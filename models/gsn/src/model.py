import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.cnn_sdf_config import Config
from .dataset import build_input_image, input_channel_counts


def spatial_dropout(p: float) -> nn.Module:
    """Channel-wise dropout for conv features; no-op when disabled or p <= 0."""
    if not Config.USE_DROPOUT2D or p <= 0:
        return nn.Identity()
    return nn.Dropout2d(p)


class GRDescriptorEncoder(nn.Module):
    """Encode a 1D GR sequence into local multi-scale descriptors."""

    def __init__(self, embed_dim: int):
        super().__init__()
        groups = max(g for g in range(min(8, embed_dim), 0, -1) if embed_dim % g == 0)
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(
                32, embed_dim, kernel_size=5, padding=4, dilation=2, bias=False
            ),
            nn.GroupNorm(groups, embed_dim),
            nn.GELU(),
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=4,
                dilation=4,
                bias=False,
            ),
            nn.GroupNorm(groups, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedGRCorrelation(nn.Module):
    """Build a cosine-similarity volume from typewell and horizontal GR.

    The two encoders have the same architecture but independent weights because
    T is sampled in TVT while H is sampled along MD; sharing convolution weights
    would incorrectly assume equal physical scale on both axes.
    """

    def __init__(
        self,
        embed_dim: int,
        scale: float = 1.0,
        h_step: int = 24,
        h_history: int = 93,
        h_future: int = 427,
    ):
        super().__init__()
        self.t_encoder = GRDescriptorEncoder(embed_dim)
        self.h_encoder = GRDescriptorEncoder(embed_dim)
        self.scale = float(scale)
        self.h_step = int(h_step)
        self.h_history = int(h_history)
        self.h_future = int(h_future)

    @staticmethod
    def _masked_standardize(
        x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask[:, None, :].to(dtype=x.dtype)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=-1, keepdim=True) / denom
        var = ((x - mean).square() * mask).sum(dim=-1, keepdim=True) / denom
        return ((x - mean) / torch.sqrt(var + 1e-6)) * mask

    def _reduce_prebin_segment(
        self,
        desc: torch.Tensor,
        *,
        history: bool,
        use_mean: bool,
        offset: int,
    ) -> torch.Tensor:
        """Apply the dataset's exact pad/truncate + H_S reduction to descriptors."""
        length = desc.shape[-1]
        if length == 0:
            return desc.new_zeros((desc.shape[0], 0))

        pad_n = (-length) % self.h_step
        if pad_n < self.h_step // 2:
            pad = (pad_n, 0) if history else (0, pad_n)
            desc = F.pad(desc[None], pad, mode="replicate")[0]
        elif pad_n > 0:
            trim = self.h_step - pad_n
            desc = desc[:, trim:] if history else desc[:, :-trim]

        grouped = desc.reshape(desc.shape[0], -1, self.h_step)
        if use_mean:
            return grouped.mean(dim=-1)
        return grouped[:, :, min(max(offset, 0), self.h_step - 1)]

    @staticmethod
    def _fit_segment(
        desc: torch.Tensor, target: int, *, history: bool
    ) -> torch.Tensor:
        """Keep last history / first future bins and edge-pad to target length."""
        length = desc.shape[-1]
        if length >= target:
            return desc[:, -target:] if history else desc[:, :target]
        if length == 0:
            return desc.new_zeros((desc.shape[0], target))
        pad_n = target - length
        pad = (pad_n, 0) if history else (0, pad_n)
        return F.pad(desc[None], pad, mode="replicate")[0]

    def _pool_prebin_descriptors(
        self,
        desc: torch.Tensor,
        lengths: torch.Tensor,
        split_ps: torch.Tensor,
        offsets: torch.Tensor,
        use_mean: torch.Tensor,
        flipped: torch.Tensor,
    ) -> torch.Tensor:
        """Pool full-resolution H descriptors onto the model's H grid."""
        pooled = []
        for b in range(desc.shape[0]):
            length = max(0, min(int(lengths[b]), desc.shape[-1]))
            ps = max(0, min(int(split_ps[b]), max(length - 1, 0)))
            hist = desc[b, :, : ps + 1] if length > 0 else desc[b, :, :0]
            fut = desc[b, :, ps + 1 : length] if length > 0 else desc[b, :, :0]

            hist = self._reduce_prebin_segment(
                hist,
                history=True,
                use_mean=bool(use_mean[b]),
                offset=int(offsets[b]),
            )
            fut = self._reduce_prebin_segment(
                fut,
                history=False,
                use_mean=bool(use_mean[b]),
                offset=int(offsets[b]),
            )
            hist = self._fit_segment(hist, self.h_history, history=True)
            fut = self._fit_segment(fut, self.h_future, history=False)
            item = torch.cat([hist, fut], dim=-1)
            if bool(flipped[b]):
                item = item.flip(-1)
            pooled.append(item)
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        t_gr: torch.Tensor,
        h_gr: torch.Tensor,
        t_mask: torch.Tensor,
        h_mask: torch.Tensor,
        *,
        h_lengths: torch.Tensor | None = None,
        h_split_ps: torch.Tensor | None = None,
        h_offsets: torch.Tensor | None = None,
        h_use_mean: torch.Tensor | None = None,
        h_flipped: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_gr = self._masked_standardize(t_gr, t_mask)
        t_desc = F.normalize(self.t_encoder(t_gr), dim=1, eps=1e-6)

        if h_lengths is not None:
            max_length = max(1, min(int(h_lengths.max()), h_gr.shape[-1]))
            h_gr = h_gr[..., :max_length]
            prebin_mask = (
                torch.arange(h_gr.shape[-1], device=h_gr.device)[None, :]
                < h_lengths.to(h_gr.device)[:, None]
            ).to(h_gr.dtype)
            h_gr = self._masked_standardize(h_gr, prebin_mask)
            h_desc = self.h_encoder(h_gr)
            h_desc = self._pool_prebin_descriptors(
                h_desc,
                h_lengths,
                h_split_ps,
                h_offsets,
                h_use_mean,
                h_flipped,
            )
        else:
            h_gr = self._masked_standardize(h_gr, h_mask)
            h_desc = self.h_encoder(h_gr)
        h_desc = F.normalize(h_desc, dim=1, eps=1e-6)

        correlation = torch.einsum("bct,bch->bth", t_desc, h_desc)
        valid = t_mask[:, :, None] * h_mask[:, None, :]
        return correlation * valid.to(correlation.dtype) * self.scale


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlockWithSE(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.drop = spatial_dropout(dropout_p)
        self.se = SEBlock(out_c)
        
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.gelu(out)

class UNetResNet(nn.Module):
    def __init__(self, in_c: int | None = None):
        if in_c is None:
            in_c, _ = input_channel_counts()
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        enc_drop = Config.DROPOUT_ENCODER_DEEP
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, dropout_p=enc_drop)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, dropout_p=enc_drop)

    def _make_layer(self, in_c, out_c, blocks, stride, dropout_p: float = 0.0):
        layers = [ResBlockWithSE(in_c, out_c, stride, dropout_p=dropout_p)]
        for _ in range(1, blocks):
            layers.append(ResBlockWithSE(out_c, out_c, 1, dropout_p=dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f0, f1, f2, f3, f4

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            spatial_dropout(Config.DROPOUT_ASPP),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(out)

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            spatial_dropout(Config.DROPOUT_DECODER),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GeoSteerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_type = ["inference", "loss"]
        self.D = nn.Parameter(torch.zeros(1))
        self.current_epoch = 1

        in_c, head_extra_c = input_channel_counts()
        if getattr(Config, "USE_LEARNED_CORRELATION", False):
            self.correlation = LearnedGRCorrelation(
                embed_dim=int(Config.CORRELATION_EMBED_DIM),
                scale=float(Config.CORRELATION_SCALE),
                h_step=int(Config.H_S),
                h_history=int(Config.H_H),
                h_future=int(Config.H_F),
            )
            in_c += 1
        else:
            self.correlation = None
        self.backbone = UNetResNet(in_c=in_c)

        self.aspp = ASPP(512, 256)
        
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.up0 = UpBlock(32, 32, 32)

        # Multi-task head: channel 0 -> SDF regression, channel 1 -> line seg logits.
        self.head = nn.Sequential(
            nn.Conv2d(32 + head_extra_c, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            spatial_dropout(Config.DROPOUT_HEAD),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, batch):
        device = self.D.device

        # All feature engineering / 2D image assembly lives in dataset.build_input_image.
        image, head_extra = build_input_image(batch, device)

        t_mask = batch["t_mask"].to(device)
        h_mask = batch["h_mask"].to(device)
        if self.correlation is not None:
            h_prebin = batch.get("h_gr_prebin")
            if h_prebin is None:
                h_corr_gr = batch["h_feat"][:, 0]
                prebin_kwargs = {}
            else:
                h_corr_gr = h_prebin
                prebin_kwargs = {
                    "h_lengths": batch["h_prebin_len"],
                    "h_split_ps": batch["h_prebin_ps"],
                    "h_offsets": batch["h_prebin_offset"],
                    "h_use_mean": batch["h_prebin_use_mean"],
                    "h_flipped": batch["h_prebin_flipped"],
                }
            correlation = self.correlation(
                batch["t_feat"][:, 0:1].to(device),
                h_corr_gr[:, None].to(device),
                t_mask,
                h_mask,
                **prebin_kwargs,
            )
            image = torch.cat([image, correlation[:, None]], dim=1)

        f0, f1, f2, f3, f4 = self.backbone(image)
        
        x = self.aspp(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.up0(x, f0)

        if head_extra.shape[1] > 0:
            x = torch.cat([x, head_extra], dim=1)

        raw = self.head(x)                    # (B, 2, T, H)
        sdf = (
            raw[:, 0:1] * Config.SDF_OUTPUT_SCALE
        ).clamp(-Config.SDF_CLIP, Config.SDF_CLIP)
        seg_logits = raw[:, 1:2]              # Channel 1: line seg logits

        mask = t_mask[:, None, :, None] * h_mask[:, None, None, :]
        eval_mask = batch["eval_mask"].to(device)
        future_mask = mask * eval_mask[:, None, None, :]
        sdf_mask = future_mask if Config.SDF_LOSS_FUTURE_ONLY else mask

        output = {}
        if "loss" in self.output_type and "sdf" in batch:
            sdf_loss = do_masked_hybrid_loss(sdf, batch["sdf"].to(device), sdf_mask)
            seg_loss = do_masked_seg_loss(
                seg_logits, batch["label"].to(device),
                future_mask if Config.SEG_LOSS_FUTURE_ONLY else mask
            )
            output["sdf_loss"] = sdf_loss
            output["seg_loss"] = seg_loss
            output["loss"] = sdf_loss + Config.SEG_LOSS_WEIGHT * seg_loss

        if "inference" in self.output_type:
            output["sdf"] = sdf
            output["seg"] = torch.sigmoid(seg_logits)

        return output

def build_viterbi_cost_matrix(
    sdf: np.ndarray,
    seg_prob: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse path scores into a Viterbi cost map (lower = more likely on path)."""
    cost = np.abs(sdf).astype(np.float32)
    w = Config.SEG_VITERBI_WEIGHT
    if seg_prob is not None and w > 0:
        cost = cost - w * seg_prob.astype(np.float32)
    return cost


def do_masked_hybrid_loss(predict, target, mask):
    mse = F.mse_loss(predict, target, reduction="none")
    l1 = F.l1_loss(predict, target, reduction="none")
    base_loss = mse + Config.SDF_LOSS_L1_WEIGHT * l1

    # W(y, x) = exp(-gamma * |s_true|) + eps
    target_weight = torch.exp(-Config.SDF_LOSS_GAMMA * torch.abs(target)) + Config.SDF_LOSS_EPS

    loss = ((base_loss * target_weight) * mask).sum() / ((mask * target_weight).sum() + 1e-8)
    return loss


def _bce_fp32(pred, target, *, reduction="none", pos_weight=None, from_logits=False):
    """BCE is unsafe under autocast; always compute in full precision."""
    device_type = "cuda" if pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()
        if from_logits:
            pw = pos_weight.float() if pos_weight is not None else None
            return F.binary_cross_entropy_with_logits(
                pred, target, reduction=reduction, pos_weight=pw
            )
        return F.binary_cross_entropy(pred, target, reduction=reduction)


def do_masked_seg_loss(logits, target, mask, eps=1e-6):
    """Joint BCE + soft-Dice loss for the sparse 2D line ('pipe') channel.

    Both terms are restricted to the valid (non-padding) region via `mask`.
    Dice is computed per-sample over the spatial dims so it stays scale-free
    w.r.t. background area, forcing the network to lock onto the thin ridge.

    Args:
        logits: (B, 1, T, H) raw segmentation logits.
        target: (B, 1, T, H) soft line label in [0, 1] (anti-aliased cv2 line).
        mask:   (B, 1, T, H) validity mask in {0, 1}.
    """
    prob = torch.sigmoid(logits) * mask
    tgt = target * mask
    dims = (1, 2, 3)
    inter = (prob * tgt).sum(dims)
    denom = prob.sum(dims) + tgt.sum(dims)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    dice = dice.mean()

    pos_weight = torch.tensor(Config.SEG_BCE_POS_WEIGHT, device=logits.device, dtype=logits.dtype)
    bce = _bce_fp32(logits, target, reduction="none", pos_weight=pos_weight, from_logits=True)
    bce = (bce * mask).sum() / (mask.sum() + eps)
    return (1 - Config.SEG_DICE_WEIGHT) * bce + Config.SEG_DICE_WEIGHT * dice