"""
exp204 WARP-相当ニューラルアーム: dTVT 増分予測 → cumsum 積分(アンカー) + typewell cross-attn。

writeup の WARP の核心を忠実に:
  - per-step 微分 dTVT を予測し、last known TVT を起点に cumsum で積分(well-conditioned:
    アンカーから大きく離れられない)。
  - GR は typewell への cross-attention で「弱い補正/ruler」としてのみ使う。
バックボーンは exp140 の SCA_U2Net_Reg(洗練済み)を流用(ユーザ了承)。強い特徴(PF/地層)は
一切与えず、入力は生 GR + 幾何(15ch) + typewell ruler のみ。

初期化: 最終 outconv を 0 付近初期化 → 初期増分≈0 → 初期予測≈persistence(良い事前)。
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp140" / "src"))
from model import SCA_U2Net_Reg, UNetRegConfig  # noqa: E402


class TypewellCrossAttn(nn.Module):
    """各水平位置が typewell トークン(GR,TVT_rel)へ cross-attend し ruler 文脈を得る。"""
    def __init__(self, in_ch: int, tw_in: int = 2, d: int = 64, ctx_ch: int = 8, heads: int = 4):
        super().__init__()
        self.q = nn.Linear(in_ch, d)
        self.kv = nn.Linear(tw_in, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, ctx_ch)

    def forward(self, feat_bcl: torch.Tensor, tw_tokens: torch.Tensor) -> torch.Tensor:
        # feat_bcl: (B, C, L) ; tw_tokens: (B, T, 2) -> ctx (B, ctx_ch, L)
        x = feat_bcl.transpose(1, 2)            # (B, L, C)
        q = self.q(x)                            # (B, L, d)
        kv = self.kv(tw_tokens)                  # (B, T, d)
        ctx, _ = self.attn(q, kv, kv, need_weights=False)  # (B, L, d)
        return self.out(ctx).transpose(1, 2)     # (B, ctx_ch, L)


class WARP_U2Net(nn.Module):
    def __init__(self, in_ch: int = 15, tw_T: int = 192, ctx_ch: int = 8,
                 mid_ch: int = 16, out_ch: int = 64, use_typewell: bool = True):
        super().__init__()
        self.use_typewell = use_typewell
        self.xattn = TypewellCrossAttn(in_ch, tw_in=2, ctx_ch=ctx_ch) if use_typewell else None
        u_in = in_ch + (ctx_ch if use_typewell else 0)
        self.unet = SCA_U2Net_Reg(UNetRegConfig(in_ch=u_in, mid_ch=mid_ch, out_ch=out_ch))
        # persistence 事前: 最終融合 conv を微小初期化 → 初期増分≈0
        nn.init.zeros_(self.unet.outconv.bias)
        nn.init.normal_(self.unet.outconv.weight, std=1e-4)
        for s in (self.unet.side1, self.unet.side2, self.unet.side3,
                  self.unet.side4, self.unet.side5, self.unet.side6):
            nn.init.zeros_(s.bias); nn.init.normal_(s.weight, std=1e-4)

    def forward(self, feat_bcl: torch.Tensor, tw_tokens: torch.Tensor):
        """入力 (B,C,L),(B,T,2) → 積分済み delta 予測(fuse, sides) を返す(cumsum 適用済み)。"""
        if self.use_typewell:
            ctx = self.xattn(feat_bcl, tw_tokens)
            x = torch.cat([feat_bcl, ctx], dim=1)
        else:
            x = feat_bcl
        inc_fuse, inc_sides = self.unet(x)       # 増分 dTVT (B,L)
        d_fuse = torch.cumsum(inc_fuse, dim=1)   # アンカー起点 cumsum 積分 → delta
        d_sides = [torch.cumsum(s, dim=1) for s in inc_sides]
        return d_fuse, d_sides


def masked_rmse(pred, tgt, mask, eps=1e-8):
    valid = mask.float(); n = valid.sum()
    if n < 1:
        return torch.zeros((), device=pred.device, requires_grad=True)
    return torch.sqrt((((pred - tgt) ** 2) * valid).sum() / n + eps)


def warp_ds_loss(d_fuse, d_sides, target_delta, mask):
    """積分済み delta に対する deep-supervision masked RMSE(fuse + 6 sides)。"""
    total = masked_rmse(d_fuse, target_delta, mask)
    for s in d_sides:
        total = total + masked_rmse(s, target_delta, mask)
    return total
