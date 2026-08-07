"""
exp207 提出用 自己完結デプロイモジュール(純GR WARP アーム standalone)。
モデル(SCA_U2Net_Reg + TypewellCrossAttn + cumsum積分ヘッド)+ 純GR特徴ビルダー + 5-fold
アンサンブル推論を1ファイルに集約。提出ノートは本ファイル + warp_{raw,ema}_bundle.pt を
Kaggle に添付して import するだけ。学習(exp204/warp_model.py, exp140/model.py, build_gr_features.py)
と数値一致することを EXP/exp207/verify_deploy.py で検証済。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ================= モデル(exp140 SCA_U2Net_Reg を忠実にインライン) =================
@dataclass
class UNetRegConfig:
    in_ch: int = 23
    mid_ch: int = 16
    out_ch: int = 64
    sam_kernel: int = 7
    cam_reduction: int = 16


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _upsample_like(x, target_len):
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch):
        super().__init__()
        self.height = height
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.enc = nn.ModuleList([REBNCONV(out_ch, mid_ch)])
        for _ in range(1, height - 1):
            self.enc.append(REBNCONV(mid_ch, mid_ch))
        self.bottleneck = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.dec = nn.ModuleList([REBNCONV(mid_ch * 2, mid_ch) for _ in range(height - 2)])
        self.dec_last = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        enc_feats = [self.enc[0](hxin)]
        for i in range(1, self.height - 1):
            enc_feats.append(self.enc[i](self.pool(enc_feats[-1])))
        hx = self.bottleneck(enc_feats[-1])
        hx = self.dec[0](torch.cat([hx, enc_feats[-1]], dim=1))
        for i in range(1, self.height - 1):
            skip = enc_feats[self.height - 2 - i]
            hx = _upsample_like(hx, skip.shape[-1])
            hx = self.dec[i](torch.cat([hx, skip], dim=1)) if i < self.height - 2 else self.dec_last(torch.cat([hx, skip], dim=1))
        return hx + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilation=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin); hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2); hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], dim=1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], dim=1))
        return hx1d + hxin


class SAM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y1):
        max_c = torch.max(y1, dim=1, keepdim=True)[0]
        avg_c = torch.mean(y1, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_c, avg_c], dim=1))) * y1


class CAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=False), nn.Linear(hidden, channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, y1):
        max_s = torch.max(y1, dim=2)[0]; avg_s = torch.mean(y1, dim=2)
        attn = self.mlp(max_s) + self.mlp(avg_s)
        return self.sigmoid(attn).unsqueeze(-1) * y1


class USAE(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch, sam_kernel=7):
        super().__init__()
        self.rsu = RSU(height, in_ch, mid_ch, out_ch)
        self.sam = SAM(out_ch, kernel_size=sam_kernel)

    def forward(self, x):
        y1 = self.rsu(x); return y1, self.sam(y1)


class UCAE(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, reduction=16, bottom=False):
        super().__init__()
        self.rsu = RSU4F(in_ch, mid_ch, out_ch)
        self.cam = CAM(out_ch, reduction=reduction)
        self.bottom = bottom

    def forward(self, x):
        y1 = self.rsu(x); y2 = self.cam(y1)
        return y2 if self.bottom else (y1, y2)


class SCA_U2Net_Reg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or UNetRegConfig()
        c, m, o = cfg.in_ch, cfg.mid_ch, cfg.out_ch
        sk, r = cfg.sam_kernel, cfg.cam_reduction
        self.e1 = USAE(7, c, m, o, sam_kernel=sk); self.e2 = USAE(6, o, m, o, sam_kernel=sk)
        self.e3 = USAE(5, o, m, o, sam_kernel=sk); self.e4 = USAE(4, o, m, o, sam_kernel=sk)
        self.e5 = UCAE(o, m, o, reduction=r, bottom=False); self.e6 = UCAE(o, m, o, reduction=r, bottom=True)
        self.d5 = RSU4F(o * 2, m, o); self.d4 = RSU(4, o * 2, m, o); self.d3 = RSU(5, o * 2, m, o)
        self.d2 = RSU(6, o * 2, m, o); self.d1 = RSU(7, o * 2, m, o)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)
        for i in range(1, 7):
            setattr(self, f"side{i}", nn.Conv1d(o, 1, 3, padding=1))
        self.outconv = nn.Conv1d(6, 1, 1)

    def forward(self, x):
        e1c1, e1c2 = self.e1(x); e2c1, e2c2 = self.e2(self.pool(e1c1))
        e3c1, e3c2 = self.e3(self.pool(e2c1)); e4c1, e4c2 = self.e4(self.pool(e3c1))
        e5c1, e5c2 = self.e5(self.pool(e4c1)); e6c = self.e6(self.pool(e5c1))
        hx6up = _upsample_like(e6c, e5c2.shape[-1]); hx5d = self.d5(torch.cat([hx6up, e5c2], dim=1))
        hx5dup = _upsample_like(hx5d, e4c2.shape[-1]); hx4d = self.d4(torch.cat([hx5dup, e4c2], dim=1))
        hx4dup = _upsample_like(hx4d, e3c2.shape[-1]); hx3d = self.d3(torch.cat([hx4dup, e3c2], dim=1))
        hx3dup = _upsample_like(hx3d, e2c2.shape[-1]); hx2d = self.d2(torch.cat([hx3dup, e2c2], dim=1))
        hx2dup = _upsample_like(hx2d, e1c2.shape[-1]); hx1d = self.d1(torch.cat([hx2dup, e1c2], dim=1))
        L = x.shape[-1]
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), L); d3 = _upsample_like(self.side3(hx3d), L)
        d4 = _upsample_like(self.side4(hx4d), L); d5 = _upsample_like(self.side5(hx5d), L)
        d6 = _upsample_like(self.side6(e6c), L)
        fuse_out = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1)).squeeze(1)
        return fuse_out, [d1.squeeze(1), d2.squeeze(1), d3.squeeze(1), d4.squeeze(1), d5.squeeze(1), d6.squeeze(1)]


class TypewellCrossAttn(nn.Module):
    def __init__(self, in_ch, tw_in=2, d=64, ctx_ch=8, heads=4):
        super().__init__()
        self.q = nn.Linear(in_ch, d); self.kv = nn.Linear(tw_in, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, ctx_ch)

    def forward(self, feat_bcl, tw_tokens):
        x = feat_bcl.transpose(1, 2)
        ctx, _ = self.attn(self.q(x), self.kv(tw_tokens), self.kv(tw_tokens), need_weights=False)
        return self.out(ctx).transpose(1, 2)


class WARP_U2Net(nn.Module):
    def __init__(self, in_ch=15, ctx_ch=8, mid_ch=16, out_ch=64, use_typewell=True):
        super().__init__()
        self.use_typewell = use_typewell
        self.xattn = TypewellCrossAttn(in_ch, tw_in=2, ctx_ch=ctx_ch) if use_typewell else None
        u_in = in_ch + (ctx_ch if use_typewell else 0)
        self.unet = SCA_U2Net_Reg(UNetRegConfig(in_ch=u_in, mid_ch=mid_ch, out_ch=out_ch))

    def forward(self, feat_bcl, tw_tokens):
        if self.use_typewell:
            x = torch.cat([feat_bcl, self.xattn(feat_bcl, tw_tokens)], dim=1)
        else:
            x = feat_bcl
        inc_fuse, _ = self.unet(x)
        return torch.cumsum(inc_fuse, dim=1)


# ================= 純GR特徴ビルダー(build_gr_features と同一) =================
TW_T = 192


def _roll(s, w, fn):
    return getattr(s.rolling(w, center=True, min_periods=1), fn)().to_numpy(np.float32)


def build_gr_features_one(hw, tw):
    """test/train well の hw,tw から純GR 15ch + typewell tokens + prep を返す(target 無し)。"""
    tw = tw.sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"]); last_Z = float(lk["Z"]); last_MD = float(lk["MD"])
    nh = len(ev); ev_start = ev.index[0]
    tw_tvt = tw["TVT"].to_numpy(np.float64); tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(np.float64)
    if len(tw_tvt) < 3:
        return None
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    gr_s = pd.Series(gr_full.values)
    def es(a): return np.asarray(a, np.float32)[ev_start:ev_start + nh]
    hgr = es(gr_full.values); gr_d1 = es(gr_s.diff().fillna(0.).values)
    gr_d2 = es(gr_s.diff().diff().fillna(0.).values); gr_env = es(_roll(gr_s, 21, "max"))
    gr_sm5 = es(_roll(gr_s, 5, "mean")); gr_sm15 = es(_roll(gr_s, 15, "mean")); gr_lstd = es(_roll(gr_s, 15, "std"))
    z_ev = ev["Z"].to_numpy(np.float32); z_rel = z_ev - np.float32(last_Z)
    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = es((hw["Z"].diff() / mdd).values); dxdmd = es((hw["X"].diff() / mdd).values); dydmd = es((hw["Y"].diff() / mdd).values)
    md_since = ev["MD"].to_numpy(np.float32) - np.float32(last_MD)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    dxy = np.sqrt((ev["X"].values - float(lk["X"])) ** 2 + (ev["Y"].values - float(lk["Y"])) ** 2).astype(np.float32)
    azimuth = np.arctan2(es(hw["Y"].diff().values), es(hw["X"].diff().values)).astype(np.float32)
    feat = np.column_stack([hgr, gr_d1, gr_d2, gr_env, gr_sm5, gr_sm15, gr_lstd,
                            z_rel, dzdmd, dxdmd, dydmd, md_since, frac, dxy, azimuth]).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    tgrid = np.linspace(tw_tvt[0], tw_tvt[-1], TW_T)
    tw_gr_rs = np.interp(tgrid, tw_tvt, tw_gr).astype(np.float32)
    tw_tokens = np.column_stack([tw_gr_rs, (tgrid - last_tvt).astype(np.float32)]).astype(np.float32)
    return dict(features=feat, tw_tokens=tw_tokens, last_tvt=last_tvt,
                ev_index=ev.index.values, n_eval=nh)


def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]; ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# ================= 推論(5-fold アンサンブル, 各 fold は自分の norm) =================
def load_bundle(path, device):
    bundle = torch.load(path, map_location=device, weights_only=False)
    models = []
    for k in sorted(bundle.keys()):
        b = bundle[k]
        m = WARP_U2Net(in_ch=15, use_typewell=True).to(device)
        m.load_state_dict(b["state"]); m.eval()
        models.append((m, b["fmean"], b["fstd"], b["tmean"], b["tstd"]))
    return models


@torch.no_grad()
def predict_ensemble(models, S, device):
    """5-fold 平均の予測 TVT(eval ゾーン, len=n_eval)。各 fold は自分の norm。"""
    preds = []
    for m, fm, fs, tm, ts in models:
        feat = ((S["features"] - fm) / fs).astype(np.float32)
        tw = ((S["tw_tokens"] - tm) / ts).astype(np.float32)
        x = torch.from_numpy(feat.T).unsqueeze(0).to(device)
        twt = torch.from_numpy(tw).unsqueeze(0).to(device)
        preds.append(m(x, twt)[0].cpu().numpy())
    return np.mean(preds, axis=0) + S["last_tvt"]
