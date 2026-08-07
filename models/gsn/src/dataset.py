import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
import cv2
import hashlib
from pathlib import Path
from config.cnn_sdf_config import Config


# ── History 2D input ablation (改这里后 reload 模块；不动 config) ─────────────
# "legacy_cross" : (h_tvt - t_tvt[row]) / scale，仅 history 列非零
# "none"         : 两通道全 0（占位，便于关特征做消融）
HISTORY_TVT_INPUT_MODE = "legacy_cross"

# ============================================================================
# Feature registry
# ----------------------------------------------------------------------------
# Each feature is a (name, fn) pair. `fn` receives a context dict of base
# arrays and returns a 1D np.ndarray aligned to the well samples.
#
#   * Channel 0 of each list is the primary "matchable" signal (used to build
#     the cross-feature gr_diff in build_input_image). KEEP GR FIRST.
#   * To ABLATE: swap fn in HORIZONTAL_FEATURES (见下方 _h_gr_* / _h_*_ps|grad 成对函数)
# ============================================================================

def _roll_mean(arr, window):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().values


def _affine_calib(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit y_true ≈ a * y_pred + b on valid pairs (good_lunck parity)."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 20:
        return 1.0, 0.0
    y = y_true[valid]
    x = y_pred[valid]
    if x.std() < 1e-6:
        return 1.0, float(y.mean() - x.mean())
    a = np.cov(x, y)[0, 1] / np.var(x)
    b = float(y.mean() - a * x.mean())
    return float(a), b


def _gr_calib_params(
    h_gr: np.ndarray,
    tvt_input: np.ndarray,
    ps_idx: int,
    tw_tvt: np.ndarray | None,
    tw_gr: np.ndarray | None,
) -> tuple[float, float]:
    """Affine calib from known zone: h_gr ≈ a * interp(tvt, tw) + b."""
    if tw_tvt is None or tw_gr is None or len(tw_tvt) == 0:
        return 1.0, 0.0
    known = np.isfinite(tvt_input[: ps_idx + 1])
    if known.sum() < 20:
        return 1.0, 0.0
    known_tvt = tvt_input[: ps_idx + 1][known]
    known_gr = h_gr[: ps_idx + 1][known]
    tw_gr_fill = pd.Series(tw_gr).interpolate().bfill().ffill().fillna(85.0).values.astype(float)
    line_interp = np.interp(known_tvt, tw_tvt.astype(float), tw_gr_fill)
    return _affine_calib(known_gr, line_interp)


def _gr_to_typewell_scale(gr: np.ndarray, calib_a: float, calib_b: float) -> np.ndarray:
    a = calib_a if abs(calib_a) > 1e-6 else 1.0
    return (gr - calib_b) / a


# --- Typewell features. ctx keys: "gr" (raw GR), "tvt" (TVT axis) -----------
def _tw_gr(ctx):        return ctx["gr"]
def _tw_gr_roll10(ctx): return _roll_mean(ctx["gr"], 10)
def _tw_gr_roll50(ctx): return _roll_mean(ctx["gr"], 50)
def _tw_dgr(ctx):       return np.nan_to_num(np.gradient(ctx["gr"], ctx["tvt"], edge_order=1), 0.0)

TYPEWELL_FEATURES = [
    ("gr",        _tw_gr),
    # ("gr_roll10", _tw_gr_roll10),
    # ("gr_roll50", _tw_gr_roll50),
    # ("dgr",       _tw_dgr),
]


# --- Horizontal features. ctx: "gr", "calib_a/b", "Z","MD","X","Y", "ps_idx" ----
# 成对变体供消融：注释/替换 HORIZONTAL_FEATURES 里的 fn 即可（改后需 reload 模块）。

def _h_gr_raw(ctx):
    return ctx["gr"]

def _h_gr_calib(ctx):
    return _gr_to_typewell_scale(ctx["gr"], ctx["calib_a"], ctx["calib_b"])

def _h_dz_ps(ctx):
    ps = int(ctx["ps_idx"])
    return ctx["Z"] - ctx["Z"][ps]

def _h_dz_grad(ctx):
    return np.nan_to_num(np.gradient(ctx["Z"]), 0.0)

def _h_dmd_ps(ctx):
    ps = int(ctx["ps_idx"])
    return ctx["MD"] - ctx["MD"][ps]

def _h_dmd_grad(ctx):
    return np.nan_to_num(np.gradient(ctx["MD"]), 1.0)

def _h_dz_dmd_ps(ctx):
    ps = int(ctx["ps_idx"])
    dz = ctx["Z"] - ctx["Z"][ps]
    dmd = ctx["MD"] - ctx["MD"][ps]
    return dz / (np.abs(dmd) + 1e-8)

def _h_dz_dmd_grad(ctx):
    dz = np.nan_to_num(np.gradient(ctx["Z"]), 0.0)
    dmd = np.nan_to_num(np.gradient(ctx["MD"]), 1.0)
    return dz / (np.abs(dmd) + 1e-8)

def _h_dx_ps(ctx):
    ps = int(ctx["ps_idx"])
    return ctx["X"] - ctx["X"][ps]

def _h_dx_grad(ctx):
    return np.nan_to_num(np.gradient(ctx["X"]), 0.0)

def _h_dy_ps(ctx):
    ps = int(ctx["ps_idx"])
    return ctx["Y"] - ctx["Y"][ps]

def _h_dy_grad(ctx):
    return np.nan_to_num(np.gradient(ctx["Y"]), 0.0)

def _h_azimuth_ps(ctx):
    ps = int(ctx["ps_idx"])
    return np.arctan2(ctx["Y"] - ctx["Y"][ps], ctx["X"] - ctx["X"][ps] + 1e-8)

def _h_azimuth_grad(ctx):
    dx = np.nan_to_num(np.gradient(ctx["X"]), 0.0)
    dy = np.nan_to_num(np.gradient(ctx["Y"]), 0.0)
    return np.arctan2(dy, dx + 1e-8)


def _h_dmd_dz_unit(ctx):
    """Unit tangent in the MD–Z plane: (dMD/ds, dZ/ds) with s along the well index."""
    dmd = np.nan_to_num(np.gradient(ctx["MD"]), 1.0)
    dz = np.nan_to_num(np.gradient(ctx["Z"]), 0.0)
    norm = np.sqrt(dmd ** 2 + dz ** 2) + 1e-8
    return dmd / norm, dz / norm


def _h_sin_dip(ctx):
    """sin component of well dip tangent (MD vs Z)."""
    sin_dip, _ = _h_dmd_dz_unit(ctx)
    return sin_dip


def _h_cos_dip(ctx):
    """cos component of well dip tangent (MD vs Z)."""
    _, cos_dip = _h_dmd_dz_unit(ctx)
    return cos_dip


def _h_dx_dy_unit(ctx):
    """Unit map-view direction tangent: (dX/dMD, dY/dMD) normalized."""
    dmd = np.nan_to_num(np.gradient(ctx["MD"]), 1.0)
    dmd = np.where(np.abs(dmd) < 1e-8, 1e-8, dmd)
    dx_dmd = np.nan_to_num(np.gradient(ctx["X"]), 0.0) / dmd
    dy_dmd = np.nan_to_num(np.gradient(ctx["Y"]), 0.0) / dmd
    norm = np.sqrt(dx_dmd ** 2 + dy_dmd ** 2) + 1e-8
    return dx_dmd / norm, dy_dmd / norm


def _h_sin_dir(ctx):
    """sin(atan2(dY/dMD, dX/dMD)) — geology / map direction tangent."""
    _, sin_dir = _h_dx_dy_unit(ctx)
    return sin_dir


def _h_cos_dir(ctx):
    """cos(atan2(dY/dMD, dX/dMD)) — geology / map direction tangent."""
    cos_dir, _ = _h_dx_dy_unit(ctx)
    return cos_dir


def _h_gr_roll10(ctx): return _roll_mean(ctx["gr"], 10)
def _h_gr_roll50(ctx): return _roll_mean(ctx["gr"], 50)
def _h_dgr(ctx):       return np.nan_to_num(np.gradient(ctx["gr"]), 0.0)
def _h_relative_md(ctx):
    r = ctx["MD"] - ctx["MD"].min()
    return r / (r.max() + 1e-8)
def _h_c_est(ctx):
    c = ctx["tvt_input"] + ctx["Z"]
    c = pd.Series(c).interpolate(limit_direction="both").ffill().bfill().fillna(0.0).values
    return (c - c.mean()) / (c.std() + 1e-8)

HORIZONTAL_FEATURES = [
    # ── ch0 GR：_h_gr_calib（仿射校准）| _h_gr_raw（平滑原始 GR）────────────
    ("gr",          _h_gr_calib),
    # ── 轨迹：_h_*_ps（相对 PS，good_lunck）| _h_*_grad（旧版 gradient）────
    ("dz",          _h_dz_ps),
    ("dx",          _h_dx_ps),
    ("dy",          _h_dy_ps),
    # ── 井斜 / 方位切向（sin/cos，沿 MD 求导；取消注释即可消融对比）────────
    ("sin_dip",     _h_sin_dip),
    ("cos_dip",     _h_cos_dip),
    ("sin_dir",     _h_sin_dir),
    ("cos_dir",     _h_cos_dir),
    ("dmd",         _h_dmd_ps),       # 或 _h_dmd_grad
    # ("dz_dmd",      _h_dz_dmd_ps),    # 或 _h_dz_dmd_grad
    # ("azimuth",     _h_azimuth_ps),   # 或 _h_azimuth_grad
    # ("gr_roll10",   _h_gr_roll10),
    # ("gr_roll50",   _h_gr_roll50),
    # ("dgr",         _h_dgr),
    # ("relative_md", _h_relative_md),
    # ("c_est",       _h_c_est),
]

K_T = len(TYPEWELL_FEATURES)
K_H = len(HORIZONTAL_FEATURES)

# Channel bookkeeping for the assembled 2D model input (see build_input_image).
# Base axis channels: t_gr + h_gr + t_extra + h_extra = K_T + K_H.
# The raw gr_diff channel is optional for learned-correlation ablation.
NUM_NORM_CHANNELS = K_T + K_H + int(getattr(Config, "USE_GR_DIFF_INPUT", True))

def input_channel_counts() -> tuple[int, int]:
    """按 Config 计算 backbone / head 输入通道数（支持 C1 history 消融）。"""
    backbone_hist = 2 if Config.USE_HISTORY_IN_BACKBONE else 0
    coord = 2 if Config.USE_COORD_ENCODING else 0
    head_extra = 2 if Config.USE_HISTORY_IN_HEAD else 0
    gr_diff = 1 if getattr(Config, "USE_GR_DIFF_INPUT", True) else 0
    in_c = K_T + K_H + gr_diff + backbone_hist + coord
    return in_c, head_extra


# 模块导入时快照（GeoSteerNet 应在改 Config 后新建实例）
NUM_INPUT_CHANNELS, NUM_HEAD_EXTRA_CHANNELS = input_channel_counts()


def build_input_image(batch, device):
    """Assemble the (B, C, T, H) model input from per-axis 1D features.

    Channel layout is controlled by USE_HISTORY_IN_BACKBONE / USE_COORD_ENCODING /
    USE_HISTORY_IN_HEAD at the top of this file.

    Returns:
        image:      (B, NUM_INPUT_CHANNELS, T, H)
        head_extra: (B, NUM_HEAD_EXTRA_CHANNELS, T, H); empty when head reinject off
    """
    t_feat = batch["t_feat"].to(device)            # (B, K_T, T)
    h_feat = batch["h_feat"].to(device)            # (B, K_H, H)
    hist_mask = batch["history_mask"].to(device)   # (B, H)

    B, _, T = t_feat.shape
    H = h_feat.shape[2]

    need_history = Config.USE_HISTORY_IN_BACKBONE or Config.USE_HISTORY_IN_HEAD
    if need_history:
        h_tvt = batch["h_seg_tvt"].to(device)
        h_tvt_img = h_tvt.view(B, 1, 1, H).expand(B, 1, T, H)
        hist_mask_img = hist_mask.view(B, 1, 1, H).expand(B, 1, T, H)

        if HISTORY_TVT_INPUT_MODE == "none":
            history_img = torch.zeros_like(hist_mask_img)
        else:
            t_tvt = batch["t_seg_tvt"].to(device)
            t_tvt_img = t_tvt.view(B, 1, T, 1).expand(B, 1, T, H)
            history_img = ((h_tvt_img - t_tvt_img) / Config.TVT_DIFF_SCALE).clamp(
                -Config.SDF_CLIP, Config.SDF_CLIP
            ) * hist_mask_img

    # Primary signal (channel 0) broadcast both ways.
    t_gr_img = t_feat[:, 0:1, :].unsqueeze(-1).expand(B, 1, T, H)
    h_gr_img = h_feat[:, 0:1, :].unsqueeze(2).expand(B, 1, T, H)
    gr_diff = (
        t_gr_img - h_gr_img
        if getattr(Config, "USE_GR_DIFF_INPUT", True)
        else None
    )

    # Remaining per-axis features (everything after channel 0).
    t_extra = t_feat[:, 1:, :].unsqueeze(-1).expand(B, t_feat.shape[1] - 1, T, H)
    h_extra = h_feat[:, 1:, :].unsqueeze(2).expand(B, h_feat.shape[1] - 1, T, H)

    normed = F.instance_norm(torch.cat([t_gr_img, h_gr_img, t_extra, h_extra], dim=1))
    n_t_extra = t_feat.shape[1] - 1
    t_gr_img = normed[:, 0:1]
    h_gr_img = normed[:, 1:2]
    t_extra = normed[:, 2 : 2 + n_t_extra]
    h_extra = normed[:, 2 + n_t_extra :]
    continuous_parts = [t_gr_img, h_gr_img]
    if gr_diff is not None:
        continuous_parts.append(gr_diff)
    continuous_parts.extend([t_extra, h_extra])
    continuous = torch.cat(continuous_parts, dim=1)

    parts = [continuous]
    if Config.USE_HISTORY_IN_BACKBONE:
        parts.extend([history_img, hist_mask_img])
    if Config.USE_COORD_ENCODING:
        t_coords = torch.linspace(-1, 1, T, device=device).view(1, 1, T, 1).expand(B, 1, T, H)
        h_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, 1, H).expand(B, 1, T, H)
        parts.extend([t_coords, h_coords])
    image = torch.cat(parts, dim=1)

    if Config.USE_HISTORY_IN_HEAD:
        head_extra = torch.cat([history_img, hist_mask_img], dim=1)
    else:
        head_extra = image.new_zeros((B, 0, T, H))

    return image, head_extra


def get_input_channel_names():
    """Channel names for `build_input_image` output, in stack order."""
    names = ["t_gr", "h_gr"]
    if getattr(Config, "USE_GR_DIFF_INPUT", True):
        names.append("gr_diff")
    names += [f"t_{n}" for n, _ in TYPEWELL_FEATURES[1:]]
    names += [f"h_{n}" for n, _ in HORIZONTAL_FEATURES[1:]]
    if Config.USE_HISTORY_IN_BACKBONE:
        names.extend(["history_tvt_diff", "history_mask"])
    if Config.USE_COORD_ENCODING:
        names.extend(["t_coord", "h_coord"])
    return names


def get_head_extra_channel_names():
    if Config.USE_HISTORY_IN_HEAD:
        return ["history_tvt_diff", "history_mask"]
    return []


def resample_typewell_by_step(t, target_step=0.5):
    t_tvt = t["TVT"].values.astype(np.float64)
    t_gr  = t["GR"].values.astype(np.float64)

    # Build the feature stack from the TYPEWELL_FEATURES registry (top of file).
    t_ctx = {"gr": t_gr, "tvt": t_tvt}
    t_feat = np.stack([fn(t_ctx) for _, fn in TYPEWELL_FEATURES], axis=1)

    diffs = np.abs(np.diff(t_tvt))
    diffs = diffs[diffs > 0]
    step = np.median(diffs) if len(diffs) > 0 else 0.5
    ratio = step / target_step

    if np.isclose(ratio, 1.0):
        pass
    elif ratio < 1.0:
        group_size = int(round(1 / ratio))
        n = len(t_tvt)
        pad_len = (-n) % group_size
        if pad_len > 0:
            t_tvt = np.pad(t_tvt, (0, pad_len), mode="edge")
            t_feat = np.pad(t_feat, ((0, pad_len), (0, 0)), mode="edge")
        t_tvt = t_tvt.reshape(-1, group_size).mean(axis=1)
        t_feat = t_feat.reshape(-1, group_size, t_feat.shape[1]).mean(axis=1)
    else:
        up_factor = int(round(ratio))
        old_idx = np.arange(len(t_tvt))
        new_idx = np.linspace(0, len(t_tvt) - 1, (len(t_tvt) - 1) * up_factor + 1)
        new_tvt = np.interp(new_idx, old_idx, t_tvt)
        new_feat = np.stack([np.interp(new_idx, old_idx, t_feat[:, k]) for k in range(t_feat.shape[1])], axis=1)
        t_tvt = new_tvt
        t_feat = new_feat

    return t_tvt, t_feat


def _odd_savgol_window(window: int) -> int:
    w = int(window)
    if w % 2 == 0:
        w += 1
    return max(5, w)


def _pad_bin_history_1d(arr: np.ndarray, step: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    pad_n = (-len(arr)) % step
    if pad_n < step // 2:
        return np.pad(arr, (pad_n, 0), mode="edge")
    if pad_n > 0:
        return arr[(step - pad_n):]
    return arr


def _pad_bin_after_1d(arr: np.ndarray, step: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    pad_n = (-len(arr)) % step
    if pad_n < step // 2:
        return np.pad(arr, (0, pad_n), mode="edge")
    if pad_n > 0:
        return arr[:-(step - pad_n)]
    return arr


def _pad_bin_history_nd(arr: np.ndarray, step: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    pad_n = (-len(arr)) % step
    if pad_n < step // 2:
        return np.pad(arr, ((pad_n, 0),) + ((0, 0),) * (arr.ndim - 1), mode="edge")
    if pad_n > 0:
        return arr[(step - pad_n):]
    return arr


def _pad_bin_after_nd(arr: np.ndarray, step: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    pad_n = (-len(arr)) % step
    if pad_n < step // 2:
        return np.pad(arr, ((0, pad_n),) + ((0, 0),) * (arr.ndim - 1), mode="edge")
    if pad_n > 0:
        return arr[:-(step - pad_n)]
    return arr


def _reduce_bins(
    grouped: np.ndarray,
    step: int,
    bin_mode: str,
    bin_offset: int,
) -> np.ndarray:
    """Reduce (n_bins, step, ...) -> (n_bins, ...); bin_mode: mean | offset."""
    if grouped.size == 0:
        return np.array([])
    if bin_mode == "mean":
        return grouped.mean(axis=1)
    idx = np.arange(grouped.shape[0])
    return grouped[idx, bin_offset]


def _bin_sample_segment(
    tvt_1d: np.ndarray,
    feat_nd: np.ndarray,
    step: int,
    bin_mode: str,
    *,
    pad_side: str,
    bin_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample one PS segment; TVT and features share the same within-bin offset."""
    pad_1d = _pad_bin_history_1d if pad_side == "history" else _pad_bin_after_1d
    pad_nd = _pad_bin_history_nd if pad_side == "history" else _pad_bin_after_nd

    tvt_p = pad_1d(tvt_1d, step)
    feat_p = pad_nd(feat_nd, step)
    if len(tvt_p) == 0:
        n_feat = feat_nd.shape[1] if feat_nd.ndim == 2 and feat_nd.size > 0 else 0
        return np.array([]), np.zeros((0, n_feat))

    g_tvt = tvt_p.reshape(-1, step)
    g_feat = feat_p.reshape(-1, step, feat_p.shape[1]) if feat_p.ndim == 2 else feat_p.reshape(-1, step)

    h_tvt = _reduce_bins(g_tvt, step, bin_mode, bin_offset)
    h_feat = _reduce_bins(g_feat, step, bin_mode, bin_offset)
    if h_feat.ndim != 2:
        h_feat = np.zeros((0, feat_nd.shape[1] if feat_nd.ndim == 2 else 0))
    return h_tvt, h_feat


def resample_horizontal_by_step(
    h,
    target_step=32,
    offset=0,
    tw_tvt=None,
    tw_gr=None,
    gr_filter_window: int | None = None,
    apply_savgol: bool = True,
    bin_mode: str = "mean",
    ps_idx: int | None = None,
):
    """bin_mode: mean (val/infer) | offset (train: one random in-bin index per well).

    ps_idx: optional split index (pseudo PS for PFE). When None, use last TVT_input.
    """
    h = h.copy()
    gr_filter_window = Config.H_GR_FILTER if gr_filter_window is None else gr_filter_window

    h_gr_raw = h["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values
    h_gr_smooth = h_gr_raw.copy()
    if apply_savgol:
        w = _odd_savgol_window(gr_filter_window)
        if len(h_gr_raw) > w:
            h_gr_smooth = savgol_filter(h_gr_raw, w, 2)
    
    Z  = h["Z"].values.astype(float)  if "Z"  in h.columns else np.zeros(len(h))
    MD = h["MD"].values.astype(float) if "MD" in h.columns else np.arange(len(h), dtype=float)
    X  = h["X"].values.astype(float)  if "X"  in h.columns else np.zeros(len(h))
    Y  = h["Y"].values.astype(float)  if "Y"  in h.columns else np.zeros(len(h))
    tvt_input = h["TVT_input"].values.astype(float) if "TVT_input" in h.columns else np.full(len(h), np.nan)

    if ps_idx is not None:
        h_ps = int(ps_idx)
    elif "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        h_ps = int(np.flatnonzero(h["TVT_input"].notna().values)[-1]) + offset
    else:
        h_ps = len(h) // 2

    calib_a, calib_b = _gr_calib_params(h_gr_smooth, tvt_input, h_ps, tw_tvt, tw_gr)

    # Build the feature stack from the HORIZONTAL_FEATURES registry (top of file).
    h_ctx = {
        "gr": h_gr_smooth,
        "calib_a": calib_a,
        "calib_b": calib_b,
        "Z": Z,
        "MD": MD,
        "X": X,
        "Y": Y,
        "tvt_input": tvt_input,
        "ps_idx": h_ps,
    }
    data = np.stack([fn(h_ctx) for _, fn in HORIZONTAL_FEATURES], axis=1)

    # Fallback logic for test set evaluation
    if "TVT" in h.columns and h["TVT"].notna().sum() > 0:
        tvt_vals = h["TVT"].values.astype(float)
    elif "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        tvt_vals = h["TVT_input"].astype(float).ffill().bfill().fillna(0.0).values
    else:
        tvt_vals = h["Z"].values.astype(float)

    tvt_all = tvt_vals
    before_tvt = tvt_all[:h_ps+1]
    after_tvt  = tvt_all[h_ps+1:]
    before_feat = data[:h_ps+1]
    after_feat  = data[h_ps+1:]

    bin_offset = np.random.randint(0, target_step) if bin_mode == "offset" else 0
    h_tvt0, h_feat0 = _bin_sample_segment(
        before_tvt, before_feat, target_step, bin_mode, pad_side="history",
        bin_offset=bin_offset,
    )
    h_tvt1, h_feat1 = _bin_sample_segment(
        after_tvt, after_feat, target_step, bin_mode, pad_side="after",
        bin_offset=bin_offset,
    )
    
    if h_feat0.ndim != 2:
        h_feat0 = np.zeros((0, K_H))
    if h_feat1.ndim != 2:
        h_feat1 = np.zeros((0, K_H))

    return (
        h_tvt0,
        h_tvt1,
        h_feat0,
        h_feat1,
        h_gr_smooth.astype(np.float32),
        int(bin_offset),
    )


def _first_known_ps_index(h: pd.DataFrame) -> int:
    if "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        return int(np.flatnonzero(h["TVT_input"].notna().values)[0])
    return 0


def _horizontal_tvt_values(h: pd.DataFrame) -> np.ndarray:
    if "TVT" in h.columns and h["TVT"].notna().sum() > 0:
        return h["TVT"].values.astype(np.float64)
    if "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        return h["TVT_input"].astype(float).ffill().bfill().fillna(0.0).values.astype(np.float64)
    return h["Z"].values.astype(np.float64)


def _true_ps_index(h: pd.DataFrame) -> int:
    if "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        return int(np.flatnonzero(h["TVT_input"].notna().values)[-1])
    return len(h) // 2


def sample_split_ps(
    h: pd.DataFrame,
    true_ps_base: int,
    *,
    config=Config,
    is_train: bool,
) -> tuple[int, int]:
    """Return (split_ps_base, true_ps_base) for horizontal resampling.

    split_ps_base is the pseudo PS used to build the H tensor; true_ps_base is
    the competition PS (for RMSE / submission).  When PFE is off, both are equal.

    PFE candidates lie in history before true PS, satisfy MD length bounds, and
    have TVT strictly above ``ps_tvt - PFE_TVT_SHIFT_THRESHOLD``.
    """
    n = len(h)
    if not is_train or not config.USE_PFE_TRAIN:
        return true_ps_base, true_ps_base

    if np.random.random() >= config.PFE_PROB:
        return true_ps_base, true_ps_base

    first_ps = _first_known_ps_index(h)
    pseudo_min = max(first_ps, config.PFE_MIN_HISTORY_ORIG - 1)
    pseudo_max = min(true_ps_base - 1, n - 1 - config.PFE_MIN_FUTURE_ORIG)

    if pseudo_min > pseudo_max:
        return true_ps_base, true_ps_base

    tvt_vals = _horizontal_tvt_values(h)
    ps_tvt = float(tvt_vals[true_ps_base])
    history_tvt_min = ps_tvt - float(config.PFE_TVT_SHIFT_THRESHOLD)
    segment = tvt_vals[pseudo_min : pseudo_max + 1]
    eligible = np.flatnonzero(segment > history_tvt_min) + pseudo_min

    if eligible.size == 0:
        return true_ps_base, true_ps_base

    split_ps_base = int(np.random.choice(eligible))
    return split_ps_base, true_ps_base


def get_crop_index_and_pad_1d(n, center, history, future):
    raw_i0 = center - history
    raw_i1 = center + future
    i0 = max(raw_i0, 0)
    i1 = min(raw_i1, n)
    pad_left = max(0, -int(raw_i0))
    pad_right = max(0, int(raw_i1 - n))
    return i0, i1, pad_left, pad_right


# ── CV split strategies (typewell group / geographic k-means) ─────────────────

CV_SPLIT_TYPEWELL_GROUP = "typewell_group"
CV_SPLIT_GEO_KMEANS = "geo_kmeans"


def get_typewell_hash(well_file: Path, train_dir: Path) -> str:
    """Group id from typewell GR curve (same typewell → same group)."""
    well_name = well_file.name.split("__")[0]
    typewell_file = train_dir / f"{well_name}{Config.TYPEWELL_SUFFIX}"
    if typewell_file.exists():
        try:
            df = pd.read_csv(typewell_file)
            rounded_gr = np.round(df["GR"].values, 2).tobytes()
            return hashlib.md5(rounded_gr).hexdigest()
        except Exception:
            return well_name
    return well_name


def well_xy_centroid(well_file: Path) -> tuple[float, float]:
    """Mean (X, Y) of a horizontal well trajectory for geographic clustering."""
    df = pd.read_csv(well_file)
    if "X" in df.columns and "Y" in df.columns:
        x = pd.to_numeric(df["X"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            return float(x[valid].mean()), float(y[valid].mean())
    return 0.0, 0.0


def _typewell_group_labels(well_files: list[Path], train_dir: Path) -> list[str]:
    return [get_typewell_hash(f, train_dir) for f in well_files]


def _geo_kmeans_labels(
    well_files: list[Path],
    *,
    n_clusters: int,
    seed: int = 5,
) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    coords = np.array([well_xy_centroid(f) for f in well_files], dtype=np.float64)
    coords = StandardScaler().fit_transform(coords)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def list_well_cv_splits(
    well_files: list[Path],
    *,
    train_dir: Path | None = None,
    n_splits: int | None = None,
    strategy: str | None = None,
    geo_kmeans_seed: int | None = None,
    geo_n_clusters: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation index splits over horizontal well files.

    Strategies:
        ``typewell_group`` (default): ``GroupKFold`` on typewell GR hash groups.
        ``geo_kmeans``: KMeans on well (X, Y) centroids (``random_state=5`` by
        default), then ``StratifiedGroupKFold`` with stratify = geo cluster and
        groups = typewell hash (no typewell leakage + balanced geography).

    Returns:
        List of ``(train_idx, val_idx)`` numpy int arrays, length ``n_splits``.
    """
    if not well_files:
        return []

    train_dir = Path(Config.TRAIN_DIR if train_dir is None else train_dir)
    n_splits = int(Config.N_FOLDS if n_splits is None else n_splits)
    strategy = strategy or getattr(Config, "CV_SPLIT_STRATEGY", CV_SPLIT_TYPEWELL_GROUP)
    geo_kmeans_seed = int(
        geo_kmeans_seed if geo_kmeans_seed is not None
        else getattr(Config, "GEO_KMEANS_SEED", 5)
    )
    geo_n_clusters = geo_n_clusters or getattr(Config, "GEO_KMEANS_N_CLUSTERS", None)
    if geo_n_clusters is None:
        geo_n_clusters = n_splits

    well_files = list(well_files)
    n_samples = len(well_files)
    groups = _typewell_group_labels(well_files, train_dir)
    X_dummy = np.zeros(n_samples)

    if strategy == CV_SPLIT_TYPEWELL_GROUP:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X_dummy, groups=groups))

    if strategy == CV_SPLIT_GEO_KMEANS:
        from sklearn.model_selection import StratifiedGroupKFold

        cluster_labels = _geo_kmeans_labels(
            well_files, n_clusters=int(geo_n_clusters), seed=geo_kmeans_seed,
        )
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
        return list(splitter.split(X_dummy, cluster_labels, groups=groups))

    raise ValueError(
        f"Unknown CV split strategy: {strategy!r}. "
        f"Use {CV_SPLIT_TYPEWELL_GROUP!r} or {CV_SPLIT_GEO_KMEANS!r}."
    )


def describe_well_cv_splits(
    well_files: list[Path],
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    train_dir: Path | None = None,
    strategy: str | None = None,
    geo_kmeans_seed: int | None = None,
    geo_n_clusters: int | None = None,
) -> pd.DataFrame:
    """Per-fold summary: sizes and geographic cluster counts in train/val."""
    train_dir = Path(Config.TRAIN_DIR if train_dir is None else train_dir)
    strategy = strategy or getattr(Config, "CV_SPLIT_STRATEGY", CV_SPLIT_TYPEWELL_GROUP)
    geo_kmeans_seed = int(
        geo_kmeans_seed if geo_kmeans_seed is not None
        else getattr(Config, "GEO_KMEANS_SEED", 5)
    )
    geo_n_clusters = geo_n_clusters or getattr(Config, "GEO_KMEANS_N_CLUSTERS", None)
    if geo_n_clusters is None:
        geo_n_clusters = len(splits)

    cluster_labels = None
    if strategy == CV_SPLIT_GEO_KMEANS:
        cluster_labels = _geo_kmeans_labels(
            well_files, n_clusters=int(geo_n_clusters), seed=geo_kmeans_seed,
        )

    rows = []
    for fold, (tr, va) in enumerate(splits):
        row = {
            "fold": fold,
            "n_train": len(tr),
            "n_val": len(va),
            "n_typewell_groups_val": len({get_typewell_hash(well_files[i], train_dir) for i in va}),
        }
        if cluster_labels is not None:
            row["n_geo_clusters_val"] = len(np.unique(cluster_labels[va]))
            row["geo_clusters_val"] = ",".join(map(str, sorted(np.unique(cluster_labels[va]))))
        rows.append(row)
    return pd.DataFrame(rows)


class WellboreSDFDataset(Dataset):
    K_T = len(TYPEWELL_FEATURES)
    K_H = len(HORIZONTAL_FEATURES)

    def __init__(self, well_files, config=Config, is_train=True, offset=0):
        self.well_files = well_files
        self.config = config
        self.is_train = is_train
        self.offset = offset

    def __len__(self):
        return len(self.well_files)

    def _flip_h_axis(self, x: torch.Tensor, *, h_dim: int) -> torch.Tensor:
        """Flip the complete H axis so the MD/time sequence stays continuous."""
        return x.flip(h_dim)

    def _apply_h_flip(self, sample):
        sample["h_mask"] = self._flip_h_axis(sample["h_mask"], h_dim=0)
        sample["h_seg_tvt"] = self._flip_h_axis(sample["h_seg_tvt"], h_dim=0)
        sample["h_gr"] = self._flip_h_axis(sample["h_gr"], h_dim=0)
        sample["h_prebin_flipped"] = ~sample["h_prebin_flipped"]
        sample["history_mask"] = self._flip_h_axis(sample["history_mask"], h_dim=0)
        sample["eval_mask"] = self._flip_h_axis(sample["eval_mask"], h_dim=0)
        sample["target"] = self._flip_h_axis(sample["target"], h_dim=0)
        sample["matched_idx"] = self._flip_h_axis(sample["matched_idx"], h_dim=0)
        sample["h_feat"] = self._flip_h_axis(sample["h_feat"], h_dim=-1)
        sample["sdf"] = self._flip_h_axis(sample["sdf"], h_dim=-1)
        sample["label"] = self._flip_h_axis(sample["label"], h_dim=-1)
        sample["history"] = self._flip_h_axis(sample["history"], h_dim=-1)
        return sample

    def _apply_h_gr_bad_segments(self, sample):
        """仅水平井 GR 局部坏段：随机区间用两端线性插值顶替（模拟坏道/缺测）。"""
        h_gr = sample["h_gr"].clone()
        H = h_gr.numel()
        if H < 4:
            return sample

        n_segs = int(np.random.randint(1, 4))
        for _ in range(n_segs):
            seg_len = int(np.random.randint(3, min(21, max(4, H // 5))))
            if seg_len >= H:
                continue
            start = int(np.random.randint(0, H - seg_len))
            end = start + seg_len
            left = h_gr[start - 1] if start > 0 else h_gr[end]
            right = h_gr[end] if end < H else h_gr[start - 1]
            t = torch.linspace(0.0, 1.0, seg_len, dtype=h_gr.dtype, device=h_gr.device)
            h_gr[start:end] = left + (right - left) * t

        sample["h_gr"] = h_gr
        sample["h_feat"][0, :] = h_gr
        return sample

    def __getitem__(self, idx):
        if self.is_train:
            gr_filter = int(np.random.randint(self.config.H_GR_FILTER[0], self.config.H_GR_FILTER[1]))
            if gr_filter % 2 == 0:
                gr_filter += 1
            apply_savgol = bool(np.random.rand() < 0.5)
            sample = self._build_sample(
                idx, gr_filter=gr_filter, apply_savgol=apply_savgol,
            )
        else:
            sample = self._build_sample(idx)

        if self.is_train:
            if np.random.rand() < self.config.H_FLIP_PROB:
                sample = self._apply_h_flip(sample)
            # if np.random.rand() < 0.5:
            #     sample = self._apply_h_gr_bad_segments(sample)

        return sample

    def _build_sample(
        self,
        idx,
        gr_filter: int | None = None,
        apply_savgol: bool | None = None,
    ):
        horiz_path = self.well_files[idx]
        sample_id = horiz_path.name.split("__")[0]

        h = pd.read_csv(horiz_path)
        typewell_path = horiz_path.parent / f"{sample_id}{self.config.TYPEWELL_SUFFIX}"
        t = pd.read_csv(typewell_path)

        t_tvt, t_feat = resample_typewell_by_step(t, target_step=0.5)
        tw_tvt = t["TVT"].values.astype(np.float64)
        tw_gr = t["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values.astype(np.float64)
        true_ps_base = _true_ps_index(h)
        split_ps_base, _ = sample_split_ps(
            h, true_ps_base, config=self.config, is_train=self.is_train,
        )
        split_ps = split_ps_base + self.offset
        true_ps = true_ps_base + self.offset
        h_step = self.config.H_S
        bin_mode = "offset" if self.is_train else "mean"
        (
            h_tvt0,
            h_tvt1,
            h_feat0,
            h_feat1,
            h_gr_prebin,
            h_bin_offset,
        ) = resample_horizontal_by_step(
            h,
            target_step=h_step,
            offset=self.offset,
            tw_tvt=tw_tvt,
            tw_gr=tw_gr,
            gr_filter_window=gr_filter,
            apply_savgol=False if apply_savgol is None else apply_savgol,
            bin_mode=bin_mode,
            ps_idx=split_ps_base,
        )

        if len(h_tvt0) > 0:
            last_tvt = h_tvt0[-1]
            last_idx = np.abs(t_tvt - last_tvt).argmin()
        else:
            last_idx = len(t_tvt) // 2

        j0, j1, pl, pr = get_crop_index_and_pad_1d(
            len(t_tvt), last_idx + 1, history=self.config.T_H, future=self.config.T_F)

        t_seg_mask = np.pad(np.ones(j1 - j0), (pl, pr))
        t_seg_tvt  = np.pad(t_tvt[j0:j1], (pl, pr), mode="edge")
        t_seg_feat = np.pad(t_feat[j0:j1], ((pl, pr), (0, 0)), mode="edge") if len(t_feat) > 0 \
            else np.zeros((self.config.T_H + self.config.T_F, self.K_T))
        t_seg_gr = t_seg_feat[:, 0]

        j0_h0, j1_h0, pl_h0, pr_h0 = get_crop_index_and_pad_1d(
            len(h_tvt0), len(h_tvt0), history=self.config.H_H, future=0)

        h_seg_mask0 = np.pad(np.ones(j1_h0 - j0_h0), (pl_h0, pr_h0))
        h_seg_tvt0  = np.pad(h_tvt0[j0_h0:j1_h0], (pl_h0, pr_h0), mode="edge") \
            if len(h_tvt0) > 0 else np.zeros(self.config.H_H)
        h_seg_feat0 = np.pad(h_feat0[j0_h0:j1_h0], ((pl_h0, pr_h0), (0, 0)), mode="edge") \
            if len(h_feat0) > 0 else np.zeros((self.config.H_H, self.K_H))

        j0_h1, j1_h1, pl_h1, pr_h1 = get_crop_index_and_pad_1d(
            len(h_tvt1), 0, history=0, future=self.config.H_F)

        h_seg_mask1 = np.pad(np.ones(j1_h1 - j0_h1), (pl_h1, pr_h1))
        h_seg_tvt1  = np.pad(h_tvt1[j0_h1:j1_h1], (pl_h1, pr_h1), mode="edge") \
            if len(h_tvt1) > 0 else np.zeros(self.config.H_F)
        h_seg_feat1 = np.pad(h_feat1[j0_h1:j1_h1], ((pl_h1, pr_h1), (0, 0)), mode="edge") \
            if len(h_feat1) > 0 else np.zeros((self.config.H_F, self.K_H))

        h_seg_mask = np.concatenate([h_seg_mask0, h_seg_mask1])
        h_seg_tvt  = np.concatenate([h_seg_tvt0, h_seg_tvt1])
        h_seg_feat = np.concatenate([h_seg_feat0, h_seg_feat1], axis=0)
        h_seg_gr   = h_seg_feat[:, 0]

        history_mask = np.zeros(self.config.H_H + self.config.H_F, dtype=np.float32)
        true_h0_len = j1_h0 - j0_h0
        if true_h0_len > 0:
            history_mask[pl_h0: pl_h0 + true_h0_len] = 1.0

        eval_mask_h = np.zeros(self.config.H_H + self.config.H_F, dtype=np.float32)
        true_h1_len = j1_h1 - j0_h1
        if true_h1_len > 0:
            eval_mask_h[self.config.H_H + pl_h1: self.config.H_H + pl_h1 + true_h1_len] = 1.0

        sdf = (h_seg_tvt[None, :] - t_seg_tvt[:, None]) / self.config.TVT_DIFF_SCALE
        clip = self.config.SDF_CLIP
        sdf = np.clip(sdf, -clip, clip)

        diff = np.abs(t_seg_tvt[:, None] - h_seg_tvt[None, :])
        matched = diff.argmin(0)
        matched_mask = (diff.min(0) < 1).astype(np.float32) * h_seg_mask

        H = len(h_seg_gr)
        T = len(t_seg_gr)
        label = np.zeros((T, H), dtype=np.float32)
        for i in range(H - 1):
            if matched_mask[i] == 0 or matched_mask[i + 1] == 0:
                continue
            cv2.line(label, (i, matched[i]), (i + 1, matched[i + 1]), 1.0, self.config.SEG_LINE_THICKNESS, cv2.LINE_AA)

        history = label.copy()
        history[:, self.config.H_H:] = 0
        target = matched

        orig_tvt = h["TVT"].values if ("TVT" in h.columns and h["TVT"].notna().sum() > 0) \
            else np.zeros(len(h))
        orig_len = len(orig_tvt)
        padded_orig = np.zeros(self.config.ORIG_PAD_LEN, dtype=np.float32)
        use_len = min(orig_len, self.config.ORIG_PAD_LEN)
        padded_orig[:use_len] = orig_tvt[:use_len]
        prebin_len = min(len(h_gr_prebin), self.config.ORIG_PAD_LEN)
        padded_h_gr_prebin = np.zeros(self.config.ORIG_PAD_LEN, dtype=np.float32)
        padded_h_gr_prebin[:prebin_len] = h_gr_prebin[:prebin_len]

        anchor_t_idx = self.config.T_H - 1

        return {
            "id": sample_id,
            "t_gr":   torch.tensor(t_seg_gr, dtype=torch.float32),
            "h_gr":   torch.tensor(h_seg_gr, dtype=torch.float32),
            "h_gr_prebin": torch.tensor(padded_h_gr_prebin, dtype=torch.float32),
            "h_prebin_len": torch.tensor(prebin_len, dtype=torch.int64),
            "h_prebin_ps": torch.tensor(split_ps_base, dtype=torch.int64),
            "h_prebin_offset": torch.tensor(h_bin_offset, dtype=torch.int64),
            "h_prebin_use_mean": torch.tensor(not self.is_train, dtype=torch.bool),
            "h_prebin_flipped": torch.tensor(False, dtype=torch.bool),
            "t_feat": torch.tensor(t_seg_feat, dtype=torch.float32).T,
            "h_feat": torch.tensor(h_seg_feat, dtype=torch.float32).T,
            "history":      torch.tensor(history, dtype=torch.float32).unsqueeze(0),
            "label":        torch.tensor(label, dtype=torch.float32).unsqueeze(0),
            "target":       torch.tensor(target, dtype=torch.int64),
            "h_mask":       torch.tensor(h_seg_mask, dtype=torch.float32),
            "t_mask":       torch.tensor(t_seg_mask, dtype=torch.float32),
            "sdf":          torch.tensor(sdf, dtype=torch.float32).unsqueeze(0),
            "eval_mask":    torch.tensor(eval_mask_h, dtype=torch.float32),
            "t_seg_tvt":    torch.tensor(t_seg_tvt, dtype=torch.float32),
            "h_seg_tvt":    torch.tensor(h_seg_tvt, dtype=torch.float32),
            "history_mask": torch.tensor(history_mask, dtype=torch.float32),
            "matched_idx":  torch.tensor(matched, dtype=torch.int64),
            "orig_tvt":     torch.tensor(padded_orig, dtype=torch.float32),
            "orig_len":     torch.tensor(orig_len, dtype=torch.int64),
            "h_ps":         torch.tensor(split_ps, dtype=torch.int64),
            "true_ps":      torch.tensor(true_ps, dtype=torch.int64),
            "h_s":          torch.tensor(h_step, dtype=torch.int64),
            "anchor_t_idx": torch.tensor(anchor_t_idx, dtype=torch.int64),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GR Noise Transfer Augmentation
# ─────────────────────────────────────────────────────────────────────────────
class GRNoiseAugDataset(Dataset):
    """Dataset wrapper: 1× original + ``n_synth`` synthetic GR-noise twins per well.

    For synthetic copy of well A with donor C:

        h_gr[future]  = h_gr_sim(A) + h_gr_noise(C)
        h_gr[history] = unchanged

    where
        h_gr_sim(A)   = typewell GR interpolated at well-A's true TVT path
        h_gr_noise(C) = smooth(h_gr_obs(C)) - h_gr_sim(C),  re-sampled to A's TVT grid
        C             = random donor (or KNN neighbour if donor="knn"), ≠ A

    Layout (N = len(base), S = n_synth):
        [0, N)           → original base[i]
        [N + s*N, N+(s+1)*N) → synth copy s∈[0,S) of well (idx % N),
                               donors for the S copies of the same well are
                               sampled distinct when possible.

    The noise cache is built lazily inside each DataLoader worker.
    """

    def __init__(
        self,
        base_dataset: WellboreSDFDataset,
        *,
        donor: str = "random",
        k_neighbors: int = 5,
        future_only: bool = True,
        n_synth: int = 2,
    ):
        self.base = base_dataset
        self.donor = donor
        self.k = k_neighbors
        self.future_only = future_only
        self.n_synth = max(0, int(n_synth))
        self._h_h = base_dataset.config.H_H
        self._knn = self._build_knn() if donor == "knn" else None
        self._noise_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Worker-local: distinct donors for all synth copies of well A until
        # the original sample is fetched again (then refreshed).
        self._donor_group: dict[int, list[int]] = {}

    # ── init helpers ────────────────────────────────────────────────────────

    def _build_knn(self) -> np.ndarray:
        """Return (N, k) int array of k nearest neighbour well indices per well."""
        from sklearn.neighbors import NearestNeighbors

        coords = np.array(
            [well_xy_centroid(f) for f in self.base.well_files], dtype=np.float64
        )
        N = len(coords)
        k_actual = min(self.k, N - 1)  # guard tiny datasets

        if k_actual < 1:
            # Single-well dataset: each well is its own neighbour (no-op noise)
            return np.zeros((N, 1), dtype=np.int64)

        nn = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree")
        nn.fit(coords)
        indices = nn.kneighbors(coords, return_distance=False)
        # Column 0 is self; drop it
        neighbours = indices[:, 1:k_actual + 1].astype(np.int64)
        return neighbours  # shape (N, k_actual)

    def _pick_donor(self, a_idx: int, *, exclude: set[int] | None = None) -> int:
        N = len(self.base)
        exclude = set(exclude or ())
        exclude.add(a_idx)
        if self.donor == "knn":
            pool = [int(i) for i in self._knn[a_idx] if int(i) not in exclude]
            if not pool:
                pool = [int(i) for i in self._knn[a_idx]]
            if not pool:
                return a_idx
            return int(pool[np.random.randint(len(pool))])
        pool = [i for i in range(N) if i not in exclude]
        if not pool:
            return a_idx
        return int(pool[np.random.randint(len(pool))])

    def _pick_synth_donors(self, a_idx: int) -> list[int]:
        """Pick ``n_synth`` donors for well A, distinct when the pool allows."""
        donors: list[int] = []
        used: set[int] = set()
        for _ in range(self.n_synth):
            c = self._pick_donor(a_idx, exclude=used)
            donors.append(c)
            used.add(c)
        return donors

    # ── noise cache ─────────────────────────────────────────────────────────

    def _get_noise(self, c_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (tvt_sorted, noise_sorted) for donor well c_idx (cached)."""
        if c_idx not in self._noise_cache:
            self._noise_cache[c_idx] = self._compute_noise(c_idx)
        return self._noise_cache[c_idx]

    def _compute_noise(self, c_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load donor well C and compute GR noise on original-resolution data.

        Returns arrays sorted by TVT so np.interp works directly.
        """
        wf = self.base.well_files[c_idx]
        sample_id = wf.name.split("__")[0]
        typewell_path = wf.parent / f"{sample_id}{Config.TYPEWELL_SUFFIX}"

        h = pd.read_csv(wf)
        t = pd.read_csv(typewell_path)

        tw_tvt = t["TVT"].values.astype(np.float64)
        tw_gr = (
            t["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values.astype(np.float64)
        )

        if "TVT" not in h.columns or h["TVT"].isna().all():
            return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        h_tvt = pd.to_numeric(h["TVT"], errors="coerce").values
        h_gr_raw = h["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values

        valid = np.isfinite(h_tvt) & np.isfinite(h_gr_raw)
        h_tvt = h_tvt[valid]
        h_gr_raw = h_gr_raw[valid]

        if len(h_tvt) < 5:
            return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        # Smooth with the same default window used during training
        win = self.base.config.H_GR_FILTER[0]
        if win % 2 == 0:
            win += 1
        if len(h_gr_raw) > win:
            h_gr_smooth = savgol_filter(h_gr_raw, win, 2).astype(np.float64)
        else:
            h_gr_smooth = h_gr_raw.copy()

        # h_gr_sim(C): typewell GR interpolated at C's TVT
        tw_order = np.argsort(tw_tvt)
        h_gr_sim = np.interp(h_tvt, tw_tvt[tw_order], tw_gr[tw_order])
        noise = h_gr_smooth - h_gr_sim

        # Sort by TVT for downstream np.interp
        sort_idx = np.argsort(h_tvt)
        return h_tvt[sort_idx].astype(np.float32), noise[sort_idx].astype(np.float32)

    # ── Dataset API ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return (1 + self.n_synth) * len(self.base)

    def __getitem__(self, idx: int):
        N = len(self.base)
        if idx < N:
            self._donor_group.pop(idx, None)
            return self.base[idx]

        rem = idx - N
        synth_i = rem // N
        a = rem % N
        sample = self.base[a]
        if a not in self._donor_group or len(self._donor_group[a]) != self.n_synth:
            self._donor_group[a] = self._pick_synth_donors(a)
        return self._apply_noise_aug(sample, a, c_idx=self._donor_group[a][synth_i])

    # ── augmentation ────────────────────────────────────────────────────────

    def _apply_noise_aug(self, sample: dict, a_idx: int, *, c_idx: int | None = None) -> dict:
        """Replace h_gr in future columns with h_gr_sim(A) + h_gr_noise(C)."""
        if c_idx is None:
            c_idx = self._pick_donor(a_idx)
        tvt_C, noise_C = self._get_noise(c_idx)

        h_tvt_A = sample["h_seg_tvt"].numpy()   # (H,) may be flipped / PFE-shifted
        h_mask  = sample["h_mask"].numpy()       # 1 = valid data column

        # h_gr_sim(A): re-interpolate typewell GR at A's current TVT grid
        t_tvt_A = sample["t_seg_tvt"].numpy()
        t_gr_A  = sample["t_feat"][0].numpy()   # t_feat shape (K_T, T) → row 0 = GR
        tw_order = np.argsort(t_tvt_A)
        h_gr_sim_A = np.interp(h_tvt_A, t_tvt_A[tw_order], t_gr_A[tw_order])

        # Resample noise from C onto A's TVT grid (TVT-aligned transfer)
        noise_at_A = np.interp(h_tvt_A, tvt_C, noise_C,
                               left=float(noise_C[0]),
                               right=float(noise_C[-1]))

        h_gr_orig = sample["h_gr"].numpy()
        h_gr_aug = h_gr_orig.copy()
        replace_mask = h_mask > 0
        if self.future_only:
            replace_mask &= np.arange(len(h_mask)) >= self._h_h
        h_gr_aug[replace_mask] = (h_gr_sim_A + noise_at_A)[replace_mask].astype(np.float32)

        sample = dict(sample)  # shallow copy so we don't mutate base's cache
        h_gr_t = torch.tensor(h_gr_aug, dtype=torch.float32)
        sample["h_gr"] = h_gr_t
        # h_feat[0, :] must mirror h_gr (channel 0 = GR in horizontal features)
        h_feat = sample["h_feat"].clone()
        h_feat[0, :] = h_gr_t
        sample["h_feat"] = h_feat

        # Rebuild the synthetic future at original H resolution so the learned
        # descriptor is extracted before H_S sampling.
        prebin_len = int(sample["h_prebin_len"])
        prebin_ps = min(int(sample["h_prebin_ps"]), prebin_len - 1)
        if prebin_len > 0 and prebin_ps < prebin_len - 1:
            prebin_tvt = sample["orig_tvt"][:prebin_len].numpy()
            sim_prebin = np.interp(
                prebin_tvt, t_tvt_A[tw_order], t_gr_A[tw_order]
            )
            noise_prebin = np.interp(
                prebin_tvt,
                tvt_C,
                noise_C,
                left=float(noise_C[0]),
                right=float(noise_C[-1]),
            )
            h_gr_prebin = sample["h_gr_prebin"].clone()
            h_gr_prebin[prebin_ps + 1 : prebin_len] = torch.from_numpy(
                (sim_prebin + noise_prebin)[prebin_ps + 1 :].astype(np.float32)
            )
            sample["h_gr_prebin"] = h_gr_prebin
        return sample