from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

VP_PREFIX_WINDOWS = (16, 32, 64)
VP_ENDPOINT_OFFSETS = (-24.0, -12.0, 0.0, 12.0, 24.0)
VP_SLOPE_OFFSETS = (-0.012, 0.0, 0.012)
VP_CURVATURE_OFFSETS = (-10.0, 0.0, 10.0)
VP_GR_WINDOW = 31
VP_HUBER_SCALE = 18.0
VP_DIVERSE_TOPK = 60
VP_DIVERSE_SCORE_HEAD = 12
VP_DIVERSE_FAMILY_QUOTA = 6
VP_DIVERSE_DELTA_BIN = 6.0
FORMATION_COLS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
_SURFACE_PRIOR_CACHE: dict[str, dict[str, object]] = {}


def _finite_xy(frame: pd.DataFrame) -> np.ndarray:
    if "X" not in frame.columns or "Y" not in frame.columns:
        return np.zeros((0, 2), dtype=np.float64)
    x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(np.float64)
    return np.column_stack([x, y])


def _build_train_space_surface_prior(
    data_root: str | Path,
    *,
    formation_cols: tuple[str, ...] = FORMATION_COLS,
    samples_per_well: int = 160,
    exclude_wells: tuple[str, ...] = (),
) -> dict[str, object]:
    """Fit a deploy-safe XY -> formation-top prior from train wells only."""

    root = Path(data_root)
    train_dir = root / "train"
    exclude = {str(x) for x in exclude_wells}
    xy_parts: list[np.ndarray] = []
    well_parts: list[np.ndarray] = []
    val_parts: dict[str, list[np.ndarray]] = {col: [] for col in formation_cols}
    usecols = {"X", "Y", *formation_cols}
    for path in sorted(train_dir.glob("*__horizontal_well.csv")):
        wid = path.name.replace("__horizontal_well.csv", "")
        if wid in exclude:
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c in usecols)
        except Exception:
            continue
        if "X" not in df.columns or "Y" not in df.columns:
            continue
        present = [col for col in formation_cols if col in df.columns]
        if not present:
            continue
        xy = _finite_xy(df)
        mask = np.isfinite(xy).all(axis=1)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        if len(idx) > int(samples_per_well) > 0:
            idx = idx[np.linspace(0, len(idx) - 1, int(samples_per_well), dtype=int)]
        xy_parts.append(xy[idx])
        well_parts.append(np.full(len(idx), wid, dtype=object))
        for col in formation_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)[idx]
            else:
                vals = np.full(len(idx), np.nan, dtype=np.float64)
            val_parts[col].append(vals)
    if not xy_parts:
        return {
            "xy": np.zeros((0, 2), dtype=np.float64),
            "scale": np.ones(2, dtype=np.float64),
            "values": {col: np.zeros(0, dtype=np.float64) for col in formation_cols},
            "sample_wells": np.zeros(0, dtype=object),
            "tree": None,
        }
    xy_all = np.vstack(xy_parts).astype(np.float64)
    scale = np.nanstd(xy_all, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    values = {col: np.concatenate(parts).astype(np.float64) if parts else np.zeros(0, dtype=np.float64) for col, parts in val_parts.items()}
    sample_wells = np.concatenate(well_parts).astype(object) if well_parts else np.zeros(0, dtype=object)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy_all / scale)
    except Exception:
        tree = None
    return {"xy": xy_all, "scale": scale, "values": values, "sample_wells": sample_wells, "tree": tree}


def _predict_surface_tops_from_prior(
    prior: dict[str, object],
    hw: pd.DataFrame,
    *,
    formation_cols: tuple[str, ...] = FORMATION_COLS,
    k: int = 32,
    power: float = 2.0,
    exclude_wells: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Query train-space formation surfaces for every row of a target well."""

    q = _finite_xy(hw)
    n = len(hw)
    out = pd.DataFrame(index=np.arange(n))
    xy = np.asarray(prior.get("xy", np.zeros((0, 2))), dtype=np.float64)
    if n == 0 or len(xy) == 0 or len(q) != n:
        for col in formation_cols:
            out[col] = np.nan
        return out
    scale = np.asarray(prior.get("scale", np.ones(2)), dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    tree = prior.get("tree")
    sample_wells = np.asarray(prior.get("sample_wells", np.zeros(0, dtype=object)), dtype=object)
    exclude = {str(x) for x in exclude_wells if str(x)}
    kk = max(1, min(int(k), len(xy)))
    query_k = kk
    if exclude and len(sample_wells) == len(xy):
        n_ex = int(np.isin(sample_wells, list(exclude)).sum())
        query_k = max(kk, min(len(xy), kk + n_ex))
    if tree is not None:
        dist, idx = tree.query(q / scale, k=query_k)
    else:
        base = xy / scale
        qq = q / scale
        dist = np.empty((len(qq), query_k), dtype=np.float64)
        idx = np.empty((len(qq), query_k), dtype=np.int64)
        for i, row in enumerate(qq):
            d = np.sqrt(np.sum((base - row[None, :]) ** 2, axis=1))
            order = np.argsort(d)[:query_k]
            dist[i] = d[order]
            idx[i] = order
    if query_k == 1:
        dist = np.asarray(dist).reshape(-1, 1)
        idx = np.asarray(idx).reshape(-1, 1)
    if exclude and len(sample_wells) == len(xy):
        keep = ~np.isin(sample_wells[idx], list(exclude))
    else:
        keep = np.ones_like(idx, dtype=bool)
    values = prior.get("values", {})
    for col in formation_cols:
        vals = np.asarray(values.get(col, np.zeros(0)), dtype=np.float64)
        pred = np.full(n, np.nan, dtype=np.float64)
        if len(vals) == len(xy):
            neigh = vals[idx]
            valid = np.isfinite(neigh) & np.isfinite(dist) & keep
            weights = 1.0 / np.maximum(dist, 1e-6) ** float(power)
            weights = np.where(valid, weights, 0.0)
            denom = weights.sum(axis=1)
            good = denom > 1e-12
            pred[good] = (weights[good] * np.where(valid[good], neigh[good], 0.0)).sum(axis=1) / denom[good]
        if np.isfinite(pred).any():
            pred = pd.Series(pred).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float64)
        out[col] = pred
    return out


def _surface_tops_for_hw(data_root: str | Path, hw: pd.DataFrame, wid: str | None = None) -> pd.DataFrame | None:
    if all(col in hw.columns for col in FORMATION_COLS):
        return hw[list(FORMATION_COLS)].copy()
    root = str(Path(data_root))
    prior = _SURFACE_PRIOR_CACHE.get(root)
    if prior is None:
        samples = int(os.environ.get("ROGII_SURFACE_PRIOR_SAMPLES_PER_WELL", "160") or "160")
        prior = _build_train_space_surface_prior(data_root, samples_per_well=samples)
        _SURFACE_PRIOR_CACHE[root] = prior
    exclude_self = str(os.environ.get("ROGII_SURFACE_PRIOR_EXCLUDE_SELF", "0")).lower() in {"1", "true", "yes", "y"}
    exclude = (str(wid),) if exclude_self and wid is not None else ()
    tops = _predict_surface_tops_from_prior(prior, hw, formation_cols=FORMATION_COLS, exclude_wells=exclude)
    if not np.isfinite(tops.to_numpy(np.float64)).any():
        return None
    return tops


def _fill(values, fill_value=None) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64)).interpolate(limit_direction="both")
    if fill_value is None:
        arr = series.to_numpy(dtype=np.float64)
        fill_value = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
    return series.fillna(float(fill_value)).to_numpy(dtype=np.float64)


def _roll(values, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    w = max(1, min(int(window), len(arr)))
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return arr.astype(np.float64, copy=True)
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.pad(arr, (pad, pad), mode="edge"), kernel, mode="valid")


def _corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return float("nan")
    x = aa[mask] - float(np.mean(aa[mask]))
    y = bb[mask] - float(np.mean(bb[mask]))
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 1e-12 else float("nan")


def _huber_score(resid, scale: float = VP_HUBER_SCALE) -> float:
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("inf")
    c = max(float(scale), 1e-6)
    a = np.abs(r)
    loss = np.where(a <= c, 0.5 * r * r, c * (a - 0.5 * c))
    return float(2.0 * np.mean(loss))


def _typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = tw[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT").drop_duplicates("TVT")
    tvt = frame["TVT"].to_numpy(np.float64)
    gr = _roll(_fill(frame["GR"].to_numpy(np.float64)), VP_GR_WINDOW)
    return tvt, gr


def _fit_heel_affine(expected, observed) -> tuple[float, float, float, int]:
    x = np.asarray(expected, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.nanstd(x)) < 1e-8:
        intercept = float(np.nanmedian(y - x)) if len(x) else 0.0
        pred = x + intercept
        rmse = float(np.sqrt(np.mean((y - pred) ** 2))) if len(x) else float("inf")
        return 1.0, intercept, rmse, int(len(x))
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    var = float(np.mean((x - xm) ** 2))
    raw_slope = float(np.mean((x - xm) * (y - ym)) / max(var, 1e-12))
    raw_slope = float(np.clip(raw_slope, 0.35, 2.50))
    weight = len(x) / (len(x) + 80.0)
    slope = float((1.0 - weight) + weight * raw_slope)
    intercept = float(np.median(y - slope * x))
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return slope, intercept, rmse, int(len(x))


def _tail_rate(md, values, *, tail: bool, window: int) -> float:
    m = np.asarray(md, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(len(m), len(v), int(window))
    if n < 2:
        return 0.0
    mm = m[-n:] if tail else m[:n]
    vv = v[-n:] if tail else v[:n]
    dm = float(mm[-1] - mm[0])
    if abs(dm) < 1e-9:
        return 0.0
    return float((vv[-1] - vv[0]) / dm)


def _compose_centered(level: float, shape_tvt: np.ndarray) -> np.ndarray:
    shape = np.asarray(shape_tvt, dtype=np.float64)
    if len(shape) == 0:
        return shape.copy()
    return float(level) + (shape - float(np.mean(shape)))


def _append_unique(out: list[dict[str, object]], seen: set[bytes], *, name: str, family: str, tvt: np.ndarray) -> None:
    arr = np.asarray(tvt, dtype=np.float64)
    if len(arr) == 0:
        return
    if not np.isfinite(arr).all():
        arr = pd.Series(arr).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float64)
    if not np.isfinite(arr).all():
        return
    key = np.round(arr, 3).astype(np.float32).tobytes()
    if key in seen:
        return
    seen.add(key)
    out.append({"name": str(name), "family": str(family), "tvt": arr.astype(np.float64)})


def _level_grid_candidates(anchor_tvt: np.ndarray, z_hidden: np.ndarray, seen: set[bytes]) -> list[dict[str, object]]:
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    z_hidden = np.asarray(z_hidden, dtype=np.float64)
    if len(anchor_tvt) == 0 or len(z_hidden) != len(anchor_tvt):
        return []
    progress = np.linspace(0.0, 1.0, len(anchor_tvt), dtype=np.float64)
    out: list[dict[str, object]] = []
    for off in (-36.0, -24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0, 36.0):
        _append_unique(out, seen, name=f"level_grid:const:{off:+.1f}", family="level_grid", tvt=anchor_tvt + float(off))
        _append_unique(out, seen, name=f"level_grid:ramp:{off:+.1f}", family="level_grid", tvt=anchor_tvt + float(off) * progress)
    return out


def _prefix_u_rate_candidates(
    known_md: np.ndarray,
    known_tvt: np.ndarray,
    known_z: np.ndarray,
    hidden_md: np.ndarray,
    hidden_z: np.ndarray,
    anchor_tvt: np.ndarray,
    seen: set[bytes],
) -> list[dict[str, object]]:
    known_md = np.asarray(known_md, dtype=np.float64)
    known_tvt = np.asarray(known_tvt, dtype=np.float64)
    known_z = np.asarray(known_z, dtype=np.float64)
    hidden_md = np.asarray(hidden_md, dtype=np.float64)
    hidden_z = np.asarray(hidden_z, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    ok = np.isfinite(known_md) & np.isfinite(known_tvt) & np.isfinite(known_z)
    known_md = known_md[ok]
    known_tvt = known_tvt[ok]
    known_z = known_z[ok]
    if len(known_md) < 2 or len(hidden_md) == 0 or len(anchor_tvt) == 0:
        return []
    known_u = known_tvt + known_z
    last_md = float(known_md[-1])
    last_u = float(known_u[-1])
    windows = [32, 64, 128, 256, 512, len(known_md)]
    rate_offsets = (-0.012, -0.006, 0.0, 0.006, 0.012)
    blends = (0.0, 0.5)
    out: list[dict[str, object]] = []
    for win in windows:
        base_rate = _tail_rate(known_md, known_u, tail=True, window=int(win))
        for off in rate_offsets:
            rate = float(base_rate) + float(off)
            u_line = last_u + rate * (hidden_md - last_md)
            tvt_line = u_line - hidden_z
            for blend in blends:
                b = float(blend)
                tvt = (1.0 - b) * tvt_line + b * _compose_centered(float(np.mean(tvt_line)), anchor_tvt)
                _append_unique(out, seen, name=f"prefix_u_rate:w{int(win)}:r{float(off):+0.4f}:b{b:.2f}", family="prefix_u_rate", tvt=tvt)
    return out


def _formation_contact_candidates(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    row_idx: np.ndarray,
    start_row: int,
    anchor_tvt: np.ndarray,
    z_hidden: np.ndarray,
    seen: set[bytes],
    surface_tops: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    top_source = surface_tops if surface_tops is not None else hw
    if top_source is None or top_source.empty:
        return []
    row_idx = np.asarray(row_idx, dtype=np.int64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    z_hidden = np.asarray(z_hidden, dtype=np.float64)
    if len(row_idx) == 0 or len(anchor_tvt) == 0:
        return []
    tvt_input = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_z = hw["Z"].to_numpy(np.float64)[:start_row]
    known_mask = np.isfinite(tvt_input) & np.isfinite(known_z)
    if int(known_mask.sum()) < 8:
        return []
    has_geology = (not tw.empty) and "Geology" in tw.columns and "TVT" in tw.columns
    geo = tw.dropna(subset=["Geology", "TVT"]) if has_geology else pd.DataFrame()
    out: list[dict[str, object]] = []
    for ref_col in FORMATION_COLS:
        if ref_col not in top_source.columns:
            continue
        top_all = pd.to_numeric(top_source[ref_col], errors="coerce").to_numpy(np.float64)
        if len(top_all) < len(hw):
            top_all = np.pad(top_all, (0, len(hw) - len(top_all)), constant_values=np.nan)
        top_known = top_all[:start_row][known_mask]
        top_hidden = top_all[row_idx]
        if int(np.isfinite(top_known).sum()) < 8 or int(np.isfinite(top_hidden).sum()) < max(4, len(top_hidden) // 5):
            continue
        ref_vals = geo.loc[geo["Geology"].astype(str).eq(ref_col), "TVT"].to_numpy(dtype=np.float64) if not geo.empty else np.asarray([], dtype=np.float64)
        finite_known = np.isfinite(top_known)
        if len(ref_vals):
            ref_tvt = float(np.nanmin(ref_vals))
            base_known = ref_tvt - (known_z[known_mask][finite_known] - top_known[finite_known])
            offset = float(np.nanmedian(tvt_input[known_mask][finite_known] - base_known))
            tvt = ref_tvt - (z_hidden - top_hidden) + offset
            name = f"formation_contact:{ref_col}"
        else:
            contact_const = float(np.nanmedian(tvt_input[known_mask][finite_known] + known_z[known_mask][finite_known] - top_known[finite_known]))
            tvt = contact_const + top_hidden - z_hidden
            name = f"formation_contact_prior:{ref_col}"
        _append_unique(out, seen, name=name, family="formation_contact", tvt=tvt)
        centered = _compose_centered(float(np.nanmean(tvt)), anchor_tvt)
        _append_unique(out, seen, name=f"{name}:anchor_shape", family="formation_contact", tvt=centered)
    return out


def _self_gr_estimate(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    hidden_gr: np.ndarray,
    *,
    half_width: int,
    stride: int,
    hidden_step: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(known_gr) < 2 * int(half_width) + 4 or len(hidden_gr) < 2:
        return None
    kgr = _roll(_fill(known_gr), 5)
    hgr = _roll(_fill(hidden_gr), 5)
    win = 2 * int(half_width) + 1
    starts = np.arange(0, len(kgr) - win + 1, max(int(stride), 1), dtype=np.int32)
    if len(starts) < 2:
        return None
    cand = kgr[starts[:, None] + np.arange(win)[None, :]]
    cand = (cand - cand.mean(axis=1, keepdims=True)) / (cand.std(axis=1, keepdims=True) + 1e-6)
    centers = np.clip(starts + int(half_width), 0, len(known_tvt) - 1)
    cand_tvt = np.asarray(known_tvt, dtype=np.float64)[centers]
    anchors = np.arange(0, len(hgr), max(int(hidden_step), 1), dtype=np.int32)
    pad = np.pad(hgr, int(half_width), mode="edge")
    hidden = pad[anchors[:, None] + np.arange(win)[None, :]]
    hidden = (hidden - hidden.mean(axis=1, keepdims=True)) / (hidden.std(axis=1, keepdims=True) + 1e-6)
    score = hidden @ cand.T / float(win)
    best = score.argmax(axis=1)
    est_anchor = cand_tvt[best]
    score_anchor = score[np.arange(len(anchors)), best]
    full_tvt = np.interp(np.arange(len(hgr)), anchors, est_anchor)
    full_score = np.interp(np.arange(len(hgr)), anchors, score_anchor)
    return full_tvt.astype(np.float64), full_score.astype(np.float64)


def _selfgr_candidates(
    known_tvt: np.ndarray,
    known_gr: np.ndarray,
    hidden_gr: np.ndarray,
    anchor_tvt: np.ndarray,
    *,
    last_known: float,
    seen: set[bytes],
) -> list[dict[str, object]]:
    known_tvt = np.asarray(known_tvt, dtype=np.float64)
    known_gr = np.asarray(known_gr, dtype=np.float64)
    hidden_gr = np.asarray(hidden_gr, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    if len(hidden_gr) < 20 or len(anchor_tvt) == 0:
        return []
    mask0 = np.isfinite(known_tvt) & np.isfinite(known_gr)
    if int(mask0.sum()) < 80:
        return []
    out: list[dict[str, object]] = []
    level_anchor = float(np.mean(anchor_tvt))
    for band in (120.0, 220.0):
        mask = mask0 & (known_tvt >= float(last_known) - float(band)) & (known_tvt <= float(last_known) + float(band))
        if int(mask.sum()) < 80:
            continue
        kt = known_tvt[mask]
        kg = known_gr[mask]
        order = np.argsort(kt)
        kt = kt[order]
        kg = kg[order]
        estimates: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for hwid in (8, 16, 28):
            est = _self_gr_estimate(kg, kt, hidden_gr, half_width=int(hwid), stride=3, hidden_step=18)
            if est is None:
                continue
            est_tvt, score = est
            smooth = _roll(est_tvt, min(101, max(1, len(est_tvt) // 3 * 2 + 1)))
            _append_unique(out, seen, name=f"selfgr:direct:b{int(band)}:w{int(hwid)}", family="selfgr", tvt=smooth)
            _append_unique(out, seen, name=f"selfgr:anchor_level:b{int(band)}:w{int(hwid)}", family="selfgr", tvt=_compose_centered(level_anchor, smooth))
            estimates.append(smooth)
            weights.append(score)
        if estimates:
            est_arr = np.vstack(estimates)
            score_arr = np.vstack(weights)
            wt = np.exp(4.0 * np.nan_to_num(score_arr))
            wt /= wt.sum(axis=0, keepdims=True) + 1e-9
            blend = (wt * est_arr).sum(axis=0)
            _append_unique(out, seen, name=f"selfgr:direct_blend:b{int(band)}", family="selfgr", tvt=blend)
            _append_unique(out, seen, name=f"selfgr:anchor_level_blend:b{int(band)}", family="selfgr", tvt=_compose_centered(level_anchor, blend))
    return out


def _piecewise_candidates(md, z, anchor_tvt, last_known: float) -> list[dict[str, object]]:
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64).copy()
    if len(md) == 0:
        return []
    anchor_tvt[0] = float(last_known)
    n = len(md)
    knots = np.unique(np.round(np.linspace(0, n - 1, min(12, max(2, n)))).astype(int))
    rows = np.arange(n, dtype=np.float64)
    progress = np.zeros(n, dtype=np.float64)
    span = float(md[-1] - md[0])
    if abs(span) > 1e-8:
        progress = np.clip((md - float(md[0])) / span, 0.0, 1.0)
    anchor_u = anchor_tvt + z
    out: list[dict[str, object]] = []
    seen: set[bytes] = set()

    def add(name: str, perturb: np.ndarray, family: str) -> None:
        ctrl_u = anchor_u[knots] + perturb[knots]
        u = np.interp(rows, knots.astype(np.float64), ctrl_u)
        u[0] = float(last_known) + float(z[0])
        tvt = u - z
        tvt[0] = float(last_known)
        key = np.round(tvt, 3).astype(np.float32).tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "family": family, "tvt": tvt.astype(np.float64)})

    add("anchor_exact", np.zeros(n, dtype=np.float64), "anchor")
    for level in VP_ENDPOINT_OFFSETS:
        add(f"level_shift:{level:+.1f}", np.full(n, float(level), dtype=np.float64), "level_shift")
    for endpoint in VP_ENDPOINT_OFFSETS:
        for slope in VP_SLOPE_OFFSETS:
            for curve in VP_CURVATURE_OFFSETS:
                perturb = float(endpoint) * progress + float(slope) * (md - float(md[0])) + float(curve) * np.sin(np.pi * progress)
                add(f"piecewise_u:e{endpoint:+.1f}:s{slope:+.4f}:c{curve:+.1f}", perturb, "piecewise_u")
    return out


def _score_path(tvt, obs_gr, tw_tvt, tw_gr, affine) -> dict[str, float]:
    exp = np.interp(np.asarray(tvt, dtype=np.float64), tw_tvt, tw_gr)
    slope, intercept, prefix_rmse, prefix_n = affine
    pred_gr = float(slope) * exp + float(intercept)
    obs = np.asarray(obs_gr, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(pred_gr)
    if int(mask.sum()) < 8:
        return {"score": float("inf"), "gr_rmse": float("inf"), "gr_corr": float("nan"), "prefix_rmse": float(prefix_rmse), "prefix_n": int(prefix_n)}
    resid = obs[mask] - pred_gr[mask]
    rmse = float(np.sqrt(np.mean(resid * resid)))
    corr = _corr(obs[mask], pred_gr[mask])
    score = _huber_score(resid)
    if np.isfinite(corr):
        score *= float(1.0 + 0.20 * max(0.0, 1.0 - corr))
    return {"score": float(score), "gr_rmse": rmse, "gr_corr": corr, "prefix_rmse": float(prefix_rmse), "prefix_n": int(prefix_n)}


def _percentile_cost(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    n, m = mat.shape
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    costs = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        col = mat[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        work = np.where(finite, col, float(np.nanmax(col[finite]) + 1.0))
        order = np.argsort(work, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        costs[:, j] = ranks / float(max(n - 1, 1))
    return costs.mean(axis=1)


def _prefix_vector(hw: pd.DataFrame, row_idx: np.ndarray, cand_tvt: np.ndarray) -> np.ndarray:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    cand_tvt = np.asarray(cand_tvt, dtype=np.float64)
    if len(row_idx) == 0 or len(cand_tvt) == 0:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    start_row = int(row_idx[0])
    tvt_input = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_mask = np.isfinite(tvt_input)
    if int(known_mask.sum()) < 2:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    known_md = hw["MD"].to_numpy(np.float64)[:start_row][known_mask]
    known_z = hw["Z"].to_numpy(np.float64)[:start_row][known_mask]
    known_tvt = tvt_input[known_mask]
    hidden_md = hw["MD"].to_numpy(np.float64)[row_idx]
    hidden_z = hw["Z"].to_numpy(np.float64)[row_idx]
    n = min(len(hidden_md), len(hidden_z), len(cand_tvt))
    if n == 0:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    hidden_md = hidden_md[:n]
    hidden_z = hidden_z[:n]
    cand_tvt = cand_tvt[:n]
    known_f = known_tvt + known_z
    cand_f = cand_tvt + hidden_z
    md_gap = float(hidden_md[0] - known_md[-1])
    vals: list[float] = []
    for win in VP_PREFIX_WINDOWS:
        ktvt_rate = _tail_rate(known_md, known_tvt, tail=True, window=int(win))
        ctvt_rate = _tail_rate(hidden_md, cand_tvt, tail=False, window=int(win))
        kf_rate = _tail_rate(known_md, known_f, tail=True, window=int(win))
        cf_rate = _tail_rate(hidden_md, cand_f, tail=False, window=int(win))
        vals.extend(
            [
                abs(float(cand_tvt[0] - (float(known_tvt[-1]) + ktvt_rate * md_gap))),
                abs(float(ctvt_rate - ktvt_rate)),
                abs(float(ctvt_rate - ktvt_rate)),
                abs(float(cand_f[0] - (float(known_f[-1]) + kf_rate * md_gap))),
                abs(float(cf_rate - kf_rate)),
                abs(float(cf_rate - kf_rate)),
            ]
        )
    return np.asarray(vals, dtype=np.float64)


def _formation_codes(tw: pd.DataFrame, tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tw.empty or "TVT" not in tw.columns:
        n = len(tvt)
        return np.full(n, -1.0), np.full(n, -1.0), np.full(n, 999.0)
    seq = tw.dropna(subset=["TVT"]).sort_values("TVT").reset_index(drop=True)
    tv = seq["TVT"].to_numpy(np.float64)
    if len(tv) == 0:
        n = len(tvt)
        return np.full(n, -1.0), np.full(n, -1.0), np.full(n, 999.0)
    labels = seq.get("Geology", pd.Series([""] * len(seq))).astype(str).to_numpy()
    label_map = {name: i for i, name in enumerate(pd.unique(labels))}
    form = np.asarray([label_map.get(x, -1) for x in labels], dtype=np.float64)
    core_order = {name: i for i, name in enumerate(FORMATION_COLS)}
    core = np.asarray([core_order.get(x, -1) for x in labels], dtype=np.float64)
    x = np.asarray(tvt, dtype=np.float64)
    idx = np.searchsorted(tv, x, side="left")
    idx = np.clip(idx, 0, len(tv) - 1)
    boundaries = tv[np.r_[True, labels[1:] != labels[:-1]]]
    if len(boundaries):
        pos = np.searchsorted(boundaries, x, side="left")
        left = np.where(pos > 0, boundaries[np.clip(pos - 1, 0, len(boundaries) - 1)], np.nan)
        right = np.where(pos < len(boundaries), boundaries[np.clip(pos, 0, len(boundaries) - 1)], np.nan)
        dist = np.nanmin(np.vstack([np.abs(x - left), np.abs(x - right)]), axis=0)
    else:
        dist = np.full(len(x), 999.0)
    return form[idx], core[idx], dist


def _add_prefix_cost_columns(records: pd.DataFrame, paths: dict[int, np.ndarray], hw: pd.DataFrame, tw: pd.DataFrame, row_idx: np.ndarray) -> pd.DataFrame:
    if records.empty:
        return records
    rows = []
    prefix_names = []
    for win in VP_PREFIX_WINDOWS:
        prefix_names.extend(
            [
                f"prefix_w{win}_tvt_start_gap",
                f"prefix_w{win}_tvt_rate_gap",
                f"prefix_w{win}_tvt_kink_abs",
                f"prefix_w{win}_f_start_gap",
                f"prefix_w{win}_f_rate_gap",
                f"prefix_w{win}_f_kink_abs",
            ]
        )
    start_row = int(row_idx[0]) if len(row_idx) else 0
    known_tvt = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_tvt = known_tvt[np.isfinite(known_tvt)]
    last_known = float(known_tvt[-1]) if len(known_tvt) else 0.0
    last_form, last_core, _ = _formation_codes(tw, np.asarray([last_known], dtype=np.float64))
    for cid in records["candidate_idx"].astype(int):
        tvt = np.asarray(paths[int(cid)], dtype=np.float64)
        vals = dict(zip(prefix_names, _prefix_vector(hw, row_idx, tvt)))
        form, core, boundary = _formation_codes(tw, tvt)
        head = slice(0, min(64, len(tvt)))
        core_d = np.diff(core[head]) if len(core[head]) > 1 else np.asarray([], dtype=np.float64)
        vals["prefix_form_code_jump"] = abs(float(form[0] - last_form[0])) if len(form) else 0.0
        vals["prefix_core_code_jump"] = abs(float(core[0] - last_core[0])) if len(core) else 0.0
        vals["prefix_core_step_down"] = float(core[0] < last_core[0]) if len(core) and last_core[0] >= 0 else 0.0
        vals["prefix_known_to_cand_relpos_gap"] = abs(float(tvt[0] - last_known)) / 50.0 if len(tvt) else 0.0
        vals["prefix_cand_head_form_change_frac"] = float(np.mean(np.diff(form[head]) != 0.0)) if len(form[head]) > 1 else 0.0
        vals["prefix_cand_head_core_step_down_frac"] = float(np.mean(core_d < 0.0)) if len(core_d) else 0.0
        vals["prefix_cand_head_boundary_min"] = float(np.nanmin(boundary[head])) if len(boundary[head]) else 999.0
        rows.append(vals)
    mat = pd.DataFrame(rows)
    out = records.reset_index(drop=True).copy()
    for col in mat.columns:
        out[col] = pd.to_numeric(mat[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
        ranks = _percentile_cost(np.abs(out[[col]].to_numpy(np.float64)))
        out[f"rel_{col}_pct_abs_asc"] = ranks
        values = out[col].to_numpy(np.float64)
        med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        mad = 1.4826 * float(np.nanmedian(np.abs(values - med))) if np.isfinite(values).any() else 1.0
        if not np.isfinite(mad) or mad < 1e-9:
            mad = float(np.nanstd(values)) if np.isfinite(values).any() else 1.0
        if not np.isfinite(mad) or mad < 1e-9:
            mad = 1.0
        out[f"rel_{col}_z"] = np.nan_to_num((values - med) / mad, nan=0.0)
    prefix_cols = [c for c in out.columns if str(c).startswith("prefix_") and pd.api.types.is_numeric_dtype(out[c])]
    refs = {
        "mapdiff": out.sort_values(["score", "candidate_idx"], ascending=[True, True]).head(1),
        "anchordiff": out[out["candidate_name"].astype(str).eq("anchor_exact")].head(1),
    }
    for ref_name, ref in refs.items():
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        for col in prefix_cols:
            out[f"{ref_name}_{col}"] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(np.float64)
                - float(pd.to_numeric(pd.Series([ref_row.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            )
    return out


def _add_surface_event_cost_columns(
    records: pd.DataFrame,
    paths: dict[int, np.ndarray],
    hw: pd.DataFrame,
    surface_tops: pd.DataFrame | None,
    row_idx: np.ndarray,
) -> pd.DataFrame:
    if records.empty or surface_tops is None or surface_tops.empty:
        return records
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if len(row_idx) == 0:
        return records
    start_row = int(row_idx[0])
    tvt_input = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_z = hw["Z"].to_numpy(np.float64)[:start_row]
    hidden_z = hw["Z"].to_numpy(np.float64)[row_idx]
    known_mask = np.isfinite(tvt_input) & np.isfinite(known_z)
    out = records.reset_index(drop=True).copy()
    raw_vals: list[float] = []
    min_vals: list[float] = []
    std_vals: list[float] = []
    counts: list[float] = []
    for cid in out["candidate_idx"].astype(int):
        tvt = np.asarray(paths[int(cid)], dtype=np.float64)
        costs: list[float] = []
        med_abs: list[float] = []
        std_abs: list[float] = []
        for col in FORMATION_COLS:
            if col not in surface_tops.columns:
                continue
            top_all = pd.to_numeric(surface_tops[col], errors="coerce").to_numpy(np.float64)
            if len(top_all) < len(hw):
                top_all = np.pad(top_all, (0, len(hw) - len(top_all)), constant_values=np.nan)
            top_known = top_all[:start_row][known_mask]
            top_hidden = top_all[row_idx]
            finite_known = np.isfinite(top_known)
            finite_hidden = np.isfinite(top_hidden) & np.isfinite(tvt) & np.isfinite(hidden_z)
            if int(finite_known.sum()) < 8 or int(finite_hidden.sum()) < max(4, len(tvt) // 5):
                continue
            contact_const = float(np.nanmedian(tvt_input[known_mask][finite_known] + known_z[known_mask][finite_known] - top_known[finite_known]))
            resid = tvt[finite_hidden] + hidden_z[finite_hidden] - top_hidden[finite_hidden] - contact_const
            if len(resid) == 0:
                continue
            med = float(np.nanmedian(np.abs(resid)))
            std = float(np.nanstd(resid))
            drift = abs(float(np.nanmean(resid[: max(1, len(resid) // 3)])) - float(np.nanmean(resid[-max(1, len(resid) // 3) :])))
            costs.append(0.55 * min(med / 18.0, 4.0) + 0.30 * min(std / 28.0, 4.0) + 0.15 * min(drift / 25.0, 4.0))
            med_abs.append(med)
            std_abs.append(std)
        if costs:
            raw_vals.append(float(np.min(costs)))
            min_vals.append(float(np.min(med_abs)))
            std_vals.append(float(np.min(std_abs)))
            counts.append(float(len(costs)))
        else:
            raw_vals.append(0.0)
            min_vals.append(0.0)
            std_vals.append(0.0)
            counts.append(0.0)
    out["pseudo_event_raw"] = np.asarray(raw_vals, dtype=np.float64)
    out["pseudo_event_min_abs"] = np.asarray(min_vals, dtype=np.float64)
    out["pseudo_event_std_min"] = np.asarray(std_vals, dtype=np.float64)
    out["pseudo_event_surface_count"] = np.asarray(counts, dtype=np.float64)
    return out


def _select_diverse_candidates(records: pd.DataFrame, paths: dict[int, np.ndarray]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if records.empty or len(records) <= VP_DIVERSE_TOPK:
        return records.copy(), paths
    rows = records.sort_values(["score_rank", "candidate_idx"], ascending=[True, True]).copy()
    selected: list[int] = []

    def add_indices(indices) -> None:
        for idx in indices:
            ii = int(idx)
            if ii not in selected:
                selected.append(ii)
            if len(selected) >= int(VP_DIVERSE_TOPK):
                return

    add_indices(rows.head(max(0, int(VP_DIVERSE_SCORE_HEAD))).index)
    if len(selected) < int(VP_DIVERSE_TOPK) and "family" in rows.columns:
        for _family, group in rows.groupby("family", sort=False):
            add_indices(group.head(max(0, int(VP_DIVERSE_FAMILY_QUOTA))).index)
            if len(selected) >= int(VP_DIVERSE_TOPK):
                break
    if len(selected) < int(VP_DIVERSE_TOPK) and "mean_level_delta" in rows.columns:
        width = max(float(VP_DIVERSE_DELTA_BIN), 1e-6)
        work = rows.copy()
        delta = pd.to_numeric(work["mean_level_delta"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        work["_delta_bin"] = np.floor(delta / width).astype(int)
        bin_heads = work.sort_values(["score_rank", "candidate_idx"]).groupby("_delta_bin", sort=False).head(1)
        add_indices(bin_heads.sort_values(["score_rank", "candidate_idx"]).index)
        bin_seconds = work.drop(index=bin_heads.index, errors="ignore").sort_values(["score_rank", "candidate_idx"]).groupby("_delta_bin", sort=False).head(1)
        add_indices(bin_seconds.sort_values(["score_rank", "candidate_idx"]).index)
    if len(selected) < int(VP_DIVERSE_TOPK):
        add_indices(rows.index)
    out = rows.loc[selected[: int(VP_DIVERSE_TOPK)]].copy()
    out["_vp_diverse_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out = out.sort_values(["_vp_diverse_rank"]).reset_index(drop=True)
    keep = set(out["candidate_idx"].astype(int).tolist())
    return out, {int(k): v for k, v in paths.items() if int(k) in keep}


def shape_cluster_key(
    mean_level_delta: float,
    end_delta: float,
    path_d1_std: float,
    path_smoothness: float,
    *,
    mean_bin: float = 2.0,
    end_bin: float = 2.0,
    d1_bin: float = 1.0,
    smooth_bin: float = 1.0,
) -> tuple[int, int, int, int]:
    """Near-duplicate shape key used by proxy diversity (does not change deploy default)."""
    return (
        int(np.round(float(mean_level_delta) / max(float(mean_bin), 1e-6))),
        int(np.round(float(end_delta) / max(float(end_bin), 1e-6))),
        int(np.round(float(path_d1_std) / max(float(d1_bin), 1e-6))),
        int(np.round(float(path_smoothness) / max(float(smooth_bin), 1e-6))),
    )


def select_diverse_candidates_proxy_shape(
    records: pd.DataFrame,
    *,
    proxy_col: str = "proxy_score",
    gr_rank_col: str = "score_rank",
    topk: int = 96,
    drop_formation_contact: bool = True,
    fc_side_channel: int = 0,
    mean_bin: float = 2.0,
    end_bin: float = 2.0,
    d1_bin: float = 1.0,
    smooth_bin: float = 1.0,
    gr_head: int = 12,
    always_keep_anchor: bool = True,
) -> pd.DataFrame:
    """Proxy-first shape-cluster diversity keep set (OOF / CV experiments only).

    Steps:
      1) optional drop / side-channel of formation_contact
      2) seed with anchor + GR score head (deploy-safe, no TVT)
      3) near-dup merge by shape key, keep best proxy per cluster
      4) fill remaining slots by proxy (GR rank tie-break)
    Higher proxy_score is better.
    """
    if records.empty:
        return records.copy()
    work = records.copy()
    fc = pd.DataFrame(columns=work.columns)
    if "family" in work.columns and (drop_formation_contact or fc_side_channel > 0):
        is_fc = work["family"].astype(str).eq("formation_contact")
        if drop_formation_contact or fc_side_channel > 0:
            fc = work.loc[is_fc].copy()
            work = work.loc[~is_fc].copy()
    if work.empty:
        return records.head(0).copy()

    proxy = pd.to_numeric(work.get(proxy_col, np.nan), errors="coerce")
    if not np.isfinite(proxy.to_numpy(np.float64)).any():
        proxy = -pd.to_numeric(work.get(gr_rank_col, np.arange(1, len(work) + 1)), errors="coerce").fillna(1e9)
    work = work.copy()
    work["_proxy"] = proxy.fillna(-1e18).to_numpy(np.float64)
    work["_gr_rank"] = pd.to_numeric(work.get(gr_rank_col, np.arange(1, len(work) + 1)), errors="coerce").fillna(1e9)

    seeded_idx: list = []
    if always_keep_anchor and "family" in work.columns:
        seeded_idx.extend(work.index[work["family"].astype(str).eq("anchor")].tolist())
    if gr_head > 0:
        seeded_idx.extend(work.sort_values(["_gr_rank", "candidate_idx"]).head(int(gr_head)).index.tolist())
    # unique preserve order
    seen = set()
    seed_rows = []
    for ix in seeded_idx:
        if ix in seen:
            continue
        seen.add(ix)
        seed_rows.append(work.loc[ix])
    seed = pd.DataFrame(seed_rows) if seed_rows else work.head(0).copy()

    rest = work.drop(index=list(seen), errors="ignore")
    if len(rest):
        keys = [
            shape_cluster_key(
                float(r.mean_level_delta) if pd.notna(getattr(r, "mean_level_delta", np.nan)) else 0.0,
                float(r.end_delta) if pd.notna(getattr(r, "end_delta", np.nan)) else 0.0,
                float(r.path_d1_std) if pd.notna(getattr(r, "path_d1_std", np.nan)) else 0.0,
                float(r.path_smoothness) if pd.notna(getattr(r, "path_smoothness", np.nan)) else 0.0,
                mean_bin=mean_bin,
                end_bin=end_bin,
                d1_bin=d1_bin,
                smooth_bin=smooth_bin,
            )
            for r in rest.itertuples(index=False)
        ]
        rest = rest.copy()
        rest["_shape_key"] = keys
        rest = rest.sort_values(["_proxy", "_gr_rank"], ascending=[False, True])
        rest = rest.groupby("_shape_key", sort=False, as_index=False).head(1)
        rest = rest.sort_values(["_proxy", "_gr_rank"], ascending=[False, True])

    keep_n = max(int(topk), 1)
    if len(seed) >= keep_n:
        main = seed.head(keep_n).copy()
    else:
        need = keep_n - len(seed)
        fill = rest.head(need) if len(rest) else rest
        main = pd.concat([seed, fill], ignore_index=True)

    if fc_side_channel > 0 and not fc.empty:
        fc = fc.copy()
        fc["_proxy"] = pd.to_numeric(fc.get(proxy_col, np.nan), errors="coerce").fillna(-1e18)
        fc["_gr_rank"] = pd.to_numeric(fc.get(gr_rank_col, np.arange(1, len(fc) + 1)), errors="coerce").fillna(1e9)
        fc = fc.sort_values(["_proxy", "_gr_rank"], ascending=[False, True]).head(int(fc_side_channel))
        main = pd.concat([main, fc], ignore_index=True)
    main = main.drop(columns=[c for c in ("_proxy", "_gr_rank", "_shape_key") if c in main.columns], errors="ignore")
    main["_vp_diverse_rank"] = np.arange(1, len(main) + 1, dtype=int)
    return main.reset_index(drop=True)


def candidate_records_for_well(
    wid: str,
    grp: pd.DataFrame,
    *,
    data_root: str | Path,
    hw: pd.DataFrame | None = None,
    tw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    data_root = Path(data_root)
    if hw is None:
        hw = pd.read_csv(data_root / "test" / f"{wid}__horizontal_well.csv")
    if tw is None:
        tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv")
    row_idx = grp["_row_idx"].to_numpy(np.int64)
    row_idx = row_idx[(row_idx >= 0) & (row_idx < len(hw))]
    if len(row_idx) == 0:
        raise RuntimeError(f"empty row index for {wid}")
    anchor = grp.set_index("_row_idx").reindex(row_idx)["tvt"].to_numpy(np.float64)
    anchor = pd.Series(anchor).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float64)
    start_row = int(row_idx[0])
    tvt_input = hw["TVT_input"].to_numpy(np.float64)
    known = tvt_input[np.isfinite(tvt_input)]
    last_known = float(known[-1]) if len(known) else float(anchor[0])
    md_hidden = hw["MD"].to_numpy(np.float64)[row_idx]
    z_hidden = hw["Z"].to_numpy(np.float64)[row_idx]
    gr_hidden = _roll(_fill(hw["GR"].to_numpy(np.float64)[row_idx]), VP_GR_WINDOW)
    tw_tvt, tw_gr = _typewell_arrays(tw)
    if len(tw_tvt) < 8:
        lo = float(np.nanmin(anchor)) - 1.0
        hi = float(np.nanmax(anchor)) + 1.0
        tw_tvt = np.array([lo, hi], dtype=np.float64)
        tw_gr = np.full(2, float(np.nanmedian(gr_hidden)) if np.isfinite(gr_hidden).any() else 0.0, dtype=np.float64)
    prefix_tvt = tvt_input[:start_row]
    prefix_gr = _roll(_fill(hw["GR"].to_numpy(np.float64)[:start_row]), VP_GR_WINDOW)
    pmask = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    affine = _fit_heel_affine(np.interp(prefix_tvt[pmask], tw_tvt, tw_gr), prefix_gr[pmask]) if int(pmask.sum()) >= 8 else (1.0, 0.0, float("nan"), 0)

    if start_row > 0:
        md_context = np.concatenate([[float(hw["MD"].iloc[start_row - 1])], md_hidden])
        z_context = np.concatenate([[float(hw["Z"].iloc[start_row - 1])], z_hidden])
        anchor_context = np.concatenate([[last_known], anchor])
    else:
        md_context = md_hidden
        z_context = z_hidden
        anchor_context = anchor
    context = _piecewise_candidates(md_context, z_context, anchor_context, last_known)
    candidates: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for cand in context:
        tvt = np.asarray(cand["tvt"], dtype=np.float64)[1:] if start_row > 0 else np.asarray(cand["tvt"], dtype=np.float64)
        if len(tvt) != len(anchor):
            continue
        family = str(cand["family"])
        if family == "level_shift":
            family = "piecewise_u"
        _append_unique(candidates, seen, name=str(cand["name"]), family=family, tvt=tvt)

    known_mask = np.isfinite(tvt_input[:start_row])
    surface_tops = _surface_tops_for_hw(data_root, hw, wid=str(wid))
    candidates.extend(_level_grid_candidates(anchor, z_hidden, seen))
    candidates.extend(
        _prefix_u_rate_candidates(
            hw["MD"].to_numpy(np.float64)[:start_row][known_mask],
            tvt_input[:start_row][known_mask],
            hw["Z"].to_numpy(np.float64)[:start_row][known_mask],
            md_hidden,
            z_hidden,
            anchor,
            seen,
        )
    )
    candidates.extend(_formation_contact_candidates(hw, tw, row_idx, start_row, anchor, z_hidden, seen, surface_tops=surface_tops))
    candidates.extend(
        _selfgr_candidates(
            tvt_input[:start_row],
            hw["GR"].to_numpy(np.float64)[:start_row],
            gr_hidden,
            anchor,
            last_known=last_known,
            seen=seen,
        )
    )

    rows: list[dict[str, object]] = []
    paths: dict[int, np.ndarray] = {}
    for ci, cand in enumerate(candidates):
        tvt = np.asarray(cand["tvt"], dtype=np.float64)
        if len(tvt) != len(anchor):
            continue
        sc = _score_path(tvt, gr_hidden, tw_tvt, tw_gr, affine)
        delta = tvt - anchor
        first = np.diff(tvt)
        second = np.diff(tvt, n=2)
        paths[ci] = tvt.astype(np.float64)
        rows.append(
            {
                "well_id": str(wid),
                "candidate_idx": int(ci),
                "candidate_name": str(cand["name"]),
                "family": str(cand["family"]),
                "score": float(sc["score"]),
                "gr_rmse": float(sc["gr_rmse"]),
                "gr_corr": float(sc["gr_corr"]),
                "prefix_rmse": float(sc["prefix_rmse"]),
                "prefix_n": int(sc["prefix_n"]),
                "mean_level_delta": float(np.mean(delta)),
                "end_delta": float(delta[-1]) if len(delta) else 0.0,
                "path_d1_std": float(np.std(first)) if len(first) else 0.0,
                "path_smoothness": float(np.sqrt(np.mean(second * second))) if len(second) else 0.0,
            }
        )
    records = pd.DataFrame(rows)
    if records.empty:
        raise RuntimeError("no live V2 VP candidates")
    score_arr = records["score"].to_numpy(np.float64)
    good = np.isfinite(score_arr)
    centered = score_arr[good] - float(np.min(score_arr[good])) if good.any() else np.zeros(0)
    temp = float(np.median(centered[centered > 0])) if np.any(centered > 0) else 1.0
    post = np.zeros(len(records), dtype=np.float64)
    if good.any():
        w = np.exp(-np.clip(centered / max(temp, 1e-9), 0.0, 700.0))
        post[good] = w / max(float(w.sum()), 1e-12)
    else:
        post[:] = 1.0 / float(len(records))
    records["posterior"] = post
    records = records.sort_values(["score"], ascending=[True]).reset_index(drop=True)
    records["score_rank"] = np.arange(1, len(records) + 1)
    records["ms_score"] = records["score"].astype(float)
    records["ms_corr_mean"] = records["gr_corr"].astype(float)
    records = _add_prefix_cost_columns(records, paths, hw, tw, row_idx)
    records = _add_surface_event_cost_columns(records, paths, hw, surface_tops, row_idx)
    records, paths = _select_diverse_candidates(records, paths)
    return records, paths


def candidate_records_with_arrays_for_well(
    wid: str,
    grp: pd.DataFrame,
    *,
    data_root: str | Path,
    hw: pd.DataFrame | None = None,
    tw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records, paths = candidate_records_for_well(str(wid), grp, data_root=data_root, hw=hw, tw=tw)
    row_idx = grp.sort_values("_row_idx")["_row_idx"].to_numpy(np.int64)
    if hw is None:
        hw = pd.read_csv(Path(data_root) / "test" / f"{wid}__horizontal_well.csv")
    if tw is None:
        tw = pd.read_csv(Path(data_root) / "test" / f"{wid}__typewell.csv")
    tvt_input = hw["TVT_input"].to_numpy(np.float64)
    known = tvt_input[np.isfinite(tvt_input)]
    last_known = float(known[-1]) if len(known) else 0.0
    tw_tvt, tw_gr = _typewell_arrays(tw)
    prefix_tvt = tvt_input[: int(row_idx[0])] if len(row_idx) else np.asarray([], dtype=np.float64)
    prefix_gr = _roll(_fill(hw["GR"].to_numpy(np.float64)[: int(row_idx[0])]), VP_GR_WINDOW) if len(row_idx) else np.asarray([], dtype=np.float64)
    pmask = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    affine = _fit_heel_affine(np.interp(prefix_tvt[pmask], tw_tvt, tw_gr), prefix_gr[pmask]) if int(pmask.sum()) >= 8 else (1.0, 0.0, float("nan"), 0)
    obs_gr = _roll(_fill(hw["GR"].to_numpy(np.float64)[row_idx]), VP_GR_WINDOW) if len(row_idx) else np.asarray([], dtype=np.float64)

    starts: list[int] = []
    drift_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    offset = 0
    for cid in records["candidate_idx"].astype(int).to_numpy():
        tvt = np.asarray(paths[int(cid)], dtype=np.float64)
        exp = np.interp(tvt, tw_tvt, tw_gr)
        slope, intercept, _rmse, _n = affine
        cal = float(slope) * exp + float(intercept)
        residual = obs_gr[: len(cal)] - cal
        drift = tvt - float(last_known)
        starts.append(offset)
        drift_parts.append(drift.astype(np.float32))
        cal_parts.append(cal.astype(np.float32))
        residual_parts.append(residual.astype(np.float32))
        offset += int(len(drift))

    records = records.reset_index(drop=True).copy()
    n = int(len(row_idx))
    records["record_start"] = np.asarray(starts, dtype=np.int64)
    records["record_len"] = int(n)
    arrays = {
        "path_drift": np.concatenate(drift_parts).astype(np.float32) if drift_parts else np.zeros(0, dtype=np.float32),
        "cal_gr": np.concatenate(cal_parts).astype(np.float32) if cal_parts else np.zeros(0, dtype=np.float32),
        "gr_residual": np.concatenate(residual_parts).astype(np.float32) if residual_parts else np.zeros(0, dtype=np.float32),
    }
    return records, arrays
