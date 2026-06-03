#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
QQ_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
NCC_QUERY_MAX = 512
NCC_QUERY_MIN = 96
TYPEWELL_MATCH_POINTS = 256
RANDOM_SEED = 20260602
EPS = 1e-9


def well_id_from_path(path: Path) -> str:
    name = path.name
    return name.split("__", 1)[0] if "__" in name else name.split(".", 1)[0]


def first_missing_index(series: pd.Series) -> int | None:
    mask = series.isna().to_numpy()
    if not mask.any():
        return None
    return int(np.flatnonzero(mask)[0])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    x = x[mask]
    y = y[mask]
    sx = x.std()
    sy = y.std()
    if sx <= EPS or sy <= EPS:
        return math.nan
    return float(np.mean(((x - x.mean()) / sx) * ((y - y.mean()) / sy)))


def linfit_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    mask = np.isfinite(x_train) & np.isfinite(y_train)
    if mask.sum() < 2 or np.nanstd(x_train[mask]) <= EPS:
        return np.full_like(x_pred, np.nan, dtype=float)
    slope, intercept = np.polyfit(x_train[mask], y_train[mask], 1)
    return slope * x_pred + intercept


def fill_interpolate(values: np.ndarray) -> np.ndarray | None:
    s = pd.Series(np.asarray(values, dtype=float))
    if s.notna().sum() < 3:
        return None
    return s.interpolate(limit_direction="both").to_numpy(dtype=float)


def zscore(values: np.ndarray) -> np.ndarray | None:
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        return None
    sd = x.std()
    if sd <= EPS:
        return None
    return (x - x.mean()) / sd


def max_window_ncc(query: np.ndarray, target: np.ndarray, step: int = 16) -> float:
    q = zscore(query)
    target = np.asarray(target, dtype=float)
    if q is None or len(target) < len(query):
        return math.nan
    windows = np.lib.stride_tricks.sliding_window_view(target, len(query))[::step]
    if len(windows) == 0:
        return math.nan
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    valid = stds > EPS
    if not valid.any():
        return math.nan
    wz = (windows[valid] - means[valid, None]) / stds[valid, None]
    scores = wz @ q / len(q)
    return float(np.max(scores))


def nan_blocks(mask: np.ndarray) -> list[tuple[int, int, int]]:
    mask = np.asarray(mask, dtype=bool)
    blocks = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i - 1
        blocks.append((start, end, end - start + 1))
    return blocks


def describe_series(series: pd.Series, name: str) -> dict[str, float | str]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"metric": name}
    desc = s.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).to_dict()
    return {"metric": name, **desc}


def describe_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([describe_series(df[c], c) for c in cols])


def md(df: pd.DataFrame, *, floatfmt: str = ".4f", index: bool = False) -> str:
    if df.empty:
        return "_无记录_"
    return df.to_markdown(index=index, floatfmt=floatfmt)


def detect_lateral_start(df: pd.DataFrame) -> tuple[int, float]:
    mdv = df["MD"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    dzdmd = np.abs(np.gradient(z, mdv))
    smooth = pd.Series(dzdmd).rolling(101, center=True, min_periods=25).median().bfill().ffill().to_numpy()
    for threshold in [0.08, 0.12, 0.18, 0.25]:
        idx = np.flatnonzero(smooth <= threshold)
        if len(idx):
            return int(idx[0]), float(threshold)
    return 0, math.nan


def segment_masks(df: pd.DataFrame, ps_idx: int, lateral_start_idx: int) -> dict[str, np.ndarray]:
    n = len(df)
    prefix = np.zeros(n, dtype=bool)
    prefix[:ps_idx] = True
    post = np.zeros(n, dtype=bool)
    post[ps_idx:] = True
    prefix_last = np.zeros(n, dtype=bool)
    prefix_last[max(0, ps_idx - 512) : ps_idx] = True
    post_first = np.zeros(n, dtype=bool)
    post_first[ps_idx : min(n, ps_idx + 512)] = True
    post_last = np.zeros(n, dtype=bool)
    post_last[max(ps_idx, n - 512) : n] = True
    lateral_prefix = np.zeros(n, dtype=bool)
    lateral_prefix[max(0, lateral_start_idx) : ps_idx] = True
    return {
        "full": np.ones(n, dtype=bool),
        "prefix": prefix,
        "post_ps": post,
        "prefix_last512": prefix_last,
        "post_first512": post_first,
        "post_last512": post_last,
        "lateral_prefix": lateral_prefix,
    }


def sample_even(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).round().astype(int)
    return x[idx], y[idx]


def pairwise_distance(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def morans_i(values: np.ndarray, dist: np.ndarray, k: int = 10) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    v = values[mask]
    d = dist[np.ix_(mask, mask)].copy()
    n = len(v)
    if n <= k + 2:
        return math.nan
    z = v - v.mean()
    denom = float(np.dot(z, z))
    if denom <= EPS:
        return math.nan
    np.fill_diagonal(d, np.inf)
    weights = np.zeros_like(d)
    for i in range(n):
        nn = np.argsort(d[i])[:k]
        weights[i, nn] = 1.0 / np.maximum(d[i, nn], 1.0)
    sw = weights.sum()
    if sw <= EPS:
        return math.nan
    return float(n / sw * (weights * (z[:, None] * z[None, :])).sum() / denom)


def semivariogram(values: np.ndarray, dist: np.ndarray, formation: str, bins: np.ndarray) -> list[dict[str, float | str]]:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    v = values[mask]
    d = dist[np.ix_(mask, mask)]
    iu = np.triu_indices(len(v), k=1)
    pair_d = d[iu]
    pair_diff = v[iu[0]] - v[iu[1]]
    rows = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (pair_d >= b0) & (pair_d < b1)
        if not m.any():
            continue
        gamma = 0.5 * np.mean(np.square(pair_diff[m]))
        abs_dip = np.mean(np.abs(pair_diff[m]) / np.maximum(pair_d[m], 1.0))
        rows.append(
            {
                "formation": formation,
                "dist_min": float(b0),
                "dist_max": float(b1),
                "pairs": int(m.sum()),
                "semivariance": float(gamma),
                "mean_abs_dip_ft_per_ft": float(abs_dip),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/ROGII/datasets/rogii-wellbore-geology-prediction"))
    parser.add_argument("--output", type=Path, default=Path("/root/ROGII/eda-2.md"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("/root/ROGII/eda/eda2_artifacts"))
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.data_root / "train"
    train_h_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    train_t_paths = sorted(train_dir.glob("*__typewell.csv"))
    train_t_by_id = {well_id_from_path(p): p for p in train_t_paths}
    well_ids = [well_id_from_path(p) for p in train_h_paths]

    typewells: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for wid in well_ids:
        tw = pd.read_csv(train_t_by_id[wid], usecols=["TVT", "GR"]).dropna()
        typewells[wid] = (tw["TVT"].to_numpy(dtype=float), tw["GR"].to_numpy(dtype=float))

    segment_rows = []
    gap_rows = []
    gap_well_rows = []
    qq_rows = []
    calibration_rows = []
    meta_rows = []
    baseline_rows = []
    z_risk_rows = []
    prefix_arrays: dict[str, np.ndarray] = {}
    query_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ps_xy = []
    formation_at_ps = []

    for path in train_h_paths:
        wid = well_id_from_path(path)
        df = pd.read_csv(path)
        ps_idx = first_missing_index(df["TVT_input"])
        if ps_idx is None or ps_idx <= 0:
            continue
        lateral_start_idx, lateral_threshold = detect_lateral_start(df)
        masks = segment_masks(df, ps_idx, lateral_start_idx)
        gr = df["GR"].to_numpy(dtype=float)
        gr_missing = ~np.isfinite(gr)

        for seg, mask in masks.items():
            rows = int(mask.sum())
            miss = int((gr_missing & mask).sum())
            segment_rows.append(
                {
                    "well_id": wid,
                    "segment": seg,
                    "rows": rows,
                    "gr_missing_rows": miss,
                    "gr_missing_rate": miss / rows if rows else math.nan,
                }
            )

        blocks = nan_blocks(gr_missing)
        for start, end, length in blocks:
            if end < ps_idx:
                seg = "prefix"
            elif start >= ps_idx:
                seg = "post_ps"
            else:
                seg = "crosses_ps"
            gap_rows.append({"well_id": wid, "segment": seg, "start_idx": start, "end_idx": end, "length": length})
        gap_well_rows.append(
            {
                "well_id": wid,
                "nan_blocks": len(blocks),
                "max_nan_gap": max((b[2] for b in blocks), default=0),
                "post_nan_blocks": sum(1 for b in blocks if b[0] >= ps_idx),
                "post_max_nan_gap": max((b[2] for b in blocks if b[0] >= ps_idx), default=0),
            }
        )

        tw_tvt, tw_gr = typewells[wid]
        for seg in ["prefix", "post_ps", "full"]:
            mask = masks[seg] & np.isfinite(df["GR"].to_numpy(dtype=float))
            tvt_col = "TVT_input" if seg == "prefix" else "TVT"
            tvt = df.loc[mask, tvt_col].to_numpy(dtype=float)
            hgr = df.loc[mask, "GR"].to_numpy(dtype=float)
            if len(tvt) < 20:
                continue
            tw_interp = np.interp(tvt, tw_tvt, tw_gr)
            diff = hgr - tw_interp
            qq_rows.append(
                {
                    "well_id": wid,
                    "segment": seg,
                    "rows": len(diff),
                    "bias_mean": float(np.mean(diff)),
                    "bias_median": float(np.median(diff)),
                    "bias_rmse": rmse(hgr, tw_interp),
                    "corr": corr(hgr, tw_interp),
                    **{f"hgr_q{int(q*100):02d}": float(np.quantile(hgr, q)) for q in QQ_QUANTILES},
                    **{f"twgr_q{int(q*100):02d}": float(np.quantile(tw_interp, q)) for q in QQ_QUANTILES},
                    **{f"diff_q{int(q*100):02d}": float(np.quantile(diff, q)) for q in QQ_QUANTILES},
                }
            )

        prefix_mask = masks["prefix"] & np.isfinite(df["GR"].to_numpy(dtype=float)) & df["TVT_input"].notna().to_numpy()
        ktvt = df.loc[prefix_mask, "TVT_input"].to_numpy(dtype=float)
        khgr = df.loc[prefix_mask, "GR"].to_numpy(dtype=float)
        if len(ktvt) >= 20:
            tw_prefix = np.interp(ktvt, tw_tvt, tw_gr)
            fit = np.polyfit(tw_prefix, khgr, 1) if np.std(tw_prefix) > EPS else [math.nan, math.nan]
            calibration_rows.append(
                {
                    "well_id": wid,
                    "prefix_points": len(khgr),
                    "prefix_corr": corr(khgr, tw_prefix),
                    "prefix_bias_mean": float(np.mean(khgr - tw_prefix)),
                    "prefix_bias_median": float(np.median(khgr - tw_prefix)),
                    "prefix_rmse": rmse(khgr, tw_prefix),
                    "calibration_slope_hgr_on_twgr": float(fit[0]),
                    "calibration_intercept": float(fit[1]),
                }
            )
            qtvt, qgr = sample_even(ktvt, khgr, TYPEWELL_MATCH_POINTS)
            query_points[wid] = (qtvt, qgr)

        filled_prefix = fill_interpolate(df.loc[masks["prefix"], "GR"].to_numpy(dtype=float))
        if filled_prefix is not None and len(filled_prefix) >= 2 * NCC_QUERY_MIN:
            prefix_arrays[wid] = filled_prefix

        hidden = df.loc[masks["post_ps"], "TVT"].to_numpy(dtype=float)
        known = df.loc[masks["prefix"], "TVT_input"].to_numpy(dtype=float)
        last_tvt = float(known[-1])
        pred_cf = np.full(len(hidden), last_tvt)
        pred_md_all = linfit_predict(df.loc[masks["prefix"], "MD"].to_numpy(dtype=float), known, df.loc[masks["post_ps"], "MD"].to_numpy(dtype=float))
        last512 = masks["prefix_last512"]
        pred_md_last512 = linfit_predict(df.loc[last512, "MD"].to_numpy(dtype=float), df.loc[last512, "TVT_input"].to_numpy(dtype=float), df.loc[masks["post_ps"], "MD"].to_numpy(dtype=float))
        lat_prefix = masks["lateral_prefix"]
        pred_z_lateral = linfit_predict(df.loc[lat_prefix, "Z"].to_numpy(dtype=float), df.loc[lat_prefix, "TVT_input"].to_numpy(dtype=float), df.loc[masks["post_ps"], "Z"].to_numpy(dtype=float))
        baseline_rows.append(
            {
                "well_id": wid,
                "hidden_rows": len(hidden),
                "hidden_tvt_span": float(np.max(hidden) - np.min(hidden)),
                "rmse_cf": rmse(hidden, pred_cf),
                "rmse_md_all": rmse(hidden, pred_md_all) if np.isfinite(pred_md_all).all() else math.nan,
                "rmse_md_last512": rmse(hidden, pred_md_last512) if np.isfinite(pred_md_last512).all() else math.nan,
                "rmse_z_lateral": rmse(hidden, pred_z_lateral) if np.isfinite(pred_z_lateral).all() else math.nan,
                "oracle_const": rmse(hidden, np.full(len(hidden), float(np.mean(hidden)))),
            }
        )

        prefix_rows = ps_idx
        lateral_prefix_len = max(0, ps_idx - lateral_start_idx)
        hidden_span = float(np.max(hidden) - np.min(hidden))
        prefix_span = float(np.nanmax(known) - np.nanmin(known))
        meta_rows.append(
            {
                "well_id": wid,
                "rows": len(df),
                "ps_idx": ps_idx,
                "lateral_start_idx": lateral_start_idx,
                "lateral_threshold_abs_dz_dmd": lateral_threshold,
                "prefix_len": prefix_rows,
                "lateral_prefix_len": lateral_prefix_len,
                "post_len": len(df) - ps_idx,
                "prefix_tvt_span": prefix_span,
                "post_tvt_span": hidden_span,
                "prefix_md_per_tvt_ft": prefix_rows / prefix_span if prefix_span > EPS else math.inf,
                "post_md_per_tvt_ft": len(hidden) / hidden_span if hidden_span > EPS else math.inf,
                "stretch_post_vs_prefix": (len(hidden) / hidden_span) / (prefix_rows / prefix_span)
                if hidden_span > EPS and prefix_span > EPS
                else math.nan,
            }
        )

        hidden_z = df.loc[masks["post_ps"], "Z"].to_numpy(dtype=float)
        pred_z_full = linfit_predict(df.loc[masks["prefix"], "MD"].to_numpy(dtype=float), df.loc[masks["prefix"], "Z"].to_numpy(dtype=float), df.loc[masks["post_ps"], "MD"].to_numpy(dtype=float))
        pred_z_lat = linfit_predict(df.loc[lat_prefix, "MD"].to_numpy(dtype=float), df.loc[lat_prefix, "Z"].to_numpy(dtype=float), df.loc[masks["post_ps"], "MD"].to_numpy(dtype=float))
        z_risk_rows.append(
            {
                "well_id": wid,
                "z_rmse_full_prefix": rmse(hidden_z, pred_z_full) if np.isfinite(pred_z_full).all() else math.nan,
                "z_rmse_lateral_prefix": rmse(hidden_z, pred_z_lat) if np.isfinite(pred_z_lat).all() else math.nan,
                "z_full_minus_lateral_rmse": (
                    rmse(hidden_z, pred_z_full) - rmse(hidden_z, pred_z_lat)
                    if np.isfinite(pred_z_full).all() and np.isfinite(pred_z_lat).all()
                    else math.nan
                ),
                "z_hidden_range": float(np.max(hidden_z) - np.min(hidden_z)),
            }
        )

        ps_row = df.iloc[max(0, ps_idx - 1)]
        ps_xy.append([float(ps_row["X"]), float(ps_row["Y"])])
        formation_at_ps.append({c: float(ps_row[c]) if pd.notna(ps_row[c]) else math.nan for c in FORMATION_COLS})

    segment_df = pd.DataFrame(segment_rows)
    gap_df = pd.DataFrame(gap_rows)
    gap_well_df = pd.DataFrame(gap_well_rows)
    qq_df = pd.DataFrame(qq_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    meta_df = pd.DataFrame(meta_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    z_risk_df = pd.DataFrame(z_risk_rows)

    self_rows = []
    prefix_wids = sorted(prefix_arrays)
    for wid in prefix_wids:
        arr = prefix_arrays[wid]
        query_len = min(NCC_QUERY_MAX, max(NCC_QUERY_MIN, len(arr) // 3))
        if len(arr) < 2 * query_len:
            continue
        query = arr[-query_len:]
        self_target = arr[: -query_len]
        self_ncc = max_window_ncc(query, self_target)
        controls = []
        control_wids = [w for w in prefix_wids if w != wid and len(prefix_arrays[w]) >= query_len]
        if control_wids:
            chosen = rng.choice(control_wids, size=min(12, len(control_wids)), replace=False)
            for cw in chosen:
                controls.append(max_window_ncc(query, prefix_arrays[cw]))
        self_rows.append(
            {
                "well_id": wid,
                "query_len": query_len,
                "self_prefix_ncc": self_ncc,
                "shuffled_control_max_ncc": float(np.nanmax(controls)) if controls else math.nan,
                "shuffled_control_mean_ncc": float(np.nanmean(controls)) if controls else math.nan,
                "self_minus_control_max": self_ncc - float(np.nanmax(controls)) if controls and np.isfinite(self_ncc) else math.nan,
            }
        )
    self_ncc_df = pd.DataFrame(self_rows)

    corr_matrix = np.full((len(well_ids), len(well_ids)), np.nan, dtype=np.float32)
    rmse_matrix = np.full((len(well_ids), len(well_ids)), np.nan, dtype=np.float32)
    for i, wid in enumerate(well_ids):
        if wid not in query_points:
            continue
        qtvt, qgr = query_points[wid]
        for j, twid in enumerate(well_ids):
            tw_tvt, tw_gr = typewells[twid]
            tw_interp = np.interp(qtvt, tw_tvt, tw_gr)
            corr_matrix[i, j] = corr(qgr, tw_interp)
            rmse_matrix[i, j] = rmse(qgr, tw_interp)

    corr_df = pd.DataFrame(corr_matrix, index=well_ids, columns=well_ids)
    rmse_df = pd.DataFrame(rmse_matrix, index=well_ids, columns=well_ids)
    corr_path = args.artifact_dir / "well_to_typewell_prefix_corr_matrix.csv"
    rmse_path = args.artifact_dir / "well_to_typewell_prefix_rmse_matrix.csv"
    corr_df.to_csv(corr_path)
    rmse_df.to_csv(rmse_path)

    match_rows = []
    for i, wid in enumerate(well_ids):
        scores = corr_matrix[i].astype(float)
        valid = np.isfinite(scores)
        if valid.sum() < 2:
            continue
        order = np.argsort(np.where(valid, scores, -np.inf))[::-1]
        top1, top2 = order[0], order[1]
        assigned_idx = well_ids.index(wid)
        assigned_score = scores[assigned_idx]
        assigned_rank = int(np.where(order == assigned_idx)[0][0] + 1)
        match_rows.append(
            {
                "well_id": wid,
                "top1_typewell": well_ids[top1],
                "top1_corr": scores[top1],
                "top2_typewell": well_ids[top2],
                "top2_corr": scores[top2],
                "top1_top2_gap": scores[top1] - scores[top2],
                "assigned_typewell_corr": assigned_score,
                "assigned_typewell_rank": assigned_rank,
                "assigned_is_top1": assigned_rank == 1,
            }
        )
    match_df = pd.DataFrame(match_rows)

    xy = np.asarray(ps_xy, dtype=float)
    spatial_dist = pairwise_distance(xy)
    spatial_df = pd.DataFrame(spatial_dist, index=well_ids, columns=well_ids)
    spatial_path = args.artifact_dir / "well_to_well_ps_xy_distance_matrix.csv"
    spatial_df.to_csv(spatial_path)

    nn_rows = []
    for i, wid in enumerate(well_ids):
        d = spatial_dist[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        nn_rows.append(
            {
                "well_id": wid,
                "nearest_well": well_ids[order[0]],
                "nearest_xy_distance_ft": float(d[order[0]]),
                "nn5_xy_distance_ft": float(d[order[4]]),
                "nn10_xy_distance_ft": float(d[order[9]]),
            }
        )
    nn_df = pd.DataFrame(nn_rows)

    formation_df = pd.DataFrame(formation_at_ps, index=well_ids)
    dist_vals = spatial_dist[np.triu_indices(len(well_ids), k=1)]
    bins = np.quantile(dist_vals, np.linspace(0, 1, 9))
    bins[0] = 0.0
    bins[-1] = bins[-1] + 1.0
    variogram_rows = []
    moran_rows = []
    for formation in FORMATION_COLS:
        values = formation_df[formation].to_numpy(dtype=float)
        moran_rows.append({"formation": formation, "morans_i_knn10_inv_dist": morans_i(values, spatial_dist, k=10)})
        variogram_rows.extend(semivariogram(values, spatial_dist, formation, bins))
    variogram_df = pd.DataFrame(variogram_rows)
    moran_df = pd.DataFrame(moran_rows)

    # Add selected EDA signals to the baseline dashboard.
    dashboard = (
        baseline_df.merge(meta_df, on="well_id", how="left")
        .merge(segment_df[segment_df["segment"] == "post_ps"][["well_id", "gr_missing_rate"]].rename(columns={"gr_missing_rate": "post_gr_missing_rate"}), on="well_id", how="left")
        .merge(calibration_df[["well_id", "prefix_corr", "prefix_bias_mean", "prefix_rmse"]], on="well_id", how="left")
        .merge(self_ncc_df[["well_id", "self_prefix_ncc", "self_minus_control_max"]], on="well_id", how="left")
        .merge(match_df[["well_id", "top1_top2_gap", "assigned_typewell_rank", "assigned_typewell_corr"]], on="well_id", how="left")
        .merge(z_risk_df[["well_id", "z_rmse_full_prefix", "z_rmse_lateral_prefix", "z_full_minus_lateral_rmse"]], on="well_id", how="left")
    )
    baseline_cols = ["rmse_cf", "rmse_md_all", "rmse_md_last512", "rmse_z_lateral"]
    dashboard["best_det_baseline_rmse"] = dashboard[baseline_cols].min(axis=1)
    dashboard["best_det_baseline_name"] = dashboard[baseline_cols].idxmin(axis=1)
    worst_cf = dashboard.sort_values("rmse_cf", ascending=False).head(30)
    worst_best = dashboard.sort_values("best_det_baseline_rmse", ascending=False).head(30)

    # Artifacts for dashboard-level follow-up.
    dashboard.to_csv(args.artifact_dir / "baseline_per_well_dashboard.csv", index=False)
    match_df.to_csv(args.artifact_dir / "typewell_top1_top2_match_scores.csv", index=False)
    self_ncc_df.to_csv(args.artifact_dir / "self_prefix_ncc_vs_shuffled.csv", index=False)
    segment_df.to_csv(args.artifact_dir / "gr_missing_by_segment.csv", index=False)
    gap_df.to_csv(args.artifact_dir / "gr_nan_gap_blocks.csv", index=False)
    variogram_df.to_csv(args.artifact_dir / "formation_variogram.csv", index=False)
    moran_df.to_csv(args.artifact_dir / "formation_morans_i.csv", index=False)
    nn_df.to_csv(args.artifact_dir / "well_to_well_nearest_neighbors.csv", index=False)

    seg_summary = (
        segment_df.groupby("segment")
        .agg(wells=("well_id", "nunique"), rows=("rows", "sum"), gr_missing_rows=("gr_missing_rows", "sum"))
        .reset_index()
    )
    seg_summary["gr_missing_rate"] = seg_summary["gr_missing_rows"] / seg_summary["rows"]

    gap_summary = describe_table(gap_well_df, ["nan_blocks", "max_nan_gap", "post_nan_blocks", "post_max_nan_gap"])
    longest_gaps = gap_df.sort_values("length", ascending=False).head(30)

    qq_summary = (
        qq_df.groupby("segment")
        .agg(
            wells=("well_id", "nunique"),
            rows=("rows", "sum"),
            bias_mean=("bias_mean", "mean"),
            bias_median=("bias_median", "median"),
            bias_rmse=("bias_rmse", "mean"),
            corr=("corr", "median"),
        )
        .reset_index()
    )
    qq_quant = (
        qq_df.groupby("segment")
        [[f"diff_q{int(q*100):02d}" for q in QQ_QUANTILES]]
        .median()
        .reset_index()
    )

    baseline_ladder = pd.DataFrame(
        [
            {"baseline": c, "row_weighted_rmse": rmse(np.zeros(len(dashboard)), np.zeros(len(dashboard)))} for c in []
        ]
    )
    ladder_rows = []
    for col in baseline_cols + ["oracle_const"]:
        sse = 0.0
        n = 0
        for _, row in baseline_df.iterrows():
            if pd.notna(row[col]):
                sse += float(row[col]) ** 2 * int(row["hidden_rows"])
                n += int(row["hidden_rows"])
        ladder_rows.append({"baseline": col, "row_weighted_rmse": math.sqrt(sse / n), "valid_wells": int(baseline_df[col].notna().sum())})
    baseline_ladder = pd.DataFrame(ladder_rows)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# ROGII EDA-2",
        "",
        f"生成时间：`{generated_at}`",
        f"数据根目录：`{args.data_root}`",
        f"生成脚本：`/root/ROGII/eda/run_eda2.py`",
        f"artifact 目录：`{args.artifact_dir}`",
        "",
        "本文件只包含数据 EDA 与 deterministic masked-tail baseline 统计；未进行模型训练或 Kaggle 外部操作。",
        "",
        "## 1. GR missing rate by segment",
        "",
        "segment 定义：`prefix` 为 `TVT_input` 非缺失段；`post_ps` 为 `TVT_input` 缺失段；`lateral_prefix` 使用 Z 轨迹 rolling median `|dZ/dMD|` 阈值检测 lateral start。",
        "",
        md(seg_summary, floatfmt=".6f"),
        "",
        "## 2. NaN gap length / dropout block",
        "",
        "GR dropout block 定义为 `GR` 连续 NaN run。",
        "",
        md(gap_summary, floatfmt=".4f"),
        "",
        "最长 GR NaN blocks Top 30：",
        "",
        md(longest_gaps[["well_id", "segment", "start_idx", "end_idx", "length"]], floatfmt=".0f"),
        "",
        "## 3. Horizontal GR vs typewell GR 的 Q-Q / bias",
        "",
        "比较方式：把水平井 TVT 坐标插值到该井 typewell GR，计算 `horizontal_GR - interp(typewell_GR, TVT)`。prefix 使用 `TVT_input`，post/full 使用训练集真实 `TVT`。",
        "",
        md(qq_summary, floatfmt=".6f"),
        "",
        "diff 分位数中位表：",
        "",
        md(qq_quant, floatfmt=".4f"),
        "",
        "## 4. self-prefix NCC / shuffled control",
        "",
        "self-prefix NCC：取 prefix 尾部 query 窗口，与同井更早 prefix GR 滑窗做最大 NCC。shuffled control：同一 query 与随机其他井 prefix GR 滑窗做对照。",
        "",
        md(describe_table(self_ncc_df, ["self_prefix_ncc", "shuffled_control_max_ncc", "shuffled_control_mean_ncc", "self_minus_control_max"]), floatfmt=".6f"),
        "",
        "self-control gap 最低的井 Top 20：",
        "",
        md(self_ncc_df.sort_values("self_minus_control_max").head(20), floatfmt=".6f"),
        "",
        "## 5. typewell prefix calibration corr",
        "",
        "prefix calibration：在 prefix 点上比较水平井 GR 与 assigned typewell GR(TVT_input)，并拟合 `horizontal_GR = slope * typewell_GR + intercept`。",
        "",
        md(describe_table(calibration_df, ["prefix_corr", "prefix_bias_mean", "prefix_bias_median", "prefix_rmse", "calibration_slope_hgr_on_twgr", "calibration_intercept"]), floatfmt=".6f"),
        "",
        "prefix corr 最低的井 Top 20：",
        "",
        md(calibration_df.sort_values("prefix_corr").head(20), floatfmt=".6f"),
        "",
        "## 6. typewell top1/top2 match score gap",
        "",
        f"每口水平井 prefix 采样最多 {TYPEWELL_MATCH_POINTS} 个有限 GR 点，对全部训练 typewell 计算 corr。矩阵保存：`{corr_path}` 与 `{rmse_path}`。",
        "",
        md(describe_table(match_df, ["top1_corr", "top2_corr", "top1_top2_gap", "assigned_typewell_corr", "assigned_typewell_rank"]), floatfmt=".6f"),
        "",
        f"assigned typewell 为 top1 的井数：{int(match_df['assigned_is_top1'].sum())} / {len(match_df)}。",
        "",
        "top1/top2 gap 最小 Top 30：",
        "",
        md(match_df.sort_values("top1_top2_gap").head(30), floatfmt=".6f"),
        "",
        "## 7. stretch/compression factor",
        "",
        "`md_per_tvt_ft` 为每 1 ft TVT 对应的 MD 行数；`stretch_post_vs_prefix = post_md_per_tvt_ft / prefix_md_per_tvt_ft`。",
        "",
        md(describe_table(meta_df, ["prefix_tvt_span", "post_tvt_span", "prefix_md_per_tvt_ft", "post_md_per_tvt_ft", "stretch_post_vs_prefix"]), floatfmt=".6f"),
        "",
        "post stretch 最大 Top 20：",
        "",
        md(meta_df.sort_values("post_md_per_tvt_ft", ascending=False).head(20), floatfmt=".6f"),
        "",
        "## 8. lateral_start_idx / lateral_prefix_len",
        "",
        "lateral_start_idx 由 Z 轨迹 rolling median `|dZ/dMD|` 检测。阈值依次尝试 0.08, 0.12, 0.18, 0.25。",
        "",
        md(describe_table(meta_df, ["lateral_start_idx", "ps_idx", "prefix_len", "lateral_prefix_len", "post_len"]), floatfmt=".4f"),
        "",
        "lateral_prefix_len 最短 Top 20：",
        "",
        md(meta_df.sort_values("lateral_prefix_len").head(20), floatfmt=".4f"),
        "",
        "## 9. Z full-vs-lateral extrapolation risk",
        "",
        "用 prefix 的 `Z ~ MD` 线性拟合外推 post-PS Z；比较 full prefix 拟合与 lateral prefix 拟合在 post-PS 的 Z RMSE。",
        "",
        md(describe_table(z_risk_df, ["z_rmse_full_prefix", "z_rmse_lateral_prefix", "z_full_minus_lateral_rmse", "z_hidden_range"]), floatfmt=".6f"),
        "",
        "full prefix 比 lateral prefix 更差的 Top 30：",
        "",
        md(z_risk_df.sort_values("z_full_minus_lateral_rmse", ascending=False).head(30), floatfmt=".6f"),
        "",
        "## 10. well-to-well / well-to-typewell distance matrix",
        "",
        f"well-to-well：PS 点 XY 欧氏距离矩阵，保存于 `{spatial_path}`。",
        "",
        md(describe_table(nn_df, ["nearest_xy_distance_ft", "nn5_xy_distance_ft", "nn10_xy_distance_ft"]), floatfmt=".4f"),
        "",
        "空间最近邻 Top 20：",
        "",
        md(nn_df.sort_values("nearest_xy_distance_ft").head(20), floatfmt=".4f"),
        "",
        f"well-to-typewell：prefix GR/TVT corr 与 RMSE 矩阵，保存于 `{corr_path}` 与 `{rmse_path}`。",
        "",
        "## 11. formation dip variogram / Moran's I",
        "",
        "formation 值取 prediction start 前一行的 6 个 formation top；距离使用 PS 点 XY 欧氏距离。Moran's I 使用 k=10 inverse-distance 权重。",
        "",
        md(moran_df, floatfmt=".6f"),
        "",
        "semivariogram / mean abs dip by distance bin：",
        "",
        md(variogram_df, floatfmt=".8f"),
        "",
        "## 12. baseline ladder 的 per-well worst dashboard",
        "",
        "masked-tail deterministic ladder：`rmse_cf`、`rmse_md_all`、`rmse_md_last512`、`rmse_z_lateral`；`oracle_const` 为隐藏段真均值 oracle。",
        "",
        md(baseline_ladder, floatfmt=".6f"),
        "",
        "CF worst Top 30：",
        "",
        md(worst_cf[["well_id", "rmse_cf", "rmse_md_all", "rmse_md_last512", "rmse_z_lateral", "oracle_const", "hidden_tvt_span", "post_gr_missing_rate", "prefix_corr", "self_prefix_ncc", "top1_top2_gap", "assigned_typewell_rank", "post_md_per_tvt_ft", "lateral_prefix_len", "z_full_minus_lateral_rmse"]], floatfmt=".6f"),
        "",
        "Best deterministic baseline worst Top 30：",
        "",
        md(worst_best[["well_id", "best_det_baseline_name", "best_det_baseline_rmse", "rmse_cf", "rmse_md_all", "rmse_md_last512", "rmse_z_lateral", "oracle_const", "hidden_tvt_span", "post_gr_missing_rate", "prefix_corr", "self_prefix_ncc", "top1_top2_gap", "assigned_typewell_rank", "post_md_per_tvt_ft", "lateral_prefix_len", "z_full_minus_lateral_rmse"]], floatfmt=".6f"),
        "",
        "Dashboard 全量 CSV：`/root/ROGII/eda/eda2_artifacts/baseline_per_well_dashboard.csv`。",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
