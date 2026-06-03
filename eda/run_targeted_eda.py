#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
SMOOTH_WINDOWS = [101, 201, 401, 801]
ACF_THRESHOLDS = [0.8, 0.5, 0.2, 0.1]
MAX_ACF_LAG = 2000
EPS = 1e-9


def well_id_from_path(path: Path) -> str:
    name = path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return name.split(".", 1)[0]


def first_missing_index(series: pd.Series) -> int | None:
    mask = series.isna().to_numpy()
    if not mask.any():
        return None
    return int(np.flatnonzero(mask)[0])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= EPS:
        out = np.full(max_lag + 1, np.nan)
        out[0] = 1.0
        return out
    max_lag = min(max_lag, len(x) - 1)
    nfft = 1 << ((2 * len(x) - 1).bit_length())
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[: max_lag + 1]
    ac = ac / ac[0]
    if max_lag < MAX_ACF_LAG:
        ac = np.pad(ac, (0, MAX_ACF_LAG - max_lag), constant_values=np.nan)
    return ac


def first_lag_below(acf: np.ndarray, threshold: float) -> float:
    if len(acf) <= 1:
        return math.nan
    values = acf[1:]
    valid = np.isfinite(values)
    below = np.flatnonzero(valid & (values <= threshold))
    if len(below) == 0:
        return math.nan
    return float(below[0] + 1)


def selected_describe(series: pd.Series, name: str | None = None) -> pd.DataFrame:
    desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    row = desc.to_frame().T
    if name is not None:
        row.insert(0, "metric", name)
    return row


def describe_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        row = {"metric": col}
        row.update(desc.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, *, index: bool = False, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_无记录_"
    return df.to_markdown(index=index, floatfmt=floatfmt)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame, cols: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def nearest_distances(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    grid = grid[np.isfinite(grid)]
    values = values[np.isfinite(values)]
    if len(values) == 0 or len(grid) == 0:
        return np.array([], dtype=float)
    grid = np.sort(grid)
    pos = np.searchsorted(grid, values)
    left = np.clip(pos - 1, 0, len(grid) - 1)
    right = np.clip(pos, 0, len(grid) - 1)
    return np.minimum(np.abs(values - grid[left]), np.abs(values - grid[right]))


def duplicate_groups_from_map(hash_to_wells: dict[str, list[str]]) -> list[list[str]]:
    return sorted(
        [sorted(wells) for wells in hash_to_wells.values() if len(wells) > 1],
        key=lambda wells: (-len(wells), wells[0]),
    )


def duplicate_summary(groups: list[list[str]]) -> dict[str, object]:
    return {
        "duplicate_groups": len(groups),
        "wells_in_duplicate_groups": int(sum(len(g) for g in groups)),
        "largest_group_size": int(max((len(g) for g in groups), default=0)),
    }


def group_preview(groups: list[list[str]], limit: int = 10) -> pd.DataFrame:
    rows = []
    for i, wells in enumerate(groups[:limit], start=1):
        rows.append(
            {
                "group": i,
                "size": len(wells),
                "wells_preview": ", ".join(wells[:8]) + (" ..." if len(wells) > 8 else ""),
            }
        )
    return pd.DataFrame(rows)


def compute_oracles(train_horizontal_paths: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    total_hidden = 0
    total_sse = Counter()

    for path in train_horizontal_paths:
        well_id = well_id_from_path(path)
        df = pd.read_csv(path, usecols=["MD", "TVT", "TVT_input"])
        hidden_mask = df["TVT_input"].isna()
        known_mask = df["TVT_input"].notna()
        if not hidden_mask.any() or not known_mask.any():
            continue

        hidden = df.loc[hidden_mask, "TVT"].to_numpy(dtype=float)
        last_known_tvt = float(df.loc[known_mask, "TVT_input"].iloc[-1])
        hidden_mean = float(np.mean(hidden))
        hidden_rows = int(len(hidden))
        hidden_min = float(np.min(hidden))
        hidden_max = float(np.max(hidden))
        hidden_span = hidden_max - hidden_min
        ps_idx = first_missing_index(df["TVT_input"])
        ps_md = float(df.loc[ps_idx, "MD"]) if ps_idx is not None else math.nan

        pred_cf = np.full(hidden_rows, last_known_tvt)
        pred_const = np.full(hidden_rows, hidden_mean)
        sse_cf = float(np.sum(np.square(hidden - pred_cf)))
        sse_const = float(np.sum(np.square(hidden - pred_const)))

        row = {
            "well_id": well_id,
            "prefix_rows": int(known_mask.sum()),
            "hidden_rows": hidden_rows,
            "prediction_start_index": ps_idx,
            "prediction_start_md": ps_md,
            "last_known_tvt": last_known_tvt,
            "hidden_mean_tvt": hidden_mean,
            "hidden_tvt_min": hidden_min,
            "hidden_tvt_max": hidden_max,
            "hidden_tvt_span": hidden_span,
            "md_rows_per_tvt_ft_hidden": hidden_rows / hidden_span if hidden_span > EPS else math.inf,
            "level_drift_mean_minus_last": hidden_mean - last_known_tvt,
            "level_component_abs": abs(hidden_mean - last_known_tvt),
            "rmse_cf": math.sqrt(sse_cf / hidden_rows),
            "rmse_const": math.sqrt(sse_const / hidden_rows),
            "rmse_cf_minus_const": math.sqrt(sse_cf / hidden_rows) - math.sqrt(sse_const / hidden_rows),
            "sse_cf": sse_cf,
            "sse_const": sse_const,
        }

        for window in SMOOTH_WINDOWS:
            smooth = (
                pd.Series(hidden)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            sse_smooth = float(np.sum(np.square(hidden - smooth)))
            row[f"rmse_smooth_{window}"] = math.sqrt(sse_smooth / hidden_rows)
            row[f"sse_smooth_{window}"] = sse_smooth

        acf = autocorr_fft(hidden - last_known_tvt, MAX_ACF_LAG)
        for threshold in ACF_THRESHOLDS:
            row[f"acf_first_below_{threshold}"] = first_lag_below(acf, threshold)

        rows.append(row)
        total_hidden += hidden_rows
        for key in ["sse_cf", "sse_const", *[f"sse_smooth_{w}" for w in SMOOTH_WINDOWS]]:
            total_sse[key] += row[key]

    oracle_df = pd.DataFrame(rows)
    weighted = {
        "total_hidden_rows": total_hidden,
        "global_rmse_cf": math.sqrt(total_sse["sse_cf"] / total_hidden),
        "global_rmse_const": math.sqrt(total_sse["sse_const"] / total_hidden),
    }
    for window in SMOOTH_WINDOWS:
        weighted[f"global_rmse_smooth_{window}"] = math.sqrt(total_sse[f"sse_smooth_{window}"] / total_hidden)

    drift_sse = float(np.sum(oracle_df["hidden_rows"] * np.square(oracle_df["level_drift_mean_minus_last"])))
    const_sse = float(oracle_df["sse_const"].sum())
    cf_sse = float(oracle_df["sse_cf"].sum())
    weighted.update(
        {
            "global_level_component_rmse": math.sqrt(drift_sse / total_hidden),
            "global_swing_component_rmse": math.sqrt(const_sse / total_hidden),
            "global_cf_mse": cf_sse / total_hidden,
            "global_level_mse_share": drift_sse / cf_sse,
            "global_swing_mse_share": const_sse / cf_sse,
            "max_abs_mse_decomposition_error": float(
                np.max(
                    np.abs(
                        np.square(oracle_df["rmse_cf"])
                        - np.square(oracle_df["rmse_const"])
                        - np.square(oracle_df["level_component_abs"])
                    )
                )
            ),
        }
    )
    return oracle_df, weighted


def compute_tvt_lookup_and_density(
    train_horizontal_paths: list[Path],
    train_typewell_paths_by_id: dict[str, Path],
    test_horizontal_paths: list[Path],
    test_typewell_paths_by_id: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    exact_counts = Counter()
    nearest_distance_frames = []

    for path in train_horizontal_paths:
        well_id = well_id_from_path(path)
        tw_path = train_typewell_paths_by_id[well_id]
        h = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        tw = pd.read_csv(tw_path, usecols=["TVT"])
        tw_values = tw["TVT"].dropna().to_numpy(dtype=float)
        tw_min = float(np.min(tw_values))
        tw_max = float(np.max(tw_values))
        tw_steps = np.diff(np.sort(tw_values))
        hidden = h.loc[h["TVT_input"].isna(), "TVT"].dropna().to_numpy(dtype=float)
        visible = h.loc[h["TVT_input"].notna(), "TVT_input"].dropna().to_numpy(dtype=float)
        full = h["TVT"].dropna().to_numpy(dtype=float)

        rounded_grid = set(np.round(tw_values, 6))
        rounded_full = np.round(full, 6)
        rounded_hidden = np.round(hidden, 6)
        full_exact = np.fromiter((v in rounded_grid for v in rounded_full), dtype=bool, count=len(rounded_full))
        hidden_exact = np.fromiter((v in rounded_grid for v in rounded_hidden), dtype=bool, count=len(rounded_hidden))
        exact_counts["full_total"] += len(full_exact)
        exact_counts["full_hits"] += int(full_exact.sum())
        exact_counts["hidden_total"] += len(hidden_exact)
        exact_counts["hidden_hits"] += int(hidden_exact.sum())

        d_full = nearest_distances(full, tw_values)
        d_hidden = nearest_distances(hidden, tw_values)
        if len(d_full):
            nearest_distance_frames.append(
                pd.DataFrame({"well_id": well_id, "split_part": "train_full", "nearest_tvt_dist": d_full})
            )
        if len(d_hidden):
            nearest_distance_frames.append(
                pd.DataFrame({"well_id": well_id, "split_part": "train_hidden", "nearest_tvt_dist": d_hidden})
            )

        rows.append(
            {
                "split": "train",
                "well_id": well_id,
                "typewell_tvt_min": tw_min,
                "typewell_tvt_max": tw_max,
                "typewell_tvt_step_median": float(np.median(tw_steps)),
                "typewell_tvt_step_min": float(np.min(tw_steps)),
                "typewell_tvt_step_max": float(np.max(tw_steps)),
                "covers_full_horizontal_tvt": bool(tw_min <= np.min(full) + EPS and tw_max + EPS >= np.max(full)),
                "covers_visible_tvt_input": bool(tw_min <= np.min(visible) + EPS and tw_max + EPS >= np.max(visible)),
                "covers_hidden_tvt": bool(tw_min <= np.min(hidden) + EPS and tw_max + EPS >= np.max(hidden)),
                "hidden_rows": int(len(hidden)),
                "hidden_tvt_span": float(np.max(hidden) - np.min(hidden)),
                "md_rows_per_tvt_ft_hidden": float(len(hidden) / (np.max(hidden) - np.min(hidden)))
                if np.max(hidden) - np.min(hidden) > EPS
                else math.inf,
            }
        )

    for path in test_horizontal_paths:
        well_id = well_id_from_path(path)
        tw_path = test_typewell_paths_by_id[well_id]
        h = pd.read_csv(path, usecols=["TVT_input"])
        tw = pd.read_csv(tw_path, usecols=["TVT"])
        tw_values = tw["TVT"].dropna().to_numpy(dtype=float)
        visible = h["TVT_input"].dropna().to_numpy(dtype=float)
        tw_steps = np.diff(np.sort(tw_values))
        rows.append(
            {
                "split": "test_visible",
                "well_id": well_id,
                "typewell_tvt_min": float(np.min(tw_values)),
                "typewell_tvt_max": float(np.max(tw_values)),
                "typewell_tvt_step_median": float(np.median(tw_steps)),
                "typewell_tvt_step_min": float(np.min(tw_steps)),
                "typewell_tvt_step_max": float(np.max(tw_steps)),
                "covers_full_horizontal_tvt": math.nan,
                "covers_visible_tvt_input": bool(
                    np.min(tw_values) <= np.min(visible) + EPS and np.max(tw_values) + EPS >= np.max(visible)
                ),
                "covers_hidden_tvt": math.nan,
                "hidden_rows": int(h["TVT_input"].isna().sum()),
                "hidden_tvt_span": math.nan,
                "md_rows_per_tvt_ft_hidden": math.nan,
            }
        )

    coverage_df = pd.DataFrame(rows)
    nearest_df = pd.concat(nearest_distance_frames, ignore_index=True)
    exact_df = pd.DataFrame(
        [
            {
                "part": "train_full_tvt",
                "rows": exact_counts["full_total"],
                "exact_grid_hits": exact_counts["full_hits"],
                "exact_hit_rate": exact_counts["full_hits"] / exact_counts["full_total"],
            },
            {
                "part": "train_hidden_tvt",
                "rows": exact_counts["hidden_total"],
                "exact_grid_hits": exact_counts["hidden_hits"],
                "exact_hit_rate": exact_counts["hidden_hits"] / exact_counts["hidden_total"],
            },
        ]
    )
    return coverage_df, nearest_df, exact_df


def compute_formation_and_geology(
    train_horizontal_paths: list[Path], train_typewell_paths: list[Path]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    any_violation_wells = set()
    any_violation_rows = 0
    total_rows_for_any = 0

    for path in train_horizontal_paths:
        well_id = well_id_from_path(path)
        df = pd.read_csv(path, usecols=FORMATION_COLS)
        well_any = np.zeros(len(df), dtype=bool)
        valid_any = np.zeros(len(df), dtype=bool)
        for upper, lower in zip(FORMATION_COLS[:-1], FORMATION_COLS[1:]):
            valid = df[[upper, lower]].notna().all(axis=1)
            sep = df[upper] - df[lower]
            violation = valid & (sep <= 0)
            if violation.any():
                any_violation_wells.add(well_id)
            well_any |= violation.to_numpy()
            valid_any |= valid.to_numpy()
            pair_rows.append(
                {
                    "pair": f"{upper}>{lower}",
                    "well_id": well_id,
                    "valid_rows": int(valid.sum()),
                    "violation_rows": int(violation.sum()),
                    "min_vertical_separation_ft": float(sep[valid].min()) if valid.any() else math.nan,
                    "p01_vertical_separation_ft": float(sep[valid].quantile(0.01)) if valid.any() else math.nan,
                    "median_vertical_separation_ft": float(sep[valid].median()) if valid.any() else math.nan,
                }
            )
        total_rows_for_any += int(valid_any.sum())
        any_violation_rows += int(well_any.sum())

    pair_detail = pd.DataFrame(pair_rows)
    pair_summary = (
        pair_detail.groupby("pair")
        .agg(
            wells=("well_id", "nunique"),
            valid_rows=("valid_rows", "sum"),
            violation_rows=("violation_rows", "sum"),
            wells_with_violation=("violation_rows", lambda s: int((s > 0).sum())),
            min_vertical_separation_ft=("min_vertical_separation_ft", "min"),
            p01_vertical_separation_ft=("p01_vertical_separation_ft", "min"),
            median_vertical_separation_ft=("median_vertical_separation_ft", "median"),
        )
        .reset_index()
    )
    pair_order = {f"{upper}>{lower}": i for i, (upper, lower) in enumerate(zip(FORMATION_COLS[:-1], FORMATION_COLS[1:]))}
    pair_summary["_pair_order"] = pair_summary["pair"].map(pair_order)
    pair_summary = pair_summary.sort_values("_pair_order").drop(columns="_pair_order").reset_index(drop=True)
    pair_summary["violation_pct"] = 100 * pair_summary["violation_rows"] / pair_summary["valid_rows"]

    label_rows = []
    for path in train_typewell_paths:
        well_id = well_id_from_path(path)
        df = pd.read_csv(path, usecols=["TVT", "Geology"])
        df = df.dropna(subset=["Geology"])
        if df.empty:
            continue
        grouped = df.groupby("Geology")
        for label, g in grouped:
            label_rows.append(
                {
                    "Geology": str(label),
                    "well_id": well_id,
                    "rows": int(len(g)),
                    "TVT_min": float(g["TVT"].min()),
                    "TVT_max": float(g["TVT"].max()),
                }
            )
    label_detail = pd.DataFrame(label_rows)
    label_summary = (
        label_detail.groupby("Geology")
        .agg(rows=("rows", "sum"), wells=("well_id", "nunique"), TVT_min=("TVT_min", "min"), TVT_max=("TVT_max", "max"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    label_summary["in_formation_top_cols"] = label_summary["Geology"].isin(FORMATION_COLS)
    lexicon_summary = pd.DataFrame(
        [
            {
                "category": "exact_formation_top_labels",
                "labels": int(label_summary["in_formation_top_cols"].sum()),
                "rows": int(label_summary.loc[label_summary["in_formation_top_cols"], "rows"].sum()),
                "wells": int(
                    label_detail.loc[label_detail["Geology"].isin(FORMATION_COLS), "well_id"].nunique()
                ),
            },
            {
                "category": "extra_geology_labels",
                "labels": int((~label_summary["in_formation_top_cols"]).sum()),
                "rows": int(label_summary.loc[~label_summary["in_formation_top_cols"], "rows"].sum()),
                "wells": int(
                    label_detail.loc[~label_detail["Geology"].isin(FORMATION_COLS), "well_id"].nunique()
                ),
            },
        ]
    )
    any_cross_summary = pd.DataFrame(
        [
            {
                "valid_rows_any_pair": total_rows_for_any,
                "rows_with_any_crossing": any_violation_rows,
                "wells_with_any_crossing": len(any_violation_wells),
                "crossing_row_pct": 100 * any_violation_rows / total_rows_for_any if total_rows_for_any else 0.0,
            }
        ]
    )
    return pair_summary, any_cross_summary, lexicon_summary, label_summary


def compute_duplicates(
    train_horizontal_paths: list[Path],
    train_typewell_paths: list[Path],
    test_horizontal_paths: list[Path],
    test_typewell_paths: list[Path],
    sample_submission_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[list[str]]], pd.DataFrame]:
    train_h_by_id = {well_id_from_path(p): p for p in train_horizontal_paths}
    train_t_by_id = {well_id_from_path(p): p for p in train_typewell_paths}
    test_t_by_id = {well_id_from_path(p): p for p in test_typewell_paths}
    sample_sub = pd.read_csv(sample_submission_path)
    sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]
    sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)

    subset_rows = []
    for path in test_horizontal_paths:
        well_id = well_id_from_path(path)
        in_train = well_id in train_h_by_id
        test_h = pd.read_csv(path)
        test_t = pd.read_csv(test_t_by_id[well_id])
        if in_train:
            train_h = pd.read_csv(train_h_by_id[well_id])
            common_h_cols = list(test_h.columns)
            horizontal_common_exact = train_h[common_h_cols].equals(test_h[common_h_cols])
            train_has_tvt = "TVT" in train_h.columns and train_h["TVT"].notna().all()
            sub_rows = sample_sub.loc[sample_sub["well_id"] == well_id, "row_index"].to_numpy(dtype=int)
            train_tvt_available_for_submission_rows = bool(
                train_has_tvt and len(sub_rows) == int(test_h["TVT_input"].isna().sum()) and train_h.iloc[sub_rows]["TVT"].notna().all()
            )
            train_t = pd.read_csv(train_t_by_id[well_id])
            common_t_cols = list(test_t.columns)
            typewell_common_exact = train_t[common_t_cols].equals(test_t[common_t_cols])
        else:
            horizontal_common_exact = False
            typewell_common_exact = False
            train_has_tvt = False
            train_tvt_available_for_submission_rows = False
        subset_rows.append(
            {
                "test_well_id": well_id,
                "test_well_id_in_train": in_train,
                "train_has_complete_TVT": train_has_tvt,
                "horizontal_common_columns_exact": horizontal_common_exact,
                "typewell_common_columns_exact": typewell_common_exact,
                "submission_rows": int((sample_sub["well_id"] == well_id).sum()),
                "test_missing_TVT_input_rows": int(test_h["TVT_input"].isna().sum()),
                "train_TVT_available_for_submission_rows": train_tvt_available_for_submission_rows,
            }
        )

    hash_maps: dict[str, defaultdict[str, list[str]]] = {
        "train_horizontal_full_file": defaultdict(list),
        "train_horizontal_trajectory_exact": defaultdict(list),
        "train_horizontal_trajectory_coarse": defaultdict(list),
        "train_typewell_full_file": defaultdict(list),
        "train_typewell_tvt_gr_exact": defaultdict(list),
    }

    for path in train_horizontal_paths:
        well_id = well_id_from_path(path)
        hash_maps["train_horizontal_full_file"][sha256_file(path)].append(well_id)
        df = pd.read_csv(path, usecols=["MD", "X", "Y", "Z"])
        hash_maps["train_horizontal_trajectory_exact"][sha256_dataframe(df, ["MD", "X", "Y", "Z"])].append(well_id)
        coarse = pd.DataFrame(
            [
                {
                    "rows": len(df),
                    "MD_min": round(float(df["MD"].min()), 0),
                    "MD_max": round(float(df["MD"].max()), 0),
                    "X_start": round(float(df["X"].iloc[0]), 0),
                    "Y_start": round(float(df["Y"].iloc[0]), 0),
                    "Z_start": round(float(df["Z"].iloc[0]), 0),
                    "X_end": round(float(df["X"].iloc[-1]), 0),
                    "Y_end": round(float(df["Y"].iloc[-1]), 0),
                    "Z_end": round(float(df["Z"].iloc[-1]), 0),
                }
            ]
        )
        hash_maps["train_horizontal_trajectory_coarse"][sha256_dataframe(coarse, list(coarse.columns))].append(well_id)

    for path in train_typewell_paths:
        well_id = well_id_from_path(path)
        hash_maps["train_typewell_full_file"][sha256_file(path)].append(well_id)
        df = pd.read_csv(path, usecols=["TVT", "GR"])
        hash_maps["train_typewell_tvt_gr_exact"][sha256_dataframe(df, ["TVT", "GR"])].append(well_id)

    groups_by_kind = {name: duplicate_groups_from_map(mapping) for name, mapping in hash_maps.items()}
    dup_summary = pd.DataFrame(
        [{"kind": kind, **duplicate_summary(groups)} for kind, groups in groups_by_kind.items()]
    )
    return pd.DataFrame(subset_rows), groups_by_kind, dup_summary


def compute_coordinate_sanity(paths_by_split: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    global_rows = []
    for split, paths in paths_by_split.items():
        total_rows = 0
        z_nonnegative = 0
        x_nonpositive = 0
        y_nonpositive = 0
        for path in paths:
            well_id = well_id_from_path(path)
            df = pd.read_csv(path, usecols=["MD", "X", "Y", "Z"])
            total_rows += len(df)
            z_nonnegative += int((df["Z"] >= 0).sum())
            x_nonpositive += int((df["X"] <= 0).sum())
            y_nonpositive += int((df["Y"] <= 0).sum())
            rows.append(
                {
                    "split": split,
                    "well_id": well_id,
                    "rows": len(df),
                    "MD_min": float(df["MD"].min()),
                    "MD_max": float(df["MD"].max()),
                    "MD_range": float(df["MD"].max() - df["MD"].min()),
                    "X_min": float(df["X"].min()),
                    "X_max": float(df["X"].max()),
                    "X_mean": float(df["X"].mean()),
                    "X_range": float(df["X"].max() - df["X"].min()),
                    "Y_min": float(df["Y"].min()),
                    "Y_max": float(df["Y"].max()),
                    "Y_mean": float(df["Y"].mean()),
                    "Y_range": float(df["Y"].max() - df["Y"].min()),
                    "Z_min": float(df["Z"].min()),
                    "Z_max": float(df["Z"].max()),
                    "Z_mean": float(df["Z"].mean()),
                    "Z_range": float(df["Z"].max() - df["Z"].min()),
                    "Z_nonnegative_rows": int((df["Z"] >= 0).sum()),
                }
            )
        global_rows.append(
            {
                "split": split,
                "total_rows": total_rows,
                "Z_nonnegative_rows": z_nonnegative,
                "X_nonpositive_rows": x_nonpositive,
                "Y_nonpositive_rows": y_nonpositive,
            }
        )

    coord_df = pd.DataFrame(rows)
    train_coord = coord_df[coord_df["split"] == "train"].copy()
    robust_cols = ["rows", "MD_min", "MD_max", "X_mean", "Y_mean", "Z_mean", "X_range", "Y_range", "Z_range"]
    for col in robust_cols:
        med = train_coord[col].median()
        mad = np.median(np.abs(train_coord[col] - med))
        if mad <= EPS:
            coord_df[f"robust_z_{col}"] = 0.0
        else:
            coord_df[f"robust_z_{col}"] = 0.6745 * (coord_df[col] - med) / mad
    rz_cols = [f"robust_z_{col}" for col in robust_cols]
    coord_df["max_abs_robust_z"] = coord_df[rz_cols].abs().max(axis=1)
    outliers = coord_df.sort_values("max_abs_robust_z", ascending=False).head(20)

    range_rows = []
    for split, g in coord_df.groupby("split"):
        range_rows.append(
            {
                "split": split,
                "wells": g["well_id"].nunique(),
                "X_min": g["X_min"].min(),
                "X_max": g["X_max"].max(),
                "Y_min": g["Y_min"].min(),
                "Y_max": g["Y_max"].max(),
                "Z_min": g["Z_min"].min(),
                "Z_max": g["Z_max"].max(),
                "MD_min": g["MD_min"].min(),
                "MD_max": g["MD_max"].max(),
            }
        )
    return pd.DataFrame(global_rows), pd.DataFrame(range_rows), outliers


def build_markdown(
    data_root: Path,
    oracle_df: pd.DataFrame,
    oracle_weighted: dict[str, object],
    coverage_df: pd.DataFrame,
    nearest_df: pd.DataFrame,
    exact_df: pd.DataFrame,
    formation_pair_summary: pd.DataFrame,
    any_cross_summary: pd.DataFrame,
    lexicon_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    test_subset_df: pd.DataFrame,
    duplicate_summary_df: pd.DataFrame,
    duplicate_groups: dict[str, list[list[str]]],
    coord_global: pd.DataFrame,
    coord_ranges: pd.DataFrame,
    coord_outliers: pd.DataFrame,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    oracle_global = pd.DataFrame(
        [
            {"oracle": "Oracle-CF", "definition": "常数=最后已知 TVT_input", "row_weighted_rmse": oracle_weighted["global_rmse_cf"]},
            {"oracle": "Oracle-Const", "definition": "常数=隐藏段真值均值", "row_weighted_rmse": oracle_weighted["global_rmse_const"]},
            *[
                {
                    "oracle": f"Oracle-Smooth-{w}",
                    "definition": f"隐藏段真值 centered rolling mean, window={w} ft",
                    "row_weighted_rmse": oracle_weighted[f"global_rmse_smooth_{w}"],
                }
                for w in SMOOTH_WINDOWS
            ],
        ]
    )
    oracle_macro = describe_table(
        oracle_df,
        ["rmse_cf", "rmse_const", "rmse_smooth_101", "rmse_smooth_201", "rmse_smooth_401", "rmse_smooth_801"],
    )
    decomp_global = pd.DataFrame(
        [
            {
                "component": "level_drift_abs(hidden_mean-last_known)",
                "row_weighted_rmse_component": oracle_weighted["global_level_component_rmse"],
                "mse_share_in_CF": oracle_weighted["global_level_mse_share"],
            },
            {
                "component": "hidden_swing_around_mean",
                "row_weighted_rmse_component": oracle_weighted["global_swing_component_rmse"],
                "mse_share_in_CF": oracle_weighted["global_swing_mse_share"],
            },
        ]
    )
    decomp_per_well = describe_table(
        oracle_df,
        ["level_component_abs", "rmse_const", "rmse_cf_minus_const", "hidden_tvt_span", "md_rows_per_tvt_ft_hidden"],
    )
    hidden_span_bins = pd.DataFrame(
        [
            {"condition": "hidden_tvt_span < 0.5 ft", "wells": int((oracle_df["hidden_tvt_span"] < 0.5).sum())},
            {"condition": "hidden_tvt_span < 1 ft", "wells": int((oracle_df["hidden_tvt_span"] < 1).sum())},
            {"condition": "hidden_tvt_span < 2 ft", "wells": int((oracle_df["hidden_tvt_span"] < 2).sum())},
            {"condition": "hidden_tvt_span < 5 ft", "wells": int((oracle_df["hidden_tvt_span"] < 5).sum())},
            {"condition": "hidden_tvt_span < 10 ft", "wells": int((oracle_df["hidden_tvt_span"] < 10).sum())},
            {"condition": "hidden_tvt_span < 20 ft", "wells": int((oracle_df["hidden_tvt_span"] < 20).sum())},
            {"condition": "hidden_tvt_span < 50 ft", "wells": int((oracle_df["hidden_tvt_span"] < 50).sum())},
        ]
    )
    acf_cols = [f"acf_first_below_{t}" for t in ACF_THRESHOLDS]
    acf_summary = describe_table(oracle_df, acf_cols)
    acf_missing = pd.DataFrame(
        [
            {
                "threshold": f"acf<={t:g}",
                "not_below_within_2000ft_or_flat": int(oracle_df[f"acf_first_below_{t}"].isna().sum()),
            }
            for t in ACF_THRESHOLDS
        ]
    )

    coverage_summary = pd.DataFrame(
        [
            {
                "split_part": "train_full_horizontal_tvt",
                "wells": int((coverage_df["split"] == "train").sum()),
                "covered_by_typewell_range": int(coverage_df.loc[coverage_df["split"] == "train", "covers_full_horizontal_tvt"].sum()),
            },
            {
                "split_part": "train_visible_tvt_input",
                "wells": int((coverage_df["split"] == "train").sum()),
                "covered_by_typewell_range": int(coverage_df.loc[coverage_df["split"] == "train", "covers_visible_tvt_input"].sum()),
            },
            {
                "split_part": "train_hidden_tvt",
                "wells": int((coverage_df["split"] == "train").sum()),
                "covered_by_typewell_range": int(coverage_df.loc[coverage_df["split"] == "train", "covers_hidden_tvt"].sum()),
            },
            {
                "split_part": "test_visible_tvt_input",
                "wells": int((coverage_df["split"] == "test_visible").sum()),
                "covered_by_typewell_range": int(coverage_df.loc[coverage_df["split"] == "test_visible", "covers_visible_tvt_input"].sum()),
            },
        ]
    )
    typewell_step_summary = describe_table(coverage_df, ["typewell_tvt_step_median", "typewell_tvt_step_min", "typewell_tvt_step_max"])
    quant_floor = float(coverage_df["typewell_tvt_step_median"].median() / math.sqrt(12))
    nearest_summary = (
        nearest_df.groupby("split_part")["nearest_tvt_dist"]
        .describe(percentiles=[0.5, 0.95, 0.99])
        .reset_index()
    )
    density_summary = describe_table(coverage_df[coverage_df["split"] == "train"], ["hidden_tvt_span", "md_rows_per_tvt_ft_hidden"])

    extra_labels = label_summary[~label_summary["in_formation_top_cols"]].head(30).copy()
    main_labels = label_summary[label_summary["in_formation_top_cols"]].copy()
    label_summary_show = label_summary.head(43).copy()

    duplicate_previews = []
    for kind, groups in duplicate_groups.items():
        preview = group_preview(groups, limit=8)
        duplicate_previews.append(f"### {kind}\n\n{md(preview)}")

    coord_outlier_cols = [
        "split",
        "well_id",
        "rows",
        "X_mean",
        "Y_mean",
        "Z_mean",
        "X_range",
        "Y_range",
        "Z_range",
        "max_abs_robust_z",
    ]

    lines = [
        "# ROGII 专项数据 EDA：可达地板 / 可辨识性",
        "",
        f"生成时间：`{generated_at}`",
        "",
        f"数据根目录：`{data_root}`",
        "",
        "生成脚本：`/root/ROGII/eda/run_targeted_eda.py`",
        "",
        "本 EDA 只使用训练集真实 `TVT` 做 masked-tail 仿真与数据核对；未进行任何模型训练。",
        "",
        "## 1. Oracle 地板与误差分解",
        "",
        "masked-tail 定义：每口训练井中 `TVT_input` 为缺失的 post-PS 段作为隐藏段，真实 `TVT` 只用于 oracle 计算。",
        "",
        f"训练井数：{len(oracle_df):,}；隐藏段总行数：{int(oracle_weighted['total_hidden_rows']):,}。",
        "",
        "### 1.1 Row-weighted oracle RMSE",
        "",
        md(oracle_global, floatfmt=".6f"),
        "",
        "### 1.2 Per-well oracle RMSE 分布",
        "",
        md(oracle_macro, floatfmt=".6f"),
        "",
        "### 1.3 CF 误差平方分解",
        "",
        "`RMSE_CF^2 = (hidden_mean - last_known)^2 + RMSE_Const^2`。本次逐井最大绝对分解误差："
        f"`{oracle_weighted['max_abs_mse_decomposition_error']:.12g}`。",
        "",
        md(decomp_global, floatfmt=".6f"),
        "",
        "逐井分布：",
        "",
        md(decomp_per_well, floatfmt=".6f"),
        "",
        "### 1.4 post-PS TVT 总跨度",
        "",
        md(hidden_span_bins, floatfmt=".0f"),
        "",
        "## 2. TVT 残差自相关长度",
        "",
        "残差定义：`hidden_TVT - last_known_TVT_input`。ACF 计算前对残差减去本井残差均值；lag 单位为 MD 行数，当前数据 MD 步长为 1 ft。",
        "",
        md(acf_summary, floatfmt=".6f"),
        "",
        "未在 2000 ft lag 内跌破阈值或序列近似常数的井数：",
        "",
        md(acf_missing, floatfmt=".0f"),
        "",
        "## 3. TVT 查表坐标与投影采样密度",
        "",
        "### 3.1 Typewell TVT 覆盖水平井 TVT",
        "",
        md(coverage_summary, floatfmt=".0f"),
        "",
        "### 3.2 Typewell TVT 步长",
        "",
        md(typewell_step_summary, floatfmt=".6f"),
        "",
        f"以全库 median typewell TVT 步长估算的均匀量化 RMSE：`{quant_floor:.6f}` ft。",
        "",
        "### 3.3 水平井 TVT 到 typewell TVT 网格的等值命中与最近距离",
        "",
        md(exact_df, floatfmt=".8f"),
        "",
        md(nearest_summary, floatfmt=".6f"),
        "",
        "### 3.4 post-PS 投影采样密度",
        "",
        md(density_summary, floatfmt=".6f"),
        "",
        "## 4. 6 个地层顶单调序 / 不交叉检查",
        "",
        "检查顺序：`ANCC > ASTNU > ASTNL > EGFDU > EGFDL > BUDA`。相邻对的 `upper - lower <= 0` 记为交叉/非单调位置。",
        "",
        md(any_cross_summary, floatfmt=".6f"),
        "",
        md(formation_pair_summary, floatfmt=".6f"),
        "",
        "## 5. Typewell Geology 标签词表对账",
        "",
        "6 个 formation top 字段：`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`。",
        "",
        md(lexicon_summary, floatfmt=".0f"),
        "",
        "### 5.1 与 formation top 字段同名的标签",
        "",
        md(main_labels[["Geology", "rows", "wells", "TVT_min", "TVT_max", "in_formation_top_cols"]], floatfmt=".4f"),
        "",
        "### 5.2 非 formation top 同名标签 Top 30",
        "",
        md(extra_labels[["Geology", "rows", "wells", "TVT_min", "TVT_max", "in_formation_top_cols"]], floatfmt=".4f"),
        "",
        "### 5.3 全部 Geology 标签",
        "",
        md(label_summary_show[["Geology", "rows", "wells", "TVT_min", "TVT_max", "in_formation_top_cols"]], floatfmt=".4f"),
        "",
        "## 6. test ⊂ train 与重复井检测",
        "",
        "### 6.1 可见 test well_id 与 train 显式核对",
        "",
        md(test_subset_df, floatfmt=".0f"),
        "",
        "### 6.2 全库重复 hash 汇总",
        "",
        md(duplicate_summary_df, floatfmt=".0f"),
        "",
        "### 6.3 重复组预览",
        "",
        "\n\n".join(duplicate_previews),
        "",
        "## 7. 坐标基准 / 单位 sanity 与离群井",
        "",
        "### 7.1 坐标符号检查",
        "",
        md(coord_global, floatfmt=".0f"),
        "",
        "### 7.2 train/test 坐标范围",
        "",
        md(coord_ranges, floatfmt=".4f"),
        "",
        "### 7.3 robust-z 最大的井 Top 20",
        "",
        md(coord_outliers[coord_outlier_cols], floatfmt=".4f"),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/ROGII/datasets/rogii-wellbore-geology-prediction"),
    )
    parser.add_argument("--output", type=Path, default=Path("/root/ROGII/eda.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    sample_submission_path = data_root / "sample_submission.csv"

    train_horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    train_typewell_paths = sorted(train_dir.glob("*__typewell.csv"))
    test_horizontal_paths = sorted(test_dir.glob("*__horizontal_well.csv"))
    test_typewell_paths = sorted(test_dir.glob("*__typewell.csv"))
    train_typewell_paths_by_id = {well_id_from_path(p): p for p in train_typewell_paths}
    test_typewell_paths_by_id = {well_id_from_path(p): p for p in test_typewell_paths}

    oracle_df, oracle_weighted = compute_oracles(train_horizontal_paths)
    coverage_df, nearest_df, exact_df = compute_tvt_lookup_and_density(
        train_horizontal_paths, train_typewell_paths_by_id, test_horizontal_paths, test_typewell_paths_by_id
    )
    formation_pair_summary, any_cross_summary, lexicon_summary, label_summary = compute_formation_and_geology(
        train_horizontal_paths, train_typewell_paths
    )
    test_subset_df, duplicate_groups, duplicate_summary_df = compute_duplicates(
        train_horizontal_paths,
        train_typewell_paths,
        test_horizontal_paths,
        test_typewell_paths,
        sample_submission_path,
    )
    coord_global, coord_ranges, coord_outliers = compute_coordinate_sanity(
        {"train": train_horizontal_paths, "test": test_horizontal_paths}
    )

    report = build_markdown(
        data_root,
        oracle_df,
        oracle_weighted,
        coverage_df,
        nearest_df,
        exact_df,
        formation_pair_summary,
        any_cross_summary,
        lexicon_summary,
        label_summary,
        test_subset_df,
        duplicate_summary_df,
        duplicate_groups,
        coord_global,
        coord_ranges,
        coord_outliers,
    )
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
