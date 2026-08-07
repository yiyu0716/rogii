#!/usr/bin/env python3
"""Train public train19 + sg_path + member_lgb posterior features.

The member_lgb selector is strong evidence inside a selected candidate mode,
but direct path overwrites were bad.  This experiment exposes the posterior as
GBDT features instead:

* compact row-level paths from the segmented member_lgb posterior;
* well-level posterior confidence, diversity, bimodality, and family mass;
* no target/oracle/RMSE columns.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/rogii")
CV_DIR = ROOT / "cv"
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

from public_train19_lgb5fold_repro import train_lgb_5fold  # noqa: E402
from public_train19_seggate_transfer_experiment import build_or_load_sidecar as build_sg_sidecar  # noqa: E402
from public_train19_trend_reliability_experiment import train_profile  # noqa: E402


ART_ROOT = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709"
DEFAULT_XY = ART_ROOT / "full_train19_cache/Xy.parquet"
DEFAULT_FEATURES = ART_ROOT / "full_train19_cache/features.json"
DEFAULT_DATA_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
DEFAULT_SEG_NPZ = ROOT / "cv/artifacts/candidate_cluster_tabular_v1_full_segmented_grid_20260707/saved_predictions.npz"
DEFAULT_WELL_GATE = ROOT / "cv/artifacts/candidate_cluster_conservative_gate_v1_20260707/well_gate_oof_features.csv"
DEFAULT_GATE_META = ROOT / "cv/artifacts/candidate_cluster_conservative_gate_deploy_reduced_v1/gate_meta.json"
DEFAULT_MEMBER_NPZ = ROOT / "cv/artifacts/candidate_member_lgb_segmented_fast_v1_noformation/saved_predictions.npz"
DEFAULT_MEMBER_SCORES = ROOT / "cv/artifacts/candidate_member_selector_v2/noformation_with_gr_likelihood/candidate_member_scores.csv"
DEFAULT_CANDIDATES = ROOT / "cv/artifacts/candidate_prefix_features_diverse_full_top60_20260706_noleak_noformation/candidate_prefix_features.csv"
DEFAULT_WELL_INDEX = ROOT / "cv/artifacts/whole_trace_piecewise_u_path_cache_v2_diverse_full_top60_20260706/well_index.csv"
OUT_ROOT = ART_ROOT / "sgpath_member_posterior_features_v1"

FAMILIES = ("anchor", "level_grid", "piecewise_u", "prefix_u_rate", "selfgr")
BAD_FEATURE_PARTS = ("target", "truth", "oracle")


def _short_mtw(key: str) -> str:
    match = re.search(r"_mtw([^_]+)_", key)
    if not match:
        return re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_").lower()
    return "mtw" + match.group(1).replace(".", "p")


def _softmax(vals: np.ndarray, tau: float) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    vals = np.nan_to_num(vals, nan=-1e9, neginf=-1e9, posinf=1e9)
    if len(vals) == 0:
        return vals
    t = max(float(tau), 1e-6)
    z = vals / t
    z = z - float(np.max(z))
    exp = np.exp(np.clip(z, -60.0, 60.0))
    s = float(np.sum(exp))
    if not math.isfinite(s) or s <= 0.0:
        return np.full(len(vals), 1.0 / max(len(vals), 1), dtype=np.float64)
    return exp / s


def _entropy_norm(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(weights) <= 1:
        return 0.0
    ent = -float(np.sum(weights * np.log(weights + 1e-12)))
    return float(ent / math.log(len(weights)))


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or len(weights) == 0:
        return 0.0, 0.0, 0.0
    weights = weights / max(float(np.sum(weights)), 1e-12)
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    abs_mean = float(np.sum(weights * np.abs(values)))
    return mean, math.sqrt(max(var, 0.0)), abs_mean


def _candidate_usecols(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    desired = {
        "well_index",
        "candidate_idx",
        "family",
        "candidate_name",
        "score_rank",
        "cache_selection_rank",
        "score",
        "posterior",
        "mean_level_delta",
        "path_ltr_score",
        "ms_score",
        "ms_score_std",
        "ms_corr_mean",
        "ms_anom_score",
        "ms_anom_corr",
        "ms_grad_score",
        "ms_grad_corr",
    }
    return [c for c in header.columns if c in desired]


def _build_member_well_features(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_csv(args.member_scores)
    for bad in ("target_abs_level", "target_rmse"):
        if bad in scores.columns:
            scores = scores.drop(columns=[bad])
    usecols = _candidate_usecols(Path(args.member_candidates))
    cand = pd.read_csv(args.member_candidates, usecols=usecols)
    merge_cols = ["well_index", "candidate_idx"]
    keep_cols = [c for c in cand.columns if c not in {"family"} or c not in scores.columns]
    cand = cand[keep_cols]
    work = scores.merge(cand, on=merge_cols, how="left", suffixes=("", "_cand"))
    if "family_cand" in work.columns and "family" not in work.columns:
        work["family"] = work["family_cand"]
    for col in (
        "member_lgb_oof",
        "prefix_member_score",
        "score_rank",
        "cache_selection_rank",
        "score",
        "posterior",
        "mean_level_delta",
        "path_ltr_score",
        "ms_score",
        "ms_score_std",
        "ms_corr_mean",
        "ms_anom_score",
        "ms_anom_corr",
        "ms_grad_score",
        "ms_grad_corr",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["cluster_id"] = pd.to_numeric(work["cluster_id"], errors="coerce").fillna(-1).astype(int)
    work["candidate_idx"] = pd.to_numeric(work["candidate_idx"], errors="coerce").fillna(-1).astype(int)
    work["well_index"] = pd.to_numeric(work["well_index"], errors="coerce").fillna(-1).astype(int)
    work["family"] = work["family"].astype(str)
    well_index = pd.read_csv(args.well_index, dtype={"well_id": str})
    wi_to_well = dict(zip(well_index["well_index"].astype(int), well_index["well_id"].astype(str)))

    rows: list[dict[str, float | int | str]] = []
    for wi, group in work.groupby("well_index", sort=False):
        g = group.copy()
        if g.empty:
            continue
        g["member_lgb_oof"] = g["member_lgb_oof"].fillna(g["member_lgb_oof"].median()).fillna(0.0)
        g["prefix_member_score"] = g["prefix_member_score"].fillna(g["prefix_member_score"].median()).fillna(0.0)
        g["mean_level_delta"] = g["mean_level_delta"].fillna(0.0)
        g["score_rank"] = g["score_rank"].fillna(99.0)
        ordered = g.sort_values(["member_lgb_oof", "score_rank", "candidate_idx"], ascending=[False, True, True]).reset_index(drop=True)
        top1 = ordered.iloc[0]
        top2 = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
        prefix_ordered = g.sort_values(["prefix_member_score", "score_rank", "candidate_idx"], ascending=[False, True, True]).reset_index(drop=True)
        ptop1 = prefix_ordered.iloc[0]

        scores_arr = ordered["member_lgb_oof"].to_numpy(np.float64)
        deltas = ordered["mean_level_delta"].to_numpy(np.float64)
        weights = _softmax(scores_arr, tau=float(args.member_softmax_tau))
        wmean, wstd, wabs = _weighted_stats(deltas, weights)
        topk = ordered.head(min(int(args.member_topk_summary), len(ordered))).copy()
        topk_scores = topk["member_lgb_oof"].to_numpy(np.float64)
        topk_deltas = topk["mean_level_delta"].to_numpy(np.float64)

        cluster_rows: list[dict[str, float | int]] = []
        for cid, cg in g.groupby("cluster_id", sort=False):
            cs = cg["member_lgb_oof"].fillna(0.0).to_numpy(np.float64)
            cdelta = cg["mean_level_delta"].fillna(0.0).to_numpy(np.float64)
            cw = _softmax(cs, tau=float(args.member_softmax_tau))
            cmean, cstd, cabs = _weighted_stats(cdelta, cw)
            cluster_rows.append(
                {
                    "cluster_id": int(cid),
                    "cluster_max": float(np.nanmax(cs)) if len(cs) else 0.0,
                    "cluster_mean": float(np.nanmean(cs)) if len(cs) else 0.0,
                    "cluster_sumexp": float(np.log(np.sum(np.exp(np.clip(cs / max(float(args.member_softmax_tau), 1e-6), -60, 60))) + 1e-12)),
                    "cluster_delta_mean": cmean,
                    "cluster_delta_std": cstd,
                    "cluster_delta_abs": cabs,
                    "cluster_n": int(len(cg)),
                }
            )
        clusters = pd.DataFrame(cluster_rows).sort_values(["cluster_max", "cluster_sumexp", "cluster_id"], ascending=[False, False, True])
        ctop1 = clusters.iloc[0]
        ctop2 = clusters.iloc[1] if len(clusters) > 1 else clusters.iloc[0]
        cw = _softmax(clusters["cluster_max"].to_numpy(np.float64), tau=float(args.member_cluster_tau))
        cluster_delta_gap = float(abs(float(ctop1["cluster_delta_mean"]) - float(ctop2["cluster_delta_mean"]))) if len(clusters) > 1 else 0.0
        cluster_margin = float(float(ctop1["cluster_max"]) - float(ctop2["cluster_max"])) if len(clusters) > 1 else 99.0
        near_tie_10_25 = float(10.0 <= cluster_delta_gap <= 25.0) * float(math.exp(-abs(cluster_margin) / max(float(args.member_tie_margin_scale), 1e-6)))

        row: dict[str, float | int | str] = {
            "well_index": int(wi),
            "well_id": wi_to_well.get(int(wi), ""),
            "memb_n_candidates": float(len(g)),
            "memb_n_clusters": float(g["cluster_id"].nunique()),
            "memb_score_mean": float(np.nanmean(scores_arr)),
            "memb_score_std": float(np.nanstd(scores_arr)),
            "memb_score_p75": float(np.nanquantile(scores_arr, 0.75)),
            "memb_score_p90": float(np.nanquantile(scores_arr, 0.90)),
            "memb_top1_score": float(top1["member_lgb_oof"]),
            "memb_top2_score": float(top2["member_lgb_oof"]),
            "memb_top_margin": float(top1["member_lgb_oof"] - top2["member_lgb_oof"]),
            "memb_top1_delta": float(top1["mean_level_delta"]),
            "memb_top2_delta": float(top2["mean_level_delta"]),
            "memb_top_delta_gap": float(abs(float(top1["mean_level_delta"]) - float(top2["mean_level_delta"]))),
            "memb_top1_abs_delta": float(abs(float(top1["mean_level_delta"]))),
            "memb_weighted_delta": float(wmean),
            "memb_weighted_delta_std": float(wstd),
            "memb_weighted_abs_delta": float(wabs),
            "memb_weight_entropy": _entropy_norm(weights),
            "memb_topk_delta_std": float(np.nanstd(topk_deltas)) if len(topk_deltas) else 0.0,
            "memb_topk_delta_range": float(np.nanmax(topk_deltas) - np.nanmin(topk_deltas)) if len(topk_deltas) else 0.0,
            "memb_topk_score_std": float(np.nanstd(topk_scores)) if len(topk_scores) else 0.0,
            "memb_prefix_top1_score": float(ptop1["prefix_member_score"]),
            "memb_prefix_top1_delta": float(ptop1["mean_level_delta"]),
            "memb_prefix_vs_lgb_candidate_diff": float(int(int(ptop1["candidate_idx"]) != int(top1["candidate_idx"]))),
            "memb_prefix_vs_lgb_cluster_diff": float(int(int(ptop1["cluster_id"]) != int(top1["cluster_id"]))),
            "memb_prefix_vs_lgb_delta_gap": float(abs(float(ptop1["mean_level_delta"]) - float(top1["mean_level_delta"]))),
            "memb_cluster_top1_score": float(ctop1["cluster_max"]),
            "memb_cluster_top2_score": float(ctop2["cluster_max"]),
            "memb_cluster_margin": float(cluster_margin),
            "memb_cluster_entropy": _entropy_norm(cw),
            "memb_cluster_top_delta": float(ctop1["cluster_delta_mean"]),
            "memb_cluster_top_abs_delta": float(abs(float(ctop1["cluster_delta_mean"]))),
            "memb_cluster_top2_delta": float(ctop2["cluster_delta_mean"]),
            "memb_cluster_delta_gap": float(cluster_delta_gap),
            "memb_cluster_neartie_10_25": float(near_tie_10_25),
            "memb_cluster_top_n": float(ctop1["cluster_n"]),
        }
        for col in ("score", "posterior", "path_ltr_score", "ms_score", "ms_score_std", "ms_corr_mean", "ms_anom_score", "ms_anom_corr", "ms_grad_score", "ms_grad_corr", "score_rank"):
            if col in ordered.columns:
                vals = ordered[col].fillna(0.0).to_numpy(np.float64)
                row[f"memb_top1_{col}"] = float(ordered[col].iloc[0]) if len(ordered) else 0.0
                row[f"memb_weighted_{col}"] = float(np.sum(weights * vals[: len(weights)])) if len(vals) == len(weights) else 0.0
        top_family = str(top1["family"])
        for fam in FAMILIES:
            row[f"memb_top_family__{fam}"] = float(top_family == fam)
            row[f"memb_weight_family__{fam}"] = float(np.sum(weights[ordered["family"].astype(str).to_numpy() == fam]))
        rows.append(row)

    out = pd.DataFrame(rows)
    numeric_cols = [c for c in out.columns if c not in {"well_id"}]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def _broadcast_well_features(well_features: pd.DataFrame, wells: np.ndarray) -> pd.DataFrame:
    keyed = well_features.set_index("well_id", drop=False)
    cols = [c for c in well_features.columns if c not in {"well_id", "well_index"}]
    out = pd.DataFrame(index=np.arange(len(wells)))
    s = pd.Series(wells.astype(str))
    for col in cols:
        out[col] = s.map(keyed[col]).fillna(0.0).to_numpy(np.float32)
    return out


def build_or_load_member_sidecar(
    args: argparse.Namespace,
    df: pd.DataFrame,
    sg_side: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    sidecar = Path(args.member_sidecar)
    groups_path = sidecar.with_suffix(".groups.json")
    if sidecar.exists() and groups_path.exists() and not args.force_rebuild_member_sidecar:
        side = pd.read_parquet(sidecar)
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        print(f"loaded member posterior sidecar {side.shape} from {sidecar}", flush=True)
        return side, groups

    t0 = time.time()
    z = np.load(args.member_npz, allow_pickle=False)
    path_keys = [k for k in z.files if k != "base" and k.startswith("seg")]
    if not path_keys:
        raise KeyError(f"no segmented member paths in {args.member_npz}; keys={z.files}")
    base = z["base"].astype(np.float32)
    if len(base) != len(df):
        raise ValueError(f"member base length mismatch: {len(base)} vs df={len(df)}")
    paths: dict[str, np.ndarray] = {}
    for key in path_keys:
        arr = z[key].astype(np.float32)
        if len(arr) != len(df):
            raise ValueError(f"member path length mismatch {key}: {len(arr)} vs df={len(df)}")
        paths[_short_mtw(key)] = arr
    best_name = "mtw4" if "mtw4" in paths else sorted(paths)[-1]
    best = paths[best_name]
    stack = np.vstack([paths[k] for k in sorted(paths)])
    mean_path = np.nanmean(stack, axis=0).astype(np.float32)
    std_path = np.nanstd(stack, axis=0).astype(np.float32)
    range_path = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)

    frac = df["frac"].to_numpy(np.float32)
    md_since = df["md_since"].to_numpy(np.float32)
    sg_prop = sg_side["sg_prop_d"].to_numpy(np.float32) if "sg_prop_d" in sg_side else df["pf_ancc_d"].to_numpy(np.float32)
    row = pd.DataFrame(
        {
            "memb_base_d": base,
            "memb_prop_best_d": best,
            "memb_prop_mean_d": mean_path,
            "memb_prop_std_d": std_path,
            "memb_prop_range_d": range_path,
            "memb_best_minus_memberbase": best - base,
            "memb_mean_minus_memberbase": mean_path - base,
            "memb_best_minus_sg_prop": best - sg_prop,
            "memb_mean_minus_sg_prop": mean_path - sg_prop,
            "memb_best_minus_ext50": best - df["ext_50"].to_numpy(np.float32),
            "memb_best_minus_ext200": best - df["ext_200"].to_numpy(np.float32),
            "memb_best_minus_pf_ancc": best - df["pf_ancc_d"].to_numpy(np.float32),
            "memb_best_minus_beam_mean": best - df["beam_mean_d"].to_numpy(np.float32),
            "memb_best_minus_sc_ens": best - df["sc_ens_d"].to_numpy(np.float32),
            "memb_best_minus_cal_proj": best - df["cal_proj_d"].to_numpy(np.float32),
            "memb_best_minus_noc_proj": best - df["noc_proj_d"].to_numpy(np.float32),
            "memb_abs_best_move_x_frac": np.abs(best - base) * frac,
            "memb_abs_best_sg_gap_x_frac": np.abs(best - sg_prop) * frac,
            "memb_prop_std_x_frac": std_path * frac,
            "memb_prop_range_x_frac": range_path * frac,
            "memb_abs_best_move_x_mdsqrt": np.abs(best - base) * np.sqrt(np.maximum(md_since, 0.0)).astype(np.float32),
        }
    )
    for name, arr in sorted(paths.items()):
        row[f"memb_path_{name}_d"] = arr
    candidate_stack = np.vstack(
        [
            best,
            mean_path,
            sg_prop,
            df["pf_ancc_d"].to_numpy(np.float32),
            df["beam_mean_d"].to_numpy(np.float32),
            df["beam_med_d"].to_numpy(np.float32),
            df["sc_ens_d"].to_numpy(np.float32),
            df["cal_proj_d"].to_numpy(np.float32),
            df["noc_proj_d"].to_numpy(np.float32),
        ]
    )
    row["memb_candidate_disagree_std"] = np.nanstd(candidate_stack, axis=0).astype(np.float32)
    row["memb_candidate_disagree_range"] = (np.nanmax(candidate_stack, axis=0) - np.nanmin(candidate_stack, axis=0)).astype(np.float32)

    well_features = _build_member_well_features(args)
    well = _broadcast_well_features(well_features, df["well"].astype(str).to_numpy())
    side = pd.concat([row, well], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    groups = {
        "path": list(row.columns),
        "well": list(well.columns),
        "all": list(side.columns),
        "path_source_keys": sorted(paths),
        "best_path": [best_name],
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    side.to_parquet(sidecar, index=False)
    groups_path.write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved member posterior sidecar {side.shape} to {sidecar} elapsed={time.time() - t0:.1f}s", flush=True)
    return side, groups


def _check_features(feats: list[str]) -> None:
    bad = [c for c in feats if any(part in c.lower() for part in BAD_FEATURE_PARTS)]
    if bad:
        raise ValueError(f"leaky feature names detected: {bad[:30]}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", type=Path, default=DEFAULT_XY)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=ART_ROOT / "full_train19_cache")
    ap.add_argument("--seg-npz", type=Path, default=DEFAULT_SEG_NPZ)
    ap.add_argument("--well-gate", type=Path, default=DEFAULT_WELL_GATE)
    ap.add_argument("--gate-meta", type=Path, default=DEFAULT_GATE_META)
    ap.add_argument("--sidecar", type=Path, default=OUT_ROOT / "seggate_transfer_features.parquet")
    ap.add_argument("--force-rebuild-sidecar", action="store_true")
    ap.add_argument("--member-npz", type=Path, default=DEFAULT_MEMBER_NPZ)
    ap.add_argument("--member-scores", type=Path, default=DEFAULT_MEMBER_SCORES)
    ap.add_argument("--member-candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--well-index", type=Path, default=DEFAULT_WELL_INDEX)
    ap.add_argument("--member-sidecar", type=Path, default=OUT_ROOT / "member_posterior_features.parquet")
    ap.add_argument("--force-rebuild-member-sidecar", action="store_true")
    ap.add_argument("--member-softmax-tau", type=float, default=0.12)
    ap.add_argument("--member-cluster-tau", type=float, default=0.12)
    ap.add_argument("--member-tie-margin-scale", type=float, default=0.08)
    ap.add_argument("--member-topk-summary", type=int, default=5)
    ap.add_argument(
        "--profiles",
        default="all_plus_sg_path_memb_path,all_plus_sg_path_memb_well,all_plus_sg_path_memb_all",
    )
    ap.add_argument("--full-5fold", action="store_true", help="Train one selected profile with full 5-fold OOF output.")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=1600)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-child-samples", type=int, default=80)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.6)
    ap.add_argument("--reg-lambda", type=float, default=10.0)
    ap.add_argument("--reg-alpha", type=float, default=1.0)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--log-period", type=int, default=300)
    ap.add_argument("--lgb-jobs", type=int, default=-1)
    ap.add_argument("--max-train-wells", type=int, default=0)
    ap.add_argument("--max-val-wells", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    needed = sorted(
        set(
            [
                "well",
                "id",
                "target",
                "frac",
                "md_since",
                "ext_all",
                "ext_200",
                "ext_50",
                "pf_ancc_d",
                "beam_mean_d",
                "beam_med_d",
                "sc_ens_d",
                "cal_proj_d",
                "noc_proj_d",
            ]
            + base_features
        )
    )
    df = pd.read_parquet(args.xy, columns=needed)
    sg_side, sg_groups = build_sg_sidecar(args, df)
    member_side, member_groups = build_or_load_member_sidecar(args, df, sg_side)
    work = pd.concat([df.reset_index(drop=True), sg_side.reset_index(drop=True), member_side.reset_index(drop=True)], axis=1)

    sg_path = sg_groups["path"]
    member_path = member_groups["path"]
    member_well = member_groups["well"]
    profile_map = {
        "baseline_all": base_features,
        "all_plus_sg_path": base_features + sg_path,
        "all_plus_sg_path_memb_path": base_features + sg_path + member_path,
        "all_plus_sg_path_memb_well": base_features + sg_path + member_well,
        "all_plus_sg_path_memb_all": base_features + sg_path + member_path + member_well,
    }
    selected_profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
    for profile in selected_profiles:
        if profile not in profile_map:
            raise KeyError(f"unknown profile={profile}; available={sorted(profile_map)}")
        _check_features(profile_map[profile])

    if args.full_5fold:
        if len(selected_profiles) != 1:
            raise ValueError("--full-5fold expects exactly one profile")
        profile = selected_profiles[0]
        args.out_dir = out_dir
        feats = profile_map[profile]
        print(f"\n=== train full 5fold profile={profile} n_features={len(feats)} ===", flush=True)
        summary, _ = train_lgb_5fold(work, feats, args)
        summary.to_csv(out_dir / "summary.csv", index=False)
        (out_dir / "feature_groups.json").write_text(
            json.dumps(
                {
                    "sg": sg_groups,
                    "member": member_groups,
                    "profile": profile,
                    "feature_count": len(feats),
                    "base_features": len(base_features),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "metadata_member.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"out_dir={out_dir}", flush=True)
        return

    rows = []
    for profile in selected_profiles:
        feats = profile_map[profile]
        print(f"\n=== train profile={profile} n_features={len(feats)} ===", flush=True)
        rows.append(train_profile(work, feats, args, profile, out_dir))
    summary = pd.concat(rows, ignore_index=True).sort_values("rmse")
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "feature_groups.json").write_text(
        json.dumps(
            {
                "sg": sg_groups,
                "member": member_groups,
                "profiles": {k: len(v) for k, v in profile_map.items()},
                "base_features": len(base_features),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("\n=== summary sorted ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
