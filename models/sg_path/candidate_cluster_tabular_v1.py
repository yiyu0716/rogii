#!/usr/bin/env python3
"""Candidate-cluster tabular posterior experiment.

This experiment uses tabular models at the candidate-cluster level instead of
rowwise TVT regression.  The default feature mode intentionally avoids the
same GR/PF residual evidence that has repeatedly failed to disambiguate
Eagle Ford aliases.  It keeps formation/order, prefix TVT/F continuity,
candidate path physics, candidate geology, and cluster geometry features.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("ROGII_ROOT", str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from candidate_cluster_selector_v1 import assign_delta_clusters
from candidate_path295_ranker_v1 import ART_DIR, DATA_ROOT, _interp_to_length, load_base_prediction
from candidate_path_ltr_prefix_posterior_gate_v1 import assign_gap_clusters
from candidate_pool_governor_v1 import apply_ramped_move, well_slices_from_wells
from cv_runner import ROOT, md_table
from private_safe_visible_prefix_candidate_posterior_v1 import PREFIX_SPECS, GEOM_COST_COLS, mean_cost
from private_safe_vp_fast_eval import _path_for_row, _score_with_cuts, _split_csv
from segmented_geology_physics_posterior_v1 import segmented_prediction_one_combo


DEFAULT_CACHE_DIR = ART_DIR / "whole_trace_piecewise_u_path_cache_v2_diverse_full_top60_20260706"
DEFAULT_PREFIX_CSV = ART_DIR / "candidate_prefix_features_diverse_full_top60_20260706_noleak_noformation" / "candidate_prefix_features.csv"
DEFAULT_BASE_NPZ = ART_DIR / "rawpfmean_current_hinge_softgr2_base_20260707" / "predictions.npz"
DEFAULT_RUN_TAG = "candidate_cluster_tabular_v1_20260707"
SEED = 20260707

ID_EXACT = {
    "well_index",
    "cluster_id",
    "candidate_idx",
    "record_start",
    "record_len",
    "fold",
    "well_id",
}
BAD_PARTS = (
    "target",
    "truth",
    "oracle",
    "selected",
    "apply",
    "ranker_pred",
    "cluster_rel",
)
SAME_EVIDENCE_PARTS = (
    "ms_",
    "score",
    "posterior",
    "path_ltr",
    "seq_resid",
    "gr_",
    "_gr",
    "pf_",
    "_pf",
    "ncc",
    "ancc",
    "rmse",
    "mae",
    "resid",
)
LEAKY_PARTS = (
    "formation_contact",
)
ORTHOGONAL_PREFIXES = (
    "prefix_",
    "rel_prefix_",
    "mapdiff_prefix_",
    "anchordiff_prefix_",
    "agg_prefix_",
    "candgeo_",
    "rel_candgeo_",
    "agg_candgeo_",
    "candphys_",
    "rel_candphys_",
    "agg_candphys_",
    "cand_drift_",
    "rel_cand_drift_",
    "agg_cand_drift_",
    "seq_path_",
    "agg_seq_path_",
    "family__",
    "agg_family__",
    "cluster_delta",
    "rel_cluster_delta",
    "cluster_n",
    "rel_cluster_n",
)


def softmax(values: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x
    x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
    z = x / max(float(tau), 1e-9)
    z -= float(np.max(z))
    w = np.exp(np.clip(z, -80.0, 0.0))
    return w / max(float(np.sum(w)), 1e-12)


def entropy_norm(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[w > 0.0]
    if len(w) <= 1:
        return 0.0
    ent = -float(np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
    return float(np.clip(ent / np.log(float(len(w))), 0.0, 1.0))


def soft_utility_from_rmse(rmse: np.ndarray, tau: float) -> np.ndarray:
    arr = np.asarray(rmse, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 99.0)
    return np.exp(-np.maximum(arr, 0.0) / max(float(tau), 1e-6))


def _safe_std(series: pd.Series) -> float:
    val = float(pd.to_numeric(series, errors="coerce").std(skipna=True) or 0.0)
    return val if np.isfinite(val) else 0.0


def _allowed_orthogonal_name(name: str) -> bool:
    low = str(name).lower()
    if low in ID_EXACT:
        return False
    if any(part in low for part in BAD_PARTS):
        return False
    if any(part in low for part in LEAKY_PARTS):
        return False
    if any(part in low for part in SAME_EVIDENCE_PARTS):
        return False
    return low.startswith(ORTHOGONAL_PREFIXES)


def formation_contact_count(frame: pd.DataFrame) -> int:
    if "family" not in frame.columns:
        return 0
    fam = frame["family"].astype(str).str.lower()
    return int(fam.eq("formation_contact").sum())


def assert_deployable_candidate_pool(frame: pd.DataFrame, *, allow_formation_contact: bool = False) -> None:
    """Guard against train-only formation-surface candidate evidence.

    Real Kaggle horizontal test files do not include the per-row formation
    surface columns used to create ``family=formation_contact`` candidates.
    Those candidates are valuable as a diagnostic oracle, but any default CV
    using them is not deployable and will overstate leaderboard value.
    """

    n_contact = formation_contact_count(frame)
    if n_contact and not bool(allow_formation_contact):
        raise RuntimeError(
            "Candidate pool contains family=formation_contact rows "
            f"({n_contact:,}). Those rows require train-only formation-surface "
            "columns and are not deployable on real Kaggle test/private data. "
            "Use the noformation candidate pool, or pass "
            "--allow-formation-contact only for explicit diagnostic/oracle runs."
        )


def cluster_tabular_feature_columns(frame: pd.DataFrame, *, feature_mode: str = "orthogonal") -> list[str]:
    """Return deployable cluster-level tabular features.

    ``orthogonal`` is the default research constraint: no target/oracle columns,
    no GR/PF residual proxies, and no existing ranker/score/posterior columns.
    """

    cols: list[str] = []
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        low = str(col).lower()
        if low in ID_EXACT or any(part in low for part in BAD_PARTS):
            continue
        if str(feature_mode) == "orthogonal":
            if not _allowed_orthogonal_name(str(col)):
                continue
        elif str(feature_mode) == "truthfree":
            if "target" in low or "oracle" in low or "truth" in low:
                continue
        else:
            raise ValueError(f"unknown feature_mode={feature_mode!r}")
        if _safe_std(frame[col]) > 1e-12:
            cols.append(str(col))
    return cols


def candidate_member_feature_columns(frame: pd.DataFrame, *, feature_mode: str = "orthogonal") -> list[str]:
    """Return deployable candidate-member evidence columns.

    This is intentionally stricter than a generic candidate ranker: in
    ``orthogonal`` mode it keeps only prefix continuity, candidate geology,
    candidate physics, family, and datum-cluster geometry style features.  It
    excludes GR/PF/residual/score proxies so the member posterior can add a
    different evidence layer instead of re-learning the same alias-prone GR fit.
    """

    return cluster_tabular_feature_columns(frame, feature_mode=str(feature_mode))


def add_family_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "family" not in out.columns:
        return out
    fam = out["family"].astype(str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True).str.lower()
    for value in sorted(fam.dropna().unique().tolist()):
        out[f"family__{value}"] = fam.eq(value).astype(float)
    return out


def assign_clusters(frame: pd.DataFrame, *, cluster_gap: float, cluster_method: str) -> pd.DataFrame:
    if str(cluster_method) == "gap":
        return assign_gap_clusters(frame, cluster_gap=float(cluster_gap))
    return assign_delta_clusters(frame, cluster_gap=float(cluster_gap), method=str(cluster_method))


def candidate_aggregate_columns(frame: pd.DataFrame, *, feature_mode: str) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if str(feature_mode) == "orthogonal" and not _allowed_orthogonal_name(str(col)):
            continue
        low = str(col).lower()
        if low in ID_EXACT or any(part in low for part in BAD_PARTS):
            continue
        if _safe_std(frame[col]) > 1e-12:
            cols.append(str(col))
    return cols


def add_cluster_relative_features(clusters: pd.DataFrame) -> pd.DataFrame:
    out = clusters.copy()
    grouped = out.groupby("well_index", sort=False)
    rel_cols = [
        c
        for c in out.columns
        if pd.api.types.is_numeric_dtype(out[c])
        and (str(c).startswith("cluster_") or str(c).startswith("agg_"))
        and "target" not in str(c).lower()
    ]
    for col in rel_cols:
        x = pd.to_numeric(out[col], errors="coerce").astype(float)
        mean = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
        std = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)).replace(0.0, np.nan)
        out[f"rel_{col}_z"] = ((x - mean.astype(float)) / std.astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"rel_{col}_pct_asc"] = grouped[col].rank(method="average", ascending=True, pct=True).astype(float)
        out[f"rel_{col}_pct_desc"] = grouped[col].rank(method="average", ascending=False, pct=True).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def build_cluster_tabular_table(
    candidates: pd.DataFrame,
    *,
    cluster_gap: float,
    cluster_method: str,
    feature_mode: str,
    use_family_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = add_family_indicators(candidates) if bool(use_family_features) else candidates.copy()
    work = assign_clusters(work, cluster_gap=float(cluster_gap), cluster_method=str(cluster_method))
    work["cluster_id"] = work["cluster_id"].astype(int)
    keys = [work["well_index"], work["cluster_id"]]
    grouped = work.groupby(["well_index", "cluster_id"], sort=False)

    sort_cols = [c for c in ["well_index", "cluster_id", "score_rank", "candidate_idx"] if c in work.columns]
    rep = work.sort_values(sort_cols, ascending=[True] * len(sort_cols)).groupby(["well_index", "cluster_id"], sort=False).head(1)
    clusters = rep.set_index(["well_index", "cluster_id"], drop=False).copy()

    stats = pd.DataFrame(index=grouped.size().index)
    stats["cluster_n"] = grouped.size().astype(float)
    delta = pd.to_numeric(work.get("mean_level_delta", 0.0), errors="coerce").fillna(0.0)
    stats["cluster_delta_mean"] = delta.groupby(keys, sort=False).mean()
    stats["cluster_delta_std"] = delta.groupby(keys, sort=False).std(ddof=0).fillna(0.0)
    stats["cluster_delta_min"] = delta.groupby(keys, sort=False).min()
    stats["cluster_delta_max"] = delta.groupby(keys, sort=False).max()
    stats["cluster_delta_span"] = stats["cluster_delta_max"] - stats["cluster_delta_min"]
    stats["cluster_delta_abs_mean"] = delta.abs().groupby(keys, sort=False).mean()

    if "target_sse_per_row" in work.columns:
        oracle_sort = work.sort_values(["well_index", "cluster_id", "target_sse_per_row"], ascending=[True, True, True])
        oracle = oracle_sort.groupby(["well_index", "cluster_id"], sort=False).head(1).set_index(["well_index", "cluster_id"])
        stats["target_cluster_oracle_sse"] = pd.to_numeric(oracle["target_sse_per_row"], errors="coerce")
        if "target_rmse" in oracle:
            stats["target_cluster_oracle_rmse"] = pd.to_numeric(oracle["target_rmse"], errors="coerce")
        else:
            stats["target_cluster_oracle_rmse"] = np.sqrt(stats["target_cluster_oracle_sse"].clip(lower=0.0))
        if "target_abs_level" in oracle:
            stats["target_cluster_oracle_abs_level"] = pd.to_numeric(oracle["target_abs_level"], errors="coerce")
        stats["target_cluster_oracle_candidate_idx"] = pd.to_numeric(oracle["candidate_idx"], errors="coerce").fillna(-999).astype(int)

    agg_cols = candidate_aggregate_columns(work, feature_mode=str(feature_mode))
    if agg_cols:
        num = work[agg_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        fg = num.groupby(keys, sort=False)
        pieces = [
            fg.mean().rename(columns={c: f"agg_{c}_mean" for c in agg_cols}),
            fg.min().rename(columns={c: f"agg_{c}_min" for c in agg_cols}),
            fg.max().rename(columns={c: f"agg_{c}_max" for c in agg_cols}),
        ]
        stats = stats.join(pd.concat(pieces, axis=1))

    clusters = clusters.join(stats).reset_index(drop=True)
    clusters = add_cluster_relative_features(clusters)
    return clusters.replace([np.inf, -np.inf], np.nan), work.replace([np.inf, -np.inf], np.nan)


def load_folds(scheme: str) -> dict[str, int]:
    frame = pd.read_csv(ROOT / "cv" / "splits" / f"{scheme}_folds.csv", dtype={"well_id": str})
    return dict(zip(frame["well_id"].astype(str), frame["fold"].astype(int)))


def make_model(kind: str, seed: int):
    if kind == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=480,
                max_depth=9,
                min_samples_leaf=4,
                max_features=0.65,
                bootstrap=False,
                random_state=int(seed),
                n_jobs=-1,
            ),
        )
    if kind == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.040,
                max_iter=360,
                max_leaf_nodes=31,
                l2_regularization=0.12,
                min_samples_leaf=18,
                random_state=int(seed),
            ),
        )
    if kind == "cat":
        from catboost import CatBoostRegressor

        return make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostRegressor(
                iterations=700,
                depth=6,
                learning_rate=0.035,
                loss_function="RMSE",
                random_seed=int(seed),
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
        )
    raise ValueError(f"unknown model kind={kind!r}")


def target_values(clusters: pd.DataFrame, *, target_mode: str, label_tau: float) -> np.ndarray:
    rmse = pd.to_numeric(clusters["target_cluster_oracle_rmse"], errors="coerce").to_numpy(np.float64)
    if str(target_mode) == "utility":
        return soft_utility_from_rmse(rmse, tau=float(label_tau))
    if str(target_mode) == "rmse":
        return rmse
    if str(target_mode) == "log_rmse":
        return np.log1p(np.maximum(rmse, 0.0))
    raise ValueError(f"unknown target_mode={target_mode!r}")


def candidate_member_target_values(candidates: pd.DataFrame, *, target_mode: str, label_tau: float) -> np.ndarray:
    rmse = pd.to_numeric(candidates["target_rmse"], errors="coerce").to_numpy(np.float64)
    if str(target_mode) == "utility":
        return soft_utility_from_rmse(rmse, tau=float(label_tau))
    if str(target_mode) == "rmse":
        return rmse
    if str(target_mode) == "log_rmse":
        return np.log1p(np.maximum(rmse, 0.0))
    raise ValueError(f"unknown target_mode={target_mode!r}")


def score_from_prediction(pred: np.ndarray, *, target_mode: str) -> np.ndarray:
    if str(target_mode) == "utility":
        return np.asarray(pred, dtype=np.float64)
    return -np.asarray(pred, dtype=np.float64)


def fit_oof_tabular_scores(
    clusters: pd.DataFrame,
    feature_cols: list[str],
    *,
    kinds: list[str],
    scheme: str,
    target_mode: str,
    label_tau: float,
    seed: int,
) -> tuple[pd.Series, pd.DataFrame]:
    folds = load_folds(str(scheme))
    fold_values = clusters["well_id"].astype(str).map(folds).fillna(-1).astype(int)
    X = clusters[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = target_values(clusters, target_mode=str(target_mode), label_tau=float(label_tau))
    all_scores: list[np.ndarray] = []
    imps: list[dict[str, object]] = []
    for kind_i, kind in enumerate(kinds):
        pred = np.full(len(clusters), np.nan, dtype=np.float64)
        for fold in sorted(v for v in fold_values.unique() if v >= 0):
            val_mask = fold_values.eq(fold).to_numpy()
            train_mask = (~val_mask) & np.isfinite(y) & fold_values.ge(0).to_numpy()
            if int(train_mask.sum()) < 20 or int(val_mask.sum()) == 0:
                continue
            model = make_model(str(kind), int(seed) + 1000 * kind_i + int(fold))
            model.fit(X.loc[train_mask], y[train_mask])
            pred[val_mask] = model.predict(X.loc[val_mask])
            final = model[-1] if hasattr(model, "__getitem__") else model
            if hasattr(final, "feature_importances_"):
                for feature, value in zip(feature_cols, final.feature_importances_):
                    imps.append({"kind": str(kind), "fold": int(fold), "feature": feature, "importance": float(value)})
            print(
                f"[cluster-tab {kind} fold {fold}] train={int(train_mask.sum()):,} val={int(val_mask.sum()):,}",
                flush=True,
            )
        all_scores.append(score_from_prediction(pred, target_mode=str(target_mode)))
    stack = np.vstack(all_scores)
    score = np.nanmean(stack, axis=0)
    return pd.Series(score, index=clusters.index, name="tab_score"), pd.DataFrame(imps)


def fit_oof_member_tabular_scores(
    candidates: pd.DataFrame,
    feature_cols: list[str],
    *,
    kinds: list[str],
    scheme: str,
    target_mode: str,
    label_tau: float,
    seed: int,
) -> tuple[pd.Series, pd.DataFrame]:
    folds = load_folds(str(scheme))
    fold_values = candidates["well_id"].astype(str).map(folds).fillna(-1).astype(int)
    X = candidates[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = candidate_member_target_values(candidates, target_mode=str(target_mode), label_tau=float(label_tau))
    all_scores: list[np.ndarray] = []
    imps: list[dict[str, object]] = []
    for kind_i, kind in enumerate(kinds):
        pred = np.full(len(candidates), np.nan, dtype=np.float64)
        for fold in sorted(v for v in fold_values.unique() if v >= 0):
            val_mask = fold_values.eq(fold).to_numpy()
            train_mask = (~val_mask) & np.isfinite(y) & fold_values.ge(0).to_numpy()
            if int(train_mask.sum()) < 20 or int(val_mask.sum()) == 0:
                continue
            model = make_model(str(kind), int(seed) + 7000 + 1000 * kind_i + int(fold))
            model.fit(X.loc[train_mask], y[train_mask])
            pred[val_mask] = model.predict(X.loc[val_mask])
            final = model[-1] if hasattr(model, "__getitem__") else model
            if hasattr(final, "feature_importances_"):
                for feature, value in zip(feature_cols, final.feature_importances_):
                    imps.append({"kind": f"member_{kind}", "fold": int(fold), "feature": feature, "importance": float(value)})
            print(
                f"[member-tab {kind} fold {fold}] train={int(train_mask.sum()):,} val={int(val_mask.sum()):,}",
                flush=True,
            )
        all_scores.append(score_from_prediction(pred, target_mode=str(target_mode)))
    stack = np.vstack(all_scores)
    score = np.nanmean(stack, axis=0)
    return pd.Series(score, index=candidates.index, name="member_tab_score"), pd.DataFrame(imps)


def combine_segment_emission_scores(
    frame: pd.DataFrame,
    *,
    member_score_weight: float,
    member_tab_weight: float,
    cluster_score_col: str = "tab_score",
    member_score_col: str = "member_score",
    member_tab_col: str = "member_tab_score",
) -> np.ndarray:
    score = pd.to_numeric(frame[cluster_score_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    if float(member_score_weight) != 0.0 and member_score_col in frame.columns:
        score += float(member_score_weight) * pd.to_numeric(frame[member_score_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    if float(member_tab_weight) != 0.0 and member_tab_col in frame.columns:
        score += float(member_tab_weight) * pd.to_numeric(frame[member_tab_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return score


def _member_score(frame: pd.DataFrame, *, prefix_spec: str, member_geom_beta: float) -> pd.Series:
    cols = PREFIX_SPECS.get(str(prefix_spec), [])
    prefix_cost = mean_cost(frame, cols, fallback=0.5) if cols else pd.Series(0.5, index=frame.index)
    geom_cost = mean_cost(frame, GEOM_COST_COLS, fallback=0.5)
    score = -prefix_cost.to_numpy(np.float64) - float(member_geom_beta) * geom_cost.to_numpy(np.float64)
    return pd.Series(score, index=frame.index, name="member_score")


def build_member_paths(
    clustered: pd.DataFrame,
    well_index: pd.DataFrame,
    path_drift: np.ndarray,
    slices: dict[str, tuple[int, int]],
    *,
    max_members: int,
    prefix_spec: str,
    member_geom_beta: float,
) -> dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray]]]:
    work = clustered.copy()
    if "member_score" not in work.columns:
        work["member_score"] = _member_score(work, prefix_spec=str(prefix_spec), member_geom_beta=float(member_geom_beta))
    wi_to_wid = dict(zip(well_index["well_index"].astype(int), well_index["well_id"].astype(str)))
    out: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray]]] = {}
    for (wi, cid), group in work.groupby(["well_index", "cluster_id"], sort=False):
        wid = wi_to_wid.get(int(wi))
        if wid is None or wid not in slices:
            continue
        n = slices[wid][1] - slices[wid][0]
        ordered = group.sort_values(["member_score", "candidate_idx"], ascending=[False, True]).head(max(1, int(max_members)))
        scores = pd.to_numeric(ordered["member_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        traces = [_path_for_row(row, path_drift, n) for _, row in ordered.iterrows()]
        out[(int(wi), int(cid))] = (scores, traces)
    return out


def posterior_expected_path_from_clusters(
    clusters: pd.DataFrame,
    member_paths: dict[tuple[int, int], list[np.ndarray] | tuple[np.ndarray, list[np.ndarray]]],
    *,
    top_clusters: int,
    tau_cluster: float,
    well_lengths: dict[int, int],
    score_col: str = "tab_score",
    member_top: int = 1,
    tau_member: float = 1.0,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    preds: dict[int, np.ndarray] = {}
    metrics: list[dict[str, float | int]] = []
    for wi, group in clusters.groupby("well_index", sort=False):
        n = int(well_lengths[int(wi)])
        ordered = group.sort_values([score_col, "cluster_id"], ascending=[False, True]).head(max(1, int(top_clusters)))
        scores = pd.to_numeric(ordered[score_col], errors="coerce").fillna(-1e9).to_numpy(np.float64)
        weights = softmax(scores, float(tau_cluster))
        acc = np.zeros(n, dtype=np.float64)
        for cw, (_, row) in zip(weights, ordered.iterrows()):
            cid = int(row["cluster_id"])
            packed = member_paths.get((int(wi), cid))
            if packed is None:
                continue
            if isinstance(packed, tuple):
                member_scores, paths = packed
                take = min(int(member_top), len(paths))
                mw = softmax(np.asarray(member_scores[:take], dtype=np.float64), float(tau_member))
                one = np.zeros(n, dtype=np.float64)
                for w, path in zip(mw, paths[:take]):
                    one += float(w) * _interp_to_length(path, n)
            else:
                paths = packed
                take = min(int(member_top), len(paths))
                one = np.mean([_interp_to_length(p, n) for p in paths[:take]], axis=0)
            acc += float(cw) * one
        preds[int(wi)] = acc
        margin = float(scores[0] - scores[1]) if len(scores) > 1 else 99.0
        metrics.append(
            {
                "well_index": int(wi),
                "n_clusters_used": int(len(ordered)),
                "cluster_entropy_norm": entropy_norm(weights),
                "cluster_margin": margin,
                "top_cluster_prob": float(np.max(weights)) if len(weights) else 1.0,
            }
        )
    return preds, pd.DataFrame(metrics)


def row_prediction_from_cluster_paths(
    clusters: pd.DataFrame,
    member_paths: dict[tuple[int, int], tuple[np.ndarray, list[np.ndarray]]],
    well_index: pd.DataFrame,
    base_pred: np.ndarray,
    slices: dict[str, tuple[int, int]],
    *,
    top_clusters: int,
    tau_cluster: float,
    member_top: int,
    tau_member: float,
    score_col: str = "tab_score",
) -> tuple[np.ndarray, pd.DataFrame]:
    wi_to_wid = dict(zip(well_index["well_index"].astype(int), well_index["well_id"].astype(str)))
    well_lengths = {int(wi): slices[wid][1] - slices[wid][0] for wi, wid in wi_to_wid.items() if wid in slices}
    pred_by_wi, metrics = posterior_expected_path_from_clusters(
        clusters,
        member_paths,
        top_clusters=int(top_clusters),
        tau_cluster=float(tau_cluster),
        well_lengths=well_lengths,
        score_col=str(score_col),
        member_top=int(member_top),
        tau_member=float(tau_member),
    )
    out = np.asarray(base_pred, dtype=np.float64).copy()
    for wi, path in pred_by_wi.items():
        wid = wi_to_wid.get(int(wi))
        if wid is None or wid not in slices:
            continue
        start, end = slices[wid]
        out[start:end] = _interp_to_length(path, end - start)
    metrics["well_id"] = metrics["well_index"].astype(int).map(wi_to_wid)
    return out, metrics


def gated_ramp(
    wells: np.ndarray,
    base_pred: np.ndarray,
    posterior_pred: np.ndarray,
    metrics: pd.DataFrame,
    *,
    alpha: float,
    clip: float,
    entropy_max: float,
    move_max: float,
    top_prob_min: float,
    slices: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    if slices is None:
        slices = well_slices_from_wells(wells.astype(str))
    lookup = metrics.set_index("well_id", drop=False) if "well_id" in metrics.columns else pd.DataFrame()
    out = np.asarray(base_pred, dtype=np.float64).copy()
    for wid, (start, end) in slices.items():
        if wid not in lookup.index:
            continue
        row = lookup.loc[wid]
        entropy = float(row.get("cluster_entropy_norm", 1.0))
        top_prob = float(row.get("top_cluster_prob", 0.0))
        move = float(np.mean(np.abs(posterior_pred[start:end] - base_pred[start:end])))
        if entropy > float(entropy_max) or top_prob < float(top_prob_min) or move > float(move_max):
            continue
        out[start:end] = apply_ramped_move(
            base_pred[start:end],
            posterior_pred[start:end],
            alpha=float(alpha),
            clip=float(clip),
            ramp=True,
        )
    return out


def cluster_score_metrics(clusters: pd.DataFrame, well_index: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    wi_to_wid = dict(zip(well_index["well_index"].astype(int), well_index["well_id"].astype(str)))
    for wi, group in clusters.groupby("well_index", sort=False):
        vals = pd.to_numeric(group[score_col], errors="coerce").fillna(-1e9).to_numpy(np.float64)
        if len(vals) == 0:
            continue
        order = np.argsort(-vals)
        w = softmax(vals[order[: min(8, len(vals))]], 1.0)
        rows.append(
            {
                "well_index": int(wi),
                "well_id": wi_to_wid.get(int(wi), ""),
                "cluster_entropy_norm": entropy_norm(w),
                "cluster_margin": float(vals[order[0]] - vals[order[1]]) if len(order) > 1 else 99.0,
                "top_cluster_prob": float(w[0]) if len(w) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def attach_well_ids(candidates: pd.DataFrame, well_index: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if "well_id" not in out.columns:
        meta = well_index[["well_index", "well_id"]].copy()
        meta["well_index"] = meta["well_index"].astype(int)
        out["well_index"] = out["well_index"].astype(int)
        out = out.merge(meta, on="well_index", how="left")
    return out


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    well_index = pd.read_csv(Path(args.cache_dir) / "well_index.csv", dtype={"well_id": str})
    candidates = pd.read_csv(args.prefix_features_csv, dtype={"well_id": str})
    assert_deployable_candidate_pool(candidates, allow_formation_contact=bool(args.allow_formation_contact))
    candidates = candidates[pd.to_numeric(candidates["candidate_idx"], errors="coerce").fillna(-1).astype(int) >= 0].copy()
    candidates = attach_well_ids(candidates, well_index)
    if int(args.max_wells) > 0:
        keep_wi = sorted(candidates["well_index"].astype(int).unique().tolist())[: int(args.max_wells)]
        candidates = candidates[candidates["well_index"].astype(int).isin(keep_wi)].reset_index(drop=True)
        well_index = well_index[well_index["well_index"].astype(int).isin(keep_wi)].reset_index(drop=True)
    path_drift = np.load(Path(args.cache_dir) / "path_cache_topk.npz", mmap_mode="r")["path_drift"]
    wells, target, base_pred = load_base_prediction(args)
    if int(args.max_wells) > 0:
        keep_ids = set(candidates["well_id"].astype(str).unique().tolist())
        mask = np.asarray([str(w) in keep_ids for w in wells.astype(str)], dtype=bool)
        wells = wells[mask]
        target = target[mask]
        base_pred = base_pred[mask]
    return candidates, well_index, path_drift, wells, target, base_pred


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    candidates, well_index, path_drift, wells, target, base_pred = load_inputs(args)
    print(f"candidate_rows={len(candidates):,} wells={candidates['well_index'].nunique():,}", flush=True)
    clusters, clustered = build_cluster_tabular_table(
        candidates,
        cluster_gap=float(args.cluster_gap),
        cluster_method=str(args.cluster_method),
        feature_mode=str(args.feature_mode),
        use_family_features=not bool(args.no_family_features),
    )
    print(f"cluster_rows={len(clusters):,} cols={len(clusters.columns):,}", flush=True)
    feature_cols = cluster_tabular_feature_columns(clusters, feature_mode=str(args.feature_mode))
    if not feature_cols:
        raise RuntimeError("No cluster tabular features selected")
    print(f"feature_cols={len(feature_cols)} mode={args.feature_mode}", flush=True)
    kinds = _split_csv(args.model_kinds, str)
    tab_score, imps = fit_oof_tabular_scores(
        clusters,
        feature_cols,
        kinds=kinds,
        scheme=str(args.scheme),
        target_mode=str(args.target_mode),
        label_tau=float(args.label_tau),
        seed=int(args.seed),
    )
    clusters["tab_score"] = tab_score
    clustered = clustered.merge(clusters[["well_index", "cluster_id", "tab_score"]], on=["well_index", "cluster_id"], how="left")
    clustered["tab_score"] = pd.to_numeric(clustered["tab_score"], errors="coerce").fillna(clustered["tab_score"].median())
    clustered["member_score"] = _member_score(clustered, prefix_spec=str(args.member_prefix_spec), member_geom_beta=float(args.member_geom_beta))
    member_tab_weights = _split_csv(args.member_tab_weights, float) if args.member_tab_weights else [float(args.member_tab_weight)]
    needs_member_tab = any(abs(float(w)) > 1e-12 for w in member_tab_weights)
    if needs_member_tab:
        member_feature_cols = candidate_member_feature_columns(clustered, feature_mode=str(args.feature_mode))
        if not member_feature_cols:
            raise RuntimeError("No candidate member tabular features selected")
        print(
            f"member_feature_cols={len(member_feature_cols)} mode={args.feature_mode} "
            f"weights={','.join(f'{float(w):g}' for w in member_tab_weights)}",
            flush=True,
        )
        member_score, member_imps = fit_oof_member_tabular_scores(
            clustered,
            member_feature_cols,
            kinds=_split_csv(args.member_tab_model_kinds, str),
            scheme=str(args.scheme),
            target_mode=str(args.member_tab_target_mode),
            label_tau=float(args.member_tab_label_tau),
            seed=int(args.seed),
        )
        clustered["member_tab_score"] = member_score
        if not member_imps.empty:
            imps = pd.concat([imps, member_imps], ignore_index=True)
    else:
        clustered["member_tab_score"] = 0.0

    slices = well_slices_from_wells(wells.astype(str))
    rows: list[dict[str, float | str | int]] = [
        {**_score_with_cuts(base_pred, target, slices, "base", cut_fracs=_split_csv(args.cut_fracs, float)), "kind": "base"}
    ]
    saved: dict[str, np.ndarray] = {"base": base_pred.astype(np.float32)}
    modes = set(_split_csv(args.modes, str))

    if "posterior" in modes:
        max_members = max(_split_csv(args.member_top, int) or [1])
        member_paths = build_member_paths(
            clustered,
            well_index,
            path_drift,
            slices,
            max_members=max_members,
            prefix_spec=str(args.member_prefix_spec),
            member_geom_beta=float(args.member_geom_beta),
        )
        for top_c in _split_csv(args.top_clusters, int):
            for member_top in _split_csv(args.member_top, int):
                for tau_c in _split_csv(args.tau_cluster, float):
                    for tau_m in _split_csv(args.tau_member, float):
                        posterior, metrics = row_prediction_from_cluster_paths(
                            clusters,
                            member_paths,
                            well_index,
                            base_pred,
                            slices,
                            top_clusters=int(top_c),
                            tau_cluster=float(tau_c),
                            member_top=int(member_top),
                            tau_member=float(tau_m),
                        )
                        base_name = f"tabpost_C{top_c}_M{member_top}_tc{tau_c:g}_tm{tau_m:g}"
                        rows.append(
                            {
                                **_score_with_cuts(posterior, target, slices, f"direct_{base_name}", cut_fracs=_split_csv(args.cut_fracs, float)),
                                "kind": "direct_tabular_posterior",
                                "top_clusters": int(top_c),
                                "member_top": int(member_top),
                                "tau_cluster": float(tau_c),
                                "tau_member": float(tau_m),
                            }
                        )
                        for alpha in _split_csv(args.alpha, float):
                            for clip in _split_csv(args.clip, float):
                                for entropy_max in _split_csv(args.entropy_max, float):
                                    for move_max in _split_csv(args.move_max, float):
                                        for top_prob_min in _split_csv(args.top_prob_min, float):
                                            pred = gated_ramp(
                                                wells,
                                                base_pred,
                                                posterior,
                                                metrics,
                                                alpha=float(alpha),
                                                clip=float(clip),
                                                entropy_max=float(entropy_max),
                                                move_max=float(move_max),
                                                top_prob_min=float(top_prob_min),
                                                slices=slices,
                                            )
                                            name = (
                                                f"{base_name}_a{alpha:g}_c{clip:g}_e{entropy_max:g}"
                                                f"_mv{move_max:g}_tp{top_prob_min:g}"
                                            )
                                            rows.append(
                                                {
                                                    **_score_with_cuts(pred, target, slices, name, cut_fracs=_split_csv(args.cut_fracs, float)),
                                                    "kind": "gated_tabular_overlay",
                                                    "top_clusters": int(top_c),
                                                    "member_top": int(member_top),
                                                    "tau_cluster": float(tau_c),
                                                    "tau_member": float(tau_m),
                                                    "alpha": float(alpha),
                                                    "clip": float(clip),
                                                    "entropy_max": float(entropy_max),
                                                    "move_max": float(move_max),
                                                    "top_prob_min": float(top_prob_min),
                                                }
                                            )
                                            if len(saved) < int(args.save_top_predictions):
                                                saved[name] = pred.astype(np.float32)

    if "segmented" in modes:
        tab_metrics = cluster_score_metrics(clusters, well_index, score_col="tab_score")
        for member_tab_w in member_tab_weights:
            scored = clustered.copy()
            scored["vp_score"] = combine_segment_emission_scores(
                scored,
                member_score_weight=float(args.member_score_weight),
                member_tab_weight=float(member_tab_w),
            )
            for seg_len in _split_csv(args.segment_lens, int):
                for max_cand in _split_csv(args.max_candidates_grid, int):
                    for score_w in _split_csv(args.score_weights, float):
                        for geo_b in _split_csv(args.geology_betas, float):
                            post = segmented_prediction_one_combo(
                                scored,
                                well_index,
                                path_drift,
                                base_pred,
                                slices,
                                segment_len=int(seg_len),
                                max_candidates=int(max_cand),
                                score_weight=float(score_w),
                                physics_beta=float(args.physics_beta),
                                geology_beta=float(geo_b),
                                transition_sigma=float(args.transition_sigma),
                                jump_sigma=float(args.jump_sigma),
                                temperature=float(args.temperature),
                                data_root=Path(args.data_root),
                                max_wells=int(args.max_wells),
                            )
                            key = f"tabseg{seg_len}_k{max_cand}_sw{score_w:g}_gb{geo_b:g}_mtw{float(member_tab_w):g}"
                            rows.append(
                                {
                                    **_score_with_cuts(post, target, slices, f"direct_{key}", cut_fracs=_split_csv(args.cut_fracs, float)),
                                    "kind": "direct_tabular_segmented",
                                    "segment_len": int(seg_len),
                                    "max_candidates": int(max_cand),
                                    "score_weight": float(score_w),
                                    "geology_beta": float(geo_b),
                                    "member_tab_weight": float(member_tab_w),
                                }
                            )
                            for alpha in _split_csv(args.seg_alpha, float):
                                for clip in _split_csv(args.seg_clip, float):
                                    for entropy_max in _split_csv(args.seg_entropy_max, float):
                                        for move_max in _split_csv(args.seg_move_max, float):
                                            pred = gated_ramp(
                                                wells,
                                                base_pred,
                                                post,
                                                tab_metrics,
                                                alpha=float(alpha),
                                                clip=float(clip),
                                                entropy_max=float(entropy_max),
                                                move_max=float(move_max),
                                                top_prob_min=float(args.seg_top_prob_min),
                                                slices=slices,
                                            )
                                            name = f"{key}_a{alpha:g}_c{clip:g}_e{entropy_max:g}_mv{move_max:g}"
                                            rows.append(
                                                {
                                                    **_score_with_cuts(pred, target, slices, name, cut_fracs=_split_csv(args.cut_fracs, float)),
                                                    "kind": "gated_tabular_segmented",
                                                    "segment_len": int(seg_len),
                                                    "max_candidates": int(max_cand),
                                                    "score_weight": float(score_w),
                                                    "geology_beta": float(geo_b),
                                                    "member_tab_weight": float(member_tab_w),
                                                    "alpha": float(alpha),
                                                    "clip": float(clip),
                                                    "entropy_max": float(entropy_max),
                                                    "move_max": float(move_max),
                                                }
                                            )
                                            if len(saved) < int(args.save_top_predictions):
                                                saved[name] = pred.astype(np.float32)

    summary = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return summary, clusters, imps, saved


def write_report(path: Path, summary: pd.DataFrame, clusters: pd.DataFrame, imps: pd.DataFrame, args: argparse.Namespace) -> None:
    cols = [
        "kind",
        "name",
        "rmse",
        "level_resid",
        "swing_resid",
        "shape_corr",
        "top_clusters",
        "member_top",
        "tau_cluster",
        "tau_member",
        "segment_len",
        "max_candidates",
        "score_weight",
        "geology_beta",
        "member_tab_weight",
        "alpha",
        "clip",
        "entropy_max",
        "move_max",
        "top_prob_min",
    ]
    have = [c for c in cols if c in summary.columns]
    lines = [
        "# Candidate Cluster Tabular V1",
        "",
        "Cluster-level tabular posterior. Default feature mode excludes target/oracle, GR/PF/ranker/residual score proxies, and train-only formation-contact candidates.",
        "",
        f"cache_dir=`{args.cache_dir}`",
        f"prefix_features_csv=`{args.prefix_features_csv}`",
        f"allow_formation_contact=`{args.allow_formation_contact}`",
        f"base_npz=`{args.base_npz}` base_col=`{args.base_col}`",
        f"feature_mode=`{args.feature_mode}` model_kinds=`{args.model_kinds}` target_mode=`{args.target_mode}` scheme=`{args.scheme}`",
        f"clusters={len(clusters)}",
        "",
        "## Summary",
        "",
        md_table(summary[have].head(100), floatfmt=".6f"),
    ]
    if not imps.empty:
        imp = imps.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(60)
        lines.extend(["", "## Feature Importances", "", md_table(imp, floatfmt=".6f")])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--prefix-features-csv", type=Path, default=DEFAULT_PREFIX_CSV)
    ap.add_argument("--allow-formation-contact", action="store_true")
    ap.add_argument("--base-npz", type=Path, default=DEFAULT_BASE_NPZ)
    ap.add_argument("--base-mode", choices=["stable2dcnn", "column"], default="column")
    ap.add_argument("--base-col", default="rawpfmean_current_hinge_softgr2")
    ap.add_argument("--shape-weight", type=float, default=0.19305455971004415)
    ap.add_argument("--level-weight", type=float, default=0.31645939520114874)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--cluster-gap", type=float, default=8.0)
    ap.add_argument("--cluster-method", choices=["gap", "single", "span"], default="gap")
    ap.add_argument("--feature-mode", choices=["orthogonal", "truthfree"], default="orthogonal")
    ap.add_argument("--no-family-features", action="store_true")
    ap.add_argument("--scheme", default="spatial")
    ap.add_argument("--model-kinds", default="et,hgb")
    ap.add_argument("--target-mode", choices=["utility", "rmse", "log_rmse"], default="utility")
    ap.add_argument("--label-tau", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--member-prefix-spec", default="interface_strat")
    ap.add_argument("--member-geom-beta", type=float, default=0.0)
    ap.add_argument("--member-score-weight", type=float, default=0.05)
    ap.add_argument("--member-tab-weight", type=float, default=0.0)
    ap.add_argument("--member-tab-weights", default="")
    ap.add_argument("--member-tab-model-kinds", default="et,hgb")
    ap.add_argument("--member-tab-target-mode", choices=["utility", "rmse", "log_rmse"], default="utility")
    ap.add_argument("--member-tab-label-tau", type=float, default=8.0)
    ap.add_argument("--modes", default="posterior,segmented")
    ap.add_argument("--top-clusters", default="2,3")
    ap.add_argument("--member-top", default="1,2")
    ap.add_argument("--tau-cluster", default="0.5,1.0")
    ap.add_argument("--tau-member", default="0.8")
    ap.add_argument("--alpha", default="0.42,0.64")
    ap.add_argument("--clip", default="6,9")
    ap.add_argument("--entropy-max", default="0.7,0.9,1.01")
    ap.add_argument("--move-max", default="12,30")
    ap.add_argument("--top-prob-min", default="0.0,0.45")
    ap.add_argument("--segment-lens", default="64")
    ap.add_argument("--max-candidates-grid", default="24")
    ap.add_argument("--score-weights", default="0.25,0.5")
    ap.add_argument("--geology-betas", default="0.6")
    ap.add_argument("--physics-beta", type=float, default=0.0)
    ap.add_argument("--transition-sigma", type=float, default=2.0)
    ap.add_argument("--jump-sigma", type=float, default=18.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seg-alpha", default="0.64")
    ap.add_argument("--seg-clip", default="9")
    ap.add_argument("--seg-entropy-max", default="0.9,1.01")
    ap.add_argument("--seg-move-max", default="30")
    ap.add_argument("--seg-top-prob-min", type=float, default=0.0)
    ap.add_argument("--cut-fracs", default="0.25,0.5,0.75")
    ap.add_argument("--save-top-predictions", type=int, default=8)
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = ART_DIR / f"{args.run_tag}_{stamp}"
    report = args.report or (ROOT / "reports" / f"{args.run_tag}.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, clusters, imps, saved = evaluate(args)
    summary.to_csv(out_dir / "summary.csv", index=False)
    clusters.to_csv(out_dir / "cluster_table_with_oof_scores.csv", index=False)
    if not imps.empty:
        imps.to_csv(out_dir / "feature_importances.csv", index=False)
    np.savez_compressed(out_dir / "saved_predictions.npz", **saved)
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_report(report, summary, clusters, imps, args)
    print(summary.head(80).to_string(index=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"report={report}", flush=True)


if __name__ == "__main__":
    main()
