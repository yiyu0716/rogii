#!/usr/bin/env python3
"""Analyze leave-query-out geo-neighbor conditions against a saved OOF geo path."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from seq_NN_geo_prior import (
    POST_TRAIN_GEO_DIAGNOSTIC_NAMES,
    make_geo_prior_for_wells,
    resolve_geo_prior_cfg,
)
from seq_NN_train import make_cv_splits, make_geo_stratified_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF_PATH = PROJECT_ROOT / "outputs/submit_ver_0719_V1_seq_nn/oof_df.pqt"
DEFAULT_BAD_QUANTILE = 0.90

RISK_FEATURES = (
    "geo_nbr_distance_mean",
    "geo_nbr_distance_q90",
    "geo_radial_extrap_score_mean",
    "geo_radial_extrap_score_q90",
    "geo_nbr_path_misalignment_mean",
    "geo_nbr_path_misalignment_q90",
)

SUMMARY_STATS = ("mean", "std", "q10", "q50", "q75", "q90", "q95", "min", "max")
SUMMARY_FAMILIES = (
    "geo_nbr_distance",
    "geo_radial_extrap_score",
    "geo_nbr_path_misalignment",
)
ALL_SUMMARY_FEATURES = tuple(
    f"{family}_{stat}"
    for family in SUMMARY_FAMILIES
    for stat in SUMMARY_STATS
)

LOGISTIC_FEATURE_GROUPS = {
    "logistic_distance": (0, 1),
    "logistic_radial": (2, 3),
    "logistic_misalignment": (4, 5),
    "logistic_distance_radial": (0, 1, 2, 3),
    "logistic_distance_misalignment": (0, 1, 4, 5),
    "logistic_all": (0, 1, 2, 3, 4, 5),
}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_cfg(cfg_path: Path):
    with cfg_path.open("rb") as handle:
        return pickle.load(handle)


def _read_oof(oof_path: Path) -> pd.DataFrame:
    columns = ["well_id", "submit_index", "TVT", "geo_prior_TVT", "TVT_pred"]
    oof = pd.read_parquet(oof_path, columns=columns)
    if oof.duplicated(["well_id", "submit_index"]).any():
        raise ValueError(f"OOF has duplicate (well_id, submit_index) rows: {oof_path}")
    if not np.isfinite(oof[["TVT", "geo_prior_TVT", "TVT_pred"]].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"OOF target/prediction columns contain non-finite values: {oof_path}")
    oof["well_id"] = oof["well_id"].astype("category")
    return oof


def _validate_cached_points(oof: pd.DataFrame, point_df: pd.DataFrame):
    if len(point_df) != len(oof):
        raise ValueError(f"cached point row count mismatch: cache={len(point_df):,}, oof={len(oof):,}")
    required = {"well_id", "submit_index", *POST_TRAIN_GEO_DIAGNOSTIC_NAMES}
    missing = sorted(required - set(point_df.columns))
    if missing:
        raise ValueError(f"cached point diagnostics missing columns: {missing}")
    cached_well = point_df["well_id"].astype(str).to_numpy()
    oof_well = oof["well_id"].astype(str).to_numpy()
    if not np.array_equal(cached_well, oof_well):
        raise ValueError("cached point diagnostic well order does not match OOF")
    if not np.array_equal(
        point_df["submit_index"].to_numpy(dtype=np.int64),
        oof["submit_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("cached point diagnostic submit_index order does not match OOF")


def make_or_load_point_diagnostics(
    oof: pd.DataFrame,
    train_path: Path,
    geo_cfg,
    output_dir: Path,
    *,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    point_path = output_dir / "point_diagnostics.pqt"
    construction_path = output_dir / "construction_summary.json"
    if point_path.exists() and not force:
        point_df = pd.read_parquet(point_path)
        _validate_cached_points(oof, point_df)
        construction = {}
        if construction_path.exists():
            with construction_path.open("r", encoding="utf-8") as handle:
                construction = json.load(handle)
        construction["reused_point_diagnostics"] = True
        return point_df, construction

    well_ids = oof["well_id"].astype(str).drop_duplicates().tolist()
    started = time.perf_counter()
    prior, prior_summary = make_geo_prior_for_wells(
        support_path=train_path,
        query_path=train_path,
        support_wells=well_ids,
        query_wells=well_ids,
        cfg=geo_cfg,
        exclude_query_from_support=True,
    )

    group_indices = oof.groupby("well_id", sort=False, observed=True).indices
    diagnostic_values = {
        name: np.empty(len(oof), dtype=np.float32)
        for name in POST_TRAIN_GEO_DIAGNOSTIC_NAMES
    }
    for well_id, positions in group_indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        submit_index = oof.iloc[positions]["submit_index"].to_numpy(dtype=np.int64)
        prior_item = prior[str(well_id)]
        for name in POST_TRAIN_GEO_DIAGNOSTIC_NAMES:
            diagnostic_values[name][positions] = np.asarray(
                getattr(prior_item, name),
                dtype=np.float32,
            )[submit_index]

    point_df = pd.DataFrame(
        {
            "well_id": pd.Categorical(oof["well_id"].astype(str)),
            "submit_index": oof["submit_index"].to_numpy(dtype=np.int32),
            **diagnostic_values,
        }
    )
    values = point_df[list(POST_TRAIN_GEO_DIAGNOSTIC_NAMES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("constructed point diagnostics contain non-finite values")
    alignment = point_df["geo_nbr_path_alignment"].to_numpy(dtype=np.float64)
    if np.any((alignment < 0.0) | (alignment > 1.0)):
        raise ValueError("geo_nbr_path_alignment falls outside [0, 1]")
    point_df.to_parquet(point_path, index=False)

    construction = {
        "elapsed_sec": float(time.perf_counter() - started),
        "exclude_query_from_support": True,
        "own_visible_prefix_support_retained": True,
        "well_count": len(well_ids),
        "point_rows": len(point_df),
        "point_diagnostics_path": str(point_path),
        "geo_prior_summary": prior_summary,
        "geo_prior_config": geo_cfg.to_dict(),
        "reused_point_diagnostics": False,
    }
    _write_json(construction_path, construction)
    return point_df, construction


def _array_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def aggregate_wells(oof: pd.DataFrame, point_df: pd.DataFrame, bad_quantile: float) -> pd.DataFrame:
    _validate_cached_points(oof, point_df)
    group_indices = oof.groupby("well_id", sort=False, observed=True).indices
    rows = []
    for well_id, positions in group_indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        submit_index = oof.iloc[positions]["submit_index"].to_numpy(dtype=np.int64)
        order = np.argsort(submit_index)
        target = oof.iloc[positions]["TVT"].to_numpy(dtype=np.float64)
        geo_pred = oof.iloc[positions]["geo_prior_TVT"].to_numpy(dtype=np.float64)
        nn_pred = oof.iloc[positions]["TVT_pred"].to_numpy(dtype=np.float64)
        geo_error = geo_pred - target
        nn_error = nn_pred - target
        distance = point_df.iloc[positions]["geo_nbr_distance"].to_numpy(dtype=np.float64)
        radial = point_df.iloc[positions]["geo_radial_extrap_score"].to_numpy(dtype=np.float64)
        alignment = point_df.iloc[positions]["geo_nbr_path_alignment"].to_numpy(dtype=np.float64)
        misalignment = 1.0 - alignment
        row = {
            "well_id": str(well_id),
            "rows": int(len(positions)),
            "geo_sse": float(np.dot(geo_error, geo_error)),
            "geo_sae": float(np.abs(geo_error).sum()),
            "geo_bias": float(np.mean(geo_error)),
            "geo_end_abs_error": float(abs(geo_error[order[-1]])),
            "nn_sse": float(np.dot(nn_error, nn_error)),
        }
        row.update(_array_stats(distance, "geo_nbr_distance"))
        row.update(_array_stats(radial, "geo_radial_extrap_score"))
        row.update(_array_stats(alignment, "geo_nbr_path_alignment"))
        row.update(_array_stats(misalignment, "geo_nbr_path_misalignment"))
        rows.append(row)

    well_df = pd.DataFrame(rows)
    well_df["geo_rmse"] = np.sqrt(well_df["geo_sse"] / well_df["rows"])
    well_df["geo_mae"] = well_df["geo_sae"] / well_df["rows"]
    well_df["nn_rmse"] = np.sqrt(well_df["nn_sse"] / well_df["rows"])
    well_df["geo_minus_nn_rmse"] = well_df["geo_rmse"] - well_df["nn_rmse"]
    threshold = float(well_df["geo_rmse"].quantile(float(bad_quantile)))
    well_df["is_bad_geo"] = well_df["geo_rmse"] >= threshold
    return well_df.sort_values("well_id").reset_index(drop=True)


def _pooled_rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(frame["geo_sse"].sum() / frame["rows"].sum()))


def analyze_features(well_df: pd.DataFrame) -> pd.DataFrame:
    target = well_df["is_bad_geo"].to_numpy(dtype=bool)
    rows = []
    for feature in ALL_SUMMARY_FEATURES:
        values = well_df[feature].to_numpy(dtype=np.float64)
        q25, q75, q90 = np.quantile(values, [0.25, 0.75, 0.90])
        low = well_df[values <= q25]
        high = well_df[values >= q75]
        top = values >= q90
        rho = (
            np.nan
            if np.ptp(values) == 0.0
            else spearmanr(values, well_df["geo_rmse"].to_numpy(dtype=np.float64)).statistic
        )
        rows.append(
            {
                "feature": feature,
                "spearman_with_geo_rmse": float(rho),
                "bad_roc_auc": float(roc_auc_score(target, values)),
                "bad_average_precision": float(average_precision_score(target, values)),
                "low_quartile_max": float(q25),
                "high_quartile_min": float(q75),
                "low_quartile_pooled_geo_rmse": _pooled_rmse(low),
                "high_quartile_pooled_geo_rmse": _pooled_rmse(high),
                "high_minus_low_pooled_geo_rmse": _pooled_rmse(high) - _pooled_rmse(low),
                "overall_bad_rate": float(target.mean()),
                "high_quartile_bad_rate": float(high["is_bad_geo"].mean()),
                "top_decile_bad_rate": float(target[top].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("bad_average_precision", ascending=False).reset_index(drop=True)


def analyze_interactions(well_df: pd.DataFrame) -> pd.DataFrame:
    features = {
        "distance": "geo_nbr_distance_mean",
        "radial_extrapolation": "geo_radial_extrap_score_q90",
        "path_misalignment": "geo_nbr_path_misalignment_q90",
    }
    rows = []
    names = list(features)
    for left_pos, left_name in enumerate(names):
        for right_name in names[left_pos + 1 :]:
            left_col = features[left_name]
            right_col = features[right_name]
            left_high = well_df[left_col] >= well_df[left_col].quantile(0.75)
            right_high = well_df[right_col] >= well_df[right_col].quantile(0.75)
            for left_value, right_value in ((False, False), (True, False), (False, True), (True, True)):
                mask = (left_high == left_value) & (right_high == right_value)
                part = well_df[mask]
                rows.append(
                    {
                        "left_feature": left_name,
                        "right_feature": right_name,
                        "left_high_quartile": bool(left_value),
                        "right_high_quartile": bool(right_value),
                        "well_count": int(len(part)),
                        "bad_count": int(part["is_bad_geo"].sum()),
                        "bad_rate": float(part["is_bad_geo"].mean()),
                        "pooled_geo_rmse": _pooled_rmse(part),
                    }
                )
    return pd.DataFrame(rows)


def _model_matrix(well_df: pd.DataFrame) -> np.ndarray:
    values = well_df[list(RISK_FEATURES)].to_numpy(dtype=np.float64)
    values[:, :4] = np.log1p(values[:, :4])
    if not np.isfinite(values).all():
        raise ValueError("model feature matrix contains non-finite values")
    return values


def _all_summary_matrix(well_df: pd.DataFrame) -> np.ndarray:
    values = well_df[list(ALL_SUMMARY_FEATURES)].to_numpy(dtype=np.float64)
    for column, feature in enumerate(ALL_SUMMARY_FEATURES):
        if feature.startswith("geo_nbr_distance") or feature.startswith("geo_radial_extrap_score"):
            values[:, column] = np.log1p(values[:, column])
    if not np.isfinite(values).all():
        raise ValueError("all-summary model feature matrix contains non-finite values")
    return values


def _rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    positive_count = int(labels.sum())
    top_positive = int(labels[order[:positive_count]].sum())
    top_20_count = int(math.ceil(0.20 * len(labels)))
    top_25_count = int(math.ceil(0.25 * len(labels)))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "positive_count": positive_count,
        "captured_at_bad_count": top_positive,
        "precision_at_bad_count": float(top_positive / positive_count),
        "recall_in_top_20pct": float(labels[order[:top_20_count]].sum() / positive_count),
        "recall_in_top_25pct": float(labels[order[:top_25_count]].sum() / positive_count),
    }


def cross_validated_detection(well_df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    well_ids = well_df["well_id"].to_numpy(dtype=str)
    labels = well_df["is_bad_geo"].to_numpy(dtype=bool)
    matrix = _model_matrix(well_df)
    splits = make_cv_splits(well_ids, cfg, log=lambda message: print(message, flush=True))
    score_sum = {
        model_name: np.zeros(len(well_df), dtype=np.float64)
        for model_name in (*LOGISTIC_FEATURE_GROUPS, "tree_depth2")
    }
    score_count = np.zeros(len(well_df), dtype=np.int64)
    fold_rows = []
    folds_per_repeat = len(splits) // int(getattr(cfg, "cv_repeats", 1))
    for split_index, (train_index, val_index) in enumerate(splits):
        models = {
            model_name: (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
                np.asarray(feature_indices, dtype=np.int64),
            )
            for model_name, feature_indices in LOGISTIC_FEATURE_GROUPS.items()
        }
        models["tree_depth2"] = (
            DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=20260724 + split_index,
            ),
            np.arange(matrix.shape[1]),
        )
        repeat = split_index // folds_per_repeat
        fold = split_index % folds_per_repeat
        for model_name, (model, feature_index) in models.items():
            model.fit(matrix[train_index][:, feature_index], labels[train_index])
            score = model.predict_proba(matrix[val_index][:, feature_index])[:, 1]
            score_sum[model_name][val_index] += score
            metrics = _rank_metrics(labels[val_index], score)
            fold_rows.append(
                {
                    "model": model_name,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "train_wells": int(len(train_index)),
                    "validation_wells": int(len(val_index)),
                    **metrics,
                }
            )
        score_count[val_index] += 1
    expected_repeats = int(getattr(cfg, "cv_repeats", 1))
    if not np.all(score_count == expected_repeats):
        raise ValueError(
            f"unexpected repeated-CV prediction counts: expected={expected_repeats}, "
            f"observed={np.unique(score_count).tolist()}"
        )

    predictions = well_df[
        ["well_id", "is_bad_geo", "geo_rmse", "nn_rmse", "geo_minus_nn_rmse", *RISK_FEATURES]
    ].copy()
    metrics_rows = []
    risk_percentiles = []
    for feature in (
        "geo_nbr_distance_q90",
        "geo_radial_extrap_score_q90",
        "geo_nbr_path_misalignment_q90",
    ):
        risk_percentiles.append(well_df[feature].rank(pct=True).to_numpy(dtype=np.float64))
    predictions["equal_rank_score"] = np.mean(risk_percentiles, axis=0)
    metrics_rows.append({"model": "equal_rank_score", **_rank_metrics(labels, predictions["equal_rank_score"])})
    for model_name, values in score_sum.items():
        score = values / score_count
        predictions[f"{model_name}_score"] = score
        metrics_rows.append({"model": model_name, **_rank_metrics(labels, score)})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
    return predictions, metrics_df, pd.DataFrame(fold_rows)


def nested_all_summary_detection(
    well_df: pd.DataFrame,
    cfg,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    model_name = "logistic_all_summaries_nested"
    matrix = _all_summary_matrix(well_df)
    well_ids = well_df["well_id"].to_numpy(dtype=str)
    labels = well_df["is_bad_geo"].to_numpy(dtype=bool)
    outer_splits = make_cv_splits(well_ids, cfg, log=lambda message: None)
    c_candidates = (0.03, 0.10, 0.30, 1.00)
    score_sum = np.zeros(len(well_df), dtype=np.float64)
    score_count = np.zeros(len(well_df), dtype=np.int64)
    fold_rows = []
    folds_per_repeat = len(outer_splits) // int(getattr(cfg, "cv_repeats", 1))
    for split_index, (train_index, val_index) in enumerate(outer_splits):
        inner_splits = make_geo_stratified_folds(
            well_ids[train_index],
            cfg,
            log=lambda message: None,
            split_seed=1000 + split_index,
        )
        best_c = None
        best_inner_ap = -np.inf
        for c_value in c_candidates:
            inner_score = np.zeros(len(train_index), dtype=np.float64)
            inner_count = np.zeros(len(train_index), dtype=np.int64)
            for inner_train, inner_val in inner_splits:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(c_value),
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                    ),
                )
                model.fit(matrix[train_index][inner_train], labels[train_index][inner_train])
                inner_score[inner_val] += model.predict_proba(matrix[train_index][inner_val])[:, 1]
                inner_count[inner_val] += 1
            inner_ap = average_precision_score(labels[train_index], inner_score / inner_count)
            if inner_ap > best_inner_ap or (
                np.isclose(inner_ap, best_inner_ap) and (best_c is None or c_value < best_c)
            ):
                best_inner_ap = float(inner_ap)
                best_c = float(c_value)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=best_c,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
            ),
        )
        model.fit(matrix[train_index], labels[train_index])
        score = model.predict_proba(matrix[val_index])[:, 1]
        score_sum[val_index] += score
        score_count[val_index] += 1
        repeat = split_index // folds_per_repeat
        fold = split_index % folds_per_repeat
        fold_rows.append(
            {
                "model": model_name,
                "repeat": int(repeat),
                "fold": int(fold),
                "train_wells": int(len(train_index)),
                "validation_wells": int(len(val_index)),
                "selected_c": best_c,
                "inner_average_precision": best_inner_ap,
                **_rank_metrics(labels[val_index], score),
            }
        )
    expected_repeats = int(getattr(cfg, "cv_repeats", 1))
    if not np.all(score_count == expected_repeats):
        raise ValueError("nested all-summary detector has incomplete outer OOF predictions")
    scores = score_sum / score_count
    metrics = {"model": model_name, **_rank_metrics(labels, scores)}
    return scores, metrics, pd.DataFrame(fold_rows)


def make_score_deciles(predictions: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        score_col = f"{model_name}_score" if model_name != "equal_rank_score" else model_name
        order = predictions[score_col].rank(method="first", pct=True)
        decile = np.minimum((order.to_numpy() * 10.0).astype(int), 9)
        for value in range(10):
            part = predictions[decile == value]
            rows.append(
                {
                    "model": model_name,
                    "risk_decile": int(value + 1),
                    "well_count": int(len(part)),
                    "bad_count": int(part["is_bad_geo"].sum()),
                    "bad_rate": float(part["is_bad_geo"].mean()),
                    "mean_geo_rmse": float(part["geo_rmse"].mean()),
                    "median_geo_rmse": float(part["geo_rmse"].median()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_detection_metrics(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    samples: int = 5000,
) -> pd.DataFrame:
    score_col = model_name if model_name == "equal_rank_score" else f"{model_name}_score"
    labels = predictions["is_bad_geo"].to_numpy(dtype=bool)
    scores = predictions[score_col].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(20260724)
    values = []
    for _ in range(int(samples)):
        index = rng.integers(0, len(predictions), size=len(predictions))
        sampled_labels = labels[index]
        if sampled_labels.all() or (~sampled_labels).all():
            continue
        values.append(
            (
                roc_auc_score(sampled_labels, scores[index]),
                average_precision_score(sampled_labels, scores[index]),
            )
        )
    values = np.asarray(values, dtype=np.float64)
    estimates = {
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
    }
    rows = []
    for column, metric in enumerate(("roc_auc", "average_precision")):
        low, median, high = np.quantile(values[:, column], [0.025, 0.50, 0.975])
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "estimate": float(estimates[metric]),
                "bootstrap_median": float(median),
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_samples": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def analyze_bad_threshold_sensitivity(well_df: pd.DataFrame, cfg) -> pd.DataFrame:
    matrix = _model_matrix(well_df)
    well_ids = well_df["well_id"].to_numpy(dtype=str)
    splits = make_cv_splits(well_ids, cfg, log=lambda message: None)
    rows = []
    for bad_quantile in (0.80, 0.85, 0.90, 0.95):
        threshold = float(well_df["geo_rmse"].quantile(bad_quantile))
        labels = well_df["geo_rmse"].to_numpy(dtype=np.float64) >= threshold
        score_sum = np.zeros(len(well_df), dtype=np.float64)
        score_count = np.zeros(len(well_df), dtype=np.int64)
        for train_index, val_index in splits:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            )
            model.fit(matrix[train_index], labels[train_index])
            score_sum[val_index] += model.predict_proba(matrix[val_index])[:, 1]
            score_count[val_index] += 1
        scores = score_sum / score_count
        rows.append(
            {
                "bad_quantile": float(bad_quantile),
                "bad_threshold": threshold,
                "base_bad_rate": float(labels.mean()),
                **_rank_metrics(labels, scores),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    oof_path: Path,
    construction: dict,
    well_df: pd.DataFrame,
    associations: pd.DataFrame,
    interactions: pd.DataFrame,
    predictions: pd.DataFrame,
    model_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    bad_quantile: float,
):
    bad_threshold = float(well_df["geo_rmse"].quantile(bad_quantile))
    best = model_metrics.iloc[0]
    strongest = associations.iloc[0]
    high_interactions = interactions[
        interactions["left_high_quartile"] & interactions["right_high_quartile"]
    ]
    strongest_interaction = high_interactions.sort_values("bad_rate", ascending=False).iloc[0]
    bad_percent = 100.0 * (1.0 - bad_quantile)
    auc_bootstrap = bootstrap_metrics[bootstrap_metrics["metric"] == "roc_auc"].iloc[0]
    ap_bootstrap = bootstrap_metrics[bootstrap_metrics["metric"] == "average_precision"].iloc[0]
    severe = threshold_sensitivity.loc[
        np.isclose(threshold_sensitivity["bad_quantile"], 0.95)
    ].iloc[0]
    best_folds = fold_metrics[fold_metrics["model"] == best["model"]]
    score_col = f"{best['model']}_score"
    bad_risk = predictions[predictions["is_bad_geo"]].copy()
    lowest_ranked_bad = bad_risk.nsmallest(1, score_col).iloc[0]
    lowest_ranked_bad_rank = int(
        predictions[score_col].rank(method="min", ascending=False).loc[lowest_ranked_bad.name]
    )
    report = f"""# Leave-Query-Out Geo-Condition Analysis

## Contract

- OOF source: `{oof_path}`
- Query suffix support: excluded from all 773 support-well pools
- Query visible-prefix support: retained because it is inference-available
- Point rows: {int(construction.get('point_rows', 0)):,}
- Saved fold-out `geo_prior_TVT` row RMSE: {math.sqrt(well_df['geo_sse'].sum() / well_df['rows'].sum()):.6f} ft
- Leave-query-out reconstructed prior row RMSE:
  {construction.get('geo_prior_summary', {}).get('rmse', float('nan')):.6f} ft
- Bad-well label: worst {bad_percent:g}% by saved geo RMSE, threshold {bad_threshold:.6f} ft,
  count {int(well_df['is_bad_geo'].sum())}

## Strongest Marginal Signal

`{strongest['feature']}` has Spearman `{strongest['spearman_with_geo_rmse']:.4f}`,
ROC AUC `{strongest['bad_roc_auc']:.4f}`, average precision
`{strongest['bad_average_precision']:.4f}`, and top-decile bad rate
`{strongest['top_decile_bad_rate']:.3f}`.

The strongest high-quartile interaction is
`{strongest_interaction['left_feature']} + {strongest_interaction['right_feature']}`:
bad-well prevalence is `{strongest_interaction['bad_rate']:.3f}` across
`{int(strongest_interaction['well_count'])}` wells, with pooled geo RMSE
`{strongest_interaction['pooled_geo_rmse']:.3f}` ft.

## Combined Detection

The best exploratory score is `{best['model']}` with repeated geographic OOF ROC
AUC `{best['roc_auc']:.4f}`, average precision `{best['average_precision']:.4f}`,
and `{int(best['captured_at_bad_count'])}/{int(best['positive_count'])}` bad wells
captured in the top risk set of the same size. Its recall is
`{best['recall_in_top_20pct']:.3f}` in the top 20% and
`{best['recall_in_top_25pct']:.3f}` in the top 25%.

Well-bootstrap 95% intervals are
`[{auc_bootstrap['ci_low']:.3f}, {auc_bootstrap['ci_high']:.3f}]` for ROC AUC and
`[{ap_bootstrap['ci_low']:.3f}, {ap_bootstrap['ci_high']:.3f}]` for average
precision. Its fold average precision ranges from
`{best_folds['average_precision'].min():.3f}` to
`{best_folds['average_precision'].max():.3f}`. For the worst 5% label, the fixed
six-summary logistic control has ROC AUC `{severe['roc_auc']:.3f}`, but only
`{int(severe['captured_at_bad_count'])}/{int(severe['positive_count'])}` extreme
wells are captured in a risk set of the same size.

The lowest-ranked bad well is `{lowest_ranked_bad['well_id']}` with geo RMSE
`{lowest_ranked_bad['geo_rmse']:.3f}` ft at risk rank
`{lowest_ranked_bad_rank}/{len(predictions)}`. This is direct evidence that local
support geometry cannot detect every incompatible geological path.

## Interpretation

These features measure local support geometry, not geological regime compatibility.
Treat them as useful risk enrichment only if the cross-validated ranking materially
exceeds the 10% base bad rate; do not use them as a hard geo fallback gate without
a separately nested candidate-mixture validation. The leave-query-out support set
is test-like and intentionally differs from the smaller repeated fold-training
support sets that produced the saved OOF geo path.
"""
    path.write_text(report, encoding="utf-8")


def run(args) -> dict:
    oof_path = args.oof_path.resolve()
    cfg_path = (args.cfg_path or (oof_path.parent / "cfg.pkl")).resolve()
    output_dir = (args.output_dir or (oof_path.parent / "geo_neighbor_condition_loo")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(cfg_path)
    geo_cfg = resolve_geo_prior_cfg(cfg)
    train_path = (args.data_dir.resolve() / "train") if args.data_dir is not None else Path(cfg.train_path)
    oof = _read_oof(oof_path)
    print(f"OOF rows={len(oof):,}, wells={oof['well_id'].nunique():,}", flush=True)
    print(f"geo config={geo_cfg.to_dict()}", flush=True)

    point_df, construction = make_or_load_point_diagnostics(
        oof,
        train_path,
        geo_cfg,
        output_dir,
        force=bool(args.force),
    )
    print(
        f"point diagnostics rows={len(point_df):,}, reused={construction.get('reused_point_diagnostics')}",
        flush=True,
    )
    well_df = aggregate_wells(oof, point_df, args.bad_quantile)
    associations = analyze_features(well_df)
    interactions = analyze_interactions(well_df)
    predictions, model_metrics, fold_metrics = cross_validated_detection(well_df, cfg)
    nested_scores, nested_metrics, nested_fold_metrics = nested_all_summary_detection(well_df, cfg)
    predictions[f"{nested_metrics['model']}_score"] = nested_scores
    model_metrics = pd.concat(
        [model_metrics, pd.DataFrame([nested_metrics])],
        ignore_index=True,
    ).sort_values("average_precision", ascending=False).reset_index(drop=True)
    fold_metrics = pd.concat([fold_metrics, nested_fold_metrics], ignore_index=True, sort=False)
    model_names = model_metrics["model"].tolist()
    deciles = make_score_deciles(predictions, model_names)
    best_model = str(model_metrics.iloc[0]["model"])
    bootstrap_metrics = bootstrap_detection_metrics(predictions, model_name=best_model)
    threshold_sensitivity = analyze_bad_threshold_sensitivity(well_df, cfg)
    best_score_col = best_model if best_model == "equal_rank_score" else f"{best_model}_score"
    top_risk = predictions.sort_values(best_score_col, ascending=False).reset_index(drop=True)
    top_risk.insert(0, "risk_rank", np.arange(1, len(top_risk) + 1))
    risk_rank = predictions[best_score_col].rank(method="min", ascending=False).astype(int)
    predictions.insert(1, "best_risk_rank", risk_rank)
    worst_geo = predictions.sort_values("geo_rmse", ascending=False).reset_index(drop=True)
    worst_geo.insert(0, "geo_error_rank", np.arange(1, len(worst_geo) + 1))

    well_df.to_csv(output_dir / "well_metrics.csv", index=False)
    associations.to_csv(output_dir / "feature_associations.csv", index=False)
    interactions.to_csv(output_dir / "risk_interactions.csv", index=False)
    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    deciles.to_csv(output_dir / "score_deciles.csv", index=False)
    bootstrap_metrics.to_csv(output_dir / "bootstrap_metrics.csv", index=False)
    threshold_sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    top_risk.head(100).to_csv(output_dir / "top100_risk_wells.csv", index=False)
    worst_geo.head(100).to_csv(output_dir / "worst100_geo_wells.csv", index=False)
    write_report(
        output_dir / "REPORT.md",
        oof_path,
        construction,
        well_df,
        associations,
        interactions,
        predictions,
        model_metrics,
        fold_metrics,
        bootstrap_metrics,
        threshold_sensitivity,
        float(args.bad_quantile),
    )

    summary = {
        "oof_path": str(oof_path),
        "cfg_path": str(cfg_path),
        "output_dir": str(output_dir),
        "bad_quantile": float(args.bad_quantile),
        "bad_threshold": float(well_df["geo_rmse"].quantile(float(args.bad_quantile))),
        "bad_well_count": int(well_df["is_bad_geo"].sum()),
        "saved_geo_row_rmse": _pooled_rmse(well_df),
        "saved_nn_row_rmse": float(np.sqrt(well_df["nn_sse"].sum() / well_df["rows"].sum())),
        "construction": construction,
        "strongest_association": associations.iloc[0].to_dict(),
        "risk_interactions": interactions.to_dict(orient="records"),
        "model_metrics": model_metrics.to_dict(orient="records"),
        "bootstrap_metrics": bootstrap_metrics.to_dict(orient="records"),
        "threshold_sensitivity": threshold_sensitivity.to_dict(orient="records"),
        "best_model": best_model,
    }
    _write_json(output_dir / "summary.json", summary)
    print(model_metrics.to_string(index=False), flush=True)
    print(f"saved analysis: {output_dir}", flush=True)
    return summary


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--cfg-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bad-quantile", type=float, default=DEFAULT_BAD_QUANTILE)
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = make_arg_parser().parse_args()
    if not 0.0 < float(args.bad_quantile) < 1.0:
        raise ValueError(f"bad_quantile must be in (0, 1), got {args.bad_quantile}")
    run(args)


if __name__ == "__main__":
    main()
