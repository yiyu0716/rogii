from __future__ import annotations

import json
import os
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

ID_EXACT = {"well_index", "cluster_id", "candidate_idx", "record_start", "record_len", "fold", "well_id"}
BAD_PARTS = ("target", "truth", "oracle", "selected", "apply", "ranker_pred", "cluster_rel")
SAME_EVIDENCE_PARTS = ("ms_", "score", "posterior", "path_ltr", "seq_resid", "gr_", "_gr", "pf_", "_pf", "ncc", "ancc", "rmse", "mae", "resid")
LEAKY_PARTS = ("formation_contact",)
ORTHOGONAL_PREFIXES = (
    "prefix_",
    "rel_prefix_",
    "mapdiff_prefix_",
    "anchordiff_prefix_",
    "agg_prefix_",
    "family__",
    "agg_family__",
    "cluster_delta",
    "rel_cluster_delta",
    "cluster_n",
    "rel_cluster_n",
)
PREFIX_COST_COLS = [
    "rel_prefix_w16_tvt_start_gap_pct_abs_asc",
    "rel_prefix_w16_tvt_rate_gap_pct_abs_asc",
    "rel_prefix_w16_tvt_kink_abs_pct_abs_asc",
    "rel_prefix_w16_f_start_gap_pct_abs_asc",
    "rel_prefix_w16_f_rate_gap_pct_abs_asc",
    "rel_prefix_w16_f_kink_abs_pct_abs_asc",
    "rel_prefix_w32_tvt_start_gap_pct_abs_asc",
    "rel_prefix_w32_tvt_rate_gap_pct_abs_asc",
    "rel_prefix_w32_tvt_kink_abs_pct_abs_asc",
    "rel_prefix_w32_f_start_gap_pct_abs_asc",
    "rel_prefix_w32_f_rate_gap_pct_abs_asc",
    "rel_prefix_w32_f_kink_abs_pct_abs_asc",
    "rel_prefix_w64_tvt_start_gap_pct_abs_asc",
    "rel_prefix_w64_tvt_rate_gap_pct_abs_asc",
    "rel_prefix_w64_tvt_kink_abs_pct_abs_asc",
    "rel_prefix_w64_f_start_gap_pct_abs_asc",
    "rel_prefix_w64_f_rate_gap_pct_abs_asc",
    "rel_prefix_w64_f_kink_abs_pct_abs_asc",
    "rel_prefix_form_code_jump_pct_abs_asc",
    "rel_prefix_core_code_jump_pct_abs_asc",
    "rel_prefix_core_step_down_pct_abs_asc",
    "rel_prefix_known_to_cand_relpos_gap_pct_abs_asc",
    "rel_prefix_cand_head_form_change_frac_pct_abs_asc",
    "rel_prefix_cand_head_core_step_down_frac_pct_abs_asc",
    "rel_prefix_cand_head_boundary_min_pct_abs_asc",
]


def _ensure_asset_dir(asset_root: Path, name: str) -> Path:
    direct = asset_root / name
    if direct.exists():
        nested = direct / name
        if nested.exists():
            return nested
        return direct
    zpath = asset_root / f"{name}.zip"
    if not zpath.exists():
        return direct
    out_root = Path("/tmp/rogii_sgpath_asset_dirs")
    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_root)
    for cand in [out_root / name, out_root / name / name, out_root]:
        if cand.exists():
            return cand
    return direct


@dataclass
class TypewellSequence:
    tvt: np.ndarray
    label_codes: np.ndarray
    core_codes: np.ndarray


def _row_index_from_id(ids: pd.Series) -> np.ndarray:
    return ids.astype(str).str.rsplit("_", n=1).str[-1].astype(np.int64).to_numpy()


def _interp_to_length(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(length)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(arr) == n:
        return arr
    if len(arr) == 0:
        return np.zeros(n, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(arr), dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.interp(x_new, x_old, arr)


def _path_for_row(row: pd.Series, path_drift: np.ndarray, length: int) -> np.ndarray:
    start = int(row.get("record_start", 0))
    n = int(row.get("record_len", length))
    return _interp_to_length(np.asarray(path_drift[start : start + n], dtype=np.float64), int(length))


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


def _add_family_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "family" not in out.columns:
        return out
    fam = out["family"].astype(str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True).str.lower()
    for value in sorted(fam.dropna().unique().tolist()):
        out[f"family__{value}"] = fam.eq(value).astype(float)
    return out


def _assign_gap_clusters(frame: pd.DataFrame, cluster_gap: float = 8.0) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("well_index", sort=False):
        work = group.copy()
        delta = pd.to_numeric(work.get("mean_level_delta", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
        order = np.argsort(delta, kind="mergesort")
        cid = np.zeros(len(work), dtype=np.int64)
        current = 0
        prev = None
        for pos in order:
            val = float(delta[pos])
            if prev is not None and abs(val - prev) > float(cluster_gap):
                current += 1
            cid[pos] = current
            prev = val
        work["cluster_id"] = cid
        parts.append(work)
    return pd.concat(parts, ignore_index=True) if parts else frame.assign(cluster_id=np.zeros(len(frame), dtype=np.int64))


def _candidate_aggregate_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        low = str(col).lower()
        if low in ID_EXACT or any(part in low for part in BAD_PARTS):
            continue
        if not _allowed_orthogonal_name(str(col)):
            continue
        if _safe_std(frame[col]) > 1e-12:
            cols.append(str(col))
    return cols


def _add_cluster_relative_features(clusters: pd.DataFrame) -> pd.DataFrame:
    out = clusters.copy()
    rel_cols = [
        c
        for c in out.columns
        if pd.api.types.is_numeric_dtype(out[c])
        and (str(c).startswith("cluster_") or str(c).startswith("agg_"))
        and "target" not in str(c).lower()
    ]
    grouped = out.groupby("well_index", sort=False)
    for col in rel_cols:
        x = pd.to_numeric(out[col], errors="coerce")
        mean = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
        std = grouped[col].transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)).replace(0.0, np.nan)
        out[f"rel_{col}_z"] = ((x - mean.astype(float)) / std.astype(float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"rel_{col}_pct_asc"] = x.groupby(out["well_index"], sort=False).rank(method="average", ascending=True, pct=True).astype(float)
        out[f"rel_{col}_pct_desc"] = x.groupby(out["well_index"], sort=False).rank(method="average", ascending=False, pct=True).astype(float)
    return out


def build_cluster_table(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _add_family_indicators(candidates)
    work = _assign_gap_clusters(work, cluster_gap=8.0)
    sort_cols = [c for c in ["well_index", "cluster_id", "score_rank", "candidate_idx"] if c in work.columns]
    rep = work.sort_values(sort_cols, ascending=True).groupby(["well_index", "cluster_id"], sort=False).head(1)
    clusters = rep.set_index(["well_index", "cluster_id"], drop=False).copy()
    keys = [work["well_index"], work["cluster_id"]]
    grouped = work.groupby(["well_index", "cluster_id"], sort=False)
    stats = pd.DataFrame(index=grouped.size().index)
    stats["cluster_n"] = grouped.size().astype(float)
    delta = pd.to_numeric(work.get("mean_level_delta", 0.0), errors="coerce")
    stats["cluster_delta_mean"] = delta.groupby(keys, sort=False).mean()
    stats["cluster_delta_std"] = delta.groupby(keys, sort=False).std().fillna(0.0)
    stats["cluster_delta_min"] = delta.groupby(keys, sort=False).min()
    stats["cluster_delta_max"] = delta.groupby(keys, sort=False).max()
    stats["cluster_delta_span"] = stats["cluster_delta_max"] - stats["cluster_delta_min"]
    stats["cluster_delta_abs_mean"] = delta.abs().groupby(keys, sort=False).mean()
    agg_cols = _candidate_aggregate_columns(work)
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
    clusters = _add_cluster_relative_features(clusters)
    return clusters.replace([np.inf, -np.inf], np.nan), work.replace([np.inf, -np.inf], np.nan)


def _mean_cost(frame: pd.DataFrame, cols: list[str], fallback: float = 0.5) -> np.ndarray:
    have = [c for c in cols if c in frame.columns]
    if not have:
        return np.full(len(frame), float(fallback), dtype=np.float64)
    return frame[have].apply(pd.to_numeric, errors="coerce").fillna(float(fallback)).mean(axis=1).to_numpy(np.float64)


def _patch_sklearn_tree_pickle_compat(obj: object, seen: set[int] | None = None) -> None:
    """Patch sklearn<=1.3 tree estimators so sklearn>=1.4 can predict them.

    Kaggle currently unpickles the selector under a newer sklearn than the
    training environment.  Old ExtraTree/DecisionTree estimators do not carry
    the monotonic_cst attribute that newer sklearn checks at predict time.
    Setting it to None matches the old unconstrained-tree semantics.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    cls_name = obj.__class__.__name__
    cls_mod = getattr(obj.__class__, "__module__", "")
    if cls_mod.startswith("sklearn.tree.") and cls_name.endswith(("TreeRegressor", "TreeClassifier")):
        if not hasattr(obj, "monotonic_cst"):
            setattr(obj, "monotonic_cst", None)

    for attr in ("steps", "estimators_", "transformers_", "named_steps"):
        if not hasattr(obj, attr):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(value, dict):
            iterable = value.values()
        else:
            iterable = value
        try:
            for item in iterable:
                child = item[-1] if isinstance(item, tuple) and item else item
                _patch_sklearn_tree_pickle_compat(child, seen)
        except TypeError:
            _patch_sklearn_tree_pickle_compat(value, seen)


def score_clusters_with_selector(clusters: pd.DataFrame, asset_root: Path) -> pd.DataFrame:
    feature_cols = json.loads((asset_root / "selector_feature_columns.json").read_text(encoding="utf-8"))
    x = clusters.reindex(columns=feature_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    preds = []
    selector_model_dir = _ensure_asset_dir(asset_root, "sg_selector_models")
    for model_path in sorted(selector_model_dir.glob("*.joblib")):
        try:
            model = joblib.load(model_path)
            _patch_sklearn_tree_pickle_compat(model)
            p = np.asarray(model.predict(x), dtype=np.float64)
        except Exception as exc:
            print(f"[sg-selector-skip] {model_path.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        preds.append(p)
        print(f"[sg-selector] {model_path.name} mean={float(np.mean(p)):.6f} std={float(np.std(p)):.6f}", flush=True)
    lgb_dir = _ensure_asset_dir(asset_root, "sg_selector_lgb")
    for model_path in sorted(lgb_dir.glob("*.txt")):
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_path))
        p = np.asarray(booster.predict(x.astype(np.float32)), dtype=np.float64)
        preds.append(p)
        print(f"[sg-selector] {model_path.name} mean={float(np.mean(p)):.6f} std={float(np.std(p)):.6f}", flush=True)
    if not preds:
        raise FileNotFoundError(f"No usable selector models under {asset_root}")
    out = clusters.copy()
    out["tab_score"] = np.mean(np.stack(preds, axis=0), axis=0)
    return out


def _softmax(values: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x
    x = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
    z = x / max(float(tau), 1e-9)
    z -= float(np.max(z))
    w = np.exp(np.clip(z, -80.0, 0.0))
    return w / max(float(np.sum(w)), 1e-12)


def _entropy_norm(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0.0]
    if len(w) <= 1:
        return 0.0
    ent = -float(np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
    return float(np.clip(ent / np.log(float(len(w))), 0.0, 1.0))


def _robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=np.float64)
    med = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - med)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return np.nan_to_num((arr - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mx = np.nanmax(arr, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(np.clip(arr - mx, -80.0, 0.0)), axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(out.reshape(()), dtype=np.float64)
    return np.squeeze(out, axis=axis)


def _forward_backward_segments(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    emit = np.asarray(emissions, dtype=np.float64)
    n_seg, n_state = emit.shape
    if n_seg == 0 or n_state == 0:
        return np.zeros_like(emit)
    trans = np.asarray(transitions, dtype=np.float64)
    if trans.ndim == 2:
        trans = np.repeat(trans[None, :, :], max(n_seg - 1, 0), axis=0)
    alpha = np.empty_like(emit)
    beta = np.zeros_like(emit)
    alpha[0] = emit[0]
    alpha[0] -= float(_logsumexp(alpha[0]))
    for t in range(1, n_seg):
        alpha[t] = emit[t] + _logsumexp(alpha[t - 1][:, None] + trans[t - 1], axis=0)
        alpha[t] -= float(_logsumexp(alpha[t]))
    for t in range(n_seg - 2, -1, -1):
        beta[t] = _logsumexp(trans[t] + emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)
        beta[t] -= float(_logsumexp(beta[t]))
    log_post = alpha + beta
    log_post -= _logsumexp(log_post, axis=1)[:, None]
    post = np.exp(np.clip(log_post, -80.0, 0.0))
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
    return post


def _segment_bounds(length: int, segment_len: int) -> list[tuple[int, int]]:
    return [(s, min(s + int(segment_len), int(length))) for s in range(0, int(length), max(1, int(segment_len)))]


def _rank01_cost(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    order = np.argsort(np.nan_to_num(arr, nan=np.nanmedian(arr)), kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(arr), dtype=np.float64)
    return ranks


def make_typewell_sequence(tw: pd.DataFrame) -> TypewellSequence:
    if tw.empty or "TVT" not in tw.columns:
        return TypewellSequence(np.zeros(0), np.zeros(0), np.zeros(0))
    seq = tw.dropna(subset=["TVT"]).sort_values("TVT").reset_index(drop=True)
    tvt = seq["TVT"].to_numpy(np.float64)
    labels = seq.get("Geology", pd.Series([""] * len(seq))).astype(str).to_numpy()
    label_map = {name: i for i, name in enumerate(pd.unique(labels))}
    core_order = {name: i for i, name in enumerate(("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"))}
    label_codes = np.asarray([label_map.get(x, -1) for x in labels], dtype=np.float64)
    core_codes = np.asarray([core_order.get(x, -1) for x in labels], dtype=np.float64)
    return TypewellSequence(tvt=tvt, label_codes=label_codes, core_codes=core_codes)


def _seq_index(seq: TypewellSequence, tvt: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(seq.tvt, np.asarray(tvt, dtype=np.float64), side="left")
    return np.clip(idx, 0, max(len(seq.tvt) - 1, 0))


def _segment_geology_cost(paths: np.ndarray, bounds: list[tuple[int, int]], last_known: float, seq: TypewellSequence | None) -> np.ndarray:
    m = int(paths.shape[0])
    costs = np.zeros((len(bounds), m), dtype=np.float64)
    if seq is None or len(seq.tvt) == 0:
        return costs
    tvt = float(last_known) + np.asarray(paths, dtype=np.float64)
    for si, (start, end) in enumerate(bounds):
        vals = tvt[:, start:end]
        raw = np.zeros(m, dtype=np.float64)
        for j in range(m):
            idx = _seq_index(seq, vals[j])
            form = seq.label_codes[idx].astype(np.float64)
            core = seq.core_codes[idx].astype(np.float64)
            core_valid = core[core >= 0]
            core_d = np.diff(core_valid) if len(core_valid) > 1 else np.asarray([], dtype=np.float64)
            runs = 1 + int(np.sum(np.diff(form) != 0.0)) if len(form) else 0
            unique = len(np.unique(form)) if len(form) else 0
            revisit = float(max(runs - unique, 0) / max(runs, 1))
            step_down = float(np.mean(core_d < 0.0)) if len(core_d) else 0.0
            raw[j] = 0.70 * step_down + 0.30 * revisit
        costs[si] = _rank01_cost(raw)
    return costs


def _transition_tensor(paths: np.ndarray, bounds: list[tuple[int, int]], transition_sigma: float, jump_sigma: float) -> np.ndarray:
    m = int(paths.shape[0])
    if len(bounds) <= 1:
        return np.zeros((0, m, m), dtype=np.float64)
    arr = np.asarray(paths, dtype=np.float64)
    sig = max(float(transition_sigma), 1e-6)
    jsig = max(float(jump_sigma), 1e-6)
    trans = np.empty((len(bounds) - 1, m, m), dtype=np.float64)
    for si in range(len(bounds) - 1):
        p0, p1 = bounds[si], bounds[si + 1]
        prev_end = arr[:, max(p0[1] - 1, p0[0])]
        next_start = arr[:, p1[0]]
        prev_mean = arr[:, p0[0] : p0[1]].mean(axis=1)
        next_mean = arr[:, p1[0] : p1[1]].mean(axis=1)
        continuity = np.abs(prev_end[:, None] - next_start[None, :]) / sig
        jump = np.abs(prev_mean[:, None] - next_mean[None, :]) / jsig
        trans[si] = -(continuity + 0.35 * jump)
    return trans


def segmented_posterior_path(paths: np.ndarray, scores: np.ndarray, seq: TypewellSequence | None, last_known: float, *, segment_len: int, score_weight: float, geology_beta: float, transition_sigma: float, jump_sigma: float, temperature: float) -> np.ndarray:
    path_arr = np.asarray(paths, dtype=np.float64)
    bounds = _segment_bounds(path_arr.shape[1], int(segment_len))
    emit = np.repeat((float(score_weight) * _robust_z(scores))[None, :], len(bounds), axis=0)
    if float(geology_beta) != 0.0:
        emit -= float(geology_beta) * _segment_geology_cost(path_arr, bounds, float(last_known), seq)
    emit = emit / max(float(temperature), 1e-6)
    trans = _transition_tensor(path_arr, bounds, float(transition_sigma), float(jump_sigma))
    posterior = _forward_backward_segments(emit, trans)
    pred = np.zeros(path_arr.shape[1], dtype=np.float64)
    for si, (start, end) in enumerate(bounds):
        pred[start:end] = posterior[si] @ path_arr[:, start:end]
    return pred


def apply_ramped_move(base: np.ndarray, cand: np.ndarray, *, alpha: float, clip: float) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float64)
    cand_arr = _interp_to_length(np.asarray(cand, dtype=np.float64), len(base_arr))
    diff = cand_arr - base_arr
    bound = abs(float(clip))
    if np.isfinite(bound) and bound > 0.0:
        diff = np.clip(diff, -bound, bound)
    tau = max(80.0, 0.12 * float(len(base_arr)))
    ramp = 1.0 - np.exp(-np.arange(len(base_arr), dtype=np.float64) / tau)
    return base_arr + float(alpha) * diff * ramp


def _candidate_records_one_well(livevp, data_root: Path, well_index: int, wid: str, grp0: pd.DataFrame):
    grp = grp0.sort_values("_row_idx").reset_index(drop=True)
    hw = pd.read_csv(data_root / "test" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv")
    row_idx = grp["_row_idx"].to_numpy(np.int64)
    known = pd.to_numeric(hw["TVT_input"], errors="coerce").dropna()
    last_known = float(known.iloc[-1]) if len(known) else float(grp["tvt"].iloc[0])
    meta = {
        "well_index": int(well_index),
        "well_id": str(wid),
        "n_hidden": int(len(grp)),
        "last_known": float(last_known),
        "row_start": int(row_idx[0]) if len(row_idx) else 0,
        "row_len": int(len(grp)),
    }
    records, arrays = livevp.candidate_records_with_arrays_for_well(str(wid), grp, data_root=data_root, hw=hw, tw=tw)
    records = records.copy()
    records["well_index"] = int(well_index)
    if "well_id" in records.columns:
        records = records.drop(columns=["well_id"])
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    return meta, records, arrays


def build_candidate_pool(test_df: pd.DataFrame, data_root: Path, base_drift: np.ndarray):
    import live_v2_candidate_generator as livevp

    source = test_df[["id", "well", "last_tvt"]].copy()
    source["_row_idx"] = _row_index_from_id(source["id"])
    source["tvt"] = pd.to_numeric(source["last_tvt"], errors="coerce").fillna(0.0).to_numpy(np.float64) + np.asarray(base_drift, dtype=np.float64)
    wells = [(i, str(wid), grp.copy()) for i, (wid, grp) in enumerate(source.groupby("well", sort=False))]
    n_jobs = int(os.environ.get("ROGII_SGPATH_CAND_WORKERS", "2") or "2")
    print(f"[sg-candidates] wells={len(wells)} workers={n_jobs}", flush=True)
    parts = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_candidate_records_one_well)(livevp, data_root, wi, wid, grp) for wi, wid, grp in wells
    )
    well_rows = []
    record_rows = []
    arrays_acc = {"path_drift": [], "cal_gr": [], "gr_residual": []}
    offsets = {key: 0 for key in arrays_acc}
    for meta, records, arrays in parts:
        well_rows.append(meta)
        records = records.copy()
        if len(records):
            records["record_start"] = pd.to_numeric(records["record_start"], errors="coerce").fillna(0).astype(int) + offsets["path_drift"]
            record_rows.append(records)
        for key in arrays_acc:
            arr = np.asarray(arrays.get(key, np.zeros(0, dtype=np.float32)), dtype=np.float32)
            arrays_acc[key].append(arr)
            offsets[key] += int(len(arr))
    candidates = pd.concat(record_rows, ignore_index=True) if record_rows else pd.DataFrame()
    well_index = pd.DataFrame(well_rows)
    arrays_out = {
        key: (np.concatenate(vals).astype(np.float32) if vals else np.zeros(0, dtype=np.float32))
        for key, vals in arrays_acc.items()
    }
    print(f"[sg-candidates] rows={len(candidates)} path_values={len(arrays_out['path_drift'])}", flush=True)
    if len(candidates) and "family" in candidates:
        print(f"[sg-candidates] family_counts={candidates['family'].astype(str).value_counts().to_dict()}", flush=True)
    return candidates, well_index, arrays_out


def _well_slices(wells: np.ndarray) -> dict[str, tuple[int, int]]:
    out = {}
    start = 0
    arr = np.asarray(wells).astype(str)
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[end] == arr[start]:
            end += 1
        out[str(arr[start])] = (start, end)
        start = end
    return out


def generate_sg_proposal(test_df: pd.DataFrame, data_root: Path, asset_root: Path, base_drift: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    candidates, well_index, arrays = build_candidate_pool(test_df, data_root, base_drift)
    if candidates.empty:
        raise RuntimeError("sg_path candidate pool is empty")
    clusters, clustered = build_cluster_table(candidates)
    clusters = score_clusters_with_selector(clusters, asset_root)
    scored = clustered.merge(clusters[["well_index", "cluster_id", "tab_score"]], on=["well_index", "cluster_id"], how="left")
    scored["tab_score"] = pd.to_numeric(scored["tab_score"], errors="coerce").fillna(float(clusters["tab_score"].median()))
    scored["member_score"] = -_mean_cost(scored, PREFIX_COST_COLS, fallback=0.5)
    scored["vp_score"] = scored["tab_score"].to_numpy(np.float64) + 0.05 * scored["member_score"].to_numpy(np.float64)
    slices = _well_slices(test_df["well"].astype(str).to_numpy())
    out = np.asarray(base_drift, dtype=np.float64).copy()
    meta_rows = []
    meta_by_wi = well_index.set_index("well_index", drop=False)
    for wi, group in scored.groupby("well_index", sort=True):
        if int(wi) not in meta_by_wi.index:
            continue
        meta = meta_by_wi.loc[int(wi)]
        wid = str(meta["well_id"])
        if wid not in slices:
            continue
        start, end = slices[wid]
        n = int(end - start)
        keep = group.sort_values(["vp_score", "score_rank", "candidate_idx"], ascending=[False, True, True]).head(24)
        if len(keep) < 2:
            continue
        paths = np.asarray([_path_for_row(row, arrays["path_drift"], n) for _, row in keep.iterrows()], dtype=np.float64)
        scores = pd.to_numeric(keep["vp_score"], errors="coerce").fillna(-1e9).to_numpy(np.float64)
        try:
            tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv")
            seq = make_typewell_sequence(tw)
        except Exception:
            seq = None
        posterior_path = segmented_posterior_path(
            paths,
            scores,
            seq,
            float(meta["last_known"]),
            segment_len=64,
            score_weight=0.15,
            geology_beta=0.4,
            transition_sigma=2.0,
            jump_sigma=18.0,
            temperature=1.0,
        )
        ramped = apply_ramped_move(base_drift[start:end], posterior_path, alpha=0.64, clip=12.0)
        move = float(np.mean(np.abs(ramped - base_drift[start:end])))
        top_scores = np.sort(scores)[::-1][: min(8, len(scores))]
        weights = _softmax(top_scores, 1.0)
        entropy = _entropy_norm(weights)
        top_prob = float(weights[0]) if len(weights) else 1.0
        if entropy <= 1.01 and move <= 30.0:
            out[start:end] = ramped
        meta_rows.append(
            {
                "well": wid,
                "n_candidates": int(len(group)),
                "n_keep": int(len(keep)),
                "move_mae": move,
                "cluster_entropy_norm": entropy,
                "top_cluster_prob": top_prob,
                "applied": float(entropy <= 1.01 and move <= 30.0),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    return out.astype(np.float32), meta_df


def build_sg_path_features(test_df: pd.DataFrame, data_root: Path, asset_root: Path, *, base_source: str = "pf_ancc_d") -> pd.DataFrame:
    if base_source in test_df.columns:
        base = pd.to_numeric(test_df[base_source], errors="coerce").fillna(0.0).to_numpy(np.float32)
    elif "pf_ancc_delta" in test_df.columns:
        base = pd.to_numeric(test_df["pf_ancc_delta"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        base_source = "pf_ancc_delta"
    else:
        base = np.zeros(len(test_df), dtype=np.float32)
        base_source = "zero"
    prop, meta = generate_sg_proposal(test_df, data_root, asset_root, base)
    move = (prop - base).astype(np.float32)
    abs_move = np.abs(move).astype(np.float32)
    frac = pd.to_numeric(test_df.get("frac", pd.Series(np.zeros(len(test_df)))), errors="coerce").fillna(0.0).to_numpy(np.float32)
    md_since = pd.to_numeric(test_df.get("md_since", pd.Series(np.zeros(len(test_df)))), errors="coerce").fillna(0.0).to_numpy(np.float32)
    zeros = np.zeros(len(test_df), dtype=np.float32)

    def col(name: str) -> np.ndarray:
        return pd.to_numeric(test_df.get(name, pd.Series(zeros)), errors="coerce").fillna(0.0).to_numpy(np.float32)

    out = pd.DataFrame(
        {
            "sg_base_d": base,
            "sg_prop_d": prop,
            "sg_move_d": move,
            "sg_abs_move_d": abs_move,
            "sg_move_x_frac": move * frac,
            "sg_abs_move_x_frac": abs_move * frac,
            "sg_abs_move_x_mdsqrt": abs_move * np.sqrt(np.maximum(md_since, 0.0)).astype(np.float32),
            "sg_prop_minus_ext50": prop - col("ext_50"),
            "sg_prop_minus_ext200": prop - col("ext_200"),
            "sg_prop_minus_extall": prop - col("ext_all"),
            "sg_prop_minus_pf_ancc": prop - col("pf_ancc_d"),
            "sg_prop_minus_beam_mean": prop - col("beam_mean_d"),
            "sg_prop_minus_sc_ens": prop - col("sc_ens_d"),
            "sg_prop_minus_cal_proj": prop - col("cal_proj_d"),
            "sg_prop_minus_noc_proj": prop - col("noc_proj_d"),
        }
    )
    stack = np.vstack([prop, col("pf_ancc_d"), col("beam_mean_d"), col("beam_med_d"), col("sc_ens_d"), col("cal_proj_d"), col("noc_proj_d")])
    out["sg_candidate_disagree_std"] = np.nanstd(stack, axis=0).astype(np.float32)
    out["sg_candidate_disagree_range"] = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    print(
        f"[sg-audit] base_source={base_source} sg_prop_mean={float(out['sg_prop_d'].mean()):.6f} "
        f"sg_move_abs_mean={float(out['sg_abs_move_d'].mean()):.6f} sg_move_abs_std={float(out['sg_abs_move_d'].std()):.6f}",
        flush=True,
    )
    if not meta.empty:
        print(
            f"[sg-audit] wells={len(meta)} applied={int(meta['applied'].sum())} "
            f"move_mae_mean={float(meta['move_mae'].mean()):.6f} n_candidates_mean={float(meta['n_candidates'].mean()):.2f}",
            flush=True,
        )
    if float(out["sg_abs_move_d"].mean()) <= 1e-6:
        raise RuntimeError("sg_path proposal is degenerate: mean absolute move is zero")
    return out
