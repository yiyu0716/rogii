from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sg_path_online_lib import (
    PREFIX_COST_COLS,
    _entropy_norm,
    _mean_cost,
    _path_for_row,
    _softmax,
    _well_slices,
    apply_ramped_move,
    build_candidate_pool,
    build_cluster_table,
    make_typewell_sequence,
    score_clusters_with_selector,
    segmented_posterior_path,
)


MEMBER_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
MEMBER_WEIGHT_NAMES = {
    0.0: "mtw0",
    0.25: "mtw0p25",
    0.5: "mtw0p5",
    1.0: "mtw1",
    2.0: "mtw2",
    4.0: "mtw4",
}
FAMILIES = ("anchor", "level_grid", "piecewise_u", "prefix_u_rate", "selfgr")


def _timer(label: str, start: float) -> float:
    now = time.time()
    print(f"[timing] {label}={now - start:.2f}s", flush=True)
    return now


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df.get(name, pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).to_numpy(np.float32)


def _cluster_metrics(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for wi, group in clusters.groupby("well_index", sort=False):
        vals = pd.to_numeric(group["tab_score"], errors="coerce").fillna(-1e9).to_numpy(np.float64)
        if len(vals) == 0:
            continue
        order = np.argsort(-vals)
        weights = _softmax(vals[order[: min(8, len(order))]], 1.0)
        rows.append(
            {
                "well_index": int(wi),
                "cluster_entropy_norm": _entropy_norm(weights),
                "top_cluster_prob": float(weights[0]) if len(weights) else 1.0,
                "cluster_margin": float(vals[order[0]] - vals[order[1]]) if len(order) > 1 else 99.0,
            }
        )
    return pd.DataFrame(rows)


def _add_member_selector_columns(scored: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    cluster_ctx = clusters[["well_index", "cluster_id", "tab_score"]].copy()
    cluster_ctx["cluster_rank_by_tab"] = cluster_ctx.groupby("well_index")["tab_score"].rank(method="first", ascending=False)
    cluster_ctx["n_clusters"] = cluster_ctx.groupby("well_index")["cluster_id"].transform("count").astype(float)
    out = out.merge(cluster_ctx, on=["well_index", "cluster_id"], how="left", suffixes=("", "_clusterctx"))
    if "tab_score_clusterctx" in out.columns:
        out["tab_score"] = pd.to_numeric(out.get("tab_score"), errors="coerce")
        out["tab_score"] = out["tab_score"].fillna(pd.to_numeric(out["tab_score_clusterctx"], errors="coerce"))
        out = out.drop(columns=["tab_score_clusterctx"])
    for col in ("tab_score", "cluster_rank_by_tab", "n_clusters"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    prefix_score = -_mean_cost(out, PREFIX_COST_COLS, fallback=0.5)
    out["member_score"] = prefix_score.astype(np.float32)
    out["prefix_member_score"] = prefix_score.astype(np.float32)

    fam = out.get("family", pd.Series([""] * len(out))).astype(str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True).str.lower()
    for name in FAMILIES:
        out[f"family_{name}"] = fam.eq(name).astype(np.float32)
    return out


def score_member_candidates(scored: pd.DataFrame, asset_root: Path) -> pd.DataFrame:
    import lightgbm as lgb

    feature_cols = json.loads((asset_root / "member_selector_feature_columns.json").read_text(encoding="utf-8"))
    x = scored.reindex(columns=feature_cols, fill_value=0.0).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    booster = lgb.Booster(model_file=str(asset_root / "member_selector_online_common_lgb.txt"))
    pred = np.asarray(booster.predict(x), dtype=np.float32)
    out = scored.copy()
    out["member_lgb_oof"] = pred
    out["member_tab_score"] = pred
    print(
        f"[member-selector] rows={len(out)} features={len(feature_cols)} "
        f"mean={float(np.nanmean(pred)):.6f} std={float(np.nanstd(pred)):.6f}",
        flush=True,
    )
    return out


def _sg_proposal_from_scored(
    test_df: pd.DataFrame,
    data_root: Path,
    scored: pd.DataFrame,
    well_index: pd.DataFrame,
    path_drift: np.ndarray,
    base: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    slices = _well_slices(test_df["well"].astype(str).to_numpy())
    out = np.asarray(base, dtype=np.float64).copy()
    meta_rows: list[dict[str, float | int | str]] = []
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
        paths = np.asarray([_path_for_row(row, path_drift, n) for _, row in keep.iterrows()], dtype=np.float64)
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
        ramped = apply_ramped_move(base[start:end], posterior_path, alpha=0.64, clip=12.0)
        move = float(np.mean(np.abs(ramped - base[start:end])))
        top_scores = np.sort(scores)[::-1][: min(8, len(scores))]
        weights = _softmax(top_scores, 1.0)
        entropy = _entropy_norm(weights)
        top_prob = float(weights[0]) if len(weights) else 1.0
        applied = bool(entropy <= 1.01 and move <= 30.0)
        if applied:
            out[start:end] = ramped
        meta_rows.append(
            {
                "well": wid,
                "n_candidates": int(len(group)),
                "n_keep": int(len(keep)),
                "move_mae": move,
                "cluster_entropy_norm": entropy,
                "top_cluster_prob": top_prob,
                "applied": float(applied),
            }
        )
    return out.astype(np.float32), pd.DataFrame(meta_rows)


def _member_paths_from_scored(
    test_df: pd.DataFrame,
    data_root: Path,
    scored: pd.DataFrame,
    clusters: pd.DataFrame,
    well_index: pd.DataFrame,
    path_drift: np.ndarray,
    base: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    slices = _well_slices(test_df["well"].astype(str).to_numpy())
    metrics = _cluster_metrics(clusters).set_index("well_index", drop=False)
    meta_by_wi = well_index.set_index("well_index", drop=False)
    paths_out = {MEMBER_WEIGHT_NAMES[w]: np.asarray(base, dtype=np.float64).copy() for w in MEMBER_WEIGHTS}
    meta_rows: list[dict[str, float | int | str]] = []
    for wi, group0 in scored.groupby("well_index", sort=True):
        if int(wi) not in meta_by_wi.index:
            continue
        meta = meta_by_wi.loc[int(wi)]
        wid = str(meta["well_id"])
        if wid not in slices:
            continue
        start, end = slices[wid]
        n = int(end - start)
        if n <= 0:
            continue
        entropy = float(metrics.loc[int(wi), "cluster_entropy_norm"]) if int(wi) in metrics.index else 1.0
        top_prob = float(metrics.loc[int(wi), "top_cluster_prob"]) if int(wi) in metrics.index else 0.0
        try:
            tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv")
            seq = make_typewell_sequence(tw)
        except Exception:
            seq = None
        for weight in MEMBER_WEIGHTS:
            name = MEMBER_WEIGHT_NAMES[weight]
            group = group0.copy()
            group["vp_score"] = (
                pd.to_numeric(group["tab_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
                + 0.05 * pd.to_numeric(group["member_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
                + float(weight) * pd.to_numeric(group["member_tab_score"], errors="coerce").fillna(0.0).to_numpy(np.float64)
            )
            keep = group.sort_values(["vp_score", "score_rank", "candidate_idx"], ascending=[False, True, True]).head(24)
            if len(keep) < 2:
                continue
            cand_paths = np.asarray([_path_for_row(row, path_drift, n) for _, row in keep.iterrows()], dtype=np.float64)
            scores = pd.to_numeric(keep["vp_score"], errors="coerce").fillna(-1e9).to_numpy(np.float64)
            posterior_path = segmented_posterior_path(
                cand_paths,
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
            direct_move = float(np.mean(np.abs(posterior_path - base[start:end])))
            if entropy <= 1.01 and top_prob >= 0.0 and direct_move <= 30.0:
                paths_out[name][start:end] = apply_ramped_move(base[start:end], posterior_path, alpha=0.64, clip=12.0)
        meta_rows.append(
            {
                "well": wid,
                "n_candidates": int(len(group0)),
                "cluster_entropy_norm": entropy,
                "top_cluster_prob": top_prob,
            }
        )
    return {k: v.astype(np.float32) for k, v in paths_out.items()}, pd.DataFrame(meta_rows)


def _sg_feature_frame(test_df: pd.DataFrame, base: np.ndarray, prop: np.ndarray, meta: pd.DataFrame, *, base_source: str) -> pd.DataFrame:
    move = (prop - base).astype(np.float32)
    abs_move = np.abs(move).astype(np.float32)
    frac = _col(test_df, "frac")
    md_since = _col(test_df, "md_since")
    out = pd.DataFrame(
        {
            "sg_base_d": base,
            "sg_prop_d": prop,
            "sg_move_d": move,
            "sg_abs_move_d": abs_move,
            "sg_move_x_frac": move * frac,
            "sg_abs_move_x_frac": abs_move * frac,
            "sg_abs_move_x_mdsqrt": abs_move * np.sqrt(np.maximum(md_since, 0.0)).astype(np.float32),
            "sg_prop_minus_ext50": prop - _col(test_df, "ext_50"),
            "sg_prop_minus_ext200": prop - _col(test_df, "ext_200"),
            "sg_prop_minus_extall": prop - _col(test_df, "ext_all"),
            "sg_prop_minus_pf_ancc": prop - _col(test_df, "pf_ancc_d"),
            "sg_prop_minus_beam_mean": prop - _col(test_df, "beam_mean_d"),
            "sg_prop_minus_sc_ens": prop - _col(test_df, "sc_ens_d"),
            "sg_prop_minus_cal_proj": prop - _col(test_df, "cal_proj_d"),
            "sg_prop_minus_noc_proj": prop - _col(test_df, "noc_proj_d"),
        }
    )
    stack = np.vstack([prop, _col(test_df, "pf_ancc_d"), _col(test_df, "beam_mean_d"), _col(test_df, "beam_med_d"), _col(test_df, "sc_ens_d"), _col(test_df, "cal_proj_d"), _col(test_df, "noc_proj_d")])
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
    return out


def _member_feature_frame(test_df: pd.DataFrame, base: np.ndarray, paths: dict[str, np.ndarray], sg_side: pd.DataFrame) -> pd.DataFrame:
    best = paths["mtw4"]
    stack = np.vstack([paths[k] for k in sorted(paths)])
    mean_path = np.nanmean(stack, axis=0).astype(np.float32)
    std_path = np.nanstd(stack, axis=0).astype(np.float32)
    range_path = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)).astype(np.float32)
    frac = _col(test_df, "frac")
    md_since = _col(test_df, "md_since")
    sg_prop = sg_side["sg_prop_d"].to_numpy(np.float32)
    out = pd.DataFrame(
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
            "memb_best_minus_ext50": best - _col(test_df, "ext_50"),
            "memb_best_minus_ext200": best - _col(test_df, "ext_200"),
            "memb_best_minus_pf_ancc": best - _col(test_df, "pf_ancc_d"),
            "memb_best_minus_beam_mean": best - _col(test_df, "beam_mean_d"),
            "memb_best_minus_sc_ens": best - _col(test_df, "sc_ens_d"),
            "memb_best_minus_cal_proj": best - _col(test_df, "cal_proj_d"),
            "memb_best_minus_noc_proj": best - _col(test_df, "noc_proj_d"),
            "memb_abs_best_move_x_frac": np.abs(best - base) * frac,
            "memb_abs_best_sg_gap_x_frac": np.abs(best - sg_prop) * frac,
            "memb_prop_std_x_frac": std_path * frac,
            "memb_prop_range_x_frac": range_path * frac,
            "memb_abs_best_move_x_mdsqrt": np.abs(best - base) * np.sqrt(np.maximum(md_since, 0.0)).astype(np.float32),
        }
    )
    for name in ("mtw0", "mtw0p25", "mtw0p5", "mtw1", "mtw2", "mtw4"):
        out[f"memb_path_{name}_d"] = paths[name]
    candidate_stack = np.vstack(
        [
            best,
            mean_path,
            sg_prop,
            _col(test_df, "pf_ancc_d"),
            _col(test_df, "beam_mean_d"),
            _col(test_df, "beam_med_d"),
            _col(test_df, "sc_ens_d"),
            _col(test_df, "cal_proj_d"),
            _col(test_df, "noc_proj_d"),
        ]
    )
    out["memb_candidate_disagree_std"] = np.nanstd(candidate_stack, axis=0).astype(np.float32)
    out["memb_candidate_disagree_range"] = (np.nanmax(candidate_stack, axis=0) - np.nanmin(candidate_stack, axis=0)).astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    print(
        f"[member-audit] best_mean={float(out['memb_prop_best_d'].mean()):.6f} "
        f"best_abs_move_mean={float(np.abs(best - base).mean()):.6f} "
        f"path_std_mean={float(out['memb_prop_std_d'].mean()):.6f}",
        flush=True,
    )
    return out


def build_sg_and_member_path_features(
    test_df: pd.DataFrame,
    data_root: Path,
    asset_root: Path,
    *,
    base_source: str = "strong_base_d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = time.time()
    if base_source in test_df.columns:
        base = pd.to_numeric(test_df[base_source], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        base = np.zeros(len(test_df), dtype=np.float32)
        base_source = "zero"

    candidates, well_index, arrays = build_candidate_pool(test_df, data_root, base)
    t = _timer("candidate_pool", t)
    if candidates.empty:
        raise RuntimeError("candidate pool is empty")

    clusters, clustered = build_cluster_table(candidates)
    clusters = score_clusters_with_selector(clusters, asset_root)
    t = _timer("cluster_selector", t)

    scored = clustered.merge(clusters[["well_index", "cluster_id", "tab_score"]], on=["well_index", "cluster_id"], how="left")
    scored["tab_score"] = pd.to_numeric(scored["tab_score"], errors="coerce").fillna(float(clusters["tab_score"].median()))
    scored = _add_member_selector_columns(scored, clusters)
    scored = score_member_candidates(scored, asset_root)
    t = _timer("member_selector", t)

    scored["vp_score"] = scored["tab_score"].to_numpy(np.float64) + 0.05 * scored["member_score"].to_numpy(np.float64)
    sg_prop, sg_meta = _sg_proposal_from_scored(test_df, data_root, scored, well_index, arrays["path_drift"], base)
    sg_side = _sg_feature_frame(test_df, base, sg_prop, sg_meta, base_source=base_source)
    t = _timer("sg_proposal", t)

    member_paths, member_meta = _member_paths_from_scored(test_df, data_root, scored, clusters, well_index, arrays["path_drift"], base)
    member_side = _member_feature_frame(test_df, base, member_paths, sg_side)
    if not member_meta.empty:
        print(
            f"[member-audit] wells={len(member_meta)} n_candidates_mean={float(member_meta['n_candidates'].mean()):.2f} "
            f"entropy_mean={float(member_meta['cluster_entropy_norm'].mean()):.6f}",
            flush=True,
        )
    _timer("member_paths", t)
    return sg_side, member_side
