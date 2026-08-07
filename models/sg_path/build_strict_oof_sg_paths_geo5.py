#!/usr/bin/env python3
"""Build fold-pure SG-path OOF sidecars for WARP, HMM, and GSN.

Each validation fold gets:
  * a formation-surface prior fit on the other four folds only;
  * candidates generated from that arm's OOF drift;
  * ET and LightGBM cluster-selector scores trained on the other folds only;
  * the production SG decoder (diversity=90, posterior KEEP=24).

Hidden TVT is used only to label training-fold candidates and to report OOF CV.
It is never exposed to validation candidate generation or selector features.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/root/rogii")
CV = ROOT / "cv"
ASSET = ROOT / "datasets_upload/rogii-online-common-memberpath-assets-v001"
FOLD_CSV = ROOT / "OOF/geo_kmeans_5fold.csv"
FULL249_EXP = ROOT / "OOF/full249_geo5fold_20260723_094041"
LEGACY_XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
LEGACY_PSEUDO = (
    ROOT
    / "cv/artifacts/public_train19_lgb5fold_20260709"
    / "sgpath_member_full249_anchor_v1/pseudo_comp_root"
)
OUT = ROOT / "OOF/sg_path_strict_oof_div90_geo5_20260723"

ARM_PATHS = {
    "warp": ROOT / "OOF/warp_exp207/oof_drift.npy",
    "hmm": ROOT / "OOF/hmm_exp224/oof.npy",
    "gs": ROOT / "OOF/OOF_exp265/gsn_d.npy",
}

DIVERSITY = 90
KEEP = 24
SG_COLS = [
    "sg_base_d", "sg_prop_d", "sg_move_d", "sg_abs_move_d",
    "sg_move_x_frac", "sg_abs_move_x_frac", "sg_abs_move_x_mdsqrt",
    "sg_prop_minus_ext50", "sg_prop_minus_ext200", "sg_prop_minus_extall",
    "sg_prop_minus_pf_ancc", "sg_prop_minus_beam_mean", "sg_prop_minus_sc_ens",
    "sg_prop_minus_cal_proj", "sg_prop_minus_noc_proj",
    "sg_candidate_disagree_std", "sg_candidate_disagree_range",
]
FRAME_COLS = [
    "id", "well", "target", "last_tvt", "frac", "md_since",
    "ext_all", "ext_200", "ext_50", "pf_ancc_d", "beam_mean_d",
    "beam_med_d", "sc_ens_d", "cal_proj_d", "noc_proj_d",
]

for p in (CV, ASSET):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import candidate_cluster_tabular_v1 as cct  # noqa: E402
import live_v2_candidate_generator as livevp  # noqa: E402
import sg_path_online_lib as sg  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def fold_map() -> dict[str, int]:
    f = pd.read_csv(FOLD_CSV, dtype={"well_id": str})
    if len(f) != 773 or f["well_id"].nunique() != 773 or set(f["fold"]) != set(range(5)):
        raise RuntimeError("invalid geo-kmeans fold map")
    return dict(zip(f["well_id"].astype(str), f["fold"].astype(int)))


def prepare_frame(force: bool = False) -> Path:
    path = OUT / "shared/canonical_fold_honest_frame.parquet"
    if path.exists() and not force:
        return path
    parts = []
    for fold in range(5):
        p = FULL249_EXP / f"features/fold_{fold}/Xy_val.parquet"
        d = pd.read_parquet(p, columns=FRAME_COLS)
        d["fold"] = np.int8(fold)
        parts.append(d)
    frame = pd.concat(parts, ignore_index=True)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    canonical = pd.read_parquet(LEGACY_XY, columns=["id"])
    canonical["id"] = canonical["id"].astype(str)
    pos = pd.Series(np.arange(len(canonical), dtype=np.int64), index=canonical["id"])
    frame["_pos"] = frame["id"].map(pos)
    if frame["_pos"].isna().any() or frame["id"].duplicated().any() or len(frame) != len(canonical):
        raise RuntimeError("fold-honest frame does not match canonical IDs")
    frame = frame.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)
    if not frame["id"].equals(canonical["id"]):
        raise RuntimeError("canonical ID order mismatch")
    fmap = fold_map()
    expected = frame["well"].map(fmap).to_numpy(np.int8)
    if not np.array_equal(expected, frame["fold"].to_numpy(np.int8)):
        raise RuntimeError("frame fold assignment mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    (OUT / "shared/frame_meta.json").write_text(
        json.dumps(
            {
                "rows": len(frame),
                "wells": int(frame["well"].nunique()),
                "id_order_hash": stable_hash(frame["id"].tolist()),
                "fold_csv": str(FOLD_CSV),
                "fold_csv_sha256": sha256_file(FOLD_CSV),
                "source": "concat full249 fold-specific Xy_val.parquet, canonical ID reorder",
            },
            indent=2,
        )
        + "\n"
    )
    return path


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_roots(force: bool = False) -> None:
    fmap = fold_map()
    src_train = ROOT / "datasets/rogii-wellbore-geology-prediction/train"
    src_test = LEGACY_PSEUDO / "test"
    if not src_test.exists():
        raise FileNotFoundError(src_test)
    for fold in range(5):
        root = OUT / f"pseudo_roots/fold_{fold}"
        marker = root / "manifest.json"
        if marker.exists() and not force:
            continue
        if root.exists():
            shutil.rmtree(root)
        (root / "train").mkdir(parents=True)
        (root / "test").mkdir(parents=True)
        train_wells = sorted(w for w, k in fmap.items() if k != fold)
        val_wells = sorted(w for w, k in fmap.items() if k == fold)
        for wid in train_wells:
            for suffix in ("__horizontal_well.csv", "__typewell.csv"):
                _link_or_copy(src_train / f"{wid}{suffix}", root / "train" / f"{wid}{suffix}")
        for wid in val_wells:
            for suffix in ("__horizontal_well.csv", "__typewell.csv"):
                _link_or_copy(src_test / f"{wid}{suffix}", root / "test" / f"{wid}{suffix}")
        marker.write_text(
            json.dumps(
                {
                    "fold": fold,
                    "n_train_wells": len(train_wells),
                    "n_test_wells": len(val_wells),
                    "train_well_hash": stable_hash(train_wells),
                    "test_well_hash": stable_hash(val_wells),
                    "test_columns": ["MD", "X", "Y", "Z", "GR", "TVT_input"],
                    "protocol": "outer-train-only surface prior; validation official-visible test files",
                },
                indent=2,
            )
            + "\n"
        )
    prepare_frame(force=force)


def _candidate_worker(task):
    wi, wid, grp, data_root = task
    return sg._candidate_records_one_well(livevp, Path(data_root), wi, wid, grp)


def build_candidate_pool_fork(frame: pd.DataFrame, data_root: Path, base: np.ndarray, workers: int):
    source = frame[["id", "well", "last_tvt"]].copy()
    source["_row_idx"] = sg._row_index_from_id(source["id"])
    source["tvt"] = pd.to_numeric(source["last_tvt"], errors="coerce").fillna(0.0).to_numpy(np.float64) + base
    wells = [(i, str(w), g.copy(), str(data_root)) for i, (w, g) in enumerate(source.groupby("well", sort=False))]
    prior = livevp._build_train_space_surface_prior(data_root, samples_per_well=160)
    livevp._SURFACE_PRIOR_CACHE[str(data_root)] = prior
    ctx = get_context("fork")
    parts = [None] * len(wells)
    with ProcessPoolExecutor(max_workers=min(workers, len(wells)), mp_context=ctx) as ex:
        futures = {ex.submit(_candidate_worker, task): i for i, task in enumerate(wells)}
        for done, fut in enumerate(as_completed(futures), 1):
            parts[futures[fut]] = fut.result()
            if done % 25 == 0 or done == len(wells):
                print(f"[candidates] {data_root.name} {done}/{len(wells)}", flush=True)
    well_rows, record_rows = [], []
    arrays_acc = {"path_drift": [], "cal_gr": [], "gr_residual": []}
    offsets = {k: 0 for k in arrays_acc}
    for meta, records, arrays in parts:
        well_rows.append(meta)
        records = records.copy()
        if len(records):
            records["record_start"] = pd.to_numeric(records["record_start"], errors="coerce").fillna(0).astype(int) + offsets["path_drift"]
            record_rows.append(records)
        for key in arrays_acc:
            arr = np.asarray(arrays.get(key, np.zeros(0)), dtype=np.float32)
            arrays_acc[key].append(arr)
            offsets[key] += len(arr)
    return (
        pd.concat(record_rows, ignore_index=True),
        pd.DataFrame(well_rows),
        {k: np.concatenate(v).astype(np.float32) for k, v in arrays_acc.items()},
    )


def configure_generator(workers: int) -> None:
    os.environ["ROGII_SGPATH_CAND_WORKERS"] = str(workers)
    os.environ["ROGII_SGPATH_DIVERSE_TOPK"] = str(DIVERSITY)
    os.environ["ROGII_SURFACE_PRIOR_EXCLUDE_SELF"] = "1"
    livevp.VP_DIVERSE_TOPK = DIVERSITY
    livevp.VP_DIVERSE_SCORE_HEAD = 12
    livevp.VP_DIVERSE_FAMILY_QUOTA = 6
    livevp.VP_DIVERSE_DELTA_BIN = 6.0

    def skip_surface_costs(records, paths, hw, surface_tops, row_idx):
        return records

    livevp._add_surface_event_cost_columns = skip_surface_costs


def label_and_cluster(frame, candidates, well_index, path_drift, base, fold):
    slices = sg._well_slices(frame["well"].astype(str).to_numpy())
    target = frame["target"].to_numpy(np.float64)
    meta = well_index.set_index("well_index", drop=False)
    labeled = candidates.copy()
    rmse = np.empty(len(labeled), dtype=np.float32)
    abs_level = np.empty(len(labeled), dtype=np.float32)
    well_ids = np.empty(len(labeled), dtype=object)
    for wi, idx in labeled.groupby("well_index", sort=False).groups.items():
        wid = str(meta.loc[int(wi), "well_id"])
        start, end = slices[wid]
        truth = target[start:end]
        for row_i in idx:
            pred = sg._path_for_row(labeled.loc[row_i], path_drift, len(truth)).astype(np.float64)
            err = pred - truth
            rmse[row_i] = np.sqrt(np.mean(err * err))
            abs_level[row_i] = abs(np.mean(err))
            well_ids[row_i] = wid
    labeled["well_id"] = well_ids
    labeled["fold"] = np.int8(fold)
    labeled["target_rmse"] = rmse
    labeled["target_abs_level"] = abs_level
    labeled["target_sse_per_row"] = rmse.astype(np.float64) ** 2
    clusters, clustered = sg.build_cluster_table(labeled)
    oracle = (
        clustered.sort_values(["well_index", "cluster_id", "target_rmse", "target_abs_level"])
        .groupby(["well_index", "cluster_id"], sort=False)
        .head(1)[["well_index", "cluster_id", "target_rmse", "target_abs_level", "well_id"]]
        .rename(columns={"target_rmse": "target_cluster_oracle_rmse", "target_abs_level": "target_cluster_oracle_abs_level"})
    )
    clusters = clusters.drop(columns=[c for c in oracle.columns if c not in {"well_index", "cluster_id"} and c in clusters], errors="ignore")
    clusters = clusters.merge(oracle, on=["well_index", "cluster_id"], how="left")
    clusters["fold"] = np.int8(fold)
    clusters["target_abs_quality"] = np.exp(-clusters["target_cluster_oracle_rmse"].clip(lower=0.0) / 8.0)
    return labeled, clusters, clustered


def build_fold_cache(tag, fold, frame, base, workers, force):
    out = OUT / tag / f"fold_{fold}"
    required = [out / "clusters.parquet", out / "clustered.parquet", out / "path_drift.npy", out / "well_index.csv"]
    if all(p.exists() for p in required) and not force:
        return
    out.mkdir(parents=True, exist_ok=True)
    root = OUT / f"pseudo_roots/fold_{fold}"
    started = time.time()
    candidates, well_index, arrays = build_candidate_pool_fork(frame, root, base, workers)
    labeled, clusters, clustered = label_and_cluster(frame, candidates, well_index, arrays["path_drift"], base, fold)
    labeled.to_parquet(out / "candidates_labeled.parquet", index=False)
    clusters.to_parquet(out / "clusters.parquet", index=False)
    clustered.to_parquet(out / "clustered.parquet", index=False)
    well_index.to_csv(out / "well_index.csv", index=False)
    np.save(out / "path_drift.npy", arrays["path_drift"])
    frame[["id", "well", "target"]].to_parquet(out / "rows.parquet", index=False)
    (out / "cache_meta.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "rows": len(frame),
                "wells": int(frame["well"].nunique()),
                "candidates": len(candidates),
                "clusters": len(clusters),
                "path_values": len(arrays["path_drift"]),
                "elapsed_s": time.time() - started,
                "candidate_family_counts": candidates["family"].astype(str).value_counts().to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    del candidates, labeled, clusters, clustered, arrays
    gc.collect()


def lgb_params(workers, seed):
    return {
        "objective": "regression", "metric": "rmse", "learning_rate": 0.04,
        "num_leaves": 31, "min_data_in_leaf": 18, "feature_fraction": 0.80,
        "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 0.12,
        "seed": seed, "feature_fraction_seed": seed + 1, "bagging_seed": seed + 2,
        "verbosity": -1, "num_threads": workers, "force_col_wise": True,
    }


def train_selectors(tag, workers, force):
    root = OUT / tag
    score_path = root / "selectors/cluster_scores_oof.parquet"
    if score_path.exists() and not force:
        return pd.read_parquet(score_path)
    pieces = []
    for fold in range(5):
        d = pd.read_parquet(root / f"fold_{fold}/clusters.parquet")
        d["fold"] = np.int8(fold)
        d["fold_row"] = np.arange(len(d), dtype=np.int32)
        pieces.append(d)
    allc = pd.concat(pieces, ignore_index=True)
    features = json.loads((ASSET / "selector_feature_columns.json").read_text())
    if any(x in c.lower() for c in features for x in ("target", "truth", "oracle")):
        raise RuntimeError("leaky selector feature name")
    x = allc.reindex(columns=features, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    y = allc["target_abs_quality"].to_numpy(np.float32)
    pred_et = np.empty(len(allc), dtype=np.float32)
    pred_lgb = np.empty(len(allc), dtype=np.float32)
    model_dir = root / "selectors/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        tr = allc["fold"].to_numpy() != fold
        va = ~tr
        model = cct.make_model("et", 20260707 + fold)
        model.steps[-1][1].set_params(n_jobs=workers)
        model.fit(x.loc[tr], y[tr])
        pred_et[va] = model.predict(x.loc[va]).astype(np.float32)
        joblib.dump(model, model_dir / f"et_fold{fold}.joblib", compress=3)
        booster = lgb.train(
            lgb_params(workers, 20261707 + 10 * fold),
            lgb.Dataset(x.loc[tr].astype(np.float32), label=y[tr], feature_name=features),
            num_boost_round=360,
        )
        pred_lgb[va] = booster.predict(x.loc[va].astype(np.float32)).astype(np.float32)
        booster.save_model(str(model_dir / f"lgb_fold{fold}.txt"))
        print(f"[{tag}:selector] fold={fold} train={int(tr.sum())} valid={int(va.sum())}", flush=True)
    allc["pred_et_oof"] = pred_et
    allc["pred_lgb_oof"] = pred_lgb
    allc["tab_score"] = (0.5 * pred_et + 0.5 * pred_lgb).astype(np.float32)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    allc.to_parquet(score_path, index=False)
    (root / "selectors/features.json").write_text(json.dumps(features, indent=2) + "\n")

    full_et = cct.make_model("et", 20260707)
    full_et.steps[-1][1].set_params(n_jobs=workers)
    full_et.fit(x, y)
    joblib.dump(full_et, model_dir / "et_full.joblib", compress=3)
    full_lgb = lgb.train(
        lgb_params(workers, 20261707),
        lgb.Dataset(x.astype(np.float32), label=y, feature_name=features),
        num_boost_round=360,
    )
    full_lgb.save_model(str(model_dir / "lgb_full.txt"))
    return allc


def decode_fold(frame, base, fold, clusters, workers):
    root = OUT / frame.attrs["tag"] / f"fold_{fold}"
    clustered = pd.read_parquet(root / "clustered.parquet")
    path_drift = np.load(root / "path_drift.npy", mmap_mode="r")
    well_index = pd.read_csv(root / "well_index.csv", dtype={"well_id": str})
    ctx = clusters[["well_index", "cluster_id", "tab_score"]]
    scored = clustered.merge(ctx, on=["well_index", "cluster_id"], how="left", validate="many_to_one")
    if scored["tab_score"].isna().any():
        raise RuntimeError(f"fold {fold}: missing OOF selector scores")
    scored["member_score"] = -sg._mean_cost(scored, sg.PREFIX_COST_COLS, fallback=0.5)
    scored["vp_score"] = scored["tab_score"].to_numpy(np.float64) + 0.05 * scored["member_score"].to_numpy(np.float64)
    slices = sg._well_slices(frame["well"].astype(str).to_numpy())
    meta_by_wi = well_index.set_index("well_index", drop=False)
    prop = base.astype(np.float64).copy()
    meta_rows = []
    for wi, group in scored.groupby("well_index", sort=True):
        wid = str(meta_by_wi.loc[int(wi), "well_id"])
        start, end = slices[wid]
        n = end - start
        keep = group.sort_values(["vp_score", "score_rank", "candidate_idx"], ascending=[False, True, True]).head(KEEP)
        if len(keep) < 2:
            continue
        paths = np.asarray([sg._path_for_row(row, path_drift, n) for _, row in keep.iterrows()], dtype=np.float64)
        scores = pd.to_numeric(keep["vp_score"], errors="coerce").fillna(-1e9).to_numpy(np.float64)
        try:
            tw = pd.read_csv(OUT / f"pseudo_roots/fold_{fold}/test/{wid}__typewell.csv")
            seq = sg.make_typewell_sequence(tw)
        except Exception:
            seq = None
        post = sg.segmented_posterior_path(
            paths, scores, seq, float(meta_by_wi.loc[int(wi), "last_known"]),
            segment_len=64, score_weight=0.15, geology_beta=0.4,
            transition_sigma=2.0, jump_sigma=18.0, temperature=1.0,
        )
        ramped = sg.apply_ramped_move(base[start:end], post, alpha=0.64, clip=12.0)
        move = float(np.mean(np.abs(ramped - base[start:end])))
        top_scores = np.sort(scores)[::-1][: min(8, len(scores))]
        weights = sg._softmax(top_scores, 1.0)
        entropy = sg._entropy_norm(weights)
        if entropy <= 1.01 and move <= 30.0:
            prop[start:end] = ramped
        meta_rows.append(
            {
                "well": wid, "n_candidates": len(group), "n_keep": len(keep),
                "move_mae": move, "cluster_entropy_norm": entropy,
                "top_cluster_prob": float(weights[0]) if len(weights) else 1.0,
                "applied": float(entropy <= 1.01 and move <= 30.0),
            }
        )
    return prop.astype(np.float32), pd.DataFrame(meta_rows)


def feature_frame(frame, base, prop):
    move = (prop - base).astype(np.float32)
    abs_move = np.abs(move).astype(np.float32)
    frac = frame["frac"].to_numpy(np.float32)
    md_since = frame["md_since"].to_numpy(np.float32)

    def col(name):
        return frame[name].to_numpy(np.float32)

    out = pd.DataFrame(
        {
            "sg_base_d": base, "sg_prop_d": prop, "sg_move_d": move,
            "sg_abs_move_d": abs_move, "sg_move_x_frac": move * frac,
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
    return out[SG_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def run_arm(tag, workers, force):
    configure_generator(workers)
    frame = pd.read_parquet(prepare_frame())
    base = np.load(ARM_PATHS[tag]).astype(np.float32)
    if len(base) != len(frame) or not np.isfinite(base).all():
        raise RuntimeError(f"{tag}: invalid base OOF")
    for fold in range(5):
        mask = frame["fold"].to_numpy() == fold
        ff = frame.loc[mask].reset_index(drop=True)
        build_fold_cache(tag, fold, ff, base[mask], workers, force)
    clusters = train_selectors(tag, workers, force)
    side = np.empty((len(frame), len(SG_COLS)), dtype=np.float32)
    meta_parts = []
    fold_rows = []
    for fold in range(5):
        mask = frame["fold"].to_numpy() == fold
        ff = frame.loc[mask].reset_index(drop=True)
        ff.attrs["tag"] = tag
        cc = clusters.loc[clusters["fold"].eq(fold)].copy()
        prop, meta = decode_fold(ff, base[mask], fold, cc, workers)
        fs = feature_frame(ff, base[mask], prop)
        side[mask] = fs.to_numpy(np.float32)
        meta["fold"] = fold
        meta_parts.append(meta)
        err = prop.astype(np.float64) - ff["target"].to_numpy(np.float64)
        fold_rows.append({"fold": fold, "rows": len(ff), "wells": int(ff["well"].nunique()), "rmse": float(np.sqrt(np.mean(err * err)))})
    side_df = pd.DataFrame(side, columns=SG_COLS)
    out = OUT / tag
    side_path = out / f"{tag}_anchor_sg_path_features_strict_oof.parquet"
    side_df.to_parquet(side_path, index=False)
    frame[["id", "well", "fold"]].to_parquet(out / "sidecar_ids.parquet", index=False)
    pd.concat(meta_parts, ignore_index=True).to_csv(out / "sg_meta_oof.csv", index=False)
    err = side_df["sg_prop_d"].to_numpy(np.float64) - frame["target"].to_numpy(np.float64)
    well_rmse = (
        pd.DataFrame({"well": frame["well"], "e2": err * err})
        .groupby("well", sort=False)["e2"].mean().pow(0.5).mean()
    )
    summary = {
        "tag": tag,
        "protocol": "geo_kmeans_5fold + fold-pure candidate prior + outer-fold OOF SG selector",
        "evaluation_strict_oof": True,
        "private_safe": True,
        "fold_map": str(FOLD_CSV),
        "fold_csv_sha256": sha256_file(FOLD_CSV),
        "base_oof": str(ARM_PATHS[tag]),
        "base_oof_sha256": sha256_file(ARM_PATHS[tag]),
        "sidecar": str(side_path),
        "sidecar_sha256": sha256_file(side_path),
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "diversity_topk": DIVERSITY,
        "score_head": 12,
        "family_quota": 6,
        "level_delta_bin_ft": 6.0,
        "posterior_keep": KEEP,
        "selector": "0.5 ExtraTrees + 0.5 LightGBM; target=exp(-cluster_oracle_rmse/8)",
        "formation_prior": "validation fold excluded from train-space prior",
        "surface_event_costs": "disabled, matching current ModelB runtime",
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mean_per_well_rmse": float(well_rmse),
        "folds": fold_rows,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(fold_rows).to_csv(out / "summary_folds.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--tag", choices=sorted(ARM_PATHS), required=True)
    r.add_argument("--workers", type=int, default=42)
    r.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare_roots(force=args.force)
    else:
        prepare_roots(force=False)
        run_arm(args.tag, args.workers, args.force)


if __name__ == "__main__":
    main()
