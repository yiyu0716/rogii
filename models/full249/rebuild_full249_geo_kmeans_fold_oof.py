#!/usr/bin/env python3
"""geo_kmeans fold-OOF full249 features + LightGBM retrain (fixed).

P0 fixes vs first draft:
  - Val targets come from truth_official.parquet AFTER feature build
    (target = tvt - last_tvt); never from is_train=False builder.
  - keep_cols deduped (last_tvt/md_since already in feature_columns).

Also:
  - Timestamped experiment dir under OOF/
  - Cache reuse gated by protocol fingerprint
  - Per-fold OOF shards + optional partial merge
  - INFER_PROTOCOL: fold models need fold-specific test features

protocol: geo_kmeans_5fold + future_only_unknown_segment + private_safe
         + features=fold_oof_honest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/rogii")
sys.path.insert(0, str(ROOT / "cv"))

from full249_reference_bundle import (  # noqa: E402
    build_reference_bundle_manifest,
    hash_wells,
    import_public_source,
    materialize_pseudo_root,
    select_outer_wells,
    sha256_file,
    stable_json_hash,
)

FOLD_CSV = ROOT / "OOF/geo_kmeans_5fold.csv"
ASSET = ROOT / "datasets_upload/rogii-public-train19-lgb5fold-assets"
SOURCE = ASSET / "public_xgb_train19_source.py"
CLEAN_XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
FEAT_JSON = ASSET / "feature_columns.json"
COMP_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "OOF"
CURRENT_LINK = OOF_ROOT / "full249_geo5fold_current"


def _set_thread_env(n_jobs: int) -> None:
    os.environ["N_JOBS"] = str(n_jobs)
    os.environ["ROGII_N_JOBS"] = str(n_jobs)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[k] = "1"


def _fork_build_well(hw_path: str, is_train: bool, mod_name: str):
    """Fork-inherited worker: parent must have FI/DI set on sys.modules[mod_name]."""
    public = sys.modules[mod_name]
    return public.build_well(Path(hw_path), is_train)


def _fork_build_surface(hw_path: str, is_train: bool, mod_name: str):
    public = sys.modules[mod_name]
    return public.build_surface_feats(Path(hw_path), public._PARALLEL_SI, is_train)


def _fork_build_dip(hw_path: str, is_train: bool, mod_name: str):
    public = sys.modules[mod_name]
    return public.build_feats_dip(Path(hw_path), public._PARALLEL_DF, is_train)


def _fork_genphys_one(hw_path: str, sdip, mod_name: str):
    public = sys.modules[mod_name]
    return public.genphys_one(hw_path, sdip)


def _parallel_map_fork(fn, tasks: list, n_jobs: int):
    """ProcessPoolExecutor with fork so large FI/SI objects are inherited, not pickled."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    if not tasks:
        return []
    nj = max(1, min(int(n_jobs), len(tasks)))
    ctx = get_context("fork")
    out = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=nj, mp_context=ctx) as ex:
        futs = {ex.submit(fn, *args): i for i, args in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"    [parallel] {done}/{len(tasks)}", flush=True)
    return out


def _patch_public_parallel(public, n_jobs: int, *, source_path: Path, data_dir: Path, feature_env: dict) -> None:
    """Replace serial per-well loops with fork-based process parallelism."""
    nj = int(n_jobs)
    mod_name = public.__name__
    sys.modules[mod_name] = public
    Path_ = public.Path
    pd = public.pd
    time = public.time

    def build_dataset_parallel(d, is_train):
        paths = [str(p) for p in sorted(Path_(d).glob("*__horizontal_well.csv"))]
        tag = "train" if is_train else "test"
        t0 = time.time()
        if getattr(public, "FI", None) is None or getattr(public, "DI", None) is None:
            raise RuntimeError("FI/DI must be fit before parallel build_dataset")
        print(f"  [{tag}] build_dataset FORK-PARALLEL start: {len(paths)} wells, n_jobs={nj}", flush=True)
        tasks = [(p, is_train, mod_name) for p in paths]
        parts = _parallel_map_fork(_fork_build_well, tasks, nj)
        parts = [r for r in parts if r is not None]
        out = pd.concat(parts, ignore_index=True)
        print(
            f"  [{tag}] build_dataset FORK-PARALLEL done: shape={out.shape} "
            f"kept={len(parts)}/{len(paths)} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        return out

    def add_surface_feats_parallel(base_df, d, SI, is_train):
        paths = [str(p) for p in sorted(Path_(d).glob("*__horizontal_well.csv"))]
        tag = "train" if is_train else "test"
        t0 = time.time()
        print(f"  [{tag}] surface FORK-PARALLEL start: {len(paths)} wells, n_jobs={nj}", flush=True)
        public._PARALLEL_SI = SI
        try:
            tasks = [(p, is_train, mod_name) for p in paths]
            parts = _parallel_map_fork(_fork_build_surface, tasks, nj)
        except Exception as e:
            print(f"  [{tag}] surface parallel failed ({e}); fallback serial", flush=True)
            return _surface_serial(public, base_df, d, SI, is_train)
        parts = [r for r in parts if r is not None]
        sf = pd.concat(parts, ignore_index=True)
        merged = base_df.merge(sf, on="id", how="left")
        assert len(merged) == len(base_df)
        print(
            f"  [{tag}] surface FORK-PARALLEL done: shape={merged.shape} "
            f"(+{sf.shape[1]-1} cols, {time.time() - t0:.0f}s)",
            flush=True,
        )
        return merged

    def add_feats_dip_parallel(base_df, d, DF, is_train):
        paths = [str(p) for p in sorted(Path_(d).glob("*__horizontal_well.csv"))]
        tag = "train" if is_train else "test"
        t0 = time.time()
        print(f"  [{tag}] dip FORK-PARALLEL start: {len(paths)} wells, n_jobs={nj}", flush=True)
        public._PARALLEL_DF = DF
        try:
            tasks = [(p, is_train, mod_name) for p in paths]
            parts = _parallel_map_fork(_fork_build_dip, tasks, nj)
        except Exception as e:
            print(f"  [{tag}] dip parallel failed ({e}); fallback serial", flush=True)
            return _dip_serial(public, base_df, d, DF, is_train)
        parts = [r for r in parts if r is not None]
        sf = pd.concat(parts, ignore_index=True)
        merged = base_df.merge(sf, on="id", how="left")
        assert len(merged) == len(base_df)
        print(
            f"  [{tag}] dip FORK-PARALLEL done: shape={merged.shape} "
            f"(+{sf.shape[1]-1} cols, {time.time() - t0:.0f}s)",
            flush=True,
        )
        return merged

    def add_phys_feats_parallel(base_df, d, DF=None, is_train=False):
        paths = sorted(Path_(d).glob("*__horizontal_well.csv"))
        tag = "train" if is_train else "test"
        t0 = time.time()
        sdip_map = {}
        if os.environ.get("DIPFUSE", "0") == "1" and DF is not None:
            for p in paths:
                wid, sd = public._sdip_one(str(p), DF, is_train)
                if sd is not None:
                    sdip_map[wid] = sd
            print(
                f"  sdip computed for {len(sdip_map)} wells "
                f"(DIPFUSE kappa={os.environ.get('DIPKAPPA', '0.1')})",
                flush=True,
            )

        def _w(p):
            return Path_(p).stem.replace("__horizontal_well", "")

        pns = getattr(public, "_PNS", os.environ.get("PNS", "48"))
        pnp = getattr(public, "_PNP", os.environ.get("PNP", "400"))
        print(
            f"  [{tag}] phys/lik-PF FORK-PARALLEL start: {len(paths)} wells, "
            f"n_jobs={nj}, PNS={pns}, PNP={pnp}",
            flush=True,
        )
        tasks = [(str(p), sdip_map.get(_w(p)), mod_name) for p in paths]
        parts = _parallel_map_fork(_fork_genphys_one, tasks, nj)
        pf = pd.concat([p for p in parts if p is not None], ignore_index=True)
        merged = base_df.merge(pf, on="id", how="left")
        assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
        before_cols = merged.shape[1]
        merged = public._add_modeconf_features(merged)
        modeconf_cols = merged.shape[1] - before_cols
        print(
            f"  [{tag}] phys/lik-PF FORK-PARALLEL done: shape={merged.shape} "
            f"(+{pf.shape[1]-1} phys cols, +{modeconf_cols} modeconf cols, "
            f"{time.time() - t0:.0f}s)",
            flush=True,
        )
        return merged

    public.build_dataset = build_dataset_parallel
    public.add_surface_feats = add_surface_feats_parallel
    public.add_feats_dip = add_feats_dip_parallel
    public.add_phys_feats = add_phys_feats_parallel
    print(f"[patch] fork-parallel build_dataset/surface/dip/phys n_jobs={nj}", flush=True)


def _surface_serial(public, base_df, d, SI, is_train):
    parts = []
    paths = sorted(public.Path(d).glob("*__horizontal_well.csv"))
    for p in paths:
        r = public.build_surface_feats(p, SI, is_train)
        if r is not None:
            parts.append(r)
    sf = public.pd.concat(parts, ignore_index=True)
    return base_df.merge(sf, on="id", how="left")


def _dip_serial(public, base_df, d, DF, is_train):
    parts = []
    paths = sorted(public.Path(d).glob("*__horizontal_well.csv"))
    for p in paths:
        r = public.build_feats_dip(p, DF, is_train)
        if r is not None:
            parts.append(r)
    sf = public.pd.concat(parts, ignore_index=True)
    return base_df.merge(sf, on="id", how="left")


def diagnose(wells, y, pred) -> dict:
    err = pred - y
    rmse = float(np.sqrt(np.mean(err * err)))
    df = pd.DataFrame({"well": wells, "e2": err * err})
    mw = float(np.sqrt(df.groupby("well")["e2"].mean()).mean())
    return {"rmse": rmse, "mean_per_well_rmse": mw}


def protocol_fingerprint(
    *,
    fold: int,
    train_wells: list[str],
    val_wells: list[str],
    feature_columns: list[str],
    feature_env: dict,
) -> dict:
    return {
        "fold": int(fold),
        "fold_csv": str(FOLD_CSV),
        "fold_csv_sha256": sha256_file(FOLD_CSV),
        "source_sha256": sha256_file(SOURCE),
        "feature_schema_sha256": sha256_file(FEAT_JSON),
        "feature_schema_hash": stable_json_hash(feature_columns),
        "feature_env_hash": stable_json_hash(feature_env),
        "train_well_hash": hash_wells(train_wells),
        "validation_well_hash": hash_wells(val_wells),
        "cut_variant": "official",
        "features_protocol": "fold_oof_honest_imputers_train_only",
        "target_join": "truth_official.parquet tvt - last_tvt AFTER feature build",
    }


def attach_val_targets(val_df: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    """Join private-safe-deferred labels onto val features after build."""
    if not truth_path.exists():
        raise FileNotFoundError(truth_path)
    truth = pd.read_parquet(truth_path)
    # truth columns: id, well, row_idx, tvt
    need = {"id", "tvt"}
    if not need.issubset(truth.columns):
        raise RuntimeError(f"truth missing cols: {truth.columns.tolist()}")
    if "last_tvt" not in val_df.columns:
        raise RuntimeError("val_df missing last_tvt; cannot form target=tvt-last_tvt")
    out = val_df.drop(columns=["target"], errors="ignore").copy()
    out["id"] = out["id"].astype(str)
    t = truth[["id", "tvt"]].copy()
    t["id"] = t["id"].astype(str)
    out = out.merge(t, on="id", how="left")
    if out["tvt"].isna().any():
        n = int(out["tvt"].isna().sum())
        raise RuntimeError(f"val target join missed {n} rows vs {truth_path}")
    out["target"] = (out["tvt"].to_numpy(np.float64) - out["last_tvt"].to_numpy(np.float64)).astype(
        np.float32
    )
    out = out.drop(columns=["tvt"])
    return out


def select_keep_cols(df: pd.DataFrame, feature_columns: list[str], *, require_target: bool) -> list[str]:
    keys = ["id", "well"]
    if require_target:
        keys.append("target")
    # feature_columns already includes last_tvt, md_since — do not prepend them again
    keep = list(dict.fromkeys(keys + feature_columns))
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns for keep: {missing[:20]}")
    return keep


def cache_reusable(fold_dir: Path, expected_fp: dict) -> bool:
    meta_p = fold_dir / "meta.json"
    val_p = fold_dir / "Xy_val.parquet"
    train_p = fold_dir / "Xy_train.parquet"
    if not (meta_p.exists() and val_p.exists() and train_p.exists()):
        return False
    meta = json.loads(meta_p.read_text())
    got = meta.get("protocol_fingerprint")
    if got != expected_fp:
        print(f"[{fold_dir.name}] cache fingerprint mismatch; rebuilding", flush=True)
        return False
    # require target present on val
    cols = pd.read_parquet(val_p, columns=["target"]).columns
    if "target" not in cols:
        print(f"[{fold_dir.name}] cached val missing target; rebuilding", flush=True)
        return False
    return True


def build_fold_matrices(
    *,
    fold: int,
    fold_dir: Path,
    n_jobs: int,
    keep_train_parquet: bool,
    feature_columns: list[str],
    feature_env: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    work = fold_dir / "workdir"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    train_wells, val_wells, split_meta = select_outer_wells(
        outer_fold_zero_based=fold,
        max_test_wells=0,
        max_train_wells=0,
        split_path=FOLD_CSV,
    )
    fp = protocol_fingerprint(
        fold=fold,
        train_wells=train_wells,
        val_wells=val_wells,
        feature_columns=feature_columns,
        feature_env=feature_env,
    )
    print(f"[fold{fold}] train_wells={len(train_wells)} val_wells={len(val_wells)}", flush=True)

    t0 = time.time()
    pseudo_manifest = materialize_pseudo_root(
        out_dir=work,
        train_wells=train_wells,
        target_wells=val_wells,
        cut_variant="official",
        source_comp_root=COMP_ROOT,
        copy_train=True,
    )
    bundle = build_reference_bundle_manifest(
        pseudo_manifest=pseudo_manifest,
        train_wells=train_wells,
        target_wells=val_wells,
        source_path=SOURCE,
        asset_root=ASSET,
        feature_columns=feature_columns,
        config={"feature_env": feature_env, "fold": fold, "split_meta": split_meta},
    )
    (fold_dir / "reference_bundle.json").write_text(bundle.to_json(), encoding="utf-8")
    (fold_dir / "train_wells.json").write_text(json.dumps(train_wells, indent=2) + "\n")
    (fold_dir / "val_wells.json").write_text(json.dumps(val_wells, indent=2) + "\n")

    _set_thread_env(n_jobs)
    for k, v in feature_env.items():
        os.environ[str(k)] = str(v)

    public = import_public_source(
        source_path=SOURCE,
        data_dir=Path(pseudo_manifest["pseudo_root"]),
        feature_env=feature_env,
        module_tag=f"geo_kmeans_fold{fold}_{int(time.time())}",
    )
    os.environ["N_JOBS"] = str(n_jobs)
    os.environ["ROGII_N_JOBS"] = str(n_jobs)
    _patch_public_parallel(
        public,
        n_jobs,
        source_path=SOURCE,
        data_dir=Path(pseudo_manifest["pseudo_root"]),
        feature_env=feature_env,
    )

    print(f"[fold{fold}] fitting imputers on train-only pseudo-root...", flush=True)
    si, dfield = public._build_imputers(t0)

    print(f"[fold{fold}] building TRAIN features (is_train=True)...", flush=True)
    train_df = public._build_full(public.DATA / "train", True, si, dfield, t0, f"fold{fold}_train")
    print(f"[fold{fold}] building VAL features (is_train=False, no target yet)...", flush=True)
    val_df = public._build_full(public.DATA / "test", False, si, dfield, t0, f"fold{fold}_val")

    missing_tr = [c for c in feature_columns if c not in train_df.columns]
    missing_va = [c for c in feature_columns if c not in val_df.columns]
    if missing_tr or missing_va:
        raise RuntimeError(f"fold{fold} missing feats train={missing_tr[:10]} val={missing_va[:10]}")

    # Attach labels AFTER feature build (truth was written by materialize_pseudo_root)
    truth_path = work / "truth_official.parquet"
    val_df = attach_val_targets(val_df, truth_path)
    # Keep truth copy for audit
    shutil.copy2(truth_path, fold_dir / "truth_official.parquet")

    keep_tr = select_keep_cols(train_df, feature_columns, require_target=True)
    keep_va = select_keep_cols(val_df, feature_columns, require_target=True)
    train_df = train_df[keep_tr].copy()
    val_df = val_df[keep_va].copy()
    assert train_df.columns.is_unique and val_df.columns.is_unique

    val_path = fold_dir / "Xy_val.parquet"
    val_df.to_parquet(val_path, index=False)
    if keep_train_parquet:
        train_df.to_parquet(fold_dir / "Xy_train.parquet", index=False)
    else:
        # Always persist train for cache reuse / resume (needed for skip fingerprint)
        train_df.to_parquet(fold_dir / "Xy_train.parquet", index=False)

    meta = {
        "fold": fold,
        "fold_map": str(FOLD_CSV),
        "n_train_wells": len(train_wells),
        "n_val_wells": len(val_wells),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "elapsed_sec": float(time.time() - t0),
        "split_meta": split_meta,
        "protocol_fingerprint": fp,
        "masked_view_hash": pseudo_manifest.get("masked_view_hash"),
        "features_protocol": "fold_oof_honest_imputers_train_only",
        "n_jobs": n_jobs,
        "note_deploy": (
            "This fold's LGB model was trained on features whose FI/DI/SI/DipField "
            "were fit on train_wells only. At inference, rebuild test features with "
            "the SAME train_wells imputer fit (see train_wells.json + rebuild recipe)."
        ),
    }
    (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[fold{fold}] matrices ready train={len(train_df)} val={len(val_df)} "
        f"in {meta['elapsed_sec']:.0f}s",
        flush=True,
    )

    # Drop bulky pseudo-root but keep cut/truth manifests already copied
    try:
        shutil.rmtree(Path(pseudo_manifest["pseudo_root"]))
    except Exception as e:
        print(f"[fold{fold}] warn: could not rm pseudo_root: {e}", flush=True)

    return train_df, val_df, meta


def train_fold_lgb(
    *,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: list[str],
    models_dir: Path,
    lgb_jobs: int,
    n_estimators: int,
    early_stopping: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    if "target" not in val_df.columns:
        raise RuntimeError("val_df missing target after truth join")
    xtr = train_df[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    ytr = train_df["target"].to_numpy(np.float32)
    xva = val_df[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    yva = val_df["target"].to_numpy(np.float32)

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.02,
        n_estimators=n_estimators,
        num_leaves=127,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=10.0,
        max_bin=256,
        random_state=seed + fold + 1,
        n_jobs=lgb_jobs,
        verbosity=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(200),
        ],
    )
    pred = model.predict(xva, num_iteration=model.best_iteration_).astype(np.float32)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(
        str(models_dir / f"model_fold{fold}.txt"), num_iteration=model.best_iteration_
    )
    rmse = float(np.sqrt(mean_squared_error(yva, pred)))
    row = {
        "fold": fold,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "best_iteration": int(model.best_iteration_ or 0),
        "rmse": rmse,
        "n_train_wells": int(train_df["well"].nunique()),
        "n_val_wells": int(val_df["well"].nunique()),
    }
    print(f"[full249-honest] fold{fold}: rmse={rmse:.6f} best_iter={model.best_iteration_}", flush=True)
    return pred, row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n-jobs", type=int, default=80)
    ap.add_argument("--lgb-jobs", type=int, default=80)
    ap.add_argument("--n-estimators", type=int, default=8000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--exp-dir", type=str, default="", help="default: OOF/full249_geo5fold_<ts>")
    ap.add_argument("--skip-existing", action="store_true", help="reuse fold cache if fingerprint matches")
    ap.add_argument("--allow-partial", action="store_true", help="write OOF shards without requiring all folds")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    feats = json.loads(FEAT_JSON.read_text())
    assert len(feats) == 249
    feature_env = json.loads((ASSET / "manifest.json").read_text()).get("public_train19_feature_env", {})

    if args.exp_dir:
        exp = Path(args.exp_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        exp = OOF_ROOT / f"full249_geo5fold_{ts}"
    feat_root = exp / "features"
    models_dir = exp / "models"
    oof_dir = exp / "oof"
    feat_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    (exp / "features.json").write_text(json.dumps(feats, indent=2) + "\n")
    protocol = {
        "protocol": "geo_kmeans_5fold + future_only_unknown_segment + private_safe + features=fold_oof_honest",
        "fold_map": str(FOLD_CSV),
        "fold_csv_sha256": sha256_file(FOLD_CSV),
        "source_sha256": sha256_file(SOURCE),
        "exp_dir": str(exp),
        "note": "SI/DipField/FI/DI fit on outer-fold train wells only; val labels joined from truth AFTER feature build.",
    }
    (exp / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (exp / "INFER_PROTOCOL.json").write_text(
        json.dumps(
            {
                "warning": (
                    "Do NOT score these models on the global all-773-fit Xy.parquet. "
                    "Each fold model requires test features rebuilt with that fold's train_wells "
                    "imputers (see features/fold_k/train_wells.json)."
                ),
                "oof_eval": "Xy_val.parquet already uses fold-honest features + truth-joined target.",
                "submit_recipe": (
                    "For each fold k: fit imputers on train_wells.json, transform test wells, "
                    "predict with models/model_fold{k}.txt; blend/average fold predictions."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    # Update convenience symlink
    if CURRENT_LINK.exists() or CURRENT_LINK.is_symlink():
        CURRENT_LINK.unlink()
    CURRENT_LINK.symlink_to(exp)

    # Also expose stable names expected by later blend scripts
    # (symlink OOF/full249 -> exp/models after success; during run point features link)
    full249_link = OOF_ROOT / "full249"
    feat_link = OOF_ROOT / "full249_fold_oof_features"

    xy_ref = pd.read_parquet(CLEAN_XY, columns=["id", "well", "target"])
    id_to_pos = {str(i): k for k, i in enumerate(xy_ref["id"].astype(str))}
    oof = np.full(len(xy_ref), np.nan, dtype=np.float64)
    rows = []

    for fold in folds:
        fold_dir = feat_root / f"fold_{fold}"
        train_wells, val_wells, _ = select_outer_wells(
            outer_fold_zero_based=fold,
            split_path=FOLD_CSV,
        )
        expected_fp = protocol_fingerprint(
            fold=fold,
            train_wells=train_wells,
            val_wells=val_wells,
            feature_columns=feats,
            feature_env=feature_env,
        )

        if args.skip_existing and cache_reusable(fold_dir, expected_fp):
            print(f"[fold{fold}] reuse fingerprint-matched cache", flush=True)
            train_df = pd.read_parquet(fold_dir / "Xy_train.parquet")
            val_df = pd.read_parquet(fold_dir / "Xy_val.parquet")
        else:
            train_df, val_df, _meta = build_fold_matrices(
                fold=fold,
                fold_dir=fold_dir,
                n_jobs=args.n_jobs,
                keep_train_parquet=True,
                feature_columns=feats,
                feature_env=feature_env,
            )

        pred, row = train_fold_lgb(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            feats=feats,
            models_dir=models_dir,
            lgb_jobs=args.lgb_jobs,
            n_estimators=args.n_estimators,
            early_stopping=args.early_stopping,
            seed=42,
        )
        rows.append(row)

        # per-fold OOF shard
        shard = pd.DataFrame(
            {
                "id": val_df["id"].astype(str),
                "well": val_df["well"].astype(str),
                "target": val_df["target"].to_numpy(np.float32),
                "pred": pred,
                "fold": fold,
            }
        )
        shard.to_parquet(oof_dir / f"oof_fold{fold}.parquet", index=False)
        np.save(oof_dir / f"oof_fold{fold}.npy", pred)

        miss = 0
        for sid, p in zip(val_df["id"].astype(str), pred):
            pos = id_to_pos.get(sid)
            if pos is None:
                miss += 1
                continue
            oof[pos] = float(p)
        if miss:
            raise RuntimeError(f"fold{fold}: {miss} val ids not in CLEAN_XY")

        del train_df, val_df, pred

    n_nan = int(np.isnan(oof).sum())
    if n_nan and not args.allow_partial:
        raise RuntimeError(f"OOF still has {n_nan} NaNs (use --allow-partial for subset folds)")

    if n_nan == 0:
        overall = diagnose(
            xy_ref["well"].astype(str).to_numpy(),
            xy_ref["target"].to_numpy(np.float64),
            oof,
        )
        rows.append({"fold": -1, "n_train": 0, "n_val": int(len(xy_ref)), "best_iteration": 0, **overall})
        np.save(models_dir / "oof.npy", oof.astype(np.float32))
        np.save(oof_dir / "oof.npy", oof.astype(np.float32))
        print(f"[full249-honest] OVERALL {overall}", flush=True)
    else:
        overall = {"rmse": None, "mean_per_well_rmse": None, "n_nan": n_nan}
        print(f"[full249-honest] partial OOF n_nan={n_nan}", flush=True)

    pd.DataFrame(rows).to_csv(models_dir / "summary.csv", index=False)
    shutil.copy2(FEAT_JSON, models_dir / "features.json")
    (models_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "full249",
                "overall": overall,
                "fold_map": str(FOLD_CSV),
                "exp_dir": str(exp),
                "features_protocol": "fold_oof_honest_imputers_train_only",
                "protocol": protocol["protocol"],
            },
            indent=2,
        )
        + "\n"
    )

    # Stable links for downstream
    for link, target in ((full249_link, models_dir), (feat_link, feat_root)):
        if link.exists() or link.is_symlink():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                # do not wipe a real dir with aborted content — rename aside
                aside = OOF_ROOT / f"_aside_{link.name}_{int(time.time())}"
                link.rename(aside)
        link.symlink_to(target)

    (exp / "DONE.json").write_text(
        json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "overall": overall}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
