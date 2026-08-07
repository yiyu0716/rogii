#!/usr/bin/env python3
"""Parity harness for online rawPFmean+hinge+softGR2 base generation.

This intentionally does not train any model.  It imports the deploy kernel that
created the historical rawpfmean_current_hinge_softgr2 base and stops before
the segmented/VP overlay.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/rogii")
DEFAULT_RAW_KERNEL = ROOT / "code/kaggle_kernel_rogii-lvlcons-gpu-softgr2-rawpfmean-segweak252-v001/rogii-lvlcons-gpu-softgr2-rawpfmean-segweak252-v001.py"
DEFAULT_DATA = ROOT / "datasets/rogii-wellbore-geology-prediction"
DEFAULT_RAVAGHI = ROOT / "datasets/ravaghi_wellbore_geology_prediction_artifacts_kaggle_current"
DEFAULT_SMOOTHER = ROOT / "datasets_upload/rogii-posterior-gate-smoother-assets"
DEFAULT_HINGE = ROOT / "datasets_upload/rogii-level-shape-hinge-dropblendpath-assets"
DEFAULT_BASE_NPZ = ROOT / "cv/artifacts/rawpfmean_current_hinge_softgr2_base_20260707/predictions.npz"


def import_raw_kernel(path: Path, data_root: Path, ravaghi_root: Path, smoother_root: Path, hinge_root: Path):
    os.environ["ROGII_DATA_DIR"] = str(data_root)
    os.environ["RAVAGHI_ARTIFACT_DIR"] = str(ravaghi_root)
    os.environ["ROGII_SMOOTHER_DIR"] = str(smoother_root)
    os.environ["ROGII_HINGE_ASSET_DIR"] = str(hinge_root)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    spec = importlib.util.spec_from_file_location("rawpfmean_segweak252_for_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import raw kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA_ROOT = data_root
    module.ART_ROOT = ravaghi_root
    module.SMOOTHER_ROOT = smoother_root
    module.HINGE_ROOT = hinge_root
    module.rl.CFG.dataset_path = data_root
    module.rl.CFG.artifacts_path = ravaghi_root
    module.ls.DATA_ROOT = data_root
    module.ls.ART_ROOT = ravaghi_root
    module.ls.CFG.dataset_path = data_root
    module.ls.CFG.artifacts_path = ravaghi_root
    module.rl.NCPU = int(os.environ.get("RAVI_WORKERS", "4"))
    return module


def compute_strong_base(raw, *, max_test_wells: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(int(getattr(raw, "STRUCT_SEED", 1745)))
    raw.install_unpickle_shims()
    sample = pd.read_csv(raw.DATA_ROOT / "sample_submission.csv")
    test_paths = sorted((raw.DATA_ROOT / "test").glob("*__horizontal_well.csv"))
    if max_test_wells > 0:
        keep_wells = [p.name.split("__", 1)[0] for p in test_paths[:max_test_wells]]
        test_paths = test_paths[:max_test_wells]
        sample = sample[sample["id"].str[:8].isin(keep_wells)].reset_index(drop=True)
        print(f"[strong-base] smoke wells={keep_wells} rows={len(sample)}", flush=True)

    test_df = raw.rl.build_dataset(test_paths, is_train=False, label="test")
    features = [c for c in test_df.columns if c not in {"well", "id", "target"}]
    x_test = test_df[features]
    print(f"[strong-base] raw test_df={test_df.shape} features={len(features)}", flush=True)

    base_preds = []
    for name in raw.MODEL_DIRS:
        print(f"[strong-base] loading/predicting {name}", flush=True)
        trainer = raw.load_trainer(name)
        base_preds.append(np.asarray(trainer.predict(x_test), dtype=np.float32))
    base_pred = np.stack(base_preds, axis=1)
    ridge_drift = raw.ridge_fold_ensemble(base_pred)
    pf_test = test_df["pf_ancc"].to_numpy(np.float32) - test_df["last_known_tvt"].to_numpy(np.float32)
    ridge_pp = raw.apply_pp(test_df, ridge_drift, pf_test)
    test_df = test_df.copy()
    test_df["pred"] = test_df["last_known_tvt"].to_numpy(np.float32) + ridge_pp
    test_df = raw.sg_smooth(test_df, "pred")
    test_df["ridge_pp_sg"] = (
        test_df["pred"].to_numpy(np.float32) - test_df["last_known_tvt"].to_numpy(np.float32)
    ).astype(np.float32)

    test_df = raw.add_posterior_smoother_features(test_df, sample)
    test_df = raw.apply_smoother_feature_engineering(test_df)
    smoother_resid = raw.predict_smoother_residuals(test_df)
    test_df["final_drift"] = (test_df["blend_base"].to_numpy(np.float32) + smoother_resid).astype(np.float32)
    test_df = raw.apply_struct_posterior_mode_prior(test_df)
    test_df = raw.add_gate_smoother_features(test_df)
    missing_gate = [c for c in raw.GATE_SMOOTHER_FEATURES if c not in test_df.columns]
    if missing_gate:
        raise RuntimeError(f"gate smoother features missing before predict: {missing_gate}")
    gate_resid = raw.predict_gate_smoother_residuals(test_df)
    test_df["final_drift_gate"] = (
        test_df["pfreplay_gate_base"].to_numpy(np.float32) + gate_resid
    ).astype(np.float32)
    test_df = raw.add_motion_selector_feature(test_df, sample)
    test_df["final_drift_motion_ridge"] = raw.apply_motion_ridge_stack(test_df)
    test_df["gate_bigru"] = test_df["final_drift_gate"].astype(np.float32)
    test_df["gate_base"] = test_df["pfreplay_gate_base"].astype(np.float32)
    test_df["pf_scale5"] = test_df["pf_scale_5"].astype(np.float32)
    test_df["final_drift_level_shape_hinge"] = raw.apply_level_shape_hinge_stack(test_df)

    source_pf = raw.ls.sub2_level_consensus_sources(sample)
    source = sample[["id"]].merge(source_pf, on="id", how="left")
    source = source.merge(
        test_df[["id", "well", "z", "md_since", "pred", "last_known_tvt", "final_drift_level_shape_hinge"]],
        on="id",
        how="left",
    )
    fallback = float(test_df["pred"].mean()) if len(test_df) else 0.0
    source["pred"] = source["pred"].fillna(fallback)
    for col in ["tvt_p0_source", "tvt_rate075_source", "tvt_starttight_source"]:
        source[col] = source[col].fillna(source["pred"])
    if source[["well", "z", "md_since", "last_known_tvt", "final_drift_level_shape_hinge"]].isna().any().any():
        raise RuntimeError("missing columns for strong base blend")

    source["tvt"] = raw.ls.FINAL_W_RIDGE * source["pred"] + raw.ls.FINAL_W_PF * source["tvt_p0_source"]
    p0_work = raw.ls.apply_chord_shrink(raw.ls.apply_f_rectification(source, raw.ls.FRECT_FRAC), raw.ls.CHORD_W)
    source["tvt_p0_chord"] = p0_work["tvt"].to_numpy(np.float32)
    source["tvt_rate075"] = (
        raw.ls.FINAL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + raw.ls.FINAL_W_PF * source["tvt_rate075_source"].to_numpy(np.float32)
    ).astype(np.float32)
    source["tvt_starttight"] = (
        raw.ls.FINAL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + raw.ls.FINAL_W_PF * source["tvt_starttight_source"].to_numpy(np.float32)
    ).astype(np.float32)
    source["tvt_rawpf_source"] = (
        source["tvt_p0_source"].to_numpy(np.float32)
        + source["tvt_rate075_source"].to_numpy(np.float32)
        + source["tvt_starttight_source"].to_numpy(np.float32)
    ) / 3.0
    source["tvt"] = (
        raw.RAWPF_LEVEL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + raw.RAWPF_LEVEL_W_PF * source["tvt_rawpf_source"].to_numpy(np.float32)
    ).astype(np.float32)
    rawpf_work = raw.ls.apply_chord_shrink(raw.ls.apply_f_rectification(source, raw.ls.FRECT_FRAC), raw.ls.CHORD_W)
    source["tvt_rawpf_mean_chord"] = rawpf_work["tvt"].to_numpy(np.float32)
    source["tvt_hinge"] = (
        source["last_known_tvt"].to_numpy(np.float32)
        + source["final_drift_level_shape_hinge"].to_numpy(np.float32)
    ).astype(np.float32)

    level = source.groupby("well", sort=False)["tvt_rawpf_mean_chord"].transform("mean").to_numpy(np.float64)
    mixed_shape_raw = 0.5 * source["tvt_p0_chord"].to_numpy(np.float64) + 0.5 * source["tvt_hinge"].to_numpy(np.float64)
    source["_mixed_shape_raw"] = mixed_shape_raw
    shape = mixed_shape_raw - source.groupby("well", sort=False)["_mixed_shape_raw"].transform("mean").to_numpy(np.float64)
    source["tvt"] = (level + shape).astype(np.float32)
    softgr2_delta = raw.compute_softgr2_level_delta(source)
    source["softgr2_delta"] = softgr2_delta.astype(np.float32)
    source["tvt"] = (level + softgr2_delta.astype(np.float64) + shape).astype(np.float32)
    source["strong_base_d"] = (
        source["tvt"].to_numpy(np.float32) - source["last_known_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    return source, test_df


def compare_to_cached(source: pd.DataFrame, base_npz: Path) -> None:
    z = np.load(base_npz, allow_pickle=True)
    wells = np.asarray(z["wells"]).astype(str)
    pred = np.asarray(z["rawpfmean_current_hinge_softgr2"], dtype=np.float32)
    target = np.asarray(z["target"], dtype=np.float32)
    offset = 0
    cached_rows = []
    for wid, grp in source.groupby("well", sort=False):
        mask = wells == str(wid)
        n = int(mask.sum())
        if n == 0:
            raise RuntimeError(f"cached base missing well {wid}")
        cached_rows.append(
            pd.DataFrame(
                {
                    "id": grp["id"].to_numpy(),
                    "cached_base_d": pred[mask],
                    "target": target[mask],
                }
            )
        )
        offset += n
    cached = pd.concat(cached_rows, ignore_index=True)
    joined = source[["id", "well", "strong_base_d"]].merge(cached, on="id", how="left")
    diff = joined["strong_base_d"].to_numpy(np.float32) - joined["cached_base_d"].to_numpy(np.float32)
    print(
        "[parity] strong_base_vs_cached "
        f"rows={len(joined)} maxabs={float(np.nanmax(np.abs(diff))):.6f} "
        f"mae={float(np.nanmean(np.abs(diff))):.6f} "
        f"rmse={float(np.sqrt(np.nanmean(diff * diff))):.6f}",
        flush=True,
    )
    err = joined["strong_base_d"].to_numpy(np.float32) - joined["target"].to_numpy(np.float32)
    print(
        "[visible] online strong base "
        f"rmse={float(np.sqrt(np.nanmean(err * err))):.6f} "
        f"mean={float(np.nanmean(joined['strong_base_d'])):.6f} "
        f"std={float(np.nanstd(joined['strong_base_d'])):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-kernel", type=Path, default=DEFAULT_RAW_KERNEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ravaghi-root", type=Path, default=DEFAULT_RAVAGHI)
    parser.add_argument("--smoother-root", type=Path, default=DEFAULT_SMOOTHER)
    parser.add_argument("--hinge-root", type=Path, default=DEFAULT_HINGE)
    parser.add_argument("--base-npz", type=Path, default=DEFAULT_BASE_NPZ)
    parser.add_argument("--max-test-wells", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "cv/artifacts/online_strong_base_public3.parquet")
    args = parser.parse_args()

    t0 = time.time()
    raw = import_raw_kernel(args.raw_kernel, args.data_root, args.ravaghi_root, args.smoother_root, args.hinge_root)
    source, _ = compute_strong_base(raw, max_test_wells=args.max_test_wells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source[["id", "well", "last_known_tvt", "tvt", "strong_base_d", "softgr2_delta"]].to_parquet(args.out, index=False)
    print(f"[out] wrote {args.out} rows={len(source)} elapsed={time.time() - t0:.1f}s", flush=True)
    if args.base_npz.exists():
        compare_to_cached(source, args.base_npz)


if __name__ == "__main__":
    main()
