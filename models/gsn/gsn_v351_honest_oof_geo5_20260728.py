#!/usr/bin/env python3
"""Reproduce teammate GSN v3.51 as honest canonical geo5 OOF predictions.

Each validation well is inferred only by its assigned fold checkpoint. One model
forward produces five decode profiles so model and post-processing gains remain
separable:

  argmin       : per-column argmin(abs(sdf))
  viterbi_orig : abs(sdf) - seg, transition penalty 0.10
  vit005       : abs(sdf), transition penalty 0.05
  argsg        : argmin followed by future-only Savitzky-Golay(101, 3)
  dual         : 0.5 * vit005 + 0.5 * argsg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader


ROOT = Path("/root/rogii")
OOF = ROOT / "OOF"
FOLD_CSV = OOF / "geo_kmeans_5fold.csv"
EXPECTED_FOLD_SHA256 = "ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da"
ARCHIVE = OOF / "kaggle_infer_v3.51.rar"
EXPECTED_ARCHIVE_SHA256 = "8ad2de8857c5f0cf32f9291433394e05e419f74a5b3ffadb3f9c35d96742336d"
BUNDLE = OOF / "_teammate_assets/kaggle_infer_v3.51/kaggle_infer_v3.51"
CODE = BUNDLE / "code"
WEIGHTS = BUNDLE / "weights"
MANIFEST = BUNDLE / "manifest.json"
DATA = ROOT / "datasets/rogii-wellbore-geology-prediction"
XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
CANON = ROOT / "experiments/repro_1086"
OUT_DIR = OOF / "gsn_v351_honest_oof_geo5_20260728"
EXPECTED_ROWS = 3_783_989
PROFILE_COLUMNS = {
    "argmin": "drift_argmin",
    "viterbi_orig": "drift_viterbi_orig",
    "vit005": "drift_vit005",
    "argsg": "drift_argsg",
    "dual": "drift_dual",
}

os.environ.setdefault("ROGII_GS_NUM_WORKERS", "8")
_ABS_DIFF: dict[int, np.ndarray] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def diagnose(wells: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(pred, float) - np.asarray(target, float)
    codes, names = pd.factorize(wells, sort=False)
    counts = np.bincount(codes).astype(float)
    level = np.bincount(codes, weights=error) / counts
    swing_error = error - level[codes]
    per_well = np.sqrt(np.bincount(codes, weights=error * error) / counts)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "level": float(np.sqrt(np.mean(level[codes] ** 2))),
        "swing": float(np.sqrt(np.mean(swing_error * swing_error))),
        "mean_per_well_rmse": float(per_well.mean()),
        "median_per_well_rmse": float(np.median(per_well)),
    }


def validate_inputs() -> dict:
    if sha256(FOLD_CSV) != EXPECTED_FOLD_SHA256:
        raise RuntimeError("canonical fold checksum mismatch")
    if sha256(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("v3.51 archive checksum mismatch")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("version") != "v3.51":
        raise RuntimeError(f"unexpected manifest version: {manifest.get('version')}")
    for item in manifest["weights"]:
        path = WEIGHTS / item["file"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"weight checksum mismatch: {path}")
    return manifest


def load_fold_map() -> dict[str, int]:
    frame = pd.read_csv(FOLD_CSV)
    if len(frame) != 773 or frame.well_id.duplicated().any():
        raise RuntimeError("canonical fold-map schema mismatch")
    return {str(row.well_id): int(row.fold) for row in frame.itertuples(index=False)}


def setup_bundle():
    sys.path.insert(0, str(CODE))
    from config.cnn_sdf_config import Config
    from src.dataset import WellboreSDFDataset, list_well_cv_splits
    from src.model import GeoSteerNet
    from src.seed import seed_everything

    Config.TRAIN_DIR = DATA / "train"
    Config.TEST_DIR = DATA / "test"
    Config.CV_SPLIT_STRATEGY = "fixed_csv"
    Config.CV_SPLIT_CSV = FOLD_CSV
    return Config, WellboreSDFDataset, list_well_cv_splits, GeoSteerNet, seed_everything


def abs_diff(size: int) -> np.ndarray:
    if size not in _ABS_DIFF:
        index = np.arange(size, dtype=np.float32)
        _ABS_DIFF[size] = np.abs(index[:, None] - index[None, :])
    return _ABS_DIFF[size]


def viterbi(cost: np.ndarray, anchor_t_idx: int, h_h: int, penalty: float) -> np.ndarray:
    rows, cols = cost.shape
    dp = np.full((rows, cols), 1e9, dtype=np.float32)
    pointer = np.zeros((rows, cols), dtype=np.int32)
    dp[anchor_t_idx, h_h - 1] = cost[anchor_t_idx, h_h - 1]
    row_index = np.arange(rows)
    distance = abs_diff(rows)
    for h in range(h_h, cols):
        transition = dp[:, h - 1][None, :] + penalty * distance
        previous = np.argmin(transition, axis=1)
        dp[:, h] = cost[:, h] + transition[row_index, previous]
        pointer[:, h] = previous
    path = np.zeros(cols, dtype=np.int32)
    path[:h_h] = anchor_t_idx
    path[-1] = np.argmin(dp[:, -1])
    for h in range(cols - 1, h_h - 1, -1):
        path[h - 1] = pointer[path[h], h]
    return path


def map_h_to_original(
    pred_h: np.ndarray,
    original_len: int,
    true_ps: int,
    h_step: int,
    h_h: int,
    h_f: int,
    anchor_tvt: float,
) -> np.ndarray:
    hist_k = np.arange(h_h, dtype=np.float64)
    hist_centers = true_ps - hist_k * h_step - (h_step - 1) / 2.0
    hist_idx = (h_h - 1 - hist_k).astype(np.int64)
    future_k = np.arange(h_f, dtype=np.float64)
    future_centers = true_ps + 1.0 + future_k * h_step + (h_step - 1) / 2.0
    future_idx = (h_h + future_k).astype(np.int64)
    centers = np.concatenate([hist_centers, future_centers])
    values = pred_h[np.concatenate([hist_idx, future_idx])]
    if np.isfinite(anchor_tvt):
        centers = np.concatenate([centers, [float(true_ps)]])
        values = np.concatenate([values, [float(anchor_tvt)]])
    order = np.argsort(centers)
    return np.interp(np.arange(original_len, dtype=np.float64), centers[order], values[order])


def decode_index_path(batch: dict, path: np.ndarray, df: pd.DataFrame, Config) -> np.ndarray:
    t_tvt = batch["t_seg_tvt"].numpy()[0]
    h_tvt = batch["h_seg_tvt"].numpy()[0]
    true_ps = int(batch["true_ps"].item())
    h_step = int(batch["h_s"].item()) if "h_s" in batch else int(Config.H_S)
    pred_h = t_tvt[path].astype(np.float64)
    pred_h[: int(Config.H_H)] = h_tvt[: int(Config.H_H)]
    known = pd.to_numeric(df["TVT_input"], errors="coerce").dropna()
    anchor = float(known.iloc[-1]) if len(known) else np.nan
    return map_h_to_original(
        pred_h,
        len(df),
        true_ps,
        h_step,
        int(Config.H_H),
        len(pred_h) - int(Config.H_H),
        anchor,
    )


def smooth_future(pred: np.ndarray, true_ps: int, window: int = 101, poly: int = 3) -> np.ndarray:
    future = pred[true_ps:]
    width = min(window, len(future) if len(future) % 2 else len(future) - 1)
    if width < 5:
        return pred.copy()
    if width % 2 == 0:
        width -= 1
    smoothed = savgol_filter(future, window_length=width, polyorder=min(poly, width - 1))
    smoothed += pred[true_ps] - smoothed[0]
    result = pred.copy()
    result[true_ps:] = smoothed
    return result


def decode_profiles(batch: dict, sdf: np.ndarray, seg: np.ndarray, df: pd.DataFrame, Config) -> dict[str, np.ndarray]:
    anchor_idx = int(batch["anchor_t_idx"].item())
    h_h = int(Config.H_H)
    argmin_path = np.argmin(np.abs(sdf), axis=0)
    original_cost = np.abs(sdf).astype(np.float32) - float(Config.SEG_VITERBI_WEIGHT) * seg.astype(np.float32)
    original_path = viterbi(original_cost, anchor_idx, h_h, 0.10)
    vit005_path = viterbi(np.abs(sdf).astype(np.float32), anchor_idx, h_h, 0.05)
    argmin = decode_index_path(batch, argmin_path, df, Config)
    original = decode_index_path(batch, original_path, df, Config)
    vit005 = decode_index_path(batch, vit005_path, df, Config)
    argsg = smooth_future(argmin, int(batch["true_ps"].item()), 101, 3)
    return {
        "argmin": argmin,
        "viterbi_orig": original,
        "vit005": vit005,
        "argsg": argsg,
        "dual": 0.5 * vit005 + 0.5 * argsg,
    }


def run_fold(fold: int, device: str) -> pd.DataFrame:
    Config, Dataset, list_splits, Model, seed_everything = setup_bundle()
    Config.DEVICE = device
    seed_everything()
    well_files = sorted(Config.TRAIN_DIR.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    # list_splits returns a list of (train_idx, validation_idx).
    splits = list_splits(well_files, train_dir=Config.TRAIN_DIR)
    valid_files = [well_files[int(i)] for i in splits[fold][1]]
    fold_map = load_fold_map()
    got = {path.name.split("__")[0] for path in valid_files}
    expected = {well for well, assigned in fold_map.items() if assigned == fold}
    if got != expected:
        raise RuntimeError(f"fold {fold} validation membership mismatch")

    checkpoint = WEIGHTS / f"fold_{fold}_best.pth"
    model = Model().to(device)
    model.output_type = ["inference"]
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    workers = int(os.environ.get("ROGII_GS_NUM_WORKERS", "8"))
    loader_args = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.startswith("cuda"),
    }
    if workers:
        loader_args.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(Dataset(well_files=valid_files, is_train=False), **loader_args)

    records: list[dict] = []
    use_amp = device.startswith("cuda")
    for well_file, batch in zip(valid_files, loader):
        well = well_file.name.split("__")[0]
        frame = pd.read_csv(well_file)
        device_batch = {
            key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            output = model(device_batch)
        sdf = output["sdf"].float().cpu().numpy()[0, 0]
        seg = output["seg"].float().cpu().numpy()[0, 0]
        cpu_batch = {
            key: (value.cpu() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        profiles = decode_profiles(cpu_batch, sdf, seg, frame, Config)
        known = pd.to_numeric(frame["TVT_input"], errors="coerce")
        last_tvt = float(known.dropna().iloc[-1])
        unknown_index = np.flatnonzero(known.isna().to_numpy())
        for row_index in unknown_index:
            row = {"id": f"{well}_{row_index}", "fold": fold}
            for profile, prediction in profiles.items():
                row[PROFILE_COLUMNS[profile]] = float(prediction[row_index] - last_tvt)
            records.append(row)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def fold_worker(fold: int, gpu_index: str) -> tuple[int, float]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    started = time.time()
    prediction = run_fold(fold, "cuda:0")
    output = OUT_DIR / f"fold_{fold}_pred.parquet"
    prediction.to_parquet(output, index=False)
    return fold, time.time() - started


def build_fold_predictions(devices: list[str], force: bool) -> None:
    gpu_ids = [item.split(":")[-1] for item in devices]
    pending = []
    for fold in range(5):
        path = OUT_DIR / f"fold_{fold}_pred.parquet"
        if path.exists() and not force:
            print(f"[reuse] fold {fold}: {path}", flush=True)
        else:
            pending.append(fold)
    offset = 0
    while offset < len(pending):
        wave = pending[offset : offset + len(gpu_ids)]
        with ProcessPoolExecutor(max_workers=len(wave)) as executor:
            futures = [
                executor.submit(fold_worker, fold, gpu_ids[i % len(gpu_ids)])
                for i, fold in enumerate(wave)
            ]
            for future in as_completed(futures):
                fold, elapsed = future.result()
                print(f"[fold {fold}] complete in {elapsed:.1f}s", flush=True)
        offset += len(gpu_ids)


def assemble() -> dict:
    xy = pd.read_parquet(XY, columns=["id", "well", "target"])
    if len(xy) != EXPECTED_ROWS or xy.id.duplicated().any():
        raise RuntimeError("canonical Xy frame mismatch")
    id_to_pos = pd.Series(np.arange(len(xy), dtype=np.int64), index=xy.id.astype(str))
    arrays = {profile: np.full(len(xy), np.nan, np.float64) for profile in PROFILE_COLUMNS}
    fold_rows = []
    for fold in range(5):
        part = pd.read_parquet(OUT_DIR / f"fold_{fold}_pred.parquet")
        position = id_to_pos.reindex(part.id.astype(str)).to_numpy()
        if np.isnan(position).any() or len(np.unique(position)) != len(position):
            raise RuntimeError(f"fold {fold} ID alignment failure")
        position = position.astype(np.int64)
        mask = np.zeros(len(xy), dtype=bool)
        mask[position] = True
        fold_result = {"fold": fold, "n_rows": int(len(part))}
        for profile, column in PROFILE_COLUMNS.items():
            arrays[profile][position] = part[column].to_numpy(np.float64)
            fold_result[profile] = diagnose(
                xy.well.to_numpy(str)[mask], xy.target.to_numpy(float)[mask], arrays[profile][mask]
            )["rmse"]
        fold_rows.append(fold_result)

    truth = xy.target.to_numpy(np.float64)
    wells = xy.well.to_numpy(str)
    results = {}
    for profile, prediction in arrays.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{profile}: incomplete OOF")
        path = OUT_DIR / f"gsn_v351_{profile}_d.npy"
        np.save(path, prediction.astype(np.float32))
        results[profile] = {**diagnose(wells, truth, prediction), "path": str(path), "sha256": sha256(path)}

    reference = np.load(OOF / "gsn_v31_honest_oof_geo5_20260726/gsn_v31_d.npy").astype(np.float64)
    comparisons = {}
    row_frame = pd.DataFrame({"well": wells, "target": truth, "v31": reference})
    for profile, prediction in arrays.items():
        row_frame[profile] = prediction
    well_parts = []
    for well, group in row_frame.groupby("well", sort=False):
        record = {"well": well, "rows": len(group)}
        for name in ["v31", *PROFILE_COLUMNS]:
            error = group[name].to_numpy(float) - group.target.to_numpy(float)
            record[f"rmse_{name}"] = float(np.sqrt(np.mean(error * error)))
            record[f"level_{name}"] = float(np.mean(error))
        well_parts.append(record)
    well_diagnostics = pd.DataFrame(well_parts)
    well_diagnostics["bucket"] = (
        pd.qcut(well_diagnostics.rmse_v31.rank(method="first"), 5, labels=False).astype(int) + 1
    )
    well_diagnostics.to_csv(OUT_DIR / "WELL_DIAGNOSTICS.csv", index=False)
    bucket_by_well = well_diagnostics.set_index("well").bucket
    row_bucket = bucket_by_well.reindex(pd.Index(wells)).to_numpy(np.int8)
    bucket_rows = []
    well_comparisons = {}
    for profile, prediction in arrays.items():
        profile_well = well_diagnostics[f"rmse_{profile}"]
        baseline_well = well_diagnostics.rmse_v31
        well_comparisons[profile] = {
            "improved_wells_vs_v31": int((profile_well < baseline_well).sum()),
            "worsened_wells_vs_v31": int((profile_well > baseline_well).sum()),
            "mean_well_delta_vs_v31": float((profile_well - baseline_well).mean()),
        }
        comparisons[profile] = {
            "prediction_pearson_vs_v31": float(np.corrcoef(prediction, reference)[0, 1]),
            "residual_pearson_vs_v31": float(np.corrcoef(prediction - truth, reference - truth)[0, 1]),
            "blend5050_v31_rmse": float(np.sqrt(np.mean((0.5 * prediction + 0.5 * reference - truth) ** 2))),
        }
    for profile in ["v31", *PROFILE_COLUMNS]:
        for bucket in range(1, 6):
            mask = row_bucket == bucket
            prediction = reference if profile == "v31" else arrays[profile]
            bucket_rows.append({
                "profile": profile,
                "bucket": bucket,
                "wells": int((well_diagnostics.bucket == bucket).sum()),
                "rmse": float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2))),
            })
    pd.DataFrame(bucket_rows).to_csv(OUT_DIR / "BUCKETS.csv", index=False)
    summary = {
        "experiment": "GSN v3.51 honest canonical geo5 OOF reproduction",
        "protocol": "geo_kmeans_5fold + future_only_unknown_segment + fold-k checkpoint only",
        "private_safe": True,
        "fully_oof": True,
        "n_rows": len(xy),
        "n_wells": int(xy.well.nunique()),
        "fold_map": str(FOLD_CSV),
        "fold_sha256": sha256(FOLD_CSV),
        "archive": str(ARCHIVE),
        "archive_sha256": sha256(ARCHIVE),
        "manifest": json.loads(MANIFEST.read_text()),
        "results": results,
        "per_fold": fold_rows,
        "comparisons_vs_v31": comparisons,
        "well_comparisons_vs_v31": well_comparisons,
        "bucket_definition": "quintiles of v3.1 per-well RMSE, Q1 easiest to Q5 hardest",
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "PER_FOLD.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_inputs()
    if not args.assemble_only:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
        if not devices:
            raise SystemExit("at least one GPU device is required")
        build_fold_predictions(devices, args.force)
    assemble()


if __name__ == "__main__":
    main()
