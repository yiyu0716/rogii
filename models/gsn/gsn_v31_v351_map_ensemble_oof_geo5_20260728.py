#!/usr/bin/env python3
"""Exact local OOF reproduction of teammate v3.1/v3.51 map ensemble."""
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
from torch.utils.data import DataLoader

ROOT = Path("/root/rogii")
OOF = ROOT / "OOF"
sys.path.insert(0, str(ROOT / "cv"))
import gsn_v351_honest_oof_geo5_20260728 as core

V31_BUNDLE = ROOT / "teammate_wody/kaggle_infer_v3.1/kaggle_infer_v3.1"
V31_WEIGHTS = V31_BUNDLE / "weights"
V31_MANIFEST = V31_BUNDLE / "manifest.json"
V351_WEIGHTS = core.WEIGHTS
OUT = OOF / "gsn_v31_v351_map_ensemble_oof_geo5_20260728"
PROFILES = core.PROFILE_COLUMNS
EXPECTED_FOLD_SHA = "ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da"


def validate_v31() -> None:
    manifest = json.loads(V31_MANIFEST.read_text())
    if manifest.get("version") != "v3.1":
        raise RuntimeError("v3.1 manifest mismatch")
    for item in manifest["weights"]:
        path = V31_WEIGHTS / item["file"]
        if core.sha256(path) != item["sha256"]:
            raise RuntimeError(f"v3.1 weight checksum mismatch: {path}")


def v351_checkpoint(fold: int) -> Path:
    nested = V351_WEIGHTS / f"fold_{fold}" / f"fold_{fold}_best.pth"
    flat = V351_WEIGHTS / f"fold_{fold}_best.pth"
    path = nested if nested.exists() else flat
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_v351() -> None:
    if core.sha256(core.FOLD_CSV) != EXPECTED_FOLD_SHA:
        raise RuntimeError("canonical Geo5 checksum mismatch")
    manifest_path = V351_WEIGHTS / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fold_sha256") != EXPECTED_FOLD_SHA or not manifest.get("strict_geo5"):
            raise RuntimeError("strict v3.51 manifest mismatch")
    for fold in range(5):
        checkpoint = v351_checkpoint(fold)
        provenance = checkpoint.parent / "PROVENANCE.json"
        if provenance.exists():
            record = json.loads(provenance.read_text())
            if record.get("fold") != fold or record.get("fold_sha256") != EXPECTED_FOLD_SHA:
                raise RuntimeError(f"v3.51 fold{fold} provenance mismatch")
            if record.get("best_checkpoint_sha256") != core.sha256(checkpoint):
                raise RuntimeError(f"v3.51 fold{fold} checkpoint checksum mismatch")


def run_fold(fold: int, device: str) -> pd.DataFrame:
    Config, Dataset, list_splits, Model, seed_everything = core.setup_bundle()
    Config.DEVICE = device
    seed_everything()
    files = sorted(Config.TRAIN_DIR.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    valid_idx = list_splits(files, train_dir=Config.TRAIN_DIR)[fold][1]
    valid_files = [files[int(i)] for i in valid_idx]
    fold_map = core.load_fold_map()
    if {p.name.split("__")[0] for p in valid_files} != {
        w for w, assigned in fold_map.items() if assigned == fold
    }:
        raise RuntimeError(f"fold {fold} membership mismatch")

    models = []
    checkpoints = (V31_WEIGHTS / f"fold_{fold}_best.pth", v351_checkpoint(fold))
    for checkpoint in checkpoints:
        model = Model().to(device)
        model.output_type = ["inference"]
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(model)

    workers = int(os.environ.get("ROGII_GS_NUM_WORKERS", "8"))
    loader_args = dict(
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )
    if workers:
        loader_args.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(Dataset(well_files=valid_files, is_train=False), **loader_args)
    records = []
    use_amp = device.startswith("cuda")
    for well_file, batch in zip(valid_files, loader):
        frame = pd.read_csv(well_file)
        well = well_file.name.split("__")[0]
        device_batch = {
            key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        outputs = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for model in models:
                outputs.append(model(device_batch))
        sdf = np.mean([out["sdf"].float().cpu().numpy()[0, 0] for out in outputs], axis=0)
        seg = np.mean([out["seg"].float().cpu().numpy()[0, 0] for out in outputs], axis=0)
        cpu_batch = {
            key: (value.cpu() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        predictions = core.decode_profiles(cpu_batch, sdf, seg, frame, Config)
        tvt_input = pd.to_numeric(frame.TVT_input, errors="coerce")
        last_tvt = float(tvt_input.dropna().iloc[-1])
        for row_idx in np.flatnonzero(tvt_input.isna().to_numpy()):
            row = {"id": f"{well}_{row_idx}", "fold": fold}
            for profile, prediction in predictions.items():
                row[PROFILES[profile]] = float(prediction[row_idx] - last_tvt)
            records.append(row)
    del models
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def worker(fold: int, gpu: str) -> tuple[int, float]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    run_fold(fold, "cuda:0").to_parquet(OUT / f"fold_{fold}_pred.parquet", index=False)
    return fold, time.time() - started


def infer(devices: list[str], force: bool) -> None:
    pending = [
        fold for fold in range(5)
        if force or not (OUT / f"fold_{fold}_pred.parquet").exists()
    ]
    offset = 0
    while offset < len(pending):
        wave = pending[offset : offset + len(devices)]
        with ProcessPoolExecutor(max_workers=len(wave)) as executor:
            futures = [executor.submit(worker, fold, devices[i]) for i, fold in enumerate(wave)]
            for future in as_completed(futures):
                fold, elapsed = future.result()
                print(f"[fold {fold}] {elapsed:.1f}s", flush=True)
        offset += len(devices)


def assemble() -> dict:
    xy = pd.read_parquet(core.XY, columns=["id", "well", "target"])
    lookup = pd.Series(np.arange(len(xy), dtype=np.int64), index=xy.id.astype(str))
    arrays = {profile: np.full(len(xy), np.nan, np.float64) for profile in PROFILES}
    per_fold = []
    for fold in range(5):
        part = pd.read_parquet(OUT / f"fold_{fold}_pred.parquet")
        pos = lookup.reindex(part.id.astype(str)).to_numpy()
        if np.isnan(pos).any() or part.id.duplicated().any():
            raise RuntimeError(f"fold {fold} alignment failure")
        pos = pos.astype(np.int64)
        row = {"fold": fold, "n_rows": len(part)}
        for profile, column in PROFILES.items():
            arrays[profile][pos] = part[column].to_numpy(float)
            row[profile] = core.diagnose(
                xy.well.to_numpy(str)[pos], xy.target.to_numpy(float)[pos], arrays[profile][pos]
            )["rmse"]
        per_fold.append(row)

    results = {}
    for profile, prediction in arrays.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{profile}: incomplete OOF")
        path = OUT / f"gsn_v31_v351_map_{profile}_d.npy"
        np.save(path, prediction.astype(np.float32))
        results[profile] = {
            **core.diagnose(xy.well.to_numpy(str), xy.target.to_numpy(float), prediction),
            "path": str(path),
            "sha256": core.sha256(path),
        }
    summary = {
        "experiment": "exact v3.1/v3.51 map-average then dual-decode OOF",
        "protocol": "canonical geo5 fold-k checkpoints; average SDF/seg before decode",
        "private_safe": True,
        "fully_oof": True,
        "n_rows": len(xy),
        "fold_map": str(core.FOLD_CSV),
        "fold_sha256": core.sha256(core.FOLD_CSV),
        "checkpoint_provenance": {
            "v31_manifest": str(V31_MANIFEST),
            "v31_manifest_sha256": core.sha256(V31_MANIFEST),
            "v31": [
                {"fold": fold, "path": str(V31_WEIGHTS / f"fold_{fold}_best.pth"),
                 "sha256": core.sha256(V31_WEIGHTS / f"fold_{fold}_best.pth")}
                for fold in range(5)
            ],
            "v351": [
                {"fold": fold, "path": str(v351_checkpoint(fold)),
                 "sha256": core.sha256(v351_checkpoint(fold))}
                for fold in range(5)
            ],
        },
        "results": results,
        "per_fold": per_fold,
        "teammate_reported_dual_cv": 7.4504,
        "local_minus_teammate_dual": results["dual"]["rmse"] - 7.4504,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(per_fold).to_csv(OUT / "PER_FOLD.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    global OUT, V31_WEIGHTS, V31_MANIFEST, V351_WEIGHTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--v31-weights", type=Path, default=V31_WEIGHTS)
    parser.add_argument("--v31-manifest", type=Path, default=V31_MANIFEST)
    parser.add_argument("--v351-weights", type=Path, default=V351_WEIGHTS)
    args = parser.parse_args()
    OUT = args.out
    V31_WEIGHTS = args.v31_weights
    V31_MANIFEST = args.v31_manifest
    V351_WEIGHTS = args.v351_weights
    OUT.mkdir(parents=True, exist_ok=True)
    if core.sha256(core.FOLD_CSV) != EXPECTED_FOLD_SHA:
        raise RuntimeError("canonical Geo5 checksum mismatch")
    validate_v31()
    validate_v351()
    if not args.assemble_only:
        devices = [item.strip().split(":")[-1] for item in args.devices.split(",") if item.strip()]
        infer(devices, args.force)
    assemble()


if __name__ == "__main__":
    main()
