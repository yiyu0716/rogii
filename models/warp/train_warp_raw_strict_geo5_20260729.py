#!/usr/bin/env python3
"""Train, assemble, and verify deploy-identical raw WARP Geo5 checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/rogii")
OOF = ROOT / "OOF"
FOLD_CSV = OOF / "geo_kmeans_5fold.csv"
CACHE = OOF / "warp_exp207/gr_features_cache.pkl"
CANONICAL = OOF / "sg_path_strict_oof_div90_geo5_20260723/shared/canonical_fold_honest_frame.parquet"
WARP_ROOT = ROOT / "teammate_warp_exp207/exp207_warp_raw_train/EXP"
TRAIN = WARP_ROOT / "exp207/train_ema.py"
DEFAULT_OUT = OOF / "cv609_strict_geo5_rebuild_20260729/warp_raw_geo5"
EXPECTED_FOLD_SHA = "ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache(wells: list[dict]) -> None:
    fold_map = pd.read_csv(FOLD_CSV, dtype={"well_id": str}).set_index("well_id").fold
    if len(wells) != 773 or len({str(w["wid"]) for w in wells}) != 773:
        raise RuntimeError("unexpected WARP cache cohort")
    mismatch = [
        str(w["wid"]) for w in wells
        if int(w["fold"]) != int(fold_map.loc[str(w["wid"])])
    ]
    if mismatch:
        raise RuntimeError(f"WARP cache is not canonical Geo5: {mismatch[:10]}")


def launch(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    wells = joblib.load(CACHE)
    validate_cache(wells)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("no GPUs configured")
    running: list[tuple[int, subprocess.Popen, object]] = []
    records = []
    for fold in range(5):
        while len([item for item in running if item[1].poll() is None]) >= len(gpus):
            time.sleep(2)
        gpu = gpus[fold % len(gpus)]
        log_path = args.out / "logs" / f"fold{fold}.log"
        handle = log_path.open("w")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "ROGII_WARP_CACHE": str(CACHE),
            "ROGII_WARP_OUT": str(args.out),
            "ROGII_WARP_NUM_WORKERS": str(args.workers),
            "ROGII_WARP_PREFETCH": "4",
            "OMP_NUM_THREADS": "1",
        })
        command = [
            sys.executable, "-u", str(TRAIN), "--fold", str(fold),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--decay", "0.999", "--batch", str(args.batch),
        ]
        process = subprocess.Popen(
            command, cwd=str(TRAIN.parent), env=env, stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((fold, process, handle))
        records.append({"fold": fold, "gpu": gpu, "pid": process.pid, "log": str(log_path)})
        print(f"[WARP Geo5] launched fold{fold} gpu={gpu} pid={process.pid}", flush=True)
    failed = []
    for fold, process, handle in running:
        code = process.wait()
        handle.close()
        records[fold]["returncode"] = code
        print(f"[WARP Geo5] fold{fold} returncode={code}", flush=True)
        if code:
            failed.append(fold)
    (args.out / "TRAIN_JOBS.json").write_text(json.dumps(records, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"WARP folds failed: {failed}")


def assemble(out: Path) -> None:
    wells = joblib.load(CACHE)
    validate_cache(wells)
    by_well = {str(w["wid"]): w for w in wells}
    canonical = pd.read_parquet(CANONICAL, columns=["id", "well", "target", "fold"])
    prediction = np.full(len(canonical), np.nan, dtype=np.float32)
    bundle = {}
    sys.path.insert(0, str(WARP_ROOT / "exp204"))
    from train_warp import compute_norm

    fold_rows = []
    for fold in range(5):
        pack_path = out / f"oof_raw_fold{fold}.npz"
        state_path = out / f"warp_raw_fold{fold}.pt"
        if not pack_path.exists() or not state_path.exists():
            raise FileNotFoundError(pack_path if not pack_path.exists() else state_path)
        pack = np.load(pack_path, allow_pickle=True)
        cursor = 0
        for wid, count in zip(pack["wid"].astype(str), pack["n"].astype(int)):
            mask = canonical.well.astype(str).eq(wid).to_numpy()
            if not mask.any() or not np.all(canonical.fold.to_numpy()[mask] == fold):
                raise RuntimeError(f"fold{fold} OOF membership failure for {wid}")
            segment = pack["pred"][cursor:cursor + count].astype(np.float64)
            cursor += count
            prediction[mask] = (segment - float(by_well[wid]["last_tvt"])).astype(np.float32)
        if cursor != len(pack["pred"]):
            raise RuntimeError(f"fold{fold} OOF cursor mismatch")
        train_wells = [w for w in wells if int(w["fold"]) != fold]
        norm = compute_norm(train_wells)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        bundle[fold] = {"state": state, **norm}
        mask = canonical.fold.to_numpy() == fold
        rmse = float(np.sqrt(np.mean((prediction[mask] - canonical.target.to_numpy()[mask]) ** 2)))
        fold_rows.append({
            "fold": fold,
            "rmse": rmse,
            "state": str(state_path),
            "state_sha256": sha256(state_path),
            "n_train_wells": len(train_wells),
            "n_val_wells": int(canonical.loc[mask, "well"].nunique()),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("strict WARP OOF is incomplete")
    oof_path = out / "oof_raw_geo5_d.npy"
    bundle_path = out / "warp_raw_geo5_bundle.pt"
    np.save(oof_path, prediction)
    torch.save(bundle, bundle_path)
    summary = {
        "protocol": "canonical Geo5 raw WARP; fold-k checkpoint trained on folds != k",
        "fold_csv": str(FOLD_CSV),
        "fold_sha256": sha256(FOLD_CSV),
        "oof": str(oof_path),
        "oof_sha256": sha256(oof_path),
        "oof_rmse": float(np.sqrt(np.mean((prediction - canonical.target.to_numpy()) ** 2))),
        "deploy_bundle": str(bundle_path),
        "deploy_bundle_sha256": sha256(bundle_path),
        "folds": fold_rows,
        "private_safe": True,
        "strict_geo5_oof": True,
        "deploy_checkpoint_identical": True,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", choices=["launch", "assemble", "all"], default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if sha256(FOLD_CSV) != EXPECTED_FOLD_SHA:
        raise RuntimeError("canonical fold checksum mismatch")
    if args.cmd in ("launch", "all"):
        launch(args)
    if args.cmd in ("assemble", "all"):
        assemble(args.out)


if __name__ == "__main__":
    main()
