#!/usr/bin/env python3
"""Retrain GeoSteerNet v3.51 with explicit canonical Geo5 provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path("/root/rogii")
OOF = ROOT / "OOF"
DATA = ROOT / "datasets/rogii-wellbore-geology-prediction"
FOLD_CSV = OOF / "geo_kmeans_5fold.csv"
BUNDLE = OOF / "_teammate_assets/kaggle_infer_v3.51/kaggle_infer_v3.51"
CODE = BUNDLE / "code"
DEFAULT_OUT = OOF / "cv609_strict_geo5_rebuild_20260729/gsn_v351_geo5_retrain"
EXPECTED_FOLD_SHA = "ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train_one(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(CODE))
    from config.cnn_sdf_config import Config
    from src.dataset import list_well_cv_splits
    from src.seed import seed_everything
    from src.train import _run_fold_training

    Config.TRAIN_DIR = DATA / "train"
    Config.TEST_DIR = DATA / "test"
    Config.CV_SPLIT_STRATEGY = "fixed_csv"
    Config.CV_SPLIT_CSV = FOLD_CSV
    Config.NUM_WORKERS = args.workers
    Config.BATCH_SIZE = args.batch
    Config.EPOCHS = args.epochs
    Config.DEVICE = "cuda:0"
    fold_map = pd.read_csv(FOLD_CSV, dtype={"well_id": str})
    files = sorted(Config.TRAIN_DIR.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    splits = list_well_cv_splits(
        files, train_dir=Config.TRAIN_DIR, strategy="fixed_csv", split_csv=FOLD_CSV
    )
    train_idx, val_idx = splits[args.fold]
    train_files = [files[int(index)] for index in train_idx]
    val_files = [files[int(index)] for index in val_idx]
    val_wells = {path.name.split("__", 1)[0] for path in val_files}
    expected = set(fold_map.loc[fold_map.fold.eq(args.fold), "well_id"].astype(str))
    if val_wells != expected:
        raise RuntimeError(f"fold{args.fold} canonical membership mismatch")
    fold_out = args.out / f"fold_{args.fold}"
    fold_out.mkdir(parents=True, exist_ok=False)
    fold_csv_copy = fold_out / "geo_kmeans_5fold.csv"
    shutil.copy2(FOLD_CSV, fold_csv_copy)
    (fold_out / "train_wells.json").write_text(
        json.dumps([path.name.split("__", 1)[0] for path in train_files], indent=2) + "\n"
    )
    (fold_out / "val_wells.json").write_text(json.dumps(sorted(val_wells), indent=2) + "\n")
    seed_everything()
    started = time.time()
    result = _run_fold_training(
        args.fold, train_files, val_files, fold_out, args.epochs, device="cuda:0"
    )
    best = fold_out / f"fold_{args.fold}_best.pth"
    last = fold_out / f"fold_{args.fold}_last.pth"
    training_log = fold_out / f"fold_{args.fold}_log.csv"
    val_log = fold_out / f"fold_{args.fold}_val_rmse.csv"
    for required in (best, last, training_log, val_log):
        if not required.exists():
            raise FileNotFoundError(required)
    record = {
        "fold": args.fold,
        "protocol": "canonical fixed_csv Geo5 v3.51 retrain",
        "fold_csv": str(FOLD_CSV),
        "fold_sha256": sha256(FOLD_CSV),
        "fold_csv_copy": str(fold_csv_copy),
        "fold_csv_copy_sha256": sha256(fold_csv_copy),
        "n_train_wells": len(train_files),
        "n_val_wells": len(val_files),
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256(best),
        "last_checkpoint": str(last),
        "last_checkpoint_sha256": sha256(last),
        "training_log": str(training_log),
        "training_log_sha256": sha256(training_log),
        "val_log": str(val_log),
        "val_log_sha256": sha256(val_log),
        "result": result,
        "elapsed_s": time.time() - started,
        "strict_geo5": True,
    }
    (fold_out / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def launch(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    running: list[tuple[int, subprocess.Popen, object]] = []
    jobs = []
    for fold in range(5):
        while len([item for item in running if item[1].poll() is None]) >= len(gpus):
            time.sleep(5)
        gpu = gpus[fold % len(gpus)]
        log_path = args.out / f"fold_{fold}_console.log"
        handle = log_path.open("w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable, "-u", str(Path(__file__)), "--cmd", "train-one",
            "--fold", str(fold), "--out", str(args.out), "--epochs", str(args.epochs),
            "--batch", str(args.batch), "--workers", str(args.workers),
        ]
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((fold, process, handle))
        jobs.append({"fold": fold, "gpu": gpu, "pid": process.pid, "log": str(log_path)})
        print(f"[GSN v3.51 Geo5] launched fold{fold} gpu={gpu} pid={process.pid}", flush=True)
    failed = []
    for fold, process, handle in running:
        code = process.wait()
        handle.close()
        print(f"[GSN v3.51 Geo5] fold{fold} returncode={code}", flush=True)
        if code:
            failed.append(fold)
        jobs[fold]["returncode"] = code
    (args.out / "TRAIN_JOBS.json").write_text(json.dumps(jobs, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"GSN folds failed: {failed}")


def assemble(out: Path) -> None:
    records = []
    for fold in range(5):
        path = out / f"fold_{fold}/PROVENANCE.json"
        if not path.exists():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text())
        checkpoint = Path(record["best_checkpoint"])
        if sha256(checkpoint) != record["best_checkpoint_sha256"]:
            raise RuntimeError(f"fold{fold} checkpoint checksum mismatch")
        records.append(record)
    manifest = {
        "name": "rogii-geosteernet-sdf",
        "version": "v3.51-strict-geo5-retrain-20260729",
        "protocol": "canonical fixed_csv Geo5, one held fold per checkpoint",
        "fold_csv": str(FOLD_CSV),
        "fold_sha256": sha256(FOLD_CSV),
        "folds": records,
        "strict_geo5": True,
        "training_logs_complete": True,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", choices=["launch", "train-one", "assemble", "all"], default="all")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if sha256(FOLD_CSV) != EXPECTED_FOLD_SHA:
        raise RuntimeError("canonical fold checksum mismatch")
    if args.cmd == "train-one":
        if not 0 <= args.fold < 5:
            raise RuntimeError("train-one requires --fold 0..4")
        train_one(args)
    else:
        if args.cmd in ("launch", "all"):
            launch(args)
        if args.cmd in ("assemble", "all"):
            assemble(args.out)


if __name__ == "__main__":
    main()
