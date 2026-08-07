#!/usr/bin/env python3
"""Replay the final canonical-Geo5 WARP checkpoints and build deployment bundle.

Unlike an OOF cache copy, every validation prediction here is regenerated from
the checkpoint and train-fold normalization. The exact same five checkpoint
states and normalizers are packed for Kaggle inference.
"""
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
CANONICAL = OOF / "sg_path_strict_oof_div90_geo5_20260723/shared/canonical_fold_honest_frame.parquet"
SOURCE = OOF / "warp_exp207"
CACHE = SOURCE / "gr_features_cache.pkl"
WARP_CODE = ROOT / "teammate_warp_exp207/exp207_warp_raw_train/EXP/exp204"
DEFAULT_OUT = OOF / "cv609_strict_geo5_rebuild_20260729/warp_final_geo5_checkpoint_replay"
EXPECTED_FOLD_SHA = "ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validated_cache() -> list[dict]:
    wells = joblib.load(CACHE)
    fold_map = pd.read_csv(FOLD_CSV, dtype={"well_id": str}).set_index("well_id").fold
    if len(wells) != 773 or len({str(w["wid"]) for w in wells}) != 773:
        raise RuntimeError("unexpected WARP cache cohort")
    mismatch = [
        str(w["wid"]) for w in wells
        if int(w["fold"]) != int(fold_map.loc[str(w["wid"])])
    ]
    if mismatch:
        raise RuntimeError(f"WARP cache fold mismatch: {mismatch[:10]}")
    return wells


def replay_fold(fold: int, out: Path) -> None:
    sys.path.insert(0, str(WARP_CODE))
    from train_warp import compute_norm, predict_well
    from warp_model import WARP_U2Net

    wells = load_validated_cache()
    train = [w for w in wells if int(w["fold"]) != fold]
    valid = [w for w in wells if int(w["fold"]) == fold]
    norm = compute_norm(train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = SOURCE / f"warp_raw_fold{fold}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = WARP_U2Net(
        in_ch=train[0]["features"].shape[1],
        tw_T=train[0]["tw_tokens"].shape[0],
        use_typewell=True,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    started = time.time()
    predictions = [predict_well(model, well, norm, device) for well in valid]
    out.mkdir(parents=True, exist_ok=True)
    pack_path = out / f"oof_checkpoint_replay_fold{fold}.npz"
    norm_path = out / f"norm_fold{fold}.npz"
    np.savez_compressed(
        pack_path,
        wid=np.asarray([str(w["wid"]) for w in valid]),
        n=np.asarray([len(p) for p in predictions], dtype=np.int32),
        pred=np.concatenate(predictions).astype(np.float32),
    )
    np.savez(norm_path, **norm)
    reference_path = SOURCE / f"oof_raw_fold{fold}.npz"
    reference = np.load(reference_path)
    replay = np.load(pack_path)
    if not np.array_equal(reference["wid"].astype(str), replay["wid"].astype(str)):
        raise RuntimeError(f"fold{fold}: historical replay well order mismatch")
    delta = replay["pred"].astype(np.float64) - reference["pred"].astype(np.float64)
    record = {
        "fold": fold,
        "protocol": "checkpoint-driven canonical Geo5 validation replay",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "n_train_wells": len(train),
        "n_val_wells": len(valid),
        "pack": str(pack_path),
        "pack_sha256": sha256(pack_path),
        "norm": str(norm_path),
        "norm_sha256": sha256(norm_path),
        "historical_training_oof": str(reference_path),
        "replay_vs_historical_rmse_delta": float(np.sqrt(np.mean(delta * delta))),
        "replay_vs_historical_max_abs_delta": float(np.max(np.abs(delta))),
        "elapsed_s": time.time() - started,
    }
    (out / f"FOLD{fold}_REPLAY.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def launch(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    devices = [item.strip() for item in args.gpus.split(",") if item.strip()]
    running: list[tuple[int, subprocess.Popen, object]] = []
    for fold in range(5):
        while len([item for item in running if item[1].poll() is None]) >= len(devices):
            time.sleep(2)
        gpu = devices[fold % len(devices)]
        log_path = args.out / f"fold{fold}_console.log"
        handle = log_path.open("w")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"})
        command = [
            sys.executable, "-u", str(Path(__file__)), "--cmd", "replay-one",
            "--fold", str(fold), "--out", str(args.out),
        ]
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((fold, process, handle))
        print(f"[WARP replay] launched fold{fold} gpu={gpu} pid={process.pid}", flush=True)
    failures = []
    for fold, process, handle in running:
        code = process.wait()
        handle.close()
        print(f"[WARP replay] fold{fold} returncode={code}", flush=True)
        if code:
            failures.append(fold)
    if failures:
        raise RuntimeError(f"WARP replay folds failed: {failures}")


def assemble(out: Path) -> None:
    wells = load_validated_cache()
    by_well = {str(w["wid"]): w for w in wells}
    canonical = pd.read_parquet(CANONICAL, columns=["id", "well", "target", "fold"])
    positions_by_well = {
        str(well): group.index.to_numpy(np.int64)
        for well, group in canonical.groupby(canonical.well.astype(str), sort=False)
    }
    prediction = np.full(len(canonical), np.nan, dtype=np.float32)
    bundle = {}
    records = []
    for fold in range(5):
        record = json.loads((out / f"FOLD{fold}_REPLAY.json").read_text())
        pack = np.load(out / f"oof_checkpoint_replay_fold{fold}.npz")
        cursor = 0
        for wid, count in zip(pack["wid"].astype(str), pack["n"].astype(int)):
            pos = positions_by_well.get(wid)
            if pos is None or not len(pos) or not np.all(canonical.fold.to_numpy()[pos] == fold):
                raise RuntimeError(f"fold{fold}: canonical membership failure for {wid}")
            segment = pack["pred"][cursor:cursor + count].astype(np.float64)
            cursor += count
            if len(segment) != len(pos):
                raise RuntimeError(f"fold{fold}: row count mismatch for {wid}")
            prediction[pos] = (segment - float(by_well[wid]["last_tvt"])).astype(np.float32)
        norm = np.load(out / f"norm_fold{fold}.npz")
        state_path = SOURCE / f"warp_raw_fold{fold}.pt"
        bundle[fold] = {
            "state": torch.load(state_path, map_location="cpu", weights_only=True),
            **{name: norm[name] for name in norm.files},
        }
        mask = canonical.fold.to_numpy() == fold
        record["fold_rmse"] = float(np.sqrt(np.mean(
            (prediction[mask].astype(np.float64) - canonical.target.to_numpy()[mask]) ** 2
        )))
        records.append(record)
    if not np.isfinite(prediction).all():
        raise RuntimeError("WARP replay OOF incomplete")
    oof_path = out / "oof_final_geo5_checkpoint_replay_d.npy"
    bundle_path = out / "warp_final_geo5_bundle.pt"
    np.save(oof_path, prediction)
    torch.save(bundle, bundle_path)
    summary = {
        "protocol": "final WARP checkpoint replay; canonical Geo5; deploy-identical checkpoint states and train-fold normalizers",
        "fold_csv": str(FOLD_CSV),
        "fold_sha256": sha256(FOLD_CSV),
        "source_checkpoint_root": str(SOURCE),
        "cache": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "oof": str(oof_path),
        "oof_sha256": sha256(oof_path),
        "oof_rmse": float(np.sqrt(np.mean(
            (prediction.astype(np.float64) - canonical.target.to_numpy()) ** 2
        ))),
        "deploy_bundle": str(bundle_path),
        "deploy_bundle_sha256": sha256(bundle_path),
        "folds": records,
        "strict_geo5_oof": True,
        "deployment_checkpoint_identical": True,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", choices=("replay-one", "launch", "assemble", "all"), default="all")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if sha256(FOLD_CSV) != EXPECTED_FOLD_SHA:
        raise RuntimeError("canonical fold checksum mismatch")
    if args.cmd == "replay-one":
        replay_fold(args.fold, args.out)
    else:
        if args.cmd in ("launch", "all"):
            launch(args)
        if args.cmd in ("assemble", "all"):
            assemble(args.out)


if __name__ == "__main__":
    main()
