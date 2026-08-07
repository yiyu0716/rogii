#!/usr/bin/env python3
"""Train one geo5 fold with nested UPF10, optionally adding nested A2."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from full249_fg1_fold0 import BASE, FG1_FEATURES
from full249_fg234_fold0 import CONSISTENCY_FEATURES
from train_full249_fg2_upf_a2_onefold import attach_well_features, fg2_well_features

ROOT = Path(__file__).resolve().parents[1]
XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
UPF_ROOT = ROOT / "OOF/upf10_nested_geo5_20260726"
A2_ROOT = ROOT / "OOF/full249_whs_a2_nested_geo5_20260724"
OUT = ROOT / "OOF/full249_fg2_nestedupf_a2_geo5_20260726"


def attach_block(frame: pd.DataFrame, positions: pd.Series, block: np.ndarray,
                 names: list[str]) -> pd.DataFrame:
    pos = positions.reindex(frame.id.astype(str)).to_numpy()
    if np.any(pd.isna(pos)):
        raise RuntimeError("canonical id position missing")
    values = np.asarray(block[pos.astype(np.int64)], dtype=np.float32)
    if values.shape != (len(frame), len(names)):
        raise RuntimeError("sidecar attachment shape mismatch")
    for j, name in enumerate(names):
        frame[name] = values[:, j]
    return frame


def train_variant(name: str, fold: int, train: pd.DataFrame, val: pd.DataFrame,
                  features: list[str], out_root: Path, threads: int,
                  n_estimators: int, early_stopping: int,
                  oof_status: str = "PROMOTION-ELIGIBLE OOF under declared nested protocol") -> dict[str, object]:
    out = out_root / f"fold_{fold}" / name
    out.mkdir(parents=True, exist_ok=False)
    xtr = train[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    ytr = train.target.to_numpy(np.float32)
    xva = val[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    yva = val.target.to_numpy(np.float32)
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.02,
        n_estimators=n_estimators, num_leaves=127, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=10.0, max_bin=256,
        random_state=43 + fold, n_jobs=threads, verbosity=-1,
    )
    t0 = time.time()
    model.fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(100)])
    pred = model.predict(xva, num_iteration=model.best_iteration_).astype(np.float32)
    rmse = float(np.sqrt(np.mean((pred - yva) ** 2)))
    model.booster_.save_model(str(out / f"model_fold{fold}.txt"), num_iteration=model.best_iteration_)
    pd.DataFrame({"id": val.id.astype(str), "well": val.well.astype(str),
                  "target": yva, "pred": pred, "fold": fold}).to_parquet(
                      out / f"oof_fold{fold}.parquet", index=False)
    np.save(out / f"oof_fold{fold}.npy", pred)
    importance = pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "gain": model.booster_.feature_importance("gain"),
        "split": model.booster_.feature_importance("split"),
    })
    importance["gain_pct"] = 100 * importance.gain / importance.gain.sum()
    importance.sort_values("gain", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    summary = {
        "variant": name, "fold": fold, "rmse": rmse,
        "best_iteration": int(model.best_iteration_ or 0),
        "n_train": len(train), "n_val": len(val), "n_features": len(features),
        "threads": threads, "runtime_seconds": time.time() - t0,
        "private_safe": True, "target_oof": True,
        "status": oof_status,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[fold{fold} {name}] {json.dumps(summary)}", flush=True)
    del xtr, xva, model, pred
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--upf-root", type=Path, default=UPF_ROOT)
    parser.add_argument("--threads", type=int, default=25)
    parser.add_argument("--n-estimators", type=int, default=8000)
    parser.add_argument("--early-stopping", type=int, default=200)
    args = parser.parse_args()
    fold_out = args.out_root / f"fold_{args.fold}"
    fold_out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    base_names = json.loads((BASE / "features.json").read_text())
    upf_names = json.loads((args.upf_root / "upf_features.json").read_text())
    a2_names = json.loads((A2_ROOT / "a2_features.json").read_text())
    if (len(base_names), len(upf_names), len(a2_names)) != (249, 10, 49):
        raise RuntimeError("feature count mismatch")
    marker, consistency = fg2_well_features(args.fold)
    fold_features = BASE / "features" / f"fold_{args.fold}"
    train = attach_well_features(pd.read_parquet(fold_features / "Xy_train.parquet"), marker, consistency)
    val = attach_well_features(pd.read_parquet(fold_features / "Xy_val.parquet"), marker, consistency)
    ids = pd.read_parquet(XY, columns=["id"]).id.astype(str)
    positions = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids.to_numpy())
    upf = np.load(args.upf_root / f"upf10_sidecar_outer{args.fold}.npy", mmap_mode="r")
    train = attach_block(train, positions, upf, upf_names)
    val = attach_block(val, positions, upf, upf_names)
    fg2_names = FG1_FEATURES + CONSISTENCY_FEATURES
    features_upf = base_names + fg2_names + upf_names
    summaries = [train_variant("FG2_NUPF", args.fold, train, val, features_upf,
                               args.out_root, args.threads, args.n_estimators, args.early_stopping)]
    a2 = np.load(A2_ROOT / "sidecars" / f"a2_sidecar_outer{args.fold}.npy", mmap_mode="r")
    train = attach_block(train, positions, a2, a2_names)
    val = attach_block(val, positions, a2, a2_names)
    summaries.append(train_variant("FG2_NUPF_A2", args.fold, train, val,
                                   features_upf + a2_names, args.out_root,
                                   args.threads, args.n_estimators, args.early_stopping))
    protocol = {
        "fold": args.fold,
        "definitions": {"FG2_NUPF": "base249+FG2_26+nested_UPF10=285",
                        "FG2_NUPF_A2": "FG2_NUPF+nested_A2_49=334"},
        "upf_source": str(args.upf_root / f"upf10_sidecar_outer{args.fold}.npy"),
        "upf_status": "outer5xinner4 target-OOF comparator; other UPF9 deterministic",
        "a2_source": str(A2_ROOT / "sidecars" / f"a2_sidecar_outer{args.fold}.npy"),
        "a2_status": "nested outer5xinner4", "threads": args.threads,
        "total_runtime_seconds": time.time() - t0, "results": summaries,
    }
    (fold_out / "SUMMARY.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
