#!/usr/bin/env python3
"""Train LightGBM full249+UPF10 on honest fold-OOF full249 features (geo_kmeans).

Reuses:
  OOF/full249_geo5fold_current/features/fold_{k}/Xy_{train,val}.parquet
Joins UPF10 sidecar by id:
  cv/artifacts/.../upf_level_fold1_v1/upf_level_cached_gap6_features.parquet

Does NOT rebuild full249 features. Outputs under OOF/:
  full249_upf_geo5fold_<timestamp>/

protocol: geo_kmeans_5fold + future_only_unknown_segment + private_safe
          + features=fold_oof_honest + upf10_cached_gap6_sidecar
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path("/root/rogii")
OOF_ROOT = ROOT / "OOF"
FOLD_CSV = OOF_ROOT / "geo_kmeans_5fold.csv"
CLEAN_XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
DEFAULT_SRC = OOF_ROOT / "full249_geo5fold_current"
DEFAULT_UPF = (
    ROOT
    / "cv/artifacts/public_train19_lgb5fold_20260709/upf_level_fold1_v1"
    / "upf_level_cached_gap6_features.parquet"
)
DEFAULT_UPF_FEATS = (
    ROOT
    / "cv/artifacts/public_train19_lgb5fold_20260709/upf_level_fold1_v1"
    / "upf_level_cached_gap6_features.features.json"
)
CURRENT_LINK = OOF_ROOT / "full249_upf_geo5fold_current"
SHORT_LINK = OOF_ROOT / "full249_upf"


def diagnose(wells: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    rmse = float(np.sqrt(np.mean(err * err)))
    df = pd.DataFrame({"well": wells, "e2": err * err})
    mw = float(np.sqrt(df.groupby("well", sort=False)["e2"].mean()).mean())
    return {"rmse": rmse, "mean_per_well_rmse": mw}


def attach_upf(df: pd.DataFrame, upf_by_id: pd.DataFrame, upf_feats: list[str]) -> pd.DataFrame:
    side = upf_by_id.loc[df["id"].astype(str), upf_feats].reset_index(drop=True)
    out = pd.concat([df.reset_index(drop=True), side], axis=1)
    missing = [c for c in upf_feats if c not in out.columns]
    if missing:
        raise RuntimeError(f"UPF attach missing columns: {missing}")
    return out


def train_fold(
    *,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats: list[str],
    models_dir: Path,
    n_estimators: int,
    early_stopping: int,
    lgb_jobs: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
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
    t0 = time.time()
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
    elapsed = time.time() - t0
    pred = model.predict(xva, num_iteration=model.best_iteration_).astype(np.float32)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_model = models_dir / f"model_fold{fold}.txt"
    model.booster_.save_model(str(out_model), num_iteration=model.best_iteration_)
    rmse = float(np.sqrt(mean_squared_error(yva, pred)))
    row = {
        "fold": fold,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "best_iteration": int(model.best_iteration_ or 0),
        "rmse": rmse,
        "n_train_wells": int(train_df["well"].nunique()),
        "n_val_wells": int(val_df["well"].nunique()),
        "elapsed_s": float(elapsed),
        "model_path": str(out_model),
    }
    print(
        f"[full249-upf] fold{fold}: rmse={rmse:.6f} best_iter={row['best_iteration']} "
        f"elapsed={elapsed:.0f}s -> {out_model.name}",
        flush=True,
    )
    return pred, row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-root", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--upf-sidecar", type=Path, default=DEFAULT_UPF)
    ap.add_argument("--upf-features-json", type=Path, default=DEFAULT_UPF_FEATS)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n-estimators", type=int, default=8000)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--lgb-jobs", type=int, default=100)
    ap.add_argument("--exp-dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_root = Path(args.feat_root).resolve()
    if not (feat_root / "features").exists() and (feat_root / "fold_0").exists():
        features_dir = feat_root
        feat_json = feat_root.parent / "features.json"
        if not feat_json.exists():
            feat_json = OOF_ROOT / "full249_geo5fold_current" / "features.json"
    else:
        features_dir = feat_root / "features"
        feat_json = feat_root / "features.json"
        if not feat_json.exists():
            feat_json = feat_root / "models" / "features.json"

    if not features_dir.exists():
        raise SystemExit(f"missing features dir: {features_dir}")
    base_feats = json.loads(feat_json.read_text())
    if len(base_feats) != 249:
        raise SystemExit(f"expected 249 base features, got {len(base_feats)}")

    upf_path = Path(args.upf_sidecar)
    upf_feat_json = Path(args.upf_features_json)
    if not upf_path.exists():
        raise SystemExit(f"missing UPF sidecar: {upf_path}")
    if not upf_feat_json.exists():
        raise SystemExit(f"missing UPF features json: {upf_feat_json}")
    upf_feats = json.loads(upf_feat_json.read_text())
    if len(upf_feats) != 10:
        raise SystemExit(f"expected 10 UPF features, got {len(upf_feats)}")
    feats = list(base_feats) + list(upf_feats)

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    for fold in folds:
        for name in ("Xy_train.parquet", "Xy_val.parquet"):
            p = features_dir / f"fold_{fold}" / name
            if not p.exists():
                raise SystemExit(f"missing {p}")

    if args.exp_dir:
        exp = Path(args.exp_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        exp = OOF_ROOT / f"full249_upf_geo5fold_{ts}"
    models_dir = exp / "models"
    oof_dir = exp / "oof"
    models_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    feat_link = exp / "features"
    if feat_link.exists() or feat_link.is_symlink():
        feat_link.unlink()
    feat_link.symlink_to(features_dir)

    (exp / "features.json").write_text(json.dumps(feats, indent=2) + "\n")
    shutil.copy2(upf_feat_json, exp / "upf_features.json")

    protocol = {
        "protocol": (
            "geo_kmeans_5fold + future_only_unknown_segment + private_safe "
            "+ features=fold_oof_honest + upf10_cached_gap6_sidecar"
        ),
        "model": "LGBMRegressor",
        "fold_map": str(FOLD_CSV),
        "feature_source": str(features_dir),
        "feature_source_resolved": str(features_dir.resolve()),
        "upf_sidecar": str(upf_path),
        "n_base_features": 249,
        "n_upf_features": 10,
        "n_features": len(feats),
        "exp_dir": str(exp),
        "params": {
            "n_estimators": args.n_estimators,
            "early_stopping": args.early_stopping,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "lgb_jobs": args.lgb_jobs,
            "seed": args.seed,
        },
        "note": (
            "full249 features reused from honest fold-OOF build; UPF10 joined by id "
            "from cached gap6 sidecar (not fold-rebuilt)."
        ),
    }
    (exp / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (exp / "INFER_PROTOCOL.json").write_text(
        json.dumps(
            {
                "warning": (
                    "Do NOT score these models on the global all-773-fit Xy.parquet. "
                    "Each fold model requires test features rebuilt with that fold's "
                    "train_wells imputers + online UPF sidecar."
                ),
                "oof_eval": (
                    "Xy_val.parquet (fold-honest full249) + UPF10 by id; truth-joined target."
                ),
                "submit_recipe": (
                    "For each fold k: rebuild test full249 with fold_k train_wells imputers, "
                    "build UPF10 sidecar online, predict with models/model_fold{k}.txt; "
                    "average fold preds."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    for link in (CURRENT_LINK, SHORT_LINK):
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(exp)

    print(f"[full249-upf] loading CLEAN_XY ids + UPF sidecar", flush=True)
    xy_ref = pd.read_parquet(CLEAN_XY, columns=["id", "well", "target"])
    id_to_pos = {str(i): k for k, i in enumerate(xy_ref["id"].astype(str))}
    upf = pd.read_parquet(upf_path, columns=upf_feats)
    if len(upf) != len(xy_ref):
        raise SystemExit(f"UPF rows {len(upf)} != Xy rows {len(xy_ref)}")
    upf_by_id = upf.copy()
    upf_by_id.index = xy_ref["id"].astype(str).to_numpy()

    oof = np.full(len(xy_ref), np.nan, dtype=np.float64)
    rows: list[dict] = []
    needed = sorted(set(base_feats + ["id", "well", "target"]))

    for fold in folds:
        fold_dir = features_dir / f"fold_{fold}"
        print(f"[full249-upf] loading fold{fold} from {fold_dir}", flush=True)
        train_df = attach_upf(
            pd.read_parquet(fold_dir / "Xy_train.parquet", columns=needed),
            upf_by_id,
            upf_feats,
        )
        val_df = attach_upf(
            pd.read_parquet(fold_dir / "Xy_val.parquet", columns=needed),
            upf_by_id,
            upf_feats,
        )
        print(
            f"[full249-upf] fold{fold} train={len(train_df)} val={len(val_df)} "
            f"n_feats={len(feats)} lgb_jobs={args.lgb_jobs}",
            flush=True,
        )
        pred, row = train_fold(
            fold=fold,
            train_df=train_df,
            val_df=val_df,
            feats=feats,
            models_dir=models_dir,
            n_estimators=args.n_estimators,
            early_stopping=args.early_stopping,
            lgb_jobs=args.lgb_jobs,
            seed=args.seed,
        )
        rows.append(row)

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
        for sid, p in zip(val_df["id"].astype(str), pred):
            oof[id_to_pos[sid]] = float(p)

    if np.isnan(oof).any():
        raise SystemExit(f"OOF has {int(np.isnan(oof).sum())} NaNs")

    overall = diagnose(
        xy_ref["well"].astype(str).to_numpy(),
        xy_ref["target"].to_numpy(np.float64),
        oof,
    )
    fold_map = pd.read_csv(FOLD_CSV)
    well_fold = fold_map.set_index("well_id")["fold"].astype(int)
    row_fold = xy_ref["well"].astype(str).map(well_fold).to_numpy()
    fold_rmses = {
        f"fold{f}": float(
            np.sqrt(np.mean((oof[row_fold == f] - xy_ref["target"].to_numpy(np.float64)[row_fold == f]) ** 2))
        )
        for f in range(5)
    }
    rows.append({"fold": -1, "n_val": len(xy_ref), **overall, **fold_rmses})

    np.save(models_dir / "oof.npy", oof.astype(np.float32))
    np.save(oof_dir / "oof.npy", oof.astype(np.float32))
    pd.DataFrame(rows).to_csv(models_dir / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(exp / "summary.csv", index=False)

    meta = {
        "name": "full249_upf",
        "overall": overall,
        "fold_rmse": fold_rmses,
        "fold_map": str(FOLD_CSV),
        "exp_dir": str(exp),
        "n_features": len(feats),
        "protocol": protocol["protocol"],
        "baseline_full249_rmse": 8.148731604085338,
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (exp / "DONE.json").write_text(
        json.dumps({"ok": True, "overall": overall, "fold_rmse": fold_rmses}, indent=2) + "\n"
    )
    print("[full249-upf] OVERALL", overall, fold_rmses, "exp=", exp, flush=True)


if __name__ == "__main__":
    main()
