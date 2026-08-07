#!/usr/bin/env python3
"""Generate strict held-out-fold exp207 WARP predictions for all 773 wells."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/rogii")
DATA_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
XY_PATH = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
FOLDS_PATH = ROOT / "cv/sub6_run/artifacts/folds5.npz"
WARP_DIR = ROOT / "my_explainer_submissions/datasets/exp207-warp"
DEFAULT_OUT = ROOT / "cv/artifacts/exp207_honest_warp_oof_20260718"

sys.path.insert(0, str(WARP_DIR))
import deploy_warp as warp  # noqa: E402


def load_fold_by_well(xy: pd.DataFrame) -> dict[str, int]:
    folds = np.load(FOLDS_PATH)
    fold_by_well: dict[str, int] = {}
    for fold in range(5):
        valid = folds[f"va_{fold}"]
        for well in xy.iloc[valid]["well"].astype(str).unique():
            previous = fold_by_well.setdefault(str(well), fold)
            if previous != fold:
                raise RuntimeError(f"well {well} appears in multiple validation folds")
    if len(fold_by_well) != 773:
        raise RuntimeError(f"fold mapping covers {len(fold_by_well)} wells, expected 773")
    return fold_by_well


@torch.no_grad()
def predict_one_model(model_tuple, features: dict, device: torch.device) -> np.ndarray:
    model, fmean, fstd, tmean, tstd = model_tuple
    feat = ((features["features"] - fmean) / fstd).astype(np.float32)
    tw = ((features["tw_tokens"] - tmean) / tstd).astype(np.float32)
    x = torch.from_numpy(feat.T).unsqueeze(0).to(device)
    twt = torch.from_numpy(tw).unsqueeze(0).to(device)
    return model(x, twt)[0].cpu().numpy() + float(features["last_tvt"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.npz"

    xy = pd.read_parquet(XY_PATH, columns=["well", "id", "target"])
    xy["well"] = xy["well"].astype(str)
    xy["id"] = xy["id"].astype(str)
    wells = xy["well"].to_numpy()
    truth = xy["target"].to_numpy(np.float64)
    fold_by_well = load_fold_by_well(xy)
    unique_wells = list(xy["well"].unique())
    indices_by_well = {
        str(well): group.index.to_numpy(np.int64)
        for well, group in xy.groupby("well", sort=False)
    }

    oof = np.full(len(xy), np.nan, dtype=np.float32)
    done_wells: set[str] = set()
    if checkpoint.exists() and not args.no_resume:
        old = np.load(checkpoint, allow_pickle=False)
        if len(old["oof"]) != len(xy):
            raise RuntimeError("checkpoint cohort mismatch")
        oof[:] = old["oof"]
        done_wells = set(old["done_wells"].astype(str).tolist())
        print(f"[resume] wells={len(done_wells)}/773", flush=True)

    models = warp.load_bundle(WARP_DIR / "warp_raw_bundle.pt", device)
    if len(models) != 5:
        raise RuntimeError(f"expected 5 WARP models, got {len(models)}")
    print(f"[warp] models={len(models)} threads={torch.get_num_threads()} device={device}", flush=True)

    started = time.time()
    pending_count = 0
    for wi, well in enumerate(unique_wells, 1):
        if well in done_wells:
            continue
        indices = indices_by_well[well]
        hw = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        features = warp.build_gr_features_one(hw, tw)
        if features is None:
            pred_drift = np.zeros(len(indices), dtype=np.float32)
        else:
            hidden_rows = np.asarray(features["ev_index"], dtype=int)
            expected_ids = np.asarray([f"{well}_{row}" for row in hidden_rows], dtype=object)
            if not np.array_equal(expected_ids, xy.iloc[indices]["id"].to_numpy(dtype=object)):
                raise RuntimeError(f"row alignment mismatch for {well}")
            fold = fold_by_well[well]
            pred_abs = predict_one_model(models[fold], features, device)
            pred_drift = np.asarray(pred_abs - float(features["last_tvt"]), dtype=np.float32)
        if len(pred_drift) != len(indices) or not np.isfinite(pred_drift).all():
            raise RuntimeError(f"invalid prediction for {well}")
        oof[indices] = pred_drift
        done_wells.add(well)
        pending_count += 1
        if pending_count % int(args.checkpoint_every) == 0 or wi == len(unique_wells):
            np.savez_compressed(
                checkpoint,
                oof=oof,
                done_wells=np.asarray(sorted(done_wells), dtype="U16"),
            )
            completed = np.isfinite(oof)
            partial_rmse = float(np.sqrt(np.mean((oof[completed] - truth[completed]) ** 2)))
            print(
                f"[warp] wells={len(done_wells)}/773 rows={int(completed.sum())}/{len(oof)} "
                f"partial_rmse={partial_rmse:.6f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    if len(done_wells) != 773 or not np.isfinite(oof).all():
        raise RuntimeError(f"incomplete OOF: wells={len(done_wells)} finite={int(np.isfinite(oof).sum())}")
    rmse = float(np.sqrt(np.mean((oof.astype(np.float64) - truth) ** 2)))
    reference_difference = abs(rmse - 11.3089)
    mapping_validated = reference_difference <= 0.02
    output_name = "warp_exp207_honest_oof.npy" if mapping_validated else "warp_exp207_localfold_candidate.npy"
    np.save(out_dir / output_name, oof)
    fold_rows = []
    folds = np.load(FOLDS_PATH)
    for fold in range(5):
        valid = folds[f"va_{fold}"]
        fold_rows.append(
            {
                "fold": fold + 1,
                "n_rows": len(valid),
                "rmse": float(np.sqrt(np.mean((oof[valid].astype(np.float64) - truth[valid]) ** 2))),
            }
        )
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rmse": rmse,
        "n_rows": len(oof),
        "n_wells": len(done_wells),
        "semantics": "one held-out bundle model per validation well; fixed GroupKFold mapping",
        "bundle_fold_key_assumption": "bundle key equals folds5 validation fold index",
        "teammate_reference_rmse": 11.3089,
        "absolute_difference_from_reference": reference_difference,
        "fold_mapping_validated": mapping_validated,
        "promotion_status": "validated_honest_oof" if mapping_validated else "quarantined_fold_mapping_mismatch",
        "output_file": output_name,
        "torch_threads": torch.get_num_threads(),
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] honest WARP RMSE={rmse:.9f} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
