#!/usr/bin/env python3
"""Average the three fixed-Geo5 Ruby repeat OOF files and report CV."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

import seq_NN_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.cfg.open("rb") as handle:
        cfg = pickle.load(handle)
    cfg.data_dir = args.data_dir.resolve()
    cfg.train_path = cfg.data_dir / "train"
    cfg.test_path = cfg.data_dir / "test"
    cfg.cv_geo_map_path = args.fold_map.resolve()
    cfg.cv_repeats = 3
    cfg.output_dir = args.output_dir.resolve()
    cfg.refresh()

    parts = []
    repeat_scores = {}
    for repeat in range(3):
        path = args.repeat_root / f"repeat{repeat}" / f"oof_repeat{repeat}.parquet"
        frame = pd.read_parquet(path)
        parts.append(frame)
        repeat_scores[str(repeat)] = float(seq_NN_train.score_prediction_df(frame))
    merged = pd.concat(parts, ignore_index=True)
    averaged = seq_NN_train.average_repeated_oof_predictions(merged, cfg, log=print)
    # No-XY runs intentionally omit geo diagnostics from their OOF frames.
    if "geo_nbr_distance" in averaged.columns:
        averaged = seq_NN_train.add_geo_prior_well_summary_columns(averaged)
    overall_rmse = float(seq_NN_train.score_prediction_df(averaged))
    per_well = seq_NN_train.well_rmse_summary(averaged)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    averaged.to_parquet(args.output_dir / "oof_geo5_3repeat.parquet", index=False)
    per_well.to_csv(args.output_dir / "oof_geo5_3repeat_per_well.csv", index=False)
    summary = {
        "protocol": "Ruby recipe, canonical fixed Geo5, three original repeat seeds",
        "fold_map": str(cfg.cv_geo_map_path),
        "fold_sha256": hashlib.sha256(Path(cfg.cv_geo_map_path).read_bytes()).hexdigest(),
        "repeat_rmse": repeat_scores,
        "oof_rmse": overall_rmse,
        "n_rows": int(len(averaged)),
        "n_wells": int(averaged["well_id"].nunique()),
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
