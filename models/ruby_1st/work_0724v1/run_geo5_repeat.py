#!/usr/bin/env python3
"""Run one Ruby CV repeat against the canonical fixed Geo5 fold map."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import seq_NN_train


_GEO_CHANNELS = {
    "geo_s_rel",
    "geo_tvt_rel",
    "geo_tvt_abs_diff",
    "geo_dS",
    "geo_tvt_diff",
    "delta_geo_trend_rbf_5",
}


def no_xy_geo_prior(_support_path, query_path, _support_wells, query_wells, _geo_cfg, **_kwargs):
    """Return a local, label-free placeholder required by the archived loader.

    The no-XY experiment removes every geo channel before training.  The
    original Dataset still requires a per-well ``geo_prior`` object even when
    no feature consumes it, so provide only zero arrays whose lengths are read
    from the query horizontal input.  No neighbor search or target TVT is read.
    """

    started = __import__("time").perf_counter()
    priors = {}
    for well_id in query_wells:
        path = Path(query_path) / f"{well_id}__horizontal_well.csv"
        n_rows = len(pd.read_csv(path, usecols=["MD"]))
        zeros = np.zeros(n_rows, dtype=np.float32)
        nans = np.full(n_rows, np.nan, dtype=np.float32)
        priors[str(well_id)] = {
            "S_rel_prior": zeros,
            "geo_nbr_std": zeros,
            "geo_nbr_dS_std": zeros,
            "geo_radial_extrap_score": nans,
            "geo_nbr_distance": nans,
            "geo_nbr_path_alignment": nans,
        }
    return priors, {"rmse": None, "elapsed_sec": __import__("time").perf_counter() - started}


def disable_xy_neighbor_features(cfg):
    """Turn an archived XY recipe into an explicitly no-XY ablation.

    Keep the version-specific CNN, losses, PF channels, augmentation and
    post-processing unchanged.  ``z_diff`` is the original non-spatial
    replacement for the main ``geo_tvt_diff`` channel.
    """

    channels = [name for name in cfg.unet_static_channels if name not in _GEO_CHANNELS]
    if "z_diff" not in channels:
        channels.append("z_diff")
    cfg.unet_static_channels = tuple(channels)
    cfg.two_stage_shared_channels = tuple(
        name for name in cfg.two_stage_shared_channels if name not in _GEO_CHANNELS
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pf-workers", type=int, default=16)
    parser.add_argument(
        "--use-xy-neighbor",
        action="store_true",
        help="Opt in to the archived XY-neighbor prior. Default is the requested no-XY ablation.",
    )
    return parser.parse_args()


def fixed_geo5_splits(well_ids, cfg, log):
    fold_map = pd.read_csv(cfg.cv_geo_map_path, dtype={"well_id": str})
    required = {"well_id", "fold"}
    if not required.issubset(fold_map.columns):
        raise ValueError(f"fixed Geo5 map needs {required}, got {fold_map.columns.tolist()}")
    if fold_map["well_id"].duplicated().any():
        raise ValueError("fixed Geo5 map has duplicated well_id")

    well_ids = np.asarray(well_ids, dtype=str)
    fold_by_well = fold_map.set_index("well_id")["fold"]
    missing = sorted(set(well_ids) - set(fold_by_well.index))
    extra = sorted(set(fold_by_well.index) - set(well_ids))
    if missing or extra:
        raise ValueError(f"fixed Geo5 map mismatch: missing={missing[:5]} extra={extra[:5]}")

    assigned = fold_by_well.loc[well_ids].to_numpy(dtype=int)
    expected = np.arange(int(cfg.fold_count), dtype=int)
    observed = np.unique(assigned)
    if not np.array_equal(observed, expected):
        raise ValueError(f"fixed Geo5 folds mismatch: expected={expected.tolist()} got={observed.tolist()}")

    digest = hashlib.sha256(Path(cfg.cv_geo_map_path).read_bytes()).hexdigest()
    counts = {int(fold): int((assigned == fold).sum()) for fold in expected}
    log(f"fixed_geo5: map={cfg.cv_geo_map_path} sha256={digest} counts={counts}")
    return [
        (np.flatnonzero(assigned != fold), np.flatnonzero(assigned == fold))
        for fold in expected
    ]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.cfg.open("rb") as handle:
        cfg = pickle.load(handle)

    original_seed = int(cfg.seed)
    original_fold_count = int(cfg.fold_count)
    original_repeats = int(cfg.cv_repeats)
    cfg.data_dir = args.data_dir.resolve()
    cfg.train_path = cfg.data_dir / "train"
    cfg.test_path = cfg.data_dir / "test"
    cfg.output_dir = args.output_dir.resolve()
    cfg.cv_geo_map_path = args.fold_map.resolve()
    cfg.cv_split_mode = "fixed_geo5"
    cfg.cv_repeats = 1
    cfg.seed = original_seed + args.repeat * original_fold_count
    cfg.num_workers = int(args.num_workers)
    cfg.PF_heatmap_num_workers = int(args.pf_workers)
    cfg.PF_heatmap_cache_dir = cfg.output_dir / "pf_heatmap_cache"
    cfg.PF_sample_cache_dir = cfg.output_dir / "pf_sample_cache"
    if not args.use_xy_neighbor:
        disable_xy_neighbor_features(cfg)
    cfg.refresh()

    # Keep Ruby's fold loop, but default to a label-free no-XY data contract.
    seq_NN_train.make_cv_splits = fixed_geo5_splits
    if not args.use_xy_neighbor:
        seq_NN_train.make_geo_prior_for_wells = no_xy_geo_prior
        seq_NN_train.add_geo_prior_diagnostic_columns = lambda df, _prior: df
        seq_NN_train.add_geo_prior_well_summary_columns = lambda df: df
    print(
        json.dumps(
            {
                "protocol": "Ruby recipe with canonical fixed Geo5 split",
                "repeat": args.repeat,
                "original_cv_repeats": original_repeats,
                "effective_cv_repeats": cfg.cv_repeats,
                "seed": cfg.seed,
                "fold_map": str(cfg.cv_geo_map_path),
                "data_dir": str(cfg.data_dir),
                "xy_neighbor_features_enabled": bool(args.use_xy_neighbor),
                "xy_neighbor_clustering_for_cv": False,
                "xy_neighbor_labels_read": bool(args.use_xy_neighbor),
                "static_channels": list(cfg.unet_static_channels),
            },
            indent=2,
        ),
        flush=True,
    )
    well_ids = seq_NN_train.discover_well_ids(cfg.train_path)
    models, oof_df = seq_NN_train.kfold_training(well_ids=well_ids, cfg=cfg, log=print)
    oof_path = cfg.output_dir / f"oof_repeat{args.repeat}.parquet"
    oof_df.to_parquet(oof_path, index=False)
    score = seq_NN_train.score_prediction_df(oof_df)
    manifest = {
        "repeat": args.repeat,
        "seed": cfg.seed,
        "oof_path": str(oof_path),
        "oof_rows": int(len(oof_df)),
        "oof_rmse": float(score),
        "fold_map": str(cfg.cv_geo_map_path),
        "fold_sha256": hashlib.sha256(Path(cfg.cv_geo_map_path).read_bytes()).hexdigest(),
        "models_discarded_after_oof": len(models),
    }
    (cfg.output_dir / "repeat_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
