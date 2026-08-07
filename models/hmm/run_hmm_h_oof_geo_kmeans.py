#!/usr/bin/env python3
"""P0: geo_kmeans-aligned HMM-H OOF (baseline / H1 / H2 / H3_shrink).

Uses locked exp224 CUDA FB decode from team notebook cell 3, but replaces
`_prep` with `hmm_h_emission.prep_hmm_h` (sigma-preserving heel adaptation).

protocol: geo_kmeans_5fold + future_only_unknown_segment + private_safe

Example:
  CUDA_VISIBLE_DEVICES=0 python cv/run_hmm_h_oof_geo_kmeans.py shard --variant h1
  python cv/run_hmm_h_oof_geo_kmeans.py merge --variant h1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/root/rogii")
sys.path.insert(0, str(ROOT / "cv"))

from exp249_full249_warp_hmm_cv_20260718 import (  # noqa: E402
    HMM_CONFIG,
    balanced_well_shards,
    diagnostics,
    load_hmm_namespace,
    load_xy,
    make_buckets,
    write_shard_checkpoint,
)
from hmm_h_emission import prep_hmm_h  # noqa: E402

DATA_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
OOF_ROOT = ROOT / "OOF"
FOLD_CSV = OOF_ROOT / "geo_kmeans_5fold.csv"
REF_HMM = OOF_ROOT / "hmm_exp224/oof.npy"


def out_dir_for(variant: str) -> Path:
    return OOF_ROOT / f"hmm_h_{variant}_geo5fold_20260723"


def run_shard(args: argparse.Namespace) -> None:
    variant = args.variant
    out_dir = Path(args.out_dir) if args.out_dir else out_dir_for(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hmm_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"

    xy = load_xy()
    shard_wells = balanced_well_shards(xy, args.num_shards)[args.shard_index]
    shard_mask = xy["well"].isin(shard_wells).to_numpy()
    global_indices = np.flatnonzero(shard_mask)
    pred = np.full(len(global_indices), np.nan, dtype=np.float32)
    std = np.full(len(global_indices), np.nan, dtype=np.float32)
    done = np.zeros(len(global_indices), dtype=bool)
    global_to_local = np.full(len(xy), -1, dtype=np.int64)
    global_to_local[global_indices] = np.arange(len(global_indices), dtype=np.int64)

    namespace, source_sha = load_hmm_namespace()
    source_sha = f"{source_sha}+hmm_h:{variant}"

    def prep_fn(hw, tw, **kw):
        return prep_hmm_h(hw, tw, variant=variant, **kw)

    if args.backend == "numba":
        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError("numba.cuda unavailable")
        device = str(cuda.get_current_device())
        decode_batch = lambda selected: namespace["run_batch_cuda"](selected, tpb=args.tpb)
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA unavailable")
        device = torch.cuda.get_device_name(0)
        decode_batch = lambda selected: namespace["_fb_torch_batch"](selected, "cuda")

    print(
        f"[hmm-h {variant}] shard {args.shard_index}/{args.num_shards} device={device} "
        f"wells={len(shard_wells)} rows={len(global_indices)}",
        flush=True,
    )

    if out_path.exists() and not args.no_resume:
        old = np.load(out_path, allow_pickle=False)
        if not np.array_equal(old["global_indices"], global_indices):
            raise RuntimeError(f"checkpoint cohort mismatch: {out_path}")
        if str(old["hmm_source_sha256"].item()) != source_sha:
            raise RuntimeError(f"checkpoint source mismatch: {out_path}")
        pred[:] = old["pred"]
        std[:] = old["std"]
        done[:] = old["done"]
        print(f"[resume] {int(done.sum())}/{len(done)}", flush=True)

    preps: list[dict[str, Any]] = []
    meta_rows = []
    t0 = time.time()
    for wi, well in enumerate(shard_wells, 1):
        well_global = np.flatnonzero(xy["well"].to_numpy() == well)
        well_local = global_to_local[well_global]
        if bool(done[well_local].all()):
            continue
        hw = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        hidden = hw["TVT_input"].isna().to_numpy()
        hidden_rows = np.flatnonzero(hidden)
        expected_ids = np.asarray([f"{well}_{row}" for row in hidden_rows], dtype=object)
        observed_ids = xy.iloc[well_global]["id"].to_numpy(dtype=object)
        if not np.array_equal(expected_ids, observed_ids):
            raise RuntimeError(f"Xy/HMM row alignment mismatch for well {well}")
        last_tvt = float(pd.to_numeric(hw.loc[~hidden, "TVT_input"], errors="coerce").iloc[-1])
        truth = pd.to_numeric(hw.loc[hidden, "TVT"], errors="coerce").to_numpy(np.float64) - last_tvt
        xy_truth = xy.iloc[well_global]["target"].to_numpy(np.float64)
        if not np.allclose(truth, xy_truth, atol=2e-5, rtol=0.0):
            raise RuntimeError(f"Xy target mismatch for well {well}")

        prep = prep_fn(hw, tw, **HMM_CONFIG)
        if prep is None:
            pred[well_local] = 0.0
            std[well_local] = 0.0
            done[well_local] = True
            continue
        if not np.array_equal(np.asarray(prep["ev_index"], dtype=int), hidden_rows):
            raise RuntimeError(f"HMM ev_index mismatch for well {well}")
        prep["_wid"] = well
        prep["_last_tvt"] = last_tvt
        prep["_local_indices"] = well_local
        preps.append(prep)
        meta_rows.append(
            {
                "well": well,
                "variant": variant,
                "gs": prep.get("_gs"),
                "gs_raw": prep.get("_gs_raw"),
                "a_use": prep.get("_a_use"),
                "b_use": prep.get("_b_use"),
            }
        )
        if wi % 25 == 0:
            print(f"[prep] {wi}/{len(shard_wells)} elapsed={time.time()-t0:.1f}s", flush=True)

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(out_dir / f"prep_meta_shard{args.shard_index}.csv", index=False)

    buckets = make_buckets(preps, args.budget_gb)
    print(f"[decode] wells={len(preps)} buckets={len(buckets)}", flush=True)
    started = time.time()
    for bi, bucket in enumerate(buckets, 1):
        selected = [preps[i] for i in bucket]
        # strip diagnostic keys that might confuse kernels? Keep them — kernels only use known keys.
        outputs = decode_batch(selected)
        for prep, (mean, posterior_std) in zip(selected, outputs):
            local = prep["_local_indices"]
            drift = np.asarray(mean, dtype=np.float64) - float(prep["_last_tvt"])
            if len(drift) != len(local) or not np.isfinite(drift).all():
                raise RuntimeError(f"invalid HMM output for well {prep['_wid']}")
            pred[local] = drift.astype(np.float32)
            std[local] = np.asarray(posterior_std, dtype=np.float32)
            done[local] = True
        write_shard_checkpoint(
            out_path,
            global_indices=global_indices,
            pred=pred,
            std=std,
            done=done,
            wells=shard_wells,
            source_sha=source_sha,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            backend=args.backend,
        )
        print(
            f"[decode] bucket={bi}/{len(buckets)} rows={int(done.sum())}/{len(done)} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    if not bool(done.all()) or not np.isfinite(pred).all():
        raise RuntimeError(f"incomplete shard: done={int(done.sum())}/{len(done)}")
    print(f"[done] {out_path}", flush=True)


def merge(args: argparse.Namespace) -> None:
    variant = args.variant
    out_dir = Path(args.out_dir) if args.out_dir else out_dir_for(variant)
    xy = load_xy()
    wells = xy["well"].to_numpy()
    truth = xy["target"].to_numpy(np.float64)
    hmm = np.full(len(xy), np.nan, dtype=np.float32)
    coverage = np.zeros(len(xy), dtype=np.int8)
    for shard_index in range(args.num_shards):
        path = out_dir / f"hmm_shard_{shard_index:02d}_of_{args.num_shards:02d}.npz"
        data = np.load(path, allow_pickle=False)
        if not bool(data["done"].all()):
            raise RuntimeError(f"incomplete shard: {path}")
        indices = data["global_indices"].astype(np.int64)
        if bool((coverage[indices] != 0).any()):
            raise RuntimeError(f"overlapping shard: {path}")
        coverage[indices] = 1
        hmm[indices] = data["pred"].astype(np.float32)
    if not bool((coverage == 1).all()) or not np.isfinite(hmm).all():
        raise RuntimeError("incomplete merge")

    diag = diagnostics(wells, hmm, truth)
    fm = pd.read_csv(FOLD_CSV).set_index("well_id")["fold"].astype(int)
    row_fold = np.array([int(fm.loc[w]) for w in wells], dtype=np.int8)
    fold_rmse = {
        f"fold{f}": float(np.sqrt(np.mean((hmm[row_fold == f] - truth[row_fold == f]) ** 2)))
        for f in range(5)
    }
    np.save(out_dir / "oof.npy", hmm.astype(np.float32))
    (out_dir / "models").mkdir(exist_ok=True)
    np.save(out_dir / "models" / "oof.npy", hmm.astype(np.float32))

    ref = {}
    if REF_HMM.exists():
        ref_arr = np.load(REF_HMM).astype(np.float64)
        ref = {
            "vs_hmm_exp224_max_abs": float(np.max(np.abs(hmm.astype(np.float64) - ref_arr))),
            "vs_hmm_exp224_rmse_delta": diag["rmse"]
            - float(np.sqrt(np.mean((ref_arr - truth) ** 2))),
            "pearson_vs_hmm_exp224": float(
                np.corrcoef(hmm.astype(np.float64), ref_arr)[0, 1]
            ),
        }

    summary = {
        "protocol": "geo_kmeans_5fold + future_only_unknown_segment + private_safe",
        "arm": f"hmm_h_{variant}",
        "variant": variant,
        "note": (
            "H1/H2 freeze gs_raw on raw typewell; only emission reference changes. "
            "H3_shrink is negative control (forbidden by writeup)."
        ),
        "hmm_config": HMM_CONFIG,
        "overall": diag,
        "fold_rmse": fold_rmse,
        "reference": ref,
        "n_rows": int(len(xy)),
        "fold_map": str(FOLD_CSV),
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "PROTOCOL.json").write_text(
        json.dumps(
            {
                "protocol": summary["protocol"],
                "variant": variant,
                "private_safe": True,
                "uses_unknown_truth": False,
            },
            indent=2,
        )
        + "\n"
    )
    link = OOF_ROOT / f"hmm_h_{variant}"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(out_dir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[merge] wrote {out_dir / 'oof.npy'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("shard")
    sp.add_argument("--variant", choices=["baseline", "h1", "h2", "h3_shrink"], required=True)
    sp.add_argument("--shard-index", type=int, default=0)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.add_argument("--backend", choices=["numba", "torch"], default="numba")
    sp.add_argument("--tpb", type=int, default=128)
    sp.add_argument("--budget-gb", type=float, default=12.0)
    sp.add_argument("--out-dir", type=str, default="")
    sp.add_argument("--no-resume", action="store_true")

    mp = sub.add_parser("merge")
    mp.add_argument("--variant", choices=["baseline", "h1", "h2", "h3_shrink"], required=True)
    mp.add_argument("--num-shards", type=int, default=1)
    mp.add_argument("--out-dir", type=str, default="")

    args = ap.parse_args()
    if args.cmd == "shard":
        run_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
