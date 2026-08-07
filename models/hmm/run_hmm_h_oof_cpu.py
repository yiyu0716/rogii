#!/usr/bin/env python3
"""P0 HMM-H OOF on CPU (exp224-equivalent FB via numba), when CUDA is unavailable.

Uses prep_hmm_h emission + exact_hmm_posterior_source._hmm2_fb with locked HMM_CONFIG.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/root/rogii")
DATA = ROOT / "datasets/rogii-wellbore-geology-prediction"
XY = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
FOLD = ROOT / "OOF/geo_kmeans_5fold.csv"
REF = ROOT / "OOF/hmm_exp224/oof.npy"

import sys

sys.path.insert(0, str(ROOT / "cv"))
from exact_hmm_posterior_source import _hmm2_fb  # noqa: E402
from exp249_full249_warp_hmm_cv_20260718 import HMM_CONFIG, diagnostics  # noqa: E402
from hmm_h_emission import prep_hmm_h  # noqa: E402


def decode_one(well: str, variant: str) -> tuple[str, np.ndarray, dict]:
    hw = pd.read_csv(DATA / "train" / f"{well}__horizontal_well.csv")
    tw = pd.read_csv(DATA / "train" / f"{well}__typewell.csv")
    prep = prep_hmm_h(hw, tw, variant=variant, **HMM_CONFIG)
    if prep is None:
        return well, np.zeros(0, np.float64), {"ok": False}
    last_tvt = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])
    post_p, _loglik = _hmm2_fb(
        prep["em"].astype(np.float32),
        prep["dm"].astype(np.float64),
        prep["dz"].astype(np.float64),
        float(prep["sp"]),
        prep["rates"].astype(np.float64),
        float(prep["sig_r"]),
        float(prep["sig_p"]),
        float(prep["start_p"]),
        float(prep["start_sig"]),
        float(prep["r0"]),
        float(prep["r0_sig"]),
        float(prep["lam"]),
        float(prep["mom"]),
    )
    mean_abs = post_p @ prep["grid"]
    drift = mean_abs.astype(np.float64) - last_tvt
    meta = {
        "ok": True,
        "gs": prep.get("_gs"),
        "gs_raw": prep.get("_gs_raw"),
        "a_use": prep.get("_a_use"),
        "b_use": prep.get("_b_use"),
        "T": int(len(prep["dm"])),
        "P": int(prep["em"].shape[1]),
    }
    return well, drift, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "h1", "h2", "h3_shrink"], required=True)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else ROOT / "OOF" / f"hmm_h_{args.variant}_geo5fold_20260723"
    out.mkdir(parents=True, exist_ok=True)

    xy = pd.read_parquet(XY, columns=["well", "id", "target"])
    wells_u = xy["well"].astype(str).unique().tolist()
    if args.max_wells > 0:
        wells_u = wells_u[: args.max_wells]

    print(f"[hmm-h-cpu {args.variant}] wells={len(wells_u)} n_jobs={args.n_jobs}", flush=True)
    t0 = time.time()
    # JIT warm
    decode_one(wells_u[0], args.variant)
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(decode_one)(w, args.variant) for w in wells_u
    )
    print(f"[hmm-h-cpu] decode {time.time()-t0:.1f}s", flush=True)

    oof = np.full(len(xy), np.nan, np.float64)
    metas = []
    by = {w: (d, m) for w, d, m in results}
    for well, g in xy.groupby("well", sort=False):
        drift, meta = by[str(well)]
        idx = g.index.to_numpy()
        if len(drift) != len(idx):
            raise RuntimeError(f"len mismatch {well}")
        oof[idx] = drift
        metas.append({"well": well, **meta})
    if np.isnan(oof).any():
        raise RuntimeError("NaNs")

    wells = xy["well"].astype(str).to_numpy()
    y = xy["target"].to_numpy(np.float64)
    diag = diagnostics(wells, oof.astype(np.float32), y)
    fm = pd.read_csv(FOLD).set_index("well_id")["fold"].astype(int)
    row_fold = np.array([int(fm.loc[w]) for w in wells], dtype=np.int8)
    fold_rmse = {
        f"fold{f}": float(np.sqrt(np.mean((oof[row_fold == f] - y[row_fold == f]) ** 2)))
        for f in range(5)
    }

    ref = {}
    if REF.exists() and args.max_wells <= 0:
        r = np.load(REF).astype(np.float64)
        ref = {
            "vs_hmm_exp224_max_abs": float(np.max(np.abs(oof - r))),
            "vs_hmm_exp224_rmse_delta": diag["rmse"] - float(np.sqrt(np.mean((r - y) ** 2))),
            "pearson_vs_hmm_exp224": float(np.corrcoef(oof, r)[0, 1]),
        }

    np.save(out / "oof.npy", oof.astype(np.float32))
    pd.DataFrame(metas).to_csv(out / "prep_meta.csv", index=False)
    summary = {
        "protocol": "geo_kmeans_5fold + future_only_unknown_segment + private_safe",
        "arm": f"hmm_h_{args.variant}",
        "variant": args.variant,
        "decode": "cpu_numba_hmm2_fb",
        "hmm_config": HMM_CONFIG,
        "overall": diag,
        "fold_rmse": fold_rmse,
        "reference": ref,
        "n_rows": int(len(xy)),
        "note": "CUDA unavailable on host; CPU FB with identical emission prep + exp224 HMM_CONFIG",
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    link = ROOT / "OOF" / f"hmm_h_{args.variant}"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
