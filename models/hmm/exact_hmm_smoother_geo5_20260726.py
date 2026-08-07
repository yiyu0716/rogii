#!/usr/bin/env python3
"""Exact second-order HMM smoother OOF arm under geo_kmeans_5fold.

Reproduces the Kaggle exact-hmm-smoother notebook (OU dip-rate dynamics +
autocorr-aware Gaussian emission, exact forward-backward posterior mean).

Fold-independent per-well decode (private-safe: known prefix + GR + trajectory
+ typewell only).  Outputs drift = posterior_mean_TVT - last_known_TVT aligned
to canonical Xy / repro_1086 row order.

protocol: geo_kmeans_5fold + future_only_unknown_segment + private_safe

Example (high load → n_jobs<=16):
  python cv/exact_hmm_smoother_geo5_20260726.py --fold 0 --n-jobs 16
  python cv/exact_hmm_smoother_geo5_20260726.py --n-jobs 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/root/rogii")
sys.path.insert(0, str(ROOT / "cv"))

from exact_hmm_posterior_source import DATA_ROOT, run_hmm2_frame  # noqa: E402

XY_PATH = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709/full_train19_cache/Xy.parquet"
FOLD_CSV = ROOT / "OOF/geo_kmeans_5fold.csv"
CANON = ROOT / "experiments/repro_1086"
OUT_DEFAULT = ROOT / "OOF/exact_hmm_smoother_geo5_20260726"
REF_HMM224 = ROOT / "OOF/OOF_exp265/hmm_d.npy"

# Validated notebook defaults (tmp_kaggle_read/exact_hmm_smoother CFG.HMM)
HMM_CFG = {
    "step": 0.35,
    "n_rates": 41,
    "rate_span": 0.10,
    "sig_r": 0.002,
    "sig_p": 0.02,
    "mom": 0.998,
    "lam": 1.0,
    "emission": "gauss",
    "sigma_mode": "std",
    "start_sig": 0.75,
    "r0_sig": 0.01,
    "band_pad": 100.0,
    "rate_center": "zero",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def diagnose(wells: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred.astype(np.float64) - y.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err * err)))
    df = pd.DataFrame({"well": wells, "e2": err * err})
    mw = float(np.sqrt(df.groupby("well")["e2"].mean()).mean())
    mean = df.groupby("well")[["e2"]].transform("mean")
    t0 = y - pd.Series(y).groupby(wells).transform("mean").to_numpy()
    p0 = pred - pd.Series(pred).groupby(wells).transform("mean").to_numpy()
    level = float(np.sqrt(np.mean((pd.Series(y).groupby(wells).mean() - pd.Series(pred).groupby(wells).mean()) ** 2)))
    swing = float(np.sqrt(np.mean((t0 - p0) ** 2)))
    return {
        "rmse": rmse,
        "mean_per_well_rmse": mw,
        "level_rmse": level,
        "swing_rmse": swing,
    }


def decode_one(well: str, data_root: Path) -> tuple[str, np.ndarray, np.ndarray, dict]:
    hpath = data_root / "train" / f"{well}__horizontal_well.csv"
    twpath = data_root / "train" / f"{well}__typewell.csv"
    hw = pd.read_csv(hpath)
    tw = pd.read_csv(twpath)
    hidden = hw["TVT_input"].isna().to_numpy()
    known = ~hidden
    if int(hidden.sum()) < 1 or int(known.sum()) < 8:
        return well, np.zeros(0, np.float64), np.zeros(0, np.float64), {"ok": False, "reason": "short"}
    last_tvt = float(pd.to_numeric(hw.loc[known, "TVT_input"], errors="coerce").iloc[-1])
    try:
        res = run_hmm2_frame(hw, tw, **HMM_CFG)
    except Exception as exc:
        return well, np.zeros(0, np.float64), np.zeros(0, np.float64), {"ok": False, "reason": repr(exc)}
    ev_index = np.flatnonzero(hidden)
    if not np.array_equal(ev_index, res.ev_index):
        return well, np.zeros(0, np.float64), np.zeros(0, np.float64), {"ok": False, "reason": "ev_index_mismatch"}
    drift = np.asarray(res.pred, dtype=np.float64)[ev_index] - last_tvt
    std = np.asarray(res.std_eval, dtype=np.float64)
    meta = {
        "ok": True,
        "reason": "ok",
        "last_tvt": last_tvt,
        "n_eval": int(len(drift)),
        "loglik": float(res.loglik),
        "std_mean": float(np.nanmean(std)) if len(std) else np.nan,
        "T": int(len(std)),
        "P": int(res.mean_eval.shape[0]) if hasattr(res, "mean_eval") else int(len(std)),
    }
    return well, drift, std, meta


def load_fold_map() -> dict[str, int]:
    fm = pd.read_csv(FOLD_CSV)
    return {str(r.well_id): int(r.fold) for r in fm.itertuples(index=False)}


def select_wells(fold: int | None, max_wells: int) -> list[str]:
    xy = pd.read_parquet(XY_PATH, columns=["well"])
    wells = xy["well"].astype(str).unique().tolist()
    if fold is not None:
        fmap = load_fold_map()
        wells = [w for w in wells if fmap.get(w) == fold]
    if max_wells > 0:
        wells = wells[:max_wells]
    return wells


def assemble_oof(results: list[tuple[str, np.ndarray, np.ndarray, dict]], xy: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(xy), np.nan, dtype=np.float64)
    std_full = np.full(len(xy), np.nan, dtype=np.float64)
    by_well = {w: (d, s, m) for w, d, s, m in results}
    for well, grp in xy.groupby("well", sort=False):
        w = str(well)
        if w not in by_well:
            continue
        drift, std, meta = by_well[w]
        if not meta.get("ok"):
            if len(drift) == 0:
                continue
            raise RuntimeError(f"failed well {w}: {meta.get('reason')}")
        idx = grp.index.to_numpy()
        if len(drift) != len(idx):
            raise RuntimeError(f"length mismatch well={w} drift={len(drift)} xy={len(idx)}")
        oof[idx] = drift
        if len(std) == len(idx):
            std_full[idx] = std
    return oof, std_full


def write_outputs(
    out_dir: Path,
    oof: np.ndarray,
    std_full: np.ndarray,
    wells_done: list[str],
    fold_filter: int | None,
    metas: list[dict],
    elapsed_s: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = np.load(CANON / "truth.npy").astype(np.float64)
    wells = np.load(CANON / "wells.npy", allow_pickle=True).astype(str)
    folds = np.load(CANON / "folds.npy").astype(np.int8)
    assert len(oof) == len(truth) == 3_783_989

    mask = np.isfinite(oof)
    n_done = int(mask.sum())
    status = "full" if n_done == len(oof) else "partial"

    overall = diagnose(wells[mask], truth[mask], oof[mask]) if n_done else {}
    per_fold = []
    for f in range(5):
        m = mask & (folds == f)
        if not int(m.sum()):
            continue
        per_fold.append({"fold": f, "n_rows": int(m.sum()), **diagnose(wells[m], truth[m], oof[m])})

    ref = {}
    if REF_HMM224.exists() and status == "full":
        ref_arr = np.load(REF_HMM224).astype(np.float64)
        ref = {
            "vs_hmm_exp224_rmse": float(np.sqrt(np.mean((ref_arr - truth) ** 2))),
            "vs_hmm_exp224_rmse_delta": overall.get("rmse", np.nan)
            - float(np.sqrt(np.mean((ref_arr - truth) ** 2))),
            "pearson_vs_hmm_exp224": float(np.corrcoef(oof, ref_arr)[0, 1]),
            "max_abs_delta_vs_hmm_exp224": float(np.max(np.abs(oof - ref_arr))),
        }

    oof_path = out_dir / "exact_hmm_d.npy"
    std_path = out_dir / "exact_hmm_std.npy"
    np.save(oof_path, np.where(mask, oof, np.nan).astype(np.float32))
    np.save(std_path, np.where(np.isfinite(std_full), std_full, np.nan).astype(np.float32))
    pd.DataFrame(metas).to_csv(out_dir / "well_meta.csv", index=False)

    protocol = {
        "protocol": "geo_kmeans_5fold + future_only_unknown_segment + private_safe",
        "arm": "exact_hmm_smoother",
        "source_notebook": "tmp_kaggle_read/exact_hmm_smoother/rogii-wellbore-geology-exact-hmm-smoother.py",
        "implementation": "cv/exact_hmm_posterior_source.run_hmm2_frame + exact_hmm_smoother_geo5_20260726.py",
        "hmm_config": HMM_CFG,
        "fold_map": str(FOLD_CSV),
        "fold_map_sha256": sha256(FOLD_CSV),
        "truth_path": str(CANON / "truth.npy"),
        "truth_sha256": sha256(CANON / "truth.npy"),
        "scoring": "future_only_unknown_segment (drift vs Xy target)",
        "private_safe": True,
        "uses_unknown_truth": False,
        "fold_filter": fold_filter,
        "wells_decoded": len(wells_done),
        "status": status,
    }
    (out_dir / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    summary = {
        "arm": "exact_hmm_smoother (OU rate + autocorr-aware emission FB smoother)",
        "protocol": protocol["protocol"],
        "status": status,
        "n_rows_total": int(len(oof)),
        "n_rows_scored": n_done,
        "wells_decoded": len(wells_done),
        "elapsed_s": elapsed_s,
        "hmm_config": HMM_CFG,
        "overall": overall,
        "per_fold": per_fold,
        "reference_vs_hmm_exp224": ref,
        "paths": {
            "exact_hmm_d": str(oof_path),
            "exact_hmm_std": str(std_path),
            "well_meta": str(out_dir / "well_meta.csv"),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if fold_filter is not None and status != "full":
        summary["note"] = f"partial run: fold={fold_filter} only; use launch_remaining_folds.sh for full OOF"
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_launch_script(out_dir: Path) -> None:
    script = out_dir / "launch_remaining_folds.sh"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd /root/rogii
LOAD=$(awk '{print $10}' /proc/loadavg | cut -d. -f1)
if (( LOAD > 200 )); then NJ=16; else NJ=32; fi
echo "[exact_hmm] load=$LOAD n_jobs=$NJ"
/usr/local/bin/python3 cv/exact_hmm_smoother_geo5_20260726.py \\
  --out-dir /root/rogii/OOF/exact_hmm_smoother_geo5_20260726 \\
  --n-jobs "$NJ" --backend threading --resume \\
  2>&1 | tee -a /root/rogii/OOF/exact_hmm_smoother_geo5_20260726/logs/full_run.log
""",
        encoding="utf-8",
    )
    script.chmod(0o755)


def decode_wells(
    wells: list[str],
    xy: pd.DataFrame,
    data_root: Path,
    *,
    n_jobs: int,
    backend: str,
    out_dir: Path,
    done: dict[str, tuple[np.ndarray, np.ndarray, dict]],
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, dict]], float]:
    pending = [w for w in wells if w not in done]
    if not pending:
        return [(w, d, s, m) for w, (d, s, m) in done.items()], 0.0
    decode_one(pending[0], data_root)  # JIT warm
    oof_partial = np.full(len(xy), np.nan, dtype=np.float64)
    if (out_dir / "exact_hmm_d.npy").exists():
        oof_partial = np.load(out_dir / "exact_hmm_d.npy").astype(np.float64)
    for w, (d, _s, _m) in done.items():
        grp = xy.loc[xy["well"].astype(str) == w]
        if len(d) == len(grp):
            oof_partial[grp.index.to_numpy()] = d

    results: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
    chunk = max(1, int(n_jobs))
    t0 = time.time()
    for start in range(0, len(pending), chunk):
        batch = pending[start : start + chunk]
        fresh = Parallel(n_jobs=min(len(batch), int(n_jobs)), backend=backend, prefer="threads")(
            delayed(decode_one)(w, data_root) for w in batch
        )
        for item in fresh:
            w, drift, _std, meta = item
            results.append(item)
            if meta.get("ok"):
                grp = xy.loc[xy["well"].astype(str) == w]
                oof_partial[grp.index.to_numpy()] = drift
        all_meta: list[dict] = []
        if (out_dir / "well_meta.csv").exists():
            all_meta = pd.read_csv(out_dir / "well_meta.csv").to_dict("records")
        seen = {str(r["well"]) for r in all_meta}
        for w, _, _, m in fresh:
            if w not in seen:
                all_meta.append({"well": w, **m})
        pd.DataFrame(all_meta).to_csv(out_dir / "well_meta.csv", index=False)
        np.save(out_dir / "exact_hmm_d.npy", oof_partial.astype(np.float32))
        print(
            f"[exact_hmm] batch {start + len(batch)}/{len(pending)} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
    results.extend((w, d, s, m) for w, (d, s, m) in done.items())
    return results, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--fold", type=int, default=None, help="optional held fold (0-4) for smoke/partial")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--backend", choices=["threading", "loky"], default="threading")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="reuse well_meta.csv for already-decoded wells")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)

    xy = pd.read_parquet(XY_PATH, columns=["well", "id", "target"])
    wells = select_wells(args.fold, args.max_wells)
    meta_path = args.out_dir / "well_meta.csv"
    done: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    if args.resume and meta_path.exists():
        old = pd.read_csv(meta_path)
        ok_wells = set(old.loc[old["ok"].astype(bool), "well"].astype(str))
        partial = args.out_dir / "exact_hmm_d.npy"
        if partial.exists():
            arr = np.load(partial).astype(np.float64)
            for well, grp in xy.groupby("well", sort=False):
                w = str(well)
                if w not in ok_wells:
                    continue
                idx = grp.index.to_numpy()
                seg = arr[idx]
                if np.isfinite(seg).all():
                    done[w] = (seg, np.zeros(len(seg)), {"ok": True, "reason": "resume"})

    print(
        f"[exact_hmm] wells={len(wells)} pending={len(wells)-len(done)} "
        f"fold={args.fold} n_jobs={args.n_jobs} backend={args.backend} out={args.out_dir}",
        flush=True,
    )
    results, elapsed = decode_wells(
        wells,
        xy,
        args.data_root,
        n_jobs=int(args.n_jobs),
        backend=str(args.backend),
        out_dir=args.out_dir,
        done=done,
    )
    print(f"[exact_hmm] decode {len(results)} wells in {elapsed:.1f}s", flush=True)

    metas = [{"well": w, **m} for w, _, _, m in results]
    oof, std_full = assemble_oof(results, xy)
    wells_decoded = [w for w, _, _, m in results if m.get("ok")]
    summary = write_outputs(
        args.out_dir,
        oof,
        std_full,
        wells_decoded,
        args.fold,
        metas,
        elapsed,
    )
    write_launch_script(args.out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
