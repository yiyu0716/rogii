#!/usr/bin/env python3
"""Deterministic grid-HMM geosteering smoother.

This is a high-ROI test of a different inference engine, not a small parameter
tweak: replace random forward-only PF sampling with an exact banded
forward-backward pass on a TVT grid.  The state is TVT; the transition prior is
on formation height F = TVT + Z, anchored by the visible-prefix dip estimate;
the observation is scalar GR matched against the typewell GR curve.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp

from cv_runner import ART_DIR, Well, WellStore, md_table
from err_diag import diagnose


SEED = 1745
OUT_DIR = ART_DIR / "hmm_grid"
REPORT = Path(__file__).resolve().parents[1] / "reports" / "hmm_grid_smoother_20260608.md"


def _prefix_params(w: Well, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> tuple[float, float]:
    hs = w.ps_idx
    tw_at_k = np.interp(w.tvt_input[:hs], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(np.nan_to_num(w.gr[:hs].astype(float)) - tw_at_k), 10.0, 60.0))
    st = max(0, hs - 30)
    dt = np.diff(w.tvt_input[st:hs])
    dz = np.diff(w.z[st:hs].astype(float))
    dm = np.diff(w.md[st:hs].astype(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    return gs, ir


def _shifted_add(prev: np.ndarray, shift: int, val: float, out: np.ndarray) -> None:
    """Append prev shifted by TVT-index shift into out candidate row."""
    if shift >= 0:
        if shift < len(prev):
            out[shift:] = prev[: len(prev) - shift] + val
    else:
        s = -shift
        if s < len(prev):
            out[: len(prev) - s] = prev[s:] + val


def _transition_candidates(
    expected_shift: float,
    grid_step: float,
    sigma: float,
    band: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = int(np.rint(expected_shift / grid_step))
    shifts = np.arange(center - band, center + band + 1, dtype=np.int32)
    d = shifts.astype(float) * grid_step - expected_shift
    logw = -0.5 * (d / max(sigma, 1e-6)) ** 2
    logw -= logsumexp(logw)
    return shifts, logw.astype(np.float32)


def _forward_backward_one(
    wid: str,
    store: WellStore,
    grid_step: float,
    trans_sigma: float,
    emit_scale: float,
    band: int,
) -> pd.DataFrame | None:
    w = store.wells[wid]
    if w.n_hidden <= 0:
        return None
    tw = store.load_typewell(wid)
    if tw is None or len(tw) < 5 or not np.isfinite(tw["GR"]).any():
        return None
    tw = tw.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float64)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(np.float64)
    gs, ir = _prefix_params(w, tw_tvt, tw_gr)
    obs_sigma = max(gs * emit_scale, 1e-6)

    hs = w.ps_idx
    n = w.n_hidden
    tvt_min = float(min(tw_tvt[0], w.last_known) - 100.0)
    tvt_max = float(max(tw_tvt[-1], w.last_known) + 100.0)
    grid = np.arange(tvt_min, tvt_max + grid_step, grid_step, dtype=np.float32)
    n_state = len(grid)
    if n_state < 16:
        return None
    tw_gr_grid = np.interp(grid.astype(np.float64), tw_tvt, tw_gr).astype(np.float32)
    gr_h = (
        pd.Series(w.gr.astype(float))
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(np.float32)[hs:]
    )
    z = w.z.astype(np.float32)
    md = w.md.astype(np.float32)

    alpha = np.empty((n, n_state), dtype=np.float32)
    prev_known_tvt = float(w.last_known)
    prev_z = float(z[hs - 1])
    prev_md = float(md[hs - 1])

    for t in range(n):
        row = hs + t
        dm = max(float(md[row] - prev_md), 1.0)
        dz = float(z[row] - prev_z)
        expected_d_tvt = ir * dm - dz
        shifts, logw = _transition_candidates(expected_d_tvt, grid_step, trans_sigma, band)
        obs = -0.5 * np.minimum(((float(gr_h[t]) - tw_gr_grid) / obs_sigma) ** 2, 600.0)

        if t == 0:
            d = (grid.astype(np.float64) - prev_known_tvt) - expected_d_tvt
            a = obs.astype(np.float64) - 0.5 * (d / max(trans_sigma, 1e-6)) ** 2
        else:
            cand = np.full((len(shifts), n_state), -np.inf, dtype=np.float32)
            for i, (shift, lw) in enumerate(zip(shifts, logw)):
                _shifted_add(alpha[t - 1], int(shift), float(lw), cand[i])
            a = obs + logsumexp(cand, axis=0).astype(np.float32)
        a = a - logsumexp(a)
        alpha[t] = a.astype(np.float32)
        prev_z = float(z[row])
        prev_md = float(md[row])

    beta = np.zeros(n_state, dtype=np.float32)
    means = np.empty(n, dtype=np.float32)
    maps = np.empty(n, dtype=np.float32)
    q40 = np.empty(n, dtype=np.float32)
    q60 = np.empty(n, dtype=np.float32)
    betas: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    for t in range(n - 1, -1, -1):
        betas[t] = beta.copy()
        if t == 0:
            prev_z = float(z[hs - 1])
            prev_md = float(md[hs - 1])
        else:
            prev_z = float(z[hs + t - 1])
            prev_md = float(md[hs + t - 1])
        row_next = hs + t
        if t > 0:
            row_next = hs + t
        if t > 0:
            pass
        if t > 0:
            # beta for alpha[t-1] uses row t as the next observation.
            dm = max(float(md[hs + t] - prev_md), 1.0)
            dz = float(z[hs + t] - prev_z)
            expected_d_tvt = ir * dm - dz
            shifts, logw = _transition_candidates(expected_d_tvt, grid_step, trans_sigma, band)
            obs_next = -0.5 * np.minimum(((float(gr_h[t]) - tw_gr_grid) / obs_sigma) ** 2, 600.0)
            next_msg = beta + obs_next.astype(np.float32)
            cand = np.full((len(shifts), n_state), -np.inf, dtype=np.float32)
            # Forward shift maps prev -> curr.  For beta(prev), gather curr =
            # prev + shift, so this is the inverse fill of _shifted_add.
            for i, (shift, lw) in enumerate(zip(shifts, logw)):
                if shift >= 0:
                    if shift < n_state:
                        cand[i, : n_state - shift] = next_msg[shift:] + float(lw)
                else:
                    s = -int(shift)
                    if s < n_state:
                        cand[i, s:] = next_msg[: n_state - s] + float(lw)
            beta = logsumexp(cand, axis=0).astype(np.float32)
            beta = beta - logsumexp(beta)

    for t in range(n):
        g = alpha[t] + betas[t]
        g = g - logsumexp(g)
        p = np.exp(g).astype(np.float64)
        s = float(p.sum())
        if s <= 0 or not np.isfinite(s):
            p = np.full(n_state, 1.0 / n_state, dtype=np.float64)
        else:
            p /= s
        means[t] = float(np.dot(p, grid.astype(np.float64)))
        maps[t] = float(grid[int(np.argmax(p))])
        cdf = np.cumsum(p)
        q40[t] = float(grid[min(np.searchsorted(cdf, 0.40), n_state - 1)])
        q60[t] = float(grid[min(np.searchsorted(cdf, 0.60), n_state - 1)])

    target = w.tvt[hs:].astype(np.float32) - np.float32(w.last_known)
    return pd.DataFrame(
        {
            "well": wid,
            "row_idx": np.arange(hs, len(w.md), dtype=np.int32),
            "target": target,
            "hmm_mean": means - np.float32(w.last_known),
            "hmm_map": maps - np.float32(w.last_known),
            "hmm_q40": q40 - np.float32(w.last_known),
            "hmm_q60": q60 - np.float32(w.last_known),
        }
    )


def _score(df: pd.DataFrame, names: list[str], label: str) -> pd.DataFrame:
    rows = []
    wells = df["well"].to_numpy()
    target = df["target"].to_numpy(np.float64)
    for name in names:
        rows.append(diagnose(wells, df[name].to_numpy(np.float64), target, name=f"{label}:{name}", verbose=False))
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def _blend_with_posterior(df: pd.DataFrame) -> pd.DataFrame:
    path = ART_DIR / "pf_posterior" / "posterior_gate_smoother_bigru_oof.pkl"
    if not path.exists():
        path = ART_DIR / "pf_posterior" / "posterior_smoother_bigru_oof.pkl"
    if not path.exists():
        return pd.DataFrame()
    base = pd.read_pickle(path)
    base = base.rename(columns={"pred": "posterior_pred"})
    # The posterior OOF frame is row-aligned by well order but does not always
    # carry row_idx in older artifacts.  Reconstruct it from group order.  Some
    # older CSV-built artifacts also contain a couple of mangled scientific-
    # notation-looking well ids, so restrict to the current HMM wells and require
    # exact length matches before assigning row_idx.
    wanted = set(df["well"].astype(str).unique())
    base = base.loc[base["well"].astype(str).isin(wanted)].copy()
    if "row_idx" not in base.columns:
        store = WellStore()
        parts = []
        for wid, grp in base.groupby("well", sort=False):
            w = store.wells[str(wid)]
            if len(grp) != w.n_hidden:
                continue
            g = grp.copy()
            g["row_idx"] = np.arange(w.ps_idx, len(w.md), dtype=np.int32)
            parts.append(g)
        if not parts:
            return pd.DataFrame()
        base = pd.concat(parts, ignore_index=True)
    merged = df.merge(base[["well", "row_idx", "target", "posterior_pred"]], on=["well", "row_idx"], suffixes=("", "_base"))
    ok = np.abs(merged["target"] - merged["target_base"]) < 1e-3
    merged = merged.loc[ok].copy()
    rows = []
    wells = merged["well"].to_numpy()
    y = merged["target"].to_numpy(np.float64)
    for hn in ["hmm_mean", "hmm_map", "hmm_q40", "hmm_q60"]:
        for w_hmm in (0.20, 0.35, 0.50, 0.65, 0.80):
            pred = (1.0 - w_hmm) * merged["posterior_pred"].to_numpy(np.float64) + w_hmm * merged[hn].to_numpy(np.float64)
            r = diagnose(wells, pred, y, name=f"posterior+hmm:{hn}@{w_hmm:.2f}", verbose=False)
            r["hmm_variant"] = hn
            r["w_hmm"] = w_hmm
            rows.append(r)
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def _write_report(summary: pd.DataFrame, blends: pd.DataFrame, args: argparse.Namespace, out_path: Path, seconds: float) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write("# HMM Grid Smoother - 2026-06-08\n\n")
        f.write("## Question\n\n")
        f.write("Can an exact deterministic forward-backward HMM over TVT states improve on the random forward-only PF/posterior smoother by using future hidden GR observations while preserving the F=TVT+Z motion prior?\n\n")
        f.write("## Run\n\n")
        f.write(f"- sample: `{args.sample}` (`0` means all wells)\n")
        f.write(f"- grid_step: `{args.grid_step}` ft\n")
        f.write(f"- trans_sigma: `{args.trans_sigma}` ft/row transition residual scale\n")
        f.write(f"- emit_scale: `{args.emit_scale}` x prefix GR residual std\n")
        f.write(f"- band: `{args.band}` grid cells around expected F-motion\n")
        f.write(f"- jobs: `{args.jobs}`\n")
        f.write(f"- runtime: `{seconds:.1f}` seconds\n")
        f.write(f"- artifact: `{out_path}`\n\n")
        f.write("## Standalone HMM CV\n\n")
        cols = ["name", "n_wells", "rmse", "level_resid", "swing_resid", "level_corr", "shape_corr"]
        f.write(md_table(summary[cols].head(20), index=False))
        f.write("\n\n")
        if not blends.empty:
            f.write("## Blend With Current Posterior Smoother\n\n")
            cols2 = ["name", "n_wells", "rmse", "level_resid", "swing_resid", "level_corr", "shape_corr", "hmm_variant", "w_hmm"]
            f.write(md_table(blends[cols2].head(20), index=False))
            f.write("\n\n")
        f.write("## Interpretation\n\n")
        best = float(summary["rmse"].iloc[0])
        best_blend = float(blends["rmse"].iloc[0]) if not blends.empty else np.nan
        f.write(f"Best standalone HMM RMSE is `{best:.4f}`. ")
        if not np.isnan(best_blend):
            f.write(f"Best posterior/HMM blend RMSE is `{best_blend:.4f}`. ")
        f.write("Compare against the current deploy-safe local best `8.5592` only after a full run; sample runs are screening only.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40, help="0 means all wells")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--grid-step", type=float, default=2.0)
    ap.add_argument("--trans-sigma", type=float, default=3.0)
    ap.add_argument("--emit-scale", type=float, default=2.0)
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.sample == 0 else f"sample{args.sample}"
    tag = f"{suffix}_gs{args.grid_step:g}_ts{args.trans_sigma:g}_es{args.emit_scale:g}_b{args.band}"
    out_path = OUT_DIR / f"hmm_grid_{tag}.pkl"
    summary_path = OUT_DIR / f"hmm_grid_{tag}_summary.csv"
    blend_path = OUT_DIR / f"hmm_grid_{tag}_blend_summary.csv"

    if out_path.exists() and not args.force:
        df = pd.read_pickle(out_path)
        t0 = time.time()
    else:
        store = WellStore()
        ids = store.ids()
        if args.sample > 0:
            rng = np.random.default_rng(SEED)
            ids = list(rng.choice(ids, min(args.sample, len(ids)), replace=False))
        print(
            f"hmm_grid | wells={len(ids)} grid={args.grid_step} sigma={args.trans_sigma} "
            f"emit={args.emit_scale} band={args.band} jobs={args.jobs}",
            flush=True,
        )
        t0 = time.time()
        out = Parallel(n_jobs=args.jobs, backend="multiprocessing")(
            delayed(_forward_backward_one)(
                wid, store, args.grid_step, args.trans_sigma, args.emit_scale, args.band
            )
            for wid in ids
        )
        parts = [x for x in out if x is not None]
        df = pd.concat(parts, ignore_index=True)
        df.to_pickle(out_path)
    seconds = time.time() - t0
    pred_cols = [c for c in df.columns if c.startswith("hmm_")]
    summary = _score(df, pred_cols, "hmm_grid")
    summary.to_csv(summary_path, index=False)
    blends = _blend_with_posterior(df)
    if not blends.empty:
        blends.to_csv(blend_path, index=False)
    _write_report(summary, blends, args, out_path, seconds)
    print("\n=== HMM standalone ===")
    print(summary[["name", "rmse", "level_resid", "swing_resid", "level_corr", "shape_corr"]].head(20).to_string(index=False))
    if not blends.empty:
        print("\n=== posterior + HMM ===")
        print(blends[["name", "rmse", "level_resid", "swing_resid", "level_corr", "shape_corr"]].head(20).to_string(index=False))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
