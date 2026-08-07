#!/usr/bin/env python3
"""HMM-H emission builders (P0): heel affine / pseudo-typewell, freeze gs_raw.

First principles (Radiant + exp255 fusion strategy):
  - Observation noise gs is measured on RAW typewell mismatch and NEVER shrunk
    after calibration (sigma-preserving).
  - H1 only remaps the emission reference: GR ≈ a * TW(TVT) + b on known prefix.
  - H2 builds a pseudo-typewell: 0.2*supplied + 0.8*smoothed heel GR@TVT_input
    inside the known TVT coverage; outside falls back to supplied TW.

Decode tables / FB kernels stay identical to locked exp224.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_heel_affine_robust(
    known_tvt: np.ndarray,
    known_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    *,
    min_pairs: int = 16,
    mad_k: float = 4.0,
    a_clip: tuple[float, float] = (0.35, 2.5),
) -> tuple[float, float]:
    """Robust affine y≈a*x+b with MAD outlier rejection. Fallback (1,0)."""
    mask = np.isfinite(known_tvt) & np.isfinite(known_gr)
    if int(mask.sum()) < min_pairs:
        return 1.0, 0.0
    x = np.interp(known_tvt[mask], tw_tvt, tw_gr)
    y = known_gr[mask].astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_pairs or float(np.std(x)) < 1e-6:
        b = float(np.nanmedian(y - x)) if len(x) else 0.0
        return 1.0, b
    a, b = np.polyfit(x, y, 1)
    for _ in range(2):
        resid = y - (a * x + b)
        med = float(np.nanmedian(resid))
        mad = float(np.nanmedian(np.abs(resid - med))) + 1e-6
        keep = np.abs(resid - med) < mad_k * 1.4826 * mad
        if int(keep.sum()) < min_pairs:
            break
        x = x[keep]
        y = y[keep]
        a, b = np.polyfit(x, y, 1)
    return float(np.clip(a, a_clip[0], a_clip[1])), float(b)


def build_pseudo_typewell_gr(
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    known_tvt: np.ndarray,
    known_gr: np.ndarray,
    *,
    w_heel: float = 0.8,
    smooth: int = 5,
) -> np.ndarray:
    """Radiant H2: TW_H = (1-w)*supplied + w*smoothed_known_GR on covered TVT."""
    out = tw_gr.astype(np.float64).copy()
    mask = np.isfinite(known_tvt) & np.isfinite(known_gr)
    if int(mask.sum()) < 8:
        return out
    kt = known_tvt[mask].astype(np.float64)
    kg = known_gr[mask].astype(np.float64)
    order = np.argsort(kt)
    kt = kt[order]
    kg = kg[order]
    # drop duplicate TVT (keep last)
    if len(kt) >= 2:
        uniq = np.ones(len(kt), dtype=bool)
        uniq[:-1] = np.diff(kt) > 1e-9
        kt = kt[uniq]
        kg = kg[uniq]
    if len(kt) < 4:
        return out
    if smooth > 1 and len(kg) >= smooth:
        kg_s = (
            pd.Series(kg)
            .rolling(int(smooth), center=True, min_periods=1)
            .mean()
            .to_numpy(np.float64)
        )
    else:
        kg_s = kg
    heel_on_tw = np.interp(tw_tvt, kt, kg_s)
    covered = (tw_tvt >= float(kt.min())) & (tw_tvt <= float(kt.max()))
    w = float(np.clip(w_heel, 0.0, 1.0))
    out[covered] = (1.0 - w) * tw_gr[covered] + w * heel_on_tw[covered]
    return out


def prep_hmm_h(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    variant: str = "baseline",
    step: float = 0.20,
    n_rates: int = 41,
    rate_span: float = 0.10,
    sig_r: float = 0.002,
    sig_p: float = 0.02,
    df: float = 4.0,
    emission: str = "gauss",
    lam: float = 1.0,
    sigma_mode: str = "std",
    start_sig: float = 0.75,
    r0_sig: float = 0.01,
    band_pad: float = 150.0,
    mom: float = 0.998,
    rate_center: str = "zero",
    h2_w_heel: float = 0.8,
    h2_smooth: int = 5,
) -> dict[str, Any] | None:
    """exp224 `_prep` with H1/H2 emission reference; gs_raw always frozen."""
    if sigma_mode != "std":
        raise NotImplementedError('prep_hmm_h only implements sigma_mode="std"')
    variant = str(variant).lower().strip()
    if variant not in {"baseline", "h1", "h2", "h3_shrink"}:
        raise ValueError(f"unknown variant {variant!r}")

    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].ffill().bfill().to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None

    ktvt = kn["TVT_input"].to_numpy(float)
    kgr = kn["GR"].fillna(0).to_numpy(float)
    tw_at_k_raw = np.interp(ktvt, tw_tvt, tw_gr)

    # ALWAYS measure gs on RAW typewell (sigma-preserving for H1/H2).
    gs_raw = float(np.clip(np.nanstd(kgr - tw_at_k_raw), 10.0, 60.0))

    # H3 negative control: shrink sigma after affine (writeup-forbidden path).
    a_use, b_use = 1.0, 0.0
    tw_gr_use = tw_gr
    gs = gs_raw

    if variant == "h1":
        a_use, b_use = fit_heel_affine_robust(ktvt, kgr, tw_tvt, tw_gr)
        # gs stays gs_raw
    elif variant == "h2":
        tw_gr_use = build_pseudo_typewell_gr(
            tw_tvt, tw_gr, ktvt, kgr, w_heel=h2_w_heel, smooth=h2_smooth
        )
        a_use, b_use = 1.0, 0.0
    elif variant == "h3_shrink":
        a_use, b_use = fit_heel_affine_robust(ktvt, kgr, tw_tvt, tw_gr)
        resid = kgr - (a_use * tw_at_k_raw + b_use)
        gs = float(np.clip(np.nanstd(resid), 10.0, 60.0))  # intentionally shrinks

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dzz = np.diff(tail["Z"].values)
    dmm = np.diff(tail["MD"].values)
    m = dmm > 0
    ir = float(np.median((dt + dzz)[m] / dmm[m])) if m.sum() >= 3 else 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    gmin = max(tw_tvt.min() - 40.0, last_tvt - band_pad)
    gmax = min(tw_tvt.max() + 40.0, last_tvt + band_pad)
    grid = np.arange(gmin, gmax + step, step)
    gr_grid = a_use * np.interp(grid, tw_tvt, tw_gr_use) + b_use

    md = ev["MD"].to_numpy(float)
    z = ev["Z"].to_numpy(float)
    gr = (
        hw["GR"]
        .interpolate(limit_direction="both")
        .fillna(float(np.nanmean(tw_gr)))
        .to_numpy(float)[ev.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))

    zsc = (gr[:, None] - gr_grid[None, :]) / gs
    if emission == "t":
        em = (-0.5 * (df + 1.0) * np.log1p(zsc**2 / df)).astype(np.float32)
    else:
        em = (-0.5 * np.minimum(zsc**2, 600.0)).astype(np.float32)

    if rate_center == "zero":
        span = max(rate_span, abs(ir) + 0.04)
        rates = np.linspace(-span, span, n_rates)
    else:
        span = rate_span
        rates = ir + np.linspace(-span, span, n_rates)
    start_p = float((last_tvt - gmin) / step)

    return dict(
        em=em,
        dm=dm,
        dz=dz,
        grid=grid.astype(np.float64),
        rates=rates.astype(np.float64),
        sp=float(step),
        sig_r=float(sig_r),
        sig_p=float(sig_p),
        start_p=start_p,
        start_sig=float(start_sig),
        r0=float(ir),
        r0_sig=float(r0_sig),
        lam=float(lam),
        mom=float(mom),
        ev_index=ev.index.values,
        n_full=len(hw),
        tvt_input=hw["TVT_input"].to_numpy(float),
        # diagnostics (ignored by FB kernels)
        _hmm_h_variant=variant,
        _gs=gs,
        _gs_raw=gs_raw,
        _a_use=float(a_use),
        _b_use=float(b_use),
    )
