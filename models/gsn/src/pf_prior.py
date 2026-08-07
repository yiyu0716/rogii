"""
Particle-filter TVT prior for MTP.

Extracted from notebook/physical_model.ipynb.  Estimates future TVT by matching
horizontal GR to typewell GR — no future TVT labels required.

    geo_z_prior = tvt_pf + Z
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config.cnn_sdf_config import Config


def run_particle_filter(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Conservative particle filter.  Returns (tvt_array, total_log_likelihood).

    Known rows (TVT_input not NaN) are left unchanged; blind-zone rows are filled
    with the weighted-mean particle TVT estimate.
    """
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    n = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_z
    pos = ls + 5.0 * rng.standard_normal(n)
    rate = ir + 0.01 * rng.standard_normal(n)
    w = np.ones(n) / n

    mom = 0.998
    vn = 0.002
    pn = 0.005
    rp = 0.1
    rr = 0.001
    resamp = 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_md = last_md
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_md, 1.0)
        rate = mom * rate + vn * rng.standard_normal(n)
        pos = pos + rate * dm_step + pn * rng.standard_normal(n)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(n) / n

        n_eff = 1.0 / (w**2).sum()
        if n_eff < resamp * n:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / n)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)
            pos = pos[idx] + rp * rng.standard_normal(n)
            rate = rate[idx] + rr * rng.standard_normal(n)
            w = np.ones(n) / n

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_md = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 500,
    n_seeds: int = 16,
    scale: float = 5.0,
    seed_offset: int = 0,
) -> np.ndarray:
    """Likelihood-weighted multi-seed PF ensemble."""
    preds: list[np.ndarray] = []
    liks: list[float] = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=seed_offset + s)
        preds.append(p)
        liks.append(ll)

    liks_arr = np.array(liks)
    liks_n = liks_arr - liks_arr.max()
    weights = np.exp(liks_n / float(scale))
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def load_typewell(horiz_path: Path, typewell_dir: Path | None = None) -> pd.DataFrame:
    tw_dir = typewell_dir or horiz_path.parent
    sid = horiz_path.name.split("__")[0]
    tw_path = tw_dir / f"{sid}{Config.TYPEWELL_SUFFIX}"
    if not tw_path.exists():
        raise FileNotFoundError(f"typewell not found: {tw_path}")
    return pd.read_csv(tw_path)


def estimate_tvt_prior(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int | None = None,
    n_seeds: int | None = None,
    scale: float | None = None,
    seed_offset: int = 0,
) -> np.ndarray:
    """Public entry: full-length TVT prior array aligned to ``hw`` rows."""
    n_particles = n_particles or Config.MTP_PF_N_PARTICLES
    n_seeds = n_seeds or Config.MTP_PF_N_SEEDS
    scale = scale if scale is not None else Config.MTP_PF_SCALE
    return run_pf_lik_ensemble(
        hw, tw,
        n_particles=n_particles,
        n_seeds=n_seeds,
        scale=scale,
        seed_offset=seed_offset,
    )


def well_pf_seed(horiz_path: Path) -> int:
    """Deterministic seed for validation / inference."""
    h = hashlib.md5(str(horiz_path).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)
