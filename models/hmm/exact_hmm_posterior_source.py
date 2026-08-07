#!/usr/bin/env python3
"""Exact HMM, heatmap/DTW evidence, and heel-calibrated two-mode sources.

These utilities are intentionally deployable: they only use the current well's
known prefix, full GR trace, trajectory, and paired typewell.  The generated
paths are absolute TVT values aligned to the hidden rows.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ROGII_ROOT", str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is available in the project env
    njit = None

from candidate_path295_ranker_v1 import (
    DATA_ROOT,
    DEFAULT_BASE_NPZ,
    DEFAULT_CACHE_DIR,
    _interp_to_length,
    load_base_prediction,
)
from candidate_pool_governor_v1 import well_slices_from_wells
from cv_runner import ART_DIR
from private_safe_visible_prefix_candidate_posterior_v1 import score_prediction


SEED = 20260707
DEFAULT_OUT_DIR = ART_DIR / "exact_hmm_heatmap_twomode_sources_20260707"


@dataclass
class HMMResult:
    pred: np.ndarray
    std_eval: np.ndarray
    loglik: float
    ev_index: np.ndarray
    mean_eval: np.ndarray


@dataclass
class TwoModeResult:
    path: np.ndarray
    commit_path: np.ndarray
    p_first: float
    shift_first: float
    shift_second: float
    mode_sep: float
    cost_first: float
    cost_second: float
    cost_ratio: float
    is_two_mode: bool


def _prep_typewell(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_numeric(tw["TVT"], errors="coerce").to_numpy(np.float64)
    g = pd.to_numeric(tw["GR"], errors="coerce")
    g = g.ffill().bfill().fillna(float(g.mean()) if np.isfinite(g.mean()) else 0.0).to_numpy(np.float64)
    m = np.isfinite(t) & np.isfinite(g)
    t, g = t[m], g[m]
    if len(t) == 0:
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    order = np.argsort(t)
    t, g = t[order], g[order]
    uniq, idx = np.unique(t, return_index=True)
    return uniq.astype(np.float64), g[idx].astype(np.float64)


def _filled_gr(values: pd.Series | np.ndarray, fallback: float = 0.0) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    return s.interpolate(limit_direction="both").bfill().ffill().fillna(float(fallback)).to_numpy(np.float64)


def _prefix_stats(hw: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray, tail_n: int = 30) -> tuple[float, float, float, float]:
    known = hw.loc[hw["TVT_input"].notna()].copy()
    if known.empty:
        return 1.0, 0.0, 30.0, 0.0
    kgr = _filled_gr(known["GR"], fallback=float(np.nanmean(tw_gr)))
    ktvt = known["TVT_input"].to_numpy(np.float64)
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if int(v.sum()) >= 20 and float(np.std(tw_at_k[v])) > 1e-6:
        a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    else:
        a, b = 1.0, float(np.nanmean(kgr[v] - tw_at_k[v])) if bool(v.any()) else 0.0
    resid = kgr[v] - (float(a) * tw_at_k[v] + float(b))
    if len(resid) >= 20:
        sigma = float(np.clip(np.nanstd(resid), 10.0, 60.0))
    else:
        sigma = 30.0
    tail = known.tail(tail_n)
    dt = np.diff(tail["TVT_input"].to_numpy(np.float64))
    dz = np.diff(tail["Z"].to_numpy(np.float64))
    dm = np.diff(tail["MD"].to_numpy(np.float64))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if int(m.sum()) >= 3 else 0.0
    return float(a), float(b), sigma, ir


if njit is not None:

    @njit(cache=True, nogil=True)
    def _hmm2_fb(em, dm, dz, sp, rates, sig_r, sig_p, start_p, start_sig, r0, r0_sig, lam, mom):
        T, P = em.shape
        R = len(rates)
        sr = rates[1] - rates[0] if R > 1 else 1.0
        NEG = np.float32(-1e18)
        alpha = np.full((T, P, R), NEG, np.float32)
        prev = np.full((P, R), NEG, np.float32)
        for p in range(P):
            dpos = (p - start_p) * sp
            lp0 = -0.5 * (dpos / start_sig) ** 2
            if lp0 < -60.0:
                continue
            for r in range(R):
                dr = (rates[r] - r0) / r0_sig
                prev[p, r] = np.float32(lp0 - 0.5 * dr * dr)
        tmp = np.empty((P, R), np.float32)
        cur = np.empty((P, R), np.float32)
        for t in range(T):
            sgr = sig_r * np.sqrt(dm[t])
            v_r = (sgr / sr) ** 2
            lrk = np.empty((R, 3), np.float64)
            for r in range(R):
                m_r = -(1.0 - mom) * rates[r] * dm[t] / sr
                pp = 0.5 * (v_r + m_r)
                pm = 0.5 * (v_r - m_r)
                if pp < 1e-12:
                    pp = 1e-12
                if pm < 1e-12:
                    pm = 1e-12
                tot = pp + pm
                if tot > 0.9:
                    pp *= 0.9 / tot
                    pm *= 0.9 / tot
                lrk[r, 0] = np.log(pm)
                lrk[r, 1] = np.log(1.0 - pp - pm)
                lrk[r, 2] = np.log(pp)
            for p in range(P):
                for r2 in range(R):
                    m2 = NEG
                    k0 = r2 - 1 if r2 - 1 >= 0 else 0
                    k1 = r2 + 1 if r2 + 1 <= R - 1 else R - 1
                    for r in range(k0, k1 + 1):
                        v = prev[p, r] + lrk[r, r2 - r + 1]
                        if v > m2:
                            m2 = v
                    if m2 > NEG / 2:
                        ss = 0.0
                        for r in range(k0, k1 + 1):
                            ss += np.exp(prev[p, r] + lrk[r, r2 - r + 1] - m2)
                        tmp[p, r2] = np.float32(m2 + np.log(ss))
                    else:
                        tmp[p, r2] = NEG
            spe = sig_p if sig_p > 0.35 * sp else 0.35 * sp
            for r2 in range(R):
                mu = rates[r2] * dm[t] - dz[t]
                b0 = int(np.floor(mu / sp + 0.5))
                lp = np.empty(5, np.float64)
                for k in range(5):
                    d = (b0 - 2 + k) * sp - mu
                    lp[k] = -0.5 * (d / spe) ** 2
                mx = lp[0]
                for k in range(1, 5):
                    if lp[k] > mx:
                        mx = lp[k]
                ss = 0.0
                for k in range(5):
                    ss += np.exp(lp[k] - mx)
                lz = mx + np.log(ss)
                for k in range(5):
                    lp[k] -= lz
                for p2 in range(P):
                    m2 = NEG
                    for k in range(5):
                        p1 = p2 - (b0 - 2 + k)
                        if p1 < 0 or p1 >= P:
                            continue
                        v = tmp[p1, r2] + lp[k]
                        if v > m2:
                            m2 = v
                    if m2 > NEG / 2:
                        ss2 = 0.0
                        for k in range(5):
                            p1 = p2 - (b0 - 2 + k)
                            if p1 < 0 or p1 >= P:
                                continue
                            ss2 += np.exp(tmp[p1, r2] + lp[k] - m2)
                        cur[p2, r2] = np.float32(m2 + np.log(ss2) + lam * em[t, p2])
                    else:
                        cur[p2, r2] = NEG
            for p in range(P):
                for r in range(R):
                    alpha[t, p, r] = cur[p, r]
                    prev[p, r] = cur[p, r]
        mx = NEG
        for p in range(P):
            for r in range(R):
                if alpha[T - 1, p, r] > mx:
                    mx = alpha[T - 1, p, r]
        ss = 0.0
        for p in range(P):
            for r in range(R):
                ss += np.exp(alpha[T - 1, p, r] - mx)
        loglik = float(mx) + np.log(ss)
        post_p = np.zeros((T, P), np.float64)
        beta_next = np.zeros((P, R), np.float32)
        beta_cur = np.empty((P, R), np.float32)
        beta_tmp = np.empty((P, R), np.float32)
        for t in range(T - 1, -1, -1):
            mxp = NEG
            for p in range(P):
                for r in range(R):
                    v = alpha[t, p, r] + beta_next[p, r]
                    if v > mxp:
                        mxp = v
            total = 0.0
            for p in range(P):
                acc = 0.0
                for r in range(R):
                    acc += np.exp(alpha[t, p, r] + beta_next[p, r] - mxp)
                post_p[t, p] = acc
                total += acc
            if total > 0.0:
                for p in range(P):
                    post_p[t, p] /= total
            if t == 0:
                break
            sgr = sig_r * np.sqrt(dm[t])
            v_r = (sgr / sr) ** 2
            lrk = np.empty((R, 3), np.float64)
            for r in range(R):
                m_r = -(1.0 - mom) * rates[r] * dm[t] / sr
                pp = 0.5 * (v_r + m_r)
                pm = 0.5 * (v_r - m_r)
                if pp < 1e-12:
                    pp = 1e-12
                if pm < 1e-12:
                    pm = 1e-12
                tot = pp + pm
                if tot > 0.9:
                    pp *= 0.9 / tot
                    pm *= 0.9 / tot
                lrk[r, 0] = np.log(pm)
                lrk[r, 1] = np.log(1.0 - pp - pm)
                lrk[r, 2] = np.log(pp)
            spe = sig_p if sig_p > 0.35 * sp else 0.35 * sp
            for r2 in range(R):
                mu = rates[r2] * dm[t] - dz[t]
                b0 = int(np.floor(mu / sp + 0.5))
                lp = np.empty(5, np.float64)
                for k in range(5):
                    d = (b0 - 2 + k) * sp - mu
                    lp[k] = -0.5 * (d / spe) ** 2
                mx2 = lp[0]
                for k in range(1, 5):
                    if lp[k] > mx2:
                        mx2 = lp[k]
                ss2 = 0.0
                for k in range(5):
                    ss2 += np.exp(lp[k] - mx2)
                lz = mx2 + np.log(ss2)
                for k in range(5):
                    lp[k] -= lz
                for p1 in range(P):
                    m2 = NEG
                    for k in range(5):
                        p2 = p1 + (b0 - 2 + k)
                        if p2 < 0 or p2 >= P:
                            continue
                        v = lp[k] + lam * em[t, p2] + beta_next[p2, r2]
                        if v > m2:
                            m2 = v
                    if m2 > NEG / 2:
                        ss3 = 0.0
                        for k in range(5):
                            p2 = p1 + (b0 - 2 + k)
                            if p2 < 0 or p2 >= P:
                                continue
                            ss3 += np.exp(lp[k] + lam * em[t, p2] + beta_next[p2, r2] - m2)
                        beta_tmp[p1, r2] = np.float32(m2 + np.log(ss3))
                    else:
                        beta_tmp[p1, r2] = NEG
            for p in range(P):
                for r in range(R):
                    m2 = NEG
                    k0 = r - 1 if r - 1 >= 0 else 0
                    k1 = r + 1 if r + 1 <= R - 1 else R - 1
                    for r2 in range(k0, k1 + 1):
                        v = lrk[r, r2 - r + 1] + beta_tmp[p, r2]
                        if v > m2:
                            m2 = v
                    if m2 > NEG / 2:
                        ss4 = 0.0
                        for r2 in range(k0, k1 + 1):
                            ss4 += np.exp(lrk[r, r2 - r + 1] + beta_tmp[p, r2] - m2)
                        beta_cur[p, r] = np.float32(m2 + np.log(ss4))
                    else:
                        beta_cur[p, r] = NEG
            mxn = NEG
            for p in range(P):
                for r in range(R):
                    if beta_cur[p, r] > mxn:
                        mxn = beta_cur[p, r]
            ssn = 0.0
            for p in range(P):
                for r in range(R):
                    ssn += np.exp(beta_cur[p, r] - mxn)
            norm = mxn + np.log(ssn) if ssn > 0.0 else 0.0
            for p in range(P):
                for r in range(R):
                    beta_next[p, r] = np.float32(beta_cur[p, r] - norm)
        return post_p, loglik


def run_hmm2_frame(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    step: float = 0.7,
    n_rates: int = 31,
    rate_span: float = 0.10,
    sig_r: float = 0.002,
    sig_p: float = 0.02,
    emission: str = "gauss",
    lam: float = 1.0,
    sigma_mode: str = "std",
    start_sig: float = 0.75,
    r0_sig: float = 0.01,
    band_pad: float = 80.0,
    mom: float = 0.998,
    rate_center: str = "zero",
) -> HMMResult:
    if njit is None:
        raise RuntimeError("numba is required for run_hmm2_frame")
    tw_tvt, tw_gr = _prep_typewell(tw)
    known_mask = hw["TVT_input"].notna().to_numpy()
    ev_mask = ~known_mask
    out = hw["TVT_input"].to_numpy(np.float64).copy()
    ev_index = np.flatnonzero(ev_mask)
    if len(ev_index) == 0:
        return HMMResult(out, np.zeros(0, dtype=np.float64), 0.0, ev_index, np.zeros(0, dtype=np.float64))
    known = hw.loc[known_mask]
    a, b, gs_mad, ir = _prefix_stats(hw, tw_tvt, tw_gr)
    if sigma_mode == "mad":
        gs = gs_mad
        a_use, b_use = a, b
    else:
        tw_at_k = np.interp(known["TVT_input"].to_numpy(np.float64), tw_tvt, tw_gr)
        kgr = _filled_gr(known["GR"], fallback=float(np.nanmean(tw_gr)))
        gs = float(np.clip(np.nanstd(kgr - tw_at_k), 10.0, 60.0))
        a_use, b_use = 1.0, 0.0
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    gmin = max(float(tw_tvt.min()) - 40.0, last_tvt - float(band_pad))
    gmax = min(float(tw_tvt.max()) + 40.0, last_tvt + float(band_pad))
    grid = np.arange(gmin, gmax + float(step), float(step), dtype=np.float64)
    if len(grid) < 4:
        grid = np.linspace(last_tvt - band_pad, last_tvt + band_pad, 16, dtype=np.float64)
    gr_grid = float(a_use) * np.interp(grid, tw_tvt, tw_gr) + float(b_use)
    ev = hw.loc[ev_mask]
    gr_all = _filled_gr(hw["GR"], fallback=float(np.nanmean(tw_gr)))
    gr = gr_all[ev_index]
    md = ev["MD"].to_numpy(np.float64)
    z = ev["Z"].to_numpy(np.float64)
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    zsc = (gr[:, None] - gr_grid[None, :]) / max(float(gs), 1e-6)
    if emission == "t":
        em = (-2.5 * np.log1p((zsc * zsc) / 4.0)).astype(np.float32)
    else:
        em = (-0.5 * np.minimum(zsc * zsc, 600.0)).astype(np.float32)
    if rate_center == "zero":
        span = max(float(rate_span), abs(float(ir)) + 0.04)
        rates = np.linspace(-span, span, int(n_rates), dtype=np.float64)
    else:
        rates = float(ir) + np.linspace(-float(rate_span), float(rate_span), int(n_rates), dtype=np.float64)
    start_p = float((last_tvt - float(grid[0])) / float(step))
    post_p, loglik = _hmm2_fb(
        em,
        dm.astype(np.float64),
        dz.astype(np.float64),
        float(step),
        rates.astype(np.float64),
        float(sig_r),
        float(sig_p),
        start_p,
        float(start_sig),
        float(ir),
        float(r0_sig),
        float(lam),
        float(mom),
    )
    mean = post_p @ grid
    var = post_p @ (grid * grid) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    out[ev_index] = mean
    return HMMResult(out, std.astype(np.float64), float(loglik), ev_index, mean.astype(np.float64))


def _segment_values(values: np.ndarray, segment: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    s = max(1, int(segment))
    n = len(arr) // s
    if n <= 0:
        return arr.copy()
    return arr[: n * s].reshape(n, s).mean(axis=1)


def _dtw_path(cost: np.ndarray) -> np.ndarray:
    c = np.asarray(cost, dtype=np.float64)
    n_t, n_h = c.shape
    dp = np.full((n_t, n_h), np.inf, dtype=np.float64)
    back = np.zeros((n_t, n_h), dtype=np.int8)
    dp[:, 0] = c[:, 0]
    for j in range(1, n_h):
        for i in range(n_t):
            best = dp[i, j - 1]
            code = 0
            if i > 0 and dp[i - 1, j - 1] < best:
                best = dp[i - 1, j - 1]
                code = -1
            if i + 1 < n_t and dp[i + 1, j - 1] < best:
                best = dp[i + 1, j - 1]
                code = 1
            dp[i, j] = c[i, j] + best
            back[i, j] = code
    path = np.zeros(n_h, dtype=np.int32)
    i = int(np.argmin(dp[:, -1]))
    path[-1] = i
    for j in range(n_h - 1, 0, -1):
        i = int(np.clip(i + int(back[i, j]), 0, n_t - 1))
        path[j - 1] = i
    return path


def candidate_heatmap_evidence(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    candidate_tvt: np.ndarray,
    *,
    segment: int = 32,
    tw_window: float = 80.0,
    run_dtw: bool = True,
) -> dict[str, float]:
    """Viewer-style GR heatmap evidence for one candidate path."""
    ev_mask = hw["TVT_input"].isna().to_numpy()
    ev_index = np.flatnonzero(ev_mask)
    if len(ev_index) == 0:
        return {
            "hm_rmse": 0.0,
            "hm_mae": 0.0,
            "hm_valley_excess": 0.0,
            "hm_second_margin": 0.0,
            "hm_dtw_rmse_ft": 0.0,
        }
    tw_tvt, tw_gr = _prep_typewell(tw)
    cand = _interp_to_length(np.asarray(candidate_tvt, dtype=np.float64), len(ev_index))
    gr_all = _filled_gr(hw["GR"], fallback=float(np.nanmean(tw_gr)))
    h_gr = _segment_values(gr_all[ev_index], segment)
    c_tvt = _segment_values(cand, segment)
    n = min(len(h_gr), len(c_tvt))
    if n == 0:
        n = len(cand)
        h_gr = gr_all[ev_index]
        c_tvt = cand
    h_gr = h_gr[:n]
    c_tvt = c_tvt[:n]
    center = float(hw.loc[hw["TVT_input"].notna(), "TVT_input"].iloc[-1])
    lo = max(float(tw_tvt.min()), min(float(np.nanmin(c_tvt)), center) - float(tw_window))
    hi = min(float(tw_tvt.max()), max(float(np.nanmax(c_tvt)), center) + float(tw_window))
    m = (tw_tvt >= lo) & (tw_tvt <= hi)
    if int(m.sum()) < 5:
        m = np.ones(len(tw_tvt), dtype=bool)
    t_axis = tw_tvt[m]
    g_axis = tw_gr[m]
    cand_gr = np.interp(c_tvt, tw_tvt, tw_gr)
    resid = h_gr - cand_gr
    heat_abs = np.abs(h_gr[None, :] - g_axis[:, None])
    best = np.min(heat_abs, axis=0)
    part = np.partition(heat_abs, 1, axis=0)
    second = part[1] if part.shape[0] > 1 else part[0]
    cand_abs = np.abs(resid)
    cand_idx = np.searchsorted(t_axis, c_tvt).clip(0, len(t_axis) - 1)
    if run_dtw and len(t_axis) >= 3 and n >= 3:
        dtw_idx = _dtw_path(heat_abs * heat_abs)
        dtw_tvt = t_axis[np.clip(dtw_idx, 0, len(t_axis) - 1)]
        dtw_rmse = float(np.sqrt(np.mean((c_tvt - dtw_tvt) ** 2)))
        dtw_abs = float(np.mean(np.abs(c_tvt - dtw_tvt)))
    else:
        dtw_rmse = 0.0
        dtw_abs = 0.0
    return {
        "hm_rmse": float(np.sqrt(np.mean(resid * resid))),
        "hm_mae": float(np.mean(cand_abs)),
        "hm_valley_excess": float(np.mean(np.maximum(cand_abs - best, 0.0))),
        "hm_second_margin": float(np.median(second - best)),
        "hm_cand_row_std": float(np.std(cand_idx.astype(np.float64))),
        "hm_dtw_rmse_ft": dtw_rmse,
        "hm_dtw_abs_ft": dtw_abs,
    }


def _path_for_candidate_row(row: pd.Series, path_drift: np.ndarray, length: int) -> np.ndarray:
    start = int(row["record_start"])
    n = int(row["record_len"])
    if start < 0 or n <= 0:
        return np.zeros(int(length), dtype=np.float64)
    return _interp_to_length(path_drift[start : start + n], int(length))


def _last_known_tvt(hw: pd.DataFrame) -> float:
    known = hw.loc[hw["TVT_input"].notna(), "TVT_input"]
    if known.empty:
        return 0.0
    return float(pd.to_numeric(known, errors="coerce").dropna().iloc[-1])


def _augment_heatmap_one_well(
    wi: int,
    wid: str,
    group: pd.DataFrame,
    *,
    path_drift: np.ndarray,
    data_root: Path,
    max_score_rank: int,
    segment: int,
    tw_window: float,
) -> pd.DataFrame:
    hw, tw = _load_train_frames(data_root, wid)
    length = int(hw["TVT_input"].isna().sum())
    rows: list[dict[str, float | int]] = []
    for idx, row in group.iterrows():
        score_rank = int(pd.to_numeric(row.get("score_rank", 999), errors="coerce"))
        base = {"_row_index": int(idx)}
        if score_rank > int(max_score_rank):
            rows.append(base)
            continue
        path = _last_known_tvt(hw) + _path_for_candidate_row(row, path_drift, length)
        ev = candidate_heatmap_evidence(hw, tw, path, segment=int(segment), tw_window=float(tw_window))
        rows.append({**base, **ev})
    return pd.DataFrame(rows)


def augment_candidate_csv_with_heatmap(args: argparse.Namespace) -> Path:
    candidates = pd.read_csv(args.candidate_csv, dtype={"well_id": str})
    well_index = pd.read_csv(Path(args.cache_dir) / "well_index.csv", dtype={"well_id": str})
    wi_to_wid = dict(zip(well_index["well_index"].astype(int), well_index["well_id"].astype(str)))
    path_drift = np.load(Path(args.cache_dir) / "path_cache_topk.npz", mmap_mode="r")["path_drift"]
    work = candidates.reset_index(drop=False).rename(columns={"index": "_orig_index"})
    prefer = "threads" if str(args.joblib_backend) == "threading" else "processes"
    pieces = Parallel(n_jobs=int(args.n_jobs), verbose=5 if args.verbose else 0, prefer=prefer)(
        delayed(_augment_heatmap_one_well)(
            int(wi),
            str(wi_to_wid[int(wi)]),
            group.set_index("_orig_index", drop=True),
            path_drift=path_drift,
            data_root=Path(args.data_root),
            max_score_rank=int(args.heatmap_top_rank),
            segment=int(args.heatmap_segment),
            tw_window=float(args.heatmap_tw_window),
        )
        for wi, group in work.groupby("well_index", sort=False)
        if int(wi) in wi_to_wid
    )
    feat = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["_row_index"])
    feat = feat.drop_duplicates("_row_index", keep="first").set_index("_row_index")
    out = candidates.copy()
    for col in ["hm_rmse", "hm_mae", "hm_valley_excess", "hm_second_margin", "hm_cand_row_std", "hm_dtw_rmse_ft", "hm_dtw_abs_ft"]:
        out[col] = feat[col] if col in feat.columns else np.nan
    out_path = Path(args.heatmap_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"heatmap_augmented_csv={out_path}", flush=True)
    print(out[["hm_rmse", "hm_valley_excess", "hm_dtw_rmse_ft"]].describe().to_string(), flush=True)
    return out_path


def _calibrate_hw_to_typewell(hw_gr: np.ndarray, tw_ref: np.ndarray) -> tuple[float, float]:
    src = np.asarray(hw_gr, dtype=np.float64)
    tgt = np.asarray(tw_ref, dtype=np.float64)
    v = np.isfinite(src) & np.isfinite(tgt)
    if int(v.sum()) < 20 or float(np.std(src[v])) < 1e-6:
        return 1.0, float(np.nanmean(tgt[v] - src[v])) if bool(v.any()) else 0.0
    coef, *_ = np.linalg.lstsq(np.vstack([src[v], np.ones(int(v.sum()))]).T, tgt[v], rcond=None)
    return float(coef[0]), float(coef[1])


def _two_minima(shifts: np.ndarray, costs: np.ndarray, min_sep: float, max_sep: float) -> tuple[int, int]:
    order = np.argsort(costs)
    i1 = int(order[0])
    i2 = i1
    for j in order[1:]:
        sep = abs(float(shifts[int(j)] - shifts[i1]))
        if sep >= float(min_sep) and sep <= float(max_sep):
            i2 = int(j)
            break
    return i1, i2


def heel_calibrated_two_mode_path(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    base_tvt: np.ndarray,
    *,
    shifts: np.ndarray | None = None,
    min_sep: float = 10.0,
    max_sep: float = 25.0,
    near_tie_ratio: float = 1.15,
    temperature: float | None = None,
) -> TwoModeResult:
    """Bounded constant-datum scan with legal heel GR calibration."""
    ev_mask = hw["TVT_input"].isna().to_numpy()
    ev_index = np.flatnonzero(ev_mask)
    base = _interp_to_length(np.asarray(base_tvt, dtype=np.float64), len(ev_index))
    if shifts is None:
        shifts = np.arange(-60.0, 60.01, 1.0, dtype=np.float64)
    else:
        shifts = np.asarray(shifts, dtype=np.float64)
    tw_tvt, tw_gr = _prep_typewell(tw)
    known = hw.loc[hw["TVT_input"].notna()]
    all_gr = _filled_gr(hw["GR"], fallback=float(np.nanmean(tw_gr)))
    kidx = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    tw_known = np.interp(known["TVT_input"].to_numpy(np.float64), tw_tvt, tw_gr)
    a, b = _calibrate_hw_to_typewell(all_gr[kidx], tw_known)
    tail_gr = a * all_gr[ev_index] + b
    v = np.isfinite(tail_gr)
    if int(v.sum()) < 10 or len(base) == 0:
        return TwoModeResult(base.copy(), base.copy(), 1.0, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf, False)
    costs = []
    for s in shifts:
        ref = np.interp(base + float(s), tw_tvt, tw_gr)
        r = tail_gr - ref
        costs.append(float(np.mean(np.minimum(r[v] * r[v], 3600.0))))
    costs_arr = np.asarray(costs, dtype=np.float64)
    i1, i2 = _two_minima(shifts, costs_arr, min_sep=min_sep, max_sep=max_sep)
    c1, c2 = float(costs_arr[i1]), float(costs_arr[i2])
    s1, s2 = float(shifts[i1]), float(shifts[i2])
    ratio = float(max(c1, c2) / max(min(c1, c2), 1e-9))
    sep = abs(s2 - s1)
    is_two = i2 != i1 and sep >= float(min_sep) and sep <= float(max_sep) and ratio <= float(near_tie_ratio)
    if temperature is None:
        temperature = max(2.0 * min(c1, c2), 1e-6)
    logits = np.array([-c1 / float(temperature), -c2 / float(temperature)], dtype=np.float64)
    logits -= float(np.max(logits))
    p = np.exp(np.clip(logits, -60.0, 60.0))
    p /= max(float(p.sum()), 1e-12)
    p_first = float(p[0])
    commit_shift = s1
    if is_two:
        mean_shift = p_first * s1 + (1.0 - p_first) * s2
    else:
        mean_shift = commit_shift
    return TwoModeResult(
        path=(base + mean_shift).astype(np.float64),
        commit_path=(base + commit_shift).astype(np.float64),
        p_first=p_first,
        shift_first=s1,
        shift_second=s2,
        mode_sep=sep,
        cost_first=c1,
        cost_second=c2,
        cost_ratio=ratio,
        is_two_mode=bool(is_two),
    )


def _load_train_frames(data_root: Path, wid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_root) / "train"
    hw = pd.read_csv(root / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(root / f"{wid}__typewell.csv")
    return hw, tw


def _build_one_source(
    wid: str,
    base_seg: np.ndarray,
    *,
    data_root: Path,
    hmm_step: float,
    hmm_rates: int,
    hmm_band_pad: float,
    add_heatmap: bool,
) -> dict[str, object]:
    hw, tw = _load_train_frames(data_root, wid)
    ev_mask = hw["TVT_input"].isna().to_numpy()
    n = int(ev_mask.sum())
    last_known = _last_known_tvt(hw)
    base_drift = _interp_to_length(np.asarray(base_seg, dtype=np.float64), n)
    base_abs = last_known + base_drift
    out: dict[str, object] = {"well_id": wid, "n": n}
    try:
        hmm = run_hmm2_frame(hw, tw, step=float(hmm_step), n_rates=int(hmm_rates), band_pad=float(hmm_band_pad))
        hmm_abs = _interp_to_length(hmm.pred[ev_mask], n)
        out["hmm2_mean"] = (hmm_abs - last_known).astype(np.float32)
        out["hmm2_std"] = hmm.std_eval.astype(np.float32)
    except Exception as exc:
        out["hmm_error"] = str(exc)
        hmm_abs = base_abs.copy()
        out["hmm2_mean"] = base_drift.astype(np.float32)
        out["hmm2_std"] = np.zeros(n, dtype=np.float32)
    try:
        tm = heel_calibrated_two_mode_path(hw, tw, base_abs)
        heel_abs = _interp_to_length(tm.path, n)
        heel_commit_abs = _interp_to_length(tm.commit_path, n)
        out["heel2mode_mean"] = (heel_abs - last_known).astype(np.float32)
        out["heel2mode_commit"] = (heel_commit_abs - last_known).astype(np.float32)
        out["twomode_meta"] = {
            "p_first": tm.p_first,
            "shift_first": tm.shift_first,
            "shift_second": tm.shift_second,
            "mode_sep": tm.mode_sep,
            "cost_ratio": tm.cost_ratio,
            "is_two_mode": tm.is_two_mode,
        }
    except Exception as exc:
        out["twomode_error"] = str(exc)
        heel_abs = base_abs.copy()
        heel_commit_abs = base_abs.copy()
        out["heel2mode_mean"] = base_drift.astype(np.float32)
        out["heel2mode_commit"] = base_drift.astype(np.float32)
        out["twomode_meta"] = {}
    if add_heatmap:
        out["base_heatmap"] = candidate_heatmap_evidence(hw, tw, base_abs)
        out["hmm_heatmap"] = candidate_heatmap_evidence(hw, tw, hmm_abs)
        out["heel_heatmap"] = candidate_heatmap_evidence(hw, tw, heel_abs)
    return out


def build_source_npz(args: argparse.Namespace) -> Path:
    wells, target, base_pred = load_base_prediction(args)
    slices = well_slices_from_wells(wells)
    all_wids = list(slices.keys())
    if int(args.max_wells) > 0:
        all_wids = all_wids[: int(args.max_wells)]
    jobs = []
    for wid in all_wids:
        start, end = slices[wid]
        jobs.append((wid, base_pred[start:end]))
    prefer = "threads" if str(args.joblib_backend) == "threading" else "processes"
    results = Parallel(n_jobs=int(args.n_jobs), verbose=5 if args.verbose else 0, prefer=prefer)(
        delayed(_build_one_source)(
            wid,
            seg,
            data_root=Path(args.data_root),
            hmm_step=float(args.hmm_step),
            hmm_rates=int(args.hmm_rates),
            hmm_band_pad=float(args.hmm_band_pad),
            add_heatmap=bool(args.add_heatmap),
        )
        for wid, seg in jobs
    )
    hmm_full = np.asarray(base_pred, dtype=np.float64).copy()
    hmm_std_full = np.zeros_like(hmm_full, dtype=np.float64)
    heel_full = np.asarray(base_pred, dtype=np.float64).copy()
    heel_commit_full = np.asarray(base_pred, dtype=np.float64).copy()
    meta_rows: list[dict[str, object]] = []
    for res in results:
        wid = str(res["well_id"])
        start, end = slices[wid]
        hmm_full[start:end] = _interp_to_length(np.asarray(res["hmm2_mean"], dtype=np.float64), end - start)
        hmm_std_full[start:end] = _interp_to_length(np.asarray(res["hmm2_std"], dtype=np.float64), end - start)
        heel_full[start:end] = _interp_to_length(np.asarray(res["heel2mode_mean"], dtype=np.float64), end - start)
        heel_commit_full[start:end] = _interp_to_length(np.asarray(res["heel2mode_commit"], dtype=np.float64), end - start)
        row: dict[str, object] = {"well_id": wid}
        row.update({f"twomode_{k}": v for k, v in dict(res.get("twomode_meta", {})).items()})
        for key in ("base_heatmap", "hmm_heatmap", "heel_heatmap"):
            if key in res:
                row.update({f"{key}_{k}": v for k, v in dict(res[key]).items()})
        if "hmm_error" in res:
            row["hmm_error"] = str(res["hmm_error"])
        if "twomode_error" in res:
            row["twomode_error"] = str(res["twomode_error"])
        meta_rows.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "sources.npz"
    np.savez_compressed(
        out_npz,
        wells=wells.astype(str),
        target=target.astype(np.float32),
        base=base_pred.astype(np.float32),
        hmm2_mean=hmm_full.astype(np.float32),
        hmm2_std=hmm_std_full.astype(np.float32),
        heel2mode_mean=heel_full.astype(np.float32),
        heel2mode_commit=heel_commit_full.astype(np.float32),
    )
    pd.DataFrame(meta_rows).to_csv(out_dir / "source_meta.csv", index=False)
    summary = pd.DataFrame(
        [
            score_prediction(base_pred, target, slices, "base"),
            score_prediction(hmm_full, target, slices, "hmm2_mean"),
            score_prediction(heel_full, target, slices, "heel2mode_mean"),
            score_prediction(heel_commit_full, target, slices, "heel2mode_commit"),
        ]
    )
    summary.to_csv(out_dir / "source_summary.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.utcnow().isoformat(),
                "base_npz": str(args.base_npz),
                "base_mode": str(args.base_mode),
                "base_col": str(args.base_col),
                "hmm_step": float(args.hmm_step),
                "hmm_rates": int(args.hmm_rates),
                "hmm_band_pad": float(args.hmm_band_pad),
                "max_wells": int(args.max_wells),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"out_npz={out_npz}", flush=True)
    return out_npz


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-npz", type=Path, default=DEFAULT_BASE_NPZ)
    ap.add_argument("--base-mode", choices=["stable2dcnn", "column"], default="column")
    ap.add_argument("--base-col", default="stable_softgr2")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--hmm-step", type=float, default=0.7)
    ap.add_argument("--hmm-rates", type=int, default=31)
    ap.add_argument("--hmm-band-pad", type=float, default=80.0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--joblib-backend", choices=["loky", "threading"], default="loky")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--add-heatmap", action="store_true")
    ap.add_argument("--candidate-csv", type=Path, default=None)
    ap.add_argument("--heatmap-out-csv", type=Path, default=DEFAULT_OUT_DIR / "candidate_prefix_features_heatmap.csv")
    ap.add_argument("--heatmap-top-rank", type=int, default=80)
    ap.add_argument("--heatmap-segment", type=int, default=32)
    ap.add_argument("--heatmap-tw-window", type=float, default=80.0)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_csv is not None:
        augment_candidate_csv_with_heatmap(args)
    else:
        build_source_npz(args)


if __name__ == "__main__":
    main()
