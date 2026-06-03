#!/usr/bin/env python3
"""Stage C v3 特征 — typewell GR 粒子滤波 drift（fold-independent，缓存）。

沿 MD 顺序跟踪 TVT 位置：P 个粒子，随机游走转移 + GR 发射似然（HW 与 typewell GR 各自 z-score 标定）。
比 NCC 的 argmax 更平滑（时序约束）。输出 per-row pf_drift + pf_ess（置信）。不跨井、不碰隐藏 TVT。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from cv_runner import WellStore, ART_DIR

P = 300
Q = 0.4          # 随机游走 sigma (ft/row)
R_SIGMA = 1.0    # 发射 sigma (z-score 单位)
INIT_SIGMA = 3.0
OUT = ART_DIR / "pf_feat.pkl"


def impute_gr(gr):
    return pd.Series(gr).ffill().bfill().to_numpy(float)


def pf_for_well(w, store, rng):
    hs, n, lk = w.ps_idx, w.n_hidden, w.last_known
    out_drift = np.zeros(n, np.float32); out_ess = np.zeros(n, np.float32)
    if n < 1:
        return out_drift, out_ess
    gr = impute_gr(w.gr.astype(float)); md = w.md.astype(float)
    tw = store.load_typewell(w.well_id)
    if tw.empty or "TVT" not in tw or "GR" not in tw:
        return out_drift, out_ess
    twd = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tt = twd["TVT"].to_numpy(float); tg = twd["GR"].to_numpy(float)
    if len(tg) < 60:
        return out_drift, out_ess
    # z-score 标定
    amu, asd = np.nanmean(gr[:hs]), np.nanstd(gr[:hs]) + 1e-6
    hwz = (gr - amu) / asd
    twz = (tg - np.mean(tg)) / (np.std(tg) + 1e-6)
    lo, hi = tt[0], tt[-1]

    parts = lk + rng.normal(0, INIT_SIGMA, P)
    for i in range(n):
        ridx = hs + i
        parts = parts + rng.normal(0, Q, P)
        parts = np.clip(parts, lo, hi)
        exp = np.interp(parts, tt, twz)
        wt = np.exp(-0.5 * ((hwz[ridx] - exp) / R_SIGMA) ** 2) + 1e-12
        wt /= wt.sum()
        out_drift[i] = np.dot(wt, parts) - lk
        ess = 1.0 / np.sum(wt * wt)
        out_ess[i] = ess / P
        if ess < P / 2:
            idx = rng.choice(P, P, p=wt)
            parts = parts[idx]
    return out_drift, out_ess


def main():
    store = WellStore(); ids = store.ids()
    rng = np.random.default_rng(0)
    parts = []
    import time; t0 = time.time()
    for k, vid in enumerate(ids):
        w = store.wells[vid]
        d, e = pf_for_well(w, store, rng)
        df = pd.DataFrame({"well_id": vid, "row_idx": np.arange(w.ps_idx, len(w.md)),
                           "pf_drift": d, "pf_ess": e})
        parts.append(df)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_pickle(OUT)
    print(f"wrote {len(out):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
