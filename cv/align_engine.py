#!/usr/bin/env python3
"""ROGII Stage B · shape 对齐引擎 v1（金牌核心）。

相对 step4 的关键升级：
  1. **top-k 典井全库检索**（assigned 仅 30/773 是最像；用 prefix-corr 矩阵选 top-k）。
  2. **z-score 标定-免疫发射**：well GR 与典井 GR 各自标准化后比较 → 自动消掉 affine(slope/intercept)，
     对齐看的是"形状/条形码"而非幅度。
  3. **多尺度去噪**（中值+Savgol）压 GR 噪声再对齐。
  4. **多典井共识**：top-k 各跑一次 Viterbi，按总代价选最优（或加权）。
  5. **分解度量**：drift_corr(level) 与 shape_corr(去均值后的段内形状) 分开看——
     破 9 必须 shape_corr 显著>0（step4 只有 level corr 0.24、shape≈0）。

用整条观测 hidden GR(offline)。oracle 不参与，只在评测里对比。
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import cv_runner as cv
from cv_runner import WellStore

EPS = 1e-9
CORR_MATRIX = "/root/ROGII/eda/eda2_artifacts/well_to_typewell_prefix_corr_matrix.csv"

P = dict(step=20, win=70.0, res=0.5, gamma=0.05, alpha=0.4, smooth=151,
         topk=5, denoise_med=9, denoise_sg=31)


def denoise(g, prm):
    """多尺度去噪：中值(去尖峰)+Savgol(去高频)，保留 NaN 掩码。"""
    g = np.asarray(g, float)
    out = g.copy()
    fin = np.isfinite(g)
    if fin.sum() < 5:
        return g
    x = pd.Series(g).interpolate(limit_direction="both").to_numpy()
    k = prm["denoise_med"] | 1
    x = medfilt(x, kernel_size=min(k, len(x) - (1 - len(x) % 2)))
    w = min(prm["denoise_sg"] | 1, len(x) - (1 - len(x) % 2))
    if w >= 5:
        x = savgol_filter(x, w, 3)
    out[fin] = x[fin]
    out[~fin] = np.nan
    return out


def znorm(a):
    a = np.asarray(a, float)
    m = np.nanmean(a); s = np.nanstd(a)
    return (a - m) / s if s > EPS else a - m


def viterbi(w, tw_tvt, tw_gr_z, g_z, prm):
    """g_z: 标准化去噪后的隐藏段 GR(锚点采样)。tw_gr_z: 标准化典井 GR(按 TVT 网格取值用)。返回 (path_tvt, mean_cost)。"""
    hs = w.ps_idx
    idx = np.arange(0, w.n_hidden, prm["step"])
    amd = w.md[hs:][idx]
    g = g_z[idx]
    A = len(idx)
    grid = w.last_known + np.arange(-prm["win"], prm["win"] + prm["res"], prm["res"])
    S = len(grid)
    cal = np.interp(grid, tw_tvt, tw_gr_z)               # 典井标准化 GR 在网格 TVT 上
    E = np.zeros((A, S))
    for i in range(A):
        E[i] = (cal - g[i]) ** 2 if np.isfinite(g[i]) else 0.0
    bins = np.arange(S)
    T = prm["gamma"] * (bins[:, None] - bins[None, :]) ** 2
    cost = E[0] + prm["alpha"] * prm["gamma"] * ((grid - w.last_known) / prm["res"]) ** 2
    back = np.zeros((A, S), np.int32)
    for i in range(1, A):
        M = cost[:, None] + T
        back[i] = M.argmin(0)
        cost = E[i] + M.min(0)
    s = int(cost.argmin()); mean_cost = float(cost.min() / A)
    path = np.empty(A, np.int32)
    for i in range(A - 1, -1, -1):
        path[i] = s; s = back[i][s]
    full = np.interp(w.md[hs:], amd, grid[path])
    return full, mean_cost


class Aligner:
    def __init__(self, store, prm):
        self.store = store; self.prm = prm
        cm = pd.read_csv(CORR_MATRIX, index_col=0)
        cm.index = cm.index.astype(str); cm.columns = cm.columns.astype(str)
        self.cm = cm

    def topk_typewells(self, wid):
        if wid in self.cm.index:
            r = self.cm.loc[wid].sort_values(ascending=False)
            cand = [c for c in r.index[:self.prm["topk"] + 2]][: self.prm["topk"]]
            if wid not in cand:
                cand = [wid] + cand[:-1]
            return cand
        return [wid]

    def predict(self, wid):
        w = self.store.wells[wid]; prm = self.prm
        gz = znorm(denoise(w.gr[w.ps_idx:], prm))
        best = None
        for twid in self.topk_typewells(wid):
            tw = self.store.load_typewell(twid).dropna(subset=["TVT", "GR"]).sort_values("TVT")
            if len(tw) < 20:
                continue
            tw_tvt = tw["TVT"].to_numpy(float)
            tw_z = znorm(denoise(tw["GR"].to_numpy(float), prm))
            try:
                path, c = viterbi(w, tw_tvt, tw_z, gz, prm)
            except Exception:
                continue
            if best is None or c < best[1]:
                best = (path, c, twid)
        if best is None:
            return np.full(w.n_hidden, w.last_known), 1e9, "none"
        path = best[0]
        sm = min(prm["smooth"] | 1, len(path) - (1 - len(path) % 2))
        if sm >= 5:
            path = savgol_filter(path, sm, 2)
        return path, best[1], best[2]


def evaluate(store, prm, ids):
    rows = []
    al = Aligner(store, prm)
    for wid in ids:
        w = store.wells[wid]
        pred, cost, twid = al.predict(wid)
        t = w.hidden_truth
        err = pred - t
        # 去均值得到形状分量
        ps = pred - pred.mean(); ts = t - t.mean()
        rows.append(dict(well_id=wid, n=w.n_hidden, rmse=float(np.sqrt(np.mean(err**2))),
                         sse=float(np.sum(err**2)),
                         pred_drift=float(pred.mean() - w.last_known),
                         true_drift=float(t.mean() - w.last_known),
                         shape_dot=float(np.sum(ps*ts)), shape_pp=float(np.sum(ps*ps)),
                         shape_tt=float(np.sum(ts*ts)), cost=cost,
                         cf_sse=float(np.sum((w.last_known - t)**2))))
    return pd.DataFrame(rows)


def summarize(df, tag=""):
    rw = np.sqrt((df["rmse"]**2 * df["n"]).sum() / df["n"].sum())
    cf = np.sqrt(df["cf_sse"].sum() / df["n"].sum())
    drift_corr = df[["pred_drift", "true_drift"]].corr().iloc[0, 1]
    shape_corr = df["shape_dot"].sum() / (np.sqrt(df["shape_pp"].sum()*df["shape_tt"].sum()) + EPS)
    win = int((df["rmse"] < np.sqrt(df["cf_sse"]/df["n"]) - 0.1).sum())
    print(f"  {tag:24s} rw_rmse={rw:.3f} (CF={cf:.3f})  drift_corr={drift_corr:+.3f}  "
          f"shape_corr={shape_corr:+.3f}  win={win}/{len(df)}")
    return rw, drift_corr, shape_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    store = WellStore()
    rng = np.random.default_rng(0)
    ids = list(rng.choice(store.ids(), args.sample, replace=False)) if args.sample else store.ids()
    print(f"[align v1] wells={len(ids)}  (ref CF=15.9, const-oracle=9.04; step4 was rw15.6 drift0.24 shape~0)")
    if args.tune:
        for gamma in (0.02, 0.05, 0.1):
            for win in (50.0, 70.0):
                for topk in (1, 5):
                    prm = {**P, "gamma": gamma, "win": win, "topk": topk}
                    df = evaluate(store, prm, ids)
                    summarize(df, f"g{gamma} w{win} k{topk}")
    else:
        df = evaluate(store, P, ids)
        summarize(df, "align_v1")


if __name__ == "__main__":
    main()
