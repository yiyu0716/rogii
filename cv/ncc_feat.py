#!/usr/bin/env python3
"""Stage C v2 特征 — 多尺度 GR NCC（对本井 typewell，fold-independent，一次性缓存）。

对每个 eval 行，取 HW GR 窗口（中心，半宽 w 行），在 typewell GR(按 TVT 排序) 的
[seed−radius, seed+radius] 带内滑动求 Pearson r，峰值位置 → tvt_est，drift = tvt_est − last_known。
seed = last_anchor_tvt（固定带，整段向量化：每井每尺度一次矩阵乘）。
多尺度半宽 {8,15,25} + softmax(峰值 corr) 融合 drift。输出按 (well_id,row_idx) 缓存，可直接 merge。

不跨井、不碰隐藏 TVT → 与折无关、无泄露。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cv_runner import WellStore, ART_DIR

HALF_WIDTHS = [8, 15, 25]
RADIUS = 150.0
OUT = ART_DIR / "ncc_feat.pkl"


def impute_gr(gr):
    return pd.Series(gr).ffill().bfill().to_numpy(float)


def ncc_for_well(w, store):
    hs = w.ps_idx; n = w.n_hidden; lk = w.last_known
    gr_full = impute_gr(w.gr.astype(float))
    tw = store.load_typewell(w.well_id)
    out = {f"ncc_drift_w{hw}": np.zeros(n, np.float32) for hw in HALF_WIDTHS}
    out.update({f"ncc_corr_w{hw}": np.zeros(n, np.float32) for hw in HALF_WIDTHS})
    out["ncc_drift_blend"] = np.zeros(n, np.float32)
    if tw.empty or "TVT" not in tw or "GR" not in tw:
        return out
    twd = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = twd["TVT"].to_numpy(float); tw_gr = twd["GR"].to_numpy(float)
    if len(tw_gr) < 60:
        return out
    # 限定候选带（固定 seed=lk）
    band = (tw_tvt >= lk - RADIUS) & (tw_tvt <= lk + RADIUS)
    tg = tw_gr[band]; tt = tw_tvt[band]
    if len(tg) < 60:
        return out

    ests = []; corrs = []
    for hw in HALF_WIDTHS:
        L = 2 * hw
        if len(tg) <= L or n < 1:
            ests.append(np.full(n, np.nan)); corrs.append(np.zeros(n)); continue
        # eval 窗口（中心半宽 hw 行；边界用 clip 索引）
        centers = np.arange(hs, hs + n)
        lo = np.clip(centers - hw, 0, len(gr_full) - L)
        Wm = gr_full[lo[:, None] + np.arange(L)[None, :]]          # (n, L)
        Wm = Wm - Wm.mean(1, keepdims=True)
        Wm /= Wm.std(1, keepdims=True) + 1e-6
        # 候选窗口
        C = sliding_window_view(tg, L)                            # (n_cand, L)
        Cn = C - C.mean(1, keepdims=True)
        Cn /= Cn.std(1, keepdims=True) + 1e-6
        cand_tvt = tt[(L // 2):(L // 2) + len(C)]                 # 候选中心 TVT
        corr = (Wm @ Cn.T) / L                                    # (n, n_cand)
        bi = corr.argmax(1)
        ests.append(cand_tvt[bi] - lk)
        corrs.append(corr[np.arange(n), bi])

    ests = np.array(ests); corrs = np.array(corrs)               # (S, n)
    for s, hw in enumerate(HALF_WIDTHS):
        e = ests[s]; e = np.where(np.isfinite(e), e, 0.0)
        out[f"ncc_drift_w{hw}"] = e.astype(np.float32)
        out[f"ncc_corr_w{hw}"] = np.nan_to_num(corrs[s]).astype(np.float32)
    # softmax(corr) 融合
    cc = np.nan_to_num(corrs); ee = np.nan_to_num(ests)
    wgt = np.exp(4.0 * cc); wgt /= wgt.sum(0, keepdims=True) + 1e-9
    out["ncc_drift_blend"] = (wgt * ee).sum(0).astype(np.float32)
    return out


def main():
    store = WellStore()
    ids = store.ids()
    parts = []
    for k, vid in enumerate(ids):
        w = store.wells[vid]
        feats = ncc_for_well(w, store)
        df = pd.DataFrame(feats)
        df.insert(0, "row_idx", np.arange(w.ps_idx, len(w.md)))
        df.insert(0, "well_id", vid)
        parts.append(df)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(ids)} wells")
    out = pd.concat(parts, ignore_index=True)
    out.to_pickle(OUT)
    print(f"wrote {len(out):,} rows -> {OUT}")
    # 快速诊断：blend NCC drift 的 shape/level 信息（对照真值需在 stage_c 里 merge）
    print(out[[c for c in out.columns if c.startswith("ncc_")]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
