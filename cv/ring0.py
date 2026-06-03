#!/usr/bin/env python3
"""Ring 0 — 零成本诊断（不写新基础设施，纯用 fold-safe 物理曲线 + 真值）。

回答三件事：
  A) 主变量 ΔF 插补质量到底多差（level 与 shape 的共同瓶颈）。
  B) Exp1：mean/dev 分离收缩能否打掉全局 β=0.3 的妥协（可达的真增益）。
  C) Exp2：天花板 = oracle-level + 物理 shape 能落到多低（决定路线值不值得打）。

用 well split（最具代表性，邻居致密≈真实 test）跑 OOF；fold-safe：点云只用训练折井。
"""
from __future__ import annotations

import numpy as np

from cv_runner import WellStore, get_or_build_splits, folds_to_pairs
from level_engine import build_cloud, LevelEngine

CONST_ORACLE = 9.04
CF = 15.91


def collect(scheme="well"):
    """OOF 收集每口井的 true_drift / pdrift / -ΔZ / ΔF_pred / ΔF_true / conf。"""
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    splits = get_or_build_splits(store, refresh=False)
    pairs = folds_to_pairs(splits[scheme])
    cols = {k: [] for k in ["TD", "PD", "NDZ", "DFp", "DFt", "CONF", "PDm", "TDm", "CONFm", "WI"]}
    nfb = 0
    for wi, (tr, va) in enumerate(pairs):
        eng = LevelEngine(XS, FS, OW, wid)
        eng.fit(tr, store)
        for vid in va:
            w = store.wells[vid]
            td = w.hidden_truth.astype(float) - w.last_known
            res = eng.physics_curve(vid, store)
            if res is None:                       # 邻井不足 → CF（drift=0, conf=0）
                nfb += 1
                pd_, ndz, conf = np.zeros_like(td), np.zeros_like(td), np.zeros_like(td)
                dfp = np.zeros_like(td)
            else:
                pd_, ndz, dfp, conf = res["pdrift"], res["negdZ"], res["dF_pred"], res["conf"]
            dft = td - ndz                         # ΔF_true = true_drift - (-ΔZ) 精确
            cols["TD"].append(td); cols["PD"].append(pd_); cols["NDZ"].append(ndz)
            cols["DFp"].append(dfp); cols["DFt"].append(dft); cols["CONF"].append(conf)
            cols["PDm"].append(np.full_like(td, pd_.mean()))
            cols["TDm"].append(np.full_like(td, td.mean()))
            cols["CONFm"].append(np.full_like(td, conf.mean()))
            cols["WI"].append(np.full(len(td), len(cols["WI"]), dtype=int))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"[{scheme}] wells={int(out['WI'].max()+1)} rows={len(out['TD']):,} fallback={nfb}")
    return out


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def pooled_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    d = collect("well")
    TD, PD, NDZ, DFp, DFt = d["TD"], d["PD"], d["NDZ"], d["DFp"], d["DFt"]
    CONF, PDm, TDm, CONFm = d["CONF"], d["PDm"], d["TDm"], d["CONFm"]

    print(f"\n参考刻度：CF {CF} / const-oracle {CONST_ORACLE} / smooth-201 0.39")
    print(f"CF (drift=0) 复核 row-wt RMSE = {rmse(TD):.3f}")

    # ---- 现行引擎复核（应≈13.4）----
    eng_pred = np.clip(0.3 * CONF * PD, -20, 20)
    print(f"现行引擎 (β0.3·conf·clip20) 复核   = {rmse(eng_pred - TD):.3f}")

    # ---- A) 主变量：ΔF 插补质量 ----
    print("\n[A] 主变量 ΔF 插补质量（−ΔZ 精确，全部误差来自 ΔF）")
    print(f"    ΔF_pred vs ΔF_true   pooled corr = {pooled_corr(DFp, DFt):+.3f}  "
          f"RMSE = {rmse(DFp - DFt):.2f} ft")
    # 段内去均值（shape 部分）
    def demean_by_well(x):
        # PDm/TDm 已是 pdrift/true 的井均值；这里对任意量按井去均值
        wi = d["WI"]; s = np.bincount(wi, x) / np.bincount(wi)
        return x - s[wi]
    DFp_dev, DFt_dev = demean_by_well(DFp), demean_by_well(DFt)
    print(f"    ΔF shape(去均值)     pooled corr = {pooled_corr(DFp_dev, DFt_dev):+.3f}")

    # ---- E) shape 从哪来：已知 -ΔZ 几何 vs 物理 pdrift ----
    print("\n[E] swing(段内 shape) 来源分解")
    TD_dev = TD - TDm
    print(f"    true swing 标准差        = {np.sqrt(np.mean(TD_dev**2)):.2f} ft (=const-oracle {CONST_ORACLE})")
    print(f"    de-meaned(-ΔZ) vs true swing corr = {pooled_corr(demean_by_well(NDZ), TD_dev):+.3f}  (只用已知几何)")
    print(f"    de-meaned(pdrift) vs true swing corr = {pooled_corr(PD - PDm, TD_dev):+.3f}  (物理 full)")

    # ---- Exp1：mean/dev 分离收缩（fold-safe 可达）----
    print("\n[Exp1] mean/dev 分离收缩  pred = conf·(bm·mean + bs·dev)")
    PD_dev = PD - PDm
    best = (1e9, None)
    grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    for bm in grid:
        for bs in grid:
            pred = CONFm * bm * PDm + CONF * bs * PD_dev
            r = rmse(pred - TD)
            if r < best[0]:
                best = (r, (bm, bs))
    # 对照：单一全局 β（bm=bs）
    single = min((rmse(CONF * b * PD - TD), b) for b in grid)
    print(f"    单一全局 β 最优:  β={single[1]:.1f}  RMSE={single[0]:.3f}")
    print(f"    mean/dev 分离最优: bm={best[1][0]:.1f} bs={best[1][1]:.1f}  RMSE={best[0]:.3f}")
    # 加 clip 看是否再降
    bm, bs = best[1]
    for clip in [15, 20, 30, 1e9]:
        pred = np.clip(CONFm * bm * PDm + CONF * bs * PD_dev, -clip, clip)
        print(f"      + clip{clip if clip<1e8 else '∞':>4}: RMSE={rmse(pred-TD):.3f}")

    # ---- Exp2：天花板 = oracle level + 物理 shape ----
    print("\n[Exp2] 天花板  pred = oracle_mean + bs·conf·dev")
    print(f"    oracle level only (bs=0) = {rmse(TDm - TD):.3f}  (应≈const-oracle {CONST_ORACLE})")
    best2 = (1e9, None)
    for bs in np.round(np.arange(0.0, 1.51, 0.05), 2):
        pred = TDm + bs * CONF * PD_dev
        r = rmse(pred - TD)
        if r < best2[0]:
            best2 = (r, bs)
    print(f"    + 物理 shape 最优: bs={best2[1]:.2f}  RMSE={best2[0]:.3f}   ← 路线天花板")
    # 对照：oracle level + 只用已知 -ΔZ 的 shape（无 ΔF）
    NDZ_dev = demean_by_well(NDZ)
    best3 = min((rmse(TDm + bs * NDZ_dev - TD), bs) for bs in np.round(np.arange(0, 1.51, 0.05), 2))
    print(f"    对照 oracle level + 只用 -ΔZ shape(无ΔF): bs={best3[1]:.2f} RMSE={best3[0]:.3f}")
    # 终极：oracle level + oracle shape 缩放（ΔF 完美时）
    best4 = min((rmse(TDm + bs * DFt_dev + NDZ_dev - TD), bs) for bs in [1.0])
    print(f"    终极 oracle level + 完美ΔF shape: RMSE={best4[0]:.3f}  (smooth-oracle 量级)")


if __name__ == "__main__":
    main()
