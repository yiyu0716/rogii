#!/usr/bin/env python3
"""Stage C v1 — 干净的行级 drift-target GBDT（从零搭，全程自控、CV 诚实）。

目标 = per-row drift = TVT_i − last_known。
v1 特征：
  core  物理 drift（fold-safe，邻井 ANCC 点云）：pdrift / dF / negdZ / conf
  anchor 线性外推 drift：tvt_rate × md_from_anchor，及 prefix 斜率/步长
  geom  隐藏段已知几何：dz=Z_i−Z_PS、md_from_anchor、dxy、frac 位置、坐标
  gr    imputed GR_i、GR 归一化、anchor GR 统计
唯一 fold 相关特征 = 物理 drift（每口井在自己被留出的折里算）；其余 within-well、与折无关。
GroupKFold = 冻结 well 折；HGB；重建 TVT → pooled RMSE + worst-SSE；可选 Savgol 后处理。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

from cv_runner import WellStore, get_or_build_splits, folds_to_pairs, ART_DIR
from level_engine import build_cloud, LevelEngine

CF, CONST_ORACLE, CEILING, FRONTIER = 15.91, 9.04, 7.69, 9.85
CACHE = ART_DIR / "stage_c_rows.pkl"


def impute_gr(gr):
    """沿 MD 前后填充缺失 GR。"""
    s = pd.Series(gr).ffill().bfill()
    return s.to_numpy(float)


def build_well_rows(vid, eng, store):
    """一口井 eval-zone 的行级特征 + 目标。物理用传入的 fold-safe eng。"""
    w = store.wells[vid]
    hs = w.ps_idx; n = w.n_hidden
    x = w.x.astype(float); y = w.y.astype(float); z = w.z.astype(float)
    md = w.md.astype(float); lk = w.last_known
    x0, y0, zps, md0 = x[hs - 1], y[hs - 1], z[hs - 1], md[hs - 1]

    # anchor 趋势
    ti = w.tvt_input
    tail = ti[max(0, hs - 200):hs]
    tail = tail[np.isfinite(tail)]
    mdtail = md[max(0, hs - 200):hs][-len(tail):] if len(tail) else md[:0]
    if len(tail) >= 2 and (mdtail[-1] - mdtail[0]) != 0:
        tvt_rate = (tail[-1] - tail[0]) / (mdtail[-1] - mdtail[0])
    else:
        tvt_rate = 0.0
    t100 = ti[max(0, hs - 100):hs]; t100 = t100[np.isfinite(t100)]
    slope100 = np.polyfit(np.arange(len(t100)), t100, 1)[0] if len(t100) > 10 else 0.0
    ap = ti[:hs]; agr = impute_gr(w.gr[:hs].astype(float))
    anchor_gr_mean = float(np.nanmean(agr)) if len(agr) else 0.0
    anchor_gr_std = float(np.nanstd(agr)) if len(agr) else 0.0

    md_from = md[hs:] - md0
    dz = z[hs:] - zps
    dxy = np.hypot(x[hs:] - x0, y[hs:] - y0)
    frac = np.arange(n) / max(n - 1, 1)
    gr_full = impute_gr(w.gr.astype(float))
    gr_ev = gr_full[hs:]
    gr_norm = (gr_ev - anchor_gr_mean) / max(anchor_gr_std, 1.0)

    res = eng.physics_curve(vid, store)
    if res is None:
        pdrift = np.zeros(n); dF = np.zeros(n); conf = np.zeros(n)
        negdZ = -dz
    else:
        pdrift, dF, conf, negdZ = res["pdrift"], res["dF_pred"], res["conf"], res["negdZ"]

    return {
        "well_id": np.full(n, vid), "fold": np.full(n, eng_fold_lookup[vid], dtype=np.int8),
        "row_idx": np.arange(hs, len(md)),
        # core 物理
        "pdrift": pdrift.astype(np.float32), "dF": dF.astype(np.float32),
        "negdZ": negdZ.astype(np.float32), "conf": conf.astype(np.float32),
        # anchor
        "extrap_drift": (tvt_rate * md_from).astype(np.float32),
        "tvt_rate": np.full(n, tvt_rate, np.float32), "slope100": np.full(n, slope100, np.float32),
        # geom
        "md_from": md_from.astype(np.float32), "dz": dz.astype(np.float32),
        "dxy": dxy.astype(np.float32), "frac": frac.astype(np.float32),
        "x": x[hs:].astype(np.float32), "y": y[hs:].astype(np.float32), "z": z[hs:].astype(np.float32),
        "x0": np.full(n, x0, np.float32), "y0": np.full(n, y0, np.float32),
        "z0": np.full(n, zps, np.float32), "lk": np.full(n, lk, np.float32),
        "n_hidden": np.full(n, n, np.int32),
        # gr
        "gr": gr_ev.astype(np.float32), "gr_norm": gr_norm.astype(np.float32),
        "agr_mean": np.full(n, anchor_gr_mean, np.float32),
        "agr_std": np.full(n, anchor_gr_std, np.float32),
        # target
        "target": (w.hidden_truth.astype(float) - lk).astype(np.float32),
    }


FEATS = ["pdrift", "dF", "negdZ", "conf", "extrap_drift", "tvt_rate", "slope100",
         "md_from", "dz", "dxy", "frac", "x", "y", "z", "x0", "y0", "z0", "lk",
         "gr", "gr_norm", "agr_mean", "agr_std"]

eng_fold_lookup = {}


def build_all():
    if CACHE.exists():
        print(f"[cache] {CACHE}")
        return pd.read_pickle(CACHE)
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    fold_of = get_or_build_splits(store, refresh=False)["well"]
    eng_fold_lookup.update(fold_of)
    pairs = folds_to_pairs(fold_of)
    parts = []
    for fi, (tr, va) in enumerate(pairs):
        eng = LevelEngine(XS, FS, OW, wid)
        eng.fit(tr, store)
        for vid in va:
            parts.append(pd.DataFrame(build_well_rows(vid, eng, store)))
        print(f"  fold {fi}: cum rows={sum(len(p) for p in parts):,}")
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(CACHE)
    return df


def savgol_per_well(df, drift_col, win=17, order=3):
    from scipy.signal import savgol_filter
    out = df[drift_col].to_numpy(float).copy()
    for _, idx in df.groupby("well_id").indices.items():
        idx = np.sort(idx)
        if len(idx) > win:
            out[idx] = savgol_filter(out[idx], win, order)
    return out


def main():
    print("[1/3] 构行级特征（fold-safe 物理）...")
    df = build_all()
    print(f"      rows={len(df):,}  wells={df['well_id'].nunique()}  feats={len(FEATS)}")

    print("[2/3] 行级 GBDT（GroupKFold=well）...")
    X = df[FEATS].to_numpy(np.float32); y = df["target"].to_numpy(np.float32)
    folds = df["fold"].to_numpy()
    oof = np.zeros(len(df), np.float32)
    for f in sorted(np.unique(folds)):
        tr = folds != f; va = ~tr
        m = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_depth=None, max_leaf_nodes=63,
            l2_regularization=1.0, min_samples_leaf=200, random_state=0)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
        print(f"      fold {f} done")
    df["oof_drift"] = oof

    print("[3/3] 重建 TVT，评分 ...")
    def pooled(drift):
        err = drift - y
        return float(np.sqrt(np.mean(err ** 2)))
    raw = pooled(oof)
    sg = savgol_per_well(df, "oof_drift")
    rsg = pooled(sg)
    print(f"\n刻度：CF {CF} / const-oracle {CONST_ORACLE} / 前沿 {FRONTIER} / 天花板 {CEILING}")
    print(f"  Stage C v1 (HGB)          pooled RMSE = {raw:.3f}")
    print(f"  + Savgol(17,3)            pooled RMSE = {rsg:.3f}")
    # 物理引擎单体对照
    eng_only = np.clip(0.3 * df["conf"] * df["pdrift"], -20, 20)
    print(f"  对照 物理引擎单体          pooled RMSE = {pooled(eng_only):.3f}")
    print(f"  对照 CF                    pooled RMSE = {pooled(np.zeros_like(y)):.3f}")

    # 特征重要性（permutation 近似：用 HGB 的 训练集 split-based 不可得，用相关性代理）
    best = sg if rsg < raw else oof
    df["pred_drift"] = best
    # worst-by-SSE
    g = df.assign(se=(best - y) ** 2).groupby("well_id").agg(
        n=("target", "size"), sse=("se", "sum"))
    g["rmse"] = np.sqrt(g["sse"] / g["n"])
    worst = g.sort_values("sse", ascending=False).head(15)
    print("\nworst-by-SSE top15:")
    print(worst.to_string())
    df[["well_id", "row_idx", "pred_drift", "target"]].to_pickle(ART_DIR / "stage_c_oof.pkl")


if __name__ == "__main__":
    main()
