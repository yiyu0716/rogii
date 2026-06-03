#!/usr/bin/env python3
"""Ring 2+3 — level GBM + 物理 shape 融合。

Ring 0 已证：瓶颈是 per-well LEVEL（段内均值 drift），shape 已够（oracle-level+物理shape=7.69）。
策略：
  LEVEL  ← 学一个 fold-safe 预测器（HGB+CatBoost）预测 hidden-mean-drift，
           特征 = PS 坐标/几何（隐藏段 X/Y/Z/MD 全已知）+ 物理 mean(ΔF) + GR 统计 + prefix 趋势。
  SHAPE  ← 物理 drift 去均值 bs·conf·(pdrift−mean)。
  融合   pred_i = last_known + level_hat + bs·conf·(pdrift_i − mean(pdrift))。
所有特征不碰隐藏段 TVT；物理特征 fold-safe（点云只用训练折井）；GroupKFold = 冻结 well split。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

from cv_runner import WellStore, get_or_build_splits, folds_to_pairs
from level_engine import build_cloud, LevelEngine

CONST_ORACLE, CF, CEILING = 9.04, 15.91, 7.69
N_FOLDS = 5

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False


def safe(v, default=0.0):
    return float(v) if np.isfinite(v) else default


def build_features(scheme="well"):
    """每口井一行：特征 + 目标 mean-drift + 保留物理曲线供融合。fold-safe 物理特征。"""
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    fold_of = get_or_build_splits(store, refresh=False)[scheme]
    pairs = folds_to_pairs(fold_of)

    rows, curves = {}, {}
    for tr, va in pairs:
        eng = LevelEngine(XS, FS, OW, wid)
        eng.fit(tr, store)
        for vid in va:
            w = store.wells[vid]
            hs = w.ps_idx; n = w.n_hidden
            x = w.x.astype(float); y = w.y.astype(float); z = w.z.astype(float)
            md = w.md.astype(float); gr = w.gr.astype(float)
            lk = w.last_known
            x0, y0, z0 = x[hs - 1], y[hs - 1], z[hs - 1]
            dx, dy = x[-1] - x0, y[-1] - y0
            norm = np.hypot(dx, dy) or 1.0
            dz_h = z[hs:] - z0
            grp, grh = gr[:hs], gr[hs:]
            # prefix TVT 趋势（已知 TVT_input）
            ti = w.tvt_input[:hs]
            tail = ti[np.isfinite(ti)][-300:]
            slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] if len(tail) > 20 else 0.0
            res = eng.physics_curve(vid, store)
            if res is None:
                pdrift = np.zeros(n); conf = np.zeros(n); dF = np.zeros(n)
            else:
                pdrift, conf, dF = res["pdrift"], res["conf"], res["dF_pred"]
            f = {
                "well_id": vid, "fold": fold_of[vid], "n_hidden": n, "lk": lk,
                "x0": x0, "y0": y0, "z0": z0,
                "md_len": md[-1] - md[hs - 1], "dxy_total": np.hypot(dx, dy),
                "az_sin": dy / norm, "az_cos": dx / norm,
                "dz_total": z[-1] - z0, "dz_mean": safe(dz_h.mean()), "dz_std": safe(dz_h.std()),
                "dz_min": safe(dz_h.min()), "dz_max": safe(dz_h.max()),
                "dF_mean": safe(np.nanmean(dF)), "dF_end": safe(dF[-1]),
                "pdrift_mean": safe(np.nanmean(pdrift)), "pdrift_end": safe(pdrift[-1]),
                "conf_mean": safe(np.nanmean(conf)),
                "grp_mean": safe(np.nanmean(grp)), "grp_std": safe(np.nanstd(grp)),
                "grh_mean": safe(np.nanmean(grh)), "grh_std": safe(np.nanstd(grh)),
                "grh_miss": safe(np.mean(~np.isfinite(grh))),
                "prefix_slope": safe(slope),
                "target": float(w.hidden_truth.mean()) - lk,   # per-well mean drift
            }
            rows[vid] = f
            curves[vid] = (w.hidden_truth.astype(float) - lk, pdrift, conf)
    df = pd.DataFrame(list(rows.values()))
    return df, curves


FEATS = ["lk", "x0", "y0", "z0", "md_len", "dxy_total", "az_sin", "az_cos",
         "dz_total", "dz_mean", "dz_std", "dz_min", "dz_max",
         "dF_mean", "dF_end", "pdrift_mean", "pdrift_end", "conf_mean",
         "grp_mean", "grp_std", "grh_mean", "grh_std", "grh_miss", "prefix_slope"]


def oof_level(df):
    """GroupKFold = 冻结 well fold；HGB + CatBoost 平均 → OOF per-well level。"""
    oof_hgb = np.zeros(len(df)); oof_cat = np.zeros(len(df))
    X = df[FEATS].to_numpy(float); ytarget = df["target"].to_numpy(float)
    for f in range(N_FOLDS):
        tr = df["fold"].to_numpy() != f
        va = ~tr
        hgb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_depth=3,
            l2_regularization=1.0, min_samples_leaf=20, random_state=0)
        hgb.fit(X[tr], ytarget[tr]); oof_hgb[va] = hgb.predict(X[va])
        if HAS_CAT:
            cat = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=4,
                                    l2_leaf_reg=6.0, random_seed=0, verbose=False)
            cat.fit(X[tr], ytarget[tr]); oof_cat[va] = cat.predict(X[va])
    return oof_hgb, (oof_cat if HAS_CAT else oof_hgb)


def pooled_rmse(level_of, curves, bs, clip=20.0):
    sse = tot = 0.0
    for vid, lvl in level_of.items():
        td, pdrift, conf = curves[vid]
        dev = pdrift - np.nanmean(pdrift)
        drift = lvl + np.clip(bs * conf * dev, -clip, clip)
        e = drift - td
        sse += np.nansum(e ** 2); tot += len(e)
    return float(np.sqrt(sse / tot))


def main():
    print("[1/3] 构特征（fold-safe 物理 OOF）...")
    df, curves = build_features("well")
    print(f"      wells={len(df)}  catboost={'on' if HAS_CAT else 'OFF'}")

    print("[2/3] 训练 level GBM（GroupKFold=well）...")
    oof_hgb, oof_cat = oof_level(df)
    df["level_hgb"] = oof_hgb; df["level_cat"] = oof_cat
    df["level"] = 0.5 * (oof_hgb + oof_cat)
    tgt = df["target"].to_numpy()
    for nm, col in [("HGB", oof_hgb), ("CatBoost", oof_cat), ("avg", df["level"].to_numpy())]:
        lvl_rmse = np.sqrt(np.mean((col - tgt) ** 2))
        corr = np.corrcoef(col, tgt)[0, 1]
        print(f"      level {nm:8s}: per-well-mean RMSE={lvl_rmse:.3f}  corr={corr:+.3f}")
    print(f"      (target std={tgt.std():.3f}; 完美 level→ 段内 RMSE 天花板 ~{CEILING})")

    print("[3/3] 融合 level + 物理 shape，扫 bs ...")
    level_of = dict(zip(df["well_id"], df["level"]))
    # 基线：level-only（bs=0）
    base = pooled_rmse(level_of, curves, bs=0.0)
    print(f"      level-only (bs=0)        pooled RMSE = {base:.3f}")
    best = (1e9, None)
    for bs in np.round(np.arange(0.0, 0.81, 0.05), 2):
        r = pooled_rmse(level_of, curves, bs=bs)
        if r < best[0]:
            best = (r, bs)
    print(f"      + 物理 shape 最优 bs={best[1]:.2f}  pooled RMSE = {best[0]:.3f}")
    print(f"\n刻度：CF {CF} / 前沿 ~9.5 / const-oracle {CONST_ORACLE} / 天花板 {CEILING} / smooth 0.39")
    # 用 oracle level 对照（验证融合实现一致）
    orc = {vid: curves[vid][0].mean() for vid in curves}
    print(f"对照 oracle-level+shape bs=0.35 = {pooled_rmse(orc, curves, 0.35):.3f}  (应≈7.69)")

    df.to_csv("artifacts/level_gbm_oof.csv", index=False)


if __name__ == "__main__":
    main()
