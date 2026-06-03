#!/usr/bin/env python3
"""Stage C v2 — 在 v1 行级特征上加 NCC 聚合信号 + 更快 GBDT，可选 CatBoost/NNLS。

复用 v1 缓存 stage_c_rows.pkl + ncc_feat.pkl（fold-independent）。
NCC 经诊断 per-row 太噪（std~50ft、shape_corr≈0），故只用**按井聚合**的 level 线索：
  ncc_mean_drift（井内中位 blend drift）、ncc_corr_mean（峰值置信）、ncc_drift_smooth（重平滑）。
更快 GBDT：max_iter=300、max_leaf_nodes=31、行下采样可选。GroupKFold=well。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
from cv_runner import ART_DIR
from stage_c import FEATS as V1_FEATS, savgol_per_well

CF, CONST_ORACLE, CEILING, FRONTIER = 15.91, 9.04, 7.69, 9.85


def main():
    df = pd.read_pickle(ART_DIR / "stage_c_rows.pkl")
    ncc = pd.read_pickle(ART_DIR / "ncc_feat.pkl")
    # 按井聚合 NCC（per-row 太噪 → 用井级 level 线索 + 重平滑曲线）
    g = ncc.groupby("well_id")
    agg = g["ncc_drift_blend"].median().rename("ncc_mean_drift").to_frame()
    agg["ncc_corr_mean"] = g["ncc_corr_w15"].mean()
    ncc = ncc.merge(agg, on="well_id", how="left")
    ncc_cols = ["ncc_drift_blend", "ncc_corr_w8", "ncc_corr_w15", "ncc_corr_w25",
                "ncc_mean_drift", "ncc_corr_mean"]
    df = df.merge(ncc[["well_id", "row_idx"] + ncc_cols], on=["well_id", "row_idx"], how="left")

    feats = V1_FEATS + ncc_cols
    X = df[feats].to_numpy(np.float32); y = df["target"].to_numpy(np.float32)
    folds = df["fold"].to_numpy()
    oof = np.zeros(len(df), np.float32)
    for f in sorted(np.unique(folds)):
        tr = folds != f
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=200,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
        m.fit(X[tr], y[tr]); oof[~tr] = m.predict(X[~tr])
        print(f"  fold {f}: trees={m.n_iter_}")
    df["oof_drift"] = oof

    def pooled(d): return float(np.sqrt(np.mean((d - y) ** 2)))
    sg = savgol_per_well(df, "oof_drift")
    print(f"\n刻度：CF {CF} / const-oracle {CONST_ORACLE} / 前沿 {FRONTIER} / 天花板 {CEILING}")
    print(f"  v2 (HGB +NCC agg)     pooled RMSE = {pooled(oof):.3f}")
    print(f"  + Savgol(17,3)        pooled RMSE = {pooled(sg):.3f}")
    df["pred_drift"] = sg if pooled(sg) < pooled(oof) else oof
    df[["well_id", "row_idx", "pred_drift", "target"]].to_pickle(ART_DIR / "stage_c_v2_oof.pkl")


if __name__ == "__main__":
    main()
