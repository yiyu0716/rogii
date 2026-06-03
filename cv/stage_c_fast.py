#!/usr/bin/env python3
"""Stage C v1 (fast) — 用已缓存特征跑快配 GBDT，确立基线 + 快迭代环。"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
from cv_runner import ART_DIR
from stage_c import FEATS, savgol_per_well

CF, CONST_ORACLE, CEILING, FRONTIER = 15.91, 9.04, 7.69, 9.85

df = pd.read_pickle(ART_DIR / "stage_c_rows.pkl")
print(f"rows={len(df):,} wells={df['well_id'].nunique()} feats={len(FEATS)}")
X = df[FEATS].to_numpy(np.float32); y = df["target"].to_numpy(np.float32)
folds = df["fold"].to_numpy()
oof = np.zeros(len(df), np.float32)
import time
for f in sorted(np.unique(folds)):
    t = time.time(); tr = folds != f
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, min_samples_leaf=200,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    m.fit(X[tr], y[tr]); oof[~tr] = m.predict(X[~tr])
    print(f"  fold {f}: trees={m.n_iter_} {time.time()-t:.0f}s", flush=True)
df["oof_drift"] = oof

def pooled(d): return float(np.sqrt(np.mean((d - y) ** 2)))
sg = savgol_per_well(df, "oof_drift")
print(f"\n刻度：CF {CF} / const-oracle {CONST_ORACLE} / 前沿 {FRONTIER} / 天花板 {CEILING}")
print(f"  Stage C v1 (HGB fast)   pooled RMSE = {pooled(oof):.3f}")
print(f"  + Savgol(17,3)          pooled RMSE = {pooled(sg):.3f}")
print(f"  对照 物理引擎单体        pooled RMSE = {pooled(np.clip(0.3*df['conf']*df['pdrift'],-20,20)):.3f}")
print(f"  对照 CF                  pooled RMSE = {pooled(np.zeros_like(y)):.3f}")
best = sg if pooled(sg) < pooled(oof) else oof
df["pred_drift"] = best
g = df.assign(se=(best - y) ** 2).groupby("well_id").agg(n=("target","size"), sse=("se","sum"))
g["rmse"] = np.sqrt(g["sse"]/g["n"])
print("\nworst-by-SSE top12:\n" + g.sort_values("sse", ascending=False).head(12).to_string())
df[["well_id","row_idx","pred_drift","target"]].to_pickle(ART_DIR/"stage_c_oof.pkl")
print("\nsaved oof.")
