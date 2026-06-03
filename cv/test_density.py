#!/usr/bin/env python3
"""快测：更密的 ANCC 点云能否打掉 LEVEL 墙（mitch 强调高空间密度）。"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "32")
import numpy as np, pandas as pd, time
from scipy.spatial import cKDTree
from cv_runner import WellStore, get_or_build_splits, folds_to_pairs
from level_engine import LevelEngine

FORMATION = "ANCC"
store = WellStore(); ids = store.ids()
fold_of = get_or_build_splits(store, refresh=False)["well"]
pairs = folds_to_pairs(fold_of)

def cloud(stride):
    XS, FS, OW = [], [], []
    for k, w in enumerate(ids):
        df = pd.read_csv(store.data_root/"train"/f"{w}__horizontal_well.csv", usecols=["X","Y",FORMATION])
        idx = np.arange(0, len(df), stride)
        xy = df[["X","Y"]].to_numpy()[idx]; F = df[FORMATION].to_numpy()[idx]
        ok = np.isfinite(F)&np.isfinite(xy[:,0])&np.isfinite(xy[:,1])
        XS.append(xy[ok]); FS.append(F[ok]); OW.append(np.full(int(ok.sum()),k))
    return np.vstack(XS).astype(float), np.concatenate(FS).astype(float), np.concatenate(OW).astype(np.int32)

def run(stride):
    t=time.time()
    XS,FS,OW = cloud(stride)
    wid = {w:i for i,w in enumerate(ids)}
    sse=tot=0.0; dp=[]; dt=[]; sxy=sxx=syy=0.0
    for tr,va in pairs:
        eng = LevelEngine(XS, FS, OW, ids, k_use=15)
        eng.fit(tr, store)
        for vid in va:
            w=store.wells[vid]; lk=w.last_known
            pred = eng.predict(vid, store)
            td = w.hidden_truth.astype(float)
            e = pred-td; sse+=np.sum(e**2); tot+=len(e)
            dp.append(pred.mean()-lk); dt.append(td.mean()-lk)
            p=pred-pred.mean(); s=td-td.mean(); sxy+=(p*s).sum(); sxx+=(p*p).sum(); syy+=(s*s).sum()
    rmse=np.sqrt(sse/tot); dc=np.corrcoef(dp,dt)[0,1]; sc=sxy/np.sqrt(sxx*syy+1e-9)
    print(f"stride={stride:3d} pts={len(XS):>7,} rmse={rmse:.3f} drift_corr={dc:+.3f} shape_corr={sc:+.3f} ({time.time()-t:.0f}s)", flush=True)

for s in [80, 20, 8]:
    run(s)
