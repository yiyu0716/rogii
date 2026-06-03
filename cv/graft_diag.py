#!/usr/bin/env python3
"""嫁接诊断 — base 的误差是 level 还是 shape 主导？物理 shape 在 base shape 之外有无边际价值？"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np, pandas as pd
from cv_runner import ART_DIR, DATA_ROOT
ART = DATA_ROOT.parent / "thbdh5765_rogii_v10_fresh_artifacts" / "diagnostics"

meta = pd.read_csv(ART / "oof_val_meta.csv")
z = np.load(ART / "oof_val_predictions.npz", allow_pickle=True)
preds = z["predictions"].astype(np.float64); tgt = z["target"].astype(np.float64)
well = meta["well"].to_numpy()
# 最优单列做 base（也可换 blend）
col = int(np.argmin([np.sqrt(np.mean((preds[:,j]-tgt)**2)) for j in range(preds.shape[1])]))
base = preds[:, col]

phys = pd.read_pickle(ART_DIR / "stage_c_rows.pkl")[["well_id","row_idx","pdrift","conf"]]
phys["id"] = phys["well_id"] + "_" + phys["row_idx"].astype(str)
order = phys.set_index("id").loc[meta["id"]]
pdrift = order["pdrift"].to_numpy()

def wmean(x): return pd.Series(x).groupby(well).transform("mean").to_numpy()
tm, bm, pm = wmean(tgt), wmean(base), wmean(pdrift)
swing = tgt - tm                      # 真 swing
base_sh = base - bm                   # base shape
phys_sh = pdrift - pm                 # 物理 shape
def rmse(d): return float(np.sqrt(np.mean((d-tgt)**2)))
def pc(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))

print(f"base=col{col}  RMSE={rmse(base):.3f}")
# 误差分解
lvl_res = np.sqrt(np.mean((bm-tm)**2))
sw_res  = np.sqrt(np.mean((base_sh-swing)**2))
print(f"  base level 残差(row-wt) = {lvl_res:.3f}   base swing 残差 = {sw_res:.3f}")
print(f"  → 总 {np.sqrt(lvl_res**2+sw_res**2):.3f} (≈RMSE 校验)")
print(f"  base level corr = {pc(bm,tm):+.3f}   base shape corr = {pc(base_sh,swing):+.3f}")
print(f"  phys level corr = {pc(pm,tm):+.3f}   phys shape corr = {pc(phys_sh,swing):+.3f}")
print(f"  base_shape vs phys_shape 冗余 corr = {pc(base_sh,phys_sh):+.3f}")

# 边际：真 swing ~ a·base_shape + b·phys_shape
from numpy.linalg import lstsq
A1 = np.column_stack([base_sh, np.ones(len(swing))])
c1,*_ = lstsq(A1, swing, rcond=None); r1 = pc(A1@c1, swing)
A2 = np.column_stack([base_sh, phys_sh, np.ones(len(swing))])
c2,*_ = lstsq(A2, swing, rcond=None); r2 = pc(A2@c2, swing)
print(f"\n  swing ~ base_shape          : R={r1:.4f}")
print(f"  swing ~ base_shape+phys_shape: R={r2:.4f}  (phys 系数={c2[1]:+.4f})")

# 反事实：oracle level + 各种 shape
print(f"\n[反事实 oracle-level + shape]")
print(f"  oracle level only           = {rmse(tm):.3f}")
print(f"  oracle + base_shape         = {rmse(tm+base_sh):.3f}")
print(f"  oracle + phys_shape(扫β)    = {min(rmse(tm+b*phys_sh) for b in np.arange(0,1.01,0.05)):.3f}")
best=1e9; bb=None
for a in np.arange(0,1.21,0.1):
  for b in np.arange(0,0.81,0.05):
    r=rmse(tm+a*base_sh+b*phys_sh)
    if r<best: best,bb=r,(round(a,1),round(b,2))
print(f"  oracle + a·base_sh + b·phys_sh = {best:.3f}  (a,b={bb})")
