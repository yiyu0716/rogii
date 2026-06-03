#!/usr/bin/env python3
"""专攻 LEVEL — base level(0.737) 之外有无正交 level 信号能把它推高？

per-well: target=真均值 drift。候选 level 源：base(thbdh5765 OOF 均值)、physics(pdrift_mean)、
mean(-ΔZ)、extrap、ncc、prefix_slope、几何。看组合能否把 level corr 从 0.737 推上去，
并把"新 level + base shape"的 pooled RMSE 压到 base(≈11.1) 以下。GroupKFold=well。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np, pandas as pd
from numpy.linalg import lstsq
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_runner import ART_DIR, DATA_ROOT, get_or_build_splits, WellStore
ART = DATA_ROOT.parent / "thbdh5765_rogii_v10_fresh_artifacts" / "diagnostics"

meta = pd.read_csv(ART/"oof_val_meta.csv"); z=np.load(ART/"oof_val_predictions.npz",allow_pickle=True)
preds=z["predictions"].astype(np.float64); tgt=z["target"].astype(np.float64); well=meta["well"].to_numpy()
col=int(np.argmin([np.sqrt(np.mean((preds[:,j]-tgt)**2)) for j in range(preds.shape[1])]))
from scipy.optimize import nnls
wts,_=nnls(preds,tgt); base=preds@wts            # blend ≈ 部署
# per-well 聚合
g=pd.DataFrame({"well":well,"base":base,"tgt":tgt}).groupby("well")
pw=g.mean(); pw["base_shape_keep"]=1
tm=pw["tgt"].to_numpy(); bm=pw["base"].to_numpy()

# 物理 + 其它 per-well 特征
lg=pd.read_csv(ART_DIR/"level_gbm_oof.csv").set_index("well_id")
lg=lg.loc[pw.index]
feats_extra=["pdrift_mean","pdrift_end","dF_mean","dz_mean","dz_total","extrap" if "extrap" in lg else "prefix_slope",
             "tvt_rate" if "tvt_rate" in lg else "slope100","grh_mean","grh_std","md_len","dxy_total","x0","y0","z0"]
feats_extra=[f for f in dict.fromkeys(feats_extra) if f in lg.columns]
pw=pw.join(lg[feats_extra])
pw["base_level"]=bm

def lc(p): return np.corrcoef(p,tm)[0,1]
print(f"wells={len(pw)}  base=NNLS-blend  base level corr={lc(bm):+.4f}  (resid {np.sqrt(np.mean((bm-tm)**2)):.3f})")
print(f"physics level corr={lc(pw['pdrift_mean']):+.4f}")

# 线性组合是否超 base？（注意：同 OOF 拟合略乐观，先看上界）
def fit_corr(cols):
    X=np.column_stack([pw[c] for c in cols]+[np.ones(len(pw))])
    c,*_=lstsq(X,tm,rcond=None); return lc(X@c)
print(f"\n线性组合 level corr（同OOF 上界）:")
for cols in [["base_level"],["base_level","pdrift_mean"],["base_level","pdrift_mean","dF_mean","dz_mean"],
             ["base_level"]+feats_extra]:
    print(f"  {cols if len(cols)<5 else 'base+all-extra'}: {fit_corr(cols):+.4f}")

# 诚实：GroupKFold 学新 level（base + 物理 + 几何），重建 = new_level + base_shape
folds=get_or_build_splits(WellStore(),refresh=False)["well"]
pw["fold"]=[folds[w] for w in pw.index]
Xcols=["base_level"]+feats_extra
X=pw[Xcols].to_numpy(float); yv=tm; oof=np.zeros(len(pw))
for f in sorted(set(pw["fold"])):
    tr=pw["fold"].to_numpy()!=f
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=15,
        l2_regularization=2.0,min_samples_leaf=20,random_state=0)
    m.fit(X[tr],yv[tr]); oof[~tr]=m.predict(X[~tr])
print(f"\nGBM 学新 level (GroupKFold): corr={lc(oof):+.4f}  resid={np.sqrt(np.mean((oof-tm)**2)):.3f}")

# pooled RMSE: new_level + base_shape  vs  base
bm_map=dict(zip(pw.index,bm)); newl_map=dict(zip(pw.index,oof))
base_shape=base - np.array([bm_map[w] for w in well])
def rmse(d): return float(np.sqrt(np.mean((d-tgt)**2)))
print(f"\n刻度 base(blend) pooled RMSE = {rmse(base):.3f}")
print(f"  new_level + base_shape     = {rmse(np.array([newl_map[w] for w in well])+base_shape):.3f}")
print(f"  oracle_level + base_shape  = {rmse(np.array([dict(zip(pw.index,tm))[w] for w in well])+base_shape):.3f}")
