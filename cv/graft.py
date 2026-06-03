#!/usr/bin/env python3
"""嫁接 — 物理 shape 叠到现有 thbdh5765 level OOF 上（CV 诚实，全 773 井 OOF）。

base = thbdh5765 OOF（drift 空间，8 模型，hill-climbing≈10.44）。
final_drift = base + β·conf·(physics_drift − per-well-mean(physics_drift)) ；可选 Savgol。
看能否把 ~10.4 往下压（物理 shape 是否在好 level 上加分）。
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from cv_runner import ART_DIR, DATA_ROOT

ART = DATA_ROOT.parent / "thbdh5765_rogii_v10_fresh_artifacts" / "diagnostics"


def main():
    meta = pd.read_csv(ART / "oof_val_meta.csv")
    z = np.load(ART / "oof_val_predictions.npz", allow_pickle=True)
    preds = z["predictions"].astype(np.float64)         # (N,8) drift
    tgt = z["target"].astype(np.float64)                # (N,) drift
    print(f"OOF rows={len(tgt):,}  pred cols={preds.shape[1]}")
    # 各列 OOF RMSE
    for j in range(preds.shape[1]):
        print(f"  col{j}: RMSE={np.sqrt(np.mean((preds[:,j]-tgt)**2)):.3f}")
    base_mean = preds.mean(1)
    print(f"  equal-mean blend RMSE = {np.sqrt(np.mean((base_mean-tgt)**2)):.3f}")
    # 取最优单列与均值里更好的做 base
    col_rmse = [np.sqrt(np.mean((preds[:,j]-tgt)**2)) for j in range(preds.shape[1])]
    best_col = int(np.argmin(col_rmse))
    base = base_mean if np.sqrt(np.mean((base_mean-tgt)**2)) < col_rmse[best_col] else preds[:,best_col]
    base_rmse = np.sqrt(np.mean((base-tgt)**2))
    print(f"  → base = {'mean-blend' if base is base_mean else f'col{best_col}'}  RMSE={base_rmse:.3f}")

    # 物理 drift（fold-safe）对齐
    phys = pd.read_pickle(ART_DIR / "stage_c_rows.pkl")[["well_id", "row_idx", "pdrift", "conf", "target"]]
    phys["id"] = phys["well_id"] + "_" + phys["row_idx"].astype(str)
    m = meta[["id", "well", "target"]].rename(columns={"target": "tgt_meta"}).merge(
        phys, on="id", how="left")
    assert m["pdrift"].notna().all(), f"未对齐 {m['pdrift'].isna().sum()} 行"
    # 校验 target 一致
    print(f"  target 校验：meta vs phys 最大差 = {np.abs(m['tgt_meta'].to_numpy()-m['target'].to_numpy()).max():.4f}")
    # 按 meta/npz 顺序排列物理量
    order = m.set_index("id").loc[meta["id"]]
    pdrift = order["pdrift"].to_numpy(); conf = order["conf"].to_numpy()
    well = meta["well"].to_numpy()

    # per-well 去均值物理 shape
    s = pd.Series(pdrift).groupby(well).transform("mean").to_numpy()
    shape = pdrift - s

    def rmse(d): return float(np.sqrt(np.mean((d - tgt) ** 2)))
    print(f"\n刻度：base {base_rmse:.3f} / const-oracle 9.04 / 天花板 7.69")
    best = (base_rmse, 0.0, "none")
    for bs in np.round(np.arange(0.0, 0.81, 0.05), 2):
        d = base + bs * conf * shape
        r = rmse(d)
        if r < best[0]: best = (r, bs, "raw")
    print(f"  base + β·conf·物理shape: 最优 β={best[1]:.2f}  RMSE={best[0]:.3f}  (Δ vs base {best[0]-base_rmse:+.3f})")

    # 不带 conf 的版本
    best2 = (base_rmse, 0.0)
    for bs in np.round(np.arange(0.0, 0.81, 0.05), 2):
        r = rmse(base + bs * shape)
        if r < best2[0]: best2 = (r, bs)
    print(f"  base + β·物理shape(无conf): 最优 β={best2[1]:.2f}  RMSE={best2[0]:.3f}")

    # 叠加后 Savgol（按井）
    bs = best[1]
    graft = base + bs * conf * shape
    sg = graft.copy()
    idx_by_well = pd.Series(np.arange(len(well))).groupby(well).apply(lambda x: x.to_numpy())
    for w, idx in idx_by_well.items():
        if len(idx) > 17:
            sg[idx] = savgol_filter(graft[idx], 17, 3)
    print(f"  + Savgol(17,3): RMSE={rmse(sg):.3f}")


if __name__ == "__main__":
    main()
