#!/usr/bin/env python3
"""生成物理 LevelEngine 的 Kaggle 提交，并报告离线真分（3 口 test 井都在 train，有真 TVT）。

- 点云 + 引擎 fit 在全部 773 train 井（self-exclusion：test 井在 train 里也不会自泄）。
- 预测用 test/ 下的几何（与 train 同井一致），写 id={well}_{row_idx}, tvt=pred。
- 离线诊断：用 train/{well}__horizontal_well.csv 的真 TVT 算 pooled RMSE，
  对照 CF（carry-forward），证明物理 drift 引擎在真任务几何上确实优于 CF。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cv_runner import DATA_ROOT, ROOT, WellStore
from level_engine import build_cloud, LevelEngine

TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]
OUT = ROOT / "submissions" / "physics_level_v001.csv"


def first_missing(arr):
    m = ~np.isfinite(arr)
    return int(np.flatnonzero(m)[0])


def main():
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    eng = LevelEngine(XS, FS, OW, wid)         # 默认 beta0.3 clip20 guard1500/3000
    eng.fit(store.ids(), store)                # 全 train 点云

    rows = []
    pool_sse_eng = pool_sse_cf = pool_n = 0.0
    print(f"{'well':10s} {'n_hidden':>9s} {'RMSE_eng':>9s} {'RMSE_CF':>9s} {'Δ':>7s}")
    for w in TEST_WELLS:
        tdf = pd.read_csv(DATA_ROOT / "test" / f"{w}__horizontal_well.csv")
        ti = tdf["TVT_input"].to_numpy(float)
        ps = first_missing(ti)
        lk = float(ti[ps - 1])
        pred = eng.predict_arrays(tdf["MD"].to_numpy(float), tdf["X"].to_numpy(float),
                                  tdf["Y"].to_numpy(float), tdf["Z"].to_numpy(float),
                                  ps, lk, self_id=w)
        idx = np.arange(ps, len(tdf))
        for i, p in zip(idx, pred):
            rows.append((f"{w}_{i}", float(p)))
        # 离线真分（train 同井有 TVT）
        truth = pd.read_csv(DATA_ROOT / "train" / f"{w}__horizontal_well.csv",
                            usecols=["TVT"])["TVT"].to_numpy(float)[ps:]
        n = len(truth)
        sse_e = float(np.sum((pred - truth) ** 2))
        sse_c = float(np.sum((lk - truth) ** 2))
        pool_sse_eng += sse_e; pool_sse_cf += sse_c; pool_n += n
        re, rc = np.sqrt(sse_e / n), np.sqrt(sse_c / n)
        print(f"{w:10s} {n:9d} {re:9.3f} {rc:9.3f} {re - rc:+7.3f}")

    pe, pc = np.sqrt(pool_sse_eng / pool_n), np.sqrt(pool_sse_cf / pool_n)
    print(f"{'POOLED':10s} {int(pool_n):9d} {pe:9.3f} {pc:9.3f} {pe - pc:+7.3f}")

    sub = pd.DataFrame(rows, columns=["id", "tvt"])
    # 对齐 sample_submission 行序
    sample = pd.read_csv(DATA_ROOT / "quick" / "sample_submission.csv", usecols=["id"])
    sub = sample.merge(sub, on="id", how="left")
    assert sub["tvt"].notna().all(), "缺失提交行！"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    print(f"\n写出: {OUT}  ({len(sub)} 行)")


if __name__ == "__main__":
    main()
