#!/usr/bin/env python3
"""Ring 1 — prefix-F 锚定，专攻 LEVEL（mean ΔF 偏置）。

当前引擎 ΔF = cloud_F(xy_i) − cloud_F(xy_PS)，用单点噪声 cloud_F_PS 当锚 → mean(ΔF) 有 per-well 偏置。
prefix 锚定：本井 prefix 已知真实相对地层面 F_rel = TVT_input + Z。
  offset = mean_prefix( cloud_F(xy) − F_rel )   （把云校准到本井 F_rel 尺度，prefix 多点降噪）
  drift_anc(i) = (cloud_F(xy_i) − offset) − Z_i − last_known
对比两者：mean-drift 的 level corr/RMSE，以及融合后 pooled RMSE。
"""
from __future__ import annotations

import numpy as np

from cv_runner import WellStore, get_or_build_splits, folds_to_pairs
from level_engine import build_cloud, LevelEngine

CONST_ORACLE, CF, CEILING = 9.04, 15.91, 7.69


def main():
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    pairs = folds_to_pairs(get_or_build_splits(store, refresh=False)["well"])

    cur_mean, anc_mean, tgt = [], [], []
    curves = {}     # vid -> (true_drift, cur_drift, anc_drift, conf)
    for tr, va in pairs:
        eng = LevelEngine(XS, FS, OW, wid)
        eng.fit(tr, store)
        for vid in va:
            w = store.wells[vid]; hs = w.ps_idx; n = w.n_hidden
            x = w.x.astype(float); y = w.y.astype(float); z = w.z.astype(float)
            md = w.md.astype(float); lk = w.last_known; zps = float(z[hs - 1])
            td = w.hidden_truth.astype(float) - lk
            self_idx = eng.wid_to_idx.get(vid, -1)
            # 隐藏段锚点
            anc = np.arange(0, n, eng.stride_anchor)
            qxy = np.column_stack([x[hs:][anc], y[hs:][anc]])
            Fh, nn = eng._idw(qxy, self_idx)
            valid = np.isfinite(Fh)
            if valid.mean() < 0.5:
                td0 = td; curves[vid] = (td, np.zeros(n), np.zeros(n), np.zeros(n))
                cur_mean.append(0.0); anc_mean.append(0.0); tgt.append(td.mean()); continue
            md_h = md[hs:]
            Fh_full = np.interp(md_h, md_h[anc][valid], Fh[valid])
            conf = np.interp(md_h, md_h[anc][valid], eng._confidence(nn[valid]))
            # --- 当前法：cloud_F_PS 锚 ---
            Fps = eng._idw(np.array([[x[hs - 1], y[hs - 1]]]), self_idx)[0][0]
            cur_drift = -(z[hs:] - zps) + (Fh_full - Fps)
            # --- prefix 锚定 ---
            L = min(hs, 800)
            pl = slice(hs - L, hs)
            Frel = w.tvt_input[pl] + z[pl]            # 已知真实相对面
            pxy = np.column_stack([x[pl], y[pl]])
            mfin = np.isfinite(Frel)
            offset = np.nan
            if mfin.sum() > 30:
                cloudFp, _ = eng._idw(pxy[mfin], self_idx)
                ok = np.isfinite(cloudFp)
                if ok.sum() > 30:
                    offset = float(np.mean(cloudFp[ok] - Frel[mfin][ok]))
            if not np.isfinite(offset):
                anc_drift = cur_drift
            else:
                anc_drift = (Fh_full - offset) - z[hs:] - lk
            curves[vid] = (td, cur_drift, anc_drift, conf)
            cur_mean.append(cur_drift.mean()); anc_mean.append(anc_drift.mean()); tgt.append(td.mean())

    cur_mean, anc_mean, tgt = map(np.array, (cur_mean, anc_mean, tgt))
    def lc(p):
        return np.corrcoef(p, tgt)[0, 1], float(np.sqrt(np.mean((p - tgt) ** 2)))
    print(f"刻度：CF {CF} / const-oracle {CONST_ORACLE} / 天花板 {CEILING}")
    print("\n[LEVEL = per-well mean drift] 预测质量（target std=%.2f）" % tgt.std())
    c0 = lc(cur_mean); c1 = lc(anc_mean)
    print(f"  当前 cloud_F_PS 锚 : corr={c0[0]:+.3f}  RMSE={c0[1]:.3f}")
    print(f"  prefix 锚定        : corr={c1[0]:+.3f}  RMSE={c1[1]:.3f}")

    # 融合 pooled RMSE：用各自的 mean 当 level + 各自 shape
    def pooled(which, bs, clip=20.0):
        sse = tot = 0.0
        for vid, (td, cur, an, conf) in curves.items():
            d = cur if which == "cur" else an
            level = d.mean()
            dev = d - d.mean()
            drift = level + np.clip(bs * conf * dev, -clip, clip)
            e = drift - td; sse += np.sum(e ** 2); tot += len(e)
        return np.sqrt(sse / tot)
    print("\n[融合 pooled RMSE] level=自身 mean + bs·conf·shape")
    for which in ["cur", "anc"]:
        best = min((pooled(which, bs), bs) for bs in np.round(np.arange(0, 0.81, 0.05), 2))
        print(f"  {which:3s}: best bs={best[1]:.2f}  RMSE={best[0]:.3f}")
    # anchored level + 物理 shape（取当前 cur 的 shape，shape 已好）
    print("\n[最优组合] prefix-anchored level + cur 物理 shape")
    sse = tot = 0.0
    for vid, (td, cur, an, conf) in curves.items():
        dev = cur - cur.mean()
        drift = an.mean() + np.clip(0.5 * conf * dev, -20, 20)
        e = drift - td; sse += np.sum(e ** 2); tot += len(e)
    print(f"  anc-level + 0.5·cur-shape  RMSE={np.sqrt(sse/tot):.3f}")


if __name__ == "__main__":
    main()
