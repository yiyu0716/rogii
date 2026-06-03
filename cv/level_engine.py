#!/usr/bin/env python3
"""ROGII level engine — 物理恒等式 drift 预测器（fold-safe），挂到 cv 基础设施。

核心物理（本环境实测 std 0.0065ft，6 top 平行）：
    TVT_i + Z_i = F_i + b_well        （F = formation top 绝对地层面，b_well 每井常数）
=>  drift_i := TVT_i − last_known = −(Z_i − Z_PS) + (F_i − F_PS)
其中 −ΔZ 在隐藏段精确已知；唯一未知是地层面变化 ΔF = F_i − F_PS。
F 空间光滑（Moran 0.88），用**训练折邻井** ANCC 点云 IDW 插补得到，PS 处自锚定（ΔF→0）。
关键：−ΔZ(~160ft) 与 ΔF 近反向抵消成 ~30ft drift，二者必须组合，单用任一都炸。

fold-safe：点云只用 train_ids 的井（验证折的井及其典井母本兄弟天然被 split 排除）。
另对 self_well 做显式排除（提交时 test 井在 train 里也安全）。

用法:
  python3 level_engine.py                  # 跑 well/typewell/spatial 三套 split 出真分
  python3 level_engine.py --beta 0.3 --clip 20
  python3 level_engine.py --sample 150     # 每折抽样验证井加速（粗测）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cv_runner import (
    ROOT, ART_DIR, DASHBOARD, SPLIT_SCHEMES, WellStore,
    get_or_build_splits, folds_to_pairs, score_oof, aggregate, md_table,
)

FORMATION = "ANCC"          # 主用地层面（6 top 平行，单 top 的 ΔF 即全体的 ΔF）
CLOUD_STRIDE = 80           # 点云沿井采样步长（ft）
CACHE = ART_DIR / "fcloud.npz"
REPORT = ROOT / "cv" / "level_engine_report.md"


# ---------------------------------------------------------------- F 点云
def build_cloud(store: WellStore, stride: int = CLOUD_STRIDE):
    """全库 (X, Y, F=ANCC) 点云，OW 记录每点归属井（整数索引）。缓存到 npz。"""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if int(z["stride"]) == stride:
            return z["XS"], z["FS"], z["OW"], list(z["wid"])
    ids = store.ids()
    XS, FS, OW = [], [], []
    for k, w in enumerate(ids):
        df = pd.read_csv(store.data_root / "train" / f"{w}__horizontal_well.csv",
                         usecols=["X", "Y", FORMATION])
        idx = np.arange(0, len(df), stride)
        xy = df[["X", "Y"]].to_numpy()[idx]
        F = df[FORMATION].to_numpy()[idx]
        ok = np.isfinite(F) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        XS.append(xy[ok]); FS.append(F[ok]); OW.append(np.full(int(ok.sum()), k))
    XS = np.vstack(XS).astype(np.float64)
    FS = np.concatenate(FS).astype(np.float64)
    OW = np.concatenate(OW).astype(np.int32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, XS=XS, FS=FS, OW=OW, wid=np.array(ids), stride=stride)
    return XS, FS, OW, ids


# ---------------------------------------------------------------- 预测器
class LevelEngine:
    """物理 drift 引擎。fit() 只用训练折的点云；predict() IDW 插补 ΔF。"""

    def __init__(self, XS, FS, OW, wid, *, beta=0.3, clip=20.0,
                 k=250, k_use=15, stride_anchor=40, min_frac=0.5,
                 conf_lo=1500.0, conf_hi=3000.0):
        self.XS, self.FS, self.OW = XS, FS, OW
        self.wid_to_idx = {w: i for i, w in enumerate(wid)}
        self.beta, self.clip = beta, clip
        self.k, self.k_use, self.stride_anchor, self.min_frac = k, k_use, stride_anchor, min_frac
        # 距离置信门：最近非 self 邻居 <conf_lo 全信，>conf_hi 退回 CF，中间余弦过渡。
        # 真实 test 井邻居密集(~400ft)→满信；spatial 留出区(>2500ft)→退 CF，不崩。
        self.conf_lo, self.conf_hi = conf_lo, conf_hi
        self.name = f"level_b{beta}_c{int(clip)}"

    def fit(self, train_ids, store):
        keep = np.fromiter((self.wid_to_idx[w] for w in train_ids if w in self.wid_to_idx),
                           dtype=np.int32)
        mask = np.isin(self.OW, keep)
        self._XS = self.XS[mask]
        self._FS = self.FS[mask]
        self._OW = self.OW[mask]
        self._tree = cKDTree(self._XS)

    def _idw(self, q, self_idx):
        """IDW 插补 F；排除 self_idx 井；每点取最近 k_use 个非 self 邻居。
        返回 (F 插值, 最近非 self 邻居距离)。"""
        d, j = self._tree.query(q, k=self.k)
        out = np.full(len(q), np.nan)
        nn = np.full(len(q), np.inf)
        for r in range(len(q)):
            jj, dd = j[r], d[r]
            m = self._OW[jj] != self_idx
            jj, dd = jj[m][:self.k_use], dd[m][:self.k_use]
            if len(jj) == 0:
                continue
            nn[r] = dd[0]
            wt = 1.0 / (dd + 1e-6)
            out[r] = float((wt * self._FS[jj]).sum() / wt.sum())
        return out, nn

    def _confidence(self, nn):
        """距离→置信 [0,1]：<conf_lo 为 1，>conf_hi 为 0，中间余弦过渡。"""
        d = np.clip(nn, self.conf_lo, self.conf_hi)
        return 0.5 * (1.0 + np.cos(np.pi * (d - self.conf_lo) / (self.conf_hi - self.conf_lo)))

    def predict(self, val_id, store):
        w = store.wells[val_id]
        return self.predict_arrays(w.md, w.x, w.y, w.z, w.ps_idx, w.last_known,
                                   self_id=val_id)

    def predict_arrays(self, md, x, y, z_all, ps_idx, last_known, self_id=None):
        """从原始几何数组预测隐藏段（供 test 提交复用；test 几何与 train 同井一致）。"""
        hs = ps_idx
        n = len(md) - hs
        lk = float(last_known)
        z = np.asarray(z_all[hs:], float); zps = float(z_all[hs - 1])
        cf = np.full(n, lk, dtype=float)
        self_idx = self.wid_to_idx.get(self_id, -1)

        anc = np.arange(0, n, self.stride_anchor)
        qxy = np.column_stack([np.asarray(x)[hs:][anc], np.asarray(y)[hs:][anc]]).astype(float)
        psxy = np.array([[float(x[hs - 1]), float(y[hs - 1])]], float)
        Fh, nn = self._idw(qxy, self_idx)
        Fps = self._idw(psxy, self_idx)[0][0]
        valid = np.isfinite(Fh)
        if not np.isfinite(Fps) or valid.mean() < self.min_frac:
            return cf                                  # 邻井不足 → 安全退回 CF
        # 用有效锚点对 F / 置信做沿 MD 的线性插补（填补 nan 锚点）
        md_h = np.asarray(md)[hs:]
        Fh_full = np.interp(md_h, md_h[anc][valid], Fh[valid])
        conf = np.interp(md_h, md_h[anc][valid], self._confidence(nn[valid]))
        pdrift = -(z - zps) + (Fh_full - Fps)          # 物理 drift 曲线
        d = np.clip(self.beta * conf * pdrift, -self.clip, self.clip)
        return lk + d

    def physics_curve(self, val_id, store):
        """Ring 0 诊断用：返回原始（未收缩/未 clip）物理 drift 分量。
        返回 dict 或 None（邻井不足应退 CF）：
          pdrift = -ΔZ + ΔF_pred ；dF_pred = ΔF 插值 ；negdZ = -(Z-Z_PS) 精确 ；conf 距离置信。
        """
        w = store.wells[val_id]
        hs = w.ps_idx; n = w.n_hidden
        z = w.z[hs:].astype(float); zps = float(w.z[hs - 1])
        self_idx = self.wid_to_idx.get(val_id, -1)
        anc = np.arange(0, n, self.stride_anchor)
        qxy = np.column_stack([w.x[hs:][anc], w.y[hs:][anc]]).astype(float)
        psxy = np.array([[w.x[hs - 1], w.y[hs - 1]]], float)
        Fh, nn = self._idw(qxy, self_idx)
        Fps = self._idw(psxy, self_idx)[0][0]
        valid = np.isfinite(Fh)
        if not np.isfinite(Fps) or valid.mean() < self.min_frac:
            return None
        md_h = w.md[hs:]
        Fh_full = np.interp(md_h, md_h[anc][valid], Fh[valid])
        conf = np.interp(md_h, md_h[anc][valid], self._confidence(nn[valid]))
        negdZ = -(z - zps)
        dF_pred = Fh_full - Fps
        return {"pdrift": negdZ + dF_pred, "negdZ": negdZ, "dF_pred": dF_pred, "conf": conf}


# ---------------------------------------------------------------- 扩展评分
def drift_shape_corr(oof, store):
    """drift_corr: 各井预测/真值的段内均值 drift 的相关；shape_corr: 段内去均值后的 pooled 相关。"""
    dp, dt = [], []
    s_xy = s_xx = s_yy = 0.0
    for wid, pred in oof.items():
        w = store.wells[wid]
        t = w.hidden_truth.astype(float)
        p = pred.astype(float)
        lk = w.last_known
        dp.append(p.mean() - lk); dt.append(t.mean() - lk)
        ps = p - p.mean(); ts = t - t.mean()
        s_xy += float((ps * ts).sum()); s_xx += float((ps * ps).sum()); s_yy += float((ts * ts).sum())
    drift_corr = float(np.corrcoef(dp, dt)[0, 1]) if len(dp) > 2 else float("nan")
    shape_corr = s_xy / np.sqrt(s_xx * s_yy + 1e-12)
    return drift_corr, shape_corr


def run(make_engine, pairs, store, sample=None, rng=None):
    oof = {}
    for train_ids, val_ids in pairs:
        eng = make_engine()
        eng.fit(train_ids, store)
        if sample is not None and len(val_ids) > sample:
            val_ids = list(rng.choice(val_ids, sample, replace=False))
        for vid in val_ids:
            oof[vid] = eng.predict(vid, store)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--k-use", type=int, default=15)
    ap.add_argument("--stride-anchor", type=int, default=40)
    ap.add_argument("--conf-lo", type=float, default=1500.0)
    ap.add_argument("--conf-hi", type=float, default=3000.0)
    ap.add_argument("--sample", type=int, default=None, help="每折抽样验证井加速（粗测）")
    args = ap.parse_args()

    print("[1/4] 加载井 + 点云 ...")
    store = WellStore()
    XS, FS, OW, wid = build_cloud(store)
    print(f"      点云点数={len(XS):,}  井数={len(wid)}")

    splits = get_or_build_splits(store, refresh=False)
    pairs = {s: folds_to_pairs(splits[s]) for s in SPLIT_SCHEMES}
    rng = np.random.default_rng(0)

    def make():
        return LevelEngine(XS, FS, OW, wid, beta=args.beta, clip=args.clip,
                           k_use=args.k_use, stride_anchor=args.stride_anchor,
                           conf_lo=args.conf_lo, conf_hi=args.conf_hi)

    print(f"[2/4] 跑 LevelEngine(beta={args.beta}, clip={args.clip}) × 三套 split ...")
    rows = []
    per_well_store = {}
    for scheme in SPLIT_SCHEMES:
        oof = run(make, pairs[scheme], store, sample=args.sample, rng=rng)
        scores = score_oof(oof, store)
        agg = aggregate(scores)
        dc, sc = drift_shape_corr(oof, store)
        pw = pd.DataFrame([{"well_id": s.well_id, "n_hidden": s.n_hidden,
                            "rmse": s.rmse, "sse": s.sse} for s in scores])
        per_well_store[scheme] = pw
        rows.append({"split": scheme, "rw_rmse": agg["row_weighted_rmse"],
                     "drift_corr": dc, "shape_corr": sc,
                     "p50": agg["per_well_rmse_p50"], "p90": agg["per_well_rmse_p90"],
                     "max": agg["per_well_rmse_max"], "n_wells": agg["n_wells"]})
        print(f"      {scheme:9s} rw_rmse={agg['row_weighted_rmse']:.4f}  "
              f"drift_corr={dc:+.3f}  shape_corr={sc:+.3f}  "
              f"p50={agg['per_well_rmse_p50']:.3f} p90={agg['per_well_rmse_p90']:.3f} max={agg['per_well_rmse_max']:.3f}")

    print("[3/4] worst-by-SSE (spatial) ...")
    sdf = per_well_store["spatial"].copy()
    sdf["sse_share"] = sdf["sse"] / sdf["sse"].sum()
    worst = sdf.sort_values("sse", ascending=False).head(20)

    print("[4/4] 写报告 ...")
    res = pd.DataFrame(rows)
    cf = 15.9099
    lines = ["# LevelEngine — 物理 drift 预测器 CV 报告\n",
             f"参数：beta={args.beta}, clip={args.clip}, k_use={args.k_use}, "
             f"stride_anchor={args.stride_anchor}, cloud_stride={CLOUD_STRIDE}, F={FORMATION}\n",
             f"参考刻度：CF {cf} / const-oracle 9.04 / smooth-201 0.39\n",
             "## 三套 split 真分\n", md_table(res), "",
             "> well=井级随机；typewell=典井母本绑同折；spatial=KMeans 地理区留出（压力下界）。\n",
             "## worst-well by SSE (spatial, top 20)\n",
             md_table(worst[["well_id", "n_hidden", "rmse", "sse", "sse_share"]]), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    sdf.sort_values("sse", ascending=False).to_csv(
        ART_DIR / f"level_engine__spatial__per_well.csv", index=False)
    print(f"      报告: {REPORT}")


if __name__ == "__main__":
    main()
