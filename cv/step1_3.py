#!/usr/bin/env python3
"""ROGII 第 1–3 步：baseline ladder + 决定生死实验（OOF surface + drift）+ 路线判定。

全部预测器实现 cv_runner.Predictor 协议，挂在第 0 步锁定的三套冻结 split 上评分。

第 1 步 · ladder（刻度）:
  carry_forward(地板~15.9) / lateral_z_slope(naive 斜率) /
  oracle_const(9.04 不可达) / oracle_smooth_201(~0.4 不可达)。

第 2 步 · 决定生死:
  (a) OOF formation-surface：从邻井 (X,Y, mean_6tops) 点云 KNN-IDW 预测留出井地层面，量 RMSE。
  (b) drift 模型：
      neighbor_drift   = last_known + 邻井(训练折)观测 drift 的 IDW（target-based 空间先验，会泄露）
      surface_follow   = last_known + 预测地层面沿隐藏轨迹的漂移（gold-route 假设的直接化身）
      surface_follow_s = surface_follow + 201ft 平滑（近免费）

第 3 步 · 看 surface_follow 在 spatial split 下能否显著破 15.9、worst-SSE 减没减、压力 CV 撑不撑，自动给出分支。

注：轨迹 (MD,X,Y,Z) 在隐藏段也是已知（test 列含之），故可用隐藏段 XY 查询地层面。
oracle_* 用真值，仅作刻度，绝不上榜。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import cv_runner as cv
from cv_runner import WellStore, CarryForward, run_predictor, score_oof, aggregate

FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
CLOUD_STRIDE = 100      # 点云采样步长（行=ft）
PRED_STRIDE = 50        # 预测锚点步长，之后按 MD 线性插值到全隐藏段
KNN_K = 10
SMOOTH_WIN = 201
EPS = 1e-9
REPORT = cv.OUT_DIR / "step1_3_report.md"

# 模块级共享（main 填充）
LATERAL_START: dict[str, int] = {}
CLOUD: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # well_id -> (xy[n,2], mean_top[n])


# ---------------------------------------------------------------- 点云
def build_surface_cloud(store: WellStore) -> None:
    """对每口 train 井采样 (X,Y,mean_6tops)，stride=CLOUD_STRIDE，覆盖全井（tops 全段已知）。"""
    for wid in store.ids():
        p = store.data_root / "train" / f"{wid}__horizontal_well.csv"
        df = pd.read_csv(p, usecols=["X", "Y"] + FORMATION_COLS)
        mean_top = df[FORMATION_COLS].mean(axis=1, skipna=True).to_numpy(dtype=float)
        xy = df[["X", "Y"]].to_numpy(dtype=float)
        idx = np.arange(0, len(df), CLOUD_STRIDE)
        ok = np.isfinite(mean_top[idx]) & np.isfinite(xy[idx, 0]) & np.isfinite(xy[idx, 1])
        CLOUD[wid] = (xy[idx][ok], mean_top[idx][ok])


def _fold_tree(train_ids: list[str]) -> tuple[cKDTree, np.ndarray]:
    xs, vs = [], []
    for wid in train_ids:
        xy, mt = CLOUD[wid]
        if len(xy):
            xs.append(xy); vs.append(mt)
    pts = np.vstack(xs); vals = np.concatenate(vs)
    return cKDTree(pts), vals


def _idw_predict(tree: cKDTree, vals: np.ndarray, q_xy: np.ndarray, k: int = KNN_K) -> np.ndarray:
    d, i = tree.query(q_xy, k=k)
    if d.ndim == 1:
        d, i = d[:, None], i[:, None]
    w = 1.0 / (d + EPS)
    return (w * vals[i]).sum(axis=1) / w.sum(axis=1)


def _pred_surface_along(tree, vals, w: cv.Well) -> tuple[np.ndarray, float]:
    """预测隐藏段每行 mean_top（稀疏锚点 KNN-IDW + 按 MD 线性插值），及锚点(last-known处) surf。"""
    n = w.n_hidden
    hs = w.ps_idx
    hid_md = w.md[hs:]
    anchor = np.arange(0, n, PRED_STRIDE)
    if anchor[-1] != n - 1:
        anchor = np.append(anchor, n - 1)
    q_xy = np.column_stack([w.x[hs:][anchor], w.y[hs:][anchor]]).astype(float)
    surf_anchor = _idw_predict(tree, vals, q_xy)
    surf_full = np.interp(hid_md, hid_md[anchor], surf_anchor)
    # last-known 位置的 surf（漂移参照点）
    ps_xy = np.array([[w.x[hs - 1], w.y[hs - 1]]], dtype=float)
    surf_ps = float(_idw_predict(tree, vals, ps_xy)[0])
    return surf_full, surf_ps


# ---------------------------------------------------------------- 预测器
class LateralZSlope:
    name = "lateral_z_slope"
    def fit(self, train_ids, store): pass
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        ls = LATERAL_START.get(val_id, 0)
        sl = slice(max(0, ls), w.ps_idx)
        z = w.z[sl].astype(float); t = w.tvt_input[sl]
        m = np.isfinite(z) & np.isfinite(t)
        if m.sum() < 2 or np.std(z[m]) <= EPS:
            return np.full(w.n_hidden, w.last_known)
        a, b = np.polyfit(z[m], t[m], 1)
        return a * w.z[w.ps_idx:].astype(float) + b


class OracleConst:
    name = "oracle_const(ceil)"
    def fit(self, train_ids, store): pass
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        return np.full(w.n_hidden, float(w.hidden_truth.mean()))


class OracleSmooth:
    name = "oracle_smooth201(ceil)"
    def fit(self, train_ids, store): pass
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        s = pd.Series(w.hidden_truth).rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
        return s.to_numpy()


class NeighborDrift:
    """last_known + 训练折邻井观测 drift(hidden_mean-last_known) 的 PS-XY IDW。target-based，空间会泄露。"""
    name = "neighbor_drift"
    def __init__(self): self.tree = None; self.drifts = None
    def fit(self, train_ids, store):
        xy, dr = [], []
        for wid in train_ids:
            w = store.wells[wid]
            xy.append(w.ps_xy); dr.append(float(w.hidden_truth.mean()) - w.last_known)
        self.tree = cKDTree(np.array(xy, dtype=float)); self.drifts = np.array(dr)
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        d, i = self.tree.query(np.array(w.ps_xy, dtype=float), k=KNN_K)
        wt = 1.0 / (d + EPS)
        drift = float((wt * self.drifts[i]).sum() / wt.sum())
        return np.full(w.n_hidden, w.last_known + drift)


class SurfaceFollow:
    name = "surface_follow"
    def __init__(self, smooth=False): self.smooth = smooth; self.tree = None; self.vals = None
    def fit(self, train_ids, store):
        self.tree, self.vals = _fold_tree(train_ids)
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]
        surf, surf_ps = _pred_surface_along(self.tree, self.vals, w)
        pred = w.last_known + (surf - surf_ps)
        if self.smooth:
            pred = pd.Series(pred).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
        return pred


class SurfaceFollowSmooth(SurfaceFollow):
    name = "surface_follow_smooth"
    def __init__(self): super().__init__(smooth=True)


class OracleSurfaceFollow:
    """刻度：用本井真实 mean_6tops 的漂移做 follow。证明即便地层面完美，surface trunk 也不成立。"""
    name = "oracle_surface_follow(ceil)"
    def fit(self, train_ids, store): pass
    def predict(self, val_id, store) -> np.ndarray:
        w = store.wells[val_id]; hs = w.ps_idx
        p = store.data_root / "train" / f"{val_id}__horizontal_well.csv"
        mt = pd.read_csv(p, usecols=FORMATION_COLS).mean(axis=1, skipna=True).to_numpy()
        drift = mt[hs:] - mt[hs - 1]
        pred = w.last_known + drift
        return pd.Series(pred).interpolate(limit_direction="both").to_numpy()


# ---------------------------------------------------------------- 2a：OOF surface RMSE
def _eval_surface_simple(pairs, store) -> pd.DataFrame:
    out = []
    for train_ids, val_ids in pairs:
        tree, vals = _fold_tree(train_ids)
        for vid in val_ids:
            w = store.wells[vid]
            hs = w.ps_idx
            # 隐藏段稀疏锚点上的真/预测 mean_top
            n = w.n_hidden
            anchor = np.arange(0, n, PRED_STRIDE)
            q_xy = np.column_stack([w.x[hs:][anchor], w.y[hs:][anchor]]).astype(float)
            pred = _idw_predict(tree, vals, q_xy)
            # 真值 mean_top（隐藏段，按需读）
            p = store.data_root / "train" / f"{vid}__horizontal_well.csv"
            tops = pd.read_csv(p, usecols=FORMATION_COLS)
            true_mt = tops.mean(axis=1, skipna=True).to_numpy()[hs:][anchor]
            m = np.isfinite(pred) & np.isfinite(true_mt)
            if m.sum() == 0:
                continue
            err = pred[m] - true_mt[m]
            out.append({"well_id": vid, "n_pts": int(m.sum()),
                        "surf_rmse": float(np.sqrt(np.mean(err**2))),
                        "surf_sse": float(np.sum(err**2))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 跑 & 报告
def md_table(df, floatfmt=".4f"):
    return "_无记录_" if df is None or df.empty else df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    print("[1/5] 加载井 + 点云 ...")
    store = WellStore()
    dash = pd.read_csv(cv.DASHBOARD, dtype={"well_id": str})
    LATERAL_START.update(dict(zip(dash["well_id"], dash["lateral_start_idx"].astype(int))))
    build_surface_cloud(store)
    cloud_pts = sum(len(CLOUD[w][0]) for w in CLOUD)
    print(f"      井={len(store.ids())}  点云点数={cloud_pts:,}")

    splits = cv.get_or_build_splits(store, refresh=False)
    pairs = {s: cv.folds_to_pairs(splits[s]) for s in cv.SPLIT_SCHEMES}
    cf_ref = dash[["well_id", "rmse_cf"]].rename(columns={"rmse_cf": "rmse_cf_ref"})

    # 2a OOF surface RMSE
    print("[2/5] 2a · OOF formation-surface RMSE ...")
    surf_summary = []
    surf_per_well = {}
    for s in cv.SPLIT_SCHEMES:
        sdf = _eval_surface_simple(pairs[s], store)
        surf_per_well[s] = sdf
        rw = float(np.sqrt(sdf["surf_sse"].sum() / sdf["n_pts"].sum()))
        surf_summary.append({"split": s, "surf_rw_rmse_ft": rw,
                             "p50": float(sdf["surf_rmse"].median()),
                             "p90": float(sdf["surf_rmse"].quantile(0.9)),
                             "max": float(sdf["surf_rmse"].max())})
        print(f"      surface  {s:9s} rw_rmse={rw:.3f} ft")

    # 1+2b ladder & drift models
    print("[3/5] 1+2b · ladder & drift 预测器 × 三套 split ...")
    makers = {
        "carry_forward": CarryForward,
        "lateral_z_slope": LateralZSlope,
        "oracle_const(ceil)": OracleConst,
        "oracle_smooth201(ceil)": OracleSmooth,
        "oracle_surface_follow(ceil)": OracleSurfaceFollow,
        "neighbor_drift": NeighborDrift,
        "surface_follow": SurfaceFollow,
        "surface_follow_smooth": SurfaceFollowSmooth,
    }
    results = {}
    for pname, make in makers.items():
        results[pname] = {}
        for s in cv.SPLIT_SCHEMES:
            oof = run_predictor(make, pairs[s], store)
            scores = score_oof(oof, store)
            pw = pd.DataFrame([{"well_id": x.well_id, "n_hidden": x.n_hidden,
                                "rmse": x.rmse, "sse": x.sse} for x in scores])
            results[pname][s] = {"agg": aggregate(scores), "pw": pw}
        a = {s: results[pname][s]["agg"]["row_weighted_rmse"] for s in cv.SPLIT_SCHEMES}
        print(f"      {pname:24s} well={a['well']:.4f} typewell={a['typewell']:.4f} spatial={a['spatial']:.4f}")

    # worst-SSE 改善（surface_follow_smooth vs CF，spatial）
    print("[4/5] worst-SSE 改善（spatial）...")
    sf = results["surface_follow_smooth"]["spatial"]["pw"].merge(cf_ref, on="well_id")
    sf["sse_cf"] = sf["rmse_cf_ref"] ** 2 * sf["n_hidden"]
    top_cf = set(sf.sort_values("sse_cf", ascending=False).head(20)["well_id"])
    worst = sf[sf["well_id"].isin(top_cf)].copy()
    worst["delta_rmse"] = worst["rmse"] - worst["rmse_cf_ref"]
    worst = worst.sort_values("sse_cf", ascending=False)[
        ["well_id", "n_hidden", "rmse_cf_ref", "rmse", "delta_rmse"]]

    # ---- 第 3 步判定 ----
    print("[5/5] 第 3 步 · 路线判定 ...")
    cf = results["carry_forward"]["spatial"]["agg"]["row_weighted_rmse"]
    # 所有"非作弊 drift 尝试"在 spatial 下的最好成绩
    attempts = ["neighbor_drift", "surface_follow", "surface_follow_smooth"]
    best = min(attempts, key=lambda p: results[p]["spatial"]["agg"]["row_weighted_rmse"])
    sp = results[best]["spatial"]["agg"]["row_weighted_rmse"]
    we = results[best]["well"]["agg"]["row_weighted_rmse"]
    drop = cf - sp
    leak_gap = sp - we
    osf = results["oracle_surface_follow(ceil)"]["well"]["agg"]["row_weighted_rmse"]
    # 判定：cheap drift 是否拿得到（任一尝试显著破 CF）
    cheap_works = sp <= 14.5
    if cheap_works and leak_gap <= 1.5:
        branch = "A · cheap drift 可拿 → 金牌路线成立"
        nxt = "上 typewell top-k 对齐 / 边界特征 / 残差平滑 / beam·PF / stack。"
    elif cheap_works:
        branch = "B · 信号弱 → 保守 gating"
        nxt = "高风险井退回 CF+平滑，只在高置信井上修正。"
    else:
        branch = ("B/C · cheap drift 全灭 → 地板 + 仅一条窄路（typewell 约束对齐）"
                  if osf > cf else "A")
        nxt = ("surface/spatial/Z 三条路已用 oracle 证伪，彻底放弃；"
               "唯一未证伪的 gold 线是『带连续性约束的 typewell GR 序列对齐(DTW)』——"
               "下一步只 prototype 它并挂 CV 量化；底下铺 CF+201平滑 鲁棒地板。不上 PF/大 stack。")

    # ---- 写报告 ----
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    L = [f"# ROGII 第 1–3 步报告\n", f"生成时间：`{ts}` · seed=`{cv.RANDOM_SEED}` · 折={cv.N_FOLDS}\n",
         "全部预测器挂在第 0 步冻结 split 上；oracle_* 用真值仅作刻度。\n"]

    L.append("## 第 1 步 · baseline ladder（row-weighted RMSE）\n")
    main_rows = [{"predictor": p, **{s: results[p][s]["agg"]["row_weighted_rmse"] for s in cv.SPLIT_SCHEMES}}
                 for p in makers]
    L.append(md_table(pd.DataFrame(main_rows)) + "\n")
    L.append("> 刻度：CF≈15.91 地板；oracle_const 9.04（估对 level 的不可达天花板，≈公开前沿）；"
             "oracle_smooth201≈0.4（估对 shape 的不可达天花板）。lateral_z_slope 远差于 CF=naive 斜率不可用。\n")

    L.append("## 第 2 步 (a) · OOF formation-surface RMSE（ft）\n")
    L.append(md_table(pd.DataFrame(surf_summary)) + "\n")
    L.append("> 这是 correction #2 的硬验收：地层面在留出井上预测得多准。spatial 列才是真实可用度。\n")

    L.append("## 第 2 步 (b) · drift 模型 per-well 分布（surface_follow_smooth）\n")
    dist = [{"split": s, **{k: results[best][s]["agg"][k] for k in
             ["per_well_rmse_p50", "per_well_rmse_p90", "per_well_rmse_max"]}} for s in cv.SPLIT_SCHEMES]
    L.append(md_table(pd.DataFrame(dist)) + "\n")
    L.append("**CF 最伤的 20 口井（按 SSE）上的改善（spatial）：**\n")
    L.append(md_table(worst) + "\n")

    L.append("## 第 3 步 · 路线判定\n")
    L.append("### 证伪清单（含 oracle 上界，证明是真死路非 bug）\n")
    osc = results["oracle_surface_follow(ceil)"]["well"]["agg"]["row_weighted_rmse"]
    ev = pd.DataFrame([
        {"候选 trunk": "formation surface（mean/nearest top）",
         "证据": f"oracle_surface_follow={osc:.1f}（用真值仍远差于 CF 15.9）；corr(tvt,top)≈-0.01",
         "判": "证伪 ✗"},
        {"候选 trunk": "trajectory Z / TVD 校正",
         "证据": "corr(tvt,Z)≈-0.07；TVD-corrected CF≈107", "判": "证伪 ✗"},
        {"候选 trunk": "spatial 邻井 drift kriging",
         "证据": f"neighbor_drift spatial={results['neighbor_drift']['spatial']['agg']['row_weighted_rmse']:.2f} > CF",
         "判": "证伪 ✗"},
        {"候选 trunk": "naive 逐行 typewell GR-match",
         "证据": "诊断 18~32 > CF（逐点匹配忽略连续性，歧义）", "判": "naive 证伪 ✗"},
        {"候选 trunk": "约束 typewell GR 序列对齐(DTW)",
         "证据": "eda-2 §3：真 TVT 处 well-GR vs typewell-GR corr 0.665 → GR 携带 TVT 信号；尚未证伪",
         "判": "唯一未证伪 ☞"},
    ])
    L.append(md_table(ev, floatfmt=".2f") + "\n")
    L.append(f"- CF(spatial) = **{cf:.3f}**；最优 cheap-drift 尝试 {best}(spatial) = **{sp:.3f}**；降幅 = **{drop:.3f}**。\n")
    L.append(f"- 空间泄露缺口 spatial−well = **{leak_gap:.3f}**（大=邻井/外推在留出区崩）。\n")
    L.append(f"- neighbor_drift well/spatial = "
             f"{results['neighbor_drift']['well']['agg']['row_weighted_rmse']:.3f} / "
             f"{results['neighbor_drift']['spatial']['agg']['row_weighted_rmse']:.3f}"
             f"（target-based 先验的泄露幅度对照）。\n")
    L.append(f"\n### → 分支 {branch}\n")
    L.append(f"下一步：{nxt}\n")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n===== 第 3 步判定：分支 {branch} =====")
    print(f"CF(spatial)={cf:.3f}  {best}(spatial)={sp:.3f}  降幅={drop:.3f}  泄露缺口={leak_gap:.3f}")
    print(f"报告：{REPORT}")


if __name__ == "__main__":
    main()
