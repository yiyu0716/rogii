#!/usr/bin/env python3
"""ROGII 第 4 步（路线验证收尾）：带连续性约束的 typewell GR 序列对齐(Viterbi/DTW)。

这是第 3 步留下的唯一未证伪 gold 线的可行性实验。问题：观测隐藏段 GR（test 也给），
典井提供 GR=f(TVT)（同尺度、覆盖 100% 隐藏段）。求 TVT 路径 t(row) 使
  Σ |calib_tw_GR(t) − wellGR|  +  连续性(平滑)罚  +  起点锚到 last_known
最小（Viterbi 动态规划）。用本井 assigned 典井 → 训练无关，CV 分数与 split 无关（如 CF）。

判据（不是看绝对分，而是看它有没有真把 level 拉离 CF 朝真值走）：
  1. row-weighted RMSE 是否显著 < CF 15.91（朝 9~12）。
  2. corr(预测 level drift, 真 level drift) 是否显著为正（CF=0）。
  3. worst-SSE 长井有没有减、有没有把好井搞崩（principle #10）。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import cv_runner as cv
from cv_runner import WellStore

EPS = 1e-9
REPORT = cv.OUT_DIR / "step4_dtw_report.md"

# 默认超参（可经 --tune 网格搜索）
P = dict(step=20, win=80.0, res=1.0, gamma=0.04, alpha=0.5, smooth=201, gr_fallback=(0.8, 17.0))


def calibrate(w, tw_tvt, tw_gr):
    """prefix overlap 上 affine: wellGR ≈ a*tw_GR(TVT)+b。不足则用 eda 中位 (0.8,17)。"""
    hs = w.ps_idx
    pl = slice(max(0, hs - 600), hs)
    ti = w.tvt_input[pl]; pg = w.gr[pl].astype(float)
    m = np.isfinite(ti) & np.isfinite(pg)
    if m.sum() < 40:
        return P["gr_fallback"]
    tw_at = np.interp(ti[m], tw_tvt, tw_gr)
    if np.std(tw_at) < EPS:
        return P["gr_fallback"]
    a, b = np.polyfit(tw_at, pg[m], 1)
    if not (0.2 <= a <= 2.0):
        return P["gr_fallback"]
    return float(a), float(b)


def anchor_gr(w, step):
    """隐藏段按 step 取锚点，GR = 窗口均值（鲁棒）。返回 (anchor_md, g)。"""
    hs = w.ps_idx
    g = w.gr[hs:].astype(float); md = w.md[hs:]
    n = len(g)
    idx = np.arange(0, n, step)
    ga = np.array([np.nanmean(g[i:i + step]) if np.isfinite(g[i:i + step]).any() else np.nan
                   for i in idx])
    return idx, md[idx], ga


def viterbi_path(w, tw_tvt, tw_gr, prm):
    hs = w.ps_idx
    a, b = calibrate(w, tw_tvt, tw_gr)
    idx, amd, g = anchor_gr(w, prm["step"])
    A = len(idx)
    grid = w.last_known + np.arange(-prm["win"], prm["win"] + prm["res"], prm["res"])
    S = len(grid)
    cal = a * np.interp(grid, tw_tvt, tw_gr) + b          # 典井 GR(网格 TVT)，已标定
    # GR 尺度（鲁棒）
    gv = g[np.isfinite(g)]
    scale = np.median(np.abs(gv - np.median(gv))) * 1.4826 if len(gv) > 5 else 1.0
    scale = max(scale, 1.0)
    # 发射代价 E[A,S]
    E = np.zeros((A, S))
    for i in range(A):
        E[i] = np.abs(cal - g[i]) / scale if np.isfinite(g[i]) else 0.0
    # 转移罚 T[s,s'] = gamma*((grid_s-grid_s')/res)^2  （按 bin 距离）
    bins = np.arange(S)
    T = prm["gamma"] * (bins[:, None] - bins[None, :]) ** 2
    # Viterbi
    cost = E[0] + prm["alpha"] * ((grid - w.last_known) / prm["res"]) ** 2 * prm["gamma"]
    back = np.zeros((A, S), dtype=np.int32)
    for i in range(1, A):
        M = cost[:, None] + T
        back[i] = M.argmin(0)
        cost = E[i] + M.min(0)
    s = int(cost.argmin())
    path_states = np.empty(A, dtype=np.int32)
    for i in range(A - 1, -1, -1):
        path_states[i] = s; s = back[i][s]
    path_tvt = grid[path_states]
    # 插值回全隐藏段 + 平滑
    full_md = w.md[hs:]
    pred = np.interp(full_md, amd, path_tvt)
    pred = pd.Series(pred).rolling(prm["smooth"], center=True, min_periods=1).mean().to_numpy()
    return pred


def predict_well(store, wid, prm):
    w = store.wells[wid]
    tw = store.load_typewell(wid).dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw) < 20:
        return np.full(w.n_hidden, w.last_known)
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    try:
        return viterbi_path(w, tw_tvt, tw_gr, prm)
    except Exception:
        return np.full(w.n_hidden, w.last_known)


def score_all(store, prm, ids):
    rows = []
    for wid in ids:
        w = store.wells[wid]
        pred = predict_well(store, wid, prm)
        truth = w.hidden_truth
        err = pred - truth
        rows.append({
            "well_id": wid, "n_hidden": w.n_hidden,
            "rmse": float(np.sqrt(np.mean(err ** 2))), "sse": float(np.sum(err ** 2)),
            "pred_drift": float(pred.mean() - w.last_known),
            "true_drift": float(truth.mean() - w.last_known),
            "cf_sse": float(np.sum((w.last_known - truth) ** 2)),
        })
    return pd.DataFrame(rows)


def rw(df, col="rmse"):
    return float(np.sqrt((df[col] ** 2 * df["n_hidden"]).sum() / df["n_hidden"].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="只跑前 N 口井（调试）")
    args = ap.parse_args()

    print("[1/3] 加载井 ...")
    store = WellStore()
    ids = store.ids()
    if args.sample:
        ids = ids[:args.sample]

    if args.tune:
        print("[tune] 在 120 口样本上网格搜 gamma/step/win ...")
        rng = np.random.default_rng(0)
        samp = list(rng.choice(store.ids(), 120, replace=False))
        best = None
        for step in (20, 30):
            for gamma in (0.01, 0.02, 0.04, 0.08, 0.16):
                for win in (60.0, 100.0):
                    prm = {**P, "step": step, "gamma": gamma, "win": win}
                    df = score_all(store, prm, samp)
                    r = rw(df); dc = df[["pred_drift", "true_drift"]].corr().iloc[0, 1]
                    tag = f"step={step} gamma={gamma} win={win}"
                    print(f"      {tag:34s} rw={r:.3f} drift_corr={dc:+.3f}")
                    if best is None or r < best[0]:
                        best = (r, prm, dc, tag)
        print(f"[tune] best: {best[3]} rw={best[0]:.3f} drift_corr={best[2]:+.3f}")
        P.update(best[1])

    print(f"[2/3] 全量评分（{len(ids)} 口井）prm={ {k:P[k] for k in ['step','gamma','win','res','alpha','smooth']} } ...")
    df = score_all(store, P, ids)
    r = rw(df)
    drift_corr = df[["pred_drift", "true_drift"]].corr().iloc[0, 1]
    cf_rw = float(np.sqrt((df["cf_sse"]).sum() / df["n_hidden"].sum()))
    # worst by CF-SSE
    worst = df.sort_values("cf_sse", ascending=False).head(20).copy()
    worst["cf_rmse"] = np.sqrt(worst["cf_sse"] / worst["n_hidden"])
    worst["delta"] = worst["rmse"] - worst["cf_rmse"]
    win_wells = int((df["rmse"] < np.sqrt(df["cf_sse"] / df["n_hidden"]) - 0.1).sum())
    lose_wells = int((df["rmse"] > np.sqrt(df["cf_sse"] / df["n_hidden"]) + 0.1).sum())

    print("[3/3] 写报告 ...")
    print(f"      DTW row-weighted RMSE = {r:.4f}  (CF={cf_rw:.4f})  drift_corr={drift_corr:+.3f}")
    print(f"      改善井={win_wells}  变差井={lose_wells}  /  {len(df)}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    verdict = ("可行 ✓ → typewell 对齐是金牌 trunk，开建模" if (r < cf_rw - 0.5 and drift_corr > 0.2)
               else ("边际 ~ → 仅高置信井修正 + CF 地板" if r < cf_rw - 0.1 or drift_corr > 0.15
                     else "证伪 ✗ → 最后窄路也死，金牌靠 CF+平滑 鲁棒地板 + 工程稳定"))
    L = [f"# ROGII 第 4 步 · 约束 typewell GR 对齐(DTW) 可行性\n",
         f"生成时间：`{ts}`  ·  训练无关（用 assigned 典井）→ split 无关\n",
         f"超参：`{ {k:P[k] for k in ['step','gamma','win','res','alpha','smooth']} }`\n",
         "## 结果\n",
         pd.DataFrame([{
             "metric": "row-weighted RMSE", "DTW": round(r, 4), "CF": round(cf_rw, 4),
             "oracle_const": 9.0354, "oracle_smooth201": 0.3926}]).to_markdown(index=False) + "\n",
         f"- **level-drift 还原相关性** corr(pred_drift, true_drift) = **{drift_corr:+.3f}**（CF=0；>0.2 才算真抓到 level）。\n",
         f"- 改善井 **{win_wells}** / 变差井 **{lose_wells}** / 共 {len(df)}。\n",
         "## CF 最伤 20 口井（按 CF-SSE）上的表现\n",
         worst[["well_id", "n_hidden", "cf_rmse", "rmse", "delta"]].to_markdown(index=False, floatfmt=".3f") + "\n",
         f"## 判定\n\n**{verdict}**\n"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    df.sort_values("cf_sse", ascending=False).to_csv(cv.ART_DIR / "dtw_per_well.csv", index=False)
    print(f"      报告：{REPORT}")
    print(f"\n===== 判定：{verdict} =====")


if __name__ == "__main__":
    main()
