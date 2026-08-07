"""Statistics for TVT-threshold PFE (sample_split_ps)."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.cnn_sdf_config import Config
from src.dataset import (
    _first_known_ps_index,
    _horizontal_tvt_values,
    _true_ps_index,
)

TRAIN = ROOT / "data/raw/rogii-wellbore-geology-prediction/train"
TW_STEP = 0.5


def resample_typewell(t_rows: pd.DataFrame) -> np.ndarray:
    tvt = t_rows["TVT"].astype(float).values
    diffs = [abs(tvt[i + 1] - tvt[i]) for i in range(len(tvt) - 1) if abs(tvt[i + 1] - tvt[i]) > 0]
    step = statistics.median(diffs) if diffs else TW_STEP
    ratio = step / TW_STEP
    if ratio > 1.05:
        up = int(round(ratio))
        new = []
        for i in range(len(tvt) - 1):
            seg = tvt[i : i + up]
            if len(seg):
                new.append(float(np.mean(seg)))
        new.append(float(tvt[-1]))
        return np.asarray(new, dtype=np.float64)
    if ratio < 0.95:
        g = int(round(1 / ratio))
        out = []
        for i in range(0, len(tvt), g):
            seg = tvt[i : i + g]
            out.append(float(np.mean(seg)))
        return np.asarray(out, dtype=np.float64)
    return tvt.astype(np.float64)


def bin_segment(arr: np.ndarray, step: int) -> list[float]:
    return [float(np.mean(arr[i : i + step])) for i in range(0, len(arr), step)]


def crop(n: int, center: int, hist: int, fut: int):
    raw0 = center - hist
    raw1 = center + fut
    i0 = max(raw0, 0)
    i1 = min(raw1, n)
    return i0, i1, max(0, -raw0), max(0, raw1 - n)


def pfe_candidates(
    h: pd.DataFrame,
    true_ps: int,
    cfg: Config,
    *,
    use_md_bounds: bool = True,
) -> np.ndarray:
    n = len(h)
    first_ps = _first_known_ps_index(h)
    if use_md_bounds:
        pseudo_min = max(first_ps, cfg.PFE_MIN_HISTORY_ORIG - 1)
        pseudo_max = min(true_ps - 1, n - 1 - cfg.PFE_MIN_FUTURE_ORIG)
    else:
        pseudo_min = first_ps
        pseudo_max = true_ps - 1
    if pseudo_min > pseudo_max:
        return np.array([], dtype=np.int64)
    tvt = _horizontal_tvt_values(h)
    history_tvt_min = float(tvt[true_ps]) - float(cfg.PFE_TVT_SHIFT_THRESHOLD)
    segment = tvt[pseudo_min : pseudo_max + 1]
    return np.flatnonzero(segment > history_tvt_min) + pseudo_min


def analyze_pseudo_oob(
    h: pd.DataFrame,
    t_tvt: np.ndarray,
    split_ps: int,
    *,
    h_s: int,
    h_h: int,
    h_f: int,
    t_h: int,
    t_f: int,
) -> dict | None:
    tvt = _horizontal_tvt_values(h)
    h_tvt0 = bin_segment(tvt[: split_ps + 1], h_s)
    h_tvt1 = bin_segment(tvt[split_ps + 1 :], h_s)
    if not h_tvt0:
        return None
    ps_tvt = h_tvt0[-1]
    last_idx = int(np.abs(t_tvt - ps_tvt).argmin())
    j0, j1, pl, pr = crop(len(t_tvt), last_idx + 1, t_h, t_f)
    t_seg = [t_tvt[j0]] * pl + list(t_tvt[j0:j1]) + [t_tvt[j1 - 1]] * pr
    t_size = t_h + t_f

    j0h, j1h, plh, prh = crop(len(h_tvt0), len(h_tvt0), h_h, 0)
    h0 = ([h_tvt0[0]] * plh + h_tvt0[j0h:j1h] + [h_tvt0[-1]] * prh)[:h_h]
    j0f, j1f, plf, prf = crop(len(h_tvt1), 0, 0, h_f)
    h1 = ([h_tvt1[0]] * plf + h_tvt1[j0f:j1f] + [h_tvt1[-1]] * prf)[:h_f]
    hseg = h0 + h1

    fut_m = [int(np.abs(np.asarray(t_seg) - hv).argmin()) for hv in hseg[h_h:]]
    full = [int(np.abs(t_tvt - hv).argmin()) for hv in hseg[h_h:]]
    above = max(full) - last_idx if full else 0
    return {
        "above": above,
        "oob": any(m in (0, t_size - 1) for m in fut_m),
        "matched_max": max(fut_m) if fut_m else 0,
    }


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(int(q * len(s)), len(s) - 1)]


def summarize_threshold(
    threshold: float,
    wells: list[dict],
    cfg: Config,
    *,
    use_md_bounds: bool = True,
) -> dict:
    cfg.PFE_TVT_SHIFT_THRESHOLD = threshold
    eligible_counts: list[int] = []
    md_span: list[int] = []
    tvt_gaps: list[float] = []
    md_bounds_fail = tvt_empty = 0
    true_oob = random_oob = worst_oob = 0
    true_above = random_above = 0
    n_true = n_rand = n_worst = 0

    for w in wells:
        h, t_tvt, tp = w["h"], w["t_tvt"], w["tp"]
        cand = pfe_candidates(h, tp, cfg, use_md_bounds=use_md_bounds)
        first_ps = _first_known_ps_index(h)
        if use_md_bounds:
            pseudo_min = max(first_ps, cfg.PFE_MIN_HISTORY_ORIG - 1)
            pseudo_max = min(tp - 1, len(h) - 1 - cfg.PFE_MIN_FUTURE_ORIG)
        else:
            pseudo_min = first_ps
            pseudo_max = tp - 1
        if pseudo_min > pseudo_max:
            md_bounds_fail += 1
            continue
        if cand.size == 0:
            tvt_empty += 1
            continue

        eligible_counts.append(int(cand.size))
        md_span.append(int(tp - cand.min()))
        tvt = _horizontal_tvt_values(h)
        gaps = tvt[tp] - tvt[cand]
        tvt_gaps.extend(gaps.tolist())

        kw = dict(
            h_s=cfg.H_S,
            h_h=cfg.H_H,
            h_f=cfg.H_F,
            t_h=cfg.T_H,
            t_f=cfg.T_F,
        )
        r_true = analyze_pseudo_oob(h, t_tvt, tp, **kw)
        if r_true:
            n_true += 1
            true_oob += int(r_true["oob"])
            true_above += int(r_true["above"] > cfg.T_F)

        rp = int(random.Random(hash(w["name"]) % 2**32).choice(cand))
        r_rand = analyze_pseudo_oob(h, t_tvt, rp, **kw)
        if r_rand:
            n_rand += 1
            random_oob += int(r_rand["oob"])
            random_above += int(r_rand["above"] > cfg.T_F)

        worst = int(cand[np.argmin(tvt[cand])])
        r_worst = analyze_pseudo_oob(h, t_tvt, worst, **kw)
        if r_worst:
            n_worst += 1
            worst_oob += int(r_worst["oob"])

    return {
        "threshold": threshold,
        "wells": len(wells),
        "md_bounds_fail": md_bounds_fail,
        "tvt_empty": tvt_empty,
        "eligible_p50": statistics.median(eligible_counts) if eligible_counts else 0,
        "eligible_p90": pct(eligible_counts, 0.9) if eligible_counts else 0,
        "eligible_max": max(eligible_counts) if eligible_counts else 0,
        "md_shift_p50": statistics.median(md_span) if md_span else 0,
        "md_shift_p90": pct(md_span, 0.9) if md_span else 0,
        "tvt_gap_p50": statistics.median(tvt_gaps) if tvt_gaps else 0,
        "tvt_gap_p90": pct(tvt_gaps, 0.9) if tvt_gaps else 0,
        "tvt_gap_max": max(tvt_gaps) if tvt_gaps else 0,
        "true_oob_pct": 100 * true_oob / n_true if n_true else 0,
        "random_oob_pct": 100 * random_oob / n_rand if n_rand else 0,
        "worst_oob_pct": 100 * worst_oob / n_worst if n_worst else 0,
        "random_above_pct": 100 * random_above / n_rand if n_rand else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-md-bounds",
        action="store_true",
        help="ignore PFE_MIN_HISTORY_ORIG / PFE_MIN_FUTURE_ORIG",
    )
    parser.add_argument("--t-f", type=int, default=None, help="override T_F (T_H unchanged)")
    parser.add_argument("--thr", type=float, default=None, help="single threshold to report")
    parser.add_argument(
        "--thr-list",
        type=str,
        default="24,32,40,48,56,64,80",
        help="comma-separated thresholds for sweep",
    )
    args = parser.parse_args()
    use_md_bounds = not args.no_md_bounds

    cfg = Config()
    if args.t_f is not None:
        cfg.T_F = args.t_f
        cfg.T_SIZE = cfg.T_H + cfg.T_F
    wells = []
    for wf in sorted(TRAIN.glob("*__horizontal_well.csv")):
        tw = TRAIN / f"{wf.name.split('__')[0]}__typewell.csv"
        if not tw.exists():
            continue
        h = pd.read_csv(wf)
        wells.append(
            {
                "name": wf.name,
                "h": h,
                "t_tvt": resample_typewell(pd.read_csv(tw)),
                "tp": _true_ps_index(h),
            }
        )

    print(f"训练井: {len(wells)}")
    print(f"T_F={cfg.T_F}  PFE_TVT_SHIFT_THRESHOLD={cfg.PFE_TVT_SHIFT_THRESHOLD} ft")
    if use_md_bounds:
        print(f"PFE_MIN_HISTORY={cfg.PFE_MIN_HISTORY_ORIG}  PFE_MIN_FUTURE={cfg.PFE_MIN_FUTURE_ORIG}")
    else:
        print("PFE_MIN_HISTORY / PFE_MIN_FUTURE: 未启用（仅 first_ps .. true_ps-1 + TVT 阈值）")
    print()

    if use_md_bounds:
        cur_thr = float(args.thr if args.thr is not None else cfg.PFE_TVT_SHIFT_THRESHOLD)
        cur = summarize_threshold(cur_thr, wells, cfg, use_md_bounds=True)
        print("=== 当前配置 ===")
        print(f"thr={cur_thr} ft  T_F={cfg.T_F}")
        print(f"MD 窗口不满足: {cur['md_bounds_fail']}  TVT 过滤后无候选: {cur['tvt_empty']}")
        print(
            f"每井候选数: p50={cur['eligible_p50']:.0f}  p90={cur['eligible_p90']:.0f}  max={cur['eligible_max']:.0f}"
        )
        print(
            f"候选 MD 前移: p50={cur['md_shift_p50']:.0f}  p90={cur['md_shift_p90']:.0f} (orig samples)"
        )
        print(
            f"候选 TVT gap: p50={cur['tvt_gap_p50']:.1f}  p90={cur['tvt_gap_p90']:.1f}  max={cur['tvt_gap_max']:.1f} ft"
        )
        print(
            f"真 PS OOB: {cur['true_oob_pct']:.1f}%  |  "
            f"随机伪 PS OOB: {cur['random_oob_pct']:.1f}%  |  "
            f"最坏(最低TVT候选) OOB: {cur['worst_oob_pct']:.1f}%"
        )
        print(f"随机伪 PS matched_above > T_F: {cur['random_above_pct']:.1f}%")
        print()

    thr_values = [float(x) for x in args.thr_list.split(",") if x.strip()]
    if args.thr is not None and len(thr_values) == 1:
        thr_values = [float(args.thr)]

    print("=== 阈值灵敏度 ===")
    print(f"{'thr(ft)':>7} | {'no_cand':>7} | {'cand_p50':>8} | {'gap_p90':>7} | {'rand_oob%':>9} | {'worst_oob%':>10}")
    for thr in thr_values:
        s = summarize_threshold(float(thr), wells, cfg, use_md_bounds=use_md_bounds)
        print(
            f"{thr:7.0f} | {s['tvt_empty']:7d} | {s['eligible_p50']:8.0f} | "
            f"{s['tvt_gap_p90']:7.1f} | {s['random_oob_pct']:8.1f}% | {s['worst_oob_pct']:9.1f}%"
        )


if __name__ == "__main__":
    main()
