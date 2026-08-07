"""Find cost-effective T_F for a given PFE TVT threshold."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.cnn_sdf_config import Config
from scripts.analyze_pfe_tvt import (
    TRAIN,
    _true_ps_index,
    resample_typewell,
    summarize_threshold,
)


def load_wells():
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
    return wells


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", type=float, default=80.0)
    parser.add_argument(
        "--t-f-list",
        type=str,
        default="144,160,176,192,208,224,240,256",
        help="comma-separated T_F values to evaluate (default: 8 points)",
    )
    parser.add_argument("--no-md-bounds", action="store_true")
    args = parser.parse_args()

    wells = load_wells()
    cfg = Config()
    base_tf = cfg.T_F
    use_md = not args.no_md_bounds
    t_f_values = [int(x) for x in args.t_f_list.split(",") if x.strip()]

    base = summarize_threshold(args.thr, wells, cfg, use_md_bounds=use_md)
    md_tag = "有 MD 限制" if use_md else "无 MD 限制"
    print(f"训练井: {len(wells)}  thr={args.thr} ft  {md_tag}")
    print(
        f"基线 T_F={base_tf}: rand_oob={base['random_oob_pct']:.1f}% "
        f"worst_oob={base['worst_oob_pct']:.1f}% true_oob={base['true_oob_pct']:.1f}%"
    )
    print()
    print(
        f"{'T_F':>4} | {'+dT':>4} | {'T_SIZE':>6} | {'true_oob':>8} | "
        f"{'rand_oob':>8} | {'worst_oob':>9} | {'above>T_F':>8}"
    )
    rows = []
    for t_f in t_f_values:
        cfg.T_F = t_f
        cfg.T_SIZE = cfg.T_H + t_f
        s = summarize_threshold(args.thr, wells, cfg, use_md_bounds=use_md)
        rows.append((t_f, s))
        print(
            f"{t_f:4d} | {t_f - base_tf:+4d} | {cfg.T_SIZE:6d} | "
            f"{s['true_oob_pct']:7.1f}% | {s['random_oob_pct']:7.1f}% | "
            f"{s['worst_oob_pct']:8.1f}% | {s['random_above_pct']:7.1f}%"
        )

    print("\n=== 推荐拐点 (相对 T_F=144) ===")
    targets = [
        ("随机 OOB <= 3%", "random_oob_pct", 3.0),
        ("随机 OOB <= 2%", "random_oob_pct", 2.0),
        ("最坏 OOB <= 5%", "worst_oob_pct", 5.0),
        ("最坏 OOB <= 2%", "worst_oob_pct", 2.0),
        ("真 PS OOB <= 1%", "true_oob_pct", 1.0),
    ]
    for label, key, target in targets:
        ans = None
        for t_f in t_f_values:
            cfg.T_F = t_f
            cfg.T_SIZE = cfg.T_H + t_f
            s = summarize_threshold(args.thr, wells, cfg, use_md_bounds=use_md)
            if s[key] <= target:
                ans = t_f
                break
        if ans is None:
            print(f"  {label}: 在 {t_f_values} 内达不到")
        else:
            print(f"  {label}: T_F>={ans} (+{ans - base_tf})  T_SIZE={cfg.T_H + ans}")

    # marginal benefit: first T_F where worst_oob < 5% and random < 3%
    combo = None
    for t_f, s in rows:
        if s["worst_oob_pct"] <= 5.0 and s["random_oob_pct"] <= 3.0:
            combo = t_f
            break
    if combo:
        print(
            f"\n划算起点 (随机<=3% 且 最坏<=5%): T_F={combo} (+{combo - base_tf}), "
            f"T_SIZE={cfg.T_H + combo}"
        )


if __name__ == "__main__":
    main()
