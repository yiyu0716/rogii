"""Evaluate PFE TVT margin for sample_split_ps (analysis only)."""
import csv
import random
import statistics
from pathlib import Path

TRAIN = Path("data/raw/rogii-wellbore-geology-prediction/train")
H_S, H_H, H_F = 24, 93, 427
T_H, T_F = 128, 144
PFE_MAX_SHIFT, PFE_MIN_HIST, PFE_MIN_FUT = 1500, 800, 1500
TW_STEP = 0.5


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def floats(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except Exception:
            pass
    return out


def true_ps(rows):
    idx = []
    for i, r in enumerate(rows):
        v = r.get("TVT_input", "")
        if v not in ("", None):
            try:
                float(v)
                idx.append(i)
            except Exception:
                pass
    return idx[-1] if idx else len(rows) // 2


def first_ps(rows):
    idx = []
    for i, r in enumerate(rows):
        v = r.get("TVT_input", "")
        if v not in ("", None):
            try:
                float(v)
                idx.append(i)
            except Exception:
                pass
    return idx[0] if idx else 0


def tvt_series(rows):
    if "TVT" in rows[0] and any(r.get("TVT", "") not in ("", None) for r in rows):
        return floats(rows, "TVT")
    return [
        float(r["TVT_input"]) if r.get("TVT_input", "") not in ("", None) else float("nan")
        for r in rows
    ]


def resample_typewell(t_rows):
    tvt = floats(t_rows, "TVT")
    diffs = [abs(tvt[i + 1] - tvt[i]) for i in range(len(tvt) - 1) if abs(tvt[i + 1] - tvt[i]) > 0]
    step = statistics.median(diffs) if diffs else TW_STEP
    ratio = step / TW_STEP
    if ratio > 1.05:
        up = int(round(ratio))
        new = []
        for i in range(len(tvt) - 1):
            seg = tvt[i : i + up]
            if seg:
                new.append(sum(seg) / len(seg))
        new.append(tvt[-1])
        return new
    if ratio < 0.95:
        g = int(round(1 / ratio))
        out = []
        for i in range(0, len(tvt), g):
            seg = tvt[i : i + g]
            out.append(sum(seg) / len(seg))
        return out
    return tvt


def bin_segment(arr, step):
    return [sum(arr[i : i + step]) / len(arr[i : i + step]) for i in range(0, len(arr), step)]


def crop(n, center, hist, fut):
    raw0 = center - hist
    raw1 = center + fut
    i0 = max(raw0, 0)
    i1 = min(raw1, n)
    return i0, i1, max(0, -raw0), max(0, raw1 - n)


def pseudo_min_tvt_bound(tvt, true_ps_i, first_ps_i, max_backshift_ft):
    thr = tvt[true_ps_i] - max_backshift_ft
    bound = true_ps_i - 1
    for i in range(first_ps_i, true_ps_i):
        if tvt[i] >= thr:
            bound = i
            break
    return bound


def analyze_pseudo(h_rows, t_tvt, split_ps):
    tvt = tvt_series(h_rows)
    h_tvt0 = bin_segment(tvt[: split_ps + 1], H_S)
    h_tvt1 = bin_segment(tvt[split_ps + 1 :], H_S)
    if not h_tvt0:
        return None
    ps_tvt = h_tvt0[-1]
    last_idx = min(range(len(t_tvt)), key=lambda i: abs(t_tvt[i] - ps_tvt))
    j0, j1, pl, pr = crop(len(t_tvt), last_idx + 1, T_H, T_F)
    t_seg = [t_tvt[j0]] * pl + t_tvt[j0:j1] + [t_tvt[j1 - 1]] * pr
    j0h, j1h, plh, prh = crop(len(h_tvt0), len(h_tvt0), H_H, 0)
    h0 = ([h_tvt0[0]] * plh + h_tvt0[j0h:j1h] + [h_tvt0[-1]] * prh)[:H_H]
    j0f, j1f, plf, prf = crop(len(h_tvt1), 0, 0, H_F)
    h1 = ([h_tvt1[0]] * plf + h_tvt1[j0f:j1f] + [h_tvt1[-1]] * prf)[:H_F]
    hseg = h0 + h1
    T = len(t_seg)
    fut_m = [min(range(T), key=lambda j: abs(t_seg[j] - hv)) for hv in hseg[H_H:]]
    full = [min(range(len(t_tvt)), key=lambda i: abs(t_tvt[i] - hv)) for hv in hseg[H_H:]]
    return {
        "above": max(full) - last_idx,
        "oob": sum(1 for m in fut_m if m in (0, T - 1)) > 0,
    }


def pct(vals, q):
    s = sorted(vals)
    return s[int(q * len(s))]


def main():
    records = []
    true_future_rows = []
    gap_at_pseudo_min = []

    for wf in sorted(TRAIN.glob("*__horizontal_well.csv")):
        h = read_csv(wf)
        n = len(h)
        tw = TRAIN / (wf.name.split("__")[0] + "__typewell.csv")
        if not tw.exists():
            continue
        t_tvt = resample_typewell(read_csv(tw))
        tvt = tvt_series(h)
        tp = true_ps(h)
        fp = first_ps(h)
        r_true = analyze_pseudo(h, t_tvt, tp)
        if r_true:
            true_future_rows.append(r_true["above"])
        pmin_md = max(fp, PFE_MIN_HIST - 1, tp - PFE_MAX_SHIFT)
        if pmin_md < tp:
            gap_at_pseudo_min.append(tvt[tp] - tvt[pmin_md])
        records.append(
            {"name": wf.name, "h": h, "t_tvt": t_tvt, "tvt": tvt, "n": n, "tp": tp, "fp": fp}
        )

    print("=== 真 PS 锚点：future 在锚点以上的 typewell 行数 (margin 应预留这部分) ===")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  p{int(q * 100):02d}: {pct(true_future_rows, q):.0f} rows ({pct(true_future_rows, q) * TW_STEP:.1f} ft)")
    print(f"  max: {max(true_future_rows):.0f} rows ({max(true_future_rows) * TW_STEP:.1f} ft)")

    print("\n=== 当前 MD-only pseudo_min 的 TVT gap (ft) ===")
    for q in [0.5, 0.9, 0.95]:
        print(f"  p{int(q * 100)}: {pct(gap_at_pseudo_min, q):.1f} ft")

    print(f"\n=== margin 扫描 (T_F={T_F}, PFE_MAX_SHIFT={PFE_MAX_SHIFT}) ===")
    print(f"{'margin':>6} | {'backshift':>9} | {'bind%':>5} | {'worst_oob%':>10} | {'rand_oob%':>9}")
    for margin in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]:
        if margin >= T_F:
            continue
        backshift_ft = (T_F - margin) * TW_STEP
        bind = viol_w = total_w = oob_r = cnt = 0
        for rec in records:
            h, t_tvt, tvt, n, tp, fp = (
                rec["h"],
                rec["t_tvt"],
                rec["tvt"],
                rec["n"],
                rec["tp"],
                rec["fp"],
            )
            pmin_md = max(fp, PFE_MIN_HIST - 1, tp - PFE_MAX_SHIFT)
            pmax = min(tp - 1, n - 1 - PFE_MIN_FUT)
            pmin_tvt = pseudo_min_tvt_bound(tvt, tp, fp, backshift_ft)
            pmin = max(pmin_md, pmin_tvt)
            if pmin > pmax:
                continue
            if pmin_tvt > pmin_md:
                bind += 1
            r_w = analyze_pseudo(h, t_tvt, pmin)
            if r_w:
                total_w += 1
                if r_w["oob"] or r_w["above"] > T_F:
                    viol_w += 1
            rp = random.Random(hash(rec["name"]) % 2**32).randint(pmin, pmax)
            r = analyze_pseudo(h, t_tvt, rp)
            if not r:
                continue
            cnt += 1
            oob_r += r["oob"]
        print(
            f"{margin:6d} | {backshift_ft:7.0f} ft | {100 * bind / len(records):4.1f}% | "
            f"{100 * viol_w / total_w:8.1f}% | {100 * oob_r / cnt:8.1f}%"
        )

    print("\n=== 达到目标 rand OOB 的最小 margin ===")
    for target in [0.5, 1, 2, 3, 5]:
        ans = None
        for margin in range(0, T_F):
            backshift_ft = (T_F - margin) * TW_STEP
            oob = cnt = 0
            for rec in records:
                h, t_tvt, tvt, n, tp, fp = (
                    rec["h"],
                    rec["t_tvt"],
                    rec["tvt"],
                    rec["n"],
                    rec["tp"],
                    rec["fp"],
                )
                pmin_md = max(fp, PFE_MIN_HIST - 1, tp - PFE_MAX_SHIFT)
                pmax = min(tp - 1, n - 1 - PFE_MIN_FUT)
                pmin = max(pmin_md, pseudo_min_tvt_bound(tvt, tp, fp, backshift_ft))
                if pmin > pmax:
                    continue
                rp = random.Random(hash(rec["name"]) % 2**32).randint(pmin, pmax)
                r = analyze_pseudo(h, t_tvt, rp)
                if not r:
                    continue
                cnt += 1
                oob += r["oob"]
            if 100 * oob / cnt <= target:
                ans = margin
                break
        if ans is not None:
            print(
                f"  OOB<={target}%: margin>={ans} "
                f"(backshift<={(T_F - ans) * TW_STEP:.0f}ft, reserve {ans}rows={ans * TW_STEP:.0f}ft)"
            )
        else:
            print(f"  OOB<={target}%: 在 margin 0..{T_F - 1} 内达不到")

    print("\n=== TVT约束 + PFE_MAX_SHIFT=500 时 margin 对比 ===")
    for margin in [48, 64, 80]:
        backshift_ft = (T_F - margin) * TW_STEP
        oob = bind = cnt = 0
        for rec in records:
            h, t_tvt, tvt, n, tp, fp = (
                rec["h"],
                rec["t_tvt"],
                rec["tvt"],
                rec["n"],
                rec["tp"],
                rec["fp"],
            )
            pmin_md = max(fp, PFE_MIN_HIST - 1, tp - 500)
            pmax = min(tp - 1, n - 1 - PFE_MIN_FUT)
            pmin_tvt = pseudo_min_tvt_bound(tvt, tp, fp, backshift_ft)
            pmin = max(pmin_md, pmin_tvt)
            if pmin > pmax:
                continue
            if pmin_tvt > pmin_md:
                bind += 1
            rp = random.Random(hash(rec["name"]) % 2**32).randint(pmin, pmax)
            r = analyze_pseudo(h, t_tvt, rp)
            if not r:
                continue
            cnt += 1
            oob += r["oob"]
        print(
            f"  margin={margin}: bind={100 * bind / len(records):.1f}% "
            f"rand_oob={100 * oob / cnt:.1f}% backshift={backshift_ft:.0f}ft"
        )


if __name__ == "__main__":
    main()
