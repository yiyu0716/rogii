# %% [code]
# ROGII train19-XGB experiment — self-contained script
# Based on train19 feature pipeline, with XGBoost trainer instead of LightGBM
# pipeline: v5 features (PF/beam/NCC/spatial) + surface (exp010) + geometry meta
#           (exp012) + dip-field (exp014) -> LightGBM 5fold -> pp -> submission.csv
import os
import time
from pathlib import Path

# Local train-only configuration: train19 = train7 + mode confidence / GR matching margin features
# Purpose: return to the train7 base; only add candidate-trajectory confidence features; do not change LGB parameters or enable geoaux.
# Characteristics: PMF=1 + MODESK=1,NMODES=3 + MODECONF=1; RATEFEAT/GEOSPECTRUM/SELFCORR/REPSECTION are off.
os.environ.setdefault("MODE", "train")
os.environ.setdefault("DATA_DIR", "/home/wdy/cyp/ROG/rogii-dataset")
os.environ.setdefault("ART_DIR", "/home/wdy/cyp/ROG/exp252_xgb_t19_modeconf_grmargin")
os.environ.setdefault("TRAIN_XGB_DEVICE", "cuda")  # cuda/gpu or cpu
os.environ.setdefault("XGB_GPU_FALLBACK", "1")
os.environ.setdefault("N_JOBS", "4")
os.environ.setdefault("LOG_WELL_EVERY", "25")
os.environ.setdefault("N_EST", "8000")
os.environ.setdefault("MODEL_SEEDS", "1")

# blend2_dir: mainly changes feature construction and aligns with blend2_likpf=8 in infer.
os.environ.setdefault("LIKPF_MSEED", "8")

# XGBoost parameters: mapped from the train19 LightGBM baseline as conservatively as possible.
# LGB num_leaves=127 roughly corresponds to max_depth=7~8; start with depth=8 and tune from there.
os.environ.setdefault("XGB_MAX_DEPTH", "8")
os.environ.setdefault("XGB_MIN_CHILD_WEIGHT", "80")
os.environ.setdefault("SUBSAMPLE", "0.8")
os.environ.setdefault("COLSAMPLE", "0.6")
os.environ.setdefault("REG_LAMBDA", "10")
os.environ.setdefault("REG_ALPHA", "1")
os.environ.setdefault("LOSS", "regression")  # regression or huber/pseudohuber
os.environ.setdefault("XGB_MAX_BIN", "256")
os.environ.setdefault("XGB_GROW_POLICY", "depthwise")
os.environ.setdefault("XGB_MAX_LEAVES", "0")  # set >0 only when grow_policy=lossguide

# Feature experiment switches: training and inference are automatically aligned through the manifest.
# train2 + PMF=1 + RATEFEAT/GEOSPECTRUM/SELFCORR/REPSECTION + MODESK=1,NMODES=3
os.environ.setdefault("MULTISCALE", "0")
os.environ.setdefault("POSTERIOR", "0")
os.environ.setdefault("DIPFUSE", "0")
os.environ.setdefault("TRAJ", "0")
os.environ.setdefault("LINMDZ", "0")
os.environ.setdefault("GRFEATS2", "0")
os.environ.setdefault("PMF", "1")
os.environ.setdefault("REPSECTION", "0")
os.environ.setdefault("SELFCORR", "0")
os.environ.setdefault("GEOSPECTRUM", "0")
os.environ.setdefault("RATEFEAT", "0")
os.environ.setdefault("DETERMINISTIC", "0")
os.environ.setdefault("EVALCAL", "0")
os.environ.setdefault("GLOBALDECODE", "0")
os.environ.setdefault("GEOPEARSON", "0")
os.environ.setdefault("NMODES", "3")
os.environ.setdefault("MODESK", "1")
os.environ.setdefault("MODECONF", "1")  # train19: mode gap/std/range, mode-vs-pmf/beam, GR match margin
# os.environ.setdefault("SMOKE_WELLS", "20")

import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree

def _find_data():
    cands = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    for c in cands:
        if c.exists():
            return True, c
    env = os.environ.get("DATA_DIR")
    if env and Path(env).exists():
        return False, Path(env)
    here = Path(__file__).resolve()
    for up in [here.parents[4] / "data", here.parents[2] / "data"]:  # Support arbitrary placement paths.
        if (up / "train").exists():
            return False, up
    return False, here.parents[2] / "data"


KAGGLE, DATA = _find_data()
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
SEED = 42
SMOKE_WELLS = int(os.environ.get("SMOKE_WELLS", "0"))
N_EST = int(os.environ.get("N_EST", "200" if SMOKE_WELLS else "8000"))


ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.0; PF_GR_SIG_MAX = 60.0; PF_GR_SIG_DEF = 30.0
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001
# pf_ancc/pf_z multi-seed: default M=1=[42] exactly matches current. PF_MSEED=8 averages [42..49] to remove seed lottery.
_PF_SEEDS = [42 + _i for _i in range(int(os.environ.get("PF_MSEED", "1")))]


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP, seed):
    np.random.seed(seed)   # Deterministic per well, independent of pass order. Single=42 / multi-seed=42..49 average.
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv; va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP, seed):
    np.random.seed(seed)   # Deterministic per well. Single=42 / multi-seed=42..49 average.
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0; pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm; va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]; pz = z_v[i]
    return pts, std_


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    md = ev["MD"].values.astype(np.float64); zz = ev["Z"].values.astype(np.float64)
    grv = ev["GR"].values.astype(np.float64)
    pacc = None; sacc = None
    for s in _PF_SEEDS:   # Single=[42] / multi-seed=[42..49] average to remove seed lottery.
        pts, std = _pf_ancc(md, zz, grv, gg, gmin, gst,
                            gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP, s)
        pacc = pts.copy() if pacc is None else pacc + pts
        sacc = std.copy() if sacc is None else sacc + std
    n = len(_PF_SEEDS)
    return (pacc / n).astype(np.float32), (sacc / n).astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values)
    dmd2 = np.diff(t2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    md = ev["MD"].values.astype(np.float64); zz = ev["Z"].values.astype(np.float64)
    grv = ev["GR"].values.astype(np.float64); grs = gr_sm.loc[ev.index].values.astype(np.float64)
    ltvt = float(kna["TVT_input"].iloc[-1])
    pacc = None; sacc = None
    for s in _PF_SEEDS:   # Single=[42] / multi-seed=[42..49] average.
        pts, std = _pf_z(md, zz, grv, grs, gg, gs2, gmin, gst, gs, ltvt, iv,
                         beta, icpt, zsig, N,
                         PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP, s)
        pacc = pts.copy() if pacc is None else pacc + pts
        sacc = std.copy() if sacc is None else sacc + std
    n = len(_PF_SEEDS)
    return (pacc / n).astype(np.float32), (sacc / n).astype(np.float32)


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]; b = hP[s, b]
    return path


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float64)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    nh = len(hgr)
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_late, b_wls


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]
        wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=2000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


FI = None
DI = None


def _pmf_fwd_delta(hw, tw, last_tvt, ev):
    """point-mass (grid Bayes) forward filter. Returns dip delta aligned to ev.
    Matches gen_pmf_feature.py(point_mass_filter.pmf_well), so cache_pmf and train/infer use identical logic.
    """
    from scipy.ndimage import gaussian_filter1d
    A_ = 0.998; RN_ = 0.002; PN_ = 0.005; IS_ = 0.3; GSTEP = 0.5
    RG = np.arange(-0.05, 0.05 + 1e-9, 0.0025)
    kn = hw[hw["TVT_input"].notna()]
    if len(ev) < 5 or len(kn) < 10:
        return None
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].astype(float).interpolate(limit_direction="both").to_numpy(float)
    if len(tw_tvt) < 5 or not np.isfinite(tw_gr).all():
        return None
    k = kn[kn["GR"].notna()]
    gs = (float(np.clip(np.std(k["GR"].values - np.interp(k["TVT_input"].values, tw_tvt, tw_gr)), 10.0, 60.0))
          if len(k) >= 20 else 30.0)
    last_z = float(kn["Z"].iloc[-1]); last_md = float(kn["MD"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    tg = last_tvt + np.arange(-50, 50 + 1e-9, GSTEP); G = len(tg)
    gg = np.interp(tg, tw_tvt, tw_gr); rate_vals = ir + RG
    grv = ev["GR"].astype(float).interpolate(limit_direction="both").to_numpy(float)
    mdv = ev["MD"].to_numpy(float); zv = ev["Z"].to_numpy(float); n = len(ev)
    P0 = (np.exp(-0.5 * ((tg - last_tvt) / max(IS_, 0.2)) ** 2)[None, :]
          * np.exp(-0.5 * ((RG) / 0.01) ** 2)[:, None]); P0 /= P0.sum()
    cols = np.arange(G)
    P = P0.copy(); est = np.full(n, np.nan); pm, pz = last_md, last_z
    sig_pos = PN_ / GSTEP; sig_rate = RN_ / 0.0025
    for j in range(n):
        dm_i = mdv[j] - pm; dz_i = zv[j] - pz
        if dm_i == 0:
            dm_i = 1.0
        P = gaussian_filter1d(P, max(sig_rate, 0.3), axis=0, mode="nearest")
        sh = (rate_vals * dm_i - dz_i) / GSTEP
        k0 = np.floor(sh).astype(int); f = sh - k0
        i0 = np.clip(cols[None, :] - k0[:, None], 0, G - 1); i1 = np.clip(cols[None, :] - k0[:, None] - 1, 0, G - 1)
        P = (1 - f)[:, None] * np.take_along_axis(P, i0, 1) + f[:, None] * np.take_along_axis(P, i1, 1)
        P = gaussian_filter1d(P, max(sig_pos, 0.3), axis=1, mode="nearest")
        if np.isfinite(grv[j]):
            d = (grv[j] - gg) / gs; L = np.exp(-0.5 * np.clip(d * d, 0, 600)); P *= L[None, :]
        s = P.sum(); P = P / s if s > 1e-300 else P0.copy()
        est[j] = float(np.dot(P.sum(0), tg)); pm, pz = mdv[j], zv[j]
    return (est - last_tvt).astype(np.float32)


def build_well(hw_path: Path, is_train: bool):
    wid = hw_path.stem.replace("__horizontal_well", "")
    tw_path = hw_path.parent / f"{wid}__typewell.csv"
    if not tw_path.exists():
        return None
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and ("TVT" not in hw.columns or hw["TVT"].isna().all()):
        return None

    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_raw = tw["GR"].astype(float).interpolate(limit_direction="both")
    tw_gr = tw_gr_raw.fillna(tw_gr_raw.mean()).to_numpy(float)
    if len(tw_tvt) < 3 or not np.isfinite(tw_gr).all():
        return None

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    kmd = kn["MD"].to_numpy(float)
    ktvt = kn["TVT_input"].to_numpy(float)
    kz = kn["Z"].to_numpy(float)

    slp_all = robust_slope(kmd, ktvt)
    slp_200 = robust_slope(kmd[-200:], ktvt[-200:])
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both")
    gr_full = gr_full.fillna(float(np.nanmean(tw_gr)))
    gr_ev = gr_full.iloc[ev.index].to_numpy(np.float32)
    gr_m21 = gr_full.rolling(21, center=True, min_periods=1).mean().iloc[ev.index].to_numpy(np.float32)
    _GRFEATS2 = os.environ.get("GRFEATS2", "0") == "1"
    if _GRFEATS2:
        gr_m101 = gr_full.rolling(101, center=True, min_periods=1).mean().iloc[ev.index].to_numpy(np.float32)
        gr_m21_slope = np.gradient(gr_m21.astype(float)).astype(np.float32)

    md_ev = ev["MD"].to_numpy(float)
    z_ev = ev["Z"].to_numpy(float).astype(np.float32)
    md_since = (md_ev - float(lk["MD"])).astype(np.float32)
    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    kgr = gr_full.iloc[: len(kn)].to_numpy(np.float32)
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)

    # --- Particle filter ---
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) != nh:
        return None
    pf_zv, std_zv = run_pf_z(hw, tw_tvt, tw_gr)
    has_z = len(pf_zv) == nh and not np.any(np.isnan(pf_zv))

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0
    beam_stack = np.stack([p - last_tvt for p in bpaths.values()], 1)

    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt.astype(np.float32), hgr)
    (sc8, sc8s), (sc15, sc15s), (sc25, sc25s) = sc_res
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = FI.impute(xy_kn, self_wid=swid)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_late, b_wls = seg_b_well(ktvt, kz, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}_d"] = tvt_f - np.float32(last_tvt)
        tvt_fs[f"tvtFw_{fn}_d"] = tvt_fw - np.float32(last_tvt)
        tvt_fs[f"tvtF50_{fn}_d"] = tvt_f50 - np.float32(last_tvt)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-kz + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = DI.impute(xy_kn, self_wid=swid)
    b_d, b_dl, b_dw = seg_b_well(ktvt, kz, d_kn)
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + kz - d_kn - b_d
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in ev.index],
        "last_tvt": sc(last_tvt),
        "md_since": md_since,
        "dz": (z_ev - np.float32(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "frac": frac,
        "slp_all": sc(slp_all),
        "slp_200": sc(slp_200),
        "slp_50": sc(slp_50),
        "slp_z": sc(slp_z),
        "ext_all": (slp_all * md_since).astype(np.float32),
        "ext_200": (slp_200 * md_since).astype(np.float32),
        "ext_50": (slp_50 * md_since).astype(np.float32),
        "gr": gr_ev,
        "gr_m21": gr_m21,
        **({} if not _GRFEATS2 else {
            "gr_m101": gr_m101,
            "gr_m21_slope": gr_m21_slope,
        }),
        "gr_vs_tw_last": gr_ev - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_na_frac": sc(hw["GR"].isna().mean()),
        "known_len": sc(len(kn)),
        "eval_len": sc(nh),
        "ktvt_range": sc(float(np.ptp(ktvt))),
        "tw_range": sc(float(np.ptp(tw_tvt))),
        # Particle filter.
        "pf_ancc_d": (pf_a - np.float32(last_tvt)).astype(np.float32),
        "pf_ancc_std": std_a,
        "pf_z_d": ((pf_zv - np.float32(last_tvt)).astype(np.float32) if has_z else sc(0.0)),
        "pf_z_std": (std_zv if has_z else sc(0.0)),
        "pf_vs_z": ((pf_a - pf_zv).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_beam": (pf_a - bpaths["cons"]).astype(np.float32),
        **{f"tdpf{int(o)}": gr_ev - np.interp(pf_a + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": beam_stack.mean(1).astype(np.float32),
        "beam_std_d": beam_stack.std(1).astype(np.float32),
        "beam_med_d": np.median(beam_stack, 1).astype(np.float32),
        "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        **{f"tdbc{int(o)}": gr_ev - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        "sc8_d": sc8 - np.float32(last_tvt), "sc8_sc": sc8s,
        "sc15_d": sc15 - np.float32(last_tvt), "sc15_sc": sc15s,
        "sc25_d": sc25 - np.float32(last_tvt), "sc25_sc": sc25s,
        "sc_cons_d": sc_cons - np.float32(last_tvt),
        "sc_ens_d": sc_ens - np.float32(last_tvt),
        "sc_trust": sc(sc_trust),
        "sc_std": np.stack([sc8, sc15, sc25], 1).std(1).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        **{f"tda{int(o)}": gr_ev - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdsc{int(o)}": gr_ev - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        "tw_gr_mean": sc(float(np.nanmean(tw_gr))),
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d,
        "form_std_d": form_std_d,
        "form_rng_d": form_rng_d,
        "spatial_knn_dist": knn_d,
        "dense_std": d_std,
        "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse),
        "dense_bias": sc(d_bias),
        "beam_vs_spatial": (bpaths["cons"] - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "sc_vs_spatial": (sc_ens - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC_d"] - (tvt_dense - last_tvt)).astype(np.float32),
        "pf_vs_spatial": (pf_a - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "pf_vs_dense": (pf_a - np.float32(last_tvt) - (tvt_dense - last_tvt)).astype(np.float32),
    }
    df = pd.DataFrame(feats)
    # exp122: hengck23 linear(md,z) global prior (disc/699326, GM). Fit TVT~a*MD+b*Z+c in the known zone,
    #  ridge-fit and extrapolate into eval. This is a richer 2D prior than existing ext_all (MD-only 1D linear). Inverse-problem insight:
    #  correct GR-only drift with a global prior. LINMDZ=0(default) appends nothing -> bitwise match. Column order rule: append new features at the end.
    if os.environ.get("LINMDZ", "0") == "1":
        mm = kmd.mean(); zm = kz.mean(); ms = kmd.std() + 1e-9; zs = kz.std() + 1e-9
        Ak = np.column_stack([(kmd - mm) / ms, (kz - zm) / zs, np.ones(len(kmd))])
        coef = np.linalg.solve(Ak.T @ Ak + 1e-3 * np.eye(3), Ak.T @ ktvt)  # ridge (countermeasure for MD/Z collinearity).
        pred_k = Ak @ coef
        Ae = np.column_stack([(md_ev - mm) / ms, (z_ev.astype(float) - zm) / zs, np.ones(len(md_ev))])
        pred_e = Ae @ coef
        lin_rmse = float(np.sqrt(np.mean((ktvt - pred_k) ** 2)))
        df["lin_mdz_d"] = (pred_e - last_tvt).astype(np.float32)
        df["lin_mdz_rmse"] = np.float32(lin_rmse)
        df["lin_mdz_vs_ext"] = ((pred_e - last_tvt) - (slp_all * md_since)).astype(np.float32)
    # exp159: append the point-mass (grid Bayes) forward filter as a decorrelated feature. PMF=0 appends nothing -> bitwise match.
    if os.environ.get("PMF", "0") == "1":
        pr = _pmf_fwd_delta(hw, tw, last_tvt, ev)
        df["pmf_fwd_d"] = pr if pr is not None else np.float32(0.0)
    if is_train:
        df["target"] = (ev["TVT"].to_numpy(float) - last_tvt).astype(np.float32)
    return df



def build_dataset(d, is_train):
    parts = []
    paths = sorted(d.glob("*__horizontal_well.csv"))
    t0 = time.time()
    tag = "train" if is_train else "test"
    print(f"  [{tag}] build_dataset start: {len(paths)} wells", flush=True)
    for i, p in enumerate(paths):
        r = build_well(p, is_train)
        if r is not None:
            parts.append(r)
        if (i + 1) % int(os.environ.get("LOG_WELL_EVERY", "25")) == 0 or (i + 1) == len(paths):
            print(f"  [{tag}] build_dataset {i + 1}/{len(paths)} wells, kept={len(parts)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    print(f"  [{tag}] build_dataset done: shape={out.shape} ({time.time() - t0:.0f}s)", flush=True)
    return out


SURF_SPW = 80   # Surface sample count per well.
SURF_K = 24     # Number of neighbor points.


class SurfaceImputer:
    """KNN interpolation over (X,Y) for the point cloud of datum-surface depth s=TVT+Z from all training wells."""

    def __init__(self, well_ids, data_dir, spw=SURF_SPW):
        xs, ys, ss, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            s = (df["TVT"] + df["Z"]).to_numpy()
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            xs.append(df["X"].to_numpy()[ix]); ys.append(df["Y"].to_numpy()[ix])
            ss.append(s[ix]); wids.extend([wid] * len(ix))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.s = np.concatenate(ss).astype(np.float64)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=SURF_K, nfetch=3000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.s))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        sn = self.s[ik]
        sp = (sn * w).sum(1) / safe
        sp = np.where(sw < 1e-9, float(self.s.mean()), sp)
        var = ((sn - sp[:, None]) ** 2 * w).sum(1) / safe
        return (sp.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


def fit_plane(x, y, s):
    """Least squares for s ~ a*x + b*y + c. Returns (a,b,c, resid_rmse)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    return coef[0], coef[1], coef[2], float(np.sqrt(np.mean(resid ** 2)))


def build_surface_feats(hw_path: Path, SI: SurfaceImputer, is_train: bool):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    Xk = kn["X"].to_numpy(float); Yk = kn["Y"].to_numpy(float); Zk = kn["Z"].to_numpy(float)
    sk = kn["TVT_input"].to_numpy(float) + Zk          # datum-surface depth (known zone).
    Xe = ev["X"].to_numpy(float); Ye = ev["Y"].to_numpy(float); Ze = ev["Z"].to_numpy(float)
    last_tvt = float(kn["TVT_input"].iloc[-1])

    # (A) self-well plane fit.
    a, b, c, self_rmse = fit_plane(Xk, Yk, sk)
    surf_self = a * Xe + b * Ye + c
    tvt_self = surf_self - Ze
    s_const = float(np.median(sk))
    tvt_const = s_const - Ze
    # Local plane using the last 50 points (prioritize dip near PS).
    tail = slice(max(0, len(kn) - 50), len(kn))
    if len(kn) >= 20:
        a2, b2, c2, _ = fit_plane(Xk[tail], Yk[tail], sk[tail])
        tvt_self50 = (a2 * Xe + b2 * Ye + c2) - Ze
    else:
        tvt_self50 = tvt_self

    # (B) neighboring-well surface.
    swid = wid if is_train else None
    surf_nbr, nbr_std, nbr_dist = SI.impute(np.column_stack([Xe, Ye]), self_wid=swid)
    tvt_nbr = surf_nbr.astype(float) - Ze
    # (C) bias calibration on the known zone of the same well.
    surf_nbr_k, _, _ = SI.impute(np.column_stack([Xk, Yk]), self_wid=swid)
    bias = float(np.median(sk - surf_nbr_k.astype(float)))
    tvt_nbrcal = (surf_nbr.astype(float) + bias) - Ze
    nbr_cal_rmse = float(np.sqrt(np.mean((sk - (surf_nbr_k.astype(float) + bias)) ** 2)))

    def d(v):
        return (v - last_tvt).astype(np.float32)

    feats = {
        "id": [f"{wid}_{i}" for i in ev.index],
        "srf_self_d": d(tvt_self),
        "srf_self50_d": d(tvt_self50),
        "srf_const_d": d(tvt_const),
        "srf_nbr_d": d(tvt_nbr),
        "srf_nbrcal_d": d(tvt_nbrcal),
        "srf_self_rmse": np.float32(self_rmse) * np.ones(len(ev), np.float32),
        "srf_nbr_std": nbr_std,
        "srf_nbr_dist": nbr_dist,
        "srf_nbrcal_rmse": np.float32(nbr_cal_rmse) * np.ones(len(ev), np.float32),
        "srf_self_vs_nbrcal": (tvt_self - tvt_nbrcal).astype(np.float32),
        "srf_self_vs_const": (tvt_self - tvt_const).astype(np.float32),
        "srf_slope_x": np.float32(a) * np.ones(len(ev), np.float32),
        "srf_slope_y": np.float32(b) * np.ones(len(ev), np.float32),
        "srf_bias": np.float32(bias) * np.ones(len(ev), np.float32),
    }
    return pd.DataFrame(feats)


def add_surface_feats(base_df, d: Path, SI, is_train):
    parts = []
    paths = sorted(d.glob("*__horizontal_well.csv"))
    tag = "train" if is_train else "test"
    t0 = time.time()
    print(f"  [{tag}] surface features start: {len(paths)} wells", flush=True)
    for i, p in enumerate(paths):
        r = build_surface_feats(p, SI, is_train)
        if r is not None:
            parts.append(r)
        if (i + 1) % int(os.environ.get("LOG_WELL_EVERY", "25")) == 0 or (i + 1) == len(paths):
            print(f"  [{tag}] surface {i + 1}/{len(paths)} wells, kept={len(parts)} ({time.time() - t0:.0f}s)", flush=True)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    print(f"  [{tag}] surface done: shape={merged.shape} (+{sf.shape[1]-1} cols, {time.time() - t0:.0f}s)", flush=True)
    return merged


def tortuosity(x, y, z, win):
    """At each point, compute window win path_len/chord_len - 1; describes wellbore tortuosity."""
    n = len(x)
    pl = np.zeros(n)  # Path length (cumulative segment lengths).
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])
    out = np.zeros(n, np.float32)
    for i in range(n):
        a = max(0, i - win)
        path_len = cum[i] - cum[a]
        chord = np.sqrt((x[i] - x[a]) ** 2 + (y[i] - y[a]) ** 2 + (z[i] - z[a]) ** 2)
        out[i] = (path_len / chord - 1.0) if chord > 1e-6 else 0.0
    return out


def local_azimuth(x, y, k):
    """At each point, compute heading from k points back (atan2). Returns sin, cos."""
    n = len(x)
    dx = np.zeros(n); dy = np.zeros(n)
    for i in range(n):
        a = max(0, i - k)
        dx[i] = x[i] - x[a]
        dy[i] = y[i] - y[a]
    az = np.arctan2(dy, dx)
    return np.sin(az).astype(np.float32), np.cos(az).astype(np.float32)


def build_feats_meta(hw_path: Path):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float)
    Z = hw["Z"].to_numpy(float); MD = hw["MD"].to_numpy(float)
    nkn = len(kn)
    ev_pos = np.arange(len(hw)) >= nkn  # after PS.

    # Inclination and horizontal progression rate (1D series).
    dmd = np.diff(MD, prepend=MD[0] - 1.0)
    dmd[dmd == 0] = 1.0
    incl = (np.gradient(Z) / np.where(np.abs(np.gradient(MD)) < 1e-6, 1.0, np.gradient(MD))).astype(np.float32)
    dxy = np.sqrt(np.gradient(X) ** 2 + np.gradient(Y) ** 2)
    dxy_dmd = (dxy / np.where(np.abs(np.gradient(MD)) < 1e-6, 1.0, np.gradient(MD))).astype(np.float32)

    # Azimuth (2D).
    az_s20, az_c20 = local_azimuth(X, Y, 20)
    # Average azimuth in known zone (overall trend before PS).
    if nkn >= 2:
        az_kn = np.arctan2(Y[nkn - 1] - Y[0], X[nkn - 1] - X[0])
    else:
        az_kn = 0.0
    az_kn_s = float(np.sin(az_kn)); az_kn_c = float(np.cos(az_kn))
    # Deviation of local azimuth from known-zone average (change in travel direction).
    az_dev = (az_s20 * az_kn_c - az_c20 * az_kn_s).astype(np.float32)  # sin(local - known).

    # Curvature (azimuth change rate).
    az_full = np.arctan2(np.gradient(Y), np.gradient(X))
    curv = np.gradient(np.unwrap(az_full)).astype(np.float32)

    # tortuosity (meta).
    tort100 = tortuosity(X, Y, Z, 100)
    tort_all_val = float(tort100[ev_pos].mean()) if ev_pos.any() else 0.0

    # Known-zone azimuth stability (meta: straight or highly curved well path).
    az_kn_std = float(np.std(np.unwrap(az_full[:nkn]))) if nkn > 5 else 0.0
    # Known-zone inclination trend (up/down tendency of the lateral).
    incl_kn_mean = float(np.mean(incl[:nkn]))
    md_total = float(MD[-1] - MD[0])

    # updip/downdip: alignment between azimuth and TVT_input slope in the known zone.
    if nkn >= 10:
        slope_tvt = float(np.polyfit(MD[:nkn], kn["TVT_input"].to_numpy(float), 1)[0])
    else:
        slope_tvt = 0.0
    # updip_proj: travel direction x TVT-change direction (signed).
    updip_proj = (az_c20 * np.sign(slope_tvt)).astype(np.float32)

    evi = ev.index
    pos = np.searchsorted(hw.index.to_numpy(), evi)

    def sc(v):
        return np.full(len(ev), np.float32(v), np.float32)

    feats = {
        "id": [f"{wid}_{i}" for i in evi],
        "m_incl": incl[pos],
        "m_dxy_dmd": dxy_dmd[pos],
        "m_az_s20": az_s20[pos],
        "m_az_c20": az_c20[pos],
        "m_az_dev": az_dev[pos],
        "m_curv": curv[pos],
        "m_tort100": tort100[pos],
        "m_updip_proj": updip_proj[pos],
        "m_az_kn_s": sc(az_kn_s),
        "m_az_kn_c": sc(az_kn_c),
        "m_az_kn_std": sc(az_kn_std),
        "m_tort_eval": sc(tort_all_val),
        "m_incl_kn_mean": sc(incl_kn_mean),
        "m_md_total": sc(md_total),
        "m_slope_tvt": sc(slope_tvt),
    }
    # exp121: second-order well-path geometry (4D). build rate = inclination change rate, a physical driver of apparent-dip drift,
    #  plus 3D dogleg severity (DLS, minimum-curvature method). Adds second-order well-path curvature information not in existing incl/az/curv.
    #  TRAJ=0(default) appends nothing -> bitwise match. Column order rule: append new features at the end.
    if os.environ.get("TRAJ", "0") == "1":
        gMD = np.gradient(MD); gMD = np.where(np.abs(gMD) < 1e-6, 1.0, gMD)
        dh = np.sqrt(np.gradient(X) ** 2 + np.gradient(Y) ** 2)
        inc_ang = np.arctan2(dh, np.gradient(Z))           # inclination angle from vertical (rad), horizontal well ~pi/2.
        sin_inc = np.sin(inc_ang)
        g_inc = np.gradient(inc_ang)
        az_un = np.unwrap(az_full)
        g_az = np.gradient(az_un)
        build_rate = (g_inc / gMD).astype(np.float32)       # inclination change rate (vertical curvature).
        turn_rate = (g_az / gMD).astype(np.float32)         # azimuth change rate (turn).
        dls = (np.sqrt(g_inc ** 2 + (sin_inc * g_az) ** 2) / gMD).astype(np.float32)  # 3D dogleg severity
        feats["m_inc_ang"] = inc_ang.astype(np.float32)[pos]
        feats["m_build_rate"] = build_rate[pos]
        feats["m_turn_rate"] = turn_rate[pos]
        feats["m_dls"] = dls[pos]
    return pd.DataFrame(feats)


def add_feats_meta(base_df, d: Path):
    parts = []
    paths = sorted(d.glob("*__horizontal_well.csv"))
    tag = "train" if "train" in str(d) else "test"
    t0 = time.time()
    print(f"  [{tag}] trajectory/meta features start: {len(paths)} wells", flush=True)
    for i, p in enumerate(paths):
        r = build_feats_meta(p)
        if r is not None:
            parts.append(r)
        if (i + 1) % int(os.environ.get("LOG_WELL_EVERY", "25")) == 0 or (i + 1) == len(paths):
            print(f"  [{tag}] meta {i + 1}/{len(paths)} wells, kept={len(parts)} ({time.time() - t0:.0f}s)", flush=True)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    print(f"  [{tag}] meta done: shape={merged.shape} (+{sf.shape[1]-1} cols, {time.time() - t0:.0f}s)", flush=True)
    return merged


DIP_SPW = 80
DIP_K = 32


class DipField:
    """Estimate local gradients (dip) in batches from the s=TVT+Z point cloud of all training wells."""

    def __init__(self, well_ids, data_dir, spw=DIP_SPW):
        xs, ys, ss, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            s = (df["TVT"] + df["Z"]).to_numpy()
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            xs.append(df["X"].to_numpy()[ix]); ys.append(df["Y"].to_numpy()[ix])
            ss.append(s[ix]); wids.extend([wid] * len(ix))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.s = np.concatenate(ss).astype(np.float64)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def gradient(self, xy_q, self_wid=None, k=DIP_K, nfetch=3000):
        """Return (a,b) of local plane s ~ a*x+b*y+c for each query point, plus neighbor distance."""
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.s))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xy[ik, 0]; yn = self.xy[ik, 1]; sn = self.s[ik]
        # Numerical stabilization: translate to the query-point center.
        xn = xn - xy_q[:, 0:1]; yn = yn - xy_q[:, 1:2]
        wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-6; A[:, 1, 1] += 1e-6; A[:, 2, 2] += 1e-6
        rhs = np.stack([(wx * sn).sum(1), (wy * sn).sum(1), (w * sn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.lstsq(A[r], rhs[r], rcond=None)[0]
                except Exception:
                    pass
        a = coef[:, 0]; b = coef[:, 1]
        ok = vk.any(1)
        a = np.where(ok, a, 0.0); b = np.where(ok, b, 0.0)
        return (a.astype(np.float64), b.astype(np.float64),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


def build_feats_dip(hw_path: Path, DF: DipField, is_train: bool):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    swid = wid if is_train else None
    lk = kn.iloc[-1]
    x0, y0, z0 = float(lk["X"]), float(lk["Y"]), float(lk["Z"])
    last_tvt = float(lk["TVT_input"])
    s0 = last_tvt + z0

    Xe = ev["X"].to_numpy(float); Ye = ev["Y"].to_numpy(float); Ze = ev["Z"].to_numpy(float)
    n = len(ev)

    # Local dip at each point on the eval path.
    a, b, dknn = DF.gradient(np.column_stack([Xe, Ye]), self_wid=swid)
    px = np.concatenate([[x0], Xe]); py = np.concatenate([[y0], Ye])
    dx = np.diff(px); dy = np.diff(py)
    ds = a * dx + b * dy
    s_pred = s0 + np.cumsum(ds)
    tvt_dipint = s_pred - Ze

    # Validate the dip field using the last 200 points of the known zone -> drift calibration and trust meta.
    t200 = kn.tail(200)
    if len(t200) >= 30:
        Xk = t200["X"].to_numpy(float); Yk = t200["Y"].to_numpy(float)
        sk = t200["TVT_input"].to_numpy(float) + t200["Z"].to_numpy(float)
        ak, bk, _ = DF.gradient(np.column_stack([Xk, Yk]), self_wid=swid)
        ds_pred_k = ak[1:] * np.diff(Xk) + bk[1:] * np.diff(Yk)
        ds_true_k = np.diff(sk)
        kn_rmse = float(np.sqrt(np.mean((ds_pred_k - ds_true_k) ** 2)))
        # Drift per ft (systematic error of the dip field).
        mdk = t200["MD"].to_numpy(float)
        dmd = np.diff(mdk); m = dmd > 0
        drift = float(np.median((ds_true_k - ds_pred_k)[m] / dmd[m])) if m.sum() >= 10 else 0.0
    else:
        kn_rmse, drift = np.nan, 0.0
    md_since = ev["MD"].to_numpy(float) - float(lk["MD"])
    tvt_dipcal = tvt_dipint + drift * md_since

    def d(v):
        return (np.asarray(v, float) - last_tvt).astype(np.float32)

    def sc(v):
        return np.full(n, np.float32(v), np.float32)

    dip_mag = np.sqrt(a ** 2 + b ** 2)
    step = np.sqrt(dx ** 2 + dy ** 2)
    align = ds / (dip_mag * step + 1e-9)  # Cosine between travel direction and dip direction.
    feats = {
        "id": [f"{wid}_{i}" for i in ev.index],
        "dipint_d": d(tvt_dipint),
        "dipcal_d": d(tvt_dipcal),
        "dip_along": (ds / np.maximum(step, 1e-6)).astype(np.float32),
        "dip_mag": dip_mag.astype(np.float32),
        "dip_align": np.clip(align, -1.5, 1.5).astype(np.float32),
        "dip_knn_dist": dknn,
        "dip_kn_rmse": sc(kn_rmse),
        "dip_drift": sc(drift),
        "dipint_vs_flat": d(tvt_dipint) - np.float32(0.0),
    }
    return pd.DataFrame(feats)


def add_feats_dip(base_df, d: Path, DF, is_train):
    parts = []
    paths = sorted(d.glob("*__horizontal_well.csv"))
    tag = "train" if is_train else "test"
    t0 = time.time()
    print(f"  [{tag}] dip-field features start: {len(paths)} wells", flush=True)
    for i, p in enumerate(paths):
        r = build_feats_dip(p, DF, is_train)
        if r is not None:
            parts.append(r)
        if (i + 1) % int(os.environ.get("LOG_WELL_EVERY", "25")) == 0 or (i + 1) == len(paths):
            print(f"  [{tag}] dip {i + 1}/{len(paths)} wells, kept={len(parts)} ({time.time() - t0:.0f}s)", flush=True)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    print(f"  [{tag}] dip done: shape={merged.shape} (+{sf.shape[1]-1} cols, {time.time() - t0:.0f}s)", flush=True)
    return merged




# ---------------- main pipeline ----------------
def apply_pp_final(md_since, model_d, pf_d, w_pf=0.05, tau=50, alpha=1.0):
    dd = model_d * (1 - w_pf) + pf_d * w_pf
    if tau:
        dd = dd * (1.0 - np.exp(-np.maximum(md_since, 0.0) / tau))
    return dd * alpha


def sg_smooth_final(wells, vals, sg_w=17, sg_p=3):
    from scipy.signal import savgol_filter
    out = vals.copy()
    df = pd.DataFrame({"w": wells})
    for _, gidx in df.groupby("w", sort=False).indices.items():
        v = vals[gidx]
        nn = len(v)
        wl = min(sg_w, nn)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            out[gidx] = savgol_filter(v, wl, sg_p)
    return out


def build_all(d, is_train, SI, DFIELD):
    tag = "train" if is_train else "test"
    t0 = time.time()
    print(f"[{tag}] build_all start: {d}", flush=True)
    df = build_dataset(d, is_train=is_train)
    print(f"[{tag}] base/PF/beam/NCC/spatial done: {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    df = add_surface_feats(df, d, SI, is_train)
    print(f"[{tag}] after surface: {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    df = add_feats_meta(df, d)
    print(f"[{tag}] after meta: {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    df = add_feats_dip(df, d, DFIELD, is_train)
    out = df.drop(columns=["m_curv", "m_dxy_dmd"])
    print(f"[{tag}] build_all done: {out.shape} ({time.time() - t0:.0f}s)", flush=True)
    return out


# ===== exp063: lik-PF (cal/nocal) physical features (embedded, numba) =====
# Make _PNS=number of filters / _PNP=number of particles configurable by env for lik-PF sweep. Default 48/400 exactly matches current.
_PNS = int(os.environ.get("PNS", "48")); _PNP = int(os.environ.get("PNP", "400"))
_PSCALE = 5.0; _PSHRINK = 0.5; _PDEG = 4


def _affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def _robfit_mod(x, v, deg=4, ni=5, c=1.5):
    n = len(x)
    if n <= deg + 1:
        return v.copy()
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9); A = np.vander(xn, deg + 1); w = np.ones(n); co = None
    for _ in range(ni):
        co, *_ = np.linalg.lstsq(A * w[:, None], v * w, rcond=None); r = v - A @ co
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9; u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
    return A @ co


@njit(cache=True, fastmath=True, nogil=True)
def _phys_pf(md_v, z_v, gr_v, grid_gr, g0, dg, a, b, gs, ir, last_U, last_MD, S, N, seed,
             sdip, kappa, liktype, likp):
    # sdip[i]=expected dU/dMD from neighboring wells (spatial dip), kappa=pullback strength. kappa=0 bitwise matches current (random consumption unchanged).
    # liktype: 0=gauss(current) 1=student-t(nu=likp) 2=cauchy 3=huber(c=likp). 0 matches current.
    np.random.seed(seed)
    ng = grid_gr.shape[0]; E = md_v.shape[0]
    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
    glo = g0; ghi = g0 + (ng - 1) * dg
    pos = np.empty((S, N)); rate = np.empty((S, N)); w = np.empty((S, N))
    for s in range(S):
        for n in range(N):
            pos[s, n] = last_U + 4.5 * np.random.normal()
            rate[s, n] = ir + 0.01 * np.random.normal()
            w[s, n] = 1.0 / N
    res = np.empty((S, E)); loglik = np.zeros(S)
    idx = np.empty(N, np.int64); cw = np.empty(N); tpos = np.empty(N); trate = np.empty(N)
    prev = last_MD
    for i in range(E):
        dm = md_v[i] - prev
        if dm < 1.0:
            dm = 1.0
        zi = z_v[i]; gri = gr_v[i]; sd = sdip[i]
        for s in range(S):
            sw = 0.0
            for n in range(N):
                rate[s, n] = MOM * rate[s, n] + kappa * (sd - rate[s, n]) + VN * np.random.normal()
                p = pos[s, n] + rate[s, n] * dm + PN * np.random.normal()
                tvt = p - zi
                if tvt < glo:
                    tvt = glo
                elif tvt > ghi:
                    tvt = ghi
                pos[s, n] = tvt + zi
                fx = (tvt - g0) / dg; k = int(fx)
                if k < 0:
                    eg = grid_gr[0]
                elif k >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - k; eg = grid_gr[k] * (1.0 - fr) + grid_gr[k + 1] * fr
                eg = a * eg + b
                d = (gri - eg) / gs; d2 = d * d
                if liktype == 0:        # gauss (current).
                    if d2 > 600.0:
                        d2 = 600.0
                    lk = np.exp(-0.5 * d2)
                elif liktype == 1:      # student-t (nu=likp): heavy tail.
                    lk = (1.0 + d2 / likp) ** (-0.5 * (likp + 1.0))
                elif liktype == 2:      # cauchy (=student-t ν=1)
                    lk = 1.0 / (1.0 + d2)
                else:                   # huber (c=likp): small d is Gaussian, large d is linear.
                    ad = abs(d)
                    if ad <= likp:
                        lk = np.exp(-0.5 * d2)
                    else:
                        lk = np.exp(-likp * ad + 0.5 * likp * likp)
                if lk < 1e-300:
                    lk = 1e-300
                nw = w[s, n] * lk; w[s, n] = nw; sw += nw
            if sw < 1e-300:
                sw = 1e-300
            loglik[s] += np.log(sw)
            for n in range(N):
                w[s, n] /= sw
            ei = 0.0
            for n in range(N):
                ei += w[s, n] * w[s, n]
            if 1.0 / ei < RESAMP * N:
                acc = 0.0
                for n in range(N):
                    acc += w[s, n]; cw[n] = acc
                u0 = np.random.random() / N; j = 0
                for n in range(N):
                    u = u0 + n / N
                    while j < N - 1 and cw[j] < u:
                        j += 1
                    idx[n] = j
                for n in range(N):
                    tpos[n] = pos[s, idx[n]] + RP * np.random.normal()
                    trate[n] = rate[s, idx[n]] + RR * np.random.normal()
                est = 0.0
                for n in range(N):
                    pos[s, n] = tpos[n]; rate[s, n] = trate[n]; w[s, n] = 1.0 / N
                    est += (pos[s, n] - zi) / N
                res[s, i] = est
            else:
                est = 0.0
                for n in range(N):
                    est += w[s, n] * (pos[s, n] - zi)
                res[s, i] = est
        prev = md_v[i]
    return res, loglik


def _phys_inputs(hw, tw_tvt, tw_gr, calibrate):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    last = kn.iloc[-1]
    last_U = float(last["TVT_input"]) + float(last["Z"]); last_MD = float(last["MD"])
    last_tvt = float(last["TVT_input"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    if calibrate:
        a, b = _affine_cal(kn["GR"].to_numpy(float), tw_at_k)
        a = 1.0 + _PSHRINK * (a - 1.0); b = _PSHRINK * b
    else:
        a, b = 1.0, 0.0
    resid = np.nan_to_num(kn["GR"].to_numpy(float), nan=0.0) - (a * tw_at_k + b)
    gs = float(np.clip(np.nanstd(resid), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float)); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    md_v = ev["MD"].to_numpy(float); z_v = ev["Z"].to_numpy(float)
    # GR missing-value fill: interp (linear at both ends, default)/ffill (LOCF hold latest prior value)/nearest. Matches infer through config travel.
    _grfill = os.environ.get("GRFILL", "interp"); _gser = hw["GR"]
    if _grfill == "ffill":
        gr_full = _gser.ffill().bfill().fillna(np.mean(tw_gr)).to_numpy(float)
    elif _grfill == "nearest":
        gr_full = _gser.interpolate(method="nearest", limit_direction="both").fillna(np.mean(tw_gr)).to_numpy(float)
    else:
        gr_full = _gser.interpolate(limit_direction="both").fillna(np.mean(tw_gr)).to_numpy(float)
    gr_v = gr_full[ev.index.to_numpy()]
    dgr = 0.25; g0 = float(tw_tvt[0]) - 50.0; ghi = float(tw_tvt[-1]) + 50.0; ng = int((ghi - g0) / dgr) + 1
    grid_gr = np.interp(g0 + dgr * np.arange(ng), tw_tvt, tw_gr).astype(np.float64)
    return (md_v, z_v, gr_v, grid_gr, g0, dgr, float(a), float(b), gs, ir, last_U, last_MD), last_tvt


@njit(cache=True, fastmath=True)
def _geo_pearson(md_v, z_v, gr_v, grid_gr, g0, dg, last_U, last_MD, last_tvt, reg_dip,
                 seg_len, ndip, dip_span, w_smooth, w_reg):
    # Reproduction of patented US11480045 big-segment geosteering: grid-search dip (=dU/dMD datum slope) in each segment and
    #  maximize K = Pearson(horizontal GR, synthetic typewell GR) / (1 + w_smooth*|delta_alpha| + w_reg*|delta_beta|).
    #  delta_alpha=adjacent-segment dip difference (smoothness), delta_beta=deviation from regional dip. synthetic GR=typewell(TVT=U-Z). Greedy forward.
    E = md_v.shape[0]; ng = grid_gr.shape[0]
    out_tvt = np.empty(E); out_conf = np.empty(E)
    cur_U = last_U; cur_md = last_MD; prev_dip = reg_dip
    i = 0
    while i < E:
        j = i + seg_len
        if j > E:
            j = E
        n = j - i
        best_K = -1.0e18; best_dip = reg_dip; best_T = 0.0
        for di in range(ndip):
            dip = reg_dip - dip_span + (2.0 * dip_span) * di / (ndip - 1)
            mh = 0.0; ms = 0.0
            for k in range(i, j):
                tvt = (cur_U + dip * (md_v[k] - cur_md)) - z_v[k]
                fx = (tvt - g0) / dg; idx = int(fx)
                if idx < 0:
                    eg = grid_gr[0]
                elif idx >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - idx; eg = grid_gr[idx] * (1.0 - fr) + grid_gr[idx + 1] * fr
                mh += gr_v[k]; ms += eg
            mh /= n; ms /= n
            cov = 0.0; vh = 0.0; vs = 0.0
            for k in range(i, j):
                tvt = (cur_U + dip * (md_v[k] - cur_md)) - z_v[k]
                fx = (tvt - g0) / dg; idx = int(fx)
                if idx < 0:
                    eg = grid_gr[0]
                elif idx >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - idx; eg = grid_gr[idx] * (1.0 - fr) + grid_gr[idx + 1] * fr
                dh = gr_v[k] - mh; ds = eg - ms
                cov += dh * ds; vh += dh * dh; vs += ds * ds
            if vh > 1e-9 and vs > 1e-9:
                T = cov / np.sqrt(vh * vs)
            else:
                T = 0.0
            if T < 0.0:
                T = 0.001          # Patent: T<0 -> small value.
            sp = dip_span + 1e-9
            pen = 1.0 + w_smooth * (abs(dip - prev_dip) / sp) + w_reg * (abs(dip - reg_dip) / sp)
            K = T / pen
            if K > best_K:
                best_K = K; best_dip = dip; best_T = T
        for k in range(i, j):
            out_tvt[k] = (cur_U + best_dip * (md_v[k] - cur_md)) - z_v[k]
            out_conf[k] = best_T
        cur_U = cur_U + best_dip * (md_v[j - 1] - cur_md)
        cur_md = md_v[j - 1]; prev_dip = best_dip
        i = j
    return out_tvt, out_conf


@njit(cache=True, fastmath=True)
def _autocorr_feat(gr_v, win, lmin, lmax):
    # exp136/137: maximize GR-window self-correlation (Pearson) over a lag range.
    #  SELFCORR(near lag 10-60)=local periodicity, REPSECTION(far lag 100-400)=repeated-layer detection from faults. High = ambiguous match.
    E = gr_v.shape[0]
    out = np.zeros(E)
    for i in range(E):
        a0 = i - win
        if a0 < 0:
            continue
        best = 0.0
        for lag in range(lmin, lmax + 1):
            b0 = a0 - lag
            if b0 < 0:
                break
            ma = 0.0; mb = 0.0
            for k in range(win):
                ma += gr_v[a0 + k]; mb += gr_v[b0 + k]
            ma /= win; mb /= win
            cov = 0.0; va = 0.0; vb = 0.0
            for k in range(win):
                da = gr_v[a0 + k] - ma; db = gr_v[b0 + k] - mb
                cov += da * db; va += da * da; vb += db * db
            if va > 1e-9 and vb > 1e-9:
                t = cov / np.sqrt(va * vb)
                if t > best:
                    best = t
        out[i] = best
    return out


@njit(cache=True, fastmath=True)
def _geo_spectrum(md_v, z_v, gr_v, grid_gr, g0, dg, last_U, last_MD, reg_dip, win, ndip, dip_span):
    # exp134 geosteering spectrum (patent Fig7-8): compute Pearson similarity over a dip range for local window +/-win at each eval point.
    #  Returns: spec_peak (max similarity=confidence), spec_spread (spectrum width=dip ambiguity), spec_nmax (number of local maxima=multi-modality).
    E = md_v.shape[0]; ng = grid_gr.shape[0]
    peak = np.empty(E); spread = np.empty(E); nmax = np.empty(E)
    sp = np.empty(ndip)
    for i in range(E):
        lo = i - win if i - win > 0 else 0
        hi = i + win + 1 if i + win + 1 < E else E
        n = hi - lo
        for di in range(ndip):
            dip = reg_dip - dip_span + (2.0 * dip_span) * di / (ndip - 1)
            mh = 0.0; ms = 0.0
            for k in range(lo, hi):
                tvt = (last_U + dip * (md_v[k] - last_MD)) - z_v[k]
                fx = (tvt - g0) / dg; idx = int(fx)
                if idx < 0:
                    eg = grid_gr[0]
                elif idx >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - idx; eg = grid_gr[idx] * (1.0 - fr) + grid_gr[idx + 1] * fr
                mh += gr_v[k]; ms += eg
            mh /= n; ms /= n
            cov = 0.0; vh = 0.0; vs = 0.0
            for k in range(lo, hi):
                tvt = (last_U + dip * (md_v[k] - last_MD)) - z_v[k]
                fx = (tvt - g0) / dg; idx = int(fx)
                if idx < 0:
                    eg = grid_gr[0]
                elif idx >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - idx; eg = grid_gr[idx] * (1.0 - fr) + grid_gr[idx + 1] * fr
                dh = gr_v[k] - mh; ds = eg - ms
                cov += dh * ds; vh += dh * dh; vs += ds * ds
            T = cov / np.sqrt(vh * vs) if (vh > 1e-9 and vs > 1e-9) else 0.0
            if T < 0.0:
                T = 0.0
            sp[di] = T
        pk = 0.0; ssum = 0.0
        for di in range(ndip):
            if sp[di] > pk:
                pk = sp[di]
            ssum += sp[di]
        # spread = entropy-like width of the normalized spectrum, nmax = number of local maxima.
        cnt = 0
        for di in range(1, ndip - 1):
            if sp[di] > sp[di - 1] and sp[di] > sp[di + 1] and sp[di] > 0.3 * pk:
                cnt += 1
        peak[i] = pk
        spread[i] = (ssum / (ndip * pk)) if pk > 1e-6 else 1.0   # close to 1 = flat = ambiguous.
        nmax[i] = cnt
    return peak, spread, nmax


@njit(cache=True, fastmath=True)
def _global_decode(md_v, z_v, gr_v, grid_gr, g0, dg, a, b, gs, last_U, last_MD, last_tvt,
                   reg_dip, nu, du, wbeta, band):
    # Phase4 global decoding (Viterbi): state = U offset from the regional-dip centerline (nu grid, du spacing).
    #  emission=GR Gaussian squared residual, transition=local dip deviation from regional dip^2 (wbeta).
    #  Solve the MAP trajectory over all observations with backward backtracking -> avoids greedy drift accumulation in the forward filter.
    E = md_v.shape[0]; ng = grid_gr.shape[0]; half = nu // 2; INF = 1.0e18

    def _eg(i, s):
        center = last_U + reg_dip * (md_v[i] - last_MD)
        tvt = (center + (s - half) * du) - z_v[i]
        fx = (tvt - g0) / dg; idx = int(fx)
        if idx < 0:
            e = grid_gr[0]
        elif idx >= ng - 1:
            e = grid_gr[ng - 1]
        else:
            fr = fx - idx; e = grid_gr[idx] * (1.0 - fr) + grid_gr[idx + 1] * fr
        return a * e + b

    cost = np.empty(nu); ncost = np.empty(nu); bp = np.empty((E, nu), np.int32)
    for s in range(nu):
        d = (gr_v[0] - _eg(0, s)) / gs
        off = (s - half) * du
        cost[s] = d * d + 0.01 * off * off / (du * du)   # Weakly anchor the start to the centerline.
        bp[0, s] = s
    prev_md = md_v[0]
    for i in range(1, E):
        dm = md_v[i] - prev_md
        if dm < 1.0:
            dm = 1.0
        for s in range(nu):
            d = (gr_v[i] - _eg(i, s)) / gs; em = d * d
            best = INF; bestp = s
            lo = s - band if s - band > 0 else 0
            hi = s + band + 1 if s + band + 1 < nu else nu
            for sp in range(lo, hi):
                dd = (s - sp) * du / dm        # Local deviation from regional dip.
                c = cost[sp] + wbeta * dd * dd
                if c < best:
                    best = c; bestp = sp
            ncost[s] = em + best; bp[i, s] = bestp
        for s in range(nu):
            cost[s] = ncost[s]
        prev_md = md_v[i]
    s = 0; bestc = INF
    for ss in range(nu):
        if cost[ss] < bestc:
            bestc = cost[ss]; s = ss
    out_tvt = np.empty(E)
    for i in range(E - 1, -1, -1):
        center = last_U + reg_dip * (md_v[i] - last_MD)
        out_tvt[i] = (center + (s - half) * du) - z_v[i]
        s = bp[i, s]
    return out_tvt


def _exact_pf_parallel_supported():
    """The exact scheduler currently covers the frozen train19 PF feature set."""
    unsupported_flags = (
        "MULTISCALE", "POSTERIOR", "EVALCAL", "GEOPEARSON", "GLOBALDECODE",
        "RATEFEAT", "GEOSPECTRUM", "SELFCORR", "REPSECTION", "MODES", "GRFEATS2",
    )
    enabled = [name for name in unsupported_flags if os.environ.get(name, "0") == "1"]
    if float(os.environ.get("TVTREG", "0") or "0") != 0.0:
        enabled.append("TVTREG")
    return enabled


def _prepare_exact_phys_one(hw_path, sdip_arr=None):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path).reset_index(drop=True)
    tw = pd.read_csv(str(hw_path).replace("__horizontal_well", "__typewell")).sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)
    ev = hw[hw["TVT_input"].isna()]
    kn = hw[hw["TVT_input"].notna()]
    if len(ev) == 0 or len(kn) == 0:
        return None
    z = ev["Z"].to_numpy(float)
    md = ev["MD"].to_numpy(float)
    E = len(ev)
    if sdip_arr is not None and len(sdip_arr) == E:
        sd = np.ascontiguousarray(sdip_arr, np.float64)
        kappa = float(os.environ.get("DIPKAPPA", "0.1"))
    else:
        sd = np.zeros(E)
        kappa = 0.0
    liktype = {"gauss": 0, "studentt": 1, "cauchy": 2, "huber": 3}.get(
        os.environ.get("LIKTYPE", "gauss"), 0
    )
    likp = float(os.environ.get("LIKP", "4.0"))
    inp_noc, last_tvt = _phys_inputs(hw, tw_tvt, tw_gr, False)
    inp_cal, _ = _phys_inputs(hw, tw_tvt, tw_gr, True)
    return {
        "wid": wid,
        "ev_index": ev.index.to_numpy(),
        "z": z,
        "mds": md - md.min(),
        "E": E,
        "last_tvt": last_tvt,
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "inputs": (inp_noc, inp_cal),
        "sd": sd,
        "kappa": kappa,
        "liktype": liktype,
        "likp": likp,
    }


def _exact_mode_summary(res, wts, Km):
    levk = res.mean(1)
    ordk = np.argsort(levk)
    cwk = np.cumsum(wts[ordk])
    totk = cwk[-1] + 1e-12
    bnd = [0]
    for j in range(1, Km):
        c = int(np.searchsorted(cwk, totk * j / Km))
        bnd.append(min(max(c, bnd[-1] + 1), len(ordk)))
    bnd.append(len(ordk))
    centers = []
    weights = []
    for j in range(Km):
        gi = ordk[bnd[j]:bnd[j + 1]]
        if len(gi) > 0:
            wk = wts[gi] / (wts[gi].sum() + 1e-12)
            center = (wk[:, None] * res[gi]).sum(0)
            weight = float(wts[gi].sum())
        else:
            center = None
            weight = 0.0
        centers.append(center)
        weights.append(weight)
    return centers, weights


def _genphys_exact_parallel(paths, sdip_map, n_jobs):
    """Run frozen train19 lik-PF tasks across seeds without changing PF math."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prepared = []
    for path in paths:
        wid = Path(path).stem.replace("__horizontal_well", "")
        item = _prepare_exact_phys_one(str(path), sdip_map.get(wid))
        if item is not None:
            prepared.append(item)
    if not prepared:
        return []

    mseed = max(1, int(os.environ.get("LIKPF_MSEED", "1")))
    Km = int(os.environ.get("NMODES", "3"))
    modes_k = os.environ.get("MODESK", "0") == "1"

    # Compile once before threads start. Every real call reseeds internally.
    warm = list(prepared[0]["inputs"][0])
    warm[0] = warm[0][:1]
    warm[1] = warm[1][:1]
    warm[2] = warm[2][:1]
    _phys_pf(
        *warm, _PNS, _PNP, 42, prepared[0]["sd"][:1], prepared[0]["kappa"],
        prepared[0]["liktype"], prepared[0]["likp"]
    )

    def run_task(task):
        wi, tag, sx = task
        item = prepared[wi]
        res, ll = _phys_pf(
            *item["inputs"][tag], _PNS, _PNP, 42 + sx, item["sd"], item["kappa"],
            item["liktype"], item["likp"]
        )
        llm = ll.max()
        wts = np.exp((ll - llm) / _PSCALE)
        wts /= wts.sum()
        phys = (wts[:, None] * res).sum(0)
        primary = None
        if sx == 0:
            primary = {
                "seed_std": float(np.mean(np.std(res, axis=0))),
                "eff": float(1.0 / np.sum(wts ** 2)),
                "loglik": float(llm / item["E"]),
            }
            if tag == 0 and modes_k:
                centers, mode_weights = _exact_mode_summary(res, wts, Km)
                centers = [phys if center is None else center for center in centers]
                primary["centers"] = centers
                primary["mode_weights"] = mode_weights
        return wi, tag, sx, phys, primary

    tasks = [
        (wi, tag, sx)
        for wi in sorted(range(len(prepared)), key=lambda i: prepared[i]["E"], reverse=True)
        for tag in (0, 1)
        for sx in range(mseed)
    ]
    results = {}
    workers = max(1, min(int(n_jobs), len(tasks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="full249_pf") as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            wi, tag, sx, phys, primary = future.result()
            results[(wi, tag, sx)] = (phys, primary)

    frames = []
    for wi, item in enumerate(prepared):
        out = {}
        diag = {}
        extra = {}
        noc_phys = None
        noc_primary = None
        for tag, name in ((0, "noc"), (1, "cal")):
            phys = results[(wi, tag, 0)][0].copy()
            primary = results[(wi, tag, 0)][1]
            for sx in range(1, mseed):
                phys += results[(wi, tag, sx)][0]
            phys /= mseed
            proj = _robfit_mod(item["mds"], phys + item["z"], _PDEG) - item["z"]
            out[name + "_raw"] = phys - item["last_tvt"]
            out[name + "_proj"] = proj - item["last_tvt"]
            if tag == 0:
                noc_phys = phys
                noc_primary = primary
                diag["seed_std"] = primary["seed_std"]
                diag["eff"] = primary["eff"]
                diag["loglik"] = primary["loglik"]
                diag["rough"] = float(np.mean(np.abs(np.diff(phys, 2)))) if item["E"] > 2 else 0.0

        if modes_k and noc_primary is not None:
            centers = noc_primary["centers"]
            mode_weights = noc_primary["mode_weights"]
            for j, center in enumerate(centers):
                extra[f"mk{Km}_{j}_d"] = center - item["last_tvt"]
                extra[f"mk{Km}_{j}_w"] = np.full(item["E"], np.float32(mode_weights[j]))
            extra[f"mk{Km}_spread"] = np.abs(centers[-1] - centers[0])
            extra[f"mk{Km}_cal_vs_mid"] = noc_phys - centers[Km // 2]
            extra[f"mk{Km}_lohi"] = (
                np.abs(centers[1] - centers[0]) - np.abs(centers[-1] - centers[-2])
                if Km >= 3 else np.zeros(item["E"])
            )
            if os.environ.get("MODECONF", "0") == "1":
                grv_m = item["inputs"][0][2]
                gr_abs = [
                    np.abs(grv_m - np.interp(center, item["tw_tvt"], item["tw_gr"])).astype(np.float32)
                    for center in centers
                ]
                GM = np.stack(gr_abs, axis=0)
                ordm = np.argsort(GM, axis=0)
                cols = np.arange(item["E"])
                best = GM[ordm[0], cols]
                second = GM[ordm[1], cols] if Km > 1 else best
                extra[f"mk{Km}_gr_minabs"] = best.astype(np.float32)
                extra[f"mk{Km}_gr_margin"] = (second - best).astype(np.float32)
                extra[f"mk{Km}_gr_best"] = ordm[0].astype(np.float32)
                extra[f"mk{Km}_gr_range"] = (GM.max(axis=0) - GM.min(axis=0)).astype(np.float32)

        frame = pd.DataFrame({
            "id": [f"{item['wid']}_{i}" for i in item["ev_index"]],
            "cal_proj_d": out["cal_proj"],
            "noc_proj_d": out["noc_proj"],
            "cal_raw_d": out["cal_raw"],
            "noc_raw_d": out["noc_raw"],
        })
        frame["cn_proj"] = frame["cal_proj_d"] - frame["noc_proj_d"]
        frame["cn_raw"] = frame["cal_raw_d"] - frame["noc_raw_d"]
        for key, value in diag.items():
            frame[key] = value
        for key, value in extra.items():
            frame[key] = value
        frames.append(frame)
    return frames


def genphys_one(hw_path, sdip_arr=None):
    try:
        wid = Path(hw_path).stem.replace("__horizontal_well", "")
        hw = pd.read_csv(hw_path).reset_index(drop=True)
        tw = pd.read_csv(str(hw_path).replace("__horizontal_well", "__typewell")).sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)
        ev = hw[hw["TVT_input"].isna()]; kn = hw[hw["TVT_input"].notna()]
        if len(ev) == 0 or len(kn) == 0:
            return None
        z = ev["Z"].to_numpy(float); md = ev["MD"].to_numpy(float); E = len(ev); mds = md - md.min()
        # Spatial-dip fusion (breakthrough): sdip_arr=expected dU/dMD from neighboring wells. kappa=0 bitwise matches current.
        if sdip_arr is not None and len(sdip_arr) == E:
            sd = np.ascontiguousarray(sdip_arr, np.float64); kappa = float(os.environ.get("DIPKAPPA", "0.1"))
        else:
            sd = np.zeros(E); kappa = 0.0
        # Robust likelihood: gauss(0,current)/studentt(1)/cauchy(2)/huber(3). likp=nu or c.
        _lt = {"gauss": 0, "studentt": 1, "cauchy": 2, "huber": 3}
        liktype = _lt.get(os.environ.get("LIKTYPE", "gauss"), 0)
        likp = float(os.environ.get("LIKP", "4.0"))
        MULTI = os.environ.get("MULTISCALE", "0") == "1"   # exp070: multi-scale proj (s3/s8/s12 + ms_std).
        POST = os.environ.get("POSTERIOR", "0") == "1"     # exp074: PF posterior quantiles (q10/q25/q75/q90).
        # exp123: hengck23 "TVT RMSE optimization" (disc/699326). Down-weight filters that fit GR well but have wrong TVT by reweighting with
        #  deviation from the linear(md,z) prior, not GR likelihood alone. TVTREG=0 matches current.
        TVTREG = float(os.environ.get("TVTREG", "0"))
        prior_tvt = None
        if TVTREG > 0:
            kmd = kn["MD"].to_numpy(float); kz = kn["Z"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float)
            mm = kmd.mean(); zm = kz.mean(); ms = kmd.std() + 1e-9; zs = kz.std() + 1e-9
            Ak = np.column_stack([(kmd - mm) / ms, (kz - zm) / zs, np.ones(len(kmd))])
            coef = np.linalg.solve(Ak.T @ Ak + 1e-3 * np.eye(3), Ak.T @ ktvt)
            prior_tvt = (np.column_stack([(md - mm) / ms, (z - zm) / zs, np.ones(E)]) @ coef)
        out = {}; diag = {}; extra = {}
        inp_noc = None; inp_cal = None
        for calf, tag in [(False, "noc"), (True, "cal")]:
            inp, last_tvt = _phys_inputs(hw, tw_tvt, tw_gr, calf)
            if tag == "noc":
                inp_noc = inp
            if tag == "cal":
                inp_cal = inp
            res, ll = _phys_pf(*inp, _PNS, _PNP, 42, sd, kappa, liktype, likp); llm = ll.max()
            if TVTREG > 0 and prior_tvt is not None:   # exp123: reweight by GR likelihood - lambda*(prior deviation).
                pen = ((res - prior_tvt[None, :]) ** 2).mean(1)
                pen = (pen - pen.min()) / (pen.std() + 1e-9)   # Nondimensionalize (scale stability).
                wts = np.exp((ll - llm) / _PSCALE - TVTREG * pen)
            else:
                wts = np.exp((ll - llm) / _PSCALE)             # s5 (=current cal_proj_d, unchanged).
            wts /= wts.sum()
            phys = (wts[:, None] * res).sum(0)
            # exp124: multi-seed average of lik-PF itself, the most important cal_proj feature, to remove stochastic noise (same mechanism as mseed8; expected larger effect).
            LIKPF_MSEED = int(os.environ.get("LIKPF_MSEED", "1"))   # 1 bitwise matches current (res/ll/wts keep seed42).
            if LIKPF_MSEED > 1:
                phys_acc = phys.copy()
                for sx in range(1, LIKPF_MSEED):
                    r2, l2 = _phys_pf(*inp, _PNS, _PNP, 42 + sx, sd, kappa, liktype, likp); lm2 = l2.max()
                    if TVTREG > 0 and prior_tvt is not None:
                        pe = ((r2 - prior_tvt[None, :]) ** 2).mean(1); pe = (pe - pe.min()) / (pe.std() + 1e-9)
                        w2 = np.exp((l2 - lm2) / _PSCALE - TVTREG * pe)
                    else:
                        w2 = np.exp((l2 - lm2) / _PSCALE)
                    w2 /= w2.sum()
                    phys_acc += (w2[:, None] * r2).sum(0)
                phys = phys_acc / LIKPF_MSEED
            proj = _robfit_mod(mds, phys + z, _PDEG) - z
            out[tag + "_raw"] = phys - last_tvt
            out[tag + "_proj"] = proj - last_tvt
            if tag == "noc":
                diag["seed_std"] = float(np.mean(np.std(res, axis=0)))
                diag["eff"] = float(1.0 / np.sum(wts ** 2))
                diag["loglik"] = float(llm / E)
                diag["rough"] = float(np.mean(np.abs(np.diff(phys, 2)))) if E > 2 else 0.0
            # exp(MTP): hengck23 multi-modality insight. Split 48 filters into 2 modes by the largest gap -> expose each mode's
            #  trajectory (upper/lower candidates), mode separation, and lower-mode weight as features. Exposes modes instead of point averaging.
            if os.environ.get("MODES", "0") == "1" and tag == "noc":
                lev = res.mean(1)                         # Average level of each filter (S,).
                order = np.argsort(lev)
                cw = np.cumsum(wts[order])                # Weighted cumulative distribution -> split by median (each mode has roughly half the weight).
                cut = int(np.searchsorted(cw, 0.5 * cw[-1]))
                cut = min(max(cut, 0), len(order) - 2)
                lo_i = order[:cut + 1]; hi_i = order[cut + 1:]
                if len(lo_i) > 0 and len(hi_i) > 0:
                    wl = wts[lo_i] / (wts[lo_i].sum() + 1e-12); wh = wts[hi_i] / (wts[hi_i].sum() + 1e-12)
                    mlo = (wl[:, None] * res[lo_i]).sum(0); mhi = (wh[:, None] * res[hi_i]).sum(0)
                    wlo_tot = float(wts[lo_i].sum())
                else:
                    mlo = phys; mhi = phys; wlo_tot = 0.5
                extra["mode_lo_d"] = mlo - last_tvt
                extra["mode_hi_d"] = mhi - last_tvt
                _mode_sep = np.abs(mhi - mlo)
                extra["mode_sep"] = _mode_sep                    # Upper/lower mode separation (=strength of multi-modality).
                extra["mode_wlo"] = np.full(E, np.float32(wlo_tot))   # Lower-mode weight.
                extra["mode_cal_vs_lo"] = phys - mlo; extra["mode_cal_vs_hi"] = phys - mhi
                if os.environ.get("GRFEATS2", "0") == "1":       # exp152: derivative features of mode trajectories.
                    extra["mode_sep_slope"] = np.gradient(_mode_sep).astype(np.float32)
                    extra["mode_lo_slope"] = np.gradient(mlo.astype(float)).astype(np.float32)
                    extra["mode_hi_slope"] = np.gradient(mhi.astype(float)).astype(np.float32)
            # MODES extension (N_best/MDN): split 48 filters into K modes by weighted K-quantiles -> expose each mode's trajectory/weight.
            #  Generalization of proven 2-mode MODES (N_best exposure idea from GM paper 2402.06377). NMODES=K. Like MODES, noc only.
            if os.environ.get("MODESK", "0") == "1" and tag == "noc":
                Km = int(os.environ.get("NMODES", "3"))
                levk = res.mean(1); ordk = np.argsort(levk)
                cwk = np.cumsum(wts[ordk]); totk = cwk[-1] + 1e-12
                bnd = [0]
                for j in range(1, Km):
                    c = int(np.searchsorted(cwk, totk * j / Km))
                    bnd.append(min(max(c, bnd[-1] + 1), len(ordk)))
                bnd.append(len(ordk))
                cen = []
                for j in range(Km):
                    gi = ordk[bnd[j]:bnd[j + 1]]
                    if len(gi) > 0:
                        wk = wts[gi] / (wts[gi].sum() + 1e-12); mj = (wk[:, None] * res[gi]).sum(0); wj = float(wts[gi].sum())
                    else:
                        mj = phys; wj = 0.0
                    extra[f"mk{Km}_{j}_d"] = mj - last_tvt
                    extra[f"mk{Km}_{j}_w"] = np.full(E, np.float32(wj))
                    cen.append(mj)
                extra[f"mk{Km}_spread"] = np.abs(cen[-1] - cen[0])      # Separation between edge modes.
                extra[f"mk{Km}_cal_vs_mid"] = phys - cen[Km // 2]       # cal vs central mode.
                extra[f"mk{Km}_lohi"] = np.abs(cen[1] - cen[0]) - np.abs(cen[-1] - cen[-2]) if Km >= 3 else np.zeros(E)
                if os.environ.get("MODECONF", "0") == "1":
                    # Per-point GR matching among exposed modes: larger margin = mode choice is more identifiable.
                    gr_abs = []
                    grv_m = inp[2]
                    for mj in cen:
                        eg = np.interp(mj, tw_tvt, tw_gr)
                        gr_abs.append(np.abs(grv_m - eg).astype(np.float32))
                    GM = np.stack(gr_abs, axis=0)
                    ordm = np.argsort(GM, axis=0)
                    cols = np.arange(E)
                    best = GM[ordm[0], cols]
                    second = GM[ordm[1], cols] if Km > 1 else best
                    extra[f"mk{Km}_gr_minabs"] = best.astype(np.float32)
                    extra[f"mk{Km}_gr_margin"] = (second - best).astype(np.float32)
                    extra[f"mk{Km}_gr_best"] = ordm[0].astype(np.float32)
                    extra[f"mk{Km}_gr_range"] = (GM.max(axis=0) - GM.min(axis=0)).astype(np.float32)
            if MULTI:   # proj at 4 scales including s5 + inter-scale variance.
                proj_sc = [out[tag + "_proj"]]
                for sc in (3.0, 8.0, 12.0):
                    w = np.exp((ll - llm) / sc); w /= w.sum()
                    pr = _robfit_mod(mds, (w[:, None] * res).sum(0) + z, _PDEG) - z
                    extra[f"{tag}_proj_s{int(sc)}"] = pr - last_tvt
                    proj_sc.append(pr - last_tvt)
                extra[tag + "_ms_std"] = np.std(np.stack(proj_sc, 0), axis=0)
            if POST and tag == "noc":   # Weighted quantiles (multi-modality).
                o = np.argsort(res, axis=0); vs = np.take_along_axis(res, o, 0)
                cw = np.cumsum(wts[o], axis=0); cw /= cw[-1:]
                Q = np.empty((E, 4))
                for ii in range(E):
                    Q[ii] = np.interp([0.1, 0.25, 0.75, 0.9], cw[:, ii], vs[:, ii])
                extra["q10"] = Q[:, 0] - last_tvt; extra["q25"] = Q[:, 1] - last_tvt
                extra["q75"] = Q[:, 2] - last_tvt; extra["q90"] = Q[:, 3] - last_tvt
        # eval-cal (ES-MDA iterative calibration): extend GR calibration into eval region using cal predictions -> affine refit -> rerun PF (K iterations, lambda).
        #  OOF improved in exp090 but LB worsened (overfitting from self-reinforcing loop). Revalidate on clean DET=0+MODES base this time (EVALCAL=1).
        if os.environ.get("EVALCAL", "0") == "1" and inp_cal is not None:
            eck = int(os.environ.get("EVALCAL_K", "2")); eclam = float(os.environ.get("EVALCAL_LAM", "1.0"))
            ec_kgr = kn["GR"].to_numpy(float)
            ec_tw_kn = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
            ec_grf = hw["GR"].interpolate(limit_direction="both").fillna(np.mean(tw_gr)).to_numpy(float)
            ec_egr = ec_grf[ev.index.to_numpy()]
            ec_sweep = os.environ.get("EVALCAL_SWEEP", "0") == "1"   # Record K=1..eck at each iteration (for K tuning).
            ec_inp = list(inp_cal)
            ec_a0, ec_b0 = float(ec_inp[6]), float(ec_inp[7])   # Calibrate using known points only (tempering anchor).
            ec_phys = np.asarray(out["cal_raw"], float) + last_tvt   # cal points (absolute TVT) iter0.
            cal_rel = np.asarray(out["cal_proj"], float)
            for kk in range(eck):
                ec_tw_ev = np.interp(ec_phys, tw_tvt, tw_gr)
                a_ec, b_ec = _affine_cal(np.concatenate([ec_kgr, ec_egr]),
                                         np.concatenate([ec_tw_kn, ec_tw_ev]))
                a_ec = 1.0 + _PSHRINK * (a_ec - 1.0); b_ec = _PSHRINK * b_ec
                ec_inp[6] = (1.0 - eclam) * ec_a0 + eclam * a_ec
                ec_inp[7] = (1.0 - eclam) * ec_b0 + eclam * b_ec
                ec_res, ec_ll = _phys_pf(*ec_inp, _PNS, _PNP, 42, sd, kappa, liktype, likp)
                ec_w = np.exp((ec_ll - ec_ll.max()) / _PSCALE); ec_w /= ec_w.sum()
                ec_phys = (ec_w[:, None] * ec_res).sum(0)
                if ec_sweep:   # Record proj at K=kk+1.
                    pr = (_robfit_mod(mds, ec_phys + z, _PDEG) - z) - last_tvt
                    extra[f"ec_proj_d_k{kk + 1}"] = pr.astype(np.float32)
                    extra[f"ec_vs_cal_k{kk + 1}"] = (pr - cal_rel).astype(np.float32)
            ec_proj_rel = (_robfit_mod(mds, ec_phys + z, _PDEG) - z) - last_tvt
            if not ec_sweep:
                extra["ec_proj_d"] = ec_proj_rel.astype(np.float32)
                extra["ec_vs_cal"] = (ec_proj_rel - cal_rel).astype(np.float32)
        # exp132 (patent US11480045): featureize Pearson big-segment geosteering as a new TVT estimator.
        if os.environ.get("GEOPEARSON", "0") == "1" and inp_noc is not None:
            md_v, z_v, gr_v, grid_gr, g0, dgr = inp_noc[0], inp_noc[1], inp_noc[2], inp_noc[3], inp_noc[4], inp_noc[5]
            last_U2, last_MD2 = float(inp_noc[10]), float(inp_noc[11])
            kmd = kn["MD"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float); kz = kn["Z"].to_numpy(float)
            reg_dip = float(robust_slope(kmd, ktvt + kz)) if len(kn) >= 2 else 0.0
            SEG = int(os.environ.get("GEO_SEG", "150")); NDIP = int(os.environ.get("GEO_NDIP", "41"))
            SPAN = float(os.environ.get("GEO_SPAN", "0.05"))
            geo_tvt, geo_conf = _geo_pearson(np.ascontiguousarray(md_v), np.ascontiguousarray(z_v),
                                             np.ascontiguousarray(gr_v), np.ascontiguousarray(grid_gr),
                                             float(g0), float(dgr), last_U2, last_MD2, float(last_tvt),
                                             reg_dip, SEG, NDIP, SPAN, 8.0, 4.0)
            extra["geo_proj_d"] = (geo_tvt - last_tvt).astype(np.float32)
            extra["geo_conf"] = geo_conf.astype(np.float32)
            extra["geo_vs_cal"] = ((geo_tvt - last_tvt) - out["cal_proj"]).astype(np.float32)
        # exp135 Phase4: global decoding (Viterbi). Uses cal inputs (a,b,gs). MAP trajectory to avoid greedy drift.
        if os.environ.get("GLOBALDECODE", "0") == "1" and inp_cal is not None:
            md_v, z_v, gr_v, grid_gr, g0, dgr = inp_cal[0], inp_cal[1], inp_cal[2], inp_cal[3], inp_cal[4], inp_cal[5]
            a_, b_, gs_ = float(inp_cal[6]), float(inp_cal[7]), float(inp_cal[8])
            last_U2, last_MD2 = float(inp_cal[10]), float(inp_cal[11])
            kmd = kn["MD"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float); kz = kn["Z"].to_numpy(float)
            reg_dip = float(robust_slope(kmd, ktvt + kz)) if len(kn) >= 2 else 0.0
            NU = int(os.environ.get("GD_NU", "121")); DU = float(os.environ.get("GD_DU", "1.0"))
            WB = float(os.environ.get("GD_WBETA", "30.0")); BAND = int(os.environ.get("GD_BAND", "8"))
            gd_tvt = _global_decode(np.ascontiguousarray(md_v), np.ascontiguousarray(z_v),
                                    np.ascontiguousarray(gr_v), np.ascontiguousarray(grid_gr),
                                    float(g0), float(dgr), a_, b_, gs_, last_U2, last_MD2,
                                    float(last_tvt), reg_dip, NU, DU, WB, BAND)
            extra["gdec_proj_d"] = (gd_tvt - last_tvt).astype(np.float32)
            extra["gdec_vs_cal"] = ((gd_tvt - last_tvt) - out["cal_proj"]).astype(np.float32)
        # exp133: lik-PF dip(rate) readout. cal_proj can follow shape (corr 0.91), but drift-rate is off (50% error) ->
        #  featureize apparent dip d(cal)/dMD, regional-dip deviation, curvature, and cumulative drift to pass drift-correction information to GBDT.
        if os.environ.get("RATEFEAT", "0") == "1":
            cal = np.asarray(out["cal_proj"], float)
            dmd = np.gradient(md); dmd[np.abs(dmd) < 1e-6] = 1.0
            dip = np.gradient(cal) / dmd                       # Apparent TVT dip.
            dip_s = pd.Series(dip).rolling(21, center=True, min_periods=1).mean().to_numpy()
            kmd = kn["MD"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float)
            reg_tvt_dip = float(robust_slope(kmd, ktvt)) if len(kn) >= 2 else 0.0
            extra["cal_dip"] = dip_s.astype(np.float32)
            extra["cal_dip_dev"] = (dip_s - reg_tvt_dip).astype(np.float32)
            extra["cal_curv"] = np.gradient(dip_s).astype(np.float32)
            extra["cal_cumdev"] = np.cumsum((dip_s - reg_tvt_dip) * dmd).astype(np.float32)
        # exp134 geosteering spectrum (patent Fig7-8): featureize dip ambiguity/multi-modality/confidence.
        if os.environ.get("GEOSPECTRUM", "0") == "1" and inp_noc is not None:
            md_v, z_v, gr_v, grid_gr, g0, dgr = inp_noc[0], inp_noc[1], inp_noc[2], inp_noc[3], inp_noc[4], inp_noc[5]
            last_U2, last_MD2 = float(inp_noc[10]), float(inp_noc[11])
            kmd = kn["MD"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float); kz = kn["Z"].to_numpy(float)
            reg_dip = float(robust_slope(kmd, ktvt + kz)) if len(kn) >= 2 else 0.0
            WIN = int(os.environ.get("SPEC_WIN", "30")); NDIP = int(os.environ.get("SPEC_NDIP", "21"))
            SPAN = float(os.environ.get("SPEC_SPAN", "0.06"))
            sp_pk, sp_spr, sp_nm = _geo_spectrum(np.ascontiguousarray(md_v), np.ascontiguousarray(z_v),
                                                 np.ascontiguousarray(gr_v), np.ascontiguousarray(grid_gr),
                                                 float(g0), float(dgr), last_U2, last_MD2, reg_dip, WIN, NDIP, SPAN)
            extra["spec_peak"] = sp_pk.astype(np.float32)
            extra["spec_spread"] = sp_spr.astype(np.float32)
            extra["spec_nmax"] = sp_nm.astype(np.float32)
        # exp136 SELFCORR (patent SC): local periodicity (near-lag autocorrelation). High = ambiguous match due to pattern repetition.
        if os.environ.get("SELFCORR", "0") == "1" and inp_noc is not None:
            gr_v = np.ascontiguousarray(inp_noc[2])
            sc = _autocorr_feat(gr_v, int(os.environ.get("SC_WIN", "25")), 10, 60)
            extra["selfcorr"] = sc.astype(np.float32)
        # exp137 REPSECTION (patent Sigma Delta gamma): far-lag autocorrelation = repeated-layer detection from faults.
        if os.environ.get("REPSECTION", "0") == "1" and inp_noc is not None:
            gr_v = np.ascontiguousarray(inp_noc[2])
            rp = _autocorr_feat(gr_v, int(os.environ.get("RS_WIN", "30")), 100, 400)
            extra["repsection"] = rp.astype(np.float32)
        df = pd.DataFrame({"id": [f"{wid}_{i}" for i in ev.index],
                           "cal_proj_d": out["cal_proj"], "noc_proj_d": out["noc_proj"],
                           "cal_raw_d": out["cal_raw"], "noc_raw_d": out["noc_raw"]})
        df["cn_proj"] = df["cal_proj_d"] - df["noc_proj_d"]
        df["cn_raw"] = df["cal_raw_d"] - df["noc_raw_d"]
        for kk, vv in diag.items():
            df[kk] = vv
        for kk, vv in extra.items():   # Append at the tail only when toggle is ON (fixed column-order rule).
            df[kk] = vv
        return df
    except Exception as e:  # noqa: BLE001
        print(f"genphys WARN {hw_path}: {e}", flush=True)
        return None


def _sdip_one(hw_path, DF, is_train):
    """For spatial-dip fusion: compute expected dU/dMD (=rate prior) for each eval point from neighboring wells using DipField."""
    import os as _os
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return wid, None
    swid = wid if is_train else None
    lk = kn.iloc[-1]
    Xe = ev["X"].to_numpy(float); Ye = ev["Y"].to_numpy(float); md = ev["MD"].to_numpy(float)
    a, b, _ = DF.gradient(np.column_stack([Xe, Ye]), self_wid=swid)
    px = np.concatenate([[float(lk["X"])], Xe]); py = np.concatenate([[float(lk["Y"])], Ye])
    mdp = np.concatenate([[float(lk["MD"])], md])
    dx = np.diff(px); dy = np.diff(py); dm = np.diff(mdp); dm[dm < 1e-6] = 1.0
    return wid, ((a * dx + b * dy) / dm).astype(np.float32)



def _add_modeconf_features(df):
    """train19: candidate confidence features built after base+phys merge.

    Adds light-weight summaries only. No new physics run, no geoaux.
    These features let LGB decide whether MODESK/PF/beam candidates agree and
    whether local GR matching around PF/beam/SC is sharp or ambiguous.
    """
    if os.environ.get("MODECONF", "0") != "1":
        return df
    Km = int(os.environ.get("NMODES", "3"))
    added = []

    # ---- MODESK agreement / gap features ----
    mcols = [f"mk{Km}_{j}_d" for j in range(Km)]
    if all(c in df.columns for c in mcols):
        M = df[mcols].to_numpy(np.float32)
        m_mean = M.mean(axis=1)
        m_std = M.std(axis=1)
        m_min = M.min(axis=1)
        m_max = M.max(axis=1)
        m_med = np.median(M, axis=1)
        df[f"mk{Km}_mean_d"] = m_mean.astype(np.float32); added.append(f"mk{Km}_mean_d")
        df[f"mk{Km}_median_d"] = m_med.astype(np.float32); added.append(f"mk{Km}_median_d")
        df[f"mk{Km}_std_d"] = m_std.astype(np.float32); added.append(f"mk{Km}_std_d")
        df[f"mk{Km}_range_d"] = (m_max - m_min).astype(np.float32); added.append(f"mk{Km}_range_d")
        df[f"mk{Km}_mean_vs_mid"] = (m_mean - M[:, Km // 2]).astype(np.float32); added.append(f"mk{Km}_mean_vs_mid")

        # Pairwise gaps: explicit for K=3/4, deterministic for any K.
        gaps = []
        for a in range(Km):
            for b in range(a + 1, Km):
                g = np.abs(M[:, b] - M[:, a]).astype(np.float32)
                df[f"mk{Km}_gap{a}{b}"] = g; added.append(f"mk{Km}_gap{a}{b}")
                gaps.append(g)
        if gaps:
            G = np.stack(gaps, axis=1)
            df[f"mk{Km}_gap_min"] = G.min(axis=1).astype(np.float32); added.append(f"mk{Km}_gap_min")
            df[f"mk{Km}_gap_max"] = G.max(axis=1).astype(np.float32); added.append(f"mk{Km}_gap_max")
            df[f"mk{Km}_gap_std"] = G.std(axis=1).astype(np.float32); added.append(f"mk{Km}_gap_std")

        # Mode weights: if one mode dominates, candidates are less ambiguous.
        wcols = [f"mk{Km}_{j}_w" for j in range(Km)]
        if all(c in df.columns for c in wcols):
            W = df[wcols].to_numpy(np.float32)
            df[f"mk{Km}_w_max"] = W.max(axis=1).astype(np.float32); added.append(f"mk{Km}_w_max")
            df[f"mk{Km}_w_min"] = W.min(axis=1).astype(np.float32); added.append(f"mk{Km}_w_min")
            df[f"mk{Km}_w_range"] = (W.max(axis=1) - W.min(axis=1)).astype(np.float32); added.append(f"mk{Km}_w_range")
            df[f"mk{Km}_w_std"] = W.std(axis=1).astype(np.float32); added.append(f"mk{Km}_w_std")

        # Candidate-vs-candidate disagreement. These are the key train19 features.
        refs = [
            ("pmf_fwd_d", "pmf"),
            ("pf_ancc_d", "pfancc"),
            ("pf_z_d", "pfz"),
            ("beam_mean_d", "beammean"),
            ("beam_med_d", "beammed"),
            ("hyb_d", "hyb"),
            ("cal_proj_d", "calproj"),
            ("noc_proj_d", "nocproj"),
        ]
        for rc, name in refs:
            if rc in df.columns:
                r = df[rc].to_numpy(np.float32)
                df[f"mk{Km}_mean_vs_{name}"] = (m_mean - r).astype(np.float32); added.append(f"mk{Km}_mean_vs_{name}")
                df[f"mk{Km}_med_vs_{name}"] = (m_med - r).astype(np.float32); added.append(f"mk{Km}_med_vs_{name}")
                # distance from reference to nearest exposed mode: how well that candidate is supported by modes.
                df[f"mk{Km}_{name}_nearest_abs"] = np.min(np.abs(M - r[:, None]), axis=1).astype(np.float32); added.append(f"mk{Km}_{name}_nearest_abs")

    # ---- GR matching margin features from existing offset residual curves ----
    def _gr_margin(prefix, offs):
        cols = [f"{prefix}{int(o)}" for o in offs]
        if not all(c in df.columns for c in cols):
            return
        A = np.abs(df[cols].to_numpy(np.float32))
        order = np.argsort(A, axis=1)
        rows = np.arange(A.shape[0])
        best = A[rows, order[:, 0]]
        second = A[rows, order[:, 1]] if A.shape[1] > 1 else best
        med = np.median(A, axis=1)
        offs_arr = np.asarray(offs, np.float32)
        df[f"{prefix}_gr_minabs"] = best.astype(np.float32); added.append(f"{prefix}_gr_minabs")
        df[f"{prefix}_gr_margin"] = (second - best).astype(np.float32); added.append(f"{prefix}_gr_margin")
        df[f"{prefix}_gr_sharp"] = (med - best).astype(np.float32); added.append(f"{prefix}_gr_sharp")
        df[f"{prefix}_gr_argmin_off"] = offs_arr[order[:, 0]].astype(np.float32); added.append(f"{prefix}_gr_argmin_off")
        if 0 in [int(o) for o in offs]:
            zidx = [int(o) for o in offs].index(0)
            df[f"{prefix}_gr_center_abs"] = A[:, zidx].astype(np.float32); added.append(f"{prefix}_gr_center_abs")
            df[f"{prefix}_gr_center_minus_best"] = (A[:, zidx] - best).astype(np.float32); added.append(f"{prefix}_gr_center_minus_best")

    _gr_margin("tdpf", PF_OFFS)
    _gr_margin("tdbc", BEAM_OFFS)
    _gr_margin("tdsc", SC_OFFS)

    if added:
        print(f"  [modeconf] added {len(added)} features", flush=True)
    return df

def add_phys_feats(base_df, d, DF=None, is_train=False):
    from joblib import Parallel, delayed
    nj = int(os.environ.get("N_JOBS", "4"))   # Kaggle default=4 / local can set N_JOBS=24, etc. genphys_one is
    paths = sorted(Path(d).glob("*__horizontal_well.csv"))  # self-contained (seed42/call) -> independent of worker count (values unchanged).
    sdip_map = {}
    if os.environ.get("DIPFUSE", "0") == "1" and DF is not None:   # Spatial-dip fusion (breakthrough PoC).
        for p in paths:
            wid, sd = _sdip_one(str(p), DF, is_train)
            if sd is not None:
                sdip_map[wid] = sd
        print(f"  sdip computed for {len(sdip_map)} wells (DIPFUSE kappa={os.environ.get('DIPKAPPA','0.1')})", flush=True)

    def _w(p):
        return Path(p).stem.replace("__horizontal_well", "")
    tag = "train" if is_train else "test"
    t0 = time.time()
    exact_parallel = os.environ.get("FULL249_EXACT_PF_PARALLEL", "1") not in {"0", "false", "False"}
    exact_jobs = int(os.environ.get("FULL249_EXACT_PF_JOBS", str(nj)) or str(nj))
    unsupported = _exact_pf_parallel_supported() if exact_parallel else []
    if exact_parallel and not unsupported:
        print(
            f"  [{tag}] phys/lik-PF EXACT-SEED-PARALLEL start: {len(paths)} wells, "
            f"n_jobs={exact_jobs}, PNS={_PNS}, PNP={_PNP}, MSEED={os.environ.get('LIKPF_MSEED','1')}",
            flush=True,
        )
        try:
            parts = _genphys_exact_parallel(paths, sdip_map, exact_jobs)
        except Exception as exc:
            print(f"  [{tag}] exact PF scheduler failed ({exc!r}); fallback per-well", flush=True)
            parts = Parallel(n_jobs=nj, prefer="processes")(
                delayed(genphys_one)(str(p), sdip_map.get(_w(p))) for p in paths)
    else:
        if unsupported:
            print(f"  [{tag}] exact PF unsupported flags={unsupported}; fallback per-well", flush=True)
        print(f"  [{tag}] phys/lik-PF features start: {len(paths)} wells, n_jobs={nj}, PNS={_PNS}, PNP={_PNP}", flush=True)
        parts = Parallel(n_jobs=nj, prefer="processes")(
            delayed(genphys_one)(str(p), sdip_map.get(_w(p))) for p in paths)
    pf = pd.concat([p for p in parts if p is not None], ignore_index=True)
    merged = base_df.merge(pf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    before_cols = merged.shape[1]
    merged = _add_modeconf_features(merged)
    modeconf_cols = merged.shape[1] - before_cols
    print(f"  [{tag}] phys/lik-PF done: shape={merged.shape} (+{pf.shape[1]-1} phys cols, +{modeconf_cols} modeconf cols, {time.time() - t0:.0f}s)", flush=True)
    return merged


# ===== Self-contained reproducible baseline: MODE=train (train -> save model) / infer (load model -> pp -> submit) =====
# No external libs (all feature code inline). PF seeds are fixed (_pf_ancc/_pf_z seed7), so split train==infer matches.
import glob
import hashlib
import json

import joblib

MODE = os.environ.get("MODE", "infer")
ART = Path(os.environ.get("ART_DIR", str(Path(__file__).resolve().parent / "artifacts_repro")))


def _feat_hash(features):
    """Hash feature column names + order (used to verify train/infer column-order consistency)."""
    return hashlib.md5(",".join(features).encode()).hexdigest()[:8]


def _build_imputers(t0):
    """Build imputers from all training wells (needed in both train/infer modes. FI/DI are global)."""
    global FI, DI
    train_wids = [p.stem.replace("__horizontal_well", "")
                  for p in sorted((DATA / "train").glob("*__horizontal_well.csv"))]
    print(f"imputers... ({len(train_wids)} wells)", flush=True)
    FI = FormationPlaneKNN(train_wids, DATA / "train")
    DI = DenseANCCImputer(train_wids, DATA / "train")
    SI = SurfaceImputer(train_wids, DATA / "train")
    DFIELD = DipField(train_wids, DATA / "train")
    print(f"imputers done ({time.time() - t0:.0f}s)", flush=True)
    return SI, DFIELD


def _build_full(data_dir, is_train, SI, DFIELD, t0, tag):
    print(f"[{tag}] full feature build START ({data_dir})", flush=True)
    df = build_all(data_dir, is_train, SI, DFIELD)
    print(f"[{tag}] full feature build: entering phys stage after {time.time() - t0:.0f}s", flush=True)
    df = add_phys_feats(df, data_dir, DFIELD, is_train)
    print(f"[{tag}] full feature build DONE: {df.shape} ({time.time() - t0:.0f}s)", flush=True)
    return df


def _robust_polyfit(x, v, deg=4, n_iter=5, c=1.5):
    if len(v) <= deg + 1:
        return v.copy()
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    A = np.vander(xn, deg + 1)
    w = np.ones(len(v))
    for _ in range(n_iter):
        co, *_ = np.linalg.lstsq(A * w[:, None], v * w, rcond=None)
        r = v - A @ co
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
    return A @ co


def _apply_pp(test_df, test_pred):
    """Projection PP: robust deg4 projection on U=tvt+Z -> blend pf_ancc 0.05 -> tau85 damping."""
    md_t = test_df["md_since"].to_numpy(float)
    pf_t = test_df["pf_ancc_d"].to_numpy(float)
    _zmap = {}
    for _p in sorted((DATA / "test").glob("*__horizontal_well.csv")):
        _wid = _p.stem.replace("__horizontal_well", "")
        _df = pd.read_csv(_p, usecols=["Z", "TVT_input"])
        _ev = _df[_df["TVT_input"].isna()]
        for _i, _z in zip(_ev.index, _ev["Z"].to_numpy(float)):
            _zmap[f"{_wid}_{_i}"] = _z
    Z_t = test_df["id"].map(_zmap).to_numpy(float)
    d_t = test_pred.copy()
    _wt = test_df["well"].to_numpy()
    for _, _gi in pd.DataFrame({"w": _wt}).groupby("w", sort=False).indices.items():
        if np.isnan(Z_t[_gi]).any():
            continue
        d_t[_gi] = _robust_polyfit(md_t[_gi], d_t[_gi] + Z_t[_gi]) - Z_t[_gi]
    d_t = d_t * (1 - 0.05) + pf_t * 0.05
    d_t = d_t * (1.0 - np.exp(-np.maximum(md_t, 0.0) / 85.0))
    # exp154: global linear recalibration after _apply_pp (correction for underestimated dip). Manifest coefficients; default no-op.
    _ra = float(os.environ.get("RECAL_A", "1.0")); _rb = float(os.environ.get("RECAL_B", "0.0"))
    if _ra != 1.0 or _rb != 0.0:
        d_t = _ra * d_t + _rb
        print(f"RECAL applied: a={_ra} b={_rb}", flush=True)
    return test_df.assign(tvt=test_df["last_tvt"].to_numpy() + d_t)


def run_train():
    """Train a train19-compatible XGBoost model.

    Why this is not a blind LightGBM -> XGBoost rename:
    - XGBoost uses max_depth/min_child_weight instead of num_leaves/min_child_samples.
    - XGBoost supports np.nan natively, but inf values must be converted to nan.
    - GPU parameters differ across xgboost versions; this function tries the modern
      tree_method='hist', device='cuda' path first and falls back to gpu_hist/CPU.
    - Early stopping / best_iteration handling differs, so OOF prediction uses
      iteration_range when available.
    """
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor
    except Exception as e:
        raise RuntimeError("xgboost is required. Try: pip install xgboost") from e
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import GroupKFold

    _seed_numba(SEED); np.random.seed(SEED); t0 = time.time()
    SI, DFIELD = _build_imputers(t0)
    _LOAD_X = os.environ.get("LOAD_X")
    if _LOAD_X:
        lp = Path(_LOAD_X) / "Xy.parquet"
        print(f"LOAD_X: loading from {lp}", flush=True)
        train_df = pd.read_parquet(lp)
        print(f"LOAD_X: loaded {train_df.shape}", flush=True)
    else:
        print("build train features...", flush=True)
        train_df = _build_full(DATA / "train", True, SI, DFIELD, t0, "train")

    features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    if os.environ.get("FEAT_SUBSET"):
        keep = json.load(open(os.environ["FEAT_SUBSET"]))
        miss = [f for f in keep if f not in train_df.columns]
        assert not miss, f"FEAT_SUBSET features missing from built columns: {miss}"
        features = list(keep)
        print(f"FEAT_SUBSET: {os.environ['FEAT_SUBSET']} -> {len(features)} feats", flush=True)

    fh = _feat_hash(features)
    print(f"feats={len(features)} feat_hash={fh}", flush=True)

    # XGBoost-specific sanitization. XGB handles NaN, but +/-inf can break hist bins.
    X = train_df[features].astype(np.float32, copy=False)
    X = X.replace([np.inf, -np.inf], np.nan)
    y = train_df["target"].astype(np.float32)
    g = train_df["well"]

    if os.environ.get("MAKE_OOF_DETAIL", "0") == "1":
        oof_path = ART / "oof.npy"
        assert oof_path.exists(), f"missing {oof_path}"
        oof = np.load(oof_path)
        assert len(oof) == len(train_df), f"oof len {len(oof)} != train_df len {len(train_df)}"
        pd.DataFrame({"id": train_df["id"].values, "well": train_df["well"].values,
                      "target": train_df["target"].values, "oof": oof}).to_csv(ART / "oof_detail.csv", index=False)
        print(f"saved oof_detail.csv -> {ART / 'oof_detail.csv'}", flush=True)
        raise SystemExit

    print(f"train matrix ready: X={X.shape}, y={y.shape}, wells={g.nunique()}, N_EST={N_EST}, N_SPLITS={N_SPLITS}", flush=True)
    vhash = hashlib.md5(np.ascontiguousarray(X.to_numpy(np.float32)).tobytes()).hexdigest()[:8]
    print(f"value_hash={vhash} (feature-value consistency check)", flush=True)
    if os.environ.get("SAVE_X"):
        cd = Path(os.environ["SAVE_X"]); cd.mkdir(parents=True, exist_ok=True)
        train_df[features + ["target", "well", "id"]].to_parquet(cd / "Xy.parquet")
        json.dump(features, open(cd / "features.json", "w"))
        print(f"SAVE_X: cached {train_df.shape} -> {cd}", flush=True)

    _LOSS = os.environ.get("LOSS", "regression").lower()
    objective = "reg:squarederror"
    if _LOSS in {"huber", "pseudohuber", "pseudo_huber"}:
        # XGBoost pseudo-Huber is robust but does not use LightGBM's alpha in the same way.
        objective = "reg:pseudohubererror"

    _TRAIN_XGB_DEVICE = os.environ.get("TRAIN_XGB_DEVICE", "cpu").strip().lower()
    _USE_XGB_GPU = _TRAIN_XGB_DEVICE in {"gpu", "cuda"}
    _EARLY = int(os.environ.get("EARLY_STOPPING", "300"))
    _MSEEDS = int(os.environ.get("MODEL_SEEDS", "1"))
    _NTHREAD = int(os.environ.get("XGB_NTHREAD", str(max(1, os.cpu_count() or 1))))

    base_params = dict(
        objective=objective,
        n_estimators=N_EST,
        learning_rate=float(os.environ.get("XGB_LR", "0.02")),
        max_depth=int(os.environ.get("XGB_MAX_DEPTH", "8")),
        min_child_weight=float(os.environ.get("XGB_MIN_CHILD_WEIGHT", "80")),
        subsample=float(os.environ.get("SUBSAMPLE", "0.8")),
        colsample_bytree=float(os.environ.get("COLSAMPLE", "0.6")),
        reg_lambda=float(os.environ.get("REG_LAMBDA", "10.0")),
        reg_alpha=float(os.environ.get("REG_ALPHA", "1.0")),
        max_bin=int(os.environ.get("XGB_MAX_BIN", "256")),
        grow_policy=os.environ.get("XGB_GROW_POLICY", "depthwise"),
        random_state=SEED,
        n_jobs=_NTHREAD,
        eval_metric="rmse",
        missing=np.nan,
        verbosity=1,
    )
    max_leaves = int(os.environ.get("XGB_MAX_LEAVES", "0"))
    if max_leaves > 0:
        base_params["max_leaves"] = max_leaves
    # Optional regularizers that often matter more in XGB than LGB.
    if os.environ.get("XGB_GAMMA") is not None:
        base_params["gamma"] = float(os.environ["XGB_GAMMA"])
    if os.environ.get("XGB_MAX_DELTA_STEP") is not None:
        base_params["max_delta_step"] = float(os.environ["XGB_MAX_DELTA_STEP"])

    if _USE_XGB_GPU:
        # Modern XGBoost: tree_method=hist + device=cuda. If unsupported, fallback below.
        base_params.update(tree_method="hist", device="cuda")
        print("XGBoost training device: cuda/hist", flush=True)
    else:
        base_params.update(tree_method="hist", device="cpu")
        print("XGBoost training device: cpu/hist", flush=True)
    print(f"XGB objective={objective}; params={base_params}", flush=True)

    def _fit_one(model, X_tr, y_tr, X_va, y_va):
        """Version-tolerant fit with early stopping."""
        try:
            model.set_params(early_stopping_rounds=_EARLY)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except TypeError:
            # Older sklearn wrapper accepts early_stopping_rounds in fit.
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=_EARLY)
        return model

    def _predict_best(model, X_part):
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            try:
                return model.predict(X_part, iteration_range=(0, int(best_iter) + 1))
            except Exception:
                pass
        return model.predict(X_part)

    cv = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(X), dtype=np.float64)
    fold_rows = []
    (ART / "models").mkdir(parents=True, exist_ok=True)

    for fold, (tr, va) in enumerate(cv.split(X, y, g)):
        print(f"[xgb train] fold {fold} START: train_rows={len(tr)} valid_rows={len(va)} ({time.time() - t0:.0f}s)", flush=True)
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]
        va_pred = np.zeros(len(va), dtype=np.float64)
        for sx in range(_MSEEDS):
            p2 = dict(base_params)
            p2["random_state"] = SEED + sx
            print(f"[xgb train] fold {fold} seed_index={sx} fitting n_estimators={N_EST} device={p2.get('device')}...", flush=True)
            model = XGBRegressor(**p2)
            try:
                model = _fit_one(model, X_tr, y_tr, X_va, y_va)
            except Exception as e:
                if _USE_XGB_GPU and os.environ.get("XGB_GPU_FALLBACK", "1") == "1":
                    print(f"[WARN] XGBoost GPU failed fold={fold} seed_index={sx}: {type(e).__name__}: {e}", flush=True)
                    print("[WARN] Retrying this model on CPU/hist. Set XGB_GPU_FALLBACK=0 to fail instead.", flush=True)
                    p2.pop("device", None)
                    p2["tree_method"] = "hist"
                    p2["device"] = "cpu"
                    model = XGBRegressor(**p2)
                    model = _fit_one(model, X_tr, y_tr, X_va, y_va)
                else:
                    raise
            best_iter = getattr(model, "best_iteration", None)
            best_score = getattr(model, "best_score", None)
            print(f"[xgb train] fold {fold} seed_index={sx} fit done; best_iteration={best_iter} best_score={best_score} ({time.time() - t0:.0f}s)", flush=True)
            va_pred += _predict_best(model, X_va) / _MSEEDS
            model_path = ART / "models" / f"fold{fold}_s{sx}.pkl"
            joblib.dump(model, model_path)
            print(f"[xgb train] saved {model_path}", flush=True)
            try:
                score = model.get_booster().get_score(importance_type="gain")
                json.dump(score, open(ART / "models" / f"fold{fold}_s{sx}_gain.json", "w"), indent=2)
            except Exception as _e:
                print(f"[WARN] feature importance save failed: {_e}", flush=True)
        oof[va] = va_pred
        frmse = root_mean_squared_error(y_va, oof[va])
        fold_rows.append({"fold": fold, "rmse": float(frmse), "valid_rows": int(len(va)), "train_rows": int(len(tr))})
        print(f"fold {fold}: rmse={frmse:.4f} ({_MSEEDS} seeds, {time.time() - t0:.0f}s)", flush=True)

    oof_rmse = float(root_mean_squared_error(y, oof))
    json.dump(features, open(ART / "features.json", "w"))
    pd.DataFrame(fold_rows).to_csv(ART / "fold_metrics.csv", index=False)

    cfg = {"model_type": "xgboost",
           "xgboost_version": getattr(xgb, "__version__", "unknown"),
           "xgb_objective": objective,
           "xgb_params": base_params,
           "n_splits": N_SPLITS,
           "model_seeds": _MSEEDS,
           "early_stopping_rounds": _EARLY,
           "pf_mseed": len(_PF_SEEDS),
           "multiscale": int(os.environ.get("MULTISCALE", "0")),
           "posterior": int(os.environ.get("POSTERIOR", "0")),
           "pns": _PNS, "pnp": _PNP,
           "dipfuse": int(os.environ.get("DIPFUSE", "0")),
           "dipkappa": float(os.environ.get("DIPKAPPA", "0.1")),
           "liktype": os.environ.get("LIKTYPE", "gauss"),
           "likp": float(os.environ.get("LIKP", "4.0")),
           "traj": int(os.environ.get("TRAJ", "0")),
           "linmdz": int(os.environ.get("LINMDZ", "0")),
           "tvtreg": float(os.environ.get("TVTREG", "0")),
           "likpf_mseed": int(os.environ.get("LIKPF_MSEED", "1")),
           "geopearson": int(os.environ.get("GEOPEARSON", "0")),
           "geo_seg": int(os.environ.get("GEO_SEG", "150")), "geo_span": float(os.environ.get("GEO_SPAN", "0.05")),
           "globaldecode": int(os.environ.get("GLOBALDECODE", "0")),
           "ratefeat": int(os.environ.get("RATEFEAT", "0")),
           "geospectrum": int(os.environ.get("GEOSPECTRUM", "0")),
           "selfcorr": int(os.environ.get("SELFCORR", "0")),
           "repsection": int(os.environ.get("REPSECTION", "0")),
           "modes": int(os.environ.get("MODES", "0")),
           "evalcal": int(os.environ.get("EVALCAL", "0")),
           "evalcal_k": int(os.environ.get("EVALCAL_K", "2")),
           "evalcal_lam": float(os.environ.get("EVALCAL_LAM", "1.0")),
           "grfill": os.environ.get("GRFILL", "interp"),
           "modesk": int(os.environ.get("MODESK", "0")),
           "nmodes": int(os.environ.get("NMODES", "3")),
           "modeconf": int(os.environ.get("MODECONF", "0")),
           "grfeats2": int(os.environ.get("GRFEATS2", "0")),
           "pmf": int(os.environ.get("PMF", "0")),
           "loss": _LOSS}
    json.dump({"feat_hash": fh, "oof_rmse": oof_rmse, "n_feats": len(features), **cfg},
              open(ART / "manifest.json", "w"), indent=2)
    print(f"cfg={cfg}", flush=True)
    np.save(ART / "oof.npy", oof)
    pd.DataFrame({"id": train_df["id"].values,
                  "well": train_df["well"].values,
                  "target": y.values,
                  "oof": oof}).to_csv(ART / "oof_detail.csv", index=False)
    print(f"OOF RMSE (raw): {oof_rmse:.4f}  feat_hash={fh}  saved->{ART}", flush=True)
    print(f"saved oof_detail.csv -> {ART / 'oof_detail.csv'}", flush=True)


def run_infer():
    t0 = time.time()
    # Look for model artifacts (Kaggle: dataset mount / local: ART).
    # IMPORTANT: prefer explicit artifact path. Otherwise glob may pick a wrong model dataset.
    art = None
    _force_art = (os.environ.get("FORCE_ART_DIR", "").strip()
                  or os.environ.get("ART_DIR", "").strip())
    if _force_art:
        _p = Path(_force_art)
        if (_p / "features.json").exists() and (_p / "models").exists() and (_p / "manifest.json").exists():
            art = _p
            print(f"FORCE_ART_DIR/ART_DIR: using {art}", flush=True)
        else:
            print(f"WARN: requested artifact path invalid: {_p}", flush=True)
    if art is None:
        target_name = os.environ.get("TARGET_ART_NAME", "exp252_xgb_t19_modeconf_grmargin")
        for fj in glob.glob("/kaggle/input/**/features.json", recursive=True):
            p = Path(fj).parent
            if p.name == target_name and (p / "models").exists() and (p / "manifest.json").exists():
                art = p; break
    if art is None:
        for fj in glob.glob("/kaggle/input/**/features.json", recursive=True):
            # primary is a dir with manifest.json (exclude blend subdirs without manifest).
            if (Path(fj).parent / "models").exists() and (Path(fj).parent / "manifest.json").exists():
                art = Path(fj).parent; break
    if art is None and (ART / "features.json").exists():
        art = ART
    assert art is not None, "train artifacts (features.json + models/) not found"
    features = json.load(open(art / "features.json"))
    models = sorted((art / "models").glob("fold*.pkl"))
    assert models, "no fold*.pkl"
    # Read train-time config (pf_mseed/multiscale/posterior) from manifest and apply it to test feature generation (train==infer).
    global _PF_SEEDS, _PNS, _PNP
    mani = json.load(open(art / "manifest.json")) if (art / "manifest.json").exists() else {}
    M = int(mani.get("pf_mseed", 1)); _PF_SEEDS = [42 + i for i in range(M)]
    os.environ["MULTISCALE"] = str(int(mani.get("multiscale", 0)))
    os.environ["POSTERIOR"] = str(int(mani.get("posterior", 0)))
    _PNS = int(mani.get("pns", _PNS)); _PNP = int(mani.get("pnp", _PNP))
    os.environ["PNS"] = str(_PNS); os.environ["PNP"] = str(_PNP)   # also for worker(spawn re-import).
    os.environ["DIPFUSE"] = str(int(mani.get("dipfuse", 0)))
    os.environ["DIPKAPPA"] = str(mani.get("dipkappa", 0.1))
    os.environ["LIKTYPE"] = str(mani.get("liktype", "gauss"))
    os.environ["LIKP"] = str(mani.get("likp", 4.0))
    os.environ["TRAJ"] = str(int(mani.get("traj", 0)))             # New toggles (infer auto-match).
    os.environ["LINMDZ"] = str(int(mani.get("linmdz", 0)))
    os.environ["TVTREG"] = str(mani.get("tvtreg", 0))
    os.environ["LIKPF_MSEED"] = str(int(mani.get("likpf_mseed", 1)))
    os.environ["GEOPEARSON"] = str(int(mani.get("geopearson", 0)))
    os.environ["GEO_SEG"] = str(int(mani.get("geo_seg", 150)))
    os.environ["GEO_SPAN"] = str(mani.get("geo_span", 0.05))
    os.environ["GLOBALDECODE"] = str(int(mani.get("globaldecode", 0)))
    os.environ["RATEFEAT"] = str(int(mani.get("ratefeat", 0)))
    os.environ["GEOSPECTRUM"] = str(int(mani.get("geospectrum", 0)))
    os.environ["SELFCORR"] = str(int(mani.get("selfcorr", 0)))
    os.environ["REPSECTION"] = str(int(mani.get("repsection", 0)))
    os.environ["MODES"] = str(int(mani.get("modes", 0)))
    os.environ["EVALCAL"] = str(int(mani.get("evalcal", 0)))        # eval-cal (infer auto-match).
    os.environ["EVALCAL_K"] = str(int(mani.get("evalcal_k", 2)))
    os.environ["EVALCAL_LAM"] = str(mani.get("evalcal_lam", 1.0))
    os.environ["GRFILL"] = str(mani.get("grfill", "interp"))        # GR missing-value fill (infer auto-match).
    os.environ["MODESK"] = str(int(mani.get("modesk", 0)))          # MODES extension (infer auto-match).
    os.environ["NMODES"] = str(int(mani.get("nmodes", 3)))
    os.environ["MODECONF"] = str(int(mani.get("modeconf", 0)))      # train19 mode confidence features
    os.environ["GRFEATS2"] = str(int(mani.get("grfeats2", 0)))      # exp152 GR/mode slopes (infer auto-match).
    os.environ["PMF"] = str(int(mani.get("pmf", 0)))                # exp159 point-mass filter feature (infer auto-match).
    os.environ["RECAL_A"] = str(mani.get("recal_a", 1.0))           # exp154 global recal (infer auto-match).
    os.environ["RECAL_B"] = str(mani.get("recal_b", 0.0))
    print(f"artifacts {art} ({len(features)}feats, {len(models)}models) cfg: pf_mseed={M} "
          f"multiscale={mani.get('multiscale',0)} posterior={mani.get('posterior',0)} "
          f"pns={_PNS} pnp={_PNP} dipfuse={mani.get('dipfuse',0)} dipkappa={mani.get('dipkappa',0.1)}", flush=True)
    print(f"Inference model_type={mani.get('model_type', 'lightgbm-compatible')} device preference: {os.environ.get('INFER_LGB_DEVICE', 'cpu')}", flush=True)
    _seed_numba(SEED); np.random.seed(SEED)
    SI, DFIELD = _build_imputers(t0)
    print("build test features...", flush=True)
    test_df = _build_full(DATA / "test", False, SI, DFIELD, t0, "test")
    fh = _feat_hash([c for c in test_df.columns if c in set(features)])
    miss = [c for c in features if c not in test_df.columns]
    assert not miss, f"missing feats: {miss[:10]}"
    print(f"infer feat_hash(overlap)={_feat_hash([c for c in features])}", flush=True)
    pred = np.zeros(len(test_df))
    print(f"[infer] predicting base models: {len(models)} model files", flush=True)
    for mi, mp in enumerate(models, 1):
        print(f"[infer] loading/predicting model {mi}/{len(models)}: {mp.name}", flush=True)
        _model = joblib.load(mp)
        if mani.get("model_type") == "xgboost":
            try:
                _model.set_params(device="cpu")
            except Exception:
                pass
            _Xte = test_df[features].astype(np.float32, copy=False).replace([np.inf, -np.inf], np.nan)
            _bi = getattr(_model, "best_iteration", None)
            if _bi is not None:
                try:
                    pred += _model.predict(_Xte, iteration_range=(0, int(_bi) + 1)) / len(models)
                except Exception:
                    pred += _model.predict(_Xte) / len(models)
            else:
                pred += _model.predict(_Xte) / len(models)
        else:
            if os.environ.get("INFER_LGB_DEVICE", "cpu").strip().lower() == "cpu":
                try:
                    _model.set_params(device_type="cpu")
                except Exception:
                    pass
            pred += _model.predict(test_df[features]) / len(models)
        print(f"[infer] model {mi}/{len(models)} done", flush=True)
    # exp157 blend: predict and blend additional model groups from the same feature build (manifest blend_dir/w0/wb).
    #  blend_dir is a subfolder name in the dataset (with features.json + models/). Assumes the same feature config.
    _bdir = mani.get("blend_dir")
    if _bdir:
        bp_root = None
        for fj in glob.glob(f"/kaggle/input/**/{_bdir}/features.json", recursive=True):
            if (Path(fj).parent / "models").exists():
                bp_root = Path(fj).parent; break
        if bp_root is None and (art / _bdir / "features.json").exists():
            bp_root = art / _bdir
        if bp_root is not None:
            bfeats = json.load(open(bp_root / "features.json"))
            bmiss = [c for c in bfeats if c not in test_df.columns]
            assert not bmiss, f"blend missing feats: {bmiss[:10]}"
            bmodels = sorted((bp_root / "models").glob("fold*.pkl"))
            bp = np.zeros(len(test_df))
            for mp in bmodels:
                _model = joblib.load(mp)
                if os.environ.get("INFER_LGB_DEVICE", "cpu").strip().lower() == "cpu":
                    try:
                        _model.set_params(device_type="cpu")
                    except Exception:
                        pass
                bp += _model.predict(test_df[bfeats]) / len(bmodels)
            w0 = float(mani.get("blend_w0", 0.7)); wb = float(mani.get("blend_wb", 0.3))
            pred = w0 * pred + wb * bp
            print(f"BLEND applied: w0={w0} wb={wb} (+{len(bmodels)} models, {len(bfeats)}feats from {bp_root})", flush=True)
    # exp158 blend2: predict and blend a model with a different likpf config using a second feature build (override LIKPF_MSEED).
    _b2 = mani.get("blend2_dir")
    if _b2:
        b2root = None
        for fj in glob.glob(f"/kaggle/input/**/{_b2}/features.json", recursive=True):
            if (Path(fj).parent / "models").exists():
                b2root = Path(fj).parent; break
        if b2root is None and (art / _b2 / "features.json").exists():
            b2root = art / _b2
        if b2root is not None:
            os.environ["LIKPF_MSEED"] = str(mani.get("blend2_likpf", 8))   # Second build uses likpf=8.
            print(f"blend2: rebuild features with LIKPF_MSEED={os.environ['LIKPF_MSEED']}...", flush=True)
            test_df2 = _build_full(DATA / "test", False, SI, DFIELD, t0, "test_b2")
            assert list(test_df2["id"]) == list(test_df["id"]), "blend2 row order mismatch"
            b2feats = json.load(open(b2root / "features.json"))
            b2models = sorted((b2root / "models").glob("fold*.pkl"))
            b2p = np.zeros(len(test_df2))
            for mp in b2models:
                _model = joblib.load(mp)
                if os.environ.get("INFER_LGB_DEVICE", "cpu").strip().lower() == "cpu":
                    try:
                        _model.set_params(device_type="cpu")
                    except Exception:
                        pass
                b2p += _model.predict(test_df2[b2feats]) / len(b2models)
            w2 = float(mani.get("blend2_w", 0.25))
            pred = pred + w2 * b2p
            print(f"BLEND2 applied: w2={w2} (+{len(b2models)} models, likpf={os.environ['LIKPF_MSEED']})", flush=True)

    # === TabNet ensemble (memory: tabnet-diversity-win) ===
    # TabNet OOF=8.83 corr 0.81 -> blend 0.21 improves OOF by -0.076.
    # Stack further on top of exp159 + IRLS-U LB 6.949.
    try:
        # Install pytorch_tabnet from bundled wheel (no internet on Kaggle)
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor as _TabNet  # type: ignore
        except ImportError:
            import subprocess as _sp
            _whl = None
            for _w in glob.glob("/kaggle/input/**/pytorch_tabnet*.whl", recursive=True):
                _whl = _w; break
            if _whl:
                _sp.check_call(["pip", "install", "--quiet", "--no-deps", _whl])
                print(f"installed pytorch_tabnet from {_whl}", flush=True)
        import pickle as _pickle
        import torch as _torch
        from pytorch_tabnet.tab_model import TabNetRegressor as _TabNet
        print("DEBUG TabNet: starting search...", flush=True)
        tn_root = None
        for fj in glob.glob("/kaggle/input/**/feat_cols.json", recursive=True):
            tn_root = Path(fj).parent
            print(f"DEBUG TabNet: found feat_cols.json at {fj}", flush=True)
            break
        if tn_root is None and (art / "tabnet_models").exists():
            tn_root = art / "tabnet_models"
        print(f"DEBUG TabNet: tn_root={tn_root}", flush=True)
        if tn_root is not None and (tn_root / "feat_cols.json").exists():
            tn_feat_cols = json.load(open(tn_root / "feat_cols.json"))
            print(f"DEBUG TabNet: loaded {len(tn_feat_cols)} feat_cols", flush=True)
            _miss_tn = [c for c in tn_feat_cols if c not in test_df.columns]
            print(f"DEBUG TabNet: missing feats count={len(_miss_tn)}, first few={_miss_tn[:3]}", flush=True)
            if not _miss_tn:
                X_tn = test_df[tn_feat_cols].fillna(0).to_numpy(dtype=np.float32)
                tn_preds = np.zeros(len(X_tn), dtype=np.float32)
                _n_tn_folds = 0
                # Kaggle auto-extracts .zip → fold{fi}/ directory exists with extracted contents.
                # pytorch_tabnet load_model expects .zip → re-zip the directory in /tmp.
                import zipfile as _zf
                import tempfile as _tmp
                for fi in range(5):
                    _scaler_path = tn_root / f"fold{fi}_scaler.pkl"
                    _model_dir = tn_root / f"fold{fi}"
                    _model_zip_src = tn_root / f"fold{fi}.zip"
                    if not _scaler_path.exists():
                        continue
                    try:
                        with open(_scaler_path, "rb") as f:
                            _scaler = _pickle.load(f)
                        X_tn_s = _scaler.transform(X_tn).astype(np.float32)
                        # Determine model path: zip exists or re-zip from extracted dir
                        if _model_zip_src.exists():
                            _model_path = str(_model_zip_src)
                        elif _model_dir.exists() and _model_dir.is_dir():
                            # Re-zip the extracted directory in /tmp
                            _tmp_zip = Path(_tmp.gettempdir()) / f"tabnet_fold{fi}.zip"
                            with _zf.ZipFile(_tmp_zip, "w") as zf:
                                for _f in _model_dir.iterdir():
                                    zf.write(_f, arcname=_f.name)
                            _model_path = str(_tmp_zip)
                        else:
                            print(f"DEBUG TabNet fold{fi}: no model found", flush=True)
                            continue
                        _tn = _TabNet()
                        _tn.load_model(_model_path)
                        tn_preds += _tn.predict(X_tn_s).flatten()
                        _n_tn_folds += 1
                        print(f"DEBUG TabNet fold{fi}: loaded + predicted OK", flush=True)
                    except Exception as ie:
                        print(f"DEBUG TabNet fold{fi}: ERR {type(ie).__name__}: {ie}", flush=True)
                if _n_tn_folds > 0:
                    tn_preds /= _n_tn_folds
                    w_tn = 0.21
                    pred = (1 - w_tn) * pred + w_tn * tn_preds
                    print(f"TabNet blend applied: w_tn={w_tn} ({_n_tn_folds} folds, {len(tn_feat_cols)} feats)", flush=True)
                else:
                    print(f"DEBUG TabNet: 0 folds succeeded, blend NOT applied", flush=True)
            else:
                print(f"TabNet skipped: missing feats {_miss_tn[:5]}", flush=True)
        else:
            print("TabNet skipped: feat_cols.json not found", flush=True)
    except Exception as e:
        print(f"TabNet skipped due to error: {e}", flush=True)

    # === Custom PP: IRLS-U (Cauchy+ramp+0.75 blend) REPLACES existing _apply_pp polyfit step ===
    # memory: irls-uspace-projection (Δ-0.079 OOF, expected LB -0.05~-0.08)
    # Replace existing _apply_pp Huber polyfit (no blend, no ramp) with Cauchy+ramp+blend.
    # Keep the later pf_ancc 0.05 blend + tau=85 damping.
    def _irls_cauchy_polyfit(x_arr, y_arr, deg=4, n_iter=4):
        if len(y_arr) <= deg + 1:
            return y_arr.copy()
        w = np.ones_like(y_arr)
        for _ in range(n_iter):
            W = np.sqrt(w)
            A = np.vander(x_arr, deg + 1)
            cf, *_ = np.linalg.lstsq(A * W[:, None], y_arr * W, rcond=None)
            r = y_arr - A @ cf
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            w = 1.0 / (1.0 + (r / (2 * s)) ** 2)  # Cauchy weight (memory recommended)
        return A @ cf
    md_t = test_df["md_since"].to_numpy(float)
    pf_t = test_df["pf_ancc_d"].to_numpy(float)
    _zmap_irls = {}
    for _p in sorted((DATA / "test").glob("*__horizontal_well.csv")):
        _wid = _p.stem.replace("__horizontal_well", "")
        _df = pd.read_csv(_p, usecols=["Z", "TVT_input"])
        _ev = _df[_df["TVT_input"].isna()]
        for _i, _z in zip(_ev.index, _ev["Z"].to_numpy(float)):
            _zmap_irls[f"{_wid}_{_i}"] = _z
    Z_t = test_df["id"].map(_zmap_irls).to_numpy(float)
    d_t = pred.copy()
    wt_arr = test_df["well"].to_numpy()
    _n_irls_ok = 0
    for _wid, _gi in pd.DataFrame({"w": wt_arr}).groupby("w", sort=False).indices.items():
        if np.isnan(Z_t[_gi]).any() or len(_gi) < 50:
            continue
        md_w = md_t[_gi]
        if (md_w.max() - md_w.min()) < 1e-6:
            continue
        x_w = (md_w - md_w.min()) / (md_w.max() - md_w.min()) * 2 - 1
        U = d_t[_gi] + Z_t[_gi]  # U-space (last_tvt is well constant, drops out in subtraction)
        Ufit = _irls_cauchy_polyfit(x_w, U, deg=4, n_iter=4)
        ramp = 0.75 * np.clip((md_w - 100) / (500 - 100), 0, 1)
        Ub = ramp * Ufit + (1 - ramp) * U
        d_t[_gi] = Ub - Z_t[_gi]
        _n_irls_ok += 1
    print(f"IRLS-U PP (Cauchy deg=4, ramp [100,500] max=0.75): {_n_irls_ok} wells", flush=True)
    # Keep remaining steps from existing PP: pf_ancc 0.05 blend + tau=85 damping.
    d_t = d_t * (1 - 0.05) + pf_t * 0.05
    d_t = d_t * (1.0 - np.exp(-np.maximum(md_t, 0.0) / 85.0))
    _ra = float(os.environ.get("RECAL_A", "1.0")); _rb = float(os.environ.get("RECAL_B", "0.0"))
    if _ra != 1.0 or _rb != 0.0:
        d_t = _ra * d_t + _rb
        print(f"RECAL applied: a={_ra} b={_rb}", flush=True)
    test_out = test_df.assign(tvt=test_df["last_tvt"].to_numpy() + d_t)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample[["id"]].merge(test_out[["id", "tvt"]], on="id", how="left")
    # sanity guards (prevent silent submission with degenerate output).
    n = len(sub); nfill = int(sub["tvt"].isna().sum())
    pstd = float(np.nanstd(test_out["tvt"]))
    wstd = float(test_out.groupby("well")["tvt"].std().median())
    print(f"SANITY: id_match={n-nfill}/{n} pred_std={pstd:.3f} perwell_std_med={wstd:.3f} "
          f"nwells={test_out['well'].nunique()} nfeats={len(features)} nmodels={len(models)}", flush=True)
    assert nfill <= 0.01 * n, f"FATAL degenerate: {nfill}/{n} ids unmatched"
    assert wstd > 0.3, f"FATAL degenerate: per-well tvt std {wstd:.4f} ~0"
    out = Path("/kaggle/working/submission.csv") if KAGGLE else Path(__file__).parent / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"submission OK: {out} rows={len(sub)} ({time.time() - t0:.0f}s)", flush=True)


def main():
    print(f"=== rogii_repro MODE={MODE} (self-contained, PF seed42 fixed) ===", flush=True)
    print(f"DATA={DATA} KAGGLE={KAGGLE} ART={ART}", flush=True)
    print(f"TRAIN_XGB_DEVICE={os.environ.get('TRAIN_XGB_DEVICE')} INFER_DEVICE=cpu N_EST={N_EST} N_JOBS={os.environ.get('N_JOBS', '4')} LOG_WELL_EVERY={os.environ.get('LOG_WELL_EVERY', '25')}", flush=True)
    assert (DATA / "train").exists(), f"DATA/train not found: {DATA / 'train'}"
    if SMOKE_WELLS:   # Validation: limit train to the first N wells (replace build_dataset used by build_all).
        def build_dataset_smoke(d, is_train):
            paths = sorted(d.glob("*__horizontal_well.csv"))
            if is_train:
                paths = paths[:SMOKE_WELLS]
            parts = [build_well(p, is_train) for p in paths]
            return pd.concat([r for r in parts if r is not None], ignore_index=True)
        globals()["build_dataset"] = build_dataset_smoke

        def add_phys_smoke(base_df, d, DF=None, is_train=False):
            from joblib import Parallel, delayed
            paths = sorted(Path(d).glob("*__horizontal_well.csv"))
            if "train" in str(d):
                paths = paths[:SMOKE_WELLS]
            sdm = {}
            if os.environ.get("DIPFUSE", "0") == "1" and DF is not None:
                for p in paths:
                    wid, sd = _sdip_one(str(p), DF, is_train)
                    if sd is not None:
                        sdm[wid] = sd
            parts = Parallel(n_jobs=int(os.environ.get("N_JOBS", "4")), prefer="processes")(
                delayed(genphys_one)(str(p), sdm.get(Path(p).stem.replace("__horizontal_well", ""))) for p in paths)
            pf = pd.concat([p for p in parts if p is not None], ignore_index=True)
            return base_df.merge(pf, on="id", how="left")
        globals()["add_phys_feats"] = add_phys_smoke
    if MODE == "train":
        run_train()
        print(f"TRAIN-ONLY DONE: artifacts saved to {ART}", flush=True)
    else:
        run_infer()


if __name__ == "__main__":
    main()
