"""
exp204 — 純 GR + 幾何のみの特徴ビルダー(強い特徴=PF/地層/beam/dtw は一切使わない)。

WARP 相当ニューラルアーム(dTVT 積分 + アンカー)の検証用。ユーザ仮説「U2Net が強いのは
アーキでなく強い特徴(likpf 等)のおかげ」を、特徴を生 GR + 幾何 + typewell ruler に絞って検証。

各 eval 位置のチャネル(全て純 GR or 軌道幾何。PF/地層/likpf/beam/dtw は無し):
  GR系:   hgr(補間GR), gr_d1, gr_d2, gr_env(rolling max), gr_sm5, gr_sm15, gr_lstd(rolling std)
  幾何系: z_rel(Z-last_Z), dzdmd, dxdmd, dydmd, md_since, frac, dxy, azimuth
typewell ruler(cross-attn 用, per-well): tw_gr / tw_tvt を固定 T グリッドへ resample。
  → dip 変換に必須の dGR/dTVT を net に読ませる(生 GR だけでは dip 盲目 [[exp101]])。

target = 真 TVT(CV 用)。last_tvt(アンカー) と共に保存。exp140 の wid 順に整列し
recovered_fold.npy を付与(= exp140 full-feature U2Net と同一分割で honest CV 比較 [[exp141]])。

  ../exp008/.venv/bin/python build_gr_features.py
出力: gr_features_cache.pkl
"""
import os, sys, time, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRAIN_DIR = ROOT / "input" / "train"
EXP140_CACHE = ROOT / "EXP" / "exp140" / "features_cache.pkl"
EXP140_FOLD = ROOT / "EXP" / "exp140" / "recovered_fold.npy"
CACHE = HERE / "gr_features_cache.pkl"

TW_T = 192   # typewell を resample する固定トークン数


def _roll(s, w, fn):
    return getattr(s.rolling(w, center=True, min_periods=1), fn)().to_numpy(np.float32)


def build_one(wid):
    hw_p = TRAIN_DIR / f"{wid}__horizontal_well.csv"
    tw_p = TRAIN_DIR / f"{wid}__typewell.csv"
    if not (hw_p.exists() and tw_p.exists()):
        return None
    hw = pd.read_csv(hw_p)
    tw = pd.read_csv(tw_p).sort_values("TVT")
    if "TVT" not in hw.columns or hw["TVT"].isna().all():
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"]); last_Z = float(lk["Z"]); last_MD = float(lk["MD"])
    nh = len(ev); ev_start = ev.index[0]

    tw_tvt = tw["TVT"].to_numpy(np.float64)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(np.float64)
    if len(tw_tvt) < 3:
        return None

    # ---- 生 GR チャネル(補間で欠損を消す = exp011/PF と同一の扱い) ----
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    gr_s = pd.Series(gr_full.values)
    def ev_slice(a): return np.asarray(a, np.float32)[ev_start:ev_start + nh]
    hgr = ev_slice(gr_full.values)
    gr_d1 = ev_slice(gr_s.diff().fillna(0.).values)
    gr_d2 = ev_slice(gr_s.diff().diff().fillna(0.).values)
    gr_env = ev_slice(_roll(gr_s, 21, "max"))
    gr_sm5 = ev_slice(_roll(gr_s, 5, "mean"))
    gr_sm15 = ev_slice(_roll(gr_s, 15, "mean"))
    gr_lstd = ev_slice(_roll(gr_s, 15, "std"))

    # ---- 幾何チャネル(軌道 = 強い特徴でない) ----
    z_ev = ev["Z"].to_numpy(np.float32)
    z_rel = (z_ev - np.float32(last_Z))
    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = ev_slice((hw["Z"].diff() / mdd).values)
    dxdmd = ev_slice((hw["X"].diff() / mdd).values)
    dydmd = ev_slice((hw["Y"].diff() / mdd).values)
    md_since = (ev["MD"].to_numpy(np.float32) - np.float32(last_MD))
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    dxy = np.sqrt((ev["X"].values - float(lk["X"])) ** 2 +
                  (ev["Y"].values - float(lk["Y"])) ** 2).astype(np.float32)
    _dx = ev_slice(hw["X"].diff().values); _dy = ev_slice(hw["Y"].diff().values)
    azimuth = np.arctan2(_dy, _dx).astype(np.float32)

    feat = np.column_stack([
        hgr, gr_d1, gr_d2, gr_env, gr_sm5, gr_sm15, gr_lstd,
        z_rel, dzdmd, dxdmd, dydmd, md_since, frac, dxy, azimuth,
    ]).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- typewell ruler: 固定 T グリッドへ resample(TVT は last_tvt 相対) ----
    tgrid = np.linspace(tw_tvt[0], tw_tvt[-1], TW_T)
    tw_gr_rs = np.interp(tgrid, tw_tvt, tw_gr).astype(np.float32)
    tw_tvt_rel = (tgrid - last_tvt).astype(np.float32)
    tw_tokens = np.column_stack([tw_gr_rs, tw_tvt_rel]).astype(np.float32)  # (T, 2)

    true_tvt = ev["TVT"].to_numpy(np.float32)
    # dTVT 増分ターゲット(アンカー基準): dtvt[0]=TVT[0]-last_tvt, 以降は階差
    dtvt = np.empty(nh, np.float32)
    dtvt[0] = true_tvt[0] - last_tvt
    dtvt[1:] = np.diff(true_tvt)

    return dict(wid=wid, features=feat, tw_tokens=tw_tokens,
                target_tvt=true_tvt, target_dtvt=dtvt, target_delta=(true_tvt - last_tvt).astype(np.float32),
                last_tvt=last_tvt, n_eval=nh)


def main():
    # [zip package 用パッチ] wid 順は exp140 cache(531MB)からでなく同梱 wid_order.json から読める
    if EXP140_CACHE.exists():
        wid_order = [d["wid"] for d in joblib.load(EXP140_CACHE)]
    else:
        import json
        wid_order = json.load(open(HERE.parent / "exp140" / "wid_order.json"))
    fold = np.load(EXP140_FOLD)
    assert len(wid_order) == len(fold)
    fold_by_wid = {w: int(f) for w, f in zip(wid_order, fold)}
    # Optional: override with shared typewell GroupKFold CSV
    shared_csv = os.environ.get("ROGII_SHARED_FOLD_CSV", "").strip()
    if shared_csv:
        fm = pd.read_csv(shared_csv)
        fold_by_wid = {str(r.well_id): int(r.fold) for r in fm.itertuples(index=False)}
        print(f"[exp204] using shared fold map {shared_csv} ({len(fold_by_wid)} wells)")

    print(f"[exp204] building pure-GR features for {len(wid_order)} wells (exp140 order)...")
    t0 = time.time()
    # Prefer sequential / small pools: large loky pools collide with concurrent LGB jobs.
    n_jobs = int(os.environ.get("ROGII_WARP_BUILD_JOBS", "1") or "1")
    if n_jobs <= 1:
        results = [build_one(w) for w in wid_order]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(build_one)(w) for w in wid_order)

    out = []
    for w, r in zip(wid_order, results):
        if r is not None:
            r["fold"] = fold_by_wid[w]
            out.append(r)
    n_eval = sum(d["n_eval"] for d in out)
    print(f"[exp204] built {len(out)}/{len(wid_order)} wells in {time.time()-t0:.0f}s | "
          f"eval pts={n_eval:,} | in_ch={out[0]['features'].shape[1]} tw_T={out[0]['tw_tokens'].shape[0]}")
    joblib.dump(out, CACHE, compress=3)
    print(f"[exp204] cached -> {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)")
    # 集計健全性: flat persistence pooled == 15.9099 か
    sse = sum(float(np.sum((d["target_tvt"] - d["last_tvt"]) ** 2)) for d in out)
    print(f"[exp204] flat-persistence pooled RMSE = {np.sqrt(sse/n_eval):.4f}  (exp001=15.9099 で整合)")


if __name__ == "__main__":
    main()
