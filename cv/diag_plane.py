import numpy as np, pandas as pd, random
from cv_runner import WellStore
from scipy.spatial import cKDTree

store = WellStore(); ids = store.ids()
# cloud of absolute formation surface F=ANCC (shared geology) sampled per well
STRIDE = 40
XS, FS, OW = [], [], []
for w in ids:
    df = pd.read_csv(store.data_root / "train" / f"{w}__horizontal_well.csv", usecols=["X", "Y", "ANCC"])
    idx = np.arange(0, len(df), STRIDE)
    xy = df[["X", "Y"]].to_numpy()[idx]; F = df["ANCC"].to_numpy()[idx]
    ok = np.isfinite(F) & np.isfinite(xy[:, 0])
    XS.append(xy[ok]); FS.append(F[ok]); OW += [w] * int(ok.sum())
XS = np.vstack(XS).astype(float); FS = np.concatenate(FS); OW = np.array(OW)
tree = cKDTree(XS)

def neighbor_dip(psxy, self_w, k=400):
    """fit plane F ~ a*x+b*y+c on nearest non-self cloud points; return (a,b)."""
    d, j = tree.query(psxy, k=k)
    m = OW[j] != self_w
    j = j[m]; d = d[m]
    if len(j) < 30:
        return 0.0, 0.0
    P = XS[j]; F = FS[j]
    wt = 1.0 / (d + 1.0)
    A = np.column_stack([P[:, 0] - psxy[0], P[:, 1] - psxy[1], np.ones(len(P))])
    W = np.sqrt(wt)
    coef, *_ = np.linalg.lstsq(A * W[:, None], F * W, rcond=None)
    return float(coef[0]), float(coef[1])

random.seed(0); sample = random.sample(ids, 150)
res = {}
for tag in ["nbr_dip", "prefix_dip", "blend_dip"]:
    res[tag] = {"errs": [], "dp": [], "dt": [], "sd": [0.0, 0.0, 0.0]}
errs_cf = []
for w in sample:
    ww = store.wells[w]; hs = ww.ps_idx
    t = ww.tvt[hs:]; z = ww.z[hs:].astype(float); zps = float(ww.z[hs - 1]); lk = ww.last_known
    psxy = np.array([ww.x[hs - 1], ww.y[hs - 1]], float)
    dx = ww.x[hs:].astype(float) - psxy[0]; dy = ww.y[hs:].astype(float) - psxy[1]
    errs_cf.append(np.full_like(t, lk) - t)
    # neighbor dip
    a, b = neighbor_dip(psxy, w)
    # prefix dip: F_prefix = TVT_input + Z over lateral prefix; fit F ~ x,y
    pl = slice(max(0, hs - 700), hs)
    Fpre = ww.tvt_input[pl] + ww.z[pl].astype(float)
    px = ww.x[pl].astype(float); py = ww.y[pl].astype(float)
    mfin = np.isfinite(Fpre)
    ap = bp = np.nan
    if mfin.sum() > 50 and np.std(px[mfin]) + np.std(py[mfin]) > 1.0:
        Ap = np.column_stack([px[mfin] - psxy[0], py[mfin] - psxy[1], np.ones(mfin.sum())])
        cp, *_ = np.linalg.lstsq(Ap, Fpre[mfin], rcond=None)
        ap, bp = float(cp[0]), float(cp[1])
    dips = {"nbr_dip": (a, b),
            "prefix_dip": (ap if np.isfinite(ap) else a, bp if np.isfinite(bp) else b),
            "blend_dip": (np.nanmean([a, ap]) if np.isfinite(ap) else a,
                          np.nanmean([b, bp]) if np.isfinite(bp) else b)}
    for tag, (aa, bb) in dips.items():
        dF = aa * dx + bb * dy
        pred = lk - (z - zps) + dF
        e = pred - t
        res[tag]["errs"].append(e); res[tag]["dp"].append((pred.mean() - lk)); res[tag]["dt"].append(t.mean() - lk)
        ps = pred - pred.mean(); ts = t - t.mean()
        sd = res[tag]["sd"]; sd[0] += np.sum(ps * ts); sd[1] += np.sum(ps * ps); sd[2] += np.sum(ts * ts)

def rw(e):
    E = np.concatenate(e); return float(np.sqrt(np.mean(E ** 2)))
print("CF                  :", round(rw(errs_cf), 3))
for tag in res:
    r = res[tag]
    print(f"{tag:12s} rmse={rw(r['errs']):.3f}  drift_corr={np.corrcoef(r['dp'],r['dt'])[0,1]:+.3f}  "
          f"shape_corr={r['sd'][0]/np.sqrt(r['sd'][1]*r['sd'][2]+1e-9):+.3f}")
print("(const-oracle 9.04, smooth-201 0.39)")
