import numpy as np, pandas as pd, random
from cv_runner import WellStore
from scipy.spatial import cKDTree

store = WellStore(); ids = store.ids()
STRIDE = 80
XS, FS, OW = [], [], []
for w in ids:
    df = pd.read_csv(store.data_root / "train" / f"{w}__horizontal_well.csv", usecols=["X", "Y", "ANCC"])
    idx = np.arange(0, len(df), STRIDE)
    xy = df[["X", "Y"]].to_numpy()[idx]; F = df["ANCC"].to_numpy()[idx]
    ok = np.isfinite(F) & np.isfinite(xy[:, 0])
    XS.append(xy[ok]); FS.append(F[ok]); OW += [w] * int(ok.sum())
XS = np.vstack(XS); FS = np.concatenate(FS); OW = np.array(OW)
tree = cKDTree(XS)

def idw(q, self_w):
    d, j = tree.query(q, k=250)
    out = np.empty(len(q))
    for r in range(len(q)):
        jj, dd = j[r], d[r]
        m = OW[jj] != self_w
        jj, dd = jj[m][:15], dd[m][:15]
        if len(jj) == 0:
            out[r] = np.nan; continue
        wt = 1.0 / (dd + 1e-6)
        out[r] = (wt * FS[jj]).sum() / wt.sum()
    return out

random.seed(0); sample = random.sample(ids, 150)
drifts_t, drifts_p, ns = [], [], []   # per-well true drift curve & physics drift curve
for w in sample:
    ww = store.wells[w]; hs = ww.ps_idx
    t = ww.tvt[hs:]; z = ww.z[hs:].astype(float); zps = float(ww.z[hs - 1]); lk = ww.last_known
    n = ww.n_hidden; anc = np.arange(0, n, 40)
    qxy = np.column_stack([ww.x[hs:][anc], ww.y[hs:][anc]]).astype(float)
    psxy = np.array([[ww.x[hs - 1], ww.y[hs - 1]]], float)
    Fh = idw(qxy, w); Fps = idw(psxy, w)[0]
    Fh_full = np.interp(ww.md[hs:], ww.md[hs:][anc], Fh)
    pdrift = -(z - zps) + (Fh_full - Fps)        # physics drift curve
    tdrift = t - lk                               # true drift curve
    drifts_p.append(pdrift); drifts_t.append(tdrift); ns.append(n)

def rw_for_beta(beta, clip=None):
    sse = tot = 0.0
    for pd_, td_ in zip(drifts_p, drifts_t):
        d = beta * pd_
        if clip is not None:
            d = np.clip(d, -clip, clip)
        e = d - td_
        sse += np.sum(e ** 2); tot += len(e)
    return np.sqrt(sse / tot)

cf = rw_for_beta(0.0)
print(f"CF (beta=0)                 : {cf:.3f}")
for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    print(f"  beta={b:.1f}                 : {rw_for_beta(b):.3f}")
# optimal global beta
num = sum(np.sum(p * t) for p, t in zip(drifts_p, drifts_t))
den = sum(np.sum(p * p) for p in drifts_p)
bopt = num / den
print(f"  beta_opt={bopt:.3f}           : {rw_for_beta(bopt):.3f}")
for c in [20, 30, 40]:
    print(f"  beta_opt + clip{c}ft        : {rw_for_beta(bopt, clip=c):.3f}")
print("(ref const-oracle 9.04, smooth-201 0.39)")
