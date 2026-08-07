from lightgbm import LGBMRegressor, log_evaluation, early_stopping
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse
    def root_mean_squared_error(y,p): return _mse(y,p)**0.5
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

from pathlib import Path
from numba import njit
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import time
import glob
import os
import json
import sys, types
ls = types.ModuleType("level_sources_lib")
sys.modules["level_sources_lib"] = ls
ls.__dict__["__file__"] = "level_sources_lib.py"
exec('from lightgbm import LGBMRegressor, log_evaluation, early_stopping\ntry:\n    from sklearn.metrics import root_mean_squared_error\nexcept ImportError:\n    from sklearn.metrics import mean_squared_error as _mse\n    def root_mean_squared_error(y,p): return _mse(y,p)**0.5\nfrom sklearn.model_selection import GroupKFold\nfrom sklearn.linear_model import Ridge\nfrom catboost import CatBoostRegressor\nfrom scipy.spatial import cKDTree\nfrom scipy.signal import savgol_filter\nfrom joblib import Parallel, delayed\n\nfrom pathlib import Path\nfrom numba import njit\nimport matplotlib.pyplot as plt\nimport multiprocessing\nimport seaborn as sns\nimport pandas as pd\nimport numpy as np\nimport warnings\nimport joblib\nimport time\nimport glob\nimport os\n\nwarnings.filterwarnings("ignore")\n\nclass CFG:\n    dataset_path = Path("/home/yiyu/rogii/datasets/rogii-wellbore-geology-prediction")\n    artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")\n    \n    seed = 42\n    n_splits = 5\n    cv = GroupKFold(n_splits=n_splits)\n    \n    metric = root_mean_squared_error\n\nSELECTOR_N_EVAL_THRESHOLD = 4840.0\nSELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)\n\nSELECTOR_BIN_VARIANTS = {\n    0: \'pf_scale_5_hold_0.2\',\n    1: \'pf_scale_3_hold_0.15\',\n    2: \'pf_scale_12_beam_0.2_hold_0.15\',\n    3: \'pf_scale_5_hold_0.15\',\n    4: \'pf_scale_5_beam_0.05_hold_0.05\',\n    5: \'pf_scale_12_beam_0.2_hold_0.05\',\n}\n\nSELECTOR_GLOBAL_VARIANT = \'pf_scale_8_hold_0.2\'\nSELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)\n\nFORMATION_COLS = [\'ANCC\', \'ASTNU\', \'ASTNL\', \'EGFDU\', \'EGFDL\', \'BUDA\']\n\nBEAM_CONFIGS = [\n    (10, 20.0, 144.0, 2),\n    (10,  8.0,  64.0, 2),\n    ( 8, 35.0, 220.0, 1),\n    (10, 14.0,  90.0, 5),\n    (20,  4.0,  36.0, 3),\n    (12, 12.0, 100.0, 3),\n    (15, 25.0, 180.0, 2),\n    (20, 30.0, 200.0, 2),\n    (15, 10.0,  80.0, 4),\n    (25,  6.0,  50.0, 3),\n    (10, 40.0, 300.0, 1),\n    (12, 18.0, 120.0, 5),\n    (30,  8.0,  70.0, 2),\n    (10, 50.0, 400.0, 0),\n]\n\n\ndef tvt_from_contacts(hw_tr, tw_tr, ref_col=\'EGFDU\'):\n    tw_g = tw_tr.dropna(subset=[\'Geology\'])\n    ref_tvt = tw_g[tw_g[\'Geology\'] == ref_col][\'TVT\'].min()\n    if np.isnan(ref_tvt):\n        ref_col = tw_g[\'Geology\'].iloc[0]\n        ref_tvt = tw_g[tw_g[\'Geology\'] == ref_col][\'TVT\'].min()\n    offset = (hw_tr[\'TVT\'] - (ref_tvt - (hw_tr[\'Z\'] - hw_tr[ref_col]))).mean()\n    return ref_tvt - (hw_tr[\'Z\'] - hw_tr[ref_col]) + offset\n\n\ndef load_well(wid, split=\'train\'):\n    base = CFG.dataset_path / split\n    hw = pd.read_csv(base / f\'{wid}__horizontal_well.csv\')\n    tw = pd.read_csv(base / f\'{wid}__typewell.csv\')\n    return hw, tw\n\n\ndef run_particle_filter(hw, tw, n_particles=500, seed=42):\n    tw_s   = tw.sort_values(\'TVT\')\n    tw_tvt = tw_s[\'TVT\'].values.astype(float)\n    tw_gr  = tw_s[\'GR\'].fillna(tw_s[\'GR\'].mean()).values.astype(float)\n\n    kn = hw[hw[\'TVT_input\'].notna()]\n    ev = hw[hw[\'TVT_input\'].isna()]\n    if len(ev) == 0:\n        return hw[\'TVT_input\'].values.astype(float).copy(), 0.0\n\n    last     = kn.iloc[-1]\n    last_tvt = float(last[\'TVT_input\'])\n    last_Z   = float(last[\'Z\'])\n    last_MD  = float(last[\'MD\'])\n\n    tw_at_k = np.interp(kn[\'TVT_input\'].values, tw_tvt, tw_gr)\n    gs = float(np.clip(np.nanstd(kn[\'GR\'].fillna(0).values - tw_at_k), 10., 60.))\n\n    tail = kn.tail(30)\n    dt = np.diff(tail[\'TVT_input\'].values)\n    dz = np.diff(tail[\'Z\'].values)\n    dm = np.diff(tail[\'MD\'].values)\n    m  = dm > 0\n    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0\n\n    N   = n_particles\n    rng = np.random.default_rng(seed)\n    ls   = last_tvt + last_Z\n    pos  = ls + 2.0 * rng.standard_normal(N)\n    rate = ir + 0.01 * rng.standard_normal(N)\n    w    = np.ones(N) / N\n\n    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5\n\n    md_v = ev[\'MD\'].values.astype(float)\n    z_v  = ev[\'Z\'].values.astype(float)\n    # Interpolate GR gaps before tracking\n    gr_interp = hw[\'GR\'].interpolate(limit_direction=\'both\').fillna(tw_gr.mean())\n    gr_v = gr_interp.values.astype(float)[ev.index]\n\n    out_vals = hw[\'TVT_input\'].values.astype(float).copy()\n    res = np.empty(len(ev))\n    prev_MD = last_MD\n    log_lik = 0.0\n\n    for i in range(len(ev)):\n        dm_step = max(md_v[i] - prev_MD, 1.0)\n        rate = MOM * rate + VN * rng.standard_normal(N)\n        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)\n        tvt_p = pos - z_v[i]\n        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)\n        pos   = tvt_p + z_v[i]\n\n        eg = np.interp(tvt_p, tw_tvt, tw_gr)\n        d  = (gr_v[i] - eg) / gs\n        lk = np.exp(-0.5 * np.minimum(d**2, 600.))\n        lk = np.maximum(lk, 1e-300)\n        avg_lk = float((w * lk).sum())\n        log_lik += np.log(max(avg_lk, 1e-300))\n        w = w * lk\n        ws = w.sum()\n        w = w / ws if ws > 0 else np.ones(N) / N\n\n        n_eff = 1.0 / (w**2).sum()\n        if n_eff < RESAMP * N:\n            cum = np.cumsum(w)\n            u0  = rng.uniform(0, 1.0 / N)\n            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)\n            pos  = pos[idx]  + RP * rng.standard_normal(N)\n            rate = rate[idx] + RR * rng.standard_normal(N)\n            w    = np.ones(N) / N\n\n        res[i] = float(np.dot(w, pos - z_v[i]))\n        prev_MD = md_v[i]\n\n    out_vals[list(ev.index)] = res\n    return out_vals, log_lik\n\n\ndef run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):\n    preds = []\n    liks  = []\n    for s in range(n_seeds):\n        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)\n        preds.append(p)\n        liks.append(ll)\n\n    liks   = np.array(liks)\n    liks_n = liks - liks.max()\n    weights = np.exp(liks_n / scale)\n    weights /= weights.sum()\n\n    return (weights[:, None] * np.stack(preds, 0)).sum(0)\n\n\ndef run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):\n    preds = []\n    liks = []\n    for s in range(n_seeds):\n        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)\n        preds.append(p)\n        liks.append(ll)\n    pred_arr = np.stack(preds, 0)\n    liks = np.array(liks)\n    liks_n = liks - liks.max()\n    out = {}\n    for scale in scales:\n        weights = np.exp(liks_n / float(scale))\n        weights /= weights.sum()\n        out[f\'pf_scale_{scale:g}\'] = (weights[:, None] * pred_arr).sum(0)\n    out[\'pf_mean\'] = pred_arr.mean(0)\n    return out\n\n\ndef beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):\n    n  = len(hgr)\n    nt = len(tw_tvt)\n    if n == 0:\n        return np.array([last_tvt])\n\n    if r > 0 and n > max(3, 2 * r + 1):\n        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)\n        sgr = savgol_filter(hgr, win, min(2, win - 1))\n    else:\n        sgr = hgr.copy()\n\n    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))\n\n    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)\n    MC    = mc * np.array([2., 1., 0., 1., 2.])\n\n    bidx  = np.full(bs, si, dtype=np.int64)\n    bcost = np.full(bs, np.inf)\n    bcost[0] = 0.\n    bn = 1\n\n    result = np.zeros(n)\n\n    for step in range(n):\n        gv = sgr[step]\n        ni = bidx[:bn, None] + MOVES[None, :]\n        ci = np.clip(ni, 0, nt - 1)\n        valid = (ni >= 0) & (ni < nt)\n\n        gr_e = (gv - tw_gr[ci])**2 / es\n        tot  = bcost[:bn, None] + gr_e + MC[None, :]\n        tot  = np.where(valid, tot, np.inf)\n\n        ni_f  = ni.flatten()\n        tot_f = tot.flatten()\n        vf    = valid.flatten()\n        ni_f  = ni_f[vf]\n        tot_f = tot_f[vf]\n\n        order = np.argsort(tot_f)\n        ni_s  = ni_f[order]\n        tot_s = tot_f[order]\n\n        _, first = np.unique(ni_s, return_index=True)\n        ni_u  = ni_s[first]\n        tot_u = tot_s[first]\n\n        kept = min(bs, len(ni_u))\n        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]\n        top  = top[np.argsort(tot_u[top])]\n\n        bidx[:kept]  = ni_u[top]\n        bcost[:kept] = tot_u[top]\n        if kept < bs:\n            bidx[kept:]  = bidx[kept - 1]\n            bcost[kept:] = np.inf\n        bn = kept\n\n        result[step] = tw_tvt[bidx[0]]\n\n    return result\n\n\ndef run_beam_ensemble(hw, tw):\n    kn = hw[hw[\'TVT_input\'].notna()]\n    ev = hw[hw[\'TVT_input\'].isna()]\n    if len(ev) == 0:\n        return hw[\'TVT_input\'].values.astype(float).copy()\n\n    last_tvt = float(kn.iloc[-1][\'TVT_input\'])\n    tw_s  = tw.sort_values(\'TVT\')\n    tw_tvt = tw_s[\'TVT\'].values.astype(float)\n    tw_gr  = tw_s[\'GR\'].fillna(tw_s[\'GR\'].mean()).values.astype(float)\n\n    gr_all = hw[\'GR\'].interpolate(limit_direction=\'both\').fillna(tw_gr.mean()).values.astype(float)\n    hgr    = gr_all[ev.index]\n\n    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)\n                    for (bs, mc, es, r) in BEAM_CONFIGS]\n\n    beam_mean = np.stack(beam_results, 0).mean(0)\n\n    out = hw[\'TVT_input\'].values.astype(float).copy()\n    out[list(ev.index)] = beam_mean\n    return out\n\n\ndef selector_well_code(hw):\n    eval_mask = hw[\'TVT_input\'].isna().to_numpy()\n    n_eval = float(eval_mask.sum())\n    z_eval = hw.loc[eval_mask, \'Z\'].values.astype(float)\n    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0\n    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)\n    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side=\'right\'))\n    code = n_bin + 2 * z_bin\n    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)\n    return code, variant, n_eval, z_span\n\n\ndef parse_selector_variant(name):\n    parts = name.split(\'_\')\n    scale = float(parts[2])\n    beam_weight = 0.0\n    hold_weight = 0.0\n    if \'beam\' in parts:\n        beam_weight = float(parts[parts.index(\'beam\') + 1])\n    if \'hold\' in parts:\n        hold_weight = float(parts[parts.index(\'hold\') + 1])\n    return scale, beam_weight, hold_weight\n\n\ndef apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):\n    scale, beam_weight, hold_weight = parse_selector_variant(name)\n    base = pf_by_scale.get(f\'pf_scale_{scale:g}\')\n    if base is None:\n        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split(\'_beam_\')[0].split(\'_hold_\')[0]]\n    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam\n    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt\n    return pred\n\nSEED=42\nNCPU=min(192,multiprocessing.cpu_count())\n\nFORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]\nPLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5\n\nBEAMS=[\n    (10,20.0,144.0,2,"cons"),\n    (10, 8.0, 64.0,2,"loose"),\n    ( 8,35.0,220.0,1,"vcons"),\n    (10,14.0, 90.0,5,"sm5"),\n    (20, 4.0, 36.0,3,"vloose"),\n    (12,12.0,100.0,3,"mid"),\n    (15,25.0,180.0,2,"stiff"),\n]\n\nPF_N=600; ANCC_N=600\nPF_MOM=0.993; PF_VN=0.005; PF_PN=0.01\nPF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.\nPF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5\nPF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3\nANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005\nANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001\n\n@njit(cache=False)\ndef _interp1(grid, v, vmin, step):\n    i = int((v - vmin) / step)\n    if i < 0: return grid[0]\n    n = len(grid) - 1\n    if i >= n: return grid[n]\n    t = (v - vmin) / step - i\n    return grid[i]*(1.-t) + grid[i+1]*t\n\n@njit(cache=False)\ndef _resamp(pos, aux, w, N, rp, rv):\n    cum = np.zeros(N+1)\n    for j in range(N): cum[j+1]=cum[j]+w[j]\n    u0=np.random.uniform(0.,1./N)\n    np2=np.empty(N); na=np.empty(N); ci=0\n    for j in range(N):\n        u=u0+j/N\n        while ci<N-1 and cum[ci+1]<u: ci+=1\n        np2[j]=pos[ci]+rp*np.random.randn()\n        na[j] =aux[ci]+rv*np.random.randn()\n    return np2,na\n\n@njit(cache=False)\ndef _beam_jit(sgr, tw_gr, si, BS, mc, es):\n    """Beam search ±2 delta, Numba JIT."""\n    n=len(sgr); nt=len(tw_gr); MAX=BS*6\n    bidx=np.zeros(BS,np.int64); bidx[0]=si\n    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)\n    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)\n    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)\n    for step in range(n):\n        gv=sgr[step]; nc=np.int64(0)\n        for bi in range(bn):\n            idx=bidx[bi]; cost=bcost[bi]\n            for d in range(-2,3):            # ±2: TVT can go down\n                ni=idx+d\n                if ni<0 or ni>=nt: continue\n                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)\n                fnd=np.int64(-1)\n                for ci in range(nc):\n                    if cI[ci]==ni: fnd=ci; break\n                if fnd>=0:\n                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi\n                else:\n                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1\n        kept=min(BS,nc)\n        for i in range(kept):\n            mi=i\n            for j in range(i+1,nc):\n                if cC[j]<cC[mi]: mi=j\n            if mi!=i:\n                cI[i],cI[mi]=cI[mi],cI[i]\n                cC[i],cC[mi]=cC[mi],cC[i]\n                cP[i],cP[mi]=cP[mi],cP[i]\n        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]\n        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept\n    best=np.int64(0)\n    for b in range(1,bn):\n        if bcost[b]<bcost[best]: best=b\n    path=np.zeros(n,np.int64); b=best\n    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]\n    return path\n\n@njit(cache=False)\ndef _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,\n              ALPHA,RN,PN,IS,RP,RR,RESAMP):\n    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N\n    for j in range(N):\n        pos[j]=ls+IS*np.random.randn()\n        rate[j]=ir+0.01*np.random.randn()\n    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.\n    for i in range(len(md_v)):\n        dm=md_v[i]-pm; dm=max(dm,1.)\n        for j in range(N):\n            rate[j]=ALPHA*rate[j]+RN*np.random.randn()\n            pos[j]+=rate[j]*dm+PN*np.random.randn()\n            tvt_j=pos[j]-z_v[i]\n            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)\n            pos[j]=tvt_j+z_v[i]\n        if not np.isnan(gr_v[i]):\n            ws=0.\n            for j in range(N):\n                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)\n                d=(gr_v[i]-eg)/gs\n                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)\n                w[j]*=lk; ws+=w[j]\n            if ws>0.:\n                for j in range(N): w[j]/=ws\n            else:\n                for j in range(N): w[j]=1./N\n        ne=0.\n        for j in range(N): ne+=w[j]*w[j]\n        if 1./ne<RESAMP*N:\n            pos,rate=_resamp(pos,rate,w,N,RP,RR)\n            for j in range(N): w[j]=1./N\n        tv=0.\n        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])\n        pts[i]=tv; va=0.\n        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2\n        std_[i]=va**0.5; pm=md_v[i]\n    return pts,std_\n\n@njit(cache=False)\ndef _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,\n          gs,ip,iv,beta,icpt,zsig,N,\n          MOM,VN,PN,GR_WT,RP,RV,RESAMP):\n    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N\n    for j in range(N):\n        pos[j]=ip+0.5*np.random.randn()\n        vel[j]=iv+0.02*np.random.randn()\n    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.\n    for i in range(len(md_v)):\n        dm=md_v[i]-pm; dm=max(dm,1.)\n        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt\n        for j in range(N):\n            vel[j]=MOM*vel[j]+VN*np.random.randn()\n            pos[j]+=vel[j]*dm+PN*np.random.randn()\n            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)\n        if not np.isnan(gr_v[i]):\n            ws=0.\n            for j in range(N):\n                ep=_interp1(gg_p,pos[j],vmin,step)\n                dp=(gr_v[i]-ep)/gs\n                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)\n                if not np.isnan(gr_sm_v[i]):\n                    es=_interp1(gg_s,pos[j],vmin,step)\n                    ds=(gr_sm_v[i]-es)/(gs*1.5)\n                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)\n                    lk=(1.-GR_WT)*lp+GR_WT*ls\n                else: lk=lp\n                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]\n            if ws>0.:\n                for j in range(N): w[j]/=ws\n            else:\n                for j in range(N): w[j]=1./N\n        ws2=0.\n        for j in range(N):\n            dv=(vel[j]-ve)/max(zsig*2.,0.005)\n            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)\n            w[j]*=lz; ws2+=w[j]\n        if ws2>0.:\n            for j in range(N): w[j]/=ws2\n        else:\n            for j in range(N): w[j]=1./N\n        ne=0.\n        for j in range(N): ne+=w[j]*w[j]\n        if 1./ne<RESAMP*N:\n            pos,vel=_resamp(pos,vel,w,N,RP,RV)\n            for j in range(N): w[j]=1./N\n        wm=0.\n        for j in range(N): wm+=w[j]*pos[j]\n        pts[i]=wm; va=0.\n        for j in range(N): va+=w[j]*(pos[j]-wm)**2\n        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]\n    return pts,std_\n\n# Dense grid for O(1) typewell lookup\ndef _grid(tw_tvt,tw_gr,step=0.2):\n    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())\n    tvt_g=np.arange(tmin,tmax+step,step)\n    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)\n\ndef _gr_sig(hw,tw_tvt,tw_gr):\n    kn=hw[hw[\'TVT_input\'].notna()&hw[\'GR\'].notna()]\n    if len(kn)<20: return float(PF_GR_SIG_DEF)\n    return float(np.clip(np.std(kn[\'GR\'].values-np.interp(kn[\'TVT_input\'].values,tw_tvt,tw_gr)),\n                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))\n\ndef _nn(arr,v):\n    i=int(np.searchsorted(arr,v,\'left\'))\n    if i>=len(arr): return len(arr)-1\n    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1\n    return i\n\ndef _smooth(vals,fb,r):\n    s=pd.Series(vals,dtype=\'float32\').interpolate(limit_direction=\'both\').fillna(fb)\n    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)\n\ndef beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):\n    si=_nn(tw_tvt,start_tvt)\n    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)\n    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))\n    return tw_tvt[path].astype(np.float32)\n\ndef run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):\n    gs=_gr_sig(hw,tw_tvt,tw_gr)\n    kn=hw[hw[\'TVT_input\'].notna()]; ev=hw[hw[\'TVT_input\'].isna()]\n    if len(ev)==0: return np.array([]),np.array([])\n    ls=float(kn[\'TVT_input\'].iloc[-1]+kn[\'Z\'].iloc[-1])\n    tail=kn.tail(30); dt=np.diff(tail[\'TVT_input\'].values)\n    dz=np.diff(tail[\'Z\'].values); dm=np.diff(tail[\'MD\'].values); m=dm>0\n    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.\n    gg,gmin,gst=_grid(tw_tvt,tw_gr)\n    pts,std=_pf_ancc(ev[\'MD\'].values.astype(np.float64),ev[\'Z\'].values.astype(np.float64),\n                      ev[\'GR\'].values.astype(np.float64),gg,gmin,gst,\n                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)\n    return pts.astype(np.float32),std.astype(np.float32)\n\ndef run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):\n    gs=_gr_sig(hw,tw_tvt,tw_gr)\n    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)\n    kna=hw[hw[\'TVT_input\'].notna()]; ev=hw[hw[\'TVT_input\'].isna()]\n    if len(ev)==0: return np.array([]),np.array([])\n    dz_k=np.diff(kna[\'Z\'].values); dvt=np.diff(kna[\'TVT_input\'].values)\n    dmd_k=np.diff(kna[\'MD\'].values); m2=dmd_k>0\n    if m2.sum()>=10:\n        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]\n        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)\n        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)\n    else: beta,icpt,zsig=-1.,0.,0.1\n    t2=kna.tail(20); dvt2=np.diff(t2[\'TVT_input\'].values); dmd2=np.diff(t2[\'MD\'].values); m3=dmd2>0\n    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.\n    gg,gmin,gst=_grid(tw_tvt,tw_gr)\n    gs2,_,_=_grid(tw_tvt,tw_s)\n    gr_sm=hw[\'GR\'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()\n    pts,std=_pf_z(ev[\'MD\'].values.astype(np.float64),ev[\'Z\'].values.astype(np.float64),\n                   ev[\'GR\'].values.astype(np.float64),\n                   gr_sm.loc[ev.index].values.astype(np.float64),\n                   gg,gs2,gmin,gst,gs,float(kna[\'TVT_input\'].iloc[-1]),iv,\n                   beta,icpt,zsig,N,\n                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)\n    return pts.astype(np.float32),std.astype(np.float32)\n\n\n_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)\n_gg=np.linspace(45,55,100,np.float64)\n_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)\n_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)\n_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)\n\ndef robust_slope(x,y,w=None):\n    x=np.asarray(x,float); y=np.asarray(y,float)\n    m=np.isfinite(x)&np.isfinite(y)\n    if m.sum()<2 or np.std(x[m])<1e-6: return 0.\n    return float(np.polyfit(x[m],y[m],1)[0])\n\ndef affine_cal(kgr,tw_at_k,min_pts=20):\n    v=np.isfinite(kgr)&np.isfinite(tw_at_k)\n    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:\n        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.\n    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)\n\ndef seg_b_well(ktvt,kz,form_col):\n    """Segment b_well: early/mid/late thirds + full prefix.\n    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""\n    bv=ktvt+kz-form_col; n=len(bv)\n    b_full=float(np.median(bv))\n    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full\n    t1,t2=n//3, 2*n//3\n    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full\n    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full\n    # WLS (tail-upweighted)\n    w=np.exp(0.02*np.arange(n)); w/=w.sum()\n    b_wls=float(np.dot(w,bv))\n    return b_full,b_early,b_mid,b_late,b_wls\n\ndef multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):\n    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""\n    out=[]\n    for hw in hws:\n        win=2*hw+1; nk=len(kgr); nh=len(hgr)\n        if nk<win+1 or nh==0:\n            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue\n        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)\n        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)\n        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)\n        if M==0:\n            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue\n        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)\n        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)\n        hp=np.pad(hg,hw,mode=\'edge\')\n        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)\n        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)\n        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)\n        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))\n    # Score-weighted ensemble (NEW: softmax-weighted combination)\n    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)\n    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9\n    sc_ens=(tvts*sw).sum(1).astype(np.float32)\n    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble\n\nclass FormationPlaneKNN:\n    def __init__(self,well_ids,data_dir):\n        rows=[]\n        for wid in well_ids:\n            p=data_dir/f\'{wid}__horizontal_well.csv\'\n            try: df=pd.read_csv(p,usecols=[\'X\',\'Y\']+FORMATIONS).dropna()\n            except: continue\n            if len(df)==0: continue\n            row={\'wid\':wid,\'x\':float(df[\'X\'].median()),\'y\':float(df[\'Y\'].median())}\n            for c in FORMATIONS: row[f\'{c}_m\']=float(df[c].median())\n            rows.append(row)\n        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df[\'wid\'])}\n        xy=self.df[[\'x\',\'y\']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))\n        self.tree=cKDTree(xy/self.scale)\n        self.xa=self.df[\'x\'].to_numpy(); self.ya=self.df[\'y\'].to_numpy()\n        self.fa=self.df[[f\'{c}_m\' for c in FORMATIONS]].to_numpy(np.float64)\n\n    def impute(self,xy_q,self_wid=None,k=PLANE_K):\n        q=xy_q/self.scale; nf=min(k+5,len(self.df))\n        dist,idx=self.tree.query(q,k=nf,workers=-1)\n        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)\n        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]\n        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)\n        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)\n        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn\n        A=np.zeros((len(q),3,3))\n        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)\n        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)\n        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)\n        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9\n        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)\n        try: coef=np.linalg.solve(A,rhs)\n        except:\n            coef=np.zeros((len(q),3,6))\n            for r in range(len(q)):\n                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]\n                except: pass\n        Xq=xy_q[:,0]; Yq=xy_q[:,1]\n        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)\n        pred[~vk.any(1)]=self.fa.mean(0)\n        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)\n\nclass DenseANCCImputer:\n    def __init__(self,well_ids,data_dir,spw=DENSE_SPW):\n        xs,ys,anccs,wids=[],[],[],[]\n        for wid in well_ids:\n            p=data_dir/f\'{wid}__horizontal_well.csv\'\n            try: df=pd.read_csv(p,usecols=[\'X\',\'Y\',\'ANCC\']).dropna()\n            except: continue\n            if len(df)==0: continue\n            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]\n            xs.append(s[\'X\'].values); ys.append(s[\'Y\'].values)\n            anccs.append(s[\'ANCC\'].values); wids.extend([wid]*len(s))\n        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])\n        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)\n        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))\n        self.tree=cKDTree(self.xy/self.scale)\n\n    def impute(self,xy_q,self_wid=None,k=DENSE_K,nfetch=5000):\n        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))\n        dist,idx=self.tree.query(q,k=nf,workers=-1)\n        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)\n        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]\n        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)\n        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)\n        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]\n        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)\n        var=((an-ap[:,None])**2*w).sum(1)/safe\n        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)\n\n_FI=None\n_DI=None\n\ndef init_spatial_imputers():\n    global _FI, _DI\n    if _FI is not None and _DI is not None:\n        return _FI, _DI\n    hw_paths=sorted((CFG.dataset_path / "train").glob(\'*__horizontal_well.csv\'))\n    train_wids=[p.stem.replace(\'__horizontal_well\',\'\') for p in hw_paths]\n    _FI=FormationPlaneKNN(train_wids,CFG.dataset_path / "train")\n    _DI=DenseANCCImputer(train_wids,CFG.dataset_path / "train")\n    return _FI, _DI\n\nANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)\nBEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)\nSC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)\nPF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)\n\ndef build_well(hw_path,tw_path,is_train):\n    global _FI,_DI\n    init_spatial_imputers()\n    wid=Path(hw_path).stem.replace(\'__horizontal_well\',\'\')\n    try:\n        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values(\'TVT\')\n    except: return None\n    if is_train and \'TVT\' not in hw.columns: return None\n    kn=hw[hw[\'TVT_input\'].notna()]; ev=hw[hw[\'TVT_input\'].isna()]\n    if len(ev)==0 or len(kn)<10: return None\n    if is_train and hw[\'TVT\'].isna().all(): return None\n    tw_tvt=tw[\'TVT\'].to_numpy(np.float32); tw_gr=tw[\'GR\'].to_numpy(np.float32)\n    if len(tw_tvt)<3: return None\n\n    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)\n    if len(pf_a)==0: return None\n    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)\n    pf_use=pf_a.astype(np.float32); std_use=std_a.astype(np.float32)\n    has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))\n\n    lk=kn.iloc[-1]; last_tvt=float(lk[\'TVT_input\'])\n    gr_full=hw[\'GR\'].astype(float).interpolate(limit_direction=\'both\').fillna(float(np.nanmean(tw_gr)))\n    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)\n    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)\n\n    # 7 beams (Numba JIT ±2)\n    bpaths={}\n    for (bs,mc,es,r,tag) in BEAMS:\n        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)\n    beam_ref=(bpaths[\'cons\']+bpaths[\'sm5\'])/2.\n\n    # Multi-scale NCC → score-weighted ensemble\n    ktvt=kn[\'TVT_input\'].to_numpy(np.float32)\n    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)\n    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]\n    sc_cons=(sc8+sc15+sc25)/3.\n    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))\n    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single\n\n    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)\n    a_cal,b_cal=affine_cal(kgr,tw_at_k)\n    kmd=kn[\'MD\'].to_numpy(np.float32); kz=kn[\'Z\'].to_numpy(np.float32)\n    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))\n    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])\n    slp_z=robust_slope(kz,ktvt)\n\n    swid=wid if is_train else None\n    xy_ev=ev[[\'X\',\'Y\']].to_numpy(np.float64); xy_kn=kn[[\'X\',\'Y\']].to_numpy(np.float64)\n    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)\n    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)\n    z_kn=kn[\'Z\'].to_numpy(np.float32); z_ev=ev[\'Z\'].to_numpy(np.float32)\n\n    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE\n    tvt_fs={}; form_rmse={}; form_list=[]\n    for fi2,fn in enumerate(FORMATIONS):\n        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])\n        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)\n        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)\n        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)\n        tvt_fs[f\'tvtF_{fn}\']=tvt_f; tvt_fs[f\'tvtFw_{fn}\']=tvt_fw\n        tvt_fs[f\'tvtF50_{fn}\']=tvt_f50\n        tvt_fs[f\'bw_{fn}\']=np.float32(b_full); tvt_fs[f\'bww_{fn}\']=np.float32(b_wls)\n        tvt_fs[f\'bw50_{fn}\']=np.float32(b_late)\n        tvt_fs[f\'bw_early_{fn}\']=np.float32(b_early)   # NEW: early segment\n        tvt_fs[f\'bw_mid_{fn}\']=np.float32(b_mid)       # NEW: mid segment\n        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))\n        form_list.append(tvt_f)\n\n    fs=np.stack(form_list,1)\n    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)\n    form_std_d =fs.std(1).astype(np.float32)\n    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)\n\n    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)\n    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)\n    b_vd=ktvt+z_kn-d_kn\n    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)\n    b_d=float(np.median(b_vd))\n    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)\n    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)\n    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)\n    res_kn=ktvt+z_kn-d_kn\n    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))\n\n    all_sigs=[pf_use]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs[\'tvtF_ANCC\'],tvt_dense]\n    sig_mat=np.stack(all_sigs,1)\n    sig_std=sig_mat.std(1).astype(np.float32)\n    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)\n\n    gr_s=pd.Series(gr_full.values); rolls={}\n    for w in [5,21,51,101]:\n        r=gr_s.rolling(w,center=True,min_periods=1)\n        rolls[f\'grm{w}\']=r.mean().iloc[ev.index].values.astype(np.float32)\n        rolls[f\'grs{w}\']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)\n    for lag in [1,5,15,30]:\n        rolls[f\'glag{lag}\']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)\n        rolls[f\'glead{lag}\']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)\n    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)\n    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)\n    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)\n    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)\n                   ).iloc[ev.index].values.astype(np.float32)\n\n    hmd=ev[\'MD\'].to_numpy(np.float32); md_since=hmd-float(lk[\'MD\'])\n    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)\n    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)\n\n    mdd=hw[\'MD\'].diff().replace(0,np.nan)\n    dzdmd=(hw[\'Z\'].diff()/mdd).iloc[ev.index].values.astype(np.float32)\n    dxdmd=(hw[\'X\'].diff()/mdd).iloc[ev.index].values.astype(np.float32)\n    dydmd=(hw[\'Y\'].diff()/mdd).iloc[ev.index].values.astype(np.float32)\n\n    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)\n    def sc(v): return np.full(nh,np.float32(v),np.float32)\n\n    feats={\n        \'well\':wid,\'id\':[f\'{wid}_{i}\' for i in ev.index],\n        \'last_known_tvt\':sc(last_tvt),\n        \'pf_ancc\':pf_use,\'pf_ancc_std\':std_use,\n        \'pf_ancc_delta\':(pf_use-last_tvt).astype(np.float32),\n        \'pf_z\':(pf_z.astype(np.float32) if has_z else sc(last_tvt)),\n        \'pf_z_delta\':((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),\n        \'pf_vs_z\':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),\n        **{f\'beam_{t}_d\':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},\n        \'beam_mean_d\':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),\n        \'beam_std_d\': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),\n        \'beam_med_d\': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),\n        \'sc8_d\':(sc8-np.float32(last_tvt)).astype(np.float32),\'sc8_sc\':sc8s,\n        \'sc15_d\':(sc15-np.float32(last_tvt)).astype(np.float32),\'sc15_sc\':sc15s,\n        \'sc25_d\':(sc25-np.float32(last_tvt)).astype(np.float32),\'sc25_sc\':sc25s,\n        \'sc_cons_d\':(sc_cons-np.float32(last_tvt)).astype(np.float32),\n        \'sc_ens_d\':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble\n        \'sc_trust\':sc(sc_trust),\'hyb_d\':(hyb_ref-np.float32(last_tvt)).astype(np.float32),\n        \'sig_std\':sig_std,\'sig_mean_d\':sig_mean,\n        **tvt_fs,\n        **{f\'frm_rmse_{fn}\':sc(form_rmse[fn]) for fn in FORMATIONS},\n        \'form_mean_d\':form_mean_d,\'form_std_d\':form_std_d,\'form_rng_d\':form_rng_d,\n        \'spatial_ancc_d\':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),\n        \'spatial_knn_dist\':knn_d,\n        \'dense_ancc\':d_ancc,\'dense_std\':d_std,\'dense_dist\':d_dist,\n        \'tvt_dense_d\' :(tvt_dense -last_tvt).astype(np.float32),\n        \'tvt_densew_d\':(tvt_densew-last_tvt).astype(np.float32),\n        \'tvt_dense50_d\':(tvt_dense50-last_tvt).astype(np.float32),\n        \'dense_rmse\':sc(d_rmse),\'dense_bias\':sc(d_bias),\'dense_nb_std\':sc(d_nb_std),\n        \'pf_vs_spatial\':(pf_use-tvt_fs[\'tvtF_ANCC\']).astype(np.float32),\n        \'pf_vs_dense\':(pf_use-tvt_dense).astype(np.float32),\n        \'spatial_vs_dense\':(tvt_fs[\'tvtF_ANCC\']-tvt_dense).astype(np.float32),\n        \'beam_vs_spatial\':(bpaths[\'cons\']-tvt_fs[\'tvtF_ANCC\']).astype(np.float32),\n        \'sc_vs_beam\':(sc_ens-bpaths[\'cons\']).astype(np.float32),\n        \'cal_a\':sc(a_cal),\'cal_b\':sc(b_cal),\n        \'pfx_rmse\':sc(pfx_rmse),\'known_len\':sc(len(kn)),\'eval_len\':sc(nh),\n        \'slp_all\':sc(slp_all),\'slp_50\':sc(slp_50),\'slp_z\':sc(slp_z),\n        \'slp_b_d_all\':(slp_b_all-last_tvt).astype(np.float32),\n        \'slp_b_d_50\': (slp_b_50 -last_tvt).astype(np.float32),\n        \'ktvt_range\':sc(float(np.ptp(ktvt))),\'ktvt_std\':sc(float(ktvt.std())),\n        \'md_since\':md_since,\'frac\':frac,\'frac2\':frac**2,\'sqrt_frac\':np.sqrt(frac),\n        \'z\':z_ev,\n        \'dx\':(ev[\'X\']-float(lk[\'X\'])).to_numpy(np.float32),\n        \'dy\':(ev[\'Y\']-float(lk[\'Y\'])).to_numpy(np.float32),\n        \'dz\':(z_ev-float(lk[\'Z\'])).astype(np.float32),\n        \'dxy\':np.sqrt((ev[\'X\']-float(lk[\'X\']))**2+(ev[\'Y\']-float(lk[\'Y\']))**2).to_numpy(np.float32),\n        \'dzdmd\':dzdmd,\'dxdmd\':dxdmd,\'dydmd\':dydmd,\n        \'gr\':hgr,\'gr_d1\':gr_d1,\'gr_d2\':gr_d2,\'gr_env\':gr_env,\'gr_nrg\':gr_nrg,\n        \'gr_vs_tw_anc\':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),\n        \'gr_vs_slp_all\':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),\n        **{f\'tda{int(o)}\' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},\n        **{f\'tdbc{int(o)}\':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},\n        **{f\'tdsc{int(o)}\':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},\n        **{f\'tdpf{int(o)}\':hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},\n        \'tw_range\':sc(float(np.ptp(tw_tvt))),\'tw_gr_mean\':sc(float(tw_gr.mean())),\n    }\n    for k,v in rolls.items(): feats[k]=v\n    result=pd.DataFrame(feats)\n    if is_train:\n        if \'TVT\' not in ev.columns or ev[\'TVT\'].isna().all(): return None\n        result[\'target\']=(ev[\'TVT\'].to_numpy(np.float32)-np.float32(last_tvt))\n    return result\n\ndef build_dataset(paths,is_train,label):\n    args=[(str(p),str(p.parent/f\'{p.stem.replace("__horizontal_well","")}__typewell.csv\'),is_train)\n          for p in paths\n          if (p.parent/f\'{p.stem.replace("__horizontal_well","")}__typewell.csv\').exists()]\n    t0=time.time()\n    res=Parallel(n_jobs=NCPU,prefer=\'threads\',verbose=3)(\n        delayed(build_well)(hp,tp,it) for hp,tp,it in args)\n    parts=[r for r in res if r is not None]\n    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()\n\n\n# ==================== Inference-only kernel entrypoint ====================\n#!/usr/bin/env python3\n"""Infer-only P0 soft055 w060 F-rect submission.\n\nLoads public Ravaghi trained base models from the artifact dataset, applies the\nlocally frozen second-level Ridge fold ensemble, blends it with a fixed\nstruct_hybrid + softobs055 PF source average at pf_scale_3, then applies the\ndeterministic F=TVT+Z rectification used in the local P0 CV audit.  This\nintentionally performs no model training on Kaggle.\n"""\nimport os\nfrom pathlib import Path\nimport sys\nimport types\nimport warnings\nimport zlib\n\nwarnings.filterwarnings("ignore")\nos.environ.setdefault("OMP_NUM_THREADS", "4")\n\nimport joblib\nimport numpy as np\nimport pandas as pd\nfrom scipy.spatial import cKDTree\nfrom scipy.signal import savgol_filter\nfrom sklearn.metrics import mean_squared_error\n\nrl = sys.modules[__name__]\n\n\nDATA_ROOT = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")\nif not DATA_ROOT.exists():\n    DATA_ROOT = Path("/kaggle/input/rogii-wellbore-geology-prediction")\nif not DATA_ROOT.exists():\n    DATA_ROOT = Path(os.environ.get("ROGII_DATA_DIR", "/home/yiyu/rogii/datasets/rogii-wellbore-geology-prediction"))\n\nART_ROOT = Path("/kaggle/input/wellbore-geology-prediction-artifacts")\nif not ART_ROOT.exists():\n    ART_ROOT = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")\nif not ART_ROOT.exists():\n    ART_ROOT = Path(os.environ.get(\n        "RAVAGHI_ARTIFACT_DIR",\n        "/home/yiyu/rogii/datasets/ravaghi_wellbore_geology_prediction_artifacts_kaggle_current",\n    ))\n\nONKAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))\nOUT_PATH = Path("/kaggle/working/submission.csv") if ONKAGGLE else Path("submission_levelcons_p0chord_rate075_starttight_local.csv")\n\nMODEL_DIRS = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]\n\nRIDGE_INTERCEPTS = np.array(\n    [-0.18320775032043457, -0.19101393222808838, -0.224159836769104, -0.191192626953125, -0.14542460441589355],\n    dtype=np.float32,\n)\nRIDGE_COEFS = np.array(\n    [\n        [0.0, 0.2911567986011505, 0.38378193974494934, 0.08868265897035599, 0.3258693218231201],\n        [0.053661931306123734, 0.39128851890563965, 0.5169801712036133, 0.0, 0.15848520398139954],\n        [0.0, 0.31343886256217957, 0.48635944724082947, 0.05501436069607735, 0.22994305193424225],\n        [0.0, 0.3842366635799408, 0.44450387358665466, 0.07253403216600418, 0.18571855127811432],\n        [0.0, 0.23798350989818573, 0.5057194232940674, 0.12051041424274445, 0.21425989270210266],\n    ],\n    dtype=np.float32,\n)\nPP_ALPHA = 1.0\nPP_TAU = 85.0\nPP_W_PF = 0.09\nFINAL_W_RIDGE = 0.40\nFINAL_W_PF = 0.60\nSTRUCT_SEED = 1745\nSTRUCT_N_SEEDS = int(os.environ.get("ROGII_STRUCT_N_SEEDS", "32"))\nSTRUCT_N_PARTICLES = int(os.environ.get("ROGII_STRUCT_N_PARTICLES", "300"))\nRATE075_N_SEEDS = int(os.environ.get("ROGII_RATE075_N_SEEDS", "16"))\nRATE075_N_PARTICLES = int(os.environ.get("ROGII_RATE075_N_PARTICLES", "120"))\nSTRUCT_RATE_OFFSETS = (-0.004, 0.0, 0.004)\nSTRUCT_RATE_CLIP = 0.08\nSTRUCT_SUBW = 60\nSTRUCT_KQ = 60\nSTRUCT_GRID_STEP = 0.5\nSTRUCT_MOM = 0.998\nSTRUCT_VN = 0.002\nSTRUCT_PN = 0.005\nSTRUCT_RP = 0.1\nSTRUCT_RR = 0.001\nSTRUCT_RESAMP = 0.5\nSTRUCT_OBS_POWER = 1.0\nSOFTOBS_OBS_POWER = 0.55\nSOFTOBS_RESAMP = 0.35\nSTRUCT_DEPLOY_SCALE = "pf_scale_3"\nSTRUCT_SCALES = (3.0, 5.0, 8.0, 12.0)\nFRECT_FRAC = 0.18\n_STRUCT_CLOUD = None\n\n\nclass DummyTrainer:\n    def __setstate__(self, state):\n        self.__dict__.update(state)\n\n    def predict(self, X):\n        preds = [m.predict(X) for m in self.estimators]\n        return np.mean(np.stack(preds, axis=0), axis=0)\n\n\ndef install_unpickle_shims() -> None:\n    koolbox = types.ModuleType("koolbox")\n    trainer_pkg = types.ModuleType("koolbox.trainer")\n    trainer_mod = types.ModuleType("koolbox.trainer.trainer")\n    koolbox.Trainer = DummyTrainer\n    trainer_pkg.Trainer = DummyTrainer\n    trainer_mod.Trainer = DummyTrainer\n    sys.modules.setdefault("koolbox", koolbox)\n    sys.modules.setdefault("koolbox.trainer", trainer_pkg)\n    sys.modules.setdefault("koolbox.trainer.trainer", trainer_mod)\n    try:\n        import sklearn.metrics._regression as reg\n\n        if not hasattr(reg, "root_mean_squared_error"):\n            reg.root_mean_squared_error = lambda y, p, **kw: mean_squared_error(y, p, **kw) ** 0.5\n    except Exception:\n        pass\n\n\ndef load_trainer(name: str) -> DummyTrainer:\n    paths = sorted((ART_ROOT / "models" / name).glob("*.pkl"))\n    if not paths:\n        # Older artifact layout fallback.\n        paths = sorted((ART_ROOT / "models" / name).glob("models.pkl"))\n    if not paths:\n        raise FileNotFoundError(f"No artifact model found for {name} under {ART_ROOT}")\n    obj = joblib.load(paths[0])\n    if hasattr(obj, "estimators"):\n        return obj\n    if isinstance(obj, list):\n        wrapper = DummyTrainer()\n        wrapper.estimators = obj\n        return wrapper\n    raise TypeError(f"Unsupported model artifact for {name}: {type(obj)}")\n\n\ndef ridge_fold_ensemble(base_pred: np.ndarray) -> np.ndarray:\n    fold_preds = base_pred @ RIDGE_COEFS.T + RIDGE_INTERCEPTS[None, :]\n    return fold_preds.mean(axis=1).astype(np.float32)\n\n\ndef apply_pp(df: pd.DataFrame, model_drift: np.ndarray, pf_drift: np.ndarray) -> np.ndarray:\n    d = model_drift * (1.0 - PP_W_PF) + pf_drift * PP_W_PF\n    d = d * (1.0 - np.exp(-np.maximum(df["md_since"].to_numpy(float), 0.0) / PP_TAU))\n    return (d * PP_ALPHA).astype(np.float32)\n\n\ndef sg_smooth(df: pd.DataFrame, col: str, sg_w: int = 17, sg_p: int = 3) -> pd.DataFrame:\n    df = df.copy()\n    for _, g in df.groupby("well", sort=False):\n        v = g[col].to_numpy(np.float32)\n        wl = min(sg_w, len(v))\n        if wl % 2 == 0:\n            wl -= 1\n        if wl >= sg_p + 2:\n            v = savgol_filter(v, wl, sg_p)\n        df.loc[g.index, col] = v\n    return df\n\n\ndef moving_average_edge(x: np.ndarray, window: int) -> np.ndarray:\n    if window <= 1 or len(x) <= 2:\n        return x.astype(np.float32, copy=True)\n    window = int(min(window, len(x)))\n    if window % 2 == 0:\n        window -= 1\n    if window <= 1:\n        return x.astype(np.float32, copy=True)\n    pad = window // 2\n    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")\n    ker = np.ones(window, dtype=np.float64) / float(window)\n    return np.convolve(xp, ker, mode="valid").astype(np.float32)\n\n\ndef apply_f_rectification(sub: pd.DataFrame, frac: float = FRECT_FRAC) -> pd.DataFrame:\n    out = sub.copy()\n    vals = out["tvt"].to_numpy(np.float64, copy=True)\n    for _, idx in out.groupby("well", sort=False).groups.items():\n        pos = np.asarray(list(idx), dtype=np.int64)\n        if len(pos) <= 2:\n            continue\n        z = out.loc[pos, "z"].to_numpy(np.float64)\n        tvt = vals[pos]\n        win = max(5, int(round(len(pos) * float(frac))))\n        if win % 2 == 0:\n            win += 1\n        vals[pos] = moving_average_edge(tvt + z, win).astype(np.float64) - z\n    out["tvt"] = vals.astype(np.float32)\n    return out\n\n\nCHORD_W = 0.10  # transfer-safe rate-persistence prior (chord de-bend). Mirrors cv/rate_persistence_prior.py\ndef apply_chord_shrink(sub: pd.DataFrame, w: float = CHORD_W) -> pd.DataFrame:\n    """Fixed global de-bend in F=TVT+Z space toward the chord connecting hidden\n    start and end (removes spurious extrapolated curvature; preserves net drift).\n    No fitting. Per well:  F_corr = (1-w)*F_pred + w*[F_start + r_ref*(md-md0)],\n    r_ref = (F_end-F_start)/(md_end-md0).  Identity at w=0."""\n    out = sub.copy()\n    vals = out["tvt"].to_numpy(np.float64, copy=True)\n    md_all = out["md_since"].to_numpy(np.float64)  # md - lk_MD; chord only needs (md - md0)\n    z_all = out["z"].to_numpy(np.float64)\n    for _, idx in out.groupby("well", sort=False).groups.items():\n        pos = np.asarray(list(idx), dtype=np.int64)\n        if len(pos) < 5:\n            continue\n        order = np.argsort(md_all[pos], kind="stable")\n        pos = pos[order]\n        md = md_all[pos]; z = z_all[pos]; tvt = vals[pos]\n        f_pred = tvt + z\n        span = md[-1] - md[0]\n        if not np.isfinite(span) or span <= 1.0:\n            continue\n        r_ref = (f_pred[-1] - f_pred[0]) / span\n        f_ref = f_pred[0] + r_ref * (md - md[0])\n        f_corr = (1.0 - w) * f_pred + w * f_ref\n        vals[pos] = f_corr - z\n    out["tvt"] = vals.astype(np.float32)\n    return out\n\n\ndef load_well(wid: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:\n    hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")\n    tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv")\n    return hw, tw\n\n\ndef struct_robust_slope(y: np.ndarray, md: np.ndarray) -> float | None:\n    dy = np.diff(y.astype(float))\n    dm = np.diff(md.astype(float))\n    ok = np.isfinite(dy) & np.isfinite(dm) & (dm > 0)\n    if ok.sum() < 3:\n        return None\n    vals = dy[ok] / dm[ok]\n    vals = vals[np.isfinite(vals)]\n    if len(vals) < 3:\n        return None\n    lo, hi = np.nanquantile(vals, [0.05, 0.95])\n    vals = vals[(vals >= lo) & (vals <= hi)]\n    if len(vals) == 0:\n        return None\n    return float(np.nanmedian(vals))\n\n\ndef struct_robust_rate(tvt: np.ndarray, z: np.ndarray, md: np.ndarray) -> float | None:\n    return struct_robust_slope(tvt.astype(float) + z.astype(float), md.astype(float))\n\n\ndef init_struct_cloud() -> dict[str, object]:\n    global _STRUCT_CLOUD\n    if _STRUCT_CLOUD is not None:\n        return _STRUCT_CLOUD\n    rows = []\n    wells = []\n    train_dir = DATA_ROOT / "train"\n    for path in sorted(train_dir.glob("*__horizontal_well.csv")):\n        wid = path.name.split("__", 1)[0]\n        try:\n            df = pd.read_csv(path, usecols=["X", "Y", "ANCC"])\n        except Exception:\n            continue\n        if len(df) == 0 or "ANCC" not in df.columns:\n            continue\n        anc = df["ANCC"].interpolate(limit_direction="both").to_numpy(np.float64)\n        idx = np.linspace(0, len(df) - 1, min(STRUCT_SUBW, len(df))).astype(int)\n        block = np.column_stack(\n            [\n                df["X"].to_numpy(np.float64)[idx],\n                df["Y"].to_numpy(np.float64)[idx],\n                anc[idx],\n            ]\n        )\n        ok = np.isfinite(block).all(axis=1)\n        if ok.any():\n            rows.append(block[ok])\n            wells.extend([wid] * int(ok.sum()))\n    if not rows:\n        raise RuntimeError(f"no ANCC structural cloud rows found under {train_dir}")\n    cloud = np.vstack(rows)\n    _STRUCT_CLOUD = {\n        "xy": cloud[:, :2],\n        "s": cloud[:, 2],\n        "well": np.asarray(wells, dtype=object),\n        "tree": cKDTree(cloud[:, :2]),\n    }\n    print(f"struct cloud rows={len(cloud)} wells={len(set(wells))}", flush=True)\n    return _STRUCT_CLOUD\n\n\ndef struct_local_plane_s(xy: np.ndarray, self_well: str) -> np.ndarray:\n    cloud = init_struct_cloud()\n    tree: cKDTree = cloud["tree"]  # type: ignore[assignment]\n    cloud_xy = np.asarray(cloud["xy"], dtype=np.float64)\n    cloud_s = np.asarray(cloud["s"], dtype=np.float64)\n    cloud_well = np.asarray(cloud["well"], dtype=object)\n    k = min(STRUCT_KQ + STRUCT_SUBW + 5, len(cloud_s))\n    _, idx = tree.query(np.asarray(xy, dtype=np.float64), k=k, workers=1)\n    if idx.ndim == 1:\n        idx = idx[:, None]\n    out = np.empty(len(xy), dtype=np.float64)\n    for i, ji0 in enumerate(idx):\n        ji = ji0[(cloud_well[ji0] != self_well) & np.isfinite(cloud_s[ji0])][:STRUCT_KQ]\n        if len(ji) < 3:\n            vals = cloud_s[ji0][np.isfinite(cloud_s[ji0])]\n            out[i] = float(np.nanmean(vals)) if len(vals) else np.nan\n            continue\n        mat = np.column_stack([cloud_xy[ji, 0], cloud_xy[ji, 1], np.ones(len(ji))])\n        try:\n            coef, *_ = np.linalg.lstsq(mat, cloud_s[ji], rcond=None)\n            out[i] = float(coef[0] * xy[i, 0] + coef[1] * xy[i, 1] + coef[2])\n        except Exception:\n            out[i] = float(np.nanmean(cloud_s[ji]))\n    return out\n\n\ndef struct_rates(hw: pd.DataFrame, wid: str) -> list[float]:\n    known = hw[hw["TVT_input"].notna()]\n    hs = int(len(known))\n    n = int(hw["TVT_input"].isna().sum())\n    if n < 5:\n        return []\n    idx = np.linspace(hs, len(hw) - 1, min(240, n)).astype(int)\n    xy = hw.iloc[idx][["X", "Y"]].to_numpy(np.float64)\n    s = struct_local_plane_s(xy, wid)\n    md = hw.iloc[idx]["MD"].to_numpy(np.float64)\n    spans: list[tuple[int, int]] = [(0, len(idx))]\n    for frac0, frac1 in [(0.0, 0.25), (0.0, 0.5), (0.25, 0.75), (0.5, 1.0), (0.75, 1.0)]:\n        a = int(frac0 * len(idx))\n        b = max(int(frac1 * len(idx)), a + 5)\n        b = min(len(idx), b)\n        if b - a >= 5:\n            spans.append((a, b))\n    rates = []\n    for a, b in spans:\n        r = struct_robust_slope(s[a:b], md[a:b])\n        if r is not None:\n            rates.append(r)\n    out: list[float] = []\n    for r in rates:\n        if np.isfinite(r) and not any(abs(r - q) < 1e-6 for q in out):\n            out.append(float(r))\n    return out\n\n\ndef struct_rate_bank(hw: pd.DataFrame, wid: str) -> np.ndarray:\n    known = hw[hw["TVT_input"].notna()]\n    hs = int(len(known))\n    tvt = hw["TVT_input"].to_numpy(float)\n    z = hw["Z"].to_numpy(float)\n    md = hw["MD"].to_numpy(float)\n    rates = []\n    for win in [30, 60, 120, 300, 800, hs]:\n        start = max(0, hs - int(win))\n        r = struct_robust_rate(tvt[start:hs], z[start:hs], md[start:hs])\n        if r is not None:\n            rates.append(r)\n    if not rates:\n        rates = [0.0]\n    bank = []\n    for rate in list(rates[:2]) + list(struct_rates(hw, wid)[:4]):\n        for off in STRUCT_RATE_OFFSETS:\n            bank.append(rate + off)\n    bank = [float(np.clip(r, -STRUCT_RATE_CLIP, STRUCT_RATE_CLIP)) for r in bank if np.isfinite(r)]\n    if not bank:\n        bank = [float(np.clip(rates[0], -STRUCT_RATE_CLIP, STRUCT_RATE_CLIP))]\n    out = []\n    for r in bank:\n        if not any(abs(r - q) < 1e-6 for q in out):\n            out.append(r)\n    return np.asarray(out, dtype=np.float64)\n\n\ndef run_struct_particle_filter(\n    hw: pd.DataFrame,\n    tw: pd.DataFrame,\n    rate_mu: float,\n    n_particles: int,\n    seed: int,\n    obs_power: float = STRUCT_OBS_POWER,\n    resamp_frac: float = STRUCT_RESAMP,\n    rate_noise_scale: float = 1.0,\n    resample_rate_noise_scale: float = 1.0,\n    init_rate_noise_scale: float = 1.0,\n    init_pos_noise_scale: float = 1.0,\n    early_rate_noise_scale: float = 1.0,\n    early_rate_frac: float = 0.0,\n) -> tuple[np.ndarray, float]:\n    tw_s = tw.sort_values("TVT")\n    tw_tvt = tw_s["TVT"].to_numpy(float)\n    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)\n    known = hw[hw["TVT_input"].notna()]\n    ev = hw[hw["TVT_input"].isna()]\n    if len(ev) == 0:\n        return hw["TVT_input"].to_numpy(float).copy(), 0.0\n    last = known.iloc[-1]\n    last_tvt = float(last["TVT_input"])\n    last_z = float(last["Z"])\n    prev_md = float(last["MD"])\n    tw_at_k = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)\n    gs = float(np.clip(np.nanstd(np.nan_to_num(known["GR"].to_numpy(float)) - tw_at_k), 10.0, 60.0))\n\n    rng = np.random.default_rng(int(seed))\n    n = int(n_particles)\n    pos = last_tvt + last_z + 2.0 * float(init_pos_noise_scale) * rng.standard_normal(n)\n    rate = float(rate_mu) + 0.01 * float(init_rate_noise_scale) * rng.standard_normal(n)\n    weights = np.ones(n, dtype=np.float64) / n\n    out_vals = hw["TVT_input"].to_numpy(float).copy()\n    gr_interp = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)\n    log_lik = 0.0\n\n    ev_indices = list(ev.index)\n    denom = max(len(ev_indices) - 1, 1)\n    for step_i, ridx in enumerate(ev_indices):\n        md_i = float(hw.at[ridx, "MD"])\n        z_i = float(hw.at[ridx, "Z"])\n        gr_i = float(gr_interp[ridx])\n        dm_step = max(md_i - prev_md, 1.0)\n        local_rate_scale = float(rate_noise_scale)\n        if float(early_rate_frac) > 0.0 and (step_i / denom) <= float(early_rate_frac):\n            local_rate_scale *= float(early_rate_noise_scale)\n        noise = STRUCT_VN * local_rate_scale * rng.standard_normal(n)\n        rate_n = STRUCT_MOM * rate + (1.0 - STRUCT_MOM) * float(rate_mu) + noise\n        pos_n = pos + rate_n * dm_step + STRUCT_PN * rng.standard_normal(n)\n        tvt_p = np.clip(pos_n - z_i, tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)\n        pos_n = tvt_p + z_i\n        expected_gr = np.interp(tvt_p, tw_tvt, tw_gr)\n        d = (gr_i - expected_gr) / gs\n        lk = np.exp(-0.5 * float(obs_power) * np.minimum(d * d, 80.0))\n        lk = np.maximum(lk, 1e-30)\n        avg_lk = float((weights * lk).sum())\n        log_lik += np.log(max(avg_lk, 1e-30))\n        weights_n = weights * lk\n        ws = float(weights_n.sum())\n        weights_n = weights_n / ws if ws > 0 else np.ones(n, dtype=np.float64) / n\n        n_eff = 1.0 / float((weights_n * weights_n).sum())\n        if n_eff < float(resamp_frac) * n:\n            cum = np.cumsum(weights_n)\n            u0 = rng.uniform(0, 1.0 / n)\n            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)\n            pos = pos_n[idx] + STRUCT_RP * rng.standard_normal(n)\n            rate = rate_n[idx] + STRUCT_RR * float(resample_rate_noise_scale) * rng.standard_normal(n)\n            weights = np.ones(n, dtype=np.float64) / n\n        else:\n            pos = pos_n\n            rate = rate_n\n            weights = weights_n\n        out_vals[ridx] = float(np.dot(weights, pos - z_i))\n        prev_md = md_i\n    return out_vals, float(log_lik)\n\n\ndef run_struct_pf_by_scale(\n    hw: pd.DataFrame,\n    tw: pd.DataFrame,\n    wid: str,\n    n_particles: int = STRUCT_N_PARTICLES,\n    n_seeds: int = STRUCT_N_SEEDS,\n    obs_power: float = STRUCT_OBS_POWER,\n    resamp_frac: float = STRUCT_RESAMP,\n    rate_noise_scale: float = 1.0,\n    resample_rate_noise_scale: float = 1.0,\n    init_rate_noise_scale: float = 1.0,\n    init_pos_noise_scale: float = 1.0,\n    early_rate_noise_scale: float = 1.0,\n    early_rate_frac: float = 0.0,\n) -> dict[str, np.ndarray]:\n    rates = struct_rate_bank(hw, wid)\n    well_seed_base = zlib.crc32(str(wid).encode("utf-8")) & 0xFFFFFFFF\n    preds = []\n    liks = []\n    for s in range(int(n_seeds)):\n        pred, ll = run_struct_particle_filter(\n            hw,\n            tw,\n            rate_mu=float(rates[s % len(rates)]),\n            n_particles=int(n_particles),\n            seed=int((STRUCT_SEED + 1009 * s + well_seed_base) % (2**32 - 1)),\n            obs_power=float(obs_power),\n            resamp_frac=float(resamp_frac),\n            rate_noise_scale=float(rate_noise_scale),\n            resample_rate_noise_scale=float(resample_rate_noise_scale),\n            init_rate_noise_scale=float(init_rate_noise_scale),\n            init_pos_noise_scale=float(init_pos_noise_scale),\n            early_rate_noise_scale=float(early_rate_noise_scale),\n            early_rate_frac=float(early_rate_frac),\n        )\n        preds.append(pred)\n        liks.append(ll)\n    pred_arr = np.stack(preds, axis=0)\n    liks = np.asarray(liks, dtype=np.float64)\n    liks_n = liks - float(np.nanmax(liks))\n    out = {}\n    for scale in STRUCT_SCALES:\n        weights = np.exp(np.clip(liks_n / float(scale), -700.0, 0.0))\n        sw = float(weights.sum())\n        weights = weights / sw if sw > 0 else np.ones_like(weights) / len(weights)\n        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(axis=0)\n    out["pf_mean"] = pred_arr.mean(axis=0)\n    return out\n\n\ndef sub2_level_consensus_sources(sample: pd.DataFrame) -> pd.DataFrame:\n    sample = sample.copy()\n    sample["well"] = sample["id"].str[:8]\n    sample["row_idx"] = sample["id"].str[9:].astype(int)\n    test_wells = sorted(sample["well"].unique())\n    init_struct_cloud()\n\n    rows = []\n    for i, wid in enumerate(test_wells, 1):\n        print(f"Level-consensus PF sources {i}/{len(test_wells)} {wid}", flush=True)\n        hw_te, tw_te = load_well(wid, "test")\n        known = hw_te["TVT_input"].dropna()\n        last_known_tvt = float(known.iloc[-1]) if len(known) > 0 else 0.0\n        try:\n            rates = struct_rate_bank(hw_te, wid)\n            pf_struct = run_struct_pf_by_scale(\n                hw_te,\n                tw_te,\n                wid,\n                n_particles=STRUCT_N_PARTICLES,\n                n_seeds=STRUCT_N_SEEDS,\n                obs_power=STRUCT_OBS_POWER,\n                resamp_frac=STRUCT_RESAMP,\n            )\n            pf_softobs = run_struct_pf_by_scale(\n                hw_te,\n                tw_te,\n                wid,\n                n_particles=STRUCT_N_PARTICLES,\n                n_seeds=STRUCT_N_SEEDS,\n                obs_power=SOFTOBS_OBS_POWER,\n                resamp_frac=SOFTOBS_RESAMP,\n            )\n            pf_rate075 = run_struct_pf_by_scale(\n                hw_te,\n                tw_te,\n                wid,\n                n_particles=RATE075_N_PARTICLES,\n                n_seeds=RATE075_N_SEEDS,\n                obs_power=0.60,\n                resamp_frac=0.35,\n                rate_noise_scale=0.75,\n                resample_rate_noise_scale=0.75,\n            )\n            pf_starttight = run_struct_pf_by_scale(\n                hw_te,\n                tw_te,\n                wid,\n                n_particles=RATE075_N_PARTICLES,\n                n_seeds=RATE075_N_SEEDS,\n                obs_power=0.60,\n                resamp_frac=0.35,\n                rate_noise_scale=0.75,\n                resample_rate_noise_scale=0.75,\n                init_rate_noise_scale=0.50,\n                init_pos_noise_scale=0.75,\n                early_rate_noise_scale=0.55,\n                early_rate_frac=0.25,\n            )\n            tvt_p0_source = 0.5 * pf_struct[STRUCT_DEPLOY_SCALE] + 0.5 * pf_softobs[STRUCT_DEPLOY_SCALE]\n            tvt_rate075_source = pf_rate075["pf_scale_8"]\n            tvt_starttight_source = pf_starttight["pf_scale_8"]\n            print(\n                f"  rates={len(rates)} min={float(np.min(rates)):.5f} max={float(np.max(rates)):.5f} "\n                f"n_eval={int(hw_te[\'TVT_input\'].isna().sum())}",\n                flush=True,\n            )\n        except Exception as exc:\n            print(f"  struct PF source failed, fallback last_known: {exc}", flush=True)\n            tvt_p0_source = hw_te["TVT_input"].fillna(last_known_tvt).to_numpy(float)\n            tvt_rate075_source = tvt_p0_source.copy()\n            tvt_starttight_source = tvt_p0_source.copy()\n\n        ws = sample[sample["well"] == wid]\n        for _, row in ws.iterrows():\n            ridx = int(row["row_idx"])\n            rows.append(\n                {\n                    "id": row["id"],\n                    "tvt_p0_source": float(tvt_p0_source[ridx]),\n                    "tvt_rate075_source": float(tvt_rate075_source[ridx]),\n                    "tvt_starttight_source": float(tvt_starttight_source[ridx]),\n                }\n            )\n    return pd.DataFrame(rows)\n\n\ndef main() -> None:\n    np.random.seed(STRUCT_SEED)\n    print(f"DATA_ROOT={DATA_ROOT}", flush=True)\n    print(f"ART_ROOT={ART_ROOT}", flush=True)\n    if not (ART_ROOT / "models").exists():\n        raise FileNotFoundError(f"Missing artifact models directory: {ART_ROOT / \'models\'}")\n    install_unpickle_shims()\n    rl.CFG.dataset_path = DATA_ROOT\n    rl.CFG.artifacts_path = ART_ROOT\n    rl.NCPU = int(os.environ.get("RAVI_WORKERS", "4"))\n\n    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")\n    max_test_wells = int(os.environ.get("ROGII_MAX_TEST_WELLS", "0"))\n    if (not ONKAGGLE) and max_test_wells > 0:\n        keep_wells = sorted(sample["id"].str[:8].unique())[:max_test_wells]\n        sample = sample[sample["id"].str[:8].isin(keep_wells)].reset_index(drop=True)\n        print(f"local smoke limited to wells={keep_wells} rows={len(sample)}", flush=True)\n    test_paths = sorted((DATA_ROOT / "test").glob("*__horizontal_well.csv"))\n    if (not ONKAGGLE) and max_test_wells > 0:\n        keep = set(sample["id"].str[:8].unique())\n        test_paths = [p for p in test_paths if p.name.split("__", 1)[0] in keep]\n    test_df = rl.build_dataset(test_paths, is_train=False, label="test")\n    features = [c for c in test_df.columns if c not in {"well", "id", "target"}]\n    X_test = test_df[features]\n    print(f"test_df={test_df.shape} features={len(features)}", flush=True)\n\n    base_preds = []\n    for name in MODEL_DIRS:\n        print(f"Loading/predicting {name}", flush=True)\n        tr = load_trainer(name)\n        base_preds.append(np.asarray(tr.predict(X_test), dtype=np.float32))\n    base_pred = np.stack(base_preds, axis=1)\n    ridge_drift = ridge_fold_ensemble(base_pred)\n    pf_test = test_df["pf_ancc"].to_numpy(np.float32) - test_df["last_known_tvt"].to_numpy(np.float32)\n    ridge_pp = apply_pp(test_df, ridge_drift, pf_test)\n    test_df = test_df.copy()\n    test_df["pred"] = test_df["last_known_tvt"].to_numpy(np.float32) + ridge_pp\n    test_df = sg_smooth(test_df, "pred")\n    sub1 = sample[["id"]].merge(test_df[["id", "pred"]].rename(columns={"pred": "tvt_sub1"}), on="id", how="left")\n\n    sub2 = sub2_level_consensus_sources(sample)\n    sub = sub1.merge(sub2, on="id", how="left")\n    sub = sub.merge(test_df[["id", "well", "z", "md_since"]], on="id", how="left")\n    fallback = float(test_df["last_known_tvt"].mean() + ridge_drift.mean()) if len(test_df) else 0.0\n    sub["tvt_sub1"] = sub["tvt_sub1"].fillna(fallback)\n    for col in ["tvt_p0_source", "tvt_rate075_source", "tvt_starttight_source"]:\n        sub[col] = sub[col].fillna(sub["tvt_sub1"])\n    if sub["z"].isna().any() or sub["well"].isna().any() or sub["md_since"].isna().any():\n        raise RuntimeError("Missing z/well/md_since columns for F rectification")\n\n    sub["tvt"] = FINAL_W_RIDGE * sub["tvt_sub1"] + FINAL_W_PF * sub["tvt_p0_source"]\n    p0_work = apply_chord_shrink(apply_f_rectification(sub, FRECT_FRAC), CHORD_W)\n    sub["tvt_p0_chord"] = p0_work["tvt"].to_numpy(np.float32)\n    sub["tvt_rate075"] = (\n        FINAL_W_RIDGE * sub["tvt_sub1"].to_numpy(np.float32)\n        + FINAL_W_PF * sub["tvt_rate075_source"].to_numpy(np.float32)\n    ).astype(np.float32)\n    sub["tvt_starttight"] = (\n        FINAL_W_RIDGE * sub["tvt_sub1"].to_numpy(np.float32)\n        + FINAL_W_PF * sub["tvt_starttight_source"].to_numpy(np.float32)\n    ).astype(np.float32)\n\n    level = (\n        sub.groupby("well", sort=False)["tvt_p0_chord"].transform("mean").to_numpy(np.float64)\n        + sub.groupby("well", sort=False)["tvt_rate075"].transform("mean").to_numpy(np.float64)\n        + sub.groupby("well", sort=False)["tvt_starttight"].transform("mean").to_numpy(np.float64)\n    ) / 3.0\n    p0_mean = sub.groupby("well", sort=False)["tvt_p0_chord"].transform("mean").to_numpy(np.float64)\n    shape = sub["tvt_p0_chord"].to_numpy(np.float64) - p0_mean\n    sub["tvt"] = (level + shape).astype(np.float32)\n    out = sub[["id", "tvt"]]\n    if len(out) != len(sample):\n        raise RuntimeError(f"submission row mismatch {len(out)} != {len(sample)}")\n    if out["tvt"].isna().any():\n        raise RuntimeError("NaN in submission")\n    print(\n        "p0_struct_soft055_frect summary "\n        f"ridge_w={FINAL_W_RIDGE:.2f} pf_w={FINAL_W_PF:.2f} "\n        f"softobs_power={SOFTOBS_OBS_POWER:.2f} pf_scale={STRUCT_DEPLOY_SCALE} frect_frac={FRECT_FRAC:.2f} chord_w={CHORD_W:.2f} "\n        f"p0_mean={sub[\'tvt_p0_chord\'].mean():.4f} rate075_mean={sub[\'tvt_rate075\'].mean():.4f} "\n        f"starttight_mean={sub[\'tvt_starttight\'].mean():.4f} "\n        f"final_mean={out[\'tvt\'].mean():.4f}",\n        flush=True,\n    )\n    out.to_csv(OUT_PATH, index=False)\n    print(f"Wrote {OUT_PATH} rows={len(out)}", flush=True)\n    print(out.head().to_string(index=False), flush=True)\n\n\nif __name__ == "__main__":\n    main()\n', ls.__dict__)

warnings.filterwarnings("ignore")

class CFG:
    dataset_path = Path("/home/yiyu/rogii/datasets/rogii-wellbore-geology-prediction")
    artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
    
    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    
    metric = root_mean_squared_error

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}

SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def load_well(wid, split='train'):
    base = CFG.dataset_path / split
    hw = pd.read_csv(base / f'{wid}__horizontal_well.csv')
    tw = pd.read_csv(base / f'{wid}__typewell.csv')
    return hw, tw


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 2.0 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):
    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred

SEED=42
NCPU=min(192,multiprocessing.cpu_count())

FORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
PLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5

BEAMS=[
    (10,20.0,144.0,2,"cons"),
    (10, 8.0, 64.0,2,"loose"),
    ( 8,35.0,220.0,1,"vcons"),
    (10,14.0, 90.0,5,"sm5"),
    (20, 4.0, 36.0,3,"vloose"),
    (12,12.0,100.0,3,"mid"),
    (15,25.0,180.0,2,"stiff"),
]

PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search ±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # ±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=False)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_

@njit(cache=False)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_

# Dense grid for O(1) typewell lookup
def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)


_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

class FormationPlaneKNN:
    def __init__(self,well_ids,data_dir):
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self,xy_q,self_wid=None,k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self,well_ids,data_dir,spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self,xy_q,self_wid=None,k=DENSE_K,nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)

_FI=None
_DI=None

def init_spatial_imputers():
    global _FI, _DI
    if _FI is not None and _DI is not None:
        return _FI, _DI
    hw_paths=sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
    train_wids=[p.stem.replace('__horizontal_well','') for p in hw_paths]
    _FI=FormationPlaneKNN(train_wids,CFG.dataset_path / "train")
    _DI=DenseANCCImputer(train_wids,CFG.dataset_path / "train")
    return _FI, _DI

ANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)
BEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)
SC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)
PF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)

def build_well(hw_path,tw_path,is_train):
    global _FI,_DI
    init_spatial_imputers()
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    try:
        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)
    pf_use=pf_a.astype(np.float32); std_use=std_a.astype(np.float32)
    has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT ±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    beam_ref=(bpaths['cons']+bpaths['sm5'])/2.

    # Multi-scale NCC → score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)

    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE
    tvt_fs={}; form_rmse={}; form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)
        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=np.float32(b_full); tvt_fs[f'bww_{fn}']=np.float32(b_wls)
        tvt_fs[f'bw50_{fn}']=np.float32(b_late)
        tvt_fs[f'bw_early_{fn}']=np.float32(b_early)   # NEW: early segment
        tvt_fs[f'bw_mid_{fn}']=np.float32(b_mid)       # NEW: mid segment
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)

    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d =fs.std(1).astype(np.float32)
    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)
    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)
    res_kn=ktvt+z_kn-d_kn
    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))

    all_sigs=[pf_use]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense]
    sig_mat=np.stack(all_sigs,1)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)

    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        'pf_ancc':pf_use,'pf_ancc_std':std_use,
        'pf_ancc_delta':(pf_use-last_tvt).astype(np.float32),
        'pf_z':(pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta':((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sig_std':sig_std,'sig_mean_d':sig_mean,
        **tvt_fs,
        **{f'frm_rmse_{fn}':sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d':form_mean_d,'form_std_d':form_std_d,'form_rng_d':form_rng_d,
        'spatial_ancc_d':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),
        'spatial_knn_dist':knn_d,
        'dense_ancc':d_ancc,'dense_std':d_std,'dense_dist':d_dist,
        'tvt_dense_d' :(tvt_dense -last_tvt).astype(np.float32),
        'tvt_densew_d':(tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d':(tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse':sc(d_rmse),'dense_bias':sc(d_bias),'dense_nb_std':sc(d_nb_std),
        'pf_vs_spatial':(pf_use-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense':(pf_use-tvt_dense).astype(np.float32),
        'spatial_vs_dense':(tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial':(bpaths['cons']-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam':(sc_ens-bpaths['cons']).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 -last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}':hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    for k,v in rolls.items(): feats[k]=v
    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result

def build_dataset(paths,is_train,label):
    args=[(str(p),str(p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv'),is_train)
          for p in paths
          if (p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0=time.time()
    res=Parallel(n_jobs=NCPU,prefer='threads',verbose=3)(
        delayed(build_well)(hp,tp,it) for hp,tp,it in args)
    parts=[r for r in res if r is not None]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


# ==================== Inference-only kernel entrypoint ====================
#!/usr/bin/env python3
"""Infer-only Ravaghi Ridge artifact submission.

Loads public Ravaghi trained base models from the artifact dataset, applies the
locally frozen second-level Ridge fold ensemble, then blends with the targetless
PF likelihood ensemble at scale=5.  This intentionally performs no model
training on Kaggle.
"""
import os
from pathlib import Path
import sys
import types
import warnings
import zlib

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import torch

rl = sys.modules[__name__]


DATA_ROOT = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("/kaggle/input/rogii-wellbore-geology-prediction")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(os.environ.get("ROGII_DATA_DIR", "/home/yiyu/rogii/datasets/rogii-wellbore-geology-prediction"))

ART_ROOT = Path("/kaggle/input/wellbore-geology-prediction-artifacts")
if not ART_ROOT.exists():
    ART_ROOT = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
if not ART_ROOT.exists():
    ART_ROOT = Path(os.environ.get(
        "RAVAGHI_ARTIFACT_DIR",
        "/home/yiyu/rogii/datasets/ravaghi_wellbore_geology_prediction_artifacts_kaggle_current",
    ))

SMOOTHER_ROOT = Path("/kaggle/input/rogii-posterior-gate-smoother-assets")
if not SMOOTHER_ROOT.exists():
    SMOOTHER_ROOT = Path("/kaggle/input/datasets/yiyu0716/rogii-posterior-gate-smoother-assets")
if not SMOOTHER_ROOT.exists():
    SMOOTHER_ROOT = Path(os.environ.get(
        "ROGII_SMOOTHER_DIR",
        "/home/yiyu/rogii/datasets_upload/rogii-posterior-gate-smoother-assets",
    ))

HINGE_ROOT = Path("/kaggle/input/rogii-level-shape-hinge-dropblendpath-assets")
if not HINGE_ROOT.exists():
    HINGE_ROOT = Path("/kaggle/input/datasets/yiyu0716/rogii-level-shape-hinge-dropblendpath-assets")
if not HINGE_ROOT.exists():
    HINGE_ROOT = Path(os.environ.get(
        "ROGII_HINGE_ASSET_DIR",
        "/home/yiyu/rogii/datasets_upload/rogii-level-shape-hinge-dropblendpath-assets",
    ))

OUT_PATH = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")

MODEL_DIRS = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]

RIDGE_INTERCEPTS = np.array(
    [-0.18320775032043457, -0.19101393222808838, -0.224159836769104, -0.191192626953125, -0.14542460441589355],
    dtype=np.float32,
)
RIDGE_COEFS = np.array(
    [
        [0.0, 0.2911567986011505, 0.38378193974494934, 0.08868265897035599, 0.3258693218231201],
        [0.053661931306123734, 0.39128851890563965, 0.5169801712036133, 0.0, 0.15848520398139954],
        [0.0, 0.31343886256217957, 0.48635944724082947, 0.05501436069607735, 0.22994305193424225],
        [0.0, 0.3842366635799408, 0.44450387358665466, 0.07253403216600418, 0.18571855127811432],
        [0.0, 0.23798350989818573, 0.5057194232940674, 0.12051041424274445, 0.21425989270210266],
    ],
    dtype=np.float32,
)
PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_PF = 0.09
FINAL_W_RIDGE = 0.50
FINAL_W_PF = 0.50
RAWPF_LEVEL_W_RIDGE = 0.30
RAWPF_LEVEL_W_PF = 0.70
SMOOTHER_ALPHAS = np.array([0.90, 0.90, 1.00, 0.65, 0.00], dtype=np.float32)
GATE_SMOOTHER_ALPHAS = np.array([0.65, 0.00, 0.80, 0.35, 0.80], dtype=np.float32)
STRUCT_MODE_RULE = dict(lam_path=1.0, lam_level=0.0, lam_shape=0.5, lam_base=0.5, lam_rank=0.0, blend_weight=0.45)
STRUCT_PREFIX_GATE_DIR_ACC = 0.5
STRUCT_PREFIX_CUT_FRACS = (0.45, 0.60, 0.75, 0.85)
STRUCT_PREFIX_MIN_VISIBLE = 80
STRUCT_PREFIX_MIN_PSEUDO = 80
PF_PREFIX_SEED = 1745
PF_PREFIX_N_SEEDS = 32
PF_PREFIX_N_PARTICLES = 200
PF_PREFIX_GATE_THRESHOLD = 8.369009
MOTION_SEED = 1745
MOTION_N_SEEDS = 32
MOTION_N_PARTICLES = 300
MOTION_RATE_OFFSETS = (-0.008, 0.0, 0.008)
MOTION_GRID_STEP = 0.5
MOTION_MOM = 0.998
MOTION_VN = 0.002
MOTION_PN = 0.005
MOTION_RP = 0.1
MOTION_RR = 0.001
MOTION_RESAMP = 0.5
MOTION_SCALES = (3.0, 5.0, 8.0, 12.0)
MOTION_RIDGE_INTERCEPT = 0.029301
MOTION_RIDGE_WEIGHTS = {
    "gate_bigru": 0.564091,
    "gate_base": 0.786030,
    "blend_base": -0.194145,
    "ridge_pp_sg": -0.076859,
    "pf_scale5": -0.311432,
    "moff_selector": 0.348050,
}

POSTERIOR_COLS = [
    "pf_seed_mean",
    "pf_seed_median",
    "pf_top1_mean",
    "pf_top2_mean",
    "pf_top4_mean",
    "pf_top8_mean",
    "pf_top16_mean",
    "pf_top32_mean",
    "beam_mean",
    "pf_scale_3",
    "pf_scale_3_wmedian",
    "pf_scale_3_q40",
    "pf_scale_3_q60",
    "pf_scale_3_trim20_80",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_scale_5_wmedian",
    "pf_scale_5_q40",
    "pf_scale_5_q60",
    "pf_scale_5_trim20_80",
    "pf_scale_8_wmedian",
    "pf_scale_8_q40",
    "pf_scale_8_q60",
    "pf_scale_8_trim20_80",
    "pf_scale_12_wmedian",
    "pf_scale_12_q40",
    "pf_scale_12_q60",
    "pf_scale_12_trim20_80",
    "sel15_current",
    "selector_pf_seed_median",
    "selector_pf_top4_mean",
    "selector_pf_top8_mean",
    "selector_pf_scale_5_wmedian",
    "selector_pf_scale_8_wmedian",
    "selector_pf_scale_5_trim20_80",
    "selector_pf_scale_8_trim20_80",
    "pf_scale_5_stdshrink",
    "selector_pf_scale_5_stdshrink",
]

STRUCT_CANDIDATE_COLS = [
    "pf_seed_mean",
    "pf_seed_median",
    "pf_top1_mean",
    "pf_top2_mean",
    "pf_top4_mean",
    "pf_top8_mean",
    "pf_top16_mean",
    "pf_top32_mean",
    "beam_mean",
    "pf_scale_3",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_scale_3_wmedian",
    "pf_scale_3_q40",
    "pf_scale_3_q60",
    "pf_scale_3_trim20_80",
    "pf_scale_5_wmedian",
    "pf_scale_5_q40",
    "pf_scale_5_q60",
    "pf_scale_5_trim20_80",
    "pf_scale_8_wmedian",
    "pf_scale_8_q40",
    "pf_scale_8_q60",
    "pf_scale_8_trim20_80",
    "pf_scale_12_wmedian",
    "pf_scale_12_q40",
    "pf_scale_12_q60",
    "pf_scale_12_trim20_80",
    "sel15_current",
    "selector_pf_seed_median",
    "selector_pf_top4_mean",
    "selector_pf_top8_mean",
    "selector_pf_scale_5_wmedian",
    "selector_pf_scale_8_wmedian",
    "selector_pf_scale_5_trim20_80",
    "selector_pf_scale_8_trim20_80",
    "pf_scale_5_stdshrink",
    "selector_pf_scale_5_stdshrink",
]

SMOOTHER_FEATURES = [
    "blend_base", "ridge_pp_sg", "pf_seed_mean", "pf_seed_median",
    "pf_top1_mean", "pf_top2_mean", "pf_top4_mean", "pf_top8_mean",
    "pf_top16_mean", "pf_top32_mean", "beam_mean", "pf_scale_3",
    "pf_scale_5", "pf_scale_8", "pf_scale_12", "pf_scale_5_wmedian",
    "pf_scale_5_q40", "pf_scale_5_q60", "pf_scale_5_trim20_80",
    "pf_scale_8_wmedian", "pf_scale_8_q40", "pf_scale_8_q60",
    "pf_scale_8_trim20_80", "pf_scale_12_trim20_80", "sel15_current",
    "selector_pf_scale_5_stdshrink", "pf_vs_ridge", "abs_pf_vs_ridge",
    "pf_scale_spread", "pf_quant_spread5", "pf_top_spread",
    "sel15_minus_raw5", "beam_minus_raw5", "pf_ancc_delta",
    "pf_z_delta", "pf_vs_z", "beam_mean_d", "beam_med_d", "beam_std_d",
    "sc8_d", "sc15_d", "sc25_d", "sc_ens_d", "sc_cons_d", "hyb_d",
    "sig_mean_d", "sig_std", "form_mean_d", "form_std_d", "tvtF_ANCC",
    "tvt_dense_d", "slp_b_d_all", "slp_b_d_50", "slp_all", "slp_50",
    "slp_z", "dzdmd", "dxdmd", "dydmd", "dz", "dxy", "z", "md_since",
    "frac", "gr", "gr_d1", "gr_d2", "gr_env", "gr_nrg", "grm5",
    "grm21", "grm51", "last_known_tvt",
]

GATE_SMOOTHER_FEATURES = [
    "pfreplay_gate_base", "path_bt_dir_base", "posterior", "path_best_blend",
    "path_best_struct", "path_minus_posterior", "path_struct_minus_posterior",
    "abs_path_minus_posterior", "path_bt_dir_weight", "pfreplay_gate_weight",
    "gate_disagrees_bt", "pfreplay_feature", "bt_dir_acc", "bt_rmse_p90",
    "bt_level_abs_p90", "bt_gain_mean", "pred_help_prob", "pred_utility",
    "xy_kdist_p90", "pf_ridge_gap_proxy", "pfbt_scale5_level_abs_mean",
    "pfbt_top1_rmse_mean", "pfbt_scale5_level_abs_p90", "pfbt_scale5_rmse_p90",
    "pfbt_scale5_dir_acc", "pfbt_top1_dir_acc", "pfbt_seed_level_std_mean",
    "pfbt_mode_signed_level_range_mean", "pfbt_lik_rmse_spearman_p90",
    "blend_base", "ridge_pp_sg", "pf_seed_mean", "pf_seed_median",
    "pf_top1_mean", "pf_top2_mean", "pf_top4_mean", "pf_top8_mean",
    "pf_top16_mean", "pf_top32_mean", "beam_mean", "pf_scale_3",
    "pf_scale_5", "pf_scale_8", "pf_scale_12", "pf_scale_5_wmedian",
    "pf_scale_5_q40", "pf_scale_5_q60", "pf_scale_5_trim20_80",
    "pf_scale_8_wmedian", "pf_scale_8_q40", "pf_scale_8_q60",
    "pf_scale_8_trim20_80", "pf_scale_12_trim20_80", "sel15_current",
    "selector_pf_scale_5_stdshrink", "pf_vs_ridge", "abs_pf_vs_ridge",
    "pf_scale_spread", "pf_quant_spread5", "pf_top_spread",
    "sel15_minus_raw5", "beam_minus_raw5", "pf_ancc_delta", "pf_z_delta",
    "pf_vs_z", "beam_mean_d", "beam_med_d", "beam_std_d", "sc8_d",
    "sc15_d", "sc25_d", "sc_ens_d", "sc_cons_d", "hyb_d", "sig_mean_d",
    "sig_std", "form_mean_d", "form_std_d", "tvtF_ANCC", "tvt_dense_d",
    "slp_b_d_all", "slp_b_d_50", "slp_all", "slp_50", "slp_z", "dzdmd",
    "dxdmd", "dydmd", "dz", "dxy", "z", "md_since", "frac", "gr",
    "gr_d1", "gr_d2", "gr_env", "gr_nrg", "grm5", "grm21", "grm51",
    "last_known_tvt",
]


class DummyTrainer:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def predict(self, X):
        preds = [m.predict(X) for m in self.estimators]
        return np.mean(np.stack(preds, axis=0), axis=0)


def install_unpickle_shims() -> None:
    koolbox = types.ModuleType("koolbox")
    trainer_pkg = types.ModuleType("koolbox.trainer")
    trainer_mod = types.ModuleType("koolbox.trainer.trainer")
    koolbox.Trainer = DummyTrainer
    trainer_pkg.Trainer = DummyTrainer
    trainer_mod.Trainer = DummyTrainer
    sys.modules.setdefault("koolbox", koolbox)
    sys.modules.setdefault("koolbox.trainer", trainer_pkg)
    sys.modules.setdefault("koolbox.trainer.trainer", trainer_mod)
    try:
        import sklearn.metrics._regression as reg

        if not hasattr(reg, "root_mean_squared_error"):
            reg.root_mean_squared_error = lambda y, p, **kw: mean_squared_error(y, p, **kw) ** 0.5
    except Exception:
        pass


def load_trainer(name: str) -> DummyTrainer:
    paths = sorted((ART_ROOT / "models" / name).glob("*.pkl"))
    if not paths:
        # Older artifact layout fallback.
        paths = sorted((ART_ROOT / "models" / name).glob("models.pkl"))
    if not paths:
        raise FileNotFoundError(f"No artifact model found for {name} under {ART_ROOT}")
    obj = joblib.load(paths[0])
    if hasattr(obj, "estimators"):
        return obj
    if isinstance(obj, list):
        wrapper = DummyTrainer()
        wrapper.estimators = obj
        return wrapper
    raise TypeError(f"Unsupported model artifact for {name}: {type(obj)}")


def ridge_fold_ensemble(base_pred: np.ndarray) -> np.ndarray:
    fold_preds = base_pred @ RIDGE_COEFS.T + RIDGE_INTERCEPTS[None, :]
    return fold_preds.mean(axis=1).astype(np.float32)


def apply_pp(df: pd.DataFrame, model_drift: np.ndarray, pf_drift: np.ndarray) -> np.ndarray:
    d = model_drift * (1.0 - PP_W_PF) + pf_drift * PP_W_PF
    d = d * (1.0 - np.exp(-np.maximum(df["md_since"].to_numpy(float), 0.0) / PP_TAU))
    return (d * PP_ALPHA).astype(np.float32)


def sg_smooth(df: pd.DataFrame, col: str, sg_w: int = 17, sg_p: int = 3) -> pd.DataFrame:
    df = df.copy()
    for _, g in df.groupby("well", sort=False):
        v = g[col].to_numpy(np.float32)
        wl = min(sg_w, len(v))
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df


SOFTGR2_DELTAS = np.arange(-8.0, 8.0 + 1.0, 1.0, dtype=np.float64)
SOFTGR2_VARIANTS = {
    "anom_corr_t4": (
        [("tw_anom_corr", 1.0), ("marker_anom_corr", 1.0), ("tw_corr", 0.5), ("tw_wide_corr", 0.5)],
        4.0,
    ),
    "rmse_anom_t4": (
        [
            ("tw_rmse", -0.6),
            ("tw_mae", -0.6),
            ("tw_anom_corr", 0.8),
            ("marker_anom_corr", 0.6),
            ("tw_corr", 0.3),
        ],
        4.0,
    ),
    "all_weak_t4": (
        [
            ("tw_corr", 0.5),
            ("tw_wide_corr", 0.5),
            ("tw_anom_corr", 0.6),
            ("marker_anom_corr", 0.6),
            ("marker_mask_jaccard", 0.2),
            ("tw_rmse", -0.3),
            ("tw_mae", -0.3),
            ("tw_wide_rmse", -0.2),
            ("resid_range_oob", -0.2),
        ],
        4.0,
    ),
}
SOFTGR2_CONFIG = {
    "variant": "anom_corr_t4",
    "entropy_max": 0.90,
    "peak_min": 0.0,
    "std_max": 999.0,
    "min_abs": 1.0,
    "sign_min": 0.75,
    "shrink": 0.90,
    "cap": 4.0,
}


def _softgr2_smooth(values, win: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    return (
        pd.Series(arr)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .rolling(int(max(1, win)), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float64)
    )


def _softgr2_corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 8:
        return 0.0
    x = aa[mask] - float(np.mean(aa[mask]))
    y = bb[mask] - float(np.mean(bb[mask]))
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 1e-12 else 0.0


def _softgr2_top_fraction_mask(values, frac: float = 0.18) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.zeros(len(arr), dtype=bool)
    if int(mask.sum()) == 0:
        return out
    threshold = float(np.nanquantile(np.abs(arr[mask]), max(0.0, min(1.0, 1.0 - float(frac)))))
    out[mask] = np.abs(arr[mask]) >= threshold
    return out


def _softgr2_unique_tvt_gr(tvt, gr) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"tvt": np.asarray(tvt, dtype=np.float64), "gr": np.asarray(gr, dtype=np.float64)})
    frame = frame[np.isfinite(frame["tvt"]) & np.isfinite(frame["gr"])].copy()
    if len(frame) < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    frame["tvt_round"] = np.round(frame["tvt"].to_numpy(np.float64), 3)
    grouped = frame.groupby("tvt_round", sort=True)["gr"].mean()
    return grouped.index.to_numpy(np.float64), _softgr2_smooth(grouped.to_numpy(np.float64), win=7)


def _softgr2_affine_calibration(prefix_gr, prefix_tvt, tw_tvt, tw_gr) -> tuple[float, float]:
    pg = np.asarray(prefix_gr, dtype=np.float64)
    pt = np.asarray(prefix_tvt, dtype=np.float64)
    if len(pg) < 12 or len(tw_tvt) < 12:
        return 1.0, 0.0
    exp = np.interp(pt, tw_tvt, tw_gr)
    mask = np.isfinite(pg) & np.isfinite(exp)
    if int(mask.sum()) < 12 or float(np.std(pg[mask])) <= 1e-9:
        return 1.0, 0.0
    x = np.column_stack([pg[mask], np.ones(int(mask.sum()), dtype=np.float64)])
    try:
        scale, bias = np.linalg.lstsq(x, exp[mask], rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    if not (np.isfinite(scale) and np.isfinite(bias)):
        return 1.0, 0.0
    return float(scale), float(bias)


def _softgr2_residual_range_oob(resid, prefix_resid) -> float:
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    p = np.asarray(prefix_resid, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(r) == 0 or len(p) == 0:
        return 0.0
    scale = 1.4826 * float(np.median(np.abs(p - float(np.median(p)))))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(p))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 2.0
    q10 = float(np.quantile(p, 0.10))
    q90 = float(np.quantile(p, 0.90))
    h10 = float(np.quantile(r, 0.10))
    h90 = float(np.quantile(r, 0.90))
    return float(max(q10 - h10, 0.0) + max(h90 - q90, 0.0)) / max(scale, 2.0)


def _softgr2_candidate_table_for_well(wid: str, frame: pd.DataFrame) -> pd.DataFrame:
    hw_path = DATA_ROOT / "test" / f"{wid}__horizontal_well.csv"
    tw_path = DATA_ROOT / "test" / f"{wid}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        return pd.DataFrame()
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path)
    if not {"TVT", "GR"}.issubset(tw.columns) or not {"TVT_input", "GR"}.issubset(hw.columns):
        return pd.DataFrame()
    tw_tvt, tw_gr = _softgr2_unique_tvt_gr(tw["TVT"].to_numpy(np.float64), tw["GR"].to_numpy(np.float64))
    if len(tw_tvt) < 20:
        return pd.DataFrame()

    row_idx = frame["row_idx"].to_numpy(np.int64) if "row_idx" in frame.columns else frame["id"].str[9:].astype(int).to_numpy(np.int64)
    row_idx = row_idx[(row_idx >= 0) & (row_idx < len(hw))]
    if len(row_idx) == 0:
        return pd.DataFrame()
    if len(row_idx) > 320:
        take = np.unique(np.linspace(0, len(row_idx) - 1, 320).round().astype(np.int64))
        row_idx = row_idx[take]
    hidden_base = frame.set_index(frame["id"].str[9:].astype(int)).reindex(row_idx)["tvt"].to_numpy(np.float64)
    if not np.isfinite(hidden_base).any():
        return pd.DataFrame()

    tvt_input = hw["TVT_input"].to_numpy(np.float64)
    gr_all = hw["GR"].to_numpy(np.float64)
    ps_idx = int(np.argmax(~np.isfinite(tvt_input))) if np.any(~np.isfinite(tvt_input)) else len(hw)
    prefix_mask = np.isfinite(tvt_input[:ps_idx]) & np.isfinite(gr_all[:ps_idx])
    prefix_tvt = tvt_input[:ps_idx][prefix_mask]
    prefix_gr = _softgr2_smooth(gr_all[:ps_idx], win=9)[prefix_mask]
    scale, bias = _softgr2_affine_calibration(prefix_gr, prefix_tvt, tw_tvt, tw_gr)
    if len(prefix_tvt) >= 12:
        prefix_exp = np.interp(prefix_tvt, tw_tvt, tw_gr)
        prefix_resid = scale * prefix_gr + bias - prefix_exp
    else:
        prefix_resid = np.zeros(0, dtype=np.float64)

    hidden_gr = _softgr2_smooth(gr_all[row_idx], win=9)
    hidden_wide = _softgr2_smooth(gr_all[row_idx], win=51)
    hidden_cal = scale * hidden_gr + bias
    hidden_cal_wide = scale * hidden_wide + bias
    rows = []
    for delta in SOFTGR2_DELTAS:
        cand_tvt = hidden_base + float(delta)
        tw_exp = np.interp(cand_tvt, tw_tvt, tw_gr)
        tw_exp_wide = _softgr2_smooth(tw_exp, win=51)
        tw_res = hidden_cal - tw_exp
        tw_res_wide = hidden_cal_wide - tw_exp_wide
        h_anom = hidden_cal - hidden_cal_wide
        t_anom = tw_exp - tw_exp_wide
        hmask = _softgr2_top_fraction_mask(h_anom, frac=0.18) | _softgr2_top_fraction_mask(np.gradient(hidden_cal), frac=0.18)
        tmask = _softgr2_top_fraction_mask(t_anom, frac=0.18) | _softgr2_top_fraction_mask(np.gradient(tw_exp), frac=0.18)
        union = hmask | tmask
        marker_idx = union & np.isfinite(h_anom) & np.isfinite(t_anom)
        marker_corr = _softgr2_corr(h_anom[marker_idx], t_anom[marker_idx]) if int(marker_idx.sum()) >= 6 else _softgr2_corr(h_anom, t_anom)
        rows.append(
            {
                "well": wid,
                "delta": float(delta),
                "tw_rmse": float(np.sqrt(np.mean(tw_res * tw_res))),
                "tw_wide_rmse": float(np.sqrt(np.mean(tw_res_wide * tw_res_wide))),
                "tw_mae": float(np.mean(np.abs(tw_res))),
                "tw_corr": _softgr2_corr(hidden_cal, tw_exp),
                "tw_wide_corr": _softgr2_corr(hidden_cal_wide, tw_exp_wide),
                "tw_anom_corr": _softgr2_corr(h_anom, t_anom),
                "marker_anom_corr": marker_corr,
                "marker_mask_jaccard": float(np.sum(hmask & tmask) / max(np.sum(union), 1)),
                "resid_range_oob": _softgr2_residual_range_oob(tw_res, prefix_resid),
            }
        )
    return pd.DataFrame(rows)


def _softgr2_robust_z(values, sign: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=np.float64)
    med = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - med)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return float(sign) * (arr - med) / scale


def _softgr2_posterior_delta(group: pd.DataFrame, variant: str) -> tuple[float, float, float]:
    terms, temp = SOFTGR2_VARIANTS[variant]
    score = np.zeros(len(group), dtype=np.float64)
    for col, coef in terms:
        if col in group.columns:
            score += abs(float(coef)) * _softgr2_robust_z(group[col].to_numpy(np.float64), 1.0 if coef > 0 else -1.0)
    score -= 0.05 * np.abs(group["delta"].to_numpy(np.float64))
    scaled = score / max(float(temp), 1e-6)
    scaled = scaled - float(np.nanmax(scaled))
    prob = np.exp(scaled)
    prob = prob / max(float(np.sum(prob)), 1e-12)
    deltas = group["delta"].to_numpy(np.float64)
    entropy = -float(np.sum(prob * np.log(prob + 1e-12))) / max(float(np.log(len(prob))), 1e-12)
    return float(np.sum(prob * deltas)), entropy, float(np.max(prob))


def _softgr2_delta_for_well(wid: str, frame: pd.DataFrame) -> float:
    candidates = _softgr2_candidate_table_for_well(wid, frame)
    if candidates.empty:
        return 0.0
    soft = {}
    entropy = np.nan
    peak = np.nan
    for variant in SOFTGR2_VARIANTS:
        delta, ent, pk = _softgr2_posterior_delta(candidates, variant)
        soft[variant] = float(delta)
        if variant == "anom_corr_t4":
            entropy = float(ent)
            peak = float(pk)
    values = np.asarray(list(soft.values()), dtype=np.float64)
    mean = float(np.mean(values)) if len(values) else 0.0
    sign_agree = float((np.sign(values) == np.sign(mean)).mean()) if len(values) else 0.0
    variant_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cfg = SOFTGR2_CONFIG
    trusted = (
        float(entropy) <= float(cfg["entropy_max"])
        and float(peak) >= float(cfg["peak_min"])
        and variant_std <= float(cfg["std_max"])
        and abs(mean) >= float(cfg["min_abs"])
        and sign_agree >= float(cfg["sign_min"])
    )
    if not trusted:
        return 0.0
    raw = float(np.clip(soft[str(cfg["variant"])], -float(cfg["cap"]), float(cfg["cap"])))
    return float(cfg["shrink"]) * raw


def compute_softgr2_level_delta(source: pd.DataFrame) -> np.ndarray:
    if source.empty or "well" not in source.columns or "tvt" not in source.columns:
        return np.zeros(len(source), dtype=np.float32)
    out = np.zeros(len(source), dtype=np.float64)
    for wid, grp in source.groupby("well", sort=False):
        try:
            delta = _softgr2_delta_for_well(str(wid), grp)
        except Exception as exc:
            print(f"  softGR2 skipped {wid}: {exc}", flush=True)
            delta = 0.0
        out[grp.index.to_numpy()] = float(delta)
    return out.astype(np.float32)


VP_PREFIX_BETA = 3.55
VP_PATH_PRIOR_WEIGHT = 0.0
VP_POSTERIOR_PRIOR_WEIGHT = 0.03
VP_ALPHA = 0.42
VP_CLIP = 7.5
VP_CLUSTER_GAP = 6.0
VP_TOP_CLUSTERS = 1
VP_MEMBER_TOP = 1
VP_TAU_CLUSTER = 0.50
VP_TAU_MEMBER = 0.80
VP_PREFIX_WINDOWS = (16, 32, 64)
VP_ENDPOINT_OFFSETS = (-24.0, -12.0, 0.0, 12.0, 24.0)
VP_SLOPE_OFFSETS = (-0.012, 0.0, 0.012)
VP_CURVATURE_OFFSETS = (-10.0, 0.0, 10.0)
VP_GR_WINDOW = 31
VP_HUBER_SCALE = 18.0
VP_DIVERSE_TOPK = 60
VP_DIVERSE_SCORE_HEAD = 12
VP_DIVERSE_FAMILY_QUOTA = 6
VP_DIVERSE_DELTA_BIN = 6.0

SEG_ENABLED = True
SEG_LEN = 64
SEG_MAX_CANDIDATES = 24
SEG_SCORE_WEIGHT = 0.25
SEG_GEOLOGY_BETA = 0.6
SEG_TRANSITION_SIGMA = 2.0
SEG_JUMP_SIGMA = 18.0
SEG_TEMPERATURE = 1.0
SEG_ALPHA = 0.64
SEG_CLIP = 9.0
SEG_CLUSTER_GAP = 8.0
SEG_CLUSTER_TAU_LSE = 0.8
SEG_CLUSTER_ENTROPY_MAX = 0.90
SEGMENTED_POSTERIOR_CACHE: dict[str, dict[str, object]] = {}
SEG_FORMATION_ORDER = {"ANCC": 0, "ASTNU": 1, "ASTNL": 2, "EGFDU": 3, "EGFDL": 4, "BUDA": 5}


def _vp_fill(values, fill_value=None) -> np.ndarray:
    x = pd.Series(np.asarray(values, dtype=np.float64)).interpolate(limit_direction="both")
    if fill_value is None:
        arr = x.to_numpy(dtype=np.float64)
        fill_value = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
    return x.fillna(float(fill_value)).to_numpy(dtype=np.float64)


def _vp_roll(values, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy()
    w = max(1, min(int(window), len(arr)))
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return arr.astype(np.float64, copy=True)
    pad = w // 2
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.pad(arr, (pad, pad), mode="edge"), kernel, mode="valid")


def _vp_corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(mask.sum()) < 3:
        return float("nan")
    x = aa[mask] - float(np.mean(aa[mask]))
    y = bb[mask] - float(np.mean(bb[mask]))
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 1e-12 else float("nan")


def _vp_huber_score(resid, scale: float = VP_HUBER_SCALE) -> float:
    r = np.asarray(resid, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("inf")
    c = max(float(scale), 1e-6)
    a = np.abs(r)
    loss = np.where(a <= c, 0.5 * r * r, c * (a - 0.5 * c))
    return float(2.0 * np.mean(loss))


def _vp_typewell_arrays(tw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = tw[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT").drop_duplicates("TVT")
    tvt = frame["TVT"].to_numpy(np.float64)
    gr = _vp_roll(_vp_fill(frame["GR"].to_numpy(np.float64)), VP_GR_WINDOW)
    return tvt, gr


def _vp_fit_heel_affine(expected, observed) -> tuple[float, float, float, int]:
    x = np.asarray(expected, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.nanstd(x)) < 1e-8:
        intercept = float(np.nanmedian(y - x)) if len(x) else 0.0
        pred = x + intercept
        rmse = float(np.sqrt(np.mean((y - pred) ** 2))) if len(x) else float("inf")
        return 1.0, intercept, rmse, int(len(x))
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    var = float(np.mean((x - xm) ** 2))
    raw_slope = float(np.mean((x - xm) * (y - ym)) / max(var, 1e-12))
    raw_slope = float(np.clip(raw_slope, 0.35, 2.50))
    weight = len(x) / (len(x) + 80.0)
    slope = float((1.0 - weight) + weight * raw_slope)
    intercept = float(np.median(y - slope * x))
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return slope, intercept, rmse, int(len(x))


def _vp_tail_rate(md, values, *, tail: bool, window: int) -> float:
    m = np.asarray(md, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(len(m), len(v), int(window))
    if n < 2:
        return 0.0
    mm = m[-n:] if tail else m[:n]
    vv = v[-n:] if tail else v[:n]
    dm = float(mm[-1] - mm[0])
    if abs(dm) < 1e-9:
        return 0.0
    return float((vv[-1] - vv[0]) / dm)


def _vp_compose_centered(level: float, shape_tvt: np.ndarray) -> np.ndarray:
    shape = np.asarray(shape_tvt, dtype=np.float64)
    if len(shape) == 0:
        return shape.copy()
    return float(level) + (shape - float(np.mean(shape)))


def _vp_append_unique(
    out: list[dict[str, object]],
    seen: set[bytes],
    *,
    name: str,
    family: str,
    tvt: np.ndarray,
) -> None:
    arr = np.asarray(tvt, dtype=np.float64)
    if len(arr) == 0:
        return
    if not np.isfinite(arr).all():
        arr = pd.Series(arr).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float64)
    if not np.isfinite(arr).all():
        return
    key = np.round(arr, 3).astype(np.float32).tobytes()
    if key in seen:
        return
    seen.add(key)
    out.append({"name": str(name), "family": str(family), "tvt": arr.astype(np.float64)})


def _vp_level_grid_candidates(anchor_tvt: np.ndarray, z_hidden: np.ndarray, seen: set[bytes]) -> list[dict[str, object]]:
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    z_hidden = np.asarray(z_hidden, dtype=np.float64)
    if len(anchor_tvt) == 0 or len(z_hidden) != len(anchor_tvt):
        return []
    progress = np.linspace(0.0, 1.0, len(anchor_tvt), dtype=np.float64)
    out: list[dict[str, object]] = []
    for off in (-36.0, -24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0, 36.0):
        _vp_append_unique(
            out,
            seen,
            name=f"level_grid:const:{off:+.1f}",
            family="level_grid",
            tvt=anchor_tvt + float(off),
        )
        _vp_append_unique(
            out,
            seen,
            name=f"level_grid:ramp:{off:+.1f}",
            family="level_grid",
            tvt=anchor_tvt + float(off) * progress,
        )
    return out


def _vp_prefix_u_rate_candidates(
    known_md: np.ndarray,
    known_tvt: np.ndarray,
    known_z: np.ndarray,
    hidden_md: np.ndarray,
    hidden_z: np.ndarray,
    anchor_tvt: np.ndarray,
    seen: set[bytes],
) -> list[dict[str, object]]:
    known_md = np.asarray(known_md, dtype=np.float64)
    known_tvt = np.asarray(known_tvt, dtype=np.float64)
    known_z = np.asarray(known_z, dtype=np.float64)
    hidden_md = np.asarray(hidden_md, dtype=np.float64)
    hidden_z = np.asarray(hidden_z, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    ok = np.isfinite(known_md) & np.isfinite(known_tvt) & np.isfinite(known_z)
    known_md = known_md[ok]
    known_tvt = known_tvt[ok]
    known_z = known_z[ok]
    if len(known_md) < 2 or len(hidden_md) == 0 or len(anchor_tvt) == 0:
        return []
    known_u = known_tvt + known_z
    last_md = float(known_md[-1])
    last_u = float(known_u[-1])
    windows = [32, 64, 128, 256, 512, len(known_md)]
    rate_offsets = (-0.012, -0.006, 0.0, 0.006, 0.012)
    blends = (0.0, 0.5)
    out: list[dict[str, object]] = []
    for win in windows:
        base_rate = _vp_tail_rate(known_md, known_u, tail=True, window=int(win))
        for off in rate_offsets:
            rate = float(base_rate) + float(off)
            u_line = last_u + rate * (hidden_md - last_md)
            tvt_line = u_line - hidden_z
            for blend in blends:
                b = float(blend)
                tvt = (1.0 - b) * tvt_line + b * _vp_compose_centered(float(np.mean(tvt_line)), anchor_tvt)
                _vp_append_unique(
                    out,
                    seen,
                    name=f"prefix_u_rate:w{int(win)}:r{float(off):+0.4f}:b{b:.2f}",
                    family="prefix_u_rate",
                    tvt=tvt,
                )
    return out


def _vp_formation_contact_candidates(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    row_idx: np.ndarray,
    start_row: int,
    anchor_tvt: np.ndarray,
    z_hidden: np.ndarray,
    seen: set[bytes],
) -> list[dict[str, object]]:
    if tw.empty or "Geology" not in tw.columns or "TVT" not in tw.columns:
        return []
    row_idx = np.asarray(row_idx, dtype=np.int64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    z_hidden = np.asarray(z_hidden, dtype=np.float64)
    if len(row_idx) == 0 or len(anchor_tvt) == 0:
        return []
    tvt_input = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_z = hw["Z"].to_numpy(np.float64)[:start_row]
    known_mask = np.isfinite(tvt_input) & np.isfinite(known_z)
    if int(known_mask.sum()) < 8:
        return []
    geo = tw.dropna(subset=["Geology", "TVT"])
    out: list[dict[str, object]] = []
    for ref_col in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"):
        if ref_col not in hw.columns:
            continue
        ref_vals = geo.loc[geo["Geology"].astype(str).eq(ref_col), "TVT"].to_numpy(dtype=np.float64)
        if len(ref_vals) == 0:
            continue
        top_all = hw[ref_col].to_numpy(np.float64)
        top_known = top_all[:start_row][known_mask]
        top_hidden = top_all[row_idx]
        if int(np.isfinite(top_known).sum()) < 8 or int(np.isfinite(top_hidden).sum()) < max(4, len(top_hidden) // 5):
            continue
        ref_tvt = float(np.nanmin(ref_vals))
        base_known = ref_tvt - (known_z[known_mask] - top_known)
        offset = float(np.nanmedian(tvt_input[known_mask] - base_known))
        tvt = ref_tvt - (z_hidden - top_hidden) + offset
        _vp_append_unique(
            out,
            seen,
            name=f"formation_contact:{ref_col}",
            family="formation_contact",
            tvt=tvt,
        )
        centered = _vp_compose_centered(float(np.nanmean(tvt)), anchor_tvt)
        _vp_append_unique(
            out,
            seen,
            name=f"formation_contact:{ref_col}:anchor_shape",
            family="formation_contact",
            tvt=centered,
        )
    return out


def _vp_self_gr_estimate(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    hidden_gr: np.ndarray,
    *,
    half_width: int,
    stride: int,
    hidden_step: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(known_gr) < 2 * int(half_width) + 4 or len(hidden_gr) < 2:
        return None
    kgr = _vp_roll(_vp_fill(known_gr), 5)
    hgr = _vp_roll(_vp_fill(hidden_gr), 5)
    win = 2 * int(half_width) + 1
    starts = np.arange(0, len(kgr) - win + 1, max(int(stride), 1), dtype=np.int32)
    if len(starts) < 2:
        return None
    cand = kgr[starts[:, None] + np.arange(win)[None, :]]
    cand = (cand - cand.mean(axis=1, keepdims=True)) / (cand.std(axis=1, keepdims=True) + 1e-6)
    centers = np.clip(starts + int(half_width), 0, len(known_tvt) - 1)
    cand_tvt = np.asarray(known_tvt, dtype=np.float64)[centers]
    anchors = np.arange(0, len(hgr), max(int(hidden_step), 1), dtype=np.int32)
    pad = np.pad(hgr, int(half_width), mode="edge")
    hidden = pad[anchors[:, None] + np.arange(win)[None, :]]
    hidden = (hidden - hidden.mean(axis=1, keepdims=True)) / (hidden.std(axis=1, keepdims=True) + 1e-6)
    score = hidden @ cand.T / float(win)
    best = score.argmax(axis=1)
    est_anchor = cand_tvt[best]
    score_anchor = score[np.arange(len(anchors)), best]
    full_tvt = np.interp(np.arange(len(hgr)), anchors, est_anchor)
    full_score = np.interp(np.arange(len(hgr)), anchors, score_anchor)
    return full_tvt.astype(np.float64), full_score.astype(np.float64)


def _vp_selfgr_candidates(
    known_tvt: np.ndarray,
    known_gr: np.ndarray,
    hidden_gr: np.ndarray,
    anchor_tvt: np.ndarray,
    z_hidden: np.ndarray,
    *,
    last_known: float,
    seen: set[bytes],
) -> list[dict[str, object]]:
    known_tvt = np.asarray(known_tvt, dtype=np.float64)
    known_gr = np.asarray(known_gr, dtype=np.float64)
    hidden_gr = np.asarray(hidden_gr, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64)
    if len(hidden_gr) < 20 or len(anchor_tvt) == 0:
        return []
    mask0 = np.isfinite(known_tvt) & np.isfinite(known_gr)
    if int(mask0.sum()) < 80:
        return []
    out: list[dict[str, object]] = []
    level_anchor = float(np.mean(anchor_tvt))
    for band in (120.0, 220.0):
        mask = mask0 & (known_tvt >= float(last_known) - float(band)) & (known_tvt <= float(last_known) + float(band))
        if int(mask.sum()) < 80:
            continue
        kt = known_tvt[mask]
        kg = known_gr[mask]
        order = np.argsort(kt)
        kt = kt[order]
        kg = kg[order]
        estimates: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for hwid in (8, 16, 28):
            est = _vp_self_gr_estimate(kg, kt, hidden_gr, half_width=int(hwid), stride=3, hidden_step=18)
            if est is None:
                continue
            est_tvt, score = est
            smooth = _vp_roll(est_tvt, min(101, max(1, len(est_tvt) // 3 * 2 + 1)))
            _vp_append_unique(
                out,
                seen,
                name=f"selfgr:direct:b{int(band)}:w{int(hwid)}",
                family="selfgr",
                tvt=smooth,
            )
            _vp_append_unique(
                out,
                seen,
                name=f"selfgr:anchor_level:b{int(band)}:w{int(hwid)}",
                family="selfgr",
                tvt=_vp_compose_centered(level_anchor, smooth),
            )
            estimates.append(smooth)
            weights.append(score)
        if estimates:
            est_arr = np.vstack(estimates)
            score_arr = np.vstack(weights)
            wt = np.exp(4.0 * np.nan_to_num(score_arr))
            wt /= wt.sum(axis=0, keepdims=True) + 1e-9
            blend = (wt * est_arr).sum(axis=0)
            _vp_append_unique(
                out,
                seen,
                name=f"selfgr:direct_blend:b{int(band)}",
                family="selfgr",
                tvt=blend,
            )
            _vp_append_unique(
                out,
                seen,
                name=f"selfgr:anchor_level_blend:b{int(band)}",
                family="selfgr",
                tvt=_vp_compose_centered(level_anchor, blend),
            )
    return out


def _vp_select_diverse_candidates(records: pd.DataFrame, paths: dict[int, np.ndarray]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if records.empty or len(records) <= VP_DIVERSE_TOPK:
        return records.copy(), paths
    rows = records.sort_values(["score_rank", "candidate_idx"], ascending=[True, True]).copy()
    selected: list[int] = []

    def add_indices(indices) -> None:
        for idx in indices:
            ii = int(idx)
            if ii not in selected:
                selected.append(ii)
            if len(selected) >= int(VP_DIVERSE_TOPK):
                return

    add_indices(rows.head(max(0, int(VP_DIVERSE_SCORE_HEAD))).index)
    if len(selected) < int(VP_DIVERSE_TOPK) and "family" in rows.columns:
        for _family, group in rows.groupby("family", sort=False):
            add_indices(group.head(max(0, int(VP_DIVERSE_FAMILY_QUOTA))).index)
            if len(selected) >= int(VP_DIVERSE_TOPK):
                break
    if len(selected) < int(VP_DIVERSE_TOPK) and "mean_level_delta" in rows.columns:
        width = max(float(VP_DIVERSE_DELTA_BIN), 1e-6)
        work = rows.copy()
        delta = pd.to_numeric(work["mean_level_delta"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        work["_delta_bin"] = np.floor(delta / width).astype(int)
        bin_heads = work.sort_values(["score_rank", "candidate_idx"]).groupby("_delta_bin", sort=False).head(1)
        add_indices(bin_heads.sort_values(["score_rank", "candidate_idx"]).index)
        bin_seconds = work.drop(index=bin_heads.index, errors="ignore").sort_values(["score_rank", "candidate_idx"]).groupby("_delta_bin", sort=False).head(1)
        add_indices(bin_seconds.sort_values(["score_rank", "candidate_idx"]).index)
    if len(selected) < int(VP_DIVERSE_TOPK):
        add_indices(rows.index)
    out = rows.loc[selected[: int(VP_DIVERSE_TOPK)]].copy()
    out["_vp_diverse_rank"] = np.arange(1, len(out) + 1, dtype=int)
    out = out.sort_values(["_vp_diverse_rank"]).reset_index(drop=True)
    keep = set(out["candidate_idx"].astype(int).tolist())
    return out, {int(k): v for k, v in paths.items() if int(k) in keep}


def _vp_piecewise_candidates(md, z, anchor_tvt, last_known: float) -> list[dict[str, object]]:
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    anchor_tvt = np.asarray(anchor_tvt, dtype=np.float64).copy()
    if len(md) == 0:
        return []
    anchor_tvt[0] = float(last_known)
    n = len(md)
    knots = np.unique(np.round(np.linspace(0, n - 1, min(12, max(2, n)))).astype(int))
    rows = np.arange(n, dtype=np.float64)
    progress = np.zeros(n, dtype=np.float64)
    span = float(md[-1] - md[0])
    if abs(span) > 1e-8:
        progress = np.clip((md - float(md[0])) / span, 0.0, 1.0)
    anchor_u = anchor_tvt + z
    out: list[dict[str, object]] = []
    seen: set[bytes] = set()

    def add(name: str, perturb: np.ndarray, family: str) -> None:
        ctrl_u = anchor_u[knots] + perturb[knots]
        u = np.interp(rows, knots.astype(np.float64), ctrl_u)
        u[0] = float(last_known) + float(z[0])
        tvt = u - z
        tvt[0] = float(last_known)
        key = np.round(tvt, 3).astype(np.float32).tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append({"name": name, "family": family, "tvt": tvt.astype(np.float64)})

    add("anchor_exact", np.zeros(n, dtype=np.float64), "anchor")
    for level in VP_ENDPOINT_OFFSETS:
        add(f"level_shift:{level:+.1f}", np.full(n, float(level), dtype=np.float64), "level_shift")
    for endpoint in VP_ENDPOINT_OFFSETS:
        for slope in VP_SLOPE_OFFSETS:
            for curve in VP_CURVATURE_OFFSETS:
                perturb = (
                    float(endpoint) * progress
                    + float(slope) * (md - float(md[0]))
                    + float(curve) * np.sin(np.pi * progress)
                )
                add(f"piecewise_u:e{endpoint:+.1f}:s{slope:+.4f}:c{curve:+.1f}", perturb, "piecewise_u")
    return out


def _vp_score_path(tvt, obs_gr, tw_tvt, tw_gr, affine) -> dict[str, float]:
    exp = np.interp(np.asarray(tvt, dtype=np.float64), tw_tvt, tw_gr)
    slope, intercept, prefix_rmse, prefix_n = affine
    pred_gr = float(slope) * exp + float(intercept)
    obs = np.asarray(obs_gr, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(pred_gr)
    if int(mask.sum()) < 8:
        return {"score": float("inf"), "gr_rmse": float("inf"), "gr_corr": float("nan"), "prefix_rmse": float(prefix_rmse), "prefix_n": int(prefix_n)}
    resid = obs[mask] - pred_gr[mask]
    rmse = float(np.sqrt(np.mean(resid * resid)))
    corr = _vp_corr(obs[mask], pred_gr[mask])
    score = _vp_huber_score(resid)
    if np.isfinite(corr):
        score *= float(1.0 + 0.20 * max(0.0, 1.0 - corr))
    return {"score": float(score), "gr_rmse": rmse, "gr_corr": corr, "prefix_rmse": float(prefix_rmse), "prefix_n": int(prefix_n)}


def _vp_prefix_vector(hw: pd.DataFrame, row_idx: np.ndarray, cand_tvt: np.ndarray) -> np.ndarray:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    cand_tvt = np.asarray(cand_tvt, dtype=np.float64)
    if len(row_idx) == 0 or len(cand_tvt) == 0:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    start_row = int(row_idx[0])
    tvt_input = hw["TVT_input"].to_numpy(np.float64)[:start_row]
    known_mask = np.isfinite(tvt_input)
    if int(known_mask.sum()) < 2:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    known_md = hw["MD"].to_numpy(np.float64)[:start_row][known_mask]
    known_z = hw["Z"].to_numpy(np.float64)[:start_row][known_mask]
    known_tvt = tvt_input[known_mask]
    hidden_md = hw["MD"].to_numpy(np.float64)[row_idx]
    hidden_z = hw["Z"].to_numpy(np.float64)[row_idx]
    n = min(len(hidden_md), len(hidden_z), len(cand_tvt))
    if n == 0:
        return np.zeros(len(VP_PREFIX_WINDOWS) * 6, dtype=np.float64)
    hidden_md = hidden_md[:n]
    hidden_z = hidden_z[:n]
    cand_tvt = cand_tvt[:n]
    known_f = known_tvt + known_z
    cand_f = cand_tvt + hidden_z
    md_gap = float(hidden_md[0] - known_md[-1])
    vals: list[float] = []
    for win in VP_PREFIX_WINDOWS:
        ktvt_rate = _vp_tail_rate(known_md, known_tvt, tail=True, window=int(win))
        ctvt_rate = _vp_tail_rate(hidden_md, cand_tvt, tail=False, window=int(win))
        kf_rate = _vp_tail_rate(known_md, known_f, tail=True, window=int(win))
        cf_rate = _vp_tail_rate(hidden_md, cand_f, tail=False, window=int(win))
        vals.extend(
            [
                abs(float(cand_tvt[0] - (float(known_tvt[-1]) + ktvt_rate * md_gap))),
                abs(float(ctvt_rate - ktvt_rate)),
                abs(float(ctvt_rate - ktvt_rate)),
                abs(float(cand_f[0] - (float(known_f[-1]) + kf_rate * md_gap))),
                abs(float(cf_rate - kf_rate)),
                abs(float(cf_rate - kf_rate)),
            ]
        )
    return np.asarray(vals, dtype=np.float64)


def _vp_percentile_cost(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    n, m = mat.shape
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    costs = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        col = mat[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        work = np.where(finite, col, float(np.nanmax(col[finite]) + 1.0))
        order = np.argsort(work, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        costs[:, j] = ranks / float(max(n - 1, 1))
    return costs.mean(axis=1)


def _vp_robust_z(values, sign: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros(len(arr), dtype=np.float64)
    med = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - med)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return float(sign) * (np.where(np.isfinite(arr), arr, med) - med) / scale


def _vp_softmax(values, tau: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(len(x), 1.0 / float(len(x)), dtype=np.float64)
    z = np.where(finite, x, float(np.min(x[finite]) - 20.0)) / max(float(tau), 1e-9)
    z = z - float(np.max(z))
    w = np.exp(np.clip(z, -60.0, 60.0))
    return w / max(float(np.sum(w)), 1e-12)


def _vp_logsumexp(values, tau: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return -1e9
    t = max(float(tau), 1e-9)
    m = float(np.max(x))
    return float(m + t * np.log(np.sum(np.exp(np.clip((x - m) / t, -60.0, 0.0)))))


def _vp_assign_clusters(level_delta: np.ndarray, gap: float = VP_CLUSTER_GAP) -> np.ndarray:
    delta = np.asarray(level_delta, dtype=np.float64)
    if len(delta) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(delta, kind="mergesort")
    labels = np.zeros(len(delta), dtype=np.int64)
    cid = 0
    prev = float(delta[order[0]])
    for idx in order:
        val = float(delta[idx])
        if abs(val - prev) > float(gap):
            cid += 1
        labels[idx] = cid
        prev = val
    return labels


def _vp_candidate_records_for_well(wid: str, grp: pd.DataFrame, hw=None, tw=None) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    if hw is None:
        hw = pd.read_csv(DATA_ROOT / "test" / f"{wid}__horizontal_well.csv")
    if tw is None:
        tw = pd.read_csv(DATA_ROOT / "test" / f"{wid}__typewell.csv")
    row_idx = grp["_row_idx"].to_numpy(np.int64)
    row_idx = row_idx[(row_idx >= 0) & (row_idx < len(hw))]
    if len(row_idx) == 0:
        raise RuntimeError(f"empty row index for {wid}")
    anchor = grp.set_index("_row_idx").reindex(row_idx)["tvt"].to_numpy(np.float64)
    anchor = pd.Series(anchor).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float64)
    start_row = int(row_idx[0])
    tvt_input = hw["TVT_input"].to_numpy(np.float64)
    known = tvt_input[np.isfinite(tvt_input)]
    last_known = float(known[-1]) if len(known) else float(anchor[0])
    md_hidden = hw["MD"].to_numpy(np.float64)[row_idx]
    z_hidden = hw["Z"].to_numpy(np.float64)[row_idx]
    gr_hidden = _vp_roll(_vp_fill(hw["GR"].to_numpy(np.float64)[row_idx]), VP_GR_WINDOW)
    tw_tvt, tw_gr = _vp_typewell_arrays(tw)
    if len(tw_tvt) < 8:
        lo = float(np.nanmin(anchor)) - 1.0
        hi = float(np.nanmax(anchor)) + 1.0
        tw_tvt = np.array([lo, hi], dtype=np.float64)
        tw_gr = np.full(2, float(np.nanmedian(gr_hidden)) if np.isfinite(gr_hidden).any() else 0.0, dtype=np.float64)
    prefix_tvt = tvt_input[:start_row]
    prefix_gr = _vp_roll(_vp_fill(hw["GR"].to_numpy(np.float64)[:start_row]), VP_GR_WINDOW)
    pmask = np.isfinite(prefix_tvt) & np.isfinite(prefix_gr)
    affine = _vp_fit_heel_affine(np.interp(prefix_tvt[pmask], tw_tvt, tw_gr), prefix_gr[pmask]) if int(pmask.sum()) >= 8 else (1.0, 0.0, float("nan"), 0)

    if start_row > 0:
        md_context = np.concatenate([[float(hw["MD"].iloc[start_row - 1])], md_hidden])
        z_context = np.concatenate([[float(hw["Z"].iloc[start_row - 1])], z_hidden])
        anchor_context = np.concatenate([[last_known], anchor])
    else:
        md_context = md_hidden
        z_context = z_hidden
        anchor_context = anchor
    context = _vp_piecewise_candidates(md_context, z_context, anchor_context, last_known)
    candidates: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for cand in context:
        tvt = np.asarray(cand["tvt"], dtype=np.float64)[1:] if start_row > 0 else np.asarray(cand["tvt"], dtype=np.float64)
        if len(tvt) != len(anchor):
            continue
        _vp_append_unique(candidates, seen, name=str(cand["name"]), family=str(cand["family"]), tvt=tvt)

    known_mask = np.isfinite(tvt_input[:start_row])
    candidates.extend(_vp_level_grid_candidates(anchor, z_hidden, seen))
    candidates.extend(
        _vp_prefix_u_rate_candidates(
            hw["MD"].to_numpy(np.float64)[:start_row][known_mask],
            tvt_input[:start_row][known_mask],
            hw["Z"].to_numpy(np.float64)[:start_row][known_mask],
            md_hidden,
            z_hidden,
            anchor,
            seen,
        )
    )
    candidates.extend(_vp_formation_contact_candidates(hw, tw, row_idx, start_row, anchor, z_hidden, seen))
    candidates.extend(
        _vp_selfgr_candidates(
            tvt_input[:start_row],
            hw["GR"].to_numpy(np.float64)[:start_row],
            gr_hidden,
            anchor,
            z_hidden,
            last_known=last_known,
            seen=seen,
        )
    )
    rows: list[dict[str, object]] = []
    paths: dict[int, np.ndarray] = {}
    for ci, cand in enumerate(candidates):
        tvt = np.asarray(cand["tvt"], dtype=np.float64)
        if len(tvt) != len(anchor):
            continue
        sc = _vp_score_path(tvt, gr_hidden, tw_tvt, tw_gr, affine)
        delta = tvt - anchor
        first = np.diff(tvt)
        second = np.diff(tvt, n=2)
        paths[ci] = tvt.astype(np.float64)
        rows.append(
            {
                "well_id": str(wid),
                "candidate_idx": int(ci),
                "candidate_name": str(cand["name"]),
                "family": str(cand["family"]),
                "score": float(sc["score"]),
                "gr_rmse": float(sc["gr_rmse"]),
                "gr_corr": float(sc["gr_corr"]),
                "prefix_rmse": float(sc["prefix_rmse"]),
                "prefix_n": int(sc["prefix_n"]),
                "mean_level_delta": float(np.mean(delta)),
                "end_delta": float(delta[-1]) if len(delta) else 0.0,
                "path_d1_std": float(np.std(first)) if len(first) else 0.0,
                "path_smoothness": float(np.sqrt(np.mean(second * second))) if len(second) else 0.0,
            }
        )
    records = pd.DataFrame(rows)
    if records.empty:
        raise RuntimeError("no live VP candidates")
    score_arr = records["score"].to_numpy(np.float64)
    good = np.isfinite(score_arr)
    centered = score_arr[good] - float(np.min(score_arr[good])) if good.any() else np.zeros(0)
    temp = float(np.median(centered[centered > 0])) if np.any(centered > 0) else 1.0
    post = np.zeros(len(records), dtype=np.float64)
    if good.any():
        w = np.exp(-np.clip(centered / max(temp, 1e-9), 0.0, 700.0))
        post[good] = w / max(float(w.sum()), 1e-12)
    else:
        post[:] = 1.0 / float(len(records))
    records["posterior"] = post
    records = records.sort_values(["score"], ascending=[True]).reset_index(drop=True)
    records["score_rank"] = np.arange(1, len(records) + 1)
    records["ms_score"] = records["score"].astype(float)
    records["ms_corr_mean"] = records["gr_corr"].astype(float)
    records, paths = _vp_select_diverse_candidates(records, paths)
    return records, paths


def _vp_prior_score(records: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(records), dtype=np.float64)
    score += 0.45 * _vp_robust_z(records["score"].to_numpy(np.float64), sign=-1.0)
    post = pd.to_numeric(records["posterior"], errors="coerce").fillna(0.0).clip(lower=1e-8).to_numpy(np.float64)
    score += 0.30 * _vp_robust_z(np.log(post), sign=1.0)
    score += 0.15 * _vp_robust_z(records["score_rank"].to_numpy(np.float64), sign=-1.0)
    score += 0.10 * _vp_robust_z(records["ms_corr_mean"].to_numpy(np.float64), sign=1.0)
    return score


def _vp_apply_ramped_move(base: np.ndarray, candidate: np.ndarray, *, alpha: float = VP_ALPHA, clip: float = VP_CLIP) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    n = min(len(base), len(cand))
    out = base.copy()
    if n == 0:
        return out
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    out[:n] = base[:n] + float(alpha) * ramp * np.clip(cand[:n] - base[:n], -float(clip), float(clip))
    return out


def apply_visible_prefix_candidate_posterior(source: pd.DataFrame) -> np.ndarray:
    work = source.copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    work["_row_idx"] = work["id"].astype(str).str.split("_").str[-1].astype(int)
    out = work["tvt"].to_numpy(np.float64).copy()
    n_used = 0
    n_failed = 0
    for wid, grp0 in work.groupby("well", sort=False):
        grp = grp0.sort_values("_row_idx").reset_index(drop=True)
        pos = grp["_pos"].to_numpy(np.int64)
        base = grp["tvt"].to_numpy(np.float64)
        try:
            records, paths = _vp_candidate_records_for_well(str(wid), grp)
            hw = pd.read_csv(DATA_ROOT / "test" / f"{wid}__horizontal_well.csv")
            row_idx = grp["_row_idx"].to_numpy(np.int64)
            prefix_cost = _vp_percentile_cost(np.vstack([_vp_prefix_vector(hw, row_idx, paths[int(ci)]) for ci in records["candidate_idx"].astype(int)]))
            post = pd.to_numeric(records["posterior"], errors="coerce").fillna(0.0).clip(lower=1e-8).to_numpy(np.float64)
            records["_vp_score"] = (
                -VP_PREFIX_BETA * prefix_cost
                + VP_PATH_PRIOR_WEIGHT * _vp_prior_score(records)
                + VP_POSTERIOR_PRIOR_WEIGHT * _vp_robust_z(np.log(post), sign=1.0)
            )
            records["_vp_prefix_cost"] = prefix_cost
            records["_vp_cluster"] = _vp_assign_clusters(records["mean_level_delta"].to_numpy(np.float64), VP_CLUSTER_GAP)
            cluster_rows = []
            for cid, cgrp in records.groupby("_vp_cluster", sort=False):
                cluster_rows.append({"cid": int(cid), "score": _vp_logsumexp(cgrp["_vp_score"].to_numpy(np.float64), VP_TAU_CLUSTER), "prefix": float(cgrp["_vp_prefix_cost"].min())})
            clusters = pd.DataFrame(cluster_rows).sort_values(["score", "prefix"], ascending=[False, True]).head(VP_TOP_CLUSTERS)
            cw = _vp_softmax(clusters["score"].to_numpy(np.float64), VP_TAU_CLUSTER)
            posterior_path = np.zeros(len(base), dtype=np.float64)
            for cweight, cid in zip(cw, clusters["cid"].astype(int).to_numpy()):
                members = records[records["_vp_cluster"].astype(int).eq(int(cid))].sort_values(
                    ["_vp_score", "_vp_prefix_cost", "score_rank"],
                    ascending=[False, True, True],
                ).head(VP_MEMBER_TOP)
                mw = _vp_softmax(members["_vp_score"].to_numpy(np.float64), VP_TAU_MEMBER)
                one = np.zeros(len(base), dtype=np.float64)
                for mweight, row in zip(mw, members.itertuples(index=False)):
                    one += float(mweight) * paths[int(row.candidate_idx)]
                posterior_path += float(cweight) * one
            out[pos] = _vp_apply_ramped_move(base, posterior_path, alpha=VP_ALPHA, clip=VP_CLIP)
            n_used += 1
        except Exception as exc:
            n_failed += 1
            print(f"  visible-prefix candidate posterior skipped {wid}: {exc}", flush=True)
            out[pos] = base
    print(
        f"visible-prefix candidate posterior used_wells={n_used} failed_wells={n_failed} "
        f"prefix_beta={VP_PREFIX_BETA:.2f} path_prior_w={VP_PATH_PRIOR_WEIGHT:.2f} "
        f"posterior_prior_w={VP_POSTERIOR_PRIOR_WEIGHT:.2f} topk={VP_DIVERSE_TOPK} "
        f"alpha={VP_ALPHA:.2f} clip={VP_CLIP:.1f}",
        flush=True,
    )
    return out.astype(np.float32)


def _seg_segment_bounds(length: int, segment_len: int) -> list[tuple[int, int]]:
    n = int(length)
    step = max(int(segment_len), 1)
    return [(start, min(start + step, n)) for start in range(0, n, step) if start < n]


def _seg_logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray(-np.inf, dtype=np.float64)
    mx = np.nanmax(arr, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(np.clip(arr - mx, -80.0, 0.0)), axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(out.reshape(()), dtype=np.float64)
    return np.squeeze(out, axis=axis)


def _seg_forward_backward(emissions: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    emit = np.asarray(emissions, dtype=np.float64)
    if emit.ndim != 2:
        raise ValueError(f"seg emissions must be 2D, got {emit.shape}")
    n_seg, n_state = emit.shape
    if n_seg == 0 or n_state == 0:
        return np.zeros_like(emit, dtype=np.float64)
    trans = np.asarray(transitions, dtype=np.float64)
    if trans.ndim == 2:
        trans = np.repeat(trans[None, :, :], max(n_seg - 1, 0), axis=0)
    if trans.shape != (max(n_seg - 1, 0), n_state, n_state):
        raise ValueError(f"seg transitions shape {trans.shape} incompatible with emissions {emit.shape}")

    alpha = np.empty_like(emit, dtype=np.float64)
    beta = np.zeros_like(emit, dtype=np.float64)
    alpha[0] = emit[0]
    alpha[0] -= float(_seg_logsumexp(alpha[0]))
    for t in range(1, n_seg):
        alpha[t] = emit[t] + _seg_logsumexp(alpha[t - 1][:, None] + trans[t - 1], axis=0)
        alpha[t] -= float(_seg_logsumexp(alpha[t]))
    for t in range(n_seg - 2, -1, -1):
        beta[t] = _seg_logsumexp(trans[t] + emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)
        beta[t] -= float(_seg_logsumexp(beta[t]))

    log_post = alpha + beta
    log_post -= _seg_logsumexp(log_post, axis=1)[:, None]
    post = np.exp(np.clip(log_post, -80.0, 0.0))
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-12)
    return post


def _seg_rank01_cost(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    order = np.argsort(np.where(finite, arr, fill), kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(arr), dtype=np.float64)
    return ranks


def _seg_formation_sequence_from_typewell(tw: pd.DataFrame) -> dict[str, np.ndarray]:
    if "TVT" not in tw.columns:
        return {"tvt": np.asarray([], dtype=np.float64), "label_codes": np.asarray([], dtype=np.int16), "core_codes": np.asarray([], dtype=np.int16)}
    frame = tw.copy()
    if "Geology" not in frame.columns:
        frame["Geology"] = "UNK"
    frame = frame[["TVT", "Geology"]].dropna(subset=["TVT"]).sort_values("TVT").drop_duplicates("TVT")
    if frame.empty:
        return {"tvt": np.asarray([], dtype=np.float64), "label_codes": np.asarray([], dtype=np.int16), "core_codes": np.asarray([], dtype=np.int16)}
    labels = frame["Geology"].fillna("UNK").astype(str).to_numpy()
    label_map: dict[str, int] = {}
    label_codes = np.empty(len(labels), dtype=np.int16)
    core_codes = np.empty(len(labels), dtype=np.int16)
    for i, label in enumerate(labels):
        if label not in label_map:
            label_map[label] = len(label_map)
        label_codes[i] = int(label_map[label])
        key = label.upper()
        core_codes[i] = int(SEG_FORMATION_ORDER.get(key, -1))
    return {
        "tvt": frame["TVT"].to_numpy(np.float64),
        "label_codes": label_codes,
        "core_codes": core_codes,
    }


def _seg_seq_index(formation_sequence: dict[str, np.ndarray], tvt_values: np.ndarray) -> np.ndarray:
    tvt_grid = np.asarray(formation_sequence.get("tvt", np.asarray([], dtype=np.float64)), dtype=np.float64)
    vals = np.asarray(tvt_values, dtype=np.float64)
    if len(tvt_grid) == 0:
        return np.zeros(vals.shape, dtype=np.int64)
    right = np.clip(np.searchsorted(tvt_grid, vals, side="left"), 0, len(tvt_grid) - 1)
    left = np.clip(right - 1, 0, len(tvt_grid) - 1)
    choose_left = np.abs(vals - tvt_grid[left]) <= np.abs(vals - tvt_grid[right])
    return np.where(choose_left, left, right).astype(np.int64)


def _seg_geology_cost(
    path_matrix: np.ndarray,
    bounds: list[tuple[int, int]],
    formation_sequence: dict[str, np.ndarray],
) -> np.ndarray:
    paths = np.asarray(path_matrix, dtype=np.float64)
    m = int(paths.shape[0]) if paths.ndim == 2 else 0
    costs = np.zeros((len(bounds), m), dtype=np.float64)
    if m == 0 or len(formation_sequence.get("tvt", [])) == 0:
        return costs
    label_codes = np.asarray(formation_sequence["label_codes"], dtype=np.int16)
    core_codes = np.asarray(formation_sequence["core_codes"], dtype=np.int16)
    for si, (start, end) in enumerate(bounds):
        if end - start <= 1:
            continue
        raw = np.zeros(m, dtype=np.float64)
        for j in range(m):
            idx = _seg_seq_index(formation_sequence, paths[j, start:end])
            form = label_codes[idx]
            core = core_codes[idx]
            runs = 1 + int(np.sum(np.diff(form) != 0)) if len(form) else 0
            unique = len(np.unique(form)) if len(form) else 0
            revisit = float(max(runs - unique, 0) / max(runs, 1))
            valid_core = core[core >= 0].astype(np.float64)
            step_down = float(np.mean(np.diff(valid_core) < 0.0)) if len(valid_core) > 1 else 0.0
            raw[j] = 0.70 * step_down + 0.30 * revisit
        costs[si] = _seg_rank01_cost(raw)
    return costs


def _seg_transition_tensor(
    path_matrix: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    transition_sigma: float = SEG_TRANSITION_SIGMA,
    jump_sigma: float = SEG_JUMP_SIGMA,
) -> np.ndarray:
    paths = np.asarray(path_matrix, dtype=np.float64)
    m = int(paths.shape[0]) if paths.ndim == 2 else 0
    if len(bounds) <= 1 or m == 0:
        return np.zeros((0, m, m), dtype=np.float64)
    trans = np.empty((len(bounds) - 1, m, m), dtype=np.float64)
    sig = max(float(transition_sigma), 1e-6)
    jsig = max(float(jump_sigma), 1e-6)
    for si in range(len(bounds) - 1):
        prev = bounds[si]
        nxt = bounds[si + 1]
        prev_end = paths[:, max(prev[1] - 1, prev[0])]
        next_start = paths[:, nxt[0]]
        prev_mean = paths[:, prev[0] : prev[1]].mean(axis=1)
        next_mean = paths[:, nxt[0] : nxt[1]].mean(axis=1)
        continuity = np.abs(prev_end[:, None] - next_start[None, :]) / sig
        jump = np.abs(prev_mean[:, None] - next_mean[None, :]) / jsig
        trans[si] = -(continuity + 0.35 * jump)
    return trans


def _seg_cluster_entropy(records: pd.DataFrame) -> float:
    if records.empty or "mean_level_delta" not in records.columns or "_vp_score" not in records.columns:
        return 1.0
    deltas = pd.to_numeric(records["mean_level_delta"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    clusters = _vp_assign_clusters(deltas, SEG_CLUSTER_GAP)
    scores = []
    for cid in np.unique(clusters):
        vals = records.loc[clusters == cid, "_vp_score"].to_numpy(np.float64)
        scores.append(_vp_logsumexp(vals, SEG_CLUSTER_TAU_LSE))
    score_arr = np.asarray(scores, dtype=np.float64)
    score_arr = score_arr[np.isfinite(score_arr)]
    if len(score_arr) <= 1:
        return 0.0
    centered = score_arr - float(np.max(score_arr))
    probs = np.exp(np.clip(centered, -80.0, 0.0))
    probs /= max(float(probs.sum()), 1e-12)
    ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return float(np.clip(ent / np.log(float(len(probs))), 0.0, 1.0))


def _seg_posterior_path_from_matrix(
    path_matrix: np.ndarray,
    bounds: list[tuple[int, int]],
    emissions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(path_matrix, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    emit = np.asarray(emissions, dtype=np.float64) / max(float(SEG_TEMPERATURE), 1e-6)
    trans = _seg_transition_tensor(paths, bounds)
    posterior = _seg_forward_backward(emit, trans)
    pred = np.zeros(paths.shape[1], dtype=np.float64)
    for si, (start, end) in enumerate(bounds):
        pred[start:end] = posterior[si] @ paths[:, start:end]
    return pred, posterior


def _seg_cached_context_for_well(wid: str, grp: pd.DataFrame) -> dict[str, object]:
    cache_key = str(wid)
    cached = SEGMENTED_POSTERIOR_CACHE.get(cache_key)
    if cached is not None and int(cached.get("n_rows", -1)) == int(len(grp)):
        return cached

    hw = pd.read_csv(DATA_ROOT / "test" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_ROOT / "test" / f"{wid}__typewell.csv")
    row_idx = grp["_row_idx"].to_numpy(np.int64)
    records, paths = _vp_candidate_records_for_well(str(wid), grp, hw=hw, tw=tw)
    prefix_matrix = np.vstack([_vp_prefix_vector(hw, row_idx, paths[int(ci)]) for ci in records["candidate_idx"].astype(int)])
    prefix_cost = _vp_percentile_cost(prefix_matrix)
    post = pd.to_numeric(records["posterior"], errors="coerce").fillna(0.0).clip(lower=1e-8).to_numpy(np.float64)
    records = records.copy()
    records["_vp_score"] = (
        -VP_PREFIX_BETA * prefix_cost
        + VP_PATH_PRIOR_WEIGHT * _vp_prior_score(records)
        + VP_POSTERIOR_PRIOR_WEIGHT * _vp_robust_z(np.log(post), sign=1.0)
    )
    records["_vp_prefix_cost"] = prefix_cost
    cluster_entropy_norm = _seg_cluster_entropy(records)
    selected = records.sort_values(
        ["_vp_score", "_vp_prefix_cost", "score_rank"],
        ascending=[False, True, True],
    ).head(max(2, int(SEG_MAX_CANDIDATES))).reset_index(drop=True)
    candidate_idx = selected["candidate_idx"].astype(int).to_numpy()
    path_matrix = np.vstack([paths[int(ci)] for ci in candidate_idx]).astype(np.float64)
    if path_matrix.shape[1] != len(grp):
        raise RuntimeError(f"seg path matrix width {path_matrix.shape[1]} != group rows {len(grp)}")
    bounds = _seg_segment_bounds(path_matrix.shape[1], SEG_LEN)
    formation_sequence = _seg_formation_sequence_from_typewell(tw)
    segment_costs = _seg_geology_cost(path_matrix, bounds, formation_sequence)
    context = {
        "n_rows": int(len(grp)),
        "records": selected,
        "candidate_idx": candidate_idx,
        "path_matrix": path_matrix,
        "bounds": bounds,
        "formation_sequence": formation_sequence,
        "segment_costs": segment_costs,
        "cluster_entropy_norm": float(cluster_entropy_norm),
    }
    SEGMENTED_POSTERIOR_CACHE[cache_key] = context
    return context


def apply_segmented_geology_posterior_overlay(source: pd.DataFrame) -> np.ndarray:
    if not SEG_ENABLED:
        return apply_visible_prefix_candidate_posterior(source)
    work = source.copy()
    work["_pos"] = np.arange(len(work), dtype=np.int64)
    work["_row_idx"] = work["id"].astype(str).str.split("_").str[-1].astype(int)
    out = work["tvt"].to_numpy(np.float64).copy()
    n_used = 0
    n_failed = 0
    n_entropy_skipped = 0
    posterior_entropy: list[float] = []
    for wid, grp0 in work.groupby("well", sort=False):
        grp = grp0.sort_values("_row_idx").reset_index(drop=True)
        pos = grp["_pos"].to_numpy(np.int64)
        base = grp["tvt"].to_numpy(np.float64)
        try:
            context = _seg_cached_context_for_well(str(wid), grp)
            records = context["records"]
            path_matrix = np.asarray(context["path_matrix"], dtype=np.float64)
            bounds = context["bounds"]
            segment_costs = np.asarray(context["segment_costs"], dtype=np.float64)
            cluster_entropy = float(context.get("cluster_entropy_norm", 1.0))
            if cluster_entropy > float(SEG_CLUSTER_ENTROPY_MAX):
                out[pos] = base
                n_entropy_skipped += 1
                continue
            score_z = _vp_robust_z(records["_vp_score"].to_numpy(np.float64), sign=1.0)
            emissions = np.repeat((SEG_SCORE_WEIGHT * score_z)[None, :], len(bounds), axis=0)
            emissions -= float(SEG_GEOLOGY_BETA) * segment_costs
            posterior_path, posterior = _seg_posterior_path_from_matrix(path_matrix, bounds, emissions)
            context["segment_posterior"] = posterior.astype(np.float32)
            if posterior.size:
                posterior_entropy.append(float(np.mean(-np.sum(posterior * np.log(posterior + 1e-12), axis=1) / np.log(max(posterior.shape[1], 2)))))
            out[pos] = _vp_apply_ramped_move(base, posterior_path, alpha=SEG_ALPHA, clip=SEG_CLIP)
            n_used += 1
        except Exception as exc:
            n_failed += 1
            print(f"  segmented posterior overlay skipped {wid}: {exc}", flush=True)
            out[pos] = base
    mean_entropy = float(np.mean(posterior_entropy)) if posterior_entropy else float("nan")
    print(
        f"segmented posterior overlay used_wells={n_used} failed_wells={n_failed} "
        f"entropy_skipped={n_entropy_skipped} cluster_entropy_max={SEG_CLUSTER_ENTROPY_MAX:.2f} "
        f"cache_wells={len(SEGMENTED_POSTERIOR_CACHE)} seg_len={SEG_LEN} max_candidates={SEG_MAX_CANDIDATES} "
        f"score_weight={SEG_SCORE_WEIGHT:.2f} geology_beta={SEG_GEOLOGY_BETA:.2f} "
        f"trans_sigma={SEG_TRANSITION_SIGMA:.1f} jump_sigma={SEG_JUMP_SIGMA:.1f} "
        f"alpha={SEG_ALPHA:.2f} clip={SEG_CLIP:.1f} mean_entropy={mean_entropy:.4f}",
        flush=True,
    )
    return out.astype(np.float32)


def _softmax_from_liks(liks: np.ndarray, scale: float) -> np.ndarray:
    x = (np.asarray(liks, dtype=np.float64) - float(np.max(liks))) / float(scale)
    x = np.clip(x, -700.0, 0.0)
    w = np.exp(x)
    s = float(w.sum())
    if s <= 0.0:
        return np.full_like(w, 1.0 / max(len(w), 1), dtype=np.float64)
    return w / s


def _weighted_quantile_seed(preds: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(preds, axis=0)
    vals = np.take_along_axis(preds, order, axis=0)
    ww = weights[order]
    cdf = np.cumsum(ww, axis=0)
    idx = np.argmax(cdf >= float(q), axis=0)
    return vals[idx, np.arange(preds.shape[1])]


def _trimmed_weighted_mean(preds: np.ndarray, weights: np.ndarray, lo: float, hi: float) -> np.ndarray:
    order = np.argsort(preds, axis=0)
    vals = np.take_along_axis(preds, order, axis=0)
    ww = weights[order]
    cdf = np.cumsum(ww, axis=0)
    keep = (cdf >= float(lo)) & (cdf <= float(hi))
    empty = ~keep.any(axis=0)
    if empty.any():
        mid = np.argmax(cdf[:, empty] >= 0.5, axis=0)
        keep[:, empty] = False
        keep[mid, np.flatnonzero(empty)] = True
    ww_keep = np.where(keep, ww, 0.0)
    denom = ww_keep.sum(axis=0)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (vals * ww_keep).sum(axis=0) / denom


# ============================================================================
# GPU particle-filter for engine A (batched torch twin of rl.run_particle_filter).
# Same constants / update order as ridge_lib.run_particle_filter (statistically
# equivalent, not bit-identical: device RNG stream differs). Produces the SAME
# pred[n_seeds, n_ev] + liks[n_seeds] the numpy seed loop produced, ~100x faster.
# Gated on cuda; ANY failure falls back to the exact numpy loop (== 6.903).
# ============================================================================
_PF_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_PF_GRID_STEP = 0.5
_PF_MOM, _PF_VN, _PF_PN, _PF_RP, _PF_RR, _PF_RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5
_PF_SEED = 1745


def _pf_pack(hw: pd.DataFrame, tw: pd.DataFrame) -> dict:
    """Build the per-well pack (mirrors rl.run_particle_filter setup exactly)."""
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    n = len(ev)
    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"]); last_Z = float(last["Z"]); last_MD = float(last["MD"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy(float) - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    md_v = ev["MD"].to_numpy(float)
    z_h = ev["Z"].to_numpy(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    gr_h = gr_interp[ev.index.to_numpy()]
    dm_h = np.empty(n)
    dm_h[0] = md_v[0] - last_MD
    if n > 1:
        dm_h[1:] = np.diff(md_v)
    dm_h = np.maximum(dm_h, 1.0)
    gmin = tw_tvt[0] - 100.0; gmax = tw_tvt[-1] + 100.0
    grid = np.arange(gmin, gmax + _PF_GRID_STEP, _PF_GRID_STEP)
    grid_gr = np.interp(grid, tw_tvt, tw_gr)
    return dict(n=n, last_tvt=last_tvt, last_Z=last_Z, gs=gs, ir=ir,
                z_h=z_h, gr_h=gr_h, dm_h=dm_h, gmin=gmin, gmax=gmax, grid_gr=grid_gr)


@torch.no_grad()
def _pf_run_chunk(packs: list, n_seeds: int, n_particles: int, gen) -> tuple:
    """Batched PF over a chunk of wells -> res[W,S,Lmax] (TVT) + loglik[W,S]."""
    DEV = _PF_DEV
    W = len(packs); S = n_seeds; N = n_particles
    Lmax = max(p["n"] for p in packs)
    Gmax = max(len(p["grid_gr"]) for p in packs)
    z_h = torch.zeros(W, Lmax, device=DEV); gr_h = torch.zeros(W, Lmax, device=DEV)
    dm_h = torch.ones(W, Lmax, device=DEV); valid = torch.zeros(W, Lmax, device=DEV)
    grid = torch.zeros(W, Gmax, device=DEV)
    gmin = torch.zeros(W, device=DEV); gmax = torch.zeros(W, device=DEV)
    last_tvt = torch.zeros(W, device=DEV); last_Z = torch.zeros(W, device=DEV)
    gs = torch.zeros(W, device=DEV); ir = torch.zeros(W, device=DEV)
    for i, p in enumerate(packs):
        n = p["n"]; g = len(p["grid_gr"])
        z_h[i, :n] = torch.tensor(p["z_h"], device=DEV); gr_h[i, :n] = torch.tensor(p["gr_h"], device=DEV)
        dm_h[i, :n] = torch.tensor(p["dm_h"], device=DEV); valid[i, :n] = 1.0
        grid[i, :g] = torch.tensor(p["grid_gr"], device=DEV); grid[i, g:] = p["grid_gr"][-1]
        gmin[i] = p["gmin"]; gmax[i] = p["gmax"]; last_tvt[i] = p["last_tvt"]
        last_Z[i] = p["last_Z"]; gs[i] = p["gs"]; ir[i] = p["ir"]
    Wd = grid.shape[1]
    grid_flat = grid.reshape(-1)
    wbase = (torch.arange(W, device=DEV) * Wd).view(W, 1, 1)

    def interp(tvt_p):
        idxf = (tvt_p - gmin.view(W, 1, 1)) / _PF_GRID_STEP
        i0 = idxf.floor().clamp(0, Wd - 2).long(); frac = (idxf - i0).clamp(0, 1)
        flat0 = (wbase + i0).reshape(-1)
        g0 = grid_flat[flat0].reshape(W, S, N); g1 = grid_flat[flat0 + 1].reshape(W, S, N)
        return g0 * (1 - frac) + g1 * frac

    pos = last_tvt.view(W, 1, 1) + last_Z.view(W, 1, 1) + 2.0 * torch.randn(W, S, N, device=DEV, generator=gen)
    rate = ir.view(W, 1, 1) + 0.01 * torch.randn(W, S, N, device=DEV, generator=gen)
    w = torch.full((W, S, N), 1.0 / N, device=DEV)
    loglik = torch.zeros(W, S, device=DEV)
    res = torch.zeros(W, S, Lmax, device=DEV)
    arangeN = torch.arange(N, device=DEV).float()
    gmin_e = gmin.view(W, 1, 1); gmax_e = gmax.view(W, 1, 1); gs_e = gs.view(W, 1, 1)
    for t in range(Lmax):
        vt = valid[:, t].view(W, 1, 1)
        z_t = z_h[:, t].view(W, 1, 1); gr_t = gr_h[:, t].view(W, 1, 1); dm_t = dm_h[:, t].view(W, 1, 1)
        rate_n = _PF_MOM * rate + _PF_VN * torch.randn(W, S, N, device=DEV, generator=gen)
        pos_n = pos + rate_n * dm_t + _PF_PN * torch.randn(W, S, N, device=DEV, generator=gen)
        tvt_p = torch.clamp(pos_n - z_t, gmin_e, gmax_e); pos_n = tvt_p + z_t
        eg = interp(tvt_p)
        d = (gr_t - eg) / gs_e
        lk = torch.exp(-0.5 * torch.clamp(d * d, max=600.)).clamp_min(1e-300)
        avg_lk = (w * lk).sum(-1)
        loglik = loglik + torch.where(valid[:, t].bool().view(W, 1), torch.log(avg_lk.clamp_min(1e-300)), torch.zeros_like(avg_lk))
        w_n = w * lk; ws = w_n.sum(-1, keepdim=True)
        w_n = torch.where(ws > 0, w_n / ws, torch.full_like(w_n, 1.0 / N))
        n_eff = 1.0 / (w_n * w_n).sum(-1)
        do_rs = (n_eff < _PF_RESAMP * N)
        cum = torch.cumsum(w_n, dim=-1)
        u0 = torch.rand(W, S, 1, device=DEV, generator=gen) / N
        positions = u0 + arangeN.view(1, 1, N) / N
        idx = torch.searchsorted(cum, positions).clamp(0, N - 1)
        pos_rs = torch.gather(pos_n, -1, idx) + _PF_RP * torch.randn(W, S, N, device=DEV, generator=gen)
        rate_rs = torch.gather(rate_n, -1, idx) + _PF_RR * torch.randn(W, S, N, device=DEV, generator=gen)
        w_rs = torch.full_like(w_n, 1.0 / N)
        rs = (do_rs & valid[:, t].bool().view(W, 1)).view(W, S, 1)
        pos_f = torch.where(rs, pos_rs, pos_n); rate_f = torch.where(rs, rate_rs, rate_n)
        w_f = torch.where(rs, w_rs, w_n)
        res[:, :, t] = (w_f * (pos_f - z_t)).sum(-1)
        pos = torch.where(vt.bool(), pos_f, pos); rate = torch.where(vt.bool(), rate_f, rate)
        w = torch.where(vt.bool(), w_f, w)
    return res, loglik


def _pf_seed_bank(hw, tw, ev, last_known, n_particles, n_seeds):
    """GPU seed bank -> (pred[n_seeds, n_ev], liks[n_seeds]); numpy fallback on any error."""
    try:
        if _PF_DEV != "cuda":
            raise RuntimeError("no cuda device")
        pack = _pf_pack(hw, tw)
        n = pack["n"]
        gen = torch.Generator(device=_PF_DEV); gen.manual_seed(_PF_SEED)
        res, loglik = _pf_run_chunk([pack], n_seeds, n_particles, gen)
        pred = (res[0, :, :n].detach().cpu().numpy() - last_known).astype(np.float32)
        liks = loglik[0].detach().cpu().numpy().astype(np.float64)
        if pred.shape != (n_seeds, n) or not np.isfinite(pred).all() or not np.isfinite(liks).all():
            raise RuntimeError(f"bad gpu pred shape={pred.shape} expected={(n_seeds, n)}")
        return pred, liks
    except Exception as e:
        print(f"  [engine-A GPU PF fallback -> numpy: {e}]", flush=True)
        preds = []; liks = []
        for seed in range(n_seeds):
            pred_full, ll = rl.run_particle_filter(hw, tw, n_particles=n_particles, seed=seed)
            preds.append(pred_full[ev.index] - last_known)
            liks.append(ll)
        return np.stack(preds, axis=0).astype(np.float32), np.asarray(liks, dtype=np.float64)


def run_pf_posterior_summary(hw: pd.DataFrame, tw: pd.DataFrame, n_particles: int = 500, n_seeds: int = 128) -> dict[str, np.ndarray]:
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return {name: np.empty(0, dtype=np.float32) for name in POSTERIOR_COLS}
    last_known = float(kn.iloc[-1]["TVT_input"])
    pred, liks = _pf_seed_bank(hw, tw, ev, last_known, n_particles, n_seeds)

    out: dict[str, np.ndarray] = {}
    out["pf_seed_mean"] = pred.mean(axis=0)
    out["pf_seed_median"] = np.median(pred, axis=0)
    top_order = np.argsort(liks)[::-1]
    for k in (1, 2, 4, 8, 16, 32):
        kk = min(k, n_seeds)
        out[f"pf_top{kk}_mean"] = pred[top_order[:kk]].mean(axis=0)

    tvt_beam_full = rl.run_beam_ensemble(hw, tw)
    tvt_beam = tvt_beam_full[ev.index] - last_known
    out["beam_mean"] = tvt_beam.astype(np.float32)
    pf_by_scale_full: dict[str, np.ndarray] = {}
    for scale in rl.SELECTOR_SCALES:
        wt = _softmax_from_liks(liks, scale)
        base = (wt[:, None] * pred).sum(axis=0)
        name = f"pf_scale_{scale:g}"
        out[name] = base.astype(np.float32)
        full = hw["TVT_input"].to_numpy(float).copy()
        full[ev.index] = base + last_known
        pf_by_scale_full[name] = full

        out[f"{name}_wmedian"] = _weighted_quantile_seed(pred, wt, 0.5).astype(np.float32)
        out[f"{name}_q40"] = _weighted_quantile_seed(pred, wt, 0.4).astype(np.float32)
        out[f"{name}_q60"] = _weighted_quantile_seed(pred, wt, 0.6).astype(np.float32)
        out[f"{name}_trim20_80"] = _trimmed_weighted_mean(pred, wt, 0.2, 0.8).astype(np.float32)

    _, selector_variant, _, _ = rl.selector_well_code(hw)
    selected_full = rl.apply_selector_variant(selector_variant, pf_by_scale_full, tvt_beam_full, last_known)
    out["sel15_current"] = (selected_full[ev.index] - last_known).astype(np.float32)

    parts = selector_variant.split("_")
    beam_w = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_w = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    wt5 = _softmax_from_liks(liks, 5.0)
    mean5 = out["pf_scale_5"].astype(np.float32)
    std5 = np.sqrt(np.maximum((wt5[:, None] * (pred - mean5[None, :]) ** 2).sum(axis=0), 0.0))
    shrunk = mean5 / (1.0 + 0.04 * std5)
    sel = (1.0 - beam_w) * shrunk + beam_w * tvt_beam
    sel = (1.0 - hold_w) * sel + hold_w * 0.0
    out["pf_scale_5_stdshrink"] = shrunk.astype(np.float32)
    out["selector_pf_scale_5_stdshrink"] = sel.astype(np.float32)

    for base_name in (
        "pf_seed_median",
        "pf_top4_mean",
        "pf_top8_mean",
        "pf_scale_5_wmedian",
        "pf_scale_8_wmedian",
        "pf_scale_5_trim20_80",
        "pf_scale_8_trim20_80",
    ):
        base = out[base_name].astype(np.float32)
        v = (1.0 - beam_w) * base + beam_w * tvt_beam
        v = (1.0 - hold_w) * v + hold_w * 0.0
        out[f"selector_{base_name}"] = v.astype(np.float32)
    return {name: np.asarray(out[name], dtype=np.float32) for name in POSTERIOR_COLS}


def add_posterior_smoother_features(test_df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    sample_meta = sample.copy()
    sample_meta["well"] = sample_meta["id"].str[:8]
    sample_meta["row_idx"] = sample_meta["id"].str[9:].astype(int)
    rows = []
    for i, wid in enumerate(sorted(sample_meta["well"].unique()), 1):
        print(f"PF posterior {i}/{sample_meta['well'].nunique()} {wid}", flush=True)
        hw_te, tw_te = load_well(wid, "test")
        summary = run_pf_posterior_summary(hw_te, tw_te, n_particles=500, n_seeds=128)
        ws = sample_meta[sample_meta["well"] == wid]
        row_idx = ws["row_idx"].to_numpy(dtype=int)
        frame = pd.DataFrame({"id": ws["id"].to_numpy(), "well": wid, "row_idx": row_idx})
        ev_index = np.flatnonzero(hw_te["TVT_input"].isna().to_numpy())
        pos = pd.Series(np.arange(len(ev_index), dtype=np.int32), index=ev_index)
        take = pos.loc[row_idx].to_numpy(dtype=np.int32)
        for name in POSTERIOR_COLS:
            frame[name] = summary[name][take].astype(np.float32)
        rows.append(frame)
    post = pd.concat(rows, ignore_index=True)
    out = test_df.merge(post.drop(columns=["well", "row_idx"]), on="id", how="left")
    missing = [c for c in POSTERIOR_COLS if out[c].isna().any()]
    if missing:
        raise RuntimeError(f"posterior feature NaNs after merge: {missing[:5]}")
    return out


def load_sequence_models(feature_columns: list[str], alphas: np.ndarray, prefix: str, label: str):
    sys.path.insert(0, str(SMOOTHER_ROOT))
    from rogii_hyformer import ROGIIHyFormer, ROGIIHyFormerConfig

    # Kaggle's GPU image can expose CUDA devices that are unsupported by the
    # bundled torch build for GRU kernels.  This infer step is tiny, so CPU is
    # the robust path.
    requested_device = os.environ.get("ROGII_SMOOTHER_DEVICE", "cpu").strip().lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    models = []
    for fold, alpha in enumerate(alphas):
        if float(alpha) == 0.0:
            continue
        cfg = ROGIIHyFormerConfig(
            input_dim=len(feature_columns),
            d_model=128,
            num_heads=4,
            num_layers=4,
            hidden_mult=4,
            dropout=0.05,
            max_seq_len=8192,
            encoder_type="bigru",
            output_mode="delta",
            use_rope=False,
        )
        model = ROGIIHyFormer(cfg)
        state = torch.load(SMOOTHER_ROOT / f"{prefix}_f{fold}_model.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        mean = np.load(SMOOTHER_ROOT / f"{prefix}_f{fold}_mean.npy").astype(np.float32)
        std = np.load(SMOOTHER_ROOT / f"{prefix}_f{fold}_std.npy").astype(np.float32)
        std[~np.isfinite(std)] = 1.0
        std[std == 0.0] = 1.0
        models.append((fold, float(alpha), model, mean, std))
    if not models:
        raise RuntimeError(f"no active {label} folds loaded")
    print(f"loaded {label} folds={[m[0] for m in models]} device={device}", flush=True)
    return models, device


def predict_sequence_residuals(
    frame: pd.DataFrame,
    feature_columns: list[str],
    alphas: np.ndarray,
    prefix: str,
    label: str,
) -> np.ndarray:
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} feature columns missing: {missing}")
    models, device = load_sequence_models(feature_columns, alphas, prefix, label)
    pred_sum = np.zeros(len(frame), dtype=np.float32)
    x_all = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    grouped = frame.assign(_row=np.arange(len(frame), dtype=np.int32)).groupby("well", sort=False)
    with torch.no_grad():
        for fold, alpha, model, mean, std in models:
            fold_pred = np.zeros(len(frame), dtype=np.float32)
            for _, g in grouped:
                idx = g["_row"].to_numpy(dtype=np.int64)
                x = np.nan_to_num(x_all[idx], nan=0.0, posinf=0.0, neginf=0.0)
                x = ((x - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
                xb = torch.from_numpy(x[None, :, :]).to(device)
                mask = torch.zeros((1, len(idx)), dtype=torch.bool, device=device)
                values = model(xb, key_padding_mask=mask).detach().cpu().numpy()[0].astype(np.float32)
                fold_pred[idx] = values
            pred_sum += float(alpha) * fold_pred
            print(f"  {label} fold={fold} alpha={alpha:.2f} residual mean={fold_pred.mean():.4f}", flush=True)
    return pred_sum / float(len(alphas))


def predict_smoother_residuals(frame: pd.DataFrame) -> np.ndarray:
    return predict_sequence_residuals(
        frame,
        SMOOTHER_FEATURES,
        SMOOTHER_ALPHAS,
        "post_smooth",
        "posterior smoother",
    )


def predict_gate_smoother_residuals(frame: pd.DataFrame) -> np.ndarray:
    return predict_sequence_residuals(
        frame,
        GATE_SMOOTHER_FEATURES,
        GATE_SMOOTHER_ALPHAS,
        "gate_smooth",
        "posterior gate smoother",
    )


def apply_smoother_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["blend_base"] = (0.5 * df["ridge_pp_sg"].astype(np.float32) + 0.5 * df["pf_scale_5"].astype(np.float32)).astype(np.float32)
    df["pf_vs_ridge"] = (df["pf_scale_5"] - df["ridge_pp_sg"]).astype(np.float32)
    df["abs_pf_vs_ridge"] = df["pf_vs_ridge"].abs().astype(np.float32)
    scale_cols = ["pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"]
    df["pf_scale_spread"] = (df[scale_cols].max(axis=1) - df[scale_cols].min(axis=1)).astype(np.float32)
    df["pf_quant_spread5"] = (df["pf_scale_5_q60"] - df["pf_scale_5_q40"]).astype(np.float32)
    top_cols = ["pf_top1_mean", "pf_top4_mean", "pf_top8_mean", "pf_top16_mean", "pf_top32_mean"]
    df["pf_top_spread"] = (df[top_cols].max(axis=1) - df[top_cols].min(axis=1)).astype(np.float32)
    df["sel15_minus_raw5"] = (df["sel15_current"] - df["pf_scale_5"]).astype(np.float32)
    df["beam_minus_raw5"] = (df["beam_mean"] - df["pf_scale_5"]).astype(np.float32)
    return df


def load_struct_cloud():
    cloud_path = SMOOTHER_ROOT / "anc_cloud.npy"
    if not cloud_path.exists():
        raise FileNotFoundError(f"Missing structural cloud: {cloud_path}")
    cloud = np.load(cloud_path).astype(np.float64)
    return cKDTree(cloud[:, :2]), cloud


_STRUCT_CACHE = None


def get_struct_cloud():
    global _STRUCT_CACHE
    if _STRUCT_CACHE is None:
        _STRUCT_CACHE = load_struct_cloud()
    return _STRUCT_CACHE


def _local_plane_predict(tree, cloud: np.ndarray, xy: np.ndarray, k: int = 60) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return np.empty(0, dtype=np.float64)
    dist, idx = tree.query(xy, k=min(k, len(cloud)), workers=1)
    if idx.ndim == 1:
        idx = idx[:, None]
    out = np.empty(len(xy), dtype=np.float64)
    for i in range(len(xy)):
        ji = idx[i]
        X = np.column_stack([cloud[ji, 0], cloud[ji, 1], np.ones(len(ji))])
        try:
            coef, *_ = np.linalg.lstsq(X, cloud[ji, 2], rcond=None)
            out[i] = coef[0] * xy[i, 0] + coef[1] * xy[i, 1] + coef[2]
        except Exception:
            out[i] = float(np.nanmean(cloud[ji, 2]))
    return out


def estimate_struct_tvt_path(wid: str) -> np.ndarray:
    try:
        tree, cloud = get_struct_cloud()
        hw, _ = load_well(wid, "test")
        known = hw[hw["TVT_input"].notna()]
        hidden = hw[hw["TVT_input"].isna()]
        if len(known) < 5 or len(hidden) == 0:
            return np.full(len(hidden), np.nan, dtype=np.float32)
        hidden_idx = np.linspace(0, len(hidden) - 1, min(240, len(hidden))).astype(int)
        visible_idx = np.linspace(0, len(known) - 1, min(60, len(known))).astype(int)
        Sh = _local_plane_predict(tree, cloud, hidden.iloc[hidden_idx][["X", "Y"]].to_numpy(np.float64))
        Sv = _local_plane_predict(tree, cloud, known.iloc[visible_idx][["X", "Y"]].to_numpy(np.float64))
        visible_F = known.iloc[visible_idx]["TVT_input"].to_numpy(np.float64) + known.iloc[visible_idx]["Z"].to_numpy(np.float64)
        offset = float(np.nanmean(visible_F - Sv))
        s_sample = Sh + offset - hidden.iloc[hidden_idx]["Z"].to_numpy(np.float64)
        ok = np.isfinite(s_sample)
        if not np.isfinite(offset) or ok.sum() < 2:
            return np.full(len(hidden), np.nan, dtype=np.float32)
        s_tvt = np.interp(np.arange(len(hidden)), hidden_idx[ok], s_sample[ok])
        last_known = float(known["TVT_input"].iloc[-1])
        return (s_tvt - last_known).astype(np.float32)
    except Exception as exc:
        print(f"  structural path failed for {wid}: {exc}", flush=True)
        return np.empty(0, dtype=np.float32)


def _prefix_struct_cut_backtest(hw: pd.DataFrame, cut: int) -> dict | None:
    tree, cloud = get_struct_cloud()
    known = hw[hw["TVT_input"].notna()]
    ps = len(known)
    if cut < STRUCT_PREFIX_MIN_VISIBLE or ps - cut < STRUCT_PREFIX_MIN_PSEUDO:
        return None
    vsel = np.linspace(0, cut - 1, min(60, cut)).astype(int)
    hsel = np.linspace(cut, ps - 1, min(120, ps - cut)).astype(int)
    sv = _local_plane_predict(tree, cloud, hw.iloc[vsel][["X", "Y"]].to_numpy(np.float64))
    sh = _local_plane_predict(tree, cloud, hw.iloc[hsel][["X", "Y"]].to_numpy(np.float64))
    f_vis = hw.iloc[vsel]["TVT_input"].to_numpy(np.float64) + hw.iloc[vsel]["Z"].to_numpy(np.float64)
    offset = float(np.nanmean(f_vis - sv))
    if not np.isfinite(offset) or not np.isfinite(sh).any():
        return None
    pred_tvt = sh + offset - hw.iloc[hsel]["Z"].to_numpy(np.float64)
    true_tvt = hw.iloc[hsel]["TVT_input"].to_numpy(np.float64)
    lk = float(hw.iloc[cut - 1]["TVT_input"])
    pred_level = float(np.nanmean(pred_tvt - lk))
    true_level = float(np.nanmean(true_tvt - lk))
    if not np.isfinite(pred_level) or not np.isfinite(true_level):
        return None
    err = pred_tvt - true_tvt
    cf_err = lk - true_tvt
    rmse = float(np.sqrt(np.nanmean(err * err)))
    cf_rmse = float(np.sqrt(np.nanmean(cf_err * cf_err)))
    return {
        "level_abs": abs(pred_level - true_level),
        "rmse": rmse,
        "dir_ok": float(np.sign(pred_level) == np.sign(true_level)),
        "signed": pred_level - true_level,
        "gain": cf_rmse - rmse,
    }


def prefix_struct_backtest_features(wid: str) -> dict[str, float]:
    try:
        hw, _ = load_well(wid, "test")
        known = hw[hw["TVT_input"].notna()]
        ps = len(known)
        cutpoints = sorted(set(int(round(ps * f)) for f in STRUCT_PREFIX_CUT_FRACS))
        rows = [r for c in cutpoints if (r := _prefix_struct_cut_backtest(hw, c)) is not None]
        if not rows:
            return {"bt_dir_acc": np.nan, "bt_rmse_p90": np.nan, "bt_level_abs_p90": np.nan, "bt_gain_mean": np.nan}
        df = pd.DataFrame(rows)
        return {
            "bt_dir_acc": float(df["dir_ok"].mean()),
            "bt_rmse_p90": float(df["rmse"].quantile(0.90)),
            "bt_level_abs_p90": float(df["level_abs"].quantile(0.90)),
            "bt_gain_mean": float(df["gain"].mean()),
        }
    except Exception as exc:
        print(f"  prefix structural backtest failed for {wid}: {exc}", flush=True)
        return {"bt_dir_acc": np.nan, "bt_rmse_p90": np.nan, "bt_level_abs_p90": np.nan, "bt_gain_mean": np.nan}


def prefix_struct_backtest_dir_acc(wid: str) -> float:
    return float(prefix_struct_backtest_features(wid).get("bt_dir_acc", np.nan))


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.full(len(x), 1e6, dtype=np.float64)
    med = float(np.nanmedian(x[finite]))
    mad = float(np.nanmedian(np.abs(x[finite] - med)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(x[finite]) + 1e-6)
    return np.where(finite, (x - med) / scale, 1e6)


def choose_struct_posterior_mode_for_well(grp: pd.DataFrame, struct_path: np.ndarray) -> tuple[np.ndarray, str, float]:
    ridge = grp["ridge_pp_sg"].to_numpy(np.float32)
    fixed = (0.55 * ridge + 0.45 * grp["pf_scale_3"].to_numpy(np.float32)).astype(np.float32)
    base_level = float(fixed.mean())
    if len(struct_path) != len(grp) or not np.isfinite(struct_path).any():
        return fixed, "fallback:blend_w045:pf_scale_3", base_level
    struct_path = np.asarray(struct_path, dtype=np.float32)
    struct_level = float(np.nanmean(struct_path))
    struct_shape = struct_path - struct_level

    candidates = []
    for rank, cand in enumerate(STRUCT_CANDIDATE_COLS, 1):
        if cand not in grp.columns:
            continue
        raw = grp[cand].to_numpy(np.float32)
        for pred_col, pred in (
            ("pred", raw),
            ("blend_w045", (0.55 * ridge + 0.45 * raw).astype(np.float32)),
            ("blend_w050", (0.50 * ridge + 0.50 * raw).astype(np.float32)),
        ):
            diff = pred - fixed
            level = float(pred.mean())
            path_err = pred - struct_path
            pred_shape = pred - level
            shape_err = pred_shape - struct_shape
            candidates.append(
                {
                    "item": f"{pred_col}:{cand}",
                    "pred": pred,
                    "rank": float(rank),
                    "path_rmse": float(np.sqrt(np.nanmean(path_err * path_err))),
                    "shape_rmse": float(np.sqrt(np.nanmean(shape_err * shape_err))),
                    "struct_abs": abs(level - struct_level),
                    "diff_base_abs_mean": float(np.abs(diff).mean()),
                    "level": level,
                }
            )
    if not candidates:
        return fixed, "fallback:blend_w045:pf_scale_3", base_level

    path_z = _robust_zscore(np.array([c["path_rmse"] for c in candidates], dtype=np.float64))
    struct_z = _robust_zscore(np.array([c["struct_abs"] for c in candidates], dtype=np.float64))
    shape_z = _robust_zscore(np.array([c["shape_rmse"] for c in candidates], dtype=np.float64))
    base_z = _robust_zscore(np.array([c["diff_base_abs_mean"] for c in candidates], dtype=np.float64))
    rank_pen = np.array([c["rank"] / 40.0 if c["rank"] > 0 else 0.5 for c in candidates], dtype=np.float64)
    score = (
        STRUCT_MODE_RULE["lam_path"] * path_z
        + STRUCT_MODE_RULE["lam_level"] * struct_z
        + STRUCT_MODE_RULE["lam_shape"] * shape_z
        + STRUCT_MODE_RULE["lam_base"] * base_z
        + STRUCT_MODE_RULE["lam_rank"] * rank_pen
    )
    best = int(np.argmin(score))
    chosen = candidates[best]
    return chosen["pred"].astype(np.float32), str(chosen["item"]), float(chosen["level"])


def apply_struct_posterior_mode_prior(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    struct_pred = np.zeros(len(df), dtype=np.float32)
    final_pred = df["final_drift"].to_numpy(np.float32).copy()
    path_best_blend = np.zeros(len(df), dtype=np.float32)
    path_best_struct = np.zeros(len(df), dtype=np.float32)
    path_bt_dir_weight = np.zeros(len(df), dtype=np.float32)
    path_bt_dir_acc = np.full(len(df), np.nan, dtype=np.float32)
    bt_rmse_p90 = np.full(len(df), np.nan, dtype=np.float32)
    bt_level_abs_p90 = np.full(len(df), np.nan, dtype=np.float32)
    bt_gain_mean = np.full(len(df), np.nan, dtype=np.float32)
    choices = []
    for wid, grp in df.groupby("well", sort=False):
        struct_path = estimate_struct_tvt_path(str(wid))
        pred, item, level = choose_struct_posterior_mode_for_well(grp, struct_path)
        idx = grp.index.to_numpy()
        struct_pred[idx] = pred
        w = float(STRUCT_MODE_RULE["blend_weight"])
        path_blend = ((1.0 - w) * df.loc[idx, "final_drift"].to_numpy(np.float32) + w * pred).astype(np.float32)
        path_best_blend[idx] = path_blend
        path_best_struct[idx] = pred
        bt_feat = prefix_struct_backtest_features(str(wid))
        bt_dir_acc = float(bt_feat.get("bt_dir_acc", np.nan))
        active = bool(np.isfinite(bt_dir_acc) and bt_dir_acc >= STRUCT_PREFIX_GATE_DIR_ACC)
        path_bt_dir_weight[idx] = 1.0 if active else 0.0
        path_bt_dir_acc[idx] = float(bt_dir_acc) if np.isfinite(bt_dir_acc) else np.nan
        bt_rmse_p90[idx] = float(bt_feat.get("bt_rmse_p90", np.nan))
        bt_level_abs_p90[idx] = float(bt_feat.get("bt_level_abs_p90", np.nan))
        bt_gain_mean[idx] = float(bt_feat.get("bt_gain_mean", np.nan))
        if active:
            final_pred[idx] = path_blend
        s_level = float(np.nanmean(struct_path)) if len(struct_path) and np.isfinite(struct_path).any() else np.nan
        choices.append((str(wid), item, s_level, level, float(bt_dir_acc), active, len(grp)))
    df["struct_mode_prior"] = struct_pred
    df["final_drift_struct"] = final_pred.astype(np.float32)
    df["path_best_struct"] = path_best_struct.astype(np.float32)
    df["path_best_blend"] = path_best_blend.astype(np.float32)
    df["path_bt_dir_weight"] = path_bt_dir_weight.astype(np.float32)
    df["bt_dir_acc"] = path_bt_dir_acc.astype(np.float32)
    df["bt_rmse_p90"] = bt_rmse_p90.astype(np.float32)
    df["bt_level_abs_p90"] = bt_level_abs_p90.astype(np.float32)
    df["bt_gain_mean"] = bt_gain_mean.astype(np.float32)
    df["path_bt_dir_base"] = (
        (1.0 - df["path_bt_dir_weight"].to_numpy(np.float32)) * df["final_drift"].to_numpy(np.float32)
        + df["path_bt_dir_weight"].to_numpy(np.float32) * df["path_best_blend"].to_numpy(np.float32)
    ).astype(np.float32)
    print("path-struct mode choices:", flush=True)
    for wid, item, s_level, level, bt_dir_acc, active, n in choices:
        print(
            f"  {wid}: {item} path_level={s_level:.4f} pred_level={level:.4f} "
            f"bt_dir_acc={bt_dir_acc:.3f} active={active} n={n}",
            flush=True,
        )
    return df


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(float)
    br = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ar) < 1e-9 or np.std(br) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _pseudo_hw_from_prefix(hw: pd.DataFrame, cut: int) -> pd.DataFrame:
    known = hw[hw["TVT_input"].notna()]
    tvt_input = known["TVT_input"].to_numpy(float).copy()
    tvt_input[cut:] = np.nan
    return pd.DataFrame(
        {
            "MD": known["MD"].to_numpy(float),
            "X": known["X"].to_numpy(float),
            "Y": known["Y"].to_numpy(float),
            "Z": known["Z"].to_numpy(float),
            "GR": known["GR"].to_numpy(float),
            "TVT_input": tvt_input,
        }
    )


def _candidate_paths_from_replay(
    pred: np.ndarray,
    liks: np.ndarray,
    pseudo_slice: slice,
    last_known: float,
) -> dict[str, np.ndarray]:
    hidden_pred = pred[:, pseudo_slice] - float(last_known)
    out = {
        "seed_mean": hidden_pred.mean(axis=0),
        "seed_median": np.median(hidden_pred, axis=0),
    }
    order = np.argsort(liks)[::-1]
    for k in (1, 4, 8, 16):
        kk = min(k, len(order))
        out[f"top{kk}_mean"] = hidden_pred[order[:kk]].mean(axis=0)
    for scale in rl.SELECTOR_SCALES:
        wt = _softmax_from_liks(liks, scale)
        out[f"scale{scale:g}"] = (wt[:, None] * hidden_pred).sum(axis=0)
    return out


def _one_pf_prefix_cut_replay(
    wid: str,
    hw: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cut: int,
) -> dict | None:
    known = hw[hw["TVT_input"].notna()]
    ps = len(known)
    if cut < STRUCT_PREFIX_MIN_VISIBLE or ps - cut < STRUCT_PREFIX_MIN_PSEUDO:
        return None
    pseudo_hw = _pseudo_hw_from_prefix(hw, cut)
    preds = []
    liks = []
    seed_key = f"{wid}:{int(cut)}:{PF_PREFIX_SEED}".encode()
    seed_base = (zlib.crc32(seed_key) % 1_000_000) + PF_PREFIX_SEED
    for s in range(PF_PREFIX_N_SEEDS):
        pred_full, ll = rl.run_particle_filter(
            pseudo_hw,
            pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr}),
            n_particles=PF_PREFIX_N_PARTICLES,
            seed=seed_base + s,
        )
        preds.append(pred_full.astype(np.float32))
        liks.append(float(ll))
    pred = np.stack(preds, axis=0)
    liks = np.asarray(liks, dtype=np.float64)

    pseudo = slice(cut, ps)
    last_known = float(known["TVT_input"].iloc[cut - 1])
    target = known["TVT_input"].iloc[cut:ps].to_numpy(np.float32) - last_known
    candidates = _candidate_paths_from_replay(pred, liks, pseudo, last_known)
    cand_rows = []
    for name, values in candidates.items():
        err = values.astype(np.float64) - target.astype(np.float64)
        level = float(np.mean(values))
        true_level = float(np.mean(target))
        cand_rows.append(
            {
                "candidate": name,
                "rmse": float(np.sqrt(np.mean(err * err))),
                "level_abs": float(abs(level - true_level)),
                "signed_level_err": float(level - true_level),
                "dir_ok": float(np.sign(level) == np.sign(true_level)),
            }
        )
    cand = pd.DataFrame(cand_rows)
    best_rmse = cand.sort_values("rmse", kind="mergesort").iloc[0]
    best_level = cand.sort_values("level_abs", kind="mergesort").iloc[0]

    seed_hidden = pred[:, pseudo] - last_known
    seed_err = seed_hidden.astype(np.float64) - target.astype(np.float64)[None, :]
    seed_rmse = np.sqrt(np.mean(seed_err * seed_err, axis=1))
    seed_level_abs = np.abs(seed_hidden.mean(axis=1) - float(target.mean()))
    top1 = cand[cand["candidate"] == "top1_mean"].iloc[0]
    scale5 = cand[cand["candidate"] == "scale5"].iloc[0]
    spread_levels = np.array([cand.loc[cand["candidate"] == c, "signed_level_err"].iloc[0] for c in cand["candidate"]])
    cf_rmse = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    return {
        "scale5_rmse": float(scale5["rmse"]),
        "scale5_level_abs": float(scale5["level_abs"]),
        "scale5_dir_ok": float(scale5["dir_ok"]),
        "scale5_gain_vs_cf": float(cf_rmse - scale5["rmse"]),
        "top1_rmse": float(top1["rmse"]),
        "top1_level_abs": float(top1["level_abs"]),
        "top1_dir_ok": float(top1["dir_ok"]),
        "top1_gain_vs_cf": float(cf_rmse - top1["rmse"]),
        "oracle_rmse": float(best_rmse["rmse"]),
        "oracle_level_abs": float(best_level["level_abs"]),
        "scale5_regret_rmse": float(scale5["rmse"] - best_rmse["rmse"]),
        "top1_regret_rmse": float(top1["rmse"] - best_rmse["rmse"]),
        "scale5_regret_level": float(scale5["level_abs"] - best_level["level_abs"]),
        "top1_regret_level": float(top1["level_abs"] - best_level["level_abs"]),
        "lik_rmse_spearman": _rank_corr(liks, -seed_rmse),
        "lik_level_spearman": _rank_corr(liks, -seed_level_abs),
        "seed_level_std": float(np.std(seed_hidden.mean(axis=1))),
        "seed_rmse_std": float(np.std(seed_rmse)),
        "mode_signed_level_std": float(np.std(spread_levels)),
        "mode_signed_level_range": float(np.max(spread_levels) - np.min(spread_levels)),
        "best_rmse_candidate": str(best_rmse["candidate"]),
        "best_level_candidate": str(best_level["candidate"]),
    }


def pf_prefix_replay_features_for_well(wid: str) -> dict:
    hw, tw = load_well(wid, "test")
    known = hw[hw["TVT_input"].notna()]
    base = {
        "n_cuts": 0,
        "pfbt_scale5_level_abs_mean": np.nan,
        "pfbt_top1_rmse_mean": np.nan,
        "pfbt_scale5_level_abs_p90": np.nan,
        "pfbt_scale5_rmse_p90": np.nan,
        "pfbt_scale5_dir_acc": np.nan,
        "pfbt_top1_dir_acc": np.nan,
        "pfbt_seed_level_std_mean": np.nan,
        "pfbt_mode_signed_level_range_mean": np.nan,
        "pfbt_lik_rmse_spearman_p90": np.nan,
    }
    if len(known) < STRUCT_PREFIX_MIN_VISIBLE + STRUCT_PREFIX_MIN_PSEUDO:
        return base
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(np.float64)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(np.float64)
    ps = len(known)
    cutpoints = sorted(set(int(round(ps * f)) for f in STRUCT_PREFIX_CUT_FRACS))
    rows = [r for c in cutpoints if (r := _one_pf_prefix_cut_replay(wid, hw, tw_tvt, tw_gr, c)) is not None]
    if not rows:
        return base
    df = pd.DataFrame(rows)
    base["n_cuts"] = int(len(rows))
    for src, out_mean, out_p90 in [
        ("scale5_level_abs", "pfbt_scale5_level_abs_mean", "pfbt_scale5_level_abs_p90"),
        ("top1_rmse", "pfbt_top1_rmse_mean", None),
        ("scale5_rmse", None, "pfbt_scale5_rmse_p90"),
        ("seed_level_std", "pfbt_seed_level_std_mean", None),
        ("mode_signed_level_range", "pfbt_mode_signed_level_range_mean", None),
        ("lik_rmse_spearman", None, "pfbt_lik_rmse_spearman_p90"),
    ]:
        if out_mean:
            base[out_mean] = float(df[src].mean())
        if out_p90:
            base[out_p90] = float(df[src].quantile(0.90))
    base["pfbt_scale5_dir_acc"] = float(df["scale5_dir_ok"].mean())
    base["pfbt_top1_dir_acc"] = float(df["top1_dir_ok"].mean())
    return base


def add_gate_smoother_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["posterior"] = df["final_drift"].astype(np.float32)
    df["path_minus_posterior"] = (df["path_best_blend"] - df["posterior"]).astype(np.float32)
    df["path_struct_minus_posterior"] = (df["path_best_struct"] - df["posterior"]).astype(np.float32)
    df["abs_path_minus_posterior"] = df["path_minus_posterior"].abs().astype(np.float32)
    df["pfreplay_gate_weight"] = 0.0
    df["pfreplay_feature"] = np.nan
    for wid, idx in df.groupby("well", sort=False).indices.items():
        print(f"PF prefix replay gate {wid}", flush=True)
        feat = pf_prefix_replay_features_for_well(str(wid))
        bt_dir_ok = float(np.nanmean(df.loc[idx, "bt_dir_acc"].to_numpy(float))) >= STRUCT_PREFIX_GATE_DIR_ACC
        value = float(feat.get("pfbt_scale5_level_abs_mean", np.nan))
        active = bool(bt_dir_ok and np.isfinite(value) and value <= PF_PREFIX_GATE_THRESHOLD)
        df.loc[idx, "pfreplay_gate_weight"] = 1.0 if active else 0.0
        df.loc[idx, "pfreplay_feature"] = value if np.isfinite(value) else np.nan
        for col, val in feat.items():
            if col == "n_cuts":
                continue
            df.loc[idx, col] = val
        print(
            f"  {wid}: bt_dir_ok={bt_dir_ok} pfbt_level_abs_mean={value:.4f} "
            f"active={active}",
            flush=True,
        )
    df["pfreplay_gate_weight"] = df["pfreplay_gate_weight"].astype(np.float32)
    df["pfreplay_gate_base"] = (
        (1.0 - df["pfreplay_gate_weight"].to_numpy(np.float32)) * df["posterior"].to_numpy(np.float32)
        + df["pfreplay_gate_weight"].to_numpy(np.float32) * df["path_best_blend"].to_numpy(np.float32)
    ).astype(np.float32)
    df["gate_disagrees_bt"] = (
        df["path_bt_dir_weight"].fillna(0).astype(np.float32)
        - df["pfreplay_gate_weight"].fillna(0).astype(np.float32)
    ).abs().astype(np.float32)
    for col in ["bt_rmse_p90", "bt_level_abs_p90", "bt_gain_mean", "pred_help_prob",
                "pred_utility", "xy_kdist_p90", "pf_ridge_gap_proxy"]:
        if col not in df.columns:
            df[col] = 0.0
    df["pred_help_prob"] = df["pred_help_prob"].fillna(0.0).astype(np.float32)
    df["pred_utility"] = df["pred_utility"].fillna(0.0).astype(np.float32)
    return df


def load_well(wid: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    hw = pd.read_csv(DATA_ROOT / split / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_ROOT / split / f"{wid}__typewell.csv")
    return hw, tw


def motion_robust_rate(tvt: np.ndarray, z: np.ndarray, md: np.ndarray) -> float | None:
    dt = np.diff(tvt.astype(float))
    dz = np.diff(z.astype(float))
    dm = np.diff(md.astype(float))
    ok = np.isfinite(dt) & np.isfinite(dz) & np.isfinite(dm) & (dm > 0)
    if ok.sum() < 3:
        return None
    vals = (dt[ok] + dz[ok]) / dm[ok]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return None
    lo, hi = np.nanquantile(vals, [0.05, 0.95])
    vals = vals[(vals >= lo) & (vals <= hi)]
    if len(vals) == 0:
        return None
    return float(np.nanmedian(vals))


def motion_rate_bank(hw: pd.DataFrame) -> np.ndarray:
    known = hw[hw["TVT_input"].notna()]
    hs = int(len(known))
    tvt = hw["TVT_input"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    md = hw["MD"].to_numpy(float)
    windows = [30, 60, 120, 300, 800, hs]
    rates = []
    for win in windows:
        start = max(0, hs - int(win))
        r = motion_robust_rate(tvt[start:hs], z[start:hs], md[start:hs])
        if r is not None:
            rates.append(r)
    if not rates:
        rates = [0.0]
    bank = []
    for r in rates[:4]:
        for off in MOTION_RATE_OFFSETS:
            bank.append(r + off)
    bank = [float(np.clip(r, -0.05, 0.05)) for r in bank if np.isfinite(r)]
    if not bank:
        bank = [float(np.clip(rates[0], -0.05, 0.05))]
    out = []
    for r in bank:
        if not any(abs(r - q) < 1e-6 for q in out):
            out.append(r)
    return np.asarray(out, dtype=np.float64)


def run_motion_particle_filter(hw: pd.DataFrame, tw: pd.DataFrame, rate_mu: float, n_particles: int, seed: int) -> tuple[np.ndarray, float]:
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    known = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].to_numpy(float).copy(), 0.0
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])
    tw_at_k = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(np.nan_to_num(known["GR"].to_numpy(float)) - tw_at_k), 10.0, 60.0))

    rng = np.random.default_rng(int(seed))
    N = int(n_particles)
    pos = last_tvt + last_z + 2.0 * rng.standard_normal(N)
    rate = float(rate_mu) + 0.01 * rng.standard_normal(N)
    weights = np.ones(N, dtype=np.float64) / N
    out_vals = hw["TVT_input"].to_numpy(float).copy()
    gr_interp = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    prev_md = last_md
    log_lik = 0.0

    for ridx in ev.index:
        md_i = float(hw.at[ridx, "MD"])
        z_i = float(hw.at[ridx, "Z"])
        gr_i = float(gr_interp[ridx])
        dm_step = max(md_i - prev_md, 1.0)
        noise = MOTION_VN * rng.standard_normal(N)
        rate_n = MOTION_MOM * rate + (1.0 - MOTION_MOM) * float(rate_mu) + noise
        pos_n = pos + rate_n * dm_step + MOTION_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos_n - z_i, tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        pos_n = tvt_p + z_i
        expected_gr = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_i - expected_gr) / gs
        lk = np.exp(-0.5 * np.minimum(d * d, 80.0))
        lk = np.maximum(lk, 1e-30)
        avg_lk = float((weights * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-30))
        weights_n = weights * lk
        ws = float(weights_n.sum())
        weights_n = weights_n / ws if ws > 0 else np.ones(N, dtype=np.float64) / N
        n_eff = 1.0 / float((weights_n * weights_n).sum())
        if n_eff < MOTION_RESAMP * N:
            cum = np.cumsum(weights_n)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos_n[idx] + MOTION_RP * rng.standard_normal(N)
            rate = rate_n[idx] + MOTION_RR * rng.standard_normal(N)
            weights = np.ones(N, dtype=np.float64) / N
        else:
            pos = pos_n
            rate = rate_n
            weights = weights_n
        out_vals[ridx] = float(np.dot(weights, pos - z_i))
        prev_md = md_i
    return out_vals, float(log_lik)


def run_motion_pf_by_scale(hw: pd.DataFrame, tw: pd.DataFrame, n_particles: int = MOTION_N_PARTICLES, n_seeds: int = MOTION_N_SEEDS) -> dict[str, np.ndarray]:
    rates = motion_rate_bank(hw)
    preds = []
    liks = []
    well_seed_base = zlib.crc32(str(hw.attrs.get("well_id", "")).encode("utf-8")) & 0xFFFFFFFF
    for s in range(int(n_seeds)):
        rate_mu = float(rates[s % len(rates)])
        pred, ll = run_motion_particle_filter(
            hw,
            tw,
            rate_mu,
            n_particles=int(n_particles),
            seed=int((MOTION_SEED + 1009 * s + well_seed_base) % (2**32 - 1)),
        )
        preds.append(pred)
        liks.append(ll)
    pred_arr = np.stack(preds, axis=0)
    liks = np.asarray(liks, dtype=np.float64)
    liks_n = liks - float(np.nanmax(liks))
    out = {}
    for scale in MOTION_SCALES:
        weights = np.exp(np.clip(liks_n / float(scale), -700.0, 0.0))
        sw = float(weights.sum())
        weights = weights / sw if sw > 0 else np.ones_like(weights) / len(weights)
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(axis=0)
    out["pf_mean"] = pred_arr.mean(axis=0)
    return out


def motion_selector_for_well(wid: str) -> pd.DataFrame:
    print(f"Motion PF selector {wid}", flush=True)
    hw, tw = load_well(wid, "test")
    hw = hw.copy()
    hw.attrs["well_id"] = wid
    known = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return pd.DataFrame(columns=["id", "moff_selector"])
    last_known = float(known.iloc[-1]["TVT_input"])
    try:
        pf_by_scale = run_motion_pf_by_scale(hw, tw, n_particles=MOTION_N_PARTICLES, n_seeds=MOTION_N_SEEDS)
        tvt_beam = rl.run_beam_ensemble(hw, tw)
        _, variant, n_eval, z_span = rl.selector_well_code(hw)
        tvt_sel = rl.apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known)
        print(
            f"  motion rates={len(motion_rate_bank(hw))} selector={variant} "
            f"n_eval={n_eval:.0f} z_span={z_span:.3f}",
            flush=True,
        )
    except Exception as exc:
        print(f"  motion PF failed, fallback gate_base: {exc}", flush=True)
        tvt_sel = hw["TVT_input"].fillna(last_known).to_numpy(float)
    rows = []
    for ridx in ev.index:
        rows.append({"id": f"{wid}_{int(ridx)}", "moff_selector": float(tvt_sel[ridx] - last_known)})
    return pd.DataFrame(rows)


def add_motion_selector_feature(test_df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    sample_meta = sample.copy()
    sample_meta["well"] = sample_meta["id"].str[:8]
    rows = []
    for wid in sorted(sample_meta["well"].unique()):
        rows.append(motion_selector_for_well(str(wid)))
    motion = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["id", "moff_selector"])
    out = test_df.merge(motion, on="id", how="left")
    if out["moff_selector"].isna().any():
        n = int(out["moff_selector"].isna().sum())
        print(f"  motion selector missing rows={n}; filling from gate_base later", flush=True)
    out["moff_selector"] = out["moff_selector"].fillna(out.get("pfreplay_gate_base", 0.0)).astype(np.float32)
    return out


def apply_motion_ridge_stack(frame: pd.DataFrame) -> np.ndarray:
    required = ["final_drift_gate", "pfreplay_gate_base", "blend_base", "ridge_pp_sg", "pf_scale_5", "moff_selector"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"motion ridge missing columns: {missing}")
    pred = np.full(len(frame), MOTION_RIDGE_INTERCEPT, dtype=np.float32)
    pred += MOTION_RIDGE_WEIGHTS["gate_bigru"] * frame["final_drift_gate"].to_numpy(np.float32)
    pred += MOTION_RIDGE_WEIGHTS["gate_base"] * frame["pfreplay_gate_base"].to_numpy(np.float32)
    pred += MOTION_RIDGE_WEIGHTS["blend_base"] * frame["blend_base"].to_numpy(np.float32)
    pred += MOTION_RIDGE_WEIGHTS["ridge_pp_sg"] * frame["ridge_pp_sg"].to_numpy(np.float32)
    pred += MOTION_RIDGE_WEIGHTS["pf_scale5"] * frame["pf_scale_5"].to_numpy(np.float32)
    pred += MOTION_RIDGE_WEIGHTS["moff_selector"] * frame["moff_selector"].to_numpy(np.float32)
    return pred.astype(np.float32)


def _orthonormalize_hinge(cols):
    q_cols = []
    for col in cols:
        v = np.asarray(col, dtype=np.float64)
        v = v - float(np.mean(v))
        for q in q_cols:
            v = v - q * float(np.mean(v * q))
        scale = float(np.sqrt(np.mean(v * v)))
        if scale > 1e-9:
            q_cols.append(v / scale)
    return np.column_stack(q_cols).astype(np.float32) if q_cols else np.zeros((len(cols[0]), 0), dtype=np.float32)


def _poly5_hinge_basis(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    z = x - float(np.mean(x))
    cols = [z ** p for p in range(1, 6)]
    for knot in [0.20, 0.35, 0.50, 0.65, 0.80]:
        cols.append(np.maximum(0.0, x - knot))
        cols.append(np.maximum(0.0, knot - x))
    return _orthonormalize_hinge(cols)


def _apply_level_shape_hinge_for_group(grp: pd.DataFrame, meta: dict) -> np.ndarray:
    names = list(meta["base_names"])
    degree = int(meta["degree"])
    if "md_since" in grp.columns:
        denom = max(float(np.nanmax(grp["md_since"].to_numpy(np.float64))), 1.0)
        x = np.clip(grp["md_since"].to_numpy(np.float64) / denom, 0.0, 1.0)
    elif "frac" in grp.columns:
        x = grp["frac"].to_numpy(np.float64)
    else:
        x = np.linspace(0.0, 1.0, len(grp))
    q = _poly5_hinge_basis(x)
    if q.shape[1] < degree:
        pad = np.zeros((len(grp), degree - q.shape[1]), dtype=np.float32)
        q = np.column_stack([q, pad])
    q = q[:, :degree].astype(np.float64)

    X = grp[names].to_numpy(np.float64)
    row_model = meta["row_model"]
    row_pred = X @ np.asarray(row_model["coef"], dtype=np.float64) + float(row_model["intercept"])
    basis_row_model = meta.get("basis_row_model", row_model)
    basis_row_pred = X @ np.asarray(basis_row_model["coef"], dtype=np.float64) + float(basis_row_model["intercept"])

    levels = X.mean(axis=0)
    shapes = X - levels[None, :]
    pred_fit = np.zeros_like(X, dtype=np.float64)
    coeff_by_name = {name: [] for name in names}
    for j in range(degree):
        coeffs = np.mean(shapes * q[:, j:j + 1], axis=0)
        pred_fit += coeffs[None, :] * q[:, j:j + 1]
        for i, name in enumerate(names):
            coeff_by_name[name].append(float(coeffs[i]))
    resid = shapes - pred_fit

    basis_shape = np.full(len(grp), float(meta["resid_model"]["intercept"]), dtype=np.float64)
    basis_shape += resid @ np.asarray(meta["resid_model"]["coef"], dtype=np.float64)
    for j in range(1, degree + 1):
        model = meta["basis_models"][f"b{j}"]
        coef = np.asarray(model["coef"], dtype=np.float64)
        feats = np.asarray([coeff_by_name[name][j - 1] for name in names], dtype=np.float64)
        bj = float(feats @ coef + float(model["intercept"]))
        basis_shape += bj * q[:, j - 1]

    basis_row_level = float(np.mean(basis_row_pred))
    basis_pred = basis_shape - float(np.mean(basis_shape)) + basis_row_level
    raw_delta = basis_pred - row_pred
    ramp = np.clip(x / float(meta["fade_frac"]), 0.0, 1.0)
    delta = raw_delta * ramp
    delta = delta - float(np.mean(delta))
    return (row_pred + float(meta["weight"]) * delta).astype(np.float32)


def apply_level_shape_hinge_stack(frame: pd.DataFrame) -> np.ndarray:
    meta_path = HINGE_ROOT / "level_shape_hinge_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing level-shape hinge asset: {meta_path}")
    meta = json.loads(meta_path.read_text())
    required = list(meta["base_names"]) + ["well", "id", "last_known_tvt"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"level-shape hinge missing columns: {missing}")
    out = np.zeros(len(frame), dtype=np.float32)
    for _, grp in frame.groupby("well", sort=False):
        vals = _apply_level_shape_hinge_for_group(grp, meta)
        out[grp.index.to_numpy()] = vals
    return out


def sub2_pf_selector(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    sample["well"] = sample["id"].str[:8]
    sample["row_idx"] = sample["id"].str[9:].astype(int)
    train_dir = DATA_ROOT / "train"
    train_wells = {p.name.split("__", 1)[0] for p in train_dir.glob("*__horizontal_well.csv")} if train_dir.exists() else set()
    test_wells = sorted(sample["well"].unique())

    rows = []
    for i, wid in enumerate(test_wells, 1):
        print(f"PF sub2 {i}/{len(test_wells)} {wid}", flush=True)
        hw_te, tw_te = load_well(wid, "test")
        tvt_phys = None
        hw_tr = None
        tw_tr = None
        if wid in train_wells:
            try:
                hw_tr = pd.read_csv(DATA_ROOT / "train" / f"{wid}__horizontal_well.csv")
                tw_tr = pd.read_csv(DATA_ROOT / "train" / f"{wid}__typewell.csv")
                hw_te["TVT_input"] = hw_tr["TVT_input"].to_numpy()
                tvt_phys = rl.tvt_from_contacts(hw_tr, tw_tr)
            except Exception as exc:
                print(f"  visible-well physical shortcut failed: {exc}", flush=True)
                tvt_phys = None

        selector_code, selector_variant, selector_n_eval, selector_z_span = rl.selector_well_code(hw_te)
        tw_ref = tw_tr if tw_tr is not None else tw_te
        try:
            pf_by_scale = rl.run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=128)
            tvt_pf = pf_by_scale["pf_scale_8"]
        except Exception as exc:
            print(f"  PF failed: {exc}", flush=True)
            last_known = hw_te["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf = hw_te["TVT_input"].fillna(last_val).to_numpy(float)
            pf_by_scale = {f"pf_scale_{scale:g}": tvt_pf.copy() for scale in rl.SELECTOR_SCALES}
        try:
            tvt_beam = rl.run_beam_ensemble(hw_te, tw_ref)
        except Exception as exc:
            print(f"  Beam failed: {exc}", flush=True)
            tvt_beam = tvt_pf.copy()

        last_known = hw_te["TVT_input"].dropna()
        last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
        tvt_selector = pf_by_scale.get("pf_scale_5", tvt_pf)
        print(
            f"  selector code={selector_code} variant=raw_pf_scale_5 "
            f"n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f}",
            flush=True,
        )

        ws = sample[sample["well"] == wid]
        for _, row in ws.iterrows():
            ridx = int(row["row_idx"])
            tvt_val = float(tvt_phys.iloc[ridx]) if tvt_phys is not None else float(tvt_selector[ridx])
            rows.append({"id": row["id"], "tvt_sub2": tvt_val})
    return pd.DataFrame(rows)


def main() -> None:
    print(f"DATA_ROOT={DATA_ROOT}", flush=True)
    print(f"ART_ROOT={ART_ROOT}", flush=True)
    print(f"SMOOTHER_ROOT={SMOOTHER_ROOT}", flush=True)
    print(f"HINGE_ROOT={HINGE_ROOT}", flush=True)
    if not (ART_ROOT / "models").exists():
        raise FileNotFoundError(f"Missing artifact models directory: {ART_ROOT / 'models'}")
    if not (SMOOTHER_ROOT / "rogii_hyformer.py").exists():
        raise FileNotFoundError(f"Missing smoother assets under {SMOOTHER_ROOT}")
    if not (HINGE_ROOT / "level_shape_hinge_meta.json").exists():
        raise FileNotFoundError(f"Missing hinge meta under {HINGE_ROOT}")
    install_unpickle_shims()
    rl.CFG.dataset_path = DATA_ROOT
    rl.CFG.artifacts_path = ART_ROOT
    rl.NCPU = int(os.environ.get("RAVI_WORKERS", "4"))
    ls.DATA_ROOT = DATA_ROOT
    ls.ART_ROOT = ART_ROOT
    ls.CFG.dataset_path = DATA_ROOT
    ls.CFG.artifacts_path = ART_ROOT

    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    test_paths = sorted((DATA_ROOT / "test").glob("*__horizontal_well.csv"))
    max_test_wells = int(os.environ.get("ROGII_MAX_TEST_WELLS", "0") or "0")
    if max_test_wells > 0:
        keep_wells = [p.name.split("__", 1)[0] for p in test_paths[:max_test_wells]]
        test_paths = test_paths[:max_test_wells]
        sample = sample[sample["id"].str[:8].isin(keep_wells)].reset_index(drop=True)
        print(f"SMOKE MODE ROGII_MAX_TEST_WELLS={max_test_wells} rows={len(sample)} wells={keep_wells}", flush=True)
    test_df = rl.build_dataset(test_paths, is_train=False, label="test")
    features = [c for c in test_df.columns if c not in {"well", "id", "target"}]
    X_test = test_df[features]
    print(f"test_df={test_df.shape} features={len(features)}", flush=True)

    base_preds = []
    for name in MODEL_DIRS:
        print(f"Loading/predicting {name}", flush=True)
        tr = load_trainer(name)
        base_preds.append(np.asarray(tr.predict(X_test), dtype=np.float32))
    base_pred = np.stack(base_preds, axis=1)
    ridge_drift = ridge_fold_ensemble(base_pred)
    pf_test = test_df["pf_ancc"].to_numpy(np.float32) - test_df["last_known_tvt"].to_numpy(np.float32)
    ridge_pp = apply_pp(test_df, ridge_drift, pf_test)
    test_df = test_df.copy()
    test_df["pred"] = test_df["last_known_tvt"].to_numpy(np.float32) + ridge_pp
    test_df = sg_smooth(test_df, "pred")
    test_df["ridge_pp_sg"] = (test_df["pred"].to_numpy(np.float32) - test_df["last_known_tvt"].to_numpy(np.float32)).astype(np.float32)
    test_df = add_posterior_smoother_features(test_df, sample)
    test_df = apply_smoother_feature_engineering(test_df)
    smoother_resid = predict_smoother_residuals(test_df)
    test_df["final_drift"] = (test_df["blend_base"].to_numpy(np.float32) + smoother_resid).astype(np.float32)
    test_df = apply_struct_posterior_mode_prior(test_df)
    test_df = add_gate_smoother_features(test_df)
    missing_gate = [c for c in GATE_SMOOTHER_FEATURES if c not in test_df.columns]
    if missing_gate:
        raise RuntimeError(f"gate smoother features missing before predict: {missing_gate}")
    gate_resid = predict_gate_smoother_residuals(test_df)
    test_df["final_drift_gate"] = (
        test_df["pfreplay_gate_base"].to_numpy(np.float32) + gate_resid
    ).astype(np.float32)
    test_df = add_motion_selector_feature(test_df, sample)
    test_df["final_drift_motion_ridge"] = apply_motion_ridge_stack(test_df)
    test_df["gate_bigru"] = test_df["final_drift_gate"].astype(np.float32)
    test_df["gate_base"] = test_df["pfreplay_gate_base"].astype(np.float32)
    test_df["pf_scale5"] = test_df["pf_scale_5"].astype(np.float32)
    test_df["final_drift_level_shape_hinge"] = apply_level_shape_hinge_stack(test_df)

    source_pf = ls.sub2_level_consensus_sources(sample)
    source = sample[["id"]].merge(source_pf, on="id", how="left")
    source = source.merge(
        test_df[["id", "well", "z", "md_since", "pred", "last_known_tvt", "final_drift_level_shape_hinge"]],
        on="id",
        how="left",
    )
    fallback = float(test_df["pred"].mean()) if len(test_df) else 0.0
    source["pred"] = source["pred"].fillna(fallback)
    for col in ["tvt_p0_source", "tvt_rate075_source", "tvt_starttight_source"]:
        source[col] = source[col].fillna(source["pred"])
    if source[["well", "z", "md_since", "last_known_tvt", "final_drift_level_shape_hinge"]].isna().any().any():
        raise RuntimeError("Missing columns for level-consensus hinge-shape blend")

    source["tvt"] = ls.FINAL_W_RIDGE * source["pred"] + ls.FINAL_W_PF * source["tvt_p0_source"]
    p0_work = ls.apply_chord_shrink(ls.apply_f_rectification(source, ls.FRECT_FRAC), ls.CHORD_W)
    source["tvt_p0_chord"] = p0_work["tvt"].to_numpy(np.float32)
    source["tvt_rate075"] = (
        ls.FINAL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + ls.FINAL_W_PF * source["tvt_rate075_source"].to_numpy(np.float32)
    ).astype(np.float32)
    source["tvt_starttight"] = (
        ls.FINAL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + ls.FINAL_W_PF * source["tvt_starttight_source"].to_numpy(np.float32)
    ).astype(np.float32)
    source["tvt_rawpf_source"] = (
        source["tvt_p0_source"].to_numpy(np.float32)
        + source["tvt_rate075_source"].to_numpy(np.float32)
        + source["tvt_starttight_source"].to_numpy(np.float32)
    ) / 3.0
    source["tvt"] = (
        RAWPF_LEVEL_W_RIDGE * source["pred"].to_numpy(np.float32)
        + RAWPF_LEVEL_W_PF * source["tvt_rawpf_source"].to_numpy(np.float32)
    ).astype(np.float32)
    rawpf_work = ls.apply_chord_shrink(ls.apply_f_rectification(source, ls.FRECT_FRAC), ls.CHORD_W)
    source["tvt_rawpf_mean_chord"] = rawpf_work["tvt"].to_numpy(np.float32)
    source["tvt_hinge"] = (
        source["last_known_tvt"].to_numpy(np.float32)
        + source["final_drift_level_shape_hinge"].to_numpy(np.float32)
    ).astype(np.float32)

    level = source.groupby("well", sort=False)["tvt_rawpf_mean_chord"].transform("mean").to_numpy(np.float64)
    mixed_shape_raw = 0.5 * source["tvt_p0_chord"].to_numpy(np.float64) + 0.5 * source["tvt_hinge"].to_numpy(np.float64)
    source["_mixed_shape_raw"] = mixed_shape_raw
    shape = mixed_shape_raw - source.groupby("well", sort=False)["_mixed_shape_raw"].transform("mean").to_numpy(np.float64)
    source["tvt"] = (level + shape).astype(np.float32)
    softgr2_delta = compute_softgr2_level_delta(source)
    source["softgr2_delta"] = softgr2_delta.astype(np.float32)
    source["tvt"] = (level + softgr2_delta.astype(np.float64) + shape).astype(np.float32)
    source["tvt_vp_base"] = source["tvt"].astype(np.float32)
    source["tvt"] = apply_segmented_geology_posterior_overlay(source)
    source["vp_delta"] = (source["tvt"].to_numpy(np.float32) - source["tvt_vp_base"].to_numpy(np.float32)).astype(np.float32)

    sub = sample[["id"]].merge(source[["id", "tvt"]], on="id", how="left")
    sub["tvt"] = sub["tvt"].fillna(fallback)
    out = sub[["id", "tvt"]]
    if len(out) != len(sample):
        raise RuntimeError(f"submission row mismatch {len(out)} != {len(sample)}")
    if out["tvt"].isna().any():
        raise RuntimeError("NaN in submission")
    print(
        "level-consensus hinge-shape summary "
        f"blend_mean={test_df['blend_base'].mean():.4f} "
        f"resid_mean={float(np.mean(smoother_resid)):.4f} "
        f"final_mean={test_df['final_drift'].mean():.4f} "
        f"pfreplay_base_mean={test_df['pfreplay_gate_base'].mean():.4f} "
        f"gate_resid_mean={float(np.mean(gate_resid)):.4f} "
        f"gate_final_mean={test_df['final_drift_gate'].mean():.4f} "
        f"moff_selector_mean={test_df['moff_selector'].mean():.4f} "
        f"motion_ridge_mean={test_df['final_drift_motion_ridge'].mean():.4f} "
        f"level_shape_hinge_mean={test_df['final_drift_level_shape_hinge'].mean():.4f} "
        f"p0_mean={source['tvt_p0_chord'].mean():.4f} rate075_mean={source['tvt_rate075'].mean():.4f} "
        f"starttight_mean={source['tvt_starttight'].mean():.4f} "
        f"fixed_rawpf_mean_level={source['tvt_rawpf_mean_chord'].mean():.4f} "
        f"rawpf_level_w={RAWPF_LEVEL_W_PF:.2f} "
        f"softgr2_nonzero_wells={(source.groupby('well')['softgr2_delta'].mean().abs() > 1e-9).sum()} "
        f"softgr2_mean={source['softgr2_delta'].mean():.4f} "
        f"vp_abs_delta_mean={source['vp_delta'].abs().mean():.4f} "
        f"final_mean={out['tvt'].mean():.4f}",
        flush=True,
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} rows={len(out)}", flush=True)
    print(out.head().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
