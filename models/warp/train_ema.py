"""
exp207 — 純GR WARP(6段, exp204)の上乗せ探索 Step2: EMA(重み指数移動平均)。

exp204 と完全同一枠組み(純GR15ch+typewell cross-attn+dTVT cumsum積分+同一recovered_fold+MPS、
6段 WARP_U2Net)を土台に、**EMA のみ追加**の clean A/B。狙い=長尺 cumsum drift による val 振動を
平滑化し安定した checkpoint を得る(writeup「WARP + multi-scale + EMA」でも併用)。

  ../exp008/.venv/bin/python train_ema.py --all-folds --epochs 40 --decay 0.999

各 step 後に shadow = decay·shadow + (1-decay)·param を更新。val/best/OOF は **EMA 重み**で評価。
比較対象: exp204 6段 raw = standalone 13.16 / PF corr0.411 / PF×arm −0.91。
"""
import os, sys, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp204"))
import joblib
from warp_model import WARP_U2Net, warp_ds_loss  # noqa: 6段
from train_warp import (WARPDataset, collate, compute_norm, pooled_rmse,  # noqa: reuse exp204
                        predict_well, DEVICE)

# Suite override: ROGII_WARP_CACHE / ROGII_WARP_OUT (shared5fold OOF suite)
CACHE = Path(os.environ.get("ROGII_WARP_CACHE", str(HERE.parent / "exp204" / "gr_features_cache.pkl")))
OUT_DIR = Path(os.environ.get("ROGII_WARP_OUT", str(HERE)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP201 = HERE.parent / "exp201" / "blend_cache.npz"


class EMA:
    """重み指数移動平均。shadow を保持し、eval 時に model へ swap-in/out。"""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)   # BN num_batches_tracked 等は非平均でコピー

    def swap_in(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def swap_out(self, model):
        model.load_state_dict(self._backup); self._backup = None


def train_fold(fold, wells, args):
    # Optional nested stacking: also drop --also-exclude-fold wells from train.
    also = getattr(args, "also_exclude_fold", -1)
    exclude = {int(fold)}
    if also is not None and int(also) >= 0:
        exclude.add(int(also))
    tr = [w for w in wells if int(w["fold"]) not in exclude]
    va = [w for w in wells if int(w["fold"]) == int(fold)]
    if not tr or not va:
        raise RuntimeError(f"empty split fold={fold} also_exclude={also}")
    norm = compute_norm(tr)
    seed_key = 49 + int(fold) * 1000 + (0 if also is None or int(also) < 0 else (int(also) + 1) * 17)
    torch.manual_seed(seed_key); np.random.seed(seed_key)
    nw = int(os.environ.get("ROGII_WARP_NUM_WORKERS", "16") or "16")
    dl_kw = dict(
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = int(os.environ.get("ROGII_WARP_PREFETCH", "4") or "4")
    print(f"  [EMA] fold{fold} DataLoader num_workers={nw} pin_memory={dl_kw['pin_memory']}", flush=True)
    dl = DataLoader(WARPDataset(tr, norm, L_max=args.lmax, train=True), **dl_kw)
    model = WARP_U2Net(in_ch=tr[0]["features"].shape[1], tw_T=tr[0]["tw_tokens"].shape[0],
                       use_typewell=not args.no_typewell).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, args.decay)
    flat = pooled_rmse(va, {w["wid"]: np.full(w["n_eval"], w["last_tvt"], np.float32) for w in va})
    best = float("inf"); best_state = None; patience = 0
    best_raw = float("inf"); best_raw_state = None   # raw-best checkpoint も別途保持
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); ls = 0.0; nb = 0
        for feat, tw, tgt, mask in dl:
            feat = feat.transpose(1, 2).to(DEVICE); tw = tw.to(DEVICE)
            tgt = tgt.to(DEVICE); mask = mask.to(DEVICE)
            opt.zero_grad()
            d_fuse, d_sides = model(feat, tw)
            loss = warp_ds_loss(d_fuse, d_sides, tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model); ls += loss.item(); nb += 1
        sch.step()
        # raw val(参考)
        model.eval()
        vr_raw = pooled_rmse(va, {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va})
        # EMA val(主指標)
        ema.swap_in(model); model.eval()
        vr = pooled_rmse(va, {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va})
        ema.swap_out(model)
        if vr_raw < best_raw:
            best_raw = vr_raw; best_raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if vr < best:
            best = vr; best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}; patience = 0
        else:
            patience += 1
        if ep == 0 or (ep + 1) % 3 == 0 or patience == 0:
            print(f"  [EMA] fold{fold} ep{ep+1:02d} loss={ls/max(nb,1):.3f} val_ema={vr:.4f} "
                  f"val_raw={vr_raw:.4f} bestEMA={best:.4f} flat={flat:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if patience >= args.patience:
            print(f"  [EMA] fold{fold} early stop ep{ep+1}"); break
    model.load_state_dict(best_state); model.eval()
    tag = f"fold{fold}" if also is None or int(also) < 0 else f"outer{also}_inner{fold}"
    torch.save(best_state, OUT_DIR / f"warp_ema_{tag}.pt")
    torch.save(best_raw_state, OUT_DIR / f"warp_raw_{tag}.pt")  # report 失敗時の保険
    oof_ema = {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va}
    model.load_state_dict(best_raw_state); model.eval()
    oof_raw = {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va}
    # Always persist per-fold OOF (needed for parallel fold jobs / shared suite)
    np.savez_compressed(
        OUT_DIR / f"oof_raw_{tag}.npz",
        wid=np.array([w["wid"] for w in va]),
        pred=np.concatenate([oof_raw[w["wid"]] for w in va]),
        true=np.concatenate([w["target_tvt"] for w in va]),
        n=np.array([w["n_eval"] for w in va]),
        also_exclude_fold=np.array([-1 if also is None else int(also)]),
    )
    np.savez_compressed(
        OUT_DIR / f"oof_ema_{tag}.npz",
        wid=np.array([w["wid"] for w in va]),
        pred=np.concatenate([oof_ema[w["wid"]] for w in va]),
        true=np.concatenate([w["target_tvt"] for w in va]),
        n=np.array([w["n_eval"] for w in va]),
        also_exclude_fold=np.array([-1 if also is None else int(also)]),
    )
    return oof_ema, oof_raw, best, best_raw, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--also-exclude-fold", type=int, default=-1,
                    help="Nested stack: also drop this fold from train (outer held).")
    ap.add_argument("--no-typewell", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--decay", type=float, default=0.999)
    args = ap.parse_args()

    wells = joblib.load(CACHE)
    print(f"[exp207 EMA decay={args.decay}] {len(wells)} wells | device={DEVICE} "
          f"| typewell={'OFF' if args.no_typewell else 'ON'} epochs={args.epochs} batch={args.batch} lmax={args.lmax}")
    folds = list(range(5)) if args.all_folds else [args.fold]
    oof_e = {}; oof_r = {}; info = {}
    t0 = time.time()
    for f in folds:
        oe, orw, best, best_raw, flat = train_fold(f, wells, args)
        oof_e.update(oe); oof_r.update(orw); info[f] = dict(best_ema=best, best_raw=best_raw, flat=flat)
        print(f"[EMA fold{f}] bestEMA={best:.4f} bestRAW={best_raw:.4f} flat={flat:.4f} Δflat={best-flat:+.4f}", flush=True)

    if not args.all_folds:
        json.dump(dict(fold=args.fold, info={str(k): v for k, v in info.items()}),
                  open(OUT_DIR / f"fold{args.fold}.json", "w"), indent=2)
        print(f"[exp207] fold{args.fold} done in {time.time()-t0:.0f}s"); return

    wid_order = [w["wid"] for w in wells]; by = {w["wid"]: w for w in wells}
    true = np.concatenate([by[w]["target_tvt"] for w in wid_order])
    n = np.array([by[w]["n_eval"] for w in wid_order])
    flat_all = pooled_rmse(wells, {w["wid"]: np.full(w["n_eval"], w["last_tvt"], np.float32) for w in wells})
    # [zip package 用パッチ] exp201 の PF blend_cache は任意。無ければ standalone OOF のみ報告
    if EXP201.exists():
        d = np.load(EXP201, allow_pickle=True)
        pf = d["base"] if list(d["wid"]) == wid_order else None
        pf_true = d["true"] if pf is not None else None
    else:
        print("[warn] exp201/blend_cache.npz 不在 → PF 相関/ブレンドはスキップ(standalone のみ)")
        pf = pf_true = None

    def report(tag, oof):
        pred = np.concatenate([oof[w] for w in wid_order])
        cv = float(np.sqrt(np.mean((pred - true) ** 2)))
        rec = dict(cv=cv)
        line = f"[{tag}] standalone OOF = {cv:.4f}"
        if pf is not None:
            corr = float(np.corrcoef(pred - true, pf - pf_true)[0, 1])
            pf_pool = float(np.sqrt(np.mean((pf - pf_true) ** 2)))
            sweep = {round(float(wb), 2): float(np.sqrt(np.mean((wb * pf + (1 - wb) * pred - true) ** 2)))
                     for wb in np.arange(0.5, 1.001, 0.05)}
            bw = min(sweep, key=sweep.get)
            rec.update(corr=corr, blend_best_w=bw, blend_best=sweep[bw], blend_delta=sweep[bw] - pf_pool)
            line += f" | PF corr={corr:.4f} | PF×arm blend w_pf={bw}={sweep[bw]:.4f} (Δ={sweep[bw]-pf_pool:+.4f})"
        print(line, flush=True)
        np.savez_compressed(OUT_DIR / f"oof_{tag}.npz", wid=np.array(wid_order), pred=pred, true=true, n=n)
        return rec

    print(f"\n=== exp207 EMA(decay={args.decay}) 5-fold honest OOF | flat={flat_all:.4f} (15.9099 で整合) ===")
    out = dict(method=f"ema{args.decay}", flat=flat_all, info={str(k): v for k, v in info.items()},
               time_s=time.time() - t0, ema=report("ema", oof_e), raw=report("raw", oof_r))
    json.dump(out, open(OUT_DIR / "res_ema.json", "w"), indent=2)
    print(f"[exp207] EMA done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
