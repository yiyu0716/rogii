"""
exp204 WARP-相当アームの訓練 + honest OOF(exp140 と同一 recovered_fold 分割)。

  ../exp008/.venv/bin/python train_warp.py --fold 0                 # fold0 proof-of-concept
  ../exp008/.venv/bin/python train_warp.py --fold 0 --no-typewell   # typewell ablation(生GRのみ)
  ../exp008/.venv/bin/python train_warp.py --all-folds              # 5-fold honest OOF(重い)

訓練=ランダムクロップ(L_max, ERF~150点なので情報損失なし)で dTVT 増分→cumsum 積分、
積分済み delta に deep-supervision RMSE。推論=全 eval を1パスで cumsum(実アンカー起点)。
判定は pooled TVT RMSE。flat persistence(15.9099)を毎回併記(集計健全性)。device=MPS。
"""
import os, sys, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from warp_model import WARP_U2Net, warp_ds_loss  # noqa

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda:0" if torch.cuda.is_available() else "cpu"))
CACHE = HERE / "gr_features_cache.pkl"


def compute_norm(wells):
    F = np.concatenate([w["features"] for w in wells], 0)
    TW = np.concatenate([w["tw_tokens"] for w in wells], 0)
    return dict(fmean=F.mean(0).astype(np.float32), fstd=np.maximum(F.std(0), 1e-6).astype(np.float32),
                tmean=TW.mean(0).astype(np.float32), tstd=np.maximum(TW.std(0), 1e-6).astype(np.float32))


class WARPDataset(Dataset):
    def __init__(self, wells, norm, L_max=2560, train=True):
        self.wells = wells; self.norm = norm; self.L_max = L_max; self.train = train

    def __len__(self):
        return len(self.wells)

    def __getitem__(self, i):
        w = self.wells[i]
        feat = (w["features"] - self.norm["fmean"]) / self.norm["fstd"]
        tw = (w["tw_tokens"] - self.norm["tmean"]) / self.norm["tstd"]
        true = w["target_tvt"]; last = w["last_tvt"]; nh = len(true)
        if self.train and nh > self.L_max:
            s = np.random.randint(0, nh - self.L_max + 1); e = s + self.L_max
        else:
            s, e = 0, nh
        anchor = last if s == 0 else float(true[s - 1])
        tgt_delta = (true[s:e] - anchor).astype(np.float32)
        return (torch.from_numpy(feat[s:e].astype(np.float32)),   # (Lc, C)
                torch.from_numpy(tw.astype(np.float32)),           # (T, 2)
                torch.from_numpy(tgt_delta))                        # (Lc,)


def collate(batch):
    Ls = [b[0].shape[0] for b in batch]; Lmax = max(Ls); C = batch[0][0].shape[1]
    B = len(batch)
    feat = torch.zeros(B, Lmax, C); tgt = torch.zeros(B, Lmax); mask = torch.zeros(B, Lmax)
    tw = torch.stack([b[1] for b in batch])
    for j, (f, _, t) in enumerate(batch):
        L = f.shape[0]; feat[j, :L] = f; tgt[j, :L] = t; mask[j, :L] = 1.0
    return feat, tw, tgt, mask


@torch.no_grad()
def predict_well(model, w, norm, device):
    feat = ((w["features"] - norm["fmean"]) / norm["fstd"]).astype(np.float32)
    tw = ((w["tw_tokens"] - norm["tmean"]) / norm["tstd"]).astype(np.float32)
    x = torch.from_numpy(feat.T).unsqueeze(0).to(device)          # (1,C,L)
    twt = torch.from_numpy(tw).unsqueeze(0).to(device)            # (1,T,2)
    d_fuse, _ = model(x, twt)
    return d_fuse[0].cpu().numpy() + w["last_tvt"]                # 予測 TVT


def pooled_rmse(wells, preds):
    sse = sum(float(np.sum((w["target_tvt"] - preds[w["wid"]]) ** 2)) for w in wells)
    n = sum(w["n_eval"] for w in wells)
    return float(np.sqrt(sse / n))


def train_fold(fold, wells, args):
    tr = [w for w in wells if w["fold"] != fold]
    va = [w for w in wells if w["fold"] == fold]
    norm = compute_norm(tr)
    torch.manual_seed(49 + fold * 1000); np.random.seed(49 + fold * 1000)

    ds = WARPDataset(tr, norm, L_max=args.lmax, train=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    collate_fn=collate, num_workers=0)
    model = WARP_U2Net(in_ch=tr[0]["features"].shape[1], tw_T=tr[0]["tw_tokens"].shape[0],
                       use_typewell=not args.no_typewell).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    flat = pooled_rmse(va, {w["wid"]: np.full(w["n_eval"], w["last_tvt"], np.float32) for w in va})
    best = float("inf"); best_state = None; patience = 0
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
            opt.step(); ls += loss.item(); nb += 1
        sch.step()
        model.eval()
        preds = {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va}
        vr = pooled_rmse(va, preds)
        if vr < best:
            best = vr; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if ep == 0 or (ep + 1) % 3 == 0 or patience == 0:
            print(f"  fold{fold} ep{ep+1:02d} trloss={ls/max(nb,1):.4f} val_tvt={vr:.4f} "
                  f"best={best:.4f} flat={flat:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if patience >= args.patience:
            print(f"  fold{fold} early stop ep{ep+1}"); break

    model.load_state_dict(best_state); model.eval()
    preds = {w["wid"]: predict_well(model, w, norm, DEVICE) for w in va}
    return model, best, preds, norm, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--no-typewell", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lmax", type=int, default=2560)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    wells = joblib.load(CACHE)
    print(f"[exp204] {len(wells)} wells | device={DEVICE} | typewell={'OFF' if args.no_typewell else 'ON'} "
          f"| epochs={args.epochs} batch={args.batch} lmax={args.lmax}")

    tag = "notw" if args.no_typewell else "tw"
    folds = list(range(5)) if args.all_folds else [args.fold]
    oof = {}; results = {}
    t0 = time.time()
    for f in folds:
        model, best, preds, norm, flat = train_fold(f, wells, args)
        oof.update(preds); results[f] = dict(best=best, flat=flat, n_val=len([w for w in wells if w["fold"] == f]))
        torch.save(model.state_dict(), HERE / f"warp_{tag}_fold{f}.pt")
        print(f"[fold{f}] best_val_tvt={best:.4f}  flat={flat:.4f}  Δflat={best-flat:+.4f}")

    if args.all_folds:
        allw = wells
        cv = pooled_rmse(allw, oof)
        flat_all = pooled_rmse(allw, {w["wid"]: np.full(w["n_eval"], w["last_tvt"], np.float32) for w in allw})
        print(f"\n=== exp204 5-fold honest OOF (TVT) = {cv:.4f}  | flat={flat_all:.4f} "
              f"(15.9099 で整合) | typewell={'OFF' if args.no_typewell else 'ON'} ===")
        import pandas as pd
        pd.DataFrame([{"wid": w["wid"], "n": w["n_eval"], "fold": w["fold"],
                       "tvt_true_last": w["last_tvt"],
                       "pred_pooled_sse": float(np.sum((w["target_tvt"] - oof[w["wid"]]) ** 2))}
                      for w in allw]).to_csv(HERE / f"oof_{tag}.csv", index=False)
        json.dump({"cv": cv, "flat": flat_all, "results": results, "time_s": time.time() - t0},
                  open(HERE / f"cv_{tag}.json", "w"), indent=2)
    else:
        json.dump({"fold": args.fold, "results": results, "time_s": time.time() - t0},
                  open(HERE / f"fold{args.fold}_{tag}.json", "w"), indent=2)
    print(f"[exp204] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
