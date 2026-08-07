import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config.cnn_sdf_config import Config
from .dataset import WellboreSDFDataset, list_well_cv_splits, describe_well_cv_splits
from .model import GeoSteerNet, build_viterbi_cost_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from .seed import dataloader_worker_init_fn, seed_everything

_VITERBI_ABS_DIFF_CACHE: dict[int, np.ndarray] = {}


def _viterbi_abs_diff(T: int) -> np.ndarray:
    if T not in _VITERBI_ABS_DIFF_CACHE:
        t_idx = np.arange(T, dtype=np.float32)
        _VITERBI_ABS_DIFF_CACHE[T] = np.abs(t_idx[:, None] - t_idx[None, :])
    return _VITERBI_ABS_DIFF_CACHE[T]



def make_dataloader(dataset, shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(Config.SEED)
        kwargs["generator"] = generator
    if Config.NUM_WORKERS > 0:
        kwargs["num_workers"] = Config.NUM_WORKERS
        kwargs["prefetch_factor"] = Config.PREFETCH_FACTOR
        kwargs["persistent_workers"] = Config.PERSISTENT_WORKERS
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
    return DataLoader(dataset, **kwargs)


def viterbi_decode_future(cost_matrix, anchor_t_idx, H_H, transition_penalty=0.10):
    """
    Finds the optimal path specifically in the evaluation (future) zone,
    forcing the sequence to start exactly from the known anchor position
    at index H_H - 1.
    """
    T, H = cost_matrix.shape
    dp = np.full((T, H), 1e9, dtype=np.float32)
    pointers = np.zeros((T, H), dtype=np.int32)
    
    # Force anchor starting state at H_H - 1
    dp[anchor_t_idx, H_H - 1] = cost_matrix[anchor_t_idx, H_H - 1]

    t_idx = np.arange(T)
    abs_diff = _viterbi_abs_diff(T)
    
    # Forward Pass starting from history boundary
    for h in range(H_H, H):
        prev_dp = dp[:, h-1]
        
        # trans_costs[i, j] = prev_dp[j] + penalty * |i - j|
        trans_costs = prev_dp[None, :] + transition_penalty * abs_diff
        
        # Best previous state index j for each current state i
        min_costs_idx = np.argmin(trans_costs, axis=1)
        
        dp[:, h] = cost_matrix[:, h] + trans_costs[t_idx, min_costs_idx]
        pointers[:, h] = min_costs_idx
        
    # Backward Pass starting from end of horizontal segment
    path = np.zeros(H, dtype=np.int32)
    path[:H_H] = anchor_t_idx  # Safe fallback for history indexes
    
    path[-1] = np.argmin(dp[:, -1])
    for h in range(H-1, H_H - 1, -1):
        path[h-1] = pointers[path[h], h]
        
    return path

def map_h_to_original(pred_tvt_H, o_len, h_ps, H_S, H_H, H_F, anchor_tvt=None):
    """Map binned H-grid TVT back to original sample indices.

    Bin centers sit at half-bin offsets from ``h_ps``; without an explicit knot at
    the last known index ``h_ps``, linear interp misses ``orig_tvt[h_ps]`` by ~1 ft.
    """
    k_hist = np.arange(H_H, dtype=np.float64)
    h_idx_hist = (H_H - 1 - k_hist).astype(np.int64)
    centers_hist = h_ps - k_hist * H_S - (H_S - 1) / 2.0

    k_fut = np.arange(H_F, dtype=np.float64)
    h_idx_fut = (H_H + k_fut).astype(np.int64)
    centers_fut = h_ps + 1.0 + k_fut * H_S + (H_S - 1) / 2.0

    centers = np.concatenate([centers_hist, centers_fut])
    vals = pred_tvt_H[np.concatenate([h_idx_hist, h_idx_fut])]

    if anchor_tvt is not None:
        centers = np.concatenate([centers, [float(h_ps)]])
        vals = np.concatenate([vals, [float(anchor_tvt)]])

    order = np.argsort(centers)
    return np.interp(np.arange(o_len, dtype=np.float64), centers[order], vals[order])


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch=None):
    model.train()
    if epoch is not None and hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    total_loss = 0.0
    total_sdf = 0.0
    total_seg = 0.0
    iterator = tqdm(dataloader, desc="Training", leave=False) if Config.SHOW_PROGRESS else dataloader
    for batch in iterator:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(batch)
            loss = output["loss"]
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_sdf += output["sdf_loss"].item()
        total_seg += output["seg_loss"].item()
    n = len(dataloader)
    return total_loss / n, total_sdf / n, total_seg / n

@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """Fast validation: forward pass + joint loss only (no Viterbi)."""
    model.eval()
    total_loss = 0.0
    total_sdf = 0.0
    total_seg = 0.0
    iterator = tqdm(dataloader, desc="Val loss", leave=False) if Config.SHOW_PROGRESS else dataloader
    for batch in iterator:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(batch)
        total_loss += output["loss"].item()
        total_sdf += output["sdf_loss"].item()
        total_seg += output["seg_loss"].item()
    n = max(len(dataloader), 1)
    return total_loss / n, total_sdf / n, total_seg / n


def _accumulate_future_rmse(
    best_t_idx: np.ndarray,
    *,
    t_seg_tvt: np.ndarray,
    h_seg_tvt: np.ndarray,
    orig_tvt: np.ndarray,
    orig_len: int,
    h_ps: int,
    h_s: int,
) -> tuple[float, int]:
    """Map decoded row path to original resolution and score future-only TVT RMSE."""
    pred_tvt_H = t_seg_tvt[best_t_idx]
    pred_tvt_H_eval = pred_tvt_H.copy()
    pred_tvt_H_eval[:Config.H_H] = h_seg_tvt[:Config.H_H]

    pred_resampled = map_h_to_original(
        pred_tvt_H_eval, orig_len, h_ps, h_s, Config.H_H, Config.H_F,
        anchor_tvt=float(orig_tvt[h_ps]) if h_ps < orig_len else None,
    )
    true_tvt = orig_tvt[:orig_len]
    if h_ps < orig_len - 1:
        eval_pred = pred_resampled[h_ps + 1:]
        eval_true = true_tvt[h_ps + 1:]
        return float(np.sum((eval_pred - eval_true) ** 2)), len(eval_true)
    return 0.0, 0


def _per_sample_future_rmse(
    sdf_np: np.ndarray,
    seg_np: np.ndarray,
    batch,
    b: int,
) -> dict[str, float | int]:
    """Future-only TVT RMSE components for one batch item."""
    t_seg_tvt = batch["t_seg_tvt"].numpy()[b]
    h_seg_tvt = batch["h_seg_tvt"].numpy()[b]
    orig_tvt = batch["orig_tvt"].numpy()[b]
    orig_len = int(batch["orig_len"].numpy()[b])
    h_ps_arr = batch["h_ps"].numpy()
    true_ps_arr = batch.get("true_ps")
    eval_ps = int(true_ps_arr.numpy()[b]) if true_ps_arr is not None else int(h_ps_arr[b])
    h_s_arr = batch.get("h_s")
    h_s = int(h_s_arr.numpy()[b]) if h_s_arr is not None else Config.H_S
    anchor_t_idx = int(batch["anchor_t_idx"].numpy()[b])

    common = dict(
        t_seg_tvt=t_seg_tvt,
        h_seg_tvt=h_seg_tvt,
        orig_tvt=orig_tvt,
        orig_len=orig_len,
        h_ps=eval_ps,
        h_s=h_s,
    )

    argmin_idx = np.argmin(np.abs(sdf_np), axis=0)
    se_a, n_a = _accumulate_future_rmse(argmin_idx, **common)

    cost_mat = build_viterbi_cost_matrix(sdf_np, seg_np)
    viterbi_idx = viterbi_decode_future(
        cost_mat,
        anchor_t_idx=anchor_t_idx,
        H_H=Config.H_H,
        transition_penalty=0.10,
    )
    se_v, n_v = _accumulate_future_rmse(viterbi_idx, **common)

    return {
        "n_future_pts": n_v,
        "se_argmin": se_a,
        "se_viterbi": se_v,
    }


@torch.no_grad()
def evaluate_rmse_diagnostics(model, dataloader, device) -> dict[str, float]:
    """Validation RMSE with both per-column argmin(|sdf|) and Viterbi decode."""
    model.eval()
    total_se = {"argmin": 0.0, "viterbi": 0.0}
    total_points = {"argmin": 0, "viterbi": 0}
    iterator = tqdm(dataloader, desc="Val RMSE", leave=False) if Config.SHOW_PROGRESS else dataloader
    for batch in iterator:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(batch)
        sdf_np = output["sdf"].squeeze(1).float().cpu().numpy()
        seg_np = output["seg"].squeeze(1).float().cpu().numpy()
        B = sdf_np.shape[0]

        for b in range(B):
            metrics = _per_sample_future_rmse(sdf_np[b], seg_np[b], batch, b)
            n = int(metrics["n_future_pts"])
            if n <= 0:
                continue
            total_se["argmin"] += float(metrics["se_argmin"])
            total_se["viterbi"] += float(metrics["se_viterbi"])
            total_points["argmin"] += n
            total_points["viterbi"] += n

    return {
        name: np.sqrt(total_se[name] / total_points[name]) if total_points[name] > 0 else 0.0
        for name in ("argmin", "viterbi")
    }


@torch.no_grad()
def evaluate_per_well_rmse(model, dataloader, device, *, fold: int | None = None) -> pd.DataFrame:
    """Per validation-well future TVT RMSE (for local CV / OOF scoring)."""
    model.eval()
    rows: list[dict] = []
    iterator = (
        tqdm(dataloader, desc="Val per-well RMSE", leave=False)
        if Config.SHOW_PROGRESS else dataloader
    )
    for batch in iterator:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            output = model(batch)
        sdf_np = output["sdf"].squeeze(1).float().cpu().numpy()
        seg_np = output["seg"].squeeze(1).float().cpu().numpy()
        B = sdf_np.shape[0]
        well_ids = batch["id"]
        if torch.is_tensor(well_ids):
            well_ids = well_ids.tolist()

        for b in range(B):
            metrics = _per_sample_future_rmse(sdf_np[b], seg_np[b], batch, b)
            n = int(metrics["n_future_pts"])
            se_a = float(metrics["se_argmin"])
            se_v = float(metrics["se_viterbi"])
            row = {
                "well_id": well_ids[b],
                "n_future_pts": n,
                "se_argmin": se_a,
                "se_viterbi": se_v,
                "rmse_argmin_ft": float(np.sqrt(se_a / n)) if n > 0 else float("nan"),
                "rmse_viterbi_ft": float(np.sqrt(se_v / n)) if n > 0 else float("nan"),
            }
            if fold is not None:
                row["fold"] = int(fold)
            rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def cache_fold_viz_rmse(
    out_dir: Path | str,
    fold: int,
    *,
    checkpoint: str = "best",
    recompute: bool = False,
    device: str | None = None,
) -> pd.DataFrame:
    """Train+val per-well viterbi RMSE for one fold; cache ``fold_{k}_viz_rmse.csv``."""
    out_dir = Path(out_dir)
    cache_path = out_dir / f"fold_{fold}_viz_rmse.csv"
    if cache_path.is_file() and not recompute:
        return pd.read_csv(cache_path)

    device = device or Config.DEVICE
    ckpt = out_dir / f"fold_{fold}_{checkpoint}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    train_dir = Path(Config.TRAIN_DIR)
    well_files = sorted(train_dir.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    tr_idx, va_idx = list_well_cv_splits(well_files, train_dir=train_dir)[fold]

    model = GeoSteerNet().to(device)
    model.output_type = ["inference"]
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    parts = []
    for split_name, idxs in (("train", tr_idx), ("val", va_idx)):
        files = [well_files[i] for i in idxs]
        if not files:
            continue
        loader = DataLoader(
            WellboreSDFDataset(well_files=files, is_train=False),
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        df = evaluate_per_well_rmse(model, loader, device, fold=fold)
        id2path = {f.name.split("__")[0]: str(f) for f in files}
        df["split"] = split_name
        df["well_path"] = df["well_id"].map(id2path)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cache_path, index=False)
    print(f"Cached {len(out)} wells -> {cache_path}")
    return out


def _pooled_rmse_from_per_well(df: pd.DataFrame, se_col: str, n_col: str = "n_future_pts") -> float:
    n = int(df[n_col].sum())
    if n <= 0:
        return float("nan")
    return float(np.sqrt(df[se_col].sum() / n))


@torch.no_grad()
def evaluate_rmse(model, dataloader, device):
    """Slow validation: Viterbi decode + original-resolution RMSE."""
    return evaluate_rmse_diagnostics(model, dataloader, device)["viterbi"]


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    val_loss, val_sdf, val_seg = evaluate_loss(model, dataloader, device)
    val_rmse = evaluate_rmse(model, dataloader, device)
    return val_loss, val_sdf, val_seg, val_rmse


def train_sdf_loop(
    model: GeoSteerNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    *,
    optimizer=None,
    scheduler=None,
    ckpt_path: Path | str | None = None,
    eval_every: int | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_epochs: int | None = None,
    tag: str = "",
) -> dict:
    """
    SDF 训练主循环：
      - 每 epoch：val loss + val RMSE（argmin + viterbi）
      - 以 argmin RMSE 改善为准保存 ckpt_path / early stop
    """
    patience = Config.EARLY_STOP_PATIENCE if early_stop_patience is None else early_stop_patience
    min_epochs = Config.EARLY_STOP_MIN_EPOCHS if early_stop_min_epochs is None else early_stop_min_epochs
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else None

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_rmse = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_best_rmse = 0
    log: list[dict] = []
    last_rmse = float("nan")
    last_argmin_rmse = float("nan")

    header = f"── SDF [{tag}]" if tag else "── SDF"
    print(f"\n{header}  epochs={epochs}  lr={Config.LR}  CosineAnnealingLR  "
          f"early_stop={patience} (argmin)  "
          f"ckpt={ckpt_path or 'none'} ──")

    for ep in range(1, epochs + 1):
        loss, sdf_l, seg_l = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch=ep,
        )
        val_loss, val_sdf, val_seg = evaluate_loss(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        rmse_diag = evaluate_rmse_diagnostics(model, val_loader, device)
        last_rmse = rmse_diag["viterbi"]
        last_argmin_rmse = rmse_diag["argmin"]
        is_best_rmse = False
        if last_argmin_rmse < best_rmse:
            best_rmse = last_argmin_rmse
            best_epoch = ep
            epochs_since_best_rmse = 0
            is_best_rmse = True
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_best_rmse += 1
        lr = optimizer.param_groups[0]["lr"]
        flag = " *best" if is_best_rmse else ""
        print(
            f"  ep {ep:02d} | loss {loss:.4f} "
            f"(sdf {sdf_l:.3f} seg {seg_l:.3f})"
            f" | lr {lr:.2e} | val loss {val_loss:.4f} "
            f"(sdf {val_sdf:.3f} seg {val_seg:.3f})"
            f" | argmin {last_argmin_rmse:.2f} | viterbi {last_rmse:.2f} ft{flag}"
        )

        log.append({
            "epoch":         ep,
            "train_loss":    round(loss, 4),
            "train_sdf_loss": round(sdf_l, 4),
            "train_seg_loss": round(seg_l, 4),
            "val_loss":      round(val_loss, 4),
            "val_sdf_loss":  round(val_sdf, 4),
            "val_seg_loss":  round(val_seg, 4),
            "val_rmse_argmin_ft": round(last_argmin_rmse, 2),
            "val_rmse_viterbi_ft": round(last_rmse, 2),
            "val_rmse_ft":   round(last_argmin_rmse, 2),  # selection metric
            "is_best_rmse":  is_best_rmse,
            "lr":            lr,
        })

        if scheduler is not None:
            scheduler.step()

        if (
            ep >= min_epochs
            and epochs_since_best_rmse >= patience
        ):
            print(
                f"  Early stop @ ep {ep}: argmin RMSE 连续 {patience} 次 eval 无改善 "
                f"(best={best_rmse:.2f} ft @ ep {best_epoch})"
            )
            break

    if ckpt_path is not None and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    return {
        "best_rmse":      best_rmse,
        "best_val_loss":  best_val_loss,
        "best_epoch":     best_epoch,
        "stopped_epoch":  log[-1]["epoch"] if log else 0,
        "log":            log,
        "rmse_hist":      [x["val_rmse_ft"] for x in log],
        "ckpt_path":      str(ckpt_path) if ckpt_path else None,
    }

def _release_gpu_memory() -> None:
    """Reclaim host RAM and GPU cache after a fold finishes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _run_fold_training(
    fold: int,
    train_files: list,
    val_files: list,
    output_dir: Path,
    epochs: int,
    device: str | None = None,
) -> dict:
    """Train a single CV fold; always releases loaders/model before returning."""
    device = device or Config.DEVICE
    print(f"\n{'='*20} Fold {fold} {'='*20}")

    train_ds = WellboreSDFDataset(well_files=train_files, is_train=True)
    if getattr(Config, "GR_NOISE_AUG", False):
        from src.dataset import GRNoiseAugDataset
        n_synth = getattr(Config, "GR_NOISE_N_SYNTH", 2)
        train_ds = GRNoiseAugDataset(
            train_ds,
            donor=getattr(Config, "GR_NOISE_DONOR", "random"),
            k_neighbors=getattr(Config, "GR_NOISE_K", 5),
            future_only=getattr(Config, "GR_NOISE_FUTURE_ONLY", True),
            n_synth=n_synth,
        )
        print(
            f"  GRNoiseAug: n_synth={n_synth} "
            f"donor={Config.GR_NOISE_DONOR} "
            f"future_only={Config.GR_NOISE_FUTURE_ONLY} "
            f"train {len(train_ds) // (1 + n_synth)} → {len(train_ds)} "
            f"({1 + n_synth}x)"
        )
    val_ds = WellboreSDFDataset(well_files=val_files, is_train=False)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader = make_dataloader(val_ds, shuffle=False)

    model = GeoSteerNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    try:
        result = train_sdf_loop(
            model,
            train_loader,
            val_loader,
            device,
            epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=output_dir / f"fold_{fold}_best.pth",
            tag=f"fold {fold}",
        )
        torch.save(model.state_dict(), output_dir / f"fold_{fold}_last.pth")

        fold_log_path = output_dir / f"fold_{fold}_log.csv"
        pd.DataFrame(result["log"]).to_csv(fold_log_path, index=False)

        # Best-checkpoint val RMSE (local CV scores for this fold)
        rmse_diag = evaluate_rmse_diagnostics(model, val_loader, device)
        per_well_rmse = evaluate_per_well_rmse(model, val_loader, device, fold=fold)
        per_well_path = output_dir / f"fold_{fold}_val_rmse.csv"
        per_well_rmse.to_csv(per_well_path, index=False)

        print(
            f"Fold {fold} Best Val argmin RMSE: {result['best_rmse']:.2f} ft "
            f"@ ep {result['best_epoch']}"
        )
        print(
            f"Fold {fold} Final val RMSE (best ckpt): "
            f"argmin {rmse_diag['argmin']:.2f} ft | "
            f"viterbi {rmse_diag['viterbi']:.2f} ft | "
            f"{len(per_well_rmse)} wells"
        )
        print(f"Fold {fold} stopped @ ep {result['stopped_epoch']}")
        print(f"Fold {fold} log: {fold_log_path}")
        print(f"Fold {fold} per-well RMSE: {per_well_path}")
        print(
            f"Fold {fold} checkpoints: {result['ckpt_path']}, "
            f"{output_dir / f'fold_{fold}_last.pth'}"
        )
        return {
            "fold": fold,
            "best_rmse": result["best_rmse"],
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"],
            "val_rmse_viterbi_ft": round(rmse_diag["viterbi"], 4),
            "val_rmse_argmin_ft": round(rmse_diag["argmin"], 4),
            "n_val_wells": len(per_well_rmse),
            "val_rmse_per_well_path": str(per_well_path),
            "ckpt_path": result["ckpt_path"],
            "last_ckpt_path": str(output_dir / f"fold_{fold}_last.pth"),
            "log_path": str(fold_log_path),
        }
    finally:
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        _release_gpu_memory()


def run_training_all_folds(
    output_dir: str | Path = ".",
    epochs: int | None = None,
    *,
    device: str | None = None,
) -> list[dict]:
    """Train every GroupKFold split sequentially; free GPU/RAM between folds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_everything()
    epochs = Config.EPOCHS if epochs is None else epochs
    device = device or Config.DEVICE

    train_dir = Path(Config.TRAIN_DIR)
    well_files = sorted(train_dir.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    if not well_files:
        print("No data found in", Config.TRAIN_DIR)
        return []

    cv_splits = list_well_cv_splits(well_files, train_dir=train_dir)

    print(f"Checkpoints will be saved to: {output_dir.resolve()}")
    print(
        f"Seed: {seed} | deterministic={Config.DETERMINISTIC} | "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )
    print(
        f"all_folds={Config.N_FOLDS}  cv={Config.CV_SPLIT_STRATEGY}  epochs={epochs}  "
        f"history_bb={Config.USE_HISTORY_IN_BACKBONE}  "
        f"history_hd={Config.USE_HISTORY_IN_HEAD}"
    )
    print(describe_well_cv_splits(well_files, cv_splits, train_dir=train_dir).to_string(index=False))

    fold_results: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        # if fold == 0 : continue
        train_files = [well_files[j] for j in train_idx]
        val_files = [well_files[j] for j in val_idx]
        fold_results.append(
            _run_fold_training(
                fold, train_files, val_files, output_dir, epochs, device=device,
            )
        )
        # break

    if fold_results:
        fold_summary = pd.DataFrame(fold_results)
        summary_path = output_dir / "all_folds_summary.csv"
        fold_summary.to_csv(summary_path, index=False)

        cv_per_well = pd.concat(
            [pd.read_csv(r["val_rmse_per_well_path"]) for r in fold_results],
            ignore_index=True,
        )
        cv_per_well_path = output_dir / "cv_per_well_rmse.csv"
        cv_per_well.to_csv(cv_per_well_path, index=False)

        oof_viterbi = _pooled_rmse_from_per_well(cv_per_well, "se_viterbi")
        oof_argmin = _pooled_rmse_from_per_well(cv_per_well, "se_argmin")
        cv_scores = {
            "cv_strategy": Config.CV_SPLIT_STRATEGY,
            "n_folds": len(fold_results),
            "n_val_wells": int(len(cv_per_well)),
            "oof_rmse_viterbi_ft": round(oof_viterbi, 4),
            "oof_rmse_argmin_ft": round(oof_argmin, 4),
            "mean_fold_rmse_viterbi_ft": round(fold_summary["val_rmse_viterbi_ft"].mean(), 4),
            "std_fold_rmse_viterbi_ft": round(fold_summary["val_rmse_viterbi_ft"].std(), 4),
            "mean_fold_rmse_argmin_ft": round(fold_summary["val_rmse_argmin_ft"].mean(), 4),
            "std_fold_rmse_argmin_ft": round(fold_summary["val_rmse_argmin_ft"].std(), 4),
        }
        cv_scores_path = output_dir / "cv_scores.csv"
        pd.DataFrame([cv_scores]).to_csv(cv_scores_path, index=False)

        print(f"\n{'='*20} All Folds Summary {'='*20}")
        for r in fold_results:
            print(
                f"  fold {r['fold']}: train-best argmin {r['best_rmse']:.2f} ft | "
                f"final val argmin {r['val_rmse_argmin_ft']:.2f} ft | "
                f"viterbi {r['val_rmse_viterbi_ft']:.2f} ft"
            )
        print(
            f"  OOF CV RMSE: argmin {oof_argmin:.2f} ft | viterbi {oof_viterbi:.2f} ft "
            f"({cv_scores['n_val_wells']} wells)"
        )
        print(f"  fold summary: {summary_path}")
        print(f"  per-well CV:  {cv_per_well_path}")
        print(f"  OOF scores:   {cv_scores_path}")

    return fold_results


def _save_training_logs(output_dir: Path, fold_logs: list[dict], fold: int) -> Path:
    fold_path = output_dir / f"fold_{fold}_train_log.csv"
    pd.DataFrame(fold_logs).to_csv(fold_path, index=False)
    return fold_path


def run_training(output_dir=".", fold: int = 0, epochs: int | None = None):
    """Train one CV fold; every epoch logs val loss + Viterbi RMSE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_everything()
    epochs = Config.EPOCHS if epochs is None else epochs

    train_dir = Path(Config.TRAIN_DIR)
    well_files = sorted(list(train_dir.glob(f"*{Config.HORIZONTAL_SUFFIX}")))
    if not well_files:
        print("No data found in", Config.TRAIN_DIR)
        return

    cv_splits = list_well_cv_splits(well_files, train_dir=train_dir)

    print(f"Checkpoints will be saved to: {output_dir.resolve()}")
    print(
        f"Seed: {seed} | deterministic={Config.DETERMINISTIC} | "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
    )
    print(
        f"fold={fold}  cv={Config.CV_SPLIT_STRATEGY}  epochs={epochs}  "
        f"history_bb={Config.USE_HISTORY_IN_BACKBONE}  "
        f"history_hd={Config.USE_HISTORY_IN_HEAD}"
    )

    fold_result = None
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        if i != fold:
            continue
        train_files = [well_files[j] for j in train_idx]
        val_files = [well_files[j] for j in val_idx]
        fold_result = _run_fold_training(
            fold, train_files, val_files, output_dir, epochs,
        )
        break

    if fold_result is not None:
        print(f"\n{'='*20} Final Results {'='*20}")
        print(f"Fold {fold} Best RMSE (ft): {fold_result['best_rmse']:.2f}")
        return fold_result["best_rmse"]
    return None
