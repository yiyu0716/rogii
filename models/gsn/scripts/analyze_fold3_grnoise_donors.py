#!/usr/bin/env python3
"""Compare GRNoise donor distributions for fold-3 under SEED=42 (v2.6) vs SEED=2026 (v2.9).

Replays 60 training epochs with a RNG-faithful fake base dataset + real
GRNoiseAugDataset donor logic (pool[randint]), matching DataLoader shuffle /
worker seeding used in ``src/train.py``.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.cnn_sdf_config import Config  # noqa: E402
from src.dataset import (  # noqa: E402
    GRNoiseAugDataset,
    list_well_cv_splits,
    sample_split_ps,
    well_xy_centroid,
    _true_ps_index,
)
from src.seed import dataloader_worker_init_fn, seed_everything  # noqa: E402
from src.train import make_dataloader  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grnoise_donor_analysis_fold3"
EPOCHS = 60
SEEDS = {"v2.6": 42, "v2.9": 2026}
FOLD = 3


class RNGFaithfulFakeBase(Dataset):
    """Consumes the same np.random calls as WellboreSDFDataset train ``__getitem__``."""

    def __init__(self, well_files: list[Path], pfe_eligible: list[np.ndarray | None]):
        self.well_files = well_files
        self.pfe_eligible = pfe_eligible
        self.is_train = True
        self.config = Config

    def __len__(self) -> int:
        return len(self.well_files)

    def __getitem__(self, idx: int):
        cfg = self.config
        _ = int(np.random.randint(cfg.H_GR_FILTER[0], cfg.H_GR_FILTER[1]))
        _ = bool(np.random.rand() < 0.5)  # apply_savgol coin
        if cfg.USE_PFE_TRAIN:
            if np.random.random() < cfg.PFE_PROB:
                elig = self.pfe_eligible[idx]
                if elig is not None and len(elig) > 0:
                    _ = int(np.random.choice(elig))
        _ = int(np.random.randint(0, cfg.H_S))  # bin_offset
        if np.random.rand() < cfg.H_FLIP_PROB:
            pass
        return {"idx": idx}


def _pfe_eligible_for_well(well_path: Path) -> np.ndarray | None:
    h = pd.read_csv(well_path)
    true_ps = _true_ps_index(h)
    # Mirror sample_split_ps eligibility without consuming RNG
    n = len(h)
    first_ps = 0
    if "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        first_ps = int(np.flatnonzero(h["TVT_input"].notna().values)[0])
    pseudo_min = max(first_ps, Config.PFE_MIN_HISTORY_ORIG - 1)
    pseudo_max = min(true_ps - 1, n - 1 - Config.PFE_MIN_FUTURE_ORIG)
    if pseudo_min > pseudo_max:
        return None
    if "TVT" in h.columns and h["TVT"].notna().sum() > 0:
        tvt_vals = h["TVT"].values.astype(np.float64)
    elif "TVT_input" in h.columns and h["TVT_input"].notna().sum() > 0:
        tvt_vals = (
            h["TVT_input"].astype(float).ffill().bfill().fillna(0.0).values.astype(np.float64)
        )
    else:
        tvt_vals = h["Z"].values.astype(np.float64)
    ps_tvt = float(tvt_vals[true_ps])
    history_tvt_min = ps_tvt - float(Config.PFE_TVT_SHIFT_THRESHOLD)
    segment = tvt_vals[pseudo_min : pseudo_max + 1]
    eligible = np.flatnonzero(segment > history_tvt_min) + pseudo_min
    if eligible.size == 0:
        return None
    return eligible.astype(np.int64)


def _compute_noise_stats(well_path: Path) -> dict:
    sample_id = well_path.name.split("__")[0]
    typewell_path = well_path.parent / f"{sample_id}{Config.TYPEWELL_SUFFIX}"
    h = pd.read_csv(well_path)
    t = pd.read_csv(typewell_path)
    tw_tvt = t["TVT"].values.astype(np.float64)
    tw_gr = (
        t["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values.astype(np.float64)
    )
    if "TVT" not in h.columns or h["TVT"].isna().all():
        return {
            "noise_std": 0.0,
            "noise_mad": 0.0,
            "noise_p95_abs": 0.0,
            "noise_rms": 0.0,
            "noise_lag1": 0.0,
            "n_pts": 0,
        }
    h_tvt = pd.to_numeric(h["TVT"], errors="coerce").values
    h_gr_raw = h["GR"].astype(float).interpolate().bfill().ffill().fillna(85.0).values
    valid = np.isfinite(h_tvt) & np.isfinite(h_gr_raw)
    h_tvt = h_tvt[valid]
    h_gr_raw = h_gr_raw[valid]
    if len(h_tvt) < 5:
        return {
            "noise_std": 0.0,
            "noise_mad": 0.0,
            "noise_p95_abs": 0.0,
            "noise_rms": 0.0,
            "noise_lag1": 0.0,
            "n_pts": int(len(h_tvt)),
        }
    win = Config.H_GR_FILTER[0]
    if win % 2 == 0:
        win += 1
    if len(h_gr_raw) > win:
        h_gr_smooth = savgol_filter(h_gr_raw, win, 2).astype(np.float64)
    else:
        h_gr_smooth = h_gr_raw.astype(np.float64)
    tw_order = np.argsort(tw_tvt)
    h_gr_sim = np.interp(h_tvt, tw_tvt[tw_order], tw_gr[tw_order])
    noise = h_gr_smooth - h_gr_sim
    # Prefer future-ish half for stats (noise transfer targets future)
    mid = len(noise) // 2
    fut = noise[mid:] if mid < len(noise) else noise
    lag1 = float(np.corrcoef(fut[:-1], fut[1:])[0, 1]) if len(fut) > 2 else 0.0
    if not np.isfinite(lag1):
        lag1 = 0.0
    return {
        "noise_std": float(np.std(fut)),
        "noise_mad": float(np.median(np.abs(fut - np.median(fut)))),
        "noise_p95_abs": float(np.percentile(np.abs(fut), 95)),
        "noise_rms": float(np.sqrt(np.mean(fut**2))),
        "noise_lag1": lag1,
        "n_pts": int(len(noise)),
    }


def _geo_rank_matrix(xy: np.ndarray) -> np.ndarray:
    """ranks[i, j] = rank of j among neighbors of i by distance (0 = nearest other)."""
    N = len(xy)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(np.argsort(d, axis=1), axis=1)


def build_fold3_train() -> tuple[list[Path], np.ndarray]:
    train_dir = Config.TRAIN_DIR
    well_files = sorted(train_dir.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    splits = list_well_cv_splits(well_files, train_dir=train_dir)
    train_idx, val_idx = splits[FOLD]
    train_files = [well_files[i] for i in train_idx]
    return train_files, train_idx


def precompute_well_table(train_files: list[Path]) -> pd.DataFrame:
    rows = []
    for i, wf in enumerate(tqdm(train_files, desc="precompute wells")):
        x, y = well_xy_centroid(wf)
        stats = _compute_noise_stats(wf)
        rows.append(
            {
                "a_idx": i,
                "well_id": wf.name.split("__")[0],
                "path": str(wf),
                "x": x,
                "y": y,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def simulate_donors(
    seed: int,
    train_files: list[Path],
    pfe_eligible: list[np.ndarray | None],
    epochs: int = EPOCHS,
) -> pd.DataFrame:
    """Run DataLoader epochs; log every donor-group refresh."""
    Config.SEED = seed
    seed_everything(seed)

    base = RNGFaithfulFakeBase(train_files, pfe_eligible)
    n_synth = int(Config.GR_NOISE_N_SYNTH)
    wrap = GRNoiseAugDataset(
        base,
        donor=Config.GR_NOISE_DONOR,
        k_neighbors=Config.GR_NOISE_K,
        future_only=Config.GR_NOISE_FUTURE_ONLY,
        n_synth=n_synth,
    )

    records: list[dict] = []
    orig_pick = wrap._pick_synth_donors

    def traced_pick(a_idx: int) -> list[int]:
        donors = orig_pick(a_idx)
        # epoch stamped externally via closure
        traced_pick.pending.append((a_idx, donors))
        return donors

    traced_pick.pending = []
    wrap._pick_synth_donors = traced_pick  # type: ignore

    loader = make_dataloader(wrap, shuffle=True)
    N = len(base)

    for ep in tqdm(range(1, epochs + 1), desc=f"seed={seed} epochs"):
        traced_pick.pending = []
        # Drain one epoch
        for batch in loader:
            pass
        # pending is per-process; with workers>0 the main process sees nothing.
        # So we must use num_workers=0 for logging OR log inside worker (hard).
        # Force: if workers logged nothing, fall back is handled by caller.
        for a_idx, donors in traced_pick.pending:
            for synth_i, c_idx in enumerate(donors):
                records.append(
                    {
                        "seed": seed,
                        "epoch": ep,
                        "a_idx": int(a_idx),
                        "synth_i": int(synth_i),
                        "c_idx": int(c_idx),
                    }
                )

    return pd.DataFrame(records)


def simulate_donors_mainprocess(
    seed: int,
    train_files: list[Path],
    pfe_eligible: list[np.ndarray | None],
    epochs: int = EPOCHS,
) -> pd.DataFrame:
    """Faithful RNG replay with num_workers=0 so donor picks are observable.

    Shuffle still uses torch Generator(Config.SEED). Base aug RNG uses
    seed_everything(seed) then continues across epochs — same as a single-worker
    training run. (Training default uses 4 workers; relative seed-42 vs 2026
    comparison remains valid on the same worker setting.)
    """
    Config.SEED = seed
    seed_everything(seed)

    base = RNGFaithfulFakeBase(train_files, pfe_eligible)
    wrap = GRNoiseAugDataset(
        base,
        donor=str(Config.GR_NOISE_DONOR),
        k_neighbors=int(Config.GR_NOISE_K),
        future_only=bool(Config.GR_NOISE_FUTURE_ONLY),
        n_synth=int(Config.GR_NOISE_N_SYNTH),
    )

    records: list[dict] = []
    orig_pick = wrap._pick_synth_donors

    def traced_pick(a_idx: int) -> list[int]:
        donors = orig_pick(a_idx)
        for synth_i, c_idx in enumerate(donors):
            records.append(
                {
                    "seed": seed,
                    "epoch": traced_pick.epoch,
                    "a_idx": int(a_idx),
                    "synth_i": int(synth_i),
                    "c_idx": int(c_idx),
                }
            )
        return donors

    traced_pick.epoch = 0
    wrap._pick_synth_donors = traced_pick  # type: ignore

    # Skip real GR rewrite; we only care about donor indices + RNG order.
    wrap._apply_noise_aug = lambda sample, a_idx, c_idx=None: sample  # type: ignore

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        wrap,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g,
        collate_fn=lambda xs: xs,  # fake samples are plain dicts
    )

    for ep in tqdm(range(1, epochs + 1), desc=f"seed={seed} nw0"):
        traced_pick.epoch = ep
        for _ in loader:
            pass

    return pd.DataFrame(records)


def enrich(df: pd.DataFrame, wells: pd.DataFrame, ranks: np.ndarray, xy: np.ndarray) -> pd.DataFrame:
    w = wells.set_index("a_idx")
    out = df.copy()
    out["a_well"] = out["a_idx"].map(w["well_id"])
    out["c_well"] = out["c_idx"].map(w["well_id"])
    out["a_noise_std"] = out["a_idx"].map(w["noise_std"])
    out["c_noise_std"] = out["c_idx"].map(w["noise_std"])
    out["c_noise_mad"] = out["c_idx"].map(w["noise_mad"])
    out["c_noise_p95"] = out["c_idx"].map(w["noise_p95_abs"])
    out["c_noise_rms"] = out["c_idx"].map(w["noise_rms"])
    out["c_noise_lag1"] = out["c_idx"].map(w["noise_lag1"])
    ax = out["a_idx"].map(w["x"]).to_numpy()
    ay = out["a_idx"].map(w["y"]).to_numpy()
    cx = out["c_idx"].map(w["x"]).to_numpy()
    cy = out["c_idx"].map(w["y"]).to_numpy()
    out["geo_dist"] = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
    out["geo_rank"] = [int(ranks[a, c]) for a, c in zip(out["a_idx"], out["c_idx"])]
    N = len(wells)
    out["geo_rank_pct"] = out["geo_rank"] / max(N - 2, 1)
    out["noise_std_ratio"] = out["c_noise_std"] / (out["a_noise_std"] + 1e-6)
    out["noise_std_diff"] = out["c_noise_std"] - out["a_noise_std"]
    out["is_knn5"] = out["geo_rank"] < 5
    out["is_knn20"] = out["geo_rank"] < 20
    out["is_knn50"] = out["geo_rank"] < 50
    # tertiles of donor noise std within train pool
    q33, q66 = w["noise_std"].quantile([0.33, 0.66]).tolist()
    out["c_noise_tertile"] = pd.cut(
        out["c_noise_std"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "mid", "high"],
    )
    q33d, q66d = np.quantile(out["geo_dist"], [0.33, 0.66])
    # use pool-wide distance tertiles from this df later in summarize
    out["dist_tertile_global"] = pd.cut(
        out["geo_dist"],
        bins=[-np.inf, q33d, q66d, np.inf],
        labels=["near", "mid", "far"],
    )
    return out


def summarize(label: str, df: pd.DataFrame, wells: pd.DataFrame) -> dict:
    N = len(wells)
    # baseline: random donor expectation
    # mean geo rank under uniform ≈ (N-2)/2
    exp_rank = (N - 2) / 2.0
    exp_knn5 = 5 / (N - 1)

    def pct(x):
        return float(100.0 * np.mean(x))

    s = {
        "label": label,
        "seed": int(df["seed"].iloc[0]),
        "n_pairs": int(len(df)),
        "n_train_wells": N,
        "n_unique_donors": int(df["c_idx"].nunique()),
        "n_unique_pairs": int(df.drop_duplicates(["a_idx", "c_idx"]).shape[0]),
        "geo_dist_mean": float(df["geo_dist"].mean()),
        "geo_dist_median": float(df["geo_dist"].median()),
        "geo_dist_p25": float(df["geo_dist"].quantile(0.25)),
        "geo_dist_p75": float(df["geo_dist"].quantile(0.75)),
        "geo_rank_mean": float(df["geo_rank"].mean()),
        "geo_rank_median": float(df["geo_rank"].median()),
        "geo_rank_expected_uniform": float(exp_rank),
        "pct_knn5": pct(df["is_knn5"]),
        "pct_knn20": pct(df["is_knn20"]),
        "pct_knn50": pct(df["is_knn50"]),
        "pct_knn5_expected": float(100 * exp_knn5),
        "c_noise_std_mean": float(df["c_noise_std"].mean()),
        "c_noise_std_median": float(df["c_noise_std"].median()),
        "c_noise_mad_mean": float(df["c_noise_mad"].mean()),
        "c_noise_p95_mean": float(df["c_noise_p95"].mean()),
        "c_noise_rms_mean": float(df["c_noise_rms"].mean()),
        "c_noise_lag1_mean": float(df["c_noise_lag1"].mean()),
        "noise_std_ratio_mean": float(df["noise_std_ratio"].mean()),
        "noise_std_ratio_median": float(df["noise_std_ratio"].median()),
        "noise_std_diff_mean": float(df["noise_std_diff"].mean()),
        "pct_donor_noise_low": pct(df["c_noise_tertile"] == "low"),
        "pct_donor_noise_mid": pct(df["c_noise_tertile"] == "mid"),
        "pct_donor_noise_high": pct(df["c_noise_tertile"] == "high"),
        "pct_dist_near": pct(df["dist_tertile_global"] == "near"),
        "pct_dist_mid": pct(df["dist_tertile_global"] == "mid"),
        "pct_dist_far": pct(df["dist_tertile_global"] == "far"),
        # popularity: Gini of donor usage
        "donor_usage_gini": float(_gini(df["c_idx"].value_counts().to_numpy())),
        "top10_donor_share_pct": float(
            100 * df["c_idx"].value_counts().head(10).sum() / len(df)
        ),
    }
    # histograms for canvas
    dist_bins = np.quantile(df["geo_dist"], np.linspace(0, 1, 11))
    dist_bins = np.unique(dist_bins)
    if len(dist_bins) < 3:
        dist_bins = np.linspace(df["geo_dist"].min(), df["geo_dist"].max() + 1e-6, 6)
    hist_d, edges_d = np.histogram(df["geo_dist"], bins=dist_bins)
    s["geo_dist_hist"] = {
        "edges": [float(x) for x in edges_d],
        "counts": [int(x) for x in hist_d],
    }
    ns_bins = np.quantile(wells["noise_std"], np.linspace(0, 1, 11))
    ns_bins = np.unique(ns_bins)
    hist_n, edges_n = np.histogram(df["c_noise_std"], bins=ns_bins)
    s["c_noise_std_hist"] = {
        "edges": [float(x) for x in edges_n],
        "counts": [int(x) for x in hist_n],
    }
    # rank histogram in 10 buckets
    rank_pct = df["geo_rank_pct"].clip(0, 1)
    hist_r, edges_r = np.histogram(rank_pct, bins=10, range=(0, 1))
    s["geo_rank_pct_hist"] = {
        "edges": [float(x) for x in edges_r],
        "counts": [int(x) for x in hist_r],
    }
    return s


def _gini(counts: np.ndarray) -> float:
    if counts.size == 0:
        return 0.0
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum()) - (n + 1) / n))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GR_NOISE_N_SYNTH={Config.GR_NOISE_N_SYNTH} DONOR={Config.GR_NOISE_DONOR}")
    train_files, _ = build_fold3_train()
    print(f"Fold {FOLD}: n_train={len(train_files)}")

    print("Caching PFE eligible sets...")
    pfe_eligible = []
    for wf in tqdm(train_files, desc="pfe"):
        pfe_eligible.append(_pfe_eligible_for_well(wf))

    wells = precompute_well_table(train_files)
    wells.to_csv(OUT_DIR / "fold3_train_wells.csv", index=False)
    xy = wells[["x", "y"]].to_numpy(dtype=np.float64)
    ranks = _geo_rank_matrix(xy)

    summaries = {}
    enriched = {}
    for label, seed in SEEDS.items():
        print(f"\n===== {label} seed={seed} =====")
        raw = simulate_donors_mainprocess(seed, train_files, pfe_eligible, epochs=EPOCHS)
        raw.to_csv(OUT_DIR / f"donors_{label}_seed{seed}.csv", index=False)
        en = enrich(raw, wells, ranks, xy)
        en.to_csv(OUT_DIR / f"donors_enriched_{label}_seed{seed}.csv", index=False)
        summaries[label] = summarize(label, en, wells)
        enriched[label] = en
        print(json.dumps(summaries[label], indent=2)[:1200])

    # Pairwise deltas
    a, b = summaries["v2.6"], summaries["v2.9"]
    delta = {
        k: (b[k] - a[k])
        for k in a
        if isinstance(a[k], (int, float)) and isinstance(b.get(k), (int, float))
    }
    report = {
        "fold": FOLD,
        "epochs": EPOCHS,
        "n_synth": int(Config.GR_NOISE_N_SYNTH),
        "note": (
            "num_workers=0 RNG-faithful replay; fold split uses GEO_KMEANS_SEED "
            "(same wells for both SEED). Compares donor draws under Config.SEED "
            "42 vs 2026. Fold3 log mins: v2.6 argmin 8.36 / v2.9 argmin 7.55."
        ),
        "summaries": summaries,
        "delta_v29_minus_v26": delta,
        "pool_noise_std_ref": {
            "mean": float(wells["noise_std"].mean()),
            "median": float(wells["noise_std"].median()),
            "p33": float(wells["noise_std"].quantile(0.33)),
            "p66": float(wells["noise_std"].quantile(0.66)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    print("\nWrote", OUT_DIR / "summary.json")
    print("Delta v2.9 - v2.6 (key):")
    for k in [
        "geo_dist_mean",
        "geo_rank_mean",
        "pct_knn5",
        "pct_knn20",
        "c_noise_std_mean",
        "pct_donor_noise_high",
        "pct_donor_noise_low",
        "pct_dist_near",
        "pct_dist_far",
        "noise_std_ratio_mean",
        "donor_usage_gini",
    ]:
        print(f"  {k}: {delta.get(k, float('nan')):+.4f}")


if __name__ == "__main__":
    main()
