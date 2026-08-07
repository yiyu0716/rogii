#!/usr/bin/env python3
"""Retrain public train19 LGB with exact online-style sg_path features.

This fixes the LB10 failure mode: the previous LGB was trained with cached
sg_path features from a strong base, while the Kaggle kernel generated sg_path
from pf_ancc_d.  This script makes train OOF use the same online generator as
the kernel.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/rogii")
CV_DIR = ROOT / "cv"
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

from public_train19_lgb5fold_repro import train_lgb_5fold  # noqa: E402

ART_ROOT = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709"
DEFAULT_XY = ART_ROOT / "full_train19_cache/Xy.parquet"
DEFAULT_FEATURES = ART_ROOT / "full_train19_cache/features.json"
DEFAULT_DATA_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
DEFAULT_ASSET_ROOT = ROOT / "datasets_upload/rogii-public-train19-sgpath-lgb5fold-assets"
DEFAULT_ONLINE_CODE = ROOT / "code/kaggle_kernel_rogii-public-train19-sgpath-lgb-v001"
OUT_ROOT = ART_ROOT / "online_sgpath_pfanchor_retrain_v1"
TEST_LIKE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def _well_id_from_path(path: Path) -> str:
    return path.name.replace("__horizontal_well.csv", "").replace("__typewell.csv", "")


def prepare_test_like_root(source_root: Path, out_root: Path, wells: list[str] | None = None) -> Path:
    """Create a data root whose test split looks like official test files.

    ``train`` remains available for train-space surface priors.  ``test`` is
    populated from train wells, but horizontal files keep only the official
    test columns, removing train-only formation surfaces and TVT truth.
    """

    source_root = Path(source_root)
    out_root = Path(out_root)
    train_src = source_root / "train"
    if not train_src.exists():
        raise FileNotFoundError(train_src)
    out_root.mkdir(parents=True, exist_ok=True)

    train_dst = out_root / "train"
    if train_dst.exists() or train_dst.is_symlink():
        if train_dst.is_symlink() and train_dst.resolve() == train_src.resolve():
            pass
        elif train_dst.is_dir() and not train_dst.is_symlink():
            shutil.rmtree(train_dst)
            train_dst.symlink_to(train_src, target_is_directory=True)
        else:
            train_dst.unlink()
            train_dst.symlink_to(train_src, target_is_directory=True)
    else:
        train_dst.symlink_to(train_src, target_is_directory=True)

    test_dst = out_root / "test"
    if test_dst.exists():
        shutil.rmtree(test_dst)
    test_dst.mkdir(parents=True, exist_ok=True)

    all_wells = sorted(_well_id_from_path(p) for p in train_src.glob("*__horizontal_well.csv"))
    selected = all_wells if wells is None else [str(w) for w in wells]
    missing = [w for w in selected if not (train_src / f"{w}__horizontal_well.csv").exists()]
    if missing:
        raise FileNotFoundError(f"missing train wells for test-like root: {missing[:10]}")

    for wid in selected:
        hw_src = train_src / f"{wid}__horizontal_well.csv"
        tw_src = train_src / f"{wid}__typewell.csv"
        hw = pd.read_csv(hw_src)
        keep = [c for c in TEST_LIKE_COLS if c in hw.columns]
        if keep != TEST_LIKE_COLS:
            raise ValueError(f"{hw_src} missing official-test columns: {sorted(set(TEST_LIKE_COLS) - set(keep))}")
        hw[TEST_LIKE_COLS].to_csv(test_dst / hw_src.name, index=False)
        if not tw_src.exists():
            raise FileNotFoundError(tw_src)
        shutil.copy2(tw_src, test_dst / tw_src.name)
    return out_root


def _load_online_lib(code_root: Path, asset_root: Path):
    for path in [Path(code_root), Path(asset_root)]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import live_v2_candidate_generator as livevp

    livevp._SURFACE_PRIOR_CACHE.clear()
    if hasattr(livevp, "VP_DIVERSE_TOPK"):
        livevp.VP_DIVERSE_TOPK = int(os.environ.get("ROGII_SGPATH_DIVERSE_TOPK", "60") or "60")
    from sg_path_online_lib import build_sg_path_features

    return build_sg_path_features


def build_or_load_online_sg_sidecar(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    sidecar = Path(args.sidecar)
    if sidecar.exists() and not args.force_rebuild_sidecar:
        side = pd.read_parquet(sidecar)
        print(f"loaded online sg sidecar {side.shape} from {sidecar}", flush=True)
        return side

    t0 = time.time()
    wells = sorted(df["well"].astype(str).unique().tolist())
    masked_root = prepare_test_like_root(Path(args.data_root), Path(args.test_like_root), wells=wells)
    build_sg_path_features = _load_online_lib(Path(args.online_code), Path(args.asset_root))
    side = build_sg_path_features(
        df,
        masked_root,
        Path(args.asset_root),
        base_source=str(args.base_source),
    )
    side = side.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    side.to_parquet(sidecar, index=False)
    print(f"saved online sg sidecar {side.shape} to {sidecar} elapsed={time.time() - t0:.1f}s", flush=True)
    return side


def visible_public_ids(data_root: Path) -> list[str]:
    sample_path = Path(data_root) / "sample_submission.csv"
    if not sample_path.exists():
        return []
    return pd.read_csv(sample_path)["id"].astype(str).tolist()


def score_visible_public(df: pd.DataFrame, pred_drift: np.ndarray, data_root: Path) -> dict[str, float]:
    ids = set(visible_public_ids(data_root))
    if not ids:
        return {}
    mask = df["id"].astype(str).isin(ids).to_numpy()
    if not mask.any() or "target" not in df.columns:
        return {}
    err = np.asarray(pred_drift, dtype=np.float64)[mask] - df.loc[mask, "target"].to_numpy(np.float64)
    out = {
        "visible_rows": float(mask.sum()),
        "visible_rmse": float(np.sqrt(np.mean(err * err))),
        "visible_mean_error": float(np.mean(err)),
        "visible_std_error": float(np.std(err)),
    }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy", type=Path, default=DEFAULT_XY)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    ap.add_argument("--online-code", type=Path, default=DEFAULT_ONLINE_CODE)
    ap.add_argument("--test-like-root", type=Path, default=OUT_ROOT / "train_as_test_root")
    ap.add_argument("--sidecar", type=Path, default=OUT_ROOT / "online_sgpath_pfanchor_features.parquet")
    ap.add_argument("--force-rebuild-sidecar", action="store_true")
    ap.add_argument("--base-source", default="pf_ancc_d")
    ap.add_argument("--full-5fold", action="store_true")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=4000)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-child-samples", type=int, default=80)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.6)
    ap.add_argument("--reg-lambda", type=float, default=10.0)
    ap.add_argument("--reg-alpha", type=float, default=1.0)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--log-period", type=int, default=400)
    ap.add_argument("--lgb-jobs", type=int, default=-1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    needed = sorted(set(["well", "id", "target"] + base_features))
    df = pd.read_parquet(args.xy, columns=needed)
    if int(args.max_wells) > 0:
        keep = sorted(df["well"].astype(str).unique().tolist())[: int(args.max_wells)]
        df = df[df["well"].astype(str).isin(keep)].reset_index(drop=True)
        args.sidecar = out_dir / f"online_sgpath_pfanchor_features_smoke{int(args.max_wells)}.parquet"
        args.test_like_root = out_dir / f"train_as_test_root_smoke{int(args.max_wells)}"
    side = build_or_load_online_sg_sidecar(args, df)
    sg_cols = [c for c in side.columns if c.startswith("sg_")]
    if len(sg_cols) != 17:
        raise RuntimeError(f"expected 17 sg columns, got {len(sg_cols)}: {sg_cols}")
    feats = base_features + sg_cols
    work = pd.concat([df.reset_index(drop=True), side.reset_index(drop=True)], axis=1)
    (out_dir / "features.json").write_text(json.dumps(feats, indent=2) + "\n", encoding="utf-8")
    (out_dir / "sg_columns.json").write_text(json.dumps(sg_cols, indent=2) + "\n", encoding="utf-8")
    meta = {
        "recipe": "public_train19_lgb_5fold_exact_online_sgpath_pfanchor",
        "args": vars(args),
        "base_features": len(base_features),
        "sg_features": len(sg_cols),
        "n_features": len(feats),
        "rows": len(work),
        "wells": int(work["well"].nunique()),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    if args.full_5fold:
        args.out_dir = out_dir
        summary, _ = train_lgb_5fold(work, feats, args)
        summary.to_csv(out_dir / "summary.csv", index=False)
        pred = np.load(out_dir / "oof.npy")
        meta["visible_public_oof"] = score_visible_public(work, pred, Path(args.data_root))
        (out_dir / "metadata_online_sgpath.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps(meta.get("visible_public_oof", {}), indent=2, sort_keys=True), flush=True)
    else:
        (out_dir / "metadata_online_sgpath.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"built sidecar only out_dir={out_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
