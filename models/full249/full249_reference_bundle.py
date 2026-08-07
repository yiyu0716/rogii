#!/usr/bin/env python3
"""Utilities for honest full249 pseudo-root bundles.

The production full249 feature builder expects a competition-like directory
layout with ``train/``, ``test/`` and ``sample_submission.csv``.  These helpers
materialize that layout from official train wells while enforcing exact-censor
visibility for target wells.

This module is deliberately small and explicit.  It is the shared contract used
by the V3.4 reconnaissance runner and, later, the honest FoldSystemPackage
builder.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path("/root/rogii")
COMP_ROOT = ROOT / "datasets/rogii-wellbore-geology-prediction"
FULL_ROOT = ROOT / "cv/artifacts/public_train19_lgb5fold_20260709"
DEFAULT_FULL249_ASSET = ROOT / "datasets_upload/rogii-public-train19-lgb5fold-assets"
DEFAULT_SOURCE = DEFAULT_FULL249_ASSET / "public_xgb_train19_source.py"
DEFAULT_SPLITS = ROOT / "cv/splits/spatial_folds.csv"

OFFICIAL_TEST_HORIZONTAL_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
OFFICIAL_TEST_TYPEWELL_COLUMNS = ["TVT", "GR"]
FORBIDDEN_TEST_COLUMNS = ["TVT", "U", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology", "target", "truth"]


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_frame(df: pd.DataFrame, *, include_index: bool = True) -> str:
    meta = {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "shape": list(df.shape),
    }
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
    vals = pd.util.hash_pandas_object(df, index=include_index).to_numpy(dtype=np.uint64)
    h.update(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()


def hash_wells(wells: Iterable[str]) -> str:
    return stable_json_hash(sorted(map(str, wells)))


def load_spatial_splits(path: Path = DEFAULT_SPLITS) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "well_id" not in df.columns or "fold" not in df.columns:
        raise ValueError(f"invalid split file columns: {df.columns.tolist()}")
    df = df.copy()
    df["well_id"] = df["well_id"].astype(str)
    df["fold"] = df["fold"].astype(int)
    return df


def select_outer_wells(
    *,
    outer_fold_zero_based: int,
    max_test_wells: int = 0,
    max_train_wells: int = 0,
    split_path: Path = DEFAULT_SPLITS,
) -> tuple[list[str], list[str], dict[str, Any]]:
    splits = load_spatial_splits(split_path)
    all_wells = sorted(splits["well_id"].astype(str).tolist())
    val_wells = sorted(splits.loc[splits["fold"].eq(int(outer_fold_zero_based)), "well_id"].astype(str).tolist())
    train_wells = sorted([w for w in all_wells if w not in set(val_wells)])
    if max_test_wells and max_test_wells > 0:
        val_wells = val_wells[: int(max_test_wells)]
    if max_train_wells and max_train_wells > 0:
        train_wells = train_wells[: int(max_train_wells)]
    meta = {
        "split_path": str(split_path),
        "outer_fold_zero_based": int(outer_fold_zero_based),
        "outer_fold_one_based_label": int(outer_fold_zero_based) + 1,
        "n_all_wells": int(len(all_wells)),
        "n_train_wells": int(len(train_wells)),
        "n_validation_wells": int(len(val_wells)),
        "train_well_hash": hash_wells(train_wells),
        "validation_well_hash": hash_wells(val_wells),
        "max_test_wells": int(max_test_wells),
        "max_train_wells": int(max_train_wells),
    }
    return train_wells, val_wells, meta


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _first_hidden_index(tvt_input: pd.Series) -> int:
    na = np.flatnonzero(tvt_input.isna().to_numpy())
    if len(na):
        return int(na[0])
    return int(len(tvt_input))


def _choose_cut(df: pd.DataFrame, cut_variant: str, earlier_holdout_rows: int = 96) -> int:
    official_cut = _first_hidden_index(df["TVT_input"])
    if cut_variant == "official":
        return official_cut
    if cut_variant == "early":
        cut = max(16, official_cut - int(earlier_holdout_rows))
        return min(cut, max(16, len(df) - 1))
    if cut_variant.startswith("frac:"):
        frac = float(cut_variant.split(":", 1)[1])
        return int(np.clip(round(len(df) * frac), 16, len(df) - 1))
    raise ValueError(f"unknown cut_variant={cut_variant}")


def materialize_pseudo_root(
    *,
    out_dir: Path,
    train_wells: list[str],
    target_wells: list[str],
    cut_variant: str,
    source_comp_root: Path = COMP_ROOT,
    earlier_holdout_rows: int = 96,
    copy_train: bool = True,
) -> dict[str, Any]:
    """Create a competition-shaped pseudo root and return its manifest."""
    pseudo_root = out_dir / f"pseudo_root_{cut_variant.replace(':', '_')}"
    if pseudo_root.exists():
        shutil.rmtree(pseudo_root)
    (pseudo_root / "train").mkdir(parents=True, exist_ok=True)
    (pseudo_root / "test").mkdir(parents=True, exist_ok=True)

    if copy_train:
        for wid in train_wells:
            for suffix in ["__horizontal_well.csv", "__typewell.csv"]:
                src = source_comp_root / "train" / f"{wid}{suffix}"
                if not src.exists():
                    raise FileNotFoundError(src)
                _link_or_copy(src, pseudo_root / "train" / src.name)

    sample_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    cut_rows: list[dict[str, Any]] = []
    masked_hash_items: dict[str, str] = {}

    for wid in target_wells:
        hw_src = source_comp_root / "train" / f"{wid}__horizontal_well.csv"
        tw_src = source_comp_root / "train" / f"{wid}__typewell.csv"
        if not hw_src.exists() or not tw_src.exists():
            raise FileNotFoundError(f"missing source files for target well {wid}")
        hw = pd.read_csv(hw_src)
        cut = _choose_cut(hw, cut_variant, earlier_holdout_rows=earlier_holdout_rows)
        if cut < 1 or cut >= len(hw):
            raise ValueError(f"bad cut={cut} for {wid} n={len(hw)}")
        truth_tvt = hw["TVT"].to_numpy(np.float64) if "TVT" in hw.columns else np.full(len(hw), np.nan)
        masked = hw.copy()
        masked.loc[cut:, "TVT_input"] = np.nan
        test_hw = masked[[c for c in OFFICIAL_TEST_HORIZONTAL_COLUMNS if c in masked.columns]].copy()
        missing = [c for c in OFFICIAL_TEST_HORIZONTAL_COLUMNS if c not in test_hw.columns]
        if missing:
            raise RuntimeError(f"{wid} missing official columns {missing}")
        forbidden_present = [c for c in FORBIDDEN_TEST_COLUMNS if c in test_hw.columns and c not in OFFICIAL_TEST_HORIZONTAL_COLUMNS]
        if forbidden_present:
            raise RuntimeError(f"forbidden test columns leaked for {wid}: {forbidden_present}")
        test_hw.to_csv(pseudo_root / "test" / f"{wid}__horizontal_well.csv", index=False)

        tw = pd.read_csv(tw_src)
        test_tw = tw[[c for c in OFFICIAL_TEST_TYPEWELL_COLUMNS if c in tw.columns]].copy()
        test_tw.to_csv(pseudo_root / "test" / f"{wid}__typewell.csv", index=False)

        for ridx in range(cut, len(test_hw)):
            sample_rows.append({"id": f"{wid}_{ridx}", "tvt": 0.0})
            truth_rows.append({"id": f"{wid}_{ridx}", "well": wid, "row_idx": ridx, "tvt": float(truth_tvt[ridx])})
        cut_rows.append(
            {
                "well": wid,
                "cut_variant": cut_variant,
                "cut_row": int(cut),
                "official_cut_row": int(_first_hidden_index(hw["TVT_input"])),
                "n_rows": int(len(hw)),
                "n_hidden_rows": int(len(hw) - cut),
                "visible_tvt_count": int(test_hw["TVT_input"].notna().sum()),
            }
        )
        masked_hash_items[wid] = hash_frame(test_hw)

    sample = pd.DataFrame(sample_rows)
    if sample.empty:
        raise RuntimeError("pseudo sample_submission is empty")
    sample.to_csv(pseudo_root / "sample_submission.csv", index=False)
    truth = pd.DataFrame(truth_rows)
    truth.to_parquet(out_dir / f"truth_{cut_variant.replace(':', '_')}.parquet", index=False)
    cut_df = pd.DataFrame(cut_rows)
    cut_df.to_csv(out_dir / f"cut_manifest_{cut_variant.replace(':', '_')}.csv", index=False)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pseudo_root": str(pseudo_root),
        "source_comp_root": str(source_comp_root),
        "cut_variant": cut_variant,
        "n_train_wells": int(len(train_wells)),
        "n_target_wells": int(len(target_wells)),
        "n_sample_rows": int(len(sample)),
        "train_well_hash": hash_wells(train_wells),
        "target_well_hash": hash_wells(target_wells),
        "masked_view_hash": stable_json_hash(masked_hash_items),
        "sample_hash": hash_frame(sample),
        "copy_train": bool(copy_train),
        "official_test_horizontal_columns": OFFICIAL_TEST_HORIZONTAL_COLUMNS,
        "official_test_typewell_columns": OFFICIAL_TEST_TYPEWELL_COLUMNS,
    }
    (out_dir / f"pseudo_root_manifest_{cut_variant.replace(':', '_')}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


@dataclass
class Full249ReferenceBundle:
    """Manifest-level representation of a full249 reference bundle."""

    bundle_id: str
    pseudo_root: str
    train_wells: list[str]
    target_wells: list[str]
    cut_variant: str
    source_path: str
    asset_root: str
    feature_schema_hash: str
    fit_well_ids_hash: str
    fit_row_ids_hash: str
    label_columns_used: list[str]
    direct_target_overlap_count: int
    transitive_target_overlap_count: str
    code_hash: str
    config_hash: str
    upstream_asset_hashes: dict[str, str]
    masked_view_hash: str
    serialization_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def build_reference_bundle_manifest(
    *,
    pseudo_manifest: dict[str, Any],
    train_wells: list[str],
    target_wells: list[str],
    source_path: Path = DEFAULT_SOURCE,
    asset_root: Path = DEFAULT_FULL249_ASSET,
    feature_columns: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> Full249ReferenceBundle:
    if feature_columns is None:
        feature_columns = json.loads((asset_root / "feature_columns.json").read_text(encoding="utf-8"))
    upstream = {
        "source": sha256_file(source_path),
        "feature_columns": sha256_file(asset_root / "feature_columns.json"),
        "manifest": sha256_file(asset_root / "manifest.json"),
    }
    payload = {
        "pseudo_manifest": pseudo_manifest,
        "train_wells": sorted(train_wells),
        "target_wells": sorted(target_wells),
        "source_hash": upstream["source"],
        "feature_schema": feature_columns,
        "config": config or {},
    }
    serialization_hash = stable_json_hash(payload)
    return Full249ReferenceBundle(
        bundle_id=f"full249_bundle_{serialization_hash[:12]}",
        pseudo_root=str(pseudo_manifest["pseudo_root"]),
        train_wells=sorted(train_wells),
        target_wells=sorted(target_wells),
        cut_variant=str(pseudo_manifest["cut_variant"]),
        source_path=str(source_path),
        asset_root=str(asset_root),
        feature_schema_hash=stable_json_hash(feature_columns),
        fit_well_ids_hash=hash_wells(train_wells),
        fit_row_ids_hash="not_materialized_row_hash_v1",
        label_columns_used=["TVT", "TVT_input", "formation_surfaces"],
        direct_target_overlap_count=int(len(set(train_wells).intersection(set(target_wells)))),
        transitive_target_overlap_count="unknown_until_runtime_trace",
        code_hash=sha256_file(source_path),
        config_hash=stable_json_hash(config or {}),
        upstream_asset_hashes=upstream,
        masked_view_hash=str(pseudo_manifest["masked_view_hash"]),
        serialization_hash=serialization_hash,
    )


def import_public_source(
    *,
    source_path: Path,
    data_dir: Path,
    feature_env: dict[str, Any] | None = None,
    module_tag: str = "runtime",
):
    """Import public_xgb_train19_source with DATA_DIR bound to a pseudo root."""
    os.environ["MODE"] = "infer"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    if feature_env:
        for key, value in feature_env.items():
            os.environ[str(key)] = str(value)
    module_name = f"public_xgb_train19_source_{module_tag}_{hashlib.md5(str(data_dir).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def transform_censored_cohort(
    *,
    bundle: Full249ReferenceBundle,
    source_path: Path = DEFAULT_SOURCE,
    asset_root: Path = DEFAULT_FULL249_ASSET,
    feature_env: dict[str, Any] | None = None,
    tag: str = "test",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build full249 features for a bundle's pseudo-test cohort."""
    t0 = time.time()
    pseudo_root = Path(bundle.pseudo_root)
    public = import_public_source(
        source_path=source_path,
        data_dir=pseudo_root,
        feature_env=feature_env,
        module_tag=tag,
    )
    if os.environ.get("ROGII_PROGRESS_GENPHYS", "0") == "1" and hasattr(public, "genphys_one"):
        orig_genphys_one = public.genphys_one

        def _progress_genphys_one(hw_path, sdip_arr=None):
            wid = Path(hw_path).stem.replace("__horizontal_well", "")
            t1 = time.time()
            print(f"    [genphys] start {wid}", flush=True)
            out = orig_genphys_one(hw_path, sdip_arr)
            rows = 0 if out is None else len(out)
            print(f"    [genphys] done {wid}: rows={rows} elapsed={time.time() - t1:.1f}s", flush=True)
            return out

        public.genphys_one = _progress_genphys_one
    si, dfield = public._build_imputers(t0)
    test_df = public._build_full(public.DATA / "test", False, si, dfield, t0, tag)
    feature_columns = json.loads((asset_root / "feature_columns.json").read_text(encoding="utf-8"))
    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"missing generated features: {missing[:30]}")
    meta = {
        "tag": tag,
        "elapsed_sec": float(time.time() - t0),
        "rows": int(len(test_df)),
        "wells": int(test_df["well"].nunique()) if "well" in test_df.columns else None,
        "columns": int(test_df.shape[1]),
        "feature_schema_hash": stable_json_hash(feature_columns),
        "feature_frame_hash": hash_frame(test_df[feature_columns].replace([np.inf, -np.inf], np.nan)),
    }
    return test_df, meta


def feature_group_fingerprints(test_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    groups: list[tuple[str, list[str]]] = [
        ("direct_visible_prefix", [c for c in feature_columns if c in {"last_tvt", "md_since", "dz", "dxy", "frac", "gr", "gr_m21", "gr_na_frac", "known_len", "eval_len", "ktvt_range", "tw_range"}]),
        ("pf_typewell_matching", [c for c in feature_columns if c.startswith(("pf_", "tdpf", "tdbc", "tdsc"))]),
        ("beam_mode_candidate", [c for c in feature_columns if c.startswith(("beam", "mk", "mode", "pmf", "mconf", "grm", "cal_", "noc_", "sc_"))]),
        ("surface_dip_formation", [c for c in feature_columns if c.startswith(("dip", "sdip", "surf", "srf", "fc_", "fm_", "ancc", "ast", "egf", "buda"))]),
    ]
    used = set(sum([cols for _, cols in groups], []))
    groups.append(("other_public_train19", [c for c in feature_columns if c not in used]))
    rows = []
    for group, cols in groups:
        present = [c for c in cols if c in test_df.columns]
        if present:
            arr = test_df[present].replace([np.inf, -np.inf], np.nan)
            nan_frac = float(arr.isna().to_numpy().mean())
            value_hash = hash_frame(arr.astype("float32", copy=False), include_index=False)
        else:
            nan_frac = float("nan")
            value_hash = ""
        rows.append(
            {
                "feature_group": group,
                "n_declared_features": int(len(cols)),
                "n_present_features": int(len(present)),
                "nan_frac": nan_frac,
                "value_hash": value_hash,
                "columns": json.dumps(present),
            }
        )
    return pd.DataFrame(rows)
