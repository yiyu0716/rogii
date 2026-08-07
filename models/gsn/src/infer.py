import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.cnn_sdf_config import Config
from .dataset import WellboreSDFDataset
from .model import GeoSteerNet, build_viterbi_cost_matrix
from .seed import seed_everything
from .train import map_h_to_original, viterbi_decode_future


def _decode_tvt_to_original(
    batch,
    sdf: np.ndarray,
    seg: np.ndarray,
    df: pd.DataFrame,
    *,
    decode: str = "viterbi",
) -> np.ndarray:
    """Decode SDF (+seg) to original-sample TVT.

    Uses the same ``t_seg_tvt`` / ``h_seg_tvt`` grid as the dataset batch so row
    indices from the SDF field align with TVT values (must not re-resample here).

    Args:
        decode: ``"viterbi"`` (default) or ``"argmin"`` (per-column |sdf| min).
    """
    decode = decode.lower()
    if decode not in ("viterbi", "argmin"):
        raise ValueError(f"decode must be 'viterbi' or 'argmin', got {decode!r}")

    t_seg_tvt = batch["t_seg_tvt"].numpy()[0]
    h_seg_tvt = batch["h_seg_tvt"].numpy()[0]
    h_ps = int(batch["h_ps"].item())
    h_s = int(batch["h_s"].item()) if "h_s" in batch else Config.H_S
    anchor_t_idx = int(batch["anchor_t_idx"].item())

    if decode == "argmin":
        best_t_idx = np.argmin(np.abs(sdf), axis=0)
    else:
        cost_mat = build_viterbi_cost_matrix(sdf, seg)
        best_t_idx = viterbi_decode_future(
            cost_mat,
            anchor_t_idx=anchor_t_idx,
            H_H=Config.H_H,
            transition_penalty=0.10,
        )

    pred_tvt_H = t_seg_tvt[best_t_idx]
    pred_tvt_H_eval = pred_tvt_H.copy()
    pred_tvt_H_eval[: Config.H_H] = h_seg_tvt[: Config.H_H]

    anchor_tvt = None
    if "TVT_input" in df.columns and df["TVT_input"].notna().any():
        anchor_tvt = float(df["TVT_input"].dropna().iloc[-1])

    return map_h_to_original(
        pred_tvt_H_eval,
        o_len=len(df),
        h_ps=h_ps,
        H_S=h_s,
        H_H=Config.H_H,
        H_F=Config.H_F,
        anchor_tvt=anchor_tvt,
    )


def _build_submission_rows(
    well_name: str,
    df: pd.DataFrame,
    pred_resampled: np.ndarray,
) -> list[dict]:
    if "TVT_input" in df.columns:
        final_tvt = df["TVT_input"].values.copy().astype(np.float64)
        submit_mask = df["TVT_input"].isna().values
    else:
        final_tvt = np.full(len(df), np.nan, dtype=np.float64)
        submit_mask = np.ones(len(df), dtype=bool)

    final_tvt[submit_mask] = pred_resampled[submit_mask]

    rows = []
    for idx in np.flatnonzero(submit_mask):
        rows.append({
            "id": f"{well_name}_{idx}",
            "tvt": float(final_tvt[idx]),
        })
    return rows


def _normalize_folds(folds: int | list[int] | tuple[int, ...] | str | None) -> list[int]:
    """``None`` / ``"all"`` → all folds; ``int`` → single fold; sequence → explicit list."""
    if folds is None or folds == "all":
        return list(range(Config.N_FOLDS))
    if isinstance(folds, int):
        return [folds]
    return [int(f) for f in folds]


def _load_models(
    model_dir: Path,
    checkpoint: str,
    folds: list[int],
) -> list[tuple[int, GeoSteerNet]]:
    use_cuda = torch.cuda.is_available()
    loaded: list[tuple[int, GeoSteerNet]] = []
    for fold in folds:
        model_path = model_dir / f"fold_{fold}_{checkpoint}.pth"
        if not model_path.is_file():
            print(f"Warning: missing {model_path}, skipped")
            continue
        model = GeoSteerNet().to(Config.DEVICE)
        model.output_type = ["inference"]
        model.load_state_dict(
            torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
        )
        model.eval()
        loaded.append((fold, model))
        print(f"Loaded fold {fold}: {model_path}")
    return loaded


def run_inference(
    model_dir=".",
    checkpoint="best",
    output_path="submission.csv",
    folds: int | list[int] | tuple[int, ...] | str | None = None,
    decode: str = "viterbi",
):
    """Run test inference and write submission CSV.

    Args:
        model_dir: Directory with ``fold_*_{checkpoint}.pth`` checkpoints.
        checkpoint: Checkpoint suffix, e.g. ``"best"`` or ``"last"``.
        output_path: Output CSV path.
        folds: Which folds to use.
            - ``None`` or ``"all"``: every fold in ``[0, N_FOLDS)`` that exists
            - ``int``: single fold, e.g. ``0``
            - sequence: explicit folds, e.g. ``[0, 2]``
            Multiple folds are averaged (sdf/seg) before decode.
        decode: ``"viterbi"`` (default) or ``"argmin"`` per-column |sdf| decode.
    """
    seed_everything()
    model_dir = Path(model_dir)
    fold_list = _normalize_folds(folds)
    test_dir = Path(Config.TEST_DIR)
    well_files = sorted(test_dir.glob(f"*{Config.HORIZONTAL_SUFFIX}"))
    if not well_files:
        print("No test data found in", test_dir)
        return None

    models = _load_models(model_dir, checkpoint, fold_list)
    if not models:
        print(f"No models loaded for folds={fold_list}. Exiting.")
        return None

    fold_ids = [f for f, _ in models]
    if len(models) == 1:
        print(f"Using single fold: {fold_ids[0]}\n")
    else:
        print(f"Ensembling {len(models)} folds: {fold_ids}\n")
    print(f"Decode: {decode.lower()}\n")

    submission_data = []
    use_cuda = torch.cuda.is_available()
    use_amp = use_cuda and str(Config.DEVICE).startswith("cuda")

    for well_file in tqdm(well_files, desc="Inference"):
        well_name = well_file.name.split("__")[0]
        df = pd.read_csv(well_file)

        typewell_path = well_file.parent / f"{well_name}{Config.TYPEWELL_SUFFIX}"
        if not typewell_path.exists():
            print(f"Typewell missing for {well_name}. Skipping.")
            continue

        loader = DataLoader(
            WellboreSDFDataset(well_files=[well_file], is_train=False),
            batch_size=1,
            shuffle=False,
        )

        for batch in loader:
            with torch.no_grad():
                fold_sdfs, fold_segs = [], []
                for _fold_id, model in models:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        output = model(batch)
                    fold_sdfs.append(output["sdf"].float().cpu().numpy())
                    fold_segs.append(output["seg"].float().cpu().numpy())

                avg_sdf = np.mean(fold_sdfs, axis=0)[0, 0]
                avg_seg = np.mean(fold_segs, axis=0)[0, 0]
                pred_resampled = _decode_tvt_to_original(
                    batch, avg_sdf, avg_seg, df, decode=decode,
                )
                submission_data.extend(
                    _build_submission_rows(well_name, df, pred_resampled)
                )

    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(output_path, index=False)
    print(sub_df.head())
    print(f"\nDone: {len(sub_df)} rows")
    print(f"Submission saved to {output_path}")
    return sub_df
