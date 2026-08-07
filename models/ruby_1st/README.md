# Ruby 0724 V1 and 0803 V2: Geo5 No-XY Reproduction

## Scope

This directory contains a local GPU reproduction of exactly two public Ruby
artifacts from the ROGII competition:

| Recipe label | Public Kaggle dataset | Author-reported CV |
|---|---|---:|
| `0724v1` | `w5833946/rogii-0724-v1-cv486` | 4.86 |
| `0803v2` | `w5833946/rogii-0803-v2-cv500` | 5.00 |

The author-reported values use Ruby's own validation protocol and include XY
neighbor information. They are not directly comparable with the local result.

## Source and Inference Code

The unmodified public code/configuration dumps are retained under:

```text
source_0724v1/
source_0803v2/
```

The public inference entry point is `seq_NN_main.py --submit-mode`, which loads
the saved `cfg.pkl` and `models.pkl` from a completed training directory. Its
deployment implementation remains intact in each `source_*` directory.
`public_submit_reproduce.ipynb` is the separately pulled public final-ensemble
inference notebook, retained as a reference only.

`work_0724v1/` and `work_0803v2/` are isolated executable copies. The only
non-source portability edit is in `seq_NN_pretrained_unet.py`: it permits the
explicit local ConvNeXt checkpoint `convnext_test.safetensors` through
`ROGII_TIMM_CHECKPOINT`, instead of attempting a remote timm download.

## Validation Protocol

All local CV assets use the canonical project fold map:

```text
/root/rogii/OOF/geo_kmeans_5fold.csv
SHA256 ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da
```

The train/validation well counts are `154/155/155/155/154`. Each recipe runs
the original three repeat seeds (`7`, `12`, `17`) as independent five-fold OOF
runs; the final prediction is the row-aligned mean of their OOF predictions.

## Requested No-XY Variant

`run_geo5_repeat.py` defaults to no XY-neighbor feature or label use. It:

1. removes every `geo_*` static channel;
2. restores `z_diff`, the non-spatial channel used by Ruby's GR-only arm;
3. replaces the archived geo-prior builder with a zero, per-query-well
   placeholder required by the Dataset API;
4. never reads neighboring-well TVT values or invokes XY clustering/search.

The active channel lists are persisted in each `repeat_manifest.json`. Passing
`--use-xy-neighbor` is an explicit opt-in and was not used for this experiment.

For `0803v2`, the public cache implementation could retain suffix labels for
diagnostics and re-align PF features with them during validation. The strict
work copy disables both behaviors: PF caches are built with
`include_target_labels=false`, and the loader masks target bins even if an old
cache is selected. The shared strict cache is
`pf_cache_shared_0803v2_noxy_strict/`; it uses only current-well observable
inputs and the fixed `PF_heatmap_base_seed`, then is reused read-only by the
three repeat jobs.

## Commands

Run from the relevant `work_*` directory, with a GPU visible through
`CUDA_VISIBLE_DEVICES`:

```bash
ROGII_TIMM_CHECKPOINT=../convnext_test.safetensors \
python -u run_geo5_repeat.py \
  --cfg cfg.pkl \
  --data-dir /root/rogii/datasets/rogii-wellbore-geology-prediction \
  --fold-map /root/rogii/OOF/geo_kmeans_5fold.csv \
  --output-dir ../runs_noxy_0724v1/repeat0 \
  --repeat 0 --num-workers 16 --pf-workers 1
```

For `0803v2`, first run `--prepare-pf-only --pf-cache-dir <shared-root>`, then
pass the same `--pf-cache-dir <shared-root>` to every training repeat. Merge
the three completed runs with `merge_geo5_repeats.py` in the matching work
directory.

## Status

The local no-XY, canonical-Geo5 runs were launched on 2026-08-07. Results are
written only under `runs_noxy_0724v1/`, `runs_noxy_0803v2/`, and their matching
`final_*` directories; no older OOF cache is used as a training input.
