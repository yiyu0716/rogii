# Handoff: Ruby 0724 V1 / 0803 V2 strict Geo5 no-XY reproduction

## Requested scope

Reproduce only the public Kaggle datasets:

- `w5833946/rogii-0724-v1-cv486`
- `w5833946/rogii-0803-v2-cv500`

using `/root/rogii/OOF/geo_kmeans_5fold.csv`, SHA256
`ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da`,
with XY-neighbor features disabled by default and local GPU training.

## Downloaded public code and inference entry point

- Untouched `0724` source/config/OOf dump:
  `source_0724v1/`
- Untouched `0803` source/config/OOf dump:
  `source_0803v2/`
- The public deployment entry point in each source dump is
  `seq_NN_main.py --submit-mode`; it loads a saved `cfg.pkl` and `models.pkl`.
- Public inference notebook referenced by the author:
  `https://www.kaggle.com/code/w5833946/submit-reproduce`
- A local copy of that public notebook is retained as
  `public_submit_reproduce.ipynb` (SHA256
  `1d3b5e8c092d25f57a800e54276b5d497f79623b4d16469182fc91c89046b252`).

No Kaggle submission has been created by this reproduction.

## Work copies and intentional changes

`work_0724v1/` and `work_0803v2/` contain only the following intentional
reproduction changes:

1. `run_geo5_repeat.py` replaces the author split generator with the canonical
   fixed Geo5 map. It asserts one assignment per train well and logs the map
   hash/counts.
2. Its default mode removes all `geo_*` channels, restores `z_diff`, and
   replaces the XY neighbor-prior builder with all-zero per-query placeholders.
   Thus no neighbor TVT labels, neighbor search, or XY clustering is used.
   `--use-xy-neighbor` is opt-in and was not passed.
3. `seq_NN_pretrained_unet.py` accepts the local ConvNeXt checkpoint through
   `ROGII_TIMM_CHECKPOINT`; no model architecture or trainable parameter is
   changed.

## Critical 0803 OOF audit finding

The public `0803v2` PF implementation is **not strict OOF as published**:

- `seq_NN_data_prep.py` could serialize suffix `TVT` into the train PF cache.
- `seq_NN_dataset.py:_make_pf_unet_features()` then used
  `common["tvt_target"]` to shift PF features when the sample has a target.
  Validation loaders have targets, so this is a direct validation-label feature
  path.

The strict work copy removes it:

- PF cache config has `include_target_labels=false`;
- the cache builder is passed `has_target=false` even for train wells;
- cache loading masks any stored target bins defensively;
- `PF_allow_target_tvt_feature=false` prevents target-based PF re-alignment.

This means the final strict `0803v2` CV is an honest **no-XY, no-target-PF
ablation**, not the author-reported 5.00 result.

## Active work

### 0724 V1

Three independent repeat runs have started, one on each currently idle GPU:

| repeat | seed | GPU | output |
|---:|---:|---:|---|
| 0 | 7 | 1 | `runs_noxy_0724v1/repeat0/` |
| 1 | 12 | 2 | `runs_noxy_0724v1/repeat1/` |
| 2 | 17 | 3 | `runs_noxy_0724v1/repeat2/` |

The source recipe has 300 epochs and 38 batches per epoch. On the local PPUs it
runs at roughly 37–38 seconds per epoch, so each five-fold repeat is expected
to require about 15 hours. Do not change batch size or epoch count if the goal
is a source-faithful no-XY comparison.

The command template is:

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
ROGII_TIMM_CHECKPOINT=/root/rogii/OOF/ruby_repro_geo5_0724v1_0803v2_20260807/convnext_test.safetensors \
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/.venv/bin/python -u run_geo5_repeat.py \
  --cfg cfg.pkl \
  --data-dir /root/rogii/datasets/rogii-wellbore-geology-prediction \
  --fold-map /root/rogii/OOF/geo_kmeans_5fold.csv \
  --output-dir ../runs_noxy_0724v1/repeat<0|1|2> \
  --repeat <0|1|2> --num-workers 16 --pf-workers 1
```

### 0803 V2

Strict PF precomputation is active:

```text
pf_cache_shared_0803v2_noxy_strict/train/
```

It is built once with 32 CPU workers and must complete all 773 training wells
before the three model repeat jobs start. The cache is independent of the
network seed and is safe to share once it is complete, because its PF state
uses only `MD`, `Z`, `GR`, visible `TVT_input`, typewell data and fixed random
seeds. The strict cache has its own digest `91559d17760b`.

Use the exact same run command as above from `work_0803v2/`, adding:

```bash
--pf-cache-dir /root/rogii/OOF/ruby_repro_geo5_0724v1_0803v2_20260807/pf_cache_shared_0803v2_noxy_strict
```

Do not use any earlier `pf_cache_*_v2` directory: those pre-date the strict
target-label removal.

## Final merge after runs complete

For each version, from its `work_*` directory:

```bash
/root/.venv/bin/python merge_geo5_repeats.py \
  --cfg cfg.pkl \
  --data-dir /root/rogii/datasets/rogii-wellbore-geology-prediction \
  --fold-map /root/rogii/OOF/geo_kmeans_5fold.csv \
  --repeat-root ../runs_noxy_<version> \
  --output-dir ../final_noxy_<version>
```

The `SUMMARY.json` written by the merge is the only final CV to report. Before
then, individual epoch validation RMSE values are training diagnostics, not
OOF CV.

An active coordinator logs to `logs/queue_0803v2_strict.log`. It waits for the
strict 0803 PF cache and all 0724 jobs, writes `final_noxy_0724v1/`, then
starts the three strict 0803 jobs and writes `final_noxy_0803v2/`. It does not
submit to Kaggle.
