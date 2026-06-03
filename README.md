# ROGII Wellbore Geology Prediction — CV Infrastructure & EDA

Private working repo for the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
Kaggle competition. Contains **only my own** cross-validation infrastructure and exploratory
analysis — **no competition data and no third-party models/kernels are included** (see below).

## What's here

```
cv/                 CV infrastructure (the core asset)
  cv_runner.py        frozen 3-split CV (well / typewell / spatial), Predictor protocol,
                      row-weighted RMSE + per-well p50/p90/max + worst-by-SSE scoring
  splits/             frozen fold assignments (well_id -> fold) for reproducibility
  level_engine.py     physics-identity drift engine (TVT+Z = formation_top + b_well)
  ring0.py / ring1_anchor.py / level_gbm.py / level_combine.py   level-vs-shape diagnostics
  stage_c*.py         row-level drift-target GBDT harness
  ncc_feat.py / pf_feat.py   multi-scale NCC & particle-filter feature builders
  graft.py / graft_diag.py   graft-onto-baseline experiments
  *_report.md         written-up results for each stage
eda.md, eda-2.md, 比赛定义.md   EDA write-ups and the problem/strategy definition
eda/                EDA scripts, notebooks (outputs cleared), and derived artifact CSVs
```

## NOT included (and why)

- **Competition data** (`datasets/.../{train,test}`) — Kaggle competition data, not redistributable.
  Download it from the competition page.
- **Third-party models / kernels / artifacts** — others' work, not mine to publish.
- **Credentials, large regenerable caches** (`cv/artifacts/*.pkl|*.npz`, etc.) — see `.gitignore`.

## Reproducing

1. Download the competition data into `datasets/rogii-wellbore-geology-prediction/{train,test}`
   plus `sample_submission.csv`.
2. Point the code at the data: `export ROGII_ROOT=$(pwd)` (paths auto-detect; see `cv/cv_runner.py`).
3. `python cv/cv_runner.py` regenerates the frozen splits and runs the baseline ladder
   (carry-forward must reproduce row-weighted RMSE ≈ 15.91 across all three splits).

## Key calibrated scales (row-weighted RMSE)

| | RMSE |
|---|---|
| carry-forward (safe floor) | 15.91 |
| best-per-well constant (oracle, unreachable) | 9.04 |
| smooth-201 (shape ceiling, needs true path) | 0.39 |

The CV is LB-aligned: carry-forward CV 15.91 ≈ public-LB constant 15.88.
