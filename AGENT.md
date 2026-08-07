# ROGII Project Principles

## Current Baseline

Use `full249` as the current feature/model baseline unless a newer handoff explicitly replaces it.

## Canonical 5-Fold Split

Use the following fixed fold map for current model training, OOF generation,
ablation comparison, and deployable five-fold assets:

```text
/root/rogii/OOF/geo_kmeans_5fold.csv
```

Its SHA256 is:

```text
ac4351cedb0f0a70edabf95308aafc73770c1190ca4d3e68c618f6b4600d64da
```

Treat this as the default `geo-stratified known-domain` protocol. The
54-group physical-typewell fold remains a secondary unseen-domain stress
diagnostic only; do not use it as the default training split or silently mix
OOF/base/sidecar assets produced under the two protocols. Reports must state
which fold map generated every learned asset.

## Canonical Training Asset Root

All assets trained or generated from now on must be written under:

```text
/root/rogii/OOF/
```

This includes feature caches, fold-specific datasets, OOF predictions,
sidecars, selector models, model weights, checkpoints, logs, diagnostics, and
submission asset staging directories. Use a new experiment-specific
subdirectory for every run; never overwrite or implicitly reuse an older run.

Legacy assets outside `/root/rogii/OOF/` may be read only when an experiment
explicitly declares them as fixed inputs and verifies their fold protocol and
train/inference semantics. Do not fall back to an old cache merely because a
new cache is missing. Every experiment report must record its output root and
the exact paths and checksums of any legacy inputs.

## Kaggle Submission Asset Ownership

Every Kaggle submission must use the locally reproduced, auditable assets
under `/root/rogii/OOF/` (uploaded under `yiyu0716` when Kernel deployment is
required). Do not depend on teammate-private datasets, teammate Kernel output
CSVs, or teammate-only checkpoints. A teammate recipe may be reproduced, but
the submission report must identify the exact local reproduction and its OOF
CV rather than describing it as the teammate's exact asset.

Before every multi-arm or routed Kaggle submission, audit reusable inference
work. Generate semantically identical WARP/HMM/GSN/STRIDE paths once, share
fold-invariant feature blocks by ID, and build common fold frames once for
multiple prediction heads. Keep fold-specific imputers and other fold-derived
state isolated; runtime reuse must preserve prediction semantics.

## Evidence Design Rules

New features and candidate-path modules should prioritize capability gaps over small tuning changes. Any proposed evidence should be judged by the following four rules.

### 1. Harder Evidence

Evidence should help exclude wrong candidates, not merely express a vague preference.

For candidate paths A and B, the feature should explain:

- why candidate A can explain a verifiable fact;
- why candidate B cannot explain that same verifiable fact.

Prefer falsifiable quantities such as replay error, likelihood ratio, violation count, physical infeasibility score, formation-order break, or segment-transition inconsistency. Avoid treating small GR/PF score differences as hard evidence by themselves.

### 2. More Independent Evidence

Evidence should not just be another rewrite of a GR/PF score.

Prefer more orthogonal facts from different evidence layers, such as visible-prefix replay, `U = TVT + Z` geology-path continuity, formation-sequence consistency, trajectory continuity, segment-level mode stability, and candidate-family reliability.

For every new feature, ask:

> Does this path follow the geological or trajectory behavior already exposed by this well?

If the answer is only "its GR residual is lower", the evidence is not independent enough.

### 3. More Candidate-Specific Evidence

Evidence should distinguish candidates A, B, and C, not only describe whether the whole well is difficult.

Useful evidence should help answer:

- does top1 truly beat top2;
- which cluster or candidate family is reliable on this well or segment;
- when should the system trust a mode switch;
- when should it avoid a hard selection and use posterior mean or hedge.

Well-level uncertainty features are allowed, but they are risk diagnostics, not candidate-selection evidence by themselves.

### 4. More Private-Safe Evidence

Evidence must be computable on the real private test set, and training must not secretly use hidden truth.

Required constraints:

- use only test-visible inputs at inference time;
- do not use hidden true TVT, hidden formation truth, oracle best cluster, or public-LB-specific overwrite logic;
- use OOF predictions whenever a learned base/model output is used during training;
- keep train/test feature semantics identical;
- prefer same-well visible-prefix replay or train-derived priors that are valid for private test.

If a feature improves CV by using information unavailable on private test, treat it as leakage and do not promote it to the main pipeline.
