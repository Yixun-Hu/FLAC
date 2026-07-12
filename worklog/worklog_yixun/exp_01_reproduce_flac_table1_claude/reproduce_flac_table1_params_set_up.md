# Params — exp_01_reproduce_flac_table1

| Parameter | Value |
|---|---|
| Checkpoint | `weights/FLAC/FLAC_EMA.ckpt` (released, EMA weights) |
| Model config | `src/configs/model_configs/FLAC/AR/FLAC_AR.json` |
| Dataset config (K=8) | `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json` (max_context=8) |
| Dataset config (K=1) | `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json` (max_context=1) |
| Eval split | `data/AR/unseen_eval.json` — full 6337 items, 17 unseen rooms (announcement 01) |
| cfg-scale | 1.0 |
| steps | 1 |
| cond-method | vanilla |
| rotate-deg | 0.0 (disabled) |
| Seeds (5 generations) | 42, 43, 44, 45, 46 |
| batch-size | 64 (script default) |
| num-workers | 4 (script default) |
| Metrics | T60, C50, EDT, FD_G, R@1/5/10 via `AGREE_fullAR.pt` (training.metrics block of model config) |
| Headline aggregation | per-scene mean (paper convention) |
| GPU | CUDA_VISIBLE_DEVICES=0 (RTX A6000 48GB) |
| Software | torch 2.7.0, pytorch_lightning 2.1.0 (repo pins) |
| Code state | branch `check-equivariance-necessity`, last commit 0bd5da0 + uncommitted diff (eval path for vanilla is unchanged from release behavior) |

## Amendment (2026-07-04 18:28)

Per Yixun's directive ("revert to the latest commit; develop commit by commit, experiment by experiment"), the first launch (18:21:52, on a working tree carrying the uncommitted equivariance-probe diff) was **aborted after ~1 partial run** and the sweep **relaunched on the pristine commit `0bd5da0`** (log: `reproduce_flac_table1_2026-07-04_18:28:50.log`). The aborted log is kept as `..._18:21:52_ABORTED_prerevert.log`; no metrics JSON from the aborted launch is used. The pre-revert code is archived in `worklog/archive_pre_revert_2026-07-04/`.

Code state for all exp_01 results: **commit 0bd5da0, no modifications** (only `.gitignore` differs, which does not affect evaluation).
