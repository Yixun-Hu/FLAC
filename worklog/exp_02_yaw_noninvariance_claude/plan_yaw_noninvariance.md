# Plan — exp_02_yaw_noninvariance

**Author:** Fable 5 (plan) · **Coder:** Opus 4.8 max (one small script) · **Reviewer:** Codex · **Date:** 2026-07-04

## Goal

Measure, on pristine `0bd5da0` and the full unseen K=1 split (6337 items / 17 rooms), how much yaw-rotating the conditioning hurts FLAC:

- **Metric 2 (accuracy vs GT):** T60/C50/EDT/FD/R@k of P_α vs ground truth, for α ∈ {0, 90, 180, 270}°, against the unrotated baseline.
- **Metric 1 (invariance gap, no GT):** how different P_α is from P_0 (waveform rel-L2 / L1, plus T60/C50/EDT computed with P_0 as reference), from stored predictions.

## Key facts (verified)

- `--rotate-deg` and `src/data/yaw_rotation.py` (`rotate_scene_metadata`, sign self-check) are already in pristine history (commit `dfb849d` "equi test") — **Metric 2 needs no new code**.
- `rotate_deg == 0.0` skips the rotation branch entirely (eval_FLAC.py:109), so the `rot0` run is computationally identical to baseline → it doubles as a determinism check: any difference is run-to-run noise, not rotation.
- Output filenames: metrics JSON gets `_rot<deg>` suffix; predictions `.pt` distinguishes runs via `--eval-name` (distinct names per run → no overwrite).

## Runs (5 total, sequential on GPU 0)

All: FLAC_AR.json model config, `acousticroom_unseeneval_1.json` (K=1), `FLAC_EMA.ckpt`, `--steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed 42 --store_predictions` (per Yixun's spec).

| # | eval-name | --rotate-deg | Role |
|---|---|---|---|
| 1 | yaw_baseline | (absent) | baseline P_0 |
| 2 | yaw_rot0 | 0 | sanity: must equal #1 |
| 3 | yaw_rot90 | 90 | rotated |
| 4 | yaw_rot180 | 180 | rotated (largest expected effect) |
| 5 | yaw_rot270 | 270 | rotated |

## Code to write (the only new code)

**`worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py`** (~100 lines), written by Opus 4.8 max effort, reviewed by Codex:

- Prepends the repo root to `sys.path` before importing `src.*` (known stale-site-packages pitfall on this machine).
- Loads two prediction `.pt` files (`P_ref`, `P_alpha`) plus the eval dataset order-independent pairing (predictions are saved in dataloader order with a fixed seed, so index i pairs across runs — the identical seed/config guarantees identical item order; the rot0-vs-baseline check validates this assumption end-to-end).
- Computes per-sample waveform rel-L2 and mean-|·| between P_alpha and P_ref, and T60/C50/EDT "gap" metrics using `AcousticMetricsCallback` with P_ref in place of GT.
- Writes a JSON + prints a table; CLI: `python compare_predictions.py --ref <pt> --alt <pt> [--out <json>]`.

No repo-source files are touched; the script lives in the experiment folder.

## Acceptance / readout

- Sanity: yaw_rot0 metrics ≡ yaw_baseline (bitwise or to 4 decimals); Metric-1 gap(rot0, baseline) ≈ 0.
- Hypothesis confirmed if T60/EDT/C50 vs GT are worse at 90/180/270 than baseline by ≫ exp_01's noise floor (T60 ±0.04, EDT ±0.37 at K=1), and Metric-1 gaps are large.
- Deliverables per SOP: params/command at launch, timestamped logs, results, analysis, codex review of the script, commits logged.

## Risks

- Disk: 5 × ~250 MB prediction tensors next to the checkpoint (gitignored; kept for reproducibility, can be pruned after analysis).
- If prediction/dataloader order were seed-dependent across processes, Metric 1 pairing would break — caught immediately by the rot0 control (gap would be nonzero even though metrics match).
