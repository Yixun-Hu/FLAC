# Yixun's queries — exp_02_yaw_noninvariance

## Query 1 (2026-07-04)

### Verbatim

> The next exp 02 should be yaw non-invariance for FLAC: ## Reproduce yaw non-invariance
>
> Use `eval_FLAC.py` with `--rotate-deg`. It rotates depth + poses before conditioning; context audio and GT stay fixed.
>
> [baseline command: unseen K=1, seed 42, batch 32, steps 1, cfg 1.0, eval-name yaw_baseline; rotated runs identical + `--rotate-deg` 90/180/270; compare saved metrics JSONs]
>
> **What to look for (Metric 2 — accuracy degradation):** rotated run should be **worse vs GT** than baseline. Example from prior runs on unseen K=1: baseline EDT ~39.3 / T60 ~10.0; rotated 180° EDT ~46.4 / T60 ~12.5. Higher EDT / T60 after rotation → model is **not yaw-invariant**, and rotation **hurts prediction quality**.
>
> **Metric 1 (Pα vs P0, no GT):** current `eval_FLAC.py` does not compute this directly. You would need `--store_predictions` on both runs and compare predictions offline, or restore `worklog/archive_pre_revert_2026-07-04/diagnose_rotation_invariance.py`.
>
> Notes: use `acousticroom_unseeneval_1.json` (unseen, one-shot K=1) — rotation effect is clearest here. Keep `--seed 42` identical across runs. `--rotate-deg 0` must match baseline (sanity check). The assumption is that FLAC is not yaw invariance, the rotation of the input panoroma depth image will damage the performance of FLAC

### Summary

Quantify FLAC's lack of yaw invariance on the frozen released checkpoint: run the full unseen K=1 eval with physically-consistent yaw rotation of the conditioning (depth panorama + poses; context audio and GT unchanged) at α ∈ {0, 90, 180, 270}°, and show (Metric 2) that accuracy vs GT degrades under rotation, plus (Metric 1) that predictions themselves change (Pα vs P0), via `--store_predictions` and an offline comparison.

### Assumption / hypothesis

FLAC is **not** yaw-invariant: a rigid yaw rotation of the whole scene leaves the ground-truth mono RIR unchanged, so an invariant model would predict identically; but FLAC's conditioning stack (DINOv3 over the panorama, Fourier features over raw positions) is orientation-sensitive, so rotating the input panorama/poses will measurably damage prediction quality (prior partial runs: EDT 39.3→46.4 ms, T60 10.0→12.5 % at 180°).

### Why this experiment needs to run

This is the documented, full-split, committed-code version of the motivating observation for the whole equivariance project (the "cylindrical sanity check" FLAC currently fails). exp_01 fixed the baseline; exp_02 fixes the size of the symmetry defect with the same rigor — establishing the reference numbers that exp_03+ (canonicalization fine-tune) must drive to zero (Metric 1) while not regressing the baseline (Metric 2 at α=0). Prior evidence was from pre-revert, partially-subset runs; per announcement 01 and the commit-by-commit discipline, it must be re-established on pristine `0bd5da0` over all 6337 items.
