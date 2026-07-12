# Plan — exp_01_reproduce_flac_table1

**Author:** Fable 5 (plan) · **Code:** none required (existing pipeline only) · **Date:** 2026-07-04

## Goal

Reproduce FLAC Table 1 (unseen AcousticRooms, geometry-conditioned rows) with the released checkpoint:

- **FLAC K=1 G✓:** T60 9.95±0.05 %, C50 1.046±0.002 dB, EDT 40.04±0.22 ms, R@1 6.80, R@5 18.92, R@10 26.87
- **FLAC K=8 G✓:** T60 8.60±0.01 %, C50 0.970±0.002 dB, EDT 37.13±0.02 ms, R@1 6.99, R@5 19.38, R@10 27.21

## Protocol (mirrors the paper exactly)

- **Checkpoint:** `weights/FLAC/FLAC_EMA.ckpt` (EMA weights, loaded via eval_FLAC.py's EMA remapping).
- **Model config:** `src/configs/model_configs/FLAC/AR/FLAC_AR.json` (metrics block uses `AGREE_fullAR.pt` — correct per CLAUDE.md: full-AR AGREE is for evaluation only).
- **Dataset configs (full split per announcement 01 — 6337 items / 17 unseen rooms, verified):**
  - K=8: `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json` (max_context=8, verified)
  - K=1: `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json` (max_context=1, verified)
- **Inference:** `--cfg-scale 1.0 --steps 1` (paper §5.4: guidance 1, single step), `--cond-method vanilla`.
- **5 generations:** paper reports mean±std over 5 generations → 5 runs per K with seeds **42, 43, 44, 45, 46**; report mean±std.
- **Headline numbers = per-scene mean** (paper convention per CLAUDE.md; the script prints both per-scene and overall).
- **GPU:** `CUDA_VISIBLE_DEVICES=0` (GPU 0 is free; GPU 1 is occupied by another job — do not touch).

## Code to write

**None.** This experiment uses only the existing `eval_FLAC.py` path, unchanged. Consequently:
- No Opus 4.8 code-writing step.
- No codex code review (`reproduce_flac_table1_codex_code_review.md` will record "N/A — no code written" for audit completeness).

One known pitfall is avoided by construction: the metrics-JSON path collision found in the code review (same eval-name overwrites) — each run gets a unique `--eval-name exp01_unseen_K{K}_seed{S}`, so all 10 JSONs coexist.

## Execution plan

One background script, sequential runs (single GPU), each run's stdout/stderr teed to a timestamped log in the experiment folder:

```bash
for K in 1 8; do
  cfg=acousticroom_unseeneval_1.json   # K=1
  [ "$K" = 8 ] && cfg=acousticroom_unseeneval.json
  for SEED in 42 43 44 45 46; do
    CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
      --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
      --dataset-config src/configs/dataset_configs/AR/eval/$cfg \
      --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
      --cfg-scale 1.0 --steps 1 --seed $SEED \
      --eval-name exp01_unseen_K${K}_seed${SEED}
  done
done
```

## Deliverables & acceptance

- `reproduce_flac_table1_params_set_up.md`, `_command.md` — written at launch.
- `reproduce_flac_table1_<timestamp>.log` — one per launch.
- `reproduce_flac_table1_results.md` — 10-run table + mean±std per K, side-by-side with Table 1.
- `reproduce_flac_table1_analysis.md` — reliability judgment: **PASS** if our mean is within ~2× the combined std of the paper values per metric (T60/C50/EDT primary; R@k secondary — retrieval is more sensitive to AGREE-checkpoint details); otherwise investigate before any further experiments.

## Risks

- Wall-clock: 10 full-split evals sequentially; if a single run proves very slow, results will land incrementally (JSONs appear per run; results file updated as they do).
- FD is reported in the paper table truncated ("0.") — we record FD_G values but can't compare against the paper's clipped column; R@k comparison covers the AGREE-space behavior.
