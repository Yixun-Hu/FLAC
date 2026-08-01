# Lab notebook — exp_10_fa_scratch_resume

## 2026-08-01 — scaffold
- **Goal** — resume exp_07's B-F (fa_invariant from-scratch, SyncBN-64 DDP recipe) from `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt`; all screens under the fa protocol (`--cond-method fa_invariant`), correcting the exp_07 mismatch.
- **Version Control** — branch check-equivariance-necessity, base = exp_09 closure (`93449cb`).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → launch.

## 2026-08-01T14:55:18-04:00 — plan Rev 2 APPROVED by Yixun → implementation round 1 (bf_resume_launch.sh)
- Coder = Opus 5 max seat; guard-tests + Codex review before use; then 15-step probe → launch.

## 2026-08-01T15:15:02-04:00 — round 1 SHIP (26/26 guards) → probe PASS (all lineage asserts; lr 4.903e-5 analytic; 15 fa steps) → COMMITTED RUN LAUNCHED (40k→67.5k, wandb)
- Acceptance criteria (pre-launch): full-state restore @40000, fa path active, 27,500 steps to 67,500, ckpt/2500, no OOM/NaN; screens per 2,500 with `--cond-method fa_invariant`; hard aborts only.
- Monitors armed (ckpt arrivals + death guard). ETA ~2.3 d → screens → R1–R3 readouts.
