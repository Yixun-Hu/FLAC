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

## 2026-08-04T11:21:39-04:00 — DECOMPOSITION CELL (vanilla P1@40k under fa eval, 5-seed both K): ensembling confound REFUTED — the invariance effect is training-side
- P1@40k + fa-eval: K8 8.817/1.0009/42.283/R4.049 · K1 10.257/1.0824/45.436/R3.926 — test-time 4-view averaging DEGRADES a vanilla model (R@1 −1.12, EDT +1.63 at K8 vs its own protocol; the model never learned averaged conditioning).
- Same-protocol matched-step comparison: fa-TRAINED 8.202/0.9778/38.79/R5.39 beats vanilla+fa-eval on all 12 cells (T60 −0.62, EDT −3.49, R@1 +1.34 at K8) → **fa's advantage is TRAINING-side invariance, not inference ensembling; "invariance makes the difference" licensed at matched steps (one training seed; 40k point).**
