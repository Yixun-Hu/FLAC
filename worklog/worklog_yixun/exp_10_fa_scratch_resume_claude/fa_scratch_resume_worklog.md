# Lab notebook — exp_10_fa_scratch_resume

## 2026-08-01 — scaffold
- **Goal** — resume exp_07's B-F (fa_invariant from-scratch, SyncBN-64 DDP recipe) from `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt`; all screens under the fa protocol (`--cond-method fa_invariant`), correcting the exp_07 mismatch.
- **Version Control** — branch check-equivariance-necessity, base = exp_09 closure (`93449cb`).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → launch.

## 2026-08-01T14:55:18-04:00 — plan Rev 2 APPROVED by Yixun → implementation round 1 (bf_resume_launch.sh)
- Coder = Opus 5 max seat; guard-tests + Codex review before use; then 15-step probe → launch.
