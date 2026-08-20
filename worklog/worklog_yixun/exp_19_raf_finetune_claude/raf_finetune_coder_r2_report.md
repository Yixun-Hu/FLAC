# exp_19 r2 Coder final report (Opus 5 subagent, 2026-08-19 ~23:20 EDT) — archived by the Planner

r2 complete: 14/14 Codex findings closed; **333 passed, 0 skipped** (`pytest src/tests/test_raf_*.py`) in worktree `raf-finetune-exp`; metric-consumer suites green individually (exp14 snapshot 32, yaw_random 112, eval_paths 42, cond_dispatch 8, frame_avg_cap 81).

localization-exp era (history of record): R3 `6c0a16e`; R1/R2/R14 `f4d34ca`; R4/R13 `ad966e4`; R5/R6/R12 `6886d40`; R7 `d20c6ab`; R9 `a08571f`; R11 `755aed6`; R8/R10 tests `07badce`; Amdt-3 diagnostic training `c7c3cab`; Amdt-4 render policy `b4c0ac1` (+ ledger rows).
raf-finetune-exp: ports `5f3e4a7`/`2f500d3`/`18c9aa9`, migration `7fd5e5e`, **R8/R10 `5da9c71`** (RAF equal-room macro block; `RT60Error.invalid_stats` additive, read before accumulator reset; `l1_stft_multires` hann window on `x.device`), ledger `f620f09`.

Key implementation decisions (carried to the re-review):
1. R3 schema: constant stays 1; per-stream field bumps to 2 (registered).
2. R8 macro scope: T60/C50/EDT/Env/L1_STFT/L1_STFT_MultiRes macro-averaged over rooms present in ALL; `Invalid T60` deliberately NOT macro-averaged (already a per-item rate; macro would be a third estimand — count and rate reported instead; pinned 2/3 vs the 0.75 macro in tests). AR/HAA regression: no new keys emitted.
3. R7: per-file `os.replace` after whole-set validation; manifest-last attestation.
4. R6: bearing inapplicability recorded, excluded from `passed`; containment/AABB/sightline unconditional.
5. Amdt-4 inpainting: pixel-Euclidean nearest neighbour, no azimuth-seam wrap (documented; immaterial ≤0.1%).
6. R9: only the DINOv3 backbone stubbed (SimpleViT); ids/dims/frequencies/STFT settings are the shipped config.
7. The scratch R8 patch was a sketch; the metric_callback hunk was re-derived by hand against the ported file.
