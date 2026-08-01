# Plan — exp_10 fa_scratch_resume (B-F resumed from 40k under the correct protocol)

**Author:** main session (Fable 5 seat) · **Date:** 2026-08-01 · **Status:** DRAFT → Codex plan review → Yixun approval.
**Commission (Q1):** "resume B-F from 40k as exp_10."

## 1. Design

**Base:** `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt` — the exp_07 B-F futility stop (full state; fa_invariant from scratch, SyncBN-64 DDP recipe, seed 42). The stop's basis was retracted (eval-protocol artifact); at 40k under its own protocol B-F read 8.190/0.9804/38.811/R5.302 (≈ P1@40k).

**Single arm:** resume under `FLAC_AR_BF.json`, identical recipe (DDP 32/GPU×2×accum1 eff-64, SyncBN-64, ViT grad-ckpt, env `flac`, wandb), warm optimizer (native continuation; exp_09 showed warm-vs-reset immaterial). **Budget: `--max-steps 67500`** — the exp_07 matched budget (primary estimand). Ckpt every 2,500.

**Screens:** every 2,500 from 42.5k, **fa protocol** (`--cond-method fa_invariant`), EMA, K=8 s42 — eval-protocol flag in the launch/screen manifest per the exp_09 SOP lesson.

**Comparators (all fixed in advance):** P1's vanilla curve at matched steps (its own protocol); released Table-1 (exp_01 5-seed); the exp_09 Fw-95000 equivariant numbers (fine-tune route).

## 2. Pre-registered readouts

- **R1 (primary — from-scratch vs fine-tune at matched *total* fa compute is NOT available; the honest primary is):** B-F endpoint/best (composite rule below) vs **P1 at 67.5k** (8.771/0.9734/36.952/6.281 s42; its own protocol) — the matched-budget from-scratch-fa vs from-scratch-vanilla comparison, each under its own eval protocol.
- **R2 (released-parity, secondary):** candidate vs Table-1, σ_c-tiered, 5-seed × K∈{1,8} — same gate machinery as exp_07/09.
- **R3 (equivariance):** C₄ sweep + 45° control at the candidate (metric-level; conditioning-level rel-L2 is architectural, exp_03-established).
- **Candidate rule (fixed):** among B-F ckpts 55k–67.5k, best seed-42 composite = max R@1 among those with T60 ≤ 9.52 ∧ EDT ≤ 40.0 (fa-eval); no qualifier → endpoint 67.5k is the candidate by default (this arm has no anchor-preservation gate — it is a from-scratch trajectory, not a fine-tune).
- **Tiers:** COMPETITIVE (R2 ≥ 6/8 SUPERIOR-or-EQUIV) / VIABLE (R1 within 2σ_c of P1@67.5k on ≥3 of 4) / SHORT (neither; report the measured bound). No stop-and-ask mid-run: hard aborts only; the run is ~4.5 d of fa steps (27.5k × ~0.14 steps/s exclusive — re-anchored at launch).

## 3. Implementation

1. **`bf_resume_launch.sh`** — thin variant of the reviewed `f_arm_launch.sh` family: MODEL_CONFIG fixed to `FLAC_AR_BF.json`, names `FLAC_exp10_BF/exp10_BF/outputs_FLAC/exp10_BF`, RESUME_CKPT required, **EXPECTED_STEP=40000** (this script's own pin — exp_09's ≥87500 floor is exp_09-scoped), MAXSTEPS 67500 default, contract gate (BV/BVp1/BF triangle) + VRAM + wandb + pin gates verbatim. Codex review before use.
2. 15-step resume-validation probe (restored step 40,000, lr == analytic InverseLR(40k) ≈ 4.9e-5, EMA continuity, fa path active) → launch → monitors/screens per exp_07/09 pattern → readouts → close per SOP.

**Risks:** none novel — the machinery is the audited exp_09 kit minus the optimizer-strip path; the one new surface is the EXPECTED_STEP pin change (guard-tested).
