# Plan — exp_10 fa_scratch_resume (B-F resumed from 40k under the correct protocol)

**Author:** main session (Fable 5 seat) · **Rev 2** (2026-08-01; all plan-review findings applied — `fa_scratch_resume_codex_plan_review.md`) · **Status:** AWAITING Yixun approval.
**Commission (Q1):** "resume B-F from 40k as exp_10."

## 1. Design

**Base:** `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt` — the exp_07 B-F futility stop (full state; fa_invariant from scratch, SyncBN-64 DDP recipe, seed 42). The stop's basis was retracted (eval-protocol artifact); at 40k under its own protocol B-F read 8.190/0.9804/38.811/R5.302 (≈ P1@40k).

**Single arm:** resume under `FLAC_AR_BF.json`, identical recipe (DDP 32/GPU×2×accum1 eff-64, SyncBN-64, ViT grad-ckpt, env `flac`, wandb), warm optimizer (native continuation; exp_09 showed warm-vs-reset immaterial). **Budget: `--max-steps 67500`** — the exp_07 matched budget (primary estimand). Ckpt every 2,500.

**Screens:** every 2,500 from 42.5k, **fa protocol** (`--cond-method fa_invariant`), EMA, K=8 s42 — eval-protocol flag in the launch/screen manifest per the exp_09 SOP lesson.

**Comparators (all fixed in advance):** P1's vanilla curve at matched steps (its own protocol); released Table-1 (exp_01 5-seed); the exp_09 Fw-95000 equivariant numbers (fine-tune route).

## 2. Pre-registered readouts (Rev 2 — review-corrected)

- **R1 (primary; FIXED matched-step):** **B-F@67,500 vs P1@67,500, both 5-eval-seed (42–46), K∈{1,8}, each under its own eval protocol.** σ_c = √(σ_BF² + σ_P1²) per cell; P1's 5-seed endpoint stats computed as part of this readout (its gate JSONs exist for 67.5k). Framing: **matched steps/samples, NOT matched compute** (fa ≈ 3.5× step cost); and an **end-to-end total-arm comparison** (training + fa inference bundled — per the exp_07 amendment, not a training-only attribution).
- **R1b (exploratory, separate):** window-best comparisons at matched steps (B-F window stat vs P1 window stat, same steps), labeled exploratory.
- **R2 (released-parity, secondary):** the R1 candidate vs Table-1, σ_c-tiered, 5-seed × K∈{1,8}.
- **R3 (equivariance):** C₄ sweep + 45° control at the candidate (metric-level; conditioning-level invariance architectural).
- **Candidate rule (fixed, tightened):** ALL screened ckpts 42.5k–67.5k eligible; qualifier = seed-42 fa-eval **T60 ≤ 8.61 ∧ C50 ≤ 0.985 ∧ EDT ≤ 38.9** (≈ the 40k point + eval-noise margin — the trajectory must not regress); candidate = max R@1 among qualifiers, 5-seed-confirmed. **No qualifier → NO confirmatory candidate**: the fixed endpoint (67.5k) is reported as the primary R1 row only, explicitly labeled non-selected.
- **Tiers (bounded):** COMPETITIVE = R2 ≥6/8 SUPERIOR-or-EQUIV **and no cell >3σ_c OUT**; VIABLE = R1 within 2σ_c of P1 on ≥3/4 **and the 4th ≤4σ_c**; SHORT = otherwise (report the measured bound). Hard aborts only.
- **Extension (pre-registered OPTION, separately approved):** if the 67.5k verdict lands VIABLE-or-better, an optional phase 2 continues 67.5k→100k (same cadence/selector applied to 70k–100k) — it CANNOT overwrite the 67.5k primary verdict; motivation: P1's best was 87.5k, B-V's R@1-best 92.5k. Yixun approves phase 2 explicitly if offered.
- **Wall-clock:** 27.5k steps ≈ 2.3 d at ~0.14 steps/s + screens ≈ **~2.5–3 d total** (re-anchored at launch).

## 3. Implementation

1. **`bf_resume_launch.sh`** — thin variant of the reviewed `f_arm_launch.sh` family. KEEP: env/PL asserts, full-state lineage check, contract gate (BV/BVp1/BF triangle), VRAM, wandb identity, DINOv3 pin gates. DELETE (exp_09-scoped, must NOT carry over): OPT_RESET*/RESET_LINEAGE, Fw/Fr/V identities, the BF/BVp1 allow-list, optimizer stripping, the ≥87,500 EXPECTED_STEP floor. ADD (review Blocker 2): initial-launch lineage = exact B-F-40k path + **embedded model_config == FLAC_AR_BF.json** + **SHA-256 pin of the 40k ckpt**; restart path = EXPECTED_STEP>40000 allowed ONLY for ckpts inside `outputs_FLAC/exp10_BF/` with MAXSTEPS>EXPECTED_STEP. Names `FLAC_exp10_BF/exp10_BF/outputs_FLAC/exp10_BF`; MAXSTEPS default 67,500. Guard-tests + Codex review before use.
2. 15-step resume-validation probe (restored step 40,000, lr == analytic InverseLR(40k) ≈ 4.9e-5, EMA continuity, fa path active) → launch → monitors/screens per exp_07/09 pattern → readouts → close per SOP.

**Risks:** none novel (audited kit minus exp_09-specific paths); not bit-exact across the resume boundary (PL restores no RNG/dataloader position; 249 unsaved exp_07 steps discarded — disclosed, per the stop record).
