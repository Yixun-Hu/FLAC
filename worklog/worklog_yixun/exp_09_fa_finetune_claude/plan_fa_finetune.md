# Plan — exp_09 fa_finetune (fa equivariant fine-tune from the 87.5k full-parity anchor)

**Author:** main session (Fable 5 seat) · **Date:** 2026-07-29 · **Status:** DRAFT → Codex plan review → Yixun approval before implementation.
**Commission (Q1):** "the fa equivariant fine-tune from the new 87.5k anchor, go ahead."

## 1. Design

**Base:** `outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt` — vanilla, 5-seed superior-or-equiv to released Table-1 at both K, **full training state (optimizer+EMA+loop)** — the state-completeness the released ckpt lacked (exp_03–06's fine-tune-damage context).

**Two arms, same launch machinery as exp_07 (all reviewed/SHIP'd, zero new training code expected):**
- **F (treatment):** fa_invariant fine-tune — resume the anchor via `--ckpt-path` under `FLAC_AR_BF.json` (cond_method `fa_invariant`, frame_avg_angles [0,90,180,270], grad-ckpt keys; state_dict-compatible by the proven arm-identity). Warm optimizer carries over; conditioning path switches at the resume boundary.
- **V (control):** continued vanilla — identical resume under `FLAC_AR_BVp1.json`. Separates fa-adaptation effects from continued-training drift at matched steps.

**Recipe (both arms):** exp_07 P1 recipe verbatim — DDP 32/GPU×2×accum1 eff-64, SyncBN(64), ViT grad-ckpt, seed 42, ckpt/2500, env `flac`, wandb; `p1_ddp_launch.sh`-style launches (BF variant needs a MODEL_CONFIG parameter or a thin `f_arm_launch.sh` mirror — the only new code, shell-only, reviewed before use).

**Budget:** `--max-steps 97500` (= anchor 87.5k + **10k fine-tune steps**), both arms. Rationale: exp_08's fa adaptation showed effects within ~2–5k steps; 10k gives margin; F-arm ≈ 1.5 d (fa ≈ 0.074–0.1 steps/s), V-arm ≈ 11 h (0.26). Sequential on the pair of GPUs: F first, then V.

**LR:** unchanged schedule (InverseLR at 87.5k ≈ 4.8e-5). exp_06 showed damage monotone in lr — but that was cold-optimizer from the released ckpt; here the warm state + native schedule IS the point. A reduced-lr arm is pre-registered as a CONDITIONAL follow-up only if F degrades error metrics >2σ_c vs the anchor.

## 2. Pre-registered gates (all 5-seed where gated; full 6,337/17 split; eval env flac)

Anchor baselines (the numbers to defend): K=8 8.2929±0.0105 / 0.9660±0.0015 / 35.9513±0.0532 / 6.9591±0.1353; K=1 9.5401±0.0231 / 1.0323±0.0060 / 38.7283±0.2263 / 6.8108±0.1766.

- **G1 (equivariance, exp_08's H-A2/A3):** conditioning-level C₄ invariance exact (rel-L2 ≤1e-6 machine-level) and generation-level rotation sweep flat (H1/H2 protocol) on the F endpoint.
- **G2 (no-damage, primary):** F endpoint (best-of-curve by the composite rule over the 10k window, seed-42 selection → seeds 43–46 confirm) within ≤2σ_c of the ANCHOR on all four metrics at both K; ≤1σ_c = clean pass.
- **G3 (released-superiority retention):** same F checkpoint still SUPERIOR-or-EQUIV vs released Table-1 in ≥7/8 cells.
- **G4 (control separation):** V at matched steps — F-vs-V differences attribute fa effects; V also guards against "the anchor was a lucky peak" (if V drifts off the anchor's numbers, the band, not the peak, is the fair reference).
- Screens: every 2,500 steps, both arms (EMA, K=8 s42), F additionally K=1 at the candidate.

## 3. Success tiers

- **FULL:** G1 + G2(≤1σ_c) + G3 → "yaw-equivariant FLAC at released-Table-1 parity or better" — the complete project narrative.
- **PARTIAL:** G1 + G2(≤2σ_c) → equivariance with bounded cost; conditional lr-reduced arm triggers.
- **NEGATIVE:** G1 fails (machinery regression — investigate) or G2 >2σ_c (fine-tune damage persists even warm-state → the exp_06 conclusion generalizes; report and stop).

## 4. Steps & artifacts

1. Codex plan review → revise → **Yixun approval** (gate).
2. Thin launch shell (`f_arm_launch.sh`, MODEL_CONFIG-parameterized mirror of `p1_ddp_launch.sh` with the contract gate pointed at the arm's config) → Codex review → commit.
3. Launch F (resume anchor, BF config, 97.5k). Screens/monitors per exp_07 pattern.
4. Launch V (same, BVp1 config). 
5. G1 equivariance block (exp_08 machinery) on the F candidate; G2/G3 5-seed gates; G4 comparison.
6. Results/analysis/HTML/closure review per SOP.

**Risks:** resume-with-different-training-config could surprise PL (mitigate: the fail-fast first minutes + arm-identity precedent); fa step-cost 3.5× (budgeted); oscillation vs 10k window (composite selection + confirm handles).
