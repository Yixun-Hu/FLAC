# Plan — exp_06_gradpath_bisect (why T60/EDT don't recover; the lr axis; gradient-path lineage)

**Author:** Fable 5 (Planner) · **Coder:** Opus 4.8 max (S2 is CODE-BEARING: L5 needs a tested --lr-schedule selector) · **Reviewer:** Codex gpt-5.5 xhigh · **Date:** 2026-07-06
**Status:** AWAITING plan review + Yixun approval.

## 0. Evidence baseline (exp_03/04/05)

T60 damage: gradient-driven, BN-independent (R1b 10.47 ≈ V1′ 10.52 vs baseline 9.97 at K=1; K=8 9.20/9.23 vs 8.61). EDT residual after freeze-bn: +1.4 ms. Falsified: Adam transient, EMA, batch noise (sole), BN mutation (sole), max_len. Never tested: **the lr axis itself** (all fine-tunes ran 5e-6 constant — a deliberate but untested deviation from the original lr 5e-5 + InverseLR).

## S1 — damage dynamics from EXISTING checkpoints (zero training cost; runs first)

R1b and V1′ saved interval checkpoints at optimizer steps 200/400 (+ final 625). Eval T60/C50/EDT at each step: **K=8, seed 42, full split** (K=8 chosen for screening: its T60 z-scores were 38–57σ — a single seed resolves differences 30× smaller than the effect; announcement 01 honored — always the full 6337-item split).

Readout (pre-registered discrimination):
- **Immediate jump** (damage ≥80% complete by step 200): objective/data mismatch at init — the gradient at the released weights points away from the T60 optimum. → S2-lr unlikely to fix; S3 (objective lineage) becomes primary.
- **Monotone accumulation** (roughly linear in steps): drift under a mismatched objective; lr changes rescale but don't redirect. → S2 informative mainly via the near-zero-lr arm.
- **Saturation to a plateau**: convergence toward a different optimum (consistent with R1b≡W1); lr sets speed, not destination — unless the plateau LEVEL is lr-dependent (S2 tests exactly this).

## S2 — the lr axis (Yixun's hypothesis; screening then confirmation)

Five fine-tune arms, all otherwise identical to the best-known recipe (batch 4×32=128, 625 opt steps, freeze-bn ON, use_ema off, seed 42):

| Arm | lr setting | Rationale |
|---|---|---|
| L1 | 5e-7 constant | 10× lower: if damage shrinks ∝ lr, it's drift-speed, not destination |
| L2 | 5e-6 constant | = R1b/V1′ (already have it — no rerun; anchor point) |
| L3 | 2e-5 constant | mid |
| L4 | ~4.2e-5 constant | **high-LR continuation proxy** (InverseLR end-value is 4.23e-5 at 400k steps but 3.54e-5 at 1M; release step count unknown — L4 is a proxy, not proven faithful) |
| L5 | 5e-5 + original InverseLR restart | pre-revert config + freeze-bn + batch parity. **Requires code (review finding 1):** `--lr-schedule {constant, inverse-restart}` in finetune_cond.py, TDD (parser, config injection preserving InverseLR(1e6, 0.5, 0.99), configure_optimizers returns the scheduler branch, recipe echo/lr trace), per-round review, smoke with lr-trace before launch |

**Screening protocol (pre-registered):** each arm evaluated K=8, seed 42, full split (1 eval ≈ 15 min). Screening is for ORDERING/shape only, never headline numbers. Finalist criteria pre-registered on BOTH metrics Yixun asked about: K=8 T60 ≤ ~8.65 OR K=8 EDT ≤ ~37.3 (3× single-eval band) → full 5-seed × K∈{1,8} gate protocol. Wall-clock: 4 new fine-tunes ≈ 3 h each + 4 screening evals ≈ 13 h sequential.

Interpretation (wording per review finding 3): if NO arm moves the plateau, the conclusion is 'no tested lr setting recovers the gate' (not 'lr has no effect') and Yixun's question is answered definitively (with the schedule-faithful arm included); if some arm recovers T60, exp_03's recipe assumption is overturned and the H3 pipeline reopens with that recipe.

## S3 — objective/data lineage audit (CPU; parallel with S2 training)

1. **Upstream diff (target identified by review):** fetch `github.com/AmandineBtto/FLAC` (official codebase per the HF model card) and diff `src/training/`, `src/data/`, `src/models/`, `src/inference/` against this fork's `0bd5da0` base. Any delta in loss assembly, timestep sampling, padding-mask semantics, augmentation defaults = a concrete lineage candidate.
2. **REVISED mechanism probes (review finding 2 — padding masks are near-absent: 5,533/291,210 train and 0/6,337 eval targets shorter than the 10,240 window, so tail masking cannot be the mechanism):** (a) target truncation — RIRs LONGER than 10,240 are cropped: measure the truncated-energy fraction and whether T60-relevant decay extends past the window; (b) metric-window coverage — the metric callback computes T60 over max 8,000 samples: quantify sensitivity of T60/EDT to the region the loss sees vs the metric sees; (c) augmentation provenance — Random Time Shift + Add Noise apply to the TARGET audio: quantify their effect on the effective target distribution (shifted/noised tails) vs the un-augmented eval targets — a train/eval target-distribution gap the released training may or may not share.
3. **Augmentation off arm (conditional):** one 200-step fine-tune with augmentations disabled + screening eval, ONLY if S3.2(c) shows a material target-distribution gap.

## Deliverables & stop rules

- S1 curves (checkpoints confirmed at steps 200/400/600 for both R1b and V1′; clean load verified by review — no partial-load flag; caveat: a full jump by step 200 cannot distinguish first-step events from fast 0–200 drift) + S2 lr-response table + S3 findings → analysis answering both of Yixun's questions: (i) the mechanism of non-recovery (dynamics + tail-loss evidence), (ii) whether lr matters (five-arm response incl. schedule-faithful).
- Source code: exactly one TDD round (--lr-schedule for L5); S1 uses existing flags (S1/S2 otherwise; S3.2 is a one-off worklog probe script like exp_05's dispersion check). If any arm passes screening, its confirmation run follows the standard full gate protocol before any claim.
- Stop: after S1+S2+S3 regardless of outcome; the analysis feeds the A-vs-B decision (matched comparison vs from-scratch) with the lr question closed.

## Risks

- The release step-count for L4's "schedule-end" lr is an assumption (~4×10⁵); L5 (schedule restart) brackets it from the other side.
- 5 sequential fine-tunes ≈ 13 h GPU; screening-eval shortcut is pre-registered to avoid 50 h of full protocols on arms that will be excluded.
- S3.1 depends on the upstream repo being locatable; if not, the lineage audit is limited to internal consistency (git history of this fork).
