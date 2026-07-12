# Analysis — exp_03_fa_invariant_cond

**Author:** Fable 5 (Planner) · **Date:** 2026-07-05

## Is the result reliable?

**Yes — high confidence, including the negative part.**

1. **Code:** six TDD rounds + plan review + integrative review, all closed through write→review→fix→re-verify; 83 tests green; every review produced at least one load-bearing fix (degenerate-fallback invariance, BN side effects, stale-depth test hole, grad-clip recipe drift, stray-checkpoint leak, autocast dtype alignment, load-integrity assertion). Reviewer: Codex gpt-5.5 xhigh, context-briefed.
2. **Physics/infrastructure claims are measured, not asserted:** conditioning invariance 4.9e-8 relative on the real DINOv3 stack; determinism control exactly 0; the end-to-end floor traced stage-by-stage to VAE-decoder amplification (×~1200) of float-level dust; pre-registered before R4 would have been read.
3. **The gate failures are trustworthy:** full 6337-item split, 5 seeds, exp_01 protocol byte-compatible (vanilla path pinned by tests), clean-load asserted, and the failure replicated across two independently-designed control recipes with a coherent dose-response (batch parity recovered ~35–40%).
4. **Honest caveats:** the first online-weights diagnostic was confounded (eval loader auto-remaps EMA keys) — caught and corrected, both runs recorded; the RIR-encoder BN and padding-mask micro-batching caveats from the accum review apply equally to R1/R1b and cannot explain a 6–42σ effect.

## Outcome

**Minimum project goal (cylindrical sanity check): effectively achieved at the conditioning level, with proof.** `fa_invariant` makes FLAC's entire conditioning path exactly yaw-invariant on C₄ (and the pose path at any angle) without touching DINOv3 — H1 evidence: float-exact conditioning, end-to-end frozen-model invariance at the decoder noise floor, 200–400× below the exp_02 defect. What H1 lacks is only ceremonial: R4's full-split Metric-1 table on a *fine-tuned* model.

**H3 (accuracy non-regression through fine-tuning): BLOCKED — and the blocker is a publishable finding in itself.** The released FLAC checkpoint cannot be non-destructively fine-tuned on its own training data with its own audited recipe: vanilla controls (never touching fa_invariant) regress T60/EDT by 6–62σ while *fully preserving* AGREE-space retrieval. This cleanly re-interprets the pre-revert "inconclusive" FA experiment: the confound was never frame averaging — fine-tuning itself is destructive at reproducible-recipe scale.

**Mechanism accounting:** EMA-vs-online ≤15%; effective-batch noise ~35–40% (measured by the R1→R1b delta); lr excluded (5e-6 is 8× below the original's final InverseLR value). **Leading unfalsified suspects:** (a) fresh Adam second-moment transient — the released artifact has no optimizer state, and bias-corrected steps in the first ~100s of steps take outsized parameter-relative moves at any lr; (b) drift between the code that trained the released checkpoint and this repo's training path (the config parity audit covers the config, not the code lineage). The T60/EDT-specific damage with intact retrieval suggests the energy-decay envelope is the sharpest direction of the loss landscape.

## Recommended next steps (in order)

1. **exp_04a (cheap, decisive):** Adam-transient test — same R1b recipe + linear lr warmup 0→5e-6 over ~200 steps, optionally β₂ 0.95 for the first steps. If the control then passes, the entire H3 path unblocks for ~3 h of GPU. This is the highest-information-per-GPU-hour move available.
2. **exp_04b (if 04a fails):** code-lineage audit — bisect the training path against the released checkpoint's provenance (loss/timestep-sampler/data-pipeline versions), possibly contacting the release authors.
3. **R2-vs-R1b matched comparison (optional, ~13 h):** even under a failing gate, fine-tuning *both* arms identically and comparing R2 against R1b (not exp_01) isolates fa_invariant's marginal effect at matched regression — and R4 on that R2 would still yield the full-split Metric-1 ≡ 0 table for H1. Scientifically valid, but weaker than fixing the recipe first.
4. **Fallback for the maximum goal:** if fine-tuning stays blocked, from-scratch training with `fa_invariant` (the infrastructure is ready and reviewed) is the clean path — expensive but confound-free.

## Bottom line

exp_03 delivers: a proven, reviewed, exactly-invariant conditioning mechanism (Route 1 works as designed); a decisive negative result about fine-tuning the released checkpoint that redefines the project's critical path; and a cheap, sharp next experiment (warmup transient test) that decides the H3 route.
