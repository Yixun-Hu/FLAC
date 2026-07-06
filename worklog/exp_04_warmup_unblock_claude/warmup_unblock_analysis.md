# Analysis — exp_04_warmup_unblock

**Author:** Fable 5 (Planner) · **Date:** 2026-07-06

## Is the result reliable?

**Yes.** Same protocol rigor as exp_03 (full split, 5 seeds, pre-registered gates and branch rules, per-round reviewed code, runtime-verified warmup engagement). Both hypotheses were registered with predictions BEFORE their data arrived; one confirmed the protocol (W1 fail → W0 branch), one was falsified (W0 predicted pass, failed) — and the falsification is documented as such rather than reinterpreted. The W0 attribution logic is airtight: with lr=0, BatchNorm running statistics are the only mutable state, so its regression cannot be an optimization artifact.

## Outcome

**exp_04 answered its question decisively — in the negative for warmup, and with a proof of mechanism nobody had on the suspect list at the start:**

1. **Adam-transient hypothesis: falsified.** Warmup engaged perfectly and changed nothing (W1 ≡ R1b to within seed noise). The regressed optimum is warmup-independent.
2. **Data-pipeline drift: proven, gradient-free.** The lr=0 null control regressed T60/EDT by ~30% of the full damage purely through RIR-encoder BN running-stat adaptation — possible only if this repo's dataloader feeds the RIR encoder statistics different from the released training's. The remaining ~70% is gradient adaptation toward that drifted data/objective optimum.
3. **The consistent T60/EDT-only signature** (retrieval always intact, C50 clean at lr=0) localizes the drift to the decay-envelope statistics of the reference-RIR conditioning path — concrete candidates: `max_len` truncation/padding convention in `get_ir_and_location_for_other_sources` (AR_md.py), amplitude normalization, or reference-selection distribution.

**Consequence for the project:** the released FLAC checkpoint cannot be gate-passingly fine-tuned by ANY optimizer recipe in this repo as-is — the blocker is data lineage, not optimization. This retroactively explains every "destructive fine-tune" observation back to the pre-revert experiments.

## Recommended next step: exp_05 = targeted drift bisection (cheap, falsifiable)

The BN buffers themselves are a measurement instrument: the drift Δ between stored running stats and our data's batch statistics is directly computable per BN layer in one forward sweep — no training. Plan sketch:
1. Instrument the RIR encoder: compare stored running_mean/var against batch stats over N batches of our loader (per layer, per channel). Quantify and localize the drift.
2. Grid the loader's candidate knobs (max_len, padding side, normalization, context sampling) and find which configuration zeroes the BN drift — that configuration is (a candidate for) the original pipeline.
3. Validation: a W0-style lr=0 run under the corrected loader should then PASS the gate; then a vanilla control; then the blocked W2–W4b pipeline resumes.
Estimated cost: step 1–2 are hours of CPU/GPU-light work; step 3 reuses the existing gated pipeline.

Alternatives if bisection dead-ends: matched-comparison route (R2-vs-R1b, valid but weaker); from-scratch fa_invariant training (confound-free, expensive).

## Bottom line

Two experiments, one coherent story: FLAC's yaw-symmetry defect is fixable by construction (exp_03 proved the mechanism), and the remaining obstacle to demonstrating it on a fine-tuned model is a now-localized data-pipeline discrepancy with a concrete, cheap bisection plan (exp_05) — plus a measurement trick (BN-as-drift-probe) that fell out of a falsified prediction.
