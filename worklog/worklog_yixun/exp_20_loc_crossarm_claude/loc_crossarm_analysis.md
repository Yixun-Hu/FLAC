# loc_crossarm_analysis — exp_20 (Planner: Claude Fable 5, 2026-08-25; CLOSED by Yixun's sign-off)

## Answer to the registered question
*Do the equivariant (BF FA) or augmented (YAW) arms carry more invertible source-position information than matched-step vanilla (P1), under the frozen exp_18 analysis-by-synthesis protocol?*

**Confirmed for equivariance, in both context regimes.** The registered Holm-4 family (top-1 sole confirmatory endpoint): BF > P1 at K_ctx=8 (+0.0182, p_adj 0.0008) and at K_ctx=1 (+0.0155, p_adj 0.0018); YAW > P1 positive in both regimes (+0.0079/+0.0068, raw p < 0.05) but not surviving Holm (p_adj 0.0596) — a suggestive, unconfirmed effect ≈ 40% of the FA effect. Ordering BF > YAW > P1 in all 18 cells; per-seed replicates within ±0.002.

## Reliability — HIGH
Machine-verified registrations (freeze `a92ff5d`; source-blob pins; per-query FA-partition end-gate); exp_15-grade checkpoint admission (EMA 210/210, step + embedded-config equality); pairing PROVEN per (regime, seed) across arms (identity streams, context digests, noise keys — common random numbers, so contrasts are genuinely paired); identity gates 6,337/6,337 × 18; announcement-05/06 FA discipline enforced in code (per-angle plan locked and verified per query; the r1 TDD round caught the driver silently sharing all four angles in one forward — precisely the announcement-06 drift class — before anything ran); fa-parity bitwise in BOTH autocast modes.

## Mechanistic notes
1. **The FA advantage composes with context-invariance** — near-equal effect at K=8 and K=1 rules out an elimination/coverage interaction; the equivariant conditioning yields generated RIRs that are more position-discriminative under AGREE, full stop.
2. **BF also reduces the context-imitation failure mode** (0.366 vs P1's 0.386 at K8) and mean error (−0.04 m) — consistent small improvements across every readout, not a single-metric artifact.
3. **Program coherence:** the training-side story (exact C₄ equivariance > augmentation, exp_07–15) transports to a downstream capability those experiments never measured. Notably, exp_11 found the FA *training* advantage recipe-contingent; exp_20's arms share one recipe family (2×A6000 lineage), so this is a within-recipe comparison — the safest form of the claim.

## Threats to validity (stated)
Single historical training run per arm (M6 caveat — arm effects are conditional on these runs; no replicated-training causal claim); AGREE scorer only (exp_18's R4 showed scorer choice moves absolute levels — the saved dumps + inline metrics allow an m2-scored replica offline if wanted); discrete M=10 candidates (the continuous version is exp_22's question); YAW = exp_17's arm (recipe-matched; exp_15's cluster arm would be a cross-recipe sensitivity only).

## Recommendations
1. exp_22 mesh-grid: P1-first (approved) — then the BF arm is the high-value addition given today's confirmation.
2. Optional offline: m2-scored exp_20 cells from the saved metrics (no GPU) — tests whether the FA edge persists under the waveform scorer.
3. Paper: the two-experiment story is complete — exp_18 (inversion works; two regimes; scorer matters) + exp_20 (equivariance confirmed better; augmentation suggestive).
