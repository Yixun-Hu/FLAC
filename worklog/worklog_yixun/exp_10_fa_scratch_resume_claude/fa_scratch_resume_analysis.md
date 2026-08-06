# Analysis — exp_10 fa_scratch_resume (closing)

**Author:** main session (Fable 5 seat; session-alternation caveat per `issue_report.md` §8) · **Date:** 2026-08-06

## Outcome

Registered tier **SHORT**: at the fixed 67.5k endpoint, vanilla (P1) beats fa on the decay/spectral metrics while **fa wins all three retrieval metrics at both K** (z 2.9–9.6). The endpoint landed band-worst for fa (outside its own screen band), so the registered verdict is conservative by construction — and the program's matched-step evidence (below) is unchanged by it.

## What the exp_10 program established (its lasting results)

1. **fa-from-scratch is a viable peer, not a failure** — the exp_07 retraction is now backed by a full trajectory: at matched recipe and steps, fa won 12/12 cells at 40k (5-seed), tracked vanilla's error metrics through 65k, and led retrieval from 50k onward. The old "2×-worse plateau" is definitively an eval-protocol artifact.
2. **The invariance effect is training-side** (decomposition 2×2, all cells 5-seed): inference ensembling alone *hurts* a vanilla model; the fa advantage exists only when trained-in. This licenses the claim "training with C₄-invariant conditioning improves matched-step performance" (one training seed; ≤65k window).
3. **Retrieval is fa's most robust edge** — R@1/R@5/R@10 lead at both K even at fa's band-worst endpoint. Consistent with the invariant-conditioning hypothesis: rotation-consistent embeddings should most help discriminative (retrieval-graded) fidelity.
4. **Equivariance is exact and free** — R3 spreads ≤1e-3 on decay metrics; the 45° control breaks. No quality-vs-equivariance tradeoff is visible within the band.

## Honest scope

- The SHORT tier is the pre-registered reading and stands. The endpoint-draw caveat is documented with its own screens (not a post-hoc rescue); the window statistic was pre-registered as exploratory (R1b) and is labeled so.
- One training seed per arm; the 42.5–65k screens are single-eval-seed steering data (only 40k/67.5k rows are 5-seed).
- Per-step, not per-compute: fa costs ≈3.5× per training step. At matched *compute* vanilla trains ~3.5× more steps — that comparison was not run (would be a follow-up estimand).
- Cluster-copy divergence (stall at 65k after the Aug-4 wipe) documented and reconciled; all gated numbers come from the completed original run.

## Recommendations

1. **Paper story (fa track):** matched-step superiority at 40k (12/12) + training-side attribution (decomposition) + exact equivariance; report the 67.5k endpoint split verdict with the band caveat. fa's headline: *equivariant retrieval gains at every budget measured*.
2. If a "best fa checkpoint" is wanted for the model zoo: 5-seed-confirm the 62.5k point (band-typical, T60 8.58 sub-released s42) rather than the band-worst endpoint — flag: selection-rule formality would need a small pre-registered addendum.
3. The matched-compute comparison (fa @X steps vs vanilla @3.5X steps) is the one estimand this program leaves genuinely open.
4. Cross-machine `metrics_json/` consolidation (proposal pending with Yixun) before any further multi-machine experiments.
