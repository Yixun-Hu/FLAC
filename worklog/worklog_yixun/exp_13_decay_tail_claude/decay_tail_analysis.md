# Analysis — exp_13 decay_tail (closing)

**Author:** main session (Fable 5 seat) · **Reviews:** Opus 5 max fallback seat (Codex 401-down, declared) · **Date:** 2026-08-08

**Outcome:** registered NULL (0/8 co-occurrence qualifiers) with the mechanism SUPPORTED (band SD ratios 0.570/0.605 — point estimates; 95% CIs include 1). INTERPRETATION (post-hoc, unregistered, n=1 arm/1 training seed): the experiment separates **oscillation width** (decay reduces it, directionally) from **metric trade structure** (the tail converges to a different trade point: lineage-best C50, released-level R@1 single-seed, T60 +0.7 off the anchor).

**Reliability:** treatment verified end-to-end (B1-class clobber excluded by the live-lr gate; bit-equal boundary lr); matched-lineage A/B control (P1's own 87.5k→97.5k segment); pre-registered bar with computed null rate (P(≥4|p₀=1/8)=0.01125). Limits: one training seed, screen-level (single eval seed), 10k tail (EMA 63% turnover), one decay shape tested.

**What it changes:** (1) stop chasing all-cells co-occurrence via schedules at this budget — it's a selection phenomenon over wide orbits; (2) the tail is a *dial*: if a deployment weights C50/R@1, S93750-class checkpoints are obtainable cheaply (5.9 h wall / 11.8 GPU-h on the 2-GPU rung) from any anchor; (3) the two-checkpoint paper story (anchor + equivariant Fw) stands unaffected; a "decay-tail S93750" row is an optional third flavor, 5-seed-confirmable on request (~1.5 h).

**Recommended next (if pursued at all):** a longer tail (≥25k, full EMA turnover) or a T60-weighted variant would test whether the Pareto point is steerable — LOW priority vs the cyl_vit track.
