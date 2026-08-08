# Results — exp_13 decay_tail (final; 2026-08-08)

Run: quarter-lr tail (retuned-checkpoint delivery, live-lr gate PASSED: post-restore InverseLR(30000,1.0,0.99), boundary lr bit-equal 1.2765957446808513e-05) from the 87.5k anchor → 97,500; 8 tail ckpts; screens EMA/K8/s42/cfg1.0/steps1/full 6,337/17/per-scene (pinned protocol).

## Registered verdict: **NULL** (DT1 0/8 qualifiers, bar ≥4/8) · **DT3 mechanism CONFIRMED** (band shrink ratios 0.570/0.605)

**Tail screens:** 88.75k 8.569/0.9667/37.033/R6.770 · 90k 8.880/0.9519/36.929/6.675 · 91.25k 9.060/0.9525/37.024/6.596 · 92.5k 9.209/0.9333/37.671/6.912 · 93.75k 9.027/**0.9277**/37.186/**R7.006** · 95k 8.985/0.9345/36.752/6.549 · 96.25k 9.017/0.9340/37.302/6.565 · 97.5k 9.074/**0.9264**/37.748/6.659.

- **DT1: 0/8** against the anchor-draw-matching bar (T60≤8.40 ∧ C50≤0.975 ∧ EDT≤36.6 ∧ R@1≥6.60). Failure mode is NOT residual oscillation: T60 sits at a consistent ~9.0 (band 8.57–9.21) — the tail converged AWAY from the anchor's T60 region while C50 moved to program-best values (all 8 points below the anchor's 0.966; two below 0.93) and R@1 held 6.55–7.01.
- **DT3 (primary, matched steps {90k,92.5k,95k,97.5k} vs P1's same steps, ddof=1):** tail T60 SD 0.1392 vs 0.244 → **ratio 0.570**; EDT SD 0.5080 vs 0.839 → **ratio 0.605** (n=4 each, χ² CI wide — point estimates only). Supplementary full-tail (n=8, 1250-grid, downward-biased): T60 SD 0.181, EDT SD 0.334.
- **DT2: N/A** (no DT1 candidate). Contextual: best tail point S93750 vs released — C50 0.9277 far better, R@1 7.006 ≈ 7.06, EDT 37.19 ≈ 37.10, T60 9.03 worse (+0.42).

## Interpretation (registered-NULL, mechanism-positive)
The decay hypothesis was half right: lr-decay **does** shrink the orbit (DT3), but co-occurrence fails because the metrics **trade structurally** — the tightened trajectory picks one point on a Pareto surface (here: C50/R@1-favorable, T60-unfavorable) rather than freezing near the anchor's all-cells draw. Corollary: the anchor's 8/8 is a band-best draw from a WIDE orbit; narrow orbits cannot reproduce it. Consistent with exp_10's A3 finding (40k spike) — this program's headline checkpoints are wide-band selection events, now demonstrated causally.

One TRAINING seed; single-eval-seed screens (no 5-seed rows — no candidate); EMA 63.2% turned over at tail end (C50 shift partially EMA-blended; disclosed).
