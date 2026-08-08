# Results — exp_13 decay_tail (final; 2026-08-08)

Run: quarter-lr tail (retuned-checkpoint delivery. Live-lr gate PASSED: closed-form agreement at probe step 87,515 = 1.2764328e-05 within 1e-12; the rewritten checkpoint field itself is bit-equal to the boundary pin 1.2765957446808513e-05) from the 87.5k anchor → 97,500; 8 tail ckpts; screens EMA/K8/s42/cfg1.0/steps1/full 6,337/17/per-scene (pinned protocol).

## Registered verdict: **NULL** (DT1 0/8 qualifiers, bar ≥4/8) · **DT3 mechanism SUPPORTED** (point estimates 0.570/0.605; 95% CIs wide and include 1 — see below)

**Tail screens:** 88.75k 8.569/0.9667/37.033/R6.770 · 90k 8.880/0.9519/36.929/6.675 · 91.25k 9.060/0.9525/37.024/6.596 · 92.5k 9.209/0.9333/37.671/6.912 · 93.75k 9.027/**0.9277**/37.186/**R7.006** · 95k 8.985/0.9345/36.752/6.549 · 96.25k 9.017/0.9340/37.302/6.565 · 97.5k 9.074/**0.9264**/37.748/6.659.

- **DT1: 0/8** against the anchor-draw-matching bar (T60≤8.40 ∧ C50≤0.975 ∧ EDT≤36.6 ∧ R@1≥6.60). Failure mode is NOT residual oscillation: T60 sits at a consistent ~9.0 (band 8.57–9.21) — the tail converged AWAY from the anchor's T60 region while C50 moved to the best values in the P1/anchor lineage (7 of 8 points below the anchor's 0.966 — S88750 reads 0.9667; two below 0.93; on par with, not better than, the exp_05/08 fine-tune line 0.922–0.930 5-seed) and R@1 held 6.55–7.01.
- **DT3 (primary, matched steps {90k,92.5k,95k,97.5k} vs P1's same steps, ddof=1):** tail T60 SD 0.1392 vs 0.244 → **ratio 0.570, 95% CI [0.145, 2.241]**; EDT SD 0.5080 vs 0.839 → **ratio 0.605, 95% CI [0.154, 2.379]** (variance-ratio F(3,3); both CIs CONTAIN 1 — the A/B does not exclude no-shrink; point estimates directional only). Independent untreated reference, same steps: exp09_V SDs 0.4623/1.390 — same direction. Supplementary full-tail (n=8, 1250-grid, downward-biased, ddof=1): T60 SD 0.189, EDT SD 0.352.
- **DT2: N/A** (no DT1 candidate). Contextual (SINGLE eval seed vs 5-seed released means): S93750 — C50 0.9277 far better, R@1 7.006 ≈ 7.057±0.102, EDT 37.186 = +0.09 (~1.3 seed-SD WORSE than 37.100±0.067), T60 9.03 worse (+0.42).

## Interpretation (registered-NULL, mechanism-positive)
The decay hypothesis was half right: lr-decay **does** shrink the orbit (DT3), but co-occurrence fails because the metrics **trade structurally** — the tightened trajectory picks one point on a Pareto surface (here: C50/R@1-favorable, T60-unfavorable) rather than freezing near the anchor's all-cells draw. Corollary: the anchor's 8/8 is a band-best draw from a WIDE orbit; narrow orbits cannot reproduce it. Consistent with exp_10's A3 finding (40k spike) — this program's headline checkpoints are wide-band selection events, now demonstrated causally.

One TRAINING seed; single-eval-seed screens (no 5-seed rows — no candidate); EMA 63.2% turned over at tail end (C50 shift partially EMA-blended; disclosed).

## Addendum — S93750 5-seed confirm (Yixun go, 2026-08-08; both K, pinned protocol)
K=8: T60 9.0264±0.0130 (OUT +23.6σ) / **C50 0.9288±0.0012 (SUPERIOR −12.1σ)** / EDT 37.1711±0.0942 (EQUIV +0.6σ) / R@1 6.9496±0.1097 (EQUIV −0.7σ) / R@5 20.1862±0.1209 / R@10 28.5561±0.0876. K=1: 10.2861±0.0284 (OUT) / **0.9951±0.0041 (SUPERIOR −6.7σ; first sub-1.0 K=1 C50 in the program)** / 39.8758±0.2523 (EQUIV) / 6.8045±0.2100 (EQUIV) / 19.9432 / 27.9312. **6/8 core cells SUPERIOR-or-EQUIV vs released; single concession T60.** Model-zoo entry: the C50/retrieval-weighted flavor (exploratory selection, now seed-confirmed).
