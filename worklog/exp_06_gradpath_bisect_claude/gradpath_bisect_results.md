# Results — exp_06_gradpath_bisect

All evals: full 6337-item unseen split. Screens: K=8, seed 42 (pre-registered as ordering-only; never headline numbers). Code `197c49a`+ (lr-schedule round closed).

## S1 — damage dynamics (interval checkpoints, zero training cost)

| step | R1b T60 | R1b EDT | V1p (frozen BN) T60 | V1p EDT |
|---|---|---|---|---|
| 0 (baseline) | 8.609 | 37.10 | 8.609 | 37.10 |
| 200 | 9.071 | 39.98 | 9.089 | 38.27 |
| 400 | 9.282 | 39.28 | 9.253 | 38.60 |
| 600 / 625* | 9.233 / 9.195* | 41.22 / 40.51* | 9.252 / 9.235* | 39.57 / 38.73* |

(*625 = 5-seed means from prior gate evals.) **~75–80% of the T60 damage lands within the first 200 optimizer steps**, then flattens at the 5e-6 trajectory scale — fast convergence away from the released point, not slow corruption.

## S2 — the lr axis (Yixun's hypothesis): **monotone in the WRONG direction**

| arm | lr | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|---|
| baseline | — | 8.609 | 0.968 | 37.100 | 7.057 |
| L1 | 5e-7 const | 9.087 | 0.953 | 38.751 | 6.817 |
| L2 (=V1p anchor) | 5e-6 const | 9.235 | 0.928 | 38.731 | 6.953 |
| L3 | 2e-5 const | 9.596 | 0.929 | 40.148 | 6.833 |
| L4 | 4.2e-5 const (continuation proxy) | 9.866 | 0.953 | 39.918 | 6.423 |
| L5 | 5e-5 + original InverseLR restart | 10.099 | 0.978 | 40.350 | 6.281 |

All arms: freeze-bn, batch 4×32 (eff. 128), 625 opt steps, seed 42. **No finalist** (thresholds T60 ≤ 8.65 or EDT ≤ 37.3). Answer to the commissioning question: **no tested lr setting — including the schedule-faithful restart — recovers the gate; damage increases monotonically with lr.** The lr-invariant-plateau (checkpoint-selection, simple form) prediction is falsified alongside: damage grows with cumulative lr-distance in the tested range.

## S3 — lineage audit

1. **Code lineage: ELIMINATED.** `git diff upstream/master 0bd5da0` over the entire training path = the fork's own equi-test addition + a 5-line device_map nit (`upstream_diff_trainpath.patch`). The objective/loss/data code is upstream-identical.
2. **Mechanism probes (train-split-only, review-corrected):** truncated energy beyond loss window 0.08% (dead); augmentation bias at effective p=0.5 ≈ T60 0.05 / EDT 0.78 ms — a partial-EDT candidate, **10× too small for T60**.
3. S3.3 aug-off arm: correctly not triggered (its pre-registered condition — S2 flat + material S3.2c — was not met).

## Unified conclusion

Training under this exact, upstream-identical objective on this data walks monotonically away from the released checkpoint (fast: most damage < 200 steps; farther with higher lr). Therefore **the released FLAC_EMA weights are not near this objective's optimum on the data/environment we possess** — the remaining explanations live outside recipe space: dataset-version or library/environment lineage, or source-side checkpoint selection. Fine-tune-based H3-vs-exp_01 is unreachable by any recipe knob tested across exp_03–06 (lr ×100 range incl. schedule restart, warmup, EMA handling, batch parity, BN freezing, grad-clip parity).
