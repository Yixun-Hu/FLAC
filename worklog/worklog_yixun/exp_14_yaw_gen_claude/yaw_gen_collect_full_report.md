# exp_14 — random-yaw generalization: collected readouts

Generated 2026-08-12T11:01:32-04:00 by `yaw_gen_collect.py` — every cell is re-validated from its own artifacts on each run; numbers never live in this file.

- campaign pin `e8ca26edb116cc56f0e62289cb65008c947079b7` · step 40000 · seeds [42, 43, 44, 45, 46] · α=0.05 · stats backend: scipy
- co-primary metrics: T60, RIR_to_GT_RIR_R@1 (Holm over the two, within each labelled hypothesis)
- **per-metric aggregation (Planner ruling, pre-registered 2026-08-12):** per-scene mean applies to the ACOUSTIC-PARAMETER family only — T60 (incl. Invalid-T60 handling), C50, EDT — matching the paper convention plan §4 intended. RETRIEVAL (RIR_to_GT_RIR_R@k, and the quarantined RIR_to_geom_R@k) and FD use the SPLIT-LEVEL global metrics: within-scene retrieval among ~370 items is a different, easier task whose levels are incomparable to every previously published number in this program, and exp_01's noise-floor calibration against released Table-1 was on the global quantity; one-room Frechet is additionally small-sample biased. Co-primaries: T60% (per-scene mean) + RIR_to_GT_RIR_R@1 (split-level).
  - scene-mean = mean over the 10 AR ROOM FAMILIES the release metric callback groups by (md['scene'] = the family, AR_md.py); the split's 17 physical rooms are its content, not its grouping.
  - scene-mean: T60, C50, EDT, Invalid T60 · split-level: FD, RIR_to_GT_RIR_R@1, RIR_to_GT_RIR_R@5, RIR_to_GT_RIR_R@10, RIR_to_geom_R@1, RIR_to_geom_R@5, RIR_to_geom_R@10
  - both sources exist in every exp_14 record — the flat `metrics` block and the `by_scene` block — so this is a reading rule, not a measurement change; by_scene stays REQUIRED for every cell because the acoustic family is read from it.

## 1. Cell inventory

- registered 106 · validated 106 · missing 0 · refused 0
- random-yaw (R): 50 · unrotated (Z): 50 · validity (V): 6

## 2. Validity gates (G1–G4) and the G5 check

| gate | status | definition |
|---|---|---|
| G1 | PASS | \|m(V@90°,s42,K8) − m(Z,s42,K8)\| ≤ 0.5·σ̂(arm's 5 Z seeds), each co-primary read through its ruled source (T60 scene-mean, R@1 split-level) |
| G2 | PASS | m_T60(VANL V@90°) − m_T60(VANL Z) ≥ 5·σ̂_T60(VANL), T60 read as the SCENE-MEAN (ruled source) |
| G3 | PASS | cell offsets == draw_yaw_offsets(n, 512, gen(rotate_seed)) |
| G4 | PASS | cross-arm input/assignment hashes equal within (K, seed); Z.input_hash == R.input_hash within (arm, K, seed) |


*G5 (external reproduction — a CHECK, never a gate): CHECK.* exp_14 Z vs exp_11 conf @40k, SPLIT-LEVEL on both sides (exp_11 never measured a scene-mean); disclose |Δ| > 3·√(σ11²+σ14²)/√5 (cross-pin: reported, never a halt)

| arm | metric | exp_11 conf | exp_14 Z | Δ | 3σ bound | beyond |
|---|---|---|---|---|---|---|
| C4L | T60 | 8.414 ± 0.006 | 8.414 ± 0.006 | 0.0000 | 0.0121 | no |
| C4L | RIR_to_GT_RIR_R@1 | 5.119 ± 0.126 | 5.091 ± 0.133 | -0.0284 | 0.2461 | no |
| C8 | T60 | 8.714 ± 0.005 | 8.714 ± 0.005 | 0.0000 | 0.0093 | no |
| C8 | RIR_to_GT_RIR_R@1 | 5.182 ± 0.078 | 5.166 ± 0.065 | -0.0158 | 0.1357 | no |
| C16 | T60 | 9.343 ± 0.016 | 9.343 ± 0.016 | 0.0000 | 0.0303 | no |
| C16 | RIR_to_GT_RIR_R@1 | 5.189 ± 0.064 | 5.176 ± 0.054 | -0.0126 | 0.1125 | no |
| C32 | T60 | 9.147 ± 0.009 | 9.147 ± 0.009 | 0.0000 | 0.0179 | no |
| C32 | RIR_to_GT_RIR_R@1 | 5.034 ± 0.086 | 5.059 ± 0.053 | 0.0252 | 0.1360 | no |


## 3. Absolute robustness m_R (the PRIMARY criterion)

**K=1** (descriptive)

| arm | K | n | T60 (scene-mean) ↓ | C50 (scene-mean) ↓ | EDT (scene-mean) ↓ | FD (split) ↓ | RIR_to_GT_RIR_R@1 (split) ↑ | RIR_to_GT_RIR_R@5 (split) ↑ | RIR_to_GT_RIR_R@10 (split) ↑ |
|---|---|---|---|---|---|---|---|---|---|
| VANL | 1 | 5 | 9.030 ± 0.057 | 1.026 ± 0.008 | 38.971 ± 0.378 | 0.324 ± 0.000 | 4.264 ± 0.274 | 14.029 ± 0.354 | 21.133 ± 0.210 |
| C4L | 1 | 5 | 9.279 ± 0.052 | 0.984 ± 0.008 | 40.554 ± 0.406 | 0.332 ± 0.001 | 4.283 ± 0.087 | 14.016 ± 0.096 | 21.297 ± 0.253 |
| C8 | 1 | 5 | 9.036 ± 0.041 | 0.940 ± 0.004 | 40.630 ± 0.280 | 0.334 ± 0.000 | 5.015 ± 0.151 | 15.310 ± 0.228 | 23.077 ± 0.192 |
| C16 | 1 | 5 | 9.412 ± 0.041 | 0.943 ± 0.007 | 42.070 ± 0.260 | 0.329 ± 0.000 | 5.047 ± 0.078 | 15.862 ± 0.116 | 23.759 ± 0.194 |
| C32 | 1 | 5 | 9.209 ± 0.041 | 0.932 ± 0.004 | 39.972 ± 0.391 | 0.322 ± 0.001 | 5.018 ± 0.209 | 15.260 ± 0.240 | 22.743 ± 0.210 |


**K=8** (confirmatory)

| arm | K | n | T60 (scene-mean) ↓ | C50 (scene-mean) ↓ | EDT (scene-mean) ↓ | FD (split) ↓ | RIR_to_GT_RIR_R@1 (split) ↑ | RIR_to_GT_RIR_R@5 (split) ↑ | RIR_to_GT_RIR_R@10 (split) ↑ |
|---|---|---|---|---|---|---|---|---|---|
| VANL | 8 | 5 | 7.724 ± 0.027 | 0.954 ± 0.005 | 36.332 ± 0.237 | 0.330 ± 0.000 | 4.444 ± 0.123 | 14.521 ± 0.304 | 21.786 ± 0.331 |
| C4L | 8 | 5 | 7.972 ± 0.028 | 0.917 ± 0.004 | 38.044 ± 0.215 | 0.338 ± 0.000 | 4.437 ± 0.109 | 14.471 ± 0.210 | 21.856 ± 0.155 |
| C8 | 8 | 5 | 7.726 ± 0.005 | 0.868 ± 0.003 | 38.080 ± 0.121 | 0.340 ± 0.000 | 5.201 ± 0.085 | 15.705 ± 0.250 | 23.516 ± 0.087 |
| C16 | 8 | 5 | 8.141 ± 0.013 | 0.878 ± 0.002 | 39.880 ± 0.071 | 0.335 ± 0.000 | 5.280 ± 0.071 | 16.721 ± 0.154 | 24.687 ± 0.270 |
| C32 | 8 | 5 | 7.974 ± 0.017 | 0.863 ± 0.003 | 37.543 ± 0.116 | 0.328 ± 0.000 | 5.094 ± 0.131 | 15.765 ± 0.164 | 23.279 ± 0.178 |


## 4. Paired degradation Δ = m_R − m_Z (mean over 5 seed-paired diffs, ±½·95% CI)

**K=1**

| arm | K | n | T60 (scene-mean) ↓ | C50 (scene-mean) ↓ | EDT (scene-mean) ↓ | FD (split) ↓ | RIR_to_GT_RIR_R@1 (split) ↑ | RIR_to_GT_RIR_R@5 (split) ↑ | RIR_to_GT_RIR_R@10 (split) ↑ |
|---|---|---|---|---|---|---|---|---|---|
| VANL | 1 | 5 | 0.537 ± 0.030 | 0.069 ± 0.006 | 2.978 ± 0.300 | 0.004 ± 0.000 | -0.508 ± 0.437 | -1.531 ± 0.617 | -2.033 ± 0.296 |
| C4L | 1 | 5 | 0.510 ± 0.017 | 0.044 ± 0.005 | 1.466 ± 0.325 | 0.004 ± 0.000 | -0.694 ± 0.264 | -1.632 ± 0.235 | -1.944 ± 0.331 |
| C8 | 1 | 5 | 0.055 ± 0.012 | 0.003 ± 0.004 | 0.266 ± 0.131 | 0.000 ± 0.000 | -0.013 ± 0.145 | -0.088 ± 0.221 | 0.019 ± 0.273 |
| C16 | 1 | 5 | -0.012 ± 0.020 | -0.002 ± 0.003 | -0.043 ± 0.033 | 0.000 ± 0.000 | 0.010 ± 0.109 | -0.136 ± 0.138 | -0.028 ± 0.203 |
| C32 | 1 | 5 | -0.001 ± 0.024 | -0.003 ± 0.002 | 0.129 ± 0.108 | 0.000 ± 0.000 | 0.054 ± 0.177 | -0.076 ± 0.229 | 0.028 ± 0.192 |


**K=8**

| arm | K | n | T60 (scene-mean) ↓ | C50 (scene-mean) ↓ | EDT (scene-mean) ↓ | FD (split) ↓ | RIR_to_GT_RIR_R@1 (split) ↑ | RIR_to_GT_RIR_R@5 (split) ↑ | RIR_to_GT_RIR_R@10 (split) ↑ |
|---|---|---|---|---|---|---|---|---|---|
| VANL | 8 | 5 | 0.521 ± 0.037 | 0.066 ± 0.007 | 2.954 ± 0.209 | 0.005 ± 0.000 | -0.505 ± 0.265 | -1.483 ± 0.487 | -1.865 ± 0.410 |
| C4L | 8 | 5 | 0.531 ± 0.029 | 0.048 ± 0.005 | 1.597 ± 0.207 | 0.005 ± 0.000 | -0.653 ± 0.243 | -1.805 ± 0.351 | -2.061 ± 0.270 |
| C8 | 8 | 5 | 0.049 ± 0.011 | 0.003 ± 0.004 | 0.276 ± 0.133 | 0.000 ± 0.000 | 0.035 ± 0.085 | -0.006 ± 0.223 | -0.073 ± 0.247 |
| C16 | 8 | 5 | -0.003 ± 0.018 | -0.003 ± 0.002 | -0.007 ± 0.035 | 0.000 ± 0.000 | 0.104 ± 0.095 | 0.019 ± 0.213 | 0.038 ± 0.228 |
| C32 | 8 | 5 | 0.006 ± 0.011 | -0.003 ± 0.002 | 0.126 ± 0.121 | 0.000 ± 0.000 | 0.035 ± 0.130 | 0.085 ± 0.152 | -0.057 ± 0.322 |


## 5. Endpoint contrasts (H-P / H-M / H-S)

**H-P (PRIMARY): m_R(C32) vs m_R(VANL)** (K=8)

| metric | mean Δ | 95% CI | p | p (Holm) | favours first | won |
|---|---|---|---|---|---|---|
| T60 (scene-mean) | 0.2502 | [0.2135, 0.2870] | 4.61e-05 | 9.219e-05 | no | no |
| RIR_to_GT_RIR_R@1 (split) | 0.6502 | [0.3591, 0.9412] | 0.003438 | 0.003438 | yes | yes |

**Verdict: PARTIAL**


**H-M (mechanism): |Δ|(C32) vs |Δ|(C4L)** (K=8)

| metric | mean Δ | 95% CI | p | p (Holm) | favours first | won |
|---|---|---|---|---|---|---|
| T60 (scene-mean) | -0.5215 | [-0.5472, -0.4957] | 6.002e-07 | 1.2e-06 | yes | yes |
| RIR_to_GT_RIR_R@1 (split) | -0.5744 | [-0.8082, -0.3406] | 0.002414 | 0.002414 | yes | yes |

**Verdict: SUPPORTED**
- alongside — |Δ|(VANL) vs |Δ|(C4L): verdict NEGATIVE


**H-S (sanity): Δ(VANL) ≠ 0** (K=8)

| metric | mean Δ | 95% CI | p | p (Holm) | favours first | won |
|---|---|---|---|---|---|---|
| T60 (scene-mean) | 0.5208 | [0.4842, 0.5573] | 2.439e-06 | 4.878e-06 | no | yes |
| RIR_to_GT_RIR_R@1 (split) | -0.5050 | [-0.7697, -0.2403] | 0.006101 | 0.006101 | no | yes |

**Verdict: SUPPORTED**


*K=1 (descriptive repeat):*

- H-P (PRIMARY): m_R(C32) vs m_R(VANL) → PARTIAL
- H-M (mechanism): |Δ|(C32) vs |Δ|(C4L) → SUPPORTED
- H-S (sanity): Δ(VANL) ≠ 0 → SUPPORTED

## 6. Adjacent fixed-order contrasts

fixed-order adjacent contrasts on the plan's arm order, unadjusted and descriptive: no verdict attaches to them and the observed ranking is never turned into a confirmatory test

*absolute m_R, fixed order VANL→C4L→C8→C16→C32*

| pair | K | metric | mean Δ | 95% CI | p |
|---|---|---|---|---|---|
| VANL→C4L | 1 | T60 (scene-mean) | 0.2489 | [0.1623, 0.3355] | 0.001336 |
| VANL→C4L | 1 | RIR_to_GT_RIR_R@1 (split) | 0.0189 | [-0.2641, 0.3020] | 0.8616 |
| C4L→C8 | 1 | T60 (scene-mean) | -0.2433 | [-0.2881, -0.1984] | 0.0001136 |
| C4L→C8 | 1 | RIR_to_GT_RIR_R@1 (split) | 0.7322 | [0.5555, 0.9089] | 0.0003258 |
| C8→C16 | 1 | T60 (scene-mean) | 0.3762 | [0.3367, 0.4157] | 1.217e-05 |
| C8→C16 | 1 | RIR_to_GT_RIR_R@1 (split) | 0.0316 | [-0.2441, 0.3073] | 0.7664 |
| C16→C32 | 1 | T60 (scene-mean) | -0.2029 | [-0.2261, -0.1796] | 1.719e-05 |
| C16→C32 | 1 | RIR_to_GT_RIR_R@1 (split) | -0.0284 | [-0.3393, 0.2825] | 0.8122 |
| VANL→C4L | 8 | T60 (scene-mean) | 0.2485 | [0.2074, 0.2895] | 7.366e-05 |
| VANL→C4L | 8 | RIR_to_GT_RIR_R@1 (split) | -0.0063 | [-0.1683, 0.1557] | 0.9192 |
| C4L→C8 | 8 | T60 (scene-mean) | -0.2460 | [-0.2841, -0.2080] | 5.663e-05 |
| C4L→C8 | 8 | RIR_to_GT_RIR_R@1 (split) | 0.7638 | [0.6979, 0.8296] | 5.543e-06 |
| C8→C16 | 8 | T60 (scene-mean) | 0.4144 | [0.3991, 0.4296] | 1.867e-07 |
| C8→C16 | 8 | RIR_to_GT_RIR_R@1 (split) | 0.0789 | [-0.0293, 0.1871] | 0.113 |
| C16→C32 | 8 | T60 (scene-mean) | -0.1665 | [-0.1915, -0.1415] | 5.02e-05 |
| C16→C32 | 8 | RIR_to_GT_RIR_R@1 (split) | -0.1862 | [-0.3965, 0.0241] | 0.06978 |


*|Δ| (flatness), same fixed order*

| pair | K | metric | mean Δ | 95% CI | p |
|---|---|---|---|---|---|
| VANL→C4L | 1 | T60 (scene-mean) | -0.0268 | [-0.0707, 0.0170] | 0.1642 |
| VANL→C4L | 1 | RIR_to_GT_RIR_R@1 (split) | 0.1862 | [-0.2919, 0.6643] | 0.3403 |
| C4L→C8 | 1 | T60 (scene-mean) | -0.4549 | [-0.4673, -0.4424] | 5.578e-08 |
| C4L→C8 | 1 | RIR_to_GT_RIR_R@1 (split) | -0.6060 | [-0.8271, -0.3848] | 0.001602 |
| C8→C16 | 1 | T60 (scene-mean) | -0.0398 | [-0.0560, -0.0237] | 0.002393 |
| C8→C16 | 1 | RIR_to_GT_RIR_R@1 (split) | -0.0221 | [-0.1449, 0.1007] | 0.6439 |
| C16→C32 | 1 | T60 (scene-mean) | -0.0005 | [-0.0198, 0.0188] | 0.9491 |
| C16→C32 | 1 | RIR_to_GT_RIR_R@1 (split) | 0.0505 | [-0.0536, 0.1546] | 0.2491 |
| VANL→C4L | 8 | T60 (scene-mean) | 0.0101 | [-0.0285, 0.0487] | 0.5067 |
| VANL→C4L | 8 | RIR_to_GT_RIR_R@1 (split) | 0.1483 | [-0.1304, 0.4271] | 0.2136 |
| C4L→C8 | 8 | T60 (scene-mean) | -0.4821 | [-0.5211, -0.4430] | 4.31e-06 |
| C4L→C8 | 8 | RIR_to_GT_RIR_R@1 (split) | -0.5933 | [-0.8414, -0.3452] | 0.00267 |
| C8→C16 | 8 | T60 (scene-mean) | -0.0359 | [-0.0457, -0.0260] | 0.0005394 |
| C8→C16 | 8 | RIR_to_GT_RIR_R@1 (split) | 0.0505 | [-0.0247, 0.1256] | 0.1357 |
| C16→C32 | 8 | T60 (scene-mean) | -0.0035 | [-0.0100, 0.0030] | 0.2083 |
| C16→C32 | 8 | RIR_to_GT_RIR_R@1 (split) | -0.0316 | [-0.1192, 0.0560] | 0.3738 |


## 7. Geometry retrieval (rotated-gallery, confounded — descriptive only)

rotated-gallery retrieval: in an R cell the gallery embeds the ROTATED point cloud through a non-yaw-invariant AGREE, so these numbers mix model robustness with AGREE's own yaw sensitivity. Cross-arm comparisons stay internally valid (the galleries are rotation-matched) but the level is confounded — descriptive only, never a headline or a co-primary.

| block | arm | K | RIR_to_geom_R@1 (split) | RIR_to_geom_R@5 (split) | RIR_to_geom_R@10 (split) |
|---|---|---|---|---|---|
| R | VANL | 1 | 3.481 ± 0.154 | 11.756 ± 0.221 | 18.151 ± 0.188 |
| R | VANL | 8 | 3.560 ± 0.141 | 11.961 ± 0.197 | 18.425 ± 0.233 |
| R | C4L | 1 | 3.270 ± 0.216 | 11.144 ± 0.290 | 17.232 ± 0.289 |
| R | C4L | 8 | 3.326 ± 0.179 | 11.406 ± 0.264 | 17.614 ± 0.330 |
| R | C8 | 1 | 3.415 ± 0.119 | 11.523 ± 0.139 | 17.971 ± 0.156 |
| R | C8 | 8 | 3.431 ± 0.065 | 11.747 ± 0.165 | 18.286 ± 0.283 |
| R | C16 | 1 | 3.513 ± 0.193 | 11.949 ± 0.212 | 18.293 ± 0.198 |
| R | C16 | 8 | 3.592 ± 0.189 | 12.157 ± 0.110 | 18.984 ± 0.175 |
| R | C32 | 1 | 3.219 ± 0.129 | 11.217 ± 0.148 | 17.431 ± 0.202 |
| R | C32 | 8 | 3.222 ± 0.210 | 11.346 ± 0.149 | 17.762 ± 0.186 |
| Z | VANL | 1 | 3.945 ± 0.133 | 13.312 ± 0.134 | 20.092 ± 0.226 |
| Z | VANL | 8 | 3.951 ± 0.108 | 13.401 ± 0.131 | 20.524 ± 0.064 |
| Z | C4L | 1 | 3.863 ± 0.127 | 13.293 ± 0.125 | 20.047 ± 0.154 |
| Z | C4L | 8 | 4.153 ± 0.075 | 13.811 ± 0.054 | 20.672 ± 0.208 |
| Z | C8 | 1 | 4.046 ± 0.098 | 13.151 ± 0.246 | 20.158 ± 0.149 |
| Z | C8 | 8 | 4.046 ± 0.068 | 13.600 ± 0.149 | 20.596 ± 0.161 |
| Z | C16 | 1 | 4.273 ± 0.122 | 13.439 ± 0.110 | 20.189 ± 0.313 |
| Z | C16 | 8 | 4.292 ± 0.140 | 13.956 ± 0.161 | 20.934 ± 0.179 |
| Z | C32 | 1 | 3.932 ± 0.044 | 12.807 ± 0.234 | 19.583 ± 0.137 |
| Z | C32 | 8 | 3.847 ± 0.124 | 13.088 ± 0.189 | 19.842 ± 0.157 |

## 8. Validity cells (V) — QA only, never a headline

| arm | angle | role | status | Δ vs unrotated (co-primaries) |
|---|---|---|---|---|
| C4L | 90° | in-group floor (G1) | OK | T60: -0.0000, RIR_to_GT_RIR_R@1: 0.0000 |
| C8 | 90° | in-group floor (G1) | OK | T60: 0.0003, RIR_to_GT_RIR_R@1: -0.0158 |
| C16 | 90° | in-group floor (G1) | OK | T60: 0.0009, RIR_to_GT_RIR_R@1: 0.0000 |
| C32 | 90° | in-group floor (G1) | OK | T60: -0.0008, RIR_to_GT_RIR_R@1: -0.0158 |
| VANL | 90° | positive control (exp_02 prior: vanilla degrades) | OK | T60: 0.1949, RIR_to_GT_RIR_R@1: -0.2367 |
| C4L | 45° | off-group mechanism control — NO gate role | OK | T60: 1.0962, RIR_to_GT_RIR_R@1: -1.2467 |

## 9. Refused and missing cells

- no cell was refused.
- 0 registered cell(s) have not landed.
