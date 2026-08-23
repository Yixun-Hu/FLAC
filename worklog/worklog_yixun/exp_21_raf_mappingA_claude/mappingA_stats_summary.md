# Mapping-A cross-arm contrasts — exp_21

Generated 2026-08-23T18:37:48Z from the 25-cell sweep (5 arms x 5 seeds).

## Design

- **1152 items** = 2 rooms x 16 placements x 36 mic slots, evaluated at seeds [42, 43, 44, 45, 46].
- **Clustering unit: placement.** The 36 items of a placement share a room position, an array, a target source and largely overlapping context, so item-i.i.d. intervals would understate the uncertainty.
- **Aggregation: equal-room macro of placement means** — the two rooms are the population, so neither is weighted by placement count.
- **Pairing is exact**: every arm evaluated the same 1,152 items under the same conditioning stream at the same five seeds, so each contrast is differenced item by item and seed by seed BEFORE any averaging.
- Intervals: room-stratified cluster bootstrap over placements (10000 resamples, alpha 0.05); p-values: paired sign-flip randomization over the same unit.
- **lower is better (all five are errors)**, so a negative difference favours the first arm.

### Flagged items (registered disclosures)

- 19 items carry a near-silent CONTEXT reference (Amendment 4.1, all FurnishedRoom p008);
- 7 items' listener map sees near-field scanned structure (Amendment 4.2/4.4).
- The **primary** row keeps all 1152; the **minus-flagged** row drops those 26. Both are reported for every contrast: the primary is the result, the sensitivity row says whether it depends on them.

> Mapping-A audio is written at x2.0 over its complete union; Mapping H is at x3.0. No audio file is shared between the two publications. Cross-mapping ABSOLUTE level-dependent comparisons (multi-resolution L1, Env) are therefore unlicensed; the contrasts reported here are WITHIN Mapping A and unaffected, and T60/C50/EDT are level-independent.

## Findings

- **24 of 50** contrasts hold at alpha 0.05; **10** also survive a Bonferroni correction over the ten pairs tested per metric.
- **The minus-flagged row changes nothing**: 0 sign flips and 0 verdict changes across all 50 contrasts. No conclusion here depends on the 26 flagged items.
- **T60**: best YAW (19.1629), worst BV (22.5860); ordering YAW < finetuned < BF < P1 < BV. Holding: P1 < BV, YAW < BV, BF < BV, finetuned < BV.
- **C50**: best finetuned (2.9008), worst YAW (3.9570); ordering finetuned < BF < P1 < BV < YAW. Holding: BF < YAW, BF < BV, finetuned < P1, finetuned < YAW, finetuned < BV, finetuned < BF.
- **EDT**: best finetuned (83.7307), worst YAW (114.7366); ordering finetuned < P1 < BF < BV < YAW. Holding: BF < YAW, finetuned < YAW, finetuned < BV.
- **L1_STFT_MultiRes**: best finetuned (2.8512), worst BV (2.9323); ordering finetuned < YAW < P1 < BF < BV. Holding: P1 < BV, P1 < BF, YAW < BV, YAW < BF.
- **Env**: best finetuned (0.5099), worst BF (0.6223); ordering finetuned < P1 < BV < YAW < BF. Holding: P1 < YAW, P1 < BF, YAW < BF, BV < BF, finetuned < YAW, finetuned < BV, finetuned < BF.

## Arms

| label | registered ckpt | cond_method | seeds | identity |
|---|---|---|---|---|
| P1 | `c4c678826cdd` (P1) | vanilla | [42, 43, 44, 45, 46] | `95aab5adbe17` |
| YAW | `ac1f26034e4f` (YAW) | vanilla | [42, 43, 44, 45, 46] | `68198bd8a02f` |
| BV | `ace9f7350707` (BV) | vanilla | [42, 43, 44, 45, 46] | `683138395437` |
| BF | `5319feb4af87` (BF) | fa_invariant | [42, 43, 44, 45, 46] | `c8f0198e7dd8` |
| finetuned | `6dfc2b2ebdc7` (finetuned) | vanilla | [42, 43, 44, 45, 46] | `2fd1e8248d42` |

Shared across every arm (asserted, not assumed): dataset config `02ba78abdde6`, prepare generation `0de97c5a1c12`, depth generation `21a8ec5fc9bd`, item stream `52e7a72ee341`, 10000-resample settings, steps 1, cfg 1.0, batch 64, source `3051f3aa`.

Invalid-T60 rate (a per-item flag, reported as the rate it is): P1 0.0000, YAW 0.0000, BV 0.0000, BF 0.0000, finetuned 0.0000.

## Headline: equal-room macro per arm

| metric | P1 | YAW | BV | BF | finetuned |
|---|---|---|---|---|---|
| T60 (% error) | 19.8724 (±0.0124) | 19.1629 (±0.0232) | 22.5860 (±0.0257) | 19.5090 (±0.0255) | 19.2887 (±0.0395) |
| C50 (dB error) | 3.8140 (±0.0017) | 3.9570 (±0.0026) | 3.9212 (±0.0028) | 3.6126 (±0.0026) | 2.9008 (±0.0016) |
| EDT (ms error) | 105.4421 (±0.0991) | 114.7366 (±0.1241) | 112.3975 (±0.0903) | 106.1463 (±0.1137) | 83.7307 (±0.1710) |
| L1_STFT_MultiRes (multi-resolution L1) | 2.8885 (±0.0006) | 2.8596 (±0.0010) | 2.9323 (±0.0013) | 2.9247 (±0.0009) | 2.8512 (±0.0009) |
| Env (envelope distance) | 0.5555 (±0.0004) | 0.5732 (±0.0005) | 0.5682 (±0.0004) | 0.6223 (±0.0004) | 0.5099 (±0.0002) |

Parenthesised value is the **seed SD** — Monte-Carlo variability of the sampler, reported beside the estimate and never inside an interval.

## Per-room means

**T60** (% error)

| arm | EmptyRoom | FurnishedRoom |
|---|---|---|
| P1 | 17.5596 | 22.1852 |
| YAW | 17.1443 | 21.1815 |
| BV | 18.9451 | 26.2269 |
| BF | 17.2077 | 21.8104 |
| finetuned | 17.0581 | 21.5193 |

**C50** (dB error)

| arm | EmptyRoom | FurnishedRoom |
|---|---|---|
| P1 | 4.0521 | 3.5759 |
| YAW | 4.2624 | 3.6517 |
| BV | 4.2426 | 3.5997 |
| BF | 3.9271 | 3.2981 |
| finetuned | 3.4309 | 2.3706 |

**EDT** (ms error)

| arm | EmptyRoom | FurnishedRoom |
|---|---|---|
| P1 | 112.9957 | 97.8885 |
| YAW | 128.0296 | 101.4437 |
| BV | 124.7855 | 100.0095 |
| BF | 118.1679 | 94.1246 |
| finetuned | 107.4051 | 60.0562 |

**L1_STFT_MultiRes** (multi-resolution L1)

| arm | EmptyRoom | FurnishedRoom |
|---|---|---|
| P1 | 2.9255 | 2.8516 |
| YAW | 2.9382 | 2.7809 |
| BV | 2.9931 | 2.8714 |
| BF | 2.9958 | 2.8535 |
| finetuned | 2.9210 | 2.7814 |

**Env** (envelope distance)

| arm | EmptyRoom | FurnishedRoom |
|---|---|---|
| P1 | 0.5554 | 0.5557 |
| YAW | 0.5915 | 0.5549 |
| BV | 0.5884 | 0.5480 |
| BF | 0.6341 | 0.6104 |
| finetuned | 0.5384 | 0.4813 |

## Contrasts

### T60 (% error)

| pair | kind | difference | 95% CI | p | holds | minus-flagged | sign flip |
|---|---|---|---|---|---|---|---|
| P1 vs YAW | AR-arm | +0.7095 | [-0.1088, +1.5173] | 0.1057 | no | no (+0.6684) | no |
| P1 vs BV | AR-arm | -2.7136 (P1 better) | [-3.5795, -1.9131] | 0.0000 | **yes** | **yes** (-2.7571) | no |
| P1 vs BF | AR-arm | +0.3634 | [-0.5516, +1.2609] | 0.4485 | no | no (+0.3673) | no |
| YAW vs BV | AR-arm | -3.4231 (YAW better) | [-4.5584, -2.2956] | 0.0000 | **yes** | **yes** (-3.4256) | no |
| YAW vs BF | AR-arm | -0.3461 | [-1.2038, +0.5574] | 0.4537 | no | no (-0.3012) | no |
| BV vs BF | AR-arm | +3.0770 (BF better) | [+2.1486, +4.0163] | 0.0000 | **yes** | **yes** (+3.1244) | no |
| P1 vs finetuned | transfer | +0.5837 | [-1.6409, +2.7605] | 0.5997 | no | no (+0.6111) | no |
| YAW vs finetuned | transfer | -0.1258 | [-2.5970, +2.0551] | 0.9190 | no | no (-0.0573) | no |
| BV vs finetuned | transfer | +3.2973 (finetuned better) | [+0.9126, +5.7700] | 0.0154 | **yes** | **yes** (+3.3683) | no |
| BF vs finetuned | transfer | +0.2203 | [-2.0953, +2.4726] | 0.8567 | no | no (+0.2439) | no |

### C50 (dB error)

| pair | kind | difference | 95% CI | p | holds | minus-flagged | sign flip |
|---|---|---|---|---|---|---|---|
| P1 vs YAW | AR-arm | -0.1430 | [-0.4049, +0.0979] | 0.2931 | no | no (-0.1340) | no |
| P1 vs BV | AR-arm | -0.1072 | [-0.4625, +0.2174] | 0.5614 | no | no (-0.1244) | no |
| P1 vs BF | AR-arm | +0.2014 | [-0.0263, +0.4099] | 0.0843 | no | no (+0.1997) | no |
| YAW vs BV | AR-arm | +0.0359 | [-0.1937, +0.2635] | 0.7576 | no | no (+0.0096) | no |
| YAW vs BF | AR-arm | +0.3445 (BF better) | [+0.1787, +0.5270] | 0.0001 | **yes** | **yes** (+0.3337) | no |
| BV vs BF | AR-arm | +0.3086 (BF better) | [+0.0832, +0.5324] | 0.0111 | **yes** | **yes** (+0.3242) | no |
| P1 vs finetuned | transfer | +0.9132 (finetuned better) | [+0.2490, +1.5766] | 0.0154 | **yes** | **yes** (+0.8895) | no |
| YAW vs finetuned | transfer | +1.0563 (finetuned better) | [+0.3345, +1.7552] | 0.0080 | **yes** | **yes** (+1.0235) | no |
| BV vs finetuned | transfer | +1.0204 (finetuned better) | [+0.3312, +1.7174] | 0.0088 | **yes** | **yes** (+1.0139) | no |
| BF vs finetuned | transfer | +0.7118 (finetuned better) | [+0.0974, +1.3111] | 0.0319 | **yes** | **yes** (+0.6897) | no |

### EDT (ms error)

| pair | kind | difference | 95% CI | p | holds | minus-flagged | sign flip |
|---|---|---|---|---|---|---|---|
| P1 vs YAW | AR-arm | -9.2945 (P1 better) | [-19.2592, -0.6353] | 0.0692 | no | no (-9.0497) | no |
| P1 vs BV | AR-arm | -6.9555 | [-19.2717, +3.3044] | 0.2676 | no | no (-7.5218) | no |
| P1 vs BF | AR-arm | -0.7042 | [-9.3372, +6.8358] | 0.8670 | no | no (-0.7362) | no |
| YAW vs BV | AR-arm | +2.3391 | [-4.6629, +9.3614] | 0.5215 | no | no (+1.5279) | no |
| YAW vs BF | AR-arm | +8.5904 (BF better) | [+4.4681, +13.2023] | 0.0001 | **yes** | **yes** (+8.3135) | no |
| BV vs BF | AR-arm | +6.2513 | [-0.4516, +13.1096] | 0.0871 | no | no (+6.7856) | no |
| P1 vs finetuned | transfer | +21.7114 | [-0.8907, +43.4594] | 0.0775 | no | no (+20.9989) | no |
| YAW vs finetuned | transfer | +31.0060 (finetuned better) | [+7.2614, +54.7033] | 0.0202 | **yes** | **yes** (+30.0486) | no |
| BV vs finetuned | transfer | +28.6669 (finetuned better) | [+3.7754, +53.1940] | 0.0367 | **yes** | **yes** (+28.5207) | no |
| BF vs finetuned | transfer | +22.4156 (finetuned better) | [+0.2458, +44.5359] | 0.0618 | no | no (+21.7351) | no |

### L1_STFT_MultiRes (multi-resolution L1)

| pair | kind | difference | 95% CI | p | holds | minus-flagged | sign flip |
|---|---|---|---|---|---|---|---|
| P1 vs YAW | AR-arm | +0.0290 | [-0.0130, +0.0671] | 0.2117 | no | no (+0.0303) | no |
| P1 vs BV | AR-arm | -0.0438 (P1 better) | [-0.0803, -0.0074] | 0.0304 | **yes** | **yes** (-0.0483) | no |
| P1 vs BF | AR-arm | -0.0361 (P1 better) | [-0.0618, -0.0106] | 0.0166 | **yes** | **yes** (-0.0341) | no |
| YAW vs BV | AR-arm | -0.0727 (YAW better) | [-0.1208, -0.0262] | 0.0052 | **yes** | **yes** (-0.0787) | no |
| YAW vs BF | AR-arm | -0.0651 (YAW better) | [-0.0970, -0.0276] | 0.0009 | **yes** | **yes** (-0.0644) | no |
| BV vs BF | AR-arm | +0.0076 | [-0.0244, +0.0419] | 0.6750 | no | no (+0.0142) | no |
| P1 vs finetuned | transfer | +0.0373 | [-0.0443, +0.1194] | 0.3924 | no | no (+0.0320) | no |
| YAW vs finetuned | transfer | +0.0084 | [-0.0690, +0.0834] | 0.8348 | no | no (+0.0017) | no |
| BV vs finetuned | transfer | +0.0811 | [-0.0012, +0.1600] | 0.0641 | no | no (+0.0803) | no |
| BF vs finetuned | transfer | +0.0735 (finetuned better) | [+0.0018, +0.1430] | 0.0559 | no | no (+0.0661) | no |

### Env (envelope distance)

| pair | kind | difference | 95% CI | p | holds | minus-flagged | sign flip |
|---|---|---|---|---|---|---|---|
| P1 vs YAW | AR-arm | -0.0176 (P1 better) | [-0.0286, -0.0072] | 0.0098 | **yes** | **yes** (-0.0172) | no |
| P1 vs BV | AR-arm | -0.0127 | [-0.0266, +0.0021] | 0.1342 | no | no (-0.0147) | no |
| P1 vs BF | AR-arm | -0.0667 (P1 better) | [-0.0778, -0.0553] | 0.0000 | **yes** | **yes** (-0.0664) | no |
| YAW vs BV | AR-arm | +0.0050 | [-0.0111, +0.0213] | 0.5625 | no | no (+0.0025) | no |
| YAW vs BF | AR-arm | -0.0491 (YAW better) | [-0.0640, -0.0325] | 0.0000 | **yes** | **yes** (-0.0492) | no |
| BV vs BF | AR-arm | -0.0541 (BV better) | [-0.0677, -0.0403] | 0.0000 | **yes** | **yes** (-0.0517) | no |
| P1 vs finetuned | transfer | +0.0457 | [-0.0051, +0.0953] | 0.0912 | no | no (+0.0427) | no |
| YAW vs finetuned | transfer | +0.0633 (finetuned better) | [+0.0155, +0.1113] | 0.0163 | **yes** | **yes** (+0.0599) | no |
| BV vs finetuned | transfer | +0.0583 (finetuned better) | [+0.0134, +0.1007] | 0.0205 | **yes** | **yes** (+0.0574) | no |
| BF vs finetuned | transfer | +0.1124 (finetuned better) | [+0.0587, +0.1640] | 0.0004 | **yes** | **yes** (+0.1091) | no |

A contrast **holds** when the 95% paired cluster-bootstrap interval excludes zero AND the randomization p-value is below 0.05. Ten pairs are tested per metric and no multiplicity correction is registered; each row's `bonferroni_10_pairs` flag in the JSON says whether it would also survive p < 0.005.

