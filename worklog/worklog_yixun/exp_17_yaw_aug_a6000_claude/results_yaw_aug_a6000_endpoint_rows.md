# exp_17 — endpoint rows (a: unseen 5-seed, seen 5-seed, c: 45° probe)

20 cells, 0 failures, 2026-08-16 21:51→22:5x EDT. Protocol identical to the C4
grid and to exp_18's seen rows (vanilla, bf16, cfg 1.0, steps 1, EMA); eval
seeds 42–46; artifacts in `outputs_FLAC/exp17_YAWAUG_extras/` (+ the grid's s42
rot0 cells). P1/B-F rows quoted from `model_comparison.md` / exp_18 records.

## UNSEEN (full published split, 6,337 items / 17 rooms), 40k, 5 seeds

| Method | K | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1) | 1 | 10.287 ± 0.026 | 1.0884 ± 0.0088 | 43.437 ± 0.397 | 4.990 ± 0.115 | 15.111 ± 0.129 | 22.509 ± 0.181 | 0.3186 ± 0.0005 |
| Per-angle FA (B-F) | 1 | 9.543 ± 0.054 | **1.0559 ± 0.0040** | **41.754 ± 0.347** | **5.166 ± 0.166** | **16.071 ± 0.241** | **23.721 ± 0.150** | 0.3287 ± 0.0004 |
| **Yaw-Aug** | 1 | **9.397 ± 0.052** | 1.0855 ± 0.0051 | 42.447 ± 0.379 | 5.031 ± 0.063 | 15.919 ± 0.256 | 23.166 ± 0.135 | 0.3223 ± 0.0004 |
| Vanilla FLAC (P1) | 8 | 8.993 ± 0.011 | 1.0093 ± 0.0035 | 40.650 ± 0.101 | 5.173 ± 0.138 | 15.430 ± 0.197 | 23.409 ± 0.056 | **0.3218 ± 0.0002** |
| Per-angle FA (B-F) | 8 | 8.202 ± 0.017 | **0.9778 ± 0.0015** | **38.793 ± 0.074** | 5.387 ± 0.075 | 16.456 ± 0.038 | **24.198 ± 0.164** | 0.3332 ± 0.0001 |
| **Yaw-Aug** | 8 | **7.965 ± 0.014** | 1.0132 ± 0.0016 | 39.923 ± 0.057 | **5.391 ± 0.097** | **16.563 ± 0.169** | 23.961 ± 0.107 | 0.3247 ± 0.0003 |

## SEEN split, 40k, 5 seeds

| Method | K | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|---|
| Per-angle FA (B-F) | 8 | 5.4848 ± 0.0194 | **0.6272 ± 0.0011** | **26.6944 ± 0.0669** | **6.6656 ± 0.0629** | **18.6778 ± 0.1168** | **26.0125 ± 0.1638** | 0.3217 ± 0.0001 |
| Vanilla FLAC (P1) | 8 | 5.7340 ± 0.0084 | 0.6604 ± 0.0016 | 27.7053 ± 0.0376 | 6.0930 ± 0.1179 | 17.4586 ± 0.1358 | 24.6646 ± 0.1358 | 0.3195 ± 0.0003 |
| **Yaw-Aug** | 8 | **5.353 ± 0.015** | 0.6555 ± 0.0012 | 27.207 ± 0.042 | 6.637 ± 0.091 | 18.610 ± 0.067 | 25.572 ± 0.133 | **0.3131 ± 0.0003** |
| Per-angle FA (B-F) | 1 | **6.5479 ± 0.0609** | **0.6861 ± 0.0055** | **29.4880 ± 0.1926** | **6.4340 ± 0.1452** | 18.1084 ± 0.1574 | **25.2501 ± 0.1673** | 0.3190 ± 0.0001 |
| Vanilla FLAC (P1) | 1 | 6.9327 ± 0.0725 | 0.7256 ± 0.0048 | 30.7589 ± 0.1845 | 5.9418 ± 0.1228 | 17.0275 ± 0.2387 | 24.0952 ± 0.2596 | 0.3164 ± 0.0005 |
| **Yaw-Aug** | 1 | 6.605 ± 0.051 | 0.7158 ± 0.0035 | 30.176 ± 0.148 | 6.366 ± 0.135 | **18.163 ± 0.177** | 24.829 ± 0.172 | **0.3111 ± 0.0002** |

## (c) 45° probe — OFF the C4 orbit (seed 42, unseen)

The augmentation draws uniformly over all 512 columns, so unlike exact-C4
frame-averaging it has no preferred sub-orbit. At 45° — exp_07 A6's negative
control angle, where the vanilla anchor demonstrably degrades:

| K=8 @45° | T60 | EDT | C50 | R@1 |
|---|---|---|---|---|
| Yaw-Aug | 7.996 (C4 band [7.940, 8.014]) | 40.523 (+0.27 above band) | 1.020 (inside) | 5.207 (inside) |
| P1 (5-seed) | 9.715 | 42.390 | 1.071 | 4.081 |

Yaw-Aug at 45° sits essentially inside its own C4 band (worst excursion:
EDT +0.27); P1 at 45° degrades to its usual rotated level (T60 +0.72, R@1
−1.09 vs its 0°). **The flatness is continuous in angle, not a C4 artifact** —
something exact-C4 FA does not guarantee off-orbit.

*Analysis by Claude Fable 5. Caveats: single training seed per arm; 45° probe
single eval seed; global-mean estimand.*
