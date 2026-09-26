# exp_24 results — HAA orientation-cue fairness ablation

*Same HAA protocol for every arm (exp_19 registered recipe: AR-40k EMA inits, 1,000 steps, batch 16×accum 4, AdamW 5e-6, seed 42; eval ckpt-1000, test split, K∈{8,1}, 5 eval seeds, each arm on its own conditioning path, bf16, cfg 1.0, one step). Cue = the loudspeaker facing direction appended as a zero-initialised XYZ triple (scale 27) to the geometry-encoder input; on the stock (world-frame) backbone it is a constant field, on CylDINO the gauge makes it column-relative. P1/YAW/YNA/CYL rows are the committed exp_19 records, CYLORI27 is exp_23, P1ORI27 and YAWORI27 are this experiment. Paper convention unless stated.*

Arms with complete records: P1, YAW, YNA, CYL, CYLORI27

## K = 8 (ckpt-1000, paper convention, 5 eval seeds)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 3.4130 ± 0.0127 | 2.2016 ± 0.0123 | 84.994 ± 0.404 | 5.184 ± 0.334 | 19.167 ± 0.239 | 31.693 ± 0.399 | 0.5778 ± 0.0009 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 4.0921 ± 0.0371 | 2.7772 ± 0.0155 | 91.755 ± 0.372 | 4.133 ± 0.176 | 16.100 ± 0.571 | 27.335 ± 0.191 | 0.5887 ± 0.0012 |
| Yaw-Aug init, aug OFF in FT | 3.3910 ± 0.0325 | 2.0956 ± 0.0107 | 77.256 ± 0.285 | 4.761 ± 0.367 | 18.543 ± 0.392 | 30.024 ± 0.304 | 0.5728 ± 0.0010 |
| CylDINO no-SSL (AR-40k→HAA) | 5.4110 ± 0.0230 | 3.4421 ± 0.0209 | 119.502 ± 0.720 | 4.100 ± 0.440 | 16.268 ± 0.238 | 27.652 ± 0.489 | 0.6035 ± 0.0008 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 3.5326 ± 0.0355 | 2.1580 ± 0.0196 | 85.824 ± 0.683 | 4.963 ± 0.253 | 18.769 ± 0.310 | 31.445 ± 0.229 | 0.5814 ± 0.0021 |

## K = 1 (ckpt-1000, paper convention, 5 eval seeds)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 3.6167 ± 0.0585 | 2.2541 ± 0.0161 | 87.950 ± 0.919 | 5.075 ± 0.421 | 19.068 ± 0.698 | 31.223 ± 0.209 | 0.5645 ± 0.0015 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 4.2378 ± 0.0545 | 2.7752 ± 0.0247 | 93.889 ± 0.406 | 4.035 ± 0.426 | 16.269 ± 0.447 | 27.199 ± 0.512 | 0.5751 ± 0.0029 |
| Yaw-Aug init, aug OFF in FT | 3.5361 ± 0.0297 | 2.0999 ± 0.0035 | 79.716 ± 0.689 | 4.611 ± 0.459 | 18.186 ± 0.360 | 30.553 ± 0.706 | 0.5629 ± 0.0012 |
| CylDINO no-SSL (AR-40k→HAA) | 5.5113 ± 0.0472 | 3.4607 ± 0.0220 | 122.045 ± 0.951 | 3.867 ± 0.111 | 16.248 ± 0.411 | 27.433 ± 0.529 | 0.5827 ± 0.0021 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 3.6803 ± 0.0590 | 2.2312 ± 0.0159 | 90.991 ± 0.779 | 4.680 ± 0.409 | 19.156 ± 0.714 | 31.443 ± 0.358 | 0.5655 ± 0.0014 |

## Controlled comparisons (ckpt-1000; first arm relative to second)

- **CYLORI27 vs P1** (CylDINO + cue vs plain FLAC (exp_23 headline)): K=8 T60 +3.5% · C50 -2.0% · EDT +1.0% · R@1 -4.3% · R@10 -0.8%; K=1 T60 +1.8% · C50 -1.0% · EDT +3.5% · R@1 -7.8% · R@10 +0.7%
- **CYLORI27 vs CYL** (the cue's effect on CylDINO): K=8 T60 -34.7% · C50 -37.3% · EDT -28.2% · R@1 +21.0% · R@10 +13.7%; K=1 T60 -33.2% · C50 -35.5% · EDT -25.4% · R@1 +21.0% · R@10 +14.6%

## Per-room (K=8, ckpt-1000, 5 seeds)

### Vanilla FLAC (P1@40k→HAA)

| room | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| classroomBase | 3.975 | 2.130 | 108.70 | 3.84 |
| complexBase | 2.837 | 1.717 | 86.09 | 8.08 |
| dampenedBase | 43.448 | 3.719 | 48.30 | 3.94 |
| hallwayBase | 3.426 | 1.241 | 96.88 | 4.87 |

### Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA)

| room | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| classroomBase | 3.605 | 1.627 | 88.22 | 4.10 |
| complexBase | 2.979 | 1.897 | 85.31 | 7.27 |
| dampenedBase | 60.163 | 5.435 | 72.41 | 1.52 |
| hallwayBase | 5.693 | 2.150 | 121.08 | 3.64 |

### Yaw-Aug init, aug OFF in FT

| room | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| classroomBase | 3.987 | 1.862 | 98.55 | 3.46 |
| complexBase | 3.010 | 1.755 | 76.56 | 7.58 |
| dampenedBase | 47.549 | 3.428 | 46.96 | 3.33 |
| hallwayBase | 3.176 | 1.336 | 86.96 | 4.68 |

### CylDINO no-SSL (AR-40k→HAA)

| room | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| classroomBase | 3.958 | 2.245 | 117.03 | 2.51 |
| complexBase | 3.429 | 2.269 | 112.29 | 9.39 |
| dampenedBase | 69.260 | 5.889 | 76.86 | 0.91 |
| hallwayBase | 8.846 | 3.366 | 171.83 | 3.59 |

### CylDINO no-SSL + orientation field s=27 (AR-40k→HAA)

| room | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| classroomBase | 4.070 | 2.083 | 111.53 | 3.28 |
| complexBase | 2.505 | 1.571 | 79.66 | 8.48 |
| dampenedBase | 51.027 | 3.441 | 44.20 | 2.93 |
| hallwayBase | 4.023 | 1.537 | 107.91 | 5.15 |

## Steps curve (K=8, seed 42, paper convention)

### T60 (%) ↓

| steps | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 8.58 | 4.47 | 3.00 | — | 3.14 | 3.26 | 3.34 | 3.35 | 3.42 | 3.43 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 9.75 | 8.03 | 6.80 | — | 4.93 | 4.56 | 4.27 | 4.21 | 4.10 | 4.15 |
| Yaw-Aug init, aug OFF in FT | — | — | — | — | — | — | — | — | — | 3.43 |
| CylDINO no-SSL (AR-40k→HAA) | — | — | — | — | — | — | — | — | — | 5.43 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 8.56 | 6.42 | 4.70 | — | 3.55 | 3.50 | 3.62 | 3.63 | 3.65 | 3.55 |

### C50 (dB) ↓

| steps | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 4.25 | 3.26 | 2.30 | — | 2.13 | 2.15 | 2.22 | 2.16 | 2.26 | 2.20 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 4.49 | 4.21 | 3.78 | — | 3.02 | 2.88 | 2.82 | 2.82 | 2.83 | 2.78 |
| Yaw-Aug init, aug OFF in FT | — | — | — | — | — | — | — | — | — | 2.09 |
| CylDINO no-SSL (AR-40k→HAA) | — | — | — | — | — | — | — | — | — | 3.47 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 4.47 | 3.70 | 2.85 | — | 2.28 | 2.30 | 2.35 | 2.28 | 2.27 | 2.16 |

### EDT (ms) ↓

| steps | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 181.42 | 115.20 | 81.62 | — | 79.86 | 80.97 | 81.34 | 81.75 | 84.54 | 84.96 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 202.62 | 156.95 | 133.88 | — | 105.44 | 99.94 | 96.42 | 94.89 | 93.60 | 91.98 |
| Yaw-Aug init, aug OFF in FT | — | — | — | — | — | — | — | — | — | 76.99 |
| CylDINO no-SSL (AR-40k→HAA) | — | — | — | — | — | — | — | — | — | 120.38 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 199.83 | 137.27 | 102.12 | — | 86.21 | 86.55 | 89.34 | 87.99 | 87.62 | 85.92 |

## K = 8 (ckpt-1000, pooled convention, 5 eval seeds)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 3.5533 ± 0.0162 | 2.0182 ± 0.0091 | 91.981 ± 0.312 | 4.165 ± 0.305 | 16.162 ± 0.613 | 27.613 ± 0.511 | 0.4379 ± 0.0005 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 4.3053 ± 0.0395 | 2.4294 ± 0.0134 | 96.170 ± 0.552 | 3.510 ± 0.110 | 13.775 ± 0.218 | 23.838 ± 0.152 | 0.4520 ± 0.0005 |
| Yaw-Aug init, aug OFF in FT | 3.4921 ± 0.0317 | 1.9141 ± 0.0081 | 83.360 ± 0.270 | 3.588 ± 0.336 | 15.304 ± 0.278 | 25.757 ± 0.384 | 0.4391 ± 0.0008 |
| CylDINO no-SSL (AR-40k→HAA) | 5.7687 ± 0.0243 | 3.1812 ± 0.0136 | 128.173 ± 0.516 | 3.354 ± 0.335 | 13.635 ± 0.178 | 22.777 ± 0.631 | 0.4574 ± 0.0008 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 3.7657 ± 0.0360 | 2.0336 ± 0.0133 | 95.014 ± 0.523 | 3.900 ± 0.402 | 15.881 ± 0.447 | 26.287 ± 0.475 | 0.4412 ± 0.0009 |

## K = 1 (ckpt-1000, pooled convention, 5 eval seeds)

| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |
|---|---|---|---|---|---|---|---|
| Vanilla FLAC (P1@40k→HAA) | 3.7863 ± 0.0557 | 2.0655 ± 0.0183 | 95.363 ± 1.077 | 4.119 ± 0.432 | 15.819 ± 0.395 | 26.958 ± 0.324 | 0.4315 ± 0.0005 |
| Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA) | 4.4744 ± 0.0600 | 2.4406 ± 0.0190 | 98.462 ± 0.636 | 3.354 ± 0.326 | 13.417 ± 0.357 | 23.573 ± 0.432 | 0.4443 ± 0.0018 |
| Yaw-Aug init, aug OFF in FT | 3.6542 ± 0.0323 | 1.9239 ± 0.0045 | 86.203 ± 0.539 | 3.697 ± 0.368 | 14.758 ± 0.658 | 25.569 ± 0.688 | 0.4330 ± 0.0011 |
| CylDINO no-SSL (AR-40k→HAA) | 5.8891 ± 0.0604 | 3.2036 ± 0.0201 | 130.979 ± 1.008 | 3.104 ± 0.289 | 13.619 ± 0.545 | 22.808 ± 0.433 | 0.4447 ± 0.0016 |
| CylDINO no-SSL + orientation field s=27 (AR-40k→HAA) | 3.9076 ± 0.0679 | 2.0983 ± 0.0182 | 100.507 ± 0.844 | 3.900 ± 0.297 | 15.694 ± 0.492 | 25.850 ± 0.400 | 0.4323 ± 0.0013 |

## LaTeX rows (\bms = best in the K block over the listed arms)

```latex
\FLAC{} & 1 & $\ms{3.617}{0.058}$ & $\ms{2.2541}{0.0161}$ & $\ms{87.950}{0.919}$ & $\bms{5.075}{0.421}$ & $\ms{19.068}{0.698}$ & $\ms{31.223}{0.209}$ \\
Yaw-aug \FLAC{} & 1 & $\ms{4.238}{0.055}$ & $\ms{2.7752}{0.0247}$ & $\ms{93.889}{0.406}$ & $\ms{4.035}{0.426}$ & $\ms{16.269}{0.447}$ & $\ms{27.199}{0.512}$ \\
Yaw-aug init, stock FT & 1 & $\bms{3.536}{0.030}$ & $\bms{2.0999}{0.0035}$ & $\bms{79.716}{0.689}$ & $\ms{4.611}{0.459}$ & $\ms{18.186}{0.360}$ & $\ms{30.553}{0.706}$ \\
\CylDINO{} & 1 & $\ms{5.511}{0.047}$ & $\ms{3.4607}{0.0220}$ & $\ms{122.045}{0.951}$ & $\ms{3.867}{0.111}$ & $\ms{16.248}{0.411}$ & $\ms{27.433}{0.529}$ \\
\CylDINO{} (+facing) & 1 & $\ms{3.680}{0.059}$ & $\ms{2.2312}{0.0159}$ & $\ms{90.991}{0.779}$ & $\ms{4.680}{0.409}$ & $\bms{19.156}{0.714}$ & $\bms{31.443}{0.358}$ \\
\addlinespace[1.5pt]
\FLAC{} & 8 & $\ms{3.413}{0.013}$ & $\ms{2.2016}{0.0123}$ & $\ms{84.994}{0.404}$ & $\bms{5.184}{0.334}$ & $\bms{19.167}{0.239}$ & $\bms{31.693}{0.399}$ \\
Yaw-aug \FLAC{} & 8 & $\ms{4.092}{0.037}$ & $\ms{2.7772}{0.0155}$ & $\ms{91.755}{0.372}$ & $\ms{4.133}{0.176}$ & $\ms{16.100}{0.571}$ & $\ms{27.335}{0.191}$ \\
Yaw-aug init, stock FT & 8 & $\bms{3.391}{0.032}$ & $\bms{2.0956}{0.0107}$ & $\bms{77.256}{0.285}$ & $\ms{4.761}{0.367}$ & $\ms{18.543}{0.392}$ & $\ms{30.024}{0.304}$ \\
\CylDINO{} & 8 & $\ms{5.411}{0.023}$ & $\ms{3.4421}{0.0209}$ & $\ms{119.502}{0.720}$ & $\ms{4.100}{0.440}$ & $\ms{16.268}{0.238}$ & $\ms{27.652}{0.489}$ \\
\CylDINO{} (+facing) & 8 & $\ms{3.533}{0.035}$ & $\ms{2.1580}{0.0196}$ & $\ms{85.824}{0.683}$ & $\ms{4.963}{0.253}$ & $\ms{18.769}{0.310}$ & $\ms{31.445}{0.229}$ \\
```
