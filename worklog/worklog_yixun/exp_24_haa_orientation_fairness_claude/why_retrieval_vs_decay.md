# Why CylDINO + cue wins retrieval and loses the decay metrics (HAA, ckpt-1000, K = 8, seed 42)

*Per-sample acoustic parameters of each arm's stored predictions and of the ground truth, recomputed with the evaluator's own code on its 9,600-sample crop; signed error e = pred − gt.*

## T60 (s)

| arm | GT mean ± sd | pred mean | MAE | bias | scatter (sd of e) | bias share of MSE | MAE after removing the bias | Pearson r | Spearman ρ |
|---|---|---|---|---|---|---|---|---|---|
| FLAC | 0.636 ± 0.146 | 0.641 | 0.042 | +0.005 | 0.083 | 0 % | 0.043 | 0.826 | 0.724 |
| FLAC+cue | 0.636 ± 0.146 | 0.644 | 0.042 | +0.008 | 0.085 | 1 % | 0.044 | 0.814 | 0.715 |
| CylDINO+cue | 0.636 ± 0.146 | 0.644 | 0.046 | +0.008 | 0.091 | 1 % | 0.047 | 0.787 | 0.697 |

## C50 (dB)

| arm | GT mean ± sd | pred mean | MAE | bias | scatter (sd of e) | bias share of MSE | MAE after removing the bias | Pearson r | Spearman ρ |
|---|---|---|---|---|---|---|---|---|---|
| FLAC | 3.744 ± 8.591 | 4.949 | 2.017 | +1.205 | 2.697 | 17 % | 1.868 | 0.959 | 0.918 |
| FLAC+cue | 3.744 ± 8.591 | 4.719 | 1.730 | +0.976 | 2.250 | 16 % | 1.643 | 0.970 | 0.938 |
| CylDINO+cue | 3.744 ± 8.591 | 5.085 | 2.031 | +1.341 | 2.684 | 20 % | 1.850 | 0.957 | 0.926 |

## EDT (ms)

| arm | GT mean ± sd | pred mean | MAE | bias | scatter (sd of e) | bias share of MSE | MAE after removing the bias | Pearson r | Spearman ρ |
|---|---|---|---|---|---|---|---|---|---|
| FLAC | 788.972 ± 483.545 | 729.766 | 91.897 | -59.206 | 120.780 | 19 % | 81.141 | 0.969 | 0.945 |
| FLAC+cue | 788.972 ± 483.545 | 734.653 | 82.613 | -54.319 | 109.362 | 20 % | 75.790 | 0.974 | 0.956 |
| CylDINO+cue | 788.972 ± 483.545 | 719.107 | 94.983 | -69.865 | 116.203 | 27 % | 83.198 | 0.971 | 0.951 |

## Ranking is bias-invariant, absolute error is not

Nearest-neighbour retrieval of each prediction's own ground truth in the 3-D (T60, C50, EDT) space, z-scored in GT statistics — a stand-in for the evaluator's AGREE-embedding ranking, computed from the same quantities as the table above, before and after subtracting each arm's own global bias:

| arm | R@1 | R@5 | R@10 | R@1 after bias removal | R@5 | R@10 |
|---|---|---|---|---|---|---|
| FLAC | 0.9 | 4.4 | 10.0 | 1.3 | 4.8 | 9.0 |
| FLAC+cue | 0.9 | 5.4 | 10.6 | 1.0 | 6.1 | 9.8 |
| CylDINO+cue | 1.3 | 4.4 | 9.3 | 0.6 | 4.3 | 8.8 |

## Per room (signed bias / scatter / Spearman ρ)

### T60 (s)

| arm | classroomBase | complexBase | dampenedBase | hallwayBase |
|---|---|---|---|---|
| FLAC | -0.02 / 0.03 / -0.02 | -0.00 / 0.03 / 0.66 | +0.05 / 0.19 / 0.71 | +0.02 / 0.03 / 0.88 |
| FLAC+cue | -0.02 / 0.03 / 0.08 | -0.00 / 0.03 / 0.65 | +0.07 / 0.19 / 0.74 | +0.01 / 0.03 / 0.86 |
| CylDINO+cue | -0.02 / 0.03 / -0.03 | -0.00 / 0.02 / 0.73 | +0.07 / 0.20 / 0.67 | +0.01 / 0.04 / 0.86 |

### C50 (dB)

| arm | classroomBase | complexBase | dampenedBase | hallwayBase |
|---|---|---|---|---|
| FLAC | +1.44 / 2.34 / 0.64 | -0.15 / 2.67 / 0.80 | +3.01 / 3.94 / 0.79 | +0.73 / 1.70 / 0.92 |
| FLAC+cue | +0.89 / 2.23 / 0.74 | +0.88 / 1.80 / 0.89 | +2.07 / 3.30 / 0.85 | +0.60 / 1.63 / 0.92 |
| CylDINO+cue | +1.58 / 2.33 / 0.66 | +0.72 / 1.96 / 0.90 | +2.62 / 4.11 / 0.76 | +0.77 / 2.22 / 0.91 |

### EDT (ms)

| arm | classroomBase | complexBase | dampenedBase | hallwayBase |
|---|---|---|---|---|
| FLAC | -82.84 / 114.11 / 0.66 | +10.18 / 141.25 / 0.80 | -32.69 / 65.58 / 0.72 | -78.24 / 123.05 / 0.88 |
| FLAC+cue | -56.67 / 107.42 / 0.75 | -44.06 / 102.46 / 0.88 | -16.50 / 50.70 / 0.83 | -74.25 / 128.24 / 0.86 |
| CylDINO+cue | -92.58 / 118.47 / 0.65 | -36.05 / 100.43 / 0.92 | -26.03 / 63.92 / 0.69 | -81.35 / 129.89 / 0.86 |
