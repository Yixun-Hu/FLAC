# Yixun's queries — exp_01_reproduce_flac_table1

## Query 1 (2026-07-04)

### Verbatim

> Great, here comes to our first experiment, reproduce FLAC's results in @FLAC_pdf.md's table 1's FLAC K=1, K=8 with Geometry condition.

### Summary

Run the released FLAC checkpoint through the paper's evaluation protocol on the unseen AcousticRooms split and reproduce the two geometry-conditioned FLAC rows of Table 1:

| Row | K | G | T60 (%)↓ | C50 (dB)↓ | EDT (ms)↓ | R@1↑ | R@5↑ | R@10↑ |
|---|---|---|---|---|---|---|---|---|
| FLAC | 1 | ✓ | 9.95 ± 0.05 | 1.046 ± 0.002 | 40.04 ± 0.22 | 6.80 ± 0.11 | 18.92 ± 0.10 | 26.87 ± 0.19 |
| FLAC | 8 | ✓ | 8.60 ± 0.01 | 0.970 ± 0.002 | 37.13 ± 0.02 | 6.99 ± 0.13 | 19.38 ± 0.15 | 27.21 ± 0.17 |

### Assumption / hypothesis

The released `FLAC_EMA.ckpt` + the repo's eval pipeline + the full unseen split reproduce the paper's Table 1 numbers within the reported standard deviations. Everything needed (checkpoint, configs, AGREE metric weights) ships in the repo.

### Why this experiment needs to run

This is the calibration anchor for the whole equivariance project. Every subsequent experiment (recipe-repaired vanilla control, canonicalization fine-tune, any method that tries to beat Table 1) will be judged against these numbers, so we must first confirm our local pipeline reproduces the published baseline — otherwise any later "improvement" or "regression" is uninterpretable. It also establishes the run-to-run noise floor (std over 5 generations) that defines how big a future delta must be to be meaningful. Per announcement 01, this must run on the full eval configuration (6337 items / 17 unseen rooms), not a subset.
