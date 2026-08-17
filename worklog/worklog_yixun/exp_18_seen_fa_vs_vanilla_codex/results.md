# FA vs Vanilla FLAC on the seen split

## Five-generation aggregate

Values are mean +/- sample standard deviation over evaluation seeds 42--46.
T60, C50, EDT, and FD are lower-is-better; retrieval R@k is
higher-is-better.

| Model | K | T60 | C50 | EDT | R@1 | R@5 | R@10 | FD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA (`exp07_BF` @ 40k) | 8 | 5.4848 +/- 0.0194 | 0.6272 +/- 0.0011 | 26.6944 +/- 0.0669 | 6.6656 +/- 0.0629 | 18.6778 +/- 0.1168 | 26.0125 +/- 0.1638 | 0.3217 +/- 0.0001 |
| Vanilla (`exp07_P1` @ 40k) | 8 | 5.7340 +/- 0.0084 | 0.6604 +/- 0.0016 | 27.7053 +/- 0.0376 | 6.0930 +/- 0.1179 | 17.4586 +/- 0.1358 | 24.6646 +/- 0.1358 | 0.3195 +/- 0.0003 |
| FA (`exp07_BF` @ 40k) | 1 | 6.5479 +/- 0.0609 | 0.6861 +/- 0.0055 | 29.4880 +/- 0.1926 | 6.4340 +/- 0.1452 | 18.1084 +/- 0.1574 | 25.2501 +/- 0.1673 | 0.3190 +/- 0.0001 |
| Vanilla (`exp07_P1` @ 40k) | 1 | 6.9327 +/- 0.0725 | 0.7256 +/- 0.0048 | 30.7589 +/- 0.1845 | 5.9418 +/- 0.1228 | 17.0275 +/- 0.2387 | 24.0952 +/- 0.2596 | 0.3164 +/- 0.0005 |

## Paired difference

The table below is `FA - Vanilla`, formed per matching seed and then
aggregated. Negative is favorable for T60/C50/EDT/FD; positive is favorable
for retrieval.

| K | T60 | C50 | EDT | R@1 | R@5 | R@10 | FD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | -0.2492 +/- 0.0172 | -0.0332 +/- 0.0022 | -1.0109 +/- 0.0697 | +0.5727 +/- 0.1615 | +1.2192 +/- 0.1792 | +1.3479 +/- 0.1971 | +0.0022 +/- 0.0002 |
| 1 | -0.3848 +/- 0.0425 | -0.0395 +/- 0.0039 | -1.2710 +/- 0.0623 | +0.4922 +/- 0.2172 | +1.0809 +/- 0.1673 | +1.1549 +/- 0.3040 | +0.0026 +/- 0.0006 |

FA improves all three acoustic-error metrics and all three retrieval metrics
at both K values. Its FD is slightly worse: +0.0022 at K=8 and +0.0026 at
K=1.

## Protocol and validation

- Full FLAC seen split: 6,217 evaluation positions from 131 rooms.
- K=8 uses `acousticroom_seeneval.json`; K=1 uses
  `acousticroom_seeneval_1.json`.
- Both checkpoints use their EMA weights at `epoch=8-step=40000`.
- FA uses `fa_invariant` conditioning with C4 angles 0/90/180/270 degrees;
  Vanilla uses `vanilla` conditioning.
- Sampling uses `cfg_scale=1.0`, one step, and bf16 conditioning.
- Seeds 42--46 are five evaluation generations from each single training
  checkpoint, not five independently trained checkpoints.
- All 20 result records were checked for the exact checkpoint, seen dataset
  config, seed, conditioning method, EMA source, source revision, 6,217 sample
  count, and finite metric set. Source revision:
  `bd927918c613afcdc5bff5557185f6bb7f6d29b7`.

Machine-readable aggregates and the complete source-file manifest are in
`seen_comparison_summary.json`. Per-run evaluator output is retained under
`logs/`; the 20 raw metric JSON files remain beside their respective 40k
checkpoints under `outputs_FLAC/exp07_BF` and `outputs_FLAC/exp07_P1`.
