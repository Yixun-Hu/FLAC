# Results — exp_02_yaw_noninvariance

**Runs:** 5/5 exit 0 (log `yaw_noninvariance_2026-07-04_20:49:36.log`, 20:49–21:42, ~8–13 min/run).
**Setup:** frozen `FLAC_EMA.ckpt`, pristine `0bd5da0`, full unseen K=1 split (6337 items / 17 rooms), seed 42, steps 1, cfg 1.0, batch 32. Metric-2 JSONs in `metrics_json/`; Metric-1 JSONs `metric1_rot{0,90,180,270}.json`.

## Determinism / pairing control (α = 0)

`yaw_rot0` vs `yaw_baseline`: **identical to the last digit** on all 7 GT metrics, and the comparator reports `max_abs_diff = 0.0` over all 6337×10240 samples. The pipeline is exactly deterministic under fixed seed, and index-pairing across prediction files is valid. Any nonzero number below is caused by rotation alone.

## Metric 2 — accuracy vs GT under rotated conditioning

| run | T60 (%)↓ | C50 (dB)↓ | EDT (ms)↓ | FD↓ | R@1↑ | R@5↑ | R@10↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (α=0) | 9.99 | 1.047 | 40.11 | 0.3027 | 6.71 | 19.00 | 26.89 |
| rot90 | 10.38 | 1.074 | 43.58 | 0.3088 | 6.30 | 17.47 | 25.09 |
| rot180 | **10.72** | **1.189** | **46.39** | 0.3032 | 6.38 | 17.85 | 25.14 |
| rot270 | 10.44 | 1.126 | 44.07 | 0.3088 | 6.11 | 17.80 | 25.33 |

Degradation vs baseline, in units of the exp_01 single-eval noise floor (T60 σ≈0.04, C50 σ≈0.006, EDT σ≈0.37):

- T60: +0.39 / **+0.73** / +0.45 → ~10–18σ
- C50: +0.027 / **+0.142** / +0.079 → ~4–22σ
- EDT: +3.47 / **+6.28** / +3.96 ms → ~9–17σ
- R@1: −0.41 / −0.33 / −0.60 (recall drops at every k and every angle)

180° is the worst angle on all perceptual metrics, consistent with Yixun's prior partial-run observation (EDT ~46.4 at 180°: reproduced here as 46.39 on the full split).

## Metric 1 — invariance gap P_α vs P_0 (no GT involved)

| α | waveform rel-L2 | mean abs diff | T60 gap (%) | C50 gap (dB) | EDT gap (ms) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 90 | 0.221 | 1.38e-4 | 3.34 | 0.563 | 18.65 |
| 180 | 0.193 | 1.21e-4 | 3.33 | 0.550 | 20.11 |
| 270 | 0.214 | 1.35e-4 | 3.41 | 0.600 | 19.98 |

The prediction itself moves by ~20% relative L2 under a conditioning rotation that, physically, should change nothing. The Metric-1 acoustic gaps (T60 ~3.4 %, EDT ~19–20 ms between P_α and P_0) are roughly 5× larger than the net Metric-2 degradation — rotations scatter predictions in both directions around GT, partially cancelling in aggregate accuracy.

## Reference targets for the fix (exp_03+)

A yaw-invariant FLAC must achieve: Metric-1 gaps ≡ 0 at all α (canonicalization gives this by construction for column-quantized angles) while Metric 2 at α=0 stays within ~2σ of baseline (T60 9.99, C50 1.047, EDT 40.11).
