# Results — exp_05_bn_drift_bisect

## Instrument + landscape (B-1/B0/B1, committed JSONs in this folder)

- Probe: fail-fast clean load (0/0), 20 BNs, no-mutation pinned; repeats tight (±0.001–0.02).
- **B0 baseline (train loader):** all 20 layers over threshold; stem max 0.082 → layer4 0.357 (depth amplification). Eval-loader reference 2–4× worse (probe discriminates).
- **B1 max_len grid:** 4800 → 0.65/1.12; 10240 → 0.12/1.39; 19200 → 0.62/1.68 (stem/all-layer max). **Shipped 9600 is the clear optimum — max_len exonerated.**
- **Dispersion check:** observed max shift 0.085 vs predicted EMA-tail noise 0.024 (≈3.5×) — EMA-tail refuted as sole cause; residual drift real but small; provenance still open (note: train loader applies Random Time Shift + Add Noise augmentations to main audio; context path raw).

## V1′ — BN-frozen vanilla control: **GATE FAIL (registered stop)**

| K | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| 1 | 10.523±0.058 (7.9σ) | **1.010±0.007 (3.7σ BETTER than baseline)** | 41.33±0.12 (3.5σ) | 6.77 (0.24σ PASS) |
| 8 | 9.235±0.005 (48.4σ) | **0.928±0.003 (10.4σ BETTER)** | 38.73±0.01 (24.2σ) | 6.95 (0.67σ PASS) |

## Per-metric damage decomposition (the experiment's central result)

| Metric | R1b (unfrozen FT) | W0 (lr=0, BN only) | V1′ (frozen FT) | Verdict |
|---|---|---|---|---|
| EDT K=1 (baseline 39.95) | 43.27 | 41.10 | 41.33 | **largely BN-mediated; gradient residual ≈ +1.4 ms** |
| C50 K=1 (baseline 1.046) | 1.078 | 1.050 | **1.010** | **BN-mediated; frozen stats + trainable affine IMPROVE clarity** |
| T60 K=1 (baseline 9.97) | 10.47 | 10.13 | 10.52 | **gradient-driven, BN-independent** |
| R@1 (all) | baseline | baseline | baseline | never damaged |

Falsified across exp_03/04/05: Adam transient, EMA-vs-online, batch noise (as sole cause), lr magnitude, BN mutation (as sole cause), max_len. Remaining: **T60-specific gradient-path lineage difference** (objective/data), unreachable by recipe repair from the released artifact.
