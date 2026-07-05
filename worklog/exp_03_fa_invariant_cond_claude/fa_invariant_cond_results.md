# Results — exp_03_fa_invariant_cond (living document; runs in progress)

All numbers: full 6337-item unseen split (announcement 01), commit `992fe49`+ for eval code, seed 42 unless noted. Baseline references from exp_01 (5-seed mean±std).

## Infrastructure verdicts (H1 at the conditioning level — settled before any fine-tune)

- **Conditioning-level invariance (real stack, real weights):** max deviation 4.9e-8 relative on C₄ — float-exact. (Ladder rung a/b diagnosis, notebook.)
- **End-to-end waveform floor:** rel-L2 4.6e-4 (fp32) / ~1.1e-3 (autocast) — entirely decoder amplification (×~1200) of float-level conditioning dust; vanilla model's exp_02 gap is 0.19–0.22, i.e. 200–400× above this floor.
- **Degenerate-source fallback:** load-bearing on real data (11/6337 unseen-eval items, 95/302,925 pairs) and invariance-correct by construction + tests.

## R0 — zero-shot fa_invariant on frozen FLAC_EMA (K=1)

| | T60↓ | C50↓ | EDT↓ | R@1↑ |
|---|---|---|---|---|
| baseline (exp_01, vanilla) | 9.969 | 1.046 | 39.95 | 6.83 |
| **R0 zero-shot fa_invariant** | 10.082 | **1.038** | 42.02 | 5.38 |

Mild OOD degradation (predicted): symmetrized conditioning is off-distribution for the frozen DiT. The fine-tune's target gap: ~0.1 T60 / ~2 ms EDT / ~1.4 pp R@1.

## R1 — vanilla-control fine-tune #1: **GATE FAIL**

Recipe: lr 5e-6 constant, use_ema off, effective batch 8 (4×2), 10 000 opt steps (80k samples), 5 seeds × K∈{1,8}:

| K | T60 | C50 | EDT | vs exp_01 |
|---|---|---|---|---|
| 1 | 10.736±0.069 | 1.101±0.008 | 44.59±0.12 | 9.7σ / 5.2σ / 11.9σ — **FAIL** |
| 8 | 9.614±0.013 | 1.032±0.002 | 42.31±0.05 | 57.6σ / 16.7σ / 62.4σ — **FAIL** |

Per protocol: R2 not launched; H3 unreadable through this control.

## Gate-failure diagnostics

1. **EMA hypothesis (falsified):** true online weights (EMA-stripped `FLAC.ckpt`) score K=1 10.06/1.087/40.71, K=8 8.68/1.011/37.96 — within ~0.1 T60 of the EMA baseline; explains ≤15% of the regression. (First diagnostic attempt was confounded by the eval loader's automatic EMA remap; corrected run documented in the notebook.)
2. **Batch-parity hypothesis (primary):** original training used **effective batch 128** (README: 32 × accum 2 × 2 GPUs) vs our 8 — 16× gradient-noise mismatch, invisible to the config parity audit (CLI-side parameter).

## R1b — amended single iteration: batch-parity control (RUNNING)

Effective batch 128 (4 × accum 32), 625 opt steps (identical 80k-sample budget), lr 5e-6, all else unchanged. Gate: same pre-registered 2σ criteria. If FAIL → registered stop: no R2, analysis instead.

*(sections below fill in as runs complete)*

## R2/R3 — fa_invariant fine-tune + evals (H3) — pending R1b gate
## R4/R4b — rotation sweeps, Metric 1 + H2 — pending R2
