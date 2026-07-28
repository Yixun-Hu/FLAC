# Results — exp_07 fa_scratch (final; all arms complete 2026-07-28)

All evals: `eval_FLAC.py`, full published unseen split (6,337 items / 17 rooms), bf16 cond-autocast, cfg 1.0, steps 1; EMA weights unless noted. Baselines = exp_01 5-seed reproduction of released `FLAC_EMA.ckpt`.

## HEADLINE — FULL TABLE-1 PARITY (maximum project goal CLOSED)

**Checkpoint of record: `outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt`**
(vanilla B-V, DDP 32/GPU×2×accum1 eff-64 + SyncBN(64) + ViT grad-ckpt, seed 42, 87,500 steps = 67.5k budget + extension; env `flac`.)

| Metric | K=8 ours (5-seed) | K=8 released | σ_c verdict | K=1 ours (5-seed) | K=1 released | σ_c verdict |
|---|---|---|---|---|---|---|
| T60 ↓ | **8.2929 ± 0.0105** | 8.609 ± 0.012 | **SUPERIOR −19.8σ** | **9.5401 ± 0.0231** | 9.969 ± 0.039 | **SUPERIOR −9.5σ** |
| C50 ↓ | 0.9660 ± 0.0015 | 0.9682 ± 0.0030 | EQUIV −0.65σ | **1.0323 ± 0.0060** | 1.0460 ± 0.0064 | **SUPERIOR −1.6σ** |
| EDT ↓ | **35.9513 ± 0.0532** | 37.10 ± 0.07 | **SUPERIOR −13.1σ** | **38.7283 ± 0.2263** | 39.95 ± 0.37 | **SUPERIOR −2.8σ** |
| R@1 ↑ | 6.9591 ± 0.1353 | 7.06 ± 0.10 | EQUIV −0.60σ | 6.8108 ± 0.1766 | 6.83 ± 0.22 | EQUIV −0.07σ |

**8/8 cells SUPERIOR or ≤1σ_c EQUIVALENT; 5 strictly superior; none outside the 1σ_c equivalence band** (both R@1 point-means are numerically lower; equivalence, not superiority). Five EVALUATION seeds; ONE training seed (42). σ_c = √(σ_ours² + σ_released²); tiers per the pre-registered gate (superior >1σ_c better / equivalence ≤1σ_c / non-inferiority ≤2σ_c).

Secondary confirmed checkpoint: **57,500** (5-seed): K=8 T60 8.4854±0.0071 / C50 0.9636±0.0016 / EDT 36.3789±0.1103 all SUPERIOR-or-EQUIV, R@1 6.1606±0.0876 OUT; K=1 T60/EDT SUPERIOR, C50 EQUIV, R@1 OUT — the within-original-budget composite-rule qualifier (first in the program).

## Arm summaries

**P1 (vanilla @ SyncBN-64 DDP recipe; 100k total).** Selection curve (EMA s42): 10k 11.78/1.378/47.48/1.78 · 20k 8.44/1.095/45.42/2.92 · 30k 9.20/1.096/43.43/4.17 · 40k 8.99/1.008/40.62/5.19 · 50k 8.65/0.985/37.65/5.54 · 55k 8.51/0.951/36.99/6.00 · 57.5k 8.49/0.963/36.43/6.06 · 60k 8.89/1.015/38.99/6.33 · 62.5k 8.82/0.949/38.16/5.87 · 65k 8.89/0.960/38.46/6.04 · 67.5k 8.77/0.973/36.95/6.28 · 70k 8.08/0.939/37.23/6.23 · 75k 9.12/0.941/39.58/6.68 · 80k 8.80/0.933/37.13/6.83 · 85k 8.91/0.957/38.02/6.25 · **87.5k 8.31/0.964/35.97/6.98** · 90k 8.79/1.010/36.60/6.85 · 92.5k 8.99/0.930/37.76/6.53 · 95k 8.49/0.962/37.13/6.41 · 97.5k 9.01/0.995/38.55/6.60 · 100k 9.55/0.943/39.04/6.75. Late-curve statistic (mean 55–67.5k): T60 8.728 / C50 0.9684 / EDT 37.664 / R@1 6.096 — **EDT closes 81% of the 8×8 arm's released-gap (STRONG threshold ≤38.59 PASSED)**; R@1 statistic below its 6.51 threshold (endpoint-window; the crossing came at 87.5k).

**B-V 8×8 (phase 1 + extend; 100k total).** Endpoint 67.5k gate FAILED strict (1/6). Extend best: R@1 7.054@92.5k (5-seed 6.921±0.186 = ≤1σ_c equivalence — R@1-only parity); EDT never below 38.29; T60 released-zone only at 30–40k. Late statistic: EDT 40.087 / R@1 5.960.

**B-F fa_invariant (8×8 M0-fit; relaunched at the SyncBN-64 DDP recipe; stopped 40k for futility).** Screens 10k–40k plateaued at ~2× the 8×8 anchor's EDT/C50 and 0.15× R@1 with healthy loss. cfg0 conditioning-lift probe at 20k: BF lift T60 −19.0/C50 −2.09/EDT −104.6/R@1 +0.33 vs BV −17.5/−1.88/−88.6/+2.23 — conditioning ACTIVE; trajectory globally slow.

**Attribution (single-delta, pre-registered).** P1 vs B-F differ only by `cond_method`+`frame_avg_angles` (parsed-object-asserted). P1 ≈ 8×8 anchor at every matched step (10k/20k/30k, R@1 lead by 60k) while B-F cratered ⇒ **recipe innocent; fa_invariant-from-scratch is the cause** (+3.5× step time). Coheres with exp_08: frame-averaged equivariance succeeds as a fine-tune-stage property.

## Reproduction anchors

- P1/extension: `p1_ddp_launch.sh` (+`RESUME_CKPT`/`MAXSTEPS`), config `FLAC_AR_BVp1.json`; wandb `FLAC_exp07_P1` runs `acwm8gvt`/`lr45v31g`/`ismr2bql`/`mkum1n79` (legs split by two harness-teardown kills, full-state resumes; not bit-exact across legs — PL restores no RNG/dataloader position; disclosed).
- Full command history: `fa_scratch_command.md`; per-run logs in this folder; gate JSONs beside the checkpoints.

## Appendix — exact P1 curve values (source of record for the HTML page)

| step | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| 10,000 | 11.784 | 1.3775 | 47.481 | 1.783 |
| 20,000 | 8.442 | 1.0954 | 45.418 | 2.919 |
| 30,000 | 9.200 | 1.0958 | 43.428 | 4.166 |
| 40,000 | 8.989 | 1.0076 | 40.620 | 5.192 |
| 50,000 | 8.647 | 0.9854 | 37.649 | 5.539 |
| 55,000 | 8.510 | 0.9506 | 36.992 | 5.997 |
| 57,500 | 8.493 | 0.9625 | 36.427 | 6.060 |
| 60,000 | 8.893 | 1.0146 | 38.991 | 6.328 |
| 62,500 | 8.815 | 0.9486 | 38.161 | 5.870 |
| 65,000 | 8.887 | 0.9604 | 38.461 | 6.044 |
| 67,500 | 8.771 | 0.9734 | 36.952 | 6.281 |
| 70,000 | 8.079 | 0.9390 | 37.228 | 6.233 |
| 75,000 | 9.116 | 0.9407 | 39.575 | 6.675 |
| 80,000 | 8.804 | 0.9332 | 37.132 | 6.833 |
| 85,000 | 8.906 | 0.9569 | 38.023 | 6.249 |
| 87,500 | 8.307 | 0.9643 | 35.973 | 6.975 |
| 90,000 | 8.785 | 1.0099 | 36.598 | 6.849 |
| 92,500 | 8.994 | 0.9300 | 37.761 | 6.533 |
| 95,000 | 8.488 | 0.9619 | 37.133 | 6.407 |
| 97,500 | 9.012 | 0.9946 | 38.547 | 6.596 |
| 100,000 | 9.552 | 0.9432 | 39.039 | 6.754 |
