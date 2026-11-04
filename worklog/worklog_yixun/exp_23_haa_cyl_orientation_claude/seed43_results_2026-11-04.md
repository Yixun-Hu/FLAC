# exp_23 seed-43 completion — results (2026-11-04, exp_21 session executing under Yixun's 11-03 go)

Runner `resume_seed43_nov03.sh` finished rc=0 at 05:39 EST: CYLORI27_s43 resumed 500→1000 (from the Sep-16 interruption boundary via the Sep-30 NAS symlink), both ckpt-1000 endpoint grids complete (K8/K1 × eval seeds 42–46, 20 cells, 0 failures). Seeds 44–46 NOT run (per Yixun).

## CYLORI27_s43 vs P1_s43 @ ckpt-1000 (mean±std over 5 eval seeds; Δ = CYL−P1)

| K | T60 Δ | C50 Δ | EDT Δ | FD Δ | R@1 Δ |
|---|---|---|---|---|---|
| 8 | **+0.275±0.035** (3.890 vs 3.615) | +0.045±0.012 | **+1.881±0.462** (94.23 vs 92.35) | +0.000 | −0.218±0.392 |
| 1 | +0.160±0.063 | +0.037±0.031 | +2.500±0.952 | −0.001 | −0.265±0.443 |

## Reading

The seed-42 generation's headline ("CYLORI27 reaches PARITY with vanilla on HAA": K8 T60 3.53 vs 3.41, C50 better) **does not fully replicate at training seed 43**: CYLORI27 is systematically slightly behind P1 on T60/C50/EDT (T60 Δ ≈ 8× the paired eval noise at K8; the s42 C50 advantage flips sign). FD and retrieval are ties within noise. Both arms are absolutely worse at s43 than s42 (training-seed variance moves both). Fair two-seed statement: **CYLORI27 is within ~0.1–0.3 T60 of vanilla on HAA across seeds — near-parity, not robust parity.**

## Disclosed asymmetry

CYLORI27_s43 carries an interruption+resume at step 500 (PL restores optimizer/EMA, NOT RNG/dataloader position — non-bit-exact), while P1_s43 trained uninterrupted. Whether the resume contributes to the s43 gap cannot be quantified without a clean rerun; the s42 pair (both uninterrupted) is the cleaner comparison.
