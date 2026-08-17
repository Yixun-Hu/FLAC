# exp_17 — C4 rotation grid RESULTS (128/128 cells, 0 failures)

**Grid:** 16 checkpoints (2,500→40,000) × {0°,90°,180°,270°} × {K=1,K=8}, full
published unseen split (6,337 items / 17 rooms), EMA, `--cond-method vanilla
--cond-autocast bf16 --cfg-scale 1.0 --steps 1 --seed 42`. Finished 2026-08-16
20:19:13 EDT, ~7.5 h on 2×A6000. Estimand: GLOBAL (sample-weighted) means —
the same flat key every `model_comparison.md` raw JSON carries. Single eval
seed (42); single training seed (42) per arm.

Full 32-orbit table: `results_yaw_aug_a6000_c4_grid_table.md` (regenerate any
time with `python -m src.tools.exp17_rotation_table --dir outputs_FLAC/exp17_YAWAUG_roteval`).

## Headline: matched-step endpoint head-to-head vs P1 vanilla @40k

The P1 control orbit at 40k ALREADY EXISTS at 5 eval seeds under the identical
protocol (0° = `exp07_P140` rows; 90° = exp_07 A6 `a6_VAN40`; 180°/270° =
`a6c4_VAN40`; every file protocol-checked vanilla+bf16). No GPU time was spent.

**C4 spread (max−min over the four angles) — the rotation-robustness number:**

| | ΔT60 | ΔC50 | ΔEDT | ΔFD | ΔR@1 |
|---|---|---|---|---|---|
| P1@40k (5-seed means), K=8 | 0.897 | 0.133 | **6.738** | 0.007 | 0.458 |
| **Yaw-Aug@40k (s42), K=8** | **0.075** | **0.013** | **0.349** | 0.001 | 0.237 |
| P1@40k (5-seed means), K=1 | 0.903 | 0.133 | 6.657 | 0.006 | 0.508 |
| **Yaw-Aug@40k (s42), K=1** | **0.061** | **0.011** | **0.384** | 0.001 | 0.047 |

**12× flatter on T60, 19× flatter on EDT, both K.** P1 degrades monotonically
away from its training frame (K=8: T60 8.993→9.889, EDT 40.65→47.39 at 180°);
Yaw-Aug is statistically flat at every angle and every checkpoint — including
step 2,500, i.e. the model never acquires a preferred orientation at all.

**Absolute quality at 0°, matched 40k steps (K=8):**

| | T60↓ | C50↓ | EDT↓ | FD↓ | R@1↑ |
|---|---|---|---|---|---|
| P1@40k (5 seeds) | 8.993±0.011 | 1.009±0.004 | 40.650±0.101 | — | 5.173±0.138 |
| Yaw-Aug@40k (s42) | **7.948** | 1.014 | **39.908** | 0.325 | **5.349** |

At its own training frame P1 loses T60 by ~1.05 to the augmented arm — larger
than the ±0.5 checkpoint band, and the neighbouring Yaw-Aug draws (35k 8.467,
37.5k 8.885) stay below P1's band too. C50 is tied; EDT and R@1 slightly favour
Yaw-Aug. **At any non-zero angle Yaw-Aug wins every metric by a wide margin.**
This matches the plan's pre-registered R1 reading: the augmentation acts as a
regulariser as well as an invariance mechanism.

**Context vs the FA arm** (B-F@40k under its own fa eval: 8.202/0.978/38.79/R@1
5.39): Yaw-Aug beats it on T60, is slightly behind on C50/EDT, ties R@1 — while
achieving comparable C4 flatness by data alone, with no inference-time
frame-averaging cost. Untested here by design: intermediate angles (45°). The
augmentation is uniform over all 512 columns, so it MAY be flat off the C4
orbit where exact-C4 FA is not — that would need a 45° probe cell to claim.

## Caveats (standing)
Single training seed per arm; Yaw-Aug side single EVAL seed (P1 side 5);
endpoint values are draws from the InverseLR oscillation band; global-mean
estimand, not per-scene; matched steps ≠ matched compute (augmentation adds no
step cost, so these ARE matched-compute at equal steps, unlike the FA arms).

*Analysis by Claude Fable 5 (main session seat). Grid tooling reviewed by
OpenAI Codex `gpt-5.6-sol` (roteval r1 + p1ctrl r1/r2, all blockers closed).*
