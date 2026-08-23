# exp_21 RESULTS — Mapping-A (unseen-source) cross-arm evaluation on RAF (2026-08-23)

**Protocol:** 1,152 items (16 placements/room × 36 mics × 1 hash-uniform held-out source), K=8 same-mic contexts, canonical generations prepare `0de97c5a1c12` / depth `21a8ec5fc9bd`, ×2.0 scalar (cross-mapping scale disclosed), 5 seeds × 5 arms, all cells from commit `3051f3a`, identity-validated end-to-end (ckpt↔label registry, one item stream, zero invalid/non-finite rows). Statistics: placement-clustered, paired at item×seed; primary + minus-flagged(26) — **zero verdict changes between them**. Full numbers: `mappingA_contrast_report.json` / `mappingA_stats_summary.md`.

## Headline (equal-room macro; ± seed SD; lower better)
| metric | P1 | YAW | BV | BF | finetuned (H-transfer) |
|---|---|---|---|---|---|
| T60 (%) | 19.87 | **19.16** | 22.59 | 19.51 | 19.29 |
| C50 (dB) | 3.81 | 3.96 | 3.92 | 3.61 | **2.90** |
| EDT (ms) | 105.4 | 114.7 | 112.4 | 106.1 | **83.7** |
| mrL1 | 2.889 | 2.860 | 2.932 | 2.925 | **2.851** |
| Env | 0.556 | 0.573 | 0.568 | 0.622 | **0.510** |

## Findings (24/50 pairwise contrasts hold; 10 survive Bonferroni-10)
1. **Unseen-source prediction is the hard task**: T60 19–23% here vs ~10–11% for the same checkpoints under Mapping-H zero-shot (exp_20) — descriptive cross-protocol reference only (item sets differ, per M6), but the gap is the motivating picture.
2. **BV is the decisive T60 loser** — every other arm beats it (−2.7 to −3.4 pts; the study's largest, tightest effects). In the matched twin pair, **BF (FA) beats BV (vanilla twin) on T60 and C50** — the FA conditioning helps unseen-source generalization on real data — while **BF loses Env to all three vanilla arms** (envelope detail cost) and sits with BV on mrL1.
3. **Mapping-H finetuning transfers metric-specifically to unseen sources**: C50 vs every AR arm (+0.71…+1.06 dB), Env vs three of four, EDT vs YAW/BV only, **T60 only vs BV, mrL1 vs nobody** — energy-ratio/envelope structure transfers; reverberation-time accuracy does not (consistent with H-training never generalizing across sources).
4. No arm dominates every metric; seed noise is 2–3 orders below arm separations.

## Caveats (registered)
Within-Mapping-A inference only; multiplicity reported not corrected (Bonferroni flags per row); ×2.0 vs ×3.0 cross-mapping scale disclosed (level-independent metrics unaffected; within-A contrasts share one scale); the 26 flagged items (near-silent references, near-field maps) change no conclusion; 2 rooms, 32 placements — no population-of-rooms claims.
