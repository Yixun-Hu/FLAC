# exp_20 RESULTS — AR 40k arms, zero-shot on canonical RAF (2026-08-21, 5 seeds, stream-audited)

Protocol: registered in the query file (announcement-05 flags per arm — BF under `fa_invariant`, its training protocol); canonical generations 46a43f4ce82b / a44a723fce4c; 768-item test + 48-item diagnostic; identical item sets and seeds across arms ⇒ within-table comparisons licensed. Zero-shot cross-domain — no comparison to the paper's AR/HAA tables is licensed. Pooled == per-room macro (equal-room design).

## Test row (mean±SD over seeds 42–46)
| Arm (40k) | T60 (%)↓ | C50 (dB)↓ | EDT (ms)↓ | mrL1↓ | Env↓ |
|---|---|---|---|---|---|
| P1 (vanilla) | 10.362±0.016 | 3.099±0.003 | 155.80±0.16 | 3.308 | 0.655 |
| YAW (yaw-aug) | 10.196±0.020 | 3.147±0.005 | 154.81±0.25 | **3.261** | **0.654** |
| **BF (FA)** | **9.644±0.020** | **3.005±0.003** | **146.88±0.18** | 3.347 | 0.725 |
| _context: released FLAC_EMA (87.5k, exp_19)_ | 11.248 | 3.015 | 144.98 | 3.281 | 0.691 |
| _context: RAF-finetuned (exp_19)_ | 5.638 | 0.849 | 38.94 | 2.821 | 0.467 |

## Reading
1. **BF leads the acoustic-parameter metrics zero-shot** (T60 −0.6 to −0.7 pts, EDT −8 to −9 ms vs the vanilla arms; margins ≫ seed noise). YAW edges mrL1/Env. Effects are real but small next to (2).
2. **Finetuning dominates arm choice for RAF transfer**: every arm sits at ~2× the finetuned row's T60/C50 and ~4× its EDT.
3. The 40k arms beat the released 87.5k anchor on zero-shot T60 (10.2–10.4 vs 11.25) but not EDT — longer AR training does not monotonically help real-data transfer.
4. Per-room and diagnostic rows recorded in the raw JSONs (diag is tiny-n as registered; not interpreted). Raw records beside the ckpts under `ar_40k_endpoints/<ARM>/`.
5. BV not evaluated (not requested); one command away.
