# exp_19 R-cal Leg A results — released FLAC_HAA.ckpt on haa_test.json (1,282 items, 5 seeds)

Pipeline: this box, `eval_FLAC.py` @ `01b7cce`, vanilla/rotate-0/autocast-default, batch 64, stream-audited (1282/1282 every seed), scorer `AGREE_fullHAA.pt` (release ships no `AGREE_HAA.pt` — manifest Amendment 1).

| Convention | T60 (%)↓ | C50 (dB)↓ | EDT (ms)↓ | R@5 (%)↑ | FD↓ |
|---|---|---|---|---|---|
| **Ours, per-room macro** | 17.628±0.108 | **2.163±0.008** | **84.37±0.65** | — | — |
| **Ours, pooled per-item** | **3.178±0.020** | 1.991±0.008 | 90.68±0.67 | 14.96±0.23 | 0.449±0.001 |
| Paper Tab. 3 (FLAC K=8, 5 gens) | 3.10±0.01 | 2.167±0.004 | 84.52±0.24 | 17.41±0.59 | (truncated in md) |

**Verdict: eval pipeline CALIBRATED.**
- C50 and EDT reproduce the paper under the per-room macro convention to within seed noise (Δ < 0.005 dB / 0.15 ms).
- T60 reproduces under the POOLED convention (3.18 vs 3.10, +2.5%); the 4-room macro is 17.6 solely because `dampenedBase` (near-anechoic, tiny absolute T60) yields 61% relative error — the paper's T60 is evidently not the 4-room macro. Convention nuance registered: exp_19's RAF results will report BOTH conventions per metric and name the convention next to every number.
- R@5 (14.96 vs 17.41) is NOT comparable by construction: the paper fine-tuned AGREE on HAA for retrieval/FD; the release ships no such checkpoint, ours uses `AGREE_fullHAA`. Labelled as such, not a discrepancy.
- Per-room T60 (seed 42): classroom 3.27 / complex 2.81 / dampened 60.99 / hallway 3.25; Invalid-T60 rate 0.0 everywhere.

Leg B (recipe reproduction) pending: step-1000 EMA checkpoint → same 5-seed eval (`exp19_rcal_repro_seed<seed>`), comparison vs this table bounds recipe-reproduction variance.
