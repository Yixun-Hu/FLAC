# Verdict: BLOCK

## 1. Numbers

- Gate recomputation, sample SD (`ddof=1`):

| K | T60 | C50 | EDT | R@1 |
|---|---:|---:|---:|---:|
| 8 | 8.46518±0.005822 | 0.95822±0.000968 | 37.49684±0.081260 | **6.92438±0.070040** |
| 1 | 9.82708±0.061231 | 1.03368±0.002453 | 40.87398±0.339350 | 6.85814±0.110825 |

- Thus K8 R@1 should round to **6.9244±0.0700**, not `6.9243±0.0701` in [results:18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_results.md:18), HTML, and generator.
- Exact exp_01 combined-σ results: K8 `−10.7649 SUP / −3.2031 SUP / +3.7743 OUT / −1.0724 NONINF`; K1 `−1.9618 SUP / −1.7941 SUP / +1.8367 NONINF / +0.1172 EQUIV`. Tiers are correct; K8 EDT should display **+3.8σ**, not `+3.7σ`, when using exact exp_01 values.
- Rotation JSONs match the table, but spreads are `0.0011/0.0001/0.0072/0.0158`; “identical to 3–4 decimals” is false, visibly for R@1 in [results:9](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_results.md:9).
- Fw and V screen values match raw JSONs.

## 2. Tier and claim scope

- **PARTIAL is not the pre-registered tier.** No checkpoint qualifies under [plan:22](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:22); G2 therefore FAILS. [Plan:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:32) makes that **NEGATIVE**, while PARTIAL requires G2≤2σ in [plan:31](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:31).
- Five-seed Fw95 vs anchor z-scores are K8 `+14.35/−4.36/+15.91/−0.23`; K1 `+4.39/+0.21/+5.26/+0.23`. G2 fails T60 and EDT at both K.
- G3 is **4 SUPERIOR + 1 EQUIV + 2 NONINF + 1 OUT**, only 5/8 SUPERIOR-or-EQUIV. “Released-Table-1-level,” “R@1 matched,” and “single concession” in [analysis:7](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_analysis.md:7) overstate the registered reading. Say: **one OUT cell, two additional NONINF-but-not-EQUIV cells**.
- `G4 ΔΔ≈0` is unsupported. Mean matched-step `F−V` is `−0.7109/+0.0147/+0.7299/+0.1263`; F95 versus the fixed V-window mean is `−0.8166/+0.0139/−0.9158/+0.4024`. Report the per-metric statistic; G4 cannot override G2, explicitly per [plan:26](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:26).
- “All within V’s band” in [results:30](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_results.md:30) is literally false: F95 is inside only for C50/EDT; it is better than the V range for T60/R@1.
- exp_03–06 tested the released-checkpoint lineage; exp_09 starts from a new exp_07 anchor with warm optimizer state. Therefore [analysis:26](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_analysis.md:26) and [HTML:22](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_01_results.html:22) do not reverse that blocker. Scope to: **the large apparent exp_09 monotone damage was an eval mismatch; strong-anchor FA fine-tuning has a much smaller, mixed delta.**

## 3. Protocol

- The s42→43–46 split was mechanically followed, but 95k is confirmation of an **unregistered fallback**, because the registered selector returned no candidate. [Analysis:12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_analysis.md:12) should not call it protocol-valid held-out confirmation.
- The architectural note does **not** satisfy pre-registered G1. [Plan:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:23) requires conditioning rel-L2, fresh-floor fixed-noise waveform evidence, and metric sweeps at both K. Current evidence is only K8 metrics at 88750. Run the full G1 block at 95000 if retaining “exactly equivariant Fw-95000”; otherwise downgrade to architecture-level evidence and record the protocol departure.
- One training seed is disclosed in results/analysis; add it to the self-contained HTML.
- Protocol-error chronology is honest and prominent. However [analysis:18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/fa_finetune_analysis.md:18) claims a B-F-40k FA spot-check was appended; no such artifact/result exists. Remove that sentence or produce and record the spot-check.

## 4. HTML/SOP

- HTML mirrors `results.md`, including its rounding and claim errors, but omits the decisive strict-G2 failure and exact G4 statistic.
- [Plan:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_09_fa_finetune_claude/plan_fa_finetune.md:3) still says “AWAITING approval.”
- Worklog ends before the closing action; commits log omits closing commit `08cf084`.
- Save this closure review, apply/reverify fixes, and log the loop. The generator is executable code and needs recorded review coverage under [SOP:66](/home/yixunhu/codespace/FLAC/worklog/experiment_SOP.md:66).

No files were changed.