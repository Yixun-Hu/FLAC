**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-21

*Round marker: r4m5 (R4 aggregation/report). REQUEST-CHANGES — headline arithmetic TRUSTWORTHY (q1–q4), promotion blocked on 6 findings (Δ=0+oracle inclusion; q6 rewording; assembly gates; ddof; q3 rooms; 2 LOW). Body verbatim.*

---

# Verdict: REQUEST-CHANGES

Do not promote the PRELIMINARY report to exp_18 R4 results yet. The supplied artifacts’ headline arithmetic is trustworthy, but the report is contract-incomplete, its input gate is not fail-closed, seed SD is calculated with the wrong convention, and q6 overclaims what the rescue/union analysis establishes.

## Findings

1. **HIGH — Required §3 outputs are absent.**

   The mandated Δ=0 alignment sensitivities are recorded but excluded from `REPORT_FAMILIES`; `ALIGNMENT_SENSITIVITY_FAMILIES` is never consumed. [metrics_report.py:27](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:27)

   Their omission is material:

   - M1 Δ=8 macro top-1: 0.5928; Δ=0: **0.4354**
   - M5 Δ=8 macro top-1: 0.5280; Δ=0: **0.2844**

   The measured-candidate oracle ceiling required by §3 is also absent. The report CLI accepts unseen and seen streams only, with no oracle/control input. [plan_loc_invert_R4.md:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert_R4.md:23), [eval_localization.py:1124](/home/yixunhu/codespace/FLAC/eval_localization.py:1124) At review time its only artifact remained `.partial` at 840/6,337 rows.

2. **HIGH — q6’s “adds information” verdict is not justified.**

   The rule returns true whenever a family rescues at least one AGREE error, making the oracle union necessarily better. [metrics_report.py:794](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:794), [metrics_report.py:828](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:828)

   That proves complementary errors or oracle-fusion potential. It does not prove:

   - a realizable fusion can choose the correct scorer per query;
   - information beyond the shared generated waveform exists; or
   - improvement beyond what another sufficiently different scorer would rescue.

   q6 must be changed from “adds information: yes” to **“complementary scoring signal observed; added information not established.”**

3. **HIGH — Report assembly is not fail-closed against the frozen experiment.**

   `iter_joined` checks only same-position `query_id`; it does not bind seed, context-stream digest, room, GT/candidate order, context membership, registration SHA, or summary provenance. [metrics_report.py:416](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:416) Because all seeds share query IDs, seed-42 metrics paired with seed-43 replay rows would pass this check while combining metric distances with the wrong context geometry.

   The driver merely accepts operator-supplied seed labels and file paths. [eval_localization.py:3534](/home/yixunhu/codespace/FLAC/eval_localization.py:3534) It neither reads the frozen manifest nor requires exactly seeds 42/43/44, five primaries, 6,337 unique identities, and ten primary comparisons. `family_scores` also silently re-derives an aggregation when the recorded block is missing instead of refusing. [metrics_report.py:75](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:75)

   The actual supplied pairs are correct, but the executable gate must enforce that fact.

4. **MEDIUM — Seed SD uses population SD, contrary to exp_18’s published convention.**

   `_mean_sd` calls `np.std` with `ddof=0`. [metrics_report.py:574](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:574) Exp_18’s prior three-seed figures use sample SD (`ddof=1`), e.g. R2’s published 0.0008 and R2b’s 0.0032. [loc_invert_results.md:46](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_results.md:46)

   Correct primary pooled top-1 SDs are:

   | Family | Reported | Correct sample SD |
   |---|---:|---:|
   | M1 | 0.000465 | 0.000569 |
   | M2 | 0.001435 | 0.001757 |
   | M3 | 0.001519 | 0.001860 |
   | M4 | 0.000932 | 0.001142 |
   | M5 | 0.001271 | 0.001557 |

   No conclusion changes at the 0.01 q3 threshold.

5. **MEDIUM — q3 reports 51 “rooms,” and seen-vs-unseen silently selects the first seed.**

   q3 appends 17 rooms from each seed and labels the resulting 51 seed-room cells as `n_rooms`. [metrics_report.py:705](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:705) Aggregating each physical room across seeds gives 17 rooms. The corrected agreement rates are M1 0.4706, M2 0.4706, M3 0.8824, M4 0.6471, M5 0.4118; the current verdicts do not change.

   Seen-vs-unseen takes `blocks[0]`, making the comparison depend on CLI input order. [metrics_report.py:868](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:868) The present run happened to select seed 42, which is defensible against seen seed 42, but it must be explicitly labelled or replaced with a three-seed unseen summary.

6. **LOW — Two secondary diagnostics need correction.**

   - Masked-retrieval MRR ranks the GT among all candidates, including masked-ineligible candidates, although prediction is restricted to the eligible set. [metrics_report.py:128](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:128)
   - `M4Accumulator` ignores the recorded feature mask and skips only non-finite columns. [metrics_report.py:267](/home/yixunhu/codespace/FLAC/src/localization/metrics_report.py:267) All current seen/unseen rows report zero drops, so current M4 results are unaffected.

## Confirmed trustworthy

- Full-stream audit found 6,337 unique queries and 17 rooms per seed, with zero supplied-pair geometry/context mismatches.
- The actual row configs and registerable payloads match the frozen manifest and registration SHA.
- Stored mean-over-K predictions match every recorded `pred_index`; secondaries are labelled.
- R4 replay fields reproduce the corresponding published R2 streams exactly for all seeds.
- Metric-matched retrieval uses `nearest_context_baseline` with correct context selection and eligible masking for prediction.
- R4 seed 42 reproduces R‑1b exactly: macro 0.688997 and pooled 0.631687. The reported 0.6873/0.6303 are three-seed averages; the small difference comes from seeds 43/44’s distinct context draws, not a convention error.
- The top-1 indicator reuse is statistically coherent as a paired mean difference with whole-room cluster resampling. Holm covers exactly ten primary comparisons per seed. Reported `p=0` should be read as zero empirical tail hits in 10,000 resamples, not a literal exact probability.

## q1–q6 trust statements

- **q1 — NUMBERS TRUSTWORTHY.** All five fail the registered macro > 0.689 rule. M2 does exceed the recomputed pooled reference, 0.7240 versus 0.6303, and that convention reversal is correctly disclosed.
- **q2 — NUMBERS TRUSTWORTHY.** All five are `beats=false` under the disclosed strict rule. M1, M2, and M5 have positive pooled deltas, but none passes per-seed Holm.
- **q3 — NUMBERS TRUSTWORTHY WITH CAVEAT.** Correct to 17 physical rooms and sample SD. Verdicts remain: only M3 is consistent—and it is consistently worse; M1/M2/M4/M5 are not room-consistent.
- **q4 — NUMBERS TRUSTWORTHY.** M1, M2, and M5 reduce context-member predictions; M3 and M4 do not. Comparing against the exact on-row AGREE rate rather than rounded 0.376 changes no sign.
- **q5 — NUMBERS TRUSTWORTHY WITH CAVEAT; conclusion incomplete.** The 120-row seen battery, power statistics, zero M4-drop result, and explicit absence of seen secondaries are correct. M1/M5 are shift-sensitive, M2 is gain-sensitive, and M3/M4 are relatively less sensitive. The missing Δ=0 rows and oracle ceiling must be included before calling robustness closed.
- **q6 — CONTINGENCY NUMBERS TRUSTWORTHY; “ADDS INFORMATION” CONCLUSION NOT TRUSTWORTHY.** Rescue rates and oracle-union accuracies are arithmetically correct, but support complementary scoring signal only.

No files, environment, packages, tests, or GPU state were changed.