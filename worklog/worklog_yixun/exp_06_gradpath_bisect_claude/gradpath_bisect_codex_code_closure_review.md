# Codex code review — consolidated: exp_06 closure round

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** exp_06 gen_page.py + results/analysis claims + exp_02 rebuilt figure-2 (deferred scope)

Verdict: **APPROVE-WITH-NITS**

**Findings**

- **Low:** [gradpath_bisect_results.md](/home/yixunhu/codespace/FLAC/worklog/exp_06_gradpath_bisect_claude/gradpath_bisect_results.md:27) says “all arms” are seed-42 screens, but L2/V1p uses the prior 5-seed mean (`9.235/0.928/38.731/6.953`), not the seed-42 JSON (`9.243/0.926/38.738/6.817`). Same provenance nit appears in the HTML figcaption at [gradpath_bisect_01_results.html](/home/yixunhu/codespace/FLAC/worklog/exp_06_gradpath_bisect_claude/gradpath_bisect_01_results.html:8). This does not change ordering: using seed-42 L2 still gives T60 gaps of `0.156, 0.353, 0.270, 0.233`.

- **Low:** Generic “damage is monotone-increasing in lr” wording at [gradpath_bisect_results.md](/home/yixunhu/codespace/FLAC/worklog/exp_06_gradpath_bisect_claude/gradpath_bisect_results.md:27) and [gradpath_bisect_analysis.md](/home/yixunhu/codespace/FLAC/worklog/exp_06_gradpath_bisect_claude/gradpath_bisect_analysis.md:13) should be read as **T60 damage**. The T60 monotone claim is statistically safe; adjacent gaps are 12-30x the K=8 single-eval T60 noise. It should not be generalized to every metric: EDT is not strictly monotone across L3/L4, and C50 improves in lower-lr arms.

**Checks Passed**

- `gen_page.py` exp_06 hardcoded T60 values match `_results.md`; L1/L3/L4/L5 match the committed screen JSONs after rounding.
- S2 SVG log-x geometry is correct; baseline dashline placement is correct.
- S1 points match the results table’s chosen 0/200/400/625 values, with the caveat that the final 625 point is a prior 5-seed mean.
- exp_02 rebuilt Figure 2 checks out: `idx=742`, envelope gap `1.4299 dB`, median `0.4341 dB`, rel-L2 `0.27843`, zoom window `6762..6982` samples, and GT index arithmetic `742 // 2 == 371`, offset `0`.
- No-finalist claim is safe: even L1 misses by `0.437` T60 and `1.451 ms` EDT, far beyond the K=8 noise floor.

**Artifacts To Regenerate**

None mandatory. If you apply the wording/provenance nits, regenerate `worklog/exp_06_gradpath_bisect_claude/gradpath_bisect_01_results.html`.
---
**Disposition (Fable 5):** Both Low nits applied same-round (L2 provenance labeled in results.md + page; monotone claim scoped to T60 in results/analysis/page); page regenerated. All numeric checks passed incl. independent recomputation of exp_02 figure-2 values.
