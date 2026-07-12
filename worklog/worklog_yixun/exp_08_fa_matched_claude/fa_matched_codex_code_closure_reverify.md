Re-ran `aggregate_results.py` and checked the JSON-backed values.

1. **BLOCKING:** PARTIAL — main framing is fixed, but `analysis.md` still overgeneralizes M5 to EDT/C50 without K=8 scoping at [fa_matched_analysis.md](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_analysis.md:29) and [fa_matched_analysis.md](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_analysis.md:49).
2. **HIGH-1:** RESOLVED — 6/6 outside, 2 superior, 4 regression is consistent in results/analysis, e.g. [fa_matched_results.md](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_results.md:74).
3. **HIGH-2:** RESOLVED — Chart C vanilla is `0.2095079`, labeled exp_02 mean; exact JSON mean is `0.2095079107` at [gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_results_assets/gen_page.py:15).
4. **HIGH-3:** RESOLVED — A-F C4 mean is `0.0023352`; ratio is `89.7x` at [gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_results_assets/gen_page.py:18).
5. **MEDIUM-1:** RESOLVED — current results/analysis use 20-185x; worklog historical “three orders” has a correction at [fa_matched_worklog.md](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_worklog.md:26).
6. **MEDIUM-2:** RESOLVED — `load_one()` asserts exactly one match and prints the M5 path at [aggregate_results.py](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/aggregate_results.py:119).
7. **MEDIUM-3:** RESOLVED — HTML table values/chips match aggregate: K=1 C50/EDT strict fail, K=8 C50/EDT warn/indeterminate at [fa_matched_01_results.html](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_01_results.html:27).
8. **MEDIUM-4:** RESOLVED — results uses 59%; worklog keeps historical 65% but bracket-corrects to 58.9% at [fa_matched_worklog.md](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_worklog.md:20).
9. **LOW-1:** RESOLVED — provenance docstring names exp_01 baseline and exp_02 Chart C source at [gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_results_assets/gen_page.py:3).
10. **LOW-2:** RESOLVED — Chart A suppresses `0.0` and emits only `base` at y=0 at [gen_page.py](/home/yixunhu/codespace/FLAC/worklog/exp_08_fa_matched_claude/fa_matched_results_assets/gen_page.py:39).

Fresh scan: no unsupported non-inferiority conclusion remains; remaining uses are hypothesis/tier labels or explicit denials. No numeric mismatch found against aggregate/JSONs.

**RE-VERIFY VERDICT: STILL-NEEDS-CHANGES.** Single residual: unscoped M5 EDT/C50 wording in `fa_matched_analysis.md` lines 29 and 49 can still be read as downgrading K=1, contradicting the strict K=1 regression framing.