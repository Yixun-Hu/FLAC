## Verdict: CLOSE-WITH-FIXES

The scientific verdict survives: **5 SUPERIOR + 3 EQUIV; C50-K8 and both R@1 cells are equivalence only.** No reruns required.

1. **Correct raw aggregations.** Using sample SD (`ddof=1`):

   - K8: T60 **8.2929±0.0105**, σc=.015943, z=−19.83 SUPERIOR; C50 **.9660±.0015**, z=−.65 EQUIV; EDT **35.9513±.0532**, z=−13.06 SUPERIOR; R@1 **6.9591±.1353**, z=−.60 EQUIV.
   - K1: **9.5401±.0231 / 1.0323±.0060 / 38.7283±.2263 / 6.8108±.1766**; z=−9.46/−1.56/−2.82/−.07; tiers correct.
   - Therefore fix K8 T60 `8.2930±.0106` and R@1 `6.9592` in [results:12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_results.md:12), [worklog:187](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_worklog.md:187), [generator:13](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gen_closing_page.py:13), and [HTML:16](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_01_results.html:16).
   - Also fix 57.5k K8 R@1 to **6.1606±.0876**, not `6.1607±.0875`, at [results:19](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_results.md:19) and [worklog:199](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_worklog.md:199). Tiers remain unchanged.

2. **Narrow overclaims.**

   - Replace “none worse” at [results:17](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_results.md:17) and [HTML:13](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_01_results.html:13) with “none outside the 1σc equivalence band”; both R@1 point means are numerically lower.
   - Delete “beyond any plausible selection … artifact” at [analysis:14](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_analysis.md:14). There is one training seed and adaptive checkpoint selection.
   - Change the causal BN claim at [analysis:18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_analysis.md:18) and [HTML:24](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_01_results.html:24) to “the bundled P1 recipe closed the gap, consistent with the BN-statistics hypothesis.” The amendment explicitly forbids isolating SyncBN/micro causally at [amendment:26](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_stop_record_and_p1_amendment.md:26).
   - Scope “fine-tune-stage property, not a from-scratch one” at [HTML:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_01_results.html:23) to the tested recipe/budget.
   - Replace “Table-1-superior base” with “Table-1 superior-or-equivalent base” at [analysis:34](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_analysis.md:34).
   - Explicitly say **five evaluation seeds, one training seed**.

3. **Protocol passed, with residual selection bias.** The composite rule at [plan:40](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_bv_parity.md:40) yields qualifiers 55k/57.5k/70k/87.5k/95k; 87.5k has maximum seed-42 R@1. Held-out-only seeds 43–46 retain all eight tiers; held-out R@1 is K8 **6.9552±.1559, EQUIV** and K1 **6.8408±.1887, EQUIV**.

   However, 87.5k lay beyond the predeclared 67.5k horizon ([plan:25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_bv_parity.md:25), [amendment:20](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_stop_record_and_p1_amendment.md:20)). Thus [analysis:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_analysis.md:11) overstates “pre-registered machinery end-to-end.” The oscillation caveat at [analysis:21](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_analysis.md:21) must add that held-out eval seeds control eval-seed noise, **not adaptive temporal/checkpoint selection**.

4. **HTML provenance fix.** The gate mirrors Markdown, including its rounding errors. Full-precision curve values in [generator:17](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/gen_closing_page.py:17) and [HTML:21](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_01_results.html:21)—especially 97.5k/100k—do not appear at that precision in [results:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_results.md:23). Put the exact source table in Markdown, then regenerate.

5. **Closure bookkeeping.** Persist this review, review/regenerate the closing-page script per [SOP:66](/home/yixunhu/codespace/FLAC/worklog/experiment_SOP.md:66), append a final ISO-timestamped worklog entry, and update [commits_fa_scratch.md:13](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/commits_fa_scratch.md:13), which stops at commit 6. Fourteen experiment artifacts also remain untracked.