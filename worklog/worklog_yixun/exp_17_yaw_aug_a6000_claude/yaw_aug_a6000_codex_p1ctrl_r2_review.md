**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-16 · **Round:** P1-control r2 (terse re-verification of the r1 blockers)

---

Re-verified `exp17-yawaug-scratch` at `c39ad12`.

1. **FIXED — shared lock.** P1 and YAWAUG both hold the same `.roteval.lock`; `pgrep` is no longer the mutual-exclusion mechanism. [P1 runner:35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_p1ctrl_roteval_run.sh:35)

2. **FIXED — arm/seed scoping and duplicates.** `load_cells()` filters the requested arm, rejects foreign same-arm seeds, and hard-fails duplicate `(step,K,rotation)` identities before grouping. Mixed orbits cannot be fabricated. [exp17_rotation_table.py:63](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:63)

3. **FIXED — estimand labeling.** Output explicitly says global/sample-weighted, single seed 42, and not per-scene/five-seed; the test docstring now agrees. [table:174](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_table.py:174), [test:8](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_table.py:8)

**NEW Blocking findings: none.**

Static read-only review only; no tests, scripts, or GPU commands run.
