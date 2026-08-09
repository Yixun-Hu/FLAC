# Q' approval — post-C32 measurement pin bb9de07

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-09

# Q-pin re-review — post-C32 measurement pin `bb9de07`

**Reviewed pin:** `bb9de072feffc4ee708e59eb37277b189f7e08ed`  
**Verdict:** **APPROVED** as the post-C32 measurement pin for the four q9 blocks.

## Blocking findings

None.

## Verified closures

1. **Per-row q9 contract:** All four q9 row specifications declare `contract="q9"`, which is threaded through `main()`, `render_row()`, and `validate_exp11_cell()`. The reviewer reproduction now proves identical populated evidence passes `q9` and fails `table`. The end-to-end test populates VANL/C4L × K1/K8 through `main()` and asserts numeric, unblocked, non-withheld rows.

2. **Complete one-pin transaction:** `check_q9_round()` requires all four cells and one shared `source_sha`. Missing cells or mixed pins rewrite every q9 row as **WITHHELD** and record the specific reason. Tests cover both missing-comparator and two-pin rounds.

3. **Non-vacuous sidecar-glob regression:** The test writes real metric and sidecar files, uses recursive globbing, first proves the metric matches, then proves the adjacent sidecar does not.

Focused read-only rerun: **6 passed**. The committed logs report **230 passed** for pytest and **151/151 passed** for screen guards. `git diff --check` is clean.

## Measurement disposition

Use `Q' = bb9de07` for all four q9 cells. Keep C32 confirmatory measurement pinned at `0c6e9ffb616cbd788b420e67d62638ad40a7b13c`. No jobs were submitted or disturbed during this review.

**Final verdict: `Q' = bb9de07` is APPROVED.**
