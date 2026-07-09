# Commits — exp_08_fa_matched

Base: `0bd5da0`. Branch: `check-equivariance-necessity`. Develop commit-by-commit per SOP.

| Order | SHA | Summary |
|---|---|---|
| 1 | `4d07611` | scaffold — Route A matched comparison (exp_07 held per Yixun) |
| 2 | `64a8cf3` | plan — matched comparison reusing V1′ as control; A-F arm + H-A1/A2/A3 gates |
| 3 | `a4d7b45` | plan review (APPROVE-WITH-CHANGES) + revision — A-V bf16 eval mirror added |
| 4 | `a022385` | Yixun's pre-approval findings solved — M5 train-seed sensitivity pair + tiered H-A1 bands |
| 5 | `f94b206` | approved; params + launch commands |
| 6 | `779bc70` | M1.5 mirror — bf16 shifts A-V T60 +0.12 (confound confirmed real); comparator registered |
| 7 | `b942357` | H-A1 verdict — strict FAIL, T60 superior (near-baseline K=8), EDT/C50 regressions; M3/M4 commands |
| 8 | `50f58e6` | H-A2 PASS + H-A3 PASS — minimum project goal achieved on fine-tuned model; M5 launching |
| 9 | `a3e8cf5` | closure — M5 verdict (T60 survives seed check; K=8 EDT/C50 downgraded, K=1 remain strict), results/analysis/HTML + `aggregate_results.py` + Codex review→re-verify |
| 10 | _(this commit)_ | bookkeeping — record closure SHA `a3e8cf5` in this log |

**No `src/` code changed in exp_08** — it consumed the exp_03/exp_04/exp_05 machinery (`fa_invariant`, `--freeze-bn`, bf16 eval, comparator) unchanged. The only new executables are analysis/reporting scripts under `worklog/exp_08_fa_matched_claude/` (`aggregate_results.py`, `fa_matched_results_assets/gen_page.py`), both covered by the closure Codex review. Hence exp_08 is one closure commit rather than several <200-line code commits.

Analyst/author of the closure artifacts: **Opus 4.8 (main session)** — role-transfer flagged per Yixun 2026-07-09 (the SOP's analysis seat, formerly Fable 5).
