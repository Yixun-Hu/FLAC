# Code review request — exp_45 rf_data_curve, code round 3 of 3 (FINAL)

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job. The CPU Slurm logs are listed below.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`, `sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads).

**Never** run `find`, `du` or `ls -R` across `/n/fs`.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve` (branch `exp-45-rf-data-curve`, HEAD `90d0202f`).

## Context

- Round 1 (`738a5de7..23110d74`) found 17 issues; its review is `rf_data_curve_codex_code_r1_review.md`.
- Fix round 1 is `23110d74..a39ded1c`.
- Round 2 found 14 issues and graded the r1 fixes; its review is `rf_data_curve_codex_code_r2_review.md`.
- Fix round 2 is `a39ded1c..90d0202f`. That is one commit per r2 finding (3 + 4 and 1 + 14 share commits), plus test-only follow-ups `da55a4a9` and `5b1b2f73`, plus the records commit.
- The disposition table and the Coder's caveats are in the last entry (03:05) of `rf_data_curve_worklog.md` and in `commits_rf_data_curve.md`.
- The plan is `plan_rf_data_curve.md` (r3, approved).

All of these files are in `worklog/worklog_yixun/exp_45_rf_data_curve_claude/`.

**This is the final round.** Anything you report now that is BLOCKING or HIGH goes to Yixun as an open blocker before any launch. Calibrate accordingly:
- Mark as BLOCKING or HIGH only defects with a concrete, reachable failure that would admit a wrong result, lose or corrupt data, write into exp_38's or another experiment's namespace, or modify `main`.
- Mark hardening that needs an adversarial actor, deliberately crafted aliases, or races that need deliberate interference as MEDIUM or NIT, and say so explicitly. Yixun is the only operator.

## What to check

1. **Each r2 finding 1–14.** Is it actually fixed? Check the code, and judge whether its test would fail on the pre-fix code. Give a one-line verdict per finding: FIXED, PARTIAL (with what is missing) or NOT FIXED.
2. **The r1 PARTIALs (F1, F4, F5, F7, F12, F13, F14).** Are they now closed by r2 findings 5, 3, 2, 7, 1 and 9, 10, and 11 respectively?
3. **Regressions introduced by fix round 2.** Pay particular attention to:
   - **r2-2:** credit only for files unchanged (same `st_dev`, `st_ino`, `st_ctime_ns`) across both scans that bracket the free-space sample. Can this under-credit so badly that a legitimate FULL-start re-check refuses while a sibling run is writing checkpoints? Free space is about 235 GiB. Each run reserves 16 × 724,000,000 B + 2 GiB. There are three runs.
   - **r2-7:** the 1 h grace period on a `submitting` intent with no job id.
   - **r2-8:** the new `release <run>` command, including whether it can ever free a hold of a running job.
   - **r2-12:** the flash-only refusal at launch and at admission. Training must be flash only; the eval-side no-flash overlay must be untouched.
   - **r2-13:** the bytecode suppression.
4. **The Coder's flagged choice under r2-6, for Yixun's decision.**
   - Materialise destinations are refused only if they equal, or lie inside, a protected exp_38 root (exp_38's two `save_root`s, `DUMP_ROOT`, `FARM`). A destination is not refused merely for containing one, because the plan-N6 generator root `/n/fs/gatrdp/codespace/FLAC` contains exp_38's neuronic save root by layout.
   - `exp45_table.py`'s import materialisation still copies exp_38's verified records to their plan r3 §3.2 paths in that generator root. Those paths lie inside exp_38's neuronic save root (`FLAC/outputs_FLAC/exp38_data_efficiency`), again by layout. They are copied only when absent or byte-identical, never through an alias, and with a no-clobber guard.
   - Is this safe? Can it ever overwrite or perturb exp_38's own live state: its run dirs, locks, owners, ledgers, or anything exp_38's tools read for admission? Or should it be refused? Give a concrete recommendation Yixun can approve or reject.
5. **A final fresh-eyes pass over the whole round** (`738a5de7..90d0202f`), limited to fail-closed admission, isolation from exp_38, never touching `main`, and the RF argv rule. Report only new, concrete issues.

## Test evidence (do not re-run)

The final full-L0 run: `rf_data_curve_2026-11-15_02-43-58_pytest_r2fix_Z3_L0_final_full_scope_green.log`. Result: 981 passed, 6 skipped, 0 failed, at `5b1b2f73`, with a clean tracked tree. Per-finding logs are `rf_data_curve_2026-11-15_*_pytest_r2fix_*.log`. Aborted attempts carry `_ABORTED_<reason>`.

## Output

1. The per-finding verdicts for r2 1–14, and the r1-PARTIAL closure lines.
2. New findings, most severe first, each with severity, file:line, failure scenario and fix.
3. A recommendation on the r2-6 materialisation question.
4. A one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
