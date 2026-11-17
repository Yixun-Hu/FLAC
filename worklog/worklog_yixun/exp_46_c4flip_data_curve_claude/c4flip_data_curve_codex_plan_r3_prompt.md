# Plan review request — exp_46 c4flip_data_curve, plan round 3 of 3 (FINAL)

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`,
`sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads). **Never** run `find`, `du` or `ls -R`
across `/n/fs` (shared filesystem); read only the paths named here and files in this repository.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve` (branch `exp-46-c4flip-data-curve`, HEAD
`bb8f67a6`). All exp_46 files are in `worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`.

## Context

- **The object of review:** `plan_c4flip_data_curve.md` **r2**, a delta to exp_45's approved plan r3
  (`worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md`). Changes r1 → r2: `git diff b96003a7 bb8f67a6
  -- worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/plan_c4flip_data_curve.md`.
- Your earlier reviews: `c4flip_data_curve_codex_plan_r1_review.md` (11 findings) and `c4flip_data_curve_codex_plan_r2_review.md`
  (6 findings; r1 items 2–5 partial). The plan's §12 dispositions every finding of both rounds.
- Yixun's decisions C1–C5: `c4flip_data_curve_yixun_query.md` (Query 8) — decided, not to be re-argued.
- Sources as in rounds 1–2 (exp_45 kit and tests, exp_38 tools and audit, `src/training/factory.py`,
  `src/training/diffusion.py`, `src/data/yaw_*.py`, `CLAUDE.md`, `worklog/experiment_SOP.md`, announcements).

**This is the final round.** Anything you report now as BLOCKING or HIGH goes to Yixun as an open blocker before any
code is written. Calibrate: BLOCKING / HIGH only for a concrete, reachable failure that would admit a wrong result,
lose or corrupt data, write into another experiment's namespace, or modify `main`. Hardening that needs an adversarial
actor or deliberate interference is MEDIUM or NIT — say so explicitly (Yixun is the only operator). Everything below
HIGH is fixed in the next plan revision without re-review and covered by a test in the code round.

## What to check

1. **Each round-2 finding r2-1 … r2-6** and the four round-1 items that were PARTIAL (2, 3, 4, 5): RESOLVED / PARTIAL
   (what is missing) / NOT RESOLVED, one line each.
2. **New problems introduced by r2**, in particular: the in-job f100 delta re-verification before sealing (placement in
   the supervisor's FULL branch, interaction with the CSV loss-record move, the end audit, the commit marker, the
   crash policy, `--time`, a cancelled job, and every admission path that must now require the verification record);
   the snapshot-root changes and the refused A6000 SMOKE path; the path-literal sweep; node-level test disposition and
   the `ENV_SKIPS` extension; E5's disclosure-block removal.
3. **A final fresh-eyes pass** over the whole plan, limited to fail-closed admission, isolation from exp_38 and exp_45,
   never touching `main`, the argv rule and comparability. Report only new, concrete issues.

## Output

1. One line per item in 1.
2. New findings, most severe first (severity, location, failure scenario, fix).
3. A one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
