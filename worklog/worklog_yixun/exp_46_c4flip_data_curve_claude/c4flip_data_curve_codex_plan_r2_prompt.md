# Plan review request — exp_46 c4flip_data_curve, plan round 2 of at most 3

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`,
`sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads). **Never** run `find`, `du` or `ls -R`
across `/n/fs` (shared filesystem); read only the paths named here and files in this repository.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve` (branch `exp-46-c4flip-data-curve`, HEAD
`b96003a7`). All exp_46 files are in `worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`.

## Context

- **The object of review:** `plan_c4flip_data_curve.md` **r1**, a delta to exp_45's approved plan r3
  (`worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md`). Changes r0 → r1: `git diff 5d4891cc b96003a7
  -- worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/plan_c4flip_data_curve.md`.
- Your round-1 review: `c4flip_data_curve_codex_plan_r1_review.md` (REVISE, 11 findings). The plan's §12 dispositions
  each finding.
- Yixun's decisions C1–C5: `c4flip_data_curve_yixun_query.md` (Query 8) — decided, not to be re-argued.
- exp_45's kit (`exp45_*.py`, `src/tests/test_exp45_*.py`, `exp45_fixtures.py`), exp_38's tools
  (`worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_*.py`, `data_efficiency_subset_audit.json`), the training
  path (`src/training/factory.py`, `src/training/diffusion.py`, `src/data/yaw_*.py`), `CLAUDE.md`,
  `worklog/experiment_SOP.md` and `worklog/worklog_yixun/announcement/` as in round 1.

## What to check

This is round 2 of 3. Whatever round 3 still finds goes to Yixun, so report **every** finding now, each with severity
BLOCKING / HIGH / MEDIUM / NIT, plan section or file:line, the concrete failure scenario, and a fix. Calibrate:
BLOCKING / HIGH only for a concrete, reachable failure that would admit a wrong result, lose or corrupt data, write into
another experiment's namespace, or modify `main`; hardening that needs an adversarial actor or deliberate interference
is MEDIUM or NIT (Yixun is the only operator).

1. **Each round-1 finding 1–11:** is it resolved by r1? One line each: RESOLVED / PARTIAL (what is missing) /
   NOT RESOLVED.
2. **New problems introduced by r1**, in particular:
   - the f100 data binding: is binding exp_38's audit `full_split` entry (read-only, schema unchanged) plus the
     context-policy check sound? Is the delta content manifest's scope ("every file f100 can read that exp_38's f075
     manifest does not cover", derived from the same loader code path) well-defined and complete, and is the
     verify-before-SMOKE / verify-after-FULL schedule sufficient? Is the I/O plan acceptable on a shared filesystem?
   - the closure fix (config renamed `exp46_…` + explicit path; the RF config out of the runtime closure);
   - the isolation guard's protected roots and "before any write" for every mutating operation;
   - E4 (fixed `X45f` endpoints) and the test disposition (which exp_45 tests are history, not kit behaviour);
   - E5, the dormant exp_45 generator contract, the 20-row table;
   - the Y46 E0 exception;
   - the restated §6 acceptance and §8 disclosures: anything still false or missing for C4F?
3. **Anything round 1 missed**, with fresh eyes on fail-closed admission, isolation from exp_38 and exp_45, never
   touching `main`, and the argv rule.

## Output

1. One line per round-1 finding.
2. New findings, most severe first (severity, location, failure scenario, fix).
3. A one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
