# Code review request — exp_46 c4flip_data_curve, code round 3 of 3 (FINAL)

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere (no `python -m py_compile`, no `__pycache__`, no pytest);
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`,
`sha256sum`, `head`, `grep` on named paths, and `python -I -B -c` that only reads). **Never** run `find`, `du` or
`ls -R` across `/n/fs` (shared filesystem); read only the paths named here and files in this repository. Do not read
the dataset (`AcousticRooms/`).

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve` (branch `exp-46-c4flip-data-curve`, HEAD
`fa6b8eee`, tracked tree clean; this prompt file is the only untracked addition besides the data link). Base:
`X45f = 1e37ead2`. exp_46 files are in `worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/` (`EXPREL46`).

## Context

- **Plan:** `plan_c4flip_data_curve.md` r3 (approved). **Your code rounds:** `c4flip_data_curve_codex_code_r1_review.md`
  (at `e0cdea5a`) and `c4flip_data_curve_codex_code_r2_review.md` (at `9cda0203`, REVISE: R2-1, R2-2 HIGH; R2-3 … R2-9
  MEDIUM; F2, F8, F10 PARTIAL).
- **Fix round 2:** commits `4b5efd90..fa6b8eee` (Coder, Claude Opus 5.5, max effort; test-first). Ledger
  `commits_c4flip_data_curve.md`; worklog entry "2026-11-18 05:46 → 07:20 EST" holds every disposition and the
  mutation-site audit list (enforced by `test_every_mutating_call_of_the_kit_is_guarded_or_audited`). New kit file
  `exp46_nas_publish.py`. Final targeted run:
  `c4flip_data_curve_2026-11-18_07-13-51_pytest_FR2_final_targeted_yawaug_first_exp45_last.log` (2,928 passed,
  15 skipped, 41 deselected, at `5827fe08`, tracked-clean, CPU Slurm job).
- **This is the last review round.** After it there is no fourth: whatever you find is fixed and covered by a test
  without re-review, or goes to Yixun if it prevents a correct result.

## Pre-review of a patch that is not yet applied (F6 / open item 13)

Yixun has not yet decided D1. If he chooses option (a), the following edit will be applied **verbatim** as declared edit
**E7** to the frozen `src/tests/test_exp45_sampler.py` (the same lines as exp_46's own copy, commit `e6920ace`),
inserted as the first lines of the body of `test_both_launch_environments_give_the_same_rank_and_world` (after its
existing `from lightning_fabric.plugins.environments import …` line, before the `for rank in (0, 1):` loop, ≈ line 88):

```python
    from lightning_fabric.utilities.rank_zero import rank_zero_only
    # exp_46 E7: LightningEnvironment.set_global_rank also sets lightning's GLOBAL rank_zero_only.rank; restored at
    # teardown so no later test in the process inherits rank 1 (silencing rank-0 banners)
    monkeypatch.setattr(rank_zero_only, "rank", rank_zero_only.rank)
```

No other line of that file changes. `test_exp46_provenance.py` would admit exactly this addition (content-checked), and
`exp46_regression_compare.py` would register that node as declared edited (it must pass). The definitive L1 then runs
the whole suite in one process with no deselection. Please judge: is this patch correct and sufficient (does the
monkeypatch teardown restore the value lightning's banner path actually reads, in every lightning module that keeps
its own reference)? What exactly must the provenance check and the L1 registration assert?

## What to check

Report **every** finding, each with severity BLOCKING / HIGH / MEDIUM / NIT, file:line, the concrete failure scenario,
and a fix. Same calibration as before: BLOCKING / HIGH only for a concrete, reachable failure that would admit a wrong
result, lose or corrupt data, write into another experiment's namespace, or modify `main`; hardening that needs an
adversarial actor is MEDIUM or NIT (Yixun is the only operator).

1. **Each round-2 finding** R2-1 … R2-9 and the PARTIAL F2, F8, F10: RESOLVED / PARTIAL (what is missing) / NOT
   RESOLVED, one line each; check that each fix's test would fail without the fix.
2. **New problems introduced by fix round 2** (`git diff 4b5efd90 fa6b8eee`), in particular:
   - `check_destination` and the guarded primitives: correct on every path (lexical containment, symlinked components,
     dangling links, the `allow=` exceptions for exp_38's worklog / farm only), no false refusal on a legitimate
     first-time path the campaign needs;
   - `exp46_nas_publish.py`: the `renameat2(RENAME_NOREPLACE)` path and the fallback, on the NAS mount (note: the
     neuronic NFS scratch refuses `renameat2(RENAME_NOREPLACE)`; does the fallback stay no-clobber and does the sheet
     use the right path?);
   - `abandon` refusing a sealed run, the recovery text;
   - the A6000 sheet under `set -e` (any `&&` / `||` list or pipeline that can hide a failure), the publication
     worktree's creation and checks;
   - the L4 dry-runs (do they submit nothing; are they the four launch paths the plan means);
   - the order of the neuronic sheet, marked "pending Yixun D2": final code round → L2 delta manifest build → pin
     commit of `F100_DELTA_SHA256` → definitive L1 → L2 union verification before any f100 SMOKE → L3 → L4 → `X46`.
3. **Anything earlier rounds missed**, with fresh eyes on fail-closed admission, isolation from exp_38 and exp_45,
   never touching `main`, and the argv rule.

## Output

1. One line per round-2 finding and the PARTIAL F2 / F8 / F10.
2. The F6 / E7 pre-review: correct / not correct, and what the provenance check and L1 registration must assert.
3. New findings, most severe first (severity, file:line, failure scenario, fix).
4. A one-line overall verdict: SHIP, SHIP-WITH-CHANGES or REVISE.
