# Code review request — exp_46 c4flip_data_curve, code round 1 of at most 3

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere (no `python -m py_compile`, no `__pycache__`, no pytest);
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`,
`sha256sum`, `head`, `grep` on named paths, and `python -I -c` that only reads). **Never** run `find`, `du` or `ls -R`
across `/n/fs` (shared filesystem); read only the paths named here and files in this repository. Do not read the
dataset (`AcousticRooms/`).

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve` (branch `exp-46-c4flip-data-curve`, HEAD
`e0cdea5a`, tracked tree clean; this prompt file is the only untracked addition besides the data links). Base:
`X45f = 1e37ead2` (exp_45's final code). All exp_46 records and kit files are in
`worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/` (`EXPREL46`).

## Context

- **The approved plan:** `plan_c4flip_data_curve.md` r3 (approved by Yixun 2026-11-17, M1–M4 as recommended), a delta
  to exp_45's approved plan r3 (`worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md`). Your three
  plan reviews: `c4flip_data_curve_codex_plan_r{1,2,3}_review.md`.
- **The code (L0):** written test-first by a Coder (Claude Opus 5.5, max effort) in the commits `1e37ead2..e0cdea5a`.
  `commits_c4flip_data_curve.md` lists every commit with its red / green logs. `c4flip_data_curve_worklog.md` (entries
  from "2026-11-17 22:28") holds the per-node test disposition of the 732 exp_45 nodes, the G7 confirmation and the
  Coder's open items 1–13.
- **Final L0 run:** `c4flip_data_curve_2026-11-18_02-24-31_pytest_L0_final2_yawaug_first_exp45_last.log`
  (2,522 passed, 15 skipped, 41 deselected, at `e6920ace`, tracked-clean, CPU Slurm job). L1 (three full CPU suites
  compared by `exp46_regression_compare.py`) has **not** run yet.

## How to read the diff (≈ 35 k inserted lines, most of them a verbatim copy)

- **Commit `407d11d1` (A)** is a renamed, verbatim copy of exp_45's kit and kit-behaviour tests at `X45f`. A test in
  that commit proves equality up to the declared rename map. Check that this proof is sound, then review the kit **as
  its diff from A**: `git diff 407d11d1 e0cdea5a -- worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/*.py
  src/tests/test_exp46_*.py src/tests/exp46_fixtures.py`. Read whole files wherever the context needs it.
- **Training path:** `git diff 1e37ead2 e0cdea5a -- src/training/` (commit `f796fe81`) and its tests
  `src/tests/test_exp46_c4flip_factory.py`, `src/tests/test_exp46_c4flip_wrapper.py` (commit `01b10973`).
- **Declared test edits E1–E5** (plan §3.1): `git diff 1e37ead2 e0cdea5a -- src/tests/test_yaw_aug_flip.py
  src/tests/test_yaw_aug_flip_factory.py src/tests/test_exp43_kit.py src/tests/test_exp45_merge_provenance.py
  src/tests/test_gen_model_comparison_depth_canon.py`.
- **Generator:** `git diff 1e37ead2 e0cdea5a -- worklog/worklog_yixun/gen_model_comparison.py` and
  `src/tests/test_gen_model_comparison_exp46.py`.
- **New exp_46-only tool:** `exp46_content_manifest.py`, derived from
  `worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_content_manifest.py`.

## What to check

This is code round 1 of 3. Whatever round 3 still finds goes to Yixun, so report **every** finding now, each with
severity BLOCKING / HIGH / MEDIUM / NIT, file:line, the concrete failure scenario, and a fix. Calibrate: BLOCKING /
HIGH only for a concrete, reachable failure that would admit a wrong result, lose or corrupt data, write into another
experiment's namespace, or modify `main`; hardening that needs an adversarial actor or deliberate interference is
MEDIUM or NIT (Yixun is the only operator).

1. **Plan conformance:** does the code implement plan r3 §2, §3.1, every row of the §3.2 table, §3.3, §5, §6 and §8?
   Name anything missing or different.
2. **Training path:** exactly the two refusals removed and nothing else (draw order, generator, quantization, mirror,
   banner unchanged); are the six §3.1 tests real tests of their claims (is the stream oracle independent of the code
   under test; would the step-equivalence test fail on a wrong transform or a wrong order of roll and mirror)?
3. **Fail-closed admission:** the f100 data binding (`full_split` audit entry + context policy), the delta manifest
   build / verify, the in-job re-verification before sealing (does every failure path — mismatch, missing file,
   missing manifest, exception, cancellation — leave the run unsealed?), receipts, online and offline admission, the
   loader-length and epoch gates for the four runs (1,148 / 2,276 / 3,413 / 4,550 steps per epoch; endpoints), the
   init-digest prediction.
4. **Isolation and side effects:** the guard over the exp_38 and exp_45 roots (realpath, symlinks, before any write,
   every mutating entry point), C3(a) storage exclusion, no exp_46 write path naming another experiment, never
   touching `main`, the argv rule.
5. **Table and generator:** the 20-row table, the exp_46 generator contract (8 rows), exp_45 dormant, E5's conditional
   expectation, the disclosures against plan §8.
6. **Tests and L1 policy:** the copied / not-copied node classification, the `ENV_SKIPS` extension,
   `exp46_regression_compare.py`'s policy for L1, the provenance test (allowed paths, content checks of E1–E5 and of
   the training-path diff, the 24-file Y46 rule).
7. **The Coder's open items** (worklog) — a verdict on each:
   - **item 1:** the neuronic `flac` env has wandb 0.26.1 while exp_43's kit pins 0.15.4, so 41 nodes of
     `test_exp43_kit.py` and 10 of `test_exp43_train_history.py` fail in every neuronic job (pre-existing; exp_46
     trains with `--logger none`). Does the L1 policy treat them correctly as inherited (failing identically in the
     parent junits)?
   - **item 3:** E4 also removed exp_45's uncommitted-edit check (it had no fixed endpoint); an equivalent lives in
     `test_exp46_provenance.py`. Acceptable?
   - **item 7:** no exp_46 command sheets exist yet (plan §4 needs one for the A6000 part). What must they contain?
   - **item 8:** f100 SMOKE needs only the pinned manifest; the re-verification runs only in FULL. Consistent with
     the plan?
   - **item 9:** the disclosure item sets differ — the collector prints items 1–6 and 8–11, the table 1–12, the
     generator 1–11. Correct against plan §8?
   - **item 10:** `F100_DELTA_SHA256` is `None` until L2, so the L2 pin commit changes the closure after L1. Which
     ordering keeps L1 valid for the frozen `X46`?
   - **item 13:** `test_exp45_sampler.py::test_both_launch_environments_give_the_same_rank_and_world` leaves
     lightning's global `rank_zero_only.rank` at 1 (pre-existing at `X45f`), so ≈ 11 pre-existing banner nodes fail at
     L1 in tree 46 when the suite runs in one process. Options: (a) a declared edit adding a 3-line restore to
     `test_exp45_sampler.py`; (b) a registered L1 allowance for exactly those nodes; (c) running that file in its own
     process. Which is sound? Is the Coder's fix in exp_46's own copies (commit `e6920ace`) correct?
8. **Anything else**, with fresh eyes.

## Output

1. Findings, most severe first (severity, file:line, failure scenario, fix).
2. One line per open item (1, 3, 7, 8, 9, 10, 13).
3. A one-line overall verdict: SHIP, SHIP-WITH-CHANGES or REVISE.
