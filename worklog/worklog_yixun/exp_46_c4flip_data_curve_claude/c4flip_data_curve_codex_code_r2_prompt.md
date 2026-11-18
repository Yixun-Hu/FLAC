# Code review request — exp_46 c4flip_data_curve, code round 2 of at most 3

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
`9cda0203`, tracked tree clean; this prompt file is the only untracked addition besides the data link). Base:
`X45f = 1e37ead2`. All exp_46 records and kit files are in `worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`
(`EXPREL46`).

## Context

- **The approved plan:** `plan_c4flip_data_curve.md` r3; your plan reviews `c4flip_data_curve_codex_plan_r{1,2,3}_review.md`.
- **Your code round 1:** `c4flip_data_curve_codex_code_r1_review.md` (REVISE: F1 HIGH, F2–F10 MEDIUM, F11–F12 NIT, plus
  verdicts on the open items), reviewed at `e0cdea5a`.
- **The fix round:** commits `671ae6c0..9cda0203` by the Coder (Claude Opus 5.5, max effort), test-first. The ledger
  `commits_c4flip_data_curve.md` and the worklog (entries from "2026-11-18 03:30") give each commit, its red / green
  logs and the disposition of each finding. Final targeted run:
  `c4flip_data_curve_2026-11-18_04-36-46_pytest_FR1_final_targeted_yawaug_first_exp45_last.log` (2,579 passed,
  15 skipped, 41 deselected, at `c8a04475`, tracked-clean, CPU Slurm job).
- **Not yet fixed, by design:** F6 / open item 13 (`test_exp45_sampler.py` leaks lightning's global rank). It waits for
  Yixun's decision D1 between your preferred (a) — a declared, content-checked 3-line restore in that file (it would
  become declared edit E7) — and (c). The fix will land in the next round and be reviewed in round 3.
- **Pending Yixun (D2):** the order written into the neuronic command sheet — final code round → L2 delta manifest
  build → pin commit of `F100_DELTA_SHA256` → definitive L1 (every node, no deselection) → L2 union verification
  (recorded) before any f100 SMOKE → L3 → L4 freeze `X46`.

## What to check

This is round 2 of 3. Round 3 is the last; whatever it still finds is fixed without re-review or goes to Yixun. So
report **every** finding now, each with severity BLOCKING / HIGH / MEDIUM / NIT, file:line, the concrete failure
scenario, and a fix. Same calibration as round 1: BLOCKING / HIGH only for a concrete, reachable failure that would
admit a wrong result, lose or corrupt data, write into another experiment's namespace, or modify `main`.

1. **Each round-1 finding F1–F12** (F6 excepted, see above) and **open item 1**: RESOLVED / PARTIAL (what is missing) /
   NOT RESOLVED, one line each. Check that each fix's test would actually fail without the fix.
2. **New problems introduced by the fix round** (`git diff 671ae6c0 9cda0203`), in particular:
   - F3's commit boundary (`exp46_supervise.py`): is there any path after the last signal check that can still seal a
     cancelled run, or a path where a sealed run is left half-written?
   - F7's `lstat` and full-schema validation of exp_45's ledger: still read-only on exp_45's paths in every case?
   - F9's `closure_digest_at(commit)`: does it compute the same digest as the live closure for the same tree?
   - F11's oracle taken from X45f's source in git.
   - The Coder's changes beyond the findings (worklog): F12's early A6000 SMOKE refusal, the exclusive-create JSON
     write of the init digest, the stricter C3 fixture, the `after_receipt` hook.
3. **The two command sheets** (new, commit `86425120`) and `src/tests/test_exp46_command_sheets.py`: do they contain
   everything you listed under open item 7, in an executable and correct order; are the commands consistent with the
   kit's real CLI (flags, modes, paths, `--time` with f100's extra hour); does the A6000 sheet stay evaluation and
   publication only; does anything in them write into another experiment's namespace or `main`'s tracked tree?
4. **Anything round 1 missed**, with fresh eyes on fail-closed admission, isolation from exp_38 and exp_45, never
   touching `main`, and the argv rule.

## Output

1. One line per round-1 finding and open item 1.
2. New findings, most severe first (severity, file:line, failure scenario, fix).
3. A one-line overall verdict: SHIP, SHIP-WITH-CHANGES or REVISE.
