# Code review request — exp_45 rf_data_curve, code round 2 of at most 3

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job. All tests ran in CPU Slurm jobs, and their logs are listed below.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`, `sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads).

**Never** run `find`, `du` or `ls -R` across `/n/fs`, which is a shared filesystem. Read only the paths named here and the files in this repository.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-45-rf-data-curve` (branch `exp-45-rf-data-curve`, HEAD `a39ded1c`).

## Context

- Round 1 reviewed `738a5de7..23110d74` and returned REVISE. Its review is `worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_codex_code_r1_review.md`, with 17 findings.
- The coordinator's triage is in `rf_data_curve_worklog.md`, entry 22:35: fix findings 1–16; finding 17 gets no change.
- The Coder's fix round is `23110d74..a39ded1c`: one commit per finding (F8 + F9 and F2 + F3 share one), plus records commits. The disposition table (finding → commit → test node → red/green logs) is in the last entry of `rf_data_curve_worklog.md` and in `commits_rf_data_curve.md`.
- The approved plan is still `plan_rf_data_curve.md` (r3), along with `worklog/experiment_SOP.md` and the announcements in `worklog/worklog_yixun/announcement/`.

## What to check

Report **every** finding in this single pass. This is round 2 of 3. Whatever round 3 still finds goes to Yixun, so be complete now. Give each finding:
- a severity: BLOCKING, HIGH, MEDIUM or NIT;
- file:line;
- the concrete failure scenario;
- a fix.

1. **Is each r1 finding 1–16 actually fixed?** Check the code, not just the test name. For each one:
   - does the new test fail on the pre-fix code? Reason about it from `git show <commit>`; you can check the red log too.
   - is the fix complete for the scenario the r1 review described?

   Give a one-line verdict per finding: FIXED, PARTIAL (with what is missing) or NOT FIXED.
2. **Regressions and new defects introduced by the fix round.** Look in particular at:
   - **F1.** The storage release is now its own `releases` entry in the reservation ledger, and `Record.seal()` blocks every rewrite after the commit marker, including the interrupt handler's. A release that fails after sealing leaves the hold in place, and no tool can release a sealed run's hold yet. Is that a deadlock for later submissions? Should `abandon` or a `release` command cover it, and does it fail closed?
   - **F2.**
     - The admission record's sha256 is sealed into the bridge result's `admission_record`, not into the descriptor, together with the admitted path, sha, bytes, config and step.
     - The descriptor gains `global_step: 40000`.
     - `exp43_kit.py` and the AR manifest it reads were added to the closure, so the closure digest changed (E0 has not run yet).
     - Is this binding sufficient? Is the closure change complete and correct, with nothing else the bridge now reads left out?
   - **F3.** The fixtures now carry real 6,337-item arrays from the exp_40 C16 sidecars, which re-hash exactly to the ten pins. Check that the integrity check really recomputes `input_hash` and `assignment_hash` from the arrays. Matching lengths alone is not enough.
   - **F5.** Written bytes are counted as min(apparent size, allocated blocks). Is that conservative in the right direction? Under-crediting is safe and over-crediting is not. Check sparse files, preallocated files, a file being truncated, and hardlinks or symlinks inside a run dir.
   - **F7.**
     - The per-(run, mode) submission lock `<save root>/.exp45_submit_<run>__<MODE>.lock`.
     - `abandon` recovers each crash window.
     - `reconcile` refuses and points to `abandon`.
     - Check for lock-ordering deadlocks against the ledger flock, stale lock files after a kill, and whether `abandon` can ever free a hold whose job is still pending or running in Slurm.
   - **F8 / F9.**
     - The comparator now needs `--merged-root`, `--exp38-root` and `--exp43-root`.
     - `ENV_SKIPS` was read from a junit capture in the neuronic CPU environment.
     - Is the allow-list narrow, keyed by node and reason? Does the L1 command in the command sheet pass the new arguments?
   - **F11.** Every launch path is now 2×32, and any other world size is refused. Is any remaining code path that still accepts world size 1 now dead or inconsistent?
   - **F12.**
     - `exp45_collect.py verify` / `materialise --dest`.
     - Generator-root copy only if a file is absent or identical.
     - `exp45_table` now also copies exp_38's configs, reference cells and kit code, and the import record has a `code` field.
     - The A6000 sheet commits R45 only after it has copied and verified the files.
     - The sheet requires `$B38` to descend from `e5a012fa`.
     - Can `materialise` ever overwrite a tracked file in the generator root, which is the plan-N6 main checkout? Does any command in the sheet modify `main`? None may.
   - **F13.** A narrow rule allows the subject `exp_45 records: E0 PASS at <40-hex>` with the 24 E0-cell files only. Is the rule anchored, so that no other subject or path can use it? Does it match the command sheet's subject exactly?
   - **F14.** A lock around environment-receipt creation, and the claim that ordering does not matter because every SMOKE re-checks the one receipt by exact equality. Is that claim true? Can a partially written receipt be read by another SMOKE?
   - **F15.** The collector table omits disclosure item 7 (there is no OrbitRIR row). The generator prints the disclosures only while an exp_45 row is published. Is this consistent with plan §8?
3. **Anything round 1 missed.** Re-read the whole kit with fresh eyes, focusing on the fail-closed admission paths and on isolation from exp_38: nothing may write into exp_38's folder, farm, dump root or owners file.

## Test evidence (do not re-run)

The final run of the full round-1 L0 scope: `worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_2026-11-15_00-34-42_pytest_r1fix_Z_L0_final_full_scope_green.log`. Result: 936 passed, 6 skipped, 0 failed, at `2f597012`, with a clean tracked tree. The per-finding red and green logs are `rf_data_curve_2026-11-1{4,5}_*_pytest_r1fix_*.log` in the same folder.

## Output

1. The per-finding verdicts for r1 F1–F16.
2. New findings, most severe first.
3. A one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
