**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code A7-C2 (exp_22 R2: the A7-C1 fixes; 9206da2f → 44c2241b) —

**Verdict:** SHIP

## F1–F5

| Item | Status | Review |
|---|---|---|
| **F1 — snapshot-bound evidence** | **RESOLVED** | [`snapshot()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:271) uses one no-follow, non-blocking descriptor and refuses links, FIFOs, directories, absent files and unreadable files. [`line()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:458) snapshots log/JUnit/outputs/inputs once, validates those bytes, re-authenticates them immediately before recording, and builds both the dispatch record and state identities from the original snapshots. [`checks()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:388) authenticates SHA-256 plus device/inode before every later line and [`finish()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:590) before automatic release. No later read establishes an expected evidence hash. Tests cover mutation/replacement between validation and record and after record at [test_exp22_r2c_snapshots.py:49](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_snapshots.py:49) and [test_exp22_r2c_snapshots.py:99](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_snapshots.py:99). |
| **F2 — complete child identity** | **RESOLVED** | The schema requires PID, start time, cwd and cmdline SHA-256, with the child cmdline equal to the expected command ([supervisor.py:117](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:117)). [`identify()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:500) retries the `/proc` race but never returns a partial identity. Early-exiting or never-matching children remain `line_pending` with `unidentified`, run to their own end without signalling, then stop with nothing judged and the reservation retained ([supervisor.py:474](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:474)). [`alive()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:208) compares the complete identity only. Schema, Planner-release and immediate/transient/never-matching tests are present. |
| **F3 — released-only resolution** | **RESOLVED** | [`active_reasons()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:221) resolves only `released`; stopped and crashed phases continue blocking after reservation disappearance. [`planner_release()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:256) refuses a pre-existing foreign replacement without unlinking it, while the later absent-record path records `released`. The phase matrix and foreign-replacement sequence are covered at [test_exp22_r2c_planner.py:29](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_planner.py:29). |
| **F4 — hygiene/regression** | **RESOLVED** | The two EOF blank-line defects are removed. The new hygiene test covers every `.py`, `.sh` and `.json` tool source ([test_exp22_r2c_hygiene.py:7](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_hygiene.py:7)); a separate committed-blob probe found all 20 clean. Detector/common, detector shell, preflight launcher and line-list blobs are unchanged from `9206da2f`; capture logic, backend rule, T rule and line validation are semantically untouched. Moving `box` into `tools/conftest.py` changes test infrastructure only. |
| **F5 — final verdict** | **RESOLVED — SHIP** | The A7-C1 BLOCKING, HIGH and NIT findings are closed. The branch may be merged into `localization-exp`; that merge commit may serve as PIN2. |

The disclosed F1 residual window is acceptable. A mutation after the pre-record authentication cannot change what the dispatch record claims was validated: the record and state still contain the original snapshot identity. A persistent change then prevents the next line or final automatic release. A crash retains the reservation for explicit Planner recovery. This is a bounded integrity-check window, not the original validation-to-different-hash substitution bug.

## Ranked findings

No BLOCKING, HIGH or NIT findings.

## Planner-audit reconciliation

Confirmed:

- All 5 commits carry the `Claude Opus 5.5` trailer.
- `git diff --name-only 9206da2f 44c2241b -- ':(exclude)worklog'` is empty.
- Merge base with `localization-exp` is `9847b45d`; its non-worklog delta is exactly the four PIN2 paths.
- The four files retain the stated hashes and are byte-identical to `5b065801`.
- `localize_meshgrid.py` and `src/localization/` are identical to `701926e5`.
- `git diff --check 9206da2f 44c2241b -- '*.py' '*.sh' '*.json'` is clean.
- AST-based counts reproduce exactly:

  - c13: production `+0/−0`, tests `+13/−0`
  - c14: production `+2/−2`, tests `+35/−5`
  - c15: production `+27/−17`, tests `+63/−1`
  - c16: production `+81/−34`, tests `+133/−44`

- c17 changes only the three final verification logs.
- Committed final evidence reports:

  - tools: 298 passed; SHA-256 `1ba056bf4f8041ffaf225c23b36d57240160906c90bcb2da3a07fb8cc4174441`
  - launcher tests: 39 passed; SHA-256 `c4faee1925c4c37a1520b2e8eccc309c76d48b0d997f46e8e8cb7e217d373ef6`
  - whole-repository collection: 6,273 tests, zero errors; SHA-256 `9aee0d870916133e5c706543a9d94f66b76a6e54cdb982afd7d49cda10ac685d`

## Read-only probes performed

I used `git status`, `log`, `show`, `diff`, `diff-tree`, `merge-base`, `ls-tree`, `cat-file`, `rg`, `sed`, `nl`, `sha256sum`, and read-only `python -B` AST/blob checks. I did not run pytest, the detector, launcher or supervisor; did not execute GPU commands; did not inspect processes; did not access `.gpu_reservations`; and did not modify any file or environment.
