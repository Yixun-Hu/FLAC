**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan A7 r3 (exp_22 R2: the no-flash backend, deltas after A7 r2)

**Verdict:** REVISE

A7 r3 resolves both prior BLOCKING findings and HIGH 2–4. One executable gap remains in HIGH 1: this pinned pytest’s JUnit XML cannot distinguish a non-strict XPASS from an ordinary pass. The Coder round must not start until that outcome is bound through another machine-readable source and tested.

## S1 — RESOLVED

The amended backend rule at [plan_loc_meshgrid_r2_bf_yaw.md:108](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:108) now matches the pinned launcher:

- H35 rejects `No module named 'flash_attn'`, requires the overlay/fallback lines back-to-back, and checks the dynamic target in the sentinel ([depth_canon_collect.py:422](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:422)).
- The launcher checks `flash_attn_func is None` before emitting the sentinel and then executes the target in the same interpreter ([exp35_no_flash_launch.py:137](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:137), [exp35_no_flash_launch.py:188](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:188)).
- Its test requires pair → sentinel and checks the exact target-specific fields ([test_exp35_no_flash_launch.py:107](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:107), [test_exp35_no_flash_launch.py:157](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:157)).
- Allowing later child-interpreter occurrences only as complete, adjacent overlay→fallback pairs is the correct relaxation. “Only as back-to-back pairs” must be implemented universally: every later occurrence of either exact line must belong to such a pair.
- None of the registered targets contains the missing-module, refusal, or sentinel literals as ordinary output.

For the pinned launcher and registered targets, I found no route for a non-SDPA main-interpreter launch to pass and no reason a valid launch would fail.

## S2 — RESOLVED

The launcher-owned admission evidence is sufficient:

- 31 polls at 60-second intervals establish 30 elapsed minutes.
- Each observation is sealed in an O_EXCL capture record.
- The 10-minute bound limits staleness.
- UUID mapping and all live blockers are checked immediately before and after reservation acquisition.
- The inherited r2 post-acquisition and pre-line checks remain in force because r3 says everything in r2 stands unless replaced.
- Admission no longer trusts the detector notification.

This closes the unauthenticated detector-to-launch handoff identified in A7 r2.

## S3 — PARTIAL

Resolved at [plan line 108](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:108):

- The session background task is now a named wake channel. It never admits or launches; if the session disappears, nothing happens.
- The durable state machine has the required pre-action atomic write plus file-and-directory fsync ordering, pending-line identity, child identity, recorded nonzero completion, crash-stays behavior, and no automatic adoption.
- `r2_preflight_<RUN>` removes the dispatch/scientific-output collision.
- The record fields are separated correctly.

Not resolved:

- The claimed JUnit-only `zero xpassed` condition is not executable with pinned pytest 9.1.1. Pytest turns a non-strict XPASS into a passed report with `wasxfail` ([skipping.py:305](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/_pytest/skipping.py:305)); the JUnit writer handles every passed call as an ordinary pass without checking `wasxfail` ([junitxml.py:553](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/_pytest/junitxml.py:553)). Such an XPASS therefore contributes to JUnit’s `passed` count.

The current eleven tests have no static xfail marks, but the amendment explicitly promises a general machine-enforced zero-XPASS rule.

## S4 — PARTIAL

The literals and lock probe are complete:

- Both UUIDs match the registered exp_35 launch script ([exp35_phase_i_launch.sh:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_i_launch.sh:32)).
- The absolute lock path is correct.
- The read-only `O_RDONLY` plus `LOCK_SH | LOCK_NB` probe matches the reviewed implementation ([exp35_supervisor.py:976](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:976)).
- Held, unheld-leftover, absent and unreadable outcomes are specified fail-closed.
- The remaining added tests cover the prior R5 requests.

The Coder list is incomplete only because it includes pass/skip/deselection but not non-strict XPASS. Add explicit XFAIL and XPASS cases, particularly an XPASS fixture whose process exits 0 and whose JUnit suite reports it as passed.

## S5 — NOT RESOLVED

**REVISE.** The Coder round may start after the pytest admission rule and test matrix are corrected.

## Ranked findings

- **BLOCKING:** None.

- **HIGH 1 — JUnit cannot prove zero XPASS.** Amend T’s rule to require:

  1. pytest exit 0;
  2. JUnit `tests=11`, `failures=errors=skipped=0`, with exactly the eleven registered case identities; and
  3. a separately validated terminal summary or reviewed machine-readable outcome artifact with exactly `passed=11` and `skipped=failed=errors=xfailed=xpassed=deselected=0`.

  Bind that artifact or summary to T’s dispatch record and add XFAIL/XPASS Coder tests.

- **NIT:** None.

## What I verified

Read-only probes only:

- Branch `localization-exp`, HEAD `09221e0c939b010fc0321ab5688e55c198e620bb`; tracked worktree and index clean. Existing untracked files were untouched.
- A7 r3 is the final line and final §9 bullet of the plan; the matching notebook entry is [loc_meshgrid_worklog.md:857](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:857).
- The verbatim A7 r2 review and every cited unresolved item.
- H35 checkout HEAD `5b065801e9d1c1844461ec84cd0558702c714326`.
- H35’s collector, launcher and launcher-test implementations using `git show 5b065801:…`.
- Registered hashes: overlay `0d0e52ca…`, launcher `bbd2847d…`, launcher test `d61a3c02…`, and `pyproject.toml` `6422eae5…`.
- The triage log SHA-256 `08c362ed…`; the plan contains exactly eleven unique T node IDs, equal to its eleven failed IDs.
- The UUID literals, absolute lock path and reviewed read-only flock semantics.
- Pinned pytest version 9.1.1 and its JUnit/xfail source behavior.
- Commands used: `git status`, `rev-parse`, `show`, `log`, `diff`, `ls-tree`, `grep`; `rg`, `nl`, `sed`, `tail`, `wc`, `sha256sum`; and `python -B -c` for read-only text inspection.

I did not run pytest, any driver, gate, probe, detector, launcher, GPU command, or process query, and did not touch, signal, or attach to any process.
