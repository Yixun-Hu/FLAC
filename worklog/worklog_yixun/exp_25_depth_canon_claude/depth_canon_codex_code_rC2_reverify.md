**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-26 · **Round:** code rC2 (re-verification of rC after fix round rC-F2)

**Verdict:** SHIP — R1–R4 are resolved in the requested scope. No BLOCKING or pre-arm HIGH finding remains. The training launcher/watcher may be armed.

## Findings

- **NIT — one stale acceptance sentence remains outside the requested v14.1 edits.** [plan_depth_canon.md:264](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:264) still says projection ≤60 h and zero foreign compute processes. The authoritative §4 watcher paragraph and §7 rung 5 are correct, so this does not block arming. Fix: change it to ≤120 h, ≥12,288 MiB/card, and co-tenants disclosed under `shared`.

- **NIT — strict-schema edge branches are implemented but not individually load-bearing.** The unexpected-record-key branch at [depth_canon_launch.sh:333](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:333) has no corresponding E-case; neither do representative float, string-GPU, non-list `co_tenants_at_launch`, or non-dict-record inputs. Current code rejects all of them correctly. Fix: add four compact `ev_refused` cases. E2–E10/L6–L8 asserting only the headline is likewise non-blocking diagnostic coverage debt.

## Item-by-item re-verification

- **R1 — RESOLVED.** The launcher passes its policy, floor, and cap into the checker and type-strictly joins them to `NEED` at [depth_canon_launch.sh:299](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:299). Exact free-memory keys and integer floors are enforced at line 319; exact record shape, ranges, no booleans, and non-empty `cwd` at lines 326–343. Extra record keys, equal-valued floats, `gpu: "0"`, and a dict instead of a list all fail. The refusal remains exit 2 with the original headline and one reason line at lines 381–389.

  The six-sha bind remains at lines 285–312. The evidence writer at lines 724–753 is unchanged. Its real output round-trips through FULL in R1c, while empty and populated co-tenant positives pass. E11a–E24 cover the required negative matrix at [depth_canon_guardtests.sh:507](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:507); removing the literal, free-memory, or record-validation blocks makes their respective RED cases accept.

- **R2 — RESOLVED.** `gpu_clean()` uses the exact `nounits` query and rejects non-decimal `used_memory` with an explicit dirty reason at [depth_canon_autolaunch.sh:182](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:182). The poll loop resets `CLEAN`, stays `WAITING_GPU`, and does not increment launcher retries at lines 242–250. The co-tenant line restores `MiB`.

  Both `na` and `na_other` are tested at [test_depth_canon_autolaunch.py:289](src/tests/test_depth_canon_autolaunch.py:289). Removing `nounits` breaks the exact-query stub and clean-poll tests; removing the numeric check breaks both J2 cases. Existing three-poll, token, rc-7 cap, literal rung, scrub, tee-drain, and idle-policy behavior remains covered.

  Projecting away the watcher-only `process_name` field, the launcher and watcher accept the same UUID/PID/used-memory grammar. The watcher additionally rejects a comma-containing process name; that is conservative and cannot cause a wrong launch.

- **R3 — RESOLVED.** The watcher is part of the clone overlay at [depth_canon_guardtests.sh:26](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:26), and U15 reads both scripts through `git -C "$R" show HEAD:` at lines 612–625. U15b’s clone-only 23,040 flip proves isolation at lines 635–643.

- **R4 — RESOLVED in requested scope.** The §4 watcher paragraph at [plan_depth_canon.md:105](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:105) and §7 rung 5 at line 184 now match 12,288 MiB/shared/120 h. The v14.1 changelog and Planner receipt accurately describe those edits. The separate §10 NIT is above.

## Deviations and open questions

All eight Coder deviations are accepted. The two pre-commit GREEN runs are byte-identical to their commits; current hashes reproduce exactly:

- Launcher `7288e6b4…`
- Watcher `d1aa5e41…`
- Guard `7a1c5130…`

Open-question adjudication:

- The `write_audit` flake remains **HIGH for any later unqualified Tier-1 “equal by construction” claim**, but is not a training-arm blocker. Root-cause it before making that claim.
- E2–E10/L6–L8 reason-line assertions are a **NIT**.
- No co-tenant/free-memory arithmetic cross-check is warranted: they are independent telemetry, and the FULL launcher repeats its live gate. Parsing the bound log would be optional provenance hardening, not a missing admission control.

## Read-only verification performed

- Confirmed no training-closure change: only `src/tests/test_depth_canon_autolaunch.py` changed under `src/`; no `train.py`, `defaults.ini`, configs, `AR_md.py`, operator, or eval-driver change.
- Generated the pin manifest to stdout and compared it directly: byte-equal, 48 entries, sha256 `631da22f…`.
- `bash -n` passed for launcher, watcher, guard, and pin generator.
- Inspected every specified RED/GREEN/ABORTED/final log. RED cycles occurred at committed heads; all six code commits are under 200 changed lines. `1ed2027` is a larger records-only commit.
- Confirmed Planner results at `1ed2027`: regression 2403/7/1, guard 207/0/0, and DRY_RUN rc 0. The standalone W&B identity result is accepted from the supplied Planner receipt.
- `git diff --check` found only captured trailing whitespace in the archived RED pytest log, not code.
- Ran no test suite, launcher, watcher, driver, audit/guard suite, GPU/process command, package command, or write operation.

The final-HEAD DRY_RUN and W&B identity check remain mandatory immediately before the token if HEAD moves again.
