**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** code P2K-1 (exp_35: the phase-(ii) launch kit; 9bd8e468 → 581a1773)

**Verdict:** REQUEST-CHANGES

## K1–K8

| Item | Status | Judgment |
|---|---|---|
| K1 — S6 / §15 conformance | **FAIL** | Most registered checks and records are present, but the phase-(i) schema gate is incomplete, subprocess timeouts violate the never-list, dry-run is not crash-proof one-shot, and post-open release refusals are not terminal. |
| K2 — safety | **FAIL** | No explicit signal/kill API, sharing, E2-FAIL, dump suppression, or relaunch path exists. However, `subprocess.run(..., timeout=...)` kills timed-out children, directly violating the no-signal/no-timeout contract. |
| K3 — correctness | **FAIL** | Manifest/CANON/digest/parser/O_EXCL/wrapper/rc checks are strong. The incomplete phase-(i) validator and non-no-follow reservation inspection remain correctness defects. |
| K4 — tests | **FAIL** | The 338-test sandbox is broad and substantially real-artifact-anchored, but it misses—and sometimes encodes—the defects below. |
| K5 — 15 readings | **FAIL** | Readings 5, 6, 7, and 11 are refuted; the other eleven are confirmed. |
| K6 — Planner commands/report | **PASS** | The CLI shapes and six-line report design are correct. They must be used only with a revised, reviewed SHIP commit. |
| K7 — SOP/audit | **PASS** | History, trailers, changed-file scope, diff hygiene, RED/GREEN evidence, and all per-commit production counts check out. |
| K8 — verdict | **REQUEST-CHANGES** | Do not merge or use `581a1773`. |

## Findings, ranked

### BLOCKING 1 — The production helper contains two killing timeouts

[exp35_phase_ii.py:268](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:268) gives Git a 600-second timeout, and [exp35_phase_ii.py:413](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:413) gives `nvidia-smi` a 120-second timeout. Python implements `subprocess.run(..., timeout=...)` by killing the child after expiry.

That contradicts S6’s never-signal rule and accepted reading 7, “no timeout anywhere.” It also means even `status` can kill its own `nvidia-smi` child.

Concrete fix:

- Remove both `timeout=` arguments; wait indefinitely and handle only normal exit or actual read/exec errors.
- Strengthen [test_exp35_phase_ii_cli.py:117](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cli.py:117) to reject a `timeout` keyword on any production `subprocess.run`, `communicate`, or `wait` call that could kill a child. The existing string search catches explicit `.kill()` but misses this implicit kill path.

Non-killing observation deadlines such as launch confirmation may return “not confirmed” while leaving the process untouched; they are not this finding.

### BLOCKING 2 — Phase-(i) “schema” validation accepts states rejected by the H35 supervisor

[exp35_phase_ii.py:314](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:314) checks the top-level key list, frozen chunk identities, headline states/rcs, E1/E2/probe results, one campaign digest, latch/lane/reservation absence, and final event text. It does not validate the real supervisor’s full evidence-bearing schema.

In contrast, the pinned supervisor validates:

- exact per-state chunk keys;
- lane/shared/start/log/end fields;
- receipt SHA-256 and campaign digest types;
- exact E1/E2/probe result schemas and probe hashes;
- each passing `final_guard`;
- baseline and training-log anchor schemas;
- event schemas;
- the registered `approved_sha256`.

See [exp35_supervisor.py:162](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:162) and [exp35_supervisor.py:234](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:234).

Consequently, a purported completed state can omit receipt hashes and final guards, carry malformed events or baseline/anchor evidence, and still be reported as complete. A subsequently committed report merely binds the hash of that invalid state; it does not repair the semantic gate.

Concrete fix:

- Port the H35 supervisor’s complete pure schema validation into the kit, including the registered approved-tuple pin.
- Then apply the extra terminal conditions: all 33 chunks `gates_verified`, no latch/lane/reservation, E2 PASS, one exact campaign digest, and the release event last.
- Add deletion/malformed-field tests for every required chunk, gate, final-guard, baseline, anchor, event, and approved-pin field. The current tests beginning at [test_exp35_phase_ii_checks.py:115](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_checks.py:115) cover headline outcomes but not the real schema.

### HIGH 1 — A crash can permit two dry runs in one orchestration

The state remains `reserved` while the runner is spawned and runs at [exp35_phase_ii.py:1114](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1114) and [exp35_phase_ii.py:1124](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1124). The first durable transition is only written after completion at [exp35_phase_ii.py:1138](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1138).

If the kit process dies after `Popen` but before that write, the state remains `reserved`. Once the original dry-run runner is gone, another `dry-run` can start in the same orchestration. This refutes reading 6’s one-dry-run guarantee.

Concrete fix:

- Durably write a terminal-on-crash `dry_run_pending`/`dry_run_started` state, including argv/environment/cwd, before spawning.
- Do not make that phase releasable or retryable by the kit; require manual review after a crash.
- Add crash tests immediately before spawn, immediately after spawn, and before rc/log publication.

### HIGH 2 — Release safety refusals are not terminal or durably recorded

After `open_mode` has admitted a releasable orchestration, [exp35_phase_ii.py:1405](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1405) judges identities, locks, processes, and all reservation records. On ambiguity, [exp35_phase_ii.py:1415](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:1415) simply returns, leaving the old releasable phase intact.

That permits silent retries and leaves no orchestration-state record of the refusal, refuting accepted reading 5. The test at [test_exp35_phase_ii_release.py:147](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_release.py:147) explicitly requires the state to remain unchanged.

Concrete fix:

- Atomically transition any post-open release refusal to `release_failed` or a terminal `release_refused`, retaining all reasons and untouched reservation identities.
- Exclude that phase from `RELEASABLE`; further action is manual Planner review, as reading 4 already requires for `release_failed`.
- Update the tests to require the durable terminal record.

### HIGH 3 — Reservation inspection follows a final symlink and can block on a non-regular file

[exp35_phase_ii.py:606](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:606) performs `lstat`, but [exp35_phase_ii.py:613](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:613) then uses ordinary `open`, which follows a non-dangling symlink. A symlink to JSON containing `{"holder":"exp22"}` is therefore accepted as an unobjectionable non-lane reservation at [exp35_phase_ii.py:684](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:684). A FIFO can block the inspection entirely.

This refutes reading 11: only a well-formed regular record belonging to another holder is allowed on a non-lane card; links and malformed/non-regular entries must refuse.

Concrete fix:

- Read through one `O_NOFOLLOW | O_NONBLOCK` descriptor and require `S_ISREG`, as `snapshot()` already does.
- Classify any link/non-regular file as unexplained or malformed; unreadability remains fail-closed.
- Add non-dangling symlink, FIFO, empty-holder, and malformed-holder tests. The current cases at [test_exp35_phase_ii_cards.py:84](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cards.py:84) cover only regular files and a dangling link.

### NIT 1 — Each mode log advertises a hash that is not the final file’s hash

[exp35_phase_ii.sh:45](/home/yixunhu/codespace/exp-35-phase-ii-kit/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh:45) hashes the log and then appends the line containing that hash. The printed value is the pre-footer prefix hash, although it is labeled `log ... sha256`.

Fix by either printing the final hash outside the log, writing a sidecar, or labeling it explicitly as the prefix hash. Add an actual digest comparison rather than the regex-only check at [test_exp35_phase_ii_cli.py:75](/home/yixunhu/codespace/exp-35-phase-ii-kit/src/tests/test_exp35_phase_ii_cli.py:75).

## K5 — the 15 readings

| # | Judgment | Reason |
|---|---|---|
| 1 | **PASS** | Exactly six recognized, unindented report lines are required; malformed, duplicate, unknown, or missing occurrences refuse. |
| 2 | **PASS** | Both kit blobs must match `EXPECT_KIT_COMMIT`, remain tracked/clean, and each orchestration records and requires its original kit commit. A later HEAD with identical reviewed blobs is deliberately allowed. |
| 3 | **PASS** | `reserved` and `dry_run_passed` are releasable withdrawals before `launch_pending`. |
| 4 | **PASS** | `reserving`, `launch_pending`, `spawning`, `running`, `reserve_failed`, and `release_failed` are not releasable. |
| 5 | **FAIL** | Post-open release refusals leave the orchestration in its prior releasable phase. |
| 6 | **FAIL** | The 3-hour launch-age gate is correct, but a crash after dry-run spawn can permit a second dry run. |
| 7 | **FAIL** | Git and `nvidia-smi` have killing `subprocess.run` timeouts. |
| 8 | **PASS** | Only the exact pinned warning text is tolerated, only during the preamble. Multiple copies are necessary because the real H35 transcript contains two. |
| 9 | **PASS** | All lanes require the last s43 training line to be `END`, any exit code. GPU 1 additionally requires `EVAL_DONE` and no live evaluator; all three exp30 files are pinned. |
| 10 | **PASS** | Supervisor/runner anywhere, wrapper anywhere, and evaluator/no-flash/probe under the run worktree are scanned; unreadable entries refuse. |
| 11 | **FAIL** | A non-dangling symlink can masquerade as another holder’s well-formed record. |
| 12 | **PASS** | Orchestration state is under the run worktree’s exp_35 folder; CLI mode logs are beside the kit. |
| 13 | **PASS** | A later successful-run orchestration’s dry run encounters exact SKIP lines, which the parser refuses. There is no automatic resume/relaunch path. |
| 14 | **PASS** | `launch` returns zero only after observing state `running` and both complete identities alive. |
| 15 | **PASS** | Real `/proc` use in tests is confined to the test process and sandbox-spawned children; production scans use the fake proc tree. |

## K6 — Planner procedure and command shapes

Do not execute these against `581a1773`. After a revised commit receives SHIP, use its full SHA as `<SHIP_COMMIT>`.

```bash
EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh status

EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh reserve 5b065801e9d1c1844461ec84cd0558702c714326 --lanes 0 --report worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun.md --report-sha256 <REPORT_SHA256>

EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh dry-run 5b065801e9d1c1844461ec84cd0558702c714326 --orch <ORCHESTRATION_DIRECTORY_PRINTED_BY_RESERVE>

EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh launch 5b065801e9d1c1844461ec84cd0558702c714326 --orch <ORCHESTRATION_DIRECTORY_PRINTED_BY_RESERVE>

EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh status --orch <ORCHESTRATION_DIRECTORY_PRINTED_BY_RESERVE>

EXPECT_KIT_COMMIT=<SHIP_COMMIT> bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh release 5b065801e9d1c1844461ec84cd0558702c714326 --orch <ORCHESTRATION_DIRECTORY_PRINTED_BY_RESERVE>
```

The Planner must manually send the phase-(i) report to Yixun, create and commit the clean report file, and include exactly these six lines with no other occurrence of `exp35-phase-ii`:

```text
exp35-phase-ii h35 5b065801e9d1c1844461ec84cd0558702c714326
exp35-phase-ii phase-i-state-sha256 <FINAL_STATE_SHA256_PRINTED_BY_STATUS>
exp35-phase-ii lanes 0
exp35-phase-ii second-lane-decision none
exp35-phase-ii eta <MEASURED_PER-CELL_TIME_AND_PHASE-II_ETA_REPORTED_TO_YIXUN>
exp35-phase-ii prior-runs none
```

A two-lane launch must instead name the complete frozen lane list and the recorded Planner/Yixun second-lane decision. GPU 1 additionally needs terminal table-job evidence. Reservation must wait for the s43 training `END` line.

## Planner audit and probes

Confirmed:

- HEAD `581a1773cd5ba28a80d3459537bbafb7d410b93f`, branch `exp-35-phase-ii-kit`, clean worktree.
- Both supplied kit SHA-256 values match.
- All 13 commits carry the exact Opus trailer.
- The Coder’s “9 files” is wrong: there are 12 `test_exp35_phase_ii*.py` files, plus the sandbox module and two kit files. No other non-log file changed.
- `git diff --check` is clean for all changed Python/shell files.
- Independent committed-counter results are exactly: `178`, `97`, `147`, `80`, `107`, `102`, `81`, `89+8`, `109`, `71+6`, `88`, `47+1`, `16+7`; every production total is below 200.
- The committed final GREEN log, SHA-256 `63a39ec7d00b8d3d5c87c6cedcda6ff76a3c8947d0dbc25ffb385191011e9888`, reports `338 passed in 40.61s`.
- H35 object probes independently reproduced: 346 manifest rows; exact 68 selected rows and counts; orders 262–329; all dumps enabled; name digest `6a88112f…d427`; 16 excluded E2-FAIL rows; both manifest hashes and the step-40000 CANON entry.
- The run worktree’s runner-relevant tree is clean and independently digests to `da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1`.
- All three live exp30 source hashes match their pins.
- The exact H35 runner output grammar and supervisor schema/end event were inspected through committed Git objects.

I did not run pytest, the kit—including `status`—the runner, supervisor, any GPU command, or any process probe; did not access `.gpu_reservations`; and made no filesystem, process, package, or environment change.

Operational disposition: P2K-1 is awaiting Coder fixes and re-review. Do not merge or use this commit.
