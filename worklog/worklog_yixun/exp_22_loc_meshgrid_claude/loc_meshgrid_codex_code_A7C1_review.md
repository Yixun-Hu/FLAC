**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code A7-C1 (exp_22 R2: the no-flash pre-flight tooling; 9847b45d → 9206da2f) —

**Verdict:** REQUEST-CHANGES

The architecture is strong and most of A7 r1–r5 is implemented faithfully, but the dispatch evidence can hash bytes different from those it validated. Two additional recovery/identity gaps conflict with the approved fail-closed state contract. These must be fixed before the merge that becomes PIN2.

## Ranked findings

### BLOCKING 1 — Dispatch records do not necessarily hash the evidence bytes that were validated

[`Supervisor.evidence()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:510) reads and validates the log and JUnit artifact. [`Supervisor.record()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:530) then independently re-reads the log and hashes the JUnit/outputs. Finally, [`line()`](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:454) hashes them a third time for the state’s integrity baseline.

A log can therefore pass the backend rule, change before `record()`, and be recorded as `admitted: true` with the changed hash. For example, appending `NO-FLASH LAUNCH REFUSED` after `evidence()` returns but before `record()` makes the dispatch hash and all subsequent integrity checks bind the unvalidated log. The same race exists for T’s JUnit evidence and the V/A text.

The current mutation tests act only after a previous dispatch record already exists ([test supervisor:181–228](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_supervisor.py:181)), so they do not exercise this interval.

Concrete fix:

- Read each log, JUnit file, input and output through one no-follow descriptor, retaining bytes plus device/inode.
- Validate those exact snapshots.
- Put hashes of those same snapshots in the dispatch record and state; do not re-read to establish the expected hashes.
- Before the next line and before final release, continue authenticating the paths against those recorded identities/hashes.
- Add a test that mutates the log and JUnit/output after evidence evaluation but before record creation; it must not produce an admitted record or release the reservation.

### HIGH 1 — `child_identified` can contain only a PID

`identify()` stops retrying when the child exits even when its `/proc` cmdline never matched, and it latches a mismatch only while the child is still running ([supervisor.py:465–482](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:465)). The result is nevertheless persisted as `child_identified` ([supervisor.py:447–449](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:447)).

The schema explicitly permits an identity without `start_time`, cwd or cmdline ([supervisor.py:116–119](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:116)), after which `alive()` falls back to PID existence alone ([supervisor.py:203–211](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:203)). That contradicts r3(e)’s mandatory child identity and “never a PID alone.”

Concrete fix: permit the `child_identified` transition only with all four `ID_KEYS` and the expected cmdline hash. If a child exits before that identity is captured, fail closed and keep the reservation rather than recording an identified child. Add a deterministic immediate-exit/transient-empty-cmdline test.

### HIGH 2 — A stopped state is silently treated as resolved if its reservation disappears

`TERMINAL` includes both `released` and `stopped` ([supervisor.py:51–52](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:51)), so `active_reasons()` ignores a stopped state ([supervisor.py:233–245](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:233)). The test explicitly deletes the reservation and declares the stopped run resolved without invoking the Planner release ([test planner:26–30](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_planner.py:26)).

A7 r3(e) says `stopped` keeps the reservation until the Planner releases it; only `released` is resolved. This matters when the reservation is removed or replaced externally—the branch can admit another run without the authenticated recovery step.

Concrete fix: only `released` should be terminal for `active_reasons()`. A stopped/crashed state must block until `planner_release()` records `released`. If a foreign replacement exists, wait for its owner to remove it, then the existing absent-record release path can close the old state safely.

### NIT 1 — Two source files have extra blank lines at EOF

Remove the final blank lines at [exp22_r2c_supervisor.py:603](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:603) and [test_exp22_r2c_card_rule.py:163](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/test_exp22_r2c_card_rule.py:163).

## K1–K7

| Item | Status | Review |
|---|---|---|
| K1 — A7 conformance | **FAIL** | PIN2 constitution, exact T/G/V/P/A order, environments, fresh directories, unchanged scientific arguments, no reachable shard, r3 backend rule, r5 T rule, dispatch location and schema are otherwise faithful. BLOCKING 1 leaves the dispatch hash unbound to the bytes actually validated. |
| K2 — detector/capture/launcher | **PASS** | The candidate rule is centralized in [common.py:290–328](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_common.py:290); unreadable input ends detection at [common.py:343–345](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_common.py:343). Capture is O_EXCL/fsynced and cadence-checked at [evidence.py:307–373](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:307). The shell implements capture → seal → fresh launch → lock → reserve → detach → confirm at [preflight.sh:88–177](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:88). |
| K3 — supervisor | **FAIL** | Durable transition ordering, pre-line checks, during-line polling, stop-first, exact reservation release and child-held locking are sound. HIGH 1 violates the authenticated `child_identified` schema; HIGH 2 bypasses the Planner’s released-state transition. |
| K4 — live-system safety | **PASS** | No path signals an existing experiment, exp_35 process or running line child. Python timeouts may terminate only helper processes the tool itself spawned. The designed writes are confined to the reservation and registered `outputs_loc/exp22` paths. Because `outputs_loc` is deliberately pinned to `/home/yixunhu/codespace/FLAC/outputs_loc` ([preflight.sh:24,69](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:24)), it does write scoped exp_22 outputs inside Yixun’s main checkout, but no tracked source. NAS and exp_35 are read/probed only. The line list contains no shard command. |
| K5 — tests/evidence | **FAIL** | The RED/GREEN history is credible; the real pinned-launcher test uses the real transformer and real argument parser, and the T outcome matrix covers the registered branches. The full supervisor uses stand-in scientific targets appropriately. It passes because no test mutates an artifact in the validation→record interval or forces an exited-before-identity child; those omissions correspond directly to BLOCKING 1 and HIGH 1. |
| K6 — SOP | **PASS** | Counts reproduce exactly. All 15 commits carry the Opus trailer; c12 is logs only. c7 is 269 production lines, but it is one cohesive state machine and the repository rule says commits are “generally” below 200; the exception is disclosed and defensible. No dependency installation or environment mutation is evidenced. NIT 1 remains. |
| K7 — verdict | **FAIL** | REQUEST-CHANGES. Fix BLOCKING 1 and HIGH 1–2, preserve the four PIN2 bytes and frozen meshgrid production code, rerun the CPU evidence ladder, and re-review the new range before merge. |

## The 19 resolved ambiguities

| # | Status | Judgment |
|---:|:---:|---|
| 1 | **PASS** | T’s exact flags/order and eleven identities are pinned at [evidence.py:58–71](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:58). |
| 2 | **PASS** | Phase-(ii) intent is represented by any exp_35 reservation; no second intent format is invented ([common.py:309–316](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_common.py:309)). |
| 3 | **PASS** | First unreadable observation produces `EVENT UNREADABLE`, rc 4. |
| 4 | **PASS** | Bash remains the detector parent; it does not `exec` the Python loop ([exp22_free_card_detect.sh:11–26](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_free_card_detect.sh:11)). |
| 5 | **PASS** | 1..7200 is enforced and GPU0 wins ties ([common.py:348–356,376–378](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_common.py:348)). |
| 6 | **PASS** | Capture accepts only GPU0/GPU1 and creates the local-second RUN identifier ([preflight.sh:182–189](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:182)). |
| 7 | **PASS** | Capture record creation precedes the first poll, every row is fsynced, and the seal follows passing checks only. |
| 8 | **PASS** | Polls are scheduled from `t0 + k×POLL_S`; a start more than 15 seconds late refuses without catch-up ([evidence.py:315–329](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:315)). |
| 9 | **PASS** | Last-poll freshness is checked before locking and again at the last look ([preflight.sh:148–159](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:148)). |
| 10 | **PASS** | Normal stops retain the reservation; only the immediate post-O_EXCL changed-world path releases during reservation ([supervisor.py:193–200,338–342](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:193)). |
| 11 | **PASS** | A foreign process latches no-further-line, never signals the child, and makes the dispatch refused ([supervisor.py:484–508](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:484)). |
| 12 | **PASS** | Wrapped and CPU environments are exact; inputs are compared to earlier recorded output hashes; V/A text is enforced ([evidence.py:245–254](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:245), [supervisor.py:510–528](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:510)). BLOCKING 1 concerns which snapshot those hashes bind. |
| 13 | **PASS** | Every normally completed line reaches `line_complete` before judgment and receives an admitted/refused dispatch record ([supervisor.py:451–460](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:451)). |
| 14 | **PASS** | Reservation records include `run` ([supervisor.py:82–96](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:82)). |
| 15 | **PASS** | Passing the lock fd to the line child is the right fail-closed behavior ([supervisor.py:300–304](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_supervisor.py:300)). It does not permanently deadlock recovery: kernel release follows child exit. A still-live or hung child intentionally prevents reservation release and requires operator adjudication. |
| 16 | **FAIL** | Planner release and `active --holding-lock` exist, but stopped states are treated as resolved without that release. See HIGH 2. |
| 17 | **PASS** | Production has one registered-config block per tool; sandbox copies replace only that block ([conftest.py:85–133](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/conftest.py:85)). |
| 18 | **PASS** | FLAC_EMA, AGREE_fullAR, depth-map and DINO cache checks precede capture ([preflight.sh:41–46,72–75](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_preflight.sh:41)). |
| 19 | **PASS** | The parser takes the whole-line summary immediately after the single short-summary outcome block ([evidence.py:162–182](/home/yixunhu/codespace/exp-22-a7-noflash/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/tools/exp22_r2c_evidence.py:162)). |

## Planner-audit reconciliation

Confirmed:

- 15 commits and 15/15 Opus trailers; last code commit `fb0d3f19`; `9206da2f` changes only three logs.
- The non-worklog delta versus both `9847b45d` and `ad6eb82d` is exactly the four PIN2 paths.
- All four SHA-256 values match the stated values and are byte-identical to exp_35 `5b065801`.
- `overlays/` tracks exactly the two permitted files.
- `localize_meshgrid.py` and `src/localization/` are identical to `701926e5`.
- Tool hashes match all six stated hashes.
- The production/test additions reproduce exactly: `143/370, 130/148, 106/124, 56/189, 126/198, 93/140, 108/132, 145/163, 81/84, 269/406, 179/230, 0/65, 5/17, 0/8`.
- Committed evidence reports 251 tool tests passed, 39 copied-launcher tests passed, and 6,226 collected with zero collection errors. I did not rerun them.
- The disclosed identify race and partial log are present; final mode history contains no committed chmod slip.

Qualifications:

- The tools directory contains 11 `test_*.py` files; the count becomes 12 only when the copied `src/tests/test_exp35_no_flash_launch.py` is included.
- Full-range `git diff --check` also reports trailing whitespace embedded in preserved pytest failure logs. Excluding `*.log`, it reports exactly the two source EOF findings above.

## Read-only probes performed

I used only `git status`, `rev-parse`, `log`, `show`, `diff`, `diff-tree`, `ls-tree`, `cat-file`, `git grep`, `rg`, `sed`, `awk`, `jq`, `sha256sum`, and a `python -B` AST-based read-only line counter. I did not run pytest, any detector, launcher, supervisor, gate, probe, driver, GPU command, or operational process query; did not access `.gpu_reservations`; and did not inspect, attach to, signal, or modify any running process or file.
