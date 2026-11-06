**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** Planner records scripts P17 (docs_1106z, docs_1106za, exp35_phase_ii_wait.sh; e9d2e21f → b919f695)

**Verdict:** REQUEST-CHANGES

No BLOCKING finding. One HIGH records-provenance defect requires correction. The wait loop remains advisory and cannot bypass the kit’s independent admission checks.

## Per-item status

| Item | Status | Assessment |
|---|:---:|---|
| Q1 — wait loop | **FAIL** | It is read-only and its s43 predicate matches the kit closely, but deadline, decimal-argument, and fail-open diagnostic edges remain. These are notification-only NITs, not launch-safety defects. |
| Q2 — records truthfulness | **FAIL** | The bindings, forecasts, issue entry, tracker, CLAUDE clause, and next-step locator are otherwise consistent, but a 12:54 card observation is incorrectly labelled “CURRENT 13:44.” |
| Q3 — records mechanics | **PASS** | Both scripts use one transaction each, exact anchor counts, rerun guards, verified sources, explicit destination-only staging, and correct substituted literals. |
| Q4 — next operational record | **PASS** | Requirements are frozen below. No reserve, dry-run, or launch occurred in P17. |

## Ranked findings

### HIGH 1 — the handoff advances the observation timestamp without new evidence

[docs_1106za_compaction.py:57](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106za_compaction.py:57) generates “CURRENT 13:44 EST Nov 6 (nvidia-smi…)” in [HANDOFF.md:8](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/HANDOFF.md:8), but its cited observation is timestamped `2026-11-06T12:54:18-05:00` at [card_observation.txt:1](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_2026-11-06_12-54-18_card_observation.txt:1). Nothing pinned in P17 establishes those processes or memory values at 13:44.

The forecasts are properly labelled; the process state is not. The same ongoing-state inference appears in the refreshed tracker and issue entry.

Concrete fix: either pin a genuinely contemporaneous observation, or say “LAST OBSERVED 12:54 EST; not re-observed during the 13:44 compaction,” and qualify the tracker/issue wording as an inference or forecast. Do not change r4 or the frozen command.

### NIT 1 — the advertised window can end up to 59 seconds early

At [exp35_phase_ii_wait.sh:35](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait.sh:35), the loop exits whenever another full 60-second sleep would exceed the deadline. For a 7,000-second window, a regular cadence can exit at 6,960 seconds; a 60-second window can exit almost immediately if setup crosses a wall-clock second. An s43 transition during that advertised remainder can therefore be missed.

Concrete fix: calculate the remaining duration, sleep `min(60, remaining)`, and perform the final observation at the deadline before emitting `window-ended`.

### NIT 2 — leading-zero arguments are interpreted as octal

[exp35_phase_ii_wait.sh:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait.sh:11) accepts any digit string, but Bash arithmetic interprets values such as `0100` as octal—giving a 64-second window—and `0080` causes an arithmetic-base error rather than the promised clean rc 2 refusal.

Concrete fix: restrict the input to at most four digits, convert with `MAX=$((10#$MAX))`, then perform the range check.

### NIT 3 — some read/parse failures are treated as ordinary state

The `grep | tail` pipeline at [exp35_phase_ii_wait.sh:16](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait.sh:16) hides a `grep` open/read failure behind `tail`’s success. A disappearance between the `-r` test and read therefore does not immediately produce `queue-log-unreadable`. Similarly, [lines 29–31](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii_wait.sh:29) treat any rc-0 `nvidia-smi` output lacking the literal GPU-0 UUID as free, whereas the kit strictly validates every row at [exp35_phase_ii.py:597](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.py:597).

Concrete fix: capture the queue-read status explicitly and validate every non-empty compute-app row before counting a free sample. Route malformed output to the diagnostic event. Encode diagnostic text so it cannot turn the claimed single EVENT line into multiple physical lines.

A process can also start after the second sample and before notification. That ordinary TOCTOU race is acceptable only because this is advisory: `reserve` repeats strict observations 60 seconds apart and checks again after acquisition.

## Q4 — first operational phase-(ii) record

Before operation, archive this P17 review verbatim and correct HIGH 1. Do not edit the currently running wait script; apply wait-loop corrections only after that invocation exits.

The reserve command must remain exactly:

```bash
EXPECT_KIT_COMMIT=7743ef4faadfc545dcce8dc5f7b58364babf1719 bash worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh reserve 5b065801e9d1c1844461ec84cd0558702c714326 --lanes 0 --report worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun_r4.md --report-sha256 054d8337a634b72b836d6c788ef96591b6f5608c66584dfcbfb3f7aeb8d98957
```

The operational record must:

1. Preserve the trigger/event and bind kit `7743ef4f…`, H35 `5b065801…`, phase-(i) state `7b6cfc8c…`, lane `0`, decision `none`, prior runs `none`, and r4 `054d8337…`.
2. Execute `reserve` once and record the exact command, rc/output, final mode-log path/hash, returned orchestration directory, and resulting `state.json` hash.
3. Stop without retry on any refusal or crash.
4. Only after successful reserve, execute exactly one dry-run, one launch, and `status --orch <exact directory>`, recording every command, rc, outcome, mode-log hash, and state hash.
5. Never hand-edit reservations, infer success, signal a process, or release before the runner exits.
6. Use one guarded records transaction and explicit destination-only staging for the resulting evidence.

## What I verified

- P16 is byte-identical to its extract at `ddcd874e…`; r4 hashes to `054d8337…`; the run-worktree state hashes to `7b6cfc8c…` at H35 `5b065801…`.
- The wait loop is 37 lines, hashes to `25e67fea…`, matches its scratch source, and passes `bash -n`.
- The shell s43 selection/final predicate matches the kit’s `S43_LINE.fullmatch` plus exact integer-END rule for the ASCII queue format.
- Both transaction manifests’ operations and hashes match their commits; parentage is `e9d2e21f → 640fd384 → b919f695`; HEAD equals `origin/localization-exp`.
- Explicit `git add` lists exactly match each transaction’s destinations; no placeholders remain.
- The default-A/no-answer history is consistent with the sent phase-(i) report, its “unless Yixun decides otherwise” rule, and the unchanged six-line `second-lane-decision none` binding.
- The cost, forecasts, tracker row, issue-report details, and CLAUDE free-card clause otherwise match r4, the pinned observation, P16, and the kit.

Probes were limited to read-only `git status/log/show/diff/diff-check/diff-tree/rev-parse/cat-file/ls-files`, `sed`, `rg`, `nl`, `find`, `sha256sum`, `wc`, `cmp`, and `bash -n`. I did not execute a records script, helper, wait loop, kit mode, test, GPU command, or process probe; did not access `.gpu_reservations`; and made no filesystem or environment change.
