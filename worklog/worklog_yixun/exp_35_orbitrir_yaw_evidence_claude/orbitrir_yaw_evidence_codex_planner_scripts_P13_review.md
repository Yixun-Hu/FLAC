**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** Planner records scripts P13 (docs_1106t, docs_1106u, exp35_cell_times.py, the completion commands, the s43-end watch; 5ff10984 → 9ee5b6ba)

**Verdict:** REQUEST-CHANGES

No BLOCKING finding and no scientific rerun is required. The phase-(i) state, guard, collector verdicts, evidence copies, and P12 provenance repair pass. Two records-only HIGH corrections are required before phase-(ii) `reserve`: qualify and bind the ETA derivation truthfully, and make the operator-facing HANDOFF actually reflect the completed state.

## Per-item status

| Item | Status | Assessment |
|---|:---:|---|
| Q1 — P12 requirements | **PASS** | P12 is verbatim; the Coder message is labeled unpreserved session evidence; the P2K-1 prompt/source chain and readings chronology verify; quoted-heredoc rule and `docs_1106n` prohibition stand; round-6b remains preserved; the first operational phase-(ii) binding is not yet due. |
| Q2 — claim truthfulness | **FAIL** | Scientific and completion claims pass, but the ETA is not sufficiently separated into measurements, extrapolations, and Planner forecasts, and HANDOFF still advertises phase (i) as running. |
| Q3 — timing script and watch | **PASS** | Both are read-only. I found no classification defect that changes the ETA. The table is a proxy dataset, not direct free-card B-F/P1 measurement. The watch has one notification-only PID-reuse NIT. |
| Q4 — readiness before `reserve` | **FAIL** | The report is technically bound correctly, but its ETA wording must be corrected and recommitted first. The stale HANDOFF should also be repaired. |
| Q5 — next-script requirements | **PASS** | Listed below. |

## Artifact status

| Artifact | Status |
|---|:---:|
| `docs_1106t_p12.py` / `fc46d5b8` | **PASS** |
| `exp35_cell_times.py` as a read-only table generator | **PASS** |
| Final state, guard, kit-status, collector, and 37-file evidence copy | **PASS** |
| `docs_1106u_phase_i_done.py` / user-facing report and HANDOFF | **FAIL** |
| Inline s43-end watch | **PASS** |

## Ranked findings

### HIGH 1 — The report does not clearly distinguish measurements from extrapolation and forecast arithmetic

[report:19](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_phase_i_report_to_yixun.md:19), [docs_1106u_phase_i_done.py:97](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106u_phase_i_done.py:97), [cell-times output:9](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_2026-11-06_05-35-38_cell_times.txt:9), [exp35_cell_times.py:9](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_cell_times.py:9)

The report places “16–20 min (OrbitRIR) on a free card” beside a parenthetical labeled “measured,” but:

- There is no free-card B-F row.
- There is no free-card P1 row.
- The observed free-card group medians are YAW eval 11.9/12.6 minutes and CANON TRAJ 13.5/14.8 minutes. The table also contains an alone-cell maximum of 20.5 minutes, so “cells alone 11.9–14.8” is only a range of selected group medians.
- Scaling the shared B-F medians by the YAW shared/alone ratios gives approximately 17.1 and 20.7 minutes, not a direct 16–20-minute measurement.
- Applying the report’s own bands to 48 P1/CANON cells and 20 B-F cells gives about 14.9–18.7 hours on one card. The reported 17–24-hour range and finish times therefore include unstated Planner safety/overhead judgment.
- The output header pins the script and snapshot, but not the exact argv or the two exp_25 runner-log paths/digests.

The s43 forecast itself is arithmetically sound: 3,020 remaining steps × 4.260 seconds is 3.574 hours, placing completion near 14:47 from the 11:13 observation. The 4.9-hour table-job cost is present in the committed plan/research notes, but it is a prior estimate, not a measurement made here.

Concrete fix: before `reserve`, commit and send a corrected report that explicitly labels:

1. the observed rows and whether values are medians or maxima;
2. the B-F/P1 free-card values as proxy-based extrapolations;
3. the exact scaling formula;
4. the 18–19-hour center, 17–24-hour range, two-card time, and finish times as Planner forecasts with assumed overhead/safety margin.

Pin the exact input log paths/digests and snapshot path/digest if the invocation is established; otherwise label the invocation unpreserved session evidence. Keep exactly six kit fields and pass the corrected report’s new SHA-256 to `reserve`.

### HIGH 2 — HANDOFF still says phase (i) is running

[HANDOFF.md:3](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/HANDOFF.md:3), [HANDOFF.md:7](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/HANDOFF.md:7), [docs_1106u_phase_i_done.py:133](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106u_phase_i_done.py:133)

The phase-completion script appends completion text after the old kit-procedure phrase, but it leaves the operator-facing header at “Last updated: 2026-11-05 20:23” and the current-status bullet beginning “Running (ours): exp_35 PHASE (i) RUNNING … supervisor pid 685273.” A later suffix says phase (i) completed, but a fresh operator is still presented with contradictory current state and a dead historical PID.

Concrete fix: replace the top/current snapshot with a concise 2026-11-06 entry: phase (i) complete, nothing of exp_35 running, no latch/reservation, phase (ii) pending the corrected report and reviewed kit. Preserve the old running details only as history.

### NIT 1 — The evidence breakdown says 33 chunk logs, but there are 32 cell-chunk logs

[docs_1106u_phase_i_done.py:128](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106u_phase_i_done.py:128), [worklog:234](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_evidence_worklog.md:234)

The 37-file manifest is correct: one supervisor log, 32 cell-chunk logs, one final state, and three probe artifacts. Chunk 5 is the probe and has no `exp35_supervisor_*chunk05*.log`.

Concrete fix: change “33 chunk logs” to “32 cell-chunk logs; the probe is represented by its JSON, digest sidecar, and probe log.”

### NIT 2 — Broad staging swept 32 transcript logs outside the transaction manifest

[git_records phase completion:14](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_records_2026-11-06_11-21-46_phase_i_done.log:14)

`git add <entire exp_35 folder>` committed 32 earlier Codex/Planner transcript logs that were not among the transaction’s 51 operations. All 51 operations and 46 sources verify, and Git now preserves the extra logs, so this is not an evidence-loss defect. It does make the commit broader than the transaction manifest and commit summary describe.

Concrete fix: attest those 32 paths/content digests in the next records correction and use an explicit staging list derived from transaction destinations in future commits.

### NIT 3 — The inline watch treats PID-directory existence as identity

The loop checks only `/proc/3027719` and `/proc/3028196`. PID reuse could delay its “both ranks gone” notification, and loss of only one rank does not wake it. It cannot authorize or launch anything, and the reviewed kit independently checks the queue ending and process/card state, so this does not affect safety or the ETA calculation.

Concrete fix for any future revision: snapshot and compare complete process identities, including start time, or treat the queue’s authenticated END line as the sole success condition and identity mismatch as an informational wake-up.

## Q1 — P12 requirements

All seven pass:

- P12 archive and extract are byte-identical at `686ce5da…`.
- The archived P2K-1 prompt is byte-identical to its source at `fe06b0f0…`, occurs verbatim in transcript `645a0c2b…`, and the session ID matches.
- The correction labels the Coder hand-back message unpreserved session evidence.
- P2K-1 refuted readings 5/6/7/11; P2K-2 records them resolved at `7743ef4f`.
- No `docs_1106n_s15ok.py` manifest appears in the reviewed time window; only `docs_1106t` and `docs_1106u` ran.
- `exp-35-round6b` remains at full head `9cc810e8c094c191511ded556c43641aa5f60984`, with no upstream.
- The first operational phase-(ii) record is not yet due: no orchestration directory exists and no `reserve` has run.

## Q2 — Scientific and completion claims

Apart from the findings above, they pass.

- Final state: 33/33 `gates_verified`, 32 cells plus the probe, every cell marked shared on GPU 1, no latch/lane/reservation, final release event at 11:12:33.
- Guard: status OK, baseline 27941, 9,039 steps, mean 5.886381 s/step, slowdown ×1.38178, calculated loss 4.0836 hours, worst windows within limits.
- Collector: exactly 68 missing phase-(ii) cells; E1 equal twice; E2 PASS on 14 rows; Tier 2-C4 PASS; six scored C4 pairs meet both caps; pooled p99 0.006189 vs 0.005707; T-dup exact; alarm absent.
- Tier 3: all 14 checks have `pass_c4=true`, all 14 have `pass_r=false`, overall pass false.
- Every quoted T60 value agrees with the pinned results Markdown/JSON.
- D2 therefore does not materially weaken the record: the creation script asserted less than the report quoted, but the quotations themselves verify.
- D3 does not weaken the binding: the committed report contains exactly the six unindented fields, once each, with the correct H35, state hash, lanes, decision, ETA, and prior-runs values.
- All 37 evidence paths pass `sha256sum -c` in both the main checkout and run worktree.

## Q3 — Timing script and watch

`exp35_cell_times.py` is read-only. It opens logs/snapshot, performs `stat`, prints a table, and writes nothing. The four-line association matches the runner-log layout: the current cell’s GPU-sharing observation is immediately before its checkpoint revalidation and RUN line. I found no structural misclassification that changes the estimate.

D4 does reduce evidentiary strength:

- birth-to-mtime is a filesystem proxy rather than timestamps recorded by the evaluator;
- absence of a nearby `shared with pid` line is interpreted as alone;
- the B-F free-card estimate is extrapolated, not observed;
- the exact invocation/input hashes are not in the output.

Those limitations require qualification, not a rerun of phase (i).

The s43 watch is read-only and safe; its only defect is the notification-only PID identity NIT above.

## Q4 — Before phase-(ii) `reserve`

The binding mechanics pass:

- Report is committed at `9ee5b6ba`, clean at its path, SHA-256 `d497cd0237ba923cbc0510957a45304797de0eadb0be865cff045608cdb92bb7`.
- State field is the full `7b6cfc8…` digest.
- Run worktree is on branch `exp-35-orbitrir-yaw` at H35 `5b065801…`; its tracked tree is clean. Its numerous untracked campaign outputs are expected and were outside the kit’s runner-relevant tree.
- The archived kit status found the relevant digest correct, locks unheld, no exp_35 process, no reservation, and no orchestration.
- The remaining cell logs, plans, preflights, receipts, and dumps are still in the run worktree. That deliberate end-of-campaign deferral is not a kit precondition, but they must not be cleaned or overwritten and must be manifest-pinned before final cleanup/branch integration.

The corrected report and HANDOFF are the only required records actions before `reserve`.

## Q5 — Requirements for the next script

1. Archive this P13 review verbatim.
2. Make a records-only correction to the report: distinguish observed medians/maxima, proxy extrapolation, and Planner forecasts; pin the exact timing inputs/formula or label missing invocation provenance.
3. Preserve exactly six valid kit fields, update the ETA field consistently, commit the corrected report, send the correction to Yixun, and use its new full SHA-256 for `reserve`.
4. Repair the top/current HANDOFF status and the “32 cell-chunk logs” wording.
5. Attest the 32 transcript logs swept into `9ee5b6ba`; future staging must enumerate transaction destinations rather than stage the whole folder.
6. Preserve the untracked phase-(i) cell logs/receipts/dumps and manifest them before cleanup or the final campaign branch commit.
7. Keep all P12 standing rules: quoted heredocs only; never rerun `docs_1106n_s15ok.py`; preserve round-6b/no-upstream; and have the first operational record bind the corrected report path/hash, full state hash, exact orchestration directory, exact commands, every mode-log hash, and outcomes.

## What I verified

I used only read-only `git status/log/show/diff/name-status/rev-parse/for-each-ref/ls-files`, `sha256sum`, `cmp`, `jq`, `rg`, `sed`, `nl`, `find`, `comm`, and `python -B -c`.

Those probes verified script/source/archive digests; prompt/transcript containment; both transaction manifests against parent and child Git blobs; exact operation/commit path sets; all 37 source and destination evidence hashes; state/guard/status/result fields; report tracking and six fields; branch identities; run-worktree state; ETA arithmetic; and absence of a phase-(ii) orchestration. `git diff --check` findings were confined to preserved transcript/evidence-log whitespace, not production code.

I did not execute any reviewed script, helper, kit mode, collector, guard, counter, test, watcher, GPU command, or process probe; did not access `.gpu_reservations`; and made no filesystem, package, environment, or process change.
