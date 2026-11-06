**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records scripts P7 (the first helper-based batch docs_1105m … docs_1105t + watcher v2; 4b2c718e → e5813bba)

**Verdict:** REQUEST-CHANGES

No BLOCKING finding. The helper was copied byte-identically and the committed bytes/manifests are internally coherent, but several HIGH issues remain: the E1 record contains incorrect and unbound measurements, watcher v2 does not fully resolve P6, four scripts use runtime-derived evidence digests, every script makes its HEAD expectation tautological, one committed destination bypassed the helper, and the E1 script is not rerun-safe.

## Per-script status

| Script | Commit | Status | Assessment |
|---|---:|---:|---|
| `docs_1105m_exp22_a7.py` | `4b2c718e` | **FAIL** | Helper/transaction use is sound, but the issue-report digest is computed from the source itself, HEAD is self-derived, and several technical/history claims have no declared source. A7 r1 also contains the defects later identified by its review. |
| `docs_1105n_p6.py` | `7ecac76d` | **FAIL** | The P6 review and three corrections are faithful, but HEAD is self-derived; live process/smoke-test/re-arm claims are unbound; and its description of watcher v2 overstates the implementation. |
| `docs_1105o_exp22_a7r2.py` | `c6d34d51` | **FAIL** | Review/triage content and all eleven node IDs verify, but the A7 review was written directly by `cp` and committed outside the transaction. HEAD is also self-derived. |
| `docs_1105p_exp22_a7r3.py` | `09221e0c` | **FAIL** | The r2 review is copied faithfully, but HEAD is self-derived and the review source is read twice. Its JUnit zero-XPASS rule was not executable, as the next review found. |
| `docs_1105q_exp22_a7r4.py` | `944ae6b0` | **FAIL** | Runtime-derived review digest, self-derived HEAD, duplicate source read, and two factual overstatements in r4: universal strict-XPASS behavior and “Coder round starts per Codex’s condition.” |
| `docs_1105r_exp22_a7r5.py` | `9847b45d` | **FAIL** | Content is faithful to the r4 review, but the review digest is runtime-derived, HEAD is self-derived, and the review is read twice. |
| `docs_1105s_exp35_e1.py` | `a70ac5e0` | **FAIL** | E1 equality itself verifies, but the training-rate/predictor paragraph is partly false and mostly absent from the only pinned source. The script can also duplicate its entry on rerun. |
| `docs_1105t_exp22_a7r5_ok.py` | `e5813bba` | **FAIL** | The APPROVE review and quote verify, but the digest is runtime-derived, HEAD is self-derived, and the review is read twice. |
| `exp35_phase_i_watch_v2.sh` | scratch | **FAIL** | Read-only throughout and much improved, but incomplete argument/schema validation and partial command identity leave P6 only partially resolved. |

## Ranked findings

### HIGH — The E1 measurement and predictor record is not source-sound

[docs_1105s_exp35_e1.py:39](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105s_exp35_e1.py:39)

The pinned snapshot records baseline step `27941` and the chunk-2 final guard at step `28439`, i.e. 498 steps—not 503: [snapshot:8](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_state_snapshot_2026-11-05_21-07.json:8), [snapshot:88](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_state_snapshot_2026-11-05_21-07.json:88). A read-only replay of the snapshot-referenced log prefix reached step `28440`—499 steps—and calculated mean `5.541`, still not 503 / 5.53. It did reproduce the quoted worst windows, 5.31 and 6.18, but that mutable log/report was never a declared full-SHA source.

The same paragraph says that “at ≈5.5 s/step the predictor allows shared spawns until ≈12:10.” H35’s actual predictor is fixed at the conservative `3.5` s/step, not the measured rate: [exp35_supervisor.py:44](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:44), [exp35_supervisor.py:73](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:73). Around this snapshot it implied a shared-spawn cutoff around 06:40–07:05 Nov 6, depending on the next chunk—not 12:10. Therefore “phase (i) runs shared almost throughout” is also overstated. At 5.5 s/step, the step/time arithmetic gives an endpoint around 14:45, slightly later than the stated 14:00–14:30.

The E1 facts themselves pass: both cells are `gates_verified`, both verdicts are `equal`, the digest/receipts match, the lane is shared GPU 1, the guard is OK, and no latch is present. H35’s E1 code confirms that `equal` binds the registered receipt, waveform, stream and metrics comparisons.

Concrete fix: append a correction that limits the snapshot claim to “final guard OK at step 28439, 498 beyond baseline.” Archive a complete guard/predictor JSON as an immutable full-SHA source before recording rates, windows or ETAs. Correct the predictor statement to its fixed 3.5-s/step rule and remove the “shared almost throughout” conclusion.

### HIGH — Watcher v2 still has notification holes

[exp35_phase_i_watch_v2.sh:8](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch_v2.sh:8), [exp35_phase_i_watch_v2.sh:15](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch_v2.sh:15), [exp35_phase_i_watch_v2.sh:25](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch_v2.sh:25)

- Missing operands expand at line 8 under `set -u` before `bad()` can issue the promised `EVENT`; argument-count validation is absent.
- The jq expression treats a missing `.stop_latch` as null and accepts `{"chunks":[]}` or an object-valued `chunks` as `OK 0`. These are malformed state documents but do not trigger an event.
- Process identity checks start time, cwd and an exact H35 argument, which closes PID reuse, but does not authenticate the complete argv or even require the supervisor script. A same-PID `exec` into another command retaining cwd/H35 would pass.
- `grep` exit 2 is treated like “no STOP LATCH”; log-read failures should be distinguished from a normal no-match.
- `/proc/<pid>/stat` field 22 is extracted with whitespace-based `awk`; a process `comm` containing spaces can cause a false event. That is fail-safe but avoidable.

The requested improvements that do pass are significant: numeric bounds, start time, cwd and H35 element; any main jq execution failure becomes an event; `.stop_latch` is read directly; the STOP line matches H35’s anchored event prefix; device:inode replacement is detected. Every reachable command is read-only: no signal, kill, process control, reservation/state mutation or persistent file write occurs.

Concrete fix: check `"$#"` before expanding operands; require an object with an explicit `stop_latch` key and an array of schema-valid chunks; compare the complete NUL-delimited expected argv including the absolute supervisor path and H35 position; parse stat after the closing `)`; and treat grep statuses other than 0/1 as an event.

### HIGH — Evidence digests are not independently pinned, and some claims have no declared source

[docs_1105m_exp22_a7.py:26](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105m_exp22_a7.py:26), [docs_1105q_exp22_a7r4.py:18](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105q_exp22_a7r4.py:18), [docs_1105r_exp22_a7r5.py:18](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105r_exp22_a7r5.py:18), [docs_1105t_exp22_a7r5_ok.py:17](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105t_exp22_a7r5_ok.py:17)

- `docs_1105m` hashes `issue_report.md` immediately before accepting that same digest.
- As disclosed, q/r/t received digests computed from the same extract at run time. Their `must_contain` checks protect decisive review conclusions, and the copied current files are exact, but this is content validation—not an independent SHA pin.
- `docs_1105m` does not source-bind the P1 log message, source-tree searches, history/diff claims or ETA.
- `docs_1105n` does not bind its 20:32 live-process check, watcher smoke tests, v1 stop, or v2 re-arm.
- `docs_1105s` pins only the state snapshot; it does not pin a guard report, training log prefix, predictor result or ETA calculation.

Concrete fix: embed an independently established literal 64-hex digest in the script. Archive and pin exact guard/process/smoke-test evidence before quoting it. Derive values from the bytes returned by `require_text`/`require_source`.

### HIGH — Every Git HEAD check is tautological

Examples: [docs_1105m_exp22_a7.py:28](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105m_exp22_a7.py:28), [docs_1105s_exp35_e1.py:29](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105s_exp35_e1.py:29)

All eight scripts run `git rev-parse HEAD` and immediately pass its result as the expected HEAD. This detects a race between declaration and commit, but cannot refuse an already-wrong starting commit. The manifests show that the actual heads happened to be the correct parents, so no committed result is wrong.

Concrete fix: hard-code the expected full parent SHA in each script—`fc445e1d…`, `4b2c718e…`, etc.—or pass it from a separately authenticated record, never from the checkout being validated.

### HIGH — `docs_1105o` committed a destination that bypassed the transaction

[docs_1105o_exp22_a7r2.py:17](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105o_exp22_a7r2.py:17), [docs_1105o_exp22_a7r2.py:98](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105o_exp22_a7r2.py:98)

The direct `cp` is disclosed and the script verifies the resulting review bytes as SHA `1c339065…`; they equal the source extract byte-for-byte. Nevertheless, `c6d34d51` contains four changed paths while its manifest contains only three operations. The review destination had no transactional creation/recovery record.

Concrete fix: require the scratch extract and call `txn.create(REV, verified_text)` or `copy_verified` inside the same transaction. Never pre-create a destination that the subsequent Git commit will include.

### HIGH — `docs_1105s` can duplicate its milestone on rerun

[docs_1105s_exp35_e1.py:32](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105s_exp35_e1.py:32), [docs_1105s_exp35_e1.py:45](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105s_exp35_e1.py:45)

There is no “E1 entry absent” check, and the HANDOFF replacement preserves the original anchor at the start of its replacement. The same script can append and insert the milestone again in a later transaction.

Concrete fix: require the milestone heading/source SHA to be absent and use a consumed anchor, or pin exact destination hashes before appending.

### NIT — Review sources are verified twice

`docs_1105n`, p, q, r and t first call `require_text`, then `copy_verified`, which rereads and re-registers the same source. The manifests consequently contain duplicate source entries. This is still hash-safe, but contradicts n’s “every source verified once” policy.

Concrete fix: retain the returned verified text and call `txn.create(REV_DST, text)`.

### NIT — Typed times/zones are not derived as required

The actual transaction times prove that `20:32`, `20:33` and `21:06` were not future events when committed. The future times in m/s are visibly forecasts. Nevertheless, P5 requires derived zone labels and source-derived historical times.

Potentially future typed values were:

- m: Nov 6 13:00–16:00, explicitly an ETA/not-before forecast.
- n: 20:32 and 20:33, actually before its 20:34:17 transaction but not source-bound or runtime-checked.
- s: 21:06, actually before its 21:07:47 transaction; Nov 6 12:10, 14:00–14:30 and 22:15 are forecasts. The 12:10 forecast is substantively wrong as described above.

Concrete fix: render current times only from `txn.iso`/`txn.hm`; extract historical ISO times from verified source objects; calculate forecasts using aware datetimes rather than typed `EST`.

## Truthfulness audit of the A7 chain

The review copies, verdicts, quotations and summaries are byte-faithful. All eleven registered node IDs exactly equal the eleven unique IDs in the pinned triage log. The four H35 hashes match.

The following proposal claims were later disproved and superseded; they should be understood as failed review submissions, not governing facts:

- A7 r1 in m proposed wrapping `-c`/`-m`, a fixed `target=localize_meshgrid.py` sentinel for pytest, and unchanged `cuda:1` behind one visible card; its review correctly rejected all three and the missing log-to-artifact binding.
- A7 r2 in o said it “answers” the earlier BLOCKING/HIGH findings, but its first-line sentinel rejected valid launches, it lacked sealed 30-minute evidence, and its T/wake/state/orchestration rules remained incomplete.
- A7 r3 in p required JUnit to establish zero XPASS, which pytest 9.1.1’s JUnit writer cannot do.
- A7 r4 in q said every XPASS becomes failed under `xfail_strict=true`; explicit `strict=False` overrides that default. Its separator-form regex was incompatible with inherited `-q`, so “the Coder round starts now per Codex’s condition” was premature.
- r corrected those points, and t accurately records the final r5 APPROVE with no findings.

Future amendments should say “intended to answer” or “submitted for review,” not “resolved” or “per Codex’s condition,” until the confirming review exists.

## Manifests and Git logs

Q4 otherwise passes.

- Exactly eight durable `done/*.json` manifests exist, corresponding in order to the eight commits.
- Every manifest old/new destination SHA matches the parent/child Git blob.
- Seven commit path sets equal their manifest operation sets exactly.
- `c6d34d51` has the one disclosed extra path: the directly copied A7 review.
- No recovery manifest exists.
- No done manifest exists for the refused first `docs_1105o` attempt.
- Every commit carries the required `Co-Authored-By: Claude Opus 5.5` trailer, is in the stated linear chain, is contained by `origin/localization-exp`, and the checkout/upstream are both `e5813bba…`.
- All eight Git logs show commit, successful pull/rebase, push and final commit. The first log also records the disclosed pre-commit pull refusal because helper-produced edits were unstaged; its later pull succeeded: [git_records…exp22_a7.log:1](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/git_records_2026-11-05_20-31-23_exp22_a7.log:1).

## Requirements for the next script

1. Append the E1 correction described above, backed by a full-SHA archived guard/predictor report.
2. Use literal expected HEAD and literal independently obtained source digests.
3. Use one verified read plus `txn.create`; do not reread review sources.
4. Put every committed destination—including review copies—inside the transaction.
5. Add an explicit rerun guard for append-only milestones.
6. Derive event times/zones from verified source data or `txn.iso`/`txn.hm`.
7. Revise watcher v2’s argc, state-schema and exact-argv checks before relying on it as the sole wake-up. It remains read-only and is not a supervisor safety gate.
8. Pull/rebase before running the transaction; then commit and push afterward, teeing all results.

## What I verified

- Read all eight scripts, watcher v2, the complete P5/P6 reviews, the installed helper, all relevant A7/P6/INT-L2 reviews, the triage log, snapshot and H35 implementations.
- Verified all stated script/helper/watcher hashes and line counts; the installed helper is SHA-256 `93eee231…`.
- Parsed all Python scripts with standard-library AST and watcher v2 with `bash -n`; executed none of them.
- Reconciled all manifest sources, Git states and operations against commit parents/children.
- Verified the eight-commit ancestry, messages, trailers, changed paths, pushes and current upstream.
- Independently checked the snapshot arithmetic, H35 E1 semantics, guard thresholds and predictor formula.
- Used only read-only probes: `rg`, `sed`, `nl`, `find`, `sha256sum`, `cmp`, `stat`, `jq`, `wc`, standard-library `python -B -c`, `bash -n`, and read-only Git commands.
- Did not run any reviewed script/helper/watcher, pytest or GPU command; did not install or modify anything; and did not inspect, signal, attach to, or control any process.
