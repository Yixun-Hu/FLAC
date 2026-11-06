**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** Planner records scripts P6 (docs_1105i … docs_1105l + the phase-(i) watcher; 74a7d39a → fc445e1d)

**Verdict:** REQUEST-CHANGES

No BLOCKING defect was found, and the committed launch evidence is overwhelmingly accurate. However, there are three record-content errors/overstatements, incomplete SHA-256 source binding, a PID-reuse hole in the frozen-entry script, and several ways for the watcher to miss its wake-up indefinitely. The already accepted lack of a cross-file transaction/recovery manifest is not counted again.

## Per-script status

| Script | Commit | Status | Assessment |
|---|---:|---:|---|
| `docs_1105i_p2.py` | `74a7d39a` | **FAIL** | P2 summary is faithful, but two mtimes do not establish when the Agent call occurred. The correction calls that interval “source-bound” when it is session-sourced. |
| `docs_1105j_reviews.py` | `86dd8c2b` | **FAIL** | Recorded facts check out, and write mechanics meet the hand-written pattern, but significant Coder/git/review-launch claims are not covered by its SHA-256 source manifest. |
| `docs_1105k_int.py` | `20fbc236` | **FAIL** | Reviews and copied ladder evidence verify. The description of the `6f6fbd88` DINOv3 prerequisite overstates the exact-set enforcement later rejected by INT-L2. Some test/launch facts are also unbound. |
| `docs_1105l_launch.py` | `fc445e1d` | **FAIL** | Launch hashes, tuple, command, environment, PID snapshot, residuals, predictor arithmetic and E5 registration largely verify. The ETA was falsely described as already “re-estimated after chunks 3–4,” and the live PID check is not reuse-safe. |
| `exp35_phase_i_watch.sh` | scratch, running read-only watcher | **FAIL** | It never signals, kills, or changes a file, but malformed/unreadable state, PID reuse, log rotation, and invalid arguments can defeat its sole notification. |

## Ranked findings

### HIGH — The correction still does not establish the Agent commissioning time

[docs_1105i_p2.py:28](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105i_p2.py:28), [docs_1105i_p2.py:45](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105i_p2.py:45), [docs_1105i_p2.py:62](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105i_p2.py:62)

The P1 review mtime is `18:21:59` and the prior script/entry/commit time is consistently `18:22:54`. Those facts do not prove the comment “the Agent call came after the first and before the second.” The exact ≈18:22 commissioning time is plausible session history, but it is not bound by either file’s content or mtime.

Concrete fix: append a correction saying only that the preserved review and entry times bound those artifacts; label the commissioning time as unpreserved session evidence unless an actual Agent-launch event/transcript with an embedded timestamp is hash-pinned.

### HIGH — `docs_1105k` overstates the DINOv3 enforcement in `6f6fbd88`

[docs_1105k_int.py:74](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105k_int.py:74)

The two expected DINO files were SHA-pinned, but `6f6fbd88` used plain `ls` for its set comparison and therefore ignored hidden snapshots/files. INT-L2 classified this exact gap as its remaining HIGH. The statement that the DINOv3 snapshot under the repository’s `assert_vit_pin` contract was a hard check is therefore overstated. Exact hidden-aware enforcement first appears in `2143fecc`, reviewed SHIP by INT-L3.

Concrete fix: append a correction distinguishing the partial `6f6fbd88` check from the exact hidden-aware `2143fecc` check.

### HIGH — The frozen launch entry claims a future re-estimation had already happened

[docs_1105l_launch.py:75](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105l_launch.py:75)

At the entry/commit time, 20:23:41 EST, chunk 1 was in flight and chunks 3–4 had not run. Therefore “re-estimated after chunks 3–4” was false at record time. The ETA itself is a reasonable launch forecast; only that parenthetical is wrong.

Concrete fix: append a correction changing it to “to be re-estimated after chunks 3–4.”

### HIGH — The scripts do not bind all claimed evidence to verified bytes

Examples:

- [docs_1105i_p2.py:36](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105i_p2.py:36) hashes the review but rereads it at line 39; its mtimes are not content-bound.
- [docs_1105j_reviews.py:14](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105j_reviews.py:14) lists only P3 and three ladder logs, while lines 66–69 also rely on the Coder log, commits, launch script/transcripts and helper test evidence.
- [docs_1105k_int.py:14](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105k_int.py:14) omits the scratch-test evidence and INT-L2 launch transcript.
- [docs_1105l_launch.py:15](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105l_launch.py:15) omits C3e, INT, INT-L2, the Coder log, plan, component files and predictor/ETA source material; it also rereads the verified review and launch log.

I independently verified most of these facts, so this is a mechanics failure rather than evidence that the committed statements are generally false.

Concrete fix: use one complete source manifest and derive every claim from the exact bytes returned by the successful SHA-256 verification—never reread a source pathname afterward.

### HIGH — The frozen-entry liveness check can accept a reused PID

[docs_1105l_launch.py:44](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1105l_launch.py:44)

The script accepts any `/proc/685273/cmdline` containing `exp35_supervisor.py` and the H35 argument. It does not bind the live process to start time `781579512`, cwd, session ID, exact argv, or the scrubbed environment recorded in the launch log. A reused PID running another exp35 supervisor could therefore be recorded as the launched process.

Concrete fix: compare `/proc/<pid>/stat` field 22, cwd, complete NUL-delimited argv, session ID and expected environment against the hash-bound launch record.

### HIGH — Watcher errors can silently turn a milestone wake-up into an indefinite wait

[exp35_phase_i_watch.sh:6](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch.sh:6), [exp35_phase_i_watch.sh:7](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch.sh:7), [exp35_phase_i_watch.sh:11](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch.sh:11)

- A missing, malformed, unreadable, or schema-invalid state makes `jq` fail; stderr is discarded, `n` becomes empty, and polling continues silently.
- A reused PID running any command containing `exp35_supervisor.py` is treated as the original supervisor because start time/H35/cwd are not checked.
- An invalid or zero `POLL` can produce a tight loop; nonnumeric `N` can never satisfy the threshold.
- If the log is renamed and replaced, the supervisor continues writing the old inode while the watcher greps the new pathname. It may miss the latch until process exit, or forever if an in-flight child never exits.

Concrete fix: validate all numeric arguments; bind PID start time, exact H35/cmdline and cwd; make any `jq -e` failure an explicit event; inspect `.stop_latch` directly; and bind the log’s device/inode or emit an error if it changes.

### NIT — The STOP LATCH match is broader than its current log contract

[exp35_phase_i_watch.sh:10](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/exp35_phase_i_watch.sh:10)

Current supervisor code prints uppercase `STOP LATCH` only for the actual latch event, so I found no present false match. Nevertheless, the unanchored substring could match future diagnostic text or a stale log.

Concrete fix: match the exact event format, e.g. an anchored timestamp followed by `STOP LATCH`, and confirm `.stop_latch != null` in the state.

## Truthfulness results

Aside from the three content defects above, the factual spot-check passed:

- All four scripts map to the stated commits, all are ancestors of `fc445e1d`, all four commits carry the required Opus 5.5 trailer, and origin reflogs show each push.
- C3d fix commits, 3/3 trailers, code-line counts, component hashes, 38-commit range, launch-script hashes, P3/P4 findings and review launch times verify.
- All eight `5b065801` ladder/dry-run copies match the run-worktree sources byte-for-byte and by SHA-256. Rung 4 reports 1513 passed, 1 skipped, 0 failed.
- The capture/launch evidence hashes all match. The logs contain exactly 62 and 69 `CHECK OK` lines. The tuple copies are byte-identical, 581 bytes, SHA-256 `238d890d…`.
- The launch log records PID 685273, start time 781579512, cwd, SID 685273, exact argv and the seven-variable scrubbed environment. The state/reservation evidence confirms chunk 1, GPU 1, epoch 1 and the 20:20:24 start.
- All component hashes, 16 external pins, versions, campaign digest, 346/116 row claims, tuple identity, residual list and four supervisor rules verify.
- Predictor arithmetic checks: at 27,930 steps and an actual ≈4.3 s/step rate, the BF K8 cutoff is about 12.37 hours after capture (≈08:41) and BF K1 about 12.82 hours (≈09:08); projected training completion is ≈10:44. Planned work is 1001.8 minutes, so “≈16.5 h” and the broad 12:30–15:00 completion forecast are reasonable launch estimates.
- `analysis/yaw_robustness/exp35_e5_offorbit_pairs.py` is absent and outside the runner’s current `RELEVANT` pathspec, so the entry genuinely pre-registers a future path rather than claiming an existing implementation.

At `2026-11-05T20:27:25-05:00`, `/proc/685273` was absent in this review environment. I therefore could not live-confirm its current cmdline/cwd/stat. That does not falsify the hash-bound 20:20–20:23 launch snapshot, but “running now” was no longer live-verifiable. I did not inspect, signal, attach to, or otherwise interact with another process.

## Watcher assessment

The watcher is read-only with respect to the filesystem and supervised processes in every code path. It only reads `/proc`, the log and state, sleeps, and writes its notification to stdout. There is no signal, kill, redirection to a file, reservation mutation or state mutation.

Its cumulative `gates_verified` count is sound under a valid atomic state: completed chunks remain `gates_verified`, so normal 60-second polling cannot miss a milestone.

The one-shot design is adequate as a non-safety wake-up if the shell/orchestrator reliably surfaces background completion and the watcher is re-armed after each milestone. The supervisor itself enforces every stop, so a lost watcher notification does not permit another spawn. It is not a durable alert mechanism, and the error-handling defects above should be fixed before relying on it operationally.

## Requirements for the next records script

The next script should:

1. Copy the exact P5-approved helper, SHA-256 `93eee2310e3ce21f052085c89c39e46c62d38a8bf0a81e3eab62d46f0f04e839`, to a stable controlled location first. It currently exists only in scratch, not in the repository.
2. Use one transaction and the shared `~/.local/state/flac_records_txn` state directory for every edit, append and copy; perform no direct destination writes.
3. Use `expect_date()` and `txn.iso`/`txn.hm`, with explicit branch, full HEAD, upstream and narrowly scoped clean-path checks.
4. Verify each source once by full SHA-256 and derive all parsed facts from those returned bytes. Do not infer event times from mtimes.
5. Append the three corrections identified above: unpreserved commissioning time, the incomplete `6f6fbd88` DINO set check, and “to be re-estimated after chunks 3–4.”
6. For live-process statements, bind full process identity—including start time—not a PID/name substring.
7. Stop on every `RecoveryNeeded`, inspect helper status, and resolve it before any later record transaction.
8. Keep pull/rebase, commit—with the Opus trailer—and push outside the helper, with an auditable command/result log.

## What I verified

Read-only probes used: `sha256sum`, `wc`, `nl`, `sed`, `rg`, `cmp`, `stat`, `jq`, `find`, standard-library Python AST/arithmetic and `/proc` reads, plus `git show/log/diff/status/reflog/rev-list/merge-base/rev-parse/for-each-ref`.

I did not execute any reviewed script, watcher, helper, test, pytest, package/environment command, or GPU command; did not install or modify anything; and did not touch, signal, attach to, or control a process.
