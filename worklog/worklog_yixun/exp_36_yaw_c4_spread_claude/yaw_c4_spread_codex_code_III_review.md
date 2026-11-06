**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_36 code review III (round 3 of 3, the last; 5ee1d6fe → d373c18e, launch commit H36_I2 d92aef10 + the Planner's records scripts)

**Verdict:** REQUEST-CHANGES

## Must be fixed BEFORE Chain I launches

1. **HIGH 1 — authenticate the local receipt before treating an existing dump as resumable.**
2. **HIGH 2 — hard-retire the superseded exp-35 phase-(ii) mutating modes.**

HIGH 1 changes launch code, so `d92aef10` cannot remain the launch commit. Record a replacement launch SHA, detach `exp-36-run` there, and repeat both dry runs. Fold the launch-code NIT fixes below into that same replacement commit. No round 4 is requested.

## BLOCKING

None.

## HIGH

### HIGH 1 — dump preflight accepts any receipt pathname as proof of ownership

**Files:** [depth_canon_collect.py:2439](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2439), [test_exp36_dump_preflight.py:54](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_dump_preflight.py:54), [depth_canon_eval_run.sh:331](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:331)

The preflight allows an existing dump or staging sibling whenever `os.path.lexists(base_dir/p.receipt)` is true. It does not open or authenticate that receipt. A dangling symlink, unrelated JSON, stale receipt, wrong pin, wrong cell, or wrong dump root therefore bypasses the collision stop. The unchanged runner can subsequently quarantine—rename—the existing dump.

**Fix:** make preflight accept an existing dump only when the receipt is regular, non-symlink, parseable and authenticated for the selected cell, current full launch pin, farm root, dump root and `dump: true`. Receipt-evidence verification should still permit recovery from a damaged/incomplete dump owned by that authenticated receipt. Pass `EXPECT_HEAD` into `--exp36-dump-preflight`.

Add tests for:

- dangling and ordinary receipt symlinks;
- unrelated/malformed JSON;
- wrong cell, pin, farm root and dump root;
- a valid current receipt with a complete dump;
- a valid current receipt whose dump needs the runner’s quarantine/recovery path.

**Pre-Chain-I:** Yes.

### HIGH 2 — the superseded exp-35 launcher can still rename the 12 shared H35 dumps

**Files:** [exp35_phase_ii.sh:33](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_phase_ii.sh:33), [depth_canon_eval_run.sh:331](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:331)

The exp-36 chain lock is local to exp-36 and is not honored by the superseded exp-35 phase-(ii) kit. Its `reserve`, `dry-run` or `launch` path can reach the generic runner and quarantine the 12 identically named H35 OFF dumps. A shared lock would prevent concurrency but not a later accidental exp-35 launch.

**Fix:** add a committed retirement gate before log creation/helper invocation that refuses `reserve`, `dry-run` and `launch`, while retaining `status` and `release` for inspection/cleanup. Update the recorded kit instructions and test that all three mutating modes refuse before orchestration, runner or GPU interaction; test that `status`/`release` remain usable.

**Pre-Chain-I:** Yes.

### HIGH 3 — an internally failed publication-manifest update can make resume permanently refuse

**Files:** [depth_canon_collect.py:2600](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2600), [depth_canon_collect.py:2636](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2636), [depth_canon_collect.py:2719](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2719)

`_Manifest.write()` removes its writer temp when manifest identity changes, but not when `os.replace()` itself fails. `_write_new()` likewise can leave a partially written newly-created file if write/fsync fails. Resume deliberately refuses whenever any manifest writer temp remains, so an ordinary internal I/O failure can strand the publication.

**Fix:** unlink a file newly created by `_write_new()` when its write/flush/fsync fails. Wrap update-temp handling in `try/finally`, removing that transaction’s temp whenever replacement did not consume it. Preserve refusal of genuinely foreign temps. Add injected write/fsync and `os.replace` failure tests proving cleanup occurs and retry/resume succeeds.

**Pre-Chain-I:** No, but it must be fixed before either publication.

### HIGH 4 — the Planner’s 17:36 record states future actions as already completed

**File:** [docs_1106zm_exp37_r1.py:53](</tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106zm_exp37_r1.py:53>)

At 17:36 the script recorded that H36_I2 existed, the run worktree had moved to it, and dry runs had been repeated. Git records `d92aef10` at 18:07:32; the move and dry runs followed at 18:07–18:14. This is the same records-integrity class as the prior false completed-action finding.

**Fix:** append—do not rewrite history—a correction saying those actions were planned/in progress at 17:36 and completed at their actual times. Make the corrective transaction assert its exact anchor/new text and retain its done manifest.

**Pre-Chain-I:** No operational dependency, although the correction should be committed promptly.

## NIT

### NIT 1 — the chain prints a publication command that always lacks required provenance arguments

**Files:** [yaw_c4_spread_chain.sh:159](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:159), [test_exp36_chain.py:191](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_chain.py:191), [yaw_c4_spread_command.md:20](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_command.md:20)

The printed command omits `--publisher-pin` and `--base-dir` and invokes the collector in the run checkout. It fails closed, but it is not an executable publication handoff. The command document has the correct publisher checkout and arguments.

**Fix:** preferably print only a pointer to the reviewed command document, since the chain cannot know the later publisher SHA. Alternatively supply every required argument from an explicitly pinned publisher checkout. Strengthen the chain test to reject output lacking `--publisher-pin`, `--base-dir` and the publisher-worktree context.

Fold this into the replacement launch commit.

### NIT 2 — multiple descriptions still say exp-36 replays are dump-free

**Files:** [depth_canon_collect.py:2462](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2462), [test_exp36_replay_check.py:4](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_replay_check.py:4), [test_exp36_a1_replays.py:66](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_a1_replays.py:66), [test_exp36_publish.py:5](/home/yixunhu/codespace/exp-36-run/src/tests/test_exp36_publish.py:5)

The implementation correctly requires and fully checks replay dumps, but these docstrings still describe the replay or publication sources as dump-free.

**Fix:** replace those statements with the A1 behavior: every replay dumps; comparison performs full waveform/marker checks; publication verifies dump evidence offline but never copies a dump. Fold the collector correction into the replacement launch commit.

### NIT 3 — the final worklog entry is future-stamped relative to its commit

**File:** [yaw_c4_spread_worklog.md:97](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_worklog.md:97)

The entry is headed `18:20`, while commit `d373c18e` containing it was authored and committed at `18:16:37`.

**Fix:** append a small chronology correction identifying `18:16:37` as the record commit time; do not silently rewrite the append-only record.

## What I verified

- **R1 — Chain I:** The normal path remains fail-closed on HEAD, chain cleanliness, manifest hash, replay runner status, exactly one captured plan, all four comparator results, and production-run status. The ordered four replays and 24 production rows are unchanged from the prior launch plans after normalizing the new dump argument and expected campaign digest. Every selected row now dumps to `/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms`. Current selected target dump/staging paths showed no collision. Resume safety is not sufficient until HIGH 1 is fixed.

- **R2 — Chain II:** It retains one YAW replay plus the same 24 A rows. A prior complete YAW cell can be reused only through the runner’s full verification/skip predicate; the new plan still includes it, and the comparator re-verifies it with its dump before any A production cell starts.

- **Dump comparison:** All four replay families require dumps and compare waveform SHA-256 and `COMPLETE` marker against pinned values, plus `n`, dtype, shape and `input_hash` against the archived reference’s dump manifest. The four archived reference receipt/manifest pins matched the reviewed files.

- **Write scope:** There are no durable writes to the exp-25 publication worktree during either chain. Durable outputs are inside the detached run checkout and NAS dump root. The chain also creates and removes a transient default-`TMPDIR` directory via `mktemp -d` at [yaw_c4_spread_chain.sh:103](/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh:103), so “nothing written outside” is not literally true if transient `/tmp` scratch is included.

- **R3 — publication:** Round-II HIGH 1 and HIGH 3 are closed: one global exp-25-root flock, pinned `O_DIRECTORY|O_NOFOLLOW` destination fd, fd-relative operations, transaction-specific names, required publisher pin, clean publisher closure and recorded provenance. HIGH 2’s authentication/resume design is present, but HIGH 3 above identifies a new internal-failure cleanup hole. Sources are verified with dump evidence; publication copies only metrics and stream sidecars, never receipts, replays or dumps.

- **R4:** `yaw_c4_spread_launch_commit.txt` contains exactly full SHA `d92aef106de09496715b26ecc42f924ead104090` plus its label. The cross-version test loads the collector from that commit, uses the relative plan in the run checkout, recomputes from another cwd with the publisher collector, and requires parsed-dict equality. The logic is correct. The supplied H36_I2 test transcript predates the launch-commit record and reports the guard skipped; I therefore do not count that transcript as evidence of the enabled test path.

- **R5:** The three Planner scripts match the stated SHA-256 values and line counts. Their completed transaction manifests exist and their recorded new hashes match the corresponding commit blobs. No unfinished transaction directory was present. `docs_1106zk_review2.py` and `docs_1106zl_fixtime.py` implement their corrections; `docs_1106zm_exp37_r1.py` has HIGH 4.

- **R6:** The Coder’s H35-name collision risk is real; HIGH 2 is the minimal durable safeguard I recommend.

- **R7 / change isolation:** Between `461c58e0` and `d92aef10`, the production runtime changes among the named campaign files are confined to `yaw_c4_spread_chain.sh` and `depth_canon_collect.py`. The runner, generator, cells manifest, no-flash overlay, evaluator and `src/` are byte-identical. Existing collector paths for `--plan`, `--verify-cell`, `--write-receipt`, `--check-backend-log` and `--count-no-flash` are unchanged; only the new dispatch paths and associated exp-36/publication logic were added.

This was a read-only static review. I did not run tests, scripts, chains, evaluators, GPU commands or process probes, and modified no files.
