**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_36 code review II (round 2 of 3; 461c58e0 → 5ee1d6fe + the Planner's exp_36 records scripts)

**Verdict:** REQUEST-CHANGES

Review-I HIGH 1 is functionally resolved. Chain I and Chain II evaluation logic remain sound, but publication must not run until the HIGH publication findings below are fixed.

## BLOCKING

None.

## HIGH

1. **Publication is not serialized, and its destination-directory check is vulnerable to concurrent interference.**

   The publisher checks the destination once, scans all shared temp names, and—during resume—deletes every matching temp without proving that no active publisher owns it. Two same-set invocations can both pass preflight; one can later overwrite the other’s manifest state, while a resume can unlink the active invocation’s temp. A concurrent rename/symlink replacement after the one-time path check could also redirect path-based writes. [depth_canon_collect.py:2557](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2557), [depth_canon_collect.py:2579](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2579), [depth_canon_collect.py:2592](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2592), [depth_canon_collect.py:2606](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2606)

   Fix:

   - Hold one nonblocking global exp_36 publication lock from before preflight through the final manifest and directory fsync.
   - Open the pinned destination once with `O_DIRECTORY|O_NOFOLLOW`, verify its inode/path, and perform temp creation, linking, hashing, and unlinking relative to that directory fd.
   - Give every publication a transaction ID; resume may remove only temps named in its validated prior manifest.
   - Test two concurrent same-set invocations, concurrent A/OFF invocations, and a destination path swap. All but the lock owner must refuse without changing destinations or manifest state.

2. **`--exp36-publish-resume` can overwrite an unrelated manifest path and does not authenticate the recovery record.**

   `resume=True` bypasses the only `lexists(out)` refusal. The code never parses or validates the existing manifest before replacing it. `_write_json_atomic()` opens a predictable temp with truncating `"w"` and then uses `os.replace`; therefore a custom `--publish-manifest-out` plus `--exp36-publish-resume` can replace any existing file, while a pre-existing temp symlink can cause an unrelated target to be truncated. The initial absent-path check also has a check/replace race. [depth_canon_collect.py:2515](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2515), [depth_canon_collect.py:2559](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2559), [depth_canon_collect.py:2600](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2600), [depth_canon_collect.py:2712](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2712)

   Fix:

   - Create the initial manifest with exclusive, no-follow, no-replace semantics and fsync both file and directory.
   - On resume, require a regular non-symlink manifest whose `check`, set, pin, digest, base/destination, verdict hash, complete entry set, source hashes, and status (`in_progress` or `partial`) exactly match the recomputed publication.
   - Use an exclusive no-follow temporary file for updates and only replace the authenticated manifest.
   - Prefer restricting production output to the exact set-specific default path.
   - Add refusal tests for resume with no manifest, an unrelated JSON/file, a symlink, the other set’s manifest, a changed verdict/source list, and a pre-existing writer temp.

3. **The recorded publisher HEAD does not bind the code that actually performed publication.**

   The manifest records `{"repo": REPO, "head": git_head(REPO)}` but does not require a valid expected publisher HEAD or a clean publisher implementation. A locally modified collector or dependency can publish while the manifest still names the unchanged Git HEAD. [depth_canon_collect.py:2598](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2598)

   Fix: require a full reviewed publisher pin—preferably `--publisher-pin`—and verify the tracked publication/replay-verification closure matches that commit. Record the collector path/hash or relevant-tree digest beside the publisher HEAD. Test dirty collector/dependency bytes, missing Git HEAD, and wrong publisher pin. Keep the artifact provenance separately fixed at `461c58e0` plus its campaign digest, as it is now.

4. **Planner records assert operational actions that had not occurred or are unsupported by their sources.**

   - The 14:44 scaffold says the latency data were “reported to him with B’s plan,” but that plan was not committed until 15:22 and the transaction records no external report. [docs_1106zf_ab_scaffold.py:63](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106zf_ab_scaffold.py:63)
   - The 16:16 notebook says Chain I “started detached” and its PID/log were recorded, while the same entry says the handshake was held, `A has NOT paused`, and the chain would start only after `PAUSED`. No PID or launch log is named. [docs_1106zj_cap.py:71](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106zj_cap.py:71), [yaw_c4_spread_worklog.md:11](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_worklog.md:11)

   Fix: append a transactionally written correction to the append-only notebook stating that Chain I had **not** launched and that detached launch/PID/log recording was a future instruction. Correct the issue-report sentence to “to be reported” or cite the later dated plan/report evidence. Do not silently rewrite the historical notebook entry.

## NIT

1. **The parsed CLI’s default per-set manifest routing is not actually tested.**

   `_cli()` always supplies `--publish-manifest-out`; the separate default-path test only formats the constant directly. Thus line 2727’s parsed-CLI default selection is uncovered. [test_exp36_publish.py:175](/home/yixunhu/codespace/exp-36-yaw-c4-spread/src/tests/test_exp36_publish.py:175), [test_exp36_publish_provenance.py:25](/home/yixunhu/codespace/exp-36-yaw-c4-spread/src/tests/test_exp36_publish_provenance.py:25)

   Fix: omit the override, redirect `EXP36_PUBLICATION` to a temporary format pattern, invoke the parsed CLI for A and OFF, and assert two distinct default files containing 48 and 12 entries.

2. **The cross-version guard does not reproduce the precise relative-plan cross-checkout scenario.**

   The old-versus-new comparison passes an absolute plan path, while the relative-path test exercises only the new collector. The production case is old collector + relative path while cwd is the run checkout, followed by new collector + the same relative string from another cwd. [test_exp36_publish.py:257](/home/yixunhu/codespace/exp-36-yaw-c4-spread/src/tests/test_exp36_publish.py:257), [test_exp36_publish.py:266](/home/yixunhu/codespace/exp-36-yaw-c4-spread/src/tests/test_exp36_publish.py:266)

   Fix: combine those conditions in one test and describe the equality as parsed-dict equality, not “byte-for-byte JSON” equality.

3. **CLI documentation still describes only the 24-cell A publication.**

   `--exp36-publish` says it copies 24 cells even though OFF copies six; the publication test module’s header likewise documents only A. [depth_canon_collect.py:2705](/home/yixunhu/codespace/exp-36-yaw-c4-spread/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:2705), [test_exp36_publish.py:4](/home/yixunhu/codespace/exp-36-yaw-c4-spread/src/tests/test_exp36_publish.py:4)

   Fix: describe the selected A/OFF set and its 48/12 files.

4. **The holding-status text is labelled “verbatim” after whitespace normalization.**

   `' '.join(ast.split())` changes the source’s whitespace and line structure. [docs_1106zj_cap.py:77](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/docs_1106zj_cap.py:77)

   Fix: preserve it in a fenced block byte-for-byte or label it “whitespace-normalized.”

## Answers to P1–P6

- **P1:** Yes, review-I HIGH 1 is resolved in behavior: OFF is exactly 6 cells/12 files, A is 24/48, YAW verdict only, distinct default manifest names, and the parsed OFF CLI is exercised. NIT 1 is the remaining test gap.
- **P2:** Normal final artifact creation is no-replace, fsynced before linking, rehashed, excludes replay/receipts, and is pin/digest/verdict bound. Publication is nevertheless not ready because concurrency, manifest ownership, and publisher-code provenance are not fail-closed.
- **P3:** Sound. The relative old plan resolves under `~/codespace/exp-36-run`; the re-check produces the same semantic verdict while artifact provenance remains `461c58e0`. The guard should be strengthened as in NIT 2.
- **P4:** No additional Chain-II correctness defect found. It selects one YAW replay plus exactly 24 rows; a prior replay is skipped only after full verification at the same HEAD/digest and is then compared again. Both dry-run phases report tree digest `49daa868…`.
- **P5:** The transaction mechanics are sound apart from the record claims in HIGH 4/NIT 4. The reviewed helper hash is exact; locking, CAS checks, exclusive staging, recovery marking, fsyncs, and completed manifests are present.
- **P6:** No other experiment-result or row-selection defect found.

## What I verified

- The exact seven-commit range `461c58e0..5ee1d6fe`, including code, tests, command file, ledger, notebook, and every supplied log.
- Run worktree detached at exact `461c58e0`; publisher checkout clean at `5ee1d6fe`; exp-25 destination checkout at exact `5b7e663b`.
- The destination and every parent are plain directories. All 48 A and 12 OFF destination paths and both default publication manifests are currently absent.
- All YAW/P1/BF/CANON archived metrics, stream, receipt, evaluator-log and CANON-sidecar hashes match the committed constants; the CANON checkpoint manifest hashes to `1e481447…`.
- Logged evidence: H36-I suite `1645 passed, 1 CUDA skip`; dev suites `635 + 242 passed`; cycle 11 `65 passed`; both DRY_RUN transcripts and all four comparator reference checks.
- All four Planner-script hashes and line counts match the supplied values. Their completed transaction manifests contain exactly 6, 3, 7, and 5 operations, and every old/new hash matches commits `241d73b1`, `de309efc`, `ce2412c5`, and `058add18`. No unfinished records transaction exists.
- I did not run tests, scripts, collectors, publishers, evaluators, runners, package operations, GPU commands, or process-control commands; I modified no file and did not access either prohibited coordination directory.
