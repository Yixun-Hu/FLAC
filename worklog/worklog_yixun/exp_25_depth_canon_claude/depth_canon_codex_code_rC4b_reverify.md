**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-26 · **Round:** code rC4b (re-verification of the E1 guard-fixture fix after rC4 SHIP)

**Verdict:** REQUEST-CHANGES

### Findings

- **BLOCKING:** None.

- **HIGH — Q2 retains the same non-hermetic evidence assumption.** The production-path section creates another fresh clone at [depth_canon_guardtests.sh:785](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:785), but Q2 invokes FULL without clearing committed evidence at [lines 795–796](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:795). Once the eventual production PASS evidence and its log are committed, that evidence remains valid across record-only commits because HEAD is provenance-only. Q2 can then:

  - return 0 if the live FULL namespace is empty; or
  - return 2 for an unrelated NAS/namespace reason if it is nonempty.

  The current assertion does not prove an evidence refusal: `harness=false` always appears in the launcher header at [depth_canon_launch.sh:152](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:152), so the second outcome falsely passes Q2. Add `clear_evidence` immediately before Q2 and assert the FULL-requires-evidence headline plus the no-candidate diagnostic. This is the only other guard case I found that committed records can invalidate or make state-dependent.

- **NIT — comment precision.** The new comment at [depth_canon_guardtests.sh:482](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:482) says “every other FULL case” clears evidence and “this HEAD.” Q2 currently does not clear, and the evidence predicate deliberately does not compare the evidence’s `head`. Prefer “other evidence-focused FULL fixtures” and “content-bound PASS.”

### E1 diagnosis and fix

The diagnosis is correct. Commit `d28afcf` added the first tracked evidence candidate, `depth_canon_smoke_evidence_2026-10-26_03-43-23.json`, whose verdict is FAIL. At `f26e42f`, E1 did not clear it. Consequently:

- E1 still passed because FULL returned rc 2.
- E1b failed because the checker reported the tracked candidate’s refusal instead of “no candidate.”
- The full run shows that sole failure at E1b.
- The isolated rerun ended `30 passed / 1 failed / 0 skipped`, again solely E1b.

The added `clear_evidence` at [depth_canon_guardtests.sh:484](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:484) is minimal, safe, clone-local, and correct for E1.

No watcher-pytest equivalent was found. Its fixture constructs a synthetic repository and copies only the watcher plus synthetic pins/launcher at [test_depth_canon_autolaunch.py:84](src/tests/test_depth_canon_autolaunch.py:84), so the directory enumeration at [line 394](src/tests/test_depth_canon_autolaunch.py:394) never sees committed archived state files.

### Production behavior

Commit `8025475` changes no production file. Committed records behave as follows:

- FULL sorts candidates by filename, not mtime, and tests every candidate newest-name-first at [depth_canon_launch.sh:387](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:387). Fresh-checkout mtimes are irrelevant. A newer FAIL cannot block an older qualifying PASS.
- A committed FAIL changes only the no-qualifying-evidence diagnostic. A committed qualifying PASS intentionally authorizes FULL.
- Evidence `head` is not part of `NEED` at [lines 320–327](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:320); content hashes and harness/provenance fields control eligibility. Record-only HEAD changes are intentionally accepted.
- The watcher uses exact active token/state paths at [depth_canon_autolaunch.sh:42](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:42). Suffixed `*.failed_smoke_*` archives, historical logs, and `.out` captures are ignored.
- The eval driver hashes an explicit relevant-file set at [depth_canon_eval_run.sh:116](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:116); evidence, watcher records, and logs are outside it. Its only experiment-folder glob is the deliberate exactly-one FULL checkpoint-manifest contract at [depth_canon_collect.py:493](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:493).

### Verification

- `8025475` is four added lines in the guard only.
- `bash -n` and `git diff --check` passed.
- Launcher remains `0794fa0e…`; pins remain 48 entries, manifest `631da22f…`.
- The isolated post-fix log currently shows all 31 E cases passing, including E1b, but has not printed its final summary because it is still in section Q.
- The full post-fix guard currently has no displayed failures and had reached E13 at my last read; it remains running.
- The prior regression is `2403 passed / 7 skipped / 1 xfailed`.

Fix Q2, re-run the relevant guard verification, and keep the launcher/watcher disarmed until that is green.
