**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-26 · **Round:** code rC4 (review of plan v15: projection cap 300 h under shared, after SMOKE 2 failed on the cap alone)

**Verdict:** SHIP, conditional only on the already-running 27-file regression and full guard suite finishing green. No BLOCKING or HIGH findings. Once both are green, the training launcher/watcher may be re-armed.

### Findings

- **BLOCKING:** None.
- **HIGH:** None.
- **NIT — records receipt pending:** [depth_canon_worklog.md](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_worklog.md:1449) accurately records through arming 2 but does not yet contain the SMOKE-2/v15 outcome; [commits_depth_canon.md](worklog/worklog_yixun/exp_25_depth_canon_claude/commits_depth_canon.md:234) likewise ends before the three v15 commits. This is consistent with the stated post-review receipt workflow and is not code-blocking. Concrete fix: in the receipt commit, append SMOKE 2, the RED/GREEN verification, rC4, and ledger rows; use 300.7 h for R6d rather than the commit subject’s 301.1 h typo.

### Review items

- **P1 — RESOLVED.** [depth_canon_launch.sh](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:71) changes only `MAX_PROJECTED_HOURS=300` plus explanatory comments and the section-(7) comment. FULL passes the literal into its evidence checker and demands exact typed equality at [lines 313–327](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:313). SMOKE 2’s evidence contains 120 and the old launcher hash, so it is doubly ineligible; a new v15 SMOKE is mandatory. The watcher contains no projection-cap reference, is byte-unchanged, and hashes to `d1aa5e41ddbd…`.

- **P2 — RESOLVED.** The evidence fixture, U15, R1b, and R6–R6d bind the literal and both boundary behaviors at [guard lines 618–724](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:618). The RED run is genuine: launcher 120 at `18ca5d4`, exactly **38 passed / 7 failed / 0 skipped**, namely U15, dependent U15b, R1b, and R6/R6b/R6c/R6d. R6c returned rc 4, reproducing SMOKE 2.

  Delete-the-branch behavior is strong:

  - Revert to 120: those same seven fail; R6c changes from pass to rc 4.
  - Inflate to 3000: those same seven fail; R6 and R6d incorrectly pass, while literal/evidence/message checks fail.

  The pytest edit at [test_depth_canon_autolaunch.py:263](src/tests/test_depth_canon_autolaunch.py:263) is fake output only. The watcher classifies solely by launcher rc at [depth_canon_autolaunch.sh:255](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:255); it never parses that message.

- **P3 — RESOLVED.** A 300 h hard sanity cap is defensible under the already-decided `shared` policy. It admits the observed 214.1 h transient projection with roughly 40% headroom and still refuses rates above about 27 s/step, including an order-of-magnitude slowdown from the 43 h uncontended baseline. It also continues to reject missing/unusable progress stamps.

  It no longer protects against a persistent 120–300 h slowdown, poor utilization within roughly 2.8–7× of baseline, or an operationally unattractive ETA. It should therefore not be presented as a runtime target; it is only a gross-stall/pathology check. The plan describes it that way and preserves the wait-for-idle alternative.

- **P4 — PARTIAL, records only.** The status line, §4(7), both acceptance sentences at [plan lines 184 and 264](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:184), and changelog at [lines 399–403](worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:399) agree with the implementation. The Planner-authored deviation and contended-rate disclosure are explicit. All SMOKE-2 log, evidence, watcher-output, token, and state artifacts remain present. Only the pending notebook/ledger receipt noted above remains.

- **P5 — RESOLVED.** The complete SMOKE log and evidence show every non-projection condition healthy: trainer rc 0 after 608 s, offline W&B run created successfully, banner/ranks/endpoint present, 51 finite losses ending at 2.470, zero checkpoints, and 296.059 ms/batch latency. Exact stamp recomputation was `(557−268)/(25−10) = 19.2667 s/step`, giving 214.074 h. The evidence’s `log_sha256` matches the log. Rank-1 TCPStore/NCCL warnings begin only after the successful W&B finish and rc-0 trainer exit; they are teardown after rank 0 closed the store. No additional pre-rearm HIGH emerged.

### Verification and outstanding runs

- `git diff d838c8e..f26e42f` names only four files; no pinned training-closure file changed.
- Pin generator output is byte-identical to the committed manifest: 48 sorted unique entries, manifest `631da22f…`.
- `bash -n` passed for launcher, guard, watcher, and pin generator; `git diff --check` passed.
- Commit sizes: `18ca5d4` 13+/8−; `ba596e3` 10+/4−; `f26e42f` 5+/2−.
- DRY_RUN evidence at `f26e42f`: rc 0, launcher `0794fa0e…`, offline-mode line before ARGV.
- Pip-freeze hash remained `e896c48f…` across the rC-F3 baseline, SMOKE 2, and v15 DRY_RUN—no environment-change evidence.
- At my last read:
  - Regression was at **56% plus 18 further passing dots**, with no failures displayed.
  - Full guard had passed through **T6**, with every reported case passing; it was still running toward the expected 227/0/0.
- I ran no pytest, launcher, watcher, guard, audit, training/eval, GPU, process-control, or network command; made no online W&B call; changed no tracked file or environment; and touched no running process.
