**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C2 (exp_35 rounds 1b, 1c and the C1 fix: 65c60dd8, 9d4e00b6, 0e4efbc0, f5a980bf)

**Verdict:** REQUEST-CHANGES

## Per-item status

| Item | Status | Assessment |
|---|---|---|
| M1 — C1 findings | **PASS for code** | The lexical expected-path fix rejects all five symlink escapes. The restored tests are meaningful. The equivalence test covers `__file__`, resolved path, cwd, launcher/target PID equality and direct-process PID inequality. SOP closure is assessed under M5. |
| M2 — exp_35 runner/backend | **PARTIAL — HIGH fix required** | The normal exp_35 route correctly uses the launcher, overlay-only `PYTHONPATH`, bytecode suppression, backend-log verification and receipt v2. The exp_25 path is nevertheless affected by unconditional exp_35 pin and digest gates. |
| M3 — identity tuple | **FAIL — BLOCKING and HIGH fixes required** | Exact-set comparison and ordinary read/query failures are fail-closed, but the approved tuple is not frozen, the purported final check precedes substantial work, and `/proc` sampling can miss PID reuse during the sample. |
| M4 — tests | **PARTIAL** | Most protections have effective positive and mutation tests. Tests do not cover tuple-file replacement, PID reuse during identity sampling, the post-check/pre-exec window, or exp_25-only admission independence. |
| M5 — SOP | **PARTIAL — NIT** | Production code counts are 90/103/72/17, all commits have the required trailer, and no environment-changing code was added. The scoped endpoint lacks committed closing records, and the final C1-fix suite log is empty. |

## Ranked findings

### BLOCKING 1 — the runner neither freezes the approved tuple nor performs the identity check at the actual pre-exec boundary

The tuple file is validated once at [depth_canon_eval_run.sh:175](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:175), but every lane check later reopens whatever content that path currently contains at [depth_canon_eval_run.sh:433](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:433). No initial or launch-entry SHA is retained. The file can therefore change between selection and final checking so that a newly observed process becomes “approved.”

The final check at [depth_canon_eval_run.sh:487](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:487) is then followed by farm validation and a full checkpoint SHA-256 at lines 493–496 before the background evaluator is started at line 498. A trainer exit, table-job replacement, or extra process during that interval is not observed, yet the cell can run and receive a receipt.

This is wider than the unavoidable, plan-disclosed race after a genuine final pre-exec snapshot and violates §12 B1.1’s frozen tuple and final runner check.

Concrete fix:

- Bind the tuple to a launch-entry SHA or read its validated bytes once and require the same digest before every check.
- Perform another `lane_state`/identity check inside `run_cell`, after every digest/checkpoint operation and immediately before invoking `$PY "$NO_FLASH_LAUNCHER"`.
- Treat failure there as a campaign hard stop with no receipt.
- Add tests that replace both the tuple and observed process set between checks, and that introduce a process after checkpoint hashing but before launch.

### HIGH 1 — `/proc` identity sampling can combine two different lifetimes of one PID

[exp35_gpu_identity.py:33](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_gpu_identity.py:33) reads `stat` once, then reads `cmdline`, cwd and fd links through line 38. If the original process exits after `stat` and the PID is reused before the remaining reads, the result can contain the old start time and the replacement process’s other fields. There is no final `stat` read proving that the sampled lifetime remained unchanged.

The existing “PID reuse” tests mutate the process before `proc_identity()` begins; they do not exercise reuse during sampling.

Concrete fix: read and parse `stat` before and after all other fields and reject unless the PID/start time is identical on both reads. Preferably also hold a `pidfd` or `/proc/<pid>` directory handle. Add a test that swaps the fake process entry after the first `stat` read and requires `ValueError`/runner refusal.

### HIGH 2 — exp_25-only runs now depend on exp_35 files

The evaluator command itself retains its old branch at [depth_canon_eval_run.sh:351](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:351), but behavior is not otherwise unchanged:

- `RELEVANT` always includes the overlay and identity helper at line 131.
- Their content therefore always changes the campaign digest.
- Overlay and launcher constants are checked unconditionally at lines 151–156, before `N_NOFLASH` is known.
- Every exp_25 plan entry gains `"no_flash": false` at [depth_canon_collect.py:698](/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:698).

Thus an exp_25-only run can refuse because an unrelated exp_35 file is absent or changed. The test at [test_exp35_eval_run.py:60](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_eval_run.py:60) checks only the spawned environment and receipt schema, so its “exactly as before” claim misses this regression.

Concrete fix: determine whether the selection contains exp_35 rows before constructing `RELEVANT`; append and pin the overlay/identity files only for such selections. Preserve the old exp_25 plan shape or explicitly default a missing `no_flash` field to false. Add an exp_25-only test that removes or mutates the exp_35 files and still obtains the old plan/spawn behavior.

### NIT 1 — closing SOP evidence is not complete at the scoped endpoint

At `f5a980bf`, the new ledger, coder log and RED/GREEN logs are not present in the committed tree, contrary to the repository convention at [CLAUDE.md:38](/home/yixunhu/codespace/exp-35-orbitrir-yaw/CLAUDE.md:38). The available `orbitrir_yaw_evidence_2026-11-05_12-57-53_C1fix_suites_at_f5a980bf.log` is zero bytes.

A later records-only commit appeared in the live worktree during review; it was excluded because the requested endpoint is `f5a980bf`. It also does not supply non-empty final-suite output.

Concrete fix: preserve a non-empty final regression log at `f5a980bf` and place the ledger/log updates in an explicitly reviewed records batch.

## What I verified

- Review scope was the committed range `da88eb38..f5a980bf`; later worktree/HEAD changes were excluded.
- C1’s lexical-path fix at [exp35_no_flash_launch.py:146](/home/yixunhu/codespace/exp-35-orbitrir-yaw/overlays/no_flash/exp35_no_flash_launch.py:146) compares imported realpaths against lexical paths below the canonical repository.
- The five symlink cases at [test_exp35_no_flash_launch.py:308](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:308) would fail under the former realpath-vs-realpath implementation.
- Direct-invocation equivalence at [test_exp35_no_flash_launch.py:173](/home/yixunhu/codespace/exp-35-orbitrir-yaw/src/tests/test_exp35_no_flash_launch.py:173) covers the requested file, cwd and PID behavior.
- Committed hashes are:
  - overlay: `0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023`;
  - launcher: `bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de`.
- The normal exp_35 spawn exports exactly overlay-only `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE=1`.
- The backend predicate requires one adjacent overlay/fallback pair, one correctly pinned sentinel after it, the expected target, and no `No module named 'flash_attn'`.
- Receipt v2 has the three registered backend fields; missing/wrong fields are refused, while exp_25 receipts retain schema 1 and the original key order.
- Every exp_35 verification rereads the named log.
- Field-22 parsing uses tokens after the last `)` and therefore handles spaces and parentheses inside `comm`.
- Exact observed-PID equality, GPU UUID equality, cwd, raw-cmdline SHA-256 and both fd links are compared. Ordinary `/proc` or `nvidia-smi` failures stop the runner.
- Existing backend, receipt, symlink, wrong-process and second-check tests are meaningful; the listed race cases are not covered.
- Production changed-code counts are `90`, `103`, `72`, and `17`; all are below 200.
- All four commits contain `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No scoped commit changes an environment/package file or contains an installation command.
- I ran no pytest, launcher, collector, runner, evaluator, GPU command or process-control command, and modified no files.
