**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rA3/rB3 (re-verification of rA2 + rB2 after fix round rB-F2)

**Verdict:** **REQUEST-CHANGES** — all F2-1 through F2-7 issues are resolved, and the training launcher/watcher may be armed. One separate HIGH, eval-side exact-byte provenance gap remains in the condition probe; fix it before the registered probe/evaluation campaign. It does not block training launch.

## Findings

**BLOCKING:** None.

**HIGH — condition-probe checkpoint provenance remains path-based.**

[depth_canon_cond_probe.py:243](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:243) still calls `torch.load(ckpt_path, map_location="cpu")`. The checkpoint is hashed later by path at [depth_canon_cond_probe.py:575](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:575). Thus, the digest bound by [depth_canon_collect.py:897](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:897) does not prove which bytes were deserialized—the same TOCTOU bracket fixed for B4 in the evaluation driver.

Concrete fix: use `eval_FLAC.read_checkpoint_exact`, load from `io.BytesIO(buf)` with the existing `map_location`, record `ckpt_sha256_loaded` and `ckpt_bytes_loaded` from that buffer, and require those values in `PROBE_CONTRACT` to match the CANON@40k manifest entry and expected length. This is required before the registered probe/evaluation campaign, not before training.

**NIT:** None newly identified.

## Per-item re-verification

- **F2-1 — RESOLVED.** The EXIT trap at [depth_canon_launch.sh:75](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:75)–89 reserves rc 7 exclusively for `gpu_refuse` and remaps every other rc 7 to documented terminal rc 8. This includes trainer exits, `set -e` failures, and Python-heredoc exits. The audit gate is explicitly mapped to rc 2 at [depth_canon_launch.sh:265](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:265). The watcher retries only rc 7 at [depth_canon_autolaunch.sh:235](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:235)–248. R9/R9b and Z9 cover trainer rc 7 and terminal watcher behavior. **Rc 8 is acceptable**: it is documented, dedicated, and terminal. The broader pre-launch remapping deviation is a sound strengthening.

- **F2-2 — RESOLVED.** No `nvidia-smi ... || true` remains in the launcher or watcher. Launcher queries are strict at [depth_canon_launch.sh:364](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:364)–406; disclosure uses the already validated values at lines 412–417. Watcher queries capture and check status at [depth_canon_autolaunch.sh:164](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:164)–191. Dropping duplicate disclosure queries is acceptable. U12/U12b/U13 make the strict behavior load-bearing.

- **F2-3 — RESOLVED.** The clone root is constrained to `/tmp` at [depth_canon_guardtests.sh:28](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:28)–31. Fixture construction fails loudly at lines 46–59. `write_audit ok` requires PASS, zero errors and zero survivors, and prints the actual gate/errors/survivors on failure at lines 104–109. T15 exercises that diagnostic. Q is followed by a fresh clone before D1 at lines 640–662. The Planner log confirms Q0 re-hashed 9,650 maps and D1 passed. Had the concurrent B1 failure recurred, the new assertion would have named the gate and first recorded errors/survivors. No retry is warranted.

- **F2-4 — RESOLVED.** Re-hashing now runs in a separate process group with a 900-second deadline at [depth_canon_audit_gate.py:197](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:197)–244. Timeout kills the group and fails closed; the launcher maps it to terminal rc 2. The fake-slow-hasher test covers the deadline. **900 seconds is reasonable**: approximately 6.7× the measured 135 seconds permits CIFS and concurrent-load variance while bounding an unattended stall to 15 minutes.

- **F2-5 — RESOLVED.** Token validation counts both logical and newline-terminated lines at [depth_canon_autolaunch.sh:127](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:127)–146 and requires both counts to equal four. Blank lines and the no-final-newline form are rejected. The added tests cover both cases.

- **F2-6 — RESOLVED within the evaluation driver/evaluator scope.** Existing cache entries must be regular, non-symlinked, exactly `0444`, correctly sized and correctly hashed at [depth_canon_eval_run.sh:203](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:203)–245. Farm and cache files are re-hashed immediately before every spawn at lines 433–438. `eval_FLAC.py` reads and hashes one exact buffer at [eval_FLAC.py:361](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:361)–387, loads it through `io.BytesIO` at lines 1917–1920, and records digest and byte length. Receipt creation and `verify_cell` enforce loaded = measured = manifest at [depth_canon_collect.py:624](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:624)–670.

  The default evaluator is behaviorally unchanged apart from those two additive provenance fields; `ckpt_load=None` record construction remains unchanged. `BytesIO` does not alter the installed PyTorch version’s `weights_only`, pickle-module, or `map_location` decisions. Only path-dependent `mmap` would differ, and it is neither passed nor enabled here. A 724 MB input buffer, possible `BytesIO` copy, and deserialized state are acceptable for two evaluators on this host: roughly 4–5 GiB aggregate against approximately 153 GiB available during review.

- **F2-7 — RESOLVED.** Retention-audit mode rejects `--dump-check receipt` before NAS selection at [depth_canon_collect.py:1480](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1480)–1484. The test confirms exit 2 and that no dumps are touched.

## Evidence and adjudications

- The regenerated pin manifest was byte-identical to the committed manifest; all 48 entries passed strict `sha256sum`. No pinned training-closure path changed between `a680726` and `ee61263`.
- The committed tie-audit report and sidecar are byte-identical to `a680726`.
- Only `depth_canon_launch.sh` and `depth_canon_audit_gate.py` changed among content-bind inputs. Both are listed with their current hashes in plan v12/receipt, so the Planner’s final-pin DRY_RUN covers the changed bind.
- RED evidence is credible and test-first. Recorded cycles were: F2-1 2 failures; F2-2 3; F2-3 T15 failure; F2-4 7; F2-5 1; F2-6 94 then 78; F2-7 1. The corresponding green subsets passed. The F2-6 stash-based counts are acceptable because the immutable red commits contain the new tests while lacking the production branches.
- The relevant tests are load-bearing: removing rc remapping, strict GPU queries, audit assertions, deadline handling, total-line validation, exact-buffer loading/cache checks, or retention rejection causes the associated red behavior.
- Every F2 commit is under 200 changed lines; the largest is 122 lines.
- The additive provenance fields on all actual `evaluate_model` records and length binding to the driver’s post-run `stat` are accepted.
- No package/environment mutation appears in the scoped changes or receipts, and this review made none.

## Probes run

I ran only permitted read-only checks: `git status/log/diff/show`, scoped `git diff --check`, manifest regeneration through process substitution and `cmp`, `sha256sum -c --strict`, historical `git show | cmp`, `bash -n`, Python AST parsing, targeted `rg`, PyTorch signature/source/config inspection, and `free -h`.

I inspected the cited Planner regression, guard, and real audit-gate logs. They report 2,369 passed / 7 skipped / 1 xfailed, guard 169/0/0 with Q0 and D1 passing, and real audit gate exit 0. I did not run pytest, the guard/audit suites, launcher, watcher, driver, training/evaluation, GPU commands, or touch any process.