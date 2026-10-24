**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rA2 (re-verification of rA: launcher / audit gate / pins / watcher)

**Verdict:** **REQUEST-CHANGES**

The original audit-authenticity BLOCKING issue is substantially resolved, but two HIGH runtime issues and the non-green guard harness remain. Do not arm the unattended launch at `a680726`.

## Findings

### BLOCKING

None.

### HIGH

1. **A6e: rc 7 is not actually exclusive to GPU-gate refusals.**

   The launcher documents rc 7 as the dedicated GPU-refusal code at [depth_canon_launch.sh:28](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:28), but it also preserves `train.py`’s exit status at [depth_canon_launch.sh:442](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:442) and returns it at [depth_canon_launch.sh:632](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:632). The guard suite explicitly has the fake trainer return 7 at [depth_canon_guardtests.sh:559](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:559).

   The watcher classifies any SMOKE rc 7 as transient at [depth_canon_autolaunch.sh:236](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:236). A real trainer failure with rc 7 would therefore be retried up to six attempts instead of becoming terminal.

   **Fix:** reserve rc 7 by remapping any post-launch/trainer rc 7 to another terminal launcher code before exit. Add an integrated guard case: fake trainer exits 7 → launcher returns the remapped code → watcher makes one attempt and enters `FAILED`.

2. **A3: two launcher `nvidia-smi` calls still explicitly ignore failure.**

   The actual memory, UUID and occupancy gates are now strict. However, immediately afterward the launcher issues two more queries with `|| true` at [depth_canon_launch.sh:390](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:390). This contradicts plan v11’s “every nvidia-smi query is strict” rule and permits launch after a query fails in the interval following the strict checks.

   **Fix:** remove these redundant best-effort queries and log the already validated values, or run them through `gpu_query` with strict parsing/status handling. Add a guard case for a failure in the disclosure query.

3. **The two guard failures are harness defects, not launcher/gate defects—but the suite must be repaired and rerun green.**

   - D1 is deterministic test-state contamination. Q replaces the harness-tagged report with the genuine production report at [depth_canon_guardtests.sh:608](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:608); D1 then invokes harness mode at [depth_canon_guardtests.sh:629](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:629). The gate correctly rejects it. Restore a fresh harness clone/report after Q.
   - B1 received an already `INCOMPLETE` synthetic report; the gate correctly refused it, as shown in the [guard log:185](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_2026-10-24_11-42-10_planner_rBF1_guardtests.log:185). `write_audit ok` builds the report at [depth_canon_guardtests.sh:99](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:99), then blindly pins and commits it at [depth_canon_guardtests.sh:127](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:127).

   For `mode == "ok"`, make `write_audit` assert `gate.verdict == PASS`, zero errors/survivors, and print the report’s actual error before returning nonzero. The original log does not establish that concurrent load caused the error; it establishes only that fixture generation failed silently.

### NIT

4. **A1 rehash has bounded work/concurrency, but no wall-clock bound.**

   It submits every row to an eight-thread pool at [depth_canon_audit_gate.py:187](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:187), with default eight workers at [depth_canon_audit_gate.py:256](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:256). There is no deadline; a hung CIFS read can stall the unattended watcher indefinitely.

   **Fix:** put the gate subprocess under a reviewed timeout or move the rehash into a killable subprocess with a deadline. A thread-future timeout alone is insufficient because running I/O threads cannot be cancelled reliably.

5. **The “exactly four lines” token check has an EOF/blank-line bypass.**

   Blank lines are skipped at [depth_canon_autolaunch.sh:131](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:131), while `wc -l` counts newline bytes at [depth_canon_autolaunch.sh:141](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:141). Four key lines plus a blank line, with no newline after the final key, has five logical lines but four newline bytes and is accepted.

   **Fix:** count every logical line read, including blanks, and require total lines = nonblank lines = 4. Extend the test at [test_depth_canon_autolaunch.py:402](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_autolaunch.py:402) with the no-final-newline form.

6. **“Clone under `/tmp` only” is the default, not enforced.**

   The guard root honors arbitrary `TMPDIR` at [depth_canon_guardtests.sh:28](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:28). Hardcode `/tmp/depth_canon_guard.XXXXXX` or validate the resolved root before cloning.

## Per-item status

- **A1 — PARTIAL.** The original self-authentication defect is resolved for the currently reviewed HEAD:
  - report and sidecar are committed and pinned at [depth_canon_make_pins.sh:29](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_make_pins.sh:29);
  - launcher and collector use the same predicate at [depth_canon_launch.sh:245](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:245) and [depth_canon_collect.py:1135](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1135);
  - exact rows/paths/counts/types are checked at [depth_canon_audit_gate.py:138](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:138);
  - every one of the 9,650 maps is rehashed—`pool.map` receives all exact-census rows, not a sample—at [depth_canon_audit_gate.py:187](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:187);
  - production rejects harness reports at [depth_canon_audit_gate.py:213](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:213).

  There is no environment or CLI switch to skip rehashing. The Python API exposes `rehash=False` for unit tests, but both production callers use the default. Partial status is solely the missing elapsed-time bound.

  A semantic forgery could still be constructed because candidate/tie results are checked for internal consistency rather than recomputed from map content. It cannot pass at the current pinned HEAD: it would require changing the report, recomputing the sidecar, regenerating the manifest, committing all three, changing `EXPECT_HEAD`, reissuing the GO token, and producing a new SMOKE. Each omitted step is caught. This is the reviewed-HEAD trust boundary, not the former “edit report and sidecar in place” bypass.

- **A2 — RESOLVED.** `assert_vit_pin()` requires the sole snapshot and exact file set at [assert_arm_configs.py:65](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py:65). The actual snapshot contains only:
  - `config.json` → `9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b`
  - `model.safetensors` → `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`

  No preprocessor/tokenizer is consumed by `AutoModel.from_pretrained`; any extra `.py` or other file is rejected by the exact file-set comparison.

- **A3 — PARTIAL.** Launcher and watcher gate queries correctly capture status separately, require one numeric memory row, and reject malformed/unknown UUID rows. The remaining two ignored launcher queries are HIGH finding 2.

- **A4 — RESOLVED.** The watcher passes literal `MB=32 ACC=1 DRY_RUN=0` and scrubs harness variables at [depth_canon_autolaunch.sh:194](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:194).

- **A5 — RESOLVED.** Timing begins before carrier unwrap and validation at [diffusion.py:642](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:642), with the focused regression at [test_depth_canon_training.py:268](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_training.py:268). The disabled path remains untouched by this enabled-only function.

- **A6 — PARTIAL.** Exact interpreter, five override refusals, dirty-tree `WAITING_TOKEN`, rc-only classification, and tee draining are implemented. The rc-7 collision is HIGH finding 1; exact token line counting is NIT 5.

## Content-scoped bind

The current evidence values are:

| Field | SHA-256 |
|---|---|
| manifest | `631da22ffc31fa1d0935b672a99d2259a24e757a7d1af44489116642d82f03fb` |
| launcher | `ec80805720c2b2c49ab9238b51a38d2a0348cb66cb06eab619668b797c5dcccf` |
| make-pins | `8da7cd9bbe35f3744aec3b6b31a27bbfebd51c7c547dfdb17e7275852de00fa2` |
| contract, `ARM || CONTROL` | `ff91b9540a037b41151b05d2bd0d091c44a9342ec03858e54d89bb83e8d2a115` |
| audit report | `b58f253ad3310b9c3701709baff9667318a208a2df2300ed743b8725cd17a62e` |
| audit gate | `282b253991151385e04edad96fb70c8a8656a7da18fc92f47f6276b0fc17086b` |

They are produced at [depth_canon_launch.sh:249](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:249) and required from FULL evidence at [depth_canon_launch.sh:261](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:261). The manifest internally pins the report and sidecar; its sidecar digest is `4cbfe1f3…`. Thus launcher, generator, gate, report, contract inputs, manifest, sidecar, and training closure are bound.

`EXPECT_HEAD` remains strict HEAD identity at [depth_canon_launch.sh:140](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:140); only the SMOKE→FULL evidence comparison is content-scoped.

## Planner adjudications

- **(a) Confirm policy, contest exclusivity.** Treating a launch-time `nvidia-smi` failure as transient rc 7 is reasonable and bounded to five retries, but rc 7 must first be reserved against trainer exit-code collision.
- **(b) Confirmed.** `AR_md` falls back to `replace=True` at [AR_md.py:103](/home/yixunhu/codespace/exp-25-depth-canon/src/configs/dataset_configs/custom_metadata/AR_md.py:103). `context_replacement()` compares the repeated-target set across every valid `(split,K)` cell at [depth_canon_collect.py:739](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:739). My read-only probe of the 11 exp_15 unseen K=8 streams found 6,337 positions, exactly 520 repeated-context targets in each, with one identical target set.
- **(c) Confirmed.** Current-map rehash uses `ThreadPoolExecutor`, default eight threads—not processes—after the audit enumeration has loaded the training stack.
- **(d) Confirmed.** Driver exit 4 is dedicated to PIN DRIFT at [depth_canon_eval_run.sh:36](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:36) and [depth_canon_eval_run.sh:430](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:430).
- **(e) Confirmed.** Relative `*_abs` ledger fields need to end with `/<relative path>`; absolute registered paths require equality at [depth_canon_audit_gate.py:121](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:121).

The opt-in Q section performs only real-NAS reads: map rehash, `findmnt`, `df`, and `stat`; `DRY_RUN=1` skips the write probe. The clone is under `/tmp` in the normal environment, subject to NIT 6.

## Verification performed

Read-only only:

- Confirmed HEAD `a680726fe12883759878e306459b31a7d5be809d`; scoped implementation is clean and byte-identical from `76de174` through `a680726`.
- `git diff --check 944636c a680726`.
- `bash -n` on launcher, watcher, guard suite, and pin generator.
- `ast.parse` on all scoped Python files.
- Regenerated the manifest to stdout and compared it byte-for-byte; all 48 `sha256sum -c --strict` entries passed.
- Parsed the committed report: PASS, 9,650 rows, 9,650 unique paths, zero row errors/survivors, no harness tag, matching sidecar.
- Inspected the real DINO snapshot file set and hashes.
- Counted replacement targets across the 11 exp_15 K=8 stream artifacts.
- Read the Planner receipts: 2,327 passed / 7 skipped / 1 xfailed; real audit gate exit 0 with 9,650 maps rehashed; guard 161 passed / 2 failed.

I did not run pytest, training/evaluation, launcher, watcher, audit, guard suite, GPU commands, package/environment changes, or process-management commands.