# exp_15 yaw_aug — Codex FINAL ARMING CHECK (evalfinal2)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (read-only) · **Date:** 2026-08-17 · **Verdict: NO-GO** — F6/readiness/generator-validation/V-withholding CONFIRMED; remaining: F1 exact-marker bypass ($0 not bound to the verified tree), incident guard detect-not-prevent, generator per-K write, evidence commit-binding. Incident adjudicated: impact assessment qualified-sound (real unauthorized scheduler action, minimal), recurrence prevention NOT yet structural.

## Verdict: NO-GO

Eval is **not armed** at origin tip `2a183bd`. The reviewed fix/evidence blobs are identical to rebased commits `89fdec5`/`7b71ddb`, but two original findings remain partially open and the incident guard is not structurally fail-closed.

### Findings

1. **BLOCKING — F1 still has an exact-marker bypass.**

   `verify_pinned_marker` correctly checks the claimed directory’s realpath and HEAD. It never verifies that the currently executing script is the script inside that directory. Therefore:

   ```text
   YAW_EVAL_PINNED_EXEC=<correct .measure_worktrees/PIN path>
   bash <main-tree>/yaw_aug_submit_grid.sh ...
   ```

   passes verification and suppresses re-exec because the marker is nonempty. Main-tree shell logic continues running while only subordinate paths point into the pinned tree.

   Single-cell path: `yaw_aug_screen_submit.sh:140`, `:400`, `:408`, `:459`, `:534`. Wave path: `yaw_aug_submit_grid.sh:125`, `:258`, `:266`, `:295`, `:305`.

   The forged-marker regression uses `${CANARY_MAIN}`, an intentionally wrong path, rather than the correct worktree path while invoking the main-tree copy; it therefore misses this case (`yaw_aug_screen_guardtests.sh:1239`). Bind the canonical current `$0` to the expected pinned script or unconditionally re-exec when it differs.

2. **BLOCKING — the incident fix detects isolation failure but does not prevent submission.**

   The harness has no `errexit` (`yaw_aug_screen_guardtests.sh:38`), and `check` merely records a failure and continues (`:190`). The rewritten PATH still includes `/usr/bin:/bin` (`:1170`), where the real Slurm clients reside on this host. Its assertion is nonfatal (`:1172`), and stub creation/`chmod` is likewise not fail-stopped before the live canary runs (`:1196`, `:1199`).

   Thus a failed rewrite, missing stub, or failed executable bit can again fall through to real `/usr/bin/sbatch`. This is not structural prevention.

3. **MAJOR — F7’s actual generator write remains per-K, not transactional.**

   `yaw_aug_publish_row.py` now provides a useful two-K preflight Boolean (`yaw_aug_publish_row.py:174`) and exits nonzero when used (`:208`). But it performs no publication, and nothing wires it into the generator.

   The generator still renders each exp_15 row independently (`gen_model_comparison.py:692`, `:915`); its enforced transaction blocks cover Q9, exp_14, and exp_11, not exp_15 (`:937`, `:955`, `:968`), before writing the file at `:1014`. Direct generator execution can therefore write a numeric K row alongside a BLOCKED other K. The new tests exercise readiness, not `both_k_ready` or the generator write path (`test_yaw_aug_collect.py:1148`).

4. **MAJOR evidence qualification — the transcripts are not commit-bound to the fix.**

   Both transcripts identify tested HEAD `431a4bf`, the pre-fix commit, rather than `1dc2652`: `slurm_guardtests_strict_3706606.out:6` and `yaw_aug_2026-08-16_17-47-37_pytest_evalfinal2.log:3`. All eight reviewed file blobs differ between those commits. The new cases appearing in the transcript show that a dirty working tree was tested, but no recorded content digest proves it exactly matched the subsequently committed fix.

### Confirmed closures

- F6’s rc-preserving capture, `squeue` resolution, and queue failure refusal are correct (`yaw_aug_screen_submit.sh:337`, `:562`, `:584`). The live-shaped missing-cell canary genuinely reaches the stub submission path (`yaw_aug_screen_guardtests.sh:1199`, `:1228`).
- Both-arm/both-K readiness and directly checked T-scoped G3 obligations are correct (`yaw_aug_publish_row.py:100`, `:114`).
- Generator validation requires five files and sidecar pins (`gen_model_comparison.py:401`, `:437`, `:466`), and the §6.10 checklist is emitted (`yaw_aug_publish_row.py:61`, `:215`).
- Defective V values are replaced by `T60: null` in JSON and `WITHHELD` in markdown (`yaw_aug_collect.py:1104`, `:1205`, `:1230`; tests at `test_yaw_aug_collect.py:1098` and `:1106`).
- The recorded evidence reports STRICT 189/0/0, union 189/189, and pytest 126/126 (`slurm_guardtests_strict_3706606.out:225`, `:231`; `yaw_aug_2026-08-16_17-48-10_union_evalfinal2.log:1`; pytest log `:24`).

### Incident adjudication

- **(a) Impact:** Qualified sound. Given the recorded scheduler/filesystem evidence, there was no GPU use, production artifact, or scientific-result impact. Calling it wholly “zero impact” is too broad: job `3706530` was a real unauthorized production-scheduler action, albeit minimal. The correction of the Coder’s false “no job existed” statement is proper (`yaw_aug_worklog.md:304`).
- **(b) Recurrence prevention:** **No.** The current rewrite/assert sequence is nonfatal and retains real Slurm binaries as PATH fallbacks.
- **(c) Other path:** **Yes.** Canary setup failure—especially absent/non-executable stubs—can still reach the real scheduler. I found no separate demonstrated route to production GPU use or eval artifacts.

No installs, environment changes, file writes, submissions, GPU use, or suite runs were performed during this review.