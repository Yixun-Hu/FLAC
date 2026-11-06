**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_36 plan review (plan_yaw_c4_spread.md r0)

**Verdict:** APPROVE-WITH-CHANGES

The experiment design is sound, but the Coder brief must incorporate every BLOCKING and HIGH item below. No further plan-review round is needed.

## BLOCKING

1. **Separate “exp_35 row” from “requires no-flash.”**

   H35 overloads `is_exp35()` for campaign identity, receipt schema, backend verification, launch planning, and `--count-no-flash`. Adding `YAW-C4S` to that predicate would make the 24 intentional `exp25_...` names fail the campaign-name gate in `verify_cell()`.

   Required change:

   - Keep `EXP35_BLOCKS` and `is_exp35()` unchanged.
   - Add `EXP36_BLOCKS = ("YAW-C4S",)` and a predicate such as `uses_no_flash(row)`.
   - Use `uses_no_flash()` for plan entry `no_flash`, receipt schema 2/backend fields, backend-log revalidation, `build_receipt`, and `--count-no-flash`.
   - Retain `is_exp35()` for exp_35 collection and identity checks.
   - Add an explicit exp_36 name validator allowing exactly:
     - 24 `exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270}` aliases;
     - one `exp36_YAW_C4_S40000_K8_s42_th90` replay.
   - Update test helpers that currently choose receipt v2 using `is_exp35()`.

   Once `--count-no-flash` uses the new predicate, the runner’s overlay/launcher inclusion in `RELEVANT`, no-flash preflight, plan-count check, and receipt backend flow work without changing `EXP35_BLOCKS`.

2. **The generator needs explicit tag/name control.**

   Current `_row()` derives both prefix and tag from the block. Given `YAW-C4S`, it would produce an incorrect `exp25_YAW_YAW-C4S...` name without `_th<angle>`, and it cannot produce the differently prefixed replay.

   Required change:

   - Add a narrow explicit name/tag override or construct these rows from the existing YAW-pin C4 templates.
   - The new block contains **25 rows total**, orders 346–370, priority 18: replay plus the 24 production cells.
   - Assert all 24 metrics suffixes are exactly  
     `1_1.0_exp25_YAW_C4_S40000_K{k}_s{s}_th{a}_rot{a}`.
   - Assert the replay differs from the K8/s42/90 template only in cell/eval name and manifest bookkeeping.

   Preserve both historical guards:

   - `build_manifest(exp36=False)` must render the complete H35 manifest exactly, SHA-256 `440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d`.
   - The exp_25-only render remains SHA-256 `1073ef647434318d219dba3c20e3a18b5a909c90aa3ee6e6bd86021c965f0575`.

   Do not merely change existing “346 rows” assertions to 371; retain an exact H35-prefix/H35-render regression test.

3. **The proposed dump-free replay cannot currently pass `_verified_cell`.**

   `_verified_cell()` has one `base_dir` and calls `verify_cell()` with its default `expect_dump=True`. The archived reference and new replay also live in different worktrees.

   Required change:

   - Extend it, or introduce an exp_36-specific helper, with explicit `base_dir`, `expect_dump`, and `expect_digest`.
   - Verify the archived reference at the exp_25 worktree, pin `d6ca307a…`, `expect_dump=True`.
   - Verify the new replay in the exp_36 worktree, pin H36, `expect_dump=False`, and require its receipt campaign digest to match the replay runner plan.
   - Directly verify the three archived raw digests before comparison: metrics, stream, and receipt.
   - The replay verdict must record H36, campaign digest, reference/new artifact hashes, receipt hash/backend, and comparison result.

4. **Make the hard stop fail-closed operationally.**

   A sequence written as separate commands is too easy to bypass accidentally.

   Required change:

   - Use one checked orchestration or an `&&` chain: replay runner → replay comparator → production runner.
   - Production must not start unless comparator exit code is exactly 0.
   - `--exp36-publish` must re-run the replay check or consume a SHA-bound equality verdict matching H36 and the campaign digest.
   - A mismatch/refusal must prevent both the 24-cell launch and publication.

## HIGH

1. **Keep `YAW-C4S` out of default selection.**

   Otherwise a default runner selection includes the replay and production rows without the replay gate. Introduce a general default-excluded set containing `E2-FAIL` and `YAW-C4S`, but keep the exp_35 E2 contingency set separate: `exp35_results()` currently assumes every `EXPLICIT_BLOCKS` row is E2-FAIL evidence.

2. **The replay equality definition is otherwise correct.**

   Type-strict equality of `metrics`, `by_scene`, and every remaining metrics-record key except `source_sha` and `eval_name`, plus raw stream SHA equality, is the right semantic definition.

   No other metrics-record key should be excluded:

   - `ckpt_path` remains the same relative farm path when `FARM_ROOT` is preserved; a different path should fail.
   - `n_samples` must remain 6337.
   - The record contains no timestamp.
   - Loaded checkpoint SHA/bytes must agree in record and receipt.
   - The new receipt must be schema 2 with SDPA, pinned overlay/launcher hashes, and a revalidated backend log.

   The archived schema-1 receipt does not prove its backend. Its current evaluator log shows the missing-flash fallback and hashes to `8cbf6e24703c9957f9fdd6c9295af8c96063e19694fcf356f1ea0aeb6a0e8ea9`; pin and validate that log if the report claims the archived run itself is backend-proven.

3. **Dumps OFF for both replay and production is sound.**

   It keeps the replay on the same execution path as the 24 cells and avoids unnecessary waveform publication. Dumping copies the already-scored tensor and does not alter RNG or the tensor; the exact replay empirically guards the remaining concern. Do not compare the archived dump receipt wholesale because `dump=true` versus `false` is intentionally different.

4. **Harden publication.**

   Before copying:

   - Offline-verify exactly the 24 source cells as dump-free, receipt-v2 SDPA cells under H36 and one campaign digest.
   - Refuse unless the destination is the pinned, plain, non-symlink exp-25 YAW directory and the exp-25 worktree is at the expected HEAD.
   - Preflight all 48 destination paths with `lexists`; refuse the entire publication if any exists.
   - Create each destination atomically with no-replace semantics, then verify destination SHA equals source SHA.
   - Never publish the replay or any `.cell.json` receipt.
   - Include source receipt hashes, source/destination paths and hashes, H36, campaign digest, exp-25 HEAD, and archived-reference hashes in the publication manifest.
   - Define partial-publication recovery explicitly; a crash must not leave a truncated final file.

   With those controls, the copy does not change any existing exp_25 artifact, receipt, campaign digest, or retention decision. The exp_25 collector selects registered manifest rows rather than hashing the directory globally.

5. **Do not run A opportunistically on both cards in one long runner invocation.**

   The runner immediately refills an available lane after a cell finishes. It cannot “return the second card after its current cell” while continuing on the first.

   Use one fixed card for A and reserve the other for B. If A temporarily uses the second card, give it a bounded invocation that terminates before B claims that card. Keep `ALLOW_SHARED_GPU=0`. Only the shared A/B coordinator may write RELEASE, after both experiments are finished.

6. **Bring the records and logs onto the H35-derived branch.**

   H35 does not contain the exp_36 plan/query directory; those files exist only at `241d73b1`. Import just the exp_36 records into the leaf branch.

   The runner also hard-codes logs and plan JSONs under the exp_25 worklog. Add a validated exp_36 log directory or acknowledge and preserve both locations; wrapping stdout with `tee` alone does not relocate per-cell, preflight, and plan logs. Any new launch-control script must be committed and included in the relevant provenance set.

7. **The paper-script update is more than `YAW_SEEDS`.**

   Changing `YAW_SEEDS` to seeds 42–46 makes the computation correct, but [plot_yaw_c4_three_models.py](/home/yixunhu/codespace/-Neurips-2026-workshop-Sound-Localization/scripts/plot_yaw_c4_three_models.py:1) still labels Yaw-Augmented FLAC as “evaluation seed 42” in its docstring and figure footer. Yixun’s paper-repo change must update those labels too.

## NIT

- Record the block count explicitly as `YAW-C4S: 25`, not “24 rows plus replay” without stating whether the replay belongs to the count.
- Put the replay first in block order even though launch safety must not depend on order.
- Keep exp_36 separate from H35 and from the phase-(ii) kit. Later integration into `localization-exp` should occur only after the exp_35 code lineage is integrated, as the plan already states.

## What I verified

- Main checkout: `localization-exp` at `241d73b1`; H35 worktree: exact `5b065801`.
- H35’s tracked runner, collector, generator, manifest, and no-flash overlay are clean against `5b065801`.
- H35 manifest is 346 rows: 230 exp_25 plus 116 exp_35, orders 0–345.
- The no-flash overlay and launcher hashes match the pinned values.
- The plot consumer expects exactly the proposed rotated filenames and fields. All ten required angle-0 YAW files for seeds 42–46 already exist; none of the 24 new rotated targets currently exists.
- Archived replay hashes and sizes match plan r0. Its metrics record has no timestamp; `ckpt_path`, `n_samples=6337`, loaded SHA/bytes, protocol fields, metrics, and per-scene fields are stable comparison keys.
- The archived evaluator log currently records the missing-flash SDPA fallback.
- The exp_25 YAW destination is presently a plain directory, and the proposed 24 destinations are collision-free.
- H35 is not an ancestor of `241d73b1`; the exp_36 plan/query files therefore must be imported explicitly into the H35-derived worktree.
- I did not run tests, scripts, evaluators, collectors, GPU commands, or processes; I did not access either prohibited coordination directory or modify any file.
