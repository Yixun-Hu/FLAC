**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-22 (integrative full review)

## Verdict: REQUEST-CHANGES

The training implementation is coherent and matches Query 2, but the whole experiment is not launch-ready. Four evaluation/admission gaps could produce plausible-looking, protocol-invalid science after the 40k run.

## BLOCKING findings

1. **Evaluation does not prove that the checkpoint was trained as `fa_cartesian`.**

   Training embeds the complete model config in every checkpoint at [train.py:16](/home/yixunhu/codespace/FLAC/train.py:16). Evaluation loads it at [eval_FLAC.py:1198](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1198), but config-to-checkpoint comparison is applied only when ARE is involved; non-ARE BFC returns without comparison at [eval_FLAC.py:309](/home/yixunhu/codespace/FLAC/eval_FLAC.py:309).

   Consequently, a same-architecture Vanilla or B-F checkpoint can load successfully, be evaluated with the CLI’s `fa_cartesian` conditioning, and produce a record claiming `cond_method: fa_cartesian`. The admission gate verifies that claim, not how the weights were trained. This is precisely the catastrophic mismatch forbidden by announcement 05.

   Required fix:

   - For BFC evaluation, require an embedded `model_config`.
   - Type-strictly bind it to `FLAC_AR_BFC.json`, including training `cond_method == fa_cartesian`, C4 angles, and training cap 32.
   - Fail before model/GPU construction for missing, Vanilla, `fa_invariant`, angle-drifted, or cap-drifted embedded configs.
   - Add mutation-resistant tests for those cases.

2. **The prior checkpoint-SHA blocker remains only conditionally implemented.**

   `build_metrics_record` writes no checkpoint digest at [eval_FLAC.py:941](/home/yixunhu/codespace/FLAC/eval_FLAC.py:941). The validator explicitly acknowledges this and lets the check “sleep” at [exp21_validate_cell.py:82](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:82); absence is accepted at [exp21_validate_cell.py:228](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:228).

   The positive fixture contains no digest yet is expected to validate at [test_exp21_table_gate.py:90](/home/yixunhu/codespace/FLAC/src/tests/test_exp21_table_gate.py:90) and [test_exp21_table_gate.py:221](/home/yixunhu/codespace/FLAC/src/tests/test_exp21_table_gate.py:221). The generator compares digests only when present at [gen_model_comparison.py:609](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:609), and the cross-K transaction compares paths but not digests at [gen_model_comparison.py:698](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:698).

   Thus all ten rows may omit the digest and pass, or K=1 may consistently use bytes A while K=8 uses bytes B at the same path. That does not satisfy plan §3g or the r4 requirement of one checkpoint digest across all ten cells.

   Required fix:

   - Compute the loaded checkpoint’s SHA-256 once and include `ckpt_sha256` in every BFC metrics record.
   - Require a well-formed digest—absence must block.
   - Require one digest across all five seeds and both K values.
   - Add tests for all-digests-missing and K1-digest-A/K8-digest-B.

3. **Registered table cells do not prove that the full split was actually evaluated.**

   The dataset silently substitutes a random item on rejection at [dataset.py:342](/home/yixunhu/codespace/FLAC/src/data/dataset.py:342) and on any loading exception at [dataset.py:358](/home/yixunhu/codespace/FLAC/src/data/dataset.py:358). Such substitution still increments `n_samples`, and usually preserves the ten-family key set, so the current validator can admit the row.

   `eval_FLAC` already has the correct positional substitution guard at [eval_FLAC.py:639](/home/yixunhu/codespace/FLAC/eval_FLAC.py:639), but it runs only when a stream is accumulated. The registered table command at [exp21_validate_cell.py:15](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/exp21_validate_cell.py:15) omits `--record-stream --expected-stream-count 6337`; those flags are currently reserved for the invariance grid.

   Required fix:

   - Add both flags to all ten registered BFC cells and the D6 comparator cells.
   - Make admission require durable proof that the positional/count checks passed—either a required sidecar validated by the gate or BFC-specific fixed-mode stream provenance in the metrics record.
   - For paired comparisons, compare the per-seed/K input identities across arms.

4. **Approved D6 comparator hygiene is not assembled or enforced.**

   Rev 3 approved re-evaluating both B-F and P1 at the current evaluator pin. The generator still contains only the historical B-F/P1 rows at [gen_model_comparison.py:70](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:70) and [gen_model_comparison.py:85](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:85), while the new BFC comment says it is read as a paired delta against them at [gen_model_comparison.py:163](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:163).

   No reviewed 20-cell comparator manifest, new row specs, or cross-arm same-pin transaction exists. BFC validation also accepts any internally uniform 40-hex `source_sha`, not the reviewed campaign pin. As assembled, the living table can visually invite the exact cross-pin paired reading D6 rejected.

   Required fix:

   - Add exact, reviewed B-F/P1 reevaluation templates or a driver importing one shared protocol definition.
   - Register separate current-pin comparator rows.
   - Require the reviewed evaluator pin across BFC, B-F, and P1; bind each arm to its proper embedded training config and checkpoint digest.
   - Keep historical rows explicitly contextual and prohibit paired-delta claims against them.

## Confirmed correct

- **Filename and field vocabulary match.** `eval_FLAC` produces `…step=40000_metrics_1_1.0_exp21_BFC_S40000_K{K}_s{seed}_fa_cartesian_a4.json` via [eval_FLAC.py:395](/home/yixunhu/codespace/FLAC/eval_FLAC.py:395), matching the exact globs at [gen_model_comparison.py:171](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:171). The actual keys are correctly named: `cond_autocast`, `frame_avg_fwd_cap`, `n_samples`, `ckpt_path`, `by_scene`; angles and `rotate_deg` are emitted as floats. There is no pending-forever vocabulary mismatch.

- **The method matches binding Query 2.** Raw Cartesian metadata receives one base pass; all four pose/geometry IDs are required and averaged over C4 at [yaw_rotation.py:625](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:625). Depth and all four pose keys rotate jointly at [yaw_rotation.py:643](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:643). `context_audio` remains from the single base pass. No cylindrical conversion or canonicalization occurs.

- **Training dispatch is real and non-vacuously tested.** Factory plumbing reaches the wrapper, whose branch calls the correct function at [diffusion.py:562](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:562). The training/validation/test-step spy at [test_fa_cartesian_dispatch.py:436](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian_dispatch.py:436) would fail on Vanilla fallthrough.

- **Train/eval cap asymmetry is structurally sound.** Training is cap 32/micro-batch 32, hence one angle per chunk. Evaluation calls the same helper with cap 64/batch 64. The evaluator recursively enters eval mode at [eval_FLAC.py:1232](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1232); DINOv3’s coordinate augmentation is guarded by `self.training`, and the relevant conditioners have no eval-time batch-statistics path. No train-only stochastic route leaks into evaluation.

  The 6,337-item split has a one-sample tail: full batches use one angle per call, while the tail groups all three nonzero angles. That tail partition should be explicitly recorded in the params/command manifests. Real-DINO cap-64 versus grouped-cap parity remains required ladder evidence.

- **Launcher trains the approved arm.** The config, dataset, two-GPU DDP/SyncBN rung, seed, precision, 40k budget, checkpoint cadence, logger, and no-validation/no-resume recipe are pinned at [bfc_launch.sh:105](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:105) and [bfc_launch.sh:193](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:193). The init-identity and DINO pin audit is fail-closed. SMOKE uses its own identity/save directory, no logger, and no checkpoint inside its window.

- **Incident mitigation is adequate for recurrence.** Every manifest-rejection guard case now carries an independent forced dry failure, so deleting the intended rejection cannot again reach `train.py`. The orphaned wandb run `fo10gff6` still needs deletion before launch.

- **Regression safety is good on valid paths.** Vanilla and `fa_invariant` dispatch, defaults, and record shapes remain unchanged; their golden tests are present. The only intended behavior change is the earlier, correctly named frame-average over-cap rejection. `eval_pl` now fails closed for unsupported future methods without changing its valid Vanilla/FA routes.

## NIT findings

- The 20-GiB per-GPU floor is reasonable against the expected roughly 16-GiB/rank footprint, but `MIN_FREE_MB` and `MIN_FREE_DISK_MB` can be lowered in registered mode at [bfc_launch.sh:299](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh:299). Do not override them for launch; preferably restrict the bypass to guard-test mode.
- The yaw-augmentation test explanation incorrectly says C4 is the exact subgroup sampled by the C512 augmentation at [test_fa_cartesian_dispatch.py:388](/home/yixunhu/codespace/FLAC/src/tests/test_fa_cartesian_dispatch.py:388); production comments are correct.
- `_rotated_variants` still describes only `*_vit` poses despite receiving all four pose keys at [yaw_rotation.py:654](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:654).
- `commits_bf_fa_cartesian.md` lists `21d423b` twice at [commits_bf_fa_cartesian.md:22](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/commits_bf_fa_cartesian.md:22).

## Required evidence after fixes

Before the registered launch:

- Re-review the four blocking fixes and rerun the focused suites plus full regression suite.
- SMOKE: two GPUs, isolated identity, logger off, no checkpoint, at least one completed optimizer step, no OOM/NaN.
- Real-data readback: all required metadata/outputs, real DINO C4 spot check, and cap-64 versus grouped-cap allclose.
- ≥200-step co-tenant rate probe with peak VRAM/headroom and stable throughput recorded.
- Final B-F parity audit: exact config/argv deltas, no validation/resume, init hash, B-F legacy per-angle provenance, and explicit train/full-eval/tail chunk plans.
- Write the SOP params and command manifests from the final clean, pushed SHA; delete wandb run `fo10gff6`.

Read-only verification performed here: scoped `git diff --check`, Python AST/JSON parsing, and `bash -n` passed. I did not rerun pytest because its temp/cache behavior would violate the strict no-write constraint; the recorded final evidence is 470 passed/1 skipped for the required set and 65/65 launcher guard cases.