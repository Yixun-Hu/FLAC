**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-21

Verdict: **REVISE**. The core method is sound: applying one column-quantized rigid rotation to depth vectors, panorama columns, and all four pose fields preserves the GeometryConditioner contract and reuses the Cartesian conditioner without new parameters. However, the plan has several blockers that could invalidate the invariance claim or make the 40k comparison misleading.

## Blocking findings

1. **BLOCKING — The arm definition is still open when this round already fixes it as single-forward.**

   [Plan line 156](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:156) leaves single versus C4 geometry as D1. The current task explicitly defines the arm as “unchanged conditioner stack, single ViT forward.” Remove D1 and register single canonical conditioning as the method. A later canonical+C4 arm would be a separate ablation.

2. **BLOCKING — The all-degenerate fallback is not yaw-invariant for this method.**

   [Plan lines 34 and 64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:34) reuse FA’s `phi_ref=None → identity` rule. That worked for `cylindrical_pose_features`, which emits invariant radii and zero angles. It does not work for target-frame rotation: after a global yaw, returning the rotated depth and poses unchanged leaves the geometry branch yaw-dependent.

   Before approval, the plan must either:

   - audit the complete AR train and both unseen-eval configurations and prove every item has a nondegenerate source or context reference, then fail closed if `reference_azimuth` returns `None`; or
   - define a new principled scene-intrinsic fallback.

   Silently returning identity is not acceptable. Degenerate-source/context-fallback and all-degenerate behavior need distinct tests. Near-tied largest-radius contexts also need a deterministic tie contract or an explicit uniqueness assumption.

3. **BLOCKING — The exactness claims and arbitrary-angle test conflate three different properties.**

   Let the panorama spacing be \(\delta=2\pi/W\). In ideal arithmetic, for a grid yaw \(\beta=m\delta\),

   \[
   q(-(\phi+\beta))+\beta=q(-\phi),
   \]

   so the method is \(C_W\)-invariant. At \(W=512\), the arithmetic claim **45° = 64 columns is correct**, as confirmed by [the current quantizer](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:305).

   But:

   - The actual 0° and rotated paths perform different sequences of float32 rotations, so their tensors are only numerically close, not bit-exact. [Plan line 118](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:118) already uses `atol=1e-6`, contradicting “exact.”
   - `rotate_scene_metadata(17.3°)` first snaps 17.3° to the column grid. It therefore tests the same \(C_{512}\) composition, not continuous-yaw error. [T-Cinf](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:119) is testing the wrong proposition.
   - For a hypothetical truly continuous input yaw, each canonical orientation lies within half a column of ideal alignment, but two independently rounded canonical orientations can differ by as much as one column. “Canonical poses agree within half a column” is not generally valid.

   Reword the claim as “ideal \(C_{512}\) invariance; numerical allclose under the implementation.” Keep the half-column bound only for target-to-x-axis residual. Define separate tolerances for metadata, conditioner output, and metric spread.

4. **BLOCKING — The proposed helper API cannot directly call the existing rotation primitive.**

   [Plan line 50](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:50) says `reference_azimuth` returns a tensor scalar, while [line 30](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:30) passes it to `rotate_scene_metadata`. `yaw_column_shift` invokes Python `round()` on its angle, which does not accept a Torch tensor. Specify the scalar conversion at the boundary, while keeping tensor behavior inside `cylindrical_pose_features` to preserve its outputs.

   The depthless contract is also contradictory: [lines 68–71](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:68) promise quantization without depth, while [line 91](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:91) proposes unquantized rotation. AR requires depth for both geometry conditioners, so the safest contract is to require depth, derive or validate each sample’s width, and fail closed otherwise.

5. **BLOCKING — The evaluation protocol omits mandatory announcement-05 flags.**

   [Plan line 142](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:142) omits `--frame-avg-angles`, `--rotate-deg`, and `--rotate-mode`. Every registered and grid invocation must explicitly contain at least:

   ```text
   --cond-method target_frame
   --frame-avg-angles 0,90,180,270
   --rotate-mode fixed
   --rotate-deg <0|45|90|180|270>
   --cond-autocast bf16
   ```

   The frame angles are explicitly present but ignored for this non-orbit method. Also pin `--batch-size 64`, `--cfg-scale 1.0`, `--steps 1`, and the exact full configs:

   - K=8: `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json`
   - K=1: `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json`

   For the invariance grid, add `--record-stream --expected-stream-count 6337` so all five cells prove identical input ordering and full-split coverage.

6. **BLOCKING — Aggregation and comparison-table integration are unspecified.**

   The SOP requires the aggregation convention to be declared. `eval_FLAC.py` defaults to flat split-level metrics; `--record-per-scene` additionally retains the by-scene payload. The plan should run with `--record-per-scene`, preserve both estimands, and state:

   - flat metrics compare directly to the existing B-F/P1 rows;
   - equal-scene means are the paper-style estimand and cannot be compared to historical flat-only artifacts without same-protocol comparator re-evaluation.

   Announcement 04 also requires a new row spec and regeneration, but [the planned-file list](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:42) omits `gen_model_comparison.py`. Add:

   - two target-frame row specs;
   - a protocol label distinct from both `vanilla eval` and `fa eval`;
   - an admission validator checking full split/count, five seeds, EMA, bf16, batch 64, target-frame conditioning, null frame angles, no orbit, step 40k, and one evaluator pin;
   - generator tests and regeneration at results time.

7. **BLOCKING — Eval provenance expectations are internally contradictory.**

   [Plan line 124](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:124) expects “no orbit fields,” but current records always contain those fields. The required target-frame record should be:

   ```json
   {
     "cond_method": "target_frame",
     "frame_avg_angles": null,
     "orbit_execution": "n/a",
     "frame_avg_fwd_cap": null
   }
   ```

   Training config guards should reject both `frame_avg_angles` and `frame_avg_max_fwd_samples` when `cond_method=target_frame`; either would otherwise be a silently ignored orbit declaration. Evaluation must still accept the explicitly supplied announcement-05 frame-angle flag and record it as unused/null.

8. **BLOCKING — Training recipe parity is not pinned tightly enough.**

   [Plan line 111](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:111) does not explicitly name several load-bearing B-F/P1 arguments. In particular, current [defaults.ini line 30](/home/yixunhu/codespace/FLAC/defaults.ini:30) says strategy `auto`, while the historical B-F/P1 launchers explicitly used `ddp_find_unused_parameters_true`.

   The exact manifest must pin:

   - train dataset and frozen VAE path;
   - no resume/pretrained checkpoint;
   - `--batch-size 32 --num-gpus 2 --accum-batches 1`;
   - `--strategy ddp_find_unused_parameters_true --sync-batchnorm true`;
   - `--precision bf16-mixed --num-workers 6 --seed 42`;
   - max steps/checkpoint cadence, offline DINO pin, and environment versions.

   Use the reviewed B-F/P1 launcher as the recipe source and transplant only generic modern safety gates from `dtail_launch.sh`; the latter is fundamentally a resume-and-retune launcher.

   Add a seeded initialization checksum proving BTF, BF, and BVp1 instantiate identical learned parameters, not merely identical trainable parameter names.

9. **BLOCKING — The experiment cannot causally isolate the pose-representation defect.**

   Canonical-XYZ versus B-F changes two mechanisms simultaneously:

   - cylindrical-to-Cartesian pose representation;
   - C4 feature averaging to a single geometry pass.

   This bundled arm is exactly what Yixun requested, so no extra arm is required now. But [the hypothesis and success wording](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:20) must not claim that an improvement identifies the pose defect. It estimates the complete Canonical-XYZ package. A null result also cannot rule out either mechanism because their effects could cancel.

10. **BLOCKING — The comparator and success rules could turn checkpoint noise into a scientific verdict.**

   B-F@40k is a valid fixed-step comparator, but exp_10 explicitly established it as a band-best spike. Its K=8 row is `8.202/0.9778/38.793/R@1 5.387`, while P1@40k is `8.993/1.0093/40.650/5.173`; that is not the same finding as exp_11’s later C4L-versus-VANL reversal. Exp_11 is contextual mechanism evidence under a different recipe/chunk plan, not a direct exp_20 comparator.

   [The “majority of four metrics” rule](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:145) has no minimum effect or uncertainty threshold, excludes FD/R@5/R@10, and could pass on three negligible improvements plus one large regression. Pre-register:

   - paired seed-42–46 deltas and their uncertainty;
   - which metrics are primary and why, while still reporting all six table metrics;
   - explicit invariance tolerances;
   - either near-endpoint checkpoint screens/trajectory context or language limiting the conclusion strictly to the 40k checkpoint.

   State prominently that five evaluation seeds quantify sampling variability only; there is still one training seed per arm.

11. **BLOCKING — The planned tests do not fully exercise the central geometry claim or old-path preservation.**

   `yaw_transform_consistency` proves only that depth remains a valid panorama; the DistEmbedder test covers only poses. Add a GeometryConditioner-level test of canonicalized `q_xyz − P_depth,xyz`, or a full lightweight conditioner-output invariance test, at 45° and C4 angles.

   Also add:

   - target-frame invariance under normal reference, context fallback, and fail-closed all-degenerate input;
   - single conditioner call/no FA call/no orbit;
   - scalar conversion and per-sample width validation;
   - exact output-name and provenance-schema tests;
   - target training-config rejection of stray orbit keys;
   - regression coverage for `test_yaw_random_eval.py`, `test_exp14_fixed_mode_snapshot.py`, and `test_frame_avg_cap_config.py`, not only the subset listed at [plan line 130](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_target_frame_claude/plan_bf_target_frame.md:130).

## Nits

- **NIT — Dispatch inventory:** `finetune_cond.py:35/76/469` is another production cond-method whitelist. HAA finetuning is out of scope, so do not extend it now, but enumerate it as intentionally unchanged and state that it must continue rejecting `target_frame` until the later HAA round.
- **NIT — Chunk-plan disclosure:** explicitly list `target_frame: N/A—no orbit`, `P1: N/A—no orbit`, and `B-F@40k: legacy per-angle C4`. This satisfies announcement 06’s cross-method disclosure.
- **NIT — Terminology:** 45° is no longer a “negative control” for target-frame conditioning. It is an off-C4 discriminator expected to remain flat for target-frame and break for historical C4 FA.
- **NIT — ETA arithmetic:** 15 evaluations at 6.5 minutes each over two GPUs is roughly one hour plus overhead, not 3–4 hours. Recalculate after the smoke/rate probe.

After these revisions, the proposed quantized target-frame method should be approvable without adding the Cyl-PE arm, HAA work, or a GPU launch decision.