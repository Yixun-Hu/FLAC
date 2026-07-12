# Codex code review — exp_03, round: evalwire (TDD cycle 5)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `8e6164a` (RED) + `337eec3` (GREEN 11a) + `1de5721` (GREEN 11b)

**Verdict: REQUEST-CHANGES**

1. **Medium: meta guard still allows wrong-method / wrong-angle comparisons.**  
   [compare_predictions.py](/home/yixunhu/codespace/FLAC/worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py:152) only guards `dataset_config`, `seed`, and `batch_size`, while the sidecar records `cond_method` and `frame_avg_angles` at [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:268). A ref/alt pair with the same dataset/seed/batch but different conditioning method or frame-average angle set will pass silently. Also, `dataset_config` is only the path string at [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:265), not resolved content/hash. For R4 evidence, guard at least `cond_method` and `frame_avg_angles` equality, with `rotate_deg` intentionally allowed to differ.

2. **Low: `evaluate_model()` silently treats unknown programmatic `cond_method` values as vanilla.**  
   The CLI is protected by `choices`, but direct callers hit the `else` branch at [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:182) and run vanilla while filenames/meta record the unknown method. Add function-level validation near [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:81).

3. **Low: tests do not assert the save-path wiring.**  
   [test_eval_paths.py](/home/yixunhu/codespace/FLAC/src/tests/test_eval_paths.py:46) tests `build_output_paths()` directly, but no test proves `evaluate_model()` actually uses it. The current implementation does use it at [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:245) and [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:261), but a future regression could leave the pure helper green while the save path drifts.

**Focus Answers**

1. **Legacy regression:** vanilla + rot0 numeric behavior looks unchanged by inspection. Rotation is skipped, the vanilla conditioner call remains inside the same autocast block, and noise/sampling/metric accumulation order is unchanged. Metrics JSON gains extra keys; prediction storage intentionally changes to the sidecar dict.

2. **Composition semantics:** correct. `--rotate-deg` is applied first at [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:169), then `invariant_conditioning()` runs. Inside it, cylindrical pose features are computed from the rotated metadata, and only ViT/depth keys get the C4 frame rotations. No double-application on the invariant pose path.

3. **Output path collisions:** yes, custom angle sets with the same count collide (`_a4` for both `0,90,180,270` and `45,135,225,315`). This is not a real risk for the planned R4 45-degree probe, because the plan runs `--rotate-deg 45` with the default frame angles, producing a distinct `_rot45` suffix.

6. **Test quality:** no wiring assertion exists. This is finding 3.

Comparator legacy interop is preserved: bare tensors still load, and single-sided meta only warns. The ValueError message for guarded mismatches is actionable, but the guarded field set is too narrow for the headline evidence chain.

I did not run tests in this read-only review.

Safe to proceed to round 6 (finetune script)? **No, fix the guard/wiring issues first.**
---
**Disposition (Fable 5):** All three findings accepted and dispatched to the Coder: (1) widen the meta guard to cond_method + frame_avg_angles (rotate_deg exempt by design); (2) function-level cond_method validation in evaluate_model; (3) wiring test that evaluate_model's save path flows through build_output_paths. Cycle 6 held until green + re-verified.
