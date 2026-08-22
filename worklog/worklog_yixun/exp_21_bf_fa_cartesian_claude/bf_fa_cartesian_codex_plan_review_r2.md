**Reviewer:** OpenAI Codex (gpt-5.6-sol, `codex exec`, read-only sandbox, reasoning=xhigh) · **Date:** 2026-08-21 (round 2)

Verdict: **REVISE**. The redirected method is implementable with the existing orbit executor, and the central `only_ids`, ViT-pinning, and cap-32 arithmetic claims are sound. Five plan-level blockers remain, principally in recipe parity and evaluation integrity.

## Confirmed code-path answers

### a) `present=POSE_KEYS` is supported

Yes, for the pinned B-F/BFC conditioner configuration.

- `MultiConditioner.only_ids` filters by conditioner **id**, not by conditioner type, so it supports both DistEmbedder ids and Geometry ids ([conditioners.py:367](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:367), [conditioners.py:374](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:374)). The actual ids match `POSE_KEYS`: `source`, `source_vit`, `context_poses`, `context_poses_vit` ([FLAC_AR_BF.json:65](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json:65)).
- The source/context DistEmbedders share one projection as claimed ([conditioners.py:594](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:594)); repeated orbit calls accumulate gradients through that same module correctly.
- `_orbit_average_batched` is role-agnostic: it accumulates every `present[i][0]` identically ([yaw_rotation.py:547](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:547), [yaw_rotation.py:568](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:568)). Therefore global ids `source`/`source_vit` and cross-attention ids `context_poses`/`context_poses_vit` are all averaged before downstream global concatenation or cross-attention concatenation ([models/diffusion.py:145](/home/yixunhu/codespace/FLAC/src/models/diffusion.py:145), [models/diffusion.py:159](/home/yixunhu/codespace/FLAC/src/models/diffusion.py:159)).
- Keeping the base mask is correct for `context_poses`: `DistEmbedderConditioner` always returns an all-ones `[B,1]` mask ([conditioners.py:350](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:350)), and yaw rotation changes neither context count nor validity. The current continuous transformer does not consume cross-attention masks anyway ([dit.py:203](/home/yixunhu/codespace/FLAC/src/models/dit.py:203)).

Add an explicit test that every averaged id retains the exact base mask, because loop-vs-batched tensor tests alone would not detect mask replacement.

### b) T2 is valid for the pinned configuration

The base-pass difference does not affect the ViT inputs:

- `cylindrical_pose_features` replaces only `source` and `context_poses`; it leaves `depth` and both `*_vit` keys untouched ([yaw_rotation.py:209](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:209), [yaw_rotation.py:269](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:269)).
- A `GeometryConditioner` reads its own `*_vit` pose plus `depth`, not the DistEmbedder pose key ([conditioners.py:388](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:388), [conditioners.py:284](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:284)).
- In the orbit passes, fa_cartesian rotates additional DistEmbedder poses, but the `*_vit` poses and depth seen by both methods remain identical. Extra DistEmbedder calls consume no RNG.

Thus eval-mode ViT outputs should be numerically equal. T2 should say `allclose` with a tolerance, not require bit equality on GPU. It proves the conditioner-level ViT branch is pinned; it does not imply identical learned ViT weights after training, because the deliberately changed pose conditioning changes the downstream loss and hence training gradients.

### c) D5 arithmetic is correct

At per-rank micro-batch 32 and cap 32:

\[
\text{angles\_per\_chunk}=\max(1,32//32)=1.
\]

The three nonzero angles therefore make three separate `MultiConditioner` calls, plus the angle-zero base call—matching the legacy loop ([yaw_rotation.py:555](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:555), [yaw_rotation.py:562](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:562)). Training uses `args.batch_size` directly per DDP rank ([train.py:112](/home/yixunhu/codespace/FLAC/train.py:112)), and the AR train loader defaults to `drop_last=True` ([dataset.py:456](/home/yixunhu/codespace/FLAC/src/data/dataset.py:456)), so there is no short training tail batch that would change the partition.

There is no residual DINO RNG-schedule difference: the relevant input shapes, DINO call order, and call count match legacy; the added DistEmbedder calls are deterministic. More precisely, the draw is per Geometry/DINO coordinate call, not literally one per angle. At K=8 the legacy/cap-32 schedule has `4 × (1 source + 8 context) = 36` DINO calls/draws per conditioning step. The plan should use that precise wording.

## Blocking findings

1. **BLOCKING — The launcher adds validation that B-F did not run, breaking recipe parity.**

   [Plan line 90](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:90) pins an AR seen-eval validation configuration. The historical B-F launcher passed only the training dataset and no `--val-dataset-config` ([bf_scratch_launch.sh:88](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh:88)).

   This is not merely extra logging. `trainer.fit` receives the validation loader ([train.py:230](/home/yixunhu/codespace/FLAC/train.py:230)), and every validation batch draws random noise ([diffusion.py:797](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:797)). Epoch validation can therefore advance RNG state and change subsequent training timesteps/noise, besides changing wall time. Remove the validation loader to reproduce B-F/P1, or stop calling the experiment single-delta.

2. **BLOCKING — The registered evaluation invocation is neither executable nor overwrite-safe.**

   The displayed command at [plan lines 126–133](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:126):

   - uses an inline K-selection comment instead of two exact commands;
   - lacks `--eval-name`;
   - omits `--record-per-scene`, despite claiming it is on;
   - leaves the grid’s `--record-stream --expected-stream-count 6337` only in prose;
   - omits the evaluation cap.

   `build_output_paths` does not include seed or dataset/K automatically ([eval_FLAC.py:373](/home/yixunhu/codespace/FLAC/eval_FLAC.py:373)). Without a unique `--eval-name` containing K and seed, all five seeds and both K values overwrite the same file.

   Rev 3 must pin two executable registered templates—or a reviewed loop—with unique K/seed names and every flag, including `--frame-avg-max-fwd-samples 64` and `--record-per-scene`. The grid invocation must separately show the stream/count flags.

3. **BLOCKING — Training cap 32 and evaluation cap 64 are conflated.**

   The training choice cap=32 is correct. It cannot be used with the registered eval batch of 64: `_orbit_average_batched` explicitly raises when `batch > cap` ([yaw_rotation.py:555](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:555)). Therefore [plan line 137](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:137), “BFC: batched, cap per D5,” is operationally wrong for evaluation, and [line 147](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:147) is wrong that either value is usable at the pinned eval batch.

   Declare separately:

   - training: micro 32, cap 32, one nonzero angle per chunk;
   - registered/grid eval: batch 64, cap 64, explicitly passed and required by admission validation.

4. **BLOCKING — The proposed paired comparisons mix evaluator pins/executors, and the aggregation admission contract is incomplete.**

   BFC would be evaluated with the current batched implementation, while the named B-F comparator is a legacy-loop row ([plan line 137](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:137)). The living table explicitly says legacy-loop and batched rows are not interchangeable ([model_comparison.md:6](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/model_comparison.md:6)). Calling historical BFC−BF/P1 differences “paired per-seed deltas” also assumes evaluator/data-context parity that different `source_sha` pins do not establish.

   Re-evaluate B-F@40k and P1@40k at the same current evaluator pin, five seeds and both K, or demote historical comparisons to contextual—not paired—comparators.

   The new-row validator at [plan line 94](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:94) must additionally reject wrong `cfg_scale`, diffusion `steps`, `rotate_deg`, eval cap, K-specific dataset path, duplicate/missing seeds, and checkpoint identity.

   Also distinguish the counts correctly: the split has 6,337 items and 17 room instances, but `--record-per-scene` groups on `md["scene"] = scene_name` ([AR_md.py:23](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:23)), producing ten room-family groups, not 17 room groups. The generator’s established scene-routed convention likewise expects ten families ([gen_model_comparison.py:787](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:787)). Do not validate `scene_count == 17`; validate exact dataset path, `n_samples == 6337`, and ten family keys. Keep the BFC table row on the flat top-level metrics, as planned.

5. **BLOCKING — The invariance-grid pass criterion is loose enough for incorrect conditioning to pass.**

   [Plan line 136](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:136) accepts a C4 metric spread as large as the five-seed sampling range. That tests whether rotation error is hidden beneath sampling variability, not whether the implementation is C4 invariant. A wrong pose-dispatch path could plausibly pass.

   Pre-register absolute per-metric C4 limits based on same-seed paired execution or the existing B-F C4 floor; exp_10 observed approximately `T60 0.0009 / C50 0.0001 / EDT 0.0011`, with retrieval spreads of only a few hundredths ([fa_scratch_resume_results.md:23](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_results.md:23)). The 45° cell should be required to exceed the C4 tolerance on a predeclared metric/conditioning statistic, not merely be investigated subjectively afterward.

## Nits

- **NIT — Dispatch inventory wording.** The production path is complete once the new helper imports are added at [diffusion.py:22](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:22) and [eval_FLAC.py:61](/home/yixunhu/codespace/FLAC/eval_FLAC.py:61). `eval_pl.py` will inherit the factory/wrapper dispatch but lacks registered-eval provenance and should be explicitly forbidden for headline rows. `unwrap_model.py` still uses upstream `stable_audio_tools` and does not preserve a conditioning policy; export is out of scope. `baselines/eval_baselines.py` evaluates non-FLAC baselines and needs no change. `finetune_cond.py` is correctly excluded and should continue rejecting `fa_cartesian`.
- **NIT — Test precision.** T2 should use `allclose`; T6 should explicitly state batch 32 so caps 32/64 exercise different chunk partitions; add exact base-mask preservation and assert `set(base) ⊇ POSE_KEYS`.
- **NIT — Comparison scope.** BFC/B-F are approximately matched-compute, but BFC/P1 are matched steps/samples, not compute; historical FA costs about 3.5× per step ([exp_10 worklog:71](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/fa_scratch_resume_worklog.md:71)). State that alongside the P1 comparison.
- **NIT — Cell arithmetic and unavailable screen.** Ten registered cells plus five grid invocations is 15 commands, with the K8/s42/0° cell reusable, so 14 unique cells—not 16 ([plan line 143](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:143)). A 42.5k screen cannot exist if training stops at 40k ([plan line 138](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:138)).

## Round-1 finding mapping

- Findings **1, 2** are moot/resolved by the redirect.
- Finding **4** is only partly moot: the reference-azimuth/scalar issue disappeared, but its depth/width fail-closed requirement was correctly carried into [plan line 36](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_20_bf_fa_cartesian_claude/plan_bf_fa_cartesian.md:36).
- Findings **3, 7, 9** are correctly remapped: C4-only invariance, populated orbit provenance, and a genuine pose-branch-only mechanism change.
- Findings **10 and 11** are substantively carried.
- Findings **5, 6, 8** appear in Rev 2, but are not yet closed because of blockers 1–4 above.

No method redesign is required. After those protocol and recipe corrections, the Cartesian C4 arm itself is approvable.
