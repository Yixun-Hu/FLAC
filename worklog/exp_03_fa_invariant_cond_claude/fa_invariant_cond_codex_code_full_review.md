# Codex code review — exp_03, round: full (integrative launch gate)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** full exp_03 diff (0dce4ce..88f69b8) + run plan §5

**Verdict: GO-WITH-CONDITIONS** for launching R0/R1.

**Findings**

1. **High: train/eval autocast dtype is not the same.**  
   R1/R2 fine-tune defaults to `bf16-mixed` in [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:302), while eval uses `torch.amp.autocast(device)` with no dtype in [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:182). In this environment, CUDA autocast default is `torch.float16`. That means a `fa_invariant` model trained on bf16-conditioned tensors may be evaluated on fp16-conditioned tensors. This violates the “exact same conditioning distribution” requirement.

2. **High: eval checkpoint loading can silently partial-load.**  
   [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:124) calls `model.load_state_dict(state_dict, strict=False)` and discards `missing/unexpected`. The final `export_model()` artifact is bare-keyed and should load cleanly, but there is no assertion or log to prove it. I verified current `FLAC_EMA.ckpt` and existing exported fine-tune artifacts are 1066 bare keys, but R1/R2 metrics should not depend on manual inspection.

3. **Medium: planned 10k fine-tune can accidentally run 2k steps.**  
   Plan §5 says R1/R2 are 10k steps, but [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:298) defaults `--max-steps 2000`. The launch command must explicitly pass `--max-steps 10000`.

4. **Medium: batch-8 `fa_invariant` fit is not proven.**  
   The worklog smoke only proves `fa_invariant` batch 2. With K=8, `fa_invariant` runs 4 frame angles × (`source_vit` + 8 context ViT passes), i.e. 36 DINO forwards per batch. Before full R2, run a storage-light batch-size probe at the planned batch size; for R1, still probe batch 8 once before the 10k run.

5. **Low: frame-angle defaults match but are duplicated.**  
   Code source of truth is [yaw_rotation.py](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:23), but CLI defaults duplicate `"0,90,180,270"` in [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:303) and [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:296). Safe for planned defaults, but easy to drift.

**Answers**

1. **Train/eval consistency:** metadata order is correct: train does raw metadata → `invariant_conditioning`; eval does optional `--rotate-deg` first, then `invariant_conditioning`. Defaults use the same numeric angles. Autocast is not consistent: bf16 training vs fp16 eval unless fixed.

2. **Checkpoint round-trip:** final exports are bare `model.* / conditioner.* / pretransform.*` keys and eval’s prefix/EMA handling is compatible. But eval has no load-integrity assertion/log; add `missing, unexpected = ...` and require zero for final exports.

3. **CFG:** training computes conditioning first, then `cfg_dropout` is applied inside DiT, so dropout sees symmetrized conditioning. Eval uses `cfg_scale=1.0`, so the unconditional CFG branch is bypassed despite `batch_cfg=True`. No raw-conditioning negative branch is used in planned evals.

4. **K=8 shapes:** no K=1-only assumption found. The helper operates over the last pose dimension, `only_ids` handles `context_poses_vit`, tests cover multi-context tensors, and the real-stack rung included K=8.

5. **Run-plan soundness:** flags exist, dataset config paths exist, and full-split configs are used. Wall-clock is likely not trivial: roughly 10-12 min per vanilla full eval from exp_01/02, `fa_invariant` evals likely higher, and two 10k fine-tunes plus ~22-28 full evals is likely an overnight-to-day run on one A6000. Do not trim R1 seeds; the 5-seed gate is needed for the exp_01 noise-floor comparison. Stage R2/R3 only after R1 passes.

**Conditions Before Launch**

1. Align eval autocast with the planned fine-tune precision, preferably explicit bf16 eval autocast for conditioning, or change the fine-tune precision and document the recipe deviation.
2. Add eval load assertions/logging for `missing/unexpected`; final exported checkpoints must load with `missing=0, unexpected=0`.
3. Write exact R0/R1 command artifacts before launch, including `--max-steps 10000` for R1.
4. Run a storage-light batch-8 fit probe before the 10k R1 run; repeat for R2 `fa_invariant` before launching it.
---
**Disposition (Fable 5):** GO-WITH-CONDITIONS accepted; launch held until all four conditions land. C1: scoped fix — eval gains --cond-autocast {default,bf16,off}; fa_invariant evals (R0/R2/R3/R4) run bf16 to match training exactly; vanilla evals keep the fp16-default for exp_01/exp_02 protocol continuity (R1 gate compares like-for-like). Metric-1 noise floor will be re-measured under bf16 and pre-registered before R4. C2: load-integrity assertion (missing/unexpected logged; hard-fail unless --allow-partial-load). C3: command artifacts will carry --max-steps 10000 explicitly. C4: storage-light batch-8 fit probes before R1 and before R2. Finding 5 (CLI default duplication) batched as a nit.
