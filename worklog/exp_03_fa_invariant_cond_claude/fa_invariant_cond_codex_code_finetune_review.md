# Codex code review — exp_03, round: finetune (TDD cycle 6)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `6d94a45` (RED) + `bd03a5c` (GREEN)

**Verdict: REQUEST-CHANGES**

1. **Medium: default gradient clipping is an undocumented recipe drift.**  
   [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:281) defaults `--gradient-clip-val` to `1.0`, while the repo training default is `0.0` in [defaults.ini](/home/yixunhu/codespace/FLAC/defaults.ini:72) and is passed through by [train.py](/home/yixunhu/codespace/FLAC/train.py:163). The approved recipe deviations are LR/scheduler, EMA off, and method injection. Unless launch commands override `--gradient-clip-val 0.0`, R1/R2 are not recipe-parity controls.

2. **Medium: `--smoke` can still get a default Lightning `ModelCheckpoint`.**  
   The smoke branch removes the explicit checkpoint callback at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:226), but the `Trainer` call at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:235) leaves `enable_checkpointing` at Lightning’s default. Installed Lightning injects `ModelCheckpoint()` when checkpointing is enabled and none is supplied. Set `enable_checkpointing=not smoke` to make the “no checkpointing, no export” smoke contract true.

3. **Low: tests don’t fully pin the recipe or scheduler absence.**  
   [test_finetune_cond.py](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:68) checks the scheduler key is removed and manually builds an optimizer, but it never exercises `DiffusionCondTrainingWrapper.configure_optimizers()` or `Trainer` scheduler state. Also [test_finetune_cond.py](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:111) ignores the training block broadly, so dropping `timestep_sampler`, `cfg_dropout_prob`, betas, or `weight_decay` would not be permanently caught.

**Focus 1-3**

1. `build_finetune_training_config` itself is recipe-faithful by inspection: it deep-copies, injects `cond_method`/`frame_avg_angles`, forces `use_ema=False`, changes LR, and removes only `scheduler`. It preserves `timestep_sampler=log_snr`, `cfg_dropout_prob=0.1`, `mask_padding=true`, `mask_padding_dropout=0.0`, AdamW betas/weight_decay, metrics, and all non-training blocks. No nested aliasing issue found.

2. `FLAC_EMA.ckpt` is bare-keyed: 0 `diffusion.*`, 0 `diffusion_ema.ema_model.*`, with `model.*`, `conditioner.*`, and `pretransform.*`. So the strip/remap branches are no-ops for the actual release checkpoint, and `missing=0/unexpected=0` is exactly expected. The EMA-remap branch is only active for PL-wrapper checkpoints and maps EMA DiT weights onto `model.*`; it does not corrupt the bare path. Optional `--pretransform-ckpt-path` mirrors `train.py`.

3. Freezing/trainability looks correct: VAE gets both `enable_grad=False` and `requires_grad_(False)` at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:213). The conditioner is trainable because `freeze=false` in `FLAC_AR.json` and the script only freezes `pretransform`. The reported 50.3M trainable / 14.2M frozen split matches DiT + unique conditioner params including DINOv3-S, with the VAE frozen.

I did not rerun pytest here; the read-only sandbox lacks a usable temp dir for Lightning/torchmetrics imports. Checkpoint-key inspection was run directly with `torch.load`.

Safe to run the parity audit and launch R0/R1? **No for launch under SOP: parity audit evidence is clean, but fix the gradient-clip drift and smoke checkpointing first.**
---
**Disposition (Fable 5):** All three accepted. Finding 1 originated in the Planner's brief (1.0 carried from the archived script; upstream default is 0.0 per defaults.ini:72) — recipe-parity requires 0.0 default; noted that the config-level parity audit cannot see Trainer args, so the fix adds it to the pinned surface. Finding 2 (Lightning default ModelCheckpoint under --smoke) and finding 3 (configure_optimizers + four-keys-only preservation test) dispatched with it. Launch held until green.
