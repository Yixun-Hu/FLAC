# Codex code review — exp_05, round: freezebn

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** commits `d1c4e5c` (RED) + `5d1c64c` (GREEN)

**Verdict**
APPROVE-WITH-NITS

**Findings**
No blocking findings.

Nit: [src/tests/test_finetune_cond.py:510](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:510) correctly pins the `on_fit_start`-only hazard, but it is still a synthetic hook-order test, not a real `Trainer.fit` integration test. It would fail an implementation that only evals in `on_fit_start` because the test calls `net.train()` before `on_train_batch_start`. Negative control at [src/tests/test_finetune_cond.py:534](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:534) is sound. Minor gap only: tests do not exercise BN1d/BN3d, though implementation covers them.

**Focus 1**
Enforcement is complete for the paths that matter. Installed PL is 2.1.0. `Trainer` calls `on_fit_start` before restore/training run, then `_FitLoop.reset()` calls `trainer.model.train()` at PL `fit_loop.py:300-304`; the callback re-asserts at [finetune_cond.py:212](/home/yixunhu/codespace/FLAC/finetune_cond.py:212). PL `training_epoch_loop.py:223-240` calls callback `on_train_batch_start` before the automatic optimization path reaches `training_step` (`automatic.py:315`). Mid-epoch validation returns through `on_validation_model_train` (`evaluation_loop.py:269-270`, `core/hooks.py:164-166`), but the next training forward is still preceded by the next `on_train_batch_start`. Resume restores state before the fit loop runs; no training forward occurs before the batch hook. I see no one-batch BN-mutation gap.

**Focus 2**
Scope is acceptable. `getattr(pl_module, "diffusion", pl_module)` at [finetune_cond.py:193](/home/yixunhu/codespace/FLAC/finetune_cond.py:193) walks the real `DiffusionCondTrainingWrapper.diffusion` subtree. Current FLAC config’s `context_audio` RIR conditioner is torchvision ResNet18, giving the expected 20 `BatchNorm2d` modules. Local DiT/transformer code uses LayerNorm/GroupNorm, and installed `dinov3_vit` uses LayerNorm only. The VAE pretransform is under `diffusion.pretransform`, but current Oobleck/VAE stack has no BatchNorm; even if it did, it is already frozen/eval and `training_step` calls `pretransform.train(False)`, so FreezeBN eval would be harmless/redundant.

Warmup interaction is fine: `WarmupLR` and `FreezeBN` both fire on `on_train_batch_start`, touch disjoint state, and are idempotent.

I did not rerun tests because this read-only sandbox has no writable temp directory for the Python import path.

Safe to launch V1'? **Yes.**
---
**Disposition (Fable 5):** Round freezebn CLOSED (no blocking findings; nits noted — BN1d/3d test coverage batched to any future round). V1′ launches with --freeze-bn only (no warmup: one-variable discipline; W1 proved warmup a no-op).
