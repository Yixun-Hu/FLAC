# Codex code review — exp_03, round: accum (launch adaptation)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commit `f472328`

**Verdict: APPROVE-WITH-NITS**

No blocking code findings in `f472328`.

Low launch nit: the tracked launch docs I saw are still stale: [fa_invariant_cond_command.md](/home/yixunhu/codespace/FLAC/worklog/exp_03_fa_invariant_cond_claude/fa_invariant_cond_command.md:20) still shows R1 with `--batch-size 8` and no `--accumulate-grad-batches 2`, and [fa_invariant_cond_params_set_up.md](/home/yixunhu/codespace/FLAC/worklog/exp_03_fa_invariant_cond_claude/fa_invariant_cond_params_set_up.md:16) still says batch size 8. The code supports the planner decision, but the command artifact must be updated before launch.

**Threading**
Correct. CLI parser adds default `1` at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:313), `main()` passes it at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:336), `finetune()` accepts it at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:168), and `build_trainer_kwargs()` passes `accumulate_grad_batches` into `pl.Trainer` kwargs at [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:129). Default `1` preserves prior behavior. No other live `finetune()` call sites found.

**Answer 2: Steps, Checkpoints, Pins**
Lightning is pinned to `pytorch_lightning==2.1.0` in [pyproject.toml](/home/yixunhu/codespace/FLAC/pyproject.toml:20). In Lightning 2.1, `global_step` is optimizer steps, not dataloader micro-batches: `epoch_loop.global_step` returns `automatic_optimization.optim_progress.optimizer_steps`. `max_steps` stops on that `global_step`, and `ModelCheckpoint(every_n_train_steps=...)` also checks `trainer.global_step % every_n_train_steps`.

So `--max-steps 10000 --accumulate-grad-batches 2` means 10,000 optimizer steps, about 20,000 micro-batches. `checkpoint_every=2500` fires every 2,500 optimizer steps. No silent 2x or 0.5x optimizer-length shift.

Recipe pins are not disturbed: accumulation is a Trainer arg, not a training-config key, and the flat-diff recipe pin remains scoped to the intended training config keys at [test_finetune_cond.py](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:278). Smoke keeps no-checkpoint behavior, and default accumulation remains `1`.

**Answer 3: Gradient Equivalence**
No sum-reduced loss component found. Training uses one `MSELoss` inside `MultiLoss` at [diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:111); `MSELoss` computes elementwise MSE, applies optional padding mask, then `.mean()` at [losses.py](/home/yixunhu/codespace/FLAC/src/training/losses/losses.py:73). Lightning then divides the returned loss by the accumulation factor internally.

Caveat: because `mask_padding=true` in [FLAC_AR.json](/home/yixunhu/codespace/FLAC/src/configs/model_configs/FLAC/AR/FLAC_AR.json:132), exact equivalence to a physical batch of 8 requires equal valid-token counts across the two micro-batches. If valid padding-mask counts differ, accumulation averages two micro-batch means instead of one all-valid-token batch mean. Statistically close, not bitwise/mathematically identical.

**Answer 4: BatchNorm Caveat**
Yes, there is trainable BatchNorm in the conditioner path: `RIRConditioner` uses torchvision `resnet18` at [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:37), and that ResNet has BN. The VAE is frozen, but the conditioner is trainable. `RIRConditioner` flattens `B * N` context audio before ResNet at [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:162), so with K=8 the BN sees 32 spectrograms for batch 4, versus 64 for batch 8. Accumulation does not merge BN statistics across micro-batches.

Severity: medium caveat for claiming exact batch-8 parity; low-to-medium for the experiment because R1 and R2 both use the same batch4/accum2 regimen, and R1 is the control gate that should catch a harmful recipe shift.

Safe to launch R1 with batch 4 accum 2? Yes, code-wise, provided the actual launch command uses `--batch-size 4 --accumulate-grad-batches 2`.
---
**Disposition (Fable 5):** Round accum CLOSED. Nit fixed in this commit (params/command updated to batch 4 × accum 2). Lightning-2.1 optimizer-step semantics confirmed (10000 = 10000 optimizer steps; checkpoints on optimizer steps). Two documented parity caveats accepted: padding-mask micro-batch averaging (statistical, not bitwise) and RIR-encoder BN statistics at 32 vs 64 spectrograms — both identical across R1/R2, and R1 gates against exp_01.
