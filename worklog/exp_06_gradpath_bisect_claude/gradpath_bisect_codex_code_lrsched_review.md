# Codex code review — exp_06, round: lrsched

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** commits `ed34c6c` (RED) + `7a6272c` (GREEN)

**Verdict: REQUEST-CHANGES**

**Finding**
- [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:374): `--lr-schedule inverse-restart` can still be combined with `--warmup-steps > 0`. That preserves the `InverseLR` scheduler, then [build_trainer_kwargs](/home/yixunhu/codespace/FLAC/finetune_cond.py:274) appends `WarmupLR`, whose [batch-start hook](/home/yixunhu/codespace/FLAC/finetune_cond.py:168) overwrites LR every micro-batch. Lightning then steps the scheduler after optimizer steps, but the next batch-start write wins before the next optimizer step. Require a fail-fast guard now: reject `lr_schedule == "inverse-restart" and warmup_steps > 0`, with a tiny test.

**Focus Answers**
- Focus 1: inverse-restart config path is correct. The only behavioral branch is the scheduler pop at [finetune_cond.py:99](/home/yixunhu/codespace/FLAC/finetune_cond.py:99); `inverse-restart` preserves the deep-copied scheduler. `configure_optimizers()` returns the scheduler branch with `interval: "step"` at [src/training/diffusion.py:195](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:195). In Lightning 2.1 local source, `global_step` is optimizer steps, and `interval="step"` schedulers are skipped while accumulating, so with `accumulate_grad_batches=32` this is per optimizer step, not per micro-batch.
- Focus 2: require the guard now; documented-unsupported is too easy to misuse silently.
- Focus 3: no active dependency found on `scheduler=constant(removed)`. Grep only found historical exp_03/04/05 logs; exp_06 worklog/tests use or tolerate `lr_schedule=...`.
- Focus 4: [test 3](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:662) asserting `base_lrs` is the right pin for “`--lr` feeds scheduler base,” because PyTorch steps once at scheduler construction and the live group LR is already warmed down. It does not prove restart/warmup-from-step-0. Add `sched.last_epoch == 0` and live LR / `get_last_lr()` approx `5e-8` for the test’s `5e-6` base.

I did not run pytest: this read-only sandbox has no usable temp dir and the local PL import path also trips on the dependency stack.

Safe to run L5? Not before the warmup/inverse guard lands; with `--warmup-steps 0`, the inverse scheduler cadence itself is correct.
---
**Disposition (Fable 5):** Both fixes dispatched (fail-fast guard + restart-semantics pin). Scheduler-cadence-under-accumulation confirmed correct — L5 unblocked once the guard lands.
