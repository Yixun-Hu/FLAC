# Codex plan review — exp_04_warmup_unblock

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** `plan_warmup_unblock.md` (scaffold commit)

**Verdict: APPROVE-WITH-CHANGES**

1. **Medium: add a pre-registered null control before interpreting W1 failure.**  
   The Adam-transient hypothesis is consistent with exp_03: R@1 recovery under R1b says the model did not globally collapse; T60/EDT/C50 remaining 6–42σ off points to sharper decay-envelope directions; batch parity partially helping is compatible with early-moment noise; and lr being far below the original final lr does not rule out Adam’s early normalized, sign-like updates.  
   But W1 failure alone does not distinguish “optimizer transient destroys” from “any train loop/export/training-mode buffer update destroys.” An lr=0 run is not trivial here because the trainable ResNet conditioner has BatchNorm running buffers, which can mutate even with zero optimizer updates. Add a `W0` or `W1-null`: R1b recipe, `--lr 0`, same 625 optimizer steps, full-split eval. If it passes, training-loop/BN/export alone are not sufficient; if it fails, warmup is no longer the right next attribution. A `5e-8` arm is secondary; lr=0 is the sharper first null.

2. **Medium: current tests do not catch the accumulation warmup bug class.**  
   The planned fake-trainer tests only verify selected `global_step` values. They would not fail if the implementation used an internal callback counter or `batch_idx`, which under `accumulate_grad_batches=32` would make warmup effectively micro-batch based. Add a test that simulates 32 repeated `on_train_batch_start` calls at `trainer.global_step == 0`, then 32 calls at `global_step == 1`, and asserts the LR remains `target * 1/200` for the first accumulation group and `target * 2/200` for the second. Alternatively implement the callback on `on_before_optimizer_step`, which Lightning calls once per optimizer step.

3. **Low/medium: the 200-step warmup has an integrated-LR confound.**  
   The plan correctly notes ~16% less full-lr-equivalent exposure: `200 * average(1..200)/200 + 425 = 525.5` vs 625 full-lr steps. That is acceptable for a repair recipe, but W1 PASS should be interpreted as “warmup/lower early LR repairs the control,” not uniquely “Adam second-moment transient proven.” A matched-area constant-lr run would be the cleaner mechanism discriminator, but I would not require it before W1.

4. **Low: pre-register the ambiguous stop/marginal language.**  
   The plan says a 2σ gate, then “passes only marginally (2–3σ).” Anything above 2σ is a fail under the stated gate. Define marginal as, for example, “all primary metrics ≤2σ but at least one ≥1.5σ,” and state whether W2 launches or pauses.

**Lightning Warmup-Step Semantics**

Verified against the pinned local `pytorch_lightning==2.1.0` package in [pyproject.toml](/home/yixunhu/codespace/FLAC/pyproject.toml:20):

- `trainer.global_step` is optimizer steps, not micro-batches: [training_epoch_loop.py](/home/yixunhu/miniconda3/envs/rir2rir/lib/python3.10/site-packages/pytorch_lightning/loops/training_epoch_loop.py:96).
- `on_train_batch_start` fires per train batch/micro-batch: [training_epoch_loop.py](/home/yixunhu/miniconda3/envs/rir2rir/lib/python3.10/site-packages/pytorch_lightning/loops/training_epoch_loop.py:223).
- accumulation boundary is determined by `ready % accumulate_grad_batches == 0`: [training_epoch_loop.py](/home/yixunhu/miniconda3/envs/rir2rir/lib/python3.10/site-packages/pytorch_lightning/loops/training_epoch_loop.py:315).
- `on_before_optimizer_step` is available and called before `optimizer.step()`: [precision_plugin.py](/home/yixunhu/miniconda3/envs/rir2rir/lib/python3.10/site-packages/pytorch_lightning/plugins/precision/precision_plugin.py:82).

So the planned `on_train_batch_start` callback is **not 32x too long if it keys only off `trainer.global_step`**. It will redundantly write the same LR for all 32 micro-batches in an optimizer step, then advance after the optimizer step increments `global_step`.

**Single Most Valuable Change**

Add the lr=0 null control as a pre-registered W1-stage arm or conditional-on-W1-fail arm. It is the cheapest way to prevent a misleading conclusion if the warmup control fails.
---
**Disposition (Fable 5):** All four findings adopted in the plan revision: W0 lr=0 null control added (conditional on W1 FAIL — saves 3 h in the PASS path; BN-buffer subtlety noted); accumulation-semantics test added to the TDD list; W1-PASS interpretation language corrected (integrated-lr confound); marginal pass defined as all ≤2σ with any ≥1.5σ ⇒ pause. Lightning global_step semantics verified by reviewer against installed 2.1.0 source.
