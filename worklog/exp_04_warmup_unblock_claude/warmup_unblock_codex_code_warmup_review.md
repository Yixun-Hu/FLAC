# Codex code review — exp_04, round: warmup

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-05
**Target:** commits `a9b5408` (RED) + `0081ee2` (GREEN)

**Verdict: APPROVE-WITH-NITS**

1. Low: [finetune_cond.py:169](/home/yixunhu/codespace/FLAC/finetune_cond.py:169) changes the default-off recipe echo: `warmup_steps=0` is now inserted between `lr` and `use_ema`. Training behavior is unchanged, and I found no in-repo grep script depending on the old exact substring, but external greps for `lr=5e-06 use_ema=False` would break.

2. Low: [src/tests/test_finetune_cond.py:376](/home/yixunhu/codespace/FLAC/src/tests/test_finetune_cond.py:376) only tests one optimizer. A wrong implementation that updates only `trainer.optimizers[0]` would pass the five new tests. The shipped implementation is correct at [finetune_cond.py:157](/home/yixunhu/codespace/FLAC/finetune_cond.py:157): it iterates all `trainer.optimizers` and all param groups.

`trainer.optimizers` answer: yes, it is the right Lightning 2.1.0 callback accessor. In the installed 2.1.0 source, `Trainer.optimizers` forwards to `strategy.optimizers`, strategy setup populates it before training, and the training loop itself uses `trainer.optimizers[0]` after `on_train_batch_start`. `trainer.global_step` is explicitly optimizer steps, so the warmup is not micro-batch keyed.

Other checks: step 0 factor `1/N` is deliberate and avoids wasting the first optimizer step at lr 0; `warmup_steps<=0` returns/behaves as off; scheduler removal means no other LR writer; the `ValueError` is programmatic-only because CLI `main()` always passes `lr`.

I did not rerun pytest because the read-only sandbox has no writable tempdir and importing Lightning fails during tempdir creation.

Safe to launch W1? Yes.
---
**Disposition (Fable 5):** Round warmup CLOSED (no blocking findings). Nit 1 accepted (echo format documented here for future greps). Nit 2 (multi-optimizer test) batched to the next code round if any. trainer.optimizers accessor confirmed correct — plus a runtime guard added to the W1 probe acceptance: first-10-step train/lr must read ~2.5e-8–2.5e-7 (warmup engaged), not 5e-6.
