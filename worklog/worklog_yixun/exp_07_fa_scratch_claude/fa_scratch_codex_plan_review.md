# Codex plan review — exp_07_fa_scratch

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-07

**Verdict: REQUEST-CHANGES**

1. **Blocking: the proposed 50k/100k budgets are not executable with current `train.py`.**  
   `train.py` hardcodes `max_steps=1000000` and exposes no CLI/config override [train.py](/home/yixunhu/codespace/FLAC/train.py:151). A 50k run would have to be manually killed after the 50k checkpoint, which is not an acceptable stop rule for a month-scale experiment. Add a tested `--max-steps` path before launch.

2. **Blocking: “original step count unknown” is false.**  
   The local released wrapper checkpoint `weights/FLAC/FLAC.ckpt` has `global_step=67500`, `epoch=14`, and includes EMA state. That is the cheapest probe of original training length. The budget table should be reframed around ~67.5k optimizer steps, not 100k/200k speculation.

3. **High: throughput math is numerically sane, but based on EMA-off fine-tunes.**  
   R1b: 20,000 micro-batches × bs4 = 80,000 samples in ~2h51-2h57 → ~7.5-7.8 samples/s shared, so ~10 samples/s free is plausible. FA probe: 0.58 micro-it/s × bs4 = 2.32 samples/s shared, ~0.3× vanilla, so ~3 samples/s free is plausible. But exp_07 wants EMA ON; exp_03/06 probes used `finetune_cond.py` with `use_ema=False`. README also says FLAC training with EMA requires H100 80GB. Add EMA-on fit/throughput probes before trusting the wall-clock table.

4. **High: “EMA + online both evaluated” needs an explicit eval protocol.**  
   Train.py checkpoints contain online `diffusion.*` plus `diffusion_ema.ema_model.*`; `eval_FLAC.py` will evaluate EMA by default when the eval config has `training.use_ema=true` [eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:193). Online evaluation needs either an eval config copy with `training.use_ema=false` or stripped EMA keys. Also, `train.py --val-every` only logs validation denoising losses, not full generated FLAC metrics or EMA metrics [diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:368). The 10k K=8 screens must be external `eval_FLAC.py` jobs.

5. **Medium: design is basically right, but preregistration is too loose.**  
   Two matched arms are the right design. EMA ON and no freeze-BN are also right for from-scratch: freeze-BN was a released-checkpoint fine-tune repair, not a from-scratch recipe. But define non-inferiority thresholds for “B-F ≈ B-V”, exact seed policy, exact decision rules for the hybrid branch, and which metrics are headline vs screening. Carry over exp_03 hardening: clean-load assertions, config diff/hash records, bf16 FA eval when matching training conditioning, and smoke/fit probes.

**Explicit Answer To 2**

The wall-clock arithmetic is right given the measured anchors: at effective batch 128, 10 samples/s gives ~7.4 days for 50k vanilla steps; 3 samples/s gives ~24.7 days for 50k FA steps. The 4× conditioner-cost story is consistent with the exp_03 FA probe because observed FA throughput was ~0.3× vanilla. The caveat is EMA ON and from-scratch memory/throughput have not been measured.

**Explicit Answer To 4**

The “no code expected” claim does not survive. Required TDD round: add `--max-steps` to `train.py`/`defaults.ini` and test that it reaches `pl.Trainer(max_steps=args.max_steps)`. Resume does not need new code: `--ckpt-path` already exists and is passed to `trainer.fit(..., ckpt_path=...)` [train.py](/home/yixunhu/codespace/FLAC/train.py:169). Seed does not need new code: `--seed` exists and is applied before model/dataloader creation [train.py](/home/yixunhu/codespace/FLAC/train.py:26). Config copies should be validated by JSON diffs plus wrapper instantiation assertions; online eval needs a documented `use_ema=false` eval config or a tested CLI selector.

**Single Most Valuable Change**

Revise the plan around the discovered 67,500-step released checkpoint target, and add the reviewed `--max-steps` control so the matched B-V/B-F runs can stop exactly at preregistered budgets.
---
**Disposition (Fable 5):** All findings adopted in the plan revision: budget re-anchored on the DISCOVERED 67,500-step training length from FLAC.ckpt (finding 2 — the review's most valuable contribution); --max-steps TDD round added as blocking pre-launch work (finding 1); EMA-on fit/throughput probe mandated before trusting wall-clock (finding 3); external-eval protocol with use_ema=false config copy for online weights (finding 4); non-inferiority thresholds, seed policy, headline-vs-screening split, and hybrid decision rule pre-registered (finding 5). Awaiting Yixun's budget decision + GPU-1 confirmation.
