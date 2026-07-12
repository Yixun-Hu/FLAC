# Codex plan review — exp_06_gradpath_bisect

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06

**Verdict: APPROVE-WITH-CHANGES**

The plan is directionally sound for answering Yixun’s two questions, but it needs revision before launch. The main blocker is that L5 is not runnable with current code, so the “no new source code expected” claim is false.

1. **High: L5 requires code, with TDD.**  
   [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:38) always removes the scheduler via `diffusion_opt.pop("scheduler", None)`, and the CLI has no inverse-scheduler arm. Existing tests also pin constant-LR behavior. Minimal change: add a tested schedule selector, e.g. `--lr-schedule constant|inverse-restart`, preserving current default; for `inverse-restart`, keep/reinject `InverseLR(inv_gamma=1000000, power=0.5, warmup=0.99)` with `lr=5e-5`, `use_ema=False`, `--freeze-bn`, batch 4 × accum 32, 625 steps. Add tests for parser, config injection, `configure_optimizers()` returning the scheduler branch, and recipe echo/LR trace.

2. **Medium: S3 tail-mask mechanism is probably weak as written.**  
   `padding_mask` comes from target-audio `PadCrop_Normalized_T`, not context audio, and masks valid prefix / padded suffix only. Training loss uses it in MSE selection in [losses.py](/home/yixunhu/codespace/FLAC/src/training/losses/losses.py:73); the DiT continuous-transformer path accepts `mask` but comments that masks are not implemented in [dit.py](/home/yixunhu/codespace/FLAC/src/models/dit.py:203). A header-only split scan found only 5,533 / 291,210 train targets shorter than 10,240 samples, and 0 / 6,337 unseen eval targets shorter than 10,240. Revise S3 to test target truncation / metric-window coverage / context `max_len=9600` / augment provenance, not assume “tail largely masked.”

3. **Medium: S2 screening is acceptable only as ordering/recovery screening.**  
   K=8 seed-42 full-split screening is statistically fine for the huge documented T60 effect, and it does not violate announcement 01 if clearly labeled non-headline. But the conclusion should be “lr does/does not recover the gate,” not “lr has no effect,” unless confirmed. Also pre-register EDT thresholds, since the user explicitly asked about EDT, not only T60.

4. **Medium: L4 schedule-end wording is overconfident.**  
   The local config uses `lr=5e-5` plus `InverseLR` in [FLAC_AR.json](/home/yixunhu/codespace/FLAC/src/configs/model_configs/FLAC/AR/FLAC_AR.json:139). The formula gives about `4.23e-5` at 400k steps, but `3.54e-5` at 1M steps; `train.py` sets `max_steps=1000000`. Treat L4 as a high-LR continuation proxy, not proven schedule-end faithful.

5. **Low/Medium: S1 is useful but coarse.**  
   Existing interval checkpoints are present at `epoch=0-step=200/400/600.ckpt` for both `outputs_FLAC/exp03_R1b_vanilla_ft/` and `outputs_FLAC/exp05_V1p_freezebn_ft/`, plus exported finals. The three trajectory shapes are discriminative at a coarse level, but a full jump by step 200 cannot distinguish an immediate first-step event from fast 0–200 drift.

6. **Low: upstream provenance needs an explicit remote/ref.**  
   Local `origin` is `git@github.com:Yixun-Hu/FLAC.git`. The HF model card points to the official codebase, resolving to `github.com/AmandineBtto/FLAC` ([HF](https://huggingface.co/AmandineBtto/FLAC), [GitHub](https://github.com/AmandineBtto/FLAC)). S3.1 should fetch/diff that exact upstream ref before making lineage claims.

**Interval-ckpt loading answer:** `--allow-partial-load` should not be needed for the R1b/V1′ interval checkpoints. Their state dicts are 1066 `diffusion.*` model keys plus one `losses.losses.0.weight`; after `eval_FLAC.py` strips `diffusion.`, the only extra key is whitelisted. No EMA keys are present, so no EMA remap occurs. Use `eval_FLAC.py`, not `eval_pl.py`.

**L5-code answer:** yes, L5 needs code. Current `finetune_cond.py` cannot run an InverseLR-restart arm.

**Single Most Valuable Change:** add the tested `--lr-schedule inverse-restart` path, update the plan to mark S2 as code-bearing, and run a smoke/LR-trace before launching L5.
---
**Disposition (Fable 5):** All six findings adopted in the plan revision (S2 marked code-bearing with the L5 TDD spec; S3 mechanism probes rewritten around the reviewer's dataset scan; L4 relabeled a proxy; EDT screening thresholds added; S1 caveat noted; upstream repo pinned to github.com/AmandineBtto/FLAC). Awaiting Yixun's approval.
