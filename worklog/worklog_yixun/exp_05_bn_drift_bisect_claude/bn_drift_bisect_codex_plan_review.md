# Codex plan review — exp_05_bn_drift_bisect

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06
**Target:** `plan_bn_drift_bisect.md` (commit fd552cc+)

**Verdict: APPROVE-WITH-CHANGES**

1. **High: V1 failure attribution is wrong.**  
   The probe concept is sound, but [plan lines 55-59](/home/yixunhu/codespace/FLAC/worklog/exp_05_bn_drift_bisect_claude/plan_bn_drift_bisect.md:55) say “BN drift zeroed but V1 still fails → gradient/target/loss path.” V1 is `lr=0`, so gradients, labels, and loss targets cannot move weights. If V1 fails after apparent BN-drift zeroing, the clean interpretation is: the instrument missed something, or train-mode BN update dynamics/microbatch noise/export state still matter. Target/loss drift only becomes a candidate at V2.

2. **Medium: the bisection grid needs tighter AR-specific knobs.**  
   `AR_md.py` currently loads same-receiver other-source refs, randomly samples them, then end-pads/head-truncates at `max_len` [AR_md.py](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:90), [AR_md.py](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:111). Add `max_len=10240` explicitly: the model sample size is 10240 [FLAC_AR.json](/home/yixunhu/codespace/FLAC/src/configs/model_configs/FLAC/AR/FLAC_AR.json:3), while train acoustic context is 9600 [acousticroom_train.json](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/AR/train/acousticroom_train.json:18). Also define `full` as a fixed-length/padded variant, define “energy-aligned truncation” precisely, and include clamp/min-max and padding/silence-fraction diagnostics. `md_variant` cannot recover samples beyond 9600 after metadata has already been truncated, so longer variants need config-time reload or explicit WAV reread.

3. **Medium: strengthen the instrument tests.**  
   The right target is BN **input** stats, because torchvision ResNet18 BNs sit after convs, and the stem change only replaces conv1 with one input channel [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:37), [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:41). Pin that with a Conv+BN pre-hook test. Also compute variance to match PyTorch BN `running_var` semantics: the stored running variance uses the unbiased estimator. Count the actual ResNet18 BN stack: 20 `BatchNorm2d` layers, no `BatchNorm1d`.

4. **Medium: pin the exact checkpoint and load path.**  
   W0 started from `weights/FLAC/FLAC_EMA.ckpt`, so the probe must compare against those exact EMA BN buffers, not `FLAC.ckpt` or a wrapper-remapped artifact. Given the exp_03 EMA-load confound, make clean load, layer names, and BN buffer count part of the report.

5. **Low: thresholds are plausible, but report estimator noise.**  
   `N>=200`, batch 16 is statistically adequate here because `RIRConditioner` flattens `B*N` context refs [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:156), so K=8 yields about 25,600 RIR spectrograms before spatial averaging. Mean-shift `<0.05` and var ratio `[0.9,1.1]` should be resolvable, but max-over-all-channels needs repeat runs or bootstrap/error bars.

**Explicit Answer 1: Instrument Validity**

Yes, probing BN inputs in eval mode is the right measurement for W0’s mechanism. The plan correctly avoids mutating running stats and uses the conditioner’s real `context_audio` path via `MultiConditioner.only_ids` [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:288). Required fixes: unbiased variance, exact 20-BN coverage, exact FLAC_EMA checkpoint, and reporting estimator noise.

**Explicit Answer 5: Cheaper First Step**

Yes. Add a B-1 pilot before the full tool/grid: one real train batch through exact FLAC_EMA, all BN pre-hook stats dumped, no-mutation asserted, with a tiny table of worst layers. It will not replace the 200-batch estimate, but it catches wrong hooks, wrong checkpoint, wrong variance convention, and gross drift immediately.

**Single Most Valuable Change**

Rewrite the V1 stop rule: V1 pass validates the bisection; V1 fail means the BN probe/update model is incomplete, not that gradient/target/loss drift dominates.
---
**Disposition (Fable 5):** All five findings + the B-1 pilot adopted in the plan revision (V1 stop-rule attribution rewritten; max_len=10240 added as suspect #1 with config-copy reload for >9600 variants; onset-aligned truncation defined; unbiased-variance/Conv+BN-stem/20-BN-count/provenance tests added; 3-repeat error bars). Per Yixun's recorded delegation (notebook 2026-07-06): review findings addressed + no outstanding REQUEST-CHANGES ⇒ plan AUTO-APPROVED; proceeding to TDD.
