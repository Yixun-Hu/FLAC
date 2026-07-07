# Codex plan review — exp_08_fa_matched

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-07

**Verdict: APPROVE-WITH-CHANGES**

1. **Medium: H-A1 has an eval-precision mismatch.** M2’s `--cond-autocast bf16` is correct for A-F and matches exp_03 C1. But the reused V1′ evals are `cond_autocast: default` from the exp_01 protocol, while `eval_FLAC.py` applies this autocast wrapper to both vanilla and FA conditioning paths. For the matched marginal comparison, rerun the existing A-V checkpoint evals with `--cond-autocast bf16` for K={1,8}, seeds 42-46, or explicitly register the mismatch. Best fix: rerun only the cheap A-V eval mirror, not the training.

2. **Low: control-reuse wording overclaims “byte-identical.”** The relevant V1′ code state is after `5d1c64c` / worklog close `51b7486`, not `992fe49` alone, since `992fe49` predates `--freeze-bn`. Diffing `5d1c64c..HEAD` shows only `--lr-schedule`, the warmup/schedule conflict guard, and recipe echo changes in [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:39). For `--lr-schedule constant` and `--warmup-steps 0`, those are behavioral no-ops. A rerun today is recipe/codepath-equivalent, but not guaranteed bit-identical due CUDA/data-loader nondeterminism.

3. **Low: statistics gate is reasonable but should be phrased as a decision gate.** V1′ sigmas exist: K1 T60 0.058, C50 0.0073, EDT 0.125; K8 T60 0.0048, C50 0.0025, EDT 0.0106. “Within 2x combined sigma” over 6 primary metric-K cells is fine for screening, but superiority should be descriptive unless coherent across cells or adjusted for multiplicity.

4. **Low: wall-clock is plausible but optimistic at K=8.** R0 K=1 FA eval was about 10-14 min; K=8 runs many more ViT forwards, and exp_03 already flags 36 DINO forwards per K=8 FA batch. M2 ~4h is plausible; M4b ~1h may be tight. M0 should update the ETA.

**Explicit Answer 1**

Control reuse is SOP-legal if exp_08 records the reused artifact, exact exp_05 command, code-diff proof, and full-split eval provenance. The later `finetune_cond.py` changes are no-ops for the V1′ constant-lr freeze-BN recipe. Do not claim strict bit comparability; claim recipe-equivalent reuse. Reusing the V1′ training arm is sound.

**Explicit Answer 2**

`--cond-method fa_invariant --freeze-bn` is safe. `FreezeBN` discovers all BN modules under `pl_module.diffusion` and reasserts eval mode on every train batch start in [finetune_cond.py](/home/yixunhu/codespace/FLAC/finetune_cond.py:225). `invariant_conditioning` runs one full conditioner pass, then only ViT ids in extra passes in [src/data/yaw_rotation.py](/home/yixunhu/codespace/FLAC/src/data/yaw_rotation.py:285); `MultiConditioner.only_ids` skips non-ViT conditioners in [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:288). The RIR encoder BN path runs once and frozen.

**Single Most Valuable Change**

Add an A-V bf16 eval mirror from the existing V1′ checkpoint and use that row for H-A1. This preserves the compute savings while removing the only material matched-comparison confound.
---
**Disposition (Fable 5):** All findings adopted: M1.5 A-V bf16 eval mirror added as the H-A1 comparator (the review's most valuable change — kills the precision confound at zero training cost); control-reuse wording corrected to recipe-equivalent with code-diff provenance (5d1c64c..HEAD); superiority phrasing made descriptive-unless-coherent; M0 mandated to update the K=8 eval ETA. FreezeBN×invariant_conditioning interaction verified safe by the review. Awaiting Yixun's approval.
