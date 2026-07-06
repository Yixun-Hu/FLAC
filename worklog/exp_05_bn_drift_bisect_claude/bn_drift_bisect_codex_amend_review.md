# Codex amendment review — exp_05 (BN-freeze pivot)

**Reviewer:** OpenAI Codex, model `gpt-5.5` at Extra High (`xhigh`) reasoning effort (codex-cli 0.142.5, `codex exec`, read-only sandbox, context-briefed per SOP) · **Date:** 2026-07-06

**Verdict: APPROVE-WITH-CHANGES**

1. **Evidence logic**
   The B1 numbers support “`max_len=9600` is the best tested setting,” not “loader is proven correct.” Baseline train drift is stable: stem `0.0820-0.0825`, worst layer `0.346-0.371`; alternatives are much worse: `4800` stem/worst `0.647/1.121`, `10240` `0.117/1.393`, `19200` `0.620/1.685`. Tight repeats prove the residual is real and well-estimated, not that it is EMA-tail noise.

   EMA-tail is plausible but not proven. With BN momentum `m=0.1`, EMA estimator noise is `sqrt(m/(2-m)) = 0.229x` the per-batch mean dispersion. For independent 256-spectrogram batches, that predicts only about `0.014σ`, matching typical stem mean drift (`~0.014`) but not the max-channel `0.08` unless correlations / room clustering / channel max effects reduce effective sample size. Do the cheap check first: estimate empirical per-batch BN input mean dispersion, then multiply by `0.229` and compare to observed residual. Existing JSONs do not store raw per-batch/signed means, so it is not extractable from artifacts alone; it can be done without repo changes by a one-off script/notebook using `BNInputRecorder`.

2. **Mechanism**
   BN freeze eliminates W0’s proven mutable-buffer damage channel. It also changes training normalization from batch stats to released running stats, so it removes train/inference BN mismatch during fine-tuning.

   Prior is mixed. If R1b/W1’s gradient damage was co-adaptation to batch-normalized / drifting-BN conditioner features, freeze should shrink it. If the remaining damage is genuine target/objective/training-lineage drift, gradients will still flow through frozen normalization and may still fail. V1′ is informative either way:
   - **Pass:** freeze-bn is a valid repair recipe; resume with `fa_invariant + freeze-bn`.
   - **Fail:** BN mutation/inconsistency was only the W0 component; the destructive fine-tune gradient path remains.

3. **Freeze scope**
   Yes: BN eval mode during training is more parity-consistent with released inference because both training and eval then use the released running stats. Affine BN `weight`/`bias` should stay trainable; freezing them would add a second intervention.

   Scope should be “all BatchNorm modules in the trainable path, with logged names/count,” which should currently equal the 20 `BatchNorm2d` layers under `context_audio` / `AudioResNet18`. Local `DiT`/transformer code uses LayerNorm/GroupNorm, not BatchNorm; DINOv3 is expected LayerNorm, but the implementation should still count and report any BN it finds.

4. **Gate logic**
   V1′ is a vanilla control against the same exp_01 2σ gate.
   - **If V1′ passes:** fine-tuning is unblocked with `freeze-bn` documented as a recipe addition; proceed to `fa_invariant + freeze-bn`.
   - **If V1′ fails:** the clean implication is that even after removing BN running-stat mutation and BN train/inference mismatch, this repo’s fine-tune path still destroys the released checkpoint. The registered stop remains right; do not run the FA fine-tune as an absolute-accuracy claim.

5. **SOP gap**
   Add explicit TDD tests for `--freeze-bn`: parser/default, recipe echo, callback installed, BN modules forced eval after parent `.train()`, BN buffers unchanged through a train forward, affine params still trainable, and BN names/count logged. Then do a smoke train step before the full V1′ launch.

**Single Most Valuable Change**

Before V1′, add the bounded per-batch BN mean-dispersion sanity check and revise the amendment wording from “loader already correct; residual = EMA-tail” to “max_len mismatch exonerated; EMA-tail is plausible; V1′ tests the BN-freeze repair regardless of residual provenance.”
---
**Disposition (Fable 5):** Amendment AUTO-APPROVED per delegation with all changes: wording corrected (loader NOT proven correct — max_len exonerated only); per-batch dispersion check runs BEFORE V1′; freeze scope = all BN in trainable path, affine trainable, names/count logged; the 7 TDD tests + pre-launch smoke mandated to the Coder verbatim.
