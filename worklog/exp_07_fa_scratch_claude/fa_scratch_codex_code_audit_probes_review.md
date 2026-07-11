**Reviewer:** OpenAI Codex gpt-5.6-sol, xhigh reasoning (codex-cli 0.144.1, `codex exec`, read-only sandbox) · **Date:** 2026-07-10

**Verdict: REQUEST-CHANGES**

## Blocking

1. **Part of the evidence was produced by an unreviewable, unreproducible inline executable.** The checkpoint probe ends without performing a model-config diff ([probe_released_ckpt.py](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py:84)), but the log appends an “embedded model_config” section ([ckpt probe log](</home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:35:28_ckpt_probe.log:29>)). Its inline source and exact command are absent, and there is no `fa_scratch_command.md`, contrary to the diagnostic-command rule ([experiment_SOP.md](/home/yixunhu/codespace/FLAC/worklog/experiment_SOP.md:34)).

   **Fix:** move the recursive leaf-level checkpoint/config diff into a checked-in, tested executable; rerun the log; record the exact commands. Record the original missing command as a post-hoc provenance deviation. Re-review that executable before closing the round.

2. **The fallback violates the defining matched-arm constraint.** The audit requires the “same micro×accum in BOTH arms” but immediately proposes `32×2` for B-V and `16×4` for B-F ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:77)). Equal effective batch does not make those trainer configurations identical.

   **Fix:** select one common pair using the more constrained B-F arm—for example, both `16×4`, or both `64×1` if both fit. An asymmetric pair must be a separately declared ablation, not the primary B-F/B-V comparison.

## High

1. **The probe compares counters from different checkpoint phases and therefore does not actually compute accumulation correctly.** It reads `batch_progress.total.completed=67,499` ([probe](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py:40)) against optimizer `completed=67,500`, then rounds `0.999985` to `1.0000` ([probe](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py:53)). Lightning saves from the checkpoint callback before batch `completed` increments; the checkpoint’s full state instead contains:

   - processed/ready micro-batches: `67,500`, current epoch `3,800`
   - optimizer steps completed: `67,500`, current epoch `3,800`
   - completed micro-batches: `67,499`, current epoch `3,799`

   The corrected arithmetic is airtight:

   `67,500 − 3,800 = 63,700 = 14 × 4,550`

   Equivalently, the lagged pair gives `67,499 − 3,799 = 14 × 4,550`. Thus accumulation is exactly 1 under like-for-like counters. Accumulation 2 is incompatible. With the shipped 291,210-item split, effective batch 128 would give `floor(291,210/128)=2,275` optimizer steps/epoch and approximately epoch 29 at step 67,500, also incompatible.

   However, the counters do **not** prove the decomposition `micro 64 × one GPU`: `2 GPUs × micro 32 × accum 1` also produces global effective batch 64 and 4,550 per-rank batches. “Single H100/micro 64” comes from the paper, not checkpoint arithmetic. The checkpoint alone also lacks dataset/trainer arguments ([probe log](</home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:35:28_ckpt_probe.log:21>)).

   **Fix:** print every counter phase, use `processed` for accumulation, explain the one-count lag, and change “arithmetically certain” ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:24)) to “effective batch 64 is implied conditional on the shipped split; the 64×1×1 decomposition is paper-specified.”

2. **The ViT initializer difference is training-relevant, not a proven no-op.** The audit calls the local authors’ path and Hub identifier “same weights” ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:48)), while later admitting that the authors’ revision is unknown ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:64)). DINO is trainable, so its initial weights affect lineage.

   The available cache currently resolves to revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`; `model.safetensors` SHA-256 is `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`. That documents **our** initializer but cannot prove equality to the missing authors’ local snapshot.

   **Fix:** pin and log that revision/hash for both arms. Downgrade “B-V ≡ released FLAC … PROVEN up to no-op keys” ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:15)) to an unresolved training-relevant initializer-provenance caveat.

3. **The proposed “faithful launch” silently chooses unresolved trainer/data settings.** The shipped training command uses batch 32, accumulation 2, two GPUs, eight workers, and validation every 2,500 steps ([README.md](/home/yixunhu/codespace/FLAC/README.md:114)); defaults specify batch 64, accumulation 1, one GPU, six workers, and `val_every=-1` ([defaults.ini](/home/yixunhu/codespace/FLAC/defaults.ini:14)). The audit adopts defaults without adjudicating this conflict ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:77)).

   These are not all logging-only:

   - Worker count changes worker RNG assignment; context RIRs use `np.random.choice` ([AR_md.py](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:104)), and augmentations are stochastic.
   - Validation samples Gaussian noise ([diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:415)), so changing validation cadence advances training-process RNG.
   - Training implicitly uses `shuffle=True` and `drop_last=True` ([dataset.py](/home/yixunhu/codespace/FLAC/src/data/dataset.py:344), [dataset.py](/home/yixunhu/codespace/FLAC/src/data/dataset.py:405)). The latter is the missing code link behind `floor(291,210/64)=4,550`.

   **Fix:** add an explicit launch manifest pinning `num_workers`, validation dataset/cadence, shuffle, drop-last, nodes, strategy, gradient clipping (`0.0`), precision, matmul policy, dependency versions, and hardware. Settings not recoverable from the checkpoint must be labeled choices, not released-run identity. Keep them identical across arms.

## Medium

1. **`assert_arm_configs.py` only partially validates the wiring it claims.** The factory argument order is correct ([factory.py](/home/yixunhu/codespace/FLAC/src/training/factory.py:5)), and the asserted wrapper attributes are real ([diffusion.py](/home/yixunhu/codespace/FLAC/src/training/diffusion.py:75)). A wrong `cond_method` would fail either construction or the assertions, so those checks bite.

   But optimizer, scheduler, and CFG checks mostly reread the input JSON ([assert_arm_configs.py](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/assert_arm_configs.py:52)); they do not prove factory wiring. The script never calls `configure_optimizers`, never checks `w.optimizer_configs` or `w.cfg_dropout_prob`, and does not compare initialized state values.

   **Fix:** assert wrapper fields directly, call `configure_optimizers()`, verify actual `AdamW` and `InverseLR` objects and their fields, reseed before each build, and compare the two model state dictionaries or hashes at initialization.

2. **The plan remains internally contradictory after its revision.** It still says the original used at least two GPUs ([plan](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/plan_fa_scratch.md:33)) and still registers 10,000-step checkpoints ([plan](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/plan_fa_scratch.md:41)), while the audit pins one GPU and 2,500-step checkpoints. The audit itself is clear that `defaults.ini` says 10,000 and launch should explicitly override it to 2,500; the unresolved ambiguity is in the governing plan.

   **Fix:** revise those original plan lines explicitly, rather than relying on a later section to supersede them implicitly.

3. **The micro-batch caveat incorrectly invokes BatchNorm.** The audit cites different “BN micro-stats” ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:81)), but there is no `BatchNorm` in `src/models` or `src/training`; DINO/DiT use normalization that is not batch-statistical.

   **Fix:** replace this with the real caveats: different accumulation arithmetic, BF16 reduction order, stochastic RNG sequencing, and data-order/resume behavior.

## Low

1. **The InverseLR comparison omits its warmup term.** The exact implementation is  
   `5e-5 × (1−0.99^(67500+1)) × (1+67500/1e6)^−0.5` ([training/utils.py](/home/yixunhu/codespace/FLAC/src/training/utils.py:56)). Here `0.99^67501 ≈ 2.3511×10^-295`, so the factor rounds to exactly 1 in float64 and the corrected result is exactly `4.839339184958273e-05`. The conclusion is correct; show the complete formula.

2. **The description of `img_h/img_w` defaults is inaccurate.** `from_scratch` defaults to false as claimed, but `img_h/img_w` are read in the SimpleViT branch when no Hugging Face path is supplied—not in the Hugging Face “from-scratch” branch ([conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:371), [conditioners.py](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:403)). They remain no-ops for these configs; correct the explanation.

3. **Paper total and training-subset counts should be separated.** The paper’s “over 300k” describes all 260 rooms ([FLAC_pdf.md](/home/yixunhu/codespace/FLAC/FLAC_pdf.md:458)); `all_data.json` contains 302,671 items, while the 243-room training split contains 291,210. The audit currently places “>300k total” opposite the 291,210 training count ([audit](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_config_identity_audit.md:38)). Correct the row to distinguish those quantities.

4. **The latest worklog timestamp lacks the mandatory UTC offset.** It uses `2026-07-10T23:50` ([worklog](/home/yixunhu/codespace/FLAC/worklog/exp_07_fa_scratch_claude/fa_scratch_worklog.md:9)); the SOP requires local time with offset ([experiment_SOP.md](/home/yixunhu/codespace/FLAC/worklog/experiment_SOP.md:44)). Correct to an offset-bearing timestamp such as `2026-07-10T23:50:00-04:00`.

Independent checks that passed: B-V is byte-identical to the repository config (SHA-256 `f3eafef…a2a7ec9`); B-F’s semantic diff is exactly the two requested training keys; split counts are 291,210/243, 6,217/131, and 6,337/17; and the 4.32M-sample, 5.0-day, and 16.7-day arithmetic is exact at the stated throughput anchors.

## Single Most Valuable Change

Replace the prose-level “PROVEN” launch identity with one reviewed, fail-closed audit executable that emits the corrected Lightning counters, canonical config diffs, dataset/drop-last counts, pinned ViT revision/hash, and a complete explicit launch manifest—and rejects any B-F/B-V trainer-setting mismatch.