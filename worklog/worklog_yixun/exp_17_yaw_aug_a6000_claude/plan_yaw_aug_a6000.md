# Plan — exp_17 matched 2×A6000 random-yaw augmentation

**Status:** Approved by Yixun for local 2×A6000 execution. Opus 5 plan review completed 2026-08-15; blocker B1 is resolved below by replacing the nonexistent source pin with an audited current-HEAD base. Implementation and launch remain pending.

## 1. Question and estimand

Train a vanilla-conditioned FLAC arm with physically consistent random yaw augmentation and compare it with the existing Vanilla P1 and legacy per-angle FA B-F arms under the same local recipe:

- 2×NVIDIA RTX A6000 (48 GiB), one node;
- DDP + SyncBatchNorm;
- micro-batch 32/GPU, accumulation 1, effective/BN batch 64;
- training seed 42, EMA on, bf16-mixed, ViT gradient checkpointing on;
- standard 291,210-item / 243-room AcousticRooms training split;
- checkpoints every 2,500 optimizer steps through the fixed 40,000-step endpoint.

Primary estimand: the effect of adding training-time random yaw augmentation to the P1 vanilla recipe. B-F is a second comparator, not the control for the augmentation effect.

The existing Neuronic exp_15 arm is explicitly out of the primary contrast because it uses 8×L40 × micro-8 and exp_11 VANL. It will be reported separately as a cross-recipe replication when available.

## 2. Treatment and single-delta contract

### 2.1 Model config

Create `FLAC_AR_YAWAUG_A6000.json` as a byte-copy of exp_07's `FLAC_AR_BVp1.json` plus exactly this final training block:

```json
"yaw_aug": {
  "enabled": true,
  "img_w": 512,
  "seed": 42
}
```

Parsed-object comparison must prove that removing `training.yaw_aug` yields exact type-strict equality with P1. The config must have vanilla conditioning (no `cond_method`), EMA enabled, and gradient checkpointing enabled for both ViT conditioners.

### 2.2 Augmentation semantics

Reuse exp_15's reviewed implementation without modification:

- one offset per sample, uniform over integer panorama columns `d ∈ {0,…,511}`;
- counter-based draw keyed by `(seed=42, global_step, global_rank, batch_index)`, leaving global RNG untouched and remaining resume-exact;
- one physically consistent yaw applied to the depth panorama and all four pose fields through `rotate_scene_metadata`;
- target RIR and context audio remain unchanged;
- training step only; validation/evaluation remain unaugmented unless an explicit eval rotation is requested;
- rank-0 pre-step banner proves the treatment is active.

### 2.3 Source pin

Build the experiment in an isolated `exp-17-yawaug-a6000` worktree/branch from audited base `41aa31dc0f5787019f26912654e4c5a14be7feeb` and bind training to the final reviewed exp_17 implementation commit. The earlier draft pin `58d0b887` does not exist in any local or remote ref and is formally withdrawn.

The current base is scientifically admissible because the later shared-path changes are default-preserving for this arm: absent `training.frame_avg_max_fwd_samples` retains the literal legacy conditioning call; absent `training.are_lambda` retains the legacy target path; and the exp_16 `RandomTimeShift` metadata publication exposes an already-drawn value without changing the waveform or consuming another RNG draw. Existing disabled/default-path and RNG-stream regression tests must pass at the final pin. Add only the exp_17 config, tests, launcher, and worklog artifacts on top. Training must print and record the final full exp_17 commit SHA.

The existing P1 checkpoint was produced across source commits due a resume at 32.5k, while this arm is a new from-scratch trajectory. Therefore the claim is a single intended algorithmic/config delta with disabled-path regression evidence, not bitwise trajectory pairing.

## 3. Training protocol

Launch from scratch after both checkpoint-curve workers finish and both GPUs pass ownership/free-memory checks:

- dataset: `src/configs/dataset_configs/AR/train/acousticroom_train.json`;
- VAE: `weights/FLAC/VAE.safetensors`, hash recorded at launch;
- max steps: 40,000;
- checkpoint cadence: 2,500;
- seed: 42;
- `--batch-size 32 --accum-batches 1 --num-workers 6`;
- `--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true`;
- `--precision bf16-mixed`, EMA inherited from config;
- W&B project/run and output namespace unique to exp_17.

The launcher must refuse nonliteral batch/accum/GPU/step/cadence values, config drift, unexpected source drift, wrong DINOv3/VAE pins, occupied GPUs, insufficient storage, or a missing augmentation banner. It must tee a timestamped log and record the exact command at launch time.

Estimated training time is approximately 43–50 hours from launch based on P1's measured ~0.26 steps/s plus bounded augmentation overhead. A 20–30 optimizer-step smoke on the same 2×A6000 rung will measure the actual post-warmup rate before the full launch; no checkpoint is written by the smoke.

**Smoke abort threshold (R3, pre-registered).** Convert the smoke's post-warmup steady-state rate to a projected wall-clock for 40,000 steps. **If the projection exceeds 55 hours, do not launch** — stop, report the measured rate, and re-plan (options: accept the longer budget explicitly, or investigate the augmentation overhead, which the exp_15 review bounded as small). Launching a run that is already known to overrun its budget is not a decision to make silently at 3 a.m.

The smoke gate is fail-closed: if its steady-state rate projects the 40k run beyond 55 wall-clock hours, or if either rank OOMs, produces non-finite loss, misses the treatment banner, or violates the expected device/batch topology, do not launch the full run; stop and re-plan with Yixun.

## 4. Validation ladder

1. Static/config checks: JSON parse, `bash -n`, `git diff --check`, source/config hash readback.
2. Existing exp_15 unit/regression suite at the source pin: yaw parser, counter draw, schema guard, physical rotation, RNG isolation, and disabled path.
3. New config-contract tests: byte construction, type-strict single delta, vanilla/EMA/grad-checkpoint pins, img width agreement.
4. Launcher guardtests: exact argv; 2×32×1; 40k cap; checkpoint cadence; config/source/data/VAE gates; no checkpoint in smoke; no accidental submission/launch in dry-run.
5. Small real-data readback: one real batch, verify depth/pose shapes and that yaw changes only the declared metadata fields.
6. Two-GPU smoke: treatment banner before step 0, at least one optimizer step, no OOM/NaN, peak memory recorded, step-rate estimate recorded.
7. Full 40k run only after all earlier rungs and the integrative opposite-family code review are green.

Every executable addition follows TDD and receives independent review before use.

## 5. Checkpoint curve and checkpoint policy

The fixed 40k endpoint is the primary matched-budget comparison; it is never replaced post hoc by whichever checkpoint looks best.

For trajectory evidence, evaluate every Yaw-Aug checkpoint from 2.5k through 40k at K=1, yaw 0°, eval seed 42 with explicit vanilla conditioning. Add it to the already-running P1/B-F six-metric curve. These single-seed curves diagnose whether an endpoint advantage is persistent or a band draw; they do not enter the five-seed headline table.

If a best-within-40k result is later desired, its scalar selection rule and selection/confirmation seed split must be registered in a plan amendment before viewing Yaw-Aug checkpoint metrics. Per-metric oracle envelopes may be plotted descriptively but cannot be presented as one deployable checkpoint.

## 6. Evaluation protocol

Use EMA and the full existing 6,337-item / 17-room unseen AcousticRooms configs. Always pass protocol flags explicitly:

```text
--cond-method vanilla
--frame-avg-angles 0,90,180,270
--frame-avg-max-fwd-samples 64
--cond-autocast bf16
--rotate-deg <0|90|180|270>
```

No new or subsampled dataset config is allowed.

### 6.1 Primary Table-1 block

Yaw-Aug@40k, K∈{1,8}, yaw 0°, evaluation seeds 42–46: 10 cells. Aggregate per-scene means and report mean ± sample SD over five evaluation seeds for T60, C50, EDT, R@1, R@5, R@10, and FD.

Compare against the already-validated P1@40k and B-F@40k five-seed rows. Re-evaluate a control cell only if checkpoint/protocol hashes do not match the existing record.

### 6.2 Fixed-yaw robustness block

Yaw-Aug@40k, K∈{1,8}, yaw ∈ {90°,180°,270°}, seeds 42–46: 30 new cells. Combine with the existing P1/B-F fixed-yaw grid to produce the three-method academic figure requested by Yixun. The yaw 0° cells are shared with §6.1, so the complete Yaw-Aug grid is 40 cells, not 50.

### 6.3 Trajectory block

Sixteen checkpoints × K=1 × yaw 0° × seed 42. The 40k cell may be reused from §6.1 after artifact validation, giving at most 15 additional jobs.

All eval manifests bind checkpoint SHA, model-config SHA, **both the TRAINING source SHA and the EVALUATION source SHA (R5 — they can differ if training is pinned and evaluation runs at a later HEAD)**, dataset config, K, seed, rotation, conditioning mode, frame angles, autocast, evaluation chunk plan, full item count, and metric output path. Existing artifacts are skipped only after validation.

## 7. Statistics and claims

- Five evaluation seeds quantify diffusion/evaluation variability, not training-seed variability. There is one training seed per arm.
- Primary clean contrast: paired by evaluation seed, Yaw-Aug−P1 at 40k for each K/metric; directions follow metric arrows.
- Secondary method contrast: Yaw-Aug−B-F, labelled architecture-vs-augmentation rather than a pure single delta.
- Pre-registered regularization interpretation: improvement at yaw 0° is compatible with generic data-diversity regularization; evidence for yaw-specific learning requires substantially larger gains at 90°/180°/270° than at 0°. Do not attribute all rotated-input gains to learned yaw structure without this degradation-relative-to-0° analysis.
- Robustness reports both absolute metrics by angle and degradation relative to each method's own 0° result.
- Avoid claiming exact yaw invariance for Yaw-Aug; the test is empirical robustness/generalization.
- **R1 — the regularization confound, registered before any Yaw-Aug metric is seen.** A yaw-augmented arm may beat the baselines on rotated inputs simply because random yaw increases effective data diversity (a generic regularizer), not because it learned anything yaw-specific. The 0° cells already collected in §6.1 are the discriminator, and the reading is fixed now: **pure regularization predicts a gain at 0° comparable to the gain at 90/180/270; yaw-specific learning predicts a much larger gain at the rotated angles than at 0°.** Report the 0° gain and the rotated gains side by side and state which pattern the data shows. Neither outcome is a failure — but the interpretation must not be chosen after seeing the numbers.
- FD is reported but interpreted separately because prior FA results improve acoustic/retrieval metrics while FD can worsen.
- The fixed endpoint, single training seed, source-generation difference, and checkpoint-band caveats are mandatory in the final analysis.

## 8. Planned files after approval

Inside `worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/`:

- `FLAC_AR_YAWAUG_A6000.json` — P1 plus one augmentation block;
- `yaw_aug_a6000_launch.sh` — fail-closed smoke/full launcher;
- `yaw_aug_a6000_guardtests.sh` — dry-run and failure-mode tests;
- `yaw_aug_a6000_eval.sh` and `yaw_aug_a6000_eval_grid.sh` — validated local two-GPU eval queues after training;
- `yaw_aug_a6000_collect.py` — validates/aggregates trajectory and five-seed grids;
- params, command, results, analysis, commit ledger, and final offline HTML/assets required by the SOP.

Permanent pytest additions under `src/tests/` cover the config and any new Python collector functions. No changes to the already-reviewed augmentation implementation are planned.

## 9. Acceptance criteria

Training is valid only if the logged source/config/VAE pins match; two A6000 ranks initialize; global/BN batch is 64; the enabled banner precedes step 0; no OOM/NaN occurs; checkpoints land at every 2.5k boundary; and the 40k checkpoint passes global-step/config/EMA/hash admission.

Evaluation is valid only if each cell loads that admitted checkpoint under explicit vanilla conditioning, evaluates all 6,337 examples, writes complete metric JSON, and passes manifest validation. Final tables/plots must be regenerated from raw metric JSONs, never hand-entered.

## 10. Schedule and resource sequencing

- Current P1/B-F K=1 curve: first priority; expected to finish around 2026-08-15 00:55 EDT.
- Plan/config/launcher review and smoke: approximately 1–2 hours after curve completion.
- Full training: approximately 43–50 hours after launch.
- Trajectory + 40 fixed-yaw five-seed cells: approximately 4–7 wall-clock hours using both GPUs after training, based on measured local eval rates.
- Earliest paper-ready result: approximately 2.5 days after full training launch, absent hardware interruption.

Launch only after ownership and free-memory checks. Co-tenancy is permitted only when the remaining per-card VRAM floor and measured smoke fit are explicitly recorded; never terminate or modify an unowned process. At the 2026-08-15 authorization check both A6000s were idle, so the intended launch is exclusive. Every training and evaluation manifest records both the pinned training SHA and the evaluation SHA.
