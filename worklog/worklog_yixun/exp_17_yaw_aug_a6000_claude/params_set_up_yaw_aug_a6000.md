# exp_17 — registered parameters (Yaw-Aug FROM-SCRATCH, 2×A6000)

Everything a reader needs to reproduce or falsify the arm. Values marked
**PINNED** are enforced by a gate that aborts the launch, not by convention.

## Identity

| | |
|---|---|
| Arm | Yaw-Aug (vanilla FLAC + training-time random yaw augmentation) |
| Control | exp_07 **P1 vanilla @40k** (same recipe, same seed, no augmentation) |
| Arm config | `worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json` |
| Control config | `worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json` (sha256 `733ca52b…a49d8`, **PINNED**) |
| Branch / base | `exp17-yawaug-scratch`, from current HEAD of `check-equivariance-necessity` |
| W&B | project `FLAC_exp17_YAWAUG`, run `exp17_YAWAUG`, account `yh4742@princeton.edu` (**PINNED**, fail-closed) |
| Output | `outputs_FLAC/exp17_YAWAUG` (**PINNED**; refuses to start if it already holds checkpoints) |

## The arm is the control plus exactly three registered deltas

| # | Delta | Why it is allowed |
|---|---|---|
| 1 | `training.yaw_aug = {enabled: true, img_w: 512, seed: 42}` | **the treatment** |
| 2 | `gradient_checkpointing: true → false` on `source_vit` | numerically inert |
| 3 | `gradient_checkpointing: true → false` on `context_poses_vit` | numerically inert |

Deltas 2/3 are a VRAM-for-time trade Yixun authorised on 2026-08-15 by freeing
both A6000s for this arm. They are admissible as *non-treatment* because exp_07
measured ViT gradient checkpointing ON vs OFF to produce **bitwise-identical
parameter gradients** (210 tensors, max abs diff 0.0; `state_dict` sha256
unchanged), pinned by `src/tests/test_vit_gradient_checkpointing.py`. Measured
effect: 3.86 → 2.200 s/opt-step, i.e. 42.9 h → 24.4 h at 40k.

Any **fourth** difference makes this a two-factor experiment and aborts the
launch: the gate reverts all three deltas and requires type-strict equality with
the control (`int 0` ≠ `float 0.0`, `1` ≠ `True`).

## Augmentation semantics

Online per-training-step re-orientation — **not** an enlarged dataset. Per
sample, draw `d ~ Uniform{0..511}` and roll the depth panorama by `d` integer
columns (exact at W=512, no interpolation), rotating the matching 4 pose fields
(`source`, `source_vit`, `context_poses`, `context_poses_vit`) through
`rotate_scene_metadata`. The RIR and context audio are untouched: this is a
change of coordinate frame, not of the room. RNG is counter-based on
`(yaw_aug.seed, global_step, global_rank, within-batch index)`, so the draw is
reproducible and differs across ranks.

## Recipe (identical to P1)

| Parameter | Value |
|---|---|
| Rung | 32/GPU × 2 GPUs × accum 1 → **eff/BN batch 64** (**PINNED** by string equality) |
| SyncBN | true (accumulation never feeds BN statistics — standing repo lesson) |
| Strategy / precision | `ddp_find_unused_parameters_true` / `bf16-mixed` |
| Seed | 42 |
| Steps | **40,000** (**PINNED**, non-overridable) |
| Checkpoint cadence | **2,500** (**PINNED**) → 16 checkpoints |
| EMA | on (**PINNED**; evaluation uses EMA weights) |
| Conditioning | vanilla — `cond_method` must be **absent** (**PINNED**) |
| Optimizer | AdamW lr 5e-5, betas (0.9, 0.999), wd 1e-3; InverseLR inv_gamma 1e6, power 0.5, warmup 0.99 |
| ViT | `facebook/dinov3-vits16-pretrain-lvd1689m`, 256×512, not frozen, 21.60 M trainable |
| Dataset | `acousticroom_train.json` — 291,210 files / 243 subfolders → 4,550 steps/epoch (40k ≈ 8.8 epochs) |
| VAE pretransform | `weights/FLAC/VAE.safetensors` (sha256 `8d82159e…f0b9`, **PINNED**) |

## Content pins checked at every launch

`src/data/yaw_rotation.py`, `src/training/diffusion.py`, `src/training/factory.py`,
`train.py`, the dataset config, the VAE weights, and the control config — seven
sha256 values. A clean-tree check only proves nothing is *modified*; these prove
it is *this* code after a checkout, rebase, or stash pop.

## What would invalidate the run

1. The treatment banner `yaw_aug ENABLED img_w=512 seed=42` absent from the log
   (matched as a whole line against `src/training/diffusion.py:407`).
2. Any config drift beyond the three registered deltas.
3. `img_w` disagreeing between `yaw_aug` and either ViT — every sample would be
   rotated by the wrong angle and nothing would raise.
4. A rung other than 32×2×1, which would change the BN batch.

*Recorded by the main session seat (Claude Opus 5, max effort).*
