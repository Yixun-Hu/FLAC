# Yaw-equi-ViT — Azimuth-Equivariant Geometry Encoding for FLAC

Experimental branch. It replaces FLAC's geometry ViT conditioner with a
**cylindrical azimuth-equivariant ViT (CylViT)** and asks:

> Does replacing the vanilla geometry ViT with a cylindrical-equivariant ViT
> reduce yaw-induced metric degradation under the same FLAC
> training/evaluation protocol?

The controlled comparison is **SimpleViT** (the xRIR-style ViT already in this
repo) vs. **CylViT** as the geometry encoder, with everything else matched:
same VAE/pretransform, same DiT, same data splits, same optimizer/schedule,
same seeds, same metric suite (T60, C50, EDT, FD, retrieval).

This is an architecture ablation line, **not** a modification of the released
FLAC checkpoint: `FLAC_EMA.ckpt` uses DINOv3-based geometry conditioners whose
weights do not fit either ViT here, so both A/B models load only the
shape-compatible weights (VAE, DiT, RIR conditioner, distance embedders) and
train the geometry branch from random init. The release config `FLAC_AR.json`
and its DINOv3 path are unchanged on this branch.

## Motivation

The geometry input is a source-relative 3D-coordinate panorama
`[3, 256, 512]` whose azimuth origin is an arbitrary gauge: rolling the
panorama columns by `k` and rotating the horizontal `(x, y)` channels by
`Rz(2πk/W)` is a label-preserving symmetry of the mono RIR. A vanilla ViT is
not invariant to this group action, so predictions degrade when the yaw gauge
changes (measured in `worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/`).

## Method — CylindricalViT

Implemented in [`src/models/cyl_vit.py`](../../../src/models/cyl_vit.py). Three pieces
give equivariance by construction:

1. **Per-pixel gauge alignment.** Each column's `(x, y)` vector is rotated by
   `Rz(-θ_c)` into that column's own azimuth frame, so channel content becomes
   roll-invariant and only spatial position carries the group action.
2. **Roll-equivariant transformer body.** Non-overlapping `16×32` patches give
   a `16×16` token grid. Absolute position encoding is kept for elevation
   only; azimuth uses a circular relative position bias (Swin-style per-head
   table indexed by signed circular azimuth distance).
3. **Invariant readout.** Mean pooling over the token grid
   (`token_pool: "mean"` in the config); the mean of an equivariant token grid
   is invariant.

The result is exact invariance for yaw rotations at patch granularity
(512 / 32 = 16 azimuth patches → the `C16` subgroup, i.e. multiples of 22.5°),
and empirically much smaller deviation than SimpleViT for sub-patch angles.

## What this branch adds

The branch is based on `check-equivariance-necessity` and inherits its
yaw-evaluation infrastructure (`src/data/yaw_rotation.py`, the
`eval_FLAC.py --rotate-deg` yaw-stress protocol, and the `cond_method`
training plumbing). On top of that it adds:

| Path | Change |
|---|---|
| `src/models/cyl_vit.py` | **New.** CylindricalViT encoder. |
| `src/models/conditioners.py` | `ViTCoordinates` gains `arch: "cyl_vit"` and `token_pool: "linear" \| "mean"`; the DINO path (selected by `hf_model_name_or_path`, as before) is untouched and defaults preserve release behavior. |
| `src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json`, `.../FLAC_AR_CylViT.json` | **New.** Matched A/B configs — identical except the geometry-encoder `arch`. |
| `src/training/diffusion.py` | Log `val/avg_loss` so `ModelCheckpoint` can monitor validation during the ablation runs. |
| `worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/` | Experiment plan, training driver, probes, run scripts, results, figures (see below). |
| `src/tests/test_phase3_vit_training.py` | Unit tests for the training driver's LR schedule and optimizer parameter groups. |

## Experiment layout

```
worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/
├── plan_cylvit_yaw_ablation.md        # full experimental design (phases 0–3)
├── train_vit_ablation.py              # matched-training driver (bounded steps,
│                                      #   shape-compatible ckpt init, grouped LRs)
├── encoder_yaw_invariance.py          # Phase 1 encoder-only invariance probe
├── summarize_phase3_convergence.py    # milestone tables from metrics JSONs
├── run_phase2_*.sh                    # short-run training / eval launchers
├── run_phase3_*.sh                    # 30k-step training / milestone eval launchers
├── phase0_phase1_results.md           # integration checks + encoder invariance
├── phase2_eval_results.md             # short-run A/B metrics + yaw deltas
├── phase3_{cylvit,simplevit}_convergence.md  # milestone metrics (updated as runs finish)
├── patches/                           # optional local patches (not applied on the branch)
└── figures/                           # metric-vs-yaw plots
```

## Reproducing

### 0. Setup

Follow the main [README](../../../README.md) install instructions, download the
FLAC weights (`bash download_weights.sh`) and the AcousticRooms dataset.

The eval metrics load the AGREE encoder, whose DINOv3 ViT-S/16 backbone is
fetched from a gated Hugging Face repo. If you cannot access that repo, apply
[`patches/agree_hf_model_local_dinov3.patch`](patches/agree_hf_model_local_dinov3.patch)
(`git apply worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/patches/agree_hf_model_local_dinov3.patch`)
— it builds the matching architecture locally and lets the AGREE checkpoint
supply all weights (strict load). It is intentionally NOT part of the branch
diff to keep DINO code untouched.

### 1. Encoder-only yaw invariance probe (no GPU / no dataset needed)

```bash
python worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py \
  --device cpu --include-context \
  --out-prefix worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance_synthetic_with_context
```

### 2. Matched training

Short-run trend test (Phase 2) and the 30k-step run (Phase 3) are launched via
the run scripts; each takes `MODEL=simplevit` or `MODEL=cylvit`:

```bash
MODEL=cylvit GPU=0 bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_phase2_short_train.sh
MODEL=cylvit GPU=0 bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_phase3_30k_one.sh
```

The underlying driver is `train_vit_ablation.py` (not `train.py`, whose step
budget is fixed at 1M). It loads only shape-compatible tensors from
`weights/FLAC/FLAC_EMA.ckpt`, freezes the VAE, and trains geometry
conditioners + DiT with grouped learning rates (see
`phase3_30k_recipe.md` for the exact recipe).

### 3. Yaw-stress evaluation

```bash
python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  --ckpt-path <exported_or_lightning_ckpt> \
  --cfg-scale 1.0 --steps 1 --seed 42 \
  --rotate-deg 90 \
  --eval-name my_eval_yaw90
```

`--rotate-deg` applies the yaw group action to the eval scenes (panorama
column roll + consistent rotation of coordinate channels and pose vectors).
Milestone sweeps are automated by `run_phase3_milestone_eval.sh`.

### 4. Tests

```bash
pytest src/tests -k "yaw_symmetry or eval_paths or cond_dispatch or invariant_conditioning or phase3_vit"
```

## Results so far

### Encoder-level invariance (Phase 1, synthetic geometry, max |Δ| of the pooled embedding)

| Encoder | patch-grid angles (n·22.5°) | sub-patch angles (5–15°) |
|---|---:|---:|
| SimpleViT | ~2.6–3.1 | ~0.21–0.33 |
| CylViT | **~2e-7** (numerically invariant) | ~0.16–0.19 |

### Matched training, total 10k steps (AcousticRooms unseen, K=1, steps=1, cfg=1.0, seed 42)

Lower is better for T60/C50/EDT/FD; higher is better for R@1/R@5/R@10.

| Model | Yaw | T60 | C50 | EDT | FD | GT R@1 | GT R@5 | GT R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleViT | 0 | 10.645 | 1.792 | 54.11 | 0.391 | 1.341 | 4.987 | 7.922 |
| SimpleViT | 90 | 10.805 | 1.833 | 54.84 | 0.394 | 1.057 | 4.340 | 7.433 |
| SimpleViT | 180 | 10.731 | 1.778 | 55.67 | 0.390 | 1.136 | 4.371 | 7.149 |
| SimpleViT | 270 | 10.615 | 1.887 | 54.91 | 0.394 | 1.089 | 4.245 | 7.511 |
| CylViT | 0 | 10.536 | 1.512 | 49.88 | 0.366 | 1.767 | 6.628 | 11.614 |
| CylViT | 90 | 10.533 | 1.550 | 49.97 | 0.368 | 1.673 | 6.644 | 11.362 |
| CylViT | 180 | 10.515 | 1.575 | 49.48 | 0.367 | 1.689 | 6.644 | 11.441 |
| CylViT | 270 | 10.532 | 1.560 | 50.04 | 0.368 | 1.752 | 6.754 | 11.504 |

At the 10k milestone CylViT is both better at yaw 0 on all seven metrics and
noticeably more yaw-stable (e.g. EDT spread 0.56 vs. 1.56; smaller retrieval
drop under rotation).

### C16 yaw sweep — matched total-25k milestone (all 16 patch-grid angles)

Full sweep over the C16 group (n·22.5°), unseen rooms, K=1, seed 42; per-angle
tables and figures in [`c16_eval_results_25k.md`](c16_eval_results_25k.md)
and `figures/c16_25k_*.png` (this directory).

| Model | Metric | yaw-0 | mean abs Δ | worst Δ | std over angles |
|---|---|---:|---:|---:|---:|
| SimpleViT | T60 ↓ | 10.455 | 0.440 | +0.823 | 0.256 |
| CylViT | T60 ↓ | **10.265** | **0.263** | **+0.514** | **0.171** |
| SimpleViT | C50 ↓ | 1.483 | 0.157 | +0.221 | 0.063 |
| CylViT | C50 ↓ | **1.295** | **0.059** | **+0.090** | **0.030** |
| SimpleViT | EDT ↓ | 49.399 | 2.802 | +5.226 | 1.357 |
| CylViT | EDT ↓ | **46.109** | **0.566** | **+1.044** | **0.390** |
| SimpleViT | FD ↓ | 0.3660 | 0.0096 | +0.0167 | 0.0048 |
| CylViT | FD ↓ | **0.3460** | **0.0009** | **+0.0017** | **0.0009** |
| SimpleViT | GT R@1 ↑ | 1.894 | 0.379 | −0.600 | 0.162 |
| CylViT | GT R@1 ↑ | **3.156** | **0.119** | **−0.237** | **0.089** |
| SimpleViT | GT R@5 ↑ | 6.738 | 1.251 | −1.973 | 0.559 |
| CylViT | GT R@5 ↑ | **11.125** | **0.284** | **−0.521** | **0.190** |
| SimpleViT | GT R@10 ↑ | 11.425 | 2.063 | −2.998 | 0.829 |
| CylViT | GT R@10 ↑ | **17.153** | **0.202** | **−0.395** | **0.169** |

At the matched 25k milestone CylViT is better at yaw 0 on all seven metrics AND
2–10× more yaw-stable across the full C16 group (e.g. mean |ΔEDT| 0.57 vs
2.80; mean |ΔFD| 0.0009 vs 0.0096).

**Status:** the 30k-step Phase 3 runs and full C16 evaluations are complete;
milestone tables are in `phase3_*_convergence.md`, and the final sweep is in
`c16_eval_results_30k.md`. Multi-seed (42–46) and K=8 evaluation from the plan
have not been run yet.

## Caveats

- The A/B models are trained for far fewer steps than the released FLAC model;
  absolute numbers are not comparable to the paper's Table 1.
- Exact invariance holds only at patch granularity (multiples of 22.5°);
  sub-patch behavior relies on the encoder's smoothness, not on symmetry.
- `cond_method="fa_invariant"` / `src/data/yaw_rotation.invariant_conditioning`
  are inherited from the base branch (a sibling experiment); every run in this
  experiment uses `cond_method="vanilla"`.
