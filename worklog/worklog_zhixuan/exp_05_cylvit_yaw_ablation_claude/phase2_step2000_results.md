# Phase 2 Step-2000 Results: SimpleViT vs CylViT Yaw Ablation

Date: 2026-07-10

This note summarizes the current reliable comparison point for the exp05 yaw-invariance ablation. The fair completed comparison is the `step=2000` checkpoint for both models, evaluated on the same K=1 unseen AcousticRooms split under yaw rotations.

## Run Status

| Model | Directory | Current status |
|---|---|---|
| SimpleViT | `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/` | Finished `global_step=2500`; final export exists. |
| CylViT | `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/` | Stopped before final export; `last.ckpt` is `global_step=2276`. |

Because CylViT did not complete the full 2500-step run, the table below uses the shared `epoch=0-step=2000.ckpt` checkpoint from both models.

## Evaluation Setup

- Train style: FLAC warm-start fine-tuning from `weights/FLAC/FLAC_EMA.ckpt`.
- Frozen modules: VAE/pretransform.
- Trainable modules: geometry conditioner + DiT.
- Eval split: `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json`.
- Context size: K=1.
- Yaw angles: `0`, `90`, `180`, `270` degrees.
- Metric direction:
  - Lower is better: T60, C50, EDT, FD.
  - Higher is better: retrieval metrics R@1, R@5, R@10.

## Step-2000 Main Metrics

| Model | Yaw | T60 (%) ↓ | C50 (dB) ↓ | EDT (ms) ↓ | FD ↓ | RIR-to-GT R@1 ↑ | RIR-to-GT R@5 ↑ | RIR-to-GT R@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SimpleViT | 0 | 11.0627 | 2.0979 | 62.1532 | 0.4295 | 0.8521 | 3.1876 | 5.4758 |
| SimpleViT | 90 | 11.2404 | 2.0065 | 61.9376 | 0.4300 | 0.7575 | 3.0456 | 5.2075 |
| SimpleViT | 180 | 11.1345 | 2.0519 | 62.0666 | 0.4299 | 0.7417 | 3.1718 | 5.5862 |
| SimpleViT | 270 | 10.9968 | 2.1783 | 62.3131 | 0.4282 | 0.9153 | 3.4243 | 5.4127 |
| CylViT | 0 | 11.2676 | 1.8649 | 57.0981 | 0.4096 | 0.9784 | 3.4559 | 6.0439 |
| CylViT | 90 | 11.3042 | 1.8975 | 57.3472 | 0.4087 | 1.0415 | 3.4717 | 5.9965 |
| CylViT | 180 | 11.2269 | 1.8964 | 56.8716 | 0.4106 | 0.9626 | 3.5979 | 6.0754 |
| CylViT | 270 | 11.2626 | 1.9225 | 56.9504 | 0.4096 | 1.0099 | 3.6926 | 6.2490 |

## Geometry Retrieval Metrics

| Model | Yaw | RIR-to-geom R@1 ↑ | RIR-to-geom R@5 ↑ | RIR-to-geom R@10 ↑ |
|---|---:|---:|---:|---:|
| SimpleViT | 0 | 0.5208 | 2.1461 | 3.8346 |
| SimpleViT | 90 | 0.4418 | 1.8463 | 3.4086 |
| SimpleViT | 180 | 0.5050 | 1.9725 | 3.5506 |
| SimpleViT | 270 | 0.4734 | 1.9252 | 3.4243 |
| CylViT | 0 | 0.6154 | 2.5722 | 4.5290 |
| CylViT | 90 | 0.5839 | 2.5406 | 4.2449 |
| CylViT | 180 | 0.6312 | 2.6195 | 4.5290 |
| CylViT | 270 | 0.5523 | 2.4302 | 4.0871 |

## Yaw Robustness Relative to Yaw 0

Positive/negative signs below are raw changes from yaw 0. For errors, smaller absolute drift is generally better. For retrieval, positive drift is better.

| Model | Yaw | ΔT60 | ΔC50 | ΔEDT | ΔFD | ΔR@1 | ΔR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleViT | 90 | +0.1777 | -0.0914 | -0.2156 | +0.0005 | -0.0946 | -0.2683 |
| SimpleViT | 180 | +0.0718 | -0.0460 | -0.0866 | +0.0004 | -0.1104 | +0.1104 |
| SimpleViT | 270 | -0.0659 | +0.0804 | +0.1599 | -0.0013 | +0.0632 | -0.0631 |
| CylViT | 90 | +0.0366 | +0.0326 | +0.2491 | -0.0009 | +0.0631 | -0.0474 |
| CylViT | 180 | -0.0407 | +0.0315 | -0.2265 | +0.0010 | -0.0158 | +0.0315 |
| CylViT | 270 | -0.0050 | +0.0576 | -0.1477 | +0.0000 | +0.0315 | +0.2051 |

## Interpretation

At the shared 2000-step checkpoint, CylViT is currently stronger than SimpleViT on the main downstream metrics:

- CylViT has lower EDT and FD at all yaw angles.
- CylViT has higher RIR-to-GT retrieval at all yaw angles.
- CylViT keeps R@1 closer to its yaw-0 value across the C4 rotations; SimpleViT has larger losses at 90 and 180 degrees.
- CylViT also has better geometry retrieval at all yaw angles.

The main caveat is that this is still a short ablation run. It is useful for comparing the two ViT choices under the same training budget, but it is not close to full FLAC paper-level performance.

## Reference: Original FLAC / exp01 Scale

For context, the reproduced/released FLAC exp01 K=1 result is much stronger than this short run:

| Source | T60 (%) ↓ | C50 (dB) ↓ | EDT (ms) ↓ | FD ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLAC exp01/release K=1 | ~9.969 | ~1.046 | ~39.95 | ~0.3031 | ~6.83 | ~19.08 | ~26.98 |
| FLAC paper Table 1 K=1 | ~9.95 | ~1.046 | ~40.04 | N/A | ~6.80 | ~18.92 | ~26.87 |

So the current exp05 results should be read as an early-budget architecture ablation, not as final FLAC-quality generation.

## Artifacts

Step-2000 metric JSON files:

- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_simplevit_step2000_K1_seed42_rot0.json`
- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_simplevit_step2000_K1_seed42_rot90_rot90.json`
- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_simplevit_step2000_K1_seed42_rot180_rot180.json`
- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_simplevit_step2000_K1_seed42_rot270_rot270.json`
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_cylvit_step2000_K1_seed42_rot0.json`
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_cylvit_step2000_K1_seed42_rot90_rot90.json`
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_cylvit_step2000_K1_seed42_rot180_rot180.json`
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000_metrics_1_1.0_exp05_cylvit_step2000_K1_seed42_rot270_rot270.json`

Checkpoints:

- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000.ckpt`
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000.ckpt`
- `FLAC/outputs_FLAC/exp05_simplevit_2500s_s42/last.ckpt` (`global_step=2500`)
- `FLAC/outputs_FLAC/exp05_cylvit_2500s_s42/last.ckpt` (`global_step=2276`)
