# Params — exp_02_yaw_noninvariance

| Parameter | Value |
|---|---|
| Checkpoint | `weights/FLAC/FLAC_EMA.ckpt` (released, frozen — no training in this exp) |
| Model config | `src/configs/model_configs/FLAC/AR/FLAC_AR.json` |
| Dataset config | `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json` (K=1, full 6337-item unseen split) |
| Rotations (deg) | baseline (flag absent), 0 (sanity), 90, 180, 270 |
| cfg-scale / steps | 1.0 / 1 |
| batch-size / workers | 32 / 4 (per Yixun's spec) |
| Seed | 42, identical across all runs |
| store_predictions | yes (for Metric-1 offline comparison) |
| GPU | CUDA_VISIBLE_DEVICES=0 |
| Code state | pristine commit `0bd5da0` for eval; comparison script in exp folder only |
| Launched | 2026-07-04_20:49:36 (log `yaw_noninvariance_2026-07-04_20:49:36.log`) |

Rotation semantics: `rotate_scene_metadata` rolls the equirectangular depth panorama by the quantized column count and rotates its per-pixel 3D vectors and all pose vectors about z by the matching angle; context audio and GT RIR are untouched (physically consistent: GT is invariant to whole-scene yaw).
