# Phase 0/1 Results — exp_05_cylvit_yaw_ablation

**Date:** 2026-07-08  
**Status:** Phase 0 and Phase 1 completed on synthetic geometry.

## Phase 0 — Code Integration

Implemented:

- Added `src/models/cyl_vit.py` from `test/flac_codec/cyl_vit.py`.
- Extended `src/models/conditioners.py` so `ViTCoordinates` supports:
  - `arch: "dino"` (existing release FLAC behavior),
  - `arch: "simple_vit"`,
  - `arch: "cyl_vit"`.
- Added optional `token_pool` for ViT-style geometry conditioners:
  - default: `"linear"` to preserve old behavior,
  - experiment configs: `"mean"` to match CylViT's invariance assumption.
- Added A/B configs:
  - `src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json`
  - `src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json`
- Added encoder-only yaw probe:
  - `worklog/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py`

Validation:

```bash
python3 -m json.tool src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json
python3 -m json.tool src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json
python3 -m py_compile \
  src/models/cyl_vit.py \
  src/models/conditioners.py \
  worklog/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py
```

All checks passed.

Build check:

```bash
../venv/bin/python - <<'PY'
import json
from src.models.factory import create_model_from_config

for path in [
    'src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json',
    'src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json',
]:
    cfg = json.load(open(path))
    model = create_model_from_config(cfg)
    print(path)
    print(type(model.conditioner.conditioners['source_vit'].vit).__name__)
    print(model.conditioner.conditioners['source_vit'].token_pool)
PY
```

Observed:

- `FLAC_AR_SimpleViT.json` builds `SimpleViT`, `token_pool=mean`.
- `FLAC_AR_CylViT.json` builds `CylindricalViT`, `token_pool=mean`.
- The release `FLAC_AR.json` remains unchanged and still uses DINOv3 when selected.

## Phase 1 — Encoder-Only Yaw Invariance

Command:

```bash
../venv/bin/python worklog/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py \
  --device cpu \
  --include-context \
  --out-prefix worklog/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance_synthetic_with_context
```

Outputs:

- `encoder_yaw_invariance_synthetic_with_context.csv`
- `encoder_yaw_invariance_synthetic_with_context.json`

The probe used deterministic synthetic metadata with:

- `depth`: `[3, 256, 512]`
- `source_vit`: `[1, 3]`
- `context_poses_vit`: `[4, 3]`

Yaw rotations use the same group action as `eval_FLAC.py`: roll panorama columns and rotate x/y vector channels plus pose vectors together.

## Summary

Combined patch-grid and sub-patch angles:

| model | key | max_abs | mean_l2 | mean_cosine_distance |
|---|---:|---:|---:|---:|
| SimpleViT | source_vit | 2.6346e+00 | 6.5464e+00 | 5.9318e-01 |
| SimpleViT | context_poses_vit | 3.0979e+00 | 1.2175e+01 | 4.4897e-01 |
| CylViT | source_vit | 1.6251e-01 | 1.4568e-01 | 2.4327e-03 |
| CylViT | context_poses_vit | 1.8670e-01 | 2.5906e-01 | 1.5679e-03 |

Patch-grid angles only (`22.5 * n` degrees):

| model | key | patch max_abs max | patch max_abs mean |
|---|---:|---:|---:|
| SimpleViT | source_vit | 2.6346e+00 | 1.5766e+00 |
| SimpleViT | context_poses_vit | 3.0979e+00 | 2.0444e+00 |
| CylViT | source_vit | 1.7881e-07 | 1.5299e-07 |
| CylViT | context_poses_vit | 2.3842e-07 | 1.9073e-07 |

Sub-patch angles only (`5, 10, 15` degrees):

| model | key | sub-patch max_abs max | sub-patch max_abs mean |
|---|---:|---:|---:|
| SimpleViT | source_vit | 2.1482e-01 | 1.7445e-01 |
| SimpleViT | context_poses_vit | 3.2765e-01 | 2.5099e-01 |
| CylViT | source_vit | 1.6251e-01 | 1.5700e-01 |
| CylViT | context_poses_vit | 1.8670e-01 | 1.7346e-01 |

## Interpretation

Phase 1 supports the structural hypothesis:

- CylViT is numerically invariant at patch-grid yaw rotations.
- SimpleViT is not yaw-invariant under the same rotations.
- CylViT is not exactly invariant for sub-patch rotations, which matches the design expectation because exact equivariance is at the 32-column patch grid.

The next step is Phase 2: train SimpleViT-FLAC and CylViT-FLAC under matched settings, then evaluate clean metrics and yaw-stress metric deltas with the exp01/Table1 metric suite.
