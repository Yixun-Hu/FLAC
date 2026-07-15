# Plan — exp_05_cylvit_yaw_ablation

**Author:** Codex + Yixun discussion · **Date:** 2026-07-08  
**Status:** planned, not yet executed

## Goal

Test whether replacing the plain geometry ViT with an azimuth-equivariant cylindrical ViT reduces yaw-induced degradation in FLAC-style RIR prediction.

The controlled comparison is:

- **Baseline:** SimpleViT geometry encoder.
- **Candidate:** CylindricalViT from `test/flac_codec/cyl_vit.py`.
- **Shared:** same FLAC latent generator, same VAE/pretransform, same context audio path, same train/eval data, same seeds, same metric suite.

The main question is not only whether CylViT improves absolute Table-1 metrics, but whether it makes the metrics more stable under yaw rotations:

```text
Does replacing vanilla geometry ViT with cylindrical-equivariant ViT reduce yaw-induced metric degradation under the same FLAC training/evaluation protocol?
```

## Background

`exp_01_reproduce_flac_table1` reproduced the released FLAC Table 1 path using:

- `eval_FLAC.py`
- `src/configs/model_configs/FLAC/AR/FLAC_AR.json`
- `weights/FLAC/FLAC_EMA.ckpt`
- `--cfg-scale 1.0 --steps 1`
- metrics: T60, C50, EDT, FD_G, R@1, R@5, R@10

That release path uses DINOv3-based `ViTCoordinates` conditioners, not the SimpleViT fallback. This experiment intentionally defines a new A/B line: SimpleViT versus CylViT as the geometry encoder. The comparison should therefore be treated as a controlled architecture ablation, not as a direct reuse of the released FLAC checkpoint without retraining.

## Hypotheses

### H1: Encoder-level invariance

CylViT should be nearly invariant at patch-grid yaw rotations, because `512 / 32 = 16` azimuth patches and the architecture uses:

- per-column cylindrical gauge alignment,
- elevation-only absolute position encoding,
- circular relative azimuth bias,
- mean pooling over the token grid.

SimpleViT should show larger embedding changes under the same yaw rotation.

### H2: Prediction-level yaw stability

After training, CylViT-FLAC should have smaller metric deltas under yaw rotation:

```text
metric_delta(angle) = metric(angle) - metric(0deg)
```

The strongest evidence would be smaller angle-to-angle variance and smaller worst-case degradation for T60, C50, EDT, FD_G, and retrieval metrics.

### H3: No unacceptable clean-performance regression

CylViT should not substantially degrade the unrotated `rotate_deg=0` performance relative to SimpleViT. If clean performance drops but yaw stability improves, the result should be reported as a robustness/accuracy tradeoff rather than a clear win.

## Experimental Design

### Phase 0 — Code integration sanity

Add CylViT to the FLAC model path without changing the evaluation protocol.

Planned code changes:

- Move or copy `test/flac_codec/cyl_vit.py` into `FLAC/src/models/cyl_vit.py`.
- Update `FLAC/src/models/conditioners.py` so the geometry conditioner can select:
  - `arch = "simple_vit"`
  - `arch = "cyl_vit"`
  - existing DINOv3 behavior remains unchanged for release FLAC configs.
- Add two model configs:
  - `src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json`
  - `src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json`
- Keep both configs identical except for the geometry encoder architecture.

Acceptance checks:

- Existing `FLAC_AR.json` still selects DINOv3 and can load `FLAC_EMA.ckpt`.
- SimpleViT config builds without touching CylViT.
- CylViT config builds and returns conditioning tensors with the same shapes as SimpleViT.

### Phase 1 — Encoder-only yaw invariance test

Before training, test the structural property directly on cached or dataset-derived geometry samples.

For each geometry sample:

```text
geom = source_or_context_pose - depth_coord
shape = [3, 256, 512]
```

Compare:

```text
E(geom)
E(yaw_rotate(geom, angle))
```

for both:

```text
E = SimpleViT geometry encoder
E = CylViT geometry encoder
```

Angles:

- Patch-grid rotations:
  - `22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5`
- Sub-patch rotations:
  - `5, 10, 15`

Metrics:

- max absolute embedding difference,
- L2 distance,
- cosine distance,
- optionally token-level deviation before pooling.

Expected:

- CylViT near-zero deviation for patch-grid rotations.
- CylViT smaller deviation than SimpleViT for sub-patch rotations, though not exact.

### Phase 2 — Training comparison

Train two matched models:

```text
SimpleViT-FLAC
CylViT-FLAC
```

Controlled variables:

- same training split,
- same validation/eval split,
- same K setup,
- same sample rate and waveform length,
- same VAE/pretransform architecture,
- same DiT architecture,
- same optimizer and LR schedule,
- same random seeds,
- same number of steps/epochs,
- same yaw augmentation policy, if enabled.

Architecture variable:

```text
source_vit / context_poses_vit geometry encoder only
```

Training strategy options:

1. **Short-run trend test first**
   - Train on a subset or smaller schedule.
   - Goal: verify that CylViT can train and yaw metric deltas move in the expected direction.

2. **Full protocol second**
   - Run the full train/eval protocol only if the short-run test is promising.

Initialization options:

- **Most fair but expensive:** train SimpleViT-FLAC and CylViT-FLAC from comparable initialization.
- **More practical first pass:** reuse compatible VAE/pretransform components and retrain the geometry conditioner plus DiT. This must be documented as adaptation, not a pure released-checkpoint comparison.

Initial recommendation:

Run the short-run trend test first. Do not spend full training budget until:

- Phase 1 confirms CylViT structural invariance.
- A small training run confirms the model learns and eval metrics are not broken.

### Phase 3 — Exp01/Table1 metric evaluation

Use the exp01 metric suite:

- T60,
- C50,
- EDT,
- FD_G,
- R@1,
- R@5,
- R@10.

Evaluate both trained models under:

```text
K = 1
K = 8
```

Seeds:

```text
42, 43, 44, 45, 46
```

Clean eval:

```text
rotate_deg = 0
```

Yaw stress eval:

```text
rotate_deg = 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180,
             202.5, 225, 247.5, 270, 292.5, 315, 337.5
```

Sub-patch stress eval:

```text
rotate_deg = 5, 10, 15
```

Primary reporting:

- clean metric at `0deg`,
- metric delta per angle,
- worst-case yaw delta,
- mean absolute yaw delta,
- standard deviation over yaw angles,
- mean and std over seeds.

The comparison table should report both absolute metrics and yaw deltas, because a method can be more yaw-stable but worse in clean accuracy.

## Proposed Result Tables

### Table A — Encoder-only invariance

Columns:

```text
model | angle | max_abs | l2 | cosine_distance | notes
```

Rows:

```text
SimpleViT, CylViT
```

### Table B — Clean exp01 metrics

Columns:

```text
model | K | T60 | C50 | EDT | FD_G | R@1 | R@5 | R@10
```

Rows:

```text
SimpleViT K=1
CylViT K=1
SimpleViT K=8
CylViT K=8
```

### Table C — Yaw robustness summary

Columns:

```text
model | K | metric | mean_abs_delta | worst_delta | angle_std | seed_std
```

### Table D — Angle sweep

Columns:

```text
model | K | angle | T60_delta | C50_delta | EDT_delta | FD_G_delta | R@1_delta | R@5_delta | R@10_delta
```

## Success Criteria

A strong positive result requires:

- CylViT has clearly lower encoder-level yaw deviation than SimpleViT.
- CylViT has lower prediction metric yaw deltas on most primary metrics.
- CylViT does not substantially degrade `rotate_deg=0` metrics.

A mixed result is still useful if:

- encoder invariance improves but final metrics do not,
- final yaw metrics improve only for patch-grid rotations,
- CylViT improves yaw stability but loses clean accuracy.

In that case, the analysis should identify whether the bottleneck is likely:

- the context audio path,
- the DiT generator,
- training schedule,
- lack of yaw augmentation,
- sub-patch rotation mismatch,
- or parameter/optimization differences.

## Risks and Controls

- **Checkpoint compatibility:** the released `FLAC_EMA.ckpt` is tied to DINOv3 conditioner weights. Do not claim a direct plug-in replacement unless the model is retrained or carefully adapted.
- **Compute cost:** full FLAC training may be expensive. Run Phase 1 and a short-run Phase 2 first.
- **Architecture mismatch:** CylViT returns token sequences like SimpleViT, while the DINOv3 path uses `pooler_output`. Keep SimpleViT and CylViT on the same conditioner branch.
- **Yaw implementation:** yaw rotation must roll panorama columns and rotate x/y channels consistently.
- **Evaluation naming:** each run must have a unique `eval_name` including model, K, seed, and angle to avoid metric JSON overwrites.
- **Metric interpretation:** AGREE is an evaluation model for FD_G/retrieval, not the predictor.

## Immediate Next Steps

1. Add `FLAC/src/models/cyl_vit.py`.
2. Extend `FLAC/src/models/conditioners.py` with architecture selection for SimpleViT/CylViT.
3. Add SimpleViT and CylViT model configs.
4. Write a small encoder-only yaw invariance script.
5. Run Phase 1 on a few geometry samples.
6. If Phase 1 passes, run a short training comparison.
7. If the short comparison is promising, launch the full exp01-style evaluation with yaw stress angles.

