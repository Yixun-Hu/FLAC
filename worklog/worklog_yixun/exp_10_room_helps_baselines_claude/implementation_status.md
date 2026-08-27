# exp_10 implementation status

## Implemented

### Shared protocol

- Exact nested `K_ctx={1,8}` prefix selection.
- Baseline waveform contract: mono `float32`, 10,240 samples, finite values, existing FLAC clamp.
- Existing frozen AGREE encoding/cosine scorer for Few-ShotRIR; exact complex DFT/Room Helps OMP selector for FEM.
- Stable candidate ordering/tie break inherited from exp_09.

### Few-ShotRIR-Waveform

- Geometry-only visual branch using the same FLAC depth-derived tensor; no RGB input exists in the model API.
- Acoustic-context encoder, FLAC-compatible coordinate Fourier features, set-style transformer memory, query decoder, and direct 10,240-sample waveform decoder.
- Reconstruction-only waveform + multi-resolution STFT + Schroeder energy-decay loss.
- Variable context training (`K=1..8`), from-scratch model/training factories, and AR train config with crop/time-shift/noise augmentation disabled.
- Strict checkpoint loader and deterministic candidate batching for localization.

### FEM-Sabine

- AR-parity context RT60 estimation (`decay_db=20`) with median aggregation and fail-closed invalid handling.
- Uniform Sabine absorption, zero-phase reflection, and normalized impedance mapping.
- First-order tetrahedral stiffness, consistent mass, and exterior boundary-mass assembly.
- Face-connected/manifold tetrahedral-air-domain validation, barycentric receiver/candidate interpolation, reciprocity-based sparse Helmholtz solves, and per-frequency relative residuals.
- Exact 80–300 Hz bins on the 22,050 Hz / 10,240-sample DFT grid.
- Room Helps §3.3 pulse-source adaptation: vertically stacked complex frequency dictionary, one frequency-independent sparse source vector, complex OMP, stable one-support selection, least-squares coefficient, and relative residual diagnostics.
- FEM selection bypasses AGREE and is invariant to a common nonzero dictionary gain; optional conjugate-symmetric IFFT remains diagnostic only.
- Fail-closed `h_max <= 0.22 m` gate plus mesh quality, volume, surface area, T60, Sabine, and solver audit fields.
- Versioned tetrahedral NPZ schema whose source OBJ SHA must match the exp_09 geometry audit.

### Execution

`localize_baseline.py` runs either method against the frozen exp_09 context, geometry, pilot, and candidate identities. Few-ShotRIR additionally pins AGREE; FEM pins `room_helps_pulse_stacked_omp` and never loads AGREE. Query outputs are atomic, content-hashed, and resume-safe. Each deterministic method writes one generic candidate-score column for `K_ctx=1` and one for `K_ctx=8`.

## Inputs still required before a real run

1. A trained Few-ShotRIR-Waveform checkpoint selected without unseen localization metrics.
2. Deterministically repaired, audited, face-connected tetrahedral air meshes for the 16 available official room OBJs. The official OBJs themselves are non-watertight and non-edge-manifold, so they are not accepted directly.
3. A tetrahedral mesh manifest:

   ```json
   {
     "schema_version": 1,
     "rooms": {
       "Apartments_idx_42": {
         "path": "tetra_meshes/Apartments_idx_42.npz",
         "npz_sha256": "<sha256>"
       }
     }
   }
   ```

4. A small-room compute/residual/phase/OMP probe before any multi-room FEM sweep.

## Entry points

Training configuration:

```bash
python train.py \
  --model-config src/configs/model_configs/baselines/FewShotRIR_Waveform_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_few_shot_waveform.json \
  --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json \
  --max-steps <preregistered_budget> \
  --logger none \
  --save-dir <inside_worktree>
```

Localization after the gated artifacts exist:

```bash
python localize_baseline.py --method few_shot_rir_waveform \
  --model-config <model_config.json> --ckpt-path <checkpoint.ckpt> \
  --agree-ckpt <agree.ckpt> --context-manifest <contexts.json> \
  --geometry-audit <geometry_audit.json> --pilot-manifest <pilot.json> \
  --dataset-root <AcousticRooms> --output-dir <inside_worktree>
```

```bash
python localize_baseline.py --method fem_sabine \
  --tetra-mesh-manifest <tetra_meshes.json> \
  --context-manifest <contexts.json> \
  --geometry-audit <geometry_audit.json> --pilot-manifest <pilot.json> \
  --dataset-root <AcousticRooms> --output-dir <inside_worktree>
```

These are interfaces, not authorization to launch the expensive jobs.
