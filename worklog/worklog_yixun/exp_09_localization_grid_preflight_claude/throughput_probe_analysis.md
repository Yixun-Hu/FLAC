# Exp_09 real cached throughput probe

Date: 2026-08-21. Device: NVIDIA RTX A6000. This probe read no localization quality value and saved no candidate score.

## Frozen inputs

| Input | SHA-256 |
|---|---|
| Vanilla `P1_40k_clean_hybrid_EMA.ckpt` | `da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643` |
| FA-BF `BF_40k_clean_hybrid_EMA.ckpt` | `0f61277f45367fb0e75d7ee70c0627b8948a23eb62be58f13fce91662551557a` |
| AGREE `AGREE_fullAR.pt` | `3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787` |
| `FLAC_AR.json` | `f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9` |
| Context manifest | `b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58` |
| Geometry audit | `ae09d9cf9416866d09dea498a1f8467e952866db8b1c914ed0bea6a75e06cf9a` |

`FLAC_AR.json` is required: strict checkpoint loading reports zero missing/unexpected keys. `FLAC_AR_InContext.json` is rejected because its 256-dimensional global embedding does not match the checkpoint's 512-dimensional layer. The gated DINO architecture is instantiated offline from the checked-in byte-equivalent config, then every FLAC/AGREE tensor is replaced by the strict full checkpoint load.

## Final measurements

Each final arm measured 512 real candidates, source batch 64, generation batches 128/256/512, one warm-up per size, and eight timed batches per size.

| Arm | Winning batch | Generation + decode + AGREE | Source cache | Peak memory | `K=8`, startup included |
|---|---:|---:|---:|---:|---:|
| Vanilla | 512 | 141.732/s | 713.546/s | 7.186 GB | 140.05 GPU-h |
| FA-BF C4 | 512 | 139.099/s | 145.935/s | 7.188 GB | 144.54 GPU-h |
| **Serial total** |  |  |  |  | **284.59 GPU-h / 11.86 days** |

Yixun's 2026-08-21 override fixes three reported settings: `K∈{1,4,8}`. They use one nested sequence of eight deterministic samples, so K=1 is prefix 0, K=4 is prefix 0–3, and K=8 is prefix 0–7. Reporting all three therefore costs one K=8 execution.

Standalone two-arm projections for K=1/4/8 are 38.31/143.86/284.59 GPU-hours. The slowest individual timed batch produces a K=8 conservative total of 311.52 GPU-hours / 12.98 days. Adding an operational 10% reserve gives 342.67 GPU-hours / 14.28 days. With two equivalent free A6000s, nominal K=8 wall time is 144.54 hours / 6.02 days.

The earlier `4 -> 2 -> 1` ladder is superseded. K=8 exceeds the former 168-hour full-execution ceiling, so the setting change is frozen before quality but the full run still needs explicit compute approval.

## Cache integrity

- Vanilla: `context_poses_vit/context_poses/context_audio` once per query; `source/source_vit` once per receiver-candidate.
- FA-BF: query cache is `context_poses_vit/context_audio`; receiver-candidate cache is `source/source_vit`; candidate-relative cylindrical `context_poses.dphi` is recomputed inside the timed generation batch.
- Shape-matched cached and uncached tokens are bit-identical for every branch in both final probes.
- Full-vectorized mixed-precision comparison is not bit-identical because CUDA kernels change with batch shape; the recorded maximum absolute difference is `0.00390625`, with masks bit-identical.
- Deterministic noise uses one counter-derived seed per query/candidate/sample and is invariant to batch partitioning.

## Evidence

- `throughput_probe_vanilla_cached_final.json`: internal canonical SHA `e60f8ead63b0fcf8c8522d7adafc84484852324d832c86a7c68a47bbcc979ca4`; file SHA-256 `e15dc30dfc3aca4609b05fa26831192b8df707ab4cd2b26fae39741bf2f93db7`.
- `throughput_probe_fa_bf_cached_run2.json`: internal canonical SHA `f59434cd31abe62fb3b055bd68841da1d7f9cd6322283077fc48d7fd993c2532`; file SHA-256 `ed5611c97a9338970d6f6eade4d54e2056c859ca658a25273c77adc07c495828`.
- `throughput_projection_k1_k4_k8.json`: deterministic CPU-only reprojection of the two final timing files under the later `K∈{1,4,8}` decision; file SHA-256 `5a0d2b08e1be28b7a49cbf478ed18b0c8d6205452dbfa3a487ab79060c29864a`.
- Earlier batch-32/64/128 and independent Vanilla repeats remain as no-quality repeatability evidence. The final same-engine files above control the estimate.

The remaining uncertainty is orchestration rather than GPU kernel throughput: a bounded one-room smoke must still validate candidate iteration, streamed aggregation, tail batches, output/resume hashes, and wall-clock overhead before the full quality run.
