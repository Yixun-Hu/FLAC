# Exp_09 GPU cost analysis — 0.20 m geometry

This analysis applies to the approved mesh-available scope: 5,337 queries in 16 rooms. `ListeningRoom_idx_2` (1,000 queries) remains excluded because its official mesh is absent.

## Geometry work count

- Physical-validity backend: deterministic 31-direction odd-ray-parity majority voting, robust to the non-watertight AcousticRooms room shells.
- Source surface-clearance prior: **0.20 m** with `eps=1e-4 m`.
- Pre-registered branch selected by the no-new-unwinnable rule: context-derived z-band.
- Query-candidate generations/scores per stochastic sample and per model arm: **8,891,826**.
- Unique receiver-candidate source branches after caching: **966,728** (9.198 query-pair uses per cached branch on average).
- Query-invariant context branches: 5,337 queries, or 42,696 context ViT forwards at eight contexts/query.
- Geometry gate: **PASS**. Every real metadata source and receiver anchor in all 16 rooms passes ray-parity validity; every source also passes the 0.20 m clearance. All 5,337 candidate sets are nonempty and finite, and zero have `e_oracle > 0.5 m`.

## Measured throughput anchor

Yixun authorized a no-quality probe after G1. It ran the actual frozen checkpoints, one-step rectified-flow sampler, VAE decoder, exact generated-audio clamp, shared `Retrieval.compute_audio_features` AGREE path, deterministic per-candidate noise, and the real branch caches on one NVIDIA RTX A6000. No localization score or winner was saved.

The final matched probes each used 512 real candidates from the highest-count audited query, source-cache batches of 64, generation batches 128/256/512, one warm-up per size, and eight timed batches per size (7,168 timed generated RIR scores per arm). Both FLAC checkpoints and AGREE loaded with zero missing/unexpected keys. The main evidence files are `throughput_probe_vanilla_cached_final.json` and `throughput_probe_fa_bf_cached_run2.json`.

FA-BF uses the released dependency split rather than pretending every context token is query-only: `source/source_vit` are receiver-candidate cached; `context_poses_vit/context_audio` are query cached; candidate-relative cylindrical `context_poses.dphi` is recomputed inside every timed generation batch. The resulting cached tokens are bit-identical to a shape-matched uncached reference. The full vectorized mixed-precision diagnostic is also retained and reports the expected batch-shape rounding drift (maximum `0.00390625`), rather than silently calling it bit-identical.

| Component / final batch-512 result | Vanilla 40k | FA-BF 40k |
|---|---:|---:|
| Generated RIR + VAE + AGREE rate | 141.732/s | 139.099/s |
| Receiver-candidate source-cache rate | 713.546/s | 145.935/s |
| Query-context cache time, measured mean | 0.14369 s | 0.40700 s |
| Batch-512 peak allocated memory | 7.186 GB | 7.188 GB |
| Batch-512 within-run CV | 0.31% | 0.17% |

The projections below include query audio/depth I/O, query-context cache, observed-RIR AGREE encoding, receiver-candidate cache, generated scoring, and one model startup. They use the exact G1 counts above.

| Score samples (`K_score`) | Vanilla | FA-BF | Serial two-arm total |
|---:|---:|---:|---:|
| 1 | 18.06 GPU-h | 20.25 GPU-h | 38.31 GPU-h |
| 2 | 35.49 GPU-h | 38.00 GPU-h | 73.49 GPU-h |
| **4** | **70.34 GPU-h** | **73.52 GPU-h** | **143.86 GPU-h (5.99 days)** |

As a measured lower-throughput bound, replacing each arm's winning rate by its slowest individual timed batch across all probe sizes gives 76.48 GPU-hours for Vanilla and 80.84 GPU-hours for FA-BF at `K=4`, or **157.32 GPU-hours (6.56 days)** serial. A separate 10% operational reserve yields **173.05 GPU-hours (7.21 days)**; this reserve covers orchestration, tail batches, filesystem contention, and resume overhead and is not used to change the pre-registered K ladder.

With two equivalent free A6000s, the nominal wall time is governed by FA-BF at about **73.52 hours (3.06 days)**; the measured conservative bound is **80.84 hours (3.37 days)**. GPU-hours remain additive.

## Artifact size

- Float32 scores only: 33.92 MiB (`K_score=1`), 67.84 MiB (`K_score=2`), 135.68 MiB (`K_score=4`), before metadata.
- Saving every 10,240-sample float32 generated waveform would require 339.20 GiB, 678.39 GiB, or 1,356.78 GiB respectively. The later engine must stream scores/features and must not persist the complete waveform cartesian product.

## Gate conclusion

The cache-enabled no-quality gate certifies `K_score=4`: nominal serial two-arm compute is 143.86 GPU-hours and the slowest-measured-batch projection is 157.32 GPU-hours, both below the pre-registered 168-hour ceiling. `K=4` is therefore frozen globally before any localization quality value is read. Candidate spacing, queries, masks, `tau`, and K remain unchanged.

This is a component-complete GPU projection, not yet a full-room wall-clock smoke. The later bounded one-room smoke must still validate job orchestration, candidate iteration, streamed aggregation, and resume behavior. The 10% reserve is the appropriate scheduling number until that smoke completes.
