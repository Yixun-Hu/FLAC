# Exp_09 GPU cost analysis

This analysis applies to the approved mesh-available scope only: 5,337 queries in 16 rooms. `ListeningRoom_idx_2` (1,000 queries) remains excluded because its official mesh is absent.

## Geometry work count

- Pre-registered branch selected by the no-new-unwinnable rule: context-derived z-band.
- Query-candidate generations/scores per stochastic sample: **6,094,936**.
- Unique receiver-candidate source branches after caching: **636,963** (9.569 query-pair uses per cached branch on average).
- Query-invariant context branches: 5,337 queries, or 42,696 context ViT forwards at eight contexts/query.
- Geometry gate: **FAIL**. Thirteen unique source anchors in seven rooms fail the declared mesh-inside/surface-clearance predicate; eight unique receivers in five rooms fail the mesh occupancy predicate. No model generation is permitted while this gate is red.

## Measured throughput anchor

The closest completed measurement is exp_01's full 6,337-query Vanilla evaluation with eight contexts, single-step sampling, VAE decoding, and AGREE/physical metrics. The five runs took 843, 875, 886, 889, and 881 seconds; the median is 881 seconds, or **7.193 generated RIR/s**. Here exp_01's `K=8` means eight contexts, not the localization score's stochastic sample count.

This is an **uncached historical whole-pipeline anchor**, not a cache-enabled localization benchmark. The planned cache should remove repeated context and receiver-candidate conditioner work, so these hours must not be presented as an exact runtime for code that has not yet been authorized or implemented.

| Score samples (`K_score`) | Historical 7.193 RIR/s | Optimistic 10 RIR/s check | 168 GPU-hour gate |
|---:|---:|---:|---:|
| 1 | 235.4 GPU-h | 169.3 GPU-h | FAIL at both anchors |
| 2 | 470.7 GPU-h | 338.6 GPU-h | FAIL |
| 4 | 941.5 GPU-h | 677.2 GPU-h | FAIL |

To fit 168 GPU-hours, the cache-enabled engine must sustain at least **10.078 RIR/s** for `K_score=1`, 20.155 RIR/s for `K_score=2`, or 40.310 RIR/s for `K_score=4`, end to end. Multiple GPUs can reduce wall-clock time but not total GPU-hours.

FA-BF must receive a separate matched cache-enabled probe after Vanilla: its C4 frame-averaged conditioner executes four angle branches, and prior training/evaluation evidence shows it is materially slower without effective caching. Applying the Vanilla rate to FA-BF would therefore be unjustified.

## Artifact size

- Float32 scores only: 23.25 MiB (`K_score=1`), 46.50 MiB (`K_score=2`), 93.00 MiB (`K_score=4`), before metadata.
- Saving every 10,240-sample float32 generated waveform would require 232.50 GiB, 465.01 GiB, or 930.01 GiB respectively. The later engine must stream scores/features and must not persist the complete waveform cartesian product.

## Gate conclusion

No score-sample rung is currently approved. Geometry is already blocked by the real-anchor failure. Even after that protocol issue is resolved, the existing evidence does not certify any rung under 168 GPU-hours: `K_score=1` needs a cache-enabled no-quality throughput probe to demonstrate at least 10.078 RIR/s. Candidate spacing, queries, and masks must not be changed in response to this estimate without a separately documented protocol decision.
