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

The closest completed measurement is exp_01's full 6,337-query Vanilla evaluation with eight contexts, single-step sampling, VAE decoding, and AGREE/physical metrics. The five runs took 843, 875, 886, 889, and 881 seconds; the median is 881 seconds, or **7.193 generated RIR/s**. Here exp_01's `K=8` means eight contexts, not the localization score's stochastic sample count.

This is an **uncached historical whole-pipeline anchor**, not a cache-enabled localization benchmark. The planned cache should remove repeated context and receiver-candidate conditioner work, so these hours must not be presented as an exact runtime for code that has not yet been implemented.

The geometry counts make the cache impact concrete: an uncached eight-context path would execute about **80,026,434 geometry ViT forwards**, whereas the frozen context/source caches require **1,009,424**, a 79.28x reduction in that component. Diffusion, VAE decode and AGREE scoring still run once per query-candidate/sample, so total runtime does not receive a 79x speedup.

| Score samples (`K_score`) | Vanilla historical 7.193 RIR/s, one arm | Optimistic 10 RIR/s, one arm | 168 GPU-hour gate |
|---:|---:|---:|---:|
| 1 | 343.4 GPU-h | 247.0 GPU-h | FAIL at both anchors |
| 2 | 686.8 GPU-h | 494.0 GPU-h | FAIL |
| 4 | 1,373.5 GPU-h | 988.0 GPU-h | FAIL |

To fit 168 GPU-hours, the cache-enabled engine must sustain at least **14.702 RIR/s** for `K_score=1`, 29.404 RIR/s for `K_score=2`, or 58.808 RIR/s for `K_score=4`, end to end. Multiple GPUs can reduce wall-clock time but not total GPU-hours.

Exp_01's one-context runs provide a rough cache-oriented bracket, because they remove seven of the eight repeated context ViT passes but still do not reuse the source branch. Their median is 10.888 RIR/s (226.8 hours here), while their two fastest runs are 15.686 and 15.381 RIR/s (157.5 and 160.6 hours). Therefore `K_score=1` is **plausible but not certified** under 168 hours; the required 14.702 RIR/s lies inside the observed machine-load range. `K_score>=2` is not plausible without a much larger end-to-end speedup.

FA-BF must receive a separate matched cache-enabled probe after Vanilla: its C4 frame-averaged conditioner executes four angle branches, and prior evidence shows it is materially slower without effective caching. Applying the Vanilla rate to FA-BF would therefore be unjustified.

The table is therefore **Vanilla-only**. Running Vanilla plus FA-BF doubles the generation count before accounting for FA-BF's additional conditioning cost; it cannot be represented by simply doubling the Vanilla GPU-hour estimate until the FA-BF cache probe exists.

## Artifact size

- Float32 scores only: 33.92 MiB (`K_score=1`), 67.84 MiB (`K_score=2`), 135.68 MiB (`K_score=4`), before metadata.
- Saving every 10,240-sample float32 generated waveform would require 339.20 GiB, 678.39 GiB, or 1,356.78 GiB respectively. The later engine must stream scores/features and must not persist the complete waveform cartesian product.

## Gate conclusion

The geometry gate now passes, but the current evidence does not certify any score-sample rung under 168 GPU-hours. `K_score=1` is the only plausible rung and needs a cache-enabled no-quality throughput probe to demonstrate at least 14.702 RIR/s. Candidate spacing, queries, and masks must not be changed in response to runtime without a separately documented protocol decision.
