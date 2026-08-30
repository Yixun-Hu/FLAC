# Five-method 64-query result with FEM coverage penalties

This end-to-end table evaluates the same 16 rooms x 4 frozen queries for all five methods. The 7 queries that fail the FEM strict-coverage gate are retained: their success indicators are zero and their localization error is the maximum true-source distance over the frozen candidate set.

FEM coverage is **57/64 (89.1%)**. The previous 112-query conditional result remains a diagnostic and is not used as this full-scope table.

## Matched localization metrics

`Localization Error [m]` is the median over all 64 queries.

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Coverage |
|---|---:|---:|---:|---:|---:|
| Vanilla FLAC | 1.009 | 15.6% | 50.0% | 23.4% | 100.0% |
| OrbitRIR (FA-BF FLAC) | 0.928 | 28.1% | 53.1% | 40.6% | 100.0% |
| Yaw-Augmented FLAC | 0.892 | 21.9% | 54.7% | 34.4% | 100.0% |
| Few-ShotRIR | 1.517 | 3.1% | 25.0% | 14.1% | 100.0% |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 1.130 | 28.1% | 48.4% | 35.9% | 89.1% |

## Localization-error distribution

| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ |
|---|---:|---:|---:|
| Vanilla FLAC | 1.748 | 1.009 | 2.972 |
| OrbitRIR (FA-BF FLAC) | 1.816 | 0.928 | 4.004 |
| Yaw-Augmented FLAC | 1.793 | 0.892 | 3.300 |
| Few-ShotRIR | 3.109 | 1.517 | 6.366 |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 2.484 | 1.130 | 5.113 |

## Interpretation boundary

The failure penalty is deliberately pessimistic but finite and query-scale-aware. Ground truth is used only to calculate an evaluation error after the method has failed to produce a valid full-candidate prediction. Success flags are forced to zero regardless of the numerical penalty value.

The four learned methods use their real predictions on all 64 queries. The three FLAC rows use the registered primary `K_gen=1`; Few-ShotRIR uses `K_ctx=8`; FEM uses eight acoustic context RIRs and Room-Helps one-support OMP.

This is a FEM--OMP result, not a FEM--AGREE result.
