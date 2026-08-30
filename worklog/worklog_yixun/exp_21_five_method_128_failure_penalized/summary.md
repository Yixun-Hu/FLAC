# Five-method 128-query result with FEM coverage penalties

This end-to-end table evaluates the same 16 rooms x 8 frozen queries for all five methods. The 16 queries that fail the FEM strict-coverage gate are retained: their success indicators are zero and their localization error is the maximum true-source distance over the frozen candidate set.

FEM coverage is **112/128 (87.5%)**. The previous 112-query conditional result remains a diagnostic and is not used as this full-scope table.

## Matched localization metrics

`Localization Error [m]` is the median over all 128 queries.

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Coverage |
|---|---:|---:|---:|---:|---:|
| Vanilla FLAC | 0.933 | 18.0% | 53.1% | 32.0% | 100.0% |
| OrbitRIR (FA-BF FLAC) | 0.893 | 18.0% | 53.1% | 35.9% | 100.0% |
| Yaw-Augmented FLAC | 0.916 | 16.4% | 53.1% | 32.8% | 100.0% |
| Few-ShotRIR | 1.558 | 4.7% | 22.7% | 13.3% | 100.0% |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 1.081 | 25.8% | 48.4% | 36.7% | 87.5% |

## Localization-error distribution

| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ |
|---|---:|---:|---:|
| Vanilla FLAC | 1.727 | 0.933 | 2.965 |
| OrbitRIR (FA-BF FLAC) | 1.982 | 0.893 | 4.465 |
| Yaw-Augmented FLAC | 1.744 | 0.916 | 3.941 |
| Few-ShotRIR | 3.024 | 1.558 | 6.490 |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 2.381 | 1.081 | 5.034 |

## Interpretation boundary

The failure penalty is deliberately pessimistic but finite and query-scale-aware. Ground truth is used only to calculate an evaluation error after the method has failed to produce a valid full-candidate prediction. Success flags are forced to zero regardless of the numerical penalty value.

The four learned methods use their real predictions on all 128 queries. The three FLAC rows use the registered primary `K_gen=1`; Few-ShotRIR uses `K_ctx=8`; FEM uses eight acoustic context RIRs and Room-Helps one-support OMP.

This is a FEM--OMP result, not a FEM--AGREE result.
