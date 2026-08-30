# Five-method 64-query result with FEM deterministic random fallback

This end-to-end table evaluates the same 16 rooms x 4 frozen queries for all five methods. The 7 queries that fail the FEM strict-coverage gate are retained: each uses the registered deterministic uniform random candidate fallback (`seed=42`), and all success indicators are calculated normally from that selected candidate.

FEM coverage is **57/64 (89.1%)**. The previous 112-query conditional result remains a diagnostic and is not used as this full-scope table.

## Matched localization metrics

`Localization Error [m]` is the median over all 64 queries.

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Native FEM coverage |
|---|---:|---:|---:|---:|---:|
| Vanilla FLAC | 1.009 | 15.6% | 50.0% | 23.4% | 100.0% |
| OrbitRIR (FA-BF FLAC) | 0.928 | 28.1% | 53.1% | 40.6% | 100.0% |
| Yaw-Augmented FLAC | 0.892 | 21.9% | 54.7% | 34.4% | 100.0% |
| Few-ShotRIR | 1.517 | 3.1% | 25.0% | 14.1% | 100.0% |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 1.059 | 28.1% | 50.0% | 35.9% | 89.1% |

## Localization-error distribution

| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ |
|---|---:|---:|---:|
| Vanilla FLAC | 1.748 | 1.009 | 2.972 |
| OrbitRIR (FA-BF FLAC) | 1.816 | 0.928 | 4.004 |
| Yaw-Augmented FLAC | 1.793 | 0.892 | 3.300 |
| Few-ShotRIR | 3.109 | 1.517 | 6.366 |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 1.999 | 1.059 | 3.799 |

## Interpretation boundary

The fallback selection uses no ground truth: it is a uniform random draw from the query's frozen candidate set, keyed independently by the fixed seed `42` and query index. Ground truth is used only after selection to calculate the standard localization metrics.

The four learned methods use their real predictions on all 64 queries. The three FLAC rows use the registered primary `K_gen=1`; Few-ShotRIR uses `K_ctx=8`; FEM uses eight acoustic context RIRs and Room-Helps one-support OMP.

This is a FEM--OMP result, not a FEM--AGREE result.
