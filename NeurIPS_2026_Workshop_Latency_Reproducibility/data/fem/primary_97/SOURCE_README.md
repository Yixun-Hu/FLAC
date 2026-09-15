# Five-method Depth-AABB matched 97-query result

This is the complete frozen strict-coverage subset over the 14 non-oversized rooms (`n=97` per K slice; 97/112 source queries, 86.6% coverage).

Localization errors were not used for selection. This is nevertheless a coverage-conditioned sample and is not representative of the 15 source queries that Depth-AABB cannot contain.

All FEM candidate coordinate arrays are byte-identical to Vanilla FLAC, all four learned arms share the same candidate-index hashes, and every Depth-AABB query passes the strict receiver/source/context/candidate coverage gate.

### K_gen = 1

| Model | Median Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.814 | 18.6% | 59.8% | 36.1% |
| FA-BF FLAC | 0.809 | 19.6% | 60.8% | 41.2% |
| YAWAUG FLAC | 0.802 | 18.6% | 60.8% | 39.2% |
| Few-ShotRIR | 1.390 | 5.2% | 27.8% | 16.5% |
| FEM-Sabine (Depth-AABB) | 0.718 | 33.0% | 59.8% | 46.4% |

### K_gen = 4

| Model | Median Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.825 | 17.5% | 57.7% | 34.0% |
| FA-BF FLAC | 0.835 | 17.5% | 55.7% | 36.1% |
| YAWAUG FLAC | 0.835 | 17.5% | 55.7% | 36.1% |
| Few-ShotRIR | 1.390 | 5.2% | 27.8% | 16.5% |
| FEM-Sabine (Depth-AABB) | 0.718 | 33.0% | 59.8% | 46.4% |

### K_gen = 8

| Model | Median Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.825 | 16.5% | 57.7% | 33.0% |
| FA-BF FLAC | 0.850 | 15.5% | 55.7% | 36.1% |
| YAWAUG FLAC | 0.807 | 17.5% | 59.8% | 38.1% |
| Few-ShotRIR | 1.390 | 5.2% | 27.8% | 16.5% |
| FEM-Sabine (Depth-AABB) | 0.718 | 33.0% | 59.8% | 46.4% |

## Interpretation

Depth-AABB FEM has median error 0.718 m, mean error 1.185 m, SR@0.5m 33.0%, and SR@1.0m 59.8% on this conditional scope.

The strict subset is imbalanced across rooms, so the room-macro view is more conservative: Depth-AABB has room-macro mean error 1.263 m and room-macro SR@1.0m 56.4%, versus 1.044 m / 62.7% for Vanilla FLAC at K_gen=1.

Candidate-domain size remains a material confounder. An exactly uniform random candidate has macro SR@1.0m 21.1%. On the 57 queries whose candidate AABB diagonal is at least 5 m, Depth-AABB FEM has median error 0.860 m and SR@1.0m 52.6%, versus 0.787 m / 61.4% for Vanilla FLAC at K_gen=1.

The three largest Depth-AABB errors are: MeetingRoom_idx_20 q1287 (4.358 m), Office_idx_11 q1692 (4.257 m), Office_idx_10 q1594 (4.110 m).

## FEM execution audit

- Wall time: 68.9 min with 2 workers x 12 MKL threads.
- Newly completed/resumed exact results: 83/14.
- Mesh nodes: 11,088--117,600.
- Maximum relative linear-solver residual: `5.810e-13`.
- FEM is deterministic and is reused as the same reference across the three K_gen slices; it is not counted as three independent runs.
