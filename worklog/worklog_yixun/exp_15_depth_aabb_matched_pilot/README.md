# Five-method Depth-AABB matched pilot

This is a deterministic one-query-per-room pilot over the 14 non-oversized rooms (`n=14` per K slice), not the final 97-query paired aggregate.

The query is the first strict-coverage record for each room in the frozen paired manifest; localization errors were not used for selection. This is nevertheless a coverage-conditioned sample and is not representative of queries that Depth-AABB cannot contain.

All FEM candidate coordinate arrays are byte-identical to Vanilla FLAC, all four learned arms share the same candidate-index hashes, and every Depth-AABB query passes the strict receiver/source/context/candidate coverage gate.

### K_gen = 1

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 1.010 | 21.4% | 50.0% | 35.7% |
| FA-BF FLAC | 0.867 | 21.4% | 57.1% | 35.7% |
| YAWAUG FLAC | 0.858 | 7.1% | 57.1% | 21.4% |
| Few-ShotRIR | 1.043 | 14.3% | 50.0% | 35.7% |
| FEM-Sabine (Depth-AABB) | 0.733 | 28.6% | 71.4% | 42.9% |

### K_gen = 4

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.808 | 28.6% | 57.1% | 42.9% |
| FA-BF FLAC | 0.811 | 21.4% | 57.1% | 35.7% |
| YAWAUG FLAC | 0.933 | 7.1% | 50.0% | 28.6% |
| Few-ShotRIR | 1.043 | 14.3% | 50.0% | 35.7% |
| FEM-Sabine (Depth-AABB) | 0.733 | 28.6% | 71.4% | 42.9% |

### K_gen = 8

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.808 | 21.4% | 57.1% | 35.7% |
| FA-BF FLAC | 0.867 | 14.3% | 57.1% | 28.6% |
| YAWAUG FLAC | 0.811 | 14.3% | 57.1% | 28.6% |
| Few-ShotRIR | 1.043 | 14.3% | 50.0% | 35.7% |
| FEM-Sabine (Depth-AABB) | 0.733 | 28.6% | 71.4% | 42.9% |

## Interpretation

The apparent aggregate advantage is materially confounded by candidate-domain size. An exactly uniform random candidate already has macro SR@1.0m = 21.0%; the two Bathroom queries have random SR@1.0m of 55.6% and 48.2%. On the nine queries whose candidate AABB diagonal is at least 5 m, Depth-AABB FEM has median error 0.860 m and SR@1.0m 55.6%, versus 0.814 m / 55.6% for Vanilla FLAC and YAWAUG at K_gen=1. Thus this pilot does not show that FEM is stronger on larger rooms.

Depth-AABB also has large individual failures in LivingRoomsWithHallway_idx_30 (3.593 m) and Apartments_idx_42 (2.291 m). Standard absolute-threshold metrics should be accompanied by exact random-hit probability or candidate-domain-normalized error in the final report.

## FEM execution audit

- Wall time: 11.7 min with 2 workers x 12 MKL threads.
- Mesh nodes: 11,088--117,600.
- Maximum relative linear-solver residual: `5.810e-13`.
- FEM is deterministic and is reused as the same reference across the three K_gen slices; it is not counted as three independent runs.
