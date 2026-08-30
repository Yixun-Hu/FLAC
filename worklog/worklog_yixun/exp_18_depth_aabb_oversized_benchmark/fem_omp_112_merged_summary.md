# Merged 112-query FEM--OMP result

Status: **COMPLETE** (`112/112` queries).

The merge uses the frozen 112-query selection. External-server queries retain explicit provenance and are counted once; query `715` is a cross-server replication check whose local artifact is canonical.

| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ |
|---|---:|---:|---:|---:|---:|---:|
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 1.842 | 0.800 | 3.339 | 29.5% | 55.4% | 42.0% |

## Source accounting

- `external_verified`: 6 queries
- `local_oversized`: 9 queries
- `local_primary_97`: 97 queries

Accepted external query IDs: `30`, `42`, `535`, `715`, `917`, `3800`, `3841`.

This is a FEM--OMP result, not a FEM--AGREE result.
