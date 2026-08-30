# Auditorium/Cafe seven-query matched results

## Provenance and scope

This is a user-provided record from an external-server execution, archived on
2026-08-30 for later matched aggregation. The external artifact, protocol, and
numerical checks were reported as passed by the user on 2026-08-30, and the
slice is accepted for the FEM--OMP 112-query aggregation. The result artifacts
have not been independently copied and recomputed on this server.

The report covers the seven completed strict-coverage queries below, evaluated
on the same frozen query identities and candidate protocol:

- Cafe: `715`, `30`, `42`, `535`, `917`
- Auditorium: `3800`, `3841`
- Query count: `n=7`
- FLAC accuracy slice: `K_gen=8`
- FEM context: `K_ctx=8`, 102 exact DFT bins from 80--300 Hz

These are descriptive results for the completed Auditorium/Cafe subset, not
the official 97-query primary table or the complete 112-query secondary table.

## Method-identity guardrail

The completed FEM row uses the deterministic Room-Helps one-support OMP
selector and has method identifier `fem_sabine_depth_aabb`. It is **not** a
FEM--AGREE accuracy result. No FEM--AGREE row is reported because the frozen
AGREE scoring stage had not been run for these seven newly completed FEM
responses when this record was supplied.

## Localization-error distribution

Values are in metres. P90 uses NumPy's default linear quantile interpolation.

| Model | Mean ↓ | Median ↓ | P90 ↓ |
|---|---:|---:|---:|
| Vanilla FLAC (`K_gen=8`) | 6.444 | 3.939 | 13.643 |
| FA-BF FLAC (`K_gen=8`) | 6.334 | 6.806 | 10.487 |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | **3.713** | **1.330** | **8.683** |

## Matched localization metrics

`Localization Error [m]` is the protocol median. Resolution-aware success at
0.5 m means `localization_error - oracle_error <= 0.5 m`.

| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ |
|---|---:|---:|---:|---:|
| Vanilla FLAC (`K_gen=8`) | 3.939 | 0.0% | 0.0% | 0.0% |
| FA-BF FLAC (`K_gen=8`) | 6.806 | 0.0% | 0.0% | 0.0% |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | **1.330** | **14.3%** | **42.9%** | **14.3%** |

## Per-query localization error

| Query | Room | Vanilla FLAC [m] | FA-BF FLAC [m] | FEM-Sabine + OMP [m] |
|---:|---|---:|---:|---:|
| 715 | Cafe | 4.804 | 4.087 | **2.099** |
| 30 | Cafe | 17.728 | 7.446 | **0.443** |
| 42 | Cafe | 2.034 | 1.593 | **0.829** |
| 535 | Cafe | 3.322 | 3.315 | **0.841** |
| 917 | Cafe | 3.939 | 6.806 | **1.894** |
| 3800 | Auditorium | 2.364 | 10.258 | **1.330** |
| 3841 | Auditorium | **10.919** | **10.831** | 18.558 |

## Observed per-query runtime

Values are seconds/query. These timings are recorded executions, not a
hardware-normalized latency comparison: the FLAC rows are legacy joint
`K_gen={1,4,8}` GPU passes, while the FEM row is the external-server
13-physical-core Depth-AABB mesh/operator/102-bin solve and OMP scoring path.

| Model | Mean [s] ↓ | Median [s] ↓ | P90 [s] ↓ |
|---|---:|---:|---:|
| Vanilla FLAC, joint `K_gen={1,4,8}` | 272.77 | 296.81 | 300.12 |
| FA-BF FLAC, joint `K_gen={1,4,8}` | 297.53 | 323.58 | 327.42 |
| FEM-Sabine + Room-Helps OMP (Depth-AABB) | 2400.38 | 1567.69 | 4503.77 |

## FEM execution audit

| Job | Queries | Elapsed | MaxRSS |
|---|---|---:|---:|
| `3773389` | Cafe `715, 30, 42` | 01:18:16 | 10.8 GiB |
| `3773391` | Cafe `535, 917` | 00:53:25 | 10.8 GiB |
| `3773390` | Auditorium `3800, 3841` | 02:31:02 | 18.5 GiB |

All seven FEM results completed 102 frequencies. The maximum relative linear-
solver residual across the subset was below `7e-13`, and every referenced
mesh/scores SHA-256 matched its result manifest.

## Interpretation boundary

Auditorium query `3841` is the main outlier: FEM localization error is
`18.558 m` despite a converged linear solve, so its scoring and geometry
behavior should be reviewed separately. These values remain separately
provenance-labelled when they are incorporated into the complete 112-query
aggregation. Query `715` is also available locally and is retained as a
cross-server replication check rather than counted twice.
