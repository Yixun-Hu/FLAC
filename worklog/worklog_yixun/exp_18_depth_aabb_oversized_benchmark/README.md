# Depth-AABB oversized-room benchmark

This benchmark measures one strict-coverage representative query in each of
the two oversized AcousticRooms geometries using the same production
Depth-AABB FEM configuration as the matched experiment.

| Room | Query | Receiver | Expected nodes | Expected tetrahedra |
|---|---:|---:|---:|---:|
| Cafe_idx_1 | 335 | R063 | 755,196 | 4,305,000 |
| Auditorium_idx_1 | 3550 | R038 | 1,092,546 | 6,309,672 |

The two queries run sequentially to bound peak host memory. Each query uses
the exact 80--300 Hz bins, MKL PARDISO, 24 solver threads, `h_max=0.22 m`,
0.05 m Depth-AABB padding, the frozen candidate set, and the strict
receiver/target/context/candidate coverage gate.

## External-server supplemental results

An externally executed seven-query slice covering five Cafe queries and two
Auditorium queries is archived separately for later matched aggregation:

- [Human-readable report](external_server_7query_omp_report.md)
- [Machine-readable summary](external_server_7query_omp_summary.json)
- [External-result verification checklist](external_server_7query_verification_checklist.md)
- [Merged 112-query FEM--OMP summary](fem_omp_112_merged_summary.md)
- [Primary full 128-query five-method comparison](../exp_21_five_method_128_failure_penalized/summary.md)

The archived FEM accuracy row is `fem_sabine_depth_aabb`, using the
deterministic Room-Helps one-support OMP selector. It is not a FEM--AGREE
accuracy row; AGREE scoring had not been run for these seven responses when
the record was supplied. The external checks were reported as passed on
2026-08-30, so the values are accepted for the 112-query FEM--OMP aggregation.
They retain external provenance and do not overwrite locally generated query
artifacts.

The 112-query merge is a strict-coverage conditional diagnostic. The primary
end-to-end comparison retains all 128 queries and applies explicit failure
penalties to the 16 queries that Depth-AABB FEM cannot evaluate.
