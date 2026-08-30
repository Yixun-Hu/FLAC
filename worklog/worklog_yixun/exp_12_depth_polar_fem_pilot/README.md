# LGT-inspired Depth-Polar FEM representative-room pilot

This is a three-query diagnostic of a deterministic, metric-depth adaptation of
LGT-Net's horizon-depth room representation. It is **not** an execution or
reproduction of the learned RGB LGT-Net model.

## Frozen protocol

- Queries: `4467` (`Bathrooms_idx_14`), `1446` (`Office_idx_10`), and
  `5494` (`LivingRoomsWithHallway_idx_30`), identical to the Depth-AABB pilot.
- Candidate arrays are byte-identical to the Depth-AABB and Full-Mesh FEM runs.
- `K_ctx=8`, 102 exact DFT bins spanning 80--300 Hz, Room-Helps single-source
  complex OMP, MKL PARDISO with 24 CPU threads, and true `h_max <= 0.22 m`.
- Geometry recovery reads only the receiver-centred metric depth panorama. The
  official room mesh is not read by the layout or meshing implementation.
- Floor and ceiling are estimated from the vertical panorama caps. Per-column
  wall radius is the 0.95 quantile of vertically supported horizontal depth,
  circular-median filtered over 5 columns, simplified at 0.03 m, and padded by
  0.05 m. The resulting non-convex star-shaped footprint is extruded and
  voxelized into a conforming tetrahedral mesh.
- Candidates outside the single-view tetrahedral domain retain their positions
  in the frozen array but receive a finite score strictly below all modeled
  candidates. Coverage is therefore a required metric, not a hidden detail.

## Per-query localization error

| Room (query) | Vanilla FLAC | FA-BF FLAC | Yaw-Aug FLAC | Full-Mesh FEM | Depth-AABB FEM | Depth-Polar FEM |
|---|---:|---:|---:|---:|---:|---:|
| Bathrooms_idx_14 (4467) | 0.680 m | 0.680 m | 0.901 m | **0.335 m** | 0.680 m | 0.680 m |
| Office_idx_10 (1446) | 2.584 m | 1.178 m | 2.771 m | 1.655 m | 2.512 m | **0.279 m** |
| LivingRoomsWithHallway_idx_30 (5494) | 0.814 m | 0.814 m | 0.814 m | **0.335 m** | 3.593 m | 3.378 m |

The Office prediction is the frozen candidate nearest the ground truth. The
Depth-AABB winner remains inside the Depth-Polar domain in all three rooms, so
the AABB-to-Polar change is not explained merely by removing the previous
winning candidate. It nevertheless remains conditional on incomplete candidate
coverage in the Office and hallway queries.

## Three-query diagnostic aggregate

| Model | Median error | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |
|---|---:|---:|---:|---:|
| Vanilla FLAC | 0.814 m | 0.0% | 66.7% | 66.7% |
| FA-BF FLAC | 0.814 m | 0.0% | 66.7% | 66.7% |
| Yaw-Aug FLAC | 0.901 m | 0.0% | 66.7% | 66.7% |
| Full-Mesh FEM | **0.335 m** | **66.7%** | 66.7% | 66.7% |
| Depth-AABB FEM | 2.512 m | 0.0% | 33.3% | 33.3% |
| Depth-Polar FEM | 0.680 m | 33.3% | 66.7% | 66.7% |

`n=3` is too small for a paper-level aggregate. The mean error is 1.446 m for
Depth-Polar FEM, 2.262 m for Depth-AABB FEM, and 0.775 m for Full-Mesh FEM.

## Geometry, coverage, and runtime audit

| Room | Depth abs. error median / P95 | Modeled candidates | Polar / AABB nodes | Polar / AABB total time | Speedup |
|---|---:|---:|---:|---:|---:|
| Bathrooms_idx_14 | 0.052 / 0.216 m | 27 / 27 | 9,936 / 11,088 | 6.23 / 16.08 s | 2.58x |
| Office_idx_10 | 0.060 / 2.906 m | 216 / 254 | 52,416 / 69,160 | 51.65 / 162.93 s | 3.15x |
| LivingRoomsWithHallway_idx_30 | 0.056 / 0.725 m | 266 / 279 | 57,132 / 81,312 | 64.19 / 184.64 s | 2.88x |

All three tetrahedral meshes are face-connected. Their maximum element edges
are 0.205--0.207 m, their audited voxel volumes exactly match the FEM element
volume sums, and maximum relative linear-solver residuals are at most
`1.53e-13`.

The depth median errors improve over AABB in all three rooms. The Office still
has a large P95 because nearby partitions occlude the outer room, while the
hallway has regions hidden around a turn. These are structural single-view
ambiguities, not tetrahedralization failures.

## Interpretation

The result supports the LGT-style horizon-depth representation as a materially
better depth-only geometry baseline than AABB: it improves two of three query
errors, substantially lowers the three-query median, and reduces FEM runtime by
2.6--3.2x. It does not yet establish a fair 384-query result because a
single-radius-per-azimuth layout cannot model candidates hidden around turns.

Before scaling, the method should either (1) add a learned completion stage and
audit it as an external prior, or (2) make candidate-domain coverage an explicit
evaluation dimension and avoid interpreting incomplete-coverage queries as a
direct all-candidate comparison.

## Artifacts

Each query has a hashed result JSON, score NPZ, depth-layout NPZ, tetrahedral
mesh NPZ, and execution log in this directory. The implementation is in
`src/baselines/depth_polar_layout.py`; the runner is
`tools/probe_depth_polar_fem.py`; regression tests are in
`src/tests/test_depth_polar_layout.py`.
