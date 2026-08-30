# Train-calibrated Depth-BoundedCompletion pilot

This experiment tests a lightweight, candidate-independent hidden-space prior
between the single-view Depth-Polar layout and Depth-AABB. It is a diagnostic,
not a learned SSC model and not a publishable 384-query result.

## No-leakage protocol

- Inference reads one fixed receiver-centred metric depth panorama. It never
  reads the unseen room mesh or candidate coordinates while reconstructing the
  domain.
- A global radial completion distance was calibrated on 243 disjoint training
  rooms, using one deterministic receiver view per room and 6,460 train source
  anchors. The frozen value is the predeclared 99.5th percentile: `4.0674 m`.
- Each polar ray can grow by at most that distance and cannot pass the AABB
  recovered from the same depth panorama. Candidate, target-source, and context
  coordinates are used only by the post-reconstruction coverage gate.
- Scoring requires the unchanged candidate array, target source, receiver, and
  all eight context sources to be inside both the continuous domain and its
  tetrahedral mesh. There is no candidate filtering or score substitution.

The train split SHA-256 is
`aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03`.
The complete calibration and selected-depth hashes are in
`train_calibration.json`.

## Representative geometry result

| Room (query) | Strict coverage | Footprint area vs Polar | FEM nodes | Relation to Depth-AABB |
|---|---:|---:|---:|---|
| Bathrooms_idx_14 (4467) | 27/27 + source + 8/8 contexts | 1.070x | 11,088 | same elements; nodes within `4.44e-16 m` |
| Office_idx_10 (1446) | 254/254 + source + 8/8 contexts | 1.346x | 69,160 | same elements; nodes within `1.78e-15 m` |
| LivingRoomsWithHallway_idx_30 (5494) | 279/279 + source + 8/8 contexts | 1.408x | 81,312 | same elements; nodes within `4.44e-16 m` |

The conservative train-calibrated setting therefore solves the coverage issue
for these three chosen queries, but expands their FEM grids back to the AABB
grids to machine precision. It loses the Depth-Polar speed advantage.

## Localization effect

| Room | Depth-Polar FEM | Depth-BoundedCompletion FEM | Depth-AABB FEM |
|---|---:|---:|---:|
| Bathrooms_idx_14 | 0.680 m | **0.680 m (independently rerun)** | 0.680 m |
| Office_idx_10 | 0.279 m (216/254 candidates; invalid direct comparison) | 2.512 m (AABB-equivalent grid) | 2.512 m |
| LivingRoomsWithHallway_idx_30 | 3.378 m (266/279 candidates; invalid direct comparison) | 3.593 m (AABB-equivalent grid) | 3.593 m |

The Bathroom completion run reproduced the AABB candidate array, score vector,
winner, and all metrics exactly. Office and Living were not redundantly
re-solved: their element arrays are exactly equal to the prior AABB meshes and
their node coordinates differ only at floating-point roundoff, so the table
labels the established AABB-equivalent result rather than a new execution.

## Frozen 128-query coverage audit

The two 64-query pilot manifests contain eight queries from each of 16 rooms.
Using the same frozen `4.0674 m` train calibration:

| Geometry | Queries with every candidate, source, and context inside |
|---|---:|
| Depth-Polar | 70/128 (54.7%) |
| Depth-BoundedCompletion | **92/128 (71.9%)** |
| Depth-AABB | 112/128 (87.5%) |

Depth-BoundedCompletion improves coverage by 22 queries, but it still fails 36
queries. Auditorium and Cafe remain 0/8. It therefore cannot support the full
384-query protocol as an air-only tetrahedral domain. Per-query counts and all
input hashes are stored in `coverage_audit_128.json`.

## Conclusion

The lightweight prior is useful as a negative result: conservative completion
that covers representative hidden candidates collapses toward AABB and inherits
its poor localization, while a less conservative polar shape leaves candidates
unscoreable. The next serious method should be learned train-only occupancy/SDF
completion coupled to a fixed background (immersed or variable-coefficient) FEM
domain. The fixed background is what guarantees all candidates remain
scoreable; the learned occupancy controls acoustics without changing the
candidate set.

Implementation: `src/baselines/depth_polar_layout.py`,
`tools/calibrate_depth_completion.py`, `tools/audit_depth_completion_coverage.py`,
and `tools/probe_depth_polar_fem.py`.
