# Depth-AABB FEM representative-room pilot

**Selected status:** `FEM-Sabine (Depth-AABB)` is now the matched-input FEM
baseline for comparison with the FLAC-family methods. Full-Mesh FEM is retained
only as a privileged-geometry oracle. The frozen paired-query protocol is in
`../exp_14_depth_aabb_matched_protocol/README.md`.

This is a three-query diagnostic, not a publishable aggregate. It tests whether
FEM-Sabine remains unusually strong when its full official room mesh is replaced
by an axis-aligned envelope fitted only to the same receiver-centred depth
panorama supplied to FLAC.

## Frozen protocol

- One frozen Batch-1 query from each representative room: `4467`
  (`Bathrooms_idx_14`), `1446` (`Office_idx_10`), and `5494`
  (`LivingRoomsWithHallway_idx_30`).
- The candidate arrays are byte-identical across Depth-AABB FEM, Full-Mesh FEM,
  and all three FLAC arms.
- FLAC is read at its registered primary `K_gen=1` result. Both FEM rows use
  `K_ctx=8`, the full 80--300 Hz band (102 exact DFT bins), Room-Helps
  single-source complex OMP, and the same observed RIR.
- The depth envelope is the min/max xyz extent of all valid panorama endpoints,
  padded by a fixed 0.05 m. It does not read the official room mesh.
- A conforming structured tetrahedral mesh is constructed with true
  `h_max <= 0.22 m`. All frozen candidates, the target source, and all eight
  context sources passed the inside-domain audit in all three queries.

## Per-query localization error

| Room (query) | Vanilla FLAC | FA-BF FLAC | Yaw-Aug FLAC | Full-Mesh FEM | Depth-AABB FEM |
|---|---:|---:|---:|---:|---:|
| Bathrooms_idx_14 (4467) | 0.680 m | 0.680 m | 0.901 m | **0.335 m** | 0.680 m |
| Office_idx_10 (1446) | 2.584 m | **1.178 m** | 2.771 m | 1.655 m | 2.512 m |
| LivingRoomsWithHallway_idx_30 (5494) | 0.814 m | 0.814 m | 0.814 m | **0.335 m** | 3.593 m |

The three-query diagnostic median is 2.512 m for Depth-AABB FEM versus 0.335 m
for Full-Mesh FEM. Depth-AABB FEM has SR@0.5m = 0/3, SR@1.0m = 1/3, and
resolution-aware SR@0.5m = 1/3. These rates are shown only to expose the pilot
trend; `n=3` is too small for a final comparison.

## Geometry audit

| Room | Depth median abs. error | Depth P95 abs. error | AABB volume | FEM nodes | FEM elements | Localization error |
|---|---:|---:|---:|---:|---:|---:|
| Bathrooms_idx_14 | 0.073 m | 0.404 m | 14.9 m3 | 11,088 | 57,960 | 0.680 m |
| Office_idx_10 | 0.094 m | 3.556 m | 109.0 m3 | 69,160 | 382,950 | 2.512 m |
| LivingRoomsWithHallway_idx_30 | 0.093 m | 0.944 m | 125.7 m3 | 81,312 | 452,790 | 3.593 m |

The low median depth errors hide large tail errors, especially in the Office.
The AABB erases internal obstacles and non-axis-aligned/non-convex boundaries;
in the hallway room it also fills substantial nonexistent volume. Its poor
localization therefore shows that the full-mesh FEM result depends strongly on
geometric fidelity, but does not establish how a better depth-only completion
would perform.

## Interpretation

The full official mesh should remain labeled a privileged geometry upper bound.
The AABB is useful as a lower-fidelity sanity check, but is too crude to become
the final matched-input baseline. The next justified reconstruction is a
depth-derived plane/polar envelope that preserves dominant walls, floor,
ceiling, and non-convex horizontal structure while retaining the frozen
candidate set.
