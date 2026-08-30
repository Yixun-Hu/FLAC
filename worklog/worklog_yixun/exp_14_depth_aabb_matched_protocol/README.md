# Depth-AABB matched-comparison protocol

## Decision

The primary FEM row is now **FEM-Sabine (Depth-AABB)**. It is the only FEM
geometry in the fair main comparison because it reads one receiver-centred
metric depth panorama and does not read the official unseen-room mesh.

- `FEM-Sabine (Full-Mesh)` remains a privileged-geometry oracle.
- `FEM-Sabine (Depth-Polar)` remains an incomplete-coverage diagnostic.
- `FEM-Sabine (Depth-BoundedCompletion)` remains a negative ablation.

The five aligned methods are Vanilla FLAC, FA-BF FLAC, YAWAUG FLAC,
Few-ShotRIR-Waveform, and FEM-Sabine (Depth-AABB).

## Shared inputs and candidates

Depth-AABB is reconstructed before looking at target or candidate coordinates:

1. Back-project the fixed receiver depth panorama to receiver-local metric xyz.
2. Take per-axis minima/maxima and add the frozen `0.05 m` padding.
3. Fill the box with a conforming tetrahedral mesh satisfying
   `h_max <= 0.22 m`.
4. Only after reconstruction, require the receiver, target source, all eight
   context sources, and every frozen candidate to be inside.

Every paired method receives the same query IDs and byte-identical candidate
arrays. No candidate inside a query may be removed and no score may be
substituted for an outside candidate.

## Paired evaluation scopes

Depth-AABB does not strictly cover every frozen query, so the comparison must
not silently discard those failures or present it as a complete-coverage
method.

The primary end-to-end comparison therefore retains all `16 x 8 = 128`
queries. For each query that fails the FEM strict-coverage gate:

- FEM SR@0.5m, SR@1.0m, and resolution-aware SR@0.5m are all zero;
- FEM localization error is the maximum true-source distance over that
  query's frozen candidate set;
- FEM coverage is reported explicitly next to the accuracy metrics.

This finite penalty is query-scale-aware and uses ground truth only to compute
an evaluation error after inference has failed. It does not expose target
information to the method. The formal full-scope result is archived in
`../exp_21_five_method_128_failure_penalized/summary.md`.

The strict paired subsets below remain conditional diagnostics for separating
localization behavior from coverage behavior; they are not the primary
end-to-end ranking.

| Scope | Source queries | Strict paired queries | Coverage | Manifest |
|---|---:|---:|---:|---|
| Conditional: 14 rooms, excluding Auditorium/Cafe | 112 | **97** | 86.6% | `depth_aabb_matched_14room_97.json` |
| Conditional: all 16 rooms | 128 | **112** | 87.5% | `depth_aabb_matched_16room_112.json` |

For each conditional paired table, metrics for all five methods must be recomputed on the
exact manifest named above. This preserves complete candidate sets, although
the conditional scope and its per-room imbalance must be disclosed.

## K alignment

FLAC is reported in three paired slices, `K_gen={1,4,8}`. Depth-AABB FEM is
deterministic and has `K_ctx=8`; it has no `K_gen` axis. Therefore:

- The primary comparison uses the registered FLAC `K_gen=1` setting and
  Few-ShotRIR `K_ctx=8`. In this naming, FA-BF FLAC is OrbitRIR.

- Full 16-room scope: three FLAC slices of 128 queries; FEM has 112 evaluated
  queries and 16 explicit coverage-failure penalties.
- Conditional 14-room scope: 97 unique FEM solves and three FLAC slices of 97
  queries (`291` FLAC query-configurations).
- Conditional 16-room scope: 112 unique FEM solves and three FLAC slices of 112
  queries (`336` FLAC query-configurations).

The same FEM result may appear as the reference in each K slice, but it must not
be counted as three independent FEM observations. In particular, the original
`128 x 3 = 384` FLAC configurations must be reported as three `n=128` slices,
not pooled into one `n=384` significance sample.

## Main metrics

The table uses the requested four metrics:

- median localization error in metres;
- success rate at `0.5 m`;
- success rate at `1.0 m`;
- resolution-aware success rate at `0.5 m`.

Depth-AABB coverage is reported next to the table. Full-Mesh FEM results may be
shown in a separate oracle block, never in the matched-input ranking.

Exact machine-readable choices are frozen in `protocol.json`. The two matched
manifests were derived from the hashed 128-query coverage audit without using
localization errors.

The audited single-query implementation is `tools/probe_depth_aabb_fem.py`.
The legacy `localize_baseline.py --method fem_sabine` path consumes official
room-derived meshes and must remain labeled Full-Mesh oracle until a dedicated
Depth-AABB batch path is wired to these manifests.
