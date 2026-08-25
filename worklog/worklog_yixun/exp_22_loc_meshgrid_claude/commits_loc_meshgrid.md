# commits_loc_meshgrid — exp_22 per-round commit ledger

Protocol source of truth: `loc_meshgrid_inherited_exp09_plan.md` (§1.1–1.3, D1/G1).
Every commit is path-scoped and TDD (red → green → commit).

## Round exp22-r1 (D1 context materializer + G1 geometry primitives)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `9b362a2` | D1 | `src/localization/meshgrid_queries.py`: runs the unmodified released `AR_md` selector under the exact exp_01 loader protocol (seed 42 / batch 64 / workers 4 / shuffle off / `pl.seed_everything(42, workers=True)` / full split order) and records per query the identity, the context fingerprints, each context RIR's sha256, the eligible-pool size and the order; materialize-then-filter enforced; exclusion exactness; content-hashed manifest with byte-stable reload that never builds a loader | +399 |
| `e05d0de` | G1 | `src/localization/meshgrid_geometry.py`: 0.5 m room-global lattice, 31 frozen non-axis-aligned directions with strict-majority odd parity, separate 0.20 m clearance prior, per-query receiver/context/z-band filters, `choose_z_branch`, `grid_oracle_error`, fail-closed mesh loading with identity metadata, and the §1.3 room-anchor audit | +411 |

**Suite after exp22-r1:** 2840 passed, 10 skipped, **0 failures**, 2 subtests passed.
(The long-standing exp_11-registry failure was fixed independently by exp_15 in `9b5ad80`;
the whole repository suite is green for the first time in this campaign.)

### Measured facts recorded this round

- **Eligible-pool census reproduces the inherited pins exactly** on the real dataset
  (pure file-tree computation, no RNG): `{6:91, 7:429, 8:5263, 9:554}` full and
  `{6:91, 7:429, 8:4363, 9:454}` filtered, 6,337 → 5,337, and every short-pool query
  is in `Cafe/Cafe_idx_1` (520 = 91 + 429).
- **Cafe_idx_1 mesh** (`e7a0b7b9…`): 93,035 vertices / 366,080 triangles, **not**
  watertight, **not** edge-manifold — which is exactly why the parity vote, not
  Open3D occupancy, is the validity rule. All 10 source anchors and all 100 receiver
  anchors classify inside by parity; all 10 sources clear 0.20 m (min 0.550 m).
  Its full lattice is 9,996 nodes → 8,381 parity-valid → 6,273 valid with the prior.

### Inherited-plan ambiguity, flagged not improvised

§1.3's fail-closed acceptance lists rule 2 ("every metadata **anchor** … inside/on the
free-space classification") and rule 3 ("every real metadata **source** anchor survives
the same inside/surface-validity predicate used for candidates"). Two Cafe **receiver**
anchors sit 0.100 m from a surface: they pass parity but not the 0.20 m prior. This
round implements the literal reading — parity for every anchor, parity **+** the source
prior for sources only — because the 0.20 m rule is described in §1.2 as a
*source-distribution* prior and receivers are neither candidates nor drawn from that
distribution (their own constraint is the ≥ 0.5 m candidate guard). Applying the prior
to receivers instead would block Cafe_idx_1, and with it the 16-room subset.
**A ruling is requested before G1's real-mesh audit runs over all 16 rooms.**
