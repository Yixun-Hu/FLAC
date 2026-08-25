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

## Round exp22-r2 (r1 code review — all 4 BLOCKERs)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `6f169c5` | F1 + F2 | the materializer reproduces the released evaluator's COMPLETE init order (seed → loader → metric/AGREE stack → iterator), resolving the configured `AGREE_fullAR` checkpoint fail-closed and recording the RNG digest at iterator creation; a parity test runs both call graphs on a bounded real slice and a counter-test proves that skipping the stack changes the draw. Every position is checked against the split enumeration (`idx` and relpath) before its draw is recorded | +214 −33 |
| `9eef028` | F3 | the 31 directions are written out as LITERALS and are the protocol (digest `9ab4339f…`, the value the review measured); `build_directions` survives as provenance and is asserted to still reproduce them; the MeetingRoom_idx_32 receiver discrepancy (15/31 odd votes, 0.25005 m) is a documented, unresolved entry that keeps the room blocked, with `odd_parity_votes` exposing the count | +134 −25 |
| `a9f796b` | F4 | `audit_meshgrid_geometry.py`: one OBJ per room, D1 contexts recovered in global coordinates and matched to metadata anchors, both oracle branches, the global branch decision, hashed per-room candidate manifests and the post-G1 cost report; `choose_z_branch` now requires equal query sets and finite oracles and never defaults a missing entry to 0.0 | +342 −18 |

**Suite after exp22-r2:** 2863 passed, 10 skipped, **0 failures**, 2 subtests passed.

### What the r2 evidence establishes

- The call-graph parity test passes on real data: our materializer and an
  `eval_FLAC`-faithful reference reach iterator creation with **the same torch RNG
  state** and draw **identical contexts** (fingerprints and context-audio digests) for
  the first four records. Omitting the metric stack changes that digest — the guard bites.
- The direction digest reproduces the reviewer's `9ab4339f…` exactly, and the
  MeetingRoom_idx_32 anchor reproduces at **15/31 odd votes, 0.25005 m** — recorded, not
  papered over, with the majority rule and anchor predicate untouched pending the exp_09
  cross-check.

## Round exp22-r3 (r2 re-review — F2 partial, F3 partial, F4 not resolved)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `d061f64` | F2 | exact canonical relpath equality (both sides reduced to one root-relative form, then string equality — four spoofs that passed the bidirectional `endswith` are now tests); an enumeration that cannot be built is a refusal, never an empty expectation; `assert_split_enumeration` proves 6,337 unique ordered identities **before** the pass and `assert_pass_census` the registered histogram **after**, both mandatory | +118 −21 |
| `1d1e491` | F3 + F4 | `REQUIRED_ROOMS` literal + refusal on deviation; a blocked room aborts **before any artifact is written**, with `--diagnostics-only` writing one stamped non-manifest report; empty z-band sets stay `inf` and disqualify the branch by name; gate counts for both branches; unique receiver-candidate pairs hashed over the actual index arrays; per-room manifests carry indices, a coordinate digest, a sidecar npz, the chosen branch and the snapped lattice origin; record-stream validation (uniqueness, positions, census, room set) | +286 −74 |

**Suite after exp22-r3:** 2877 passed, 10 skipped, **0 failures**, 2 subtests passed.

### Deliberate refinement, for the re-review

r2's F4 hardening made `choose_z_branch` refuse any non-finite oracle; r3's F3(b)
requires empty z-band sets to stay `inf` **in** the distribution. Both cannot hold
literally, so the rule now distinguishes them: NaN is missing evidence on either side
and still refuses, a non-finite **full-height** oracle still refuses (an empty
full-height set is a hard failure upstream), and `+inf` on the **band** side is
meaningful — it disqualifies the branch, by name and with a count.

## Round exp22-r4 (r3 re-review — four blockers + one partial)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `3b27412` | 1–5 | root matched as a whole **component sequence** (`NotAcousticRooms/…` and `AcousticRoomsOld/…` no longer canonicalize onto the split, plus a multi-component root case) and `build/write/load_manifest` refuse `census_verified != True`; `GateCounter` computes the **per-receiver union** ({0,1,2}+{0,1,3} ⇒ 4 calls, 4 pairs, 2 distinct sets); `--expected-queries` restricted to `--diagnostics-only` with the registered census always enforced and `assert_registered_census` in `main()`; `-inf` refuses on both sides while `+inf` stays meaningful on the band side only; `verify_room_manifest` + `verify_report_chain` re-accept every published artifact from its own files as the LAST publish step, and publishing refuses a non-empty output directory | +398 −86 |

**Suite after exp22-r4:** 2889 passed, 10 skipped, **0 failures**, 2 subtests passed.

### D1 manifest — REGENERATED after the F2 fix

The superseded `outputs_loc/exp22/d1_context_manifest.json` was **deleted** first (r4
`--overwrite` semantics), then written fresh by the same driver calls
(`loc_meshgrid_2026-08-25_02:12:43_d1_manifest.log`):

| fact | value |
|---|---|
| full stream sha256 | `15d229c0b5c56107475141e629504e86f2f9b8b3f3a3eeaa0995755380f5abc4` |
| filtered stream sha256 | `99f8da609ef30456faa8251ad000c4675cdb2065013457cff110f905980894e9` |
| census | 6,337 → 5,337, `census_verified: true` |
| eligible histograms | full `{6:91, 7:429, 8:5263, 9:554}` · filtered `{6:91, 7:429, 8:4363, 9:454}` |
| short-context queries | 520, all in `Cafe/Cafe_idx_1` |
| call graph | `seed_everything → build_dataloader → build_metric_stack → create_iterator` |
| AGREE | `weights/AGREE/AGREE_fullAR.pt`, resolution **configured**, sha `3a13243d6c6a1108…` |
| RNG digest at iterator creation | `7b625f96cc52808b8cc092ee9f037fcaae8b6c2e3230a53acb86ddfa08ad7f08` |

**Both stream hashes are identical to the superseded manifest**, which is the expected
result and worth stating: the F2 fix changed how a path is *compared*, not what the
released loader *drew*. The inherited-exp09 hash comparison remains **pending** the rsync.

## Micro-round exp22-r5 (r4 re-review — the last G1 blocker)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `60e91e4` | staged publish | the audit writes every artifact into a `.staging_geometry_audit_*` sibling, runs `verify_room_manifest` + `verify_report_chain` against the STAGED files, and only then publishes — a whole-directory `os.replace` where possible, per-file otherwise. A verifier failure removes the staging directory and leaves the final directory untouched and empty. The ordering defect is fixed: the verification block is written into the report **before** its disk copy (and the chain result rewritten once the chain verifies), so the published report states what was checked | +64 −22 |

**Suite after exp22-r5:** 2892 passed, 10 skipped, **0 failures**, 2 subtests passed.

## Round exp22-r6 (Yixun directive — self-authoritative direction selection)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `910ead7` | anchor-driven selection | `select_direction_seed` / `evaluate_direction_seed` / `anchor_scenes` implement the registered rule — the smallest generator seed whose 31 directions classify EVERY metadata source and receiver anchor of all 16 required rooms as interior at ≥ 16/31 — plus `select_direction_seed.py` (anchors only, ~700 points per seed; one mesh load). The pin is replaced by the selected set, `KNOWN_PARITY_DISCREPANCIES` cleared and the old failure kept as `RESOLVED_PARITY_DISCREPANCIES`, and the seed + rule recorded in the docstring, the audit report and every room manifest | +286 −74 |

**Suite after exp22-r6:** 2900 passed, 10 skipped, **0 failures**, 2 subtests passed.

### The selection run (`loc_meshgrid_direction_selection.json`, log `…_direction_selection.log`)

| fact | value |
|---|---|
| rule | smallest seed s ≥ 0 with ≥ 16/31 odd parity for every anchor in all 16 rooms |
| anchors swept | **700** (sources + receivers, 16 rooms) |
| seed 0 | **FAILS** — exactly one anchor: `MeetingRoom/MeetingRoom_idx_32` receiver `[2.26, 0.48, 1.2]` at 15/31 |
| seed 1 | **PASSES** all 700, minimum 16/31 |
| selected | **seed 1**, digest `79544f2dbc880a37a4826aa527d40e99a3e54ce849cfd0ec9f1c6e847c528a8d` |
| previous pin | seed 0, digest `9ab4339f…` — confirmed to be exactly `build_directions(31, seed=0)` |
| tightest anchors under the pin | MeetingRoom_idx_20 and MeetingRoom_idx_32 receivers, both 16/31 |

**Investigation answered, not assumed:** the directive asked what happens if seed 0 passes.
It does not — the 16-room sweep reproduces the reviewer's single failure exactly, so the
r1-r5 discrepancy and the sweep were measuring the same generation. `MeetingRoom_idx_32`
is now ACCEPTED by `audit_room_anchors`, which removes the last G1 blocker.

**Margin caveat for the audit:** the selected set clears the majority by exactly one vote
(16/31) on two MeetingRoom receiver anchors. That is the rule's own bar, but it is a thin
margin worth knowing before the cost gate.

