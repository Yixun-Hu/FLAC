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


## Micro-round exp22-r6b (real G1 audit abort — metadata pair naming)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `2962330` | numeric-tolerant pair lookup | `audit_meshgrid_geometry._metadata_for` reconstructed `f"{source_node}_{receiver_node}.json"` from the IR filename, producing `S007_R019.json` where the release stores `S007_R0019.json` — the release writes `"S00" + str(src) + "_R00" + str(rec)`, so no single reconstructed format matches both a two-digit and a single-digit receiver. It now parses `(src, rec)` with `candidates.parse_ir_filename` and resolves the file by **parsed numeric identity over the directory listing** via `candidates.find_pair_metadata`, reusing those semantics rather than re-deriving them. A missing pair still raises the same fail-closed `ValueError`, now naming `(S<src>, R<rec>)` and the room directory | +27 −8 |

**Suite after exp22-r6b:** 2907 passed, 10 skipped, **0 failures**, 2 subtests passed.

**Tests (7, RED first — 5 failed / 2 passed before the fix; the 2 were the cases where the
reconstructed name coincidentally equalled the stored one):** two-digit receiver stored as
`S007_R0019.json`; single-digit `S007_R008.json`; `S010`-style source `S0010_R0015.json`;
three-digit `S001_R00100.json`; the missing-pair refusal; a probe asserting no single
reconstructed format can satisfy the mixed-format directory; and a real-data test against
`AcousticRooms/metadata/MeetingRoom/MeetingRoom_idx_32` (skipped where the dataset is absent).

**D1 is unaffected — verified, not assumed.** `src/localization/meshgrid_queries.py` builds
exactly one filename, `f"S00{node}_{receiver}_hybrid_IR.wav"` in `eligible_context_pool`
(line 101), which mirrors released `AR_md.py:99` and is an IR wav name, not a metadata name.
It constructs no `.json` path at all: its only `.json` strings are the model config, the
`AR_md.py` module path and `data/AR/unseen_eval.json`. Pair metadata is read solely inside
the **unmodified released** `AR_md.get_receiver_source_location`, which uses the release's own
concatenation and is therefore correct by construction. The committed manifest hashes
(full `15d229c0…`, filtered `99f8da60…`) are unchanged by this round.

## Round exp22-r7 (the I1 engine — inherited plan §1.4/§1.5 + Yixun's decisions)

| SHA | Item | Description | changed lines (code / tests) |
|---|---|---|---|
| `8e0cbc8` | noise + score core | `noise_key(seed, query_id, candidate_index, k)`, `noise_block` (candidate-major, chunk invariant by construction), `nested_scores` reading K ∈ {1,4,8} from prefixes of ONE 8-sample sequence at τ = 0.1, `argmax_by_global_index` (tie-break stated on the GLOBAL candidate index, so a re-ordered slice cannot move a prediction), `score_query` publishing the log-mean-exp headline and the `S_mean` diagnostic in exact float32 hex | +208 / +182 |
| `08b74b9` | the two caches | conditioning split along §1.5's boundary through the released `MultiConditioner.only_ids` seam: context branch once per QUERY, source branch once per (receiver, candidate) over the ascending union. `ReceiverCache` is single-instance and carries the digest of the panorama it was built from — `source_vit` reads `depth`, so a receiver whose queries disagreed about it would be silently mis-served. Cached-vs-uncached assembly pinned bit-identical; chunking proven to change batching only | +167 / +148 |
| `6d8d7c5` | D1 + G1 bindings | the four published-artifact verifiers move from the audit driver into `meshgrid_geometry` and are re-exported (one implementation; no logic change; audit suite green). `load_audit_plan` re-accepts the whole G1 chain before a query is scored and refuses a diagnostics-only report or a branch other than the audit's. `verify_context_record` makes the D1 manifest EXECUTABLE — fingerprints and every context RIR's float32 sha256 recomputed from the live stream before the draw conditions anything. `assert_receiver_consistent` recomputes G1's oracle from the loader's own `md['source']` | +351 −93 / +213 |
| `5db5f06` | binding, artifacts, resume, dumps, probe | 17-field run binding under crossarm's type-sensitive canonical digest; sidecar-first atomic publication (the row is the completion marker and carries the sidecar's digest); `completed_queries` re-verifies both files so a resume skips only what still re-accepts; dumps bounded to the registered probe set + a digest-carried case list (announcement 08); `probe_record` cannot carry a score by construction | +255 / +147 |
| `ac051d4` | the pass | `run_pass` walks the released loader ONCE in D1 order, verifies each draw, conditions the context branch once, embeds the observation once, and then runs the room receiver-group by receiver-group with exactly one cache resident. Rooms arrive as contiguous blocks — asserted, and true on the registered split. Generation is candidate-major in `batch_rows // K` chunks; identical scores at batch 32 and batch 1 | +297 −5 / +215 −2 |
| `c8d8a01` | real stack + driver | `build_mesh_engine` (eval_FLAC's lines of record + the `only_ids` seam), `cache_parity_check` (§1.5's proof on the REAL conditioner), `assert_release_rng_state`, and `localize_meshgrid.py` | +457 −16 / +101 |
| `a31c357` | source-chunk default | `source_vit` is a `GeometryConditioner`: every candidate is a full ViT forward over a `[3, 256, 512]` map, so the default chunk is 16, not 256 | +14 −5 / — |

| `12bff5c` | probe accounting + progress | the source cache is billed to the GROUP under a name that says so (summing a per-query key would count it k times); the driver prints a live rate and ETA every 25 queries, because the registered pass runs for days | +30 −6 / +3 |
| `e9e5b81` | bounded dumps | an admitted dump now writes something. exp_18 dumped every candidate because M was ~10; here M averages 1,667, so a full dump is 546 MB/query and 1.7 TB/pass. A dumped query keeps every prefix's prediction, every prefix's `S_mean` prediction and the top-N at the largest prefix, plus the observation — REGENERATED from their own noise keys after scoring, so nothing is held in memory and a test proves the regenerated waveforms reproduce the similarities that were scored | +102 −4 / +57 |
| `d9fdc75` | parity proof split | see the real-stack finding below | +79 −20 / +62 |
| `321cf16` | advisory binding tier + driver fix | `source_chunk`/`batch_rows` are recorded, compared and REPORTED on resume rather than refused (binding them strictly would make an OOM unrecoverable); fixes a `NameError` that killed `--cache-parity-check` after the model load and that no fixture test could reach | +45 −8 / +14 |
| `43c2e37` | ARE refusal ordering | both checkpoint refusals (ARE artifact, conditioning binding) are CPU-only file reads and now run together before anything is built or moved — exp_18's r3 finding 9, which the engine build had re-introduced | +25 −7 / +17 |
| `4eceb6d` | probe-room, diagnostics claim nothing, measured dtype | `--probe-room` bounds the probe to one room (Cafe's smallest group is 380 k waveforms, so there was no affordable real smoke); `writes_query_artifacts` stops a probe or parity check leaving a binding a scored pass would resume; `BATCHING_CAVEAT` records the MEASURED float16 ulp instead of an assumed bfloat16 one | +38 −16 / +14 |

**Suite after exp22-r7:** **2,989 passed, 10 skipped, 0 failures**, 2 subtests passed (11 m 37 s) — 2,907 before the round plus 82 new engine tests.

### Real-artifact cross-check (not a fixture)

`load_audit_plan` re-verified the published 16-room G1 audit in 48.5 s, and the engine's own
`receiver_groups` accounting reproduces the cost gate **exactly**: **8,896,540** candidate-query
pairs and **966,147** union members (= source-conditioner calls) — the same numbers the audit
published. The 16 registered off-grid probe queries were computed from the manifests
(`S001_R001` in most rooms; `Cafe/Cafe_idx_1` → `1|…/S001_R040_hybrid_IR.wav`).

### Declared deviations, stamped in the binding and in every row

1. **Scorer readout** — inherited §1.4 names `encode_audio(..., normalize=True)`; this engine uses
   exp_18/exp_20's deterministic VAE-mean readout, because the sampled path draws from AGREE's
   bottleneck (~7e-5 cosine noise, exp_18 measurement) and consumes the global RNG stream.
2. **Noise key** — the dispatched key is `(seed, query_id, candidate_index, k)`; inherited §1.1
   says candidates of a query share their seeds (common random numbers). The dispatched key is
   the default and `shared_across_candidates` is implemented and selectable, so a ruling either
   way costs no code round.
3. **Sidecar granularity** — per-QUERY float16 `.npy`, not per-room `.npz`: the atomic-resume
   contract is per query, and a room-level pack would lose finished queries on a mid-room kill.
4. **Dump content** (`DUMP_CONTENT_RULE`) — a bounded, score-derived selection, because the
   exp_18 dump-everything rule is 1.7 TB here. The rule is stamped in every dumped row.
5. **`--probe-room`** — an addition beyond the dispatch, diagnostics-only, without which no
   affordable real smoke of the pass exists (the split's first room has no small group).

### The real-stack §1.5 parity finding (`d9fdc75`)

The first run of `--cache-parity-check` on the frozen P1 checkpoint reported **MISMATCH on all
five branches**, max |diff| 2e-3. That was not a cache error: the cached side computes the
source branch in chunks and the context branch at batch 1, while the uncached side ran one call
over 64 candidates, and the conditioners run in **float16** (`--cond-autocast default` on CUDA),
where a changed batch shape moves an output by ~1 ulp. A check that cannot separate those two
proves nothing, so it was rebuilt to answer them apart:

| half | question | result on the real conditioner |
|---|---|---|
| `memoization` | same batching on both sides (one candidate per call, cache chunk 1) — does the cache serve what the direct call computes, through the released `only_ids` seam? | **MATCH — all five branches bit-identical, max \|diff\| 0.0** |
| `batched` | cache at its production chunk vs one uncached call, 32 candidates over two chunks of 16 | informational, and mechanistically sensible: `context_audio` 3.9e-3 and `context_poses_vit` 2.0e-3 (batch 1 vs batch 32), `source_vit` 2.0e-3 (2×16 vs 1×32), `context_poses` 7.6e-6, and **`source` exactly 0.0** — the one branch that is elementwise and therefore batch-invariant |
| `counter_test` | can the comparison fail at all? | **bit** — perturbed positions move it by 2.12 |

The memoization half also proves the `only_ids` split is FAITHFUL, since the cached side asks
for one branch and the direct side asks for all five. The batching asymmetry is disclosed in
`BATCHING_CAVEAT`, published in every run binding.

### Real-stack smoke of the whole pass (NOT the registered probe — that is the ladder step)

`--probe 1 --probe-room Bathrooms/Bathrooms_idx_14` ran the complete engine against the frozen
P1 checkpoint on cuda:0: released call graph reproduced, RNG state matched D1's
`rng_digest_at_iter`, 4,422 stream positions verified against the frozen manifest, checkpoint
loaded EMA with 0 missing / 0 stray, one receiver group of **10 queries / 276 candidate-query
pairs / 2,208 generated waveforms**, **no scores and no `rows/` directory written**.

| component | per query (224 waveforms) | share |
|---|---|---|
| decode (VAE) | 1.42 s | **83 %** |
| embed (AGREE) | 0.21 s | 12 % |
| sampling (DiT, 1 step) | 0.087 s | 5 % |
| conditioning (cache expand) | 0.002 s | 0.1 % |
| source cache | 0.075 s for the whole GROUP | — |
| context branch + observation | 0.17 s per query, phase 1 | — |

**7.67 ms per generated waveform at the default `--batch-rows 64`**, i.e. ~**152 GPU-hours** for
the 71,172,320-waveform pass — inside the 175 h stop-rule band, and dominated 83 % by the VAE
decode at a batch of 64 rows, which is the obvious knob for the registered probe to sweep.
Treat this as a wiring smoke on one small room, not a cost measurement: it is a single
27-candidate-per-query room and the coordinator's probe owns the number.

## Round exp22-r8 (r7 review — 7 BLOCKERs + 1 MAJOR, all items)

| SHA | Item | Description | changed lines (code / tests) |
|---|---|---|---|
| `3f56cef` | CRN | `REGISTERED_NOISE_POLICY = shared_across_candidates` is the default everywhere; `run_pass` refuses anything else without an explicit opt-in and the driver refuses `--noise-policy per_candidate` outright. Inherited §1.1 fixes the shared draw, and it is the variance reduction the comparison rests on: with one draw per query, a score difference between two candidates is a difference between the CANDIDATES. The per-candidate key stays reachable by opt-in so the r7 evidence is reproducible | +11 −4 / +33 −4 |
| `783be68` | GT prep | every fixture candidate set is now DERIVED by the same `meshgrid_geometry` filter the audit ran, from a real 0.5 m lattice with real receivers and contexts: the receiver guard, the 0.25 m context guard and the z-band all bite, two queries share a receiver with different contexts, and the branch choice changes the answer. Precondition for the GT removal — reconstructing a candidate set is only a test if the fixture's sets were filtered, not written down | +0 / +173 −119 |
| `fc873c6` | GT | all three doors closed. `assert_query_geometry_consistent` replaces the oracle cross-check and proves MORE without the target: from the manifest receiver, the live context poses and the base bank it re-derives the z-band, both drop counts and the entire candidate index set. A differing candidate is tolerated only within `CONTEXT_JOIN_TOLERANCE` of the boundary deciding it, and every tolerated case is counted and summed into the summary. `context_record` grows `prove_target_absent`: D1 materialization still proves absence, the engine REQUIRES the frozen verdict. `GuardedMetadata` makes the rest structural — `source`/`source_vit` stay visible but any read, `dict()` copy included, raises `LeakageError` | +202 −41 / +144 −14 |
| `747cd37` | BINDING | `cond_autocast` and `dataset_config_sha256` become strict binding fields (a resume could otherwise mix float16/float32 tokens or an edited split); every row and the summary record the batching they were produced with, and an advisory change on resume is appended to the published binding's `advisory_history` with a timestamp instead of scrolling past in stdout | +40 −4 / +63 |
| `57017a6` | RESUME | `--resume` without a published binding is a refusal (it used to CREATE one, legitimizing whatever rows were there); every row carries `row_sha256` over everything it claims plus its binding digest; and `assert_published_matches` checks a skipped query's identity, receiver, branch and full candidate list against the loaded G1 plan, because nothing regenerates a skipped row. Verification also follows any waveform dump a row names | +104 −13 / +89 |
| `4a5b910` | DETERMINISM | `DETERMINISM_CONTRACT` states the claim and `score_fingerprint` makes it operational: at fixed batching a replay must be bit-exact through scoring. `SCORE_TOLERANCE` is the registered bound for changed batching, and `compare_scored_runs` reports the max delta per prefix, every query whose argmax MOVED and how many sat within tolerance and could have. Rows carry their top-1 margin and an `argmax_stable` flag; the summary aggregates per K. `replay_check` (driver `--replay-check`) runs the claim with everything rebuilt; a ladder-gated test drives it against the frozen checkpoint | +193 −2 / +129 −3 |
| `a3d8b53` | PROBE | each record names its receiver group and union size, so the cache cost is attributable and not double counted, and carries its own context time. `project_cost` measures three rates against three denominators — generation per WAVEFORM, source per unique (receiver, candidate) ROW, context per QUERY — and projects each against `REGISTERED_TOTALS` (16 / 5,337 / 8,896,540 / 966,147 / 71,172,320). The artifact carries the whole strict binding, its digest, the batching, the determinism contract and the leakage caveat, and refuses to publish a probe measured under an unregistered noise policy | +109 −9 / +93 |
| `e14c68e` | SHARDING a | `--rooms` with canonical ids; empty, duplicate or unknown selections refuse. A shard still walks and verifies the COMPLETE D1 stream — the draws depend on the whole pass — while opening no candidate manifest and conditioning nothing for a room it does not own. Declared rooms are published beside the binding, pinned for that shard's resume, and kept out of the strict digest so a merge can require identical base bindings | +57 −8 / +89 |
| `b38ec2b` | SHARDING b | `merge_shards` publishes into a FRESH directory only after: identical strict bindings; one pinned advisory batching, in the bindings and in every row; disjoint declared rooms whose union is exactly the registered set; exactly the registered query identities at their registered positions; no duplicate and no extra row; every row/sidecar/waveform digest re-verified against the binding and every row's identity re-checked against the G1 plan; and the three totals. The merged directory is itself a complete, re-verifiable run | +205 −2 / +137 |
| `0895184` | DUMP | `load_dump_cases` REQUIRES the registered sha256 and compares it — an unregistered or edited list can no longer extend the announcement-08 exemption by existing — and the summary records which list and which digest authorized each dumped query | +40 −10 / +61 −3 |

### Real-stack re-verification after the r8 fixes

`--cache-parity-check` re-run against the frozen P1 checkpoint on cuda:0 with the GT guard,
the CRN default and the new binding fields in place:

- **no `LeakageError`** — the released `MultiConditioner`, asked for the context branch through
  `only_ids`, never reaches for `md['source']`; the whole driver path runs with the target
  structurally unreadable;
- the D1 binding still holds: released call graph reproduced, RNG digest matched at iterator
  creation, and the frozen manifest's `target_absent` (True for all 5,337 records) is what the
  engine now requires in place of re-deriving it;
- **MEMOIZATION: MATCH** (all five branches bit-identical, counter-test bit), `BATCHED` 3.91e-3
  at float16 — unchanged from r7, as expected: none of the r8 fixes touch the caches.

The reviewer's exact probe-sweep and shard command lines parse and validate against the new
driver (`--branch z_band --noise-policy shared_across_candidates --probe-room … --rooms …`);
running them is the ladder's, not this round's.

**Suite after exp22-r8:** **3,036 passed, 11 skipped, 0 failures**, 2 subtests passed
(10 m 51 s) — 2,989 before the round; +47 net (the engine file goes 79 → 129 tests, and the
r7 GT/oracle test is replaced by the GT-free reconstruction ones). The one added skip is the
ladder-gated real-stack replay (`EXP22_REAL_STACK=1`).

## Micro-round exp22-r8b (r8 re-review — 5 RESOLVED / 3 PARTIAL)

| SHA | Item | Description | changed lines (code / tests) |
|---|---|---|---|
| `718fc35` | 5 · stability boundary | a per-score bound of ε moves a GAP by up to 2ε — the leader can lose ε while the runner-up gains it — so r8's `margin > ε` declared exactly the band that can flip (ε, 2ε] stable. `ARGMAX_STABILITY_FACTOR` / `argmax_stability_bound` / `is_argmax_stable` are now the single place the rule lives; rows, the run summary and `compare_scored_runs` all apply it and publish the bound they applied. A 1.5ε margin is pinned at-risk in all three | +29 −6 / +63 |
| `486c48f` | 7 · merge trust | the merge compared shards by their STORED `binding_sha256` — the very string a tampered shard keeps saying — so every digest is recomputed from the binding's own content and a stored value that disagrees refuses; all later comparisons use the recomputed one. The source-row census summed each shard's FINAL `run_summary`, which a normal restart legitimately reduces, so a shard resumed after completing receiver groups could not merge; the census is now DERIVED from the G1 plan (one row per (receiver, candidate) in each receiver's union over exactly the merged rooms), which is restart-invariant by construction, with the observed number published beside it | +29 −3 / +49 |
| `16b3ff1` | 8 · dump authority + nit | the registered case-list digest becomes a STRICT binding field: changing it, adding it or dropping it all refuse, so a resume can no longer skip an already-complete query while owing it a newly requested dump — and shards must share one dump authority to merge. The driver docstring's rejected per-candidate deviation text is replaced by the CRN statement | +10 −4 / +51 |

**Suite after exp22-r8b:** **3,044 passed, 11 skipped, 0 failures**, 2 subtests passed
(10 m 44 s) — 3,036 after r8 plus the 8 new r8b tests (engine file 130 → 138).
- `bc1ebea` exp_22 r9 (1): meshgrid_report.py + tests — R1 aggregation, gates-before-numbers
- `2d44d8e` exp_22 r9 (2): meshgrid_offgrid_probe.py + tests — §2 off-grid truth + calibration controls
- `07a7242` exp_22 r9b (1): meshgrid_retrieval_control.py + 76 tests — §2 sparse/metadata-bank AGREE retrieval
- `dc46a70` exp_22 r9b (2): report controls_elsewhere -> "built (r9b), run pending"
- `9ba13cf` exp_22 r9b (3): truth-pinning hardening (dense-grid oracle re-derivation gate)
- `b3f08e0` exp_22 r9b (4): context-overlap disclosure
- `0a06416` exp_22 r9c (1): report joins artifacts to the run it names (B1-B3, M6-M8, minors)
- `f92cb26` exp_22 r9c (2): offgrid probe gates before device, stages dumps (B4, B5, M9)
- `a6e7176` exp_22 r9e: sparse bank pre-registered (bank_digest + --print/--expect-bank-sha256), K=1 score = raw cosine, model_config into binding, full input-surface canonicality
- `52cb570` exp_22 r9d (1): report re-derives the merge receipt; pre-registered truth required (B1/B3/M6-M8)
- `cda347e` exp_22 r9d (2): probe gates before import, one-direction publish (B4/B5/M9, disclosures)
- `613193e` `94eddaa` exp_22 r9g: complete batching stamps mandatory; probe computes+compares bank digest pre-import; publication gate unified; renames inside rollback
- `fbdffc1` `e2bd859` `8528283` `40aab70` exp_22 r9h: one root for digest+scoring; byte continuity (hash-verified reads, verify-before-read); device/alpha/loader-values registered-bound
- `4e9817a` `e1d6193` exp_22 r9j: read-once truth parse; observation tied to rows; journaled crash-atomic publication; fail-closed canonicality
- `60d7a5d` exp_22 r9k: one read per pair file; verdict cannot outrank walk gates
- `926093a` exp_22 r9j2: observation PINNED by pre-registered digest (print/expect-observation-bank-sha256)
- `a94346d` `dd715f0` exp_22 r9m: rows/sidecars/probe-truth/observation all parsed from verified buffers (read-once everywhere); frozen digests re-verified unchanged
- `4315a91` `2d1034c` exp_22 r9m2: byte-level artifact snapshot; census keeps what it verified, walk held to it
- `ec5d771` `33dae58` exp_22 r9p: tie recalibrated to engine changed-batching envelope (4.14e-3), per-query deltas published; diagnosis row-verified
