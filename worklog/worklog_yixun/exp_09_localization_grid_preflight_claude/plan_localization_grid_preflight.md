# Plan — exp_09_localization_grid_preflight (3-D mesh-valid analysis-by-synthesis localization)

**Author:** OpenAI Codex (Planner / Analyst) · **Coder:** strongest coding-tier subagent at max effort after approval · **Reviewer:** Anthropic Claude Opus 5 (`claude-opus-5`, Claude Code 2.1.237 interactive session) · **Date:** 2026-08-20
**Status:** APPROVED by Yixun on 2026-08-20 through R0→D1→G1, with protocol amendments for the original exp_01 global-RNG context draw and, after measured B7, 31-direction ray-parity validity plus a 0.20 m source-surface prior. Yixun subsequently authorized the cache-enabled no-quality throughput probe; the measured gate selected and froze `K=4` on 2026-08-21. No localization quality score has been read.

## 0. Decision being tested

Can the frozen Vanilla FLAC 40k clean EMA checkpoint be inverted on the mesh-available AcousticRooms unseen rooms by evaluating a common, physically valid three-dimensional candidate grid and ranking generated RIRs with the frozen AGREE acoustic embedding?

This is inference-only. There is no learned localization head and no optimization loss. The quantity called “score” below follows Eq. (3) of `acoustic_localization_brief.pdf`; all other choices follow the attached experiment text plus Yixun's later overrides.

Protocol precedence is explicit: the PDF's §4 sentence about a common valid **2-D** grid at target source height is superseded by Yixun's approved isotropic **3-D** grid. The 2-D rule would leak the unknown target height. The manuscript protocol sentence and reserved Table 1 caption must be updated before exp_09 is reported. Exp_09 is also canonical-heading-only; it fills the diagnostic canonical column, not the brief's primary random-yaw column or random-yaw degradation result.

## 1. Frozen protocol

### 1.1 Data and leakage control

- Source manifest: existing `data/AR/unseen_eval.json` (6,337 queries / 17 rooms); do not author a reduced eval config.
- Exp_09 test scope, explicitly directed by Yixun on 2026-08-20: exclude exactly the 1,000 queries from `ListeningRoom_idx_2` because its official OBJ is absent. The fixed preflight denominator is **5,337 queries in 16 mesh-available rooms**. Filter this room in memory, record the exclusion in every manifest/report, and fail if any additional query is lost.
- This experiment-specific exception supersedes announcement 01 only for the missing-asset room in this mesh preflight. Results must say “mesh-available preflight subset”; they are not the complete published 6,337-query unseen-room protocol. Debugging may use further bounded records, but those numbers never enter `_results.md`.
- One query is a target RIR `h_obs = h(x*_s, x_r)` with continuous global `x*_s` and known global receiver `x_r` read from metadata.
- Context tensor width remains the released FLAC `N_ctx=8`, but uniqueness is not assumed. Materialize contexts through the unmodified released `AR_md.py` selection path under the exact exp_01 K=8 loader protocol: `seed=42`, `batch_size=64`, `num_workers=4`, `shuffle=False`, full `data/AR/unseen_eval.json` order, and `pl.seed_everything(42, workers=True)`. This preserves the released eligible-path pool, its `f"S00{node}"` S010 quirk, its global/per-worker NumPy RNG streams, and its `replace=False`→exception→`replace=True` short-pool behavior.
- Contexts must be materialized for all 6,337 records before filtering. Only after the complete original loader pass may the 1,000 `ListeningRoom_idx_2` records be excluded; filtering first would alter worker assignment/RNG consumption for the retained queries. Save the selected context identities and order in a content-hashed manifest, then reuse that frozen manifest for every candidate, stochastic generation, Vanilla arm, and later FA-FLAC arm. No model arm redraws contexts.
- The full-manifest eligible-count histogram is pinned to `{6: 91, 7: 429, 8: 5263, 9: 554}`. After excluding `ListeningRoom_idx_2`, the in-scope histogram is `{6: 91, 7: 429, 8: 4363, 9: 454}`. The 520 short-context queries all remain in `Cafe_idx_1` and receive the released loader's global-RNG replacement draws to width eight; none is dropped. The target RIR and target source remain absent.
- All candidates for a query share receiver, depth panorama, context RIRs, context poses, sample count, and seeds. Only the candidate source pose changes.
- Global metadata coordinates are converted to receiver-relative FLAC coordinates exactly once at the model boundary: `candidate_relative = candidate_global - receiver_global`; context poses use the existing same transform.

### 1.2 Three-dimensional candidate grid

- Spacing: `delta = (0.5, 0.5, 0.5) m`.
- Lattice: for each axis, integer multiples of `0.5 m` between `ceil(aabb_min / 0.5) * 0.5` and `floor(aabb_max / 0.5) * 0.5`. The lattice is therefore room-global and independent of query and ground truth.
- Base room-valid mask (Yixun-approved B7 revision, 2026-08-20): physical validity is determined by strict-majority odd ray parity over 31 frozen non-axis-aligned directions, which is robust to the official meshes being non-watertight/non-manifold. Separately apply a source-distribution surface-clearance prior `distance + 1e-4 m >= 0.20 m`. The `1e-4 m` tolerance applies consistently to surface, receiver, context, and z-band boundaries.
- Query-valid mask: distance to the known receiver `>= 0.5 m` and a **0.25 m numerical-duplicate guard** around each selected context-source coordinate. The 0.25 m guard is half a lattice step, prevents an effectively coincident context coordinate, and caused 0.0% measured oracle@0.5 damage in the review audit; the rejected 1.0 m rule caused 21.4%.
- Observation-derived z rule (pre-registered, no target access): from the selected global context coordinates retain lattice heights in `[min(z_ctx)-0.5 m, max(z_ctx)+0.5 m]`, intersected with the mesh-valid full-height lattice. G1 computes both full-height and z-band oracle distributions over all 5,337 queries. Use the z-band globally only if every query remains nonempty and it creates zero additional `e_oracle>0.5 m` queries versus full height; otherwise use full height globally. The chosen branch is decided from geometry/oracle coverage before any FLAC generation and recorded in every manifest.
- Ground truth is never inserted or snapped into the candidates.
- Save the candidate manifest per room/query, including lattice origin, spacing, mesh identity, validity backend, base/query counts, exclusions, and SHA-256 so every arm receives byte-identical candidates.
- Every query must have a nonempty candidate set and finite `e_oracle`; otherwise G1 fails before generation. Report the continuous-grid oracle `e_oracle = min_c ||c-x*_s||`, raw error `e_loc`, excess-over-oracle error `e_excess=max(0,e_loc-e_oracle)`, and raw/oracle-normalized success side by side as co-primary readouts.

### 1.3 Geometry validity and the upstream missing-mesh exclusion

Primary backend: Open3D 0.19.0 `RaycastingScene`, using the official OBJ triangles for point occupancy and closest-surface distance. Before candidate generation, audit every mesh for parse success, finite vertices, nonempty triangles, AABB, edge/manifold/watertight diagnostics, and occupancy/distance consistency at all known source/receiver metadata anchors.

Fail-closed acceptance for a mesh-backed room:

1. one unambiguous OBJ resolves to the split room;
2. every metadata anchor is finite and inside/on the free-space classification after the documented `1e-4 m` tolerance;
3. every real metadata source anchor survives the same inside/surface-validity predicate used for candidates; any failure blocks the room rather than warning;
4. every query-valid grid is nonempty with finite oracle error, and the per-room full-height/z-band oracle distributions are emitted before generation;
5. the generated base grid is deterministic and two independently chunked calls return byte-identical candidate arrays.

Known upstream gap: official AcousticRooms commit `3c87318a...` has no `ListeningRoom_idx_2.obj`, although that room contributes 1,000 queries to the full unseen split. Per Yixun's 2026-08-20 decision:

- record the missing path, official repository commit, excluded room, and excluded query IDs/count in the run manifest and results;
- exclude exactly `ListeningRoom_idx_2` from exp_09, leaving 5,337 queries / 16 rooms;
- do not build the previously proposed multi-depth fallback;
- do not use a convex hull, AABB-only rule, single panorama, nearest room mesh, or any substitute geometry;
- fail closed if a mesh is missing for any of the 16 included rooms.

The authoritative mesh is still required before a future result can claim the complete 17-room/6,337-query protocol.

### 1.4 Frozen model and score

- Experiment-1 arm: `/home/zhixuanzhao/projects/Frame_Average/Checkpoint/P1_40k_clean_hybrid_EMA.ckpt`, SHA-256 `da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643`.
- Frozen AGREE used by the probe: `/home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt`, SHA-256 `3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787`.
- Model config: existing `src/configs/model_configs/FLAC/AR/FLAC_AR.json`, SHA-256 `f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9`; dataset config: existing `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json`. `FLAC_AR.json` is the architecture that strictly loads the frozen checkpoint with zero missing/unexpected keys; the previously named `FLAC_AR_InContext.json` has a 256-vs-512 global-embedding mismatch and is rejected. Do not author a reduced eval config.
- Default sampler settings mirror the existing FLAC evaluation path: rectified-flow discrete Euler, `steps=1`, `cfg_scale=1.0`, checkpoint load integrity fail-closed, frozen/eval mode. No CFG sweep is part of exp_09.
- Nominal stochastic protocol: `K=4`, with the pre-registered runtime-only fallback in §1.5; `tau=0.1` stays fixed in every branch. Samples use counter-based deterministic seeds independent of batching.
- AGREE input parity is pinned to the reviewed FLAC retrieval path: observed RIRs come from the existing dataloader crop, mono float32 `[B,1,10240]` at 22,050 Hz; generated RIRs are decoded to the same shape and clamped to `[-1,1]` exactly as `eval_FLAC.py`. Reuse `Retrieval.compute_audio_features` or extract one shared helper used by both metric and localization paths; do not create an independent preprocessing implementation. The observation is encoded once per query and generated RIRs use the same `encode_audio(..., normalize=True)` path. Then

  `s[x,k] = cosine(E_a(h_obs), E_a(h_hat[x,k]))`

  `S[x] = tau * (logsumexp_k(s[x,k] / tau) - log(K))`.

  Use float32 accumulation and the stable `torch.logsumexp` form. Prediction is deterministic tie-breaking argmax by lexicographically sorted global candidate index. Also save `S_mean[x]=mean_k s[x,k]` and its argmax as a zero-cost diagnostic; it never replaces the PDF-controlled headline score. A pre-registered debug slice records the empirical cosine spread relative to fixed `tau=0.1` but cannot tune tau.
- The heatmap softmax temperature `T` is visualization-only and cannot affect the predicted candidate. Default `T=0.1`, labeled uncalibrated.

### 1.5 Conditioning cache and pre-registered compute ladder

- Vanilla query cache contract: compute `context_poses_vit`, `context_poses`, and `context_audio` once per query; recompute only the source branches for candidate changes. A bit-identity test must show the cached and uncached prepend-conditioning tokens are identical on the same inputs. For the later FA-BF cost probe, the released cylindrical transform makes `context_poses.dphi` candidate-dependent; therefore cache only `context_poses_vit` and `context_audio` per query and recompute the cheap `context_poses` branch inside each scored batch. This dependency split is mandatory and tested against the released full C4 path.
- Receiver-candidate cache: within a receiver group, `source_vit` and `source` depend on receiver/depth/candidate but not which target source was held out. Compute them once for the union of that receiver's candidates and reuse them across its target queries, with an uncached-vs-cached equality test and bounded in-memory lifetime per receiver.
- **Post-G1 gate before I1:** G1 reports exact valid candidate-query pairs for both full-height and admissible z-band branches, unique receiver-candidate pairs, estimated conditioner calls, and artifact bytes. Stop and present these geometry-only counts to Yixun; I1 cannot open until this second cost gate is accepted.
- Runtime ladder, chosen globally from a cache-enabled no-quality throughput probe and frozen before reading any localization score: use `K=4` if projected full execution is `<=168 GPU-hours`; otherwise `K=2` if that projection is `<=168 GPU-hours`; otherwise use `K=1`. If cached `K=1` still projects above 168 GPU-hours, do not change spacing, candidates, or queries—request more compute/time and remain blocked. Every later model arm uses the same selected K and candidate manifests.
- **Measured decision (2026-08-21):** final same-engine A6000 projections including one model startup are 70.34 GPU-hours for Vanilla and 73.52 GPU-hours for FA-BF at `K=4`, 143.86 GPU-hours serial total. Recomputing with the slowest individual measured batch gives 157.32 GPU-hours total. Both satisfy the 168-hour ladder, so `K=4` is selected globally. The optional 10% operations reserve is scheduling guidance, not a different ladder input.

## 2. Evaluation and controls

Primary canonical-heading preflight metrics on the fixed 5,337-query / 16-room mesh-available subset, aggregated first per room and bootstrapped by room:

- median and mean continuous Euclidean localization error `e_loc` (m);
- raw success `1[e_loc<=r]` and oracle-normalized success `1[e_excess<=r]` at `r in {0.5,1.0} m`, reported side by side as co-primary;
- median/mean `e_oracle` and `e_excess`, plus per-room oracle distributions and the fraction with `e_oracle>0.5 m`;
- 95% room-bootstrap confidence intervals;
- latency per query, candidate, and generated RIR.

Controls under the identical candidate manifest:

- deterministic uniform-random candidate baseline, repeated with pre-registered seeds;
- AGREE oracle retrieval using real candidate-bank RIRs only where an exact dataset RIR exists, labeled sparse/metadata-bank and not confused with the dense-grid model oracle;
- off-grid truth probe on exactly the lexicographically first query from each of the 16 included rooms: generate at continuous `x*_s`, record its score/rank against grid candidates, but never insert it into the argmax candidate set;
- real-vs-generated AGREE calibration: for the same 16 fixed probe queries, compare `cos(E(h_obs),E(h_real,other))` with `cos(E(h_obs),E(h_generated))` and report both distributions to diagnose embedding domain gap;
- save the mean-aggregated-score localization metrics beside the PDF log-mean-exp metrics and report score/candidate-count associations; these are diagnostics only;
- score ablations (waveform / multiscale STFT) are deferred unless separately approved; they are not needed for the first Vanilla verdict.

Visualizations are selected by pre-registered quantiles after all queries finish: lowest-error sharp case, median-error/ambiguous case, and highest-error failure case, with boundary/valid region, receiver, continuous truth, prediction, and uncalibrated normalized score. Selection cannot be hand-picked.

Primary success criterion: Vanilla FLAC's raw and oracle-normalized median/error success readouts beat the room-matched random baseline with non-overlapping or clearly shifted room-bootstrap intervals. Reliability gates (candidate/oracle coverage, exact 5,337-query subset coverage, load integrity, leakage guards, AGREE parity) must pass before interpreting localization quality. These numbers are canonical-coordinate diagnostics only and cannot fill the brief's primary random-yaw result.

## 3. Planned files and TDD rounds

Tests remain in `src/tests/` per announcement 02. Each round is test-first (red), minimal implementation (green), review/fix/reverification, and one small commit generally below 200 changed code lines.

### R0 — restore the permanent-suite baseline and geometry runtime

- Fix `src/tests/test_eval_paths.py` before extending it: resolve the repository root by walking parents to the `.git` marker, then load the comparator from `worklog/worklog_yixun/exp_02_yaw_noninvariance_claude`. This closes the announcement-03 migration regression that currently prevents test collection.
- Add `open3d==0.19.0` to `pyproject.toml` and install it in the FLAC runtime interpreter used for `src/tests/`; record interpreter path and installed version. The fact that `/usr/bin/python3` has Open3D is not accepted as proof for the FLAC environment.
- Run the complete existing `src/tests/` suite and record an all-green, no-collection-error baseline in the worklog. R0 receives its own Opus code review and must close before G1.

### D1 — release-parity global-RNG query/context construction

Create `src/localization/ar_queries.py`:

- parse existing split entries and metadata into immutable query records;
- load/pad/crop the observed RIR exactly like `AR_md.py`;
- invoke the original K=8 evaluation loader over the complete 6,337-record split with exp_01's seed-42/batch-64/four-worker/no-shuffle settings, preserving its global/per-worker NumPy draw order, S010 quirk, and replacement fallback;
- record the selected context identities/order, protocol settings, full split hash, and manifest hash before excluding `ListeningRoom_idx_2`; reloads use the frozen manifest and never redraw contexts;
- materialize one shared FLAC metadata/context object and clone it with one candidate-relative pose;
- emit stable candidate/sample seeds independent of batching while treating the frozen context manifest—not a query-local RNG—as the context source of truth.

Create `src/tests/test_localization_ar_queries.py` first. Tests: IR-versus-metadata filename conventions, including the released S010 quirk; full 17-room/6,337-record source-manifest parse invariant; exact exclusion of the 1,000 `ListeningRoom_idx_2` records yields 16 rooms/5,337 records; any additional omission fails; full/in-scope eligible-count histograms equal `{6:91,7:429,8:5263,9:554}` / `{6:91,7:429,8:4363,9:454}`; all 520 short queries receive width-eight replacement draws and are not dropped; target source/RIR excluded; a reference harness proves the materializer matches the original loader for fixed exp_01 settings; changing/filtering the split before materialization is rejected; manifest reload is byte-stable and arm-shared; candidate cloning changes only `source`/`source_vit`; global-to-relative transform; dataloader audio length/rate checks.

### G1 — geometry audit primitives

Create `src/localization/geometry.py`:

- `snap_axis_to_lattice(bounds, spacing)` — exact global lattice endpoints;
- `build_lattice(aabb_min, aabb_max, spacing)` — stable lexicographic `float32/float64` candidate order;
- `load_raycast_scene(mesh_path)` — fail-closed OBJ load and identity metadata;
- `classify_free_space(scene, points)` — bounded, deterministic 31-direction odd-ray-parity majority validity; and `classify_mesh_candidates(scene, points, surface_clearance)` — validity plus the separate distance prior;
- `filter_query_candidates(points, receiver, context_sources, ...)` — receiver/context/z-band clearances;
- `grid_oracle_error(points, truth)`.

Create `src/tests/test_localization_geometry.py` first. Tests: negative/non-finite spacing rejected; negative AABB coordinates snap correctly; exact expected cubic lattice/order; synthetic room shell plus enclosed obstacle distinguishes room air/solid/outside under ray parity; exact 0.20 m surface boundary with `1e-4 m` tolerance; 0.5 m receiver and 0.25 m context boundary behavior; context-derived z-band and its global full-height fallback rule; no ground-truth insertion; per-query nonempty/finite-oracle guard; deterministic chunking; empty/missing/malformed mesh fails closed; all real source and receiver anchors pass validity and all sources pass the 0.20 m prior.

Create `tools/audit_localization_geometry.py` after its helper contracts are green. It verifies the documented missing room, audits all 16 included meshes, emits per-room source-anchor survival and full-height/z-band oracle distributions, and writes JSON/Markdown only under this experiment folder. The tool itself is included in the G1 code review.

### S1 — scoring and metric aggregation

Create `src/localization/scoring.py`:

- normalized cosine score contract;
- stable log-mean-exp for arbitrary `tau>0`;
- lexicographic stable argmax;
- visualization-only softmax;
- localization/random/oracle/excess metrics and room bootstrap.

Create `src/tests/test_localization_scoring.py` first. Tests: `K=1`; equal scores return the same score; small-`tau` approaches max; numerical stability at extreme logits; fixed-tau and mean-score outputs; invalid tau/shape/NaN rejection; tie rule; visualization temperature cannot change prediction; exact `e_loc/e_oracle/e_excess` and raw/oracle-normalized success; room bootstrap resamples rooms, not RIR rows.

### I1 — frozen FLAC/AGREE localization engine

Create `src/localization/engine.py` and `localize_FLAC.py`:

- reuse/refactor the already-reviewed checkpoint-integrity and Vanilla/FA conditioner dispatch from `eval_FLAC.py` without changing legacy evaluation behavior;
- freeze/load FLAC and AGREE once;
- encode observation once through the shared Retrieval preprocessing helper;
- cache context branches once per query and source branches once per receiver-candidate union, while bounding memory to one receiver group;
- batch candidates and `K` samples with batch-invariant seeds;
- generate/decode RIRs, encode AGREE audio, score, resume from atomic per-query artifacts, and write a schema-versioned manifest/result;
- expose `vanilla` now and `fa_invariant` for the later matched arm, with no arm-specific candidate logic.

Create `src/tests/test_localization_engine.py` and extend the now-green `src/tests/test_eval_paths.py` first. Tests: clean/dirty checkpoint load; all params frozen/eval; localization and `Retrieval.compute_audio_features` return identical embeddings for the same mono float32 `[B,1,10240]` waveform; generated clamp parity; observation encoded once; cached and uncached context tokens bit-identical; receiver-candidate source-cache and uncached tokens bit-identical; candidate batch shape/metadata; exact K seeds and batch-size invariance; fake sampler recovers the known synthetic winner; Vanilla/FA dispatch; interrupted resume skips only complete verified queries; config/split/checkpoint/candidate/K/cache hash mismatch refuses resume; output stays inside `NeuriPs_Workshop`.

### R1 — reporting

Create `tools/aggregate_localization_results.py` and `tools/render_localization_results.py`, with tests in `src/tests/test_localization_reporting.py`. Tests: exact 5,337-query/16-room coverage plus explicit 1,000-query `ListeningRoom_idx_2` exclusion; no other duplicate/missing IDs; per-room then room-bootstrap aggregation; random seed accounting; raw/oracle/excess and mean-score diagnostics; fixed 16-query off-grid/calibration probe coverage; candidate-count association; canonical-only and “mesh-available preflight subset” labels; quantile case selection; Markdown/HTML values agree with aggregate JSON. Both scripts enter review before use.

## 4. Validation ladder and launch gates

0. **R0 precondition:** pin/install Open3D in the actual FLAC interpreter, repair `test_eval_paths.py`, and record the complete pre-exp_09 suite green with no collection errors.
1. Static per round: `py_compile`, config JSON parse, `git diff --check`.
2. Permanent tests per round: new localization tests plus all existing `src/tests/`; no round advances on red.
3. D1 full census: run the exp_01-compatible global-RNG materializer on all 6,337 queries, prove its protocol and manifest hash, then prove the exact 5,337 post-materialization exclusion, both context histograms, replacement behavior, and leakage guards.
4. G1 synthetic then real geometry: box mesh first; then verify missing `ListeningRoom_idx_2`, audit all 16 included meshes, real-anchor survival, per-query nonempty sets, full-height/z-band candidate counts and oracle distributions.
5. **Post-G1 user cost gate before I1:** publish exact geometry-only counts/call/storage estimates and the pre-registered z-band branch. Stop until Yixun accepts the cost evidence.
6. I1 tiny synthetic forward with fake FLAC/AGREE, including cached-vs-uncached token identity and resume/hash guards.
7. Real-data readback: several records; record 22.05 kHz, `[B,1,10240]`, dtype/range/min/max/std, context transforms, generated clamp, and exact localization-vs-Retrieval embedding equality.
8. Cache-enabled, no-quality throughput probe chooses one global `K` by §1.5's `4 -> 2 -> 1`/168-GPU-hour rule; record the selected branch before reading scores.
9. One real query bounded-candidate smoke, then one complete-room debugging smoke: verify resume, artifacts, memory, and measured throughput. These are debugging-only and never enter `_results.md`.
10. Independent full-diff Opus review, parity audit against `eval_FLAC.py`/AGREE, params + exact commands + acceptance criteria recorded at launch.
11. Fixed 5,337-query / 16-room mesh-available preflight run and controls, followed by explicitly subset-labeled results, reliability analysis, offline HTML, and commits log.

## 5. Explicitly out of scope for exp_09

- training or fine-tuning FLAC/AGREE/localization heads;
- changing `0.5 m` grid spacing after reading quality results;
- inserting ground truth or snapping it into candidates;
- any query reduction beyond the explicitly excluded 1,000-query `ListeningRoom_idx_2` room;
- yaw augmentation, randomized headings, HAA, cylindrical/SSP models, or the FA-BF headline comparison (the engine is made compatible, but those are later matched arms);
- exact flow likelihood or calibrated posterior claims;
- reconstructing or substituting geometry for `ListeningRoom_idx_2` in exp_09.

## 6. Risks requiring review/approval

1. **Subset claim boundary:** Yixun resolved the missing mesh by excluding `ListeningRoom_idx_2`. Every result must expose the 5,337/16 denominator and cannot be described as the complete unseen split.
2. **Compute scale:** the review measured 25,312,262 raw candidate-query pairs before mesh/z masking. Query/receiver caching, the geometry-only post-G1 gate, and the pre-registered global `K=4 -> 2 -> 1` runtime ladder prevent post-quality protocol tuning; no spacing/query reduction is automatic.
3. **Mesh topology:** architectural OBJs may be non-watertight or self-intersecting. Metadata-anchor validation is the empirical fail-closed criterion; topology warnings cannot be ignored merely because a grid is nonempty.
4. **AGREE input parity:** mono float32 `[B,1,10240]`, dataloader crop, generated clamp, and the shared Retrieval embedding helper are mandatory and byte/number audited.
5. **Context parity:** exp_09 intentionally preserves the released exp_01 seed-42 global/per-worker RNG path, S010 eligibility quirk, and replacement draws for 520 Cafe queries. The full 6,337-query context manifest is frozen before the mesh-missing room is filtered, then shared by every arm.

## 7. Review-response ledger and approval decisions

- **B1 accepted:** replace 1.0 m context exclusion with the reviewer-recommended 0.25 m duplicate guard; add per-query nonempty/finite-oracle gates.
- **B2 option (a), amended by Yixun:** preserve the released eligible context pool including the S010 quirk and use the exact exp_01 seed-42 global/per-worker RNG replacement path for the 520 short Cafe queries; materialize all 6,337 first, freeze the manifest, then filter the missing-mesh room; never drop a short-context query.
- **B3 accepted:** cache query- and receiver-candidate-invariant conditioning, move exact counts/cost approval after G1 and before I1, adopt context-derived z-band only under its no-new-unwinnable gate, and pre-register the global `K=4 -> 2 -> 1` runtime ladder.
- **B4 accepted:** R0 repairs the stale worklog path and must establish the green permanent-suite baseline before localization code.
- **B5 accepted:** bind localization to mono float32 10,240-sample Retrieval preprocessing and generated clamping, with an embedding-equality test/audit.
- **B6 accepted:** use `eps=1e-4 m` for every clearance boundary and fail if a real source anchor does not survive.
- **N1–N3 accepted:** pre-register conditional context z-band, document 3-D's explicit precedence over the PDF's 2-D sentence, and label exp_09 canonical-only.
- **N4–N7 accepted:** save mean score, fixed 16-room off-grid truth probes, raw/oracle/excess co-primary readouts, and real-vs-generated AGREE calibration.
- **N8 accepted:** keep `cfg_scale=1.0`; no CFG sweep.
- **N9 accepted:** pin/install Open3D 0.19.0 in the interpreter that runs this repository before G1.

Yixun's 2026-08-20 approval, including the global-RNG amendment above, authorizes only R0 and the subsequent small reviewed TDD rounds through G1 plus the geometry/cost analysis. The workflow stops again after G1 for the exact cost gate before I1, so no expensive generation is implied by this approval.
