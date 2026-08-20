# Plan — exp_09_localization_grid_preflight (3-D mesh-valid analysis-by-synthesis localization)

**Author:** OpenAI Codex (Planner / Analyst) · **Coder:** strongest coding-tier subagent at max effort after approval · **Reviewer:** Claude Opus 4.8, max effort, via Claude Code CLI · **Date:** 2026-08-20  
**Status:** DRAFT — independent review attempted 2026-08-20 but blocked by expired Claude authentication; implementation remains blocked on a real Opus review, Planner revision, and Yixun approval.

## 0. Decision being tested

Can the frozen Vanilla FLAC 40k clean EMA checkpoint be inverted on AcousticRooms unseen rooms by evaluating a common, physically valid three-dimensional candidate grid and ranking generated RIRs with the frozen AGREE acoustic embedding?

This is inference-only. There is no learned localization head and no optimization loss. The quantity called “score” below follows Eq. (3) of `acoustic_localization_brief.pdf`; all other choices follow the attached experiment text plus Yixun's later grid override.

## 1. Frozen protocol

### 1.1 Data and leakage control

- Headline split: existing `data/AR/unseen_eval.json`, all 6,337 queries in all 17 unseen rooms. Debugging may use bounded records but those numbers never enter `_results.md`.
- One query is a target RIR `h_obs = h(x*_s, x_r)` with continuous global `x*_s` and known global receiver `x_r` read from metadata.
- Context size is the existing FLAC `N_ctx=8`. At the same receiver, sort the nine non-target source IDs and choose eight with a deterministic RNG derived from `(protocol_seed, scene, room, receiver_id, target_source_id)`. The target RIR and target source are absent. The chosen context is materialized once and reused unchanged for every candidate and every stochastic draw of that query.
- All candidates for a query share receiver, depth panorama, context RIRs, context poses, sample count, and seeds. Only the candidate source pose changes.
- Global metadata coordinates are converted to receiver-relative FLAC coordinates exactly once at the model boundary: `candidate_relative = candidate_global - receiver_global`; context poses use the existing same transform.

### 1.2 Three-dimensional candidate grid

- Spacing: `delta = (0.5, 0.5, 0.5) m`.
- Lattice: for each axis, integer multiples of `0.5 m` between `ceil(aabb_min / 0.5) * 0.5` and `floor(aabb_max / 0.5) * 0.5`. The lattice is therefore room-global and independent of query and ground truth.
- Base room-valid mask: inside the room free space and unsigned distance to the nearest mesh surface `>= 0.5 m`, matching AcousticRooms' published source-placement clearance.
- Query-valid mask: distance to the known receiver `>= 0.5 m` and distance to every selected context-source location `>= 1.0 m`. The context exclusion prevents an exact context position/RIR from becoming a privileged candidate and matches the dataset's published inter-source separation. The threshold comparison is tolerant at numerical boundaries (`distance + eps >= threshold`).
- Ground truth is never inserted or snapped into the candidates.
- Save the candidate manifest per room/query, including lattice origin, spacing, mesh identity, validity backend, base/query counts, exclusions, and SHA-256 so every arm receives byte-identical candidates.
- Report the continuous-grid oracle `e_oracle = min_c ||c-x*_s||`. If `e_oracle > 0.5 m`, the query cannot count as a model success at 0.5 m; report both raw success and oracle-normalized diagnostic coverage, without redefining the headline metric.

### 1.3 Geometry validity and the upstream missing-mesh gate

Primary backend: Open3D 0.19.0 `RaycastingScene`, using the official OBJ triangles for point occupancy and closest-surface distance. Before candidate generation, audit every mesh for parse success, finite vertices, nonempty triangles, AABB, edge/manifold/watertight diagnostics, and occupancy/distance consistency at all known source/receiver metadata anchors.

Fail-closed acceptance for a mesh-backed room:

1. one unambiguous OBJ resolves to the split room;
2. every metadata anchor is finite and inside/on the free-space classification after the documented tolerance;
3. published placement checks are consistent up to mesh/metadata numerical tolerance;
4. the generated base grid is nonempty and deterministic;
5. two independently batched calls return byte-identical candidate arrays.

Known pre-plan failure: official AcousticRooms commit `3c87318a...` has no `ListeningRoom_idx_2.obj`, although that room is in the full unseen split. Therefore:

- Round G1 may implement and validate the mesh path on the 16 available rooms, but no full/headline experiment may launch by silently dropping `ListeningRoom_idx_2`.
- Preferred resolution: obtain the missing authoritative OBJ from the dataset authors/user and record its hash/provenance.
- Proposed fallback, requiring explicit approval after its audit: derive a conservative free-space grid for `ListeningRoom_idx_2` from all receiver-centered global depth panoramas. A point is retained only when at least one receiver ray observes free space through it with `0.5 m` radial margin and it is at least `0.5 m` from the fused depth-surface point set. Validate against every known source/receiver anchor, report fallback candidate/oracle coverage separately, and stratify results by `mesh` versus `depth_fallback`. This preserves all 6,337 queries but is not claimed geometrically identical to the primary backend.

No implicit convex-hull, AABB-only, single-panorama, nearest-mesh substitution, or room deletion is allowed.

### 1.4 Frozen model and score

- Experiment-1 arm: `/home/zhixuanzhao/projects/Frame_Average/Checkpoint/P1_40k_clean_hybrid_EMA.ckpt`, SHA-256 `da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643`.
- Frozen AGREE: `/home/zhixuanzhao/projects/rir2rir/FLAC/weights/AGREE/AGREE_fullAR.pt`, SHA-256 `3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787`.
- Model config: existing `src/configs/model_configs/FLAC/AR/FLAC_AR_InContext.json`; dataset config: existing `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json`. Do not author a reduced eval config.
- Default sampler settings mirror the existing FLAC evaluation path: rectified-flow discrete Euler, `steps=1`, `cfg_scale=1.0`, checkpoint load integrity fail-closed, frozen/eval mode.
- Provisional pre-registered stochastic protocol, subject to plan review and Yixun approval: `K=4` independent samples per candidate with counter-based deterministic seeds; `tau=0.1`. A bounded latency probe estimates the full cost but may not silently change `K`, `tau`, grid spacing, or sampler settings.
- Encode the observed RIR once and each generated RIR through frozen `AGREE.encode_audio(..., normalize=True)`. Then

  `s[x,k] = cosine(E_a(h_obs), E_a(h_hat[x,k]))`

  `S[x] = tau * (logsumexp_k(s[x,k] / tau) - log(K))`.

  Use float32 accumulation and the stable `torch.logsumexp` form. Prediction is deterministic tie-breaking argmax by lexicographically sorted global candidate index.
- The heatmap softmax temperature `T` is visualization-only and cannot affect the predicted candidate. Default `T=0.1`, labeled uncalibrated.

## 2. Evaluation and controls

Headline metrics on the full split, aggregated first per room and bootstrapped by room:

- median and mean continuous Euclidean localization error (m);
- success within `0.5 m` and `1.0 m`;
- 95% room-bootstrap confidence intervals;
- grid-oracle median/mean/success coverage;
- latency per query, candidate, and generated RIR.

Controls under the identical candidate manifest:

- deterministic uniform-random candidate baseline, repeated with pre-registered seeds;
- AGREE oracle retrieval using real candidate-bank RIRs only where an exact dataset RIR exists, labeled sparse/metadata-bank and not confused with the dense-grid model oracle;
- score ablations (waveform / multiscale STFT) are deferred unless separately approved; they are not needed for the first Vanilla verdict.

Visualizations are selected by pre-registered quantiles after all queries finish: lowest-error sharp case, median-error/ambiguous case, and highest-error failure case, with boundary/valid region, receiver, continuous truth, prediction, and uncalibrated normalized score. Selection cannot be hand-picked.

Primary success criterion: Vanilla FLAC's median error and success rates beat the room-matched random baseline with non-overlapping or clearly shifted room-bootstrap intervals. Reliability gates (candidate coverage, full split, load integrity, leakage guards) must pass before interpreting localization quality.

## 3. Planned files and TDD rounds

Tests remain in `src/tests/` per announcement 02. Each round is test-first (red), minimal implementation (green), review/fix/reverification, and one small commit generally below 200 changed code lines.

### G1 — geometry audit primitives

Create `src/localization/geometry.py`:

- `snap_axis_to_lattice(bounds, spacing)` — exact global lattice endpoints;
- `build_lattice(aabb_min, aabb_max, spacing)` — stable lexicographic `float32/float64` candidate order;
- `load_raycast_scene(mesh_path)` — fail-closed OBJ load and identity metadata;
- `classify_mesh_candidates(scene, points, surface_clearance)` — occupancy plus distance mask in bounded chunks;
- `filter_query_candidates(points, receiver, context_sources, ...)` — receiver/context clearances;
- `grid_oracle_error(points, truth)`.

Create `src/tests/test_localization_geometry.py` first. Tests: negative/non-finite spacing rejected; negative AABB coordinates snap correctly; exact expected cubic lattice/order; synthetic watertight box retains only known interior points with clearance; outside/surface points rejected; receiver/context boundary behavior; no ground-truth insertion; deterministic chunking; empty/missing/malformed mesh fails closed.

Create `tools/audit_localization_geometry.py` after its helper contracts are green. It audits all full-split rooms and writes JSON/Markdown only under this experiment folder. The tool itself is included in the G1 code review.

### D1 — deterministic query/context construction

Create `src/localization/ar_queries.py`:

- parse existing split entries and metadata into immutable query records;
- load/pad/crop the observed RIR exactly like `AR_md.py`;
- deterministically choose eight same-receiver non-target contexts without replacement;
- materialize one shared FLAC metadata/context object and clone it with one candidate-relative pose;
- emit stable query/candidate/sample seeds independent of batching/worker count.

Create `src/tests/test_localization_ar_queries.py` first. Tests: filename/metadata mapping; full 17-room/6337-record parse invariant; target source and target RIR excluded; exactly eight unique contexts from the other nine; fixed seed reproducibility; different target may change selection; candidate cloning changes only `source`/`source_vit`; global-to-relative transform; audio length/rate checks; missing context fails closed rather than replacement sampling.

### S1 — scoring and metric aggregation

Create `src/localization/scoring.py`:

- normalized cosine score contract;
- stable log-mean-exp for arbitrary `tau>0`;
- lexicographic stable argmax;
- visualization-only softmax;
- localization/random/oracle metrics and room bootstrap.

Create `src/tests/test_localization_scoring.py` first. Tests: `K=1`; equal scores return the same score; small-`tau` approaches max; numerical stability at extreme logits; invalid tau/shape/NaN rejection; tie rule; visualization temperature cannot change prediction; exact synthetic errors/success rates; room bootstrap resamples rooms, not RIR rows.

### I1 — frozen FLAC/AGREE localization engine

Create `src/localization/engine.py` and `localize_FLAC.py`:

- reuse/refactor the already-reviewed checkpoint-integrity and Vanilla/FA conditioner dispatch from `eval_FLAC.py` without changing legacy evaluation behavior;
- freeze/load FLAC and AGREE once;
- encode observation once;
- batch candidates and `K` samples with batch-invariant seeds;
- generate/decode RIRs, encode AGREE audio, score, resume from atomic per-query artifacts, and write a schema-versioned manifest/result;
- expose `vanilla` now and `fa_invariant` for the later matched arm, with no arm-specific candidate logic.

Create `src/tests/test_localization_engine.py` and extend `src/tests/test_eval_paths.py` first. Tests: clean/dirty checkpoint load; all params frozen/eval; observation encoded once; candidate batch shape/metadata; exact K seeds and batch-size invariance; fake sampler recovers the known synthetic winning candidate; Vanilla and FA dispatch; interrupted resume skips only complete verified queries; config/split/checkpoint/candidate hash mismatch refuses resume; output stays inside the requested `NeuriPs_Workshop` directory.

### R1 — reporting

Create `tools/aggregate_localization_results.py` and `tools/render_localization_results.py`, with tests in `src/tests/test_localization_reporting.py`. Tests: complete 6,337-query coverage; no duplicate/missing query IDs; per-room then room-bootstrap aggregation; random baseline seed accounting; mesh/fallback strata; oracle coverage; quantile-based case selection; Markdown/HTML values agree with aggregate JSON. Both scripts enter review before use.

## 4. Validation ladder and launch gates

1. Static: `py_compile`, config JSON parse, `git diff --check`.
2. Permanent tests: new localization tests plus all existing `src/tests/`.
3. Tiny synthetic box/mesh forward with fake FLAC/AGREE.
4. Real-data readback: several records, then the complete metadata/split integrity scan; verify 22.05 kHz, lengths, coordinate transforms, context leakage guards.
5. Geometry audit: all room assets, anchors, candidate counts, oracle coverage; stop on missing/invalid room. Resolve `ListeningRoom_idx_2` explicitly.
6. One real query with a bounded candidate slice for shape/memory only; no scientific result.
7. One complete room smoke, no headline interpretation; verify resume, artifact size, and measured candidate/s throughput.
8. Full-cost projection and storage guardrail. If projected compute/storage is unacceptable, stop and ask; do not tune the protocol after seeing localization quality.
9. Independent full-diff review, parity audit against `eval_FLAC.py`/AGREE, params + exact command + acceptance criteria recorded at launch.
10. Full 6,337-query run and controls, followed by results, reliability analysis, offline HTML, and commits log.

## 5. Explicitly out of scope for exp_09

- training or fine-tuning FLAC/AGREE/localization heads;
- changing `0.5 m` grid spacing after reading quality results;
- inserting ground truth or snapping it into candidates;
- reducing the full split for headline claims;
- yaw augmentation, randomized headings, HAA, cylindrical/SSP models, or the FA-BF headline comparison (the engine is made compatible, but those are later matched arms);
- exact flow likelihood or calibrated posterior claims;
- silently substituting a different room mesh for `ListeningRoom_idx_2`.

## 6. Risks requiring review/approval

1. **Missing official mesh:** full-split mesh-only execution is currently impossible for `ListeningRoom_idx_2`. Preferred authoritative-mesh path versus the proposed multi-depth conservative fallback must be decided explicitly.
2. **Compute scale:** dense 3-D candidates × 6,337 queries × `K=4` may be expensive even at one flow step. The full-cost projection is a hard gate; no protocol reduction is automatic.
3. **Mesh topology:** architectural OBJs may be non-watertight or self-intersecting. Metadata-anchor validation is the empirical fail-closed criterion; topology warnings cannot be ignored merely because a grid is nonempty.
4. **AGREE input parity:** scoring must use exactly the audio normalization/length convention on which AGREE was trained. Real-RIR and generated-RIR preprocessing must be byte/number audited.
5. **Context proximity bias:** the proposed `1.0 m` context-source exclusion is scientifically motivated but was not explicit in the PDF; it is frozen here for review and user approval before any result is observed.
