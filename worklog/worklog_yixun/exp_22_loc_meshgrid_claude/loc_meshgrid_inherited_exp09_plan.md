# Plan — exp_09_localization_grid_preflight (3-D mesh-valid analysis-by-synthesis localization)
*(Inherited verbatim from Yixun's 2026-08-23 message; authored by OpenAI Codex as Planner in the zhixuanzhao/Frame_Average checkout; approved there by Yixun through R0→D1→G1 with the amendments recorded inline. Serves as exp_22's registered protocol source of truth.)*

**Author:** OpenAI Codex (Planner / Analyst) · **Coder:** strongest coding-tier subagent at max effort after approval · **Reviewer:** Anthropic Claude Opus 5 (`claude-opus-5`, Claude Code 2.1.237 interactive session) · **Date:** 2026-08-20
**Status:** APPROVED by Yixun on 2026-08-20 through R0→D1→G1, with protocol amendments for the original exp_01 global-RNG context draw and, after measured B7, 31-direction ray-parity validity plus a 0.20 m source-surface prior. Yixun subsequently authorized the cache-enabled no-quality throughput probe. On 2026-08-21 Yixun superseded the provisional single-K ladder and fixed three reported stochastic settings: `K∈{1,4,8}`. No localization quality score has been read; the full K=8 cost now requires a renewed launch decision.

## 0. Decision being tested
Can the frozen Vanilla FLAC 40k clean EMA checkpoint be inverted on the mesh-available AcousticRooms unseen rooms by evaluating a common, physically valid three-dimensional candidate grid and ranking generated RIRs with the frozen AGREE acoustic embedding?
This is inference-only. There is no learned localization head and no optimization loss. The quantity called "score" below follows Eq. (3) of `acoustic_localization_brief.pdf`; all other choices follow the attached experiment text plus Yixun's later overrides.
Protocol precedence is explicit: the PDF's §4 sentence about a common valid **2-D** grid at target source height is superseded by Yixun's approved isotropic **3-D** grid. The 2-D rule would leak the unknown target height. The manuscript protocol sentence and reserved Table 1 caption must be updated before exp_09 is reported. Exp_09 is also canonical-heading-only; it fills the diagnostic canonical column, not the brief's primary random-yaw column or random-yaw degradation result.

## 1. Frozen protocol
### 1.1 Data and leakage control
- Source manifest: existing `data/AR/unseen_eval.json` (6,337 queries / 17 rooms); do not author a reduced eval config.
- Exp_09 test scope, explicitly directed by Yixun on 2026-08-20: exclude exactly the 1,000 queries from `ListeningRoom_idx_2` because its official OBJ is absent. The fixed preflight denominator is **5,337 queries in 16 mesh-available rooms**. Filter this room in memory, record the exclusion in every manifest/report, and fail if any additional query is lost.
- This experiment-specific exception supersedes announcement 01 only for the missing-asset room in this mesh preflight. Results must say "mesh-available preflight subset"; they are not the complete published 6,337-query unseen-room protocol. Debugging may use further bounded records, but those numbers never enter `_results.md`.
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
- Stochastic protocol: report `K∈{1,4,8}` with `tau=0.1` fixed in every branch. Generate one nested, counter-seeded sequence through K=8 per query/candidate, then compute K=1 from sample 0, K=4 from samples 0–3, and K=8 from samples 0–7. Thus all three settings share samples exactly, remain batch-size invariant, and cost no more than one K=8 execution.
- AGREE input parity is pinned to the reviewed FLAC retrieval path: observed RIRs come from the existing dataloader crop, mono float32 `[B,1,10240]` at 22,050 Hz; generated RIRs are decoded to the same shape and clamped to `[-1,1]` exactly as `eval_FLAC.py`. Reuse `Retrieval.compute_audio_features` or extract one shared helper used by both metric and localization paths; do not create an independent preprocessing implementation. The observation is encoded once per query and generated RIRs use the same `encode_audio(..., normalize=True)` path. Then
  `s[x,k] = cosine(E_a(h_obs), E_a(h_hat[x,k]))`
  `S[x] = tau * (logsumexp_k(s[x,k] / tau) - log(K))`.
  Use float32 accumulation and the stable `torch.logsumexp` form. Prediction is deterministic tie-breaking argmax by lexicographically sorted global candidate index. Also save `S_mean[x]=mean_k s[x,k]` and its argmax as a zero-cost diagnostic; it never replaces the PDF-controlled headline score. A pre-registered debug slice records the empirical cosine spread relative to fixed `tau=0.1` but cannot tune tau.
- The heatmap softmax temperature `T` is visualization-only and cannot affect the predicted candidate. Default `T=0.1`, labeled uncalibrated.

### 1.5 Conditioning cache and pre-registered compute ladder
- Vanilla query cache contract: compute `context_poses_vit`, `context_poses`, and `context_audio` once per query; recompute [as] candidate changes. A bit-identity test must show the cached and uncached prepend-conditioning tokens are identical on the same inputs. For the later FA-BF cost probe, the released cylindrical transform makes `context_poses.dphi` candidate-dependent; therefore cache only `context_poses_vit` and `context_audio` per query and recompute the cheap `context_poses` branch inside each scored batch. This dependency split is mandatory and tested against the released full C4 path.
- Receiver-candidate cache: within a receiver group, `source_vit` and `source` depend on receiver/depth/candidate but not which target source was held out. Compute them once for the union of that receiver's candidates and reuse them across its target queries, with an uncached-vs-cached equality test and bounded in-memory lifetime per receiver.
- **Post-G1 gate before I1:** G1 reports exact valid candidate-query pairs for both full-height and admissible z-band branches, unique receiver-candidate pairs, estimated conditioner calls, and artifact bytes. Stop and present these geometry-only counts to Yixun; I1 cannot open until this second cost gate is accepted.
- **Yixun override (2026-08-21):** the earlier `4 -> 2 -> 1` runtime ladder is superseded. The experiment reports all three fixed nested prefixes `K∈{1,4,8}`; runtime cannot drop one of them automatically. Every model arm uses the same candidate manifest, sample seeds, and prefix definition.
- **Measured cost:** standalone two-arm projections are 38.31, 143.86, and 284.59 GPU-hours for K=1, K=4, and K=8. The requested nested three-setting execution costs the K=8 value: 140.05 hours Vanilla + 144.54 hours FA-BF = **284.59 GPU-hours / 11.86 serial days**. The slowest-measured-batch bound is 311.52 hours / 12.98 days. This exceeds the earlier 168-hour launch ceiling, so changing K is recorded but full generation remains unopened pending explicit compute approval.

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

## 3–7. TDD rounds (R0/D1/G1/S1/I1/R1), validation ladder, out-of-scope, risks, review ledger
*(Sections 3–7 inherited verbatim in substance: R0 baseline + Open3D pin; D1 release-parity global-RNG context materializer + tests; G1 geometry primitives + audit tool + tests; S1 scoring/aggregation + tests; I1 frozen engine + localize_FLAC.py + tests; R1 reporting tools + tests. Ladder: static → permanent tests → D1 census → G1 synthetic+real geometry → post-G1 Yixun cost gate → I1 synthetic → readback → K{1,4,8} throughput probe + post-probe cost decision → bounded smokes → full-diff review + parity audit → 5,337-query run + controls. Out of scope: training; spacing changes after reading quality; GT insertion; query reduction beyond ListeningRoom_idx_2; yaw/random-heading arms; likelihood claims; substitute geometry. Risks: subset claim boundary; compute scale; mesh topology; AGREE input parity; context parity. Review ledger B1–B6/N1–N9 as approved.)*
