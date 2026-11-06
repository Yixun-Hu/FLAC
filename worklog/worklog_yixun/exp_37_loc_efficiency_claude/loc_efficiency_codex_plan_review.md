**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** exp_37 plan review (plan_loc_efficiency.md r0)

**Verdict:** REVISE

Apply the changes below to the worklog, then proceed without another plan review.

## BLOCKING

1. **The instrumented stage pass cannot also be the B1 headline pass.**

   The bundle’s headline timer contains no CUDA-event recording or per-stage allocator resets. Those operations occur inside its `perf_counter` region and can bias the total; checking `Σstages ≈ total` does not detect that bias because both values come from the same instrumented pass.

   Required change:

   - Preserve an optional, byte-near-original **uninstrumented** mode for headline `core_forward_total`.
   - Run the stage split separately and compare its per-query and aggregate core totals with the headline pass; the proposed 2% threshold is acceptable. Outside 2% stops publication of the split, not the valid uninstrumented headline.
   - Put the candidate-conditioning boundary **after** `merge_cached_conditioning()` and `get_conditioning_inputs()`. The planned “before `generate_rir_batch` → after `deterministic_noise`” interval incorrectly charges those conditioning operations to the DiT stage.
   - DiT must be exactly deterministic noise plus the one Euler/model step; decode begins after `sample_discrete_euler`.
   - Preserve a running whole-query memory maximum. Per-stage `reset_peak_memory_stats()` otherwise destroys the bundle-compatible query peak. Report stage peaks as absolute allocated peaks, not additive memory.
   - Measure selector peak separately and report the combined inference peak as a maximum, never a sum.

2. **B2’s time axis is not yet defined faithfully enough to run.**

   “Use the accuracy runs’ own timings” is ambiguous and could select `elapsed_seconds`, which includes a different boundary. The bundle quantity is:

   `core_forward_total + shape-matched AGREE selector`.

   Required change:

   - Never use `elapsed_seconds` for the Pareto axis.
   - Adapt the AGREE-only benchmark to arbitrary query/candidate-count manifests, retaining its warm-up, batch 64, three repeats per query, and median-of-three fields.
   - Re-measure selector vectors for 0.75 m and 1.0 m counts. The 0.5 m/B3 arms may share one selector vector because their candidate arrays are identical.
   - For coarse-to-fine, measure the actual sequential path: observed encoding once; coarse generated encoding/similarity/top-k; neighbour construction; refined generated encoding/similarity; union argmax. Do not add two complete selector totals, which would double-count observed encoding, and do not substitute one selector over the final union, which omits the top-k dependency.
   - State whether the plotted `s/query` is the query-micro mean over the 64 matched queries; use the same definition for every point.
   - Dedicated 1.0 m/c2f timing must retain the full 0.5-grid candidate indices for noise seeds, matching the offline accuracy readout.

3. **The B3 rotation wording admits a double rotation.**

   If query metadata is rotated and `candidate_metadata()` is then derived from that rotated object and rotated again, depth and context poses receive two rotations.

   Required implementation contract:

   - Keep immutable `base_metadata`.
   - Use `rotate_scene_metadata(base_metadata, θ, 512)` once for the query/context cache.
   - Build every candidate item from `base_metadata`, then rotate that completed candidate item once.
   - Candidate arrays, filtering, truth and metrics remain in the original global frame.
   - Add gates that θ=0 is byte-identical to the existing path and θ=90 gives exactly one 128-column roll/pose rotation for query and candidate metadata.

4. **GPU-card exclusivity is satisfied literally, but CPU co-tenancy is not controlled enough for publishable timing.**

   The timed candidate section includes CPU metadata construction, seed hashing, and OrbitRIR’s CPU depth rolls/einsums. Merely disclosing exp_36 on GPU 1 does not prevent arm-dependent CPU interference.

   Required change: either run all publishable timing passes after exp_36 leaves the host, or pre-assign disjoint CPU affinities/thread pools to both campaigns and validate at least the B-F C4/K8 reference again after exp_36 ends. Record CPU affinity, `torch` intra/inter-op counts, OMP/MKL variables and co-tenants. Accuracy runs may proceed concurrently.

   The one-query warm-up is faithful. Open3D mesh reconstruction is outside the bundle’s core timer and is not itself a problem.

## HIGH

1. **Pin G0 to an exact query.**

   “The 27-candidate Bathrooms_idx_14 query” is ambiguous: the seed-42 manifest has three such queries. Pin:

   - index `4467`
   - `single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav`
   - candidate hash `4f212dcb…86bf1`
   - bundle file `query_04467.npz`

   The K=1 NPZ is present in git for P1, Yaw-Aug and B-F, with `candidates [27,3]` and `similarities [27,1]`.

   The `1e-3` maximum-similarity-delta gate is a reasonable conservative threshold, but it is not an established cross-host equivalence bound. Keep it fixed and do not relax it post hoc. Compare candidate arrays exactly and run byte-identical YAW first as the environment/backend control:

   - YAW failure ⇒ environment/backend parity failed.
   - YAW pass but P1/B-F failure ⇒ wrapped-versus-extracted EMA parity is suspect.

   Argmax equality remains mandatory. For query 4467, YAW’s bundle top-two margin is about `0.00456`, so the proposed threshold is meaningfully below that margin.

2. **Tighten the B2 mathematical contract.**

   The 1.0 m offline readout and `k=5`, Chebyshev-radius-0.5 m refinement are correct, provided they are described as the **nested 1.0 m subset with full-grid common random numbers**, not a native reindexed 1.0 m rerun.

   Also require:

   - deterministic top-k ordering: descending score, then original full-grid index;
   - union ordering/ties inherited from the full 0.5 m lattice;
   - a fresh oracle over each actual candidate set: 0.5, 0.75, nested 1.0, and the model/seed-dependent c2f union;
   - the c2f resolution-aware metric must not reuse the full 0.5 m oracle.

   For 0.75 m, pin the audit’s single global branch to `z_band`—the current plan incorrectly says “per room”—verify all 5,337 grids remain nonempty, then require the same exact 64 query IDs. Record that shared coordinates receive different generation noise because the existing seed is grid-index-based; seeds 42–44 mitigate this but do not remove it.

3. **Treat FewShotRIR and FEM as contextual reference markers, not members of the new same-host Pareto frontier.**

   The FewShotRIR accuracy and latency checkpoint/config digests match, and restricting bundle per-query latency to the same 64 queries is legitimate. FEM is less clean: its bundle runtime is policy-aware and heterogeneous, including six externally recovered queries, and is explicitly not hardware-normalized.

   Label:

   - FewShotRIR: same 64 queries, exp_23 accuracy, bundle-host latency.
   - FEM: exp_23 deterministic-random fallback accuracy, policy-aware mixed-host latency, seven failures on this 64-query subset.
   - Draw both as external/reference markers and exclude them from any computed frontier envelope.

4. **Bind new B3 synthesis cells to the reused exp_35 cells more tightly.**

   Reusing BF-pin T θ=0 and C4 θ=90, seed 42, K={1,8} is valid. New `{0}` and `{0,180}` cells must use the same evaluator closure, checkpoint, dataset, batch size, conditioner autocast, no-flash backend and metric aggregation. Prefer the H35 evaluator with only cell admission/provenance changes. If an exp_36 leaf is used, prove the evaluator closure relevant to these cells is unchanged.

   State explicitly that synthesis is single-seed evidence; it has no seed uncertainty. Localization seeds 42–44 are sufficient and correctly paired across θ because the candidate arrays and generation seed keys stay fixed.

5. **Correct two provenance claims before reporting them to Yixun.**

   - Vanilla fragility is fully auditable: K=1-only reproduces `61/64` prediction indices, with median `1.10822` versus `1.00927`.
   - Yaw-Aug fragility is real, with median `0.91615` versus `0.89172` and `60/64` matching per-query errors. But the original Yaw per-query NPZ/indices are absent from git, so “60/64 predictions” is not verifiable. Say “60/64 per-query localization errors match.”

   B4 should read “checkpoint not found in the searched known local/NAS roots; skipped per Yixun’s conditional instruction,” not an unconditional proof of nonexistence.

6. **The paper prose and actual latency boundary disagree.**

   The bundle/code exclude ordinary query input loading from the headline, while Appendix `app:latency` contains a sentence saying timing begins before loading the observed/context RIRs and depth panorama. The equation and bundle agree with the plan’s core-plus-selector boundary. Describe exp_37 as a reproduction of the executable bundle protocol and flag that manuscript sentence for correction before publication.

7. **Expand run provenance slightly.**

   Record the resolved mesh-root override, verified 16-mesh digest map, overlay and launcher digests, selected attention backend, exact context index/path used for K_ctx=1, frame angles and rotation. The audit itself should remain byte-identical.

## NIT

- D1, D2, D3, D4, D6 and D7 are otherwise sound. No-flash is the best-supported match, but remains inferred from the exp_09 logs rather than proven for the original latency run.
- One repeat matches the released bundle; do not imply that it estimates timing variance.
- Keep `k={1,3,10}` strictly labelled as post-primary sensitivity. It must not alter the registered `k=5` result.
- Report `max_memory_allocated`, `max_memory_reserved`, and loaded-weight footprint as different quantities.

## Minimal Coder split

- **B-a first:** dca796dd worktree/configuration, fail-closed no-flash launcher, mesh remap and provenance, exact query-4467 G0, plus an unchanged/uninstrumented headline timer mode. Once G0 passes, start a K_ctx=8 arm if the timing-host gate is satisfied; otherwise start a 0.5 m accuracy run.
- **B-b:** K_ctx=1, arbitrary-manifest AGREE selector, three-arm summarizer, optional event/memory split pass and headline-vs-split comparison.
- **B-c:** nested subset/c2f readout and timing, parameterized 0.75 m audit, frame-angle/rotation changes, aggregation and figure.
- **B-synthesis:** separate H35-compatible leaf; it need not block B-a/B-b.

## Verified statically

- `54bc7d59` is the requested `localization-exp` plan commit; `dca796dd` is `origin/NIPS-Experiment`.
- Bundle manifests confirm 128 queries, K_ctx=8, K_gen=1, batch 64, seed 42, warm-up 1, synchronized core timing, one Euler step and separate median-of-three AGREE timing.
- Bundle statistics and the paper table agree exactly, including OrbitRIR `10.9185 s/query` and `15.0913 ms/candidate`; matched P1 and Yaw rows are already present.
- The 64-query seed-42 pilot is exactly one half of the frozen 128-query latency selection.
- The 1.0 m integer-coordinate grid is a pointwise subset of the globally anchored 0.5 m grid; 0.75 m is not nested.
- All 16 local mesh files exist and match the committed audit SHA-256 values.
- The paper’s FewShotRIR row `1.595 / 1.6 / 26.6 / 15.6` is absent from committed exp_10/22/23 aggregates; those consistently give `1.517 / 3.1 / 25.0 / 14.1`.
- The exp_35 θ=0 and θ=90 C4 synthesis cells required for reuse are committed and were no-flash.
- No 87.5k checkpoint was found in the known local `weights` and NAS checkpoint roots; the highest parsed NAS step is 67,500. Git records the expected 87.5k digest/path only.
