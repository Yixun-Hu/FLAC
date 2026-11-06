# exp_37 loc_efficiency — plan r0 (2026-11-06, Planner: Claude Opus 5.5)

**Request:** Yixun 2026-11-06 14:35 + 14:40 (relayed by the desktop session; verbatim in `loc_efficiency_yixun_query.md`):
the accuracy–efficiency trade-off answering the OrbitRIR reviewer's "Computational overhead" — B1 matched latency
breakdown, B2 candidate density + coarse-to-fine, B3 inference orbit size, B4 P1@87.5k. Top priority on this box (with
exp_36); cards through the `claude_gpu_yield/` handshake; timings only on a card with nothing else on it; review chain:
this brief plan + ONE Codex plan review, then go. Research reports (read-only, Opus agents):
`research_latency_bundle.md`, `research_localization_64q.md` (this folder).

## 1. Facts
- **The latency protocol is findable** — `origin/NIPS-Experiment` (`dca796dd`), bundle `NeurIPS_2026_Workshop_Latency_Reproducibility/`
  (exp_29 "core_forward_latency_16room_128", another host: 2× RTX A6000, driver 595.84, Xeon 5220R, torch 2.7.0+cu126). The paper's
  OrbitRIR / FewShotRIR / FEM rows equal its `expected/summary_final.json` to the last digit. Timers live in
  `code/reference/src/localization/runner.py::execute_query` (`context_conditioning`; `candidate_conditioning_and_generation` per
  64-candidate batch; `core_forward_total`), driven by `localize_FLAC.py --synchronize-latency --warmup-query-count 1
  --measure-core-forward --candidate-batch-size 64 --sample-seed 42 --tau 0.1 --score-sample-counts 1`; the AGREE selector is timed
  separately (`benchmark_localization_selector_latency.py`: zeros `[b,1,10240]`, 3 repeats per query, median); per query
  L_q = core_forward_total_q + selector_q; s/query mean / median / P90 over the 128 frozen queries
  (`data/selection/frozen_16room_128.json`, 16 rooms × 8, 92,608 candidates); ms/candidate dataset-wide = 1000·ΣL/Σ|C|, query
  median / P90 of 1000·L_q/|C_q|.
- **Matched FLAC / Yaw-Aug rows already exist** in that bundle (same run as OrbitRIR's 15.091): Vanilla FLAC 6.456 / 2.104 / 31.742
  s/query and 8.923 / 9.556 / 10.410 ms/candidate; Yaw-Aug 6.389 / 2.073 / 32.398 and 8.830 / 9.177 / 10.640. The paper's
  commented-out rows (57.53 / 80.59 ms/candidate) come from `exp_19_five_method_latency_97` (14 rooms / 97 queries, a joint
  K_gen {1,4,8} pass, no warm-up) — not the same protocol, as Yixun suspected.
- **Backend:** the bundle's interpreter (a `.venv` on the other host) printed "flash_attn not installed" in its exp_09 pilot logs,
  so its DiT most likely ran SDPA fp32 (UNVERIFIED for the latency runs themselves). This box's `flac` env has flash-attn
  2.7.4.post1 and the DiT uses it whenever importable → every exp_37 run uses a no-flash launcher (the reviewed exp_35 overlay
  pattern) and records the backend actually selected (`src.models.transformer.flash_attn_func is None`).
- **This box:** same CPU model, 2× RTX A6000 (different cards), driver 590.48.01. YAW checkpoint identical (`ac1f2603…`); P1 / B-F
  here are the wrapped 40k checkpoints (`c4c67882…` / `5319feb4…`) whose EMA weights `load_flac_module` swaps in — the bundle used
  EMA extracts (`da127485…` / `0f61277f…`, absent here); same weights path, different file digests (disclosed). The geometry audit
  binds absolute mesh paths of the other host; all 16 meshes exist here under `AcousticRooms/room_mesh_obj_format/` with the
  audit's sha256 (16/16). All 1,280 data files of the 128 queries exist. K_ctx is hard-coded to 8 (runner / summarize).
- **Expected wall time** (exclusive A6000, bundle-derived): P1 ≈ 15 min, B-F ≈ 25 min, YAW ≈ 15 min per 128-query repeat; selector
  ≈ 6 min; 3 arms × K_ctx {8, 1} + selector ≈ 2 h.

## 2. B1 — matched latency breakdown (exclusive card)
- **Worktree** `~/codespace/exp-37-loc-efficiency`, branch `exp-37-loc-efficiency` cut from `origin/NIPS-Experiment` `dca796dd` (the
  code that produced the bundle and the paper's localization; this checkout's `src/localization` is a different stack).
- **Code round (Opus Coder, TDD):** (1) a no-flash launcher for `localize_FLAC.py` (port of the reviewed exp_35 overlay; backend
  recorded in every run manifest; refuse to start if it resolves to flash); (2) a local run config re-pointing python, the dataset
  root, AGREE (`weights/AGREE/AGREE_fullAR.pt`, sha `3a13243d…` = the bundle's) and the checkpoints; an in-memory mesh-path remap
  (other-host prefix → this box's `AcousticRooms/room_mesh_obj_format/`), each mesh's sha256 verified against the audit before use,
  the audit file itself untouched; (3) `--context-count {8,1}`: K_ctx = 1 feeds contexts[0] of the exp_09 manifest to the model while
  the candidate filter keeps the 8-context rule; `latency_protocol.context_count` records the value; (4) a stage split with
  `torch.cuda.Event` marks inside the existing timed region (after the batch-start sync; before `generate_rir_batch`; after
  `deterministic_noise`; after `sample_discrete_euler`; after the VAE decode) read after the existing end-of-batch sync — no extra
  host syncs, so the headline totals keep the bundle's sync placement; per-stage peak memory (`reset_peak_memory_stats` /
  `max_memory_allocated` around stages; `max_memory_reserved`; the weight footprint after loading); a per-query check that the
  stages sum to the timed total within 2 %; (5) a 3-arm summarize variant (formulas unchanged, K_ctx ∈ {8, 1}).
- **Stages reported:** context encoding (the existing `context_conditioning`; FA = 32 batch-1 ViT passes, vanilla 8) / candidate
  conditioning incl. the FA rotated ViT passes / single-step DiT (noise + one Euler step) / VAE decode / AGREE selector.
- **Runs** (1 repeat each, warm-up 1, seed 42, batch 64; nothing else on the card; CPU co-tenants recorded): the selector once, then
  P1, YAW, B-F at K_ctx 8, then at K_ctx 1. Cross-check: OrbitRIR K_ctx 8 here vs the bundle's 10.919 s/query / 15.091
  ms/candidate (host difference reported, never hidden).
- **Output:** per-query JSONs (bundle schema + stage fields + backend), `b1_latency_summary.{json,md}` (3 arms × 2 K: s/query
  mean / median / P90, ms/candidate dataset-wide / median / P90, peak memory, stage split) — the drop-in replacement for the
  paper's commented-out rows (plus the bundle's matched rows as the immediate answer).
- **The 64-query localization code** — `origin/NIPS-Experiment`: `localize_FLAC.py` → `src/localization/runner.py::run_localization`
  (pilot code `cf6de194`, results `0834969d` in `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_results/`:
  per query `candidates [N,3]` + `similarities [N,8]`, nested K_gen 8, the paper reads K=1). Queries = `pilot_manifest_seed42_4_per_room.json`
  (4 per room × 16 mesh rooms, rng 42) = EXACTLY the seed-42 half of the latency bundle's 128. Grid = `audit["grid_spacing_m"]` (0.5,
  hash-locked per room / per query), lattice anchored at the origin, free-space by ray voting, per-query filter (≥ 0.5 m from the
  receiver, ≥ 0.25 m from each context source, z band). Score = τ·(logsumexp(s/τ) − log K), τ 0.1 (K = 1: the AGREE cosine). Metrics:
  e_loc, oracle (δ_grid), excess, SR@0.5 / SR@1.0, resolution-aware SR@0.5 (excess ≤ 0.5), pooled over the 64. The committed per-query
  files reproduce the paper's K=1 medians (Vanilla 1.0093 m, OrbitRIR 0.9280 m). No spacing / frame-angle / rotation flag exists.
- **Three findings Yixun must see** (before any number is published): (i) the paper's FewShotRIR localization row (1.595 / 1.6 / 26.6 /
  15.6) appears in no committed aggregate — exp_10 / exp_22 / exp_23 all give 1.517 / 3.1 / 25.0 / 14.1; (ii) the paper's FLAC and
  Yaw-Aug medians are fragile: the bundle's K_gen=1-only rerun (batch 64×1 instead of 64×8, same checkpoints and seed) reproduces 64/64
  OrbitRIR predictions but 61/64 Vanilla (median 1.108 m vs 1.009) and 60/64 Yaw-Aug (0.916 vs 0.892); (iii) B4 cannot run here — no
  P1 step-87,500 checkpoint exists on this box (highest NAS step 67,500; its recorded digest `bd3fc7db…` points to neuronic).
- **The 1.0 m grid is an exact subset of the 0.5 m grid** (integer coordinates, same seeds keyed by the 0.5 m index): its accuracy and a
  coarse-to-fine search are exact offline readouts of a 0.5 m run's per-candidate scores. 0.75 m is not nested (new audit + GPU runs).

## 3. Common to every exp_37 run
- Worktree `~/codespace/exp-37-loc-efficiency` (branch from `dca796dd`), run from its own tree; no-flash launcher (backend recorded,
  refuse if flash); the in-memory mesh-root remap with per-mesh sha256 checks; `--candidate-batch-size 64`, `--tau 0.1`,
  `--score-sample-counts 1` (K_gen 1 only); generation seeds `--sample-seed` 42 / 43 / 44 (the query set stays the seed-42 pilot).
- **Gate G0 (hard stop, ~5 min):** the 27-candidate Bathrooms_idx_14 query, K_gen 1, seed 42, for P1, YAW and B-F against the bundle's
  K=1-only per-query npz: argmax equal and max |Δ similarity| ≤ 1e-3 (exact equality reported if it holds). It validates the wrapped
  P1 / B-F checkpoints, the remap and the backend in one step.
- **Timing validity:** every timed pass runs alone on its card (GPU 0; exp_36 on GPU 1 — CPU co-tenancy disclosed), with the bundle's
  flags (`--synchronize-latency --warmup-query-count 1 --measure-core-forward`); because B runs one process at a time on GPU 0, every
  B accuracy run is also a valid timing run. Never publish a timing from a card that held another compute process.

## 4. B2 — candidate density (OrbitRIR C4 and FLAC P1; seeds 42–44)
- 0.5 m: 6 GPU runs (2 arms × 3 seeds) on the 64 queries. 1.0 m: offline readout of each 0.5 m run (integer-coordinate subset; the
  resolution-aware oracle recomputed over that subset). Coarse-to-fine (pre-registered): the 1.0 m pass → top-**k = 5** → add every 0.5 m
  candidate of the query within Chebyshev distance 0.5 m of each top-k point (the ≤ 26 lattice neighbours) → argmax over coarse ∪
  refined; accuracy offline from the 0.5 m runs; k ∈ {1, 3, 10} reported as a labelled sensitivity only. 0.75 m: the audit tool gets
  a spacing parameter (the z-band branch pinned to the 0.5 m audit's choice per room), a 0.75 m audit + pilot manifest regenerated
  (must select the same 64 queries — refuse otherwise), 6 GPU runs. Time axis: the 0.5 / 0.75 m runs' own timings (seed 42 headline,
  43 / 44 as repeats); 1.0 m and coarse-to-fine get dedicated timing passes (seed 42, 2 arms) through a candidate-subset mode.
- Pareto figure (2 panels: s/query vs median error; s/query vs SR@1m): OrbitRIR and FLAC at 0.5 / 0.75 / 1.0 / c2f (accuracy mean ± SD
  over seeds), plus FewShotRIR and FEM (accuracy from the committed exp_23 aggregate — the paper row's discrepancy flagged; time from the
  bundle's per-query latencies restricted to the 64-query half, measured on the other host — labelled).

## 5. B3 — inference orbit size (OrbitRIR only)
- A `--frame-avg-angles` flag ({0}, {0,180}, C4; recorded in the run identity) and a `--rotate-deg` option (`rotate_scene_metadata` on the
  query metadata and every candidate's metadata; candidates and metrics stay in the global frame, as eval_FLAC `--rotate-deg`).
- Localization: 3 orbit sets × θ {0, 90} × seeds 42–44 (C4 θ0 = B2's 0.5 m runs; 15 new runs). Synthesis: 8 new eval_FLAC cells
  ({0}, {0,180} × θ {0, 90} × K {1, 8}, seed 42, no-flash, through the exp_35 / exp_36 runner path on a leaf branch of exp_36's code);
  C4 at θ0 / θ90 reused from exp_35 phase (i) (its BF-pin T θ0 and C4 θ90 cells at seed 42, K 1 / 8, run no-flash at H35). Latency: B1-style passes on the 128
  latency queries, K_ctx 8, for {0} and {0,180}.

## 6. B4 — not runnable here (no P1@87.5k checkpoint); reported to Yixun with the path where it probably lives.

## 7. Schedule (forecasts)
After PAUSED: GPU 1 = exp_36 (≈ 5 h); GPU 0 = exp_37, one process at a time: G0 → B1 (selector + 3 arms × K_ctx 8 → 1, ≈ 2 h) → B2
0.5 m (≈ 1.2 h) → B3 localization (≈ 2.5 h) → B2 0.75 m + the 1.0 m / c2f timing passes (≈ 0.7 h) → B3 latency (≈ 0.6 h); B3's 8
synthesis cells on GPU 1 once exp_36 is done (≈ 2 h). Code first: Coder round B-a (worktree, launcher, remap, configs, G0) → the first
B GPU job; then B-b (stage split, K_ctx 1, 3-arm summarize), B-c (subset / c2f, 0.75 m audit, frame-angle and rotation flags, readout,
figure) while the GPU runs. RELEASE when exp_36 and exp_37 no longer need the cards.

## 8. Planner defaults (Yixun may override; each reversible by a rerun)
D1 backend no-flash (SDPA) for every run, as the bundle / pilot; D2 K_ctx 1 = contexts[0] of the exp_09 manifest, the 8-context candidate
filter kept; D3 the wrapped P1 / B-F 40k checkpoints (EMA swapped in), digests disclosed, G0 guards; D4 the mesh-root remap; D5 stage split
by CUDA events in the headline pass (no extra syncs), per-query Σstages vs total within 2 %; D6 one repeat (as the bundle); D7 the AGREE
selector re-measured here; D8 c2f k = 5, the 26-neighbourhood; D9 1.0 m / c2f accuracy offline-exact; D10 the 0.75 m audit as above;
D11 B3 seeds 42–44 for localization, seed 42 for synthesis.

## 9. Hard stops
G0 failure; a run resolving to flash; a mesh sha mismatch; the 0.75 m pilot selecting other queries; any timed pass with another compute
process on its card (discard + rerun, never publish). Each stops the affected arm and is reported.

## 10. r1 — the Codex plan review's required changes (binding; applied 2026-11-06 per the review: "apply the changes, then proceed without another plan review")
- **B1 timing passes:** the headline `core_forward_total` comes from an UNINSTRUMENTED pass (byte-near the bundle's timed region, no CUDA
  events, no allocator resets inside the timer); the stage split runs as a SEPARATE instrumented pass whose per-query and aggregate core totals
  are compared with the headline (≤ 2 %; outside 2 % only the split is withheld, never the headline). Stage boundaries: candidate conditioning =
  batch start → after `merge_cached_conditioning()` + `get_conditioning_inputs()`; DiT = `deterministic_noise` + the one Euler / model step;
  VAE decode = after `sample_discrete_euler` → decoded batch. Memory: a running whole-query maximum kept (bundle-compatible); stage peaks
  reported as absolute allocated peaks (not additive); the selector's peak measured separately; the combined inference peak = a maximum, never a
  sum; `max_memory_allocated`, `max_memory_reserved` and the loaded-weight footprint reported as different quantities.
- **B2 time axis:** s/query = query-micro mean over the 64 matched queries of `core_forward_total + shape-matched AGREE selector`, the same
  definition for every point; never `elapsed_seconds`. The AGREE-only benchmark is generalized to arbitrary query / candidate-count manifests
  (warm-up, batch 64, 3 repeats, median-of-three kept); selector vectors re-measured for the 0.75 m and the nested 1.0 m counts (0.5 m and B3 share
  one vector). Coarse-to-fine timing measures the actual sequential path (observed encoding once; coarse generation + encoding + similarity +
  top-k; neighbour construction; refined generation + encoding + similarity; union argmax). Dedicated 1.0 m / c2f timing keeps the full 0.5 m
  candidate indices for the noise seeds.
- **B2 mathematics:** "1.0 m" = the nested integer-coordinate subset with full-grid common random numbers (not a native reindexed rerun);
  top-k ordering by descending score then the original full-grid index; union ordering and ties inherited from the 0.5 m lattice; a fresh oracle
  over each actual candidate set (0.5, 0.75, nested 1.0, and each model / seed's c2f union — never the full 0.5 m oracle for c2f). 0.75 m: the
  audit's single GLOBAL height branch pinned to `z_band`, all 5,337 grids verified non-empty, the exact same 64 query IDs required; shared
  coordinates get different generation noise (grid-index seeds) — disclosed; seeds 42–44 mitigate it.
- **B3 rotation contract:** an immutable `base_metadata`; `rotate_scene_metadata(base_metadata, θ, 512)` applied once for the query / context
  cache; every candidate item built from `base_metadata` and then rotated once; candidate arrays, filtering, truth and metrics in the global frame;
  gates: θ = 0 byte-identical to the existing path, θ = 90 exactly one 128-column roll / pose rotation for query and candidate metadata.
- **CPU co-tenancy:** publishable timing passes run only after exp_36 has left the host (Yixun's 15:30 order — the off-orbit run, then exp_36,
  then exp_37 — makes this the schedule anyway); CPU affinity, torch intra / inter-op threads, OMP / MKL variables and every co-tenant recorded.
- **G0 pinned:** query index 4467 (`single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav`, candidate hash `4f212dcb…86bf1`,
  bundle `query_04467.npz`, K=1 `similarities [27,1]` present for P1, YAW and B-F); candidate arrays compared exactly; argmax equal; max
  |Δ similarity| ≤ 1e-3 fixed (never relaxed post hoc); YAW (byte-identical checkpoint) runs first as the environment / backend control — YAW fail
  ⇒ environment / backend parity failed; YAW pass but P1 / B-F fail ⇒ wrapped-vs-extracted EMA parity suspect.
- **Pareto references:** FewShotRIR (same 64 queries, exp_23 accuracy, bundle-host latency) and FEM (exp_23 deterministic-random fallback
  accuracy, policy-aware mixed-host latency, 7 failures on this subset) drawn as labelled external markers, excluded from any frontier.
- **B3 synthesis:** the new {0} / {0,180} cells use the same evaluator closure, checkpoint, dataset, batch, conditioner autocast, no-flash backend
  and aggregation as the reused exp_35 cells (H35 evaluator; on a leaf branch, the closure proved unchanged); synthesis is single-seed evidence.
- **Corrected claims:** Vanilla fragility = 61/64 identical prediction indices, median 1.10822 vs 1.00927; Yaw-Aug = 60/64 per-query localization
  errors match, median 0.91615 vs 0.89172 (its per-query indices are not in git); B4 = "no P1 step-87,500 checkpoint found in the searched known
  local / NAS roots; skipped per Yixun's conditional instruction". The paper's `app:latency` sentence saying timing begins before loading the
  observed / context RIRs and depth contradicts the bundle (input loading excluded) — flagged for correction; exp_37 reproduces the executable
  bundle protocol.
- **Provenance:** the resolved mesh-root override, the 16-mesh digest map, overlay + launcher digests, the selected attention backend, the exact
  K_ctx = 1 context index / path, frame angles and rotation recorded per run; the audit itself byte-identical.
- **NITs:** no-flash = best-supported match, inferred from the exp_09 logs; one repeat = the bundle's protocol, not a variance estimate; k ∈ {1,3,10}
  strictly post-primary sensitivity.
- **Coder split:** B-a (worktree / config, fail-closed no-flash launcher, mesh remap + provenance, exact G0, the uninstrumented headline mode);
  B-b (K_ctx 1, arbitrary-manifest selector, 3-arm summarizer, the split pass + comparison); B-c (nested subset / c2f readout + timing, 0.75 m
  audit, frame-angle / rotation, aggregation, figure); B-synthesis on a separate H35-compatible leaf.
- **Schedule (Yixun 15:30):** exp_37's GPU work starts after the seed-42 off-orbit run and exp_36; its Coder rounds proceed now on CPU.
