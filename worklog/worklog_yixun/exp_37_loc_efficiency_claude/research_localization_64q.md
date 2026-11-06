> Read-only research report by an Opus 5.5 subagent (2026-11-06), committed verbatim as the plan's evidence; model output, checked by the Planner where the plan relies on it.

# 64-query localization: provenance and reuse report (read-only)

Refs are `path:line` on `origin/NIPS-Experiment` (head dca796dd). E9 = `worklog/worklog_yixun/exp_09_localization_grid_preflight_claude`; LB = `NeurIPS_2026_Workshop_Latency_Reproducibility`. No GPU was used; every hash and file check below was recomputed on this box.

## 1. Entry point, flags, checkpoints, results
- Entry point: `localize_FLAC.py` → `src/localization/runner.py::run_localization`. Pilot code is cf6de194 (Aug 21); results were committed in 0834969d.
  - The numerical path is unchanged at head (317b12e1 only split generate/score and added latency flags). Head adds 3 run-identity keys (runner.py:467-469), so reruns need fresh output dirs.
  - CLI is localize_FLAC.py:21-38. It has no grid, frame-angle, rotation or autocast flag.
- Exact commands for Vanilla and OrbitRIR: `E9/run_pilot_commands.sh:7-35`.
  - Common args (:11-20): `--model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json` (sha f3eafef4…, identical on NE, cf6de194 and localization-exp), `--agree-ckpt …/AGREE_fullAR.pt`, `--context-manifest E9/context_manifest_exp01_seed42.json`, `--geometry-audit E9/geometry_audit.json`, `--pilot-manifest E9/pilot_manifest_seed42_4_per_room.json`, `--dataset-root …/AcousticRooms --device cuda:0 --candidate-batch-size 64 --sample-seed 42 --tau 0.1`. Score counts are the default `1 4 8` (nested); the paper reads K_gen=1.
  - Vanilla (:25-29): `--ckpt-path /home/zhixuanzhao/projects/Frame_Average/Checkpoint/P1_40k_clean_hybrid_EMA.ckpt --cond-method vanilla`. OrbitRIR (:31-35): `…/BF_40k_clean_hybrid_EMA.ckpt --cond-method fa_invariant`. Python is zhixuanzhao's `FLAC-vanilla/.venv` (:8).
- Yaw-Aug 64-query command: NOT committed (UNVERIFIED). Its identity (`E9/pilot_results/yawaug_ar_40k_vanilla/run_manifest.json`) shows the same flags, vanilla conditioning and ckpt `…/Checkpoint/YAWAUG_AR_40k_epoch8-step40000.ckpt` (path from its summary.json). The batch-2 twin script is `E9/run_yawaug_batch2_16x8x3.sh:18-31`.
- Checkpoint digests (`E9/pilot_results/<arm>/run_manifest.json:5`, AGREE at :3) compared with this box:
  - Vanilla P1 `da12748586912c5f…` and OrbitRIR BF `0f61277f45367fb0…` are EMA extracts and are absent here. The local `weights/exp20/P1_40k.ckpt` (`c4c678826cdd…`) and `BF_40k.ckpt` (`5319feb4af87…`) are wrapped PL ckpts, so the bytes differ.
  - Yaw-Aug `ac1f26034e4f…` = `weights/exp20/YAW_40k.ckpt`, and AGREE `3a13243d6c6a…` = `weights/AGREE/AGREE_fullAR.pt`: both identical.
  - No `*clean_hybrid*` file exists on this box. Whether the extracts hold the same EMA weights as the wrapped ckpts is UNVERIFIED; localization-exp `src/localization/meshgrid_report.py:198-203` records it "by authority".
  - `load_flac_module` (engine.py:239-267) remaps `diffusion_ema.ema_model.*` and whitelists leftover keys (eval_FLAC.py:106), so the wrapped ckpts should load (not run). Cheap parity check: rerun one 27-candidate Bathrooms_idx_14 query (about 2 s) and diff it against the committed npz.
- Conditioning settings:
  - OrbitRIR uses C4, `frame_average_angles [0,90,180,270]` (fa_bf run_manifest.json:8-13), hard-coded via `DEFAULT_FRAME_ANGLES` (runner.py:14, 105-113, 459-461; yaw_rotation.py:23).
  - Conditioner caches run under `torch.amp.autocast("cuda")`, i.e. fp16 = eval_FLAC `--cond-autocast default` (runner.py:283, 312). TF32 on, matmul "medium" (:475-477). Sampler: 1-step rectified-flow Euler, cfg 1.0 (engine.py:420-462).
  - The pilot ran WITHOUT flash-attn (`E9/pilot_results/pilot_run.log:10-11, 95-96`).
- Seeds are all 42, single seed: generation noise per (query, candidate index, sample) via blake2b (engine.py:177-201; runner.py:333-337); query selection (pilot.py:66); context RNG (manifest `protocol`); random baseline (scoring.py:70-74).
- Raw results in git:
  - `E9/pilot_results/{vanilla,fa_bf}/queries/query_*.{json,npz}`: 64 per arm; each npz holds `candidates [N,3]` and `similarities [N,8]` (f32). Aggregate: `E9/pilot_results/pilot_results.md:7,10` (K=1) and `:13` (random).
  - Yaw-Aug: only `yawaug_ar_40k_vanilla/{run_manifest,summary}.json` and `summary.md:7` are in git; its per-query files are NOT.
- Paper table source: `worklog/worklog_yixun/exp_23_five_method_64_random_fallback/summary.{md:13-17,json}` (FEM 1.059 with a random fallback; exp_22 is the FEM-penalty variant, 1.130).
  - Built by `tools/aggregate_five_method_128_failure_penalized.py` (reads K key "1" at :34-39; `--query-scope-dir`, `--fem-failure-policy random_candidate` at :122-146). The exact invocation is not committed (UNVERIFIED). Random row = `pilot_results.md:13`.
  - The paper's FewShotRIR row (1.595 / 1.6 / 26.6 / 15.6) is in no committed aggregate; exp_10/22/23 all give 1.517 / 3.1 / 25.0 / 14.1. Source UNVERIFIED.
- Check: the committed per-query JSONs give K1 medians of 1.0093 m (Vanilla) and 0.9280 m (OrbitRIR), matching the paper.
- Reproducibility warning: LB's K_gen=1-only rerun (`LB/data/repeats/repeat_001`: same ckpt digests, seed 42, tau, batch 64×1) reproduces 64/64 OrbitRIR predictions but only 61/64 Vanilla (median 1.108 m instead of 1.009) and 60/64 Yaw-Aug (0.916 instead of 0.892).
  - So the Vanilla and Yaw-Aug medians move about 0.1 m from numerics alone. Cause UNVERIFIED: batch shape, card or backend.

## 2. Query selection
- Frozen manifest: `E9/pilot_manifest_seed42_4_per_room.json` (sha 6eeeec40…, 46,301 query-candidate pairs), built by `tools/build_localization_pilot.py` (defaults `--queries-per-room 4 --seed 42`, :24-29).
  - Algorithm (pilot.py:66-79): `default_rng(42)`; rooms sorted; each pool sorted by index; `rng.choice(len(pool), 4, replace=False)`.
  - Pool: the 5,337 mesh-available unseen queries in the audit (ListeningRoom_idx_2 excluded, OBJ absent). It binds the context and audit shas (pilot.py:155-158).
- Rooms (4 each, 27–5,296 candidates per query): Apartments_idx_42, Apartments_idx_50, Auditorium_idx_1, Bathrooms_idx_14, Bathrooms_idx_18, Bedrooms_idx_18, Bedrooms_idx_33, Cafe_idx_1, LivingRoomsWithHallway_idx_25, LivingRoomsWithHallway_idx_30, MeetingRoom_idx_20, MeetingRoom_idx_32, Office_idx_10, Office_idx_11, Restaurants_idx_22, Restaurants_idx_24.
- Relation to the bundle's 128 queries: `LB/data/selection/frozen_16room_128.json` (9bf75810…) = union of this seed-42 pilot and the disjoint seed-43 `E9/pilot_manifest_seed43_batch2_4_per_room.json` (805093b3…, built with `--exclude-pilot-manifest`).
  - Verified: the 64 queries are exactly its seed-42 half, with no overlap. The LB protocol copies of the context manifest and audit are byte-identical to E9's.

## 3. Candidate grid
- How it is built:
  - `reconstruct_room_base_candidates` (engine.py:349-365) opens the OBJ at `audit.rooms[r].mesh_path` with Open3D (geometry.py:72-107).
  - Lattice: `build_lattice(aabb_min, aabb_max, audit["grid_spacing_m"])`, k·Δ per axis from ceil(min/Δ)·Δ to floor(max/Δ)·Δ, so it is anchored at the global origin (geometry.py:45-69).
  - Validity: free space by 31-ray odd-parity majority, surface clearance ≥ 0.20 m (:14-16, :165-188), hash-checked against `base_points_sha256`.
  - Per-query filter (engine.py:368-411 → geometry.py:191-219): receiver ≥ 0.5 m, context sources ≥ 0.25 m, z within [min z_ctx − 0.5, max z_ctx + 0.5]; checked against `z_indices_sha256` and `chosen_count`. Ground truth is never inserted.
- Files on this box:
  - The audit stores absolute `mesh_path`s under `/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/room_mesh_obj_format/`, used verbatim by engine.py:353.
  - All 16 OBJs exist at `/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/<Scene>/<Room>.obj` (via the FLAC/AcousticRooms symlink), and all 16 sha256 match the audit's `mesh_sha256`. All 640 query/context wavs and depth npys for the 64 queries are present.
  - `outputs_loc/` holds only exp18/20/22 (localization-exp) output.
- Spacing is not a CLI flag. The engine reads `audit["grid_spacing_m"]` (engine.py:354), but the audit tool hard-codes 0.5 (`tools/audit_localization_geometry.py:88, 243`; `--mesh-root` at :335), and rooms and query masks are hash-locked (engine.py:362-364, 400-410).
- 0.75 m or 1.0 m through the engine:
  1. Parametrize the spacing in the audit tool and regenerate the audit. Pin `z_band`: `choose_z_band_branch` (geometry.py:235-244) uses a 0.5 m threshold and may flip on coarser grids.
  2. Rebuild the pilot with seed 42 and 4 per room. The same 64 indices are expected, because selection depends only on pool sizes (UNVERIFIED).
  3. Run localize_FLAC unchanged. Note that the noise seed uses the in-grid index, so a re-gridded run draws different noise.
- The 1.0 m grid is an exact subset of the 0.5 m grid (same origin, pointwise filters).
  - Verified on the committed npz: all 46,301 candidates are on the 0.5 m lattice; 6,610 (14.3%) have integer coordinates; every query keeps ≥ 6 (median 31).
  - So for P1 and OrbitRIR a 1.0 m readout is just the integer rows of the existing similarity matrices (K = 1, 4 or 8), with no GPU, because the 0.5 m indices keep the samples identical. Yaw-Aug needs a rerun, or LB's separate K1-only npz.
  - 0.75 m is not nested (only multiples of 1.5 m coincide), so it needs a new audit and a GPU run. The resolution-aware oracle must be computed over the grid actually used (scoring.py:54-57).
- Coarse-to-fine (training-free): write a subset-scoring variant of `execute_query` (runner.py:244-400).
  - Reuse `load_frozen_query` (engine.py:303-334), `encode_audio_features` (:161), the per-query `_cache_branch` context cache (runner.py:105-119), `candidate_metadata` (engine.py:337), `candidate_seed` with the FULL 0.5 m index, `generate_and_score_batch` (:466-490), and `log_mean_exp_scores` / `stable_argmax` / `localization_metrics` (scoring.py).
  - Steps: (1) score the integer-coordinate subset of the frozen 0.5 m set; (2) take the top-k; (3) add the 0.5 m candidates within 0.5 m on every axis of each top-k point (at most 26 new each; validity = membership in the 0.5 m set); (4) take the argmax over the union. The fine set is well defined because the integer lattice sits inside the half-integer lattice with the same origin.
  - Cost: about 14% of the 0.5 m pass plus at most 26·k per query. It can be simulated offline from the committed npz (P1, OrbitRIR) before any GPU run.

## 4. Metrics
- Per query (scoring.py:45-67): e_loc = ‖ĉ − x_gt‖; oracle = min over the query's candidates of ‖c − x_gt‖ (= δ_grid); excess = max(0, e_loc − oracle); SR@r flag is e_loc ≤ r; resolution-aware SR@0.5 (`oracle_normalized_success_0_5m`) is excess ≤ 0.5.
- Score: S = τ·(logsumexp(s/τ) − log K) with τ = 0.1 (scoring.py:11-35); at K=1 it is the plain AGREE cosine. Features are normalized (Retrieval.py:48). Ties go to the first candidate in lexicographic order (scoring.py:38-42).
- Aggregation: pooled over the 64 queries; `np.median` for error, flag means for SR (aggregator :85-105; reporting.py:45-71).
- No synthesis metrics: the run saves only candidates and cosines (runner.py:173-191). The paper's RIR table comes from separate `eval_FLAC.py` runs.

## 5. Frame averaging at inference
- Angles are not a CLI argument; the runner always passes `DEFAULT_FRAME_ANGLES` (runner.py:105-113). {0} or {0,180} needs a small change: thread a `--frame-avg-angles` into `_cache_branch` and record it in the identity (:459-461).
- The engine already accepts any tuple starting at 0 (engine.py:74-110; check at :84-85): {0} returns the base pass (:91), i.e. cylindrical pose invariants plus one unrotated ViT pass; {0,180} is C2, an exact 256-column roll; C4 is the default.
- The FA cache is already split: per-query `context_poses_vit` + `context_audio`; per-candidate `source_vit` + `source` + dynamic `context_poses` (dphi measured relative to the candidate) (engine.py:36-44; runner.py:283-330).
- A 90° query rotation has no code in `src/localization` or the localize scripts. It needs about 10 lines: apply `rotate_scene_metadata(md, radians(90), 512)` (yaw_rotation.py:160-218; 128-column roll plus rotation of all pose keys) to the query metadata after `load_frozen_query` and to each `candidate_metadata` item.
  - Candidates and metrics stay in the global frame; this matches eval_FLAC `--rotate-deg`. OrbitRIR (C4 + cylindrical features) is invariant up to float roundoff; Vanilla and Yaw-Aug are not.
  - A depth-only roll (`pose_keys=()`) would be physically inconsistent.

## 6. Runtime and memory (2× RTX A6000 48 GB, torch 2.7.0+cu126, per `LB/ENVIRONMENT.md`)
- Pilot, nested K_gen=8 (64 candidates × 8 = 512 RIRs per forward):
  - Summed per-query time: Vanilla 2,673.6 s and OrbitRIR 2,922.5 s (from the committed JSONs); Yaw-Aug 3,688.9 s (`yawaug…/summary.md:12`). That is about 58 / 63 / 80 ms per candidate.
  - Per query: 1.7 s (27 candidates) up to 304.5 / 332.9 s (Cafe, 5,296 candidates). Peak allocated memory: 7.19 GB for every arm.
  - Pre-registered estimate: 0.75 / 0.84 GPU-h (`E9/localization_pilot_experiment_report.md:202`). Model load time was not logged.
- K_gen=1 (LB repeat_001, synchronized, 1 warm-up, core forward only): on the 64 queries Σ elapsed ≈ 420 s (Vanilla), 702 s (OrbitRIR), 410 s (Yaw-Aug). Over all 128 queries: 8.9 / 15.1 / 8.8 ms per candidate (`LB/expected/summary_final.md`). Peak memory: 1.29 GB.
- Flags: `--candidate-batch-size` (default 64; generation batch = that × max K), `--score-sample-counts`, `--query-limit`, `--warmup-query-count`, `--synchronize-latency`, `--measure-core-forward` (requires K=1). CUDA is mandatory (runner.py:442-443).

## 7. Environment
- Imports OK in the `flac` env (checked with `CUDA_VISIBLE_DEVICES=`): open3d 0.19.0, torch/torchaudio 2.7.0+cu126, transformers 4.57.0, PL 2.1.0, numpy 1.23.5, scipy, einops, ema_pytorch, k_diffusion, alias_free_torch, librosa, prefigure, safetensors, threadpoolctl, wandb.
  - Missing: `sparse_dot_mkl`, which is only the optional FEM extra.
  - Backend: `flash_attn 2.7.4.post1` IS installed, and NE `src/models/transformer.py:12-18` uses it when importable. The pilot ran without it, so a faithful rerun needs the reviewed no-flash mechanism (exp_22/35) or a disclosed backend change.
- Run NE code from its own tree: it needs `AGREE/AGREE/model_configs/dinov3_vits16_local/config.json` (engine.py:48-66), which is absent on localization-exp, and NE `src/` differs from localization-exp.
- Machine-specific dependencies:
  - The scripts' `/home/zhixuanzhao/...` paths are plain CLI args and easy to swap.
  - Hard binding: the audit's absolute `mesh_path`s are used verbatim, and the audit sha is bound by the pilot manifest (pilot.py:157) and the run identity (runner.py:455). Running here needs either a mesh-root remap in the engine (the local meshes are hash-identical, and the base-points hash still guards the grid) or a rebuilt audit plus pilot, which gives new shas.
  - The P1/BF EMA extracts and the Yaw-Aug per-query files exist only on that machine. `--output-dir` must lie inside the repo (localize_FLAC.py:41-45).

## 8. P1 checkpoint at step 87,500
- None on this box.
  - The requested `find -maxdepth 6 -name '*87500*' -o -name '*87.5k*'` over `/home/yixunhu/codespace/*/weights`, `FLAC/weights` and `/media/diskstation/yixunhu/FLAC/checkpoints` returns nothing. The largest step of any ckpt on the NAS is 67,500.
  - An extended name search (depth 7-8 over `/home/yixunhu/codespace` and `/media/diskstation/yixunhu`) found only 87.5k metric JSONs (e.g. `FLAC/outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500_metrics_*.json`) and logs. A depth-10 search timed out, so anything deeper is UNVERIFIED.
- Recorded digest: `bd3fc7db…`, in localization-exp `worklog/worklog_yixun/exp_13_decay_tail_claude/anchor_87500.sha256`, for `outputs_FLAC/exp07_P1/.../epoch=19-step=87500.ckpt`. Probably on neuronic (UNVERIFIED).
