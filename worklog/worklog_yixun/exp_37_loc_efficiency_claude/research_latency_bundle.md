> Read-only research report by an Opus 5.5 subagent (2026-11-06), committed verbatim as the plan's evidence; model output, checked by the Planner where the plan relies on it.

# B1 prep: the NIPS-Experiment latency bundle (origin/NIPS-Experiment dca796dd), read-only research
Refs: B = origin/NIPS-Experiment:NeurIPS_2026_Workshop_Latency_Reproducibility/, M = B/code/measurement, C = B/code/calculation, R = B/code/reference, N = origin/NIPS-Experiment: (branch root). Nothing was run on a GPU; all numbers below come from the shipped JSONs or static reading.

## 1. benchmark_five_method_latency.py: an orchestrator only (the timers are in R/src/localization/runner.py)
- CLI (M/benchmark_five_method_latency.py:370-378): --config, --stage {run,summarize,all}, --repeat-count, --warmup-query-count, --resume, --dry-run. The config (schema 2) must name all 4 learned methods plus the FEM and selector paths (:94-115). validate_config requires every path to exist and output_root to be inside the repo (:118-153).
- Inputs (B/data/protocol/original_benchmark_config.json): selection frozen_16room_128.json (:41), exp_09 context_manifest_exp01_seed42.json (:4), geometry_audit.json (:12), dataset_root (:6), AGREE_fullAR.pt (:2). All 3 FLAC arms use src/configs/model_configs/FLAC/AR/FLAC_AR.json (sha f3eafef4…). Checkpoints are P1_40k_clean_hybrid_EMA, BF_40k_clean_hybrid_EMA and YAWAUG_AR_40k_epoch8-step40000 (:14-32). Batch 64 (:3), score_seed 42, tau 0.1, warmup 1 (:43), repeat_count 3 (:37; overridden to 1 on the CLI).
- Per-arm call it builds (:192-219): `<python> localize_FLAC.py --model-config … --ckpt-path … --agree-ckpt … --context-manifest … --geometry-audit … --pilot-manifest <selection> --dataset-root … --output-dir <repeat>/<method> --device cuda:0 --cond-method {vanilla|fa_invariant} --candidate-batch-size 64 --sample-seed 42 --tau 0.1 --score-sample-counts 1 --synchronize-latency --warmup-query-count 1 --measure-core-forward`.
  - The environment sets CUDA_VISIBLE_DEVICES from the config ("0") and MPLCONFIGDIR (:187-190). Arms run serially: vanilla, fa_bf, yawaug, few_shot (:26-33).
  - Output goes to <repeat>/logs/<method>.log (:340-347) and benchmark_manifest.json (:436-470; context_count 8 is hard-coded at :449).
- Timed regions (R/src/localization/runner.py, execute_query :244-400, under inference_mode):
  - context_conditioning: sync :281 → perf_counter :282 → fp16 autocast _cache_branch(context ids) :283-292 → sync :294 → :295-297.
  - candidate_conditioning_and_generation, per 64-candidate batch (loop :302; last batch partial): sync :305 → t :306 → candidate_metadata :307-311 → autocast source branch (+ FA dynamic branch) :312-330 → _repeat_branch + seeds :331-337 → generate_rir_batch :339-345 (outside autocast, fp32 with TF32 set at :475-477) → sync :347 → accumulate :348. AGREE scoring of the batch (:349-350) is outside the timer.
  - core_forward_total = the sum of the two (:395-399). elapsed_seconds (sync :267-269 … :369-371) is not used by the table; it also covers query I/O, filtering, the observed AGREE encode (:277-278) and the AGREE scoring.
  - Peak memory: reset_peak_memory_stats before each measured query (:526), max_memory_allocated after the batch loop (:366-368) → peak_memory_bytes.
- Warm-up (:484-507): the first selection record (index 5182, Apartments_idx_42 S007_R006, 176 candidates) runs through the full execute_query and its result is discarded; the measured loop then runs it again as query 1.
- K_ctx has no CLI flag: it is whatever the context manifest holds (8 per record; manifest protocol max_context 8). It is hard-guarded at :274-275 (raises unless 8), identity n_context 8 (:462) and latency_protocol context_count 8 (:386-388). Few-ShotRIR alone gets --context-counts 8 (M/…:238-239).
- OrbitRIR FA: conditioning_method fa_invariant → --cond-method fa_invariant → _cache_branch → cache_invariant_conditioning_branch(…, DEFAULT_FRAME_ANGLES) (runner.py:105-113).
  - The angles (0, 90, 180, 270) are fixed in R/src/data/yaw_rotation.py:23 (no CLI) and recorded as identity frame_average_angles (:459-461).
  - FA context ids = (context_poses_vit, context_audio). A per-candidate dynamic context_poses branch adds cylindrical poses relative to the candidate azimuth (R/src/localization/engine.py:36-45; yaw_rotation.py:114-135).
- Per-query JSON (runner.py:222-241, :384-399; sha/arrays fields :186-190): query/candidate identity, n_context, score_sample_counts, metrics, elapsed_seconds, peak_memory_bytes; latency_protocol{name fem_core_aligned_kctx8_kgen1, context_count 8, generated_rirs_per_candidate 1, all five *_included false}; latency_seconds{context_conditioning, candidate_conditioning_and_generation, core_forward_total}.
- run_manifest.json (:450-473): sha256s of the model config, checkpoint, AGREE, context manifest, geometry audit and pilot; query_indices, conditioning_method, frame_average_angles, n_context 8, score_sample_counts [1], tau, sample_seed 42, candidate_batch_size 64, synchronize_timing / warmup 1 / measure_core_forward; sampler {rectified_flow_discrete_euler, steps 1, cfg 1.0}. The 3 shipped manifests (B/data/repeats/repeat_001/*/run_manifest.json) match these flags.
- Recorded command: only the driver line, B/RAW_MEASUREMENT.md:10-13 (`CUDA_VISIBLE_DEVICES=0 python tools/benchmark_five_method_latency.py --config worklog/worklog_yixun/exp_29_core_forward_latency_16room_128/benchmark_config.json --repeat-count 1`).
  - The rendered per-arm commands were in repeat_001/logs/*.log and data/execution_logs/driver_resume_gpu0*.log. These 7 logs are listed in B/SHA256SUMS but are not in git (N/.gitignore:59 `*.log`), so the per-arm line above is a reconstruction (UNVERIFIED as executed).
  - B/ENVIRONMENT.md:17: Vanilla/FA ran on physical card 1 and Yaw/Few-Shot on card 0 after a resume (18 contended Yaw rows were deleted), so repeat_001 was assembled from at least 2 runs.

## 2. benchmark_localization_selector_latency.py (AGREE selector)
- CLI M/…:159-171; original call B/RAW_MEASUREMENT.md:21-33 (--candidate-batch-size 64 --repeat-count 3). It refuses unless the 128/112/97/15/16 FEM accounting holds (:181-197), so the FEM inputs (B/data/selection/depth_aabb_matched_16room_112.json, B/data/selector/fem_response_cache_97/) are needed even for AGREE-only use.
- Warm-up: one unreported encode of zeros [64,1,10240] (:206-210). Then per query, 3 repeats (:65):
  - observed RIR: sync → t → H2D + encode → sync (:66-73);
  - per batch: zeros((count,1,10240), float32, device) is allocated outside the timer (:82), then sync → t → encode_audio_features → sync (:83-89);
  - similarity is timed without a sync (:91-94); features are already on the CPU because Retrieval.compute_audio_features returns .cpu() (N:src/metrics/modules/Retrieval.py:44-50); then cat + argmax (:96-101).
- Each field is a statistics.median over the 3 repeats (:46-50, :116-126); the table uses agree.median_seconds.scoring_total_seconds (the median of the totals). There is no autocast.
- The cost is shape-only, so it is arm-independent by design: Σ of the medians = 86.8 s over 128 queries (0.678 s/query; B/README.md:91-99), about 0.94 ms per candidate.

## 3. Statistics
- Per query L_q = core_forward_total_q + selector median scoring_total_q (C/summarize_core_forward_latency.py:317-322). s/query = mean, np.median and np.quantile(·, 0.9) (linear) over the 128 L_q (:169-181, :376).
- ms/candidate: dataset-wide mean = 1000·ΣL_q / Σ|C_q| (amortized_seconds :382-384; Σ|C| = 92,608); query median / P90 = median / P90 of {1000·L_q/|C_q|} (:380-381; ×1000 at :235-239).
- C/aggregate_kctx8_kgen1_latency.py takes the per-query median over repeats (:173-188; with one repeat this is the identity), then the same statistics (:197-212).
- Both tools require all 5 methods; summarize also requires n_context == 8 (:312), selector repeat_count == 3 (:190), and the Few-ShotRIR and FEM inputs (:258-298). B1 therefore needs a reviewed 3-arm variant; the formulas carry over unchanged.
- My recomputation of the bundle from the 3×128 query JSONs (P1 / BF / YAW):
  - context s/query mean 0.090 / 0.324 / 0.091; candidate part 7.86 / 13.71 / 7.77 ms/candidate (dataset-wide);
  - adding context (0.12 / 0.45 / 0.13) and AGREE (0.94) gives 8.92 / 15.09 / 8.83, matching the table;
  - peak_memory_bytes max 1,228 / 1,229 / 1,228 MiB, min 762 MiB: the same for every arm, so it does not separate stages.

## 4. Splitting candidate_conditioning_and_generation (one batch = runner.py:306-347 + engine.py:409-452)
- (a) Candidate conditioning = runner.py:307-337, plus the glue at engine.py:426-432 (merge_cached_conditioning + get_conditioning_inputs):
  - candidate_metadata (CPU dicts; engine.py:337-346);
  - source_vit/source through MultiConditioner (R/src/models/conditioners.py:288-337). GeometryConditioner (:194-224) does one H2D copy of the 3×256×512 depth per candidate (:199-201), then one DINOv3 pass on [64,3,256,512] (:209-221), plus the DistEmbedder;
  - OrbitRIR uses cache_invariant_conditioning_branch (engine.py:74-110): CPU cylindrical features (:86), a base pass (:87), 3 rotated passes (:95-107) each preceded by a CPU torch.roll + einsum of every candidate's depth (yaw_rotation.py:198-216), then the mean (:108-109); plus the dynamic context_poses branch (runner.py:320-330);
  - _repeat_branch + blake2b seeds (:331-337).
- (b) Single DiT step = deterministic_noise (engine.py:433-440; one CUDA Generator + randn per seed) + sample_discrete_euler with steps=1 (:441-450 → R/src/inference/sampling.py:121-130, one forward; cfg 1.0 means no batch doubling, R/src/models/dit.py:320).
- (c) VAE decode = pretransform.decode(...).float() + clamp (engine.py:451-452). FLAC_AR.json:9 sets iterate_batch true, so this is 64 sequential single-sample bottleneck + decoder calls (R/src/models/autoencoders.py:312-327, pretransforms.py:64-76).
- Already timed separately: context encoding (runner.py:280-297; FA = 32 batch-1 ViT passes, vanilla 8) and the AGREE selector (§2).
- Minimal hook: an optional stage-timer argument to generate_rir_batch (None = unchanged code path) plus one mark in the runner before :338.
  - Preferred: torch.cuda.Event(enable_timing=True) recorded after :305, before :338, and after engine.py:440, :450 and :452, read after the existing sync at :347. Events add no host syncs, so CPU/GPU overlap and the totals are preserved; CPU-only work lands in the interval where it happens.
  - Adding sync + perf_counter at those boundaries instead means about 3 extra syncs per batch × 1,504 batches per repeat. That serialises kernel launch against execution between stages and biases the totals upward (size UNVERIFIED; probably small).
  - Either way, take the headline numbers from a pass with exactly the bundle's sync placement. Run the split separately, or show the event pass's totals equal the headline, and report Σstages − headline.
- Memory: keep the per-query reset/max (:526, :366-368). For per-stage peaks, call reset_peak_memory_stats before and max_memory_allocated after each stage (host-side allocator counters, no device sync), keeping a running max per query. Also log max_memory_reserved and memory_allocated() after loading the model and AGREE (:478-479) as the weight footprint. Allocator numbers exclude the CUDA context.

## 5. Environment and flash-attn
- Bundle host (B/ENVIRONMENT.md:5-15): 2× RTX A6000 48 GB (GPU-d4f1b10f…, GPU-295d7083…), driver 595.84, Xeon Gold 5220R 24c/48t, Linux 6.8.0-136, Python 3.10.12, NumPy 1.23.5, torch/torchaudio 2.7.0+cu126, SciPy 1.15.3, CUDA 12.6. Interpreter: /home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python (config :36).
- flash-attn is not recorded anywhere in the bundle: no backend field in the manifests, nothing in RAW_MEASUREMENT or ENVIRONMENT; the only mention is the upstream install note at R/README.md:32. The logs that would show it were dropped by .gitignore.
  - Indirect evidence: N:worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_results/pilot_run.log:10-11 (also pilot_results_batch2/pilot_run_batch2.log:10-11) comes from the same .venv and prints "No module named 'flash_attn' / flash_attn not installed, disabling Flash Attention".
  - So the timed DiT most likely ran on SDPA in fp32. This is UNVERIFIED for the latency runs themselves.
- This box: same CPU model (5220R, 48 threads), 2× RTX A6000 but different cards (GPU-0a1aaa90…, GPU-6ac2604a…), driver 590.48.01, kernel 6.17; the flac env has Python 3.10.20 with the same torch/numpy/scipy pins.
- flash_attn 2.7.4.post1 is installed in the env (2026-11-01 23:14). R/src/models/transformer.py:12-18 and :413/:429-440 (identical to this checkout's src/models/transformer.py) then send every DiT self/cross-attention through flash_attn_func, casting q/k/v from fp32 to fp16, instead of fp32 SDPA.
  - Only the DiT is affected: HF DINOv3 uses flash only on explicit request (conditioners.py:374-379 does not ask), and AGREE has no flash_attn reference.
- Implication: to match the bundle, B1 needs a no-flash DiT (e.g. by adapting the reviewed overlays/no_flash/{flash_attn.py, exp35_no_flash_launch.py} on localization-exp HEAD) and must log the backend actually selected via `src.models.transformer.flash_attn_func is None` (find_spec cannot see a shadow). All 3 arms must share one backend; a flash pass is optional extra data.
- The timed region includes CPU work (metadata, seeds, FA CPU rotations, per-candidate depth H2D), so a free card alone is not enough: CPU co-tenants also perturb the numbers.

## 6. Dependencies, paths, data on this box
- Imports: a static scan of the 68-file closure (localize_FLAC.py, both benchmarks, summarize, aggregate) found that every third-party module imports in flac (CUDA_VISIBLE_DEVICES= checks), except two that are never reached: horovod is guarded by try/except (N:AGREE/AGREE/loss.py:15-18), and umap is a lazy import used only with method='umap' (N:src/interface/aeiou.py:48-49). Importing the repo modules themselves is UNVERIFIED (needs a checkout).
- B/code is not standalone: engine.py imports eval_FLAC (R/…/engine.py:16) and the AGREE package (:271-272), and neither is in the bundle. N (dca796dd) has the full layout:
  - localize_FLAC.py, both benchmarks and summarize are byte-identical to the bundle;
  - N:src/localization/engine.py adds only the optional candidate_filter_context_sources_global override (:368-381; no effect when the key is absent); src/models/factory.py, src/training/factory.py and src/data/dataset.py only add FewshotRiR branches;
  - this checkout's localization-exp src/localization is a different (meshgrid) stack, so B1 needs its own worktree from N (none created).
- Paths to re-point (/home/zhixuanzhao does not exist here): python (:36) → /home/yixunhu/miniconda3/envs/flac/bin/python; dataset_root (:6) → AcousticRooms (a NAS link); AGREE (:2) → weights/AGREE/AGREE_fullAR.pt (sha 3a13243d…, same as the bundle); checkpoints (:15/:24/:29) → weights/exp20/{BF,P1,YAW}_40k.ckpt.
- Checkpoints: YAW_40k.ckpt has sha ac1f2603…, identical to the bundle's. P1/BF here are the wrapped 40k checkpoints (c4c67882…, 5319feb4…), not the bundle's EMA extracts (da127485…, 0f61277f…), which are absent on this box.
  - load_flac_module swaps in the EMA weights (engine.py:248-256; FLAC_AR.json:137 use_ema true; leftover keys whitelisted at N:eval_FLAC.py:106), the same path the bundle's wrapped YAW checkpoint took. The run_manifest checkpoint_sha256 will differ for P1 and BF.
- Geometry audit: rooms[].mesh_path is absolute (/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/room_mesh_obj_format/…) and is read at engine.py:353, outside the timers.
  - The audit is SHA-bound (runner.py:439-441; R/…/pilot.py:155-158), so editing it breaks those checks. It needs either a reviewed path remap or a rebound audit + selection (pattern: N:tools/rebind_fewshot_rir_candidate_protocol.py).
  - All 16 meshes exist at AcousticRooms/room_mesh_obj_format/<scene>/<room>.obj with sha256 equal to the audit's mesh_sha256 (16/16).
- The orchestrator refuses on this box because the Few-ShotRIR checkpoint and the FEM result dirs (config :8-11, :20) are not in git. So call localize_FLAC.py per arm directly; its output dir must be inside the worktree (localize_FLAC.py:41-45).
- Data: all 1,280 files the 128 queries need (128 targets, 1,024 contexts, 128 depth .npy) exist under AcousticRooms. The exp_09 context manifest and geometry audit in N are byte-identical to B/data/protocol.
- K_ctx=1 needs code changes: runner.py:274/:387/:462 and summarize :312 hard-code 8. Candidate filtering uses the contexts (engine.py:371-381, with a mask-hash check at :396-397), so K=1 model inputs must keep the 8-context filter (the N override above), and which single context is used (e.g. contexts[0]) must be decided in advance.

## 7. Expected wall time (exclusive A6000; bundle numbers, same GPU/CPU model)
| arm (K_ctx=8, 1 repeat) | Σ core (timed) | Σ (core + AGREE) | Σ elapsed_seconds | estimate incl. startup |
|---|---:|---:|---:|---:|
| P1 vanilla | 739.5 s | 826.3 s | 839.0 s | ≈ 15 min |
| BF OrbitRIR | 1,310.8 s | 1,397.6 s | 1,405.0 s | ≈ 25 min |
| YAW | 731.0 s | 817.8 s | 819.7 s | ≈ 15 min |
- Startup is UNVERIFIED, about 1-3 min per arm (model + AGREE load, 16 open3d mesh rebuilds, warm-up query 1.6 / 3.2 / 1.6 s). The load is skewed: Cafe_idx_1 (42,365) and Auditorium_idx_1 (29,782) hold most of the 92,608 candidates; 1,504 batches per repeat.
- K_ctx=1 only shrinks context encoding (8→1 and 32→4 ViT passes): about −10 s for P1/YAW and −35 s for BF per repeat, so wall time stays about 15/25/15 min. The selector runs once (independent of arm and K): 3 × 86.8 s timed plus loads, about 5-6 min.
- Total: one repeat of 3 arms × K_ctx {8, 1} plus the selector takes about 2 h. A separate synchronized split pass adds about 1 h 50 min; CUDA events in the same pass add about 0. Three repeats triple the arm time.
- The commented-out paper rows (57.53 / 80.59 ms/candidate) come from N:worklog/worklog_yixun/exp_19_five_method_latency_97/summary.md: 14 rooms / 97 queries, a joint K_gen {1,4,8} pass, and no warm-up.

## Decisions for Yixun
1. K_ctx=1: which single context (rec. contexts[0] of the exp_09 manifest), keeping the 8-context candidate filter. 2. Backend: no-flash to match the bundle (rec.), plus an optional flash column.
3. P1/BF: accept the wrapped 40k checkpoints, whose digests differ from the bundle's EMA extracts (rec.; same weights path). 4. mesh_path: in-memory remap (rec., it is outside the timers) vs a rebound audit + selection.
5. Stage split: CUDA events in the headline pass after an equality check (rec.) vs a separate synced pass. 6. Repeats: 1 (as the bundle) or 3. 7. Selector: re-measure on this box (rec.) vs reuse the bundle's file.
Process note: I briefly wrote a stray scratch file (scratchpad/sums_list.txt) by mistake and deleted it at once. Nothing else outside this report was created or changed.
