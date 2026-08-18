# plan_loc_invert — exp_18: Inverting Vanilla FLAC for Source Localization (preflight)

**Author:** Claude Fable 5 (Planner seat), 2026-08-18. Status: DRAFT → Codex plan review → revision → **awaiting Yixun approval before any implementation**.

## 1. Objective & research question

Test whether a frozen, pretrained Vanilla FLAC contains enough source-position information to localize the source of a held-out RIR in an **unseen room**, purely by analysis-by-synthesis inversion (no localization training). Success = localization error substantially below the random-candidate baseline, with heatmaps whose dominant mode sits near the true source. exp_18 also delivers the protocol + code that the later cross-arm experiment (FA B-F, yaw-aug, cyl-DINOv3 FLAC) will reuse unchanged.

## 2. Protocol (registered)

### 2.1 Queries
- Split: the **full** existing unseen config `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json` → `data/AR/unseen_eval.json`, 6,337 items / 17 rooms (announcement 01; no new or subsampled eval configs). Each item = one query: h_obs = h(S_gt, R), receiver R known, source hidden.
- Context: the standard pipeline (K_ctx = 8 references drawn from OTHER source nodes at the SAME receiver — `AR_md.get_ir_and_location_for_other_sources`). Target-source leakage is excluded **by construction**; the driver additionally asserts it per query (context fingerprint via `eval_FLAC.sample_context_ids` must not contain the GT source position) and hard-fails on violation.
- All M candidates for a query share the same receiver pose, depth panorama, and the SAME drawn context (context computed once per query).
- Dataloader: existing config, `shuffle=False`, `batch_size` for iteration only; `pl.seed_everything(seed)` fixes the context draws. Every item's identity is verified with `sample_target_id`; silent dataset substitutions (`SampleDataset` retry-on-failure) are detected by comparing against the split enumeration and **logged + excluded**, count reported in `_results.md` (expected 0).

### 2.2 Candidate set
- C = all source nodes present in the query's room directory (unique S-nodes from filenames), world coordinates from the room's `metadata/` pair JSONs (`src_loc`); GT source included. Expected M ≈ 10; per-query M logged.
- Node parsing is numeric and naming-tolerant (handles `S010` vs the release code's `"S00"+str(node)` quirk); src_loc consistency across receivers is asserted (tolerance 1e-6) when multiple pair files exist.
- Per-candidate conditioning: `source` / `source_vit` = candidate position projected into the receiver camera frame (same transform as `AR_md.get_3d_point_camera_coord`; parity-tested). `depth`, `context_*` unchanged.

### 2.3 Generation
- Frozen FLAC, EMA weights, rectified flow, `steps=1`, `cfg_scale=1.0` (paper protocol), `--cond-method vanilla` for this arm (announcement 05: the flag is declared in every manifest; the driver carries the full cond-method dispatch so later arms run `fa_invariant` correctly).
- K = 8 samples per candidate (fresh noise each; conditioning computed once per query for all M candidates, tiled K× for sampling). All M×K generations for a query run in ≤2 GPU batches.

### 2.4 Scoring
- Embed with the frozen AGREE **audio branch**: `E_a = model.encode_audio(·, normalize=True)`, inputs padded/cropped to 10,240 samples exactly as `Retrieval.compute_audio_features` does.
- s_{m,k} = cos(E_a(h_obs), E_a(ĥ_{m,k})); per-candidate aggregation S_m = τ·log((1/K)Σ_k exp(s_{m,k}/τ)) (log-mean-exp); prediction = argmax_m S_m (deterministic lowest-index tie-break).
- **All per-sample s_{m,k} are logged**, so aggregator/τ/K′≤K sensitivity is offline re-aggregation — no regeneration, no test-set tuning.

### 2.5 Hyperparameter registration (no unseen-set tuning)
- τ, aggregator (lme/mean/max), and effective K are selected on the **seen-room dev split** (`acousticroom_seeneval.json`, existing full config): one dev generation run at K=8, then an offline sweep (τ ∈ {0.02, 0.05, 0.1, 0.2, 0.5}, methods {lme, mean, max}, K′ ∈ {1,2,4,8}) minimizing dev median e_loc. The winning triple is registered in `_params_set_up.md` BEFORE the unseen run; unseen results additionally report mean/max as labelled secondaries.

### 2.6 Metrics (headline on the unseen split)
- Primary: median e_loc (m); also mean e_loc, success@0.5 m, success@1.0 m. Aggregation follows the paper convention: per-room means first, then average over the 17 rooms; pooled-over-samples values reported as secondaries.
- Secondary (natural for a discrete C): top-1 candidate accuracy, MRR of the GT candidate.
- **Random-candidate baseline (lower)**: exact, per query — uniform over C gives E[e_loc] = mean of candidate distances-to-GT, success@r = fraction of candidates within r, median from the pooled per-candidate distance distribution (weight 1/M per query). No Monte Carlo.
- **GT-RIR oracle (upper)**: replace generated RIRs by measured candidate RIRs at the same receiver, same scorer. ⚠️ For candidate = GT this IS h_obs (cos = 1), so the plain oracle is only a pipeline sanity check (must hit ~100% top-1). A non-trivial oracle needs a second measurement of the same pair (e.g. a different `single_channel_ir_*` channel, if present — verify at data readback rung); if none exists, we report the identity-oracle as sanity-only and say so.
- Seeds: primary seed 42; seeds 43, 44 repeated on the unseen split for headline error bars (cheap at ~1–4 h/run; final say at approval).

### 2.7 Heatmaps
Top-down (x–y) per-query maps: candidates colored by softmax(S/T) (T for display only), receiver, GT source, prediction, candidate/receiver extent as the room region proxy (z encoded by marker size or annotation). Representative cases chosen by a **pre-registered rule**: sharp-success = correct top-1 with largest top-2 margin; ambiguous = smallest top-2 margin; failure = largest e_loc. Three per category across rooms, picked by the rule, not by hand.

## 3. Checkpoints & assets (blocked on Yixun; see §8)
- FLAC arm (this exp): a Vanilla FLAC ckpt — preflight smoke can use released HF `FLAC_EMA.ckpt` (downloadable now); the registered run should use the program's vanilla checkpoint (recommended: exp07_P1 anchor `epoch=19-step=87500.ckpt`; the matched-step VANL@40k row belongs to the later cross-arm experiment).
- Scorer: recommended primary `AGREE_AR.pt` (train-split-only — unseen rooms genuinely unseen by the scorer); `AGREE_fullAR.pt` only as a labelled leaky diagnostic.
- Dataset: `AcousticRooms/` must land on this box (rsync or re-download) with `metadata/` and `single_channel_ir_1/`.

## 4. Implementation plan (per file, with per-function test list — announcement 02)

New code is additive; no edits to release files (`AR_md.py`, `src/data/`, `train.py`, …). Tests in `src/tests/`.

### 4.1 `src/localization/__init__.py` — empty package marker.

### 4.2 `src/localization/candidates.py`
| Function | Contract | Tests (`src/tests/test_loc_candidates.py`) |
|---|---|---|
| `parse_ir_filename(name) -> (src_node, rec_node)` | numeric parse of `S…_R…_hybrid_IR.wav`, any digit count | std names; `S010`; malformed → ValueError |
| `find_pair_metadata(meta_room_dir, src, rec) -> Path\|None` | locate pair JSON by numeric identity, naming-tolerant (`S010`/`S0010`) | both namings (tmp_path fixtures); missing → None |
| `load_source_location(meta_room_dir, src, prefer_rec) -> np.ndarray` | src_loc from prefer_rec's pair file, fallback any receiver; cross-receiver consistency assert | fixture tree; fallback path; inconsistent src_loc → error |
| `build_candidate_set(ir_path, metadata_path) -> CandidateSet` | dataclass(nodes, xyz_world [M,3], rec_loc, gt_node, gt_xyz); nodes = unique S in room dir, sorted; GT ∈ C | fixture room (3 sources); GT flagged; det. order; missing metadata → error |
| `project_to_camera(rec_loc, xyz) -> np.ndarray` | receiver-frame translation (no rotation), reimpl of `get_3d_point_camera_coord` | **parity test**: importlib-load `AR_md.py`, assert equality on random points |
| `candidate_metadata(base_md, cand_cam_xyz) -> dict` | deepcopy of the query metadata with ONLY `source` ([3] f32) and `source_vit` ([1,3] f32) replaced | only those keys differ; dtypes/shapes; base_md not mutated |

### 4.3 `src/localization/scoring.py` (pure torch/numpy; no I/O)
| Function | Contract | Tests (`src/tests/test_loc_scoring.py`) |
|---|---|---|
| `cosine_sims(obs_emb [D], gen_embs [M,K,D]) -> [M,K]` | inputs assumed L2-normalized; asserts norms ≈ 1 | hand-built vectors; norm-guard raises |
| `aggregate(sims, method, tau) -> [M]` | lme (τ>0), mean, max | lme→max as τ→0⁺; lme→mean for large τ; K=1 ⇒ all equal; τ≤0 with lme → ValueError |
| `predict_index(scores) -> int` | argmax, lowest-index tie-break | tie case |
| `softmax_map(scores, T) -> [M]` | sums to 1, T>0 | invariance to score shift; sum=1 |
| `localization_error(a, b) -> float`, `success_within(err, r)` | L2; boundary r inclusive (registered: err ≤ r) | hand values; boundary |
| `random_baseline(cand_xyz, gt_xyz) -> dict` | exact uniform-candidate expectations (mean, success@r, pooled distance list for medians) | hand example; agreement with seeded MC to 1e-3 |
| `summarize(records, radii) -> dict` | per-room-then-macro + pooled; median/mean/success/top-1/MRR | small synthetic record set with known answers |

### 4.4 `src/localization/agree_embed.py`
| Function | Contract | Tests (`src/tests/test_loc_agree_embed.py`) |
|---|---|---|
| `load_agree_audio(ckpt, device)` | thin wrapper over `metric_callback.loading_AGREE_model` (reuse, no duplication) | integration test, `skipif` no ckpt on box |
| `embed_rirs(model, wavs [B,1,T], device) -> [B,D]` | pad/crop to 10240 mirroring `Retrieval.compute_audio_features`; returns L2-normalized f32 | stub model with `encode_audio`; pad & crop branches; normalization |

### 4.5 `eval_localization.py` (repo root; argparse in the `eval_FLAC.py` house style)
Flags: `--model-config --dataset-config --ckpt-path --agree-ckpt --num-samples(K) --tau --agg {lme,mean,max} --steps --cfg-scale --seed --cond-method {vanilla,fa_invariant} --frame-avg-angles --cond-autocast --score-source {flac,gt_rir} --out-dir --eval-name --smoke --max-queries` (`--max-queries` REFUSES to run without `--smoke`; smoke outputs are stamped `smoke` in filename + record and can never be aggregated into headline files).
Structure: reuse `eval_FLAC.evaluate_model`'s loading path (EMA remap, `check_load_integrity`, cond-method dispatch, `resolve_cond_autocast`, `source_sha`, `orbit_provenance`); per query → `build_candidate_set` → M conditioner metadata → conditioner once → tile K → `sample_discrete_euler` (objective dispatch as in eval_FLAC) → VAE decode → `embed_rirs` → sims → JSONL row {query id, room, receiver, gt node/xyz, candidates, per-sample sims, S, prediction, e_loc, context fingerprint, substitution flag} → summary JSON with full provenance (git sha, ckpt sha256, flags, protocol fields).
Tests (`src/tests/test_eval_localization.py`): smoke-guard refusal; JSONL row schema round-trip; summary aggregation vs `scoring.summarize`; gt_rir mode uses measured files & marks identity candidate; cond-method flag threading (unit level, mocked model).

### 4.6 `worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_heatmaps.py`
Reads JSONL → per-query PNGs + the 3×3 pre-registered case gallery into `loc_invert_results_assets/`. Reviewed under universal-review (batched round); dataviz guidance loaded before chart code.

### 4.7 Explicitly NOT touched
`train.py`, `defaults.ini`, `src/data/`, `src/models/`, `src/training/`, `data/AR/`, `AR_md.py`, all release eval scripts. `src/localization/` + `src/tests/test_loc_*` are new files only; `eval_localization.py` is a new root script.

## 5. Run matrix
| Run | Split | Ckpt | Mode | Purpose |
|---|---|---|---|---|
| R0 smoke | 2 rooms, `--smoke --max-queries 4` | released FLAC_EMA (or first rsynced) | flac | end-to-end pipe + timing probe |
| R1 dev-tune | full seen split | registered vanilla ckpt | flac, K=8 | generate once; offline τ/agg/K′ sweep → register |
| R2 registered | full unseen 6,337 | registered vanilla ckpt | flac, registered (τ, agg, K) | headline; seeds 42 (+43, 44) |
| R3 oracle | full unseen | — | gt_rir | sanity (identity) + non-trivial variant if 2nd channel exists |
| R4 random | full unseen | — | analytic | lower baseline (CPU, minutes) |

## 6. Validation ladder mapping
1. Static: `py_compile` all new files; `git diff --check`. 2. Unit tests (all of §4, fixture-based, no GPU/data). 3. Tiny synthetic forward: model from `FLAC_AR.json` config with random init, 1 query × 2 candidates × K=2 on GPU, asserts shapes/finiteness (no weights needed). 4. Real-data readback (needs dataset): 1 room — verify metadata naming (S010 question), M, src_loc consistency, context fingerprint excludes GT. 5. R0 smoke (needs weights): sims ∈ [−1,1], JSONL schema, identity-oracle top-1 = 100% on smoke queries, per-query wall-time → full-run projection logged. 6. (no fit probe needed — inference only, M×K ≤ 80 ≪ eval batch 64 memory at batch 64 already proven). 7. Full runs per §5, acceptance criteria written in `_worklog.md` at each launch.

## 7. Integrity controls (summary)
- Leakage: context excludes target source by construction + per-query fingerprint assert; scorer = train-split-only AGREE_AR (recommended); no unseen-set tuning (dev-split registration, §2.5); no subsets in headline numbers (smoke stamped and quarantined); silent-substitution audit (§2.1).
- Protocol flags declared in params/command/logs per announcement 05; vanilla arm runs `--cond-method vanilla` explicitly.
- Every run teed to a timestamped log in the exp folder; commands recorded at launch in `loc_invert_command.md`; SHAs in `commits_loc_invert.md`.

## 8. Open decisions for Yixun (recommendation first)
1. **Registered vanilla ckpt**: exp07_P1 anchor 87.5k (recommended — best vanilla, answers "can Vanilla FLAC be inverted" at its best) vs released FLAC_EMA vs exp11_VANL@40k (defer to cross-arm exp for matched-step rows). Smoke uses released FLAC_EMA either way.
2. **Scorer**: AGREE_AR.pt primary + AGREE_fullAR.pt labelled diagnostic (recommended) vs fullAR primary.
3. **Seeds**: 42 + {43,44} robustness (recommended) vs single seed.
4. **K=8** per candidate (recommended; K′ sensitivity free offline) — confirm.
5. **Dataset onto this box**: rsync from the other machine vs fresh download — and confirm the rsync manifest: dataset + chosen FLAC ckpt(s) + (if not HF-downloaded) AGREE ckpts.

## 9. Compute budget
Basis: exp_01 full unseen eval ≈ 6.5 min (batch 64, A6000). Localization ≈ 10× conditioner + 80× 1-step DiT/VAE + AGREE embeds per query ⇒ est. 1–4 h per full-split run; R1 dev similar; R3 oracle minutes (no generation). Timing probe at R0 gates the launch; if projection exceeds ~12 h, options return to Yixun before R1.

## 10. Deliverables
`loc_invert_params_set_up.md`, `loc_invert_command.md`, per-run logs, `loc_invert_results.md`, `loc_invert_analysis.md`, `loc_invert_01_results.html` + assets (incl. heatmap gallery), `commits_loc_invert.md`. Cross-arm comparison table extension is OUT OF SCOPE for exp_18 (later experiment owns it).
