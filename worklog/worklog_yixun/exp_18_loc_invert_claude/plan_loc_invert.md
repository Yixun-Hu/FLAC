# plan_loc_invert — exp_18: Inverting Vanilla FLAC for Source Localization (preflight)

**Author:** Claude Fable 5 (Planner seat). **Rev 2, 2026-08-18** — folds all 11 findings of the Codex plan review (`loc_invert_codex_plan_review.md`, REQUEST-CHANGES on Rev 1 `4f3658e`; disposition in `loc_invert_worklog.md`). Status: awaiting the supplementary Opus fallback review delta, then **Yixun approval before any implementation**.

## 1. Objective & research question

Test whether a frozen, pretrained Vanilla FLAC contains enough source-position information to localize the source of a held-out RIR in an **unseen room**, purely by analysis-by-synthesis inversion (no localization training). exp_18 also delivers the protocol + code that the later cross-arm experiment (FA B-F, yaw-aug, cyl-DINOv3 FLAC) will reuse unchanged.

**Registered success criterion (Codex C1):** the model must beat the **information-matched (context-conditioned) random baseline** (§2.6), not merely uniform-over-C — the context's target-exclusion structure alone already narrows the plausible candidate set.

## 2. Protocol (registered)

### 2.1 Queries & identity audit
- Split: the **full** existing unseen config `src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json` → `data/AR/unseen_eval.json`, 6,337 items / 17 rooms (announcement 01). Each item = one query: h_obs = h(S_gt, R), receiver known, source hidden.
- Context: the standard pipeline (8 references from OTHER source nodes at the SAME receiver). Target-source exclusion holds by construction; the driver asserts it per query via the `sample_context_ids` fingerprint (positions rendered at 6 decimals; GT source position must be absent) and hard-fails on violation.
- **Identity audit is fail-closed (Codex C2):** before any GPU generation, the driver enumerates the split and verifies each yielded item's `sample_target_id` (idx + relpath) against the expected enumeration — the FIRST mismatch (a `SampleDataset` silent substitution) aborts the run; precedent: `eval_FLAC.verify_stream_positions`. A headline artifact is written only after proving exactly 6,337 identities, no duplicates/omissions, 17 physical rooms; the split hash (sha256 over the ordered identity list) is recorded in the summary record. Smoke runs may *diagnose* substitutions; full runs must be repaired and rerun.
- `pl.seed_everything(seed)` fixes context draws; `shuffle=False`.

### 2.2 Candidate set (metadata-defined; Codex C7)
- C = the room's **valid source set enumerated from `metadata/` pair JSONs** (unique source ids with a `src_loc`), NOT from RIR filenames. GT included; expected M ≈ 10, logged per query. Node ids parsed numerically, naming-tolerant (`S010`/`S0010` quirk); `src_loc` uniqueness + cross-receiver consistency asserted (tol 1e-6).
- Data-readback rung cross-checks the metadata source set against the RIR files present and fails on unexplained differences. Missing measured RIRs may shrink only the *oracle's* explicitly-reported eligibility set — never FLAC's candidate set or the headline denominator.
- Per-candidate conditioning: `source`/`source_vit` = candidate position in the receiver camera frame (same transform as `AR_md.get_3d_point_camera_coord`, parity-tested); `depth`, `context_*` shared across the query's candidates.

### 2.3 Generation
- Frozen FLAC, EMA weights, rectified flow, `steps=1`, `cfg_scale=1.0`. Protocol flags pinned and recorded everywhere per announcement 05 (Codex C5): `--cond-method vanilla --rotate-deg 0 --cond-autocast default`, frame-avg angles recorded `n/a`. The driver **fails closed** on `--rotate-deg != 0` and on unsupported checkpoint modes (`are_lambda != 0` declared, non-`rectified_flow` objective) (Codex C8).
- K = 8 samples per candidate. **Deterministic noise bank (Codex C10):** noise for sample k of a query is keyed `(seed, query_id, k)` and **shared across that query's candidates** (common random numbers — candidate ranking is then execution-layout-independent and lower-variance). Tests: candidate-permutation invariance and one-batch-vs-two-batch equality of s_{m,k}; resume reproduces recorded context fingerprints and noise keys.
- Conditioning computed once per query over the M candidate-metadata dicts; sampling tiles it K×. Numerical faithfulness to `eval_FLAC` is not assumed but **tested**: a one-query parity test runs the same ckpt/metadata/noise through both paths and asserts identical generated waveforms (§4.5).

### 2.4 Scoring (protocol-parity with the established AR retrieval path; Codex C6)
- E_a = AGREE audio branch. Preprocessing registered as the *established callback path*: decoded RIRs clamped to [-1,1], truncated to the **first 8,000 samples** (`AcousticMetricsCallback.max_len` for AR), then zero-padded to 10,240 inside `encode_audio` handling — verified by an **embedding-equality test** against the actual `update_metrics→Retrieval.compute_audio_features` route. h_obs takes the identical route.
- AGREE loaded `.to(device).eval().requires_grad_(False)`, scored under `torch.inference_mode()`; batch-size-invariance test; checkpoint sha256 recorded.
- s_{m,k} = cos(E_a(h_obs), E_a(ĥ_{m,k})); S_m = τ·log((1/K)Σ_k exp(s_{m,k}/τ)); prediction = argmax (lowest-index tie-break). All s_{m,k} logged.

### 2.5 Hyperparameter registration (Codex C4)
- **LME with K=8 is the registered method** (per Yixun's spec) — dev selection may NOT replace it. The dev run (full seen split, `acousticroom_seeneval.json`) selects **τ only**, from the pre-registered grid {0.02, 0.05, 0.1, 0.2, 0.5} minimizing dev pooled-median e_loc; deterministic tie-break = smallest τ. Registered before the unseen run in `_params_set_up.md`.
- mean, max, and K′ ∈ {1,2,4} are computed offline from the logged s_{m,k} and reported **only as labelled sensitivity analyses**.

### 2.6 Metrics & baselines (Codex C1, C3)
- **Primary: pooled median e_loc over all 6,337 query errors** (matches the spec). Also pooled mean, success@0.5 m, success@1.0 m (err ≤ r).
- Room identity for reporting = **physical room `scene_name/scene_id`** (17 rooms; `md['scene']` alone is the 10-way scene *type* and is not used for grouping). Labelled secondaries: equal-room macro stats (incl. mean of per-room medians), top-1 candidate accuracy, MRR.
- **Baselines, identically weighted to the primary:**
  - *Uniform-over-C* (the spec's literal baseline): exact per-query — E[e_loc] = mean candidate distance, success@r = fraction within r, pooled median from the per-candidate distance distribution (weight 1/M per query).
  - *Context-conditioned (information-matched, REGISTERED comparison target)*: uniform over candidates whose source is NOT among the query's 8 context sources — exact, same formulas over the eligible set. Report the eligible-set-size distribution, each candidate's context-membership in the JSONL, and how often FLAC predicts a context-member candidate.
- **Uncertainty:** metrics per seed; seed mean ± SD reported as run-to-run variability, SEPARATELY from a 17-room clustered bootstrap CI (resample rooms, 10,000 draws) on the primary — the SD over 3 seeds is never presented as a CI.
- **GT-RIR oracle (upper diagnostic):** same scorer on measured candidate RIRs at the query receiver. The same-pair candidate IS h_obs ⇒ identity-oracle is a pipeline sanity check (~100% top-1) and labelled as such; a non-trivial oracle variant uses a second measurement of the same pair (e.g. another `single_channel_ir_*` channel) **iff** data readback confirms one exists; its (possibly reduced) eligibility set is reported explicitly.

### 2.7 Heatmaps (Codex C11)
Top-down per-query maps: candidates colored by softmax(S/T_disp), receiver, GT, prediction. **T_disp pre-registered = the registered τ.** The shaded region is labelled *candidate extent* (not a room boundary; a true boundary only if derivable from metadata). Case gallery by pre-registered rule (sharp-success = correct top-1, max top-2 margin; ambiguous = min margin; failure = max e_loc), 3 per category, rule-selected — no hand-picking.

## 3. Checkpoints & assets (open decisions §8)
As Rev 1: smoke may use released HF `FLAC_EMA.ckpt`; registered run on the program vanilla ckpt (recommended exp07_P1 anchor 87.5k). Scorer: recommended `AGREE_AR.pt` primary (train-split-only), `AGREE_fullAR.pt` labelled leaky diagnostic. Dataset must land on this box.

## 4. Implementation plan (per file, per-function tests — announcement 02)

Additive only; no edits to release files. Tests in `src/tests/`. (Tables give the contract; tests named test_loc_*.)

### 4.1 `src/localization/__init__.py` — package marker.

### 4.2 `src/localization/candidates.py` (`src/tests/test_loc_candidates.py`)
| Function | Contract / tests |
|---|---|
| `parse_ir_filename(name)` | numeric (src, rec) from `S…_R…_hybrid_IR.wav`; tests: std, `S010`, malformed→ValueError |
| `enumerate_metadata_sources(meta_room_dir) -> dict[node, xyz]` | **candidate authority (C7)**: unique sources with `src_loc` from pair JSONs, naming-tolerant; tests: fixture tree incl. `S010`-style, uniqueness, cross-receiver consistency assert, inconsistent→error |
| `find_pair_metadata(meta_room_dir, src, rec)` | naming-tolerant pair lookup; tests: both namings, missing→None |
| `build_candidate_set(ir_path, metadata_path) -> CandidateSet` | dataclass(nodes, xyz_world, rec_loc, gt_node, gt_xyz, context-membership filled later); GT∈C asserted; deterministic order; tests: fixture room, GT missing from metadata→error |
| `project_to_camera(rec_loc, xyz)` | translation-only; **parity test** vs importlib-loaded `AR_md.get_3d_point_camera_coord` on random points |
| `candidate_metadata(base_md, cand_cam_xyz)` | deepcopy; only `source` ([3] f32) and `source_vit` ([1,3] f32) replaced; tests: key-diff exactness, dtypes, base not mutated |
| `crosscheck_sources_vs_files(meta_nodes, room_dir)` | readback-rung check; tests: match, extra-file, missing-file cases |

### 4.3 `src/localization/scoring.py` (pure; `src/tests/test_loc_scoring.py`)
| Function | Contract / tests |
|---|---|
| `cosine_sims(obs [D], gen [M,K,D])` | norm-guard (≈1) raises; hand-built vectors |
| `aggregate(sims, method, tau)` | lme/mean/max; lme→max (τ→0⁺), lme→mean (τ large), K=1 equality, τ≤0 lme→ValueError |
| `predict_index(scores)` | argmax, lowest-index tie-break (test) |
| `softmax_map(scores, T)` | sums to 1; shift-invariance; T>0 |
| `localization_error`, `success_within` | L2; boundary err ≤ r (test) |
| `uniform_baseline(cand_xyz, gt_xyz)` | exact mean/success/pooled-distances; hand example + seeded-MC agreement 1e-3 |
| `context_conditioned_baseline(cand_xyz, gt_xyz, context_member_mask)` | **(C1)** exact over non-context candidates; eligible-size returned; tests: hand example, all-but-one case, GT-always-eligible invariant |
| `noise_key(seed, query_id, k) -> generator seed` | **(C10)** stable hash, collision test over sample grid, k/query independence |
| `summarize(records, radii)` | pooled primary + per-room(17) macro secondaries + top-1/MRR; identical weighting applied to both baselines; synthetic set with known answers |
| `clustered_bootstrap_ci(records, by='room_id', n=10000, seed)` | **(C3)** resample rooms; tests: degenerate 1-room, reproducibility w/ seed |

### 4.4 `src/localization/agree_embed.py` (`src/tests/test_loc_agree_embed.py`)
| Function | Contract / tests |
|---|---|
| `load_agree_audio(ckpt, device)` | wraps `loading_AGREE_model`; returns model `.eval()`, all `requires_grad=False` (asserted in test w/ stub); integration test skipif no ckpt |
| `preprocess_for_scoring(wav [B,1,T])` | clamp[-1,1] → first-8000 → shape contract; **embedding-equality test vs the real `update_metrics→Retrieval` route** (C6; integration, skipif no ckpt) |
| `embed_rirs(model, wavs, device)` | inference_mode; L2-normalized f32 [B,D]; batch-size-invariance test (B=1 vs B=8, stub + integration) |

### 4.5 `eval_localization.py` (root driver; `src/tests/test_eval_localization.py`)
Flags: `--model-config --dataset-config --ckpt-path --agree-ckpt --num-samples --tau --agg --steps --cfg-scale --seed --cond-method --frame-avg-angles --rotate-deg --cond-autocast --score-source {flac,gt_rir} --out-dir --eval-name --smoke --max-queries` (`--max-queries` refuses without `--smoke`; smoke stamped in filename+record).
**Reuse boundary (explicit, C8):** imports from `eval_FLAC`: `sample_target_id`, `sample_context_ids`, `source_sha`, `orbit_provenance`, `resolve_cond_autocast`, `check_load_integrity`, `resolve_are_from_checkpoint` (to fail closed on ARE ckpts); model build/EMA-remap follows `evaluate_model` lines-of-record and is covered by the parity test below — any divergence is a bug, not a protocol variant.
| Unit | Contract / tests |
|---|---|
| `audit_split_identities(dl, expected)` | **(C2)** fail-closed on first mismatch; returns split hash; tests: clean pass, injected substitution→SystemExit, hash stability |
| `build_noise_bank(seed, query_id, K, shape)` | keyed per §2.3; permutation + batch-split equivalence tests |
| `run_query(...)` | M×K layout deterministic (candidate-major); layout test with stub sampler |
| JSONL row schema | round-trip test; includes context-membership mask, noise keys, per-sample sims, protocol fields |
| summary record | aggregation equals `scoring.summarize` on the same rows (test); provenance: git sha, ckpt sha256, agree sha256, split hash, all §2.3 flags |
| `--score-source gt_rir` | uses measured files; marks identity candidate; eligibility-set reporting; unit test with fixture wavs |
| smoke guard | refusal test |
| **Parity harness** `parity_check_one_query(...)` | same ckpt/md/noise through driver path vs `eval_FLAC`-style reference path ⇒ identical waveforms (atol 0); integration, skipif no ckpt/data; ALSO run at rung 5 on the real smoke query |

### 4.6 `worklog/…/loc_invert_heatmaps.py` (reviewed in a consolidated round; `src/tests/test_loc_heatmaps.py` for the pure parts)
| Unit | Contract / tests |
|---|---|
| `select_cases(records, rule, n_per_cat)` | pre-registered rule of §2.7, deterministic; synthetic-records test |
| `room_extent(records)` | candidate-extent bbox, honest labelling; test |
| `render_heatmap(record, T_disp, out)` | writes PNG; smoke-tested on synthetic record (mpl Agg) |

### 4.7 Explicitly NOT touched
`train.py`, `defaults.ini`, `src/data/`, `src/models/`, `src/training/`, `data/AR/`, `AR_md.py`, release eval scripts.

## 5. Run matrix
| Run | Split | Ckpt | Mode | Purpose |
|---|---|---|---|---|
| R0 probe | pre-registered query ids from 2 rooms, `--smoke --max-queries 4` | released FLAC_EMA (or first rsynced) | flac, K=8, max-M batch | end-to-end pipe + **fit & timing probe (C9): peak memory + per-component times (cond / sample / decode / embed) at batch 64** |
| R1 dev-tune | full seen split | registered vanilla ckpt | flac, K=8 | τ selection per §2.5 |
| R2 registered | full unseen 6,337 | registered vanilla ckpt | flac, LME/K=8/τ* | headline; seeds 42/43/44, **one seed per GPU concurrently** (no sharding; if a sweep must shard, a merge gate proves disjoint-union = 6,337 identities) |
| R3 oracle | full unseen | — | gt_rir | identity sanity + 2nd-channel variant iff available |
| R4 baselines | full unseen | — | analytic | uniform + context-conditioned (CPU) |

## 6. Validation ladder
1. Static (`py_compile`, `git diff --check`). 2. Unit tests (§4, fixtures, no GPU/data). 3. Tiny synthetic forward (random-init model from config, 1 query × 2 cand × K=2). 4. Real-data readback (1 room): metadata naming (`S010`), source enumeration vs files, `src_loc` consistency, context fingerprint excludes GT, second-channel existence for the oracle. 5. R0 smoke incl. parity harness on a real query + identity-oracle 100% on smoke queries. 6. **Fit/timing probe = R0's measurement half — NOT waived (C9)**; full-run budget projected from it; if projection > 12 h/sweep, options go back to Yixun before R1. 7. Full runs per §5 with pre-launch acceptance criteria in `_worklog.md`.

## 7. Integrity controls (delta over Rev 1)
Fail-closed identity audit (no exclusions in headline); metadata-defined candidates; information-matched baseline as the registered comparison; LME/K=8 fixed, τ-only dev tuning; scorer protocol = established 8,000-sample AR path with equality test; deterministic noise bank; announcement-05 flags pinned (`vanilla / rotate 0 / autocast default / fa-angles n/a`) in params, command file, every log and output row; smoke quarantined; per-seed + clustered-CI reporting.

## 8. Open decisions for Yixun (recommendation first)
1. **Registered vanilla ckpt**: exp07_P1 anchor 87.5k (recommended) vs released FLAC_EMA vs exp11_VANL@40k (defer matched-step row to cross-arm exp). Smoke uses released FLAC_EMA either way.
2. **Scorer**: `AGREE_AR.pt` primary + `AGREE_fullAR.pt` labelled diagnostic (recommended) vs fullAR primary.
3. **Seeds**: 42+43+44 (recommended; one per GPU, third follows) vs single seed.
4. **K=8** registered (recommended; K′ sensitivity offline) — confirm.
5. **Dataset onto this box**: rsync vs fresh download; confirm manifest = AcousticRooms (incl. `metadata/`, `single_channel_ir_1/`, any `single_channel_ir_2+` for the oracle variant) + chosen FLAC ckpt(s) + AGREE ckpts (HF-downloadable otherwise).

## 9. Compute budget (post-C9 honesty)
Naive scaling of exp_01's 6.5 min basis by generated-sample count (×80) bounds a full sweep at ~8.7–13.7 h; conditioner amortization (shared context per query) should cut this substantially but is **unproven until R0 measures it**. Budget and the seed recommendation are finalized from R0's probe; >12 h/sweep projection returns to Yixun before R1.

## 10. Deliverables
As Rev 1 (params, command, logs, results, analysis, HTML+assets incl. rule-selected heatmap gallery, commit ledger). Cross-arm table remains OUT OF SCOPE for exp_18.
