# loc_invert_impl_contracts — authoritative per-function contracts for the Coder

**Precedence: Rev 3 deltas OVERRIDE Rev 2 tables where they conflict** (e.g. candidate_metadata is shallow-copy per O19, aggregate uses torch.logsumexp per O18). Assembled 2026-08-19 from plan Rev 2 (`e71df84`) §4 + plan Rev 3 (`f404cde`) §4.

## PART A — Rev 2 §4 (base contracts)

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


## PART B — Rev 3 §4 (binding deltas)

## 4. Implementation plan (per file, per-function tests)

As Rev 2 §4 with these deltas (full contracts live in the Rev 2 tables, which remain binding):
- `candidates.py`: + `assert_gt_matches_loader(cand_set, md)` (O6; test: exact match passes, 1e-6 perturbation aborts); `candidate_metadata` becomes shallow-copy-with-key-swap (O19; base-intact test). `enumerate_metadata_sources` remains the candidate authority.
- `scoring.py`: `aggregate` via `torch.logsumexp` + τ=0.02 stability test (O18); + `context_conditioned_baseline` handles the empty-eligible/GT-only case explicitly (LRH_idx_30 test); + `nearest_context_baseline(cand_xyz, ctx_xyz, ctx_sims)` (O10; hand-example test); + `paired_room_clustered_test(records_a, records_b)` (O12; synthetic test); + `power_statistic(sims)` (§2.8.2; test).
- `agree_embed.py`: `embed_rirs(..., readout={'mean','sample'})`, mean = registered; tests: determinism, global-RNG-state unchanged (O2), stub stdev→0 equality, batch invariance, preprocessing-tensor equality vs real callback route (C6), pads-only-never-crops edge (O18).
- `eval_localization.py`: + `--context-k` passthrough by dataset-config choice only (existing `_1/_4` configs); + pinned `--batch-size --num-workers` recorded in provenance (O8); + smoke query ids pinned to SEEN rooms (O16); + constant-source control mode `--control constant_source` (§2.8.1); + serialization precision round-trip test (O18); everything else per Rev 2 (audit, noise bank, layout, parity harness, gt_rir mode, smoke guard).
- `loc_invert_heatmaps.py`: as Rev 2 + optional depth-silhouette helper (pure-function test on synthetic point cloud).

