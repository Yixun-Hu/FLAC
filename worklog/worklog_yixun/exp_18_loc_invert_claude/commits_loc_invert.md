# commits_loc_invert — exp_18 commit ledger (branch `localization-exp`, base `6170007`)

| SHA | Description |
|---|---|
| `4f3658e` | exp_18 scaffold: query (verbatim + summary), recon worklog (incl. Codex-401 fallback entry appended after), plan draft (pre-review) |
| `09dfec6` | commit ledger + Codex-401 fallback worklog entry |
| `e71df84` | Codex plan review saved (REQUEST-CHANGES, 11 findings) + plan Rev 2 folding all findings + disposition worklog entry |
| `20586ad` | Opus supplementary review saved + plan Rev 3 folding O1–O23 + disposition worklog entry |

## Round 1 (Coder: Claude Opus 5) — `src/localization/{candidates,scoring}.py` + tests, TDD cycles

| SHA | Description | changed lines |
|---|---|---|
| `64fa6be` | r1 cycle 1: `parse_ir_filename` + `src/localization` package marker | +79 |
| `9bfa509` | r1 cycle 2: `find_pair_metadata` + `enumerate_metadata_sources` (candidate authority, C7) | +200 −1 |
| `80ad3aa` | r1 cycle 3: `project_to_camera` + bit-exact parity test vs importlib-loaded `AR_md` | +69 |
| `45bad42` | r1 cycle 4: `CandidateSet` dataclass + `build_candidate_set` | +156 |
| `c3537d1` | r1 cycle 5: `candidate_metadata` (shallow copy, O19), `assert_gt_matches_loader` (O6), `crosscheck_sources_vs_files` | +204 |
| `88f8989` | r1 cycle 6: `cosine_sims` (norm guard) + `aggregate` via `torch.logsumexp` (τ=0.02 stability, O18) | +202 |
| `43d1401` | r1 cycle 7: `predict_index`, `softmax_map`, `localization_error`, `success_within` | +113 −1 |
| `abd5021` | r1 cycle 8: `uniform_baseline`, `context_conditioned_baseline` (C1 + GT-only edge case), `nearest_context_baseline` (O10) | +212 |
| `d9cba5d` | r1 cycle 9: `noise_key` (sha256 canonical payload, C10) + `power_statistic` (§2.8.2) | +130 |
| `69f52e0` | r1 cycle 10: `summarize` — pooled-median primary + labelled per-room/macro secondaries | +227 |
| `e8b6d49` | r1 cycle 11: `clustered_bootstrap_ci` (C3) + `paired_room_clustered_test` (O12) | +248 |

### Round 1 fix batch (Codex review `loc_invert_codex_code_r1_review.md`, REQUEST-CHANGES)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `c09b7e8` | F1 (HIGH, finding 1) | finiteness guards in `scoring.py` — NaN/Inf rejected at every numerical entry point (`_require_finite` / `_finite_scalar` / `_check_radii`) | +150 −13 |
| `089ab2f` | F1 (HIGH, finding 1) | finiteness guards in `candidates.py` — JSON `NaN`/`Infinity` coordinates, `_xyz`, `xyz_world`, `candidate_metadata` torch path | +75 −2 |
| `cf4c0bc` | F2 (MEDIUM, finding 2) | cross-node `src_loc` uniqueness enforced in `enumerate_metadata_sources` (the candidate authority, plan §2.2) at `SRC_LOC_TOL` | +53 −3 |
| `5693876` | F3 (MEDIUM, finding 3) | explicit `method="linear"` percentile + golden CI-endpoint and two-sided p-value fixtures (mutation-verified) | +70 −2 |
| `2ecc62f` | F4 (NIT, finding 4) | Monte-Carlo agreement now genuinely holds at 1e-3 for all three metrics (chunked 6.4e7 draws, ~5σ) | +19 −6 |
| `1cb5d92` | F5 (Planner ruling 4b) | optional `eligible_mask` on `nearest_context_baseline` (raw behaviour unchanged at `None`; all-False / length mismatch refused) | +65 −2 |
| `c33617e` | — | ledger append for the r1 fix batch | +14 |

## Round 2 (Coder: Claude Opus 5) — `src/localization/agree_embed.py` + tests

| SHA | Description | changed lines |
|---|---|---|
| `56d321a` | r2 cycle 1: `preprocess_for_scoring` — clamp → `max_len`=8000 slice → pad to 10240, pinned by a C6 composition test against the cited release expressions | +152 |
| `a664041` | r2 cycle 2: `embed_rirs` — registered deterministic VAE-mean readout (+ `sample` diagnostic); stub wired to the REAL `VAEBottleneck` | +155 −1 |
| `e16ac40` | r2 cycle 3: `load_agree_audio` — reuses `loading_AGREE_model`, freezes + asserts eval/no-grad, CWD guard for the CWD-relative `VAE.ckpt`, `LoadedAgree` with ckpt sha256 | +123 −1 |
| `836f16a` | r2 cycle 4: integration tests on the real AGREE scorer (skipif on assets; frozen/eval, deterministic unit-norm [2,512], sampled path stochastic, RNG isolation) | +71 |
| `4a4ff7e` | r2 cycle 5: explicit O18 pads-only-never-crops test over T ∈ {1 … 20000} | +15 |
| `ec496f3` | ledger append for round 2 | +12 |

### Round 2 fix batch (Codex review `loc_invert_codex_code_r2_review.md`, APPROVE-WITH-CHANGES)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `4d8f11c` | F3 (MED #3) + F4 (NIT #4) | `config_name` removed — the guard validates the config the reused loader builds (tied to its source), `os.path.isfile` for relative **and** absolute `pretrained`; eval-mode refusal walks `named_modules` so a train-mode child is refused | +75 −14 |
| `58054d7` | F5 (NIT #5) | parity test now *traverses* the dependency route (real `AcousticMetricsCallback.max_len` + real `Retrieval.update` → `compute_audio_features` with a recording fake AGREE); mutation-verified | +37 |
| `50b3f66` | F1 (MED #1) | integration asset detection anchored at the repo root via `__file__`; module-scoped `repo_root_cwd` fixture; regression test that present assets never skip | +37 −6 |
| `22bf0ad` | F2 (MED #2) | real-model B=8 vs 8×B=1 batch invariance (atol 1e-6) + CUDA-conditional CPU&CUDA RNG-isolation test with a sampled-path companion | +56 |
| `7335248` | — | ledger append for the r2 fix batch | +19 |

## Round 3 (Coder: Claude Opus 5) — `eval_localization.py` driver + tests

| SHA | Unit | Description | changed lines |
|---|---|---|---|
| `7af1979` | a | fail-closed split identity audit (`expected_split_identities`, `split_hash`, `audit_split_identities` → SystemExit on first mismatch) | +197 |
| `210fbeb` | b | `build_noise_bank` — per-`(seed, query_id, k)` generators via `scoring.noise_key`, global RNG untouched | +83 |
| `e682216` | c | `run_query` + `Engine` seam — candidate-major `m*K+k` layout, one conditioner call, O6 invariant, `constant_source` control | +271 −1 |
| `667a017` | d | exact `float.hex` sims serialization, `room_id_from_relpath`, context-membership by eval_FLAC's fingerprint rule, `gt_reciprocal_rank` | +122 −1 |
| `52d5a81` | d | `build_row` schema + `write_row`/`read_rows` (flush per row, bitwise round trip) | +191 −1 |
| `221a195` | e | `summarize_run` — FLAC + both baselines + O10 control (raw & masked), gt_only handling, eligible-set stats | +225 −2 |
| `ab326e4` | e | `build_provenance`, `output_paths` (K/seed/smoke stamped), `write_summary` + `jsonable` | +179 −1 |
| `cbd7f2b` | f | `gt_rir` measured-RIR oracle (`measured_rir_paths`, `load_measured_rirs`, `run_query_gt_rir`) | +152 −1 |
| `1cf834e` | h | CLI `parse_args` + `validate_args` + `assert_rectified_flow` / `assert_no_are` startup refusals | +159 −2 |
| `56a5848` | h | `prepare_state_dict` — evaluate_model's EMA lines of record, fixtures shaped like the real ckpts | +89 −1 |
| `6bcbc8b` | g | `build_engine` + `parity_check_one_query`; real-ckpt integration parity **match=True, diff 0.0** | +235 −2 |
| `afb8fc3` | h | per-query wiring (`dataset_folder_from_md`, `query_candidate_set`, `context_evidence`) | +130 −4 |
| `b33f3ed` | h | `process_query` + `run_evaluation` (audit-first, JSONL streaming, smoke truncation) | +216 −4 |
| `12b8ecc` | h | `main()` + `scoring_only_engine` + `--parity-check` | +98 −1 |
| `42cf879` | — | ledger append for round 3 | +22 |

### Round 3 fix batch (Codex review `loc_invert_codex_code_r3_review.md`, REQUEST-CHANGES)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `156c714` | F3 (HIGH #3) | `constant_source` no longer overwrites candidate geometry — immutable `candidate_positions` vs substituted `conditioning_positions` | +61 −8 |
| `3ead435` | F7 (MED #7) | fail-closed context membership (fingerprint→index map, collisions, unresolvable ids, GT-in-context) resolved BEFORE generation | +104 −11 |
| `4e3b4a5` | F9 (MED #9) | finite/domain guards on all numeric flags + `load_and_validate_artifacts` on CPU before the scorer or generator is built | +122 −11 |
| `bdaeb04` | F8 (MED #8) | gt_rir: duplicate `(src, rec)` matches raise, rank over available only, identity must exist, mode refuses ckpt/control/parity, `gt_rir_K1` stamping | +115 −13 |
| `f11f137` | F5 (HIGH #5) | O16 — `--smoke`/`--parity-check` require a SEEN split, checked from the dataset-config CONTENT | +69 |
| `e9de5c0` + `4d3369e` + `380dfe3` | F1 (BLOCKER #1) | identity TOCTOU closed: split-JSON-derived expectation, in-loop pre-generation check, end gate on count+rooms, hash over the SCORED stream, `.partial` atomic publish; + `main()` refusal-order follow-ups | +227 −36, +1, +4 −4 |
| `90a62da` | F6 (HIGH #6) | provenance: registration SHA (O17), config content hashes, context K, loader semantics, context-stream digest (O8), device/precision/versions/TF32/flash_attn | +161 −6 |
| `a39c9b3` | F4 (HIGH #4) | summary: `flac_excl_gt_only` + masked control on the same retained subset, clustered CI, paired room-clustered tests, power statistic per row + aggregate | +128 −4 |
| `da8a284` | F2 (HIGH #2) | M×K parity — always-running synthetic (full conditioning, bitwise) + real-asset seen-room test (bitwise at `--cond-autocast off`; batch split bitwise at the registered default) | +285 −1 |
| `06588eb` | — | `__main__` guard defect found by the CLI parity re-run + `--num-workers >= 1` guard | +26 −6 |
