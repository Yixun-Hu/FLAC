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
| `5114f9e` | — | ledger append for the r3 fix batch | +25 |

## Round 4 (integrative full review `loc_invert_codex_code_full_review.md` + plan Rev 3.1)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `c3ad3c8` | R4-1 (F1) | frozen room manifest (`build_room_manifest`, `manifest_sha256`, `candidate_set_from_manifest`); per-query disk enumeration deleted; real 17-room unseen test pins Rev 3.1 §1 (M=10 everywhere, LRH S10 wav-absent) | +261 −21 |
| `03c6fe3` | R4-6 + R4-7 | resolved device block (requested/index/name/capability/UUID); finite `--frame-avg-angles` at parse time; registration validated before any model load | +85 −6 |
| `ba73723` | R4-3 (F3) | fail-closed context evidence: `resolve_context_k`, `assert_query_context` (pre-generation), `assert_context_evidence_complete` (publication gate); no leniency flag | +154 −3 |
| `af708e3` | R4-5 (F5) | cell-unique `artifact_stem` + `artifact_paths` refusing existing final/`.partial` targets unless `--overwrite`; manifest published as an artifact | +102 −35 |
| `4c0a1fc` | R4-4 (F4) | machine-checked registration: `--registration-manifest`, git commit + byte-equality verification, every locked field + seed-membership checked before model loads | +254 −15 |
| `e851c0a` | R4-2a (F2) | `--mode readback` — R-1's dataset gate (crosscheck both ways, split-vs-metadata counts, wav readback at 22050, depth presence), JSON report, nonzero exit | +207 −3 |
| `92f132a` + `ac1d1ab` | R4-2b (F2) | always-on per-query component timings + CUDA peak memory; `probe_summary` in the run summary (R0's probe = a smoke summary) | +156 −4, +3 −1 |
| `0693f59` | R4-2c (F2) | `--mode scorer-noise` — §2.8.3 sampled-readout measurement (pairwise cos + vs-mean stats), AGREE only | +192 −3 |
| `f86de94` | R4-2d (F2) | `src/localization/reaggregate.py` + `--mode reaggregate` — R1's offline τ/agg/K′ sweep and registered selection; the sims codec moves there (single definition) | +413 −20 |
| `1dd2e25` | — | `__main__` guard defect recurred; now asserted structurally on the module AST | +16 −3 |
| `01b7cce` | — | ledger append for round 4 | +18 |

## Round 5 (launch-gate review `loc_invert_codex_code_r4_review.md`)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `a69f96a` | R5-1 (H1) | wall-correct GPU timing: `_sync` on the resolved index, `_timed` with leading+trailing sync (scoring's `.cpu()` inside its interval), `context` timed, separately synchronized whole-query wall time; two-GPU discrimination test | +243 −83 |
| `fb41be1` | R5-2 (H2) | readback ENFORCES the registered R-1 invariants (17 rooms × M=10, LRH S10 the only allowed anomaly), loads every depth map (256,512)/float/finite, decodes one wav per (room, source) | +265 −39 |
| `34fee8e` | R5-3 (M3) | one atomic no-clobber `write_json_atomic` for all five writers; content-addressed auxiliary stems | +158 −35 |
| `50f5396` | R5-4 (M4) | registration requires a full-hex immutable id, in-repo manifest, byte equality AND ancestry of HEAD; resolved id recorded | +106 −22 |
| `36e77ec` | R5-5 (M5) | main(): configs → registration → K_ctx → output claim → **then** `torch.load` → AGREE → generator → dataloader; call-order spy test | +111 −30 |
| `f84c197` | R5-6 (M6) | scorer-noise wavs must belong to the configured split; draws seeded from `--seed` and the seed recorded | +116 −24 |
| `3302e2d` | — | ledger append for round 5 | +12 |

### Round 5b (launch gate v2 `loc_invert_codex_code_r5_review.md` — H2 residual)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `03b7641` | R5b-1 + R5b-2 + 3 nits | canonical unseen split pinned by byte digest / 6337 identities / 17 rooms / room-node-map digest, enforced in readback **and** run startup and locked into the registration manifest (`split_file_sha256`); readback requires wavs ≥ `MIN_WAV_SAMPLES` (10240) and reports min/max/mean length; scorer-noise stem hashes the selected wav set; `PROBE_WALL` and `write_json_atomic` docstrings clarified | +331 −20 |
| `30f26d1` | — | ledger append for round 5b | +8 |

## Round 6 (plan Rev 3.2 — duplicate-position sources merge into one candidate)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `12553e8` | r6-1 | `merge_position_duplicates` in the candidate authority (canonical = lowest id, every group recorded); `enumerate_metadata_sources(..., allow_duplicate_positions=True)` for the manifest only — cross-receiver drift still aborts | +87 −2 |
| `b46f777` | r6-2 | manifest merges positions (`merge_map` non-trivial only, `member_nodes` kept); GT resolves through its group; merged positions end the F7 collision case; rows + provenance record merges | +142 −11 |
| `f2b912f` | r6-3 | `gt_rir` prefers the canonical node's measured file and falls back to any member, recording `oracle_source_nodes`; identity bound to the GT's group | +161 −18 |
| `b504dea` | r6-4 | `survey_duplicate_sources.py` — the Planner's R0-abort probe as reviewed, tested tooling (2/131 seen rooms, 0/17 unseen) | +85 |
| `31a37cd` | — | ledger append for round 6 | +11 |

## Round 7 (announcement 08 — predicted waveforms are a required artifact)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `29ff806` | R7-1 | `--dump-waveforms DIR`: per-query `.npz` with `pred [M,K,10240]` (exactly-as-scored) + `obs [10240]`; row `waveform_path`/`waveform_sha256`; atomic `*_waveforms.json` index carrying checksums **and** geometry; self-describing README; non-empty dir refused; oracle refuses the flag | +343 −13 |
| `1ec2365` | R7-2 | `--verify-against ROWS.jsonl`: full-pipeline replay comparing every per-sample sim on exact float hex, aborting at the first differing `(m,k)`; `verify_against` summary block; `_replay` stem so the original artifacts are never touched | +182 −4 |
| `76f7072` | R7-3 + R7-4 | `--readback-decode-all` (whole-split decode + `is_silence` detection); clean-row golden schema test and "computation-identical" wording; survey reports `n_errors` and exits nonzero unless `--allow-errors` | +196 −22 |
| `7d0d740` | — | `__main__` guard restored to the module end (caught by the r4 AST test) | +4 −4 |
| `d6e531d` | — | ledger append for round 7 | +12 |

## Round R4-r1 (plan_loc_invert_R4 §1–§3 — non-AGREE metric families, exploratory)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `837ed46` | scaffold + M1/M5 | REGISTERABLE constant set (+ JSON payload), windows (9600 / 8000), shared zero-padded lag machinery via FFT correlation + prefix energies in float64, M1 = 1−max ρ², M5 = 1−max NCC with peak lag + GCC-PHAT secondary, off-grid `delta_max` refused | +490 |
| `08e747b` | M2 + M3 | repo scale set pinned by test + spectral convergence (λ=1, raw amplitudes); Schroeder EDC L1 over the observation-defined [0,−30] dB region + band/Hilbert secondaries | +267 |
| `f94ee27` | M4 + entry points | repo estimators only, per-query uniform validity mask, frozen z-norm L1; `compute_metrics` (candidates + context + diagnostics + config echo); `metric_matched_retrieval` delegating to `scoring.nearest_context_baseline` | +469 |
| `f509abd` | — | ledger append for R4-r1 | +10 |

## Round R4-r2 (driver integration of the R4 metric families)

| SHA | Item | Description | changed lines |
|---|---|---|---|
| `b1373e7` | ruling | per-query M4 dropped-feature diagnostic (count + causing candidate/context/obs) alongside the kept uniform-drop rule | +79 |
| `37054d5` | items 1 + 4 | `--metrics` on the replay path: one immutable snapshot feeds dump AND metrics with a digest guard; metrics-JSONL per query (all families, hex distances, aggregations, predictions, M4 block, lags, tail provenance); replay preflight cross-checking protocol + cardinality/uniqueness + sibling-summary provenance; M4 estimator exceptions → NaN | +485 −20 |
| `b7ecca1` | item 2 | `--metric-registration` gate reusing `verify_registration_commit`; locks the whole REGISTERABLE set + applied metric_config; seen runs record the payload without a manifest | +150 |
| `a90f9dc` | item 5 | `--mode metrics-calibrate`: seen-only Δmax selection (dev top-1, tie→smallest), M4 μ/σ freeze, draft manifest, per-feature between/within/top-1 diagnostics | +181 |
| `529ee53` | item 3 | `--mode metrics-retrieval`: metric-matched control (raw + masked), measured oracle ceiling, context/non-context split, no generation; `__main__` guard restored | +233 |
| `d2371f4` | — | ledger append for R4-r2 | +12 |

## Round r4m3 (consolidated R4 review `loc_invert_codex_code_r4m_review.md`)

| SHA | Finding | Description | changed lines |
|---|---|---|---|
| `d4f471a` | **F7 (first)** | literal r7 dump/provenance route when `--metrics` is off; byte-level golden parity against the r7 module loaded from `7d0d740` — **unblocked R2b seed 44** | +105 −13 |
| `f360b3c` | F2 + F3 | M1 evaluates the literal registered residual per lag (not `1−ρ²`); pyroomacoustics' `-1` sentinel and any negative decay time become NaN, tested against the real wrapper | +129 −13 |
| `408a093` | F4 + F1 (constants) | M2 complex-STFT, M3 band/Hilbert and M5 GCC peak similarity as first-class candidate+context families; Δ=0 always emitted; three seen sensitivities; Holm–Bonferroni; REGISTERABLE gains every missing formula constant | +286 −16 |
| `81dda12` | F1 + F4 + F8 | manifest authoritative for the whole `MetricConfig`; registration required for every unseen metric mode; calibration authenticates its identity stream; draft gains seeds/digests; secondaries + sensitivities serialized; vestigial param removed | +268 −54 |
| `a4102a5` | F5 + F6 | prediction-based context split, compact-index oracle mapping, fallback source nodes, seed in stem, paired context-digest check; all metric outputs stay `.partial` until every gate passes | +215 −34 |
