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
