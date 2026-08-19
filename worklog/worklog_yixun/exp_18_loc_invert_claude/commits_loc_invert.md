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
