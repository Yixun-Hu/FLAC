# exp_19 r1 Coder report (Opus 5 subagent, 2026-08-19 ~19:25 EDT) — archived verbatim by the Planner

All 12 cycles landed. Final state: **190 RAF tests, all green**; working tree clean for every file owned.

| # | SHA | Deliverable | Tests added (file total) | Result |
|---|---|---|---|---|
| 1 | `8ac6fad` | `data/RAF/raf_common.py`: `parse_tx_line`, `parse_rx_line`, `canonicalize_quat` | 27 (`test_raf_common.py`) | pass |
| 2 | `250cf03` | `equirect_directions`, `stable_context_seed` | +15 (42) | pass |
| 3 | `7802290` | `RAF_TO_PIPELINE`, `farthest_point_selection` | +16 (58) | pass |
| 4 | `c773bda` | `prepare_data.load_room_index` + `crosscheck_captures` | 13 (`test_raf_prepare_data.py`) | pass |
| 5 | `40e0946` | `group_captures` + placement clustering | +12 (25) | pass |
| 6 | `6f0c511` | `select_splits`, `assemble_split_jsons`, `write_split_files`, `build_splits_record` | +16 (41) | pass |
| 7 | `4665018` | `resample_and_write` + audit, `build_runtime_metadata`, CLI `main` | +15 (56) | pass |
| 8 | `59495a5` | `render_depth.py`: `render_depth`, `depth_qa`, `load_mesh_pipeline`, CLI | 18 (`test_raf_render_depth.py`) | pass |
| 9 | `e2fba7c` | `RAF_md.py` scene/poses/depth + caches | 15 (`test_raf_md.py`) | pass |
| 10 | `b8ae279` | context draws + provenance + collation contract | +15 (30) | pass |
| 11 | `0a3e600` | `FLAC_RAF_finetune.json` + 3 dataset JSONs | 7 (`test_raf_configs.py`) | pass |
| 12 | `6c1432e` | `metric_callback.py` + `RT60.py` RAF policy + FD guard | 21 (`test_raf_metrics.py`) | pass |
| — | `92bb9d3` | ledger row for cycle 12 | — | — |

`pytest src/tests/test_raf_*.py -q` tail: `190 passed, 2 warnings in 17.19s`.
Directly-touched existing suites re-run individually, all green: test_cond_dispatch 8, test_yaw_random_eval 112, test_are_lambda_config 82, test_eval_paths 42, test_exp14_fixed_mode_snapshot 32, test_frame_avg_cap_config 81, test_loc_agree_embed 46 (read-only run).

## DEVIATIONS / QUESTIONS (verbatim; Planner dispositions in impl-contracts Amendment 1)

1. **BLOCKER — the `all_rx` off-by-one is a `nan,nan,nan` sentinel.** EmptyRoom 47484 tx / 47485 rx / 47484 dirs; FurnishedRoom 39132 / 39133 / 39132; the extra rx line is literally `nan,nan,nan`. Contract-strict `load_room_index` aborts on the real corpus; no escape hatch added. → Planner: registered narrow rule, cycle 13.
2. `--allow-nonuniform` will be required for FurnishedRoom (1085×36 + one group of 72; confirmed real).
3. Contract C "y_p" read as the height (third pipeline) component under `RAF_TO_PIPELINE=(X,Z,Y)`. → registered.
4. Exact equirect round-trip attainable only at unit depth (float non-associativity, ~23% pixels 1 ULP off in general); tests assert bit-exact at depth==1 + rtol=1e-6/atol=0 general; `equirect_directions` in torch (numpy trig differs by ULP).
5. Val groups get a 12-mic FPS support pool (else val items have no context); val groups contribute all 36 as val items, pool = context only; recorded as `support_ids`. → registered.
6. `group_key` = 16-hex sha256 of the canonical 7-tuple (filename-safe); tuple stored alongside; distinct-tuple/same-key collision aborts.
7. `groups_metadata.json`/`poses_metadata.json` cover selected groups only; reserve groups live in `raf_splits_record.json` (avoids rendering 2,638 reserve maps + 47k-entry per-worker JSON loads).
8. Metadata root = `<runtime dataset>/<Room>/metadata/` (per-room, RAF-specific).
9. Added CLI flags: `--split-dir` (tests never write into the repo), `--crosscheck-sample`, `--allow-nonuniform`, renderer `--img-h/--img-w/--floor-tol`.
10. Clipping checked on source AND resampled output (output-only was fail-open under anti-alias attenuation + loader clamp).
11. `depth_qa["passed"]` excludes the floor check (occluder under speaker plausible in FurnishedRoom): `floor_ok=False` + warning, not abort; misses/non-finite/shape still abort.
12. `canonicalize_quat` raises on all-zero quaternion.
13. Contract D's collation rationale for int64 ids doesn't literally hold (collation passes metadata dicts through; pin_memory skips str) — implemented as specified anyway.
14. **Latent landmine (pre-existing, untouched):** `src/metrics/modules/l1_stft_multires.py:41` hardcodes `torch.hann_window(n_fft).cuda()` — CPU eval of that metric cannot work; RAF registration tested without exercising `update` on CPU.
15. TDD ordering nit: RAF_md written as one file in cycle 9; cycle 10 added assertions (splitting mid-function across commits would commit an incoherent module).
16. Peer commit `9627449` (non-path-scoped) swept `test_raf_md.py`'s cycle-10 additions; no content lost; peer asked to path-scope.
17. Ledger rows lag one commit by construction (no amend policy).

No cycle incomplete. Fences respected (no exp_18/localization/peer files, no installs, no GPU, no network, synthetic fixtures only).
