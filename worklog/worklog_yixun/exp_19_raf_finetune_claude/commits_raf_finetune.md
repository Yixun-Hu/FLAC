# commits_raf_finetune — exp_19 commit ledger (branch `localization-exp`)

| SHA | Description |
|---|---|
| `138dc26` | exp_19 scaffold: query (verbatim + summary) + worklog (state reconciliation) |
| `1b1b118` | scoping answers + mesh download + HAA-template recon logged |
| `2746b49` | plan Rev 1 + EmptyRoom readback worklog |
| `5268a1f` | Codex plan review (REQUEST-CHANGES, C1–C16) + plan Rev 2 + disposition |
| `f8a5643` | approval recorded (plan Rev 2 + 6 decisions locked; date typo in entry corrected next commit) |
| `8ac6fad` | r1 TDD cycle 1: `raf_common` fail-closed tx/rx parsers + `canonicalize_quat` (27 tests) |
| `250cf03` | r1 TDD cycle 2: `equirect_directions` (exact inverse of the pipeline pixel->ray map) + `stable_context_seed` (42 tests) |
| `7802290` | r1 TDD cycle 3: `RAF_TO_PIPELINE` gauge constant + deterministic `farthest_point_selection` (58 tests) |
| `c773bda` | r1 TDD cycle 4: `load_room_index` (positional, fail-closed count invariant) + seeded per-capture cross-check (13 tests) |
| `40e0946` | r1 TDD cycle 5: `group_captures` (canonical 7-tuple key, exactly-36 invariant, `--allow-nonuniform`) + placement clustering (25 tests) |
| `6f0c511` | r1 TDD cycle 6: `select_splits` (group-atomic FPS split, 12/24 support/test, val support pool) + split JSONs + splits record (41 tests) |
| `4665018` | r1 TDD cycle 7: `resample_and_write` + amplitude audit, runtime metadata emission, prepare CLI (56 tests) |
| `59495a5` | r1 TDD cycle 8: `render_depth` (open3d raycast, abort-on-miss) + per-map QA + CLI; hand-derived six-wall oracle (18 tests) |
| `e2fba7c` | r1 TDD cycle 9: `RAF_md` scene/poses/depth (no flipud) + bounded per-worker JSON + depth caches (15 tests) |
| `b8ae279` | r1 TDD cycle 10: RAF_md context (train stochastic / eval deterministic), provenance tensors, collation contract (30 tests) |
| `0a3e600` | r1 TDD cycle 11: RAF model + 3 dataset configs, pinned by a HAA-template diff whitelist (7 tests) |
| `6c1432e` | r1 TDD cycle 12: RAF in `metric_callback` + `RT60Error` (9600 window, per-scene, T30) + fail-closed FD/retrieval guard + AR/HAA regressions (21 tests) |
