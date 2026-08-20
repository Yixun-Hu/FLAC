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
| `92bb9d3` | r1 ledger catch-up row for cycle 12 |
| `1d1af86` | r1 TDD cycle 13: `all_rx` trailing all-NaN sentinel rule (Amendment 1 D1) — accepted once, recorded in `RoomIndex` + splits record; every other mismatch/NaN still aborts (70 tests) |
| `6c0a16e` | r2 fix R3: capture-id stream fingerprints in `eval_FLAC.py` (AR/HAA byte-identical; per-run fingerprint_schema 1/2) (29 tests) |
| `f4d34ca` | r2 fix R1+R2+R14: FPS eligibility (exactly-36 only, forced reserve + atom uniqueness), `diagnostic` role (12 context-only supports / 24 targets, own manifest + config), val supports context-only |
| `ad966e4` | r2 fix R4+R13: `readback_audit.py` (onset/delay fit, crop-vs-full T30, quaternion readings, gauge pinning) + publish gates in prepare/render; WAV read-back, JSON-safe dB, role-split amplitude distributions |
| `6886d40` | r2 fix R5+R6+R12: canonical 256x512 float32 enforcement (+`--non-canonical` taint), loader-side depth contract, real-mesh QA (containment/bounds/sightline/bearing/scale), one scene per room + render benchmark |
| `d20c6ab` | r2 fix R7: atomic staged publish with sha256 manifest written last + failure-injection tests |
| `a08571f` | r2 fix R9: real MultiConditioner + `get_conditioning_inputs` pass over the real collated RAF batch |
| `755aed6` | r2 fix R11: structural clone-whitelist (recursive node/type/length comparison against the patched HAA template) |
| `07badce` | r2: pre-written R8/R10 tests, skipped behind a feature probe during the peer's `src/metrics` freeze |
| `c7c3cab` | r2 fix R2 (Amendment 3): diagnostic supports join training (train 408 = 2x(16x12+12)); diagnostic targets stay eval-only |
