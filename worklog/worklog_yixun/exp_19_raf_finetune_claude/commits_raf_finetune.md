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
| `b4c0ac1` | r2 Amendment 4: ray-miss cap 0.1% + recorded nearest-valid inpainting, bearing tie rule (2%/20 deg), floor tol 0.15 m (44 render tests) |

## Branch migration (2026-08-19 ~23:00)
Rows above record development SHAs on `localization-exp` (history of record for r1/r2 + R-cal). exp_19 now lives on **`raf-finetune-exp`** (worktree `~/codespace/exp-19-raf-finetune`, base `6170007`): state-ported as `5f3e4a7` + `2f500d3` + `18c9aa9`, verified 326-passed. New rows below ledger THIS branch.

| SHA | Description |
|---|---|
| `5f3e4a7` | port 1/2: all exp_19-owned files @ localization-exp e23c45e |
| `2f500d3` | port 1b: test suite (glob miss caught) |
| `18c9aa9` | port 2/2: exp_19-authored shared-file deltas (eval_FLAC R3; metric-stack c12) |
| `5da9c71` | r2 fix R8+R10: RAF equal-room macro aggregation + invalid T60 count/rate (additive `RT60Error.invalid_stats`) + `l1_stft_multires` window on `x.device`; the 7 held tests now run (28 metric tests) |
| `dbae7f2` | r3 fix S4: per-item STFT attribution (per-scene L1_STFT no longer contains other rooms) + fail-closed RAF scene labels + independent unequal-room macro oracles |
| `2897b2b` | r3 fix S1: canonical publication authenticates the pinned record (sha256 e879768f, exact (X,Z,Y)+xyzw pins, both rooms + measurement blocks + raf_root); synthetic flows declare `--non-canonical` and taint every artifact |
| `3253c2f` | r3 fix S2: canonical miss cap may only be lowered; QA re-derives rate/count/hash and applies the registered cap itself |
| `6288d8a` | r3 fix S3: `PublishTransaction` — invalidate-then-swap across all roots, one generation-bound commit marker written last; no marker = unpublished |
| `0c96b72` | r3 fix S5: mesh-independent rx-sightline evidence (pose-file receivers) + real on-disk AR/HAA scale references; landmark bearing demoted to recorded-only |
| `c7882e6` | r3 fix S6: amplitude scalar derives from trained supports only, with an id-set hash required before a scalar applies |
| `cf439b5` | r3 fix S7: CPU bf16-autocast coverage for the multires-l1 window (+ opt-in non-default-CUDA case) |
| `de2075a` | r3 fix S8: contract text updated to the pinned gauge/quaternion and the cap-plus-inpaint miss policy |
| `ad631e9` | r4 fix T8: AR/HAA global L1_STFT restored bug-compatible (legacy B-squared weighting), corrected per-item weighting for RAF only; goldens hand-derived outside the callback |
| `6bff9c7` | r4 fix T6: tuple scene labels normalised+indexed, RAF labels validated against the room set, macro requires exactly both rooms |
| `4742954` | r4 fix T7: miss audit reads the RAW pre-inpaint hit mask; coordinates+hash mandatory (incl. empty set), unique/in-bounds/count/ray-count checked |
| `6fdda0e` | r4 fix T4: markers namespaced per kind (prepare/depth), empty/unexpected root sets rejected, `verify_combined_publication`, composition + crash tests |
| `c68d11b` | r4 fix T2: canonical split-parameter set enforced and bound into the marker identity |
| `5f40ebc` | r4 fix T1: read-once digest carried to provenance, every sub-verdict validated, corpus binding via room-index digests (counts fallback for the pinned record) |
| `d3381d9` | r4 fix T3: RAF_md first-load publication gate, cached per process (RAF_REQUIRE_PUBLICATION=1) |
| `ae9d308` | r4 fix T5: vertical-axis nadir-vs-tracked-height gate joins publication; horizontal-permutation detectability boundary RECORDED; candidate-error tests transform mesh+poses together |
| `cde1083` | r4 fix T9: committed synthetic depth-reference fixtures; real-HAA band as a skip-gated integration test pinned by hash |
| `0e100f0` | r4 fix T10: stale text purged (wxyz usage example, 'release does not state', 'full hit rate') |
| `36c92f3` | r4 re-pin: canonical record regenerated with `room_index` pose digests (sha256 `9288181b`); T1 corpus binding now full, counts fallback retained for legacy records |
| `9728e98` | r5 finding 1 (T3): mandatory combined publication gate in RAF_md — runtime pointer names the split dir, prepare+depth markers verified, canonical identity checked; env var removed, test-only `_RAF_MD_TEST_MODE` |
| `9e64a73` | r5 finding 5 (T5): raw RAF Y published as `tx_height_raf_m` and fed to the vertical check; end-to-end candidate-gauge CLI test (mesh+poses) |
| `c7b29f3` | r5 finding 4 (T7): miss audit derives count/coords/rate/hash from the raw mask; `mask_verified is True` required |
| `d48d708` | r5 findings 2+3 (T2/T4): seed in the prepare identity, new canonical RENDER identity validated pre-I/O, internal combined completeness + marker provenance validation |
| `a738fa1` | r5 finding 6: separate `bearing_delta_deg` / `vertical_delta_m` (the reused `delta` wrote vertical metres into the bearing field) |
