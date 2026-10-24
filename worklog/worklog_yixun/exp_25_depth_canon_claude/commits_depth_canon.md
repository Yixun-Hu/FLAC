# exp_25 depth_canon — commits

Branch `exp-25-depth-canon` (worktree `~/codespace/exp-25-depth-canon`), cut from `localization-exp` @ `6eb2b5e`.

| SHA | Author seat | One line |
|---|---|---|
| `6eb2b5e` | Planner | scaffold + plan v1 + Codex plan review r1 (REVISE) + tie-prevalence probe |
| `08ac314` | Planner | plan v2 — Codex r1 findings addressed; Yixun decisions D1–D6 recorded (base of TDD round 1) |
| `a385250` | Coder r1 | cycle 1: `recover_depth_proxy` + `column_profile` (bitwise yaw-stable proxy; contract rejections) |
| `f5476cf` | Coder r1 | cycle 2a: `canonical_column` single-candidate classes (`none`/`same_column`) + fail-closed `StabilizedTieError`, `TIE_CLASSES` |
| `33deb59` | Coder r1 | cycle 2b: rooted-image tie-break key (C-1) → `cross_column_image`; Option-A strict xfail |
| `517fe82` | Coder r1 | cycle 3: source key (C-2) + ordered pair-preserving context key (C-3); stabilized all-roll raise |
| `5dee91f` | Coder r1 | cycle 4: `canonical_shift` + `canonicalize_scene_metadata` (input contract, post-condition, raise propagation) |
| `649f98e` | Coder r1 | cycle 5: `canonical_depth_sha256` (float64, signed-zero normalized) + canonical-output invariance test 7 |
| `551d537` | Coder r1 | docstring precision: the 1e-6 radius is a degeneracy threshold, not a comparison tolerance |
| `fbd00c8` | Coder r1 | records: round-1 lab-notebook entries + this file |
| `518e03a` | Planner | plan v3→v5 (Codex plan r2 REVISE → r3 → r4 → r5 APPROVE) + plan reviews r2–r5, comparator ledger, training-path diff audit rev. 2 (P1 legs `e50d098`/`1b56e7a`), worklog |
| `c1d229b` | Coder r1 fix | F1 (red by design): tests for proxy-only selection + fail-closed periodic ties whatever the poses (code review r1 BLOCKING) |
| `f85c412` | Coder r1 fix | F2a (still red by design): retire the v3–v5 pose-key tests and fixtures |
| `c8a58b0` | Coder r1 fix | F2b: proxy-only `canonical_column`; any rooted-image tie → `StabilizedTieError`; 3 tie classes; wording without stabilizer claims |
| `1b1a47e` | Coder r1 fix | F3: all four pose fields required; `canonicalize_scene_metadata` has no `pose_keys` (all of `POSE_KEYS` rotated) |
| `6cad8a6` | Coder r1 fix | F4: `StabilizedTieError` carries the loader sample identifier (`idx`/`path`/`relpath`), picklable |
| `f734c67` | Coder r1 fix | F5: planning probe asserts map count + float64; read-only re-run reproduces every recorded number |
| `054b986` | Coder r1 fix | F6: docstrings — hash formula names the `+ 0.0`; re-wrap a long line |
| `de6e774` | Coder r1 fix | records: lab-notebook entries F1–F6 + this file (round-1 fix round) |
| `be7c1d0` | Planner | plan v6 (image-only tie-break, fail-closed survivors, round-2 CPU-float64 contract) after Codex code review r1; worklog |
| `71670d9` | Planner | plan v7 (stale v5 wording removed; round-2 transfer-hook tests 12b/12c) — round 1 closed after Codex re-verification; worklog |
| `a08254b` | Coder r2 | cycle 1: `_parse_canonicalize_config` (`CANON_KEYS`; fail-closed; yaw_aug / fa_invariant exclusions) + `**canon_kwargs` splice into the `diffusion_cond` wrapper call (test 10) |
| `5d8b893` | Coder r2 | cycle 2: wrapper `canonicalize_enabled` / `canonicalize_img_w` raw validation + exclusions, `canon_tie_counts`, rank-0 banner `canonicalize ENABLED img_w=<w> tiebreak=image`, `on_fit_end` tie totals (tests 11, 13, 14) |
| `c9ddb92` | Coder r2 | cycle 3a-i: `_check_canon_metadata` — yaw_aug schema rules under `canonicalize:` + depth CPU / float64 contract (tests 13, 12c) |
| `4ccf54b` | Coder r2 | cycle 3a-ii: `_canonicalize_batch` — per-sample operator, atomic tie counters, training-path `self.log`, `StabilizedTieError` propagates (tests 13, 12c) |
| `1b43d34` | Coder r2 | cycle 3b: canonicalization hook first in `training_step` / `validation_step` / `test_step` (test 12) |
| `3a60e9c` | Coder r2 | cycle 4: `transfer_batch_to_device` override — metadata left on the CPU when enabled, parent otherwise; meta-device + CUDA + real-fit order tests (12b, 12c) |
| `aab5164` | Coder r2 | cycle 5a (red by design): arm-config contract test 22 |
| `e72142c` | Coder r2 | cycle 5b: `FLAC_AR_CANON.json` = `FLAC_AR_BVp1.json` bytes + `training.canonicalize {enabled: true, img_w: 512}` (sha `aba94032…`) |
| `9350054` | Coder r2 | records: lab-notebook entries (device-consumer verification, cycles 1–5b, round summary) + this file |
| `a7f8893` | Planner | round-2 receipt: independent verification (576 passed, 1 xfailed), ledger rows for the Planner commits, worklog |
| `1adda49` | Planner | plan v8: DDP-opaque metadata carrier (Codex code review r2 BLOCKING), tests 12b/12c amended |
| `c7771ec` | Coder r2 fix | F1 (code review r2 BLOCKING): `_CpuMetadataCarrier` — enabled `transfer_batch_to_device` hands the untouched metadata on in an opaque `__slots__` carrier that DDP's `_to_kwargs`/`_recursive_to` does not descend into; `_canonicalize_batch` unwraps it first; real-`_to_kwargs` regression tests (meta + CUDA) |
| `8f16dfb` | Coder r2 fix | F2 (NIT): operator `ValueError`s re-raised as `canonicalize: metadata[i]: …` (`StabilizedTieError` untouched) |
| `b424969` | Coder r2 fix | F3 + records: ledger one row per SHA (chronological; dup `518e03a` merged; `de6e774`/`9350054`/`a7f8893`/`1adda49`/`c7771ec`/`8f16dfb` added) + fix-round notebook entries |
| `df31f44` | Coder r2 fix | records: F3 notebook-entry timestamp corrected to the `date(1)` value (18:57:13, was hand-typed 18:58:24) |
| `cd31610` | Planner | round-2 fix-round receipt (Planner verification 586 passed, 1 xfailed); worklog |
| `2be08f6` | Planner | round 2 closed (Codex re-verification SHIP); round-3 split (3a / 3b / 3c) recorded; worklog — base of TDD round 3a |
| `83504a0` | Coder r3a | cycle 1a: `canon_suffix` + `build_output_paths(..., canonicalize, canon_mismatch)` tokens after the method suffix, before rotation (test 16; golden + exp_01 names byte-identical; 72-combo injectivity) |
| `3904928` | Coder r3a | cycle 1b: `build_metrics_record(..., canon_provenance=None)` → `_canon_provenance`, `CANON_RECORD_KEYS` (8 fields, validated fail-closed, appended last) (test 17) |
| `d9148b3` | Coder r3a | cycle 2a: `CanonProtocol` / `LEGACY_CANON_PROTOCOL`, `_canon_strict_diff`, `canon_vit_img_w`, `model_config_sha256` (test 18 helpers) |
| `1428ba8` | Coder r3a | cycle 2b: `resolve_canonicalize_protocol` — fail-closed checkpoint binding, every plan §2.2 rule (test 18) |
| `f9e0e3c` | Coder r3a | cycle 3a: `CanonStream`, `canon_sidecar_path`, `build_canon_record` (`.canon.json`) (test 20) |
| `2981bf8` | Coder r3a | cycle 3b-i: `prepare_eval_metadata` — rotation, then per-sample canonicalization on the CPU float64 cloud (test 19) |
| `46521f9` | Coder r3a | cycle 3b-ii: `build_canon_provenance` (record fields from the protocol + stream + both config digests) |
| `0f5d100` | Coder r3a | cycle 4a: `ScoredWaveformDump` construction refusals + `append`; `scored_waveform_dir` (test 21) |
| `6eb790b` | Coder r3a | cycle 4b: `ScoredWaveformDump.finalize` (staging → verify → `COMPLETE` → atomic rename) + `is_complete` (test 21) |
| `c62ac96` | Coder r3a | cycle 5a: `evaluate_model` canonicalization wiring — binding before model construction, one early `build_output_paths`, `prepare_eval_metadata` in the loop, canon checks, provenance, `.canon.json` (toy-stack drive of the real loop) |
| `bd38095` | Coder r3a | cycle 5b: `evaluate_model` scored-waveform dump wiring — stream required, target refused before the model, `append` right before `update_metrics` (spied), `finalize` last |
| `7e51b11` | Coder r3a | cycle 5c: CLI `--canonicalize` / `--allow-canon-mismatch` / `--dump-scored-waveforms` threaded into `evaluate_model` (behavioural subprocess tests) |
| `bd6bddc` | Coder r3a | cycle 5d: `src/tools/depth_canon_stats.py` — `rel_l2` / `rel_linf` / `paired_summary` with the pinned denominators (test 21b) |
| `666e0f9` | Coder r3a | cycle 5e: dump verification compares bits through a `uint32` view (no 2×260 MB `tobytes` copies) + one-bit / dtype / length corruption test |
| `9efcf13` | Coder r3a | records: lab-notebook entries (start, cycles 1a-5e, round summary) + ledger rows |
| `e0bcae0` | Planner | r3a receipt — independent regression 1755 passed / 7 skipped / 1 xfailed at `9efcf13`, static checks, provisional positions on the Coder's open questions; Codex r3a requested |
| `180fea4` | Planner | Codex code review r3a (REQUEST-CHANGES: BLOCKING metric geometry frame = A4, 2 HIGH, 3 NIT) + Planner receipt |
| `503a2da` | Planner | plan v9: arm-independent metric path (A4), verified completion predicate, `canonical_depth_hash`, dump ROOT semantics, p99 definition pinned |
| `5e99cad` | Coder r3b | cycle 1a: tie audit — raw-side selector (numpy rooted-image key) + circular candidate separation |
| `92c0372` | Coder r3b | cycle 1b: tie audit — split item -> depth map via AR_md's own path logic (np.load spy test) |
| `4bdc9a3` | Coder r3b | cycle 2: tie audit — split configs -> loader items (`get_audio_filenames`) -> unique maps + inputs ledger |
| `1a18bab` | Coder r3b | cycle 3: tie audit — per-map row: AR_md cloud -> `canonicalize_scene_metadata` vs the raw selector |
| `3bfc2d6` | Coder r3b | cycle 4a: tie audit — ordered (multi-)worker run, per-split summaries, gate (exit 0/1/3/4), deterministic JSON + sha256 sidecar |
| `c879ba7` | Coder r3b | cycle 4b: tie audit — CLI (`--out --workers --limit --splits`), markdown summary, throughput projection |
| `af8f02b` | Coder r3b | cycle 5a: composition probe — registered offsets + fixed per-item noise (test 21b) |
| `fbae181` | Coder r3b | cycle 5b: composition probe — stagewise relL2 / relL∞ via `depth_canon_stats` (test 21b) |
| `91ba965` | Coder r3b | cycle 6a: composition probe — A = canon(md), B = canon(rot_d(md)) + the probe items' Tier-1 record |
| `3212b91` | Coder r3b | cycle 6b: composition probe — stage runner mirroring eval_FLAC's 1-step loop (fp32 / eval protocols) |
| `408553e` | Coder r3b | cycle 7: composition probe — checkpoint load through `evaluate_model`'s path (equivalence test) |
| `1da0f34` | Coder r3b | cycle 8a: composition probe — the probe loop: stage records, determinism floor, ε_cond, control p99, Tier 1, verdict |
| `2ad63ee` | Coder r3b | cycle 8b-i: composition probe — first-n loader items, markdown summary, provenance helpers |
| `881575a` | Coder r3b | cycle 8b-ii: composition probe — CLI `main` (load, seeded loader, probe, JSON + sidecar, exit = verdict) |
| `0a271e4` | Coder r3b | cycle 9: launcher kit — pin generator + `depth_canon_pins.sha256` (44 files) + guard-suite skeleton (P cases, throwaway clone) |
| `5e1c181` | Coder r3b | cycle 10a-i: launcher skeleton — mode, string-pinned rung, harness location, teed log, the exact argv (P1 recipe) |
| `417e1ed` | Coder r3b | cycle 10a-ii: launcher — HEAD pin (`EXPECT_HEAD`) |
| `cce2a34` | Coder r3b | cycle 10b: launcher — content pins (coverage + bytes + committed) and the config contract |
| `e7e8902` | Coder r3b | cycle 11a: launcher — smoke-evidence prerequisite, NAS gate, run-owned namespaces |
| `7048ccf` | Coder r3b | cycle 11b: launcher — GPU free-memory gate, W&B identity gate, DINOv3 gate on this checkout's src |
| `e490a88` | Coder r3b | cycle 12a: launcher — the launch path: marker, drained single log read, scans, run-owned checkpoint set + manifest |
| `84d7403` | Coder r3b | cycle 12b: launcher — steady-state smoke projection + the smoke evidence bound to the final log; opt-in production guard section |
| `58c296f` | Coder r3b | records: lab-notebook entries (start, cycles 1a-12b, real-data throughput, round summary) + ledger rows |
| `b9f83b9` | Coder r3a-F1 | fix 1 (BLOCKING, A4): `PreparedMetadata(cond, metric)` — only the conditioner sees the canonical frame; metric inputs + ARE add-back from the post-rotation, pre-canonicalization list |
| `f50a436` | Coder r3a-F1 | fix 2 (HIGH): `finalize` reads the checksum sidecar back; `is_complete(final_dir, expected_n=None)` verifies the whole cell |
| `bb6bbfa` | Coder r3a-F1 | fix 3 (HIGH 2): `.canon.json` gains the rotation-invariant `canonical_depth_hash` |
| `b2e59da` | Coder r3a-F1 | fix 4 (NIT): `canon_vit_img_w` refuses duplicate conditioner ids |
| `fdf5267` | Coder r3a-F1 | fix 5 (NIT): dump target — refuse ANY existing final path (never remove one) and reserved `.staging-` components |
| `f42ffa9` | Coder r3a-F1 | fix 6 (NIT): `append` fails closed on non-float32 |
| `2d1c15b` | Coder r3a-F1 | fix 7: `cond_method` / `file_model_cfg` keyword-only; `file_model_cfg` required in play |
| `0b21c12` | Coder r3a-F1 | fix 8: a dump requires `--record-stream` (published stream sidecar) |
| `1d28e41` | Coder r3a-F1 | fix 9: routing test spies on the real functions (replaces `inspect.getsource`) |
| `958bc9f` | Coder r3a-F1 | records: lab-notebook entries (F1 start, fixes 1–9, round summary + final 3b/F1 regression) + ledger rows |
| `9afeb9e` | Planner | records: receipt for rounds 3b + r3a-F1 (part 1) — regression 1911 passed / 7 skipped / 1 xfailed at `958bc9f`; Codex F1 re-verification requested |
| `0e60b57` | Planner | records: Codex r3a-F1 re-verification SHIP — round 3a closed; 2 non-blocking NITs queued |
| `1d438c7` | Planner | records: guard-suite receipt (118 passed incl. read-only NAS production cases); Codex code review r3b requested |
| `5bd522b` | Coder r3c | cycle 0a: `is_complete` `schema_version` type-strict (true / 1.0 / "1" refused) |
| `2bd0aa3` | Coder r3c | cycle 0b: toy-loop spy pins the ARE add-back to the post-rotation, pre-canonical list (test only) |
| `a768604` | Coder r3c | cycle 1a (red by design): cell-manifest tests — counts, order, injective names/stems, canon tokens, common flags |
| `c9e7918` | Coder r3c | cycle 1b: `depth_canon_make_cells.py` — the deterministic 230-cell manifest generator |
| `c22fca5` | Coder r3c | cycle 1c: manifest tests per block class, canon protocol, configs / split counts / checkpoint pins (mutation-red) |
| `a13154f` | Coder r3c | cycle 1d: `depth_canon_cells.json` (generated data) + drift tests |
| `4ab894b` | Coder r3c | cycle 2a-i (red by design): collector fixtures (real evaluator functions) + `record_reasons` tests |
| `b8bbf06` | Coder r3c | cycle 2a-ii: `depth_canon_collect.py` — `record_reasons` |
| `bb70812` | Coder r3c | cycle 2b: `stream_reasons` + `canon_reasons` |
| `c50d5bd` | Coder r3c | cycle 2c-i (red by design): dump + receipt tests |
| `d9744f8` | Coder r3c | cycle 2c-ii: `dump_reasons` (is_complete once / manifest mode) + `receipt_reasons` + `cell_argv` |
| `199d80f` | Coder r3c | cycle 2d-i (red by design): `verify_cell` / CANON index / CLI tests |
| `89713cd` | Coder r3c | cycle 2d-ii: `verify_cell`, `cell_paths`, `load_canon_index`, `--verify-cell` |
| `f5c768f` | Coder r3c | cycle 3: `select_cells` / `plan_cells` / `write_receipt`; `--plan` / `--write-receipt` |
| `3d196bd` | Coder r3c | cycle 3b: `--write-receipt --verify-after` |
| `639d186` | Coder r3c | cycle 4a-i (red by design): driver tests — throwaway clone + harness, DRY_RUN argv, refusals (206 changed lines, disclosed) |
| `7ac11d0` | Coder r3c | cycle 4a-ii: `depth_canon_eval_run.sh` skeleton — gates, plan, checkpoint sha, DRY_RUN |
| `eacf3b7` | Coder r3c | cycle 4b-i (red by design): run-path tests — lanes, skip, four quarantine states, failure, busy GPU, no-dump block |
| `ee42356` | Coder r3c | cycle 4b-ii: the driver's run path — predicate, quarantine, GPU lanes, receipt + verify, summary |
| `d4a1184` | Coder r3c | cycle 5a: full mode — `collect_cells`, completeness, `make_pairs`, Tier 1 |
| `d31adca` | Coder r3c | cycle 5b-i (red by design): Tier 2 tests |
| `e07283b` | Coder r3c | cycle 5b-ii: Tier 2 + `load_control` (registered probes only) |
| `46cba20` | Coder r3c | cycle 5c: Tier 3 + the table block |
| `4c3eaa4` | Coder r3c | cycle 5d: TRAJ grid, B-F historical quote, tie counts, seen/unseen overlap disclosure |
| `fd646b0` | Coder r3c | cycle 5e-i (red by design): results / refusal / `--partial` tests |
| `2929e12` | Coder r3c | cycle 5e-ii: `build_results`, `render_md`, `--collect` |
| `d9fa986` | Coder r3c | cycle 6-i (red by design): generator exp25 tests |
| `e93c42a` | Coder r3c | cycle 6-ii: `gen_model_comparison.py` exp25 rows + contract (additive) |
| `6c92209` | Coder r3c | records: lab-notebook entries (start, cycles 0a–6-ii, round summary) + ledger rows |
| `c6feecf` | Coder r3b-F1 | fix 3 (HIGH 3): tie audit enforces the training input contract (float64 (256,512) raw, float64 cpu (3,256,512) cloud, depth modality) |
| `ef43d03` | Coder r3b-F1 | fix 8 (NIT 8): tie-audit precedence INCOMPLETE > FAIL > REFUSED_SUBSET > PASS |
| `7333a57` | Coder r3b-F1 | fix 2a (BLOCKING 2): `_canonicalize_batch` latency (perf_counter), `canon/latency_ms`, the rank-0 latency line at fit end |
| `dc52861` | Coder r3b-F1 | fix 4 (HIGH 4): probe registered contract vs `--diagnostic` (registered:false, exit 6) |
| `ddea813` | Coder r3b-F1 | fix 5 (HIGH 5): pin closure (+ FLAC_AR_BV / FLAC_AR_BF), manifest regenerated (46); guard `GUARD_ONLY` filter |
| `22d520c` | Coder r3b-F1 | fix 6 (HIGH 6): the launcher requires the flac interpreter for every gate and train.py |
| `ce92947` | Coder r3b-F1 | fix 7 (HIGH 7): GPU occupancy gate (no compute process on GPU 0/1; pid + cwd printed, never signalled) |
| `f8db9dc` | Coder r3b-F1 | fix 1 (BLOCKING 1): SMOKE and FULL gated by the complete PASS tie audit (+ sidecar, counts, hashes) |
| `4256fc7` | Coder r3b-F1 | fix 2b (BLOCKING 2): smoke evidence requires finite train/loss + the measured canonicalization latency; FULL checks both |
| `8404388` | Coder r3b-F1 | fix 9 (NIT 9): content-scoped SMOKE->FULL binding (manifest, launcher, pin generator, contract, audit; HEAD provenance) |
| `f8dcb5f` | Coder r3b-F1 | fix 10 (NIT 10): launcher header documents its write targets incl. repo-local `wandb/` |
| `5408a68` | Coder r3b-F1 | fix 9b: guard state — re-write the E9 evidence after section B |
| `929576d` | Coder r3d | cycle 1a (red by design): autolaunch watcher tests — token + GPU waits |
| `827548d` | Coder r3d | cycle 1b: `depth_canon_autolaunch.sh` — preconditions, GO token, GPU condition, atomic state (fire point refuses) |
| `0ba37ac` | Coder r3d | cycle 1c: precondition tests (mutation-red) |
| `0f22293` | Coder r3d | cycle 2a (red by design): fire-sequence + state tests |
| `33a8d72` | Coder r3d | cycle 2b: the fire sequence (SMOKE -> FULL, transient GPU-gate retries <= 5, final failures) |
| `dff44c8` | Planner records | Codex code-review artifacts r1 / r1 re-verify / r2 / r2 re-verify / r3b committed (previously untracked) |
| `1c0f6fb` | Planner records | plan v10: Codex r3b amendments, round 3c as built, round 3d watcher, A4/A5 resolved (Yixun's pre-approved GO, freeze rule), A6 eval GPU policy, schedule |
| `dbbe07f` | Planner records | receipt for rounds 3c + r3b-F1 + 3d (regression 2155/7/1, guard 150/0/0), Codex r3b verdict, pre-approved GO, adjudications Q1–Q6 + A6, reviews A/B launched |
| `84f0a9e` | Coder rB-F1 | B1 cycle 1a-i (red by design): tie-audit gate tests on a real tiny audit — structural mutations, sidecar, non-JSON |
| `dcdd945` | Coder rB-F1 | B1 cycle 1a-ii (red by design): gate tests — limit/subset/survivor runs, stale split JSON, cardinalities, the registered contract vs the real splits, real audit, CLI |
| `6780dc9` | Coder rB-F1 | B1 cycle 1b: `depth_canon_audit_gate.py` part A — sidecar, exact schema, tool hashes + clean tree, inputs ledger recomputed by the audit's own code, cardinalities |
| `6178158` | Coder rB-F1 | B1 cycle 1c (red): gate row tests — census, per-row contract, recomputed summaries, re-hashed map bytes |
| `9134922` | Coder rB-F1 | B1 cycle 1d: gate part B — exact census, per-row contract, per_split/agreement recomputed, current map bytes re-hashed |
| `de8981a` | Coder rB-F1 | B1 cycle 1e (red): collector tests on a REAL audit — every invalid audit / disagreeing cell refuses full collection (exit 3, nothing written) |
| `6434db7` | Coder rB-F1 | B1 cycle 1f: collector `load_tie_audit` = the shared gate; `tie_summary` before the refusal, contradictions are problems |
| `f52b90d` | Coder rB-F1 | B1 cycle 1g (regression fix): the driver stub evaluator puts src/tests on sys.path (1f's fixture imports sibling test modules) |
| `cb08bd8` | Coder rB-F1 | B2 cycle 2a (red): driver lane-gate tests (uuid-mapped occupancy, fail-closed queries, opt-in sharing, re-check before spawn) |
| `3fd2b8d` | Coder rB-F1 | B2 cycle 2b: `lane_state` — >= 10 GiB free AND no compute process, fail-closed queries, re-check before each spawn, `ALLOW_SHARED_GPU=1` relaxes occupancy only |
| `93ef8e2` | Coder rB-F1 | B3 cycle 3a (red): receipts written by the real writer; campaign/spawn/post digest binding tests; one digest per collection |
| `fe61023` | Coder rB-F1 | B3 cycle 3b (red): driver tests — startup digest in the receipt, edit between cells -> exit 4 before the 2nd spawn, edit during a cell -> that cell fails |
| `c7ef06b` | Coder rB-F1 | B3 cycle 3c: relevant-tree digest at start / before every spawn / after every cell, PIN DRIFT exit 4, receipt binds the three digests, verify_cell + collection check them |
| `8d0335f` | Coder rB-F1 | B4 cycle 4a (red): checkpoint TOCTOU tests — verified cache + hard link, measured digest, space gate, symlink/foreign entry, tampered cache, bytes changed during the cell |
| `00ac606` | Coder rB-F1 | B4 cycle 4b: content-addressed verified checkpoint cache, free-space gate, hard-linked farm entries re-checked before the spawn, post-run measured sha256 in the receipt |
| `7338ceb` | Coder rB-F1 | B5 cycle 5a (red): probe tests — dirty tree = registered deviation; main() publishes via build_report / write_report |
| `2d427a1` | Coder rB-F1 | B5 cycle 5b: probe `tree_deviations` (checked before loading) + `build_report` / `write_report` |
| `69ee428` | Coder rB-F1 | B5 cycle 5c (red): collector probe-binding tests on a real toy probe artifact (26 refusals, legitimate FAIL, contract mirror) |
| `55c835d` | Coder rB-F1 | B5 cycle 5d: `PROBE_CONTRACT` + `probe_reasons`; `load_control` -> (control, reasons, eps); eps_cond threshold + PASS/FAIL rendered |
| `56cad9b` | Coder rB-F1 | B6 cycle 6a (red): offline-verifiable receipt tests (artifact digests, dump evidence, receipt mode, written only after the full verification) |
| `e126fe7` | Coder rB-F1 | B6 cycle 6b: `build_receipt` + `dump_evidence` + `evidence_reasons`, `verify_cell(dump_check='receipt', receipt_doc=)`, verify-then-write receipts |
| `0662305` | Coder rB-F1 | B6 cycle 6c (red): generator offline validation + NAS retention audit tests |
| `e24c9f2` | Coder rB-F1 | B6 cycle 6d: generator validates offline (receipt mode) with one campaign digest per row / round; `--nas-retention-audit` |
| `b6de552` | Coder rB-F1 | NIT a cycle 7a (red): K nonempty context ids; the registered repetition rule; the pairing note |
| `57cf336` | Coder rB-F1 | NIT a cycle 7b: `stream_reasons` K ids, `context_replacement` (same repeated targets per split and K), `PAIRING_NOTE` |
| `cf14d57` | Coder rB-F1 | NIT b cycle 8a (red): overlap derived from the validated audit, compared with the registered 1,093 / 112 |
| `51bc8b8` | Coder rB-F1 | NIT b cycle 8b: `split_overlap(summary)` + `REGISTERED_OVERLAP`; mismatch flagged + a problem |
| `557f3dc` | Coder rB-F1 | NIT c cycle 9 (test-only): golden byte-identity of every pre-exp25 generator row vs `e5902b9` |
| `356ef98` | Coder rB-F1 | NIT d cycle 10 (test-only): two-cell failure-then-success driver test |
| `dfbaaef` | Coder rB-F1 | A1 cycle 11a (red): gate harness-contract mode + checkout-portable ledger tests |
| `6de842f` | Coder rB-F1 | A1 cycle 11b: `--harness-contract` (only under DEPTH_CANON_HARNESS, tagged reports only), `*_abs` ledger fields portable |
| `268bd22` | Coder rB-F1 | data: the full rung-3 tie audit + sidecar committed unchanged (+ run log, gate re-validation log) |
| `a5c28fd` | Coder rB-F1 | A1 cycle 11c (red, guard): T section on real tiny harness audits; P1b/M6 48 pins; B4/B5; Q0 |
| `343bdf1` | Coder rB-F1 | A1 cycle 11d: the launcher calls depth_canon_audit_gate.py; gate sha in the SMOKE->FULL bind; audit + sidecar pinned (48) |
| `08ef666` | Coder rB-F1 | A2 cycle 12a (red): DINOv3 snapshot pin tests on synthetic HF caches |
| `36765c0` | Coder rB-F1 | A2 cycle 12b: `assert_vit_pin` pins config.json + model.safetensors (exact file set); pins regenerated (48) |
| `2122aa2` | Coder rB-F1 | A3 + A6e cycle 13a (red, guard): strict nvidia-smi cases, GPU refusals rc 7 |
| `6b42489` | Coder rB-F1 | A3 + A6e cycle 13b: launcher `gpu_query` (status apart from stdout), strict parsing, every GPU-gate refusal exits 7 |
| `ca8d59a` | Coder rB-F1 | A3/A4/A6 cycle 14a (red): watcher strictness, literal rung + scrub, rc-7 transient, interpreter, token, dirty pins, drain |
| `8fd226a` | Coder rB-F1 | A3/A4/A6 cycle 14b: the watcher fixes (strict gpu_clean, env -u + MB=32 ACC=1 DRY_RUN=0, PY_PIN, 4-line ISO token, clean pins, rc 7, tee drain) |
| `27a87b7` | Coder rB-F1 | A5 cycle 15a (red): the latency covers unwrap + validation |
| `999d515` | Coder rB-F1 | A5 cycle 15b: `perf_counter` before the unwrap; pins regenerated (48) |
| `76de174` | Coder rB-F1 | records: notebook A2–A6 + ledger rows |
| `5111f4e` | Planner records | Codex code review B (round 3c) REQUEST-CHANGES receipt + artifact; fix round rB-F1 opened |
| `3688275` | Coder rB-F2 | F2-1a (red, guard + watcher): train.py's exit 7 must come back as the terminal 8 (R9, Z9; R9b passthrough); the watcher makes one attempt on it |
| `978de18` | Coder rB-F2 | F2-1b: `gpu_refuse` is the only emitter of rc 7; an EXIT trap remaps any other 7 to 8; header lists every exit code |
| `2998f7e` | Coder rB-F2 | F2-2a (red, guard): strict-only nvidia-smi stub (other queries logged + fail); U12/U12b/U13 |
| `01cd41a` | Coder rB-F2 | F2-2b: the best-effort disclosure queries removed; the disclosure logs the validated gate answers |
| `d348bd1` | Coder rB-F2 | F2-3a (red, guard): T15 a must-PASS fixture that is INCOMPLETE fails loudly |
| `c652249` | Coder rB-F2 | F2-3b (guard repair): Q restores a fresh harness clone; `write_audit ok` asserts a complete PASS + prints the audit's error; tmp root /tmp/depth_canon_guard.XXXXXX validated |
| `93de567` | Coder rB-F2 | F2-4a (red): re-hash child under a 900 s deadline; hung / failing children fail closed |
| `84ced81` | Coder rB-F2 | F2-4b: the gate re-hashes in a killable child (`--rehash-worker`), `REHASH_DEADLINE_S = 900`, process group SIGKILLed past it |
| `0480900` | Coder rB-F2 | F2-5a (red): token with a blank line and no final newline must wait |
| `8b177d5` | Coder rB-F2 | F2-5b: `token_ok` counts every logical line (== key lines == newlines == 4) |
| `85c81c2` | Coder rB-F2 | F2-6a (red): eval_FLAC hashes and deserializes one buffer; `ckpt_sha256_loaded` / `ckpt_bytes_loaded` in the record |
| `8e87d17` | Coder rB-F2 | F2-6b: `read_checkpoint_exact`, `torch.load(io.BytesIO(buf), map_location='cpu')`, `build_metrics_record(ckpt_load=)` |
| `6885b3c` | Coder rB-F2 | F2-6c (red): receipt binds loaded == measured == manifest; writable cache entry rejected; pre-spawn byte flip refused |
| `b724af6` | Coder rB-F2 | F2-6d: collector LOAD_KEYS + receipt binding + `--ckpt-bytes`; driver `cache_ok` (0444, size, re-hash) at staging and before each spawn |
| `39ccdf4` | Coder rB-F2 | F2-7a (red): the retention audit refuses `--dump-check receipt` |
| `bf7bf38` | Coder rB-F2 | F2-7b: `--nas-retention-audit` accepts only full / manifest |
