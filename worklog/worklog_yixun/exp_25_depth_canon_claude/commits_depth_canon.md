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
