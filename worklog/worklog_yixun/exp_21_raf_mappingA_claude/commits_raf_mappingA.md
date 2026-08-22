# commits_raf_mappingA — exp_21 commit ledger (branch `raf-mapping-a`)

Base: `raf-finetune-exp` @ `263ef27` (full exp_19 pipeline + 557-test suite inherited).

| SHA | Description |
|---|---|
| `6f1c0f6` | r1 cycle 1: `cluster_placements` (complete linkage, 5 cm cap, medoid template) + `match_mics` (Hungarian, p95/max/ambiguity gates) — 21 tests |
| `49acbbb` | r1 cycle 2: `select_target` (hash-uniform, M8) + `stable_item_context` (per-item K=8, same-xyz exclusion per M5) — 32 tests |
| `13c8f77` | r1 cycle 3: audio-union enumeration + M1 amplitude audit (abort-with-measured-report, never drop/auto-adjust) — 46 tests |
| `ef7166d` | r1 cycle 4: `write_union` + Mapping-H byte-identity provenance (abort on disagreement) — 21 prepare tests |
| `a1fef7b` | r1 cycle 6: publication FLAVORS (`mappingA_prepare`/`mappingA_depth` kinds + registered identities); H→A, A→H, republish, injected-crash composition tests — 11 tests |
| `dce9610` | r1 cycle 5: `build_items` + M5 static manifest validator (18 tests) |
| `34cfcd3` | r1 cycles 8-9: `RAF_A_md` (AR_md semantics, per-context own-rx, listener depth) + mappingA publication gate + conditioner/C4 tests — 20 tests |
| `3bd6dae` | r1 cycle 7: listener-positioned render mode (`--positions-from mappingA`, raw rx height nadir gate, transmitter sightline probe, dedup) — 9 tests |
| `f38fd21` | r1: inherited exception-provenance test learns the second RAF hook |
| `7c0626e` | r1 cycle 10: `raf_mappingA.json` eval config + `mappingA_stats.py` (placement clustering unit, equal-room macro, room-stratified bootstrap, exact paired randomization) — 14 tests |
| `a14662d` | r1 cycle 11: end-to-end integration (chain composes; amplitude-gate and failed-correspondence negative paths) — 128 mappingA tests |
| `93e14c6` | r1: complete linkage via scipy (corpus-scale; cycle-1 semantics unchanged) |
| `f13263f` | r1 readback rung: correspondence record over both rooms — 74/91 placements, 73/86 eligible, 1,152-item identity ACHIEVABLE |
| `bdad9f5` | r1 cycle 12: `prepare_mappingA` CLI main() — survey→items→union→audit→staged publish, registered identity + 3 digests (30 prepare tests) |
| `1743de9` | r2 N2: disjoint Mapping-A split root (`data/RAF_mappingA`) — composition tests over the ACTUAL CLI defaults (the r4-T4 failure mode re-entering through a default value) |
| `96c267a` | r2: the eval-config assertion follows `MAPPINGA_SPLIT_ROOT` instead of a hardcoded copy |
| `9b9e696` | r2 N1: listener render publishes the `mappingA_depth` marker with its own derived identity; the RENDERER-produced marker passes `RAF_A_md` end to end |
| `b8c5a89` | r2 N3: audit the loader's 10,240-sample CROP for silence (full waveform for clipping), re-check at write, and split `amplitude_derivation_target` (0.75) from `clip_ceiling` (0.999) |
| `f158307` | r2 N4: `--mappingH-dir` located/required/verified in the CLI (generation, manifest coverage, scalar agreement); "byte-identical" replaced by CONTENT identity — float WAVs carry a PEAK-chunk timestamp |
| `a7414b5` | r2 N5: `correspondence_sha256` = digest of the committed record (read once), digests pinned exactly, canonical refused while the audio union is a placeholder, pointer↔marker cross-check |
| `bc5e70c` | r2 N6: validator recomputes displacement/source identity, full schema without defaults, per-context evidence + slot checks, REGISTERED item count, config `expected_items` → runtime stream gate |
| `1326cdf` | r2 N7: `eval_FLAC --record-per-item` sidecar (RAF-gated, additive) + `mappingA_stats` paired ingestion (exact item×seed pairing, registered design, paired placement bootstrap) |
| `f1b1808` | r2 N8: tracked height read from the RAW RAF row (`RAF_UP_AXIS`), not from a pipeline coordinate that holds it only under the current gauge |
| `044585e` | r2 N9: rigid residual (recorded), per-slot evidence digest, duplicate-receiver and zero/zero-margin refusals; algorithm → `mappingA-correspondence-2`; readback re-run + record re-pinned (`d4d79b49…`) |
| `1b8e8d7` | r3 P1: Mapping-A runtime root DERIVED as `<mappingH-dir>/mappingA`; equal/ancestor/outside/room-inside roots refused before any survey or write; both real CLIs composed at true defaults |
| `e56bb55` | r3 P2: `rx_row == assignment[group_key][mic_slot]` attested from the authoritative per-slot correspondence (published in the splits record, fail-closed by default); MATCH_SCHEMA gains the N9 digest + residual, all distances finite; reviewer's slot-0/row-23 probe committed as a negative test |
| `fd68f9d` | r3 P3: arm identity derived from sidecar provenance (ckpt sha, config digests, publication generation, stream hash); non-seed fields constant within an arm; registered mode = exactly seeds 42-46; cross-arm shared-identity + self-comparison checks |
| `b138a9b` | r3: N1 e2e test upgraded to a CANONICAL render whose marker is consumed untouched (bytes asserted unchanged); wrong-map-count refusal added |
| `0c4f6f9` | r4 Q1: protected rooms taken from the Mapping-H publication POINTER (`pointer_rooms`), not the A-run's `--rooms`; reviewer's EmptyRoom-run-writes-FurnishedRoom probe committed as a unit + CLI negative test |
| `b9a16e5` | r4 Q2: identity gains frame_avg_angles/fwd_cap, orbit_execution, cond_autocast, batch_size, source_sha and BOTH publication generations (RAF_A_md records prepare + depth); `assert_paired` requires every held-fixed control shared (1-vs-8-step probe) |
| `9dfe6bd` | r4 Q3: registered ckpt↔label registry (P1/BF/YAW/BV from ar_40k_endpoints MANIFEST.sha256, finetuned from exp_19 rcal_weights_sha256.txt), mismatched assertions refused, duplicate labels refused, arm identities kept positionally |
| `a273ff7` | r4 R1: the DERIVED `<H>/mappingA` default now runs the empty-protected-list refusal and the room-disjointness gates instead of returning ahead of them; reviewer's two probes committed as negative tests |
