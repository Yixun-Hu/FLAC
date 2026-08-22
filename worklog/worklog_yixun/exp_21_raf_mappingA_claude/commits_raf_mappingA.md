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
