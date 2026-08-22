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
