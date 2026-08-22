# VERDICT: REQUEST-CHANGES

The implementation has genuine in-boundary operator-error/crash-state defects. Canonical Mapping-A cannot currently reach a valid consumer-verifiable publication.

## Findings

1. **N1 — Critical — [render_depth.py:989](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/render_depth.py:989), [render_depth.py:1002](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/render_depth.py:1002), [render_depth.py:1148](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/render_depth.py:1148)**  
   `--positions-from mappingA` still validates the Mapping-H render identity, creates `PublishTransaction(..., kind="depth")`, and emits the Mapping-H parameter payload. `RAF_A_md` requires `mappingA_depth`, so a canonical listener render can report success but can never satisfy the consumer gate. The advertised Mapping-A `n_maps`, `positions_from`, and readback fields are never derived or written by this CLI.  
   **Fix:** select `mappingA_depth` and its complete identity when listener mode is active; derive and enforce the actual depth-file/map count and readback digest. Add a canonical-equivalent end-to-end test where the renderer-produced marker—not a hand-built marker—passes `RAF_A_md`.

2. **N2 — Critical — [prepare_mappingA.py:584](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:584), [prepare_mappingA.py:704](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:704), [publish.py:23](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/publish.py:23), [test_mappingA_publish.py:92](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/tests/test_mappingA_publish.py:92)**  
   The real Mapping-A CLI defaults its split root to `data/RAF`, the same root used by Mapping-H. Both flavors therefore overwrite the single `raf_publish_manifest.json`; publishing either flavor invalidates the other flavor’s marker-to-manifest digest. The four composition tests conceal this by giving A a separate `data_RAF_mappingA` root. A crash after invalidating the shared manifest likewise makes H unpublished.  
   **Fix:** use a genuinely disjoint Mapping-A split root and update [raf_mappingA.json:7](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/RAF/eval/raf_mappingA.json:7), or implement flavor-scoped manifests as well as markers. Exercise the actual CLI/default topology in H→A, A→H, republish, and injected-crash tests.

3. **N3 — High — [prepare_mappingA.py:124](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:124), [prepare_mappingA.py:170](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:170), [dataset.py:289](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/data/dataset.py:289)**  
   The amplitude audit tests the peak of the entire resampled waveform. The eval loader crops to the first 10,240 samples and then applies its −60 dB test. A file with a quiet initial crop and a later loud peak passes this audit but is silently substituted at evaluation. Exp_19’s correct implementation explicitly audits `out[:sample_size]` at [prepare_data.py:1064](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_data.py:1064). Also, the marker claims `amplitude_ceiling=0.75` while the actual clip gate is 0.999, conflating the historical derivation target with the enforced limit.  
   **Fix:** audit and record both full and first-10,240-sample peaks, fail on crop silence, and recheck during writing. Distinguish the 0.75 derivation target from the 0.999 clip ceiling. Add a delayed-impulse negative test.

4. **N4 — High — [prepare_mappingA.py:710](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:710), [prepare_mappingA.py:233](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:233)**  
   `write_union` can compare shared captures against Mapping-H, but the production CLI never passes `mappingH_room_dir` or `mappingH_generation`. Every real run will therefore record zero shared files and cannot detect differing bytes. The cycle-4 unit tests exercise an API path the shipped workflow does not use.  
   **Fix:** locate and verify the Mapping-H publication, require its generation, pass both into every room’s `write_union`, and reject a claimed Mapping-H tree/file not covered by that generation’s manifest.

5. **N5 — High — [prepare_mappingA.py:691](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:691), [publish.py:402](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/publish.py:402), [publish.py:481](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/publish.py:481)**  
   `correspondence_sha256` covers only placement counts, selected IDs, and pass/fail totals—not assignments, per-slot displacements, group gates, medoids, or the committed correspondence record. Additionally, every Mapping-A digest remains `SHA256_SHAPE`, so any 64-character lowercase hex value is accepted as canonical. This permits stale/wrong correspondence or audio-union evidence through the operator-error gate.  
   **Fix:** digest the full canonical correspondence evidence and require the committed record as a read-once input; pin its now-known digest. Pin the existing exp_19 readback digest exactly, and refuse canonical publication while the audio-union pin remains a placeholder. Cross-check pointer, prepare marker, and depth marker identities.

6. **N6 — High — [prepare_mappingA.py:400](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:400), [prepare_mappingA.py:440](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:440), [prepare_mappingA.py:451](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:451), [prepare_mappingA.py:677](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:677)**  
   The static validator trusts the stored `xyz_key`, `rx_displacement_m`, and match summaries instead of recomputing them from poses/correspondence evidence. Missing match fields default to passing values. It does not attest each context group’s successful correspondence or actual mic-slot assignment. Canonical `main()` also validates against `expected_items=n_items`, making the item-count gate tautological. The config’s `expected_items=1152` is otherwise unused.  
   **Fix:** require the full schema without defaults; recompute source-position identity and receiver displacement; validate context group/slot assignments against correspondence evidence; enforce 16×36×2 and exact room/slot structure in canonical mode; wire the expected count into the runtime stream gate.

7. **N7 — High — [mappingA_stats.py:23](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:23), [mappingA_stats.py:38](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:38), [mappingA_stats.py:97](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:97)**  
   The local arithmetic examples are correct, but the promised statistical workflow is absent. No evaluator output contains per-item metrics and `mappingA_stats.py` has no record/prediction ingestion surface. Pairing checks only placement keys, so different items or seeds within the same placements pass; `macro_two_room` accepts one or extra rooms; and `paired_randomization` returns only a p-value, not the registered paired interval.  
   **Fix:** retain or compute per-item metrics keyed by arm/seed/item, require exact item×seed equality before aggregation, enforce two rooms/16 placements each, bootstrap paired placement differences within room, and emit the registered contrast interval plus separately labelled seed SD.

8. **N8 — Medium — [prepare_mappingA.py:364](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:364), [prepare_mappingA.py:390](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:390)**  
   `rx_target_height_raf_m` is taken from transformed `rx_target_p[2]`, not raw RAF Y. It happens to be numerically equal under the current correct `(X,Z,Y)` transform, but it is not independent evidence: a wrong vertical transform could change both the rendered position and the purported “raw” reference together.  
   **Fix:** persist `target["rx_xyz"][target_row][RAF_UP_AXIS]` directly, following exp_19’s pattern at [prepare_data.py:1180](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_data.py:1180), and test a wrong candidate gauge while leaving raw height untouched.

9. **N9 — Medium — [mappingA_common.py:105](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_common.py:105), [run_mappingA_readback.py:72](/home/yixunhu/codespace/exp-21-raf-mapping-a/worklog/worklog_yixun/exp_21_raf_mappingA_claude/run_mappingA_readback.py:72)**  
   The registered rigid-array residual is not computed anywhere. The committed record also discards Hungarian assignments, per-slot displacements, and per-slot margins. Additionally, exact duplicate microphone coordinates produce zero matched and second-nearest distances, but the zero-displacement branch labels the margin infinite and can pass an intrinsically ambiguous array.  
   **Fix:** calculate and record the preregistered rigid residual, preserve or digest full per-slot correspondence evidence, reject duplicate receiver coordinates, and treat `matched==second==0` as ambiguous.

## Verified claims

- **M3 passes:** [RAF_A_md.py:56](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/RAF_A_md.py:56) implements `tx_target − rx_target`; [RAF_A_md.py:90](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/RAF_A_md.py:90) uses every context capture’s own receiver; and [RAF_A_md.py:99](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/RAF_A_md.py:99) loads the target-receiver panorama. These match [AR_md.py:31](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/AR_md.py:31), [AR_md.py:120](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/AR_md.py:120), and [AR_md.py:48](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/configs/dataset_configs/custom_metadata/AR_md.py:48).

- **Normal-case M2 gating passes:** scipy complete linkage retains the fixed-input complete-linkage/cap semantics; p95, maximum, and ambiguity gates are applied per group, and `survey_room` puts only passing groups into eligibility/items.

- **M1 ordering partly passes:** the union is exactly the deduplicated target/context union, and both room audits finish before the publishing transaction starts, so an amplitude-policy failure writes no publication. N3 and N4 prevent full approval.

- **FA/C₄ is semantically applicable:** listener-centered depth and `*_vit` poses rotate together, while the pose path uses cylindrical features. The Mapping-A tests exercise vanilla, FA, and C₄ behavior; N8 concerns QA independence, not FA geometry.

## Correspondence-record assessment

I independently recomputed both rooms from the raw pose files in memory. The committed headline values match exactly:

- EmptyRoom: 47,484 captures, 1,319 groups, 74 placements, 892/427 pass/fail, 73 eligible.
- FurnishedRoom: 39,132 captures, 1,086 groups, one 72-capture exclusion, 91 placements, 927/158 pass/fail, 86 eligible.

The bimodality is real: median passing-group p95 is approximately 0.410/0.246 mm, while median failing-group p95 is 4.13/4.51 cm. “Failures sit at 4–7 cm” should be read as the dominant mode, not a universal range: the smallest failed p95 is about 1.03–1.07 cm and FurnishedRoom has a long tail to 34 cm p95/41 cm maximum displacement. Eligibility flags and stored pass/fail gates are internally consistent.

The full committed record SHA-256 is `f2da911b5de82e7914a0cf234c0f0713051880784a4a71c163fa92b377288da4`; it is currently not pinned or consumed by publication.

## Coder review notes

- **`n_placements` coupling:** `16` is consistently used as a per-room count and normal construction yields `16×36×2=1152`; this is not independently blocking. The canonical validator’s self-count and unused config count remain N6.
- **Summary-not-full-record digest:** confirmed as N5.
- **`SHA256_SHAPE` pins:** confirmed as N5. Shape checking is acceptable only during a pre-canonical development rung, not in a verifier that already accepts canonical publication.

The source tree contains 137 Mapping-A test cases after parametrization, matching the claim. I did not execute pytest because of the strict no-writes constraint; no files or installs were changed.