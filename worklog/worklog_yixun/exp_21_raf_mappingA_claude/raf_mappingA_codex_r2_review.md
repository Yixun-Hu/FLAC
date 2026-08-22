# VERDICT: REQUEST-CHANGES

Three genuine in-boundary operator-error defects remain.

## Findings

1. **P1 — Critical — [prepare_mappingA.py:1031](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:1031), [prepare_mappingA.py:1160](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:1160), [prepare_mappingA.py:1232](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:1232), [test_mappingA_publish.py:91](/home/yixunhu/codespace/exp-21-raf-mapping-a/src/tests/test_mappingA_publish.py:91)**  
   **Problem:** N2 is only partially closed. The split defaults are now disjoint, but there is no Mapping-A runtime default or disjointness check: `--output-dir` is required and staged verbatim. The test calling its topology “ACTUAL” manually invents `runtime/RAF/mappingA`; it is not derived from either CLI. Passing the natural Mapping-H root as both `--mappingH-dir RAF` and `--output-dir RAF` overwrites `RAF/raf_publication.json` and its root manifest, invalidating H; listener rendering then also collides with H’s depth roots. This is exactly a registered wrong-flag/operator-error state.  
   **Fix:** derive `<Mapping-H runtime>/mappingA`, or require and verify that Mapping-A’s output is a proper disjoint child of `--mappingH-dir`; refuse equal or ancestor roots before surveying/writing. Exercise that refusal and both real CLI invocations in composition tests.

2. **P2 — High — [prepare_mappingA.py:735](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:735), [prepare_mappingA.py:764](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:764), [prepare_mappingA.py:827](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:827), [prepare_mappingA.py:895](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/prepare_mappingA.py:895)**  
   **Problem:** N6’s promised group/slot attestation remains absent. The validator checks only that `rx_row` is within 0–35; it never proves that it is the Hungarian row assigned to the item’s `mic_slot`. `MATCH_SCHEMA` also omits the N9 evidence digest and rigid-residual fields. A read-only probe confirmed that an item for slot 0 with target row 23, context rows 17–24, and no evidence digests passes validation. Thus “full schema” and “context group/slot attested against correspondence evidence” are false.  
   **Fix:** retain authoritative assignment/per-slot evidence through validation; require finite full match evidence, recompute its digest, and enforce `target/context rx_row == assignment[group_key][mic_slot]`.

3. **P3 — High — [mappingA_stats.py:105](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:105), [mappingA_stats.py:146](/home/yixunhu/codespace/exp-21-raf-mapping-a/data/RAF/mappingA_stats.py:146), [eval_FLAC.py:1683](/home/yixunhu/codespace/exp-21-raf-mapping-a/eval_FLAC.py:1683)**  
   **Problem:** N7 enforces item/seed equality only after the operator manually groups files into an arm. `arm_from_sidecars` ignores all provenance except `seed`, accepts an externally supplied label as the arm identity, and does not require registered seeds 42–46. Sidecars from different checkpoints/conditions can therefore be pooled as one arm, and a one-seed or wrong-seed experiment still receives the registered contrast report.  
   **Fix:** derive and validate arm identity from sidecar provenance, require all non-seed protocol fields to be constant within an arm, enforce exactly seeds 42–46 in registered mode, and compare shared protocol/publication or stream identity across arms.

## Verified closures

- **N1:** renderer selects `mappingA_depth`, derives the 1,152-file identity, carries the pinned readback digest, and RAF_A_md requests the Mapping-A flavor.
- **N3:** full-wave clipping and first-10,240-sample crop silence are separately checked; write-time recheck and 0.75/0.999 identity split are present.
- **N4:** Mapping-H generation/manifest coverage and scalar are checked up front; shared audio uses exact float32 content identity.
- **N5 core:** pins exactly match `d4d79b49677b…` and `9288181be62b…`; canonical publication correctly refuses the placeholder audio-union digest.
- **N8:** tracked height comes directly from raw RAF axis 1.
- **N9:** algorithm is `mappingA-correspondence-2`; duplicate/zero-zero ambiguity refusals and rigid/per-slot evidence are implemented.

The regenerated correspondence record recomputed exactly from the read-only corpus: 892/427 and 927/158 pass/fail, 73/86 eligible, zero duplicates or degenerate margins. All 2,404 sized groups carry evidence digests; the sole 165-way digest duplicate is the placement-medoid identity evidence. Passing rigid residual maxima remain ≤1.585 µm; FurnishedRoom failing RMS reaches 0.1827 m.

## Residuals

- `audio_union_sha256` remains intentionally unpinned; the registered dry-run → pin → canonical-run workflow is still required.
- Shared WAV identity is content rather than file-byte identity because of the PEAK timestamp.
- Rigid residual remains recorded, not gated, as registered.
- The N1 “end-to-end” test edits a non-canonical renderer marker into canonical form; code inspection closes N1, but the test is not literally an untouched canonical renderer output.

No writes or installs were performed. Pytest was not run because its cache/tmp fixtures violate the strict read-only constraint; verification used code inspection, hashing, and `python -B` in-memory recomputation only.