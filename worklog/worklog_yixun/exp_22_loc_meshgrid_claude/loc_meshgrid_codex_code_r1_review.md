**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-25

*Round exp22-r1. REQUEST-CHANGES / both launches NO-GO: 4 BLOCKERs (worker-RNG call-graph divergence; substitution guard; direction-set not pinned + MeetingRoom_idx_32 anchor failure; audit driver absent). Cafe claims independently confirmed. Body verbatim.*

---

Verdict: **REQUEST CHANGES — exp22-r1 remains open.** Both launches are **NO-GO**.

## Findings

1. **[BLOCKER][D1] Worker RNG streams do not reproduce the released evaluator.**

   The materializer seeds, builds the loader, and immediately iterates it ([meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:147)). The released evaluator instead seeds, builds the loader, constructs the metric/AGREE stack, and only then creates the iterator ([eval_FLAC.py](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1333)). AGREE construction consumes torch RNG through randomized module initialization ([metric_callback.py](/home/yixunhu/codespace/FLAC/src/metrics/metric_callback.py:444), [audio_model.py](/home/yixunhu/codespace/FLAC/AGREE/AGREE/audio_model.py:14)).

   PyTorch draws the DataLoader worker base seed when the iterator is created; each worker’s NumPy seed derives from it. Therefore the intervening AGREE construction changes every worker’s `np.random.choice` stream, including short-pool replacement draws.

   The pool-equality test only compares one synthetic pool’s enumeration ([test_loc_meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_queries.py:55)); it never compares ordered draws under the two complete initialization call graphs. It cannot catch this defect.

2. **[BLOCKER][D1] Silent dataset substitution is not rejected.**

   `SampleDataset` replaces any failed item with a random other item ([dataset.py](/home/yixunhu/codespace/FLAC/src/data/dataset.py:357)). D1 records the substituted item’s identity but never verifies `md["idx"] == stream position`. Counts remain 6,337 and the eligible histogram can remain unchanged, so `census_verified` can still pass ([meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_queries.py:270)).

   The “extra loss” test removes an excluded-room record only ([test_loc_meshgrid_queries.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_queries.py:159)); it does not exercise an in-scope loss or same-count substitution. D1 needs the positional guard already implemented for evaluation streams in `eval_FLAC.py`, plus exact split-order/uniqueness validation.

3. **[BLOCKER][G1] The current direction set fails a real included room and is not actually pinned.**

   `FROZEN_DIRECTIONS` is regenerated from the implementation at import ([meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_geometry.py:45)); its test compares that function with its own output, so a changed generator remains green ([test_loc_meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/tests/test_loc_meshgrid_geometry.py:61)). No inherited-exp09 direction digest or literal array is pinned.

   My read-only 16-room anchor sweep found `MeetingRoom/MeetingRoom_idx_32` rejected: receiver `[2.26, 0.48, 1.2]` receives only **15/31** odd-parity votes, below the strict-majority requirement. Its surface distance is 0.25005 m, so this is not a clearance or boundary-epsilon issue. Current direction digest: `9ab4339fa893c00dca817b901a149c292b080d0e6971c90f0b8b0b88e858c261`.

4. **[BLOCKER][G1] The contracted 16-room audit tool/output path is absent.**

   The commit provides primitives, but no reviewed consumer resolves exactly one OBJ per room, joins D1 contexts in global coordinates, audits exactly 5,337 queries, computes both oracle branches, chooses the global branch, writes hashed candidate manifests, or reports candidate-query pairs, unique receiver-candidate pairs, conditioner calls, and artifact bytes required by the post-G1 cost gate.

   Additionally, `choose_z_branch` does not validate equal query sets or finite oracle values and defaults a missing full-height query to `0.0` ([meshgrid_geometry.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_geometry.py:311)). The current two-query test cannot detect incomplete audit maps.

## Launch calls

- **D1 full-pass manifest generation: NO-GO.** Fix the pre-iterator RNG-order mismatch and substitution guard, then pin the complete ordered stream against the inherited exp09 manifest/artifact.
- **G1 16-room audit: NO-GO.** Pin/cross-check the inherited direction set, resolve the MeetingRoom failure, and add the committed end-to-end audit/manifest/cost-report driver.

The Cafe claims were independently confirmed exactly: SHA `e7a0b7b9…`, 93,035 vertices, 366,080 triangles, all 10 sources and 100 receivers parity-valid, sources-only prior passes with minimum 0.550 m, and lattice `9,996 → 8,381 parity-valid → 6,273 valid`. Whole versus chunked Cafe classifications were byte-identical. The sources-only prior implementation agrees with the binding ruling.

No files were modified and no pytest run was attempted because the review was strictly read-only; validation used source inspection and CPU-only, no-write probes.