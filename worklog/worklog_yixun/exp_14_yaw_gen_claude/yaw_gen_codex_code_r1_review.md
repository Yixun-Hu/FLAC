# Code review — exp_14_yaw_gen round 1 (eval-side random-yaw mode)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec -s read-only`) · **Date:** 2026-08-11 · **Commits under review:** `66e6ca5` `9e737a1` `ebd7983` `dbf0fae` `16d7d13` (base `44788e6`) · **Tokens:** 233,820 · Raw transcript: session scratchpad `yaw_gen_codex_r1_review_raw.log`

VERDICT: REVISE

1. BLOCKING — `eval_FLAC.py:817`, `src/tests/test_yaw_random_eval.py:858`: count enforcement is tautological. The code compares the stream with `len(dataset)`, so even the test’s zero-item dataset produces a valid random artifact. This does not enforce the pre-registered 6,337 samples; the pending `idx == i` assertion will not fix that. Require the campaign’s expected count explicitly, verify `len(dataset) == len(stream) == 6337`, and add wrong-size/empty-dataset rejection tests.

2. BLOCKING — `worklog/worklog_yixun/exp_14_yaw_gen_claude/yaw_gen_worklog.md:59`, `plan_yaw_gen.md:53`, `plan_yaw_gen.md:73`: ruling (1)’s proposed sidecar payload is incomplete. Recording only `input_hash`, target IDs, and count omits the ordered context fingerprints and `img_w` needed to reproduce the input hash, plus offsets needed by G3 and the R/V assignment hash. Keep the separate opt-in sidecar, but store the full canonical input tuples and R/V offsets/assignment tuples, their hashes, schema version, and count; write it only after validation.

3. BLOCKING — `yaw_gen_worklog.md:60`, `worklog/worklog_yixun/gen_model_comparison.py:16`, `src/metrics/modules/Retrieval.py:19`, `src/metrics/modules/Retrieval.py:59`: ruling (2) rests on the wrong retrieval branch. The reported table R@1 is `RIR_to_GT_RIR_R@1`, computed from predicted-audio versus GT-audio embeddings. Only `RIR_to_geom_R@k` consumes the rotated point cloud. FD is indeed audio-only (`src/metrics/modules/FD.py:24`), but the claimed confound does not justify replacing R@1. Restore T60/R@1 as co-primaries and disclose only geometry retrieval as confounded. Any independent FD substitution needs a new scientific rationale and user approval before data collection.

4. NIT — `eval_FLAC.py:227`, `src/configs/dataset_configs/custom_metadata/AR_md.py:33`, `AR_md.py:124`: the context fingerprint is stable only because the current loader pins `context_poses` to float32. It is not dtype-stable: across the actual 6,415 unseen source/receiver pairs, float32 versus float64 changed two six-decimal strings, and float16 changed 5,032. Add fail-closed float32/finite/shape assertions and version the fingerprint schema.

5. NIT — `src/tests/test_yaw_random_eval.py:806`: the `evaluate_model` wiring test uses an empty stub loader, so actual multi-worker ordering remains untested. Add a small in-memory map-style DataLoader test with `num_workers > 0`, two batches, and ordered `idx` values; assert the golden offsets attach in sampler order. This remains a strengthening test, since PyTorch 2.7’s default `in_order=True` and the repository’s `shuffle=False`/`drop_last=false` were verified.

Planner rulings:

1. REVISE AS WRITTEN. A separate opt-in `.stream.json` is sound and preserves fixed metrics-record/path bytes, but it must carry canonical tuple preimages and offsets, not merely hashes/target IDs.

2. REJECT. Geometry retrieval is confounded; the experiment’s existing R@1 is audio-to-audio and is not.

3. ACCEPT FOR THIS CAMPAIGN WITH A DTYPE PIN. The fingerprint is ordered, captured pre-rotation, and collision-safe on the actual unseen data: zero six-decimal collisions and minimum source separation 0.2 m. It is not generally dtype-stable. The pending `idx == position` assertion is sound for detecting this dataset’s recursive substitution behavior.

Verified vs trusted: I ran the two new suites with cache/bytecode disabled: 98 passed, 3 warnings in 27.13s. I independently re-derived the seed-42 offsets and both canonical hashes; all matched. I executed the pre-change `44788e6` helper definitions in memory against every golden fixture value; zero mismatches. Commit ancestry confirms `66e6ca5` was captured before source edits, and `44788e6`/`89f24cd` have identical eval/yaw-helper bytes. Fixed records, paths, prediction metadata, and dispatch behavior are pinned; `AR_md.py` and `dataset.py` are untouched. Cycle structure and sub-200-line source deltas are verified; the reported red-first counts and 394-test combined run are trusted from the worklog/Coder report, not rerun here.
