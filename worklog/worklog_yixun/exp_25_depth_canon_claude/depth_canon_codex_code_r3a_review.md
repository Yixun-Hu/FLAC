**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r3a (evaluator binding + sidecar + scored-waveform dump + stats tool)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING — Retrieval geometry is measured in a treatment-dependent frame

At [eval_FLAC.py:1949](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1949), `metadata` is replaced by the canonicalized list. The geometry supplied to metrics at [eval_FLAC.py:2010](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2010) is consequently canonical for CANON cells but merely post-rotation for P1/YAW/B-F cells.

This affects only `RIR_to_geom_R@k`: retrieval consumes `depth` at [metric_callback.py:323](/home/yixunhu/codespace/exp-25-depth-canon/src/metrics/metric_callback.py:323) and encodes it at [Retrieval.py:52](/home/yixunhu/codespace/exp-25-depth-canon/src/metrics/modules/Retrieval.py:52). `RIR_to_GT_RIR`, FD, T60, C50, EDT, and waveform metrics use audio only. Scene grouping is unchanged.

I agree with the Planner: metric geometry must be arm-independent. Preserve the post-rotation/pre-canonicalization list and use it for `scene_list`, `depth_list`, and `query_list`; only conditioning should receive canonical metadata. For semantic consistency, ARE’s explicitly post-rotation metadata input should also use that list, although CANON+ARE is not a registered cell.

Concrete test: in fixed nonzero and random modes, assert that:

- the conditioner receives `j*=0` metadata;
- the metric callback’s depth tensor is exactly derived from the post-rotation/pre-canonicalization metadata;
- the two differ on a fixture with a nonzero canonical shift;
- the disabled legacy path still returns/uses the same metadata objects.

### HIGH — The completion predicate is not actually a verified-completion predicate

`finalize` has the correct publication order: staging → read-back → `COMPLETE` → rename. However:

- The written `waveforms.sha256` file is never read back or validated before the marker is written at [eval_FLAC.py:1637](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1637).
- `is_complete` at [eval_FLAC.py:1657](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1657) returns true for any marker plus any JSON object—even `manifest.json == {}`. It checks neither row count, array presence/shape/dtype, checksum, nor marker contents.
- The test at [test_depth_canon_eval.py:700](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:700) explicitly enshrines that weak behavior.

That is safe only if round 3c never treats `is_complete` as sufficient, but its docstring calls it “the driver’s resume predicate,” while the plan requires complete-and-verified.

Concrete fix:

- Read back and compare the checksum sidecar before writing `COMPLETE`.
- Make `is_complete(final_dir, expected_n=None)` validate schema/core manifest fields, `n == shape[0]`, optional expected count, array dtype/shape/hash, checksum-sidecar text, and `COMPLETE` contents matching the waveform hash.
- Leave external metrics/stream/canon digest verification to the collector, and require both predicates before resume skips.
- Add marker-bearing tests for `{}` manifest, missing array, wrong `n`, bad checksum, and bad marker hash.

### HIGH — The sidecar still lacks the rotation-invariant Tier-1 aggregate requested for collection

`canon_hash` hashes full position tuples, including `dj`, at [eval_FLAC.py:1072](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1072). It therefore differs across rotations by construction; the test at [test_depth_canon_eval.py:359](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:359) asserts that difference. This does not satisfy plan §6 test 20’s intended one-line equality evidence.

I agree with the Planner: keep `canon_hash` as the cell-specific digest and add, for example, `canonical_depth_hash`, hashing ordered `(target_id, canonical_depth_sha256)` pairs. Test equality across rolls and sensitivity to reordering, target substitution, and hash alteration.

Stream order is guaranteed for registered cells: `shuffle=False`, sequential sampling, `drop_last:false` in all four planned AR configs, and `verify_stream_positions` rejects loader substitution. The paired `input_hash` additionally establishes identical item/context assignment.

### NIT — ViT unanimity can be bypassed by duplicate conditioner IDs

[eval_FLAC.py:420](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:420) stores widths in a dictionary keyed by conditioner ID. Duplicate IDs overwrite earlier widths. My isolated AST probe passed widths `512` and `256` under the same ID and `canon_vit_img_w` returned `256` instead of rejecting disagreement.

Use an indexed list of `(position, id, width)` entries, or explicitly reject duplicate IDs.

### NIT — Dump-target hygiene is weaker than its operational role

[eval_FLAC.py:1585](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1585) correctly refuses a missing root, symlink/file targets, and nonempty final directories. It permits:

- an already-existing empty final cell directory, which is later silently removed;
- a root/final path nested under a reserved `.staging-<pid>` name.

Prefer refusing every pre-existing final cell directory and any reserved staging-path component. The driver should quarantine and report existing targets instead of the evaluator erasing an empty one.

### NIT — Enforce the float32 assumption behind “exactly as scored”

`append` silently casts arbitrary input dtypes to float32 at [eval_FLAC.py:1608](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1608). The registered FLAC path starts with float32 noise and samples/decodes outside autocast, so its `fakes` should be float32 and the dump is exact. Still, fail closed on a non-float32 input rather than silently weakening the standing “exact tensor handed to `update_metrics`” contract. The current double-input test should become a rejection test.

## Contract review

### Legacy path

For a checkpoint/file with no `training.canonicalize` block and no new flags, the path remains behaviorally and serially identical to `2be08f6` for a fixed `source_sha`:

- output paths retain their exact names;
- metrics fields and order are unchanged;
- predictions metadata is unchanged;
- no `.canon.json` is written;
- the existing `.stream.json` construction is unchanged;
- `prepare_eval_metadata(..., canonicalize=False)` directly returns `apply_rotation_plan`;
- fixed zero rotation returns the original list object;
- `build_output_paths` is called once.

Moving `build_output_paths` earlier is sound. By then `rotation_plan`, frame-average angles, and resolved ARE lambda all exist in fixed/random, vanilla/FA, and ARE paths. `test_eval_paths` still spies one call with the same substantive arguments, although its “save stage” wording is stale.

There is a legacy behavior test beyond names, but not a full parent-vs-head loop snapshot: `test_legacy_cell_is_untouched` drives the real loop and checks metadata identity, output file set, and absence of new fields/files. Golden record serialization and the older save-path integration supplement it.

### Protocol binding

The requested truth table is implemented correctly:

- completely absent blocks and flags → exact legacy tuple;
- matching enabled blocks plus flag → accepted;
- matching enabled blocks without flag → fatal;
- matching disabled blocks without flags → accepted as in-play, trained/applied false;
- flag with absent blocks → fatal unless explicitly authorized as the P1+canon diagnostic;
- trained-but-not-applied and applied-but-not-trained diagnostics require the allow flag;
- unnecessary allow flag → fatal;
- missing checkpoint `model_config['training']` while in play → fatal;
- absent versus `{enabled:false}` → fatal;
- embedded-only enabled or disabled block → fatal;
- `1` versus `true`, differing key sets, or other disagreement → fatal even with allow;
- `fa_invariant` plus `--canonicalize` → fatal;
- trained block `img_w` must equal the unanimous file-config ViT width;
- binding precedes model construction and cannot be bypassed by `--allow-partial-load`.

“In play” includes the embedded block and is computed from it before deciding on the legacy branch.

The sixth `file_model_cfg` parameter is operationally safe in the sole production caller because it is named. As API hardening, make `cond_method` and `file_model_cfg` keyword-only, with `file_model_cfg` required whenever in play; the current optional positional shape is easy for tests or later drivers to miscall.

### Metadata, sidecar, and records

Rotation precedes canonicalization. `rotate_scene_metadata` shallow-copies dictionaries and replaces rotated tensors, preserving the loader dictionaries and the post-rotation/pre-canonicalization objects needed for the BLOCKING fix.

A real AR probe found:

- `.npy` depth dtype `float64`, shape `[256,512]`;
- converted depth `torch.float64` on CPU;
- `collation_fn` returns a list containing a tuple of the original metadata dictionaries, without stacking or moving their depths.

`sample_target_id` uses `idx` plus `relpath/path`, which rotation does not modify. Position-by-position comparison with the pre-rotation `RotationStream` is therefore correct in random mode. `StabilizedTieError` remains uncaught and carries the sample identifier.

The existing `.canon.json` fields match the plan, and `canonical_depth_sha256` is imported directly from the training-side `yaw_rotation.py`, so Tier 1 compares like with like. Canon provenance is written whenever the protocol is in play; the sidecar is written only when canonicalization is actually applied. `canon_sidecar_path` is injective whenever the metrics path is injective.

### Scored waveforms

`append(fakes)` receives the exact object immediately passed to `update_metrics`, after clamp and both padding branches. If `reals` is padded, the unchanged scored `fakes` is still what both consumers receive. Stored rows are detached, contiguous CPU float32 `[B,T]`.

The implementation accumulates every row in RAM, then creates a concatenated tensor and a full `np.load` copy during finalization. For 6,337×10,240:

- final array: 259,563,520 bytes = 247.54 MiB;
- rows + concatenation + read-back: about 742.62 MiB per cell, before comparison temporaries;
- two concurrent evaluators: roughly 1.45 GiB extra.

This is not cumulative across 230 cell processes and is probably tolerable, but a staging `open_memmap` would better satisfy the SOP’s bounded-working-set preference.

On a mid-loop exception, no staging exists because all rows are still in RAM. On a finalization/disk-full failure, staging remains without `COMPLETE`; a rename failure leaves a verified, marked staging directory; a crash after rename leaves a complete final directory. No incomplete final directory is published.

The manifest’s `input_hash` correctly uses the pre-rotation `RotationStream`. For the campaign this is sound because every cell passes `--record-stream`. For general CLI safety, either require `record_stream=True` for dumps or automatically write the stream sidecar; random mode alone currently permits a hash with no published preimages.

All planned AR configs have `drop_last:false`, so `len(dataset) == c == appended rows`. Other configs with `drop_last:true` fail finalization rather than silently publishing a short dump.

### Stats and CLI

`rel_l2` and `rel_linf` match the registered denominators exactly, with the second argument as reference. Inputs are converted to finite float64 `[N,T]`; malformed, empty, nonfinite, or mismatched shapes are rejected.

`paired_summary` uses NumPy’s default linear-interpolation p99—not nearest-rank. This is documented and tested; add “NumPy linear percentile” to the plan for reproducibility. The `1/sqrt(2)` correction was mathematically correct and affected the test expectation only; my probe reproduced `0.7071067811865475`.

CLI defaults are correct:

- `--canonicalize`: `store_true`;
- `--allow-canon-mismatch`: `store_true`;
- `--dump-scored-waveforms`: `None`.

Treating the dump argument as a root is preferable and acceptable. Amend plan §5’s common command by removing `<cell stem>` from the argument. The driver must create/verify the root once, pass that root for every cell, and poll `<root>/<metrics-stem>/COMPLETE`. No additional evaluator flag is presently missing.

## Coder deviations

1. **Sixth `file_model_cfg` parameter:** Accept, with the keyword-only hardening noted above.
2. **Dump argument is a root:** Accept; amend plan §5 and the future driver.
3. **Dump requires an accumulated stream:** Accept for planned cells; strengthen to require or automatically publish the stream sidecar.
4. **`expected_n = len(dataset)`:** Accept. Planned configs have `drop_last:false`; mismatches fail closed.
5. **Early single `build_output_paths` call:** Accept.
6. **Additional manifest artifact digests:** Accept and useful; completion verification still needs the HIGH fix.
7. **Canon/rotation target-ID consistency check:** Accept; correct in random mode.
8. **Static `is_complete`:** Form accepted, current semantics rejected pending the HIGH fix.
9. **`--store_predictions` untouched except filename:** Accept; it remains the legacy pre-clamp artifact.
10. **Toy CPU stack:** Accept. It drives the real `evaluate_model`, not a copied loop.
11. **Importing `_parse_canonicalize_config`:** Accept; one validation definition is preferable. A public shared helper could replace the private import later.

## Open questions

1. **Metric geometry frame:** Confirm the Planner’s position. This is the BLOCKING fix above.
2. **Rotation-invariant canonical digest:** Confirm. Add the ordered `(target_id, canonical_depth_sha256)` digest while retaining `canon_hash`.
3. **Whole-config hash equality:** Confirm the Planner’s position. The evaluator should record both hashes but not globally require equality; legitimate evaluation files can differ outside `training`. The round-3c collector should require equality for registered table cells.

## Tests and SOP discipline

The 147+20 cases are generally strong, especially the binding truth table, sidecar preimages, real-loop toy integration, corruption mutation, and CLI subprocess coverage. They are not all independently load-bearing:

- the source-inspection test at [test_depth_canon_eval.py:855](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:855) is refactor-brittle and should be replaced by call-order/count spies;
- the `is_complete` test locks in the weak predicate;
- the canonical-hash test locks in the absence of a rotation-invariant aggregate;
- no test currently pins the arm-independent metric geometry frame.

The three skipped tests in the Coder’s core run are:

- the tiny real-CUDA DDP transfer test, because CUDA was hidden;
- the two exp_14 real-rung1 artifact tests, because that artifact is absent.

The optional real AR depth-map test ran because the `AcousticRooms` mount is present.

SOP evidence is otherwise satisfactory:

- the invalid 5a red was disclosed and rerun against the parent;
- 5d’s implementation was correct and only the erroneous test expectation changed;
- all 14 code commits are below 200 changed lines, largest 192;
- the scoped evaluator diff is +610/−13, with the deletions accounted for as moved/refactored lines;
- `git diff --check` is clean;
- no tracked environment/package changes are present;
- the Coder reports CPU-only work and no training/evaluation launch.

Two docstrings should be corrected: [eval_FLAC.py:1049](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1049) calls `md` canonicalized although it is pre-canonicalization, and `is_complete` overstates itself as the driver predicate.

I did not run pytest, training, evaluation, or GPU commands. I verified the exact diff and commit statistics, traced all affected paths against `2be08f6`, inspected the real AR loader/converter/collation dtype path, calculated dump memory, and ran isolated in-memory probes for the stats formulas, percentile behavior, helper signature, and duplicate-ID ViT-width counterexample.