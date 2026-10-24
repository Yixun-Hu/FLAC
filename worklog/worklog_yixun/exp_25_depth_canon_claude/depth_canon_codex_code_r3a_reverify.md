**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r3a-F1 re-verification

**Verdict:** SHIP

All prior BLOCKING, HIGH, and NIT findings are closed. I found two non-blocking test/schema hardening NITs; neither affects registered exp_25 cells.

## Closure table

| r3a finding | Status | Evidence |
|---|---|---|
| BLOCKING — arm-dependent metric geometry | **Closed** | `PreparedMetadata(cond, metric)` preserves the post-rotation list as `metric` and produces canonical `cond`; disabled mode returns the same list object for both at [eval_FLAC.py:1190](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1190). Both vanilla and FA branches receive `cond` at [eval_FLAC.py:2030](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2030); FA+CANON remains prohibited, so an FA cell gets the shared noncanonical list. ARE receives `metric` at [eval_FLAC.py:2071](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2071), and scene/depth/query are all built from it at [eval_FLAC.py:2091](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2091). Fixed-90 and random real-loop tests prove exact pre-canonical metric geometry, canonical conditioner input, and a nonzero difference at [test_depth_canon_eval.py:968](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:968). Disabled identity is pinned at [test_depth_canon_eval.py:475](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:475). |
| HIGH — weak completion predicate | **Closed** | Finalization performs array write → checksum sidecar and manifest → array/manifest/sidecar read-back and rehash → hash-bearing `COMPLETE` → rename at [eval_FLAC.py:1664](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1664). `is_complete` validates the requested manifest fields, `n`, optional `expected_n`, dtype, shape, array hash, sidecar, and marker at [eval_FLAC.py:1707](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1707). The positive case and marker-bearing `{}`, missing-array, wrong-count, checksum, marker, and altered-array cases are covered at [test_depth_canon_eval.py:788](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:788); checksum read-back before marker is specifically mutation-tested at [test_depth_canon_eval.py:830](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:830). |
| HIGH — missing rotation-invariant Tier-1 aggregate | **Closed** | `canonical_depth_hash` hashes ordered `[target_id, canonical_depth_sha256]` pairs while leaving `canon_hash` unchanged at [eval_FLAC.py:1086](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1086). Tests establish equality across rolls and sensitivity to reorder, target substitution, and hash alteration at [test_depth_canon_eval.py:391](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:391). |
| NIT — duplicate conditioner IDs | **Closed** | Indexed `(position, id, width)` entries and explicit duplicate rejection at [eval_FLAC.py:414](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:414), including agreeing and disagreeing duplicate-width tests. |
| NIT — dump-target hygiene | **Closed** | Every existing final path is refused and retained; reserved `.staging-` path components are rejected at [eval_FLAC.py:1632](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1632). Tests cover empty/nonempty directory, file, symlink, and reserved components at [test_depth_canon_eval.py:671](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:671). |
| NIT — silent float32 cast | **Closed** | `append` now rejects every non-float32 scored tensor at [eval_FLAC.py:1647](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1647), tested for float64, bfloat16, and float16. |
| Hardening — positional protocol arguments | **Closed** | `cond_method` and `file_model_cfg` are keyword-only; `file_model_cfg` is required whenever the protocol is in play at [eval_FLAC.py:450](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:450). The production caller uses names. |
| Hardening — dump without published stream | **Closed** | A dump now requires literal `record_stream=True` before any file/config/model work at [eval_FLAC.py:1841](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1841), including rejection of random mode alone. |
| Brittle `inspect.getsource` routing test | **Closed** | Replaced by spies around the real resolver, path builder, metadata preparation, rotation, conditioner, dump append, and metric update at [test_depth_canon_eval.py:1030](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_eval.py:1030). Recorded mutation checks demonstrate it detects reversed append/score order and duplicate path construction. |

The whole-array hash in `is_complete` is acceptable for a predicate invoked once per cell: it streams roughly 260 MB without materializing a second array. Across 230 cells that is about 60 GB of verification reads; it should not be used as a tight polling loop.

## New findings

### NIT — The ARE two-list route is correct but not mutation-pinned

[eval_FLAC.py:2073](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2073) correctly passes `metric_metadata`, but existing CANON tests do not enter the ARE branch. The disabled same-object contract cannot distinguish `metadata` from `metric_metadata`.

This does not block exp_25 because CANON+ARE is unregistered. Before treating that composition as supported, add a toy-loop spy with canonicalization and a mocked ARE bank, asserting `apply_are_addback` receives the post-rotation/pre-canonicalization list.

### NIT — `schema_version` validation is not type-strict

At [eval_FLAC.py:1726](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1726), `m.get("schema_version") != 1` accepts JSON `true` and `1.0` because both compare equal to integer `1` in Python. Other integer fields correctly use `type(x) is int`.

Concrete fix: require `type(version) is int and version == SCORED_DUMP_SCHEMA_VERSION`, with tests for `true`, `1.0`, and `"1"`. This is schema hardening only; array, checksum, and marker integrity remain verified.

## Legacy and regression verification

- The definitions of `canon_suffix`, `build_output_paths`, `build_metrics_record`, `build_predictions_meta`, `stream_sidecar_path`, and `build_stream_record` are byte-identical between `84d7403` and `1d28e41`.
- Golden tests still pin output names, metrics field order/JSON bytes, and predictions metadata at [test_exp14_fixed_mode_snapshot.py:87](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_exp14_fixed_mode_snapshot.py:87).
- `PreparedMetadata` does not reach prediction storage; predictions still append the sampled tensor directly at [eval_FLAC.py:2080](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:2080).
- The item counter, `verify_stream_positions`, stream-sidecar serialization, and scored-tensor identity are unchanged.
- Both previously flagged docstrings are now accurate at [eval_FLAC.py:1061](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1061) and [eval_FLAC.py:1707](/home/yixunhu/codespace/exp-25-depth-canon/eval_FLAC.py:1707).

## What I verified

- Exact scoped diff `84d7403..1d28e41`, `git diff --check`, commit ordering, and per-commit statistics. Largest fix commit is 118 changed lines, below the SOP’s 200-line guideline.
- RED evidence is credible and parent-sensitive for fixes 1–8; fix 9 records two meaningful mutation reds. No after-the-fact test is apparent.
- Read-only AST probes confirmed both conditioner branches use `cond`, while ARE uses `metric`, and confirmed the disabled `PreparedMetadata(rotated, rotated)` identity.
- Inspected the Planner’s receipt log: **1911 passed, 7 skipped, 1 xfailed, 0 failed in 367.48 s**.
- HEAD differs from `1d28e41` only in the notebook/commit ledger; the two scoped code files are identical.
- I did not run pytest, training, evaluation, or GPU commands. A direct import probe was stopped by the read-only sandbox when a framework import attempted temporary-directory allocation; no files or processes were modified.