**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rB4 (re-verification of the rA3/rB3 probe-provenance HIGH after fix round rB-F3)

**Verdict:** **SHIP** — the sole open probe-provenance HIGH is resolved. rB-F3 affects no training-launch input.

## Findings

- **BLOCKING:** None.
- **HIGH:** None.
- **NIT:** None.

## Per-item status

**F3-1 — RESOLVED.**

- The probe reads the checkpoint once through `eval_FLAC.read_checkpoint_exact`, validates exactly the two provenance fields, deserializes that buffer with `torch.load(io.BytesIO(buf), map_location="cpu")`, then releases `buf`: [depth_canon_cond_probe.py:250](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:250).
- No `weights_only`, pickle, or other load argument changed.
- The three-tuple carries the validated provenance to `main`: [depth_canon_cond_probe.py:290](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:290).
- `ckpt_sha256_loaded` and `ckpt_bytes_loaded` are inserted immediately after the legacy path digest and therefore name the deserialized bytes: [depth_canon_cond_probe.py:587](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:587).
- Type strictness comes from `_ckpt_load_provenance`; `None` is explicitly refused because that shared helper intentionally treats `None` as “no additive provenance.”
- The spy test proves one read and one `BytesIO` load; the rewrite-after-read test proves loaded provenance remains bound to the deserialized prefix while the later path digest changes: [test_depth_canon_cond_probe.py:391](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_cond_probe.py:391), [test_depth_canon_cond_probe.py:695](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_cond_probe.py:695).
- Registered/deviation, sidecar, EMA selection, canonicalization, and ARE behavior are otherwise unchanged.

**F3-2 — RESOLVED.**

- `PROBE_CONTRACT["ckpt_step"]` binds the control to CANON@40000: [depth_canon_collect.py:825](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:825).
- Receipt sourcing is correct: only cells already accepted by `verify_cell`, with arm `CANON` and step 40000, contribute receipts: [depth_canon_collect.py:883](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:883).
- No receipt, another digest, non-positive/non-int length, disagreeing lengths, or probe/receipt length mismatch produces a named reason: [depth_canon_collect.py:891](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:891).
- Both legacy `ckpt_sha256` and `ckpt_sha256_loaded` must equal the CANON manifest digest; loaded fields also receive the shared type-strict validation: [depth_canon_collect.py:915](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:915).
- `load_control` requires `canon_receipts`, and `--collect` supplies receipts from the cells just verified: [depth_canon_collect.py:962](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:962), [depth_canon_collect.py:1557](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1557).
- An invalid control is refused in full mode; partial mode labels it `NOT COMPUTABLE`: [depth_canon_collect.py:1270](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:1270).
- Mutation coverage includes every requested missing/malformed/mismatched probe field and receipt condition: [test_depth_canon_collect.py:878](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:878), [test_depth_canon_collect.py:958](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:958), [test_depth_canon_collect.py:1425](/home/yixunhu/codespace/exp-25-depth-canon/src/tests/test_depth_canon_collect.py:1425).

The byte length adds no independent cryptographic guarantee beyond a matching SHA-256 digest: that digest already commits to the exact byte sequence and hence its length. It does add useful cross-artifact/schema evidence that the probe, evaluator record, verified receipt, and driver-measured size followed the intended provenance path. It is correctly sourced from verified receipts, never from the probe’s own file.

**F3-3 — RESOLVED.**

`gen_model_comparison.py` and `test_gen_model_comparison_depth_canon.py` are byte-identical between `6cdcf9f` and `ad03da8`. The generator continues to consume the collector’s `verify_cell` contract and does not duplicate probe validation. The final recorded regression includes the differential golden that preserves every pre-exp25 rendered row.

## Scope and launch safety

- At code HEAD `833ab9a`, exactly four files changed: the probe, collector, and their two tests. The other two scoped generator files are untouched.
- At records HEAD `ad03da8`, the only additional paths are the declared notebook, ledger, and six rB-F3 pytest logs. No unexpected product path changed.
- The regenerated pin manifest is byte-identical to the committed manifest: 48 entries. `sha256sum -c --strict` passed all 48, and no pinned path differs from `6cdcf9f`.
- Launcher, watcher, audit gate, and guard suite are byte-identical to `6cdcf9f`; their SHA-256 values remain respectively `2068af00…`, `c5f2f1bd…`, `452ece21…`, and `7d62e5ae…`.
- Pin generator, pins, CANON config, all non-test `src/`, `eval_FLAC.py`, and `gen_model_comparison.py` are unchanged. No environment/package file changed. Therefore nothing in rB-F3 can affect the training launch.

## TDD and deviations

The immutable RED commits are load-bearing: F3-1 failed 11 tests/42 passed without exact-buffer provenance; F3-2 failed 46/198, including both CLI cases silently succeeding before the collector branch existed. Green subsets report 282 and 299 passed. Commits changed 92, 34, 90, and 58 lines—each below 200. The committed final seven-file log reports 614 passed in 543.83 seconds. The Planner’s 27-file run was still in progress at 11% when inspected and was not used to support this verdict.

Deviations (1)–(6) are accepted: the three-tuple and explicit `None` refusal are fail-closed; `ckpt_step` removes duplicated literals; rejecting an other-digest receipt is sound defense-in-depth; no-CANON selections correctly require `--partial`; the Claude Opus 5.5 trailer is metadata-only; and committing the six logs changes records only.

Open questions: **(a) confirm no probe-side deviation is required**—the artifact exposes both values and the collector rejects either digest mismatch; **(b) confirm every verified CANON@40k receipt, including D-i, is intended**—all name the same verified bytes; **(c) confirm no-CANON selections are partial by construction** because the probe is the campaign-level control.

## Read-only probes run

I used `git status/log/diff/show`, scoped path and blob comparisons, `git diff --check`, manifest regeneration piped to `cmp`, strict verification of all 48 hashes, `sha256sum`, `bash -n`, Python AST parsing, and inspection of the committed RED/GREEN/final logs. I did not run pytest, training/evaluation, the probe, collector, generator, launcher, watcher, driver, audit/guard suites, GPU commands, or touch any process or file.