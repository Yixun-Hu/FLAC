**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-03 · **Round:** code rB-F1 (exp_22 R2-B fixes: gate integrity, A4, executed partition; e837e173 → 1c2b0bfa)

**Verdict:** REQUEST-CHANGES

The three prior BLOCKING findings are resolved. The remaining changes concern evidence self-sufficiency, ledger integrity across resume, and synchronization of the A5 command record.

## C1 — RESOLVED

All three prior blockers are closed.

- Complete probe set: `_probe_cases` refuses an incomplete stream at [localize_meshgrid.py:828](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:828); G3/G0 also carry an exact registered/consumed set clause through [meshgrid_fa_parity.py:268](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:268). Early- and late-deletion tests are present.
- Pinned G0 anchor: the bytes are read once and checked against `8c5fd5e8…` at [meshgrid_fa_parity.py:753](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:753). The driver loads it before the observation-bank digest or replay at [localize_meshgrid.py:754](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:754).
- Identity/provenance: `gate_identity` uses full registered-protocol admission and exact deviation-set equality at [meshgrid_fa_parity.py:380](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:380). Dirty/unstated tracked state, malformed/missing SHA, missing host, or failed CUDA UUID capture make the final verdict non-PASS at [meshgrid_fa_parity.py:417](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:417).

## C2 — PARTIAL

A4’s execution behavior is substantially correct:

- G3 is forced to `--cond-autocast off`, production 256/16, with hard `1e-3`.
- G3D uses resolved production autocast, has `verdict: measurement`, `match: null`, `tolerance: null`, and publishes the 3.9e-3 vanilla caveat.
- Enabled state, dtype, and device type are read through the same autocast-context helper used by the engine.
- Keeping `inputs.binding.cond_autocast = "default"` is acceptable because it is the registered identity token and explicit resolved fields sit beside it. A4(iii) should be clarified to mean “never `default` as the sole execution description,” rather than literally forbidding the token.

One schema defect remains: G3D still publishes the numeric `score_tolerance` because [meshgrid_fa_parity.py:744](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:744) sets it unconditionally. It is not enforced in measurement mode, but the artifact is ambiguous and contradicts “no fp16 tolerance.”

## C3 — PARTIAL

Resolved portions:

- Headline partition values are call-derived from observed base/orbit calls.
- Declared versus executed comparison walks the exact schema both ways.
- Configured strides are consistency checks, not reported execution.
- The ledger is written atomically before each row, accumulated across resume segments, and re-derived before merge.
- `merge_shards` validates each shard before writing the merged directory and preserves a merged counts digest.
- The synthetic BF report fixture’s contract change is acceptable: it is explicitly synthetic and uses the engine ledger functions.

The tail-prefix rule is sound for canonical admission. Rows may be written after a prefix-only check at [meshgrid_engine.py:2486](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2486), while the exact check occurs at finalization through [meshgrid_engine.py:2423](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2423). If it fails, the rows remain on disk, but no valid completed summary/segment is produced and merge refuses them. Thus it is not a numerical-admission hole, although “publishes nothing” is inaccurate.

The durability chain is incomplete: existing segment `counts_sha256` and the stored `merged` block are not validated when reopening a ledger at [meshgrid_engine.py:805](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:805). Merge reconstructs raw counts but checks only the summary’s `executed` and aggregate digest at [meshgrid_engine.py:843](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:843). An edited count that preserves the headline partition can be re-legitimized by a no-op resume, which rewrites the aggregate digest and summary.

## C4 — RESOLVED

The 13 delete-the-check mutations cover the requested clauses:

- score tolerance;
- argmax;
- source-tail coverage;
- replay orbit/partition;
- mask value, dtype, absence, and shape;
- DiT-input-only drift;
- partial query set and stream-end refusal;
- provenance;
- identity;
- anchor pin;
- autocast.

`assert_cacheable` now checks an engaged recorder and a finite, nonempty, identity-first, duplicate-free orbit at [meshgrid_engine.py:2026](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2026). Exact C4 registration remains enforced by driver/protocol admission.

## C5 — PARTIAL

G3/G3D remain quality-blind: no prediction, raw score, margin, similarity, or localization error is serialized. Only deltas, agreement counts/flags, registered identities, candidate identities, and structural evidence appear.

The JSON is not yet sufficient to re-verify every PASS clause alone:

- The driver verifies the pinned report but discards `_anchor` before constructing G3/G3D evidence at [localize_meshgrid.py:890](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:890). The JSON therefore does not establish that its registered query digest belongs to the pinned `8c5fd5e8…` report.
- Each G3 chunk publishes only `record["partition"]`, not the complete replay-partition record whose cap, micro-batch, angle count, and forward count were judged, at [meshgrid_fa_parity.py:675](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:675).
- G3D’s numeric `score_tolerance` makes its no-threshold status ambiguous.

## C6 — PARTIAL

Verified:

- Independent AST count, excluding tests/comments/docstrings:  
  `d6361fc2 +34`, `b18cd9ed +42`, `f382a582 +63`, `65a7b825 +83`, `9831a2a4 +21`, `b6735a7b +48`, `0c696c78 +115` = **+406**.
- All seven code commits are below 200 net code lines.
- All seven code commits and records commit carry the Claude Opus 5.5 trailer.
- Scope is exactly the stated five code/test files; `meshgrid_segments.py` and the other protected modules are unchanged.
- Records report CPU-only work, no GPU launch, and no package/environment operation.
- The Coder’s recorded run is 1,134 passed / 9 environmental skips / 0 failed; collection is 5,029 / 0 errors. I did not rerun pytest.

The command list contains GV, G0–G3, G3D, both probe forms, both fully expanded arm shard/merge sequences with `--expect-arm`, same-pin resume, handoff, report/crossarm, and restart-not-resume.

A5 is now committed on `localization-exp` as `969ae01c` and appears at [plan_loc_meshgrid_r2_bf_yaw.md:103](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:103). However, this branch’s closing handoff still says the conversion is open and omits `× 0.5145` at [loc_meshgrid_worklog.md:499](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:499).

## C7 — NOT RESOLVED

- **Push:** blocked for reviewed release by the evidence and ledger findings above, their focused tests, the Planner rerun, and committed review records.
- **Final integrative review/merge:** blocked until this fix round is corrected and re-reviewed. Preserve committed A1–A5 from `localization-exp` during integration.
- **G0–G3/G3D/probe:** blocked until a corrected, reviewed, pushed, tracked-clean pin is available on a Yixun-released card. G0 remains A6000-class-bound.
- **YAW/BF passes:** require GV and G0–G3 PASS, G3D structurally complete, and the A5 projection `factor × GPU-hours × 0.5145 ≤ 150 h`, followed by complete launch/merge provenance and retrieval-handoff admission.

No new scientific threshold or estimand needs preregistration beyond A4/A5. Before measurements, clarify A4(iii)’s requested-token wording and define the two-lane A5 calculation explicitly as the slower/max lane projection.

## Ranked findings

1. **HIGH — G3D publishes an apparent numeric score tolerance despite being threshold-free.**  
   Set `score_tolerance` to `null` for measurement mode and add a schema assertion covering both tolerance fields.

2. **HIGH — committed G3/G3D JSON cannot independently re-evaluate the anchor and full replay-partition clauses.**  
   Include the pinned anchor digest/query digest in evidence and serialize the complete replay-partition record checked by `replay_partition_reasons`.

3. **HIGH — ledger checksums are written but not validated across resume/merge.**  
   Add one central ledger reader that validates every segment index, nonnegative count, per-segment digest, stored merged digest/executed/violations, and the full summary ledger block. Use it in both `open_fa_ledger` and `fa_partition_census`; add corruption-before-no-op-resume tests.

4. **HIGH — the branch handoff remains pre-A5.**  
   Replace the open conversion note with the committed A5 formula, record the max-lane rule for two-lane probes, and state that the arithmetic/verdict is written before any BF shard starts.

5. **NIT — final partition failure leaves partial row artifacts despite “publishes nothing” wording.**  
   Document these as incomplete, non-admissible artifacts, or stage/quarantine rows until the final exact check succeeds.

6. **NIT — shell preflight examples only print `STOP` and return success.**  
   The tracked-clean and handoff examples at [loc_meshgrid_worklog.md:492](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:492) and [loc_meshgrid_worklog.md:513](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:513) should exit nonzero on mismatch. The downstream code is fail-closed, so this is procedural hardening.

## Read-only verification performed

- `git status`, revision/log/trailer inspection, scoped/full diff stat, name-status, numstat, and `git diff --check`.
- Line-by-line source, test, mutation-log, worklog, ledger, and command review.
- Independent AST-based per-commit production-code counts.
- SHA-256 verification of the probe report (`8c5fd5e8…`) and retrieval handoff (`1bf62c05…`).
- Re-read committed A1–A5 from `localization-exp`; A5 is committed at `969ae01c`.
- No pytest, driver, gate, probe, model import, training/evaluation, GPU, package-management, process-control, or filesystem-writing command was run.
