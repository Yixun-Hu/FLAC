**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-03 · **Round:** code rB-F2 (exp_22 R2-B fixes 2: evidence self-sufficiency, ledger integrity, A5/A6 command list; 1c2b0bfa → 26436759)

**Verdict:** REQUEST-CHANGES

The command-list/A5 closure and G3D tolerance fix are sound, but evidence self-sufficiency and ledger integrity still have blocking fail-open cases.

## Findings, ranked

1. **BLOCKING — the merged BF ledger is invalid under its own reader, and canonical reporting does not verify it.**  
   [`fa_partition_census`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1017) copies shard segments while retaining each shard-local `segment_index` ([line 1029](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1029)). The merge then writes those flattened segments unchanged ([line 3126](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:3126)). Two one-segment shards therefore produce indices `[0, 0]`, which `_ledger_reasons` correctly rejects as not `0..1`. My pure probe reproduced that exact refusal.

   Worse, [`evaluate_run`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:2387) checks only the failure marker at line 2433; [`assert_merge_report`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:641) ignores both the merged ledger and `merge_report.json.fa_partition_executed`. Thus a merged BF artifact’s ledger or partition receipt can be changed after merge without invalidating the canonical per-arm or inherited crossarm report.

   **Fix:** globally renumber merged segments while preserving `shard_segment_index`, make the merged ledger reader-valid, and have `evaluate_run` re-read it exactly and compare its re-derived digest/executed block/per-shard census with `merge_report.json`. Vanilla reporting must conversely require no FA ledger or receipt.

2. **BLOCKING — the pinned anchor is asserted, not authenticated from the JSON.**  
   [`anchor_reasons`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:325) verifies that the evidence *claims* SHA `8c5fd5e8…`, then accepts any unique query list whose digest and count are internally consistent. It has no registered expected query-list digest and no serialized report bytes to connect those IDs to that SHA. I supplied an arbitrary query ID, recalculated its digest, retained the pinned SHA strings, and `anchor_reasons(..., True)` returned `[]`.

   G0 has the same deeper gap: [`ANCHOR_EVIDENCE`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:882) excludes the frozen digest map, while [`_reverify_replay`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1194) merely checks `frozen_digests == published_digests`. Both sides can be replaced together. Therefore G0’s two additive fields are not sufficient, and a coordinated “swapped anchor” edit can still re-verify.

   **Fix:** bind the JSON to registered anchor-content constants or include/hash the complete relevant anchor content. The current committed file yields `query_ids_sha256 = 4f5743a8…` and frozen digest-map SHA `9ecef47d…`; both should be independently pinned and checked, or the exact serialized anchor records should be included and hashed back to the registered report.

3. **HIGH — reopen admission still permits missing or truncated execution evidence.**  
   [`read_fa_ledger`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:940) returns `None` when both ledger and summary block are absent ([lines 961–966](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:961)), even if an existing binding and rows show that this is not a fresh directory. Deleting both files can therefore bootstrap a replacement ledger over existing rows.

   The summary-prefix rule also authenticates only internal consistency. A truncated tail segment with recomputed self-digests and no summary can pass `final=False`; later full-sized calls can saturate the maxima and make the final exact partition pass despite the lost call counts. My pure probe demonstrated both the prefix acceptance and the subsequent exact pass. A genuinely disagreeing summary is correctly refused, but deleting the summary removes that anchor. Segment indices also accept `False`/`True` as `0`/`1`.

   **Fix:** refuse a missing ledger whenever binding, rows, launch state, or another run artifact exists; require real integer segment indices; and bind each segment state to its published rows. A practical design is to stamp each row with that segment’s cumulative `n_queries` and counts digest, allowing only a single one-query-ahead state for an interrupted final write.

4. **HIGH — `reverify_gate_evidence` does not always return named reasons, and several required structural clauses remain weak.**  
   The four clause calls are guarded by `MALFORMED`, but `_verdict_reasons` is invoked outside that protection ([lines 1253–1265](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1253)). `{"gate":"G3","verdict_clauses":"malformed"}` raises `AttributeError` instead of returning a reason.

   In addition:

   - [`autocast_reasons`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:372) ignores `cond_autocast_device_type` and accepts any nonempty dtype. My probe showed missing device type plus dtype `"banana"` returns `[]`.
   - G3D “complete” does not require score deltas to be finite numeric measurements; it only skips the threshold predicate.
   - `inputs_reasons` requires a nonempty DiT-input map, not the exact registered key set.
   - G3/G3D raw executed counts are not checked for canonical shape, nonnegative integers, or booleans as the ledger counts are.

   **Fix:** validate the full evidence schema and wrap verdict verification too; check exact resolved dtype/device against `inputs.device`; add threshold-free finiteness/type checks for G3D score facts; register exact DiT input keys; and reuse canonical counts validation for executed evidence.

5. **NIT — a hard kill between detection and marker publication is fail-open.**  
   Both synchronous failure paths write the marker ([final check](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2581), [before-row check](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2648)). But if the process is killed after the check fails and before `write_partition_failure` completes, the before-row ledger still holds the previous good prefix, while the final ledger can be accepted under the prefix rule. Reopen does not necessarily catch either case.

   **Fix:** atomically publish a pending-check sentinel before each check and clear it only after success. A kill may then conservatively force a restart, which is consistent with A6(iv).

## C1–C9

- **C1 — RESOLVED.** G3D publishes `tolerance`, `score_tolerance`, and `match` as null in report/evidence/JSON; G3 retains `1e-3` and `SCORE_TOLERANCE`. No numeric threshold is applied to G3D tensor or score deltas. Its remaining structure checks are not implicit tolerances.

- **C2 — NOT RESOLVED.** The function is pure with respect to runtime artifacts: it imports engine/report/crossarm code and registered constants but reads no model, tensor, anchor file, or other data file. It substantially replays the six gates’ current predicates, summaries, identity, and provenance checks. However, the anchor contents and G0 digest map are not tied to the pin, malformed verdict fields can escape as exceptions, and the autocast/device, DiT-key, executed-count, and G3D-finiteness clauses are incomplete. Simple flipped verdicts/booleans, above-threshold G3 deltas, missing chunks, invalid anchor SHA, and missing provenance are caught; a coordinated swapped anchor or valid-looking substituted provenance is not authenticated.

- **C3 — NOT RESOLVED.** The reader correctly checks the marker, JSON object, declaration, segment ordering, nonnegative count fields, canonical count rows and digests, unknown branches, merged block, summary ledger metadata, exact/prefix summary blocks, declared partition, and vanilla absence. Tested corruptions are refused before writes and at source-shard merge. But the existing-run/no-ledger case, recomputed truncated prefix, boolean indices, invalid merged output, and absent report re-verification remain holes.

- **C4 — RESOLVED.** The authoritative entry at [`loc_meshgrid_worklog.md:567`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:567) closes A5. It reads the correct `projection.projected_gpu_hours.total`; one-card arithmetic is `1.5 × h × 0.5145`; two-lane arithmetic is `max(h0,h1) × 0.5145`; both exit nonzero above 150 h. Evidence, arithmetic, and verdict must be recorded before BF, with stop-and-ask above cap. The order is gates and probe, then YAW on both cards, then BF. I extracted the old/new inline commands: all 21 old explicit command snippets recur verbatim except the intentionally replaced handoff check; the two-lane commands are now fully expanded.

- **C5 — PARTIAL.** Every ordinary Python failure at final/before-row writes the detailed marker; marker refusal covers reader/reopen, `_read_shard`, per-arm report, and inherited crossarm admission. Probe exemption, wording, and retained incomplete rows are correct. The one-line report edit is necessary and acceptable. The hard-kill window remains.

- **C6 — RESOLVED.** Script preflights, evidence re-verification, probe lane status, arithmetic, marker checks, and handoff checks fail nonzero under `set -euo pipefail`. The interactive-shell subshell instruction is explicit.

- **C7 — PARTIAL.** A6(ii) arithmetic and A6(iii) null fields are correct. Every gate receives the four resolved autocast fields through `_gate_inputs`. For G1/G3/G3D there is an autocast verdict clause, but it does not actually validate device type or the resolved dtype value. A6(iv) is enforced for synchronous failures but not the hard-kill/reopen cases above.

- **C8 — PARTIAL.** SOP records, RED→GREEN cycles, CPU-only claims, no-environment-change statement, nine Opus trailers, commit ledger, and dated worklog entries are present. My independent AST count matches the ledger:

  | Commit | AST code delta |
  |---|---:|
  | 82c73c1a | +2/−1, net +1 |
  | 50ce4e33 | +34/−9, net +25 |
  | 787b2efa | +60/−40, net +20 |
  | 4b2af1f5 | +154/−3, net +151 |
  | 9f60dede | +115/−3, net +112 |
  | 659e0d4b | +116/−39, net +77 |
  | fc8836a7 | +53/−7, net +46 |
  | 3163658c | +6/−3, net +3 |
  | 01f693cb | 0 |

  Range total is +532/−97, net +435. Largest raw non-test Python commit is 659e0d4b at 199 changed lines; 4b2af1f5 is 198.

  The recorded 20/20 mutations are real but not exhaustive. No delete mutation exercises the new anchor-content binding, malformed outer verdict path, exact autocast dtype/device, G3D finite measurement, missing-ledger existing run, prefix truncation, merged-ledger validity/report join, or kill-window sentinel. Several implemented re-verifier paths—registered-input equalities, chunk tiling, digest chains, coverage/score summaries, executed counts, and much of G0/G1/GV—also have tamper tests but no individual delete-the-check mutation.

- **C9 — NOT READY.**
  - **Push:** do not push this as the reviewed integration candidate until the blocking fixes are committed and re-reviewed. The existing untracked Codex artifacts do not themselves affect a Git push.
  - **Final integrative review/merge:** blocked on C2/C3/C7/C5 fixes and another focused fix review.
  - **GV/G1/G2/G3/G3D/G0 and BF probe:** do not run them at this pin; fixes will change the reviewed SHA and the evidence contract.
  - **YAW/BF passes:** additionally wait for the final integrative SHIP, Yixun’s go, released cards, successful gates, and the BF cap verdict.
  - **Further preregistration:** none is required if the fixes remain validation/integrity-only. A1–A6 already determine thresholds, estimands, and scheduling arithmetic.

## What I verified

Read-only probes included `git status`, `rev-parse`, log/trailer inspection, scoped `diff --stat`/`--numstat`/`--check`, changed and untouched file inventories, independent AST counting from Git objects, SHA-256 checks of the anchor and handoff, source/test/worklog inspection with `rg`/`sed`, and CPU-only pure-function probes with bytecode writes disabled and CUDA hidden. I did not run pytest, drivers, gates, probes, training/evaluation, or GPU commands, and did not modify any file or process.
