**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`) · **Date:** 2026-11-14

Reviewed r2 against the exp_38 kit at `df849937`. No files or environments changed; no tests or jobs run. **No BLOCKING findings remain.** Four concrete changes are still needed.

“RESOLVED” below means the plan contract is corrected; implementation remains subject to the planned code review.

| Round-2 finding | Status | Verification |
|---|---|---|
| r2-1 | **RESOLVED** | §3.2 sets Tree 45’s exact campaign root and redirects all reference readers/enumerators through `REF_DIR`, matching the checks in `exp38_eval.py:315,335,664,884`. Shared card locks are preserved. |
| r2-2 | **RESOLVED** | The CSV move is explicitly FULL-only, with a separate SMOKE check. This matches the completion branches in `exp38_supervise.py:632,645`. |
| r2-3 | **RESOLVED** | Bridge enumeration, runner/checkpoint dispatch, receipt admission and collection are connected explicitly. These cover the inherited call sites at `exp38_eval.py:560,831,898` and `exp38_collect.py:163`. |
| r2-4 | **RESOLVED** | Separate publication worktree, all 24 backing E0 files, unchanged closure, and matching SMOKE/FULL training HEADs address `cmd_run()`’s HEAD restriction at `exp38_eval.py:909`. |
| r2-5 | **PARTIAL** | Runtime commit pin and separate historical worktree are correct. The offline API and portable bundle remain incomplete; finding 1 below. |
| r2-6 | **RESOLVED** | A new two-parent comparator is explicitly specified, replacing reliance on the incompatible single-baseline comparator. |
| r2-7 | **PARTIAL** | Atomic reservation addresses the race, but the FULL-start formula counts the candidate twice; finding 2. |
| r2-8 | **PARTIAL** | Generator/test scope and regeneration timing are corrected. Generated-table scope and exp_38 evidence placement remain incomplete; findings 3–4. |
| r2-9 | **RESOLVED** | Header uniqueness is explicit, and the `src/` wording now exempts the declared test changes. |

The round-1 items previously marked PARTIAL are:

| Round-1 item | Status | Reason |
|---|---|---|
| 2 | **RESOLVED** | RF registry, evaluator and collector changes are enumerated. |
| 5 | **RESOLVED** | Parser compatibility edits and the dedicated two-parent comparator are specified. |
| 6 | **RESOLVED** | References, decision bytes and every reference reader are covered. |
| 7 | **RESOLVED** | The external bridge now has an execution and admission path. |
| 8 | **PARTIAL** | Historical identity is preserved; portable offline import still needs finding 1’s fix. |
| 10 | **RESOLVED** | Unique, order-independent headers and separate FULL/SMOKE handling are specified. |
| 11 | **PARTIAL** | Findings 3–4 still prevent the required shared-table publication. |
| 15 | **PARTIAL** | Filesystem budgets and reservation locking are specified; reservation rechecking needs correction. |

1. **HIGH — remaining r2-5: the portable import cannot run as specified.**  
   [Plan §3.2, lines 169–178](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:169).

   **Scenario:** `exp38_collect` has no offline mode. Its [`collect()`](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:161) invokes online E0 admission and artifact enumeration that opens NAS checkpoints, dumps and other external files. The cited generator test instead calls the evaluator’s offline validator.

   The bundle also omits explicit `*.dump_manifest.json` copies, which [`validate_cell(..., offline=True)`](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:601) still opens. Offline training admission additionally reads retained training snapshot manifests, train/SMOKE logs, evidence, checkpoint manifest, completion marker and terminal launch record; these cannot all be treated as external attestations.

   **Fix:** before freezing `X`, specify an isolated import adapter using `validate_e0_receipt(offline=True)`, `validate_cell(..., offline=True)` and pure aggregation. Enumerate and publish its complete local dependency set, including dump-manifest copies. Keep NAS checkpoint and waveform payloads as attestations. Verify import with NAS unavailable and refusal when a required local manifest is missing or changed.

2. **MEDIUM — new in r2: reservation rechecking subtracts the current run twice.**  
   [Plan §3.2, line 110](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:110).

   **Scenario:** submission inserts the candidate into the ledger. At FULL start, the “same inequality” subtracts both its outstanding reservation within the sum and another candidate allowance. For example, 70 GiB free with three 13-GiB reservations satisfies the 30-GiB floor at submission, but the repeated formula reports only 18 GiB and refuses.

   **Fix:** subtract the candidate once before insertion; at start, subtract only the ledger’s outstanding reservations. Make rechecks idempotent and test submission through FULL start, including partially written runs.

3. **MEDIUM — remaining r2-8: the generated comparison table is still outside the allowed change scope.**  
   [Plan §3.1, lines 73–76](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:73), versus §3.2 lines 189–191.

   **Scenario:** regeneration must commit `worklog/worklog_yixun/model_comparison.md`, but the exhaustive allowlist permits the generator and its test, not that output. The required publication therefore fails the plan’s own path audit.

   **Fix:** permit that exact generated file in qualifying results-publication commits.

4. **MEDIUM — new in r2: importing into a temporary worktree does not populate the shared generator’s exp_38 rows.**  
   [Plan §3.2, lines 172–190](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md:172).

   **Scenario:** `import-exp38` writes a sealed derived record after reading a temporary worktree. The shared generator instead [globs raw exp_38 farm paths](/n/fs/gatrdp/codespace/exp-38-campaign/worklog/worklog_yixun/gen_model_comparison.py:188) beneath its selected root. It never consumes that import record. Regeneration #2 can therefore leave all twelve exp_38 rows pending despite a successful import.

   **Fix:** explicitly materialize the hash-verified portable exp_38 farm bundle at its exact relative paths in the neuronic generator root, record its source commit, and require all twelve rows to populate before publication. Preserve the frozen campaign checkouts.

APPROVE-WITH-CHANGES