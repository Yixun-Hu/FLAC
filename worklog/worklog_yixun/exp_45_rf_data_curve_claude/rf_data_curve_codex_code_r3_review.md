**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`) · **Date:** 2026-11-15

Reviewed `90d0202f` read-only. No tests or jobs rerun. The saved final L0 log reports **981 passed, 6 skipped** at `5b1b2f73`.

| R2 | Verdict | Code and pre-fix test assessment |
|---|---|---|
| 1 | **FIXED** | Both regenerations write into B through `--out`. Tests reject the old sheet commands and missing CLI option. |
| 2 | **FIXED** | Credit requires matching `(st_dev, st_ino, st_ctime_ns)` across the bracketing scans; multiply linked files refuse. Shrinking/replacement/deletion and hardlink cases expose the old behavior. |
| 3 | **FIXED** | The collector checks the comparison’s current metrics, stream, receipt and dump-manifest bindings. The new test fails pre-fix first at its missing-binding assertion. |
| 4 | **FIXED** | Bundle files are captured once; hashing and comparison use those captured bytes. Read-count and post-capture mutation tests expose the old separate reads. |
| 5 | **FIXED** | `Record.seal()` precedes marker publication. The injected post-rename fsync failure exposes the old record rewrite. |
| 6 | **FIXED** | Canonical namespace checks precede output mutations. Materialisation exception assessed below. The recorded pre-fix isolation failures are missing-API errors, rather than behavioral demonstrations. |
| 7 | **FIXED** | The one-hour grace prevents immediate abandonment; late jobs face owner and reservation-token checks before training. The test simulates parent death and scheduler outcomes; it does not exercise a surviving subprocess. |
| 8 | **FIXED** | `release --run` checks owner/token, sealed-record bindings and terminal scheduler evidence while preserving the run. Pre-fix tests fail because the operation is absent; current cases cover running and unproven jobs. |
| 9 | **FIXED** | Both materialisers refuse missing tracked destinations and publish with `os.link`, preventing replacement. Tracked/racing destinations fail pre-fix; dangling symlinks were already rejected by r2-6. |
| 10 | **FIXED** | The exception enumerates exactly 24 paths. The exhaustive test exposes the old 96-path acceptance. |
| 11 | **PARTIAL** | A paused live writer is tolerated and crash leftovers refuse. Normal writer completion can still cause a false stale-file refusal; see finding 2. The test misses that transition. |
| 12 | **FIXED** | Launch and SMOKE/FULL admission require flash. The consistent SDPA-chain and launch tests expose the old acceptance. Evaluation’s no-flash overlay remains unchanged. |
| 13 | **FIXED** | Both sbatch interpreter invocations use `-B`. The dynamic pre-fix evidence detects kit bytecode; permitting unrelated stdlib caches is appropriate. |
| 14 | **FIXED** | The sheet supplies three XML inputs and all three checkout roots. Its tests fail on the previous absent command. |

The storage change cannot cause the proposed FULL-start refusal at the supplied capacity. Each reservation is **12.788 GiB**, totaling **38.365 GiB**. At 235 GiB free, zero credit still leaves **196.635 GiB**, above the **30 GiB** floor. Even charging another three complete budgets against that starting capacity leaves **158.269 GiB**.

The grace period does not establish that an orphaned `sbatch` has exited. Its practical safety comes from removing the old owner and checking the reservation token before training. The new `release` operation separately refuses running jobs and missing scheduler evidence.

| R1 PARTIAL | Closure |
|---|---|
| F1 → r2-5 | **CLOSED** — record frozen before possible marker publication. |
| F4 → r2-3 | **CLOSED** for the collector’s compared-byte binding; separate table-consumer gap below. |
| F5 → r2-2 | **CLOSED** — bracketed accounting and hardlink refusal. |
| F7 → r2-7 | **CLOSED** for the supported recovery workflow, with the bounded-grace qualification above. |
| F12 → r2-1 and r2-9 | **CLOSED** — output leaves main’s tracked table untouched; materialisation cannot clobber destinations. |
| F13 → r2-10 | **CLOSED** — exact 24-path publication exception. |
| F14 → r2-11 | **PARTIAL** — writer-completion race remains. |

New findings, most severe first:

1. **HIGH — Required bridge admission is bypassed by both table writers.**  
   [gen_model_comparison.py:1045](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/gen_model_comparison.py:1045), [exp45_table.py:391](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_table.py:391).

   With otherwise valid fraction evidence, a missing bridge comparison—or a missing/corrupt compared bridge artifact—does not prevent the shared generator from rendering numeric RF rows. It validates fraction cells and two-K coverage without reading the bridge. `build()` similarly checks fraction cells against cached collector values without revalidating bridge evidence.

   Thus these publication entrypoints admit a campaign that the collector rejects under the registered bridge integrity criterion. Missing evidence suffices; no crafted aliases or deliberate race are required.

   **Fix:** require a shared publication gate in both writers that validates the sealed comparison against current bridge artifacts and carries its verdict/discrepancies into the output. Cover absent comparison and changed bridge-artifact cases through both entrypoints.

2. **MEDIUM — A completed receipt publication can still reject another healthy SMOKE.**  
   [exp45_registry.py:1539](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:1539).

   `git ls-files` can capture a receipt temp filename. The writer then renames it and unlocks before `_env_receipt_lock_held()` runs. Hygiene reports that now-absent filename as a stale crash leftover because it never rechecks existence. This is ordinary concurrent completion, requiring no interference.

   **Fix:** inspect candidate leftovers while holding the receipt lock, ignoring filenames that have disappeared. Extend the regression to let publication finish between the Git listing and lock inspection.

**Recommendation on r2-6: approve the narrow materialisation exception for the registered G.** The imported farm records land beneath `exp38_data_efficiency/eval/`, separate from live `de_*` run directories and their ownership/lock state. Existing files are preserved or refused; missing tracked files refuse; atomic no-clobber publication protects newly appearing destinations.

The import does populate files that exp_38’s **offline readers in G** consume. It does not alter admission inputs in the separate frozen campaign checkouts. Approval should cover exactly the verified import-file/code set at the registered generator root. Refusing G merely because it contains exp_38’s save root is unnecessary.

The RF argv still follows the exact registered substitutions. I found no additional concrete regression in the frozen runtime paths, main’s tracked output paths, or exp_38’s live training namespace.

**Overall verdict: REVISE — finding 1 remains an open HIGH blocker for Yixun before launch.**