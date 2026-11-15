**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-45-rf-data-curve`) · **Date:** 2026-11-15

**Reviewer: OpenAI Codex · 2026-11-15 · HEAD `a39ded1c`.** Read-only review; no tests, jobs, installations, or file writes performed. The saved final log reports **936 passed, 6 skipped** at `2f597012`; subsequent commits change records only.

| R1 | Verdict | Pre-fix test evidence and current assessment |
|---|---|---|
| F1 | **PARTIAL** | The completion test exposes the original hash mismatch. Normal completion is fixed; an exception after marker publication can still trigger a rewrite. |
| F2 | **FIXED** | Negative admission cases fail under the old substring check. The reused validator and sealed result bind the admitted checkpoint correctly. |
| F3 | **FIXED** | Identically missing, shortened, or changed arrays expose the old equality shortcut. Both hashes are now recomputed from complete arrays. |
| F4 | **PARTIAL** | The old collector ignores absent or invalid comparisons. The new gate works, but its seal does not bind the compared stream bytes. |
| F5 | **PARTIAL** | The red tests expose the old observation order and sparse-file credit. Hardlinks and truncation still permit over-crediting. |
| F6 | **FIXED** | The old build accepts jointly changed cells/rows. The new implementation checks hashes, values, attestations, and campaign against commit blobs. |
| F7 | **PARTIAL** | The named pre-intent and post-archive windows are recovered. A killed submitter can still leave an unprotected, in-flight `sbatch` child. |
| F8 | **FIXED** | Checkout roots are normalized while substantive errors remain distinguishable. The recorded red stops at the newly added API argument. |
| F9 | **FIXED** | The allow-list contains exact node/reason pairs. The recorded red includes missing-symbol/API failures; the current rejection logic is narrow. |
| F10 | **FIXED** | The old launcher ignores the isolated root. The new hook and occupied-production-directory test cover that failure. |
| F11 | **FIXED** | Old single-rank launch/log acceptance fails the new tests. Production launch and receipt paths now require 2×32; the leftover single-rank argv branch is unreachable. |
| F12 | **PARTIAL** | The red tests expose missing transfer APIs/dependencies. Transfers now exist, but regeneration writes a tracked file in main, and copying lacks complete no-clobber protection. |
| F13 | **PARTIAL** | The publication subject now matches and is anchored. The path rule admits 96 filename combinations instead of the registered 24. |
| F14 | **PARTIAL** | The sharpened red observes three receipt writes. The lock prevents partial receipt reads, but the earlier hygiene scan can reject another writer’s temporary files. |
| F15 | **FIXED** | Missing disclosure output fails the new tests. Omitting OrbitRIR-specific item 7 from the RF-only collector is reasonable; published generator rows carry all disclosures. |
| F16 | **FIXED** | The old gate accepts a directory named `hparams.yaml`. The new regular-file and symlink checks reject it. |

For F2, keeping the runtime admission-record hash in the sealed bridge result is sufficient; it need not become a descriptor constant. The added closure dependencies cover the reused validator’s record-only reads: `exp43_kit.py` and its AR manifest. For F3, I independently recomputed both hashes for all ten fixture streams: all arrays contain 6,337 items and match their pins.

I found no submission/ledger lock-order cycle. Persistent lock files alone do not retain a flock after process death.

The new and remaining findings, most severe first:

1. **HIGH — [rf_data_curve_neuronic_commands.md:95](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_neuronic_commands.md:95): both regenerations modify main’s tracked table.**  
   `--repo-root $G` selects both evidence and output. The generator writes `$G/worklog/worklog_yixun/model_comparison.md`; copying it into B afterward does not prevent that write. The second regeneration repeats it.  
   **Fix:** provide a separate output destination in B while reading evidence from G, and publish both regenerations from B.

2. **HIGH — [exp45_registry.py:958](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:958): written-byte credit remains unsafe for hardlinks and shrinking files.**  
   Each hardlink earns the same allocation credit again, including links across reserved runs. Also, an 8 KiB file scanned before truncation still earns 8 KiB against free space sampled afterward. With 25 KiB then free, a 20 KiB reservation appears to leave 13 KiB instead of the actual 5 KiB. Sparse and preallocated files are conservatively handled at the observation instant; these cases are not.  
   **Fix:** deduplicate or reject multiply linked files and credit only allocation guaranteed to remain consumed through admission. Cover truncation, replacement, and cross-reservation hardlinks.

3. **HIGH — [exp45_eval.py:1291](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_eval.py:1291): a stale bridge stream can still publish as equal.**  
   The comparison binds current metrics bytes but records no hash of the compared exp_45 stream. After comparison, remove its arrays while retaining the declared pins and refresh its receipt/dump-manifest cross-links. Offline `validate_cell()` accepts that stream, and the comparison’s metrics binding remains valid. Collection can publish equality despite missing comparison evidence.  
   **Fix:** seal and recheck the compared stream hashes, preferably with the complete cell artifact bindings. Add a post-seal stream-mutation regression.

4. **HIGH — [exp45_eval.py:1225](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_eval.py:1225): bundle hashes can describe different bytes from those compared.**  
   `load_bridge_bundle()` hashes references, then `cmd_bridge()` separately opens them. A reference changed between those operations can determine the verdict while the result records its earlier manifest hash. There is no comparison-read hash check for those reference files.  
   **Fix:** read each reference once, verify those captured bytes against the manifest, and compare the parsed captured bytes.

5. **HIGH — [exp45_supervise.py:667](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_supervise.py:667): publication can precede `Record.seal()` even when the writer raises.**  
   `atomic_write_bytes()` replaces the marker JSON before its directory fsync. If that fsync raises, the complete marker and sidecar already exist, but `rec.seal()` has not executed. The outer exception handler rewrites the terminal record and invalidates the published marker. The release-error tests do not exercise this window.  
   **Fix:** freeze the terminal record before marker publication, or preserve it whenever marker publication may have occurred. Inject a post-rename failure in the regression.

6. **HIGH — [exp45_submit_neuronic.py:204](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_submit_neuronic.py:204): output-root aliases can defeat exp_38 isolation.**  
   The dedicated path names are not checked for physical aliasing. If the exp_45 save root points to exp_38’s save root, submission creates locks, ledger entries, owners, and run directories there. Likewise, an aliased dump root receives the write probe and new dumps. Filesystem-type checks permit both cases.  
   **Fix:** validate canonical output namespaces against protected exp_38 roots before any mutation, including probes and locks. Apply equivalent containment checks to materialisation destinations.

7. **MEDIUM — [exp45_submit_neuronic.py:228](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_submit_neuronic.py:228): the submission lock does not survive a live `sbatch` child.**  
   Killing the Python submitter releases its lock while its ordinary subprocess can continue submitting. `abandon` can then see a `submitting` intent, no recorded job ID, and empty scheduler queries, archive the owner, and release the hold before that submission resolves.  
   **Fix:** retain ownership through the submission subprocess, durably record its outcome, and keep the hold when an issued submission has an unknown outcome. Test parent death while `sbatch` remains alive.

8. **MEDIUM — [exp45_submit_neuronic.py:449](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_submit_neuronic.py:449): releasing a stranded sealed hold requires archiving the successful run.**  
   A failed post-seal release conservatively retains the hold. Terminal `abandon` can release it, so this is not an unavoidable lock deadlock, but it moves the successful run away from its registered path. Residual outstanding credit can block later admissions without a supported preservation path.  
   **Fix:** add an idempotent release operation that validates the seal, token/owner binding, and terminal scheduler state while preserving the run directory and records.

9. **MEDIUM — [exp45_table.py:289](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_table.py:289): materialisation can overwrite an existing destination.**  
   `os.path.exists()` treats a dangling destination symlink as absent, and the subsequent replacement overwrites it. Both materialisers also check existence before calling the replacing writer, allowing a destination created or restored between those operations to be clobbered. Neither protects tracked-but-missing destinations through the Git index.  
   **Fix:** use `lexists`, check tracked destinations, and publish absent files with an atomic no-clobber primitive.

10. **MEDIUM — [test_exp45_merge_provenance.py:74](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/src/tests/test_exp45_merge_provenance.py:74): the E0 exception admits unregistered paths.**  
    Epoch `{34,8}` and the optional FA suffix are independent of the gate. For example, E0a at epoch 8 and E0b without `_fa_invariant_a4` pass. Subject and path matching are anchored, but the accepted path set is four times too large.  
    **Fix:** enumerate the exact 24 paths or enforce each gate’s registered epoch and conditioning suffix.

11. **MEDIUM — [exp45_supervise.py:403](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_supervise.py:403): concurrent SMOKEs can fail hygiene before reaching the receipt lock.**  
    The scan runs before `ensure_env_receipt()`. Another writer’s `.rf_data_curve_env_receipt_neuronic.json[.sha256].tmp.<pid>` is neither ignored nor allowed, so the arriving SMOKE refuses. A killed writer can leave this obstruction behind. The concurrency test mocks `check_untracked()` away.  
    **Fix:** coordinate hygiene with publication or narrowly handle the kit’s temporary record names, including crash recovery. Test the real hygiene gate during a paused write.

12. **MEDIUM — [exp45_registry.py:2043](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_registry.py:2043): primary FULL admission still accepts SDPA training.**  
    Submitting SMOKE and FULL with `--backend sdpa` can produce an admitted primary run: the gates validate SDPA evidence and the collector accepts it. The registered recipe requires flash, and the disclosures do not register this additional training-backend difference.  
    **Fix:** refuse non-flash primary SMOKE/FULL launches and receipts, or register a separate diagnostic namespace that cannot populate primary rows.

13. **NIT — [exp45_train.sbatch:44](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_train.sbatch:44): imported exp_38 tools can write bytecode into their folder.**  
    The launcher unsets `PYTHONDONTWRITEBYTECODE`; the supervisor then imports the reused tools from exp_38’s folder. A fresh launch checkout can acquire `__pycache__` files there. Tracked bytes remain frozen, but the literal no-write requirement does not hold.  
    **Fix:** run the importing driver with bytecode writing disabled.

14. **NIT — [rf_data_curve_neuronic_commands.md:38](/n/fs/gatrdp/codespace/exp-45-rf-data-curve/worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_neuronic_commands.md:38): neither command sheet contains an L1 comparator invocation.**  
    There is consequently no sheet command supplying the three newly required root arguments. The CLI docstring is current.  
    **Fix:** add the concrete L1 comparison command with all three XML inputs and `--merged-root`, `--exp38-root`, and `--exp43-root`.

**Overall verdict: REVISE.**