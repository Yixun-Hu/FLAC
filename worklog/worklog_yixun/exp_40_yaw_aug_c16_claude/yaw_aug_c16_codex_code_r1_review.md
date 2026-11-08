**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_40 code review 1 of 3 (HEAD 15f73ff5)

**Verdict:** REQUEST-CHANGES

The treatment implementation itself is narrowly correct, but the experiment is not launchable or publishable yet. The main blockers are provenance of the four-pin golden, neuronic submission durability, interruption cleanup, admission validation, and multiple evaluation paths that can either fail before the first cell or return success without complete verified results.

## Q1 — Treatment correctness and absent-key equivalence

The production delta implements plan §2 correctly:

- A uniform draw over 512 columns followed by `32 * floor(d / 32)` is exactly uniform over the 16-point grid.
- The factory passes no new keyword when `offset_step` is absent.
- The constructor parameter is appended last.
- The quantization branch is guarded and occurs after the original draw.
- The C16 config is the reference config plus exactly `"offset_step": 32`.
- `training_step` applies yaw augmentation; validation and test paths do not.
- Canonicalization remains mutually exclusive with yaw augmentation, and ARE remains absent.
- `eval_FLAC.py` constructs the wrapper but does not call `training_step` or `_apply_yaw_aug`.

I found no production path that silently trains a different treatment when the pinned C16 config and launcher contract are actually enforced. However, the claimed historical absent-key proof is not presently established: the committed fixture can be assembled from captures made by another tool and labelled with arbitrary commit SHAs. The “independent” pre-rotated test also feeds the production rotation into the control path.

## Q2 — Kit and fail-closed behavior

The run-tree, config, sealed-HF, quick AR, prefigure, evidence instrumentation, and checkpoint inventory designs are generally strong. I also verified from the installed PL 2.1 source that:

- `ModelCheckpoint._save_checkpoint` is entered on every rank.
- `Trainer.save_checkpoint` contains the distributed barrier.
- `strategy.broadcast` supports the tuple used by the publication hook.
- TorchElastic detection has priority over SLURM detection.

The kit nevertheless has blocking fail-closed gaps: state transitions are not durably directory-fsynced or internally locked; claims are scoped to a checkout rather than a canonical shared control directory; the claim-probe receipt is almost unvalidated; admission trusts a shallow evidence header and rehashes only the final NAS checkpoint; and interruption can strand both the owned training process and GPU locks. The actual CIFS publication primitives are not preflighted on the destination filesystem.

## Q3 — Launchers

Normal success/failure sequencing is coherent, and the local `setsid` ownership check is directionally correct. Crash and signal behavior is not acceptable:

- There are no `EXIT`, `INT`, or `TERM` traps.
- Ctrl-C while waiting can let the separate training session continue.
- Early failures after lock acquisition can leave locks behind.
- Final unlock failure is only a warning, so the launcher can report PASS with stale locks.
- The watcher sends one SIGTERM and then returns; it does not wait or escalate for its owned process group.
- The neuronic GPU check assumes physical indices `0..N-1` correspond to the Slurm allocation.

The torchrun/PL TorchElastic choice itself is consistent with the installed PL 2.1 code.

## Q4 — Evaluation

The generated 85-cell grid and argv derivation match the planned exp_25 protocol, and the off-grid C16-vs-YAW formula is implemented correctly. The operational driver and evidence gates are not yet reliable:

- It can dirty the detached worktree with `__pycache__` before its own per-spawn cleanliness check.
- Two lanes race the same checkpoint cache and hard link.
- An uncaught worker exception can still produce overall exit 0.
- E0 reads mutable, unpinned exp_25 references.
- The C16 admission check accepts any JSON with `admitted: true`, `arm: C16`.
- Collection does not bind receipts to current EVAL_SHA, closure, or runtime-input manifest.
- The `verify` command does not verify artifacts.
- Comparator registration and the table validator check only a fraction of their declared contracts.
- Collection omits P1’s off-grid `d`, the promised per-angle tables, and the paper-import CSV.

The `gen_model_comparison.py` addition is structurally additive and does not alter existing row specs, but its exp_40 validator is too weak to authorize publication.

## Q5 — Tests

The core helper/config tests are useful, but several assertions are vacuous or test a weaker predicate than their name/documentation claims. Most importantly, there is no stub runner test for `run_cells`, no worktree-cleanliness/bytecode test, no real E0 reference-mismatch test, no forged-admission negative test, no admission rehash test, and no concurrent cache or submission-state test.

I did not start another pytest process because the user-reported full CPU pytest run is still in progress.

## Q6 — Other issues

The implementation has accumulated several parallel representations of the same trust boundary—check JSON, evidence receipt, audit, admission, runtime manifest, cell receipt, registry—without one shared strict validator. That duplication is the main source of the fail-open/self-consistent-but-untrusted paths below. The command sheets also describe outputs and guarantees the code does not currently produce.

# Findings

## BLOCKING

1. **The four-pin absent-key golden is not provenance-bound to the final capture tool or the claimed commits.**  
   `src/tests/exp40_yaw_aug_golden_capture.py:262`, `:271`, `:282`, `:298`; `src/tests/test_yaw_aug_c16.py:73`, `:277`; `worklog/.../yaw_aug_c16_2026-11-07_21-21-00_E1_golden_capture_four_pins.log:2`

   `capture` records only a caller-supplied source path, torch version, and record. `assemble` accepts an arbitrary `<sha>=file` label without proving that file was captured from that SHA, and it inserts the assembler’s current tool hash rather than the tool hash used by each capture. The only committed capture log names draft tool hash `c120cf…`; the fixture claims final tool hash `69d92e…`. Python version is absent.

   **Required change:** make each capture record and validate the exact source commit/blob manifest, Python and torch versions, and the capture-tool hash actually executing the capture. Make assembly reject duplicate/unrecognized pins and mismatched tool/source identities. Recapture all four pins using the final committed bytes and commit the per-pin capture evidence.

2. **The neuronic at-most-one guarantee is neither globally scoped nor durably serialized.**  
   `exp40_submit_neuronic.sh:39-41`, `:89`; `exp40_kit.py:170-177`, `:1699-1748`

   Claims, state files, and `.submit.lock` live below the submitting checkout’s `outputs_FLAC`. Two checkouts can therefore submit the same RUN_ID using different locks, states, save roots, and claim directories, allowing both jobs to train. Within one checkout, `neuronic_submit` has no internal shared flock, and `atomic_write_json` does not fsync the parent directory, so the binding `sbatch_attempted` boundary is not durable.

   **Required change:** place claims and submission states in one fixed, canonical shared control directory independent of the checkout. Hold a flock there across reconciliation, state creation, `sbatch_attempted`, `sbatch`, and submitted-state recording. Fsync the parent after every state rename and add real cross-process/cross-checkout contention tests.

3. **The two-node claim probe does not prove or bind the property the FULL gate relies on.**  
   `exp40_claim_probe.sbatch:23-49`; `exp40_kit.py:1994-1997`

   The wall-clock modulo loop is not a barrier and can be skewed across hosts. Only the batch shell records a mount view. The receipt is written non-atomically, without fsync, and the submitter checks only `verdict == PASS`; it ignores hosts, rounds, mount source/type, tested directory, job ID, age, script SHA, and source SHA. A stale or fabricated PASS from another filesystem is accepted.

   **Required change:** use a real inter-node barrier for every round, record the claim-path mount identity from both tasks, and durably write a receipt containing exact hosts, 20 rounds, job/source/script identities, tested directory, and timestamp. Strictly validate all fields and freshness before FULL submission.

4. **The launchers have no safe interruption cleanup and may report PASS with stale locks.**  
   `exp40_train_local.sh:107-118`, `:132-170`; `exp40_train.sbatch:72-109`; `exp40_kit.py:982-1011`

   Ctrl-C or TERM while the shell waits can leave the `setsid` training session running and leave GPU locks held. Unexpected exits after lock acquisition have the same problem. Final unlock failure is only a warning. The watcher sends SIGTERM once and returns without confirming that its owned process group exited.

   **Required change:** install `EXIT`, `INT`, and `TERM` traps immediately after acquiring resources. Track whether the child/PGID and watcher were created; signal only the owned PGID, wait with a bounded grace period, escalate only that PGID if needed, reap both children, and release only owned locks. Any cleanup or unlock failure must make the launcher fail.

5. **Audit/admission can accept a fabricated or stale FULL evidence receipt.**  
   `exp40_kit.py:1455-1464`, `:1467-1509`, `:1512-1518`

   `full_evidence_reasons` validates only four top-level strings. It does not check platform, rung, train SHA, tree/environment/recipe digests, reference receipt, rank count, summaries, offset evidence, finite-loss flag, checkpoint acknowledgements, or exit fields. Audit and admission then trust its file hash.

   **Required change:** add one strict FULL-receipt validator covering the complete schema and every acceptance field, including binding to the check record and DDP_SMOKE/PILOT receipt SHA. Use it from both audit and admission, and add mutation tests for every critical field.

6. **Admission rehashes only the final NAS checkpoint and lets caller metadata override identity.**  
   `exp40_kit.py:1519-1553`

   Admission compares the NAS manifest to the old audit, then rereads only step 40k. Any of the other 15 files may change after audit and admission will still pass. `{"admitted": True, "arm": "C16", **meta}` also permits caller metadata to override `admitted` and `arm`; platform/rung/train SHA are not derived from evidence, and the promised run-tree digest is absent.

   **Required change:** rehash all 16 NAS checkpoint files during admission, reject extras/partials, and compare every byte hash to the manifest and audit. Accept an exact metadata schema and derive run ID, platform, rung, train SHA, run-tree digest, evidence SHA, and audit SHA from validated records rather than caller claims.

7. **The evaluation driver can dirty its detached worktree before its first spawn and then refuse itself.**  
   `exp40_eval.py:321-329`, `:482-506`

   `stem_of` imports `eval_FLAC` from the detached worktree in the driver process without disabling bytecode. The documented command does not export `PYTHONDONTWRITEBYTECODE`. This can create `__pycache__` under the eval worktree; the subsequent `eval_worktree_reasons` call then detects an unallowlisted untracked file and aborts before the first evaluator spawn.

   **Required change:** disable bytecode in the driver before any eval-tree import, or calculate stems in an isolated `-B` subprocess. Add a real detached-worktree test proving repeated stem generation leaves `git status` clean.

8. **Two-lane evaluation races the checkpoint cache, and worker crashes can still return success.**  
   `exp40_eval.py:271-305`, `:482-534`

   Both worker threads can observe a missing cache entry, use the same PID-based temporary name, race `RENAME_NOREPLACE`, and race creation of the arm hard link. Any exception in a worker is uncaught; Python prints a thread traceback, but `run_cells` may return an empty failure list and `main` returns 0 with cells unprocessed.

   **Required change:** serialize each content-addressed checkpoint/link with a per-SHA lock and unique temporary name, adopting an identical winner. Wrap every worker iteration in exception handling that records a failure, and finish by proving that every selected cell has exactly one verified receipt before returning success.

9. **E0 compares against mutable external files rather than committed reference hashes.**  
   `exp40_eval.py:91-99`, `:332-340`, `:428-444`, `:788-798`, `:905-909`

   The exp_25 metrics, stream files, input hashes, and waveform hashes are read directly from mutable absolute paths. No expected SHA is declared or verified. The E0 receipt contains only verdict, EVAL_SHA, closure, cell names, and evaluator pins; it omits reference hashes, environment, checkpoint/config identity, input-hash map, and per-cell receipt hashes.

   **Required change:** commit a reference manifest for the three E0 cells and all expected-hash inputs, validate every referenced byte before comparison, and bind the complete reference/environment/checkpoint/config identity plus the three new cell receipts into the durable E0 receipt.

10. **Post-E0 evaluation and table publication accept self-consistent but unadmitted evidence.**  
    `exp40_eval.py:260-267`, `:636-668`, `:766-785`; `gen_model_comparison.py:517-531`, `:895-909`

    Any hash-pinned JSON with `admitted: true`, `arm: C16`, a checkpoint path, and SHA is treated as admission. Collection re-verifies cells against the receipt’s own `head` and checkpoint fields but does not require current EVAL_SHA, closure, runtime-input SHA, E0 receipt, or validated admission. `validate_row` checks only that metrics bytes equal a receipt field; it does not validate stream bytes, receipt identity, admission, exact path, closure, or five distinct seeds.

    **Required change:** implement strict admission and cell-receipt validators shared by run, collect, figure, and table publication. Require exact EVAL_SHA/closure/runtime manifest/E0/admission identities and exactly the registered five T cells per K. Commit and validate the admission plus metrics, stream, and receipt manifest for offline table regeneration.

## HIGH

1. **The “independent pre-rotated reference” test ultimately compares the production transform with itself.**  
   `src/tests/test_yaw_aug_c16.py:344-359`

   The first-principles result is checked only with `allclose`; the plain wrapper is fed `pre`, which was produced by `rotate_scene_metadata`. Exact conditioning, loss, and gradient equality therefore exercise production rotation on both sides.

   **Required change:** feed the first-principles metadata into the plain wrapper and define appropriate tolerances through conditioning, loss, and all gradients.

2. **Save-time NAS acknowledgement has stale-file and unbounded-background-write holes, and no real CIFS preflight.**  
   `exp40_kit.py:1219-1240`, `:1304-1325`, `:1328-1357`; `yaw_aug_c16_worklog.md:53`, `:60`

   If a name is already in `MANIFEST.sha256`, publication skips rehashing the NAS file. Identical-orphan adoption returns without a final source rehash. On timeout, the daemon publication thread continues writing after the caller has failed. Tests exercise local temporary files, not `RENAME_NOREPLACE`, directory fsync, O_EXCL, and flock behavior on the actual CIFS destination. The worklog establishes libc availability, not NAS semantics.

   **Required change:** rehash every already-manifested destination before acknowledgement, rehash the source after orphan adoption, use a cancellable bounded publication mechanism, and add a pre-FULL capability receipt produced on the real NAS directory.

3. **A failure writing the GPU-lock ownership record leaks locks.**  
   `exp40_kit.py:1953-1957`

   Locks are acquired before the CLI writes `LOCK_JSON`. If that JSON write fails, the shell has no token with which to release the locks.

   **Required change:** catch output-write failure and release exactly the just-acquired paths, or create a durable owner record before acquisition and update it transactionally.

4. **The pinned A6000 driver is never enforced, and evaluation has no environment/resolution gate.**  
   `exp40_kit.py:95`, `:561-600`; `exp40_eval.py:201-238`, `:458-467`

   `DRIVER_A6000` is unused. The driver enters the environment digest but is not compared with the stated pin. Evaluation checks sealed-HF bytes but does not run the HF resolution probe, version/origin checks, CUDA/runtime checks, or record an environment digest in E0/cell receipts.

   **Required change:** enforce the A6000 driver pin and module origins during training. Add a strict evaluation environment probe, including offline HF resolution inside the sealed hub, and bind its digest into E0 and every cell receipt.

5. **The neuronic GPU gate does not identify the Slurm-allocated physical GPUs.**  
   `exp40_train.sbatch:39-43`, `:65-67`; `exp40_kit.py:897-926`

   It checks physical indices `0..N-1` from `nvidia-smi`, while Slurm may allocate arbitrary devices and expose a remapped `CUDA_VISIBLE_DEVICES`. On a larger node it may inspect unallocated GPUs or fail on total visible count while torchrun uses different devices.

   **Required change:** record and validate `SLURM_JOB_GPUS`/allocated UUIDs before clearing rank variables, map CUDA local ranks to those UUIDs, and gate occupancy/model/count on the actual allocation.

6. **Scheduler reconciliation mishandles duplicate accepted jobs and does not implement the declared state machine.**  
   `exp40_kit.py:1680-1696`, `:1712-1748`

   A successful claim winner plus a loser that exits 3 appears as mixed COMPLETED/FAILED and is classified as failed. State stays `submitted` or `sbatch_attempted`; it never durably transitions through running/terminal, and all discovered duplicate IDs are only placed in an informal `reconciled` field. Malformed state JSON raises an uncaught exception rather than a named refusal.

   **Required change:** reconcile against the durable claim owner job ID, distinguish claim-refused duplicates, record all job IDs, and durably transition to running/terminal under the shared flock. Treat malformed/truncated state as an explicit ambiguous refusal.

7. **The required evaluation dataset freeze is a command-sheet convention, not a gate.**  
   `exp40_eval.py:201-238`, `:866-872`; `yaw_aug_c16_eval_commands.md:48-55`

   Evaluation checks IR counts but not the 16-item content sample, and it neither consumes nor age-checks the full `ar_verify_eval.json` receipt. The suite runs normally if step 3 in the command sheet is skipped.

   **Required change:** require the current full-rehash receipt as a runtime input before E0, validate manifest digest, root, and age, and run the quick count plus sample gates before every cell batch.

8. **Evaluation does not take the planned cooperative card locks and does not recheck occupancy immediately before spawn.**  
   `exp40_eval.py:450-467`, `:504-513`

   `lane_free` is advisory and races other launchers. The driver then performs worktree/HF/data checks before spawning without another occupancy check. It never participates in `GPU_LOCK_<uuid>` or the exp_40 `.cards.lock`.

   **Required change:** acquire the same UUID lock for the chosen lane, recheck UUID/model/occupancy immediately before spawn, hold the lock for the evaluator lifetime, and release it through signal-safe cleanup.

9. **Per-cell verification is materially weaker than §5.3.**  
   `exp40_eval.py:343-425`

   It does not enforce the exact top-level metrics schema; accepts booleans as numeric metrics; checks only `by_scene` names, not per-scene keys/finiteness; does not check tuple/offset lengths; does not bind fixed assignment tuples or random offsets to the expected draw; accepts a substring in `COMPLETE`; and does not require the no-flash launcher SHA in the sentinel.

   **Required change:** implement the exact type-strict metrics/scene/stream/dump schema, tuple counts and relationships, expected random offsets, exact COMPLETE contents, and both overlay and launcher SHAs.

10. **Comparator registration and collection do not validate the declared registry contract.**  
    `exp40_eval.py:561-629`, `:658-668`

    Registration does not verify source receipts, stream protocol, dataset/model config hashes, dump bytes, exp_36 replay-verdict SHA, or a predeclared exp_36 evaluator pin. Collection later rehashes only metrics/stream/receipt, not dump manifests or waveform evidence, and it does not require exact registry count/content.

    **Required change:** validate each comparator with its source experiment’s authoritative validator, pin the exact expected cell set and evaluator/replay identities, record all promised fields, and rehash every registered artifact during collection.

11. **The `verify` subcommand performs no verification.**  
    `exp40_eval.py:892-897`

    It only prints argv and the E0 gate, despite the module contract saying it re-verifies every receipt.

    **Required change:** make `verify` call the same strict validators as run/collect for every selected receipt and return nonzero on any missing or invalid artifact. Keep argv printing as a separate dry-run action.

12. **Collection and figure outputs do not implement the planned results contract.**  
    `exp40_eval.py:540-558`, `:681-731`, `:843-858`

    The off-grid result omits P1’s `d`. The results Markdown contains only θ=0 and the C16-vs-YAW off-grid summary, not the promised C4/G45/OFF/R per-angle tables. No `yaw_aug_c16_paper_import.csv` is produced. The figure manifest omits the raw-input CSV SHA, and `figure` bypasses that CSV entirely. The `allowed` contract set is computed but unused.

    **Required change:** generate all promised tables, include P1’s descriptive `d`, write the paper-import CSV, build the figure from a validated frozen CSV, and bind its SHA plus only the contract-selected cell hashes into the figure manifest.

13. **A neuronic FULL can print PASS before satisfying the experiment’s FULL acceptance condition.**  
    `exp40_train.sbatch:88-109`

    The job audits only local checkpoints and prints PASS even though no checkpoint is yet on NAS and no admission exists. Plan §14 explicitly says the local manifest is not a substitute for NAS publication and admission.

    **Required change:** label this state `LOCAL_COMPLETE_PENDING_PUBLICATION` rather than PASS, with a durable pending receipt. Reserve final FULL PASS for post-pull NAS rehash, audit, and admission.

## MEDIUM

1. **The capture helper leaks a global flash-attention mutation, and only validation—not test—is checked for no augmentation.**  
   `exp40_yaw_aug_golden_capture.py:100-111`; `test_yaw_aug_c16.py:365-373`

   `_mods()` permanently assigns `transformer.flash_attn_func = None`, making the combined pytest process order-dependent.

   **Required change:** scope and restore the patch or run capture cases in isolated subprocesses; add the equivalent `test_step` no-augmentation test.

2. **Several claimed test guarantees are vacuous or absent.**  
   `test_exp40_eval.py:123-131`, `:150-183`, `:244-252`; `test_exp40_kit.py:225-249`, `:291-369`, `:702-714`

   The off-grid numerical assertion accepts any negative result. The passing cell fixture uses empty per-scene payloads. E0 comparison itself is not tested. “Race” tests are sequential. The cadence PASS audit is disabled with `if False`, and admission is not exercised.

   **Required change:** assert exact formula values; populate and validate full scene schemas; add E0 mismatch cases, true concurrent claim/cache/state tests, truncated-state tests, successful full audit/admission, and mutation of every NAS checkpoint.

3. **AR receipt freshness accepts future timestamps, and the submitter performs a weaker check than the in-job FULL gate.**  
   `exp40_kit.py:487-500`; `exp40_submit_neuronic.sh:43-47`

   Negative age passes. The submitter ignores age and `manifest_digest`, allowing an irreversible FULL identity to be claimed by a job that immediately fails its in-job gate.

   **Required change:** require `0 <= age <= 48h` and run the same strict AR receipt validator before submission and in the job.

4. **Partition MaxTime discovery is ambiguous and pipeline-sensitive.**  
   `exp40_submit_neuronic.sh:59-60`

   `sinfo | head -1` can hide heterogeneous limits and can fail under `pipefail` if `sinfo` receives SIGPIPE.

   **Required change:** parse all returned limits without `head`, require one unambiguous value or conservatively use the minimum, and record the raw query in the submission state.

5. **The evaluation command sheet is not independently runnable before training.**  
   `yaw_aug_c16_eval_commands.md:33-55`

   It assumes the receipts directory and local sealed HF hub already exist, although E0 is documented as runnable before C16 training. It also does not pass the AR verification receipt to the driver because the driver has no such gate.

   **Required change:** add explicit directory creation and sealed-hub build/verify commands, then pass the full dataset/environment receipts into E0 and later cells.

6. **The launch/admission command sheet does not provide a reproducible audit path or full admission metadata.**  
   `yaw_aug_c16_neuronic_commands.md:34-40`, `:90-100`

   `<ts>` is not resolvable mechanically, and the proposed metadata omits the run-tree digest and validated evidence identities. The neuronic post-pull instruction says “exactly as in A.4” without giving the required remote `expected-dirpath`.

   **Required change:** have the launcher write a stable run-result manifest naming the audit and evidence files, and provide exact copy/publish/audit/admit commands whose metadata is derived from those records.

7. **The generator contract does not require exactly five unique, correctly located files.**  
   `gen_model_comparison.py:517-531`, `:895-915`, `:1201-1209`

   Duplicate byte-identical copies with registered basenames can inflate `n` and be aggregated; the two-K gate then sees a numeric row and passes.

   **Required change:** validate exact canonical farm paths and the complete seed set `{42,…,46}` for each K before aggregation.

## NIT

1. **Several comments/docstrings still describe superseded behavior.**  
   `exp40_kit.py:16`, `:22`; `exp40_evidence_driver.py:49-52`; `exp40_train.sbatch:13`; `exp40_eval.py:2`, `:19`

   They refer to publisher “watch” mode, a loss digest dropped by §14 H4, and plan r1/r2 rather than binding §14.

   **Required change:** update the operational documentation to the actual save-hook, finite-flag, and §14 contracts so future reviewers and operators do not follow removed paths.

## What I verified

- Read the requested SOP, announcements, final r3 plan and all three plan reviews, worklog/smoke evidence, and cited exp_15/17/25/36 records.
- Reviewed the complete `2b42468c..15f73ff5` diff.
- Verified the C16 quantization mathematics, factory behavior, constructor ordering, config delta, banner, and train/eval dispatch sites.
- Verified the 85-cell counts and exp_25 argv-substitution construction.
- Inspected installed PL 2.1 source for checkpoint-save barriers, broadcast behavior, and TorchElastic-before-SLURM selection.
- Confirmed the committed smoke evidence records live C16 offsets, finite loss, flash calls, and attributed checkpoint calls.
- Ran only read-only inspection commands. I did not run pytest, training, evaluation, GPU, Slurm, checkpoint loading, or any process-control command, and modified no files.