**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_38 code review 1 of 3 (2b42468c..0d6edf46)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING

1. **E0 and SMOKE_DDP receipt consumers verify only a small subset of their claimed bindings.**  
   [`exp38_registry.py:901`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:901) checks E0 verdict, closure, and nonempty evaluator hashes, but ignores the exact eight-cell set/equality results, `failed`, HEAD, environment-receipt SHA, checkpoint map, reference-manifest/current reference hashes, and input-hash pins. [`exp38_registry.py:924`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:924) accepts a smoke receipt based only on mode/run/platform/verdict/closure; it ignores backend, environment receipt, argv projection, config/split hashes, GPU identity, HEAD, and recorded evidence. Thus an SDPA smoke can authorize a flash FULL, and an old smoke can authorize a recreated environment receipt. The writer at [`exp38_registry.py:937`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:937) also trusts the caller-provided strings `"LOG GATE PASS"` and `"EVIDENCE GATE PASS"` instead of revalidating their underlying artifacts.  
   **Required:** define exact receipt schemas and reject missing/extra/wrongly typed fields; validate every plan binding against current files/receipts and requested backend; re-run evidence validation when creating or consuming the smoke receipt.

2. **The first-step gate records but does not validate the RNG state that controls the trajectory.**  
   [`exp38_registry.py:731`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:731) only checks that ranks share one `torch_cpu` digest. It ignores Python, NumPy, and CUDA RNG digests, plus recorded `strategy`, `local_rank`, and `n_state_tensors`. A wrong CUDA initial state can therefore alter the run while passing the evidence gate.  
   **Required:** pin all RNG components explicitly—rank-indexed where they should differ and equality-constrained where they should agree—and validate the complete evidence schema.

3. **Result evaluation is not bound to the accepted E0 campaign, commit, closure, or hardware.**  
   [`exp38_eval.py:370`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:370) accepts any clean campaign-tree HEAD, computes a closure without comparing it to E0, and never calls `check_e0_receipt`. It records `head`/`closure_sha256`, but [`exp38_eval.py:195`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:195) ignores both. [`exp38_collect.py:62`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:62) calls validation without an expected source SHA and merely requires a run’s ten cells to agree with each other. GPU model/UUID is not admitted either.  
   **Required:** make result-cell launch, validation, and collection consume the E0 receipt and require its HEAD/closure/environment/evaluator pins; require metrics `source_sha`, receipt HEAD/closure, and expected A6000 identity to match that campaign.

4. **A checkpoint/run can be admitted without a durable successful-training admission record.**  
   [`exp38_eval.py:270`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:270) audits once, then copies a mutable `CHECKPOINTS.sha256`; later [`exp38_eval.py:176`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:176) trusts that manifest. There is no terminal `COMPLETE` receipt proving rc=0 plus log/evidence/snapshot/end-checkpoint audits, and `ckpt_bytes_loaded` is never compared. A failed run that happened to write the 40k checkpoint, or a checkpoint and manifest replaced together after preparation, can be admitted.  
   **Required:** publish a terminal training receipt only after every end audit succeeds; bind run, closure, snapshot digest, checkpoint SHA and byte count, logs/evidence, and config. `prepare`, evaluation, and collection must require that receipt and compare both loaded SHA and byte count.

5. **Transactional cell admission does not prove the admitted artifacts are complete or numeric.**  
   [`exp38_eval.py:145`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:145) treats strings, `None`, and other unsupported leaves as finite; by-scene inner metric schemas are not checked. [`exp38_eval.py:239`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:239) only checks dump metadata and marker existence—it does not verify `waveforms.npy`, shape `[6337, 10240]`, dtype, `waveforms.sha256`, or `COMPLETE` contents. The supposedly valid fixture at [`test_exp38_eval.py:88`](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:88) contains no waveform file and is admitted. Receipt cell/path/log/head/closure/AGREE and checkpoint-byte fields are also ignored.  
   **Required:** require exact finite non-boolean numeric schemas globally and per scene; call the evaluator’s full `ScoredWaveformDump.is_complete(..., expected_n=6337)` predicate or an equivalent exact check; validate every receipt and checkpoint provenance field.

6. **Evaluation dependencies and output storage are not pinned or mount-verified.**  
   `AGREE` is only a pathname at [`exp38_eval.py:42`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:42); its post-run hash is recorded at line 352 but never checked against an expected value or rechecked on admission. The evaluator’s `VAE.ckpt` dependency is likewise unbound. [`exp38_eval.py:400`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:400) only calls `statvfs` on the dump root, so an unmounted NAS mountpoint can silently receive local waveforms; it also budgets one cell rather than the requested block. `PYTHONOPTIMIZE` is not sanitized at line 335.  
   **Required:** pre- and post-pin AGREE and all evaluator weight dependencies in E0 and every result admission; sanitize optimization/import flags; require the expected mount type/source plus read/write probe and block-level free-space budget.

7. **Reference rows can be published from modified, unmanifested anchor files.**  
   [`exp38_eval.py:298`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:298) verifies only whatever lines happen to be present, not the exact required 68-file set or duplicates. More importantly, [`exp38_collect.py:55`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:55) loads anchors directly without calling `check_refs`, checking E0, or comparing the current manifest and files with the E0 receipt. These reference files are excluded from the closure.  
   **Required:** define and enforce the exact reference set, invoke it during collection, and compare the current manifest SHA and every reference SHA with the accepted E0 receipt.

### HIGH

1. **The “full-state” checkpoint audit is partial.**  
   [`exp38_registry.py:391`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:391) validates selected fields, a hard-coded optimizer-state count, shapes/finiteness, selected scheduler keys, three loop counters, and callback `dirpath`. Extra/missing optimizer keys, scheduler entries, loop state, and callback payload can pass. Mutation tests use `inventory=False`.  
   **Required:** enforce exact schemas/types/key sets for optimizer, scheduler, loops, callbacks, and state inventory, and exercise the real inventory path for every arm/rung.

2. **The registry still exposes unsafe primary-run resume argv.**  
   [`exp38_registry.py:502`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:502) and CLI line 998 generate `--ckpt-path` using the same primary ID/save directory, without a distinct diagnostic ID or `RESUMED_AT_*` marker. Tests explicitly bless this behavior.  
   **Required:** remove resume from the primary registry/CLI, or implement the plan’s separate non-primary diagnostic identity, marker, terminal receipt, and collector exclusion.

3. **Unplanned run IDs are launchable.**  
   The permissive regex at [`exp38_registry.py:147`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:147), local validation at [`exp38_train_local.sh:45`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train_local.sh:45), and sbatch validation at [`exp38_train.sbatch:31`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train.sbatch:31) allow `de_p1_f050`, `de_p1_f075`, and their 8×8 variants, although only P1 f025 is optional in r2.  
   **Required:** launch/describe only membership in the registered matrix; keep sensitivity rungs behind explicit registered IDs.

4. **The local launcher can orphan training and later signal an unrelated process.**  
   [`exp38_train_local.sh:217`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train_local.sh:217) creates a new process group but installs no TERM/INT/HUP/EXIT trap. If the driver dies, its flock FDs close while training continues. `kill_ours` has no grace timeout/KILL escalation and then waits indefinitely; PGID/PID reuse is not guarded by process start time. The SMOKE sampler at lines 124–129 has the same PID-reuse risk. Partial records are not finalized/copied on these exits.  
   **Required:** install ownership-checked traps, forward only to the recorded child PGID, wait with a bound and escalation, clean the sampler, preserve the crash marker, and atomically finalize the partial record.

5. **Neuronic training has no immutable snapshot.**  
   [`exp38_train.sbatch:52`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train.sbatch:52) checks the mutable checkout once, then launches and runs end audits from that checkout. A long-running job can therefore train/audit against files changed after its initial clean check.  
   **Required:** create the same closure-bound tool/config snapshot used locally, launch from it, use `--repo <snapshot>` for every later registry audit, and verify the snapshot at completion.

6. **Neuronic “exactly once” and job ownership are incomplete.**  
   [`exp38_submit_neuronic.py:51`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.py:51) gives every SMOKE a new token-specific directory, so duplicate SMOKEs are unrestricted. `find_intent` at line 146 also includes renamed `.abandoned_*` directories. Reconciliation at line 179 silently leaves zero/multiple-job ambiguity unresolved. The job only checks that its name contains the token prefix and allows any `.exp38_submitted` marker at [`exp38_train.sbatch:79`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train.sbatch:79); it never validates the intent’s exact run/mode/backend/HEAD/closure/save-dir/job-id.  
   **Required:** use one stable ownership record per `(run, mode)`, separate ownership from attempt payloads, exclude abandoned records, refuse ambiguous scheduler states, and have sbatch exact-compare its allocation and environment with the intent.

7. **Neuronic allocation/storage/signal gates are weaker than the plan.**  
   [`exp38_train.sbatch:73`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_train.sbatch:73) counts `nvidia-smi` rows rather than proving the Slurm-visible allocation/CUDA mapping, has no filesystem/free-space/write probe, and lines 121–142 have no signal trap or bounded termination.  
   **Required:** validate `CUDA_VISIBLE_DEVICES`/Slurm allocation plus torch-visible devices and names, perform the registered storage probes, and add signal-safe child termination and terminal intent/record updates.

8. **The subset audit’s exact realpath binding is not portable to the advertised Neuronic setup.**  
   [`exp38_subset_audit.py:147`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_subset_audit.py:147) includes `realpath(AcousticRooms)` in the exact bound map. The artifact records `/media/diskstation/...`, while [`data_efficiency_neuronic_commands.md:12`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_neuronic_commands.md:12) constructs the Neuronic link from `/n/fs/...`. Unless both happen to resolve to the identical A6000 spelling, sbatch line 58 refuses before training.  
   **Required:** bind dataset content/root fingerprint and record the actual platform path, or register explicit expected canonical roots per platform.

9. **The environment and Neuronic untracked-tree receipts are not exact enough.**  
   [`exp38_registry.py:838`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:838) converts NCCL/driver probe failures to pinnable `None` and omits interpreter identity, relevant environment flags, and a complete dependency fingerprint. [`exp38_registry.py:805`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:805) allows Neuronic symlink names without enforcing their targets.  
   **Required:** fail closed on required probes, bind `sys.executable`, relevant numeric/import flags and a normalized package inventory, and validate each allowed symlink by registered target or content identity.

10. **The result pair is not committed in the planned order and lacks a campaign manifest.**  
    [`exp38_collect.py:107`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:107) renames JSON first and Markdown second, even though JSON is the designated last commit point. A crash can expose new JSON with old Markdown. The JSON contains no manifest of admitted raw artifacts/hashes, and `main` always returns zero with pending cells despite documenting a `--partial` mode that does not exist.  
    **Required:** fsync and publish Markdown first, JSON last, directory-fsync the commit, include the complete admitted-artifact manifest, and make pending cells fail unless an explicit `--partial` option is supplied.

11. **Tests do not exercise the gates most likely to fail open.**  
    Receipt consumers have no schema/backend/env/input-pin mutation suite; launcher tests are dry-run/refusal tests only; there is no signal/lock/orphan test; exactly-once tests cover FULL but not SMOKE or intent ownership; checkpoint mutations bypass inventory; evidence tests are single-process and do not establish DDP equality or numeric/gradient equivalence. The waveform-free fixture at [`test_exp38_eval.py:88`](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:88) directly demonstrates vacuous admission.  
    **Required:** add negative mutation coverage for every receipt/admission field, synthetic launcher lifecycle tests, SMOKE exactly-once/reconciliation tests, full checkpoint-schema mutations, and a real two-rank evidence test or equivalent disclosed integration proof.

### NIT

1. **Flag-absent behavior is numerically unchanged, but not literally byte-for-byte old behavior.**  
   `evidence_json` is added to parsed arguments at [`defaults.ini:48`](/home/yixunhu/codespace/exp-38-data-efficiency/defaults.ini:48), and `vars(args)` is sent to W&B/Comet at [`train.py:194`](/home/yixunhu/codespace/exp-38-data-efficiency/train.py:194). Thus old jobs can acquire an extra logged hyperparameter even when the callback is disabled.  
   **Required:** omit the empty field from logger metadata or document and regression-test this metadata-only difference.

2. **Smoke receipts are not written atomically.**  
   [`exp38_registry.py:963`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:963) writes the JSON and sidecar directly. Crashes fail closed but leave awkward unrecoverable partial artifacts.  
   **Required:** temporary files, file fsync, ordered `os.replace`, and directory fsync.

3. **The stored validation evidence does not substantiate “all paths green.”**  
   [`data_efficiency_2026-11-07_21-36-46_dry_run_all_paths.log:151`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_2026-11-07_21-36-46_dry_run_all_paths.log:151) shows all twelve sbatch paths refusing a dirty tree; the later sbatch log still refuses its final paths at line 81. The full-suite CPU log has no final summary, while the focused exp38 log does show 195 passed/1 skipped.  
   **Required:** after fixes, record one clean-final-HEAD dry run covering every local/sbatch path and one complete test summary.

4. **`git diff --check` is not clean.**  
   The first failures begin at [`data_efficiency_2026-11-07_20-34-13_pytest_A3_red.log:11`](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_2026-11-07_20-34-13_pytest_A3_red.log:11).  
   **Required:** normalize committed logs or explicitly exclude raw logs from the repository’s whitespace check.

## Answers to Q1–Q6

### Q1 — Correctness versus plan r2

Not yet compliant. The deterministic primary argv, effective batch-size invariant, B-F per-angle cap configuration, fraction configs, local subset audit, local closure snapshot, and basic lock/run-directory structure are present. The major gaps are the shallow receipts, incomplete RNG/full-state gates, unsafe resume and extra run IDs, missing evaluation/campaign bindings, incomplete transactional admission, non-atomic results, and incomplete Neuronic snapshot/exactly-once implementation described above.

### Q2 — Production safety

`run_evidence.py` is statically observe-only in the numeric sense: digesting state/RNG does not draw RNG; wrappers return the original call results; wrappers are removed after step one and in teardown; and filenames are global-rank-local. I found no numeric mutation in the callback itself. The missing issue is validation of the recorded RNG state, not recording it.

With the flag absent, trainer callbacks and numerical behavior remain the old behavior. It is not literally byte-for-byte externally observable because the empty argument enters logger metadata.

The `dataset.py`/`AR_md.py` port does not change the 100% or evaluation configurations: `allowed_basenames=None` preserves the original candidate list and draw, and only `restrict_to_split=true` activates the new path. `DatasetContractError` only changes handling of that new fatal contract failure.

### Q3 — Fail-open paths

Yes: wrong E0 contents, wrong smoke backend/environment, wrong CUDA RNG state, evaluation at another accepted-looking HEAD/closure/GPU, mutable checkpoint manifests, changed AGREE/VAE assets, incomplete waveform dumps, and modified reference cells can all pass current consumers.

I did not find a separate generic shell parse-error bypass: checked command substitutions generally feed explicit comparisons, `PIPESTATUS` is used for piped gates, JSON exceptions terminate, and mapfile’s swallowed producer status is followed by an argv-length check. The substantive fail-open paths are semantic under-validation rather than uncaught parser failures.

### Q4 — Long-running bash robustness

The local snapshot invocation is correct: the snapshot registry is called with `--repo "$SNAP"`. Normal local failures preserve the run-directory crash marker, and flock FDs are released when the driver exits.

Signal/process ownership is not robust: both launchers lack traps, termination has no escalation or PID-start-time validation, partial records are not finalized, and local driver death can release card locks while its process group continues. Neuronic also lacks immutable tools and storage probes. Local pre-launch NAS mount/free/write checks are good; a mid-run NAS hang remains fail-closed but can block indefinitely.

### Q5 — Tests

The focused test log is green, but the suite does not prove several stated properties. The clearest counterexample is the “valid” evaluation fixture being admitted without `waveforms.npy`. Receipt-field mutations, DDP RNG evidence, real full-state inventory, process lifecycle, Neuronic SMOKE exactly-once, snapshot drift, and publication crash ordering are materially under-tested.

### Q6 — Other comparability/auditability issues

The largest additional risks are mutable reference anchors, missing evaluator-weight pins, lack of a terminal successful-training receipt, missing campaign artifact manifest, permissive partial publication, and incomplete clean-final-HEAD validation evidence. These can make otherwise plausible results non-comparable or impossible to reconstruct.

## What I verified

- Reviewed the complete `2b42468c..0d6edf46` diff and the cited SOP, plan r2, worklog, command sheets, receipts/logs, tests, and reference cells.
- Confirmed the AR metadata port, dataset test, fraction splits, manifest, and fraction configs are byte-identical to the exp_14 sources cited in the worklog.
- Confirmed the subset artifact reports PASS with 73,473 / 145,665 / 218,463 targets, 243 rooms each, no empty restricted pools, and a root fingerprint.
- Inspected both recorded GPU-0 smoke artifacts and their in-process evidence.
- Confirmed the focused exp38 pytest log reports 195 passed, 1 skipped.
- Performed static/read-only review only; I did not run tests, training, evaluation, launchers, GPU commands, or process inspection.