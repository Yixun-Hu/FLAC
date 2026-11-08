**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-07 · **Round:** exp_38 code review 2 of 3 (2b42468c..e053c4c6; fix round 0d6edf46..e053c4c6)

**Verdict:** REQUEST-CHANGES

The fix round materially improves the experiment, but four blocking paths remain: terminal receipts are not fully consumed, evaluation runs from a mutable checkout, dataset identity covers filenames rather than audio bytes, and inherited distributed-launch variables can contaminate an A6000 launch before the evidence gate stops it.

## Q1 — Round-1 findings

| Finding | Status | Evidence |
|---|---|---|
| B1 — shallow E0/SMOKE consumers | **PARTIAL** | E0 now checks exact top/cell schemas and campaign bindings at [exp38_eval.py:260](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:260); SMOKE revalidates request fields and embedded evidence at [exp38_registry.py:1308](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1308), with mutation tests at [test_exp38_receipts.py:253](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:253). However E0 cell hashes are only syntax-checked, and SMOKE’s log/snapshot bindings are not rehashed. |
| B2 — RNG recorded but not validated | **RESOLVED** | All evidence keys, topology-specific RNG digests, ranks, strategy and initial-state fields are enforced at [exp38_registry.py:1023](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1023). The mutation suite covers every RNG component at [test_exp38_receipts.py:89](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:89), and the real two-rank CPU test establishes the DDP reset rule at [test_exp38_ddp_evidence.py:59](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_ddp_evidence.py:59). |
| B3 — evaluation not bound to E0/campaign/GPU | **PARTIAL** | Cell receipts now bind E0, HEAD, closure, environment and A6000 UUID/name at [exp38_eval.py:407](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:407), with mutation tests at [test_exp38_eval.py:260](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:260). But `eval_FLAC.py` still executes from the mutable campaign checkout at [exp38_eval.py:542](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:542), so transient code/config changes can be stamped with the old campaign identity. |
| B4 — no terminal training receipt | **PARTIAL** | The writer is now strong: it gates rc, logs, evidence, checkpoints and manifest before sealing at [exp38_registry.py:1372](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1372). `prepare` audits the final checkpoint at [exp38_eval.py:464](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:464). The consumer at [exp38_registry.py:1409](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1409) ignores most semantic fields, and there is no terminal-receipt mutation suite. |
| B5 — vacuous numeric/dump/cell admission | **RESOLVED** | Exact finite non-boolean metric schemas are enforced at [exp38_eval.py:172](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:172), complete cell admission at [exp38_eval.py:357](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:357), and waveform bytes/shape/dtype/checksums at [exp38_eval.py:329](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:329). Mutation tests are non-vacuous at [test_exp38_eval.py:233](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:233) and [test_exp38_eval.py:302](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:302). |
| B6 — evaluator dependencies/storage unpinned | **PARTIAL** | AGREE/VAE/DINO pins run before and after each evaluation at [exp38_eval.py:531](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:531), and block-level fstype/free-space/write probing is at [exp38_eval.py:601](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:601). Admission merely compares recorded values with constants at [exp38_eval.py:412](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:412); it never calls `eval_pins()` against the current dependencies. |
| B7 — mutable/unmanifested reference rows | **RESOLVED** | The exact 68-file set, duplicates, manifest and bytes are enforced at [exp38_eval.py:207](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:207). Collection compares the current set with E0 at [exp38_collect.py:52](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:52). Negative tests are at [test_exp38_eval.py:94](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:94). |
| H1 — partial checkpoint audit | **PARTIAL** | Outer schemas, inventory, optimizer/scheduler fields and selected loop/callback data are much stronger at [exp38_registry.py:431](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:431), with real inventory/readback tests at [test_exp38_registry.py:457](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:457). Arbitrary optimizer-state parameter IDs, nested loop payloads and most callback values still pass. |
| H2 — unsafe resume argv | **RESOLVED** | Primary argv has no resume path at [exp38_registry.py:570](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:570); removal is tested at [test_exp38_registry.py:301](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:301). |
| H3 — unplanned IDs | **RESOLVED** | Registry membership is authoritative and tested against all disallowed P1/rung variants at [test_exp38_registry.py:247](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:247). Shell-level overacceptance cannot reach training because the supervisor calls the registry first. |
| H4 — orphaning/unrelated signaling | **PARTIAL** | PID/start-time and unreaped-leader ownership make process-group signaling safe at [exp38_supervise.py:104](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:104), with meaningful signal/grace/reap tests at [test_exp38_supervise.py:92](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:92). The card-lock guarantee does not extend to Lightning’s spawned rank process; see new HIGH finding 1. |
| H5 — no Neuronic snapshot | **RESOLVED** | Both platforms create and launch from a closure snapshot at [exp38_supervise.py:515](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:515), verify its digest, and audit it on completion. Snapshot mutation/removal tests are at [test_exp38_registry.py:651](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:651). |
| H6 — Neuronic exactly-once/ownership | **PARTIAL** | Stable `(run, mode)` owner records, exact intent comparison, reconciliation and abandon logic exist at [exp38_submit_neuronic.py:127](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.py:127) and [exp38_submit_neuronic.py:176](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.py:176). Attempt claiming has a check-then-create race that can admit two early jobs. |
| H7 — Neuronic allocation/storage/signal gates | **RESOLVED** | Slurm/CUDA/torch/nvidia allocation checks are at [exp38_supervise.py:300](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:300), storage checks at [exp38_registry.py:849](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:849), and bounded signal handling is shared with A6000. Tests would fail if these gates were removed. |
| H8 — nonportable subset realpath | **PARTIAL** | Exact path binding was removed, but its replacement at [exp38_subset_audit.py:135](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_subset_audit.py:135) hashes filenames only, not audio bytes. A different dataset with identical room listings passes. |
| H9 — environment/untracked tree inexact | **PARTIAL** | Required probes now fail closed and the receipt records interpreter/stack/package versions at [exp38_registry.py:1155](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1155); links are content-checked at [exp38_registry.py:1090](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1090). Package bytes/import origins and untracked non-`.py` shadows under `worklog/` remain unbound. |
| H10 — result order/manifest/partial CLI | **PARTIAL** | Markdown is written first and JSON last, with explicit partial behavior, at [exp38_collect.py:116](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:116) and [exp38_collect.py:152](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:152). The manifest at [exp38_collect.py:84](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:84) is not a manifest of every admitted artifact. |
| H11 — missing high-risk tests | **PARTIAL** | Many useful suites were added, including real DDP and process ownership. Missing coverage remains for terminal receipt semantics, actual Lightning-rank lock inheritance, atomic attempt claiming, transient evaluation-tree mutation, E0 cell digest binding, current eval-pin admission and complete result manifests. |
| N1 — empty flag changes logger metadata | **RESOLVED** | Empty `evidence_json` is removed before logger publication at [train.py:51](/home/yixunhu/codespace/exp-38-data-efficiency/train.py:51), with regression coverage at [test_exp38_receipts.py:482](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:482). |
| N2 — non-atomic smoke receipts | **RESOLVED** | Sealed JSON/sidecar writes use fsync, ordered replacement and directory fsync at [exp38_registry.py:780](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:780), tested at [test_exp38_receipts.py:145](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:145). |
| N3 — evidence did not show all paths green | **PARTIAL** | The focused suite is complete—511 passed at [pytest log:48](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_2026-11-07_23-04-05_pytest_exp38_final_code_r1.log:48)—and the dry run records expected successes/refusals. The full CPU regression still says shard 2 was running at [full-suite log:23](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_2026-11-07_22-53-30_pytest_full_suite_cpu_sharded.log:23), so “all paths green” is not established. |
| N4 — `git diff --check` dirty | **NOT** | `git diff --check 2b42468c..HEAD` still reports committed trailing whitespace in [data_efficiency_codex_code_r1_review.md:9](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_codex_code_r1_review.md:9) onward and a new blank EOF line at [plan_data_efficiency.md:247](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md:247). |

## Q2 — New findings introduced or exposed by the fix round

### BLOCKING

1. **`TRAINING_COMPLETE` is stronger when written than when consumed.**

   [exp38_registry.py:1418](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1418) validates only run/rc/gate strings/final name/config-split hashes, optional closure/E0 and checkpoint manifest/final bytes. It ignores semantic validation of `platform`, `backend`, `head`, snapshot, environment receipt, SMOKE receipt/path/hash, full argv, log/path/hash, evidence hash, GPU identity and timestamps. Both evaluation admission at [exp38_eval.py:319](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:319) and the Neuronic publication command at [exp38_registry.py:1553](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1553) use this shallow consumer.

   **Required change:** make `check_training_receipt` validate every field against the requested platform/backend/campaign, exact registered argv and GPU topology; rehash and re-run log/evidence gates; validate the referenced E0, SMOKE and environment receipts; and make `training-check` require the expected closure/E0/platform. Add a resealed-field mutation suite covering every key.

2. **Evaluation executes from a mutable checkout while claiming the frozen campaign identity.**

   `campaign_gates()` checks cleanliness only once at [exp38_eval.py:573](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:573). The evaluator then runs with `cwd=reg.REPO` at [exp38_eval.py:542](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:542). `source_sha()` records only `git rev-parse HEAD` at [eval_FLAC.py:1291](/home/yixunhu/codespace/exp-38-data-efficiency/eval_FLAC.py:1291). A tracked evaluator/config/split file can be changed transiently during a long cell and restored before inspection; the result can still pass with the original HEAD/closure.

   **Required change:** evaluate from a sealed E0-closure snapshot, including configs, split, evaluator and launcher, and verify that snapshot before and after every cell. Add a fake evaluator test that mutates the campaign checkout during execution and proves the executed snapshot remains unchanged.

3. **The portable dataset fingerprint does not bind dataset content.**

   [exp38_subset_audit.py:135](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_subset_audit.py:135) hashes only `scene/room` plus sorted directory entry names. Replacing an audio file’s bytes without changing its filename passes both platforms’ subset verification. This permits a FULL to train on different data while retaining the registered split and audit.

   **Required change:** bind the audio bytes used by the campaign—preferably a committed per-file content manifest or an independently immutable dataset release digest—and test same-name/different-byte mutation. File sizes alone are insufficient.

4. **Inherited distributed-launch variables are neither sanitized nor receipted.**

   `child_env()` removes Slurm rank variables only for Neuronic at [exp38_supervise.py:224](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:224). A6000 preserves `SLURM_PROCID`, and both paths preserve inherited `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `GROUP_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_*` and `TORCHELASTIC_*`. These are not in the environment receipt. The current test explicitly blesses `SLURM_PROCID` surviving into A6000 at [test_exp38_supervise.py:243](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:243). That can change `train.py`’s seed or Lightning’s cluster environment, causing a wrong-topology launch or a destructive false refusal after the run directory is claimed.

   **Required change:** reject or strip all external rank/launcher variables before A6000 launch and before the Neuronic `torchrun` parent starts; let only the chosen launcher create rank variables. Include the complete set in environment-negative tests.

### HIGH

1. **The card-lock inheritance test does not cover Lightning’s actual rank subprocess.**

   Locks are passed only to the direct child at [exp38_supervise.py:542](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:542). PL 2.1 starts the second rank with a plain `subprocess.Popen(..., env=..., cwd=...)` at [subprocess_script.py:140](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/lightning_fabric/strategies/launchers/subprocess_script.py:140), so the lock FDs close in rank 1. If the supervisor is SIGKILLed and rank 0 later exits while rank 1 remains, the card lock can be released under a running trainer. [test_exp38_supervise.py:134](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:134) covers only a direct sleeping child.

   **Required change:** use a dedicated lock-holder whose lifetime covers the entire process group, or explicitly arrange inheritance to all ranks; test with the real Lightning subprocess launcher and a surviving rank.

2. **Neuronic’s one-attempt rule has a check-then-create race.**

   [exp38_submit_neuronic.py:193](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.py:193) lists existing attempts, then creates a job-specific `attempt_<jid>.json` at line 198. While intent state is `submitting` and `job_id` is `None`, two accidentally duplicated scheduler jobs can both observe no attempt and atomically create different filenames. The sequential test at [test_exp38_launchers.py:199](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_launchers.py:199) cannot expose this.

   **Required change:** claim one fixed owner-scoped marker using `mkdir` or `O_EXCL`, then create the informational job-specific record. Add a simultaneous two-process claim test.

3. **E0 and SMOKE retain unverified durable bindings.**

   E0’s `metrics_sha256` and `receipt_sha256` are only checked for SHA syntax at [exp38_eval.py:273](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:273). The “valid” fixture deliberately supplies invented `a…`/`b…` hashes at [exp38_fixtures.py:46](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/exp38_fixtures.py:46). SMOKE similarly checks that log/snapshot hashes look like hashes but does not reopen the log or snapshot at [exp38_registry.py:1321](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1321).

   **Required change:** retain immutable copies or registered paths and rehash them in consumers. Add E0 cell-hash and SMOKE log/hash/snapshot mutations with valid-looking alternate SHA values.

4. **Checkpoint “full-state” admission still accepts malformed optimizer/loop/callback state.**

   [exp38_registry.py:462](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:462) checks optimizer-state count but permits arbitrary out-of-range integer state IDs because shape validation is skipped when no matching parameter exists. [exp38_registry.py:484](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:484) checks only selected nested loop counters, and callback payload values other than `dirpath`/`monitor` are ignored at line 498.

   **Required change:** enforce the exact optimizer state-ID set and shape for every entry, recursively exact nested loop schemas/types, and registered callback payload values. Add those mutations to [test_exp38_registry.py:396](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:396).

5. **Current evaluator dependencies are not checked on admission.**

   `validate_cell()` trusts constant values recorded in the receipt at [exp38_eval.py:412](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:412). It does not execute [exp38_registry.py:752](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:752), despite that function’s documented “and on admission” contract.

   **Required change:** call `eval_pins()` during online E0/cell validation and collection, with a mutation test for each external dependency.

6. **Training snapshots omit the receipts the plan says they seal.**

   `RECEIPT_FILES` contains only the subset-audit sidecar at [exp38_registry.py:916](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:916); E0, SMOKE and environment receipts are not retained in the snapshot. The terminal receipt contains their hashes, but no durable copy of the SMOKE bytes is placed in the run/farm/result manifest.

   **Required change:** snapshot exact E0/SMOKE/environment receipt bytes and sidecars after validation, and carry those immutable copies through terminal admission and result publication.

7. **The result manifest is not complete.**

   Per-cell entries contain only metrics, stream, receipt and compact dump-manifest hashes at [exp38_collect.py:84](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_collect.py:84). It omits admitted logs, dump `COMPLETE`/sidecar/waveforms, checkpoint manifest/final checkpoint, training-receipt sidecar, E0 sidecar, reference manifest and retained upstream receipts.

   **Required change:** emit a path-and-hash manifest for every artifact validation consumed, or explicitly package a complete immutable admission bundle and manifest that.

8. **Launch-record persistence fails open.**

   [exp38_supervise.py:210](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:210) catches and logs every record-write `OSError`; training continues even if both record locations fail. This can leave a successful trainer with no durable launch/ownership history.

   **Required change:** require at least one designated authoritative record write to succeed before launch and on each terminal transition; secondary copies may remain best-effort. Test one-copy failure and all-copy failure separately.

9. **Evaluation has no per-cell ownership claim.**

   `cmd_run()` holds only a per-card lock at [exp38_eval.py:604](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:604). Two invocations on different cards can select the same cell, both pass the absent-artifact checks at [exp38_eval.py:526](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:526), then race on metrics, receipt and dump publication.

   **Required change:** atomically claim each cell before any execution and retain a crash marker requiring explicit archive/recovery.

10. **Tree/environment identity still permits import drift.**

    [exp38_registry.py:1095](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1095) ignores every untracked non-`.py` entry under `worklog/`; an untracked extension module can shadow an exp38 module before the check runs. The environment receipt at [exp38_registry.py:1169](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1169) hashes package names/versions, not installed module bytes or import origins.

    **Required change:** allowlist exact record filename patterns rather than whole directories, reject executable/importable extensions and `.pth` files, and record critical module origins plus an immutable environment/image identity.

11. **The pre-launch backend gate proves discoverability, not the backend binding.**

    [exp38_supervise.py:237](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:237) uses `find_spec("flash_attn")` only. It does not import the production transformer or verify `src.models.transformer.flash_attn_func`. The first-step evidence eventually detects a mismatch, but only after the run directory is claimed and training starts.

    **Required change:** perform the import/binding probe in the exact child environment before launch; retain first-step evidence as the in-process confirmation.

### NIT

1. **The primary command sheet was corrupted during the fix.**

   [data_efficiency_command.md:16](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_command.md:16) concatenates a truncated `MODE=SMOKE RUN=de` command with the next heading, loses the two recorded smoke commands and closing fence, and [line 66](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_command.md:66) starts with the orphaned fragment `00.json``. Restore the commands, heading, receipt paragraph and code-fence boundaries.

## Q3 — Production safety

The three specifically requested production surfaces are safe by static inspection:

- [run_evidence.py](/home/yixunhu/codespace/exp-38-data-efficiency/src/training/run_evidence.py:31) is observe-only numerically. State/RNG hashing does not draw RNG; wrappers return the original operation; wrappers are removed after step one and in teardown. With the flag enabled it adds synchronization/copy overhead, but I found no value, gradient or RNG mutation.

- With the flag absent, [logger_args_dict()](/home/yixunhu/codespace/exp-38-data-efficiency/train.py:51) removes the empty field from the live namespace before logger publication; `extra_callbacks()` subsequently sees it as absent and returns `[]`. The existing callback list and Trainer kwargs remain unchanged. The regression test is meaningful.

- `src/data/dataset.py` and `src/configs/dataset_configs/custom_metadata/AR_md.py` have no fix-round diff from `0d6edf46`; the flag-off/default path remains the old path, and `DatasetContractError` still bypasses generic resampling.

Production safety overall is not yet sufficient because of the launch/evaluation findings above.

## Q4 — Tests

The strongest new tests prove their intended contracts:

- The evidence mutation suite is comprehensive and non-vacuous.
- The real two-rank CPU DDP test genuinely proves PL 2.1’s post-build reset behavior and rank agreement. Its CUDA/SyncBN/full-model limitations are correctly disclosed.
- Signal forwarding, bounded grace, group termination, PID/start-time ownership and “never signal after reap” tests are meaningful.
- Metrics, receipt and complete-waveform admission mutation suites are strong.
- The dump implementation is statically equivalent to `ScoredWaveformDump.is_complete` for the shared predicate and additionally enforces the registered shape/cross-links. The test covers representative mutations, not every possible schema/type mutation, but is not vacuous.

The following claims are overstated or missing:

- The fake FULL lifecycle at [test_exp38_supervise.py:289](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:289) stubs common/campaign/E0/SMOKE/storage/snapshot and terminal receipt writing; its fake trainer creates no checkpoints. It proves orchestration, not an integrated terminal lifecycle.

- The lock test covers a direct child only, not Lightning’s spawned rank.

- The E0 “valid receipt” fixture uses unrelated cell hashes, so it cannot prove E0 artifact binding.

- There is no terminal-receipt all-fields mutation suite. Current tests cover writer refusal, closure, final bytes and manifest only at [test_exp38_receipts.py:326](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:326).

- The checkpoint mutation suite exercises outer schemas well, but misses arbitrary optimizer IDs, nested loop structures and callback values.

- There is no coverage for transient evaluator-tree changes, current eval-pin admission, duplicate per-cell execution, concurrent Neuronic attempts, inherited launcher variables, all-record-write failure or complete result-manifest membership.

## Q5 — Comparability and auditability

Results remain at risk of being wrong or non-comparable if:

- identically named but byte-different audio is used;
- evaluator/config files change during a cell;
- inherited distributed variables change topology/seed;
- a shallowly consumed terminal receipt misstates backend/environment/evidence.

Even if numeric results are correct, the missing receipt copies, incomplete result manifest, fail-open launch records and weak E0 cell-digest binding can make the campaign unauditable later.

The stored focused evidence is credible for the 511-test exp38 selection and the dry-run paths, but it is not a complete green repository regression: one CPU shard was still running, several other shards had classified failures, and `git diff --check` remains dirty.

## What I verified

- HEAD `e053c4c67698eb5b634be73e3802a556eee3e9b0`, base `2b42468c`, and the complete `0d6edf46..HEAD` fix diff.
- All local and Neuronic launch, ownership, snapshot, receipt and run-directory paths.
- E0, training, checkpoint, evaluation-cell, collection and table-publication admission paths.
- New tests and the committed focused/full-suite/dry-run evidence.
- Static review only: no tests, training, GPU commands, package/environment changes, file modifications or process interaction.