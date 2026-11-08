**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-08 · **Round:** exp_38 code review 3 of 3 (2b42468c..61037926; r2 fix round e053c4c6..61037926)

**Verdict:** REQUEST-CHANGES

One remaining defect can make the trained curves non-comparable: the AR tree is authenticated only before a multi-day run, while training continues to read the mutable network tree. Additional HIGH findings affect snapshot immutability, GPU/process isolation, installed-package identity, and durable auditability.

## Q1 — Prior findings at HEAD

The r2 review’s table actually contains 12 `PARTIAL` plus one `NOT` r1 row, despite the §15 summary saying 13 `PARTIAL` plus one `NOT`. All 13 non-resolved table rows are included below.

| Prior finding | Status at HEAD | Evidence / remaining issue |
|---|---|---|
| r2 BLOCKING 1 — shallow terminal receipt consumer | **PARTIAL** | Consumption is substantially stronger at [exp38_registry.py:1595](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1595), with mutation coverage at [test_exp38_receipts.py:413](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:413). However snapshot identity is only syntactic, retained E0/environment/SMOKE records are only partly interpreted, L40 UUIDs need not be unique, and the CLI supplies no expected campaign/platform at [exp38_registry.py:1742](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1742) and [exp38_registry.py:1796](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1796). |
| r2 BLOCKING 2 — mutable evaluation checkout | **PARTIAL** | Cells now execute from the evaluation snapshot and reverify it at [exp38_eval.py:675](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:675); reverting the snapshot cwd would fail [test_exp38_eval.py:342](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:342). But the verifier permits unmanifested extra files; see HIGH 1. |
| r2 BLOCKING 3 — dataset names, not bytes | **PARTIAL** | The committed manifest correctly binds the split-derived file set and current bytes at [exp38_content_manifest.py:39](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_content_manifest.py:39), with same-name mutation coverage at [test_exp38_content_manifest.py:53](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_content_manifest.py:53). It is verified only before launch, so it does not bind the bytes subsequently read during training; see BLOCKING 1. |
| r2 BLOCKING 4 — inherited launcher variables | **RESOLVED** | Refusal and child sanitization cover rank, TorchElastic, PL, Lightning, and relevant Slurm variables at [exp38_registry.py:1205](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1205) and [exp38_supervise.py:226](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:226). Removal would fail [test_exp38_supervise.py:249](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:249). |
| r2 HIGH 1 — spawned ranks drop card locks | **RESOLVED** | The detached holder retains locks until the process group is empty at [exp38_supervise.py:331](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:331). The surviving non-inheriting grandchild test at [test_exp38_supervise.py:282](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:282) would fail without it. Exception paths remain a separate HIGH 2. |
| r2 HIGH 2 — Neuronic attempt race | **RESOLVED** | Fixed owner-scoped `mkdir` claim at [exp38_submit_neuronic.py:177](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_submit_neuronic.py:177), covered by the six-process test at [test_exp38_launchers.py:283](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_launchers.py:283). |
| r2 HIGH 3 — E0/SMOKE durable bindings | **PARTIAL** | A6000 online/offline E0 re-admission and SMOKE log/snapshot reopening are implemented at [exp38_eval.py:296](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:296) and [exp38_registry.py:1442](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1442). Neuronic explicitly disables E0 artifact re-admission at [exp38_registry.py:1357](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1357); see HIGH 5. |
| r2 HIGH 4 — optimizer/loop/callback checkpoint state | **RESOLVED** | Exact optimizer IDs, recursive loop template/counters, and callback values are checked at [exp38_registry.py:364](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:364). The mutation suite at [test_exp38_registry.py:413](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_registry.py:413) covers the formerly accepted cases. |
| r2 HIGH 5 — live evaluator pins absent from admission | **RESOLVED** | Online validation calls the live dependency probe at [exp38_eval.py:475](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:475), tested for all three pins at [test_exp38_eval.py:310](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:310). |
| r2 HIGH 6 — receipts not retained | **RESOLVED** | Exact sealed E0, environment and SMOKE copies plus the train log are retained at [exp38_registry.py:1523](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1523), then copied to the farm at [exp38_eval.py:608](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:608). |
| r2 HIGH 7 — incomplete result manifest | **PARTIAL** | Cell/run/campaign coverage was expanded at [exp38_eval.py:520](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:520), but E0 logs, evaluator weights/snapshot manifests are omitted, and several hashes are copied from declarations rather than recomputed; see HIGH 6. |
| r2 HIGH 8 — launch record fails open | **PARTIAL** | `Record` is strict at creation and explicit terminal updates at [exp38_supervise.py:196](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:196). The unexpected-error terminal path remains non-strict at [exp38_supervise.py:649](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:649); see HIGH 7. |
| r2 HIGH 9 — duplicate evaluation cells | **RESOLVED** | Atomic cell claims occur before evaluation at [exp38_eval.py:659](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:659), with concurrent coverage at [test_exp38_eval.py:321](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:321). |
| r2 HIGH 10 — import/environment drift | **PARTIAL** | The untracked/importable allowlist and origin checks are meaningful at [exp38_registry.py:1143](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1143). Installed-file bytes are not actually verified when a distribution has `RECORD`; see HIGH 4. |
| r2 HIGH 11 — backend gate checks discoverability only | **RESOLVED** | The exact child environment imports the production transformer and tests `flash_attn_func` binding at [exp38_supervise.py:242](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:242), covered at [test_exp38_supervise.py:332](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_supervise.py:332). |
| r2 NIT 1 — corrupted command sheet | **PARTIAL** | The sheet is restored at [data_efficiency_command.md:19](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_command.md:19), but this is documentation-only and no test detects recurrence, so it does not meet the requested code-plus-regression-test definition of `RESOLVED`. |
| r1 B1 — shallow E0/SMOKE consumers | **PARTIAL** | A6000 paths are strong, but Neuronic E0 and the terminal bundle still lack complete durable re-admission. |
| r1 B3 — evaluation not bound to campaign/GPU | **PARTIAL** | Campaign/GPU and snapshot execution are now bound, but unmanifested snapshot entries remain accepted at [exp38_registry.py:1008](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1008). |
| r1 B4 — no terminal training receipt | **PARTIAL** | Strong writer and substantial consumer exist, but HIGH 3 remains. |
| r1 B6 — evaluator dependencies/storage | **RESOLVED** | Live dependencies and storage are enforced and regression-tested at [exp38_eval.py:475](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:475) and [exp38_eval.py:766](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:766). |
| r1 H1 — partial checkpoint audit | **RESOLVED** | Exact state checks and mutation tests now close the identified holes. |
| r1 H4 — orphaning/unrelated signaling | **PARTIAL** | Normal training supervision and lock lifetime are strong, but unexpected exceptions can still abandon a live child, and evaluation has no persistent lock holder; see HIGH 2. |
| r1 H6 — Neuronic exactly-once | **RESOLVED** | Stable owner plus atomic attempt claim and concurrent test close the race. |
| r1 H8 — portable dataset binding | **PARTIAL** | Portable content identity exists, but it is not an immutable identity of the bytes used throughout training. |
| r1 H9 — environment/untracked identity | **PARTIAL** | Importable-tree handling is improved, but installed-byte identity remains incomplete. |
| r1 H10 — result order/manifest | **PARTIAL** | Publication ordering is correct; manifest completeness and current-byte hashing remain incomplete. |
| r1 H11 — missing high-risk tests | **PARTIAL** | Many requested tests were added, but the gaps listed under Q4 remain. |
| r1 N3 — incomplete test evidence | **RESOLVED** | For exp_38 scope, the committed focused run completed with 587 passes, and the full CPU suite completed with all 11 failures classified. The three exp_25 failures remain the explicitly documented merge-time decision. |
| r1 N4 — dirty `git diff --check` | **RESOLVED** | `git diff --check 2b42468c..HEAD` is clean at HEAD. This is a mechanical hygiene check rather than a behavioral test. |

## Q2 — Remaining and new findings

### BLOCKING

1. **The AR digest does not bind the bytes actually used during the run.** `[RESULTS]`

   FULL verifies the current network tree once at [exp38_supervise.py:526](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:526). Training then reads the `AcousticRooms` symlink for up to several days, while terminal publication merely passes the previously obtained constant into [exp38_registry.py:1543](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1543). A WAV, metadata JSON, or depth array changed after preflight can affect training while `TRAINING_COMPLETE` still records the pinned manifest.

   **Required change:** make the training snapshot’s `AcousticRooms` link point to an immutable, read-only filesystem snapshot or content-addressed release of exactly the manifested files, and bind that release identity in the receipt. A post-run rehash is useful defense-in-depth but is not sufficient by itself because a transient mutation could be restored. Add a lifecycle test that changes a targeted file after preflight and proves no terminal receipt can be published.

### HIGH

1. **Snapshot verification accepts unmanifested extra files.** `[RESULTS]`

   [exp38_registry.py:1008](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1008) checks every listed file/link but never walks the snapshot to reject additions. An added extension module or other importable can shadow listed source while `snapshot_digest()` remains unchanged. This applies to both training and evaluation snapshots.

   **Required change:** compare the complete lexical snapshot tree against `SNAPSHOT.sha256`, `SNAPSHOT.links`, their required parent directories, and nothing else; do not follow the dataset/output links. Reject every unexpected file, symlink, or importable directory before and after execution. Test extra `.py`, compiled-extension, and symlink entries added before and during a fake cell/run.

2. **Child ownership and card-lock lifetime are not exception-safe.** `[RESULTS]`

   Evaluation uses plain `subprocess.run` at [exp38_eval.py:695](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:695), while the lock is held only by its Python parent at [exp38_eval.py:769](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:769). If that parent alone dies, the evaluator can survive after the card unlocks. Training also creates its child before starting the holder at [exp38_supervise.py:593](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:593), and unexpected exceptions from holder creation, record updates, or gate callbacks reach [exp38_supervise.py:649](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:649) without terminating/reaping the live group.

   **Required change:** use a shared owned-process abstraction for evaluation and training; start a detached group-aware lock holder or otherwise transfer the lock before the child can proceed, and terminate/reap the owned group in every non-`SIGKILL` exceptional path. Test a killed evaluation parent with a surviving child, holder-start failure, and an exception raised by each gate callback after spawn.

3. **The terminal bundle is still not independently reproducible, and the publication CLI is context-free.** `[AUDIT/ROBUSTNESS]`

   At [exp38_registry.py:1625](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1625), `snapshot_sha256`, paths, and times receive only syntax/type checks. The retained E0 copy is reduced to verdict/closure, environment to platform, and SMOKE to a subset of its fields at [exp38_registry.py:1641](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1641). Published Neuronic bundles exclude `.exp38_tools`, so the snapshot manifest is unavailable. A coherent reseal of top-level and retained records can pass. The NAS command invokes the context-free CLI at [data_efficiency_neuronic_commands.md:53](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/data_efficiency_neuronic_commands.md:53).

   **Required change:** retain snapshot manifests and all durable SMOKE/E0 backing artifacts in the run bundle; fully validate internal environment hashes, SMOKE fields/backing log, snapshot closure, unique GPU UUIDs, and timestamp ordering. Make `training-check` require explicit expected platform/backend/closure/E0, or derive and validate them from a specified campaign tree. Update the NAS command accordingly. Add valid-looking snapshot-SHA, duplicate-L40, coherent multi-record reseal, and wrong-campaign CLI tests.

4. **Installed-package “byte identity” hashes RECORD metadata, not installed bytes.** `[RESULTS]`

   [exp38_registry.py:1249](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1249) hashes the text of each distribution’s `RECORD`. Editing an installed `.py` or shared library without changing `RECORD` yields the same environment receipt. The fallback tree digest is used only when `RECORD` is absent.

   **Required change:** parse `RECORD`, recompute every current file’s size/hash, verify recorded hashes where present, and derive the environment identity from the current bytes—not the RECORD text alone. Include unhashed RECORD entries through an actual-byte tree digest. Add a synthetic distribution test that mutates an installed file while preserving `RECORD`.

5. **Neuronic E0 admission trusts only a self-resealed JSON plus sidecar.** `[AUDIT/ROBUSTNESS]`

   Neuronic deliberately calls `validate_e0_receipt(..., readmit=False)` at [exp38_registry.py:1357](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1357). The existing test explicitly demonstrates that a changed E0 metric is accepted in this mode at [test_exp38_eval.py:192](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:192). A `.sha256` sidecar detects accidental edits but provides no independent authenticity after coherent resealing.

   **Required change:** copy an E0 admission bundle containing all eight metrics, streams and receipts to Neuronic and run the existing offline re-admission there, or anchor the bundle in an independently immutable/signed artifact. Test valid-looking alternate cell hashes and E0 HEAD with the Neuronic path.

6. **The result manifest still omits consumed artifacts and sometimes records claimed rather than current hashes.** `[AUDIT/ROBUSTNESS]`

   [exp38_eval.py:520](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:520) uses the dump manifest’s declared waveform hash without hashing `waveforms.npy`; [exp38_eval.py:537](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:537) copies the final-checkpoint SHA from the training receipt; and [exp38_eval.py:551](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:551) omits the eight E0 logs, evaluator weights, and evaluation snapshot manifests.

   **Required change:** rehash every manifested current file at collection and refuse mismatch. Add E0 logs, AGREE/VAE/VIT artifacts, `SNAPSHOT.sha256`/`SNAPSHOT.links`, and any other file online admission consumed. Extend the manifest test to assert these paths and to mutate the checkpoint, waveform array, E0 log, and evaluator weight after admission.

7. **A terminal receipt can survive without a terminal authoritative launch record.** `[AUDIT/ROBUSTNESS]`

   `write_training_receipt()` commits `TRAINING_COMPLETE` at [exp38_supervise.py:624](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:624), before the run lock is released and the terminal launch record is written at lines 639–640. If that strict write fails, the receipt remains consumable. Unexpected-error recording is also non-strict at line 650.

   **Required change:** make the terminal launch record and training receipt a cross-linked transaction with one final commit marker; `prepare` must require the sealed terminal launch record and matching digest. Use `strict=True` on every terminal error path, after first stopping/reaping any child. Add failures at the final record write and the outer exception path and prove no admissible completion remains.

8. **A legitimate Neuronic SDPA environment without `flash_attn` is refused.** `[AUDIT/ROBUSTNESS]`

   `backend_gate()` correctly permits absent `flash_attn` for SDPA at [exp38_supervise.py:257](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py:257), but `module_identity()` unconditionally requires all `CRITICAL_MODULES`, including `flash_attn`, at [exp38_registry.py:1235](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_registry.py:1235). Thus the first Neuronic SDPA SMOKE cannot create its environment receipt when flash-attn is intentionally absent.

   **Required change:** record an explicit `absent` identity for optional backend modules, or parameterize environment admission by backend. Flash mode must still require the registered version and transformer binding. Add Neuronic SDPA tests for both absent and overlay-shadowed flash-attn.

### NIT

1. **Evaluation preserves inherited `GIT_*` variables while installing `GIT_DIR`.** `[AUDIT/ROBUSTNESS]`

   [exp38_eval.py:647](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:647) removes Python/launcher/Slurm variables but retains `GIT_WORK_TREE`, `GIT_OBJECT_DIRECTORY`, `GIT_COMMON_DIR`, and similar variables. These can make `source_sha()` fail or resolve against unintended Git storage, causing a late false refusal.

   **Required change:** remove all inherited `GIT_*` entries before setting the one intended `GIT_DIR`; add an environment test with conflicting Git variables.

2. **The repaired command sheet has no regression guard.** `[AUDIT/ROBUSTNESS]`

   The content is currently correct, but no test would fail if its code fences or canonical launch/publication commands were corrupted again.

   **Required change:** add a static test checking balanced fences and the required current command blocks, including the strengthened Neuronic `training-check` invocation.

## Targeted checks with no additional finding

- Evaluation resolution itself is correct: the no-flash launcher places the snapshot root ahead of site-packages at [exp35_no_flash_launch.py:84](/home/yixunhu/codespace/exp-38-data-efficiency/overlays/no_flash/exp35_no_flash_launch.py:84); `src`, repo `AGREE`, configs and splits come from the snapshot; weights and `outputs_FLAC` use its registered links. `GIT_DIR=<worktree gitdir> git rev-parse HEAD` returned `61037926…` in the read-only probe. Bytecode is disabled, and intended outputs leave through the output link.

- The AR manifest’s file set matches what `dataset.py` and `AR_md.py` read: targets, target/context metadata, context WAVs restricted to the split, and receiver depth arrays. Its three-component grouping produces the committed 729 groups. The issue is run-time mutability, not file-set derivation.

- A legitimate Neuronic run copied to the NAS is prepared correctly: [exp38_eval.py:599](/home/yixunhu/codespace/exp-38-data-efficiency/worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_eval.py:599) derives the original training directory from the receipt argv, so `ModelCheckpoint.dirpath` is not incorrectly compared with the NAS destination.

- The optimizer-ID, loop-template and callback pins are compatible by static inspection with this `train.py`, `--logger none`, and both 2×32 and 8×8. Evidence callbacks have no checkpoint state, topology does not alter parameter ordering, and the exact formerly accepted mutations are covered. No new real exp_38 endpoint exists yet, so this conclusion rests on the registered real YAW/B-F/dc_van endpoints plus the mutation suite.

- The training lock holder intentionally retains the card indefinitely while any process in the owned group remains; that is fail-safe. In its normal path I found no early release. The remaining problem is holder startup/exception handling and the separate evaluation lock path.

- The untracked allowlist and launcher-variable rules do not falsely reject the documented clean campaign/Slurm flows. Neuronic retains allocation-level Slurm facts while stripping rank variables from the torchrun child.

## Q3 — Production safety

The requested numerical production surfaces are unchanged by `e053c4c6..HEAD` and remain safe by static inspection:

- [run_evidence.py:31](/home/yixunhu/codespace/exp-38-data-efficiency/src/training/run_evidence.py:31) hashes state/RNG without drawing randomness. Its wrappers at [run_evidence.py:77](/home/yixunhu/codespace/exp-38-data-efficiency/src/training/run_evidence.py:77) forward original arguments/results and are removed after step one or teardown. It adds synchronization/copy overhead when enabled but no numerical mutation found.

- With the flag absent/empty, [train.py:42](/home/yixunhu/codespace/exp-38-data-efficiency/train.py:42) adds no callback, and [train.py:51](/home/yixunhu/codespace/exp-38-data-efficiency/train.py:51) removes the empty field from the live argparse dictionary before logger publication. The regression at [test_exp38_receipts.py:613](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_receipts.py:613) is meaningful.

- The default `dataset.py`/`AR_md.py` path remains unchanged. `restrict_to_split=False` preserves legacy context selection; only explicit contract violations bypass generic resampling at [dataset.py:362](/home/yixunhu/codespace/exp-38-data-efficiency/src/data/dataset.py:362).

Production numerical behavior is therefore unchanged, but campaign-level safety is not shippable until the `[RESULTS]` findings above are fixed.

## Q4 — Tests

The new tests genuinely prove:

- same-name AR byte changes are detected;
- exact checkpoint optimizer/loop/callback mutations refuse;
- launcher variables are rejected/stripped;
- concurrent attempt and per-cell claims admit one owner;
- the training lock holder survives a killed supervisor while a non-inheriting descendant remains;
- online/offline A6000 E0 re-admission, live evaluator pins, and retained receipt/log/evidence checks work.

Missing or overstated coverage:

- No dataset test changes a file after preflight but before terminal publication.
- The snapshot execution test at [test_exp38_eval.py:342](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:342) mocks `verify_snapshot`; it does not test extra-file refusal, real no-flash resolution, `GIT_DIR`, or output-link behavior.
- No test kills an evaluation parent while its child survives, or injects a post-spawn training exception.
- Terminal receipt mutations are one-field-at-a-time. They do not test a coherent reseal, a valid alternate snapshot SHA, duplicate Neuronic UUIDs, or the context-free CLI.
- [test_exp38_eval.py:202](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:202) explicitly blesses Neuronic’s lack of E0 artifact re-admission.
- No test mutates an installed distribution file while leaving `RECORD` unchanged, or exercises Neuronic SDPA with flash-attn absent.
- The result-manifest test at [test_exp38_eval.py:546](/home/yixunhu/codespace/exp-38-data-efficiency/src/tests/test_exp38_eval.py:546) checks only a subset of paths and does not prove current-byte hashing.
- The record test checks the `Record` class, not the non-strict outer exception or terminal commit ordering.

The stored focused summary reports 587 passes. The full CPU regression completed with 6,567 passes and 11 classified failures: three planned exp_14/exp_25 byte-pin conflicts, one flaky timing test, and seven pre-existing GPU-only failures. I did not execute either suite.

## Q5 — Overall correctness and auditability

If left unfixed, the following can make results wrong or non-comparable:

- data changes after the preflight manifest pass;
- an extra snapshot importable shadows the sealed source set;
- installed package bytes drift while RECORD text remains unchanged;
- an evaluator or trainer survives its lock-owning supervisor and overlaps another GPU workload.

The remaining terminal/E0/manifest/record issues primarily threaten later auditability and can falsely publish or accept an incomplete provenance bundle, although `prepare` currently catches some wrong-campaign cases after NAS publication.

## What I verified

- HEAD/base/fix-round commits and the complete `e053c4c6..61037926` diff.
- `git diff --check 2b42468c..HEAD`: clean.
- Every r2 finding and every non-resolved r1 table row against code and tests.
- Training/evaluation snapshots, no-flash import order, GIT_DIR behavior, links, receipts, Neuronic publication/prepare, process ownership, content manifest, checkpoint schema, collection and publication.
- The committed manifest identity/coverage and stored AR verification PASS.
- The committed 587-pass focused summary, full CPU regression classification, and all-path dry-run record.
- Static/read-only review only: no test, training, evaluation, launcher, scheduler, GPU, package, file-write, or process-control action.