# exp_15 yaw_aug — Codex INTEGRATIVE FULL review (pre-launch)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox) · **Date:** 2026-08-12 · **Range:** `d3a0312..b1bcd38` (exp_15 surfaces) · **Verdict: NO-GO** — F1/F2/F3 fixed in the full-fix loop, F4 resolved by branch state, F5 deferral recorded (eval kit gets its own rounds + second integrative review); dispositions in `yaw_aug_worklog.md`.

# exp_15 `yaw_aug` — integrative full review

**Reviewed range:** `d3a03129f4b070881ae1a8c8d37e6fb8321cb8ba..b1bcd38735a4d4b6e9b5225802c17af5c995ef99`  
**Control pin:** `81ddac372076ea92751ae09cbaf371df70f396e5`  
**Runtime evidence:** `09bfa6e` — 145/145 guardtests, DRYRUN, 495 pytest passes; closure recorded at `b1bcd38`.

The three round closures remain accepted; none of their findings are reopened below.

## Findings

1. **BLOCKING — the completion audit is protected while queued, but late-bound after the job starts.**

   The queued-job closure correctly includes `yaw_aug_record_control.py` and the exp_11 helpers ([yaw_aug_train.sbatch:242](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:242)), and EXPECT-vs-HEAD content binding catches committed or dirty changes before startup ([yaw_aug_train.sbatch:252](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:252)). Thus a signature change cannot slip under a **queued** job.

   However, after training finishes, the worker imports that module again from the live shared checkout ([yaw_aug_train.sbatch:1003](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1003)); there is no second closure/hash check. The end-of-run W&B reader and classifier are likewise executed live ([yaw_aug_train.sbatch:960](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:960)). A mid-run edit can therefore change admission behavior or turn an otherwise successful 11-hour run into class 9.

   The current calls do match the recorder signatures exactly: `snapshot_checkpoint(path)`, `canonical_bytes(obj)`, `summarize_ema(state_dict)`, and `canonical_sha256(obj)` ([yaw_aug_record_control.py:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:127), [yaw_aug_record_control.py:156](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:156), [yaw_aug_record_control.py:251](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py:251)).

   **Fix:** snapshot the recorder, arm config, classifier, and W&B helper into a run-owned directory immediately after the start closure; record their hashes in the manifest and use only those copies. Add a guardtest that mutates the live recorder after snapshot and proves the completion path remains bound.

2. **BLOCKING — plan §7 rung 4, the bounded real-AR readback, has not been performed or recorded.**

   The plan explicitly requires a few records through the actual dataloader, checking `[3,256,512]`, finite values, pose shapes, roll equivalence, norm preservation, and unchanged z ([plan_yaw_aug.md:184](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:184)). No such evidence exists in the worklog or transcripts; the notebook instead proceeds directly from round closure to full review and smoke ([yaw_aug_worklog.md:118](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:118)).

   **Fix:** before smoke, perform the bounded CPU real-data readback through `acousticroom_train.json`, tee the assertions, and append the exact command/result to the worklog.

3. **MAJOR — the smoke is not storage-light and does not enforce promotion to production.**

   `SMOKE=1` defaults to 30 steps with `checkpoint_every=10` ([yaw_aug_train.sbatch:175](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:175)); `train.py` always installs that checkpoint callback ([train.py:182](/n/fs/gatrdp/codespace/FLAC/train.py:182)). This contradicts the registered storage-light smoke ([plan_yaw_aug.md:186](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:186)) and can write roughly three 724 MB checkpoints while distorting the step-rate measurement.

   In addition, the smoke exit code does not enforce the `≥0.9×` rate or peak-VRAM criteria. The worklog says “submit both,” but there is no dependency or promotion record; smoke and production share the same lock ([yaw_aug_train.sbatch:663](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:663)). Independently queued jobs can start in either order or make one abort on the lock.

   **Fix:** set the smoke checkpoint interval beyond its maximum step and verify no checkpoint lands; capture rate/VRAM in a smoke acceptance record. Submit production only after that record passes—do not queue both independently.

4. **BLOCKING — the live checkout is not currently the pushed review pin.**

   `origin/check-equivariance-necessity` is at `b1bcd38`, but live HEAD is the rewritten `e684a09`, **ahead 5 / behind 6**. Their exp_15 training closures are content-identical, but the submitter always exports current HEAD ([yaw_aug_submit.sh:97](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:97), [yaw_aug_submit.sh:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:107)). Launching now would therefore record unpushed `e684a09`, not reviewed/pushed `b1bcd38`.

   The exp_14 tracked artifacts are dirty, so checkout/rebase/stash in this shared tree is unsafe. Five exp_11 8×L40 jobs, `3684149–3684153`, are currently pending. Their content-scoped closure means exp_15 kit/test/worklog fixes are safe, but any new non-test `src/` change could abort them.

   **Fix:** coordinate with the exp_14 owner, reconcile without stashing or overwriting artifacts, and confirm the submission SHA is both pushed and the reviewed closure. Rerun DRYRUN at that exact SHA.

5. **MAJOR — plan §6.7–§6.8 were silently left out of this purported whole-experiment `full` review.**

   The planned eval kit and tested collector are specified at [plan_yaw_aug.md:150](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:150), but neither exists at `b1bcd38`. Consequently the control record is strong but its promised per-cell rehash gate is not wired, and the §8 eval acceptance criteria are not executable.

   This need not prevent training if explicitly deferred, but then this marker is “training-launch full,” not the final integrative review of the complete experiment.

   **Fix:** record the deferral explicitly; implement §6.7–§6.8 with their TDD/review loops and run another integrative review before any eval submission. Sections §6.9–§6.10 are legitimately results-dependent and remain pending.

6. **MINOR — RESTART provenance is not added to the registry.**

   The registry initializes a `restarts` object but only INITIAL mode writes a record ([yaw_aug_train.sbatch:826](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:826)). Restart manifests exist, so the lineage is recoverable, but the declared registry schema is incomplete.

   **Fix:** append a restart entry containing job, manifest hash, source checkpoint hash/step, and launch UUID.

7. **NIT — evidence bookkeeping has two small inconsistencies.**

   The allowlist commentary says 15 files ([yaw_aug_pin_allowlist.txt:14](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_pin_allowlist.txt:14)); the immutable recomputation is **16 gate-scope files**: 13 tests plus the three production files `yaw_rotation.py`, `diffusion.py`, and `factory.py`. All 16 are matched, so the gate itself is correct.

   Also, the 495-pass transcript records the result but not the exact pytest argv. Record the exact seven adjacent suites named by commit `09bfa6e`.

## Cross-round seams that pass

- The armed config is exactly `{"enabled": true, "img_w": 512, "seed": 42}` ([FLAC_AR_YAWAUG.json:195](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json:195)).
- The launcher checks that literal block, vanilla conditioning, EMA, both ViTs, and `ViT img_w == yaw_aug.img_w == 512` ([yaw_aug_train.sbatch:346](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:346)); the argv separately pins training seed 42 ([yaw_aug_train.sbatch:438](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:438)).
- Factory validation and wrapper enforcement agree ([factory.py:21](/n/fs/gatrdp/codespace/FLAC/src/training/factory.py:21), [diffusion.py:197](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:197)).
- Wrapper and watcher banner text match exactly: `yaw_aug ENABLED img_w=512 seed=42` ([diffusion.py:349](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:349), [yaw_aug_train.sbatch:895](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:895)).
- The hook is confined to `training_step` ([diffusion.py:476](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:476)); validation/test have no augmentation branch. The absent path passes no new kwargs and remains protected by the `d3a0312` golden regression.
- No quiet scientific bias was found in seed derivation, global RNG isolation, target preservation, or the control-admission contents.

## Plan §7 ladder

| Rung | Status |
|---|---|
| 1. Static | **Partial.** Read-only syntax/JSON checks and executable-surface `diff --check` passed during this review. No committed rung transcript; literal whole-diff checking fails only on raw pytest/review transcript whitespace. |
| 2. Pytest | **Satisfied.** `495 passed` at evidence commit `09bfa6e` ([transcript:25](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_10-45-10_pytest_r3fix.log:25)). |
| 3. Tiny synthetic forward | **Satisfied.** Real wrapper/DiT steps with augmentation on and off, including RNG and geometry assertions ([test_yaw_aug_training.py:752](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_training.py:752)). |
| 4. Real-data readback | **Pending — blocking.** No evidence. |
| 5. Guardtests + DRYRUN | **Satisfied.** 145/145 and zero submissions ([guard transcript:172](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-12_10-42-44_guardtests_r3fix.log:172)); exact production argv recorded in the DRYRUN. |
| 6. 8×L40 smoke | **Pending**, after Findings 1–4 are closed. |

## Plan §8 acceptance checklist

| Criterion | Status |
|---|---|
| Pushed pinned SHA reported | **Pending / blocking:** pushed `b1bcd38` ≠ live `e684a09`. |
| Current control-to-pin allowlist green | **Satisfied statically:** 16 gate-scope files, three non-test production files, all matched. |
| 1 node × 8 L40, micro 8, effective 64 | **Configured; runtime pending.** |
| SyncBN active, accumulation 1 | **Configured and guard-tested; runtime pending.** |
| Grad-checkpointing gate | **Satisfied statically.** |
| Exact banner before step 0 | **Static seam satisfied; runtime pending.** |
| ≥1 step, no OOM/NaN | **Pending smoke.** |
| Rate ≥0.9× VANL and peak VRAM accepted | **Pending; promotion gate currently absent.** |
| Production checkpoints every 2,500 | **Configured; runtime pending.** |
| Step-40k checkpoint audited and SHA-recorded | **Pending; late-bound import must be fixed first.** |
| Eval argv verified in manifests | **Pending; eval kit absent.** |
| Metrics JSON + 6,337-tuple sidecar | **Pending; eval kit absent.** |
| G1–G5 pass | **Pending; collector absent.** |

## Verdict

**NO-GO.**

Do not submit the smoke yet. Close Findings 1–4, record rung 4, regenerate guard/DRYRUN evidence, and rerun the affected integrative review. Findings 5–7 may be deferred as stated, but §6.7–§6.8 and a second full review are mandatory before evaluation.