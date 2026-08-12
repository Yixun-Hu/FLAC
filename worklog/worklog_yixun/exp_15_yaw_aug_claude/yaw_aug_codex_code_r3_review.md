# exp_15 yaw_aug — Codex CODE review, Round 3 (training launch kit)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox; self-reported generically as "API workspace agent") · **Date:** 2026-08-12 (clean rerun — the 2026-08-11 invocation was lost with the idle session) · **Commits reviewed:** `50a526d` `155d57a` `0dd59a9` `889898d` `5829ef2` · **Verdict: REVISE** — findings 1–6 fixed in-round before any smoke/launch; loop closure in `yaw_aug_worklog.md`.

# exp_15 `yaw_aug` — Round 3 code review

**Reviewer:** OpenAI Codex (API workspace agent; exact serving subversion not exposed, bounded read-only inspection) · **Date:** 2026-08-12  
**Commits:** `50a526d`, `155d57a`, `0dd59a9`, `889898d`, `5829ef2`  
**Verdict: REVISE — do not run the smoke or training launch yet.**

## Findings

1. **BLOCKING — the required smoke poisons the production launch registry and permits the wrong topology.**

   `SMOKE=1` accepts `32x2`, `16x4`, or `8x8` ([yaw_aug_train.sbatch:168](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:168), [yaw_aug_submit.sh:54](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:54)), although the approved smoke is explicitly the real 8×L40 rung ([plan_yaw_aug.md:186](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:186)).

   More seriously, every INITIAL—including a smoke—writes `arms["YAWAUG"]` into the production `yaw_aug_launch_registry.json` ([yaw_aug_train.sbatch:735](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:735), [yaw_aug_train.sbatch:745](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:745)). The required sequence “smoke → full launch” therefore makes the full INITIAL abort as already registered. Completion excludes smoke, but registration does not ([yaw_aug_train.sbatch:881](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:881)). DRYRUN exits before this block, so the 98-case suite never exercised it.

   **Fix:** hard-pin smoke to `8x8`/micro 8/NGPU 8; skip the production registry entirely for smoke or use a separate smoke-only registry. Add an extracted/temp-registry guardtest proving smoke leaves no `arms.YAWAUG` entry and the subsequent production INITIAL registers successfully.

2. **BLOCKING — completion is neither round-2-rigorous nor fail-closed.**

   The code chooses the newest checkpoint by mtime ([yaw_aug_train.sbatch:874](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:874)), hashes it without loading or validating it, and records `final_step` from the requested `MAXSTEPS`, not the checkpoint’s embedded step ([yaw_aug_train.sbatch:882](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:882), [yaw_aug_train.sbatch:895](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:895)). It does not verify a stable inode, safe-load the checkpoint, enforce `global_step == 40000`, compare the embedded YAWAUG config, or verify the exact EMA mirror/inventory established in Round 2.

   If no checkpoint exists, the block is silently skipped. If registry writing fails, it only prints a warning and a green job can still exit zero ([yaw_aug_train.sbatch:902](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:902)). That violates the launch acceptance criterion requiring a SHA-recorded 40k checkpoint.

   **Fix:** validate and hash one fd-pinned snapshot using the Round-2 contract: `weights_only=True`, before/after inode identity, strict non-boolean integer step 40000, canonical embedded config equality, exact EMA suffix/shape/dtype inventory, and SHA from that same inode. Missing checkpoint or verification/registry failure must turn an otherwise green job into a provenance failure. Add synthetic wrong-step/config/EMA, replacement-race, missing-checkpoint, and registry-write-failure cases.

3. **BLOCKING — the runtime closure omits the actual training split and the drift command can fail open.**

   The pinned dataset config loads tracked `data/AR/train.json` ([acousticroom_train.json:7](/n/fs/gatrdp/codespace/FLAC/src/configs/dataset_configs/AR/train/acousticroom_train.json:7)), but neither launcher nor submitter checks it ([yaw_aug_train.sbatch:221](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:221), [yaw_aug_submit.sh:83](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:83)). An uncommitted edit to that split therefore passes exact-HEAD binding and changes the samples actually trained.

   Both `git status` calls also discard stderr and ignore the command’s exit status; a Git/pathspec failure yielding empty stdout is treated as a clean tree.

   **Fix:** add the exact runtime split—preferably `data/AR/train.json`—to submitter and worker closure; make `git status` failure fatal; use quoted Git pathspecs so deletions remain detectable. Add mutation/deletion and forced-status-failure regressions.

4. **MAJOR — exp_15 has the same over-broad pending-job closure that `2b75036` fixed in exp_11.**

   Real launches require literal `HEAD == EXPECT_SHA` and include all of `src`, including `src/tests`, in drift ([yaw_aug_train.sbatch:221](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:221), [yaw_aug_train.sbatch:232](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:232)). Since `5829ef2`, the current branch moved only through test files under the training scope; a job submitted at that pin would already abort despite unchanged non-test training content.

   This fails safely, so it is not a scientific corruption path. A strict checkout freeze can make it survivable for one job, as the plan anticipated. It is nevertheless not operationally acceptable in this shared checkout with active sessions and potentially long queueing—the exact failure already killed exp_11 legs and led to reviewed fix `2b75036` ([yaw_aug_worklog.md:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:107)).

   **Fix:** port exp_11’s current content-scoped binding: exclude `src/tests` from drift and EXPECT-vs-captured-HEAD closure, retain all actual runtime files/helpers/configs plus `data/AR/train.json`, require a full commit OID, and re-read HEAD after the comparison. Add a synthetic test-only commit case.

5. **MAJOR — the banner gate is buffered correctly, but it does not prove the exact banner appeared before step evidence and class 8 masks unrelated failures.**

   The positive pieces are correct: `PYTHONUNBUFFERED=1` is exported to `torchrun` children ([yaw_aug_train.sbatch:136](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:136)), and the actual source prints exactly `yaw_aug ENABLED img_w=512 seed=42` with `flush=True` ([diffusion.py:349](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:349), [diffusion.py:355](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:355)).

   However:

   - `grep -qF` is substring matching, so `seed=420` satisfies the seed-42 gate.
   - Presence is checked before step presence without comparing file positions; a log with step evidence first and the banner later returns `OK` ([yaw_aug_train.sbatch:808](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:808)).
   - The guard suite tests banner-then-step, but not step-then-banner or prefix collisions ([yaw_aug_train_guardtests.sh:287](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:287)).
   - Class 8 is assigned unconditionally whenever the final verdict is not `OK`, so an early OOM/infrastructure failure before `on_fit_start` is reclassified as banner failure rather than retaining its real taxonomy ([yaw_aug_train.sbatch:947](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:947)).

   **Fix:** require an exact full line (`grep -Fx`) and compare first banner/step positions. Add `seed=420`, suffixed/prefixed text, and reversed-order cases. Set class 8 for definite `MISSING`, or for a green result lacking `OK`; preserve an existing nonzero class when the verdict is merely `PENDING`.

6. **MAJOR — the VANL manifest’s digest is correct, but parsing is not bound to the bytes that were verified.**

   The hard-coded digest equals exp_11’s current VANL registry value ([arm_launch_registry.json:107](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:107)), and torch/VAE comparisons are strict strings. But the launcher hashes the file with `sha256sum` and then Python reopens it for parsing ([yaw_aug_train.sbatch:541](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:541), [yaw_aug_train.sbatch:547](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:547)). A replacement between those opens permits unverified values to be trusted—the same snapshot-binding class fixed in Round 2.

   **Fix:** in one Python invocation, read manifest bytes once, hash those bytes against the VANL registry pin, and parse those same bytes. If a live registry cross-check is intended, load `arms.VANL.manifest_sha256` explicitly and assert it equals the reviewed constant before parsing.

7. **MINOR — the “98 passed / 0 failed” transcript contains unhandled guard-harness errors.**

   The count is genuinely 98 `PASS` records and zero `FAIL`, but the final green transcript contains two `sed` errors ([guardtests transcript:31](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-21-08_guardtests_r3.log:31)). `spool()` ignores `cp`/`sed` return codes ([yaw_aug_train_guardtests.sh:104](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:104)), and the restart-cap case incorrectly passes literal `-e` arguments ([yaw_aug_train_guardtests.sh:167](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:167)). The intended two substitutions still ran, so this does not invalidate the cap result, but the helper can produce future false greens.

   **Fix:** remove the stray `-e` arguments; make every copy/substitution failure return nonzero; assert the expected mutation exists before running each spooled launcher; regenerate a clean transcript.

8. **NIT — one provenance comment retains the superseded pre-rebase SHA.**

   The launcher says its copy commit was `a4bbe86` ([yaw_aug_train.sbatch:5](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:5)); the reconciliation maps that commit to current `50a526d` ([yaw_aug_worklog.md:105](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:105)).

   **Fix:** update the comment to `50a526d`.

## Six reported deviations

| # | Adjudication |
|---|---|
| 1. Invoke three exp_11 helpers rather than copy | **ACCEPT IN PRINCIPLE.** They are argument-driven and arm-agnostic, and none changed between `5829ef2` and current HEAD. Keep them in the corrected content closure; Findings 3–4 must be fixed. |
| 2. Retire init-hash gate N | **ACCEPT.** A one-arm cross-arm init hash is meaningless. The raw-byte construction test proves the control-plus-one-block config contract ([test_yaw_aug_arm_config.py:68](/n/fs/gatrdp/codespace/FLAC/src/tests/test_yaw_aug_arm_config.py:68)); the launcher independently pins both ViTs and width coupling. |
| 3. Drop `--extension` | **ACCEPT.** Extension contradicts the registered 40k endpoint. Ordinary restart preflight plus the 40k cap is the correct stricter contract. |
| 4. Retain unreachable restart-cap check | **ACCEPT.** It is sensible defense in depth. Repair the noisy/fail-open spool harness in Finding 7. |
| 5. Add class 8 | **ACCEPT INTENT; REVISE IMPLEMENTATION.** A green run proven unaugmented must become class 8, but `PENDING` must not erase an existing OOM/infrastructure/bug class. |
| 6. Backfill completion fields absent from exp_11 | **ACCEPT INTENT; BLOCKING IMPLEMENTATION DEFECT.** The schema extension is justified, but the current completion writer does not establish that the hashed artifact is the strict 40k YAWAUG checkpoint and can fail without failing the job. |

## Requested risk checks

- **Accumulation/topology:** no current path reaches `train.py` with accumulation other than 1; the value is read back from constructed argv. Production micro/NGPU are pinned 8/8, but smoke bypasses them—Finding 1.
- **Allowlist:** scope is exactly `train.py`, `defaults.ini`, `src`; shell `case` patterns match the whole path, `src/tests/*` does not match `src/tests_evil/...`, and Git emits no leading `./`. It is fatal in REAL mode. Current historical diff is 16 files rather than the transcript’s then-current 15 because of a later test file; all remain matched.
- **40k re-pin:** complete. All executable production pins/assertions and restart bounds are 40k. Remaining 100k/Q10 references are explanatory comments; 67,500 is only the exp_07 parity reference.
- **Argv parity:** dropping `--sync-batchnorm` is caught as a missing reference flag; changing `--precision` is caught because only exact `bf16-mixed` is admitted. The committed DRYRUN argv matches plan §3.2, including 8×8, accumulation 1, seed 42, SyncBN, bf16, and 2,500 cadence ([DRYRUN transcript:10](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-11_14-21-46_dryrun_r3.log:10)).
- **Copy identity:** at `50a526d`, target/source Git blob IDs match exactly against parent `de7c41d`: train `e7f6d547…`, submitter `011f5bc2…`, guardtests `6aed5778…`; file modes also match. The later exp_11 `2b75036` gate change necessarily makes the current originals differ.
- **Transcripts:** the red→green progression is coherent, final count is exactly 98 PASS records, and DRYRUN/submitter output is internally consistent. The final guard transcript is not clean because of Finding 7.

**VERDICT: REVISE.** Findings 1–3 are launch blockers. Findings 4–6 should be closed in the same Round-3 fix loop before regenerating guard/DRYRUN evidence and proceeding to the integrative full review.