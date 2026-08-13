# exp_15 yaw_aug — Codex CHAIN-ROUND review

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox) · **Date:** 2026-08-13 · **Commits reviewed:** `51a8c34` `9944863` `84c382b` `7dda3e2` `a542ea9` · **Verdict: NO-GO** — 5 BLOCKING (leg-2 preflight admission; W&B resume=must; epilogue ordering/lock race; non-idempotent submission/registry; unwired waiver rate gate) + 3 MAJOR; PL 2.1 resume core adjudicated SOUND. Fix loop follows; dispositions in `yaw_aug_worklog.md`.

# Verdict: NO-GO

The chain must not launch after push. Leg 1 can run, but the reviewed implementation cannot safely advance to leg 2, and the self-chaining epilogue is neither idempotent nor ordered after all per-leg failure gates.

## Findings

1. **BLOCKING — every chain RESTART is rejected by the inherited checkpoint preflight before Lightning runs.** Introduced/exposed by `51a8c34`.

   The INITIAL manifest records `max_steps 2500` ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:914)). Leg 2 constructs preflight arguments with `--max-steps 5000` and the INITIAL manifest ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:807)). But the exact helper blob reviewed at `a542ea9`, `fa_orbit_ckpt_preflight.py:81-105`, requires the manifest’s `max_steps` and commit to equal the restarting job’s values. It therefore reports `manifest max_steps '2500' != 5000`.

   This also rejects a later leg after an unrelated exp_11 commit moves HEAD, despite exp_15’s content-scoped start gate permitting such a move. The DRYRUN transcript never reaches preflight because DRYRUN exits at [yaw_aug_train.sbatch:670](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:670).

   **Fix:** add an exp_15 chain-aware preflight contract. It must allow increasing per-leg `Trainer(max_steps)` up to cap 40,000 while binding the original launch identity, and require the resume checkpoint’s step and SHA to equal the last audited registry tip. Pass the chain registry/cap/target explicitly. Do not rely on the currently dirty, unreviewed exp_11 helper edits.

2. **BLOCKING — W&B resume semantics independently kill leg 2.** Exposed by `51a8c34`.

   Every RESTART reuses the INITIAL W&B ID with `WANDB_RESUME=must` ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:842)), while leg 2 changes both `max_steps` and `ckpt_path`. `train.py` pushes that changed config at [train.py:193](/n/fs/gatrdp/codespace/FLAC/train.py:193). This is the exact failure already demonstrated and fixed for exp_11 in `3847212`: jobs 3684149/50 died because the resumed W&B run refused those config changes.

   **Fix:** give each chain RESTART a fresh lineage-tagged W&B run ID, do not export `WANDB_RESUME`, and record the parent W&B ID in the manifest/registry. Port the `3847212` behavior and add a regression for changed `max_steps` plus `ckpt_path`.

3. **BLOCKING — the successor is submitted before the current leg’s final acceptance, and it can race the still-held run lock.** Introduced by `51a8c34`; falsely certified structurally by `9944863`.

   Submission occurs at [yaw_aug_train.sbatch:1305](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1305), before:

   - final-record tee and transcript-copy provenance can set class 7 ([yaw_aug_train.sbatch:1324](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1324));
   - the final banner classification can set class 8 ([yaw_aug_train.sbatch:1360](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1360)).

   Thus a successor can exist even though its parent later fails a retained per-leg gate. Guardtest line 961 only proves “audit condition appears before submission”; it cannot detect later condemnation.

   Also, the successor has no Slurm dependency. If it backfills immediately, it reaches the nonblocking flock while the parent still owns it and aborts ([yaw_aug_train.sbatch:793](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:793)).

   Audit failure itself correctly becomes class 9 with no submission, and the final leg correctly emits END without submitting. Those subpaths pass.

   **Fix:** determine the final per-leg verdict—including banner and log provenance—before advancing. Submit the child with an `afterok:<current-job>` dependency so it cannot contend with the parent’s lock. Add functional cases proving class 7, class 8, class 9, and final-cap legs submit nothing; submission failure alone must remain class 12.

4. **BLOCKING — submission/recovery is not idempotent, and it corrupts manifest provenance.** Introduced by `51a8c34`.

   - `next_leg_command` and status writes ignore append failures ([yaw_aug_train.sbatch:193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:193)).
   - The worker manifest was already copied and SHA-recorded in the registry at lines 936/971 or 1009. Appending chain state later changes the registered file, leaving `manifest_sha256` stale.
   - Each replay creates a random new submission intent ([yaw_aug_submit.sh:324](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:324)). The registry refuses only the same Slurm job ID, not a second job for the same resume/target boundary ([yaw_aug_train.sbatch:1004](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1004)).
   - A crash after audit but before command publication orphans the chain. A crash after `sbatch` succeeds but before status publication makes manual replay double-submit.
   - `legs` rejects duplicate steps but does not require monotonic order, continuity from the previous tip, or parent-checkpoint hash equality ([yaw_aug_train.sbatch:1249](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1249)).

   **Fix:** use an atomic registry state machine keyed by `(arm, target_step)`—for example `AUDITED → INTENDED → SUBMITTED`—under a dedicated flock. Record parent step/SHA, command, unique intent token, and JID. On replay, return the existing submission instead of issuing another. Keep launch manifests immutable; store epilogue state in a separate hashed record.

5. **BLOCKING — the standing waiver is cited, but its mandatory post-hoc rate gate is not enforced.**

   The waiver requires both 0.849 and 0.843 windowed floors and says breach stops the chain ([yaw_aug_worklog.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_worklog.md:162); [plan_yaw_aug.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/plan_yaw_aug.md:238)). The implementation merely embeds those numbers in a reference string ([yaw_aug_submit.sh:170](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:170)); no launcher or epilogue evaluates or records either floor before submitting leg 2. With the default 2,500-step INITIAL, both endpoints are available before its epilogue.

   The ordinary scope is otherwise correct: environment identity, VRAM floor, and banner watcher still execute per leg at launcher lines 700, 776, and 1079. Only the acceptance-record/waiver promotion gate is skipped for RESTART. Each submission-intent manifest cites the standing waiver.

   **Fix:** add an INITIAL-only, tested window-rate evaluator before chain advancement. Record both rates and refuse successor submission on either breach. Also make `chain_initial_manifest()` validate the INITIAL entry’s mode, chain flag/cap, readable manifest, and same-byte registry SHA; it currently returns an unchecked path at [yaw_aug_submit.sh:124](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:124).

6. **MAJOR — leg-boundary math is safe for ordinary inputs, but the tunable leg size is incompatible with its wall-time pin.**

   For representable inputs, no accepted worker path can produce a target above 40,000 or off the 2,500 cadence: inputs are aligned, the sum is capped, and the result is rechecked at [yaw_aug_train.sbatch:274](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:274). INITIAL 0→2,500, mid legs, and 37,500→40,000 are correct.

   But any aligned `LEG_STEPS` is accepted while every leg receives the fixed 1:30 limit. The suite explicitly blesses 2,500→12,500 with `LEG_STEPS=10000` at [yaw_aug_train_guardtests.sh:894](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:894), although approximately 10,000 seconds already exceeds 1:30 before startup/audit.

   **Fix:** either require `LEG_STEPS=2500`, or introduce reviewed step-to-time pins. Add an upper numeric bound before Bash arithmetic to fail closed on overflow-sized digit strings.

7. **MAJOR — CHAIN=0 compatibility evidence is not airtight, and registry behavior is not monolith-compatible.**

   The argv comparison uses `git show "${HEAD_SHA}:${LAUNCHER}"` at [yaw_aug_train_guardtests.sh:913](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:913), so it compares the worktree launcher against the same post-chain launcher at HEAD. It does not compare against the reviewed pre-chain behavior (`44df1a2` or an immutable golden argv). Static inspection indicates the actual CHAIN-unset training argv remains unchanged, but this test does not prove it.

   Moreover, the completion writer appends a `legs` entry even when `CHAIN=0` and prints `CHAIN END` for a monolith ([yaw_aug_train.sbatch:1249](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1249)). That is not byte-compatible registry/output behavior.

   **Fix:** compare a NUL-delimited argv against a fixed pre-chain golden or `44df1a2`, testing both unset and `CHAIN=0`. Gate the new `legs` schema and chain wording on `CHAIN=1`; retain the reviewed monolith completion writer unchanged.

8. **MAJOR — the two-transcript “union covers every case” claim is false, and SKIP handling is not fail-safe.** `84c382b`/`7dda3e2`/`a542ea9`.

   Both committed transcripts skip the real-mode allowlist case at line 58, so it is green in neither transcript. Literal counts are:

   - clean-worktree transcript: **3 SKIP** records, not 2;
   - main-checkout transcript: **18 SKIP** records, not 11;
   - zero literal FAIL records in both.

   See [clean transcript](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-46-29_guardtests_chain_worktree.log:58) and [main transcript](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_12-51-16_guardtests_chain.log:58).

   `closure_clean()` also suppresses Git stderr/status in a command substitution ([yaw_aug_train_guardtests.sh:139](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:139)), and the suite can exit zero with all later cases skipped.

   **Fix:** add a strict clean-evidence mode in which Git-status failure or any required SKIP fails the suite. Point the real-mode spooled launcher’s literal `REPO` at the clean worktree, then commit a transcript where that case actually passes. Add functional epilogue-order/idempotency tests rather than the current static awk check.

## PL 2.1 resume/fencepost adjudication

The Lightning behavior itself is correct once Findings 1–2 are fixed:

- `train.py` forwards `args.max_steps` into `Trainer` at [train.py:78](/n/fs/gatrdp/codespace/FLAC/train.py:78), installs `ModelCheckpoint(every_n_train_steps=2500)` at [train.py:182](/n/fs/gatrdp/codespace/FLAC/train.py:182), and passes `ckpt_path` to `fit()` at [train.py:230](/n/fs/gatrdp/codespace/FLAC/train.py:230).
- PL restores loop progress, so `global_step` resumes at 2,500.
- PL stops when `global_step >= max_steps` ([fit_loop.py](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:165)).
- `on_train_batch_end` runs after the optimizer step, and ModelCheckpoint saves when `global_step % 2500 == 0` ([model_checkpoint.py](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:285)).

Therefore a valid step-2,500 checkpoint plus `--max-steps 5000` performs exactly 2,500 additional optimizer steps, saves `*-step=5000.ckpt`, and then stops. There is no 4,999/5,001 fencepost and no “stop without boundary save” path.

## Coder deviations

1. **Per-leg registry entries:** **ACCEPT IN PRINCIPLE; REVISE IMPLEMENTATION.** Stronger provenance is desirable, but entries must form a tip-bound hash chain, be idempotent by target, link to job/manifest/parent SHA, and be CHAIN-only.

2. **Two-run guardtest pattern:** **ACCEPT THE PATTERN; CURRENT EVIDENCE REJECTED.** Complementary environments are reasonable on this shared checkout, but coverage needs machine-enforced reconciliation and at least one strict run for every required case.

3. **Suite no-mutation hardening:** **ACCEPT.** Redirecting the tracked acceptance record to the temp spool and checking before/after state is the correct repair from `84c382b`. It should remain.

## Current scheduler/worktree interaction

- Job **3687499** is still PENDING, currently estimated for 2026-08-13 23:40, with `EXPECT_SHA=bd6d8b9`. Because the reviewed launcher/submitter surfaces have changed, it will fail the content-binding gate before acquiring the lock or writing the registry. It should not corrupt the chain.
- No `yaw_aug_launch_registry.json` exists. `outputs_FLAC/exp15_YAWAUG.lock` contains stale smoke metadata but has no active kernel flock.
- exp_11 jobs **3687569–3687573** are pending under separate output namespaces and locks. They cannot directly corrupt exp_15. However, the currently dirty exp_11 preflight/helper files are inside exp_15’s closure, so exp_15 submission currently fails closed.
- The branch is **ahead 6 / behind 4** relative to upstream. A direct push is not currently possible; reconciliation will rewrite or merge the reviewed pin, after which exact-pin evidence must be regenerated.

**Final verdict: NO-GO.** Fix the two leg-2 admission blockers, epilogue ordering/dependency, idempotent registry state, mandatory rate gate, and evidence gaps; then rerun this chain-round review before any launch.