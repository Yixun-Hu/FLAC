# exp_15 yaw_aug — Codex CHAIN RE-REVIEW

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, read-only) · **Date:** 2026-08-13 · **Verdict: NO-GO** — F2/F3/F5/F6/F7/F8 CLOSED; F4 submission transaction still open (1 BLOCKING) + F1 identity completeness + new stale-final-record MAJOR. exp_11 chunk-kit interaction confirmed DISJOINT. Dispositions in worklog.

# Verdict: NO-GO

Do **not** launch the chain INITIAL at `001ce68`. Most F1–F8 repairs are genuine, but F4’s submission transaction remains race-prone and its post-`sbatch` crash recovery can permanently adopt an unrunnable child.

## Findings

1. **BLOCKING — F4 remains open: submission is not atomic under the advertised flock.** `yaw_aug_chain_state.py` locks only one helper invocation and releases the lock on exit ([yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:47)). After `intend` returns, the separate `squeue → sbatch → mark-submitted` sequence runs unlocked ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:197)). Two replayers can therefore obtain the same token, both see an empty queue, and both submit. Additionally:

   - `squeue` failure is suppressed and treated like “no job,” causing submission rather than fail-closed refusal ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:201)).
   - A crash after `sbatch` makes the parent non-successful, so the child’s `afterok:<parent>` dependency cannot release; recovery nevertheless finds and marks that child `SUBMITTED` ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:203), [dependency](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:209)).
   - The recorded `next_leg_command` omits the intent token, dependency, and state transition, so running the advertised recovery command directly is not idempotent ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1594)).
   - The state schema supports `parent_ckpt_sha256`, but `chain_advance()` never supplies it ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1569), [yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:159)).

   **Fix:** hold one per-boundary submission lock across state re-read, scheduler query, `sbatch`, and state publication; treat scheduler-query failure as fatal; verify the old parent’s terminal state before adoption; provide a state-aware recovery operation; and persist/verify the parent checkpoint SHA. The different-SHA/same-boundary refusal itself is correct.

2. **MAJOR — F1’s INITIAL-identity validation is incomplete.** The core audited-tip check is correct: the last registry entry’s step and SHA are bound to the fd-checked checkpoint ([yaw_aug_chain_preflight.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_preflight.py:98), [yaw_aug_chain_preflight.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_preflight.py:107)). However, the INITIAL-mode condition accepts the identity if **either** the manifest or registry says INITIAL; both must agree ([yaw_aug_chain_preflight.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_preflight.py:168)). The submit-time reader still returns only an unchecked path ([yaw_aug_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:124)), and neither the worker manifest nor registry records a chain flag/cap for the preflight to validate.

   **Fix:** independently require manifest mode and registry mode to be INITIAL; record and validate `chain=1`, cap `40000`, cadence/leg size, and initial target `2500`; perform readable-file/same-byte SHA validation in `chain_initial_manifest()` before submission.

3. **MAJOR, new — terminal chain outcomes are absent from the durable final record.** `FINAL_RECORD` is constructed and written before class 7/8 adjustments and before chain classes 12/13 ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1364)); the same stale value is printed after advancement ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1618)). A rate-gate breach can therefore durably say `classified rc=0` while the job exits 13.

   **Fix:** atomically publish a terminal post-advancement status containing the actual exit class, state-machine status, successor JID or no-submit reason, and rate-gate verdict.

## F1–F8 disposition

- **F1:** Partially closed; tip step/SHA, non-tip, fork, and snapshot-helper wiring are sound, but Finding 2 remains.
- **F2:** Closed. Every restart mints a fresh current-leg ID, scrubs `WANDB_RESUME`, records the INITIAL/root W&B ID, and readback verifies the fresh current ID—not its parent ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:855), [readback call](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1200)).
- **F3:** Closed for no-submit ordering. The class-7/8/9 and cap cases functionally drive the extracted real `chain_advance()` block, and `afterok` is present ([guardtests](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:1078)).
- **F4:** **Open/blocking**, per Finding 1.
- **F5:** Closed. The window math reproduces `200/212 = 0.943`; breach and insufficient data become class 13 before submission, INITIAL-only ([yaw_aug_rate_gate.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_rate_gate.py:46), [chain gate](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:1580)).
- **F6:** Closed: `LEG_STEPS=2500` is enforced in both submitter and worker ([yaw_aug_submit.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit.sh:213), [yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:286)).
- **F7:** Closed. The `44df1a2` launcher blob is identical to the rebased pre-chain parent’s blob; the golden is non-empty, unset/`CHAIN=0` match, and `CHAIN=1` differs ([guardtests](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:961), [golden](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_monolith_argv_golden.txt:1)). The printed-argv rather than NUL-tuple deviation is acceptable for this fixed, whitespace-free production argv.
- **F8:** Closed for the committed pin. `287316c`, `098d8ea`, `4221002`, and `e0bfa3a` yield 311 unique, well-formed ledger cases; all 311 pass somewhere, with no failures. The six `skip_env` cases are genuine absent-control-manifest cases and pass in the main checkout ([union result](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_14-21-03_union_coverage.log:2), [STRICT result](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-13_14-15-42_guardtests_chainfix_worktree.log:354)). Deviation 3 is accepted.

## Pin and exp_11 interaction

`HEAD == origin/check-equivariance-necessity == 001ce6838bc8d67a5f68eefc4292d46f71bfe78a`; the tracked launch closure is clean. The full worktree is not literally clean because untracked exp_11/exp_15 runtime artifacts remain, but the sanctioned gates intentionally use `--untracked-files=no`.

Commit `8ff4dbc` precedes the final exp_15 guard evidence. Operational write namespaces are disjoint:

- exp_11: `outputs_FLAC/exp11_<ARM>`, `exp11_<ARM>.lock`, `.submit_<ARM>.lock`, `.chunk_watchdog.lock`, `arm_launch_registry.json`.
- exp_15: `outputs_FLAC/exp15_YAWAUG`, `exp15_YAWAUG.lock`, `yaw_aug_launch_registry.json`, and run-local `yaw_aug_chain_state.json`.

The only interaction is intentional, read-only validation of exp_11’s VANL control registry/helper surfaces. No lock, job-name, state, output, or writable-registry collision exists. The production exp_15 run directory and launch registry are absent; the old lock file has no active kernel lock.

**Final verdict: NO-GO — do not launch the chain INITIAL at `001ce68`.**