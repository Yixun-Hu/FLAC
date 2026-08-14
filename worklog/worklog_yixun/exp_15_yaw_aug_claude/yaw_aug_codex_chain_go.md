# exp_15 yaw_aug — Codex chain GO verdicts (verification chain)

**Reviewer:** OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only) · Chain verification sequence: chain review NO-GO (8) → re-review NO-GO (F4+2) → final-verify NO-GO (4) → GO check NO-GO (3) → **GO check #2 2026-08-14: ZERO FINDINGS — GO at pin `fb4fc74`**.

## Findings

None.

- `canonical_state()` uniquely resolves `OUT_OF_ME+` to `OUT_OF_MEMORY`; `COMP` collides between `COMPLETED` and `COMPLETING` and therefore fails closed. All `sacct` state reads specify `State%-40`; name lookup also specifies `JobID%-20`. `squeue` uses full-state `%T` ([yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:152)).
- SUBMITTED replay probes the recorded successor, treats query failures/unknown state as fatal, records terminal successors as corpses, rotates the token, and submits fresh—all under the same lock ([yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:234), [yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:274), [yaw_aug_chain_state.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_chain_state.py:431)).
- The composed test invokes the real helper through all three phases and verifies exactly two submissions, corpse persistence, and no third submission ([yaw_aug_train_guardtests.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:1652)).
- The completeness test extracts the real writer and covers all five added keys plus a non-empty repeated `snapshot_sha256` line ([yaw_aug_train_guardtests.sh](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train_guardtests.sh:1571), [yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:924)).
- Scheduler audit confirmed: the submitter has no executable scheduler reads; the launcher’s sole `squeue` reads `%l` only ([yaw_aug_train.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_train.sbatch:667)).
- Committed evidence is clean: STRICT `373/0` at `b1314b1`, with every new case passing ([STRICT transcript](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-14_10-27-46_guardtests_chainfix4_worktree.log:405)); union coverage is `376/376`, zero failures ([union transcript](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_2026-08-14_10-36-32_union_coverage_chainfix4.log:4)).

**Verdict: GO — submit chain INITIAL at `fb4fc74`.**

Local `HEAD` and `origin/check-equivariance-necessity` both resolve to `fb4fc7497c3e024bfa35b7c02744a32d9a1a0d48`.