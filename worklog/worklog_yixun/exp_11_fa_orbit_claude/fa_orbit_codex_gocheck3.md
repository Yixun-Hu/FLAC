# GO re-check 3 — exp_11 measurement execution (e650d6a)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access (bwrap unavailable); read-only · **Date:** 2026-08-07

# exp_11 final GO/NO-GO re-check — `e650d6a`

**Overall verdict: NO-GO for measurement execution and table publication.**

`e650d6a607d1a093fe2f954fd49555886cd74c51` is both local `HEAD` and pushed branch HEAD. The asset, CLI, and validator-path fixes close their punch-list items, but the lease handoff still has a prune race and an unsafe cancellation-failure path. The disclosed table overwrite also exposes a publication guard that is required now.

## Punch-list disposition

| Item | Status | Evidence and judgment |
|---|---|---|
| **1. Store lock, held-job lease, and failed-`squeue` behavior** | **PARTIALLY CLOSED** | All individual helper commands acquire `.store.lock` before dispatch ([fa_orbit_measure_worktree.sh:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:127)); `squeue` controller/auth failures retain the lease ([fa_orbit_measure_worktree.sh:65](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:65)); and the happy path is `sbatch --hold` → real-ID lease → `scontrol release` ([fa_orbit_screen_submit.sh:39](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:39)). However, the lock ends when worktree preparation returns, before the held job and lease exist. A concurrent `--prune` can therefore remove the unleased tree between lines 35 and 55. `add_lease()` then recreates only `<removed-tree>/.leases` without validating that the target remains a live worktree ([fa_orbit_measure_worktree.sh:56](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:56)), after which the job is released and aborts against a non-worktree. The prior pin-to-lease/prune race is therefore not closed. |
| **2. AGREE SHA-256 and canonical AcousticRooms target** | **CLOSED** | `AcousticRooms` is pinned to `/n/fs/gatrdp/datasets/AcousticRooms`, and the helper refuses a different resolved target ([fa_orbit_measure_worktree.sh:46](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:46), [fa_orbit_measure_worktree.sh:180](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:180)). The screen pins AGREE to `3a13243d…c787`; that equals the current file’s SHA-256 ([fa_orbit_screen.sbatch:211](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:211)). |
| **3. Real `--repo-root` argv plus foreign-cwd subprocess coverage** | **CLOSED** | `main(argv=None)` now passes `None` to argparse, making the real command line reachable ([gen_model_comparison.py:253](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:253)). The new subprocess tests run from a foreign cwd, target a synthetic repository, provide nonempty valid two-K exp_11 evidence, and require numeric—not pending/blocked—rows ([test_gen_model_comparison_gate.py:362](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:362), [test_gen_model_comparison_gate.py:383](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:383)). |
| **4. Validator config resolution against `REPO`** | **CLOSED** | Relative `model_config` paths resolve against `REPO`; relative checkpoints resolve against `OUTPUT_ROOT_BASE` ([exp11_validate_rows.py:375](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:375)). The foreign-cwd test exercises successful recomputation and verifies that a wrong hash remains rejected ([test_exp11_validate_rows.py:591](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_validate_rows.py:591)). |

## Fresh-eyes findings

### Blocking lifecycle failures

1. **Preparation and lease creation are not one locked transaction.**  
   The mock demonstrates the expected happy-path calls, but it never injects a prune between preparation and lease creation ([fa_orbit_screen_guardtests.sh:413](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:413)). The worktree must remain protected continuously until its real job-ID lease is durable, and `--lease` must reject a missing, foreign, or half-removed target.

2. **An ambiguous release/cancel failure can produce a live unleased job.**  
   If `scontrol release` fails and `scancel` also fails, the script nevertheless deletes the lease ([fa_orbit_screen_submit.sh:60](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:60)). A release command may have reached Slurm before its client observed failure; dropping the lease can therefore expose a running job’s tree to pruning. The mock hard-codes successful `scancel`, so it misses this case ([fa_orbit_screen_guardtests.sh:431](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:431)). The lease must be retained unless cancellation is positively confirmed.

3. **The screen’s fallback release bypasses the store lock.**  
   The EXIT trap directly unlinks the lease if the helper fails ([fa_orbit_screen.sbatch:192](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:192)). A stale lease is safe and already reapable; an unlocked removal contradicts the store-wide-lock invariant.

### Bonus half-removed-worktree fix

**Status: PARTIALLY CLOSED.**

The identity predicate correctly requires `.git`, self as top-level, and a matching common directory ([fa_orbit_measure_worktree.sh:153](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:153)), closing the reported false reuse of an empty directory.

However, any identity-check failure is treated as proof of a half-removed tree and triggers unconditional recursive deletion without consulting leases ([fa_orbit_measure_worktree.sh:160](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:160)). A transient Git/NFS failure must not delete a leased tree. Cleanup should require a proven unleased/stale entry, and the deletion-name check should enforce exactly 40 hexadecimal characters rather than merely an eight-hex prefix.

## Table-overwrite incident

**The proposed numbered-row regression guard is required now, not deferred.**

The committed table has a numeric exp_10 K=1 endpoint row and a 4/5 K=8 row ([model_comparison.md:32](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/model_comparison.md:32)). The registered source globs currently find zero files for both cells, yet the generator renders missing evidence as pending and writes directly over the table ([gen_model_comparison.py:206](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:206), [gen_model_comparison.py:339](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:339)). The incident demonstrates this is an active failure mode.

The recovered table is currently byte-identical to `HEAD`, but the pipeline must default-abort before any existing numeric row becomes pending. Evidence-count regressions such as 4/5 → 0/5 should also abort or require an explicit audited override.

## Verdict by execution scope

| Scope from pushed `e650d6a` | Verdict | Reason |
|---|---|---|
| **(a) Backfill re-runs** | **NO-GO** | The preparation-to-lease prune race and ambiguous release/cancel handling remain. |
| **(b) Arm screens, including `CELL=conf`** | **NO-GO** | The same lifecycle defects apply to every submitted seed/K cell. |
| **(c) Table pipeline** | **NO-GO** | CLI and validator resolution are fixed, but the demonstrated numbered-row-to-pending overwrite remains unguarded. |

Fresh project-environment pytest completed successfully; the committed evidence records **186 passing tests**, and the inspected mocked guard log records **68/68 passing cases**. These suites do not cover the blocking interleavings above. No Slurm jobs were submitted. Already-running arm training is outside this NO-GO; the verdict gates measurement submissions and table publication.
