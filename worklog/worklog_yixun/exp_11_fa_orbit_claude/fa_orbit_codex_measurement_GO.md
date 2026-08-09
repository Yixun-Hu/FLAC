# FINAL GO — exp_11 measurement campaign (c27af15)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-07

# Final confirmation review — exp_11 measurement GO (`c27af15`)

**Decision: GO for both scopes under C1–C3 plus a continuously engaged campaign freeze. No blocking findings remain.**

Commit `c27af1581cbcda713e3c6e89f1701c03f0633eeb` is the pushed branch HEAD, and the reviewed lifecycle scripts match that commit.

## Freeze and deletion-path audit

| Surface | Disposition while frozen | Evidence |
|---|---|---|
| Freeze serialization | **Closed.** Store lock acquisition precedes all command dispatch; `--freeze`, `--thaw`, and `--frozen` therefore execute under the same lock as create, lease, release, and prune. | [fa_orbit_measure_worktree.sh:255](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:255), [fa_orbit_measure_worktree.sh:270](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:270), [fa_orbit_measure_worktree.sh:284](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:284), [fa_orbit_measure_worktree.sh:286](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:286) |
| Explicit `--prune` and ordinary preparation sweep | **Inert for worktrees.** `prune_all` may inspect entries and reap stale lease files, but every prunable worktree is retained under freeze. The ordinary preparation call reaches the same guarded function. | [fa_orbit_measure_worktree.sh:207](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:207), [fa_orbit_measure_worktree.sh:223](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:223), [fa_orbit_measure_worktree.sh:309](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:309), [fa_orbit_measure_worktree.sh:383](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:383) |
| `remove_entry` | **Fail-closed.** Its freeze guard precedes asset removal, `git worktree remove`, and root `rm -rf`; callers cannot bypass the campaign freeze through this deleting function. | [fa_orbit_measure_worktree.sh:179](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:179), [fa_orbit_measure_worktree.sh:183](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:183), [fa_orbit_measure_worktree.sh:189](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:189), [fa_orbit_measure_worktree.sh:202](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:202) |
| `git worktree prune` | **Suppressed.** The normal sweep refuses the command under freeze. The second invocation in stale-target preparation is unreachable because that path aborts first. | [fa_orbit_measure_worktree.sh:234](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:234), [fa_orbit_measure_worktree.sh:236](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:236), [fa_orbit_measure_worktree.sh:331](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:331), [fa_orbit_measure_worktree.sh:339](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:339) |
| Implicit half-removed-target cleanup | **Aborts without mutation.** A definitively invalid target under freeze exits with instructions to thaw, clean manually, and re-freeze; lease checks, metadata pruning, and `remove_entry` are not reached. | [fa_orbit_measure_worktree.sh:324](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:324), [fa_orbit_measure_worktree.sh:330](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:330) |
| Submission without freeze | **Refused before preparation or `sbatch`.** The submitter first establishes the outer store-lock span, then requires the marker before preparing a worktree. | [fa_orbit_screen_submit.sh:30](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:30), [fa_orbit_screen_submit.sh:52](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:52), [fa_orbit_screen_submit.sh:71](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:71), [fa_orbit_screen_submit.sh:79](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:79) |

## Lease-file carve-out

**Accepted.** Freeze deliberately permits only bookkeeping-file removal:

- Normal release unlinks exactly `.leases/<jobid>` at [fa_orbit_measure_worktree.sh:123](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:123).
- Reaping unlinks exactly a stale lease file after Slurm successfully proves the job absent, or explicitly reports an invalid job ID, at [fa_orbit_measure_worktree.sh:125](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:125) and [fa_orbit_measure_worktree.sh:156](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:156).

Neither operation removes code, assets, Git administration, or the worktree directory. Even after a lease file is reaped, the freeze prevents that state from cascading into worktree deletion. The carve-out therefore preserves the promised invariant.

## Evidence

The neutralized-freeze RED run produced the expected four failures: explicit pruning removed its candidate, `git worktree prune` ran, stale-target preparation cleared and recreated the target, and an unfreezed submission proceeded with `rc=0` ([RED log:87](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_04-07-53_screen_guardtests.log:87), [RED log:100](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_04-07-53_screen_guardtests.log:100)). This demonstrates guard sensitivity.

The committed green run covers all freeze behaviors and reports **94 passed, 0 failed** ([green log:86](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_04-09-40_screen_guardtests.log:86), [green log:114](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_04-09-40_screen_guardtests.log:114)); the accompanying Python suite reports **192 passed** ([pytest log:17](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_04-10-55_pytest_condgo.log:17)). The supplied live dry-run is green; its durable post-state is also present: the campaign marker is engaged and a valid detached worktree for full SHA `c27af1581cbcda713e3c6e89f1701c03f0633eeb` exists.

## Guard-suite caveat

The guard suite directly unlinks the marker for one lock-isolation test at [fa_orbit_screen_guardtests.sh:627](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:627) and deliberately leaves the store thawed at [fa_orbit_screen_guardtests.sh:749](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:749).

That is not a production-path defect, but **“guards not run mid-campaign” is a necessary operating condition**, not an informational note. Under that condition it is non-blocking.

## Verdict

| Scope | Verdict under C1–C3 plus engaged freeze |
|---|---|
| **(a) Backfill re-runs** | **GO** — restricted to `CELL=screen`, seed 42, K=8, steps 20000/30000. |
| **(b) Arm screens, including `CELL=conf`** | **GO** — `screen` remains seed 42/K=8; `conf` remains seeds 42–46 with K in `{1,8}`. |

Commit `c27af15` satisfies the prior conditional-GO criterion: all previously reachable worktree-deletion paths are inert or abort while frozen. The remaining lifecycle outcomes are unreachable under the conditions below or fail-closed availability/storage risks.

## Final operating conditions — verbatim

> C1. SINGLE SERIALIZED SUBMITTER: exactly one fa_orbit_screen_submit.sh invocation at a time, sequential, from one operator shell (the Planner's own session; no parallel submissions ever).
>
> C2. NO AUTO-PRUNE: --prune is never invoked during the measurement campaign; worktrees accumulate (disk cost ~200 MB per SHA, accepted); any cleanup is manual, only when squeue shows zero exp11 measurement jobs, and logged.
>
> C3. All submissions via the locked submitter from a pushed HEAD, as already required.
>
> C4. ENGAGED CAMPAIGN FREEZE: before the first submission, engage the freeze through fa_orbit_measure_worktree.sh --freeze and keep .measure_worktrees/.campaign_freeze continuously present until squeue shows zero exp11 measurement jobs. Do not invoke --thaw, directly unlink the marker, run fa_orbit_screen_guardtests.sh, or perform cleanup mid-campaign. Only after zero exp11 measurement jobs is confirmed may the store be thawed and manual cleanup performed and logged.
