# GO re-check 4 — exp_11 measurement execution (253658f)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access (bwrap unavailable); read-only · **Date:** 2026-08-07

# GO/NO-GO re-check 4 — exp_11 measurement execution (`253658f`)

**Overall verdict: NO-GO for new measurement submissions. GO for the fail-closed table pipeline.**

`253658f3cd2f47684270b8b7a5cd2407a6bdb3a7` matches local `HEAD` and `origin/check-equivariance-necessity`; reviewed paths have no post-commit changes. Already-running arm training is unaffected.

## Per-item disposition

| Item | Status | Judgment |
|---|---|---|
| **1. Lock-spanned submission lifecycle** | **PARTIALLY** | The normal path correctly re-execs through `--with-lock`, inherits fd 8, validates inherited fd 8 inside the helper, and does not explicitly unlock it ([worktree helper:209](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:209), [worktree helper:223](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:223), [worktree helper:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:306)). The supplied RED reproduces the original race—lock free, tree deleted, submit rc=5 ([RED log:73](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_03-23-01_screen_guardtests.log:73))—and the enabled run passes ([green log:73](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_03-24-16_screen_guardtests.log:73)). However, two lifecycle blockers remain below. |
| **2. Three-valued identity, safe deletion, and lease reaping** | **PARTIALLY** | Exact 40-lowercase-hex checking, centralized tree removal, and the opportunistic stale-lease reaping repair are correct ([worktree helper:80](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:80), [worktree helper:137](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:137), [worktree helper:157](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:157)). The half-removed target path requires `invalid`, no live/unverifiable lease, and a 40-hex name ([worktree helper:250](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:250)). But the claimed global deletion invariant is not actually enforced, and canonicalization failures are not fully three-valued. |
| **3. Published-row regression guard** | **CLOSED** | A guarded live-table execution returned **4**, did not write, and named exactly `fa scratch @67.5k (exp_10, pending gates)` at K=1 and K=8. An in-memory `--allow-row-regression` execution returned 0 and rendered both affected rows into the audit paragraph. The real table remained byte-identical. The implementation aborts before the write at lines 378–397 ([generator:378](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:378)); focused regression tests cover numeric loss, count loss, dropped rows, override auditing, and non-regressing growth ([tests:442](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:442)). |

## Fresh-eyes findings on the lock re-exec

### Blocking: the outer submitter trusts the marker without validating fd 8

The helper validates both `FA_ORBIT_STORE_LOCK_HELD=1` and `/proc/self/fd/8`, but the submitter decides whether to enter the wrapper using only the environment marker ([submitter:30](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:30)).

Consequently:

```text
FA_ORBIT_STORE_LOCK_HELD=1, no inherited fd 8
→ submitter skips --with-lock
→ preparation helper briefly locks and exits
→ sbatch window is unlocked
→ lease helper takes a separate lock
```

That recreates the exact preparation-to-lease race the commit is intended to close. The `/proc/self/fd` check is therefore correct but placed too far downstream to protect the transaction. The submitter must validate fd 8 itself or unconditionally enter the trusted wrapper unless a verified inherited lock is present.

### Blocking: failed release still discards the lease without confirmed cancellation

On `scontrol release` failure, the submitter calls `scancel` but removes the lease regardless of whether cancellation succeeds ([submitter:76](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:76)). A release request can have reached Slurm even when the client reports failure; `scancel` can then fail or merely initiate asynchronous cancellation. Dropping the lease exposes a possibly running job’s tree to pruning.

The guard mock makes `scancel` unconditionally succeed and explicitly treats “failed release … drops its lease” as PASS ([guard tests:432](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:432), [guard tests:459](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:459)). The safe behavior is to retain the lease after any ambiguous release path and let the existing `squeue` reaper remove it only after Slurm proves the job absent.

Additionally, the code says “validate the lease, then release” but performs validation after `scontrol release` ([submitter:65](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:65), [submitter:76](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:76), [submitter:83](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:83)).

The prior unlocked EXIT fallback also remains: if helper-based release fails, the job directly unlinks its lease outside the store lock ([screen driver:192](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:192)). A stale lease is safe; this fallback should be removed.

### Secondary invocation defect

`--with-lock` changes cwd to the main repository before executing its command, while the submitter passes `$0` unchanged ([worktree helper:195](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:195), [submitter:34](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:34)). A relative invocation from outside the repository root can therefore fail during re-exec. The absolute script path should be passed.

## Safe-cleanup residuals

`worktree_identity()` checks `git` failures, but it does not check failures from either `readlink -f`; a canonicalization/NFS error can become `invalid` or even compare as two empty strings instead of becoming `indeterminate` ([worktree helper:65](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:65)).

Also, `prune_all()` skips only `indeterminate`; both `valid` and `invalid` entries reach `remove_entry()` when unleased ([worktree helper:173](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:173)). `remove_entry()` then escalates a failed `git worktree remove` into raw `rm -rf` ([worktree helper:160](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:160)). Thus the stated “recursive deletion only after definitively invalid identity” invariant is false. Normal pruning of valid unleased worktrees should be separated from the raw cleanup path; a failed Git removal should leave the entry for inspection.

The reaping regression itself is fixed: `prunable()` now visits every lease and reaps every provably dead neighbor even when another lease is live.

## Scope verdict

| Scope from pushed `253658f` | Verdict | Operating condition |
|---|---|---|
| **(a) Backfill re-runs** | **NO-GO** | Do not submit new backfills until the outer fd-8 validation and ambiguous release/cancel handling are fixed and exercised. |
| **(b) Arm screens, including `CELL=conf`** | **NO-GO** | Same lifecycle defects apply to every screen and confirmatory cell. Already-running arm training may continue. |
| **(c) Table pipeline** | **GO** | Run with explicit `--repo-root /n/fs/gatrdp/codespace/FLAC`. With today’s missing exp_10 evidence, exit 4 without writing is the required successful guard outcome. Publish normally only after the missing evidence returns; use `--allow-row-regression` solely for a deliberate reviewed retraction whose rendered audit is committed. |

Before another measurement GO-check, add guards for a forged marker with missing/wrong fd 8, release failure plus `scancel` failure, lease validation before release, removal of the unlocked EXIT fallback, `readlink -f` failure, and valid-worktree removal failure. The committed evidence reports 75/75 shell guards and 192 pytest cases, but none covers those blocking paths. No Slurm jobs were submitted and no project file was changed during this review.
