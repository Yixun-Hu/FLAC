# GO re-check 5 — exp_11 measurement execution (2e189fe)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access (bwrap unavailable); read-only · **Date:** 2026-08-07

# GO/NO-GO re-check 5 — exp_11 measurement execution (`2e189fe`)

**Overall verdict: NO-GO for new measurement submissions.**

`2e189fe306fbee1a952450bdf51561baf2340f6b` matches local `HEAD` and `origin/check-equivariance-necessity`; reviewed paths have no post-commit changes. Already-running arm training is unaffected.

## Per-item disposition

| Item | Status | Disposition |
|---|---|---|
| Outer-entry check precedes preparation | **CLOSED** | The submitter checks fd 8 before the first transaction step ([submitter:42](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:42), [submitter:71](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:71)). |
| Forged marker with missing/wrong/unresolvable fd 8 | **CLOSED as tested** | All three paths are refused; GREEN evidence is recorded at [guard log:72](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_03-45-10_screen_guardtests.log:72). |
| Non-empty readlink equality in submitter and helper | **CLOSED** | Both predicates require non-empty resolved paths before equality ([submitter:45](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:45), [helper:228](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:228)). |
| RED evidence for absent outer check | **CORROBORATED, workspace-only** | The local RED log records all three forged cases accepted with `rc=0` ([RED log:80](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_03-42-23_screen_guardtests.log:80)); that RED log is untracked, not pushed. |
| Validate before release | **CLOSED** | Lease validation occurs at lines 112–117, before release at line 119; the runtime mock confirms the lease was valid at release ([green log:71](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-07_03-45-10_screen_guardtests.log:71)). |
| Retain lease after ambiguous release/scancel | **CLOSED** | Neither failed release nor failed cancellation removes the lease ([submitter:119](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:119)); both cases pass in the guard log. |
| Remove unlocked EXIT unlink | **CLOSED** | EXIT release runs only through the locked helper; failure leaves the lease for reaping ([screen:192](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:192)). |
| Registered-worktree removal failure never escalates to root `rm -rf` | **NOT CLOSED** | The normal valid-worktree case is protected, but the claimed invariant is conditional on a fallible identity classification and is not global. |

## Fresh blocking findings

### 1. fd-path identity is not proof that the flock is held

Both predicates verify that fd 8 resolves to `.store.lock`, but neither verifies or acquires the advisory lock. This sequence passes the outer gate while remaining unlocked:

```text
exec 8<.measure_worktrees/.store.lock   # correct file, no flock
export FA_ORBIT_STORE_LOCK_HELD=1
bash fa_orbit_screen_submit.sh ...
```

A read-only probe with that state reached argument parsing instead of the “only CLAIMS” refusal:

```text
rc=2
unknown argument 'NOT_A_VALID_ARGUMENT'
```

The nested helper makes the same path-only decision and therefore also skips `flock`. If another process owns the real lock, the forged invocation still proceeds concurrently. The preparation → held submission → lease transaction is consequently not proven locked.

Both predicates need a kernel-level check/acquisition such as `flock -n 8` after path validation, with guards for a correct-path-but-unlocked fd 8.

### 2. Registered-but-damaged worktrees can still reach root `rm -rf`

`remove_entry()` protects a failed Git removal only when its earlier `worktree_identity()` result is exactly `valid` ([helper:164](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:164), [helper:168](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:168)). A registered worktree with a missing/damaged `.git`, or a failed unchecked `readlink -f`, can be classified `invalid`; if `git worktree remove` then fails, execution falls through to root `rm -rf` at line 180.

Thus the green guard proves only “valid worktree + mocked Git failure.” It does not prove “registered worktree removal failure never escalates.” Registration must be established independently from `git worktree list --porcelain`, with failure treated as indeterminate, or every failed Git removal must leave the root intact.

## Verification evidence

- Shell syntax and `git show --check` pass.
- Committed evidence reports **85/85 guards** and **192 pytest**.
- The live dry-run is green as supplied, but does not exercise either fresh concurrency case.
- The generator and table were untouched by `2e189fe`; the established fail-closed table exit-4 path remains intact.
- No Slurm job was submitted and no project file was changed by this review.

## Scope verdict

| Scope from pushed `2e189fe` | Verdict | Operating condition |
|---|---|---|
| **(a) Backfill re-runs** | **NO-GO** | Do not resubmit the 20k/30k backfills until both blocking findings are fixed, pushed, and guard-tested. |
| **(b) Arm screens, including `CELL=conf`** | **NO-GO** | The same submitter and worktree store are shared by futility and confirmatory cells. |
| **Already-running arm training** | **CONTINUE** | Outside this measurement-submission gate. |

After remediation, invoke the submitter by absolute path or from the main repository root, from a sanitized production environment with `FA_ORBIT_STORE_LOCK_HELD`, Slurm mock overrides, `DRYRUN`, and non-production `OUTPUT_ROOT` unset. Backfills remain restricted to `CELL=screen`, seed 42, K=8, steps 20000/30000; arm confirmatory cells remain seeds 42–46 with K in `{1,8}`.
