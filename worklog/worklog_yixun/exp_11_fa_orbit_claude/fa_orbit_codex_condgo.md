# Conditional-GO ruling — exp_11 measurement under operating conditions (2e189fe)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-07

# Operating-conditions analysis — `2e189fe`

**Decision: NO-GO as written for both scopes.** The remaining defects do not expose a demonstrated “wrong number publishes” path under the proposed serialization and lease discipline; they are availability/store-lifecycle risks. However, verbatim C2 does not actually prevent pruning or deletion at this commit, so the removal-escalation finding remains reachable.

## Reachability

| Finding | Reachable under C1–C3? | Analysis |
|---|---:|---|
| Preparation-to-lease race caused by an unlocked forged fd 8 | **No damaging race under the intended conditions** | The incorrect “fd path implies flock held” state can still be constructed, but C1 excludes another submitter and intended C2 excludes a concurrent pruner. The preparation helper’s own sweep completes before returning and excludes the worktree being prepared. Thus nothing can delete that tree during the preparation → held submission → lease window. |
| Registered-worktree removal failure escalating to root `rm -rf` | **Yes** | C2 bans only an explicit `--prune` invocation. Every ordinary preparation nevertheless calls `prune_all "$WT"` unconditionally at [fa_orbit_measure_worktree.sh:324](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:324). Preparation can also call `remove_entry` directly when the target entry is classified `invalid` at [fa_orbit_measure_worktree.sh:268](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:268). Therefore “`--prune` is never invoked” does not imply “nothing ever deletes,” and worktrees are not guaranteed to accumulate. |
| Ambiguous release/cancel | **Closed; no unsafe path remains** | The lease is validated before release at [fa_orbit_screen_submit.sh:112](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:112), and every release/cancellation ambiguity retains it at [fa_orbit_screen_submit.sh:119](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:119). A possibly released job therefore remains protected; a cancelled or permanently held job leaves only a stale lease. |
| Identity misclassification if deletion truly never occurs | **Fail-closed only** | A false `invalid`/`indeterminate` result prevents preparation or lease creation and causes cancellation/resubmission. It does not select different measurement inputs. The problem is that deletion does occur through the two paths above, so this hypothetical protection is not established by verbatim C2. |

### Exact reachable removal path

A normal serialized submission can execute:

```text
fa_orbit_screen_submit.sh
  → default worktree preparation
  → prune_all "$WT"                         # no --prune argument required
  → another 40-hex entry is unleased
  → worktree_identity classifies a registered-but-damaged entry as invalid
  → remove_entry recomputes a non-valid verdict
  → git worktree remove --force fails
  → rm -rf "$entry"
```

The same escalation can occur for the target SHA during stale-entry preparation:

```text
existing target entry
  → identity = invalid
  → no live/unverifiable lease
  → remove_entry
  → git removal fails
  → root rm -rf
```

C1–C3 do not exclude damaged Git metadata, a canonicalization failure, or a failed Git removal.

## Validity versus availability

| Finding | Classification |
|---|---|
| Forged fd 8 / missing actual flock | **Availability under C1–C3.** With no concurrent deleter, it has no effect. If concurrency were reintroduced, the held-job sequence, lease identity check, startup lease check, and `EXPECT_SHA` gate make deletion/replacement fail closed rather than publish a result from the wrong commit. |
| Removal escalation after identity misclassification | **Availability/store-integrity.** For a properly submitted queued or running job, its live lease prevents pruning. Deleting an inactive registered tree can leave Git administration inconsistent, make a later preparation fail, or make later provenance validation fail closed. No wrong-number publication path is established. |
| Lease creation or identity-check failure | **Availability.** The job remains held and is cancelled or requires manual cancellation/resubmission. |
| Ambiguous release/cancel with retained lease | **Availability.** Outcomes are a correctly leased job, a cancelled job, or a held/stale lease requiring cleanup. |
| EXIT lease-release failure | **Availability/storage.** The lease remains until safely reaped; measurement validity is preserved. |

## Verdict

| Scope | Verdict under verbatim C1–C3 |
|---|---|
| **(a) Backfill re-runs** | **NO-GO** |
| **(b) Arm screens, including `CELL=conf`** | **NO-GO** |

This is a NO-GO for the proposed operating-conditions acceptance, not a demonstrated scientific-validity failure: verbatim C2 does not enforce its stated “no auto-prune/worktrees accumulate” premise.

If the implementation or operating mechanism actually prevented **all** implicit pruning and stale-entry deletion during the campaign—not merely explicit `--prune` calls—the remaining lifecycle findings would be unreachable or fail-closed availability risks, and both scopes would be **GO** under C1–C3.

The proposed conditions, reproduced verbatim, are:

> C1. SINGLE SERIALIZED SUBMITTER: exactly one fa_orbit_screen_submit.sh invocation at a time, sequential, from one operator shell (the Planner's own session; no parallel submissions ever).
>
> C2. NO AUTO-PRUNE: --prune is never invoked during the measurement campaign; worktrees accumulate (disk cost ~200 MB per SHA, accepted); any cleanup is manual, only when squeue shows zero exp11 measurement jobs, and logged.
>
> C3. All submissions via the locked submitter from a pushed HEAD, as already required.
