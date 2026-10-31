**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** plan r4

**Verdict:** REVISE — 2 HIGH findings remain.

## Resolution table

| Round-3 finding | Status | Section | Assessment |
|---|---|---|---|
| H1 — identity claim | **PARTIALLY RESOLVED** | §2.1 C-3/C-4; §6 test 4 | C-4 now correctly disclaims input identity, and the stabilized test only requires `StabilizedTieError`. However, C-3 still claims key equality yields the same `context_poses` and `context_poses_vit` tensors. That remains false for independently valued `*_vit` fields and for rows treated as degenerate at `0 < r < 1e-6`. |
| H2 — atomic completion | **PARTIALLY RESOLVED** | §5; §6 test 21 | §5 has the correct order: stage → verify → write `COMPLETE` in staging → rename. Test 21 still specifies `temporary-directory → verify → rename → COMPLETE`, recreating the crash window. Recovery also covers only unmarked final directories, not marker-bearing directories that fail collector verification. |
| H3 — P1 lineage | **RESOLVED** | §7 rung 4; training-path audit rev. 2 | The audit enumerates all legs, identifies `1b56e7a` as the leg producing step 40,000, and separates the post-40k extension. Git confirms the relevant training path did not change. |
| Nit — B-F wording | **PARTIALLY RESOLVED** | §1; §5; §9; comparator ledger | §1 and the ledger correctly identify the rotated evidence as method-level A-F evidence. §5 still says B-F’s “fixed-angle cells are historical,” and §9 says “B-F fixed-angle cells historical,” again implying B-F@40k rotated measurements exist. |
| Nit — Tier-2 alarm | **RESOLVED** | §1 Tier 2 | Exceeding 0.1 explicitly suspends an unqualified invariance conclusion pending item-level triage; it remains separate from the p99 PASS predicate, and relL∞ is descriptive. |

## Remaining findings

1. **HIGH — Make the atomic-completion specification consistent.** In §6 test 21, change the sequence to `staging → verify → COMPLETE → atomic rename`. Test crashes immediately before and after rename. Quarantine every final directory that fails the complete-and-verified predicate, including marker-bearing but collector-invalid directories.

2. **HIGH — Remove the remaining identity assertion in C-3.** State only that preserving row order preserves pose/audio association. Do not infer numerical equality of candidate outputs from the context key.

3. **NIT — Replace the remaining B-F fixed-angle wording** in §§5 and 9 with “historical θ=0 B-F row plus method-level A-F evidence.”

4. **NIT — Cell count remains inconsistent:** §5 says 230, while §11 A3 still says 232.

5. **NIT — Update §1’s shorthand** “P1 trained at `e50d098`” to note initial launch at `e50d098` and the step-40k resumed leg at `1b56e7a`.

## Verified as correct

- `git log e50d098..1b56e7a -- train.py defaults.ini src/ data/AR` returned empty.
- `git show --stat 1b56e7a` touched only `worklog/experiment_SOP.md`, `worklog/worklog_yixun/HANDOFF.md`, and `p1_ddp_launch.sh`.
- The revised lineage audit correctly identifies the step-40k producing leg.
- The fail-closed C-4 policy, zero-survivor pre-FULL gate, Tier-2 alarm consequence, and comparator ledger’s A-F/B-F distinction are otherwise correct.
- I made no file, environment, or package changes and ran no tests, training, evaluation, or GPU commands.