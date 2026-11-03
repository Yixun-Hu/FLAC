**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-03 · **Round:** code rB-F3 (exp_22 R2-B fixes 3: anchor content, re-verifier completeness, ledger row binding, merged ledger, kill window; 383a1ab5 → 3fa55fe2)

**Verdict:** REQUEST-CHANGES

The production fixes resolve the rB-F2 correctness findings. I found no remaining BLOCKING production defect. The request is for one evidence-closure issue: the claimed exhaustive 70/70 mutation proof does not mutate several distinct new refusal predicates.

## Findings, ranked

### HIGH 1 — The delete-the-check proof is not exhaustive

The logged mutations all being caught is valid, but “70/70 caught” proves only those 70 selected mutations. Several new checks have no corresponding deletion mutation:

- G3D’s `argmax_equal` and `at_risk` Boolean requirements at [meshgrid_fa_parity.py:769](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:769). M11–M13 cover numeric deltas and rejecting a Boolean as a number, not these two flags.
- Merged segments returning to an earlier shard—the non-contiguous `A, B, A` case—at [meshgrid_engine.py:1123](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1123).
- A shard’s local indices not being exactly `0..k−1` in order at [meshgrid_engine.py:1127](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1127). M25 only deletes the origin/type predicate.
- Separate stamp-form predicates for real indices, `n_queries >= 1`, and string `counts_sha256` at [meshgrid_engine.py:1002](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1002). M38 removes only the dictionary/exact-field portion.
- A stamp naming a nonexistent segment at [meshgrid_engine.py:1011](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1011). M41 deletes only the launch-record mismatch half.
- A stamp claiming a state beyond the segment at [meshgrid_engine.py:1016](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1016). M40 deletes only the duplicate-state half.

The mutation driver confirms those narrower edits in its embedded source at [the mutation log:1](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-11-03_16:59:45_r2bf3_delete_the_check_mutations.log:1).

Concrete fix: add targeted negative tests where absent, delete each predicate independently, and record a new all-caught mutation run. No production-code change is otherwise required unless one of those mutations survives.

## Per-item review

### C1 — RESOLVED

The anchor now authenticates content, not merely self-consistent claims.

- Registered content digests are defined beside the byte pin at [meshgrid_fa_parity.py:78](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:78).
- `anchor_reasons` joins evidence to both registrations at [meshgrid_fa_parity.py:360](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:360).
- Loading independently verifies the parsed content at [meshgrid_fa_parity.py:921](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:921).
- G0 re-verification authenticates its published frozen map at [meshgrid_fa_parity.py:1307](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1307).

I independently hashed the committed JSON bytes and recomputed the two canonical JSON digests:

- File: `8c5fd5e8…`
- Ordered IDs: `4f5743a8c7dcead5486281560a60e709ff93e44e315acf16722962790ab82382`
- Frozen map: `9ecef47d94165e375c87c98597815a8b0c1ef3b8e83fe75112519ab20f31011c`

All match the registered constants. An arbitrary ID list carrying recomputed claims and a coordinated G0 frozen/published-map replacement were both refused.

The synthetic-anchor fixtures temporarily re-register content, but the autouse fixture restores the committed values, and the real committed-content tests independently derive the actual file. I found no test that passes only because a replacement constant leaks across tests.

### C2 — RESOLVED

`reverify_gate_evidence` now turns malformed evidence into named reasons rather than exceptions at [meshgrid_fa_parity.py:1424](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1424).

My malformed probes included non-dictionary evidence, non-string gates, strings/numbers in place of `verdict_clauses`, malformed `inputs`, and malformed G0/G1/G2/G3/G3D/GV structures. All returned reason lists; none raised.

Also verified:

- Exact autocast token/enabled/dtype/device matching at [meshgrid_fa_parity.py:427](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:427).
- Finite, non-negative, non-Boolean G3D measurements at [meshgrid_fa_parity.py:763](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:763).
- Exact registered DiT input set at [meshgrid_fa_parity.py:284](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:284).
- Shared canonical counts validation at [meshgrid_engine.py:889](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:889).
- Per-gate registered schemas match what the live producers write.

Rejecting a zero-valued count is not over-strict: the recorder’s first observation produces count one. An empty invocation is represented by an empty count list, not a zero-count entry.

G1’s executed partition being descriptive and G0/G2/GV publishing autocast without independently judging it remain disclosed, pre-registered semantics, not defects in this round.

### C3 — RESOLVED

The missing-ledger, truncation, index-type, and row-binding findings are fixed.

- Missing ledger is admitted only in a fresh directory at [meshgrid_engine.py:1154](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1154) and [meshgrid_engine.py:1205](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1205).
- Boolean segment indices are refused.
- The driver opens and durably publishes the BF ledger before the binding at [localize_meshgrid.py:434](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:434).
- The reader verifies row stamps and the exact/one-ahead state at [meshgrid_engine.py:985](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:985).

The sidecar stamp is equivalent to the requested in-row stamp for the canonical integrity bar. Defeating either requires rewriting the ledger plus one other artifact: the sidecar here, or the row and its recomputed public digest for an in-row design. Orphan stamps are ignored without a valid row and overwritten on regeneration. A missing or malformed stamp for a valid row is refused.

Every R2 scored BF driver invocation carries a launch record; the merge also re-verifies that provenance before using `(launch_record_sha256, shard_segment_index)`. Thus the merged lookup does not reject a legitimate canonical shard. The protected row schema need not change.

The merge-exact interpretation is sound:

- The shard’s last invocation must be exact.
- An earlier one-ahead crash segment is admitted only when every lost call shape was executed again later.
- Requiring every historical segment to be exact would make the registered resume path impossible.

The lost attempt’s exact call multiplicities and intermediate stamped states are not cryptographically bound. That is an accepted residual: they are execution telemetry, while every partition shape is re-executed and every published score row binds to an admitted ledger state.

### C4 — RESOLVED

The merged ledger is now in one global index space while retaining shard origin and local index at [meshgrid_engine.py:1296](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1296).

My synthetic two-shard/one-segment-each probe produced global indices `[0, 1]`, local indices `[0, 0]`, and no `_merged_reasons`.

The receipt is exactly re-derived from the ledger—including executed partition, violations, count digest, and per-shard blocks—at [meshgrid_engine.py:1110](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1110). The merge verifies the constructed ledger before its first output write at [meshgrid_engine.py:3315](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:3315).

`evaluate_run` now performs the required exact reader/receipt join at [meshgrid_report.py:2438](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:2438).

Vanilla behavior is correct: no ledger, summary partition, or non-null receipt is admitted. The legacy P1 absent receipt and YAW null receipt remain valid.

### C5 — RESOLVED

The pending sentinel closes the check-to-marker hard-kill window.

For each query:

| Kill window | Durable result |
|---|---|
| Before sentinel | Only in-memory work moved; no published evidence |
| Sentinel through check/end | Directory is refused |
| Failed check before/while failure marker writes | Sentinel still makes the directory inadmissible |
| End to ledger write | Previous good ledger remains; query regenerates |
| Ledger to stamp | Exactly one query ahead; recoverable, but not merge-final if it remains last |
| Stamp to sims | Orphan stamp has no valid row and is ignored/overwritten |
| Sims to row | Incomplete query regenerates; earlier one-ahead segment must satisfy merge reconciliation |
| After atomic row | Ledger, stamp, and row are mutually bound |

For the final check, a kill after the final ledger persistence but before the sentinel leaves only resumable prefix evidence; exact report/merge admission still refuses an incomplete partition. From sentinel creation until its removal the directory is inadmissible. After removal, the exact check has passed.

I found no kill window that leaves incorrect evidence admissible as a final canonical result. The cost is one small atomic JSON write and one unlink per scored query on the local output directory.

### C6 — PARTIAL

The 70 recorded mutations were all caught, including the 22 added re-verifier mutations. However, the proof is not exhaustive for the reasons in HIGH 1.

This is the only remaining request-change item.

### C7 — RESOLVED

The vanilla execution path remains frozen:

- Ledger opening is conditional on `fa_partition` being a dictionary.
- Vanilla still writes the binding in its old order and passes `fa_ledger=None`.
- The row builder and P1/YAW row schema are unchanged.
- No protected model/data/configuration module changed.

The P1 golden test at [test_loc_meshgrid_report.py:2302](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2302) and the three frozen-vanilla tests at [test_loc_meshgrid_fa_split.py:89](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:89), line 96, and line 1314 were not edited. The supplied Coder/Planner runs report them passing.

### C8 — RESOLVED

Independent AST count, excluding blank lines, comments, docstrings, and tests:

| Commit | Added | Removed | Changed |
|---|---:|---:|---:|
| 66024ecd | 34 | 5 | 39 |
| 45f492e9 | 118 | 66 | 184 |
| ea6b0fa7 | 53 | 2 | 55 |
| a56d4a59 | 59 | 20 | 79 |
| c0fbadf4 | 56 | 14 | 70 |
| 3b410489 | 130 | 8 | 138 |
| 7375693c | 26 | 1 | 27 |
| 701926e5 | 0 | 0 | 0 |

Total: `+468/−108`, net `+360`. Commit `45f492e9` is 184 changed code lines under the project’s established AST convention, so it is within the “generally <200” guideline; its 232-line raw count was disclosed.

Also confirmed:

- Code/test scope: seven files, `+1708/−155`.
- `git diff --check` clean.
- No protected module or environment/package file changed.
- Every range commit carries the Opus trailer.
- RED results and subsequent GREEN results are recorded per cycle.
- Code ends at `701926e5`, mutation evidence at `d5a4a3c3`, records at `3fa55fe2`.
- The 17:17:46 command list has exactly one executable change: the pre-merge loop additionally names `partition_check_pending.json`. Other changes are prose.
- The changes conform to plan §3 R2-C and §9 A1–A6.

### C9 — PARTIAL

Readiness is blocked only by C6:

- **Push:** add the omitted mutation cases, record the all-caught run, update records, and obtain SHIP.
- **Final integrative review/merge:** push the reviewed tip, then perform the requested whole-branch R2-A through rB-F3 review before merging into `localization-exp`.
- **GV/G1/G2/G3/G3D/G0 and BF probe:** after the reviewed integration point is fixed and a card is free, run the registered sequence.
- **YAW/BF passes:** proceed only after the gates/probe pass and the remaining registered operational conditions, including the BF makespan condition, are satisfied.

The fresh-directory rule, sidecar stamps, pending sentinel, and merge one-ahead reconciliation are validation/recovery mechanisms. They do not change the estimand, thresholds, gate semantics, query set, schedule, or revealed evidence. D-R2-6 already permits recorded resume. No additional pre-registration is required.

## What I verified

I performed only read-only static work:

- Confirmed the requested diff scope and stat, commit range, trailers, worktree state, and diff hygiene.
- Independently rederived the anchor file and content digests.
- Ran pure-function malformed-evidence, anchor-forgery, autocast, measurement, DiT-input, and count probes with CUDA hidden.
- Constructed in-memory merged-ledger and one-ahead reconciliation probes.
- Audited every ledger/stamp/sentinel write and read path and the report-side exact join.
- Compared the command lists and independently recounted AST code lines per commit.
- Audited the embedded source of all 70 mutation cases.

I did not run pytest, drivers, gates, training/evaluation, GPU commands, package operations, or any write-producing command.
