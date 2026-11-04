**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-03 · **Round:** code rB-F4 (exp_22 R2-B tests-and-proof round: the exhaustive delete-the-check proof; dec16c55 → 36ba90f4)

**Verdict:** REQUEST-CHANGES

No production-code defect was found, and the six gaps named in rB-F3 are closed. The request changes are confined to the claimed exhaustiveness and isolation proof.

## Findings, ranked

### HIGH 1 — The 244-row inventory is not exhaustive under its own scope rule

The declared rule includes every predicate in every rB-F3-created function, but these atoms have no mutation row and are not among the six justified exclusions:

- [`pair_reasons`: `tolerance is None`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:201), the gate-versus-measurement dispatch.
- [`_shape_reasons`: `field in payload`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1402), the guard that lets a missing field produce its named reason instead of raising.
- [`_row_binding_reasons`: `blocks is not None`, the fallback blocks, and `if merged_mode`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:994), plus the separate merged/shard lookup selection at [line 1009](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1009).
- [`_merged_reasons`: `blocks and blocks[-1][0] == shard`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1121), including both the nonempty guard and same-shard comparison.
- [`fresh_out_dir`: the final `not found` verdict](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1170).
- [`read_fa_ledger`: merge-report existence selects merged mode](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1196).

The three `continue` statements at engine lines 1008, 1014, and 1019 are also incorrectly dismissed as inert control flow. Removing the first two can turn a named refusal into a `TypeError`/`AttributeError` before the accumulated reasons are returned.

Concrete fix: add focused tests and independent mutations for these atoms, including both positive guards and negative admission paths where applicable, then rerun the complete proof.

### HIGH 2 — At least two `P` classifications have hidden admission paths

[M133 and M134](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-11-03_19:08:20_r2bf4_delete_the_check_mutations.log:1344) are not purely permissive:

- M133 deletes `created_utc:str`. An evidence object that also omits `created_utc` then passes the mutated schema; the field is not checked elsewhere. The logged positive failures occur because unchanged honest evidence now carries an “extra” field.
- M134 changes `device:str` to `device:int`. A G2, G0, or GV evidence with `inputs.device = 7` passes the mutated schema; those gates return early from `autocast_reasons`, and their comparisons do not otherwise validate that field.

The supposedly exhaustive schema test also derives its field/kind oracle from `fp.EVIDENCE_SCHEMA` itself at [test line 4404](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4404). When that registration is mutated, the test’s oracle moves with it.

Concrete fix: freeze an independent expected `(gate, field, kind)` manifest in the tests, reclassify admission-capable entries as N/C, mutate every schema registration atom, and require the corresponding missing/retyped-field test to fail. Re-audit all 20 P rows; M198’s honest-input failure alone likewise does not prove that no forged merged directory can satisfy the shard-mode fallback.

### HIGH 3 — The driver’s `CAUGHT` rule does not enforce the isolation standard

The rule is only:

`returncode != 0 and "failed" in summary and bool(failed)`

at [proof-log line 1172](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-11-03_19:08:20_r2bf4_delete_the_check_mutations.log:1172). Any failing test selected by a broad expression makes the mutation “caught,” even if the atom’s own test passes. Only three failed IDs are retained.

M244 demonstrates the evidence problem: eleven tests fail, but the log shows three older merged-corruption tests and hides the new single-shard exactness test in “+8 more.” Static tracing indicates the target test should fail, but the recorded artifact does not establish that.

Concrete fix: each mutation must carry required failed test IDs (or exact parameterized-ID patterns); `CAUGHT` should require `returncode == 1` and `required_ids ⊆ failed_ids`, reject collection/errors, and log every required ID. Prefer selectors that select only the atom’s test family.

## C1 — RESOLVED

All six named rB-F3 gaps are now directly represented:

- (a) M079/M080 delete `argmax_equal` and `at_risk` separately. The test asserts the exact boolean reason both directly and through re-verification at [lines 4234–4249](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4234).
- (b) M228 deletes only the A,B,A return check; the direct and real-reader test asserts the contiguous-block reason at [line 4940](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4940).
- (c) M229 deletes only the local-index sequence comparison; gap, duplicate, and reorder all assert its reason at [line 4956](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4956).
- (d) M172/M038/M039/M173–M176 separately remove the stamp object, field set, query binding, real indices, positive query count, and string digest checks. The chosen edits otherwise bind successfully; see [STAMP_FORM_DEFECTS](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4712).
- (e) M177 removes exactly `segment is None`; the shard and merged cases assert “which the ledger does not hold” at [line 4748](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4748).
- (f) M178 removes only the beyond-segment disjunct. Its test asserts that predicate’s reason at [line 4769](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:4769). It cannot be an only-violation end-to-end input because the range check is deliberately redundant; treating it as R is accurate.

The logged failed IDs name these tests.

## C2 — NOT RESOLVED

My independent inventory grouped the diff as follows:

- Parity: measured-number/tolerance, registered anchor content, autocast resolution, DiT inputs, G3/G3D measurement, content loader, counts re-verification, re-hosted production-query/coverage checks, malformed handling, and schema machinery.
- Engine: sentinel, canonical counts, delta subtraction, stamps, row/segment binding, merged origins/receipt, freshness, reader dispatch, adoption, census, pass dispatch, and merge publication.
- Driver/report: binding/ledger claim order and exact report reread.

The table covers most of these through FP/RV, EN, DR, and RP rows, including the pre-existing predicates re-hosted into new functions. It nevertheless misses the atoms listed in HIGH 1 and leaves the EVIDENCE_SCHEMA family inadequately justified.

The scope rule is appropriate and not harmful through over-breadth; its execution is incomplete within that declared scope. A transitive mutation sweep of every old helper is not necessary, provided each new call-site predicate is independently deleted and its behavior isolated.

## C3 — NOT RESOLVED

The driver can report a false catch from another selected failure. The Planner’s 40-row rerun reproduces the rows, but it reproduces the same rule and truncated evidence.

I traced more than ten N rows from edit to assertion:

| Mutation | Own assertion |
|---|---|
| M071 | exact registered-tolerance reason at test line 4147 |
| M079/M080 | exact `K=8: <flag> is not a boolean` |
| M081 | exact ordered-query-content reason |
| M089 | single autocast-clause reason |
| M114 | candidate list “does not tile” |
| M123 | exact full/tail coverage reason |
| M145 | one canonical-counts reason |
| M158 | exact non-real `n_queries` reason |
| M173–M176 | stamp-form reason via `_stamp_needle` |
| M177 | exact nonexistent-segment reason |
| M228 | exact noncontiguous-shard reason |
| M229 | exact local-index-order reason |
| M241 | `ckpt_sha256` refusal plus unchanged snapshot |
| M244 | exact last-invocation/one-ahead report reason |

These spot checks are sound, but they cannot substitute for a machine-enforced requirement across all 244 rows.

## C4 — NOT RESOLVED

- C rows: sound. The two committed digests are re-derived from the committed report, the four autocast registrations are checked against Torch/engine state, and registered DiT inputs are derived from the model configuration.
- D rows: direct behavioral tests are acceptable evidence for state/data atoms.
- R rows: direct-helper or simulated-upstream-defect tests are acceptable for proving that a defense exists, but they are unit evidence rather than end-to-end reachability. The thirteen disclosed R classifications are broadly reasonable; M184 is the clearest removable-but-useful internal defense.
- P rows: not resolved because M133 and M134 have actual admission paths. The claim that every P deletion can only over-refuse or crash is false.

## C5 — PARTIAL

- `_collect(schema, ...)` malformed branch: accepted as unreachable for serialized JSON after the valid-gate/dict entry checks.
- Three `continue`s: refuted. They preserve named-refusal behavior and prevent invalid state from being dereferenced.
- EVIDENCE_SCHEMA entries: refuted. They are independently mutable, and the current oracle is self-referential.
- Merge receipt keys: accepted. One union comparison implements the predicate, while the fixed-key test covers each required key, extra, and missing keys.
- Artifact producer content: accepted as outside the refusal-predicate sweep. Sentinel content is deliberately unread; stamp content is separately pinned and read by the binding checks.
- `fa_ledger` handoff/return tuple: accepted as non-verdict plumbing. Dropping it creates a redundant empty segment rather than admitting false counts, although a structural test would be appropriate if “one segment per invocation” becomes a required invariant.

## C6 — PARTIAL

Confirmed:

- The test diff is exactly `+939/−0`, appended after the previous final test.
- No existing test was changed or weakened.
- The P1 golden and three frozen-vanilla tests are unedited and recorded green by name.
- Fixture re-registration is restored by the autouse fixture; mutations of the actual committed constants use separate committed-content tests.
- Simulated defects are confined to disclosed R/defense-in-depth cases.
- No timing assertions, global ordering dependence, or shared filesystem state was found; temporary paths and explicit RNG seeds make the new tests deterministic.

Not resolved: the schema tests and positive P tests do not establish the claimed independent admission coverage.

## C7 — PARTIAL

Procedural evidence is otherwise clean:

- Production diff against `701926e5` is empty.
- `23a89009..36ba90f4` changes no production or test source.
- Five round commits carry the Opus trailer.
- `git diff --check` is clean.
- Recorded focused run: 1,511 passed, 9 environmental skips.
- Planner CPU rerun: 1,511 passed, 9 skipped; 5,399 collected under `src/tests`.
- Whole-repository record: 5,406 collected, zero errors.
- Planner clone sample: 40/40 rows reproduced with clean tracked state.
- No tracked environment/package changes were present; all recorded execution used `CUDA_VISIBLE_DEVICES=""`.
- The 2026-11-03T17:17:46 authoritative GPU command list is unchanged.

Readiness:

- (a) Do not push this round as reviewed until the missing atoms, P reclassification, and isolation-enforced mutation rerun are committed.
- (b) The final integrative review and merge into `localization-exp` remain blocked by this REQUEST-CHANGES result.
- (c) GV/G1/G2/G3/G3D/G0 plus the BF probe remain blocked by review completion, Yixun’s go, and a free card; use the unchanged 17:17:46 command list.
- (d) YAW/BF passes remain downstream of those gates/probe and released cards.

No new pre-registration is needed: this remains tests-and-proof work with frozen production behavior. If one of the added mutations survives and requires a production change, reassess registration then.

## What I verified

Read-only probes only:

- `git status`, commit log/messages/trailers, `git show`, diff stats/name-status, and `git diff --check`.
- Production-freeze diffs against `701926e5` and source freeze from `23a89009` to `36ba90f4`.
- Full production diff `383a1ab5..701926e5`, current function bodies, appended tests, worklog entries, proof source, coverage table, and final logs.
- Parsed the embedded 1,200-line driver in memory and enumerated all 244 mutation tuples: N 180, P 20, D 24, C 7, R 13.
- Static AST comparison of changed/new decision atoms against the mutation edit strings.
- Read the Planner’s external pytest and 40-row mutation-sample logs.

I did not run pytest, the mutation driver, any driver/gate/probe, GPU command, or process-management command, and modified no file.
