**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-04 · **Round:** code rB-F5 (exp_22 R2-B tests-and-proof round: the mechanically exhaustive, isolation-enforcing delete-the-check proof; 36643835 → 612f4bb7)

**Verdict:** REQUEST-CHANGES

Production remains frozen and I found no production-code defect. The previously named HIGH-1 atoms, schema coverage, and CAUGHT isolation requirements are addressed. However, the atom enumerator still omits material state/effect nodes from its own whole-function scope, so the 391-atom/0-unmapped inventory is not mechanically exhaustive.

## Findings

### HIGH 1 — The enumerator omits verdict-producing state/effect atoms

**Blocks SHIP:** Yes.

The enumerator’s statement handling in [atoms_rbf5.py:145](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf5/atoms_rbf5.py:145) recognizes expression calls, selected mutator calls, `+=`, and assignments whose values are reason literals. It does not generally enumerate assignments or calls nested under `return`/assignment, even though its declared scope includes “state a verdict is computed from.”

At least these three in-scope atoms are absent from the 391-row inventory:

- [`bucket[key] = rest` at meshgrid_engine.py:942](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:942). Deleting this update changes `_counts_minus` residual state and therefore later binding/count verdicts. The inventory covers the surrounding `if rest` and `bucket.pop(...)`, but no mutation independently deletes this assignment.
- [`return write_json(...)` at meshgrid_engine.py:970](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:970). This is the actual stamp-publication effect of the newly created `write_fa_stamp`. The inventory moves from the directory-creation call to `read_fa_stamp`; the publication call has no atom of its own. M042 deletes a caller, not this function-body effect.
- [`top[id(segment)] = (...)` at meshgrid_engine.py:1022](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1022). This updates the highest-state tracker consumed by `_row_binding_reasons`. The inventory covers its guard and later reason update, but not deletion of the tracker update itself.

The containment/independence mapping and M290 ambiguity guard are conservative once an atom exists. The defect is the upstream atom set.

Concrete fix:

1. Extend enumeration to verdict-relevant `Assign`/`AnnAssign`/subscript writes and effects nested in `Return` or assignment expressions across all 26 scoped functions.
2. Re-enumerate the complete scope rather than adding only these three examples.
3. Give every newly discovered atom an independent mutation and an exact required observable.
4. Rerun the complete proof and update the inventory/history and readiness records.

### NIT 1 — Planner comparison helpers do not enforce their advertised sample sets

**Blocks SHIP:** No.

[rbf5_compare_sample.py:72](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf5_compare_sample.py:72) compares every parsed sample row but does not require the advertised 75 IDs, reject missing/extra IDs, or enforce the count independently. [rbf4_compare_sample.py:4](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf4_compare_sample.py:4) similarly lacks an expected-ID/count assertion and would overwrite duplicate mutation IDs while parsing.

The actual rB-F5 Planner log does contain 75 distinct rows and `TOTAL 75/75`, so this did not invalidate the reported rerun.

Fix: supply or hard-code the expected ID set, reject duplicates/missing/extras, and assert the expected cardinality before comparing fields.

## C1–C7

### C1 — PARTIAL

Every atom specifically named in rB-F4 now has an independent mapping:

- A0198 → M245/M246; A0378 → M247.
- A0043–A0046 → M272–M277; A0058 → M278/M279.
- The three continues → M280/M281/M282.
- A0101–A0103 → M288/M289/M357; ambiguous M290 is correctly mapped to neither.
- A0125/A0126 → M294–M297.

Their required tests exercise the intended reason/effect. M282 is correctly D: deleting the continue binds a refused state and introduces an additional binding reason without raising.

A0160/A0161 are equivalent within reachable contracts: the operands are internal ordinary dictionaries/`None`, so the two `or {}` deletions cannot create an admission path. Exotic custom falsy mappings are outside this path.

Nevertheless, the full-scope inventory remains incomplete due to the omitted state/effect nodes above.

### C2 — RESOLVED

The literal [`FROZEN_SCHEMA_MANIFEST` at test_loc_meshgrid_fa_split.py:5077](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:5077) is independent of `fp.EVIDENCE_SCHEMA`. My read-only reconstruction found 211 literal, unique entries and exact ordered equality with the frozen registration.

The registration extractor faithfully executes the `_schema` definition and schema-registration statements relevant to these mutants. Inserted per-entry mutations operate after registration and change exactly one entry.

The tests at [line 5251](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:5251) and [line 5270](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:5270) check each entry’s own missing/retyped reason using real gate evidence. The `_is_kind` truth table independently covers its five predicates.

The reclassifications are sound:

- M133: N.
- M134: N.
- M135: R.
- M198: N.

M198’s test at [line 5590](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:5590) uses a valid single FA shard carrying `merge_report.json`, first verifies shard-mode reading, then demonstrates that the frozen merged dispatcher rejects the origin. This closes the fallback path.

The remaining 49 P rows are genuine over-refusal/crash mutations; I found no deletion among them that admits forged or wrong evidence.

### C3 — RESOLVED

The implemented rule enforces:

- `returncode == 1`;
- no JUnit `<error>` cases;
- zero collection errors and no error summary;
- exactly as many collected cases as required IDs;
- every required node ID in the failed set.

Exact node selection preserves parametrized IDs. Passed, skipped, or xfailed required cases cannot satisfy the failed-set condition; setup/teardown and collection errors are rejected.

An independent parse found 890 distinct rows with the declared totals, all return code 1, zero errors, and equal selected/required/failed cardinalities. I spot-checked more than ten old, new, schema, P, D, N, and R declarations—including M001, M014, M041, M042, M070, M100, M133–M135, M198, M220, M244, M245, M272, M282, M328, and M388—against their test designs and logged failures.

Like any ID-based oracle, the rule cannot mechanically prove semantic causality inside a test body. The logged failure location/message supplies that audit trail; I found no false catch in the inspected rows. The 14 D and 7 P helper-surfaced cases are acceptable because those mutations intentionally break the honest pipeline observable rather than a direct reason assertion.

### C4 — RESOLVED

The sampled kinds agree end to end:

- N removes or bypasses the atom’s own refusal.
- P causes only honest-input refusal/crash and has no admission path.
- D changes published/bound state or another required effect.
- R leaves an independent defense active while its own direct test fails.
- C covers independently frozen constants.

The 18 non-schema R rows are valid defense-in-depth cases. The 373 schema R rows are also valid: their own schema clause is directly tested, while the classifier demonstrates that another independent clause still refuses every forged affected entry.

### C5 — RESOLVED

The test diff is one purely appended file: +610/−0. No existing test was edited or weakened.

The P1 golden test and three frozen-vanilla-digest tests remain unchanged and are reported green by name. The new tests’ monkeypatches establish fixtures or isolate documented defenses; they do not restore the mutated production predicate as their oracle. The direct hand-merged EN-135 test and EN-186’s independent-subtraction observable are appropriate corrections. I found no timing, randomness, or shared mutable state likely to make the additions flaky.

### C6 — PARTIAL

The five committed tool copies are line-for-line identical to the scratchpad versions: driver 2,653 lines, enumerator 480, classifier 65, closing script 73, history helper 102.

The driver restores frozen bytes in `finally`, compares after each mutation, refuses tracked dirt, rejects an unexpected frozen registration, and turns SIGTERM into the same restoration path. The disclosed SIGKILL/background-limit incident is inherently uncatchable and was followed by an explicit restore and frozen hash check.

The classifier isolation and mutation-table/JUnit parsing are sound. The material tooling defect is the enumerator omission in HIGH 1. The two Planner scripts have the non-blocking sample-set validation weakness in NIT 1.

### C7 — PARTIAL

The four commits have the required Opus trailer. The records consistently describe the appended tests, proof execution, interruptions, totals, CPU-only suite, collection checks, and unchanged GPU command list. No tracked environment/package change or production change is present.

The records’ “mechanically exhaustive”/0-unmapped conclusion must be revised after the enumerator repair and proof rerun.

Readiness:

- **Push as review-ready:** blocked by HIGH 1 and the resulting proof/record refresh.
- **Final integrative R2-A … rB-F5 review:** should wait for that repair and rereview.
- **Merge into `localization-exp`:** additionally waits for Yixun’s sidecar-stamp-location decision.
- **GV/G1/G2/G3/G3D/G0 and BF probe:** still require a reviewed/pushed pin, Yixun’s go-ahead, and a released/free GPU; BF remains contingent on the within-cap probe.
- **YAW/BF passes:** remain downstream of those gates/probe and card availability.

No new pre-registration is required for a tests/tooling-only proof repair. Reassess only if re-enumeration exposes a production change or changes thresholds, estimands, schedules, or GPU commands.

## Read-only verification performed

- `git status`, commit metadata/trailers, `git diff --stat`, `--numstat`, and `--check`.
- Confirmed empty production diffs from `701926e5` to HEAD and from proof HEAD `08a314c2` to `612f4bb7`.
- Inspected the complete appended test diff, frozen production diff, proof inventory/table, P audit, schema classification, records, and final logs with `git show`, `sed`, and `rg`.
- Ran only write-disabled Python one-liners to invoke the enumerator, reconstruct/compare the 211-entry registration, compare embedded tool sources, and parse the 890 logged rows.
- Did not run pytest, mutation/classification/history drivers, closing scripts, gates, probes, training/evaluation, GPU commands, or process-control commands.
