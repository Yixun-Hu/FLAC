**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-04 · **Round:** code rB-F6 (exp_22 R2-B tests-and-proof round: the statement-complete delete-the-check proof; 873b230f → d5d02423)

**Verdict:** **SHIP**

No BLOCKING or HIGH findings. rB-F5 HIGH 1 is resolved. The proof now exhaustively covers its explicitly declared operator set, production remained frozen, and the six appended tests provide the missing direct observables.

## C1 — RESOLVED

My independent AST enumeration agreed exactly with the inventory:

- 307 scoped statements; enumerator 307; missing 0; extra 0.
- 81 rule exclusions: 52 module-level definition containers and 29 docstrings; missing 0; extra 0.
- 606 total atoms: 603 mapped, 3 equivalent mutants excluded, 0 unmapped.
- The scoped created-function set is exactly 26.
- I found no unenumerated scoped writer, argument/container mutator, nested publishing call, or verdict-relevant assignment.

The three named statements map correctly:

- [`bucket[key] = rest`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:942) → M925. Its required test checks the exact residual, whole-shape removal, and underflow.
- [`return write_json(...)`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:970) → M933. Its new test checks the exact published stamp.
- [`top[id(segment)] = ...`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1022) → M949. Its parametrized test checks both the genuine state and forged-state refusal.

The AST-exact mapping rule is sound: a statement maps only to the mutant whose module AST is exactly the frozen AST with that statement replaced by `pass` or `else: pass`. Inside/enclosing edits do not count, and ambiguous multi-statement deletions map neither.

[`blocks = None`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1325) is equivalent: the error path raises before reading it, while the successful path necessarily rebinds it at line 1327.

Treating module-level definitions as scope containers is acceptable for the declared function-body operator set. The records’ stronger “union” wording is imprecise; see NIT 4.

## C2 — RESOLVED

I traced the requested rows from generated edit through selected test and failure observable, including:

- Six re-declared misses: M978, M1013, M1014, M1019, M1040, M1094.
- Nine cross-file rows: M896, M901, M907, M1034, M1097–M1101.
- Nested effect M1118.
- The three named rows M925, M933, M949.

The five schema-string deletions M1097–M1101 are not disguised collection errors. The selected launch-record test collects, enters its body, calls `_preflight`, and only then triggers the lazy import and `NameError`. M901/M907 similarly fail ordinary runtime helper calls in independently collected test modules. The JUnit evidence reports failures, not errors.

The generic CAUGHT rule cannot alone prove predicate-level causality: any deletion-induced failure of the selected test could technically satisfy it. Here that risk is controlled by isolated one-edit mutants, exact selection, logged deepest frames, and manual inspection of the required test designs. For the inspected rows, each failure is causally tied to the deleted statement’s stated effect.

M1118 correctly removes only `_production_query_reasons(...)`; the tests observe both loss of the query-specific reason and the resulting incorrect folds.

## C3 — RESOLVED

The 18 new P rows’ no-admission arguments are sound. They either remove permissive branches/returns or make honest execution fail through an unbound binding or missing normalization; none creates an admission path.

The eight R rows are genuinely redundant end to end:

- M977: `fresh_out_dir` string normalization; its production caller already supplies a string.
- M1012–M1015: census-side re-verification of a merged ledger already constrained during construction.
- M1027–M1029: merge-side row binding re-verification of already verified/copied shard evidence.

Their direct tests still observe the defenses in isolation.

The two reclassifications are correct:

- M948/M949 are N: without the highest-state tracker, a forged one-query row can bind to the empty prefix.
- M977 is R under the frozen production call graph.

Helper-surfaced N/C failures are acceptable evidence here. They occur during runtime helper calls—not pytest collection or infrastructure setup—and deletion makes every relevant invocation unusable before the final assertion. Calling these “setup helper” failures is slightly loose terminology but does not change the evidence.

## C4 — RESOLVED

The operator claim is now honest and adequately bounded:

1. Decision-atom deletion/neutralization.
2. AST-exact whole-statement deletion.
3. Scoped effect deletion.
4. Schema-entry deletion/retyping and registration-token deletion.
5. Registered constants.

Operator replacement, boundary shifts, alternative arguments, unregistered constants, and out-of-scope code are explicitly excluded. Therefore this is not universal semantic mutation completeness, but it is a valid exhaustive claim over the stated operator set.

I found no particular excluded boundary/operator that must be mutation-run before the gates. Important boundaries already have direct two-sided tests, including exact/one-ahead/beyond ledger states, residual zero/positive/negative, registered/no tolerance, and finite/non-finite measurements.

No new pre-registration is required.

## C5 — RESOLVED

The test change is exactly +133/−0 in one appended hunk after the rB-F5 tests. No existing assertion, fixture, golden, or frozen-digest test was modified.

[`test_a_scored_pass_through_main_claims_its_out_dir_before_the_pass_runs`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:5741) exercises the real `driver.main` → `claim_out_dir` path. The stack, preflight, launch admission, and pass body are stubbed, but binding publication, ledger opening, launch-record attachment, handoff to `run_pass`, and second-binding refusal are real.

The new tests use deterministic synthetic fixtures and `tmp_path`; I found no credible flake source. Logs report 2028 passed, 9 environmental skips, and the Planner independently reproduced that result. I did not run pytest.

## C6 — RESOLVED, with NITs

The enumerator/driver changes implement the claimed behavior:

- Generated edits must be unique, compile, and match exact statement-deletion AST digests.
- Literal statement declarations refuse stale source.
- Every required node ID must collect.
- Dirty tracked trees are refused.
- Every mutant is restored and byte-compared.
- Post-run production diff and tracked state are recorded.
- The classifier is byte-identical to rB-F5.
- Embedded driver, enumerator, classifier, final-check, history, and declaration-aid sources match their scratchpad copies.

The Planner’s comparison script fixes rB-F5 NIT 1: it enforces the expected ID set and count, rejects duplicates/out-of-range IDs, requires exact missing/extra equality and `TOTAL n/n caught`, checks all 1118 Coder rows, compares normalized row bodies and header/inventory blocks, and requires `UNMAPPED 0`.

The pick/sample and old-row identity scripts are consistent with their stated jobs. The assembler and declaration aids did not introduce an inaccurate committed record. The two FLAC record-patching scripts use exact-count replacements; commits `645c61b7` and `8ef0e1ee` touch only their stated record documents, and their resulting descriptions match the reviewed rounds.

## C7 — RESOLVED for rB-F6

The records accurately preserve RED/GREEN evidence, CPU-only execution, no environment change, frozen production, zero code lines, and the Opus trailer on all three commits. The authoritative GPU command list remains the 2026-11-03T17:17:46-05:00 entry.

Remaining readiness steps are procedural:

- **Push:** no code/evidence blocker; commit this review through the established workflow, then the Planner can push.
- **Final integrative review and merge:** perform the whole-branch R2-A…rB-F6 review after push. Merge also waits for that review’s SHIP verdict and Yixun’s sidecar stamp-location decision.
- **GV/G1/G2/G3/G3D/G0 and BF probe:** wait for settled integration, explicit go, and a free card; use the unchanged registered command list.
- **YAW/BF passes:** wait for successful gates/probe, free cards, explicit go, and the existing BF-duration rule. Any production change requires the corresponding evidence review/re-run.

## Findings, ranked

### NIT 1 — a SIGTERM can land before restoration is protected

The driver writes the mutant at [`mutate_rbf6.py:3344`](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf6/mutate_rbf6.py:3344), then enters `try` on the next line. A SIGTERM in that narrow interval could bypass the restoration `finally`.

Fix: place the mutation write itself inside an enclosing `try/finally`, with the original captured before it.

**Blocks SHIP:** No; the completed run ended tracked-clean with byte-identical production.  
**Known limitation acceptable:** Yes; recording it does not invalidate the R2 evidence.

### NIT 2 — `final_checks.sh` does not fail fast

[`final_checks.sh:5`](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf6/final_checks.sh:5) uses `set -uo pipefail`, not `set -euo pipefail`. A failing pytest or collection command can therefore be followed by later successful commands and a successful script exit.

Fix: enable `-e` and explicitly assert pytest/collection status and zero collection errors.

**Blocks SHIP:** No; the logged summaries are green and the Planner independently reproduced them.  
**Known limitation acceptable:** Yes; it weakens the helper’s exit-status enforcement, not the recorded results.

### NIT 3 — ten mutation-table locations show edit context rather than the atom

The table derives location and code from `edits[0][0]` at [`mutate_rbf6.py:3364`](/tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/rbf6/mutate_rbf6.py:3364). Consequently M903, M929, M953, M976, M977, M991, M999, M1002, M1048, and M1092 show the context-grown edit’s first line. M977 is the clearest example: the atom is line 1160, while the table displays line 1158 and an import statement.

Fix: print the atom declaration’s source line/code separately from the edit-context start.

**Blocks SHIP:** No; atom IDs, exact AST checks, and actual edits identify the correct targets.  
**Known limitation acceptable:** Yes; this is an evidence-readability defect only.

### NIT 4 — whole-definition deletion is not literally a union of body-statement deletions

The worklog says a module-level definition’s deletion is “the union of its statements’ deletions” at [`loc_meshgrid_worklog.md:801`](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:801). Removing a definition also removes its name binding and is not AST- or behaviorally identical to deleting every body statement.

Fix: say that whole-definition deletion is an intentionally excluded coarse container mutation, while its scoped body statements are independently covered.

**Blocks SHIP:** No; excluding container deletion is sound for the declared function-body operator set.  
**Known limitation acceptable:** Yes; correcting the rationale is sufficient and does not invalidate R2 evidence.

## Read-only verification performed

- `git log/show/diff/status`, per-commit frozen-production comparisons, `git diff --check`, stats, trailers, and FLAC record-commit inspection.
- Independent read-only Python AST enumeration and the permitted read-only enumerator collection: 606 atoms, 307 statements, 81 rule exclusions, 0 unmapped.
- Static tracing of the requested mutation rows, required tests, failure frames, production call paths, P/R arguments, and equivalent mutants.
- `sed`, `rg`, `nl`, `diff`, `cmp`, and `sha256sum` over committed logs, frozen code, tests, embedded tools, scratchpad tooling, comparison scripts, assembler, and record-patching scripts.
- No files written or modified; no pytest, mutation driver, classifier, final-check/history/lane script, package operation, GPU command, or process interaction was performed.
