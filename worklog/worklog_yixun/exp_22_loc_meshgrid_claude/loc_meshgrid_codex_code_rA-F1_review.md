**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-02 · **Round:** code rA-F1 (exp_22 R2-A fixes: mixed-SHA refusal, retrieval-handoff consumption, NITs; 3dc94b9b → 463b1976)

**Verdict:** SHIP

The rA BLOCKING and HIGH findings are closed. I found no remaining BLOCKING or HIGH code defect. Two record/documentation NITs should be closed before the round is formally pushed/handed back.

## Ranked findings

### NIT — one Planner-audit statement is literally inaccurate

`reconcile_controls_elsewhere` is AST-identical to `3dc94b9b`, but `read_retrieval_handoff` is not: its parsing body was intentionally factored into `_parse_retrieval_handoff`, and the reader is now a wrapper at [meshgrid_offgrid_probe.py:704](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_offgrid_probe.py:704). My AST comparison returned `False` for the reader and `True` for reconciliation.

Concrete fix: change the audit wording to “`reconcile_controls_elsewhere` is AST-identical; `read_retrieval_handoff` is behavior/output-compatible after its old parsing body was factored into `_parse_retrieval_handoff`.” The regression test adequately pins that compatibility at [test_loc_meshgrid_offgrid_probe.py:3459](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_offgrid_probe.py:3459).

### NIT — SOP review records remain untracked

The tracked tree is clean, but `git status` shows the prior rA review artifact and Codex review logs untracked, including `loc_meshgrid_codex_code_rA_review.md`. Before formal push/handback, commit the rA review, this rA-F1 review, their review logs, and the Planner rerun result on the appropriate records branch.

The large R2-A collection listing may remain excluded as previously disclosed. Also preserve the authoritative §9 amendments already committed on `origin/localization-exp` as `be2aa0fe`; the feature branch’s local plan copy still ends at §8.

## F1 — RESOLVED

The one-reviewed-pin rule is now enforced at every required layer:

- `one_pin_reason` requires exactly one Git SHA: [meshgrid_segments.py:63](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:63).
- A resume at a different commit is refused before record copying or ledger mutation: [meshgrid_segments.py:124](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:124).
- `provenance_census` derives SHAs from every ledger segment, including row-less segments, and makes `complete=False` when their union is not exactly one: [meshgrid_segments.py:247](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:247).
- `merge_ledgers` checks the cross-shard union before its dry-run return: [meshgrid_segments.py:308](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:308). Because the engine dry-runs this before creating the merged artifact, refusal writes nothing.
- Report canonicality continues to consume the re-derived census, so hand-assembled mixed-SHA runs become NON-CANONICAL.

The load-bearing controls are present: different-commit resume, row-less segment, single-commit success, dry-run and engine merge refusal, and hand-assembled report refusal at [test_loc_meshgrid_launch_record.py:731](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_launch_record.py:731) and [test_loc_meshgrid_report.py:2584](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2584).

## F2 — RESOLVED

The sparse-control handoff is now a real admission input rather than prose:

- The registered historical handoff is pinned in code at [meshgrid_report.py:212](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:212).
- `consume_retrieval_handoff` reads once and checks the digest, control key, exact completed/canonical status, registered census, report digest, and complete registered headline: [meshgrid_offgrid_probe.py:775](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_offgrid_probe.py:775).
- Per-arm reports gate canonicality and emit the reconciled or NOT-RECONCILED entry at [meshgrid_report.py:384](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:384).
- Crossarm admission defers this one gate, then the crossarm applies it once at its own level: [meshgrid_crossarm.py:448](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:448), [meshgrid_crossarm.py:505](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:505).
- Both CLIs expose `--retrieval-handoff`. An absent or unreconciled handoff leaves BF/YAW reports and the entire crossarm NON-CANONICAL while preserving labelled diagnostic statistics.
- The published legacy P1 comparator remains canonical without the flag, explicitly says STATUS NOT RECONCILED in this newly built report, and never says “reused.” Its branch is tested at [test_loc_meshgrid_report.py:2711](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2711).

Pinning the digest in code is sound. This is one previously completed, registered control, so its identity belongs in the registry rather than being operator-selected through an `--expect` flag. The optional expected-digest parameter is used only as a test seam; both production call sites use the registered default.

The committed-pair test at [test_loc_meshgrid_offgrid_probe.py:3418](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_offgrid_probe.py:3418) is load-bearing for repository integrity: it proves the constant names the handoff consumed by the canonical P1 probe and that the committed report bytes match the report digest inside that pinned handoff. Runtime intentionally does not reread the report itself.

I independently reproduced:

- Handoff: `1bf62c05d1bf796025590fc2585ca4041a695471fba6fd460e947cbbf351bbf9`
- Report: `65d735356c5b5611bba46e1d3c83a64e0a1469f9456ff1e3089601e0b7a884c8`
- Canonical P1 probe references that same handoff digest.
- Handoff states `run (canonical)`, `canonical=true`, and `5,337 / 16`.

## F3 — RESOLVED

All three NITs are now delete-the-code tested:

- Every registered room and both primary endpoints must exactly match the arm report: [meshgrid_crossarm.py:485](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:485), with compensating-difference and missing-room tests at [test_loc_meshgrid_crossarm.py:719](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:719).
- The independent test pin enumerates all 24 whitelist members, exercises each individually, and derives/refuses every fixture field outside the whitelist plus an unseen future field: [test_loc_meshgrid_crossarm.py:423](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:423), [test_loc_meshgrid_crossarm.py:568](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:568), [test_loc_meshgrid_crossarm.py:592](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:592).
- The complete P1 single-shard note, not merely its tail, is pinned at [test_loc_meshgrid_report.py:2439](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2439).

## F4 — PARTIAL only for the audit-wording NIT above

The tests themselves are load-bearing and honest.

The five RED logs show the intended failures. Cycle 4 transparently records an initial circular test-import mistake and then the real nine-test RED rerun; it does not present the collection mistake as product evidence. M1–M5 each kill the intended assertion, and the currently tracked implementation is byte-identical to HEAD.

The key rename to `arm_independent_citation` did not weaken coverage. Tests inspect serialized JSON plus rendered Markdown and require:

- “reused” absent from every unreconciled artifact;
- “REUSED, NOT RERUN” present only after reconciliation;
- no such word for legacy P1;
- both full digests and the reconciled headline present when cited.

The four pre-existing behavioral tests were changed appropriately: two stale-static-text expectations were inverted, and two YAW-report tests now pass the valid handoff so their original launch-provenance assertions remain isolated. The single-shard test was separately strengthened.

## F5 — PARTIAL pending record finalization

My independent AST counter reproduced the ledger exactly, counting nonblank/noncomment/nondocstring Python outside `src/tests/`:

| Commit | Net code lines |
|---|---:|
| `d9c3ac05` | +26 |
| `b55bd7f6` | +15 |
| `e2adccba` | +69 |
| `166caa18` | +84 |
| `05b9c455` | +78 |

All are below 200. Every code commit and the records commit carries the disclosed `Co-Authored-By: Claude Opus 5.5` trailer. The scratchpad counter is acceptable because it is not shipped code and the ledger has now been independently reproduced; commit it only if it becomes a reusable experiment tool.

The scoped diff changes only the four named localization modules, five tests, and records. No dependency/environment file changed, no installation command appears, and the logs consistently record CPU execution with `CUDA_VISIBLE_DEVICES=''`.

Coder evidence records 990 passed, 9 environmental skips, 0 failed and 4,885 collected with zero collection errors. I did not rerun pytest, per instruction.

## F6 — RESOLVED as a readiness audit

### (a) Pushing the branch

No code defect blocks pushing. Before formal handback:

1. Let the Planner’s focused rerun finish cleanly and record its result.
2. Commit the prior rA review artifact, this review artifact, and their review logs.
3. Preserve `be2aa0fe`’s authoritative §9 amendments when this feature branch is eventually integrated.

The tracked tree is otherwise clean; the branch is 20 commits ahead of `origin/exp-22-r2-bf-yaw`.

### (b) R2-B

CPU implementation can begin after record closure. R2-B still must deliver the entire A8 list already copied into [loc_meshgrid_worklog.md:306](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:306):

- Five-way FA dependency split.
- FA accepted only with that split engaged.
- Complete executed partition and runtime evidence.
- Probe serialization through `binding_fields(binding)`; the current truncating code remains at [meshgrid_engine.py:1558](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1558).
- Three parity gates with full/tail coverage.
- Frozen vanilla replay digest.
- Fake-conditioner dependency/counter-tests.

Completion of R2-B, specifically its GPU parity evidence, remains gated on a free card and Yixun’s explicit go. Current `assert_cacheable` still correctly refuses FA at [meshgrid_engine.py:1630](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1630).

Add two operational items to the R2-B/R2-C brief:

- Update every canonical BF/YAW report and crossarm command to pass `--retrieval-handoff` and preflight its SHA-256 against `1bf62c05…`. The current plan command at [plan_loc_meshgrid_r2_bf_yaw.md:67](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/plan_loc_meshgrid_r2_bf_yaw.md:67) would now intentionally produce a NON-CANONICAL report because that flag is absent.
- State explicitly that any code change after a shard starts invalidates resume for that run pin: restart the affected shard instead of resuming it at a new commit.

### (c) YAW in R2-C

The vanilla YAW execution path is technically available at this head, but the registered R2-C launch remains blocked by:

- R2-B completion and the final integrative review;
- a tracked-clean final code pin;
- Yixun’s go and free GPUs;
- YAW cache parity plus per-shard launch records;
- corrected report/crossarm commands including the pinned retrieval handoff and legacy-P1 flag.

A YAW-only crossarm can publish its registered raw p-values, but the four-test Holm family and final verdicts remain incomplete until BF finishes.

## What I verified

Read-only probes included:

- `git status`, `rev-parse`, branch ancestry, full commit messages/trailers, `diff --stat`, `--name-status`, `--numstat`, scoped diffs, and `git diff --check`.
- Line-by-line inspection of the implementation, relevant tests, plan, prior review, worklog, ledger, RED logs, final-suite log, and collection/check log.
- Independent AST LOC counting from `git show` blobs for every code commit.
- Independent AST comparison of the two reused helper functions.
- `sha256sum` of all seven rA-F1 logs and the handoff/report/probe artifact trio.
- Read-only JSON extraction confirming the handoff status/census, report digest, and canonical probe reference.
- Confirmation that `localize_meshgrid.py`, `meshgrid_engine.py`, data/model/config/training files, and dependency files are unchanged in the reviewed range.

I did not run pytest, repository modules, drivers, training/evaluation, GPU commands, package managers, compilation that could create caches, or process-management commands, and I made no file changes.
