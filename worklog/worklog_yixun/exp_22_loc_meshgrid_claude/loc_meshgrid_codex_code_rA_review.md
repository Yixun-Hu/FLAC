**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-02 · **Round:** code rA (exp_22 R2-A: provenance, arm-aware registration, paired crossarm; 3f595e19 → 3dc94b9b)

**Verdict:** REQUEST-CHANGES

The registered statistics, arm-aware protocol gates, legacy-P1 behavior, and most launch-segment provenance are correctly implemented. One explicit Planner ruling remains blocking: a canonical run can currently merge and report segments produced at different Git SHAs.

## Ranked findings

### BLOCKING — mixed Git SHAs are published, not refused

The census collects `git_shas` but does not turn multiple values into a failure: [meshgrid_segments.py:247](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:247), [meshgrid_segments.py:267](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:267). `merge_ledgers` checks each shard independently and returns successfully from its dry run without checking the union of SHAs: [meshgrid_segments.py:283](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:283). The report then accepts any complete census: [meshgrid_report.py:735](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:735).

Thus two clean, environment-verified segments from different commits can form a canonical merged run, contrary to the Planner’s one-reviewed-pin ruling.

Minimal fix:

- In `provenance_census`, add a reason whenever launch-record generation has `len(git_shas) != 1`. This also rejects a same-shard resume after a code change.
- In `merge_ledgers`, check the union of all shard-record Git SHAs before the `dry_run` return, so merge refuses before writing anything.
- Preserve report behavior: the same census reason must make a hand-assembled mixed-SHA report NON-CANONICAL.
- Add three tests: mixed-SHA resume, mixed-SHA cross-shard dry-run merge, and report NON-CANONICAL for a mixed-SHA ledger.

### HIGH — sparse-control reuse is asserted without consuming its canonical handoff

Plan §1 requires a BF/YAW report to cite the sparse retrieval control only after consuming the completed canonical handoff and its digest. Instead:

- The per-arm report still publishes static, stale “built (r9b), run pending” text: [meshgrid_report.py:340](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:340).
- Crossarm states that sparse retrieval is “reused” unconditionally: [meshgrid_crossarm.py:421](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:421).
- Its CLI has no retrieval-handoff input: [meshgrid_crossarm.py:549](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:549).

The completed handoff exists and is canonical, but this code never reads it.

Fix: require the handoff for canonical BF/YAW and crossarm reports; hash and parse the same buffer; validate its control key, canonical status, completed status and census; publish the handoff/report digests and reconciled headline. If absent, say “STATUS NOT RECONCILED,” never “reused.” The existing reviewed helpers in `meshgrid_offgrid_probe` provide the intended semantics, though shared code may need factoring to avoid a circular import.

### NIT — several binding claims are correct in code but not fully delete-the-code tested

- Crossarm and report currently compute the same per-room values because both use `mr.room_block`, but `run_crossarm` only asserts equality of the across-room point, so compensating per-room differences would pass: [meshgrid_crossarm.py:459](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:459). Compare the full `per_room` mappings.
- The whitelist is pinned exactly, but the permissive test mutates only five representative entries, not every whitelist member: [test_loc_meshgrid_crossarm.py:523](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:523).
- The P1 single-shard note test pins only its tail rather than the complete base string: [test_loc_meshgrid_report.py:2429](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2429).

## A1 — RESOLVED

- K=8/LME per-room median `e_loc` and `success_raw@1.0`, followed by an unweighted mean of treatment−P1 room deltas, are correct: [meshgrid_crossarm.py:72](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:72).
- The matrix is produced by the report’s registered `default_rng(20260825)`/10,000 constructor and shared across all four tests: [meshgrid_crossarm.py:94](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:94), [meshgrid_crossarm.py:171](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:171).
- Inclusive-zero tails, no correction, BF one-sided and YAW two-sided arithmetic are exact: [meshgrid_crossarm.py:104](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:104).
- Holm refuses anything other than the exact four labels: [meshgrid_crossarm.py:130](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:130).
- Verdict classes, including wrong-sign BF refusal, are correct: [meshgrid_crossarm.py:146](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:146).
- Incomplete families report raw p only: [meshgrid_crossarm.py:186](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:186).
- The expectations are genuinely independent numerically: direct hand examples cover tails and endpoints, while the end-to-end expectation reconstructs draws and Holm directly with NumPy rather than calling the implementation: [test_loc_meshgrid_crossarm.py:103](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:103), [test_loc_meshgrid_crossarm.py:225](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_crossarm.py:225).

The current per-room results agree structurally with `meshgrid_report.build_cell`; the NIT above asks for an explicit full-mapping gate.

## A2 — PARTIAL

Admission correctly goes through `evaluate_run` and `build_report`, with canonical gates forced and authenticated results retained: [meshgrid_crossarm.py:283](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:283). Pairing is fail-closed over all non-whitelisted binding and row fields, with explicit candidate-coordinate, truth, batching, receipt, census, metadata-bank and CRN checks: [meshgrid_crossarm.py:352](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:352).

The sparse-bank digest is correctly absent as a pairing axis. The P1 exception is explicit and printed, and the claim boundary is printed. Status is PARTIAL solely because the sparse-control handoff is not consumed.

## A3 — PARTIAL

Resolved portions:

- Probe launch-record builder/admission are imported, not copied: [meshgrid_segments.py:26](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_segments.py:26).
- Records remain outside the shared binding.
- Rows are segment-stamped; resume rejects unknown stamps: [meshgrid_engine.py:1997](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1997), [meshgrid_engine.py:1322](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1322).
- R2-generation merge requires complete provenance before writing, while legacy R1 merge remains unchanged: [meshgrid_engine.py:2343](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:2343).
- Records are hash-verified while copied and row ranges are published.
- Report re-derives the census and gaps become NON-CANONICAL.
- Diagnostics are record-free.

The blocking mixed-SHA rule remains.

## A4 — RESOLVED

The complete arm registry, including the disclosed BF partition, is correct: [meshgrid_report.py:145](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:145). It registers context-ViT micro/cap 1, scored-batch micro 32, source-cache micro/cap 16, one angle per chunk, one shared angle and three additional orbit forwards.

Arm resolution precedes method comparison; FA partition omissions and `expect_arm` mismatches are hard refusals even under deviation mode: [meshgrid_report.py:749](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:749). Embedded training protocol checks are present. Driver preflight ordering is binding → arm protocol → embedded protocol → ARE/condition binding → `assert_cacheable`: [localize_meshgrid.py:305](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:305).

Parsing enforces exact vanilla/C4 orbits, zero rotation and normalized `-0`: [localize_meshgrid.py:85](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:85), [localize_meshgrid.py:184](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:184). A scored pass requires a record and all six explicit flags: [localize_meshgrid.py:199](/home/yixunhu/codespace/exp-22-r2-bf-yaw/localize_meshgrid.py:199). No vanilla-evaluated BF or FA-evaluated YAW path can become canonical.

`assert_cacheable` still refuses FA, as R2-A requires: [meshgrid_engine.py:1630](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1630).

## A5 — RESOLVED

The P1 metric/oracle/baseline/association/census/case/crosscheck/protocol goldens are hard-pinned and exercised against the current report: [test_loc_meshgrid_report.py:2232](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2232). The implementation diff does not alter the P1 metric computation.

`RUN_BINDING_FIELDS` is unchanged; all-or-none R2 fields preserve the legacy digest path and reject partial generations: [meshgrid_engine.py:913](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:913), [meshgrid_engine.py:945](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:945). Legacy provenance requires P1, legacy generation and no record anywhere: [meshgrid_report.py:725](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:725).

The current P1 note bytes match the base diff. Strengthening the full single-shard-note test remains a NIT.

## A6 — PARTIAL

The tests are strong and cover statistics, exact tails, shared matrix, all verdict classes, hard arm/partition refusals, embedded contradictions, segment corruption, resume attribution, unknown records, merge/report census paths, and preflight call ordering.

They are incomplete under the requested delete-the-code standard because:

- Mixed-SHA refusal is absent.
- Not every whitelist item is independently exercised.
- Full per-room report/crossarm equality is not asserted.
- The complete P1 single-shard string is not pinned.

## A7 — RESOLVED

- History is exactly 13 code commits plus one records commit.
- My read-only LOC counter reproduced the ledger counts exactly: `13, 89, 151, 76, 26, 90, 60, 73, 148, 161, 188, 2, 18`; all are below 200.
- The three raw implementation additions over 200 are 202, 220 and 215 lines, attributable to docstrings/prose; acceptable under the plan’s defined code-line measure.
- Rebuilding the unpushed 227-code-line attempt with `reset --soft` into 151/76-line commits is acceptable.
- The records commit contains the 13 RED logs, final focused-suite log, ledger and append-only worklog entries. The whole-repo collection log remains untracked as disclosed.
- Every commit has the disclosed co-author trailer; accepted per the brief.
- No dependency/environment files changed. Logs record CPU-only test execution. I found no install/upgrade commands.

## A8 — RESOLVED as a boundary audit

R2-C can currently run YAW and produce a YAW-vs-P1 raw-p comparison, but it cannot:

- Launch BF: `assert_cacheable("fa_invariant")` refuses.
- Complete the registered four-test family or issue Holm verdicts without BF.
- Proceed canonically until the mixed-SHA and retrieval-handoff findings above are fixed.

R2-B must deliver:

- The five-way FA split in `meshgrid_engine.py`: `context_audio` once; C4-averaged `context_poses_vit`; candidate-batch `context_poses`; C4-averaged cached `source_vit`; cached cylindrical-base `source`.
- `assert_cacheable(cond_method, split)` accepting FA only when that split is engaged.
- An executed partition matching every `run_binding.json.fa_partition` path:
  `frame_avg_chunk_plan`, `cap_policy`, `orbit_size`, `angles_per_chunk`, `n_orbit_forwards`, `shared_angle_count`, `context_vit.{candidate_micro_batch,frame_avg_fwd_cap}`, `context_scored_batch.candidate_micro_batch`, and `source_cache.{candidate_micro_batch,frame_avg_fwd_cap}`.
- Runtime evidence of that executed partition, not merely the current CLI-derived declaration. The same block should be published in the run summary/parity evidence.
- Update [meshgrid_engine.py:1558](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1558) to serialize `binding_fields(binding)`; otherwise the required BF throughput-probe artifact drops the R2 protocol/partition fields.
- The three committed parity gates: split versus released invariant conditioning at matched batching/autocast off; cached versus uncached matched-batching bit identity with candidate counter-tests; production 256/16/default-autocast replay with per-key ≤1e-3, shapes/dtypes/masks/orbit partition, score/argmax stability, and full/tail batches.
- The frozen vanilla replay digest `ee2ba80a…` and fake-conditioner tests for every dependency-table cell.

The implemented `crossarm_r2.json/.md` CLI, optional descriptive BF-vs-YAW pairing, and unchanged per-arm `meshgrid_r1_report.*` names match the amended brief.

## What I verified

Read-only probes included:

- `git status`, `rev-parse`, `diff --stat`, `diff --name-status`, scoped diffs, and `git diff --check`.
- Confirmation that nothing under `src/data`, `src/models`, `train.py`, `eval_FLAC.py`, configs, or `gen_model_comparison.py` changed.
- Full commit log, commit/trailer inspection, raw stats, and an AST-based nonblank/noncomment/nondocstring LOC count.
- Line-by-line inspection of the approved plan, prior Codex plan review, worklog, ledger, implementation and relevant tests.
- RED/final log tails and SHA-256s; recorded final result is 932 passed, 9 environmental skips, 0 failed, with 4,820 tests collected and no import errors.
- A standalone NumPy seed probe was unavailable because this shell Python has no NumPy; I did not install or alter anything. The RNG construction was instead verified source-to-source against the independent test.

I did not run pytest, repository modules, drivers, training/evaluation, GPU commands, package managers, or process-management commands, and made no file changes.
