**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-04 · **Round:** code R2 FULL (exp_22 R2 integrative review: R2-A … rB-F6; 3f595e19 → e89206eb)

**Verdict:** **SHIP**

No new BLOCKING, HIGH, or NIT findings. The 80-commit branch composes correctly against plan r2 plus A1–A6, is technically ready to merge, and is ready for the registered GPU gates once the procedural launch conditions are met. This verdict does not authorize a GPU launch.

## Findings, ranked

- **BLOCKING:** None.
- **HIGH:** None.
- **NIT:** None new.

The four rB-F6 NITs remain accepted, recorded limitations; none blocks the merge, gates, or passes:

1. `scratchpad/rbf6/mutate_rbf6.py:3344`: narrow SIGTERM-before-`try/finally` restoration window. Fix: put the mutation write inside the restoring `try/finally`.
2. `scratchpad/rbf6/final_checks.sh:5`: uses `set -uo pipefail` rather than `set -euo pipefail`. Fix: enable `-e` and explicitly check pytest/collection status.
3. `scratchpad/rbf6/mutate_rbf6.py:3364`: ten table locations show edit-context starts rather than atom locations. Fix: print atom source and edit-context source separately.
4. [loc_meshgrid_worklog.md:801](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:801): “union” wording for whole-definition deletion was imprecise; the corrective interpretation is recorded at [line 826](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:826).

## F1 — PASS: registered protocol

The number-producing paths implement the binding registration exactly.

- The registered checkpoint identities and arm protocols are fixed in [meshgrid_report.py:156](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:156) and [meshgrid_report.py:192](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:192). I independently hashed the three checkpoint files in the FLAC checkout; all equal the registered P1/BF/YAW digests.
- `room_block` computes the within-room median `e_loc` and within-room mean success indicator, and `across_rooms` takes the unweighted mean over sorted rooms, at [meshgrid_report.py:1764](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:1764) and [meshgrid_report.py:1816](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:1816).
- Crossarm constructs treatment-minus-P1 paired room deltas, uses one registered room-resample matrix, applies the registered one-/two-sided tails without finite correction, and passes exactly four p-values to Holm at [meshgrid_crossarm.py:72](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:72), [meshgrid_crossarm.py:83](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:83), and [meshgrid_crossarm.py:130](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:130).
- The exact census is registered at [meshgrid_engine.py:66](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:66): 5,337 queries, 16 rooms, 8,896,540 candidate-query pairs, 966,147 source rows, and 71,172,320 waveforms.
- BF is bound to `fa_invariant` plus C4; YAW and P1 are bound to vanilla plus the explicit `[0.0]` no-op orbit. Batch rows 256, source chunk 16, rotation zero, and requested autocast `default` are part of the binding.

I found no route that can silently substitute a pooled estimator, another arm protocol, another checkpoint, or a different confirmatory family while remaining canonical.

## F2 — PASS: vanilla/P1 path

P1’s frozen behavior is preserved end to end, and YAW uses that same vanilla execution path.

- `REGISTERED_ARM_PROTOCOL` assigns both P1 and YAW the identical vanilla method definition.
- The vanilla engine remains outside the FA split; vanilla shards, merges, and reports refuse FA ledger residue.
- G0 authenticates the committed P1 probe report against `8c5fd5e8…`, requires the exact 16-query anchor, and compares the replayed frozen digests. The registered anchor constant is at [meshgrid_fa_parity.py:78](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:78).
- The recorded focused run includes the P1 semantic golden and three frozen-vanilla-digest guards. The core frozen-path test starts at [test_loc_meshgrid_fa_split.py:89](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_fa_split.py:89); the P1 report golden is at [test_loc_meshgrid_report.py:2302](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/tests/test_loc_meshgrid_report.py:2302).
- `git diff 701926e5..e89206eb -- localize_meshgrid.py src/localization/` is empty.

## F3 — PASS: BF FA path

The R2-B through rB-F3 components compose without a new lifecycle defect.

- The five-way split matches the registered dependency table and calls the released `invariant_conditioning`: query-cached `context_audio` and C4-averaged `context_poses_vit`; candidate-batch `context_poses`; receiver/candidate-cached base `source` and C4-averaged `source_vit`. See [meshgrid_engine.py:510](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:510).
- Executed partition calls are recorded from actual execution and checked by prefix before rows and exactly at completion.
- The validating ledger reader is shared by reopen, merge, census, and report admission at [meshgrid_engine.py:1173](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:1173).
- Each FA row has a sidecar state stamp under `fa_ledger_stamps/`, written after its ledger state and before the row; merged stamps are rebound to the merged segment namespace. See [meshgrid_engine.py:949](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_engine.py:949).
- Failure markers and pending-check sentinels fail closed at reopen, merge, per-arm reporting, and crossarm.
- `--resume` opens a new launch-record segment and new ledger segment at the same reviewed pin. Mixed-SHA resume is refused; any code change or non-admissible marker requires restart in a fresh directory.

I recommend retaining the sidecar stamp. Moving it into each row would change frozen production and row schemas without improving the coordinated-rewrite threat boundary; both representations rely on the same self-authenticated ledger/receipt joins.

## F4 — PASS: gates and throughput cap

The gate set implements the amended rules:

- GV, G1, G2, G0 are exact gates.
- G3 is the fp32 logic gate with the hard `1e-3` per-key tolerance and the structural, mask, orbit, score, argmax, and full/tail requirements.
- G3D is a production-autocast measurement only; `tolerance`, `score_tolerance`, and `match` are null.
- G3 and G0 require all 16 registered probe queries.
- A nominal numerical PASS cannot survive an unregistered binding, missing registered identity, dirty tracked tree, malformed SHA, missing host, or missing CUDA UUID. Committed evidence is independently re-verifiable through [meshgrid_fa_parity.py:1424](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_fa_parity.py:1424).

The command list records A5/A6 exactly:

- One card: `1.5 × GPU_hours_proj × 0.5145`.
- Two concurrent lanes: `1.0 × max(lane0, lane1) × 0.5145`.
- `>150 h` exits nonzero, stops, and asks Yixun before any BF shard starts.

## F5 — PASS: reports and crossarm

Per-arm reporting and crossarm use the same authenticated facts as F1.

- `evaluate_run` starts with non-admissibility refusal, recomputes the run binding, validates the exact FA ledger where applicable, authenticates rows/truth/census/provenance, and only then calculates metrics at [meshgrid_report.py:2387](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_report.py:2387).
- The retrieval handoff is pinned to `1bf62c05…`; I independently hashed the registered file and obtained that value.
- The crossarm CLI exposes the registered `--run-p1 --run-yaw --run-bf --allow-legacy-p1-provenance --retrieval-handoff` interface at [meshgrid_crossarm.py:658](/home/yixunhu/codespace/exp-22-r2-bf-yaw/src/localization/meshgrid_crossarm.py:658).
- P1’s legacy-provenance exception is arm-specific and explicit. It cannot admit a BF or YAW run.
- Holm is produced only with both treatment arms present and exactly four registered tests.

## F6 — PASS: authoritative command list

The authoritative list at [loc_meshgrid_worklog.md:691](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:691) is complete and consistent with the actual parsers.

Its order is correct: tracked-clean/pin preflight → GV/G1/G2/G3/G3D/G0 → JSON re-verification → BF throughput probe and cap arithmetic → YAW → BF → per-arm reports → crossarm.

All operational checks fail closed:

- `set -euo pipefail`, exact pin, and tracked-clean checks.
- Both concurrent probe PIDs are awaited and their statuses checked.
- Above-cap arithmetic exits 1.
- Failure-marker and pending-sentinel checks exit 1.
- Handoff mismatch exits 1.
- Reports and crossarm repeat the code-side admission checks.

`PIN`, card, and per-arm `TAG` placeholders must be concretized at launch. The checkout must have the registered `outputs_loc/`, `weights/`, and `AcousticRooms/` assets, as the command-list precondition states.

## F7 — PASS for merge; GPU evidence still required

The recorded evidence is adequate for the static merge decision:

- Focused suite: **2028 passed, 9 environmental skips, 0 failed** in [the final log](/home/yixunhu/codespace/exp-22-r2-bf-yaw/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_2026-11-04_09:03:05_r2bf6_final_focused_suite.log).
- Whole-repository collection: 5,923 tests, zero collection errors; 5,916 under `src/tests`.
- Statement-complete delete-the-check proof: **1118/1118 caught**, with 606 declared atoms, 603 mapped, three equivalent exclusions, and zero unmapped.
- The proof ended tracked-clean with production byte-identical to `701926e5`.

I did not rerun pytest or the mutation tooling. The remaining unexecuted behavior is intentionally dynamic: the real checkpoint/real dataset stack and the GPU gates. That is why the gates remain mandatory; it is not a static merge gap.

## F8 — PASS: merge integration

`origin/localization-exp` has 39 commits after the merge base, covering records plus exp_14/exp_25 work. There are:

- Zero paths changed by both branches since `3f595e19`.
- No changes upstream under `src/localization/` or `src/tests/test_loc_*`.
- No conflict markers or changed/added-in-both cases in a read-only `git merge-tree` simulation.

The upstream §9 plan amendments survive automatically: the R2 branch did not modify its inherited §8-only plan blob.

Carry the tracked R2 source, seven test files, and records. Leave out the eleven untracked Codex/collection transcript logs. No scratchpad tooling is tracked in the branch; its relevant sources are embedded in the committed proof records.

## F9 — PASS: SOP and readiness

- HEAD and its upstream both resolve to `e89206eb1fc61c50b4913efe29c713d1ec3571d4`.
- All 80 commits have exactly one `Co-Authored-By` trailer.
- Under the project’s defined code-line metric—nonblank, noncomment, nondocstring Python outside tests—the largest production commit is 188 lines, below the plan’s 200-line target. Six commits exceed 200 raw changed lines only because raw numstat includes comments/docstrings.
- The source/test patch passes `git diff --check`. A whole-range check reports trailing spaces inside committed verbatim proof logs; I do not treat archival transcript whitespace as a code/SOP defect.
- No dependency, environment, package-lock, or container file changed. The records state that no environment modification occurred, and I made none.

No additional pre-registration is needed.

## Remaining holds

| Stage | What still blocks it |
|---|---|
| Merge | No technical blocker. Procedurally, record Yixun’s stamp-location decision; I recommend accepting the current sidecar and merging unchanged. Exclude untracked logs. |
| GPU gates | Merge and push the final pin; use that exact 40-hex pin on a tracked-clean checkout with assets; wait for Yixun’s explicit go and released cards after his trainings. |
| YAW/BF passes | All gate JSONs must complete and re-verify; the BF probe/arithmetic/verdict must be recorded; projected makespan must be ≤150 h. Run YAW first, then BF, before any exp_09 seeds. |

## Read-only verification performed

- `git status`, `log`, `show`, `diff`, `diff-tree`, `rev-list`, `merge-base`, `merge-tree`, name-overlap, trailer, commit-size, frozen-production, and whitespace probes.
- Static tracing with `rg`, `sed`, `nl`, and `awk` over the plan, A1–A6, P1 results, all prior reviews, command list, production code, tests, proof records, and commit ledger.
- `sha256sum` of P1/BF/YAW checkpoints, the P1 probe report, retrieval handoff, and preregistered-digests record.
- No files written or modified; no pytest, driver, gate, probe, training/eval, GPU, package operation, mutation tooling, or process interaction.
