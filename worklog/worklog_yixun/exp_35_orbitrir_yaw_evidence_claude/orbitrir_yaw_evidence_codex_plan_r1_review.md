**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan r1 (exp_35 orbitrir_yaw_evidence: plan r1 at 52305fb8)

**Verdict:** REVISE

The scientific scope and preregistration are substantially repaired. Two implementation-design gaps still stop launch: the no-flash mechanism is not yet fail-closed under Python’s actual import order, and GPU sharing does not mechanically restrict the co-tenant to the approved s43 trainer. Neither requires a new Yixun decision.

## R1 — r0 findings

| Finding | Status | Assessment |
|---|---|---|
| P1 scope | **RESOLVED** | Phase (i) now contains 30 B-F cells, two P1 controls, and the B-F probe, with no CSV splice ([plan:12](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:12), [plan:32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:32)). |
| P2 arm-aware protocol/regressions | **RESOLVED** | Per-arm protocol, exp_25 cell-manifest byte equality, and results-JSON byte equality are specified and tested ([plan:51](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:51), [plan:232](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:232)). |
| P3 Tier 2 semantics | **RESOLVED** | Tier 2-C4 uses B-F’s own matched-offset FA control; 45°/random are descriptive; T-dup remains exact ([plan:106](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:106)). |
| P4 backend provenance | **PARTIAL** | The committed hash-bound overlay, sentinels, receipt v2, and CTL gate are good, but Python path precedence and ignored bytecode leave the backend assertion non-fail-closed ([plan:61](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:61)). |
| P5 E2 reference | **RESOLVED** | The exact A6 CSV and per-K rows are controlling, with thresholds and units fixed ([plan:94](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:94)). |
| P6 collector separation | **RESOLVED** | Hash/schema/pin/missing/invalid validation, external namespace, H35 digest isolation, stream registry, and exp_25 regression are specified ([plan:130](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:130)). |
| P7 figure | **PARTIAL** | CANON-45 ×8 is mandatory and provenance columns exist, but `source_sha256 = CSV sha plus row key` is not a SHA-256 field and cannot be validated cleanly ([plan:165](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:165)). |
| P8 GPU plan | **PARTIAL** | Lane, chunks, guard thresholds, reservation and ETA are present, but the runner accepts any foreign process under `ALLOW_SHARED_GPU=1`; the s43→table-job handover is not mechanically enforced ([plan:177](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:177)). |
| P9 preregistration | **RESOLVED** | E1–E5 separate rules from expectations, FD is retained, and the launch checklist is frozen ([plan:91](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:91), [plan:215](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:215)). |
| P10 exp_22 | **RESOLVED for exp_35** | Deferral is safe: the exp_22 watcher is unbuilt/suspended and exp_35 needs only the reservation contract. Nothing numerical in exp_35 depends on exp_22 ([plan:59](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:59), [plan:265](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:265)). |

## R2 — no-flash overlay: NOT RESOLVED

The stub content has the stated SHA-256, and the intended fallback in [transformer.py:12](/home/yixunhu/codespace/FLAC/src/models/transformer.py:12) is correct. However:

1. Launching `python /repo/eval_FLAC.py` puts the script directory `/repo` before `PYTHONPATH`. Therefore `PYTHONPATH=/repo/overlays/no_flash` is not “exactly first.” A repo-root `flash_attn.py` or `flash_attn/` can win without being covered by `RELEVANT`. The current evaluator launch is directly at [depth_canon_eval_run.sh:308](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:308).

2. `__pycache__/` and `*.pyc` are ignored at [.gitignore:2](/home/yixunhu/codespace/FLAC/.gitignore:2), while `tree_digest` hashes only tracked and unignored files ([runner:119](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:119)). A valid ignored overlay bytecode file can execute without entering the digest.

3. The receipt’s `"sdpa"` value remains a runner-supplied assertion. The sentinel is strong evidence only after path and bytecode ambiguity are removed.

4. The “outside packaging” claim is false as written. `[tool.setuptools.packages.find] where=["."]` at [pyproject.toml:40](/home/yixunhu/codespace/FLAC/pyproject.toml:40) uses namespace-package discovery, so the nested package can be included as `overlays.no_flash.flash_attn`.

Concrete fix: use a committed same-interpreter launcher that establishes an explicit path order with the overlay first, rejects overlay bytecode, sets `PYTHONDONTWRITEBYTECODE=1`, asserts the exact `flash_attn` spec origin, asserts `src.models.transformer.__file__`, and asserts `flash_attn_func is None` before executing `eval_FLAC.py`. Hash the launcher and overlay. Alternatively use a single `overlays/no_flash/flash_attn.py` plus an explicit packaging exclusion. Add tests for repo-root shadowing, ignored `.pyc`, wrong spec origin and inherited path removal.

No other direct repository import of `flash_attn` was found. AGREE has no separate route; DINOv3 is loaded through `AutoModel` without an explicit flash implementation. Evaluator-created workers/subprocesses inherit the environment. Once import isolation is repaired, I found no alternate route to the real package.

## R3 — E1 exact control equality: RESOLVED

The fields at [plan:80](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:80) are necessary and sufficient in combination with normal `verify_cell`:

- Waveform SHA plus shape/dtype/n prevents numerical substitution.
- Stream-sidecar, input and assignment hashes bind the sample sequence.
- Type-strict `metrics` and `by_scene` comparisons cover reported results.
- Loaded checkpoint hash/bytes bind the model.
- Receipt backend binds the corrected execution path.

`source_sha` and `eval_name` legitimately differ between pins/names. Ignoring them in E1 is safe because the campaign pin, expected H35 cell identity and record validation are checked elsewhere. `ckpt_path` legitimately remains the common farm link.

## R4 — preregistration: RESOLVED, with one precision nit

E2’s CSV rows and all 14 thresholds match the pinned A6 CSV. Tier 2-C4’s six matched pairs, matched-offset controls, pooled rule, factor three, `1e-2` cap, alarm and exact duplicates are frozen and data-independent.

Tier 3 correctly contains 28 checks: two K values × seven metrics × `pass_c4`/`pass_r`, including FD.

The exp_25 JSON contains:

- K1: `0.9222999999999999, 0.1331, 6.498399999999997, 0.005900000000000016, 0.6470000000000002, 0.7101000000000006, 1.0731000000000002`
- K8: `0.9032999999999998, 0.13369999999999993, 6.806699999999999, 0.007400000000000018, 0.5522999999999998, 0.7259000000000011, 1.088799999999999`

The displayed values at [plan:118](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:118) are correctly rounded. Implementation must read the full values from the hash-pinned JSON, not copy the rounded prose constants.

E3, E4 and E5 clearly distinguish verdicts, expectations and descriptive analyses.

## R5 — collector/regression guards: RESOLVED

The fail-closed list is complete for the planned reuse:

- Exact external artifact hash, schema, pin and known partial state.
- Finite/non-null required fields and expected sample counts.
- External-reference namespace.
- One H35 campaign digest.
- Full input/assignment stream registry.
- CSV isolation to E2/figure.
- E1 gate.

The byte-for-byte exp_25 re-collect command is an appropriate regression guard. The external exp_25 R/45 cells remain pin-verified evidence rather than H35 cells.

## R6 — probe and figure provenance: PARTIAL

The scientific B-F probe contract is sound: first 64 K8 unseen items, fixed checkpoint/config/seed/noise, FA without canonicalization, three C4 offsets, fp32/eval protocols and a determinism floor.

What is missing is backend binding equivalent to evaluator cells. “Overlay sentinel present” at [plan:161](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:161) does not define its launch mechanism or require backend fields in the probe artifact.

Concrete fix: launch the probe through the same corrected import-isolation launcher and record/validate `attention_backend`, overlay SHA, launcher/campaign digest, log path and log SHA in `orbitrir_bf_cond_probe.json`.

For the figure, split the ambiguous field into:

- `source_artifact_path`
- `source_artifact_sha256`
- `source_row_key`
- `provenance_basis` or `orbit_execution_basis`

The “A6 batched, inferred from run date after 1479304” claim should remain explicitly labelled as an inference.

## R7 — GPU plan: NOT RESOLVED

The numerical planning checks out:

- Worst completed aligned 500-step window through s43 step 4,000: **9.712 s/step**.
- Mean through step 4,000: `34083 / 4000 = 8.52075`, correctly capped at **8.521**.
- Worst aligned 100-step window: **9.950 s/step**, ending at step 3,400.
- ETA arithmetic gives about **1,002 minutes = 16.7 hours**; “≈16.5 h (14–21 h)” is reasonable.

The blocking defect is identity enforcement. The current lane gate accepts any occupant when `ALLOW_SHARED_GPU=1` ([runner:344](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh:344)). Thus:

- A new unrelated process can replace/join the s43 trainer and still be accepted.
- Once s43 ends and its table job starts on GPU 1, the next chunk can race and co-tenant with that table job, contrary to [plan:203](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:203).
- The fixed reservation names GPU 1 only; the GPU 0 handover is not an atomic reservation transfer.
- The launch template retains `ALLOW_SHARED_GPU=1`; the GPU 0 path must explicitly remove it.

Concrete fix: before every spawn, require the foreign-occupant set to equal an approved s43 identity tuple—GPU UUID, PID plus process start time, cwd and command/log identity. Any disappearance or change triggers handover, not another shared spawn. Atomically move the reservation to GPU 0, require zero foreign compute processes there, and launch with `ALLOW_SHARED_GPU` unset. Test trainer exit/table-job replacement, extra PID, PID reuse and reservation-transfer races.

The guard also needs persistent baseline state defining “our first spawn” and a reviewed supervisor/loop that runs the parser every ≤5 minutes and refuses the next chunk on any nonzero result.

## R8 — checklist and TDD: PARTIAL

The rounds are implementable after incorporating the fixes above.

Critical before phase (i):

- Rounds 1–4.
- Round 5a.
- Round 5b: Tier 2/Tier 3 implementation should be frozen and reviewed before observing phase-(i) data, even if final rendering occurs later.
- The guard half of Round 6.
- All validation-ladder steps through integrative review.

The phase-(ii) figure half of Round 6 can follow while phase (i) runs. Splitting figure and guard into separate reviewed commits would clarify that critical path.

Add checklist/tests for:

- Script-directory and `.pyc` shadow attacks.
- Probe backend provenance.
- Exact approved co-tenant identity.
- GPU 1→0 reservation handover and removal of shared mode.
- Explicit confirmation that the exp_22 watcher remains absent/suspended.
- Guard baseline persistence, log truncation/rotation, epoch boundaries and co-tenant replacement.

## R9 — launch decision: NOT RESOLVED

Phase (i) should **not launch from plan r1**. The blockers are implementation-contract deficiencies, not unanswered scientific choices:

1. Make no-flash execution genuinely fail-closed for every evaluator and probe process.
2. Restrict shared execution to the exact approved s43 trainer and make the free-GPU handover atomic.

Yixun does not need to be asked another decision. These fixes implement decisions he already approved. A focused r2/code review can clear the launch.

## Ranked findings

### BLOCKING

1. **No-flash import isolation is bypassable** through script-directory precedence or ignored bytecode; the receipt does not yet prove actual dispatch fail-closed.
2. **GPU co-tenancy is not identity-bound** and can silently continue beside the s43 table job or another process.

### HIGH

3. **The B-F probe lacks the runner’s backend/overlay provenance contract.**
4. **Guard baseline and chunk-supervisor state are underspecified**, especially across training exit and GPU handover.
5. **The proposed overlay is not actually excluded from setuptools namespace discovery.**
6. **Figure `source_sha256` conflates an artifact digest with a row locator.**

### NIT

7. The Tier 3 prose constants are rounded; tests and computation must use full values directly from the pinned exp_25 JSON.

## What I verified

Read-only probes only:

- Confirmed branch `localization-exp` and HEAD `52305fb8660ff50f78cf3fc6ef52db9f93f82ef8`.
- Read plan r1, r0 review, Yixun’s seven YES decisions, relevant runner/collector/evaluator/model code, exp_22 records and packaging configuration.
- Recomputed the exact overlay content SHA-256: `0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023`.
- Verified A6 CSV SHA `0ae554c0…23da`, exp_25 JSON SHA `ec3fcd08…b6e0`, its schema/pin/partial/missing/invalid fields, E2 rows, CTL receipt/stream/dump hashes and full P1 Δ_C4 values.
- Confirmed the production diff from `d6ca307a` is limited to `src/localization/*`.
- Parsed the exp_30 s43 log using Python’s standard library only; verified 9.712, 8.52075 and 9.950.
- Probed Python path order with `PYTHONDONTWRITEBYTECODE=1`; the script/current directory precedes the overlay supplied through `PYTHONPATH`.
- Searched the evaluator stack for other `flash_attn` imports and subprocess routes.
- Ran no GPU command, torch import, evaluation, training, runner, collector, probe, pytest, package operation or process operation. Modified no files.
