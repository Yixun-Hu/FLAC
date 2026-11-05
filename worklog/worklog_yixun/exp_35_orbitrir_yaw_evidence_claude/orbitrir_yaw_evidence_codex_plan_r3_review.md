**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan r3 (exp_35 orbitrir_yaw_evidence: plan r3 at 2cf63019)

**Verdict:** REVISE

§12 resolves the supervisor, probe, and child-import findings substantially. One launch blocker remains: the timing predictor reduces risk but does not enforce the frozen requirement that a shared exp_35 cell never overlap the exp_30 evaluation FIFO.

## Per-item status

| r2 finding | Status | Assessment |
|---|---|---|
| **B1 — GPU hand-over** | **PARTIAL** | The final runner-side lane check, mechanical `/proc` identity binding, `O_EXCL` reservation acquisition, persisted transfer epoch, ordered GPU-1→GPU-0 transfer, and fail-closed restart states resolve the earlier reservation defects ([plan:299](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:299), [plan:303](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:303), [plan:304](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:304)). The remaining gap is the GPU-1 queue: its launcher does not honor the reservation, and the predictor is not an exclusion mechanism. The plan also needs launch evidence that the earlier s44 FIFO item is durably finished; otherwise it can start another table cell during s43 independently of the s43-end predictor. |
| **B2 — supervisor persistence** | **RESOLVED** | The schema now binds campaign/code/cells/log state, records all chunk lifecycle states, uses atomic writes, permanently latches failures, re-runs unfinished gates, and refuses ordinary log replacement or shrinkage ([plan:306](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:306)). Remaining durability hardening is a NIT below. |
| **B3 — probe transaction** | **RESOLVED** | Chunk 5 now has a distinct launcher path, frozen argv, the same admission gates, close-before-hash ordering, non-logged provenance finalization, sidecar regeneration, and validation before `gates_verified` ([plan:308](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:308)). |
| **H4 — child imports** | **RESOLVED** | §12 contracts the observed absence of Python subprocesses, tests DataLoader `fork` and `spawn`, and exports a controlled overlay-only `PYTHONPATH` with a fresh-interpreter test ([plan:310](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:310)). |

## Ranked findings

### BLOCKING 1 — the end predictor is not queue exclusion

The predictor at [plan:302](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:302) is a sensible safety margin, but it cannot substantiate “the lane never co-tenants with his table job” from [plan:203](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:203):

- `4.208 s/step` is the fastest observed window, not an enforced lower bound on future step time.
- The listed cell durations are planning estimates, not runtime deadlines.
- The exp_30 evaluator tests memory before every table cell and has explicitly “No exclusivity” ([queue:5](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh:5), [eval:115](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_eval.sh:115)). If training ends while exp_35 is still running and enough memory remains, the table cell may start.
- The same FIFO contains the preceding s44 evaluation. A momentary absence of an s44 compute process between its cells does not prove that FIFO item is complete.

Before shared launch, the plan must therefore do one of the following:

1. Use an interlock honored by the exp_30 evaluator, and verify the earlier s44 FIFO item is durably complete; or
2. Wait for s43 to finish and hand over to GPU 0 before launching phase (i); or
3. Obtain an explicit revision of the no-table-co-tenancy requirement accepting this forecast risk.

The arbitrary GPU-0 race already disclosed at [plan:304](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:304)—an unrelated session starting after the final check—is an acceptable residual risk. Unlike the exp_30 FIFO, it is not a known scheduled launcher that this plan claims to exclude.

### NIT 1 — strengthen state durability

After `rename`, fsync the state directory as well as the temporary file. Persist a content anchor or parsed offset/hash for the training log so copy-truncate-and-regrow between observations cannot evade the inode/size tests.

### NIT 2 — spell out chunk 5’s ledger fields

The generic schema names a runner receipt SHA, but chunk 5 has no runner receipt. Define its corresponding fields explicitly—probe process rc, JSON SHA, sidecar SHA, final log SHA, and `PROBE_CONTRACT_BF` result.

### NIT 3 — freeze the external queue evidence

Record the queue/evaluator/admission-helper SHAs in the launch entry. The cited queue file currently has SHA-256 `9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa` and is modified relative to its checkout HEAD; the only diff is the `TRAIN_JOBS` array.

## Launch answers

**(a)** Yes. The GPU-1 FIFO exclusion above remains the sole blocker once §11 and §12 are faithfully implemented. B2, B3, and H4 no longer block phase (i).

**(b)** No—not as a guarantee. `4.208 s/step` plus twice the planning duration is a conservative operational heuristic and makes overlap unlikely, but the queue’s admission rule is memory-only and does not honor the reservation. It is acceptable only as a disclosed, explicitly accepted risk; it does not satisfy the currently frozen “never co-tenants” requirement by itself.

## What I verified

- FLAC is on `localization-exp` at `2cf630199494748a7715be88d5428a707cfd1fe1`; the reviewed plan matches HEAD and has SHA-256 `f59a37c153e74bf542b7fbc2e68a8fc58e660145b566d6a9943fb5e2c331702f`.
- The cited current queue file says:
  - `TRAIN_JOBS=('ACEattn 44' 'ACEattn 43')`, so s43 is last ([queue:239](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh:239)).
  - Training uses GPUs 0 and 1 ([queue:123](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh:123)).
  - The evaluation FIFO invokes GPU 1 ([queue:66](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh:66)).
  - Admission is based on free memory and expressly provides no exclusivity ([queue:5](/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh:5)).
- The queue appends each evaluation only after its training succeeds, but the evaluator performs memory admission before every table cell; after the queue closes, its threshold falls from `T+E` to `E`.
- I did not inspect the exp-35 implementation worktree. I ran no GPU command, runner, evaluator, collector, probe, pytest, package operation, or process operation and modified no files.
