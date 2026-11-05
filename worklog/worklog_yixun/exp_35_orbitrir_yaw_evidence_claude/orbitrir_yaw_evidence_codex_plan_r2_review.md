**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan r2 (exp_35 orbitrir_yaw_evidence: plan r2 at 682df5db)

**Verdict:** REVISE

§11 substantially repairs r1, but phase (i) remains blocked. The approved co-tenant is now identity-bound, yet the GPU transition is still a time-of-check/time-of-use protocol rather than an atomic hand-over. The supervisor’s durable-state and probe-chunk contracts also need tightening.

## Per-item disposition

| r1 finding | Status | Assessment |
|---|---|---|
| 1. No-flash isolation | **PARTIAL** | The top-level launcher closes script-directory precedence, overlay bytecode, repo-root shadowing, inherited `PYTHONPATH`, wrong origin and wrong transformer dispatch ([plan:270–275](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:270)). `runpy.run_path()` on a file does not prepend the target directory. A forked DataLoader worker inherits the asserted modules/path; multiprocessing `spawn` copies the parent’s explicit `sys.path` before re-running the main file, so those two worker modes are safe in principle. However, an independently exec’d Python subprocess does **not** inherit an in-memory `sys.path`; §11 removes inherited `PYTHONPATH` without exporting a controlled child path or requiring every Python child to use the launcher. The current pinned evaluator/probe only exec `git`, so no present Python-subprocess bypass was found, but the binding contract needs to forbid such children or route/test them, plus test both DataLoader `fork` and `spawn`. |
| 2. Identity-bound co-tenancy / hand-over | **PARTIAL** | Exact foreign-set equality and PID-reuse protection are good ([plan:277–278](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:277)). They do not make the launch atomic: the trainer can exit after the supervisor check but before the runner’s evaluator exec; its table job can replace it or start while a cell is running. Likewise, two GPU-0 observations 60 seconds apart do not reserve GPU 0 against a job starting immediately afterward ([plan:279](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:279)). |
| 3. Probe provenance | **PARTIAL** | All requested fields and collector validation are now required ([plan:284](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:284)). But chunk 5 is a probe, not a manifest cell, while the supervisor is specified to launch all chunks through `runner --only <one cell>` ([plan:280](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:280)). The contract also does not define who adds the SHA of the final, closed probe log to the JSON and regenerates its sidecar. |
| 4. Guard / supervisor state | **PARTIAL** | A persistent baseline, chunk ledger and reservation holder are added, with ≤5-minute polling and named post-chunk gates ([plan:280–281](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:280)). Missing are atomic state writes, a persisted stop latch, in-flight/gate-complete states, restart reconciliation, and defined fail-closed behavior for log truncation/rotation. Listing tests does not specify whether rotation must be refused or how continuity is proven. |
| 5. Packaging exclusion | **RESOLVED** | `exclude = ["overlays*"]` expressly removes the namespace tree from setuptools discovery ([plan:274](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:274), [plan:286](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:286)). |
| 6. Figure provenance fields | **RESOLVED** | Artifact path, artifact SHA, row key and execution-basis fields are separated; the A6 inference remains explicitly labelled ([plan:288](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:288)). |
| 7. Full-precision exp_25 values | **RESOLVED** | Computation and tests must read all values directly from the hash-pinned JSON; prose values are explicitly non-computational ([plan:290](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:290)). |
| R8. Critical path | **PARTIAL** | Round 5b and round 6a are correctly moved before observation of phase-(i) data, and the figure is separated ([plan:292](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:292)). The path is not executable fail-closed until the probe’s non-cell launch, supervisor recovery semantics, child-import tests and genuinely atomic GPU acquisition are specified. |

## Ranked findings

### BLOCKING 1 — the GPU hand-over is not atomic

“Checked twice, 60 s apart” establishes quiescence, not ownership. It does not close any of these races:

- The s43 trainer exits between identity verification and the runner’s actual evaluator exec.
- The follow-on table job starts on GPU 1 while the current shared cell is still running. The supervisor polls the training guard, not foreign-process identity, and promises not to signal the running cell.
- The table job selects GPU 0 rather than the assumed GPU 1 after the second free check.
- Another launcher claims GPU 0 after the second check but before or after the reservation write.
- The GPU-0 reservation already belongs to someone else; §11 says “writes” rather than atomically creates and verifies ownership.
- A crash/restart leaves both reservation files absent or state inconsistent. The “both never absent” test is restricted to “while a cell runs,” although the dangerous interval is also between chunks and during acquisition.

The GPU-1→0 transition needs one exclusion protocol honored by all relevant launchers, including the s43 follow-on table launcher. If that launcher cannot honor the reservation, the supervisor must not claim that table co-tenancy is impossible. At minimum:

1. Pass the approved tuple into the runner and enforce it in the runner’s final pre-exec lane check—not only in the outer supervisor.
2. Bind the tuple’s “log path” mechanically, such as through the process cmdline or `/proc/<pid>/fd/{1,2}`, rather than treating it as an asserted string.
3. Atomically acquire GPU 0 without overwriting another holder; require the owned GPU-1 reservation to exist; persist the transfer; only then release GPU 1.
4. Refuse inconsistent restart states, including neither reservation present.
5. Prove the table job is committed to GPU 1, or make its launcher honor the reservation. A 60-second sample cannot provide this guarantee.

### BLOCKING 2 — supervisor persistence is not yet a fail-closed state machine

The state file needs a binding schema and atomic lifecycle. It should persist at least the campaign/H35 identity, baseline and source-log identity, `planned → in_flight → runner_verified → gates_verified` status per chunk, lane/reservation epoch, and a permanent stop reason. Writes must be atomic; corrupt or inconsistent state must refuse resume.

On restart, a completed receipt must not cause the supervisor to skip E1, E2, the probe gate, or the post-chunk guard. Nonzero runner status, absent receipt, failed verification, a final guard failure, or an invalid probe must be latched before any next spawn. Log truncation/rotation must either prove continuity or hard-stop; resetting the baseline is not acceptable.

### BLOCKING 3 — chunk 5 has no defined launch and finalization transaction

The 33 chunks comprise 32 evaluator cells plus one standalone probe ([plan:184–190](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/plan_orbitrir_yaw_evidence.md:184)). Therefore `runner --only <one cell>` cannot launch chunk 5.

The contract should explicitly branch:

- Chunks 1–4 and 6–33: runner, one manifest cell.
- Chunk 5: the same no-flash launcher directly invokes the exact probe target/argv under the same guard, identity and reservation gates.
- After the probe process closes its raw log, a non-logged finalization step hashes that final log, writes the campaign/backend/overlay/launcher/log provenance into the JSON, regenerates `.sha256`, and runs `PROBE_CONTRACT_BF`.
- Only successful finalization and validation mark chunk 5 gate-complete.

### HIGH 4 — child-process import isolation is not fully contracted

The launcher is fail-closed for the parent. Fork and normal multiprocessing spawn preserve the explicit path, but a fresh `subprocess` Python interpreter reconstructs its path from argv/environment and would not inherit the launcher’s in-memory `sys.path`.

The implementation contract should require one of:

- no independently exec’d Python children in the pinned evaluator/probe tree;
- every such child runs through the launcher; or
- a controlled child environment plus a child-side origin/backend assertion.

Add tests that force DataLoader `fork` and `spawn`, with each worker reporting the transformer path and `flash_attn_func is None`, and a fresh-child test proving that the real installed package cannot win.

## Launch decision

Phase (i) must not launch from plan r2 as written. No new scientific decision from Yixun is required; these are implementation-contract corrections. Once the atomic ownership protocol, supervisor recovery/gates, and standalone probe transaction are added, the remaining r1 items are adequately specified.

## What I verified

- Reviewed the plan and r1 review explicitly from commit `682df5db`; plan SHA-256 was `c88825e94159164952e9a0af2d7f957e3747f325db3a156832b23ab484f10ba6`.
- The shared checkout advanced during review, so all artifact conclusions were pinned with `git show 682df5db`; the reviewed plan artifacts and relevant evaluator/probe files had no diff from that commit.
- Inspected Python’s standard-library `runpy` and multiprocessing preparation logic with `PYTHONDONTWRITEBYTECODE=1`: file-based `run_path` does not prepend the target directory, and multiprocessing spawn copies the parent `sys.path` before re-running the main file.
- At `682df5db`, the evaluator and probe create multi-worker DataLoaders; their only explicit subprocesses invoke `git`, not Python.
- Did not inspect the exp-35 implementation worktree. Ran no GPU command, runner, collector, probe, evaluation, pytest, package operation or process operation; modified no files.
