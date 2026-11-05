**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C3d (exp_35: the C3c fixes; b15d8a82 → 1f31c7a7)

**Verdict:** REQUEST-CHANGES — C3c’s BLOCKING finding is resolved, but three HIGH correctness gaps remain. `1f31c7a7` is not ready to become H35.

## C1–C9

### C1 — RESOLVED

The state schema now requires the evidence its status claims:

- `gates_verified` requires an exact `final_guard` with `status == "OK"`, `exit == 0`, non-negative timestamp/step, and an anchor accepted by the real guard schema: `exp35_supervisor.py:161-201`.
- The record is populated from the final `observe()` result, after the gate: `exp35_supervisor.py:727-735`.
- A latched completed failure in `in_flight` permits exactly `{ended, rc}`; all other extra fields are refused: `exp35_supervisor.py:172-179`.
- Baseline log path, inode and size, reservation directory/path, pins, and manifest/self hashes are validated: `exp35_supervisor.py:83-95, 204-266, 941-950`.
- Old states lacking `final_guard` are refused, rather than upgraded.

A partial/corrupt write is rejected, and merely flipping `runner_verified` to `gates_verified` no longer works. A deliberately fabricated, fully schema-valid state remains possible because the state is not cryptographically authenticated; that is outside the structural state contract reviewed in C3c.

### C2 — PARTIAL / NOT RESOLVED

Ordinary execution now re-authenticates immediately before the spawn path, and each watch poll re-authenticates before examining co-tenants: `exp35_supervisor.py:615-621, 647-650`. A detected in-flight loss latches without signalling the child.

However:

- `ready()` and the 60-second hand-over do not authenticate the currently held reservation.
- `handover()` creates GPU 0’s file, changes state, then unconditionally unlinks the old GPU 1 path: `exp35_supervisor.py:565-593`.
- Final release also unconditionally unlinks the state’s path: `exp35_supervisor.py:781-783`.
- The check at `:615` precedes the state fsync at `:619-620`, leaving a non-trivial check-to-spawn interval.

Therefore a reservation replaced during hand-over can be erased as though it were still ours, after which a GPU 0 cell can spawn. See HIGH 1.

### C3 — RESOLVED

For the supervisor dry run:

- `sys.dont_write_bytecode` is set before any local dynamic import: `exp35_supervisor.py:34, 424-437`.
- It validates exactly two GPU cards, uses the state’s current lane, requires the full tuple plus predictor only on shared GPU 1, and requires an empty GPU 0 lane without invoking the predictor: `exp35_supervisor.py:877-920`.
- Hidden `--proc` is threaded into both identity and supervisor configuration: `exp35_supervisor.py:935-950`.
- The fresh subprocess test and ladder 2 ran with `PYTHONDONTWRITEBYTECODE` unset and no `-B`; all 5009 pre-existing entries were unchanged.

The runner’s separate `DRY_RUN=1` did write ignored bytecode through collector imports. That is disclosed by ladder 3b and cleaned up; it does not refute the supervisor dry-run fix.

### C4 — NOT RESOLVED

Complete approved identities are compared, but `replaced()` returns false when the identity of a PID just reported by `nvidia-smi` cannot be read: `exp35_supervisor.py:652-665`.

That is not safely equivalent to “gone.” A process that actually exited between the two observations is harmless, but a still-listed foreign process whose `/proc` identity is persistently inaccessible is accepted at every poll. Such a process can co-tenant an entire cell undetected. See HIGH 2.

### C5 — RESOLVED

After finalizer rc 0, the supervisor now requires:

- Three 64-hex hashes.
- A sidecar sealing the recorded JSON hash.
- A launch record naming this log, its hash, and this campaign digest.

See `exp35_supervisor.py:667-685, 716-724`.

In the production path, the probe writes its report and sidecar before returning success; the finalizer refuses already-finalized input, checks the closed log, and atomically rewrites both JSON and sidecar: `depth_canon_collect.py:1256-1285`. Thus an earlier finalized report cannot silently pass as the current attempt. There is, however, a restart/idempotence problem after successful finalization; see HIGH 3.

### C6 — PARTIAL

Confirmed:

- The supervisor never signals the runner.
- A stop latch is never cleared by supervisor code.
- GPU 0 commands omit `ALLOW_SHARED_GPU` and tuple variables: `exp35_supervisor.py:452-462`.
- GPU 0 still requires two free checks 60 seconds apart.
- A `runner_verified` chunk re-runs its gates.
- An old state without `final_guard` is refused.

The transaction is not fully sound because hand-over/removal does not authenticate the reservation, and a crash between probe-ledger persistence and the final guard cannot successfully reconcile.

### C7 — PASS, qualified

#### (a) Exp_25 byte guard

The equivalent-environment clone is an acceptable discharge of §5 for code compatibility. With the same two external path conditions as the exp_25 worktree, the current collector reproduced the registered JSON and Markdown byte-for-byte. The run-worktree failure is explained by two environment-dependent inputs:

1. CANON provenance compares resolved paths at `depth_canon_collect.py:1038-1040`. Because the recorded `model_config` is absolute, `os.path.join(REPO, absolute_path)` preserves the exp_25 path. A byte-identical symlink makes the realpaths agree.

2. Tie-audit loading calls `validate_tie_audit(path, REPO)` at `depth_canon_collect.py:1496-1508`. The audit resolves relative roots such as `AcousticRooms` against the current repository in `depth_canon_tie_audit.py:150-183, 193-217`, then rehashes the map bytes in `depth_canon_audit_gate.py:236-244, 306+`.

Without the asset link, the baseline Markdown difference consists of:

- The composition-control path mismatch being added to the PARTIAL reasons.
- The tie audit reporting missing/unreadable depth maps rather than 112 validated maps.
- Tier 2 reporting no registered composition control, so `eps_cond` and its pair/pooled details are unavailable.
- The final tie summary having empty/unknown per-split and shared-map values.

The scientific cell metrics, Tier 1, and Tier 3 values were otherwise unchanged.

#### (b) Exp_35 reachability

Those two exp_25-specific path dependencies do not reach:

- E1: `depth_canon_collect.py:1800-1832`.
- E2: `depth_canon_collect.py:1835-1867`.
- BF probe finalization/contract: `depth_canon_collect.py:1160-1225, 1256-1285`.
- `--verify-cell`: `depth_canon_collect.py:2435-2450`.
- Final exp_35 collection/Tier 3: `depth_canon_collect.py:2153-2183`.

Actual evaluation still needs the planned `AcousticRooms → NAS` and weights/checkpoint symlinks. Creating and recording those links before launch is sufficient, provided the NAS manifest/dumps, checkpoint bytes, DINO/FLAC/VAE weights, farm/cache writability, interpreter, and external exp_25 artifacts are also validated as required by §7–§8.

The runner dry run does not read the raw `AcousticRooms` maps, and these links are outside `RELEVANT`; rerunning it solely because the links were added would not test their correctness. Record and check their resolved targets separately.

### C8 — PARTIAL

The RED/GREEN evidence is credible and deletion-sensitive for the implemented fixes:

- Initial targeted RED failures cover the new schema, reservation, PID-reuse, ledger, and dry-run assertions.
- Final supervisor suites passed.
- Ladder 4 reports 1462 passed, 1 skipped, 0 failed.
- The hidden `/proc` tests use the real identity helper against a fake tree, while ladder 2 supplies the live-system check.
- Probe-ledger mutation tests establish the supervisor’s envelope; separate collector tests establish the production probe contract.

Coverage omissions correspond directly to the findings:

- No mutation of the held reservation during the 60-second hand-over or immediately before final release.
- No test where a PID remains in the `nvidia-smi` table while its `/proc` identity is unreadable.
- No restart from probe chunk 5 after the production finalizer has succeeded but before the final guard. `FakeGates.finalize()` rewrites an already-finalized document unconditionally (`test_exp35_supervisor_lanes.py:145-158`), unlike the production finalizer.

### C9 — FAIL / NOT READY

`1f31c7a7` should not become H35. After the findings are fixed, use the successor commit for the integrative branch review and rerun the authoritative ladder at that exact SHA.

The final ladder should include all four rungs. Running the runner dry run after asset-link creation is useful as final-environment evidence, but it does not replace a direct read-only validation of those links.

Beyond §8, §13 NIT 3, and the C3c launch list, record:

- The fail-closed policy for a GPU-listed but unreadable PID.
- Authenticated reservation transfer and authenticated removal/release.
- The probe-finalization restart rule and its recovery test.
- Resolved asset-link targets and relevant asset/checkpoint hashes.
- Final ladder transcript hashes, including disclosure/cleanup of runner-created bytecode.

No new scientific decision rule needs pre-registration; these are operational integrity rules.

## Ranked findings

### HIGH 1 — Reservation transfer and removal can act on a file no longer owned by this campaign

`exp35_supervisor.py:529-593, 781-783`

A reservation can be deleted/replaced after startup or the preceding watch. Hand-over never validates the old record before committing the new lane and unlinking the old path. Final release has the same issue.

Concrete fix:

- Authenticate the held reservation before each `ready()` decision, before and after the hand-over wait, immediately before committing the transfer, and immediately before every removal.
- Remove only a record whose complete fields, epoch, and campaign still equal the expected held record.
- On mismatch, latch and do not unlink anything.
- Re-authenticate after the in-flight state save as the last pre-spawn operation, or use an ownership mechanism that closes the check-to-spawn race.

Add tests that replace/delete/corrupt/change the epoch during the hand-over gap and immediately before final release, asserting no further spawn and no foreign record deletion.

### HIGH 2 — A GPU-listed PID with unreadable identity is accepted as the approved trainer

`exp35_supervisor.py:652-665`

`replaced()` conflates “PID disappeared” with “identity could not be read.” That permits a persistently unreadable, unapproved process to share the lane indefinitely.

Concrete fix:

- Make identity lookup tri-state.
- If a PID was reported on the lane and its complete identity cannot be read, re-query the compute table: treat it as gone only if it is no longer listed; otherwise latch.
- Never classify a still-listed unreadable PID as approved.

Add a test that leaves `(TR1, GPU1)` in the fake `nvidia-smi` table while deleting or corrupting its fake `/proc/TR1` identity. The cell may finish, but the latch must prevent the next spawn.

### HIGH 3 — A valid finalized probe cannot survive the documented `runner_verified` restart path

`exp35_supervisor.py:716-727, 737-748`; `depth_canon_collect.py:1262-1267`

There is a persisted crash point after finalization and ledger save, but before the final guard. On restart the chunk is still `runner_verified`, so reconciliation invokes `gates()` again. The production finalizer then refuses because `provenance.launch` already exists, permanently latching an otherwise valid probe.

Concrete fix:

- Make finalization idempotent only when the existing launch, sidecar, current log hash, campaign digest, overlay and launcher hashes all match exactly; or
- Represent “finalized/ledger verified, final guard pending” explicitly and rerun the contract plus final guard without trying to finalize again.

Add a production-faithful restart test at probe chunk 5 from the persisted post-finalizer/pre-final-guard state.

## What I verified

- Read the binding plan r4 through §14 A1–A2 and prior C1/C2/C3a/C3b/C3c reviews.
- Reviewed committed content with `git show`/`git diff`; no project command was executed.
- Both commits have the Opus 5.5 trailer.
- `git diff --check b15d8a82 1f31c7a7 -- '*.py'` is clean.
- Code count for `dce4f27a`: 139 production and 217 test lines; +102/−37 code and +209/−8 tests.
- `1f31c7a7` changes no `.py`, `.sh`, or `src/**` content relative to `dce4f27a`.
- Confirmed unchanged hashes:
  - Overlay: `0d0e52ca…c023`
  - Launcher: `bbd2847d…43de`
  - Manifest: `440eb1be…037d`, 346 rows
- Verified all four supplied untracked ladder hashes before reading them; all matched.
- Read the RED/GREEN records, final suite transcript, live supervisor dry run, runner dry run/cleanup, and equivalent-environment byte guard.
- Final `git status` had no tracked modifications; it showed only untracked ladder/dry-run material and an untracked `AcousticRooms` path. I did not modify or rely on that path.
- I did not run pytest, the runner, supervisor, guard, probe, collector, training/evaluation, `nvidia-smi`, or any GPU/process-attaching command.
