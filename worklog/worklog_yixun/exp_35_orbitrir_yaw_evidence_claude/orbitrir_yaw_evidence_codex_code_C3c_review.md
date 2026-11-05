**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C3c (exp_35: the C3b fixes, C3a HIGH 1, round 6a (iii)–(iv): lane ownership, the chunk transaction, the World adapter and dry run; 8efdaf5a → b15d8a82)

**Verdict:** REQUEST-CHANGES

The ordinary 33-chunk path is largely faithful, and R1/R2 are correct. However, one part of C3b BLOCKING 1 remains unresolved, and the reservation, live identity-watch, probe-finalization, and dry-run edges are not fail-closed enough for H35.

## Ranked findings

### BLOCKING 1 — `gates_verified` still has no evidence that the final guard passed

`GATE_KEYS` contains only E1, E2, and PROBE, while chunks without a named gate require no additional evidence when moved from `runner_verified` to `gates_verified`:

- `b15d8a82:worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor.py:112-185`
- The final guard runs at `:669`, but `:670-671` stores only `state="gates_verified"`.

Consequently, a valid chunk 6+ at `runner_verified` can have only its `state` changed to `gates_verified` and pass `state_reasons`. The global `his_log_anchor` cannot prove that the observation happened after that particular chunk. The existing fake-prefix mutation begins from bare planned chunks, so it does not catch this.

The “exact schema” is also incomplete: latched `in_flight` chunks may carry receipt fields without their values being validated (`:164-171`), baseline paths are merely non-empty, reservation paths are checked only by basename (`:191-205`), and `main` can supply `None` as a cells/supervisor hash (`:872-875`).

Concrete fix:

- Persist a per-chunk final-guard result containing an exact success schema, observation time/step and anchor, and require it for every `gates_verified` chunk.
- Enforce exact allowed key sets and types for latched `in_flight` states.
- Pass the expected log path and reservation directory into state validation.
- Refuse non-hex/missing cells and supervisor hashes before creating/loading state.
- Add mutations that flip a fully evidenced `runner_verified` chunk to `gates_verified`, alter a valid baseline path, use negative inode/size, and place a same-basename reservation path outside the configured directory.

### HIGH 1 — reservation ownership is checked only at supervisor startup

`reservation_reasons` is called once at `exp35_supervisor.py:702-705`. Neither `ready()` (`:509-543`) nor `watch()` (`:605-627`) revalidates the held reservation. `execute()` creates a file only when the in-memory state says no reservation exists (`:582-588`).

If the current file is deleted, replaced, or corrupted after startup, the next cell can run while both reservation files are absent or while another holder owns the lane. This answers C4’s final question: **yes, current code can leave no valid reservation while a cell runs after an external reservation mutation**.

Additionally, because state validation checks only the reservation path’s basename, a corrupt state can name an arbitrary same-basename path; hand-over later unlinks that path at `:571-573`.

Concrete fix:

- Require `reservation.path == reservation_path(configured_dir, uuid)`.
- Re-run exact reservation authentication immediately before every spawn and during each watch poll.
- A loss/change during a cell should latch after that cell without signalling it.
- Add deletion, replacement, wrong-epoch and wrong-path tests between chunks and during a running chunk.

### HIGH 2 — the supervisor dry run is not guaranteed read-only and checks the wrong lane after hand-over

`_module()` dynamically imports the identity and guard helpers at `exp35_supervisor.py:405-409`. `dry_run()` invokes both at `:816` without setting `sys.dont_write_bytecode`. In a clean writable checkout, those imports may create ignored `__pycache__/*.pyc` files.

The tests do not observe this: the supervisor is already imported before the snapshot, and `_snapshot` covers only `tmp_path`, not the run worktree (`test_exp35_supervisor_cli.py:19-20,110-119`).

There is a second dry-run defect: although `:825` reads the current lane from state, `:833-845` always checks GPU 1’s approved tuple and shared predictor. A legitimate resumed GPU-0 state can therefore be falsely refused, while GPU-0 occupancy is not checked as the current lane.

Concrete fix:

- Set `sys.dont_write_bytecode = True` before any dynamic import.
- Test the CLI in a fresh committed snapshot with `PYTHONDONTWRITEBYTECODE` absent and compare the entire snapshot, including ignored files.
- Branch the live check by state lane: exact approved tuple and predictor on shared GPU 1; zero foreign processes and no predictor on GPU 0.
- Enforce the exact two-card map in dry-run as in `Supervisor.tables()`.

### HIGH 3 — the running-cell watch loses PID-reuse protection

At `exp35_supervisor.py:608-609`, the approved tuple is reduced to a set of PID numbers. At `:623-626`, any process with an approved PID is exempt without rechecking its start time, cwd, cmdline hash or fd links.

If the trainer exits and a table job reuses PID 3028196 during a cell, it is not classified as a new process and no stop latch is set. The next pre-spawn identity check prevents another shared spawn, but §13 requires the replacement to latch and be reported after the running cell.

Concrete fix: while polling, compare every non-descendant process with the complete approved identity. A missing trainer alone may lead to hand-over; an extra or same-PID/different-identity process must latch. Add an in-flight PID-reuse test.

### HIGH 4 — successful probe finalization can continue with missing ledger artifacts

After the finalizer returns, chunk 5 records three hashes at `exp35_supervisor.py:659-663`, but `bad` depends only on the finalizer return code. If a file is missing or unreadable, `_sha_file` returns `None`; the supervisor can still mark the probe `gates_verified` at `:669-671` and spawn chunk 6. Restart validation would later reject the state, but that is too late.

Concrete fix: require all three hashes to be 64-hex and consistent with the finalized probe provenance before marking the gate verified; otherwise latch. Add a finalizer-rc-0 test with a missing/deleted JSON, sidecar, and log.

## Per-item status

| Item | Status | Assessment |
|---|---|---|
| **C1 — C3b findings** | **PARTIAL** | BLOCKING 1 remains partial because final-guard evidence and several exact-state constraints are missing. HIGH 1 is **RESOLVED**: only `FileNotFoundError` means absence, with PermissionError/EIO tests. HIGH 2 is **RESOLVED for reservation-file field authentication**, but live ownership has HIGH 1 above. NIT 1 is **RESOLVED**: malformed anchors return REFUSED. |
| **C2 — C3a HIGH 1 / R1** | **PASS / RESOLVED** | R1 is the correct reconciliation. The config pin check follows the selected plan, while an exp_25-only digest universe contains CANON/P1/YAW configs and excludes B-F. Any exp_35 selection uses all four arm configs, preserving one digest across P1 and B-F single-cell chunks. This matches §5 item 4. The relevant config *set* matches d6ca307a; the literal digest cannot equal d6ca307a because H35’s code and manifest differ. |
| **C3 — §14 A1 / R2** | **PASS** | R2 is correct: fallback is independently decided for each `(method, K)`. A missing OrbitRIR K8 contingency cell makes only OrbitRIR K8 fall back to its s42 point; other method/K points can remain n=5. Default exclusion and the test-only follow-up are correct. |
| **C4 — lane ownership** | **PARTIAL** | The full tuple is correctly bound through `/proc`, rehashed against the launch entry, and checked before shared starts; the predictor is exact and conservative for stale observations; hand-over order, O_EXCL, epoch increment, GPU-0 checks and unset `ALLOW_SHARED_GPU` are correct. A checked shared cell cannot intentionally start beside an unapproved process, and a checked GPU-0 cell cannot intentionally start beside an observed foreign process. HIGH 1 and HIGH 3 leave reservation continuity and same-PID in-flight replacement unresolved. |
| **C5 — chunk transaction** | **PARTIAL** | The one-cell runner commands, ≤5-minute polling, no-signalling rule, E1/E2 semantics, campaign-digest verification, restart reconciliation, probe invocation/finalization and permanent latch behavior are otherwise correct. BLOCKING 1 and HIGH 4 leave the persisted gate claim and one probe fail-closed edge incomplete. |
| **C6 — World, gates, dry-run, CLI** | **FAIL** | The real adapters are otherwise faithful: fail-closed `nvidia-smi`, directory-fd `/proc` reads, O_EXCL, file/directory fsync, repo-cwd spawn with closed stdin and combined output, and collector CLI construction. Components have real tests rather than a wholly replaced World. The advertised read-only dry run is false because of possible bytecode writes and is not lane-aware after hand-over. |
| **C7 — tests/evidence** | **PARTIAL** | The per-commit RED/GREEN evidence is credible and most mutation tests are deletion-sensitive. Missing tests correspond directly to the findings above. The dry-run read-only test passes because it snapshots the fake tmp tree, not helper-import writes in the checkout. The restart receipt test relies on `FakeGates`, though the production receipt adapter has separate tests. |
| **C8 — SOP** | **PASS** | Independent production/test counts are `19/26`, `0/3`, `149/210`, `132/196`, `17/27`, `164/121`, `138/122`; b15d8a82 is records-only. Every production count is below 200. All 8 commits carry the Opus 5.5 trailer; Python/shell diff-check is clean; no package/environment file changed. |
| **C9 — readiness** | **FAIL / NOT READY** | b15d8a82 should not become H35. The findings require a successor commit, followed by the complete validation ladder and integrative review at that successor SHA. A2/round 6b remains non-blocking for phase (i). |

## Test and evidence judgment

The 6145f93d all-suite failure is a clone-location artifact. Both `test_depth_canon_autolaunch.py` and `depth_canon_autolaunch.sh` are unchanged since d6ca307a. The test passes `REPO/train.py` as a supposedly outside-`/tmp` launcher, but in the scratch clone `REPO` itself was under `/tmp`; the harness correctly accepted it. That explains why the same test passes in the normal worktree.

The committed evidence shows:

- Targeted RED/GREEN for each of the seven code/test commits.
- `149 passed` for supervisor and guard files at 0b9e1a36.
- Earlier all-suite evidence: `1236 passed, 1 skipped, 1` location-dependent failure.
- Final CPU suites, exp_25 byte guard, runner DRY_RUN and live supervisor dry run remain pending. The existing supervisor dry run cannot serve as read-only evidence until HIGH 2 is fixed.

## Launch-entry additions

After the fixes, the frozen launch entry should additionally record:

- The approved tuple’s exact bytes, path, size and SHA-256, plus its decoded PID/start/cwd/cmdline/fd1/fd2/GPU UUID.
- Exact SHA-256 values for the supervisor, guard and identity helper—not merely H35.
- Initial state-file status: absent for a new campaign, or exact bytes/SHA and successful schema validation for a resume; stop latch absent.
- Reservation directory snapshot and the expected GPU-1 reservation path/schema: holder, H35, branch, UUID, epoch 1; hand-over’s GPU-0 epoch 2 rule.
- The exact supervisor command and environment, including state/log/reservation paths and shared/unshared environment rules.
- The live GPU index→UUID map, complete compute-process table, guard log inode/device/size/mtime/tail anchor/current step, and predictor calculation.
- The exact 33-chunk order, cells SHA, 116-row exp_35 subset SHA, campaign digest, and runner dry-run transcript hash.
- Exact queue/evaluator/admission-helper bytes and hashes, s44 completion evidence, and confirmation that the exp_22 watcher remains absent.
- Hashes of the final CPU-suite, exp_25 byte-for-byte collect, runner DRY_RUN and corrected read-only supervisor dry-run transcripts.

No new scientific decision needs pre-registration.

## What I verified

Read-only committed-object probes only:

- HEAD is `b15d8a82` on `exp-35-orbitrir-yaw`; the range contains exactly eight commits.
- 8/8 trailers; `git diff --check 8efdaf5a b15d8a82 -- '*.py' '*.sh'` is clean.
- Overlay `0d0e52ca…7c023`, launcher `bbd2847d…43de`, and manifest `440eb1be…e7037d` are unchanged from C3b.
- Manifest: 346 rows, 116 exp_35 rows; its first 230 rows are object-identical to d6ca307a.
- The evaluator closure since d6ca307a still changes only the six `src/localization/meshgrid_*` files.
- Independent AST-aware changed-code counts match the Coder’s reported figures.
- The autolaunch test/script are unchanged since d6ca307a, and the `/tmp` failure mechanism is confirmed from their committed code.
- No worktree source or worktree status was inspected. I ran no pytest, runner, supervisor, guard, probe, collector, evaluator, training, GPU command, package command or process-control command, and modified no file.
