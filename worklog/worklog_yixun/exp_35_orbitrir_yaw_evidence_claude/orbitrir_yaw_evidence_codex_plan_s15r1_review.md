**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-06 · **Round:** plan §15 r1 (exp_35: the phase-(ii) launch; ef4a446a)

**Verdict:** APPROVE

No BLOCKING or HIGH findings. The NIT clarifications and the S6 implementation checklist below are binding for the Coder/code review.

## S1 — PASS: scope and pins

The selected scope is exact and sufficient:

- H35’s manifest hashes to `440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d` and contains 346 rows.
- Orders 262–329 are exactly 68 unique rows: OFF-P1 20, OFF-BF 20, OFF-CANON 20, CANON-45 8.
- OFF blocks are 135°/225° × K1/K8 × s42–46. CANON-45 is 45° × K1/K8 × s43–46.
- Every selected row has `dump: true`; the manifest dump root is `/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms`.
- E2-FAIL is separately confined to orders 330–345 and is not selected. The phase-(i) state records E2 PASS on all 14 checks.
- P1, B-F, and CANON protocols are correct: vanilla/non-canonical P1, `fa_invariant` B-F, canonicalized CANON.
- H35 is the requested `5b065801e9d1c1844461ec84cd0558702c714326`. I independently recomputed the runner-relevant tree digest in the run worktree as `da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1`, matching phase-(i) receipts/state.
- The CANON manifest hashes to `1e4814472b72375d6f3f42b3773f7bdb5d137719f882d00aef0387b770218cba`; its step-40000 entry is exactly `52c64b525c8a34f1721b73abd5c4aad67295ea2f729426e9330104b6d1441ace`.
- `--canon-ckpt-manifest` reaches planning, receipt writing, cell verification, and collection. The runner verifies the source when staging and fully re-hashes the immutable cache before every spawn.
- The H35 collector requires the OFF/CANON-45 cells for a complete `--exp35` result and rejects E2-FAIL artifacts after E2 PASS.
- Round 6b pins H35, the cells-manifest hash, and the CANON-manifest hash; its figure registers exactly these 135°/225° and CANON-45 sources. E5 consumes the required B-F/CANON dump pairs.

## S2 — PASS: preconditions and ordering

The ordering matches Yixun’s instructions:

- Launch is gated on 33/33 phase-(i) chunks being `gates_verified`, no stop latch, the supervisor gone, and its GPU-1 reservation removed.
- The phase-(i) report must be sent before reservation/launch and must include measured slowdown, phase-(ii) per-cell timing, ETA, and card plan. If the new one-card ETA exceeds roughly 24 hours, that fact must be reported before proceeding.
- The default is one frozen lane on GPU 0. GPU 1 remains available to exp_22 only after the s43 table job.
- The live queue file still hashes to `9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa`; it has `TRAIN_JOBS=('ACEattn 44' 'ACEattn 43')`, trains on GPUs 0/1, and dispatches evaluation only to GPU 1. The evaluator hash is `94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c`.
- The complete lane list must be decided before the initial runner launch. The existing global runner lock does not permit dynamically adding a second runner/lane later.

The static probe found phase (i) still incomplete, as expected at review time; §15 correctly treats completion as a future hard precondition.

## S3 — PASS: admission and reservation

The reservation design is mutually exclusive with exp_22:

- exp_22’s detector is read-only. It treats any reservation on that card as busy, treats GPU 0 as busy while the exp_35 supervisor lock is held, and requires the s43 table job to be terminally complete before GPU 1 is eligible.
- exp_22’s actual launcher uses the same per-UUID O_EXCL namespace and rechecks after acquisition. Whichever launcher wins O_EXCL owns the card; the loser must refuse without touching the winner’s record.
- Operational priority still depends on obeying the registered order: the Planner must not launch exp_22 on the intended exp_35 card before exp_35 reserves it.
- A leftover phase-(i) reservation, a live supervisor, a held runner lock, a foreign reservation, or a process observed during admission must cause refusal.
- The table job is pinned to GPU 1, so it does not race the default GPU-0 lane.
- Release must authenticate exact bytes plus device/inode immediately before unlinking and must refuse while the launcher, runner, evaluator descendant, or inherited campaign lock may still live.

## S4 — PASS: runner invocation

The command and environment are correct:

- The dry run and real launch select only `OFF-P1,OFF-BF,OFF-CANON,CANON-45`, pass the explicit CANON manifest, and omit `--no-dump-blocks`.
- `env -i` makes `ALLOW_SHARED_GPU` and `EXP35_APPROVED_COTENANTS` absent. The implementation should also explicitly record that they were absent.
- The runner defaults `GPU_MAX_POLLS=0`; an observed occupied or low-memory card therefore waits indefinitely.
- The runner checks the lane at selection, immediately before spawn, and again inside the exp_35 child immediately before the launcher.

One precision point: “never shares” means “never intentionally launches onto a card already observed busy.” If a foreign job starts after the final pre-exec check, one in-flight cell can overlap it—the TOCTOU residual already disclosed in §12. If occupation appears between the parent and child checks, the child refuses, and the campaign exits rc 2 rather than waiting.

The dry-run gate must prove 68 exact cell IDs and 68 argv records, with no SKIP, E2-FAIL, refusal, or stale-artifact quarantine. The runner’s dry run does create its log, plan JSON, and no-flash preflight log; it merely avoids evaluation, links, quarantine moves, and dumps.

## S5 — PASS: during and after

The handling is consistent with the H35 runner:

- rc 0: every selected cell is complete and verified.
- rc 1: one or more cells failed; ordinary rc-1 cell failures do not make this multi-cell runner fail-fast, so later cells may still execute.
- rc 2: gate/backend/lane refusal or another abort.
- rc 4: HEAD or relevant-tree pin drift.
- The monitor remains read-only and must never signal the runner, including after an observed rc-1 cell failure.
- Any nonzero final rc requires a Planner review and report to Yixun before any relaunch.
- On rc 0, the next products are `orbitrir_yaw_results.{md,json}` via `--exp35`, then the round-6b off-orbit figure/raw/summary CSVs and descriptive E5. Collection must reuse H35, the same CANON manifest, and the single campaign digest.

## S6 — PASS: binding kit and test contract

`exp35_phase_ii.sh` must implement the following pre-registered contract.

### Checks and durable records

- Self-pin the reviewed kit bytes/commit and target the explicit H35 worktree; do not add or commit the kit inside the pinned worktree.
- Record H35, manifest hash and exact 68-cell-name digest, CANON-manifest/checkpoint hashes, campaign digest `da82d012…50c1`, dump root, queue/evaluator hashes, lane indices/UUIDs, report evidence, ETA, and any second-lane decision.
- Validate the phase-(i) state hash/schema, 33/33 completion, no latch, E2 PASS, final release event, unheld supervisor/runner locks, and absence of live phase-(i) identities.
- Freeze the full lane list before `reserve`; a two-lane acquisition is all-or-nothing.
- Perform two process/reservation observations 60 seconds apart, resolve index↔UUID each time, then create each JSON with O_CREAT|O_EXCL, fully write/fsync it and its directory, retain exact bytes’ SHA-256 plus device/inode, and recheck afterward.
- Persist an atomic, file-and-directory-fsynced state machine such as `reserved → dry_run_passed → launch_pending → running → exited → released`, with failure states retained for review.
- Record the dry-run environment, cwd, argv, outer/internal log hashes, rc, exact planned set, and all 68 argv records.
- Before launch, reauthenticate every reservation, recheck the cards and runner lock, and require the matching successful dry run.
- Launch through a detached wrapper/supervisor that durably records the runner’s exact rc. Record complete identities—PID, `/proc` start time, cwd, cmdline/argv SHA-256—for the detached wrapper and actual runner.
- `status` must be read-only and report identity validity, held locks/reservations, log/summary state, receipt counts for the exact 68 cells, and final rc.
- `release` must require a durable exit/refusal state, no live wrapper/runner/evaluator descendant, and no inherited phase-ii or runner lock. It must authenticate each exact record immediately before unlinking and fsync the reservation directory afterward.

### Required tests

Tests must cover:

- Every wrong pin/hash/cell count/block/dump root/CANON entry/campaign digest.
- Incomplete or latched phase (i), E2 not PASS, missing report, stale queue pin, and an unrecorded second lane.
- UUID/index changes, occupied-card observations, O_EXCL loss, post-acquisition occupation, reservation replacement, and partial two-lane acquisition.
- Exact dry-run parsing: 68 IDs, 68 argv records, no E2-FAIL/SKIP/refusal/quarantine.
- Environment contamination proving both sharing variables remain absent.
- Detachment and PID reuse; crashes before/after spawn and before identity/rc publication.
- rc 0/1/2/4, including rc 1 continuing later cells.
- Release refusal for a live supervisor, runner, child, held lock, missing/replaced bytes, or changed inode.
- A completely sandboxed harness: no real runner, GPU command, NAS write, real `/proc` identity, or real reservation path.

### The kit must never

- Move or modify H35, install packages, alter the environment, select E2-FAIL, suppress dumps, or enable GPU sharing.
- Overwrite, adopt, or remove another holder’s reservation.
- Signal, kill, attach to, or otherwise control any process.
- Automatically relaunch, clear a failure, change lanes, or start exp_22.
- Release on ambiguous process/state evidence.
- Let `status` mutate state.

## S7 — PASS: approval and user decisions

APPROVE. Yixun need not make another scientific or protocol decision. He receives the measured ETA/card plan first; if he does not answer, the registered default is one lane on GPU 0. Any second lane requires a recorded Planner decision before the initial runner launch.

## Findings, ranked

- **BLOCKING:** none.
- **HIGH:** none.
- **NIT 1:** Replace the absolute phrase “never shares” with “never intentionally launches on an observed-busy card”; preserve the accepted post-check TOCTOU disclosure.
- **NIT 2:** Define “clean” as the runner-relevant tree being committed/clean. The whole worktree necessarily contains untracked phase-(i) evidence. Record and compare the exact campaign digest `da82d012…50c1`.
- **NIT 3:** Clarify that a second lane must be added to the frozen lane list before the single initial runner launch; it cannot be attached dynamically under the runner’s global lock.

## What I verified

Read-only probes included:

- Full plan and parent diff at `ef4a446ad2b71ec302da415d69bebc66749b4173`.
- All 532 lines of the H35 runner, including selection, locks, CANON handling, GPU polling, sharing gates, digest checks, and exit codes.
- H35 manifest counts/combinations/protocols and both manifest SHA-256 values.
- Run-worktree HEAD/tracked cleanliness and an independent reconstruction of the relevant-tree campaign digest.
- Read-only phase-(i) state fields, including E2 PASS; phase completion was not assumed.
- The pinned exp_30 queue/evaluator hashes and static GPU dispatch lines.
- exp_22 PIN2’s detector, card rule, sealed launcher, O_EXCL reservation, authenticated release, and A7 ordering.
- Round-6b source registration, H35/CANON pins, figure inputs, and E5 implementation.

I did not run any runner, dry run, supervisor, script, test, GPU command, or process probe; did not touch or inspect the reservation directory; and made no filesystem or environment changes.
