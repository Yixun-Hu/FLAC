**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code C3e (exp_35: the C3d fixes + the Planner's single-instance lock; 1f31c7a7 → 5b065801e9d1c1844461ec84cd0558702c714326)

**Verdict:** SHIP — the three C3d HIGH findings are resolved and Planner P1 is implemented. `5b065801e9d1c1844461ec84cd0558702c714326` is ready to become H35, conditional on INT accepting the branch and ladder 4 finishing with zero failures.

## E1 — RESOLVED

The state now binds the reservation to `{uuid, epoch, path, sha256}`; creation returns the hash of the exact written bytes, and authentication checks bytes, fields, epoch, and inode: `exp35_supervisor.py:218-224,295-348`.

Authentication occurs:

- Before every `ready()` decision: `:588-625`.
- Before and after the hand-over gap, before committing both records, and immediately before removing GPU 1’s record: `:627-668`.
- After the `in_flight` save as the last pre-spawn check; failure restores the chunk to `planned` and spawns nothing: `:687-705`.
- Immediately before final release: `:878-883`.

No non-trivial path remains that spawns with a known-invalid reservation or intentionally removes a foreign record. The inherent userspace TOCTOU remains: an external writer could race the final authentication and `Popen`, or the inode check and `unlink`. The latter was disclosed; the launch entry should describe both as the same microscopic check/action residual.

## E2 — RESOLVED

`listed_identity()` is tri-state and re-queries the compute table after an unreadable identity: `exp35_supervisor.py:555-567`.

- Still listed, or table re-query failure: unreadable and latched.
- No longer listed: gone.
- Complete identity available: compared field-by-field.

This governs the approved trainer in `ready()`: `:569-621`, and every approved foreign PID during the running-cell watch: `:713-748`. Other identity paths either refuse directly—the runner and dry run—or cannot approve an extra PID because the process-set equality already fails. I found no path that approves an unreadable listed PID.

## E3 — RESOLVED

The collector’s idempotent branch requires:

- A valid sidecar sealing the current JSON.
- The finalizer’s canonical JSON bytes.
- Exact ordered launch provenance for backend, overlay hash, launcher hash, campaign digest, log path, and current log hash.

It returns without writing only on complete equality: `depth_canon_collect.py:1256-1294`. The CLI then re-runs `PROBE_CONTRACT_BF`: `:2354-2365`.

On restart, the supervisor reruns finalization/contract, recomputes JSON, sidecar, and log hashes, compares them with the persisted ledger, and only then performs the final guard: `exp35_supervisor.py:750-768,787-833`.

The restart test kills the supervisor after persisted finalization/ledger but before the final guard and uses the real collector finalizer: `test_exp35_supervisor_loop.py:471-522`. The supervisor harness stubs the heavyweight contract result, but the actual idempotent CLI plus complete toy probe contract is separately exercised at `test_exp35_bf_probe_contract.py:138-174,333-345`; the combination is production-faithful for the defect.

Ladder 1 independently reproduced exp_25 JSON `ec3fcd08…b6e0` and identical Markdown.

## E4 — PASS

`Supervisor.run()` opens and exclusively locks `<state>.lock` before `_run()` can load the state, retains the descriptor through reconciliation, every chunk, gates, and final release, and returns rc 3 with the required message on contention: `exp35_supervisor.py:835-848`.

The dry run executes before normal `run()`, opens an existing lock read-only, probes it with `LOCK_SH | LOCK_NB`, and never creates it: `:976-992,995-1040,1071-1078`.

Tests cover contention with and without an existing state, state-byte preservation, whole-run lock lifetime, and dry-run absence/read-only probing: `test_exp35_supervisor_loop.py:528-575`; `test_exp35_supervisor_cli.py:237-253`.

For the fixed launch-entry state path and a lock file that is not externally removed/replaced, two supervisors cannot act on the state.

## E5 — PASS

- No signalling API appears in the supervisor; a latched running cell is still polled to completion.
- The latch is assigned only when currently `None` and is never cleared: `exp35_supervisor.py:531-534`.
- GPU 0 inherits neither `ALLOW_SHARED_GPU` nor the approved-tuple variables: `:456-499`.
- `STATE_SCHEMA` remains 1, but old states are refused through the exact `supervisor_sha256` pin and the newly required reservation hash: `:47,218-239`.
- In-flight cell reconciliation, `runner_verified` gate replay, probe restart, and fail-closed reservation reconciliation remain intact: `:822-833,850-883`.
- The runner, guard, identity helper, and probe are unchanged from `1f31c7a7`.
- The previously resolved C3c/C3d state evidence, dry-run, complete-identity watch, probe ledger, and exp_25 compatibility items remain resolved.

## E6 — PASS, ladder 4 pending

RED/GREEN evidence:

- E1/E2: **49 failed, 148 passed** → **197 passed**.
- E3/P1: **7 failed, 159 passed** → **264 passed**.
- Other collector/guard/identity suites: **183 passed**.

The RED failures directly include exact-byte ownership, hand-over-gap/commit/removal mutation, last-pre-spawn mutation, final release, unreadable identity in `ready()` and watch, finalized-probe restart, lock contention/lifetime, and dry-run lock behavior.

The deliberately non-RED regression cases are disclosed: a PID genuinely leaving the table, refusal of any non-identical finalized record, and changed-ledger restart rejection. None of the fixes passes solely because of its fake: reservation and `flock` operations are real filesystem operations, identity tests use the real helper over fake `/proc`, and probe tests use the real finalizer plus a separate real contract/CLI test.

Ladder 4 was still incomplete at my final read: its file contained only the two start lines, so no pytest summary was available.

## E7 — PASS

Independent `exp35_count_code_lines.py` results:

- `955ac38c`: production **+112/−49 = 161**; tests **+206/−26 = 232**.
- `d3381fed`: production **+45/−7 = 52**; tests **+151/−18 = 169**.
- `5b065801`: no code or test changes; records only.

All three commits carry `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git diff --check 1f31c7a7 5b065801 -- '*.py' '*.sh'` is clean. No environment/package files changed, and the RED/GREEN commands used `CUDA_VISIBLE_DEVICES=`.

`git diff localization-exp 5b065801 -- eval_FLAC.py src ':(exclude)src/tests'` is empty.

## E8 — PASS / READY

C3e is ready for H35 after INT and the final ladder-4 result. No new scientific or operational decision is required.

The launch entry should retain the Coder’s four rules and four disclosed residuals, and explicitly state that the last pre-spawn authentication—like compare-and-unlink—is a userspace check/action boundary rather than an atomic primitive. It should also record the final ladder-4 summary/hash when available.

## Ranked findings

No BLOCKING, HIGH, or NIT findings.

Accepted operational residuals:

- Microscopic authentication-to-spawn and inode-check-to-unlink races.
- A trainer caught mid-exit but still GPU-listed latches fail-closed.
- The dry-run shared-lock probe can briefly cause a starting supervisor to refuse.
- A crash after hand-over state commit but before GPU 1 release leaves both files and latches on restart.

## What I verified

- Read binding plan r4 through §14 and the C3d review.
- Reviewed committed blobs using `git show` and `git diff`.
- Verified all stated production hashes, including supervisor `0b5d94a8…`, collector `61cbda5e…`, manifest `440eb1be…`, overlay, launcher, runner, guard, identity helper, and probe.
- Confirmed 346 manifest rows, 116 exp_35 rows, and exp_35-row digest `500d6f5e…`.
- Verified ladder hashes before reading: rungs 1, 2, 3, and 3b exactly matched the supplied values. Ladder 4’s start-only snapshot hash was `dbf001a8…`.
- Confirmed ladder 2’s 5,022-entry read-only snapshot, absent lock/state/reservation files, and ladder 3’s 32-cell plan with campaign digest `da82d012…`.
- Did not run pytest, project code, the runner, supervisor, guard, probe, collector, training/evaluation, `nvidia-smi`, or any process-attaching command; modified nothing.
