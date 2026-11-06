**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code INT-L2 (exp_35: the Planner's revised phase-(i) launch script; 8c71e258 → 6f6fbd88)

**Verdict:** REQUEST-CHANGES — L1’s capture/launch binding is resolved, but one HIGH prerequisite defect remains: the purported exact DINOv3 cache-set check ignores hidden entries and is therefore weaker than the repository’s `assert_vit_pin()` contract.

## Per-item status

| Item | Status | Review |
|---|---|---|
| **L1** | **RESOLVED** | `exp35_phase_i_launch.sh:114-115,189-268,271-302,325-335`. All pre-tuple CHECKs must pass at `:235`; dry-run failure prevents the record copy and seal at `:249-254`; the outer pipeline waits for the capture log to close, checks its rc/no-failure condition, and only then writes the noclobber seal at `:331-334`. Launch verifies the supplied seal hash, H35, script hash, campaign digest, tuple/copy/log hashes, age and ladder hashes, then reruns mutable/common/live-tuple/dry-run checks and a final live-state check. With the exact capture-emitted seal path/hash frozen in the launch entry, failed, partial, stale, or subsequently modified captures are not launchable. |
| **L2** | **PARTIAL** | `:43-65,103-112,117-172`. NAS P1/BF lines, exact interpreter/package versions, exp_22 file/process absence, all four A6 CSVs, campaign digest, and four ladder rungs are hard checks. The digest implementation is equivalent to the runner’s exp_35 `tree_digest`: it expands 116 files and recomputes `da82d012…`. DINO hashes are hard-pinned, but the exact-set check is incomplete; see HIGH 1. |
| **L3** | **RESOLVED** | `:39-40,111-112,152,202,219,227-231,239-248`. Direct audit Python calls use `-B`. The dry-run and real supervisor invocations at `:178,307` do not spell `-B`, but run under `env -i` with `PYTHONDONTWRITEBYTECODE=1`. The supervisor’s `_env` does not strip it, so it reaches runner processes and collector calls; exp_35 evaluator/probe launches also set it explicitly. This changes only bytecode caching, not evaluation semantics. |
| **L4** | **PARTIAL** | Draft handoff from the tee subshell is correct; the generated seal’s awk parsing and age arithmetic work for genuine seals; `find -newer` covers ordinary worktree writes; the stub is faithful through the last pre-detachment checks. No signal operation or unintended write to the exp_30 checkout/NAS was found, and `flock` prevents two active supervisors. Several hardening NITs remain below. |
| **L5** | **NOT RESOLVED** | Do not run the production script with `EXPECT_SCRIPT_SHA256=82155db5…` yet. Fix HIGH 1, obtain/review the new script hash, then run a fresh capture and its exact emitted launch line. The external launch entry must also carry the disclosures and E5 registration listed below. |

## Ranked findings

### HIGH 1 — the “exact” DINOv3 cache-set check ignores hidden entries

Evidence: `exp35_phase_i_launch.sh:47-48,142-144`.

The two required files are correctly hashed, but both set comparisons use plain `ls`:

```bash
ls "$DINO/snapshots"
ls "$DINO/snapshots/$VIT_REV"
```

Plain `ls` omits dot-prefixed entries. Therefore an extra hidden snapshot or a hidden file inside the pinned snapshot still yields `CHECK OK`. The repository contract uses `os.listdir()` and `os.walk()` in `assert_arm_configs.py:65-89`, both of which include hidden names.

Concrete fix:

- Replace the set enumeration with an exact hidden-aware check, such as sorted `find -mindepth 1 -maxdepth 1 -printf '%f\n'`, or a small `"$PY" -B` checker matching `os.listdir`/`os.walk`.
- Retain both existing SHA-256 checks.
- Add negative test-copy cases for an extra `.snapshot` and an extra hidden file under the pinned snapshot.

### NIT 1 — the tuple “canonical path” predicate permits lexical traversal

Evidence: `exp35_phase_i_launch.sh:285`.

Bash pattern `*` inside `[[ … ]]` can match `/`; I confirmed that a string such as:

```text
$TUPLE_DIR/exp35_approved_cotenants_../../outside.json
```

matches the current pattern. A genuine, hash-frozen capture cannot produce this value, so this does not break L1, but the verifier’s “canonical” claim is too broad.

Concrete fix: separately require `dirname "$tuple" == "$TUPLE_DIR"`, validate only the basename against `exp35_approved_cotenants_*.json`, and reject `/`, `..`, symlinks, and non-regular files. Apply equivalent canonical checks to the capture log and ladder basenames.

### NIT 2 — freshness is not rechecked at the final boundary

Evidence: `exp35_phase_i_launch.sh:289-301`.

Age is checked before mutable checks and the dry run. A seal just under three hours old can cross the limit before detachment. Validate `sealed_at` as decimal before arithmetic and repeat the age check in the final-look block.

### NIT 3 — dry-run transcripts are not noclobber files

Evidence: `exp35_phase_i_launch.sh:174-186`.

`> "$log"` can overwrite a pre-existing orphan transcript, contrary to the script’s “NEW files, never overwritten” statement. Create it under `set -o noclobber` before redirecting.

The supplied tests also do not exercise stale seals, tuple/copy/log/ladder tampering, failed dry runs, malformed `sealed_at`, path traversal, or the DINO hidden-entry case. Add those mutation cases when revising.

## Launch-entry requirements

After the revised script is reviewed, the launch entry must record:

- H35, C3e’s SHIP verdict, the newly reviewed launch-script SHA-256, capture/dry-run log hashes, tuple and byte-identical copy hashes, seal path/hash, and campaign digest.
- All four ladder hashes and Ladder 4’s `1513 passed, 1 skipped, 0 failed` result.
- The accepted residuals from INT I6 and C3e: shared-memory OOM risk; NAS stalls without in-flight termination; sampled guard/identity blind spots; predictor risk; the GPU-0 post-check race; authentication-to-spawn and compare-to-unlink userspace windows; mid-exit trainer latch; dry-run lock contention; and crash-between-handover-commit-and-GPU-1-release behavior.
- The future E5 script’s exact path/hash, outside the runner’s `RELEVANT` set. Phase (ii) must separately re-pin its external CANON manifest/checkpoint.

## What I verified

- Confirmed `6f6fbd88` and production script SHA-256 `82155db541b6c42395bb27fae169defacb2e3c316a1a19645210c167f2ac4bd8`; confirmed the prior blob’s `4adeb160…`.
- Inspected the full `8c71e258..6f6fbd88` Git diff; `git diff --check` and `bash -n` passed for production and test-copy scripts.
- Independently expanded the runner-equivalent digest set to 116 files and obtained `da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1`.
- Verified ladder hashes `c3f62847…`, `587ee8f9…`, `321dfc7d…`, and `1fc44f92…`; Ladder 4 ends with `1513 passed, 1 skipped`.
- Verified the test copy differs only in `MAINX`, `TUPLE_DIR`, supervisor-log destination, and the no-launch stub.
- Verified supplied evidence hashes and contents: t1 refused before tuple/seal; t2 had 62 CHECK OK and sealed; t3 had 62 CHECK OK and reached `WOULD LAUNCH`; t4 rejected the wrong seal hash.
- Did not run either launch script, its test copy, pytest, supervisor, runner, or any GPU/process command; made no filesystem or environment changes.
