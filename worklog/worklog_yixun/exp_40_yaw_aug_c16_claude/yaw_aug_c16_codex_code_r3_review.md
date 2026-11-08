**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-08 · **Round:** exp_40 code review 3 of 3 (HEAD 09150566)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING 1 — LAUNCH-BLOCKING: admission ignores the sealed HF hub

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2111-2119,2804-2806,3005-3009`  
`src/models/conditioners.py:449-458`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_neuronic_commands.md:50-52,133-134`

`admission_state_subprocess` sets offline flags but inherits `HF_HOME`/`HF_HUB_CACHE`; `admit` has no sealed-hub argument. The launchers’ exports occur in child shells and do not survive to the standalone admission command. Strict wrapper construction calls `AutoModel.from_pretrained`, so admission either:

- resolves from the mutable ordinary cache instead of the verified sealed hub; or
- fails on an otherwise correct installation where only the sealed hub is available.

This leaves both A6000 and neuronic paths without an acceptable experiment-level admission.

**Required change:** make `admit --hf-sealed` mandatory; validate it with `hf_sealed_reasons` and `hf_resolution_reasons`; clear competing HF cache variables and explicitly set `HF_HOME`, `HF_HUB_CACHE`, and offline flags in the admission subprocess. Bind the sealed snapshot identity into the admission record and validator. Update both command-sheet admission calls and add a strict-load subprocess test proving resolution occurs only under the supplied sealed hub.

### HIGH 1 — The “exact” smoke/PILOT receipt validator remains semantically incomplete

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:844-927,930-971,1929-1935`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:116-124,158-173`

The top-level and most nested shapes are now exact, and the real producer’s SyncBN counts, UUID strings, sampled keys, and rank-zero banner are compatible. However:

- `rng_signature` accepts any nonempty dictionary, while the GPU producer emits exactly `python`, `numpy`, `torch_cpu`, `torch_cuda`, and `sobol`, each a SHA-256.
- Reference smoke/PILOT offset digests, histograms, and sampled offsets are checked only structurally—not against the counter-based stream.
- Histogram bins may be negative if their sum remains correct.
- `sec_per_step` need only be finite and positive. The durable PILOT receipt contains no timing-window timestamps from which the rate can be recomputed.

Thus a structurally valid fabricated DDP_SMOKE can authorize FULL, and a fabricated PILOT rate can request inadequate wall time.

**Required change:** require the exact RNG sub-schema and nonnegative histogram bins; recompute the DDP_SMOKE/PILOT counter stream in the immutable run tree; include the PILOT timing-window evidence in the receipt and derive `sec_per_step` in the validator. Mutation-test balanced histogram corruption, a different valid offset digest/sample set, every RNG omission/addition, and a rate inconsistent with its recorded timestamps.

### HIGH 2 — The claim probe records live checkout bytes, not the source Slurm actually executed

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_claim_probe.sbatch:26-37`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2643-2648`

Slurm executes a spooled copy of the sbatch script, but the job hashes the live checkout’s script, runs the live checkout’s kit, and obtains `source_sha` from the live checkout. If that checkout changes while the probe is queued, the receipt can claim the newer commit/script although the spooled orchestration came from the older source.

The round-3 `EXPECT_SHA` and current-mount checks do not close this gap.

**Required change:** submit the probe from an immutable archived worktree at `EXPECT_SHA`, export and verify that SHA in the job, run its kit from that tree, and hash the actual spooled script (`BASH_SOURCE[0]`) or compare a submission-time script digest with the spooled bytes. Add a queued-checkout-drift test.

### HIGH 3 — `neuronic-time` validates tree/environment/rung against values taken from the receipt itself

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_submit_neuronic.sh:72-84`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2952-2961`

The validator receives `rec["tree_digest"]` and `rec["env_digest"]` as its expected values and derives the rung/run ID from the receipt. The selected shell `RUNG`, `RUN_ID`, and prospective FULL tree/environment are not independently supplied.

The in-job FULL check eventually detects a mismatch, but only after submission and the execution-side claim, burning the immutable FULL identity.

**Required change:** build and verify the prospective run tree before computing time; pass the selected `--rung`, `--run-id`, tree digest, and environment digest independently into `neuronic-time`; reject any mismatch before `neuronic-submit` and before the claim can be created. Add wrong-rung, wrong-run-ID, wrong-tree, and wrong-environment tests.

### MEDIUM 1 — Cell receipts can use future `t_start`/end times

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:569-605,1031-1038`

The validator requires numeric, internally consistent timestamps but never requires `t_start` or `t` to be no later than the current time. A future-dated cell can therefore make an older AR receipt appear fresh and make any admission predate the cell.

**Required change:** require `0 <= t - t_start`, `seconds >= 0`, and `t_start <= t <= now + a small clock-skew allowance` in the shared receipt validator, including offline table validation. Add future-start, future-end, negative-duration, and excessive-clock-skew tests.

### MEDIUM 2 — Admission’s provenance objects are not validated beyond being dictionaries

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2173-2188,2250-2253`

The generator records fixed `historical` and `current_assets` values, but `admission_reasons` accepts any dictionaries. A rewritten admission can claim false historical knowledge or arbitrary AR/DINO/VAE identities and still pass if subsequently pinned as a runtime input.

**Required change:** require `historical == HISTORICAL`; require the exact `current_assets` keys and the pinned DINO revision/config/weights, VAE digest, and committed AR-manifest digest; require a finite, non-future admission time. Add one mutation test per nested field.

### MEDIUM 3 — Tests miss the remaining acceptance holes

`src/tests/test_exp40_kit.py:1312-1320,1374-1416,1613-1625,1656-1667`  
`src/tests/test_exp40_eval.py:923-970`

Notable gaps:

- The supposed exact receipt fixture itself has an incomplete RNG dictionary.
- No balanced histogram/digest/sample corruption is tested.
- The admission subprocess test uses `strict_load=False`, so it never exercises DINO/HF resolution.
- Neuronic timing tests do not independently vary selected rung/run ID/tree/environment.
- Runtime tests cover stale and pre-rehash times but not a future cell clock.
- No test models Slurm spooled-source versus live-checkout drift.

**Required change:** add the specific negative cases listed in BLOCKING 1 and HIGH/MEDIUM 1–3 and assert named refusals.

### NIT

None.

## Q1 — Round-2 finding status

- **B1 fixed completely:** the control directory and claim namespace are fixed and non-overridable.
- **B2 partial:** `EXPECT_SHA`, mount schemas, and current-mount comparison are present, but the job hashes/runs live checkout files rather than the spooled/immutable source.
- **B3 partial:** top-level and most nested schemas are exact, but RNG, stream semantics, and PILOT rate provenance remain incomplete.
- **B4 partial:** runtime inputs, admission binding, E0 derivation, and per-cell AR age are fixed; future `t_start` remains accepted.
- **H1 fixed completely:** capture assembly re-hashes each declared source tree.
- **H2 partial:** the admission’s kit/config/recomputation run in an immutable training-commit tree, but strict reconstruction resolves DINO through inherited external cache state.
- **H3 fixed completely:** NAS receipt exactness and already-manifested source rehash are implemented.
- **M1 fixed completely:** the command-sheet order is mechanically coherent.
- **M2 fixed completely:** dry-run gate failures are nonzero refusals.
- **M3 partial:** several newly relevant negative tests remain missing.
- **M4 fixed completely:** `test_step` is now actually executed with raising augmentation/rotation stubs.
- **NIT fixed:** training completion is described as pending admission rather than experiment PASS.

## Q2 — Round-3 defects

The new/remaining defects are BLOCKING 1, HIGH 1–3, and MEDIUM 1–2.

Specific requested checks:

- The admission tree, cwd, training-commit kit, and subprocess path are correct.
- Real DDP producer shapes are compatible: SyncBN is counted as `_BatchNorm`, bare and `GPU-` UUIDs are accepted, sampled keys match, and the banner is rank-zero-only.
- The probe now compares compute-node and current login-node `SOURCE/FSTYPE`; it intentionally does not compare cross-host `st_dev`. This is conservative and fail-closed.
- Control-directory pinning is complete.
- Eval runtime identity is substantially fixed, except for the future-time hole.
- Neuronic time validation is not independently bound to the prospective FULL identity.

## Q3 — Treatment and absent-key equivalence

Unchanged and correct.

- The original uniform draw is snapped down as `s * floor(d/s)`.
- For `img_w=512`, `s=32`, the result is exactly uniform over 16 offsets.
- Quantization consumes no randomness.
- `offset_step` is optional, type-strict, a proper divisor, and appended last to the constructor.
- With the key absent, the factory passes the original three exp_15 kwargs only.
- `offset_step=1` preserves the original offsets.
- `src/data/yaw_rotation.py` remains byte-identical to base.
- Validation and test paths do not dispatch training augmentation.

## Q4 — End-to-end launchability

**A6000:** training through DDP_SMOKE, FULL, audit, and run result is statically launchable on correct inputs. The complete path is not acceptable because admission is not bound to the sealed HF hub. After that fix, the E0 → cells → verify → collect → figure → table ordering is coherent.

**Neuronic:** PILOT and FULL job execution are mechanically coherent, but the path is not acceptable end-to-end because:

- probe source identity is not the spooled source;
- the time gate is self-bound and can burn the FULL identity before the in-job mismatch is detected;
- post-pull admission has the same sealed-HF defect.

The admission step is the one that may directly fail on otherwise correct inputs containing a valid sealed hub but no usable ordinary HF cache.

## Q5 — Tests

The committed records report:

- `735 passed` in the round-3 four-file suite;
- `204 passed` in the eval subset;
- the earlier full suite: `6883 passed`, seven documented CPU/CUDA-environment failures.

My exact authorized pytest attempt did not collect tests because the read-only sandbox had no usable writable temporary directory:

```text
FileNotFoundError: No usable temporary directory found in
['/tmp', '/var/tmp', '/usr/tmp', '<worktree>']
```

I did not alter `TMPDIR`, install anything, or rerun outside the authorized command.

The missing/non-probing tests are listed in MEDIUM 3.

## Q6 — Anything else

No additional treatment, checkpoint-publication, DDP banner, GPU UUID, SyncBN, sampled-key, command-order, or table/figure defect was found. `git diff --check 78c64d29..HEAD` is clean.

## What I verified

- Read the SOP, announcements, binding plan §14, both prior reviews, round-3 worklog mapping, commit ledger, and requested logs/DRY_RUN records.
- Reviewed `78c64d29..09150566` and the complete `2b42468c..09150566` experiment diff.
- Traced A6000 and neuronic execution from prerequisite receipts through admission and all evaluation/publication stages.
- Compared exact validators with the real evidence producer fields and distributed launch behavior.
- Rechecked treatment mathematics, absent-key equivalence, E1 capture provenance, NAS publication, and evaluation runtime/E0 binding.
- Ran no training, evaluation, GPU, Slurm, checkpoint loading, process-control, package-management, or write command, and modified no files.