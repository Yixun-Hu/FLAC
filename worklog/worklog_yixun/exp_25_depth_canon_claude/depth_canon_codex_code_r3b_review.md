**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-23 · **Round:** code r3b (tie audit + composition probe + launcher/guard suite/pins)

**Verdict:** REQUEST-CHANGES

## Findings

### BLOCKING

1. **FULL is not mechanically gated by the tie audit.**

   [depth_canon_launch.sh:207](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:207) validates only the SMOKE receipt. It never requires the pre-FULL audit artifact, verifies its sidecar, or checks zero survivors/errors. A FULL run can therefore start without auditing all 9,650 maps.

   **Fix:** require a qualifying audit JSON before FULL: exact three registered splits, `limit=null`, complete union, zero survivors, zero errors, PASS, expected item/map counts, matching split/config/operator/audit-script hashes, and a valid SHA-256 sidecar. Add guard cases for absent, subset, stale, tampered, survivor-containing, and error-containing reports.

2. **SMOKE can PASS without proving finite loss or measuring canonicalization latency.**

   [depth_canon_launch.sh:359](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:359) proves banner/ranks/endpoint and scans for the strings `nan|inf`, but it does not require any parseable loss value. A log containing no loss still passes. More directly, [depth_canon_launch.sh:478](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:478) writes `"canonicalization_latency": null`.

   This does not meet rung 5’s finite-loss and measured-latency requirements.

   **Fix:** instrument `_canonicalize_batch` with a stable rank-0 latency record, require a positive finite measurement in the SMOKE receipt, and require at least one parseable finite training loss after a completed step. Make both fields mandatory when FULL qualifies evidence. Add guard cases for missing loss, nonnumeric/nonfinite loss, missing latency, and invalid latency.

### HIGH

3. **The audit can PASS float32 depth data that production training rejects.**

   [depth_canon_tie_audit.py:225](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py:225) accepts any floating NumPy dtype. The AR_md cross-check effectively pins the `[256,512] → [3,256,512]` layout, but does not require `np.float64`/`torch.float64`. Production explicitly rejects non-float64 depth at [diffusion.py:597](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:597).

   **Fix:** assert raw shape `(256,512)`, raw dtype `np.float64`, cloud shape `(3,256,512)`, cloud dtype `torch.float64`, CPU residency, and the registered dataset’s depth modality. Add float32 and wrong-contract fixtures.

4. **The composition probe does not distinguish registered references from diagnostics.**

   [depth_canon_cond_probe.py:445](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:445) permits arbitrary checkpoint, configs, item count, batch size, seed, device, and canonicalization mismatch. [depth_canon_cond_probe.py:503](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:503) can publish the same successful schema for a five-item CPU run, a non-EMA checkpoint, or a mismatch diagnostic.

   **Fix:** either enforce the registered contract—64 unseen items, batch 64, seed 42, CUDA, exact configs, EMA weights, no mismatch—or add an explicit `--diagnostic` mode producing `registered:false` and a non-qualifying exit/status. The 3c collector must reject nonregistered artifacts.

5. **The pin closure omits two files consumed by the launch gate.**

   [depth_canon_make_pins.sh:23](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_make_pins.sh:23) pins `FLAC_AR_BVp1.json`, CANON, and `assert_arm_configs.py`, but the assertion script actually opens `FLAC_AR_BV.json` and `FLAC_AR_BF.json` at [assert_arm_configs.py:105](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py:105). Uncommitted edits to those files can alter the supposedly pinned preflight.

   **Fix:** add both configs to the manifest and dirty-path check, or refactor the gate to exercise only the pinned P1/CANON inputs. Regenerate the manifest at the reviewed commit.

6. **The required `flac` execution environment is observed, not enforced.**

   [depth_canon_launch.sh:105](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:105) invokes whichever `python` appears in `PATH`; lines 328–331 merely print its identity and a `pip freeze` hash. A compatible but numerically different environment can launch the registered run.

   **Fix:** require `/home/yixunhu/miniconda3/envs/flac/bin/python`, or fail closed on the resolved interpreter and an approved environment/package digest. Use that interpreter consistently for all gates and training.

7. **The GPU gate proves memory availability, not that GPUs are unoccupied.**

   The launcher never signals or kills another process, which is good. But its memory-only threshold can still admit training alongside a foreign xRIR job that leaves 22.5 GiB free, causing compute contention and possibly memory pressure.

   **Fix:** before launch, query compute processes on GPUs 0 and 1 and fail closed if any foreign compute process exists; do not kill it. Add a mocked guard case. Until then, the operator must independently ensure the xRIR jobs have ended.

### NIT

8. **Mixed survivors plus processing errors return survivor code 1 rather than error code 3.**

   [depth_canon_tie_audit.py:324](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py:324) gives survivors precedence. It cannot produce a false PASS, but a partially processed run is reported as ordinary FAIL rather than incomplete/error.

   **Fix:** give errors precedence or define an explicit combined status.

9. **The SMOKE→FULL HEAD equality is safe but unnecessarily broad.**

   It prevents harmless receipt/worklog commits between SMOKE and FULL, contrary to the content-scoped lesson.

   **Fix:** bind evidence to the training manifest plus launcher/config-contract hashes, retaining HEAD as provenance rather than an equality condition. Any training-closure, launcher, or gate change must still require a new SMOKE.

10. **W&B writes repo-local state outside the NAS save root and experiment directory.**

    `train.py` constructs `WandbLogger` without `save_dir`, so normal W&B metadata goes under the checkout’s `wandb/`. This matches existing behavior but should be documented as the one expected extra write target.

## Technical conclusions

### A. Tie audit

The survivor logic is faithful for the relevant `StabilizedTieError`: it builds depth through the actual `AR_md.get_custom_metadata`, invokes the actual `canonical_column`/`recover_depth_proxy` path through `canonicalize_scene_metadata`, and records a survivor exactly when that stabilized-tie exception is raised. Other operator/loader failures become errors and cannot yield PASS.

Path resolution matches `dataset.py` plus AR_md’s inline transformation. Dedupe by resolved depth-map path is correct: one path names one file, while identical bytes at separate paths remain separate audit units. The depth-only call cannot change AR_md’s depth branch relative to full metadata loading; it does not certify unrelated pose/audio branches.

The independent NumPy raw selector correctly performs the rooted-image lexicographic comparison in column-major order. Keeping raw/proxy agreement diagnostic rather than gate-defining is the preregistered choice and is appropriate.

Sorted paths plus ordered `Pool.imap` make output worker-count independent. The split ledger, input/code hashes, per-map rows, and sidecar are adequate records once the dtype contract and launcher binding are fixed. Roughly 9,650 rows is a reasonable artifact size.

### B. Composition-control probe

The numerical chain matches evaluation:

- conditioner under fp32/autocast-off and bf16 evaluation policies;
- identical per-item fixed noise for A/B;
- actual conditioning-input adapter;
- actual one-step `sample_discrete_euler`, `cfg=1.0`, `batch_cfg=True`, and `dist_shift`;
- one recorded DiT velocity;
- decoder, clamp, then padding in evaluation order.

The EMA/prefix logic is genuinely pinned against the real evaluator by the test. Relative metrics come from `depth_canon_stats`; Tier-1 hashes, `dj`, tie classes, determinism floor, checkpoint/config/split hashes, device, dtype policy, seed, and ARE refusal are all present.

The offset mapping is exact: `d64=45°`, `d128=90°`, `d256=180°`, and `d384=270°` bit-for-bit. Use those fixed controls for the matching C4 cells and random→random; use pooled p99 as the registered Tier-2 reference. Running from the repository root is acceptable because the configs contain cwd-relative paths.

### C. Launcher

The training argv is semantically P1-exact for the first 40k steps: 2 GPUs, batch 32, accumulation 1, SyncBN, DDP unused-parameter strategy, bf16-mixed, seed 42, six workers, 40k endpoint, 2,500 cadence, training split only, correct VAE/config/save/W&B identities, and offline HF. P1 indeed supplies no validation dataset. Explicit precision merely makes the matching default visible.

Checkpoint ownership, exact cadence/set verification, NAS identity/free-space/write probe, log draining, nonfinite scan, exact banner, rank and endpoint gates, and harness isolation are otherwise sound. `mkdir -p` plus a create/read/remove probe is an acceptable production-side NAS gate.

The DINO workaround imports this checkout’s `src`, checks `__file__`, and then runs the assertion script; it avoids the worktree/site-packages bug. Its transitive config inputs still need pinning as noted above.

A steady-state projection from step ≥10 through step 25 is acceptable with the measured P1 rate and large 60-hour margin. If a future result approaches the threshold, extend the smoke rather than treating a short estimate as decisive.

Harness evidence is typed `harness:true`, production requires `false`, and normal harness runs relocate the save root under a marked `/tmp` directory. A harness run cannot qualify as production evidence through the provided path.

### D–E. Guards and pins

All requested existing guard scenarios are present and exercise the real launcher branches: argv, rung pins, config mutation, HEAD/pin failures, stale SMOKE, occupied namespace, foreign/stale/off-cadence checkpoints, missing banner, missing endpoint, nonfinite output, NAS failure, and DRY_RUN GPU bypass. Deleting the corresponding gates would make their cases fail.

The opt-in production section is read-only with respect to the NAS: DRY_RUN performs `findmnt`/`df`/namespace reads and skips both `mkdir` and the write probe. Its approximately 11-minute duration is acceptable given repeated real DINO initialization.

The committed 44-entry manifest is sorted and reproducible from the generator, and all entries verify. The VAE symlink is represented by its stable repository path while hashing target bytes; the machine-specific `AcousticRooms` symlink is not included. The principal closure omission is the two assertion configs above.

## Coder deviations

The current round summary actually enumerates 16 deviations; I adjudicated all of them:

1. Claude Opus 5.5 co-author trailers: **accept**; they accurately identify authorship.
2. AR_md production construction in the audit: **accept with HIGH dtype fix**.
3. Independent raw selector/non-gating agreement: **accept**.
4. Exit-code scheme: **accept**, with the mixed error/survivor NIT.
5. Per-item random rotation: **accept**.
6. Per-item deterministic noise derived from `(seed,i)`: **accept**.
7. Both fp32 and evaluation-conditioning protocols: **accept**; retain the A/A floor.
8. Copied checkpoint re-key/load path: **accept**, but registered output must enforce EMA/config contract.
9. No validation config and explicit bf16 precision: **accept**; P1-equivalent.
10. Forty-four-file pin superset: **not yet complete** because BV/BF gate inputs and environment identity are unbound.
11. `/tmp` marked harness: **accept**.
12. Steady-state projection: **accept conditionally**, with longer smoke near threshold.
13. NAS launch-marker mtime ownership: **accept**.
14. 22.5-GiB threshold: **accept as stricter than plan**, but it is not an idle-GPU guarantee.
15. Missing two-rank/endpoint status as rc 6: **accept**.
16. Approximately 11-minute guard runtime: **accept**.

## Open questions

1. **Worktree-safe DINO gate:** the launcher workaround works. The assertion script should eventually acquire a proper repository-root override, but that is not independently blocking.

2. **Commits between SMOKE and FULL:** allow record-only commits. Bind FULL to content digests—training manifest, launcher, configuration contract, and audit inputs—not global HEAD equality.

3. **Steady-state projection:** trustworthy at the current margin; require longer evidence if projection is near 60 hours.

4. **3c offset mapping:** adopt `45→d64`, `90→d128`, `180→d256`, `270→d384`, and random→random.

5. **Seen/unseen overlap:** union dedupe for the audit is correct. However, the splits share 112 depth maps and my read-only enumeration found 1,093 exact item paths in common, roughly 17–18% of each split. The published cells remain valid, but seen and unseen results are correlated and must not be presented as independent estimates. Disclose the overlap; an overlap/non-overlap diagnostic may supplement, not replace, the registered cells.

## Verification performed

I performed only static/read-only checks:

- Confirmed the scoped files are unchanged after `84d7403`, despite later record/fix commits.
- Ran `git diff --check` and `bash -n` on the shell scripts.
- Parsed the reviewed Python and JSON files without importing training/GPU code.
- Regenerated the pin listing to stdout and compared it byte-for-byte; verified all 44 hashes.
- Compared the P1 and exp_17 launchers/review, evaluator, sampler, operator, AR_md, dataset resolution, plan v9, SOP, announcements, worklog, and commit record.
- Independently enumerated split item/map counts and overlap.
- Verified the fixed-offset radian values bitwise.
- Checked scoped commit sizes; the largest is under 200 changed lines.
- Reviewed the RED/GREEN cycle records and Planner receipts.

I did **not** run pytest, the launcher, guard suite, training/evaluation, GPU/NAS write commands, or process-management commands. I made no files, environment, package, or process changes. The reported 1,911-test and 118-guard-test results are Planner receipts, not reruns by this reviewer.

**Status:** exp_25 remains pre-FULL; no SMOKE/FULL authorization until the blocking fixes are reviewed and the required complete audit is bound into FULL. The foreign xRIR jobs on GPUs 0/1 were untouched.