**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** integrative pre-launch review (training closure, gates, watcher, GO procedure)

**Verdict:** **ARM** — the pinned training tree is safe to write the GO token and start the watcher after the final records/eval-fix commits, final pin verification, and a fresh `DRY_RUN=1 MODE=SMOKE EXPECT_HEAD=<final HEAD>` pass. No training-launch blocker remains.

## Launch-gate walk

1. **Rung/mode constants:** `MB=32`, `ACC=1`, `DRY_RUN∈{0,1}`, `MODE∈{SMOKE,FULL}`; FULL fixes 40,000 steps and cadence 2,500. Invalid input exits 2. [depth_canon_launch.sh:60](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:60), [depth_canon_launch.sh:98](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:98)

2. **Interpreter:** resolved `python` must be exactly `/home/yixunhu/miniconda3/envs/flac/bin/python`; failure exits 2. [depth_canon_launch.sh:133](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:133)

3. **HEAD identity:** `EXPECT_HEAD` must be 40 hex characters and equal current `HEAD`; failure exits 2. [depth_canon_launch.sh:160](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:160)

4. **Pin manifest:** regenerated path list must equal the committed list; all 48 hashes must pass; every pinned path plus the launcher kit must be committed and clean. Failure exits 2. [depth_canon_launch.sh:168](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:168)

5. **Configuration contract:** validates schema/types, exact canonicalization block, and literal byte-equivalence to P1 after removing that block. Failure exits 2. [depth_canon_launch.sh:195](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:195)

6. **Tie-audit gate:** committed report and sidecar are pinned; production rejects harness-tagged reports, re-enumerates the path set, checks exact equality, validates current audit-input hashes, and re-hashes every referenced map byte. Any gate failure is converted to launcher rc 2. [depth_canon_launch.sh:255](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:255), [depth_canon_audit_gate.py:247](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:247)

7. **Killable re-hash:** re-hashing runs in a child process with a 900-second deadline; timeout terminates/kills the child and invalidates the audit. Through the launcher this surfaces as rc 2, never 7. [depth_canon_audit_gate.py:192](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py:192)

8. **Smoke-evidence content binding:** production FULL binds six hashes: pins manifest, launcher, pin generator, concatenated CANON+P1 contract, audit report, and audit-gate program. It also requires the successful smoke verdict, banner, endpoint, both ranks, finite losses, latency, no smoke checkpoints, and the smoke-log hash. Failure exits 2. [depth_canon_launch.sh:269](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:269), [depth_canon_launch.sh:279](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:279)

9. **NAS:** requires source `//10.249.7.28/shared_data`, CIFS, numeric free space ≥30 GiB, and a create/delete write probe in production. Failure exits 2. DRY_RUN skips only the write probe. [depth_canon_launch.sh:316](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:316)

10. **Run namespace:** FULL requires a fresh empty checkpoint namespace; SMOKE must not have checkpoints. Failure exits 2. [depth_canon_launch.sh:347](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:347)

11. **GPU admission:** strict index-to-UUID mapping, numeric memory ≥23,040 MiB on each card, known compute-app query results, and no process on either UUID. Every GPU refusal comes only through `gpu_refuse` as rc 7; the EXIT trap remaps any unrelated 7 to 8. [depth_canon_launch.sh:75](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:75), [depth_canon_launch.sh:357](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:357)

12. **Cached GPU disclosure:** the post-gate disclosure uses the already captured readings; no later `nvidia-smi` query creates a new unhandled refusal. [depth_canon_launch.sh:412](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:412)

13. **W&B:** production requires the expected authenticated identity; query or identity mismatch exits 2. DRY_RUN and the explicit test harness skip it. [depth_canon_launch.sh:420](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:420)

14. **DINOv3/arm identity:** exactly one snapshot revision is admitted, including pinned `config.json` and `model.safetensors`; architecture, 753 tensors/64.50M parameters, seed 42, and initial state-dict identity are asserted. Failure exits 2. [assert_arm_configs.py:49](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/assert_arm_configs.py:49), [assert_arm_configs.py:94](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/assert_arm_configs.py:94)

15. **Launch marker and trainer:** production writes the marker before invoking `train.py`; launch-marker failure exits 2. Trainer rc is retained, with trainer rc 7 remapped to 8. [depth_canon_launch.sh:463](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:463)

16. **Post-run log gates:** missing banner → 3; rank/endpoint failure → 6; non-finite loss → 4, subject to preserving an earlier trainer failure. [depth_canon_launch.sh:485](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:485)

17. **Checkpoint ownership/cadence:** SMOKE must emit none; FULL hashes every run-owned checkpoint, rejects foreign/stale/missing/extra cadence points, and requires 40k. Checkpoint-set failure → 5; missing endpoint → 6. [depth_canon_launch.sh:507](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:507)

18. **SMOKE verdict:** requires finite loss, the exact latency line, a calculable steady-state projection ≤60 hours, and atomic evidence publication; metric failure → 4, publication failure → 2. [depth_canon_launch.sh:562](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:562)

No production gate is bypassable through the watcher: it passes literal `MB=32 ACC=1 DRY_RUN=0`, unsets launcher overrides and `DEPTH_CANON_HARNESS`, and supplies token HEAD as `EXPECT_HEAD`. `DRY_RUN=1` and the `/tmp` harness intentionally skip gates but cannot launch the registered production run through this watcher. [depth_canon_autolaunch.sh:195](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:195)

## Unattended-wait failure modes

| Event | Watcher/launcher response | Wrong run possible? |
|---|---|---|
| Originating session dies | `setsid nohup` keeps the watcher alive. Host reboot or watcher death leaves the last state and stops future launches. | No; inspect state/processes before re-arming. |
| NAS hiccup | Waiting does not touch NAS. Smoke/FULL audit or NAS failure becomes terminal rc 2; in-run I/O failure prevents a successful verdict. | No accepted wrong run; GPU work could be spent before a post-run failure. |
| Transient `nvidia-smi` error | Resets the three-poll count. Launcher refusal is rc 7; SMOKE retries up to five times. FULL rc 7 is terminal. | No. |
| Another job starts on one card | Before launch: dirty poll/refusal. After the final launcher query: a narrow resource TOCTOU remains because there is deliberately no later query. | Correct tree/argv still launch, but co-tenancy is possible; A8 GPU priority/no-new-job policy is the control. |
| Record-only commit | Token HEAD no longer equals HEAD; watcher returns to `WAITING_TOKEN`. | No. |
| EXPDIR fills | Atomic state/log/evidence writes fail and the watcher or launcher fails; it cannot publish successful completion. | No accepted wrong run. |
| `train.py` exits 7 | EXIT trap remaps it to rc 8; watcher records terminal failure rather than treating it as transient GPU occupancy. | No. |
| Audit re-hash hangs on CIFS | Child is killed after 900 seconds; launcher returns rc 2. | No. |
| SMOKE crashes after GPU admission | Trainer/post-run checks fail; no smoke evidence means FULL is not invoked. | No. |

The watcher additionally validates an exactly four-line, newline-terminated token; HEAD, manifest hash, timestamp, and `by=planner`; pinned-path cleanliness; zero UUID-mapped processes; ≥23,040 MiB; three clean polls; token again between SMOKE and FULL; atomic state replacement; flock; and the exact interpreter. It contains no process-killing operation and writes only below EXPDIR. [depth_canon_autolaunch.sh:52](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:52), [depth_canon_autolaunch.sh:101](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:101), [depth_canon_autolaunch.sh:126](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:126)

## Findings

**BLOCKING:** None.

**HIGH — evaluation only, nonblocking for training:** `depth_canon_cond_probe.py` still loads a checkpoint by pathname and hashes the path later, so the registered probe lacks exact-byte load provenance. [depth_canon_cond_probe.py:243](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:243), [depth_canon_cond_probe.py:575](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:575)

Concrete fix: complete rB-F3 using `eval_FLAC.read_checkpoint_exact` plus `io.BytesIO`, record `ckpt_sha256_loaded`/`ckpt_bytes_loaded`, bind both through `PROBE_CONTRACT`, and obtain Codex re-verification before the probe/evaluation campaign.

**HIGH — recovery path, nonblocking for the first launch:** there is no gated resume MODE. The production argv has no `--ckpt-path`, FULL rejects a nonempty namespace, and its freshness/checkpoint-set logic assumes one launch marker. [depth_canon_launch.sh:150](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:150), [depth_canon_launch.sh:347](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:347), [depth_canon_launch.sh:532](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:532)

Concrete fix: do not perform an ad-hoc autonomous resume. Add/review a `MODE=RESUME` path or follow the minimum HANDOFF prescription below.

**NIT — arming-shell hardening:** the documented command should also unset `DEPTH_CANON_HARNESS`, `GO_TOKEN`, and all `AUTOLAUNCH_*` overrides before starting the watcher. The current environment contains none, and the watcher validates harness inputs, so this does not prevent ARM. [depth_canon_autolaunch.sh:52](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:52)

## Training and treatment closure

The CANON and P1 configuration files differ only by `training.canonicalize = {"enabled": true, "img_w": 512}`; P1 hashes to `733ca52b…`. The launcher reproduces the historical 40k P1 recipe: two GPUs, 32/GPU, accumulation 1, SyncBN, DDP find-unused true, bf16-mixed, seed 42, six workers, cadence 2,500, the same dataset/VAE paths, no validation configuration, and HF offline. [depth_canon_launch.sh:150](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:150), [p1_ddp_launch.sh:100](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_07_fa_scratch_claude/p1_ddp_launch.sh:100)

The disabled factory path contributes no kwargs; disabled-wrapper behavior is covered by the supplied golden regression evidence. [factory.py:206](/home/yixunhu/codespace/exp-25-depth-canon/src/training/factory.py:206), [factory.py:325](/home/yixunhu/codespace/exp-25-depth-canon/src/training/factory.py:325) The static training-path audit remains applicable because all code changes after `bf7bf38` are records-only and the current pinned contents reproduce exactly. [depth_canon_training_path_diff_audit.md:19](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_training_path_diff_audit.md:19)

The treatment matches the plan:

- `recover_depth_proxy` reconstructs depth deterministically. [yaw_rotation.py:656](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:656)

- `canonical_column` uses only the image-derived rooted lexicographic key and fails closed with `StabilizedTieError`. [yaw_rotation.py:702](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:702), [yaw_rotation.py:763](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:763)

- `canonicalize_scene_metadata` validates all four pose fields, calls the reviewed rotation operator once, and verifies postcondition `j*=0`. [yaw_rotation.py:805](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:805), [yaw_rotation.py:835](/home/yixunhu/codespace/exp-25-depth-canon/src/data/yaw_rotation.py:835)

- The wrapper preserves CPU float64 metadata in an opaque carrier, canonicalizes exactly once at the beginning of each step before conditioning, and emits the specified banner, fit-end tie count, and latency lines. [diffusion.py:146](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:146), [diffusion.py:627](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:627), [diffusion.py:803](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:803), [diffusion.py:468](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:468)

## GO procedure

The order is correct. The review receipt/artifact commit and preferably rB-F3 must precede pin step 3 and DRY_RUN step 4. Any later commit changes HEAD even if it changes only records; therefore the final DRY_RUN and token must be taken at the very last pre-freeze commit.

Pins need not be rewritten for records-only or non-pinned eval changes, but they must still regenerate identically, remain at 48 entries, and be clean. Write the four-line token only after that final DRY_RUN succeeds, then freeze this worktree until FULL has launched. Record `$!` immediately in the external FLAC HANDOFF and monitor the state file. [plan_depth_canon.md:187](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:187), [HANDOFF.md:18](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/HANDOFF.md:18)

A fresh session can execute the initial launch from plan v12 plus the external HANDOFF. It cannot safely infer recovery from those documents alone; that is the HIGH above.

Because both cards are expected occupied until October 25 and rB-F3 is outside the 48 training pins, the orderly choice is to finish rB-F3 and its verification first, then perform the final HEAD DRY_RUN and arm. A8 permits arming first if rB-F3 threatens the GPU-free window, but there is presently no technical reason to prefer that ordering.

The eval driver, collector, cells manifest, generator, and probe cannot modify the training checkpoints: they use separate local/NAS evaluation namespaces and treat the CANON checkpoint manifest as input. The open probe issue therefore does not affect training or checkpoint integrity.

## Autonomous recovery minimum

If FULL crashes, HANDOFF must require:

1. Prove no rank or launcher remains; preserve the failed namespace and logs. Select the latest complete 2,500-step boundary and record its exact bytes, SHA-256, length, step, parent launch marker, and parent manifest.

2. Re-run every prelaunch gate manually: interpreter, final HEAD/pins/cleanliness, configuration, audit plus map re-hash, NAS identity/free/write, W&B identity, DINO/init/environment, and strict two-GPU occupancy/memory immediately before launch.

3. Launch the exact recipe with only `--ckpt-path <verified-boundary>` added; retain 40k endpoint/cadence/offline settings, create a resume marker, and log parent/child W&B identities.

4. Apply the same banner, both-rank, finite-loss, endpoint, tie-count, latency, checkpoint ownership, and exact-byte hashing checks. Aggregate tie/latency totals across both legs because process counters restart.

5. Build one explicit CANON lineage manifest: original hashes through boundary N and resumed hashes after N, no duplicate/foreign step, endpoint included. Review any collector change needed to consume that lineage and pass its manifest explicitly. [collect_depth_canon.py:475](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/collect_depth_canon.py:475)

6. Disclose the crash, resume boundary, parent/child logs and identities, and RNG/dataloader discontinuity.

A resumed arm remains usable as a historical matched-recipe comparison—P1 itself resumed at 32.5k—but it is not a monolithic, identical-random-trajectory causal pair. Lightning restores optimizer/scheduler/EMA/loop state, not RNG streams. [depth_canon_training_path_diff_audit.md:5](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_training_path_diff_audit.md:5), [CLAUDE.md:158](/home/yixunhu/codespace/exp-25-depth-canon/CLAUDE.md:158)

## Required result disclosures

- Historical matched-recipe P1 control, including its 32.5k resume.
- One training seed per arm.
- Seen/unseen overlap: 1,093 items.
- YAW results re-measured at the exp_25 pin.
- B-F shown only as a historical row.
- Measured canonicalization latency and the matched-steps-versus-compute limitation.
- Unattended watcher launch.
- SOP deviation: round 3c began before r3b-F1 closed.
- rB-F2 constants: rc 8 remap and 900-second audit re-hash deadline.
- Any FULL resume, boundary, reason, lineage, and RNG discontinuity.

## What I verified

Read-only probes established:

- HEAD `916aa3e722b9ef62139b0709080e6096bf99348d`; all changes after code HEAD `bf7bf38` are records-only.
- Tracked tree clean; pinned-path scoped status empty.
- Pin regeneration matched byte-for-byte; 48 entries; manifest SHA-256 `631da22f…`; all pinned hashes passed.
- CANON/P1 structural diff contains exactly the canonicalize block.
- Audit report SHA-256 `b58f253a…`, sidecar, and bytes are unchanged from the committed audited artifact; 9,650 maps, zero survivors/errors.
- Launcher and watcher pass `bash -n`; reviewed Python files pass AST parsing.
- `git diff --check` and repository connectivity pass.
- Planner logs show 2,369 passed/7 skipped/1 xfailed, guard suite 169/0/0, real audit-gate rc 0, and the documented HEAD dry-run rc 0.
- I did not run pytest, training, evaluation, GPU queries, launcher, watcher, audit, driver, or guards. A `py_compile` attempt was refused by the read-only filesystem before creating `__pycache__`; AST parsing was used instead.