# Code re-review — exp_11 consolidated round (commit 5557974)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable, `max_user_namespaces=0`); read-only instruction, tree verified clean post-review · **Date:** 2026-08-05 · *(reviewer's self-identification line below retained verbatim)*

# Code re-review — exp_11_fa_orbit consolidated round

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only review) · **Date:** 2026-08-05 · **Reviewed commit:** `5557974` (`HEAD` is documentation-only follow-up `0df4103`)

## Verdict

**REJECT — 2 NEW BLOCKING, 3 NEW NIT**

The config tests and `awk` precision diagnosis are sound, and several round-3 findings are closed. However, the checkpointing pivot was not propagated through the complete P0 pipeline, and the production output root remains operator-controlled. In addition, B2/B3/B5/B7 remain only partially closed.

The expected “only pin commit + SMOKE evidence + sign-off” state has therefore not yet been reached. **Launch authorization: none.**

## Round-3 finding disposition

| Finding | Status | Evidence |
|---|---|---|
| **B1 — post-P0 recipe pinning and submitter** | **CLOSED for this pre-P0 stage** | The launcher defines literal placeholders plus fixed `MAXSTEPS=40000` and checkpoint cadence at [fa_orbit_train.sbatch:58](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:58), refuses every unpinned normal launch at [fa_orbit_train.sbatch:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:127), and cross-checks rung/MB/GPU/budget at [fa_orbit_train.sbatch:138](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:138). The submitter reads those pins and derives GRES/CPU/RAM/time at [fa_orbit_submit.sh:45](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:45) and [fa_orbit_submit.sh:65](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:65); the job validates the resulting allocation at [fa_orbit_train.sbatch:318](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:318). The actual values are legitimately deferred. At pin review, the single common VRAM floor must be justified as the maximum safe floor across all arms, or changed to per-arm pins. |
| **B2 — restart lineage** | **PARTIALLY-CLOSED** | Checkpoints are restricted to the exact arm checkpoint directory at [fa_orbit_train.sbatch:201](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:201). CPU loading, exact step/config, future budget, warm optimizer, scheduler, EMA and SHA checks are implemented at [fa_orbit_ckpt_preflight.py:88](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:88) and [fa_orbit_ckpt_preflight.py:104](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:104). But manifest commit binding fails open when `commit` is missing: [fa_orbit_ckpt_preflight.py:69](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:69) compares only when both strings are nonempty. Require an exact, nonempty commit match and add missing/changed-commit tests. |
| **B3 — exclusive run ownership** | **PARTIALLY-CLOSED** | The normal `mkdir` acquisition is atomic at [fa_orbit_train.sbatch:373](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:373). Stale recovery is not: after failed `mkdir`, an absent/incomplete owner file is treated as stale and the job writes into the already-existing lock directory at [fa_orbit_train.sbatch:380](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:380). A contender can arrive between the first job’s `mkdir` and owner write; release also removes the directory without checking its UUID at [fa_orbit_train.sbatch:388](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:388). Use `flock`, or atomically quarantine/reacquire stale directories and release only when the stored UUID still matches. |
| **B4 — startup watchdog** | **CLOSED** | The unbounded 300-second absence timer and `scancel` are gone. The watcher reacts only to an observed wrong rank count, terminates the torchrun tree, and leaves the parent alive for classification at [fa_orbit_train.sbatch:478](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:478); post-hoc classification follows at [fa_orbit_train.sbatch:495](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:495). |
| **B5 — provenance/log durability and taxonomy** | **PARTIALLY-CLOSED** | Manifest publication/copy is checked at [fa_orbit_train.sbatch:435](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:435); FIFO tee records both torchrun and tee statuses at [fa_orbit_train.sbatch:467](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:467); the tested precedence is implemented at [fa_orbit_classify.py:46](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_classify.py:46). Remaining fail-open paths: `pip freeze` failure is not checked before its digest is recorded at [fa_orbit_train.sbatch:439](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:439), and the final dual-tee status is discarded at [fa_orbit_train.sbatch:505](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:505). The classifier’s detailed result and earlier preflight transcript also remain only in Slurm output. Check the environment command, tee the classifier/final record through a captured status, and fail with class 7 if either durable copy fails. |
| **B6 — environment and external artifact identity** | **CLOSED** | Exact Python, PL and Torch versions are gated at [fa_orbit_train.sbatch:342](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:342); the VAE digest is enforced at [fa_orbit_train.sbatch:358](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:358). Driver/CUDA are recorded, and the existing DINO revision/hash plus init-identity gate runs inside the allocation at [fa_orbit_train.sbatch:432](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:432). Restart checkpoint hashes enter the manifest at [fa_orbit_train.sbatch:454](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:454). |
| **B7 — W&B destination and lineage** | **PARTIALLY-CLOSED** | Ambient controls are scrubbed, online mode is forced, a collision-resistant ID is created, and restart uses the original ID with `resume=must` at [fa_orbit_train.sbatch:403](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:403). The authenticated email is gated and `viewer.entity` is recorded, but `WANDB_ENTITY` is never set and the actual logger-created entity/project/name/ID is not verified. Export the approved entity explicitly and verify the created run identity, at least in the required SMOKE and preferably post-run in the launcher. |
| **B8 — guard coverage and exact smoke** | **PARTIALLY-CLOSED** | Tests now use a temporary output root at [fa_orbit_train_guardtests.sh:46](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:46), exercise classification at [fa_orbit_train_guardtests.sh:158](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:158), checkpoint preflight at [fa_orbit_train_guardtests.sh:186](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:186), and submitter derivation at [fa_orbit_train_guardtests.sh:234](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:234). The worklog reports 52 guards and 76 pytest cases green at [fa_orbit_worklog.md:90](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:90), but the only committed guard log is the obsolete 33-case run at [fa_orbit_2026-08-05_18-29-31_guardtests.log:41](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_2026-08-05_18-29-31_guardtests.log:41). Lock recovery, final-status tee, environment/W&B and watcher behavior are not exercised directly. Most importantly, the exact `train.py`/W&B/checkpoint SMOKE remains pending, as expected. |
| **N9 — argv escape hatch and duplicates** | **CLOSED** | Additions have exact literal values at [fa_orbit_train.sbatch:251](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:251), and duplicate flags are rejected at [fa_orbit_train.sbatch:263](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:263). |
| **N10 — explicit gradient-checkpointing leaf** | **CLOSED** | The runtime gate requires exactly the two expected ViT IDs, key presence, and literal `True` at [fa_orbit_train.sbatch:187](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:187). This is the correct post-pivot equivalent of the original literal-`False` request. |

## `awk` OFMT diagnosis and replay

**Diagnosis verified.** The old `print v+0` applies AWK’s default `OFMT=%.6g`, producing values such as `1.78597e+09`. The following AWK invocation then interprets that rounded value as `1785970000`, so the liveness comparison measures thousands of seconds of formatting error. The committed fix uses `printf "%.6f"` at [p0_profile.sbatch:286](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:286), preserving sufficient epoch precision.

The three real replay cases confirm the worklog’s stated true gaps:

| Job | Complete ticks | Last CSV timestamp | Recorded train end | True gap |
|---|---:|---:|---:|---:|
| VAN 8×8 `3638700` | 76 × 8 = 608 rows | [CSV:608](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_VAN_8x8_2026-08-05_19-59-57_jid3638700_vram.csv:608) | [Slurm:193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_p0_p0-VAN_8x8-w6_3638700.out:193) | 0.552585 s |
| FA1 8×8 `3638701` | 68 × 8 = 544 rows | [CSV:544](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_FA1_8x8_2026-08-05_20-00-57_jid3638701_vram.csv:544) | [Slurm:193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_p0_p0-FA1_8x8-w6_3638701.out:193) | 0.163680 s |
| C4L 8×8 `3638702` | 91 × 8 = 728 rows | [CSV:728](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_C4L_8x8_2026-08-05_20-01-57_jid3638702_vram.csv:728) | [Slurm:193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_p0_p0-C4L_8x8-w6_3638702.out:193) | 0.331248 s |

Thus the `valid=0` results were false negatives caused by precision loss, not dead pollers.

## Config pivot

The requested config contract is enforced correctly:

- The deep diff permits only `training.frame_avg_angles` for FA1/C8/C16/C32 and no parsed differences for C4L at [test_exp11_orbit_configs.py:131](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:131).
- Both ViT leaves must be literal `True`, not merely truthy, at [test_exp11_orbit_configs.py:153](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:153), with `1`, `1.0`, `False` and `0` mutation coverage at [test_exp11_orbit_configs.py:178](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:178).
- C4L byte identity is asserted at [test_exp11_orbit_configs.py:196](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:196) and independently verified in this review with `cmp`.
- Orbit values, float types, spacing, exact column shifts, patch alignment and filename cardinality remain checked at [test_exp11_orbit_configs.py:206](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_orbit_configs.py:206).

There is no hole in the five-config test itself. The hole is in downstream P0 consumers, covered by NEW-1.

## New findings

### NEW-1 — BLOCKING: the official P0 pipeline was not fully pivoted to uniform gradient checkpointing

The official matrix still submits VAN at every rung and a separate CKPT4 cell at [p0_submit_matrix.sh:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:127). `p0_profile.sbatch` maps VAN to the canonical vanilla config at [p0_profile.sbatch:57](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:57), then requires literal `gradient_checkpointing: true` for every family at [p0_profile.sbatch:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:169). The canonical VAN ViT configs have no such leaf at [FLAC_AR.json:55](/n/fs/gatrdp/codespace/FLAC/src/configs/model_configs/FLAC/AR/FLAC_AR.json:55) and [FLAC_AR.json:71](/n/fs/gatrdp/codespace/FLAC/src/configs/model_configs/FLAC/AR/FLAC_AR.json:71). Every official VAN cell will therefore abort before profiling.

The collector is also still semantically pre-pivot: it treats C4L as checkpointing-off and CKPT4 as checkpointing-on at [p0_collect.py:540](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:540), and renders a false “disabling checkpointing” comparison at [p0_collect.py:684](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:684). C4L and CKPT4 are now the same checkpointed recipe. The stale assumption is still positively encoded in [test_exp11_p0_collect.py:388](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_p0_collect.py:388).

**Fix:** add a reviewed checkpointed VAN P0 config, or remove VAN from the promised matrix; remove/relabel the now-redundant CKPT4 comparison; update submitter cardinality, collector semantics/rendering, and tests before the official P0 launch.

### NEW-2 — BLOCKING: the test-only `OUTPUT_ROOT` hook remains active in real Slurm jobs

The submitter propagates ambient `OUTPUT_ROOT` into the allocation at [fa_orbit_submit.sh:81](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:81). The launcher honors it without restricting Slurm jobs at [fa_orbit_train.sbatch:98](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:98), and derives the real save root, lock and restart namespace from it at [fa_orbit_train.sbatch:123](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:123) and [fa_orbit_train.sbatch:135](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:135).

An exported typo or stale testing value can redirect an official arm to an arbitrary directory, defeating the pinned output identity and creating a separate lock/lineage.

**Fix:** force and validate `OUTPUT_ROOT=outputs_FLAC` whenever `SLURM_JOB_ID` is set; honor the override only for non-Slurm guard dry-runs. The submitter should export the fixed value, not ambient state.

### NEW-3 — NIT: the submitter records provenance only after `sbatch` accepts the job

`sbatch` runs at [fa_orbit_submit.sh:93](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:93); manifest creation begins only afterward at [fa_orbit_submit.sh:97](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh:97). A local manifest failure returns nonzero while the job remains queued/running.

**Fix:** atomically publish an intent manifest containing the exact command before submission, then append/rename with the returned job ID; alternatively cancel the exact submitted job if publication fails.

### NEW-4 — NIT: FIFO creation uses race-prone `mktemp -u`

[fa_orbit_train.sbatch:471](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:471) selects a nonexistent name without creating it, then calls `mkfifo`.

**Fix:** create a temporary file with `mktemp`, remove that exact file, create the FIFO immediately, and include FIFO removal in the exit trap.

### NEW-5 — NIT: the poller comment overstates replay uniformity, and the new tolerance is looser than the evidence

[p0_profile.sbatch:292](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:292) says all three cells had `76x8 = 608` rows. They actually had 76, 68 and 91 complete ticks. The replay proves a maximum true gap of 0.553 seconds; changing the old two-second tolerance to five seconds at [p0_profile.sbatch:296](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:296) is not supported by that replay.

**Fix:** correct the comment and retain the two-second bound, or document measured evidence justifying five seconds.

## Validation performed

- Shell syntax passed for the changed launch/guard/P0 scripts.
- Python AST parsing passed for the new Python modules, collector and config test.
- Source/config-only `git diff --check` passed.
- Full-commit `git diff --check` reports trailing whitespace only inside committed raw Slurm output artifacts.
- C4L is byte-identical to exp_07 `FLAC_AR_BF.json`.
- No tests or GPU/Slurm jobs were executed because this was a read-only review.

## Launch preconditions

1. Fix NEW-1 and NEW-2, plus the unresolved B2/B3/B5/B7 code paths; update tests for each fix.
2. Commit and persist a fresh green result for the rebuilt 52-case guard suite and relevant pytest set.
3. Run the **fixed uniform-grad-checkpointing official P0 matrix**. All required rows must be provenance-valid; VAN semantics and collector labels must match the pivot.
4. Produce and review the official P0 report, including the selected common rung, C16/C32 spot evidence, throughput, VRAM margins and justified wall limits.
5. Land a dedicated pin commit replacing every `TO-PIN-AFTER-P0` value: rung, MB, GPU count, VRAM floor(s), per-arm time limits and exact P0 manifest SHA-256. Keep `MAXSTEPS=40000` and checkpoint cadence 2,500.
6. Obtain focused reviewer sign-off on that pin commit against the official report.
7. Run the exact multi-GPU `SMOKE=1` path at the selected rung and preserve evidence of:
   - the expected number of distinct ranks/devices;
   - literal grad-checkpointing-on config;
   - one actual W&B run with verified entity/project/name/ID;
   - completion of the bounded optimizer-step budget;
   - one checkpoint directory containing a readable full-state checkpoint with embedded config, optimizer, scheduler and EMA;
   - identical durable logs, duplicated manifest and successful final classification;
   - no stale lock, watcher or torchrun processes afterward.
8. Obtain reviewer sign-off on the SMOKE evidence.
9. Commit/push the final reviewed SHA and record SOP acceptance criteria and exact commands before submitting C4L, C8, C16 and C32.

**No arm may launch until all nine conditions are satisfied.**
