# Code review — exp_11 round 3 (arm training launcher, commit 72a8114)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable, `max_user_namespaces=0`); read-only instruction, tree verified clean post-review · **Date:** 2026-08-05 · *(reviewer's self-identification line below retained verbatim)*

# Code review — exp_11_fa_orbit Coder Round 3

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only review) · **Date:** 2026-08-05 · **Commit:** `72a811416b35c4bfb9282c5ce1c4370776329eb9`

## Verdict

**REJECT — 8 BLOCKING, 2 NIT**

The `torchrun`/Lightning DDP core is sound, but this commit is not safe for multi-day launches. In particular, the P0-selected recipe is not pinned, restart lineage is substantially weaker than exp_10, the watchdog can cancel a legitimate startup, and critical provenance/log failures do not abort training.

Read-only static checks passed: both shell files parse, the Python gate parses, and `git diff --check` is clean. No GPU/Slurm execution was performed during this review.

## Findings

1. **BLOCKING — The launcher is still a pre-P0 scaffold, not the literally pinned post-P0 recipe required by §10.**

   The approved plan requires one P0 winner shared by every arm and then “literally pinned in the launch script” ([plan_fa_orbit.md:78](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/plan_fa_orbit.md:78), [plan_fa_orbit.md:84](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/plan_fa_orbit.md:84)). At this commit, the notebook records only that the matrix was launched and collection/rung selection remained pending ([fa_orbit_worklog.md:69](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_worklog.md:69)).

   Nevertheless, the launcher admits either `16x4` or `8x8`, arbitrary positive `MAXSTEPS`, an arbitrary `MIN_FREE_MB`, and a manually supplied wall limit ([fa_orbit_train.sbatch:18](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:18), [fa_orbit_train.sbatch:79](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:79), [fa_orbit_train.sbatch:96](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:96)). A mixed-rung sweep changes rank count, sampler partitioning and rank/worker augmentation seeds; it is not the registered orbit-only comparison even though global and SyncBN batch remain 64. `MAXSTEPS=1` also passes the full-launch contract despite the registered 40,000-step estimand.

   **Fix:** after P0 collection, commit the selected rung, exact MB/GPU count, `MAXSTEPS=40000`, checkpoint cadence, per-arm P0-derived VRAM threshold, and reviewed wall limits. Restore the planned submission wrapper so operators do not hand-assemble resource flags. Bind every value to the exact P0 manifest/report SHA and reject mismatched Slurm CPU, GPU, memory and time allocations inside the job.

2. **BLOCKING — The “exp_10-style” restart gate omits the checks that made exp_10 fail-closed.**

   The new gate accepts any regular file anywhere below the broad arm save root and checks only the operator’s claimed step against `MAXSTEPS` ([fa_orbit_train.sbatch:175](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:175)). It never loads the checkpoint or verifies:

   - embedded `global_step == EXPECTED_STEP`;
   - embedded model config equals the exact arm config;
   - full optimizer and scheduler state;
   - EMA state;
   - original rung, target budget or launch manifest;
   - checkpoint SHA-256.

   The guard suite actually declares a zero-byte synthetic checkpoint to be a valid restart ([fa_orbit_train_guardtests.sh:89](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:89)). This is materially weaker than exp_10’s `torch.load` preflight and step/config/optimizer/scheduler/EMA checks ([bf_resume_launch.sh:170](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch.sh:170)).

   A renamed C4 checkpoint copied under the C16 root would pass and likely load because orbit size does not change the module tree. A checkpoint already beyond `MAXSTEPS` can also produce Lightning’s exact “max_steps reached” message without doing another optimizer step because PL stops when `global_step >= max_steps` ([fit_loop.py:165](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:165)).

   **Fix:** restrict the resolved path to the exact `RUNDIR/checkpoints/` tree, load it on CPU, require exact global step and parsed model-config equality, and require one nonempty optimizer state, scheduler state and EMA keys as exp_10 does. Bind the restart to the original manifest’s rung, target and commit; pin `MAXSTEPS=40000`; hash the checkpoint into the restart manifest.

3. **BLOCKING — Run-directory ownership is race-prone; duplicate jobs can concurrently write one checkpoint tree.**

   Initial mode performs a non-atomic existence check early in preflight ([fa_orbit_train.sbatch:169](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:169)), but the directory is created later by W&B/ModelCheckpoint. Two queued copies of the same arm can both observe absence and then train concurrently into the same fixed `RUNDIR`. Restart mode has no active-job lock at all.

   This can overwrite identically named step checkpoints and make lineage unrecoverable.

   **Fix:** atomically reserve the exact run directory for INITIAL mode and acquire one arm/run lock for RESTART mode before expensive gates. Store job ID and launch UUID in the lock. Stale-lock recovery must explicitly verify that the recorded Slurm job is no longer active.

4. **BLOCKING — The 300-second watchdog can cancel a legitimate startup and bypass all classification/cleanup.**

   The timer starts before `torchrun`; its window includes Python imports, dataset discovery, model/DINO/VAE construction and rank-zero W&B initialization. The P0 smoke used `p0_runner.py` with `LOGGER=none`, so its 60-second result does not establish a cold-start bound for this path. `WATCHDOG_SEC` is also inherited through `--export=ALL` without numeric or lower-bound validation.

   At the deadline, an absent message causes `scancel` of the whole allocation ([fa_orbit_train.sbatch:357](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:357)). If correct DDP registration occurs immediately after the grep, the valid run is still cancelled. Because `scancel` kills the batch shell and pipeline, sections N and the final status line normally never run.

   The hardcoded `torchrun --nproc_per_node=N` plus PL’s TorchElastic world-size validation already prevents the earlier P0 world-size-one failure mode.

   **Fix:** remove the absence timer unless a cold-cache/W&B startup bound is measured. Prefer immediate rejection of an observed wrong world-size message plus the post-hoc gate. If a rendezvous timeout remains necessary, use a conservatively measured limit, validate it, terminate the `torchrun` process group rather than `scancel` the allocation, and let the parent emit an explicit watchdog classification.

5. **BLOCKING — Manifest/log durability fails open, and the advertised exit taxonomy is not implemented.**

   The script uses `set -uo pipefail`, not `set -e`. Manifest creation and publication are unchecked, so a write or `mv` failure does not prevent training ([fa_orbit_train.sbatch:329](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:329)). The training pipeline records only `PIPESTATUS[0]`, discarding a nonzero `tee` status ([fa_orbit_train.sbatch:375](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:375)). GNU `tee` can continue consuming input after one destination fails, allowing days of training without the promised duplicate log.

   Classification is also incomplete:

   - a normal torchrun OOM returns nonzero, so the `rc=3` assignment guarded by `rc == 0` is normally unreachable;
   - missing world-size evidence becomes `rc=6` only when torchrun returned zero;
   - watchdog cancellation produces no launcher exit classification;
   - tee/manifest failure has no class.

   **Fix:** explicitly check directory creation, environment hashing, manifest write/publication and both tee destinations. Capture every pipeline status. Define precedence while preserving the raw torchrun code, e.g. world-size invalid, watchdog, OOM, missing completion, log/provenance failure, generic runtime failure. Duplicate the complete preflight/final-status record—not only torchrun output—to durable storage.

6. **BLOCKING — The runtime and external training artifacts are recorded but not identity-gated.**

   The launcher’s correctness depends specifically on PL 2.1.0 behavior, yet it has no version gate. `pip freeze` is merely hashed into the manifest; any digest is accepted ([fa_orbit_train.sbatch:333](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:333)). A shared-environment update while jobs wait in queue could change cluster election, rank behavior, checkpoint broadcast or the completion literal. This is exactly why exp_10 gates PL 2.1.0 before launch.

   The external `weights/FLAC/VAE.safetensors` is not tracked by Git and is neither pinned nor hashed. Restart checkpoints likewise lack a manifest hash, despite §5 requiring config/checkpoint hashes.

   **Fix:** require the reviewed Python executable and exact PL version, plus a P0/cohort-bound environment digest or explicit torch/PL pins. Hash and verify the VAE file and every restart checkpoint before launch. Record driver/CUDA versions and require the same reviewed environment contract for every arm.

7. **BLOCKING — W&B is rank-safe, but online destination and run lineage are not fail-closed.**

   The parent gate verifies only `wandb.Api().viewer.email` ([fa_orbit_train.sbatch:312](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:312)). Because submission uses `--export=ALL`, ambient `WANDB_MODE`, `WANDB_DISABLED`, `WANDB_ENTITY`, `WANDB_RUN_ID` or `WANDB_RESUME` can redirect, disable, reuse or offline the actual logger while the account check still succeeds. The manifest records only project/display name, not the actual entity and run ID ([fa_orbit_train.sbatch:350](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:350)).

   **Fix:** scrub all W&B control variables, explicitly set online mode and the approved entity, and create a collision-proof run ID recorded before `torchrun`. Define restart behavior explicitly: either resume the original W&B ID with `resume=must`, or create a new lineage-linked ID. Require rank zero to emit and verify the actual entity/project/name/ID.

8. **BLOCKING — The guard evidence does not cover the fail-closed branch inventory or the exact risky runtime path.**

   The 33-case log is green, but DRYRUN exits before Slurm, GPU, VRAM, W&B, init identity, manifest, watchdog, torchrun and post-hoc classification. The two “real” cases abort at commit/Slurm checks; contrary to the test header, no valid invocation reaches the impossible VRAM threshold. There is no live smoke of `train.py` with torchrun, W&B and ModelCheckpoint—the P0 smoke used the logger-free, checkpoint-free `p0_runner.py`.

   The guard itself is unsafe: it temporarily overwrites a tracked production config without a restoration trap and conditionally executes `rm -rf outputs_FLAC/exp11_C8` after a race-prone existence check ([fa_orbit_train_guardtests.sh:89](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:89), [fa_orbit_train_guardtests.sh:126](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh:126)).

   **Fix:** test validators against in-memory or temporary-root inputs; never alter tracked configs or production output prefixes. Add mocked tests for every downstream failure and exit class. Before launch, run a reviewed multi-GPU smoke of the exact `train.py` path proving one W&B run, N distinct ranks/devices, the expected single checkpoint directory, a readable checkpoint with model config/optimizer/scheduler/EMA, dual durable logs, and clean final classification.

9. **NIT — The current exp_07 reference argv is faithful, but the parity checker has an unnecessarily broad escape hatch.**

   The embedded tokens match exp_07’s actual command, including its intended 67,500-step target. The current explicit additions also equal the corresponding defaults. However, any future new flag is accepted merely because its value equals the mutable current `defaults.ini`, despite the stated “enumerated differences only” contract ([fa_orbit_train.sbatch:245](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:245)). `as_map` also silently collapses duplicate flags.

   **Fix:** whitelist the exact default-restating additions and expected values (`num-nodes`, `precision`, `val-every`, empty val config, gradient clip); reject all other added flags and duplicate flags.

10. **NIT — The launch semantic gate treats a missing gradient-checkpointing key as explicit `false`.**

    `get("gradient_checkpointing", False)` admits an absent leaf ([fa_orbit_train.sbatch:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:140)). The committed configs are currently protected by the round-1 deep-diff tests, so this does not change this commit’s recipe, but the runtime gate is weaker than its message.

    **Fix:** require the key to exist and its value to be literally `False`, and require the exact two expected ViT conditioner IDs.

## Verified critical behavior

- `torchrun --standalone --nproc_per_node=N` is coherent under the one-task sbatch allocation. PL 2.1.0 prioritizes TorchElastic over Slurm ([accelerator_connector.py:417](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:417)); TorchElastic consumes `RANK`, `LOCAL_RANK` and `WORLD_SIZE` and validates `devices × nodes == WORLD_SIZE`. The selected SLURM-variable unsets do not create a self-spawn path.

- W&B does not create one run per rank. `RANK` exists before imports; `rank_zero_experiment` returns a dummy object on nonzero ranks ([logger.py:102](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/lightning_fabric/loggers/logger.py:102)). `logger.watch` is not itself decorated as the launcher comment claims, but its call through the dummy experiment is a no-op. `push_wandb_config` likewise skips nonzero ranks.

- Rank zero resolves the intended checkpoint path; nonzero ranks initially have `dirpath=None`, after which `ModelCheckpoint.setup` broadcasts rank zero’s resolved path to all ranks ([model_checkpoint.py:264](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:264)).

- Unsetting `SLURM_PROCID` makes every rank seed model initialization with 42. PL later incorporates global rank into dataloader-worker seeding, so ranks receive deterministic but distinct worker RNG streams. This preserves cross-arm identity only if every arm uses the same pinned rung.

- EMA updating on every rank is appropriate: DDP synchronizes the online model’s gradients/weights, and every rank applies the same deterministic EMA update before rank zero checkpoints. No separate EMA all-reduce is required.

- C4L is the correct internal initialization reference. The exp_11 configs are already deep-diff-gated against historical B-F, while C4L is the contemporaneous bridge used for the sweep’s estimand.

## Coder deviations 1–5

| # | Deviation | Judgment |
|---|---|---|
| 1 | One self-contained sbatch instead of the planned submission wrapper plus sbatch | **Conditionally acceptable structure, but not acceptable as delivered.** Without the wrapper, critical Slurm resources are hand-entered and unvalidated; covered by BLOCKING 1. |
| 2 | Both surviving rungs remain selectable pending P0 | **Not acceptable for launch.** §10 requires one literal winner shared by all arms; covered by BLOCKING 1. |
| 3 | `MIN_FREE_MB` and wall time remain operator inputs | **Not acceptable for launch.** Omission fails closed, but a wrong low value permits an avoidable OOM and a wrong high value/time rejects or truncates a valid run. Pin them from the reviewed P0 report. |
| 4 | “The two-model seeded init-identity build is not executed at launch” | **The description is incorrect.** The real launcher invokes the gate inside the allocation ([fa_orbit_train.sbatch:326](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:326)); C8/C16/C32 each build the requested arm and C4L ([assert_arm_configs_exp11.py:149](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:149), [assert_arm_configs_exp11.py:159](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py:159)). C4L’s one-build self-reference is acceptable. The gate should remain inside the allocation. |
| 5 | Added a 300-second `scancel` watchdog beyond the approved machinery | **Not acceptable in its present form.** Its bound is unproven for the exact W&B/checkpoint path and cancellation bypasses classification; covered by BLOCKING 4. |

**Launch authorization: none. Do not submit C4L, C8, C16 or C32 from commit `72a8114`.**
