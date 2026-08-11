**Reviewer:** OpenAI Codex (gpt-5.6-sol, codex-cli 0.147.0, `codex exec`, GitHub-connector read-only — local sandbox unavailable on della) · **Date:** 2026-08-11
**Round:** code round B (della Slurm kit: eval/train/submit) · **Verdict:** REVISE

> Invocation note (Planner): GitHub-connector review of `Yixun-Hu/FLAC` @ `della-flac-chequity`, commits `60edd0a` + `85a928b` (HEAD `85a928b` at review time). 141,557 tokens.

### BLOCKING

1. Full training remains vulnerable to inherited recipe overrides. The wrapper uses `--export=ALL`, while the full-run path accepts ambient `MAXSTEPS`, `VAL_EVERY`, and `CHECKPOINT_EVERY`. An exported login-shell value can silently change the 67,500/2,500/2,500 recipe; invalid values instead waste an allocation on argument parsing. Direct documented `sbatch` use can also inherit `SMOKE` or `DRYRUN`. Make full-run constants literal and reject conflicting exports, or stop exporting the ambient environment. `della_submit.sh:119-135`; `della_train.sbatch:96-109,153-168`.

2. The resume search is not isolated from smoke or other layouts. For a full run, `SAVEDIR` is the parent of `${BASE_SAVEDIR}/smoke`, and `"$SAVEDIR"/**/*.ckpt` recursively accepts every checkpoint at every depth. Therefore any smoke/debug/foreign `.ckpt` can win by mtime; there is no canonical-directory, global-step, config, or EMA-state validation. Discovery errors from `ls` are also suppressed and unchecked, potentially turning "resume lookup failed" into "start from scratch." Search only the exact production checkpoint directory, keep smoke as a sibling rather than a child, fail on discovery errors, and validate checkpoint identity before adding `--ckpt-path`. `della_train.sbatch:70,77-87,95-106,139-144`.

3. Submission and command recording are not an atomic, fail-closed transaction. `sbatch` succeeds before the job ID is validated or the record is opened/appended. A parse failure, disk-full condition, permission error, or partial append leaves a live job with no valid required record; no cancellation occurs. Conversely, failed `sbatch` itself correctly writes nothing. Prefer `sbatch --hold --parsable`, append under a lock, then release; cancel or retain the job held if recording fails. `della_submit.sh:146-176`.

   There is also a lifecycle conflict: once `della_vanilla_repro_command.md` is committed, every successful submission immediately dirties that tracked file before the queued worker runs, causing its clean-tree gate to fail after acquiring a GPU. The first wave only avoids this because the record does not yet exist at HEAD and is therefore untracked. The record location or code-pinning strategy must make repeated submissions compatible with the tracked-clean gate. `della_submit.sh:36,155-176`; `della_eval.sbatch:102-106`; `della_train.sbatch:120-124`.

4. Tee durability is checked only after the expensive workload completes. GNU `tee` can fail to open or later write the experiment log while continuing to copy output to stdout; `main` can consequently run the entire eval or multi-day training job before rc 8 is noticed. Pre-open the log successfully before invoking `main` and arrange for a runtime tee failure to terminate the producer rather than merely change the final exit status. `della_eval.sbatch:178-181`; `della_train.sbatch:205-208`.

5. DRYRUN is not write-free as specified. The submit dry run executes `git fetch`, which writes `FETCH_HEAD` and potentially remote-tracking metadata, while suppressing fetch failure and possibly comparing against stale state. The sbatch dry runs invoke Python without `-B`/`PYTHONDONTWRITEBYTECODE`, so imports can create bytecode caches after the clean-tree check. Use a non-mutating remote query such as `git ls-remote`, disable optional Git index refreshes, and run provenance imports with bytecode writes disabled. `della_submit.sh:102-107,141-143`; `della_eval.sbatch:113-147,174-176`; `della_train.sbatch:128-176,201-203`.

### NON-BLOCKING

1. The detailed GPU disclosure command is not gated: failure inside `echo "$(nvidia-smi …)"` is masked by `echo`, allowing the run to proceed without recording model/memory/UUID. Capture it into a checked variable. `della_eval.sbatch:158`; `della_train.sbatch:192`.

2. Eval exports `WANDB_DIR` but does not create or verify it; only training does. This is currently harmless because eval does not initialize W&B and the worklog records prior creation, but the shared runtime contract would be more self-contained if the directory were ensured for real runs. `della_eval.sbatch:53-57`; `della_train.sbatch:182-183`.

3. The train banner always includes `(SMOKE=0)` because parameter expansion with `:+` tests non-emptiness and `"0"` is non-empty. Cosmetic only. `della_train.sbatch:63,91`.

4. The submit-time format check accepts values such as `99:99:99`; Slurm will reject them before recording, so this does not corrupt a run, but validating minute/second ranges would produce a clearer usage failure. `della_submit.sh:78-80`.

The core protocol mapping is otherwise strong: all three eval cells bind the correct config/seed/count and exact permitted argv; offline variables precede Python; the hardened GPU-count and single-task gates are correct; array execution and `PIPESTATUS`/`TRAIN_RC` propagation are sound; effective seed, GPU identity, resume disclosure, full argv, and jid-suffixed logs are present. The A100-80GB train constraint, unconstrained A100 eval, absence of explicit partition, and time-derived QOS are consistent with the amended della plan.
