# Code review — exp_12 mem_probe round 1 (`mem_probe.sbatch` + config copy)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` — bwrap namespaces unavailable on this host (`max_user_namespaces=0`); reviewer instructed read-only, tree verified clean post-review · **Date:** 2026-08-05 · *(the reviewer's own self-identification line below is retained verbatim)*

**Reviewer:** OpenAI Codex (GPT-5, Codex workspace/API agent, read-only review) · **Date:** 2026-08-05

# Code review — exp_12 memory probe

Read-only review; no files were modified.

The JSON copy is byte-identical to canonical `FLAC_AR.json` at SHA-256 `f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9`. At the reviewed snapshot, `train.py` resolves one device with `strategy="auto"` to the single-process path, omits SyncBN, uses `bf16-mixed`, accumulation 1, EMA enabled, and no ViT checkpointing. `--max-steps 5` reaches the Trainer correctly. Bash syntax and JSON parsing pass. The script contains no operation against job 3637217.

## Findings

1. **BLOCKING — The VRAM artifact does not fail closed or prove that its peak belongs to exactly the allocated probe GPU.**  
   [mem_probe.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch:69) queries every GPU row returned by `nvidia-smi`, then takes the maximum across them. This cluster currently has `ConstrainDevices=yes`, so one row is expected, but the script never asserts that invariant or records a stable UUID. Polling errors are discarded, and an empty CSV makes the `awk` expression report `0`, after which a successful training return code still produces overall success. The denominator is also hard-coded rather than bound to the sampled device.  
   **Concrete fix:** synchronously resolve and assert exactly one visible L40 UUID, query only that UUID, truncate a job-unique CSV before polling, make sampler/query failure fatal, validate that every retained row has that UUID and numeric memory fields, require at least one sample, and obtain `memory.total` dynamically from the same UUID. A missing/invalid measurement must override `rc=0` and fail the job.

2. **BLOCKING — The canonical-config gate is relative and its semantic checks are bypassable.**  
   [mem_probe.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch:35) only checks that the probe copy equals the current canonical file. If both drift together, the gate passes; the documented expected hash is printed but never enforced. The Python gate uses `assert`, which disappears when inherited `PYTHONOPTIMIZE` is enabled.  
   **Concrete fix:** require both files to equal the reviewed literal SHA-256 above, unset/refuse `PYTHONOPTIMIZE`, and replace operational `assert` statements with explicit conditional failures. Explicitly verify `training.use_ema is True` alongside vanilla conditioning and both checkpointing values.

3. **BLOCKING — A queued job is not bound to the reviewed code/defaults, and several defining settings remain mutable defaults.**  
   [mem_probe.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch:31) merely prints a short HEAD. The experiment files are currently untracked, and the shared checkout can change while the job waits. Precision, strategy, node count, SyncBN state, validation state, and fresh-start state are not all explicit on the command line. They are correct in the current [defaults.ini](/n/fs/gatrdp/codespace/FLAC/defaults.ini:23), but could drift silently before execution.  
   **Concrete fix:** commit and push the reviewed/fixed round, require a full `EXPECT_FLAC_SHA` at submission, and fail unless runtime HEAD matches it and the relevant tracked surfaces are unchanged. Pass at least `--num-nodes 1 --strategy auto --precision bf16-mixed --sync-batchnorm false --val-every -1 --val-dataset-config '' --ckpt-path '' --pretrained-ckpt-path '' --gradient-clip-val 0.0`; add `#SBATCH --nodes=1 --ntasks=1` and runtime assertions for one task, one node, and one GPU.

4. **BLOCKING — The W&B fallback changes the instrumented execution path.**  
   [mem_probe.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch:55) silently continues with `logger=none` after identity or availability failure while claiming the measured configuration is unchanged. In [train.py](/n/fs/gatrdp/codespace/FLAC/train.py:164), W&B enables `logger.watch(training_wrapper)`; `none` does not. Thus the fallback changes hooks and potentially the measured peak, and it eliminates the mandated isolated W&B run.  
   **Concrete fix:** keep `LOGGER=wandb` fixed and abort on identity/availability failure, matching the prior launcher convention. Set an experiment-specific `WANDB_DIR` if local W&B artifacts must also be isolated.

5. **BLOCKING — OOM reporting and successful completion are not synchronized or asserted.**  
   Global logging uses asynchronous process substitution at [mem_probe.sbatch](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch:29), but the script greps that same file immediately after `train.py` exits. There is no guarantee `tee` has flushed the terminal OOM traceback before the grep. Conversely, `rc=0` is accepted without confirming that Lightning reported reaching optimizer step 5.  
   **Concrete fix:** run training through a synchronous `python … 2>&1 | tee …` pipeline, preserve `${PIPESTATUS[0]}`, then stop/wait for the sampler and validate the artifacts. Accept success only with `rc=0`, a valid peak, and the `max_steps=5 reached` completion marker. Classify nonzero exits as CUDA OOM only when the fully flushed log contains the exact error; otherwise report a distinct non-OOM failure.

6. **NIT — Run artifacts are collision-prone and stale output is not rejected.**  
   Logs and CSVs use second-resolution timestamps without the Slurm job ID, use append redirections, and the fixed save/W&B names can be reused by an accidental duplicate submission.  
   **Concrete fix:** include `${SLURM_JOB_ID}` in every per-run artifact name, truncate newly created measurement files, and fail if the intended output/run directory already exists.

7. **NIT — Launch-stage documentation currently claims a launch that has not happened.**  
   [mem_probe_params_set_up.md](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe_params_set_up.md:1) says “written at launch,” while the worklog still says review pending and no job ID or command artifact exists. The acceptance criterion also does not state an unambiguous five-step success condition or full worker SHA.  
   **Concrete fix:** mark it as a pre-launch draft now; at actual submission, add the exact `sbatch` command and job ID to `mem_probe_command.md`, record the full worker SHA, and require either all five optimizer steps or a fail-honest OOM with exact error and validated peak.

## Verdict

**REJECT** — the intended training path is presently correct, but the measurement/provenance gates can still yield an untrustworthy or configuration-dependent peak.
