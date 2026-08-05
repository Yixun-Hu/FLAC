# Code review — exp_11 round 2 (P0 profiling kit, commit e566513)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap namespaces unavailable on this host, `max_user_namespaces=0`); reviewer instructed read-only, tree verified clean post-review · **Date:** 2026-08-05 · *(reviewer's own self-identification line below retained verbatim)*

# Code review — exp_11_fa_orbit, Coder Round 2

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only review) · **Date:** 2026-08-05 · **Commit:** `e566513f7d098d180f129749bfaec93f7447b6ff`

**Verdict: REJECT — 7 BLOCKING, 2 NIT**

Read-only review; no files were modified. `bash -n`, `git diff --check`, and the collector tests passed (`18 passed`). These checks do not exercise the Slurm/DDP launch path.

## Findings

1. **BLOCKING — Every advertised multi-GPU cell is launched as one Slurm task, so it does not establish N-rank DDP or BN-64 semantics.**

   [p0_profile.sbatch:27](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:27) fixes `--ntasks=1`, while [the training command](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:124) directly runs `python train.py --num-gpus N`. In the installed Lightning 2.1.0, the presence of `SLURM_NTASKS` selects `SLURMEnvironment`, whose `creates_processes_externally=True` prevents Lightning’s DDP launcher from spawning the other local ranks. With only one Slurm task, the job therefore either fails on incomplete Slurm rank variables or operates with world size one; `--num-gpus N` alone cannot create the missing processes.

   Consequently, SyncBatchNorm would see only the micro-batch, not `micro × N = 64`, and the alleged 2/4/8-GPU scaling measurements could actually be one-GPU measurements with progressively smaller micro-batches. The VRAM gate does not catch this: it validates that all allocated UUIDs appear in `nvidia-smi`, not that each UUID hosts a training rank.

   **Concrete fix:** launch N processes explicitly, for example with `torchrun --standalone --nproc_per_node="$NGPU"` inside the one-task allocation, or request N Slurm tasks and use a coherent `srun` layout. Add a runtime gate proving world size N, exactly one rank per allocated UUID, and non-idle training allocation on every UUID. Validate the corrected path with a two-GPU smoke before running the matrix.

2. **BLOCKING — `WALL_FIT` is full-process wall time from two independent jobs, not elapsed time between optimizer steps 10 and 30.**

   The timer at [p0_profile.sbatch:123](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:123) starts before Python imports, model/VAE construction, DDP rendezvous, dataloader startup and the first batch, and stops after Trainer teardown. The submitter launches the 10- and 30-step runs as unrelated Slurm jobs at [p0_submit_matrix.sh:60](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:60), with no same-node, same-UUID, or same-cotenant guarantee.

   Thus

   \[
   W_{30}-W_{10}
   \]

   contains differences in two independent startups, first-ten-step warmups, dataloader scheduling, compilation/kernel selection and teardown. Those terms do not cancel merely because both commands execute nominally similar phases. They can dominate a twenty-step delta and reverse the selected rung or orbit-cost slope.

   **Concrete fix:** obtain monotonic timestamps at completed optimizer steps 10 and 30 within one 30-step fit, using a P0-only Lightning callback/runner so no library source change is needed. The independent 10-step run may remain as a startup diagnostic, but must not determine decision throughput. Bind timing to rank zero and synchronize CUDA before both timestamps.

3. **BLOCKING — The P0 contract’s GPU-utilization/power trace and conditional worker control are absent, so the kit cannot support its promised bottleneck attribution.**

   The poller records only UUID and memory at [p0_profile.sbatch:116](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:116), while workers are hard-coded to six at [p0_profile.sbatch:128](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:128). There is no utilization/power evidence with which to decide whether the residual is GPU compute, communication, input starvation, VAE/DiT work, or optimizer overhead, and no mode for the required zero-versus-six-worker spot check.

   The collector nevertheless labels its fitted intercept “everything else in the step” and renders it as “residual step cost.” That is a decomposition, not a bottleneck diagnosis.

   **Concrete fix:** add timestamped, UUID-bound `utilization.gpu` and `power.draw` samples over the actual step-10→30 window; summarize them per UUID. Add a conditional worker-pair submission mode at the selected rung. Until those measurements exist, label the residual “unattributed” and do not issue a definitive bottleneck report.

4. **BLOCKING — Collection is not bound to a submission manifest and can silently omit or mix cells.**

   The submitter writes a manifest at [p0_submit_matrix.sh:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:37), but the collector ignores it and scans every historical `slurm_p0_*.out` at [p0_collect.py:367](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:367). A cell for which both submissions fail produces no log and therefore no `INCOMPLETE` row at all. The collector can write a zero-cell or partial report and still return success at [p0_collect.py:410](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:410). It can also pair stale 10- and 30-step results from different launches or commits.

   Multiple result markers are reported as a problem but the last one is still admitted into calculations at [p0_collect.py:384](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:384). Submission failures likewise do not make the submitter exit nonzero.

   **Concrete fix:** create a collision-proof run ID and atomic manifest containing the exact expected cells, steps, job IDs and SHA. Pass that ID plus commit SHA, config hash and job ID into every `P0RESULT`. Require the collector to consume one explicit manifest, reject cross-run rows, emit every expected cell, and return nonzero before derived calculations if any expected half is missing, duplicated, malformed, pending or failed.

5. **BLOCKING — The config gate allows any cell label to use any allow-listed config.**

   [p0_profile.sbatch:89](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:89) checks only whether `MODEL_CONFIG` belongs to a six-path allow-list. It never binds `VAN_*` to canonical vanilla, `C4L_*` to C4L, `C8_*` to C8, or `CKPT4_*` to the checkpointed exp_07 config. `CELL` itself accepts arbitrary underscore-separated labels.

   A valid but mislabeled config would pass every gate and directly corrupt the orbit slope, grad-checkpointing cost and selected rung.

   **Concrete fix:** derive the expected config from the parsed cell family inside the sbatch script and require exact resolved-path equality. Also enforce the expected orbit and strict boolean checkpointing semantics, emit the config SHA-256, and have the collector verify it. Restrict spot rungs to `{32x2,16x4,8x8}`.

6. **BLOCKING — The attribution fit treats an underdetermined two-point line as a valid bottleneck model.**

   [orbit_pass_fit](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:213) accepts any two of VAN/C4L/C8. Two points necessarily produce a perfect line and `R²=1`, even though the missing third control is exactly what checks the assumed linear orbit-cost model. It then reports the extrapolated intercept as residual cost. Negative slopes/intercepts and non-finite values are also accepted; `float("nan")` passes parsing, bypasses the non-positive-delta comparison, and can propagate through every derived table.

   **Concrete fix:** require the exact `{VAN,C4L,C8}` set before reporting slope, intercept or R². Report a C8−C4L marginal contrast separately if desired, but do not call it a fitted residual model. Reject non-finite/non-positive walls and rates, invalid `valid` values, negative peaks, and physically implausible fit outputs; mark attribution ambiguous rather than interpreting them.

7. **BLOCKING — The multi-UUID poller can terminate or sample partially and still produce `valid=1`.**

   The background sampler is started at [p0_profile.sbatch:115](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:115), but its lifetime/exit status is discarded at [p0_profile.sbatch:138](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:138). Validation only requires that every expected UUID appeared at least once somewhere in the file. With no timestamp or sample identifier, it cannot prove that every polling interval contained exactly N UUIDs. A prematurely dead sampler with earlier valid rows can therefore understate the peak and still pass.

   This reintroduces the substance of exp_12 review finding B1 for the multi-GPU case.

   **Concrete fix:** record a sample sequence/timestamp on every poll, require exactly one row for every expected UUID in every retained sample, and terminate the sampler through a clean stop condition whose exit status is checked. Any premature sampler exit, partial sample, duplicate UUID or missing UUID must force `valid=0`.

8. **NIT — The completion marker is broader than the exact Lightning success message.**

   [p0_profile.sbatch:158](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:158) accepts any line containing `max_steps=N`, a non-digit, and later `reached`. Use the exact fixed substring `` `Trainer.fit` stopped: `max_steps=N` reached.`` and require the expected rank-zero occurrence. The shell, rather than a Lightning rank, emits `P0RESULT`, so rank duplication of that final marker is otherwise not a problem.

9. **NIT — The `rc=5` measurement-invalid branch is rendered as generic `FAILED`.**

   The sbatch correctly changes a clean training return to `rc=5` when VRAM measurement is invalid, but [summarize](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:172) checks generic nonzero `rc` before `valid==0`. The row is safely excluded from calculations, but its report loses the actual reason.

   **Concrete fix:** classify `rc=5` or `valid=0` as `INVALID` with a measurement-specific note before the generic failure branch.

## Verified points

- The requested matrix cardinality is correct: nine default paired cells plus the CKPT4 pair, with C16/C32 excluded from the default submission.
- The strategy and SyncBN CLI choices would reproduce global BN-64 at 16×4 and 8×8 **if** N ranks were actually launched.
- Resource arithmetic itself fits the stated nodes: the 8-GPU job requests 64 CPUs and 108 GiB, below 104 cores and approximately 503 GiB. Forty minutes is comfortably above the C8-at-32×2 planning estimate; a C32 spot at 32×2 is more borderline and should receive a measured, spot-specific limit.
- A C32 spot OOM that reaches `P0RESULT` is isolated by job-ID artifacts, rendered as `OOM`, and excluded from derived calculations; it does not contaminate other cells. A timeout or pre-marker death is unsafe only because of finding 4.
- The committed VAN, C4L, C8 and CKPT4 configs have the intended vanilla/orbit/checkpointing semantics. The defect is failure to bind those semantics to the submitted cell label.
- No arm launcher or library change was introduced in this round.
