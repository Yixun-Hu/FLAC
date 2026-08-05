# Code re-review — exp_11 round 2 fixes (commit ec0250d)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, reasoning effort xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** `--sandbox danger-full-access` (bwrap unavailable, `max_user_namespaces=0`); read-only instruction, tree verified clean post-review · **Date:** 2026-08-05 · *(reviewer's self-identification line below retained verbatim)*

# Code re-review — exp_11_fa_orbit round-2 fixes

**Reviewer:** OpenAI Codex (GPT-5, API invocation, read-only re-review) · **Date:** 2026-08-05 · **Commit:** `ec0250d294368eafdfaa953f059e17d6faa00284`

**Verdict: REJECT — 6 BLOCKING, 2 NIT**

Static checks, `bash -n`, `py_compile`, `git diff --check`, and the focused tests passed (`37 passed`). No GPU/Slurm smoke was performed.

## Prior findings

| # | Verdict | One-line evidence |
|---|---|---|
| 1 | **CLOSED** | N processes are now launched with `torchrun`, while Lightning elects TorchElastic before Slurm and maps `LOCAL_RANK` to one device ([p0_profile.sbatch:173](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:173), [accelerator_connector.py:420](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:420), [ddp.py:113](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/strategies/ddp.py:113)). |
| 2 | **CLOSED** | One fit records rank-zero step-10/30 timestamps after local CUDA synchronization, and PL checks `max_steps` only after batch-end hooks complete ([p0_runner.py:59](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_runner.py:59), [fit_loop.py:164](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:164)). |
| 3 | **PARTIALLY-CLOSED** | Utilization/power polling and a worker mode now exist, but the collector does not require the poller artifact and the worker pair is split into independent manifests ([p0_profile.sbatch:182](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:182), [p0_submit_matrix.sh:123](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:123), [p0_collect.py:333](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:333)). |
| 4 | **PARTIALLY-CLOSED** | Rows are bound by run/SHA/job/cell/config, but `maxsteps`, MB and GPU count are not compared and the seconds-resolution run ID can overwrite an existing manifest ([p0_collect.py:162](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:162), [p0_submit_matrix.sh:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:37)). |
| 5 | **CLOSED** | Legal cell grammar, derived config mapping, config hash, orbit semantics and strict checkpointing booleans are now gated ([p0_profile.sbatch:57](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:57), [p0_profile.sbatch:92](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:92), [p0_profile.sbatch:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:140)). |
| 6 | **PARTIALLY-CLOSED** | Exact VAN+C4L+C8 membership and implausible-fit marking are fixed, but parsing still admits non-positive `wall_fit` and other internally inconsistent result fields ([p0_collect.py:89](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:89), [p0_collect.py:364](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:364)). |
| 7 | **CLOSED** | Poller exit status, query failures, per-tick UUID completeness/uniqueness, sample count and end-of-training liveness are checked before `valid=1` ([p0_profile.sbatch:223](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:223), [p0_profile.sbatch:247](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:247)). |
| 8 | **CLOSED** | The gate now uses PL’s exact fixed completion string ([p0_profile.sbatch:297](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:297), [fit_loop.py:167](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py:167)). |
| 9 | **CLOSED** | `rc=5` or `valid=0` is classified as `INVALID` before generic nonzero return codes ([p0_collect.py:245](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:245)). |

## Runner and torchrun assessment

`p0_runner.py` follows `train.py` closely enough for profiling: identical argument parsing, seeding behavior under the same environment, dataloader/model/wrapper factories, VAE load, EMA construction, precision, accumulation, trainer factory and SyncBN wiring ([p0_runner.py:101](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_runner.py:101), [train.py:93](/n/fs/gatrdp/codespace/FLAC/train.py:93)). With accumulation one and this single-optimizer wrapper, `global_step` reaches 10 and 30 once each; the callback deduplicates marks and synchronizes before timestamping.

The multi-process launch itself is coherent: `torchrun --standalone` supplies the single-node rendezvous and TorchElastic rank variables; `devices=NGPU`, `num_nodes=1` matches `WORLD_SIZE`, Lightning does not spawn children, and each process selects `parallel_devices[LOCAL_RANK]` ([torchelastic.py:50](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/lightning_fabric/plugins/environments/torchelastic.py:50), [torchelastic.py:76](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/lightning_fabric/plugins/environments/torchelastic.py:76), [subprocess_script.py:96](/n/fs/gatrdp/envs/flac/lib/python3.10/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py:96)).

## New findings

1. **BLOCKING — The advertised `spot` and `workers` collectors always return “AMBIGUOUS” even when every job is valid.**

   These manifests contain no VAN+C4L+C8 triple, so `orbit_pass_fit()` returns `{}`; `attribution_ok({})` is false and `main()` consequently returns 2 ([p0_submit_matrix.sh:115](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:115), [p0_collect.py:404](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:404), [p0_collect.py:643](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:643)).

   **Fix:** record manifest mode and require an attribution fit only for matrix manifests that promise one. Successful spot/worker collections should return zero when all mode-specific expected rows are valid.

2. **BLOCKING — Utilization/power evidence is optional at collection time.**

   A missing or corrupt poller CSV merely causes `poller_summaries()` to omit the cell; it does not change `complete`, withhold derived results, or make the collector nonzero ([p0_collect.py:333](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:333), [p0_collect.py:641](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:641)). The sbatch validator also checks only memory syntax, allowing missing/non-finite utilization or power values to retain `valid=1` ([p0_profile.sbatch:252](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:252)).

   **Fix:** emit and bind the expected UUID set plus poller SHA-256; require the file, complete in-window UUID ticks, and finite utilization/power evidence per UUID. Otherwise classify the cell `INVALID` and return nonzero.

3. **BLOCKING — The manifest is neither collision-proof nor an exact execution binding.**

   `short-SHA + epoch-seconds` can collide between concurrent submissions, and plain `mv` overwrites the existing manifest ([p0_submit_matrix.sh:37](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:37), [p0_submit_matrix.sh:53](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:53)). Admission checks omit manifest `maxsteps` and cell-derived MB/NGPU; a row claiming `maxsteps=31`, `mb=8`, `ngpu=8` is admitted for a manifest expecting `C4L_32x2`, and reporting masks the mismatch by reconstructing shape from the label ([p0_collect.py:173](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:173), [p0_collect.py:242](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_collect.py:242)).

   **Fix:** use a random UUID or nanosecond-plus-random ID with no-clobber publication; validate full SHA/config hashes and exact `maxsteps`, MB, NGPU and worker count against each manifest row. Pin P0 to exactly 30 steps unless a separately named mode is approved.

4. **BLOCKING — The worker comparison is not a manifest-bound pair and worker count is not provenance-bound.**

   The two halves are separate one-cell manifests, while neither manifest nor `P0RESULT` carries `NUM_WORKERS` ([p0_submit_matrix.sh:127](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:127), [p0_profile.sbatch:320](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:320)). Worse, matrix mode exports `ALL` without explicitly setting the default, so an ambient exported `NUM_WORKERS` silently changes every matrix cell ([p0_submit_matrix.sh:71](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:71), [p0_profile.sbatch:87](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:87)). Collecting both suggested commands also overwrites the same default `p0_report.md`.

   **Fix:** use one manifest with rows keyed by `(cell, workers)`, explicitly export six workers for ordinary matrix/spot cells, include workers in `P0RESULT`, require both worker variants before computing the contrast, and produce a run-specific report path.

5. **BLOCKING — The unchanged 40-minute limit can terminate the required C32 spot before it emits a result.**

   Every mode receives `00:40:00` ([p0_submit_matrix.sh:83](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_submit_matrix.sh:83)), while the registered C32 prior is 0.012–0.018 steps/s—approximately 28–42 minutes for 30 steps before imports, rendezvous and first-batch startup ([plan_fa_orbit.md:53](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/plan_fa_orbit.md:53)). The slow end therefore cannot fit.

   **Fix:** give C16/C32 measured, spot-specific limits with startup margin, and record the limit in the manifest.

6. **BLOCKING — The three-point “per-orbit-pass” fit remains structurally confounded.**

   VAN uses the ordinary conditioner and Cartesian pose path, whereas C4L/C8 first replace poses with cylindrical features before running the conditioner ([diffusion.py:215](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:215), [yaw_rotation.py:282](/n/fs/gatrdp/codespace/FLAC/src/data/yaw_rotation.py:282)). Thus the VAN→C4L difference is not solely three extra ViT passes; fitting it as such biases both slope and intercept.

   **Fix:** add an FA1 control using `cond_method=fa_invariant`, `frame_avg_angles=[0.0]`, and checkpointing off, then fit FA1/C4L/C8. Keep canonical VAN as a separately reported contrast.

7. **NIT — The config-map hook can short-circuit a scheduled job through inherited environment state.**

   The hook runs before every Slurm and commit/config gate ([p0_profile.sbatch:65](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/p0_profile.sbatch:65)). This cannot produce an admitted result—the manifest-bound collector sees no `P0RESULT`—but Slurm reports a misleading successful no-op.

   **Fix:** permit the hook only when `SLURM_JOB_ID` is unset and explicitly remove `P0_PRINT_CONFIG` from exported job state.

8. **NIT — The corrected launch has only duck-typed callback tests, not the requested real DDP validation.**

   The new test explicitly avoids a Trainer, GPU and data path ([test_exp11_p0_runner.py:10](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_p0_runner.py:10)); therefore environment election, SyncBN conversion, device placement and final markers remain unexercised.

   **Fix:** after the blockers are fixed, require a reviewed two-GPU smoke proving world size two, distinct local devices, SyncBatchNorm, both timing marks, complete poller evidence and an admitted collector row.

## Coder-reported deviations

| Deviation | Judgment |
|---|---|
| Config-map early-exit hook | **Acceptable only as a query mechanism**; it cannot create an admitted unvalidated cell, but NIT 7 should prevent scheduled no-op success. |
| Exit code 6 | **Acceptable**; the same branch sets `valid=0`, so the collector classifies it `INVALID`. |
| `MAXSTEPS >= 30` | **Not acceptable**; the approved P0 contract is exactly 30 steps and peak VRAM otherwise covers a different workload. Covered by BLOCKING 3. |
| Dual worker manifests | **Not acceptable**; the pair and worker value are not jointly provenance-bound. Covered by BLOCKING 1 and 4. |
| Unchanged 40-minute limit | **Not acceptable** for the required C32 spot at the registered slow-rate prior. Covered by BLOCKING 5. |
