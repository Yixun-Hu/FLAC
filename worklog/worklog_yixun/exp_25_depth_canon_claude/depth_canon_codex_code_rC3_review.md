**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-26 · **Round:** code rC3 (review of fix round rC-F3: pinned offline W&B mode after SMOKE 1 failed)

**Verdict:** SHIP — conditional only on the Planner’s currently running full guard finishing green. Once green, the training launcher/watcher may be re-armed.

No BLOCKING or HIGH findings. One pre-existing, non-blocking diagnostic NIT.

### Per-item status

- **D1 — RESOLVED.** `WANDB_MODE_PIN="offline"` is a single pinned literal at [depth_canon_launch.sh:90](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:90), reported before ARGV at [line 181](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:181), and exported immediately before the shared SMOKE/FULL train invocation at [line 570](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:570). Caller values are overridden.

  The export remains set for post-run commands, but that is harmless: NAS/identity/DINO gates have already run; the checkpoint scan and evidence writer at [lines 613 and 668](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:613) use only stdlib/filesystem operations and never import W&B.

  Static SDK tracing confirms offline mode:

  - W&B maps `offline` to `_offline=True` at [wandb_settings.py:647](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py:647).
  - Init skips login and user-server settings at [wandb_init.py:288](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/wandb_init.py:288).
  - The handler acknowledges run creation locally at [handler.py:193](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/handler.py:193).
  - The writer persists records but does not forward ordinary records to the sender offline at [writer.py:129](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/writer.py:129).
  - Consequently `sender._init_run()` and its `upsert_run()` at [sender.py:980](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:980) do not run. Finish’s server-info request also short-circuits offline at [sender.py:1574](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/wandb/sdk/internal/sender.py:1574).

  Lightning’s `watch`, config/hyperparameter logging, `define_metric`, metrics, `wandb.Audio/Image`, and finish therefore use the local record path. The current arm has no `wandb.save`, artifact, or `use_artifact` call: [train.py:164](train.py:164) uses the ordinary logger/watch/config flow, while the CANON diffusion wrapper only imports scalar `log_metric`. `WandbLogger(log_model=False)` is the default, so checkpoint artifacts are not enabled.

- **D2 — RESOLVED.** The fake trainer records only `WANDB_MODE` at [depth_canon_guardtests.sh:195](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:195). Section W at [line 796](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:796) covers SMOKE, FULL, caller `online`/`disabled`, DRY_RUN ordering, literal uniqueness, export uniqueness, and missing-W&B refusal. Removing the export is caught by W2b/W3b/W4b/W5b/W6b and W8.

  Genuine RED: 7 passed / 12 failed / 1 skipped at `62dc642`; GREEN: 19 / 0 / 1 at `6c7c1fe`. Coder full guard: 225 / 0 / 0.

- **D3 — RESOLVED.** Watcher and watcher pytest are byte-unchanged from `692f4f5`; watcher SHA remains `d1aa5e41…`. Its production call still pins `MB=32 ACC=1 DRY_RUN=0` at [depth_canon_autolaunch.sh:219](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:219). Its scrub need not include `WANDB_MODE`, because the launcher overwrites it.

  I agree with the Planner’s arming scrub: [arm_exp25_v14.sh:34](</tmp/claude-1007/-home-yixunhu-codespace-FLAC/1630c6fe-5d3e-42b4-94ec-1b6d3bcb0e4e/scratchpad/r3a/arm_exp25_v14.sh:34>) removes both `WANDB_MODE` and `WANDB_DIR`. This is defense-in-depth. In fact, PL currently passes explicit `dir="."` at [wandb.py:331](/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytorch_lightning/loggers/wandb.py:331), whose INIT precedence exceeds environment precedence, so the Coder’s `WANDB_DIR` concern is overstated for this exact logger path. The scrub is still prudent.

- **D4 — RESOLVED.** The launcher header states offline storage/sync consequences at [depth_canon_launch.sh:41](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:41); the notebook records the cause, fix, lack of live dashboard, later sync, and no Coder online call at [depth_canon_worklog.md:1416](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_worklog.md:1416) and [line 1445](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_worklog.md:1445).

  The failed SMOKE artifacts remain present. This was **not a v14 gate defect**: pins/config/audit/NAS passed; the GPU gate recorded 16,264/16,098 MiB and both shared co-tenants; identity was `yh4742@princeton.edu`. Failure occurred later at `logger.watch → wandb.init`, as shown at [smoke log:78](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_2026-10-26_03-43-23_exp25_CANON_smoke_train.log:78). The FAIL evidence correctly preserves rc 1 and GPU provenance at [evidence:31](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_smoke_evidence_2026-10-26_03-43-23.json:31).

- **D5 — PARTIAL, NIT only.** All material post-run checks behaved correctly: missing banner/ranks/endpoint were reported without replacing trainer rc 1; non-finite scan was clean; zero checkpoints were confirmed; zero losses and absent latency were recorded; verdict/evidence were FAIL; FULL was not attempted. The evidence’s stored log SHA matches the actual log.

### Findings

- **NIT — misleading early-failure projection wording:** [depth_canon_launch.sh:694](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:694) divides elapsed time by the configured 25-step budget and [line 695](worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:695) prints “25 steps in 14 s” even though zero steps completed. The verdict remains correct because rc 1 wins at line 722. Suggested later fix: label this as the “25-step budget” and emit startup-inclusive projection `n/a` unless progress confirms completed steps. This does not justify delaying re-arm.

### Verification performed

- Branch/HEAD: `exp-25-depth-canon`, `5d3f7865670290823f43ebcfc65f314dc6f228f3`.
- No training-closure file changed. Generated manifest is byte-equal to committed manifest: 48 entries, SHA `631da22f…`.
- `bash -n`: launcher, guard, and pin generator passed.
- `git diff --check 692f4f5 5d3f786`: clean.
- Commit sizes: `62dc642` 63+/0−; `6c7c1fe` 14+/1−; `5d3f786` records 791+/0−.
- Launcher SHA `2c3453e8…`; guard SHA `2c65d254…`; watcher unchanged.
- Environment evidence: pip-freeze SHA remained `e896c48f…` across failed SMOKE, Coder DRY_RUN, and Planner DRY_RUN. No package/environment file changed.
- Planner regression: 2403 passed / 7 skipped / 1 xfailed / 0 failed.
- Planner DRY_RUN at final HEAD: rc 0; one mode line immediately before ARGV; launcher binding `2c3453e8…`.
- Planner guard latest read at **05:18:12 EDT**: still running, through section R case R6, all emitted cases PASS and no FAIL/abort line. Final SHIP activation remains conditional on that log ending green.
