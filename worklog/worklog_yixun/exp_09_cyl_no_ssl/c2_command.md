# C2 full training — command (recorded BEFORE launch)

**Yixun C2 go:** covered by the 2026-07-21 verbatim "start exp_06/FLAC exp-09's GPU
training" (exp_06 worklog @ cyl `3e416db`); the co-tenant interaction was flagged to
Yixun at C1 with an explicit hold-window before this launch. **C1 evidence:** fit peaks
22,145/22,259 MiB (ours ≈6.3 GiB/GPU); frozen gate 26,355 MiB; smoke PASSED (100/100
steps observed==declared, finite loss, strict reload, **sustained 0.16863 steps/s**
ceiled-monotonic, 9 backbone calls/batch, zero extra frame passes). **Projection:
67,500 / 0.16863 ≈ 4.63 d co-tenant** (vs B-F's own 0.079 → exp-09's one-pass
conditioning is ~2.1× faster).

Pins: EXPECT_PACKAGE_SHA `3e416db…` (cyl repo unchanged since freeze, verified at
launch); EXPECT_EXP09_SHA = THIS records commit (self-ref convention, post-hoc
verified). Co-tenant with B-F ranks 855996/856706 (never touched; mutual-slowdown
flag stands — B-F's ETA extends while C2 runs).

**Screening prohibition (integrative r2 §4):** C1 did NOT prove a combined
train+eval peak ⇒ **NO co-tenant screening during C2**. Per-10k screens are SKIPPED;
all evaluation happens at D on the saved checkpoints (ckpt every 2,500 steps).

**Resume policy:** any stop/resume uses `--ckpt-path <last.ckpt>` (launcher
passthrough) and is DISCLOSED as a fresh stochastic continuation (Lightning does not
restore RNG/dataloader position) — pre-registered, plan §3.

```bash
cd /home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export EXPECT_PACKAGE_SHA=3e416db1b6933dd842a3667432ff21436e7089ca
export EXPECT_EXP09_SHA=$(git rev-parse HEAD)
export EXP09_LOG_DIR=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude
LOGGER=wandb nohup setsid bash worklog/worklog_yixun/exp_09_cyl_no_ssl/exp09_launch.sh \
  > "$EXP09_LOG_DIR/exp09_c2_wrapper_$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &
```
(wandb run `FLAC_exp09_cylNoSSL`; save-dir `outputs_FLAC/exp09_cylNoSSL` in the
worktree; the launcher re-runs the pin gate + frozen-threshold + free-VRAM gates
fail-closed before train.py; 67,500 steps, ckpt every 2,500, seed 42.)
Acceptance at launch: gates ALL PASS in the teed log; two ranks resident; wandb run
live. Completion acceptance: step 67,500 reached; per-2,500 ckpts present; D-stage
evaluation follows separately.
