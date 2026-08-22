# bf_fa_cartesian_command.md — exact reproduction commands (exp_21)

Env for all: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac`, repo root, branch `exp17-yawaug-scratch`.

**Two sections, and the distinction is the point.** *RUN* records what was actually executed, with the timestamp of the log it produced. *PLANNED* records the exact command a not-yet-executed step WILL use — it is a pre-registration, not a launch record, and nothing in it may be read as evidence that anything ran. (The previous "NOT YET LAUNCHED (added at launch time)" rows were self-contradictory: the commands were already pre-entered.)

---

# RUN

## SMOKE rehearsal — 2026-08-22 03:50:03 EDT
```bash
SMOKE=1 bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
```
25 steps, isolated identity `exp21_BFC_smoke`, LOGGER none forced, no checkpoints, co-tenant. Log: `bf_fa_cartesian_2026-08-22_03-50-03_smoke.log`. (An earlier attempt was refused by the conda gate, as designed.)

## Rung-4 cap-parity probe, FP32 only — 2026-08-22 03:53:41 EDT
```bash
python worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py
```
Log: `bf_fa_cartesian_2026-08-22_03-53-41_rung4_cap_parity.log`. **Superseded** by the run below: it measured only FP32, while the registered protocol conditions under bf16.

## Rung-4 cap-parity probe, REGISTERED bf16 + FP32 — 2026-08-22 04:10:10 EDT
```bash
python worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py
```
Log: `bf_fa_cartesian_2026-08-22_04-10-10_rung4_cap_parity_bf16.log`. Result: cap-32-vs-cap-64 **bit-exact (0.000e+00) at both precisions**; C4 90° 1.192e-07 fp32 (OK) but 7.812e-03 under bf16 (= 2⁻⁷, one bf16 mantissa step) → probe exits **1** and escalates rather than relaxing its own tolerance. Awaiting the Planner's judgement against the plan §5 metric limits.

## Launcher guard exercise — 2026-08-22 04:16:50 EDT
```bash
bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch_guardtests.sh
```
80 passed / 0 failed. Log: `bf_fa_cartesian_2026-08-22_04-16-50_guardtests.log`.

---

# PLANNED — pre-registered, NOT a launch record

Nothing below has been executed. Each entry is the exact command, recorded in advance so the run cannot drift from what was approved; a launch adds a RUN entry above with its real timestamp and log name.

## Rung-6 rate probe (>=200 steady co-tenant steps) — Planner schedules
```bash
RATE_PROBE=1 bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
```
320 steps FIXED (passing `MAXSTEPS` aborts), isolated identity `exp21_BFC_rateprobe`, LOGGER none forced, no checkpoints, own `*_rateprobe.log`. **Measure the steps-100..300 window** — 100 discarded to warmup, 200 steady, 20 tail slack. `SMOKE=1` together with `RATE_PROBE=1` aborts.

## Registered 40k training — Planner schedules
```bash
bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
```
REGISTERED mode. Every recipe value is pinned inside the script and env overrides are refused, not honoured: `--max-steps 40000 --checkpoint-every 2500 --logger wandb --batch-size 32 --num-gpus 2 --accum-batches 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed --num-workers 6 --seed 42`, no val loader, no resume, save-dir `outputs_FLAC/exp21_BFC`.

## Eval block (34 cells) — Planner schedules, after 40k
```bash
DRY_RUN=1 bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_eval_driver.sh   # list all 34 first
bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_eval_driver.sh
```
10 BFC registered + 4 invariance-grid angles + 10 BFre + 10 P1re. Every command is generated from `exp21_protocol.py`; the driver restates no flag.

### The conditioning protocol, inline (announcement 05)
These flags are part of the experiment, never defaults — `--cond-autocast bf16` versus the CLI default alone moved the same checkpoint's T60 between 8.202 and 10.652 in the exp_10 record, so they are recorded here as well as in the module:

```
--cond-method fa_cartesian            # BFC   (BFre: fa_invariant | P1re: vanilla, which carries NO orbit flags)
--frame-avg-angles 0,90,180,270       # frame-averaged arms only
--frame-avg-max-fwd-samples 64        # the EVALUATION cap; the arm TRAINS at 32
--rotate-mode fixed --rotate-deg 0    # grid cells: --rotate-deg {45,90,180,270}
--cond-autocast bf16
--batch-size 64 --cfg-scale 1.0 --steps 1
--record-per-scene
--record-stream --expected-stream-count 6337
--seed {42..46}
--eval-name exp21_{BFC,BFre,P1re}_S40000_K{1,8}_s{seed}[_rot{deg}]
```

### The three chunk plans, inline (announcement 06)
`angles_per_chunk = max(1, cap // batch)` over the 3 nonzero C4 angles:

| rung | batch | cap | angles/chunk | chunks |
|---|---|---|---|---|
| TRAIN | 32/rank | 32 | 1 | `[1,1,1]` — one DINOv3 RoPE draw per angle (B-F's schedule, D5) |
| EVAL, full batch | 64 | 64 | 1 | `[1,1,1]` |
| EVAL, tail batch | 1 (6337 % 64) | 64 | 64 | `[3]` — all three nonzero angles share one chunk |

⚠ The full-eval rung is **1** angle/chunk, not 2 (corrected in the r5 prelaunch fix; 2 arises only at batch 32 with cap 64, which is the rung-4 probe's configuration). Eval mode draws no RoPE, so the partition is numerically inert — measured bit-exact at both precisions in rung 4 — but the declared plan must still be the true one.

Train tail: 291,210 records at effective batch 64 leaves 10 globally / 5 per rank; `drop_last=True` discards them.

## REGISTERED 40k training — LAUNCHED 2026-08-22 05:50 EDT from HEAD `a98543f11f23eee5e2bbe3cf4863ec7e32f05f9b`
```bash
bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
```
REGISTERED mode: all recipe values pinned in-script (config FLAC_AR_BFC.json, 40k steps, ckpt 2500, wandb, DDP 32×2 SyncBN bf16-mixed seed 42, no val loader, no resume). Rate-gate baseline 13.94 s/step co-tenant (steps 100–300, ±25% same-tenancy).
