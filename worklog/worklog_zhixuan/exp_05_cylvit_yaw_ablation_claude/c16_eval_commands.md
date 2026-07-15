# C16 Yaw-Sweep Evaluation Commands

Recorded at launch time per experiment_SOP.md.

## 2026-07-12T10:20-04:00 — C16 sweep at matched total-25k milestone

Both 30k training runs were still in progress (~step 20,150 / 25,000), so the
sweep uses the latest milestone checkpoint available for BOTH models:
`epoch=4-step=20000.ckpt` (total-step label 25k). GPUs had ample headroom
(~4.3 GiB / 49 GiB used per card), so evaluation runs concurrently with
training, one model per GPU.

Angles: all 16 patch-grid yaw angles (n * 22.5°, n = 0..15) — the C16 group on
which CylViT is equivariant by construction.

```bash
MODEL=cylvit GPU=0 bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_c16_eval.sh \
  > worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/c16_eval_cylvit_total25k_gpu0.log 2>&1 &

MODEL=simplevit GPU=1 bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_c16_eval.sh \
  > worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/c16_eval_simplevit_total25k_gpu1.log 2>&1 &
```

- Checkpoints: `outputs_FLAC/exp05_{model}_phase3_total30k_s42/epoch=4-step=20000.ckpt`
- Eval config: `acousticroom_unseeneval_1.json`, K=1, steps=1, cfg_scale=1.0, seed 42, batch 64
- Metrics JSONs land next to the checkpoint as
  `epoch=4-step=20000_metrics_1_1.0_exp05_{model}_c16_total25k_yaw{angle}[_rot{int}].json`
- The script skips angles whose metrics JSON already exists, so it is safe to
  re-run after an interruption, and can be re-pointed at the final 30k
  checkpoint later via `CKPT=... TOTAL_LABEL=30k`.

## PLANNED — C16 sweep at final total-30k (queued 2026-07-12, runs after training ends)

Per Yixun's request the sweep is repeated on the final 30k model. Sequencing:
wait until BOTH 30k trainings AND their auto-launched milestone evals
(`run_phase3_30k_one.sh` trailing `run_phase3_milestone_eval.sh`) have exited,
then:

`STORE_PRED_C4=1` additionally stores prediction tensors at the C4 angles
(0/90/180/270), consumed by the exp_02-style waveform figure below.

```bash
MODEL=cylvit GPU=0 TOTAL_LABEL=30k STORE_PRED_C4=1 \
  CKPT="$(ls outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=*-step=25000.ckpt)" \
  bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_c16_eval.sh \
  > worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/c16_eval_cylvit_total30k_gpu0.log 2>&1 &

MODEL=simplevit GPU=1 TOTAL_LABEL=30k STORE_PRED_C4=1 \
  CKPT="$(ls outputs_FLAC/exp05_simplevit_phase3_total30k_s42/epoch=*-step=25000.ckpt)" \
  bash worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/run_c16_eval.sh \
  > worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/c16_eval_simplevit_total30k_gpu1.log 2>&1 &
```

Then summarize with:

```bash
../venv/bin/python worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/summarize_c16_sweep.py \
  --ckpt-stem "epoch=<N>-step=25000" --total-label 30k
```

(The Lightning `step=25000` checkpoint = total-30k label, consistent with the
milestone convention; the 25k C16 run used `step=20000` the same way.)

## PLANNED — exp_02-style waveform rotation figure (Yixun request, 2026-07-12)

After the 30k C16 sweep (with `STORE_PRED_C4=1`) finishes, generate the
4-panel waveform-level rotation figure (log-RMS envelope, raw-waveform zoom,
P180−P0 difference trace, Schroeder decay) for BOTH models, showcasing the
SAME sample (selected at the 90th percentile of SimpleViT's P180-vs-P0
envelope gaps, so the comparison is like-for-like):

```bash
STEM="$(basename "$(ls outputs_FLAC/exp05_cylvit_phase3_total30k_s42/epoch=*-step=25000.ckpt)" .ckpt)"
../venv/bin/python worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/gen_rotation_waveform_visuals.py \
  --model cylvit --ckpt-stem "$STEM" --total-label 30k --select-by simplevit
../venv/bin/python worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/gen_rotation_waveform_visuals.py \
  --model simplevit --ckpt-stem "$STEM" --total-label 30k --select-by simplevit
```

Outputs: `figures/rir_rotation_{model}_30k.{png,pdf}`. Expectation: SimpleViT
reproduces the exp_02 signature (nonzero difference trace, diverging Schroeder
curves); CylViT's difference trace should be near-flat and its C4 Schroeder
curves should coincide (exact patch-grid invariance of the geometry encoder).
Note the two exported checkpoints may have different epoch numbers; if so, run
each model with its own stem instead of the shared `$STEM`.
