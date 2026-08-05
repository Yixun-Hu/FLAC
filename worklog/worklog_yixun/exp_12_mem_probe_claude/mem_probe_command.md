# exp_12 mem_probe — reproduction commands (recorded at launch)

## Probe submission (2026-08-05T17:15 EDT) — job 3637984

```bash
cd /n/fs/gatrdp/codespace/FLAC
sbatch --export=ALL,EXPECT_SHA=$(git rev-parse HEAD) \
  worklog/worklog_yixun/exp_12_mem_probe_claude/mem_probe.sbatch
# Submitted batch job 3637984
# worker-bound SHA: 4c095ae517aa5f9e8e5dd847d40199ce4d98c9f0
# job state at +1 min: RUNNING (started 2026-08-05T16:29:26)
```

The sbatch file is self-contained (gates + measurement + classification); the inner training command it runs is:

```bash
python train.py \
  --model-config worklog/worklog_yixun/exp_12_mem_probe_claude/FLAC_AR_exp12_memprobe.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 5 --batch-size 64 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 1 --num-nodes 1 --strategy auto --precision bf16-mixed \
  --sync-batchnorm false --val-every -1 --val-dataset-config '' \
  --ckpt-path '' --pretrained-ckpt-path '' --gradient-clip-val 0.0 \
  --logger wandb --checkpoint-every 10000 \
  --name FLAC_exp12_memprobe --experiment-name exp12_memprobe \
  --save-dir outputs_FLAC/exp12_memprobe
```

Artifacts land in this folder: `slurm_exp12-mem-probe_3637984.out`, `mem_probe_<TS>_jid3637984_S5_train.log`, `mem_probe_<TS>_jid3637984_vram.csv`.
