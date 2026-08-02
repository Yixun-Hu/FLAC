# CylindricalViT: FLAC-compatible training and inference

This directory is the end-to-end entry point for replacing FLAC's two geometry
encoders with the repository-native `CylindricalViT` in
`src/models/cyl_vit.py`. It does **not** use the separate
`cylindrical_dinov3` package.

The experiment keeps FLAC's train/eval stack and recipe unchanged outside the
geometry-ViT blocks:

- standard `train.py` and `eval_FLAC.py`;
- full AcousticRooms K=8 training split;
- VAE initialization from `weights/FLAC/VAE.safetensors`;
- AdamW (`5e-5`, betas `0.9/0.999`, weight decay `1e-3`);
- InverseLR (`inv_gamma=1e6`, `power=0.5`, `warmup=0.99`);
- EMA, BF16 mixed precision, effective batch size 64;
- 67,500 optimizer steps and a checkpoint every 2,500 steps;
- full unseen K=1/K=8 evaluation with seeds 42--46.

The two custom ViTs are a shared, randomly initialized CylindricalViT with a
linear `16x32` patch embedding, 512-dimensional tokens, 12 blocks, 8 heads,
and mean token pooling. Loading DINOv3 weights into this architecture is not
possible; all non-ViT training settings remain matched.

## Files

- `FLAC_AR_CylViT.json`: experiment-frozen config; the original exp-05 config
  remains untouched.
- `verify_config.py`: fail-closed audit proving that only the two geometry-ViT
  config blocks differ from `FLAC_AR.json`; optionally instantiates the model.
- `run_train.sh`: standard 67.5k training, explicit resume, and dry-run support.
- `run_predict.sh`: one K/seed/yaw inference run, with optional waveform storage.
- `run_eval_suite.sh`: full unseen K=1/K=8, seeds 42--46 metric suite.
- `run_yaw_suite.sh`: C16 K=1 yaw sweep plus K=8 yaw 0/90 checks.
- `run_pipeline.sh`: train, resolve the final checkpoint, evaluate, summarize.
- `find_checkpoint.py`: unambiguous recursive checkpoint resolver.
- `summarize_metrics.py`: aggregate metric JSON files into Markdown.

All scripts locate the repository with Git, so they can be launched from any
working directory.

## Prerequisites

Follow the root README and provide:

1. AcousticRooms at the path expected by the tracked dataset configs;
2. `weights/FLAC/VAE.safetensors` for training;
3. `weights/AGREE/AGREE_fullAR.pt` and its DINOv3 dependency for metrics;
4. a Python environment containing the FLAC requirements.

Run the structural preflight without a GPU. Environment assignments must
precede the `bash` command:

```bash
PYTHON=/path/to/flac/python DRY_RUN=1 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_train.sh
```

## Train

One GPU, micro-batch 8 and accumulation 8 gives effective batch 64:

```bash
PYTHON=/path/to/flac/python GPU_IDS=0 BATCH_SIZE=8 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_train.sh
```

Two GPUs, micro-batch 8 per GPU and accumulation 4:

```bash
PYTHON=/path/to/flac/python GPU_IDS=0,1 BATCH_SIZE=8 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_train.sh
```

The launcher derives accumulation from `GLOBAL_BATCH_SIZE=64` and refuses a
non-integral or mismatched effective batch. Useful overrides include
`LOGGER=wandb`, `NUM_WORKERS`, `CHECKPOINT_EVERY`, `SAVE_DIR`, and `SEED`.

Resume explicitly from a Lightning checkpoint:

```bash
CKPT_PATH=/path/to/epoch=...-step=25000.ckpt GPU_IDS=0 BATCH_SIZE=8 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_train.sh
```

For a wiring smoke test, use a separate output directory:

```bash
MAX_STEPS=2 CHECKPOINT_EVERY=1 SAVE_DIR=outputs_FLAC/cyl_vit_test_smoke \
  GPU_IDS=0 BATCH_SIZE=1 GLOBAL_BATCH_SIZE=1 LOGGER=none \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_train.sh
```

## Predict one configuration

`run_predict.sh` runs `eval_FLAC.py`, computes metrics, and stores decoded
predictions by default. `SPLIT` accepts `unseen` (default) or `seen`:

```bash
CKPT_PATH=/path/to/epoch=...-step=67500.ckpt SPLIT=unseen K=1 SEED=42 YAW=0 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_predict.sh
```

Set `STORE_PREDICTIONS=0` for metrics only. Use `K=8` for the eight-context
protocol. Output JSON/PT files are written beside the checkpoint by the
existing `eval_FLAC.py` naming contract.

## Full evaluation

Full unseen K=1 and K=8 metrics for seeds 42--46:

```bash
CKPT_PATH=/path/to/epoch=...-step=67500.ckpt \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_eval_suite.sh
```

C16 yaw stress test at K=1 and the K=8 yaw 0/90 spot check:

```bash
CKPT_PATH=/path/to/epoch=...-step=67500.ckpt \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_yaw_suite.sh
```

Both suites skip metric files that already exist. `run_eval_suite.sh` writes a
summary beside the checkpoint; the summary can also be regenerated directly:

```bash
python worklog/worklog_zhixuan/cyl_vit_test/summarize_metrics.py \
  --checkpoint /path/to/epoch=...-step=67500.ckpt
```

## Complete chain

The complete train -> final-checkpoint resolution -> K=1/K=8 evaluation ->
summary chain is:

```bash
GPU_IDS=0 BATCH_SIZE=8 \
  bash worklog/worklog_zhixuan/cyl_vit_test/run_pipeline.sh
```

Set `RUN_YAW=1` to append the yaw suite. To evaluate an existing run without
training, set `RUN_TRAIN=0 CKPT_PATH=/path/to/checkpoint.ckpt`. Use
`DRY_RUN=1 SEEDS=42` to print the complete chain without allocating a GPU.

`run_eval_suite.sh` defaults to the paper-facing unseen split. Set
`SPLITS="unseen seen"` to evaluate both tracked splits without filename
collisions.
