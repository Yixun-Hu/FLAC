# Commands — exp_05_bn_drift_bisect

Code `8e673d3`+, `CUDA_VISIBLE_DEVICES=0`. Every launched run's command lands here at launch time.

## B-1 pilot (one real batch; LAUNCHED 2026-07-06)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/bn_drift_probe.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --n-batches 1 --batch-size 16 --seed 42 --device cuda \
  --out worklog/exp_05_bn_drift_bisect_claude/bn_drift_pilot_B-1.json
```

## B0 baseline (LAUNCHED 2026-07-06, after B-1 pass)

```bash
# train loader, 3 repeats (seeds 42/43/44 for data order):
for S in 42 43 44; do
CUDA_VISIBLE_DEVICES=0 python tools/bn_drift_probe.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --n-batches 200 --batch-size 16 --seed $S --device cuda \
  --out worklog/exp_05_bn_drift_bisect_claude/bn_drift_B0_train_seed$S.json
done
# eval loader reference (K=8 eval config):
CUDA_VISIBLE_DEVICES=0 python tools/bn_drift_probe.py ... \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --n-batches 200 --batch-size 16 --seed 42 --device cuda \
  --out worklog/exp_05_bn_drift_bisect_claude/bn_drift_B0_eval_seed42.json
```
