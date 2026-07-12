# Commands — exp_05_bn_drift_bisect

Code `8e673d3`+, `CUDA_VISIBLE_DEVICES=0`. Every launched run's command lands here at launch time.

## B-1 pilot (one real batch; LAUNCHED 2026-07-06)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/bn_drift_probe.py \
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

## B1 grid stage 1 — max_len via config copies (LAUNCHED 2026-07-06 ~02:55)

```bash
for ML in 4800 10240 19200; do
CUDA_VISIBLE_DEVICES=0 python tools/bn_drift_probe.py \
  --dataset-config worklog/exp_05_bn_drift_bisect_claude/train_maxlen$ML.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --n-batches 200 --batch-size 16 --seed 42 --device cuda \
  --out worklog/exp_05_bn_drift_bisect_claude/bn_drift_B1_maxlen$ML.json
done
```

## Dispersion check (pre-V1', review-mandated; LAUNCHED 2026-07-06 ~03:50)

```bash
CUDA_VISIBLE_DEVICES=0 python worklog/exp_05_bn_drift_bisect_claude/dispersion_check.py  # one-off; uses tools.bn_drift_probe.BNInputRecorder; output committed as dispersion_check_result.json
```

## V1' — BN-frozen vanilla control (LAUNCHED 2026-07-06 ~02:36)

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_cond.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --save-dir outputs_FLAC/exp05_V1p_freezebn_ft --name FLAC_exp05_V1p_freezebn \
  --cond-method vanilla --lr 5e-6 --freeze-bn --max-steps 625 --checkpoint-every 200 \
  --batch-size 4 --accumulate-grad-batches 32 --num-workers 4 --seed 42
# then gate evals: eval_FLAC.py with the V1' ckpt, K in {1,8} x seeds 42..46, exp_01 protocol,
#   --eval-name exp05_V1p_K${K}_seed${SEED}
```
