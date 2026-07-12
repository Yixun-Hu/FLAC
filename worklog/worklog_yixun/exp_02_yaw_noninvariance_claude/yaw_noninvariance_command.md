# Command — exp_02_yaw_noninvariance

Launched: 2026-07-04_20:49:36

```bash
cd /home/yixunhu/codespace/FLAC
bash worklog/exp_02_yaw_noninvariance_claude/run_exp02.sh
```

= 5 sequential runs of (per Yixun's spec):

```bash
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed 42 \
  --store_predictions --eval-name yaw_<name> [--rotate-deg <0|90|180|270>]
```

Metric-1 offline comparison (after runs):

```bash
python worklog/exp_02_yaw_noninvariance_claude/compare_predictions.py \
  --ref weights/FLAC/FLAC_EMA_predictions_1_1.0_yaw_baseline.pt \
  --alt weights/FLAC/FLAC_EMA_predictions_1_1.0_yaw_rot<deg>.pt \
  --out worklog/exp_02_yaw_noninvariance_claude/metric1_rot<deg>.json
```
