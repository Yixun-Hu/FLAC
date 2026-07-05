# Command — exp_01_reproduce_flac_table1

Launched: 2026-07-04_18:21:52 (log: `reproduce_flac_table1_2026-07-04_18:21:52.log`)

## Reproduce

```bash
cd /home/yixunhu/codespace/FLAC
bash worklog/exp_01_reproduce_flac_table1_claude/run_exp01.sh
```

which runs, for K in {1, 8} and SEED in {42..46}:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval[_1].json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --cfg-scale 1.0 --steps 1 --seed $SEED \
  --eval-name exp01_unseen_K${K}_seed${SEED}
```

Outputs: `weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K{K}_seed{SEED}.json` (10 files).
