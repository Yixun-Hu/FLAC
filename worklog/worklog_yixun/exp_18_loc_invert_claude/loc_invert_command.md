# loc_invert_command — exact reproduction commands (appended AT LAUNCH per SOP)

## R-1a readback gate (2026-08-19_21:19:20 EDT, HEAD 30f26d1)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
python eval_localization.py --mode readback \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1a_readback \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:19:20_R1a_readback.log
```

## R-1b oracle + baselines (2026-08-19_21:19:20 EDT, HEAD 30f26d1)
```bash
cd /home/yixunhu/codespace/FLAC && source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
python eval_localization.py --mode run --score-source gt_rir \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --agree-ckpt weights/AGREE/AGREE_AR.pt \
  --tau 0.1 --agg lme --seed 42 \
  --cond-method vanilla --rotate-deg 0 --cond-autocast default \
  --batch-size 4 --num-workers 4 --device cuda:1 \
  --out-dir outputs_loc/exp18 --eval-name exp18_R1b_oracle \
  2>&1 | tee worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_2026-08-19_21:19:20_R1b_oracle.log
```
