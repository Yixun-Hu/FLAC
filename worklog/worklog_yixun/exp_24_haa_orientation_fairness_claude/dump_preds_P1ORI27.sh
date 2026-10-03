#!/bin/bash
# One eval cell with --store_predictions for the FLAC+cue arm (P1ORI27, ckpt-1000, HAA test K=8, seed 42), so the
# per-sample predicted waveforms exist for the bias-vs-ranking analysis (Yixun 2026-10-03: why is CylDINO+cue better on
# retrieval but worse on T60/C50/EDT than FLAC+cue?). Same protocol as haa_ft_ori_eval.sh.
set -uo pipefail
cd /home/yixunhu/codespace/FLAC; E=worklog/worklog_yixun/exp_24_haa_orientation_fairness_claude; E23=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
echo "[dump] start $(date -Is)"
HF_HUB_OFFLINE=1 PYTHONPATH="" CUDA_VISIBLE_DEVICES=0 $PY eval_FLAC.py \
  --model-config $E/FLAC_HAA_finetune_P1ORI27.json --dataset-config $E23/haa_test_ori.json \
  --ckpt-path "outputs_FLAC/exp24_HAA_P1ORI27/FLAC_exp24_HAA_P1ORI27/exp24_HAA_P1ORI27/checkpoints/epoch=999-step=1000.ckpt" --cond-method vanilla --cond-autocast bf16 --record-per-scene \
  --cfg-scale 1.0 --steps 1 --seed 42 --eval-name qual_real_flacori_K8_s42 --store_predictions > $E/dump_preds_P1ORI27.log 2>&1
echo "[dump] rc=$? $(date -Is)"
