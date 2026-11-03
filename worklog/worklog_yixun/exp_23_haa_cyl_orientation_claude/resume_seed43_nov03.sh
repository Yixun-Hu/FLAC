#!/usr/bin/env bash
# exp_23 seed-43 completion (Yixun go 2026-11-03 via overview session, relayed verbatim in the
# exp_21 session's record): resume CYLORI27_s43 500->1000 from the INTERRUPTION_2026-09-16 doc's
# command, then endpoints, then P1_s43 FULL + endpoints. ONE card (GPU env, default 0), co-tenant
# beside CERPA (<= ~16 GB ours; floor checks below). Executed by the exp_21 session on the absent
# web-session owner's behalf. Seeds 44-46 NOT run (explicitly not chosen).
set -u
cd /home/yixunhu/codespace/FLAC
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
GPU="${GPU:-0}"
eval "$(grep -E '^\s*export\s+WANDB_API_KEY=' ~/.bashrc | tail -1)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU); [ "$free" -ge 18000 ] || { echo "GPU $GPU only ${free} MiB free (<18000) - abort"; exit 2; }
CK=outputs_FLAC/exp23_HAA_CYLORI27_s43/FLAC_exp23_HAA_CYLORI27_s43/exp23_HAA_CYLORI27_s43/checkpoints/epoch=499-step=500.ckpt
[ -r "$CK" ] || { echo "resume ckpt unreadable - abort"; exit 2; }
echo "=== seed43 completion start $(date -Is) gpu $GPU pid $$ ==="
HF_HUB_OFFLINE=1 PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src CUDA_VISIBLE_DEVICES=$GPU \
python train.py --dataset-config $E/haa_train_ori.json --val-dataset-config $E/haa_val_ori.json \
  --model-config $E/FLAC_HAA_finetune_CYLORI27.json --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --pretrained-ckpt-path outputs_FLAC/exp19_inits/HAA_init_CYLORI.ckpt --ckpt-path "$CK" \
  --max-steps 1000 --batch-size 16 --accum-batches 4 --num-workers 8 --seed 43 --num-gpus 1 \
  --precision bf16-mixed --val-every 10 --checkpoint-every 100 --logger wandb \
  --name FLAC_exp23_HAA_CYLORI27_s43 --experiment-name exp23_HAA_CYLORI27_s43 \
  --save-dir outputs_FLAC/exp23_HAA_CYLORI27_s43
rc=$?; echo "RESUME CYLORI27_s43 rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "SEED43 STOPPED after resume"; exit $rc; }
ARM=CYLORI27 TAG=_s43 ENDPOINTS="1000" GPU=$GPU bash $E/haa_ft_cylori_eval_v2.sh endpoints
rc=$?; echo "ENDPOINTS CYLORI27_s43 rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "SEED43 STOPPED after CYLORI27 endpoints"; exit $rc; }
ARM=P1 SEED=43 CADENCE=100 MODE=FULL GPU=$GPU VRAM_FLOOR=8000 DISK_FLOOR=60000 bash $E/haa_ft_cylori_launch.sh
rc=$?; echo "FULL P1_s43 rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "SEED43 STOPPED after P1 FULL"; exit $rc; }
ARM=P1 TAG=_s43 ENDPOINTS="1000" GPU=$GPU bash $E/haa_ft_cylori_eval_v2.sh endpoints
rc=$?; echo "ENDPOINTS P1_s43 rc=$rc $(date -Is)"
echo "SEED43 COMPLETION DONE rc=$rc $(date -Is)"
