#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export MPLCONFIGDIR=/tmp/matplotlib-yawaug-localization-batch2

repo_root="/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp"
experiment_dir="${repo_root}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
result_root="${experiment_dir}/pilot_results_batch2"
output_dir="${result_root}/yawaug_ar_40k_vanilla"
log_path="${result_root}/yawaug_ar_40k_vanilla.log"
python_bin="/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python"

mkdir -p "${result_root}"
exec >>"${log_path}" 2>&1
cd "${repo_root}"

"${python_bin}" localize_FLAC.py \
  --model-config "${repo_root}/src/configs/model_configs/FLAC/AR/FLAC_AR.json" \
  --ckpt-path "/home/zhixuanzhao/projects/Frame_Average/Checkpoint/YAWAUG_AR_40k_epoch8-step40000.ckpt" \
  --agree-ckpt "/home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt" \
  --context-manifest "${experiment_dir}/context_manifest_exp01_seed42.json" \
  --geometry-audit "${experiment_dir}/geometry_audit.json" \
  --pilot-manifest "${experiment_dir}/pilot_manifest_seed43_batch2_4_per_room.json" \
  --dataset-root "/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms" \
  --output-dir "${output_dir}" \
  --device cuda:0 \
  --cond-method vanilla \
  --candidate-batch-size 64 \
  --sample-seed 42 \
  --tau 0.1
