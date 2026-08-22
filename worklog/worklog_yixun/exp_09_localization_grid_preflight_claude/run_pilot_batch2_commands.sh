#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:-0}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"

repo_root="/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp"
python_bin="/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python"
experiment_dir="${repo_root}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
result_dir="${experiment_dir}/pilot_results_batch2"
pilot_manifest="${experiment_dir}/pilot_manifest_seed43_batch2_4_per_room.json"
common_args=(
  --model-config "${repo_root}/src/configs/model_configs/FLAC/AR/FLAC_AR.json"
  --agree-ckpt "/home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt"
  --context-manifest "${experiment_dir}/context_manifest_exp01_seed42.json"
  --geometry-audit "${experiment_dir}/geometry_audit.json"
  --pilot-manifest "${pilot_manifest}"
  --dataset-root "/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms"
  --device cuda:0
  --candidate-batch-size 64
  --sample-seed 42
  --tau 0.1
)

cd "${repo_root}"

"${python_bin}" localize_FLAC.py \
  "${common_args[@]}" \
  --ckpt-path "/home/zhixuanzhao/projects/Frame_Average/Checkpoint/P1_40k_clean_hybrid_EMA.ckpt" \
  --output-dir "${result_dir}/vanilla" \
  --cond-method vanilla

"${python_bin}" localize_FLAC.py \
  "${common_args[@]}" \
  --ckpt-path "/home/zhixuanzhao/projects/Frame_Average/Checkpoint/BF_40k_clean_hybrid_EMA.ckpt" \
  --output-dir "${result_dir}/fa_bf" \
  --cond-method fa_invariant

"${python_bin}" tools/aggregate_localization_results.py \
  --pilot-manifest "${pilot_manifest}" \
  --vanilla-dir "${result_dir}/vanilla" \
  --fa-bf-dir "${result_dir}/fa_bf" \
  --output-json "${result_dir}/pilot_results_batch2.json" \
  --output-md "${result_dir}/pilot_results_batch2.md"
