#!/usr/bin/env bash
set -euo pipefail

experiment_dir="/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
result_dir="${experiment_dir}/pilot_results_batch2"
log_path="${result_dir}/pilot_run_batch2.log"

mkdir -p "${result_dir}"
exec >>"${log_path}" 2>&1
exec /bin/bash "${experiment_dir}/run_pilot_batch2_commands.sh" "${1:-0}"
