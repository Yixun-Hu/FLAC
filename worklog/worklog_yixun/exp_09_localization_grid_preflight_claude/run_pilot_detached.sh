#!/usr/bin/env bash
set -euo pipefail

experiment_dir="/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
log_path="${experiment_dir}/pilot_results/pilot_run.log"

exec >>"${log_path}" 2>&1
exec /bin/bash "${experiment_dir}/run_pilot_commands.sh" "${1:-0}"
