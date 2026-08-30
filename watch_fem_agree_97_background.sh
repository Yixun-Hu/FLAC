#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_output=${task_repo}/worklog/worklog_yixun/exp_17_fem_agree_97
task_pattern='run_depth_aabb_fem_agree.py --stage solve'

mkdir -p "${task_output}"
printf '%s\n' "$$" > "${task_output}/watcher.pid"
while pgrep -f "${task_pattern}" > /dev/null; do
    task_completed=$(find "${task_output}/responses" -maxdepth 1 -name 'query_*.json' 2>/dev/null | wc -l)
    echo "WAIT active response shards: ${task_completed}/97 $(date --iso-8601=seconds)"
    sleep 60
done

echo "HANDOFF to resume-safe launcher $(date --iso-8601=seconds)"
exec /bin/bash "${task_repo}/run_fem_agree_97_background.sh"
