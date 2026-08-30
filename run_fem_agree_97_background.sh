#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp14=${task_repo}/worklog/worklog_yixun/exp_14_depth_aabb_matched_protocol
task_exp16=${task_repo}/worklog/worklog_yixun/exp_16_depth_aabb_matched_97
task_output=${task_repo}/worklog/worklog_yixun/exp_17_fem_agree_97
task_dataset=/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms
task_agree=/home/zhixuanzhao/projects/Frame_Average/FLAC-C4-FA-reproduction/weights/AGREE/AGREE_fullAR.pt
task_mkl=/opt/anaconda3/lib/libmkl_rt.so
task_tool=${task_repo}/tools/run_depth_aabb_fem_agree.py

mkdir -p "${task_output}"
printf '%s\n' "$$" > "${task_output}/launcher.pid"

common_solve=(
    "${task_python}" "${task_tool}"
    --stage solve
    --selection "${task_exp14}/depth_aabb_matched_14room_97.json"
    --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json"
    --geometry-audit "${task_exp9}/geometry_audit.json"
    --dataset-root "${task_dataset}"
    --source-result-dir "${task_exp16}/results"
    --output-dir "${task_output}"
    --solver-backend mkl_pardiso
    --solver-threads 12
    --mkl-runtime "${task_mkl}"
    --shard-count 2
)

echo "START FEM response shards $(date --iso-8601=seconds)"
MKL_RT="${task_mkl}" MPLCONFIGDIR=/tmp/matplotlib-exp17-fem-agree \
    "${common_solve[@]}" --shard-index 0 \
    > "${task_output}/solve_shard0.log" 2>&1 &
task_shard0_pid=$!
MKL_RT="${task_mkl}" MPLCONFIGDIR=/tmp/matplotlib-exp17-fem-agree \
    "${common_solve[@]}" --shard-index 1 \
    > "${task_output}/solve_shard1.log" 2>&1 &
task_shard1_pid=$!
printf '%s\n' "${task_shard0_pid}" > "${task_output}/solve_shard0.pid"
printf '%s\n' "${task_shard1_pid}" > "${task_output}/solve_shard1.pid"
wait "${task_shard0_pid}"
wait "${task_shard1_pid}"
echo "DONE FEM response shards $(date --iso-8601=seconds)"

while true; do
    task_gpu_line=$(nvidia-smi \
        --query-gpu=index,utilization.gpu \
        --format=csv,noheader,nounits | sort -t, -k2,2n | head -n 1)
    task_gpu_index=${task_gpu_line%%,*}
    task_gpu_util=${task_gpu_line##*,}
    task_gpu_index=$(echo "${task_gpu_index}" | tr -d ' ')
    task_gpu_util=$(echo "${task_gpu_util}" | tr -d ' ')
    if [[ "${task_gpu_util}" -le 50 ]]; then
        break
    fi
    echo "WAIT GPU availability: lowest utilization ${task_gpu_util}% $(date --iso-8601=seconds)"
    sleep 60
done

echo "START AGREE scoring on physical GPU ${task_gpu_index} $(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="${task_gpu_index}" MPLCONFIGDIR=/tmp/matplotlib-exp17-fem-agree \
    "${task_python}" "${task_tool}" \
    --stage score \
    --selection "${task_exp14}/depth_aabb_matched_14room_97.json" \
    --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
    --geometry-audit "${task_exp9}/geometry_audit.json" \
    --dataset-root "${task_dataset}" \
    --source-result-dir "${task_exp16}/results" \
    --agree-ckpt "${task_agree}" \
    --output-dir "${task_output}" \
    --device cuda:0 \
    --candidate-batch-size 32 \
    --score-seed 42 \
    --target-peak 0.95 \
    --tau 0.1
echo "DONE FEM--AGREE 97-query run $(date --iso-8601=seconds)"
