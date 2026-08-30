#!/usr/bin/env bash
set -u -o pipefail

BENCH_REPO=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
BENCH_PYTHON=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
BENCH_OUTPUT=${BENCH_REPO}/worklog/worklog_yixun/exp_18_depth_aabb_oversized_benchmark
BENCH_CONTEXT=${BENCH_REPO}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json
BENCH_GEOMETRY=${BENCH_REPO}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json
BENCH_DATASET=/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms
BENCH_MKL=/opt/anaconda3/lib/libmkl_rt.so

mkdir -p "${BENCH_OUTPUT}/logs"
cd "${BENCH_REPO}" || exit 1

run_query() {
  local query_index=$1
  local room_label=$2
  local result_path
  local log_path
  local exit_code
  result_path=$(printf '%s/query_%05d_depth_aabb_result.json' "${BENCH_OUTPUT}" "${query_index}")
  log_path=$(printf '%s/logs/query_%05d.log' "${BENCH_OUTPUT}" "${query_index}")

  if [[ -f "${result_path}" ]]; then
    printf '[%s] query %d already completed\n' "${room_label}" "${query_index}"
    return 0
  fi

  printf '[%s] query %d started at %s\n' "${room_label}" "${query_index}" "$(date --iso-8601=seconds)"
  MKL_RT="${BENCH_MKL}" MPLCONFIGDIR=/tmp/matplotlib-exp18-depth-aabb-oversized \
    "${BENCH_PYTHON}" tools/probe_depth_aabb_fem.py \
      --query-index "${query_index}" \
      --context-manifest "${BENCH_CONTEXT}" \
      --geometry-audit "${BENCH_GEOMETRY}" \
      --dataset-root "${BENCH_DATASET}" \
      --output-dir "${BENCH_OUTPUT}" \
      --maximum-edge-m 0.22 \
      --padding-m 0.05 \
      --solver-backend mkl_pardiso \
      --solver-threads 24 \
      --mkl-runtime "${BENCH_MKL}" \
      >"${log_path}" 2>&1
  exit_code=$?
  printf '[%s] query %d exited %d at %s\n' "${room_label}" "${query_index}" "${exit_code}" "$(date --iso-8601=seconds)"
  return "${exit_code}"
}

run_query 335 Cafe_idx_1 || true
run_query 3550 Auditorium_idx_1 || true
