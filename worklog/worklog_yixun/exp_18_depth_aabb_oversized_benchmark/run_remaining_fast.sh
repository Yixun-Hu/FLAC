#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_REPO=${BENCH_REPO:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}
BENCH_PYTHON=${BENCH_PYTHON:-${BENCH_REPO}/../../FLAC-vanilla/.venv/bin/python}
BENCH_OUTPUT=${BENCH_OUTPUT:-${SCRIPT_DIR}}
BENCH_SELECTION=${BENCH_REPO}/worklog/worklog_yixun/exp_14_depth_aabb_matched_protocol/depth_aabb_matched_16room_112.json
BENCH_CONTEXT=${BENCH_REPO}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/context_manifest_exp01_seed42.json
BENCH_GEOMETRY=${BENCH_REPO}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json
BENCH_DATASET=${BENCH_DATASET:-${BENCH_REPO}/../../../rir2rir/FLAC/AcousticRooms}
BENCH_MKL=${BENCH_MKL:-/opt/anaconda3/lib/libmkl_rt.so}

if [[ ! -x "${BENCH_PYTHON}" ]]; then
  printf 'BENCH_PYTHON is not executable: %s\n' "${BENCH_PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${BENCH_DATASET}" ]]; then
  printf 'BENCH_DATASET is not a directory: %s\n' "${BENCH_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${BENCH_MKL}" ]]; then
  printf 'BENCH_MKL is not a file: %s\n' "${BENCH_MKL}" >&2
  exit 2
fi

cd "${BENCH_REPO}"

# Preserve the already-running 24-thread Auditorium query.  The resumable batch
# will skip it if it succeeds and retry it if it exits without a valid result.
while pgrep -f -- "tools/probe_depth_aabb_fem.py --query-index 3550" >/dev/null; do
  printf '[queue] waiting for active query 3550 at %s\n' "$(date --iso-8601=seconds)"
  sleep 30
done

printf '[queue] starting two-worker oversized batch at %s\n' "$(date --iso-8601=seconds)"
exec env \
  MKL_RT="${BENCH_MKL}" \
  MPLCONFIGDIR=/tmp/matplotlib-exp18-fast-batch \
  "${BENCH_PYTHON}" tools/run_depth_aabb_matched_pilot.py \
    --selection "${BENCH_SELECTION}" \
    --context-manifest "${BENCH_CONTEXT}" \
    --geometry-audit "${BENCH_GEOMETRY}" \
    --dataset-root "${BENCH_DATASET}" \
    --output-dir "${BENCH_OUTPUT}" \
    --rooms Auditorium_idx_1 Cafe_idx_1 \
    --workers 2 \
    --solver-threads 12 \
    --cpu-sets 0-11 12-23 \
    --mkl-runtime "${BENCH_MKL}"
