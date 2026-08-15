#!/usr/bin/env bash
set -euo pipefail

PINNED_ROOT=/tmp/flac-a6c4-069b72a
MAIN_FLAC=/home/yixunhu/codespace/FLAC
ANALYSIS_ROOT=/home/yixunhu/codespace/cylindrical-dinov3/analysis/yaw_robustness
RUN_ROOT="$ANALYSIS_ROOT/runs/a6c4_40k"
PYTHON_BIN=/home/yixunhu/miniconda3/envs/flac/bin/python
CKPT="$MAIN_FLAC/outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=8-step=40000.ckpt"
EVAL_NAME=a6c4_VAN40_rot180_unseeneval_s42
METRICS_PATH="${CKPT%.ckpt}_metrics_1_1.0_${EVAL_NAME}_rot180.json"
CELL_LOG="$RUN_ROOT/logs/VAN40_K8_rot180_s42.log"
PROBE_EVENTS="$RUN_ROOT/two_way_probe.events"

mkdir -p "$RUN_ROOT/logs"
printf '%s | START VAN40 K=8 rot=180 seed=42\n' "$(date --iso-8601=seconds)" >>"$PROBE_EVENTS"

if [[ -e "$METRICS_PATH" ]]; then
    "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_c4_cell.py" "$METRICS_PATH" \
        --arm VAN40 --k 8 --angle 180 --seed 42 >>"$CELL_LOG" 2>&1
else
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    (
        cd "$PINNED_ROOT"
        "$PYTHON_BIN" eval_FLAC.py \
            --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json \
            --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
            --ckpt-path "$CKPT" \
            --cfg-scale 1.0 \
            --steps 1 \
            --batch-size 64 \
            --num-workers 4 \
            --device cuda \
            --eval-name "$EVAL_NAME" \
            --seed 42 \
            --rotate-deg 180 \
            --cond-method vanilla \
            --cond-autocast bf16
    ) >"$CELL_LOG" 2>&1
    "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_c4_cell.py" "$METRICS_PATH" \
        --arm VAN40 --k 8 --angle 180 --seed 42 >>"$CELL_LOG" 2>&1
fi

printf '%s | DONE VAN40 K=8 rot=180 seed=42 metrics=%s\n' \
    "$(date --iso-8601=seconds)" "$METRICS_PATH" >>"$PROBE_EVENTS"
