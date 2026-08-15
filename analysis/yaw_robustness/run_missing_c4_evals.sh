#!/usr/bin/env bash
set -euo pipefail

PINNED_ROOT=/tmp/flac-a6c4-069b72a
MAIN_FLAC=/home/yixunhu/codespace/FLAC
ANALYSIS_ROOT=/home/yixunhu/codespace/cylindrical-dinov3/analysis/yaw_robustness
RUN_ROOT="$ANALYSIS_ROOT/runs/a6c4_40k"
PYTHON_BIN=/home/yixunhu/miniconda3/envs/flac/bin/python
EXPECTED_SOURCE=069b72a5f0ccd43e52551ad4bde1355e8caab92d
FA_SHA=5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328
VAN_SHA=c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c
FA_CKPT="$MAIN_FLAC/outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt"
VAN_CKPT="$MAIN_FLAC/outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=8-step=40000.ckpt"
FA_CONFIG=worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
VAN_CONFIG=worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json
EVENT_LOG="$RUN_ROOT/events.log"
CUDA_PHYSICAL=${CUDA_PHYSICAL:-1}
K_VALUES=${K_VALUES:-"8 1"}
ANGLE_VALUES=${ANGLE_VALUES:-"180 270"}
ARM_VALUES=${ARM_VALUES:-"FA40 VAN40"}
SEED_VALUES=${SEED_VALUES:-"42 43 44 45 46"}
PLOT_ON_COMPLETE=${PLOT_ON_COMPLETE:-1}
WORKER_NAME=${WORKER_NAME:-main}

mkdir -p "$RUN_ROOT/logs"

event() {
    printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$EVENT_LOG"
}

fail() {
    event "FATAL: $*"
    exit 1
}

[[ -d "$PINNED_ROOT" ]] || fail "missing pinned worktree $PINNED_ROOT"
[[ "$(git -C "$PINNED_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE" ]] || fail "wrong source revision"
[[ -x "$PYTHON_BIN" ]] || fail "missing FLAC Python environment"
[[ "$(sha256sum "$FA_CKPT" | awk '{print $1}')" == "$FA_SHA" ]] || fail "FA checkpoint hash mismatch"
[[ "$(sha256sum "$VAN_CKPT" | awk '{print $1}')" == "$VAN_SHA" ]] || fail "Vanilla checkpoint hash mismatch"
[[ -e "$PINNED_ROOT/weights/FLAC/VAE.ckpt" ]] || fail "missing VAE weights bridge"
[[ -e "$PINNED_ROOT/weights/AGREE/AGREE_fullAR.pt" ]] || fail "missing AGREE weights bridge"
[[ -e "$PINNED_ROOT/AcousticRooms" ]] || fail "missing AcousticRooms data bridge"

export CUDA_VISIBLE_DEVICES="$CUDA_PHYSICAL"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_cell() {
    local arm=$1
    local k=$2
    local angle=$3
    local seed=$4
    local ckpt config method dataset_name dataset_config eval_name metrics_path cell_log

    if [[ "$arm" == FA40 ]]; then
        ckpt=$FA_CKPT
        config=$FA_CONFIG
        method=fa_invariant
    else
        ckpt=$VAN_CKPT
        config=$VAN_CONFIG
        method=vanilla
    fi

    if [[ "$k" == 8 ]]; then
        dataset_name=unseeneval
        dataset_config=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
    else
        dataset_name=unseeneval_1
        dataset_config=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
    fi

    eval_name="a6c4_${arm}_rot${angle}_${dataset_name}_s${seed}"
    if [[ "$method" == fa_invariant ]]; then
        metrics_path="${ckpt%.ckpt}_metrics_1_1.0_${eval_name}_fa_invariant_a4_rot${angle}.json"
    else
        metrics_path="${ckpt%.ckpt}_metrics_1_1.0_${eval_name}_rot${angle}.json"
    fi
    cell_log="$RUN_ROOT/logs/${arm}_K${k}_rot${angle}_s${seed}.log"

    if [[ -e "$metrics_path" ]]; then
        if "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_c4_cell.py" "$metrics_path" \
            --arm "$arm" --k "$k" --angle "$angle" --seed "$seed" >>"$cell_log" 2>&1; then
            event "SKIP valid existing cell: $arm K=$k rot=$angle seed=$seed"
            return
        fi
        fail "refusing to overwrite invalid existing artifact $metrics_path"
    fi

    event "START $arm K=$k rot=$angle seed=$seed"
    local -a cmd=(
        "$PYTHON_BIN" eval_FLAC.py
        --model-config "$config"
        --dataset-config "$dataset_config"
        --ckpt-path "$ckpt"
        --cfg-scale 1.0
        --steps 1
        --batch-size 64
        --num-workers 4
        --device cuda
        --eval-name "$eval_name"
        --seed "$seed"
        --rotate-deg "$angle"
        --cond-method "$method"
        --cond-autocast bf16
    )
    if [[ "$method" == fa_invariant ]]; then
        cmd+=(--frame-avg-angles 0,90,180,270)
    fi
    (
        cd "$PINNED_ROOT"
        "${cmd[@]}"
    ) >"$cell_log" 2>&1

    "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_c4_cell.py" "$metrics_path" \
        --arm "$arm" --k "$k" --angle "$angle" --seed "$seed" >>"$cell_log" 2>&1
    event "DONE $arm K=$k rot=$angle seed=$seed metrics=$metrics_path"
}

event "CHAIN START worker=$WORKER_NAME source=$EXPECTED_SOURCE gpu_physical=$CUDA_PHYSICAL k=[$K_VALUES] angles=[$ANGLE_VALUES] arms=[$ARM_VALUES] seeds=[$SEED_VALUES]"

# Finish the paper's primary K=8 plot first, then backfill K=1.
for k in $K_VALUES; do
    for angle in $ANGLE_VALUES; do
        for arm in $ARM_VALUES; do
            for seed in $SEED_VALUES; do
                run_cell "$arm" "$k" "$angle" "$seed"
            done
        done
    done
done

if [[ "$PLOT_ON_COMPLETE" == 1 ]]; then
    MPLCONFIGDIR=/tmp/codex-mpl-cache \
        /usr/bin/python3 "$ANALYSIS_ROOT/plot_yaw_curves.py" --mode c4 >>"$RUN_ROOT/plot.log" 2>&1
    event "CHAIN COMPLETE worker=$WORKER_NAME figure=$ANALYSIS_ROOT/generated/yaw_c4_metrics.png"
else
    event "CHAIN COMPLETE worker=$WORKER_NAME plot=deferred"
fi
