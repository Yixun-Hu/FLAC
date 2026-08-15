#!/usr/bin/env bash
set -euo pipefail

PINNED_ROOT=/tmp/flac-a6c4-069b72a
MAIN_FLAC=/home/yixunhu/codespace/FLAC
ANALYSIS_ROOT=/home/yixunhu/codespace/cylindrical-dinov3/analysis/checkpoint_curves
RUN_ROOT="$ANALYSIS_ROOT/runs/k1_seed42_yaw0"
PYTHON_BIN=/home/yixunhu/miniconda3/envs/flac/bin/python
EXPECTED_SOURCE=069b72a5f0ccd43e52551ad4bde1355e8caab92d
CUDA_PHYSICAL=${CUDA_PHYSICAL:?set CUDA_PHYSICAL to 0 or 1}
ARM=${ARM:?set ARM to FA or VAN}
STEPS=${STEPS:-"2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000 32500 35000 37500 40000"}

[[ "$ARM" == FA || "$ARM" == VAN ]] || { echo "ARM must be FA or VAN" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs"
EVENT_LOG="$RUN_ROOT/events_${ARM}.log"
MANIFEST="$RUN_ROOT/manifest_${ARM}.tsv"

event() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$EVENT_LOG"
}

fail() {
    event "FATAL $*"
    exit 1
}

[[ -d "$PINNED_ROOT" ]] || fail "missing pinned evaluator worktree"
[[ "$(git -C "$PINNED_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE" ]] || fail "wrong evaluator source"
[[ -x "$PYTHON_BIN" ]] || fail "missing FLAC Python"
[[ -e "$PINNED_ROOT/weights/FLAC/VAE.ckpt" ]] || fail "missing VAE bridge"
[[ -e "$PINNED_ROOT/weights/AGREE/AGREE_fullAR.pt" ]] || fail "missing AGREE bridge"
[[ -e "$PINNED_ROOT/AcousticRooms" ]] || fail "missing AcousticRooms bridge"

if [[ "$ARM" == FA ]]; then
    CKPT_ROOT="$MAIN_FLAC/outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
    CONFIG=worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
    METHOD=fa_invariant
    SKIP_STEPS="25000 40000"
else
    CKPT_ROOT="$MAIN_FLAC/outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints"
    CONFIG=worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json
    METHOD=vanilla
    SKIP_STEPS="40000"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_PHYSICAL"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FRAME_AVG_MAX_FWD_SAMPLES=64

printf 'arm\tstep\tcheckpoint\tsha256\n' >"$MANIFEST.tmp"
for step in $STEPS; do
    if [[ " $SKIP_STEPS " == *" $step "* ]]; then
        continue
    fi
    mapfile -t matches < <(find "$CKPT_ROOT" -maxdepth 1 -type f -name "*-step=${step}.ckpt" -print)
    [[ ${#matches[@]} -eq 1 ]] || fail "expected one checkpoint for $ARM step=$step, found ${#matches[@]}"
    ckpt=${matches[0]}
    printf '%s\t%s\t%s\t%s\n' "$ARM" "$step" "$ckpt" "$(sha256sum "$ckpt" | awk '{print $1}')" >>"$MANIFEST.tmp"
done
mv "$MANIFEST.tmp" "$MANIFEST"

event "CHAIN START arm=$ARM gpu=$CUDA_PHYSICAL source=$EXPECTED_SOURCE cells=$(($(wc -l < "$MANIFEST") - 1))"
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r arm step ckpt ckpt_sha; do
    [[ "$(sha256sum "$ckpt" | awk '{print $1}')" == "$ckpt_sha" ]] || fail "checkpoint changed: $ckpt"
    eval_name="curve0_${arm}_S${step}_K1_s42"
    if [[ "$METHOD" == fa_invariant ]]; then
        metrics_path="${ckpt%.ckpt}_metrics_1_1.0_${eval_name}_fa_invariant_a4.json"
    else
        metrics_path="${ckpt%.ckpt}_metrics_1_1.0_${eval_name}.json"
    fi
    cell_log="$RUN_ROOT/logs/${arm}_S${step}_K1_s42.log"
    if [[ -e "$metrics_path" ]]; then
        "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_curve_cell.py" "$metrics_path" --arm "$arm" --step "$step" >>"$cell_log" 2>&1 \
            || fail "invalid existing artifact: $metrics_path"
        event "SKIP arm=$arm step=$step valid=$metrics_path"
        continue
    fi

    event "START arm=$arm step=$step"
    cmd=(
        "$PYTHON_BIN" eval_FLAC.py
        --model-config "$CONFIG"
        --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
        --ckpt-path "$ckpt"
        --cfg-scale 1.0 --steps 1 --batch-size 64 --num-workers 4
        --device cuda --eval-name "$eval_name" --seed 42 --rotate-deg 0
        --cond-method "$METHOD" --cond-autocast bf16
    )
    if [[ "$METHOD" == fa_invariant ]]; then
        cmd+=(--frame-avg-angles 0,90,180,270)
    fi
    (cd "$PINNED_ROOT" && "${cmd[@]}") >"$cell_log" 2>&1
    "$PYTHON_BIN" "$ANALYSIS_ROOT/validate_curve_cell.py" "$metrics_path" --arm "$arm" --step "$step" >>"$cell_log" 2>&1
    event "DONE arm=$arm step=$step metrics=$metrics_path"
done
event "CHAIN COMPLETE arm=$ARM"

