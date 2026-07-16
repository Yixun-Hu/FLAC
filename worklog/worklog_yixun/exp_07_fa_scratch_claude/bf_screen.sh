#!/usr/bin/env bash
# ============================================================================
# bf_screen.sh <step> - screen one exp07_BF checkpoint (EMA then online),
# K=8, eval-seed 42, full unseen split, bf16 - the B-F analog of
# bvext_screen.sh (same reviewed skeleton: direct redirect, no pipeline -
# see the tail -0 SIGPIPE note there), co-located on GPU 1 with training.
# Exit nonzero if either eval failed, so a background launcher surfaces it.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

S="${1:?usage: bf_screen.sh <step>}"
EXPD="worklog/worklog_yixun/exp_07_fa_scratch_claude"
CKPT="$(ls outputs_FLAC/exp07_BF/*step="${S}".ckpt 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "no ckpt for step ${S} under outputs_FLAC/exp07_BF/"; exit 1; }
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPD}/fa_scratch_${TS}_BF_screen_S${S}.log"

{
  echo "=== BF screen S${S} (EMA then online) - ${TS} - $(git rev-parse --short HEAD) - ckpt ${CKPT} ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_BF.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp07_BF_screen_S${S}_ema"
  echo "=== EMA eval exit $? ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_BF_online_eval.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp07_BF_screen_S${S}_online"
  echo "=== ONLINE eval exit $? ==="
} >> "$LOG" 2>&1

echo "SCREEN S${S} DONE -> ${LOG}"
grep -aE "eval exit [0-9]+" "$LOG"
if grep -aqE "eval exit [1-9]" "$LOG" || ! grep -aqE "eval exit 0" "$LOG"; then
  echo "SCREEN S${S} HAD FAILURES - inspect ${LOG}"
  exit 1
fi
