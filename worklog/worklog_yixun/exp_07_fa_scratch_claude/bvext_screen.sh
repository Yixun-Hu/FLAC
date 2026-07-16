#!/usr/bin/env bash
# ============================================================================
# bvext_screen.sh <step> - screen one exp07_BVextend checkpoint (EMA then
# online), K=8, eval-seed 42, full unseen split, bf16 - mirrors the phase-1
# screen protocol. Co-located on GPU 1 with the extend training run (safe:
# phase 1 ran identical co-located screens; card has ~37 GiB headroom).
#
# Output: everything appended to a timestamped log via DIRECT REDIRECT.
# (v1 inline driver piped through `tail -0`, which exits without reading ->
# SIGPIPE killed tee and the block before the first eval; do not pipe here.)
# Exit nonzero if either eval failed, so a background launcher surfaces it.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

S="${1:?usage: bvext_screen.sh <step>}"
EXPD="worklog/worklog_yixun/exp_07_fa_scratch_claude"
# recursive: the wandb-logged resume (2026-07-16) nests new ckpts under
# <save-dir>/<project>/<run>/checkpoints/ (train.py:129); pre-resume ckpts are flat
CKPT="$(find outputs_FLAC/exp07_BVextend -name "*step=${S}.ckpt" 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "no ckpt for step ${S} under outputs_FLAC/exp07_BVextend/"; exit 1; }
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPD}/fa_scratch_${TS}_BVext_screen_S${S}.log"

{
  echo "=== BVext screen S${S} (EMA then online) - ${TS} - $(git rev-parse --short HEAD) - ckpt ${CKPT} ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_BV.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp07_BVext_screen_S${S}_ema"
  echo "=== EMA eval exit $? ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_BV_online_eval.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp07_BVext_screen_S${S}_online"
  echo "=== ONLINE eval exit $? ==="
} >> "$LOG" 2>&1

echo "SCREEN S${S} DONE -> ${LOG}"
grep -aE "eval exit [0-9]+" "$LOG"
if grep -aqE "eval exit [1-9]" "$LOG" || ! grep -aqE "eval exit 0" "$LOG"; then
  echo "SCREEN S${S} HAD FAILURES - inspect ${LOG}"
  exit 1
fi
