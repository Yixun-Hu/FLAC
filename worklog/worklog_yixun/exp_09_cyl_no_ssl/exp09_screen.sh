#!/usr/bin/env bash
# ============================================================================
# exp09_screen.sh <step> - screen one exp09_cylNoSSL checkpoint (EMA, then the
# online variant IF its config exists), K=8, eval-seed 42, full unseen split,
# bf16 - the exp-09 analog of exp_07's bf_screen.sh (same reviewed skeleton:
# direct redirect, no pipeline, exit nonzero if any eval failed so a background
# launcher surfaces it). Co-located on GPU 1 with training.
#
# The EMA pass uses FLAC_AR_exp09.json (use_ema=true -> eval_FLAC loads EMA
# weights). The online pass uses FLAC_AR_exp09_online_eval.json; that variant is a
# D-stage artifact (grad-checkpointing off, use_ema=false) and is NOT part of the
# Stage-B deliverable, so the online pass is skipped cleanly until the file exists.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

S="${1:?usage: exp09_screen.sh <step>}"
EXPD="worklog/worklog_yixun/exp_09_cyl_no_ssl"
# recursive: with --logger wandb, train.py:129 nests ckpts under
# <save-dir>/<project>/<run-name>/checkpoints/ (flat with --logger none)
CKPT="$(find outputs_FLAC/exp09_cylNoSSL -name "*step=${S}.ckpt" 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "no ckpt for step ${S} under outputs_FLAC/exp09_cylNoSSL/"; exit 1; }
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPD}/exp09_${TS}_cylNoSSL_screen_S${S}.log"
ONLINE_CFG="${EXPD}/FLAC_AR_exp09_online_eval.json"

{
  echo "=== exp09 screen S${S} (EMA; online if present) - ${TS} - $(git rev-parse --short HEAD) - ckpt ${CKPT} ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_exp09.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp09_cylNoSSL_screen_S${S}_ema"
  echo "=== EMA eval exit $? ==="
  if [ -f "$ONLINE_CFG" ]; then
    CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
      --model-config "$ONLINE_CFG" \
      --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
      --ckpt-path "$CKPT" \
      --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp09_cylNoSSL_screen_S${S}_online"
    echo "=== ONLINE eval exit $? ==="
  else
    echo "=== ONLINE eval SKIPPED (no ${ONLINE_CFG}; D-stage artifact) ==="
  fi
} >> "$LOG" 2>&1

echo "SCREEN S${S} DONE -> ${LOG}"
grep -aE "eval exit [0-9]+" "$LOG"
if grep -aqE "eval exit [1-9]" "$LOG" || ! grep -aqE "eval exit 0" "$LOG"; then
  echo "SCREEN S${S} HAD FAILURES - inspect ${LOG}"
  exit 1
fi
