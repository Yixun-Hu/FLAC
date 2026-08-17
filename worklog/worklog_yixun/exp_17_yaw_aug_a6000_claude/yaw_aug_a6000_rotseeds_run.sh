#!/usr/bin/env bash
# exp_17 — fill the per-angle 5-seed table for Yaw-Aug@40k (Yixun 2026-08-17):
# rotations {90,180,270} x seeds {43..46} x {K1,K8} = 24 cells. 0° already has
# 5 seeds; rotated angles had seed 42 only (from the C4 grid). Same protocol
# byte-for-byte; same extras farm/lock. Review: batched into the closure round.
# Written by the main session seat (Claude Fable 5).
set -uo pipefail
EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
FOREIGN_MODEL_CFG="$HOME/codespace/exp-17-yawaug-a6000/${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
EXTRAS="outputs_FLAC/exp17_YAWAUG_extras"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_rotseeds.log"
LOCK="${EXPDIR}/.roteval.lock"; exec 9>"$LOCK"
flock -n 9 || { echo "rotation lock held - abort"; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 rot-seeds (90/180/270 x s43-46 x K1/K8) — ${TS} — $(git rev-parse --short HEAD) ==="
pgrep -f "eval_FLA[C].py" >/dev/null && { echo "eval already running - abort"; exit 2; }
pgrep -f "trai[n].py" >/dev/null && { echo "train.py running - abort"; exit 2; }
[ -f "$(readlink -f "${EXTRAS}/epoch=8-step=40000.ckpt")" ] || { echo "40k ckpt link missing - abort"; exit 2; }

i=0; FAILED=0; PIDS=(); LABELS=()
drain(){ local j; for j in "${!PIDS[@]}"; do wait "${PIDS[$j]}" || { echo "  !! FAILED: ${LABELS[$j]}"; FAILED=$((FAILED+1)); }; done; PIDS=(); LABELS=(); }
for ROT in 90 180 270; do for S in 43 44 45 46; do for K in 1 8; do
  NAME="exp17_YAWAUG_S40000_K${K}_rot${ROT}_seed${S}"
  ls "${EXTRAS}"/*"${NAME}"*.json >/dev/null 2>&1 && { echo "skip: ${NAME}"; continue; }
  [ "$K" = "1" ] && CFG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json" \
                 || CFG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
  echo "[$((i+1))/24] gpu$((i%2)) ${NAME}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$((i%2)) python eval_FLAC.py \
    --model-config "$FOREIGN_MODEL_CFG" --dataset-config "$CFG" \
    --ckpt-path "${EXTRAS}/epoch=8-step=40000.ckpt" \
    --cond-method vanilla --cond-autocast bf16 \
    --rotate-mode fixed --rotate-deg "${ROT}.0" \
    --cfg-scale 1.0 --steps 1 --seed "$S" \
    --eval-name "$NAME" > "${EXPDIR}/roteval_${NAME}.log" 2>&1 &
  PIDS+=("$!"); LABELS+=("$NAME"); i=$((i+1))
  [ $((i%2)) -eq 0 ] && drain
done; done; done
drain
echo "failures: ${FAILED}"
N=$(ls "${EXTRAS}"/*rot{90,180,270}_seed4[3-6]*.json 2>/dev/null | wc -l)
echo "rot-seed JSONs: ${N}/24"
[ "$FAILED" -eq 0 ] && [ "$N" -ge 24 ] && echo "ROTSEEDS COMPLETE" || { echo "INCOMPLETE - re-run to resume"; exit 1; }
