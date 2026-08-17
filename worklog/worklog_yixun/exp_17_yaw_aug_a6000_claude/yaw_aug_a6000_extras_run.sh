#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_extras_run.sh — exp_17 decisions (a)+(c), Yixun 2026-08-16.
#
# (a) Yaw-Aug@40k, 0°, eval seeds 43–46, both K  -> 8 cells. Purpose: turn the
#     single-seed endpoint into a 5-seed row admissible to model_comparison.md
#     (gen_model_comparison structurally excludes single-seed screens).
# (seen) Yaw-Aug@40k, SEEN split, 0°, seeds 42–46, both K -> 10 cells. Purpose:
#     fill the Yaw-Aug rows of Yixun's seen-split table alongside exp_18's
#     P1/B-F seen rows (protocol verified identical: vanilla+bf16).
# (c) Yaw-Aug@40k, 45°, seed 42, both K          -> 2 cells. Purpose: probe
#     OFF the C4 orbit. The augmentation draws uniformly over all 512 columns,
#     so it may be flat where exact-C4 frame-averaging is not — 45° is exp_07
#     A6's negative control angle, making the numbers directly comparable.
#
# These cells live in their OWN directory (exp17_YAWAUG_extras), NOT the grid
# farm: the orbit aggregator hard-errors on foreign seeds and would drop the
# S40000 orbit if a 45° record joined its rotation set — both by design.
#
# Protocol is byte-identical to the reviewed grid cells (vanilla, bf16, cfg 1.0,
# steps 1, fixed rotation) — only --seed / --rotate-deg / --eval-name vary.
# Takes the SHARED .roteval.lock so it can never overlap another grid.
#
# Written by the main session seat (Claude Fable 5). Review: batched into the
# exp_17 closure round per the small-script consolidation rule.
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
FOREIGN_MODEL_CFG="$HOME/codespace/exp-17-yawaug-a6000/${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
GRID_FARM="outputs_FLAC/exp17_YAWAUG_roteval"
EXTRAS="outputs_FLAC/exp17_YAWAUG_extras"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_extras.log"

LOCK="${EXPDIR}/.roteval.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "rotation lock held - abort"; exit 2; }

mkdir -p "$EXTRAS"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 extras (a: 5-seed endpoint, c: 45° probe) — ${TS} — $(git rev-parse --short HEAD) ==="

pgrep -f "eval_FLAC.py" >/dev/null && { echo "eval_FLAC already running - abort"; exit 2; }
pgrep -f "train.py" >/dev/null && { echo "train.py running on this box - abort"; exit 2; }

CKPT_REAL="$(readlink -f "${GRID_FARM}/epoch=8-step=40000.ckpt")"
[ -f "$CKPT_REAL" ] || { echo "40k checkpoint not resolvable - abort"; exit 2; }
ln -sfn "$CKPT_REAL" "${EXTRAS}/epoch=8-step=40000.ckpt"
echo "checkpoint: ${EXTRAS}/epoch=8-step=40000.ckpt -> ${CKPT_REAL}"

# cell spec: "<split> <K> <seed> <rot>"
CELLS=()
for S in 43 44 45 46; do CELLS+=("unseen 1 $S 0" "unseen 8 $S 0"); done
CELLS+=("unseen 1 42 45" "unseen 8 42 45")
for S in 42 43 44 45 46; do CELLS+=("seen 1 $S 0" "seen 8 $S 0"); done

run_cell() {  # gpu split K seed rot
  local GPU="$1" SPLIT="$2" K="$3" SEED="$4" ROT="$5"
  local CFG NAME
  if [ "$SPLIT" = "unseen" ]; then
    [ "$K" = "1" ] && CFG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"                    || CFG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
    NAME="exp17_YAWAUG_S40000_K${K}_rot${ROT}_seed${SEED}"
  else
    [ "$K" = "1" ] && CFG="src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json"                    || CFG="src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json"
    NAME="exp17_YAWAUG_seen_S40000_K${K}_s${SEED}"
  fi
  echo "gpu${GPU} ${NAME}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" python eval_FLAC.py \
    --model-config "$FOREIGN_MODEL_CFG" \
    --dataset-config "$CFG" \
    --ckpt-path "${EXTRAS}/epoch=8-step=40000.ckpt" \
    --cond-method vanilla --cond-autocast bf16 \
    --rotate-mode fixed --rotate-deg "${ROT}.0" \
    --cfg-scale 1.0 --steps 1 --seed "$SEED" \
    --eval-name "$NAME" > "${EXPDIR}/roteval_${NAME}.log" 2>&1
}

i=0; FAILED=0; PIDS=(); LABELS=()
drain() {
  local j
  for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { echo "  !! FAILED: ${LABELS[$j]}"; FAILED=$((FAILED+1)); }
  done
  PIDS=(); LABELS=()
}
for SPEC in "${CELLS[@]}"; do
  read -r SPLIT K SEED ROT <<<"$SPEC"
  if [ "$SPLIT" = "unseen" ]; then NM="exp17_YAWAUG_S40000_K${K}_rot${ROT}_seed${SEED}";
  else NM="exp17_YAWAUG_seen_S40000_K${K}_s${SEED}"; fi
  if ls "${EXTRAS}"/*"${NM}"*.json >/dev/null 2>&1; then
    echo "skip (exists): ${NM}"; continue
  fi
  run_cell $((i % 2)) "$SPLIT" "$K" "$SEED" "$ROT" &
  PIDS+=("$!"); LABELS+=("${SPLIT}_K${K}_rot${ROT}_s${SEED}")
  i=$((i+1))
  [ $((i % 2)) -eq 0 ] && drain
done
drain
echo "failures: ${FAILED}"
N="$(ls "${EXTRAS}"/*.json 2>/dev/null | grep -vc stream || true)"
echo "extras JSONs present: ${N}/20"
[ "$FAILED" -eq 0 ] && [ "$N" -ge 20 ] && echo "EXTRAS COMPLETE" || { echo "EXTRAS INCOMPLETE - re-run to resume"; exit 1; }
