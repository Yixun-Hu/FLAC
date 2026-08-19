#!/usr/bin/env bash
# exp_19 curve eval (Yixun 2026-08-19): FA/Vanilla/Yaw-Aug HAA performance vs
# finetuning steps. Steps 100..900 by 100 (410/1000 already exist from the
# grid), K=8, seed 42, per-arm protocol identical to haa_ft_eval.sh. GPU1 only
# (GPU0 is training YNA). Resume-safe: skips existing non-empty records.
# Batched into the closure review round per the small-script rule.
set -uo pipefail
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/haa_ft_curve_${TS}.log"
exec 9>"${EXPDIR}/.haa_eval.lock"
flock -n 9 || { echo "eval lock held - abort"; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_19 curve eval — ${TS} — $(git rev-parse --short HEAD) — GPU1 only ==="
K_CFG="src/configs/dataset_configs/HAA/eval/haa_test.json"
FAILED=0; RUN=0; SKIP=0
for STEP in 100 200 300 500 600 700 800 900; do :; done  # (docs: 400/410 note below)
for ARM in P1 BF YAW; do
  case "$ARM" in
    P1)  CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"; PROT=(--cond-method vanilla) ;;
    BF)  CFG="${EXPDIR}/FLAC_HAA_finetune_BF.json"; PROT=(--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64) ;;
    YAW) CFG="${EXPDIR}/FLAC_HAA_finetune_YAW.json"; PROT=(--cond-method vanilla) ;;
  esac
  for STEP in 100 200 300 500 600 700 800 900; do
    CKS=( outputs_FLAC/exp19_HAA_${ARM}/*/*/checkpoints/epoch=*-step=${STEP}.ckpt )
    [ -e "${CKS[0]}" ] || { echo "!! no ckpt for ${ARM} S${STEP}"; FAILED=$((FAILED+1)); continue; }
    [ "${#CKS[@]}" -eq 1 ] || { echo "!! ${#CKS[@]} ckpts for ${ARM} S${STEP}"; FAILED=$((FAILED+1)); continue; }
    NAME="exp19_HAA_${ARM}_S${STEP}_K8_s42"
    if ls "$(dirname "${CKS[0]}")"/*"${NAME}"*.json >/dev/null 2>&1; then
      echo "skip: ${NAME}"; SKIP=$((SKIP+1)); continue
    fi
    echo "[run] gpu1 ${NAME}"
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
      --model-config "$CFG" --dataset-config "$K_CFG" \
      --ckpt-path "${CKS[0]}" \
      "${PROT[@]}" --cond-autocast bf16 --record-per-scene \
      --cfg-scale 1.0 --steps 1 --seed 42 \
      --eval-name "$NAME" > "${EXPDIR}/eval_${NAME}.log" 2>&1 \
      || { echo "  !! FAILED ${NAME}"; FAILED=$((FAILED+1)); }
    RUN=$((RUN+1))
  done
done
echo "curve eval done: ran ${RUN}, skipped ${SKIP}, failed ${FAILED} (410/1000 points come from the existing grid)"
[ "$FAILED" -eq 0 ] || exit 1
