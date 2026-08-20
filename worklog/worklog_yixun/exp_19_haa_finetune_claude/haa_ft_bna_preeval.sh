#!/usr/bin/env bash
# exp_19-BNA pre-eval (Yixun 2026-08-19): evaluate BNA's ALREADY-STORED ckpts on
# GPU1 while training continues on GPU0. Cells are byte-identical in protocol
# and naming to haa_ft_eval.sh's, so the post-training auto-chain will skip them
# via record validation and only run the S1000 cells. Holds .haa_eval.lock so
# the chain cannot collide; must release before ~00:34 (14 cells ~77 min: OK).
# Addendum-review batch per the small-script rule.
set -uo pipefail
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"
exec 9>"${EXPDIR}/.haa_eval.lock"
flock -n 9 || { echo "eval lock held - abort"; exit 2; }
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
exec > >(tee -a "${EXPDIR}/haa_ft_bna_preeval_${TS}.log") 2>&1
echo "=== BNA pre-eval on GPU1 — ${TS} — $(git rev-parse --short HEAD) ==="
STOCK="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
K8="src/configs/dataset_configs/HAA/eval/haa_test.json"
K1="src/configs/dataset_configs/HAA/eval/haa_test_1.json"
FAILED=0
run() { # step K seed
  local STEP=$1 K=$2 SEED=$3
  local CFG; [ "$K" = 1 ] && CFG="$K1" || CFG="$K8"
  local CKS=( outputs_FLAC/exp19_HAA_BNA/*/*/checkpoints/epoch=*-step=${STEP}.ckpt )
  [ -e "${CKS[0]}" ] && [ "${#CKS[@]}" -eq 1 ] || { echo "!! ckpt glob ${STEP}: ${#CKS[@]}"; FAILED=$((FAILED+1)); return; }
  local NAME="exp19_HAA_BNA_S${STEP}_K${K}_s${SEED}"
  ls "$(dirname "${CKS[0]}")"/*"${NAME}"*.json >/dev/null 2>&1 && { echo "skip ${NAME}"; return; }
  echo "[run] ${NAME}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "$STOCK" --dataset-config "$CFG" --ckpt-path "${CKS[0]}" \
    --cond-method vanilla --cond-autocast bf16 --record-per-scene \
    --cfg-scale 1.0 --steps 1 --seed "$SEED" --eval-name "$NAME" \
    > "${EXPDIR}/eval_${NAME}.log" 2>&1 || { echo "  !! FAILED ${NAME}"; FAILED=$((FAILED+1)); }
}
for S in 42 43 44 45 46; do run 410 8 $S; run 410 1 $S; done   # 端点 S410 全 10 格
for STEP in 100 200 300 500; do run $STEP 8 42; done            # 曲线点
echo "pre-eval done, failures=${FAILED}"
[ "$FAILED" -eq 0 ] || exit 1
