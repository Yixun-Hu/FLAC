#!/usr/bin/env bash
# exp_23 — HAA finetune of the CylDINO no-SSL AR-40k init WITH the orientation field (arm CYLORI).
# LEAN launcher (exp_12 precedent: results-first, no gate apparatus). The train.py argv is the
# exp_19 registered HAA recipe VERBATIM (1,000 steps, batch 16 x accum 4, AdamW 5e-6 InverseLR,
# VAE frozen, weights-only init, val/ckpt every 10, seed 42, bf16-mixed); only the model config,
# dataset configs (metadata module adds md['facing']) and the widened init differ from arm CYL.
#   MODE=SMOKE|FULL GPU=0 bash haa_ft_cylori_launch.sh
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
MODE="${MODE:-SMOKE}"; GPU="${GPU:-0}"; ARM="${ARM:-CYLORI}"; FULL_CADENCE="${CADENCE:-10}"; SEED="${SEED:-42}"; DISK_FLOOR="${DISK_FLOOR:-80000}"
case "$ARM" in CYLORI|CYLORI27|P1) ;; *) echo "ARM must be CYLORI, CYLORI27 or P1"; exit 2 ;; esac
# seed 42 keeps the original run names; any other seed gets a _s<seed> tag on names and dirs
TAG=""; [ "$SEED" = "42" ] || TAG="_s${SEED}"
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
CYL_PKG=/home/yixunhu/codespace/cylindrical-dinov3/src
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
case "$MODE" in
  FULL)  STEPS=1000; CADENCE=$FULL_CADENCE; VALEVERY=10; NAME=FLAC_exp23_HAA_${ARM}${TAG}; EXPNAME=exp23_HAA_${ARM}${TAG}; SAVEDIR=outputs_FLAC/exp23_HAA_${ARM}${TAG} ;;
  SMOKE) STEPS=3; CADENCE=1000000; VALEVERY=1000000; NAME=FLAC_exp23_HAA_${ARM}${TAG}_smoke; EXPNAME=exp23_HAA_${ARM}${TAG}_smoke; SAVEDIR=outputs_FLAC/exp23_HAA_${ARM}${TAG}_smoke ;;
  *) echo "MODE must be SMOKE or FULL"; exit 2 ;;
esac
LOG="$E/haa_ft_${TS}_${ARM}${TAG}_${MODE}.log"; RUNLOG="$E/haa_ft_${TS}_${ARM}${TAG}_${MODE}_train.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_23 HAA finetune | ARM=${ARM} SEED=${SEED} MODE=${MODE} GPU=${GPU} cadence=${FULL_CADENCE} | ${TS} ==="
echo "FLAC HEAD: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD)) | dirty: $(git status --porcelain -- src $E | wc -l) tracked-path changes"
echo "cylindrical_dinov3 HEAD: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
if [ "$ARM" = "P1" ]; then   # vanilla FLAC, stock recipe files (exp_19 arm P1), seed-varied
  INIT=outputs_FLAC/exp19_inits/HAA_init_P1.ckpt; INIT_SHA_FILE=worklog/worklog_yixun/exp_19_haa_finetune_claude/exp19_init_shas.txt
  CFG=src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json; DS=src/configs/dataset_configs/HAA/train/haa_train.json; VDS=src/configs/dataset_configs/HAA/eval/haa_val.json
else
  INIT=outputs_FLAC/exp19_inits/HAA_init_CYLORI.ckpt; INIT_SHA_FILE=$E/exp23_init_sha.txt
  CFG=$E/FLAC_HAA_finetune_${ARM}.json; DS=$E/haa_train_ori.json; VDS=$E/haa_val_ori.json
fi
VAE=weights/FLAC/VAE.safetensors
for f in "$INIT" "$CFG" "$DS" "$VDS" "$VAE" "$E/HAA_md_ori.py" "$E/haa_speaker_facing.json" src/models/conditioners.py src/data/yaw_rotation.py train.py; do
  [ -f "$f" ] || { echo "missing $f - abort"; exit 2; }; sha256sum "$f"; done
grep -q "^$(sha256sum "$INIT" | cut -c1-64)  " "$INIT_SHA_FILE" || { echo "INIT sha of $INIT not in $INIT_SHA_FILE - abort"; exit 2; }
# resources (co-tenancy allowed; never touch another process)
VRAM_FLOOR="${VRAM_FLOOR:-20000}"   # measured need of this recipe: ~3.9 GiB (CYLORI/CYLORI27 runs); 20000 is exp_19's conservative floor
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU"); echo "GPU ${GPU} free ${FREE} MiB (floor ${VRAM_FLOOR})"
[ "$FREE" -ge "$VRAM_FLOOR" ] || { echo "not enough free VRAM - abort"; exit 2; }
DISK=$(df -Pm . | awk 'NR==2{print $4}'); echo "disk free ${DISK} MiB (floor ${DISK_FLOOR})"
[ "$DISK" -ge "$DISK_FLOOR" ] || { echo "not enough disk - abort"; exit 2; }
echo "--- co-tenancy at launch ---"; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
[ -e "$SAVEDIR" ] && [ "$MODE" = "FULL" ] && { echo "$SAVEDIR exists - refusing to overwrite a run - abort"; exit 2; }
ARGV=("$PY" train.py --dataset-config "$DS" --val-dataset-config "$VDS" --model-config "$CFG"
  --pretransform-ckpt-path "$VAE" --pretrained-ckpt-path "$INIT"
  --max-steps "$STEPS" --batch-size 16 --accum-batches 4 --num-workers 8 --seed "$SEED" --num-gpus 1 --precision bf16-mixed
  --val-every "$VALEVERY" --checkpoint-every "$CADENCE" --logger wandb --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR")
echo "ARGV: ${ARGV[*]}"
START=$(date +%s)
PYPATH="$CYL_PKG"; [ "$ARM" = "P1" ] && PYPATH=""   # vanilla arm: the cylindrical package is deliberately NOT on the path
env HF_HUB_OFFLINE=1 PYTHONPATH="$PYPATH" CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}" 2>&1 | tee -a "$RUNLOG"
rc="${PIPESTATUS[0]}"
echo "=== exp_23 ${ARM} ${MODE} exit rc=${rc} after $(( $(date +%s) - START ))s at $(date '+%F %T') ==="
NORM="$(mktemp)"; tr '\r' '\n' < "$RUNLOG" > "$NORM"
if [ "$ARM" = "P1" ]; then
  grep -q "Loading ViT model from facebook/dinov3-vits16-pretrain-lvd1689m" "$NORM" && echo "banner: vanilla DINOv3 backbone found" || { echo "!! vanilla banner MISSING - run invalid"; rc=3; }
  grep -q "cylindrical_dinov3\|orientation_field ENABLED" "$NORM" && { echo "!! cylindrical/orientation banner present in a VANILLA run - run invalid"; rc=3; } || echo "banner: no cylindrical/orientation banner (correct for P1)"
else
  grep -q "orientation_field ENABLED: patch conv widened to 6 input channels (scale=$(python3 -c "import json;print(json.load(open('$CFG'))['model']['conditioning']['configs'][1]['config']['orientation_scale'])")" "$NORM" && echo "banner: orientation_field ENABLED found" || { echo "!! orientation_field banner MISSING - run invalid"; rc=3; }
  grep -q "Loading cylindrical_dinov3 ViT" "$NORM" && echo "banner: cylindrical backbone found" || { echo "!! cylindrical banner MISSING - run invalid"; rc=3; }
fi
MARKER="\`Trainer.fit\` stopped: \`max_steps=${STEPS}\` reached."
awk -v m="$MARKER" 'substr($0, length($0)-length(m)+1) == m { found=1 } END { exit found ? 0 : 1 }' "$NORM" && echo "endpoint marker: found (max_steps=${STEPS})" || { echo "!! endpoint marker NOT found"; [ "$rc" -eq 0 ] && rc=4; }
rm -f "$NORM"
if [ "$MODE" = "FULL" ]; then N=$(ls "$SAVEDIR"/*/*/checkpoints/*.ckpt 2>/dev/null | wc -l); echo "checkpoints written: $N (expect $((STEPS/CADENCE)))"; fi
echo "=== launcher done rc=${rc} ==="; exit "$rc"
