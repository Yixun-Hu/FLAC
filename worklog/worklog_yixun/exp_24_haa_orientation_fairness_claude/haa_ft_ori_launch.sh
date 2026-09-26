#!/bin/bash
# exp_24 HAA orientation-cue fairness ablation — fine-tune a STOCK-backbone arm with the facing cue.
#   ARM=P1ORI27  : vanilla FLAC (P1 AR-40k init) + orientation cue s=27       (control for CYLORI27)
#   ARM=YAWORI27 : yaw-augmented FLAC (exp_17 AR-40k init, aug ON in FT) + orientation cue s=27
# exp_19 registered HAA recipe VERBATIM (1,000 steps, batch 16 x accum 4, AdamW 5e-6 InverseLR, VAE frozen,
# val every 10, seed 42, bf16-mixed) = exactly what exp_23 used for CYLORI27; dataset = exp_23's *_ori configs
# (stock HAA configs + md['facing']); the cylindrical package is deliberately NOT on the path.
#   MODE={SMOKE|FULL} GPU={0|1} ARM={P1ORI27|YAWORI27} [CADENCE=100] [SEED=42] bash haa_ft_ori_launch.sh
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
MODE="${MODE:-SMOKE}"; GPU="${GPU:-0}"; ARM="${ARM:-P1ORI27}"; FULL_CADENCE="${CADENCE:-100}"; SEED="${SEED:-42}"; DISK_FLOOR="${DISK_FLOOR:-20000}"; VRAM_FLOOR="${VRAM_FLOOR:-8000}"
case "$ARM" in P1ORI27) INIT_ARM=P1ORI ;; YAWORI27) INIT_ARM=YAWORI ;; *) echo "ARM must be P1ORI27 or YAWORI27"; exit 2 ;; esac
TAG=""; [ "$SEED" = "42" ] || TAG="_s${SEED}"
E=worklog/worklog_yixun/exp_24_haa_orientation_fairness_claude; E23=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
case "$MODE" in
  FULL)  STEPS=1000; CADENCE=$FULL_CADENCE; VALEVERY=10; NAME=FLAC_exp24_HAA_${ARM}${TAG}; EXPNAME=exp24_HAA_${ARM}${TAG}; SAVEDIR=outputs_FLAC/exp24_HAA_${ARM}${TAG} ;;
  SMOKE) STEPS=3; CADENCE=1000000; VALEVERY=1000000; NAME=FLAC_exp24_HAA_${ARM}${TAG}_smoke; EXPNAME=exp24_HAA_${ARM}${TAG}_smoke; SAVEDIR=outputs_FLAC/exp24_HAA_${ARM}${TAG}_smoke ;;
  *) echo "MODE must be SMOKE or FULL"; exit 2 ;;
esac
LOG="$E/haa_ft_${TS}_${ARM}${TAG}_${MODE}.log"; RUNLOG="$E/haa_ft_${TS}_${ARM}${TAG}_${MODE}_train.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_24 HAA finetune | ARM=${ARM} SEED=${SEED} MODE=${MODE} GPU=${GPU} cadence=${FULL_CADENCE} | ${TS} ==="
echo "FLAC HEAD: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD)) | dirty: $(git status --porcelain -- src $E | wc -l) tracked-path changes"
INIT=outputs_FLAC/exp19_inits/HAA_init_${INIT_ARM}.ckpt; INIT_SHA_FILE=$E/exp24_init_sha.txt
CFG=$E/FLAC_HAA_finetune_${ARM}.json; DS=$E23/haa_train_ori.json; VDS=$E23/haa_val_ori.json; VAE=weights/FLAC/VAE.safetensors
for f in "$INIT" "$CFG" "$DS" "$VDS" "$VAE" "$E23/HAA_md_ori.py" "$E23/haa_speaker_facing.json" src/models/conditioners.py src/data/yaw_rotation.py src/training/diffusion.py train.py; do
  [ -f "$f" ] || { echo "missing $f - abort"; exit 2; }; sha256sum "$f"; done
grep -q "^$(sha256sum "$INIT" | cut -c1-64)  " "$INIT_SHA_FILE" || { echo "INIT sha of $INIT not in $INIT_SHA_FILE - abort"; exit 2; }
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU"); echo "GPU ${GPU} free ${FREE} MiB (floor ${VRAM_FLOOR}; measured need of this recipe ~4 GiB)"
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
env HF_HUB_OFFLINE=1 PYTHONPATH="" CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}" 2>&1 | tee -a "$RUNLOG"
rc="${PIPESTATUS[0]}"
echo "=== exp_24 ${ARM} ${MODE} exit rc=${rc} after $(( $(date +%s) - START ))s at $(date '+%F %T') ==="
NORM="$(mktemp)"; tr '\r' '\n' < "$RUNLOG" > "$NORM"
grep -q "Loading ViT model from facebook/dinov3-vits16-pretrain-lvd1689m" "$NORM" && echo "banner: vanilla DINOv3 backbone found" || { echo "!! vanilla banner MISSING - run invalid"; rc=3; }
grep -q "orientation_field ENABLED (vanilla backbone): patch conv widened to 6 input channels (scale=27.0)" "$NORM" && echo "banner: orientation cue (vanilla) found" || { echo "!! orientation-cue banner MISSING - run invalid"; rc=3; }
grep -q "Loading cylindrical_dinov3 ViT" "$NORM" && { echo "!! cylindrical banner present in a VANILLA run - run invalid"; rc=3; } || echo "banner: no cylindrical backbone (as required)"
MARKER="\`Trainer.fit\` stopped: \`max_steps=${STEPS}\` reached."
awk -v m="$MARKER" 'substr($0, length($0)-length(m)+1) == m { found=1 } END { exit found ? 0 : 1 }' "$NORM" && echo "endpoint marker: found (max_steps=${STEPS})" || { echo "!! endpoint marker NOT found"; rc=3; }
rm -f "$NORM"
if [ "$MODE" = "FULL" ]; then N=$(ls "$SAVEDIR"/*/*/checkpoints/*.ckpt 2>/dev/null | wc -l); echo "checkpoints written: $N (expect $((STEPS/CADENCE)))"; fi
echo "=== launcher done rc=${rc} ==="; exit "$rc"
