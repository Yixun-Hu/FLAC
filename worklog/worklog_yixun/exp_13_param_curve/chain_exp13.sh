#!/bin/bash
# exp_13 performance-parameter curve, tier B (ViT-B/16, 85.66M encoder), unattended:
#   train cylB (2-GPU DDP+SyncBN, eff 64, seed 42, 40k) -> eval (fa_invariant [0])
#   -> train vanB (same recipe) -> eval (vanilla path)
#   -> NAS archive both (verify) -> swap local ckpts to symlinks (Yixun's standing
#      pattern, 2026-09-12) -> assemble curve table.
# Tier S of the curve = existing exp07_P1@40k + exp09_cylNoSSL@40k 5-seed cells.
# LEAN mode (standing no-gates directive). Hard stop at every missing artifact.
set -uo pipefail
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
REC=worklog/worklog_yixun/exp_13_param_curve
NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp13_param_curve
LOG=$REC/chain_exp13.log
say () { echo "[exp13] $* | $(date -Is)" >> "$LOG"; }

train () {  # cfg run
  local CFG="$1" RUN="$2"
  say "launch $RUN (2-GPU DDP+SyncBN, 40k)"
  { echo "launched_at: $(date -Is)"; echo "run: $RUN"
    echo "exp12_sha: $(git rev-parse HEAD)"
    echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
    echo "config_sha256: $(sha256sum "$CFG" | cut -d' ' -f1)"
  } > "$REC/at_launch_$RUN.txt"
  nohup python train.py --model-config "$CFG" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger wandb --checkpoint-every 2500 \
    --name "$RUN" --experiment-name "$RUN" --save-dir "outputs_FLAC/$RUN" \
    > "$REC/train_$RUN.log" 2>&1 &
  local PID=$!
  echo "pid: $PID" >> "$REC/at_launch_$RUN.txt"
  say "$RUN pid $PID"
  setsid nohup bash worklog/worklog_yixun/exp_12_arms/disk_guard.sh "$PID" "$RUN" 20 > /dev/null 2>&1 &
  while kill -0 "$PID" 2>/dev/null; do sleep 180; done
  local CK; CK=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*step=40000.ckpt 2>/dev/null | head -1)
  [ -n "$CK" ] || { say "STOP: $RUN has no step=40000 checkpoint"; return 3; }
  say "$RUN complete: $CK"
}

evalrun () {  # run mode(cyl|van)
  local RUN="$1" MODE="$2" CK
  CK=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*step=40000.ckpt 2>/dev/null | head -1)
  [ -n "$CK" ] || { say "STOP: eval $RUN no ckpt"; return 3; }
  local CFG="$REC/FLAC_AR_exp13_${RUN#exp13_}.json"
  say "eval $RUN start (K1 gpu0 || K8 gpu1)"
  for K in 1 8; do
    ( DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
      [ "$K" = "1" ] && DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
      GPU=$((K==8))
      for SEED in 42 43 44 45 46; do
        NAME="${RUN}_D1_K${K}_s${SEED}"
        if [ "$MODE" = "cyl" ]; then
          OUT="$(dirname "$CK")/$(basename "$CK" .ckpt)_metrics_1_1.0_${NAME}_fa_invariant_a1.json"
          [ -s "$OUT" ] && continue
          CUDA_VISIBLE_DEVICES=$GPU python eval_FLAC.py --model-config "$CFG" --dataset-config "$DS" \
            --ckpt-path "$CK" --cond-method fa_invariant --frame-avg-angles 0 \
            --cond-autocast bf16 --seed "$SEED" --steps 1 --cfg-scale 1.0 --eval-name "$NAME" \
            >> "$REC/eval_$RUN.log" 2>&1
        else
          OUT="$(dirname "$CK")/$(basename "$CK" .ckpt)_metrics_1_1.0_${NAME}.json"
          [ -s "$OUT" ] && continue
          CUDA_VISIBLE_DEVICES=$GPU python eval_FLAC.py --model-config "$CFG" --dataset-config "$DS" \
            --ckpt-path "$CK" --cond-method vanilla \
            --cond-autocast bf16 --seed "$SEED" --steps 1 --cfg-scale 1.0 --eval-name "$NAME" \
            >> "$REC/eval_$RUN.log" 2>&1
        fi
        echo "[exp13] cell $NAME done | $(date -Is)" >> "$LOG"
      done ) &
  done
  wait
  say "eval $RUN done: $(ls $(dirname "$CK")/*_metrics_*${RUN}_D1_* 2>/dev/null | wc -l)/10 cells"
}

archive_swap () {  # run
  local RUN="$1"
  mkdir -p "$NAS/$RUN"
  local ok=0
  for f in outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue
    local base tgt; base=$(basename "$f"); tgt="$NAS/$RUN/$base"
    cp "$f" "$tgt.part" && mv "$tgt.part" "$tgt"
    local hl hn; hl=$(sha256sum "$f" | cut -d' ' -f1); hn=$(sha256sum "$tgt" | cut -d' ' -f1)
    if [ "$hl" = "$hn" ]; then
      echo "$hn  $base" >> "$NAS/$RUN/MANIFEST.sha256"
      rm "$f"; ln -s "$tgt" "$f"; ok=$((ok+1))
    else
      say "SHA MISMATCH $RUN/$base -- kept local, NOT swapped"
    fi
  done
  say "$RUN archived+symlinked: $ok checkpoints"
}

train "$REC/FLAC_AR_exp13_cylB.json" exp13_cylB || exit 3
evalrun exp13_cylB cyl || exit 3
train "$REC/FLAC_AR_exp13_vanB.json" exp13_vanB || exit 3
evalrun exp13_vanB van || exit 3
archive_swap exp13_cylB
archive_swap exp13_vanB
say "EXP13 TIER-B COMPLETE -- curve assembly is analysis-side"
