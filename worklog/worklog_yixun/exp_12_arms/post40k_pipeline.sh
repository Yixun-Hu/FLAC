#!/bin/bash
# exp_12: pipeline after the curve run stops at 40k (Yixun 2026-08-25: "Arm B is only
# needed to run for 40k training steps"). Stages:
#   1. wait for the training to exit (the stop_at_ckpt watcher halts it at 40,000)
#   2. registered curve evals over its retained checkpoints (K={1,8} x s42)
#   3. arm A @40k cells (2) -- the knob-matched SSL isolation at the 40k budget
#   4. P1 vanilla late-window backfill (missing s42 cells, P1's own vanilla protocol)
#   5. NAS-archive the curve run's checkpoints (copy + sha verify; never deletes)
set -uo pipefail
PID="${1:?curve train pid}"
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/post40k_pipeline.log
say () { echo "[post40k] $* | $(date -Is)" >> "$LOG"; }

say "waiting on curve train pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
N=$(ls outputs_FLAC/exp12B_curve_ddp/*/*/checkpoints/*.ckpt 2>/dev/null | wc -l)
TOP=$(ls outputs_FLAC/exp12B_curve_ddp/*/*/checkpoints/*.ckpt 2>/dev/null | sed -E 's/.*step=([0-9]+)\.ckpt/\1/' | sort -n | tail -1)
say "train exited; $N checkpoints, top step ${TOP:-none}"
if [ "${TOP:-0}" -lt 40000 ]; then
  say "STOP: top checkpoint ${TOP:-none} < 40000 -- run did not reach the target; needs a look"
  exit 3
fi

say "stage 2: registered curve evals (exp12B_curve_ddp)"
bash $REC/curve_eval.sh exp12B_curve_ddp 0 1 >> "$LOG" 2>&1
say "curve evals rc=$?"

say "stage 3: arm A @40k cells (knob-matched budget point)"
ACK=$(ls outputs_FLAC/exp12A_c3c4_ddp/*/*/checkpoints/*step=40000.ckpt | head -1)
for K in 1 8; do
  [ "$K" = "8" ] && DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
                 || DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
  NAME="exp12A_c3c4_ddp_CURVE_S40000_K${K}_s42"
  OUT="$(dirname "$ACK")/$(basename "$ACK" .ckpt)_metrics_1_1.0_${NAME}_fa_invariant_a1.json"
  if [ -s "$OUT" ]; then say "SKIP $NAME"; continue; fi
  say "RUN $NAME"
  CUDA_VISIBLE_DEVICES=$((K==8)) python eval_FLAC.py \
    --model-config $REC/FLAC_AR_exp12A.json --dataset-config "$DS" --ckpt-path "$ACK" \
    --cond-method fa_invariant --frame-avg-angles 0 --cond-autocast bf16 \
    --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "$NAME" >> "$LOG" 2>&1
  say "DONE $NAME rc=$?"
done

say "stage 4: P1 vanilla s42 backfill (missing steps only, vanilla protocol)"
( cd /home/yixunhu/codespace/FLAC
  for STEP in $(seq 2500 2500 65000); do
    CK=$(ls outputs_FLAC/exp07_P1/*/*/checkpoints/*step=${STEP}.ckpt 2>/dev/null | head -1)
    [ -n "$CK" ] || continue
    for K in 1 8; do
      # any existing s42 vanilla rot-0 cell for this step+K counts as covered
      if [ "$K" = "1" ]; then PAT="*step=${STEP}_metrics_1_1.0_*K1*s42*.json unseeneval_1"; DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json; else DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json; fi
      COVERED=$(python3 - "$CK" "$K" <<'PY'
import glob, json, os, sys
ck, k = sys.argv[1], sys.argv[2]
d = os.path.dirname(ck); stem = os.path.basename(ck)[:-5]
want_k1 = (k == "1")
for f in glob.glob(f"{d}/{stem}_metrics_*.json"):
    try: r = json.load(open(f))
    except Exception: continue
    if r.get("cond_method") != "vanilla" or float(r.get("rotate_deg", 0)) != 0.0: continue
    n = os.path.basename(f)
    if ("42" not in n): continue
    is_k1 = ("K1" in n) or ("unseeneval_1" in n) or ("_1_s42" in n)
    if is_k1 == want_k1: print("yes"); break
PY
)
      [ "$COVERED" = "yes" ] && continue
      NAME="exp07_P1_CURVE_S${STEP}_K${K}_s42"
      echo "[post40k] RUN $NAME | $(date -Is)" >> /home/yixunhu/codespace/exp-12-arms/worklog/worklog_yixun/exp_12_arms/post40k_pipeline.log
      HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$((K==8)) python eval_FLAC.py \
        --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --dataset-config "$DS" \
        --ckpt-path "$CK" --cond-method vanilla --cond-autocast bf16 \
        --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "$NAME" \
        >> /home/yixunhu/codespace/exp-12-arms/worklog/worklog_yixun/exp_12_arms/post40k_pipeline.log 2>&1
      echo "[post40k] DONE $NAME rc=$? | $(date -Is)" >> /home/yixunhu/codespace/exp-12-arms/worklog/worklog_yixun/exp_12_arms/post40k_pipeline.log
    done
  done )
say "stage 4 done"

say "stage 5: NAS archive of the curve run"
bash $REC/archive_to_nas.sh exp12B_curve_ddp >> "$LOG" 2>&1
say "archive rc=$?"
say "POST-40K PIPELINE COMPLETE"
