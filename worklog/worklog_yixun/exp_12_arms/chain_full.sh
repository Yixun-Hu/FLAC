#!/bin/bash
# exp_12 full pipeline (Yixun, 2026-08-20), unattended:
#   stage 1: wait current arm B (DDP) -> eval (K=1 || K=8 split across cards)
#   stage 2: arm A re-run from step 0, 2-GPU DDP + SyncBN -> eval
#   stage 3: arm B CURVE re-run from step 0, same recipe -> eval
# Stage 3 exists because arm B's intermediate checkpoints 2,500-40,000 were deleted during
# the disk emergency; a fresh run with FULL retention is the only way to the complete
# acoustic-vs-steps curve. Ordered A-then-B2 per Yixun's explicit instruction.
#
# STANDING RULE: nothing here deletes checkpoints -- every run keeps every checkpoint; the
# disk guard STOPS a run gracefully at a complete checkpoint if space runs out.
# Every handoff hard-stops if its predecessor lacks a 67,500-step checkpoint.
set -uo pipefail

B_PID="${1:?current arm B pid}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/chain_full.log
say () { echo "[chain full] $* | $(date -Is)" >> "$LOG"; }

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 120; done; }
final_ckpt () { ls outputs_FLAC/$1/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1; }

eval_both_k () {   # $1=run  $2=config
  local ck; ck=$(final_ckpt "$1")
  if [ -z "$ck" ]; then
    say "STOP: $1 has no step=67500 checkpoint -- eval skipped, GPUs left free."
    return 3
  fi
  say "$1 eval start (K=1 gpu0 || K=8 gpu1) ckpt=$ck"
  K_VALUES=1 bash $REC/eval_arm.sh "$1" "$2" 0 "$ck" >> $LOG 2>&1 &
  local e1=$!
  K_VALUES=8 bash $REC/eval_arm.sh "$1" "$2" 1 "$ck" >> $LOG 2>&1 &
  local e8=$!
  wait $e1; local r1=$?
  wait $e8; local r8=$?
  say "$1 eval done: K=1 rc=$r1, K=8 rc=$r8, cells $(find outputs_FLAC/$1 -name '*step=67500_metrics*D1*.json' 2>/dev/null | wc -l)/10"
  return 0
}

train_stage () {   # $1=config  $2=run
  say "launching $2 (2-GPU DDP + SyncBN, from step 0, ALL checkpoints retained)"
  local pid
  pid=$(bash $REC/launch_arm_ddp.sh "$1" "$2" | grep -oP '^pid: \K[0-9]+')
  if [ -z "$pid" ]; then say "STOP: $2 launch produced no pid"; return 4; fi
  say "$2 running pid $pid"
  setsid nohup bash $REC/disk_guard.sh "$pid" "$2" 20 > /dev/null 2>&1 &
  say "disk guard armed on $2 (stops gracefully; never deletes)"
  wait_pid "$pid"
  say "$2 pid $pid exited"
  return 0
}

# ---- stage 1: current arm B ---------------------------------------------------------------
say "waiting on current arm B pid $B_PID"
wait_pid "$B_PID"
say "arm B pid $B_PID exited"
eval_both_k exp12B_ssl_cond_ddp "$REC/FLAC_AR_exp12B.json" || exit 3

# ---- stage 2: arm A re-run ----------------------------------------------------------------
train_stage "$REC/FLAC_AR_exp12A.json" exp12A_c3c4_ddp || exit 4
eval_both_k exp12A_c3c4_ddp "$REC/FLAC_AR_exp12A.json" || exit 3

# ---- stage 3: arm B curve re-run ----------------------------------------------------------
train_stage "$REC/FLAC_AR_exp12B.json" exp12B_curve_ddp || exit 4
eval_both_k exp12B_curve_ddp "$REC/FLAC_AR_exp12B.json" || exit 3

say "PIPELINE COMPLETE: current-B evaluated, arm A re-trained+evaluated, arm B curve run"
say "complete with all 27 checkpoints. Curve per-checkpoint evals are a SEPARATE decision."
