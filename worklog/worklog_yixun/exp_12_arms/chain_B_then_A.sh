#!/bin/bash
# exp_12 full pipeline, unattended:
#   wait arm B (DDP) -> eval B (K=1 || K=8 on the two cards)
#   -> train arm A on the SAME 2-GPU DDP + SyncBN recipe from step 0 -> eval A
#
#   setsid nohup bash chain_B_then_A.sh <arm_B_pid> &
#
# Why arm A is re-trained: arm B moved to 2-GPU DDP + SyncBN (BN statistics over 64), while
# the original arm A ran 1-GPU accum-2 (BN over 32). Those are not numerically equivalent --
# this model has 20 BatchNorm modules in the context_audio CNN -- so A-vs-B currently carries
# an execution difference. Re-running A on B's exact recipe removes it, and BOTH then sit on
# the original exp-09 recipe, so A, B and the no-SSL baseline become mutually comparable.
#
# The original arm A run and its results are NOT touched: this trains a separate run name.
# Every stage hard-stops if its predecessor did not produce a 67,500-step checkpoint.
set -uo pipefail

B_PID="${1:?arm B pid}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/chain_B_then_A.log
say () { echo "[chain B->A] $* | $(date -Is)" >> "$LOG"; }

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 120; done; }
final_ckpt () { ls outputs_FLAC/$1/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1; }

eval_both_k () {   # $1=run  $2=config
  local ck; ck=$(final_ckpt "$1")
  if [ -z "$ck" ]; then
    say "STOP: $1 has no step=67500 checkpoint -- eval skipped, GPUs left free."
    return 3
  fi
  say "$1 eval start (K=1 on GPU0, K=8 on GPU1, concurrent) ckpt=$ck"
  K_VALUES=1 bash $REC/eval_arm.sh "$1" "$2" 0 "$ck" >> $LOG 2>&1 &
  local e1=$!
  K_VALUES=8 bash $REC/eval_arm.sh "$1" "$2" 1 "$ck" >> $LOG 2>&1 &
  local e8=$!
  wait $e1; local r1=$?
  wait $e8; local r8=$?
  local n; n=$(find outputs_FLAC/$1 -name "*step=67500_metrics*D1*.json" 2>/dev/null | wc -l)
  say "$1 eval done: K=1 rc=$r1, K=8 rc=$r8, cells on disk $n/10"
  return 0
}

# ---- stage 1: arm B (already running) ----------------------------------------------------
say "waiting on arm B DDP pid $B_PID"
wait_pid "$B_PID"
say "arm B pid $B_PID exited"
eval_both_k exp12B_ssl_cond_ddp "$REC/FLAC_AR_exp12B.json" || exit 3

# ---- stage 2: arm A on arm B's exact recipe ----------------------------------------------
say "launching arm A on 2-GPU DDP + SyncBN (fresh from step 0)"
A_PID=$(bash $REC/launch_arm_ddp.sh "$REC/FLAC_AR_exp12A.json" exp12A_c3c4_ddp | grep -oP '^pid: \K[0-9]+')
if [ -z "$A_PID" ]; then say "STOP: arm A launch produced no pid"; exit 4; fi
say "arm A DDP running pid $A_PID"
setsid nohup bash $REC/disk_guard.sh "$A_PID" exp12A_c3c4_ddp 20 > /dev/null 2>&1 &
# Keep every checkpoint when there is room (a full 27-point checkpoint curve for arm A on
# the DDP recipe is worth 19.5 GB); fall back to the newest 2 if space is tight.
FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
if [ "$FREE_GB" -gt 100 ]; then A_KEEP=999; else A_KEEP=2; fi
say "arm A checkpoint retention: keep=$A_KEEP (free ${FREE_GB} GB)"
setsid nohup bash $REC/ckpt_reaper.sh "$A_PID" exp12A_c3c4_ddp "$A_KEEP" > /dev/null 2>&1 &
say "disk guard + checkpoint reaper armed on arm A"
wait_pid "$A_PID"
say "arm A pid $A_PID exited"

# ---- stage 3: arm A eval -------------------------------------------------------------------
eval_both_k exp12A_c3c4_ddp "$REC/FLAC_AR_exp12A.json" || exit 3
say "PIPELINE COMPLETE -- arm B and arm A both trained and evaluated on the 2-GPU DDP + SyncBN recipe"
