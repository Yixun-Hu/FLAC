#!/bin/bash
# exp_12 per-GPU chain. Detached, so no GPU-day is lost when a stage finishes while the
# agent session is not live.
#
#   bash chain.sh <pid> <run> <model_config> <gpu> <then_arm_B:0|1>
#
# GPU0 (then_arm_B=1):  wait A -> eval A -> SSL -> wait SSL -> B conditioning -> eval B
# GPU1 (then_arm_B=0):  wait C -> eval C
#
# Every handoff verifies its predecessor's artifact and STOPS rather than helpfully filling
# the card: a missing 67.5k checkpoint or a missing SSL export means something needs a human
# look, and the GPU is more useful free than running the next thing on a broken input.
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; CFG="${3:?config}"; GPU="${4:?gpu}"; THEN_B="${5:-0}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/chain_gpu$GPU.log
say () { echo "[chain gpu$GPU] $* | $(date -Is)" >> $LOG; }

wait_pid () { while kill -0 "$1" 2>/dev/null; do sleep 120; done; }

final_ckpt () { ls outputs_FLAC/$1/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1; }

eval_run () {   # $1=run $2=config
  local ck; ck=$(final_ckpt "$1")
  if [ -z "$ck" ]; then
    say "STOP: $1 has NO step=67500 checkpoint -- it did not finish. Eval skipped, GPU left free."
    return 3
  fi
  say "$1 eval start ($ck)"
  bash $REC/eval_arm.sh "$1" "$2" "$GPU" "$ck" >> $LOG 2>&1
  say "$1 eval done rc=$?"
  return 0
}

# ---- stage 1: the arm already training on this card --------------------------------------
say "waiting on $RUN pid $PID"
wait_pid "$PID"
say "$RUN pid $PID exited"
eval_run "$RUN" "$CFG" || exit 3
[ "$THEN_B" = "1" ] || { say "chain complete"; exit 0; }

# ---- stage 2: arm B, SSL ------------------------------------------------------------------
say "arm B SSL launch"
bash $REC/ssl/launch_ssl.sh "$GPU" >> $LOG 2>&1
SSL_PID=$(grep -oP '^pid: \K[0-9]+' $REC/at_launch_exp12B_ssl.txt | tail -1)
if [ -z "$SSL_PID" ]; then say "STOP: SSL launch produced no pid"; exit 4; fi
say "arm B SSL running pid $SSL_PID"
wait_pid "$SSL_PID"
say "arm B SSL pid $SSL_PID exited"

EXPORT=outputs_FLAC/exp12B_ssl/backbone_final.pt
if [ ! -f "$EXPORT" ]; then
  say "STOP: SSL produced no $EXPORT -- conditioning NOT launched, GPU left free."
  exit 5
fi
say "arm B SSL export present ($(stat -c %s "$EXPORT") bytes)"

# ---- stage 3: arm B, C2 conditioning on the SSL backbone -----------------------------------
say "arm B conditioning launch"
B_PID=$(bash $REC/launch_arm.sh "$GPU" "$REC/FLAC_AR_exp12B.json" exp12B_ssl_cond | grep -oP '^pid: \K[0-9]+')
if [ -z "$B_PID" ]; then say "STOP: arm B conditioning launch produced no pid"; exit 6; fi
say "arm B conditioning running pid $B_PID"
wait_pid "$B_PID"
say "arm B conditioning pid $B_PID exited"

# ---- stage 4: arm B eval --------------------------------------------------------------------
eval_run exp12B_ssl_cond "$REC/FLAC_AR_exp12B.json" || exit 3
say "chain complete -- all three arms evaluated"
