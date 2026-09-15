#!/usr/bin/env bash
# Yixun 2026-09-15 09:4x EDT: "co-tenant when free". Poll both cards every 60 s; the first card
# showing >= 8 GB free on THREE consecutive polls (skips the 1-2 min gap between another queue's
# jobs, which would otherwise lure us onto a card about to be refilled) gets the seed-43 chain.
# GPU 1 is checked first (its xRIR job ends first). Gives up after 72 h.
cd /home/yixunhu/codespace/FLAC
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
echo "watcher start $(date -Is) pid $$"
declare -A OK=([0]=0 [1]=0); t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 259200 ]; do
  for g in 1 0; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g 2>/dev/null || echo 0)
    if [ "${free:-0}" -ge 8000 ]; then OK[$g]=$(( ${OK[$g]} + 1 )); else OK[$g]=0; fi
    if [ "${OK[$g]}" -ge 3 ]; then
      echo "GPU $g free ${free} MiB on 3 consecutive polls -> launching seed-43 chain $(date -Is)"
      GPU=$g bash $E/chain_seed43.sh; rc=$?; echo "WATCHER: chain exited rc=$rc $(date -Is)"; exit $rc
    fi
  done
  sleep 60
done
echo "WATCHER GAVE UP after 72 h $(date -Is)"; exit 9
