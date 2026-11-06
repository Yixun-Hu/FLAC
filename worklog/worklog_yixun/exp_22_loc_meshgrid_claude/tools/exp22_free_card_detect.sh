#!/usr/bin/env bash
# exp_22 R2-C: the READ-ONLY free-card detector (plan_loc_meshgrid_r2_bf_yaw.md §9 A7 r2 (v)(1), r3 (d) (h)).
#
#   bash exp22_free_card_detect.sh <max seconds, 1..7200>
#
# Observes both cards every 60 s under the card rule of exp22_r2c_common.py and exits with exactly ONE stdout line:
#   rc 0  EVENT CANDIDATE <GPU0|GPU1> <uuid> index <i>: free at 31 consecutive polls (GPU 0 first)
#   rc 3  EVENT TIMEOUT (no candidate within the bound: the Planner re-arms; the window restarts, no memory)
#   rc 4  EVENT UNREADABLE (an input of the card rule could not be read: busy, nothing admitted)
#   rc 2  EVENT ARGUMENT ERROR
# It never writes a file, reserves a card, starts a workload or signals a process. The Planner's session runs it as a
# background task (one at a time, re-armed at most every two hours) whose completion is the wake channel; nothing is
# admitted without a Planner session. ONLY the launcher's own sealed capture (exp22_r2c_preflight.sh capture) admits
# a card. The bash process stays the parent (no exec), so the exp_35 launch guard's *free_card* scan still sees it.
#
# Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
set -u
# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
# --- END REGISTERED CONFIG ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! { [ "$#" -eq 1 ] && [[ "$1" =~ ^[0-9]+$ ]] && [ "$((10#$1))" -ge 1 ] && [ "$((10#$1))" -le 7200 ]; }; then
  echo "EVENT ARGUMENT ERROR: usage: bash $0 <max seconds, 1..7200>"
  exit 2
fi
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$HERE/exp22_r2c_common.py" detect --max-seconds "$1"
