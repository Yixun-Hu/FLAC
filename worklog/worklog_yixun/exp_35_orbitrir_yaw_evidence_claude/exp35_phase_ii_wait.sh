#!/usr/bin/env bash
# exp_35 phase (ii): a READ-ONLY notification loop for the Planner. It never reserves, launches, signals or writes
# anything; the kit's `reserve` re-checks every precondition itself. It prints one EVENT line and exits 0 when
#   (1) the LAST s43 lifecycle line of Yixun's queue log first becomes its END line (the kit's exact predicate,
#       exp35_phase_ii.py::s43_end: S43_LINE fullmatch, then '\S+ END training ACEattn s43 exit=-?\d+'), or
#   (2) that holds AND GPU 0 shows no compute process at two checks 60 s apart, or
#   (3) the queue log is unreadable or nvidia-smi fails, or
#   (4) its own window (argument 1, seconds, 60..7000) ends.
set -u
MAX=${1:-}
if (( $# != 1 )) || ! [[ $MAX =~ ^[0-9]+$ ]] || (( MAX < 60 || MAX > 7000 )); then
    echo "REFUSED: usage: exp35_phase_ii_wait.sh <max-seconds 60..7000>"; exit 2
fi
Q=/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/queue_20261101T233607_741947.log
G0=GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e
last_s43() { LC_ALL=C grep -E '^[^[:space:]]+ (START|END) training ACEattn s43( .*)?$' "$Q" | tail -1; }
is_end() { [[ $1 =~ ^[^[:space:]]+\ END\ training\ ACEattn\ s43\ exit=-?[0-9]+$ ]]; }
deadline=$(( $(date +%s) + MAX ))
[[ -r $Q ]] || { echo "EVENT queue-log-unreadable $(date -Is): $Q"; exit 0; }
last=$(last_s43)
if is_end "$last"; then was_ended=1; else was_ended=0; fi
echo "armed $(date -Is) for ${MAX} s: s43 ended at arm time = $was_ended (its last s43 line: ${last:-none})"
free=0
while :; do
    [[ -r $Q ]] || { echo "EVENT queue-log-unreadable $(date -Is): $Q"; exit 0; }
    last=$(last_s43)
    if is_end "$last"; then ended=1; else ended=0; fi
    if (( ended == 1 && was_ended == 0 )); then echo "EVENT s43-ended $(date -Is): $last"; exit 0; fi
    apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>&1) \
        || { echo "EVENT nvidia-smi-failed $(date -Is): $apps"; exit 0; }
    if grep -q -F "$G0" <<<"$apps"; then free=0; else free=$(( free + 1 )); fi
    if (( ended == 1 && free >= 2 )); then
        echo "EVENT gpu0-free $(date -Is): no compute process on GPU 0 at two checks 60 s apart; s43's last line is END"; exit 0
    fi
    if (( $(date +%s) + 60 > deadline )); then echo "EVENT window-ended $(date -Is): no event (ended=$ended free=$free)"; exit 0; fi
    sleep 60
done
