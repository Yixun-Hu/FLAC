#!/usr/bin/env bash
# exp_35 phase (ii): a READ-ONLY notification loop for the Planner -- v2 (Codex P17 NIT 1-3; v1 exp35_phase_ii_wait.sh
# stays unchanged). It never reserves, launches, signals or writes anything; the kit's `reserve` re-checks every
# precondition itself. It prints exactly one EVENT line (its detail flattened to one line) and exits 0 when
#   (1) the LAST s43 lifecycle line of Yixun's queue log first becomes its END line (the kit's exact predicate,
#       exp35_phase_ii.py::s43_end: S43_LINE fullmatch, then '\S+ END training ACEattn s43 exit=-?\d+'), or
#   (2) that holds AND, at every check over at least 60 s, every nvidia-smi compute-app row is well formed (the kit's
#       ROW) and none is on GPU 0, or
#   (3) the queue log cannot be read, or nvidia-smi fails or prints a malformed row, or
#   (4) its window (argument 1: decimal seconds 60..7000, at most four digits) ends -- after a final check AT the deadline.
set -u
export LC_ALL=C
if (( $# != 1 )) || ! [[ $1 =~ ^[0-9]{1,4}$ ]] || (( 10#$1 < 60 || 10#$1 > 7000 )); then
    echo "REFUSED: usage: exp35_phase_ii_wait_v2.sh <max-seconds 60..7000>"; exit 2
fi
MAX=$(( 10#$1 ))
Q=/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/queue_20261101T233607_741947.log
G0=GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e
flat() { local d=${1//[[:cntrl:]]/ }; printf '%s' "${d:0:600}"; }
event() { echo "EVENT $1 $(date -Is): $(flat "$2")"; exit 0; }
s43_last() {   # stdout: the last s43 lifecycle line ('' if none); rc 1 when the queue log cannot be read
    local out rc
    out=$(grep -a -E '^[^[:space:]]+ (START|END) training ACEattn s43( .*)?$' "$Q"); rc=$?
    (( rc <= 1 )) || return 1
    printf '%s' "${out##*$'\n'}"
}
is_end() { [[ $1 =~ ^[^[:space:]]+\ END\ training\ ACEattn\ s43\ exit=-?[0-9]+$ ]]; }
gpu0_state() {   # stdout: free | busy | 'failed: rc=<rc>' | 'malformed: <row>' (stdout parsed, stderr left to the log)
    local apps rc row state=free
    apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader); rc=$?
    (( rc == 0 )) || { printf 'failed: rc=%s' "$rc"; return; }
    while IFS= read -r row; do
        [[ $row =~ ^[[:space:]]*$ ]] && continue
        if [[ $row =~ ^[[:space:]]*([0-9]+),[[:space:]]*(GPU-[0-9A-Za-z-]+)[[:space:]]*$ ]]; then
            [[ ${BASH_REMATCH[2]} == "$G0" ]] && state=busy
        else
            printf 'malformed: %s' "$row"; return
        fi
    done <<<"$apps"
    printf '%s' "$state"
}
deadline=$(( $(date +%s) + MAX ))
last=$(s43_last) || event queue-log-unreadable "$Q"
if is_end "$last"; then was_ended=1; else was_ended=0; fi
echo "armed $(date -Is) for ${MAX} s (v2): s43 ended at arm time = $was_ended (its last s43 line: $(flat "${last:-none}"))"
free_since=0
while :; do
    now=$(date +%s)
    last=$(s43_last) || event queue-log-unreadable "$Q"
    if is_end "$last"; then ended=1; else ended=0; fi
    if (( ended == 1 && was_ended == 0 )); then event s43-ended "$last"; fi
    g=$(gpu0_state)
    case $g in
        free) (( free_since > 0 )) || free_since=$now ;;
        busy) free_since=0 ;;
        *) event nvidia-smi-unusable "$g" ;;
    esac
    if (( ended == 1 && free_since > 0 && now - free_since >= 60 )); then
        event gpu0-free "no compute process on GPU 0 at every check since $(date -d "@$free_since" -Is); s43's last line is END"
    fi
    if (( now >= deadline )); then event window-ended "no event (ended=$ended free_since=$free_since)"; fi
    rem=$(( deadline - now ))
    sleep $(( rem < 60 ? rem : 60 ))
done
