#!/usr/bin/env bash
# exp_35 phase (i) watcher v6 (Planner; READ-ONLY: never signals, kills or writes anything; revised after Codex P6-P10).
# Exits -- one EVENT line -- on: wrong arguments; the supervisor gone or no longer the launched process (pid + /proc start
# time parsed after the comm field + cwd + the sha256 of its exact NUL-delimited argv + its fd 1 AND fd 2 still the given
# log's device:inode); the log replaced or unreadable; a stop latch (.stop_latch != null) or an anchored
# "[<iso>] STOP LATCH" event line; a state outside the MILESTONE-RELEVANT SUBSET of H35's schema (an object with a
# stop_latch key; exactly 33 chunks whose (n, kind, cell, gate) equal H35's frozen chunk_plan(), embedded literally below;
# states only planned | in_flight | runner_verified | gates_verified, contiguous: gates_verified*, at most one in_flight |
# runner_verified, then planned*) -- per-state evidence is the supervisor's own schema; or >= N gates_verified chunks.
# v6 (Codex P10 HIGH 1): the state and log ARGUMENTS are bound to the authenticated process -- the state path must equal the
# single "--state" value of the supervisor's argv (canonical, absolute), and the log must be the very file its fd 1 and
# fd 2 write (same device:inode; a link or a copy is refused) -- so another snapshot or log can never stand in.
# Usage: bash "$0" <pid> <start time> <argv sha256> <supervisor log> <state json> <N> [poll s, 10..600]
#        bash "$0" --check-state-file <json>   (the state predicate alone, for smoke tests; prints its verdict, exits)
set -u
PLAN='[[1,"cell","exp35_P1_T_S40000_K1_s42","E1"],[2,"cell","exp35_P1_T_S40000_K8_s42","E1"],[3,"cell","exp35_BF_T_S40000_K1_s42",null],[4,"cell","exp35_BF_T_S40000_K8_s42","E2"],[5,"probe",null,"PROBE"],[6,"cell","exp35_BF_T_S40000_K1_s43",null],[7,"cell","exp35_BF_T_S40000_K8_s43",null],[8,"cell","exp35_BF_T_S40000_K1_s44",null],[9,"cell","exp35_BF_T_S40000_K8_s44",null],[10,"cell","exp35_BF_T_S40000_K1_s45",null],[11,"cell","exp35_BF_T_S40000_K8_s45",null],[12,"cell","exp35_BF_T_S40000_K1_s46",null],[13,"cell","exp35_BF_T_S40000_K8_s46",null],[14,"cell","exp35_BF_R_S40000_K1_s42",null],[15,"cell","exp35_BF_R_S40000_K8_s42",null],[16,"cell","exp35_BF_R_S40000_K1_s43",null],[17,"cell","exp35_BF_R_S40000_K8_s43",null],[18,"cell","exp35_BF_R_S40000_K1_s44",null],[19,"cell","exp35_BF_R_S40000_K8_s44",null],[20,"cell","exp35_BF_R_S40000_K1_s45",null],[21,"cell","exp35_BF_R_S40000_K8_s45",null],[22,"cell","exp35_BF_R_S40000_K1_s46",null],[23,"cell","exp35_BF_R_S40000_K8_s46",null],[24,"cell","exp35_BF_T_S40000_K1_s42_dup",null],[25,"cell","exp35_BF_T_S40000_K8_s42_dup",null],[26,"cell","exp35_BF_C4_S40000_K1_s42_th90",null],[27,"cell","exp35_BF_C4_S40000_K8_s42_th90",null],[28,"cell","exp35_BF_C4_S40000_K1_s42_th180",null],[29,"cell","exp35_BF_C4_S40000_K8_s42_th180",null],[30,"cell","exp35_BF_C4_S40000_K1_s42_th270",null],[31,"cell","exp35_BF_C4_S40000_K8_s42_th270",null],[32,"cell","exp35_BF_C4_S40000_K1_s42_th45",null],[33,"cell","exp35_BF_C4_S40000_K8_s42_th45",null]]'
SCHEMA='def ok_states: (map(.state)) as $s
          | ([$s | to_entries[] | select(.value != "gates_verified") | .key] | first // ($s | length)) as $k
          | ($s[$k:]) as $r
          | if ($r | length) == 0 then true
            elif ($r[0] == "in_flight" or $r[0] == "runner_verified") then ($r[1:] | all(. == "planned"))
            else ($r | all(. == "planned")) end;
        if (type == "object") and has("stop_latch") and has("chunks") and (.chunks | type == "array") and (.chunks | length == 33)
           and ([.chunks | to_entries[] | (.value | type == "object") and (.value.n == .key + 1)
                 and ((.value.state // null) as $x | ["planned", "in_flight", "runner_verified", "gates_verified"] | index($x) != null)] | all)
           and ([.chunks[] | [.n, .kind, .cell, .gate]] == $plan)
           and (.chunks | ok_states)
        then (if .stop_latch != null then "LATCH" else "OK" end) + " " + ([.chunks[] | select(.state == "gates_verified")] | length | tostring)
        else error("schema") end'
verdict() { jq -e -r --argjson plan "$PLAN" "$SCHEMA" "$1" 2>&1; }
if [ "${1:-}" = "--check-state-file" ]; then
  [ "$#" -eq 2 ] || { echo "usage: bash $0 --check-state-file <json>"; exit 2; }
  verdict "$2"; exit $?
fi
bad() { echo "EVENT: WATCHER ARGUMENT ERROR: $*"; exit 2; }
[ "$#" -eq 6 ] || [ "$#" -eq 7 ] || bad "expected 6 or 7 arguments, got $#"
PID="$1"; START="$2"; ARGV_SHA="$3"; LOG="$4"; STATE="$5"; N="$6"; POLL="${7:-60}"
RUN=/home/yixunhu/codespace/exp-35-orbitrir-yaw
[[ "$PID" =~ ^[0-9]+$ ]] || bad "pid $PID"; [[ "$START" =~ ^[0-9]+$ ]] || bad "start time $START"
[[ "$ARGV_SHA" =~ ^[0-9a-f]{64}$ ]] || bad "argv sha256 $ARGV_SHA"
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] && [ "$N" -le 33 ] || bad "N $N"
[[ "$POLL" =~ ^[0-9]+$ ]] && [ "$POLL" -ge 10 ] && [ "$POLL" -le 600 ] || bad "poll $POLL"
[ -f "$LOG" ] && [ ! -L "$LOG" ] || bad "log $LOG is not a regular file (a link is refused)"
INODE="$(stat -c '%d:%i' -- "$LOG" 2>/dev/null)" || bad "log $LOG unreadable"
same_process() {
  local stat; stat="$(cat "/proc/$PID/stat" 2>/dev/null)" || return 1
  [ "$(printf '%s' "${stat##*) }" | awk '{print $20}')" = "$START" ] || return 1     # field 22 = the 20th after "comm) "
  [ "$(readlink "/proc/$PID/cwd" 2>/dev/null)" = "$RUN" ] || return 1
  [ "$(sha256sum < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-64)" = "$ARGV_SHA" ] || return 1
  [ "$(stat -L -c '%d:%i' "/proc/$PID/fd/1" 2>/dev/null)" = "$INODE" ] || return 1     # its stdout IS the given log
  [ "$(stat -L -c '%d:%i' "/proc/$PID/fd/2" 2>/dev/null)" = "$INODE" ] || return 1     # and its stderr
}
same_process || bad "pid $PID is not the launched supervisor writing $LOG (start time, cwd, argv sha256, fd 1 / fd 2)"
mapfile -d '' ARGV < "/proc/$PID/cmdline" || bad "the supervisor's argv is unreadable"
want=""; seen=0
for ((i = 0; i < ${#ARGV[@]}; i++)); do
  if [ "${ARGV[$i]}" = "--state" ]; then seen=$((seen + 1)); want="${ARGV[$((i + 1))]:-}"; fi
done
[ "$seen" -eq 1 ] && [ -n "$want" ] || bad "the supervisor's argv does not carry exactly one --state value"
[ "$want" = "$(realpath -m -- "$want")" ] && [ "${want:0:1}" = / ] || bad "the supervisor's --state $want is not canonical"
[ "$STATE" = "$want" ] || bad "state $STATE is not the supervisor's --state $want"
event=""
while :; do
  if ! same_process; then event="the supervisor (pid $PID, start $START) is gone, no longer the launched process or no longer writing $LOG"; break; fi
  if [ "$(stat -c '%d:%i' -- "$LOG" 2>/dev/null)" != "$INODE" ]; then event="the supervisor log $LOG was replaced or removed"; break; fi
  grep -q -E '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}[-+][0-9:]{5}\] STOP LATCH' "$LOG" 2>/dev/null; g=$?
  if [ "$g" -eq 0 ]; then event="a STOP LATCH event in the log"; break; fi
  if [ "$g" -ne 1 ]; then event="the supervisor log could not be read (grep status $g)"; break; fi
  out="$(verdict "$STATE")"; j=$?
  if [ "$j" -ne 0 ] || ! [[ "$out" =~ ^(LATCH|OK)\ [0-9]+$ ]]; then event="the state is unreadable or malformed: ${out:0:200}"; break; fi
  read -r latch n <<<"$out"
  if [ "$latch" = LATCH ]; then event="the state holds a stop latch"; break; fi
  if [ "$n" -ge "$N" ]; then event="milestone: $n chunks gates_verified (>= $N)"; break; fi
  sleep "$POLL"
done
echo "EVENT: $event"
date -Is
jq -c '{latch: .stop_latch, lane: .lane, states: [.chunks[] | .state]}' "$STATE" 2>&1 | cut -c1-900
tail -n 6 "$LOG" 2>&1 | cut -c1-300
