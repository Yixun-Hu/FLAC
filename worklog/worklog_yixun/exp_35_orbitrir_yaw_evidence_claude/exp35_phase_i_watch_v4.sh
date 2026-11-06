#!/usr/bin/env bash
# exp_35 phase (i) watcher v4 (Planner; READ-ONLY: never signals, kills or writes anything; revised after Codex P6, P7 + P8).
# Exits -- one EVENT line -- on: wrong arguments; the supervisor gone or no longer the launched process (pid + /proc start
# time parsed after the comm field + cwd + the sha256 of its exact NUL-delimited argv); the log replaced (device:inode) or
# unreadable; a stop latch (.stop_latch != null) or an anchored "[<iso>] STOP LATCH" event line; a state that breaks H35's
# schema (an object with a stop_latch key; exactly 33 chunks, n = position; states only planned | in_flight |
# runner_verified | gates_verified, contiguous: gates_verified*, at most one in_flight | runner_verified, then planned*); or
# >= N gates_verified chunks.
# Usage: bash exp35_phase_i_watch_v4.sh <pid> <start time> <argv sha256> <supervisor log> <state json> <N> [poll s, 10..600]
set -u
bad() { echo "EVENT: WATCHER ARGUMENT ERROR: $*"; exit 2; }
[ "$#" -eq 6 ] || [ "$#" -eq 7 ] || bad "expected 6 or 7 arguments, got $#"
PID="$1"; START="$2"; ARGV_SHA="$3"; LOG="$4"; STATE="$5"; N="$6"; POLL="${7:-60}"
RUN=/home/yixunhu/codespace/exp-35-orbitrir-yaw
[[ "$PID" =~ ^[0-9]+$ ]] || bad "pid $PID"; [[ "$START" =~ ^[0-9]+$ ]] || bad "start time $START"
[[ "$ARGV_SHA" =~ ^[0-9a-f]{64}$ ]] || bad "argv sha256 $ARGV_SHA"
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] && [ "$N" -le 33 ] || bad "N $N"
[[ "$POLL" =~ ^[0-9]+$ ]] && [ "$POLL" -ge 10 ] && [ "$POLL" -le 600 ] || bad "poll $POLL"
INODE="$(stat -c '%d:%i' -- "$LOG" 2>/dev/null)" || bad "log $LOG unreadable"
same_process() {
  local stat; stat="$(cat "/proc/$PID/stat" 2>/dev/null)" || return 1
  [ "$(printf '%s' "${stat##*) }" | awk '{print $20}')" = "$START" ] || return 1     # field 22 = the 20th after "comm) "
  [ "$(readlink "/proc/$PID/cwd" 2>/dev/null)" = "$RUN" ] || return 1
  [ "$(sha256sum < "/proc/$PID/cmdline" 2>/dev/null | cut -c1-64)" = "$ARGV_SHA" ] || return 1
}
SCHEMA='def ok_states: (map(.state)) as $s
          | ([$s | to_entries[] | select(.value != "gates_verified") | .key] | first // ($s | length)) as $k
          | ($s[$k:]) as $r
          | if ($r | length) == 0 then true
            elif ($r[0] == "in_flight" or $r[0] == "runner_verified") then ($r[1:] | all(. == "planned"))
            else ($r | all(. == "planned")) end;
        if (type == "object") and has("stop_latch") and has("chunks") and (.chunks | type == "array") and (.chunks | length == 33)
           and ([.chunks | to_entries[] | (.value | type == "object") and (.value.n == .key + 1)
                 and ((.value.state // null) as $x | ["planned", "in_flight", "runner_verified", "gates_verified"] | index($x) != null)] | all)
           and (.chunks | ok_states)
        then (if .stop_latch != null then "LATCH" else "OK" end) + " " + ([.chunks[] | select(.state == "gates_verified")] | length | tostring)
        else error("schema") end'
event=""
while :; do
  if ! same_process; then event="the supervisor (pid $PID, start $START) is gone or no longer the launched process"; break; fi
  if [ "$(stat -c '%d:%i' -- "$LOG" 2>/dev/null)" != "$INODE" ]; then event="the supervisor log $LOG was replaced or removed"; break; fi
  grep -q -E '^\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}[-+][0-9:]{5}\] STOP LATCH' "$LOG" 2>/dev/null; g=$?
  if [ "$g" -eq 0 ]; then event="a STOP LATCH event in the log"; break; fi
  if [ "$g" -ne 1 ]; then event="the supervisor log could not be read (grep status $g)"; break; fi
  out="$(jq -e -r "$SCHEMA" "$STATE" 2>&1)"; j=$?
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
