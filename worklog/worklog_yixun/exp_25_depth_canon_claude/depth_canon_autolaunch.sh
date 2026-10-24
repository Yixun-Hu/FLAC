#!/usr/bin/env bash
# ============================================================================
# depth_canon_autolaunch.sh -- exp_25 depth_canon: the unattended watcher that fires the
# CANON training (MODE=SMOKE, then MODE=FULL of depth_canon_launch.sh) the moment both A6000s
# are free, with no human in the loop (Yixun pre-approved the GO, 2026-10-24).
#
#   GO token   $EXPDIR/depth_canon_GO.token, written by the Planner, never by this watcher:
#              head=<40 hex> / manifest_sha256=<sha256 of depth_canon_pins.sha256's bytes> /
#              issued=<ISO-8601> / by=planner, one key=value per line. Valid only while head ==
#              `git rev-parse HEAD` and manifest_sha256 == the pins file's sha256, re-checked on
#              every poll AND immediately before each launcher call (a commit after arming refuses).
#   GPUs       GPU 0 and GPU 1 each: ZERO compute processes (uuid -> index mapped) AND >= 23040 MiB
#              free; any nvidia-smi failure or unparsable output is DIRTY. Fire after 3 consecutive
#              clean polls (60 s apart); a dirty poll resets the count. Foreign processes are logged
#              once per change (pid, used memory, readlink /proc/<pid>/cwd); a heartbeat every 30 min.
#   Fire       SMOKE (output tee'd into the watcher log) -> rc 0 -> FULL -> rc 0: DONE (exit 0); FULL
#              rc != 0: FAILED (exit 1). SMOKE refused by the launcher's own GPU gates (the two
#              strings below) is transient: back to WAITING_GPU, at most 5 retries; any other SMOKE
#              failure is FAILED (exit 1) and FULL is never attempted.
#   State      $EXPDIR/depth_canon_autolaunch_state.json, replaced atomically (tmp + mv) on every
#              change; log $EXPDIR/depth_canon_<TS>_autolaunch.log.
#   Never      terminates or signals any process, writes outside $EXPDIR, or touches the NAS (the
#              launcher does, behind its own gates).
#   Harness    DEPTH_CANON_HARNESS=<dir under /tmp with .depth_canon_harness> (tests only) enables
#              GO_TOKEN / AUTOLAUNCH_LAUNCHER (both under /tmp) / AUTOLAUNCH_POLL_S /
#              AUTOLAUNCH_STABLE_POLLS / AUTOLAUNCH_MAX_POLLS (exit 3 "poll budget exhausted").
#              Outside the harness any of them is refused (exit 2): production values are fixed.
# Exit: 0 DONE | 1 FAILED | 2 a precondition refused | 3 harness poll budget exhausted.
#
# Written by the exp_25 Coder seat (Claude Opus 5.5, max effort), round 3d.
# ============================================================================
set -euo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "$REPO"
EXPDIR="worklog/worklog_yixun/exp_25_depth_canon_claude"
LAUNCHER="$EXPDIR/depth_canon_launch.sh"
PINS="$EXPDIR/depth_canon_pins.sha256"
TOKEN="$EXPDIR/depth_canon_GO.token"
STATE="$EXPDIR/depth_canon_autolaunch_state.json"
MIN_GPU_FREE_MIB=23040           # = depth_canon_launch.sh MIN_GPU_FREE_MIB (22.5 GiB)
POLL_S=60; STABLE_POLLS=3; MAX_POLLS=0; HEARTBEAT_S=1800
MAX_GPU_RETRIES=5                # fixed
# The launcher's two transient GPU refusals (depth_canon_launch.sh): the memory floor
# ("GPU GATE FAILED: GPU <g> free ...", gate (8)) and the occupancy gate of r3b-F1 fix 7
# ("GPU OCCUPIED: foreign compute process(es) on GPU 0/1 ..."). Defined ONCE, here.
GPU_GATE_STR="GPU GATE FAILED"
GPU_OCCUPIED_STR="GPU OCCUPIED: foreign compute process"

die() { echo "$*"; exit 2; }

# --- preconditions -------------------------------------------------------------------- #
HARNESS="${DEPTH_CANON_HARNESS:-}"
under_tmp() { case "$(readlink -f -- "$1" 2>/dev/null)" in /tmp/?*) return 0 ;; esac; return 1; }
if [ -n "$HARNESS" ]; then
  HREAL="$(readlink -f -- "$HARNESS" 2>/dev/null || true)"
  case "$HREAL" in /tmp/?*) ;; *) die "DEPTH_CANON_HARNESS must be a directory under /tmp (got '${HARNESS}') - abort" ;; esac
  [ -f "$HREAL/.depth_canon_harness" ] || die "DEPTH_CANON_HARNESS '${HARNESS}' lacks the .depth_canon_harness marker - abort"
  if [ -n "${GO_TOKEN:-}" ]; then under_tmp "$GO_TOKEN" || die "GO_TOKEN must be under /tmp - abort"; TOKEN="$GO_TOKEN"; fi
  if [ -n "${AUTOLAUNCH_LAUNCHER:-}" ]; then
    under_tmp "$AUTOLAUNCH_LAUNCHER" || die "AUTOLAUNCH_LAUNCHER must be a path under /tmp - abort"
    LAUNCHER="$AUTOLAUNCH_LAUNCHER"
  fi
  POLL_S="${AUTOLAUNCH_POLL_S:-$POLL_S}"; STABLE_POLLS="${AUTOLAUNCH_STABLE_POLLS:-$STABLE_POLLS}"
  MAX_POLLS="${AUTOLAUNCH_MAX_POLLS:-$MAX_POLLS}"
  [[ "$POLL_S" =~ ^[0-9]+(\.[0-9]+)?$ && "$STABLE_POLLS" =~ ^[1-9][0-9]*$ && "$MAX_POLLS" =~ ^[0-9]+$ ]] \
    || die "AUTOLAUNCH_POLL_S / _STABLE_POLLS / _MAX_POLLS must be numbers - abort"
else
  for v in GO_TOKEN AUTOLAUNCH_LAUNCHER AUTOLAUNCH_POLL_S AUTOLAUNCH_STABLE_POLLS AUTOLAUNCH_MAX_POLLS; do
    [ -z "${!v:-}" ] || die "${v} is honoured only under DEPTH_CANON_HARNESS (production values are fixed) - abort"
  done
fi
exec 8>"$EXPDIR/.depth_canon_autolaunch.lock" || die "cannot open ${EXPDIR}/.depth_canon_autolaunch.lock - abort"
flock -n 8 || die "another autolaunch watcher holds ${EXPDIR}/.depth_canon_autolaunch.lock - abort"
PY="$(command -v python 2>/dev/null || true)"
case "$PY" in */envs/flac/bin/python*) ;; *) die "'python' on PATH is '${PY:-absent}', not the flac env (*/envs/flac/bin/python) - abort" ;; esac
[ -f "$LAUNCHER" ] || die "the launcher ${LAUNCHER} is missing - abort"
command -v nvidia-smi >/dev/null || die "nvidia-smi is not on PATH - abort"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="$EXPDIR/depth_canon_${TS}_autolaunch.log"
exec > >(tee -a "$LOG") 2>&1
STARTED="$(date -Iseconds)"
echo "=== exp_25 depth_canon autolaunch -- ${TS} -- pid $$ -- launcher ${LAUNCHER} -- harness=${HARNESS:-none} ==="
echo "poll ${POLL_S} s, fire after ${STABLE_POLLS} clean polls, <= ${MAX_GPU_RETRIES} GPU-gate retries, min free ${MIN_GPU_FREE_MIB} MiB"

# --- state (atomic: tmp + mv) ------------------------------------------------------------ #
PHASE=""; REASON=""; TOKEN_HEAD=""; CLEAN=0; RETRIES=0; SMOKE_RC=""; FULL_RC=""; LAST_STATE=""
write_state() {   # write_state <phase> <reason>: only when something changed
  PHASE="$1"; REASON="$2"
  local key="${PHASE}|${REASON}|${TOKEN_HEAD}|${CLEAN}|${RETRIES}|${SMOKE_RC}|${FULL_RC}"
  [ "$key" != "$LAST_STATE" ] || return 0
  LAST_STATE="$key"
  "$PY" - "$STATE.tmp.$$" "$PHASE" "$REASON" "$$" "$STARTED" "$TOKEN_HEAD" "$CLEAN" "$RETRIES" \
      "$SMOKE_RC" "$FULL_RC" "$LOG" <<'PY'
import datetime, json, sys
out, phase, reason, pid, started, head, clean, retries, smoke, full, log = sys.argv[1:12]
num = lambda v: int(v) if v != "" else None  # noqa: E731
st = {"schema_version": 1, "kind": "depth_canon_autolaunch_state", "phase": phase, "reason": reason,
      "watcher_pid": int(pid), "started": started,
      "updated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "token_head": head or None, "clean_polls": int(clean), "gpu_retries": int(retries),
      "smoke_rc": num(smoke), "full_rc": num(full), "log": log}
with open(out, "w") as f:
    json.dump(st, f, indent=1)
    f.write("\n")
PY
  mv -f "$STATE.tmp.$$" "$STATE"
  echo "[$(date -Iseconds)] phase ${PHASE}: ${REASON}"
}

# --- the GO token ------------------------------------------------------------------------ #
token_ok() {   # 0 = valid (TOKEN_HEAD set); 1 = invalid (TOKEN_WHY set)
  local line key val head="" man="" issued="" by="" n=0 cur
  TOKEN_WHY=""
  [ -f "$TOKEN" ] || { TOKEN_WHY="no GO token at ${TOKEN}"; return 1; }
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    key="${line%%=*}"; val="${line#*=}"
    [ "$key" != "$line" ] || { TOKEN_WHY="GO token line '${line}' is not key=value"; return 1; }
    case "$key" in
      head) head="$val" ;; manifest_sha256) man="$val" ;; issued) issued="$val" ;; by) by="$val" ;;
      *) TOKEN_WHY="GO token has an unknown key '${key}'"; return 1 ;;
    esac
    n=$((n + 1))
  done < "$TOKEN"
  [ "$n" -eq 4 ] || { TOKEN_WHY="GO token must hold exactly head, manifest_sha256, issued, by (got ${n} lines)"; return 1; }
  [[ "$head" =~ ^[0-9a-f]{40}$ && "$man" =~ ^[0-9a-f]{64}$ && "$issued" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] \
    && [ "$by" = planner ] || { TOKEN_WHY="GO token is malformed (head / manifest_sha256 / issued / by)"; return 1; }
  cur="$(git rev-parse HEAD)"
  [ "$head" = "$cur" ] || { TOKEN_WHY="GO token head ${head} != HEAD ${cur} (re-issue the token at this HEAD)"; return 1; }
  cur="$(sha256sum "$PINS" 2>/dev/null | awk '{print $1}')"
  [ "$man" = "$cur" ] || { TOKEN_WHY="GO token manifest_sha256 ${man} != ${PINS} sha256 ${cur:-unreadable}"; return 1; }
  TOKEN_HEAD="$head"
}

# --- the GPUs ---------------------------------------------------------------------------- #
gpu_clean() {   # 0 = both cards idle with enough memory; 1 = dirty (GPU_WHY set)
  local uuids apps g u line uuid pid pname mem free busy=""
  declare -A IDX=()
  GPU_WHY=""
  uuids="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)" \
    || { GPU_WHY="nvidia-smi --query-gpu=index,uuid failed"; return 1; }
  for g in 0 1; do
    u="$(awk -F', *' -v g="$g" '$1 == g {print $2}' <<<"$uuids")"
    [ -n "$u" ] || { GPU_WHY="no uuid for GPU ${g} in nvidia-smi output"; return 1; }
    IDX[$u]="$g"
  done
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null)" \
    || { GPU_WHY="nvidia-smi --query-compute-apps failed"; return 1; }
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    IFS=',' read -r uuid pid pname mem <<<"$line"
    uuid="${uuid// /}"; pid="${pid// /}"; pname="${pname# }"; mem="${mem# }"
    [[ "$pid" =~ ^[0-9]+$ && -n "$uuid" ]] || { GPU_WHY="unparsable compute-apps line '${line}'"; return 1; }
    [ -n "${IDX[$uuid]:-}" ] || continue                                   # another card
    busy="${busy}GPU ${IDX[$uuid]}: pid ${pid} (${pname}, ${mem}, cwd $(readlink "/proc/${pid}/cwd" 2>/dev/null || echo '?')); "
  done <<<"$apps"
  [ -z "$busy" ] || { GPU_WHY="foreign compute process(es): ${busy}"; return 1; }
  for g in 0 1; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -n1 | tr -d ' ')" || true
    [[ "$free" =~ ^[0-9]+$ ]] || { GPU_WHY="memory.free of GPU ${g} unparsable ('${free}')"; return 1; }
    [ "$free" -ge "$MIN_GPU_FREE_MIB" ] || { GPU_WHY="GPU ${g} has ${free} MiB free < ${MIN_GPU_FREE_MIB} MiB"; return 1; }
  done
}

run_launcher() {   # run_launcher <MODE> <capture file>; returns the launcher's rc
  local rc
  set +e
  MODE="$1" EXPECT_HEAD="$TOKEN_HEAD" DRY_RUN=0 bash "$LAUNCHER" 2>&1 | tee "$2"
  rc="${PIPESTATUS[0]}"
  set -e
  return "$rc"
}

# --- the loop ---------------------------------------------------------------------------- #
write_state WAITING_TOKEN "starting"
POLLS=0; LAST_BEAT="$(date +%s)"
while :; do
  POLLS=$((POLLS + 1))
  if [ "$MAX_POLLS" -gt 0 ] && [ "$POLLS" -gt "$MAX_POLLS" ]; then
    echo "poll budget exhausted (${MAX_POLLS}) in phase ${PHASE}"; exit 3
  fi
  if [ $(( $(date +%s) - LAST_BEAT )) -ge "$HEARTBEAT_S" ]; then
    echo "[$(date -Iseconds)] heartbeat: phase ${PHASE} (${REASON}), clean polls ${CLEAN}/${STABLE_POLLS}"
    LAST_BEAT="$(date +%s)"
  fi
  if ! token_ok; then
    CLEAN=0; TOKEN_HEAD=""; write_state WAITING_TOKEN "$TOKEN_WHY"; sleep "$POLL_S"; continue
  fi
  if ! gpu_clean; then   # write_state prints (and records) the reason once per change
    CLEAN=0; write_state WAITING_GPU "GPUs not free: ${GPU_WHY}"; sleep "$POLL_S"; continue
  fi
  CLEAN=$((CLEAN + 1))
  write_state WAITING_GPU "clean poll ${CLEAN}/${STABLE_POLLS}"
  [ "$CLEAN" -ge "$STABLE_POLLS" ] || { sleep "$POLL_S"; continue; }
  # fire -- the token is re-checked immediately before EACH launcher call
  if ! token_ok; then CLEAN=0; TOKEN_HEAD=""; write_state WAITING_TOKEN "$TOKEN_WHY"; sleep "$POLL_S"; continue; fi
  write_state SMOKE_RUNNING "MODE=SMOKE EXPECT_HEAD=${TOKEN_HEAD}"
  CAP="$EXPDIR/depth_canon_${TS}_autolaunch_smoke_$((RETRIES + 1)).out"
  rc=0; run_launcher SMOKE "$CAP" || rc=$?
  SMOKE_RC="$rc"
  if [ "$rc" -ne 0 ]; then
    if grep -qF -- "$GPU_GATE_STR" "$CAP" || grep -qF -- "$GPU_OCCUPIED_STR" "$CAP"; then
      RETRIES=$((RETRIES + 1)); CLEAN=0
      if [ "$RETRIES" -gt "$MAX_GPU_RETRIES" ]; then
        write_state FAILED "stage=SMOKE rc=${rc}: the launcher's GPU gate refused ${RETRIES} times (> ${MAX_GPU_RETRIES} retries)"
        exit 1
      fi
      write_state WAITING_GPU "SMOKE refused by the launcher's GPU gate (rc ${rc}); retry ${RETRIES}/${MAX_GPU_RETRIES} after re-stabilizing"
      sleep "$POLL_S"; continue
    fi
    write_state FAILED "stage=SMOKE rc=${rc} (see ${CAP}); FULL not attempted"
    exit 1
  fi
  if ! token_ok; then CLEAN=0; TOKEN_HEAD=""; write_state WAITING_TOKEN "$TOKEN_WHY (after a passing SMOKE)"; sleep "$POLL_S"; continue; fi
  write_state FULL_RUNNING "MODE=FULL EXPECT_HEAD=${TOKEN_HEAD}"
  rc=0; run_launcher FULL /dev/null || rc=$?       # FULL's output reaches this log through the tee
  FULL_RC="$rc"
  if [ "$rc" -eq 0 ]; then write_state DONE "SMOKE and FULL both exited 0"; exit 0; fi
  write_state FAILED "stage=FULL rc=${rc}"
  exit 1
done
