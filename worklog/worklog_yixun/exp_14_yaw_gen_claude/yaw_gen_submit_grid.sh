#!/usr/bin/env bash
# ============================================================================
# yaw_gen_submit_grid.sh — submit one WAVE of the exp_14 grid (plan §5.5/§6).
#
# The grid is 106 cells at one campaign pin. This script is the only thing that
# decides WHICH of them to submit; yaw_gen_screen_submit.sh still owns each
# individual submission (store lock -> pinned worktree -> sbatch --hold -> lease
# -> release), so nothing about that transaction is duplicated here.
#
#   WAVE=vctl   the six validity controls — plan §6 runs them FIRST and alone:
#               C4L@90 exercises worktree provisioning (the VAE.ckpt leaf-asset
#               regression that killed exp_11's R3), the fa path, fixed rotation
#               and metrics landing, and the six together are gates G1/G2.
#   WAVE=zref   the 50 theta=0 reference cells (their sigma feeds G1/G2, and the
#               VANL block completes the model_comparison.md vanilla row).
#   WAVE=rgen   the 50 headline random-yaw cells.
#   WAVE=all    everything, for a resume sweep after the waves have run.
#
# Wave ORDER is the Planner's decision, not this script's: it submits exactly the
# wave it is given.
#
# DEDUP IS VALIDATE-BEFORE-SKIP (review B6). Before submitting anything, every
# cell of the wave is classified by exp14_validate_cell.py:
#   VALID    -> skipped (already measured at this pin)
#   MISSING  -> submitted
#   INVALID  -> the whole wave HALTS. An artifact that exists but does not
#               validate is a triage question, never a thing to silently skip
#               (it would be read as evidence) or overwrite (it would destroy
#               the evidence of what went wrong).
# A cell whose job is already queued or running is skipped with a note.
#
# Usage:
#   DRYRUN=1 bash yaw_gen_submit_grid.sh WAVE=vctl            # prints, submits nothing
#   bash yaw_gen_submit_grid.sh WAVE=vctl EXCLUDE=neu303,neu332
#   bash yaw_gen_submit_grid.sh WAVE=rgen MAX_INFLIGHT=16
# ============================================================================
set -uo pipefail
# --- FIRST EXECUTABLE STATEMENTS: sanitize the world, then gate it -----------
# THREAT MODEL (Planner, 2026-08-11): this kit defends against ACCIDENTS and
# stray environment state — a leftover export, a seam left set by a previous
# run, a test that forgets isolation. It is not hardened against an adversary
# with arbitrary control of the calling shell; the preamble below is the cheap
# part of that boundary, not a claim to have crossed it.
#
# Order is the whole point (review W1/V1). Everything after this — the gate, the
# reader, the store-lock re-exec — used to run under whatever PATH and whatever
# exported shell functions the caller supplied.
#
# POSIXLY_CORRECT is an ASSIGNMENT, so it involves no command lookup and cannot
# be shadowed; in bash it turns on POSIX mode, where SPECIAL BUILTINS (set,
# unset, export, eval, ...) are resolved BEFORE functions. The `unset -f` that
# follows is therefore guaranteed to be the real builtin even if an exported
# unset() function was inherited. POSIX mode is dropped again immediately after
# (verified: "${!YAW_GEN_@}" expands identically in both modes).
POSIXLY_CORRECT=1
unset -f unset set export builtin command exec trap readonly eval declare \
         typeset local source env 2>/dev/null || true
set +o posix
unset POSIXLY_CORRECT
PATH=/usr/bin:/bin:/usr/local/bin; export PATH
unset -f sbatch scontrol scancel squeue sync git grep sed awk head tail tr \
         cat printf id date hostname readlink flock mktemp sha256sum cut wc ls \
         rm mv cp mkdir sleep bash python3 2>/dev/null || true

# ENTRY DOCTRINE, part 1: enumerate the environment NATIVELY. "${!YAW_GEN_@}"
# expands to the NAMES of set shell variables with that prefix — bash's own
# symbol table, so there is no command here to substitute or shadow.
env_names_to_gate() { printf '%s\n' ${!YAW_GEN_@} ${!FA_ORBIT_@} 2>/dev/null; }
env_allowlist_gate() {   # $1 = space-separated allowed names
  local name a allowed
  for name in $(env_names_to_gate); do
    [ -n "$name" ] || continue
    allowed=0
    for a in $1; do [ "$name" = "$a" ] && allowed=1 && break; done
    [ "$allowed" = "1" ] || {
      echo "environment variable ${name} is not on this mode's allowlist: only the documented DATA seams are honoured (nothing here may name a command)" >&2
      exit 2
    }
  done
}

DRYRUN="${DRYRUN:-0}"
TEST_MODE="${YAW_GEN_TEST_MODE:-0}"
[ "$DRYRUN" = "1" ] && TEST_MODE=1      # a dry run submits nothing by construction
ALLOW_LIVE="YAW_GEN_TEST_MODE"
#   YAW_GEN_TEST_MODE      "1" selects simulation (0/unset = live)
ALLOW_TEST="YAW_GEN_TEST_MODE YAW_GEN_TEST_RECORD YAW_GEN_TEST_JOBID \
YAW_GEN_TEST_SUBMIT_RC YAW_GEN_SQUEUE_FIXTURE YAW_GEN_SQUEUE_FAILS \
YAW_GEN_SYNC_FAILS YAW_GEN_PIN_FILE YAW_GEN_COMMAND_LOG YAW_GEN_WT_DIR \
YAW_GEN_INTENT_DIR YAW_GEN_MAIN_REPO"
#   YAW_GEN_TEST_RECORD    path of a text file this script APPENDS its argv to
#   YAW_GEN_TEST_JOBID     the job id a simulated submission reports (data)
#   YAW_GEN_TEST_SUBMIT_RC the rc a simulated submission returns (data)
#   YAW_GEN_SQUEUE_FIXTURE path of a text file holding "<jobid> <name>" lines —
#                          READ as data, never executed
#   YAW_GEN_SQUEUE_FAILS   "1" makes the simulated queue query fail
#   YAW_GEN_SYNC_FAILS     "1" makes the simulated flush fail
#   YAW_GEN_MAIN_REPO      test-only root this script reads/writes under
#   YAW_GEN_PIN_FILE       path of the campaign-pin file to read
#   YAW_GEN_COMMAND_LOG    path of the command log to append to
#   YAW_GEN_WT_DIR         directory whose .leases/ prove an in-flight job
#   YAW_GEN_INTENT_DIR     not read here: allowed because a test environment that
#                          drives the single-cell submitter carries it (data path)
if [ "$TEST_MODE" = "1" ]; then env_allowlist_gate "$ALLOW_TEST"
else                            env_allowlist_gate "$ALLOW_LIVE"; fi

MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
# TEST MODE may relocate the whole tree this script reads and writes (review V3).
# It is a DATA path like every other allowlisted seam, and it exists so a test
# harness is isolated BY DEFAULT rather than by remembering to be: a case that
# forgets isolation lands in the temp root, and one that tries it in live mode is
# refused by the allowlist above.
if [ "$TEST_MODE" = "1" ] && [ -n "${YAW_GEN_MAIN_REPO:-}" ]; then
  MAIN_REPO="$YAW_GEN_MAIN_REPO"
fi
EXPDIR="$MAIN_REPO/worklog/worklog_yixun/exp_14_yaw_gen_claude"
SUBMIT="$EXPDIR/yaw_gen_screen_submit.sh"
VALIDATE="$EXPDIR/exp14_validate_cell.py"
PIN_FILE="$EXPDIR/yaw_gen_campaign_pin"
COMMAND_LOG="$EXPDIR/yaw_gen_command.md"
# (both may be redirected in TEST MODE only — see the TEST_MODE block below)
PY=/n/fs/gatrdp/envs/flac/bin/python     # the campaign interpreter, not overridable
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_REPO/outputs_FLAC}"
# NO ENVIRONMENT VARIABLE CARRIES AN EXECUTABLE, in any mode. Every seam that
# used to name a command (the submitter, squeue, sync, even the interpreter) is
# gone: an env var that is exec'd is a way to run something else, and no amount
# of "is it a mock?" reasoning makes that safe. Test mode SIMULATES those calls
# in-process; live mode runs binaries resolved to absolute paths at entry (Z2).
# What test mode may still be given is DATA: a queue FIXTURE FILE to read, flags
# that make a simulated call fail, and paths to write records to.

MAX_INFLIGHT="${MAX_INFLIGHT:-16}"
POLL_SECONDS="${POLL_SECONDS:-120}"

WAVE=""; EXCLUDE=""; PIN_SHA=""
is_num()      { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
is_hex40()    {
  case "${1:-}" in
    *[!0-9a-f]*) return 1 ;;
    ????????????????????????????????????????) return 0 ;;
    *) return 1 ;;
  esac
}
is_nodelist() { case "${1:-}" in ''|*[!a-z0-9,]*) return 1 ;; esac; }
in_set()      { local v="$1"; shift; local t; for t in "$@"; do [ "$v" = "$t" ] && return 0; done; return 1; }
reject()      { echo "$1" >&2; exit 2; }

# Arguments are DATA, never shell: every key is whitelisted and every value is
# shape-checked BEFORE assignment, exactly as in yaw_gen_screen_submit.sh.
for kv in "$@"; do
  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    WAVE)     in_set "$val" vctl zref rgen all || reject "WAVE='${val}' is not vctl|zref|rgen|all" ;;
    EXCLUDE)  is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
    PIN_SHA)  is_hex40 "$val" || reject "PIN_SHA='${val}' is not 40 hex characters" ;;
    *)        reject "unknown argument '${kv}' (expected WAVE=/EXCLUDE=/PIN_SHA=)" ;;
  esac
  printf -v "$key" '%s' "$val"
done
[ -n "$WAVE" ] || { echo "usage: [DRYRUN=1] bash $0 WAVE={vctl|zref|rgen|all} [EXCLUDE=n1,n2] [PIN_SHA=<40hex>]" >&2; exit 2; }
is_num "$MAX_INFLIGHT" && [ "$MAX_INFLIGHT" -ge 1 ] && [ "$MAX_INFLIGHT" -le 16 ] \
  || reject "MAX_INFLIGHT='${MAX_INFLIGHT}' must be 1..16 (plan §7 caps this campaign at 16 slots)"
[ -f "$VALIDATE" ] || reject "validator missing: ${VALIDATE}"
[ -f "$SUBMIT" ]   || reject "single-cell submitter missing: ${SUBMIT}"
# An overridable SUBMITTER is exactly the hole a "is this a mock?" test cannot
# close: a wrapper, a copy or a hard link that calls the real one passes every
# such test. There is no override any more — in test mode the submission is
# SIMULATED in-process (submit_cell), and in live mode it is always this
# campaign's own submitter.
if [ "$TEST_MODE" = "1" ]; then
  # Only in test mode may the campaign's own state be redirected. A test that
  # writes the live command log and the live pin file is not a test of the wave,
  # it is an edit of the campaign — and a RED run of this suite once submitted
  # four real jobs that way.
  [ -n "${YAW_GEN_PIN_FILE:-}" ] && PIN_FILE="$YAW_GEN_PIN_FILE"
  [ -n "${YAW_GEN_COMMAND_LOG:-}" ] && COMMAND_LOG="$YAW_GEN_COMMAND_LOG"
else
  # LIVE: resolve the binaries ONCE, absolutely, on the PATH sanitized at entry
  # and with any exported function of the same name already dropped (Z2).
  for _n in squeue sync; do
    _p="$(command -v "$_n" 2>/dev/null)" || _p=""
    case "$_p" in
      /*) ;;
      *) reject "cannot resolve an absolute path for '${_n}' on a sanitized PATH" ;;
    esac
    [ -x "$_p" ] || reject "resolved '${_p}' for ${_n} is not executable"
    printf -v "BIN_${_n}" '%s' "$_p"
  done
  echo "resolved: squeue=${BIN_squeue} sync=${BIN_sync}" >&2
fi

# --- the campaign pin --------------------------------------------------------
# A live wave REQUIRES the pin: 106 cells are comparable only if they ran at one
# commit, and without the file each submission would pin at whatever HEAD happens
# to be. DRYRUN does not need it (it submits nothing).
# PIN_SHA is an ASSERTION, never a substitute: passing it used to satisfy the
# "one commit" rule with no pin file at all, which is exactly the rail the round
# claimed to have (review B3).
CAMPAIGN_PIN=""
if [ -f "$PIN_FILE" ]; then
  CAMPAIGN_PIN="$(head -1 "$PIN_FILE" | tr -d '[:space:]')"
  is_hex40 "$CAMPAIGN_PIN" || reject "campaign pin file ${PIN_FILE} does not hold a 40-hex sha"
fi
if [ "$DRYRUN" != "1" ]; then
  if [ -z "$CAMPAIGN_PIN" ]; then
    echo "refusing to submit a wave with no campaign pin FILE (${PIN_FILE})." >&2
    echo "  106 cells are comparable only if they ran at ONE commit, and PIN_SHA on a" >&2
    echo "  command line is not that commitment — set it deliberately, once:" >&2
    echo "  printf '%s\\n' \$(git -C ${MAIN_REPO} rev-parse HEAD) > ${PIN_FILE}" >&2
    exit 2
  fi
  git -C "$MAIN_REPO" rev-parse --verify --quiet "${CAMPAIGN_PIN}^{commit}" >/dev/null \
    || reject "campaign pin ${CAMPAIGN_PIN} is not a commit in this repository"
fi
if [ -n "$PIN_SHA" ] && [ -n "$CAMPAIGN_PIN" ] && [ "$PIN_SHA" != "$CAMPAIGN_PIN" ]; then
  reject "PIN_SHA=${PIN_SHA} disagrees with the campaign pin ${CAMPAIGN_PIN} (${PIN_FILE})"
fi
[ -n "$CAMPAIGN_PIN" ] && PIN_SHA="$CAMPAIGN_PIN"

# --- the grid ----------------------------------------------------------------
# Enumerated by the VALIDATOR, which is also what the collector and the guard
# suite read: one definition of "the registered grid", not three.
mapfile -t CELLS < <("$PY" "$VALIDATE" grid --wave "$WAVE") \
  || reject "could not enumerate the ${WAVE} wave"
[ "${#CELLS[@]}" -gt 0 ] || reject "the ${WAVE} wave is empty"
# The banner goes to STDERR so a DRYRUN's stdout is exactly one line per cell
# (the guard suite diffs it against the registered grid).
echo "=== exp_14 ${WAVE} wave: ${#CELLS[@]} registered cells | pin ${PIN_SHA:-<none: DRYRUN>} ===" >&2

test_record() { [ -n "${YAW_GEN_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_GEN_TEST_RECORD"; return 0; }

sync_file() {   # flush one file's bytes to disk; NONZERO when it cannot be done.
                # A record that is not durable is not a record — that is the whole
                # point of writing it before the launch — so an unconditional
                # success here made the fatal check unreachable.
  local f="$1"
  if [ "$TEST_MODE" = "1" ]; then
    test_record "sync ${f}"
    [ "${YAW_GEN_SYNC_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_sync" "$f" 2>/dev/null && return 0
  "$BIN_sync" 2>/dev/null && return 0         # coreutils without per-file sync
  return 1
}

queue_lines() {  # "<jobid> <name>" per line; NONZERO when the queue cannot be read
  if [ "$TEST_MODE" = "1" ]; then
    [ "${YAW_GEN_SQUEUE_FAILS:-0}" = "1" ] && return 1
    # A FIXTURE FILE, read as data. There is no command here to substitute.
    [ -n "${YAW_GEN_SQUEUE_FIXTURE:-}" ] || return 0
    cat "$YAW_GEN_SQUEUE_FIXTURE" 2>/dev/null || return 1
    return 0
  fi
  "$BIN_squeue" -h -u "$(id -un)" -o "%i %j" 2>/dev/null
}

submit_argv() {  # <arm> <cell> <step> <seed> <k> <rot>  -> the cell's submit argv
  local arm="$1" cell="$2" step="$3" seed="$4" k="$5" rot="$6"
  local argv="ARM=${arm} CELL=${cell} STEP=${step} SEED=${seed} K=${k}"
  [ "$rot" = "-" ] || argv="${argv} ROTATE_DEG=${rot}"
  [ -z "$EXCLUDE" ] || argv="${argv} EXCLUDE=${EXCLUDE}"
  [ -z "$PIN_SHA" ] || argv="${argv} PIN_SHA=${PIN_SHA}"
  printf '%s' "$argv"
}

# --- DRYRUN: print the exact submission line for every cell, submit nothing ---
if [ "$DRYRUN" = "1" ]; then
  for line in "${CELLS[@]}"; do
    # shellcheck disable=SC2086
    set -- $line
    echo "bash ${SUBMIT} $(submit_argv "$1" "$2" "$3" "$4" "$5" "$6")"
  done
  echo "DRY RUN: ${#CELLS[@]} cells enumerated, nothing submitted, nothing classified" >&2
  exit 0
fi

# --- classify the whole wave BEFORE submitting anything ----------------------
# One python process for the wave (importing per cell would cost a process each);
# the classification is a snapshot taken at wave start, which is what dedup needs
# — it is about artifacts that already existed, not ones this wave will produce.
mapfile -t STATUS < <("$PY" "$VALIDATE" classify --wave "$WAVE" \
                        --output-root "$OUTPUT_ROOT" --pin "$PIN_SHA") \
  || reject "could not classify the ${WAVE} wave"
[ "${#STATUS[@]}" -eq "${#CELLS[@]}" ] \
  || reject "classifier returned ${#STATUS[@]} rows for ${#CELLS[@]} cells - refusing to guess"

INVALID=()
for row in "${STATUS[@]}"; do
  case "$row" in *" INVALID "*) INVALID+=("$row") ;; esac
done
if [ "${#INVALID[@]}" -gt 0 ]; then
  echo "HALT: ${#INVALID[@]} cell(s) of the ${WAVE} wave have artifacts that do NOT validate." >&2
  echo "Nothing was submitted. An artifact that exists but fails validation is a" >&2
  echo "triage question: it must not be skipped (it would be read as evidence) and" >&2
  echo "must not be overwritten (that destroys the evidence of what went wrong)." >&2
  for row in "${INVALID[@]}"; do echo "  ${row}" >&2; done
  echo "Triage: re-run the named checks, then either fix the cell's inputs and" >&2
  echo "  move the artifact aside deliberately, or re-pin the campaign." >&2
  exit 3
fi

# --- in-flight jobs: job ID *and* name, and the LEASE that proves the pairing --
# A name alone is a claim; the lease is the evidence. exp_11's whole worktree
# discipline rests on "a running job holds a lease on the tree it reads code
# from", so a queued job whose lease is missing is either not ours or running
# unprotected — and skipping the cell would then mean waiting for a job that may
# never produce it (review B5).
#
# A squeue that FAILS is not evidence that nothing is running (exp_11's lease
# reaper learned this the expensive way): refuse rather than flood the queue.
# The pinned worktree whose .leases/ prove a queued job is really ours. In a LIVE
# wave this is DERIVED from the campaign pin and nothing can redirect it: an
# overridable lease directory is an overridable invariant — any directory holding
# .leases/<jid> would let a wave skip a job that holds no lease in the campaign's
# own worktree, which is the fail-open case the re-verify caught.
#
# The guard suite still needs to point it elsewhere, so the seam survives behind
# TWO locks: YAW_GEN_TEST_MODE=1 must be set explicitly, and test mode itself
# refuses to run unless the submitter is a mock (checked above). A live wave can
# therefore never reach this branch.
WT_DIR="${MAIN_REPO}/.measure_worktrees/${PIN_SHA}"
# In test mode only — a LIVE wave carrying this variable never gets here at all:
# the entry allowlist refuses it outright, which is stronger than ignoring it.
if [ -n "${YAW_GEN_WT_DIR:-}" ] && [ "$TEST_MODE" = "1" ]; then
  WT_DIR="$YAW_GEN_WT_DIR"
  echo "TEST MODE: lease directory overridden to ${WT_DIR}" >&2
fi
QUEUED="$(queue_lines)" \
  || reject "squeue failed - refusing to submit without knowing what is already queued"
inflight_count() { printf '%s\n' "$QUEUED" | awk '{print $2}' | grep -c '^exp14-' ; }
refresh_queue() {
  QUEUED="$(queue_lines)" || return 1
  return 0
}
queued_jobid() {   # <job name> -> the job id Slurm reports for it, or ""
  printf '%s\n' "$QUEUED" | awk -v n="$1" '$2 == n {print $1; exit}'
}

SUBMITTED=0; SKIPPED=0; RUNNING=0
for i in "${!CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- ${CELLS[$i]}
  ARM="$1"; CELL="$2"; STEP="$3"; SEED="$4"; K="$5"; ROT="$6"
  row="${STATUS[$i]}"
  case "$row" in
    "${ARM} ${CELL} ${STEP} ${SEED} ${K} ${ROT} "*) ;;
    *) reject "classifier row '${row}' does not describe cell '${CELLS[$i]}' - refusing to guess" ;;
  esac
  # Identical rendering to yaw_gen_screen_submit.sh (both pinned to
  # exp14_validate_cell.job_name by a guard case): the token is what keeps the
  # two C4L vctl angles from reading as one in-flight cell (review B2).
  JOB_TOKEN=""
  case "$CELL" in
    rgen) JOB_TOKEN="-rotrand${SEED}" ;;
    vctl) JOB_TOKEN="-rot${ROT%.0}" ;;
  esac
  JOB_NAME="exp14-screen-${ARM}-${CELL}${JOB_TOKEN}-${STEP}-s${SEED}-K${K}"
  case "$row" in
    *" VALID "*)
      echo "SKIP  ${JOB_NAME}: already measured and valid at this pin"; SKIPPED=$((SKIPPED + 1)); continue ;;
  esac
  JID_INFLIGHT="$(queued_jobid "$JOB_NAME")"
  if [ -n "$JID_INFLIGHT" ]; then
    if [ -f "${WT_DIR}/.leases/${JID_INFLIGHT}" ] \
       && grep -q "^jobid ${JID_INFLIGHT}$" "${WT_DIR}/.leases/${JID_INFLIGHT}" 2>/dev/null; then
      echo "SKIP  ${JOB_NAME}: job ${JID_INFLIGHT} is in flight and holds its lease"
      RUNNING=$((RUNNING + 1)); continue
    fi
    echo "HALT: ${JOB_NAME} is queued as job ${JID_INFLIGHT} but holds NO lease under" >&2
    echo "  ${WT_DIR}/.leases/${JID_INFLIGHT}" >&2
    echo "  Either it is reading code from an unprotected tree (a sweep may delete it" >&2
    echo "  underneath the job), or the name belongs to a job this campaign did not" >&2
    echo "  submit. Skipping would mean waiting for a cell that may never land, and" >&2
    echo "  submitting would duplicate it: neither is safe, so the wave stops." >&2
    echo "  Triage: scontrol show job ${JID_INFLIGHT}; ls ${WT_DIR}/.leases/" >&2
    exit 5
  fi
  # --- concurrency: never more than MAX_INFLIGHT of OUR jobs in the queue -----
  while [ "$(inflight_count)" -ge "$MAX_INFLIGHT" ]; do
    echo "  waiting: $(inflight_count) exp14- jobs in flight (cap ${MAX_INFLIGHT}); sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    refresh_queue || reject "squeue failed while waiting - stopping the wave"
  done
  ARGV="$(submit_argv "$ARM" "$CELL" "$STEP" "$SEED" "$K" "$ROT")"
  echo "SUBMIT ${JOB_NAME}"
  # In test mode the submission is simulated HERE — no process is started, so a
  # test cannot reach sbatch through any executable at all. The simulation still
  # observes what matters: whether the command log carried this cell's LAUNCHING
  # line at the moment the submission would have happened.
  submit_cell() {
    if [ "$TEST_MODE" = "1" ]; then
      test_record "submit $*"
      if grep -q 'LAUNCHING' "$COMMAND_LOG" 2>/dev/null; then
        test_record "  launching-line-present"
      else
        test_record "  launching-line-MISSING"
      fi
      [ -n "${YAW_GEN_TEST_SUBMIT_RC:-}" ] && return "$YAW_GEN_TEST_SUBMIT_RC"
      echo "submitted HELD as ${YAW_GEN_TEST_JOBID:-5550001}"
      return 0
    fi
    bash "$SUBMIT" "$@"
  }
  # The command log is written BEFORE the submission and flushed to disk, and a
  # failure to write it is FATAL (review B7). Recording afterwards means a crash
  # between sbatch and the append produces a job nobody can trace back to a
  # command; recording first can only produce the harmless converse — a launch
  # line for a job that never started, which the "rc" line that follows resolves.
  log_command() {   # <state> <jid> <rc>
    printf -- '- `%s` **%s** %s jid `%s` rc %s — `bash %s %s`\n' \
      "$(date -Is)" "$JOB_NAME" "$1" "${2:-<none>}" "${3:-<pending>}" "$SUBMIT" "$ARGV" \
      >> "$COMMAND_LOG" && sync_file "$COMMAND_LOG"
  }
  log_command LAUNCHING "" "" \
    || { echo "HALT: cannot durably record the command in ${COMMAND_LOG}; refusing to" >&2
         echo "  submit a job that would have no launch record." >&2; exit 6; }
  # shellcheck disable=SC2086
  OUT="$(submit_cell $ARGV 2>&1)"; RC=$?
  echo "$OUT" | sed 's/^/       | /'
  JID="$(printf '%s\n' "$OUT" | sed -n 's/^submitted HELD as \([0-9][0-9]*\)$/\1/p' | tail -1)"
  log_command SUBMITTED "$JID" "$RC" \
    || echo "WARNING: could not append the outcome to ${COMMAND_LOG}" >&2
  if [ "$RC" -ne 0 ]; then
    echo "HALT: submission of ${JOB_NAME} failed (rc=${RC}); ${SUBMITTED} cell(s) already submitted." >&2
    echo "  The wave stops here rather than continuing past an unexplained failure." >&2
    exit 4
  fi
  SUBMITTED=$((SUBMITTED + 1))
  refresh_queue || reject "squeue failed after submitting ${JOB_NAME} - stopping the wave"
done

echo "=== ${WAVE} wave done: ${SUBMITTED} submitted, ${SKIPPED} already valid, ${RUNNING} in flight ==="
exit 0
