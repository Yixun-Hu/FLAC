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
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$MAIN_REPO/worklog/worklog_yixun/exp_14_yaw_gen_claude"
SUBMIT="$EXPDIR/yaw_gen_screen_submit.sh"
VALIDATE="$EXPDIR/exp14_validate_cell.py"
PIN_FILE="$EXPDIR/yaw_gen_campaign_pin"
COMMAND_LOG="$EXPDIR/yaw_gen_command.md"
# (both may be redirected in TEST MODE only — see the TEST_MODE block below)
PY="${YAW_GEN_PY:-/n/fs/gatrdp/envs/flac/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_REPO/outputs_FLAC}"
# Test seams, same shape as the single-cell submitter's: a real run uses the
# real binaries. They exist so the guard suite can exercise the SEQUENCE without
# submitting anything.
SQUEUE="${YAW_GEN_SQUEUE:-squeue}"
SYNC_CMD="${YAW_GEN_SYNC:-sync}"
SUBMIT_CMD="${YAW_GEN_SUBMIT:-$SUBMIT}"

DRYRUN="${DRYRUN:-0}"
# TEST MODE exists so the guard suite can drive the wave's DECISIONS with mocked
# Slurm and a fake lease directory. It is inert unless the submitter itself is a
# mock: a run that could really submit may not also relax an invariant.
TEST_MODE="${YAW_GEN_TEST_MODE:-0}"
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
if [ "$TEST_MODE" = "1" ]; then
  [ -n "${YAW_GEN_SUBMIT:-}" ] \
    && [ "$(readlink -f "${YAW_GEN_SUBMIT}" 2>/dev/null)" != "$(readlink -f "$SUBMIT")" ] \
    || reject "YAW_GEN_TEST_MODE=1 may only run against a mocked submitter (set YAW_GEN_SUBMIT=...); refusing to run a wave that could really submit with test seams honoured"
  # ...and only THEN may the campaign's own state be redirected. A test that
  # writes the live command log and the live pin file is not a test of the wave,
  # it is an edit of the campaign — and a RED run of this suite once submitted
  # four real jobs that way.
  [ -n "${YAW_GEN_PIN_FILE:-}" ] && PIN_FILE="$YAW_GEN_PIN_FILE"
  [ -n "${YAW_GEN_COMMAND_LOG:-}" ] && COMMAND_LOG="$YAW_GEN_COMMAND_LOG"
elif [ -n "${YAW_GEN_PIN_FILE:-}${YAW_GEN_COMMAND_LOG:-}${YAW_GEN_WT_DIR:-}" ]; then
  echo "NOTE: YAW_GEN_PIN_FILE / YAW_GEN_COMMAND_LOG / YAW_GEN_WT_DIR are IGNORED" >&2
  echo "      without YAW_GEN_TEST_MODE=1 (which itself requires a mocked submitter)" >&2
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

sync_file() {   # flush one file's bytes to disk; NONZERO when it cannot be done.
                # A record that is not durable is not a record — that is the whole
                # point of writing it before the launch — so an unconditional
                # success here made the fatal check unreachable.
  local f="$1"
  command -v "$SYNC_CMD" >/dev/null 2>&1 || return 1
  "$SYNC_CMD" "$f" 2>/dev/null && return 0
  "$SYNC_CMD" 2>/dev/null && return 0        # coreutils without per-file sync
  return 1
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
if [ -n "${YAW_GEN_WT_DIR:-}" ]; then
  if [ "$TEST_MODE" = "1" ]; then
    WT_DIR="$YAW_GEN_WT_DIR"
    echo "TEST MODE: lease directory overridden to ${WT_DIR}" >&2
  else
    echo "NOTE: YAW_GEN_WT_DIR is set but IGNORED — leases are read from the pinned" >&2
    echo "      worktree ${WT_DIR} (override needs YAW_GEN_TEST_MODE=1 AND a mocked" >&2
    echo "      submitter; a live wave may not relax the lease invariant)" >&2
  fi
fi
QUEUED="$("$SQUEUE" -h -u "$(id -un)" -o "%i %j" 2>/dev/null)" \
  || reject "squeue failed - refusing to submit without knowing what is already queued"
inflight_count() { printf '%s\n' "$QUEUED" | awk '{print $2}' | grep -c '^exp14-' ; }
refresh_queue() {
  QUEUED="$("$SQUEUE" -h -u "$(id -un)" -o "%i %j" 2>/dev/null)" || return 1
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
  OUT="$(bash "$SUBMIT_CMD" $ARGV 2>&1)"; RC=$?
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
