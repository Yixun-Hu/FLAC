#!/usr/bin/env bash
# ============================================================================
# yaw_aug_submit_grid.sh — submit one WAVE of the exp_15 eval grid (plan §4.1).
#
# The grid is 42 cells at one campaign pin. This script is the only thing that
# decides WHICH of them to submit; yaw_aug_screen_submit.sh still owns each
# individual submission (store lock -> pinned worktree -> sbatch --hold -> lease
# -> release), so nothing about that transaction is duplicated here.
#
#   WAVE=vctl   the two validity controls — plan §7-8 runs them FIRST and alone:
#               VANL@90 exercises worktree provisioning (the weights/FLAC/VAE.ckpt
#               leaf-asset regression that killed exp_11's R3), fixed rotation and
#               metrics landing, and IS gate G1's positive control. YAWAUG@90 is
#               descriptive only.
#   WAVE=tbl    the 20 theta=0 headline cells (H1's clean contrast; their sigma
#               also feeds G1's threshold).
#   WAVE=rrob   the 20 random-yaw robustness cells (H2/H3).
#   WAVE=all    everything, for a resume sweep after the waves have run.
#
# Wave ORDER is the Planner's decision, not this script's: it submits exactly the
# wave it is given.
#
# DEDUP IS VALIDATE-BEFORE-SKIP (exp_14 review B6). Before submitting anything,
# every cell of the wave is classified by exp15_validate_cell.py:
#   VALID    -> skipped (already measured at this pin)
#   MISSING  -> submitted
#   INVALID  -> the whole wave HALTS. An artifact that exists but does not
#               validate is a triage question, never a thing to silently skip
#               (it would be read as evidence) or overwrite (it would destroy
#               the evidence of what went wrong).
# A cell whose job is already queued or running is skipped with a note.
#
# ⚠️ Classification is ALSO fail-closed on checkpoint admission: while the
# training chain has not recorded arms.YAWAUG.final_ckpt_sha256, the classifier
# refuses the whole wave by name rather than reporting YAWAUG cells as MISSING.
# "not measured yet" and "cannot be judged yet" are different states, and only
# the second one must stop a wave. DRYRUN still enumerates the grid, because it
# submits nothing and classifies nothing.
#
# Usage:
#   DRYRUN=1 bash yaw_aug_submit_grid.sh WAVE=all              # prints, submits nothing
#   bash yaw_aug_submit_grid.sh WAVE=vctl EXCLUDE=neu303,neu332
#   bash yaw_aug_submit_grid.sh WAVE=rrob MAX_INFLIGHT=16
# ============================================================================
# U1: the assignment and the special-builtin sweep come BEFORE `set`, so a
# shadowed set() cannot intercept errexit/nounset/pipefail either. An assignment
# involves no command lookup; POSIX mode then makes special builtins outrank
# functions for the sweep that follows.
POSIXLY_CORRECT=1
unset -f unset set export builtin command exec trap readonly eval declare \
         typeset local source env 2>/dev/null || true
set +o posix
unset POSIXLY_CORRECT
set -uo pipefail
# --- FIRST EXECUTABLE STATEMENTS: sanitize the world, then gate it -----------
# THREAT MODEL (inherited from exp_14): this kit defends against ACCIDENTS and
# stray environment state. It is not hardened against an adversary with arbitrary
# control of the calling shell; the preamble below is the cheap part of that
# boundary, not a claim to have crossed it.
PATH=/usr/bin:/bin:/usr/local/bin; export PATH
unset -f sbatch scontrol scancel squeue sync git grep sed awk head tail tr \
         cat printf id date hostname readlink flock mktemp sha256sum cut wc ls \
         rm mv cp mkdir sleep bash python3 2>/dev/null || true

# ENTRY DOCTRINE, part 1: enumerate the environment NATIVELY. The neighbouring
# namespaces (exp_15's TRAINING kit YAW_AUG_*, exp_14's YAW_GEN_*) are enumerated
# too and appear on no allowlist, so inheriting one is a hard refusal.
env_names_to_gate() { printf '%s\n' ${!YAW_EVAL_@} ${!YAW_AUG_@} ${!YAW_GEN_@} ${!FA_ORBIT_@} 2>/dev/null; }
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
TEST_MODE="${YAW_EVAL_TEST_MODE:-0}"
[ "$DRYRUN" = "1" ] && TEST_MODE=1      # a dry run submits nothing by construction
ALLOW_LIVE="YAW_EVAL_TEST_MODE YAW_EVAL_PINNED_EXEC"
#   YAW_EVAL_PINNED_EXEC  set BY THIS SCRIPT for its own re-exec from the pinned
#                         worktree; its presence means "you are the pinned copy"
#   YAW_EVAL_TEST_MODE      "1" selects simulation (0/unset = live)
ALLOW_TEST="YAW_EVAL_TEST_MODE YAW_EVAL_TEST_RECORD YAW_EVAL_TEST_JOBID \
YAW_EVAL_TEST_SUBMIT_RC YAW_EVAL_SQUEUE_FIXTURE YAW_EVAL_SQUEUE_FAILS \
YAW_EVAL_SYNC_FAILS YAW_EVAL_PIN_FILE YAW_EVAL_COMMAND_LOG YAW_EVAL_WT_DIR \
YAW_EVAL_INTENT_DIR YAW_EVAL_MAIN_REPO YAW_EVAL_CONTROL YAW_EVAL_REGISTRY \
YAW_EVAL_PINNED_EXEC"
#   YAW_EVAL_TEST_RECORD    path of a text file this script APPENDS its argv to
#   YAW_EVAL_TEST_JOBID     the job id a simulated submission reports (data)
#   YAW_EVAL_TEST_SUBMIT_RC the rc a simulated submission returns (data)
#   YAW_EVAL_SQUEUE_FIXTURE path of a text file holding "<jobid> <name>" lines —
#                           READ as data, never executed
#   YAW_EVAL_SQUEUE_FAILS   "1" makes the simulated queue query fail
#   YAW_EVAL_SYNC_FAILS     "1" makes the simulated flush fail
#   YAW_EVAL_MAIN_REPO      test-only root this script reads/writes under
#   YAW_EVAL_PIN_FILE       path of the campaign-pin file to read
#   YAW_EVAL_COMMAND_LOG    path of the command log to append to
#   YAW_EVAL_WT_DIR         directory whose .leases/ prove an in-flight job
#   YAW_EVAL_CONTROL        synthetic VANL admission record (classification only)
#   YAW_EVAL_REGISTRY       synthetic YAWAUG chain registry (classification only)
#   YAW_EVAL_INTENT_DIR     not read here: allowed because a test environment that
#                           drives the single-cell submitter carries it (data path)
if [ "$TEST_MODE" = "1" ]; then env_allowlist_gate "$ALLOW_TEST"
else                            env_allowlist_gate "$ALLOW_LIVE"; fi

MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
# TEST MODE may relocate the whole tree this script reads and writes. It is a
# DATA path like every other allowlisted seam, and it exists so a test harness is
# isolated BY DEFAULT rather than by remembering to be.
if [ "$TEST_MODE" = "1" ] && [ -n "${YAW_EVAL_MAIN_REPO:-}" ]; then
  MAIN_REPO="$YAW_EVAL_MAIN_REPO"
fi
EXPDIR="$MAIN_REPO/worklog/worklog_yixun/exp_15_yaw_aug_claude"
# See yaw_aug_screen_submit.sh: MAIN_REPO is the production tree (pin file,
# command log, outputs); CODE_ROOT is where the EXECUTABLE control plane is read
# from, which for a live wave is the pinned worktree (integrative review F1).
# Enumeration and classification decide which cells run and which are skipped as
# "already measured" — that is control-plane logic and must be pinned too.
CODE_ROOT="$MAIN_REPO"
[ -n "${YAW_EVAL_PINNED_EXEC:-}" ] && CODE_ROOT="$YAW_EVAL_PINNED_EXEC"
CODE_EXPDIR="$CODE_ROOT/worklog/worklog_yixun/exp_15_yaw_aug_claude"
HELPER="$MAIN_REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh"
SUBMIT="$CODE_EXPDIR/yaw_aug_screen_submit.sh"
VALIDATE="$CODE_EXPDIR/exp15_validate_cell.py"
PIN_FILE="$EXPDIR/yaw_aug_screen_campaign_pin"
# ONE command log for the whole experiment: the training legs and the eval cells
# are the same experiment's launches, and a reader reconstructing what was run
# should not have to know there were two kits. Eval rows are distinguishable by
# their exp15-screen- job names.
COMMAND_LOG="$EXPDIR/yaw_aug_command.md"
# (both may be redirected in TEST MODE only — see the TEST_MODE block below)
PY=/n/fs/gatrdp/envs/flac/bin/python     # the campaign interpreter, not overridable
# THE PRODUCTION TREE IS A CONSTANT IN LIVE MODE (exp_14 round-3 closure B4). An
# inherited OUTPUT_ROOT used to be accepted from the ambient environment and then
# classified against — while the single-cell submitter forwards the environment
# to sbatch with --export=ALL and the job itself ABORTS on a non-production
# value. A stray export therefore deduplicated against the wrong tree and
# released jobs guaranteed to abort. Redirection stays a TEST-MODE datum only.
PRODUCTION_OUTPUT_ROOT="$MAIN_REPO/outputs_FLAC"
# Redirection is honoured ONLY under an explicit YAW_EVAL_TEST_MODE=1. Not under
# DRYRUN: a dry run exits before classification, so a redirect buys it nothing and
# accepting one would leave a mode in which the printed plan and a later live wave
# disagree about which tree they mean.
if [ "${YAW_EVAL_TEST_MODE:-0}" = "1" ]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
else
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "NOTE: ignoring inherited OUTPUT_ROOT='${OUTPUT_ROOT}' - this wave" >&2
    echo "      classifies and submits against ${PRODUCTION_OUTPUT_ROOT} only." >&2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
fi
# exported EXPLICITLY so the job the submitter releases with --export=ALL reads
# the same tree this wave classified — the split between them was the defect.
export OUTPUT_ROOT
# NO ENVIRONMENT VARIABLE CARRIES AN EXECUTABLE, in any mode. Every seam that
# used to name a command (the submitter, squeue, sync, even the interpreter) is
# gone: an env var that is exec'd is a way to run something else, and no amount
# of "is it a mock?" reasoning makes that safe. Test mode SIMULATES those calls
# in-process; live mode runs binaries resolved to absolute paths at entry.

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
# shape-checked BEFORE assignment, exactly as in yaw_aug_screen_submit.sh.
for kv in "$@"; do
  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    WAVE)     in_set "$val" vctl tbl rrob all || reject "WAVE='${val}' is not vctl|tbl|rrob|all" ;;
    EXCLUDE)  is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
    PIN_SHA)  is_hex40 "$val" || reject "PIN_SHA='${val}' is not 40 hex characters" ;;
    *)        reject "unknown argument '${kv}' (expected WAVE=/EXCLUDE=/PIN_SHA=)" ;;
  esac
  printf -v "$key" '%s' "$val"
done
[ -n "$WAVE" ] || { echo "usage: [DRYRUN=1] bash $0 WAVE={vctl|tbl|rrob|all} [EXCLUDE=n1,n2] [PIN_SHA=<40hex>]" >&2; exit 2; }
is_num "$MAX_INFLIGHT" && [ "$MAX_INFLIGHT" -ge 1 ] && [ "$MAX_INFLIGHT" -le 16 ] \
  || reject "MAX_INFLIGHT='${MAX_INFLIGHT}' must be 1..16 (plan §10 caps this campaign at 16 slots)"
[ -f "$VALIDATE" ] || reject "validator missing: ${VALIDATE}"
[ -f "$SUBMIT" ]   || reject "single-cell submitter missing: ${SUBMIT}"
# An overridable SUBMITTER is exactly the hole a "is this a mock?" test cannot
# close: a wrapper, a copy or a hard link that calls the real one passes every
# such test. There is no override — in test mode the submission is SIMULATED
# in-process (submit_cell), and in live mode it is always this campaign's own
# submitter.
if [ "$TEST_MODE" = "1" ]; then
  # Only in test mode may the campaign's own state be redirected. A test that
  # writes the live command log and the live pin file is not a test of the wave,
  # it is an edit of the campaign — and a RED run of exp_14's suite once
  # submitted four real jobs that way.
  [ -n "${YAW_EVAL_PIN_FILE:-}" ] && PIN_FILE="$YAW_EVAL_PIN_FILE"
  [ -n "${YAW_EVAL_COMMAND_LOG:-}" ] && COMMAND_LOG="$YAW_EVAL_COMMAND_LOG"
else
  # LIVE: resolve the binaries ONCE, absolutely, on the PATH sanitized at entry
  # and with any exported function of the same name already dropped.
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
# A live wave REQUIRES the pin: 42 cells are comparable only if they ran at one
# commit, and without the file each submission would pin at whatever HEAD happens
# to be. DRYRUN does not need it (it submits nothing).
# PIN_SHA is an ASSERTION, never a substitute: passing it used to satisfy the
# "one commit" rule with no pin file at all, which is exactly the rail exp_14's
# review B3 said the round claimed to have.
# --- THE RE-EXEC MARKER IS VERIFIED, NEVER TRUSTED (re-review finding 1) -------
# YAW_EVAL_PINNED_EXEC used to be believed on sight: nonempty meant "you are the
# pinned copy", so an inherited or forged value redirected CODE_ROOT anywhere and
# skipped the bootstrap entirely — the pin bound nothing. A marker that selects
# which code executes has to be checked against something the caller cannot
# choose, so it is checked against BOTH:
#   (a) the canonical path the bootstrap would itself prepare for the pin file's
#       SHA — realpath equality, so a symlink or a sibling tree cannot pass; and
#   (b) that tree's own HEAD, which must equal the campaign pin exactly.
# A mismatch is a HARD REFUSAL, not a fallback to re-exec: if the marker is wrong
# we are already in a state nobody designed, and the safe move is to stop.
# DRYRUN is permissive (it submits nothing) and labels itself as such.
verify_pinned_marker() {   # $1 = claimed CODE_ROOT, $2 = campaign pin sha
                           # rc 0 = verified AND we ARE the pinned copy
                           # rc 3 = verified but $0 is elsewhere -> caller re-execs
                           # rc 1 = forged/mismatched -> caller HARD REFUSES
  local claimed="$1" pin="$2" want head self
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN: pinned-exec marker accepted UNVERIFIED (a dry run submits nothing)" >&2
    return 0
  fi
  if [ -z "$pin" ]; then
    echo "YAW_EVAL_PINNED_EXEC is set but there is no campaign pin to verify it against" >&2
    return 1
  fi
  want="$(readlink -f "${MAIN_REPO}/.measure_worktrees/${pin}" 2>/dev/null)" || want=""
  claimed="$(readlink -f "$claimed" 2>/dev/null)" || claimed=""
  if [ -z "$want" ] || [ -z "$claimed" ] || [ "$want" != "$claimed" ]; then
    echo "REFUSING: YAW_EVAL_PINNED_EXEC='${claimed}' is not the worktree this pin" >&2
    echo "  prepares ('${want}'). A marker that selects which code executes may not" >&2
    echo "  be taken on trust." >&2
    return 1
  fi
  head="$(git -C "$claimed" rev-parse HEAD 2>/dev/null)" || head=""
  if [ "$head" != "$pin" ]; then
    echo "REFUSING: the tree at '${claimed}' is at HEAD '${head}', not the campaign" >&2
    echo "  pin '${pin}'." >&2
    return 1
  fi
  # ...and the EXECUTING SCRIPT must itself live in that tree (arming-check
  # finding 1). Verifying the directory said nothing about who was running: a
  # correct marker handed to the MAIN-tree copy passed every check above and then
  # continued executing main-tree shell logic, with only subordinate paths
  # pointing into the pinned tree. A correct marker with the wrong $0 is not an
  # attack, it is a mis-invocation, so the safe move is to ROUTE to the pinned
  # copy (rc 3) rather than refuse; only a bad marker refuses.
  self="$(readlink -f "$0" 2>/dev/null)" || self=""
  case "$self" in
    "$claimed"/*) return 0 ;;
    *) echo "pinned-exec marker is correct but this script is '${self}', outside" >&2
       echo "  the verified tree — re-execing from that tree's own copy." >&2
       return 3 ;;
  esac
}

CAMPAIGN_PIN=""
if [ -f "$PIN_FILE" ]; then
  CAMPAIGN_PIN="$(head -1 "$PIN_FILE" | tr -d '[:space:]')"
  is_hex40 "$CAMPAIGN_PIN" || reject "campaign pin file ${PIN_FILE} does not hold a 40-hex sha"
fi
if [ "$DRYRUN" != "1" ]; then
  if [ -z "$CAMPAIGN_PIN" ]; then
    echo "refusing to submit a wave with no campaign pin FILE (${PIN_FILE})." >&2
    echo "  42 cells are comparable only if they ran at ONE commit, and PIN_SHA on a" >&2
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
if [ -n "${YAW_EVAL_PINNED_EXEC:-}" ] && [ "$TEST_MODE" != "1" ]; then
  MARKER_RC=0
  verify_pinned_marker "$YAW_EVAL_PINNED_EXEC" "$CAMPAIGN_PIN" || MARKER_RC=$?
  case "$MARKER_RC" in
    0) echo "pinned-exec marker VERIFIED against ${CAMPAIGN_PIN}" >&2 ;;
    3) PINNED_SELF="${YAW_EVAL_PINNED_EXEC}/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh"
       [ -f "$PINNED_SELF" ] || reject "the verified tree has no wave submitter at ${PINNED_SELF}"
       echo "re-exec from the VERIFIED pinned copy: ${PINNED_SELF}" >&2
       exec bash "$PINNED_SELF" "$@" ;;
    *) exit 2 ;;
  esac
fi

# --- RE-EXEC FROM THE PINNED WORKTREE (integrative review F1) ------------------
# A live wave prepares (or reuses) the pinned tree and then runs ITSELF from it,
# so enumeration, classification and the single-cell submitter it calls are all
# the pinned versions. DRYRUN stays main-tree on purpose — it submits nothing —
# and says so out loud below.
if [ "$DRYRUN" != "1" ] && [ "$TEST_MODE" != "1" ] && [ -z "${YAW_EVAL_PINNED_EXEC:-}" ]; then
  [ -f "$HELPER" ] || reject "measure-worktree helper missing: ${HELPER}"
  WT_PREPARED="$(bash "$HELPER" "$PIN_SHA" | tail -1)"
  [ -d "$WT_PREPARED" ] || reject "could not prepare the pinned worktree for ${PIN_SHA}"
  PINNED_SELF="${WT_PREPARED}/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_submit_grid.sh"
  [ -f "$PINNED_SELF" ] || reject "the pinned worktree has no wave submitter at ${PINNED_SELF}"
  echo "re-exec from the PINNED control plane: ${PINNED_SELF}" >&2
  YAW_EVAL_PINNED_EXEC="$WT_PREPARED"; export YAW_EVAL_PINNED_EXEC
  exec bash "$PINNED_SELF" "$@"
fi

# --- the grid ----------------------------------------------------------------
# Enumerated by the VALIDATOR, which is also what the driver and the guard suite
# read: one definition of "the registered grid", not three. The seventh field is
# the canonical Slurm job name (exp15_validate_cell.job_name) — emitted with the
# grid so this script needs no shell re-implementation of it and no python
# process per cell. exp_14 rendered it in shell here and pinned it with a guard
# case; that whole drift class is gone.
mapfile -t CELLS < <("$PY" "$VALIDATE" grid --wave "$WAVE" --with-jobname) \
  || reject "could not enumerate the ${WAVE} wave"
[ "${#CELLS[@]}" -gt 0 ] || reject "the ${WAVE} wave is empty"
# The banner goes to STDERR so a DRYRUN's stdout is exactly one line per cell
# (the guard suite diffs it against the registered grid).
echo "=== exp_15 ${WAVE} wave: ${#CELLS[@]} registered cells | pin ${PIN_SHA:-<none: DRYRUN>} ===" >&2
if [ -n "${YAW_EVAL_PINNED_EXEC:-}" ]; then
  echo "=== control plane: PINNED (${CODE_ROOT}) ===" >&2
else
  echo "=== control plane: MAIN-TREE (unpinned) — dry run only; a live wave re-execs from the pin ===" >&2
fi

test_record() { [ -n "${YAW_EVAL_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_EVAL_TEST_RECORD"; return 0; }

sync_file() {   # flush one file's bytes to disk; NONZERO when it cannot be done.
                # A record that is not durable is not a record — that is the whole
                # point of writing it before the launch — so an unconditional
                # success here made the fatal check unreachable.
  local f="$1"
  if [ "$TEST_MODE" = "1" ]; then
    test_record "sync ${f}"
    [ "${YAW_EVAL_SYNC_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_sync" "$f" 2>/dev/null && return 0
  "$BIN_sync" 2>/dev/null && return 0         # coreutils without per-file sync
  return 1
}

queue_lines() {  # "<jobid> <name>" per line; NONZERO when the queue cannot be read
  if [ "$TEST_MODE" = "1" ]; then
    [ "${YAW_EVAL_SQUEUE_FAILS:-0}" = "1" ] && return 1
    # A FIXTURE FILE, read as data. There is no command here to substitute.
    [ -n "${YAW_EVAL_SQUEUE_FIXTURE:-}" ] || return 0
    cat "$YAW_EVAL_SQUEUE_FIXTURE" 2>/dev/null || return 1
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
# DRYRUN runs BEFORE classification on purpose, so it still enumerates the whole
# grid while the training chain is unfinished and YAWAUG has no admission record
# yet. It submits nothing and reads no checkpoint.
if [ "$DRYRUN" = "1" ]; then
  echo "=== classification root: ${OUTPUT_ROOT} ===" >&2
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
CLASSIFY_ARGV=(classify --wave "$WAVE" --output-root "$OUTPUT_ROOT" --pin "$PIN_SHA")
# Synthetic admission sources are a TEST datum, gated exactly like every other
# seam: only under YAW_EVAL_TEST_MODE=1, where no submission can reach sbatch.
if [ "$TEST_MODE" = "1" ]; then
  [ -z "${YAW_EVAL_CONTROL:-}" ]  || CLASSIFY_ARGV+=(--control "$YAW_EVAL_CONTROL")
  [ -z "${YAW_EVAL_REGISTRY:-}" ] || CLASSIFY_ARGV+=(--registry "$YAW_EVAL_REGISTRY")
fi
mapfile -t STATUS < <("$PY" "$VALIDATE" "${CLASSIFY_ARGV[@]}") \
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
# never produce it (exp_14 review B5).
#
# A squeue that FAILS is not evidence that nothing is running (exp_11's lease
# reaper learned this the expensive way): refuse rather than flood the queue.
WT_DIR="${MAIN_REPO}/.measure_worktrees/${PIN_SHA}"
# In test mode only — a LIVE wave carrying this variable never gets here at all:
# the entry allowlist refuses it outright, which is stronger than ignoring it.
if [ -n "${YAW_EVAL_WT_DIR:-}" ] && [ "$TEST_MODE" = "1" ]; then
  WT_DIR="$YAW_EVAL_WT_DIR"
  echo "TEST MODE: lease directory overridden to ${WT_DIR}" >&2
fi
QUEUED="$(queue_lines)" \
  || reject "squeue failed - refusing to submit without knowing what is already queued"
inflight_count() { printf '%s\n' "$QUEUED" | awk '{print $2}' | grep -c '^exp15-screen-' ; }
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
  ARM="$1"; CELL="$2"; STEP="$3"; SEED="$4"; K="$5"; ROT="$6"; JOB_NAME="$7"
  row="${STATUS[$i]}"
  case "$row" in
    "${ARM} ${CELL} ${STEP} ${SEED} ${K} ${ROT} "*) ;;
    *) reject "classifier row '${row}' does not describe cell '${CELLS[$i]}' - refusing to guess" ;;
  esac
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
    echo "  waiting: $(inflight_count) exp15-screen- jobs in flight (cap ${MAX_INFLIGHT}); sleeping ${POLL_SECONDS}s"
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
      [ -n "${YAW_EVAL_TEST_SUBMIT_RC:-}" ] && return "$YAW_EVAL_TEST_SUBMIT_RC"
      echo "submitted HELD as ${YAW_EVAL_TEST_JOBID:-5550001}"
      return 0
    fi
    bash "$SUBMIT" "$@"
  }
  # The command log is written BEFORE the submission and flushed to disk, and a
  # failure to write it is FATAL (exp_14 review B7). Recording afterwards means a
  # crash between sbatch and the append produces a job nobody can trace back to a
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
