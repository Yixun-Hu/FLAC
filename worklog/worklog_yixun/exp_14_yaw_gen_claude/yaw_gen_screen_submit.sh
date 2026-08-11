#!/usr/bin/env bash
# ============================================================================
# yaw_gen_screen_submit.sh — submit ONE exp_14 cell against a pinned, LEASED
# measurement worktree. Adapted from exp_11's fa_orbit_screen_submit.sh; the
# transaction below is unchanged, only the campaign it submits for.
#
# For a whole wave use yaw_gen_submit_grid.sh, which enumerates the registered
# grid, dedups against landed artifacts and calls THIS script per cell.
#
# Screens used to be submitted by hand, which is how job 3649599 ended up
# reading code that moved underneath it. Submitting through this script makes
# the three things that must be atomic actually atomic:
#
#   0. LOCK SPAN — the whole sequence runs inside the store lock (the script
#                  re-execs itself under `fa_orbit_measure_worktree.sh
#                  --with-lock`). Locking each step separately still left a gap
#                  between "tree prepared" and "lease written" in which a prune
#                  sweep sees a brand-new unleased tree and deletes it.
#   1. pin      — a worktree at the submission SHA, assets provisioned
#   2. LEASE     — via sbatch --hold. Submitting held is what removes the last
#                  two races: the job id exists BEFORE the lease is written, and
#                  the job cannot start until it is released, so there is no
#                  window in which a queued job has no lease (a pruner would
#                  have seen its tree as free) and none in which a lease names a
#                  placeholder instead of a job. If the lease write or the
#                  release fails, the held job is CANCELLED — never orphaned.
#   3. EXPECT_SHA — taken from the pinned tree, not from the (moving) main HEAD
#
set -euo pipefail
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

# The MODE is known from the environment alone (no arguments needed), so the gate
# runs HERE — before the manifest reader, before the store-lock re-exec, before
# anything else this script can do.
DRYRUN="${DRYRUN:-0}"
TEST_MODE=0
[ "${YAW_GEN_TEST_MODE:-0}" = "1" ] && TEST_MODE=1
[ "$DRYRUN" = "1" ] && TEST_MODE=1        # a dry run submits nothing by construction
ALLOW_LIVE="FA_ORBIT_STORE_LOCK_HELD YAW_GEN_TEST_MODE"
#   FA_ORBIT_STORE_LOCK_HELD  "1" from the shared store helper's lock protocol
#   YAW_GEN_TEST_MODE         "1" selects simulation (0/unset = live)
ALLOW_TEST="FA_ORBIT_STORE_LOCK_HELD YAW_GEN_TEST_MODE YAW_GEN_TEST_RECORD \
YAW_GEN_TEST_JOBID YAW_GEN_TEST_SUBMIT_SLEEP YAW_GEN_TEST_RELEASE_SLEEP \
YAW_GEN_TEST_RELEASE_FAILS YAW_GEN_TEST_SCANCEL_FAILS YAW_GEN_SYNC_FAILS \
YAW_GEN_PIN_FILE YAW_GEN_INTENT_DIR YAW_GEN_MAIN_REPO"
#   YAW_GEN_TEST_RECORD        path of a text file this script APPENDS its argv to
#   YAW_GEN_TEST_JOBID         the job id a simulated submission reports (data)
#   YAW_GEN_TEST_SUBMIT_SLEEP  seconds a simulated submission stalls (data)
#   YAW_GEN_TEST_RELEASE_SLEEP seconds a simulated release stalls (data)
#   YAW_GEN_TEST_RELEASE_FAILS "1" makes the simulated release fail
#   YAW_GEN_TEST_SCANCEL_FAILS "1" makes the simulated cancel fail
#   YAW_GEN_SYNC_FAILS         "1" makes the simulated flush fail
#   YAW_GEN_MAIN_REPO          test-only root this script reads/writes under
#   YAW_GEN_PIN_FILE           path of the campaign-pin file to read
#   YAW_GEN_INTENT_DIR         directory the intent manifest is written to
# The re-exec below runs this script again from the top, so the child re-runs
# this gate on the environment it actually received.
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
# The measure-worktree store is SHARED (one lock, one freeze, one lease space for
# every campaign on this machine) and its helper lives in exp_11's folder, which
# is read-only to this experiment. Referenced in place, never copied: a second
# copy would be a second lock discipline over the same directory.
HELPER="$MAIN_REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh"

# --- OUTER ENTRY: prove the lock before anything transactional happens -------
# Re-exec under the store lock so that preparation, submission and leasing are
# ONE atomic span. fd 8 is inherited, so the nested helper calls below re-enter
# the same lock instead of deadlocking on a second one.
#
# The marker alone is not proof. It is an ordinary environment variable, and
# anything that inherits it — a stale export, a wrapper, a rerun of a crashed
# shell — would carry it into a run holding no lock at all. Verifying it inside
# the helper is too late: by then this script has already prepared a tree. So
# the check lives HERE, at the outer entry, before any transaction step.
# The intent manifest ends with a fixed SENTINEL, and this is the reader that
# checks it. A nonempty file is not a complete one: a crash mid-write leaves a
# plausible-looking manifest that a collector would read as the launch record.
MANIFEST_SENTINEL="manifest_complete yes"

verify_manifest_complete() {   # <path>; nonzero unless it ends with the sentinel
  [ -f "$1" ] || return 1
  [ "$(tail -1 "$1" 2>/dev/null)" = "$MANIFEST_SENTINEL" ] || return 1
  return 0
}

sync_file() {   # flush one file's bytes; NONZERO when durability cannot be had.
                # No environment variable names the flusher: in test mode this is
                # an internal no-op that RECORDS itself, and in live mode it is the
                # absolute path resolved at entry (Z1/Z2).
  if [ "${TEST_MODE:-0}" = "1" ]; then
    test_record "sync $1"
    [ "${YAW_GEN_SYNC_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_sync" "$1" 2>/dev/null && return 0
  "$BIN_sync" 2>/dev/null && return 0
  return 1
}

# `--verify-manifest <path>` is the read-only reader entry point (guard suite and
# operators); it takes no lock, submits nothing and reads no environment seam.
if [ "${1:-}" = "--verify-manifest" ]; then
  verify_manifest_complete "${2:?a manifest path}" \
    && { echo "manifest complete: $2"; exit 0; }
  echo "manifest INCOMPLETE (no '${MANIFEST_SENTINEL}' sentinel): $2" >&2
  exit 1
fi

LOCKFILE="${MAIN_REPO}/.measure_worktrees/.store.lock"

fd8_is_the_store_lock() {
  local have want
  [ -e /proc/self/fd/8 ] || return 1
  have="$(readlink -f /proc/self/fd/8 2>/dev/null)" || return 1
  want="$(readlink -f "$LOCKFILE" 2>/dev/null)" || return 1
  # An unresolvable path yields the EMPTY string, and two empty strings compare
  # equal — which would turn a double failure into a "match". Require both.
  [ -n "$have" ] && [ -n "$want" ] && [ "$have" = "$want" ]
}

if [ "${FA_ORBIT_STORE_LOCK_HELD:-0}" = "1" ]; then
  fd8_is_the_store_lock || {
    echo "FA_ORBIT_STORE_LOCK_HELD is set but fd 8 is not ${LOCKFILE} —" >&2
    echo "refusing to run a submission that only CLAIMS to hold the store lock" >&2
    exit 2
  }
elif [ "$DRYRUN" = "1" ]; then
  # A dry run performs no transaction: it prepares no worktree, writes no lease
  # and submits nothing, so it takes no store lock and cannot contend with a live
  # campaign. It exists so the argv this script would build is inspectable
  # WITHOUT submitting (the reviewer could not exercise this script at all).
  :
else
  exec bash "$HELPER" --with-lock bash "$0" "$@"
fi

ARM=""; STEP=40000; SEED=42; K=8; CELL=""; EXCLUDE=""; ROTATE_DEG=""
PIN_SHA=""; LOG=""

# --- argument parsing: NEVER eval a value ------------------------------------
# The old parser ran `eval "KEY='VALUE'"`, so a value carrying a quote executed
# whatever followed it. Every key is whitelisted, every VALUE is shape-checked
# BEFORE it is assigned, and assignment goes through printf -v into a name that
# came from the whitelist -- the value is never parsed as shell at any point.
is_num()      { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
is_decimal()  { case "${1:-}" in ''|*[!0-9.]*|.*|*.) return 1 ;; *.*.*) return 1 ;; esac; }
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

for kv in "$@"; do
  case "$kv" in
    *=*) ;;
    *) reject "argument '${kv}' is not KEY=VALUE" ;;
  esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    ARM)        in_set "$val" C4L C8 C16 C32 VANL \
                  || reject "ARM='${val}' is not a registered arm" ;;
    CELL)       in_set "$val" rgen zref vctl \
                  || reject "CELL='${val}' is not a registered cell type" ;;
    STEP)       is_num "$val" || reject "STEP='${val}' is not numeric" ;;
    SEED)       is_num "$val" || reject "SEED='${val}' is not numeric" ;;
    K)          in_set "$val" 1 8 || reject "K='${val}' is not 1 or 8" ;;
    ROTATE_DEG) is_decimal "$val" || reject "ROTATE_DEG='${val}' is not a decimal number" ;;
    PIN_SHA)    is_hex40 "$val" || reject "PIN_SHA='${val}' is not 40 hex characters" ;;
    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
    LOG)        # kept to a durable, provenance-preserving location on purpose:
                # the overnight incidents were a tee into the pinned worktree and
                # node-local /tmp paths, both of which this shape excludes
                case "$val" in
                  "${EXPDIR}"/*_screen.log) ;;
                  *) reject "LOG='${val}' must be an absolute ${EXPDIR}/..._screen.log path" ;;
                esac
                case "$val" in *[!A-Za-z0-9/._-]*) reject "LOG='${val}' has unsafe characters" ;; esac ;;
    *)          reject "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/PIN_SHA=/LOG=)" ;;
  esac
  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
done

# --- the CAMPAIGN PIN is the default, and disagreement is refused ------------
# exp_14 keeps its OWN pin file rather than the store-wide .campaign_pin marker:
# the store is shared with other campaigns (exp_11's screens, exp_15's), one
# marker cannot mean two commits at once, and a campaign that read someone else's
# pin would measure half its grid somewhere else. Refusal semantics are the
# store marker's, unchanged.
# TEST SEAMS, and the ONE rule that governs them: they are honoured only when the
# submission path is provably a MOCK. A run that could really submit may never
# redirect the campaign's own state — the guard suite writing the live
# yaw_gen_command.md and yaw_gen_campaign_pin is precisely how a RED test run of
# this suite once submitted four real jobs.
# TEST MODE is a MODE, not a mock. Deciding "is this executable a mock?" is not
# decidable — a wrapper, a copy, a hard link or an absolute path to the real
# sbatch all differ from the string "sbatch" — so nothing is asked about the
# executable any more: in test mode this script EXECUTES NO SUBMIT COMMAND AT
# ALL. It simulates the Slurm interaction internally and records the argv it
# would have run. A test therefore cannot reach sbatch even by accident, which is
# the only property worth having (an earlier "mock" rule let a guard run submit
# four real jobs).
# The mode and the allowlist were settled at the top of this script. What remains
# for LIVE mode is resolving the binaries it will actually run.
if [ "$TEST_MODE" != "1" ]; then
  # Resolve every Slurm binary ONCE, absolutely, on the sanitized PATH set at
  # entry and with any exported shell function of the same name already dropped.
  # Only these stored paths are invoked afterwards (Z2).
  for _n in sbatch scontrol scancel sync; do
    _p="$(command -v "$_n" 2>/dev/null)" || _p=""
    case "$_p" in
      /*) ;;
      *) reject "cannot resolve an absolute path for '${_n}' on the sanitized PATH" ;;
    esac
    [ -x "$_p" ] || reject "resolved '${_p}' for ${_n} is not executable"
    printf -v "BIN_${_n}" '%s' "$_p"
  done
  echo "resolved: sbatch=${BIN_sbatch} scontrol=${BIN_scontrol} scancel=${BIN_scancel}"
fi
PIN_FILE="${EXPDIR}/yaw_gen_campaign_pin"
INTENT_DIR="$EXPDIR"
if [ "$TEST_MODE" = "1" ]; then
  [ -n "${YAW_GEN_PIN_FILE:-}" ] && PIN_FILE="$YAW_GEN_PIN_FILE"
  [ -n "${YAW_GEN_INTENT_DIR:-}" ] && INTENT_DIR="$YAW_GEN_INTENT_DIR"
elif [ -n "${YAW_GEN_PIN_FILE:-}${YAW_GEN_INTENT_DIR:-}" ]; then
  reject "YAW_GEN_PIN_FILE / YAW_GEN_INTENT_DIR are test seams: they need YAW_GEN_TEST_MODE=1, never a run that could really submit"
fi
CAMPAIGN_PIN=""
if [ -f "$PIN_FILE" ]; then
  CAMPAIGN_PIN="$(head -1 "$PIN_FILE" | tr -d '[:space:]')"
  is_hex40 "$CAMPAIGN_PIN" \
    || reject "campaign pin file ${PIN_FILE} does not hold a 40-hex commit sha"
fi
if [ -n "$CAMPAIGN_PIN" ]; then
  if [ -z "$PIN_SHA" ]; then
    PIN_SHA="$CAMPAIGN_PIN"
    echo "campaign pin (from ${PIN_FILE}): ${PIN_SHA}"
  elif [ "$PIN_SHA" != "$CAMPAIGN_PIN" ]; then
    echo "PIN_SHA=${PIN_SHA} disagrees with the campaign pin ${CAMPAIGN_PIN}." >&2
    echo "A campaign measures every arm at ONE commit; submitting one cell elsewhere" >&2
    echo "would silently make it incomparable. To change the pin deliberately:" >&2
    echo "  printf '%s\\n' ${PIN_SHA} > ${PIN_FILE}" >&2
    exit 2
  fi
fi
if [ -n "$PIN_SHA" ]; then
  git -C "$MAIN_REPO" rev-parse --verify --quiet "${PIN_SHA}^{commit}" >/dev/null \
    || reject "PIN_SHA ${PIN_SHA} is not a commit in this repository"
fi
[ -n "$ARM" ] && [ -n "$CELL" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C4L CELL=rgen [STEP=40000] [SEED=42] [K=8] [EXCLUDE=node[,node]] [ROTATE_DEG=90] [PIN_SHA=<40hex>] [LOG=...]" >&2; exit 2; }
# CELL is REQUIRED and has no default here or in the driver: a cell type is a
# protocol choice (announcement 05), and defaulting one would let an omitted
# argument decide the experiment.

# --- DRYRUN: print what would be submitted, touch nothing --------------------
if [ "$DRYRUN" = "1" ]; then
  CELL_EXPORT_DRY=""
  [ -n "$ROTATE_DEG" ] && CELL_EXPORT_DRY="${CELL_EXPORT_DRY},ROTATE_DEG=${ROTATE_DEG}"
  [ -n "$LOG" ] && CELL_EXPORT_DRY="${CELL_EXPORT_DRY},LOG=${LOG}"
  JOB_TOKEN_DRY=""
  case "$CELL" in
    rgen) JOB_TOKEN_DRY="-rotrand${SEED}" ;;
    vctl) JOB_TOKEN_DRY="-rot${ROTATE_DEG%.0}" ;;
  esac
  echo "DRYRUN job-name exp14-screen-${ARM}-${CELL}${JOB_TOKEN_DRY}-${STEP}-s${SEED}-K${K}"
  echo "DRYRUN pin ${PIN_SHA:-<none: HEAD>}"
  echo "DRYRUN exclude ${EXCLUDE:-<none>}"
  echo "DRYRUN export ARM=${ARM},STEP=${STEP},SEED=${SEED},K=${K},CELL=${CELL}${CELL_EXPORT_DRY}"
  echo "DRYRUN driver ${EXPDIR}/yaw_gen_screen.sbatch"
  echo "DRY RUN complete: no worktree prepared, no lease written, nothing submitted"
  exit 0
fi

# 0b. PREFLIGHT: the campaign freeze must be engaged.
# The campaign's validity argument rests on "no worktree is deleted while it
# runs". Leaving that to operator discipline makes it a promise; requiring the
# marker here makes it a precondition — a submission cannot happen unless the
# guarantee is mechanically in force.
FREEZE_MARKER="${MAIN_REPO}/.measure_worktrees/.campaign_freeze"
if [ -e "$FREEZE_MARKER" ]; then
  echo "campaign freeze: ACTIVE — $(head -1 "$FREEZE_MARKER" 2>/dev/null)"
else
  echo "campaign freeze: ABSENT" >&2
  echo "refusing to submit: a measurement campaign requires the deletion freeze." >&2
  echo "  engage it with: bash ${HELPER} --freeze" >&2
  exit 2
fi

# 1. pin + assets (the helper refuses a dirty or mismatched tree). With PIN_SHA
#    the tree is prepared at THAT commit and EXPECT_SHA follows it, so the job's
#    commit binding checks the pin, not wherever the branch has drifted to.
WT="$("$HELPER" ${PIN_SHA:+"$PIN_SHA"} | tail -1)"
[ -d "$WT" ] || { echo "could not prepare a measurement worktree" >&2; exit 3; }
EXPECT_SHA="$(git -C "$WT" rev-parse HEAD)"
if [ -n "$PIN_SHA" ] && [ "$EXPECT_SHA" != "$PIN_SHA" ]; then
  echo "the prepared tree is at ${EXPECT_SHA}, not the requested PIN_SHA ${PIN_SHA}" >&2
  exit 3
fi
[ -n "$PIN_SHA" ] && echo "campaign pin: ${PIN_SHA}"

# 2. submit HELD: the id exists before the lease, the job runs after it
# The job name starts with exp14- so the wave submitter can count THIS campaign's
# jobs in squeue without catching a neighbour's, and it carries the ROTATION
# TOKEN: C4L vctl@45 and vctl@90 differ in nothing else, so without it a wave
# watching squeue reads one as the other's in-flight job and silently drops a
# registered validity control (review B2). Rendered in shell here and in
# yaw_gen_submit_grid.sh; a guard case pins both against
# exp14_validate_cell.job_name, which is the one definition.
JOB_TOKEN=""
case "$CELL" in
  rgen) JOB_TOKEN="-rotrand${SEED}" ;;
  vctl) JOB_TOKEN="-rot${ROTATE_DEG%.0}" ;;
esac
JOB_NAME="exp14-screen-${ARM}-${CELL}${JOB_TOKEN}-${STEP}-s${SEED}-K${K}"
# --- the ONLY three places this script talks to Slurm ------------------------
# In TEST MODE each one records what it would have done and returns a scripted
# result; no process is started. In LIVE mode each runs the canonical binary.
test_record() { [ -n "${YAW_GEN_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_GEN_TEST_RECORD"; return 0; }

slurm_submit_hold() {   # prints a job id on stdout, like `sbatch --parsable`
  if [ "$TEST_MODE" = "1" ]; then
    test_record "sbatch $*"
    [ -n "${YAW_GEN_TEST_SUBMIT_SLEEP:-}" ] \
      && { : > "${YAW_GEN_TEST_RECORD:-/dev/null}.submitting"; sleep "$YAW_GEN_TEST_SUBMIT_SLEEP"; }
    case "$*" in *--hold*) ;; *) echo "TEST: submit called WITHOUT --hold" >&2; return 9 ;; esac
    printf '%s\n' "${YAW_GEN_TEST_JOBID:-7654321}"
    return 0
  fi
  "$BIN_sbatch" "$@"
}
slurm_release() {       # $1 = job id
  if [ "$TEST_MODE" = "1" ]; then
    test_record "scontrol release $1"
    grep -q "^jobid $1$" "${WT}/.leases/$1" 2>/dev/null \
      && test_record "release saw a VALID lease for $1"
    # a window the suite can interrupt, to prove a signal cancels the held job
    [ -n "${YAW_GEN_TEST_RELEASE_SLEEP:-}" ] \
      && { : > "${YAW_GEN_TEST_RECORD:-/dev/null}.releasing"; sleep "$YAW_GEN_TEST_RELEASE_SLEEP"; }
    [ "${YAW_GEN_TEST_RELEASE_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_scontrol" release "$1"
}
slurm_cancel() {        # $1 = job id, or --name=<job name>
  if [ "$TEST_MODE" = "1" ]; then
    test_record "scancel $1"
    [ "${YAW_GEN_TEST_SCANCEL_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_scancel" "$1"
}
# Node exclusion is passed as an EXPLICIT FLAG, never through the environment.
# SBATCH_EXCLUDE does not exist: of the 58 input environment variables sbatch
# documents there is no --exclude equivalent (the lookalike SBATCH_EXCLUSIVE is
# --exclusive, a different option entirely). sbatch therefore ignored it in
# silence, and every batch that believed it was excluding sick nodes was not.
EXCLUDE_ARGV=()
if [ -n "$EXCLUDE" ]; then
  EXCLUDE_ARGV=(--exclude="$EXCLUDE")
  echo "excluding nodes: ${EXCLUDE}"
elif [ -n "${SBATCH_EXCLUDE:-}" ]; then
  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
  exit 2
fi
# CELL-specific parameters travel with the job, not as ambient state.
CELL_EXPORT=""
[ -n "$ROTATE_DEG" ] && CELL_EXPORT="${CELL_EXPORT},ROTATE_DEG=${ROTATE_DEG}"
[ -n "$LOG" ] && CELL_EXPORT="${CELL_EXPORT},LOG=${LOG}"
# --- the abort guard is armed BEFORE the submission, not after ---------------
# `sbatch --hold` returning and the id being parsed are two different moments,
# and between them lived a fatal exit (a malformed id) plus a signal window: a
# job could exist with nothing arranged to cancel it. The guard is therefore
# armed FIRST, and it can cancel with or without an id — the job NAME is known
# before submission and is cell-injective (review B2), so `scancel --name=` is
# the fallback. Cancelling a name that was never submitted is harmless; leaving a
# held job that nobody knows about is not.
ABORT_ACTIVE=1
HELD_JOBID=""
cancel_held_job() {
  [ "${ABORT_ACTIVE:-0}" = "1" ] || return 0
  ABORT_ACTIVE=0
  if [ -n "${HELD_JOBID:-}" ]; then
    echo "leaving job ${HELD_JOBID} HELD is not an option - cancelling it (lease RETAINED)" >&2
    slurm_cancel "$HELD_JOBID" \
      || echo "scancel FAILED - cancel job ${HELD_JOBID} by hand" >&2
  else
    echo "aborting with no job id in hand - cancelling by NAME ${JOB_NAME} in case the" >&2
    echo "submission created a job we never learned the id of (lease RETAINED)" >&2
    slurm_cancel "--name=${JOB_NAME}" \
      || echo "scancel --name=${JOB_NAME} FAILED - check squeue by hand" >&2
  fi
}
trap cancel_held_job EXIT
trap 'cancel_held_job; exit 130' INT      # a signal must TERMINATE, not fall
trap 'cancel_held_job; exit 143' TERM     # through into the release below

JOBID="$(slurm_submit_hold --hold --parsable \
  --job-name="$JOB_NAME" \
  --output="${EXPDIR}/slurm_screen_%x_%j.out" \
  "${EXCLUDE_ARGV[@]}" \
  --export=ALL,MEASURE_ROOT="$WT",EXPECT_SHA="$EXPECT_SHA",ARM="$ARM",STEP="$STEP",SEED="$SEED",K="$K",CELL="$CELL""$CELL_EXPORT" \
  "$EXPDIR/yaw_gen_screen.sbatch")" || { echo "sbatch FAILED - nothing submitted" >&2; exit 4; }
JOBID="${JOBID%%;*}"
case "$JOBID" in
  ''|*[!0-9]*)
    # The submission may well have SUCCEEDED and only its output be unreadable,
    # so this exits through the same guard — by name, since there is no id.
    echo "sbatch returned '${JOBID}', not a job id - abort" >&2; exit 4 ;;
esac
HELD_JOBID="$JOBID"
echo "submitted HELD as ${JOBID}"

# 3. lease it by its real id, VALIDATE the lease, and only THEN release the job.
#    All of this still runs inside the store lock taken at step 0.
#
#    THE LEASE IS NEVER DROPPED ON AN ERROR PATH. Once a lease exists, the only
#    safe thing to do with an ambiguous outcome is to keep it: if scancel failed,
#    or succeeded but we cannot be sure, the job may yet run — and a job whose
#    lease we tidied away is a job whose worktree a sweep will delete underneath
#    it. A lease that outlives its job costs one held worktree until the reaper
#    proves, via a successful squeue, that the id is gone. That asymmetry is the
#    whole design: retaining is cheap, dropping is unrecoverable.
#
#    add_lease re-checks the worktree's identity before writing: a lease on a
#    directory that is not a live registered worktree would send the job to a
#    tree with no code in it.
if ! bash "$HELPER" --lease "$JOBID" "$WT"; then
  echo "could not write the lease for ${JOBID} - cancelling the held job" >&2
  cancel_held_job
  exit 5
fi

# Validate BEFORE releasing: while the job is still held, a bad lease costs a
# cancellation; after release it is a running job we cannot vouch for.
if ! grep -q "^jobid ${JOBID}$" "${WT}/.leases/${JOBID}" 2>/dev/null; then
  echo "lease ${WT}/.leases/${JOBID} does not name ${JOBID} - cancelling (lease RETAINED)" >&2
  cancel_held_job
  exit 7
fi
echo "lease validated: ${WT}/.leases/${JOBID}"

# --- intent manifest: written while the job is STILL HELD (review B7) ---------
# Writing it after the release produced a window in which a running job had no
# launch record, and a failure to write it was only a warning — so a crash could
# leave an undocumented job reading a pinned worktree. The manifest is therefore
# published (atomically, tmp+rename in the same directory) BEFORE the release,
# and a failure to publish it CANCELS the held job: a job with no launch record
# is worse than no job.
INTENT="${INTENT_DIR}/yaw_gen_submission_${ARM}_${CELL}_S${STEP}_s${SEED}_K${K}_jid${JOBID}.txt"
# The intent manifest records the EVAL-PROTOCOL flags this cell will run under,
# not just its identity (announcement 05: a mismatched flag produces
# plausible-looking, catastrophically wrong numbers, so the launch record has to
# be readable for protocol compliance on its own).
case "$CELL" in
  rgen) ROT_INTENT="rotate_mode random rotate_seed ${SEED} rotate_deg <null>" ;;
  vctl) ROT_INTENT="rotate_mode fixed rotate_seed <n/a> rotate_deg ${ROTATE_DEG}" ;;
  *)    ROT_INTENT="rotate_mode fixed rotate_seed <n/a> rotate_deg 0" ;;
esac
publish_intent() {   # write, flush, verify, then atomically replace. Any failure
                     # is a failure of the whole publication, never a warning.
  local tmp="${INTENT}.tmp"
  rm -f "$tmp"
  { printf 'job %s name %s submitted_at %s by %s@%s\n' "$JOBID" "$JOB_NAME" "$(date -Is)" \
           "$(id -un)" "$(hostname)"
    printf 'arm %s cell %s step %s seed %s K %s\n' "$ARM" "$CELL" "$STEP" "$SEED" "$K"
    printf 'pin_sha %s campaign_pin %s\n' "${PIN_SHA:-<none:HEAD>}" "${CAMPAIGN_PIN:-<none>}"
    printf 'expect_sha %s\n' "$EXPECT_SHA"
    printf 'measure_root %s\n' "$WT"
    printf 'exclude %s\n' "${EXCLUDE:-<none>}"
    printf '%s\n' "$ROT_INTENT"
    printf 'batch_size 64 num_workers 4 expected_stream_count 6337 record_stream yes\n'
    printf 'cond_autocast bf16 cfg_scale 1.0 steps 1 use_ema yes\n'
    printf 'lease %s\n' "${WT}/.leases/${JOBID}"
    printf '%s\n' "$MANIFEST_SENTINEL"
  } > "$tmp" || return 1
  sync_file "$tmp" || return 1
  # the RENAME SOURCE must itself be complete before it is published
  verify_manifest_complete "$tmp" || return 1
  mv -f "$tmp" "$INTENT" || return 1
  verify_manifest_complete "$INTENT" || return 1
}
if ! publish_intent; then
  rm -f "${INTENT}.tmp"
  echo "could not publish the intent manifest ${INTENT} - cancelling the held job" >&2
  echo "(a job with no launch record is worse than no job; the lease is RETAINED)" >&2
  cancel_held_job
  exit 8
fi
echo "intent manifest: ${INTENT}"

# 4. ...and only now let it run.
if ! slurm_release "$JOBID"; then
  echo "could not release ${JOBID} - cancelling it; the lease is RETAINED" >&2
  if slurm_cancel "$JOBID"; then
    echo "cancelled ${JOBID}; its lease stays until squeue proves the id is gone" >&2
  else
    echo "scancel FAILED too - job ${JOBID} may still run; cancel it by hand." >&2
    echo "The lease is deliberately kept: dropping it would let a sweep delete" >&2
    echo "the worktree out from under a job that is still alive." >&2
  fi
  exit 6
fi
ABORT_ACTIVE=0; HELD_JOBID=""; trap - EXIT INT TERM   # released: no longer ours to cancel

echo "released ${JOB_NAME} (${JOBID})"
echo "  MEASURE_ROOT ${WT}"
echo "  EXPECT_SHA   ${EXPECT_SHA}"
echo "  PIN_SHA      ${PIN_SHA:-<none: HEAD>}"
echo "  lease        ${WT}/.leases/${JOBID}"
