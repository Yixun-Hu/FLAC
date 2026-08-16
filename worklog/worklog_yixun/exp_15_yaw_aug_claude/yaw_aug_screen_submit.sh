#!/usr/bin/env bash
# ============================================================================
# yaw_aug_screen_submit.sh — submit ONE exp_15 eval cell against a pinned, LEASED
# measurement worktree. Adapted from exp_14's yaw_gen_screen_submit.sh (itself
# exp_11's); the TRANSACTION below is unchanged, only the campaign it submits for
# and the namespace its test seams live in.
#
# For a whole wave use yaw_aug_submit_grid.sh, which enumerates the registered
# 42-cell grid, dedups against landed artifacts and calls THIS script per cell.
#
# ⚠️ This script is the EVAL kit. It has nothing to do with yaw_aug_submit.sh,
# which submits TRAINING legs; the two share a folder and a command log and
# nothing else. The seams here are namespaced YAW_EVAL_* and the entry gate
# REFUSES any inherited YAW_AUG_* / YAW_GEN_* variable outright, so a leftover
# from the training kit or from exp_14's cannot be read as one of ours.
#
# Screens used to be submitted by hand, which is how exp_11's job 3649599 ended
# up reading code that moved underneath it. Submitting through this script makes
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
# U1: the assignment and the special-builtin sweep come BEFORE `set`, so a
# shadowed set() cannot intercept errexit/nounset/pipefail either. An assignment
# involves no command lookup; POSIX mode then makes special builtins outrank
# functions for the sweep that follows.
POSIXLY_CORRECT=1
unset -f unset set export builtin command exec trap readonly eval declare \
         typeset local source env 2>/dev/null || true
set +o posix
unset POSIXLY_CORRECT
set -euo pipefail
# --- FIRST EXECUTABLE STATEMENTS: sanitize the world, then gate it -----------
# THREAT MODEL (inherited from exp_14, Planner 2026-08-11): this kit defends
# against ACCIDENTS and stray environment state — a leftover export, a seam left
# set by a previous run, a test that forgets isolation. It is not hardened
# against an adversary with arbitrary control of the calling shell; the preamble
# below is the cheap part of that boundary, not a claim to have crossed it.
#
# Order is the whole point. Everything after this — the gate, the reader, the
# store-lock re-exec — would otherwise run under whatever PATH and whatever
# exported shell functions the caller supplied.
#
# POSIXLY_CORRECT is an ASSIGNMENT, so it involves no command lookup and cannot
# be shadowed; in bash it turns on POSIX mode, where SPECIAL BUILTINS (set,
# unset, export, eval, ...) are resolved BEFORE functions. The `unset -f` that
# follows is therefore guaranteed to be the real builtin even if an exported
# unset() function was inherited. POSIX mode is dropped again immediately after.
PATH=/usr/bin:/bin:/usr/local/bin; export PATH
unset -f sbatch scontrol scancel squeue sync git grep sed awk head tail tr \
         cat printf id date hostname readlink flock mktemp sha256sum cut wc ls \
         rm mv cp mkdir sleep bash python3 2>/dev/null || true

# ENTRY DOCTRINE, part 1: enumerate the environment NATIVELY. "${!YAW_EVAL_@}"
# expands to the NAMES of set shell variables with that prefix — bash's own
# symbol table, so there is no command here to substitute or shadow. The three
# NEIGHBOURING namespaces are enumerated too (exp_15's training kit, exp_14's
# eval kit, exp_11's store helper): nothing named YAW_AUG_* or YAW_GEN_* is on
# either allowlist, so inheriting one is a hard refusal rather than a value that
# quietly means nothing here.
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

# The MODE is known from the environment alone (no arguments needed), so the gate
# runs HERE — before the manifest reader, before the store-lock re-exec, before
# anything else this script can do.
DRYRUN="${DRYRUN:-0}"
TEST_MODE=0
[ "${YAW_EVAL_TEST_MODE:-0}" = "1" ] && TEST_MODE=1
[ "$DRYRUN" = "1" ] && TEST_MODE=1        # a dry run submits nothing by construction
ALLOW_LIVE="FA_ORBIT_STORE_LOCK_HELD YAW_EVAL_TEST_MODE YAW_EVAL_PINNED_EXEC"
#   YAW_EVAL_PINNED_EXEC  set BY THIS SCRIPT for its own re-exec from the pinned
#                         worktree; its presence means "you are the pinned copy"
#   FA_ORBIT_STORE_LOCK_HELD  "1" from the shared store helper's lock protocol
#   YAW_EVAL_TEST_MODE        "1" selects simulation (0/unset = live)
ALLOW_TEST="FA_ORBIT_STORE_LOCK_HELD YAW_EVAL_TEST_MODE YAW_EVAL_PINNED_EXEC YAW_EVAL_TEST_RECORD \
YAW_EVAL_TEST_JOBID YAW_EVAL_TEST_SUBMIT_SLEEP YAW_EVAL_TEST_RELEASE_SLEEP \
YAW_EVAL_TEST_RELEASE_FAILS YAW_EVAL_TEST_SCANCEL_FAILS YAW_EVAL_SYNC_FAILS \
YAW_EVAL_PIN_FILE YAW_EVAL_INTENT_DIR YAW_EVAL_MAIN_REPO"
#   YAW_EVAL_TEST_RECORD        path of a text file this script APPENDS its argv to
#   YAW_EVAL_TEST_JOBID         the job id a simulated submission reports (data)
#   YAW_EVAL_TEST_SUBMIT_SLEEP  seconds a simulated submission stalls (data)
#   YAW_EVAL_TEST_RELEASE_SLEEP seconds a simulated release stalls (data)
#   YAW_EVAL_TEST_RELEASE_FAILS "1" makes the simulated release fail
#   YAW_EVAL_TEST_SCANCEL_FAILS "1" makes the simulated cancel fail
#   YAW_EVAL_SYNC_FAILS         "1" makes the simulated flush fail
#   YAW_EVAL_MAIN_REPO          test-only root this script reads/writes under
#   YAW_EVAL_PIN_FILE           path of the campaign-pin file to read
#   YAW_EVAL_INTENT_DIR         directory the intent manifest is written to
# The re-exec below runs this script again from the top, so the child re-runs
# this gate on the environment it actually received.
if [ "$TEST_MODE" = "1" ]; then env_allowlist_gate "$ALLOW_TEST"
else                            env_allowlist_gate "$ALLOW_LIVE"; fi

MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
# TEST MODE may relocate the whole tree this script reads and writes. It is a
# DATA path like every other allowlisted seam, and it exists so a test harness is
# isolated BY DEFAULT rather than by remembering to be: a case that forgets
# isolation lands in the temp root, and one that tries it in live mode is refused
# by the allowlist above.
if [ "$TEST_MODE" = "1" ] && [ -n "${YAW_EVAL_MAIN_REPO:-}" ]; then
  MAIN_REPO="$YAW_EVAL_MAIN_REPO"
fi
EXPDIR="$MAIN_REPO/worklog/worklog_yixun/exp_15_yaw_aug_claude"
# --- WHERE THE EXECUTABLE CONTROL PLANE COMES FROM (integrative review F1) -----
# The campaign pin used to bind only the CODE the job read; the submitter itself,
# the validator that rendered identity/contract, and the driver script handed to
# sbatch all came from the MAIN checkout, which moves. A post-pin edit could
# therefore change what executed while the artifact still recorded the pinned
# commit — the announcement-05 mismatch class the pin exists to eliminate.
#
# So: MAIN_REPO stays the production tree (outputs, logs, intents, pin file,
# command log — all of which must NOT be pinned), and CODE_ROOT is the tree the
# executable pieces are read from. A live run re-execs itself out of the pinned
# worktree and sets YAW_EVAL_PINNED_EXEC, at which point CODE_ROOT is that tree.
CODE_ROOT="$MAIN_REPO"
[ -n "${YAW_EVAL_PINNED_EXEC:-}" ] && CODE_ROOT="$YAW_EVAL_PINNED_EXEC"
CODE_EXPDIR="$CODE_ROOT/worklog/worklog_yixun/exp_15_yaw_aug_claude"
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
                # absolute path resolved at entry.
  if [ "${TEST_MODE:-0}" = "1" ]; then
    test_record "sync $1"
    [ "${YAW_EVAL_SYNC_FAILS:-0}" = "1" ] && return 1
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
  # WITHOUT submitting.
  :
else
  # --- PRE-LOCK: a live submission needs the campaign pin FILE ----------------
  # (eval-r1 review finding 2.) Checked HERE, before the store-lock re-exec, for
  # two reasons: a submission that is going to be refused must not take the
  # store-wide lock that every campaign on this machine contends for, and the
  # refusal must not depend on the re-exec succeeding. The child re-runs this
  # same block and then re-checks the pin against the parsed PIN_SHA below.
  #
  # No argument parsing is needed for this: the question is only whether the file
  # exists and holds a commit-shaped sha. TEST MODE is exempt (it executes no
  # submit command at all) and DRYRUN never reaches this branch.
  if [ "$TEST_MODE" != "1" ]; then
    _PRELOCK_PIN_FILE="${EXPDIR}/yaw_aug_screen_campaign_pin"
    _PRELOCK_PIN=""
    [ -f "$_PRELOCK_PIN_FILE" ] \
      && _PRELOCK_PIN="$(head -1 "$_PRELOCK_PIN_FILE" 2>/dev/null | tr -d '[:space:]')"
    case "$_PRELOCK_PIN" in
      ????????????????????????????????????????)
        case "$_PRELOCK_PIN" in *[!0-9a-f]*) _PRELOCK_PIN="" ;; esac ;;
      *) _PRELOCK_PIN="" ;;
    esac
    if [ -z "$_PRELOCK_PIN" ]; then
      echo "refusing to submit a cell with no campaign pin FILE (${_PRELOCK_PIN_FILE})." >&2
      echo "  42 cells are comparable only if they ran at ONE commit, and PIN_SHA on a" >&2
      echo "  command line is not that commitment — set it deliberately, once:" >&2
      echo "  printf '%s\\n' \$(git -C ${MAIN_REPO} rev-parse HEAD) > ${_PRELOCK_PIN_FILE}" >&2
      echo "  (refused BEFORE taking the shared store lock)" >&2
      exit 2
    fi
  fi
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
    ARM)        in_set "$val" YAWAUG VANL \
                  || reject "ARM='${val}' is not a registered exp_15 arm (YAWAUG|VANL)" ;;
    CELL)       in_set "$val" tbl rrob vctl \
                  || reject "CELL='${val}' is not a registered cell type (tbl|rrob|vctl)" ;;
    STEP)       is_num "$val" || reject "STEP='${val}' is not numeric" ;;
    SEED)       is_num "$val" || reject "SEED='${val}' is not numeric" ;;
    K)          in_set "$val" 1 8 || reject "K='${val}' is not 1 or 8" ;;
    ROTATE_DEG) is_decimal "$val" || reject "ROTATE_DEG='${val}' is not a decimal number" ;;
    PIN_SHA)    is_hex40 "$val" || reject "PIN_SHA='${val}' is not 40 hex characters" ;;
    EXCLUDE)    is_nodelist "$val" || reject "EXCLUDE='${val}' is not a comma-separated node list" ;;
    LOG)        # kept to a durable, provenance-preserving location on purpose:
                # exp_11's overnight incidents were a tee into the pinned worktree
                # and node-local /tmp paths, both of which this shape excludes
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
# exp_15's eval campaign keeps its OWN pin file rather than the store-wide
# .campaign_pin marker: the store is shared with other campaigns (exp_11's
# screens, exp_14's), one marker cannot mean two commits at once, and a campaign
# that read someone else's pin would measure half its grid somewhere else.
# TEST MODE is a MODE, not a mock. Deciding "is this executable a mock?" is not
# decidable — a wrapper, a copy, a hard link or an absolute path to the real
# sbatch all differ from the string "sbatch" — so nothing is asked about the
# executable: in test mode this script EXECUTES NO SUBMIT COMMAND AT ALL. It
# simulates the Slurm interaction internally and records the argv it would have
# run. A test therefore cannot reach sbatch even by accident, which is the only
# property worth having (exp_14's earlier "mock" rule let a guard run submit four
# real jobs).
if [ "$TEST_MODE" != "1" ]; then
  # Resolve every Slurm binary ONCE, absolutely, on the sanitized PATH set at
  # entry and with any exported shell function of the same name already dropped.
  # Only these stored paths are invoked afterwards.
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
PIN_FILE="${EXPDIR}/yaw_aug_screen_campaign_pin"
INTENT_DIR="$EXPDIR"

# --- the cell's IDENTITY and PROTOCOL CONTRACT, from the ONE definition ------
# Announcement 05: the eval-protocol flags are part of the experiment, and the
# pre-launch record must be readable for protocol compliance on its own — the
# post-run screenmeta cannot repair it, being written by the job that already
# ran. Rendered by exp15_validate_cell so the intent, the driver's argv, the job
# name and the admissibility rules are ONE definition rather than four that agree
# today. exp_14 rendered the job name in shell and pinned it with a guard case;
# exp15_validate_cell is torch-free, so there is no reason to keep a second copy.
VALIDATOR="$CODE_EXPDIR/exp15_validate_cell.py"
render_cell_identity() {
  local args=(--arm "$ARM" --cell "$CELL" --step "$STEP" --seed "$SEED" --k "$K")
  [ "$CELL" = "vctl" ] && args+=(--rotate-deg "$ROTATE_DEG")
  JOB_NAME="$(python3 "$VALIDATOR" jobname "${args[@]}")" || {
    echo "could not render the cell's job name - nothing submitted" >&2
    return 1
  }
  CELL_CONTRACT="$(python3 "$VALIDATOR" contract "${args[@]}")" || {
    echo "could not render the cell's protocol contract - nothing submitted" >&2
    return 1
  }
}
if [ "$TEST_MODE" = "1" ]; then
  [ -n "${YAW_EVAL_PIN_FILE:-}" ] && PIN_FILE="$YAW_EVAL_PIN_FILE"
  [ -n "${YAW_EVAL_INTENT_DIR:-}" ] && INTENT_DIR="$YAW_EVAL_INTENT_DIR"
elif [ -n "${YAW_EVAL_PIN_FILE:-}${YAW_EVAL_INTENT_DIR:-}" ]; then
  reject "YAW_EVAL_PIN_FILE / YAW_EVAL_INTENT_DIR are test seams: they need YAW_EVAL_TEST_MODE=1, never a run that could really submit"
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
    echo "A campaign measures every cell at ONE commit; submitting one cell elsewhere" >&2
    echo "would silently make it incomparable. To change the pin deliberately:" >&2
    echo "  printf '%s\\n' ${PIN_SHA} > ${PIN_FILE}" >&2
    exit 2
  fi
fi
if [ -n "$PIN_SHA" ]; then
  git -C "$MAIN_REPO" rev-parse --verify --quiet "${PIN_SHA}^{commit}" >/dev/null \
    || reject "PIN_SHA ${PIN_SHA} is not a commit in this repository"
fi
# --- a LIVE single submission REQUIRES the pin file, exactly as a wave does ----
# (eval-r1 review finding 2.) This path used to fall back to whatever HEAD was if
# the file was absent, so the planned single V/probe launches — the FIRST cells
# of the campaign — could have been pinned somewhere the other 41 were not, and
# the DRYRUN said so only as an easily-read-past `<none: HEAD>`.
#
# PIN_SHA on the command line is an ASSERTION about the file's content, never a
# substitute for it: a campaign measures every cell at ONE commit, and a commit
# named in an argument is a claim nothing durable records. The mismatch branch
# above already refuses disagreement; this makes the file mandatory.
#
# DRYRUN and TEST MODE are exempt because neither can submit: a dry run prepares
# no worktree and takes no lock, and test mode executes no submit command at all.
if [ "$DRYRUN" != "1" ] && [ "$TEST_MODE" != "1" ] && [ -z "$CAMPAIGN_PIN" ]; then
  echo "refusing to submit a cell with no campaign pin FILE (${PIN_FILE})." >&2
  echo "  42 cells are comparable only if they ran at ONE commit, and PIN_SHA on a" >&2
  echo "  command line is not that commitment — set it deliberately, once:" >&2
  echo "  printf '%s\\n' \$(git -C ${MAIN_REPO} rev-parse HEAD) > ${PIN_FILE}" >&2
  exit 2
fi
[ -n "$ARM" ] && [ -n "$CELL" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=YAWAUG CELL=tbl [STEP=40000] [SEED=42] [K=8] [EXCLUDE=node[,node]] [ROTATE_DEG=90] [PIN_SHA=<40hex>] [LOG=...]" >&2; exit 2; }
# CELL is REQUIRED and has no default here or in the driver: a cell type is a
# protocol choice (announcement 05), and defaulting one would let an omitted
# argument decide the experiment.
render_cell_identity || exit 2

# --- DRYRUN: print what would be submitted, touch nothing --------------------
if [ "$DRYRUN" = "1" ]; then
  CELL_EXPORT_DRY=""
  [ -n "$ROTATE_DEG" ] && CELL_EXPORT_DRY="${CELL_EXPORT_DRY},ROTATE_DEG=${ROTATE_DEG}"
  [ -n "$LOG" ] && CELL_EXPORT_DRY="${CELL_EXPORT_DRY},LOG=${LOG}"
  echo "DRYRUN job-name ${JOB_NAME}"
  if [ -n "$PIN_SHA" ]; then
    echo "DRYRUN pin ${PIN_SHA}"
  else
    # Not a green light: say plainly that the live path would refuse this.
    echo "DRYRUN pin <none: no campaign pin file at ${PIN_FILE}>"
    echo "DRYRUN NOTE a LIVE submission would REFUSE here — the pin file is required"
  fi
  echo "DRYRUN exclude ${EXCLUDE:-<none>}"
  echo "DRYRUN export ARM=${ARM},STEP=${STEP},SEED=${SEED},K=${K},CELL=${CELL}${CELL_EXPORT_DRY}"
  echo "DRYRUN driver ${CODE_EXPDIR}/yaw_aug_screen.sbatch"
  if [ -n "${YAW_EVAL_PINNED_EXEC:-}" ]; then
    echo "DRYRUN control-plane PINNED (${CODE_ROOT})"
  else
    echo "DRYRUN control-plane MAIN-TREE (unpinned): this dry run reads the moving"
    echo "DRYRUN   checkout. A LIVE submission re-execs from the pinned worktree first."
  fi
  # the PROTOCOL CONTRACT the intent manifest will carry, shown here so a dry run
  # is a full protocol review (announcement 05) and not just an identity check
  printf '%s\n' "$CELL_CONTRACT" | while IFS= read -r line; do
    echo "DRYRUN intent ${line}"
  done
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

# --- RE-EXEC FROM THE PINNED WORKTREE (integrative review F1) ------------------
# Everything after this point — identity rendering, the contract in the intent
# manifest, and the driver script handed to sbatch — must come from the pinned
# tree, not from the checkout this process was started in. The child inherits fd
# 8 and FA_ORBIT_STORE_LOCK_HELD, so it re-enters the same store lock rather than
# deadlocking on a second one, and it re-runs every gate above on the environment
# it actually receives.
if [ -z "${YAW_EVAL_PINNED_EXEC:-}" ]; then
  PINNED_SELF="${WT}/worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_screen_submit.sh"
  [ -f "$PINNED_SELF" ] \
    || { echo "the pinned worktree has no submitter at ${PINNED_SELF} - abort" >&2; exit 3; }
  echo "re-exec from the PINNED control plane: ${PINNED_SELF}"
  YAW_EVAL_PINNED_EXEC="$WT"; export YAW_EVAL_PINNED_EXEC
  exec bash "$PINNED_SELF" "$@"
fi
echo "control plane: PINNED (${CODE_ROOT})"

# --- VALIDATE-BEFORE-SKIP + IN-FLIGHT GUARD + COMMAND LOG (review F6) ---------
# The wave path has had all three since round 1; this path — the one the runbook
# uses for the first V and probe launches — had none, so it could re-run a landed
# cell, duplicate a queued one, or submit with no entry in yaw_aug_command.md
# (plan §6.7 requires EVERY submission to be appended). Same predicates, one cell.
COMMAND_LOG="${EXPDIR}/yaw_aug_command.md"
[ "$TEST_MODE" = "1" ] && [ -n "${YAW_EVAL_COMMAND_LOG:-}" ] && COMMAND_LOG="$YAW_EVAL_COMMAND_LOG"
CELL_STATUS_ARGS=(cellstatus --arm "$ARM" --cell "$CELL" --step "$STEP"
                  --seed "$SEED" --k "$K" --output-root "${MAIN_REPO}/outputs_FLAC"
                  --pin "$EXPECT_SHA")
[ "$CELL" = "vctl" ] && CELL_STATUS_ARGS+=(--rotate-deg "$ROTATE_DEG")
CELL_STATUS="$(python3 "$VALIDATOR" "${CELL_STATUS_ARGS[@]}" 2>&1)"; STATUS_RC=$?
case "$STATUS_RC" in
  0) echo "SKIP ${JOB_NAME}: already measured and VALID at this pin"
     echo "  ${CELL_STATUS}"
     exit 0 ;;
  3) echo "cell not yet measured: ${CELL_STATUS}" ;;
  1) echo "HALT: this cell's artifacts EXIST but do not validate." >&2
     echo "  ${CELL_STATUS}" >&2
     echo "  An artifact that exists but fails validation is a triage question: it" >&2
     echo "  must not be skipped (it would be read as evidence) and must not be" >&2
     echo "  overwritten (that destroys the evidence of what went wrong)." >&2
     exit 3 ;;
  *) echo "HALT: could not classify this cell - refusing to submit blind" >&2
     echo "  ${CELL_STATUS}" >&2; exit 3 ;;
esac
# ...and it must not already be queued. A name alone is a claim; the lease is the
# evidence (the wave path's rule, applied here).
if [ "$TEST_MODE" != "1" ]; then
  INFLIGHT_JID="$("$BIN_squeue" -h -u "$(id -un)" -o "%i %j" 2>/dev/null \
                  | awk -v n="$JOB_NAME" '$2 == n {print $1; exit}')" || INFLIGHT_JID=""
  if [ -n "$INFLIGHT_JID" ]; then
    if grep -q "^jobid ${INFLIGHT_JID}$" "${WT}/.leases/${INFLIGHT_JID}" 2>/dev/null; then
      echo "SKIP ${JOB_NAME}: job ${INFLIGHT_JID} is in flight and holds its lease"
      exit 0
    fi
    echo "HALT: ${JOB_NAME} is queued as ${INFLIGHT_JID} but holds NO lease under" >&2
    echo "  ${WT}/.leases/${INFLIGHT_JID} — skipping would wait for a cell that may" >&2
    echo "  never land, submitting would duplicate it. Triage by hand." >&2
    exit 5
  fi
fi
log_command() {   # <state> <jid> <rc> — durable, and BEFORE the submission
  printf -- '- `%s` **%s** %s jid `%s` rc %s — `bash %s ARM=%s CELL=%s STEP=%s SEED=%s K=%s%s`\n' \
    "$(date -Is)" "$JOB_NAME" "$1" "${2:-<none>}" "${3:-<pending>}" "$0" \
    "$ARM" "$CELL" "$STEP" "$SEED" "$K" \
    "$([ -n "$ROTATE_DEG" ] && printf ' ROTATE_DEG=%s' "$ROTATE_DEG")" \
    >> "$COMMAND_LOG" && sync_file "$COMMAND_LOG"
}
log_command LAUNCHING "" "" \
  || { echo "HALT: cannot durably record the command in ${COMMAND_LOG}; refusing to" >&2
       echo "  submit a job that would have no launch record." >&2; exit 6; }

# 2. submit HELD: the id exists before the lease, the job runs after it
# JOB_NAME was rendered above by exp15_validate_cell.job_name. It starts with
# exp15- so the wave submitter can count THIS campaign's jobs in squeue without
# catching a neighbour's, and it carries the ROTATION TOKEN so two cells that
# differ only in their rotation cannot read as one in-flight job.
# --- the ONLY three places this script talks to Slurm ------------------------
# In TEST MODE each one records what it would have done and returns a scripted
# result; no process is started. In LIVE mode each runs the canonical binary.
test_record() { [ -n "${YAW_EVAL_TEST_RECORD:-}" ] && printf '%s\n' "$*" >> "$YAW_EVAL_TEST_RECORD"; return 0; }

slurm_submit_hold() {   # prints a job id on stdout, like `sbatch --parsable`
  if [ "$TEST_MODE" = "1" ]; then
    test_record "sbatch $*"
    [ -n "${YAW_EVAL_TEST_SUBMIT_SLEEP:-}" ] \
      && { : > "${YAW_EVAL_TEST_RECORD:-/dev/null}.submitting"; sleep "$YAW_EVAL_TEST_SUBMIT_SLEEP"; }
    case "$*" in *--hold*) ;; *) echo "TEST: submit called WITHOUT --hold" >&2; return 9 ;; esac
    printf '%s\n' "${YAW_EVAL_TEST_JOBID:-7654321}"
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
    [ -n "${YAW_EVAL_TEST_RELEASE_SLEEP:-}" ] \
      && { : > "${YAW_EVAL_TEST_RECORD:-/dev/null}.releasing"; sleep "$YAW_EVAL_TEST_RELEASE_SLEEP"; }
    [ "${YAW_EVAL_TEST_RELEASE_FAILS:-0}" = "1" ] && return 1
    return 0
  fi
  "$BIN_scontrol" release "$1"
}
slurm_cancel() {        # $1 = job id, or --name=<job name>
  if [ "$TEST_MODE" = "1" ]; then
    test_record "scancel $1"
    [ "${YAW_EVAL_TEST_SCANCEL_FAILS:-0}" = "1" ] && return 1
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
# before submission and is cell-injective, so `scancel --name=` is the fallback.
# Cancelling a name that was never submitted is harmless; leaving a held job that
# nobody knows about is not.
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
  "$CODE_EXPDIR/yaw_aug_screen.sbatch")" || { echo "sbatch FAILED - nothing submitted" >&2; exit 4; }
JOBID="${JOBID%%;*}"
case "$JOBID" in
  ''|*[!0-9]*)
    # The submission may well have SUCCEEDED and only its output be unreadable,
    # so this exits through the same guard — by name, since there is no id.
    echo "sbatch returned '${JOBID}', not a job id - abort" >&2; exit 4 ;;
esac
HELD_JOBID="$JOBID"
echo "submitted HELD as ${JOBID}"
log_command SUBMITTED "$JOBID" "0" \
  || echo "WARNING: could not append the outcome to ${COMMAND_LOG}" >&2

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

# --- intent manifest: written while the job is STILL HELD ---------------------
# Writing it after the release produced a window in which a running job had no
# launch record, and a failure to write it was only a warning — so a crash could
# leave an undocumented job reading a pinned worktree. The manifest is therefore
# published (atomically, tmp+rename in the same directory) BEFORE the release,
# and a failure to publish it CANCELS the held job: a job with no launch record
# is worse than no job.
#
# The name carries `screen` so an eval intent can never be confused with the
# TRAINING kit's yaw_aug_submission_*.txt in the same folder.
INTENT="${INTENT_DIR}/yaw_aug_screen_submission_${ARM}_${CELL}_S${STEP}_s${SEED}_K${K}_jid${JOBID}.txt"
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
    printf '%s\n' "$CELL_CONTRACT"
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
