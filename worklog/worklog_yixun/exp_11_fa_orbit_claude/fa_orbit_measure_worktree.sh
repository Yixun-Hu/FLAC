#!/usr/bin/env bash
# ============================================================================
# fa_orbit_measure_worktree.sh — prepare a pinned, LEASED code tree for
# measurement jobs, provision the untracked runtime assets, and print its path.
#
# WHY. A measurement job reads code from the checkout it runs in while
# development keeps committing to that same checkout; jobs 3649599/3649600 were
# refused after 13 good minutes because source_sha moved mid-run. Measurements
# therefore run from a git worktree pinned to the submission SHA under
# .measure_worktrees/<sha>/, while outputs go to the MAIN tree. Development and
# measurement are decoupled — no commit freezes.
#
# TWO THINGS THIS SCRIPT MUST GET RIGHT, both found by review:
#
# 1. ASSETS. A fresh worktree contains only TRACKED files. `AcousticRooms` (the
#    dataset symlink, untracked) and `weights/` (AGREE + VAE, gitignored) are
#    both resolved RELATIVELY by the eval dataset configs and the arm configs'
#    AGREE_ckpt, so without them a pinned evaluation crashes on startup. They are
#    linked in here, pointing at exactly what the main tree points at.
#
# 2. LIFECYCLE. Fixed-count pruning could delete the tree of a QUEUED job (submit
#    A, then three more submissions, and A's tree is gone before it starts).
#    Trees are now LEASED: a submission submits HELD, writes .leases/<jobid> for
#    the id Slurm just gave it, then releases the job — so a queued job always
#    has a lease and a lease always names a real job. The job removes its lease
#    on exit; a tree is prunable only when it holds no lease AND none of its
#    lease job ids is still known to Slurm (a squeue that FAILS keeps the tree).
#    Every operation — create, lease, release, prune — takes one store-wide
#    flock, so a sweep can never interleave with a submission.
#
#   MEASURE_ROOT="$(bash .../fa_orbit_measure_worktree.sh)"          # pin HEAD
#   bash .../fa_orbit_measure_worktree.sh --lease <jobid> <root>     # add a lease
#     (submissions go through fa_orbit_screen_submit.sh, which does the held-job
#      dance; call --lease by hand only for an already-known job id)
#   bash .../fa_orbit_measure_worktree.sh --release <jobid> <root>   # drop a lease
#   bash .../fa_orbit_measure_worktree.sh --prune                    # sweep
#   bash .../fa_orbit_measure_worktree.sh --freeze ["reason"]        # campaign:
#   bash .../fa_orbit_measure_worktree.sh --thaw                     #  no deletes
#   bash .../fa_orbit_measure_worktree.sh --frozen                   # rc 0=frozen
#   bash .../fa_orbit_measure_worktree.sh --pin-campaign <sha>       # campaign
#   bash .../fa_orbit_measure_worktree.sh --unpin-campaign           #  commit
#   bash .../fa_orbit_measure_worktree.sh --pinned                   # rc 0=pinned
# ============================================================================
set -uo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
ROOT="${MAIN_REPO}/.measure_worktrees"
# Untracked runtime inputs the eval resolves RELATIVE to the code root, each
# pinned to the target it MUST resolve to. The main tree's symlink is mutable
# (anyone can repoint it at a different corpus between screens), so following it
# blindly would let the measurement stack silently change datasets. Registered
# target, or abort.
# LEAF paths, not top-level directories: `weights/` became a TRACKED directory
# when the exp_01 metric JSONs were force-added (commit 7543476), so a fresh
# worktree materialises a real weights/FLAC/ holding those JSONs and nothing
# else. Linking the parent is then impossible and linking nothing leaves the
# evaluation without its conditioner. Each leaf the eval actually opens is
# provisioned on its own, whatever its parent happens to be.
# VAE.ckpt is here because the AGREE config chain loads it: it was the packaging
# gap behind the halt — the worktree had VAE.safetensors but not VAE.ckpt, so a
# pinned evaluation died on a file the main tree has had all along. No semantic
# change on any surface; the tree was simply missing an input.
ASSETS=(AcousticRooms weights/AGREE weights/FLAC/VAE.safetensors weights/FLAC/VAE.ckpt)
ASSET_TARGETS=(/n/fs/gatrdp/datasets/AcousticRooms
               /n/fs/gatrdp/codespace/FLAC/weights/AGREE
               /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors
               /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.ckpt)
# Bookkeeping this script itself creates in the tree; never code.
BOOKKEEPING=(.leases)

die() { echo "$1" >&2; exit "${2:-2}"; }

# --- CAMPAIGN FREEZE ---------------------------------------------------------
# While a measurement campaign is in flight, no worktree may be deleted by
# anything, for any reason. Leases already protect trees whose jobs Slurm knows
# about, but that is a proof about job state; a campaign wants a proof about
# THIS SCRIPT: while the marker exists, every deletion path here is inert.
#
# Reaping lease FILES continues (a lease whose job provably never existed is
# garbage either way, and removing it cannot lose a worktree). Preparation
# continues too — trees are still created and reused. Only deletion stops.
FREEZE_MARKER="${ROOT}/.campaign_freeze"
frozen() { [ -e "$FREEZE_MARKER" ]; }

# --- CAMPAIGN PIN ------------------------------------------------------------
# The commit every measurement of this campaign is taken at. Arms measured at
# different code SHAs are not comparable, and "remember to pass PIN_SHA" is not a
# mechanism — so the pin lives in a file, the submitter defaults to it, and a
# submission that names a DIFFERENT commit is refused rather than quietly
# honoured. Changing the pin is therefore an explicit, logged act.
PIN_MARKER="${ROOT}/.campaign_pin"
campaign_pin() { [ -f "$PIN_MARKER" ] && head -1 "$PIN_MARKER" | tr -d '[:space:]'; }

freeze_note() {   # $1 = what was being attempted
  echo "  CAMPAIGN FREEZE (${FREEZE_MARKER}): refusing to ${1}" >&2
}

lease_dir() { echo "${1}/.leases"; }

# --- worktree identity -------------------------------------------------------
# A DIRECTORY IS NOT A WORKTREE. `git worktree remove` can unregister a tree and
# leave its directory behind (our own symlinks and .leases/ defeat its cleanup),
# and because that leftover sits INSIDE the main checkout, `git -C` on it walks
# UP and answers for the MAIN repo — HEAD matches, so a directory containing no
# code at all looks reusable. Verify identity, never presence.
#
# The verdict is THREE-valued on purpose. "git could not answer" is not "this is
# not a worktree": on a shared NFS filesystem a transient failure is ordinary,
# and treating it as invalid would license deleting a live measurement tree.
worktree_identity() {   # $1 = directory -> echoes valid | invalid | indeterminate
  local top rc common main_common
  if [ ! -e "$1/.git" ]; then echo invalid; return; fi   # definitive: no gitlink
  top="$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)"; rc=$?
  if [ "$rc" -ne 0 ]; then echo indeterminate; return; fi
  if [ "$(readlink -f "$top")" != "$(readlink -f "$1")" ]; then echo invalid; return; fi
  common="$(git -C "$1" rev-parse --git-common-dir 2>/dev/null)" || { echo indeterminate; return; }
  main_common="$(git -C "$MAIN_REPO" rev-parse --git-common-dir 2>/dev/null)" \
    || { echo indeterminate; return; }
  if [ "$(readlink -f "$common")" != "$(readlink -f "$main_common")" ]; then echo invalid; return; fi
  echo valid
}

is_live_worktree() { [ "$(worktree_identity "$1")" = "valid" ]; }

# Exactly 40 hex characters — the only shape this store ever creates, and the
# only shape anything here is ever allowed to delete.
is_sha_name() {
  case "$1" in
    *[!0-9a-f]*) return 1 ;;
    ????????????????????????????????????????) return 0 ;;
    *) return 1 ;;
  esac
}

# --- lease helpers -----------------------------------------------------------
add_lease() {   # $1 = job id, $2 = worktree
  # A lease on something that is not a live registered worktree is a lease on
  # nothing: the caller would go on to run a job against a directory that holds
  # no code. Refuse, and let the caller cancel.
  local verdict; verdict="$(worktree_identity "$2")"
  [ "$verdict" = "valid" ] \
    || die "refusing to lease ${2}: worktree identity is ${verdict}, not a live registered worktree" 4
  mkdir -p "$(lease_dir "$2")" || die "cannot create the lease directory in $2" 3
  printf 'jobid %s\nheld_since %s\nhost %s\n' "$1" "$(date -Is)" "$(hostname)" \
    > "$(lease_dir "$2")/$1" || die "cannot write lease $1 in $2" 3
  echo "$(lease_dir "$2")/$1"
}

release_lease() { rm -f "$(lease_dir "$2")/$1"; }

lease_is_live() {  # $1 = lease job id; 0 = live/unknown (KEEP), 1 = provably gone
  local out rc
  # Anything that is not a job id is not a question Slurm can answer, and an
  # unanswerable question must never be resolved as "collectable".
  case "$1" in ''|*[!0-9]*) return 0 ;; esac
  out="$(squeue -h -j "$1" -o %i 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ]; then
    case "$out" in *"$1"*) return 0 ;; *) return 1 ;; esac   # listed vs finished
  fi
  # A nonzero squeue is NOT evidence that the job is gone. Only Slurm's explicit
  # "that id does not exist" answer proves absence; anything else (controller
  # down, socket timeout, auth) must fail SAFE — keep the lease, keep the tree.
  case "$out" in
    *"Invalid job id"*) return 1 ;;
    *) echo "  squeue failed while checking lease $1 (rc=${rc}: ${out}) — treating the lease as LIVE" >&2
       return 0 ;;
  esac
}

# PURE predicate: does any lease name a job that is live — or that we could not
# prove is gone? No side effects, so it is safe to ask before deciding to delete.
has_live_lease() {  # $1 = worktree; 0 = yes (KEEP), 1 = every lease provably gone
  local d f; d="$(lease_dir "$1")"
  [ -d "$d" ] || return 1
  for f in "$d"/*; do
    [ -e "$f" ] || continue
    lease_is_live "$(basename "$f")" && return 0
  done
  return 1
}

prunable() {  # $1 = worktree; prunable iff no lease names a live job (and reaps)
  # Reap every PROVABLY dead lease as we go — a lease whose job Slurm says never
  # existed is garbage whether or not its neighbours are live — then decide on
  # what is left. (Deciding first and reaping second made reaping depend on
  # which lease the loop happened to see first.)
  local d f live=0; d="$(lease_dir "$1")"
  if [ -d "$d" ]; then
    for f in "$d"/*; do
      [ -e "$f" ] || continue
      if lease_is_live "$(basename "$f")"; then
        live=1
      else
        echo "  stale lease $(basename "$f") in $1 (Slurm reports no such job)" >&2
        rm -f "$f"
      fi
    done
  fi
  [ "$live" -eq 0 ]
}

# The ONLY code path that deletes anything. Every caller must already have
# established that the entry is unleased; this enforces the name shape and the
# "a failure to remove is a warning, never an escalation" rule.
remove_entry() {   # $1 = store entry, already proven unleased
  local verdict
  # The freeze is checked HERE, in the one function that deletes, so no caller
  # can route around it — and again at each call site, so the log says why.
  if frozen; then freeze_note "remove ${1}"; return 1; fi
  is_sha_name "$(basename "$1")" \
    || { echo "  (refusing to remove ${1}: not a 40-hex store entry)" >&2; return 1; }
  verdict="$(worktree_identity "$1")"
  # our own symlinks and .leases/ make `git worktree remove` bail out half-done,
  # so clear them first, then let git remove what git registered
  rm -rf "${1}/.leases" "${1}/AcousticRooms" "${1}/weights" 2>/dev/null
  if ! git -C "$MAIN_REPO" worktree remove --force "$1" >&2 2>/dev/null; then
    # For a REGISTERED worktree git is the authority. If it declines, forcing an
    # rm -rf would delete the checkout while leaving git's administrative files
    # pointing at it — a corrupt store, from a sweep that was only tidying up.
    # Leave it; the next sweep tries again, and preparation re-provisions it.
    if [ "$verdict" = "valid" ]; then
      echo "  (git declined to remove the registered worktree ${1}; leaving it in place)" >&2
      return 1
    fi
  fi
  # unregistered leftovers (git no longer knows them) are ours to clear
  if [ -d "$1" ]; then
    rm -rf "$1" 2>/dev/null || { echo "  (could not remove ${1}; leaving it for inspection)" >&2; return 1; }
  fi
  return 0
}

prune_all() {  # never touches $KEEP_WT
  local keep="${1:-}"
  local d ID
  if frozen; then
    echo "campaign freeze active: no worktree will be removed by this sweep " \
         "(stale lease FILES are still reaped)" >&2
  fi
  for d in "$ROOT"/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    [ "$d" = "$keep" ] && continue
    ID="$(worktree_identity "$d")"
    if [ "$ID" = "indeterminate" ]; then
      echo "skipping ${d}: identity indeterminate (git/filesystem error), never deleting on a maybe" >&2
      continue
    fi
    if prunable "$d"; then
      if frozen; then
        echo "  freeze: keeping unleased worktree ${d} (would have been pruned)" >&2
        continue
      fi
      echo "pruning unleased measurement worktree ${d}" >&2
      remove_entry "$d"
    else
      echo "keeping leased worktree ${d}" >&2
    fi
  done
  # `git worktree prune` only drops administrative records, but it is still a
  # pruning operation and the campaign guarantee is "nothing prunes".
  if frozen; then
    freeze_note "run 'git worktree prune'"
  else
    git -C "$MAIN_REPO" worktree prune >&2 2>/dev/null || true
  fi
}

cd "$MAIN_REPO" || die "cannot cd ${MAIN_REPO}" 3
mkdir -p "$ROOT" || die "cannot create ${ROOT}" 3

# ONE lock for the whole store, held by EVERY operation — creation, lease,
# release and prune alike.
#
# LOCK SPAN. Locking each operation individually is not enough: a submission
# prepares a tree, releases the lock, then leases it, and a prune sweep that runs
# in that gap sees a brand-new, unleased tree and deletes it. `--with-lock CMD`
# therefore holds the lock across the WHOLE prepare -> submit -> lease sequence.
# The lock lives on fd 8, which children inherit, so nested calls from inside CMD
# re-enter the SAME open file description instead of deadlocking on a second one.
LOCK="${ROOT}/.store.lock"

lock_already_held() {
  local have want
  [ "${FA_ORBIT_STORE_LOCK_HELD:-0}" = "1" ] || return 1
  # Trust the marker only if fd 8 really is this lock: a stray environment
  # variable must never be able to turn locking off. An unresolvable path gives
  # the EMPTY string, and two empty strings compare equal — so a double readlink
  # failure would read as a match. Require both to resolve.
  [ -e /proc/self/fd/8 ] || return 1
  have="$(readlink -f /proc/self/fd/8 2>/dev/null)" || return 1
  want="$(readlink -f "$LOCK" 2>/dev/null)" || return 1
  [ -n "$have" ] && [ -n "$want" ] && [ "$have" = "$want" ]
}

take_store_lock() {
  lock_already_held && return 0
  exec 8>"$LOCK" || die "cannot open the store lock ${LOCK}" 3
  flock 8 || die "cannot take the store lock ${LOCK}" 3
  export FA_ORBIT_STORE_LOCK_HELD=1
}

if [ "${1:-}" = "--with-lock" ]; then
  shift
  [ "$#" -gt 0 ] || die "--with-lock needs a command to run" 3
  take_store_lock
  exec "$@"          # fd 8 (and therefore the lock) is inherited for its lifetime
fi

take_store_lock

case "${1:-}" in
  --freeze)
    # under the store lock, like every other store operation
    if frozen; then
      echo "campaign freeze already active since $(head -1 "$FREEZE_MARKER" 2>/dev/null)" >&2
    else
      printf 'frozen_at %s\nby %s@%s\nreason %s\n' "$(date -Is)" "$(id -un)" "$(hostname)" \
        "${2:-exp_11 measurement campaign}" > "$FREEZE_MARKER" \
        || die "cannot write the freeze marker ${FREEZE_MARKER}" 3
      echo "CAMPAIGN FREEZE ENGAGED at $(date -Is): no worktree in ${ROOT} can be deleted" >&2
    fi
    exit 0 ;;
  --thaw)
    if frozen; then
      echo "campaign freeze LIFTED at $(date -Is) (was: $(head -1 "$FREEZE_MARKER" 2>/dev/null))" >&2
      rm -f "$FREEZE_MARKER" || die "cannot remove the freeze marker ${FREEZE_MARKER}" 3
    else
      echo "no campaign freeze was active" >&2
    fi
    exit 0 ;;
  --frozen)  frozen && { echo "frozen"; exit 0; }; echo "thawed"; exit 1 ;;
  --pin-campaign)
    NEWPIN="${2:?a 40-hex commit sha}"
    case "$NEWPIN" in
      *[!0-9a-f]*) die "campaign pin '${NEWPIN}' is not 40 hex characters" ;;
      ????????????????????????????????????????) ;;
      *) die "campaign pin '${NEWPIN}' is not 40 hex characters" ;;
    esac
    git rev-parse --verify --quiet "${NEWPIN}^{commit}" >/dev/null \
      || die "campaign pin ${NEWPIN} is not a commit in this repository"
    OLDPIN="$(campaign_pin)"
    # Writing the pin is not a deletion, so it is allowed under a freeze.
    printf '%s\n' "$NEWPIN" > "$PIN_MARKER" || die "cannot write ${PIN_MARKER}" 3
    printf '# set %s by %s@%s (previous: %s)\n' "$(date -Is)" "$(id -un)" "$(hostname)" \
      "${OLDPIN:-<none>}" >> "$PIN_MARKER"
    echo "CAMPAIGN PIN set to ${NEWPIN} at $(date -Is) (previous: ${OLDPIN:-<none>})" >&2
    exit 0 ;;
  --unpin-campaign)
    if [ -f "$PIN_MARKER" ]; then
      echo "CAMPAIGN PIN cleared at $(date -Is) (was: $(campaign_pin))" >&2
      rm -f "$PIN_MARKER" || die "cannot remove ${PIN_MARKER}" 3
    else
      echo "no campaign pin was set" >&2
    fi
    exit 0 ;;
  --pinned)  P="$(campaign_pin)"; [ -n "$P" ] && { echo "$P"; exit 0; }; echo "<none>"; exit 1 ;;
  --lease)   add_lease "${2:?job id}" "${3:?worktree}" >/dev/null; exit 0 ;;
  --release) release_lease "${2:?job id}" "${3:?worktree}"; exit 0 ;;
  --prune)   prune_all ""; exit 0 ;;
esac

SHA="${1:-$(git rev-parse HEAD)}"
git rev-parse --verify --quiet "${SHA}^{commit}" >/dev/null || die "not a commit: ${SHA}"
SHA="$(git rev-parse "${SHA}^{commit}")"
WT="${ROOT}/${SHA}"

# creation runs under the store lock taken above: concurrent submissions for one
# SHA are idempotent, and no sweep can observe a half-built tree.
#
# A failed identity check is NOT a licence to delete. An entry is cleared only if
# it is BOTH definitively invalid AND provably unleased; an indeterminate verdict
# (git or NFS having a bad day) or any lease we cannot prove is dead leaves the
# directory exactly where it is, with a warning.
if [ -d "$WT" ]; then
  VERDICT="$(worktree_identity "$WT")"
  case "$VERDICT" in
    valid) ;;
    indeterminate)
      die "cannot establish the identity of ${WT} (git/filesystem error) - refusing to touch it" 3 ;;
    invalid)
      if frozen; then
        die "${WT} is a half-removed worktree, but a CAMPAIGN FREEZE is active - thaw (--thaw) and clean it manually, then re-freeze" 3
      fi
      if has_live_lease "$WT"; then
        die "${WT} is not a valid worktree but still holds a lease that is live or unverifiable - refusing to clear it" 3
      fi
      is_sha_name "$(basename "$WT")" || die "refusing to clear ${WT}: not a 40-hex store entry" 3
      echo "clearing a half-removed, unleased worktree directory at ${WT}" >&2
      git worktree prune >&2 2>/dev/null || true
      remove_entry "$WT" || die "cannot clear the stale directory ${WT}" 3 ;;
  esac
fi

if [ -d "$WT" ]; then
  HAVE="$(git -C "$WT" rev-parse HEAD 2>/dev/null)" || die "existing ${WT} is not a git worktree"
  [ "$HAVE" = "$SHA" ] || die "existing ${WT} is at ${HAVE}, not ${SHA} - refusing to reuse"
  echo "reusing pinned worktree ${WT}" >&2
else
  if ! git worktree add --detach "$WT" "$SHA" >&2; then
    # "missing but already registered" — a registration whose directory is gone.
    # Re-creating it destroys nothing (there is nothing there), so --force is
    # safe here even under a campaign freeze, which forbids DELETING trees, not
    # creating them. `git worktree prune` would also fix it, but that is a
    # pruning operation and the freeze suppresses those on purpose.
    if [ ! -e "$WT" ]; then
      echo "clearing a stale registration for the missing ${WT} and re-creating it" >&2
      git worktree add --detach --force "$WT" "$SHA" >&2 || die "git worktree add failed" 3
    else
      die "git worktree add failed" 3
    fi
  fi
  echo "created pinned worktree ${WT}" >&2
fi

# --- provision the untracked runtime assets (the crasher) --------------------
for i in "${!ASSETS[@]}"; do
  A="${ASSETS[$i]}"; WANT="${ASSET_TARGETS[$i]}"
  SRC="$(readlink -f "${MAIN_REPO}/${A}" 2>/dev/null)"
  [ -n "$SRC" ] && [ -e "$SRC" ] || die "main-tree asset '${A}' does not resolve (${SRC:-<unset>}) - the worktree would crash at eval start"
  [ "$SRC" = "$(readlink -f "$WANT" 2>/dev/null)" ] \
    || die "main-tree asset '${A}' resolves to ${SRC}, not the REGISTERED target ${WANT} - refusing to pin a measurement tree to an unregistered corpus"
  if [ -L "${WT}/${A}" ]; then
    CUR="$(readlink -f "${WT}/${A}")"
    [ "$CUR" = "$SRC" ] || die "worktree asset ${WT}/${A} points at ${CUR}, not ${SRC}"
  elif [ -e "${WT}/${A}" ]; then
    # A TRACKED path of the same name is the checkout's own content, not ours:
    # accept it (it came from the pinned commit) rather than overwrite it.
    if git -C "$WT" ls-files --error-unmatch "$A" >/dev/null 2>&1; then
      echo "asset ${A} is tracked at this commit; using the checkout's own copy" >&2
    else
      die "worktree already contains an untracked non-symlink ${A} - refusing to touch it"
    fi
  else
    mkdir -p "$(dirname "${WT}/${A}")" || die "cannot create the parent of ${A} in ${WT}" 3
    ln -s "$SRC" "${WT}/${A}" || die "cannot link ${A} into ${WT}" 3
    echo "linked ${A} -> ${SRC}" >&2
  fi
done

# --- the tree must be pristine: tracked AND untracked (assets excepted) -------
DIRTY="$(git -C "$WT" status --porcelain --untracked-files=no)" \
  || die "git status failed in ${WT} - refusing to treat an error as clean"
[ -z "$DIRTY" ] || { echo "pinned worktree ${WT} has modified tracked files:" >&2; echo "$DIRTY" >&2; exit 2; }
# -uall lists untracked FILES, so match on the leading path component: the
# exception is "the assets and the lease directory", not "these exact names".
ALLOW="$(printf '%s\n' "${ASSETS[@]}" "${BOOKKEEPING[@]}" | sed 's/\./\\./g' | paste -sd'|' -)"
UNTRACKED="$(git -C "$WT" status --porcelain --untracked-files=all -- . 2>/dev/null \
             | awk '$1=="??"{print $2}' \
             | grep -vE "^(${ALLOW})(/|\$)" || true)"
[ -z "$UNTRACKED" ] || { echo "pinned worktree ${WT} carries untracked files (importable code?):" >&2; echo "$UNTRACKED" >&2; exit 2; }

prune_all "$WT"
# No explicit unlock: under --with-lock fd 8 is the CALLER's open file
# description, and releasing it here would open exactly the gap this span
# exists to close. A standalone run releases it by exiting.
echo "$WT"
