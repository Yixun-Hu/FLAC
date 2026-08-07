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
# ============================================================================
set -uo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
ROOT="${MAIN_REPO}/.measure_worktrees"
# Untracked runtime inputs the eval resolves RELATIVE to the code root, each
# pinned to the target it MUST resolve to. The main tree's symlink is mutable
# (anyone can repoint it at a different corpus between screens), so following it
# blindly would let the measurement stack silently change datasets. Registered
# target, or abort.
ASSETS=(AcousticRooms weights)
ASSET_TARGETS=(/n/fs/gatrdp/datasets/AcousticRooms /n/fs/gatrdp/codespace/FLAC/weights)
# Bookkeeping this script itself creates in the tree; never code.
BOOKKEEPING=(.leases)

die() { echo "$1" >&2; exit "${2:-2}"; }

lease_dir() { echo "${1}/.leases"; }

# --- lease helpers -----------------------------------------------------------
add_lease() {   # $1 = job id (or pending token), $2 = worktree
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

prunable() {  # $1 = worktree; prunable iff no lease file names a live job
  local d; d="$(lease_dir "$1")"
  [ -d "$d" ] || return 0
  local f
  for f in "$d"/*; do
    [ -e "$f" ] || continue
    lease_is_live "$(basename "$f")" && return 1
    echo "  stale lease $(basename "$f") in $1 (Slurm reports no such job)" >&2
    rm -f "$f"
  done
  return 0
}

prune_all() {  # never touches $KEEP_WT
  local keep="${1:-}"
  local d
  for d in "$ROOT"/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    [ "$d" = "$keep" ] && continue
    if prunable "$d"; then
      echo "pruning unleased measurement worktree ${d}" >&2
      # our own symlinks and .leases/ make `git worktree remove` bail out
      # half-done, so clear them first and confirm the directory is really gone
      rm -rf "${d}/.leases" "${d}/AcousticRooms" "${d}/weights" 2>/dev/null
      git -C "$MAIN_REPO" worktree remove --force "$d" >&2 2>/dev/null || true
      if [ -d "$d" ]; then
        case "$(basename "$d")" in
          [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
            rm -rf "$d" 2>/dev/null || echo "  (could not remove ${d}; leaving it for inspection)" >&2 ;;
          *) echo "  (refusing to remove non-sha entry ${d})" >&2 ;;
        esac
      fi
    else
      echo "keeping leased worktree ${d}" >&2
    fi
  done
  git -C "$MAIN_REPO" worktree prune >&2 2>/dev/null || true
}

cd "$MAIN_REPO" || die "cannot cd ${MAIN_REPO}" 3
mkdir -p "$ROOT" || die "cannot create ${ROOT}" 3

# ONE lock for the whole store, held by EVERY operation — creation, lease,
# release and prune alike. Locking creation only still allowed a prune sweep to
# enumerate leases while a submission was writing one; the two must never
# interleave, so they take the same lock.
LOCK="${ROOT}/.store.lock"
exec 8>"$LOCK" || die "cannot open the store lock ${LOCK}" 3
flock 8 || die "cannot take the store lock ${LOCK}" 3

case "${1:-}" in
  --lease)   add_lease "${2:?job id}" "${3:?worktree}" >/dev/null; exit 0 ;;
  --release) release_lease "${2:?job id}" "${3:?worktree}"; exit 0 ;;
  --prune)   prune_all ""; exit 0 ;;
esac

SHA="${1:-$(git rev-parse HEAD)}"
git rev-parse --verify --quiet "${SHA}^{commit}" >/dev/null || die "not a commit: ${SHA}"
SHA="$(git rev-parse "${SHA}^{commit}")"
WT="${ROOT}/${SHA}"

# creation runs under the store lock taken above: concurrent submissions for one
# SHA are idempotent, and no sweep can observe a half-built tree
# A DIRECTORY IS NOT A WORKTREE. `git worktree remove` can unregister a tree and
# still leave its directory behind (our own symlinks and .leases/ defeat its
# cleanup), and because that leftover sits INSIDE the main checkout, `git -C` on
# it walks UP and answers for the MAIN repo — HEAD matches, and the tree looks
# reusable while containing no code at all. Verify identity, never presence.
is_live_worktree() {   # $1 = candidate directory
  [ -e "$1/.git" ] || return 1
  [ "$(git -C "$1" rev-parse --show-toplevel 2>/dev/null)" = "$1" ] || return 1
  [ "$(readlink -f "$(git -C "$1" rev-parse --git-common-dir 2>/dev/null)")" \
    = "$(readlink -f "$(git -C "$MAIN_REPO" rev-parse --git-common-dir)")" ] || return 1
}

if [ -d "$WT" ] && ! is_live_worktree "$WT"; then
  # only ever inside our own store, and only a <40-hex> directory
  case "$(basename "$WT")" in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
    *) die "refusing to clear ${WT}: not a sha-named store entry" 3 ;;
  esac
  echo "clearing a half-removed worktree directory at ${WT}" >&2
  git worktree prune >&2 2>/dev/null || true
  rm -rf "$WT" || die "cannot clear the stale directory ${WT}" 3
fi

if [ -d "$WT" ]; then
  HAVE="$(git -C "$WT" rev-parse HEAD 2>/dev/null)" || die "existing ${WT} is not a git worktree"
  [ "$HAVE" = "$SHA" ] || die "existing ${WT} is at ${HAVE}, not ${SHA} - refusing to reuse"
  echo "reusing pinned worktree ${WT}" >&2
else
  git worktree add --detach "$WT" "$SHA" >&2 || die "git worktree add failed" 3
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
    die "worktree already contains a non-symlink ${A} - refusing to touch it"
  else
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
flock -u 8            # released here anyway when the script exits and fd 8 closes
echo "$WT"
