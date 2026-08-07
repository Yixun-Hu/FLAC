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
#    Trees are now LEASED: a submission writes .leases/<jobid> and the job
#    removes it on exit; a tree is prunable only when it holds no lease AND none
#    of its lease job ids is still known to Slurm. Creation is serialised by a
#    flock so concurrent submissions for the same SHA are idempotent.
#
#   MEASURE_ROOT="$(bash .../fa_orbit_measure_worktree.sh)"          # pin HEAD
#   bash .../fa_orbit_measure_worktree.sh --lease <jobid> <root>     # add a lease
#   bash .../fa_orbit_measure_worktree.sh --release <jobid> <root>   # drop a lease
#   bash .../fa_orbit_measure_worktree.sh --prune                    # sweep
# ============================================================================
set -uo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
ROOT="${MAIN_REPO}/.measure_worktrees"
# Untracked runtime inputs the eval resolves RELATIVE to the code root.
ASSETS=(AcousticRooms weights)
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

lease_is_live() {  # $1 = lease job id -- unknown ids are treated as DEAD only
  case "$1" in pending-*) return 0 ;; esac      # a pending submission is live
  squeue -h -j "$1" -o %i 2>/dev/null | grep -q "$1"
}

prunable() {  # $1 = worktree; prunable iff no lease file names a live job
  local d; d="$(lease_dir "$1")"
  [ -d "$d" ] || return 0
  local f
  for f in "$d"/*; do
    [ -e "$f" ] || continue
    lease_is_live "$(basename "$f")" && return 1
    echo "  stale lease $(basename "$f") in $1 (job is gone from squeue)" >&2
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
      git -C "$MAIN_REPO" worktree remove --force "$d" >&2 2>/dev/null \
        || echo "  (git worktree remove declined; leaving ${d} in place for inspection)" >&2
    else
      echo "keeping leased worktree ${d}" >&2
    fi
  done
  git -C "$MAIN_REPO" worktree prune >&2 2>/dev/null || true
}

cd "$MAIN_REPO" || die "cannot cd ${MAIN_REPO}" 3
mkdir -p "$ROOT" || die "cannot create ${ROOT}" 3

promote_lease() {  # $1 = pending token, $2 = real job id, $3 = worktree
  local d; d="$(lease_dir "$3")"
  [ -f "${d}/$1" ] || die "no pending lease ${d}/$1 to promote" 3
  add_lease "$2" "$3" >/dev/null   # write the real one FIRST (never a gap)
  rm -f "${d}/$1"
  echo "${d}/$2"
}

case "${1:-}" in
  --lease)   add_lease "${2:?job id}" "${3:?worktree}" >/dev/null; exit 0 ;;
  --promote) promote_lease "${2:?pending token}" "${3:?job id}" "${4:?worktree}" >/dev/null; exit 0 ;;
  --release) release_lease "${2:?job id}" "${3:?worktree}"; exit 0 ;;
  --prune)   prune_all ""; exit 0 ;;
esac

SHA="${1:-$(git rev-parse HEAD)}"
git rev-parse --verify --quiet "${SHA}^{commit}" >/dev/null || die "not a commit: ${SHA}"
SHA="$(git rev-parse "${SHA}^{commit}")"
WT="${ROOT}/${SHA}"

# --- creation is serialised: concurrent submissions for one SHA are idempotent -
LOCK="${ROOT}/.create.lock"
exec 8>"$LOCK" || die "cannot open the creation lock ${LOCK}" 3
flock 8 || die "cannot take the creation lock" 3

if [ -d "$WT" ]; then
  HAVE="$(git -C "$WT" rev-parse HEAD 2>/dev/null)" || die "existing ${WT} is not a git worktree"
  [ "$HAVE" = "$SHA" ] || die "existing ${WT} is at ${HAVE}, not ${SHA} - refusing to reuse"
  echo "reusing pinned worktree ${WT}" >&2
else
  git worktree add --detach "$WT" "$SHA" >&2 || die "git worktree add failed" 3
  echo "created pinned worktree ${WT}" >&2
fi

# --- provision the untracked runtime assets (the crasher) --------------------
for A in "${ASSETS[@]}"; do
  SRC="$(readlink -f "${MAIN_REPO}/${A}" 2>/dev/null)"
  [ -n "$SRC" ] && [ -e "$SRC" ] || die "main-tree asset '${A}' does not resolve (${SRC:-<unset>}) - the worktree would crash at eval start"
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
flock -u 8
echo "$WT"
