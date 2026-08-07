#!/usr/bin/env bash
# ============================================================================
# fa_orbit_measure_worktree.sh — prepare a READ-ONLY, pinned code tree for one
# measurement job, and print its path.
#
# WHY (the commit-freeze trap, final form). A measurement job reads code from the
# checkout it runs in, while development keeps committing to that same checkout.
# Jobs 3649599/3649600 evaluated correctly for 13 minutes and were then refused
# because the evaluator stamped source_sha from a HEAD that had moved during the
# run, while the driver had pinned the submission commit. The historical
# workaround was to freeze development while measurements were in flight — which
# does not scale and has already cost this experiment several rounds.
#
# The fix: measurements run from a git worktree pinned to the submission SHA under
# .measure_worktrees/<sha>/. Development continues in the main tree; the job's
# code identity is immutable for its whole lifetime, so source_sha ALWAYS equals
# the pinned SHA. Outputs still go to the MAIN tree (outputs_FLAC, exp-folder
# logs) via absolute paths, so results land where everything else expects them.
#
#   MEASURE_ROOT="$(bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh)"
#   sbatch --export=ALL,EXPECT_SHA=$(git rev-parse HEAD),MEASURE_ROOT=$MEASURE_ROOT,... <driver>
#
# With no argument it pins the current HEAD. Existing worktrees are reused, and
# all but the newest 3 are pruned so the directory cannot grow without bound.
# ============================================================================
set -uo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
KEEP=3
cd "$MAIN_REPO" || { echo "cannot cd ${MAIN_REPO}" >&2; exit 3; }

SHA="${1:-$(git rev-parse HEAD)}"
git rev-parse --verify --quiet "${SHA}^{commit}" >/dev/null \
  || { echo "not a commit: ${SHA}" >&2; exit 2; }
SHA="$(git rev-parse "${SHA}^{commit}")"

ROOT="${MAIN_REPO}/.measure_worktrees"
WT="${ROOT}/${SHA}"
mkdir -p "$ROOT" || { echo "cannot create ${ROOT}" >&2; exit 3; }

if [ -d "$WT" ]; then
  HAVE="$(git -C "$WT" rev-parse HEAD 2>/dev/null)"
  if [ "$HAVE" = "$SHA" ]; then
    echo "reusing pinned worktree ${WT}" >&2
  else
    echo "existing ${WT} is at ${HAVE}, not ${SHA} - refusing to reuse" >&2
    exit 2
  fi
else
  git worktree add --detach "$WT" "$SHA" >&2 || { echo "git worktree add failed" >&2; exit 3; }
  echo "created pinned worktree ${WT}" >&2
fi

# it must be pristine: a measurement may not read edited code
DIRTY="$(git -C "$WT" status --porcelain --untracked-files=no)"
[ -z "$DIRTY" ] || { echo "pinned worktree ${WT} is dirty - refusing:" >&2; echo "$DIRTY" >&2; exit 2; }

# prune: keep the newest KEEP worktrees, drop the rest (they are pure caches)
mapfile -t OLD < <(ls -1dt "${ROOT}"/*/ 2>/dev/null | tail -n +$((KEEP + 1)))
for O in "${OLD[@]:-}"; do
  [ -n "$O" ] || continue
  case "$(basename "${O%/}")" in "$SHA") continue ;; esac
  echo "pruning old measurement worktree ${O%/}" >&2
  git worktree remove --force "${O%/}" >&2 2>/dev/null || rm -rf "${O%/}"
done
git worktree prune >&2 2>/dev/null || true

echo "$WT"
