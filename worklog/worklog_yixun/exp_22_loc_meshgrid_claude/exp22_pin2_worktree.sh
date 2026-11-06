#!/usr/bin/env bash
# exp_22 R2-C (plan A7 r6): the Planner's pre-flight worktree -- create it, or verify it (read-only).
#
#   bash exp22_pin2_worktree.sh create <PIN2> <worktree>   git worktree add --detach, the asset links, then verify
#   bash exp22_pin2_worktree.sh verify <PIN2> <worktree>   read-only: HEAD, detached, clean tracked tree, every link
#
# Layout: the worktree keeps PIN2's TRACKED weights/ directory (PIN2 tracks weights/FLAC/*.json); every UNTRACKED entry
# of the main checkout's weights/ gets one link to the main checkout's entry; outputs_loc -> the main checkout's;
# AcousticRooms -> the NAS. It never writes a file of the main checkout (`git worktree add` writes only its own
# metadata under .git/worktrees/). Written by the Planner (Claude Opus 5.5).
set -uo pipefail
MAIN=/home/yixunhu/codespace/FLAC
ROOMS=/media/diskstation/yixunhu/FLAC/AcousticRooms
if ! { [ "$#" -eq 3 ] && [[ "$1" =~ ^(create|verify)$ ]] && [[ "$2" =~ ^[0-9a-f]{40}$ ]]; }; then
  echo "usage: bash $0 create|verify <PIN2 40-hex> <worktree>"; exit 2
fi
MODE=$1; PIN2=$2; WT=$3
die() { echo "REFUSED: $*"; exit 1; }
case "$WT" in /home/yixunhu/codespace/?*) ;; *) die "the worktree must lie under /home/yixunhu/codespace/" ;; esac
case "$WT" in *[!A-Za-z0-9/_.-]*|*/../*|*/..|*/./*) die "the worktree path must be plain" ;; esac
git -C "$MAIN" cat-file -e "${PIN2}^{commit}" 2>/dev/null || die "${PIN2} is not a commit of ${MAIN}"
git -C "$MAIN" diff --quiet "$PIN2" HEAD -- weights || die "the main checkout's tracked weights/ differ from PIN2's"
mapfile -d '' RAW < <(git -C "$MAIN" ls-files -o --directory -z -- weights)
ENTRIES=(); for e in "${RAW[@]}"; do ENTRIES+=("${e%/}"); done
[ "${#ENTRIES[@]}" -gt 0 ] || die "no untracked entry under ${MAIN}/weights"
echo "=== exp22_pin2_worktree ${MODE} -- $(date -Is) -- PIN2 ${PIN2} -- ${WT} -- this script $(sha256sum < "$0" | cut -c1-64)"
echo "untracked weights entries of the main checkout (${#ENTRIES[@]}): ${ENTRIES[*]}"

if [ "$MODE" = create ]; then
  [ ! -e "$WT" ] && [ ! -L "$WT" ] || die "${WT} exists"
  git -C "$MAIN" worktree add --detach "$WT" "$PIN2" || die "git worktree add failed"
  ln -s "$MAIN/outputs_loc" "$WT/outputs_loc" || die "the outputs_loc link"
  ln -s "$ROOMS" "$WT/AcousticRooms" || die "the AcousticRooms link"
  for rel in "${ENTRIES[@]}"; do
    [ ! -e "$WT/$rel" ] && [ ! -L "$WT/$rel" ] || die "${WT}/${rel} exists"
    parent="$(dirname "$WT/$rel")"
    [ -d "$parent" ] && [ ! -L "$parent" ] || die "the parent of ${rel} is not a real directory of the worktree"
    ln -s "$MAIN/$rel" "$WT/$rel" || die "the link ${rel}"
  done
  echo "created; verifying"
fi

FAIL=0
ok() { echo "OK: $*"; }
bad() { echo "FAILED: $*"; FAIL=1; }
if [ "$(git -C "$WT" rev-parse HEAD 2>/dev/null)" = "$PIN2" ]; then ok "HEAD = PIN2"; else bad "HEAD = PIN2"; fi
if ! git -C "$WT" symbolic-ref -q HEAD >/dev/null; then ok "HEAD is detached"; else bad "HEAD is detached"; fi
st="$(git -C "$WT" status --porcelain --untracked-files=no 2>&1)"
if [ -z "$st" ]; then ok "the tracked tree is clean"; else bad "the tracked tree is clean: ${st}"; fi
if [ -d "$WT/weights" ] && [ ! -L "$WT/weights" ]; then ok "weights/ is a real directory"; else bad "weights/ is a real directory"; fi
if [ -L "$WT/outputs_loc" ] && [ "$(readlink -f "$WT/outputs_loc")" = "$MAIN/outputs_loc" ]; then ok "outputs_loc -> ${MAIN}/outputs_loc"; else bad "outputs_loc"; fi
if [ -L "$WT/AcousticRooms" ] && [ "$(readlink -f "$WT/AcousticRooms")" = "$ROOMS" ]; then ok "AcousticRooms -> ${ROOMS}"; else bad "AcousticRooms"; fi
for rel in "${ENTRIES[@]}"; do
  if [ -L "$WT/$rel" ] && [ "$(readlink "$WT/$rel")" = "$MAIN/$rel" ] && [ "$WT/$rel" -ef "$MAIN/$rel" ]; then
    ok "${rel} -> ${MAIN}/${rel} (same device + inode)"
  else
    bad "${rel} -> ${MAIN}/${rel}"
  fi
done
mapfile -d '' WRAW < <(git -C "$WT" ls-files -o --directory -z -- weights)
got="$(for e in "${WRAW[@]}"; do printf '%s\n' "${e%/}"; done | sort)"; want="$(printf '%s\n' "${ENTRIES[@]}" | sort)"
if [ "$got" = "$want" ]; then ok "the worktree's untracked weights entries are exactly the links"; else bad "untracked weights entries: ${got}"; fi
if [ "$FAIL" = 0 ]; then echo "VERIFY PASS"; else echo "VERIFY FAIL"; fi
exit "$FAIL"
