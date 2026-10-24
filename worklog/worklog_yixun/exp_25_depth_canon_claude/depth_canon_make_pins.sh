#!/usr/bin/env bash
# ============================================================================
# depth_canon_make_pins.sh -- the content-pin manifest of exp_25's training
# path (plan_depth_canon.md v8 §4 gate (1)). One definition of WHAT is pinned:
# the launcher compares the committed manifest's path column with `--list`
# (coverage) and verifies it with `sha256sum -c --strict` (content).
#
#   --list    the pinned paths, one per line, LC_ALL=C sorted, unique
#   (none)    the manifest on stdout: "<sha256>  <path>" per path, same order
#   --write   (re)write depth_canon_pins.sha256 next to this script (atomic)
#
# Pinned: the fixed files below + every *.py directly inside each GLOB_DIRS
# entry (src/training/losses/ and src/interface/ are imported by the training
# wrapper, so they are pinned too -- a superset of the plan's list). Any missing
# fixed file or empty directory is an error. Deterministic: same tree, same bytes.
#
# Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §4.
# ============================================================================
set -euo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
EXPDIR="worklog/worklog_yixun/exp_25_depth_canon_claude"
EXP07="worklog/worklog_yixun/exp_07_fa_scratch_claude"
FIXED=(train.py defaults.ini src/__init__.py
       src/configs/dataset_configs/custom_metadata/AR_md.py
       src/configs/dataset_configs/AR/train/acousticroom_train.json data/AR/train.json
       weights/FLAC/VAE.safetensors "$EXP07/FLAC_AR_BVp1.json" "$EXPDIR/FLAC_AR_CANON.json"
       "$EXP07/assert_arm_configs.py"
       "$EXP07/FLAC_AR_BV.json" "$EXP07/FLAC_AR_BF.json")   # opened by assert_arm_configs.py:105 (r3b-F1 fix 5)
GLOB_DIRS=(src/data src/models src/training src/training/losses src/inference src/interface)

pin_paths() {
  local p d n
  for p in "${FIXED[@]}"; do
    [ -f "$p" ] || { echo "make_pins: missing pinned file $p" >&2; return 1; }
    echo "$p"
  done
  for d in "${GLOB_DIRS[@]}"; do
    n=0
    for p in "$d"/*.py; do
      [ -f "$p" ] || continue
      echo "$p"; n=$((n + 1))
    done
    [ "$n" -gt 0 ] || { echo "make_pins: no *.py directly inside $d" >&2; return 1; }
  done
}

PATHS="$(pin_paths | LC_ALL=C sort -u)"
manifest() { local p; while IFS= read -r p; do sha256sum -- "$p"; done <<<"$PATHS"; }

case "${1:-}" in
  --list)  printf '%s\n' "$PATHS" ;;
  "")      manifest ;;
  --write) tmp="$(mktemp "$EXPDIR/.depth_canon_pins.XXXXXX")"
           manifest > "$tmp" && mv -f "$tmp" "$EXPDIR/depth_canon_pins.sha256"
           echo "wrote $EXPDIR/depth_canon_pins.sha256 ($(wc -l < "$EXPDIR/depth_canon_pins.sha256") files)" ;;
  *)       echo "usage: $0 [--list | --write]" >&2; exit 2 ;;
esac
