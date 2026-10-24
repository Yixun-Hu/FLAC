#!/usr/bin/env bash
# ============================================================================
# depth_canon_guardtests.sh -- the launcher guard suite of exp_25 depth_canon
# (plan_depth_canon.md v8 §6 "Launcher guard suite", §4 gates).
#
# Isolation (exp_17 r3 BLOCKING findings): every case runs against a THROWAWAY
# clone in `mktemp -d` -- a `git clone --shared` (objects read through the
# alternates file, nothing written to the source repository) of the source HEAD,
# sparse-checked-out, with the working-tree versions of this round's launcher
# files overlaid and committed INSIDE the clone, `weights/` symlinked read-only.
# Synthetic logs, smoke evidence, checkpoints and stubs (python -> fake trainer
# for `train.py` only, findmnt, df, nvidia-smi) live in the same tmp dir. The
# source repo, its experiment folder and the NAS are never written; the suite
# never moves, hides or deletes a file outside its tmp dir.
#
# Usage (from anywhere, flac env first on PATH):  bash depth_canon_guardtests.sh
# Optional: GUARD_NAS_READONLY=1 adds the production-path DRY_RUN cases, which
# READ the real NAS (findmnt / df / stat of the save root; no writes).
#
# Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §4/§6.
# ============================================================================
set -euo pipefail

SRC_REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
EXPREL="worklog/worklog_yixun/exp_25_depth_canon_claude"
OVERLAY=(depth_canon_launch.sh depth_canon_guardtests.sh depth_canon_make_pins.sh depth_canon_pins.sha256)
H="$(mktemp -d "${TMPDIR:-/tmp}/depth_canon_guard.XXXXXX")"
trap 'rm -rf "$H"' EXIT
R="$H/repo"
PASS=0; FAIL=0; SKIP=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1${2:+ -- $2}"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP  $1${2:+ -- $2}"; SKIP=$((SKIP + 1)); }
expect() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

make_clone() {   # a fresh clone of the source HEAD + the overlaid launcher files, committed
  rm -rf "$R"
  git clone -q --shared --no-checkout "$SRC_REPO" "$R"
  git -C "$R" sparse-checkout set --cone src data/AR \
    worklog/worklog_yixun/exp_07_fa_scratch_claude "$EXPREL"
  git -C "$R" checkout -q --detach "$(git -C "$SRC_REPO" rev-parse HEAD)"
  local f
  for f in "${OVERLAY[@]}"; do
    [ -e "$SRC_REPO/$EXPREL/$f" ] && cp "$SRC_REPO/$EXPREL/$f" "$R/$EXPREL/$f"
  done
  ln -s "$SRC_REPO/weights" "$R/weights"
  ( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write )
  commit_clone "guard overlay"
}

commit_clone() {
  git -C "$R" add -A "$EXPREL" src
  git -C "$R" -c user.name=guard -c user.email=guard@localhost commit -q --allow-empty -m "$1"
}

echo "=== exp_25 depth_canon guard suite -- source HEAD $(git -C "$SRC_REPO" rev-parse --short HEAD) -- tmp $H ==="
make_clone

echo "--- P. the content-pin manifest generator ---"
LIST="$(cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --list)"
expect "P1 the list is LC_ALL=C sorted and unique" \
  test "$LIST" = "$(LC_ALL=C sort -u <<<"$LIST")"
expect "P1b the list covers the training path" bash -c '
  for p in train.py defaults.ini src/__init__.py src/data/yaw_rotation.py src/data/dataset.py \
           src/models/conditioners.py src/training/diffusion.py src/training/factory.py \
           src/training/losses/losses.py src/inference/sampling.py src/interface/aeiou.py \
           src/configs/dataset_configs/custom_metadata/AR_md.py \
           src/configs/dataset_configs/AR/train/acousticroom_train.json data/AR/train.json \
           weights/FLAC/VAE.safetensors worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json \
           '"$EXPREL"'/FLAC_AR_CANON.json worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py; do
    grep -qxF "$p" <<<"$1" || { echo "missing $p"; exit 1; }; done
  ! grep -q "^src/tests/" <<<"$1"' _ "$LIST"
M1="$(cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh")"
M2="$(cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh")"
expect "P2 the manifest is deterministic and lists exactly the --list paths" \
  bash -c '[ "$1" = "$2" ] && [ "$(awk "{print \$2}" <<<"$1")" = "$3" ]' _ "$M1" "$M2" "$LIST"
expect "P3 --write produces a manifest sha256sum verifies" \
  bash -c 'cd "$1" && sha256sum -c --strict --quiet "$2/depth_canon_pins.sha256"' _ "$R" "$EXPREL"
expect "P3b the source repo's committed manifest is current" \
  bash -c 'cd "$1" && [ "$(bash "$2/depth_canon_make_pins.sh")" = "$(cat "$2/depth_canon_pins.sha256")" ]' \
  _ "$SRC_REPO" "$EXPREL"
touch "$R/src/models/zz_guard_new_module.py"
expect "P4 a new src/models/*.py is covered automatically" \
  bash -c 'cd "$1" && bash "$2/depth_canon_make_pins.sh" --list | grep -qxF src/models/zz_guard_new_module.py' _ "$R" "$EXPREL"
rm -f "$R/src/models/zz_guard_new_module.py"
mv "$R/weights" "$R/weights.off"
expect "P5 a missing pinned file fails the generator" \
  bash -c 'cd "$1" && ! bash "$2/depth_canon_make_pins.sh" --list >/dev/null 2>"$3/p5.err" && grep -q "missing pinned file weights/FLAC/VAE.safetensors" "$3/p5.err"' _ "$R" "$EXPREL" "$H"
mv "$R/weights.off" "$R/weights"

echo
echo "=== ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ==="
[ "$FAIL" -eq 0 ]
