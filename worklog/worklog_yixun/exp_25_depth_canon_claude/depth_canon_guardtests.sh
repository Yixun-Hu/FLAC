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

# --- harness: marker, stubs on PATH for the launcher only --------------------- #
BIN="$H/bin"; mkdir -p "$BIN"; touch "$H/.depth_canon_harness"
REAL_PY="$(command -v python)"
cat > "$BIN/python" <<STUB
#!/usr/bin/env bash
# guard stub: 'python train.py ...' runs the fake trainer; everything else is the real python
if [ "\${1:-}" = "train.py" ]; then shift; exec bash "$H/fake_train.sh" "\$@"; fi
exec "$REAL_PY" "\$@"
STUB
chmod +x "$BIN"/*

# launch <VAR=value ...>: the CLONE's launcher, harness mode, stubs first on PATH; sets OUT, RC
launch() {
  set +e
  OUT="$(cd "$R" && env PATH="$BIN:$PATH" DEPTH_CANON_HARNESS="$H" \
        EXPECT_HEAD="$(git -C "$R" rev-parse HEAD)" "$@" bash "$EXPREL/depth_canon_launch.sh" 2>&1)"
  RC=$?
  set -e
}
# case_rc <name> <want_rc> <want_text> <VAR=value ...>
case_rc() {
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  launch "$@"
  if [ "$RC" = "$want_rc" ] && grep -qF -- "$want_txt" <<<"$OUT"; then ok "$name (rc=$RC)"
  else bad "$name" "want rc=$want_rc + '$want_txt'; got rc=$RC; last: $(tail -1 <<<"$OUT")"; fi
}
argv_of() { grep -m1 '^ARGV: ' <<<"$1" | sed 's/^ARGV: //' || true; }
argval() {   # the token after <flag>, printed only when the flag occurs exactly once
  local -a t; read -r -a t <<<"$1"; local i n=0 v=""
  for ((i = 0; i < ${#t[@]}; i++)); do
    if [ "${t[$i]}" = "$2" ]; then n=$((n + 1)); v="${t[$((i + 1))]:-}"; fi
  done
  [ "$n" -eq 1 ] && printf '%s' "$v"
}
has_value() { [ "$(argval "$1" "$2")" = "$3" ]; }

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

echo "--- A. mode, dry-run flag, rung (string-pinned), harness location ---"
case_rc "A1 MODE unset"          2 "MODE must be exactly SMOKE or FULL"  MODE=
case_rc "A2 MODE lowercase"      2 "MODE must be exactly SMOKE or FULL"  MODE=full DRY_RUN=1
case_rc "A3 MODE near-miss"      2 "MODE must be exactly SMOKE or FULL"  MODE=FULL_ DRY_RUN=1
case_rc "A4 DRY_RUN not 0/1"     2 "DRY_RUN must be 0 or 1"              MODE=SMOKE DRY_RUN=yes
case_rc "A5 accumulation smuggled" 2 "only the BN-compliant rung"        MODE=FULL DRY_RUN=1 MB=8 ACC=8
case_rc "A6 micro-batch halved"  2 "only the BN-compliant rung"          MODE=FULL DRY_RUN=1 MB=16
case_rc "A7 non-numeric rung"    2 "only the BN-compliant rung"          MODE=FULL DRY_RUN=1 MB=32x
case_rc "A8 harness outside /tmp" 2 "must be a directory under /tmp"     MODE=SMOKE DRY_RUN=1 DEPTH_CANON_HARNESS="$HOME"
case_rc "A9 harness without its marker" 2 "lacks the .depth_canon_harness marker" MODE=SMOKE DRY_RUN=1 DEPTH_CANON_HARNESS="$BIN"

echo "--- K. HEAD pin ---"
case_rc "K1 EXPECT_HEAD unset"   2 "EXPECT_HEAD must be the full 40-hex"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD=
case_rc "K2 short EXPECT_HEAD"   2 "EXPECT_HEAD must be the full 40-hex"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD="$(git -C "$R" rev-parse --short HEAD)"
case_rc "K3 wrong EXPECT_HEAD"   2 "HEAD PIN FAILED"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD=0000000000000000000000000000000000000000
case_rc "K4 the pin passes"      0 "HEAD pin OK: $(git -C "$R" rev-parse HEAD)"  MODE=SMOKE DRY_RUN=1

echo "--- F. the ACTUAL train.py argv (whole tokens, each flag exactly once) ---"
launch MODE=FULL DRY_RUN=1; FULL_ARGV="$(argv_of "$OUT")"
launch MODE=SMOKE DRY_RUN=1; SMOKE_ARGV="$(argv_of "$OUT")"
expect "F1 argv starts with 'python train.py'" bash -c '[[ "$1" == "python train.py "* ]]' _ "$FULL_ARGV"
for fv in "--model-config $EXPREL/FLAC_AR_CANON.json" \
          "--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json" \
          "--pretransform-ckpt-path weights/FLAC/VAE.safetensors" "--max-steps 40000" \
          "--checkpoint-every 2500" "--batch-size 32" "--accum-batches 1" "--num-gpus 2" \
          "--sync-batchnorm true" "--strategy ddp_find_unused_parameters_true" \
          "--precision bf16-mixed" "--seed 42" "--num-workers 6" "--logger wandb" \
          "--name FLAC_exp25_CANON" "--experiment-name exp25_CANON" "--save-dir $H/save"; do
  expect "F2 FULL ${fv}" has_value "$FULL_ARGV" "${fv%% *}" "${fv#* }"
done
for fv in "--max-steps 25" "--checkpoint-every 1000000" "--name FLAC_exp25_CANON_smoke" \
          "--experiment-name exp25_CANON_smoke" "--save-dir $H/save" "--sync-batchnorm true"; do
  expect "F3 SMOKE ${fv}" has_value "$SMOKE_ARGV" "${fv%% *}" "${fv#* }"
done
expect "F4 no validation set, no resume, no pretrained init (P1 passed none)" bash -c '
  [ -n "$1" ] || exit 1
  for f in --val-dataset-config --ckpt-path --pretrained-ckpt-path; do grep -qw -- "$f" <<<"$1" && exit 1; done; exit 0' _ "$FULL_ARGV"
expect "F5 every emitted flag is a defaults.ini key (prefigure)" bash -c '
  [ -n "$1" ] || exit 1
  for f in $(grep -o -- "--[a-z-]*" <<<"$1"); do k="${f#--}"; grep -qE "^${k//-/_} *=" "$2/defaults.ini" || { echo "unknown $f"; exit 1; }; done' \
  _ "$FULL_ARGV" "$R"
case_rc "D1 SMOKE dry run reaches the boundary" 0 "DRY_RUN: all gates passed" MODE=SMOKE DRY_RUN=1

echo
echo "=== ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ==="
[ "$FAIL" -eq 0 ]
