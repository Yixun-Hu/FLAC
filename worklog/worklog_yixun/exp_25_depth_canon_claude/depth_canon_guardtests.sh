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

commit_clone() {   # everything the throwaway clone holds (ignored files stay ignored)
  git -C "$R" add -A
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
cat > "$BIN/findmnt" <<'STUB'
#!/usr/bin/env bash
# guard stub: the CIFS share for every target (GUARD_FINDMNT=real defers to the real findmnt)
[ "${GUARD_FINDMNT:-cifs}" = real ] && exec /usr/bin/findmnt "$@"
printf '%s\n' "${GUARD_FINDMNT_LINE:-cifs   //10.249.7.28/shared_data}"
STUB
cat > "$BIN/df" <<'STUB'
#!/usr/bin/env bash
# guard stub: a df -P -B1 table whose 'Available' column is GUARD_DF_AVAIL bytes
echo "Filesystem 1-blocks Used Available Capacity Mounted on"
echo "//10.249.7.28/shared_data 47764605435904 31138083921920 ${GUARD_DF_AVAIL:-16626521513984} 66% /media/diskstation"
STUB
cat > "$BIN/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
# guard stub: memory.free queries answer GUARD_GPU_FREE (MiB); other queries print nothing
case "$*" in *memory.free*) echo "${GUARD_GPU_FREE:-40000}" ;; esac
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
expect "P3b the source repo's COMMITTED manifest is current" \
  bash -c 'cd "$1" && [ "$(bash "$2/depth_canon_make_pins.sh")" = "$(git show "HEAD:$2/depth_canon_pins.sha256")" ]' \
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
echo "--- M. content pins: coverage, bytes (incl. the gitignored VAE), committed ---"
printf '\n# guard tamper\n' >> "$R/src/data/utils.py"
case_rc "M1 a tampered pinned file"  2 "PIN MANIFEST FAILED: content"  MODE=SMOKE DRY_RUN=1
git -C "$R" checkout -q -- src/data/utils.py
touch "$R/src/models/zz_guard_new_module.py"
case_rc "M2 an unpinned new module"  2 "PIN MANIFEST FAILED: coverage" MODE=SMOKE DRY_RUN=1
rm -f "$R/src/models/zz_guard_new_module.py"
printf '\n# guard edit\n' >> "$R/src/data/utils.py"
( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null )
case_rc "M3 consistent but uncommitted" 2 "PINNED TREE NOT COMMITTED"  MODE=SMOKE DRY_RUN=1
git -C "$R" checkout -q -- src/data/utils.py "$EXPREL/depth_canon_pins.sha256"
printf '\n# guard edit\n' >> "$R/$EXPREL/depth_canon_launch.sh"
case_rc "M4 an uncommitted launcher edit" 2 "PINNED TREE NOT COMMITTED"  MODE=SMOKE DRY_RUN=1
git -C "$R" checkout -q -- "$EXPREL/depth_canon_launch.sh"
mkdir -p "$H/otherweights/FLAC"; printf 'not the VAE' > "$H/otherweights/FLAC/VAE.safetensors"
rm "$R/weights"; ln -s "$H/otherweights" "$R/weights"
case_rc "M5 another VAE behind the (gitignored) weights link" 2 "PIN MANIFEST FAILED: content"  MODE=SMOKE DRY_RUN=1
rm "$R/weights"; ln -s "$SRC_REPO/weights" "$R/weights"
printf '\n# guard edit\n' >> "$R/src/data/utils.py"
( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null ); commit_clone "guard: new pin commit"
case_rc "M6 a committed change with its regenerated manifest is a new valid pin" 0 "pin manifest OK: 44 files" MODE=SMOKE DRY_RUN=1
make_clone >/dev/null

echo "--- C. config contract: the arm is the P1 control plus exactly training.canonicalize ---"
mutate() {   # mutate <file> '<python on cfg>': rewrite in the clone, regenerate pins, commit
  make_clone >/dev/null
  ( cd "$R" && python - "$1" "$2" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))
exec(expr, {"cfg": cfg})
open(path, "w").write(json.dumps(cfg, indent=4))
PY
    bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null )
  commit_clone "guard: mutated $1"
}
ARM="$EXPREL/FLAC_AR_CANON.json"; CTL="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
mutate "$ARM" 'cfg["training"]["canonicalize"]["img_w"] = 256'
case_rc "C1 block width 256"            2 "canonicalize block is not the registered treatment" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["canonicalize"]["enabled"] = 1'
case_rc "C2 enabled as int 1"           2 "wrong TYPES" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["canonicalize"]["img_w"] = 512.0'
case_rc "C3 width as float"             2 "wrong TYPES" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"].pop("canonicalize")'
case_rc "C4 treatment missing"          2 "canonicalize block is not the registered treatment" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["cond_method"] = "fa_invariant"'
case_rc "C5 FA conditioning added"      2 "must be vanilla-conditioned" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["yaw_aug"] = {"enabled": False, "img_w": 512, "seed": 42}'
case_rc "C6 a yaw_aug block added"      2 "yaw_aug must be absent" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["model"]["conditioning"]["configs"][1]["config"]["gradient_checkpointing"] = False'
case_rc "C7 grad-ckpt off on one ViT"   2 "gradient_checkpointing must be true" MODE=SMOKE DRY_RUN=1
mutate "$ARM" '[c["config"]["ViT"].update(img_w=256) for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]'
case_rc "C8 ViT widths 256"             2 "ViT widths" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_rc "C9 a second, silent delta"     2 "NOT the control plus the one registered delta" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'b = cfg["training"].pop("canonicalize"); u = cfg["training"].pop("use_ema"); cfg["training"].update(canonicalize=b, use_ema=u)'
case_rc "C10 block not the last training key" 2 "last key of training" MODE=SMOKE DRY_RUN=1
mutate "$ARM" 'cfg["training"]["use_ema"] = False'
case_rc "C11 EMA off"                   2 "use_ema must be true" MODE=SMOKE DRY_RUN=1
mutate "$CTL" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_rc "C12 the control itself moved"  2 "control config sha256" MODE=SMOKE DRY_RUN=1
make_clone >/dev/null
sed -i 's/"img_w": 512$/"img_w":  512/' "$R/$ARM"
( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null ); commit_clone "guard: whitespace"
case_rc "C13 parse-equal but not byte-equal" 2 "not the control's bytes plus the literal insertion" MODE=SMOKE DRY_RUN=1
make_clone >/dev/null
case_rc "C14 the genuine arm passes"    0 "config contract OK" MODE=SMOKE DRY_RUN=1

echo "--- E. FULL needs a PASSING smoke evidence of this HEAD + manifest (+ harness flag) ---"
write_evidence() {   # write_evidence <name> '<python dict of overrides>'
  ( cd "$R" && python - "$EXPREL" "$1" "$2" <<'PY'
import hashlib, json, subprocess, sys
exp, name, over = sys.argv[1:4]
log = f"{exp}/depth_canon_guard_{name}_smoke_train.log"
open(log, "w").write("synthetic smoke log\n")
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
man = hashlib.sha256(open(f"{exp}/depth_canon_pins.sha256", "rb").read()).hexdigest()
ev = {"schema_version": 1, "kind": "depth_canon_smoke_evidence", "verdict": "PASS", "head": head,
      "manifest_sha256": man, "harness": True, "config_contract_ok": True, "banner_seen": True,
      "endpoint_seen": True, "ranks_seen": True, "nonfinite_seen": False, "smoke_checkpoints": 0,
      "log": log, "log_sha256": hashlib.sha256(open(log, "rb").read()).hexdigest()}
ev.update(eval(over))
json.dump(ev, open(f"{exp}/depth_canon_smoke_evidence_{name}.json", "w"))
PY
  )
}
clear_evidence() { rm -f "$R/$EXPREL"/depth_canon_smoke_evidence_*.json; }
NEED="FULL requires a PASSING smoke evidence"
case_rc "E1 no evidence"                    2 "$NEED" MODE=FULL DRY_RUN=1
write_evidence e2 '{"head": "1" * 40}'
case_rc "E2 evidence of another HEAD"       2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e3 '{"manifest_sha256": "2" * 64}'
case_rc "E3 evidence of another manifest"   2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e4 '{"verdict": "FAIL"}'
case_rc "E4 a failed smoke"                 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e5 '{"harness": False}'
case_rc "E5 production evidence in harness mode" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e6 '{}'; printf 'edited\n' >> "$R/$EXPREL/depth_canon_guard_e6_smoke_train.log"
case_rc "E6 evidence whose log was edited"  2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e7 '{"smoke_checkpoints": 1}'
case_rc "E7 a smoke that wrote a checkpoint" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e8 '{"banner_seen": 1}'
case_rc "E8 a truthy non-bool field"        2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e9 '{}'
case_rc "E9 genuine evidence accepted"      0 "smoke evidence: $EXPREL/depth_canon_smoke_evidence_e9.json" MODE=FULL DRY_RUN=1

echo "--- N. NAS gate on the save root: identity, free space, write probe ---"
case_rc "N1 a non-CIFS save root (real findmnt)" 2 "NAS GATE FAILED: identity" MODE=SMOKE DRY_RUN=1 GUARD_FINDMNT=real
case_rc "N2 another CIFS share"             2 "NAS GATE FAILED: identity" MODE=SMOKE DRY_RUN=1 GUARD_FINDMNT_LINE="cifs //10.0.0.1/other"
case_rc "N3 under 30 GiB free"              2 "NAS GATE FAILED: free space" MODE=SMOKE DRY_RUN=1 GUARD_DF_AVAIL=1000
case_rc "N4 dry run skips the write probe"  0 "write probe SKIPPED under DRY_RUN" MODE=SMOKE DRY_RUN=1
mkdir -p "$H/save"; chmod 555 "$H/save"
case_rc "N5 an unwritable save root"        2 "NAS GATE FAILED: write probe" MODE=SMOKE
chmod 755 "$H/save"

echo "--- S. run-owned namespaces ---"
mkdir -p "$H/save/FLAC_exp25_CANON/exp25_CANON/checkpoints"; touch "$H/save/FLAC_exp25_CANON/exp25_CANON/checkpoints/old.ckpt"
case_rc "S1 FULL into a non-empty namespace" 2 "RUN NAMESPACE NOT EMPTY" MODE=FULL DRY_RUN=1
rm -rf "$H/save/FLAC_exp25_CANON"
mkdir -p "$H/save/FLAC_exp25_CANON_smoke/exp25_CANON_smoke/checkpoints"; touch "$H/save/FLAC_exp25_CANON_smoke/exp25_CANON_smoke/checkpoints/x.ckpt"
case_rc "S2 SMOKE namespace holding a checkpoint" 2 "SMOKE NAMESPACE HOLDS CHECKPOINTS" MODE=SMOKE DRY_RUN=1
rm -rf "$H/save/FLAC_exp25_CANON_smoke"

case_rc "D1 SMOKE dry run reaches the boundary" 0 "DRY_RUN: all gates passed" MODE=SMOKE DRY_RUN=1

echo
echo "=== ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ==="
[ "$FAIL" -eq 0 ]
