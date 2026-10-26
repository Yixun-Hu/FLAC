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
OVERLAY=(depth_canon_launch.sh depth_canon_guardtests.sh depth_canon_make_pins.sh depth_canon_pins.sha256
         depth_canon_audit_gate.py)
H="$(mktemp -d /tmp/depth_canon_guard.XXXXXX)"   # rB-F2 F2-3 (c): /tmp, never $TMPDIR
case "$(readlink -f -- "$H")" in /tmp/depth_canon_guard.??????) ;;
  *) echo "GUARD ABORTED: the tmp root '${H}' does not resolve under /tmp/depth_canon_guard.*"; exit 2 ;; esac
trap 'rm -rf "$H"' EXIT
R="$H/repo"
PASS=0; FAIL=0; SKIP=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1${2:+ -- $2}"; FAIL=$((FAIL + 1)); }
skip() { echo "  SKIP  $1${2:+ -- $2}"; SKIP=$((SKIP + 1)); }
FILTERED=0
want() {   # GUARD_ONLY=P,T,...: run only the cases of those sections (the case name's leading letters)
  [ -z "${GUARD_ONLY:-}" ] && return 0
  case ",${GUARD_ONLY}," in *",${1%%[0-9]*},"*) return 0 ;; esac
  FILTERED=$((FILTERED + 1)); return 1
}
expect() { local name="$1"; shift; want "$name" || return 0; if "$@"; then ok "$name"; else bad "$name"; fi; }

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
  write_audit ok || { echo "GUARD ABORTED: make_clone could not build the harness tie-audit fixture (reason above)" >&2; exit 1; }
}

write_audit() {   # write_audit <mode>: a REAL tie audit (the clone's audit script) of a tiny synthetic tree in
                  # $H/audit_tree, harness-tagged, + sidecar + the harness contract; the audit report is a
                  # PINNED file (rA A1 i), so the manifest is regenerated and the clone committed
  ( cd "$R" && "$REAL_PY" - "$EXPREL" "$1" "$H" <<'PY'
import importlib.util, json, os, shutil, sys
import numpy as np
exp, mode, h = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("guard_tie_audit", os.path.join(exp, "depth_canon_tie_audit.py"))
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
tree = os.path.join(h, "audit_tree")
shutil.rmtree(tree, ignore_errors=True)
os.makedirs(tree)
root, ar_md = os.path.join(tree, "AcousticRooms"), os.path.abspath("src/configs/dataset_configs/custom_metadata/AR_md.py")
template = json.load(open("src/configs/dataset_configs/AR/train/acousticroom_train.json"))
SPLITS = {"train": {"Cafe": {"Cafe_idx_0": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav", "S001_R002_hybrid_IR.wav"]}},
          "unseen": {"Office": {"Office_idx_2": ["S001_R005_hybrid_IR.wav", "S002_R006_hybrid_IR.wav"]}},
          "seen": {"Cafe": {"Cafe_idx_0": ["S003_R001_hybrid_IR.wav"]}, "Office": {"Office_idx_2": ["S003_R005_hybrid_IR.wav"]}}}
cfgs = {}
for name, split in SPLITS.items():
    jp, cp = os.path.join(tree, f"{name}.json"), os.path.join(tree, f"{name}_cfg.json")
    json.dump(split, open(jp, "w"))
    cfg = json.loads(json.dumps(template))
    cfg["datasets"][0].update(path=root, json_file_path=jp, custom_metadata_module=ar_md)
    json.dump(cfg, open(cp, "w"), indent=4)
    cfgs[name] = cp
sp = {n: audit.read_split_config(cfgs[n]) for n in SPLITS}
maps, ledger = audit.collect_maps(sp)
rng = np.random.default_rng(3)
for k, key in enumerate(sorted(maps)):
    os.makedirs(os.path.dirname(maps[key]["path"]), exist_ok=True)
    d = 0.5 + 7.0 * rng.random((256, 512))
    if mode == "survivor" and k == 0:
        d = np.tile(d[:, :128], (1, 4))                       # exactly periodic -> an image-key survivor
    if mode in ("error", "ok_but_error") and k == 1:
        d[0, 0] = np.inf                                      # an unprocessed map
    np.save(maps[key]["path"], d)
limit = 2 if mode == "limit" else None
rows = audit.run_rows(maps, ar_md, root, 1, limit)
rep = audit.build_report(rows, ledger, list(SPLITS), len(maps), limit, ar_md)
rep["harness"] = True                                         # the guard's reports are never production
if mode in ("ok", "ok_but_error"):   # rB-F2 F2-3 (b): a fixture meant to PASS must BE a complete PASS -- loudly
    g = rep["gate"]
    if g.get("verdict") != "PASS" or g.get("n_errors") or g.get("n_survivors") or rep["errors"] or rep["survivors"]:
        sys.stderr.write(f"GUARD FIXTURE FAILED: write_audit({mode}) is not a complete PASS: gate {g}; "
                         f"errors {rep['errors'][:5]}; survivors {rep['survivors'][:5]}\n")
        sys.exit(3)
if mode == "script":
    rep["tool"]["script_sha256"] = "0" * 64
if mode == "counts":
    rep["inputs"][2]["n_items"] -= 1
if mode == "dup":
    rep["rows"].insert(1, dict(rep["rows"][0]))
if mode == "arbitrary":
    rep["rows"][0]["path"] = "m0"
if mode == "untagged":
    del rep["harness"]
out = os.path.join(exp, "depth_canon_tie_audit.json")
audit.write_report(rep, out)                                  # the audit's own writer: sidecar recomputed
if mode == "tamper":
    open(out + ".sha256", "w").write("0" * 64 + "  depth_canon_tie_audit.json\n")
if mode == "stale":                                           # a split JSON changed after the audit
    json.dump(dict(SPLITS["unseen"], Extra={"Extra_idx_0": ["S001_R001_hybrid_IR.wav"]}), open(sp["unseen"]["json_abs"], "w"))
if mode == "map_byte":                                        # a depth map changed after the audit
    p = maps[sorted(maps)[0]]["path"]
    d = np.load(p)
    d[0, 0] += 1e-9
    np.save(p, d)
json.dump({"splits": {n: {"dataset_config": cfgs[n], "n_items": led["n_items"]} for n, led in zip(SPLITS, ledger)},
           "n_maps": len(maps), "custom_metadata_module": ar_md}, open(os.path.join(h, "audit_contract.json"), "w"))
PY
  ) || return 1
  ( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null )
  commit_clone "guard: tie audit ($1)"
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
# guard stub: memory.free -> GUARD_GPU_FREE (MiB, printf %b), exit GUARD_GPU_MEM_RC; index,uuid -> three
# fixed uuids (GPU 2 is another card), exit GUARD_GPU_UUID_RC; the compute-app query -> GUARD_GPU_APPS
# ('pid, uuid, MiB' lines, printf %b -- rC-F1: the v14 3-column query), exit GUARD_GPU_APPS_RC
# (GUARD_GPU_APPS_FAIL=1: no output, exit 1).
# A nonzero exit AFTER plausible output is the rA A3 case: it must still refuse.
# rB-F2 F2-2: ONLY the launcher's three strict gate queries are answered; any other query (e.g. a
# best-effort disclosure query) is logged to $H/nvsmi_unexpected and FAILS (exit 9).
H="$(dirname "$(dirname "$(readlink -f "$0")")")"
case "$*" in
  "-i "[01]" --query-gpu=memory.free --format=csv,noheader,nounits")
    printf '%b\n' "${GUARD_GPU_FREE:-40000}"; exit "${GUARD_GPU_MEM_RC:-0}" ;;
  "--query-gpu=index,uuid --format=csv,noheader")
    printf '0, GPU-guard-0\n1, GPU-guard-1\n2, GPU-guard-2\n'; exit "${GUARD_GPU_UUID_RC:-0}" ;;
  "--query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits")
    [ "${GUARD_GPU_APPS_FAIL:-0}" = 1 ] && exit 1
    printf '%b' "${GUARD_GPU_APPS:-}"; exit "${GUARD_GPU_APPS_RC:-0}" ;;
  *) printf '%s\n' "$*" >> "$H/nvsmi_unexpected"; echo "guard stub: unexpected nvidia-smi query"; exit 9 ;;
esac
STUB
cat > "$H/fake_train.sh" <<'FAKE'
#!/usr/bin/env bash
# guard fake trainer: prints what a real run prints and writes what it writes, per GUARD_SCENARIO
set -euo pipefail
declare -A A
while [ $# -gt 0 ]; do case "$1" in --*) A["${1#--}"]="${2:-}"; shift 2 ;; *) shift ;; esac; done
sc="${GUARD_SCENARIO:-ok}"; steps="${A[max-steps]}"; cad="${A[checkpoint-every]}"; rate="${GUARD_RATE:-3.9}"
ns="${A[save-dir]}/${A[name]}/${A[experiment-name]}/checkpoints"
stamp() { local t; t="$(python -c "print(int(2 + $1 * $rate))")"
          if [ "$t" -ge 3600 ]; then printf '%d:%02d:%02d' $((t / 3600)) $((t % 3600 / 60)) $((t % 60))
          else printf '%02d:%02d' $((t / 60)) $((t % 60)); fi; }
[ "$sc" = no_ranks ] || echo "All distributed processes registered. Starting with 2 processes"
[ "$sc" = no_banner ] || echo "canonicalize ENABLED img_w=512 tiebreak=image"
if [ "$steps" -le 25 ]; then ns_list="$(seq 1 "$steps")"; else ns_list="1 10 2500 20000 $steps"; fi
for n in $ns_list; do
  loss=0.123; [ "$sc" = nan ] && [ "$n" = "$steps" ] && loss=nan; [ "$sc" = text_loss ] && loss=abc
  post=", train/loss=${loss}"; [ "$sc" = no_loss ] && post=""
  [ "$sc" = no_stamps ] || printf '\rEpoch 0:   0%%|          | %s/4550 [%s<1:00:00,  %ss/it, v_num=ab12%s]' "$n" "$(stamp "$n")" "$rate" "$post"
done
echo
case "$sc" in   # src/training/diffusion.py on_fit_end, rank 0 (r3b-F1 fix 2)
  no_latency) ;;
  bad_latency) echo "canonicalize latency rank0 batches=0 samples=0 total_s=0.000000 mean_ms_per_batch=0.000" ;;
  *) echo "canonicalize latency rank0 batches=${steps} samples=$((steps * 32)) total_s=0.087500 mean_ms_per_batch=3.500" ;;
esac
if [ "$cad" -le "$steps" ] || [ "$sc" = smoke_ckpt ]; then
  mkdir -p "$ns"
  for ((s = cad; s <= steps; s += cad)); do
    [ "$sc" = no_endpoint_ckpt ] && [ "$s" = "$steps" ] && continue
    printf 'ckpt %s\n' "$s" > "$ns/epoch=$((s / 4550))-step=$s.ckpt"
  done
  [ "$sc" = smoke_ckpt ] && printf 'x\n' > "$ns/epoch=0-step=10.ckpt"
  [ "$sc" = foreign ] && printf 'x\n' > "$ns/notes.txt"
  [ "$sc" = wrong_cadence ] && printf 'x\n' > "$ns/epoch=0-step=1234.ckpt"
  [ "$sc" = stale ] && touch -d '2000-01-01 00:00:00' "$ns/epoch=0-step=2500.ckpt"
fi
[ "$sc" = no_endpoint ] || printf '`Trainer.fit` stopped: `max_steps=%s` reached.' "$steps"
exit "${GUARD_TRAIN_RC:-0}"
FAKE
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
  want "$name" || return 0
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
           '"$EXPREL"'/FLAC_AR_CANON.json worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py \
           worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BV.json \
           worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json \
           '"$EXPREL"'/depth_canon_tie_audit.json '"$EXPREL"'/depth_canon_tie_audit.json.sha256; do
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

echo "--- I. the interpreter (r3b-F1 fix 6): the flac env for every gate and for training ---"
mkdir -p "$H/otherpy"; printf '#!/usr/bin/env bash\necho "NOT THE FLAC PYTHON"; exit 1\n' > "$H/otherpy/python"
chmod +x "$H/otherpy/python"
case_rc "I1 production mode with another python first on PATH" 2 "INTERPRETER GATE FAILED" \
  MODE=SMOKE DRY_RUN=1 DEPTH_CANON_HARNESS= PATH="$H/otherpy:$PATH"
case_rc "I2 harness mode with a python outside the harness first on PATH" 2 "INTERPRETER GATE FAILED" \
  MODE=SMOKE DRY_RUN=1 PATH="$(dirname "$REAL_PY"):$BIN:$PATH"
case_rc "I3 the harness stub is accepted and named" 0 "interpreter OK: $BIN/python" MODE=SMOKE DRY_RUN=1

echo "--- K. HEAD pin ---"
case_rc "K1 EXPECT_HEAD unset"   2 "EXPECT_HEAD must be the full 40-hex"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD=
case_rc "K2 short EXPECT_HEAD"   2 "EXPECT_HEAD must be the full 40-hex"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD="$(git -C "$R" rev-parse --short HEAD)"
case_rc "K3 wrong EXPECT_HEAD"   2 "HEAD PIN FAILED"  MODE=SMOKE DRY_RUN=1 EXPECT_HEAD=0000000000000000000000000000000000000000
case_rc "K4 the pin passes"      0 "HEAD pin OK: $(git -C "$R" rev-parse HEAD)"  MODE=SMOKE DRY_RUN=1

echo "--- F. the ACTUAL train.py argv (whole tokens, each flag exactly once) ---"
launch MODE=FULL DRY_RUN=1; FULL_ARGV="$(argv_of "$OUT")"
launch MODE=SMOKE DRY_RUN=1; SMOKE_ARGV="$(argv_of "$OUT")"
expect "F1 argv starts with the pinned interpreter + train.py" bash -c '[[ "$1" == "$2 train.py "* ]]' _ "$FULL_ARGV" "$BIN/python"
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
case_rc "M6 a committed change with its regenerated manifest is a new valid pin" 0 "pin manifest OK: 48 files" MODE=SMOKE DRY_RUN=1
printf '\n' >> "$R/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
case_rc "M7 an edit to a config the DINOv3 gate opens (FLAC_AR_BF.json) is caught" 2 "PIN MANIFEST FAILED: content" MODE=SMOKE DRY_RUN=1
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

echo "--- T. SMOKE and FULL need the complete PASSING tie audit, through ONE gate (rB B1 / rA A1) ---"
AUDIT_NEED="TIE AUDIT GATE FAILED"
git -C "$R" rm -q "$EXPREL/depth_canon_tie_audit.json" "$EXPREL/depth_canon_tie_audit.json.sha256"; commit_clone "guard: no audit"
case_rc "T1 no tie audit (a pinned file): the pin gate refuses" 2 "PIN MANIFEST FAILED" MODE=FULL DRY_RUN=1
case_rc "T1b SMOKE refuses without it too"        2 "PIN MANIFEST FAILED" MODE=SMOKE DRY_RUN=1
write_audit limit;     case_rc "T2 a --limit audit (REFUSED_SUBSET)" 2 "$AUDIT_NEED" MODE=FULL DRY_RUN=1
write_audit stale;     case_rc "T3 a split JSON changed after the audit" 2 "split JSONs" MODE=FULL DRY_RUN=1
write_audit tamper;    case_rc "T4 a tampered sidecar (re-pinned)" 2 "sha256 sidecar" MODE=FULL DRY_RUN=1
write_audit survivor;  case_rc "T5 an audit that found a survivor" 2 "$AUDIT_NEED" MODE=FULL DRY_RUN=1
write_audit error;     case_rc "T6 an audit with an unprocessed map" 2 "$AUDIT_NEED" MODE=FULL DRY_RUN=1
write_audit script;    case_rc "T7 an audit by another audit script" 2 "audit script" MODE=FULL DRY_RUN=1
write_audit counts;    case_rc "T8 item counts that are not the census" 2 "inputs" MODE=FULL DRY_RUN=1
write_audit dup;       case_rc "T10 duplicated rows (sidecar recomputed)" 2 "duplicate" MODE=FULL DRY_RUN=1
write_audit arbitrary; case_rc "T11 arbitrary row paths (sidecar recomputed)" 2 "rows are not the" MODE=FULL DRY_RUN=1
write_audit map_byte;  case_rc "T12 a depth map changed after the audit" 2 "changed since the audit" MODE=FULL DRY_RUN=1
write_audit untagged;  case_rc "T13 an untagged report in harness mode" 2 "harness-tagged" MODE=FULL DRY_RUN=1
fixture_selftest() {   # rB-F2 F2-3 (b): a fixture that must be a complete PASS but is not fails LOUDLY, naming why
  local out
  if out="$(write_audit ok_but_error 2>&1)"; then echo "write_audit ok_but_error returned 0"; return 1; fi
  grep -qF "GUARD FIXTURE FAILED" <<<"$out" && grep -qF "INCOMPLETE" <<<"$out" && grep -qF "non-finite" <<<"$out" \
    || { echo "no loud reason: ${out}"; return 1; }
}
expect "T15 a write_audit fixture that must PASS but came out INCOMPLETE fails loudly with the row error" fixture_selftest
write_audit ok
case_rc "T14 the harness report in PRODUCTION mode is refused" 2 "a harness tag is never production" \
  MODE=SMOKE DRY_RUN=1 DEPTH_CANON_HARNESS= PATH="$(dirname "$REAL_PY"):$BIN:$PATH"
case_rc "T9 the complete PASS audit is accepted"  0 "tie audit OK" MODE=SMOKE DRY_RUN=1

echo "--- E. FULL needs a PASSING smoke evidence of this HEAD + manifest (+ harness flag) ---"
write_evidence() {   # write_evidence <name> '<python dict of overrides>' (DROP deletes a key; cot(**kw) = a co-tenant record)
  ( cd "$R" && python - "$EXPREL" "$1" "$2" <<'PY'
import hashlib, json, subprocess, sys
exp, name, over = sys.argv[1:4]
DROP = object()                   # rC-F2: an override value that DELETES the key
COT = {"gpu": 0, "pid": 4242, "used_mib": 30000, "cwd": "/tmp/guard_cotenant"}   # a valid co-tenant record
def cot(**kw):
    return {k: v for k, v in dict(COT, **kw).items() if v is not DROP}
log = f"{exp}/depth_canon_guard_{name}_smoke_train.log"
open(log, "w").write("synthetic smoke log\n")
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sha = lambda *ps: hashlib.sha256(b"".join(open(p, "rb").read() for p in ps)).hexdigest()
man = sha(f"{exp}/depth_canon_pins.sha256")
ev = {"schema_version": 2, "kind": "depth_canon_smoke_evidence", "verdict": "PASS", "head": head,
      "manifest_sha256": man, "launcher_sha256": sha(f"{exp}/depth_canon_launch.sh"),
      "make_pins_sha256": sha(f"{exp}/depth_canon_make_pins.sh"),
      "contract_sha256": sha(f"{exp}/FLAC_AR_CANON.json",
                             "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"),
      "audit_sha256": sha(f"{exp}/depth_canon_tie_audit.json"),
      "audit_gate_sha256": sha(f"{exp}/depth_canon_audit_gate.py"),
      "harness": True, "config_contract_ok": True, "banner_seen": True,
      "endpoint_seen": True, "ranks_seen": True, "nonfinite_seen": False, "smoke_checkpoints": 0,
      "finite_loss_seen": True, "canonicalization_latency_ms": 3.5,
      "gpu_policy": "shared", "min_gpu_free_mib": 12288, "max_projected_hours": 120,
      "free_mib_at_launch": {"0": 40000, "1": 40000},
      "co_tenants_at_launch": [], "log": log, "log_sha256": hashlib.sha256(open(log, "rb").read()).hexdigest()}
ev.update(eval(over))
for k in [k for k, v in ev.items() if v is DROP]:
    del ev[k]
json.dump(ev, open(f"{exp}/depth_canon_smoke_evidence_{name}.json", "w"))
PY
  )
}
clear_evidence() { rm -f "$R/$EXPREL"/depth_canon_smoke_evidence_*.json; }
NEED="FULL requires a PASSING smoke evidence"
case_rc "E1 no evidence"                    2 "$NEED" MODE=FULL DRY_RUN=1
expect "E1b ... and the line after the headline says there is no candidate (rC-F2)" \
  grep -qxF "evidence check: no depth_canon_smoke_evidence_*.json in $EXPREL" <<<"$OUT"
write_evidence e2 '{"launcher_sha256": "1" * 64}'
case_rc "E2 evidence of another launcher"   2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
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
write_evidence e10 '{"schema_version": 1}'
case_rc "E10 v13 (schema 1) evidence is refused: FULL needs the v14 schema (rC-F1)" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
# rC-F2 1 (Codex rC HIGH, C3): FULL enforces the schema-2 GPU provenance. Each case takes E9's PASSING schema-2
# evidence and changes ONE thing: FULL refuses (rc 2, the headline once) and the NEXT line names the key + the value seen.
EVWHY="evidence check: newest candidate depth_canon_smoke_evidence_ev.json refused: "
ev_refused() {   # ev_refused <name> '<key> = <value seen>' '<python dict of overrides>'
  local name="$1" why="$2" nxt; want "$name" || return 0
  clear_evidence; write_evidence ev "$3"
  launch MODE=FULL DRY_RUN=1
  nxt="$(grep -A1 -F -- "$NEED" <<<"$OUT" | sed -n 2p || true)"
  if [ "$RC" = 2 ] && [ "$(grep -cF -- "$NEED" <<<"$OUT" || true)" = 1 ] && [[ "$nxt" == "${EVWHY}${why} (want "* ]]; then
    ok "$name (rc=$RC)"
  else bad "$name" "want rc=2 + '$NEED' + next line '${EVWHY}${why} (want ...'; got rc=$RC; next: ${nxt:-$(tail -1 <<<"$OUT")}"; fi
  clear_evidence
}
ev_refused "E11a gpu_policy deleted"            'gpu_policy = <missing>'            '{"gpu_policy": DROP}'
ev_refused "E11b min_gpu_free_mib deleted"      'min_gpu_free_mib = <missing>'      '{"min_gpu_free_mib": DROP}'
ev_refused "E11c max_projected_hours deleted"   'max_projected_hours = <missing>'   '{"max_projected_hours": DROP}'
ev_refused "E11d free_mib_at_launch deleted"    'free_mib_at_launch = <missing>'    '{"free_mib_at_launch": DROP}'
ev_refused "E11e co_tenants_at_launch deleted"  'co_tenants_at_launch = <missing>'  '{"co_tenants_at_launch": DROP}'
ev_refused "E12 gpu_policy idle (not the pinned shared)" 'gpu_policy = "idle"'      '{"gpu_policy": "idle"}'
ev_refused "E13 the v13 floor 23040"            'min_gpu_free_mib = 23040'          '{"min_gpu_free_mib": 23040}'
ev_refused "E14 the v13 cap 60 h"               'max_projected_hours = 60'          '{"max_projected_hours": 60}'
ev_refused "E15 free_mib_at_launch null (the writer's no-gate form)" 'free_mib_at_launch = null' '{"free_mib_at_launch": None}'
ev_refused "E16 a card missing"                 'free_mib_at_launch = {"0": 20000}' '{"free_mib_at_launch": {"0": 20000}}'
ev_refused "E17 GPU 1 below the floor"          'free_mib_at_launch["1"] = 12287'   '{"free_mib_at_launch": {"0": 20000, "1": 12287}}'
ev_refused "E18 a free value as a string"       'free_mib_at_launch["0"] = "20000"' '{"free_mib_at_launch": {"0": "20000", "1": 20000}}'
ev_refused "E19 a co-tenant record without cwd" 'co_tenants_at_launch[0].cwd = <missing>' '{"co_tenants_at_launch": [cot(cwd=DROP)]}'
ev_refused "E20 a co-tenant pid 0"              'co_tenants_at_launch[0].pid = 0'   '{"co_tenants_at_launch": [cot(pid=0)]}'
ev_refused "E21 a co-tenant on GPU 2"           'co_tenants_at_launch[0].gpu = 2'   '{"co_tenants_at_launch": [cot(gpu=2)]}'
ev_refused "E22 a co-tenant used_mib -1"        'co_tenants_at_launch[0].used_mib = -1' '{"co_tenants_at_launch": [cot(used_mib=-1)]}'
ev_refused "E23 a co-tenant with an empty cwd"  'co_tenants_at_launch[0].cwd = ""'  '{"co_tenants_at_launch": [cot(cwd="")]}'
ev_refused "E24 a bool smuggled as the pid"     'co_tenants_at_launch[0].pid = true' '{"co_tenants_at_launch": [cot(pid=True)]}'
write_evidence e25 '{"co_tenants_at_launch": [cot(), cot(gpu=1, pid=4243, used_mib=0, cwd="?")]}'
case_rc "E25 valid co-tenant records (incl. cwd '?', 0 MiB) are accepted" 0 \
  "smoke evidence: $EXPREL/depth_canon_smoke_evidence_e25.json" MODE=FULL DRY_RUN=1; clear_evidence
clear_evidence; printf 'not json\n' > "$R/$EXPREL/depth_canon_smoke_evidence_ev.json"
case_rc "E26 an unreadable evidence file is refused" 2 "${EVWHY}not a JSON object" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence e9 '{}'
case_rc "E9 genuine evidence accepted"      0 "smoke evidence: $EXPREL/depth_canon_smoke_evidence_e9.json" MODE=FULL DRY_RUN=1

echo "--- B. the SMOKE->FULL binding is content-scoped (r3b-F1 fix 9): HEAD is provenance only ---"
bind_case() {   # bind_case <name> <want_rc> <want_text> <change>: evidence, then the change, then FULL
  make_clone >/dev/null; write_evidence b '{}'
  ( cd "$R" && eval "$4" ) >/dev/null
  case_rc "$1" "$2" "$3" MODE=FULL DRY_RUN=1
}
if want B; then
bind_case "B1 a record-only commit between SMOKE and FULL is accepted" 0 "smoke evidence:" \
  'echo "- note" >> "$EXPREL/depth_canon_worklog.md" && commit_clone "records only"'
bind_case "B2 a training-closure change (+ its regenerated manifest) needs a new SMOKE" 2 "$NEED" \
  'printf "\n# edit\n" >> src/data/utils.py && bash "$EXPREL/depth_canon_make_pins.sh" --write && commit_clone "src"'
bind_case "B3 a committed launcher change needs a new SMOKE" 2 "$NEED" \
  'printf "\n# edit\n" >> "$EXPREL/depth_canon_launch.sh" && commit_clone "launcher"'
bind_case "B4 a new (still qualifying) tie audit needs a new SMOKE" 2 "$NEED" \
  'printf " \n" >> "$EXPREL/depth_canon_tie_audit.json" && sha256sum "$EXPREL/depth_canon_tie_audit.json" | sed "s#  .*/#  #" > "$EXPREL/depth_canon_tie_audit.json.sha256" && bash "$EXPREL/depth_canon_make_pins.sh" --write && commit_clone "audit"'
bind_case "B5 a committed change to the audit gate needs a new SMOKE" 2 "$NEED" \
  'printf "\n# edit\n" >> "$EXPREL/depth_canon_audit_gate.py" && commit_clone "gate"'
make_clone >/dev/null
fi
write_evidence e9 '{}'   # the genuine evidence the N / S / G sections run against (E9's state)

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

echo "--- G. GPU memory (never under DRY_RUN), W&B (never in harness), DINOv3 pin ---"
case_rc "G1 a busy card refuses (rc 7: the watcher's transient class)" 7 "GPU GATE FAILED: GPU 0 free 1000 MiB" MODE=SMOKE GUARD_GPU_FREE=1000
case_rc "G2 the GPU gate is skipped under DRY_RUN" 0 "GPU memory gate SKIPPED under DRY_RUN" MODE=SMOKE DRY_RUN=1 GUARD_GPU_FREE=1000
case_rc "G3 the W&B identity gate is skipped in harness mode" 0 "W&B identity gate SKIPPED (harness)" MODE=SMOKE
mkdir -p "$H/emptyhub"
case_rc "G4 a missing DINOv3 snapshot refuses" 2 "DINOv3/init-identity GATE FAILED" MODE=SMOKE DRY_RUN=1 HF_HUB_CACHE="$H/emptyhub"
case_rc "G5 the gate imports THIS checkout's src" 0 "src imported from $R/src" MODE=SMOKE DRY_RUN=1

echo "--- U. v14: shared policy - co-tenants disclosed, memory floor decides; strict queries kept (rC-F1; rA A3) ---"
COT="GPU occupancy gate: policy=shared -"
case_rc "U1 a co-tenant on GPU 0 is disclosed, not refused (policy=shared)" 0 \
  "$COT 1 co-tenant compute process(es) on GPU 0/1 disclosed, not refused:" MODE=SMOKE GUARD_GPU_APPS='4242, GPU-guard-0, 32256\n'
expect "U1b ... naming its pid, card, uuid, MiB and cwd" grep -qE '^  pid 4242 on GPU 0 \(GPU-guard-0\): 32256 MiB, cwd .+$' <<<"$OUT"
case_rc "U2 a co-tenant on GPU 1 is named with its card, MiB and cwd" 0 "pid 4243 on GPU 1 (GPU-guard-1): 4806 MiB, cwd " \
  MODE=SMOKE GUARD_GPU_APPS='4243, GPU-guard-1, 4806\n'
case_rc "U3 a failed compute-process query refuses" 7 "query-compute-apps" \
  MODE=SMOKE GUARD_GPU_APPS_FAIL=1
case_rc "U4 a process on another (known) card is not a co-tenant" 0 "$COT no compute process on GPU 0 or GPU 1" \
  MODE=SMOKE GUARD_GPU_APPS='4244, GPU-guard-2, 100\n'
rm -f "$H/nvsmi_unexpected"
case_rc "U12 the co-tenancy disclosure logs the VALIDATED answers (rB-F2 F2-2)" 0 \
  "GPU 1 GPU-guard-1: 40000 MiB free, 0 co-tenant process(es)" MODE=SMOKE GUARD_GPU_APPS='4244, GPU-guard-2, 100\n'
expect "U12b ... and names the compute process seen on another card" grep -qF "pid 4244 on GPU 2 (GPU-guard-2)" <<<"$OUT"
case_rc "U12c the per-card co-tenant counts" 0 "GPU 0 GPU-guard-0: 40000 MiB free, 1 co-tenant process(es)" \
  MODE=SMOKE GUARD_GPU_APPS='4245, GPU-guard-1, 100\n4246, GPU-guard-1, 200\n4247, GPU-guard-0, 300\n'
expect "U12d ... GPU 1 holds two, three in all" bash -c 'grep -qF "GPU 1 GPU-guard-1: 40000 MiB free, 2 co-tenant process(es)" <<<"$1" &&
  grep -qF "$2 3 co-tenant compute process(es) on GPU 0/1 disclosed, not refused:" <<<"$1"' _ "$OUT" "$COT"
expect "U13 the launcher issues no nvidia-smi query beyond the strict gate (a failing disclosure query is impossible)" \
  test ! -e "$H/nvsmi_unexpected"
case_rc "U6 a process on an UNKNOWN uuid refuses" 7 "unknown uuid" \
  MODE=SMOKE GUARD_GPU_APPS='4245, GPU-guard-9, 100\n'
case_rc "U7 a numeric memory answer with a nonzero exit refuses" 7 "exited 3" MODE=SMOKE GUARD_GPU_MEM_RC=3
case_rc "U8 two memory rows for one card refuse" 7 "not exactly one number" MODE=SMOKE GUARD_GPU_FREE='40000\n40000'
case_rc "U9 a malformed compute-app row refuses" 7 "malformed compute-app row" MODE=SMOKE GUARD_GPU_APPS='garbage\n'
case_rc "U10 a plausible uuid map with a nonzero exit refuses" 7 "exited 4" MODE=SMOKE GUARD_GPU_UUID_RC=4
case_rc "U11 plausible compute-app rows with a nonzero exit refuse" 7 "exited 5" \
  MODE=SMOKE GUARD_GPU_APPS='4244, GPU-guard-2, 100\n' GUARD_GPU_APPS_RC=5
case_rc "U14 a co-tenant AND free memory below the floor: the floor refuses (rc 7)" 7 \
  "GPU GATE FAILED: GPU 0 free 12287 MiB < 12288 MiB (12 GiB) - abort" MODE=SMOKE GUARD_GPU_FREE=12287 GUARD_GPU_APPS='4242, GPU-guard-0, 30000\n'
case_rc "U14b exactly the floor beside a co-tenant passes" 0 "GPU memory gate OK: GPU 0 and GPU 1 each >= 12288 MiB free" \
  MODE=SMOKE GUARD_GPU_FREE=12288 GUARD_GPU_APPS='4242, GPU-guard-0, 30000\n'
u15() {   # the v14 pinned literals, equal in the launcher (clone = the overlaid working tree) and the watcher
  local l="$R/$EXPREL/depth_canon_launch.sh" w="$SRC_REPO/$EXPREL/depth_canon_autolaunch.sh" n v
  grep -qE '^MIN_GPU_FREE_MIB=12288([^0-9]|$)' "$l" && grep -qE '^MAX_PROJECTED_HOURS=120([^0-9]|$)' "$l" \
    && grep -qx 'GPU_OCCUPANCY_POLICY=shared' "$l" || { echo "launcher literals"; return 1; }
  grep -qE '^MIN_GPU_FREE_MIB=12288([^0-9]|$)' "$w" && grep -qx 'GPU_OCCUPANCY_POLICY=shared' "$w" || { echo "watcher literals"; return 1; }
  for n in MIN_GPU_FREE_MIB GPU_OCCUPANCY_POLICY; do
    v="$(grep -E "^${n}=" "$l" | sed -E "s/^${n}=([^[:space:]]*).*/\1/")"
    [ "$(grep -cE "^${n}=" "$l")" = 1 ] && [ "$(grep -cE "^${n}=" "$w")" = 1 ] \
      && [ "$v" = "$(grep -E "^${n}=" "$w" | sed -E "s/^${n}=([^[:space:]]*).*/\1/")" ] || { echo "${n} differs"; return 1; }
  done
}
expect "U15 the pinned literals 12288 / 120 / shared, mirrored by the watcher" u15
case_rc "U16 a 2-field compute-app row (the v13 query's shape) is malformed" 7 "malformed compute-app row" \
  MODE=SMOKE GUARD_GPU_APPS='4242, GPU-guard-0\n'
case_rc "U16b a non-numeric used_memory refuses" 7 "malformed compute-app row" MODE=SMOKE GUARD_GPU_APPS='4242, GPU-guard-0, [N/A]\n'
set_policy() {   # set_policy <literal>: the clone's launcher with another policy literal, committed
  sed -i "s/^GPU_OCCUPANCY_POLICY=shared\$/GPU_OCCUPANCY_POLICY=$1/" "$R/$EXPREL/depth_canon_launch.sh"
  grep -qx "GPU_OCCUPANCY_POLICY=$1" "$R/$EXPREL/depth_canon_launch.sh" && commit_clone "guard: policy $1"
}
if want U; then
  u15b() {   # rC-F2 3 (Codex rC NIT, C5): U15 must read the CLONE's committed watcher -- a clone watcher whose floor
             # literal differs (committed in the clone only; the source repo untouched) fails it, on the watcher half
    sed -i 's/^MIN_GPU_FREE_MIB=12288/MIN_GPU_FREE_MIB=23040/' "$R/$EXPREL/depth_canon_autolaunch.sh"
    grep -qE '^MIN_GPU_FREE_MIB=23040([^0-9]|$)' "$R/$EXPREL/depth_canon_autolaunch.sh" || { echo "no floor literal to flip"; return 1; }
    commit_clone "guard: watcher floor 23040"
    local msg; msg="$(u15)" && { echo "U15 still passed: it did not read the clone's watcher"; return 1; }
    [ "$msg" = "watcher literals" ] || { echo "U15 failed for another reason: ${msg}"; return 1; }
  }
  expect "U15b U15 reads the clone's committed watcher: a clone watcher with floor 23040 fails it (rC-F2)" u15b
  if set_policy idle; then
    case_rc "U17 GPU_OCCUPANCY_POLICY=idle restores the v10 refusal (rc 7)" 7 "GPU OCCUPIED: foreign compute process(es) on GPU 0/1" \
      MODE=SMOKE GUARD_GPU_APPS='4242, GPU-guard-0, 100\n'
    case_rc "U17b idle: a process on another card does not block" 0 "GPU occupancy gate OK: no compute process on GPU 0 or GPU 1" \
      MODE=SMOKE GUARD_GPU_APPS='4244, GPU-guard-2, 100\n'
  else bad "U17 / U17b" "the launcher has no 'GPU_OCCUPANCY_POLICY=shared' line to flip"; fi
  make_clone >/dev/null
  if set_policy relaxed; then
    case_rc "U18 any other policy literal is a CONFIG ERROR (rc 2)" 2 "CONFIG ERROR: GPU_OCCUPANCY_POLICY must be idle or shared" \
      MODE=SMOKE DRY_RUN=1
  else bad "U18" "the launcher has no 'GPU_OCCUPANCY_POLICY=shared' line to flip"; fi
  make_clone >/dev/null
fi
clear_evidence; rm -rf "$H/save"
case_rc "U5 the occupancy gate is skipped under DRY_RUN" 0 "GPU memory gate SKIPPED under DRY_RUN" \
  MODE=SMOKE DRY_RUN=1 GUARD_GPU_APPS='4242, GPU-guard-0, 100\n'

echo "--- X. the preflight cannot satisfy the post-run scans ---"
launch MODE=SMOKE DRY_RUN=1; PRE="$OUT"
expect "X1 the preflight never prints the banner as a whole line" bash -c '! tr "\r" "\n" <<<"$1" | grep -qxF "$2"' _ "$PRE" "canonicalize ENABLED img_w=512 tiebreak=image"
expect "X2 the preflight never matches the non-finite scan" bash -c '! grep -qiE "\b(nan|inf)\b" <<<"$1"' _ "$PRE"
expect "X3 the scan does match a non-finite loss (non-vacuous)" bash -c 'grep -qiE "\b(nan|inf)\b" <<<"x, train/loss=nan]"'

echo "--- R. SMOKE: the real launch path on the fake trainer ---"
clear_evidence; rm -rf "$H/save"
case_rc "R1 a healthy smoke passes"          0 "SMOKE VERDICT: PASS" MODE=SMOKE
EV="$(ls "$R/$EXPREL"/depth_canon_smoke_evidence_*.json 2>/dev/null | tail -1 || true)"
expect "R1b it wrote evidence bound to HEAD, manifest and its final log" bash -c '
  cd "$1" && python - "$2" <<PY
import hashlib, json, subprocess, sys
ev = json.load(open(sys.argv[1]))
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
man = hashlib.sha256(open("$3/depth_canon_pins.sha256", "rb").read()).hexdigest()
assert ev["verdict"] == "PASS" and ev["head"] == head and ev["manifest_sha256"] == man, ev
assert ev["harness"] is True and ev["smoke_checkpoints"] == 0 and ev["max_projected_hours"] == 120
assert ev["schema_version"] == 2 and ev["gpu_policy"] == "shared" and type(ev["min_gpu_free_mib"]) is int
assert ev["min_gpu_free_mib"] == 12288 and ev["free_mib_at_launch"] == {"0": 40000, "1": 40000}, ev
assert ev["co_tenants_at_launch"] == [], ev
fm = ev["free_mib_at_launch"]   # rC-F2: the typed keys FULL now requires (R1c = the launcher round trip)
assert type(ev["max_projected_hours"]) is int and type(fm) is dict and sorted(fm) == ["0", "1"], ev
assert all(type(fm[g]) is int and fm[g] >= ev["min_gpu_free_mib"] for g in fm), ev
assert abs(ev["steady_state_s_per_step"] - 3.9) < 0.1 and 40 < ev["projection_h"] < 46, ev
assert hashlib.sha256(open(ev["log"], "rb").read()).hexdigest() == ev["log_sha256"]
PY' _ "$R" "$EV" "$EXPREL"
case_rc "R1c FULL accepts that evidence (the writer -> FULL predicate round trip)" 0 "smoke evidence: $EXPREL/$(basename "$EV")" MODE=FULL DRY_RUN=1
case_rc "R1d a SMOKE beside a co-tenant on GPU 0 passes (rC-F1)" 0 "SMOKE VERDICT: PASS" \
  MODE=SMOKE GUARD_GPU_APPS='4242, GPU-guard-0, 30000\n4244, GPU-guard-2, 100\n'
EV2="$(ls "$R/$EXPREL"/depth_canon_smoke_evidence_*.json 2>/dev/null | tail -1 || true)"
expect "R1e its evidence discloses the co-tenant (pid, gpu 0, used_mib, cwd) and the free memory" bash -c '
  [ "$1" != "$2" ] && "$3" - "$1" <<PY
import json, sys
ev = json.load(open(sys.argv[1]))
cot = ev["co_tenants_at_launch"]
assert ev["verdict"] == "PASS" and ev["gpu_policy"] == "shared" and len(cot) == 1, ev
c = cot[0]
assert sorted(c) == ["cwd", "gpu", "pid", "used_mib"] and type(c["cwd"]) is str and c["cwd"], c
assert (type(c["gpu"]), type(c["pid"]), type(c["used_mib"])) == (int, int, int), c
assert (c["gpu"], c["pid"], c["used_mib"]) == (0, 4242, 30000), c
assert ev["free_mib_at_launch"] == {"0": 40000, "1": 40000}, ev
PY' _ "$EV2" "$EV" "$REAL_PY"
case_rc "R1f FULL accepts the co-tenant SMOKE's evidence" 0 "smoke evidence: $EXPREL/$(basename "$EV2")" MODE=FULL DRY_RUN=1
[ -n "$EV2" ] && [ "$EV2" != "$EV" ] && rm -f -- "${EV2:?}"   # R10 / L1 below count and read R1's evidence only
case_rc "R2 no banner"                       3 "TREATMENT BANNER NOT FOUND" MODE=SMOKE GUARD_SCENARIO=no_banner
case_rc "R3 no endpoint marker"              6 "ENDPOINT NOT REACHED" MODE=SMOKE GUARD_SCENARIO=no_endpoint
case_rc "R4 no two-rank start"               6 "TWO-RANK DDP START NOT FOUND" MODE=SMOKE GUARD_SCENARIO=no_ranks
case_rc "R5 a non-finite loss"               4 "NON-FINITE VALUES" MODE=SMOKE GUARD_SCENARIO=nan
case_rc "R6 too slow: 12 s/step (133 h > 120 h)" 4 "SMOKE VERDICT: FAIL - projected 133.3 h > MAX_PROJECTED_HOURS=120" MODE=SMOKE GUARD_RATE=12
case_rc "R6b 6 s/step (66.7 h) passes under the v14 120 h cap" 0 "SMOKE VERDICT: PASS - projected 66.7 h <= 120 h" MODE=SMOKE GUARD_RATE=6
EV6="$(grep -oE 'smoke evidence \(PASS\): [^ ]+' <<<"$OUT" | sed 's/^smoke evidence (PASS): //' || true)"
[ -n "$EV6" ] && rm -f -- "${R:?}/${EV6:?}"   # R10 counts R1's PASS evidence only
case_rc "R7 no rate stamps"                  4 "SMOKE VERDICT: FAIL - the projection could not be computed" MODE=SMOKE GUARD_SCENARIO=no_stamps
case_rc "R8 a smoke that wrote a checkpoint" 5 "SMOKE WROTE 1 CHECKPOINT" MODE=SMOKE GUARD_SCENARIO=smoke_ckpt
rm -rf "$H/save/FLAC_exp25_CANON_smoke"
case_rc "R9 train.py exited 7: remapped to the terminal 8 (rc 7 is the GPU gate's alone; Codex A2 HIGH 1)" 8 \
  "is remapped to the terminal 8" MODE=SMOKE GUARD_TRAIN_RC=7
rm -rf "$H/save/FLAC_exp25_CANON_smoke"
case_rc "R9b another train.py failure passes through unchanged" 3 "train.py exited rc=3" MODE=SMOKE GUARD_TRAIN_RC=3
expect "R10 only R1 left PASS evidence" bash -c '
  cd "$1" && [ "$(grep -l "\"verdict\": \"PASS\"" "$2"/depth_canon_smoke_evidence_*.json | wc -l)" = 1 ]' _ "$R" "$EXPREL"

echo "--- L. SMOKE proves a finite loss and a measured canonicalization latency (r3b-F1 fix 2) ---"
expect "L1 R1's evidence carries the latency and a finite loss" bash -c '
  "$2" - "$1" <<PY
import json, math, sys
ev = json.load(open(sys.argv[1]))
assert ev["verdict"] == "PASS" and ev["finite_loss_seen"] is True and ev["last_train_loss"] == 0.123, ev
lat = ev["canonicalization_latency_ms"]
assert type(lat) is float and math.isfinite(lat) and lat == 3.5, ev
assert ev["canonicalization_latency"] == {"batches": 25, "samples": 800, "total_s": 0.0875, "mean_ms_per_batch": 3.5}
PY' _ "$R/$EXPREL/$(basename "$EV")" "$REAL_PY"
case_rc "L2 no latency line"        4 "SMOKE VERDICT: FAIL - the 'canonicalize latency" MODE=SMOKE GUARD_SCENARIO=no_latency
case_rc "L3 an invalid latency"     4 "SMOKE VERDICT: FAIL - the 'canonicalize latency" MODE=SMOKE GUARD_SCENARIO=bad_latency
case_rc "L4 no training loss"       4 "SMOKE VERDICT: FAIL - no parseable finite training loss" MODE=SMOKE GUARD_SCENARIO=no_loss
case_rc "L5 a non-numeric loss"     4 "SMOKE VERDICT: FAIL - no parseable finite training loss" MODE=SMOKE GUARD_SCENARIO=text_loss
clear_evidence
write_evidence l6 '{"canonicalization_latency_ms": None}'
case_rc "L6 FULL refuses evidence without the latency" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence l7 '{"canonicalization_latency_ms": -1.0}'
case_rc "L7 FULL refuses a non-positive latency" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence
write_evidence l8 '{"finite_loss_seen": False}'
case_rc "L8 FULL refuses evidence without a finite loss" 2 "$NEED" MODE=FULL DRY_RUN=1; clear_evidence

echo "--- Z. FULL: banner, endpoint, non-finite, run-owned checkpoints, manifest ---"
full_case() { rm -rf "$H/save/FLAC_exp25_CANON"; case_rc "$@"; }
clear_evidence; write_evidence z_synthetic '{}'   # FULL's prerequisite (the real writer: cycle 12b)
full_case "Z1 a healthy FULL passes"         0 "checkpoint set OK: exactly the 16 cadence checkpoints" MODE=FULL
MAN="$(ls "$R/$EXPREL"/depth_canon_*_exp25_CANON_checkpoints.sha256 2>/dev/null | tail -1 || true)"
expect "Z1b the checkpoint manifest hashes all 16 files" bash -c '[ -n "$1" ] && [ "$(wc -l < "$1")" = 16 ] && sha256sum -c --quiet "$1"' _ "$MAN"
full_case "Z2 no banner"                     3 "TREATMENT BANNER NOT FOUND" MODE=FULL GUARD_SCENARIO=no_banner
full_case "Z3 the endpoint checkpoint is missing" 6 "ENDPOINT CHECKPOINT MISSING" MODE=FULL GUARD_SCENARIO=no_endpoint_ckpt
full_case "Z4 no endpoint marker"            6 "ENDPOINT NOT REACHED" MODE=FULL GUARD_SCENARIO=no_endpoint
full_case "Z5 a non-finite loss in FULL"     4 "NON-FINITE VALUES" MODE=FULL GUARD_SCENARIO=nan
full_case "Z6 a foreign file"                5 "CHECKPOINT SET REJECTED: foreign files" MODE=FULL GUARD_SCENARIO=foreign
full_case "Z7 a stale checkpoint"            5 "older than the launch marker" MODE=FULL GUARD_SCENARIO=stale
full_case "Z8 off-cadence checkpoint"        5 "cadence set" MODE=FULL GUARD_SCENARIO=wrong_cadence
full_case "Z9 FULL: train.py exited 7 is remapped to the terminal 8" 8 "is remapped to the terminal 8" MODE=FULL GUARD_TRAIN_RC=7

echo "--- Q. production path (opt-in GUARD_NAS_READONLY=1: READS the real NAS -- findmnt, df, stat; writes nothing) ---"
if [ "${GUARD_NAS_READONLY:-0}" = 1 ]; then
  launch_prod() {   # no harness, no stubs: the production save root and the real findmnt / df
    set +e
    OUT="$(cd "$R" && env DEPTH_CANON_HARNESS= EXPECT_HEAD="$(git -C "$R" rev-parse HEAD)" "$@" \
          bash "$EXPREL/depth_canon_launch.sh" 2>&1)"
    RC=$?
    set -e
  }
  make_clone >/dev/null   # the GENUINE committed audit (rA A1): restored from the source HEAD, real data read-only
  git -C "$R" checkout -q "$(git -C "$SRC_REPO" rev-parse HEAD)" -- "$EXPREL/depth_canon_tie_audit.json" \
    "$EXPREL/depth_canon_tie_audit.json.sha256"
  ( cd "$R" && bash "$EXPREL/depth_canon_make_pins.sh" --write >/dev/null ); commit_clone "guard: the genuine audit"
  ln -s "$SRC_REPO/AcousticRooms" "$R/AcousticRooms"
  launch_prod MODE=SMOKE DRY_RUN=1
  expect "Q0 production validates the genuine committed tie audit (all maps re-hashed)" \
    bash -c '[ "$1" = 0 ] && grep -qF "tie audit OK" <<<"$2"' _ "$RC" "$OUT"
  expect "Q1 production SMOKE dry run passes the real NAS gate" bash -c '[ "$1" = 0 ] && grep -qF "NAS gate OK: /media/diskstation/yixunhu/FLAC/checkpoints" <<<"$2"' _ "$RC" "$OUT"
  expect "Q1b production argv saves to the NAS pin" has_value "$(argv_of "$OUT")" --save-dir /media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon
  launch_prod MODE=FULL DRY_RUN=1
  expect "Q2 production FULL refuses without production evidence" bash -c '[ "$1" = 2 ] && grep -qF "harness=false" <<<"$2"' _ "$RC" "$OUT"
  for fv in "--save-dir /media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon" "--max-steps 40000" "--checkpoint-every 2500"; do
    expect "Q2b production FULL argv ${fv}" has_value "$(argv_of "$OUT")" "${fv%% *}" "${fv#* }"
  done
  make_clone >/dev/null   # rB-F2 F2-3 (a): back to a fresh harness clone + harness report (D1 needs it)
else
  skip "Q1-Q2 production-path dry runs" "set GUARD_NAS_READONLY=1 to read the real NAS"
fi

expect "W1 the launcher header documents W&B's repo-local wandb/ write target (r3b-F1 fix 10)" \
  bash -c 'sed -n "1,/^set -euo pipefail/p" "$1" | grep -q "repo-local .wandb/."' _ "$R/$EXPREL/depth_canon_launch.sh"
case_rc "D1 SMOKE dry run reaches the boundary" 0 "DRY_RUN: all gates passed" MODE=SMOKE DRY_RUN=1

echo
[ "$FILTERED" -eq 0 ] || echo "(${FILTERED} case(s) not run: GUARD_ONLY=${GUARD_ONLY})"
echo "=== ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ==="
[ "$FAIL" -eq 0 ]
