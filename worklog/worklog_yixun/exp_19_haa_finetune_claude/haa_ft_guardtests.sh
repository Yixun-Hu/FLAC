#!/usr/bin/env bash
# ============================================================================
# haa_ft_guardtests.sh — exercise every gate in haa_ft_launch.sh.
#
# Each case drives the REAL launcher and asserts both the exit code and the
# reason printed, so a gate that fires for the WRONG reason fails here. Every
# reject case also passes DRY_RUN=1: if the gate under test were DELETED, the
# case must still stop at the dry-run boundary rather than start a real run
# (exp_17 Codex r2 finding).
#
# What this suite does NOT do, deliberately:
#  * it never runs train.py, never runs eval, and never puts work on a GPU. The
#    R1 probe is stubbed through PROBE_CMD in every BF/YAW case. ⚠️ PROBE_CMD is
#    refused by the launcher outside DRY_RUN, so PRODUCTION runs the REAL probe,
#    unstubbed, against the real HAA stack — the stub exists only so this suite
#    can prove the WIRING (pass / fail / refusal / silent-zero) without a GPU.
#  * it writes no synthetic evidence into a production namespace. The init
#    decoys and the decoy manifests live in a private mktemp tree; the launcher
#    routes DRY_RUN output to ${EXPDIR}/.dryrun_logs by construction, so the
#    cleanup here can never delete a concurrent real run's log — the exp_17
#    suite's cleanup could, and its review said so.
#
# The one production file this suite temporarily edits is an ARM CONFIG, and
# only to prove the config-contract gate fires. Restoration is by SHA and FATAL:
# a suite that cannot prove it put the file back must not report success. If it
# somehow did not, a concurrent launcher would REFUSE (the arm-config byte pin
# is gate 2) rather than train on the mutated file — the safe direction.
#
# Usage:  bash worklog/worklog_yixun/exp_19_haa_finetune_claude/haa_ft_guardtests.sh
#
# Written by the exp_19 coder seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 2          # repo root
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"
LAUNCH="${EXPDIR}/haa_ft_launch.sh"
BF_CFG="${EXPDIR}/FLAC_HAA_finetune_BF.json"
YAW_CFG="${EXPDIR}/FLAC_HAA_finetune_YAW.json"
STOCK_CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
LOCKFILE="${EXPDIR}/.haa_ft.lock"
SRC_TAMPER="src/data/yaw_rotation.py"          # content-PINNED: trips gate 2
CLOSURE_TAMPER="src/models/utils.py"           # unpinned but IN the closure: trips the dirty check
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
GUARDLOG="${EXPDIR}/haa_ft_${TS}_guardtests.log"

PASS=0; FAIL=0; RESTORE_FAILED=0

# --- a live training run must not be disturbed ------------------------------- #
# Bracket trick: "[t]rain.py" cannot match pgrep's own command line, which would
# otherwise make this check fire on itself. Per CLAUDE.md, a train.py on this box
# may belong to a SIBLING checkout, so the cwd decides whether it is ours.
OURS="$(pwd -P)"
for PID in $(pgrep -f "[t]rain\.py" 2>/dev/null); do
  CWD="$(readlink -f "/proc/${PID}/cwd" 2>/dev/null)"
  [ "$CWD" = "$OURS" ] || continue
  echo "!! a train.py from THIS worktree is running (pid ${PID}, cwd ${CWD})."
  echo "   This suite briefly mutates an arm config; refusing while a run could read it."
  exit 2
done

# --- private scratch: decoy inits and decoy manifests ------------------------ #
TMPROOT="$(mktemp -d)"
INIT_OK="${TMPROOT}/inits";        mkdir -p "$INIT_OK"
INIT_EMPTY="${TMPROOT}/inits_empty"; mkdir -p "$INIT_EMPTY"
for A in P1 BF YAW; do head -c 4096 /dev/urandom > "${INIT_OK}/HAA_init_${A}.ckpt"; done

MAN_OK="${TMPROOT}/manifest_ok.txt"
sha256sum "${INIT_OK}"/HAA_init_*.ckpt > "$MAN_OK"
MAN_P1ONLY="${TMPROOT}/manifest_p1only.txt"
grep 'HAA_init_P1' "$MAN_OK" > "$MAN_P1ONLY"
MAN_BADSHA="${TMPROOT}/manifest_badsha.txt"
# A DETERMINISTIC wrong sha. The first version clobbered the leading nibble with
# '0', which is a no-op ~1 time in 16 — and it duly no-op'd on this suite's first
# run, so the case reported "accepted" for a launcher that was behaving
# correctly. A flaky negative control is worse than none.
awk '{print "0000000000000000000000000000000000000000000000000000000000000000  " $2}' \
    "$MAN_OK" > "$MAN_BADSHA"
MAN_DUP="${TMPROOT}/manifest_dup.txt"
{ cat "$MAN_OK"; grep 'HAA_init_BF' "$MAN_OK"; } > "$MAN_DUP"
MAN_MISSINGFILE="${TMPROOT}/manifest_missingfile.txt"
sed "s#${INIT_OK}#${INIT_EMPTY}#" "$MAN_OK" > "$MAN_MISSINGFILE"

# Stubs. The launcher requires BOTH rc=0 and the verdict line, so "silent zero"
# is its own case.
STUB_PASS='echo "R1 GATE PASS - stubbed"; exit 0'
STUB_FAIL='echo "  R1 GATE FAIL - do NOT launch"; exit 1'
STUB_REFUSE='echo "probe_haa_fa_invariance REFUSED: could not build the stack"; exit 2'
STUB_SILENT='echo "measured some things"; exit 0'

# --- fatal restoration ------------------------------------------------------- #
BF_BACKUP="${TMPROOT}/BF.json.bak"; cp "$BF_CFG" "$BF_BACKUP"
YAW_BACKUP="${TMPROOT}/YAW.json.bak"; cp "$YAW_CFG" "$YAW_BACKUP"
BF_SHA="$(sha256sum "$BF_CFG" | cut -d' ' -f1)"
YAW_SHA="$(sha256sum "$YAW_CFG" | cut -d' ' -f1)"
# The two source files this suite tampers with are restored by MOVE, but a move
# that silently failed would leave reviewed code modified; both are sha-checked
# in cleanup and a mismatch makes the whole suite untrustworthy.
SRC_SHA="$(sha256sum "$SRC_TAMPER" | cut -d' ' -f1)"
CLOSURE_SHA="$(sha256sum "$CLOSURE_TAMPER" | cut -d' ' -f1)"
restore_cfgs() {
  cp "$BF_BACKUP" "$BF_CFG"; cp "$YAW_BACKUP" "$YAW_CFG"
  local nb ny
  nb="$(sha256sum "$BF_CFG" | cut -d' ' -f1)"; ny="$(sha256sum "$YAW_CFG" | cut -d' ' -f1)"
  if [ "$nb" != "$BF_SHA" ] || [ "$ny" != "$YAW_SHA" ]; then
    echo "!! ARM CONFIG NOT RESTORED (BF ${nb} vs ${BF_SHA}; YAW ${ny} vs ${YAW_SHA})"
    RESTORE_FAILED=1
  fi
}
cleanup() {
  restore_cfgs
  [ -e "${SRC_TAMPER}.guardbak" ] && mv -f "${SRC_TAMPER}.guardbak" "$SRC_TAMPER"
  [ -e "${CLOSURE_TAMPER}.guardbak" ] && mv -f "${CLOSURE_TAMPER}.guardbak" "$CLOSURE_TAMPER"
  local ns nc
  ns="$(sha256sum "$SRC_TAMPER" | cut -d' ' -f1)"
  nc="$(sha256sum "$CLOSURE_TAMPER" | cut -d' ' -f1)"
  if [ "$ns" != "$SRC_SHA" ] || [ "$nc" != "$CLOSURE_SHA" ]; then
    echo "!! TAMPERED SOURCE NOT RESTORED (${SRC_TAMPER} ${ns} vs ${SRC_SHA}; ${CLOSURE_TAMPER} ${nc} vs ${CLOSURE_SHA})"
    RESTORE_FAILED=1
  fi
  # .dryrun_logs holds ONLY dry-run scratch by construction (the launcher routes
  # it there when DRY_RUN=1), so removing it cannot destroy production evidence.
  rm -rf "${EXPDIR}/.dryrun_logs"
  rm -rf "$TMPROOT"
}
trap cleanup EXIT

# --- helpers ----------------------------------------------------------------- #
# case_run <name> <expected_rc> <expected_substring> <env assignments...>
case_run() {
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  local out rc
  out="$(env "$@" bash "$LAUNCH" 2>&1)"; rc=$?
  if [ "$rc" = "$want_rc" ] && grep -qF -- "$want_txt" <<<"$out"; then
    echo "  PASS  ${name}  (rc=${rc})"; PASS=$((PASS+1))
  else
    echo "  FAIL  ${name}: want rc=${want_rc} and text '${want_txt}'; got rc=${rc}"
    echo "        last line: $(tail -1 <<<"$out")"; FAIL=$((FAIL+1))
  fi
}
expect() {
  local name="$1" ok="$2"
  if [ "$ok" = "1" ]; then echo "  PASS  ${name}"; PASS=$((PASS+1))
  else echo "  FAIL  ${name}"; FAIL=$((FAIL+1)); fi
}
eq() { [ "$1" = "$2" ] && echo 1; }
# argval <argv line> <flag> -> the token AFTER the flag, with the FLAG matched as
# a whole token. exp_17's sed version also matched '--not-max-steps 1000'
# (its own r3 non-blocking finding); field equality cannot.
argval() { awk -v f="$2" '{for(i=1;i<=NF;i++) if($i==f){print $(i+1); exit}}' <<<"$1"; }
mutate() { python - "$1" "$2" <<'PY'
import json, sys
p, expr = sys.argv[1], sys.argv[2]
cfg = json.load(open(p))
exec(expr, {"cfg": cfg})
json.dump(cfg, open(p, "w"), indent=4)
PY
}
shaof() { sha256sum "$1" | cut -d' ' -f1; }

# The environment every ACCEPT case needs: decoy inits + decoy manifest + a
# stubbed probe. Kept in one place so a case cannot accidentally accept for a
# reason this suite never intended.
OKENV=(DRY_RUN=1 MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" PROBE_CMD="$STUB_PASS")

exec > >(tee -a "$GUARDLOG") 2>&1
echo "=== exp_19 haa_ft_launch.sh guardtests — ${TS} — HEAD $(git rev-parse --short HEAD) ==="
echo "scratch: ${TMPROOT} (decoy inits + manifests; nothing synthetic enters a production namespace)"
echo "NOTE: the R1 probe is STUBBED here (PROBE_CMD). Production refuses that override; the real probe runs unstubbed."

# --------------------------------------------------------------------------- #
echo "--- A. ARM / GPU / MODE are matched exactly, never inferred ---"
case_run "A1 ARM unset"        2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM= GPU=0 MODE=FULL
case_run "A2 ARM lowercase"    2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM=bf GPU=0 MODE=FULL
case_run "A3 ARM invented"     2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM=BFX GPU=0 MODE=FULL
case_run "A4 GPU unset"        2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU= MODE=FULL
case_run "A5 GPU out of range" 2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU=2 MODE=FULL
case_run "A6 GPU non-numeric"  2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU=cuda:0 MODE=FULL
case_run "A7 MODE unset"       2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=
case_run "A8 MODE lowercase"   2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=full
case_run "A9 MODE near-miss"   2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL_

echo "--- B. the test overrides are a DRY_RUN-only facility ---"
# Without this, every gate below could be bypassed from the command line that
# wants to launch. Asserted with DRY_RUN unset, i.e. on the production path.
case_run "B1 PROBE_CMD refused outside DRY_RUN"   2 "PROBE_CMD is a DRY_RUN-only test override"   ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_PASS"
case_run "B2 MANIFEST refused outside DRY_RUN"    2 "MANIFEST is a DRY_RUN-only test override"    ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_OK"
case_run "B3 INIT_DIR refused outside DRY_RUN"    2 "INIT_DIR is a DRY_RUN-only test override"    ARM=BF GPU=0 MODE=FULL INIT_DIR="$INIT_OK"
case_run "B4 ARM_CFG_SHA refused outside DRY_RUN" 2 "ARM_CFG_SHA is a DRY_RUN-only test override" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA=deadbeef
case_run "B5 the override is disclosed when used" 0 "TEST OVERRIDE ACTIVE: PROBE_CMD"             "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- C. the init gate: manifest line, file, sha ---"
case_run "C1 manifest file absent"     2 "init manifest ${TMPROOT}/nope.txt not found"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="${TMPROOT}/nope.txt"
case_run "C2 no line for this arm"     2 "has 0 line(s) for"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_P1ONLY"
case_run "C3 ambiguous duplicate line" 2 "has 2 line(s) for"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_DUP"
case_run "C4 wrong init sha"           2 "INIT SHA MISMATCH"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_BADSHA"
case_run "C5 pinned init file missing" 2 "does not exist"        "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_MISSINGFILE" INIT_DIR="$INIT_EMPTY"
case_run "C6 the real default manifest is the one production uses" 2 "init manifest ${EXPDIR}/exp19_init_shas.txt not found" DRY_RUN=1 ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_PASS" INIT_DIR="$INIT_OK"

echo "--- D. config contract: one registered treatment per arm ---"
# ARM_CFG_SHA re-pins gate 2 to the MUTATED bytes so gate 4 is actually reached;
# case G2 covers the unre-pinned path (the byte pin fires first).
mutate "$BF_CFG" 'cfg["training"]["yaw_aug"] = {"enabled": True, "img_w": 512, "seed": 42}'
case_run "D1 BF with yaw_aug smuggled in"  2 "BF arm must NOT carry yaw_aug"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
mutate "$BF_CFG" 'cfg["training"]["frame_avg_angles"] = [0.0, 45.0, 90.0, 135.0]'
case_run "D2 BF orbit is not B-F's"        2 "must be B-F's float C4 orbit"   "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
mutate "$BF_CFG" 'cfg["training"]["frame_avg_angles"] = [0, 90, 180, 270]'
case_run "D3 BF orbit ints for floats"     2 "must be B-F's float C4 orbit"   "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
mutate "$BF_CFG" 'cfg["training"]["cond_method"] = "vanilla"'
case_run "D4 BF conditioning dropped"      2 "cond_method must be 'fa_invariant'" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
mutate "$BF_CFG" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "D5 BF non-delta drift"           2 "NOT the stock plus exactly its two registered deltas" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
mutate "$BF_CFG" 'cfg["training"]["mask_padding_dropout"] = 0'
case_run "D6 BF type-only drift"           2 "NOT the stock plus exactly its two registered deltas" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs
# Reported BY NAME, not as a generic "not the stock plus its deltas": use_ema
# decides whether the HAA rows are EMA rows at all, and the first version of the
# gate checked it AFTER the strict comparison, where it was unreachable.
mutate "$BF_CFG" 'cfg["training"]["use_ema"] = False'
case_run "D7 BF EMA off"                   2 "use_ema must be true"           "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL ARM_CFG_SHA="$(shaof "$BF_CFG")"
restore_cfgs

mutate "$YAW_CFG" 'cfg["training"]["cond_method"] = "fa_invariant"'
case_run "D8 YAW with cond_method smuggled in" 2 "YAW arm must NOT carry cond_method" "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL ARM_CFG_SHA="$(shaof "$YAW_CFG")"
restore_cfgs
mutate "$YAW_CFG" 'cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["img_w"] = 256'
case_run "D9 YAW width mismatch (ViT side)"    2 "roll the panorama by the wrong number of columns" "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL ARM_CFG_SHA="$(shaof "$YAW_CFG")"
restore_cfgs
mutate "$YAW_CFG" 'cfg["training"]["yaw_aug"]["enabled"] = 1'
case_run "D10 YAW bool-as-int"                 2 "wrong TYPES"                "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL ARM_CFG_SHA="$(shaof "$YAW_CFG")"
restore_cfgs
mutate "$YAW_CFG" 'cfg["training"]["yaw_aug"]["seed"] = 43'
case_run "D11 YAW treatment altered"           2 "not the registered treatment" "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL ARM_CFG_SHA="$(shaof "$YAW_CFG")"
restore_cfgs
mutate "$YAW_CFG" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "D12 YAW non-delta drift"             2 "NOT the stock plus exactly training.yaw_aug" "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL ARM_CFG_SHA="$(shaof "$YAW_CFG")"
restore_cfgs

echo "--- E. the R1 probe gate is wired to the probe's real verdict ---"
case_run "E1 probe FAIL stops the arm"      2 "R1 GATE REFUSED (probe rc=1)"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "E2 probe REFUSAL is not a skip"   2 "R1 GATE REFUSED (probe rc=2)"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_REFUSE"
case_run "E3 silent zero is not a pass"     2 "printed no 'R1 GATE PASS' verdict" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_SILENT"
case_run "E4 YAW is gated too (it rotates)" 2 "R1 GATE REFUSED (probe rc=1)"  "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "E5 P1 skips the probe by name"    0 "R1 probe: SKIPPED for the P1 arm" "${OKENV[@]}" ARM=P1 GPU=0 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "E6 the standing r1 limitation is disclosed in every gated run" 0 \
         "subject and oracle share rotate_scene_metadata" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
case_run "E7 the disclosure names the init the probe loads" 0 \
         "R1 probe DISCLOSURE: loads ${INIT_OK}/HAA_init_BF.ckpt through train.py's consumer path" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- F. accept paths: all three arms reach the dry-run boundary ---"
case_run "F1 P1 FULL accepted"    0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=P1  GPU=0 MODE=FULL
case_run "F2 BF FULL accepted"    0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF  GPU=1 MODE=FULL
case_run "F3 YAW FULL accepted"   0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=YAW GPU=0 MODE=FULL
case_run "F4 BF SMOKE accepted"   0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF  GPU=0 MODE=SMOKE
case_run "F5 P1 contract states the stock file is used directly" 0 \
         "config contract OK: ${STOCK_CFG} == ${STOCK_CFG} + none (stock config used directly)" \
         "${OKENV[@]}" ARM=P1 GPU=0 MODE=FULL
case_run "F6 the init gate reports a match"  0 "init gate OK:"   "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
case_run "F7 source pins are reported"       0 "source pins OK"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- G. reviewed-source pins fire, and fire FIRST ---"
cp "$SRC_TAMPER" "${SRC_TAMPER}.guardbak"
printf '\n# guardtest tamper\n' >> "$SRC_TAMPER"
case_run "G1 code tamper caught"   2 "SOURCE PIN FAILED for 'src/data/yaw_rotation.py'"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
mv -f "${SRC_TAMPER}.guardbak" "$SRC_TAMPER"
case_run "G2 pins pass once restored" 0 "source pins OK"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
mutate "$BF_CFG" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "G3 an edited arm config trips the byte pin before the contract" 2 \
         "SOURCE PIN FAILED for '${BF_CFG}'"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
restore_cfgs
case_run "G4 restored arm config passes again" 0 "config contract OK"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
# The clean-tree gate had NO case of its own (exp_17 Codex r3: "no case
# independently reaches the clean-tree gate"). Driven here with an UNPINNED file
# inside the training closure, so it is the dirty check that fires and not gate 2.
cp "$CLOSURE_TAMPER" "${CLOSURE_TAMPER}.guardbak"
printf '\n# guardtest tamper\n' >> "$CLOSURE_TAMPER"
case_run "G5 a dirty training closure is refused" 2 "tracked training closure is dirty" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
mv -f "${CLOSURE_TAMPER}.guardbak" "$CLOSURE_TAMPER"
case_run "G6 and clean again once restored" 0 "tree clean across the training closure" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- H. the ACTUAL train.py argv, per arm and per mode ---"
# Capture, THEN grep. Piping the launcher into `grep -m1` would SIGPIPE it once
# the match is found — harmless here, but the same shape is what section K2
# forbids in the launcher, and a suite that models the trap it polices is a
# suite nobody will trust.
argv_of() {
  local out; out="$(env "${OKENV[@]}" ARM="$1" GPU="$2" MODE="$3" bash "$LAUNCH" 2>&1)"
  grep -m1 '^ARGV: ' <<<"$out"
}
FULL_BF="$(argv_of BF 0 FULL)"
SMOKE_BF="$(argv_of BF 0 SMOKE)"
FULL_P1="$(argv_of P1 0 FULL)"
FULL_YAW="$(argv_of YAW 1 FULL)"

expect "H1 FULL endpoint exactly 1000"       "$(eq "$(argval "$FULL_BF" --max-steps)" 1000)"
expect "H2 FULL cadence exactly 10"          "$(eq "$(argval "$FULL_BF" --checkpoint-every)" 10)"
expect "H3 FULL val-every exactly 10"        "$(eq "$(argval "$FULL_BF" --val-every)" 10)"
expect "H4 SMOKE endpoint exactly 20"        "$(eq "$(argval "$SMOKE_BF" --max-steps)" 20)"
expect "H5 SMOKE cadence 1e6 (no ckpt)"      "$(eq "$(argval "$SMOKE_BF" --checkpoint-every)" 1000000)"
expect "H6 SMOKE val-every 1e6"              "$(eq "$(argval "$SMOKE_BF" --val-every)" 1000000)"
expect "H7 batch exactly 16"                 "$(eq "$(argval "$FULL_BF" --batch-size)" 16)"
expect "H8 accumulation exactly 4"           "$(eq "$(argval "$FULL_BF" --accum-batches)" 4)"
expect "H9 single GPU"                       "$(eq "$(argval "$FULL_BF" --num-gpus)" 1)"
expect "H10 precision bf16-mixed"            "$(eq "$(argval "$FULL_BF" --precision)" bf16-mixed)"
expect "H11 seed exactly 42"                 "$(eq "$(argval "$FULL_BF" --seed)" 42)"
expect "H12 num-workers exactly 8"           "$(eq "$(argval "$FULL_BF" --num-workers)" 8)"
expect "H13 train split is haa_train.json"   "$(eq "$(argval "$FULL_BF" --dataset-config)" src/configs/dataset_configs/HAA/train/haa_train.json)"
expect "H14 val split is haa_val.json"       "$(eq "$(argval "$FULL_BF" --val-dataset-config)" src/configs/dataset_configs/HAA/eval/haa_val.json)"
expect "H15 VAE is the frozen pretransform"  "$(eq "$(argval "$FULL_BF" --pretransform-ckpt-path)" weights/FLAC/VAE.safetensors)"
expect "H16 BF uses the BF config"           "$(eq "$(argval "$FULL_BF" --model-config)" "$BF_CFG")"
expect "H17 YAW uses the YAW config"         "$(eq "$(argval "$FULL_YAW" --model-config)" "$YAW_CFG")"
expect "H18 P1 uses the STOCK config"        "$(eq "$(argval "$FULL_P1" --model-config)" "$STOCK_CFG")"
expect "H19 BF inits from the BF weights"    "$(eq "$(argval "$FULL_BF" --pretrained-ckpt-path)" "${INIT_OK}/HAA_init_BF.ckpt")"
expect "H20 P1 inits from the P1 weights"    "$(eq "$(argval "$FULL_P1" --pretrained-ckpt-path)" "${INIT_OK}/HAA_init_P1.ckpt")"

echo "--- H'. namespace separation (no arm can write into another's) ---"
expect "H21 BF FULL save-dir"   "$(eq "$(argval "$FULL_BF" --save-dir)" outputs_FLAC/exp19_HAA_BF)"
expect "H22 BF SMOKE save-dir"  "$(eq "$(argval "$SMOKE_BF" --save-dir)" outputs_FLAC/exp19_HAA_BF_smoke)"
expect "H23 P1 FULL save-dir"   "$(eq "$(argval "$FULL_P1" --save-dir)" outputs_FLAC/exp19_HAA_P1)"
expect "H24 YAW FULL save-dir"  "$(eq "$(argval "$FULL_YAW" --save-dir)" outputs_FLAC/exp19_HAA_YAW)"
expect "H25 wandb run name per arm"        "$(eq "$(argval "$FULL_BF" --name)" FLAC_exp19_HAA_BF)"
expect "H26 wandb experiment per arm"      "$(eq "$(argval "$FULL_BF" --experiment-name)" exp19_HAA_BF)"
expect "H27 smoke wandb identity differs"  "$(eq "$(argval "$SMOKE_BF" --experiment-name)" exp19_HAA_BF_smoke)"
ALLDIRS="$(printf '%s\n' \
  "$(argval "$FULL_BF" --save-dir)" "$(argval "$SMOKE_BF" --save-dir)" \
  "$(argval "$FULL_P1" --save-dir)" "$(argval "$FULL_YAW" --save-dir)" | sort)"
expect "H28 the four namespaces are pairwise distinct" \
  "$(eq "$(printf '%s\n' "$ALLDIRS" | sort -u | wc -l)" 4)"

echo "--- I. argval itself is strict (non-vacuity for section H) ---"
expect "I1 argval matches the flag as a whole token" \
  "$(eq "$(argval 'ARGV: python train.py --not-max-steps 999 --max-steps 1000' --max-steps)" 1000)"
expect "I2 argval returns nothing for an absent flag" \
  "$(eq "$(argval 'ARGV: python train.py --max-steps 1000' --sync-batchnorm)" '')"
expect "I3 eq rejects a value substring" "$(eq "$(eq 10000 1000)" '')"

echo "--- J. single instance through the gate phase ---"
# A separate PROCESS holds the lock, so nothing depends on fd inheritance.
flock -x "$LOCKFILE" -c 'sleep 6' &
LOCKPID=$!
sleep 1
case_run "J1 a second launcher is refused while the gate lock is held" 2 \
         "another exp_19 launcher holds"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
wait "$LOCKPID" 2>/dev/null
case_run "J2 and accepted once it is released" 0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- K. properties of the launcher that must not silently regress ---"
expect "K1 the budget is not env-overridable" \
  "$(! grep -qE 'FULL_STEPS:-|FULL_CADENCE:-|SMOKE_STEPS:-|SMOKE_CADENCE:-|FULL_VAL_EVERY:-|MID_STEPS:-|BATCH:-|ACCUM:-|SEED:-' "$LAUNCH" && echo 1)"
expect "K2 post-run log reading never uses 'tr | grep -q' (pipefail SIGPIPE)" \
  "$(! grep -qE "tr '..' '..' <[^>]*\| *grep" "$LAUNCH" && echo 1)"
expect "K3 the endpoint marker carries Lightning's backticks" \
  "$(grep -qF 'MARKER="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."' "$LAUNCH" && echo 1)"
expect "K4 the endpoint is matched as line-ENDS-WITH, not substring" \
  "$(grep -qF 'substr($0, length($0)-length(m)+1) == m' "$LAUNCH" && echo 1)"
expect "K5 the banner is matched as a WHOLE line" \
  "$(grep -qF 'grep -qxF -- "$BANNER" "$NORM"' "$LAUNCH" && echo 1)"
# Captured ONCE, then inspected from a herestring. The first version piped the
# launcher straight into `grep -qF`, and under this suite's own `pipefail` the
# `-q` early exit SIGPIPE'd the launcher and turned a successful match into a
# FAIL — the exact defect exp_17's Codex r3 review flagged in the launcher,
# reproduced here in the test that polices it.
YAW_PREFLIGHT="$(env "${OKENV[@]}" ARM=YAW GPU=0 MODE=FULL bash "$LAUNCH" 2>&1)"
YAW_PREFLIGHT_N="$(tr '\r' '\n' <<<"$YAW_PREFLIGHT")"
expect "K6 the preflight never prints the banner text (self-satisfaction guard)" \
  "$(grep -qxF 'yaw_aug ENABLED img_w=512 seed=42' <<<"$YAW_PREFLIGHT_N" && echo 0 || echo 1)"
expect "K7 but the preflight IS non-vacuous about the treatment" \
  "$(grep -qF 'preflight treatment plan:' <<<"$YAW_PREFLIGHT_N" && echo 1)"
expect "K8 NaN/Inf is checked in BOTH modes, not SMOKE-only" \
  "$(grep -qF "grep -qiE 'train/loss=(nan|-?inf(inity)?)' \"\$NORM\"" "$LAUNCH" \
     && ! grep -q 'MODE" = "SMOKE" \].*train/loss' "$LAUNCH" && echo 1)"
expect "K9 both registered readings (410 and 1000) are required for FULL" \
  "$(grep -qF 'for S in "$MID_STEPS" "$FULL_STEPS"' "$LAUNCH" && echo 1)"
expect "K10 the checkpoint count is REPORTED, not asserted equal" \
  "$(grep -qF 'cadence ${CADENCE} alone would give' "$LAUNCH" && echo 1)"
expect "K11 the gate lock is released before training" \
  "$(grep -qF 'exec 9>&-' "$LAUNCH" && echo 1)"
expect "K12 train.py output is piped synchronously (PIPESTATUS), not async tee" \
  "$(grep -qF 'rc="${PIPESTATUS[0]}"' "$LAUNCH" && echo 1)"
expect "K13 DRY_RUN output never lands in the evidence directory" \
  "$(grep -qF 'LOGDIR="${EXPDIR}/.dryrun_logs"' "$LAUNCH" && echo 1)"
# Codex r1 finding 5: the gate is only about the arm if it measures the arm's
# weights. The launcher must hand the probe the very init it pinned in gate 3.
expect "K14 the probe is given the arm's init via --ckpt-path" \
  "$(grep -qF -- '--ckpt-path "$INIT"' "$LAUNCH" && echo 1)"
expect "K15 the pinned probe sha is the current file" \
  "$(grep -qF "$(sha256sum "${EXPDIR}/probe_haa_fa_invariance.py" | cut -d' ' -f1)" "$LAUNCH" && echo 1)"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
echo "log: ${GUARDLOG}"
[ "$RESTORE_FAILED" -eq 0 ] || { echo "!! arm-config restoration could not be proven - suite result is NOT trustworthy"; exit 3; }
[ "$FAIL" -eq 0 ]
