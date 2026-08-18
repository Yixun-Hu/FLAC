#!/usr/bin/env bash
# ============================================================================
# haa_ft_guardtests.sh — exercise every gate in haa_ft_launch.sh.
#
# Each case drives the REAL launcher and asserts both the exit code and the
# reason printed, so a gate that fires for the wrong reason fails here.
#
# Three properties this suite is built around, each from a Codex finding:
#
#  * **Nothing here can start training.** Every reject case runs under DRY_RUN=1
#    (the launcher exits before train.py) OR, where the case must test a
#    PRODUCTION refusal, it is poisoned: those cases are MODE=FULL with no
#    EXPECT_SHA, so if the refusal under test were deleted the revision gate
#    still refuses. r2-B5 named the one case that could previously have reached
#    real training.
#  * **It cannot race a production launcher.** The suite takes `.haa_ft.lock`
#    EXCLUSIVELY for its whole duration and aborts if it cannot (r2-B3), so no
#    production launcher can be inside its gate phase while an arm config is
#    momentarily mutated. Its own launcher children are told the caller holds it.
#  * **Restoration failure is fatal FROM THE TRAP** (r2-B4): the EXIT trap
#    re-checks every restored file by sha and exits non-zero itself, so a
#    restoration that silently failed cannot be reported as a green suite. Every
#    cp/mv in a restoration path has its status checked.
#
# The R1 probe is stubbed through PROBE_CMD in every BF/YAW case. ⚠️ PROBE_CMD is
# refused by the launcher in production, so a real launch runs the REAL probe
# against the real HAA stack; the stub exists only so this suite can prove the
# WIRING (pass / fail / refusal / silent-zero) without a GPU.
#
# The post-run verdicts (endpoint, banner, NaN, 410/1000, smoke checkpoints, rc
# propagation) are EXECUTED, not grepped for, via the launcher's DRY_RUN=2
# rehearsal mode with a stubbed TRAIN_CMD writing into a private REHEARSAL_DIR.
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

# --- (r2-B3) exclude production launchers for the WHOLE suite ----------------- #
# The window this closes: a production launcher passes its byte pin and config
# contract, the suite then mutates the config, and train.py opens the mutated
# bytes after the launcher releases its lock. Holding the gate lock exclusively
# for the whole suite makes that sequence impossible. Non-blocking on purpose —
# if a launcher is mid-gates right now, the safe action is to NOT mutate.
exec 9>"$LOCKFILE" || { echo "!! cannot open ${LOCKFILE}"; exit 2; }
flock -n 9 || {
  echo "!! ${LOCKFILE} is held: an exp_19 launcher may be mid-gates."
  echo "   Refusing to run — this suite mutates arm configs and must never do so"
  echo "   while a production launcher could read them."
  exit 2; }

# --- private scratch: decoy inits, manifests, dataset fixture, rehearsal dir -- #
TMPROOT="$(mktemp -d)" || exit 2
INIT_OK="${TMPROOT}/inits";          mkdir -p "$INIT_OK"
INIT_EMPTY="${TMPROOT}/inits_empty"; mkdir -p "$INIT_EMPTY"
REHEARSE="${TMPROOT}/rehearsal";     mkdir -p "$REHEARSE"
# The launcher mints a PER-INVOCATION dry-run log dir with `mktemp -d -t`, which
# honours TMPDIR. Pointing that at the suite's own scratch means those dirs are
# removed with everything else at exit — no bookkeeping to get wrong, and nothing
# of another session's is ever a candidate for deletion.
DRYTMP="${TMPROOT}/dry";             mkdir -p "$DRYTMP"
for A in P1 BF YAW; do head -c 4096 /dev/urandom > "${INIT_OK}/HAA_init_${A}.ckpt"; done

MAN_OK="${TMPROOT}/manifest_ok.txt"
sha256sum "${INIT_OK}"/HAA_init_*.ckpt > "$MAN_OK"
MAN_P1ONLY="${TMPROOT}/manifest_p1only.txt"
grep 'HAA_init_P1' "$MAN_OK" > "$MAN_P1ONLY"
MAN_BADSHA="${TMPROOT}/manifest_badsha.txt"
# A DETERMINISTIC wrong sha. The first version clobbered the leading nibble with
# '0', which is a no-op ~1 time in 16 — and it duly no-op'd on this suite's first
# run. A flaky negative control is worse than none.
awk '{print "0000000000000000000000000000000000000000000000000000000000000000  " $2}' \
    "$MAN_OK" > "$MAN_BADSHA"
MAN_DUP="${TMPROOT}/manifest_dup.txt"
{ cat "$MAN_OK"; grep 'HAA_init_BF' "$MAN_OK"; } > "$MAN_DUP"
MAN_MISSINGFILE="${TMPROOT}/manifest_missingfile.txt"
sed "s#${INIT_OK}#${INIT_EMPTY}#" "$MAN_OK" > "$MAN_MISSINGFILE"

# Dataset fixture: the FIRST and LAST file each split actually references, built
# by reading the real split JSONs so the fixture cannot drift from the gate.
# ⚠️ The real HAA tree on this machine is NOT prepared (no mono_rirs_22050Hz /
# metadata / depth_images yet, and a relocation to /media/diskstation is in
# flight), so production launches legitimately refuse at that gate today. This
# fixture exercises the gate's logic without pretending the data is ready.
DATA_OK="${TMPROOT}/haa_ok"
DATA_EMPTY="${TMPROOT}/haa_empty"; mkdir -p "$DATA_EMPTY"
python - "$DATA_OK" <<'PY' || { echo "!! could not build the dataset fixture"; exit 2; }
import json, os, sys
root = sys.argv[1]
for cfg_path in ("src/configs/dataset_configs/HAA/train/haa_train.json",
                 "src/configs/dataset_configs/HAA/eval/haa_val.json"):
    ds = json.load(open(cfg_path))["datasets"][0]
    split = json.load(open(ds["json_file_path"]))
    scenes = [s for s in split if split[s]]
    for scene, name in ((scenes[0], split[scenes[0]][0]), (scenes[-1], split[scenes[-1]][-1])):
        p = os.path.join(root, scene, ds.get("folder_name", "binaural_rirs"), name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"RIFF----WAVEfmt ")          # non-empty; the gate stats size
    print(f"fixture: {cfg_path} -> {scenes[0]}/{split[scenes[0]][0]}, {scenes[-1]}/{split[scenes[-1]][-1]}")
PY
# A root whose FIRST/LAST files exist but one is zero-length: the "half-copied
# relocation" case, which a plain existence check would pass.
DATA_TRUNC="${TMPROOT}/haa_trunc"
cp -a "$DATA_OK" "$DATA_TRUNC" || { echo "!! fixture copy failed"; exit 2; }
find "$DATA_TRUNC" -name '*.wav' | head -1 | while read -r f; do : > "$f"; done
# A symlink to the good root: the relocation shape the gate must accept.
DATA_LINK="${TMPROOT}/haa_link"
ln -s "$DATA_OK" "$DATA_LINK" || { echo "!! fixture symlink failed"; exit 2; }

# Probe stubs. The launcher requires BOTH rc=0 and the verdict line, so "silent
# zero" is its own case.
STUB_PASS='echo "R1 GATE PASS - stubbed"; exit 0'
STUB_FAIL='echo "  R1 GATE FAIL - do NOT launch"; exit 1'
STUB_REFUSE='echo "probe_haa_fa_invariance REFUSED: could not build the stack"; exit 2'
STUB_SILENT='echo "measured some things"; exit 0'

# --- fatal restoration (r2-B4) ----------------------------------------------- #
BF_BACKUP="${TMPROOT}/BF.json.bak"; cp "$BF_CFG" "$BF_BACKUP" || exit 2
YAW_BACKUP="${TMPROOT}/YAW.json.bak"; cp "$YAW_CFG" "$YAW_BACKUP" || exit 2
BF_SHA="$(sha256sum "$BF_CFG" | cut -d' ' -f1)"
YAW_SHA="$(sha256sum "$YAW_CFG" | cut -d' ' -f1)"
SRC_SHA="$(sha256sum "$SRC_TAMPER" | cut -d' ' -f1)"
CLOSURE_SHA="$(sha256sum "$CLOSURE_TAMPER" | cut -d' ' -f1)"

restore_cfgs() {
  cp "$BF_BACKUP" "$BF_CFG"  || { echo "!! cp BF restore FAILED";  RESTORE_FAILED=1; }
  cp "$YAW_BACKUP" "$YAW_CFG" || { echo "!! cp YAW restore FAILED"; RESTORE_FAILED=1; }
  local nb ny
  nb="$(sha256sum "$BF_CFG" | cut -d' ' -f1)"; ny="$(sha256sum "$YAW_CFG" | cut -d' ' -f1)"
  if [ "$nb" != "$BF_SHA" ] || [ "$ny" != "$YAW_SHA" ]; then
    echo "!! ARM CONFIG NOT RESTORED (BF ${nb} vs ${BF_SHA}; YAW ${ny} vs ${YAW_SHA})"
    RESTORE_FAILED=1
  fi
}
restore_sources() {
  if [ -e "${SRC_TAMPER}.guardbak" ]; then
    mv -f "${SRC_TAMPER}.guardbak" "$SRC_TAMPER" || { echo "!! mv ${SRC_TAMPER} restore FAILED"; RESTORE_FAILED=1; }
  fi
  if [ -e "${CLOSURE_TAMPER}.guardbak" ]; then
    mv -f "${CLOSURE_TAMPER}.guardbak" "$CLOSURE_TAMPER" || { echo "!! mv ${CLOSURE_TAMPER} restore FAILED"; RESTORE_FAILED=1; }
  fi
  local ns nc
  ns="$(sha256sum "$SRC_TAMPER" | cut -d' ' -f1)"
  nc="$(sha256sum "$CLOSURE_TAMPER" | cut -d' ' -f1)"
  if [ "$ns" != "$SRC_SHA" ] || [ "$nc" != "$CLOSURE_SHA" ]; then
    echo "!! TAMPERED SOURCE NOT RESTORED (${SRC_TAMPER} ${ns} vs ${SRC_SHA}; ${CLOSURE_TAMPER} ${nc} vs ${CLOSURE_SHA})"
    RESTORE_FAILED=1
  fi
}
# The trap itself decides the exit status: checking RESTORE_FAILED before the
# trap runs (as the previous revision did) cannot see a failure that happens
# INSIDE it, so the "fatal" claim was false (r2-B4).
cleanup() {
  local pending=$?
  restore_cfgs
  restore_sources
  rm -rf "$TMPROOT"
  if [ "$RESTORE_FAILED" -ne 0 ]; then
    echo "!! RESTORATION COULD NOT BE PROVEN - the repository may be left modified"
    echo "!! suite result is NOT trustworthy"
    exit 3
  fi
  exit "$pending"
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
# a whole token. exp_17's sed version also matched '--not-max-steps 1000'.
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

# The environment every ACCEPT case needs. GATE_LOCK_HELD_BY_CALLER is not a
# convenience: this suite owns .haa_ft.lock for its whole duration (above), so a
# child that tried to take it would deadlock against its own suite.
OKENV=(DRY_RUN=1 MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" PROBE_CMD="$STUB_PASS"
       HAA_ROOT="$DATA_OK" GATE_LOCK_HELD_BY_CALLER=1 TMPDIR="$DRYTMP")
# Production-refusal cases: MODE=FULL and NO EXPECT_SHA, so the revision gate is
# an independent second refusal behind whichever one is under test (r2-B5).
POISON=(ARM=BF GPU=0 MODE=FULL)

exec > >(tee -a "$GUARDLOG") 2>&1
echo "=== exp_19 haa_ft_launch.sh guardtests — ${TS} — HEAD $(git rev-parse --short HEAD) ==="
echo "scratch: ${TMPROOT} (decoy inits, manifests, dataset fixture, rehearsal dir)"
echo "gate lock: held exclusively by this suite for its whole duration"
echo "NOTE: the R1 probe is STUBBED here (PROBE_CMD). Production refuses that override."

# --------------------------------------------------------------------------- #
echo "--- A. ARM / GPU / MODE / DRY_RUN are matched exactly, never inferred ---"
case_run "A1 ARM unset"        2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM= GPU=0 MODE=FULL
case_run "A2 ARM lowercase"    2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM=bf GPU=0 MODE=FULL
case_run "A3 ARM invented"     2 "ARM must be exactly P1, BF or YAW"  "${OKENV[@]}" ARM=BFX GPU=0 MODE=FULL
case_run "A4 GPU unset"        2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU= MODE=FULL
case_run "A5 GPU out of range" 2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU=2 MODE=FULL
case_run "A6 GPU non-numeric"  2 "GPU must be exactly 0 or 1"         "${OKENV[@]}" ARM=BF GPU=cuda:0 MODE=FULL
case_run "A7 MODE unset"       2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=
case_run "A8 MODE lowercase"   2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=full
case_run "A9 MODE near-miss"   2 "MODE must be exactly SMOKE or FULL" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL_
case_run "A10 DRY_RUN invented" 2 "DRY_RUN must be exactly 0"          "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL DRY_RUN=yes

echo "--- B. the test overrides are a DRY-only facility (each poisoned by EXPECT_SHA) ---"
case_run "B1 PROBE_CMD refused in production"   2 "PROBE_CMD is a DRY_RUN-only test override"   "${POISON[@]}" PROBE_CMD="$STUB_PASS"
case_run "B2 MANIFEST refused in production"    2 "MANIFEST is a DRY_RUN-only test override"    "${POISON[@]}" MANIFEST="$MAN_OK"
case_run "B3 INIT_DIR refused in production"    2 "INIT_DIR is a DRY_RUN-only test override"    "${POISON[@]}" INIT_DIR="$INIT_OK"
case_run "B4 ARM_CFG_SHA refused in production" 2 "ARM_CFG_SHA is a DRY_RUN-only test override" "${POISON[@]}" ARM_CFG_SHA=deadbeef
case_run "B5 HAA_ROOT refused in production"    2 "HAA_ROOT is a DRY_RUN-only test override"    "${POISON[@]}" HAA_ROOT="$DATA_OK"
case_run "B6 TRAIN_CMD refused in production"   2 "TRAIN_CMD is a DRY_RUN-only test override"   "${POISON[@]}" TRAIN_CMD="true"
case_run "B7 caller-held lock refused in production" 2 "GATE_LOCK_HELD_BY_CALLER is a DRY_RUN-only test override" "${POISON[@]}" GATE_LOCK_HELD_BY_CALLER=1
case_run "B8 THE POISON ITSELF: no EXPECT_SHA, no override, still refused" 2 \
         "EXPECT_SHA is REQUIRED for a production FULL launch" "${POISON[@]}"
case_run "B9 the override is disclosed when used" 0 "TEST OVERRIDE ACTIVE: PROBE_CMD" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- B'. revision binding (r2-B1) ---"
case_run "B10 a wrong EXPECT_SHA is refused" 2 "is not the revision you reviewed" \
         ARM=BF GPU=0 MODE=FULL EXPECT_SHA=0000000000000000000000000000000000000000
case_run "B11 a DRY FULL may omit EXPECT_SHA" 0 "DRY_RUN: all gates passed" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
# A production SMOKE must get PAST the revision gate. It cannot reach the dry-run
# boundary here (this suite owns the gate lock, and a production run may not be
# told otherwise), so the provable statement is the negative one: whatever stops
# it, it is not EXPECT_SHA. Captured then searched — piping into `grep -q` would
# SIGPIPE the launcher under this suite's own `pipefail`, the very defect L2
# polices in the launcher.
B11_OUT="$(env ARM=BF GPU=0 MODE=SMOKE bash "$LAUNCH" 2>&1)"
expect "B12 a production SMOKE is not stopped by the revision gate" \
  "$(grep -qF 'EXPECT_SHA is REQUIRED' <<<"$B11_OUT" && echo 0 || echo 1)"
B12_OUT="$(env "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL bash "$LAUNCH" 2>&1)"
expect "B13 HEAD is recorded in every mode" \
  "$(grep -qF "HEAD $(git rev-parse HEAD)" <<<"$B12_OUT" && echo 1)"

echo "--- C. the init gate: manifest line, file, sha ---"
case_run "C1 manifest file absent"     2 "init manifest ${TMPROOT}/nope.txt not found"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="${TMPROOT}/nope.txt"
case_run "C2 no line for this arm"     2 "has 0 line(s) for"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_P1ONLY"
case_run "C3 ambiguous duplicate line" 2 "has 2 line(s) for"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_DUP"
case_run "C4 wrong init sha"           2 "INIT SHA MISMATCH"     "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_BADSHA"
case_run "C5 pinned init file missing" 2 "does not exist"        "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MANIFEST="$MAN_MISSINGFILE" INIT_DIR="$INIT_EMPTY"
# C6 used to assert that the DEFAULT manifest is absent, which would become
# permanently red the day the inits are extracted (r2 finding). The durable
# statement is that the default PATH is the production one.
expect "C6 the default manifest path is the production one" \
  "$(grep -qF 'MANIFEST_DEFAULT="${EXPDIR}/exp19_init_shas.txt"' "$LAUNCH" && echo 1)"

echo "--- D. config contract: one registered treatment per arm ---"
# ARM_CFG_SHA re-pins gate 2 to the MUTATED bytes so gate 4 is actually reached;
# case G3 covers the unre-pinned path (the byte pin fires first).
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

echo "--- E. the dataset inventory gate (relocation-aware) ---"
case_run "E1 a missing dataset root is refused"  2 "referenced file missing"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL HAA_ROOT="${TMPROOT}/no_such_root"
case_run "E2 an empty (mid-copy) root is refused" 2 "referenced file missing" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL HAA_ROOT="$DATA_EMPTY"
case_run "E3 a zero-length file is refused"       2 "is EMPTY"                "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL HAA_ROOT="$DATA_TRUNC"
case_run "E4 a SYMLINKED root is accepted (the relocation shape)" 0 "dataset inventory OK" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL HAA_ROOT="$DATA_LINK"
case_run "E5 the resolved target is logged"       0 "-> $(readlink -f "$DATA_OK")"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL HAA_ROOT="$DATA_LINK"
case_run "E6 P1 is gated on the dataset too"      2 "referenced file missing"  "${OKENV[@]}" ARM=P1 GPU=0 MODE=FULL HAA_ROOT="$DATA_EMPTY"

echo "--- F. the R1 probe gate is wired to the probe's real verdict ---"
case_run "F1 probe FAIL stops the arm"      2 "R1 GATE REFUSED (probe rc=1)"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "F2 probe REFUSAL is not a skip"   2 "R1 GATE REFUSED (probe rc=2)"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_REFUSE"
case_run "F3 silent zero is not a pass"     2 "printed no 'R1 GATE PASS' verdict" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL PROBE_CMD="$STUB_SILENT"
case_run "F4 YAW is gated too (it rotates)" 2 "R1 GATE REFUSED (probe rc=1)"  "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "F5 P1 skips the probe by name"    0 "R1 probe: SKIPPED for the P1 arm" "${OKENV[@]}" ARM=P1 GPU=0 MODE=FULL PROBE_CMD="$STUB_FAIL"
case_run "F6 the standing r1 limitation is disclosed" 0 "subject and oracle share rotate_scene_metadata" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
case_run "F7 the disclosure names the init the probe loads" 0 \
         "R1 probe DISCLOSURE: loads ${INIT_OK}/HAA_init_BF.ckpt through train.py's consumer path" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- G. accept paths: all three arms reach the dry-run boundary ---"
case_run "G1 P1 FULL accepted"    0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=P1  GPU=0 MODE=FULL
case_run "G2 BF FULL accepted"    0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF  GPU=1 MODE=FULL
case_run "G3 YAW FULL accepted"   0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=YAW GPU=0 MODE=FULL
case_run "G4 BF SMOKE accepted"   0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF  GPU=0 MODE=SMOKE
case_run "G5 P1 contract states the stock file is used directly" 0 \
         "config contract OK: ${STOCK_CFG} == ${STOCK_CFG} + none (stock config used directly)" \
         "${OKENV[@]}" ARM=P1 GPU=0 MODE=FULL
case_run "G6 the init gate reports a match"  0 "init gate OK:"   "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
case_run "G7 source pins are reported (incl. both splits)" 0 "incl. both HAA split inventories" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL

echo "--- H. reviewed-source pins and the closure, and the order they fire in ---"
cp "$SRC_TAMPER" "${SRC_TAMPER}.guardbak" || exit 2
printf '\n# guardtest tamper\n' >> "$SRC_TAMPER"
case_run "H1 pinned code tamper caught" 2 "SOURCE PIN FAILED for 'src/data/yaw_rotation.py'"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
restore_sources
case_run "H2 pins pass once restored"   0 "source pins OK"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
mutate "$BF_CFG" 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "H3 an edited arm config trips the byte pin before the contract" 2 \
         "SOURCE PIN FAILED for '${BF_CFG}'"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
restore_cfgs
cp "$CLOSURE_TAMPER" "${CLOSURE_TAMPER}.guardbak" || exit 2
printf '\n# guardtest tamper\n' >> "$CLOSURE_TAMPER"
case_run "H4 a dirty training closure is refused" 2 "tracked training closure is dirty" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
restore_sources
case_run "H5 and clean again once restored" 0 "tree clean across the training closure" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
expect "H6 the HAA splits are inside the closure (r2-B1)" \
  "$(grep -qF 'data/HAA/train_base.json data/HAA/val_base.json' "$LAUNCH" && echo 1)"

echo "--- I. the ACTUAL train.py argv, per arm and per mode ---"
# Capture, THEN grep: piping the launcher into `grep -m1` would SIGPIPE it, the
# same shape section L forbids in the launcher.
argv_of() {
  local out; out="$(env "${OKENV[@]}" ARM="$1" GPU="$2" MODE="$3" bash "$LAUNCH" 2>&1)"
  grep -m1 '^ARGV: ' <<<"$out"
}
FULL_BF="$(argv_of BF 0 FULL)"
SMOKE_BF="$(argv_of BF 0 SMOKE)"
FULL_P1="$(argv_of P1 0 FULL)"
FULL_YAW="$(argv_of YAW 1 FULL)"

expect "I1 FULL endpoint exactly 1000"       "$(eq "$(argval "$FULL_BF" --max-steps)" 1000)"
expect "I2 FULL cadence exactly 10"          "$(eq "$(argval "$FULL_BF" --checkpoint-every)" 10)"
expect "I3 FULL val-every exactly 10"        "$(eq "$(argval "$FULL_BF" --val-every)" 10)"
expect "I4 SMOKE endpoint exactly 20"        "$(eq "$(argval "$SMOKE_BF" --max-steps)" 20)"
expect "I5 SMOKE cadence 1e6 (no ckpt)"      "$(eq "$(argval "$SMOKE_BF" --checkpoint-every)" 1000000)"
expect "I6 SMOKE val-every 1e6"              "$(eq "$(argval "$SMOKE_BF" --val-every)" 1000000)"
expect "I7 batch exactly 16"                 "$(eq "$(argval "$FULL_BF" --batch-size)" 16)"
expect "I8 accumulation exactly 4"           "$(eq "$(argval "$FULL_BF" --accum-batches)" 4)"
expect "I9 single GPU"                       "$(eq "$(argval "$FULL_BF" --num-gpus)" 1)"
expect "I10 precision bf16-mixed"            "$(eq "$(argval "$FULL_BF" --precision)" bf16-mixed)"
expect "I11 seed exactly 42"                 "$(eq "$(argval "$FULL_BF" --seed)" 42)"
expect "I12 num-workers exactly 8"           "$(eq "$(argval "$FULL_BF" --num-workers)" 8)"
expect "I13 train split is haa_train.json"   "$(eq "$(argval "$FULL_BF" --dataset-config)" src/configs/dataset_configs/HAA/train/haa_train.json)"
expect "I14 val split is haa_val.json"       "$(eq "$(argval "$FULL_BF" --val-dataset-config)" src/configs/dataset_configs/HAA/eval/haa_val.json)"
expect "I15 VAE is the frozen pretransform"  "$(eq "$(argval "$FULL_BF" --pretransform-ckpt-path)" weights/FLAC/VAE.safetensors)"
expect "I16 BF uses the BF config"           "$(eq "$(argval "$FULL_BF" --model-config)" "$BF_CFG")"
expect "I17 YAW uses the YAW config"         "$(eq "$(argval "$FULL_YAW" --model-config)" "$YAW_CFG")"
expect "I18 P1 uses the STOCK config"        "$(eq "$(argval "$FULL_P1" --model-config)" "$STOCK_CFG")"
expect "I19 BF inits from the BF weights"    "$(eq "$(argval "$FULL_BF" --pretrained-ckpt-path)" "${INIT_OK}/HAA_init_BF.ckpt")"
expect "I20 P1 inits from the P1 weights"    "$(eq "$(argval "$FULL_P1" --pretrained-ckpt-path)" "${INIT_OK}/HAA_init_P1.ckpt")"

echo "--- I'. namespace separation (no arm can write into another's) ---"
expect "I21 BF FULL save-dir"   "$(eq "$(argval "$FULL_BF" --save-dir)" outputs_FLAC/exp19_HAA_BF)"
expect "I22 BF SMOKE save-dir"  "$(eq "$(argval "$SMOKE_BF" --save-dir)" outputs_FLAC/exp19_HAA_BF_smoke)"
expect "I23 P1 FULL save-dir"   "$(eq "$(argval "$FULL_P1" --save-dir)" outputs_FLAC/exp19_HAA_P1)"
expect "I24 YAW FULL save-dir"  "$(eq "$(argval "$FULL_YAW" --save-dir)" outputs_FLAC/exp19_HAA_YAW)"
expect "I25 wandb run name per arm"        "$(eq "$(argval "$FULL_BF" --name)" FLAC_exp19_HAA_BF)"
expect "I26 wandb experiment per arm"      "$(eq "$(argval "$FULL_BF" --experiment-name)" exp19_HAA_BF)"
expect "I27 smoke wandb identity differs"  "$(eq "$(argval "$SMOKE_BF" --experiment-name)" exp19_HAA_BF_smoke)"
ALLDIRS="$(printf '%s\n' \
  "$(argval "$FULL_BF" --save-dir)" "$(argval "$SMOKE_BF" --save-dir)" \
  "$(argval "$FULL_P1" --save-dir)" "$(argval "$FULL_YAW" --save-dir)" | sort)"
expect "I28 the four namespaces are pairwise distinct" \
  "$(eq "$(printf '%s\n' "$ALLDIRS" | sort -u | wc -l)" 4)"
expect "I29 argval matches the flag as a whole token" \
  "$(eq "$(argval 'ARGV: python train.py --not-max-steps 999 --max-steps 1000' --max-steps)" 1000)"
expect "I30 eq rejects a value substring" "$(eq "$(eq 10000 1000)" '')"

echo "--- J. locks: per-arm and per-GPU, held for the WHOLE run (r2-B2) ---"
case_run "J1 a launcher that must take the gate lock is refused while it is held" 2 \
         "another exp_19 launcher holds"  DRY_RUN=1 MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" \
         PROBE_CMD="$STUB_PASS" HAA_ROOT="$DATA_OK" TMPDIR="$DRYTMP" ARM=BF GPU=0 MODE=FULL
case_run "J2 the caller-held path is accepted"  0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
# Hold the per-arm / per-GPU locks from a SEPARATE process, as a concurrent run
# would, and prove a second launch cannot claim the same arm or the same card.
flock -x "${EXPDIR}/.haa_ft_BF.lock" -c 'sleep 8' & ARMPID=$!
flock -x "${EXPDIR}/.haa_gpu1.lock"  -c 'sleep 8' & GPUPID=$!
sleep 1
case_run "J3 the same ARM cannot be launched twice"  2 "arm BF is already running or gating" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL
case_run "J4 the same GPU cannot be claimed twice"   2 "GPU 1 is already reserved"           "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL
wait "$ARMPID" "$GPUPID" 2>/dev/null
case_run "J5 both are free again once released"      0 "DRY_RUN: all gates passed"  "${OKENV[@]}" ARM=YAW GPU=1 MODE=FULL
expect "J6 the run locks are NOT closed before exec (the child inherits them)" \
  "$(grep -qE 'exec 9>&-' "$LAUNCH" && ! grep -qE 'exec 8>&-|exec 7>&-' "$LAUNCH" && echo 1)"

echo "--- K. post-run verdicts, EXECUTED via the rehearsal mode ---"
# Every case below runs the launcher's real post-run block against a stubbed
# train run. Codex r2: no dynamic case exercised rc propagation, endpoint
# rejection, banner rejection, NaN rejection, 410/1000 or the smoke ckpt refusal.
GOOD_FULL_LOG='printf "Epoch 0: 100%%|##| 1000/1000 [10:00<00:00, train/loss=0.31]\`Trainer.fit\` stopped: \`max_steps=1000\` reached.\n"'
MK_CKPTS='mkdir -p "$SAVEDIR" && touch "$SAVEDIR/epoch=0-step=410.ckpt" "$SAVEDIR/epoch=0-step=1000.ckpt"'
REHEARSE_ENV=(DRY_RUN=2 MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" TMPDIR="$DRYTMP"
              PROBE_CMD="$STUB_PASS" HAA_ROOT="$DATA_OK" GATE_LOCK_HELD_BY_CALLER=1 LOGGER=none)
# Each rehearsal case gets its OWN namespace. Sharing one made every case after
# the first trip the FULL namespace-occupancy gate on the PREVIOUS case's
# checkpoints — a fixture defect that hid the post-run gates behind an earlier
# (correct) refusal, and exactly the kind of cross-case leakage the per-arm
# namespaces exist to prevent in production.
rdir() { local d="${REHEARSE}/$1"; mkdir -p "$d" || exit 2; printf '%s' "$d"; }

case_run "K1 a healthy FULL rehearsal ends rc=0" 0 "final rc=0" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k1)" ARM=BF GPU=0 MODE=FULL TRAIN_CMD="${MK_CKPTS} && ${GOOD_FULL_LOG}"
case_run "K2 a failing train propagates its rc" 7 "final rc=7" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k2)" ARM=BF GPU=0 MODE=FULL TRAIN_CMD="${MK_CKPTS} && ${GOOD_FULL_LOG} && exit 7"
case_run "K3 a log without the endpoint marker is rejected" 6 "ENDPOINT NOT REACHED" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k3)" ARM=BF GPU=0 MODE=FULL TRAIN_CMD="${MK_CKPTS} && echo 'Epoch 0: 10/1000'"
case_run "K4 a log that merely QUOTES the marker is rejected" 6 "ENDPOINT NOT REACHED" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k4)" ARM=BF GPU=0 MODE=FULL \
         TRAIN_CMD="${MK_CKPTS} && echo 'diagnostic: \`Trainer.fit\` stopped: \`max_steps=1000\` reached. was not found'"
case_run "K5 non-finite loss is rejected even with the endpoint reached" 4 "NON-FINITE LOSS" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k5)" ARM=BF GPU=0 MODE=FULL \
         TRAIN_CMD="${MK_CKPTS} && echo 'train/loss=nan' && ${GOOD_FULL_LOG}"
case_run "K6 FULL without the step-410 reading is rejected" 6 "REQUIRED CHECKPOINT(S) MISSING" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k6)" ARM=BF GPU=0 MODE=FULL \
         TRAIN_CMD='mkdir -p "$SAVEDIR" && touch "$SAVEDIR/epoch=0-step=1000.ckpt" && '"${GOOD_FULL_LOG}"
case_run "K7 FULL without the endpoint checkpoint is rejected" 6 "REQUIRED CHECKPOINT(S) MISSING" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k7)" ARM=BF GPU=0 MODE=FULL \
         TRAIN_CMD='mkdir -p "$SAVEDIR" && touch "$SAVEDIR/epoch=0-step=410.ckpt" && '"${GOOD_FULL_LOG}"
case_run "K8 a YAW run whose banner is absent is rejected" 3 "TREATMENT BANNER" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k8)" ARM=YAW GPU=1 MODE=FULL TRAIN_CMD="${MK_CKPTS} && ${GOOD_FULL_LOG}"
case_run "K9 a YAW run WITH the banner passes" 0 "treatment banner: FOUND" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k9)" ARM=YAW GPU=1 MODE=FULL \
         TRAIN_CMD="${MK_CKPTS} && echo 'yaw_aug ENABLED img_w=512 seed=42' && ${GOOD_FULL_LOG}"
case_run "K10 a banner PARAPHRASE does not satisfy the whole-line check" 3 "TREATMENT BANNER" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k10)" ARM=YAW GPU=1 MODE=FULL \
         TRAIN_CMD="${MK_CKPTS} && echo 'note: yaw_aug ENABLED img_w=512 seed=42 (preflight)' && ${GOOD_FULL_LOG}"
case_run "K11 a SMOKE that wrote a checkpoint is rejected" 5 "SMOKE wrote 1 checkpoint" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k11)" ARM=BF GPU=0 MODE=SMOKE \
         TRAIN_CMD='mkdir -p "$SAVEDIR" && touch "$SAVEDIR/epoch=0-step=20.ckpt" && printf "\`Trainer.fit\` stopped: \`max_steps=20\` reached.\n"'
case_run "K12 a clean SMOKE reports zero checkpoints" 0 "smoke checkpoints: 0" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k12)" ARM=BF GPU=0 MODE=SMOKE \
         TRAIN_CMD='printf "\`Trainer.fit\` stopped: \`max_steps=20\` reached.\n"'
case_run "K13 rehearsal artifacts never land in a production namespace" 0 \
         "save-dir=${REHEARSE}/k13/exp19_HAA_BF" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k13)" ARM=BF GPU=0 MODE=FULL TRAIN_CMD="${MK_CKPTS} && ${GOOD_FULL_LOG}"
case_run "K14 rehearsal requires TRAIN_CMD" 2 "requires TRAIN_CMD" \
         DRY_RUN=2 REHEARSAL_DIR="$REHEARSE" MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" TMPDIR="$DRYTMP" \
         PROBE_CMD="$STUB_PASS" HAA_ROOT="$DATA_OK" GATE_LOCK_HELD_BY_CALLER=1 ARM=BF GPU=0 MODE=FULL
case_run "K15 rehearsal requires a private REHEARSAL_DIR" 2 "requires REHEARSAL_DIR" \
         DRY_RUN=2 MANIFEST="$MAN_OK" INIT_DIR="$INIT_OK" PROBE_CMD="$STUB_PASS" TMPDIR="$DRYTMP" \
         HAA_ROOT="$DATA_OK" GATE_LOCK_HELD_BY_CALLER=1 ARM=BF GPU=0 MODE=FULL TRAIN_CMD=true
# (r2-B6) The init is re-hashed at the point of consumption; swap it AFTER the
# gates would have passed by making TRAIN_CMD irrelevant — the check runs before
# TRAIN_CMD, so corrupting the file mid-suite is what this exercises.
cp "${INIT_OK}/HAA_init_P1.ckpt" "${TMPROOT}/P1.orig" || exit 2
printf 'tampered' >> "${INIT_OK}/HAA_init_P1.ckpt"
case_run "K16 an init replaced after gate 3 is caught before exec" 2 "INIT SHA MISMATCH" \
         "${REHEARSE_ENV[@]}" REHEARSAL_DIR="$(rdir k16)" ARM=P1 GPU=0 MODE=FULL TRAIN_CMD="${MK_CKPTS} && ${GOOD_FULL_LOG}"
cp "${TMPROOT}/P1.orig" "${INIT_OK}/HAA_init_P1.ckpt" || exit 2

echo "--- M. resource floors: raisable, never lowerable ---"
case_run "M1 a raised VRAM floor is honoured"  2 "< required 99999999 MiB" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MIN_FREE_MB=99999999
case_run "M2 a raised disk floor is honoured"  2 "< required 999999999 MiB" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MIN_FREE_DISK_MB=999999999
case_run "M3 a LOWERED VRAM floor is clamped back to the registered one" 0 \
         "MIN_FREE_MB=1 is below the registered floor 20000; CLAMPED" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MIN_FREE_MB=1
case_run "M4 a LOWERED disk floor is clamped back to the registered one" 0 \
         "MIN_FREE_DISK_MB=1 is below the registered floor 72000; CLAMPED" \
         "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MIN_FREE_DISK_MB=1
case_run "M5 the clamped floor is the one actually enforced" 0 \
         ">= 20000 MiB floor" "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL MIN_FREE_MB=1

echo "--- L. properties of the launcher that must not silently regress ---"
expect "L1 the budget is not env-overridable" \
  "$(! grep -qE 'FULL_STEPS:-|FULL_CADENCE:-|SMOKE_STEPS:-|SMOKE_CADENCE:-|FULL_VAL_EVERY:-|MID_STEPS:-|BATCH:-|ACCUM:-|SEED:-' "$LAUNCH" && echo 1)"
expect "L2 post-run log reading never uses 'tr | grep -q' (pipefail SIGPIPE)" \
  "$(! grep -qE "tr '..' '..' <[^>]*\| *grep" "$LAUNCH" && echo 1)"
expect "L3 the endpoint marker carries Lightning's backticks" \
  "$(grep -qF 'MARKER="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."' "$LAUNCH" && echo 1)"
expect "L4 the floors are clamped, never lowered, in production" \
  "$(grep -qF 'CLAMPED (floors may be raised, never lowered)' "$LAUNCH" && echo 1)"
expect "L5 the preflight never prints the banner text (self-satisfaction guard)" \
  "$(grep -qxF 'yaw_aug ENABLED img_w=512 seed=42' \
      <<<"$(tr '\r' '\n' <<<"$(env "${OKENV[@]}" ARM=YAW GPU=0 MODE=FULL bash "$LAUNCH" 2>&1)")" \
     && echo 0 || echo 1)"
expect "L6 the probe is given the arm's init via --ckpt-path" \
  "$(grep -qF -- '--ckpt-path "$INIT"' "$LAUNCH" && echo 1)"
expect "L7 the pinned probe sha is the current file" \
  "$(grep -qF "$(sha256sum "${EXPDIR}/probe_haa_fa_invariance.py" | cut -d' ' -f1)" "$LAUNCH" && echo 1)"
expect "L8 both HAA split files are content-pinned" \
  "$(grep -qF "$(sha256sum data/HAA/train_base.json | cut -d' ' -f1)" "$LAUNCH" \
     && grep -qF "$(sha256sum data/HAA/val_base.json | cut -d' ' -f1)" "$LAUNCH" && echo 1)"
expect "L9 train.py output is piped synchronously (PIPESTATUS), not async tee" \
  "$(grep -qF 'rc="${PIPESTATUS[0]}"' "$LAUNCH" && echo 1)"
# Asserted on the ASSIGNMENT, not on the string: the launcher's comment explains
# why the shared `.dryrun_logs` directory was abandoned, and a bare substring
# search would read that explanation as the defect it describes.
expect "L10 DRY output goes to a per-invocation mktemp dir, not a shared one" \
  "$(grep -qF 'mktemp -d -t haa_ft_dryrun' "$LAUNCH" \
     && ! grep -qE '^\s*LOGDIR=.*\.dryrun_logs' "$LAUNCH" && echo 1)"
expect "L11 two dry runs never share a log directory (non-vacuity for L10)" \
  "$(a="$(env "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL bash "$LAUNCH" 2>&1 | sed -n 's/^dry-run log dir: //p')"; \
     b="$(env "${OKENV[@]}" ARM=BF GPU=0 MODE=FULL bash "$LAUNCH" 2>&1 | sed -n 's/^dry-run log dir: //p')"; \
     [ -n "$a" ] && [ "$a" != "$b" ] && echo 1)"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
echo "log: ${GUARDLOG}"
[ "$FAIL" -eq 0 ]
