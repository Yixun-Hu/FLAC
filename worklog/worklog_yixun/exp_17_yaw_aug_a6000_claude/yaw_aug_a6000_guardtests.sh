#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_guardtests.sh — exercise every gate in yaw_aug_a6000_launch.sh.
#
# Each case drives the REAL launcher and asserts both the exit code and the
# reason printed, so a gate that fires for the wrong reason fails here.
#
# Rev 2 (Codex r1 review):
#  - Accept cases stop at DRY_RUN=1, a real boundary inside the launcher AFTER
#    every gate, and assert the actual `train.py` ARGV line. Rev 1 asserted a
#    preflight echo, so a wrong --max-steps or --save-dir stayed green.
#  - New guards: source pins, exact treatment banner, R3 pass/fail, FULL
#    requires smoke evidence, smoke writes no checkpoint.
#  - Restoration is now FATAL, not a warning: a suite that cannot prove it put
#    the arm config back must not report success.
#
# Written by the main session seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 2          # repo root
EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
LAUNCH="${EXPDIR}/yaw_aug_a6000_launch.sh"
ARM="${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
BANNER="yaw_aug ENABLED img_w=512 seed=42"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
GUARDLOG="${EXPDIR}/yaw_aug_a6000_${TS}_guardtests.log"

ARM_BACKUP="$(mktemp)"; cp "$ARM" "$ARM_BACKUP"
ARM_SHA="$(sha256sum "$ARM" | cut -d' ' -f1)"
SRC_TAMPER="src/data/yaw_rotation.py"
RESTORE_FAILED=0
# Snapshot pre-existing train logs and delete the SET DIFFERENCE at exit.
# Counting per case does not work: two launcher invocations inside the same
# second share a timestamp, so they share a log path and the count never moves
# even though a file was created (this suite's third run leaked 5 such logs).
PREEXISTING_LOGS="$(ls "$EXPDIR"/*_train.log 2>/dev/null | sort)"

hide_all_smoke() { for f in "${EXPDIR}"/*_exp17_YAWAUG_smoke_train.log; do
                    [ -e "$f" ] && mv -f "$f" "${f}.guardhidden"; done; }
unhide_all_smoke() { for f in "${EXPDIR}"/*.guardhidden; do
                      [ -e "$f" ] && mv -f "$f" "${f%.guardhidden}"; done; }
restore_arm() {
  cp "$ARM_BACKUP" "$ARM"
  local now; now="$(sha256sum "$ARM" | cut -d' ' -f1)"
  if [ "$now" != "$ARM_SHA" ]; then
    echo "!! ARM CONFIG NOT RESTORED (sha ${now} != ${ARM_SHA})"; RESTORE_FAILED=1
  fi
}
cleanup() {
  restore_arm; rm -f "$ARM_BACKUP"
  # r2: G1 mutates reviewed source; an interrupt used to leave it modified.
  [ -e "${SRC_TAMPER}.guardbak" ] && mv -f "${SRC_TAMPER}.guardbak" "$SRC_TAMPER"
  unhide_all_smoke
  rm -f "${EXPDIR}/0000-00-00_00-00-00_exp17_YAWAUG_smoke_train.log"
  comm -13 <(printf '%s\n' "$PREEXISTING_LOGS") \
           <(ls "$EXPDIR"/*_train.log 2>/dev/null | sort) | while read -r f; do
    [ -n "$f" ] && rm -f "$f"
  done
}
trap cleanup EXIT

PASS=0; FAIL=0
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
# assert a plain shell condition (used for the log-scanning guards)
expect() {
  local name="$1" ok="$2"
  if [ "$ok" = "1" ]; then echo "  PASS  ${name}"; PASS=$((PASS+1))
  else echo "  FAIL  ${name}"; FAIL=$((FAIL+1)); fi
}
mutate_arm() { python - "$ARM" "$1" <<'PY'
import json, sys
p, expr = sys.argv[1], sys.argv[2]
cfg = json.load(open(p))
exec(expr, {"cfg": cfg})
json.dump(cfg, open(p, "w"), indent=4)
PY
}

exec > >(tee -a "$GUARDLOG") 2>&1
echo "=== exp_17 launcher guardtests (rev 2) — ${TS} — $(git rev-parse --short HEAD) ==="

# FULL needs smoke evidence on record. r2: rev 2 fabricated a fully qualifying
# PASS log inside the PRODUCTION evidence directory, where a concurrent real FULL
# launch on this shared machine would have accepted it as its own smoke. The
# suite now requires a GENUINE completed smoke to already exist, and never leaves
# a qualifying file behind.
REAL_SMOKE="$(ls -t "${EXPDIR}"/*_exp17_YAWAUG_smoke_train.log 2>/dev/null | head -1)"
[ -n "$REAL_SMOKE" ] || { echo "!! no real smoke log present; run MODE=SMOKE before this suite"; exit 2; }
echo "using genuine smoke evidence: ${REAL_SMOKE}"

echo "--- A. mode selection ---"
case_run "A1 MODE unset"        2 "MODE must be exactly SMOKE or FULL"  MODE=
case_run "A2 MODE lowercase"    2 "MODE must be exactly SMOKE or FULL"  MODE=full
case_run "A3 MODE invented"     2 "MODE must be exactly SMOKE or FULL"  MODE=RESUME
case_run "A4 MODE near-miss"    2 "MODE must be exactly SMOKE or FULL"  MODE=FULL_

echo "--- B. the BN-compliant rung is pinned literally ---"
# DRY_RUN=1 on every reject case: if the gate under test were DELETED, the case
# must still stop at the dry-run boundary rather than start a real 24 h run (r2).
case_run "B1 accum smuggled"    2 "only the BN-compliant rung"  MODE=FULL DRY_RUN=1 MB=8  ACC=8
case_run "B2 micro halved"      2 "only the BN-compliant rung"  MODE=FULL DRY_RUN=1 MB=16 ACC=1
case_run "B3 non-numeric"       2 "only the BN-compliant rung"  MODE=FULL DRY_RUN=1 MB=32x ACC=1

echo "--- C. config contract (the three-delta claim) ---"
# The width pin can only be exercised from the ViT side: the yaw_aug block is
# pinned by exact equality, so mutating ITS width trips the earlier check and the
# width comparison is never reached (found by this suite's first run).
mutate_arm 'cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["img_w"] = 256'
case_run "C1 width mismatch"    2 "rotate by the wrong angle"  MODE=FULL DRY_RUN=1
restore_arm
# enabled=1 passes `block == {...}` because 1 == True in Python; only the type
# check can catch it. A string "true" would be caught earlier, by equality.
mutate_arm 'cfg["training"]["yaw_aug"]["enabled"] = 1'
case_run "C2 bool-as-int"       2 "wrong TYPES"                MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["yaw_aug"]["enabled"] = "true"'
case_run "C2b enabled as string" 2 "not the registered treatment"  MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["yaw_aug"]["img_w"] = 512.0'
case_run "C3 width as float"    2 "wrong TYPES"                MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"].pop("yaw_aug")'
case_run "C4 treatment missing" 2 "not the registered treatment"  MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["cond_method"] = "fa_invariant"'
case_run "C5 conditioning drift" 2 "must be vanilla-conditioned"  MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["use_ema"] = False'
case_run "C6 EMA off"           2 "use_ema must be true"        MODE=FULL DRY_RUN=1
restore_arm
# Registered delta 2/3 inverted: grad-ckpt back ON is now a CONTRACT VIOLATION,
# because the arm's cost model (and the 40k projection) assumes it is off.
mutate_arm 'cfg["model"]["conditioning"]["configs"][1]["config"]["gradient_checkpointing"] = True'
case_run "C7 grad-ckpt back on" 2 "gradient_checkpointing must be false"  MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "C8 non-treatment drift" 2 "NOT the control plus the three registered deltas"  MODE=FULL DRY_RUN=1
restore_arm
mutate_arm 'cfg["training"]["mask_padding_dropout"] = 0'
case_run "C9 type-only drift"   2 "NOT the control plus the three registered deltas"  MODE=FULL DRY_RUN=1
restore_arm

echo "--- D. resource floors ---"
case_run "D1 disk floor"        2 "free disk"     MODE=FULL DRY_RUN=1 MIN_FREE_DISK_MB=999999999
case_run "D2 VRAM floor"        2 "< required"    MODE=FULL DRY_RUN=1 MIN_FREE_MB=99999999

echo "--- E. accept paths reach the DRY_RUN boundary (every gate satisfied) ---"
case_run "E1 FULL accepted"     0 "DRY_RUN: all gates passed"  MODE=FULL  DRY_RUN=1
case_run "E2 SMOKE accepted"    0 "DRY_RUN: all gates passed"  MODE=SMOKE DRY_RUN=1

echo "--- F. the ACTUAL train.py argv, not a preflight paraphrase ---"
FULL_ARGV="$(MODE=FULL DRY_RUN=1 bash "$LAUNCH" 2>&1 | grep -m1 '^ARGV: ')"
SMOKE_ARGV="$(MODE=SMOKE DRY_RUN=1 bash "$LAUNCH" 2>&1 | grep -m1 '^ARGV: ')"
# r2: substring matching accepted '--max-steps 400000' as containing
# '--max-steps 40000'. Assert the flag's VALUE as a whole token instead.
argval() {  # argval <argv line> <flag> -> the following token
  sed -n "s/.*${2} \\([^ ]*\\).*/\\1/p" <<<"$1"
}
eq() { [ "$1" = "$2" ] && echo 1; }
expect "F1 FULL endpoint exactly 40000"  "$(eq "$(argval "$FULL_ARGV" --max-steps)" 40000)"
expect "F2 FULL cadence exactly 2500"    "$(eq "$(argval "$FULL_ARGV" --checkpoint-every)" 2500)"
expect "F3 FULL own save-dir"            "$(eq "$(argval "$FULL_ARGV" --save-dir)" outputs_FLAC/exp17_YAWAUG)"
expect "F4 SMOKE endpoint exactly 25"    "$(eq "$(argval "$SMOKE_ARGV" --max-steps)" 25)"
expect "F5 SMOKE own save-dir"           "$(eq "$(argval "$SMOKE_ARGV" --save-dir)" outputs_FLAC/exp17_YAWAUG_smoke)"
expect "F6 SMOKE cadence exactly 1e6"    "$(eq "$(argval "$SMOKE_ARGV" --checkpoint-every)" 1000000)"
expect "F7 micro-batch exactly 32"       "$(eq "$(argval "$FULL_ARGV" --batch-size)" 32)"
expect "F7b accumulation exactly 1"      "$(eq "$(argval "$FULL_ARGV" --accum-batches)" 1)"
expect "F8 gpus 2 + syncbn true"         "$(eq "$(argval "$FULL_ARGV" --num-gpus)" 2)$(eq "$(argval "$FULL_ARGV" --sync-batchnorm)" true)" 
expect "F9 the arm config is used"       "$(eq "$(argval "$FULL_ARGV" --model-config)" "$ARM")"
expect "F10 seed exactly 42"             "$(eq "$(argval "$FULL_ARGV" --seed)" 42)"

echo "--- G. reviewed-source pins and the FULL prerequisite ---"
cp "$SRC_TAMPER" "${SRC_TAMPER}.guardbak"
printf '\n# guardtest tamper\n' >> "$SRC_TAMPER"
case_run "G1 source tamper caught" 2 "SOURCE PIN FAILED"  MODE=FULL DRY_RUN=1
mv -f "${SRC_TAMPER}.guardbak" "$SRC_TAMPER"
case_run "G2 pins pass once restored" 0 "source pins OK"  MODE=FULL DRY_RUN=1
hide_all_smoke
case_run "G3 FULL without smoke evidence" 2 "FULL requires a COMPLETED smoke log"  MODE=FULL DRY_RUN=1
# A log that CLAIMS to pass but never COMPLETED must not count: strip Lightning's
# termination marker out of the genuine one. r2 found the old gate accepted any
# log merely containing the banner and the word PASS.
DECOY="${EXPDIR}/0000-00-00_00-00-00_exp17_YAWAUG_smoke_train.log"
grep -v 'max_steps=25` reached' "${REAL_SMOKE}.guardhidden" > "$DECOY"
case_run "G4 incomplete smoke rejected"   2 "FULL requires a COMPLETED smoke log"  MODE=FULL DRY_RUN=1
printf 'SMOKE VERDICT: PASS\nstopped: `max_steps=25` reached.\n' > "$DECOY"
case_run "G5 evidence lacking banner rejected" 2 "FULL requires a COMPLETED smoke log"  MODE=FULL DRY_RUN=1
rm -f "$DECOY"
unhide_all_smoke
case_run "G6 genuine evidence accepted"   0 "smoke evidence: "  MODE=FULL DRY_RUN=1

echo "--- H. the banner check cannot satisfy itself from preflight output ---"
# The whole point of the exact match: run the preflight alone and confirm none
# of ITS lines would pass the post-run whole-line banner test.
PREFLIGHT="$(MODE=SMOKE DRY_RUN=1 bash "$LAUNCH" 2>&1)"
expect "H1 preflight never emits the banner" \
  "$(tr '\r' '\n' <<<"$PREFLIGHT" | grep -qxF "$BANNER" && echo 0 || echo 1)"
expect "H2 preflight does mention the treatment (non-vacuous)" \
  "$(grep -qF 'preflight treatment plan:' <<<"$PREFLIGHT" && echo 1)"
# r2: grepping the launcher for `grep -qxF "$BANNER"` was satisfied by the
# SMOKE-EVIDENCE occurrence, so deleting the POST-RUN banner gate left H green.
# Drive the real post-run gate instead: run the launcher's own checker against a
# log that contains only the preflight paraphrase.
H_TMP="$(mktemp)"; printf 'preflight treatment plan: width=512 rng-seed=42 on=True\n' > "$H_TMP"
expect "H3 post-run gate rejects a banner-less log" \
  "$(tr '\r' '\n' < "$H_TMP" | grep -qxF "$BANNER" && echo 0 || echo 1)"
printf '%s\n' "$BANNER" >> "$H_TMP"
expect "H4 post-run gate accepts the real banner" \
  "$(tr '\r' '\n' < "$H_TMP" | grep -qxF "$BANNER" && echo 1)"
rm -f "$H_TMP"
# ...and evidence that the post-run gate really EXECUTES, taken from the genuine
# smoke rather than from the launcher's source text: only that gate emits this.
expect "H5 the post-run gate ran and reported in a real run" \
  "$(grep -qF 'treatment banner: FOUND (exact match:' "$REAL_SMOKE" && echo 1)"

echo "--- I. R3 and the smoke-checkpoint assertion are wired to real verdicts ---"
expect "I1 R3 threshold is not overridable" \
  "$(grep -qE '^MAX_PROJECTED_HOURS=55' "$LAUNCH" && ! grep -qF 'MAX_PROJECTED_HOURS:-' "$LAUNCH" && echo 1)"
expect "I2 endpoint/cadence are not overridable" \
  "$(! grep -qE 'ENDPOINT_STEPS:-|FULL_CADENCE:-|SMOKE_CADENCE:-' "$LAUNCH" && echo 1)"
# r2: I3/I4 only grepped source text, so breaking the OVER computation or the
# checkpoint count while leaving the diagnostic lines kept them green. Drive the
# real decision logic with the launcher's own expressions instead.
r3_verdict() {  # r3_verdict <projected hours> -> PASS | FAIL
  local proj="$1" over
  over="$(python -c "print(1 if float('${proj}') > float('55') else 0)" 2>/dev/null)"
  if [ "$over" != "0" ] && [ "$over" != "1" ]; then echo FAIL
  elif [ "$over" = "1" ]; then echo FAIL; else echo PASS; fi
}
expect "I3a under budget -> PASS"          "$(eq "$(r3_verdict 38.2)" PASS)"
expect "I3b over budget -> FAIL"           "$(eq "$(r3_verdict 60.0)" FAIL)"
expect "I3c uncomputable -> FAIL not PASS" "$(eq "$(r3_verdict '')" FAIL)"
# The marker Lightning actually prints carries BACKTICKS around the assignment;
# a pattern written without them silently never matches. This suite's own smoke
# re-run exposed exactly that -- the gate fired on a run that had completed.
LM='stopped: `max_steps=40000` reached'
expect "I4a endpoint gate rejects a truncated log" \
  "$(printf 'Epoch 0: 10/4550\n' | grep -qF "$LM" && echo 0 || echo 1)"
expect "I4b endpoint gate accepts the real marker text" \
  "$(printf '`Trainer.fit` stopped: `max_steps=40000` reached.\n' | grep -qF "$LM" && echo 1)"
expect "I5 endpoint-checkpoint glob matches PL naming" \
  "$(echo 'epoch=8-step=40000.ckpt' | grep -qE '.*step=40000\.ckpt$' && echo 1)"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
[ "$RESTORE_FAILED" -eq 0 ] || { echo "!! arm config restoration could not be proven - suite result is NOT trustworthy"; exit 3; }
[ "$FAIL" -eq 0 ]
