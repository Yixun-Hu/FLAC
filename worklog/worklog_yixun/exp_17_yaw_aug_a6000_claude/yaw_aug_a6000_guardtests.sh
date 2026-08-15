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
CREATED_LOGS=()
FAKE_SMOKE=""
RESTORE_FAILED=0

restore_arm() {
  cp "$ARM_BACKUP" "$ARM"
  local now; now="$(sha256sum "$ARM" | cut -d' ' -f1)"
  if [ "$now" != "$ARM_SHA" ]; then
    echo "!! ARM CONFIG NOT RESTORED (sha ${now} != ${ARM_SHA})"; RESTORE_FAILED=1
  fi
}
cleanup() {
  restore_arm; rm -f "$ARM_BACKUP"
  [ -n "$FAKE_SMOKE" ] && rm -f "$FAKE_SMOKE"
  for f in "${CREATED_LOGS[@]:-}"; do [ -n "$f" ] && rm -f "$f"; done
}
trap cleanup EXIT

PASS=0; FAIL=0
# case_run <name> <expected_rc> <expected_substring> <env assignments...>
case_run() {
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  local before; before="$(ls "$EXPDIR"/*_train.log 2>/dev/null | wc -l)"
  local out rc
  out="$(env "$@" bash "$LAUNCH" 2>&1)"; rc=$?
  local after; after="$(ls "$EXPDIR"/*_train.log 2>/dev/null | wc -l)"
  while IFS= read -r f; do CREATED_LOGS+=("$f"); done < <(ls -t "$EXPDIR"/*_train.log 2>/dev/null | head -n $(( after - before )) )
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

# A FULL run needs smoke evidence on record. Provide a synthetic one so the FULL
# cases can reach the gates under test, and assert its ABSENCE separately (G3).
FAKE_SMOKE="${EXPDIR}/9999-99-99_00-00-00_exp17_YAWAUG_smoke_train.log"
printf '%s\nSMOKE VERDICT: PASS\n' "$BANNER" > "$FAKE_SMOKE"

echo "--- A. mode selection ---"
case_run "A1 MODE unset"        2 "MODE must be exactly SMOKE or FULL"  MODE=
case_run "A2 MODE lowercase"    2 "MODE must be exactly SMOKE or FULL"  MODE=full
case_run "A3 MODE invented"     2 "MODE must be exactly SMOKE or FULL"  MODE=RESUME
case_run "A4 MODE near-miss"    2 "MODE must be exactly SMOKE or FULL"  MODE=FULL_

echo "--- B. the BN-compliant rung is pinned literally ---"
case_run "B1 accum smuggled"    2 "only the BN-compliant rung"  MODE=FULL MB=8  ACC=8
case_run "B2 micro halved"      2 "only the BN-compliant rung"  MODE=FULL MB=16 ACC=1
case_run "B3 non-numeric"       2 "only the BN-compliant rung"  MODE=FULL MB=32x ACC=1

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
case_run "D1 disk floor"        2 "free disk"     MODE=FULL MIN_FREE_DISK_MB=999999999
case_run "D2 VRAM floor"        2 "< required"    MODE=FULL MIN_FREE_MB=99999999

echo "--- E. accept paths reach the DRY_RUN boundary (every gate satisfied) ---"
case_run "E1 FULL accepted"     0 "DRY_RUN: all gates passed"  MODE=FULL  DRY_RUN=1
case_run "E2 SMOKE accepted"    0 "DRY_RUN: all gates passed"  MODE=SMOKE DRY_RUN=1

echo "--- F. the ACTUAL train.py argv, not a preflight paraphrase ---"
FULL_ARGV="$(MODE=FULL DRY_RUN=1 bash "$LAUNCH" 2>&1 | grep -m1 '^ARGV: ')"
SMOKE_ARGV="$(MODE=SMOKE DRY_RUN=1 bash "$LAUNCH" 2>&1 | grep -m1 '^ARGV: ')"
while IFS= read -r f; do CREATED_LOGS+=("$f"); done < <(ls -t "$EXPDIR"/*_train.log 2>/dev/null | head -2)
expect "F1 FULL endpoint 40000"   "$(grep -qF -- '--max-steps 40000' <<<"$FULL_ARGV" && echo 1)"
expect "F2 FULL cadence 2500"     "$(grep -qF -- '--checkpoint-every 2500' <<<"$FULL_ARGV" && echo 1)"
expect "F3 FULL own save-dir"     "$(grep -qF -- '--save-dir outputs_FLAC/exp17_YAWAUG' <<<"$FULL_ARGV" && ! grep -qF -- 'exp17_YAWAUG_smoke' <<<"$FULL_ARGV" && echo 1)"
expect "F4 SMOKE never 40k"       "$(grep -qF -- '--max-steps 25' <<<"$SMOKE_ARGV" && ! grep -qF -- '40000' <<<"$SMOKE_ARGV" && echo 1)"
expect "F5 SMOKE own save-dir"    "$(grep -qF -- '--save-dir outputs_FLAC/exp17_YAWAUG_smoke' <<<"$SMOKE_ARGV" && echo 1)"
expect "F6 SMOKE cadence>>steps"  "$(grep -qF -- '--checkpoint-every 1000000' <<<"$SMOKE_ARGV" && echo 1)"
expect "F7 rung reaches argv"     "$(grep -qF -- '--batch-size 32 --accum-batches 1' <<<"$FULL_ARGV" && echo 1)"
expect "F8 syncbn + 2 gpus"       "$(grep -qF -- '--num-gpus 2' <<<"$FULL_ARGV" && grep -qF -- '--sync-batchnorm true' <<<"$FULL_ARGV" && echo 1)"
expect "F9 the arm config is used" "$(grep -qF -- "$ARM" <<<"$FULL_ARGV" && echo 1)"

echo "--- G. reviewed-source pins and the FULL prerequisite ---"
SRC="src/data/yaw_rotation.py"
cp "$SRC" "${SRC}.guardbak"
printf '\n# guardtest tamper\n' >> "$SRC"
case_run "G1 source tamper caught" 2 "SOURCE PIN FAILED"  MODE=FULL DRY_RUN=1
mv "${SRC}.guardbak" "$SRC"
case_run "G2 pins pass once restored" 0 "source pins OK"  MODE=FULL DRY_RUN=1
mv "$FAKE_SMOKE" "${FAKE_SMOKE}.hidden"
case_run "G3 FULL without smoke evidence" 2 "FULL requires a SMOKE log"  MODE=FULL DRY_RUN=1
mv "${FAKE_SMOKE}.hidden" "$FAKE_SMOKE"
# Evidence that only *claims* to pass, without the banner, must not count.
printf 'SMOKE VERDICT: PASS\n' > "${FAKE_SMOKE}"
case_run "G4 evidence lacking banner rejected" 2 "FULL requires a SMOKE log"  MODE=FULL DRY_RUN=1
printf '%s\nSMOKE VERDICT: PASS\n' "$BANNER" > "$FAKE_SMOKE"

echo "--- H. the banner check cannot satisfy itself from preflight output ---"
# The whole point of the exact match: run the preflight alone and confirm none
# of ITS lines would pass the post-run whole-line banner test.
PREFLIGHT="$(MODE=SMOKE DRY_RUN=1 bash "$LAUNCH" 2>&1)"
while IFS= read -r f; do CREATED_LOGS+=("$f"); done < <(ls -t "$EXPDIR"/*_train.log 2>/dev/null | head -1)
expect "H1 preflight never emits the banner" \
  "$(tr '\r' '\n' <<<"$PREFLIGHT" | grep -qxF "$BANNER" && echo 0 || echo 1)"
expect "H2 preflight does mention the treatment (non-vacuous)" \
  "$(grep -qF 'preflight treatment plan:' <<<"$PREFLIGHT" && echo 1)"
expect "H3 banner is matched as a whole line in the launcher" \
  "$(grep -qF 'grep -qxF "$BANNER"' "$LAUNCH" && echo 1)"

echo "--- I. R3 and the smoke-checkpoint assertion are wired to real verdicts ---"
expect "I1 R3 threshold is not overridable" \
  "$(grep -qE '^MAX_PROJECTED_HOURS=55' "$LAUNCH" && ! grep -qF 'MAX_PROJECTED_HOURS:-' "$LAUNCH" && echo 1)"
expect "I2 endpoint/cadence are not overridable" \
  "$(! grep -qE 'ENDPOINT_STEPS:-|FULL_CADENCE:-|SMOKE_CADENCE:-' "$LAUNCH" && echo 1)"
expect "I3 R3 FAIL sets a non-zero rc" \
  "$(grep -A1 'SMOKE VERDICT: FAIL - projected' "$LAUNCH" | grep -qF 'rc=4' && echo 1)"
expect "I4 smoke checkpoints are counted and fatal" \
  "$(grep -qF 'SMOKE wrote ${NCKPT} checkpoint(s)' "$LAUNCH" && grep -qF 'rc=5' "$LAUNCH" && echo 1)"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
[ "$RESTORE_FAILED" -eq 0 ] || { echo "!! arm config restoration could not be proven - suite result is NOT trustworthy"; exit 3; }
[ "$FAIL" -eq 0 ]
