#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_guardtests.sh — exercise every gate in yaw_aug_a6000_launch.sh.
#
# Each case drives the REAL launcher and asserts both the exit code and the
# reason printed, so a gate that fires for the wrong reason fails here. Accept
# cases dry-fail at the VRAM gate (MIN_FREE_MB set absurdly high), which sits
# after every cheap gate and before wandb / the DINOv3 pin / train.py — so no
# case can start training, contact W&B, or touch the real output namespaces.
#
# The arm config is mutated in place for the contract cases and restored by
# sha256 comparison; an EXIT trap restores it even on interrupt.
#
# Written by the main session seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 2          # repo root
EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
LAUNCH="${EXPDIR}/yaw_aug_a6000_launch.sh"
ARM="${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
GUARDLOG="${EXPDIR}/yaw_aug_a6000_${TS}_guardtests.log"

ARM_BACKUP="$(mktemp)"; cp "$ARM" "$ARM_BACKUP"
ARM_SHA="$(sha256sum "$ARM" | cut -d' ' -f1)"
CREATED_LOGS=()

restore_arm() {
  cp "$ARM_BACKUP" "$ARM"
  local now; now="$(sha256sum "$ARM" | cut -d' ' -f1)"
  [ "$now" = "$ARM_SHA" ] || echo "!! ARM CONFIG NOT RESTORED (sha ${now} != ${ARM_SHA})"
}
cleanup() {
  restore_arm; rm -f "$ARM_BACKUP"
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
  # collect any launcher log this case created, for cleanup
  while IFS= read -r f; do CREATED_LOGS+=("$f"); done < <(ls -t "$EXPDIR"/*_train.log 2>/dev/null | head -n $(( $(ls "$EXPDIR"/*_train.log 2>/dev/null | wc -l) - before )) )
  if [ "$rc" = "$want_rc" ] && grep -qF -- "$want_txt" <<<"$out"; then
    echo "  PASS  ${name}  (rc=${rc})"; PASS=$((PASS+1))
  else
    echo "  FAIL  ${name}: want rc=${want_rc} and text '${want_txt}'; got rc=${rc}"
    echo "        last line: $(tail -1 <<<"$out")"; FAIL=$((FAIL+1))
  fi
}
# mutate the arm config with a python expression over the parsed object
mutate_arm() { python - "$ARM" "$1" <<'PY'
import json, sys
p, expr = sys.argv[1], sys.argv[2]
cfg = json.load(open(p))
exec(expr, {"cfg": cfg})
json.dump(cfg, open(p, "w"), indent=4)
PY
}

exec > >(tee -a "$GUARDLOG") 2>&1
echo "=== exp_17 launcher guardtests — ${TS} — $(git rev-parse --short HEAD) ==="

echo "--- A. mode selection ---"
case_run "A1 MODE unset"        2 "MODE must be exactly SMOKE or FULL"  MODE=
case_run "A2 MODE lowercase"    2 "MODE must be exactly SMOKE or FULL"  MODE=full
case_run "A3 MODE invented"     2 "MODE must be exactly SMOKE or FULL"  MODE=RESUME
case_run "A4 MODE near-miss"    2 "MODE must be exactly SMOKE or FULL"  MODE=FULL_

echo "--- B. the BN-compliant rung is pinned literally ---"
case_run "B1 accum smuggled"    2 "only the BN-compliant rung"  MODE=FULL MB=8  ACC=8
case_run "B2 micro halved"      2 "only the BN-compliant rung"  MODE=FULL MB=16 ACC=1
case_run "B3 non-numeric"       2 "only the BN-compliant rung"  MODE=FULL MB=32x ACC=1

echo "--- C. config contract (the single-delta claim) ---"
# The width pin can only be exercised from the ViT side: the yaw_aug block is
# pinned by exact equality, so mutating ITS width trips the earlier check and the
# width comparison is never reached (found by this suite's first run).
mutate_arm 'cfg["model"]["conditioning"]["configs"][1]["config"]["ViT"]["img_w"] = 256'
case_run "C1 width mismatch"    2 "rotate by the wrong angle"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
# enabled=1 passes `block == {...}` because 1 == True in Python; only the type
# check can catch it. A string "true" would be caught earlier, by equality.
mutate_arm 'cfg["training"]["yaw_aug"]["enabled"] = 1'
case_run "C2 bool-as-int"       2 "wrong TYPES"                MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["yaw_aug"]["enabled"] = "true"'
case_run "C2b enabled as string" 2 "not the registered treatment"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["yaw_aug"]["img_w"] = 512.0'
case_run "C3 width as float"    2 "wrong TYPES"                MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"].pop("yaw_aug")'
case_run "C4 treatment missing" 2 "not the registered treatment"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["cond_method"] = "fa_invariant"'
case_run "C5 conditioning drift" 2 "must be vanilla-conditioned"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["use_ema"] = False'
case_run "C6 EMA off"           2 "use_ema must be true"        MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["model"]["conditioning"]["configs"][1]["config"]["gradient_checkpointing"] = False'
case_run "C7 grad-ckpt off"     2 "gradient_checkpointing must be true"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["cfg_dropout_prob"] = 0.2'
case_run "C8 non-treatment drift" 2 "NOT the control plus one block"  MODE=FULL MIN_FREE_MB=99999999
restore_arm
mutate_arm 'cfg["training"]["mask_padding_dropout"] = 0'
case_run "C9 type-only drift"   2 "NOT the control plus one block"  MODE=FULL MIN_FREE_MB=99999999
restore_arm

echo "--- D. resource floors ---"
case_run "D1 disk floor"        2 "free disk"     MODE=FULL MIN_FREE_DISK_MB=999999999
case_run "D2 VRAM floor"        2 "< required"    MODE=FULL MIN_FREE_MB=99999999

echo "--- E. accept paths reach the VRAM gate with everything else satisfied ---"
case_run "E1 FULL accepted"     2 "config contract OK"  MODE=FULL  MIN_FREE_MB=99999999
case_run "E2 SMOKE accepted"    2 "config contract OK"  MODE=SMOKE MIN_FREE_MB=99999999

echo "--- F. the smoke/full namespaces are distinct (no cross-contamination) ---"
# NOTE: capture then match. Piping the launcher into grep would make `pipefail`
# report the launcher's non-zero dry-fail status as the pipeline's, so the `if`
# never sees grep's verdict (this suite's second run found exactly that).
case_run "F1 SMOKE own namespace"  2 "save-dir=outputs_FLAC/exp17_YAWAUG_smoke" MODE=SMOKE MIN_FREE_MB=99999999
case_run "F2 FULL own namespace"   2 "save-dir=outputs_FLAC/exp17_YAWAUG"       MODE=FULL  MIN_FREE_MB=99999999
case_run "F3 FULL endpoint pinned" 2 "endpoint=40000 | cadence=2500"            MODE=FULL  MIN_FREE_MB=99999999
case_run "F4 SMOKE never 40k"      2 "endpoint=25"                              MODE=SMOKE MIN_FREE_MB=99999999

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ]
