#!/usr/bin/env bash
# exp_22 R2-C: the Planner-run pre-flight launcher (plan_loc_meshgrid_r2_bf_yaw.md §9 A7 r2 (v)(2), r3 (b) (f) (h)),
# after exp_35's reviewed exp35_phase_i_launch.sh (capture -> seal -> launch).
#
#   EXPECT_SCRIPT_SHA256=<reviewed sha> bash exp22_r2c_preflight.sh capture <PIN2> <GPU0|GPU1>
#       The hard checks (HEAD = PIN2 on a clean tracked tree, the four PIN2 pins, the assets, the run's fresh
#       directories, no exp22 run active, the card free under its rule), the launcher's OWN 31 sealed polls of the
#       card 60 s apart (an O_EXCL capture record), the checks again; ONLY when every check passed, the seal
#       (noclobber) binding PIN2, this script's sha256, the run, the card + index, the capture record and log, the
#       tools (the line list among them) and the pins. Prints the launch line. Writes only into
#       outputs_loc/exp22/r2_preflight_<RUN>/.
#   EXPECT_SCRIPT_SHA256=<reviewed sha> bash exp22_r2c_preflight.sh launch <PIN2> <seal> <seal sha256>
#       Within 10 min of the last sealed poll: the seal and all it binds verified, every check again, the
#       single-instance exp22 lock taken, the card re-checked, then (exp22_r2c_supervisor.py reserve) the O_EXCL
#       reservation, the re-check after it and the first durable state "reserved"; the supervisor DETACHED (setsid +
#       nohup, an env -i environment, cwd = the worktree) with the lock fd inherited; its identity confirmed.
#
# The layout of the PIN2 worktree W (checked by every capture and launch; A7-C3):
#   outputs_loc   -> /home/yixunhu/codespace/FLAC/outputs_loc    one link (untracked at PIN2)
#   AcousticRooms -> /media/diskstation/yixunhu/FLAC/AcousticRooms   one link (untracked at PIN2)
#   weights/      the worktree's OWN directory -- PIN2 tracks files under weights/FLAC/, so it can never be one link
#                 (git would read them as deleted) -- holding one link per UNTRACKED entry of the main checkout's
#                 weights/: weights/AGREE, weights/exp20 and weights/AGREE_*.pt -> main; weights/FLAC/{FLAC.ckpt,
#                 FLAC_EMA.ckpt, VAE.ckpt, VAE.safetensors, ...} -> main. Each registered asset under weights/ must BE
#                 the main checkout's file (test -ef: same device + inode once every link is resolved).
#
# The Planner runs it by hand; nothing here acts unattended. It never signals, kills or edits any process; nvidia-smi
# queries only. Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
set -uo pipefail
# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
PYTEST_MAIN=/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytest/__main__.py
OUTPUTS_TARGET=/home/yixunhu/codespace/FLAC/outputs_loc
WEIGHTS_TARGET=/home/yixunhu/codespace/FLAC/weights
ROOMS_TARGET=/media/diskstation/yixunhu/FLAC/AcousticRooms
CONFIRM_TIMEOUT_S=60
DINO_CACHE=/home/yixunhu/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m
# --- END REGISTERED CONFIG ---
MODE="${1:-}"; PIN2="${2:-}"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; W="$(cd "$TOOLS/../../../.." && pwd)"
COMMON="$TOOLS/exp22_r2c_common.py"; EVID="$TOOLS/exp22_r2c_evidence.py"; SUP="$TOOLS/exp22_r2c_supervisor.py"
EX="$W/outputs_loc/exp22"; LOCK="$EX/r2_preflight.lock"
GPU0=GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e; GPU1=GPU-6ac2604a-e84e-0615-1221-ccbed7c268da
PINS=("overlays/no_flash/flash_attn.py|0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"
      "overlays/no_flash/exp35_no_flash_launch.py|bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de"
      "src/tests/test_exp35_no_flash_launch.py|d61a3c02a7e1295a48b3edb60e8abccab508cadb6f4c11bdca9872234bd961dd"
      "pyproject.toml|6422eae5482773db252372760825749f41ac061487be892e350fda997b619a74")
TOOL_FILES=(exp22_r2c_common.py exp22_r2c_evidence.py exp22_r2c_supervisor.py exp22_free_card_detect.sh
            exp22_r2c_preflight.sh exp22_r2c_preflight_lines.json)
ASSETS=(outputs_loc/exp22/d1_context_manifest.json outputs_loc/exp22/g1_audit/geometry_audit_report.json)
# every weights/ file a line opens (A7-C3, statically): the gates' / probe's --ckpt-path (exp20), the model config's
# AGREE checkpoint and the AGREE dinoV3 audio tower's pretrained VAE (every AGREE scorer build), T's released FLAC_EMA
WEIGHT_ASSETS=(weights/exp20/YAW_40k.ckpt weights/exp20/BF_40k.ckpt weights/exp20/P1_40k.ckpt
               weights/FLAC/FLAC_EMA.ckpt weights/FLAC/VAE.ckpt weights/AGREE/AGREE_fullAR.pt)
ASSET_DIRS=(outputs_loc/exp22/i1_P1_CRN_br256_20260825_194053_merged AcousticRooms/metadata AcousticRooms/depth_map
            AcousticRooms/single_channel_ir_1)                     # the last three: T's real-data tests skip without them
LAUNCH_ENV=(HOME=/home/yixunhu USER=yixunhu LOGNAME=yixunhu LANG=en_US.UTF-8
            PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONDONTWRITEBYTECODE=1)
RUN_RE='^r2_preflight_[0-9]{8}T[0-9]{6}$'
FAIL=0

say() { printf '%s\n' "$*"; }
need() {  # <what> <command...>: a hard check
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then say "CHECK OK: $what"; else say "CHECK FAILED: $what"; FAIL=1; fi
}
sha() { sha256sum < "$1" 2>/dev/null | cut -c1-64; }
eq() { [ "$1" = "$2" ]; }
py() { env -i "${LAUNCH_ENV[@]}" CUDA_VISIBLE_DEVICES= "$PY" -B "$@"; }   # every helper in the launch environment
uuid_of() { case "$1" in GPU0) say "$GPU0" ;; GPU1) say "$GPU1" ;; *) say "" ;; esac; }
field() { awk -v k="$2" '{for (i = 1; i < NF; i++) if ($i == k) {print $(i + 1); exit}}' <<<"$1"; }
ok_capture() { [[ "$1" == "CAPTURE OK "* ]]; }
script_check() { need "EXPECT_SCRIPT_SHA256 is set and equals this script's sha256 (${SELF_SHA})" eq "${EXPECT_SCRIPT_SHA256:-unset}" "$SELF_SHA"; }

common_checks() {   # <orch> <run>: the preconditions both modes check right before they act
  local head e rel want out rc
  head="$(git -C "$W" rev-parse HEAD 2>/dev/null)"
  need "HEAD ${head} = PIN2 ${PIN2}" eq "$head" "$PIN2"
  need "the tracked tree is clean" eq "$(git -C "$W" status --porcelain --untracked-files=no 2>&1)" ""
  for e in "${PINS[@]}"; do IFS='|' read -r rel want <<<"$e"; need "pin ${rel} ${want:0:8}..." eq "$(sha "$W/$rel")" "$want"; done
  need "outputs_loc -> ${OUTPUTS_TARGET}" eq "$(readlink -f "$W/outputs_loc")" "$OUTPUTS_TARGET"
  need "weights/ is this worktree's own directory, not a link (PIN2 tracks files under it)" test -d "$W/weights" -a ! -L "$W/weights"
  for rel in "${WEIGHT_ASSETS[@]}"; do
    need "asset ${rel} is the main checkout's file (${WEIGHTS_TARGET}/${rel#weights/}: same device + inode)" \
      test -f "$W/$rel" -a "$W/$rel" -ef "$WEIGHTS_TARGET/${rel#weights/}"
  done
  need "AcousticRooms -> ${ROOMS_TARGET}" eq "$(readlink -f "$W/AcousticRooms")" "$ROOMS_TARGET"
  for rel in "${ASSETS[@]}"; do need "asset ${rel}" test -f "$W/$rel"; done
  for rel in "${ASSET_DIRS[@]}"; do need "asset ${rel}/" test -d "$W/$rel"; done
  need "the flac interpreter and pytest's own __main__.py" test -x "$PY" -a -f "$PYTEST_MAIN"
  need "the DINOv3 HF cache T's tests need (${DINO_CACHE})" test -d "$DINO_CACHE"
  need "the run's r2_gates_${2} and r2_probe_BF_${2} do not exist" test ! -e "$EX/r2_gates_$2" -a ! -e "$EX/r2_probe_BF_$2"
  out="$(py "$SUP" active --except-orch "$1")"; rc=$?; say "$out"
  need "no exp22 run is active or unresolved" test "$rc" = 0
}

card_now() {   # <card> <sealed index or empty>: one observation under the card rule
  local out rc
  out="$(py "$COMMON" observe --card "$1")"; rc=$?; say "observation: $out"
  need "$1 is free under its rule now" test "$rc" = 0
  [ -z "$2" ] || need "$1 is at the sealed index $2" eq "$(jq -r .index <<<"$out" 2>/dev/null)" "$2"
}

capture() {   # <card> <run> <orch> <draft>
  local card="$1" run="$2" orch="$3" draft="$4" rec="$3/capture_polls.jsonl" chk rsha t e rel want
  say "=== exp_22 R2-C pre-flight CAPTURE -- $(date -Is) -- host $(hostname) -- PIN2 ${PIN2} -- ${card} -- run ${run} ==="
  say "this script: ${SELF_SHA}  $0"
  script_check
  common_checks "$orch" "$run"
  card_now "$card" ""
  [ "$FAIL" = 0 ] || { say "CAPTURE REFUSED before the polls: at least one CHECK failed (see above)"; return 1; }
  say ""; say "### the sealed admission polls of ${card} (A7 r3 (b)) -> ${rec}"
  py "$EVID" capture-polls --card "$card" --record "$rec"
  need "every sealed poll on time and free" test "$?" = 0
  rsha="$(sha "$rec")"
  chk="$(py "$EVID" check-capture --card "$card" --record "$rec" --sha256 "$rsha" --fresh)"; say "$chk"
  need "the capture record verifies (sha256 ${rsha})" ok_capture "$chk"
  say ""; say "### the checks again, after the window"
  common_checks "$orch" "$run"
  card_now "$card" "$(field "$chk" index)"
  [ "$FAIL" = 0 ] || { say "CAPTURE REFUSED: at least one CHECK failed (see above); no seal"; return 1; }
  { printf 'pin2 %s\nscript_sha256 %s\nrun %s\ncard %s %s %s\n' "$PIN2" "$SELF_SHA" "$run" "$card" "$(uuid_of "$card")" \
      "$(field "$chk" index)"
    printf 'capture_record %s %s\nfirst_poll %s\nlast_poll %s\n' "$rec" "$rsha" "$(field "$chk" first)" "$(field "$chk" last)"
    for t in "${TOOL_FILES[@]}"; do printf 'tool %s %s\n' "$t" "$(sha "$TOOLS/$t")"; done
    for e in "${PINS[@]}"; do IFS='|' read -r rel want <<<"$e"; printf 'pin %s %s\n' "$rel" "$want"; done; } > "$draft"
  say "CAPTURE CHECKS OK (the seal is written after this log is closed)"
}

seal_write() {   # <draft> <capture log> <orch>: the seal, a NEW file, only after a fully passing capture
  local seal="$3/seal.txt"
  ( set -o noclobber; { cat "$1"; printf 'capture_log %s %s\nsealed_at %s %s\n' "$2" "$(sha "$2")" "$(date +%s)" \
      "$(date -Is)"; } > "$seal" ) 2>/dev/null || { say "SEAL REFUSED: ${seal} exists"; return 1; }
  say "SEAL ${seal} sha256 $(sha "$seal")"
  say "LAUNCH LINE (within 10 min of the last poll): EXPECT_SCRIPT_SHA256=${SELF_SHA} bash $0 launch ${PIN2} ${seal} $(sha "$seal")"
}

launch() {   # <seal> <seal sha256>
  local seal="$1" ssha="$2" orch run card cuuid idx rec rsha clog clsha chk out rc pid suplog extra t want
  orch="$(dirname "$seal")"; run="${orch##*_}"
  say "=== exp_22 R2-C pre-flight LAUNCH -- $(date -Is) -- host $(hostname) -- PIN2 ${PIN2} -- seal ${seal} ${ssha} ==="
  say "this script: ${SELF_SHA}  $0"
  script_check
  need "the seal is the regular file seal.txt of this run's orchestration directory" \
    bash -c '[ -f "$1" ] && [ ! -L "$1" ] && [ "${1##*/}" = seal.txt ]' _ "$seal"
  need "the seal hashes to the given sha256" eq "$(sha "$seal")" "$ssha"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED: the seal"; return 1; }
  cat "$seal"
  sv() { awk -v k="$1" '$1 == k {$1 = ""; sub(/^ /, ""); print; exit}' "$seal"; }
  need "the seal names PIN2" eq "$(sv pin2)" "$PIN2"
  need "the seal names this script" eq "$(sv script_sha256)" "$SELF_SHA"
  need "the seal names this run" eq "$(sv run)" "$run"
  read -r card cuuid idx <<<"$(sv card)"
  need "the sealed card ${card} is registered as ${cuuid}" bash -c '[ -n "$1" ] && [ "$1" = "$2" ]' _ "$(uuid_of "$card")" "$cuuid"
  read -r rec rsha <<<"$(sv capture_record)"; read -r clog clsha <<<"$(sv capture_log)"
  need "the capture record is this run's" eq "$rec" "$orch/capture_polls.jsonl"
  need "the capture log is this run's, unchanged, with CAPTURE CHECKS OK and no CHECK FAILED" bash -c \
    '[ "$1" = "$2/capture.log" ] && [ -f "$1" ] && [ ! -L "$1" ] && [ "$(sha256sum < "$1" | cut -c1-64)" = "$3" ] && grep -q "^CAPTURE CHECKS OK" "$1" && ! grep -q "^CHECK FAILED" "$1"' \
    _ "$clog" "$orch" "$clsha"
  need "the seal binds all ${#TOOL_FILES[@]} tools" eq "$(grep -c '^tool ' "$seal")" "${#TOOL_FILES[@]}"
  while read -r _ t want; do need "tool ${t} unchanged since the capture" eq "$(sha "$TOOLS/$t")" "$want"; done < <(grep '^tool ' "$seal")
  extra="$(ls -A "$orch" | grep -v -x -E 'capture\.log|capture_polls\.jsonl|seal\.txt|launch_[0-9]{8}T[0-9]{6}\.log')"
  need "the orchestration directory holds only the capture and launch logs (no state yet)" eq "$extra" ""
  chk="$(py "$EVID" check-capture --card "$card" --record "$rec" --sha256 "$rsha" --fresh)"; say "$chk"
  need "the sealed capture verifies and is fresh (<= 10 min after its last poll)" ok_capture "$chk"
  need "the capture's index is the sealed index" eq "$(field "$chk" index)" "$idx"
  common_checks "$orch" "$run"
  card_now "$card" "$idx"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED: at least one CHECK failed (see above)"; return 1; }
  exec {LFD}<>"$LOCK" || { say "LAUNCH REFUSED: the exp22 lock ${LOCK} cannot be opened"; return 1; }
  flock -x -n "$LFD" || { say "LAUNCH REFUSED: another exp22 launcher or supervisor holds ${LOCK}"; return 1; }
  say "the single-instance exp22 lock ${LOCK} is held (fd ${LFD})"
  chk="$(py "$EVID" check-capture --card "$card" --record "$rec" --sha256 "$rsha" --fresh)"; say "$chk"
  need "last look: the sealed capture is still fresh" ok_capture "$chk"
  card_now "$card" "$idx"
  out="$(py "$SUP" active --except-orch "$orch" --holding-lock)"; rc=$?; say "$out"
  need "last look: no other exp22 run is active or unresolved" test "$rc" = 0
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED at the last look (nothing reserved)"; return 1; }
  out="$(py "$SUP" reserve --orch "$orch" --pin2 "$PIN2" --card "$card" --index "$idx" --seal "$seal" --seal-sha256 "$ssha")"
  rc=$?; say "$out"
  [ "$rc" = 0 ] || { say "LAUNCH REFUSED: the reservation (see above)"; return 1; }
  suplog="$orch/supervisor.log"
  ( set -o noclobber; : > "$suplog" ) 2>/dev/null || { say "LAUNCH NOT STARTED: ${suplog} exists (the reservation and the state stay for the Planner)"; return 1; }
  say "starting: cd ${W} && env -i ${LAUNCH_ENV[*]} /usr/bin/setsid /usr/bin/nohup ${PY} -B ${SUP} run --orch ${orch} --lock-fd ${LFD} >> ${suplog} 2>&1 < /dev/null &"
  ( cd "$W" && exec env -i "${LAUNCH_ENV[@]}" /usr/bin/setsid /usr/bin/nohup "$PY" -B "$SUP" run --orch "$orch" \
      --lock-fd "$LFD" >> "$suplog" 2>&1 < /dev/null ) &
  pid=$!
  out="$(py "$SUP" confirm --orch "$orch" --pid "$pid" --timeout "$CONFIRM_TIMEOUT_S")"; rc=$?; say "$out"
  if [ "$rc" = 0 ]; then
    say "LAUNCHED: supervisor pid ${pid}; log ${suplog}; state ${orch}/state.json; heartbeat ${orch}/heartbeat.json"; return 0
  fi
  say "LAUNCH NOT CONFIRMED: inspect ${suplog} and ${orch}/state.json (never signalled here; the reservation and the state stay for the Planner)"
  return 1
}

SELF_SHA="$(sha "$0")"
case "$MODE" in
  capture) [ "$#" -eq 3 ] && [[ "${3:-}" =~ ^GPU[01]$ ]] || { say "usage: EXPECT_SCRIPT_SHA256=<sha> bash $0 capture <PIN2> <GPU0|GPU1>"; exit 2; } ;;
  launch) [ "$#" -eq 4 ] || { say "usage: EXPECT_SCRIPT_SHA256=<sha> bash $0 launch <PIN2> <seal> <seal sha256>"; exit 2; } ;;
  *) say "usage: EXPECT_SCRIPT_SHA256=<sha> bash $0 capture <PIN2> <GPU0|GPU1> | launch <PIN2> <seal> <seal sha256>"; exit 2 ;;
esac
[[ "$PIN2" =~ ^[0-9a-f]{40}$ ]] || { say "PIN2 must be a 40-hex commit"; exit 2; }
if [ "$MODE" = capture ]; then
  RUN="$(date +%Y%m%dT%H%M%S)"; ORCH="$EX/r2_preflight_$RUN"
  mkdir "$ORCH" 2>/dev/null || { say "CAPTURE REFUSED: ${ORCH} exists or ${EX} is missing"; exit 1; }
  OUT="$ORCH/capture.log"; DRAFT="$(mktemp)"
  ( set -o noclobber; : > "$OUT" ) || exit 1
  capture "$3" "$RUN" "$ORCH" "$DRAFT" 2>&1 | tee -a "$OUT"; rc=${PIPESTATUS[0]}
  say "rc ${rc}; capture log ${OUT} sha256 $(sha "$OUT")"
  if [ "$rc" = 0 ] && ! grep -q '^CHECK FAILED' "$OUT"; then seal_write "$DRAFT" "$OUT" "$ORCH" || rc=1; fi
  rm -f -- "$DRAFT"; exit "$rc"
fi
ORCH="$(dirname "$3")"
[[ "${ORCH##*/}" =~ $RUN_RE ]] && [ "$(dirname "$ORCH")" = "$EX" ] && [ -d "$ORCH" ] && [ ! -L "$ORCH" ] \
  || { say "LAUNCH REFUSED: ${3} is not inside an orchestration directory ${EX}/r2_preflight_<RUN>"; exit 1; }
OUT="$ORCH/launch_$(date +%Y%m%dT%H%M%S).log"
( set -o noclobber; : > "$OUT" ) 2>/dev/null || { say "LAUNCH REFUSED: ${OUT} exists"; exit 1; }
launch "$3" "$4" 2>&1 | tee -a "$OUT"; rc=${PIPESTATUS[0]}
say "rc ${rc}; launch log ${OUT} sha256 $(sha "$OUT")"
exit "$rc"
