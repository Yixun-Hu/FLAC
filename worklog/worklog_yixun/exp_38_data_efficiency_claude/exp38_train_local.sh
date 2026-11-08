#!/usr/bin/env bash
# ============================================================================
# exp38_train_local.sh — exp_38 data_efficiency launcher for THIS 2×A6000 box
# (plan_data_efficiency.md r2 §10 B6). Every recipe fact comes from exp38_registry.py.
#
#   MODE=DRY_RUN   RUN=<id>   CPU-side gates + the exact argv; spawns nothing
#   MODE=SMOKE     RUN=<id>   1 rank on SMOKE_GPU (default 0), micro 32, SMOKE_STEPS (20), no
#                             checkpoint, smoke namespace; PRELIMINARY only (no DDP / SyncBN)
#   MODE=SMOKE_DDP RUN=<id>   EXPECT_SHA=<HEAD>  production topology (2 × 32 + SyncBN), 30 steps,
#                             no checkpoint, smoke namespace -> a bound SMOKE_DDP receipt
#   MODE=FULL      RUN=<id>   EXPECT_SHA=<HEAD>  the monolithic 40k run; needs the E0 receipt and a
#                             PASS SMOKE_DDP receipt for this run at the same campaign closure
#   GPU_POLICY=idle (default) | shared ; ATTN_BACKEND=flash (default) | sdpa
# No resume mode: a crashed primary run is archived and re-run from step 0 (plan §6).
#
# Exit: 0 ok · 2 a gate refused · 3 environment / launch error · 4 the start gate failed (only
# OUR process group is terminated) · 5 the end gate / audit failed · else train.py's status.
# The body is one function called on the last line (bash parses it all before running).
# Never signals a process it did not start; never reads or writes another session's handshake.
# ============================================================================
set -uo pipefail

main() {
  local HERE REPO
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 3
  REPO="$(git -C "$HERE" rev-parse --show-toplevel)" || exit 3
  cd "$REPO" || exit 3
  local EXPREL="worklog/worklog_yixun/exp_38_data_efficiency_claude"

  local MODE="${MODE:-DRY_RUN}" RUN="${RUN:-}" GPU_POLICY="${GPU_POLICY:-idle}"
  local ATTN_BACKEND="${ATTN_BACKEND:-flash}" SMOKE_GPU="${SMOKE_GPU:-0}" SMOKE_STEPS="${SMOKE_STEPS:-20}"
  local TS; TS="$(date '+%Y-%m-%d_%H-%M-%S')"
  local SAVE_ROOT="/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency"
  local CAMPAIGN="/home/yixunhu/codespace/exp-38-campaign"
  local LOCKDIR="${EXP38_LOCKDIR:-/home/yixunhu/codespace}"   # .exp38_card<i>.lock, shared with exp38_eval.py
  # DRY_RUN-only test hook (a real run always uses the registered NAS root)
  if [ "$MODE" = "DRY_RUN" ]; then SAVE_ROOT="${EXP38_DRY_SAVE_ROOT:-$SAVE_ROOT}"; fi
  if [ "$MODE" != "DRY_RUN" ] && [ -n "${EXP38_LOCKDIR:-}" ]; then
    echo "EXP38 LAUNCH REFUSED: EXP38_LOCKDIR is a DRY_RUN test hook" >&2; exit 2
  fi

  die() { echo "EXP38 LAUNCH REFUSED: $1" >&2; exit "${2:-2}"; }

  case "$MODE" in DRY_RUN|SMOKE|SMOKE_DDP|FULL) ;; *) die "MODE must be DRY_RUN|SMOKE|SMOKE_DDP|FULL (got '${MODE}')";; esac
  [[ "$RUN" =~ ^de_(yaw|bf|p1)_f(025|050|075)$ ]] || die "RUN must be a 2x32 exp_38 run id (got '${RUN}')"
  case "$GPU_POLICY" in idle|shared) ;; *) die "GPU_POLICY must be idle|shared";; esac
  [[ "$SMOKE_STEPS" =~ ^[1-9][0-9]?$ ]] || die "SMOKE_STEPS must be 1..99"
  [[ "$SMOKE_GPU" =~ ^[01]$ ]] || die "SMOKE_GPU must be 0 or 1"
  [ -z "${RESUME_STEP:-}" ] || die "there is no resume: archive the run dir and re-run from step 0 (plan §6)"

  # ---- interpreter + sanitized environment ---------------------------------------
  local PY; PY="$(command -v python)" || die "no python on PATH" 3
  case "$(readlink -f "$PY")" in */envs/flac/bin/python*) ;; *) die "python resolves to $(readlink -f "$PY"), not the flac env" 3;; esac
  unset PYTHONPATH PYTHONOPTIMIZE PYTHONSTARTUP PYTHONINSPECT ENABLE_TORCH_COMPILE
  export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
  case "$ATTN_BACKEND" in
    flash) ;;
    sdpa)  export PYTHONPATH="$REPO/overlays/no_flash" PYTHONDONTWRITEBYTECODE=1 ;;
    *) die "ATTN_BACKEND must be flash|sdpa";;
  esac
  local REG=("$PY" "$HERE/exp38_registry.py")
  local ARM="${RUN#de_}"; ARM="${ARM%%_*}"
  local MIN_FREE=12288; [ "$ARM" = "bf" ] && MIN_FREE=21900

  "${REG[@]}" check "$RUN" --platform a6000 --save-root "$SAVE_ROOT" > /dev/null || die "registry check failed for ${RUN}"
  "$PY" "$HERE/exp38_subset_audit.py" --verify "$HERE/data_efficiency_subset_audit.json" > /dev/null \
    || die "subset audit --verify failed (dataset root / bound inputs changed)"
  local PINS; PINS="$("${REG[@]}" pins --backend "$ATTN_BACKEND")" || die "VAE / DINOv3 / backend pins failed"
  echo "pins: ${PINS}"

  gpu_gate() {   # $@ = card indices; strict parsing; refuses, never signals
    local rows apps idx uuid free name n pid
    rows="$(nvidia-smi --query-gpu=index,uuid,memory.free,name --format=csv,noheader,nounits)" || die "nvidia-smi query failed" 3
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)" || die "nvidia-smi apps query failed" 3
    [ "$(grep -c . <<< "$rows")" = 2 ] || die "expected exactly two visible GPUs"
    for idx in "$@"; do
      uuid="$(awk -F', ' -v i="$idx" '$1==i {print $2}' <<< "$rows")"
      free="$(awk -F', ' -v i="$idx" '$1==i {print $3}' <<< "$rows")"
      name="$(awk -F', ' -v i="$idx" '$1==i {print $4}' <<< "$rows")"
      [[ "$uuid" =~ ^GPU- ]] && [[ "$free" =~ ^[0-9]+$ ]] || die "GPU ${idx}: unparseable nvidia-smi row" 3
      [ "$name" = "NVIDIA RTX A6000" ] || die "GPU ${idx} is '${name}', not an NVIDIA RTX A6000"
      n="$(awk -F', ' -v u="$uuid" '$1==u' <<< "$apps" | grep -c . || true)"
      echo "GPU ${idx} ${uuid} (${name}): ${free} MiB free, ${n} foreign compute process(es)"
      for pid in $(awk -F', ' -v u="$uuid" '$1==u {print $2}' <<< "$apps"); do
        echo "  co-tenant pid ${pid} cwd $(readlink "/proc/${pid}/cwd" 2>/dev/null || echo '?')"
      done
      if [ "$GPU_POLICY" = "idle" ] && [ "$n" -gt 0 ]; then die "GPU ${idx} is not idle (GPU_POLICY=idle)"; fi
      [ "$free" -ge "$MIN_FREE" ] || die "GPU ${idx}: ${free} MiB free < ${MIN_FREE} MiB"
    done
  }

  # ---- DRY_RUN -------------------------------------------------------------------
  if [ "$MODE" = "DRY_RUN" ]; then
    "${REG[@]}" run-dir-check "$SAVE_ROOT/$RUN" --run "$RUN" --leg-start 0 || die "run dir is not fresh"
    echo "closure: $("${REG[@]}" closure)"
    echo "FULL argv: CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python $("${REG[@]}" argv "$RUN" --platform a6000 --save-root "$SAVE_ROOT" | tr '\n' ' ')"
    echo "SMOKE_DDP argv: CUDA_VISIBLE_DEVICES=0,1 python $("${REG[@]}" smoke-argv "$RUN" --steps 30 --ddp --save-dir "outputs_FLAC/exp38_smoke/${RUN}_ddp_<ts>" | tr '\n' ' ')"
    echo "SMOKE argv: CUDA_VISIBLE_DEVICES=${SMOKE_GPU} python $("${REG[@]}" smoke-argv "$RUN" --steps "$SMOKE_STEPS" --save-dir "outputs_FLAC/exp38_smoke/${RUN}_<ts>" | tr '\n' ' ')"
    "${REG[@]}" e0-check --platform a6000 2>&1 | sed 's/^/E0 receipt (FULL needs it): /'
    "${REG[@]}" smoke-check "$RUN" --platform a6000 2>&1 | sed 's/^/SMOKE_DDP receipt (FULL needs it): /'
    "${REG[@]}" env-receipt --platform a6000 2>&1 | sed 's/^/environment receipt: /'
    [ "$(readlink -f "$REPO")" = "$CAMPAIGN" ] || echo "note: not the campaign tree ${CAMPAIGN} (SMOKE_DDP / FULL refuse here)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
    echo "DRY_RUN OK ${RUN} (nothing spawned)"
    exit 0
  fi

  # ---- the exp_38 per-card locks (shared with exp38_eval.py; held until this script exits) ----
  lock_card() {   # $1 = card, $2 = fd
    eval "exec $2>> \"\$LOCKDIR/.exp38_card$1.lock\"" || die "cannot open the card-$1 lock" 3
    flock -n "$2" || die "another exp_38 training / evaluation holds card $1 (${LOCKDIR}/.exp38_card$1.lock)"
  }
  if [ "$MODE" = "SMOKE" ]; then lock_card "$SMOKE_GPU" 8; else lock_card 0 8; lock_card 1 9; fi

  # ---- SMOKE (single rank, preliminary) --------------------------------------------------
  if [ "$MODE" = "SMOKE" ]; then
    gpu_gate "$SMOKE_GPU"
    local SDIR="$REPO/outputs_FLAC/exp38_smoke/${RUN}_${TS}" LOG="$HERE/data_efficiency_${TS}_${RUN}_SMOKE.log"
    [ ! -e "$SDIR" ] || die "smoke dir ${SDIR} exists"
    mkdir -p "$SDIR" || die "cannot create ${SDIR}" 3
    local ARGV; mapfile -t ARGV < <("${REG[@]}" smoke-argv "$RUN" --steps "$SMOKE_STEPS" --save-dir "$SDIR")
    [ "${#ARGV[@]}" -gt 10 ] || die "smoke argv empty" 3
    local SAMP="$SDIR/vram_samples.txt"
    nvidia-smi -i "$SMOKE_GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$SAMP" 2>/dev/null &
    local SAMPLER=$!
    echo "=== SMOKE ${RUN} on GPU ${SMOKE_GPU}: python ${ARGV[*]}" | tee "$LOG"
    CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PY" "${ARGV[@]}" < /dev/null >> "$LOG" 2>&1
    local rc=$?
    kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
    local NCK; NCK="$(find "$SDIR" -name '*.ckpt' | wc -l)"
    local GATE; GATE="$("${REG[@]}" log-gate "$LOG" --run "$RUN" --phase end --backend "$ATTN_BACKEND" --ngpu 1 --max-steps "$SMOKE_STEPS" --min-losses 1)"
    local EVG; EVG="$("${REG[@]}" evidence-gate "$SDIR/evidence" --run "$RUN" --ngpu 1 --backend "$ATTN_BACKEND" --max-steps "$SMOKE_STEPS")"
    local PEAK; PEAK="$(sort -n "$SAMP" | tail -1)"
    local VERDICT=PASS
    [ "$rc" -eq 0 ] && [ "$NCK" -eq 0 ] && [ "$GATE" = "LOG GATE PASS" ] && [ "$EVG" = "EVIDENCE GATE PASS" ] || VERDICT=FAIL
    EXP38_REC="$(printf '%s\n' "SMOKE" "$RUN" "$TS" "$SMOKE_GPU" "$SMOKE_STEPS" "$rc" "$NCK" "$GATE" "$EVG" "$PEAK" "$ATTN_BACKEND" "$(git rev-parse HEAD)" "$VERDICT")" \
      "$PY" -c 'import json, os, sys
k = ["mode", "run", "ts", "gpu", "steps", "rc", "checkpoints_written", "log_gate", "evidence_gate", "peak_vram_mib_card", "backend", "head", "verdict"]
rec = dict(zip(k, os.environ["EXP38_REC"].split("\n")))
rec["note"] = "single rank: no DDP / SyncBN; preliminary only"
json.dump(rec, open(sys.argv[1], "w"), indent=1)' "$HERE/data_efficiency_${TS}_${RUN}_SMOKE.json"
    echo "SMOKE VERDICT: ${VERDICT} (log ${LOG})"
    [ "$VERDICT" = PASS ] && exit 0 || exit 5
  fi

  # ---- SMOKE_DDP / FULL: campaign-tree gates ---------------------------------------------
  [ "$(readlink -f "$REPO")" = "$CAMPAIGN" ] || die "${MODE} runs only from the campaign tree ${CAMPAIGN}"
  local HEAD; HEAD="$(git rev-parse HEAD)"
  [ -n "${EXPECT_SHA:-}" ] && [ "$EXPECT_SHA" = "$HEAD" ] || die "EXPECT_SHA (${EXPECT_SHA:-unset}) != HEAD ${HEAD}"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked tree is not clean"
  "${REG[@]}" untracked --platform a6000 || die "untracked entries / import shadows in the campaign tree"
  "${REG[@]}" env-receipt --platform a6000 > /dev/null || die "environment drift vs the campaign receipt"
  local CLOSURE; CLOSURE="$("${REG[@]}" closure)" || die "closure digest failed" 3
  gpu_gate 0 1

  if [ "$MODE" = "SMOKE_DDP" ]; then
    local SDIR="$REPO/outputs_FLAC/exp38_smoke/${RUN}_ddp_${TS}" LOG="$HERE/data_efficiency_${TS}_${RUN}_SMOKE_DDP.log"
    [ ! -e "$SDIR" ] || die "smoke dir ${SDIR} exists"
    mkdir -p "$SDIR" || die "cannot create ${SDIR}" 3
    local ARGV; mapfile -t ARGV < <("${REG[@]}" smoke-argv "$RUN" --steps 30 --ddp --save-dir "$SDIR")
    [ "${#ARGV[@]}" -gt 10 ] || die "smoke argv empty" 3
    echo "=== SMOKE_DDP ${RUN}: CUDA_VISIBLE_DEVICES=0,1 python ${ARGV[*]}" | tee "$LOG"
    CUDA_VISIBLE_DEVICES=0,1 "$PY" "${ARGV[@]}" < /dev/null >> "$LOG" 2>&1
    local rc=$?
    local NCK; NCK="$(find "$SDIR" -name '*.ckpt' | wc -l)"
    local GATE; GATE="$("${REG[@]}" log-gate "$LOG" --run "$RUN" --phase end --backend "$ATTN_BACKEND" --ngpu 2 --max-steps 30 --min-losses 1)"
    local EVG; EVG="$("${REG[@]}" evidence-gate "$SDIR/evidence" --run "$RUN" --ngpu 2 --backend "$ATTN_BACKEND" --max-steps 30)"
    local VERDICT=PASS
    [ "$rc" -eq 0 ] && [ "$NCK" -eq 0 ] && [ "$GATE" = "LOG GATE PASS" ] && [ "$EVG" = "EVIDENCE GATE PASS" ] || VERDICT=FAIL
    local OUT="$HERE/data_efficiency_smoke_${RUN}_${TS}.json"
    "${REG[@]}" smoke-receipt "$OUT" --run "$RUN" --platform a6000 --evidence "$SDIR/evidence" --rc "$rc" \
      --checkpoints "$NCK" --log-gate "$GATE" --evidence-gate "$EVG" --verdict "$VERDICT" --head "$HEAD" \
      || die "could not write the SMOKE_DDP receipt" 3
    echo "SMOKE_DDP VERDICT: ${VERDICT} (receipt ${OUT}, log ${LOG})"
    [ "$VERDICT" = PASS ] && exit 0 || exit 5
  fi

  # ---- FULL ----------------------------------------------------------------------------
  "${REG[@]}" e0-check --platform a6000 > /dev/null || die "no valid E0 receipt for this campaign closure"
  "${REG[@]}" smoke-check "$RUN" --platform a6000 > /dev/null || die "no PASS SMOKE_DDP receipt for ${RUN} at this closure"
  findmnt -n -o FSTYPE -T "$(dirname "$SAVE_ROOT")" | grep -q cifs || die "the save root is not on the NAS (CIFS) mount"
  mkdir -p "$SAVE_ROOT" || die "cannot create ${SAVE_ROOT}" 3
  local AVAIL; AVAIL="$(df -B1G --output=avail "$SAVE_ROOT" | tail -1 | tr -dc '0-9')"
  [ "${AVAIL:-0}" -ge 30 ] || die "NAS free ${AVAIL:-?} GiB < 30 GiB"
  local PROBE="$SAVE_ROOT/.exp38_probe_${TS}_$$"
  echo probe > "$PROBE" && [ "$(cat "$PROBE")" = probe ] && rm -f "$PROBE" || die "NAS write probe failed" 3
  local RUN_DIR="$SAVE_ROOT/$RUN"
  "${REG[@]}" run-dir-check "$RUN_DIR" --run "$RUN" --leg-start 0 > /dev/null || die "run dir is not fresh (archive a failed run first)"
  mkdir -p "$RUN_DIR" || die "cannot create ${RUN_DIR}" 3
  mkdir "$RUN_DIR/.exp38_lock" || die "run lock ${RUN_DIR}/.exp38_lock exists"
  echo "pid $$ host $(hostname) ts ${TS} head ${HEAD}" > "$RUN_DIR/.exp38_lock/holder"
  local SNAP="$RUN_DIR/.exp38_tools"
  "${REG[@]}" snapshot "$RUN" --dest "$SNAP" > /dev/null || die "tool snapshot failed" 3
  local SREG=("$PY" "$SNAP/$EXPREL/exp38_registry.py" --repo "$SNAP")
  local ARGV; mapfile -t ARGV < <("${REG[@]}" argv "$RUN" --platform a6000)
  [ "${#ARGV[@]}" -gt 10 ] || die "argv empty" 3
  local LOG="$HERE/data_efficiency_${TS}_${RUN}_FULL_train.log" REC="$HERE/at_launch_${RUN}_${TS}.txt"
  {
    echo "run: ${RUN}"; echo "mode: FULL (monolithic 0 -> 40000)"; echo "launched_at: $(date -Iseconds)"
    echo "host: $(hostname)"; echo "head: ${HEAD}"; echo "closure: ${CLOSURE}"
    echo "launcher_sha256: $(sha256sum "${BASH_SOURCE[0]}" | cut -c1-64)"
    echo "check: $("${REG[@]}" check "$RUN" --platform a6000 | tr -d '\n ')"
    echo "pins: ${PINS}"; echo "env_receipt: $("${REG[@]}" env-receipt --platform a6000)"
    echo "e0: $("${REG[@]}" e0-check --platform a6000)"; echo "smoke: $("${REG[@]}" smoke-check "$RUN" --platform a6000)"
    echo "gpu_policy: ${GPU_POLICY}"
    echo "gpus: $(nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader | tr '\n' ';')"
    echo "co_tenants: $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr '\n' ';')"
    echo "cpus: $(nproc)"; echo "python: $(readlink -f "$PY")"
    echo "pip_freeze_sha256: $("$PY" -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)"
    echo "snapshot: ${SNAP}"
    echo "argv: CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python ${ARGV[*]}"
    echo "train_log: ${LOG}"
  } > "$REC" || die "cannot write ${REC}" 3
  cp "$REC" "$RUN_DIR/" || die "cannot copy the at-launch record" 3
  cat "$REC"

  CUDA_VISIBLE_DEVICES=0,1 setsid "$PY" "${ARGV[@]}" < /dev/null > "$LOG" 2>&1 &
  local TPID=$!
  sleep 2
  local PGID; PGID="$(ps -o pgid= -p "$TPID" | tr -d ' ')"
  local MYPGID; MYPGID="$(ps -o pgid= -p $$ | tr -d ' ')"
  echo "train.py pid ${TPID} pgid ${PGID:-?}; log ${LOG}"
  kill_ours() {
    if [ -n "$PGID" ] && [ "$PGID" = "$TPID" ] && [ "$PGID" != "$MYPGID" ]; then
      echo "START GATE FAILED: terminating OUR process group ${PGID}" | tee -a "$REC"; kill -TERM -- "-${PGID}" 2>/dev/null
    fi
  }
  local wrc=0
  "${SREG[@]}" log-gate "$LOG" --run "$RUN" --phase start --backend "$ATTN_BACKEND" --ngpu 2 --wait 5400 --pid "$TPID" | tee -a "$REC"
  wrc=${PIPESTATUS[0]}
  if [ "$wrc" -ne 0 ]; then kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
  local w=0
  until [ -f "$RUN_DIR/evidence/run_evidence.json.rank0.json" ] && [ -f "$RUN_DIR/evidence/run_evidence.json.rank1.json" ]; do
    sleep 10; w=$((w + 10))
    if [ "$w" -ge 1800 ] || [ ! -e "/proc/${TPID}" ]; then echo "no run evidence from both ranks" | tee -a "$REC"; kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
  done
  "${SREG[@]}" evidence-gate "$RUN_DIR/evidence" --run "$RUN" --ngpu 2 --backend "$ATTN_BACKEND" | tee -a "$REC"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
  wait "$TPID"; local rc=$?
  echo "train.py exited rc=${rc} watcher rc=${wrc} at $(date -Iseconds)" | tee -a "$REC"
  mv "$RUN_DIR/.exp38_lock" "$RUN_DIR/.exp38_lock.released_${TS}" 2>/dev/null
  [ "$rc" -eq 0 ] || exit "$rc"
  local erc=0
  "${SREG[@]}" verify-snapshot "$SNAP" >> "$REC" || erc=5
  "${SREG[@]}" log-gate "$LOG" --run "$RUN" --phase end --backend "$ATTN_BACKEND" --ngpu 2 >> "$REC" || erc=5
  "${SREG[@]}" run-dir-check "$RUN_DIR" --run "$RUN" --leg-start 40000 >> "$REC" || erc=5
  local FCK; FCK="$(ls "$RUN_DIR"/epoch=*-step=40000.ckpt 2>/dev/null)"
  [ -n "$FCK" ] && "${SREG[@]}" audit-ckpt "$FCK" --run "$RUN" --step 40000 --run-dir "$RUN_DIR" >> "$REC" || erc=5
  [ "$erc" -eq 0 ] && "${SREG[@]}" ckpt-manifest "$RUN_DIR" --run "$RUN" >> "$REC" || erc=5
  cp "$REC" "$RUN_DIR/" 2>/dev/null
  tail -8 "$REC"
  [ "$erc" -eq 0 ] || { echo "EXP38 ${RUN} END AUDIT FAILED (see ${REC})"; exit 5; }
  echo "EXP38 ${RUN} COMPLETE: ${FCK}"
  exit 0
}

main "$@"
exit $?
