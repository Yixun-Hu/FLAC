#!/bin/bash
# ============================================================================
# exp38_train.sbatch — ONE monolithic exp_38 job on neuronic (plan_data_efficiency.md r2 §10 B7;
# sensitivity platform, D2). Submit ONLY through exp38_submit_neuronic.py, which derives the Slurm
# resources, names the job with its intent token and exports:
#   RUN  MODE (SMOKE | FULL)  EXPECT_SHA  EXP38_TOKEN  ATTN_BACKEND (flash|sdpa)
#   EXPECT_FLASH_VERSION (flash only)  ALLOW_NONMATCHED_RUNG (1 for an _r8x8 run)  SAVE_DIR
# SMOKE = the 2-rank (or 8-rank) 30-step storage-light pilot (rate, memory, evidence) that FULL
# requires; it also writes this platform's environment receipt the first time.
# Test hook (refused under a real allocation): EXP38_SBATCH_DRYRUN=1 runs every gate that needs no
# allocation and prints the torchrun line.
# Exit: 0 ok · 2 a gate refused · 3 environment error · 4 start gate failed (our torchrun group
# terminated) · 5 end gate / audit failed · else torchrun's status.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -uo pipefail

main() {
  local REPO; REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "not inside the FLAC checkout" >&2; exit 3; }
  cd "$REPO" || exit 3
  local EXP="worklog/worklog_yixun/exp_38_data_efficiency_claude"
  local DRY="${EXP38_SBATCH_DRYRUN:-0}"
  die() { echo "EXP38 JOB REFUSED: $1" >&2; exit "${2:-2}"; }
  if [ "$DRY" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then die "EXP38_SBATCH_DRYRUN is a test hook, never under a real allocation"; fi
  if [ "$DRY" != 1 ] && [ -z "${SLURM_JOB_ID:-}" ]; then die "a real job must run under sbatch"; fi

  local RUN="${RUN:-}" MODE="${MODE:-}" EXPECT_SHA="${EXPECT_SHA:-}" TOKEN="${EXP38_TOKEN:-}"
  local ATTN_BACKEND="${ATTN_BACKEND:-flash}" ALLOW="${ALLOW_NONMATCHED_RUNG:-0}" SD="${SAVE_DIR:-}"
  [[ "$RUN" =~ ^de_(yaw|bf|p1)_f(025|050|075)(_r8x8)?$ ]] || die "RUN '${RUN}' is not an exp_38 run id"
  case "$MODE" in SMOKE|FULL) ;; *) die "MODE must be SMOKE|FULL";; esac
  [[ "$TOKEN" =~ ^[0-9a-f]{32}$ ]] || die "EXP38_TOKEN must be the submission's 32-hex intent token"
  local NGPU=2; [[ "$RUN" == *_r8x8 ]] && NGPU=8
  local ALLOW_FLAG=(); [ "$ALLOW" = 1 ] && ALLOW_FLAG=(--allow-nonmatched-rung)

  # ---- environment (exp_15 conventions) ----------------------------------------------
  local ENV_PREFIX="/n/fs/gatrdp/envs/flac"
  [ "$DRY" = 1 ] && ENV_PREFIX="${EXP38_ENV_PREFIX:-$ENV_PREFIX}"
  unset PYTHONPATH PYTHONOPTIMIZE PYTHONSTARTUP PYTHONINSPECT ENABLE_TORCH_COMPILE
  export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
  [ "$DRY" = 1 ] || export HF_HOME=/n/fs/gatrdp/hf_cache
  case "$ATTN_BACKEND" in
    flash) ;;
    sdpa) export PYTHONPATH="$REPO/overlays/no_flash" PYTHONDONTWRITEBYTECODE=1 ;;
    *) die "ATTN_BACKEND must be flash|sdpa" ;;
  esac
  local PY; PY="$(command -v python)" || die "no python" 3
  [ "$(readlink -f "$PY")" = "$(readlink -f "$ENV_PREFIX/bin/python")" ] || die "python is $(readlink -f "$PY"), not ${ENV_PREFIX}/bin/python" 3
  local REG=("$PY" "$REPO/$EXP/exp38_registry.py")

  # ---- code / closure / recipe gates --------------------------------------------------------
  local HEAD; HEAD="$(git rev-parse HEAD)"
  [ -n "$EXPECT_SHA" ] && [ "$HEAD" = "$EXPECT_SHA" ] || die "HEAD ${HEAD} != EXPECT_SHA ${EXPECT_SHA:-unset}"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die "tracked tree is not clean"
  "${REG[@]}" untracked --platform neuronic || die "import shadows / unexpected untracked entries"
  "${REG[@]}" check "$RUN" --platform neuronic "${ALLOW_FLAG[@]}" > /dev/null || die "registry check failed"
  "$PY" "$REPO/$EXP/exp38_subset_audit.py" --verify "$REPO/$EXP/data_efficiency_subset_audit.json" > /dev/null \
    || die "subset audit --verify failed on this platform's AcousticRooms tree"
  local PINS; PINS="$("${REG[@]}" pins --backend "$ATTN_BACKEND")" || die "VAE / DINOv3 / backend pins failed"
  if [ "$ATTN_BACKEND" = flash ]; then
    local FV; FV="$("$PY" -c 'import importlib.metadata as m; print(m.version("flash-attn"))' 2>/dev/null)"
    [ -n "${EXPECT_FLASH_VERSION:-}" ] && [ "$FV" = "$EXPECT_FLASH_VERSION" ] || die "flash-attn ${FV:-absent} != EXPECT_FLASH_VERSION ${EXPECT_FLASH_VERSION:-unset}"
  fi
  local RG=0
  "${REG[@]}" e0-check --platform neuronic > /dev/null || { RG=1; echo "receipt gate: no E0 receipt for this campaign closure (E0 runs on the A6000 box)" >&2; }
  if [ "$MODE" = FULL ]; then
    "${REG[@]}" env-receipt --platform neuronic > /dev/null || { RG=1; echo "receipt gate: environment drift vs this platform's receipt (or no SMOKE yet)" >&2; }
    "${REG[@]}" smoke-check "$RUN" --platform neuronic > /dev/null || { RG=1; echo "receipt gate: no PASS SMOKE receipt for ${RUN} on neuronic at this closure" >&2; }
  fi
  if [ "$RG" -ne 0 ] && [ "$DRY" != 1 ]; then die "a receipt gate refused (see above)"; fi

  # ---- allocation gates ---------------------------------------------------------------------
  if [ "$DRY" != 1 ]; then
    local GPUS; GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || die "nvidia-smi failed" 3
    [ "$(grep -c . <<< "$GPUS")" = "$NGPU" ] || die "allocation has $(grep -c . <<< "$GPUS") GPUs, rung needs ${NGPU}"
    [ "$(grep -c 'L40' <<< "$GPUS")" = "$NGPU" ] || die "allocation GPUs are not all L40: ${GPUS}"
    [ "${SLURM_JOB_NUM_NODES:-1}" = 1 ] && [ "${SLURM_NTASKS:-1}" = 1 ] || die "need 1 node / 1 task"
    [[ "${SLURM_JOB_NAME:-}" == *"${TOKEN:0:8}" ]] || die "job name ${SLURM_JOB_NAME:-?} does not carry this intent token"
  fi
  local ARGV
  if [ "$MODE" = SMOKE ]; then
    mapfile -t ARGV < <("${REG[@]}" smoke-argv "$RUN" --steps 30 --ddp --platform neuronic --save-dir "$SD")
  else
    mapfile -t ARGV < <("${REG[@]}" argv "$RUN" --platform neuronic "${ALLOW_FLAG[@]}")
    [ "$(awk '$0=="--save-dir"{getline; print; exit}' < <(printf '%s\n' "${ARGV[@]}"))" = "$SD" ] || die "SAVE_DIR ${SD} != the registry's run dir"
  fi
  [ "${#ARGV[@]}" -gt 10 ] || die "registry refused the argv"
  "${REG[@]}" run-dir-check "$SD" --run "$RUN" --leg-start 0 --allow-submission-marker > /dev/null || die "run dir ${SD} is not fresh"
  local TS; TS="$(date '+%Y-%m-%d_%H-%M-%S')"
  local TAG="${RUN}_${MODE}_jid${SLURM_JOB_ID:-dry}"
  local LOG="$REPO/$EXP/data_efficiency_${TS}_${TAG}_train.log" REC="$REPO/$EXP/at_launch_${TAG}_${TS}.txt"
  local CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "${ARGV[@]}")
  if [ "$DRY" = 1 ]; then
    echo "pins: ${PINS}"; echo "closure: $("${REG[@]}" closure)"
    echo "DRYRUN ${MODE} ${RUN} ngpu ${NGPU}: ${CMD[*]}"
    [ "$RG" -eq 0 ] && echo "DRYRUN receipt gates: PASS" || echo "DRYRUN receipt gates: WOULD REFUSE (E0 / SMOKE receipts are produced at campaign time)"
    exit 0
  fi
  mkdir -p "$SD" && mkdir "$SD/.exp38_lock" || die "run lock ${SD}/.exp38_lock exists (live or crashed run)"
  echo "job ${SLURM_JOB_ID} host $(hostname) token ${TOKEN}" > "$SD/.exp38_lock/holder"
  {
    echo "run: ${RUN}"; echo "mode: ${MODE}"; echo "job: ${SLURM_JOB_ID} ${SLURM_JOB_NAME} host $(hostname) token ${TOKEN}"
    echo "launched_at: $(date -Iseconds)"; echo "head: ${HEAD}"; echo "closure: $("${REG[@]}" closure)"
    echo "sbatch_sha256: $(sha256sum "$REPO/$EXP/exp38_train.sbatch" | cut -c1-64)"
    echo "check: $("${REG[@]}" check "$RUN" --platform neuronic "${ALLOW_FLAG[@]}" | tr -d '\n ')"
    echo "pins: ${PINS}"; echo "gpus: $(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | tr '\n' ';')"
    echo "slurm_rank_env: PROCID=${SLURM_PROCID:-} LOCALID=${SLURM_LOCALID:-} NODEID=${SLURM_NODEID:-} NTASKS=${SLURM_NTASKS:-}"
    echo "cpus: $(nproc)"; echo "python: $(readlink -f "$PY")"
    echo "pip_freeze_sha256: $("$PY" -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)"
    echo "command: ${CMD[*]}"; echo "train_log: ${LOG}"
  } > "$REC" || die "cannot write ${REC}" 3
  cp "$REC" "$SD/" || die "cannot copy the record" 3
  if [ "$MODE" = SMOKE ] && [ ! -e "$REPO/$EXP/data_efficiency_env_receipt_neuronic.json" ]; then
    "${REG[@]}" env-receipt --platform neuronic --write >> "$REC" || die "cannot write the neuronic environment receipt" 3
  fi
  "${REG[@]}" env-receipt --platform neuronic >> "$REC" || die "environment drift vs this platform's receipt"

  # PL 2.1 elects TorchElastic; the SLURM rank variables must not reach train.py (recorded above)
  unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS_PER_NODE SLURM_NPROCS
  setsid "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
  local TPID=$!
  sleep 2
  local PGID; PGID="$(ps -o pgid= -p "$TPID" | tr -d ' ')"
  kill_ours() { [ -n "$PGID" ] && [ "$PGID" = "$TPID" ] && kill -TERM -- "-${PGID}" 2>/dev/null; }
  local STEPS=40000; [ "$MODE" = SMOKE ] && STEPS=30
  if [ "$MODE" = FULL ]; then
    "${REG[@]}" log-gate "$LOG" --run "$RUN" --phase start --backend "$ATTN_BACKEND" --ngpu "$NGPU" --wait 5400 --pid "$TPID" | tee -a "$REC"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
    local w=0 r
    for r in $(seq 0 $((NGPU - 1))); do
      until [ -f "$SD/evidence/run_evidence.json.rank${r}.json" ]; do
        sleep 10; w=$((w + 10))
        if [ "$w" -ge 1800 ] || [ ! -e "/proc/${TPID}" ]; then echo "no run evidence from rank ${r}" | tee -a "$REC"; kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
      done
    done
    "${REG[@]}" evidence-gate "$SD/evidence" --run "$RUN" --ngpu "$NGPU" --backend "$ATTN_BACKEND" | tee -a "$REC"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then kill_ours; wait "$TPID" 2>/dev/null; exit 4; fi
  fi
  wait "$TPID"; local rc=$?
  echo "torchrun rc=${rc} at $(date -Iseconds)" | tee -a "$REC"
  mv "$SD/.exp38_lock" "$SD/.exp38_lock.released_${TS}" 2>/dev/null
  if [ "$MODE" = SMOKE ]; then
    local NCK; NCK="$(find "$SD" -name '*.ckpt' | wc -l)"
    local GATE; GATE="$("${REG[@]}" log-gate "$LOG" --run "$RUN" --phase end --backend "$ATTN_BACKEND" --ngpu "$NGPU" --max-steps 30 --min-losses 1)"
    local EVG; EVG="$("${REG[@]}" evidence-gate "$SD/evidence" --run "$RUN" --ngpu "$NGPU" --backend "$ATTN_BACKEND" --max-steps 30)"
    local VERDICT=PASS
    [ "$rc" -eq 0 ] && [ "$NCK" -eq 0 ] && [ "$GATE" = "LOG GATE PASS" ] && [ "$EVG" = "EVIDENCE GATE PASS" ] || VERDICT=FAIL
    "${REG[@]}" smoke-receipt "$REPO/$EXP/data_efficiency_smoke_${RUN}_${TS}.json" --run "$RUN" --platform neuronic \
      --evidence "$SD/evidence" --rc "$rc" --checkpoints "$NCK" --log-gate "$GATE" --evidence-gate "$EVG" \
      --verdict "$VERDICT" --head "$HEAD" | tee -a "$REC"
    echo "SMOKE VERDICT ${VERDICT}; rate: $(tr '\r' '\n' < "$LOG" | grep -oE '[0-9.]+(it/s|s/it)' | tail -1)" | tee -a "$REC"
    [ "$VERDICT" = PASS ] && exit 0 || exit 5
  fi
  [ "$rc" -eq 0 ] || exit "$rc"
  local erc=0
  "${REG[@]}" log-gate "$LOG" --run "$RUN" --phase end --backend "$ATTN_BACKEND" --ngpu "$NGPU" >> "$REC" || erc=5
  "${REG[@]}" run-dir-check "$SD" --run "$RUN" --leg-start 40000 --allow-submission-marker >> "$REC" || erc=5
  local ECK; ECK="$(ls "$SD"/epoch=*-step=40000.ckpt 2>/dev/null)"
  [ -n "$ECK" ] && "${REG[@]}" audit-ckpt "$ECK" --run "$RUN" --step 40000 --run-dir "$SD" >> "$REC" || erc=5
  [ "$erc" -eq 0 ] && "${REG[@]}" ckpt-manifest "$SD" --run "$RUN" >> "$REC" || erc=5
  cp "$REC" "$SD/" 2>/dev/null
  [ "$erc" -eq 0 ] || { echo "EXP38 ${RUN} END AUDIT FAILED"; exit 5; }
  echo "EXP38 ${RUN} COMPLETE (${ECK})"
  exit 0
}

main "$@"
exit $?
