#!/usr/bin/env bash
# ============================================================================
# exp38_train_local.sh — exp_38 data_efficiency launcher for THIS 2×A6000 box
# (plan_data_efficiency.md r2 §10 B6). It only sanitises the environment and hands over to
# exp38_supervise.py (exec), which runs every gate, holds the card locks, launches train.py from an
# immutable closure snapshot, owns its process group and publishes the sealed receipts.
#
#   MODE=DRY_RUN   RUN=<id>   CPU-side gates + the exact commands; spawns nothing
#   MODE=SMOKE     RUN=<id>   1 rank on SMOKE_GPU (default 0), micro 32, SMOKE_STEPS (20), no
#                             checkpoint; PRELIMINARY only (no DDP / SyncBN, never a FULL gate)
#   MODE=SMOKE_DDP RUN=<id>   EXPECT_SHA=<HEAD>  production topology (2 × 32 + SyncBN), 30 steps
#                             -> the sealed SMOKE_DDP receipt FULL requires
#   MODE=FULL      RUN=<id>   EXPECT_SHA=<HEAD>  the monolithic 0 -> 40,000 run; needs the E0 receipt
#                             and the newest SMOKE_DDP receipt of this run passing every field check
#   GPU_POLICY=idle (default) | shared ; ATTN_BACKEND=flash (default) | sdpa
# No resume mode: a crashed primary run is archived and re-run from step 0 (plan §6).
# Exit codes: see exp38_supervise.py (0 ok · 2 gate · 3 env · 4 start gate · 5 end audit · 128+N signal).
# Test hooks (DRY_RUN only): EXP38_DRY_SAVE_ROOT, EXP38_LOCKDIR.
# The body is one function called on the last line (bash parses it all before running).
# ============================================================================
set -uo pipefail

main() {
  local HERE REPO
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 3
  REPO="$(git -C "$HERE" rev-parse --show-toplevel)" || exit 3
  cd "$REPO" || exit 3
  local MODE="${MODE:-DRY_RUN}" RUN="${RUN:-}" GPU_POLICY="${GPU_POLICY:-idle}"
  local ATTN_BACKEND="${ATTN_BACKEND:-flash}" SMOKE_GPU="${SMOKE_GPU:-0}" SMOKE_STEPS="${SMOKE_STEPS:-20}"
  die() { echo "EXP38 LAUNCH REFUSED: $1" >&2; exit "${2:-2}"; }

  case "$MODE" in DRY_RUN|SMOKE|SMOKE_DDP|FULL) ;; *) die "MODE must be DRY_RUN|SMOKE|SMOKE_DDP|FULL (got '${MODE}')";; esac
  [[ "$RUN" =~ ^de_(yaw|bf|p1)_f(025|050|075)$ ]] || die "RUN must be a registered 2x32 exp_38 run id (got '${RUN}')"
  case "$GPU_POLICY" in idle|shared) ;; *) die "GPU_POLICY must be idle|shared";; esac
  case "$ATTN_BACKEND" in flash|sdpa) ;; *) die "ATTN_BACKEND must be flash|sdpa";; esac
  [[ "$SMOKE_STEPS" =~ ^[1-9][0-9]?$ ]] || die "SMOKE_STEPS must be 1..99"
  [[ "$SMOKE_GPU" =~ ^[01]$ ]] || die "SMOKE_GPU must be 0 or 1"
  [ -z "${RESUME_STEP:-}" ] || die "there is no resume: archive the run dir and re-run from step 0 (plan §6)"
  local HOOKS=()
  if [ -n "${EXP38_DRY_SAVE_ROOT:-}" ] || [ -n "${EXP38_LOCKDIR:-}" ]; then
    [ "$MODE" = DRY_RUN ] || die "EXP38_DRY_SAVE_ROOT / EXP38_LOCKDIR are DRY_RUN test hooks"
    [ -n "${EXP38_DRY_SAVE_ROOT:-}" ] && HOOKS+=(--dry-save-root "$EXP38_DRY_SAVE_ROOT")
    [ -n "${EXP38_LOCKDIR:-}" ] && HOOKS+=(--lockdir "$EXP38_LOCKDIR")
  fi

  # ---- interpreter + sanitized environment (the supervisor re-checks every item) ----------
  local PY; PY="$(command -v python)" || die "no python on PATH" 3
  case "$(readlink -f "$PY")" in */envs/flac/bin/python*) ;; *) die "python resolves to $(readlink -f "$PY"), not the flac env" 3;; esac
  unset PYTHONPATH PYTHONOPTIMIZE PYTHONSTARTUP PYTHONINSPECT PYTHONHOME ENABLE_TORCH_COMPILE PYTHONDONTWRITEBYTECODE
  export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

  local ARGS=(--platform a6000 --mode "$MODE" --run "$RUN" --backend "$ATTN_BACKEND" --gpu-policy "$GPU_POLICY"
              --smoke-gpu "$SMOKE_GPU" --smoke-steps "$SMOKE_STEPS")
  [ -n "${EXPECT_SHA:-}" ] && ARGS+=(--expect-sha "$EXPECT_SHA")
  exec "$PY" "$HERE/exp38_supervise.py" "${ARGS[@]}" "${HOOKS[@]}"
}

main "$@"
exit $?
