#!/bin/bash
# ============================================================================
# exp38_train.sbatch — ONE monolithic exp_38 job on neuronic (plan_data_efficiency.md r2 §10 B7;
# sensitivity platform, D2). Submit ONLY through exp38_submit_neuronic.py, which derives the Slurm
# resources, owns (run, mode) with a stable intent record, names the job with its intent token and
# exports:
#   RUN  MODE (SMOKE | FULL)  EXPECT_SHA  EXP38_TOKEN  ATTN_BACKEND (flash|sdpa)
#   EXPECT_FLASH_VERSION (flash only)  ALLOW_NONMATCHED_RUNG (1 for an _r8x8 run)  SAVE_DIR
# SMOKE = the production-topology 30-step pilot that FULL requires (sealed SMOKE_DDP receipt); the
# first SMOKE also writes this platform's sealed environment receipt.
# This script only sanitises the environment and execs exp38_supervise.py, which exact-compares the
# allocation and the environment with the owner intent, launches torchrun FROM a closure snapshot
# in the run dir, owns its process group (TERM/INT/HUP forwarded, bounded KILL) and publishes the
# receipts. Test hook (refused under a real allocation): EXP38_SBATCH_DRYRUN=1 -> MODE=DRY_RUN.
# Exit codes: see exp38_supervise.py.
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
  [[ "$RUN" =~ ^de_(yaw|bf)_f(025|050|075)(_r8x8)?$|^de_p1_f025$ ]] || die "RUN '${RUN}' is not a registered exp_38 run id"
  case "$MODE" in SMOKE|FULL) ;; *) die "MODE must be SMOKE|FULL";; esac
  [[ "$TOKEN" =~ ^[0-9a-f]{32}$ ]] || die "EXP38_TOKEN must be the submission's 32-hex intent token"
  case "$ATTN_BACKEND" in flash|sdpa) ;; *) die "ATTN_BACKEND must be flash|sdpa" ;; esac
  case "$ALLOW" in 0|1) ;; *) die "ALLOW_NONMATCHED_RUNG must be 0|1" ;; esac

  # ---- environment (exp_15 conventions) ----------------------------------------------
  local ENV_PREFIX="/n/fs/gatrdp/envs/flac"
  [ "$DRY" = 1 ] && ENV_PREFIX="${EXP38_ENV_PREFIX:-$ENV_PREFIX}"
  unset PYTHONPATH PYTHONOPTIMIZE PYTHONSTARTUP PYTHONINSPECT PYTHONHOME ENABLE_TORCH_COMPILE PYTHONDONTWRITEBYTECODE
  export PATH="$ENV_PREFIX/bin:$PATH" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
  [ "$DRY" = 1 ] || export HF_HOME=/n/fs/gatrdp/hf_cache
  local PY; PY="$(command -v python)" || die "no python" 3
  [ "$(readlink -f "$PY")" = "$(readlink -f "$ENV_PREFIX/bin/python")" ] || die "python is $(readlink -f "$PY"), not ${ENV_PREFIX}/bin/python" 3

  # no untracked importable file may sit next to the kit the supervisor imports (Codex code r2 HIGH 10)
  local KITLIST; KITLIST="$(git -C "$REPO/$EXP" ls-files --others -- .)" || die "git ls-files failed in the kit folder" 3
  local SHADOW; SHADOW="$(grep -v '__pycache__/' <<< "$KITLIST" | grep -E '\.(py|pyc|pyo|so|pyd|pth)$|(^|/)(site|user)customize' || true)"
  [ -z "$SHADOW" ] || die "untracked importable files in the kit folder: ${SHADOW//$'\n'/ }" 3
  local ARGS=(--platform neuronic --run "$RUN" --backend "$ATTN_BACKEND" --expect-python "$ENV_PREFIX/bin/python"
              --expect-flash-version "${EXPECT_FLASH_VERSION:-}")
  [ "$ALLOW" = 1 ] && ARGS+=(--allow-nonmatched-rung)
  if [ "$DRY" = 1 ]; then
    exec "$PY" "$REPO/$EXP/exp38_supervise.py" --mode DRY_RUN "${ARGS[@]}"
  fi
  exec "$PY" "$REPO/$EXP/exp38_supervise.py" --mode "$MODE" "${ARGS[@]}" --expect-sha "$EXPECT_SHA" \
    --token "$TOKEN" --save-dir "$SD"
}

main "$@"
exit $?
