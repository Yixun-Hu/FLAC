#!/usr/bin/env bash
# p1_eval_driver.sh <K> <SEED> <GPU> — exp_06-addendum P1 (BVp1) eval, exp_01 convention.
# Runs eval_FLAC.py on the IMPORTED (copied+sha-pinned) P1 ckpt/config with the vanilla
# native path. Gates: import-pin sha256 re-verification, frozen MIN_FREE exact-match,
# free-VRAM, refuse-while-P1-training-alive, external log dir. DRY_RUN=1 prints and exits.
set -o pipefail
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
PDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl/p1_matched_d1"
USAGE="usage: p1_eval_driver.sh <K:1|8> <SEED> <GPU>  (env: EVAL_PYTHON DRY_RUN)"
K="${1:-}"; SEED="${2:-}"; GPU="${3:-}"
[ -n "$K" ] && [ -n "$SEED" ] && [ -n "$GPU" ] || { echo "$USAGE"; exit 3; }
case "$K" in 1|8) ;; *) echo "REFUSING: K must be 1 or 8"; exit 3;; esac
CKPT="$PDIR/p1_import/p1_step67500.ckpt"
CFG="$PDIR/p1_import/FLAC_AR_BVp1.json"
PINS="$PDIR/p1_import/p1_import_pins.txt"
LOGDIR="/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude/d_eval_logs"
EVAL_PYTHON="${EVAL_PYTHON:-/home/yixunhu/miniconda3/envs/flac/bin/python}"
[ -f "$CKPT" ] && [ -f "$CFG" ] && [ -f "$PINS" ] || { echo "REFUSING: import (ckpt/config/pins) incomplete — run the import step first"; exit 3; }
if [ "$K" = "1" ]; then DS="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"; else DS="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"; fi
NAME="exp09_P1D1_K${K}_s${SEED}"
ARGS=(--model-config "$CFG" --dataset-config "$DS" --ckpt-path "$CKPT"
      --cond-method vanilla --seed "$SEED" --steps 1 --cfg-scale 1.0 --eval-name "$NAME")
CMD_DISPLAY="CUDA_VISIBLE_DEVICES=${GPU} ${EVAL_PYTHON} eval_FLAC.py ${ARGS[*]}"
if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN (not executed):"; echo "EVAL: ${CMD_DISPLAY}"; exit 0; fi
# --- REVIEWED-CODE gate (protocol r1 blocking #2): exact full worktree HEAD must equal the
# r2-CLEARED pin AND tracked files must be clean (untracked p1_import artifacts are allowed;
# any tracked modification = unreviewed code = refuse). Applies to every REAL invocation. ---
: "${EXPECT_P1KIT_SHA:?EXPECT_P1KIT_SHA required (the r2-CLEARED worktree commit)}"
HEAD_FULL="$(git rev-parse HEAD)"
[ "$HEAD_FULL" = "$EXPECT_P1KIT_SHA" ] || { echo "REVIEWED-CODE GATE FAILED: HEAD ${HEAD_FULL} != EXPECT_P1KIT_SHA ${EXPECT_P1KIT_SHA}"; exit 1; }
[ -z "$(git status --porcelain -uno)" ] || { echo "REVIEWED-CODE GATE FAILED: tracked files modified (unreviewed code)"; git status --porcelain -uno | head; exit 1; }
# --- refuse while the FLAC-owned P1 training is alive (never co-tenant their run) ---
if pgrep -f "exp07_P1" > /dev/null 2>&1; then echo "REFUSING: P1 training process still alive"; exit 2; fi
# --- import-pin gate (replaces the exp-09 pin gate for this non-exp-09 config): the pins
# manifest must contain EXACTLY the two expected entries (protocol r1 blocking #2), then
# every entry must verify (--strict: any bad/improper line fails). ---
[ "$(grep -c . "$PINS")" -eq 2 ] || { echo "IMPORT PIN GATE FAILED: pins manifest must have exactly 2 entries"; exit 1; }
grep -qE '^[0-9a-f]{64}  p1_step67500\.ckpt$' "$PINS" || { echo "IMPORT PIN GATE FAILED: no exact ckpt entry"; exit 1; }
grep -qE '^[0-9a-f]{64}  FLAC_AR_BVp1\.json$' "$PINS" || { echo "IMPORT PIN GATE FAILED: no exact config entry"; exit 1; }
(cd "$PDIR/p1_import" && sha256sum -c "$(basename "$PINS")" --quiet --strict) || { echo "IMPORT PIN GATE FAILED — copies do not match p1_import_pins.txt"; exit 1; }
# --- frozen MIN_FREE exact-match + free-VRAM gate (same discipline as d_eval_driver) ---
FROZEN_FILE="worklog/worklog_yixun/exp_09_cyl_no_ssl/c1_frozen_min_free.txt"
[ -f "$FROZEN_FILE" ] || { echo "REFUSING: frozen threshold file absent"; exit 3; }
FROZEN="$(tr -d '[:space:]' < "$FROZEN_FILE")"
case "$FROZEN" in ''|*[!0-9]*) echo "REFUSING: frozen value not an integer"; exit 3;; esac
MIN_FREE_MB="${MIN_FREE_MB:-$FROZEN}"
[ "$MIN_FREE_MB" = "$FROZEN" ] || { echo "REFUSING: MIN_FREE_MB override mismatch"; exit 3; }
mkdir -p "$LOGDIR" || exit 3
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${LOGDIR}/exp09_${TS}_P1D1_${NAME}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_06-addendum P1 eval — ${TS} — worktree ${HEAD_FULL} (reviewed-pin OK, tracked-clean) — ${NAME} gpu=${GPU} MIN_FREE_MB=${MIN_FREE_MB} ==="
sha256sum "$CKPT" "$CFG"
FREE="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
[ -n "$FREE" ] || { echo "nvidia-smi query failed — refusing blind"; exit 2; }
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${GPU} free ${FREE} < ${MIN_FREE_MB} — refusing"; exit 2; }
echo "--- launching: ${CMD_DISPLAY} ---"
CUDA_VISIBLE_DEVICES="$GPU" "$EVAL_PYTHON" eval_FLAC.py "${ARGS[@]}"
rc=${PIPESTATUS[0]}
echo "=== P1 eval '${NAME}' exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') — log ${LOG} ==="
exit "$rc"
