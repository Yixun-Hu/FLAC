#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_roteval_run.sh — exp_17 C4 rotation-evaluation grid.
#
# Yixun, 2026-08-15: evaluate every finished checkpoint at 0/90/180/270 degrees
# at K=1 and K=8 — 16 x 4 x 2 = 128 cells — and run it ONLY AFTER training
# finishes, so the 40k run keeps both A6000s to itself. That choice is enforced
# here, not remembered: gate 1 refuses to start while training is alive, and
# gate 2 refuses until the completion audit passes.
#
# The checkpoints belong to ANOTHER worktree's run (exp-17-yawaug-a6000), which
# this script must never disturb. eval_FLAC.py writes its metric JSON to
# os.path.dirname(--ckpt-path), so pointing it at the foreign directory would
# scatter 128 files through another session's namespace. Instead we build a
# SYMLINK FARM in our own output tree and evaluate through it: the reads pass
# through to their checkpoints, every write lands in ours.
#
# Both GPUs run one cell each. Resumable: a cell with a non-empty metric JSON is
# skipped, so an interrupted 12-hour queue picks up where it stopped.
#
# Written by the main session seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
FOREIGN_ROOT="${FOREIGN_ROOT:-$HOME/codespace/exp-17-yawaug-a6000}"
FOREIGN_CKPT_DIR="${FOREIGN_ROOT}/outputs_FLAC/exp17_YAWAUG"
FOREIGN_LOG="${FOREIGN_ROOT}/${EXPDIR}/yaw_aug_a6000_2026-08-15_14-54-31_FULL.log"
FOREIGN_MODEL_CFG="${FOREIGN_ROOT}/${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
TRAIN_PID="${TRAIN_PID:-1012326}"

FARM="outputs_FLAC/exp17_YAWAUG_roteval"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_roteval.log"

# Single instance. Two starts would schedule duplicate cells, put two processes
# on each card, and race writes to the same JSONs (Codex review).
LOCK="${EXPDIR}/.roteval.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "another roteval run holds ${LOCK} - abort"; exit 2; }

mkdir -p "$FARM"
[ -L "$FARM" ] && { echo "${FARM} is a symlink; refusing (writes could land elsewhere)"; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 C4 rotation-eval grid — ${TS} — $(git rev-parse --short HEAD) ==="
echo "foreign run : ${FOREIGN_CKPT_DIR}"
echo "symlink farm: ${FARM}   (all writes land here; the foreign tree is read-only)"

# --- gate 1: training must be FINISHED, not merely quiet --------------------- #
# The user's explicit choice was zero interference. A liveness check is the only
# thing standing between that promise and a 50 % slowdown of a 20-hour run.
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "REFUSING TO START: training pid ${TRAIN_PID} is still alive."
  echo "  Yixun chose 'run after training, zero interference' (2026-08-15)."
  echo "  Wait for it to finish, or re-run with TRAIN_PID= if that pid is stale."
  exit 2
fi
echo "gate 1 OK: training pid ${TRAIN_PID} is not running"

# --- gate 2: the run must have COMPLETED, not just stopped ------------------- #
# Evaluating an interrupted run's checkpoints would produce 128 real-looking
# numbers for a trajectory that never reached its endpoint.
# The foreign launcher records its own exit status: "=== exp_17 FULL exit rc=N".
# Hard-coding --rc 0 would defeat the audit's "an unknown status is not a zero
# one" rule (Codex review), so parse the real one and refuse if it is absent.
FRC="$(tr '\r' '\n' < "$FOREIGN_LOG" | sed -n 's/.*exit rc=\([0-9][0-9]*\).*/\1/p' | tail -1)"
[ -n "$FRC" ] || { echo "REFUSING TO START: the foreign log records no exit status yet"; exit 2; }
echo "foreign training exit status: rc=${FRC}"
python -m src.tools.exp17_full_audit --log "$FOREIGN_LOG" \
       --ckpt-dir "$FOREIGN_CKPT_DIR" --rc "$FRC" || {
  echo "REFUSING TO START: the FULL run did not pass its completion audit (above)."
  echo "  Its checkpoints are not a completed 40k trajectory; evaluating them would"
  echo "  produce 128 plausible numbers for an experiment that did not finish."
  exit 2; }
echo "gate 2 OK: completion audit passed"

# --- build the symlink farm -------------------------------------------------- #
# Clear stale links first: an old link could silently point the grid at a
# different run's checkpoint for the same step (Codex review).
find "$FARM" -maxdepth 1 -name '*.ckpt' -delete
N=0
while IFS= read -r real; do
  if ln -sfn "$real" "${FARM}/$(basename "$real")"; then N=$((N+1));
  else echo "failed to link ${real} - abort"; exit 2; fi
done < <(find "$FOREIGN_CKPT_DIR" -name '*.ckpt' | sort)
echo "symlinked ${N} checkpoints into ${FARM}"
[ "$N" -eq 16 ] || { echo "expected 16 checkpoints, linked ${N} - abort"; exit 2; }

echo "--- provenance: the real checkpoints behind the links ---"
for l in "${FARM}"/*.ckpt; do
  printf '%s -> %s\n' "$(basename "$l")" "$(readlink -f "$l")"
done

# --- plan, then run two cells at a time (one per GPU) ------------------------ #
python - "$FARM" "$FOREIGN_MODEL_CFG" <<'PY' > "${EXPDIR}/.roteval_queue.txt"
import sys, re
from pathlib import Path
sys.path.insert(0, ".")
from src.tools.exp17_rotation_grid import build_grid, cell_argv, cell_name, pending_cells

farm, model_cfg = Path(sys.argv[1]), sys.argv[2]
ckpts = {int(re.search(r"step=(\d+)", p.name).group(1)): str(p)
         for p in farm.glob("*.ckpt")}
grid = build_grid(ckpts)
todo = pending_cells(grid, out_dir=farm)
print(f"# {len(todo)} of {len(grid)} cells pending", file=sys.stderr)
for c in todo:
    print("\t".join(cell_argv(c, model_config=model_cfg)))
PY
rc=$?
[ "$rc" -eq 0 ] || { echo "grid planning FAILED (rc=${rc}) - abort"; exit 2; }

TOTAL="$(wc -l < "${EXPDIR}/.roteval_queue.txt")"
echo "queue: ${TOTAL} pending cells (~11 min each; 2 in flight => ~$((TOTAL*11/2/60)) h)"

# A no-argument `wait` returns 0 regardless of child failures (GNU bash manual),
# so rev 1 discarded every evaluator error. Wait on each PID and count failures.
i=0; FAILED=0; PIDS=(); LABELS=()
drain() {
  local j
  for j in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$j]}"; then
      echo "  !! cell FAILED: ${LABELS[$j]} (see ${EXPDIR}/roteval_${LABELS[$j]}.log)"
      FAILED=$((FAILED+1))
    fi
  done
  PIDS=(); LABELS=()
}
while IFS= read -r line; do
  IFS=$'\t' read -r -a ARGV <<<"$line"
  GPU=$((i % 2))
  NAME="${ARGV[-1]}"
  CELLLOG="${EXPDIR}/roteval_${NAME}.log"
  echo "[$((i+1))/${TOTAL}] gpu${GPU} ${NAME}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}" > "$CELLLOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("$NAME")
  i=$((i+1))
  [ $((i % 2)) -eq 0 ] && drain
done < "${EXPDIR}/.roteval_queue.txt"
drain
echo "evaluator failures: ${FAILED}"

echo "=== grid finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
LEFT="$(python -c "
import sys, re; sys.path.insert(0,'.')
from pathlib import Path
from src.tools.exp17_rotation_grid import build_grid, pending_cells
farm=Path('${FARM}')
ck={int(re.search(r'step=(\d+)', p.name).group(1)): str(p) for p in farm.glob('*.ckpt')}
print(len(pending_cells(build_grid(ck), out_dir=farm)))
")"
echo "cells still pending: ${LEFT}"
[ "$LEFT" -eq 0 ] || { echo "INCOMPLETE GRID - re-run this script to resume"; exit 1; }
echo "all 128 cells complete"
