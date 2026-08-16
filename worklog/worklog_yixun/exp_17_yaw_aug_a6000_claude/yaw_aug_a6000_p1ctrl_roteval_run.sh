#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_p1ctrl_roteval_run.sh — exp_17 P1-CONTROL C4 rotation grid.
#
# The same 16-checkpoint x {0,90,180,270} x {K1,K8} grid as the Yaw-Aug arm,
# run over P1 vanilla's OWN checkpoints (this repo, exp_07) under the IDENTICAL
# protocol. Without this, "augmentation bought rotational flatness" has no
# same-protocol baseline: exp_07's A6 head-to-head is supporting evidence but a
# different protocol vintage. The comparison of interest is spread-vs-spread,
# per (step, K), same evaluator, same seed, same bf16 autocast.
#
# DO NOT LAUNCH without Yixun's explicit GO (decision (b) of 2026-08-16): it
# owns both GPUs for ~7 h. Gate 1 additionally refuses while any eval_FLAC or
# the Yaw-Aug grid runner is alive, so the two grids can never overlap.
#
# P1's checkpoint directory is a HISTORICAL ARCHIVE that already holds the
# original P1 metric JSONs — and eval_FLAC.py writes its outputs next to the
# checkpoint it reads. So this grid, too, evaluates through a symlink farm:
# reads pass through, all 128 new JSONs land in outputs_FLAC/exp17_P1CTRL_roteval.
#
# Written by the main session seat (Claude Fable 5).
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
P1_CKPT_DIR="outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints"
P1_MODEL_CFG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
# P1's training config, pinned since exp_07 (also the exp_17 control config).
PIN_p1cfg="733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8  ${P1_MODEL_CFG}"

FARM="outputs_FLAC/exp17_P1CTRL_roteval"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_p1ctrl_roteval.log"

# THE SAME lock file the Yaw-Aug runner holds (review: a private lock plus
# pgrep snapshots is one-sided and racy — the running YAWAUG process already
# owns .roteval.lock, so acquiring it here is mutual exclusion with NO change
# to the active script, and a YAWAUG resume cannot start under us either).
LOCK="${EXPDIR}/.roteval.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "the rotation-grid lock ${LOCK} is held (Yaw-Aug grid or another control) - abort"; exit 2; }

mkdir -p "$FARM"
[ -L "$FARM" ] && { echo "${FARM} is a symlink; refusing"; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 P1-CONTROL C4 grid — ${TS} — $(git rev-parse --short HEAD) ==="

# --- gate 1: the GPUs must be free of the other grid and of training --------- #
if pgrep -f "yaw_aug_a6000_roteval_run.sh" >/dev/null 2>&1; then
  echo "REFUSING: the Yaw-Aug rotation grid is still running - the two grids must never overlap"; exit 2
fi
if pgrep -f "eval_FLAC.py" >/dev/null 2>&1; then
  echo "REFUSING: eval_FLAC processes are already on the GPUs"; exit 2
fi
if pgrep -f "train.py" >/dev/null 2>&1; then
  echo "REFUSING: a train.py is running somewhere on this box - check readlink /proc/<pid>/cwd before anything else"; exit 2
fi
echo "gate 1 OK: no grid, no eval, no training on the box"

# --- gate 2: the config is P1's own, byte-pinned ----------------------------- #
echo "$PIN_p1cfg" | sha256sum -c --status - || {
  echo "P1 config pin FAILED - ${P1_MODEL_CFG} is not the file P1 was trained with - abort"; exit 2; }
echo "gate 2 OK: P1 config pin matches"

# --- symlink farm over P1's 2500..40000 window ------------------------------- #
find "$FARM" -maxdepth 1 -name '*.ckpt' -delete
N=0
for S in $(seq 2500 2500 40000); do
  MATCHES=( "${P1_CKPT_DIR}"/epoch=*-step=${S}.ckpt )
  [ -e "${MATCHES[0]}" ] || { echo "P1 checkpoint for step ${S} not found - abort"; exit 2; }
  [ "${#MATCHES[@]}" -eq 1 ] || { echo "step ${S} has ${#MATCHES[@]} checkpoint files (${MATCHES[*]}) - refusing to pick one silently"; exit 2; }
  REAL="${MATCHES[0]}"
  ln -sfn "$(readlink -f "$REAL")" "${FARM}/$(basename "$REAL")" || { echo "link failed for ${REAL}"; exit 2; }
  N=$((N+1))
done
echo "symlinked ${N} P1 checkpoints into ${FARM}"
[ "$N" -eq 16 ] || { echo "expected 16, linked ${N} - abort"; exit 2; }
for l in "${FARM}"/*.ckpt; do printf '%s -> %s\n' "$(basename "$l")" "$(readlink -f "$l")"; done

# --- plan (arm=P1CTRL), then run two at a time ------------------------------- #
python - "$FARM" "$P1_MODEL_CFG" <<'PY' > "${EXPDIR}/.p1ctrl_queue.txt"
import sys, re
from pathlib import Path
sys.path.insert(0, ".")
from src.tools.exp17_rotation_grid import (
    build_grid, cell_argv, pending_cells, P1_CONTROL_ARM)

farm, model_cfg = Path(sys.argv[1]), sys.argv[2]
ckpts = {int(re.search(r"step=(\d+)", p.name).group(1)): str(p)
         for p in farm.glob("*.ckpt")}
grid = build_grid(ckpts)
todo = pending_cells(grid, out_dir=farm, arm=P1_CONTROL_ARM)
print(f"# {len(todo)} of {len(grid)} control cells pending", file=sys.stderr)
for c in todo:
    print("\t".join(cell_argv(c, model_config=model_cfg, arm=P1_CONTROL_ARM)))
PY
rc=$?
[ "$rc" -eq 0 ] || { echo "control-grid planning FAILED (rc=${rc}) - abort"; exit 2; }

TOTAL="$(wc -l < "${EXPDIR}/.p1ctrl_queue.txt")"
echo "queue: ${TOTAL} pending control cells (~6.5 min each; 2 in flight => ~$((TOTAL*65/10/2/60)) h)"

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
  echo "[$((i+1))/${TOTAL}] gpu${GPU} ${NAME}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}" > "${EXPDIR}/roteval_${NAME}.log" 2>&1 &
  PIDS+=("$!"); LABELS+=("$NAME")
  i=$((i+1))
  [ $((i % 2)) -eq 0 ] && drain
done < "${EXPDIR}/.p1ctrl_queue.txt"
drain
echo "evaluator failures: ${FAILED}"

LEFT="$(python -c "
import sys, re; sys.path.insert(0,'.')
from pathlib import Path
from src.tools.exp17_rotation_grid import build_grid, pending_cells, P1_CONTROL_ARM
farm=Path('${FARM}')
ck={int(re.search(r'step=(\d+)', p.name).group(1)): str(p) for p in farm.glob('*.ckpt')}
print(len(pending_cells(build_grid(ck), out_dir=farm, arm=P1_CONTROL_ARM)))
")"
echo "control cells still pending: ${LEFT}"
[ "$LEFT" -eq 0 ] || { echo "INCOMPLETE CONTROL GRID - re-run to resume"; exit 1; }
echo "all 128 control cells complete"
