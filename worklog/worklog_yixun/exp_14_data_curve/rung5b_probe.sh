#!/bin/bash
# exp_14 ladder rung 5b: contract-violation termination probe (van arm, 2-GPU DDP) on a split where every receiver keeps ONE source
# => every restricted context pool is empty => DatasetContractError on the first batch.
# Hardened in round F (codex finding 4). Acceptance, ALL of: train.py exits non-zero but not
# by timeout, the log carries DatasetContractError, no train.py of this worktree survives,
# and no checkpoint was written. Only then the .done marker; otherwise .failed + exit 1.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
NAS="${NAS:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
SETTLE="${SETTLE:-5}"; PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"
TS=$(date +%Y-%m-%d_%H-%M-%S); LOG=$REC/rung5b_probe_van_${TS}.log
fail () { printf 'RUNG5B_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung5b_probe_${TS}.failed"
  echo "rung5b FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
echo "=== rung5b start $(date -Is) FLAC=$(git rev-parse --short HEAD)" | tee "$LOG"
RC=0
CUDA_VISIBLE_DEVICES=0,1 timeout "$PROBE_TIMEOUT" python train.py --model-config "$KIT/configs/FLAC_AR_exp14_vanS.json" \
  --dataset-config worklog/worklog_yixun/exp_14_data_curve/rung5b_probe_dataset_config.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 3 --batch-size 2 --accum-batches 1 --num-workers 2 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every 100000 \
  --name smoke5b_probe --experiment-name smoke5b_probe --save-dir "$NAS/smoke5b_probe" >> "$LOG" 2>&1 || RC=$?
echo "=== rung5b rc=$RC end $(date -Is)" | tee -a "$LOG"
sleep "$SETTLE"
# A pid only counts as a survivor when it is a python process whose cwd IS this worktree
# (CLAUDE.md: never attribute a train.py process to this worktree without checking
# /proc/<pid>/cwd). Bare `pgrep -f` matched any shell whose command line merely quoted the
# pattern -- including the one that started the probe -- and reported a phantom survivor.
survivors () {
  local pid exe cwd want
  want="$(readlink -f "$WT" 2>/dev/null || echo "$WT")"
  { pgrep -f "$WT/train.py|smoke5b_probe" || true; } | while read -r pid; do
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    case "$exe" in *python*) : ;; *) continue ;; esac
    if [ "$cwd" = "$want" ]; then echo "$pid"; fi
  done
}
SURV=$(survivors | wc -l)
CKPTS=$( { find "$NAS/smoke5b_probe" -name '*.ckpt' 2>/dev/null || true; } | wc -l)
echo "=== surviving train.py procs from this worktree: $SURV" | tee -a "$LOG"
echo "=== checkpoints written: $CKPTS" | tee -a "$LOG"
[ "$RC" != 0 ] || fail "train.py exited 0: the contract violation did not terminate the run"
[ "$RC" != 124 ] || fail "the probe hit the ${PROBE_TIMEOUT}s timeout instead of failing fast"
grep -q "DatasetContractError" "$LOG" || fail "rc=$RC but no DatasetContractError in $LOG"
[ "$SURV" = 0 ] || fail "$SURV train.py process(es) of this worktree survived the violation"
[ "$CKPTS" = 0 ] || fail "$CKPTS checkpoint(s) were written by a run that must not produce any"
echo "RUNG5B_DONE rc=$RC $(date -Is)" > "$REC/rung5b_probe_${TS}.done"
