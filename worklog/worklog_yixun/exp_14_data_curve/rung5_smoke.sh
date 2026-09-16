#!/bin/bash
# exp_14 ladder rung 5: smallest-batch, NO-checkpoint smoke per arm, 2-GPU DDP+SyncBN (co-tenant with exp13_vanB by Yixun's decision).
# Planner one-off; hardened in round F (codex finding 4: these scripts used to write their
# .done marker whatever happened) and corrected in round G (codex full-r2 finding 3: the
# round-F banner check searched for `max_steps=3 reached`, which matches NEITHER recorded
# successful log -- Lightning writes it with backticks -- so the rung would have failed on
# its own evidence). Acceptance, per arm, all of: train.py exits 0, Lightning's real
# ``\`Trainer.fit\` stopped: \`max_steps=<STEPS>\` reached`` banner, THIS arm's backbone
# banner (the arms differ only there), the <STEPS>/18368 batch count of the 25 % split at
# micro-batch 2 x 2 GPUs, a finite train/loss, and ZERO checkpoints under a save dir this
# invocation made (round H, codex full-r3 finding 2: those last two checks existed in
# ladder_checks.sh but only rung 6 called them, and a FIXED save dir would make the
# checkpoint count a statement about some earlier run). Only then is the .done marker
# written; otherwise a .failed marker names the reason and the script exits
# non-zero. The checks live in ladder_checks.sh, so the recorded logs can be replayed
# through exactly this code. Never touches other processes.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
NAS="${NAS:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
STEPS="${STEPS:-3}"; BATCHES="${BATCHES:-3/18368}"   # the recorded run's own evidence
TS=$(date +%Y-%m-%d_%H-%M-%S)
fail () { printf 'RUNG5_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung5_smoke_${TS}.failed"
  echo "rung5 FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$REC/ladder_checks.sh" || fail "cannot source $REC/ladder_checks.sh"
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
for ARM in cyl van; do
  CFG=$KIT/configs/FLAC_AR_exp14_${ARM}S.json; LOG=$REC/rung5_smoke_${ARM}_${TS}.log
  [ -f "$CFG" ] || fail "$ARM: no arm config at $CFG"
  # Unique per INVOCATION, as rung 7's run dir is: nothing an earlier run left behind can
  # be in here, so "zero checkpoints" is a fact about THIS smoke.
  SD="$NAS/smoke5_${ARM}_${TS}-$$"; mkdir -p "$SD"
  lc_no_checkpoints "$SD" || fail "$ARM: $SD is not a fresh save dir"
  echo "=== rung5 $ARM start $(date -Is) FLAC=$(git rev-parse --short HEAD) cfg=$(sha256sum "$CFG" | cut -c1-16)" | tee "$LOG"
  RC=0
  CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps "$STEPS" --batch-size 2 --accum-batches 1 --num-workers 2 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 100000 \
    --name smoke5_dc_$ARM --experiment-name smoke5_dc_$ARM --save-dir "$SD" >> "$LOG" 2>&1 || RC=$?
  echo "=== rung5 $ARM rc=$RC end $(date -Is)" | tee -a "$LOG"
  [ "$RC" = 0 ] || fail "$ARM: train.py exited $RC (see $LOG)"
  lc_fit_banner "$LOG" "$STEPS" \
    || fail "$ARM: rc 0 but no Lightning max_steps=$STEPS banner in $LOG"
  lc_backbone "$LOG" "$ARM" || fail "$ARM: its own backbone banner is missing from $LOG"
  lc_batches "$LOG" "$BATCHES" || fail "$ARM: the fit never reached $BATCHES in $LOG"
  lc_finite_loss "$LOG" || fail "$ARM: no finite train/loss in $LOG"
  # --checkpoint-every 100000 over $STEPS steps must leave NOTHING behind.
  lc_no_checkpoints "$SD" \
    || fail "$ARM: checkpoint(s) under $SD, written by a smoke that must write none"
done
echo "RUNG5_DONE $(date -Is)" > "$REC/rung5_smoke_${TS}.done"
