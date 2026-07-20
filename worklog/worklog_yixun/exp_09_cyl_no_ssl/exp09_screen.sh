#!/usr/bin/env bash
# ============================================================================
# exp09_screen.sh <step> <REQUIRED_FREE_MB> - screen one exp09_cylNoSSL checkpoint
# (EMA, then the online variant IF its config exists), K=8, eval-seed 42, full unseen
# split, bf16 - the exp-09 analog of exp_07's bf_screen.sh (same reviewed skeleton:
# direct redirect, no pipeline, exit nonzero if any eval failed so a background
# launcher surfaces it). Co-located on GPU 1 with training.
#
# !!! C2 CO-TENANT-SCREENING PROHIBITION (integrative-review finding 4) !!!
# The C2 records PROHIBIT running this screen co-tenant with C2 training UNLESS C1 has
# EXPLICITLY proved the combined train+eval peak fits. An eval launched into a low-memory
# training phase can consume headroom the next training peak needs and TERMINATE C2. This
# script therefore takes a REQUIRED free-memory headroom argument ($2) and refuses (via an
# nvidia-smi check on the eval GPU) BEFORE launching any eval when the headroom is not free.
#
# BOTH passes pass --cond-method fa_invariant --frame-avg-angles 0 EXPLICITLY:
# eval_FLAC.py defaults to 'vanilla' and does NOT read cond_method from the JSON
# training block (unlike train.py, where the JSON is authoritative), so omitting the
# flags would screen the documented raw-pose vanilla trap (Codex Stage-B blocker 1).
#
# The EMA pass uses FLAC_AR_exp09.json (use_ema=true -> eval_FLAC loads EMA
# weights). The online pass uses FLAC_AR_exp09_online_eval.json; that variant is a
# D-stage artifact (grad-checkpointing off, use_ema=false) and is NOT part of the
# Stage-B deliverable, so the online pass is skipped cleanly until the file exists.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

S="${1:?usage: exp09_screen.sh <step> <REQUIRED_FREE_MB>}"
EXPD="worklog/worklog_yixun/exp_09_cyl_no_ssl"
EVAL_GPU=1   # this screen co-locates the eval on GPU 1 (CUDA_VISIBLE_DEVICES=1 below)

# --- REQUIRED free-memory headroom ($2): refuse (fail-closed) BEFORE launching any eval if
# the eval GPU lacks the headroom. See the C2 co-tenant-screening prohibition in the header.
REQUIRED_FREE_MB="${2:-}"
[ -n "$REQUIRED_FREE_MB" ] || { echo "usage: exp09_screen.sh <step> <REQUIRED_FREE_MB>  (headroom argument is REQUIRED; C2 records prohibit co-tenant screening without proven headroom)"; exit 3; }
case "$REQUIRED_FREE_MB" in
  ''|*[!0-9]*) echo "REFUSING: REQUIRED_FREE_MB='${REQUIRED_FREE_MB}' is not a plain integer."; exit 3;;
esac
FREE="$(nvidia-smi -i "$EVAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
rc_q=$?
[ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on eval GPU ${EVAL_GPU} (rc=${rc_q}) - refusing to screen blind"; exit 2; }
[ "$FREE" -ge "$REQUIRED_FREE_MB" ] || { echo "eval GPU ${EVAL_GPU} free ${FREE} MiB < required ${REQUIRED_FREE_MB} MiB - refusing to screen (co-tenant headroom inadequate; C2 prohibition)"; exit 2; }

# recursive: with --logger wandb, train.py:129 nests ckpts under
# <save-dir>/<project>/<run-name>/checkpoints/ (flat with --logger none)
CKPT="$(find outputs_FLAC/exp09_cylNoSSL -name "*step=${S}.ckpt" 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "no ckpt for step ${S} under outputs_FLAC/exp09_cylNoSSL/"; exit 1; }
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXP09_LOG_DIR:-$EXPD}/exp09_${TS}_cylNoSSL_screen_S${S}.log"
ONLINE_CFG="${EXPD}/FLAC_AR_exp09_online_eval.json"

{
  echo "=== exp09 screen S${S} (EMA; online if present) - ${TS} - $(git rev-parse --short HEAD) - ckpt ${CKPT} ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config "${EXPD}/FLAC_AR_exp09.json" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-method fa_invariant --frame-avg-angles 0 \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp09_cylNoSSL_screen_S${S}_ema"
  echo "=== EMA eval exit $? ==="
  if [ -f "$ONLINE_CFG" ]; then
    CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
      --model-config "$ONLINE_CFG" \
      --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
      --ckpt-path "$CKPT" \
      --cond-method fa_invariant --frame-avg-angles 0 \
      --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name "exp09_cylNoSSL_screen_S${S}_online"
    echo "=== ONLINE eval exit $? ==="
  else
    echo "=== ONLINE eval SKIPPED (no ${ONLINE_CFG}; D-stage artifact) ==="
  fi
} >> "$LOG" 2>&1

echo "SCREEN S${S} DONE -> ${LOG}"
grep -aE "eval exit [0-9]+" "$LOG"
if grep -aqE "eval exit [1-9]" "$LOG" || ! grep -aqE "eval exit 0" "$LOG"; then
  echo "SCREEN S${S} HAD FAILURES - inspect ${LOG}"
  exit 1
fi
