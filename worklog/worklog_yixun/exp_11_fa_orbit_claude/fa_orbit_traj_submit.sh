#!/usr/bin/env bash
# ============================================================================
# fa_orbit_traj_submit.sh — submit ONE Q10 trajectory block: 5 seeds x 2 K.
#
# Intended to be called from the checkpoint watcher as each new (>40k) ckpt
# lands. It is a thin loop over the LOCKED submitter, not a second submission
# path: every cell still goes through fa_orbit_screen_submit.sh and therefore
# through the store lock, the campaign pin, the freeze precondition, the held-job
# lease dance and every gate. Submitting sequentially is deliberate — the
# submitter serialises on the store lock anyway, and a failure part-way should
# stop the block rather than race ten jobs into the queue.
#
#   bash fa_orbit_traj_submit.sh ARM=C8 STEP=42500 [EXCLUDE=node,node] [DRYRUN=1]
#
# Exits non-zero on the first failed cell, naming it. Already-submitted cells are
# not detected here: re-running a block resubmits it, so let the watcher own
# "has this step been done".
# ============================================================================
set -uo pipefail
EXPDIR=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude
SUBMITTER="$EXPDIR/fa_orbit_screen_submit.sh"

ARM=""; STEP=""; EXCLUDE=""; DRYRUN=0
for kv in "$@"; do
  case "$kv" in
    ARM=*|STEP=*|EXCLUDE=*|DRYRUN=*) eval_free_key="${kv%%=*}"; val="${kv#*=}" ;;
    *) echo "unknown argument '${kv}' (expected ARM=/STEP=/EXCLUDE=/DRYRUN=)" >&2; exit 2 ;;
  esac
  # values are validated by the locked submitter; shape-check the two we branch on
  case "$eval_free_key" in
    ARM)     ARM="$val" ;;
    STEP)    case "$val" in ''|*[!0-9]*) echo "STEP='${val}' is not numeric" >&2; exit 2;; esac; STEP="$val" ;;
    EXCLUDE) EXCLUDE="$val" ;;
    DRYRUN)  DRYRUN="$val" ;;
  esac
done
[ -n "$ARM" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C8 STEP=42500 [EXCLUDE=n1,n2] [DRYRUN=1]" >&2; exit 2; }

SEEDS=(42 43 44 45 46)
KS=(8 1)
echo "Q10 trajectory block: ARM=${ARM} STEP=${STEP} — ${#SEEDS[@]} seeds x ${#KS[@]} K = $(( ${#SEEDS[@]} * ${#KS[@]} )) cells"
n=0
for K in "${KS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    n=$((n + 1))
    ARGS=(ARM="$ARM" STEP="$STEP" SEED="$SEED" K="$K" CELL=traj)
    [ -n "$EXCLUDE" ] && ARGS+=(EXCLUDE="$EXCLUDE")
    if [ "$DRYRUN" = "1" ]; then
      echo "  [${n}/10] DRYRUN: bash ${SUBMITTER} ${ARGS[*]}"
      continue
    fi
    echo "  [${n}/10] submitting ${ARM} S${STEP} s${SEED} K${K} ..."
    if ! bash "$SUBMITTER" "${ARGS[@]}"; then
      echo "cell ${n}/10 (${ARM} S${STEP} s${SEED} K${K}) FAILED to submit — stopping the block" >&2
      echo "  (cells 1..$((n-1)) are already queued; resubmit the remainder by hand)" >&2
      exit 3
    fi
  done
done
echo "Q10 trajectory block complete: ${n}/10 cells submitted for ${ARM} S${STEP}"
