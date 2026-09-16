#!/bin/bash
# exp_14 ladder chain: rungs 5b -> 6 -> 7, sequentially, on the same box.
# Hardened in round F (codex finding 4): the previous version had no `set -e` and no rung
# acceptance at all, so it wrote CHAIN_DONE even when every rung had failed. Each rung script
# now exits non-zero unless its own acceptance holds; the chain additionally requires that the
# rung wrote a FRESH .done marker, and stops at the first rung that does not.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
TS=$(date +%Y-%m-%d_%H-%M-%S); START=$(date +%s)
fail () { printf 'CHAIN_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung5b67_chain_${TS}.failed"
  echo "chain FAILED: $*" >&2; exit 1; }
# <script>:<marker prefix> -- the marker each rung writes only when its acceptance holds.
for RUNG in rung5b_probe.sh:rung5b_probe rung6_fit_probe.sh:rung6_fit rung7_nas_ckpt.sh:rung7_nas; do
  SCRIPT="${RUNG%%:*}"; PREFIX="${RUNG##*:}"; RC=0
  echo "=== chain $SCRIPT start $(date -Is)"
  bash "$REC/$SCRIPT" || RC=$?
  [ "$RC" = 0 ] || fail "$SCRIPT exited $RC"
  MARKER=$(find "$REC" -maxdepth 1 -name "${PREFIX}_*.done" -newermt "@$START" -print -quit)
  [ -n "$MARKER" ] || fail "$SCRIPT exited 0 but wrote no fresh ${PREFIX}_*.done marker"
  echo "=== chain $SCRIPT ok $(date -Is) | $MARKER"
done
echo "CHAIN_DONE $(date -Is)" > "$REC/rung5b67_chain_${TS}.done"
