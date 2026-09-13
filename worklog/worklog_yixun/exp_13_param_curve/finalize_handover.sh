#!/bin/bash
# exp_13 handoff finalizer: after the stop-at-7500 lands, produce the authoritative NAS
# handover state. Copy-only + sha-verified; never deletes.
set -uo pipefail
TPID="${1:?train pid}"
cd /home/yixunhu/codespace/exp-12-arms
NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp13_param_curve/runtime_snapshot
R=worklog/worklog_yixun/exp_13_param_curve
LOG=$R/finalize_handover.log
say () { echo "[finalize] $* | $(date -Is)" >> "$LOG"; }
say "waiting for train pid $TPID to exit"
while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
grep -q "stopped cleanly at step 7500" worklog/worklog_yixun/exp_12_arms/stop_at_ckpt_exp13_cylB.log \
  || { say "STOP marker missing -- manual look needed"; exit 3; }
CK=$(ls outputs_FLAC/exp13_cylB/*/*/checkpoints/*step=7500.ckpt | head -1)
[ -n "$CK" ] || { say "no 7500 ckpt"; exit 3; }
B=$(basename "$CK")
cp "$CK" "$NAS/checkpoints/$B.part" && mv "$NAS/checkpoints/$B.part" "$NAS/checkpoints/$B"
HL=$(sha256sum "$CK" | cut -d' ' -f1); HN=$(sha256sum "$NAS/checkpoints/$B" | cut -d' ' -f1)
[ "$HL" = "$HN" ] || { say "SHA MISMATCH on final sync"; exit 3; }
cp $R/train_exp13_cylB.log $R/chain_exp13.log worklog/worklog_yixun/exp_12_arms/stop_at_ckpt_exp13_cylB.log "$NAS/" 2>/dev/null
{ echo "AUTHORITATIVE HANDOVER -- exp13_cylB"
  echo "handover_at: $(date -Is)"
  echo "origin training: STOPPED cleanly at step 7500 (stop_at_ckpt log copied here)"
  echo "authoritative_checkpoint: checkpoints/$B"
  echo "sha256: $HN"
  echo "resume: RESUME_CKPT_CYLB=<this file> with chain v2; trains to TOTAL 40000"
  echo "origin will NOT touch exp13_cylB again."
} > "$NAS/AUTHORITATIVE_HANDOVER.txt"
echo "$HN  $B" >> "$NAS/SNAPSHOT.sha256"
pkill -f "snapshot_refresher.sh" 2>/dev/null || true
say "HANDOVER READY: $B sha $HN"
