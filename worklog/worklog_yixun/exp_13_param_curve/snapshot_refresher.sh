#!/bin/bash
# exp_13: keep the NAS runtime snapshot current while cylB trains. Copy-only; sha-verified;
# every ~2h; exits when training ends (final state then archived by the chain itself).
set -uo pipefail
SRC=/home/yixunhu/codespace/exp-12-arms
NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp13_param_curve/runtime_snapshot
R=$SRC/worklog/worklog_yixun/exp_13_param_curve
while pgrep -f "train.py.*exp13_cylB" > /dev/null || pgrep -f "train.py.*exp13_vanB" > /dev/null; do
  for RUN in exp13_cylB exp13_vanB; do
    CK=$(ls -t $SRC/outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null | grep -v '\.part' | head -1)
    [ -n "$CK" ] || continue
    B=$(basename "$CK")
    if [ ! -f "$NAS/checkpoints/$B" ]; then
      cp "$CK" "$NAS/checkpoints/$B.part" && mv "$NAS/checkpoints/$B.part" "$NAS/checkpoints/$B"
      HL=$(sha256sum "$CK" | cut -d' ' -f1); HN=$(sha256sum "$NAS/checkpoints/$B" | cut -d' ' -f1)
      if [ "$HL" = "$HN" ]; then echo "$HN  $B" >> "$NAS/SNAPSHOT.sha256"
      else rm -f "$NAS/checkpoints/$B"; fi
    fi
  done
  cp $R/chain_exp13.log $R/train_exp13_cylB.log $NAS/ 2>/dev/null
  cp $R/train_exp13_vanB.log $NAS/ 2>/dev/null
  sleep 7200
done
echo "trainings ended; refresher standing down $(date -Is)" >> "$NAS/refresher.log"
