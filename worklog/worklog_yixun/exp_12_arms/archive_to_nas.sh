#!/bin/bash
# exp_12: archive one run's checkpoints to the NAS (Yixun's standing instruction,
# 2026-08-20: "all the training checkpoints you should store on
# /media/diskstation/yixunhu/FLAC/checkpoints/").
#
#   bash archive_to_nas.sh <run_name>
#
# COPY + VERIFY, never move: every file is rsync'd to
#   /media/diskstation/yixunhu/FLAC/checkpoints/exp12_cyl_dinov3_arms/<run_name>/
# then re-read from the NAS and sha256-compared against the local file. Local copies are
# NOT deleted (standing no-deletion rule); a manifest records every hash. Re-runnable:
# rsync skips files already archived, and verification always runs on everything.
set -uo pipefail

RUN="${1:?run name}"
SRC_ROOT=/home/yixunhu/codespace/exp-12-arms/outputs_FLAC
NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp12_cyl_dinov3_arms
LOG=/home/yixunhu/codespace/exp-12-arms/worklog/worklog_yixun/exp_12_arms/archive_$RUN.log
say () { echo "[archive $RUN] $* | $(date -Is)" | tee -a "$LOG"; }

mapfile -t files < <(ls $SRC_ROOT/$RUN/*/*/checkpoints/*.ckpt $SRC_ROOT/$RUN/*.pt 2>/dev/null)
if [ "${#files[@]}" -eq 0 ]; then say "REFUSE: no checkpoints found for $RUN"; exit 2; fi

mkdir -p "$NAS/$RUN"
say "archiving ${#files[@]} files -> $NAS/$RUN/"

fail=0
for f in "${files[@]}"; do
  base=$(basename "$f")
  rsync -a --partial "$f" "$NAS/$RUN/$base"
  h_local=$(sha256sum "$f" | cut -d' ' -f1)
  h_nas=$(sha256sum "$NAS/$RUN/$base" | cut -d' ' -f1)
  if [ "$h_local" = "$h_nas" ]; then
    echo "$h_nas  $base" >> "$NAS/$RUN/MANIFEST.sha256"
    say "OK  $base ($h_nas)"
  else
    say "MISMATCH  $base local=$h_local nas=$h_nas -- NOT verified"
    fail=1
  fi
done
sort -u -k2,2 "$NAS/$RUN/MANIFEST.sha256" -o "$NAS/$RUN/MANIFEST.sha256" 2>/dev/null || true

if [ "$fail" -ne 0 ]; then say "ARCHIVE INCOMPLETE -- at least one hash mismatch"; exit 1; fi
say "archive VERIFIED: ${#files[@]} files, manifest at $NAS/$RUN/MANIFEST.sha256"
say "local copies retained (deletion only ever with Yixun's explicit permission)"
