#!/bin/bash
# Yixun 2026-09-30 10:08 EDT (verbatim): "How about you move the needed to moved checkpoint and symlink them to NAS"
# Archive-then-symlink of finished-experiment checkpoints: copy to the NAS if no copy exists, sha256 BOTH sides, and only
# on an exact match replace the local file with a symlink to the NAS copy. On any mismatch/failure both copies are kept
# and a MISMATCH line is logged. Nothing else is touched (metrics JSONs, prediction dumps, logs stay local).
# Scope (the candidates listed to Yixun 2026-09-30 07:51): exp_23 CYLORI/CYLORI27/CYLORI27_s43, exp_24 four HAA arms,
# exp13_cylB early checkpoints (2.5k/5k/7.5k). Detach: setsid nohup bash archive_ckpts_to_nas.sh > /dev/null 2>&1 &
set -uo pipefail
K=/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/housekeeping_nas_archive_claude
N=/media/diskstation/yixunhu/FLAC/checkpoints; LOG=$K/archive.log; MAN=$K/MANIFEST.sha256
say(){ echo "[nas-archive] $* | $(date -Is)" | tee -a "$LOG"; }
freed=0; nsym=0; ncopy=0; nmis=0; nskip=0
archive_one(){ # LOCAL NAS_TARGET
  local L="$1" R="$2"
  [ -L "$L" ] && { nskip=$((nskip+1)); return; }
  [ -f "$L" ] || { say "missing local $L"; return; }
  if lsof -- "$L" >/dev/null 2>&1; then say "SKIP (open by a process): $L"; nskip=$((nskip+1)); return; fi
  mkdir -p "$(dirname "$R")"
  local sz; sz=$(stat -c %s "$L")
  if [ ! -e "$R" ]; then
    say "copy $sz bytes -> $R"
    if cp -- "$L" "$R.part" && mv -f -- "$R.part" "$R"; then ncopy=$((ncopy+1)); else say "COPY FAILED $L"; rm -f -- "$R.part"; nmis=$((nmis+1)); return; fi
  fi
  local ls rs; ls=$(sha256sum -- "$L" | cut -c1-64); rs=$(sha256sum -- "$R" | cut -c1-64)
  if [ "$ls" = "$rs" ]; then
    if ln -s -- "$R" "$L.__lnk" && mv -f -T -- "$L.__lnk" "$L"; then
      echo "$ls  $R" >> "$MAN"; freed=$((freed+sz)); nsym=$((nsym+1)); say "verified + symlinked $L ($ls)"
    else say "SYMLINK FAILED $L"; rm -f -- "$L.__lnk"; nmis=$((nmis+1)); fi
  else say "MISMATCH (both copies kept): $L local=$ls nas=$rs"; nmis=$((nmis+1)); fi
}
cd /home/yixunhu/codespace/FLAC
say "START pid $$ | local free $(df -h / | tail -1 | awk '{print $4}') | NAS free $(df -h /media/diskstation | tail -1 | awk '{print $4}')"
for ARM in CYLORI CYLORI27 CYLORI27_s43; do
  while IFS= read -r f; do rel=${f#outputs_FLAC/exp23_HAA_$ARM/}; archive_one "$f" "$N/exp23_haa_orientation/$ARM/$rel"; done < <(find outputs_FLAC/exp23_HAA_$ARM -name '*.ckpt' -type f | sort)
  say "exp23 $ARM done | local free $(df -h / | tail -1 | awk '{print $4}')"
done
for ARM in P1ORI27 YAWORI27 P1ZUP27 CYLZUP27; do
  while IFS= read -r f; do rel=${f#outputs_FLAC/exp24_HAA_$ARM/}; archive_one "$f" "$N/exp24_haa_orientation_fairness/$ARM/$rel"; done < <(find outputs_FLAC/exp24_HAA_$ARM -name '*.ckpt' -type f | sort)
  say "exp24 $ARM done | local free $(df -h / | tail -1 | awk '{print $4}')"
done
W=/home/yixunhu/codespace/exp-12-arms
while IFS= read -r f; do archive_one "$f" "$N/exp13_param_curve/exp13_cylB/$(basename "$f")"; done < <(find $W/outputs_FLAC/exp13_cylB -name '*.ckpt' -type f | sort)
say "exp13_cylB early done | local free $(df -h / | tail -1 | awk '{print $4}')"
cp -- "$MAN" "$N/exp23_haa_orientation/MANIFEST_housekeeping_2026-09-30.sha256" 2>/dev/null
say "DONE: symlinked $nsym, copied $ncopy, mismatches $nmis, skipped $nskip, freed $((freed/1000000000)) GB | local free $(df -h / | tail -1 | awk '{print $4}')"
echo "NAS_ARCHIVE_DONE $(date -Is)" >> "$LOG"
