#!/usr/bin/env bash
# Copy AcousticRooms (27G, 615,265 files) from local NVMe to the CIFS share.
# Non-destructive: the source is never touched. Restartable + idempotent.
set -uo pipefail

SRC=/home/yixunhu/codespace/FLAC/AcousticRooms
DST=/media/diskstation/yixunhu/FLAC/AcousticRooms

# The share is mounted uid=0/gid=0, file_mode=0777, so as a non-root user we can
# set neither ownership, permissions, nor mtimes -- utime() returns EPERM. That
# makes rsync's default size+mtime quick check useless (every dest file would
# look newer than its source and be re-sent on every pass), so the resume check
# is --size-only. Sizes are authoritative here: this is a read-only dataset, so
# a size match cannot mask an in-place content edit, and any file cut short by
# an interrupted pass has the wrong size and is re-sent.
# No -t/-p/-o/-g => rsync never attempts the calls that would EPERM, so a clean
# pass exits 0 instead of 23 and the retry loop can trust the exit code.
RSYNC_OPTS=(-r --size-only --no-perms --no-owner --no-group --info=progress2,stats2)

rc=1
for attempt in 1 2 3 4 5; do
    echo "=== rsync attempt ${attempt} @ $(date -Is) ==="
    rsync "${RSYNC_OPTS[@]}" "$SRC/" "$DST/"
    rc=$?
    echo "=== attempt ${attempt} exit=${rc} @ $(date -Is) ==="
    if [ "$rc" -eq 0 ]; then
        echo "COPY_OK after ${attempt} attempt(s)"
        exit 0
    fi
    # The share is a 'soft' mount, so a server hiccup surfaces as an I/O error
    # rather than blocking forever. Pause, then re-pass: already-copied files
    # are skipped on size, so a retry costs a metadata scan, not a re-transfer.
    sleep 30
done

echo "COPY_FAILED after 5 attempts (last rc=${rc})"
exit 1
