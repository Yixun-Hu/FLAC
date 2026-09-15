#!/usr/bin/env bash
# OPTIONAL, RUN BY HAND. Swap the local AcousticRooms folder for a symlink to
# the NAS copy. Reversible at every step: nothing is deleted until the symlink
# is in place AND proven to serve real file content.
#
# Ordering per Codex review finding #13: `rm -rf` followed by `ln -s` is unsafe,
# because if ln fails (ENOSPC, permissions, mount vanished) the dataset is gone
# AND all five sibling checkouts that symlink into this path are broken.
# Instead: stage -> rename aside -> swap -> prove -> only then delete.
set -uo pipefail

SRC=/home/yixunhu/codespace/FLAC/AcousticRooms
DST=/media/diskstation/yixunhu/FLAC/AcousticRooms
BACKUP="${SRC}.local-backup-$$"
STAGE="${SRC}.newlink-$$"

die() { echo "!! ABORT: $*" >&2; exit 1; }

[ -d "$SRC" ] && [ ! -L "$SRC" ] || die "$SRC is not a real directory (already migrated?)"
[ -d "$DST" ] || die "destination $DST is not present -- is the share mounted?"
mountpoint -q /media/diskstation || die "/media/diskstation is not mounted"

echo "==> run ar_verify.sh first and confirm VERIFY_OK. Continuing in 10s (Ctrl-C to stop)"
sleep 10

# 1. stage the symlink BESIDE the source. If this fails we have lost nothing.
ln -s "$DST" "$STAGE" || die "could not create staged symlink (nothing was changed)"
[ -d "$STAGE/" ] || { rm -f "$STAGE"; die "staged symlink does not resolve (nothing was changed)"; }
echo "[1/5] staged symlink resolves"

# 2. move the real directory aside. Same filesystem, so this is an atomic
#    rename, not a copy -- instant, and instantly undoable.
mv -T "$SRC" "$BACKUP" || { rm -f "$STAGE"; die "could not rename source aside (nothing was changed)"; }
echo "[2/5] source renamed to $BACKUP"

# 3. put the symlink at the original path.
if ! mv -T "$STAGE" "$SRC"; then
    mv -T "$BACKUP" "$SRC" && echo "rolled back cleanly"
    die "could not install symlink; source restored"
fi
echo "[3/5] symlink installed at $SRC"

# 4. prove the path serves real content through the link, including for the
#    sibling checkouts that chain through it.
probe=$(find "$SRC/single_channel_ir_1" -type f -name '*.wav' -print -quit 2>/dev/null)
if [ -z "$probe" ] || [ ! -s "$probe" ]; then
    mv -T "$SRC" "$STAGE" && mv -T "$BACKUP" "$SRC" && echo "rolled back cleanly"
    die "symlink does not serve file content; source restored"
fi
echo "[4/5] reads through symlink OK: $probe"
for sib in exp-09-cyl-dinov3-no-ssl exp-10-cyl-distill exp-12-arms \
           exp-17-yawaug-a6000 rir2rir-exp17; do
    p="/home/yixunhu/codespace/$sib/AcousticRooms"
    [ -e "$p/single_channel_ir_1" ] && echo "      sibling OK: $sib" \
                                    || echo "      !! sibling BROKEN: $sib"
done

# 5. only now is deleting the local copy safe.
echo
echo "[5/5] migration complete. The local copy is still on disk at:"
echo "          $BACKUP"
echo "      Reclaim the 27G when you are satisfied:"
echo "          rm -rf --one-file-system -- \"$BACKUP\""
echo "      To undo instead:"
echo "          rm -f \"$SRC\" && mv -T \"$BACKUP\" \"$SRC\""
