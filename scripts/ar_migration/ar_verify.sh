#!/usr/bin/env bash
# Verify the NAS copy of AcousticRooms against the local source.
# STRICTLY READ-ONLY: deletes nothing, moves nothing, creates no symlink.
#
# Codex review (gpt-5.6-sol, xhigh) killed the previous version for two
# blocking reasons, both fixed here:
#   (a) it ignored every command's exit status, so on this 100%-full NVMe an
#       ENOSPC from sort would leave BOTH manifests empty -> 0 == 0 -> a
#       "VERIFY_OK" that had checked nothing. Every step is now status-checked
#       and there is an explicit non-empty floor on the manifests.
#   (b) a 300-of-615265 sample detects a single bad file 0.05% of the time.
#       The sample is retained only as a labelled spot check, never as proof,
#       and it now uses real entropy instead of `yes 42`.
set -uo pipefail

SRC=/home/yixunhu/codespace/FLAC/AcousticRooms
DST=/media/diskstation/yixunhu/FLAC/AcousticRooms
SAMPLE_N=${SAMPLE_N:-500}
FIND_TIMEOUT=${FIND_TIMEOUT:-3600}

die() { echo; echo "VERIFY_FAILED -- $*"; exit 1; }

case "$SAMPLE_N" in ''|*[!0-9]*) die "SAMPLE_N must be a positive integer";; esac
[ "$SAMPLE_N" -gt 0 ] || die "SAMPLE_N must be > 0"

# The copy must be finished: a tree still being written can pass a scan and
# change afterwards.
if pgrep -f "rsync .*AcousticRooms" >/dev/null 2>&1; then
    die "an rsync on AcousticRooms is still running -- verify only after COPY_OK"
fi

# Confirm DST really sits on the expected CIFS share. Without this we could be
# verifying an empty local mountpoint directory that merely has the right path.
dst_fstype=$(findmnt -n -o FSTYPE --target "$DST" 2>/dev/null) || dst_fstype=""
dst_src=$(findmnt -n -o SOURCE --target "$DST" 2>/dev/null) || dst_src=""
[ "$dst_fstype" = "cifs" ] || die "$DST is not on a cifs mount (fstype='$dst_fstype') -- share not mounted?"
echo "dest mount: $dst_src ($dst_fstype)"

# Manifests are ~40MB each plus sort scratch. The root fs is full, so pick the
# work dir deliberately and prove it has room instead of discovering ENOSPC
# halfway through.
REUSE=0
if [ -n "${VERIFY_REUSE_WORK:-}" ]; then
    [ -s "${VERIFY_REUSE_WORK}/src.mf" ] && [ -s "${VERIFY_REUSE_WORK}/dst.mf" ] \
        || die "VERIFY_REUSE_WORK=${VERIFY_REUSE_WORK} lacks a complete src.mf/dst.mf pair"
    WORK="${VERIFY_REUSE_WORK}"; REUSE=1
else
    WORK=$(mktemp -d "${VERIFY_TMPDIR:-/tmp}/ar_verify.XXXXXX") || die "mktemp failed"
fi
avail_k=$(df -Pk "$WORK" | awk 'NR==2{print $4}') || die "df failed on $WORK"
[ -n "$avail_k" ] && [ "$avail_k" -ge 1048576 ] \
    || die "need >=1GiB scratch in $WORK, have ${avail_k}K -- set VERIFY_TMPDIR"
export TMPDIR="$WORK"
echo "work dir: $WORK  (${avail_k}K free)"
echo

# --- manifest: every path, its type, and (for files) its byte size ----------
# NUL-delimited end to end so newlines/spaces/unicode in names cannot split a
# record. Directories and symlinks are inventoried too, so VERIFY_OK means the
# whole tree matched, not just the regular files.
manifest() {
    local root=$1 out=$2
    ( cd "$root" || exit 1
      find . \
          \( -type f -printf 'F %s %p\0' \) -o \
          \( -type d -printf 'D 0 %p\0' \) -o \
          \( -type l -printf 'L 0 %p\0' \) -o \
          \( -printf 'X 0 %p\0' \)
    ) > "$out.raw"
    local rc=${PIPESTATUS[0]:-$?}
    [ "$rc" -eq 0 ] || return 1
    LC_ALL=C sort -z "$out.raw" > "$out" || return 1
    rm -f "$out.raw"
    [ -s "$out" ] || return 1
    return 0
}

if [ "$REUSE" -eq 1 ]; then
    echo "[1/4] reusing source manifest from $WORK"
    echo "[2/4] reusing destination manifest from $WORK"
else
    echo "[1/4] scanning source ..."
    timeout "$FIND_TIMEOUT" bash -c "$(declare -f manifest); manifest '$SRC' '$WORK/src.mf'" \
        || die "source manifest failed (rc=$?)"
    echo "[2/4] scanning destination (CIFS, this is the slow one) ..."
    timeout "$FIND_TIMEOUT" bash -c "$(declare -f manifest); manifest '$DST' '$WORK/dst.mf'" \
        || die "destination manifest failed (rc=$?) -- CIFS error or timeout"
fi

# A pathname may legally contain a newline, which would let `tr` forge extra
# awk records. Rather than assume none do, prove it: the NUL-record count must
# equal the line count after translation. If they differ, bail out loudly.
assert_no_newlines() {
    local mf=$1 recs lines
    recs=$(tr -cd '\0' < "$mf" | wc -c) || return 1
    lines=$(tr '\0' '\n' < "$mf" | wc -l) || return 1
    [ "$recs" -eq "$lines" ] || { echo "  ($mf: $recs records vs $lines lines)"; return 1; }
    return 0
}
assert_no_newlines "$WORK/src.mf" || die "a source pathname contains a newline -- stats would be wrong"
assert_no_newlines "$WORK/dst.mf" || die "a dest pathname contains a newline -- stats would be wrong"

# All four counters use %.0f: %d truncates to int32 and saturates at INT_MAX,
# which is exactly how the byte total silently became 2147483647 on an earlier
# run and made the equality check vacuous.
stats() {  # -> "files dirs others bytes"
    tr '\0' '\n' < "$1" | LC_ALL=C awk '
        $1=="F"{f++; b+=$2} $1=="D"{d++} $1=="L"||$1=="X"{o++}
        END{printf "%.0f %.0f %.0f %.0f\n", f+0, d+0, o+0, b+0}'
}
# Capture the producer's status first: `read < <(...)` only reports read's own
# status, so a failed pipeline could otherwise slip through as success.
src_stats=$(stats "$WORK/src.mf") || die "source stats pipeline failed"
dst_stats=$(stats "$WORK/dst.mf") || die "dest stats pipeline failed"
read -r sf sd so sb <<<"$src_stats" || die "source stats malformed: '$src_stats'"
read -r df_ dd do_ db <<<"$dst_stats" || die "dest stats malformed: '$dst_stats'"

printf '    source: %d files, %d dirs, %d other, %d bytes\n' "$sf" "$sd" "$so" "$sb"
printf '    dest  : %d files, %d dirs, %d other, %d bytes\n' "$df_" "$dd" "$do_" "$db"

# Explicit floor: an empty or truncated manifest can never read as success.
# The manifest records a symlink's path but not its target, and collapses all
# special files to 'X'. This dataset is documented as files+dirs only, so any
# such entry means the manifest comparison is weaker than it claims -- reject
# rather than silently under-verify.
[ "$so" -eq 0 ] || die "source has $so symlink/special entries; their targets are not compared"
[ "$do_" -eq 0 ] || die "dest has $do_ symlink/special entries; their targets are not compared"
[ "$sf" -ge 600000 ] || die "source manifest implausible ($sf files) -- scan did not complete"
[ "$sb" -gt 0 ]      || die "source byte total is zero -- scan did not complete"

echo
echo "[3/4] comparing full trees (path + type + size, every entry) ..."
[ "$sf" -eq "$df_" ] || die "file count mismatch: src=$sf dst=$df_"
[ "$sd" -eq "$dd"  ] || die "dir count mismatch: src=$sd dst=$dd"
[ "$sb" -eq "$db"  ] || die "total byte mismatch: src=$sb dst=$db (delta $((sb-db)))"
if cmp -s "$WORK/src.mf" "$WORK/dst.mf"; then
    echo "    inventories matched: $sf files + $sd dirs, same paths, equal file sizes"
else
    diff <(tr '\0' '\n' < "$WORK/src.mf") <(tr '\0' '\n' < "$WORK/dst.mf") \
        > "$WORK/tree.diff" 2>&1
    echo "    first differing entries:"; head -20 "$WORK/tree.diff"
    die "tree manifests differ -- see $WORK/tree.diff"
fi

# --- spot check: real hashes on a real random sample ------------------------
# Labelled honestly: this raises confidence against silent corruption, it does
# NOT prove it absent (500/615265 = 0.08% detection for a single bad file).
echo
echo "[4/4] SPOT CHECK ONLY: sha256 on $SAMPLE_N randomly chosen files ..."
( cd "$SRC" && find . -type f -print0 ) > "$WORK/all.z" || die "sample enumerate failed"
shuf -z -n "$SAMPLE_N" "$WORK/all.z" > "$WORK/sample.z" || die "shuf failed"
got=$(tr -cd '\0' < "$WORK/sample.z" | wc -c) || die "sample count failed"
[ "$got" -eq "$SAMPLE_N" ] || die "sample has $got entries, expected $SAMPLE_N"

bad=0; checked=0
while IFS= read -r -d '' rel; do
    s=$(sha256sum -- "$SRC/$rel" 2>/dev/null | cut -d' ' -f1) || s=""
    d=$(sha256sum -- "$DST/$rel" 2>/dev/null | cut -d' ' -f1) || d=""
    if [ -z "$s" ] || [ -z "$d" ]; then
        echo "    !! unreadable: $rel (src='$s' dst='$d')"; bad=$((bad+1))
    elif [ "$s" != "$d" ]; then
        echo "    !! HASH MISMATCH: $rel"; bad=$((bad+1))
    fi
    checked=$((checked+1))
done < "$WORK/sample.z"

[ "$checked" -eq "$SAMPLE_N" ] || die "only $checked/$SAMPLE_N sampled files were checked"
[ "$bad" -eq 0 ] || die "$bad of $checked sampled files failed"
echo "    all $checked sampled files: SHA-256 digests matched"

echo
echo "VERIFY_OK"
echo "  - $sf files / $sd dirs: same paths, equal recorded sizes (exhaustive)"
echo "  - $sb bytes accounted for on both sides"
echo "  - $checked randomly sampled files: SHA-256 digests matched"
echo "  SCOPE: verified as of the scan times above. Content was compared only"
echo "         for the $checked sampled files ($checked/$sf); the rest matched on"
echo "         size alone. For an irreversible delete, run a full-checksum pass."
echo "  work dir kept for inspection: $WORK"
exit 0
