Reading prompt from stdin...
2026-08-19T19:56:39.697578Z ERROR codex_models_manager::cache: failed to load models cache: missing field `base_instructions` at line 97 column 5
OpenAI Codex v0.144.1
--------
workdir: /home/yixunhu/codespace/FLAC
model: gpt-5.6-sol
provider: openai
approval: on-request
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 01a01b98-e4e9-7133-bf59-739e018102ca
--------
user
You are reviewing two one-off ops shell scripts, at xhigh rigor.

HARD CONSTRAINT: This is a READ-ONLY review. Do NOT install anything, do not
run pip/conda/apt, do not modify any environment, do not create/modify/delete
any file, and do not execute either script. Read and reason only.

CONTEXT
- Task: relocate a 27 GB, 615,265-file read-only research dataset
  (AcousticRooms) off a 100%-full local NVMe onto a CIFS/SMB share, then
  replace the original directory with a symlink to the new location.
- Source: /home/yixunhu/codespace/FLAC/AcousticRooms  (local ext4, 615,265
  files, 816 dirs, 0 symlinks, 9 of the files are .DS_Store)
- Dest:   /media/diskstation/yixunhu/FLAC/AcousticRooms
- Mount:  //mae-cab-nas.princeton.edu/shared_data on /media/diskstation type
  cifs (rw,vers=3.0,cache=strict,username=server8,uid=0,noforceuid,gid=0,
  noforcegid,file_mode=0777,dir_mode=0777,soft,nounix,serverino,mapposix,
  actimeo=1,closetimeo=1)
- We run as non-root user yixunhu. On this share utime()/chown()/chmod() all
  return EPERM, so mtimes/owner/perms CANNOT be preserved. The dataset is
  read-only and never edited in place.
- 5 sibling git checkouts already symlink to the SOURCE path, so the final
  symlink must keep that path resolving.
- ar_copy.sh has ALREADY RUN and is mid-flight. ar_verify.sh has NOT run.
  After verification passes, the plan is: rm -rf the source, then
  ln -s <dest> <source path>.

REVIEW FOCUS, in priority order:
1. CORRECTNESS OF THE VERIFICATION GATE. ar_verify.sh is the only thing
   standing between a good copy and irreversibly deleting 27 GB. Is its
   evidence actually sufficient to conclude the destination is complete and
   faithful? Name any way it could report VERIFY_OK while the destination is
   missing files, has truncated files, or has corrupt content. Be specific
   and adversarial.
2. Any way ar_verify.sh could crash, hang, or silently skip checks --
   including on filenames with spaces/newlines/unicode, empty files, very
   large file counts, or a transient CIFS I/O error mid-scan. Note that
   `set -uo pipefail` is used WITHOUT `set -e`.
3. Is `shuf --random-source=<(yes 42)` a valid, actually-varying sample, or
   does it degenerate (e.g. always pick the first N, or bias the draw)?
4. Whether --size-only in ar_copy.sh is a sound resume/skip predicate given
   the EPERM-on-utime constraint, and whether the 5-attempt retry loop can
   mask a real failure.
5. Anything about the planned rm -rf + ln -s follow-up worth flagging.

Report concrete defects with severity (blocking / should-fix / nit) and the
minimal fix for each. If a check is sound, say so briefly rather than padding.

===== FILE: ar_copy.sh =====
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

===== FILE: ar_verify.sh =====
#!/usr/bin/env bash
# Verify the NAS copy of AcousticRooms against the local source.
# READ-ONLY: deletes nothing, moves nothing. Exit 0 == safe to delete source.
set -uo pipefail

SRC=/home/yixunhu/codespace/FLAC/AcousticRooms
DST=/media/diskstation/yixunhu/FLAC/AcousticRooms
WORK="$(dirname "$0")/verify"
SAMPLE_N=${SAMPLE_N:-300}
mkdir -p "$WORK"

fail=0

# --- 1. exact per-file manifest: relative path + byte size -------------------
# Sizes are the load-bearing check. mtimes are NOT comparable (the share
# refuses utime(), so every dest file carries its copy time) and a full
# checksum of 27G over SMB is too slow, so path+size over all 615k files is
# the exhaustive tier and hashes are sampled below.
echo "[1/4] building manifests ..."
( cd "$SRC" && find . -type f -printf '%s %p\n' | sort ) > "$WORK/src.manifest"
( cd "$DST" && find . -type f -printf '%s %p\n' | sort ) > "$WORK/dst.manifest"

src_n=$(wc -l < "$WORK/src.manifest")
dst_n=$(wc -l < "$WORK/dst.manifest")
echo "    source files: $src_n"
echo "    dest   files: $dst_n"

# --- 2. count + total bytes --------------------------------------------------
echo "[2/4] comparing counts and total bytes ..."
src_b=$(awk '{s+=$1} END{print s+0}' "$WORK/src.manifest")
dst_b=$(awk '{s+=$1} END{print s+0}' "$WORK/dst.manifest")
echo "    source bytes: $src_b"
echo "    dest   bytes: $dst_b"
[ "$src_n" -eq "$dst_n" ] || { echo "    !! FILE COUNT MISMATCH"; fail=1; }
[ "$src_b" -eq "$dst_b" ] || { echo "    !! TOTAL BYTE MISMATCH"; fail=1; }

# --- 3. exhaustive diff of the two manifests --------------------------------
echo "[3/4] diffing manifests (path+size, all files) ..."
if diff "$WORK/src.manifest" "$WORK/dst.manifest" > "$WORK/manifest.diff" 2>&1; then
    echo "    manifests identical"
else
    echo "    !! MANIFEST DIFF -- $(wc -l < "$WORK/manifest.diff") lines, see $WORK/manifest.diff"
    head -20 "$WORK/manifest.diff"
    fail=1
fi

# --- 4. sampled content hashes ----------------------------------------------
# Size equality cannot catch silent bit corruption in transit, so hash a random
# sample of files on both sides and compare. Sampling is seeded for
# reproducibility; -R feeds the same permutation to both sides.
echo "[4/4] hashing a ${SAMPLE_N}-file random sample on both sides ..."
( cd "$SRC" && find . -type f | sort | shuf -n "$SAMPLE_N" --random-source=<(yes 42) ) > "$WORK/sample.list"

: > "$WORK/sample.src.sha"
: > "$WORK/sample.dst.sha"
while IFS= read -r rel; do
    s=$(sha256sum "$SRC/$rel" | cut -d' ' -f1)
    d=$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)
    printf '%s  %s\n' "$s" "$rel" >> "$WORK/sample.src.sha"
    printf '%s  %s\n' "$d" "$rel" >> "$WORK/sample.dst.sha"
done < "$WORK/sample.list"

if diff "$WORK/sample.src.sha" "$WORK/sample.dst.sha" > "$WORK/sample.diff" 2>&1; then
    echo "    all ${SAMPLE_N} sampled hashes match"
else
    echo "    !! SAMPLED HASH MISMATCH, see $WORK/sample.diff"
    head -20 "$WORK/sample.diff"
    fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "VERIFY_OK  -- dest is a faithful copy; source is safe to remove"
    exit 0
fi
echo "VERIFY_FAILED -- DO NOT delete the source"
exit 1

codex
I’m doing a read-only, adversarial shell-semantics review. I’ll inspect the actual files and relevant tool behavior without executing either script or writing anything.
exec
/bin/bash -lc "sed -n '1,240p' ar_copy.sh; sed -n '1,280p' ar_verify.sh" in /home/yixunhu/codespace/FLAC
 exited 2 in 0ms:
sed: can't read ar_copy.sh: No such file or directory
sed: can't read ar_verify.sh: No such file or directory

codex
The pasted scripts aren’t present under those filenames at the workspace root, so I’m treating the supplied text as authoritative. The first blocking pattern is already clear: many critical command failures are ignored, allowing empty or stale evidence to pass.
web search: 
web search: site:gnu.org software coreutils manual shuf --random-source random bytes ...
web search: 
web search: github coreutils shuf.c randperm_new shuf random source source code ...
web search: 
web search: https://raw.githubusercontent.com/coreutils/coreutils/master/src/shuf.c
web search: 
web search: https://raw.githubusercontent.com/coreutils/coreutils/master/lib/randperm.c
web search: 
web search: https://raw.githubusercontent.com/coreutils/coreutils/master/gl/lib/randperm.c
codex
Verdict: **do not use the current `VERIFY_OK` to authorize deletion.** It can succeed with corrupt content and, under several error conditions, with little or no valid verification evidence.

## Blocking defects

1. **Blocking — content verification is only a 300-file sample**

A destination file with the correct path and byte length but incorrect contents passes unless sampled. This includes:

- A pre-existing same-sized wrong file, which `rsync --size-only` never transfers.
- Silent corruption during or after copying.
- A file truncated and then extended or sparsely recreated to its original logical size.
- An unreadable unsampled file.

Even with a genuinely uniform sample, one bad file among 615,265 is detected with probability only:

`300 / 615265 ≈ 0.0488%`

So it passes approximately **99.9512%** of the time.

Minimal fix: hash or byte-compare every regular file. For example, build NUL-delimited SHA-256 manifests from both roots and require them to compare identically. Every `find`, `sort`, `xargs`, and `sha256sum` failure must abort verification. A suitable pattern is conceptually:

```bash
hash_tree() (
    cd -- "$1"
    find . -type f -print0 |
        LC_ALL=C sort -z |
        xargs -0 -r sha256sum -z --
)
```

Run it separately for source and destination, checking both exit statuses, then compare with `cmp`. Given irreversible deletion, “full checksum is too slow” is not a sufficient correctness argument.

2. **Blocking — critical command failures are ignored**

`pipefail` only makes a failed pipeline return nonzero. Without `set -e` or explicit status checks, that status is discarded.

Unchecked operations include:

- `mkdir -p`
- Both manifest-generation pipelines
- `wc` and `awk`
- The sampling pipeline
- Every `sha256sum | cut`
- Every write to the checksum files

A concrete false-positive scenario is especially plausible because the NVMe is full:

- Existing manifest files are truncated successfully.
- `sort` cannot write because of `ENOSPC`, leaving both manifests empty.
- Both counts and totals become zero.
- Empty manifests compare equal.
- Sampling output also fails or is empty.
- Empty checksum files compare equal.
- The script prints `VERIFY_OK`.

Likewise, if both `sha256sum` calls fail for a sampled entry, `s` and `d` are both empty and the entry is recorded as matching.

Minimal fix:

- Use `set -euo pipefail`, but also explicitly guard every evidence-producing command with `if ! ...; then exit 1; fi`.
- Use a fresh, non-symlink work directory on a filesystem with verified free space.
- Never reuse stale artifacts.
- Assert the known source inventory: exactly 615,265 regular files, 816 directories, and zero symlinks.
- Treat any hash read error as an immediate failure.
- If sampling remains, require exactly `min(SAMPLE_N, file_count)` successful hashes.

3. **Blocking — the manifests and sample list are not filename-safe**

Spaces, tabs, ordinary Unicode names, and empty files are handled adequately. Newlines in filenames are not.

An adversarial example:

- Source has directory `x\n1 .` containing a one-byte file `y`.
- Destination has directory `y\n1 .` containing a one-byte file `x`.

The source manifest becomes the two lines:

```text
1 ./x
1 ./y
```

The destination produces the same two lines in the opposite order, and `sort` makes them identical. The script reports two files and two bytes on each side even though each tree actually has one differently named file.

The sample list similarly splits the source path into nonexistent paths. Both source and destination hashes fail, both captured hashes become empty, and the checksum files compare equal. This can produce `VERIFY_OK` with the destination missing the actual source file.

Minimal fix: use NUL-delimited records throughout: `find -print0` or `-printf '...\0'`, `sort -z`, `shuf -z`, `read -r -d ''`, and `sha256sum -z`. GNU documents `-z` specifically for arbitrary filenames, including embedded newlines. [GNU `shuf` documentation](https://www.gnu.org/s/coreutils/manual/html_node/shuf-invocation.html)

4. **Blocking — no mount or alias validation**

The verifier can pass while not verifying a NAS copy at all:

- The CIFS mount is absent and `DST` resolves to a fallback local directory.
- `DST` or one of its components is a symlink back to `SRC`.
- `SRC` is already a symlink to `DST`.
- Source and destination otherwise resolve to the same underlying tree.

In the alias case, every comparison is naturally identical; deleting the source can then destroy the only data.

Minimal fix, both before scanning and immediately before deletion:

- Require `SRC` to be a real directory, not a symlink.
- Require `DST` to be a real directory, not a symlink.
- Use `findmnt -T` to require filesystem type `cifs`, expected mount target, and expected remote share.
- Compare canonical paths and device/inode identities to prove source and destination are distinct.
- Recheck the mount after verification and immediately before switching paths.

5. **Blocking — verification is not safe while copying or otherwise mutable**

`ar_copy.sh` is currently running. Do not start verification until it exits successfully and no process is writing the destination.

A file can be present and full-sized when `find` examines it, then be removed, truncated, or overwritten after that point. If it is not sampled, verification still passes.

Minimal fix: orchestrate copy and verification sequentially under an exclusive lock or otherwise establish that both trees remain immutable from the first scan through source retirement.

## Sampling defects

6. **Should-fix — `yes 42` is not a seeded random generator**

`yes 42` supplies the repeating byte sequence `34 32 0a ...`. GNU `shuf` expects its random source to provide random bytes. The official reproducible example uses a seeded pseudorandom byte stream, not repeated seed text. [GNU random-source documentation](https://www.gnu.org/s/coreutils/manual/html_node/Random-sources.html)

It does not simply select the first 300 entries: GNU’s permutation/reservoir calculations will produce varying indices within the run. However:

- The subset is deterministic for the same ordered input.
- Re-running verification examines the same subset.
- The byte source has no randomness and does not justify a uniform-sample interpretation.
- Its structured periodic input can bias the selection.

Minimal fix: omit `--random-source` for a new sample each run, or use a proper seeded pseudorandom byte stream. With exhaustive hashes, remove sampling entirely.

Also validate `SAMPLE_N` as a positive integer. Currently `SAMPLE_N=0`, a negative value, or an invalid value can yield an empty sample and still print success.

The comment mentioning “`-R` feeds the same permutation to both sides” is inaccurate: no `-R` is used, and only one sample list is generated anyway.

## Other verification issues

7. **Should-fix — only regular files are inventoried**

Empty directories, extra directories, symlinks, sockets, and other node types are ignored. Therefore `VERIFY_OK` does not mean the directory tree is an exact copy.

Minimal fix: compare a separate NUL-delimited manifest of every path and its type. This also verifies all 816 directories and confirms zero destination symlinks.

8. **Should-fix — CIFS operations can stall for a long time**

A `soft` mount makes errors more likely to surface eventually, but it is not an application-level deadline. A `find`, `stat`, or file read can wait through SMB reconnect and request timeouts, and 615,000 metadata operations can already take a long time.

Minimal fix: run verification under an external deadline/watchdog, with any timeout treated as failure. An unkillable kernel I/O wait can still delay termination, but must never be converted into success.

9. **Should-fix — the work location is inappropriate for a full filesystem**

`WORK="$(dirname "$0")/verify"` may place tens or hundreds of megabytes of manifests and sort temporary data on the already-full NVMe. It also is not reliably the script directory when invoked through `PATH`.

Minimal fix: use a fresh `mktemp -d` under an explicitly selected filesystem with sufficient free space, set `TMPDIR` there for `sort`, and validate every write.

## `ar_copy.sh`

10. **Should-fix — `--size-only` is a resume optimization, not an integrity predicate**

It is reasonable when all of these are true:

- The destination began empty or was previously verified.
- The source is immutable.
- A failed transfer cannot leave an installed full-length but incorrect file.
- A final exhaustive content verification follows.

It is not safe against pre-existing same-sized content or equal-length corruption. Source immutability only rules out source edits; it says nothing about destination correctness.

Minimal fix: keep `--size-only` for the five efficient repair attempts if desired, then run a distinct checksum-based repair pass without `--size-only`, or rely on a corrected exhaustive verifier.

11. **Should-fix — a later zero exit can conceal earlier equal-size damage**

The retry mechanics themselves are correct: the script only prints `COPY_OK` when an `rsync` invocation returns zero, and a final nonzero exits 1. Retrying transient errors is sound.

The weakness is that a later `--size-only` pass can skip an incorrect full-length destination file and return zero. Thus the loop’s final zero is only evidence that all files selected by the size predicate succeeded.

Minimal fix: final checksum repair/verification as above.

12. **Nit — the fifth failed attempt sleeps unnecessarily**

After attempt five, the script still sleeps 30 seconds before reporting failure. Move the sleep so it occurs only before another attempt.

## Destructive follow-up

13. **Blocking — do not `rm -rf` first and hope `ln -s` succeeds**

After deletion, symlink creation can fail because of `ENOSPC`, permissions, a changed mount, or an unexpected path. That would leave the dataset deleted and all five sibling links broken.

Use a reversible switch:

1. Create a temporary symlink beside `SRC` before deleting anything.
2. Confirm it resolves to the expected live CIFS destination.
3. Rename `SRC` to a uniquely named local backup.
4. Rename the staged symlink into the exact `SRC` pathname.
5. Verify access through `SRC` and recheck the CIFS mount.
6. Only then remove the renamed local backup.

If the full filesystem cannot create even the staged symlink, stop; do not delete first. When eventually removing the backup, use the exact quoted path, no trailing slash, and preferably `rm -rf --one-file-system -- "$backup"`.

The intended absolute symlink at the exact original source pathname is otherwise correct and will preserve the five sibling symlink chains.

Finally, the CIFS mount advertises `rw` with mode `0777`; the destination is not filesystem-enforced read-only. If immutability is an operational premise, enforce it with NAS ACLs or a read-only mount after migration.
tokens used
70,670
Verdict: **do not use the current `VERIFY_OK` to authorize deletion.** It can succeed with corrupt content and, under several error conditions, with little or no valid verification evidence.

## Blocking defects

1. **Blocking — content verification is only a 300-file sample**

A destination file with the correct path and byte length but incorrect contents passes unless sampled. This includes:

- A pre-existing same-sized wrong file, which `rsync --size-only` never transfers.
- Silent corruption during or after copying.
- A file truncated and then extended or sparsely recreated to its original logical size.
- An unreadable unsampled file.

Even with a genuinely uniform sample, one bad file among 615,265 is detected with probability only:

`300 / 615265 ≈ 0.0488%`

So it passes approximately **99.9512%** of the time.

Minimal fix: hash or byte-compare every regular file. For example, build NUL-delimited SHA-256 manifests from both roots and require them to compare identically. Every `find`, `sort`, `xargs`, and `sha256sum` failure must abort verification. A suitable pattern is conceptually:

```bash
hash_tree() (
    cd -- "$1"
    find . -type f -print0 |
        LC_ALL=C sort -z |
        xargs -0 -r sha256sum -z --
)
```

Run it separately for source and destination, checking both exit statuses, then compare with `cmp`. Given irreversible deletion, “full checksum is too slow” is not a sufficient correctness argument.

2. **Blocking — critical command failures are ignored**

`pipefail` only makes a failed pipeline return nonzero. Without `set -e` or explicit status checks, that status is discarded.

Unchecked operations include:

- `mkdir -p`
- Both manifest-generation pipelines
- `wc` and `awk`
- The sampling pipeline
- Every `sha256sum | cut`
- Every write to the checksum files

A concrete false-positive scenario is especially plausible because the NVMe is full:

- Existing manifest files are truncated successfully.
- `sort` cannot write because of `ENOSPC`, leaving both manifests empty.
- Both counts and totals become zero.
- Empty manifests compare equal.
- Sampling output also fails or is empty.
- Empty checksum files compare equal.
- The script prints `VERIFY_OK`.

Likewise, if both `sha256sum` calls fail for a sampled entry, `s` and `d` are both empty and the entry is recorded as matching.

Minimal fix:

- Use `set -euo pipefail`, but also explicitly guard every evidence-producing command with `if ! ...; then exit 1; fi`.
- Use a fresh, non-symlink work directory on a filesystem with verified free space.
- Never reuse stale artifacts.
- Assert the known source inventory: exactly 615,265 regular files, 816 directories, and zero symlinks.
- Treat any hash read error as an immediate failure.
- If sampling remains, require exactly `min(SAMPLE_N, file_count)` successful hashes.

3. **Blocking — the manifests and sample list are not filename-safe**

Spaces, tabs, ordinary Unicode names, and empty files are handled adequately. Newlines in filenames are not.

An adversarial example:

- Source has directory `x\n1 .` containing a one-byte file `y`.
- Destination has directory `y\n1 .` containing a one-byte file `x`.

The source manifest becomes the two lines:

```text
1 ./x
1 ./y
```

The destination produces the same two lines in the opposite order, and `sort` makes them identical. The script reports two files and two bytes on each side even though each tree actually has one differently named file.

The sample list similarly splits the source path into nonexistent paths. Both source and destination hashes fail, both captured hashes become empty, and the checksum files compare equal. This can produce `VERIFY_OK` with the destination missing the actual source file.

Minimal fix: use NUL-delimited records throughout: `find -print0` or `-printf '...\0'`, `sort -z`, `shuf -z`, `read -r -d ''`, and `sha256sum -z`. GNU documents `-z` specifically for arbitrary filenames, including embedded newlines. [GNU `shuf` documentation](https://www.gnu.org/s/coreutils/manual/html_node/shuf-invocation.html)

4. **Blocking — no mount or alias validation**

The verifier can pass while not verifying a NAS copy at all:

- The CIFS mount is absent and `DST` resolves to a fallback local directory.
- `DST` or one of its components is a symlink back to `SRC`.
- `SRC` is already a symlink to `DST`.
- Source and destination otherwise resolve to the same underlying tree.

In the alias case, every comparison is naturally identical; deleting the source can then destroy the only data.

Minimal fix, both before scanning and immediately before deletion:

- Require `SRC` to be a real directory, not a symlink.
- Require `DST` to be a real directory, not a symlink.
- Use `findmnt -T` to require filesystem type `cifs`, expected mount target, and expected remote share.
- Compare canonical paths and device/inode identities to prove source and destination are distinct.
- Recheck the mount after verification and immediately before switching paths.

5. **Blocking — verification is not safe while copying or otherwise mutable**

`ar_copy.sh` is currently running. Do not start verification until it exits successfully and no process is writing the destination.

A file can be present and full-sized when `find` examines it, then be removed, truncated, or overwritten after that point. If it is not sampled, verification still passes.

Minimal fix: orchestrate copy and verification sequentially under an exclusive lock or otherwise establish that both trees remain immutable from the first scan through source retirement.

## Sampling defects

6. **Should-fix — `yes 42` is not a seeded random generator**

`yes 42` supplies the repeating byte sequence `34 32 0a ...`. GNU `shuf` expects its random source to provide random bytes. The official reproducible example uses a seeded pseudorandom byte stream, not repeated seed text. [GNU random-source documentation](https://www.gnu.org/s/coreutils/manual/html_node/Random-sources.html)

It does not simply select the first 300 entries: GNU’s permutation/reservoir calculations will produce varying indices within the run. However:

- The subset is deterministic for the same ordered input.
- Re-running verification examines the same subset.
- The byte source has no randomness and does not justify a uniform-sample interpretation.
- Its structured periodic input can bias the selection.

Minimal fix: omit `--random-source` for a new sample each run, or use a proper seeded pseudorandom byte stream. With exhaustive hashes, remove sampling entirely.

Also validate `SAMPLE_N` as a positive integer. Currently `SAMPLE_N=0`, a negative value, or an invalid value can yield an empty sample and still print success.

The comment mentioning “`-R` feeds the same permutation to both sides” is inaccurate: no `-R` is used, and only one sample list is generated anyway.

## Other verification issues

7. **Should-fix — only regular files are inventoried**

Empty directories, extra directories, symlinks, sockets, and other node types are ignored. Therefore `VERIFY_OK` does not mean the directory tree is an exact copy.

Minimal fix: compare a separate NUL-delimited manifest of every path and its type. This also verifies all 816 directories and confirms zero destination symlinks.

8. **Should-fix — CIFS operations can stall for a long time**

A `soft` mount makes errors more likely to surface eventually, but it is not an application-level deadline. A `find`, `stat`, or file read can wait through SMB reconnect and request timeouts, and 615,000 metadata operations can already take a long time.

Minimal fix: run verification under an external deadline/watchdog, with any timeout treated as failure. An unkillable kernel I/O wait can still delay termination, but must never be converted into success.

9. **Should-fix — the work location is inappropriate for a full filesystem**

`WORK="$(dirname "$0")/verify"` may place tens or hundreds of megabytes of manifests and sort temporary data on the already-full NVMe. It also is not reliably the script directory when invoked through `PATH`.

Minimal fix: use a fresh `mktemp -d` under an explicitly selected filesystem with sufficient free space, set `TMPDIR` there for `sort`, and validate every write.

## `ar_copy.sh`

10. **Should-fix — `--size-only` is a resume optimization, not an integrity predicate**

It is reasonable when all of these are true:

- The destination began empty or was previously verified.
- The source is immutable.
- A failed transfer cannot leave an installed full-length but incorrect file.
- A final exhaustive content verification follows.

It is not safe against pre-existing same-sized content or equal-length corruption. Source immutability only rules out source edits; it says nothing about destination correctness.

Minimal fix: keep `--size-only` for the five efficient repair attempts if desired, then run a distinct checksum-based repair pass without `--size-only`, or rely on a corrected exhaustive verifier.

11. **Should-fix — a later zero exit can conceal earlier equal-size damage**

The retry mechanics themselves are correct: the script only prints `COPY_OK` when an `rsync` invocation returns zero, and a final nonzero exits 1. Retrying transient errors is sound.

The weakness is that a later `--size-only` pass can skip an incorrect full-length destination file and return zero. Thus the loop’s final zero is only evidence that all files selected by the size predicate succeeded.

Minimal fix: final checksum repair/verification as above.

12. **Nit — the fifth failed attempt sleeps unnecessarily**

After attempt five, the script still sleeps 30 seconds before reporting failure. Move the sleep so it occurs only before another attempt.

## Destructive follow-up

13. **Blocking — do not `rm -rf` first and hope `ln -s` succeeds**

After deletion, symlink creation can fail because of `ENOSPC`, permissions, a changed mount, or an unexpected path. That would leave the dataset deleted and all five sibling links broken.

Use a reversible switch:

1. Create a temporary symlink beside `SRC` before deleting anything.
2. Confirm it resolves to the expected live CIFS destination.
3. Rename `SRC` to a uniquely named local backup.
4. Rename the staged symlink into the exact `SRC` pathname.
5. Verify access through `SRC` and recheck the CIFS mount.
6. Only then remove the renamed local backup.

If the full filesystem cannot create even the staged symlink, stop; do not delete first. When eventually removing the backup, use the exact quoted path, no trailing slash, and preferably `rm -rf --one-file-system -- "$backup"`.

The intended absolute symlink at the exact original source pathname is otherwise correct and will preserve the five sibling symlink chains.

Finally, the CIFS mount advertises `rw` with mode `0777`; the destination is not filesystem-enforced read-only. If immutability is an operational premise, enforce it with NAS ACLs or a read-only mount after migration.
