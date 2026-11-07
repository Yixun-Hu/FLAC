"""Incident archive (2026-11-06, Planner; exp_37 review 3 B-02): move the untracked incident quarantine and recovery logs OUT of
the exp_37 worktree into a NAS archive, preserving them byte-exactly: list the worktree's untracked files under
incident_2026-11-06/ (git), copy each to the archive with exclusive creation, verify its sha256, write a sha256 manifest, and only
then remove the source; empty directories removed afterwards. Refuses if the archive exists or any copy mismatches."""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path

W = Path("/home/yixunhu/codespace/exp-37-loc-efficiency")
REL = "worklog/worklog_yixun/exp_37_loc_efficiency_claude/incident_2026-11-06"
A = Path("/media/diskstation/yixunhu/FLAC/incident_2026-11-06_exp37_fixture_archive")
out = subprocess.run(["git", "-C", str(W), "status", "--porcelain", "--untracked-files=all", "--", REL],
                     capture_output=True, text=True, check=True).stdout.splitlines()
files = [l[3:] for l in out if l.startswith("?? ")]
if any(not l.startswith("?? ") for l in out):
    sys.exit(f"REFUSED: tracked changes under {REL}: {[l for l in out if not l.startswith('?? ')]}")
if not files:
    sys.exit("REFUSED: nothing untracked to archive")
if A.exists():
    sys.exit(f"REFUSED: {A} exists")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for rel in files:
    src = W / rel
    if src.is_symlink() or not src.is_file():
        sys.exit(f"REFUSED: {src} is not a regular file")
    rows.append((sha(src), rel))
A.mkdir(parents=False)
for digest, rel in rows:
    dst = A / Path(rel).relative_to(REL)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(W / rel, "rb") as fi, open(dst, "xb") as fo:
        shutil.copyfileobj(fi, fo); fo.flush(); os.fsync(fo.fileno())
    if sha(dst) != digest:
        sys.exit(f"REFUSED: the archived copy of {rel} does not match; nothing removed")
manifest = "".join(f"{d}  {Path(r).relative_to(REL)}\n" for d, r in rows)
(A / "SHA256SUMS").write_text(manifest)
for _digest, rel in rows:
    (W / rel).unlink()
for d in sorted({(W / r).parent for _x, r in rows}, key=lambda p: len(p.parts), reverse=True):
    while d != W / REL and d.is_dir() and not any(d.iterdir()):
        d.rmdir(); d = d.parent
print(f"archived {len(rows)} files to {A}; SHA256SUMS sha256 {hashlib.sha256(manifest.encode()).hexdigest()}")
