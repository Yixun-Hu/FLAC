#!/usr/bin/env python3
"""exp_38 — content binding of every AcousticRooms file the training runs read (Codex code r2 BLOCKING 3).

The subset audit binds the AR tree by its room LISTINGS (names); this tool binds the BYTES. The
file set is derived from the largest registered split (``train_frac075`` ⊇ 050 ⊇ 025, nested by
construction and test-covered): for every target ``<scene>/<room>/S00{s}_R00{r}_hybrid_IR.wav`` the
loader reads the target wav, its metadata ``metadata/<scene>/<room>/S00{s}_R00{r}.json`` (AR_md's own
name rebuild) and the receiver's depth panorama ``depth_map/<scene>/<room>/{r}.npy``; every context
it draws is another in-split target of the same receiver (``restrict_to_split``), so it is covered.

    python exp38_content_manifest.py --build  <out.txt.gz> [--dataset-root AcousticRooms] [--threads 16]
    python exp38_content_manifest.py --verify <manifest.txt.gz> [--dataset-root AcousticRooms]

The manifest is one line per (sub-tree, scene, room) group -- ``<group sha256>  <files>  <bytes>
<group>`` with the group sha256 over ``<relpath>\0<file sha256>\0<bytes>\n`` of its sorted files --
sorted, gzip with mtime 0 (byte-stable; ~730 lines instead of 445k); its identity is the sha256 of the
UNCOMPRESSED text, pinned in exp38_registry.py (``AR_CONTENT_SHA256``). ``--verify`` re-reads and
re-hashes EVERY file of the split-implied set (parallel reads; ~15 min over CIFS) and compares the
groups. Read-only on the dataset.

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp38_registry as reg  # noqa: E402

CONTENT_SPLIT_TAG = "075"


def content_files(repo=None, split_rel=None):
    """Sorted dataset-relative paths the registered runs can read."""
    split = json.load(open(reg._abs(split_rel or reg.FRACS[CONTENT_SPLIT_TAG]["split"], repo)))
    out = set()
    for scene, rooms in split.items():
        for room, names in rooms.items():
            for name in names:
                s, r = int(name.split("_")[0][1:]), int(name.split("_")[1][1:])
                out.add(f"single_channel_ir_1/{scene}/{room}/{name}")
                out.add(f"metadata/{scene}/{room}/S00{s}_R00{r}.json")
                out.add(f"depth_map/{scene}/{room}/{r}.npy")
    return sorted(out)


def _hash_one(root, rel):
    h = hashlib.sha256()
    n = 0
    with open(os.path.join(root, rel), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return rel, h.hexdigest(), n


def hash_files(root, files, threads=16):
    """{rel: (sha256, bytes)}; a missing / unreadable file is reported as (None, None)."""
    out = {}
    with cf.ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(_hash_one, root, rel): rel for rel in files}
        for f in cf.as_completed(futs):
            try:
                rel, digest, n = f.result()
                out[rel] = (digest, n)
            except OSError:
                out[futs[f]] = (None, None)
    return out


def group_of(rel):
    return "/".join(rel.split("/")[:3])


def groups(hashes):
    """{group: (sha256, n files, bytes)} of per-file {rel: (sha256, bytes)}."""
    acc = {}
    for rel in sorted(hashes):
        d, n = hashes[rel]
        g = acc.setdefault(group_of(rel), [hashlib.sha256(), 0, 0])
        g[0].update(f"{rel}\0{d}\0{n}\n".encode())
        g[1] += 1
        g[2] += n
    return {k: (v[0].hexdigest(), v[1], v[2]) for k, v in acc.items()}


def manifest_text(hashes):
    return "".join(f"{d}  {n}  {b}  {g}\n" for g, (d, n, b) in sorted(groups(hashes).items()))


def write_manifest(path, text):
    if os.path.exists(path):
        raise reg.GateError(f"{path} exists")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(text.encode())
    reg.atomic_write_bytes(path, buf.getvalue())
    return hashlib.sha256(text.encode()).hexdigest()


def read_manifest(path, want_sha=None):
    text = gzip.decompress(open(path, "rb").read()).decode()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if want_sha is not None and digest != want_sha:
        raise reg.GateError(f"{path}: content manifest sha {digest} != pinned {want_sha}")
    rows = {}
    for line in text.splitlines():
        d, n, b, g = line.split("  ", 3)
        rows[g] = (d, int(n), int(b))
    return rows, digest


def verify(manifest, dataset_root="AcousticRooms", repo=None, want_sha=None, threads=16):
    """Problems (empty = PASS): the pinned manifest, exactly the split-implied file set, and every
    file's current bytes."""
    repo = repo or reg.REPO
    root = dataset_root if os.path.isabs(dataset_root) else os.path.join(repo, dataset_root)
    try:
        rows, digest = read_manifest(manifest, want_sha)
    except (OSError, ValueError, reg.GateError) as err:
        return [f"content manifest: {err}"]
    p = []
    want = content_files(repo)
    if sorted(rows) != sorted({group_of(r) for r in want}) or sum(v[1] for v in rows.values()) != len(want):
        p.append(f"manifest groups ({len(rows)}, {sum(v[1] for v in rows.values())} files) != the split's "
                 f"({len({group_of(r) for r in want})}, {len(want)} files)")
        return p
    now = hash_files(root, want, threads)
    missing = [rel for rel, (d, _) in now.items() if d is None]
    if missing:
        p.append(f"{len(missing)} file(s) unreadable, e.g. {missing[:3]}")
        return p
    cur = groups(now)
    bad = sorted(g for g in rows if cur.get(g) != rows[g])
    if bad:
        p.append(f"{len(bad)} group(s) differ from the content manifest, e.g. {bad[:3]}")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build")
    g.add_argument("--verify")
    ap.add_argument("--dataset-root", default="AcousticRooms")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    root = a.dataset_root if os.path.isabs(a.dataset_root) else os.path.join(reg.REPO, a.dataset_root)
    if a.verify:
        probs = verify(a.verify, a.dataset_root, want_sha=reg.AR_CONTENT_SHA256, threads=a.threads)
        print("AR CONTENT VERIFY " + ("PASS" if not probs else "FAIL: " + " | ".join(probs)))
        return 0 if not probs else 1
    files = content_files()
    hashes = hash_files(root, files, a.threads)
    missing = [rel for rel, (d, _) in hashes.items() if d is None]
    if missing:
        print(f"AR CONTENT BUILD FAILED: {len(missing)} unreadable, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    digest = write_manifest(a.build, manifest_text(hashes))
    total = sum(n for _, n in hashes.values())
    print(f"AR CONTENT MANIFEST {a.build}: {len(files)} files, {total} bytes, sha256(text) {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
