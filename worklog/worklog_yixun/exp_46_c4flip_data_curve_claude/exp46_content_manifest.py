#!/usr/bin/env python3
"""exp_46 -- content binding of every AcousticRooms file f100 can read that exp_38's f075 content manifest does not
cover (plan_c4flip_data_curve.md r3 §3.2 "Data binding"; Codex plan r1 finding 2, r2 findings 2 + 3).

Derived from exp_38's exp38_content_manifest.py, which is reused READ-ONLY from exp_38's folder: its file-set rule
(``content_files``: per target the waveform, its metadata file and the receiver's depth panorama), its group / text /
gzip format and its reader. The delta is the full split's (data/AR/train.json, f100) content files minus exp_38's
f075 content files -- by exp_38's verified full_split audit exactly ``F100_DELTA_FILES`` = 145,494 files (72,747 target
waveforms + their 72,747 metadata files; every depth panorama f100 needs is already in the f075 manifest). The count
is a registered constant: a different delta refuses. The union (exp_38's f075 manifest + this delta) covers every file
f100 can read; exp_38's manifest is verified by exp_38's own tool, this one by this tool.

    python exp46_content_manifest.py --build  <out.txt.gz> [--dataset-root AcousticRooms]      # L2, once
    python exp46_content_manifest.py --verify <manifest.txt.gz> [--dataset-root AcousticRooms]  # L2 and in the f100 FULL

Every read is SEQUENTIAL -- one file at a time, no thread pool: no parallel fan-out on /n/fs (plan r3 §3.2, §5 L2).
``verify_record`` is what the supervisor's f100 FULL branch runs after the child and before the end audit (plan r3
§3.2, Codex plan r2 finding 2): its sealed record is retained in receipts/ and bound into the training receipt. The
caller's ``check`` is called between files, so a cancelled job (the supervisor's signal check) stops the verification
and leaves the run unsealed. Read-only on the dataset.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

# Codex code r1 finding 1 (HIGH): exp_38's tools are imported from exp_38's own folder below; the plain standalone
# invocation (no -B) must write no bytecode there -- nor anywhere: no import of this process caches bytecode
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_registry as reg  # noqa: E402

sys.path.insert(0, reg._abs(reg.EXP38_DIR))                  # exp_38's tool, read-only (its folder, its bytes)
import exp38_content_manifest as cm38  # noqa: E402

RECORD_KEYS = frozenset({"schema", "kind", "verdict", "problems", "manifest", "manifest_sha256", "files", "bytes",
                         "dataset_root", "host", "started_at", "finished_at"})


def full_files(repo=None):
    """Sorted dataset-relative paths f100 can read: exp_38's content-file rule over the full split."""
    return cm38.content_files(repo, split_rel=reg.FRACS[reg.FULL_TAG]["split"])


def delta_files(repo=None):
    """f100's files minus exp_38's f075 content files (its manifest's file set), refused unless the count is the
    registered F100_DELTA_FILES and the delta holds no depth panorama (the f075 manifest covers every receiver)."""
    f075 = set(cm38.content_files(repo))                        # exp_38's CONTENT_SPLIT_TAG (075) rule, unchanged
    full = set(full_files(repo))
    if not f075 <= full:
        raise reg.GateError("exp_38's f075 content files are not a subset of the full split's")
    delta = sorted(full - f075)
    if len(delta) != reg.F100_DELTA_FILES or any(rel.startswith("depth_map/") for rel in delta):
        raise reg.GateError(f"the f100 delta has {len(delta)} files (registered {reg.F100_DELTA_FILES}) or depth files")
    return delta


def hash_sequential(root, files, check=None):
    """{rel: (sha256, bytes)}, one file at a time; a missing / unreadable file is (None, None). ``check()`` runs before
    every file (the supervisor's signal check)."""
    out = {}
    for rel in files:
        if check is not None:
            check()
        try:
            _, digest, n = cm38._hash_one(root, rel)
            out[rel] = (digest, n)
        except OSError:
            out[rel] = (None, None)
    return out


def _root(dataset_root, repo):
    return dataset_root if os.path.isabs(dataset_root) else os.path.join(repo, dataset_root)


def verify(manifest, dataset_root="AcousticRooms", repo=None, want_sha=None, check=None):
    """Problems (empty = PASS): the manifest at its pinned sha, exactly the registered delta's groups and file count,
    and every delta file's CURRENT bytes (re-read sequentially)."""
    repo = repo or reg.REPO
    if want_sha is None:
        return ["the f100 delta manifest is not pinned in the registry (F100_DELTA_SHA256; built at L2)"]
    try:
        rows, _ = cm38.read_manifest(manifest, want_sha)
    except (OSError, ValueError, EOFError, reg.GateError, cm38.reg.GateError) as err:
        return [f"f100 delta manifest: {err}"]
    try:
        want = delta_files(repo)
    except (OSError, ValueError, reg.GateError) as err:
        return [f"f100 delta file set: {err}"]
    if sorted(rows) != sorted({cm38.group_of(r) for r in want}) or sum(v[1] for v in rows.values()) != len(want):
        return [f"manifest groups ({len(rows)}, {sum(v[1] for v in rows.values())} files) != the delta's "
                f"({len({cm38.group_of(r) for r in want})}, {len(want)} files)"]
    now = hash_sequential(_root(dataset_root, repo), want, check)
    missing = [rel for rel, (d, _) in now.items() if d is None]
    if missing:
        return [f"{len(missing)} f100 delta file(s) unreadable, e.g. {missing[:3]}"]
    cur = cm38.groups(now)
    bad = sorted(g for g in rows if cur.get(g) != rows[g])
    return [f"{len(bad)} group(s) differ from the f100 delta manifest, e.g. {bad[:3]}"] if bad else []


def verify_record(manifest, dataset_root="AcousticRooms", repo=None, want_sha=None, check=None):
    """The verification as a record (RECORD_KEYS): verdict PASS / FAIL, its problems, the manifest and its pinned text
    sha256, the delta's file count and bytes, the dataset root, host and times. An exception or a stop of ``check``
    propagates (no record: the run stays unsealed)."""
    repo = repo or reg.REPO
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    problems = verify(manifest, dataset_root, repo, want_sha, check)
    rows = {}
    if not problems:
        rows, _ = cm38.read_manifest(manifest, want_sha)
    return {"schema": 1, "kind": "exp46_f100_delta_verification", "verdict": "FAIL" if problems else "PASS",
            "problems": problems, "manifest": os.path.relpath(os.path.realpath(manifest), os.path.realpath(repo)),
            "manifest_sha256": want_sha, "files": sum(v[1] for v in rows.values()),
            "bytes": sum(v[2] for v in rows.values()), "dataset_root": os.path.realpath(_root(dataset_root, repo)),
            "host": socket.gethostname(), "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def record_problems(rec):
    """Problems of a sealed verification record as its consumers admit it (the training receipt's writer and every
    admission): the exact schema, verdict PASS with no problem, the registered manifest at its pinned sha, the
    registered file count."""
    if not isinstance(rec, dict) or set(rec) != RECORD_KEYS:
        return ["f100 delta verification record: not the registered schema"]
    p = []
    if rec["verdict"] != "PASS" or rec["problems"] or rec["kind"] != "exp46_f100_delta_verification" or rec["schema"] != 1:
        p.append(f"f100 delta verification verdict {rec['verdict']!r}: {rec['problems'][:2]}")
    if rec["manifest"] != reg.F100_DELTA_MANIFEST or rec["manifest_sha256"] != reg.F100_DELTA_SHA256 \
            or reg.F100_DELTA_SHA256 is None:
        p.append("f100 delta verification record: not the registered, pinned manifest")
    if rec["files"] != reg.F100_DELTA_FILES:
        p.append(f"f100 delta verification record: {rec['files']} files != {reg.F100_DELTA_FILES}")
    return p


def build(out, dataset_root="AcousticRooms", repo=None):
    """L2: hash every delta file (sequentially) and write the manifest (never over an existing file)."""
    repo = repo or reg.REPO
    if os.path.lexists(out):
        raise reg.GateError(f"{out} exists (the delta manifest is built once)")
    files = delta_files(repo)
    hashes = hash_sequential(_root(dataset_root, repo), files)
    missing = [rel for rel, (d, _) in hashes.items() if d is None]
    if missing:
        raise reg.GateError(f"{len(missing)} f100 delta file(s) unreadable, e.g. {missing[:3]}")
    digest = cm38.write_manifest(out, cm38.manifest_text(hashes))
    return {"manifest": out, "files": len(files), "bytes": sum(n for _, n in hashes.values()), "sha256_text": digest}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build")
    g.add_argument("--verify")
    ap.add_argument("--dataset-root", default="AcousticRooms")
    a = ap.parse_args(argv)
    try:
        if a.verify:
            rec = verify_record(a.verify, a.dataset_root, want_sha=reg.F100_DELTA_SHA256)
            print(json.dumps(rec, sort_keys=True))
            print("F100 DELTA VERIFY " + ("PASS" if rec["verdict"] == "PASS" else "FAIL: " + " | ".join(rec["problems"])))
            return 0 if rec["verdict"] == "PASS" else 1
        out = build(a.build, a.dataset_root)
        print(f"F100 DELTA MANIFEST {out['manifest']}: {out['files']} files, {out['bytes']} bytes, "
              f"sha256(text) {out['sha256_text']}")
        return 0
    except (reg.GateError, cm38.reg.GateError, OSError, ValueError) as err:
        print(f"EXP46 CONTENT MANIFEST REFUSED: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
