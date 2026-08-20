"""Atomic staged publishing for RAF artifacts (exp_19 r2, Codex R7).

A RAF publish is a SET of artifacts that only mean anything together: the WAVs,
the runtime metadata that indexes them, the split manifests that name them, and
the audits that describe them. Writing them straight to their destinations one
after another means any late failure -- a bad file, a NAS interruption, a full
disk -- leaves a plausible mixture of old and new, and a rerun can overwrite audio
before replacing the split that described it.

So: stage the whole publish under a hidden sibling directory of the destination
(same directory => same filesystem => ``os.replace`` is atomic), validate it
there, swap every file in, and write the hash manifest LAST. The manifest's
presence is the evidence that a publish committed, and ``verify_manifest`` can
re-check the published tree at any later point.
"""
import datetime
import hashlib
import json
import os
import shutil
import uuid

MANIFEST_NAME = "raf_publish_manifest.json"
STAGING_PREFIX = ".{name}.staging-"


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class StagedPublish:
    """Context manager that stages a publish and commits it atomically.

    Usage::

        with StagedPublish(dest_root) as staged:
            write_something(staged.path("sub", "file.json"))
            staged.commit(expected=["sub/file.json"], validate_json=True)

    Leaving the block without a successful ``commit`` -- by exception or by
    omission -- removes the staging directory and leaves the destination exactly
    as it was.
    """

    def __init__(self, dest_root):
        self.dest_root = os.path.abspath(dest_root)
        parent = os.path.dirname(self.dest_root)
        os.makedirs(parent, exist_ok=True)
        prefix = STAGING_PREFIX.format(name=os.path.basename(self.dest_root))
        self.staging_dir = os.path.join(parent, f"{prefix}{uuid.uuid4().hex[:12]}")
        os.makedirs(self.staging_dir)
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def path(self, *parts):
        """Staged path for a destination-relative artifact; parents are created."""
        if not parts:
            raise ValueError("a staged artifact needs a relative path")
        target = os.path.join(self.staging_dir, *parts)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        return target

    def staged_files(self):
        """Destination-relative paths of everything staged so far, sorted."""
        out = []
        for root, _dirs, files in os.walk(self.staging_dir):
            for name in files:
                full = os.path.join(root, name)
                out.append(os.path.relpath(full, self.staging_dir).replace(os.sep, "/"))
        return sorted(out)

    def commit(self, expected=None, validate_json=False):
        """Validate the staged tree, swap it in, then write the hash manifest.

        ``expected`` names artifacts that MUST be present (a publish missing one of
        its parts is not a publish); ``validate_json`` re-parses every staged
        ``.json`` so a truncated write is caught before anything is visible.
        """
        staged = self.staged_files()
        if expected:
            missing = [name for name in expected if name not in set(staged)]
            if missing:
                raise ValueError(
                    f"refusing to publish an incomplete set: {missing} were never staged")
        if validate_json:
            for name in staged:
                if name.endswith(".json"):
                    with open(os.path.join(self.staging_dir, name)) as f:
                        try:
                            json.load(f)
                        except ValueError as e:
                            raise ValueError(
                                f"refusing to publish {name}: it is not valid JSON ({e})")

        files = {}
        for name in staged:
            source = os.path.join(self.staging_dir, name)
            files[name] = {"sha256": sha256_file(source), "bytes": os.path.getsize(source)}

        os.makedirs(self.dest_root, exist_ok=True)
        for name in staged:
            source = os.path.join(self.staging_dir, name)
            target = os.path.join(self.dest_root, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(source, target)   # atomic: same filesystem by construction

        manifest = {
            "committed_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "root": self.dest_root,
            "n_files": len(files),
            "files": files,
        }
        # Written last, and itself via replace, so a reader either sees the whole
        # manifest of a completed publish or no manifest at all.
        tmp_manifest = os.path.join(self.staging_dir, MANIFEST_NAME + ".tmp")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=4, allow_nan=False)
        os.replace(tmp_manifest, os.path.join(self.dest_root, MANIFEST_NAME))

        self.committed = True
        self.cleanup()
        return manifest

    def cleanup(self):
        shutil.rmtree(self.staging_dir, ignore_errors=True)


def verify_manifest(dest_root):
    """Re-check a published tree against its manifest.

    Returns ``{"missing": [...], "mismatched": [...]}``; both empty means the
    published artifacts are byte-identical to what was committed.
    """
    manifest_path = os.path.join(dest_root, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"no publish manifest at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    missing, mismatched = [], []
    for name, entry in sorted(manifest["files"].items()):
        path = os.path.join(dest_root, name)
        if not os.path.isfile(path):
            missing.append(name)
        elif sha256_file(path) != entry["sha256"]:
            mismatched.append(name)
    return {"missing": missing, "mismatched": mismatched}
