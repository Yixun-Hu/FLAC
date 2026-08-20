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
COMMIT_MARKER_NAME = "raf_publish_commit.json"
SUPERSEDED_SUFFIX = ".superseded"
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

    def __init__(self, dest_root, generation=None):
        self.generation = generation
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

    def validate(self, expected=None, validate_json=False):
        """Everything checkable BEFORE the destination is touched.

        Separated from the swap so a transaction can validate every root first and
        only then start replacing files anywhere.
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
        return staged

    def invalidate(self):
        """Rename any existing attestation aside BEFORE replacement begins (S3).

        While files are being swapped the tree holds a mixture of generations, so
        the previous manifest must not keep claiming it is published.
        """
        for name in (MANIFEST_NAME, COMMIT_MARKER_NAME):
            path = os.path.join(self.dest_root, name)
            if os.path.exists(path):
                os.replace(path, path + SUPERSEDED_SUFFIX)

    def swap_in(self, staged=None):
        """Replace the destination files from the staged tree."""
        staged = self.staged_files() if staged is None else staged
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
        return files

    def write_manifest(self, files):
        manifest = {
            "committed_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "root": self.dest_root,
            "generation": self.generation,
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
        return manifest

    def commit(self, expected=None, validate_json=False):
        """Single-root publish: validate, invalidate, swap, attest.

        Multi-root publishes go through ``PublishTransaction`` instead, which
        interleaves those phases across roots so one commit marker covers them all.
        """
        staged = self.validate(expected=expected, validate_json=validate_json)
        self.invalidate()
        files = self.swap_in(staged)
        manifest = self.write_manifest(files)
        self.cleanup()
        return manifest

    def cleanup(self):
        shutil.rmtree(self.staging_dir, ignore_errors=True)


class PublishTransaction:
    """One generation over many roots, attested by a single commit marker (S3).

    Phases, in order: stage everything -> validate every root -> invalidate every
    root's old attestation -> swap every root in -> write every root's manifest ->
    write ONE commit marker, last. A failure anywhere after invalidation leaves no
    valid marker, and a reader that finds no valid marker treats the tree as
    UNPUBLISHED rather than as the previous generation.
    """

    def __init__(self, marker_root, generation=None):
        self.marker_root = os.path.abspath(marker_root)
        self.generation = generation or uuid.uuid4().hex
        self.roots = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def stage(self, dest_root):
        staged = StagedPublish(dest_root, generation=self.generation)
        self.roots.append(staged)
        return staged

    def commit(self, expectations=None, validate_json=True, extra=None):
        expectations = expectations or {}
        plans = []
        for root in self.roots:
            expected = expectations.get(root.dest_root)
            plans.append((root, root.validate(expected=expected,
                                              validate_json=validate_json)))

        marker_path = os.path.join(self.marker_root, COMMIT_MARKER_NAME)
        if os.path.exists(marker_path):
            os.replace(marker_path, marker_path + SUPERSEDED_SUFFIX)
        for root, _ in plans:
            root.invalidate()

        manifests = {}
        for root, staged in plans:
            manifests[root.dest_root] = root.write_manifest(root.swap_in(staged))

        marker = {
            "generation": self.generation,
            "committed_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "marker_root": self.marker_root,
            "roots": {
                dest: {
                    "manifest": MANIFEST_NAME,
                    "manifest_sha256": sha256_file(os.path.join(dest, MANIFEST_NAME)),
                    "n_files": manifest["n_files"],
                }
                for dest, manifest in manifests.items()
            },
        }
        if extra:
            marker.update(extra)
        os.makedirs(self.marker_root, exist_ok=True)
        tmp = marker_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(marker, f, indent=4, allow_nan=False)
        os.replace(tmp, marker_path)          # LAST: this is the attestation

        self.committed = True
        self.cleanup()
        return marker

    def cleanup(self):
        for root in self.roots:
            root.cleanup()


def verify_publication(marker_root):
    """Read a published tree the way a consumer must: marker first.

    Returns ``{"published": bool, "reason": str, "generation": ..., "roots": {...}}``.
    No valid marker means UNPUBLISHED -- not "the previous generation" -- because a
    tree interrupted mid-swap holds a mixture of generations.
    """
    marker_path = os.path.join(os.path.abspath(marker_root), COMMIT_MARKER_NAME)
    if not os.path.isfile(marker_path):
        return {"published": False, "reason": f"no commit marker at {marker_path}",
                "generation": None, "roots": {}}
    with open(marker_path) as f:
        marker = json.load(f)

    roots, reasons = {}, []
    for dest, entry in sorted(marker.get("roots", {}).items()):
        manifest_path = os.path.join(dest, entry["manifest"])
        if not os.path.isfile(manifest_path):
            reasons.append(f"{dest}: manifest missing")
            roots[dest] = {"missing": [entry["manifest"]], "mismatched": []}
            continue
        if sha256_file(manifest_path) != entry["manifest_sha256"]:
            reasons.append(f"{dest}: manifest does not match the marker")
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("generation") != marker["generation"]:
            reasons.append(
                f"{dest}: manifest generation {manifest.get('generation')} != marker "
                f"generation {marker['generation']}")
        result = verify_manifest(dest)
        roots[dest] = result
        if result["missing"] or result["mismatched"]:
            reasons.append(f"{dest}: {len(result['missing'])} missing, "
                           f"{len(result['mismatched'])} mismatched")

    return {
        "published": not reasons,
        "reason": "; ".join(reasons),
        "generation": marker["generation"],
        "roots": roots,
    }


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
