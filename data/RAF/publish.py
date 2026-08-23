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
# Markers are NAMESPACED per transaction kind (T4). Rendering stores its marker at
# the runtime root, which a later preparation transaction also publishes into; a
# single shared name meant a prepare rerun renamed the depth attestation aside and
# orphaned depth evidence it never regenerates.
# Flavor-scoped kinds: one corpus tree can carry a Mapping-H publication and a
# Mapping-A one simultaneously, and neither may disturb the other (exp_19 r4-T4:
# a shared marker name let a prepare rerun orphan the depth attestation).
MARKER_KINDS = ("prepare", "depth", "mappingA_prepare", "mappingA_depth")
FLAVOR_KINDS = {"mappingH": ("prepare", "depth"),
                "mappingA": ("mappingA_prepare", "mappingA_depth")}
SUPERSEDED_SUFFIX = ".superseded"
STAGING_PREFIX = ".{name}.staging-"


def marker_name(kind):
    """Commit-marker filename for one transaction kind."""
    if kind not in MARKER_KINDS:
        raise ValueError(f"unknown transaction kind {kind!r}, expected one of {MARKER_KINDS}")
    return f"raf_publish_commit.{kind}.json"


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
        # Only THIS root's manifest is invalidated. Commit markers are namespaced
        # per kind and are handled by the transaction that owns them, so a prepare
        # rerun can no longer rename the depth attestation aside (T4).
        path = os.path.join(self.dest_root, MANIFEST_NAME)
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

    def __init__(self, marker_root, kind="prepare", generation=None):
        self.marker_root = os.path.abspath(marker_root)
        self.kind = kind
        self.marker_name = marker_name(kind)
        self.generation = generation or uuid.uuid4().hex
        self.roots = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def stage(self, dest_root):
        dest = os.path.abspath(dest_root)
        if any(root.dest_root == dest for root in self.roots):
            raise ValueError(f"duplicate root in one transaction: {dest}")
        staged = StagedPublish(dest, generation=self.generation)
        self.roots.append(staged)
        return staged

    def commit(self, expectations=None, validate_json=True, extra=None):
        if not self.roots:
            raise ValueError("refusing to commit a transaction with no roots")
        expectations = expectations or {}
        plans = []
        for root in self.roots:
            expected = expectations.get(root.dest_root)
            plans.append((root, root.validate(expected=expected,
                                              validate_json=validate_json)))

        marker_path = os.path.join(self.marker_root, self.marker_name)
        if os.path.exists(marker_path):
            os.replace(marker_path, marker_path + SUPERSEDED_SUFFIX)
        for root, _ in plans:
            root.invalidate()

        manifests = {}
        for root, staged in plans:
            manifests[root.dest_root] = root.write_manifest(root.swap_in(staged))

        marker = {
            "generation": self.generation,
            "kind": self.kind,
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


def verify_publication(marker_root, kind="prepare", expected_roots=None):
    """Read a published tree the way a consumer must: marker first.

    Returns ``{"published": bool, "reason": str, "generation": ..., "roots": {...}}``.
    No valid marker means UNPUBLISHED -- not "the previous generation" -- because a
    tree interrupted mid-swap holds a mixture of generations. An EMPTY root set is
    likewise not a publication, and ``expected_roots`` pins the exact set a complete
    publication must cover (T4), so a Furnished-only depth run cannot read as one.
    """
    marker_path = os.path.join(os.path.abspath(marker_root), marker_name(kind))
    if not os.path.isfile(marker_path):
        return {"published": False, "reason": f"no commit marker at {marker_path}",
                "generation": None, "roots": {}}
    with open(marker_path) as f:
        marker = json.load(f)

    roots, reasons = {}, []
    declared = marker.get("roots") or {}
    if not declared:
        reasons.append("marker declares no roots: an empty publication is not one")
    if marker.get("kind", kind) != kind:
        reasons.append(f"marker kind {marker.get('kind')!r} != requested {kind!r}")
    if expected_roots is not None:
        resolved = [os.path.abspath(r) for r in expected_roots]
        if not resolved:
            reasons.append("an empty expected-root list is not a publication")
        if len(set(resolved)) != len(resolved):
            # reducing to a set would hide the caller's own duplicate expectation
            reasons.append(f"duplicate expected roots: {sorted(resolved)}")
        expected = set(resolved)
        actual = {os.path.abspath(r) for r in declared}
        if expected != actual:
            reasons.append(
                f"expected roots {sorted(expected)} but the marker covers "
                f"{sorted(actual)}")
    for dest, entry in sorted(declared.items()):
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
        "kind": marker.get("kind"),
        "marker": marker,
        "roots": roots,
    }


CANONICAL_ROOMS = ("EmptyRoom", "FurnishedRoom")

# Sentinel for an identity field whose exact value is only knowable after the
# canonical run: the consumer requires the SHAPE (64 lowercase hex characters).
SHA256_SHAPE = "<sha256>"

# The registered per-kind parameter identities (Amendment 7). They live HERE, in
# the verifier, so the consumer checks a marker against the same dictionary the
# producer was validated against -- rather than against the producer's own claim
# that it was canonical (r5 finding 3).
CANONICAL_PREPARE_PARAMS = {
    "rooms": list(CANONICAL_ROOMS),
    "n_groups": 16,
    "n_val_groups": 4,
    "n_train": 12,
    "n_diagnostic_groups": 1,
    "seed": 0,
    "full_crosscheck": True,
    "allow_nonuniform": True,
    # Amendment 9: the ceiling is the registered RULE input; the scalar is DERIVED
    # from the corpus and equals 3.0 on it, which is what a canonical publication
    # must reproduce.
    "amplitude_ceiling": 0.75,
    "amplitude_scalar": 3.0,
    # F3: the derivation provenance is part of the identity -- formula version, id
    # COUNT ((16 train/test + 1 diagnostic) x 12 x 2 rooms = 408), and the id-set
    # hash, PINNED from the canonical generation 46a43f4ce82b (scalar x3 =
    # min(support 5.0, clamp 3.0)). A consumer can now verify that the registered
    # trained-ID union is what produced the scalar, not merely that some union did.
    "amplitude_formula_version": "9.2",
    "amplitude_derivation_ids": 408,
    "amplitude_derivation_sha256": "8a740feef8f430dbc2e65d8f3d5eefa3d6b191c00c615ff758163c7428eef00d",
}
CANONICAL_RENDER_PARAMS = {
    "rooms": list(CANONICAL_ROOMS),
    "img_h": 256,
    "img_w": 512,
    "floor_tol": 0.15,
    "max_miss_rate": 0.0025,
    "rx_sightline_receivers": 8,
    # Amendment 9: recorded diagnostic in BOTH rooms (environmental obstruction
    # proven by a full yaw sweep), never a gate.
    "rx_sightline_policy": "recorded",
    # F4: the fingerprint of the on-disk HAA reference the scale gate ran against
    # (4 processed base rooms, 0.5038-11.5523 m).
    "haa_reference_sha256":
        "1d59babdbc1b0b6075b32216c864588acf5516454a92a4a6af946bd832656eb3",
}
# --------------------------------------------------------------------------- #
# Mapping-A registered identities (exp_21 plan section 4)
# --------------------------------------------------------------------------- #
CANONICAL_MAPPINGA_PREPARE_PARAMS = {
    "rooms": list(CANONICAL_ROOMS),
    "n_placements": 16,
    "k": 8,
    "n_items": 1152,
    # the correspondence algorithm IS the scientific claim, so its version and
    # every tolerance are part of what a consumer verifies
    "match_algorithm_version": "mappingA-correspondence-2",
    "match_p95_m": 0.01,
    "match_max_m": 0.02,
    "match_ambiguity_margin": 3.0,
    "placement_cap_m": 0.05,
    # Amendment 4: Mapping A publishes its COMPLETE union at x2.0 -- the registered
    # formula re-derived over ITS union, whose clip clamp binds at 2.0401 because
    # two EmptyRoom union captures clip at x3. Mapping H stays at x3.0; the two
    # corpora are therefore at DIFFERENT levels, which is disclosed rather than
    # reconciled (see CROSS_MAPPING_SCALE_DISCLOSURE).
    "amplitude_scalar": 2.0,
    # the target the scalar is DERIVED against, kept distinct from the ceiling
    # every written file is CHECKED against (N3)
    "amplitude_derivation_target": 0.75,
    "clip_ceiling": 0.999,
    # N5: PINNED, not shaped. A canonical publication that could name any 64-hex
    # string was not identified by it at all. The audio-union digest was the one
    # value knowable only from the generation itself; it is now pinned from the
    # clean dry run of 2026-08-22 (generation 5fc096147bec, scalar x2 support-bound,
    # peaks 0.979/0.944, 21 near-silent context references over 19 p008 items), so
    # NOTHING in this identity is a placeholder any more.
    "correspondence_sha256":
        "d4d79b49677b7bc7541bf0e7dfe7f32f532912f6a268fca1d421b280c799663e",
    "audio_union_sha256":
        "b19eff06c7a13e0aaeafcdf95ad58f7f4f24bb3def794889102440437a220a21",
    "readback_record_sha256":
        "9288181be62bf8b4669880522fadaab18527facb2749837f768572069f4876c3",
}
CANONICAL_MAPPINGA_DEPTH_PARAMS = {
    "rooms": list(CANONICAL_ROOMS),
    "positions_from": "mappingA",
    "img_h": 256,
    "img_w": 512,
    "floor_tol": 0.15,
    # Amendment 4.3: LISTENER maps carry their own registered cap. Measured over
    # the full 1,152-position sweep -- worst 0.656%, 17 positions over the
    # source-calibrated 0.25%, all FurnishedRoom furniture-adjacent. The SOURCE
    # identity (CANONICAL_RENDER_PARAMS) keeps 0.0025: the published Mapping-H
    # marker is bound to it and must stay valid.
    "max_miss_rate": 0.007,
    "n_maps": 1152,
    "readback_record_sha256":
        "9288181be62bf8b4669880522fadaab18527facb2749837f768572069f4876c3",
}

CANONICAL_IDENTITIES = {"prepare": CANONICAL_PREPARE_PARAMS,
                        "depth": CANONICAL_RENDER_PARAMS,
                        "mappingA_prepare": CANONICAL_MAPPINGA_PREPARE_PARAMS,
                        "mappingA_depth": CANONICAL_MAPPINGA_DEPTH_PARAMS}


def unpinned_identity_keys(kind):
    """Registered digests that are still placeholders for this kind (N5).

    ``SHA256_SHAPE`` accepts any well-formed digest, which is right while a value
    is genuinely unknowable and wrong the moment it is knowable: a publication
    identified by "some sha256" is not identified. Producers ask this before
    claiming a canonical publication.
    """
    return sorted(key for key, value in CANONICAL_IDENTITIES[kind].items()
                  if value == SHA256_SHAPE)


def resolve_rooms(rooms, canonical=True):
    """Completeness is defined HERE, not by the caller (r5 finding 3).

    A canonical publication covers exactly the registered rooms; asking for a
    subset cannot make a one-room publication complete. Non-canonical runs may name
    their own rooms, but never an empty or duplicated list.
    """
    if canonical:
        if rooms is not None and tuple(rooms) != CANONICAL_ROOMS:
            raise ValueError(
                f"canonical publication covers exactly {list(CANONICAL_ROOMS)}, "
                f"not {list(rooms)}")
        return list(CANONICAL_ROOMS)
    if not rooms:
        raise ValueError("a publication with no rooms is not a publication")
    if len(set(rooms)) != len(rooms):
        raise ValueError(f"duplicate rooms in {list(rooms)}")
    return list(rooms)


def canonical_record_digest():
    """The pinned readback digest, imported lazily to keep the module graph acyclic."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readback_audit.py")
    spec = importlib.util.spec_from_file_location("raf_readback_for_publish", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CANONICAL_RECORD_SHA256


def marker_identity_problems(kind, marker):
    """Does this marker actually attest the registered identity for its kind?

    Checks the COMPLETE parameter payload -- exact key set and exact values -- plus
    the pinned readback digest and the absence of taint. A marker that merely says
    ``canonical_parameters: true`` proves nothing; the r5 oracle accepted markers
    with no parameters or digest at all.
    """
    expected = CANONICAL_IDENTITIES[kind]
    problems = []
    if marker.get("canonical") is not True:
        problems.append(f"{kind} marker does not declare canonical publication")
    if marker.get("taint"):
        problems.append(f"{kind} marker is tainted: {marker['taint']}")

    parameters = marker.get("parameters")
    if not isinstance(parameters, dict):
        problems.append(f"{kind} marker carries no parameter payload")
    else:
        missing = sorted(set(expected) - set(parameters))
        extra = sorted(set(parameters) - set(expected))
        if missing:
            problems.append(f"{kind} marker parameters are missing {missing}")
        if extra:
            problems.append(f"{kind} marker parameters carry unregistered {extra}")
        for key in sorted(set(expected) & set(parameters)):
            actual, want = parameters[key], expected[key]
            if want == SHA256_SHAPE:
                if not (isinstance(actual, str) and len(actual) == 64
                        and all(c in "0123456789abcdef" for c in actual)):
                    problems.append(
                        f"{kind} marker parameter {key}={actual!r} is not a sha256")
                continue
            if isinstance(want, list):
                actual, want = list(actual or []), list(want)
            if actual != want:
                problems.append(
                    f"{kind} marker parameter {key}={actual!r} != registered {want!r}")

    digest = ((marker.get("readback_record") or {}).get("sha256")
              if isinstance(marker.get("readback_record"), dict) else None)
    if digest != canonical_record_digest():
        problems.append(
            f"{kind} marker readback digest {digest} is not the pinned "
            f"{canonical_record_digest()}")
    return problems


# The runtime tree's pointer at the Mapping-H/Mapping-A split directory. Written
# by both prepare CLIs and covered by the runtime manifest.
PUBLICATION_POINTER = "raf_publication.json"


def pointer_identity_problems(output_dir, rooms, flavor, prepare_marker,
                              depth_marker):
    """The pointer a consumer reads must agree with the markers it points at (N5).

    A reader reaches the publication through the pointer: it names the split
    directory, the rooms and the parameters. Verifying the markers alone left the
    pointer free to disagree with them -- a stale pointer beside a fresh generation
    would send RAF_md/RAF_A_md at the right tree with the wrong identity.
    """
    path = os.path.join(output_dir, PUBLICATION_POINTER)
    if not os.path.isfile(path):
        return [f"{path} does not exist: the runtime tree names no publication"]
    try:
        with open(path) as f:
            pointer = json.load(f)
    except ValueError as e:
        return [f"{path} is not valid JSON ({e})"]

    problems = []
    if pointer.get("flavor", "mappingH") != flavor:
        problems.append(f"pointer declares flavor {pointer.get('flavor')!r}, not "
                        f"{flavor!r}")
    if list(pointer.get("rooms") or []) != list(rooms):
        problems.append(f"pointer names rooms {pointer.get('rooms')!r}, not "
                        f"{list(rooms)!r}")
    if pointer.get("canonical") is not True:
        problems.append("pointer does not declare a canonical publication")
    if pointer.get("taint"):
        problems.append(f"pointer is tainted: {pointer['taint']}")

    marker_params = (prepare_marker.get("parameters") or {})
    for key, value in sorted((pointer.get("parameters") or {}).items()):
        if key not in marker_params:
            problems.append(f"pointer parameter {key} is not in the prepare marker")
        elif marker_params[key] != value:
            problems.append(f"pointer parameter {key}={value!r} != marker "
                            f"{marker_params[key]!r}")

    digests = {
        "pointer": ((pointer.get("readback_record") or {}).get("sha256")
                    if isinstance(pointer.get("readback_record"), dict) else None),
        "prepare marker": ((prepare_marker.get("readback_record") or {}).get("sha256")
                           if isinstance(prepare_marker.get("readback_record"), dict)
                           else None),
        "depth marker": ((depth_marker.get("readback_record") or {}).get("sha256")
                         if isinstance(depth_marker.get("readback_record"), dict)
                         else None),
    }
    if len(set(digests.values())) != 1:
        problems.append("readback digests disagree across the publication: "
                        + ", ".join(f"{where}={digest}"
                                    for where, digest in sorted(digests.items())))
    return problems


def verify_combined_publication(split_dir, output_dir, rooms=None,
                                depth_subdir="depth_images", canonical=True,
                                flavor="mappingH"):
    """A publication is complete only when BOTH of its kinds attest their sets.

    Prepare covers the runtime tree and the split directory; depth covers one
    directory per room. Either alone is a partial state, and this is the check a
    consumer runs before treating the corpus as canonical (T4). ``flavor`` selects
    the kind pair, so Mapping H and Mapping A are verified independently of one
    another on the same tree.
    """
    if flavor not in FLAVOR_KINDS:
        raise ValueError(
            f"unknown publication flavor {flavor!r}, expected one of "
            f"{sorted(FLAVOR_KINDS)}")
    prepare_kind, depth_kind = FLAVOR_KINDS[flavor]
    rooms = resolve_rooms(rooms, canonical=canonical)
    prepare = verify_publication(
        split_dir, kind=prepare_kind,
        expected_roots=[os.path.abspath(output_dir), os.path.abspath(split_dir)])
    depth = verify_publication(
        output_dir, kind=depth_kind,
        expected_roots=[os.path.join(os.path.abspath(output_dir), room, depth_subdir)
                        for room in rooms])
    reasons = []
    if not prepare["published"]:
        reasons.append(f"{prepare_kind}: {prepare['reason']}")
    if not depth["published"]:
        reasons.append(f"{depth_kind}: {depth['reason']}")
    # r6 finding 3: verify the markers' COMPLETE parameter payload against the
    # registered identity for their kind, and the pinned digest -- never the
    # producer's own `canonical_parameters` boolean, which is just a claim.
    if canonical:
        for kind, report in ((prepare_kind, prepare), (depth_kind, depth)):
            reasons.extend(marker_identity_problems(kind, report.get("marker") or {}))
        reasons.extend(pointer_identity_problems(
            output_dir, rooms, flavor, prepare.get("marker") or {},
            depth.get("marker") or {}))
    return {
        "published": not reasons,
        "reason": "; ".join(reasons),
        "rooms": rooms,
        "flavor": flavor,
        "kinds": {prepare_kind: prepare, depth_kind: depth},
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
