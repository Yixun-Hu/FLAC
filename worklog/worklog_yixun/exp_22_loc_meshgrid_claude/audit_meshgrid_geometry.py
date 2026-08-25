#!/usr/bin/env python
"""exp_22 G1 -- the 16-room geometry audit and post-G1 cost gate.

The consumer the inherited plan §1.3/§1.5 contracts for. It resolves exactly one
OBJ per room, joins the D1 context manifest in GLOBAL coordinates, audits every
query's candidate filters, computes BOTH oracle branches (full height and the
context-derived z-band), applies the pre-registered branch rule once for the
whole experiment, writes a hashed candidate manifest per room, and reports the
counts the cost gate is decided on.

Everything is fail-closed: a room with no unambiguous mesh, a context that does
not resolve to a metadata anchor, an empty candidate set, a non-finite oracle or
a query count that is not the registered one blocks the audit rather than
producing a partial manifest.

Usage:
    python audit_meshgrid_geometry.py --context-manifest <D1 manifest.json> \\
        --mesh-root <.../room_mesh_obj_format> --metadata-root AcousticRooms/metadata \\
        --out-dir <dir> [--expected-queries 5337]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.localization import meshgrid_geometry as mg          # noqa: E402
from src.localization import meshgrid_queries as mq           # noqa: E402

#: The 16 rooms the audit MUST cover: the unseen split minus the room whose
#: official OBJ is absent. Pinned as a literal so a missing room cannot be
#: silently absorbed into "the exclusion" (r2 re-review F3).
REQUIRED_ROOMS = (
    "Apartments/Apartments_idx_42",
    "Apartments/Apartments_idx_50",
    "Auditorium/Auditorium_idx_1",
    "Bathrooms/Bathrooms_idx_14",
    "Bathrooms/Bathrooms_idx_18",
    "Bedrooms/Bedrooms_idx_18",
    "Bedrooms/Bedrooms_idx_33",
    "Cafe/Cafe_idx_1",
    "LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25",
    "LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30",
    "MeetingRoom/MeetingRoom_idx_20",
    "MeetingRoom/MeetingRoom_idx_32",
    "Office/Office_idx_10",
    "Office/Office_idx_11",
    "Restaurants/Restaurants_idx_22",
    "Restaurants/Restaurants_idx_24",
)

#: how far a recovered context coordinate may sit from a metadata anchor.
ANCHOR_TOLERANCE = 1e-3
#: one conditioner call per unique (receiver, candidate) pair, per the §1.5 cache.
CONDITIONER_CALLS_PER_PAIR = 1
#: bytes a stored candidate costs in the manifest (3 float64 + index bookkeeping).
BYTES_PER_CANDIDATE = 32


def resolve_room_meshes(room_ids, mesh_root):
    """Exactly ONE OBJ per room, or a refusal naming the room."""
    resolved = {}
    for room_id in room_ids:
        scene, scene_id = room_id.split("/")
        directory = os.path.join(mesh_root, scene)
        matches = []
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                stem, extension = os.path.splitext(name)
                if stem == scene_id and extension.lower() == ".obj":
                    matches.append(os.path.join(directory, name))
        if not matches:
            raise ValueError(f"{room_id}: no OBJ found under {directory!r}; a room without "
                             "an unambiguous mesh blocks the audit (inherited plan §1.3)")
        if len(matches) > 1:
            raise ValueError(f"{room_id}: {len(matches)} OBJs resolve to the room "
                             f"({[os.path.basename(m) for m in matches]}); exactly one is "
                             "required")
        resolved[room_id] = matches[0]
    return resolved


def _parse_fingerprint(text):
    return np.array([float(part) for part in str(text).split(",")], dtype=np.float64)


def resolve_context_globals(record, receiver, anchors, tolerance=ANCHOR_TOLERANCE):
    """The drawn contexts in GLOBAL coordinates, matched to metadata anchors.

    D1 stores each context as its receiver-relative pose; the audit needs global
    coordinates. Adding the receiver recovers them under the released camera
    transform, and every recovered point must land on a real metadata source
    anchor -- which both identifies the context and proves the transform
    assumption on this room rather than assuming it.
    """
    receiver = np.asarray(receiver, dtype=np.float64).reshape(3)
    sources = np.asarray(anchors["sources"], dtype=np.float64).reshape(-1, 3)
    out = []
    for fingerprint in record["context_fingerprints"]:
        candidate = _parse_fingerprint(fingerprint) + receiver
        distances = np.linalg.norm(sources - candidate, axis=1)
        best = int(np.argmin(distances))
        if distances[best] > tolerance:
            raise ValueError(
                f"query {record['query_id']!r}: the context at relative "
                f"{fingerprint!r} recovers to {candidate.tolist()}, which is "
                f"{distances[best]:.4f} m from the nearest metadata source anchor "
                f"(tolerance {tolerance}); the receiver-relative transform does not hold "
                "for this room and the contexts cannot be placed")
        out.append(sources[best])
    return out


def _metadata_for(record, metadata_root):
    """``(receiver_global, target_global)`` from the query's own metadata JSON.

    The pair file is found by PARSED NUMERIC IDENTITY over the directory
    listing, never by reconstructing a name. The release writes these files as
    ``"S00" + str(src) + "_R00" + str(rec)``, so receiver 19 is stored as
    ``S007_R0019.json`` while receiver 8 is ``S007_R008.json`` -- one
    reconstructed format cannot match both, which is what aborted the first real
    G1 audit. ``candidates.find_pair_metadata`` already had these semantics and
    is reused rather than re-derived.
    """
    from src.localization.candidates import find_pair_metadata, parse_ir_filename

    scene, scene_id = record["room_id"].split("/")
    src_node, rec_node = parse_ir_filename(os.path.basename(record["relpath"]))
    room_dir = os.path.join(metadata_root, scene, scene_id)
    path = find_pair_metadata(room_dir, src_node, rec_node)
    if path is None:
        raise ValueError(f"query {record['query_id']!r}: no pair metadata for "
                         f"(S{src_node}, R{rec_node}) in {room_dir}")
    with open(path) as handle:
        payload = json.load(handle)
    return (np.asarray(payload["rec_loc"], dtype=np.float64),
            np.asarray(payload["src_loc"], dtype=np.float64))


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode()
                          + b"\n").hexdigest()


def _summary(values, allow_infinite=False):
    """An oracle distribution. Infinities are KEPT when the branch allows them:
    an empty z-band set is exactly what disqualifies that branch, and replacing
    it with the full-height value hides the disqualification (r2 re-review)."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("an oracle distribution must be nonempty")
    finite = np.isfinite(array)
    if not allow_infinite and not finite.all():
        raise ValueError("an oracle distribution must be finite")
    return {"n_queries": int(array.size), "median": float(np.median(array)),
            "mean": float(array.mean()), "max": float(array.max()),
            "n_infinite": int((~finite).sum()),
            "median_finite": (float(np.median(array[finite])) if finite.any() else None),
            "n_over_threshold": int((array > mg.ORACLE_THRESHOLD).sum()),
            "fraction_over_threshold": float((array > mg.ORACLE_THRESHOLD).mean())}


def coordinates_digest(array):
    """sha256 over the exact float64 coordinate bytes."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def candidate_set_key(receiver_id, indices):
    """``(receiver_id, sha256(index set))`` -- the true dedup key.

    Deduplicating ``(receiver, candidate_count)`` can over- or under-count: two
    different candidate sets of the same size collapse, and the same set counted
    twice under different receivers does not (r2 re-review F4).
    """
    array = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256(np.sort(array).tobytes()).hexdigest()
    return (str(receiver_id), digest)


class GateCounter:
    """The post-G1 gate's counts, over the UNION of each receiver's candidates.

    The conditioner cache is per (receiver, candidate): a receiver whose queries
    ask for ``{0,1,2}`` and ``{0,1,3}`` needs FOUR conditioner calls, not six.
    Summing distinct sets over-counted the shared candidates (r3 review F4); the
    number of distinct sets is kept as a separate, labelled diagnostic.
    """

    def __init__(self):
        self.unions = {}
        self.distinct = set()
        self.scored_pairs = 0

    def add(self, receiver_id, indices):
        array = np.asarray(indices, dtype=np.int64)
        self.scored_pairs += int(array.size)
        if array.size == 0:
            return
        self.unions.setdefault(str(receiver_id), set()).update(int(i) for i in array)
        self.distinct.add(candidate_set_key(receiver_id, array))

    def summary(self):
        union_total = int(sum(len(members) for members in self.unions.values()))
        return {
            "candidate_query_pairs": int(self.scored_pairs),
            # one conditioner call per (receiver, candidate) in the receiver's union
            "unique_receiver_candidate_pairs": union_total,
            "conditioner_calls_estimate": union_total,
            "distinct_candidate_sets": int(len(self.distinct)),
            "n_receivers": int(len(self.unions)),
            "artifact_bytes": int(self.scored_pairs * BYTES_PER_CANDIDATE),
        }


def verify_room_manifest(manifest_path, out_dir=None):
    """Re-accept a published room manifest from its own artifacts, fail-closed.

    Reconstructs BOTH branches from the sidecar npz and the recorded indices and
    re-derives every digest. The audit runs this as its last publish step, so
    nothing is published that the verifier would reject (r3 review F4).
    """
    out_dir = out_dir or os.path.dirname(os.path.abspath(manifest_path))
    reasons = []
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    npz_path = os.path.join(out_dir, manifest.get("coordinates_npz", ""))
    if not os.path.isfile(npz_path):
        return {"ok": False, "reasons": [f"the sidecar {npz_path!r} is missing"],
                "manifest": manifest_path}
    with np.load(npz_path) as data:
        base = np.asarray(data["base_candidates"], dtype=np.float64)

    if coordinates_digest(base) != manifest.get("base_candidates_sha256"):
        reasons.append(f"the npz base candidates do not match base_candidates_sha256 "
                       f"({coordinates_digest(base)[:12]}... vs "
                       f"{str(manifest.get('base_candidates_sha256'))[:12]}...)")
    if int(manifest.get("n_base_valid", -1)) != int(base.shape[0]):
        reasons.append(f"n_base_valid is {manifest.get('n_base_valid')} but the npz holds "
                       f"{base.shape[0]} candidates")

    branches = set()
    for query in manifest.get("queries", []):
        for branch, key, count_key in (("full_height", "candidate_indices", "n_candidates"),
                                       ("z_band", "candidate_indices_z_band",
                                        "n_candidates_z_band")):
            indices = np.asarray(query.get(key, []), dtype=np.int64)
            if indices.size != int(query.get(count_key, -1)):
                reasons.append(f"{query['query_id']}: {branch} carries {indices.size} indices "
                               f"but reports {query.get(count_key)}")
                continue
            if indices.size and (indices.min() < 0 or indices.max() >= base.shape[0]):
                reasons.append(f"{query['query_id']}: {branch} index out of range "
                               f"[{indices.min()}, {indices.max()}] for {base.shape[0]} "
                               "candidates")
                continue
            if len(set(indices.tolist())) != indices.size:
                reasons.append(f"{query['query_id']}: {branch} repeats an index")
                continue
            branches.add(branch)
            if branch == "full_height":
                digest = coordinates_digest(base[indices])
                if digest != query.get("candidate_coordinates_sha256"):
                    reasons.append(f"{query['query_id']}: reconstructed coordinates hash to "
                                   f"{digest[:12]}... but the manifest records "
                                   f"{str(query.get('candidate_coordinates_sha256'))[:12]}...")
    return {"ok": not reasons, "reasons": reasons, "manifest": manifest_path,
            "n_queries": len(manifest.get("queries", [])),
            "branches_reconstructed": sorted(branches),
            "room_id": manifest.get("room_id")}


def verify_report_chain(report_path):
    """The report's per-room digests must still match the manifests on disk."""
    out_dir = os.path.dirname(os.path.abspath(report_path))
    with open(report_path) as handle:
        report = json.load(handle)
    reasons = []
    for room_id, entry in sorted((report.get("rooms") or {}).items()):
        path = os.path.join(out_dir, entry["candidate_manifest"])
        if not os.path.isfile(path):
            reasons.append(f"{room_id}: {entry['candidate_manifest']} is missing")
            continue
        with open(path) as handle:
            payload = json.load(handle)
        digest = _sha256_json(payload)
        if digest != entry.get("candidate_manifest_sha256"):
            reasons.append(f"{room_id}: the manifest hashes to {digest[:12]}... but the "
                           f"report records {str(entry.get('candidate_manifest_sha256'))[:12]}"
                           "...; it was edited after publication")
        verdict = verify_room_manifest(path, out_dir=out_dir)
        if not verdict["ok"]:
            reasons.append(f"{room_id}: {verdict['reasons'][0]}")
    return {"ok": not reasons, "reasons": reasons, "n_rooms": len(report.get("rooms") or {}),
            "report": report_path}


def validate_records(records, expected_queries, required_rooms, expected_histogram):
    """The record stream IS the registered subset: count, uniqueness, order,
    census and the exact room set (r2 re-review F3)."""
    if expected_queries is not None and len(records) != int(expected_queries):
        raise ValueError(f"the context manifest carries {len(records)} queries but the "
                         f"registered in-scope count is {expected_queries}; the audit covers "
                         "the whole subset or refuses")
    identities = [record["query_id"] for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("the context manifest has duplicate query ids; every query must "
                         "appear exactly once")
    positions = [int(record["position"]) for record in records]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError(f"the records are not in a unique ascending position order "
                         f"(first offenders {positions[:5]})")
    rooms = sorted({record["room_id"] for record in records})
    if required_rooms is not None and rooms != sorted(required_rooms):
        missing = sorted(set(required_rooms) - set(rooms))
        extra = sorted(set(rooms) - set(required_rooms))
        raise ValueError(f"the audit room set is wrong: missing {missing}, unexpected "
                         f"{extra}; the required set is exactly the {len(required_rooms)} "
                         "mesh-available rooms")
    if expected_histogram is not None:
        histogram = mq.eligible_histogram(records)
        if histogram != expected_histogram:
            raise ValueError(f"the eligible-pool histogram is {histogram}, not the registered "
                             f"census {expected_histogram}")
    return True


def run_audit(context_manifest, mesh_root, metadata_root, out_dir,
              expected_queries=None, spacing=mg.LATTICE_SPACING,
              chunk=4096, required_rooms=REQUIRED_ROOMS,
              expected_histogram=None, diagnostics_only=False,
              _allow_fixture_census=False):
    """Audit every query, then -- only if every gate passed -- write artifacts.

    Nothing reaches disk until the whole audit has succeeded: a blocked room, a
    wrong room set, a broken record stream or a failed census leaves the output
    directory untouched. ``diagnostics_only`` writes ONE clearly stamped
    non-manifest report for inspection and never writes candidate manifests.
    """
    records = context_manifest["records"]
    # A REGISTERED run always enforces the registered scope: 5,337 queries and
    # the registered histogram. --expected-queries can only narrow a DIAGNOSTICS
    # run, which writes no candidate manifests (r3 review F3/F4).
    if not diagnostics_only and not _allow_fixture_census:
        if expected_queries not in (None, mq.FILTERED_COUNT):
            raise ValueError(f"--expected-queries {expected_queries} is only available to a "
                             "--diagnostics-only run; a registered audit covers exactly the "
                             f"{mq.FILTERED_COUNT}-query subset")
        expected_queries = mq.FILTERED_COUNT
        expected_histogram = mq.FILTERED_ELIGIBLE_HISTOGRAM
    validate_records(records, expected_queries, required_rooms, expected_histogram)

    # publish into a FRESH directory only: a leftover artifact could be mistaken
    # for part of this audit (r3 review, publish-phase)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f"the output directory {out_dir!r} is not empty; an audit publishes "
                         "into a fresh directory so no existing artifact can be mistaken for "
                         "part of it")

    by_room = {}
    for record in records:
        by_room.setdefault(record["room_id"], []).append(record)
    meshes = resolve_room_meshes(sorted(by_room), mesh_root)

    rooms, payloads, arrays = {}, {}, {}
    full_oracle, band_oracle = {}, {}
    blocked = []
    band_nonempty = True
    counts = {"full_height": GateCounter(), "z_band": GateCounter()}

    for room_id in sorted(by_room):
        scene, scene_id = room_id.split("/")
        scene_obj = mg.load_raycast_scene(meshes[room_id])
        anchors = mg.metadata_anchors(os.path.join(metadata_root, scene, scene_id))
        anchor_audit = mg.audit_room_anchors(scene_obj, anchors, room_id=room_id)
        if not anchor_audit["accepted"]:
            blocked.append({"room_id": room_id, "audit": anchor_audit})

        aabb_min, aabb_max = mg.scene_aabb(scene_obj)
        lattice = mg.build_lattice(aabb_min, aabb_max, spacing=spacing)
        base = mg.classify_mesh_candidates(scene_obj, lattice, chunk=chunk)
        base_candidates = lattice[base["valid"]]
        if base_candidates.shape[0] == 0:
            raise ValueError(f"{room_id}: the mesh-valid base grid is empty")
        # the SNAPPED origin: the first lattice node, not the raw AABB minimum
        snapped_origin = [float(v) for v in lattice[0]]

        queries = []
        for record in by_room[room_id]:
            receiver, target = _metadata_for(record, metadata_root)
            contexts = resolve_context_globals(record, receiver, anchors)
            full = mg.filter_query_candidates(base_candidates, receiver=receiver,
                                              context_sources=contexts)
            band_bounds = mg.context_z_band(contexts)
            try:
                banded = mg.filter_query_candidates(base_candidates, receiver=receiver,
                                                    context_sources=contexts,
                                                    z_band=band_bounds)
            except ValueError:
                band_nonempty = False
                banded = None

            full_indices = np.flatnonzero(full["mask"])
            band_indices = (np.flatnonzero(banded["mask"]) if banded is not None
                            else np.zeros(0, dtype=np.int64))
            full_error = mg.grid_oracle_error(full["candidates"], target)
            # an empty z-band set stays INFINITE: that is what disqualifies it
            band_error = (mg.grid_oracle_error(banded["candidates"], target)
                          if banded is not None else float("inf"))
            full_oracle[record["query_id"]] = full_error
            band_oracle[record["query_id"]] = band_error

            receiver_id = f"{room_id}|" + ",".join(f"{v:.6f}" for v in receiver)
            for branch, indices in (("full_height", full_indices), ("z_band", band_indices)):
                counts[branch].add(receiver_id, indices)

            queries.append({
                "query_id": record["query_id"], "position": record["position"],
                "n_candidates": int(full_indices.size),
                "n_candidates_z_band": int(band_indices.size),
                "candidate_indices": [int(i) for i in full_indices],
                "candidate_indices_z_band": [int(i) for i in band_indices],
                "candidate_coordinates_sha256": coordinates_digest(full["candidates"]),
                "n_dropped_receiver": full["n_dropped_receiver"],
                "n_dropped_context": full["n_dropped_context"],
                "z_band": [float(band_bounds[0]), float(band_bounds[1])],
                "oracle": {"full_height": full_error, "z_band": band_error},
                "receiver": [float(v) for v in receiver], "receiver_id": receiver_id,
                "n_contexts": len(contexts),
            })

        payloads[room_id] = {
            "room_id": room_id, "spacing": float(spacing),
            "lattice_origin": snapped_origin,
            "lattice_aabb_min": [float(v) for v in aabb_min],
            "lattice_aabb_max": [float(v) for v in aabb_max],
            "n_lattice": int(lattice.shape[0]),
            "n_parity_valid": int(base["parity_valid"].sum()),
            "n_base_valid": int(base["n_valid"]),
            "base_candidates_sha256": coordinates_digest(base_candidates),
            "coordinates_npz": f"candidates_{scene}_{scene_id}.npz",
            "clearance": base["clearance"], "eps": base["eps"],
            "directions_sha256": mg.FROZEN_DIRECTIONS_SHA256,
            "directions_seed": mg.FROZEN_DIRECTIONS_SEED,
            "direction_selection_rule": mg.DIRECTION_SELECTION_RULE,
            "mesh": scene_obj.identity, "anchor_audit": anchor_audit,
            "queries": queries,
            "exclusion": {"room_id": mq.EXCLUDED_ROOM, "n_excluded": mq.EXCLUDED_COUNT,
                          "reason": "no official OBJ; mesh-available preflight subset"},
        }
        arrays[room_id] = base_candidates
        rooms[room_id] = {
            "mesh": {"path": meshes[room_id], "sha256": scene_obj.identity["sha256"],
                     "n_triangles": scene_obj.identity["n_triangles"],
                     "watertight": scene_obj.identity["watertight"]},
            "anchor_audit_accepted": anchor_audit["accepted"],
            "n_queries": len(queries), "n_base_valid": int(base["n_valid"]),
            "candidate_manifest": f"candidates_{scene}_{scene_id}.json",
        }

    branch = mg.choose_z_branch(full_oracle, band_oracle, band_nonempty=band_nonempty)
    cost = {name: counts[name].summary() for name in ("full_height", "z_band")}
    cost["chosen_branch"] = branch["branch"]

    report = {
        "experiment": "exp_22 loc_meshgrid G1 geometry audit",
        "n_rooms": len(rooms), "n_queries": len(records), "spacing": float(spacing),
        "directions_sha256": mg.FROZEN_DIRECTIONS_SHA256,
        "directions_seed": mg.FROZEN_DIRECTIONS_SEED,
        "direction_selection_rule": mg.DIRECTION_SELECTION_RULE,
        "resolved_parity_discrepancies": list(mg.RESOLVED_PARITY_DISCREPANCIES),
        "context_manifest_sha256": context_manifest.get("filtered_stream_sha256"),
        "required_rooms": list(required_rooms or []),
        "rooms": rooms,
        "oracle": {"full_height": _summary(full_oracle.values()),
                   "z_band": _summary(band_oracle.values(), allow_infinite=True)},
        "branch": branch, "cost": cost,
        "anchor_audit": {"accepted_rooms": sorted(r for r, v in rooms.items()
                                                  if v["anchor_audit_accepted"]),
                         "blocked_rooms": sorted(entry["room_id"] for entry in blocked),
                         "blocked_detail": blocked},
        "diagnostics_only": bool(diagnostics_only),
        "status": ("DIAGNOSTICS ONLY -- not a registration artifact; candidate manifests "
                   "were NOT written" if diagnostics_only else "audit complete"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if blocked and not diagnostics_only:
        raise ValueError(
            f"the anchor audit blocked {[entry['room_id'] for entry in blocked]}; the audit "
            "writes nothing until every required room is accepted (a blocked room is a "
            "pending ruling, not a warning). Re-run with diagnostics_only=True to inspect.")

    if diagnostics_only:
        os.makedirs(out_dir, exist_ok=True)
        _write_json(os.path.join(out_dir, "geometry_diagnostics_report.json"), report)
        return report

    # Everything is written into a STAGING sibling, verified there, and only then
    # published: a verifier failure must leave the final directory as it found it
    # (r4 review). The verification block is written INTO the report before its
    # disk copy, so the published report is not missing what was checked.
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".staging_geometry_audit_", dir=parent)
    try:
        for room_id, payload in payloads.items():
            payload["chosen_branch"] = branch["branch"]
            np.savez(os.path.join(staging, payload["coordinates_npz"]),
                     base_candidates=arrays[room_id])
            rooms[room_id]["candidate_manifest_sha256"] = _sha256_json(payload)
            _write_json(os.path.join(staging, rooms[room_id]["candidate_manifest"]), payload)

        verifications = {}
        for room_id in payloads:
            verdict = verify_room_manifest(
                os.path.join(staging, rooms[room_id]["candidate_manifest"]), out_dir=staging)
            verifications[room_id] = verdict
            if not verdict["ok"]:
                raise ValueError(f"the staged manifest for {room_id} does not verify: "
                                 f"{verdict['reasons'][:3]}; nothing was published")
        report["verification"] = {"rooms": {room: verdict["ok"]
                                            for room, verdict in verifications.items()},
                                  "chain_ok": None, "staged": True}
        _write_json(os.path.join(staging, "geometry_audit_report.json"), report)

        chain = verify_report_chain(os.path.join(staging, "geometry_audit_report.json"))
        if not chain["ok"]:
            raise ValueError(f"the staged report does not verify against its manifests: "
                             f"{chain['reasons'][:3]}; nothing was published")
        report["verification"]["chain_ok"] = True
        _write_json(os.path.join(staging, "geometry_audit_report.json"), report)

        _publish_staging(staging, out_dir)
        staging = None
    finally:
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return report


def _publish_staging(staging, out_dir):
    """Move the verified staging directory into place, atomically where possible."""
    if os.path.isdir(out_dir):
        if os.listdir(out_dir):
            raise ValueError(f"{out_dir!r} became non-empty during the audit; refusing to "
                             "publish over it")
        os.rmdir(out_dir)                       # so the whole-dir rename can land
    try:
        os.replace(staging, out_dir)            # one atomic rename
        return out_dir
    except OSError:                             # different filesystems: file by file
        os.makedirs(out_dir, exist_ok=True)
        for name in sorted(os.listdir(staging)):
            os.replace(os.path.join(staging, name), os.path.join(out_dir, name))
        os.rmdir(staging)
        return out_dir


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--metadata-root", default=os.path.join("AcousticRooms", "metadata"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-queries", type=int, default=None,
                        help="only for --diagnostics-only; a registered audit always covers "
                             f"the {mq.FILTERED_COUNT}-query subset")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="write ONE clearly stamped non-manifest report for inspection "
                             "instead of refusing on a blocked room; never writes candidate "
                             "manifests")
    args = parser.parse_args(argv)

    manifest = mq.load_manifest(args.context_manifest)
    # the context manifest's own census, before any geometry runs
    mq.assert_registered_census(manifest)
    report = run_audit(manifest, mesh_root=args.mesh_root,
                       metadata_root=args.metadata_root, out_dir=args.out_dir,
                       expected_queries=args.expected_queries,
                       diagnostics_only=args.diagnostics_only)
    print(f"rooms {report['n_rooms']} | queries {report['n_queries']} | "
          f"branch {report['branch']['branch']}")
    for name in ("full_height", "z_band"):
        block = report["oracle"][name]
        print(f"oracle {name:12s} median {block['median']:.3f} m, over-threshold "
              f"{block['n_over_threshold']}, infinite {block['n_infinite']}")
        cost = report["cost"][name]
        print(f"  cost[{name}]: {cost['candidate_query_pairs']} candidate-query pairs, "
              f"{cost['unique_receiver_candidate_pairs']} unique receiver-candidate pairs, "
              f"~{cost['conditioner_calls_estimate']} conditioner calls, "
              f"{cost['artifact_bytes'] / 1e6:.1f} MB")
    if report["anchor_audit"]["blocked_rooms"]:
        print(f"BLOCKED rooms (anchor audit): {report['anchor_audit']['blocked_rooms']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
