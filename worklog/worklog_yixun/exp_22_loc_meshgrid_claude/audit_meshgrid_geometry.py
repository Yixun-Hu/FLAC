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
import sys
from datetime import datetime, timezone

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.localization import meshgrid_geometry as mg          # noqa: E402
from src.localization import meshgrid_queries as mq           # noqa: E402

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
    """``(receiver_global, target_global)`` from the query's own metadata JSON."""
    room_id = record["room_id"]
    scene, scene_id = room_id.split("/")
    name = os.path.basename(record["relpath"])
    source_node, receiver_node = name.split("_")[0], name.split("_")[1]
    path = os.path.join(metadata_root, scene, scene_id, f"{source_node}_{receiver_node}.json")
    if not os.path.isfile(path):
        raise ValueError(f"query {record['query_id']!r}: metadata not found at {path}")
    with open(path) as handle:
        payload = json.load(handle)
    return (np.asarray(payload["rec_loc"], dtype=np.float64),
            np.asarray(payload["src_loc"], dtype=np.float64))


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode()
                          + b"\n").hexdigest()


def _summary(values):
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("an oracle distribution must be nonempty and finite")
    return {"n_queries": int(array.size), "median": float(np.median(array)),
            "mean": float(array.mean()), "max": float(array.max()),
            "n_over_threshold": int((array > mg.ORACLE_THRESHOLD).sum()),
            "fraction_over_threshold": float((array > mg.ORACLE_THRESHOLD).mean())}


def run_audit(context_manifest, mesh_root, metadata_root, out_dir,
              expected_queries=mq.FILTERED_COUNT, spacing=mg.LATTICE_SPACING,
              chunk=4096):
    """Audit every query in the manifest; write per-room manifests and the report."""
    records = context_manifest["records"]
    if expected_queries is not None and len(records) != int(expected_queries):
        raise ValueError(f"the context manifest carries {len(records)} queries but the "
                         f"registered in-scope count is {expected_queries}; the audit covers "
                         "the whole subset or refuses")
    os.makedirs(out_dir, exist_ok=True)

    by_room = {}
    for record in records:
        by_room.setdefault(record["room_id"], []).append(record)
    meshes = resolve_room_meshes(sorted(by_room), mesh_root)

    rooms, full_oracle, band_oracle = {}, {}, {}
    band_nonempty = True
    candidate_query_pairs = 0
    receiver_candidate_pairs = set()

    for room_id in sorted(by_room):
        scene, scene_id = room_id.split("/")
        scene_obj = mg.load_raycast_scene(meshes[room_id])
        anchors = mg.metadata_anchors(os.path.join(metadata_root, scene, scene_id))
        audit = mg.audit_room_anchors(scene_obj, anchors, room_id=room_id)

        lattice = mg.build_lattice(*mg.scene_aabb(scene_obj), spacing=spacing)
        base = mg.classify_mesh_candidates(scene_obj, lattice, chunk=chunk)
        base_candidates = lattice[base["valid"]]
        if base_candidates.shape[0] == 0:
            raise ValueError(f"{room_id}: the mesh-valid base grid is empty")

        queries = []
        for record in by_room[room_id]:
            receiver, target = _metadata_for(record, metadata_root)
            contexts = resolve_context_globals(record, receiver, anchors)
            full = mg.filter_query_candidates(base_candidates, receiver=receiver,
                                              context_sources=contexts)
            band = mg.context_z_band(contexts)
            try:
                banded = mg.filter_query_candidates(base_candidates, receiver=receiver,
                                                    context_sources=contexts, z_band=band)
            except ValueError:
                band_nonempty = False
                banded = None

            full_error = mg.grid_oracle_error(full["candidates"], target)
            band_error = (mg.grid_oracle_error(banded["candidates"], target)
                          if banded is not None else float("inf"))
            full_oracle[record["query_id"]] = full_error
            band_oracle[record["query_id"]] = band_error if banded is not None else full_error

            candidate_query_pairs += full["n_candidates"]
            receiver_key = (room_id, tuple(np.round(receiver, 6)))
            receiver_candidate_pairs.add((receiver_key, full["n_candidates"]))
            queries.append({
                "query_id": record["query_id"], "position": record["position"],
                "n_candidates": full["n_candidates"],
                "n_candidates_z_band": (banded["n_candidates"] if banded is not None else 0),
                "n_dropped_receiver": full["n_dropped_receiver"],
                "n_dropped_context": full["n_dropped_context"],
                "z_band": [float(band[0]), float(band[1])],
                "oracle": {"full_height": full_error, "z_band": band_error},
                "receiver": [float(v) for v in receiver],
                "n_contexts": len(contexts),
            })

        payload = {
            "room_id": room_id, "spacing": float(spacing),
            "lattice_origin": [float(v) for v in mg.scene_aabb(scene_obj)[0]],
            "n_lattice": int(lattice.shape[0]),
            "n_parity_valid": int(base["parity_valid"].sum()),
            "n_base_valid": int(base["n_valid"]),
            "clearance": base["clearance"], "eps": base["eps"],
            "directions_sha256": mg.FROZEN_DIRECTIONS_SHA256,
            "mesh": scene_obj.identity, "anchor_audit": audit,
            "queries": queries,
            "exclusion": {"room_id": mq.EXCLUDED_ROOM, "n_excluded": mq.EXCLUDED_COUNT,
                          "reason": "no official OBJ; mesh-available preflight subset"},
        }
        name = f"candidates_{scene}_{scene_id}.json"
        digest = _sha256_json(payload)
        with open(os.path.join(out_dir, name), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        rooms[room_id] = {
            "mesh": {"path": meshes[room_id], "sha256": scene_obj.identity["sha256"],
                     "n_triangles": scene_obj.identity["n_triangles"],
                     "watertight": scene_obj.identity["watertight"]},
            "anchor_audit_accepted": audit["accepted"],
            "n_queries": len(queries), "n_base_valid": int(base["n_valid"]),
            "candidate_manifest": name, "candidate_manifest_sha256": digest,
        }

    branch = mg.choose_z_branch(full_oracle, band_oracle, band_nonempty=band_nonempty)
    unique_pairs = sum(count for _key, count in receiver_candidate_pairs)
    report = {
        "experiment": "exp_22 loc_meshgrid G1 geometry audit",
        "n_rooms": len(rooms), "n_queries": len(records),
        "spacing": float(spacing),
        "directions_sha256": mg.FROZEN_DIRECTIONS_SHA256,
        "context_manifest_sha256": context_manifest.get("filtered_stream_sha256"),
        "rooms": rooms,
        "oracle": {"full_height": _summary(full_oracle.values()),
                   "z_band": _summary(band_oracle.values())},
        "branch": branch,
        "anchor_audit": {"accepted_rooms": sorted(r for r, v in rooms.items()
                                                  if v["anchor_audit_accepted"]),
                         "blocked_rooms": sorted(r for r, v in rooms.items()
                                                 if not v["anchor_audit_accepted"])},
        "cost": {
            "candidate_query_pairs": int(candidate_query_pairs),
            "unique_receiver_candidate_pairs": int(unique_pairs),
            "conditioner_calls_estimate": int(unique_pairs * CONDITIONER_CALLS_PER_PAIR),
            "artifact_bytes": int(candidate_query_pairs * BYTES_PER_CANDIDATE),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "geometry_audit_report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--metadata-root", default=os.path.join("AcousticRooms", "metadata"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-queries", type=int, default=mq.FILTERED_COUNT)
    args = parser.parse_args(argv)

    manifest = mq.load_manifest(args.context_manifest)
    report = run_audit(manifest, mesh_root=args.mesh_root,
                       metadata_root=args.metadata_root, out_dir=args.out_dir,
                       expected_queries=args.expected_queries)
    print(f"rooms {report['n_rooms']} | queries {report['n_queries']} | "
          f"branch {report['branch']['branch']}")
    print(f"oracle full-height median {report['oracle']['full_height']['median']:.3f} m, "
          f"over-threshold {report['oracle']['full_height']['n_over_threshold']}")
    print(f"oracle z-band      median {report['oracle']['z_band']['median']:.3f} m, "
          f"over-threshold {report['oracle']['z_band']['n_over_threshold']}")
    cost = report["cost"]
    print(f"cost: {cost['candidate_query_pairs']} candidate-query pairs, "
          f"{cost['unique_receiver_candidate_pairs']} unique receiver-candidate pairs, "
          f"~{cost['conditioner_calls_estimate']} conditioner calls, "
          f"{cost['artifact_bytes'] / 1e6:.1f} MB of candidate artifacts")
    if report["anchor_audit"]["blocked_rooms"]:
        print(f"BLOCKED rooms (anchor audit): {report['anchor_audit']['blocked_rooms']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
