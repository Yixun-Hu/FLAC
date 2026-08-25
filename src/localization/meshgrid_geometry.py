"""exp_22 G1 -- mesh-valid 3-D candidate geometry (inherited plan §1.2/§1.3).

The candidate set is a room-global 0.5 m lattice filtered by physical validity.
Validity is deliberately NOT Open3D's occupancy: the official AcousticRooms
meshes are neither watertight nor manifold, so a single ray (or any rule that
assumes closure) inherits that fragility. The approved B7 rule is a strict
majority of odd ray-parity votes over 31 frozen, non-axis-aligned directions --
odd so a vote cannot tie, non-axis-aligned because these rooms are built from
axis-aligned triangles that a coordinate-parallel ray grazes.

Separately from parity, a source-distribution prior keeps only candidates at
least 0.20 m from any surface. The two are reported separately because they mean
different things: parity says "inside the room", the prior says "somewhere a
source could plausibly sit".

Everything else is per query and never touches the target: at least 0.5 m from
the known receiver, a 0.25 m duplicate guard around each selected context source,
and an optional z-band derived from the CONTEXT heights only. Ground truth is
never inserted or snapped into the candidate set.
"""
import hashlib
import json
import os

import numpy as np

#: the lattice step, in metres (inherited plan §1.2).
LATTICE_SPACING = 0.5
#: one tolerance for every boundary: surface, receiver, context, z-band.
EPS = 1e-4
#: source-distribution surface-clearance prior.
SURFACE_CLEARANCE = 0.20
#: minimum distance from the known receiver.
RECEIVER_MIN_DISTANCE = 0.5
#: numerical-duplicate guard around each selected context source (half a step).
CONTEXT_GUARD_RADIUS = 0.25
#: the z-band pad applied to the context height range.
Z_BAND_PAD = 0.5
#: an oracle error above this is what the z-band branch may not create.
ORACLE_THRESHOLD = 0.5
#: how many ray directions vote; odd, so a strict majority always exists.
N_DIRECTIONS = 31


def build_directions(count=N_DIRECTIONS, seed=0):
    """``count`` unit directions, deterministic and never axis-aligned.

    A seeded Fibonacci sphere gives an even, reproducible spread; the golden-angle
    offset is rotated by a fixed irrational amount so no sample lands in a
    coordinate plane, which is what an axis-aligned mesh grazes.
    """
    count = int(count)
    if count < 3 or count % 2 == 0:
        raise ValueError(f"the direction count must be odd and >= 3, got {count}")
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    indices = np.arange(count, dtype=np.float64) + 0.5
    # tilt both angles by fixed irrationals: keeps every component away from 0
    z = 1.0 - 2.0 * indices / count
    z = z * 0.987654321 + 0.012345679 * np.cos(indices + float(seed))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * indices / golden + 0.3927 + float(seed)
    directions = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    if np.abs(directions).min() <= 1e-3:
        raise ValueError("a generated direction lies in a coordinate plane")
    return directions


#: FROZEN at import: the 31 directions every validity vote uses, forever.
FROZEN_DIRECTIONS = build_directions(N_DIRECTIONS)


# --------------------------------------------------------------------------- #
# the lattice
# --------------------------------------------------------------------------- #
def snap_axis_to_lattice(low, high, spacing=LATTICE_SPACING):
    """Integer multiples of ``spacing`` inside ``[low, high]``, inclusive."""
    spacing = float(spacing)
    if not spacing > 0 or not np.isfinite(spacing):
        raise ValueError(f"spacing must be a positive finite number, got {spacing}")
    low, high = float(low), float(high)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"axis bounds must be finite, got [{low}, {high}]")
    first = int(np.ceil(low / spacing - EPS))
    last = int(np.floor(high / spacing + EPS))
    if last < first:
        return np.zeros(0, dtype=np.float64)
    return np.arange(first, last + 1, dtype=np.float64) * spacing


def build_lattice(aabb_min, aabb_max, spacing=LATTICE_SPACING):
    """The room-global lattice, in lexicographic (x, y, z) order.

    Independent of query and of ground truth by construction: only the room's
    AABB and the spacing enter.
    """
    low = np.asarray(aabb_min, dtype=np.float64).reshape(-1)
    high = np.asarray(aabb_max, dtype=np.float64).reshape(-1)
    if low.size != 3 or high.size != 3:
        raise ValueError("an AABB is two 3-vectors")
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(f"the AABB must be finite, got {low.tolist()} .. {high.tolist()}")
    if np.any(high < low):
        raise ValueError(f"the AABB is inverted: {low.tolist()} .. {high.tolist()}")
    axes = [snap_axis_to_lattice(low[i], high[i], spacing) for i in range(3)]
    if any(axis.size == 0 for axis in axes):
        return np.zeros((0, 3), dtype=np.float64)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.ascontiguousarray(grid, dtype=np.float64)


# --------------------------------------------------------------------------- #
# the mesh
# --------------------------------------------------------------------------- #
class RaycastScene:
    """An Open3D raycasting scene plus the identity of the mesh behind it."""

    def __init__(self, scene, identity):
        self.scene, self.identity = scene, identity


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raycast_scene(path):
    """Load an OBJ fail-closed and return the scene with its identity.

    A mesh that does not exist, does not parse, has no triangles or carries a
    non-finite vertex blocks the room rather than warning (inherited plan §1.3).
    """
    import open3d as o3d

    path = str(path)
    if not os.path.isfile(path):
        raise ValueError(f"mesh not found: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    if triangles.size == 0:
        raise ValueError(f"{path}: the mesh parsed to zero triangles; a room cannot be "
                         "classified from vertices alone")
    if vertices.size == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"{path}: the mesh has non-finite or missing vertices")

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    identity = {
        "path": path, "sha256": _sha256_file(path),
        "n_vertices": int(vertices.shape[0]), "n_triangles": int(triangles.shape[0]),
        "aabb_min": [float(v) for v in vertices.min(axis=0)],
        "aabb_max": [float(v) for v in vertices.max(axis=0)],
        "watertight": bool(mesh.is_watertight()),
        "edge_manifold": bool(mesh.is_edge_manifold()),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": None,          # O(n^2) on these meshes; recorded as unknown
        "backend": f"open3d {o3d.__version__} RaycastingScene",
        "directions_sha256": hashlib.sha256(
            FROZEN_DIRECTIONS.tobytes()).hexdigest(),
        "n_directions": int(FROZEN_DIRECTIONS.shape[0]),
    }
    return RaycastScene(scene, identity)


def scene_aabb(scene):
    """``(aabb_min, aabb_max)`` of the loaded mesh."""
    return scene.identity["aabb_min"], scene.identity["aabb_max"]


def _as_points(points):
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if array.size and not np.isfinite(array).all():
        raise ValueError("candidate points must be finite")
    return array


def classify_free_space(scene, points, directions=None, chunk=None):
    """Strict-majority odd ray parity over the frozen directions.

    A point is inside iff more than half of the 31 rays cross an odd number of
    triangles. Chunking changes nothing but memory: the vote is per point.
    """
    import open3d as o3d

    directions = FROZEN_DIRECTIONS if directions is None else np.asarray(
        directions, dtype=np.float64).reshape(-1, 3)
    points = _as_points(points)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    n_directions = directions.shape[0]
    majority = n_directions // 2 + 1
    size = int(chunk) if chunk else points.shape[0]

    odd_votes = np.zeros(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], size):
        block = points[start:start + size]
        rays = np.empty((block.shape[0] * n_directions, 6), dtype=np.float32)
        rays[:, :3] = np.repeat(block, n_directions, axis=0)
        rays[:, 3:] = np.tile(directions, (block.shape[0], 1))
        counts = scene.scene.count_intersections(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)).numpy()
        odd_votes[start:start + size] = (counts.reshape(block.shape[0], n_directions) % 2
                                         == 1).sum(axis=1)
    return odd_votes >= majority


def surface_distance(scene, points, chunk=None):
    """Unsigned distance from each point to the nearest triangle."""
    import open3d as o3d

    points = _as_points(points)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    size = int(chunk) if chunk else points.shape[0]
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], size):
        block = points[start:start + size].astype(np.float32)
        out[start:start + size] = scene.scene.compute_distance(
            o3d.core.Tensor(block, dtype=o3d.core.Dtype.Float32)).numpy().astype(np.float64)
    return out


def classify_mesh_candidates(scene, points, chunk=None, clearance=SURFACE_CLEARANCE,
                             eps=EPS):
    """Parity AND the surface-clearance prior, reported separately."""
    points = _as_points(points)
    parity = classify_free_space(scene, points, chunk=chunk)
    distance = surface_distance(scene, points, chunk=chunk)
    clearance_valid = distance + eps >= float(clearance)
    return {"valid": parity & clearance_valid, "parity_valid": parity,
            "clearance_valid": clearance_valid, "distance": distance,
            "clearance": float(clearance), "eps": float(eps),
            "n_points": int(points.shape[0]),
            "n_valid": int((parity & clearance_valid).sum())}


# --------------------------------------------------------------------------- #
# per-query filters
# --------------------------------------------------------------------------- #
def context_z_band(context_sources, pad=Z_BAND_PAD):
    """``[min(z)-pad, max(z)+pad]`` from the CONTEXT heights only."""
    sources = np.asarray(list(context_sources), dtype=np.float64).reshape(-1, 3)
    if sources.shape[0] == 0:
        raise ValueError("a z-band needs at least one context source")
    if not np.isfinite(sources).all():
        raise ValueError("context sources must be finite")
    return (float(sources[:, 2].min() - pad), float(sources[:, 2].max() + pad))


def filter_query_candidates(candidates, receiver, context_sources=(), z_band=None,
                            eps=EPS):
    """The per-query mask: receiver distance, context guard, optional z-band.

    Never inserts or snaps the ground truth, which is not an argument here.
    """
    candidates = _as_points(candidates)
    receiver = np.asarray(receiver, dtype=np.float64).reshape(3)
    if not np.isfinite(receiver).all():
        raise ValueError("the receiver must be finite")
    keep = np.ones(candidates.shape[0], dtype=bool)

    far_enough = (np.linalg.norm(candidates - receiver, axis=1) + eps
                  >= RECEIVER_MIN_DISTANCE)
    n_dropped_receiver = int((~far_enough).sum())
    keep &= far_enough

    n_dropped_context = 0
    for source in (np.asarray(s, dtype=np.float64).reshape(3) for s in context_sources):
        if not np.isfinite(source).all():
            raise ValueError("context sources must be finite")
        near = np.linalg.norm(candidates - source, axis=1) - eps <= CONTEXT_GUARD_RADIUS
        n_dropped_context += int((near & keep).sum())
        keep &= ~near

    n_dropped_band = 0
    if z_band is not None:
        low, high = float(z_band[0]), float(z_band[1])
        inside = (candidates[:, 2] + eps >= low) & (candidates[:, 2] - eps <= high)
        n_dropped_band = int((~inside & keep).sum())
        keep &= inside

    kept = np.ascontiguousarray(candidates[keep])
    if kept.shape[0] == 0:
        raise ValueError("every candidate was filtered away; a query must keep a nonempty "
                         "candidate set (inherited plan §1.2)")
    return {"candidates": kept, "mask": keep, "n_candidates": int(kept.shape[0]),
            "n_dropped_receiver": n_dropped_receiver,
            "n_dropped_context": n_dropped_context,
            "n_dropped_z_band": n_dropped_band,
            "z_band": None if z_band is None else [float(z_band[0]), float(z_band[1])],
            "receiver_min_distance": RECEIVER_MIN_DISTANCE,
            "context_guard_radius": CONTEXT_GUARD_RADIUS, "eps": float(eps)}


def grid_oracle_error(candidates, truth):
    """``min_c ||c - x*_s||`` -- the continuous-grid oracle error."""
    candidates = _as_points(candidates)
    if candidates.shape[0] == 0:
        raise ValueError("the oracle needs a nonempty candidate set")
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    if not np.isfinite(truth).all():
        raise ValueError("the ground-truth source must be finite")
    return float(np.linalg.norm(candidates - truth, axis=1).min())


def choose_z_branch(full_height_oracle, band_oracle, band_nonempty,
                    threshold=ORACLE_THRESHOLD):
    """The pre-registered branch rule, decided from geometry alone.

    Use the z-band globally only if every query stays nonempty under it AND it
    creates no query with ``e_oracle > threshold`` that full height did not have;
    otherwise full height, globally.
    """
    if not band_nonempty:
        return {"branch": "full_height", "n_new_over_threshold": None,
                "reason": "at least one query's z-band candidate set was not nonempty"}
    new_over = [query for query, error in band_oracle.items()
                if error > threshold and float(full_height_oracle.get(query, 0.0)) <= threshold]
    if new_over:
        return {"branch": "full_height", "n_new_over_threshold": len(new_over),
                "queries": sorted(new_over)[:10],
                "reason": f"the z-band created {len(new_over)} queries with e_oracle > "
                          f"{threshold} m that full height did not have"}
    return {"branch": "z_band", "n_new_over_threshold": 0,
            "reason": f"every query stays nonempty and no new e_oracle > {threshold} m "
                      "query appears"}


# --------------------------------------------------------------------------- #
# helpers for the audits
# --------------------------------------------------------------------------- #
def metadata_anchors(metadata_dir):
    """Every distinct source and receiver coordinate a room's metadata names."""
    sources, receivers = {}, {}
    for name in sorted(os.listdir(metadata_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(metadata_dir, name)) as handle:
            payload = json.load(handle)
        source, receiver = payload.get("src_loc"), payload.get("rec_loc")
        if source is not None:
            sources[tuple(round(float(v), 6) for v in source)] = [float(v) for v in source]
        if receiver is not None:
            receivers[tuple(round(float(v), 6) for v in receiver)] = [float(v)
                                                                      for v in receiver]
    return {"sources": [sources[key] for key in sorted(sources)],
            "receivers": [receivers[key] for key in sorted(receivers)],
            "metadata_dir": str(metadata_dir)}


def write_box_obj(path, boxes):
    """Write an OBJ of axis-aligned boxes -- the synthetic rooms the tests use."""
    vertices, faces = [], []
    for (low, high) in boxes:
        (x0, y0, z0), (x1, y1, z1) = low, high
        base = len(vertices) + 1
        vertices += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                     (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                 (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
        for a, b, c, d in quads:
            faces.append((base + a, base + b, base + c))
            faces.append((base + a, base + c, base + d))
    with open(path, "w") as handle:
        for vertex in vertices:
            handle.write("v {:.6f} {:.6f} {:.6f}\n".format(*vertex))
        for face in faces:
            handle.write("f {} {} {}\n".format(*face))
    return str(path)


def audit_room_anchors(scene, anchors, clearance=SURFACE_CLEARANCE, eps=EPS):
    """The §1.3 fail-closed room acceptance, as the plan states it.

    Rule 2 applies to EVERY metadata anchor: finite, and inside the free-space
    classification after the ``1e-4 m`` tolerance. Rule 3 names the SOURCE
    anchors specifically: they must also survive the candidate predicate, i.e.
    the 0.20 m source-distribution prior. Receivers are not candidates and are
    not drawn from the source distribution -- their own constraint is the
    >= 0.5 m candidate-distance guard -- so the prior is not applied to them.
    """
    report = {"metadata_dir": anchors.get("metadata_dir"), "clearance": float(clearance),
              "eps": float(eps), "rules": {}}
    accepted = True
    for label in ("sources", "receivers"):
        points = np.asarray(anchors[label], dtype=np.float64).reshape(-1, 3)
        verdict = classify_mesh_candidates(scene, points, clearance=clearance, eps=eps)
        block = {"n": int(points.shape[0]),
                 "n_parity_valid": int(verdict["parity_valid"].sum()),
                 "n_clearance_valid": int(verdict["clearance_valid"].sum()),
                 "min_distance": (float(verdict["distance"].min()) if points.size else None)}
        if not bool(verdict["parity_valid"].all()):
            accepted = False
            block["failure"] = "rule 2: an anchor is outside the free-space classification"
        if label == "sources" and not bool(verdict["valid"].all()):
            accepted = False
            block["failure"] = ("rule 3: a source anchor fails the candidate predicate "
                                "(parity + the 0.20 m source prior)")
        report["rules"][label] = block
    report["accepted"] = accepted
    return report
