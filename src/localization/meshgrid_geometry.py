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


#: THE frozen direction set, written out as literals.
#:
#: r1 review F3: regenerating these from the generator at import is not a pin --
#: a changed generator stays green against its own output. These 31 unit vectors
#: ARE the protocol; ``build_directions`` is kept only as the provenance of how
#: they were produced, and a test asserts the two still agree AND that the digest
#: below is unchanged.
FROZEN_DIRECTIONS_LITERAL = (
    (-0.1771400678047207, 0.18506768919042715, 0.9666288567986445),
    (0.44918969783162255, -0.02953273079508564, 0.8929481693657928),
    (-0.4482095600950777, -0.3594773005568645, 0.8184645750573001),
    (0.1003951590430564, 0.6502305325817338, 0.7530744096961431),
    (0.39731027876809805, -0.5954013140943477, 0.6983135524676147),
    (-0.7413713938219922, 0.1819974362937909, 0.6459453456794346),
    (0.7110598798867792, 0.3892872398747092, 0.5855333398587393),
    (-0.2765451918273667, -0.8119662651815595, 0.5140365172672801),
    (-0.36099339915426826, 0.822988003214889, 0.4386051895855965),
    (0.8498688337349825, -0.3752569253095685, 0.370007034329281),
    (-0.8998681240823277, -0.30403777553453665, 0.3127273417988316),
    (0.46565484132321705, 0.8456508257901678, 0.26084525986235),
    (0.23110306920210177, -0.9514138530985036, 0.2034773981002008),
    (-0.8228745234287446, 0.5520061602870696, 0.13478396676549745),
    (0.9869163087380934, 0.14991737324948992, 0.059337852545819574),
    (-0.6275136393809939, -0.7785118585521303, -0.012079672368400381),
    (-0.06300703989133183, 0.9953841603953444, -0.07239120221537622),
    (0.7150943824199973, -0.6878098708455005, -0.12473013188022608),
    (-0.9834560824042765, 0.023924926730223638, -0.17955982808798462),
    (0.7306018569799332, 0.6373143895564594, -0.2450536582908952),
    (-0.10577313475482022, -0.9416370797499404, -0.3195804343262319),
    (-0.5415184806977659, 0.7430667908309223, -0.39320411929050003),
    (0.8719340220460582, -0.17620296861865783, -0.45681897404612765),
    (-0.7365252345925201, -0.44373251919877316, -0.5105213317912817),
    (0.2337982728223017, 0.792327215638578, -0.5635210297617358),
    (0.34260366140477083, -0.7008244201131407, -0.6256739273496421),
    (-0.6660910120575932, 0.26178541699578206, -0.698420474431009),
    (0.5915988447453302, 0.22752652263004627, -0.7734613683926753),
    (-0.24158905966343766, -0.4851627693106215, -0.8403878946806556),
    (-0.12241418302241346, 0.4263188313322928, -0.8962516509588271),
    (0.2707082469830636, -0.16580413333099558, -0.9482752946195077),
)

FROZEN_DIRECTIONS = np.array(FROZEN_DIRECTIONS_LITERAL, dtype=np.float64)

#: sha256 over ``FROZEN_DIRECTIONS.tobytes()``; recorded in every manifest so a
#: drifted direction set is visible in the artifacts, not only in the code.
FROZEN_DIRECTIONS_SHA256 = (
    "9ab4339fa893c00dca817b901a149c292b080d0e6971c90f0b8b0b88e858c261")


#: Documented, UNRESOLVED anchor discrepancies (r1 review F3).
#:
#: The reviewer's 16-room sweep found one metadata anchor that the strict-majority
#: rule rejects. It is recorded here rather than papered over: neither the
#: majority rule nor the anchor predicate is changed, because that decision waits
#: for the rsynced exp_09 artifact cross-check. An audit that hits one of these
#: reports it as a KNOWN discrepancy and still refuses the room -- fail-closed --
#: so the ruling cannot be skipped by accident.
KNOWN_PARITY_DISCREPANCIES = (
    {"room_id": "MeetingRoom/MeetingRoom_idx_32", "kind": "receivers",
     "point": [2.26, 0.48, 1.2], "odd_votes": 15, "n_directions": 31,
     "surface_distance_m": 0.25005,
     "status": "documented, unresolved",
     "note": "15/31 odd votes is one below the strict majority; the surface distance "
             "is 0.25005 m, so this is not a clearance or epsilon effect. Pending the "
             "exp_09 artifact cross-check; the majority rule and the anchor predicate "
             "are unchanged in this round."},
)


def known_discrepancy(room_id, point, kind, tolerance=1e-6):
    """The documented entry for this anchor, or ``None``."""
    for entry in KNOWN_PARITY_DISCREPANCIES:
        if entry["room_id"] != room_id or entry["kind"] != kind:
            continue
        if np.allclose(np.asarray(entry["point"], dtype=np.float64),
                       np.asarray(point, dtype=np.float64), atol=tolerance):
            return dict(entry)
    return None


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
        "directions_sha256": hashlib.sha256(FROZEN_DIRECTIONS.tobytes()).hexdigest(),
        "directions_sha256_pinned": FROZEN_DIRECTIONS_SHA256,
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


def odd_parity_votes(scene, points, directions=None, chunk=None):
    """How many of the frozen rays cross an ODD number of triangles, per point.

    Exposed because the vote count is the evidence: a documented discrepancy is
    "15 of 31", not merely "rejected".
    """
    import open3d as o3d

    directions = FROZEN_DIRECTIONS if directions is None else np.asarray(
        directions, dtype=np.float64).reshape(-1, 3)
    points = _as_points(points)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    n_directions = directions.shape[0]
    size = int(chunk) if chunk else points.shape[0]

    votes = np.zeros(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], size):
        block = points[start:start + size]
        rays = np.empty((block.shape[0] * n_directions, 6), dtype=np.float32)
        rays[:, :3] = np.repeat(block, n_directions, axis=0)
        rays[:, 3:] = np.tile(directions, (block.shape[0], 1))
        counts = scene.scene.count_intersections(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)).numpy()
        votes[start:start + size] = (counts.reshape(block.shape[0], n_directions) % 2
                                     == 1).sum(axis=1)
    return votes


def classify_free_space(scene, points, directions=None, chunk=None):
    """Strict-majority odd ray parity over the frozen directions.

    A point is inside iff MORE THAN HALF of the 31 rays cross an odd number of
    triangles. Chunking changes nothing but memory: the vote is per point.
    """
    directions = FROZEN_DIRECTIONS if directions is None else np.asarray(
        directions, dtype=np.float64).reshape(-1, 3)
    majority = directions.shape[0] // 2 + 1
    return odd_parity_votes(scene, points, directions=directions, chunk=chunk) >= majority


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

    Fail-closed about its inputs (r1 review F4): both maps must cover exactly the
    same queries and every oracle must be finite. A missing entry used to default
    to 0.0, which made the band look as if it had created a regression it did not.
    """
    full_keys, band_keys = set(full_height_oracle or {}), set(band_oracle or {})
    if not full_keys:
        raise ValueError("the branch rule needs a non-empty full-height oracle map")
    if full_keys != band_keys:
        missing, extra = sorted(full_keys - band_keys), sorted(band_keys - full_keys)
        raise ValueError(f"the two oracle maps must cover the same queries; the band is "
                         f"missing {missing[:5]} and has {extra[:5]} the full height does not")
    # NaN is missing evidence on either side and is always a refusal. A full-height
    # oracle must also be finite: an empty full-height set is a hard failure
    # upstream. On the BAND side, +inf is meaningful -- it is how an empty z-band
    # candidate set reports itself -- and it disqualifies the branch rather than
    # being replaced by the full-height value (r2 F4 hardening + r3 F3(b)).
    for label, mapping, allow_inf in (("full height", full_height_oracle, False),
                                      ("z band", band_oracle, True)):
        bad = []
        for query, value in mapping.items():
            if value is None:
                bad.append(query)
                continue
            number = float(value)
            if np.isnan(number) or (not allow_inf and not np.isfinite(number)):
                bad.append(query)
        if bad:
            raise ValueError(f"the {label} oracle is not finite for {sorted(bad)[:5]}; a "
                             "branch decision may not be taken over missing values")

    empty_band = sorted(query for query, value in band_oracle.items()
                        if not np.isfinite(float(value)))
    if empty_band:
        return {"branch": "full_height", "n_new_over_threshold": None,
                "n_queries": len(full_keys), "threshold": float(threshold),
                "n_empty_band": len(empty_band), "queries": empty_band[:10],
                "reason": f"{len(empty_band)} queries have an EMPTY z-band candidate set "
                          "(infinite oracle), which disqualifies the band branch"}

    if not band_nonempty:
        return {"branch": "full_height", "n_new_over_threshold": None,
                "n_queries": len(full_keys), "threshold": float(threshold),
                "reason": "at least one query's z-band candidate set was not nonempty"}
    new_over = [query for query in sorted(band_keys)
                if float(band_oracle[query]) > threshold
                and float(full_height_oracle[query]) <= threshold]
    if new_over:
        return {"branch": "full_height", "n_new_over_threshold": len(new_over),
                "queries": new_over[:10], "n_queries": len(full_keys),
                "threshold": float(threshold),
                "reason": f"the z-band created {len(new_over)} queries with e_oracle > "
                          f"{threshold} m that full height did not have"}
    return {"branch": "z_band", "n_new_over_threshold": 0, "n_queries": len(full_keys),
            "threshold": float(threshold),
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


def _room_id_from_metadata_dir(metadata_dir):
    if not metadata_dir:
        return None
    parts = os.path.normpath(str(metadata_dir)).split(os.sep)
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


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


def audit_room_anchors(scene, anchors, clearance=SURFACE_CLEARANCE, eps=EPS,
                       room_id=None):
    """The §1.3 fail-closed room acceptance, as the plan states it.

    Rule 2 applies to EVERY metadata anchor: finite, and inside the free-space
    classification after the ``1e-4 m`` tolerance. Rule 3 names the SOURCE
    anchors specifically: they must also survive the candidate predicate, i.e.
    the 0.20 m source-distribution prior. Receivers are not candidates and are
    not drawn from the source distribution -- their own constraint is the
    >= 0.5 m candidate-distance guard -- so the prior is not applied to them.
    """
    room_id = room_id or _room_id_from_metadata_dir(anchors.get("metadata_dir"))
    report = {"metadata_dir": anchors.get("metadata_dir"), "room_id": room_id,
              "clearance": float(clearance), "eps": float(eps), "rules": {},
              "directions_sha256": FROZEN_DIRECTIONS_SHA256}
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
            failing = np.asarray(points)[~verdict["parity_valid"]]
            block["failing_points"] = [[float(v) for v in point] for point in failing]
            documented = [known_discrepancy(room_id, point, label) for point in failing]
            block["known_discrepancies"] = [entry for entry in documented if entry]
            block["all_failures_documented"] = bool(documented and all(documented))
        if label == "sources" and not bool(verdict["valid"].all()):
            accepted = False
            block["failure"] = ("rule 3: a source anchor fails the candidate predicate "
                                "(parity + the 0.20 m source prior)")
        report["rules"][label] = block
    report["accepted"] = accepted
    return report
