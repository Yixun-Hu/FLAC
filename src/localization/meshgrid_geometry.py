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

The direction set is SELECTED, not assumed: exp_22 is self-authoritative about
it (Yixun, 2026-08-25), and the registered rule -- the smallest generator seed
whose 31 directions classify every metadata anchor of all 16 required rooms as
interior at >= 16/31 -- returned **seed 1** over the real meshes (700 anchors).
Seed 0 failed exactly one of them, MeetingRoom_idx_32's receiver, at 15/31.

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
#: SELECTED, not merely generated: exp_22 is self-authoritative about these
#: constants (Yixun, 2026-08-25), and the registered rule -- see
#: :data:`DIRECTION_SELECTION_RULE` -- picks the smallest generator seed whose
#: 31 directions classify EVERY metadata source and receiver anchor in all 16
#: required rooms as interior at >= 16/31 odd parity. The sweep over the real
#: meshes (700 anchors) returned **seed 1**: seed 0 fails exactly one anchor,
#: MeetingRoom_idx_32's receiver [2.26, 0.48, 1.2], at 15/31.
#:
#: The literals ARE the protocol; ``build_directions`` is the provenance of how
#: they were produced, and a test asserts the two still agree.
FROZEN_DIRECTIONS_LITERAL = (
    (-0.28579100209807223, -0.05576932726592917, 0.9566678029786961),
    (0.27987880138315685, 0.3787068503595556, 0.8821842086702032),
    (0.06057012745266738, -0.5737410134156589, 0.8167940433090465),
    (-0.48511696581013153, 0.42891368921398, 0.7620331860805178),
    (0.7044296389112167, 0.0124298426948464, 0.7096649792923378),
    (-0.5516712219738683, -0.5235737190540501, 0.6492529734716426),
    (0.05700174717725607, 0.8142165749595468, 0.5777561508801832),
    (0.5381288881952615, -0.6768212996709899, 0.5023248231984997),
    (-0.8899014828832577, 0.14126757688142563, 0.4337266679421843),
    (0.7727927879003984, 0.510958884522509, 0.37644697541173466),
    (-0.22939672068693973, -0.9176245280404595, 0.32456489347525325),
    (-0.45916776871848897, 0.8472134951792204, 0.2671970317131041),
    (0.9264172915101602, -0.31991768101777546, 0.1985036003784007),
    (-0.9104968856226278, -0.3947813019531085, 0.12305748615872275),
    (0.4072564987285918, 0.9118527614949657, 0.051639961244502916),
    (0.3160592512489688, -0.9486998227036378, -0.008671568602473043),
    (-0.8722946271518841, 0.4851595640025336, -0.06101049826732268),
    (0.9661869842750447, 0.230355726565393, -0.11584019447508148),
    (-0.5513132478150761, -0.8143535315068475, -0.18133402467799206),
    (-0.14112753665552724, 0.956356768709187, -0.25586080071332873),
    (0.7325741791941771, -0.5956299569999376, -0.3294844856775967),
    (-0.9178975518752759, -0.054194029305257685, -0.39309934043322453),
    (0.6229100855868855, 0.6421458306172021, -0.44680169817837834),
    (-0.024737458828122498, -0.8657867073005941, -0.4998013961488327),
    (-0.5410963165210391, 0.6256373933823063, -0.5619542937367387),
    (0.7675555361917533, -0.08951726942986739, -0.634700840818106),
    (-0.5710751811652167, -0.41248006905657203, -0.7097417347797722),
    (0.1273897418662951, 0.6168940475618422, -0.7766682610677523),
    (0.28386385325612706, -0.4757223485485104, -0.8325320173459239),
    (-0.44680086470936253, 0.13390395765672902, -0.8845556610066044),
    (0.29833757357514296, 0.14428002052873334, -0.9434924312730465),
)

FROZEN_DIRECTIONS = np.array(FROZEN_DIRECTIONS_LITERAL, dtype=np.float64)

#: sha256 over ``FROZEN_DIRECTIONS.tobytes()``; recorded in every manifest so a
#: drifted direction set is visible in the artifacts, not only in the code.
FROZEN_DIRECTIONS_SHA256 = (
    "79544f2dbc880a37a4826aa527d40e99a3e54ce849cfd0ec9f1c6e847c528a8d")

#: The generator seed the pinned literals correspond to -- the selection's answer.
FROZEN_DIRECTIONS_SEED = 1


#: RESOLVED by the registered selection, kept as provenance.
#:
#: The anchor this records is not a defect in the geometry: it is what the
#: PREVIOUS direction set (seed 0) did, and it is why the set was selected rather
#: than assumed. Under the pinned seed-1 set the same anchor classifies interior
#: at 16/31, and all 700 anchors of the 16 rooms pass.
KNOWN_PARITY_DISCREPANCIES = ()

RESOLVED_PARITY_DISCREPANCIES = (
    {"room_id": "MeetingRoom/MeetingRoom_idx_32", "kind": "receivers",
     "point": [2.26, 0.48, 1.2], "odd_votes_under_previous_pin": 15, "n_directions": 31,
     "previous_seed": 0,
     "previous_sha256": "9ab4339fa893c00dca817b901a149c292b080d0e6971c90f0b8b0b88e858c261",
     "surface_distance_m": 0.25005,
     "status": "resolved by the registered anchor-driven selection (seed 0 -> seed 1)",
     "note": "the only anchor of 700 that the seed-0 set rejected; not a clearance or "
             "epsilon effect (0.25005 m from the surface). The selection rule was fixed "
             "before the sweep and applied to every anchor of all 16 rooms."},
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
            # -inf is not a meaningful oracle anywhere: only POSITIVE infinity,
            # and only on the band side, carries the "empty set" meaning (r3 F4)
            if np.isnan(number) or number < 0 or (not np.isfinite(number)
                                                  and not (allow_inf and number > 0)):
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


# --------------------------------------------------------------------------- #
# anchor-driven direction-set selection (Yixun directive, 2026-08-25)
# --------------------------------------------------------------------------- #
#: The REGISTERED selection rule. exp_22 is self-authoritative about these
#: constants: the 31 directions exist only to test interior free space, and the
#: metadata anchors are known-interior points by construction, so the classifier
#: must agree with them. Choosing the set BEFORE any generation, by a rule fixed
#: in advance, is pre-registration of a geometry constant -- the same class as
#: the plan's own pre-generation z-branch rule -- not tuning against a result.
DIRECTION_SELECTION_RULE = (
    "the smallest generator seed s >= 0 whose build_directions(31, seed=s) set gives "
    "strict-majority odd parity (>= 16 of 31) for EVERY metadata source AND receiver anchor "
    "in ALL 16 required rooms"
)


def anchor_scenes(rooms, mesh_root, metadata_root, resolve=None):
    """Load each room's mesh ONCE and collect its anchors.

    The sweep then costs ~116 ray-parity classifications per room per seed, not a
    lattice: loading is the expensive part and it happens once.
    """
    resolve = resolve or _default_resolve
    out = {}
    for room_id in rooms:
        scene_name, scene_id = room_id.split("/")
        scene = load_raycast_scene(resolve(room_id, mesh_root))
        anchors = metadata_anchors(os.path.join(metadata_root, scene_name, scene_id))
        out[room_id] = {
            "scene": scene,
            "sources": np.asarray(anchors["sources"], dtype=np.float64).reshape(-1, 3),
            "receivers": np.asarray(anchors["receivers"], dtype=np.float64).reshape(-1, 3),
        }
    return out


def _default_resolve(room_id, mesh_root):
    scene_name, scene_id = room_id.split("/")
    return os.path.join(mesh_root, scene_name, f"{scene_id}.obj")


def evaluate_direction_seed(seed, scenes, n_directions=N_DIRECTIONS, votes_fn=None):
    """Does this seed's set classify EVERY anchor of every room as interior?"""
    votes_fn = votes_fn or (lambda _seed, directions, scene, points:
                            odd_parity_votes(scene, points, directions=directions))
    directions = build_directions(n_directions, seed=seed)
    majority = n_directions // 2 + 1
    rooms, worst, failures = {}, None, []
    for room_id in sorted(scenes):
        entry = scenes[room_id]
        room_report = {}
        for label in ("sources", "receivers"):
            points = entry[label]
            if points.size == 0:
                room_report[label] = {"n": 0, "min_votes": None, "n_failing": 0}
                continue
            votes = np.asarray(votes_fn(seed, directions, entry["scene"], points),
                               dtype=np.int64)
            failing = np.flatnonzero(votes < majority)
            room_report[label] = {
                "n": int(points.shape[0]), "min_votes": int(votes.min()),
                "n_failing": int(failing.size),
                "failing_points": [[float(v) for v in points[i]] for i in failing[:5]],
                "failing_votes": [int(votes[i]) for i in failing[:5]],
            }
            worst = int(votes.min()) if worst is None else min(worst, int(votes.min()))
            for index in failing:
                failures.append({"room_id": room_id, "kind": label,
                                 "point": [float(v) for v in points[index]],
                                 "odd_votes": int(votes[index])})
        rooms[room_id] = room_report
    return {"seed": int(seed), "n_directions": int(n_directions), "majority": majority,
            "ok": not failures, "min_votes": worst, "n_failures": len(failures),
            "failures": failures[:10], "rooms": rooms,
            "directions_sha256": hashlib.sha256(directions.tobytes()).hexdigest()}


def select_direction_seed(scenes, max_seed=64, n_directions=N_DIRECTIONS, votes_fn=None,
                          on_seed=None):
    """The registered rule: the SMALLEST passing seed, or a refusal.

    Deterministic: seeds are tried in ascending order and the first that passes
    every anchor wins, so the same rooms always yield the same set.
    """
    attempts = []
    for seed in range(int(max_seed) + 1):
        report = evaluate_direction_seed(seed, scenes, n_directions=n_directions,
                                         votes_fn=votes_fn)
        attempts.append({"seed": seed, "ok": report["ok"], "min_votes": report["min_votes"],
                        "n_failures": report["n_failures"],
                         "first_failures": report["failures"][:3]})
        if on_seed is not None:
            on_seed(report)
        if report["ok"]:
            return {"seed": seed, "directions": build_directions(n_directions, seed=seed),
                    "report": report, "attempts": attempts, "rule": DIRECTION_SELECTION_RULE}
    raise ValueError(f"no seed in [0, {max_seed}] classifies every anchor as interior under "
                     f"the registered rule; the closest attempts were "
                     f"{sorted(attempts, key=lambda a: a['n_failures'])[:3]}")


# --------------------------------------------------------------------------- #
# published-artifact verifiers (shared by the G1 audit tool and the I1 engine)
# --------------------------------------------------------------------------- #
def manifest_json_sha256(payload):
    """sha256 over a manifest payload's exact published bytes."""
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode()
                          + b"\n").hexdigest()


def coordinates_digest(array):
    """sha256 over the exact float64 coordinate bytes."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


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
        digest = manifest_json_sha256(payload)
        if digest != entry.get("candidate_manifest_sha256"):
            reasons.append(f"{room_id}: the manifest hashes to {digest[:12]}... but the "
                           f"report records {str(entry.get('candidate_manifest_sha256'))[:12]}"
                           "...; it was edited after publication")
        verdict = verify_room_manifest(path, out_dir=out_dir)
        if not verdict["ok"]:
            reasons.append(f"{room_id}: {verdict['reasons'][0]}")
    return {"ok": not reasons, "reasons": reasons, "n_rooms": len(report.get("rooms") or {}),
            "report": report_path}
