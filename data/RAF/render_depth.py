"""Equirectangular depth rendering for RAF (exp_19, contract section C).

Renders a 256x512 float32 depth panorama at each selected group's source (tx)
position by raycasting the room mesh with open3d, using the SAME pixel->ray
equation the pipeline inverts at load time (``raf_common.equirect_directions``,
which is the exact inverse of ``convert_equirect_to_camera_coord``). Row 0 is the
zenith; there is no flipud anywhere in the RAF path.

Miss policy (Amendment 4): real scanned meshes have holes, so a per-map miss rate
at or below ``DEFAULT_MAX_MISS_RATE`` is repaired by nearest-valid-neighbour
inpainting and RECORDED (count + a hash of the repaired coordinates); only a rate
above the cap aborts. The gauge is PINNED, and canonical renders are gated on the
committed readback record plus mesh-independent receiver-sightline evidence.

Usage:
    python data/RAF/render_depth.py --raf-root /path/to/raf_dataset \\
        --output-dir /path/to/runtime/RAF --rooms EmptyRoom FurnishedRoom
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import RAF_TO_PIPELINE, equirect_directions  # noqa: E402
from publish import PublishTransaction  # noqa: E402
from readback_audit import load_passing_record, record_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEPTH_H = 256
DEPTH_W = 512
# The loader hard-codes this grid (RAF_md -> convert_equirect_to_camera_coord), so
# a map rendered at any other size cannot be consumed: it would pass QA and then
# fail inside metadata loading, where SampleDataset silently substitutes another
# item. Canonical renders are therefore exactly this shape, and anything else has
# to be asked for explicitly and taints the QA record.
CANONICAL_SHAPE = (DEPTH_H, DEPTH_W)
# Plausibility band for a room-scale depth panorama, used to sanity-check the RAF
# maps against the AR/HAA distributions the ViT normalisation expects.
EXPECTED_DEPTH_RANGE_M = (0.05, 50.0)
LANDMARK_BEARING_TOL_DEG = 15.0
# The pipeline's vertical axis is the THIRD component (AR/HAA poses carry the
# height there), so the camera height above the RAF ground plane (y = 0) is
# position_p[2].
HEIGHT_AXIS = 2
# Real scans put content above the nominal y=0 ground (speaker stands, carpet), so
# the nadir distance runs short of the camera height by up to ~0.10 m on measured
# EmptyRoom positions. 0.15 m is the recorded threshold (Amendment 4); it stays a
# WARNING, never an abort.
DEFAULT_FLOOR_TOL = 0.15
# Ray-miss policy (Amendment 4). Real scanned meshes have holes -- FurnishedRoom
# missed 62 of 131,072 rays at a real tx -- so a miss rate at or below this cap is
# repaired by nearest-valid-neighbour inpainting and RECORDED (count + a hash of
# the repaired coordinates). Above the cap the render still aborts: that is a
# broken mesh or a camera outside the room, not a scan hole.
DEFAULT_MAX_MISS_RATE = 0.001
# sha256 of the empty coordinate list, i.e. "nothing was repaired".
EMPTY_FILL_HASH = hashlib.sha256(b"").hexdigest()
# Bearing tie rule (Amendment 4): a second surface within this fraction of the
# farthest distance but further than this angle away in bearing is a TIE, and the
# landmark check declares itself inapplicable. Exactly this configuration produced
# a 96-degree false alarm on the real EmptyRoom mesh.
BEARING_TIE_DISTANCE_FRAC = 0.02
BEARING_TIE_ANGLE_DEG = 20.0
# Slack on a receiver sightline: one pixel of angular quantisation plus mesh
# thickness. A blocked receiver is short by metres, not centimetres.
RX_SIGHTLINE_TOL_M = 0.10
# How many receivers to probe per map, farthest first: long sightlines are the
# discriminating ones.
RX_SIGHTLINE_MAX_RECEIVERS = 8
# Rooms whose receiver sightlines must be unobstructed. FurnishedRoom legitimately
# occludes some tx->rx paths, so there the evidence is recorded, not required.
RX_SIGHTLINE_REQUIRED_ROOMS = ("EmptyRoom",)
# A RAF depth panorama should live in the same range as the AR/HAA maps the ViT
# normalisation was trained on; these bounds widen the measured HAA band
# (0.50 - 11.55 m over the four base rooms) rather than replace it.
SCALE_REFERENCE_TOLERANCE = 3.0
# What the render evidence can and cannot decide about the gauge (Amendment 6, T5).
# Recorded in every QA file so no reader mistakes a passing render for a proof of
# the horizontal assignment.
DETECTABILITY_BOUNDARY = {
    "vertical_axis": "gauge-discriminating (nadir vs tracked height)",
    "horizontal_permutation": "undetectable by render",
    "horizontal_basis": ("pinned by derivation: the documented RAF axis convention "
                         "plus the Metashape export convention, covered by unit "
                         "tests on RAF_TO_PIPELINE -- not by these render checks, "
                         "which a consistently permuted horizontal pair satisfies "
                         "identically"),
    "inconsistency_detection": ("the receiver sightline check catches transforms "
                                "applied inconsistently across mesh, tx and rx"),
}


def load_mesh_pipeline(obj_path):
    """Load ``mesh.obj`` (RAF world coordinates) and move it into the pipeline frame.

    The gauge has determinant -1, so triangle winding inverts; raycast distances
    are unaffected by winding, and no normal-dependent quantity is computed here.
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"mesh not found: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"mesh {obj_path} holds no geometry")
    mesh.vertices = o3d.utility.Vector3dVector(vertices @ RAF_TO_PIPELINE.T)
    return mesh


def build_scene(mesh):
    """One ``RaycastingScene`` per room, reused for every camera (R12).

    The real OBJs are ~215 MB; converting the mesh and rebuilding the acceleration
    structure per camera cost that work ~21 times per room for no benefit.
    """
    if isinstance(mesh, o3d.t.geometry.RaycastingScene):
        return mesh
    tmesh = (mesh if isinstance(mesh, o3d.t.geometry.TriangleMesh)
             else o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def fill_hash(coordinates):
    """sha256 over the repaired pixel coordinates, in their canonical rendering."""
    payload = ";".join(f"{int(r)},{int(c)}" for r, c in coordinates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fill_missing(hits, max_miss_rate=DEFAULT_MAX_MISS_RATE):
    """Repair a scan hole by nearest-valid-neighbour inpainting, or abort.

    A filled pixel is a RECORDED REPAIR, not a fabricated wall: the count and a
    hash of the repaired coordinates travel with the map, so a later reader can
    tell exactly which pixels were never measured. Above ``max_miss_rate`` nothing
    is repaired -- that is a broken mesh or a camera outside the room, and the
    registered abort stands.

    The nearest neighbour is Euclidean in PIXEL space and does not wrap around the
    azimuth seam; for the isolated sub-0.1% holes this policy admits, the nearest
    valid pixel is adjacent either way.
    """
    arr = np.asarray(hits)
    missed = ~np.isfinite(arr)
    count = int(missed.sum())
    total = int(arr.size)
    rate = count / total if total else 0.0
    report = {
        "miss_count": count,
        "miss_rate": rate,
        "max_miss_rate": float(max_miss_rate),
        "within_cap": bool(rate <= max_miss_rate),
        "filled_pixels": [],
        "filled_pixels_sha256": EMPTY_FILL_HASH,
        "hit_mask_sha256": EMPTY_FILL_HASH,
        "miss_mask": missed,
        "n_rays": total,
    }
    if count == 0:
        return arr, report
    if rate > max_miss_rate:
        raise RuntimeError(
            f"depth render missed {count} of {total} rays ({rate:.4%}), above the "
            f"registered cap of {max_miss_rate:.1%} ({DEFAULT_MAX_MISS_RATE}). A scan "
            "hole is repaired and recorded; this is not one -- the mesh is broken or "
            "the camera is outside the room.")
    if count == total:
        raise RuntimeError("depth render missed every ray: nothing to inpaint from")

    rows, cols = np.nonzero(missed)
    coordinates = [[int(r), int(c)] for r, c in zip(rows, cols)]
    report["filled_pixels"] = coordinates
    report["filled_pixels_sha256"] = fill_hash(coordinates)
    report["hit_mask_sha256"] = fill_hash(coordinates)
    # The RAW pre-inpaint mask travels with the report so QA can audit against the
    # evidence itself rather than the report's claims (T7). Not serialised.
    report["miss_mask"] = missed

    # distance_transform_edt measures to the nearest ZERO, i.e. the nearest VALID
    # pixel, and hands back that pixel's index for every position.
    indices = ndimage.distance_transform_edt(missed, return_distances=False,
                                             return_indices=True)
    return arr[tuple(indices)].astype(arr.dtype, copy=False), report


def render_depth(mesh, position_p, h=DEPTH_H, w=DEPTH_W,
                 max_miss_rate=DEFAULT_MAX_MISS_RATE, return_report=False):
    """Euclidean distance to the mesh along every equirect ray from ``position_p``.

    ``mesh`` is a pipeline-frame mesh (see ``load_mesh_pipeline``) or a prebuilt
    ``RaycastingScene`` from ``build_scene``; ``position_p`` is a pipeline-frame
    point. Rays are unit vectors, so open3d's ``t_hit`` is the Euclidean distance
    directly.

    Miss policy (Amendment 4): a per-map miss rate at or below ``max_miss_rate`` is
    repaired by ``fill_missing`` (nearest-valid-neighbour inpainting) and RECORDED
    -- count, rate, and a hash of the repaired coordinates -- while a rate above the
    cap aborts. Real scanned meshes are not watertight (FurnishedRoom missed 62 of
    131,072 rays at a real tx), so demanding a 100% hit rate would reject every
    canonical render; an unrecorded fill, on the other hand, would put a fabricated
    wall into training. Hence repair PLUS provenance.

    Returns the ``[h, w]`` float32 map, or ``(map, miss_report)`` when
    ``return_report`` is set.
    """
    position = np.asarray(position_p, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"position_p must have shape (3,), got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError(f"position_p is not finite: {position.tolist()}")

    scene = build_scene(mesh)

    dirs = equirect_directions(h, w).reshape(-1, 3)
    origins = np.tile(position.astype(np.float32), (dirs.shape[0], 1))
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(h, w)

    try:
        hits, miss_report = fill_missing(hits, max_miss_rate=max_miss_rate)
    except RuntimeError as e:
        raise RuntimeError(f"depth render at {position.tolist()}: {e}")
    if (hits <= 0).any():
        raise RuntimeError(
            f"depth render at {position.tolist()} produced non-positive distances "
            f"(min {float(hits.min())})")
    depth = hits.astype(np.float32)
    return (depth, miss_report) if return_report else depth


def depth_qa(depth, position_p, floor_tol=DEFAULT_FLOOR_TOL, img_h=DEPTH_H,
             img_w=DEPTH_W, canonical=True, miss_report=None):
    """Per-map quality report (plan Rev 2 section 8.3).

    ``passed`` covers the STRUCTURAL checks — shape/dtype, finiteness, positivity,
    the canonical grid, and the miss audit (a repaired map passes only while its
    miss rate is within the registered cap and its repair evidence checks out).
    The floor check (nadir distance vs the camera's height above the RAF ground
    plane) is reported as ``floor_ok`` + a warning instead: an occluder under the
    speaker is physically possible in FurnishedRoom, so it is a fact to record, not
    a structural defect. The gauge-discriminating vertical check lives in
    ``real_mesh_qa``, which has the tracked height to compare against.
    """
    arr = np.asarray(depth)
    position = np.asarray(position_p, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    finite = bool(finite_mask.all())
    positive = bool((arr[finite_mask] > 0).all()) if finite_mask.any() else False
    expected_shape = (int(img_h), int(img_w))
    shape_ok = arr.shape == expected_shape
    dtype_ok = arr.dtype == np.float32
    canonical_grid = expected_shape == CANONICAL_SHAPE

    height = float(position[HEIGHT_AXIS])
    nadir = float(np.median(arr[-1][np.isfinite(arr[-1])])) if shape_ok and finite_mask[-1].any() else float("nan")
    floor_delta = nadir - height
    floor_ok = bool(np.isfinite(floor_delta) and abs(floor_delta) <= floor_tol)

    values = arr[finite_mask]
    warnings = []
    if not shape_ok:
        warnings.append(
            f"depth map is {arr.shape}, expected {expected_shape} "
            f"(the loader consumes exactly {CANONICAL_SHAPE[0]}x{CANONICAL_SHAPE[1]})")
    if not dtype_ok:
        warnings.append(f"depth map dtype is {arr.dtype}, expected float32")
    if canonical and not canonical_grid:
        warnings.append(
            f"non-canonical grid {expected_shape}: this map cannot be loaded by RAF_md")
    misses = None
    if miss_report is not None:
        misses, miss_warnings = audit_miss_report(
            miss_report, arr, canonical=canonical,
            miss_mask=miss_report.get("miss_mask"))
        warnings.extend(miss_warnings)
    if not floor_ok:
        warnings.append(
            f"floor distance at nadir ({nadir:.4f} m) differs from the camera height "
            f"({height:.4f} m) by {floor_delta:.4f} m (> {floor_tol} m)")

    return {
        "shape": list(arr.shape),
        "expected_shape": list(expected_shape),
        "canonical_shape": bool(shape_ok),
        "canonical": bool(canonical and canonical_grid),
        "dtype": str(arr.dtype),
        "finite": finite,
        "positive": positive,
        "hit_rate": float(finite_mask.mean()),
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "camera_height": height,
        "nadir_distance": nadir,
        "floor_delta": float(floor_delta),
        "floor_tol": float(floor_tol),
        "floor_ok": floor_ok,
        # Recorded per map: how many pixels were never measured, and exactly which
        # ones (by hash). Without the coordinates a repaired map is indistinguishable
        # from a measured one.
        "misses": misses,
        "warnings": warnings,
        "passed": bool(finite and positive and shape_ok and dtype_ok
                       and (canonical_grid or not canonical)
                       and (misses is None or misses["audit_ok"])),
    }


# The registered canonical RENDER identity (Amendment 7). Validated before any
# I/O and written into the depth marker, so a Furnished-only render, a loosened
# floor tolerance that disables the vertical gate, or a non-canonical grid can no
# longer publish "canonical": true.
CANONICAL_RENDER_PARAMS = {
    "rooms": ("EmptyRoom", "FurnishedRoom"),
    "img_h": DEPTH_H,
    "img_w": DEPTH_W,
    "floor_tol": DEFAULT_FLOOR_TOL,
    "max_miss_rate": DEFAULT_MAX_MISS_RATE,
}


def render_identity(args):
    """The parameter set a depth publication's marker is bound to."""
    return {
        "rooms": list(args.rooms),
        "img_h": int(args.img_h),
        "img_w": int(args.img_w),
        "floor_tol": float(args.floor_tol),
        "max_miss_rate": float(args.max_miss_rate),
    }


def canonical_render_deviations(args):
    identity = render_identity(args)
    deviations = []
    if tuple(identity["rooms"]) != CANONICAL_RENDER_PARAMS["rooms"]:
        deviations.append(
            f"rooms {identity['rooms']} != {list(CANONICAL_RENDER_PARAMS['rooms'])}")
    for key in ("img_h", "img_w", "floor_tol"):
        if identity[key] != CANONICAL_RENDER_PARAMS[key]:
            deviations.append(f"{key} {identity[key]} != {CANONICAL_RENDER_PARAMS[key]}")
    # the cap may be LOWERED canonically (resolve_miss_cap), never raised
    if identity["max_miss_rate"] > CANONICAL_RENDER_PARAMS["max_miss_rate"]:
        deviations.append(
            f"max_miss_rate {identity['max_miss_rate']} > "
            f"{CANONICAL_RENDER_PARAMS['max_miss_rate']}")
    return deviations


def assert_canonical_render(args):
    """Fail-closed render-parameter gate, run BEFORE any I/O (r5 finding 2)."""
    deviations = canonical_render_deviations(args)
    if deviations:
        raise ValueError(
            "refusing a canonical render with non-registered parameters: "
            + "; ".join(deviations)
            + ". Pass --non-canonical to render an experiment (its artifacts are "
              "tainted).")
    return deviations


def resolve_miss_cap(requested, canonical=True):
    """Miss-cap policy (S2): canonical runs may only LOWER the registered cap.

    Returns ``(cap, taint)``. A looser cap is not a knob a canonical run gets:
    ``--max-miss-rate 0.05`` would otherwise publish 5% inpainted pixels with QA
    reporting them as within cap.
    """
    requested = float(requested)
    if requested <= DEFAULT_MAX_MISS_RATE:
        return requested, []
    if canonical:
        raise ValueError(
            f"refusing a canonical render with --max-miss-rate {requested}: the "
            f"registered cap is {DEFAULT_MAX_MISS_RATE} and may only be LOWERED. "
            "Pass --non-canonical to render with a looser cap (the outputs are "
            "tainted).")
    return requested, [f"miss cap {requested} above the registered {DEFAULT_MAX_MISS_RATE}"]


def audit_miss_report(miss_report, depth, canonical=True, miss_mask=None):
    """Re-derive the miss verdict FROM THE RAW HIT MASK (r5 finding 4).

    The mask is mandatory. Count, coordinates, rate and hash are all computed from
    it here; the report is then required to agree with every one of them. Making
    the mask optional left ``mask_verified=None`` acceptable, so a stale or forged
    zero-miss report -- empty coordinates, the public empty-set hash, the right ray
    count -- passed QA with nothing tying it to this map.
    """
    warnings = []
    n_rays = int(np.asarray(depth).size)
    shape = np.asarray(depth).shape[:2] if np.asarray(depth).ndim >= 2 else (0, 0)

    if miss_mask is None:
        return ({k: v for k, v in miss_report.items()
                 if k not in ("filled_pixels", "miss_mask")}
                | {"n_rays_recomputed": n_rays, "mask_verified": False,
                   "audit_ok": False},
                ["no raw pre-inpaint hit mask: the miss report cannot be tied to "
                 "this map, so it cannot be audited"])

    mask = np.asarray(miss_mask, dtype=bool)
    if mask.shape != shape or mask.size != n_rays:
        return ({k: v for k, v in miss_report.items()
                 if k not in ("filled_pixels", "miss_mask")}
                | {"n_rays_recomputed": n_rays, "mask_verified": False,
                   "audit_ok": False},
                [f"raw hit mask is {mask.shape}, not the map's {shape}"])

    # EVERYTHING below is derived from the mask, never from the report.
    rows, cols = np.nonzero(mask)
    true_coordinates = [[int(r), int(c)] for r, c in zip(rows, cols)]
    true_count = len(true_coordinates)
    true_rate = true_count / n_rays if n_rays else 0.0
    true_hash = fill_hash(true_coordinates)

    declared_cap = float(miss_report.get("max_miss_rate", DEFAULT_MAX_MISS_RATE))
    cap = min(declared_cap, DEFAULT_MAX_MISS_RATE) if canonical else declared_cap
    within = bool(true_rate <= cap)

    reported = [[int(v) for v in c] for c in (miss_report.get("filled_pixels") or [])]
    coordinates_match = reported == true_coordinates
    count_ok = int(miss_report.get("miss_count", -1)) == true_count
    hash_ok = miss_report.get("filled_pixels_sha256") == true_hash
    rays_ok = int(miss_report.get("n_rays", -1)) == n_rays
    cap_claim_ok = miss_report.get("within_cap") is within

    if not coordinates_match:
        warnings.append(
            "repaired coordinates do not match the raw pre-inpaint hit mask")
    if not count_ok:
        warnings.append(
            f"miss report count {miss_report.get('miss_count')} is not the mask's "
            f"{true_count}")
    if not hash_ok:
        warnings.append("repaired-pixel hash does not match the mask's coordinates")
    if not rays_ok:
        warnings.append(
            f"declared ray count {miss_report.get('n_rays')} is not the map's {n_rays}")
    if not cap_claim_ok:
        warnings.append(
            f"miss report claims within_cap={miss_report.get('within_cap')}, "
            f"recomputed {within}")
    if not within:
        warnings.append(
            f"{true_count} rays missed ({true_rate:.4%}), above the {cap:.3%} cap "
            "applied here")
    elif true_count:
        warnings.append(
            f"{true_count} rays missed and were repaired by nearest-valid-neighbour "
            f"inpainting (hash {true_hash[:12]})")

    mask_verified = bool(coordinates_match and count_ok and hash_ok)
    audit = {k: v for k, v in miss_report.items()
             if k not in ("filled_pixels", "miss_mask")}
    audit.update({
        "n_rays_recomputed": n_rays,
        "miss_count_from_mask": true_count,
        "miss_rate_recomputed": true_rate,
        "filled_pixels_sha256_from_mask": true_hash,
        "cap_applied": cap,
        "within_cap_recomputed": within,
        "count_matches_coordinates": bool(count_ok),
        "hash_matches_coordinates": bool(hash_ok),
        "coordinates_match_mask": bool(coordinates_match),
        "ray_count_matches_map": bool(rays_ok),
        "mask_verified": mask_verified,
        # mask_verified must be TRUE: None is no longer tolerated.
        "audit_ok": bool(mask_verified and within and rays_ok and cap_claim_ok),
    })
    return audit, warnings


def direction_to_pixel(direction, img_h=DEPTH_H, img_w=DEPTH_W):
    """Inverse of ``equirect_directions``: which pixel looks along ``direction``.

    theta = atan2(y, x) and phi = -asin(z) invert
    ``dir = (cos(phi)cos(theta), cos(phi)sin(theta), -sin(phi))``; the pixel indices
    then follow from the same half-pixel offsets the forward map uses.
    """
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError(f"direction must be a finite non-zero vector, got {d.tolist()}")
    d = d / norm
    theta = float(np.arctan2(d[1], d[0]))
    phi = float(np.arcsin(np.clip(-d[2], -1.0, 1.0)))
    col = int(np.floor((theta + np.pi) * img_w / (2.0 * np.pi)))
    row = int(np.floor((phi + np.pi / 2.0) * img_h / np.pi))
    return (min(max(row, 0), img_h - 1), col % img_w)


def rx_sightline_check(depth, position_p, rx_positions_p, img_h=DEPTH_H,
                       img_w=DEPTH_W, tol=RX_SIGHTLINE_TOL_M,
                       max_receivers=RX_SIGHTLINE_MAX_RECEIVERS, required=True):
    """MESH-INDEPENDENT gauge evidence (S5).

    The receiver positions come from the tracked pose files, not from the mesh, so
    this asks a question the renderer cannot answer by construction: looking from
    the source toward a receiver that is known to be inside the room, does the
    rendered map reach at least as far as that receiver? A consistently wrong
    horizontal gauge points those bearings at a wall and fails, while the old
    landmark check -- rendered map versus the same transformed mesh -- could not
    see it.

    Receivers are probed farthest-first: long sightlines discriminate, short ones
    barely constrain anything.
    """
    arr = np.asarray(depth, dtype=np.float64)
    tx = np.asarray(position_p, dtype=np.float64)
    rx = np.asarray(rx_positions_p, dtype=np.float64).reshape(-1, 3)
    if rx.size == 0:
        # No receivers to probe: the evidence is ABSENT, which is recorded as
        # ``checked: False`` rather than silently scored either way. The canonical
        # CLI refuses to publish unchecked maps; a unit call simply has no evidence.
        return {"n_receivers": 0, "n_blocked": 0, "passed": True, "checked": False,
                "required": bool(required), "tol_m": float(tol),
                "worst_deficit_m": 0.0, "per_receiver": [],
                "reason": "no receiver positions available"}

    distances = np.linalg.norm(rx - tx, axis=1)
    order = np.argsort(-distances)[:max_receivers]
    per_receiver, blocked, worst = [], 0, 0.0
    for index in order:
        distance = float(distances[index])
        if distance <= tol:
            continue
        row, col = direction_to_pixel((rx[index] - tx) / distance, img_h, img_w)
        # a 3x3 neighbourhood absorbs one pixel of angular quantisation; the
        # question is "is there an unobstructed sightline in roughly this
        # direction", not "exactly through this pixel centre"
        rows = slice(max(row - 1, 0), min(row + 2, img_h))
        cols = [(col + k) % img_w for k in (-1, 0, 1)]
        seen = float(arr[rows][:, cols].max())
        deficit = max(0.0, distance - tol - seen)
        if deficit > 0:
            blocked += 1
            worst = max(worst, distance - seen)
        per_receiver.append({
            "distance_m": distance,
            "depth_m": seen,
            "pixel": [int(row), int(col)],
            "clear": bool(deficit <= 0),
        })
    return {
        "n_receivers": len(per_receiver),
        "n_blocked": blocked,
        "passed": bool(blocked == 0),
        "checked": bool(per_receiver),
        "required": bool(required),
        "tol_m": float(tol),
        "max_receivers": int(max_receivers),
        "worst_deficit_m": float(worst),
        "per_receiver": per_receiver,
    }


def reference_depth_stats(root, pattern="*/depth_images/*.npy"):
    """Depth statistics of an ON-DISK reference corpus (AR or HAA), or its absence.

    ``scale_plausible`` only becomes a gate once a real reference is present; a
    missing corpus is RECORDED as missing rather than silently treated as a pass.
    """
    import glob

    paths = sorted(glob.glob(os.path.join(str(root), pattern))) if root else []
    if not paths:
        return {"available": False, "n_maps": 0, "source": str(root),
                "reason": f"no depth maps under {root!r} (not readable or absent)"}
    lo, hi, means, count = np.inf, -np.inf, [], 0
    for path in paths:
        arr = np.load(path)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lo = min(lo, float(finite.min()))
        hi = max(hi, float(finite.max()))
        means.append(float(finite.mean()))
        count += 1
    return {
        "available": count > 0,
        "n_maps": count,
        "source": str(root),
        "min": float(lo), "max": float(hi),
        "mean_of_means": float(np.mean(means)) if means else None,
        "paths": [os.path.basename(p) for p in paths],
    }


def scale_band(references, tolerance=SCALE_REFERENCE_TOLERANCE):
    """Plausible depth band from whichever references are actually present."""
    available = [r for r in references.values() if r.get("available")]
    if not available:
        return None
    lo = min(r["min"] for r in available) / tolerance
    hi = max(r["max"] for r in available) * tolerance
    return (lo, hi)


def real_mesh_qa(depth, position_p, mesh, img_h=DEPTH_H, img_w=DEPTH_W,
                 bearing_tol_deg=LANDMARK_BEARING_TOL_DEG, bounds_tol=0.05,
                 scene=None, tie_distance_frac=BEARING_TIE_DISTANCE_FRAC,
                 tie_angle_deg=BEARING_TIE_ANGLE_DEG, rx_positions_p=None,
                 rx_sightline_required=True, references=None, tracked_height_m=None,
                 vertical_tol_m=DEFAULT_FLOOR_TOL):
    """Checks that only the REAL mesh can answer (plan Rev 2 section 4.ii, R6).

    * camera containment -- the source must be inside the closed room, or every
      distance in the map is measured from outside it;
    * room bounds -- the reconstructed point cloud must lie inside the mesh's
      bounding box, which catches a mis-scaled or mis-signed gauge;
    * receiver sightlines -- MESH-INDEPENDENT evidence (S5): rendered depth toward
      receivers taken from the tracked pose files must reach those receivers.
      Required in EmptyRoom, recorded in FurnishedRoom (real occlusions);
    * depth scale -- the distribution the ViT normalisation will see, checked
      against the ACTUAL on-disk AR/HAA reference maps when they are present;
    * landmark bearing -- RECORDED ONLY. It compares the rendered map against the
      same transformed mesh, so a consistently wrong gauge satisfies both sides;
      it is diagnostics, never a gate (S5).
    """
    arr = np.asarray(depth, dtype=np.float64)
    position = np.asarray(position_p, dtype=np.float64)
    # ``scene`` is passed in by the CLI so the acceleration structure is built
    # once per room (R12) rather than once per QA call.
    scene = build_scene(mesh) if scene is None else scene
    vertices = np.asarray(mesh.vertices if hasattr(mesh, "vertices")
                          else mesh.vertex["positions"].numpy(), dtype=np.float64)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)

    occupancy = scene.compute_occupancy(
        o3d.core.Tensor(position.astype(np.float32).reshape(1, 3)))
    camera_inside = bool(int(occupancy.numpy().reshape(-1)[0]) == 1)

    dirs = equirect_directions(int(img_h), int(img_w))
    cloud = position.reshape(1, 1, 3) + arr[..., None] * dirs
    points = cloud.reshape(-1, 3)
    bounds_ok = bool(np.all(points >= lo - bounds_tol)
                     and np.all(points <= hi + bounds_tol))

    # Sightline bound: no ray may travel further than the room's bounding box
    # allows in that direction. Derived from the mesh AABB by the slab method here
    # (not from the renderer), so a transposed or mis-scaled gauge breaks it.
    flat_dirs = dirs.reshape(-1, 3).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_lo = (lo - position) / flat_dirs
        t_hi = (hi - position) / flat_dirs
    t_exit = np.nanmin(np.where(flat_dirs > 0, t_hi, np.where(flat_dirs < 0, t_lo, np.inf)),
                       axis=1)
    sightline_slack = float(np.nanmax(arr.reshape(-1) - t_exit))
    sightline_ok = bool(sightline_slack <= bounds_tol)

    flat = int(np.argmax(arr))
    far_dir = dirs.reshape(-1, 3)[flat]
    vertex_dist = np.linalg.norm(vertices - position, axis=1)
    mesh_far = vertices[int(np.argmax(vertex_dist))]
    mesh_dir = mesh_far - position
    map_bearing = float(np.degrees(np.arctan2(far_dir[1], far_dir[0])))
    mesh_bearing = float(np.degrees(np.arctan2(mesh_dir[1], mesh_dir[0])))
    delta = abs((map_bearing - mesh_bearing + 180.0) % 360.0 - 180.0)
    # Applicable only when the farthest surface is unique in DIRECTION (Amendment
    # 4). A second surface within BEARING_TIE_DISTANCE_FRAC of the farthest
    # distance but more than BEARING_TIE_ANGLE_DEG away in bearing is a TIE: which
    # of them argmax picks is arbitrary, so a mismatch says nothing about the
    # gauge. Exactly this configuration produced a 96-degree false alarm on the
    # real EmptyRoom mesh.
    contenders = vertices[vertex_dist >= (1.0 - tie_distance_frac)
                          * float(vertex_dist.max())] - position
    azimuths = np.degrees(np.arctan2(contenders[:, 1], contenders[:, 0]))
    spread = float(np.max(np.abs((azimuths - mesh_bearing + 180.0) % 360.0 - 180.0))) \
        if azimuths.size else 0.0
    bearing_applicable = bool(spread <= tie_angle_deg)
    bearing_ok = bool(delta <= bearing_tol_deg) if bearing_applicable else True

    finite = arr[np.isfinite(arr)]
    scale = {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "p50": float(np.percentile(finite, 50)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
    }
    references = {"AR": {"available": False, "n_maps": 0, "reason": "not provided"},
                  "HAA": {"available": False, "n_maps": 0, "reason": "not provided"}} \
        if references is None else dict(references)
    band = scale_band(references)
    scale_checked = band is not None
    band = band if scale_checked else EXPECTED_DEPTH_RANGE_M
    plausible = bool(finite.size and band[0] <= scale["min"] and scale["max"] <= band[1])

    # T5: the VERTICAL axis is gauge-discriminating and mesh-independent -- the
    # nadir distance comes from the render, the height comes from the tracked pose
    # file. A candidate gauge that puts the wrong RAF axis in the pipeline's
    # vertical slot moves one and not the other, so this gates publication.
    nadir = float(np.median(arr[-1])) if arr.ndim == 2 and arr.shape[0] else float("nan")
    if tracked_height_m is None:
        vertical = {"ok": True, "checked": False, "nadir_m": nadir,
                    "tracked_height_m": None, "delta_m": None,
                    "tol_m": float(vertical_tol_m),
                    "reason": "no tracked height supplied"}
    else:
        delta = nadir - float(tracked_height_m)
        vertical = {"ok": bool(abs(delta) <= vertical_tol_m), "checked": True,
                    "nadir_m": nadir, "tracked_height_m": float(tracked_height_m),
                    "delta_m": float(delta), "tol_m": float(vertical_tol_m)}

    sightline = rx_sightline_check(
        arr, position, rx_positions_p if rx_positions_p is not None else np.zeros((0, 3)),
        img_h=img_h, img_w=img_w, required=rx_sightline_required)

    warnings = []
    if not camera_inside:
        warnings.append(f"camera {position.tolist()} is not inside the room mesh")
    if not bounds_ok:
        warnings.append("reconstructed point cloud leaves the mesh bounding box")
    if not sightline_ok:
        warnings.append(
            f"a sightline overshoots the room bounding box by {sightline_slack:.3f} m: "
            "the gauge may be transposed or mis-scaled")
    if not bearing_applicable:
        warnings.append(
            f"landmark bearing not applicable: a surface within {tie_distance_frac:.0%} "
            f"of the farthest distance lies {spread:.1f} deg away in bearing "
            f"(> {tie_angle_deg} deg), so the farthest direction is a tie-break")
    elif not bearing_ok:
        warnings.append(
            f"landmark bearing disagrees with the mesh by {delta:.1f} deg "
            f"(> {bearing_tol_deg} deg): the gauge may be transposed")
    if not plausible:
        warnings.append(
            f"depth range [{scale['min']}, {scale['max']}] m is outside the "
            f"plausible {band} m band"
            + (" (from the on-disk AR/HAA references)" if scale_checked else ""))
    if vertical["checked"] and not vertical["ok"]:
        warnings.append(
            f"vertical axis: nadir {vertical['nadir_m']:.3f} m vs tracked height "
            f"{vertical['tracked_height_m']:.3f} m (delta {vertical['delta_m']:.3f} m "
            f"> {vertical['tol_m']} m) -- the wrong RAF axis may be in the vertical slot")
    if not sightline["passed"]:
        detail = (f"{sightline['n_blocked']} of {sightline['n_receivers']} receiver "
                  f"sightlines blocked, worst deficit {sightline['worst_deficit_m']:.2f} m")
        warnings.append(detail if sightline["required"]
                        else f"{detail} (recorded only for this room)")

    return {
        "camera_inside": camera_inside,
        "mesh_bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
        "bounds_ok": bounds_ok,
        "bounds_tol": float(bounds_tol),
        "landmark_bearing_deg": map_bearing,
        "mesh_landmark_bearing_deg": mesh_bearing,
        "bearing_delta_deg": float(delta),
        "bearing_tol_deg": float(bearing_tol_deg),
        "bearing_applicable": bearing_applicable,
        "bearing_spread_deg": spread,
        "bearing_tie_distance_frac": float(tie_distance_frac),
        "bearing_tie_angle_deg": float(tie_angle_deg),
        "bearing_ok": bearing_ok,
        "sightline_slack_m": sightline_slack,
        "sightline_ok": sightline_ok,
        "bearing_gates_publication": False,
        "depth_scale": scale,
        "scale_reference": references,
        "scale_band_m": list(band),
        "scale_checked": scale_checked,
        "scale_plausible": plausible,
        "rx_sightline": sightline,
        "vertical_axis": vertical,
        # Recorded honestly (T5): what this evidence can and cannot decide.
        "detectability": DETECTABILITY_BOUNDARY,
        "warnings": warnings,
        # S5: the gauge evidence that gates publication is mesh-INDEPENDENT
        # (receiver sightlines) plus containment/bounds/sightline-bound and, once a
        # real reference corpus is present, the depth scale. The landmark bearing
        # is recorded but never gates.
        "passed": bool(camera_inside and bounds_ok and sightline_ok
                       and (sightline["passed"] or not sightline["required"])
                       and (plausible or not scale_checked)
                       and vertical["ok"]),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render RAF equirect depth panoramas at each group's tx position")
    parser.add_argument('--raf-root', required=True,
                        help="RAF release root (holds 3d_models/<Room>/mesh.obj)")
    parser.add_argument('--output-dir', required=True,
                        help="runtime dataset root written by prepare_data.py")
    parser.add_argument('--rooms', nargs='+', default=['EmptyRoom', 'FurnishedRoom'])
    parser.add_argument('--img-h', type=int, default=DEPTH_H)
    parser.add_argument('--img-w', type=int, default=DEPTH_W)
    parser.add_argument('--floor-tol', type=float, default=DEFAULT_FLOOR_TOL)
    parser.add_argument('--haa-depth-root', default='HAA',
                        help="processed HAA root; its depth_images are the on-disk "
                             "scale reference (S5)")
    parser.add_argument('--ar-depth-root', default=None,
                        help="AcousticRooms depth_map root, if readable; its absence "
                             "is recorded and HAA alone activates the scale check")
    parser.add_argument('--rx-sightline-receivers', type=int,
                        default=RX_SIGHTLINE_MAX_RECEIVERS)
    parser.add_argument('--max-miss-rate', type=float, default=DEFAULT_MAX_MISS_RATE,
                        help="per-map ray-miss rate tolerated and repaired by "
                             "nearest-valid-neighbour inpainting; above it the render "
                             "aborts (Amendment 4)")
    parser.add_argument('--non-canonical', action='store_true',
                        help="synthetic/test mode: allows a grid other than 256x512 "
                             "(unloadable by RAF_md) AND an unauthenticated readback "
                             "record; every artifact this run publishes is tainted")
    parser.add_argument('--readback-record', required=True,
                        help="path to a PASSING, adjudicated raf_readback_record.json; "
                             "canonical depth maps are rendered under a PINNED gauge, "
                             "the pinned one (Amendment 4)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # R4 gate: the maps encode the RAF->pipeline gauge, so they may only be
    # rendered once that gauge has been pinned from the readback audit.
    canonical = not args.non_canonical
    readback = load_passing_record(
        args.readback_record, canonical=canonical,
        expected_raf_root=args.raf_root if canonical else None)
    readback_provenance = record_provenance(args.readback_record, readback,
                                            canonical=canonical)
    logger.info("readback record %s (gauge %s)", readback_provenance["sha256"][:12],
                readback_provenance["gauge_pinned"])

    taint = list(readback_provenance["taint"])
    render_params = render_identity(args)
    render_deviations = canonical_render_deviations(args)
    if canonical:
        assert_canonical_render(args)
    elif render_deviations:
        taint.append("non-registered render parameters: " + "; ".join(render_deviations))
    taint.extend(resolve_miss_cap(args.max_miss_rate, canonical)[1])
    if (args.img_h, args.img_w) != CANONICAL_SHAPE:
        # canonical mode already refused this in assert_canonical_render
        taint.append(f"non-canonical grid {args.img_h}x{args.img_w}: unusable by RAF_md")

    # S3: one transaction over every room's depth directory, so two rooms can
    # never be published under different generations.
    publish_txn = PublishTransaction(args.output_dir, kind="depth")
    expectations, records = {}, {}
    # S5: real on-disk reference statistics, read once per run.
    references = {
        "HAA": reference_depth_stats(args.haa_depth_root),
        "AR": reference_depth_stats(args.ar_depth_root, pattern="**/*.npy")
        if args.ar_depth_root else
        {"available": False, "n_maps": 0, "source": None,
         "reason": "AR depth_map root not provided (often unreadable from this mount)"},
    }
    logger.info("scale references: HAA %s, AR %s",
                references["HAA"].get("n_maps"), references["AR"].get("n_maps"))
    sightline_policy = {room: ("required" if room in RX_SIGHTLINE_REQUIRED_ROOMS
                               else "recorded") for room in args.rooms}
    for room in args.rooms:
        mesh = load_mesh_pipeline(os.path.join(args.raf_root, "3d_models", room, "mesh.obj"))
        meta_path = os.path.join(args.output_dir, room, "metadata", "groups_metadata.json")
        with open(meta_path) as f:
            groups_meta = json.load(f)

        # Receiver positions come from the TRACKED POSE FILES, never from the mesh:
        # that independence is what makes the sightline check real evidence (S5).
        poses_path = os.path.join(args.output_dir, room, "metadata", "poses_metadata.json")
        rx_positions = np.zeros((0, 3))
        if os.path.isfile(poses_path):
            with open(poses_path) as f:
                rx_positions = np.asarray(
                    [entry["rx_p"] for entry in json.load(f).values()], dtype=np.float64)
        sightline_required = room in RX_SIGHTLINE_REQUIRED_ROOMS

        depth_dir = os.path.join(args.output_dir, room, "depth_images")

        # R12: one conversion + one acceleration structure per room.
        t0 = time.perf_counter()
        scene = build_scene(mesh)
        scene_build_s = time.perf_counter() - t0

        # R7: every map and the QA record are staged and swapped in together, so
        # a failure mid-room can never leave half a depth set beside a QA file
        # that describes the other half.
        staged = publish_txn.stage(depth_dir)
        maps, failed, warned, bearings = {}, [], [], {}
        render_s = 0.0
        for group_key, entry in groups_meta.items():
            position = np.asarray(entry["tx_xyz_p"], dtype=np.float64)
            t1 = time.perf_counter()
            depth, miss_report = render_depth(scene, position, h=args.img_h,
                                              w=args.img_w,
                                              max_miss_rate=args.max_miss_rate,
                                              return_report=True)
            render_s += time.perf_counter() - t1
            np.save(staged.path(entry["depth_file"]), depth)

            qa = depth_qa(depth, position, floor_tol=args.floor_tol,
                          img_h=args.img_h, img_w=args.img_w, canonical=canonical,
                          miss_report=miss_report)
            qa["depth_file"] = entry["depth_file"]
            # R6: the checks only the real mesh can answer, fail-closed.
            # r5 finding 5: the tracked height comes from the PUBLISHED RAW RAF Y,
            # never from position[HEIGHT_AXIS] -- that is the gauge-transformed
            # vector this very map was rendered from, so it cannot witness a wrong
            # vertical assignment.
            if "tx_height_raf_m" not in entry:
                raise ValueError(
                    f"{room} group {group_key}: groups_metadata carries no "
                    "tx_height_raf_m. The vertical gauge check needs the raw RAF "
                    "height; re-run data/RAF/prepare_data.py to republish it.")
            qa["real_mesh"] = real_mesh_qa(
                depth, position, mesh, img_h=args.img_h, img_w=args.img_w, scene=scene,
                rx_positions_p=rx_positions, rx_sightline_required=sightline_required,
                references=references,
                tracked_height_m=float(entry["tx_height_raf_m"]),
                vertical_tol_m=args.floor_tol)
            if canonical and not qa["real_mesh"]["rx_sightline"]["checked"]:
                qa["warnings"].append(
                    "no receiver sightline evidence: canonical maps require it")
                qa["passed"] = False
            qa["warnings"] = qa["warnings"] + qa["real_mesh"]["warnings"]
            qa["passed"] = bool(qa["passed"] and qa["real_mesh"]["passed"])
            bearings[group_key] = qa["real_mesh"]["landmark_bearing_deg"]
            maps[group_key] = qa
            if not qa["passed"]:
                failed.append(group_key)
            if qa["warnings"]:
                warned.append(group_key)
            logger.info("%s %s: range [%.3f, %.3f] m, nadir %.3f m (height %.3f m), "
                        "bearing %.1f deg (mesh %.1f deg%s), %d rays repaired",
                        room, group_key, qa["min"], qa["max"], qa["nadir_distance"],
                        qa["camera_height"], qa["real_mesh"]["landmark_bearing_deg"],
                        qa["real_mesh"]["mesh_landmark_bearing_deg"],
                        "" if qa["real_mesh"]["bearing_applicable"] else ", tie",
                        miss_report["miss_count"])

        record = {
            "room": room,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "canonical": canonical,
            "taint": taint,
            "floor_tol": args.floor_tol,
            "max_miss_rate": args.max_miss_rate,
            "n_maps": len(maps),
            "n_failed": len(failed),
            "n_warned": len(warned),
            "readback_record": readback_provenance,
            "scale_reference": references,
            "detectability": DETECTABILITY_BOUNDARY,
            "rx_sightline_policy": {room: ("required" if room in RX_SIGHTLINE_REQUIRED_ROOMS
                                           else "recorded")
                                    for room in ("EmptyRoom", "FurnishedRoom")},
            # Recorded per room so the two rooms' landmark bearings can be compared
            # across rooms at the run rung (a shared gauge error would show up as a
            # consistent offset in both).
            "landmark_bearings": bearings,
            "render_benchmark": {
                "n_maps": len(maps),
                "scene_build_s": scene_build_s,
                "total_render_s": render_s,
                "mean_render_s": render_s / len(maps) if maps else 0.0,
                "maps_per_s": len(maps) / render_s if render_s > 0 else None,
                "rays_per_map": int(args.img_h) * int(args.img_w),
            },
            "maps": maps,
        }
        with open(staged.path("raf_depth_qa.json"), "w") as f:
            json.dump(record, f, indent=4, allow_nan=False)

        if failed:
            # Nothing is published: every staging directory is discarded.
            publish_txn.cleanup()
            raise RuntimeError(f"{room}: {len(failed)} depth maps failed QA: {failed}")

        expectations[staged.dest_root] = (
            [entry["depth_file"] for entry in groups_meta.values()]
            + ["raf_depth_qa.json"])
        records[room] = record
        logger.info("%s: staged %d depth maps in %.2fs (scene build %.2fs), %d warnings",
                    room, len(maps), render_s, scene_build_s, len(warned))

    marker = publish_txn.commit(expectations=expectations, validate_json=True,
                                extra={"canonical": canonical, "taint": taint,
                                       "parameters": render_params,
                                       "canonical_parameters": not render_deviations,
                                       "readback_record": readback_provenance})
    logger.info("published generation %s over %d depth directories",
                marker["generation"][:12], len(marker["roots"]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
