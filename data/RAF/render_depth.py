"""Equirectangular depth rendering for RAF (exp_19, contract section C).

Renders a 256x512 float32 depth panorama at each selected group's source (tx)
position by raycasting the room mesh with open3d, using the SAME pixel->ray
equation the pipeline inverts at load time (``raf_common.equirect_directions``,
which is the exact inverse of ``convert_equirect_to_camera_coord``). Row 0 is the
zenith; there is no flipud anywhere in the RAF path.

Usage:
    python data/RAF/render_depth.py --raf-root /path/to/raf_dataset \\
        --output-dir /path/to/runtime/RAF --rooms EmptyRoom FurnishedRoom
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import open3d as o3d

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import RAF_TO_PIPELINE, equirect_directions  # noqa: E402
from publish import StagedPublish  # noqa: E402
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
DEFAULT_FLOOR_TOL = 0.05


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


def render_depth(mesh, position_p, h=DEPTH_H, w=DEPTH_W):
    """Euclidean distance to the mesh along every equirect ray from ``position_p``.

    ``mesh`` is a pipeline-frame mesh (see ``load_mesh_pipeline``) or a prebuilt
    ``RaycastingScene`` from ``build_scene``; ``position_p`` is a pipeline-frame
    point. Rays are unit vectors, so open3d's ``t_hit`` is the Euclidean distance
    directly.

    Registered miss policy: ANY ray that fails to hit aborts the render with a
    report. There is no silent fill — an unhit ray means the camera is outside the
    watertight room or the mesh is broken, and a filled value would propagate into
    training as a plausible-looking wall.
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

    missed = ~np.isfinite(hits)
    if missed.any():
        rows, cols = np.nonzero(missed)
        sample = ", ".join(f"({int(r)},{int(c)})" for r, c in
                           zip(rows[:5], cols[:5]))
        raise RuntimeError(
            f"depth render at {position.tolist()} missed {int(missed.sum())} of "
            f"{h * w} rays (first: {sample}). The registered policy is to abort: a "
            "filled value would enter training as a fabricated wall.")
    if (hits <= 0).any():
        raise RuntimeError(
            f"depth render at {position.tolist()} produced non-positive distances "
            f"(min {float(hits.min())})")
    return hits.astype(np.float32)


def depth_qa(depth, position_p, floor_tol=DEFAULT_FLOOR_TOL, img_h=DEPTH_H,
             img_w=DEPTH_W, canonical=True):
    """Per-map quality report (plan Rev 2 section 8.3).

    ``passed`` covers the STRUCTURAL checks only — shape/dtype, finiteness,
    positivity, full hit rate. The floor check (nadir distance vs the camera's
    height above the RAF ground plane) is reported as ``floor_ok`` + a warning
    instead: an occluder under the speaker is physically possible in
    FurnishedRoom, so it is a fact to record, not a structural defect.
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
        "warnings": warnings,
        "passed": bool(finite and positive and shape_ok and dtype_ok
                       and (canonical_grid or not canonical)),
    }


def real_mesh_qa(depth, position_p, mesh, img_h=DEPTH_H, img_w=DEPTH_W,
                 bearing_tol_deg=LANDMARK_BEARING_TOL_DEG, bounds_tol=0.05,
                 scene=None):
    """Checks that only the REAL mesh can answer (plan Rev 2 section 4.ii, R6).

    * camera containment -- the source must be inside the closed room, or every
      distance in the map is measured from outside it;
    * room bounds -- the reconstructed point cloud must lie inside the mesh's
      bounding box, which catches a mis-scaled or mis-signed gauge;
    * landmark bearing -- the direction of the map's LONGEST sightline must agree
      with the direction of the mesh's farthest surface point. This is the
      non-circular gauge check: a transposed or flipped axis swings it by ~90/180
      degrees while every synthetic-box test still passes;
    * depth scale -- the distribution the ViT normalisation will see, recorded
      against the AR/HAA reference band.
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
    # Applicable only when the farthest surface is unique in DIRECTION. A
    # symmetric room has several equally distant corners, and then a bearing
    # mismatch is a tie-break, not evidence about the gauge -- asserting on it
    # would be a false alarm rather than a check.
    contenders = vertices[vertex_dist >= 0.99 * float(vertex_dist.max())] - position
    azimuths = np.degrees(np.arctan2(contenders[:, 1], contenders[:, 0]))
    spread = float(np.max(np.abs((azimuths[:, None] - azimuths[None, :] + 180.0)
                                 % 360.0 - 180.0))) if azimuths.size else 0.0
    bearing_applicable = bool(spread <= bearing_tol_deg)
    bearing_ok = bool(delta <= bearing_tol_deg) if bearing_applicable else True

    finite = arr[np.isfinite(arr)]
    scale = {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "p50": float(np.percentile(finite, 50)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
    }
    plausible = bool(finite.size and EXPECTED_DEPTH_RANGE_M[0] <= scale["min"]
                     and scale["max"] <= EXPECTED_DEPTH_RANGE_M[1])

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
            f"landmark bearing not applicable: the farthest surface spans {spread:.1f} deg "
            "of azimuth (a symmetric room), so the direction is a tie-break")
    elif not bearing_ok:
        warnings.append(
            f"landmark bearing disagrees with the mesh by {delta:.1f} deg "
            f"(> {bearing_tol_deg} deg): the gauge may be transposed")
    if not plausible:
        warnings.append(
            f"depth range [{scale['min']}, {scale['max']}] m is outside the "
            f"expected {EXPECTED_DEPTH_RANGE_M} m band")

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
        "bearing_ok": bearing_ok,
        "sightline_slack_m": sightline_slack,
        "sightline_ok": sightline_ok,
        "depth_scale": scale,
        "scale_reference": {"AR": None, "HAA": None,
                            "expected_range_m": list(EXPECTED_DEPTH_RANGE_M)},
        "scale_plausible": plausible,
        "warnings": warnings,
        "passed": bool(camera_inside and bounds_ok and bearing_ok and sightline_ok),
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
    parser.add_argument('--non-canonical', action='store_true',
                        help="allow a grid other than 256x512; such maps CANNOT be "
                             "loaded by RAF_md and their QA record is tainted")
    parser.add_argument('--readback-record', required=True,
                        help="path to a PASSING, adjudicated raf_readback_record.json; "
                             "canonical depth maps are rendered under a PINNED gauge, "
                             "never a candidate one")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # R4 gate: the maps encode the RAF->pipeline gauge, so they may only be
    # rendered once that gauge has been pinned from the readback audit.
    readback_provenance = record_provenance(args.readback_record,
                                            load_passing_record(args.readback_record))
    logger.info("readback record %s (gauge %s)", readback_provenance["sha256"][:12],
                readback_provenance["gauge_pinned"])

    canonical = (args.img_h, args.img_w) == CANONICAL_SHAPE
    if not canonical and not args.non_canonical:
        raise ValueError(
            f"refusing to render a {args.img_h}x{args.img_w} grid: RAF_md consumes "
            f"exactly {CANONICAL_SHAPE[0]}x{CANONICAL_SHAPE[1]}, so these maps would "
            "pass QA and then fail inside metadata loading, where the dataloader "
            "silently substitutes another item. Pass --non-canonical to render them "
            "anyway (the QA record is tainted).")
    taint = [] if canonical else [
        f"non-canonical grid {args.img_h}x{args.img_w}: unusable by RAF_md"]

    for room in args.rooms:
        mesh = load_mesh_pipeline(os.path.join(args.raf_root, "3d_models", room, "mesh.obj"))
        meta_path = os.path.join(args.output_dir, room, "metadata", "groups_metadata.json")
        with open(meta_path) as f:
            groups_meta = json.load(f)

        depth_dir = os.path.join(args.output_dir, room, "depth_images")

        # R12: one conversion + one acceleration structure per room.
        t0 = time.perf_counter()
        scene = build_scene(mesh)
        scene_build_s = time.perf_counter() - t0

        # R7: every map and the QA record are staged and swapped in together, so
        # a failure mid-room can never leave half a depth set beside a QA file
        # that describes the other half.
        staged = StagedPublish(depth_dir)
        maps, failed, warned, bearings = {}, [], [], {}
        render_s = 0.0
        for group_key, entry in groups_meta.items():
            position = np.asarray(entry["tx_xyz_p"], dtype=np.float64)
            t1 = time.perf_counter()
            depth = render_depth(scene, position, h=args.img_h, w=args.img_w)
            render_s += time.perf_counter() - t1
            np.save(staged.path(entry["depth_file"]), depth)

            qa = depth_qa(depth, position, floor_tol=args.floor_tol,
                          img_h=args.img_h, img_w=args.img_w, canonical=canonical)
            qa["depth_file"] = entry["depth_file"]
            # R6: the checks only the real mesh can answer, fail-closed.
            qa["real_mesh"] = real_mesh_qa(depth, position, mesh, img_h=args.img_h,
                                           img_w=args.img_w, scene=scene)
            qa["warnings"] = qa["warnings"] + qa["real_mesh"]["warnings"]
            qa["passed"] = bool(qa["passed"] and qa["real_mesh"]["passed"])
            bearings[group_key] = qa["real_mesh"]["landmark_bearing_deg"]
            maps[group_key] = qa
            if not qa["passed"]:
                failed.append(group_key)
            if qa["warnings"]:
                warned.append(group_key)
            logger.info("%s %s: range [%.3f, %.3f] m, nadir %.3f m (height %.3f m), "
                        "bearing %.1f deg (mesh %.1f deg)", room, group_key, qa["min"],
                        qa["max"], qa["nadir_distance"], qa["camera_height"],
                        qa["real_mesh"]["landmark_bearing_deg"],
                        qa["real_mesh"]["mesh_landmark_bearing_deg"])

        record = {
            "room": room,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "canonical": canonical,
            "taint": taint,
            "floor_tol": args.floor_tol,
            "n_maps": len(maps),
            "n_failed": len(failed),
            "n_warned": len(warned),
            "readback_record": readback_provenance,
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
            # Nothing is published: the staging directory is discarded by cleanup.
            staged.cleanup()
            raise RuntimeError(f"{room}: {len(failed)} depth maps failed QA: {failed}")

        manifest = staged.commit(
            expected=[entry["depth_file"] for entry in groups_meta.values()]
                     + ["raf_depth_qa.json"],
            validate_json=True)
        if manifest["n_files"] != len(maps) + 1:
            raise RuntimeError(
                f"{room}: published {manifest['n_files']} files for {len(maps)} maps "
                "plus one QA record; the publish is not the set that was rendered")
        logger.info("%s: published %d depth maps in %.2fs (scene build %.2fs), "
                    "%d warnings, QA -> %s", room, len(maps), render_s, scene_build_s,
                    len(warned), os.path.join(depth_dir, "raf_depth_qa.json"))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
