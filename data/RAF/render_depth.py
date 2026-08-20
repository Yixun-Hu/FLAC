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

import numpy as np
import open3d as o3d

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import RAF_TO_PIPELINE, equirect_directions  # noqa: E402
from readback_audit import load_passing_record, record_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEPTH_H = 256
DEPTH_W = 512
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


def render_depth(mesh, position_p, h=DEPTH_H, w=DEPTH_W):
    """Euclidean distance to the mesh along every equirect ray from ``position_p``.

    ``mesh`` must already be in the PIPELINE frame (see ``load_mesh_pipeline``) and
    ``position_p`` is a pipeline-frame point. Rays are unit vectors, so open3d's
    ``t_hit`` is the Euclidean distance directly.

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

    if isinstance(mesh, o3d.t.geometry.TriangleMesh):
        tmesh = mesh
    else:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

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


def depth_qa(depth, position_p, floor_tol=DEFAULT_FLOOR_TOL):
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
    shape_ok = arr.ndim == 2
    dtype_ok = arr.dtype == np.float32

    height = float(position[HEIGHT_AXIS])
    nadir = float(np.median(arr[-1][np.isfinite(arr[-1])])) if shape_ok and finite_mask[-1].any() else float("nan")
    floor_delta = nadir - height
    floor_ok = bool(np.isfinite(floor_delta) and abs(floor_delta) <= floor_tol)

    values = arr[finite_mask]
    warnings = []
    if not floor_ok:
        warnings.append(
            f"floor distance at nadir ({nadir:.4f} m) differs from the camera height "
            f"({height:.4f} m) by {floor_delta:.4f} m (> {floor_tol} m)")

    return {
        "shape": list(arr.shape),
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
        "passed": bool(finite and positive and shape_ok and dtype_ok),
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

    for room in args.rooms:
        mesh = load_mesh_pipeline(os.path.join(args.raf_root, "3d_models", room, "mesh.obj"))
        meta_path = os.path.join(args.output_dir, room, "metadata", "groups_metadata.json")
        with open(meta_path) as f:
            groups_meta = json.load(f)

        depth_dir = os.path.join(args.output_dir, room, "depth_images")
        os.makedirs(depth_dir, exist_ok=True)

        maps, failed, warned = {}, [], []
        for group_key, entry in groups_meta.items():
            depth = render_depth(mesh, np.asarray(entry["tx_xyz_p"], dtype=np.float64),
                                 h=args.img_h, w=args.img_w)
            np.save(os.path.join(depth_dir, entry["depth_file"]), depth)
            qa = depth_qa(depth, entry["tx_xyz_p"], floor_tol=args.floor_tol)
            qa["depth_file"] = entry["depth_file"]
            maps[group_key] = qa
            if not qa["passed"]:
                failed.append(group_key)
            if qa["warnings"]:
                warned.append(group_key)
            logger.info("%s %s: range [%.3f, %.3f] m, nadir %.3f m (height %.3f m)",
                        room, group_key, qa["min"], qa["max"], qa["nadir_distance"],
                        qa["camera_height"])

        record = {
            "room": room,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "floor_tol": args.floor_tol,
            "n_maps": len(maps),
            "n_failed": len(failed),
            "n_warned": len(warned),
            "readback_record": readback_provenance,
            "maps": maps,
        }
        qa_path = os.path.join(depth_dir, "raf_depth_qa.json")
        with open(qa_path, "w") as f:
            json.dump(record, f, indent=4)
        logger.info("%s: %d depth maps, %d warnings, QA -> %s", room, len(maps),
                    len(warned), qa_path)

        if failed:
            raise RuntimeError(f"{room}: {len(failed)} depth maps failed QA: {failed}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
