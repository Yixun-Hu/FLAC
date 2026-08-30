#!/usr/bin/env python3
"""Repair official AcousticRooms triangle soups and build audited FEM air meshes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.fem_meshing import (
    FTETWILD_COMMIT,
    FTETWILD_DETERMINISM_PATCH,
    FTETWILD_IDEAL_EDGE_M,
    FTETWILD_MAXIMUM_THREADS,
    MeshResolutionError,
    audit_indexed_surface,
    build_ftetwild_command,
    coarsened_ideal_edge_m,
    extract_boundary_triangles,
    parse_gmsh22_tetrahedra_largest_component,
    production_mesh_audit,
    read_triangle_obj_preserve_indices,
    refined_ideal_edge_m,
    snap_surface_vertices_to_reference,
    surfaces_have_same_triangle_geometry,
    write_indexed_triangle_obj,
)
from src.baselines.fem_pipeline import FEM_MAXIMUM_EDGE_M, save_tetrahedral_mesh_npz
from src.baselines.fem_solver import barycentric_interpolation_matrix
from src.localization.engine import reconstruct_room_base_candidates
from src.localization.pilot import canonical_sha256


EXP09_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "worklog/worklog_yixun/exp_10_room_helps_baselines_claude/fem_meshes"
)
# fTetWild writes the volume MSH and manifold surface OBJ independently in
# decimal text. Allow sub-0.1 mm serialization drift before snapping the
# template onto the exact tetrahedral boundary; centimeter-scale repair motion
# remains far outside this gate.
SURFACE_GEOMETRY_TOLERANCE_M = 1e-4
MAXIMUM_REMOVED_AIR_VOLUME_FRACTION = 1e-3


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    content["sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def exclusive_file_lock(path: Path):
    """Serialize manifest read-modify-write cycles across room workers."""
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_hashed_json(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text())
    expected = payload.pop("sha256", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise RuntimeError(f"hashed JSON identity mismatch: {path}")
    payload["sha256"] = expected
    return payload


def room_artifacts_are_valid(
    output_dir: Path,
    manifest_entry: dict,
    audit_entry: dict,
) -> bool:
    try:
        mesh_relative = manifest_entry["path"]
        mesh_hash = manifest_entry["npz_sha256"]
        surface_relative = audit_entry["repaired_surface_path"]
        surface_hash = audit_entry["repaired_surface_sha256"]
        if audit_entry["tetra_npz_path"] != mesh_relative:
            return False
        if audit_entry["tetra_npz_sha256"] != mesh_hash:
            return False
        mesh_path = output_dir / mesh_relative
        surface_path = output_dir / surface_relative
        return (
            mesh_path.is_file()
            and surface_path.is_file()
            and file_sha256(mesh_path) == mesh_hash
            and file_sha256(surface_path) == surface_hash
        )
    except (KeyError, TypeError, OSError):
        return False


def _resume_payload(path: Path, expected: dict, *, mutable_keys: set[str]) -> dict:
    if not path.exists():
        return expected
    existing = load_hashed_json(path)
    identity_keys = set(expected) - mutable_keys - {"sha256"}
    if (set(existing) - mutable_keys - {"sha256"}) != identity_keys:
        raise RuntimeError(f"resume identity keys differ: {path}")
    for key in identity_keys:
        if existing[key] != expected[key]:
            raise RuntimeError(f"resume identity differs at {key!r}: {path}")
    return existing


def _validate_repaired_surface(
    path: Path, volume_nodes: np.ndarray, boundary_faces: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    template_vertices, repaired_faces = read_triangle_obj_preserve_indices(path)
    template_audit = audit_indexed_surface(template_vertices, repaired_faces)
    used_nodes = np.unique(boundary_faces)
    remap = np.full(len(volume_nodes), -1, dtype=np.int64)
    remap[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
    volume_boundary_vertices = np.asarray(volume_nodes, dtype=np.float64)[used_nodes]
    volume_boundary_faces = remap[boundary_faces]
    volume_boundary_audit = audit_indexed_surface(
        volume_boundary_vertices, volume_boundary_faces
    )
    repaired_vertices, maximum_snap_distance = snap_surface_vertices_to_reference(
        volume_boundary_vertices,
        template_vertices,
        tolerance_m=SURFACE_GEOMETRY_TOLERANCE_M,
    )
    repaired_audit = audit_indexed_surface(repaired_vertices, repaired_faces)
    geometry_matches = surfaces_have_same_triangle_geometry(
        volume_boundary_vertices,
        volume_boundary_faces,
        repaired_vertices,
        repaired_faces,
        tolerance_m=0.0,
    )
    volume_boundary_passed = (
        volume_boundary_audit["closed_edge_manifold"]
        and volume_boundary_audit["orientable"]
        and volume_boundary_audit["penetrating_intersection_pair_count"] == 0
    )
    template_topology_passed = (
        template_audit["closed_edge_manifold"]
        and template_audit["vertex_manifold"]
        and template_audit["orientable"]
    )
    if (
        not template_topology_passed
        or not repaired_audit["passed"]
        or not volume_boundary_passed
        or not geometry_matches
    ):
        raise RuntimeError(
            "repaired surface failed indexed topology/geometry gate: "
            f"template={template_audit}, snapped={repaired_audit}, "
            f"volume_boundary={volume_boundary_audit}, "
            f"geometry_matches={geometry_matches}"
        )
    return (
        {
            "ftetwild_manifold_template": template_audit,
            "repaired_indexed_surface": repaired_audit,
            "tetrahedral_boundary": volume_boundary_audit,
            "maximum_template_to_fem_snap_distance_m": maximum_snap_distance,
            "maximum_allowed_snap_distance_m": SURFACE_GEOMETRY_TOLERANCE_M,
            "triangle_geometry_matches_tetrahedral_boundary": True,
            "triangle_geometry_match_tolerance_m": 0.0,
            "zero_measure_contacts_allowed": True,
            "penetrating_intersections_allowed": False,
        },
        repaired_vertices,
        repaired_faces,
    )


def _room_anchor_points(room: str, context_manifest: dict, geometry_audit: dict):
    records = [record for record in context_manifest["records"] if record["room"] == room]
    if not records:
        raise RuntimeError(f"room {room!r} has no frozen context records")
    sources = np.unique(
        np.asarray([record["source_global"] for record in records], dtype=np.float64), axis=0
    )
    receivers = np.unique(
        np.asarray([record["receiver_global"] for record in records], dtype=np.float64), axis=0
    )
    candidates = reconstruct_room_base_candidates(room, geometry_audit)
    return sources, receivers, candidates


def _validate_room_points(mesh, room: str, context_manifest: dict, geometry_audit: dict):
    sources, receivers, candidates = _room_anchor_points(
        room, context_manifest, geometry_audit
    )
    points = np.concatenate((sources, receivers, candidates), axis=0)
    interpolation = barycentric_interpolation_matrix(mesh, points, tolerance=1e-7)
    row_sums = np.asarray(interpolation.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0):
        raise RuntimeError("tetrahedral interpolation weights do not sum to one")
    return {
        "source_anchor_count": int(len(sources)),
        "receiver_anchor_count": int(len(receivers)),
        "candidate_count": int(len(candidates)),
        "all_points_inside": True,
    }


def _clean_working_outputs(msh_path: Path) -> None:
    for suffix in (
        "",
        "__sf.obj",
        "__cutting.stl",
        "__tracked_surface.stl",
        "__simplify.off",
    ):
        path = Path(str(msh_path) + suffix)
        if path.exists():
            path.unlink()


def generate_room(
    *,
    room: str,
    binary: Path,
    output_dir: Path,
    geometry_audit: dict,
    context_manifest: dict,
    ideal_edge_m: float,
    maximum_edge_m: float,
    maximum_threads: int,
    minimum_edge_utilization: float = 0.0,
    target_edge_utilization: float = 0.90,
    smooth_open_boundary: bool = True,
    reuse_current_working_mesh: bool = False,
    current_working_ideal_edge_m: float | None = None,
) -> tuple[dict, dict]:
    room_audit = geometry_audit["rooms"][room]
    source_path = Path(room_audit["mesh_path"])
    if file_sha256(source_path) != room_audit["mesh_sha256"]:
        raise RuntimeError(f"official OBJ hash mismatch for {room}")
    working_dir = output_dir / "working"
    mesh_dir = output_dir / "rooms"
    surface_dir = output_dir / "repaired_surfaces"
    log_dir = output_dir / "logs"
    for directory in (working_dir, mesh_dir, surface_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    msh_path = working_dir / f"{room}.msh"
    current_ideal_edge_m = float(ideal_edge_m)
    total_meshing_seconds = 0.0
    total_parse_and_validate_seconds = 0.0
    attempts = []
    mesh = None
    component_audit = None
    geometric_audit = None
    command = None
    if reuse_current_working_mesh:
        if current_working_ideal_edge_m is None:
            raise ValueError("working-mesh reuse requires its exact ideal edge length")
        current_ideal_edge_m = float(current_working_ideal_edge_m)
        generated_surface = Path(str(msh_path) + "__sf.obj")
        if not msh_path.is_file() or not generated_surface.is_file():
            raise FileNotFoundError("completed working MSH and repaired surface are required")
        command = build_ftetwild_command(
            binary,
            source_path,
            msh_path,
            ideal_edge_m=current_ideal_edge_m,
            maximum_threads=maximum_threads,
            smooth_open_boundary=smooth_open_boundary,
            log_path=log_dir / f"{room}.attempt_2.ftetwild.log",
        )
        start = time.perf_counter()
        mesh, component_audit = parse_gmsh22_tetrahedra_largest_component(
            msh_path,
            maximum_removed_volume_fraction=MAXIMUM_REMOVED_AIR_VOLUME_FRACTION,
        )
        geometric_audit = production_mesh_audit(
            mesh.nodes, mesh.elements, maximum_edge_m=maximum_edge_m
        )
        parse_seconds = time.perf_counter() - start
        total_parse_and_validate_seconds += parse_seconds
        attempts.append(
            {
                "attempt": "recovered_completed_working_mesh",
                "ideal_edge_m": current_ideal_edge_m,
                "observed_maximum_edge_m": geometric_audit[
                    "maximum_element_edge_m"
                ],
                "passed_hmax": True,
                "meshing_seconds": None,
                "parse_and_audit_seconds": parse_seconds,
                "working_msh_sha256_before_finalization": file_sha256(msh_path),
                "working_surface_sha256_before_finalization": file_sha256(
                    generated_surface
                ),
                "air_domain_components": component_audit,
                "command": command,
            }
        )
    for attempt_index in (() if reuse_current_working_mesh else range(1, 4)):
        _clean_working_outputs(msh_path)
        ftetwild_log = log_dir / f"{room}.attempt_{attempt_index}.ftetwild.log"
        console_log = log_dir / f"{room}.attempt_{attempt_index}.console.log"
        command = build_ftetwild_command(
            binary,
            source_path,
            msh_path,
            ideal_edge_m=current_ideal_edge_m,
            maximum_threads=maximum_threads,
            smooth_open_boundary=smooth_open_boundary,
            log_path=ftetwild_log,
        )
        start = time.perf_counter()
        with console_log.open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        attempt_meshing_seconds = time.perf_counter() - start
        total_meshing_seconds += attempt_meshing_seconds
        if not msh_path.is_file():
            raise RuntimeError(f"fTetWild did not produce a volume mesh for {room}")

        start = time.perf_counter()
        mesh, component_audit = parse_gmsh22_tetrahedra_largest_component(
            msh_path,
            maximum_removed_volume_fraction=MAXIMUM_REMOVED_AIR_VOLUME_FRACTION,
        )
        try:
            geometric_audit = production_mesh_audit(
                mesh.nodes, mesh.elements, maximum_edge_m=maximum_edge_m
            )
        except MeshResolutionError as error:
            attempt_parse_seconds = time.perf_counter() - start
            total_parse_and_validate_seconds += attempt_parse_seconds
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ideal_edge_m": current_ideal_edge_m,
                    "observed_maximum_edge_m": error.observed_edge_m,
                    "passed_hmax": False,
                    "meshing_seconds": attempt_meshing_seconds,
                    "parse_and_audit_seconds": attempt_parse_seconds,
                    "air_domain_components": component_audit,
                    "command": command,
                }
            )
            if attempt_index == 3:
                raise
            next_ideal_edge_m = refined_ideal_edge_m(
                current_ideal_edge_m=current_ideal_edge_m,
                observed_edge_m=error.observed_edge_m,
                allowed_edge_m=maximum_edge_m,
            )
            print(
                f"[{room}] hmax={error.observed_edge_m:.6f} m failed; "
                f"retry la={next_ideal_edge_m:.6f} m",
                flush=True,
            )
            current_ideal_edge_m = next_ideal_edge_m
            continue
        attempt_parse_seconds = time.perf_counter() - start
        total_parse_and_validate_seconds += attempt_parse_seconds
        observed_edge_m = geometric_audit["maximum_element_edge_m"]
        underutilized = (
            minimum_edge_utilization > 0
            and observed_edge_m < maximum_edge_m * minimum_edge_utilization
        )
        attempts.append(
            {
                "attempt": attempt_index,
                "ideal_edge_m": current_ideal_edge_m,
                "observed_maximum_edge_m": geometric_audit[
                    "maximum_element_edge_m"
                ],
                "passed_hmax": True,
                "passed_utilization": not underutilized,
                "meshing_seconds": attempt_meshing_seconds,
                "parse_and_audit_seconds": attempt_parse_seconds,
                "air_domain_components": component_audit,
                "command": command,
            }
        )
        if underutilized:
            if attempt_index == 3:
                raise RuntimeError(
                    f"{room} hmax={observed_edge_m:.9g} remains below the "
                    f"minimum utilization {minimum_edge_utilization:.3g} of the "
                    f"{maximum_edge_m:.9g} m gate"
                )
            next_ideal_edge_m = coarsened_ideal_edge_m(
                current_ideal_edge_m=current_ideal_edge_m,
                observed_edge_m=observed_edge_m,
                allowed_edge_m=maximum_edge_m,
                target_utilization=target_edge_utilization,
            )
            print(
                f"[{room}] hmax={observed_edge_m:.6f} m under-utilized; "
                f"retry la={next_ideal_edge_m:.6f} m",
                flush=True,
            )
            current_ideal_edge_m = next_ideal_edge_m
            continue
        break
    if (
        mesh is None
        or component_audit is None
        or geometric_audit is None
        or command is None
    ):
        raise RuntimeError(f"adaptive tetrahedralization did not complete for {room}")

    start = time.perf_counter()
    boundary_faces = extract_boundary_triangles(mesh.nodes, mesh.elements)
    generated_surface = Path(str(msh_path) + "__sf.obj")
    if not generated_surface.is_file():
        raise RuntimeError(f"fTetWild did not produce its repaired surface for {room}")
    repaired_surface = surface_dir / f"{room}.obj"
    surface_audit, repaired_vertices, repaired_faces = _validate_repaired_surface(
        generated_surface, mesh.nodes, boundary_faces
    )
    write_indexed_triangle_obj(repaired_vertices, repaired_faces, repaired_surface)
    point_audit = _validate_room_points(mesh, room, context_manifest, geometry_audit)
    total_parse_and_validate_seconds += time.perf_counter() - start

    destination = mesh_dir / f"{room}.npz"
    temporary = mesh_dir / f"{room}.partial.npz"
    save_tetrahedral_mesh_npz(
        mesh,
        temporary,
        source_mesh_sha256=room_audit["mesh_sha256"],
    )
    temporary.replace(destination)
    npz_hash = file_sha256(destination)
    surface_hash = file_sha256(repaired_surface)
    _clean_working_outputs(msh_path)

    manifest_entry = {
        "path": str(destination.relative_to(output_dir)),
        "npz_sha256": npz_hash,
    }
    audit_entry = {
        "room": room,
        "source_obj_path": str(source_path),
        "source_obj_sha256": room_audit["mesh_sha256"],
        "repaired_surface_path": str(repaired_surface.relative_to(output_dir)),
        "repaired_surface_sha256": surface_hash,
        "tetra_npz_path": str(destination.relative_to(output_dir)),
        "tetra_npz_sha256": npz_hash,
        "initial_ideal_edge_m": float(ideal_edge_m),
        "final_ideal_edge_m": float(current_ideal_edge_m),
        "meshing_attempts": attempts,
        "meshing_seconds": float(total_meshing_seconds),
        "parse_and_validate_seconds": float(total_parse_and_validate_seconds),
        "surface": surface_audit,
        "air_domain_components": component_audit,
        "volume": geometric_audit,
        "points": point_audit,
        "command": command,
    }
    return manifest_entry, audit_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftetwild-bin", type=Path, required=True)
    parser.add_argument(
        "--geometry-audit", type=Path, default=EXP09_DIR / "geometry_audit.json"
    )
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=EXP09_DIR / "context_manifest_exp01_seed42.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rooms", nargs="*")
    parser.add_argument("--ideal-edge-m", type=float, default=FTETWILD_IDEAL_EDGE_M)
    parser.add_argument("--maximum-edge-m", type=float, default=FEM_MAXIMUM_EDGE_M)
    parser.add_argument("--maximum-threads", type=int, default=FTETWILD_MAXIMUM_THREADS)
    parser.add_argument(
        "--minimum-edge-utilization",
        type=float,
        default=0.0,
        help="optional lower hmax/maximum-edge ratio; zero disables coarsening retries",
    )
    parser.add_argument("--target-edge-utilization", type=float, default=0.90)
    parser.add_argument(
        "--no-smooth-open-boundary",
        action="store_true",
        help=(
            "disable fTetWild open-boundary smoothing when it excludes frozen "
            "source, receiver, or candidate points"
        ),
    )
    parser.add_argument("--reuse-current-working-mesh", action="store_true")
    parser.add_argument("--current-working-ideal-edge-m", type=float)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record a room failure and continue processing the remaining rooms",
    )
    args = parser.parse_args()

    if not 0.0 <= args.minimum_edge_utilization < 1.0:
        raise ValueError("minimum edge utilization must lie in [0, 1)")
    if not 0.0 < args.target_edge_utilization < 1.0:
        raise ValueError("target edge utilization must lie in (0, 1)")
    if (
        args.minimum_edge_utilization > 0
        and args.target_edge_utilization <= args.minimum_edge_utilization
    ):
        raise ValueError("target utilization must exceed minimum utilization")

    binary = args.ftetwild_bin.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("FEM mesh outputs must stay inside the NeuriPs_Workshop worktree") from error
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_audit = json.loads(args.geometry_audit.read_text())
    context_manifest = json.loads(args.context_manifest.read_text())
    available_rooms = tuple(sorted(geometry_audit["rooms"]))
    rooms = tuple(args.rooms) if args.rooms else available_rooms
    if len(set(rooms)) != len(rooms) or any(room not in available_rooms for room in rooms):
        raise ValueError("--rooms must be unique names from the frozen geometry audit")
    if args.reuse_current_working_mesh and len(rooms) != 1:
        raise ValueError("working-mesh recovery is restricted to exactly one named room")
    if args.reuse_current_working_mesh != (
        args.current_working_ideal_edge_m is not None
    ):
        raise ValueError(
            "working-mesh recovery and its exact ideal edge length must be supplied together"
        )

    manifest_path = output_dir / "tetra_mesh_manifest.json"
    audit_path = output_dir / "mesh_generation_audit.json"
    failures_path = output_dir / "mesh_generation_failures.json"
    expected_manifest = {
        "schema_version": 1,
        "mesh_generator": "fTetWild",
        "mesh_generator_commit": FTETWILD_COMMIT,
        "mesh_generator_determinism_patch": FTETWILD_DETERMINISM_PATCH,
        "mesh_generator_binary_sha256": file_sha256(binary),
        "geometry_audit_sha256": geometry_audit["sha256"],
        "ideal_edge_m": float(args.ideal_edge_m),
        "maximum_edge_m": float(args.maximum_edge_m),
        "maximum_threads": int(args.maximum_threads),
        "rooms": {},
    }
    expected_audit = {
        "schema_version": 1,
        "mesh_generator": "fTetWild",
        "mesh_generator_commit": FTETWILD_COMMIT,
        "mesh_generator_determinism_patch": FTETWILD_DETERMINISM_PATCH,
        "mesh_generator_binary_sha256": file_sha256(binary),
        "geometry_audit_sha256": geometry_audit["sha256"],
        "context_manifest_sha256": context_manifest["sha256"],
        "ideal_edge_m": float(args.ideal_edge_m),
        "maximum_edge_m": float(args.maximum_edge_m),
        "maximum_threads": int(args.maximum_threads),
        "requested_rooms": list(rooms),
        "rooms": {},
    }
    expected_failures = {
        "schema_version": 1,
        "mesh_generator_binary_sha256": file_sha256(binary),
        "geometry_audit_sha256": geometry_audit["sha256"],
        "maximum_removed_air_volume_fraction": (
            MAXIMUM_REMOVED_AIR_VOLUME_FRACTION
        ),
        "rooms": {},
    }
    if args.minimum_edge_utilization > 0:
        for payload in (expected_manifest, expected_audit):
            payload["minimum_edge_utilization"] = float(
                args.minimum_edge_utilization
            )
            payload["target_edge_utilization"] = float(
                args.target_edge_utilization
            )

    state_lock_path = output_dir / ".manifest.lock"

    def load_current_state() -> tuple[dict, dict]:
        if manifest_path.exists() != audit_path.exists():
            raise RuntimeError("mesh manifest and generation audit must resume together")
        current_manifest = _resume_payload(
            manifest_path, expected_manifest, mutable_keys={"rooms"}
        )
        current_audit = _resume_payload(
            audit_path,
            expected_audit,
            mutable_keys={"rooms", "requested_rooms"},
        )
        current_audit["requested_rooms"] = sorted(
            set(current_audit.get("requested_rooms", ()))
            | set(rooms)
            | set(current_audit["rooms"])
        )
        return current_manifest, current_audit

    def update_failure(room: str, error: BaseException | None) -> None:
        with exclusive_file_lock(state_lock_path):
            failures = _resume_payload(
                failures_path, expected_failures, mutable_keys={"rooms"}
            )
            if error is None:
                failures["rooms"].pop(room, None)
            else:
                failures["rooms"][room] = {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            atomic_json(failures_path, failures)

    for index, room in enumerate(rooms, start=1):
        with exclusive_file_lock(state_lock_path):
            manifest, audit = load_current_state()
            manifest_entry = manifest["rooms"].get(room)
            audit_entry = audit["rooms"].get(room)
        if (
            manifest_entry is not None
            and audit_entry is not None
            and room_artifacts_are_valid(output_dir, manifest_entry, audit_entry)
        ):
            print(f"[{index}/{len(rooms)}] resume {room}", flush=True)
            continue
        print(f"[{index}/{len(rooms)}] mesh {room}", flush=True)
        try:
            manifest_entry, audit_entry = generate_room(
                room=room,
                binary=binary,
                output_dir=output_dir,
                geometry_audit=geometry_audit,
                context_manifest=context_manifest,
                ideal_edge_m=args.ideal_edge_m,
                maximum_edge_m=args.maximum_edge_m,
                maximum_threads=args.maximum_threads,
                minimum_edge_utilization=args.minimum_edge_utilization,
                target_edge_utilization=args.target_edge_utilization,
                smooth_open_boundary=not args.no_smooth_open_boundary,
                reuse_current_working_mesh=args.reuse_current_working_mesh,
                current_working_ideal_edge_m=args.current_working_ideal_edge_m,
            )
        except Exception as error:
            if not args.continue_on_error:
                raise
            update_failure(room, error)
            print(
                f"[{index}/{len(rooms)}] fail {room}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        with exclusive_file_lock(state_lock_path):
            manifest, audit = load_current_state()
            existing_manifest_entry = manifest["rooms"].get(room)
            existing_audit_entry = audit["rooms"].get(room)
            if existing_manifest_entry is not None or existing_audit_entry is not None:
                if (
                    existing_manifest_entry != manifest_entry
                    or existing_audit_entry != audit_entry
                ):
                    raise RuntimeError(
                        f"concurrent worker committed a different result for {room}"
                    )
            manifest["rooms"][room] = manifest_entry
            audit["rooms"][room] = audit_entry
            atomic_json(manifest_path, manifest)
            atomic_json(audit_path, audit)
        update_failure(room, None)
        print(
            f"[{index}/{len(rooms)}] pass {room}: "
            f"{audit_entry['volume']['node_count']} nodes, "
            f"{audit_entry['volume']['element_count']} tets, "
            f"hmax={audit_entry['volume']['maximum_element_edge_m']:.6f} m",
            flush=True,
        )


if __name__ == "__main__":
    main()
