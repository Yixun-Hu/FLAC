from pathlib import Path
import json

import numpy as np
import pytest

from generate_fem_meshes import atomic_json, load_hashed_json, room_artifacts_are_valid

from src.baselines.fem_meshing import (
    FTETWILD_COMMIT,
    MeshResolutionError,
    audit_indexed_surface,
    build_ftetwild_command,
    extract_boundary_triangles,
    classify_triangle_pair,
    coarsened_ideal_edge_m,
    parse_gmsh22_tetrahedra,
    parse_gmsh22_tetrahedra_largest_component,
    production_mesh_audit,
    refined_ideal_edge_m,
    snap_surface_vertices_to_reference,
    surfaces_have_same_triangle_geometry,
    write_boundary_obj,
)


GMSH22_ONE_TET = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
11 0 0 0
25 1 0 0
40 0 1 0
99 0 0 1
$EndNodes
$Elements
2
1 2 0 11 25 40
2 4 0 11 25 40 99
$EndElements
"""


def test_parse_gmsh22_tetrahedra_maps_sparse_node_tags(tmp_path):
    path = tmp_path / "one.msh"
    path.write_text(GMSH22_ONE_TET)

    mesh = parse_gmsh22_tetrahedra(path)

    assert mesh.nodes.shape == (4, 3)
    assert mesh.elements.tolist() == [[0, 1, 2, 3]]
    assert np.array_equal(mesh.nodes[mesh.elements[0]], np.eye(4, 3, k=-1))


def test_parse_gmsh22_tetrahedra_rejects_binary_or_missing_volume(tmp_path):
    binary = tmp_path / "binary.msh"
    binary.write_text(GMSH22_ONE_TET.replace("2.2 0 8", "2.2 1 8"))
    with pytest.raises(ValueError, match="ASCII Gmsh 2.2"):
        parse_gmsh22_tetrahedra(binary)

    surface = tmp_path / "surface.msh"
    surface.write_text(GMSH22_ONE_TET.replace("2\n1 2 0 11 25 40\n2 4 0 11 25 40 99", "1\n1 2 0 11 25 40"))
    with pytest.raises(ValueError, match="tetrahedral"):
        parse_gmsh22_tetrahedra(surface)


def test_disconnected_gmsh_keeps_only_a_negligible_largest_air_component(tmp_path):
    path = tmp_path / "disconnected.msh"
    path.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
8
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 3 0 0
6 3.05 0 0
7 3 0.05 0
8 3 0 0.05
$EndNodes
$Elements
2
1 4 0 1 2 3 4
2 4 0 5 6 7 8
$EndElements
"""
    )
    with pytest.raises(ValueError, match="face-connected"):
        parse_gmsh22_tetrahedra(path)

    mesh, audit = parse_gmsh22_tetrahedra_largest_component(
        path, maximum_removed_volume_fraction=1e-3
    )

    assert mesh.nodes.shape == (4, 3)
    assert mesh.elements.tolist() == [[0, 1, 2, 3]]
    assert audit["component_count"] == 2
    assert audit["removed_component_count"] == 1
    assert audit["removed_volume_fraction"] == pytest.approx(0.05**3 / (1 + 0.05**3))
    with pytest.raises(ValueError, match="removed volume fraction"):
        parse_gmsh22_tetrahedra_largest_component(
            path, maximum_removed_volume_fraction=1e-5
        )


def test_triangle_pair_classifier_rejects_open3d_candidates_that_are_disjoint():
    first = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    disjoint = np.array([[0.8, 0.8, 0], [1.8, 0.8, 0], [0.8, 1.8, 0]], dtype=float)
    touching = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
    overlapping = np.array(
        [[0.25, 0.25, 0], [1.25, 0.25, 0], [0.25, 1.25, 0]], dtype=float
    )
    crossing = np.array(
        [[0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], dtype=float
    )
    folded_shared_edge = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float
    )

    assert classify_triangle_pair(first, disjoint, tolerance_m=1e-6) == "disjoint"
    assert classify_triangle_pair(first, touching, tolerance_m=1e-6) == "contact"
    assert classify_triangle_pair(first, overlapping, tolerance_m=1e-6) == "penetrating"
    assert classify_triangle_pair(first, crossing, tolerance_m=1e-6) == "penetrating"
    assert (
        classify_triangle_pair(first, folded_shared_edge, tolerance_m=1e-6)
        == "contact"
    )


def test_ftetwild_command_pins_repair_and_resolution_contract(tmp_path):
    binary = tmp_path / "FloatTetwild_bin"
    source = tmp_path / "room.obj"
    output = tmp_path / "room.msh"

    command = build_ftetwild_command(
        binary,
        source,
        output,
        ideal_edge_m=0.10,
        maximum_threads=16,
        log_path=tmp_path / "room.log",
    )

    assert FTETWILD_COMMIT == "d7d99bb4387a07895b9adce058dc7305f6b6e5ab"
    assert command[:6] == [str(binary), "-i", str(source), "-o", str(output), "--la"]
    assert command[6] == "0.1"
    for flag in (
        "--correct-surface-orientation",
        "--use-floodfill",
        "--manifold-surface",
        "--smooth-open-boundary",
        "--no-binary",
        "--no-color",
    ):
        assert flag in command
    assert command[command.index("--max-threads") + 1] == "16"


def test_production_mesh_audit_enforces_true_maximum_edge():
    nodes = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=float
    )
    elements = np.array([[0, 1, 2, 3]])

    audit = production_mesh_audit(nodes, elements, maximum_edge_m=0.18)

    assert audit["passed"] is True
    assert audit["maximum_element_edge_m"] == pytest.approx(np.sqrt(0.02))
    with pytest.raises(MeshResolutionError, match="maximum edge") as caught:
        production_mesh_audit(nodes, elements, maximum_edge_m=0.13)
    assert caught.value.observed_edge_m == pytest.approx(np.sqrt(0.02))
    assert caught.value.allowed_edge_m == pytest.approx(0.13)


def test_adaptive_target_tightens_from_observed_true_maximum_edge():
    refined = refined_ideal_edge_m(
        current_ideal_edge_m=0.10,
        observed_edge_m=0.242992627,
        allowed_edge_m=0.18,
    )

    assert refined == pytest.approx(0.070372505, rel=1e-7)
    assert refined < 0.10


def test_adaptive_target_coarsens_toward_but_below_active_hmax_gate():
    coarsened = coarsened_ideal_edge_m(
        current_ideal_edge_m=0.07,
        observed_edge_m=0.155,
        allowed_edge_m=0.22,
        target_utilization=0.90,
    )

    assert coarsened == pytest.approx(0.07 * 0.22 * 0.90 / 0.155)
    assert coarsened > 0.07

    with pytest.raises(ValueError, match="under-utilized"):
        coarsened_ideal_edge_m(
            current_ideal_edge_m=0.1,
            observed_edge_m=0.2,
            allowed_edge_m=0.22,
            target_utilization=0.90,
        )


def test_surface_audit_allows_only_exact_coordinate_zero_measure_contacts():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
            [5, 7, 6],
            [4, 6, 7],
            [4, 7, 5],
            [4, 5, 6],
        ]
    )

    audit = audit_indexed_surface(vertices, faces)

    assert audit["closed_edge_manifold"] is True
    assert audit["vertex_manifold"] is True
    assert audit["orientable"] is True
    assert audit["penetrating_intersection_pair_count"] == 0
    assert audit["zero_measure_contact_pair_count"] > 0
    assert audit["passed"] is True


def test_surface_geometry_comparison_ignores_manifold_vertex_splits():
    first_vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    first_faces = np.array([[0, 1, 2], [0, 3, 1]])
    split_vertices = np.vstack((first_vertices, first_vertices[0]))
    split_faces = np.array([[0, 1, 2], [4, 3, 1]])

    assert surfaces_have_same_triangle_geometry(
        first_vertices, first_faces, split_vertices, split_faces
    )
    split_vertices[4, 0] = 1e-6
    assert not surfaces_have_same_triangle_geometry(
        first_vertices, first_faces, split_vertices, split_faces
    )
    assert surfaces_have_same_triangle_geometry(
        first_vertices,
        first_faces,
        split_vertices,
        split_faces,
        tolerance_m=1e-5,
    )


def test_surface_vertex_snap_preserves_splits_and_enforces_output_tolerance():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    surface = np.array(
        [[1e-6, 0, 0], [1 + 1e-6, 0, 0], [0, 1 - 1e-6, 0], [1e-6, 0, 0]]
    )

    snapped, maximum = snap_surface_vertices_to_reference(
        reference, surface, tolerance_m=1e-5
    )

    assert maximum == pytest.approx(1e-6)
    assert np.array_equal(snapped, reference[[0, 1, 2, 0]])
    with pytest.raises(ValueError, match="tolerance"):
        snap_surface_vertices_to_reference(reference, surface, tolerance_m=1e-7)


def test_extract_boundary_triangles_removes_shared_face_and_orients_outward():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    elements = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])

    faces = extract_boundary_triangles(nodes, elements)

    assert faces.shape == (6, 3)
    assert (0, 1, 2) not in {tuple(sorted(face)) for face in faces}
    for face in faces:
        owner = next(element for element in elements if set(face).issubset(element))
        omitted = next(index for index in owner if index not in face)
        coordinates = nodes[face]
        normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        assert np.dot(normal, nodes[omitted] - coordinates[0]) < 0.0


def test_write_boundary_obj_compacts_unused_volume_nodes(tmp_path):
    nodes = np.array(
        [[9, 9, 9], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    faces = np.array([[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]])
    path = tmp_path / "boundary.obj"

    audit = write_boundary_obj(nodes, faces, path)

    lines = path.read_text().splitlines()
    assert audit == {"vertex_count": 4, "triangle_count": 4}
    assert len([line for line in lines if line.startswith("v ")]) == 4
    assert len([line for line in lines if line.startswith("f ")]) == 4
    assert not any("9 9 9" in line for line in lines)


def test_hashed_json_rewrite_does_not_hash_the_previous_hash(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"schema_version": 1, "rooms": {}})
    first = load_hashed_json(path)
    first["rooms"]["room"] = {"path": "rooms/room.npz"}

    atomic_json(path, first)

    second = load_hashed_json(path)
    assert second["rooms"] == {"room": {"path": "rooms/room.npz"}}
    assert json.loads(path.read_text())["sha256"] == second["sha256"]


def test_resume_requires_both_artifacts_with_matching_hashes(tmp_path):
    room_path = tmp_path / "rooms" / "room.npz"
    surface_path = tmp_path / "repaired_surfaces" / "room.obj"
    room_path.parent.mkdir()
    surface_path.parent.mkdir()
    room_path.write_bytes(b"mesh")
    surface_path.write_bytes(b"surface")
    manifest_entry = {
        "path": "rooms/room.npz",
        "npz_sha256": "d30ca7a7a32bf5772dc5eb2a2e7bd35737eff795ad74f2479b359716b59abdfa",
    }
    audit_entry = {
        "repaired_surface_path": "repaired_surfaces/room.obj",
        "repaired_surface_sha256": "763cdc62a869262b6ff432a40eae29a00bb96f96f7a3320845abc8cd144d12e2",
        "tetra_npz_path": "rooms/room.npz",
        "tetra_npz_sha256": manifest_entry["npz_sha256"],
    }

    assert room_artifacts_are_valid(tmp_path, manifest_entry, audit_entry)
    surface_path.write_bytes(b"changed")
    assert not room_artifacts_are_valid(tmp_path, manifest_entry, audit_entry)
