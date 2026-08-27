"""Deterministic fTetWild repair and Gmsh-to-FEM conversion primitives."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .fem_solver import TetrahedralMesh


FTETWILD_COMMIT = "d7d99bb4387a07895b9adce058dc7305f6b6e5ab"
FTETWILD_DETERMINISM_PATCH = "seed42-v1"
FTETWILD_IDEAL_EDGE_M = 0.10
FTETWILD_MAXIMUM_THREADS = 1


class MeshResolutionError(ValueError):
    def __init__(self, observed_edge_m: float, allowed_edge_m: float):
        self.observed_edge_m = float(observed_edge_m)
        self.allowed_edge_m = float(allowed_edge_m)
        super().__init__(
            f"tetrahedral mesh maximum edge {self.observed_edge_m:.9g} m exceeds "
            f"the {self.allowed_edge_m:.9g} m gate"
        )


def refined_ideal_edge_m(
    *,
    current_ideal_edge_m: float,
    observed_edge_m: float,
    allowed_edge_m: float,
    safety_factor: float = 0.95,
) -> float:
    """Deterministically tighten fTetWild's target after a true-hmax failure."""

    values = tuple(
        map(float, (current_ideal_edge_m, observed_edge_m, allowed_edge_m, safety_factor))
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("adaptive mesh-refinement inputs must be finite and positive")
    current, observed, allowed, safety = values
    if observed <= allowed:
        raise ValueError("adaptive refinement requires an observed hmax failure")
    if safety >= 1.0:
        raise ValueError("adaptive mesh-refinement safety factor must be below one")
    return current * allowed / observed * safety


def coarsened_ideal_edge_m(
    *,
    current_ideal_edge_m: float,
    observed_edge_m: float,
    allowed_edge_m: float,
    target_utilization: float = 0.90,
) -> float:
    """Deterministically coarsen an unnecessarily fine mesh toward the active gate."""

    values = tuple(
        map(
            float,
            (
                current_ideal_edge_m,
                observed_edge_m,
                allowed_edge_m,
                target_utilization,
            ),
        )
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("adaptive mesh-coarsening inputs must be finite and positive")
    current, observed, allowed, target = values
    if target >= 1.0:
        raise ValueError("mesh-coarsening target utilization must be below one")
    if observed >= allowed * target:
        raise ValueError("mesh is not under-utilized relative to the active hmax gate")
    return current * allowed * target / observed


def _required_line(stream, context: str) -> str:
    line = stream.readline()
    if not line:
        raise ValueError(f"truncated Gmsh file while reading {context}")
    return line.strip()


def _parse_gmsh22_arrays(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read first-order tetrahedra from an ASCII Gmsh 2.2 file."""

    source = Path(path)
    node_tags: np.ndarray | None = None
    nodes: np.ndarray | None = None
    raw_elements: np.ndarray | None = None
    with source.open("r", encoding="utf-8") as stream:
        while True:
            marker = stream.readline()
            if not marker:
                break
            marker = marker.strip()
            if marker == "$MeshFormat":
                fields = _required_line(stream, "mesh format").split()
                if len(fields) != 3 or fields[:2] != ["2.2", "0"]:
                    raise ValueError("only ASCII Gmsh 2.2 tetrahedral meshes are supported")
                if _required_line(stream, "mesh format terminator") != "$EndMeshFormat":
                    raise ValueError("invalid Gmsh mesh-format terminator")
            elif marker == "$Nodes":
                count = int(_required_line(stream, "node count"))
                if count < 4:
                    raise ValueError("Gmsh mesh contains too few nodes")
                nodes = np.empty((count, 3), dtype=np.float64)
                node_tags = np.empty(count, dtype=np.int64)
                for index in range(count):
                    fields = _required_line(stream, "nodes").split()
                    if len(fields) != 4:
                        raise ValueError("invalid Gmsh node record")
                    node_tags[index] = int(fields[0])
                    nodes[index] = [float(value) for value in fields[1:]]
                if _required_line(stream, "node terminator") != "$EndNodes":
                    raise ValueError("invalid Gmsh node terminator")
            elif marker == "$Elements":
                count = int(_required_line(stream, "element count"))
                raw_elements = np.empty((count, 4), dtype=np.int64)
                tetrahedron_count = 0
                for _index in range(count):
                    fields = [int(value) for value in _required_line(stream, "elements").split()]
                    if len(fields) < 3:
                        raise ValueError("invalid Gmsh element record")
                    element_type, tag_count = fields[1], fields[2]
                    node_fields = fields[3 + tag_count :]
                    if element_type == 4:
                        if len(node_fields) != 4:
                            raise ValueError("only first-order Gmsh tetrahedra are supported")
                        raw_elements[tetrahedron_count] = node_fields
                        tetrahedron_count += 1
                raw_elements = raw_elements[:tetrahedron_count]
                if _required_line(stream, "element terminator") != "$EndElements":
                    raise ValueError("invalid Gmsh element terminator")

    if nodes is None or node_tags is None or raw_elements is None or len(raw_elements) == 0:
        raise ValueError("Gmsh file does not contain a tetrahedral volume mesh")
    if len(np.unique(node_tags)) != len(node_tags):
        raise ValueError("Gmsh node tags must be unique")
    if np.array_equal(node_tags, np.arange(1, len(node_tags) + 1)):
        elements = raw_elements - 1
    else:
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
        try:
            elements = np.asarray(
                [[tag_to_index[tag] for tag in element] for element in raw_elements],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("Gmsh element references an unknown node tag") from error
    return nodes, elements


def parse_gmsh22_tetrahedra(path: Path | str) -> TetrahedralMesh:
    """Read an ASCII Gmsh 2.2 file and require one face-connected air domain."""

    nodes, elements = _parse_gmsh22_arrays(path)
    return TetrahedralMesh(nodes=nodes, elements=elements)


def _tetrahedral_face_components(elements: np.ndarray) -> np.ndarray:
    """Label tetrahedra connected through complete triangular faces."""

    parents = np.arange(len(elements), dtype=np.int64)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    face_owner: dict[tuple[int, int, int], int] = {}
    repeated_faces: set[tuple[int, int, int]] = set()
    for element_index, element in enumerate(elements):
        for omitted in range(4):
            face = tuple(
                sorted(
                    int(value)
                    for local_index, value in enumerate(element)
                    if local_index != omitted
                )
            )
            if face in repeated_faces:
                raise ValueError("tetrahedral air domain contains a non-manifold face")
            owner = face_owner.pop(face, None)
            if owner is None:
                face_owner[face] = element_index
            else:
                union(owner, element_index)
                repeated_faces.add(face)

    roots = np.fromiter(
        (find(index) for index in range(len(elements))),
        dtype=np.int64,
        count=len(elements),
    )
    _roots, labels = np.unique(roots, return_inverse=True)
    return labels


def _component_volumes(
    nodes: np.ndarray,
    elements: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    *,
    chunk_size: int = 250_000,
) -> np.ndarray:
    volumes = np.zeros(component_count, dtype=np.float64)
    for start in range(0, len(elements), chunk_size):
        stop = start + chunk_size
        coordinates = nodes[elements[start:stop]]
        differences = coordinates[:, 1:] - coordinates[:, :1]
        element_volumes = np.abs(np.linalg.det(differences)) / 6.0
        if not np.all(element_volumes > 1e-14):
            raise ValueError("mesh contains a degenerate tetrahedron")
        volumes += np.bincount(
            labels[start:stop],
            weights=element_volumes,
            minlength=component_count,
        )
    return volumes


def parse_gmsh22_tetrahedra_largest_component(
    path: Path | str,
    *,
    maximum_removed_volume_fraction: float = 1e-3,
) -> tuple[TetrahedralMesh, dict]:
    """Keep a dominant air component only when discarded volume is negligible.

    fTetWild can occasionally leave a tiny, sealed tetrahedral pocket outside the
    usable room.  Selection is by physical volume, not element count.  The caller
    must still prove that every frozen source, receiver, and candidate lies in the
    returned component.
    """

    maximum_removed_volume_fraction = float(maximum_removed_volume_fraction)
    if (
        not math.isfinite(maximum_removed_volume_fraction)
        or maximum_removed_volume_fraction < 0.0
        or maximum_removed_volume_fraction >= 1.0
    ):
        raise ValueError(
            "maximum removed volume fraction must be finite and in [0, 1)"
        )
    nodes, elements = _parse_gmsh22_arrays(path)
    try:
        mesh = TetrahedralMesh(nodes=nodes, elements=elements)
    except ValueError as error:
        if str(error) != "tetrahedral air domain must be face-connected":
            raise
    else:
        return mesh, {
            "component_count": 1,
            "selected_component": 0,
            "selected_element_count": int(len(elements)),
            "removed_component_count": 0,
            "removed_element_count": 0,
            "removed_volume_m3": 0.0,
            "removed_volume_fraction": 0.0,
            "maximum_allowed_removed_volume_fraction": maximum_removed_volume_fraction,
            "filter_applied": False,
        }

    labels = _tetrahedral_face_components(elements)
    component_count = int(labels.max()) + 1
    volumes = _component_volumes(nodes, elements, labels, component_count)
    selected_component = int(np.argmax(volumes))
    total_volume = float(volumes.sum())
    selected_volume = float(volumes[selected_component])
    removed_volume = total_volume - selected_volume
    removed_volume_fraction = removed_volume / total_volume
    if removed_volume_fraction > maximum_removed_volume_fraction:
        raise ValueError(
            f"removed volume fraction {removed_volume_fraction:.9g} exceeds "
            f"the {maximum_removed_volume_fraction:.9g} largest-component gate"
        )

    keep = labels == selected_component
    kept_elements = elements[keep]
    used_nodes = np.unique(kept_elements)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
    mesh = TetrahedralMesh(nodes=nodes[used_nodes], elements=remap[kept_elements])
    return mesh, {
        "component_count": component_count,
        "selected_component": selected_component,
        "selected_element_count": int(keep.sum()),
        "removed_component_count": component_count - 1,
        "removed_element_count": int((~keep).sum()),
        "original_volume_m3": total_volume,
        "selected_volume_m3": selected_volume,
        "removed_volume_m3": removed_volume,
        "removed_volume_fraction": removed_volume_fraction,
        "maximum_allowed_removed_volume_fraction": maximum_removed_volume_fraction,
        "filter_applied": True,
    }


def build_ftetwild_command(
    binary_path: Path | str,
    source_mesh_path: Path | str,
    output_msh_path: Path | str,
    *,
    ideal_edge_m: float = FTETWILD_IDEAL_EDGE_M,
    maximum_threads: int = FTETWILD_MAXIMUM_THREADS,
    log_path: Path | str,
) -> list[str]:
    """Build the pinned triangle-soup repair and tetrahedralization command."""

    ideal_edge_m = float(ideal_edge_m)
    if not math.isfinite(ideal_edge_m) or ideal_edge_m <= 0:
        raise ValueError("ideal_edge_m must be finite and positive")
    if not isinstance(maximum_threads, int) or isinstance(maximum_threads, bool):
        raise ValueError("maximum_threads must be an integer")
    if maximum_threads <= 0:
        raise ValueError("maximum_threads must be positive")
    return [
        str(Path(binary_path)),
        "-i",
        str(Path(source_mesh_path)),
        "-o",
        str(Path(output_msh_path)),
        "--la",
        format(ideal_edge_m, ".12g"),
        "--correct-surface-orientation",
        "--use-floodfill",
        "--manifold-surface",
        "--smooth-open-boundary",
        "--no-binary",
        "--no-color",
        "--max-threads",
        str(maximum_threads),
        "--log",
        str(Path(log_path)),
        "--level",
        "6",
        "--is-quiet",
    ]


def production_mesh_audit(
    nodes: np.ndarray,
    elements: np.ndarray,
    *,
    maximum_edge_m: float,
    chunk_size: int = 250_000,
) -> dict:
    """Compute scalable geometric gates without assembling Helmholtz matrices."""

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    maximum_edge_m = float(maximum_edge_m)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes must be finite with shape [N, 3]")
    if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
        raise ValueError("elements must have shape [E, 4]")
    if not math.isfinite(maximum_edge_m) or maximum_edge_m <= 0:
        raise ValueError("maximum_edge_m must be finite and positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    total_volume = 0.0
    minimum_volume = math.inf
    minimum_mean_ratio = math.inf
    observed_maximum_edge = 0.0
    for start in range(0, len(elements), chunk_size):
        coordinates = nodes[elements[start : start + chunk_size]]
        differences = coordinates[:, 1:] - coordinates[:, :1]
        volumes = np.abs(np.linalg.det(differences)) / 6.0
        if not np.all(volumes > 1e-14):
            raise ValueError("tetrahedral mesh contains a degenerate element")
        squared_edges = np.stack(
            [
                np.sum((coordinates[:, first] - coordinates[:, second]) ** 2, axis=1)
                for first in range(4)
                for second in range(first)
            ],
            axis=1,
        )
        total_volume += float(volumes.sum())
        minimum_volume = min(minimum_volume, float(volumes.min()))
        observed_maximum_edge = max(
            observed_maximum_edge, float(np.sqrt(squared_edges.max()))
        )
        mean_ratios = 12.0 * (3.0 * volumes) ** (2.0 / 3.0) / squared_edges.sum(axis=1)
        minimum_mean_ratio = min(minimum_mean_ratio, float(mean_ratios.min()))
    if observed_maximum_edge > maximum_edge_m + 1e-12:
        raise MeshResolutionError(observed_maximum_edge, maximum_edge_m)
    return {
        "passed": True,
        "node_count": int(len(nodes)),
        "element_count": int(len(elements)),
        "volume_m3": total_volume,
        "minimum_element_volume_m3": minimum_volume,
        "minimum_element_mean_ratio": minimum_mean_ratio,
        "maximum_element_edge_m": observed_maximum_edge,
        "maximum_allowed_edge_m": maximum_edge_m,
    }


def extract_boundary_triangles(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Extract the true exterior of a tetrahedral domain with outward winding."""

    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes must be finite with shape [N, 3]")
    if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
        raise ValueError("elements must have shape [E, 4]")
    if elements.min() < 0 or elements.max() >= len(nodes):
        raise ValueError("elements reference invalid node indices")

    local_faces = np.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64
    )
    oriented_faces = elements[:, local_faces].reshape(-1, 3).copy()
    omitted_nodes = elements.reshape(-1)
    coordinates = nodes[oriented_faces]
    normals = np.cross(
        coordinates[:, 1] - coordinates[:, 0],
        coordinates[:, 2] - coordinates[:, 0],
    )
    points_to_omitted = nodes[omitted_nodes] - coordinates[:, 0]
    points_inward = np.einsum("ij,ij->i", normals, points_to_omitted) > 0.0
    oriented_faces[points_inward, 1:] = oriented_faces[points_inward, 2:0:-1]

    canonical_faces = np.sort(oriented_faces, axis=1)
    _unique, first_indices, counts = np.unique(
        canonical_faces, axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("tetrahedral air domain contains a non-manifold face")
    boundary_faces = oriented_faces[first_indices[counts == 1]]
    if len(boundary_faces) == 0:
        raise ValueError("tetrahedral air domain has no exterior boundary")
    return np.ascontiguousarray(boundary_faces)


def write_boundary_obj(
    nodes: np.ndarray, faces: np.ndarray, path: Path | str
) -> dict[str, int]:
    """Write a compact, deterministic OBJ for an extracted tetrahedral boundary."""

    nodes = np.asarray(nodes, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
        raise ValueError("nodes must be finite with shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must have shape [F, 3]")
    if faces.min() < 0 or faces.max() >= len(nodes):
        raise ValueError("faces reference invalid node indices")
    used_nodes = np.unique(faces)
    compact_indices = np.full(len(nodes), -1, dtype=np.int64)
    compact_indices[used_nodes] = np.arange(1, len(used_nodes) + 1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Extracted from the final fTetWild tetrahedral air domain.\n")
        for coordinate in nodes[used_nodes]:
            stream.write("v " + " ".join(format(value, ".17g") for value in coordinate) + "\n")
        for face in compact_indices[faces]:
            stream.write("f " + " ".join(str(int(value)) for value in face) + "\n")
    return {"vertex_count": int(len(used_nodes)), "triangle_count": int(len(faces))}


def _normalized_triangle_normal(triangle: np.ndarray, tolerance_m: float) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = float(np.linalg.norm(normal))
    if length <= tolerance_m * tolerance_m:
        raise ValueError("surface contains a degenerate triangle")
    return normal / length


def _coplanar_triangle_relation(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    tolerance_m: float,
) -> str:
    """Classify two coplanar convex triangles using the separating-axis theorem."""

    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    minimum_overlap = math.inf
    for triangle in (first_2d, second_2d):
        for index in range(3):
            edge = triangle[(index + 1) % 3] - triangle[index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axis_length = float(np.linalg.norm(axis))
            if axis_length <= tolerance_m:
                raise ValueError("surface contains a degenerate triangle")
            axis /= axis_length
            first_projection = first_2d @ axis
            second_projection = second_2d @ axis
            overlap = min(first_projection.max(), second_projection.max()) - max(
                first_projection.min(), second_projection.min()
            )
            if overlap < -tolerance_m:
                return "disjoint"
            minimum_overlap = min(minimum_overlap, float(overlap))
    return "penetrating" if minimum_overlap > tolerance_m else "contact"


def _triangle_plane_section(
    triangle: np.ndarray,
    signed_distances: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for index, distance in enumerate(signed_distances):
        if abs(float(distance)) <= tolerance_m:
            points.append(triangle[index])
    for first_index in range(3):
        second_index = (first_index + 1) % 3
        first_distance = float(signed_distances[first_index])
        second_distance = float(signed_distances[second_index])
        if (
            first_distance < -tolerance_m and second_distance > tolerance_m
        ) or (
            first_distance > tolerance_m and second_distance < -tolerance_m
        ):
            fraction = first_distance / (first_distance - second_distance)
            points.append(
                triangle[first_index]
                + fraction * (triangle[second_index] - triangle[first_index])
            )
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance_m for existing in unique):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64).reshape(-1, 3)


def classify_triangle_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    tolerance_m: float = 1e-6,
) -> str:
    """Return ``disjoint``, zero-measure ``contact``, or ``penetrating``.

    Open3D deliberately returns broad self-intersection candidates.  This exact
    second-stage classifier prevents coplanar, overlapping-AABB false positives
    from being mistaken for crossings while retaining a strict penetration gate.
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    tolerance_m = float(tolerance_m)
    if (
        first.shape != (3, 3)
        or second.shape != (3, 3)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("triangles must be finite with shape [3, 3]")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("triangle contact tolerance must be finite and positive")

    first_normal = _normalized_triangle_normal(first, tolerance_m)
    second_normal = _normalized_triangle_normal(second, tolerance_m)
    second_to_first_plane = (second - first[0]) @ first_normal
    first_to_second_plane = (first - second[0]) @ second_normal
    normals_parallel = abs(float(np.dot(first_normal, second_normal))) >= 1.0 - 1e-10
    coplanar = (
        normals_parallel
        and np.max(np.abs(second_to_first_plane)) <= tolerance_m
        and np.max(np.abs(first_to_second_plane)) <= tolerance_m
    )
    if coplanar:
        return _coplanar_triangle_relation(
            first, second, first_normal, tolerance_m
        )
    if normals_parallel:
        return "disjoint"
    shared_vertex_count = int(
        np.any(
            np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
            <= tolerance_m,
            axis=1,
        ).sum()
    )
    if shared_vertex_count >= 2:
        # Two non-coplanar triangles sharing two vertices meet only on that
        # complete edge: their planes have no second intersection locus.
        return "contact"
    if (
        np.all(second_to_first_plane > tolerance_m)
        or np.all(second_to_first_plane < -tolerance_m)
        or np.all(first_to_second_plane > tolerance_m)
        or np.all(first_to_second_plane < -tolerance_m)
    ):
        return "disjoint"

    first_section = _triangle_plane_section(
        first, first_to_second_plane, tolerance_m
    )
    second_section = _triangle_plane_section(
        second, second_to_first_plane, tolerance_m
    )
    if len(first_section) == 0 or len(second_section) == 0:
        return "disjoint"
    line_direction = np.cross(first_normal, second_normal)
    line_direction /= np.linalg.norm(line_direction)
    first_projection = first_section @ line_direction
    second_projection = second_section @ line_direction
    overlap = min(first_projection.max(), second_projection.max()) - max(
        first_projection.min(), second_projection.min()
    )
    if overlap < -tolerance_m:
        return "disjoint"
    return "penetrating" if overlap > tolerance_m else "contact"


def audit_indexed_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    contact_tolerance_m: float = 1e-6,
) -> dict:
    """Audit raw indexed topology without merging intentional vertex splits."""

    import open3d as o3d

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("surface vertices must be finite with shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("surface faces must have shape [F, 3]")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("surface faces reference invalid vertex indices")
    surface = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
    )
    intersection_pairs = np.asarray(
        surface.get_self_intersecting_triangles(), dtype=np.int64
    ).reshape(-1, 2)
    zero_measure_contacts = 0
    disjoint_candidates = 0
    penetrating = 0
    for first_index, second_index in intersection_pairs:
        relation = classify_triangle_pair(
            vertices[faces[first_index]],
            vertices[faces[second_index]],
            tolerance_m=contact_tolerance_m,
        )
        if relation == "disjoint":
            disjoint_candidates += 1
        elif relation == "contact":
            zero_measure_contacts += 1
        else:
            penetrating += 1
    closed_edge_manifold = bool(surface.is_edge_manifold(False))
    vertex_manifold = bool(surface.is_vertex_manifold())
    orientable = bool(surface.is_orientable())
    passed = closed_edge_manifold and vertex_manifold and orientable and penetrating == 0
    return {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "closed_edge_manifold": closed_edge_manifold,
        "vertex_manifold": vertex_manifold,
        "orientable": orientable,
        "intersection_pair_count": int(len(intersection_pairs)),
        "disjoint_candidate_pair_count": disjoint_candidates,
        "zero_measure_contact_pair_count": zero_measure_contacts,
        "penetrating_intersection_pair_count": penetrating,
        "contact_tolerance_m": float(contact_tolerance_m),
        "passed": bool(passed),
    }


def surfaces_have_same_triangle_geometry(
    first_vertices: np.ndarray,
    first_faces: np.ndarray,
    second_vertices: np.ndarray,
    second_faces: np.ndarray,
    *,
    tolerance_m: float = 0.0,
) -> bool:
    """Compare triangle geometry exactly while ignoring topological vertex splits."""

    first_vertices = np.asarray(first_vertices, dtype=np.float64)
    second_vertices = np.asarray(second_vertices, dtype=np.float64)
    first_faces = np.asarray(first_faces, dtype=np.int64)
    second_faces = np.asarray(second_faces, dtype=np.int64)
    for vertices, faces in (
        (first_vertices, first_faces),
        (second_vertices, second_faces),
    ):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("surface vertices must be finite with shape [N, 3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("surface faces must have shape [F, 3]")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("surface faces reference invalid vertex indices")
    if len(first_faces) != len(second_faces):
        return False
    tolerance_m = float(tolerance_m)
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("surface geometry tolerance must be finite and nonnegative")
    first_coordinates, first_ids = np.unique(
        first_vertices, axis=0, return_inverse=True
    )
    if tolerance_m == 0.0:
        combined = np.concatenate((first_coordinates, second_vertices), axis=0)
        _coordinates, inverse = np.unique(combined, axis=0, return_inverse=True)
        remapped_first_ids = inverse[: len(first_coordinates)]
        second_ids = inverse[len(first_coordinates) :]
        first_ids = remapped_first_ids[first_ids]
    else:
        from scipy.spatial import cKDTree

        distances, second_ids = cKDTree(first_coordinates).query(second_vertices, k=1)
        if np.any(distances > tolerance_m):
            return False
    first_triangles = np.sort(first_ids[first_faces], axis=1)
    second_triangles = np.sort(second_ids[second_faces], axis=1)
    first_order = np.lexsort(
        (first_triangles[:, 2], first_triangles[:, 1], first_triangles[:, 0])
    )
    second_order = np.lexsort(
        (second_triangles[:, 2], second_triangles[:, 1], second_triangles[:, 0])
    )
    return bool(
        np.array_equal(first_triangles[first_order], second_triangles[second_order])
    )


def read_triangle_obj_preserve_indices(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangle OBJ without merging equal-coordinate manifold splits."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError("OBJ surface vertex must have three coordinates")
                vertices.append([float(value) for value in fields[1:]])
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError("OBJ surface must contain triangles only")
                face = []
                for value in fields[1:]:
                    vertex_token = value.split("/", 1)[0]
                    index = int(vertex_token)
                    if index <= 0:
                        raise ValueError("OBJ surface requires positive absolute indices")
                    face.append(index - 1)
                faces.append(face)
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise ValueError("OBJ surface contains no valid vertices")
    if face_array.ndim != 2 or face_array.shape[1:] != (3,) or len(face_array) == 0:
        raise ValueError("OBJ surface contains no valid triangles")
    if not np.isfinite(vertex_array).all():
        raise ValueError("OBJ surface contains non-finite vertices")
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError("OBJ surface face references an invalid vertex")
    return vertex_array, face_array


def snap_surface_vertices_to_reference(
    reference_vertices: np.ndarray,
    surface_vertices: np.ndarray,
    *,
    tolerance_m: float,
) -> tuple[np.ndarray, float]:
    """Snap a split surface onto the quantized FEM boundary without merging indices."""

    from scipy.spatial import cKDTree

    reference = np.asarray(reference_vertices, dtype=np.float64)
    surface = np.asarray(surface_vertices, dtype=np.float64)
    tolerance_m = float(tolerance_m)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) == 0
        or not np.isfinite(reference).all()
    ):
        raise ValueError("reference vertices must be finite with shape [N, 3]")
    if (
        surface.ndim != 2
        or surface.shape[1] != 3
        or len(surface) == 0
        or not np.isfinite(surface).all()
    ):
        raise ValueError("surface vertices must be finite with shape [N, 3]")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("surface snap tolerance must be finite and positive")
    unique_reference = np.unique(reference, axis=0)
    distances, indices = cKDTree(unique_reference).query(surface, k=1)
    maximum_distance = float(distances.max())
    if maximum_distance > tolerance_m:
        raise ValueError(
            f"surface-to-FEM snap distance {maximum_distance:.9g} m exceeds "
            f"the {tolerance_m:.9g} m tolerance"
        )
    return unique_reference[indices], maximum_distance


def write_indexed_triangle_obj(
    vertices: np.ndarray, faces: np.ndarray, path: Path | str
) -> None:
    """Write all indexed vertices, including intentional equal-coordinate splits."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("surface vertices must be finite with shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("surface faces must have shape [F, 3]")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("surface faces reference invalid vertex indices")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Manifold-split fTetWild topology snapped to the final FEM boundary.\n")
        for coordinate in vertices:
            stream.write("v " + " ".join(format(value, ".17g") for value in coordinate) + "\n")
        for face in faces + 1:
            stream.write("f " + " ".join(str(int(value)) for value in face) + "\n")
