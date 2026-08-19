"""Candidate-set construction for exp_18 (analysis-by-synthesis source localization).

The AcousticRooms wav namespace (``S008_R089_hybrid_IR.wav``) and the metadata
namespace (``S008_R0089.json``) use different zero-padding conventions, so every
identity here is the *parsed integer* node id; file lookups scan the directory
listing and match numerically instead of reconstructing one fixed name format.
"""
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch

_IR_NAME_RE = re.compile(r"^S(\d+)_R(\d+)_hybrid_IR\.wav$")
_PAIR_NAME_RE = re.compile(r"^S(\d+)_R(\d+)\.json$")

#: cross-receiver ``src_loc`` agreement tolerance (metres).
SRC_LOC_TOL = 1e-6


def _require_finite(array, what):
    """Fail closed on NaN / +-Inf coordinates.

    JSON admits ``NaN``/``Infinity`` literals, so metadata is a real entry point
    for non-finite values; letting one through would silently poison every
    projection and distance downstream.
    """
    if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
        raise ValueError(f"{what} must be finite (no NaN or Inf)")
    return array


def parse_ir_filename(name):
    """Return ``(src_node, rec_node)`` parsed from an AR IR file name.

    ``name`` may be a bare file name or a full path. Raises ``ValueError`` for
    anything that is not an ``S<digits>_R<digits>_hybrid_IR.wav`` name.
    """
    base = os.path.basename(str(name))
    match = _IR_NAME_RE.match(base)
    if match is None:
        raise ValueError(f"not an AR IR file name: {name!r}")
    return int(match.group(1)), int(match.group(2))


def _pair_files(meta_room_dir):
    """``{(src, rec): path}`` for every pair JSON in ``meta_room_dir``.

    Raises ``ValueError`` if the directory is missing or if two file names parse
    to the same numeric identity (unresolvable ambiguity).
    """
    room = str(meta_room_dir)
    if not os.path.isdir(room):
        raise ValueError(f"metadata room directory not found: {room}")
    pairs = {}
    for fname in sorted(os.listdir(room)):
        match = _PAIR_NAME_RE.match(fname)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in pairs:
            raise ValueError(
                f"ambiguous pair metadata for S{key[0]}_R{key[1]} in {room}: "
                f"{os.path.basename(pairs[key])} and {fname}")
        pairs[key] = os.path.join(room, fname)
    return pairs


def find_pair_metadata(meta_room_dir, src, rec):
    """Path of the ``(src, rec)`` pair JSON, or ``None`` if absent.

    Matching is on parsed numeric identity over the directory listing, so both
    ``S008_R0089.json`` and ``S008_R089.json`` resolve for ``(8, 89)``.
    """
    return _pair_files(meta_room_dir).get((int(src), int(rec)))


def _load_pair(path):
    with open(path, "r") as fin:
        return json.load(fin)


def _loc_array(meta, key, path):
    if key not in meta:
        raise ValueError(f"pair metadata {path} has no {key!r} key")
    xyz = np.asarray(meta[key], dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"pair metadata {path}: {key!r} must be 3 floats, got shape {xyz.shape}")
    return _require_finite(xyz, f"pair metadata {path}: {key!r}")


def enumerate_metadata_sources(meta_room_dir):
    """``{src_node: xyz_world}`` for every source in the room's pair JSONs.

    This is the candidate authority (C7): the candidate set is what ``metadata/``
    declares, not what the wav directory happens to contain. ``src_loc`` must be
    consistent across the receivers that observe the same source (``SRC_LOC_TOL``).
    """
    pairs = _pair_files(meta_room_dir)
    if not pairs:
        raise ValueError(f"no pair metadata (S*_R*.json) in {meta_room_dir}")
    sources, seen_in = {}, {}
    for (src, _rec), path in sorted(pairs.items()):
        xyz = _loc_array(_load_pair(path), "src_loc", path)
        if src in sources:
            if not np.allclose(sources[src], xyz, rtol=0.0, atol=SRC_LOC_TOL):
                raise ValueError(
                    f"inconsistent src_loc for source {src}: {sources[src].tolist()} in "
                    f"{os.path.basename(seen_in[src])} vs {xyz.tolist()} in {os.path.basename(path)}")
        else:
            sources[src], seen_in[src] = xyz, path
    return {node: sources[node] for node in sorted(sources)}


def _xyz(value, what):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{what} must be 3 floats, got shape {arr.shape}")
    return _require_finite(arr, what)


def project_to_camera(rec_loc, xyz):
    """World -> receiver-frame coordinates: the translation the loader applies.

    Identical arithmetic (one subtraction per axis) to
    ``AR_md.get_3d_point_camera_coord(source_pose=rec_loc, point_3d=xyz)``, so
    the candidate conditioning reproduces ``md['source']`` bit-exactly.
    """
    return _xyz(xyz, "xyz") - _xyz(rec_loc, "rec_loc")


@dataclass
class CandidateSet:
    """The metadata-defined candidate set C of one query (GT included).

    ``nodes`` is sorted ascending and ``xyz_world[i]`` is the world position of
    ``nodes[i]``; the order is therefore deterministic across runs and machines.
    """
    nodes: list
    xyz_world: np.ndarray
    rec_loc: np.ndarray
    gt_node: int
    gt_xyz: np.ndarray

    def __post_init__(self):
        self.nodes = [int(n) for n in self.nodes]
        self.xyz_world = np.asarray(self.xyz_world, dtype=np.float64)
        self.rec_loc = _xyz(self.rec_loc, "rec_loc")
        self.gt_node = int(self.gt_node)
        self.gt_xyz = _xyz(self.gt_xyz, "gt_xyz")
        if self.xyz_world.ndim != 2 or self.xyz_world.shape[1] != 3:
            raise ValueError(f"xyz_world must be [M, 3], got shape {self.xyz_world.shape}")
        _require_finite(self.xyz_world, "xyz_world")
        if len(self.nodes) != self.xyz_world.shape[0]:
            raise ValueError(
                f"nodes ({len(self.nodes)}) and xyz_world ({self.xyz_world.shape[0]}) disagree")
        if self.nodes != sorted(set(self.nodes)):
            raise ValueError(f"nodes must be sorted and unique, got {self.nodes}")
        if self.gt_node not in self.nodes:
            raise ValueError(f"GT node {self.gt_node} is not in the candidate set {self.nodes}")
        row = self.xyz_world[self.nodes.index(self.gt_node)]
        if not np.array_equal(row, self.gt_xyz):
            raise ValueError(
                f"gt_xyz {self.gt_xyz.tolist()} != candidate row {row.tolist()} for node {self.gt_node}")

    @property
    def gt_index(self):
        return self.nodes.index(self.gt_node)

    def __len__(self):
        return len(self.nodes)


def build_candidate_set(ir_path, metadata_path):
    """Candidate set for the query IR at ``ir_path``.

    ``metadata_path`` is the dataset's ``metadata`` root; the room directory is
    ``<metadata_path>/<scene>/<scene_id>/``, derived from ``ir_path`` exactly as
    the release loader derives it.
    """
    ir_path = str(ir_path)
    gt_node, rec_node = parse_ir_filename(ir_path)
    parts = ir_path.replace(os.sep, "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"ir_path must contain <scene>/<scene_id>/<file>: {ir_path}")
    scene, scene_id = parts[-3], parts[-2]
    room_dir = os.path.join(str(metadata_path), scene, scene_id)

    sources = enumerate_metadata_sources(room_dir)
    if gt_node not in sources:
        raise ValueError(
            f"GT source {gt_node} of {os.path.basename(ir_path)} is absent from {room_dir}")
    pair_path = find_pair_metadata(room_dir, gt_node, rec_node)
    if pair_path is None:
        raise ValueError(f"no pair metadata for S{gt_node}_R{rec_node} in {room_dir}")
    rec_loc = _loc_array(_load_pair(pair_path), "rec_loc", pair_path)

    nodes = sorted(sources)
    return CandidateSet(
        nodes=nodes,
        xyz_world=np.stack([sources[n] for n in nodes], axis=0),
        rec_loc=rec_loc,
        gt_node=gt_node,
        gt_xyz=sources[gt_node],
    )


def candidate_metadata(base_md, cand_cam_xyz):
    """``base_md`` with only ``source``/``source_vit`` swapped for a candidate.

    Shallow copy (O19): every untouched key -- ``depth``, ``context_*``, ... --
    is the *same object* in the returned dict, so the M candidate variants of a
    query share those tensors and ``base_md`` is never mutated.
    ``cand_cam_xyz`` is the candidate position in the receiver camera frame.
    """
    if isinstance(cand_cam_xyz, torch.Tensor):
        source = cand_cam_xyz.detach().to(torch.float32).reshape(-1).clone()
        if source.numel() != 3:
            raise ValueError(f"cand_cam_xyz must be 3 floats, got {tuple(cand_cam_xyz.shape)}")
        _require_finite(source.numpy(), "cand_cam_xyz")
    else:
        source = torch.as_tensor(_xyz(cand_cam_xyz, "cand_cam_xyz"), dtype=torch.float32)
    md = dict(base_md)
    md["source"] = source
    md["source_vit"] = source.unsqueeze(0)
    return md


def assert_gt_matches_loader(cand_set, md, atol=0.0):
    """Fail closed unless the GT candidate's projection *is* the loader's ``source``.

    The per-query geometry invariant (O6): the candidate pipeline and the release
    loader must agree bit-exactly on where the ground-truth source sits in the
    receiver frame (``atol=0``), otherwise the whole candidate sweep is offset.
    """
    if "source" not in md:
        raise AssertionError("metadata has no 'source' key; cannot verify the geometry invariant")
    got = md["source"]
    if not isinstance(got, torch.Tensor):
        got = torch.as_tensor(got)
    if tuple(got.shape) != (3,):
        raise AssertionError(f"md['source'] must be shape [3], got {tuple(got.shape)}")
    expected = torch.as_tensor(
        project_to_camera(cand_set.rec_loc, cand_set.gt_xyz), dtype=torch.float32)
    diff = (got.detach().to(torch.float32).double() - expected.double()).abs().max().item()
    if not diff <= atol:
        raise AssertionError(
            f"GT candidate projection {expected.tolist()} != loader md['source'] "
            f"{got.tolist()} (max |diff| = {diff:g} > atol {atol:g})")
    return None


def crosscheck_sources_vs_files(meta_nodes, room_dir):
    """Readback rung: metadata-declared sources vs the room's IR file names.

    Reports both directions without changing the candidate set -- ``metadata/``
    stays the authority; missing files only shrink what a measured-RIR oracle can
    report.
    """
    room = str(room_dir)
    if not os.path.isdir(room):
        raise ValueError(f"IR room directory not found: {room}")
    file_nodes = set()
    for fname in sorted(os.listdir(room)):
        match = _IR_NAME_RE.match(fname)
        if match is not None:
            file_nodes.add(int(match.group(1)))
    meta = {int(n) for n in meta_nodes}
    missing = sorted(meta - file_nodes)
    extra = sorted(file_nodes - meta)
    return {
        "meta_nodes": sorted(meta),
        "file_nodes": sorted(file_nodes),
        "missing_files": missing,
        "extra_files": extra,
        "ok": not missing and not extra,
    }
