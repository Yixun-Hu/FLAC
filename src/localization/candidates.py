"""Candidate-set construction for exp_18 (analysis-by-synthesis source localization).

The AcousticRooms wav namespace (``S008_R089_hybrid_IR.wav``) and the metadata
namespace (``S008_R0089.json``) use different zero-padding conventions, so every
identity here is the *parsed integer* node id; file lookups scan the directory
listing and match numerically instead of reconstructing one fixed name format.
"""
import json
import os
import re

import numpy as np

_IR_NAME_RE = re.compile(r"^S(\d+)_R(\d+)_hybrid_IR\.wav$")
_PAIR_NAME_RE = re.compile(r"^S(\d+)_R(\d+)\.json$")

#: cross-receiver ``src_loc`` agreement tolerance (metres).
SRC_LOC_TOL = 1e-6


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
    return xyz


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
    return arr


def project_to_camera(rec_loc, xyz):
    """World -> receiver-frame coordinates: the translation the loader applies.

    Identical arithmetic (one subtraction per axis) to
    ``AR_md.get_3d_point_camera_coord(source_pose=rec_loc, point_3d=xyz)``, so
    the candidate conditioning reproduces ``md['source']`` bit-exactly.
    """
    return _xyz(xyz, "xyz") - _xyz(rec_loc, "rec_loc")
