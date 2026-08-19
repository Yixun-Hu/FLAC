"""RAF dataset preparation for FLAC (exp_19, contract section B).

Mirrors ``data/HAA/prepare_data.py`` in role: it turns the raw RAF release into
the runtime layout the FLAC dataloader expects, plus the canonical split JSONs.

    <raf-root>/archived/<Room>/data/<6-digit id>/{rir.wav, tx_pos.txt, rx_pos.txt}
    <raf-root>/archived/<Room>/metadata/all_{tx,rx}_pos.txt
        ->
    <output>/<Room>/mono_rirs_22050Hz/<id>.wav        (22.05 kHz float32)
    <output>/<Room>/metadata/{poses,groups}_metadata.json
    data/RAF/{train,val,test}_base.json + raf_splits_record.json + raf_amplitude_audit.json

Fail-closed throughout: RAF's pose text files are the only description of the
capture geometry, and every silent repair here would surface as a mislabelled
split rather than as an error.

Usage:
    python data/RAF/prepare_data.py --raf-root /path/to/raf_dataset \\
        --output-dir /path/to/runtime/RAF --rooms EmptyRoom FurnishedRoom
"""
import argparse
import hashlib
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import (  # noqa: E402
    RAF_TO_PIPELINE,
    canonicalize_quat,
    parse_rx_line,
    parse_tx_line,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CAPTURE_ID_WIDTH = 6
DEFAULT_CROSSCHECK_SAMPLE = 200


def _read_pose_lines(path):
    """Read a pose file, dropping only TRAILING blank lines.

    A blank line in the middle of the file is left in place so the parser rejects
    it: it would shift every subsequent capture's pose by one.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pose file not found: {path}")
    with open(path) as f:
        lines = f.read().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _capture_dirs(room_dir):
    """Sorted capture directory names under ``<room_dir>/data``, validated.

    Requires exactly the contiguous ids ``000000 .. N-1``: the ``all_*_pos.txt``
    files are positional (line i describes capture i), so a gap or a stray name
    would silently re-key every capture after it.
    """
    data_dir = os.path.join(room_dir, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"capture directory not found: {data_dir}")
    names = sorted(e.name for e in os.scandir(data_dir) if e.is_dir())
    expected = [f"{i:0{CAPTURE_ID_WIDTH}d}" for i in range(len(names))]
    if names != expected:
        mismatch = next((n for n, e in zip(names, expected) if n != e), None)
        raise ValueError(
            f"capture directories in {data_dir} are not the contiguous ids "
            f"000000..{len(names) - 1:0{CAPTURE_ID_WIDTH}d} (first mismatch: {mismatch!r}); "
            "the all_*_pos.txt files are positional, so this cannot be repaired here")
    return names


def load_room_index(room_dir):
    """Read one room's positional pose index.

    Returns:
        list of dicts ``{index, capture_id, quat [4], tx_xyz [3], rx_xyz [3]}``,
        ordered by capture id (== line order in the ``all_*`` files).

    Aborts unless ``len(all_tx) == len(all_rx) == #capture dirs``. The released
    corpus violates this (``all_rx_pos.txt`` carries one extra ``nan,nan,nan``
    line); resolving it is a readback-rung decision, not something this loader may
    absorb.
    """
    meta_dir = os.path.join(room_dir, "metadata")
    tx_lines = _read_pose_lines(os.path.join(meta_dir, "all_tx_pos.txt"))
    rx_lines = _read_pose_lines(os.path.join(meta_dir, "all_rx_pos.txt"))
    capture_ids = _capture_dirs(room_dir)

    if not (len(tx_lines) == len(rx_lines) == len(capture_ids)):
        raise ValueError(
            f"{room_dir}: pose/capture count mismatch — all_tx_pos.txt has "
            f"{len(tx_lines)} lines, all_rx_pos.txt has {len(rx_lines)} lines, "
            f"and there are {len(capture_ids)} capture directories. Refusing to "
            "guess which lines are data (the released corpus ships a trailing "
            "'nan,nan,nan' line in all_rx_pos.txt — that must be resolved "
            "explicitly at the readback rung).")

    index = []
    for i, capture_id in enumerate(capture_ids):
        try:
            quat, tx_xyz = parse_tx_line(tx_lines[i])
            rx_xyz = parse_rx_line(rx_lines[i])
        except ValueError as e:
            raise ValueError(f"{room_dir}: capture {capture_id} (line {i}): {e}")
        index.append({
            "index": i,
            "capture_id": capture_id,
            "quat": quat,
            "tx_xyz": tx_xyz,
            "rx_xyz": rx_xyz,
        })
    return index


def crosscheck_captures(room_dir, index, n_sample=DEFAULT_CROSSCHECK_SAMPLE, seed=0,
                        full=False):
    """Verify per-capture ``tx_pos.txt``/``rx_pos.txt`` against the ``all_*`` lines.

    ``full=True`` checks every capture; otherwise a seeded sample of ``n_sample``
    (capped at the corpus size) is checked. Any mismatch aborts — the two sources
    disagreeing means the positional assumption behind the whole index is wrong.
    """
    if full:
        chosen = list(range(len(index)))
        mode = "full"
    else:
        n = min(int(n_sample), len(index))
        rng = np.random.default_rng(seed)
        chosen = sorted(int(i) for i in rng.choice(len(index), size=n, replace=False))
        mode = "sample"

    for i in chosen:
        record = index[i]
        cdir = os.path.join(room_dir, "data", record["capture_id"])
        tx_lines = _read_pose_lines(os.path.join(cdir, "tx_pos.txt"))
        rx_lines = _read_pose_lines(os.path.join(cdir, "rx_pos.txt"))
        if len(tx_lines) != 1 or len(rx_lines) != 1:
            raise ValueError(
                f"{cdir}: expected exactly one tx and one rx line, got "
                f"{len(tx_lines)}/{len(rx_lines)}")
        quat, tx_xyz = parse_tx_line(tx_lines[0])
        rx_xyz = parse_rx_line(rx_lines[0])
        if not (np.array_equal(quat, record["quat"])
                and np.array_equal(tx_xyz, record["tx_xyz"])):
            raise ValueError(
                f"capture {record['capture_id']}: tx_pos.txt disagrees with line {i} of "
                f"all_tx_pos.txt ({quat.tolist()}+{tx_xyz.tolist()} vs "
                f"{record['quat'].tolist()}+{record['tx_xyz'].tolist()})")
        if not np.array_equal(rx_xyz, record["rx_xyz"]):
            raise ValueError(
                f"capture {record['capture_id']}: rx_pos.txt disagrees with line {i} of "
                f"all_rx_pos.txt ({rx_xyz.tolist()} vs {record['rx_xyz'].tolist()})")

    return {
        "mode": mode,
        "checked": len(chosen),
        "seed": None if full else int(seed),
        "capture_ids": [index[i]["capture_id"] for i in chosen],
        "mismatches": 0,
    }


def _canonical_group_repr(quat_canon, tx_xyz):
    """Text form of the canonical 7-tuple used to derive the group key.

    RAF's pose files carry exactly 6 decimals, so ``%.6f`` is lossless on parsed
    values; it also makes the key robust to a re-emission that differs by <5e-7.
    ``-0.0`` is normalised to ``0.0`` so a canonicalisation sign flip cannot
    produce two spellings of one pose.
    """
    values = np.concatenate([np.asarray(quat_canon, dtype=np.float64),
                             np.asarray(tx_xyz, dtype=np.float64)])
    values = np.where(values == 0.0, 0.0, values)
    return ",".join(f"{v:.6f}" for v in values)


def _group_key(canonical_repr):
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:16]


def _placement_key(centroid_p, ndigits=2):
    """Array-placement bucket: the group's rx centroid rounded to 1 cm.

    Informational in v1 (plan Rev 2 section 5) — it reports how often a physical
    placement was re-occupied; it does not drive the split.
    """
    rounded = np.round(np.asarray(centroid_p, dtype=np.float64), ndigits)
    rounded = np.where(rounded == 0.0, 0.0, rounded)
    return "_".join(f"{v:.{ndigits}f}" for v in rounded)


def group_captures(index, allow_nonuniform=False, expected_size=36):
    """Group captures by the canonicalised full tx pose (quaternion + xyz).

    One group == one (source pose, array placement) session, which is the atomic
    unit of the split. Grouping is by the FULL pose line, never by position alone:
    two orientations at one xyz are different sources.

    Returns:
        (groups, report). ``groups`` is ordered by first capture id; each entry
        carries both the RAF-frame identity (``group_tuple``) and the pipeline-frame
        positions (``tx_xyz_p``, ``rx_xyz_p``, ``rx_centroid_p``) that the runtime
        metadata and the renderer consume.
    """
    if not index:
        raise ValueError("empty capture index")

    by_key = {}
    order = []
    for record in index:
        quat_canon = canonicalize_quat(record["quat"])
        repr_ = _canonical_group_repr(quat_canon, record["tx_xyz"])
        key = _group_key(repr_)
        if key not in by_key:
            by_key[key] = {
                "group_key": key,
                "group_repr": repr_,
                "quat_canon": quat_canon,
                "tx_xyz": np.asarray(record["tx_xyz"], dtype=np.float64),
                "capture_ids": [],
                "rx_xyz": [],
            }
            order.append(key)
        elif by_key[key]["group_repr"] != repr_:
            raise ValueError(
                f"group key collision: {by_key[key]['group_repr']} and {repr_} "
                f"both map to {key}")
        by_key[key]["capture_ids"].append(record["capture_id"])
        by_key[key]["rx_xyz"].append(np.asarray(record["rx_xyz"], dtype=np.float64))

    groups = []
    nonuniform = []
    for key in order:
        g = by_key[key]
        rx = np.vstack(g["rx_xyz"])
        rx_p = rx @ RAF_TO_PIPELINE.T
        centroid_p = rx_p.mean(axis=0)
        g.update({
            "size": len(g["capture_ids"]),
            "group_tuple": np.concatenate([g["quat_canon"], g["tx_xyz"]]).tolist(),
            "tx_xyz_p": RAF_TO_PIPELINE @ g["tx_xyz"],
            "rx_xyz": rx,
            "rx_xyz_p": rx_p,
            "rx_centroid_p": centroid_p,
            "placement_key": _placement_key(centroid_p),
        })
        if g["size"] != expected_size:
            nonuniform.append({"group_key": key, "size": g["size"]})
        groups.append(g)

    if nonuniform:
        detail = ", ".join(f"{d['group_key']}:{d['size']}" for d in nonuniform)
        if not allow_nonuniform:
            raise ValueError(
                f"{len(nonuniform)} of {len(groups)} groups do not hold exactly "
                f"{expected_size} captures ({detail}). Pass --allow-nonuniform to "
                "record the deviation and continue.")
        logger.warning("non-uniform groups recorded: %s", detail)

    placements = {}
    for g in groups:
        placements.setdefault(g["placement_key"], []).append(g["group_key"])

    sizes = {}
    for g in groups:
        sizes[g["size"]] = sizes.get(g["size"], 0) + 1

    report = {
        "n_groups": len(groups),
        "n_captures": len(index),
        "expected_size": expected_size,
        "size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "nonuniform": nonuniform,
        "n_placements": len(placements),
        "placements": placements,
    }
    return groups, report
