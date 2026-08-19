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
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import (  # noqa: E402
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
