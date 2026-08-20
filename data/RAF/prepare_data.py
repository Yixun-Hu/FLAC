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
import json
import logging
import math
import os
import sys

import librosa
import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # raf_common.py is a sibling script, not an installed package
    sys.path.insert(0, _HERE)
from publish import StagedPublish  # noqa: E402
from readback_audit import load_passing_record, record_provenance  # noqa: E402
from raf_common import (  # noqa: E402
    DBFS_FLOOR,
    RAF_TO_PIPELINE,
    canonicalize_quat,
    dbfs as _dbfs,
    distance_stats as _distance_stats,
    farthest_point_selection,
    parse_rx_line,
    parse_tx_line,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

CAPTURE_ID_WIDTH = 6
DEFAULT_CROSSCHECK_SAMPLE = 200
SOURCE_SR = 48000
TARGET_SR = 22050
RIR_FOLDER = "mono_rirs_22050Hz"
DEPTH_SUFFIX = "_depth_image.npy"
# The FLAC loader crops to sample_size and drops anything below -60 dBFS
# (src/data/dataset.py::is_silence) — both mirrored here so the audit measures
# exactly what the runtime will see.
LOADER_SAMPLE_SIZE = 10240
SILENCE_THRESHOLD_DB = -60.0


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


class RoomIndex(list):
    """The capture list PLUS how the pose files had to be read to produce it.

    A plain list would carry no record of the dropped ``all_rx`` sentinel, and the
    drop must be visible in ``raf_splits_record.json`` — so the provenance travels
    with the data rather than beside it. Slicing/copying yields a plain list, which
    is deliberate: only the object that was actually read may make the claim (see
    ``sentinel_flag_of``).
    """

    def __init__(self, records, rx_trailing_sentinel_dropped):
        super().__init__(records)
        self.rx_trailing_sentinel_dropped = bool(rx_trailing_sentinel_dropped)


def sentinel_flag_of(index):
    """Read the sentinel provenance off an index, FAIL-CLOSED.

    A direct attribute read with no default: something that never observed the
    pose files (a plain list, a slice) raises instead of quietly asserting the
    comfortable "no line was dropped".
    """
    return index.rx_trailing_sentinel_dropped


def _is_nan_triplet(line):
    """True iff the line is exactly three fields that ALL parse as NaN."""
    fields = line.strip().split(",")
    if len(fields) != 3:
        return False
    for field in fields:
        try:
            value = float(field)
        except (TypeError, ValueError):
            return False
        if not math.isnan(value):   # inf is a corrupt value, not a sentinel
            return False
    return True


def load_room_index(room_dir):
    """Read one room's positional pose index.

    Returns:
        ``RoomIndex`` (a list of ``{index, capture_id, quat [4], tx_xyz [3],
        rx_xyz [3]}`` ordered by capture id == line order in the ``all_*`` files)
        carrying ``rx_trailing_sentinel_dropped``.

    Requires ``len(all_tx) == len(all_rx) == #capture dirs``, with exactly ONE
    registered exception (contracts Amendment 1, D1): a single trailing
    ``nan,nan,nan`` line in ``all_rx_pos.txt`` — which the released corpus ships —
    is dropped iff it is the final line, all three fields are NaN, and the counts
    line up once it is gone. The drop is recorded, never silent. Every other
    mismatch aborts, and NaN anywhere else in either file aborts at the parser.
    """
    meta_dir = os.path.join(room_dir, "metadata")
    tx_path = os.path.join(meta_dir, "all_tx_pos.txt")
    rx_path = os.path.join(meta_dir, "all_rx_pos.txt")
    tx_lines = _read_pose_lines(tx_path)
    rx_lines = _read_pose_lines(rx_path)
    capture_ids = _capture_dirs(room_dir)

    sentinel_dropped = False
    if (len(rx_lines) == len(tx_lines) + 1
            and len(tx_lines) == len(capture_ids)
            and _is_nan_triplet(rx_lines[-1])):
        logger.warning(
            "%s: dropping the trailing all-NaN sentinel line %d of all_rx_pos.txt "
            "(%r) per contracts Amendment 1; %d data lines remain",
            room_dir, len(rx_lines), rx_lines[-1].strip(), len(tx_lines))
        rx_lines = rx_lines[:-1]
        sentinel_dropped = True

    if not (len(tx_lines) == len(rx_lines) == len(capture_ids)):
        raise ValueError(
            f"{room_dir}: pose/capture count mismatch — all_tx_pos.txt has "
            f"{len(tx_lines)} lines, all_rx_pos.txt has {len(rx_lines)} lines, "
            f"and there are {len(capture_ids)} capture directories. Refusing to "
            "guess which lines are data (only a SINGLE trailing all-NaN line in "
            "all_rx_pos.txt is droppable, and only when the counts line up "
            "afterwards).")

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
    return RoomIndex(index, sentinel_dropped)


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


CANONICAL_GROUP_SIZE = 36


def _render_xyz(xyz):
    """6-decimal rendering of a position, matching the group-key convention."""
    values = np.where(np.asarray(xyz, dtype=np.float64) == 0.0, 0.0, xyz)
    return ",".join(f"{v:.6f}" for v in values)


def count_duplicate_atoms(group):
    """How many captures of ``group`` repeat an earlier ``(tx pose, rx)`` atom."""
    seen, duplicates = set(), 0
    for row in group["rx_xyz"]:
        atom = _render_xyz(row)
        if atom in seen:
            duplicates += 1
        seen.add(atom)
    return duplicates


def partition_eligibility(groups, expected_size=CANONICAL_GROUP_SIZE):
    """Split groups into FPS-eligible ones and explicitly excluded ones (R1).

    Eligibility is decided BEFORE any selection: a group that does not hold
    exactly ``expected_size`` captures cannot yield the registered 12/24 manifest,
    so letting it into the farthest-point sequence would either produce a
    differently-sized split or silently displace an eligible group.
    """
    eligible, excluded = [], []
    for g in groups:
        if g["size"] != expected_size:
            excluded.append({
                "group_key": g["group_key"],
                "size": g["size"],
                "exclusion_reason": f"size!={expected_size}",
            })
        else:
            eligible.append(g)
    return eligible, excluded


def assert_unique_atoms(groups):
    """Every eligible capture must be a distinct ``(tx pose, rx)`` measurement.

    A repeated atom is the same physical measurement twice, so it could land in
    both a support pool and the test half of the same group — leakage that no
    downstream check could see. Reserve groups are exempt by construction: they are
    never split, and their anomalies are recorded instead (the real FurnishedRoom
    72-capture group is 36 duplicated atoms).
    """
    seen, duplicates = {}, []
    for g in groups:
        for capture_id, row in zip(g["capture_ids"], g["rx_xyz"]):
            atom = (g["group_key"], _render_xyz(row))
            if atom in seen:
                duplicates.append((atom[0], atom[1], seen[atom], capture_id))
            else:
                seen[atom] = capture_id
    if duplicates:
        detail = "; ".join(f"group {k} rx({rx}) captures {a}+{b}"
                           for k, rx, a, b in duplicates[:5])
        raise ValueError(
            f"{len(duplicates)} duplicate (tx-pose, rx) atoms among eligible groups "
            f"({detail}). The same measurement cannot be both support and test.")


def select_splits(groups, n_groups=16, n_val_groups=4, n_train=12,
                  n_diagnostic_groups=1, expected_size=CANONICAL_GROUP_SIZE):
    """Preregistered, deterministic, group-atomic split (plan Rev 2 section 5).

    Only exactly-``expected_size`` groups are eligible (R1). One farthest-point
    sequence over the eligible groups' tx positions selects ``n_groups`` train/test
    groups, then *continues* into ``n_val_groups`` validation groups and
    ``n_diagnostic_groups`` HAA-parity diagnostic groups; everything else — the
    ineligible groups included, each with an ``exclusion_reason`` — is reserve
    (listed, never evaluated).

    Within every selected group a farthest-point sequence over the receiver
    positions picks ``n_train`` support mics, and the remaining captures are that
    group's targets:

    * ``train_test`` — supports are the TRAINING items, the other 24 the test items;
    * ``val``        — supports are CONTEXT ONLY, the other 24 are the val targets
      (R14: HAA parity, so every val target draws from the same 12 candidates
      rather than 12 of them drawing from 11);
    * ``diagnostic`` — same 12-support/24-target manifest, evaluated as a separate
      HAA-analogue row. Per contracts Amendment 3 its 12 supports JOIN the training
      set (the literal HAA analogue finetunes on the eval room's supports), so the
      canonical training set is 16x12 + 12 = 204 items per room; its 24 targets stay
      eval-only.
    """
    for name, value in (("n_groups", n_groups), ("n_val_groups", n_val_groups),
                        ("n_train", n_train)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if (not isinstance(n_diagnostic_groups, (int, np.integer))
            or isinstance(n_diagnostic_groups, bool) or n_diagnostic_groups < 0):
        raise ValueError(
            f"n_diagnostic_groups must be a non-negative int, got {n_diagnostic_groups!r}")

    eligible, excluded = partition_eligibility(groups, expected_size)
    assert_unique_atoms(eligible)

    n_needed = int(n_groups) + int(n_val_groups) + int(n_diagnostic_groups)
    if n_needed > len(eligible):
        raise ValueError(
            f"need {n_needed} groups ({n_groups} train/test + {n_val_groups} val + "
            f"{n_diagnostic_groups} diagnostic) but only {len(eligible)} of "
            f"{len(groups)} groups are eligible (exactly {expected_size} captures)")

    tx = np.vstack([g["tx_xyz_p"] for g in eligible])
    order = farthest_point_selection(tx, n_needed)
    train_test = [eligible[i]["group_key"] for i in order[:n_groups]]
    val = [eligible[i]["group_key"] for i in order[n_groups:n_groups + n_val_groups]]
    diagnostic = [eligible[i]["group_key"] for i in order[n_groups + n_val_groups:]]
    selected = set(train_test) | set(val) | set(diagnostic)
    reserve = [g["group_key"] for g in groups if g["group_key"] not in selected]

    by_key = {g["group_key"]: g for g in groups}
    support_ids, train_ids, test_ids, val_ids, diagnostic_ids = {}, {}, {}, {}, {}
    target_bucket = dict([(k, test_ids) for k in train_test]
                         + [(k, val_ids) for k in val]
                         + [(k, diagnostic_ids) for k in diagnostic])
    for key in train_test + val + diagnostic:
        g = by_key[key]
        if n_train > g["size"]:
            raise ValueError(
                f"group {key} holds {g['size']} captures, cannot pick {n_train} support mics")
        picks = farthest_point_selection(g["rx_xyz_p"], n_train)
        # kept in FPS order: the record then shows the selection sequence itself.
        # Consumers that need an order-independent draw (RAF_md's deterministic
        # eval context) sort the pool themselves rather than trusting this one.
        support = [g["capture_ids"][i] for i in picks]
        support_ids[key] = support
        targets = [c for c in g["capture_ids"] if c not in set(support)]
        target_bucket[key][key] = targets
        # Amendment 3: train/test AND diagnostic supports are training items; only
        # a val group's supports stay context-only (they support the val loop,
        # which must not see gradient).
        if key in set(train_test) or key in set(diagnostic):
            train_ids[key] = list(support)

    _assert_disjoint(train_ids, test_ids, val_ids, diagnostic_ids, by_key,
                     val, diagnostic, reserve)

    roles = {k: "train_test" for k in train_test}
    roles.update({k: "val" for k in val})
    roles.update({k: "diagnostic" for k in diagnostic})
    roles.update({k: "reserve" for k in reserve})

    return {
        "train_test_groups": train_test,
        "val_groups": val,
        "diagnostic_groups": diagnostic,
        "reserve_groups": reserve,
        "excluded_groups": excluded,
        "roles": roles,
        "support_ids": support_ids,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "val_ids": val_ids,
        "diagnostic_ids": diagnostic_ids,
        "params": {"n_groups": int(n_groups), "n_val_groups": int(n_val_groups),
                   "n_train": int(n_train),
                   "n_diagnostic_groups": int(n_diagnostic_groups),
                   "expected_size": int(expected_size)},
    }


def _assert_disjoint(train_ids, test_ids, val_ids, diagnostic_ids, by_key,
                     val_groups, diagnostic_groups, reserve_groups):
    """Fail-closed leakage check: the whole experiment rests on this."""
    buckets = {"train": train_ids, "test": test_ids, "val": val_ids,
               "diagnostic": diagnostic_ids}
    names = sorted(buckets)
    flat = {name: {i for v in ids.values() for i in v} for name, ids in buckets.items()}
    for a in names:
        for b in names:
            if a < b and flat[a] & flat[b]:
                raise ValueError(f"split leakage: {len(flat[a] & flat[b])} captures in both {a} and {b}")
    # A val or reserve group is protected whole. A diagnostic group is protected
    # only in its TARGETS: Amendment 3 puts its supports in the training set on
    # purpose (that is what makes the row the literal HAA analogue), so protecting
    # the whole group would forbid the registered design rather than a leak.
    protected = set()
    for key in list(val_groups) + list(reserve_groups):
        protected |= set(by_key[key]["capture_ids"])
    protected |= {c for v in diagnostic_ids.values() for c in v}
    leaked = protected & (flat["train"] | flat["test"])
    if leaked:
        raise ValueError(
            f"split leakage: {len(leaked)} protected captures (val/reserve groups, or "
            "diagnostic TARGETS) appear in train/test")


def assemble_split_jsons(per_room):
    """Build the three HAA-shaped split dicts ``{room: ["<id>.wav", ...]}``."""
    jsons = {"train": {}, "val": {}, "test": {}, "diagnostic": {}}
    for room, payload in per_room.items():
        split = payload["split"]
        for name, ids in (("train", split["train_ids"]),
                          ("test", split["test_ids"]),
                          ("val", split["val_ids"]),
                          ("diagnostic", split["diagnostic_ids"])):
            files = sorted(f"{cid}.wav" for group_ids in ids.values() for cid in group_ids)
            jsons[name][room] = files
    return jsons


def write_split_files(split_dir, jsons):
    """Write ``{train,val,test}_base.json`` into ``split_dir``."""
    os.makedirs(split_dir, exist_ok=True)
    paths = {}
    for name, payload in jsons.items():
        path = os.path.join(split_dir, f"{name}_base.json")
        _write_json(path, payload)
        paths[name] = path
    return paths


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
    return path


def _role_distances(groups, split, group_keys, target_ids):
    """Target-to-nearest-support and support-support distances for one role."""
    by_key = {g["group_key"]: g for g in groups}
    nearest, pairwise = [], []
    for key in group_keys:
        g = by_key[key]
        pos = {cid: g["rx_xyz_p"][i] for i, cid in enumerate(g["capture_ids"])}
        support = np.vstack([pos[c] for c in split["support_ids"][key]])
        for cid in target_ids.get(key, []):
            nearest.append(float(np.linalg.norm(support - pos[cid], axis=1).min()))
        d = np.linalg.norm(support[:, None, :] - support[None, :, :], axis=-1)
        pairwise.extend(d[np.triu_indices(len(support), k=1)].tolist())
    return {
        "target_to_nearest_support": _distance_stats(nearest),
        "support_pairwise": _distance_stats(pairwise),
    }


def _room_distances(groups, split):
    """Support/target distance distributions, overall and per role."""
    by_role = {
        "train_test": _role_distances(groups, split, split["train_test_groups"],
                                      split["test_ids"]),
        "val": _role_distances(groups, split, split["val_groups"], split["val_ids"]),
        "diagnostic": _role_distances(groups, split, split["diagnostic_groups"],
                                      split["diagnostic_ids"]),
    }
    return {
        # kept under its r1 name: the headline "how far is a test mic from its
        # support set" distribution, which is the train/test row
        "test_to_nearest_support": by_role["train_test"]["target_to_nearest_support"],
        "support_pairwise": by_role["train_test"]["support_pairwise"],
        "by_role": by_role,
    }


def _git_describe():
    try:
        import subprocess
        out = subprocess.run(["git", "describe", "--always", "--dirty"],
                             cwd=_HERE, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def build_splits_record(per_room, params):
    """The canonical, committed description of how the split was cut."""
    rooms = {}
    for room, payload in per_room.items():
        groups, split = payload["groups"], payload["split"]
        rooms[room] = {
            "counts": {
                "train": sum(len(v) for v in split["train_ids"].values()),
                "test": sum(len(v) for v in split["test_ids"].values()),
                "val": sum(len(v) for v in split["val_ids"].values()),
                "diagnostic": sum(len(v) for v in split["diagnostic_ids"].values()),
                "train_test_groups": len(split["train_test_groups"]),
                "val_groups": len(split["val_groups"]),
                "diagnostic_groups": len(split["diagnostic_groups"]),
                "reserve_groups": len(split["reserve_groups"]),
            },
            "train_test_groups": list(split["train_test_groups"]),
            "val_groups": list(split["val_groups"]),
            "diagnostic_groups": list(split["diagnostic_groups"]),
            "reserve_groups": list(split["reserve_groups"]),
            "excluded_groups": list(split["excluded_groups"]),
            # Recorded, never asserted (R1): an ineligible group is already kept out
            # of every split, so its anomalies are a fact about reserve rather than
            # a reason to stop. The real FurnishedRoom 72-capture group shows up
            # here as 36 duplicate atoms.
            "reserve_anomalies": {
                entry["group_key"]: {
                    "size": entry["size"],
                    "exclusion_reason": entry["exclusion_reason"],
                    "duplicate_atoms": count_duplicate_atoms(
                        next(g for g in groups if g["group_key"] == entry["group_key"])),
                }
                for entry in split["excluded_groups"]
            },
            "distances": _room_distances(groups, split),
            "placements": {
                "n_placements": payload["group_report"]["n_placements"],
                "groups_per_placement": {k: len(v) for k, v
                                         in payload["group_report"]["placements"].items()},
            },
            "group_report": {k: v for k, v in payload["group_report"].items()
                             if k != "placements"},
            "crosscheck": payload["crosscheck"],
            # How the pose files were read (contracts Amendment 1, D1). Indexed,
            # not .get(): a payload that never observed them may not claim False.
            "rx_trailing_sentinel_dropped": bool(payload["rx_trailing_sentinel_dropped"]),
            "group_details": [
                {
                    "group_key": g["group_key"],
                    "group_tuple": [float(v) for v in g["group_tuple"]],
                    "tx_xyz_p": [float(v) for v in g["tx_xyz_p"]],
                    "placement_key": g["placement_key"],
                    "size": g["size"],
                    "role": split["roles"][g["group_key"]],
                    "support_ids": split["support_ids"].get(g["group_key"], []),
                }
                for g in groups
            ],
        }
    return {"params": params, "git_describe": _git_describe(), "rooms": rooms}


def resample_and_write(room_dir, out_room_dir, capture_ids, target_sr=TARGET_SR,
                       orig_sr=SOURCE_SR, sample_size=LOADER_SAMPLE_SIZE,
                       silence_db=SILENCE_THRESHOLD_DB, folder_name=RIR_FOLDER,
                       roles=None, scale=None):
    """Resample the given captures to 22.05 kHz float32 WAVs + amplitude audit.

    ``subtype='FLOAT'`` is a declared divergence from HAA's PCM16 default: RAF RIRs
    peak near 0.01, so 16-bit would leave the decay tail only ~56 dB above the
    quantisation floor.

    Fail-closed on a wrong source rate, a multi-channel file, non-finite samples,
    or |x| > 1 in either the source or the output. The per-file peak and the
    sub-``silence_db`` flag are recorded but NOT fatal: the flag is what tells us
    how many items the dataloader would silently substitute
    (``src/data/dataset.py::is_silence``), which is a fact about the corpus rather
    than a preparation failure.

    R13: every written file is READ BACK and compared against what was meant to be
    written (a publish that silently truncated or re-quantised would otherwise be
    invisible), all dB values are JSON-safe, the distributions are separated by
    split role, and ``scale`` -- if the readback audit ever calls for one -- is a
    single scalar applied identically to every file, with the decision recorded
    from TRAIN SUPPORTS ONLY.
    """
    dest = os.path.join(out_room_dir, folder_name)
    os.makedirs(dest, exist_ok=True)
    if scale is not None:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be a positive finite scalar, got {scale}")

    files, peaks, n_silent = {}, [], 0
    roundtrip_max = 0.0
    for capture_id in capture_ids:
        src = os.path.join(room_dir, "data", capture_id, "rir.wav")
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        if sr != orig_sr:
            raise ValueError(f"{src}: expected {orig_sr} Hz, got {sr} Hz")
        if audio.shape[1] != 1:
            raise ValueError(f"{src}: expected mono, got {audio.shape[1]} channels")
        wave = audio[:, 0]
        if not np.isfinite(wave).all():
            raise ValueError(f"{src}: source holds non-finite samples")
        src_peak = float(np.abs(wave).max())
        # Checked on the SOURCE as well as the output: the anti-alias filter can
        # pull a lone out-of-range spike back under 1.0, and the loader clamps to
        # [-1, 1], so an over-range source would be silently distorted at runtime.
        if src_peak > 1.0:
            raise ValueError(f"{src}: source signal is out of range (peak {src_peak:.6f} > 1.0)")

        out = librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr)
        out = np.asarray(out, dtype=np.float32)
        if scale is not None:
            out = np.asarray(out * scale, dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError(f"{src}: resampling produced non-finite samples")
        peak = float(np.abs(out).max())
        if peak > 1.0:
            raise ValueError(f"{src}: resampled signal clips (peak {peak:.6f} > 1.0)")

        out_path = os.path.join(dest, f"{capture_id}.wav")
        sf.write(out_path, out, target_sr, subtype="FLOAT")

        # R13: read back what was actually published.
        back, back_sr = sf.read(out_path, dtype="float32", always_2d=True)
        if back_sr != target_sr or back.shape[1] != 1:
            raise ValueError(
                f"{out_path}: read back as {back_sr} Hz / {back.shape[1]} ch, "
                f"expected {target_sr} Hz mono")
        if back.shape[0] != out.shape[0]:
            raise ValueError(
                f"{out_path}: read back {back.shape[0]} samples, wrote {out.shape[0]}")
        roundtrip = float(np.abs(back[:, 0] - out).max())
        roundtrip_max = max(roundtrip_max, roundtrip)

        crop_peak = float(np.abs(out[:sample_size]).max())
        dbfs = _dbfs(peak)
        dbfs_crop = _dbfs(crop_peak)
        # The loader crops to sample_size BEFORE its silence test, so the crop is
        # the number that decides substitution.
        silent = bool(dbfs_crop < silence_db)
        n_silent += int(silent)
        peaks.append(peak)
        files[capture_id] = {
            "peak": peak,
            "peak_crop": crop_peak,
            "dbfs": dbfs,
            "dbfs_crop": dbfs_crop,
            "silent_at_threshold": silent,
            "n_samples": int(out.shape[0]),
            "roundtrip_max_abs_error": roundtrip,
            "roundtrip_samples": int(back.shape[0]),
            "role": None if roles is None else roles.get(capture_id),
        }

    by_role = {}
    for capture_id, entry in files.items():
        if entry["role"] is not None:
            by_role.setdefault(entry["role"], []).append(entry["peak"])
    support_peaks = [e["peak"] for e in files.values() if e["role"] in ("train", "support")]

    return {
        "n_files": len(files),
        "orig_sr": int(orig_sr),
        "target_sr": int(target_sr),
        "subtype": "FLOAT",
        "sample_size": int(sample_size),
        "silence_threshold_db": float(silence_db),
        "dbfs_floor": DBFS_FLOOR,
        "n_silent": n_silent,
        "peak_stats": _distance_stats(peaks),
        "roundtrip_max_abs_error": roundtrip_max,
        "by_role": {role: _distance_stats(values) for role, values in sorted(by_role.items())},
        # The approved decision rule (plan Rev 2 section 10.4): no rescaling unless
        # RAF is off-scale against HAA/AR, and any scalar must come from the train
        # supports alone -- never from statistics that saw a test item.
        "scale_decision": {
            "rule": ("none unless RAF is off-scale versus HAA/AR; if applied, ONE "
                     "scalar applied identically to targets and context"),
            "derived_from": "train supports only",
            "n_train_supports": len(support_peaks),
            "train_support_peak_median": (float(np.median(support_peaks))
                                          if support_peaks else None),
            "train_support_peak_stats": _distance_stats(support_peaks),
            "applied_scalar": None if scale is None else float(scale),
        },
        "comparison": {
            "HAA": None,
            "AR": None,
            "note": ("reference peak distributions are filled in from the processed "
                     "HAA/AR corpora at the readback rung"),
        },
        "files": files,
    }


def build_runtime_metadata(index, groups, split):
    """Loader-visible metadata for one room: (poses_metadata, groups_metadata).

    Only captures of SELECTED groups appear: those are the only files the runtime
    ever opens, and every worker loads this JSON. The reserve groups are listed in
    ``raf_splits_record.json`` instead (they carry no depth map and no support pool,
    so an entry here would also make the depth renderer render them).
    """
    role_of = {}
    # Context-only supports first, so a support that is ALSO a training item (the
    # train/test role) is overwritten by its stronger role below.
    for gk, ids in split["support_ids"].items():
        role_of.update({c: "support" for c in ids})
    for gk, ids in split["train_ids"].items():
        role_of.update({c: "train" for c in ids})
    for gk, ids in split["test_ids"].items():
        role_of.update({c: "test" for c in ids})
    for gk, ids in split["val_ids"].items():
        role_of.update({c: "val" for c in ids})
    for gk, ids in split["diagnostic_ids"].items():
        role_of.update({c: "diagnostic" for c in ids})

    by_id = {r["capture_id"]: r for r in index}
    selected = (list(split["train_test_groups"]) + list(split["val_groups"])
                + list(split["diagnostic_groups"]))
    by_key = {g["group_key"]: g for g in groups}

    poses, groups_meta = {}, {}
    for gk in selected:
        g = by_key[gk]
        tx_p = [float(v) for v in g["tx_xyz_p"]]
        for i, capture_id in enumerate(g["capture_ids"]):
            if capture_id not in role_of:
                raise ValueError(
                    f"capture {capture_id} of selected group {gk} has no split role")
            poses[capture_id] = {
                "tx_xyz_p": tx_p,
                "quat_raw": [float(v) for v in by_id[capture_id]["quat"]],
                "rx_p": [float(v) for v in g["rx_xyz_p"][i]],
                "group_key": gk,
                "split_role": role_of[capture_id],
            }
        groups_meta[gk] = {
            "tx_xyz_p": tx_p,
            "depth_file": f"{gk}{DEPTH_SUFFIX}",
            "train_ids": list(split["support_ids"][gk]),
            "role": split["roles"][gk],
        }
    return poses, groups_meta


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare the RAF dataset for FLAC")
    parser.add_argument('--raf-root', required=True,
                        help="RAF release root (holds archived/<Room>/ and 3d_models/)")
    parser.add_argument('--output-dir', required=True,
                        help="runtime dataset root; rooms are written as <output>/<Room>/")
    parser.add_argument('--split-dir', default='data/RAF',
                        help="where the canonical split JSONs + records are written")
    parser.add_argument('--rooms', nargs='+', default=['EmptyRoom', 'FurnishedRoom'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-groups', type=int, default=16)
    parser.add_argument('--n-val-groups', type=int, default=4)
    parser.add_argument('--n-diagnostic-groups', type=int, default=1,
                        help="HAA-parity diagnostic groups, taken next in the same "
                             "FPS sequence (12 context-only supports / 24 targets)")
    parser.add_argument('--n-train', type=int, default=12)
    parser.add_argument('--crosscheck-sample', type=int, default=DEFAULT_CROSSCHECK_SAMPLE)
    parser.add_argument('--full-crosscheck', action='store_true',
                        help="cross-check every capture instead of a seeded sample")
    parser.add_argument('--allow-nonuniform', action='store_true',
                        help="record rather than abort on groups that do not hold 36 captures")
    parser.add_argument('--readback-record', required=True,
                        help="path to a PASSING, adjudicated raf_readback_record.json "
                             "(data/RAF/readback_audit.py); canonical artifacts are "
                             "never published without one")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # R4 publish gate, BEFORE anything is written: the onset/delay fit, the T30
    # truncation decision and the gauge/quaternion pinning must already exist and
    # have been adjudicated, or these artifacts would be canonical under
    # assumptions nobody checked.
    readback = load_passing_record(args.readback_record)
    readback_provenance = record_provenance(args.readback_record, readback)
    logger.info("readback record %s (gauge %s, quat %s, T60 %s)",
                readback_provenance["sha256"][:12],
                readback_provenance["gauge_pinned"],
                readback_provenance["quat_order_pinned"],
                readback_provenance["t60_headline"])

    # R7: the runtime tree and the split directory are each staged whole and
    # swapped in only once every room has been read, resampled and audited. Until
    # both commits happen, the previous publish is the only thing on disk.
    with StagedPublish(args.output_dir) as staged_runtime, \
            StagedPublish(args.split_dir) as staged_splits:
        per_room, audits = {}, {}
        for room in args.rooms:
            room_dir = os.path.join(args.raf_root, "archived", room)
            logger.info("reading %s", room_dir)
            index = load_room_index(room_dir)
            crosscheck = crosscheck_captures(room_dir, index, n_sample=args.crosscheck_sample,
                                             seed=args.seed, full=args.full_crosscheck)
            logger.info("%s: cross-checked %d captures (%s)", room, crosscheck["checked"],
                        crosscheck["mode"])
            groups, group_report = group_captures(index, allow_nonuniform=args.allow_nonuniform)
            logger.info("%s: %d captures in %d groups over %d placements", room,
                        len(index), group_report["n_groups"], group_report["n_placements"])
            split = select_splits(groups, n_groups=args.n_groups,
                                  n_val_groups=args.n_val_groups, n_train=args.n_train,
                                  n_diagnostic_groups=args.n_diagnostic_groups)

            poses, groups_meta = build_runtime_metadata(index, groups, split)
            selected_ids = [c for gk in (split["train_test_groups"] + split["val_groups"]
                                         + split["diagnostic_groups"])
                            for c in next(g for g in groups
                                          if g["group_key"] == gk)["capture_ids"]]
            role_of = {c: entry["split_role"] for c, entry in poses.items()}
            audits[room] = resample_and_write(
                room_dir, os.path.join(staged_runtime.staging_dir, room),
                selected_ids, roles=role_of)
            logger.info("%s: staged %d resampled RIRs (%d below %g dBFS)", room,
                        audits[room]["n_files"], audits[room]["n_silent"],
                        audits[room]["silence_threshold_db"])

            _write_json(staged_runtime.path(room, "metadata", "poses_metadata.json"), poses)
            _write_json(staged_runtime.path(room, "metadata", "groups_metadata.json"),
                        groups_meta)

            per_room[room] = {"groups": groups, "split": split,
                              "group_report": group_report, "crosscheck": crosscheck,
                              "rx_trailing_sentinel_dropped": sentinel_flag_of(index)}

        jsons = assemble_split_jsons(per_room)
        for name, payload in jsons.items():
            _write_json(staged_splits.path(f"{name}_base.json"), payload)
        params = {"seed": args.seed, "n_groups": args.n_groups,
                  "n_val_groups": args.n_val_groups, "n_train": args.n_train,
                  "n_diagnostic_groups": args.n_diagnostic_groups,
                  "rooms": list(args.rooms), "raf_root": args.raf_root,
                  "output_dir": args.output_dir,
                  "crosscheck": "full" if args.full_crosscheck else f"sample:{args.crosscheck_sample}",
                  "allow_nonuniform": bool(args.allow_nonuniform)}
        splits_record = build_splits_record(per_room, params)
        splits_record["readback_record"] = readback_provenance
        _write_json(staged_splits.path("raf_splits_record.json"), splits_record)
        _write_json(staged_splits.path("raf_amplitude_audit.json"),
                    {"params": params, "readback_record": readback_provenance,
                     "rooms": audits})

        expected_runtime = [f"{room}/metadata/{name}"
                            for room in args.rooms
                            for name in ("poses_metadata.json", "groups_metadata.json")]
        expected_splits = [f"{name}_base.json" for name in sorted(jsons)] + [
            "raf_splits_record.json", "raf_amplitude_audit.json"]
        runtime_manifest = staged_runtime.commit(expected=expected_runtime,
                                                 validate_json=True)
        splits_manifest = staged_splits.commit(expected=expected_splits,
                                               validate_json=True)

    logger.info("published %d runtime artifacts to %s and %d split artifacts to %s",
                runtime_manifest["n_files"], args.output_dir,
                splits_manifest["n_files"], args.split_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
