#!/usr/bin/env python3
"""exp_25 depth_canon -- the cell verifier and results collector (plan_depth_canon.md v9 §1
tiers + PASS rules, §5 cells / atomic completion / "Drivers:", §9 statistics).

``--verify-cell <stem>`` checks ONE cell's artifacts against its cell class in
``depth_canon_cells.json`` (the driver's resume predicate and the generator's exp25
contract); the full mode pairs the cells and writes ``depth_canon_results.{md,json}``.

Every check is fail-closed and named: a function returns the list of reasons the
artifact is not the cell it claims to be ([] = valid). Comparisons are TYPE-strict
(eval_FLAC's ``_canon_strict_diff``): ``1`` is not ``1.0`` is not ``true``.

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9.
"""
import json
import math
import os
import sys


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # this checkout's eval_FLAC / src, never a stale installed copy

import eval_FLAC  # noqa: E402
from src.data.yaw_rotation import TIE_CLASSES  # noqa: E402

EXPREL = "worklog/worklog_yixun/exp_25_depth_canon_claude"
EXPDIR = os.path.join(REPO, EXPREL)
CELLS_JSON = os.path.join(EXPDIR, "depth_canon_cells.json")
IMG_W = 512
# the evaluator's emission set for AR (as exp_11's validator pins it) and the table subset
METRIC_KEYS = ("T60", "C50", "EDT", "FD", "Invalid T60", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
               "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
TABLE_KEYS = ("T60", "C50", "EDT", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10")
TIER3_KEYS = ("T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
              "RIR_to_GT_RIR_R@10")  # plan §1 m_Z: T60, C50, EDT, R@1/5/10, FD
SCENE_KEYS = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
              "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants")  # both AR eval splits
# eval_FLAC.build_metrics_record's key order, per cell class
BASE_KEYS = ("metrics", "ckpt_path", "rotate_deg", "cond_method", "frame_avg_angles", "cond_autocast",
             "orbit_execution", "frame_avg_fwd_cap", "source_sha", "batch_size", "n_samples",
             "dataset_config", "seed", "cfg_scale", "steps", "eval_name", "weights_source", "device")
RANDOM_KEYS = ("rotate_mode", "rotate_seed", "input_hash", "assignment_hash", "stream_count", "img_w")
SCENE_RECORD_KEYS = ("by_scene", "per_scene_schema", "scene_count")
CANON_KEYS = tuple(eval_FLAC.CANON_RECORD_KEYS)


def load_cells(path=CELLS_JSON):
    with open(path) as f:
        return json.load(f)


def _eq(reasons, label, got, want):
    """Append a named reason unless ``got`` equals ``want`` TYPE-strictly."""
    if eval_FLAC._canon_strict_diff(got, want, label) is not None:
        reasons.append(f"{label}: got {got!r}, expected {want!r}")


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


_DIGESTS = {}


def model_config_digest(row):
    """eval_FLAC.model_config_sha256 of the row's COMMITTED model config (cached)."""
    path = os.path.join(REPO, row["model_config"])
    if path not in _DIGESTS:
        with open(path) as f:
            _DIGESTS[path] = eval_FLAC.model_config_sha256(json.load(f))
    return _DIGESTS[path]


def expected_record_keys(row):
    return (list(BASE_KEYS) + (list(RANDOM_KEYS) if row["rotate_mode"] == "random" else [])
            + list(SCENE_RECORD_KEYS) + (list(CANON_KEYS) if row["canon_in_play"] else []))


def _metric_block_reasons(label, block):
    if not isinstance(block, dict):
        return [f"{label} is not an object"]
    if set(block) != set(METRIC_KEYS):
        return [f"{label} metrics keys {sorted(block)} != the evaluator's emission set {sorted(METRIC_KEYS)}"]
    return [f"{label}[{k}] = {block[k]!r} is not a finite number" for k in METRIC_KEYS
            if not _finite(block[k])]


def record_reasons(rec, row, *, pin, link):
    """The metrics JSON against its cell class: every protocol field (announcement 05: never
    a default), the key ORDER eval_FLAC writes for this class, finite metrics over the
    emission set, the per-scene block over the ten AR families, and -- when
    canonicalization is in play -- the B2 binding fields, with the file and embedded
    config digests equal to each other AND to the committed config's digest."""
    if not isinstance(rec, dict):
        return ["metrics record is not a JSON object"]
    reasons, n = [], row["expected_stream_count"]
    want = expected_record_keys(row)
    if list(rec) != want:
        reasons.append(f"record key order {list(rec)} != {want} (eval_FLAC's for this cell class)")
    random = row["rotate_mode"] == "random"
    expect = {"ckpt_path": link, "rotate_deg": None if random else float(row["rotate_deg"]),
              "cond_method": "vanilla", "frame_avg_angles": None, "cond_autocast": "bf16",
              "orbit_execution": "n/a", "frame_avg_fwd_cap": None, "source_sha": pin, "batch_size": 64,
              "n_samples": n, "dataset_config": row["dataset_config"], "seed": row["seed"],
              "cfg_scale": 1.0, "steps": 1, "eval_name": row["cell"], "weights_source": "ema",
              "device": "cuda", "per_scene_schema": eval_FLAC.PER_SCENE_SCHEMA,
              "scene_count": len(SCENE_KEYS)}
    if random:
        expect.update(rotate_mode="random", rotate_seed=row["seed"], stream_count=n, img_w=IMG_W)
        reasons += [f"{k} {rec.get(k)!r} is not a sha256 digest" for k in ("input_hash", "assignment_hash")
                    if not eval_FLAC._is_sha256_hex(rec.get(k))]
    for key, value in expect.items():
        _eq(reasons, key, rec.get(key), value)
    reasons += _metric_block_reasons("metrics", rec.get("metrics"))
    by_scene = rec.get("by_scene")
    if not isinstance(by_scene, dict) or set(by_scene) != set(SCENE_KEYS):
        reasons.append(f"by_scene groups {sorted(by_scene) if isinstance(by_scene, dict) else by_scene!r}"
                       f" != the ten AR families {list(SCENE_KEYS)}")
    else:
        for scene in SCENE_KEYS:
            reasons += _metric_block_reasons(f"by_scene[{scene}]", by_scene[scene])
    if row["canon_in_play"]:
        applied = row["canonicalize"]
        for key, value in (("canonicalize_trained", row["canon_trained"]),
                           ("canonicalize_applied", applied),
                           ("canonicalize_mismatch", row["canon_mismatch"]),
                           ("canonicalize_img_w", IMG_W),
                           ("canonicalize_tiebreak", "image" if applied else None)):
            _eq(reasons, key, rec.get(key), value)
        counts = rec.get("canonicalize_tie_counts")
        if applied:
            if not (isinstance(counts, dict) and set(counts) == set(TIE_CLASSES)
                    and all(type(counts[c]) is int and counts[c] >= 0 for c in TIE_CLASSES)
                    and sum(counts.values()) == n):
                reasons.append(f"canonicalize_tie_counts {counts!r} are not {n} items over {TIE_CLASSES}")
        else:
            _eq(reasons, "canonicalize_tie_counts", counts, None)
        f, e = rec.get("model_config_sha256_file"), rec.get("model_config_sha256_embedded")
        if not eval_FLAC._is_sha256_hex(f) or f != e:
            reasons.append(f"model_config_sha256_file {f!r} != model_config_sha256_embedded {e!r}: the "
                           "checkpoint's embedded config and --model-config differ")
        elif f != model_config_digest(row):
            reasons.append(f"model_config_sha256 {f} != the committed {row['model_config']} digest")
    return reasons


def stream_reasons(stream, row, rec=None):
    """The ``.stream.json`` assignment audit against the cell class: schema, rotation,
    exactly ``n`` well-formed positions in order, offsets (fixed: the constant column
    shift of the angle; random: in-grid draws), recomputed hashes, and -- in random mode --
    the record's hashes equal to the sidecar's (two views of ONE stream)."""
    if not isinstance(stream, dict):
        return ["stream sidecar is not a JSON object"]
    reasons, n, random = [], row["expected_stream_count"], row["rotate_mode"] == "random"
    for key, value in (("schema_version", eval_FLAC.STREAM_SCHEMA_VERSION), ("fingerprint_schema", 1),
                       ("rotate_mode", row["rotate_mode"]), ("rotate_seed", row["rotate_seed"]),
                       ("rotate_deg", None if random else float(row["rotate_deg"])),
                       ("img_w", IMG_W), ("stream_count", n)):
        _eq(reasons, f"stream {key}", stream.get(key), value)
    inp, asg, offs = (stream.get(k) for k in ("input_tuples", "assignment_tuples", "offsets"))
    if not all(isinstance(x, list) for x in (inp, asg, offs)):
        return reasons + ["stream is missing input_tuples / assignment_tuples / offsets"]
    reasons += [f"stream {label} holds {len(seq)} positions, expected {n}"
                for label, seq in (("input_tuples", inp), ("assignment_tuples", asg), ("offsets", offs))
                if len(seq) != n]
    for i, t in enumerate(inp):
        if not (isinstance(t, list) and len(t) == 4 and t[0] == i and isinstance(t[1], str)
                and isinstance(t[2], list) and t[3] == IMG_W):
            reasons.append(f"stream input tuple at position {i} is malformed or out of order: {t!r}")
            break
    for i, t in enumerate(asg):
        if not (isinstance(t, list) and len(t) == 3 and t[0] == i and i < len(inp) and i < len(offs)
                and isinstance(inp[i], list) and len(inp[i]) > 1 and t[1] == inp[i][1] and t[2] == offs[i]):
            reasons.append(f"stream assignment tuple at position {i} disagrees with the input stream "
                           f"or the offsets: {t!r}")
            break
    if random:
        if any(type(o) is not int or not 0 <= o < IMG_W for o in offs):
            reasons.append(f"stream offsets outside the {IMG_W}-column grid")
    else:
        shift = eval_FLAC.yaw_column_shift(math.radians(row["rotate_deg"]), IMG_W)
        if any(type(o) is not int or o != shift for o in offs):
            reasons.append(f"stream offsets are not the constant column shift {shift} of "
                           f"{row['rotate_deg']} deg")
    _eq(reasons, "stream input_hash (recomputed)", stream.get("input_hash"),
        eval_FLAC.canonical_stream_hash(inp))
    _eq(reasons, "stream assignment_hash (recomputed)", stream.get("assignment_hash"),
        eval_FLAC.canonical_stream_hash(asg))
    if rec is not None and random:
        for key in ("input_hash", "assignment_hash"):
            _eq(reasons, f"record {key} vs stream", rec.get(key), stream.get(key))
    return reasons


CANON_SIDECAR_KEYS = ("schema_version", "img_w", "tiebreak", "stream_count", "tie_counts", "positions",
                      "canon_hash", "canonical_depth_hash")  # eval_FLAC.build_canon_record's order


def canon_reasons(canon, row, stream, rec):
    """The ``.canon.json`` sidecar: schema, ``n`` well-formed positions ``[i, target, dj,
    tie_class, depth_sha]`` whose targets are the stream's position by position, tie counts
    recounted and equal to the record's, and both digests recomputed from the preimages."""
    if not isinstance(canon, dict):
        return ["canon sidecar is not a JSON object"]
    reasons, n = [], row["expected_stream_count"]
    if list(canon) != list(CANON_SIDECAR_KEYS):
        reasons.append(f"canon sidecar keys {list(canon)} != {list(CANON_SIDECAR_KEYS)}")
    for key, value in (("schema_version", eval_FLAC.CANON_SIDECAR_SCHEMA_VERSION), ("img_w", IMG_W),
                       ("tiebreak", "image"), ("stream_count", n)):
        _eq(reasons, f"canon {key}", canon.get(key), value)
    pos = canon.get("positions")
    if not isinstance(pos, list) or len(pos) != n:
        return reasons + [f"canon positions hold {len(pos) if isinstance(pos, list) else pos!r}, expected {n}"]
    targets = [t[1] if isinstance(t, list) and len(t) > 1 else None
               for t in ((stream or {}).get("input_tuples") or [])]
    for i, p in enumerate(pos):
        if not (isinstance(p, list) and len(p) == 5 and p[0] == i and type(p[2]) is int
                and 0 <= p[2] < IMG_W and p[3] in TIE_CLASSES and eval_FLAC._is_sha256_hex(p[4])):
            reasons.append(f"canon position {i} is malformed: {p!r}")
            return reasons
        if i >= len(targets) or p[1] != targets[i]:
            reasons.append(f"canon position {i} names target {p[1]!r}, the stream evaluated "
                           f"{targets[i] if i < len(targets) else None!r}")
            break
    counts = {c: sum(p[3] == c for p in pos) for c in TIE_CLASSES}
    _eq(reasons, "canon tie_counts (recounted)", canon.get("tie_counts"), counts)
    _eq(reasons, "record canonicalize_tie_counts vs the canon sidecar recount",
        (rec or {}).get("canonicalize_tie_counts"), counts)
    _eq(reasons, "canon canon_hash (recomputed)", canon.get("canon_hash"),
        eval_FLAC.canonical_stream_hash(pos))
    _eq(reasons, "canon canonical_depth_hash (recomputed)", canon.get("canonical_depth_hash"),
        eval_FLAC.canonical_stream_hash([[p[1], p[4]] for p in pos]))
    return reasons


def _file_digest(path):
    return eval_FLAC._file_sha256(path) if path and os.path.isfile(path) else None


DUMP_FILES = ("COMPLETE", "manifest.json", "waveforms.npy", "waveforms.sha256")
DUMP_MANIFEST_KEYS = ("schema_version", "n", "dtype", "shape", "sha256", "input_hash", "metrics_stem",
                      "source_sha", "ckpt_path", "metrics_path", "metrics_sha256", "stream_sidecar_path",
                      "stream_sidecar_sha256", "canon_sidecar_path", "canon_sidecar_sha256")


def staging_leftovers(dump_dir):
    """Sibling ``<dump_dir>.staging-*`` entries (a crashed or failed finalize)."""
    parent, name = os.path.split(os.path.abspath(dump_dir))
    try:
        return sorted(e for e in os.listdir(parent) if e.startswith(name + ".staging-"))
    except OSError:
        return []


def dump_reasons(dump_dir, row, *, stem, stream, expect, pin, mode="full"):
    """The published waveform dump of one cell. ``mode='full'`` runs the evaluator's verified
    predicate ``ScoredWaveformDump.is_complete(dir, expected_n)`` (streams the whole array
    ONCE -- never call this in a poll loop); ``'manifest'`` reads only the marker + manifest.
    Both then bind the manifest to THIS cell: the stream's input_hash, the stem, the pin,
    the checkpoint link and the paths + sha256 digests of the metrics / stream / canon files
    as they are on disk now. ``expect``: the evaluator's path strings + the files' locations."""
    if mode not in ("full", "manifest"):
        raise ValueError(f"dump check mode must be 'full' or 'manifest', got {mode!r}")
    reasons = [f"leftover staging directory {s}" for s in staging_leftovers(dump_dir)]
    if os.path.islink(dump_dir) or not os.path.isdir(dump_dir):
        return reasons + [f"dump directory {dump_dir} is absent (or not a plain directory)"]
    n = row["expected_stream_count"]
    if sorted(os.listdir(dump_dir)) != sorted(DUMP_FILES):
        reasons.append(f"dump directory holds {sorted(os.listdir(dump_dir))}, expected {list(DUMP_FILES)}")
    if mode == "full" and not eval_FLAC.ScoredWaveformDump.is_complete(dump_dir, n):
        return reasons + [f"dump is not complete (ScoredWaveformDump.is_complete, expected_n={n})"]
    try:
        with open(os.path.join(dump_dir, "manifest.json")) as f:
            m = json.load(f)
        with open(os.path.join(dump_dir, eval_FLAC.SCORED_DUMP_MARKER)) as f:
            marker = f.read()
    except (OSError, ValueError) as err:
        return reasons + [f"dump COMPLETE marker / manifest unreadable ({err})"]
    if not isinstance(m, dict):
        return reasons + ["dump manifest is not a JSON object"]
    if list(m) != list(DUMP_MANIFEST_KEYS):
        reasons.append(f"dump manifest keys {list(m)} != {list(DUMP_MANIFEST_KEYS)}")
    if marker != f"{m.get('sha256')}\n" or not eval_FLAC._is_sha256_hex(m.get("sha256")):
        reasons.append("dump COMPLETE marker does not hold the manifest's waveform sha256")
    shape = m.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(x) is int for x in shape)
            and shape[0] == n and shape[1] > 0):
        reasons.append(f"dump shape {shape!r} is not [{n}, T]")
    for key, want in (("schema_version", eval_FLAC.SCORED_DUMP_SCHEMA_VERSION), ("n", n),
                      ("dtype", "float32"),
                      ("input_hash", stream.get("input_hash") if isinstance(stream, dict) else None),
                      ("metrics_stem", stem), ("source_sha", pin), ("ckpt_path", expect["ckpt_path"]),
                      ("metrics_path", expect["metrics_path"]),
                      ("metrics_sha256", _file_digest(expect["metrics_file"])),
                      ("stream_sidecar_path", expect["stream_sidecar_path"]),
                      ("stream_sidecar_sha256", _file_digest(expect["stream_file"])),
                      ("canon_sidecar_path", expect["canon_sidecar_path"]),
                      ("canon_sidecar_sha256", _file_digest(expect["canon_file"]))):
        _eq(reasons, f"dump manifest {key}", m.get(key), want)
    return reasons


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_KEYS = ("schema_version", "cell", "stem", "block", "arm", "step", "ckpt_link", "ckpt_target",
                "ckpt_sha256", "model_config", "model_config_sha256", "dataset_config",
                "dataset_config_sha256", "expected_stream_count", "dump", "dump_root", "argv",
                "expect_head", "lane", "log", "rc")


def cell_argv(row, link, dump_root, dump):
    """The evaluator argv of one cell: the manifest's flags + the farm link (+ the dump ROOT)."""
    return list(row["eval_args"]) + ["--ckpt-path", link] + (
        ["--dump-scored-waveforms", dump_root] if dump else [])


def receipt_reasons(receipt, row, *, stem, link, target, ckpt_sha256, dump, dump_root, pin):
    """The driver's ``<metrics stem>.cell.json``: what was run, on which checkpoint (its
    sha256 verified BEFORE the run against the endpoint manifest / the CANON launcher's
    checkpoint manifest), at which pin, with which argv, and that the evaluator exited 0.
    The committed configs must still hash to the manifest's pins."""
    if not isinstance(receipt, dict):
        return ["receipt is not a JSON object"]
    reasons = []
    if list(receipt) != list(RECEIPT_KEYS):
        reasons.append(f"receipt keys {list(receipt)} != {list(RECEIPT_KEYS)}")
    want = {"schema_version": RECEIPT_SCHEMA_VERSION, "cell": row["cell"], "stem": stem,
            "block": row["block"], "arm": row["arm"], "step": row["step"], "ckpt_link": link,
            "ckpt_target": target, "ckpt_sha256": ckpt_sha256, "model_config": row["model_config"],
            "model_config_sha256": row["model_config_sha256"], "dataset_config": row["dataset_config"],
            "dataset_config_sha256": row["dataset_config_sha256"],
            "expected_stream_count": row["expected_stream_count"], "dump": dump,
            "dump_root": dump_root if dump else None, "argv": cell_argv(row, link, dump_root, dump),
            "expect_head": pin, "rc": 0}
    for key, value in want.items():
        _eq(reasons, f"receipt {key}", receipt.get(key), value)
    if type(receipt.get("lane")) is not int or receipt["lane"] < 0:
        reasons.append(f"receipt lane {receipt.get('lane')!r} is not a GPU index")
    if not isinstance(receipt.get("log"), str) or not receipt["log"]:
        reasons.append("receipt names no evaluator log")
    for key in ("model_config", "dataset_config"):
        if _file_digest(os.path.join(REPO, row[key])) != row[key + "_sha256"]:
            reasons.append(f"{row[key]} no longer hashes to the manifest's {key}_sha256")
    return reasons
