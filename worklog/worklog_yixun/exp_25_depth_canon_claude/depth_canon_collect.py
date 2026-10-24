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
import glob
import json
import math
import os
import re
import sys
from typing import NamedTuple


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


class CellPaths(NamedTuple):
    link: str
    metrics: str
    stream: str
    canon: str      # where a canon sidecar WOULD be (it must exist iff the cell canonicalizes)
    receipt: str
    dump_dir: str
    stem: str


def cell_paths(row, link, dump_root):
    """Every path of one cell, derived exactly as eval_FLAC derives them from ``link``."""
    m = eval_FLAC.build_output_paths(
        link, 1, 1.0, row["cell"], cond_method="vanilla", rotate_deg=float(row["rotate_deg"] or 0),
        n_angles=4, rotate_mode=row["rotate_mode"], rotate_seed=row["rotate_seed"], are_lambda=None,
        canonicalize=row["canonicalize"], canon_mismatch=row["canon_mismatch"])["metrics"]
    return CellPaths(link, m, eval_FLAC.stream_sidecar_path(m), eval_FLAC.canon_sidecar_path(m),
                     m[:-len(".json")] + ".cell.json", eval_FLAC.scored_waveform_dir(dump_root, m),
                     os.path.basename(m)[:-len(".json")])


def expected_checkpoint(row, manifest, canon_index):
    """``(target path, sha256)`` the cell must have been evaluated on."""
    ck = manifest["arms"][row["arm"]]["ckpt"]
    if ck["kind"] == "endpoint":
        return ck["path"], ck["sha256"]
    if not canon_index or row["step"] not in canon_index:
        raise KeyError(f"no CANON checkpoint index entry for step {row['step']} (the launcher's "
                       "depth_canon_*_exp25_CANON_checkpoints.sha256 is required for CANON cells)")
    return canon_index[row["step"]]


def load_canon_index(path, run_ns):
    """``{step: (path, sha256)}`` from the launcher's FULL checkpoint manifest (``sha256sum``
    lines). Every line must be one checkpoint of the CANON run namespace; any other line,
    a malformed digest or a duplicate step is a ``ValueError`` (fail-closed)."""
    pat = re.compile(re.escape(run_ns.rstrip("/")) + r"/checkpoints/epoch=\d+-step=(\d+)\.ckpt")
    index = {}
    with open(path) as f:
        for line in f.read().splitlines():
            sha, sep, ckpt = line.partition("  ")
            m = pat.fullmatch(ckpt) if sep and eval_FLAC._is_sha256_hex(sha) else None
            if m is None:
                raise ValueError(f"{path}: {line!r} is not a checkpoint of the CANON run {run_ns}")
            if int(m.group(1)) in index:
                raise ValueError(f"{path}: duplicate step {m.group(1)}")
            index[int(m.group(1))] = (ckpt, sha)
    return index


def find_canon_manifest(expdir=EXPDIR):
    """The ONE launcher checkpoint manifest in the experiment folder, else ``FileNotFoundError``."""
    hits = sorted(glob.glob(os.path.join(expdir, "depth_canon_*_exp25_CANON_checkpoints.sha256")))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one depth_canon_*_exp25_CANON_checkpoints.sha256 in "
                                f"{expdir}, found {len(hits)}: pass --canon-ckpt-manifest")
    return hits[0]


def verify_cell(stem, *, manifest, farm_root, dump_root, pin, canon_index=None, base_dir=REPO,
                dump_check="full", expect_dump=True, check_link=True):
    """Named reasons the cell ``stem`` is not complete-and-verified ([] = the driver may skip
    it, the generator may publish it). Checks the metrics record, stream and canon sidecars,
    the driver's receipt, the dump (``dump_check`` 'full' = ``is_complete`` once, or
    'manifest'), and -- ``check_link`` -- that the farm link still points at the checkpoint.
    Relative artifact paths resolve against ``base_dir`` (the evaluator's cwd)."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin)):
        return [f"pin {pin!r} is not a full 40-hex commit"]
    ckpt_name, _, suffix = stem.partition("_metrics_")
    rows = [c for c in manifest["cells"] if c["metrics_suffix"] == suffix]
    if len(rows) != 1 or not ckpt_name:
        return [f"{stem!r} names no registered cell of the manifest ({len(rows)} suffix matches)"]
    row = rows[0]
    try:
        target, sha = expected_checkpoint(row, manifest, canon_index)
    except KeyError as err:
        return [str(err).strip("'\"")]
    if os.path.basename(target) != ckpt_name + ".ckpt":
        return [f"{stem!r}: checkpoint {ckpt_name!r} is not this cell's {os.path.basename(target)!r}"]
    link = os.path.join(farm_root, row["arm"], os.path.basename(target))
    paths = cell_paths(row, link, dump_root)
    at = lambda p: os.path.join(base_dir, p)  # noqa: E731 -- an absolute p stays absolute
    docs, reasons = {}, []
    wanted = [("metrics", paths.metrics), ("stream", paths.stream), ("receipt", paths.receipt)]
    for key, path in wanted + ([("canon", paths.canon)] if row["canonicalize"] else []):
        try:
            with open(at(path)) as f:
                docs[key] = json.load(f)
        except (OSError, ValueError) as err:
            reasons.append(f"missing or unreadable {key} artifact {path} ({type(err).__name__})")
    if reasons:
        return reasons
    if not row["canonicalize"] and os.path.lexists(at(paths.canon)):
        reasons.append(f"unexpected canon sidecar {paths.canon} for a cell without canonicalization")
    reasons += record_reasons(docs["metrics"], row, pin=pin, link=link)
    reasons += stream_reasons(docs["stream"], row, docs["metrics"])
    if row["canonicalize"]:
        reasons += canon_reasons(docs["canon"], row, docs["stream"], docs["metrics"])
    reasons += receipt_reasons(docs["receipt"], row, stem=stem, link=link, target=target,
                               ckpt_sha256=sha, dump=expect_dump, dump_root=dump_root, pin=pin)
    if expect_dump:
        canon = paths.canon if row["canonicalize"] else None
        expect = {"ckpt_path": link, "metrics_path": paths.metrics, "metrics_file": at(paths.metrics),
                  "stream_sidecar_path": paths.stream, "stream_file": at(paths.stream),
                  "canon_sidecar_path": canon, "canon_file": at(canon) if canon else None}
        reasons += dump_reasons(at(paths.dump_dir), row, stem=stem, stream=docs["stream"],
                                expect=expect, pin=pin, mode=dump_check)
    elif os.path.lexists(at(paths.dump_dir)) or staging_leftovers(at(paths.dump_dir)):
        reasons.append(f"a dump exists at {paths.dump_dir} for a cell run without one")
    if check_link:
        here = at(link)
        if not os.path.islink(here):
            reasons.append(f"farm link {link} is not a symlink")
        elif os.path.realpath(here) != os.path.realpath(target):
            reasons.append(f"farm link {link} -> {os.path.realpath(here)}, not {target}")
    return reasons


def select_cells(manifest, blocks=None, only=None, limit=None):
    """The rows to run, in the manifest's (priority) order: ``blocks`` (e.g. the A1 switch =
    every block but YAW-pin), ``only`` (cell names), ``limit`` (first N). Unknown or
    duplicate names, an empty list and a non-positive limit are ``ValueError``."""
    for label, names, universe in (("block", blocks, list(manifest["blocks"])),
                                   ("cell", only, [c["cell"] for c in manifest["cells"]])):
        if names is not None and (not names or len(set(names)) != len(names)
                                  or any(n not in universe for n in names)):
            raise ValueError(f"unknown, duplicate or empty {label} selection {names!r}")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"--limit must be a positive int, got {limit!r}")
    rows = [c for c in manifest["cells"] if (blocks is None or c["block"] in blocks)
            and (only is None or c["cell"] in only)]
    return rows[:limit] if limit else rows


def plan_cells(manifest, selected, *, farm_root, dump_root, canon_index, no_dump_blocks):
    """One driver entry per selected row: the checkpoint to verify (target + expected
    sha256 + NAS manifest key), the farm link, every artifact path, whether it dumps
    (``no_dump_blocks`` deselects a block's dumps, A3), and the exact evaluator argv."""
    bad = [b for b in no_dump_blocks if b not in manifest["blocks"]]
    if bad:
        raise ValueError(f"unknown --no-dump-blocks {bad}")
    out = []
    for row in selected:
        ck = manifest["arms"][row["arm"]]["ckpt"]
        target, sha = expected_checkpoint(row, manifest, canon_index)
        link = os.path.join(farm_root, row["arm"], os.path.basename(target))
        dump = bool(row["dump"]) and row["block"] not in no_dump_blocks
        p = cell_paths(row, link, dump_root)
        out.append({"cell": row["cell"], "block": row["block"], "arm": row["arm"], "step": row["step"],
                    "K": row["K"], "seed": row["seed"], "target": target, "sha256": sha,
                    "nas_manifest": ck.get("nas_manifest"), "nas_manifest_key": ck.get("nas_manifest_key"),
                    "link": link, "metrics": p.metrics, "stream": p.stream,
                    "canon": p.canon if row["canonicalize"] else None, "canon_probe": p.canon,
                    "receipt": p.receipt, "dump": dump, "dump_root": dump_root, "dump_dir": p.dump_dir,
                    "stem": p.stem, "expected_stream_count": row["expected_stream_count"],
                    "argv": cell_argv(row, link, dump_root, dump)})
    return out


def write_receipt(entry, row, *, base_dir, lane, log, rc, pin, ckpt_sha256):
    """Atomically write the cell's ``.cell.json`` (temp file + ``os.replace``). Refuses a
    measured checkpoint digest that is not the expected one (the driver must not get here)."""
    if ckpt_sha256 != entry["sha256"]:
        raise ValueError(f"{entry['cell']}: measured checkpoint sha256 {ckpt_sha256} != the expected "
                         f"{entry['sha256']}")
    rec = {"schema_version": RECEIPT_SCHEMA_VERSION, "cell": entry["cell"], "stem": entry["stem"],
           "block": row["block"], "arm": row["arm"], "step": row["step"], "ckpt_link": entry["link"],
           "ckpt_target": entry["target"], "ckpt_sha256": ckpt_sha256, "model_config": row["model_config"],
           "model_config_sha256": row["model_config_sha256"], "dataset_config": row["dataset_config"],
           "dataset_config_sha256": row["dataset_config_sha256"],
           "expected_stream_count": row["expected_stream_count"], "dump": entry["dump"],
           "dump_root": entry["dump_root"] if entry["dump"] else None, "argv": entry["argv"],
           "expect_head": pin, "lane": lane, "log": log, "rc": rc}
    path = os.path.join(base_dir, entry["receipt"])
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)
    return path


def _csv(value):
    return None if value is None else [v for v in value.split(",") if v]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-cell", metavar="STEM", help="verify ONE cell; exit 0 valid / 1 invalid")
    ap.add_argument("--pin", help="the campaign evaluator pin (40-hex)")
    ap.add_argument("--cells", default=CELLS_JSON)
    ap.add_argument("--farm-root", default=None, help="as passed to the evaluator (default: manifest)")
    ap.add_argument("--dump-root", default=None, help="the dump ROOT (default: manifest)")
    ap.add_argument("--base-dir", default=REPO, help="the evaluator's cwd (default: the repo root)")
    ap.add_argument("--canon-ckpt-manifest", default=None)
    ap.add_argument("--dump-check", choices=("full", "manifest"), default="full")
    ap.add_argument("--no-dump", action="store_true", help="the cell was run without a dump")
    ap.add_argument("--no-link-check", action="store_true")
    ap.add_argument("--plan", metavar="OUT", help="write the driver's plan JSON; exit 2 if refused")
    ap.add_argument("--blocks", default=None, help="comma list (default: every block)")
    ap.add_argument("--only", default=None, help="comma list of cell names")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-dump-blocks", default="", help="comma list of blocks run WITHOUT a dump")
    ap.add_argument("--write-receipt", metavar="CELL", help="write CELL's receipt from --plan-json")
    ap.add_argument("--plan-json", default=None)
    ap.add_argument("--lane", type=int, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--rc", type=int, default=None)
    ap.add_argument("--ckpt-sha", default=None, help="the checkpoint sha256 the driver MEASURED")
    args = ap.parse_args(argv)
    manifest = load_cells(args.cells)
    farm = args.farm_root or manifest["farm_root"]
    dump_root = args.dump_root or manifest["dump_root"]
    if args.verify_cell:
        suffix = args.verify_cell.partition("_metrics_")[2]
        arms = {c["arm"] for c in manifest["cells"] if c["metrics_suffix"] == suffix}
        index = None
        if arms == {"CANON"}:   # only CANON-checkpoint cells need the launcher's manifest
            index = load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                     manifest["arms"]["CANON"]["ckpt"]["run_ns"])
        reasons = verify_cell(args.verify_cell, manifest=manifest, farm_root=farm, dump_root=dump_root,
                              pin=args.pin, canon_index=index, base_dir=args.base_dir,
                              dump_check=args.dump_check, expect_dump=not args.no_dump,
                              check_link=not args.no_link_check)
        print(("VALID " if not reasons else "INVALID ") + args.verify_cell)
        for reason in reasons:
            print("  - " + reason)
        return 0 if not reasons else 1
    if args.plan:
        try:
            selected = select_cells(manifest, _csv(args.blocks), _csv(args.only), args.limit)
            index = None
            if any(c["arm"] == "CANON" for c in selected):
                try:
                    index = load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                             manifest["arms"]["CANON"]["ckpt"]["run_ns"])
                except (OSError, ValueError) as err:
                    raise ValueError(f"CANON cells need the launcher's checkpoint manifest: {err}") from err
            entries = plan_cells(manifest, selected, farm_root=farm, dump_root=dump_root,
                                 canon_index=index, no_dump_blocks=_csv(args.no_dump_blocks))
        except (ValueError, KeyError, OSError) as err:
            print(f"PLAN REFUSED: {err}")
            return 2
        with open(args.plan, "w") as f:
            json.dump({"schema_version": 1, "cells_manifest": os.path.abspath(args.cells),
                       "farm_root": farm, "dump_root": dump_root, "n_selected": len(entries),
                       "cells": entries}, f, indent=1)
        print(f"plan: {len(entries)} cell(s) -> {args.plan}")
        return 0
    if args.write_receipt:
        with open(args.plan_json) as f:
            entry = next(e for e in json.load(f)["cells"] if e["cell"] == args.write_receipt)
        row = next(c for c in manifest["cells"] if c["cell"] == args.write_receipt)
        print("receipt: " + write_receipt(entry, row, base_dir=args.base_dir, lane=args.lane,
                                          log=args.log, rc=args.rc, pin=args.pin,
                                          ckpt_sha256=args.ckpt_sha))
        return 0
    ap.error("nothing to do: pass --verify-cell, --plan or --write-receipt")
    return 2


if __name__ == "__main__":
    sys.exit(main())
