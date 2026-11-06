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
import copy
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
EXP25_BLOCKS = ("T", "T-dup", "C4+45", "R", "D-i", "D-ii", "S", "P1-pin", "YAW-pin", "TRAJ")
EXP35_BLOCKS = ("CTL", "BF-pin", "BF-dup", "OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45",   # plan_orbitrir r2 §1
                "E2-FAIL")                                                                  # + §14 A1
EXPLICIT_BLOCKS = ("E2-FAIL",)   # §14 A1.3: selected only when named (the block or its cells), never by default
EXP36_BLOCKS = ("YAW-C4S",)      # exp_36 (plan_yaw_c4_spread.md r0; Codex plan review B1, H1): the Yaw-Aug C4 spread
DEFAULT_EXCLUDED_BLOCKS = EXPLICIT_BLOCKS + EXP36_BLOCKS   # Codex H1: never in a default selection (E2 semantics apart)
# exp_35 (plan_orbitrir_yaw_evidence.md r2 §3 P2): ONE per-arm evaluation protocol, vanilla by default and
# frame-averaged for B-F; it drives record validation, the artifact paths, the probe binding and rendering
VANILLA_PROTOCOL = {"cond_method": "vanilla", "frame_avg_angles": None, "orbit_execution": "n/a",
                    "frame_avg_fwd_cap": None, "cond_autocast": "bf16"}
ARM_PROTOCOL = {"BF": {"cond_method": "fa_invariant", "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
                       "orbit_execution": "batched", "frame_avg_fwd_cap": 64, "cond_autocast": "bf16"}}
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
LOAD_KEYS = tuple(eval_FLAC.CKPT_LOAD_KEYS)   # rB-F2 F2-6: the digest of the bytes torch.load deserialized


def arm_protocol(arm):
    """The protocol fields every metrics record of an ``arm`` cell must hold (a fresh copy)."""
    return copy.deepcopy(ARM_PROTOCOL.get(arm, VANILLA_PROTOCOL))


def is_exp35(row):
    """An exp_35 row: one of the blocks plan_orbitrir_yaw_evidence.md r2 §1 appends."""
    return row["block"] in EXP35_BLOCKS


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
    return (list(BASE_KEYS) + list(LOAD_KEYS) + (list(RANDOM_KEYS) if row["rotate_mode"] == "random" else [])
            + list(SCENE_RECORD_KEYS) + (list(CANON_KEYS) if row["canon_in_play"] else []))


def _metric_block_reasons(label, block):
    if not isinstance(block, dict):
        return [f"{label} is not an object"]
    if set(block) != set(METRIC_KEYS):
        return [f"{label} metrics keys {sorted(block)} != the evaluator's emission set {sorted(METRIC_KEYS)}"]
    return [f"{label}[{k}] = {block[k]!r} is not a finite number" for k in METRIC_KEYS
            if not _finite(block[k])]


def _load_reasons(label, doc):
    """rB-F2 F2-6: the evaluator's digest + length of the checkpoint bytes it deserialized."""
    sha, n = doc.get("ckpt_sha256_loaded"), doc.get("ckpt_bytes_loaded")
    return ([] if eval_FLAC._is_sha256_hex(sha) else [f"{label} ckpt_sha256_loaded {sha!r} is not a sha256 digest"]) + (
        [] if type(n) is int and n > 0 else [f"{label} ckpt_bytes_loaded {n!r} is not a positive int"])


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
    random, P = row["rotate_mode"] == "random", arm_protocol(row["arm"])
    expect = {"ckpt_path": link, "rotate_deg": None if random else float(row["rotate_deg"]),
              "cond_method": P["cond_method"], "frame_avg_angles": P["frame_avg_angles"],
              "cond_autocast": P["cond_autocast"], "orbit_execution": P["orbit_execution"],
              "frame_avg_fwd_cap": P["frame_avg_fwd_cap"], "source_sha": pin, "batch_size": 64,
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
    reasons += _load_reasons("metrics record", rec)
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
        if not _context_ok(t[2], row["K"]):   # rB-F1 NIT a: K from the cell's dataset config
            reasons.append(f"stream position {i}: context_ids {t[2]!r:.80} are not exactly K={row['K']} nonempty "
                           "strings")
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
    """Sibling ``<dump_dir>.staging-*`` entries (a crashed or failed finalize) -- except the
    ones the driver already parked as ``….incomplete-<ts>`` (quarantined, never deleted)."""
    parent, name = os.path.split(os.path.abspath(dump_dir))
    try:
        return sorted(e for e in os.listdir(parent) if e.startswith(name + ".staging-")
                      and ".incomplete-" not in e[len(name):])
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
    return reasons + manifest_reasons(m, marker, row, stem=stem, stream=stream, expect=expect, pin=pin)


def manifest_reasons(m, marker, row, *, stem, stream, expect, pin, label="dump manifest"):
    """A dump manifest (+ its COMPLETE marker) bound to THIS cell; shared by the on-disk check
    and the receipt's offline dump evidence (rB-F1 B6)."""
    n, reasons = row["expected_stream_count"], []
    if not isinstance(m, dict):
        return [f"{label} is not a JSON object"]
    if list(m) != list(DUMP_MANIFEST_KEYS):
        reasons.append(f"{label} keys {list(m)} != {list(DUMP_MANIFEST_KEYS)}")
    if marker != f"{m.get('sha256')}\n" or not eval_FLAC._is_sha256_hex(m.get("sha256")):
        reasons.append(f"{label}: the COMPLETE marker does not hold the manifest's waveform sha256")
    shape = m.get("shape")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(x) is int for x in shape)
            and shape[0] == n and shape[1] > 0):
        reasons.append(f"{label} shape {shape!r} is not [{n}, T]")
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
        _eq(reasons, f"{label} {key}", m.get(key), want)
    return reasons


def dump_evidence(dump_dir):
    """What a receipt binds of a fully verified dump (rB-F1 B6, decision A7): the manifest's exact
    bytes (digest + content), the COMPLETE marker and the waveforms sidecar -- enough to
    re-verify the publication offline, without the NAS."""
    import hashlib
    with open(os.path.join(dump_dir, "manifest.json"), "rb") as f:
        raw = f.read()
    with open(os.path.join(dump_dir, eval_FLAC.SCORED_DUMP_MARKER)) as f:
        marker = f.read()
    with open(os.path.join(dump_dir, "waveforms.sha256")) as f:
        side = f.read()
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "manifest": json.loads(raw), "marker": marker,
            "waveforms_sidecar": side}


def evidence_reasons(ev, row, *, stem, stream, expect, pin):
    """The receipt's dump evidence, OFFLINE: self-consistent (the recorded manifest serializes,
    as the evaluator writes it, to ``manifest_sha256``), the marker and sidecar the evaluator
    writes for its waveform sha256, and the manifest bound to this cell and to the committed
    metrics / stream / canon files."""
    import hashlib
    if not (isinstance(ev, dict) and set(ev) == {"manifest_sha256", "manifest", "marker", "waveforms_sidecar"}):
        return [f"receipt dump_evidence {ev!r:.80} is missing or malformed"]
    m, out = ev["manifest"], []
    if not isinstance(m, dict) or hashlib.sha256(json.dumps(m, indent=2).encode()).hexdigest() != ev["manifest_sha256"]:
        out.append("receipt dump_evidence.manifest_sha256 is not the digest of the recorded manifest")
    out += manifest_reasons(m, ev["marker"], row, stem=stem, stream=stream, expect=expect, pin=pin,
                            label="receipt dump_evidence manifest")
    if ev["waveforms_sidecar"] != f"{m.get('sha256') if isinstance(m, dict) else None}  waveforms.npy\n":
        out.append("receipt dump_evidence waveforms sidecar is not the sha256sum line of the waveform digest")
    return out


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_KEYS = ("schema_version", "cell", "stem", "block", "arm", "step", "ckpt_link", "ckpt_target",
                "ckpt_sha256", "ckpt_cache", "ckpt_sha256_measured", "ckpt_sha256_loaded", "ckpt_bytes_loaded",
                "model_config", "model_config_sha256", "dataset_config",
                "dataset_config_sha256", "expected_stream_count", "dump", "dump_root", "argv",
                "expect_head", "campaign_digest", "spawn_digest", "post_digest", "artifact_sha256",
                "dump_evidence", "lane", "log", "rc")
# exp_35 (plan_orbitrir_yaw_evidence.md r2 §3b, §11 B1): every exp_35 cell runs through the committed no-flash
# launcher; its receipt (schema 2) appends the backend it ran on, and its log must prove it
RECEIPT_SCHEMA_VERSION_EXP35 = 2
RECEIPT_V2_KEYS = ("attention_backend", "no_flash_overlay_sha256", "no_flash_launcher_sha256")
RECEIPT_KEYS_V2 = RECEIPT_KEYS + RECEIPT_V2_KEYS
NO_FLASH = {"overlay": "overlays/no_flash/flash_attn.py",
            "overlay_sha256": "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023",
            "launcher": "overlays/no_flash/exp35_no_flash_launch.py",
            "launcher_sha256": "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de",
            "backend": "sdpa", "sentinel": "EXP35_NO_FLASH_LAUNCHER", "target": "eval_FLAC.py",
            "overlay_line": "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA",
            "fallback_line": "flash_attn not installed, disabling Flash Attention",   # src/models/transformer.py:16
            "missing_line": "No module named 'flash_attn'"}


def backend_fields():
    """The receipt-v2 backend fields of every exp_35 cell (the pinned overlay + launcher)."""
    return {"attention_backend": NO_FLASH["backend"], "no_flash_overlay_sha256": NO_FLASH["overlay_sha256"],
            "no_flash_launcher_sha256": NO_FLASH["launcher_sha256"]}


def backend_log_reasons(text, target=NO_FLASH["target"]):
    """The log of a launcher run proves the SDPA backend: transformer.py printed the overlay's ImportError
    and its fallback line exactly once, back to back; the launcher then printed ONE sentinel with the
    backend, the pinned overlay + launcher sha256 and ``target``; no line reports a MISSING flash_attn
    (an import that never met the overlay)."""
    lines, nf, out = text.splitlines(), NO_FLASH, []
    if any(nf["missing_line"] in ln for ln in lines):
        out.append(f"the log reports {nf['missing_line']!r}: an import ran without the overlay")
    ov = [i for i, ln in enumerate(lines) if ln == nf["overlay_line"]]
    fb = [i for i, ln in enumerate(lines) if ln == nf["fallback_line"]]
    if not (len(ov) == len(fb) == 1 and fb[0] == ov[0] + 1):
        out.append(f"the log holds {len(ov)} overlay ImportError line(s) and {len(fb)} SDPA fallback line(s), "
                   "not one back-to-back pair")
    sent = [i for i, ln in enumerate(lines) if ln.startswith(nf["sentinel"] + " ")]
    if len(sent) != 1:
        return out + [f"the log holds {len(sent)} no-flash launcher sentinel line(s), not exactly one"]
    got = dict(tok.partition("=")[::2] for tok in lines[sent[0]].split()[1:])
    want = {"attention_backend": nf["backend"], "overlay_sha256": nf["overlay_sha256"],
            "launcher_sha256": nf["launcher_sha256"], "target": target}
    if got != want:
        out.append(f"the launcher sentinel {got} != {want}")
    if fb and sent[0] < fb[-1]:
        out.append("the launcher sentinel precedes the SDPA fallback line")
    return out


def backend_log_file_reasons(path, target=NO_FLASH["target"]):
    try:
        with open(path, errors="replace") as f:
            return backend_log_reasons(f.read(), target)
    except (OSError, TypeError) as err:
        return [f"the cell log {path!r} is unreadable ({err}): the attention backend is unproven"]


DIGEST_KEYS = ("campaign_digest", "spawn_digest", "post_digest")   # rB-F1 B3: the evaluated tree


def ckpt_cache_path(link, sha256):
    """The content-addressed, verified copy a farm entry hard-links (rB-F1 B4)."""
    return os.path.join(os.path.dirname(os.path.dirname(link)), ".ckpt_cache", f"{sha256}.ckpt")


def cell_argv(row, link, dump_root, dump):
    """The evaluator argv of one cell: the manifest's flags + the farm link (+ the dump ROOT)."""
    return list(row["eval_args"]) + ["--ckpt-path", link] + (
        ["--dump-scored-waveforms", dump_root] if dump else [])


def receipt_reasons(receipt, row, *, stem, link, target, ckpt_sha256, dump, dump_root, pin, expect_digest=None,
                    expect=None, stream=None):
    """The driver's ``<metrics stem>.cell.json``: what was run, on which checkpoint (its
    sha256 verified BEFORE the run against the endpoint manifest / the CANON launcher's
    checkpoint manifest), at which pin, with which argv, and that the evaluator exited 0.
    The committed configs must still hash to the manifest's pins."""
    if not isinstance(receipt, dict):
        return ["receipt is not a JSON object"]
    reasons, keys = [], RECEIPT_KEYS_V2 if is_exp35(row) else RECEIPT_KEYS
    if list(receipt) != list(keys):
        reasons.append(f"receipt keys {list(receipt)} != {list(keys)}")
    want = {"schema_version": RECEIPT_SCHEMA_VERSION_EXP35 if is_exp35(row) else RECEIPT_SCHEMA_VERSION,
            "cell": row["cell"], "stem": stem,
            "block": row["block"], "arm": row["arm"], "step": row["step"], "ckpt_link": link,
            "ckpt_target": target, "ckpt_sha256": ckpt_sha256, "ckpt_cache": ckpt_cache_path(link, ckpt_sha256),
            "ckpt_sha256_measured": ckpt_sha256, "ckpt_sha256_loaded": ckpt_sha256, "model_config": row["model_config"],
            "model_config_sha256": row["model_config_sha256"], "dataset_config": row["dataset_config"],
            "dataset_config_sha256": row["dataset_config_sha256"],
            "expected_stream_count": row["expected_stream_count"], "dump": dump,
            "dump_root": dump_root if dump else None, "argv": cell_argv(row, link, dump_root, dump),
            "expect_head": pin, "rc": 0, **(backend_fields() if is_exp35(row) else {})}
    for key, value in want.items():
        _eq(reasons, f"receipt {key}", receipt.get(key), value)
    d = receipt.get("campaign_digest")
    if not eval_FLAC._is_sha256_hex(d):
        reasons.append(f"receipt campaign_digest {d!r} is not a sha256 digest")
    for key in DIGEST_KEYS[1:]:   # the relevant tree at the spawn / after the evaluator exited
        _eq(reasons, f"receipt {key} vs campaign_digest", receipt.get(key), d)
    if expect_digest is not None:
        _eq(reasons, "receipt campaign_digest vs this campaign's", d, expect_digest)
    if expect is not None:   # rB-F1 B6: the committed artifacts and the dump evidence, offline
        _eq(reasons, "receipt artifact_sha256 (the committed metrics / stream / canon now)",
            receipt.get("artifact_sha256"), {"metrics": _file_digest(expect["metrics_file"]),
                                             "stream": _file_digest(expect["stream_file"]),
                                             "canon": _file_digest(expect["canon_file"])})
        if dump:
            reasons += evidence_reasons(receipt.get("dump_evidence"), row, stem=stem, stream=stream,
                                        expect=expect, pin=pin)
        elif receipt.get("dump_evidence") is not None:
            reasons.append("receipt dump_evidence present for a cell run without a dump")
    reasons += _load_reasons("receipt", receipt)
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
    """Every path of one cell, derived exactly as eval_FLAC derives them from ``link`` (the arm's
    ``cond_method`` names it: a B-F cell carries ``_fa_invariant_a4``)."""
    P = arm_protocol(row["arm"])
    m = eval_FLAC.build_output_paths(
        link, 1, 1.0, row["cell"], cond_method=P["cond_method"], rotate_deg=float(row["rotate_deg"] or 0),
        n_angles=4 if P["frame_avg_angles"] is None else len(P["frame_avg_angles"]), rotate_mode=row["rotate_mode"],
        rotate_seed=row["rotate_seed"], are_lambda=None,
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
                dump_check="full", expect_dump=True, check_link=True, expect_digest=None, receipt_doc=None,
                backend_target=NO_FLASH["target"]):
    """Named reasons the cell ``stem`` is not complete-and-verified ([] = the driver may skip
    it, the generator may publish it). Checks the metrics record, stream and canon sidecars,
    the driver's receipt (its committed-artifact digests and dump evidence), the dump
    (``dump_check`` 'full' = ``is_complete`` once, 'manifest' = marker + manifest, both + the
    receipt's evidence == the dump on disk; 'receipt' = OFFLINE, the receipt's evidence stands
    for the dump, no NAS read), and -- ``check_link`` -- the farm entry's cache identity.
    ``receipt_doc``: a draft receipt verified before it is written. Relative artifact paths
    resolve against ``base_dir`` (the evaluator's cwd)."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin)):
        return [f"pin {pin!r} is not a full 40-hex commit"]
    ckpt_name, _, suffix = stem.partition("_metrics_")
    rows = [c for c in manifest["cells"] if c["metrics_suffix"] == suffix]
    if len(rows) != 1 or not ckpt_name:
        return [f"{stem!r} names no registered cell of the manifest ({len(rows)} suffix matches)"]
    row = rows[0]
    if is_exp35(row) != row["cell"].startswith("exp35_"):
        return [f"{row['cell']!r}: the name and the block {row['block']!r} name different campaigns"]
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
    if dump_check not in ("full", "manifest", "receipt"):
        raise ValueError(f"dump_check must be 'full', 'manifest' or 'receipt', got {dump_check!r}")
    if receipt_doc is not None:   # a draft receipt, verified BEFORE it is written (rB-F1 B6)
        docs["receipt"] = receipt_doc
    wanted = [("metrics", paths.metrics), ("stream", paths.stream)] + (
        [("receipt", paths.receipt)] if receipt_doc is None else [])
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
    canon = paths.canon if row["canonicalize"] else None
    expect = {"ckpt_path": link, "metrics_path": paths.metrics, "metrics_file": at(paths.metrics),
              "stream_sidecar_path": paths.stream, "stream_file": at(paths.stream),
              "canon_sidecar_path": canon, "canon_file": at(canon) if canon else None}
    reasons += receipt_reasons(docs["receipt"], row, stem=stem, link=link, target=target,
                               ckpt_sha256=sha, dump=expect_dump, dump_root=dump_root, pin=pin,
                               expect_digest=expect_digest, expect=expect, stream=docs["stream"])
    if isinstance(docs["metrics"], dict) and isinstance(docs["receipt"], dict):   # rB-F2 F2-6: loaded == receipt
        for key in LOAD_KEYS:
            _eq(reasons, f"metrics record {key} vs the receipt's", docs["metrics"].get(key), docs["receipt"].get(key))
    if is_exp35(row) and isinstance(docs["receipt"], dict):   # exp_35 §3b: its log re-read at every verification
        log = docs["receipt"].get("log")
        reasons += backend_log_file_reasons(at(log) if isinstance(log, str) else None, backend_target)
    if dump_check == "receipt":
        pass                       # offline: the receipt's evidence stands for the dump (decision A7)
    elif expect_dump:
        reasons += dump_reasons(at(paths.dump_dir), row, stem=stem, stream=docs["stream"],
                                expect=expect, pin=pin, mode=dump_check)
        try:
            if not isinstance(docs["receipt"], dict) or \
                    dump_evidence(at(paths.dump_dir)) != docs["receipt"].get("dump_evidence"):
                reasons.append("receipt dump_evidence != the dump on disk now")
        except (OSError, ValueError):
            pass                   # the dump itself is reported unreadable above
    elif os.path.lexists(at(paths.dump_dir)) or staging_leftovers(at(paths.dump_dir)):
        reasons.append(f"a dump exists at {paths.dump_dir} for a cell run without one")
    if check_link:   # rB-F1 B4: a regular file on the verified cache file's inode, never a symlink
        here, cache = at(link), at(ckpt_cache_path(link, sha))
        if os.path.islink(here):
            reasons.append(f"farm entry {link} is a symlink, not the hard link of the cache file {cache}")
        elif not os.path.isfile(here):
            reasons.append(f"farm entry {link} is absent or not a regular file")
        elif os.path.islink(cache) or not os.path.isfile(cache) or not os.path.samestat(os.stat(here), os.stat(cache)):
            reasons.append(f"farm entry {link} is not a hard link of the verified cache file {cache}")
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
            and (only is None or c["cell"] in only)
            and (blocks is not None or only is not None or c["block"] not in DEFAULT_EXCLUDED_BLOCKS)]
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
                    "link": link, "cache": ckpt_cache_path(link, sha), "metrics": p.metrics, "stream": p.stream,
                    "canon": p.canon if row["canonicalize"] else None, "canon_probe": p.canon,
                    "receipt": p.receipt, "dump": dump, "dump_root": dump_root, "dump_dir": p.dump_dir,
                    "stem": p.stem, "expected_stream_count": row["expected_stream_count"],
                    "argv": cell_argv(row, link, dump_root, dump), **({"no_flash": True} if is_exp35(row) else {})})
    return out


def build_receipt(entry, row, *, base_dir, lane, log, rc, pin, ckpt_sha256, ckpt_bytes, campaign_digest,
                  spawn_digest, post_digest, backend=None):
    """The cell's receipt (``.cell.json``) as a dict: refuses a measured checkpoint digest that
    is not the expected one and a tree digest at the spawn / after the run that is not the
    campaign's; binds the committed artifacts' digests and, for a dump, its evidence (B6); and
    (rB-F2 F2-6) the evaluator's digest of the bytes it deserialized, which must equal the
    expected one, with the length the driver measured (``ckpt_bytes``)."""
    if ckpt_sha256 != entry["sha256"]:
        raise ValueError(f"{entry['cell']}: measured checkpoint sha256 {ckpt_sha256} != the expected "
                         f"{entry['sha256']}")
    if backend != (backend_fields() if is_exp35(row) else None):   # exp_35: the runner's backend claim
        raise ValueError(f"{entry['cell']}: attention backend fields {backend!r} are not "
                         f"{backend_fields() if is_exp35(row) else None!r} for this cell")
    if not eval_FLAC._is_sha256_hex(campaign_digest) or {spawn_digest, post_digest} != {campaign_digest}:
        raise ValueError(f"{entry['cell']}: tree digest moved (campaign {campaign_digest}, spawn {spawn_digest}, "
                         f"post {post_digest})")
    at = lambda p: os.path.join(base_dir, p) if p else None  # noqa: E731
    artifacts = {"metrics": _file_digest(at(entry["metrics"])), "stream": _file_digest(at(entry["stream"])),
                 "canon": _file_digest(at(entry["canon"]))}
    if artifacts["metrics"] is None or artifacts["stream"] is None or (entry["canon"] and artifacts["canon"] is None):
        raise ValueError(f"{entry['cell']}: nothing to bind -- the metrics / stream / canon files are not all there")
    try:
        with open(at(entry["metrics"])) as f:
            met = json.load(f)
        loaded = {k: met.get(k) for k in LOAD_KEYS} if isinstance(met, dict) else {}
    except (OSError, ValueError) as err:
        raise ValueError(f"{entry['cell']}: the metrics record is unreadable ({err})") from err
    if loaded.get("ckpt_sha256_loaded") != entry["sha256"]:
        raise ValueError(f"{entry['cell']}: the evaluator loaded checkpoint bytes with sha256 "
                         f"{loaded.get('ckpt_sha256_loaded')!r}, not the expected {entry['sha256']}")
    if type(ckpt_bytes) is not int or ckpt_bytes <= 0 or loaded.get("ckpt_bytes_loaded") != ckpt_bytes:
        raise ValueError(f"{entry['cell']}: the evaluator loaded {loaded.get('ckpt_bytes_loaded')!r} checkpoint "
                         f"bytes, the driver measured {ckpt_bytes!r} bytes")
    try:
        evidence = dump_evidence(at(entry["dump_dir"])) if entry["dump"] else None
    except (OSError, ValueError) as err:
        raise ValueError(f"{entry['cell']}: the dump cannot be bound ({err})") from err
    return {"schema_version": RECEIPT_SCHEMA_VERSION_EXP35 if backend else RECEIPT_SCHEMA_VERSION,
            "cell": entry["cell"], "stem": entry["stem"],
           "block": row["block"], "arm": row["arm"], "step": row["step"], "ckpt_link": entry["link"],
           "ckpt_target": entry["target"], "ckpt_sha256": entry["sha256"],
           "ckpt_cache": ckpt_cache_path(entry["link"], entry["sha256"]), "ckpt_sha256_measured": ckpt_sha256,
           "ckpt_sha256_loaded": loaded["ckpt_sha256_loaded"], "ckpt_bytes_loaded": loaded["ckpt_bytes_loaded"],
           "model_config": row["model_config"],
           "model_config_sha256": row["model_config_sha256"], "dataset_config": row["dataset_config"],
           "dataset_config_sha256": row["dataset_config_sha256"],
           "expected_stream_count": row["expected_stream_count"], "dump": entry["dump"],
           "dump_root": entry["dump_root"] if entry["dump"] else None, "argv": entry["argv"],
           "expect_head": pin, "campaign_digest": campaign_digest, "spawn_digest": spawn_digest,
           "post_digest": post_digest, "artifact_sha256": artifacts, "dump_evidence": evidence,
           "lane": lane, "log": log, "rc": rc, **(backend or {})}


def write_receipt(entry, row, *, base_dir, receipt=None, **kw):
    """Atomically write the cell's receipt (``receipt`` = an already-built, verified dict, or
    built here from ``kw``) -- temp file + ``os.replace``."""
    rec = receipt if receipt is not None else build_receipt(entry, row, base_dir=base_dir, **kw)
    path = os.path.join(base_dir, entry["receipt"])
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# full mode: load every selected cell through verify_cell, pair, tiers
# --------------------------------------------------------------------------- #
def collect_cells(manifest, selected, *, farm_root, dump_root, pin, canon_index, base_dir=REPO,
                  check_link=True, no_dump_blocks=(), dump_check="full"):
    """``{cell: entry}`` in selection order; an entry is ``valid`` only if ``verify_cell``
    returned no reason, and only then carries the loaded record / stream / canon sidecar."""
    out = {}
    for row in selected:
        e = {"row": row, "valid": False, "missing": False, "reasons": [], "record": None,
             "stream": None, "canon": None, "receipt": None, "stem": None, "dump_dir": None,
             "dump": bool(row["dump"]) and row["block"] not in no_dump_blocks}
        out[row["cell"]] = e
        try:
            target, _ = expected_checkpoint(row, manifest, canon_index)
        except KeyError as err:
            e.update(missing=True, reasons=[str(err).strip("'\"")])
            continue
        p = cell_paths(row, os.path.join(farm_root, row["arm"], os.path.basename(target)), dump_root)
        e.update(stem=p.stem, dump_dir=os.path.join(base_dir, p.dump_dir))
        if not os.path.exists(os.path.join(base_dir, p.metrics)):
            e.update(missing=True, reasons=[f"metrics JSON {p.metrics} absent"])
            continue
        e["reasons"] = verify_cell(p.stem, manifest=manifest, farm_root=farm_root, dump_root=dump_root,
                                   pin=pin, canon_index=canon_index, base_dir=base_dir,
                                   dump_check=dump_check, expect_dump=e["dump"], check_link=check_link)
        if e["reasons"]:
            continue
        e["valid"] = True
        for key, path in (("record", p.metrics), ("stream", p.stream), ("receipt", p.receipt)) + (
                (("canon", p.canon),) if row["canonicalize"] else ()):
            with open(os.path.join(base_dir, path)) as f:
                e[key] = json.load(f)
    return out


def completeness(cells):
    """``(missing, invalid)`` cell names, in selection order."""
    return ([n for n, e in cells.items() if e["missing"]],
            [n for n, e in cells.items() if not e["missing"] and not e["valid"]])


def _ref_key(row):
    """The theta = 0 unseen reference a rotated cell pairs with: same arm, K, seed, step."""
    return (row["arm"], row["K"], row["seed"], row["step"])


def make_pairs(cells):
    """Every rotated cell with its theta = 0 partner of the same (arm, K, seed, step):
    C4 (90/180/270) and 45 vs T [seed 42], R vs T [seed], T-dup vs T, TRAJ theta vs TRAJ 0.
    ``input_hash_equal`` proves the pair saw identical items and context draws."""
    refs = {}
    for name, e in cells.items():
        r = e["row"]
        if e["valid"] and r["split"] == "unseen" and r["rotate_mode"] == "fixed" and r["rotate_deg"] == 0 \
                and r["block"] in ("T", "P1-pin", "YAW-pin", "TRAJ") and r["subblock"] in ("T", "TRAJ"):
            refs[_ref_key(r)] = name
    pairs = []
    for name, e in cells.items():
        r = e["row"]
        if r["block"] == "T-dup":
            kind = "dup"
        elif r["rotate_mode"] == "random" and r["block"] in ("R", "P1-pin", "YAW-pin"):
            kind = "R"
        elif r["rotate_mode"] == "fixed" and r["rotate_deg"] and r["split"] == "unseen" \
                and r["block"] in ("C4+45", "P1-pin", "YAW-pin", "TRAJ"):
            kind = "TRAJ" if r["block"] == "TRAJ" else ("45" if r["rotate_deg"] == 45 else "C4")
        else:
            continue
        ref = refs.get(_ref_key(r))
        same = bool(ref and e["valid"] and e["stream"]["input_hash"] == cells[ref]["stream"]["input_hash"])
        pairs.append({"arm": r["arm"], "kind": kind, "K": r["K"], "seed": r["seed"], "step": r["step"],
                      "deg": r["rotate_deg"] if r["rotate_mode"] == "fixed" else None, "cell": name,
                      "ref": ref, "valid": e["valid"], "input_hash_equal": same})
    return pairs


PAIRING_NOTE = ("input_hash equality proves that a pair saw the same targets and the same context draws in "
                "the same order; identical generative noise is not in the hash preimage -- it follows from the "
                "execution protocol: same arm / K / step / eval seed, batch size 64 and the same stream census "
                "under the one pinned evaluator, hence the same global torch.randn sequence")


def context_replacement(cells):
    """The registered context-uniqueness rule (rB-F1 NIT a): AR_md draws the K references without
    replacement unless the room has fewer than K other sources, and then WITH replacement (ids
    repeat). Which targets repeat is a property of the split, not of the seed (520 / 6,337 unseen
    K=8 targets in every one of exp_15's eleven streams), so every valid cell of a (split, K)
    must repeat on exactly the same targets."""
    by = {}
    for name, e in cells.items():
        if e["valid"]:
            r = e["row"]
            hit = tuple(sorted(t[1] for t in e["stream"]["input_tuples"] if len(set(t[2])) != len(t[2])))
            by.setdefault(f"{r['split']}/K{r['K']}", {}).setdefault(hit, []).append(name)
    return {key: {"cells": sum(len(v) for v in sets.values()),
                  "targets_with_repeated_context": len(next(iter(sets))) if len(sets) == 1 else None,
                  "consistent": len(sets) == 1} for key, sets in sorted(by.items())}


TIER1_KINDS = ("C4", "45", "R", "dup")


def _tier1_pair(p, cells):
    if not (p["valid"] and p["ref"]):
        return {**p, "n": 0, "n_equal": 0, "canonical_depth_hash_equal": False, "pass": False}
    pa, pb = cells[p["cell"]]["canon"]["positions"], cells[p["ref"]]["canon"]["positions"]
    n_equal = sum(1 for x, y in zip(pa, pb) if x[1] == y[1] and x[4] == y[4])
    digest = lambda pos: eval_FLAC.canonical_stream_hash([[x[1], x[4]] for x in pos])  # noqa: E731
    same = digest(pa) == digest(pb)
    return {**p, "n": len(pa), "n_equal": n_equal, "canonical_depth_hash_equal": same,
            "pass": bool(p["input_hash_equal"] and same and n_equal == len(pa) == len(pb))}


def tier1(pairs, cells):
    """Tier 1 (structural, exact): every registered rotated CANON cell (C4, 45, R) and T-dup
    holds the SAME canonical recovered-depth image as its theta = 0 pair, position by position
    and as the one-line ``canonical_depth_hash`` (recomputed from the positions). TRAJ pairs
    are reported descriptively (canonicalization does not depend on the checkpoint)."""
    reg = [_tier1_pair(p, cells) for p in pairs if p["arm"] == "CANON" and p["kind"] in TIER1_KINDS]
    traj = [_tier1_pair(p, cells) for p in pairs if p["arm"] == "CANON" and p["kind"] == "TRAJ"]
    unseen = {eval_FLAC.canonical_stream_hash([[x[1], x[4]] for x in e["canon"]["positions"]])
              for e in cells.values() if e["valid"] and e["canon"] and e["row"]["split"] == "unseen"}
    return {"pass": bool(reg) and all(p["pass"] for p in reg), "n_pairs": len(reg), "pairs": reg,
            "trajectory": {"n_pairs": len(traj), "n_pass": sum(p["pass"] for p in traj)},
            "distinct_canonical_depth_hashes_unseen": len(unseen)}


# Tier 2 (plan v9 §1, pre-registered): p99(relL2) <= 3 x the control p99 AND <= 1e-2; max > 1e-1
# is an ALARM (suspends an unqualified invariance claim pending item triage); T-dup exact.
TIER2_FACTOR, TIER2_CAP, TIER2_ALARM = 3.0, 1e-2, 1e-1
CONTROL_KEYS = {90: "d128", 180: "d256", 270: "d384", 45: "d64"}   # r3b review: bit-exact offsets
CONTROL_PROTOCOL = "eval"   # bf16 cond autocast, the cells' own protocol


def control_key(deg):
    """The composition-control offset a rotated cell is compared with (random -> 'random')."""
    return "random" if deg is None else CONTROL_KEYS[deg]


PROBE_CONTRACT = {   # plan v10 §1 Tier 2 (ii) + eps_cond, §5 "registered probe" (mirrors the probe's REGISTERED)
    "n_items": 64, "batch_size": 64, "seed": 42, "device": "cuda", "img_w": 512, "split": "unseen",
    "config_K": 8, "K": 8,   # the registered unseen K=8 config; its stream census: K context ids per item
    "offset_keys": ("d64", "d128", "d256", "d384", "random"), "stages": ("cond", "velocity", "latent", "waveform"),
    "protocols": {"fp32": {"cond_autocast": "off", "matmul_precision": "highest"},
                  "eval": {"cond_autocast": "bf16", "matmul_precision": "medium"}},
    "eps_threshold": 1e-5,
    "ckpt_step": 40000}   # rB-F3: the CANON checkpoint the probe is bound to (digest AND loaded length)


def _probe_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("depth_canon_cond_probe_view",
                                                  os.path.join(EXPDIR, "depth_canon_cond_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _context_ok(ids, k):
    """Exactly K nonempty strings (draws with replacement may repeat: see pairing_note)."""
    return isinstance(ids, list) and len(ids) == k and all(isinstance(x, str) and x for x in ids)


def _probe_number_reasons(rep, P):
    """eps_cond, control p99 and the verdict must be what the probe's own results imply."""
    res, keys, out = rep.get("results"), list(P["offset_keys"]), []
    if not (isinstance(res, dict) and list(res) == list(P["protocols"]) and all(
            isinstance(res[p], dict) and list(res[p]) == keys and all(
                isinstance(res[p][k], dict) and all(st in res[p][k] for st in P["stages"]) for k in keys)
            for p in res)):
        return ["probe results are not protocols x registered offsets x stages"]
    vals = [(v, k, i, pos) for k in keys for i, d in res["fp32"][k]["cond"]["per_id"].items()
            for pos, v in enumerate(d["relL2"])]
    eps = rep.get("eps_cond")
    if not (isinstance(eps, dict) and set(eps) == {"value", "threshold", "pass", "argmax"}):
        return out + [f"probe eps_cond {eps!r:.120} is missing or malformed"]
    if not (vals and all(_finite(v) for v, *_ in vals) and _finite(eps["value"]) and eps["value"] >= 0):
        return out + ["probe eps_cond (or the fp32 conditioning relL2 it is the max of) is not finite"]
    top = max(vals)
    _eq(out, "probe eps_cond value (max fp32 conditioning relL2 of the results)", eps["value"], top[0])
    _eq(out, "probe eps_cond argmax", eps["argmax"], {"key": top[1], "id": top[2], "position": top[3]})
    _eq(out, "probe eps_cond threshold (registered)", eps["threshold"], P["eps_threshold"])
    _eq(out, "probe eps_cond pass", eps["pass"], top[0] <= P["eps_threshold"])
    v = rep.get("verdict")
    if not (isinstance(v, dict) and set(v) == {"pass", "eps_cond", "tier1", "masks"}
            and all(type(x) is bool for x in v.values()) and v["eps_cond"] is eps["pass"]
            and v["pass"] is all(v[k] for k in ("eps_cond", "tier1", "masks"))):
        out.append(f"probe verdict {v!r:.120} is inconsistent with eps_cond")
    for p in P["protocols"]:
        want = {k: res[p][k]["waveform"]["summary"]["relL2"]["p99"] for k in keys}
        want["pooled"] = ((rep.get("pooled") or {}).get(p) or {}).get("waveform", {}).get("relL2", {}).get("p99")
        _eq(out, f"probe control_p99[{p}] (the results' waveform relL2 p99)", (rep.get("control_p99") or {}).get(p), want)
        out += [f"probe control p99 {p}/{k} = {x!r} is not a finite non-negative number" for k, x in want.items()
                if not (_finite(x) and x >= 0)]
    return out


def canon_endpoint_receipts(cells):
    """rB-F3 F3-2: the receipts (``<stem>.cell.json``) of every VERIFIED CANON@``ckpt_step`` cell of
    the selection -- each holds the digest + length of the checkpoint bytes its evaluator loaded."""
    step = PROBE_CONTRACT["ckpt_step"]
    return [e["receipt"] for e in cells.values()
            if e["valid"] and e["row"]["arm"] == "CANON" and e["row"]["step"] == step]


def _probe_length_reasons(prov, ckpt_sha256, receipts, label="CANON"):
    """rB-F3 F3-2 (Codex rA3/rB3 HIGH): the probe's ``ckpt_bytes_loaded`` == the one length every
    verified CANON@40k cell receipt holds for the same checkpoint. No such receipt, receipts that
    disagree or name other bytes -> the length is NOT verifiable (a reason, never a silent pass).
    exp_35: ``label="B-F"`` -- the same rule over the verified B-F@40k cell receipts."""
    arm, label, step = label, "probe provenance.ckpt_bytes_loaded", PROBE_CONTRACT["ckpt_step"]
    if not receipts:
        return [f"{label} is not verifiable: the selection holds no verified {arm}@{step} cell receipt "
                "(the length of the checkpoint bytes the campaign loaded)"]
    out = [f"{label} is not verifiable: a verified {arm}@{step} cell receipt's ckpt_sha256_loaded "
           f"{r.get('ckpt_sha256_loaded')!r} is not the probe's checkpoint {ckpt_sha256}"
           for r in receipts if r.get("ckpt_sha256_loaded") != ckpt_sha256]
    sizes = [r.get("ckpt_bytes_loaded") for r in receipts]
    bad = [n for n in sizes if type(n) is not int or n <= 0]
    if bad:
        return out + [f"{label} is not verifiable: a verified {arm}@{step} cell receipt's ckpt_bytes_loaded "
                      f"{bad[0]!r} is not a positive int"]
    if len(set(sizes)) != 1:
        return out + [f"{label} is not verifiable: the verified {arm}@{step} cell receipts disagree on "
                      f"ckpt_bytes_loaded ({sorted(set(sizes))})"]
    _eq(out, f"{label} (vs the verified {arm}@{step} cell receipts' ckpt_bytes_loaded)",
        prov.get("ckpt_bytes_loaded"), sizes[0])
    return out


def probe_reasons(rep, *, ckpt_sha256, canon_receipts, pin, manifest):
    """The registered composition control of THIS campaign (Codex rB HIGH B5): registered, no
    deviation, the CANON@40k checkpoint, the campaign pin on a clean tree, the registered
    arguments, the campaign's configs + split, the EMA canonical load, the offsets the probe's
    own draw gives, the stream census, and eps / control / verdict consistent with the results.
    rB-F3 F3-2: the bytes the probe deserialized -- ``ckpt_sha256_loaded`` == the manifest sha,
    ``ckpt_bytes_loaded`` == the length ``canon_receipts`` (:func:`canon_endpoint_receipts`) hold."""
    P, out = PROBE_CONTRACT, []
    _eq(out, "probe schema_version", rep.get("schema_version"), 1)
    if rep.get("registered") is not True:
        out.append(f"control probe is not a registered control (registered: {rep.get('registered')!r})")
    _eq(out, "probe deviations", rep.get("deviations"), [])
    prov = rep.get("provenance") if isinstance(rep.get("provenance"), dict) else {}
    canon = manifest["arms"]["CANON"]
    ds = next(c for c in manifest["cells"] if c["split"] == P["split"] and c["K"] == P["config_K"])
    split_json = manifest["splits"][P["split"]]["split_json"]
    digest = model_config_digest(next(c for c in manifest["cells"] if c["arm"] == "CANON"))
    want = {"ckpt_sha256": ckpt_sha256, "ckpt_sha256_loaded": ckpt_sha256,
            "source_sha": pin, "dirty": "", "device": P["device"], "seed": P["seed"],
            "random_offset_seed": P["seed"] + 1, "n_items": P["n_items"], "batch_size": P["batch_size"],
            "img_w": P["img_w"], "protocols": P["protocols"], "allow_canon_mismatch": False,
            "model_config_file_sha256": canon["model_config_sha256"], "dataset_config": ds["dataset_config"],
            "dataset_config_sha256": ds["dataset_config_sha256"], "split_json": split_json,
            "split_json_sha256": _file_digest(os.path.join(REPO, split_json)),
            "load": {"weights_source": "ema", "canonicalize_trained": True, "canonicalize_applied": True,
                     "canonicalize_mismatch": False, "img_w": P["img_w"], "model_config_sha256_file": digest,
                     "model_config_sha256_embedded": digest}}
    for key, value in want.items():
        _eq(out, f"probe provenance.{key}", prov.get(key), value)
    out += _load_reasons("probe provenance", prov) + _probe_length_reasons(prov, ckpt_sha256, canon_receipts)
    if not isinstance(prov.get("model_config"), str) or os.path.realpath(os.path.join(REPO, prov["model_config"])) \
            != os.path.realpath(os.path.join(REPO, canon["model_config"])):
        out.append(f"probe provenance.model_config {prov.get('model_config')!r} is not {canon['model_config']}")
    _eq(out, "probe offsets (the probe's own draw)", rep.get("offsets"),
        _probe_module().draw_offsets(P["n_items"], P["img_w"], P["seed"]))
    items = rep.get("items")
    if not (isinstance(items, list) and len(items) == P["n_items"] and all(
            isinstance(it, dict) and it.get("position") == i and isinstance(it.get("target_id"), str)
            and it["target_id"] and _context_ok(it.get("context_ids"), P["K"]) for i, it in enumerate(items))):
        out.append(f"probe items: not {P['n_items']} ordered positions with a target and K={P['K']} context ids")
    else:
        _eq(out, "probe stream_input_hash (recomputed from the items)", rep.get("stream_input_hash"),
            eval_FLAC.canonical_stream_hash([[it["position"], it["target_id"], it["context_ids"], P["img_w"]]
                                             for it in items]))
    return out + _probe_number_reasons(rep, P)


def load_control(path, *, ckpt_sha256, canon_receipts, pin, manifest):
    """``(control_p99, reasons, eps)`` from the composition-control probe JSON: only a valid
    registered artifact of this campaign (``probe_reasons``) with a matching sidecar is a
    control; ``eps`` = ``{value, threshold, verdict PASS|FAIL}`` (a legitimate FAIL stays FAIL).
    ``canon_receipts``: :func:`canon_endpoint_receipts` of the collected cells (rB-F3, required)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            text = f.read()
        with open(path + ".sha256") as f:
            side = f.read()
        rep = json.loads(text)
    except (OSError, ValueError) as err:
        return None, [f"control probe {path} unreadable ({err})"], None
    reasons = []
    if side != f"{hashlib.sha256(text).hexdigest()}  {os.path.basename(path)}\n":
        reasons.append(f"control probe sha256 sidecar does not match {path}")
    if not isinstance(rep, dict):
        return None, reasons + ["control probe is not a JSON object"], None
    reasons += probe_reasons(rep, ckpt_sha256=ckpt_sha256, canon_receipts=canon_receipts, pin=pin, manifest=manifest)
    if reasons:
        return None, reasons, None
    eps = rep["eps_cond"]
    return ({k: float(v) for k, v in rep["control_p99"][CONTROL_PROTOCOL].items()}, [],
            {"value": eps["value"], "threshold": eps["threshold"], "verdict": "PASS" if eps["pass"] else "FAIL"})


# exp_35 (plan_orbitrir_yaw_evidence.md r2 §5, §11 H3, r3 §12 B3): the B-F composition control (mirrors the
# probe's REGISTERED_BF): quarter turns only, FA conditioning, no canonicalization, launched through the no-flash
# launcher; Tier 2-C4 compares 90/180/270 cells with d128/d256/d384 (eval protocol) and the pooled value
PROBE_CONTRACT_BF = dict(PROBE_CONTRACT, offset_keys=("d128", "d256", "d384"), offsets=(128, 256, 384),
                         cond_method="fa_invariant", frame_avg_angles=[0.0, 90.0, 180.0, 270.0],
                         frame_avg_max_fwd_samples=64, ckpt_bytes=723922667, target=f"{EXPREL}/depth_canon_cond_probe.py")
BF_CONTROL_KEYS = {90: "d128", 180: "d256", 270: "d384"}
BF_PROVENANCE_KEYS = ("ckpt_path", "ckpt_sha256", "ckpt_sha256_loaded", "ckpt_bytes_loaded", "model_config",
                      "model_config_file_sha256", "dataset_config", "dataset_config_sha256", "split_json",
                      "split_json_sha256", "source_sha", "dirty", "device", "torch", "cuda_device", "seed", "noise",
                      "n_items", "batch_size", "num_workers", "img_w", "latent_shape", "protocols", "load",
                      "allow_canon_mismatch", "cond_method", "frame_avg_angles", "frame_avg_max_fwd_samples", "launch")


def _bf_vector_reasons(node, label, n):
    """One stage record: ``relL2`` / ``relLinf`` exactly ``n`` finite, non-negative values each, and its summary the
    recomputation from those items (Codex C3a BLOCKING 1: no stored summary is trusted)."""
    from src.tools import depth_canon_stats as stats
    if not (isinstance(node, dict) and all(isinstance(node.get(m), list) and len(node[m]) == n and all(
            _finite(x) and x >= 0 for x in node[m]) for m in ("relL2", "relLinf"))):
        return [f"{label}: relL2 / relLinf are not {n} finite, non-negative values each"]
    out = []
    _eq(out, f"{label}.summary (recomputed from its items)", node.get("summary"),
        {m: stats.paired_summary(node[m]) for m in ("relL2", "relLinf")})
    return out


def _bf_block_reasons(blk, label, P, ids):
    """One protocol x quarter turn (or determinism) record: the four stages, the conditioner ids, a BOOL mask, and
    every per-id / all / stage vector with its summary."""
    if not (isinstance(blk, dict) and list(blk) == list(P["stages"]) and isinstance(blk["cond"], dict)):
        return [f"{label}: not the stages {list(P['stages'])}"]
    c = blk["cond"]
    out = [] if type(c.get("masks_equal")) is bool else [f"{label}.cond.masks_equal {c.get('masks_equal')!r} is not a bool"]
    if not (c.get("ids") == ids and isinstance(c.get("per_id"), dict) and sorted(c["per_id"]) == ids):
        return out + [f"{label}.cond ids / per_id {c.get('ids')!r} are not the conditioner ids {ids}"]
    for lab, node in [(f"{label}.cond.per_id.{i}", c["per_id"][i]) for i in ids] + [(f"{label}.cond.all", c.get("all"))] \
            + [(f"{label}.{st}", blk[st]) for st in P["stages"][1:]]:
        out += _bf_vector_reasons(node, lab, P["n_items"])
    return out


def _bf_number_reasons(rep, P):
    """Every number of the B-F report is RECOMPUTED from its item vectors (Codex C3a BLOCKING 1): each vector holds
    exactly n_items finite, non-negative values; every summary, the pooled block, the control p99 (per offset and
    pooled), eps_cond (reported, NO verdict) and the verdict must equal the recomputation -- and the verdict hold."""
    from src.tools import depth_canon_stats as stats
    res, det, keys, protos = rep.get("results"), rep.get("determinism"), list(P["offset_keys"]), list(P["protocols"])
    if not (isinstance(res, dict) and list(res) == protos and isinstance(det, dict) and list(det) == protos
            and all(isinstance(res[p], dict) and list(res[p]) == keys and isinstance(res[p][k], dict)
                    for p in protos for k in keys)):
        return ["probe results / determinism are not protocols x quarter turns x stages"]
    ids = (res[protos[0]][keys[0]].get("cond") or {}).get("ids")
    if not (isinstance(ids, list) and ids and all(isinstance(i, str) for i in ids) and ids == sorted(set(ids))):
        return [f"probe conditioner ids {ids!r} are not a sorted, non-empty list of names"]
    out = [x for p in protos for k in keys for x in _bf_block_reasons(res[p][k], f"results.{p}.{k}", P, ids)]
    out += [x for p in protos for x in _bf_block_reasons(det[p], f"determinism.{p}", P, ids)]
    if out:
        return out
    rows = lambda p, st: [res[p][k]["cond"]["all"] if st == "cond" else res[p][k][st] for k in keys]  # noqa: E731
    pooled = {p: {st: {m: stats.paired_summary([v for r in rows(p, st) for v in r[m]]) for m in ("relL2", "relLinf")}
                  for st in P["stages"]} for p in protos}
    _eq(out, "probe pooled (recomputed from the items)", rep.get("pooled"), pooled)
    _eq(out, "probe control_p99 (recomputed from the waveform relL2 items)", rep.get("control_p99"), {
        p: dict({k: stats.paired_summary(res[p][k]["waveform"]["relL2"])["p99"] for k in keys},
                pooled=pooled[p]["waveform"]["relL2"]["p99"]) for p in protos})
    top = max((v, k, i, pos) for k in keys for i, d in res["fp32"][k]["cond"]["per_id"].items()
              for pos, v in enumerate(d["relL2"]))
    _eq(out, "probe eps_cond (reported without a verdict)", rep.get("eps_cond"),
        {"value": top[0], "argmax": {"key": top[1], "id": top[2], "position": top[3]}})
    masks = all(b["cond"]["masks_equal"] for p in protos for b in [res[p][k] for k in keys] + [det[p]])
    exact = all(v == 0.0 for p in protos for st in P["stages"]
                for v in (det[p]["cond"]["all"] if st == "cond" else det[p][st])["relL2"])
    _eq(out, "probe verdict (masks equal AND the determinism floor exact)", rep.get("verdict"),
        {"pass": masks and exact, "masks": masks, "determinism": exact})
    return out + ([] if masks and exact else ["probe verdict FAILS: the B-F control is invalid (a hard stop)"])


def probe_bf_reasons(rep, *, pin, campaign_digest, manifest, bf_receipts=None, base_dir=REPO):
    """The B-F composition control of THIS campaign: registered, no deviation, arm BF, the B-F@40k bytes
    (digest + the registered loaded length; ``bf_receipts``: those of the verified B-F cells, when given),
    the pin on a clean tree, the registered arguments and frame-average recipe, the campaign's configs and
    split, an EMA load WITHOUT canonicalization, the quarter-turn offsets, the stream census, numbers
    consistent with the results, and the launch provenance: the pinned overlay + launcher, the campaign
    digest, and the closed log (its sha256 re-checked) carrying the backend evidence for the probe target."""
    P, out = PROBE_CONTRACT_BF, []
    for key, want in (("schema_version", 1), ("arm", "BF"), ("deviations", [])):
        _eq(out, f"probe {key}", rep.get(key), want)
    if rep.get("registered") is not True:
        out.append(f"the B-F probe is not a registered control (registered: {rep.get('registered')!r})")
    prov = rep.get("provenance") if isinstance(rep.get("provenance"), dict) else {}
    if list(prov) != list(BF_PROVENANCE_KEYS):
        out.append(f"probe provenance keys {list(prov)} != {list(BF_PROVENANCE_KEYS)}")
    bf, ds = manifest["arms"]["BF"], next(c for c in manifest["cells"] if c["split"] == P["split"] and c["K"] == P["config_K"])
    digest = model_config_digest(next(c for c in manifest["cells"] if c["arm"] == "BF"))
    split_json, sha = manifest["splits"][P["split"]]["split_json"], bf["ckpt"]["sha256"]
    want = {"ckpt_sha256": sha, "ckpt_sha256_loaded": sha, "ckpt_bytes_loaded": P["ckpt_bytes"], "source_sha": pin,
            "dirty": "", "device": P["device"], "seed": P["seed"], "n_items": P["n_items"], "batch_size": P["batch_size"],
            "img_w": P["img_w"], "protocols": P["protocols"], "allow_canon_mismatch": False,
            "cond_method": P["cond_method"], "frame_avg_angles": P["frame_avg_angles"],
            "frame_avg_max_fwd_samples": P["frame_avg_max_fwd_samples"],
            "model_config_file_sha256": bf["model_config_sha256"], "dataset_config": ds["dataset_config"],
            "dataset_config_sha256": ds["dataset_config_sha256"], "split_json": split_json,
            "split_json_sha256": _file_digest(os.path.join(REPO, split_json)),
            "load": {"weights_source": "ema", "canonicalize_trained": None, "canonicalize_applied": False,
                     "canonicalize_mismatch": False, "img_w": None, "model_config_sha256_file": digest,
                     "model_config_sha256_embedded": digest}}
    for key, value in want.items():
        _eq(out, f"probe provenance.{key}", prov.get(key), value)
    if bf_receipts is not None:
        out += _probe_length_reasons(prov, sha, bf_receipts, label="B-F")
    if not isinstance(prov.get("model_config"), str) or os.path.realpath(os.path.join(REPO, prov["model_config"])) \
            != os.path.realpath(os.path.join(REPO, bf["model_config"])):
        out.append(f"probe provenance.model_config {prov.get('model_config')!r} is not {bf['model_config']}")
    _eq(out, "probe offsets (the quarter turns)", rep.get("offsets"), {f"d{d}": [d] * P["n_items"] for d in P["offsets"]})
    items = rep.get("items")
    if not (isinstance(items, list) and len(items) == P["n_items"] and all(
            isinstance(it, dict) and it.get("position") == i and isinstance(it.get("target_id"), str)
            and it["target_id"] and _context_ok(it.get("context_ids"), P["K"]) for i, it in enumerate(items))):
        out.append(f"probe items: not {P['n_items']} ordered positions with a target and K={P['K']} context ids")
    else:
        _eq(out, "probe stream_input_hash (recomputed from the items)", rep.get("stream_input_hash"),
            eval_FLAC.canonical_stream_hash([[it["position"], it["target_id"], it["context_ids"], P["img_w"]]
                                             for it in items]))
    return out + _bf_number_reasons(rep, P) + _bf_launch_reasons(prov.get("launch"), campaign_digest, base_dir)


def _bf_launch_reasons(launch, campaign_digest, base_dir):
    import hashlib
    out = []
    if not isinstance(launch, dict) or list(launch) != ["attention_backend", "overlay_sha256", "launcher_sha256",
                                                        "campaign_digest", "log_path", "log_sha256"]:
        return [f"probe provenance.launch {launch!r:.120} is missing or malformed (finalization not run?)"]
    for key, want in (("attention_backend", NO_FLASH["backend"]), ("overlay_sha256", NO_FLASH["overlay_sha256"]),
                      ("launcher_sha256", NO_FLASH["launcher_sha256"]), ("campaign_digest", campaign_digest)):
        _eq(out, f"probe provenance.launch.{key}", launch[key], want)
    path = os.path.join(base_dir, launch["log_path"]) if isinstance(launch["log_path"], str) else None
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except (OSError, TypeError) as err:
        return out + [f"probe provenance.launch.log_path {launch['log_path']!r} is unreadable ({err})"]
    _eq(out, "probe provenance.launch.log_sha256 (the log re-hashed)", launch["log_sha256"], hashlib.sha256(raw).hexdigest())
    return out + [f"probe log: {x}" for x in backend_log_reasons(raw.decode(errors="replace"), PROBE_CONTRACT_BF["target"])]


def _read_probe(path):
    """``(raw bytes, report, reasons)``: the probe JSON and its sha256 sidecar (as write_report wrote them)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            raw = f.read()
        with open(path + ".sha256") as f:
            side = f.read()
        rep = json.loads(raw)
    except (OSError, ValueError) as err:
        return None, None, [f"B-F probe {path} unreadable ({err})"]
    ok = side == f"{hashlib.sha256(raw).hexdigest()}  {os.path.basename(path)}\n" and isinstance(rep, dict)
    return raw, rep, [] if ok else [f"the B-F probe's sha256 sidecar does not match {path} (or it is no object)"]


def load_control_bf(path, *, pin, campaign_digest, manifest, bf_receipts=None, base_dir=REPO):
    """``(control_p99 of the eval protocol, reasons, eps)`` of a valid, registered B-F control of this
    campaign (:func:`probe_bf_reasons`); eps_cond is reported without a verdict."""
    _, rep, reasons = _read_probe(path)
    if rep is not None and not reasons:
        reasons = probe_bf_reasons(rep, pin=pin, campaign_digest=campaign_digest, manifest=manifest,
                                   bf_receipts=bf_receipts, base_dir=base_dir)
    if reasons:
        return None, reasons, None
    return ({k: float(v) for k, v in rep["control_p99"][CONTROL_PROTOCOL].items()}, [],
            {"value": rep["eps_cond"]["value"], "argmax": rep["eps_cond"]["argmax"], "verdict": None})


def finalize_bf_probe(path, log_path, *, campaign_digest, base_dir=REPO):
    """r3 §12 B3, the NON-logged step after the probe process exited and its log closed: verify the probe's own
    JSON + sidecar and the log's backend evidence, then write ``provenance.launch`` (backend, overlay +
    launcher sha256, the campaign digest, the log path + sha256), the JSON (atomically) and a fresh sidecar.
    Idempotent (Codex C3d HIGH 3, a supervisor restart after the finalization): an already-finalized JSON is
    accepted UNCHANGED only when it is the finalizer's own bytes, sealed by its sidecar, and its launch record is
    exactly the one this call would write (the CURRENT log sha256). ``ValueError`` (nothing written) on any refusal."""
    import hashlib
    raw_json, rep, reasons = _read_probe(path)
    if reasons:
        raise ValueError("; ".join(reasons))
    prov = rep.get("provenance")
    if rep.get("arm") != "BF" or not isinstance(prov, dict):
        raise ValueError(f"{path}: not a B-F probe awaiting finalization (another arm, or no provenance)")
    if not eval_FLAC._is_sha256_hex(campaign_digest):
        raise ValueError(f"campaign digest {campaign_digest!r} is not a sha256 digest")
    with open(os.path.join(base_dir, log_path), "rb") as f:
        raw = f.read()
    bad = backend_log_reasons(raw.decode(errors="replace"), PROBE_CONTRACT_BF["target"])
    if bad:
        raise ValueError(f"the probe log {log_path} lacks the backend evidence: " + "; ".join(bad))
    launch = {"attention_backend": NO_FLASH["backend"], "overlay_sha256": NO_FLASH["overlay_sha256"],
              "launcher_sha256": NO_FLASH["launcher_sha256"], "campaign_digest": campaign_digest,
              "log_path": log_path, "log_sha256": hashlib.sha256(raw).hexdigest()}
    if "launch" in prov:
        if not (isinstance(prov["launch"], dict) and list(prov["launch"].items()) == list(launch.items())
                and raw_json == (json.dumps(rep, indent=1) + "\n").encode()):
            raise ValueError(f"{path}: already finalized with another launch record (log, its sha256, the campaign "
                             "digest, the overlay or the launcher) or in other bytes")
        return path
    prov["launch"] = launch
    text = json.dumps(rep, indent=1) + "\n"
    for target, body in ((path, text), (path + ".sha256",
                                        f"{hashlib.sha256(text.encode()).hexdigest()}  {os.path.basename(path)}\n")):
        tmp = f"{target}.tmp-{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(body)
        os.replace(tmp, target)
    return path


def _waveforms(entry):
    import numpy as np
    return np.load(os.path.join(entry["dump_dir"], "waveforms.npy"), mmap_mode="r", allow_pickle=False)


def _paired(a, b, chunk=512):
    """Per-item relL2 / relLinf of ``a`` against the reference ``b``, chunked (memory-bounded)."""
    import numpy as np
    from src.tools import depth_canon_stats as stats
    l2, linf = [], []
    for start in range(0, b.shape[0], chunk):
        l2.append(stats.rel_l2(a[start:start + chunk], b[start:start + chunk]))
        linf.append(stats.rel_linf(a[start:start + chunk], b[start:start + chunk]))
    return np.concatenate(l2), np.concatenate(linf)


def tier2(pairs, cells, control):
    """Tier 2 -- practical waveform invariance of CANON from the scored dumps (rotated vs its
    theta = 0 pair, identical input_hash). Verdict = every pair's matched rule (its offset's
    control p99) AND the pooled rule (all pairs' items vs the pooled control p99) AND the 1e-2
    cap; requiring both readings of "control p99" is the fail-closed resolution. The alarm and
    the T-dup determinism requirement are reported separately (plan v9 §1)."""
    import numpy as np
    from src.tools import depth_canon_stats as stats
    if control is None:
        return {"pass": None, "note": "no registered composition control: Tier 2 not computable"}
    out, pooled, alarm = [], [], {}
    for p in [p for p in pairs if p["arm"] == "CANON" and p["kind"] in ("C4", "45", "R")]:
        key = control_key(p["deg"])
        e, ref = cells[p["cell"]], cells.get(p["ref"]) if p["ref"] else None
        if not (p["input_hash_equal"] and e["dump"] and ref and ref["dump"]):
            out.append({**p, "control_key": key, "computable": False, "pass_matched": False,
                        "pass_cap": False})
            continue
        l2, linf = _paired(_waveforms(e), _waveforms(ref))
        s2, sinf = stats.paired_summary(l2), stats.paired_summary(linf)
        pooled.append(l2)
        if s2["max"] > TIER2_ALARM:
            targets = [t[1] for t in ref["stream"]["input_tuples"]]
            alarm[p["cell"]] = [targets[i] for i in np.flatnonzero(l2 > TIER2_ALARM)]
        out.append({**p, "control_key": key, "control_p99": control[key], "computable": True,
                    "relL2": s2, "relLinf": sinf,
                    "pass_matched": s2["p99"] <= TIER2_FACTOR * control[key], "pass_cap": s2["p99"] <= TIER2_CAP})
    pool = stats.paired_summary(np.concatenate(pooled)) if pooled else None
    pool_ok = bool(pool and pool["p99"] <= TIER2_FACTOR * control["pooled"] and pool["p99"] <= TIER2_CAP)
    dup = []
    for p in [p for p in pairs if p["arm"] == "CANON" and p["kind"] == "dup"]:
        e, ref = cells[p["cell"]], cells.get(p["ref"]) if p["ref"] else None
        ok = bool(p["input_hash_equal"] and e["dump"] and ref and ref["dump"])
        mx = float(_paired(_waveforms(e), _waveforms(ref))[0].max()) if ok else None
        sha = lambda x: json.load(open(os.path.join(x["dump_dir"], "manifest.json")))["sha256"]  # noqa: E731
        same = bool(ok and sha(e) == sha(ref))
        dup.append({**p, "relL2_max": mx, "dump_sha_equal": same, "pass": bool(ok and mx == 0 and same)})
    return {"pass": bool(out) and all(p["computable"] and p["pass_matched"] and p["pass_cap"] for p in out)
            and pool_ok, "pairs": out,
            "pooled": {"relL2": pool, "control_p99": control["pooled"], "pass": pool_ok},
            "determinism": {"pass": bool(dup) and all(d["pass"] for d in dup), "pairs": dup},
            "alarm": {"suspended": bool(alarm), "items": alarm,
                      "note": ("invariance conclusion suspended pending item triage" if alarm else "none")},
            "rule": f"p99(relL2) <= {TIER2_FACTOR:g} x control p99 (matched offset AND pooled) and <= "
                    f"{TIER2_CAP:g}; alarm max > {TIER2_ALARM:g}; T-dup max relL2 == 0"}


# Tier 3 (plan v9 §1): Delta_C4 of CANON <= 10 % of P1-pin's on every metric, and |Delta_R| of
# CANON <= one eval-seed sd of its m_Z; item-weighted (the table's estimand), per-scene reported.
SEEDS = (42, 43, 44, 45, 46)
TIER3_FRACTION = 0.10
AGGS = ("item", "scene")   # flat `metrics` (item-weighted) / mean over the ten by_scene families


def _value(entry, key, agg):
    rec = entry["record"]
    if agg == "item":
        return float(rec["metrics"][key])
    return sum(float(rec["by_scene"][s][key]) for s in SCENE_KEYS) / len(SCENE_KEYS)


def _index(cells):
    """``(arm, kind F|R|S, K, seed, step, deg) -> entry`` over the valid table / rotation cells."""
    idx = {}
    for e in cells.values():
        r = e["row"]
        if e["valid"] and r["block"] not in ("D-i", "D-ii", "T-dup"):
            kind = "S" if r["split"] == "seen" else ("R" if r["rotate_mode"] == "random" else "F")
            idx[(r["arm"], kind, r["K"], r["seed"], r["step"], r["rotate_deg"])] = e
    return idx


def _same_items(a, b):
    return a is not None and b is not None and a["stream"]["input_hash"] == b["stream"]["input_hash"]


def _stats(values):
    import statistics as st
    return {"n": len(values), "mean": st.mean(values) if values else None,
            "sd": st.stdev(values) if len(values) > 1 else None, "values": values}


def _arm_tier3(idx, arm, k, agg, key):
    v = lambda e: _value(e, key, agg)  # noqa: E731
    z = {s: idx.get((arm, "F", k, s, 40000, 0)) for s in SEEDS}
    c4 = [idx.get((arm, "F", k, 42, 40000, d)) for d in (0, 90, 180, 270)]
    p45, rr = idx.get((arm, "F", k, 42, 40000, 45)), {s: idx.get((arm, "R", k, s, 40000, None)) for s in SEEDS}
    zs = _stats([v(e) for e in z.values() if e])
    ok_c4 = all(_same_items(e, c4[0]) for e in c4)
    d = [v(rr[s]) - v(z[s]) for s in SEEDS if _same_items(rr[s], z[s])]
    return {"n_z": zs["n"], "mean_z": zs["mean"], "sd_z": zs["sd"],
            "c4_values": [v(e) for e in c4] if ok_c4 else None,
            "delta_c4": (max(v(e) for e in c4) - min(v(e) for e in c4)) if ok_c4 else None,
            "probe45": v(p45) - v(c4[0]) if _same_items(p45, c4[0]) else None,
            "n_r_pairs": len(d), "delta_r": (sum(d) / len(d)) if len(d) == len(SEEDS) else None,
            "sd_r_pairs": _stats(d)["sd"]}


def tier3(cells):
    """Tier 3 per arm, K, aggregation and metric, plus the CANON-vs-P1 PASS checks
    (``pass`` None when an input is missing -- never a silent PASS)."""
    idx = _index(cells)
    arms = {a: {k: {agg: {key: _arm_tier3(idx, a, k, agg, key) for key in TIER3_KEYS} for agg in AGGS}
                for k in (1, 8)} for a in ("CANON", "P1", "YAW")}
    checks = []
    for k in (1, 8):
        for key in TIER3_KEYS:
            c, p = arms["CANON"][k]["item"][key], arms["P1"][k]["item"][key]
            known = None not in (c["delta_c4"], p["delta_c4"], c["delta_r"], c["sd_z"])
            checks.append({"K": k, "metric": key, "computable": known,
                           "pass_c4": known and c["delta_c4"] <= TIER3_FRACTION * p["delta_c4"],
                           "pass_r": known and abs(c["delta_r"]) <= c["sd_z"]})
    verdict = None if not all(c["computable"] for c in checks) else all(
        c["pass_c4"] and c["pass_r"] for c in checks)
    return {"arms": arms, "checks": checks, "pass": verdict,
            "rule": f"Delta_C4(CANON) <= {TIER3_FRACTION:g} x Delta_C4(P1-pin) and |Delta_R(CANON)| <= "
                    "sd(m_Z CANON), every metric, both K, item-weighted"}


def table_block(cells):
    """theta = 0 rows, mean +- sample sd over the five eval seeds: unseen (CANON T, P1-pin T,
    YAW-pin T) and ``seen`` (CANON S, YAW-pin S); item-weighted and per-scene."""
    idx = _index(cells)
    def block(kind, arm):
        return {k: {agg: {key: _stats([_value(e, key, agg) for e in
                                       (idx.get((arm, kind, k, s, 40000, 0)) for s in SEEDS) if e])
                          for key in TIER3_KEYS} for agg in AGGS} for k in (1, 8)}
    out = {arm: block("F", arm) for arm in ("CANON", "P1", "YAW")}
    out["seen"] = {arm: block("S", arm) for arm in ("CANON", "YAW")
                   if any(key[0] == arm and key[1] == "S" for key in idx)}
    return out


def traj_grid(cells, t3):
    """The D5c trajectory: CANON checkpoints 2.5k..40k x theta {0,90,180,270} x K, seed 42,
    item-weighted (40k = the T / C4 cells). Descriptive, single seed; each step's Delta_C4 is
    set against 10 % of P1-pin's Delta_C4 @40k (the only P1 reference measured at this pin)."""
    idx, grid = _index(cells), []
    for step in range(2500, 40001, 2500):
        for k in (1, 8):
            ent = {d: idx.get(("CANON", "F", k, 42, step, d)) for d in (0, 90, 180, 270)}
            ok = all(_same_items(e, ent[0]) for e in ent.values())
            delta = {key: (max(_value(e, key, "item") for e in ent.values())
                           - min(_value(e, key, "item") for e in ent.values())) for key in TIER3_KEYS} if ok else None
            p1 = {key: t3["arms"]["P1"][k]["item"][key]["delta_c4"] for key in TIER3_KEYS}
            grid.append({"step": step, "K": k, "cells": {str(d): e["row"]["cell"] if e else None for d, e in ent.items()},
                         "values": {str(d): {key: _value(e, key, "item") for key in TIER3_KEYS} if e else None
                                    for d, e in ent.items()},
                         "delta_c4": delta,
                         "within_10pct_of_p1_40k": {key: (delta[key] <= TIER3_FRACTION * p1[key])
                                                    if delta and p1[key] is not None else None
                                                    for key in TIER3_KEYS}})
    return {"grid": grid, "note": "descriptive, single seed 42, item-weighted; 40k = the T / C4 cells; "
                                  "reference = 10 % of P1-pin Delta_C4 @40k"}


BF_LABEL = "fa scratch B-F @40k (equivariant, 59% budget)"
MODEL_COMPARISON = os.path.join(REPO, "worklog", "worklog_yixun", "model_comparison.md")


def bf_historical(path=MODEL_COMPARISON):
    """B-F's HISTORICAL theta = 0 row, quoted verbatim (plan §5 ledger: not re-measured)."""
    rows = {}
    with open(path) as f:
        for line in f:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 10 and cells[0] == BF_LABEL:
                rows[int(cells[2])] = {"protocol": cells[1], "n": int(cells[3]),
                                       "values": dict(zip(("T60", "C50", "EDT", "R@1", "R@5", "R@10"), cells[4:10]))}
    return {"label": BF_LABEL, "rows": rows,
            "provenance": "quoted from model_comparison.md (exp_07 B-F @40k, fa_invariant eval, legacy "
                          "per-angle loop, its own historical evaluator pin; raw JSONs "
                          "outputs_FLAC/exp07_BF/**/*exp10_BF40_K{1,8}_s4[2-6]*.json) -- not re-measured "
                          "at the exp_25 pin, no rotated or random-yaw cells (plan §5 ledger)"}


_GATE = None


def audit_gate():
    """``depth_canon_audit_gate.py`` of this experiment folder, loaded once: the ONE tie-audit
    predicate (Codex rB B1), the same module the launcher applies."""
    global _GATE
    if _GATE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("depth_canon_audit_gate",
                                                      os.path.join(EXPDIR, "depth_canon_audit_gate.py"))
        _GATE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GATE)
    return _GATE


def load_tie_audit(path):
    """``(report, summary, reasons)`` -- the rung-3 tie audit only when the shared gate
    (``validate_tie_audit``: sidecar, exact schema, the audit's own census, current hashes,
    re-hashed map bytes) accepts it; otherwise ``(None, None, reasons)``."""
    import hashlib
    ok, problems, summary = audit_gate().validate_tie_audit(path, REPO)
    if not ok:
        return None, None, problems
    with open(path, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != summary["audit_sha256"]:
        return None, None, [f"tie audit {path} changed while it was being validated"]
    return json.loads(raw), summary, []


def tie_summary(cells, audit, summary=None):
    """Tie classes per canonicalized cell (records) and per split (the VALIDATED audit,
    item-weighted); every cell's counts must equal its split's audit counts (a per-map
    property), else the cell is listed in ``mismatches`` (a problem in full mode)."""
    per_cell = {n: e["record"]["canonicalize_tie_counts"] for n, e in cells.items()
                if e["valid"] and e["row"]["canonicalize"]}
    per_split = {s: audit["per_split"][s] for s in ("train", "unseen", "seen")} if audit else {}
    mismatches = []
    if audit:
        for n, counts in per_cell.items():
            items = per_split[cells[n]["row"]["split"]]["items"]
            if counts != {c: items[c] for c in TIE_CLASSES}:
                mismatches.append(n)
    shared = (len(set(summary["split_maps"]["unseen"]) & set(summary["split_maps"]["seen"]))
              if audit and summary else None)
    return {"per_cell": per_cell, "per_split": per_split, "mismatches": mismatches, "shared_maps": shared}


REGISTERED_OVERLAP = {"items": 1093, "maps": 112}   # the r3b data finding (plan v10 §5)
_DEFAULT_SPLITS = (("unseen", "data/AR/unseen_eval.json"), ("seen", "data/AR/seen_eval.json"))


def split_overlap(summary=None):
    """The seen / unseen overlap (r3b review; rB-F1 NIT b): the shared ITEMS recomputed from the
    split JSONs the validated audit names (the registered defaults without one), the shared
    depth MAPS derived from the validated audit's per-split map sets (None without one) --
    both compared with the registered overlap; a difference is flagged in ``mismatch``."""
    files = ({i["split"]: i["json_abs"] for i in summary["inputs"]} if summary
             else {n: os.path.join(REPO, rel) for n, rel in _DEFAULT_SPLITS})
    items = {}
    for name in ("unseen", "seen"):
        with open(files[name]) as f:
            data = json.load(f)
        items[name] = {(room, rid, w) for room, ids in data.items() for rid, ws in ids.items() for w in ws}
    shared = len(items["unseen"] & items["seen"])
    u, sn = len(items["unseen"]), len(items["seen"])
    maps = len(set(summary["split_maps"]["unseen"]) & set(summary["split_maps"]["seen"])) if summary else None
    reg = dict(REGISTERED_OVERLAP)
    mismatch = ([f"overlap: {shared} shared items != the registered {reg['items']}"] if shared != reg["items"] else []) \
        + ([f"overlap: {maps} shared depth maps != the registered {reg['maps']}"]
           if maps is not None and maps != reg["maps"] else [])
    maps_txt = (f"{maps} depth maps (derived from the validated rung-3 tie audit)" if maps is not None
                else "an unknown number of depth maps (not derivable: no validated tie audit)")
    return {"unseen_items": u, "seen_items": sn, "shared_items": shared, "shared_maps": maps, "registered": reg,
            "mismatch": mismatch,
            "disclosure": (f"DISCLOSURE: the seen and unseen AR eval splits share {shared:,} exact item paths "
                           f"({100 * shared / u:.1f} % of unseen, {100 * shared / sn:.1f} % of seen) and {maps_txt}: "
                           "seen and unseen results are correlated and must not be presented as independent "
                           "estimates." + (f" MISMATCH with the registered overlap: {'; '.join(mismatch)}."
                                           if mismatch else ""))}


ARM_LABEL = {"CANON": "CANON (deepest-column canonicalization) @40k",
             "P1": "P1 vanilla @40k (P1-pin, re-measured at the exp_25 pin)",
             "YAW": "Yaw-Aug @40k (YAW-pin, re-measured at the exp_25 pin)"}
SHORT = {"RIR_to_GT_RIR_R@1": "R@1", "RIR_to_GT_RIR_R@5": "R@5", "RIR_to_GT_RIR_R@10": "R@10"}


def build_results(cells, *, control, control_reasons, eps, audit, audit_reasons, pin, blocks, partial,
                  audit_summary=None):
    """Every tier, table and disclosure; raises ``ValueError`` (the refusal) when a required
    cell, the registered control or the tie audit is missing/invalid, or a cell's tie counts
    contradict the validated audit -- unless ``partial``."""
    missing, invalid = completeness(cells)
    ties = tie_summary(cells, audit, audit_summary)       # BEFORE the refusal (Codex rB B1)
    problems = ([f"{len(missing)} required cell(s) missing"] if missing else []) + (
        [f"{len(invalid)} cell(s) failed verification"] if invalid else []) + (
        [f"composition control: {'; '.join(control_reasons)}"] if control is None else []) + (
        [f"tie audit: {'; '.join(audit_reasons)}"] if audit is None else []) + (
        [f"tie audit: {len(ties['mismatches'])} canonicalized cell(s) carry tie counts that differ from the "
         f"validated audit's split counts ({', '.join(ties['mismatches'][:3])})"] if ties["mismatches"] else [])
    overlap = split_overlap(audit_summary)
    problems += overlap["mismatch"]
    repl = context_replacement(cells)
    problems += [f"{k}: repeated context ids on different targets in different cells (the registered rule: "
                 "the same targets in every cell of a split and K)" for k, v in repl.items() if not v["consistent"]]
    digests = sorted({e["receipt"]["campaign_digest"] for e in cells.values() if e["valid"]})
    if len(digests) > 1:   # rB-F1 B3: one campaign = one evaluated tree
        problems.append(f"the cells span {len(digests)} campaign digests ({', '.join(d[:12] for d in digests)})")
    if problems and not partial:
        raise ValueError("RESULTS REFUSED (pass --partial for a labelled partial table): " + "; ".join(problems)
                         + "".join(f"\n  missing {n}" for n in missing)
                         + "".join(f"\n  invalid {n}: {cells[n]['reasons'][:3]}" for n in invalid))
    pairs = make_pairs(cells)
    t3 = tier3(cells)
    diag = {cells[n]["row"]["block"] + f" K={cells[n]['row']['K']}": {
        "cell": n, "metrics": {k: cells[n]["record"]["metrics"][k] for k in TIER3_KEYS}}
        for n, e in cells.items() if e["valid"] and e["row"]["block"] in ("D-i", "D-ii")}
    return {"schema_version": 1, "pin": pin, "blocks": blocks, "partial": bool(problems), "problems": problems,
            "missing": missing, "invalid": {n: cells[n]["reasons"] for n in invalid},
            "pairing": {"note": PAIRING_NOTE, "context_replacement": repl},
            "tier1": tier1(pairs, cells), "tier2": tier2(pairs, cells, control), "tier3": t3,
            "eps_cond": eps or {"value": None, "threshold": PROBE_CONTRACT["eps_threshold"],
                                "verdict": "NOT COMPUTABLE (no valid registered composition control)"},
            "table": table_block(cells), "bf_historical": bf_historical(), "traj": traj_grid(cells, t3),
            "ties": ties, "overlap": overlap, "diagnostics": diag}


def _fmt(st, key):
    if not st or st.get("mean") is None:
        return "—"
    nd = 4 if key == "C50" else 3
    return f"{st['mean']:.{nd}f}" + (f" ± {st['sd']:.{nd}f}" if st.get("sd") is not None else "")


def _num(v, nd=4):
    return "—" if v is None else f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def render_md(res):
    part = res["partial"]
    h = lambda t: f"## {'PARTIAL — ' if part else ''}{t}"  # noqa: E731
    keys = TIER3_KEYS
    head = "| row | K | " + " | ".join(SHORT.get(k, k) for k in keys) + " |"
    sep = "|---|---|" + "---|" * len(keys)
    out = [f"# exp_25 depth_canon — results ({'PARTIAL' if part else 'complete'})", "",
           f"Evaluator pin `{res['pin']}`; blocks {', '.join(res['blocks'])}. Generated by "
           "`depth_canon_collect.py --collect` from the raw cell artifacts (never hand-edited).", "",
           res["overlap"]["disclosure"], ""]
    if part:
        out += [f"**PARTIAL:** {'; '.join(res['problems'])}. Missing: " + (", ".join(res["missing"]) or "none")
                + ". Invalid: " + (", ".join(res["invalid"]) or "none") + ".", ""]
    t1, t2, t3 = res["tier1"], res["tier2"], res["tier3"]
    out += [f"Pairing: {res['pairing']['note']}. Context draws with replacement (repeated ids, same targets in "
            f"every cell of a split and K): {res['pairing']['context_replacement']}.", "",
            h("Tier 1 — canonical depth identity (exact)"), "",
            f"PASS: {t1['pass']} over {t1['n_pairs']} rotated/duplicate CANON pairs; TRAJ pairs "
            f"{t1['trajectory']['n_pass']}/{t1['trajectory']['n_pairs']} equal (descriptive); distinct unseen "
            f"canonical_depth_hash values: {t1['distinct_canonical_depth_hashes_unseen']}.", "",
            h("Tier 2 — practical waveform invariance (scored dumps)"), "",
            f"PASS: {t2['pass']} — rule: {t2.get('rule', t2.get('note'))}.", "",
            f"eps_cond (fp32 conditioning, max relL2) = {_num(res['eps_cond']['value'])}; registered threshold "
            f"{res['eps_cond']['threshold']:.0e}: {res['eps_cond']['verdict']}.", ""]
    if t2.get("pairs") is not None:
        out += ["| cell | control | ctrl p99 | relL2 p99 | relL2 max | relLinf max | matched | cap |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {p['cell']} | {p['control_key']} | {_num(p.get('control_p99'))} | "
                f"{_num((p.get('relL2') or {}).get('p99'))} | {_num((p.get('relL2') or {}).get('max'))} | "
                f"{_num((p.get('relLinf') or {}).get('max'))} | {p['pass_matched']} | {p['pass_cap']} |"
                for p in t2["pairs"]]
        pool = t2["pooled"]
        out += ["", f"pooled: relL2 p99 {_num((pool['relL2'] or {}).get('p99'))} vs 3 x {_num(pool['control_p99'])}"
                f" -> {pool['pass']}; T-dup determinism: {t2['determinism']['pass']}; alarm: {t2['alarm']['note']}"
                + (f" {t2['alarm']['items']}" if t2["alarm"]["suspended"] else ""), ""]
    out += [h("Tier 3 — metric spreads (item-weighted verdict; per-scene in the JSON)"), "",
            f"PASS: {t3['pass']} — {t3['rule']}.", "",
            "| arm | K | metric | Delta_C4 | 45-deg probe | Delta_R | sd(m_Z) |", "|---|---|---|---|---|---|---|"]
    for arm in ("CANON", "P1", "YAW"):
        for k in (1, 8):
            for key in keys:
                x = t3["arms"][arm][k]["item"][key]
                out.append(f"| {arm} | {k} | {SHORT.get(key, key)} | {_num(x['delta_c4'])} | {_num(x['probe45'])} | "
                           f"{_num(x['delta_r'])} | {_num(x['sd_z'])} |")
    for agg, title in (("item", "item-weighted (the table estimand)"), ("scene", "per-scene mean (recorded)")):
        out += ["", h(f"Table block — unseen, theta = 0, mean ± sd over seeds 42–46, {title}"), "", head, sep]
        for arm in ("CANON", "P1", "YAW"):
            for k in (1, 8):
                out.append(f"| {ARM_LABEL[arm]} | {k} | " + " | ".join(
                    _fmt(res["table"][arm][k][agg][key], key) for key in keys) + " |")
        if agg == "item":
            bf = res["bf_historical"]
            out += [f"| {bf['label']} — B-F HISTORICAL, quoted | {k} | " + " | ".join(
                bf["rows"].get(k, {}).get("values", {}).get(SHORT.get(key, key), "—") for key in keys) + " |"
                for k in (1, 8)] + ["", f"B-F: {bf['provenance']}."]
    out += ["", h("Seen split (S block) — correlated with unseen, see the disclosure"), "", head, sep]
    out += [f"| {ARM_LABEL[a]} | {k} | " + " | ".join(_fmt(res["table"]["seen"][a][k]["item"][key], key)
                                                       for key in keys) + " |"
            for a in res["table"]["seen"] for k in (1, 8)]
    out += ["", h("Diagnostics D-i (CANON ckpt, no canonicalization) / D-ii (P1 ckpt + canonicalization)"), ""]
    out += [f"- {name}: `{d['cell']}` " + ", ".join(f"{SHORT.get(k, k)} {v:.4g}" for k, v in d["metrics"].items())
            for name, d in sorted(res["diagnostics"].items())]
    out += ["", h("TRAJ — Delta_C4 per checkpoint (seed 42, item-weighted, descriptive)"), "",
            "| step | K | " + " | ".join(SHORT.get(k, k) for k in keys) + " | <= 10 % of P1@40k (all) |",
            "|---|---|" + "---|" * (len(keys) + 1)]
    for g in res["traj"]["grid"]:
        d, w = g["delta_c4"] or {}, g["within_10pct_of_p1_40k"]
        out.append(f"| {g['step']} | {g['K']} | " + " | ".join(_num(d.get(k)) for k in keys)
                   + f" | {all(w.values()) if None not in w.values() else '—'} |")
    ties = res["ties"]
    out += ["", h("tie classes (per split: tie audit; per cell: records)"), "",
            f"per split (items): {ties['per_split']}; cells whose counts differ from their split: "
            f"{ties['mismatches'] or 'none'}; shared unseen/seen maps: {ties['shared_maps']}.", ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# exp_35 collect mode, part 1 (plan_orbitrir_yaw_evidence.md r2 §3c E1, §4 E2, §5 items 1-5): the hash-pinned
# external exp_25 results, the hash-pinned stream registry, the CTL hard stop and E2
# --------------------------------------------------------------------------- #
EXP25_RESULTS_SHA256 = "ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0"
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
EXP25_MISSING = tuple(f"exp25_{a}_S_S40000_K{k}_s{s}" for a in ("CANON", "YAW") for k in (1, 8) for s in SEEDS)
EXP35_DIR = os.path.join(REPO, "worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude")
EXP35_REGISTRY = os.path.join(EXP35_DIR, "exp35_exp25_stream_registry.json")
EXP35_REGISTRY_SHA256 = "9ec31f32a9d89cf9ffa90938d7f1198658af2eafa3b53675605d9fb925365a24"
A6_CSV = os.path.join(REPO, "analysis", "yaw_robustness", "generated", "yaw_metrics_raw.csv")
A6_CSV_SHA256 = "0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da"
E2_TOLERANCE = {"T60": 0.001, "C50": 0.0005, "EDT": 0.005, "FD": 0.0005, "RIR_to_GT_RIR_R@1": 0.2,
                "RIR_to_GT_RIR_R@5": 0.2, "RIR_to_GT_RIR_R@10": 0.2}   # §4: T60 %, C50 dB, EDT ms, FD, R@k pp
CTL_REFERENCE = {   # §3c: the force-added exp_25 P1-pin T s42 cells (receipt, waveform, stream sidecar)
    1: {"receipt_sha256": "fa1ea729708e5dc139336673eb2c7b87cbbad224d2ec3315109d9a2e67443714",
        "waveform_sha256": "8cb47c945d55d3787f76bf0cf8e78e445f8db1387fefca827316401da8d9e8f8",
        "stream_sha256": "31aa02dcd5dcb2f6d80247f13e8a4e76ae28bc050f7f2c8f353e2b0808e0793d"},
    8: {"receipt_sha256": "54490e2ab131d323ec3ea9471273d87834f0b5344cd6d806ee74b08d8c376bec",
        "waveform_sha256": "1aace782f22267bb4a39712ee459cf3b7ec9b891e97695455e63862d74eb0fd3",
        "stream_sha256": "8d8018cabfd803be559b4224ef7c8833fe0af7b96542cbce65cd0a3e1d1d4af6"}}
CTL_LOADED = ("c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c", 723927531)
CTL_DUMP_SHAPE = [6337, 10240]


def _pinned_json(path, sha256):
    """``(doc, reasons)`` of a hash-pinned JSON file (None when unreadable or not the pinned bytes)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as err:
        return None, [f"{path} unreadable ({err})"]
    if hashlib.sha256(raw).hexdigest() != sha256:
        return None, [f"{path} hashes to {hashlib.sha256(raw).hexdigest()}, not the pinned sha256 {sha256}"]
    return json.loads(raw), []


def exp25_reference_reasons(doc):
    """§5 items 1-2: the exp_25 results as registered -- schema, pin, the known partial state -- and every
    reused field finite: Tier 3 of CANON / P1 / YAW per K, item-weighted metric, and the theta = 0 table."""
    out = []
    for key, want in (("schema_version", 1), ("pin", EXP25_PIN), ("partial", True), ("missing", list(EXP25_MISSING)),
                      ("invalid", {})):
        _eq(out, f"exp_25 results {key}", doc.get(key), want)
    for arm in ("CANON", "P1", "YAW"):
        for k in ("1", "8"):
            for key in TIER3_KEYS:
                x = (((doc.get("tier3") or {}).get("arms") or {}).get(arm) or {}).get(k, {}).get("item", {}).get(key, {})
                t = (((doc.get("table") or {}).get(arm) or {}).get(k) or {}).get("item", {}).get(key, {})
                out += [f"exp_25 tier3 {arm} K{k} {key} {f} = {x.get(f)!r} != 5" for f in ("n_z", "n_r_pairs")
                        if x.get(f) != 5]
                out += [f"exp_25 tier3 {arm} K{k} {key} {f} = {x.get(f)!r} is not finite"
                        for f in ("mean_z", "sd_z", "delta_c4", "probe45", "delta_r") if not _finite(x.get(f))]
                if t.get("n") != 5 or not (_finite(t.get("mean")) and _finite(t.get("sd"))):
                    out.append(f"exp_25 table {arm} K{k} {key} {t!r:.80} is not a finite 5-seed mean +- sd")
    return out


def load_exp25_reference(path=None):
    """The external exp_25 results (§5 items 1-3): ``(doc, reasons)``; doc is None unless valid."""
    doc, reasons = _pinned_json(path or os.path.join(EXPDIR, "depth_canon_results.json"), EXP25_RESULTS_SHA256)
    reasons = reasons or exp25_reference_reasons(doc)
    return (None if reasons else doc), reasons


def load_registry(path=EXP35_REGISTRY):
    """The frozen stream registry (§5 item 5): ``(doc, reasons)``."""
    return _pinned_json(path, EXP35_REGISTRY_SHA256)


def registry_reasons(row, stream, registry):
    """An exp_35 cell's stream reproduces exp_25's: the input_hash of its K / seed, and the assignment of its
    seed (random) or of its angle (fixed)."""
    out, k, s = [], str(row["K"]), str(row["seed"])
    _eq(out, f"stream input_hash vs the exp_25 registry (K{k} s{s})", stream.get("input_hash"),
        registry["input_hash"].get(k, {}).get(s))
    want = (registry["assignment_hash_random"].get(s) if row["rotate_mode"] == "random"
            else registry["assignment_hash_fixed"].get(str(row["rotate_deg"])))
    _eq(out, f"stream assignment_hash vs the exp_25 registry ({row['rotate_mode']} {row['rotate_deg'] or s})",
        stream.get("assignment_hash"), want)
    return out


def _verified_cell(name, *, manifest, pin, base_dir, farm_root=None, dump_root=None, canon_index=None):
    """``(row, paths, docs, reasons)`` of one cell verified OFFLINE (the receipt's dump evidence; no link check)."""
    rows = [c for c in manifest["cells"] if c["cell"] == name]
    if len(rows) != 1:
        return None, None, None, [f"{name} is not one registered cell"]
    farm, dump_root = farm_root or manifest["farm_root"], dump_root or manifest["dump_root"]
    target, _ = expected_checkpoint(rows[0], manifest, canon_index)
    p = cell_paths(rows[0], os.path.join(farm, rows[0]["arm"], os.path.basename(target)), dump_root)
    reasons = verify_cell(p.stem, manifest=manifest, farm_root=farm, dump_root=dump_root, pin=pin,
                          canon_index=canon_index, base_dir=base_dir, dump_check="receipt", check_link=False)
    if reasons:
        return rows[0], p, None, [f"{name} does not verify at {pin[:12]}: {x}" for x in reasons[:3]]
    docs = {}
    for key in ("metrics", "stream", "receipt"):
        with open(os.path.join(base_dir, getattr(p, key)), "rb") as f:
            docs[key + "_raw"] = f.read()
        docs[key] = json.loads(docs[key + "_raw"])
    return rows[0], p, docs, []


def exp35_ctl_check(k, *, pin, manifest, base_dir=REPO, farm_root=None, dump_root=None):
    """§3c E1, the hard stop: ``(status, reasons)`` -- "equal", "mismatch" (no B-F cell may run) or "refused"
    (undecidable: a cell does not verify, or the reference is not the registered one)."""
    import hashlib
    ref_name, new_name = f"exp25_P1_T_S40000_K{k}_s42", f"exp35_P1_T_S40000_K{k}_s42"
    pinned = CTL_REFERENCE[k]
    _, _, ref, why = _verified_cell(ref_name, manifest=manifest, pin=EXP25_PIN, base_dir=base_dir, farm_root=farm_root,
                                    dump_root=dump_root)
    _, _, new, why2 = _verified_cell(new_name, manifest=manifest, pin=pin, base_dir=base_dir, farm_root=farm_root,
                                     dump_root=dump_root)
    refused = why + why2
    if ref and hashlib.sha256(ref["receipt_raw"]).hexdigest() != pinned["receipt_sha256"]:
        refused.append(f"the reference receipt of {ref_name} is not the registered {pinned['receipt_sha256'][:12]}")
    if new and new["receipt"].get("attention_backend") != "sdpa":
        refused.append(f"{new_name}: receipt attention_backend {new['receipt'].get('attention_backend')!r} != sdpa")
    if refused:
        return "refused", refused
    out, rm, nm = [], ref["receipt"]["dump_evidence"]["manifest"], new["receipt"]["dump_evidence"]["manifest"]
    for label, a, b in (("waveform sha256", nm["sha256"], pinned["waveform_sha256"]),
                        ("COMPLETE marker", new["receipt"]["dump_evidence"]["marker"], pinned["waveform_sha256"] + "\n"),
                        ("dump n / dtype / shape", [nm["n"], nm["dtype"], nm["shape"]],
                         [CTL_DUMP_SHAPE[0], "float32", CTL_DUMP_SHAPE]),
                        ("dump input_hash", nm["input_hash"], rm["input_hash"]),
                        ("stream sidecar sha256", hashlib.sha256(new["stream_raw"]).hexdigest(), pinned["stream_sha256"]),
                        ("ckpt loaded", [new["metrics"].get(x) for x in LOAD_KEYS], list(CTL_LOADED))):
        _eq(out, f"CTL K{k} {label}", a, b)
    for key in ("metrics", "by_scene"):
        diff = eval_FLAC._canon_strict_diff(new["metrics"].get(key), ref["metrics"].get(key), key)
        if diff is not None:
            out.append(f"CTL K{k} record {key} differs from the exp_25 cell: {diff}")
    rest = lambda d: {x: v for x, v in d.items() if x not in ("metrics", "by_scene", "source_sha", "eval_name")}  # noqa
    _eq(out, f"CTL K{k} record (every other key but source_sha / eval_name)", rest(new["metrics"]), rest(ref["metrics"]))
    return ("equal" if not out else "mismatch"), out


def exp35_e2(*, pin, manifest, base_dir=REPO, csv_path=None, farm_root=None, dump_root=None):
    """§4 E2 (decides only the 45-degree source): ``exp35_BF_T_S40000_K{1,8}_s42`` vs the sha-pinned A6 CSV row
    (method ``FA B-F@40k``, angle 0, seed 42) -- PASS iff all 14 |differences| are within E2_TOLERANCE."""
    import csv
    import hashlib
    path = csv_path or A6_CSV
    res = {"status": "REFUSED", "pass": None, "csv": path, "csv_sha256": None, "rows": [], "rows45": [], "reasons": [],
           "rule": "|new - A6 CSV| <= " + ", ".join(f"{k} {v:g}" for k, v in E2_TOLERANCE.items()) + " (all 14)"}
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as err:
        return dict(res, reasons=[f"the A6 CSV {path} is unreadable ({err})"])
    res["csv_sha256"] = hashlib.sha256(raw).hexdigest()
    if res["csv_sha256"] != A6_CSV_SHA256:
        return dict(res, reasons=[f"the A6 CSV hashes to {res['csv_sha256']}, not the pinned {A6_CSV_SHA256}"])
    table = list(csv.DictReader(raw.decode().splitlines()))
    for k in (1, 8):
        hits = [r for r in table if (r["method"], r["K"], r["angle_deg"], r["seed"]) == ("FA B-F@40k", str(k), "0", "42")]
        _, _, new, why = _verified_cell(f"exp35_BF_T_S40000_K{k}_s42", manifest=manifest, pin=pin, base_dir=base_dir,
                                        farm_root=farm_root, dump_root=dump_root)
        res["reasons"] += why + ([] if len(hits) == 1 else [f"the A6 CSV holds {len(hits)} FA B-F@40k K{k} theta 0 s42 rows"])
        if why or len(hits) != 1:
            continue
        for key, tol in E2_TOLERANCE.items():
            a, b = float(new["metrics"]["metrics"][key]), float(hits[0][key])
            res["rows"].append({"K": k, "metric": key, "new": a, "a6": b, "abs_diff": abs(a - b), "tolerance": tol,
                                "pass": abs(a - b) <= tol})
    if res["reasons"]:
        return dict(res, rows=[])
    ok = len(res["rows"]) == 14 and all(r["pass"] for r in res["rows"])
    rows45 = [r for r in table if r["angle_deg"] == "45" and r["method"] in A6_METHODS.values() and int(r["seed"]) in SEEDS]
    return dict(res, status="PASS" if ok else "FAIL", rows45=rows45 if ok else [], **{"pass": ok})   # (b)'s 45 deg: on PASS


# --------------------------------------------------------------------------- #
# exp_35 collect mode, part 2 (plan r2 §4, frozen before any phase-(i) data, §11 R8): Tier 2-C4, Tier 3 for B-F
# --------------------------------------------------------------------------- #
def _bf_index(cells):
    """``(block, subblock, K, seed, deg) -> entry`` over the VALID B-F rows (BF-pin, BF-dup, OFF-BF)."""
    return {(e["row"]["block"], e["row"]["subblock"], e["row"]["K"], e["row"]["seed"], e["row"]["rotate_deg"]): e
            for e in cells.values() if e["valid"] and e["row"]["arm"] == "BF"}


def _distance(e, ref):
    """``(relL2 summary, relLinf summary, relL2 per item)`` of a dumped cell against its dumped theta = 0 pair."""
    from src.tools import depth_canon_stats as stats
    l2, linf = _paired(_waveforms(e), _waveforms(ref))
    return stats.paired_summary(l2), stats.paired_summary(linf), l2


def tier2_c4(cells, control):
    """§4 Tier 2-C4 -- OrbitRIR's invariance claim: each (K1, K8) x (90, 180, 270) pair at seed 42 against BF-pin T
    s42 with equal input_hash, p99(relL2) <= 3 x the matched B-F control p99 (d128 / d256 / d384) AND <= 1e-2,
    AND the pooled p99 of the six pairs' items <= 3 x the pooled control p99 (a control p99 of 0 demands a p99 of
    0, as registered for exp_25); alarm: any item > 0.1; T-dup exact (max 0, dump sha256 equal). The 45-degree and
    random-yaw distances are reported, outside the claim, with no verdict."""
    import numpy as np
    from src.tools import depth_canon_stats as stats
    if control is None:
        return {"pass": None, "claim": "NOT COMPUTABLE",
                "note": "no valid registered B-F composition control: Tier 2-C4 not computable"}
    idx, pairs, pooled, alarm = _bf_index(cells), [], [], {}
    for k in (1, 8):
        ref = idx.get(("BF-pin", "T", k, 42, 0))
        for deg in (90, 180, 270):
            e, key = idx.get(("BF-pin", "C4", k, 42, deg)), BF_CONTROL_KEYS[deg]
            p = {"cell": e["row"]["cell"] if e else None, "K": k, "deg": deg, "control_key": key,
                 "control_p99": control[key]}
            if not (e and ref and _same_items(e, ref) and e["dump"] and ref["dump"]):
                pairs.append(dict(p, computable=False, pass_matched=False, pass_cap=False))
                continue
            s2, sinf, l2 = _distance(e, ref)
            pooled.append(l2)
            if s2["max"] > TIER2_ALARM:
                alarm[p["cell"]] = [t[1] for t, v in zip(ref["stream"]["input_tuples"], l2) if v > TIER2_ALARM]
            pairs.append(dict(p, computable=True, relL2=s2, relLinf=sinf, pass_matched=s2["p99"] <= TIER2_FACTOR * control[key],
                              pass_cap=s2["p99"] <= TIER2_CAP))
    pool = stats.paired_summary(np.concatenate(pooled)) if len(pooled) == 6 else None
    pool_ok = bool(pool and pool["p99"] <= TIER2_FACTOR * control["pooled"])   # no pooled cap (Codex C3a HIGH 2)
    dup, descriptive = [], {"45": [], "R": []}
    for k in (1, 8):
        e, ref = idx.get(("BF-dup", "T", k, 42, 0)), idx.get(("BF-pin", "T", k, 42, 0))
        ok = bool(e and ref and _same_items(e, ref) and e["dump"] and ref["dump"])
        mx = float(_distance(e, ref)[2].max()) if ok else None
        same = bool(ok and e["receipt"]["dump_evidence"]["manifest"]["sha256"]
                    == ref["receipt"]["dump_evidence"]["manifest"]["sha256"])
        dup.append({"cell": e["row"]["cell"] if e else None, "K": k, "relL2_max": mx, "dump_sha_equal": same,
                    "pass": bool(ok and mx == 0 and same)})
        for kind, key, seed in [("45", ("BF-pin", "C4", k, 42, 45), 42)] + [("R", ("BF-pin", "R", k, s, None), s)
                                                                           for s in SEEDS]:
            e, ref = idx.get(key), idx.get(("BF-pin", "T", k, seed, 0))
            if e and ref and _same_items(e, ref) and e["dump"] and ref["dump"]:
                s2, sinf, _ = _distance(e, ref)
                descriptive[kind].append({"cell": e["row"]["cell"], "K": k, "seed": seed, "relL2": s2, "relLinf": sinf})
    verdict = len(pairs) == 6 and all(p["computable"] and p["pass_matched"] and p["pass_cap"] for p in pairs) and pool_ok
    return {"pass": verdict, "claim": "SUSPENDED" if alarm else "PASS" if verdict else "FAIL",   # Codex C3a HIGH 3
            "pairs": pairs, "pooled": {"relL2": pool, "control_p99": control["pooled"], "pass": pool_ok},
            "determinism": {"pass": all(d["pass"] for d in dup), "pairs": dup},
            "alarm": {"suspended": bool(alarm), "items": alarm,
                      "note": "the C4 invariance claim is suspended pending item triage" if alarm else "none"},
            "descriptive": descriptive,
            "rule": f"every C4 pair p99(relL2) <= {TIER2_FACTOR:g} x its matched B-F control p99 and <= {TIER2_CAP:g}, "
                    f"the pooled C4 p99 <= {TIER2_FACTOR:g} x the pooled control p99; alarm > {TIER2_ALARM:g}; "
                    "T-dup max relL2 == 0 and dump sha256 equal; 45-degree / random-yaw: descriptive"}


def tier3_bf(cells, exp25):
    """§4 Tier 3 for B-F (the registered rule, item-weighted, 28 checks with FD): pass_c4 = Delta_C4(BF) <= 0.1 x
    Delta_C4(P1-pin) -- the P1-pin value read at FULL precision from the hash-pinned exp_25 JSON (§11 N7) -- and
    pass_r = |Delta_R(BF)| <= sd(m_Z BF); the B-F estimators are depth_canon's own over the BF-pin cells only."""
    idx = {("BF", "R" if e["row"]["rotate_mode"] == "random" else "F", e["row"]["K"], e["row"]["seed"],
            e["row"]["step"], e["row"]["rotate_deg"]): e
           for e in cells.values() if e["valid"] and e["row"]["block"] == "BF-pin"}
    arms = {"BF": {k: {agg: {key: _arm_tier3(idx, "BF", k, agg, key) for key in TIER3_KEYS} for agg in AGGS}
                   for k in (1, 8)}}
    checks = []
    for k in (1, 8):
        for key in TIER3_KEYS:
            b, p1 = arms["BF"][k]["item"][key], exp25["tier3"]["arms"]["P1"][str(k)]["item"][key]["delta_c4"]
            known = None not in (b["delta_c4"], p1, b["delta_r"], b["sd_z"])
            checks.append({"K": k, "metric": key, "computable": known, "p1_delta_c4": p1,
                           "pass_c4": known and b["delta_c4"] <= TIER3_FRACTION * p1,
                           "pass_r": known and abs(b["delta_r"]) <= b["sd_z"]})
    verdict = None if not all(c["computable"] for c in checks) else all(c["pass_c4"] and c["pass_r"] for c in checks)
    return {"arms": arms, "checks": checks, "pass": verdict,
            "rule": f"Delta_C4(BF) <= {TIER3_FRACTION:g} x Delta_C4(P1-pin, the exp_25 JSON) and |Delta_R(BF)| <= "
                    "sd(m_Z BF), every metric incl. FD, both K, item-weighted"}


# --------------------------------------------------------------------------- #
# exp_35 collect mode, part 3 (plan r2 §1 (a)/(b)/(c), §5 items 1-7): the tables and the fail-closed build
# --------------------------------------------------------------------------- #
EXP25_WORKTREE = os.path.expanduser("~/codespace/exp-25-depth-canon")
EXP35_PROBE_JSON = os.path.join(EXP35_DIR, "orbitrir_bf_cond_probe.json")
A6_METHODS = {"FLAC": "Vanilla P1@40k", "OrbitRIR": "FA B-F@40k"}
EXP25_EXTERNAL = tuple(f"exp25_{a}_R_S40000_K{k}_s{s}" for a in ("P1", "CANON") for k in (1, 8) for s in SEEDS) + tuple(
    f"exp25_{a}_C4_S40000_K{k}_s42_th45" for a in ("P1", "CANON") for k in (1, 8))   # table (b) and the figure


_BF_ROWS = {1: "| fa scratch B-F @40k (equivariant, 59% budget) | fa eval (legacy-loop) | 1 | 5 | 9.543 ± 0.054 | "
                "1.0559 ± 0.0040 | 41.754 ± 0.347 | 5.166 ± 0.166 | 16.071 ± 0.241 | 23.721 ± 0.150 |",
            8: "| fa scratch B-F @40k (equivariant, 59% budget) | fa eval (legacy-loop) | 8 | 5 | 8.202 ± 0.017 | "
                "0.9778 ± 0.0015 | 38.793 ± 0.074 | 5.387 ± 0.075 | 16.456 ± 0.038 | 24.198 ± 0.164 |"}
BF_HISTORICAL_EXP35 = {   # Codex C3a NIT 1: table (a)'s historical B-F reference row, frozen (never read live)
    "label": BF_LABEL,
    "rows": {k: {"protocol": "fa eval (legacy-loop)", "n": 5, "values": dict(zip(
        ("T60", "C50", "EDT", "R@1", "R@5", "R@10"), [c.strip() for c in row.strip().strip("|").split("|")][4:10]))}
        for k, row in _BF_ROWS.items()},
    "provenance": "quoted from model_comparison.md (exp_07 B-F @40k, fa_invariant eval, legacy per-angle loop, its own "
                  "historical evaluator pin) -- frozen for exp_35 as the exp_25 results quoted it; not re-measured",
    "source": {"path": "worklog/worklog_yixun/model_comparison.md", "commit": "ed048609c9ca47ed413ab416d6e50a6225edfe04",
               "blob": "bc523e0a8ccc0afb59e93e90b536116c087ed295", "row_text": _BF_ROWS}}


def resolve(res, entry):
    """A table entry ``{"ref": "a.b.c"}`` -> ``res["a"]["b"]["c"]``: a reused row lives once (§5 item 3)."""
    node = res
    for key in entry["ref"].split("."):
        node = node[key]
    return node


def cell_artifacts(row, manifest, *, farm_root, dump_root, canon_index, base_dir):
    """EVERY artifact of one cell that exists -- record, stream / canon sidecar, receipt, dump -- with every partial or
    stale output beside them (``<path>.incomplete-<ts>``, ``<dump>.staging-*``): Codex C3a C7 for the E2-FAIL block."""
    target, _ = expected_checkpoint(row, manifest, canon_index)
    p, found = cell_paths(row, os.path.join(farm_root, row["arm"], os.path.basename(target)), dump_root), []
    for rel in (p.metrics, p.stream, p.canon, p.receipt, p.dump_dir):
        parent, name = os.path.split(os.path.join(base_dir, rel))
        try:
            entries = os.listdir(parent)
        except FileNotFoundError:                  # only a missing directory is absence (Codex C3b HIGH 1)
            continue
        except OSError as err:
            raise ValueError(f"RESULTS REFUSED: the artifacts of {row['cell']} cannot be listed ({parent}: {err})") from err
        found += [os.path.join(parent, e) for e in entries if e == name or e.startswith(name + ".")]
    return sorted(set(found))


def external_exp25_cells(manifest, *, base_dir, canon_index, farm_root=None, dump_root=None):
    """The exp_25 cells of table (b), verified OFFLINE at the exp_25 pin against the exp-25 worktree (§5):
    ``({cell: its flat item-weighted metrics}, reasons)``."""
    out, reasons = {}, []
    for name in EXP25_EXTERNAL:
        _, _, docs, why = _verified_cell(name, manifest=manifest, pin=EXP25_PIN, base_dir=base_dir, farm_root=farm_root,
                                         dump_root=dump_root, canon_index=canon_index)
        reasons += why
        if docs:
            out[name] = docs["metrics"]["metrics"]
    return out, reasons


EXT_BACKEND = "sdpa (plan §3b: exp_25's backend; its schema-1 receipts do not record it)"
POINT_FIELDS = ("seed", "cell", "campaign", "pin", "attention_backend")


def _point(e=None, *, name=None, external=None):
    """One seed of a pooled row: an H35 cell entry, or an external exp_25 cell (s42, verified at d6ca307a)."""
    if e is not None:
        return {"seed": e["row"]["seed"], "cell": e["row"]["cell"], "campaign": "exp35", "metrics": e["record"]["metrics"],
                "pin": e["receipt"]["expect_head"], "attention_backend": e["receipt"].get("attention_backend")}
    return {"seed": 42, "cell": name, "campaign": "exp25", "pin": EXP25_PIN, "attention_backend": EXT_BACKEND,
            "metrics": external[name]}


def _pointed(points, source):
    """A row pooled over more than one source (§14 A1.4): per K the TABLE_KEYS stats over its points, and every
    point's provenance (campaign, pin, backend) -- descriptive pooling, never a Tier 2 / Tier 3 pairing."""
    return dict(_seed_stats({k: [p["metrics"] for p in pts] for k, pts in points.items()}, source),
                points={str(k): [{f: p[f] for f in POINT_FIELDS} for p in pts] for k, pts in points.items()})


def _e2_fail_45(method, idx, external, e2f):
    """§14 A1.4, an E2 FAIL: s42 from the one-pin SDPA cell (P1: exp_25's C4+45 th45, external; B-F: BF-pin C4 th45)
    + s43-46 from the E2-FAIL block -- per K, n 5 if its four E2-FAIL cells verify, else the labelled s42 point (n 1)."""
    arm, pts, notes = {"FLAC": "P1", "OrbitRIR": "BF"}[method], {}, []
    for k in (1, 8):
        p1, bf = f"exp25_P1_C4_S40000_K{k}_s42_th45", ("BF-pin", "C4", k, 42, 45)
        first = ([_point(name=p1, external=external)] if p1 in external else []) if arm == "P1" else (
            [_point(idx[bf])] if bf in idx else [])
        rest = [e2f.get(f"exp35_{arm}_OFF_S40000_K{k}_s{s}_th45") for s in SEEDS[1:]]
        whole = all(e and e["valid"] for e in rest)
        pts[k] = first + ([_point(e) for e in rest] if whole else [])
        notes += [] if whole else [f"K{k}: the E2-FAIL cells are incomplete, so the s42 one-pin point only (n 1)"]
    one = "exp_25 P1-pin C4 th45 (external, d6ca307a)" if arm == "P1" else "BF-pin C4 th45 (this pin)"
    mix = "; the P1 pin mix is justified by E1 (the CTL cells are exactly equal at d6ca307a and this pin)"
    return _pointed(pts, f"E2 FAIL: s42 = the one-pin SDPA cell {one}, s43-46 = the E2-FAIL cells (this pin)"
                    + (mix if arm == "P1" else "") + "".join(f"; {x}" for x in notes))


def _seed_stats(flat, source):
    """``{"source", "1", "8"}``: per K the TABLE_KEYS mean ± sd over flat metric dicts, one per seed."""
    return dict({str(k): {key: _stats([float(m[key]) for m in ms]) for key in TABLE_KEYS} for k, ms in flat.items()},
                source=source)


def tables(cells, external, e2, e2f):
    """§1 (a) theta = 0, (b) 45 degrees and random yaw, (c) the Tier 3 spread. A reused exp_25 row is a pointer
    into ``external_reference`` (§5 item 3); every computed mean names its source; the A6 CSV reaches (b) only as
    E2's PASS rows (§5 item 6); on an E2 FAIL the 45-degree FLAC / OrbitRIR points are the s42 one-pin cells (§4)."""
    idx, ks = _bf_index(cells), (1, 8)
    h35 = lambda sub, k, seeds, deg: [idx[("BF-pin", sub, k, s, deg)]["record"]["metrics"]  # noqa: E731
                                      for s in seeds if ("BF-pin", sub, k, s, deg) in idx]
    ext = lambda fmt, k, seeds: [external[fmt.format(k=k, s=s)] for s in seeds  # noqa: E731
                                 if fmt.format(k=k, s=s) in external]
    if e2["status"] == "PASS":
        b45 = {m: _seed_stats({k: [r for r in e2["rows45"] if (r["method"], r["K"]) == (label, str(k))] for k in ks},
                              f"A6 CSV '{label}' 45 deg s42-46 (E2 PASS)") for m, label in A6_METHODS.items()}
    elif e2["status"] == "FAIL":
        b45 = {m: _e2_fail_45(m, idx, external, e2f) for m in A6_METHODS}
    else:
        b45 = {m: {"source": f"pending: E2 {e2['status']}", "1": None, "8": None} for m in A6_METHODS}
    c45 = sorted((e for e in cells.values() if e["valid"] and e["row"]["block"] == "CANON-45"), key=lambda e: e["row"]["seed"])
    b45["CANON"] = _pointed({k: [_point(name=n, external=external) for n in [f"exp25_CANON_C4_S40000_K{k}_s42_th45"]
                                 if n in external] + [_point(e) for e in c45 if e["row"]["K"] == k] for k in ks},
                            "exp_25 C4+45 th45 s42 (external, d6ca307a) + exp_35 CANON-45 s43-46")
    b = {"45": b45, "R": {"FLAC": _seed_stats({k: ext("exp25_P1_R_S40000_K{k}_s{s}", k, SEEDS) for k in ks},
                                              "exp_25 P1-pin R s42-46 (external, d6ca307a)"),
                          "CANON": _seed_stats({k: ext("exp25_CANON_R_S40000_K{k}_s{s}", k, SEEDS) for k in ks},
                                               "exp_25 CANON R s42-46 (external, d6ca307a)"),
                          "OrbitRIR": _seed_stats({k: h35("R", k, SEEDS, None) for k in ks}, "exp_35 BF-pin R s42-46")}}
    a = {"FLAC": {"ref": "external_reference.exp25.table.P1"}, "CANON": {"ref": "external_reference.exp25.table.CANON"},
         "OrbitRIR": _seed_stats({k: h35("T", k, SEEDS, 0) for k in ks}, "exp_35 BF-pin T s42-46"),
         "historical_bf": BF_HISTORICAL_EXP35}
    c = dict({arm: {"ref": f"external_reference.exp25.tier3.{arm}"} for arm in ("CANON", "P1", "YAW")},
             BF={"ref": "tier3.BF.arms.BF"})
    return a, b, c


def build_exp35_results(cells, *, pin, blocks, exp25, registry, registry_why, ctl, e2, control, control_why,
                        probe_present, eps, external, external_why, exp25_base_dir, partial, e2f=None,
                        e2f_present=()):
    """§5's fail-closed list -- it refuses even with ``partial``; cells not run yet, a B-F probe not run yet and an
    E2 that did not PASS are the only gaps, labelled under ``partial``."""
    e2f, fail, present = e2f or {}, e2["status"] == "FAIL", list(e2f_present)   # present: ANY artifact (C3a C7)
    every = {**cells, **(e2f if fail else {})}     # §14 A1.4: after an E2 FAIL the block is part of the campaign
    missing, invalid = completeness(every)
    digests = sorted({e["receipt"]["campaign_digest"] for e in every.values() if e["valid"]})
    problems = [f"E2 {e2['status']} but {len(present)} E2-FAIL cell(s) exist ({', '.join(present[:3])}): that block "
                "runs only after an E2 FAIL (§14 A1)"] if present and not fail else []
    problems += [f"stream registry: {x}" for x in registry_why] + [
        f"{n}: {x} (stream registry)" for n, e in every.items() if e["valid"] and registry
        for x in registry_reasons(e["row"], e["stream"], registry)]
    problems += [f"{len(invalid)} cell(s) failed verification ({', '.join(invalid[:3])})"] if invalid else []
    problems += [f"the cells span {len(digests)} campaign digests"] if len(digests) > 1 else []
    problems += [f"CTL K{k} {st} (E1 must be equal): {'; '.join(why[:2])}" for k, (st, why) in ctl.items() if st != "equal"]
    problems += [f"external exp_25 cells: {x}" for x in external_why]
    problems += [f"the A6 CSV is not the pinned one: {'; '.join(e2['reasons'][:1])}"] if e2["csv_sha256"] != A6_CSV_SHA256 else []
    problems += [f"the B-F probe: {x}" for x in control_why[:3]] if control is None and probe_present else []
    if problems:
        raise ValueError("RESULTS REFUSED (an integrity problem; --partial does not apply): " + "; ".join(problems))
    gaps = [f"{len(missing)} cell(s) not run ({', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''})"] if missing else []
    gaps += ["the B-F probe has not run: Tier 2-C4 is not computable"] if control is None else []
    gaps += [f"E2 {e2['status']}: the 45-degree FLAC / OrbitRIR points are pending"] if e2["status"] not in (
        "PASS", "FAIL") else []
    gaps += [f"E2 FAIL: {sum(n in missing for n in e2f)} E2-FAIL cell(s) not run, so those 45-degree points fall back "
             "to the s42 one-pin cell (n 1)"] if fail and any(n in missing for n in e2f) else []
    if gaps and not partial:
        raise ValueError("RESULTS REFUSED (pass --partial for a labelled partial table): " + "; ".join(gaps))
    a, b, c = tables(cells, external, e2, e2f)
    return {"schema_version": 1, "campaign": "exp35", "pin": pin, "blocks": list(blocks), "partial": bool(gaps),
            "gaps": gaps, "missing": missing, "campaign_digest": digests[0] if digests else None,
            "external_reference": {
                "exp25": {"path": os.path.relpath(os.path.join(EXPDIR, "depth_canon_results.json"), REPO),
                          "sha256": EXP25_RESULTS_SHA256, "pin": EXP25_PIN,
                          "table": {arm: exp25["table"][arm] for arm in ("P1", "CANON")},
                          "tier3": {arm: exp25["tier3"]["arms"][arm] for arm in ("CANON", "P1", "YAW")}},
                "exp25_cells": {"pin": EXP25_PIN, "base_dir": exp25_base_dir, "metrics": external,
                                "check": "verify_cell(pin=d6ca307a, dump_check='receipt', check_link=False)"},
                "registry_sha256": EXP35_REGISTRY_SHA256,
                "a6_csv": {"path": e2["csv"], "sha256": e2["csv_sha256"],
                           "read_by": "E2 only; its 45-degree rows enter table (b) on an E2 PASS"}},
            "e1_ctl": {str(k): {"status": st, "reasons": why} for k, (st, why) in ctl.items()}, "e2": e2,
            "tier2_c4": tier2_c4(cells, control), "eps_cond_bf": eps, "tier3": {"BF": tier3_bf(cells, exp25)},
            "table_a": a, "table_b": b, "table_c": c}


def exp35_results(manifest, *, pin, canon_index, blocks=None, base_dir=REPO, farm_root=None, dump_root=None,
                  exp25_base_dir=EXP25_WORKTREE, exp25_dump_root=None, bf_probe_json=EXP35_PROBE_JSON, a6_csv=None,
                  partial=False, check_link=True, dump_check="full"):
    """§5, the exp35 collect mode over exp_35 blocks only: the results dict, or ValueError("RESULTS REFUSED ...")."""
    blocks = list(blocks or EXP35_BLOCKS)
    if any(b not in EXP35_BLOCKS for b in blocks):
        raise ValueError(f"RESULTS REFUSED: the exp35 mode takes exp_35 blocks only, got {blocks}")
    farm, dump_root = farm_root or manifest["farm_root"], dump_root or manifest["dump_root"]
    kw = dict(pin=pin, manifest=manifest, base_dir=base_dir, farm_root=farm, dump_root=dump_root)
    exp25, exp25_why = load_exp25_reference()
    if exp25 is None:
        raise ValueError("RESULTS REFUSED: the external exp_25 results: " + "; ".join(exp25_why))
    kw_c = dict(farm_root=farm, dump_root=dump_root, pin=pin, canon_index=canon_index, base_dir=base_dir,
                check_link=check_link, dump_check=dump_check)
    cells = collect_cells(manifest, select_cells(manifest, [b for b in blocks if b not in EXPLICIT_BLOCKS]), **kw_c)
    e2f_rows = select_cells(manifest, list(EXPLICIT_BLOCKS))                         # §14 A1.4: always examined
    e2f = collect_cells(manifest, e2f_rows, **kw_c)
    e2f_present = [r["cell"] for r in e2f_rows if cell_artifacts(r, manifest, farm_root=farm, dump_root=dump_root,
                                                                 canon_index=canon_index, base_dir=base_dir)]
    registry, registry_why = load_registry()
    digests = {e["receipt"]["campaign_digest"] for e in cells.values() if e["valid"]}
    control, control_why, eps = load_control_bf(
        bf_probe_json, pin=pin, campaign_digest=next(iter(digests)) if len(digests) == 1 else None, manifest=manifest,
        bf_receipts=[e["receipt"] for e in cells.values() if e["valid"] and e["row"]["arm"] == "BF"], base_dir=base_dir)
    external, external_why = external_exp25_cells(manifest, base_dir=exp25_base_dir, canon_index=canon_index,
                                                  farm_root=farm, dump_root=exp25_dump_root or dump_root)
    return build_exp35_results(cells, pin=pin, blocks=blocks, exp25=exp25, registry=registry, registry_why=registry_why,
                               ctl={k: exp35_ctl_check(k, **kw) for k in (1, 8)}, e2=exp35_e2(csv_path=a6_csv, **kw),
                               control=control, control_why=control_why, probe_present=os.path.exists(bf_probe_json),
                               eps=eps, external=external, external_why=external_why, exp25_base_dir=exp25_base_dir,
                               partial=partial, e2f=e2f, e2f_present=e2f_present)


EXP35_RESULTS = os.path.join(EXP35_DIR, "orbitrir_yaw_results")   # §1: orbitrir_yaw_results.{md,json}


def _mean_n(st, key):
    s = (st or {}).get(key)
    return "—" if not s or s.get("mean") is None else _fmt(s, key) + f" (n {s['n']})"


def _verdict(v):
    return "PASS" if v else "not computable" if v is None else "FAIL"


def render_exp35_md(res):
    """orbitrir_yaw_results.md: the provenance, E1 / E2, tables (a) / (b) / (c) with the Tier 3 verdict, Tier 2-C4.
    Pointers resolve into ``external_reference``; nothing is recomputed here."""
    res = json.loads(json.dumps(res))                              # the results file's own view (string K keys)
    part, keys, ks = res["partial"], TABLE_KEYS, ("1", "8")
    h = lambda t: f"## {'PARTIAL — ' if part else ''}{t}"  # noqa: E731
    name = lambda key: SHORT.get(key, key)  # noqa: E731
    head, sep = "| method (source) | K | " + " | ".join(name(k) for k in keys) + " |", "|---|---|" + "---|" * len(keys)
    x = res["external_reference"]
    out = [f"# exp_35 orbitrir_yaw_evidence — results{' (PARTIAL)' if part else ''}", "",
           f"Evaluator pin `{res['pin']}`; campaign digest `{res['campaign_digest']}`; blocks {', '.join(res['blocks'])}. "
           "Written by `depth_canon_collect.py --exp35`; never hand-edited.", "",
           f"external_reference: the exp_25 results `{x['exp25']['path']}` (sha256 `{x['exp25']['sha256']}`, pin "
           f"`{x['exp25']['pin']}`; its rows are quoted, never paired with this campaign's cells); "
           f"{len(x['exp25_cells']['metrics'])} exp_25 cells verified offline at `{x['exp25_cells']['pin']}` "
           f"({x['exp25_cells']['check']}); the stream registry `{x['registry_sha256']}`; the A6 CSV "
           f"`{x['a6_csv']['sha256']}` ({x['a6_csv']['read_by']}).", ""]
    out += [f"**PARTIAL:** {'; '.join(res['gaps'])}.", ""] if part else []
    out += [h("E1 — the control cells (exact; a hard stop)"), ""] + [f"- CTL K{k}: {v['status']}"
                                                                    for k, v in res["e1_ctl"].items()]
    e2 = res["e2"]
    out += ["", h("E2 — decides the 45-degree source only"), "", f"{e2['status']} — {e2['rule']}.", ""] + [
        f"- K{r['K']} {name(r['metric'])}: new {r['new']:.6g}, A6 {r['a6']:.6g}, |diff| {r['abs_diff']:.3g} <= "
        f"{r['tolerance']:g}: {r['pass']}" for r in e2["rows"]]
    a = res["table_a"]
    out += ["", h("(a) theta = 0, unseen, mean ± sd over seeds 42–46, item-weighted"), "", head, sep]
    for m in ("FLAC", "CANON", "OrbitRIR"):
        d = a[m]
        out += [f"| {m} ({d.get('ref') or d['source']}) | {k} | " + " | ".join(
            _mean_n(resolve(res, d)[k]["item"] if "ref" in d else d[k], key) for key in keys) + " |" for k in ks]
    bf = a["historical_bf"]
    out += [f"| {bf['label']} — B-F HISTORICAL, quoted, reference only | {k} | " + " | ".join(
        bf["rows"].get(k, {}).get("values", {}).get(name(key), "—") for key in keys) + " |" for k in ks]
    out += ["", f"B-F HISTORICAL: {bf['provenance']}."]
    for angle, title in (("45", "45 degrees"), ("R", "random yaw")):
        out += ["", h(f"(b) {title}, unseen, mean ± sd over seeds 42–46, item-weighted"), "", head, sep]
        out += [f"| {m} ({res['table_b'][angle][m]['source']}) | {k} | " + " | ".join(
            _mean_n(res["table_b"][angle][m][k], key) for key in keys) + " |" for m in ("FLAC", "CANON", "OrbitRIR")
            for k in ks]
        pooled = [(m, d) for m, d in res["table_b"][angle].items() if "points" in d]   # §14 A1.4: per-point provenance
        out += ["", "Pooled rows, per point (seed, campaign, pin, attention backend):"] + [
            f"- {m} K{k}: " + "; ".join(f"s{p['seed']} {p['campaign']} @{p['pin'][:12]} ({p['attention_backend']})"
                                        for p in d["points"][k]) for m, d in pooled for k in ks] if pooled else []
    t3 = res["tier3"]["BF"]
    fails = {r: [f"K{c['K']} {name(c['metric'])}" for c in t3["checks"] if c["computable"] and not c[r]]
             for r in ("pass_c4", "pass_r")}
    out += ["", h("(c) Tier 3 spread, item-weighted (CANON / P1-pin / YAW-pin quoted from the exp_25 JSON) and the "
                  "B-F verdict"), "", f"**Tier 3 B-F: {_verdict(t3['pass'])}** — {t3['rule']}. pass_c4 fails: "
            f"{', '.join(fails['pass_c4']) or 'none'}; pass_r fails: {', '.join(fails['pass_r']) or 'none'}.", "",
            "| arm (source) | K | metric | Delta_C4 | 45-deg probe | Delta_R | sd(m_Z) |", "|---|---|---|---|---|---|---|"]
    out += [f"| {arm} ({d['ref']}) | {k} | {name(key)} | " + " | ".join(
        _num(resolve(res, d)[k]["item"][key][f]) for f in ("delta_c4", "probe45", "delta_r", "sd_z")) + " |"
        for arm, d in res["table_c"].items() for k in ks for key in TIER3_KEYS]
    t2 = res["tier2_c4"]
    alarm = (t2.get("alarm") or {}).get("items") or {}
    out += ["", h("Tier 2-C4 — OrbitRIR's invariance claim (scored dumps vs the B-F composition control)"), "",
            f"**{t2['claim']}** (the registered p99 verdict: {_verdict(t2['pass'])})" + (
                " — the alarm suspends the claim pending item triage; items " + "; ".join(
                    f"{c}: {', '.join(v)}" for c, v in alarm.items()) if t2["claim"] == "SUSPENDED" else "")
            + f" — {t2.get('rule') or t2.get('note')}.", ""]
    if "pairs" in t2:
        out += ["| cell | control | control p99 | relL2 p99 | relL2 max | matched | cap |", "|---|---|---|---|---|---|---|"]
        out += [f"| {p['cell']} | {p['control_key']} | {_num(p['control_p99'])} | {_num((p.get('relL2') or {}).get('p99'))}"
                f" | {_num((p.get('relL2') or {}).get('max'))} | {p['pass_matched']} | {p['pass_cap']} |" for p in t2["pairs"]]
        out += ["", f"Pooled C4: {_verdict(t2['pooled']['pass'])} (p99 {_num((t2['pooled']['relL2'] or {}).get('p99'))}, "
                f"control {_num(t2['pooled']['control_p99'])}). T-dup determinism: {_verdict(t2['determinism']['pass'])} ("
                + "; ".join(f"{d['cell']}: max {_num(d['relL2_max'])}, dump sha256 equal {d['dump_sha_equal']}"
                            for d in t2["determinism"]["pairs"]) + f"). Alarm: {t2['alarm']['note']}. eps_cond (B-F, "
                f"fp32; reported, no verdict): {res['eps_cond_bf']}.", "",
                "Descriptive only (outside the claim, no verdict) — 45 degrees and random yaw vs theta = 0:"]
        out += [f"- {d['cell']}: relL2 p99 {_num(d['relL2']['p99'])}, max {_num(d['relL2']['max'])}; relLinf max "
                f"{_num(d['relLinf']['max'])}" for kind in ("45", "R") for d in t2["descriptive"][kind]]
    return "\n".join(out) + "\n"


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
    ap.add_argument("--dump-check", choices=("full", "manifest", "receipt"), default="full",
                    help="receipt = offline: the receipt's dump evidence stands for the dump (no NAS)")
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
    ap.add_argument("--ckpt-bytes", type=int, default=None,
                    help="--write-receipt: the checkpoint size the driver measured (== the evaluator's loaded length)")
    ap.add_argument("--campaign-digest", default=None,
                    help="--plan: the relevant-tree digest at campaign start (required); --verify-cell: expected")
    ap.add_argument("--spawn-digest", default=None, help="--write-receipt: the tree digest at the spawn")
    ap.add_argument("--post-digest", default=None, help="--write-receipt: the tree digest after the run")
    ap.add_argument("--collect", action="store_true", help="the full mode: write the results files")
    ap.add_argument("--nas-retention-audit", action="store_true",
                    help="every PUBLISHED dump (receipt with dump evidence) of the selection still exists on "
                         "the NAS and matches its receipt (--dump-check full re-hashes, manifest is quick; "
                         "receipt is refused: it never reads the NAS)")
    ap.add_argument("--partial", action="store_true", help="write a labelled PARTIAL table")
    ap.add_argument("--probe-json", default=os.path.join(EXPDIR, "depth_canon_cond_probe.json"))
    ap.add_argument("--tie-audit-json", default=os.path.join(EXPDIR, "depth_canon_tie_audit.json"))
    ap.add_argument("--out-md", default=os.path.join(EXPDIR, "depth_canon_results.md"))
    ap.add_argument("--out-json", default=os.path.join(EXPDIR, "depth_canon_results.json"))
    ap.add_argument("--verify-after", action="store_true",
                    help="with --write-receipt: then verify the cell in the same process (exit 1 if invalid)")
    ap.add_argument("--check-backend-log", metavar="LOG", help="exp_35: exit 0 iff LOG proves the SDPA backend")
    ap.add_argument("--finalize-bf-probe", metavar="JSON", help="exp_35 §12 B3: finalize + validate the B-F probe")
    ap.add_argument("--probe-log", default=None, help="--finalize-bf-probe: the closed probe log (repo-relative)")
    ap.add_argument("--exp35-ctl-check", type=int, metavar="K", help="exp_35 E1: exit 0 equal / 1 mismatch / 2 refused")
    ap.add_argument("--exp35-e2", action="store_true", help="exp_35 E2: exit 0 PASS / 1 FAIL / 2 refused")
    ap.add_argument("--a6-csv", default=None, help="--exp35-e2: the A6 CSV (default: the pinned file)")
    ap.add_argument("--exp35", action="store_true", help="exp_35 §5: <--exp35-out>.{md,json} from exp_35 blocks only")
    ap.add_argument("--exp35-out", default=EXP35_RESULTS, help="--exp35: the output prefix (default: the exp_35 folder)")
    ap.add_argument("--exp25-base-dir", default=EXP25_WORKTREE, help="--exp35: the exp-25 worktree (the external cells)")
    ap.add_argument("--exp25-dump-root", default=None, help="--exp35: their dump root (default: the manifest's)")
    ap.add_argument("--bf-probe-json", default=EXP35_PROBE_JSON, help="--exp35: the finalized B-F probe")
    ap.add_argument("--count-no-flash", action="store_true",
                    help="exp_35: print how many cells of the --blocks / --only / --limit selection are exp_35 cells")
    ap.add_argument("--backend-target", default=NO_FLASH["target"],
                    help="exp_35: the launcher target the log must name (default eval_FLAC.py)")
    ap.add_argument("--attention-backend", default=None, help="--write-receipt, exp_35 cells: sdpa")
    ap.add_argument("--no-flash-overlay-sha256", default=None, help="--write-receipt, exp_35 cells")
    ap.add_argument("--no-flash-launcher-sha256", default=None, help="--write-receipt, exp_35 cells")
    args = ap.parse_args(argv)
    if args.check_backend_log:
        reasons = backend_log_file_reasons(args.check_backend_log, args.backend_target)
        print(("BACKEND OK " if not reasons else "BACKEND REFUSED ") + args.check_backend_log
              + "".join(f"\n  - {x}" for x in reasons))
        return 0 if not reasons else 1
    manifest = load_cells(args.cells)
    if args.finalize_bf_probe:   # exp_35 §12 B3: chunk 5's non-logged finalization, then the contract
        try:
            finalize_bf_probe(args.finalize_bf_probe, args.probe_log, campaign_digest=args.campaign_digest,
                              base_dir=args.base_dir)
        except (ValueError, OSError, TypeError) as err:
            print(f"BF PROBE FINALIZATION REFUSED: {err}")
            return 1
        _, reasons, _ = load_control_bf(args.finalize_bf_probe, pin=args.pin, campaign_digest=args.campaign_digest,
                                        manifest=manifest, base_dir=args.base_dir)
        print(("BF PROBE VALID " if not reasons else "BF PROBE INVALID ") + args.finalize_bf_probe
              + "".join(f"\n  - {x}" for x in reasons))
        return 0 if not reasons else 1
    if args.exp35:   # exp_35 §5: the exp35 collect mode
        if args.collect or (args.out_json, args.out_md) != (os.path.join(EXPDIR, "depth_canon_results.json"),
                                                            os.path.join(EXPDIR, "depth_canon_results.md")):
            ap.error("--exp35 writes <--exp35-out>.json / .md; --collect / --out-json / --out-md are the exp_25 mode")
        try:
            index = load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                     manifest["arms"]["CANON"]["ckpt"]["run_ns"])
            res = exp35_results(manifest, pin=args.pin, canon_index=index, blocks=_csv(args.blocks),
                                base_dir=args.base_dir, farm_root=args.farm_root, dump_root=args.dump_root,
                                exp25_base_dir=args.exp25_base_dir, exp25_dump_root=args.exp25_dump_root,
                                bf_probe_json=args.bf_probe_json, a6_csv=args.a6_csv, partial=args.partial,
                                check_link=not args.no_link_check, dump_check=args.dump_check)
        except (ValueError, OSError, KeyError) as err:
            print(str(err))
            return 3
        with open(args.exp35_out + ".json", "w") as f:
            json.dump(res, f, indent=1)
        with open(args.exp35_out + ".md", "w") as f:
            f.write(render_exp35_md(res))
        print(f"wrote {args.exp35_out}.md and .json ({'PARTIAL' if res['partial'] else 'complete'})")
        return 0
    if args.exp35_ctl_check is not None:   # exp_35 §3c E1, the hard stop after chunks 1-2
        if args.exp35_ctl_check not in (1, 8):
            print(f"CTL REFUSED: K must be 1 or 8, got {args.exp35_ctl_check}")
            return 2
        status, reasons = exp35_ctl_check(args.exp35_ctl_check, pin=args.pin, manifest=manifest,
                                          base_dir=args.base_dir, farm_root=args.farm_root, dump_root=args.dump_root)
        print(f"CTL K{args.exp35_ctl_check} {status.upper()}" + "".join(f"\n  - {x}" for x in reasons))
        return {"equal": 0, "mismatch": 1}.get(status, 2)
    if args.exp35_e2:   # exp_35 §4 E2, after chunks 3-4: the 45-degree source only
        res = exp35_e2(pin=args.pin, manifest=manifest, base_dir=args.base_dir, csv_path=args.a6_csv,
                       farm_root=args.farm_root, dump_root=args.dump_root)
        print(f"E2 {res['status']} ({res['rule']})" + "".join(f"\n  - {x}" for x in res["reasons"]) + "".join(
            f"\n  K{r['K']} {r['metric']}: new {r['new']!r} A6 {r['a6']!r} |diff| {r['abs_diff']:.6g} "
            f"<= {r['tolerance']:g}: {r['pass']}" for r in res["rows"]))
        return {"PASS": 0, "FAIL": 1}.get(res["status"], 2)
    if args.count_no_flash:   # exp_35 (Codex C2 HIGH 2): the runner's RELEVANT tree depends on it
        try:
            print(sum(is_exp35(c) for c in select_cells(manifest, _csv(args.blocks), _csv(args.only), args.limit)))
        except ValueError as err:
            print(f"SELECTION REFUSED: {err}")
            return 2
        return 0
    farm = args.farm_root or manifest["farm_root"]
    dump_root = args.dump_root or manifest["dump_root"]
    if args.write_receipt:   # rB-F1 B6: the receipt exists only after the FULL verification passed
        if not args.verify_after:
            print("RECEIPT REFUSED: a receipt is publication evidence; pass --verify-after (it is written only "
                  "after the cell's full verification)")
            return 2
        with open(args.plan_json) as f:
            plan = json.load(f)
        entry = next(e for e in plan["cells"] if e["cell"] == args.write_receipt)
        row = next(c for c in manifest["cells"] if c["cell"] == args.write_receipt)
        flags = (args.attention_backend, args.no_flash_overlay_sha256, args.no_flash_launcher_sha256)
        backend = None if flags == (None, None, None) else dict(zip(RECEIPT_V2_KEYS, flags))
        try:
            rec = build_receipt(entry, row, base_dir=args.base_dir, lane=args.lane, log=args.log, rc=args.rc,
                                pin=args.pin, ckpt_sha256=args.ckpt_sha, ckpt_bytes=args.ckpt_bytes,
                                campaign_digest=plan["campaign_digest"],
                                spawn_digest=args.spawn_digest, post_digest=args.post_digest, backend=backend)
            index = (load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                      manifest["arms"]["CANON"]["ckpt"]["run_ns"]) if row["arm"] == "CANON" else None)
        except (ValueError, OSError) as err:
            print(f"RECEIPT REFUSED: {err}")
            return 1
        reasons = verify_cell(entry["stem"], manifest=manifest, farm_root=farm, dump_root=dump_root, pin=args.pin,
                              canon_index=index, base_dir=args.base_dir, dump_check=args.dump_check,
                              expect_dump=entry["dump"], check_link=not args.no_link_check,
                              expect_digest=plan["campaign_digest"], receipt_doc=rec,
                              backend_target=args.backend_target)
        if reasons:
            print(f"INVALID {entry['stem']}\n" + "".join(f"  - {x}\n" for x in reasons)
                  + "RECEIPT REFUSED: the cell did not verify; no receipt written")
            return 1
        print("receipt: " + write_receipt(entry, row, base_dir=args.base_dir, receipt=rec))
        print("VALID " + entry["stem"])
        return 0
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
                              check_link=not args.no_link_check, expect_digest=args.campaign_digest,
                              backend_target=args.backend_target)
        print(("VALID " if not reasons else "INVALID ") + args.verify_cell)
        for reason in reasons:
            print("  - " + reason)
        return 0 if not reasons else 1
    if args.plan:
        if not (args.campaign_digest and eval_FLAC._is_sha256_hex(args.campaign_digest)):
            print("PLAN REFUSED: --campaign-digest (the relevant-tree sha256 at campaign start) is required")
            return 2
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
                       "campaign_digest": args.campaign_digest,
                       "farm_root": farm, "dump_root": dump_root, "n_selected": len(entries),
                       "cells": entries}, f, indent=1)
        print(f"plan: {len(entries)} cell(s) -> {args.plan}")
        return 0
    if args.nas_retention_audit:   # rB-F1 B6: the table never needs the NAS; this audits retention
        if args.dump_check not in ("full", "manifest"):   # rB-F2 F2-7: 'receipt' never reads the NAS
            print(f"RETENTION AUDIT REFUSED: --dump-check {args.dump_check} never reads the NAS; a retention "
                  "audit needs --dump-check full (re-hash) or manifest (quick)")
            return 2
        selected = select_cells(manifest, _csv(args.blocks), _csv(args.only), args.limit)
        index = load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                 manifest["arms"]["CANON"]["ckpt"]["run_ns"]) \
            if any(c["arm"] == "CANON" for c in selected) else None
        published, lost = 0, []
        for row in selected:
            target, _ = expected_checkpoint(row, manifest, index)
            p = cell_paths(row, os.path.join(farm, row["arm"], os.path.basename(target)), dump_root)
            try:
                with open(os.path.join(args.base_dir, p.receipt)) as f:
                    if json.load(f).get("dump_evidence") is None:
                        continue          # not published, or published without a dump: nothing to retain
            except (OSError, ValueError, AttributeError):
                continue
            published += 1
            reasons = verify_cell(p.stem, manifest=manifest, farm_root=farm, dump_root=dump_root, pin=args.pin,
                                  canon_index=index, base_dir=args.base_dir, dump_check=args.dump_check,
                                  expect_dump=True, check_link=False)
            if reasons:
                lost.append(f"  NOT RETAINED {row['cell']}: " + "; ".join(reasons[:3]))
        print(f"NAS retention: {published - len(lost)}/{published} published dump(s) retained and matching their "
              f"receipts (dump check {args.dump_check})")
        print("\n".join(lost))
        return 0 if not lost else 1
    if args.collect:
        try:
            blocks = _csv(args.blocks) or [b for b in manifest["blocks"] if b in EXP25_BLOCKS]
            if any(b not in EXP25_BLOCKS for b in blocks):   # exp_35 (plan r2 §3 P6): its own mode
                raise ValueError(f"RESULTS REFUSED: {[b for b in blocks if b not in EXP25_BLOCKS]} are not exp_25 "
                                 "blocks; --collect is the exp_25 mode (exp_35 cells: the exp35 mode)")
            selected = select_cells(manifest, blocks)
            index = load_canon_index(args.canon_ckpt_manifest or find_canon_manifest(),
                                     manifest["arms"]["CANON"]["ckpt"]["run_ns"])
            cells = collect_cells(manifest, selected, farm_root=farm, dump_root=dump_root, pin=args.pin,
                                  canon_index=index, base_dir=args.base_dir,
                                  check_link=not args.no_link_check,
                                  no_dump_blocks=_csv(args.no_dump_blocks), dump_check=args.dump_check)
            step = PROBE_CONTRACT["ckpt_step"]
            control, c_reasons, eps = load_control(args.probe_json, ckpt_sha256=index[step][1],
                                                   canon_receipts=canon_endpoint_receipts(cells), pin=args.pin,
                                                   manifest=manifest) \
                if step in index else (None, [f"no CANON@{step} checkpoint in the index"], None)
            audit, a_summary, a_reasons = load_tie_audit(args.tie_audit_json)
            res = build_results(cells, control=control, control_reasons=c_reasons, eps=eps, audit=audit,
                                audit_reasons=a_reasons, pin=args.pin, blocks=blocks, partial=args.partial,
                                audit_summary=a_summary)
        except (ValueError, OSError, KeyError) as err:
            print(str(err))
            return 3
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=1)
        with open(args.out_md, "w") as f:
            f.write(render_md(res))
        print(f"wrote {args.out_md} and {args.out_json} ({'PARTIAL' if res['partial'] else 'complete'})")
        return 0
    ap.error("nothing to do: pass --verify-cell, --plan, --write-receipt or --collect")
    return 2


if __name__ == "__main__":
    sys.exit(main())
