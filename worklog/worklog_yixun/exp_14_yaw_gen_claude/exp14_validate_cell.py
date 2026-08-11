#!/usr/bin/env python3
"""exp_14 — validate ONE yaw-generalization cell's artifacts (plan §3.1/§3.2/§3.3).

A landed cell is three files: the metrics record, its ``.screenmeta.json``
submission manifest and its ``.stream.json`` assignment audit. This module
answers one question about them — *are these the artifacts of the cell the
campaign registered, produced under the protocol it registered?* — and returns a
list of NAMED reasons, empty when the cell is valid.

It is deliberately the same predicate in three places:

* ``yaw_gen_screen.sbatch`` runs it before emitting SCREENRESULT, so a cell that
  cannot be validated never announces a result;
* ``yaw_gen_submit_grid.sh`` runs it for dedup, where the rule is
  validate-BEFORE-skip (review B6): a cell is skipped only when its artifacts
  pass every check, and an artifact that exists but fails one HALTS the wave for
  triage rather than being silently skipped or overwritten;
* the round-3 collector runs it before any contrast.

What it does NOT do is cross-cell reasoning. Z<->R pairing, cross-arm rotation
matching and 5/5 seed blocks are equalities BETWEEN cells (plan §3.3, gates
G1-G4) and belong to the collector; claiming them per cell would be claiming
evidence this file cannot see.

**No torch.** The submitter classifies ~100 cells on a shared login node, so this
module imports nothing heavier than the standard library. Two rules that live in
``eval_FLAC`` are therefore mirrored here — the canonical stream serialization
and the metrics-path shape — and ``src/tests/test_exp14_validate_cell.py`` pins
both mirrors to their originals so they cannot drift.

Usage
-----
    python3 exp14_validate_cell.py grid [--wave {vctl,zref,rgen,all}]
    python3 exp14_validate_cell.py check --metrics PATH --arm ARM --cell CELL \\
        --step 40000 --seed SEED --k K [--rotate-deg DEG] [--pin SHA] \\
        [--ckpt-sha SHA] [--expected-count 6337]
    python3 exp14_validate_cell.py classify --wave WAVE --output-root DIR \\
        [--pin SHA] [--expected-count 6337]

``check`` exits 0 (valid), 1 (invalid — reasons on stdout), 2 (unregistered cell
or usage error) or 3 (the artifact is absent, which is a normal "not run yet").
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import namedtuple

# --- the registered campaign (plan §3.1/§3.2) --------------------------------
ARMS = ("VANL", "C4L", "C8", "C16", "C32")
TRAIN_ORBIT = {"VANL": 0, "C4L": 4, "C8": 8, "C16": 16, "C32": 32}
CELLS = ("rgen", "zref", "vctl")
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
STEP = 40000
# The six validity controls, exhaustively. VANL@45 is deliberately absent: the
# positive control is VANL@90, and an off-group vanilla angle answers no question
# this campaign asked.
VCTL_TUPLES = (("C4L", 90.0), ("C8", 90.0), ("C16", 90.0), ("C32", 90.0),
               ("VANL", 90.0), ("C4L", 45.0))
# Campaign constants. Every cell must report exactly these or it is not a member
# of the block it claims to belong to (plan §3.2).
EXPECTED_COUNT = 6337
BATCH_SIZE = 64
NUM_WORKERS = 4
COND_AUTOCAST = "bf16"
CFG_SCALE = 1.0
STEPS = 1
IMG_W = 512
# Schema versions this validator understands (eval_FLAC.STREAM_SCHEMA_VERSION /
# CONTEXT_FINGERPRINT_SCHEMA). A bump means the meaning of a field changed, so a
# sidecar written under another version is not comparable and is refused.
STREAM_SCHEMA_VERSION = 1
FINGERPRINT_SCHEMA = 1
SPLIT_K8 = "acousticroom_unseeneval.json"
SPLIT_K1 = "acousticroom_unseeneval_1.json"

Cell = namedtuple("Cell", "arm cell step seed k rotate_deg")


def expected_grid():
    """The 106 registered cells: 50 rgen + 50 zref + 6 vctl, all at STEP."""
    cells = []
    for name in ("rgen", "zref"):
        for arm in ARMS:
            for k in KS:
                for seed in SEEDS:
                    cells.append(Cell(arm, name, STEP, seed, k, None))
    for arm, deg in VCTL_TUPLES:
        cells.append(Cell(arm, "vctl", STEP, 42, 8, float(deg)))
    return tuple(cells)


_GRID = expected_grid()
_GRID_SET = frozenset(_GRID)

WAVES = ("vctl", "zref", "rgen", "all")


def wave_cells(wave):
    """The cells of one submission wave (plan §6: vctl first, then zref, then rgen)."""
    if wave not in WAVES:
        raise ValueError(f"unknown wave {wave!r}; registered waves: {list(WAVES)}")
    if wave == "all":
        return _GRID
    return tuple(c for c in _GRID if c.cell == wave)


def is_registered(cell):
    return cell in _GRID_SET


# --- naming ------------------------------------------------------------------
def fmt_deg(deg):
    """``90.0 -> '90'``, ``5.625 -> '5p625'`` — eval_FLAC.rot_token's rendering."""
    d = float(deg)
    return str(int(d)) if d.is_integer() else repr(d).replace(".", "p")


def rotation_suffix(cell):
    """The cell's rotation token: ``_rotrand<seed>`` / ``_rot<deg>`` / ``''``.

    Mirrors eval_FLAC.rotation_suffix for the protocols this campaign uses, and is
    what the screen driver renders in shell for the eval name.
    """
    if cell.cell == "rgen":
        return f"_rotrand{int(cell.seed)}"
    if cell.cell == "vctl":
        return f"_rot{fmt_deg(cell.rotate_deg)}"
    return ""


def eval_name(cell):
    """``exp14_<ARM>_<CELL>_S<STEP>_s<SEED>_K<K><token>`` — injective over the grid."""
    return (f"exp14_{cell.arm}_{cell.cell}_S{int(cell.step)}_s{int(cell.seed)}"
            f"_K{int(cell.k)}{rotation_suffix(cell)}")


def parse_eval_name(name):
    """Inverse of :func:`eval_name`, restricted to REGISTERED cells.

    A name that parses but names no registered cell is rejected: the point of a
    registered grid is that an unregistered artifact cannot be read as evidence.
    """
    parts = name.split("_")
    if len(parts) < 6 or parts[0] != "exp14":
        raise ValueError(f"{name!r} is not an exp_14 eval name")
    arm, cellname = parts[1], parts[2]
    tail = parts[3:]
    if cellname not in CELLS:
        raise ValueError(f"{name!r} names no exp_14 cell type")
    try:
        step = int(tail[0][1:]) if tail[0].startswith("S") else None
        seed = int(tail[1][1:]) if tail[1].startswith("s") else None
        k = int(tail[2][1:]) if tail[2].startswith("K") else None
    except (IndexError, ValueError):
        raise ValueError(f"{name!r} does not carry S<step>_s<seed>_K<k>")
    if None in (step, seed, k):
        raise ValueError(f"{name!r} does not carry S<step>_s<seed>_K<k>")
    deg = None
    if cellname == "vctl":
        tok = tail[3] if len(tail) > 3 else ""
        if not tok.startswith("rot") or tok.startswith("rotrand"):
            raise ValueError(f"{name!r} is a vctl cell without a fixed-angle token")
        try:
            deg = float(tok[len("rot"):].replace("p", "."))
        except ValueError:
            raise ValueError(f"{name!r} carries an unreadable rotation token {tok!r}")
    cell = Cell(arm, cellname, step, seed, k, deg)
    if eval_name(cell) != name:
        raise ValueError(f"{name!r} is not the registered name of the cell it describes "
                         f"(expected {eval_name(cell)!r})")
    if not is_registered(cell):
        raise ValueError(f"{name!r} names an UNREGISTERED cell {tuple(cell)}")
    return cell


# --- rules mirrored from eval_FLAC (pinned by tests) -------------------------
def canonical_stream_hash(tuples):
    """sha256 over an ordered tuple stream, in the pre-registered serialization.

    Mirror of ``eval_FLAC.canonical_stream_hash`` (plan §3.3): one JSON array per
    tuple, ``sort_keys=True``, ``separators=(",", ":")``, LF-joined, UTF-8.
    """
    payload = "\n".join(
        json.dumps(list(t), sort_keys=True, separators=(",", ":")) for t in tuples
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_column_shift(deg, img_w=IMG_W):
    """The integer column shift a fixed angle quantises to (yaw_column_shift's rule)."""
    return int(round(float(deg) / 360.0 * int(img_w))) % int(img_w)


def cond_method(arm):
    return "vanilla" if arm == "VANL" else "fa_invariant"


def frame_avg_angles(arm):
    n = TRAIN_ORBIT[arm]
    return None if n == 0 else [k * 360.0 / n for k in range(n)]


def metrics_path(ckpt_path, cell):
    """Where eval_FLAC writes this cell's metrics JSON (mirror of build_output_paths)."""
    ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
    directory = os.path.dirname(ckpt_path)
    method = cond_method(cell.arm)
    method_suffix = "" if method == "vanilla" else f"_{method}_a{TRAIN_ORBIT[cell.arm]}"
    stem = f"{STEPS}_{CFG_SCALE}_{eval_name(cell)}{method_suffix}{rotation_suffix(cell)}"
    return os.path.join(directory, f"{ckpt_name}_metrics_{stem}.json")


def stream_path(metrics):
    base = metrics[:-len(".json")] if metrics.endswith(".json") else metrics
    return base + ".stream.json"


def screenmeta_path(metrics):
    return metrics + ".screenmeta.json"


# The audited per-arm checkpoint expectation, published once by
# exp14_hash_ckpts.py and READ (never recomputed) from here on: re-hashing five
# 724 MB files per wave on a shared login node is not a thing to do repeatedly.
CKPT_EXPECT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exp14_ckpt_expect.json")


def load_ckpt_expect(path=CKPT_EXPECT, arms=ARMS, step=STEP):
    """``arm -> sha256`` from the audited expectation; anything missing is fatal.

    Fail-closed by construction (review B4): a dedup SKIP is a decision to keep a
    number without re-measuring it, so it must rest on WHICH checkpoint produced
    that number. An absent or incomplete expectation therefore raises rather than
    letting the check quietly not run.
    """
    if not os.path.isfile(path):
        raise ValueError(
            f"audited checkpoint expectation missing: {path}. Publish it once with "
            "exp14_hash_ckpts.py --write; a cell cannot be skipped as 'already "
            "measured' without knowing which checkpoint measured it.")
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("arms"), dict):
        raise ValueError(f"checkpoint expectation {path} is malformed")
    if int(raw.get("step", -1)) != int(step):
        raise ValueError(f"checkpoint expectation {path} is for step {raw.get('step')!r}, "
                         f"not the campaign endpoint {step}")
    out = {}
    for arm in arms:
        row = raw["arms"].get(arm)
        sha = (row or {}).get("sha256")
        if not (isinstance(sha, str) and len(sha) == 64):
            raise ValueError(f"checkpoint expectation {path} has no sha256 for arm {arm}")
        out[arm] = sha
    return out


def checkpoint_path(output_root, arm, step=STEP):
    """The arm's single checkpoint at ``step``, or None if it is not unique."""
    pat = os.path.join(output_root, f"exp11_{arm}", f"FLAC_exp11_{arm}",
                       f"exp11_{arm}", "checkpoints", f"epoch=*-step={step}.ckpt")
    hits = sorted(glob.glob(pat))
    return hits[0] if len(hits) == 1 else None


# --- the expectations one cell must meet -------------------------------------
def rotation_expectation(cell):
    """``(rotate_mode, rotate_deg, rotate_seed)`` for a registered cell.

    The angle and the seed are mutually exclusive by design: a fixed cell has an
    angle and no seed, a random cell has a seed and a NULL angle. Recording 0.0
    for a random cell would make it indistinguishable from an unrotated one.
    """
    if cell.cell == "rgen":
        return "random", None, int(cell.seed)
    if cell.cell == "vctl":
        return "fixed", float(cell.rotate_deg), None
    return "fixed", 0.0, None


def _eq(reasons, label, got, want):
    if got != want:
        reasons.append(f"{label} {got!r} != expected {want!r}")


def validate_metrics_record(rec, cell, pin=None, expected_count=EXPECTED_COUNT):
    """Named reasons why ``rec`` is not this cell's metrics record ([] = valid)."""
    reasons = []
    mode, deg, rseed = rotation_expectation(cell)
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        reasons.append("metrics block is missing or empty")
    _eq(reasons, "eval_name", rec.get("eval_name"), eval_name(cell))
    _eq(reasons, "seed", rec.get("seed"), int(cell.seed))
    _eq(reasons, "cfg_scale", rec.get("cfg_scale"), CFG_SCALE)
    _eq(reasons, "steps", rec.get("steps"), STEPS)
    _eq(reasons, "cond_autocast", rec.get("cond_autocast"), COND_AUTOCAST)
    _eq(reasons, "batch_size", rec.get("batch_size"), BATCH_SIZE)
    _eq(reasons, "cond_method", rec.get("cond_method"), cond_method(cell.arm))
    _eq(reasons, "frame_avg_angles", rec.get("frame_avg_angles"), frame_avg_angles(cell.arm))
    ds = rec.get("dataset_config") or ""
    want_split = SPLIT_K8 if int(cell.k) == 8 else SPLIT_K1
    if not str(ds).endswith(want_split):
        reasons.append(f"dataset_config {ds!r} is not the K={cell.k} split ({want_split})")
    # n_samples is REQUIRED, not "checked when present": an absent count is
    # exactly what a partially-evaluated split looks like.
    if rec.get("n_samples") != int(expected_count):
        reasons.append(f"n_samples {rec.get('n_samples')!r} != expected {int(expected_count)} "
                       "(required: a cell that does not report its item count is not a "
                       "complete evaluation of the split)")
    # eval_FLAC falls back to ONLINE weights when the EMA entries are absent, and
    # says so only in this field. A non-EMA row is not the registered cell.
    _eq(reasons, "weights_source", rec.get("weights_source"), "ema")
    ckpt = str(rec.get("ckpt_path") or "")
    if f"step={int(cell.step)}." not in ckpt:
        reasons.append(f"ckpt_path {ckpt!r} is not a step={cell.step} checkpoint")
    if pin is not None:
        _eq(reasons, "source_sha (campaign pin)", rec.get("source_sha"), pin)
    # --- the yaw protocol (announcement 05) ---
    # Fixed-mode records are FROZEN (review B4): eval_FLAC adds the random-mode
    # provenance keys only in random mode, so their presence on a zref/vctl cell
    # means it ran a protocol other than the one it claims.
    random_keys = ("rotate_mode", "rotate_seed", "input_hash", "assignment_hash",
                   "stream_count", "img_w")
    if mode == "random":
        _eq(reasons, "rotate_mode", rec.get("rotate_mode"), "random")
        if "rotate_deg" not in rec or rec.get("rotate_deg") is not None:
            reasons.append(f"rotate_deg {rec.get('rotate_deg')!r} is not null: a random-yaw "
                           "cell has no single angle, and 0.0 would read as unrotated")
        _eq(reasons, "rotate_seed", rec.get("rotate_seed"), rseed)
        _eq(reasons, "stream_count", rec.get("stream_count"), int(expected_count))
        _eq(reasons, "img_w", rec.get("img_w"), IMG_W)
        for key in ("input_hash", "assignment_hash"):
            val = rec.get(key)
            if not (isinstance(val, str) and len(val) == 64):
                reasons.append(f"{key} {val!r} is not a sha256 digest")
    else:
        present = [k for k in random_keys if k in rec]
        if present:
            reasons.append(f"fixed-mode record carries random-mode keys {present}: "
                           "rotate_mode/hash provenance is added only in random mode")
        got = rec.get("rotate_deg")
        if got is None or float(got) != float(deg):
            reasons.append(f"rotate_deg {got!r} != expected {deg!r}")
    return reasons


def validate_stream_record(stream, cell, expected_count=EXPECTED_COUNT, record=None):
    """Named reasons why the ``.stream.json`` audit is not this cell's ([] = valid)."""
    reasons = []
    mode, deg, rseed = rotation_expectation(cell)
    _eq(reasons, "stream schema_version", stream.get("schema_version"), STREAM_SCHEMA_VERSION)
    _eq(reasons, "stream fingerprint_schema", stream.get("fingerprint_schema"), FINGERPRINT_SCHEMA)
    _eq(reasons, "stream rotate_mode", stream.get("rotate_mode"), mode)
    _eq(reasons, "stream rotate_seed", stream.get("rotate_seed"), rseed)
    stream_deg = stream.get("rotate_deg")
    if (deg is None) != (stream_deg is None) or (deg is not None
                                                 and float(stream_deg) != float(deg)):
        reasons.append(f"stream rotate_deg {stream_deg!r} != expected {deg!r}")
    inp = stream.get("input_tuples")
    asg = stream.get("assignment_tuples")
    offs = stream.get("offsets")
    if not isinstance(inp, list) or not isinstance(asg, list) or not isinstance(offs, list):
        reasons.append("stream is missing input_tuples / assignment_tuples / offsets")
        return reasons
    n = int(expected_count)
    for label, seq in (("stream_count", stream.get("stream_count")),
                       ("input_tuples", len(inp)), ("assignment_tuples", len(asg)),
                       ("offsets", len(offs))):
        if seq != n:
            reasons.append(f"{label} {seq!r} != expected {n} positions")
    for i, row in enumerate(inp):
        if not isinstance(row, list) or len(row) != 4 or row[0] != i:
            reasons.append(f"stream position {i} is malformed or out of order: {row!r}")
            break
    if [row[2] if isinstance(row, list) and len(row) == 3 else None for row in asg] != list(offs):
        reasons.append("stored offsets disagree with the assignment tuples' offsets")
    # Each assignment tuple must name the position and the TARGET the input stream
    # says was evaluated there. Without this the two hashes can each be internally
    # consistent while attributing an offset to an item that never received it —
    # which is precisely what the audit exists to make impossible.
    for i, row in enumerate(asg):
        if not isinstance(row, list) or len(row) != 3 or row[0] != i:
            reasons.append(f"assignment tuple {i} is malformed or out of position: {row!r}")
            break
        if i < len(inp) and isinstance(inp[i], list) and len(inp[i]) == 4 \
                and row[1] != inp[i][1]:
            reasons.append(f"assignment tuple {i} names target {row[1]!r} but the input "
                           f"stream evaluated {inp[i][1]!r} at that position")
            break
    _eq(reasons, "recomputed input_hash", stream.get("input_hash"), canonical_stream_hash(inp))
    _eq(reasons, "recomputed assignment_hash", stream.get("assignment_hash"),
        canonical_stream_hash(asg))
    if record is not None and mode == "random":
        # The record and its sidecar must be two views of ONE stream.
        _eq(reasons, "record input_hash vs stream", record.get("input_hash"),
            stream.get("input_hash"))
        _eq(reasons, "record assignment_hash vs stream", record.get("assignment_hash"),
            stream.get("assignment_hash"))
    # The estimand is defined on the 512-column grid (theta_i = d_i * 360/512), so
    # the width is a campaign constant, not a value to be inferred: an absent img_w
    # used to default to 512 and a different width used to pass.
    img_w = stream.get("img_w")
    if img_w != IMG_W:
        reasons.append(f"stream img_w {img_w!r} != the campaign column grid {IMG_W}")
        img_w = IMG_W if not isinstance(img_w, int) or img_w <= 0 else img_w
    bad = [o for o in offs if not isinstance(o, int) or o < 0 or o >= int(img_w)]
    if bad:
        reasons.append(f"{len(bad)} offset(s) fall outside the [0,{img_w}) column grid "
                       f"(first {bad[0]!r})")
    if mode == "fixed":
        want = expected_column_shift(deg, img_w)
        wrong = [o for o in offs if o != want]
        if wrong:
            reasons.append(f"{len(wrong)} offset(s) != the constant shift {want} that "
                           f"{deg} deg quantises to (first {wrong[0]!r})")
    elif offs and len(set(offs)) == 1:
        reasons.append(f"all {len(offs)} offsets are {offs[0]!r}: {n} independent draws over "
                       f"{img_w} columns are never constant — the random path did not run")
    return reasons


def validate_screenmeta(meta, cell, pin=None, ckpt_sha=None,
                        expected_count=EXPECTED_COUNT):
    """Named reasons why the submission manifest does not describe this cell."""
    reasons = []
    mode, deg, rseed = rotation_expectation(cell)
    _eq(reasons, "screenmeta arm", meta.get("arm"), cell.arm)
    _eq(reasons, "screenmeta cell", meta.get("cell"), cell.cell)
    _eq(reasons, "screenmeta step", meta.get("step"), int(cell.step))
    _eq(reasons, "screenmeta seed", meta.get("seed"), int(cell.seed))
    _eq(reasons, "screenmeta K", meta.get("K"), int(cell.k))
    _eq(reasons, "screenmeta eval_name", meta.get("eval_name"), eval_name(cell))
    _eq(reasons, "screenmeta cond_method", meta.get("cond_method"), cond_method(cell.arm))
    _eq(reasons, "screenmeta cond_autocast", meta.get("cond_autocast"), COND_AUTOCAST)
    _eq(reasons, "screenmeta rotate_mode", meta.get("rotate_mode"), mode)
    _eq(reasons, "screenmeta rotate_seed", meta.get("rotate_seed"), rseed)
    meta_deg = meta.get("rotate_deg")
    if (deg is None) != (meta_deg is None) or (deg is not None and float(meta_deg) != float(deg)):
        reasons.append(f"screenmeta rotate_deg {meta_deg!r} != expected {deg!r}")
    _eq(reasons, "screenmeta batch_size", meta.get("batch_size"), BATCH_SIZE)
    _eq(reasons, "screenmeta num_workers", meta.get("num_workers"), NUM_WORKERS)
    _eq(reasons, "screenmeta expected_stream_count", meta.get("expected_stream_count"),
        int(expected_count))
    _eq(reasons, "screenmeta record_stream", meta.get("record_stream"), True)
    _eq(reasons, "screenmeta use_ema", meta.get("use_ema"), True)
    _eq(reasons, "screenmeta training_orbit", meta.get("training_orbit"), TRAIN_ORBIT[cell.arm])
    _eq(reasons, "screenmeta eval_orbit", meta.get("eval_orbit"), TRAIN_ORBIT[cell.arm])
    if pin is not None:
        _eq(reasons, "screenmeta commit (campaign pin)", meta.get("commit"), pin)
    if ckpt_sha is not None:
        _eq(reasons, "screenmeta ckpt_sha256", meta.get("ckpt_sha256"), ckpt_sha)
    return reasons


def _read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _read_record(path, label):
    """``(record, reasons)`` — a malformed or non-object payload NAMES itself.

    Valid JSON that is not an object (a list, a string, ``null``) used to reach
    ``.get`` and raise, which turns a triage question into a crashed submitter.
    """
    try:
        obj = _read_json(path)
    except (ValueError, OSError) as exc:
        return None, [f"could not parse {label} JSON {path}: {exc}"]
    if not isinstance(obj, dict):
        return None, [f"{label} {path} is not a JSON object at the top level "
                      f"(got {type(obj).__name__})"]
    return obj, []


def validate_cell(metrics, cell, pin=None, ckpt_sha=None, expected_count=EXPECTED_COUNT):
    """Every named reason this cell's artifacts are not valid; ``[]`` when they are.

    Raises ``ValueError`` for an UNREGISTERED cell before touching the filesystem:
    an artifact for a cell the plan never registered is not a validation failure,
    it is a question that should not have been asked.
    """
    if not is_registered(cell):
        raise ValueError(f"cell {tuple(cell)} is not registered in the exp_14 grid")
    if not os.path.isfile(metrics):
        return [f"metrics artifact missing: {metrics}"]
    rec, bad = _read_record(metrics, "metrics")
    if rec is None:
        return bad
    # A VALID verdict must mean every CAMPAIGN check ran, not that some of them
    # had no input (review B6). The pin and the checkpoint digest are the two that
    # a caller could previously omit and still be told the cell is valid.
    reasons = []
    if pin is None:
        reasons.append("campaign pin not supplied: a cell cannot be declared valid "
                       "without checking which commit produced it")
    if ckpt_sha is None:
        reasons.append("expected ckpt sha256 not supplied: a cell cannot be declared "
                       "valid without checking WHICH checkpoint it evaluated")
    reasons += validate_metrics_record(rec, cell, pin=pin, expected_count=expected_count)

    meta_p = screenmeta_path(metrics)
    if not os.path.isfile(meta_p):
        reasons.append(f"screenmeta sidecar missing: {meta_p}")
    else:
        meta, bad = _read_record(meta_p, "screenmeta")
        reasons += bad
        if meta is not None:
            reasons += validate_screenmeta(meta, cell, pin=pin, ckpt_sha=ckpt_sha,
                                           expected_count=expected_count)

    stream_p = stream_path(metrics)
    if not os.path.isfile(stream_p):
        reasons.append(f"stream sidecar missing: {stream_p} (--record-stream is "
                       "mandatory for every exp_14 cell)")
    else:
        stream, bad = _read_record(stream_p, "stream")
        reasons += bad
        if stream is not None:
            reasons += validate_stream_record(stream, cell, expected_count=expected_count,
                                              record=rec)
    return reasons


# --- CLI ---------------------------------------------------------------------
def parse_deg(value):
    """Tolerant reading of a rotation angle from a COMMAND LINE: float or None.

    The shell has no null. A caller that means "no angle" can only say so by
    omitting the flag, or by passing 0, "0", "0.0" or "" — and a validator that
    treated the string "0" as "an angle was requested" rejected 100 of the 106
    cells AFTER they had spent their GPU (review B1). Everything that spells
    "nothing" therefore reads as None here, and everything else must parse as a
    float or it is a hard usage error.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "none", "null", "-"):
        return None
    deg = float(text)                      # ValueError -> caller reports usage
    return None if deg == 0.0 else deg


def check_argv(cell, metrics, pin=None, ckpt_sha=None, expected_count=EXPECTED_COUNT):
    """The canonical ``check`` argv for one cell — the ONE definition of it.

    The screen driver renders these flags in shell (it already knows every value
    and a python round-trip would cost a process); a guard case pins its rendering
    against this function, so the two cannot drift. Note what is NOT here: a
    ``--rotate-deg`` outside vctl. rgen draws its angles and zref has none, so
    passing 0 would be a claim about a protocol neither of them ran.
    """
    argv = ["check", "--metrics", str(metrics), "--arm", cell.arm, "--cell", cell.cell,
            "--step", str(int(cell.step)), "--seed", str(int(cell.seed)),
            "--k", str(int(cell.k))]
    if cell.cell == "vctl":
        argv += ["--rotate-deg", fmt_deg(cell.rotate_deg)]
    if pin is not None:
        argv += ["--pin", str(pin)]
    if ckpt_sha is not None:
        argv += ["--ckpt-sha", str(ckpt_sha)]
    argv += ["--expected-count", str(int(expected_count))]
    return argv


def _fmt_cell(cell):
    rot = "-" if cell.rotate_deg is None else fmt_deg(cell.rotate_deg)
    return f"{cell.arm} {cell.cell} {cell.step} {cell.seed} {cell.k} {rot}"


def _cmd_grid(args):
    for cell in wave_cells(args.wave):
        print(_fmt_cell(cell))
    return 0


def _cell_from_args(args):
    """The Cell a CLI invocation names, or a usage error (ValueError)."""
    deg = parse_deg(getattr(args, "rotate_deg", None))
    if args.cell == "vctl":
        if deg is None:
            raise ValueError("a vctl cell needs a non-zero --rotate-deg")
    elif deg is not None:
        raise ValueError(f"--rotate-deg {args.rotate_deg} is meaningless for a "
                         f"{args.cell} cell: rgen draws its own angles and zref has none")
    return Cell(args.arm, args.cell, int(args.step), int(args.seed), int(args.k), deg)


def _cmd_argv(args):
    """Print the canonical check argv (the driver's rendering is pinned to it)."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"argv: {exc}", file=sys.stderr)
        return 2
    print(" ".join(check_argv(cell, args.metrics, pin=args.pin, ckpt_sha=args.ckpt_sha,
                              expected_count=args.expected_count)))
    return 0


def _cmd_check(args):
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"check: {exc}", file=sys.stderr)
        return 2
    if not is_registered(cell):
        print(f"CELL {tuple(cell)} is not registered in the exp_14 grid")
        return 2
    if not os.path.isfile(args.metrics):
        print(f"MISSING {eval_name(cell)}: {args.metrics}")
        return 3
    reasons = validate_cell(args.metrics, cell, pin=args.pin, ckpt_sha=args.ckpt_sha,
                            expected_count=args.expected_count)
    if reasons:
        print(f"INVALID {eval_name(cell)} ({len(reasons)} reason(s)):")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print(f"VALID {eval_name(cell)} ({args.metrics})")
    return 0


def _cmd_classify(args):
    """One line per cell: identity, status, reasons. The wave submitter's input."""
    try:
        expect = load_ckpt_expect(args.ckpt_expect, arms=ARMS)
    except ValueError as exc:
        print(f"classify: {exc}", file=sys.stderr)
        return 2
    ckpts = {}
    for cell in wave_cells(args.wave):
        if cell.arm not in ckpts:
            ckpts[cell.arm] = checkpoint_path(args.output_root, cell.arm, cell.step)
        ckpt = ckpts[cell.arm]
        if ckpt is None:
            print(f"{_fmt_cell(cell)} MISSING no unique step={cell.step} checkpoint under "
                  f"{args.output_root}/exp11_{cell.arm}")
            continue
        path = metrics_path(ckpt, cell)
        if not os.path.isfile(path):
            print(f"{_fmt_cell(cell)} MISSING {path}")
            continue
        reasons = validate_cell(path, cell, pin=args.pin, ckpt_sha=expect[cell.arm],
                                expected_count=args.expected_count)
        if reasons:
            print(f"{_fmt_cell(cell)} INVALID {path} :: " + "; ".join(reasons))
        else:
            print(f"{_fmt_cell(cell)} VALID {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("grid", help="print the registered grid (one cell per line)")
    g.add_argument("--wave", default="all", choices=list(WAVES))
    g.set_defaults(func=_cmd_grid)

    c = sub.add_parser("check", help="validate ONE cell's artifacts")
    c.add_argument("--metrics", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--cell", required=True)
    c.add_argument("--step", required=True)
    c.add_argument("--seed", required=True)
    c.add_argument("--k", required=True)
    c.add_argument("--rotate-deg", default=None)
    c.add_argument("--pin", default=None)
    c.add_argument("--ckpt-sha", default=None)
    c.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    c.set_defaults(func=_cmd_check)

    a = sub.add_parser("argv", help="print the canonical `check` argv for one cell")
    a.add_argument("--metrics", required=True)
    a.add_argument("--arm", required=True)
    a.add_argument("--cell", required=True)
    a.add_argument("--step", required=True)
    a.add_argument("--seed", required=True)
    a.add_argument("--k", required=True)
    a.add_argument("--rotate-deg", default=None)
    a.add_argument("--pin", default=None)
    a.add_argument("--ckpt-sha", default=None)
    a.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    a.set_defaults(func=_cmd_argv)

    cl = sub.add_parser("classify", help="status of every cell in a wave (dedup input)")
    cl.add_argument("--wave", default="all", choices=list(WAVES))
    cl.add_argument("--output-root", required=True)
    cl.add_argument("--pin", default=None)
    cl.add_argument("--ckpt-expect", default=CKPT_EXPECT,
                    help="audited arm -> checkpoint sha256 map (exp14_ckpt_expect.json)")
    cl.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    cl.set_defaults(func=_cmd_classify)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
