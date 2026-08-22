#!/usr/bin/env python3
"""exp_21 (bf_fa_cartesian): the registered evaluation protocol, stated ONCE.

This module is the admission predicate for a BFC table cell. It exists as its
own file, in this experiment's folder, for the same reason exp_11's, exp_14's
and exp_15's do: the model-comparison generator must not be the place where a
protocol is *defined*, only where it is *applied*, and the eval driver that
produces these cells (a later round) has to agree with the table about what a
registered cell is. One file, imported by both, is how the other three campaigns
guarantee that.

WHAT IS REGISTERED (plan §5, and announcement 05's rule that eval-protocol flags
are part of the experiment, not defaults):

    python eval_FLAC.py --model-config <FLAC_AR_BFC.json> \\
      --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval[_1].json \\
      --ckpt-path <...>/epoch=..-step=40000.ckpt \\
      --cond-method fa_cartesian --frame-avg-angles 0,90,180,270 \\
      --frame-avg-max-fwd-samples 64 --rotate-mode fixed --rotate-deg 0 \\
      --cond-autocast bf16 --batch-size 64 --cfg-scale 1.0 --steps 1 \\
      --record-per-scene --seed ${SEED} --eval-name exp21_BFC_S40000_K{8,1}_s${SEED}

Every constant below is one flag of that command, and the checks are equalities
against the record ``eval_FLAC.build_metrics_record`` writes. A cell that differs
in any of them is a different measurement: `--cond-autocast bf16` versus the CLI
default alone made the same checkpoint read 8.202 or 10.652 on T60 in the exp_10
record, and it is not recoverable after the fact.

TWO CAPS, DELIBERATELY DIFFERENT (announcement 06). ``FRAME_AVG_FWD_CAP`` here is
the EVALUATION cap, 64, which is what a batch of 64 requires (a chunk is whole
angles). The arm's TRAINING cap is 32, declared in FLAC_AR_BFC.json and asserted
by the launcher. Eval mode draws no RoPE rescale, so the eval cap is protocol
identity rather than numerics -- but it is recorded, so it is checked.

TWO ESTIMANDS, BOTH REQUIRED. The table publishes the FLAT split-level metrics
(the comparator estimand: B-F's and P1's rows are flat). The registered command
also passes ``--record-per-scene``, so every cell carries the ten-room-family
block that makes the paper-style per-scene mean computable -- and a row published
without it has silently dropped a deliverable the command produced. The families
are derived from the split file, never typed (see ``EXPECTED_SCENE_KEYS``).

CHECKPOINT IDENTITY. The step is parsed and compared as an integer, because
``step=400000`` contains ``step=40000``. Identity across the cell -- one path,
and one digest once the eval driver records one -- is a cross-record property and
lives in the caller, together with the five-seed and one-pin rules, next to the
table's own KEYS, exactly as exp_15's split does.
"""
import json
import math
import os
import re
from collections import namedtuple

ARM = "BFC"
STEP = 40000                     # the registered endpoint (plan §5)
SEEDS = (42, 43, 44, 45, 46)     # eval seeds = sampling variability only
KS = (1, 8)

# --- the conditioning protocol (this is the arm) -----------------------------
COND_METHOD = "fa_cartesian"
FRAME_AVG_ANGLES = [0.0, 90.0, 180.0, 270.0]
FRAME_AVG_FWD_CAP = 64           # EVAL cap; the arm TRAINS at 32 (see above)
ORBIT_EXECUTION = "batched"
COND_AUTOCAST = "bf16"           # registered table protocol, NOT the CLI default

# --- the sampling / split protocol -------------------------------------------
BATCH_SIZE = 64
CFG_SCALE = 1.0
STEPS = 1
ROTATE_DEG = 0.0                 # the unrotated cell, in FIXED mode
WEIGHTS_SOURCE = "ema"
EXPECTED_COUNT = 6337            # the full published unseen split
DATASET_CONFIG = {
    8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
    1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
}
# --- checkpoint identity (r4 review BLOCKING 3) -------------------------------
# A substring test for "step=40000" accepts `epoch=99-step=400000.ckpt`, i.e. a
# checkpoint ten times further along, published under the 40k row's name. The
# step is therefore PARSED and compared as an integer.
CKPT_STEP_RE = re.compile(r"step=(\d+)")
# The digest field the eval driver will record. `build_metrics_record` carries no
# checkpoint digest today (only `ckpt_path`), so exact PATH uniformity across the
# cell is the floor and this check sleeps until the field exists -- at which
# point it engages on its own, for every cell that carries it. Named here, once,
# so the driver and the table cannot disagree about the spelling.
CKPT_SHA_FIELD = "ckpt_sha256"
CKPT_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

# --- the per-scene (auxiliary) estimand (r4 review BLOCKING 2) -----------------
# The ten AR room families are DERIVED, never typed: `data/AR/unseen_eval.json`
# is a dict keyed by room family, and `AR_md.py:23` sets `md['scene']` to exactly
# that key -- which is what `--record-per-scene` groups on. So the registered
# split file IS the canonical grouping, it travels with the repo, and it agrees
# with the list exp_15 read back from a real committed exp_14 artifact (pinned by
# test). Deriving it fail-closed at import is deliberate: a table gate that
# cannot state the grouping must refuse the row, not guess it.
SPLIT_REL_PATH = os.path.join("data", "AR", "unseen_eval.json")
PER_SCENE_SCHEMA = 1                 # eval_FLAC.PER_SCENE_SCHEMA
# Per family, the ACOUSTIC family is what the per-scene mean is taken over (plan
# §5); FD and retrieval stay split-level. exp_15's ratified list additionally
# names "Invalid T60", which is checked HERE when present but not required, since
# requiring it would be exp_21 inventing a contract for itself.
REQUIRED_SCENE_METRICS = ("T60", "C50", "EDT")
OPTIONAL_SCENE_METRICS = ("Invalid T60",)


def _repo_root(start=None):
    """Marker-walk to the repo root; survives worklog relocations."""
    path = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while not os.path.isdir(os.path.join(path, ".git")):
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("repo root (.git) not found from " + str(start))
        path = parent
    return path


def _derive_scene_keys():
    path = os.path.join(_repo_root(), SPLIT_REL_PATH)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot derive the AR room families from {path} ({exc}): the exp_21 table "
            "gate states the per-scene grouping from the registered split itself and "
            "refuses to fall back on a hand-written list") from exc
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"{path} is not a non-empty scene->items map")
    return tuple(sorted(payload))


EXPECTED_SCENE_KEYS = _derive_scene_keys()
EXPECTED_SCENES = len(EXPECTED_SCENE_KEYS)


def _finite(value):
    """A measured number, not a NaN, an Inf, a bool or a numeric string."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def parse_ckpt_step(ckpt_path):
    """The optimizer step a checkpoint path names; ``ValueError`` if it does not.

    Exactly one ``step=<n>`` token, on a ``.ckpt``: two tokens are ambiguous and
    none is unprovable, and both are more likely to be a path built by hand than
    one PL wrote.
    """
    if not isinstance(ckpt_path, str) or not ckpt_path.endswith(".ckpt"):
        raise ValueError(f"ckpt_path {ckpt_path!r} is not a .ckpt path")
    steps = CKPT_STEP_RE.findall(os.path.basename(ckpt_path))
    if len(steps) != 1:
        raise ValueError(
            f"ckpt_path {ckpt_path!r} names {len(steps)} 'step=<n>' tokens; a registered "
            "cell evaluates one PL checkpoint, whose basename carries exactly one")
    return int(steps[0])


def per_scene_reasons(record):
    """``[reason, ...]`` for the per-scene payload ``--record-per-scene`` writes.

    The flat metrics remain the table estimand; this block is the registered
    AUXILIARY estimand (the paper-style per-scene mean), and a row published
    without it silently drops a deliverable the registered command produced.
    """
    reasons = []
    by_scene = record.get("by_scene")
    if not isinstance(by_scene, dict) or not by_scene:
        return [f"by_scene is {by_scene!r}: the registered command passes "
                "--record-per-scene, so the per-scene payload is part of the record "
                "(plan §3g requires the ten room families)"]
    if record.get("per_scene_schema") != PER_SCENE_SCHEMA:
        reasons.append(f"per_scene_schema is {record.get('per_scene_schema', '<absent>')!r}, "
                       f"not {PER_SCENE_SCHEMA}: a payload written under another schema "
                       "does not mean what this gate reads")
    if record.get("scene_count") != len(by_scene):
        reasons.append(f"scene_count {record.get('scene_count', '<absent>')!r} != the "
                       f"{len(by_scene)} scene(s) actually recorded")
    got = tuple(sorted(by_scene))
    if got != EXPECTED_SCENE_KEYS:
        missing = [k for k in EXPECTED_SCENE_KEYS if k not in by_scene]
        extra = [k for k in got if k not in EXPECTED_SCENE_KEYS]
        reasons.append(
            "by_scene is not the registered AR room-family grouping"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + " -- two different groupings of the same size are the same number of "
              "scenes and a DIFFERENT estimand")
    # Every family's payload must itself be usable evidence: the per-seed
    # observation is the mean OVER these ten values, so one NaN or one missing
    # T60 does not degrade the estimate, it destroys it.
    for scene in sorted(set(EXPECTED_SCENE_KEYS) & set(by_scene)):
        payload = by_scene[scene]
        if not isinstance(payload, dict):
            reasons.append(f"by_scene[{scene!r}] is {payload!r}, not a metric map")
            continue
        for metric in REQUIRED_SCENE_METRICS:
            if metric not in payload:
                reasons.append(f"by_scene[{scene!r}] does not report {metric}")
            elif not _finite(payload[metric]):
                reasons.append(f"by_scene[{scene!r}].{metric} = {payload[metric]!r} is not "
                               "a finite number")
        for metric in OPTIONAL_SCENE_METRICS:
            if metric in payload and not _finite(payload[metric]):
                reasons.append(f"by_scene[{scene!r}].{metric} = {payload[metric]!r} is not "
                               "a finite number")
    return reasons


def ckpt_reasons(record):
    """``[reason, ...]`` for the checkpoint this record names.

    Identity ACROSS the cell (one path, one digest) is a cross-record property
    and lives with the caller; this is the per-record half: the right step, and a
    well-formed digest if the driver recorded one.
    """
    reasons = []
    try:
        step = parse_ckpt_step(record.get("ckpt_path"))
    except ValueError as exc:
        return [str(exc)]
    if step != STEP:
        reasons.append(f"ckpt_path {record.get('ckpt_path')!r} is at step {step}; this row "
                       f"publishes the step-{STEP} checkpoint only (a substring test would "
                       f"have accepted it: {STEP} is a prefix of {step} for step=400000)")
    sha = record.get(CKPT_SHA_FIELD)
    if sha is not None and not (isinstance(sha, str) and CKPT_SHA_RE.match(sha)):
        reasons.append(f"{CKPT_SHA_FIELD} {sha!r} is not a 64-hex digest")
    return reasons

EVAL_NAME_TEMPLATE = "exp21_BFC_S{step}_K{k}_s{seed}"
EVAL_NAME_RE = re.compile(r"^exp21_BFC_S(?P<step>\d+)_K(?P<k>\d+)_s(?P<seed>\d+)$")

Cell = namedtuple("Cell", "step k seed")


def eval_name(k, seed):
    """The ``--eval-name`` of one registered cell.

    ``build_output_paths`` does not add K or the seed itself (eval_FLAC.py:373),
    so the eval name is the ONLY thing separating the ten registered cells'
    artifacts from each other -- and from the invariance grid's, which appends
    ``_rot<deg>``."""
    return EVAL_NAME_TEMPLATE.format(step=STEP, k=int(k), seed=int(seed))


def parse_eval_name(name):
    """``Cell`` for a registered table cell; ``ValueError`` for anything else.

    Deliberately strict about the three registered facts. The §5 invariance grid
    writes ``..._s42_rot45`` cells of the SAME checkpoint and an optional
    band-context screen writes ``S37500``: both are legitimate artifacts of this
    experiment and neither is a model row, so they must fail to parse rather than
    be silently averaged into one.
    """
    match = EVAL_NAME_RE.match(str(name or ""))
    if match is None:
        raise ValueError(
            f"eval_name {name!r} is not a registered exp_21 table cell "
            f"(expected {EVAL_NAME_TEMPLATE.format(step=STEP, k='<K>', seed='<seed>')})"
        )
    step, k, seed = (int(match.group("step")), int(match.group("k")),
                     int(match.group("seed")))
    if step != STEP:
        raise ValueError(f"eval_name {name!r} is at step {step}; this row publishes "
                         f"the step-{STEP} checkpoint only")
    if k not in KS:
        raise ValueError(f"eval_name {name!r} has K={k}; the registered context sizes "
                         f"are {list(KS)}")
    if seed not in SEEDS:
        raise ValueError(f"eval_name {name!r} has seed {seed}, which is not one of the "
                         f"registered eval seeds {list(SEEDS)}")
    return Cell(step, k, seed)


def same_dataset_config(recorded, expected_relative):
    """Is ``recorded`` the registered split for this K?

    An identity, not a spelling: operators paste absolute paths, and the record
    stores the path exactly as given. An absolute path INTO a checkout ends with
    the registered relative path, so it is the same file. The two AR unseen
    configs cannot be confused this way -- ``..._unseeneval_1.json`` does not end
    with ``..._unseeneval.json`` and vice versa -- so a K=1 split can never be
    accepted for a K=8 row.
    """
    if not isinstance(recorded, str):
        return False
    got = os.path.normpath(recorded).replace(os.sep, "/")
    want = os.path.normpath(expected_relative).replace(os.sep, "/")
    return got == want or got.endswith("/" + want)


def validate_metrics_record(record, cell):
    """``[reason, ...]`` naming every way this record is not ``cell``'s registered
    measurement. Empty means it is.

    One reason per deviation, each naming the FIELD and the value found: a row
    that renders BLOCKED is only useful if it says which knob to fix.
    """
    if not isinstance(record, dict):
        return ["the metrics record is missing or is not an object"]
    reasons = []

    def eq(field, want):
        got = record.get(field, "<absent>")
        if got != want:
            reasons.append(f"{field} is {got!r}, the registered protocol is {want!r}")

    eq("cond_method", COND_METHOD)
    eq("frame_avg_angles", FRAME_AVG_ANGLES)
    eq("frame_avg_fwd_cap", FRAME_AVG_FWD_CAP)
    eq("orbit_execution", ORBIT_EXECUTION)
    eq("cond_autocast", COND_AUTOCAST)
    eq("batch_size", BATCH_SIZE)
    eq("n_samples", EXPECTED_COUNT)
    eq("cfg_scale", CFG_SCALE)
    eq("steps", STEPS)
    eq("weights_source", WEIGHTS_SOURCE)

    # The rotation protocol needs both halves: in RANDOM mode eval_FLAC nulls
    # rotate_deg and appends rotate_mode, so checking the angle alone would let a
    # randomly-rotated cell through on `None != 45.0`.
    if record.get("rotate_deg") != ROTATE_DEG:
        reasons.append(f"rotate_deg is {record.get('rotate_deg', '<absent>')!r}, the "
                       f"registered cell is the UNROTATED one ({ROTATE_DEG})")
    if "rotate_mode" in record:
        reasons.append(f"rotate_mode {record['rotate_mode']!r} is recorded: the "
                       "registered cell runs in fixed mode, whose record carries no "
                       "rotation provenance block")

    want_ds = DATASET_CONFIG[int(cell.k)]
    if not same_dataset_config(record.get("dataset_config"), want_ds):
        reasons.append(f"dataset_config is {record.get('dataset_config', '<absent>')!r}, "
                       f"the registered K={cell.k} split is {want_ds!r}")

    reasons += ckpt_reasons(record)
    reasons += per_scene_reasons(record)

    if record.get("seed") != cell.seed:
        reasons.append(f"seed is {record.get('seed', '<absent>')!r} but the eval name says "
                       f"{cell.seed}: the file was renamed, or the run was seeded "
                       "differently than it claims")

    want_name = eval_name(cell.k, cell.seed)
    if record.get("eval_name") != want_name:
        reasons.append(f"eval_name is {record.get('eval_name', '<absent>')!r}, not "
                       f"{want_name!r}")

    sha = record.get("source_sha")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        reasons.append(f"source_sha {sha!r} is not a 40-hex evaluator pin (eval_FLAC "
                       "records 'unknown' when git is unavailable; a row whose "
                       "provenance is unknown is not a measured one)")
    return reasons
