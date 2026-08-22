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
      --record-per-scene --record-stream --expected-stream-count 6337 \\
      --seed ${SEED} --eval-name exp21_BFC_S40000_K{8,1}_s${SEED}

The executable form of that command lives in ``exp21_protocol.py``, which builds
it from these constants for every arm; this module never restates a flag.

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
``step=400000`` contains ``step=40000``. A digest is REQUIRED per record (round-5
full review, BLOCKING 2): the path names which file was ASKED FOR, only the
digest names which bytes were LOADED. Identity across the cell -- one path, one
digest -- is a cross-record property and lives in the caller, together with the
five-seed and one-pin rules, next to the table's own KEYS, exactly as exp_15's
split does.

TRAINED-AS. ``cond_method`` records how the EVALUATION conditioned; a record that
says ``fa_cartesian`` proves nothing about the weights, because a vanilla or B-F
checkpoint of the same architecture would load cleanly and produce exactly that
record (round-5 full review, BLOCKING 1). ``eval_FLAC`` now binds the run to the
checkpoint's embedded ``model_config`` and records what it found, so the gate
checks the ARTIFACT and not the claim.

FULL SPLIT. ``n_samples == 6337`` counts what the loop saw, and the loop counts
substitutions too: ``SampleDataset`` silently serves a random OTHER item on a load
or silence failure (dataset.py:342/358), which still increments the counter and
usually leaves the ten family keys intact. The registered command therefore also
passes ``--record-stream --expected-stream-count 6337``, and admission requires
the resulting ``.stream.json`` sidecar -- whose per-position payload lets this
gate RE-RUN eval_FLAC's positional check on durable evidence rather than trust
that it ran (BLOCKING 3).
"""
import hashlib
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
# ...and what the CHECKPOINT must have been trained as. Recorded by eval_FLAC's
# trained-as binding (eval_FLAC.bind_fa_cartesian_checkpoint), which reads the
# config train.py embedded in the checkpoint. Equal to COND_METHOD for this arm
# by construction; they are separate constants because they answer different
# questions, and the whole of BLOCKING 1 is that only one of them was checked.
TRAINED_COND_METHOD = "fa_cartesian"
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
# The digest field, now REQUIRED (round-5 full review, BLOCKING 2). Until this
# round `build_metrics_record` wrote no digest, so the check "slept" and absence
# was accepted -- which meant all ten rows could omit it and pass, or K=1 could
# consistently load bytes A while K=8 loaded bytes B at the same path. Absence
# now blocks. Named here, once, so the driver and the table cannot disagree
# about the spelling (it is `eval_FLAC.build_metrics_record`'s key).
CKPT_SHA_FIELD = "ckpt_sha256"
# Matched with .fullmatch(), never .match() (r5 re-review, BLOCKING 4): Python's
# `$` matches BEFORE a terminal newline, so the old pattern admitted a digest
# with a trailing "\n" -- what a shell capture of `sha256sum` yields verbatim.
CKPT_SHA_RE = re.compile(r"[0-9a-f]{64}")
TRAINED_FIELD = "trained_cond_method"

# --- the full-split proof (r5 review BLOCKING 3) ------------------------------
# `eval_FLAC.stream_sidecar_path`: the metrics file's `.json` becomes
# `.stream.json`. Restated (not imported) because this validator must stay
# importable from a worktree that carries only the worklog -- and pinned equal to
# eval_FLAC's own rule by test, which is what stops the two from drifting.
STREAM_SUFFIX = ".stream.json"
STREAM_SCHEMA_VERSION = 1        # eval_FLAC.STREAM_SCHEMA_VERSION
FINGERPRINT_SCHEMA = 1           # eval_FLAC.CONTEXT_FINGERPRINT_SCHEMA

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


# --- what varies BETWEEN arms, and only that (r5 review BLOCKING 4) -----------
# D6 re-evaluates B-F and P1 at this same pin, so their cells are admitted by
# THIS machinery rather than a second copy of it. Exactly five things differ
# between the three arms; everything else -- the split, the sampling protocol,
# the autocast, the per-scene estimand, the digest, the stream sidecar, the seeds
# and the pin -- is the campaign's, identical by construction. A profile is that
# list of five, so an arm cannot quietly relax anything else.
#
# ``exp21_protocol.py`` owns the arm registry and builds these profiles from it;
# the BFC profile lives here because this module IS the BFC contract.
def profile(eval_prefix, cond_method, trained_cond_method, frame_averaged):
    return {
        "eval_prefix": eval_prefix,
        "cond_method": cond_method,
        "trained_cond_method": trained_cond_method,
        # A vanilla evaluation runs NO orbit. Recording angles or a cap for it
        # would claim a protocol it did not execute, and would make P1's row look
        # protocol-compatible with the frame-averaged arms.
        "frame_avg_angles": FRAME_AVG_ANGLES if frame_averaged else None,
        "frame_avg_fwd_cap": FRAME_AVG_FWD_CAP if frame_averaged else None,
        "orbit_execution": ORBIT_EXECUTION if frame_averaged else "n/a",
    }


BFC_PROFILE = profile("exp21_BFC", COND_METHOD, TRAINED_COND_METHOD, True)


def ckpt_reasons(record, prof=None):
    """``[reason, ...]`` for the checkpoint this record names.

    Identity ACROSS the cell (one path, one digest) is a cross-record property
    and lives with the caller; this is the per-record half: the right step, a
    REQUIRED well-formed digest, and the trained-as receipt.
    """
    prof = prof or BFC_PROFILE
    reasons = []
    try:
        step = parse_ckpt_step(record.get("ckpt_path"))
    except ValueError as exc:
        return [str(exc)]
    if step != STEP:
        reasons.append(f"ckpt_path {record.get('ckpt_path')!r} is at step {step}; this row "
                       f"publishes the step-{STEP} checkpoint only (a substring test would "
                       f"have accepted it: {STEP} is a prefix of {step} for step=400000)")
    sha = record.get(CKPT_SHA_FIELD, None)
    if sha is None:
        reasons.append(
            f"the record carries no {CKPT_SHA_FIELD}: a path proves which file was "
            "NAMED, only a digest proves which bytes were LOADED, and the record "
            "without one is exactly where a substituted or re-trained checkpoint "
            "would hide (eval_FLAC records it for every run)")
    elif not (isinstance(sha, str) and CKPT_SHA_RE.fullmatch(sha)):
        reasons.append(f"{CKPT_SHA_FIELD} {sha!r} is not a lowercase 64-hex digest")

    trained = record.get(TRAINED_FIELD, None)
    if trained is None:
        reasons.append(
            f"the record carries no {TRAINED_FIELD}: cond_method records how this "
            "EVALUATION conditioned, which a vanilla or B-F checkpoint would also "
            "produce under --cond-method fa_cartesian. Its absence means the "
            "trained-as binding did not run, so the arm is unproven (announcement 05)")
    elif trained != prof["trained_cond_method"]:
        reasons.append(
            f"{TRAINED_FIELD} is {trained!r}: these weights were TRAINED under a "
            f"different conditioning than {prof['trained_cond_method']!r}, whatever "
            "the evaluation claims")
    return reasons


def stream_sidecar_path(metrics_path):
    """``<...>_metrics_<stem>.json`` -> ``<...>_metrics_<stem>.stream.json``.

    The same rule as ``eval_FLAC.stream_sidecar_path`` (pinned equal by test).
    """
    path = str(metrics_path)
    base = path[: -len(".json")] if path.endswith(".json") else path
    return base + STREAM_SUFFIX


def _positional_reasons(tuples):
    """Re-run eval_FLAC's substitution guard on the DURABLE payload.

    ``verify_stream_positions`` (eval_FLAC.py:639) raises inside the run, so a
    passing cell leaves no positive trace of it in the metrics record. The
    sidecar, however, carries the whole ordered stream --- each entry is
    ``[position, "<dataset_idx>|<relpath>", [context ids], img_w]`` --- so the
    check is recomputable here, from evidence, by anyone.

    That matters because the event it catches is silent: ``SampleDataset``
    substitutes a RANDOM OTHER item on a load or silence failure, and the
    substituted sample carries its own ``idx``. Position i then holds item j.
    """
    reasons = []
    for i, entry in enumerate(tuples):
        if not (isinstance(entry, list) and len(entry) == 4):
            reasons.append(f"input_tuples[{i}] is {entry!r}, not "
                           "[position, target_id, context_ids, img_w]")
            break
        position, target_id = entry[0], entry[1]
        if position != i:
            reasons.append(f"input_tuples[{i}] is recorded at position {position!r}: "
                           "the stream is not in evaluation order")
            break
        head = str(target_id).split("|", 1)[0]
        try:
            dataset_idx = int(head)
        except (TypeError, ValueError):
            reasons.append(f"input_tuples[{i}] target id {target_id!r} carries no "
                           "dataset index, so it cannot be checked against its "
                           "position")
            break
        if dataset_idx != i:
            reasons.append(
                f"stream position {i} carries dataset idx {dataset_idx} "
                f"(target {target_id!r}): the dataset served a DIFFERENT item than "
                "the sequential sampler asked for -- SampleDataset substitutes a "
                "random item on a load/silence failure, and the substituted sample "
                "still counts toward n_samples")
            break
    return reasons


def read_stream_payload(metrics_path):
    """The sidecar payload beside one metrics JSON, or ``None`` if unusable."""
    path = stream_sidecar_path(metrics_path)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def canonical_input_hash(payload):
    """Recompute the cell's INPUT identity from its durable ``input_tuples``.

    ``eval_FLAC.canonical_stream_hash``'s serialization, restated (and pinned
    equal to it by test): one JSON array per tuple, ``sort_keys=True``,
    ``separators=(",", ":")``, LF-joined, UTF-8.

    Recomputed rather than read out of ``input_hash`` on purpose. The digest in
    the payload is a claim by the writer; the tuples are the preimages, so
    hashing them here is what makes "these two arms saw the same items and the
    same reference draws" checkable by anyone holding the artifacts.

    The input identity is rotation-INDEPENDENT by construction (it carries the
    item and its context sources, never the offset), which is exactly why it is
    the right key for pairing arms.
    """
    tuples = (payload or {}).get("input_tuples")
    if not isinstance(tuples, list):
        return None
    return hashlib.sha256(
        "\n".join(json.dumps(list(t), sort_keys=True, separators=(",", ":"))
                  for t in tuples).encode("utf-8")
    ).hexdigest()


def stream_sidecar_reasons(metrics_path):
    """``[reason, ...]`` for the full-split proof beside one metrics JSON.

    Round-5 full review, BLOCKING 3. ``n_samples == 6337`` is NOT proof that the
    published split was evaluated: a silent per-item substitution keeps the count
    and usually the ten family keys, so the row would be admitted. The registered
    command passes ``--record-stream --expected-stream-count 6337``, and
    ``eval_FLAC`` writes this sidecar ONLY after ``verify_stream_count`` and
    ``verify_stream_positions`` have both passed (eval_FLAC.py:1394-1457) --- so
    its existence is a receipt, and its contents let the receipt be audited.

    Required, not optional: a cell that cannot show it did not prove coverage.
    """
    path = stream_sidecar_path(metrics_path)
    if not os.path.isfile(path):
        return [f"no assignment sidecar at {os.path.basename(path)}: the registered "
                "command passes --record-stream --expected-stream-count "
                f"{EXPECTED_COUNT}, and without the sidecar there is no durable proof "
                "that the full split -- rather than a silently substituted stand-in "
                "for some of it -- was evaluated"]
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        return [f"{os.path.basename(path)} is unreadable ({exc})"]
    if not isinstance(payload, dict):
        return [f"{os.path.basename(path)} is not a stream record object"]

    reasons = []
    if payload.get("schema_version") != STREAM_SCHEMA_VERSION:
        reasons.append(f"stream schema_version is "
                       f"{payload.get('schema_version', '<absent>')!r}, not "
                       f"{STREAM_SCHEMA_VERSION}: a payload written under another "
                       "schema does not mean what this gate reads")
    if payload.get("fingerprint_schema") != FINGERPRINT_SCHEMA:
        reasons.append(f"stream fingerprint_schema is "
                       f"{payload.get('fingerprint_schema', '<absent>')!r}, not "
                       f"{FINGERPRINT_SCHEMA}")
    # the registered cell is the UNROTATED one, in fixed mode
    if payload.get("rotate_mode") != "fixed":
        reasons.append(f"stream rotate_mode is {payload.get('rotate_mode', '<absent>')!r}, "
                       "the registered cell runs in fixed mode")
    if payload.get("rotate_deg") != ROTATE_DEG:
        reasons.append(f"stream rotate_deg is {payload.get('rotate_deg', '<absent>')!r}, "
                       f"the registered cell is the unrotated one ({ROTATE_DEG})")

    tuples = payload.get("input_tuples")
    if not isinstance(tuples, list):
        reasons.append(f"stream input_tuples is {type(tuples).__name__}, not a list: "
                       "without the per-position preimages the coverage claim is "
                       "unfalsifiable")
        return reasons
    # COUNT: three independent statements of the same number must agree, and all
    # three must be the pre-registered split size.
    counts = {"stream_count": payload.get("stream_count"),
              "len(input_tuples)": len(tuples),
              "len(assignment_tuples)": (len(payload["assignment_tuples"])
                                         if isinstance(payload.get("assignment_tuples"), list)
                                         else None),
              "len(offsets)": (len(payload["offsets"])
                               if isinstance(payload.get("offsets"), list) else None)}
    wrong = {name: got for name, got in counts.items() if got != EXPECTED_COUNT}
    if wrong:
        reasons.append(
            f"the assignment stream does not cover the registered split: "
            + ", ".join(f"{name} = {got!r}" for name, got in sorted(wrong.items()))
            + f" (the published unseen split is {EXPECTED_COUNT} items; announcement "
              "01 forbids new or subsampled eval configurations)")
    reasons += _positional_reasons(tuples)
    return reasons

EVAL_NAME_TEMPLATE = "exp21_BFC_S{step}_K{k}_s{seed}"
EVAL_NAME_RE = re.compile(r"^exp21_BFC_S(?P<step>\d+)_K(?P<k>\d+)_s(?P<seed>\d+)$")
#: The same three facts for ANY campaign arm (``exp21_<prefix>_S<step>_K<k>_s<seed>``).
ARM_EVAL_NAME_RE = re.compile(
    r"^(?P<prefix>exp21_[A-Za-z0-9]+)_S(?P<step>\d+)_K(?P<k>\d+)_s(?P<seed>\d+)$")

Cell = namedtuple("Cell", "step k seed")


def parse_arm_eval_name(name, eval_prefix):
    """``Cell`` for one cell of ``eval_prefix``'s row; ``ValueError`` otherwise.

    The arm-generic twin of :func:`parse_eval_name`, and just as strict about the
    three registered facts -- including that the PREFIX is the row's own. A B-F
    re-evaluation renamed into P1's row would otherwise be averaged under P1's
    label, which is the cross-arm version of the exp_14 round-3 B2 lesson.
    """
    match = ARM_EVAL_NAME_RE.match(str(name or ""))
    if match is None or match.group("prefix") != eval_prefix:
        raise ValueError(
            f"eval_name {name!r} is not a registered exp_21 {eval_prefix} cell "
            f"(expected {eval_prefix}_S{STEP}_K<K>_s<seed>)")
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


def validate_metrics_record(record, cell, prof=None):
    """``[reason, ...]`` naming every way this record is not ``cell``'s registered
    measurement. Empty means it is.

    One reason per deviation, each naming the FIELD and the value found: a row
    that renders BLOCKED is only useful if it says which knob to fix.

    ``prof`` selects the ARM (default: BFC, so every existing caller is
    unchanged). The five per-arm values are the only ones it can vary -- see
    :func:`profile` -- so the D6 comparator rows are held to the same campaign in
    everything else.
    """
    if not isinstance(record, dict):
        return ["the metrics record is missing or is not an object"]
    prof = prof or BFC_PROFILE
    reasons = []

    def eq(field, want):
        got = record.get(field, "<absent>")
        if got != want:
            reasons.append(f"{field} is {got!r}, the registered protocol is {want!r}")

    eq("cond_method", prof["cond_method"])
    eq("frame_avg_angles", prof["frame_avg_angles"])
    eq("frame_avg_fwd_cap", prof["frame_avg_fwd_cap"])
    eq("orbit_execution", prof["orbit_execution"])
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

    reasons += ckpt_reasons(record, prof)
    reasons += per_scene_reasons(record)

    if record.get("seed") != cell.seed:
        reasons.append(f"seed is {record.get('seed', '<absent>')!r} but the eval name says "
                       f"{cell.seed}: the file was renamed, or the run was seeded "
                       "differently than it claims")

    want_name = (f"{prof['eval_prefix']}_S{STEP}_K{int(cell.k)}_s{int(cell.seed)}")
    if record.get("eval_name") != want_name:
        reasons.append(f"eval_name is {record.get('eval_name', '<absent>')!r}, not "
                       f"{want_name!r}")

    sha = record.get("source_sha")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        reasons.append(f"source_sha {sha!r} is not a 40-hex evaluator pin (eval_FLAC "
                       "records 'unknown' when git is unavailable; a row whose "
                       "provenance is unknown is not a measured one)")
    return reasons
