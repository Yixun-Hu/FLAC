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

Used by ``gen_model_comparison.validate_exp21_cell``; cell-level rules (five
seeds, one pin, the metrics payload the table prints) live there, next to the
table's own KEYS, exactly as exp_15's split does.
"""
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
CKPT_STEP_TOKEN = f"step={STEP}"

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

    ckpt = record.get("ckpt_path")
    if not isinstance(ckpt, str) or CKPT_STEP_TOKEN not in ckpt:
        reasons.append(f"ckpt_path {ckpt!r} is not the {CKPT_STEP_TOKEN} checkpoint this "
                       "row publishes")

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
