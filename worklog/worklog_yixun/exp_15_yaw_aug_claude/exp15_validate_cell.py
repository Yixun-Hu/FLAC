#!/usr/bin/env python3
"""exp_15 — validate ONE yaw-augmentation eval cell's artifacts (plan §4.1-§4.3, §5).

Adapted from ``exp14_validate_cell.py`` (Yixun-approved plan deviation, worklog
2026-08-15T14:45): exp_14's campaign is review-closed and already implements the
assignment-integrity contract exp_15 adopted verbatim, so the machinery below is
exp_14's with exp_15's grid, arms and protocol substituted. exp_14's own file is
untouched.

A landed cell is three files: the metrics record, its ``.screenmeta.json``
submission manifest and its ``.stream.json`` assignment audit. This module
answers one question about them — *are these the artifacts of the cell the
campaign registered, produced under the protocol it registered?* — and returns a
list of NAMED reasons, empty when the cell is valid.

It is deliberately the same predicate in three places:

* ``yaw_aug_screen.sbatch`` runs it before emitting SCREENRESULT, so a cell that
  cannot be validated never announces a result;
* ``yaw_aug_submit_grid.sh`` runs it for dedup, where the rule is
  validate-BEFORE-skip (exp_14 review B6): a cell is skipped only when its
  artifacts pass every check, and an artifact that exists but fails one HALTS the
  wave for triage rather than being silently skipped or overwritten;
* the collector (round E2) runs it before any contrast.

What it does NOT do is cross-cell reasoning. T<->R pairing, cross-arm rotation
matching and 5/5 seed blocks are equalities BETWEEN cells (plan §4.3, gates
G1-G5) and belong to the collector; claiming them per cell would be claiming
evidence this file cannot see.

**No torch.** The submitter classifies 42 cells on a shared login node, so this
module imports nothing heavier than the standard library. The DEEP checkpoint
admission (embedded step, canonical embedded-config equality, the EMA<->online
mirror, the inventory digest) needs torch and therefore lives in
``exp15_admit_ckpt.py``, which the screen driver runs per cell on the compute
node before it spends the GPU. What this module does is read the two COMMITTED
admission records for the expected sha256 — never a record's own boolean
``checks`` block, which is a claim, not evidence.

Two rules that live in ``eval_FLAC`` are mirrored here — the canonical stream
serialization and the metrics-path shape — exactly as in exp_14.

Usage
-----
    python3 exp15_validate_cell.py grid [--wave {tbl,rrob,vctl,all}]
    python3 exp15_validate_cell.py check --metrics PATH --arm ARM --cell CELL \\
        --step 40000 --seed SEED --k K [--rotate-deg DEG] [--pin SHA] \\
        [--ckpt-sha SHA] [--expected-count 6337]
    python3 exp15_validate_cell.py classify --wave WAVE --output-root DIR \\
        [--pin SHA] [--expected-count 6337]
    python3 exp15_validate_cell.py expect [--arm ARM]

``check`` exits 0 (valid), 1 (invalid — reasons on stdout), 2 (unregistered cell
or usage error) or 3 (the artifact is absent, which is a normal "not run yet").
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import namedtuple

# --- the registered campaign (plan §4.1/§4.2) --------------------------------
# TWO arms, both VANILLA-conditioned. The arm's identity is not an orbit (that is
# exp_11/exp_14's axis) but whether its TRAINING applied the random-yaw
# augmentation — which is why TRAIN_YAW_AUG, not TRAIN_ORBIT, is the map the
# checkpoint gate asserts against the embedded config.
ARMS = ("YAWAUG", "VANL")
TRAIN_YAW_AUG = {"YAWAUG": True, "VANL": False}
CELLS = ("tbl", "rrob", "vctl")
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
STEP = 40000
# The two validity controls, exhaustively, both at s42/K=8 (plan §4.1 block V):
#   VANL   @90 — the POSITIVE control, gate G1: a vanilla model must degrade.
#   YAWAUG @90 — DESCRIPTIVE ONLY (plan §5, review F3). It carries no gate role;
#                gating on it would presuppose the augmentation succeeded.
# No 45-degree control: exp_15 has no group structure for an off-group angle to
# probe (that was exp_14's C4L mechanism control), so it would answer no question
# this campaign asked.
VCTL_TUPLES = (("VANL", 90.0), ("YAWAUG", 90.0))
# Campaign constants. Every cell must report exactly these or it is not a member
# of the block it claims to belong to (plan §4.2; exp_14 §3.2's values, kept
# identical so exp_15's VANL T rows remain comparable to exp_14's VANL Z rows as
# the pre-declared external check).
EXPECTED_COUNT = 6337
BATCH_SIZE = 64
NUM_WORKERS = 4
COND_AUTOCAST = "bf16"
CFG_SCALE = 1.0
STEPS = 1
IMG_W = 512
# BOTH arms are vanilla-conditioned models, so every cell runs --cond-method
# vanilla. --frame-avg-angles is INERT under vanilla but is pinned explicitly on
# the command line per announcement 05 (never rely on a default): eval_FLAC's own
# default is this same C4 list, and a recipe that relied on it would change
# silently when the default did.
COND_METHOD = "vanilla"
FRAME_AVG_ANGLES_FLAG = "0,90,180,270"
# ...and what eval_FLAC RECORDS for it. eval_FLAC.py:1127 writes the angle list
# into the metrics record only for cond_method='fa_invariant' and None otherwise,
# so the flag's value never reaches the record. The manifest therefore carries
# BOTH — the effective value (None, mirroring the record) and the literal flag —
# and this validator checks each against its own source. Asserting the flag
# against the record would fail every cell; asserting only the record would let a
# cell that never pinned the flag pass.
FRAME_AVG_ANGLES_RECORDED = None
# Schema versions this validator understands (eval_FLAC.STREAM_SCHEMA_VERSION /
# CONTEXT_FINGERPRINT_SCHEMA / PER_SCENE_SCHEMA). A bump means the meaning of a
# field changed, so a sidecar written under another version is not comparable and
# is refused.
STREAM_SCHEMA_VERSION = 1
FINGERPRINT_SCHEMA = 1
PER_SCENE_SCHEMA = 1
# WHAT "PER-SCENE" ACTUALLY GROUPS BY (established by exp_14 from its rung-1
# artifact). The released metric callback groups on ``md['scene']``, and
# ``AR_md.py`` sets that to the room FAMILY -- ``rel_path.split('/')[-3]``, e.g.
# "Cafe" -- while the per-room id (``[-2]``) never reaches the callback. So the
# release convention's per-scene mean is a mean over the split's 10 room
# FAMILIES, not its 17 physical rooms; "6,337 items / 17 rooms" describes the
# split's CONTENT, not the grouping.
#
# The key SET is pinned, not just its size: two different ten-family groupings
# would be the same number of scenes and a different estimand.
#
# VERIFIED, not assumed (eval-r1 review finding 1): this key set was read back
# from a real committed exp_14 artifact —
# outputs_FLAC/exp11_VANL/.../epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8_rotrand42_rotrand42.json
# — whose by_scene block has exactly these ten groups and whose scene_count is 10.
EXPECTED_SCENE_KEYS = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe",
                       "ListeningRoom", "LivingRoomsWithHallway", "MeetingRoom",
                       "Office", "Restaurants")
EXPECTED_SCENES = len(EXPECTED_SCENE_KEYS)          # 10

# --- the METRIC SCHEMA the collector (E2) will consume (review finding 1) -----
# "Non-empty dict" was not a validation. A cell whose metrics block was missing
# FD, spelled T60 in lower case, or carried NaN/Inf/True classified VALID and was
# then SKIPPED by the wave submitter's dedup as "already measured" — i.e. the
# fail-open case validate-before-skip exists to prevent, one level down.
#
# The required names are the ones a real exp_14 cell actually carries (same eval
# code path, same metric callback), so this is a readback, not a guess. Extra
# keys are allowed — a future callback may add one — but EVERY value present must
# be a finite real number.
REQUIRED_SPLIT_METRICS = (
    "T60", "Invalid T60", "C50", "EDT", "FD",
    "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10",
    "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10",
)
# Per scene, only the ACOUSTIC family is read (FD and retrieval come from the
# split-level block — plan §13's ratified routing), so these are what a per-scene
# payload must carry to be usable.
#
# `Invalid T60` is in that family BY RATIFICATION (plan §13 names it explicitly),
# and the integrative review caught it missing here: the ten-group mean cannot be
# routed from payloads that were never required to carry it. Verified present in
# all ten groups of the real committed exp_14 artifact before being required.
REQUIRED_SCENE_METRICS = ("T60", "Invalid T60", "C50", "EDT")
SPLIT_K8 = "acousticroom_unseeneval.json"
SPLIT_K1 = "acousticroom_unseeneval_1.json"

HERE = os.path.dirname(os.path.abspath(__file__))
# The two COMMITTED admission sources (plan §5 G4). Neither is recomputed here —
# this module is torch-free — but neither is TRUSTED either: what is read is the
# recorded sha256 and the recorded facts, and exp15_admit_ckpt.py re-derives all
# of them from the checkpoint itself before any GPU is spent.
CONTROL_ADMISSION = os.path.join(HERE, "yaw_aug_control_admission.json")
LAUNCH_REGISTRY = os.path.join(HERE, "yaw_aug_launch_registry.json")
# The arms' model configs, repo-relative. YAWAUG's is exp_15's own; VANL's is
# exp_11's, referenced in place (copying an audited artifact creates a second
# thing that can drift out of agreement with the checkpoint it vouches for).
ARM_CONFIG_REL = {
    "YAWAUG": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
    "VANL": "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json",
}
# Where each arm's 40k checkpoint lives, as a glob relative to an output root.
ARM_CKPT_GLOB = {
    "YAWAUG": ("exp15_YAWAUG", "FLAC_exp15_YAWAUG", "exp15_YAWAUG", "checkpoints"),
    "VANL": ("exp11_VANL", "FLAC_exp11_VANL", "exp11_VANL", "checkpoints"),
}

Cell = namedtuple("Cell", "arm cell step seed k rotate_deg")


def expected_grid():
    """The 42 registered cells: 20 tbl + 20 rrob + 2 vctl, all at STEP."""
    cells = []
    for name in ("tbl", "rrob"):
        for arm in ARMS:
            for k in KS:
                for seed in SEEDS:
                    cells.append(Cell(arm, name, STEP, seed, k, None))
    for arm, deg in VCTL_TUPLES:
        cells.append(Cell(arm, "vctl", STEP, 42, 8, float(deg)))
    return tuple(cells)


_GRID = expected_grid()
_GRID_SET = frozenset(_GRID)
GRID_SIZE = 42

WAVES = ("vctl", "tbl", "rrob", "all")


def wave_cells(wave):
    """The cells of one submission wave (plan §7-8: vctl first, then tbl, then rrob)."""
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
    if cell.cell == "rrob":
        return f"_rotrand{int(cell.seed)}"
    if cell.cell == "vctl":
        return f"_rot{fmt_deg(cell.rotate_deg)}"
    return ""


def eval_name(cell):
    """``exp15_<ARM>_<CELL>[_rot...]_S<STEP>_s<SEED>_K<K>`` — injective over the grid.

    Note the token's POSITION: exp_15's registered naming (plan §6.7) puts it
    directly after the cell type, where exp_14 put it at the end. Both are
    injective; this module is the one definition of exp_15's spelling and every
    other file renders it from here or is pinned against it by a guard case.
    """
    return (f"exp15_{cell.arm}_{cell.cell}{rotation_suffix(cell)}"
            f"_S{int(cell.step)}_s{int(cell.seed)}_K{int(cell.k)}")


def job_name(cell):
    """The Slurm job name for one cell — injective over the grid.

    The rotation token is load-bearing here, not decoration: VANL vctl@90 and a
    hypothetical second VANL fixed-angle control would differ in nothing else, and
    more immediately a wave watching squeue must not read one cell as another's
    in-flight job (exp_14 review B2). Rendered with '-' separators because a job
    name is also a file name (slurm_screen_%x_%j.out); the character set is
    [A-Za-z0-9._-].
    """
    token = rotation_suffix(cell).replace("_", "-")     # _rot90 -> -rot90
    return (f"exp15-screen-{cell.arm}-{cell.cell}{token}"
            f"-{int(cell.step)}-s{int(cell.seed)}-K{int(cell.k)}")


def parse_eval_name(name):
    """Inverse of :func:`eval_name`, restricted to REGISTERED cells.

    A name that parses but names no registered cell is rejected: the point of a
    registered grid is that an unregistered artifact cannot be read as evidence.
    """
    parts = name.split("_")
    if len(parts) < 6 or parts[0] != "exp15":
        raise ValueError(f"{name!r} is not an exp_15 eval name")
    arm, cellname = parts[1], parts[2]
    if cellname not in CELLS:
        raise ValueError(f"{name!r} names no exp_15 cell type")
    tail = parts[3:]
    token = ""
    if tail and not tail[0].startswith("S"):
        token, tail = tail[0], tail[1:]
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
        if not token.startswith("rot") or token.startswith("rotrand"):
            raise ValueError(f"{name!r} is a vctl cell without a fixed-angle token")
        try:
            deg = float(token[len("rot"):].replace("p", "."))
        except ValueError:
            raise ValueError(f"{name!r} carries an unreadable rotation token {token!r}")
    cell = Cell(arm, cellname, step, seed, k, deg)
    if eval_name(cell) != name:
        raise ValueError(f"{name!r} is not the registered name of the cell it describes "
                         f"(expected {eval_name(cell)!r})")
    if not is_registered(cell):
        raise ValueError(f"{name!r} names an UNREGISTERED cell {tuple(cell)}")
    return cell


# --- rules mirrored from eval_FLAC (pinned by guard cases) -------------------
def canonical_stream_hash(tuples):
    """sha256 over an ordered tuple stream, in the pre-registered serialization.

    Mirror of ``eval_FLAC.canonical_stream_hash`` (plan §4.3): one JSON array per
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
    """Both exp_15 arms are VANILLA models; the arm axis is training augmentation.

    Kept as a function (rather than inlining the constant) so the driver, the
    intent contract and this validator go on having exactly one definition of it —
    and so a future third arm cannot silently inherit 'vanilla'.
    """
    if arm not in TRAIN_YAW_AUG:
        raise ValueError(f"{arm!r} is not an exp_15 arm")
    return COND_METHOD


def metrics_path(ckpt_path, cell):
    """Where eval_FLAC writes this cell's metrics JSON (mirror of build_output_paths).

    The rotation suffix appears TWICE in the stem — once inside the eval name we
    chose and once appended by eval_FLAC itself — which is exp_14's observed
    behaviour on real artifacts and is deliberately reproduced rather than
    "fixed": the path has to be whatever eval_FLAC actually writes.
    """
    ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", "")
    directory = os.path.dirname(ckpt_path)
    # method_suffix is empty for every exp_15 cell (vanilla); spelled out rather
    # than dropped so the mirror of build_output_paths stays readable as a mirror.
    method_suffix = "" if cond_method(cell.arm) == "vanilla" else "_unreachable"
    stem = f"{STEPS}_{CFG_SCALE}_{eval_name(cell)}{method_suffix}{rotation_suffix(cell)}"
    return os.path.join(directory, f"{ckpt_name}_metrics_{stem}.json")


def stream_path(metrics):
    base = metrics[:-len(".json")] if metrics.endswith(".json") else metrics
    return base + ".stream.json"


def screenmeta_path(metrics):
    return metrics + ".screenmeta.json"


# --- checkpoint admission expectation (plan §5 G4) ---------------------------
def _read_json_strict(path, label):
    if not os.path.isfile(path):
        raise ValueError(f"{label} missing: {path}")
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{label} {path} could not be parsed: {exc}")
    if not isinstance(obj, dict):
        raise ValueError(f"{label} {path} is not a JSON object at the top level")
    return obj


def _sha256_or_raise(value, what, source):
    if not (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{what} in {source} is {value!r}, not a lowercase sha256 digest")
    return value


def _vanl_expectation(path=CONTROL_ADMISSION, step=STEP):
    """VANL's admission facts, from the committed control-admission record.

    The record's own ``checks`` block is deliberately NOT read: it is the
    recorder's claim that it validated the file, and a claim is not evidence.
    What is taken is the MEASUREMENTS — sha256, size, embedded step, canonical
    embedded-config digest, the EMA inventory digest — every one of which
    exp15_admit_ckpt.py re-derives from the checkpoint itself.
    """
    rec = _read_json_strict(path, "VANL control-admission record")
    ckpt = rec.get("checkpoint")
    cfg = rec.get("config")
    meta = rec.get("_meta") or {}
    if not isinstance(ckpt, dict) or not isinstance(cfg, dict):
        raise ValueError(f"{path} has no checkpoint/config block")
    if int(meta.get("expect_step", -1)) != int(step):
        raise ValueError(f"{path} was recorded for step {meta.get('expect_step')!r}, "
                         f"not exp_15's endpoint {step}")
    if int(ckpt.get("global_step", -1)) != int(step):
        raise ValueError(f"{path} records global_step {ckpt.get('global_step')!r}, "
                         f"not exp_15's endpoint {step}")
    xref = rec.get("exp_11_cross_references") or {}
    ckpt_rel = ckpt.get("path")
    if not isinstance(ckpt_rel, str) or not ckpt_rel:
        raise ValueError(f"{path} records no checkpoint path")
    return {
        "arm": "VANL",
        "source": os.path.basename(path),
        "sha256": _sha256_or_raise(ckpt.get("sha256"), "checkpoint sha256", path),
        "bytes": int(ckpt["bytes"]) if isinstance(ckpt.get("bytes"), int) else None,
        "step": int(step),
        # WHERE the file must live and WHICH recorded launch produced it. A sha256
        # says two files are the same bytes; it does not say the bytes came from a
        # run this campaign registered, and a checkpoint dropped into the tree by
        # hand would otherwise satisfy every other check.
        "ckpt_path_rel": ckpt_rel,
        "manifest_sha256": _sha256_or_raise(
            xref.get("manifest_sha256"),
            "exp_11_cross_references.manifest_sha256", path),
        "embedded_config_canonical_sha256": _sha256_or_raise(
            ckpt.get("embedded_config_canonical_sha256"),
            "embedded_config_canonical_sha256", path),
        "config_canonical_sha256": _sha256_or_raise(
            cfg.get("canonical_sha256"), "config canonical_sha256", path),
        "config_sha256": _sha256_or_raise(cfg.get("sha256"), "config sha256", path),
        "config_rel": ARM_CONFIG_REL["VANL"],
        "ema_inventory_sha256": _sha256_or_raise(
            ckpt.get("ema_inventory_sha256"), "ema_inventory_sha256", path),
        "ema_key_count": ckpt.get("ema_key_count"),
        "online_model_key_count": ckpt.get("online_model_key_count"),
    }


def _yawaug_expectation(path=LAUNCH_REGISTRY, step=STEP):
    """YAWAUG's admission facts, from the chain's committed launch registry.

    ``final_ckpt_sha256`` is written by the chain's FINAL leg. Until 40k lands it
    is ``null``, and this raises with that fact stated plainly — a cell must not
    be admitted, skipped as "already valid", or evaluated against an expectation
    that does not exist yet.

    Two independent records of the same checkpoint are cross-checked: the arm
    entry's ``final_ckpt_sha256``/``final_step`` and the leg list's step-40000
    entry (with its own audit block). Disagreement is fatal — it means the
    registry describes two different files.
    """
    reg = _read_json_strict(path, "YAWAUG launch registry")
    arm = ((reg.get("arms") or {}).get("YAWAUG"))
    if not isinstance(arm, dict):
        raise ValueError(f"{path} has no arms.YAWAUG entry")
    final_sha = arm.get("final_ckpt_sha256")
    final_step = arm.get("final_step")
    if final_sha is None or final_step is None:
        raise ValueError(
            f"{path}: arms.YAWAUG.final_ckpt_sha256={final_sha!r} / final_step="
            f"{final_step!r} — the training chain has NOT recorded its final "
            f"checkpoint yet (legs recorded so far: "
            f"{sorted(int(l.get('step', -1)) for l in (reg.get('legs') or {}).get('YAWAUG') or [])}). "
            "No YAWAUG eval cell can be admitted, classified or skipped until the "
            "chain completes 40,000 steps and its completion gate backfills that "
            "field. This is the pre-registered fail-closed state, not an error to "
            "work around.")
    if int(final_step) != int(step):
        raise ValueError(f"{path}: arms.YAWAUG.final_step {final_step!r} is not exp_15's "
                         f"pre-registered endpoint {step}")
    legs = (reg.get("legs") or {}).get("YAWAUG") or []
    hits = [l for l in legs if isinstance(l, dict) and int(l.get("step", -1)) == int(step)]
    if len(hits) != 1:
        raise ValueError(f"{path}: expected exactly one legs.YAWAUG entry at step {step}, "
                         f"found {len(hits)}")
    leg = hits[0]
    audit = leg.get("audit") or {}
    leg_sha = _sha256_or_raise(leg.get("ckpt_sha256"), f"legs.YAWAUG[step={step}].ckpt_sha256", path)
    final_sha = _sha256_or_raise(final_sha, "arms.YAWAUG.final_ckpt_sha256", path)
    if leg_sha != final_sha:
        raise ValueError(f"{path}: arms.YAWAUG.final_ckpt_sha256 {final_sha[:12]} != the "
                         f"step-{step} leg's ckpt_sha256 {leg_sha[:12]} — the registry "
                         "describes two different checkpoints")
    leg_path = leg.get("ckpt_path")
    if not isinstance(leg_path, str) or not leg_path:
        raise ValueError(f"{path}: legs.YAWAUG[step={step}] records no ckpt_path")
    return {
        "arm": "YAWAUG",
        "source": os.path.basename(path),
        "sha256": final_sha,
        "bytes": int(leg["ckpt_bytes"]) if isinstance(leg.get("ckpt_bytes"), int) else None,
        "step": int(step),
        # See the VANL branch: identity of the bytes is not provenance of the run.
        "ckpt_path_rel": leg_path,
        "manifest_sha256": _sha256_or_raise(arm.get("manifest_sha256"),
                                            "arms.YAWAUG.manifest_sha256", path),
        "embedded_config_canonical_sha256": _sha256_or_raise(
            audit.get("embedded_config_canonical_sha256"),
            f"legs.YAWAUG[step={step}].audit.embedded_config_canonical_sha256", path),
        # The registry pins the config by RAW file bytes; its canonical form is
        # what the checkpoint embeds, so the two are recorded separately and
        # exp15_admit_ckpt.py checks the file against both.
        "config_canonical_sha256": _sha256_or_raise(
            audit.get("embedded_config_canonical_sha256"),
            f"legs.YAWAUG[step={step}].audit.embedded_config_canonical_sha256", path),
        "config_sha256": _sha256_or_raise(arm.get("config_sha256"),
                                          "arms.YAWAUG.config_sha256", path),
        "config_rel": ARM_CONFIG_REL["YAWAUG"],
        "ema_inventory_sha256": _sha256_or_raise(
            audit.get("ema_inventory_sha256"),
            f"legs.YAWAUG[step={step}].audit.ema_inventory_sha256", path),
        "ema_key_count": audit.get("ema_key_count"),
        "online_model_key_count": audit.get("online_model_key_count"),
    }


def admission_expectation(arm, control=CONTROL_ADMISSION, registry=LAUNCH_REGISTRY,
                          step=STEP):
    """Every admission fact recorded for ``arm``; raises (never guesses) otherwise."""
    if arm == "VANL":
        return _vanl_expectation(control, step)
    if arm == "YAWAUG":
        return _yawaug_expectation(registry, step)
    raise ValueError(f"{arm!r} is not an exp_15 arm")


def load_ckpt_expect(control=CONTROL_ADMISSION, registry=LAUNCH_REGISTRY,
                     arms=ARMS, step=STEP):
    """``arm -> sha256`` from the committed records; anything missing is fatal.

    Fail-closed by construction (exp_14 review B4): a dedup SKIP is a decision to
    keep a number without re-measuring it, so it must rest on WHICH checkpoint
    produced that number. An absent or incomplete record therefore raises rather
    than letting the check quietly not run — which, today, is exactly what happens
    for YAWAUG while the chain is still running.
    """
    return {arm: admission_expectation(arm, control, registry, step)["sha256"]
            for arm in arms}


def checkpoint_path(output_root, arm, step=STEP):
    """The arm's single checkpoint at ``step``, or None if it is not unique."""
    if arm not in ARM_CKPT_GLOB:
        raise ValueError(f"{arm!r} is not an exp_15 arm")
    pat = os.path.join(output_root, *ARM_CKPT_GLOB[arm], f"epoch=*-step={step}.ckpt")
    hits = sorted(glob.glob(pat))
    return hits[0] if len(hits) == 1 else None


# --- the expectations one cell must meet -------------------------------------
def rotation_expectation(cell):
    """``(rotate_mode, rotate_deg, rotate_seed)`` for a registered cell.

    The angle and the seed are mutually exclusive by design: a fixed cell has an
    angle and no seed, a random cell has a seed and a NULL angle. Recording 0.0
    for a random cell would make it indistinguishable from an unrotated one.
    """
    if cell.cell == "rrob":
        return "random", None, int(cell.seed)
    if cell.cell == "vctl":
        return "fixed", float(cell.rotate_deg), None
    return "fixed", 0.0, None


def _eq(reasons, label, got, want):
    if got != want:
        reasons.append(f"{label} {got!r} != expected {want!r}")


def is_finite_real(value):
    """A metric value must be a FINITE REAL number — and ``bool`` is not one.

    ``isinstance(True, int)`` is True in Python, so a naive numeric check admits
    ``True``/``False`` as 1/0; ``float('nan')`` and ``float('inf')`` are floats
    that pass every isinstance test and then poison a mean, a paired difference
    and a t-statistic silently. Both are excluded explicitly.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _metric_block_reasons(block, label, required):
    """Named reasons a metric mapping is not usable evidence ([] = usable)."""
    reasons = []
    if not isinstance(block, dict) or not block:
        return [f"{label} is missing or empty "
                f"({type(block).__name__ if block is not None else None})"]
    missing = [m for m in required if m not in block]
    if missing:
        # Named individually: a wrong-cased 't60' shows up here as a MISSING
        # 'T60', which is the failure a reader needs to see.
        reasons.append(f"{label} is missing required metric(s) {missing} "
                       f"(present: {sorted(block)})")
    bad = sorted(k for k, v in block.items() if not is_finite_real(v))
    if bad:
        shown = {k: block[k] for k in bad[:4]}
        reasons.append(f"{label} has {len(bad)} non-finite/non-numeric value(s) "
                       f"{bad}: {shown} — a metric that is NaN, Inf, a bool or a "
                       "string is not a measurement")
    return reasons


def validate_metrics_record(rec, cell, pin=None, expected_count=EXPECTED_COUNT,
                            expected_scenes=EXPECTED_SCENES,
                            expected_keys=EXPECTED_SCENE_KEYS):
    """Named reasons why ``rec`` is not this cell's metrics record ([] = valid)."""
    reasons = []
    mode, deg, rseed = rotation_expectation(cell)
    reasons += _metric_block_reasons(rec.get("metrics"), "metrics block",
                                     REQUIRED_SPLIT_METRICS)
    _eq(reasons, "eval_name", rec.get("eval_name"), eval_name(cell))
    _eq(reasons, "seed", rec.get("seed"), int(cell.seed))
    _eq(reasons, "cfg_scale", rec.get("cfg_scale"), CFG_SCALE)
    _eq(reasons, "steps", rec.get("steps"), STEPS)
    _eq(reasons, "cond_autocast", rec.get("cond_autocast"), COND_AUTOCAST)
    _eq(reasons, "batch_size", rec.get("batch_size"), BATCH_SIZE)
    _eq(reasons, "cond_method", rec.get("cond_method"), cond_method(cell.arm))
    # eval_FLAC records the angle list only under fa_invariant; both exp_15 arms
    # are vanilla, so the record must carry None here whatever the (pinned, inert)
    # flag said. The flag itself is checked against the MANIFEST — see
    # validate_screenmeta — because that is the only artifact that witnesses it.
    _eq(reasons, "frame_avg_angles", rec.get("frame_avg_angles"), FRAME_AVG_ANGLES_RECORDED)
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
    # Fixed-mode records are FROZEN (exp_14 review B4): eval_FLAC adds the
    # random-mode provenance keys only in random mode, so their presence on a
    # tbl/vctl cell means it ran a protocol other than the one it claims. This
    # holds even though exp_15's tbl cells pass --rotate-mode fixed --rotate-deg 0
    # EXPLICITLY: those flag values are eval_FLAC's own defaults, so the record is
    # byte-identical to one written with no rotation flags at all.
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
    reasons += _per_scene_reasons(rec, expected_scenes, expected_keys)
    return reasons


def _per_scene_reasons(rec, expected_scenes=EXPECTED_SCENES,
                       expected_keys=EXPECTED_SCENE_KEYS):
    """Named reasons the per-scene block is not this campaign's (plan §5).

    Required, with NO fallback. The estimand is the per-scene mean, so a cell
    that did not record `by_scene` did not measure it — and a collector that
    silently averaged over items instead would answer a different question in the
    same table cell.
    """
    reasons = []
    by_scene = rec.get("by_scene")
    if not isinstance(by_scene, dict) or not by_scene:
        return [f"by_scene {type(by_scene).__name__ if by_scene is not None else None} "
                "is missing or empty: this campaign's observation is the PER-SCENE "
                "mean, so a cell without per-scene results did not measure the "
                "estimand (pass --record-per-scene)"]
    if rec.get("per_scene_schema") != PER_SCENE_SCHEMA:
        reasons.append(f"per_scene_schema {rec.get('per_scene_schema')!r} != "
                       f"{PER_SCENE_SCHEMA} — written under another contract")
    if expected_keys is not None:
        got, want = set(by_scene), set(expected_keys)
        if got != want:
            missing, extra = sorted(want - got), sorted(got - want)
            reasons.append(
                f"by_scene groups {sorted(got)}, not the release grouping's "
                f"{sorted(want)}"
                + (f" (missing {missing})" if missing else "")
                + (f" (unexpected {extra})" if extra else "")
                + ": a mean over a different grouping is a different estimand")
    elif len(by_scene) != int(expected_scenes):
        reasons.append(f"by_scene covers {len(by_scene)} scene(s), not the expected "
                       f"{int(expected_scenes)}: a per-scene mean over a different "
                       "grouping is a different number")
    if rec.get("scene_count") != len(by_scene):
        reasons.append(f"scene_count {rec.get('scene_count')!r} != the {len(by_scene)} "
                       "scene(s) actually recorded")
    # Every scene's payload must itself be usable evidence: the per-seed
    # observation for the acoustic family is the mean OVER these ten values, so
    # one NaN or one missing T60 does not degrade the estimate, it destroys it.
    for scene, payload in sorted(by_scene.items()):
        scene_reasons = _metric_block_reasons(payload, f"by_scene[{scene!r}]",
                                              REQUIRED_SCENE_METRICS)
        if scene_reasons:
            # Report the FIRST offending scene in full rather than ten copies of
            # the same sentence; the cell is refused either way.
            reasons += scene_reasons
            break
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
    # BOTH spellings of the frame-average angles, from their two different
    # sources: the effective value eval_FLAC recorded (None under vanilla) and the
    # LITERAL flag the driver pinned (announcement 05). A manifest that carried
    # only the first could not witness that the flag was ever passed.
    _eq(reasons, "screenmeta frame_avg_angles", meta.get("frame_avg_angles"),
        FRAME_AVG_ANGLES_RECORDED)
    _eq(reasons, "screenmeta frame_avg_angles_flag", meta.get("frame_avg_angles_flag"),
        FRAME_AVG_ANGLES_FLAG)
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
    _eq(reasons, "screenmeta record_per_scene", meta.get("record_per_scene"), True)
    _eq(reasons, "screenmeta use_ema", meta.get("use_ema"), True)
    # exp_15's arm axis: what the TRAINING did. exp_14 recorded training/eval
    # orbit here; exp_15 has no orbit, and recording one would invite a reader to
    # compare two campaigns' rows on a field that means different things.
    _eq(reasons, "screenmeta train_yaw_aug", meta.get("train_yaw_aug"),
        TRAIN_YAW_AUG[cell.arm])
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


def validate_payloads(record, meta, stream, cell, pin=None, ckpt_sha=None,
                      expected_count=EXPECTED_COUNT, expected_scenes=EXPECTED_SCENES,
                      expected_keys=EXPECTED_SCENE_KEYS):
    """Every named reason these ALREADY-PARSED payloads are not this cell's.

    This is the core; :func:`validate_cell` is the path-reading wrapper around it.
    The split exists because validating one snapshot and aggregating a different
    one is a real failure mode (eval-r2 review finding 5): the collector parses
    the three artifacts once, keeps those objects, and must validate EXACTLY the
    objects it will go on to average. A path-based validator re-reads the files,
    so a concurrent replacement could be validated as version B while version A's
    numbers are the ones that reach the table.

    ``None`` for a payload means "that artifact was absent", which is named
    rather than skipped.
    """
    if not is_registered(cell):
        raise ValueError(f"cell {tuple(cell)} is not registered in the exp_15 grid")
    # A VALID verdict must mean every CAMPAIGN check ran, not that some of them
    # had no input (exp_14 review B6). The pin and the checkpoint digest are the
    # two that a caller could previously omit and still be told the cell is valid.
    reasons = []
    if pin is None:
        reasons.append("campaign pin not supplied: a cell cannot be declared valid "
                       "without checking which commit produced it")
    if ckpt_sha is None:
        reasons.append("expected ckpt sha256 not supplied: a cell cannot be declared "
                       "valid without checking WHICH checkpoint it evaluated")
    if not isinstance(record, dict):
        return reasons + ["metrics payload is absent or is not a JSON object"]
    reasons += validate_metrics_record(record, cell, pin=pin,
                                       expected_count=expected_count,
                                       expected_scenes=expected_scenes,
                                       expected_keys=expected_keys)
    if meta is None:
        reasons.append("screenmeta payload is absent")
    elif not isinstance(meta, dict):
        reasons.append("screenmeta payload is not a JSON object")
    else:
        reasons += validate_screenmeta(meta, cell, pin=pin, ckpt_sha=ckpt_sha,
                                       expected_count=expected_count)
    if stream is None:
        reasons.append("stream payload is absent (--record-stream is mandatory for "
                       "every exp_15 cell)")
    elif not isinstance(stream, dict):
        reasons.append("stream payload is not a JSON object")
    else:
        reasons += validate_stream_record(stream, cell, expected_count=expected_count,
                                          record=record)
    return reasons


def validate_cell(metrics, cell, pin=None, ckpt_sha=None, expected_count=EXPECTED_COUNT,
                  expected_scenes=EXPECTED_SCENES, expected_keys=EXPECTED_SCENE_KEYS):
    """Read this cell's three artifacts from disk and validate them.

    The entry point the screen driver and the wave submitter use, where reading
    from paths IS the job. It reads once and hands the parsed payloads to
    :func:`validate_payloads`, so both callers run the same checks over one
    definition of them.

    Raises ``ValueError`` for an UNREGISTERED cell before touching the filesystem:
    an artifact for a cell the plan never registered is not a validation failure,
    it is a question that should not have been asked.
    """
    if not is_registered(cell):
        raise ValueError(f"cell {tuple(cell)} is not registered in the exp_15 grid")
    if not os.path.isfile(metrics):
        return [f"metrics artifact missing: {metrics}"]
    rec, bad = _read_record(metrics, "metrics")
    if rec is None:
        return bad
    prefix = []
    meta = stream = None
    meta_p, stream_p = screenmeta_path(metrics), stream_path(metrics)
    if not os.path.isfile(meta_p):
        prefix.append(f"screenmeta sidecar missing: {meta_p}")
    else:
        meta, bad = _read_record(meta_p, "screenmeta")
        prefix += bad
    if not os.path.isfile(stream_p):
        prefix.append(f"stream sidecar missing: {stream_p} (--record-stream is "
                      "mandatory for every exp_15 cell)")
    else:
        stream, bad = _read_record(stream_p, "stream")
        prefix += bad
    reasons = validate_payloads(rec, meta, stream, cell, pin=pin, ckpt_sha=ckpt_sha,
                                expected_count=expected_count,
                                expected_scenes=expected_scenes,
                                expected_keys=expected_keys)
    # A sidecar that was MISSING on disk is reported with its path (useful for an
    # operator); validate_payloads only knows it was absent, so drop its
    # path-free duplicate of the same fact.
    generic = {"screenmeta payload is absent",
               "stream payload is absent (--record-stream is mandatory for every "
               "exp_15 cell)"}
    return prefix + [r for r in reasons if r not in generic]


# --- CLI ---------------------------------------------------------------------
def parse_deg(value):
    """Tolerant reading of a rotation angle from a COMMAND LINE: float or None.

    The shell has no null. A caller that means "no angle" can only say so by
    omitting the flag, or by passing 0, "0", "0.0" or "" — and a validator that
    treated the string "0" as "an angle was requested" rejected 100 of exp_14's
    106 cells AFTER they had spent their GPU (exp_14 review B1). Everything that
    spells "nothing" therefore reads as None here, and everything else must parse
    as a float or it is a hard usage error.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "none", "null", "-"):
        return None
    deg = float(text)                      # ValueError -> caller reports usage
    return None if deg == 0.0 else deg


def check_argv(cell, metrics, pin=None, ckpt_sha=None, expected_count=EXPECTED_COUNT,
               expected_scenes=EXPECTED_SCENES):
    """The canonical ``check`` argv for one cell — the ONE definition of it.

    The screen driver renders these flags in shell (it already knows every value
    and a python round-trip would cost a process); a guard case pins its rendering
    against this function, so the two cannot drift. Note what is NOT here: a
    ``--rotate-deg`` outside vctl. rrob draws its angles and tbl's angle is zero,
    so passing 0 would be a claim about a protocol neither of them ran.
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
    argv += ["--expected-count", str(int(expected_count)),
             "--expected-scenes", str(int(expected_scenes))]
    return argv


def eval_argv_tail(cell):
    """The PROTOCOL half of the eval_FLAC argv for one cell (plan §4.2, literal).

    Rendered here so the screen driver, the pre-launch intent and the guard suite
    all read ONE definition of the per-class flags. The identity half
    (--model-config/--ckpt-path/--eval-name/...) stays in the driver, which is
    what knows the paths.

    Announcement 05 in one line: every value below is passed explicitly, including
    the ones that equal an argparse default, because a recipe that relies on a
    default changes silently when the default does.
    """
    mode, deg, rseed = rotation_expectation(cell)
    argv = ["--cond-method", COND_METHOD,
            "--frame-avg-angles", FRAME_AVG_ANGLES_FLAG,
            "--cond-autocast", COND_AUTOCAST,
            "--rotate-mode", mode]
    if mode == "random":
        # --rotate-seed is legal ONLY in random mode (eval_FLAC makes passing it in
        # fixed mode a hard error, never a silent no-op), and the angle stays 0:
        # a random cell draws per sample, so a non-zero --rotate-deg is refused by
        # eval_FLAC as a contradiction.
        argv += ["--rotate-seed", str(int(rseed)), "--rotate-deg", "0"]
    else:
        argv += ["--rotate-deg", fmt_deg(deg)]
    return argv


def _fmt_angle(deg):
    """``90.0 -> '90'``, ``11.25 -> '11.25'`` — the flag's own spelling."""
    d = float(deg)
    return str(int(d)) if d.is_integer() else repr(d)


def contract_lines(cell):
    """The PROTOCOL CONTRACT for one cell, as the pre-launch intent records it.

    Announcement 05: the eval-protocol flags are part of the experiment, and a
    launch record has to be readable for protocol compliance on its own. The
    post-run ``.screenmeta.json`` cannot repair an incomplete pre-launch intent —
    it is written by the job that already ran under whatever flags it got.

    Rendered HERE so the intent, the screen driver's argv and this validator's
    admissibility rules are one definition rather than three that agree today.
    """
    mode, deg, rseed = rotation_expectation(cell)
    split = SPLIT_K8 if int(cell.k) == 8 else SPLIT_K1
    return [
        f"cond_method {COND_METHOD}",
        # The LITERAL flag value, not the recorded one: this line documents what
        # the command line will carry. It is inert under vanilla and pinned anyway.
        f"frame_avg_angles {FRAME_AVG_ANGLES_FLAG} (inert under vanilla; pinned per announcement 05)",
        f"train_yaw_aug {'yes' if TRAIN_YAW_AUG[cell.arm] else 'no'}",
        f"rotate_mode {mode} rotate_seed "
        + (str(int(rseed)) if rseed is not None else "<n/a>")
        + " rotate_deg " + ("<null>" if deg is None else fmt_deg(deg)),
        f"split {split} expected_stream_count {EXPECTED_COUNT} record_stream yes",
        f"record_per_scene yes expected_scenes {EXPECTED_SCENES}"
        " (release grouping: AR room families)",
        f"batch_size {BATCH_SIZE} num_workers {NUM_WORKERS}",
        f"cond_autocast {COND_AUTOCAST} cfg_scale {CFG_SCALE} steps {STEPS} use_ema yes",
        "eval_argv_protocol " + " ".join(eval_argv_tail(cell)),
    ]


def _cmd_contract(args):
    """Print the cell's protocol contract (the submitter writes it into the intent)."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"contract: {exc}", file=sys.stderr)
        return 2
    if not is_registered(cell):
        print(f"contract: {tuple(cell)} is not registered in the exp_15 grid",
              file=sys.stderr)
        return 2
    for line in contract_lines(cell):
        print(line)
    return 0


def _fmt_cell(cell):
    rot = "-" if cell.rotate_deg is None else fmt_deg(cell.rotate_deg)
    return f"{cell.arm} {cell.cell} {cell.step} {cell.seed} {cell.k} {rot}"


def _cmd_grid(args):
    """One cell per line. ``--with-jobname`` appends the canonical Slurm job name.

    The wave submitter needs a job name per cell to read squeue, and rendering
    one python process per cell for that would cost 42 of them. Emitting it as a
    seventh field keeps the ONE definition (:func:`job_name`) without the process
    tax — and without the shell re-implementation exp_14 had to pin with a guard
    case. ``classify`` deliberately keeps the six-field prefix, so its rows stay
    parseable by position.
    """
    for cell in wave_cells(args.wave):
        line = _fmt_cell(cell)
        if args.with_jobname:
            line = f"{line} {job_name(cell)}"
        print(line)
    return 0


def _cell_from_args(args):
    """The Cell a CLI invocation names, or a usage error (ValueError)."""
    deg = parse_deg(getattr(args, "rotate_deg", None))
    if args.cell == "vctl":
        if deg is None:
            raise ValueError("a vctl cell needs a non-zero --rotate-deg")
    elif deg is not None:
        raise ValueError(f"--rotate-deg {args.rotate_deg} is meaningless for a "
                         f"{args.cell} cell: rrob draws its own angles and tbl is theta=0")
    return Cell(args.arm, args.cell, int(args.step), int(args.seed), int(args.k), deg)


def _cmd_jobname(args):
    """Print the canonical Slurm job name (both submitters are pinned to it)."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"jobname: {exc}", file=sys.stderr)
        return 2
    print(job_name(cell))
    return 0


def _cmd_evalname(args):
    """Print the canonical eval name (the driver's shell rendering is pinned to it)."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"evalname: {exc}", file=sys.stderr)
        return 2
    print(eval_name(cell))
    return 0


def _cmd_evalargv(args):
    """Print the cell class's PROTOCOL flags (plan §4.2), the one definition."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"evalargv: {exc}", file=sys.stderr)
        return 2
    if not is_registered(cell):
        print(f"evalargv: {tuple(cell)} is not registered in the exp_15 grid",
              file=sys.stderr)
        return 2
    print(" ".join(eval_argv_tail(cell)))
    return 0


def _cmd_argv(args):
    """Print the canonical check argv (the driver's rendering is pinned to it)."""
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"argv: {exc}", file=sys.stderr)
        return 2
    print(" ".join(check_argv(cell, args.metrics, pin=args.pin, ckpt_sha=args.ckpt_sha,
                              expected_count=args.expected_count,
                              expected_scenes=args.expected_scenes)))
    return 0


def _cmd_expect(args):
    """Print the admission expectation for one arm (or all) — the G4 input."""
    arms = (args.arm,) if args.arm else ARMS
    rc = 0
    for arm in arms:
        try:
            exp = admission_expectation(arm, args.control, args.registry)
        except ValueError as exc:
            print(f"EXPECT-UNAVAILABLE {arm}: {exc}", file=sys.stderr)
            rc = 1
            continue
        print(json.dumps(exp, sort_keys=True))
    return rc


def _cmd_check(args):
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"check: {exc}", file=sys.stderr)
        return 2
    if not is_registered(cell):
        print(f"CELL {tuple(cell)} is not registered in the exp_15 grid")
        return 2
    if not os.path.isfile(args.metrics):
        print(f"MISSING {eval_name(cell)}: {args.metrics}")
        return 3
    reasons = validate_cell(args.metrics, cell, pin=args.pin, ckpt_sha=args.ckpt_sha,
                            expected_count=args.expected_count,
                            expected_scenes=args.expected_scenes)
    if reasons:
        print(f"INVALID {eval_name(cell)} ({len(reasons)} reason(s)):")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print(f"VALID {eval_name(cell)} ({args.metrics})")
    return 0


def _cmd_cellstatus(args):
    """VALID / MISSING / INVALID for ONE cell — the single-cell path's dedup.

    The wave submitter has had validate-before-skip since round 1; the single-cell
    path (which the runbook uses for the first V and probe launches) had no
    equivalent and could therefore re-run a cell that had already landed
    (integrative review F6). This is the same predicate, for one cell.
    """
    try:
        cell = _cell_from_args(args)
    except ValueError as exc:
        print(f"cellstatus: {exc}", file=sys.stderr)
        return 2
    if not is_registered(cell):
        print(f"cellstatus: {tuple(cell)} is not registered", file=sys.stderr)
        return 2
    try:
        expect = admission_expectation(cell.arm, args.control, args.registry)["sha256"]
    except ValueError as exc:
        print(f"cellstatus: {exc}", file=sys.stderr)
        return 2
    ckpt = checkpoint_path(args.output_root, cell.arm, cell.step)
    if ckpt is None:
        print(f"MISSING no unique step={cell.step} checkpoint under {args.output_root}")
        return 3
    path = metrics_path(ckpt, cell)
    if not os.path.isfile(path):
        print(f"MISSING {path}")
        return 3
    reasons = validate_cell(path, cell, pin=args.pin, ckpt_sha=expect,
                            expected_count=args.expected_count,
                            expected_scenes=args.expected_scenes)
    if reasons:
        print(f"INVALID {path} :: " + "; ".join(reasons))
        return 1
    print(f"VALID {path}")
    return 0


def _cmd_classify(args):
    """One line per cell: identity, status, reasons. The wave submitter's input.

    Fail-closed on the ADMISSION side too: if an arm in this wave has no recorded
    checkpoint expectation (today: YAWAUG, whose chain has not reached 40k), the
    whole classification refuses rather than reporting its cells as MISSING —
    "not measured yet" and "cannot be judged yet" are different states and only
    the second one must stop a wave.
    """
    expect = {}
    for arm in sorted({c.arm for c in wave_cells(args.wave)}):
        try:
            expect[arm] = admission_expectation(arm, args.control, args.registry)["sha256"]
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
                  f"{args.output_root}/{ARM_CKPT_GLOB[cell.arm][0]}")
            continue
        path = metrics_path(ckpt, cell)
        if not os.path.isfile(path):
            print(f"{_fmt_cell(cell)} MISSING {path}")
            continue
        reasons = validate_cell(path, cell, pin=args.pin, ckpt_sha=expect[cell.arm],
                                expected_count=args.expected_count,
                                expected_scenes=args.expected_scenes)
        if reasons:
            print(f"{_fmt_cell(cell)} INVALID {path} :: " + "; ".join(reasons))
        else:
            print(f"{_fmt_cell(cell)} VALID {path}")
    return 0


def _add_cell_args(parser, metrics=False):
    if metrics:
        parser.add_argument("--metrics", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--step", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--k", required=True)
    parser.add_argument("--rotate-deg", default=None)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("grid", help="print the registered grid (one cell per line)")
    g.add_argument("--wave", default="all", choices=list(WAVES))
    g.add_argument("--with-jobname", action="store_true",
                   help="append the canonical Slurm job name as a seventh field")
    g.set_defaults(func=_cmd_grid)

    c = _add_cell_args(sub.add_parser("check", help="validate ONE cell's artifacts"),
                       metrics=True)
    c.add_argument("--pin", default=None)
    c.add_argument("--ckpt-sha", default=None)
    c.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    c.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
    c.set_defaults(func=_cmd_check)

    j = _add_cell_args(sub.add_parser("jobname", help="print the canonical Slurm job name"))
    j.set_defaults(func=_cmd_jobname)

    e = _add_cell_args(sub.add_parser("evalname", help="print the canonical eval name"))
    e.set_defaults(func=_cmd_evalname)

    ea = _add_cell_args(sub.add_parser("evalargv",
                                       help="print the cell class's eval-protocol flags"))
    ea.set_defaults(func=_cmd_evalargv)

    a = _add_cell_args(sub.add_parser("argv", help="print the canonical `check` argv"),
                       metrics=True)
    a.add_argument("--pin", default=None)
    a.add_argument("--ckpt-sha", default=None)
    a.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    a.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
    a.set_defaults(func=_cmd_argv)

    ct = _add_cell_args(sub.add_parser("contract",
                                       help="print the cell's protocol contract"))
    ct.set_defaults(func=_cmd_contract)

    x = sub.add_parser("expect", help="print the recorded checkpoint admission facts")
    x.add_argument("--arm", default=None, choices=list(ARMS))
    x.add_argument("--control", default=CONTROL_ADMISSION)
    x.add_argument("--registry", default=LAUNCH_REGISTRY)
    x.set_defaults(func=_cmd_expect)

    cs = _add_cell_args(sub.add_parser("cellstatus",
                                       help="VALID/MISSING/INVALID for ONE cell"))
    cs.add_argument("--output-root", required=True)
    cs.add_argument("--pin", default=None)
    cs.add_argument("--control", default=CONTROL_ADMISSION)
    cs.add_argument("--registry", default=LAUNCH_REGISTRY)
    cs.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    cs.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
    cs.set_defaults(func=_cmd_cellstatus)

    cl = sub.add_parser("classify", help="status of every cell in a wave (dedup input)")
    cl.add_argument("--wave", default="all", choices=list(WAVES))
    cl.add_argument("--output-root", required=True)
    cl.add_argument("--pin", default=None)
    cl.add_argument("--control", default=CONTROL_ADMISSION,
                    help="committed VANL control-admission record")
    cl.add_argument("--registry", default=LAUNCH_REGISTRY,
                    help="committed YAWAUG chain launch registry")
    cl.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    cl.add_argument("--expected-scenes", type=int, default=EXPECTED_SCENES)
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
