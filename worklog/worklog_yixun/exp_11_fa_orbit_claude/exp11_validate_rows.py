#!/usr/bin/env python3
"""exp_11 row validator — nothing enters model_comparison.md unproven (plan §5).

Yixun's standing request (Q7): every number in a comparison table must be
traceable to the protocol it claims. This validator is the gate. For one cell
(arm x step x K x seed set) it reads each metrics JSON **and** the screen sidecar
written beside it, and refuses the cell unless it can prove:

  * ``cond_method == 'fa_invariant'`` and the EXACT uniform orbit of that arm
    (float-for-float: an int ``45`` is not ``45.0``, and a C4 orbit on a C8 row
    is a different experiment);
  * the checkpoint is the arm's OWN checkpoint at the claimed step;
  * K matches the dataset config actually used (the ``_1`` split is K=1);
  * cfg 1.0, 1 diffusion step, bf16 conditioning autocast, EMA weights;
  * the eval seeds appear EXACTLY once each — no missing seed, no duplicate;
  * the exp_11 execution provenance: ``orbit_execution == 'batched'`` with the
    matching cap and a non-empty source SHA, so a legacy-loop row can never be
    averaged in with batched ones.

Everything it cannot prove is a problem, never a default. Usage:

    python exp11_validate_rows.py --arm C8 --step 10000 --k 8 --seeds 42,43,44,45,46 <metrics.json ...>
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys


def _repo_root(p):
    p = os.path.abspath(p)
    # `.git` is a DIRECTORY in a normal checkout and a FILE in a linked worktree —
    # measurements run from a pinned worktree, so both must count as the root.
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Where the OUTPUTS live. Under worktree-pinned measurement the code root is the
# pinned worktree while checkpoints and metrics stay in the main tree, so the
# containment check must resolve against the main tree, not the code root.
OUTPUT_ROOT_BASE = os.environ.get("EXP11_OUTPUT_ROOT") or REPO

from src.data.yaw_rotation import (  # noqa: E402
    FRAME_AVG_MAX_FWD_SAMPLES, ORBIT_EXECUTION)

# arm -> orbit size. C4BACKFILL is the exp_07 B-F lineage screened under the fa
# protocol (plan §3); it is C4 by construction.
# VANL has NO orbit: not 1, none. Frame averaging is the single delta it exists
# to remove, so every orbit-shaped question about it has to answer "n/a".
ARM_ORBITS = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32, "C4BACKFILL": 4, "VANL": None}
ARM_RUN_PREFIX = {
    "C4L": "outputs_FLAC/exp11_C4L/", "C8": "outputs_FLAC/exp11_C8/",
    "C16": "outputs_FLAC/exp11_C16/", "C32": "outputs_FLAC/exp11_C32/",
    "VANL": "outputs_FLAC/exp11_VANL/",
    "C4BACKFILL": "outputs_FLAC/exp07_BF/",
}
EVAL_CONFIG_FOR_K = {
    8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
    1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
}
EXPECTED_CFG_SCALE = 1.0
EXPECTED_STEPS = 1
EXPECTED_AUTOCAST = "bf16"
EXPECTED_WEIGHTS = "ema"
EXPECTED_N_SAMPLES = 6337          # the full published unseen split (announcement 01)
# The metric keys eval_FLAC ACTUALLY emits, pinned from a real record produced by
# the C4 backfill screen (job 3649599, exp11_C4backfill_S20000_s42_K8): the
# earlier "exact six" was the TABLE subset and rejected every genuine row.
EMITTED_METRIC_KEYS = (
    "T60", "C50", "EDT", "FD", "Invalid T60",
    "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10",
    "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10",
)
# The six a comparison row reports (gen_model_comparison.KEYS) — a strict subset:
# they must be present and finite, and the emission set must not drift either way.
REQUIRED_METRIC_KEYS = ("T60", "C50", "EDT",
                        "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10")
assert set(REQUIRED_METRIC_KEYS) <= set(EMITTED_METRIC_KEYS)
# Every one of these must be PRESENT in the evaluator's own record with the right
# type — a missing or null field used to make its cross-check silently skip
# (re-review item 1). (field, type, expected-or-None)
# VANL (Q9) is the vanilla arm of this lineage. Three record fields are
# necessarily different for it -- the conditioning method, and the two orbit
# provenance fields that describe an orbit it never executes -- so they are
# lifted out of the shared tuple and asserted per-protocol below.
VANILLA_ARMS = ("VANL",)
PROTOCOL_SPECIFIC_FIELDS = ("cond_method", "orbit_execution", "frame_avg_fwd_cap",
                            "frame_avg_angles")
MANDATORY_RECORD_FIELDS = (
    ("cond_method", str, "fa_invariant"),
    ("cond_autocast", str, EXPECTED_AUTOCAST),
    ("orbit_execution", str, None),
    ("frame_avg_fwd_cap", int, None),
    ("source_sha", str, None),
    ("ckpt_path", str, None),
    ("rotate_deg", float, None),
    ("seed", int, None),
    ("cfg_scale", float, EXPECTED_CFG_SCALE),
    ("steps", int, EXPECTED_STEPS),
    ("eval_name", str, None),
    ("dataset_config", str, None),
    ("batch_size", int, None),
    ("n_samples", int, EXPECTED_N_SAMPLES),
    ("weights_source", str, EXPECTED_WEIGHTS),
    ("device", str, None),
    ("frame_avg_angles", list, None),
)
EXPECTED_BATCH_SIZE = 64           # eval_FLAC's registered evaluation batch
EXPECTED_DEVICE_PREFIX = "cuda"    # a CPU evaluation is not the registered protocol
MANDATORY_SIDECAR_FIELDS = (
    "arm", "step", "seed", "K", "eval_name", "cfg_scale", "steps", "model_config",
    "model_config_sha256", "dataset_config", "ckpt_path", "ckpt_sha256", "use_ema",
    "frame_avg_angles", "cond_method", "cond_autocast", "commit",
)

# Purpose-specific contracts (round-4 review B4): the seed policy is REGISTERED,
# never supplied by the caller, and an R3 rotation row is never table-admissible.
CONTRACTS = {
    # Trajectory screens run at BOTH K (full curves at K=1 and K=8). The futility
    # GATES are narrower and stay K=8 only: `gate_K` records that separately so
    # widening the cadence cannot drift the gate semantics. A K=1 screen is a
    # perfectly good trajectory point and is NOT gate-admissible evidence.
    "futility": {"cells": ("screen", "backfill"), "seeds": (42,), "K": (1, 8),
                 "gate_K": (8,), "table_admissible": False},
    "table":    {"cells": ("conf",), "seeds": (42, 43, 44, 45, 46), "K": (1, 8),
                 "table_admissible": True},
    # The Q9 round: VANL and C4L measured at ONE new pin, five seeds, both K.
    # Same shape as `table` — it IS a table contract — but a distinct cell so the
    # original campaign's conf evidence is preserved rather than overwritten.
    # Q10 trajectory cells: five seeds x both K at each checkpoint ABOVE 40k, so
    # the extended curve carries error bars. NOT table-admissible -- the table's
    # comparison point stays 40k -- and not gate evidence either. Figure-admissible
    # at the futility provenance bar (hash recomputation optional), because a
    # trajectory point is read as a curve, not as a published number.
    "traj":     {"cells": ("traj",), "seeds": (42, 43, 44, 45, 46), "K": (1, 8),
                 "min_step_exclusive": 40000, "max_step": 100000, "step_grid": 2500,
                 "table_admissible": False, "figure_admissible": True},
    "q9":       {"cells": ("q9",), "seeds": (42, 43, 44, 45, 46), "K": (1, 8),
                 "step": 40000, "arms": ("VANL", "C4L"), "table_admissible": True},
    # R3 (plan §4) is ONE seed evaluated at five registered yaw offsets — the
    # exactly-once SEED logic would call those five files duplicates, so this
    # contract is keyed by ROTATIONS instead (re-review item 4).
    "r3":       {"cells": ("r3",), "seeds": (42,), "K": (8,),
                 "rotations": (0.0, 5.625, 11.25, 22.5, 45.0),
                 "step": 40000, "table_admissible": False},
    # CROSS (R2 mechanism / D2): one checkpoint evaluated under orbits it was
    # NOT trained on. Keyed by EVAL ORBIT, and the registered set is per-arm --
    # every orbit except the arm's own, because "evaluated at its own orbit" is
    # what screen/conf already are. Never table-admissible: a cross row is
    # mechanism evidence, not a result for the model.
    "cross":    {"cells": ("cross",), "seeds": (42,), "K": (8,),
                 "orbits": "all-but-training", "step": 40000,
                 "table_admissible": False},
}
REGISTERED_ROTATIONS = CONTRACTS["r3"]["rotations"]
ALL_ORBITS = (4, 8, 16, 32)
R3_STEP = CONTRACTS["r3"]["step"]        # R3 and CROSS are registered at the
CROSS_STEP = CONTRACTS["cross"]["step"]  # 40k endpoint only (plan §4)


def cross_orbits_for(arm):
    """The eval orbits a cross sweep must cover for ``arm``: all but its own."""
    return tuple(n for n in ALL_ORBITS if n != ARM_ORBITS[arm])
# Provenance that must be IDENTICAL across every row of a cell.
CELL_IDENTITY_FIELDS = ("cell", "ckpt_path", "ckpt_sha256", "model_config_sha256",
                        "source_sha", "commit", "orbit_execution")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# q9 is a SEPARATE conf namespace for the Q9 fa-vs-vanilla round. Re-measuring
# C4L at the new pin under the old `conf` name would overwrite the published
# 0c6e9ff evidence file-for-file; a distinct cell keeps both rounds on disk.
_SCREEN_RE = re.compile(r"^exp11_(C4L|C8|C16|C32|VANL)_(screen|conf|q9|traj)_S(\d+)_s(\d+)_K(\d+)$")
_BACKFILL_RE = re.compile(r"^exp11_C4backfill_S(\d+)_s(\d+)_K(\d+)$")
# R3 carries the ROTATION in the name: the five rows of an R3 cell otherwise
# share one eval name and are distinguishable only by a field inside the file.
_R3_RE = re.compile(r"^exp11_(C4L|C8|C16|C32)_r3_rot(\d+(?:p\d+)?)_s(\d+)_K(\d+)$")
# CROSS carries the EVAL orbit, which is exactly what varies across the cell.
_CROSS_RE = re.compile(r"^exp11_(C4L|C8|C16|C32)_cross_a(\d+)_S(\d+)_s(\d+)_K(\d+)$")
_BACKFILL_CROSS_RE = re.compile(r"^exp11_C4backfill_cross_a(\d+)_S(\d+)_s(\d+)_K(\d+)$")


def rotation_from_token(tok):
    """``'5p625' -> 5.625``. The inverse of eval_FLAC.rot_token."""
    return float(tok.replace("p", "."))


def rot_token(rotate_deg):
    """Re-exported from eval_FLAC so callers render R3 names exactly one way."""
    from eval_FLAC import rot_token as _t
    return _t(rotate_deg)


def gate_admissible(k, contract="futility"):
    """May a row at this K be read as FUTILITY-GATE evidence?

    Trajectory screens exist at K=1 and K=8; the pre-registered gates are defined
    on K=8. Keeping the two apart is the whole point of a separate `gate_K`."""
    # NO fallback to the contract's ordinary K. Falling back made
    # gate_admissible(8, "traj") return True, so the "trajectory rows are never
    # gate evidence" claim was false in the one function that encodes it. A
    # contract is gate-bearing only if it says so, in gate_K, explicitly.
    spec = CONTRACTS.get(contract, {})
    return k in spec.get("gate_K", ())


def is_vanilla_arm(arm):
    return arm in VANILLA_ARMS


def orbit_for(arm):
    """The exact uniform orbit an arm's rows must carry (``[]`` for a vanilla arm)."""
    if arm not in ARM_ORBITS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARM_ORBITS)}")
    n = ARM_ORBITS[arm]
    if n is None:
        return []
    return [k * 360.0 / n for k in range(n)]


def parse_eval_name(name):
    """Parse the plan §4 eval-name schema into its identity fields."""
    m = _SCREEN_RE.match(name or "")
    if m:
        return {"arm": m.group(1), "cell": m.group(2), "step": int(m.group(3)),
                "seed": int(m.group(4)), "K": int(m.group(5))}
    m = _R3_RE.match(name or "")
    if m:
        # The step is not in an R3 name because the contract registers exactly
        # one (the 40k endpoint); the rotation takes the distinguishing slot.
        return {"arm": m.group(1), "cell": "r3", "step": R3_STEP,
                "rotate_deg": rotation_from_token(m.group(2)),
                "seed": int(m.group(3)), "K": int(m.group(4))}
    m = _CROSS_RE.match(name or "")
    if m:
        return {"arm": m.group(1), "cell": "cross", "eval_orbit": int(m.group(2)),
                "step": int(m.group(3)), "seed": int(m.group(4)), "K": int(m.group(5))}
    m = _BACKFILL_CROSS_RE.match(name or "")
    if m:
        return {"arm": "C4BACKFILL", "cell": "cross", "eval_orbit": int(m.group(1)),
                "step": int(m.group(2)), "seed": int(m.group(3)), "K": int(m.group(4))}
    m = _BACKFILL_RE.match(name or "")
    if m:
        return {"arm": "C4BACKFILL", "cell": "backfill", "step": int(m.group(1)),
                "seed": int(m.group(2)), "K": int(m.group(3))}
    raise ValueError(f"eval name {name!r} does not match the exp_11 schema "
                     "(exp11_<arm>_<screen|conf>_S<step>_s<seed>_K<k>, "
                     "exp11_<arm>_r3_rot<deg>_s<seed>_K<k>, "
                     "exp11_<arm>_cross_a<orbit>_S<step>_s<seed>_K<k>, "
                     "or exp11_C4backfill[_cross_a<orbit>]_S..)")


def sidecar_path_for(metrics_path):
    """The screen sidecar written beside a metrics JSON."""
    return f"{metrics_path}.screenmeta.json"


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        raise ValueError(f"duplicate JSON key(s): {dups}")
    return dict(pairs)


def _reject_constant(name):
    raise ValueError(f"non-standard JSON constant {name!r} (NaN/Infinity)")


def _load_json(path):
    """Strict: duplicate keys and NaN/Infinity constants are corruption, not data."""
    with open(path, "r") as fh:
        return json.load(fh, object_pairs_hook=_no_duplicate_keys,
                         parse_constant=_reject_constant)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_HASH_CACHE = {}


def _cached_sha256(path):
    """Checkpoints are ~700 MB; hash each one once per validation run."""
    real = os.path.realpath(path)
    if real not in _HASH_CACHE:
        _HASH_CACHE[real] = sha256_file(real)
    return _HASH_CACHE[real]


def _finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _filename_protocol(metrics_path):
    """``(steps, cfg, eval_name)`` from ``<ckpt>_metrics_<steps>_<cfg>_<name>...``."""
    base = os.path.basename(metrics_path)
    m = re.search(r"_metrics_(\d+)_([0-9.]+)_(.+?)(?:_fa_invariant_a\d+)?(?:_rot\d+)?\.json$", base)
    if not m:
        return None, None, None
    return int(m.group(1)), float(m.group(2)), m.group(3)


def _angles_equal(got, want):
    """Exact, type-strict comparison: an int 45 is not the float 45.0."""
    if not isinstance(got, list) or len(got) != len(want):
        return False
    return all(isinstance(a, float) and abs(a - b) < 1e-9 for a, b in zip(got, want))


def validate_row(metrics_path, verify_hashes=False):
    """Validate ONE metrics JSON. Returns ``(row_info, problems)``.

    Fail-closed everywhere: strict JSON, all six finite table metrics, the exact
    filename ``build_output_paths`` would generate, an unrotated row, EMA weights
    PROVEN by the record's own ``weights_source``, the full-split item count, a
    real 40-hex source commit, every mandatory sidecar field present, and the
    record and sidecar telling the same story field by field.
    """
    problems = []
    try:
        rec = _load_json(metrics_path)
    except Exception as exc:
        return {}, [f"{os.path.basename(metrics_path)}: unreadable/invalid metrics JSON: {exc}"]

    side_path = sidecar_path_for(metrics_path)
    if not os.path.isfile(side_path):
        return {}, [f"{os.path.basename(metrics_path)}: screen sidecar missing "
                    f"({os.path.basename(side_path)}) — the protocol cannot be proven"]
    try:
        side = _load_json(side_path)
    except Exception as exc:
        return {}, [f"{os.path.basename(metrics_path)}: unreadable/invalid sidecar: {exc}"]

    tag = os.path.basename(metrics_path)
    missing_side = [f for f in MANDATORY_SIDECAR_FIELDS if f not in side]
    if missing_side:
        return {}, [f"{tag}: sidecar is missing mandatory field(s) {missing_side}"]
    try:
        ident = parse_eval_name(side["eval_name"])
    except ValueError as exc:
        return {}, [f"{tag}: {exc}"]

    arm, cell, step, seed, k = ident["arm"], ident["cell"], ident["step"], ident["seed"], ident["K"]
    # A CROSS row is the one case where the evaluated orbit is deliberately NOT
    # the arm's training orbit: that mismatch is the measurement. Everything else
    # (the checkpoint's own embedded angles, checked by the screen driver) still
    # has to be the training orbit — this is about what the EVAL ran.
    # A vanilla arm has no orbit integer at all, so every orbit-shaped quantity
    # below degenerates to zero/empty rather than raising on None.
    train_orbit = ARM_ORBITS[arm] or 0
    eval_orbit = ident.get("eval_orbit", train_orbit) or 0
    if cell == "cross":
        if eval_orbit not in ALL_ORBITS:
            problems.append(f"{os.path.basename(metrics_path)}: eval orbit a{eval_orbit} is not "
                            f"one of the registered orbits {ALL_ORBITS}")
        if eval_orbit == train_orbit:
            problems.append(f"{os.path.basename(metrics_path)}: cross row evaluates {arm} at its "
                            f"OWN training orbit C{train_orbit} — that is a screen/conf row, not "
                            "a cross-orbit measurement")
    want_angles = [j * 360.0 / eval_orbit for j in range(eval_orbit)] if eval_orbit else []

    # --- the sidecar must agree with its own eval name -----------------------
    for field, want in (("arm", arm), ("step", step), ("seed", seed), ("K", k)):
        if side.get(field) != want:
            problems.append(f"{tag}: sidecar {field}={side.get(field)!r} != {want!r} from the eval name")

    # --- protocol constants ---------------------------------------------------
    if side.get("cfg_scale") != EXPECTED_CFG_SCALE:
        problems.append(f"{tag}: cfg_scale={side.get('cfg_scale')!r} != {EXPECTED_CFG_SCALE}")
    if side.get("steps") != EXPECTED_STEPS:
        problems.append(f"{tag}: steps={side.get('steps')!r} != {EXPECTED_STEPS}")
    if side.get("use_ema") is not True:
        problems.append(f"{tag}: use_ema={side.get('use_ema')!r} — screens evaluate EMA weights")
    want_ds = EVAL_CONFIG_FOR_K.get(k)
    if want_ds is None:
        problems.append(f"{tag}: K={k} has no registered eval dataset config")
    elif os.path.normpath(str(side.get("dataset_config", ""))) != os.path.normpath(want_ds):
        problems.append(f"{tag}: dataset_config={side.get('dataset_config')!r} is not the "
                        f"K={k} unseen split ({want_ds})")

    # --- the metrics themselves: EXACTLY the six, finite, numeric -------------
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict):
        problems.append(f"{tag}: metrics block is missing or not a mapping")
    else:
        if set(metrics) != set(EMITTED_METRIC_KEYS):
            extra = sorted(set(metrics) - set(EMITTED_METRIC_KEYS))
            absent = sorted(set(EMITTED_METRIC_KEYS) - set(metrics))
            problems.append(f"{tag}: metric key set drifted from the registered emission set "
                            f"(unexpected {extra}, missing {absent})")
        missing_table = [m for m in REQUIRED_METRIC_KEYS if m not in metrics]
        if missing_table:
            problems.append(f"{tag}: the table metrics {missing_table} are absent")
        bad = [m for m in sorted(set(metrics) & set(EMITTED_METRIC_KEYS))
               if not _finite_number(metrics[m])]
        if bad:
            problems.append(f"{tag}: metrics {bad} are not finite numbers (bools are not numbers)")

    # --- every evaluator field: present, right type, right value (item 1) -----
    vanilla = is_vanilla_arm(arm)
    for field, typ, expect in MANDATORY_RECORD_FIELDS:
        if vanilla and field in PROTOCOL_SPECIFIC_FIELDS:
            continue                      # asserted per-protocol immediately below
        if field not in rec or rec[field] is None:
            problems.append(f"{tag}: evaluator record is missing {field} — an absent field is "
                            "not a passing check")
            continue
        val = rec[field]
        ok_type = isinstance(val, float) if typ is float else isinstance(val, typ)
        if typ is float and isinstance(val, int) and not isinstance(val, bool):
            ok_type = True                                   # 1 is an acceptable 1.0
        if typ is int and isinstance(val, bool):
            ok_type = False
        if not ok_type:
            problems.append(f"{tag}: {field}={val!r} is {type(val).__name__}, expected "
                            f"{typ.__name__}")
            continue
        if expect is not None and val != expect:
            problems.append(f"{tag}: {field}={val!r} != {expect!r}")
    if isinstance(rec.get("batch_size"), int) and rec["batch_size"] != EXPECTED_BATCH_SIZE:
        problems.append(f"{tag}: batch_size={rec['batch_size']} != the registered "
                        f"{EXPECTED_BATCH_SIZE}")
    if isinstance(rec.get("device"), str) and not rec["device"].startswith(EXPECTED_DEVICE_PREFIX):
        problems.append(f"{tag}: device={rec['device']!r} is not a {EXPECTED_DEVICE_PREFIX} "
                        "evaluation")

    # --- the VANILLA protocol statement (Q9 / NEW-6) --------------------------
    if vanilla:
        # FAIL-CLOSED: an ABSENT key is not a passing check. The orbit provenance
        # fields must be PRESENT and exactly None -- "the evaluator declared no
        # orbit" and "the evaluator forgot to say" are different claims, and only
        # the first one is evidence.
        for f in ("frame_avg_fwd_cap", "frame_avg_angles"):
            if f not in rec:
                problems.append(f"{tag}: a vanilla record must contain {f} (explicitly null), "
                                "not omit it — an absent field is not a declaration")
            elif rec[f] is not None:
                problems.append(f"{tag}: {f}={rec[f]!r} must be exactly null on a vanilla row")
        if "frame_avg_angles" not in side:
            problems.append(f"{tag}: a vanilla sidecar must contain frame_avg_angles (explicitly null)")
        elif side["frame_avg_angles"] is not None:
            problems.append(f"{tag}: sidecar frame_avg_angles={side['frame_avg_angles']!r} must be "
                            "exactly null on a vanilla row")
        if rec.get("cond_method") != "vanilla":
            problems.append(f"{tag}: cond_method={rec.get('cond_method')!r} — a {arm} row must be "
                            "a vanilla evaluation")
        if rec.get("orbit_execution") != "n/a":
            problems.append(f"{tag}: orbit_execution={rec.get('orbit_execution')!r} != 'n/a' — "
                            "a vanilla evaluation executes no orbit, and labelling it 'batched' "
                            "would make it look protocol-compatible with a frame-averaged row")
        if cell not in ("screen", "conf", "q9"):
            problems.append(f"{tag}: cell {cell!r} is not registered for {arm} — r3 and cross are "
                            "UNREGISTERED for a vanilla arm in this campaign (yaw sensitivity is a "
                            "meaningful question for it, just not one this round asks)")

    # --- the record's own protocol statement ---------------------------------
    if not vanilla and not _angles_equal(rec.get("frame_avg_angles"), want_angles):
        problems.append(f"{tag}: frame_avg_angles={rec.get('frame_avg_angles')!r} is not the "
                        f"exact C{ARM_ORBITS[arm]} orbit of {arm}")
    if float(rec.get("rotate_deg") or 0.0) != 0.0 and cell != "r3":
        problems.append(f"{tag}: rotate_deg={rec.get('rotate_deg')!r} — a rotated evaluation is "
                        "not a screen/table row")
    if cell == "r3":
        # The name says which rotation this row is; the record must agree, or the
        # five rows of the cell are not the five rotations they claim to be.
        named = float(ident["rotate_deg"])
        got = float(rec.get("rotate_deg") or 0.0)
        if abs(got - named) > 1e-9:
            problems.append(f"{tag}: eval name says rotate_deg={named} but the record says {got}")
        if named not in REGISTERED_ROTATIONS:
            problems.append(f"{tag}: rotation {named} is not one of the registered R3 offsets "
                            f"{REGISTERED_ROTATIONS}")
    if cell == "cross":
        # The sidecar must state BOTH orbits explicitly: a reader of one row has
        # to be able to see what was trained and what was evaluated.
        for field, want in (("training_orbit", train_orbit), ("eval_orbit", eval_orbit)):
            if field not in side:
                problems.append(f"{tag}: a cross sidecar must record {field}")
            elif int(side[field]) != want:
                problems.append(f"{tag}: sidecar {field}={side[field]!r} != {want}")
        if len(rec.get("frame_avg_angles") or []) != eval_orbit:
            problems.append(f"{tag}: the record evaluated {len(rec.get('frame_avg_angles') or [])} "
                            f"angles, but the name claims a{eval_orbit}")
    if not vanilla and rec.get("orbit_execution") != ORBIT_EXECUTION:
        problems.append(f"{tag}: orbit_execution={rec.get('orbit_execution')!r} != "
                        f"{ORBIT_EXECUTION!r} — legacy-loop rows are not comparable with these")
    if not vanilla and rec.get("frame_avg_fwd_cap") != FRAME_AVG_MAX_FWD_SAMPLES:
        problems.append(f"{tag}: frame_avg_fwd_cap={rec.get('frame_avg_fwd_cap')!r} != "
                        f"{FRAME_AVG_MAX_FWD_SAMPLES}")
    src = str(rec.get("source_sha") or "")
    if not _COMMIT_RE.match(src):
        problems.append(f"{tag}: source_sha={rec.get('source_sha')!r} is not a 40-hex commit sha")
    elif src != str(side.get("commit", "")):
        problems.append(f"{tag}: source_sha {src[:12]} != sidecar commit "
                        f"{str(side.get('commit'))[:12]} — evaluator and driver ran different code")

    # --- record vs sidecar: one story, field by field -------------------------
    for field in ("cond_method", "cond_autocast", "seed", "cfg_scale", "steps", "eval_name"):
        if side.get(field) != rec.get(field):
            problems.append(f"{tag}: sidecar and metrics disagree on {field} "
                            f"({side.get(field)!r} vs {rec.get(field)!r})")
    if os.path.normpath(str(rec.get("dataset_config", ""))) != \
            os.path.normpath(str(side.get("dataset_config", ""))):
        problems.append(f"{tag}: sidecar and metrics disagree on dataset_config")
    if not vanilla and not _angles_equal(side.get("frame_avg_angles"), want_angles):
        problems.append(f"{tag}: sidecar frame_avg_angles are not {arm}'s exact orbit")

    # --- the checkpoint -------------------------------------------------------
    ckpt = str(rec.get("ckpt_path", ""))
    if side.get("ckpt_path") != ckpt:
        problems.append(f"{tag}: sidecar and metrics disagree on ckpt_path")
    if not os.path.basename(ckpt).endswith(f"step={step}.ckpt"):
        problems.append(f"{tag}: ckpt {os.path.basename(ckpt)} is not step {step}")
    # Containment by resolved PATH, not substring: "…/exp11_C8_backup/…" contains
    # the C8 prefix but is not the arm's run directory (re-review item 2).
    prefix = ARM_RUN_PREFIX[arm]
    base = OUTPUT_ROOT_BASE
    ckpt_real = os.path.realpath(ckpt if os.path.isabs(ckpt) else os.path.join(base, ckpt))
    root_real = os.path.realpath(os.path.join(base, prefix))
    if os.path.commonpath([ckpt_real, root_real]) != root_real:
        problems.append(f"{tag}: ckpt {ckpt} is not inside this arm's own run directory ({prefix})")
    for field, rx, what in (("model_config_sha256", _SHA256_RE, "64-hex sha256"),
                            ("ckpt_sha256", _SHA256_RE, "64-hex sha256"),
                            ("commit", _COMMIT_RE, "40-hex commit")):
        if not isinstance(side.get(field), str) or not rx.match(side[field]):
            problems.append(f"{tag}: sidecar {field}={side.get(field)!r} is not a {what}")

    # --- hashes are recomputed, not trusted (round-4 review B3/B6) ------------
    if verify_hashes:
        for field, path_field in (("model_config_sha256", "model_config"),
                                  ("ckpt_sha256", "ckpt_path")):
            # The sidecar records the model config REPO-RELATIVE (an absolute
            # path into a pinned worktree dangles once that tree is pruned), so
            # resolve it against the registered root — never the ambient cwd,
            # which for a validator run from a worktree is the wrong tree, and
            # for a run from anywhere else is nothing at all.
            target = str(side.get(path_field, ""))
            if target and not os.path.isabs(target):
                target = os.path.join(OUTPUT_ROOT_BASE if path_field == "ckpt_path" else REPO,
                                      target)
            if not os.path.isfile(target):
                problems.append(f"{tag}: cannot recompute {field}: {target} is not readable")
                continue
            got = _cached_sha256(target)
            if got != side.get(field):
                problems.append(f"{tag}: {field} mismatch — sidecar {side.get(field)}, "
                                f"file {got} ({target})")

    # --- the filename must be EXACTLY what build_output_paths generates --------
    want_name = _expected_filename(ckpt, side, arm, rec.get("rotate_deg", 0.0), eval_orbit,
                                   cond_method="vanilla" if vanilla else "fa_invariant")
    if want_name is None:
        problems.append(f"{tag}: cannot derive the expected filename (bad ckpt/sidecar)")
    elif os.path.basename(metrics_path) != want_name:
        problems.append(f"{tag}: filename is not the one build_output_paths would generate "
                        f"({want_name})")

    row = {"path": metrics_path, "arm": arm, "cell": cell, "step": step,
           "rotate_deg": float(rec.get("rotate_deg") or 0.0),
           "eval_orbit": eval_orbit, "training_orbit": train_orbit,
           "seed": seed, "K": k, "ckpt_path": ckpt, "metrics": metrics if isinstance(metrics, dict) else {},
           "source_sha": rec.get("source_sha"), "eval_name": side.get("eval_name"),
           "ckpt_sha256": side.get("ckpt_sha256"),
           "model_config_sha256": side.get("model_config_sha256"),
           "commit": side.get("commit"), "orbit_execution": rec.get("orbit_execution")}
    return row, problems


def _expected_filename(ckpt_path, side, arm, rotate_deg, n_angles=None, cond_method=None):
    """The filename ``eval_FLAC.build_output_paths`` would produce for this row.

    ``n_angles`` is the EVALUATED orbit, which for a cross row is not the arm's
    training orbit — the ``aN`` in the filename follows what actually ran."""
    if not n_angles:
        n_angles = ARM_ORBITS[arm] or 0
    try:
        from eval_FLAC import build_output_paths
        paths = build_output_paths(ckpt_path, int(side["steps"]), float(side["cfg_scale"]),
                                   str(side["eval_name"]),
                                   cond_method=cond_method or "fa_invariant",
                                   rotate_deg=float(rotate_deg or 0.0),
                                   n_angles=int(n_angles))
        return os.path.basename(paths["metrics"])
    except Exception:
        return None


def validate_cell(metrics_paths, arm, step, k, contract, verify_hashes=False):
    """Validate a whole cell against a REGISTERED contract (round-4 review B4).

    ``contract`` fixes the admissible cell types, the seed set and the K values —
    the caller cannot widen them — and every row of the cell must share one
    checkpoint, one config hash and one evaluator identity."""
    problems = []
    if contract not in CONTRACTS:
        return [], [f"unknown contract {contract!r}; known: {sorted(CONTRACTS)}"]
    spec = CONTRACTS[contract]
    if k not in spec["K"]:
        problems.append(f"contract {contract} does not admit K={k} (allowed {spec['K']})")

    rows = []
    for path in metrics_paths:
        row, probs = validate_row(path, verify_hashes=verify_hashes)
        problems.extend(probs)
        if row:
            rows.append(row)

    for row in rows:
        base = os.path.basename(row["path"])
        if row["arm"] != arm:
            problems.append(f"{base}: arm {row['arm']} does not belong to the {arm} cell")
        if row["step"] != step:
            problems.append(f"{base}: step {row['step']} != {step}")
        if row["K"] != k:
            problems.append(f"{base}: K={row['K']} != {k}")
        if row["cell"] not in spec["cells"]:
            problems.append(f"{base}: cell type {row['cell']!r} is not admissible under the "
                            f"{contract} contract (allowed {spec['cells']})")

    # --- the registered replication set, exactly once each --------------------
    # For screen/table cells the replication axis is the SEED; for R3 it is the
    # yaw offset (one seed, five registered rotations) — the seed-keyed logic
    # called those five files duplicates and made the block unvalidatable.
    if "rotations" in spec:
        axis, wanted = "rotate_deg", spec["rotations"]
    elif "orbits" in spec:
        # per-ARM: every registered orbit except the one it trained on
        axis, wanted = "eval_orbit", cross_orbits_for(arm)
    else:
        axis, wanted = "seed", spec["seeds"]
    lo = spec.get("min_step_exclusive")
    if lo is not None and step <= lo:
        problems.append(f"contract {contract} covers steps strictly above {lo} (got {step}) — "
                        f"the <= {lo} curve is the single-seed screen record")
    hi = spec.get("max_step")
    if hi is not None and step > hi:
        problems.append(f"contract {contract} stops at the {hi} budget (got {step})")
    grid = spec.get("step_grid")
    if grid is not None and step % grid:
        problems.append(f"contract {contract} sits on the {grid}-step checkpoint grid (got {step})")
    if spec.get("step") is not None and step != spec["step"]:
        problems.append(f"contract {contract} is registered at step {spec['step']} only, got {step}")
    if spec.get("arms") and arm not in spec["arms"]:
        problems.append(f"contract {contract} is registered for {spec['arms']} only, got {arm} — "
                        "the Q9 round is the fa-vs-vanilla pair, not the whole sweep")
    seen = {}
    for row in rows:
        seen.setdefault(row[axis], []).append(row["path"])
    if axis in ("rotate_deg", "eval_orbit"):
        for row in rows:
            if row["seed"] not in spec["seeds"]:
                problems.append(f"{os.path.basename(row['path'])}: seed {row['seed']} is not the "
                                f"registered {contract} seed {spec['seeds']}")
    for want in wanted:
        n = len(seen.get(want, []))
        if n == 0:
            problems.append(f"{arm} S{step} K{k} [{contract}]: {axis} {want} is missing")
        elif n > 1:
            problems.append(f"{arm} S{step} K{k} [{contract}]: {axis} {want} appears more than once "
                            f"({[os.path.basename(p) for p in seen[want]]})")
    for got in sorted(seen):
        if got not in wanted:
            problems.append(f"{arm} S{step} K{k} [{contract}]: unexpected {axis} {got} "
                            f"(the contract registers {wanted})")

    # --- one checkpoint, one config, one evaluator across the whole cell ------
    for field in CELL_IDENTITY_FIELDS:
        values = {str(r.get(field)) for r in rows}
        if len(values) > 1:
            problems.append(f"{arm} S{step} K{k} [{contract}]: rows disagree on {field} — a cell "
                            f"must be one identity, got {sorted(values)}")
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate exp_11 metric rows before they enter a table")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_ORBITS))
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--contract", required=True, choices=sorted(CONTRACTS),
                    help="futility (screens/backfill, seed 42, K8), table (conf, seeds 42-46), r3")
    ap.add_argument("--output-root", default=None,
                    help="where outputs_FLAC lives (default: this checkout; set to the MAIN tree "
                         "when the code runs from a pinned measurement worktree)")
    ap.add_argument("--verify-hashes", action="store_true",
                    help="recompute the config/checkpoint sha256 instead of trusting the sidecar")
    ap.add_argument("paths", nargs="+", help="metrics JSONs (globs are expanded)")
    args = ap.parse_args(argv)
    if args.output_root:
        global OUTPUT_ROOT_BASE
        OUTPUT_ROOT_BASE = args.output_root

    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    rows, problems = validate_cell(paths, args.arm, args.step, args.k, args.contract,
                                   verify_hashes=args.verify_hashes)
    for p in problems:
        print(f"  !! {p}")
    if problems:
        print(f"REJECTED {args.arm} S{args.step} K{args.k} [{args.contract}]: {len(problems)} "
              f"problem(s) across {len(paths)} file(s)")
        return 1
    print(f"VALIDATED {args.arm} S{args.step} K{args.k} [{args.contract}]: {len(rows)} row(s), "
          f"seeds {sorted(r['seed'] for r in rows)}, ckpt {os.path.basename(rows[0]['ckpt_path'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
