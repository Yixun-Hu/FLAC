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
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data.yaw_rotation import (  # noqa: E402
    FRAME_AVG_MAX_FWD_SAMPLES, ORBIT_EXECUTION)

# arm -> orbit size. C4BACKFILL is the exp_07 B-F lineage screened under the fa
# protocol (plan §3); it is C4 by construction.
ARM_ORBITS = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32, "C4BACKFILL": 4}
ARM_RUN_PREFIX = {
    "C4L": "outputs_FLAC/exp11_C4L/", "C8": "outputs_FLAC/exp11_C8/",
    "C16": "outputs_FLAC/exp11_C16/", "C32": "outputs_FLAC/exp11_C32/",
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
# The six metrics every comparison row reports (gen_model_comparison.KEYS).
REQUIRED_METRIC_KEYS = ("T60", "C50", "EDT",
                        "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10")
MANDATORY_SIDECAR_FIELDS = (
    "arm", "step", "seed", "K", "eval_name", "cfg_scale", "steps", "model_config",
    "model_config_sha256", "dataset_config", "ckpt_path", "ckpt_sha256", "use_ema",
    "frame_avg_angles", "cond_method", "cond_autocast", "commit",
)

# Purpose-specific contracts (round-4 review B4): the seed policy is REGISTERED,
# never supplied by the caller, and an R3 rotation row is never table-admissible.
CONTRACTS = {
    "futility": {"cells": ("screen", "backfill"), "seeds": (42,), "K": (8,),
                 "table_admissible": False},
    "table":    {"cells": ("conf",), "seeds": (42, 43, 44, 45, 46), "K": (1, 8),
                 "table_admissible": True},
    "r3":       {"cells": ("r3",), "seeds": (42,), "K": (8,), "table_admissible": False},
}
# Provenance that must be IDENTICAL across every row of a cell.
CELL_IDENTITY_FIELDS = ("cell", "ckpt_path", "ckpt_sha256", "model_config_sha256",
                        "source_sha", "commit", "orbit_execution")
_SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")

_SCREEN_RE = re.compile(r"^exp11_(C4L|C8|C16|C32)_(screen|conf|r3)_S(\d+)_s(\d+)_K(\d+)$")
_BACKFILL_RE = re.compile(r"^exp11_C4backfill_S(\d+)_s(\d+)_K(\d+)$")


def orbit_for(arm):
    """The exact uniform orbit an arm's rows must carry."""
    if arm not in ARM_ORBITS:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARM_ORBITS)}")
    n = ARM_ORBITS[arm]
    return [k * 360.0 / n for k in range(n)]


def parse_eval_name(name):
    """Parse the plan §4 eval-name schema into its identity fields."""
    m = _SCREEN_RE.match(name or "")
    if m:
        return {"arm": m.group(1), "cell": m.group(2), "step": int(m.group(3)),
                "seed": int(m.group(4)), "K": int(m.group(5))}
    m = _BACKFILL_RE.match(name or "")
    if m:
        return {"arm": "C4BACKFILL", "cell": "backfill", "step": int(m.group(1)),
                "seed": int(m.group(2)), "K": int(m.group(3))}
    raise ValueError(f"eval name {name!r} does not match the exp_11 schema "
                     "(exp11_<arm>_<cell>_S<step>_s<seed>_K<k> or exp11_C4backfill_S..)")


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
    want_angles = orbit_for(arm)

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

    # --- the metrics themselves ----------------------------------------------
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict):
        problems.append(f"{tag}: metrics block is missing or not a mapping")
    else:
        absent = [m for m in REQUIRED_METRIC_KEYS if m not in metrics]
        if absent:
            problems.append(f"{tag}: metrics missing {absent}")
        bad = [m for m in REQUIRED_METRIC_KEYS if m in metrics and not _finite_number(metrics[m])]
        if bad:
            problems.append(f"{tag}: metrics {bad} are not finite numbers")

    # --- the record's own protocol statement ---------------------------------
    if rec.get("cond_method") != "fa_invariant":
        problems.append(f"{tag}: cond_method={rec.get('cond_method')!r} != 'fa_invariant'")
    if not _angles_equal(rec.get("frame_avg_angles"), want_angles):
        problems.append(f"{tag}: frame_avg_angles={rec.get('frame_avg_angles')!r} is not the "
                        f"exact C{ARM_ORBITS[arm]} orbit of {arm}")
    if rec.get("cond_autocast") != EXPECTED_AUTOCAST:
        problems.append(f"{tag}: cond_autocast={rec.get('cond_autocast')!r} != {EXPECTED_AUTOCAST!r}")
    if rec.get("rotate_deg") != 0.0 and cell != "r3":
        problems.append(f"{tag}: rotate_deg={rec.get('rotate_deg')!r} — a rotated evaluation is "
                        "not a screen/table row")
    if rec.get("weights_source") != EXPECTED_WEIGHTS:
        problems.append(f"{tag}: weights_source={rec.get('weights_source')!r} != {EXPECTED_WEIGHTS!r} "
                        "— EMA must be proven by the evaluator, not asserted by the driver")
    if rec.get("n_samples") != EXPECTED_N_SAMPLES:
        problems.append(f"{tag}: n_samples={rec.get('n_samples')!r} != {EXPECTED_N_SAMPLES} "
                        "(the full unseen split)")
    if rec.get("orbit_execution") != ORBIT_EXECUTION:
        problems.append(f"{tag}: orbit_execution={rec.get('orbit_execution')!r} != "
                        f"{ORBIT_EXECUTION!r} — legacy-loop rows are not comparable with these")
    if rec.get("frame_avg_fwd_cap") != FRAME_AVG_MAX_FWD_SAMPLES:
        problems.append(f"{tag}: frame_avg_fwd_cap={rec.get('frame_avg_fwd_cap')!r} != "
                        f"{FRAME_AVG_MAX_FWD_SAMPLES}")
    src = str(rec.get("source_sha") or "")
    if not _SHA_RE.match(src):
        problems.append(f"{tag}: source_sha={rec.get('source_sha')!r} is not a commit sha")

    # --- record vs sidecar: one story, field by field -------------------------
    for field in ("cond_method", "cond_autocast", "seed", "cfg_scale", "steps", "eval_name"):
        if field in side and rec.get(field) is not None and side[field] != rec.get(field):
            problems.append(f"{tag}: sidecar and metrics disagree on {field} "
                            f"({side[field]!r} vs {rec.get(field)!r})")
    if rec.get("dataset_config") is not None and os.path.normpath(str(rec["dataset_config"])) != \
            os.path.normpath(str(side.get("dataset_config", ""))):
        problems.append(f"{tag}: sidecar and metrics disagree on dataset_config")
    if not _angles_equal(side.get("frame_avg_angles"), want_angles):
        problems.append(f"{tag}: sidecar frame_avg_angles are not {arm}'s exact orbit")

    # --- the checkpoint -------------------------------------------------------
    ckpt = str(rec.get("ckpt_path", ""))
    if side.get("ckpt_path") != ckpt:
        problems.append(f"{tag}: sidecar and metrics disagree on ckpt_path")
    if not os.path.basename(ckpt).endswith(f"step={step}.ckpt"):
        problems.append(f"{tag}: ckpt {os.path.basename(ckpt)} is not step {step}")
    prefix = ARM_RUN_PREFIX[arm]
    if prefix not in ckpt.replace("\\", "/"):
        problems.append(f"{tag}: ckpt {ckpt} is not under this arm's own run directory ({prefix})")
    for field in ("model_config_sha256", "ckpt_sha256", "commit"):
        if not _SHA_RE.match(str(side.get(field) or "")):
            problems.append(f"{tag}: sidecar {field}={side.get(field)!r} is not a hash/commit")

    # --- hashes are recomputed, not trusted (round-4 review B3/B6) ------------
    if verify_hashes:
        for field, path_field in (("model_config_sha256", "model_config"),
                                  ("ckpt_sha256", "ckpt_path")):
            target = str(side.get(path_field, ""))
            if not os.path.isfile(target):
                problems.append(f"{tag}: cannot recompute {field}: {target} is not readable")
                continue
            got = _cached_sha256(target)
            if got != side.get(field):
                problems.append(f"{tag}: {field} mismatch — sidecar {side.get(field)}, "
                                f"file {got} ({target})")

    # --- the filename must be EXACTLY what build_output_paths generates --------
    want_name = _expected_filename(ckpt, side, arm, rec.get("rotate_deg", 0.0))
    if want_name is None:
        problems.append(f"{tag}: cannot derive the expected filename (bad ckpt/sidecar)")
    elif os.path.basename(metrics_path) != want_name:
        problems.append(f"{tag}: filename is not the one build_output_paths would generate "
                        f"({want_name})")

    row = {"path": metrics_path, "arm": arm, "cell": cell, "step": step,
           "seed": seed, "K": k, "ckpt_path": ckpt, "metrics": metrics if isinstance(metrics, dict) else {},
           "source_sha": rec.get("source_sha"), "eval_name": side.get("eval_name"),
           "ckpt_sha256": side.get("ckpt_sha256"),
           "model_config_sha256": side.get("model_config_sha256"),
           "commit": side.get("commit"), "orbit_execution": rec.get("orbit_execution")}
    return row, problems


def _expected_filename(ckpt_path, side, arm, rotate_deg):
    """The filename ``eval_FLAC.build_output_paths`` would produce for this row."""
    try:
        from eval_FLAC import build_output_paths
        paths = build_output_paths(ckpt_path, int(side["steps"]), float(side["cfg_scale"]),
                                   str(side["eval_name"]), cond_method="fa_invariant",
                                   rotate_deg=float(rotate_deg or 0.0),
                                   n_angles=ARM_ORBITS[arm])
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

    # --- the registered seed set, exactly once each --------------------------
    seen = {}
    for row in rows:
        seen.setdefault(row["seed"], []).append(row["path"])
    for want in spec["seeds"]:
        n = len(seen.get(want, []))
        if n == 0:
            problems.append(f"{arm} S{step} K{k} [{contract}]: seed {want} is missing")
        elif n > 1:
            problems.append(f"{arm} S{step} K{k} [{contract}]: seed {want} appears more than once "
                            f"({[os.path.basename(p) for p in seen[want]]})")
    for got in sorted(seen):
        if got not in spec["seeds"]:
            problems.append(f"{arm} S{step} K{k} [{contract}]: unexpected seed {got} "
                            f"(the contract registers {spec['seeds']})")

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
    ap.add_argument("--verify-hashes", action="store_true",
                    help="recompute the config/checkpoint sha256 instead of trusting the sidecar")
    ap.add_argument("paths", nargs="+", help="metrics JSONs (globs are expanded)")
    args = ap.parse_args(argv)

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
