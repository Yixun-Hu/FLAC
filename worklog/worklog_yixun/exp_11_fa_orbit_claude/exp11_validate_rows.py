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
import json
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


def _load_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


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


def validate_row(metrics_path):
    """Validate ONE metrics JSON. Returns ``(row_info, problems)``."""
    problems = []
    try:
        rec = _load_json(metrics_path)
    except Exception as exc:
        return {}, [f"{os.path.basename(metrics_path)}: unreadable metrics JSON: {exc}"]

    side_path = sidecar_path_for(metrics_path)
    if not os.path.isfile(side_path):
        return {}, [f"{os.path.basename(metrics_path)}: screen sidecar missing "
                    f"({os.path.basename(side_path)}) — the protocol cannot be proven"]
    try:
        side = _load_json(side_path)
    except Exception as exc:
        return {}, [f"{os.path.basename(metrics_path)}: unreadable sidecar: {exc}"]

    tag = os.path.basename(metrics_path)
    try:
        ident = parse_eval_name(side.get("eval_name", ""))
    except ValueError as exc:
        return {}, [f"{tag}: {exc}"]

    arm, step, seed, k = ident["arm"], ident["step"], ident["seed"], ident["K"]
    want_angles = orbit_for(arm)

    # --- the sidecar must agree with its own eval name -----------------------
    for field, want in (("arm", arm), ("step", step), ("seed", seed), ("K", k)):
        got = side.get(field)
        if got != want:
            problems.append(f"{tag}: sidecar {field}={got!r} != {want!r} from the eval name")

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

    # --- the metrics record itself -------------------------------------------
    if not rec.get("metrics"):
        problems.append(f"{tag}: metrics block is empty")
    if rec.get("cond_method") != "fa_invariant":
        problems.append(f"{tag}: cond_method={rec.get('cond_method')!r} != 'fa_invariant'")
    if not _angles_equal(rec.get("frame_avg_angles"), want_angles):
        problems.append(f"{tag}: frame_avg_angles={rec.get('frame_avg_angles')!r} is not the "
                        f"exact C{ARM_ORBITS[arm]} orbit of {arm}")
    if rec.get("cond_autocast") != EXPECTED_AUTOCAST:
        problems.append(f"{tag}: cond_autocast={rec.get('cond_autocast')!r} != {EXPECTED_AUTOCAST!r}")
    if rec.get("orbit_execution") != ORBIT_EXECUTION:
        problems.append(f"{tag}: orbit_execution={rec.get('orbit_execution')!r} != "
                        f"{ORBIT_EXECUTION!r} — legacy-loop rows are not comparable with these")
    if rec.get("frame_avg_fwd_cap") != FRAME_AVG_MAX_FWD_SAMPLES:
        problems.append(f"{tag}: frame_avg_fwd_cap={rec.get('frame_avg_fwd_cap')!r} != "
                        f"{FRAME_AVG_MAX_FWD_SAMPLES}")
    if not rec.get("source_sha"):
        problems.append(f"{tag}: source_sha is missing/empty")

    # --- sidecar vs record: they must tell the same story ---------------------
    for field in ("cond_method", "cond_autocast"):
        if field in side and side[field] != rec.get(field):
            problems.append(f"{tag}: sidecar and metrics disagree on {field} "
                            f"({side[field]!r} vs {rec.get(field)!r})")
    if "frame_avg_angles" in side and not _angles_equal(side["frame_avg_angles"], want_angles):
        problems.append(f"{tag}: sidecar frame_avg_angles are not {arm}'s exact orbit")

    # --- the checkpoint -------------------------------------------------------
    ckpt = str(rec.get("ckpt_path", ""))
    if side.get("ckpt_path") and side["ckpt_path"] != ckpt:
        problems.append(f"{tag}: sidecar and metrics disagree on ckpt_path")
    if f"step={step}." not in os.path.basename(ckpt) and not os.path.basename(ckpt).endswith(
            f"step={step}.ckpt"):
        problems.append(f"{tag}: ckpt {os.path.basename(ckpt)} is not step {step}")
    prefix = ARM_RUN_PREFIX[arm]
    if prefix not in ckpt.replace("\\", "/"):
        problems.append(f"{tag}: ckpt {ckpt} is not under this arm's own run directory ({prefix})")

    # --- the filename must not contradict the sidecar -------------------------
    f_steps, f_cfg, f_name = _filename_protocol(metrics_path)
    if f_steps is not None:
        if f_steps != side.get("steps"):
            problems.append(f"{tag}: filename steps {f_steps} != sidecar {side.get('steps')}")
        if f_cfg != side.get("cfg_scale"):
            problems.append(f"{tag}: filename cfg_scale {f_cfg} != sidecar {side.get('cfg_scale')}")
        if f_name and f_name != side.get("eval_name"):
            problems.append(f"{tag}: filename eval name {f_name!r} != sidecar "
                            f"{side.get('eval_name')!r}")

    row = {"path": metrics_path, "arm": arm, "cell": ident["cell"], "step": step,
           "seed": seed, "K": k, "ckpt_path": ckpt, "metrics": rec.get("metrics", {}),
           "source_sha": rec.get("source_sha"), "eval_name": side.get("eval_name")}
    return row, problems


def validate_cell(metrics_paths, arm, step, expected_seeds, k):
    """Validate a whole cell: every row valid, and each expected seed exactly once."""
    rows, problems = [], []
    for path in metrics_paths:
        row, probs = validate_row(path)
        problems.extend(probs)
        if row:
            rows.append(row)

    for row in rows:
        if row["arm"] != arm:
            problems.append(f"{os.path.basename(row['path'])}: arm {row['arm']} does not belong "
                            f"to the {arm} cell")
        if row["step"] != step:
            problems.append(f"{os.path.basename(row['path'])}: step {row['step']} != {step}")
        if row["K"] != k:
            problems.append(f"{os.path.basename(row['path'])}: K={row['K']} != {k}")

    seen = {}
    for row in rows:
        seen.setdefault(row["seed"], []).append(row["path"])
    for want in expected_seeds:
        n = len(seen.get(want, []))
        if n == 0:
            problems.append(f"{arm} S{step} K{k}: seed {want} is missing")
        elif n > 1:
            problems.append(f"{arm} S{step} K{k}: seed {want} appears more than once "
                            f"({[os.path.basename(p) for p in seen[want]]})")
    for got in sorted(seen):
        if got not in expected_seeds:
            problems.append(f"{arm} S{step} K{k}: unexpected seed {got}")
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate exp_11 metric rows before they enter a table")
    ap.add_argument("--arm", required=True, choices=sorted(ARM_ORBITS))
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", default="42", help="comma-separated expected seeds (default 42)")
    ap.add_argument("paths", nargs="+", help="metrics JSONs (globs are expanded)")
    args = ap.parse_args(argv)

    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    rows, problems = validate_cell(paths, args.arm, args.step, seeds, args.k)
    for p in problems:
        print(f"  !! {p}")
    if problems:
        print(f"REJECTED {args.arm} S{args.step} K{args.k}: {len(problems)} problem(s) across "
              f"{len(paths)} file(s)")
        return 1
    print(f"VALIDATED {args.arm} S{args.step} K{args.k}: {len(rows)} row(s), seeds "
          f"{sorted(r['seed'] for r in rows)}, ckpt {os.path.basename(rows[0]['ckpt_path'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
