#!/usr/bin/env python3
"""exp_37 B2 / B3 aggregation spec builder (Codex code review 1, BLOCKING 5): the deterministic, complete spec.

From the fixed inputs (``--inputs``: pilots, audits, nested manifest, selector vectors, the latency selection /
selector, the exp_23 accuracy and the bundle latency CSV) and the runs root, it enumerates EVERY registered run by its
registered stem -- B2 ``b2_{0p50,0p75}_{P1,BF}_s{42,43,44}`` (seed 42 the timing pass + ``_replay``; 43 / 44
accuracy-only unless ``--seeds-43-44-timing``), ``b2t_{nested,c2f}_{arm}_s42``; B3 ``b3_{o0,o0_180,c4}_t{0,90}_s{seed}``
(C4 theta 0 is B2's B-F 0.50) and the latency passes ``b3_lat_o0_s42`` / ``b3_lat_o0_180_s42`` / ``b1_k8_BF_s42`` --
plus the B3 synthesis section (``--synthesis`` JSON from the exp_36 leaf, else the PENDING placeholder) and checks it
with ``exp37_aggregate_b2b3.check_spec``. Missing run directories are refused unless ``--plan``. Write-once output.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from tools import exp37_aggregate_b2b3 as tool  # noqa: E402

SYNTHESIS_PLACEHOLDER = (
    "PENDING -- the 8 new eval_FLAC synthesis cells ({0}, {0,180} x theta {0, 90} x K {1, 8}, seed 42, no-flash) run on "
    "the exp_36 Coder's H35-compatible leaf; the 4 C4 cells are reused from exp_35 phase (i). The Planner fills "
    "leaf_commit, runbook, metrics_keys and the 12 cells (orbit, theta, k_context, seed, source, metrics_json, "
    "metrics_sha256); until then the B2 / B3 summary is diagnostic only.")
B3_STEM = {"0": "o0", "0_180": "o0_180", "c4": "c4"}
LATENCY_STEMS = {"0": "b3_lat_o0_s42", "0_180": "b3_lat_o0_180_s42", "c4": "b1_k8_BF_s42"}


def b2_stem(arm: str, grid: str, seed: int) -> str:
    return f"b2_{grid.replace('.', 'p')}_{arm}_s{seed}"


def build_spec(inputs: dict, runs_root, *, synthesis=None, check_exists: bool = True, seeds_43_44_timing: bool = False,
               require_timing_replays: bool = False, timing_replays=()) -> dict:
    root = Path(runs_root)

    def run(stem):
        return str(root / stem)

    def optional_replay(stem):
        return run(f"{stem}_replay") if require_timing_replays or stem in timing_replays else None
    b2 = [{"arm": arm, "grid": g, "seed": seed, "timing": seed == 42 or seeds_43_44_timing, "dir": run(b2_stem(arm, g, seed)),
           "replay": run(f"{b2_stem(arm, g, seed)}_replay") if seed == 42 or seeds_43_44_timing else None}
          for arm in tool.ARMS for g in tool.B2_GRIDS for seed in tool.SEEDS]
    timing = [{"arm": arm, "kind": kind, "dir": run(stem), "replay": optional_replay(stem)} for arm in tool.ARMS
              for kind, stem in (("nested_1m", f"b2t_nested_{arm}_s42"), ("c2f", f"b2t_c2f_{arm}_s42"))]
    b3 = [{"orbit": o, "theta": t, "seed": s, "dir": run(f"b3_{B3_STEM[o]}_t{t}_s{s}")} for o in tool.B3_ORBITS
          for t in tool.B3_THETAS for s in tool.SEEDS if (o, t) != ("c4", 0)]
    latency = {"selection": inputs["latency_selection"], "selector": inputs["latency_selector"],
               "runs": [{"orbit": o, "dir": run(stem), "replay": optional_replay(stem)} for o, stem in LATENCY_STEMS.items()]}
    section = json.loads(Path(synthesis).read_text()) if synthesis else {
        "status": "pending", "leaf_commit": None, "runbook": None, "metrics_keys": [], "cells": [],
        "note": SYNTHESIS_PLACEHOLDER}
    spec = {"schema_version": tool.SPEC_SCHEMA, **{key: inputs[key] for key in tool.INPUT_KEYS}, "b2_runs": b2,
            "b2_timing": timing, "b3_runs": b3, "b3_latency": latency, "b3_synthesis": section,
            "external": {"accuracy": inputs["external_accuracy"], "latency_csv": inputs["external_latency_csv"]},
            "require_timing_replays": bool(require_timing_replays)}
    if check_exists:
        dirs = [i[k] for i in b2 + timing + latency["runs"] for k in ("dir", "replay") if i.get(k)] + [i["dir"] for i in b3]
        missing = [d for d in dirs if not Path(d).is_dir()]
        if missing:
            raise FileNotFoundError("missing run directories: " + ", ".join(missing))
    return spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--synthesis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="do not require the run directories (the planned spec)")
    parser.add_argument("--seeds-43-44-timing", action="store_true")
    parser.add_argument("--require-timing-replays", action="store_true")
    parser.add_argument("--timing-replay", action="append", default=[], help="a timing-only stem that has a replay")
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"REFUSED: {args.output} exists (specs are written once)", file=sys.stderr)
        return 2
    try:
        spec = build_spec(json.loads(args.inputs.read_text()), args.runs_root, synthesis=args.synthesis,
                          check_exists=not args.plan, seeds_43_44_timing=args.seeds_43_44_timing,
                          require_timing_replays=args.require_timing_replays, timing_replays=tuple(args.timing_replay))
        tool.check_spec(spec)
    except (FileNotFoundError, tool.SpecError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "b2_runs": len(spec["b2_runs"]), "b3_runs": len(spec["b3_runs"]),
                      "synthesis": spec["b3_synthesis"]["status"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
