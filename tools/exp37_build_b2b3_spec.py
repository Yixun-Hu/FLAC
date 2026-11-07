#!/usr/bin/env python3
"""exp_37 B2 / B3 aggregation spec builder (Codex code review 1, BLOCKING 5): the deterministic, complete spec.

From the fixed inputs (``--inputs``: pilots, audits, nested manifest, selector vectors, the latency selection /
selector, the exp_23 accuracy and the bundle latency CSV) and the runs root, it enumerates EVERY registered run by its
registered stem -- B2 ``b2_{0p50,0p75}_{P1,BF}_s{42,43,44}`` (seed 42 the timing pass + ``_replay``; 43 / 44
accuracy-only, D-a), ``b2t_{nested,c2f}_{arm}_s42`` + ``_replay`` (D-c), ``b2a_c2f_{arm}_s{43,44}`` (the c2f accuracy
runs); B3 ``b3_{o0,o0_180,c4}_t{0,90}_s{seed}``
(C4 theta 0 is B2's B-F 0.50) and the latency passes ``b3_lat_o0_s42`` / ``b3_lat_o0_180_s42`` / ``b1_k8_BF_s42`` --
plus the B3 synthesis section (``--synthesis``: the section ``tools/exp37_build_synthesis_cells.py`` builds at runbook
step B3S.cells from the leaf's 8 new cells and the 4 reused exp_35 cells, else the PENDING placeholder) and checks it
with ``exp37_aggregate_b2b3.check_spec``. Missing run directories are refused unless ``--plan`` (the planned spec);
``--require-timing-replays`` is mandatory (D-c). Write-once output.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix rounds 1-2.
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
    "PENDING -- the 8 new eval_FLAC synthesis cells ({0}, {0,180} x theta {0, 90} x K {1, 8}, seed 42, no-flash) run "
    "by the leaf chain b3 (exp-37-b3-synthesis at d3013734, runbook step B3S.run); the 4 C4 cells are reused from "
    "exp_35 phase (i) (H35 5b065801). Runbook step B3S.cells (tools/exp37_build_synthesis_cells.py) validates all 12 "
    "against their embedded identities and writes the complete section; until then the B2 / B3 summary is "
    "diagnostic only.")
B3_STEM = {"0": "o0", "0_180": "o0_180", "c4": "c4"}
LATENCY_STEMS = {"0": "b3_lat_o0_s42", "0_180": "b3_lat_o0_180_s42", "c4": "b1_k8_BF_s42"}


def b2_stem(arm: str, grid: str, seed: int) -> str:
    return f"b2_{grid.replace('.', 'p')}_{arm}_s{seed}"


def build_spec(inputs: dict, runs_root, *, synthesis=None, check_exists: bool = True) -> dict:
    """The schema-3 spec (D-a: seed 42 is the only timing pass of a grid; D-c: every timing pass has its replay)."""
    root = Path(runs_root)

    def run(stem):
        return str(root / stem)
    b2 = [{"arm": arm, "grid": g, "seed": seed, "timing": seed == 42, "dir": run(b2_stem(arm, g, seed)),
           "replay": run(f"{b2_stem(arm, g, seed)}_replay") if seed == 42 else None}
          for arm in tool.ARMS for g in tool.B2_GRIDS for seed in tool.SEEDS]
    timing = [{"arm": arm, "kind": kind, "dir": run(stem), "replay": run(f"{stem}_replay")} for arm in tool.ARMS
              for kind, stem in (("nested_1m", f"b2t_nested_{arm}_s42"), ("c2f", f"b2t_c2f_{arm}_s42"))]
    c2f_runs = [{"arm": arm, "seed": seed, "dir": run(f"b2a_c2f_{arm}_s{seed}")} for arm in tool.ARMS
                for seed in tool.C2F_ACCURACY_SEEDS]
    b3 = [{"orbit": o, "theta": t, "seed": s, "dir": run(f"b3_{B3_STEM[o]}_t{t}_s{s}")} for o in tool.B3_ORBITS
          for t in tool.B3_THETAS for s in tool.SEEDS if (o, t) != ("c4", 0)]
    latency = {"selection": inputs["latency_selection"], "selector": inputs["latency_selector"],
               "runs": [{"orbit": o, "dir": run(stem), "replay": run(f"{stem}_replay")} for o, stem in LATENCY_STEMS.items()]}
    section = json.loads(Path(synthesis).read_text()) if synthesis else {
        "status": "pending", "leaf_commit": None, "runbook": None, "metrics_keys": [], "cells": [],
        "note": SYNTHESIS_PLACEHOLDER}
    spec = {"schema_version": tool.SPEC_SCHEMA, **{key: inputs[key] for key in tool.INPUT_KEYS}, "b2_runs": b2,
            "b2_timing": timing, "b2_c2f_runs": c2f_runs, "b3_runs": b3, "b3_latency": latency, "b3_synthesis": section,
            "external": {"accuracy": inputs["external_accuracy"], "latency_csv": inputs["external_latency_csv"]},
            "require_timing_replays": True}
    if check_exists:
        dirs = [i[k] for i in b2 + timing + latency["runs"] for k in ("dir", "replay") if i.get(k)] + [
            i["dir"] for i in b3 + c2f_runs]
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
    parser.add_argument("--require-timing-replays", action="store_true",
                        help="required: every timing pass has its artifact replay (D-c; no exp_37 exemption)")
    args = parser.parse_args(argv)
    if not args.require_timing_replays:
        print("REFUSED: --require-timing-replays is mandatory (announcement 08; no written exp_37 exemption)",
              file=sys.stderr)
        return 2
    if args.output.exists():
        print(f"REFUSED: {args.output} exists (specs are written once)", file=sys.stderr)
        return 2
    try:
        spec = build_spec(json.loads(args.inputs.read_text()), args.runs_root, synthesis=args.synthesis,
                          check_exists=not args.plan)
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
