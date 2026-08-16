#!/usr/bin/env python3
"""exp_15 — the plan §6.9 model-row transaction, executable (integrative review F7).

§6.9 registered a TRIGGER in prose — "when the YAWAUG T block reaches 5/5 seeds at
both K and the §5 gates for those cells pass, the row spec lands, regenerate,
commit, push" — with nothing that could execute or test it. This is that path.

Two things it does, and one it deliberately does not:

* ``ready`` evaluates a **T-ONLY** readiness predicate. The global collector stays
  PENDING until the R blocks land, and waiting for those would delay a row that is
  already earned: the row publishes the θ=0 Table-1 numbers, so what must be
  complete is the YAWAUG T block at both K, plus the gates that bear on those
  cells (G4 admission and the T-scoped part of G3). G1/G2/G5 concern the V and R
  blocks and are reported but do not gate this row.
* ``verify`` re-derives the two row specs' evidence and checks that the shared
  generator would publish exactly the §13-routed numbers.
* It does NOT edit ``gen_model_comparison.py``. The two exp_15 row specs are
  already registered there (additively, ``contract="exp15"``), so publishing is
  "regenerate + commit + push" — a decision for the operator, not a side effect of
  a readiness check.

Usage::

    python3 yaw_aug_publish_row.py ready  --output-root outputs_FLAC --pin <sha>
    python3 yaw_aug_publish_row.py verify --output-root outputs_FLAC --pin <sha>
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKLOG = os.path.dirname(_HERE)
for _p in (_HERE, _WORKLOG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp15_validate_cell as V              # noqa: E402
import yaw_aug_collect as C                  # noqa: E402

ROW_LABEL = "yaw-aug vanilla YAWAUG @40k (exp_15)"
ROW_ARM = "YAWAUG"
ROW_CELL = C.T_BLOCK
# The gates that bear on a T-only row. G1 (VANL@90 control), G2 (the R probe's
# golden draw) and G5 (all eight T+R blocks) concern cells this row does not
# publish; they are reported for context and do not gate it.
ROW_GATES = ("G3", "G4")


def row_specs():
    """The two registered specs, read from the SHARED generator, not restated."""
    import importlib.util
    path = os.path.join(_WORKLOG, "gen_model_comparison.py")
    spec = importlib.util.spec_from_file_location("gen_model_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [r for r in module.ROWS if len(r) > 4 and r[4] == "exp15"], module


def readiness(output_root, pin=None, expected_count=V.EXPECTED_COUNT,
              control=None, registry=None):
    """``{ready, reasons, seeds, gates}`` — the T-only §6.9 trigger, evaluated."""
    cells, _missing, rejected = C.collect_cells(output_root, pin=pin,
                                                expected_count=expected_count)
    reasons = []
    seeds = {}
    for k in V.KS:
        got = sorted(int(a.cell.seed) for a in cells
                     if a.cell.arm == ROW_ARM and a.cell.cell == ROW_CELL
                     and int(a.cell.k) == int(k))
        seeds[str(k)] = got
        missing = [s for s in V.SEEDS if s not in got]
        if missing:
            reasons.append(f"YAWAUG T K={k}: {len(got)}/5 seeds on disk, missing {missing}")
    bad = [name for name, why in ((V.eval_name(c), w) for c, w in rejected)
           if name.startswith(f"exp15_{ROW_ARM}_{ROW_CELL}")]
    if bad:
        reasons.append(f"{len(bad)} YAWAUG T cell(s) failed validation: {bad[:2]}")

    # BOTH gates are evaluated T-SCOPED. The global G3/G4 obligation sets span the
    # R cells, which by construction have not run when this predicate is asked —
    # the whole point of a T-only trigger is that the θ=0 row is earned before the
    # robustness campaign finishes (plan §6.9). What is required is that nothing
    # touching a T cell is wrong.
    gates = {"G3": C.gate_g3(cells), "G4": C.gate_g4(cells, control, registry)}
    t_violations = [v for v in gates["G3"].get("violations", [])
                    if v.get("cell_class") == C.T_BLOCK]
    if t_violations:
        reasons.append(f"G3: {len(t_violations)} hash violation(s) touch the T block")
    # G4, T-scoped: every landed YAWAUG T cell must carry the admitted digest.
    try:
        admitted = V.admission_expectation(
            ROW_ARM, control or V.CONTROL_ADMISSION,
            registry or V.LAUNCH_REGISTRY)["sha256"]
    except ValueError as exc:
        reasons.append(f"G4: {ROW_ARM} has no admission record yet ({exc})")
        admitted = None
    if admitted is not None:
        for art in cells:
            if art.cell.arm != ROW_ARM or art.cell.cell != ROW_CELL:
                continue
            if art.meta.get("ckpt_sha256") != admitted:
                reasons.append(f"G4: {V.eval_name(art.cell)} did not evaluate the "
                               "admitted checkpoint")
    return {"ready": not reasons, "reasons": reasons, "seeds": seeds,
            "gates": {n: gates[n]["status"] for n in ROW_GATES},
            "label": ROW_LABEL}


def verify(output_root, pin=None, expected_count=V.EXPECTED_COUNT,
           control=None, registry=None):
    """What the generator WOULD publish, recomputed here from the same raws."""
    specs, module = row_specs()
    out = {"specs": [], "ready": readiness(output_root, pin=pin,
                                           expected_count=expected_count,
                                           control=control, registry=registry)}
    import glob
    for label, note, k, patterns, contract in specs:
        files = sorted(set(sum((glob.glob(os.path.join(output_root, "..", p),
                                          recursive=True) for p in patterns), [])))
        files = [f for f in files if not f.endswith((".screenmeta.json", ".stream.json"))]
        entry = {"label": label, "note": note, "K": k, "contract": contract,
                 "files": len(files)}
        if len(files) < module.MIN_SEEDS:
            entry["status"] = f"pending ({len(files)}/{module.MIN_SEEDS} seeds on disk)"
        else:
            ok, problems = module.validate_exp15_cell(files, expected_k=k)
            if not ok:
                entry["status"] = "BLOCKED"
                entry["problems"] = problems[:3]
            else:
                values, n = module.agg_files_exp15(files)
                entry["status"] = "READY"
                entry["values"] = {m: [round(mean, 4), round(sd, 4)]
                                   for m, (mean, sd) in values.items()}
        out["specs"].append(entry)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("ready", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--output-root", required=True)
        sp.add_argument("--pin", default=None)
        sp.add_argument("--expected-count", type=int, default=V.EXPECTED_COUNT)
        sp.add_argument("--control", default=None)
        sp.add_argument("--registry", default=None)
        sp.set_defaults(cmd=name)
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    fn = readiness if args.cmd == "ready" else verify
    result = fn(args.output_root, pin=args.pin, expected_count=args.expected_count,
                control=args.control, registry=args.registry)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    ready = result["ready"] if args.cmd == "ready" else result["ready"]["ready"]
    if not ready:
        print("\nNOT READY — the §6.9 transaction must not fire yet.", file=sys.stderr)
        return 1
    print("\nREADY — regenerate model_comparison.md, then commit and push "
          "(announcement 04). Do NOT repin a running eval campaign.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
