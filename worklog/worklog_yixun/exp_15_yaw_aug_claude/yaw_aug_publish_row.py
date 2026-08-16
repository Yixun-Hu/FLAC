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


PUBLISH_CHECKLIST = """\
§6.10 publication — who does what, in order (nothing below is automatic):

 1. REGENERATE (operator, main checkout):
      python3 worklog/worklog_yixun/gen_model_comparison.py
 2. INSPECT the two new rows in worklog/worklog_yixun/model_comparison.md.
    Both K rows must be present and neither may read *pending* or **BLOCKED**.
 3. STAGE exactly these paths (path-scoped; the generator is shared):
      git add worklog/worklog_yixun/model_comparison.md
      git add worklog/worklog_yixun/gen_model_comparison.py
 4. COMMIT, message shape:
      exp_15 results: YAWAUG @40k Table-1 rows (K=1 and K=8), §13-routed
 5. PULL-REBASE THEN PUSH, immediately (announcement 04):
      git pull --rebase && git push
 6. DO NOT repin a running eval campaign. The R wave continues at its own pin.
 7. The remaining §6.10 artifacts are NOT produced by this script and remain
    owed: yaw_aug_results.md, yaw_aug_analysis.md (Planner, incl. the mandatory
    scope-of-inference statement), yaw_aug_01_results.html + assets,
    yaw_aug_params_set_up.md / yaw_aug_command.md eval sections, and
    commits_yaw_aug.md.
"""


def readiness(output_root, pin=None, expected_count=V.EXPECTED_COUNT,
              control=None, registry=None):
    """``{ready, reasons, seeds, gates}`` — the T-only §6.9 trigger, evaluated.

    T-only, but NOT YAWAUG-only (re-review finding 3). The row's validity rests on
    the cross-arm T input-hash equalities, and those cannot be tested from one
    arm's cells: with all ten YAWAUG T cells and no VANL counterparts the
    equalities are simply UNTESTED, which the previous version scored as ready.
    So both arms' T blocks are required at both K, and the T-scoped G3
    obligations are evaluated DIRECTLY rather than by looking for violations that
    could not have been detected.
    """
    cells, _missing, rejected = C.collect_cells(output_root, pin=pin,
                                                expected_count=expected_count)
    reasons = []
    seeds = {}
    for arm in C.ARM_ORDER:                       # YAWAUG and VANL
        for k in V.KS:
            got = sorted(int(a.cell.seed) for a in cells
                         if a.cell.arm == arm and a.cell.cell == ROW_CELL
                         and int(a.cell.k) == int(k))
            seeds[f"{arm}/K{k}"] = got
            missing = [s for s in V.SEEDS if s not in got]
            if missing:
                reasons.append(f"{arm} T K={k}: {len(got)}/5 seeds on disk, "
                               f"missing {missing}")
    bad = [V.eval_name(c) for c, _w in rejected if c.cell == ROW_CELL]
    if bad:
        reasons.append(f"{len(bad)} T cell(s) failed validation: {bad[:2]}")

    # The T-scoped G3 obligations, enumerated and CHECKED (not merely "no
    # violation was observed"): every (K, seed) cross-arm input_hash pair.
    required_t = {o for o in C.g3_obligations() if o[0] == "input_hash"
                  and o[1] == ROW_CELL}
    checked_t = C.g3_checked(cells) & required_t
    if checked_t != required_t:
        reasons.append(f"G3 (T-scoped): only {len(checked_t)}/{len(required_t)} "
                       "cross-arm input_hash equalities have both artifacts present")
    gates = {"G3": C.gate_g3(cells), "G4": C.gate_g4(cells, control, registry)}
    t_violations = [v for v in gates["G3"].get("violations", [])
                    if v.get("cell_class") == ROW_CELL]
    if t_violations:
        reasons.append(f"G3: {len(t_violations)} hash violation(s) touch the T block")
    # G4, T-scoped: every landed T cell of EITHER arm carries its admitted digest.
    for arm in C.ARM_ORDER:
        try:
            admitted = V.admission_expectation(
                arm, control or V.CONTROL_ADMISSION,
                registry or V.LAUNCH_REGISTRY)["sha256"]
        except ValueError as exc:
            reasons.append(f"G4: {arm} has no admission record yet ({exc})")
            continue
        for art in cells:
            if art.cell.arm != arm or art.cell.cell != ROW_CELL:
                continue
            if art.meta.get("ckpt_sha256") != admitted:
                reasons.append(f"G4: {V.eval_name(art.cell)} did not evaluate the "
                               "admitted checkpoint")
    return {"ready": not reasons, "reasons": reasons, "seeds": seeds,
            "gates": {n: gates[n]["status"] for n in ROW_GATES},
            "label": ROW_LABEL, "checklist": PUBLISH_CHECKLIST}


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
    # ONE TRANSACTION over BOTH K rows (re-review finding 3). The generator
    # renders each row independently, so nothing there prevents one K publishing
    # while the other is pending — the pairing has to be enforced here.
    statuses = [e["status"] for e in out["specs"]]
    out["both_k_ready"] = (len(out["specs"]) == 2
                           and all(st == "READY" for st in statuses))
    if not out["both_k_ready"]:
        out["transaction"] = ("REFUSED: both K rows publish together or neither. "
                              f"statuses={statuses}")
    else:
        out["transaction"] = "READY: both K rows are publishable as one transaction"
    out["checklist"] = PUBLISH_CHECKLIST
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
    if args.cmd == "ready":
        ready = result["ready"]
    else:
        ready = result["ready"]["ready"] and result["both_k_ready"]
    if not ready:
        print("\nNOT READY — the §6.9 transaction must not fire yet.", file=sys.stderr)
        return 1
    print(PUBLISH_CHECKLIST, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
