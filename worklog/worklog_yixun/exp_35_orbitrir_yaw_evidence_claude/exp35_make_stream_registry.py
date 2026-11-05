#!/usr/bin/env python3
"""exp_35 (plan_orbitrir_yaw_evidence.md r2 §5 item 5): freeze ``exp35_exp25_stream_registry.json`` -- the item,
context and rotation streams of exp_25's VERIFIED unseen 40k cells of the three arms (P1-pin, CANON, YAW-pin;
T, R, C4+45), which agree across arms and are what every exp_35 cell must reproduce:

  input_hash[K][seed]            the T and R cells (a rotation does not enter the input tuples)
  assignment_hash_random[seed]   the R cells (K does not enter the assignment tuples)
  assignment_hash_fixed[deg]     0 / 90 / 180 / 270 / 45 from the cells; 135 / 225 DERIVED from the split's
                                 target order and eval_FLAC's yaw_column_shift -- the rule is first checked to
                                 reproduce every observed angle

Each source cell must pass ``verify_cell`` at the exp_25 pin, OFFLINE (``dump_check="receipt"``,
``check_link=False``) against the exp-25 worktree's artifacts. Read-only on that worktree; writes only --out.

  --write   (re)write the registry      --check   exit 1 if the committed registry drifted from a rebuild

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
COLLECT = os.path.join(REPO, "worklog", "worklog_yixun", "exp_25_depth_canon_claude", "depth_canon_collect.py")
OUT = os.path.join(HERE, "exp35_exp25_stream_registry.json")
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
SOURCE_BLOCKS = ("T", "C4+45", "R", "P1-pin", "YAW-pin")
SEEDS = (42, 43, 44, 45, 46)
OBSERVED, DERIVED = (0, 90, 180, 270, 45), (135, 225)


def collector():
    spec = importlib.util.spec_from_file_location("depth_canon_collect_registry", COLLECT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shift_of(deg):
    """The constant column shift of a fixed rotation, by eval_FLAC's own rule."""
    import eval_FLAC
    return eval_FLAC.yaw_column_shift(math.radians(deg), 512)


def _one(bucket, label):
    if len(bucket) != 1:
        raise ValueError(f"{label}: the exp_25 cells disagree ({sorted(bucket)})")
    return next(iter(bucket))


def build_registry(C, manifest, *, base_dir, canon_index, pin=EXP25_PIN, farm_root=None, dump_root=None):
    """The registry dict; ``ValueError`` when a source cell is missing / does not verify or the cells disagree."""
    farm, dump_root = farm_root or manifest["farm_root"], dump_root or manifest["dump_root"]
    rows = [c for c in manifest["cells"] if c["block"] in SOURCE_BLOCKS and c["split"] == "unseen"
            and c["step"] == 40000 and c["subblock"] in ("T", "R", "C4")]
    inp, rnd, fixed, order, cells = {}, {}, {}, set(), {}
    for row in rows:
        target, _ = C.expected_checkpoint(row, manifest, canon_index)
        p = C.cell_paths(row, os.path.join(farm, row["arm"], os.path.basename(target)), dump_root)
        reasons = C.verify_cell(p.stem, manifest=manifest, farm_root=farm, dump_root=dump_root, pin=pin,
                                canon_index=canon_index, base_dir=base_dir, dump_check="receipt", check_link=False)
        if reasons:
            raise ValueError(f"{row['cell']} does not verify at {pin[:12]}: {reasons[:2]}")
        with open(os.path.join(base_dir, p.stream), "rb") as f:
            raw = f.read()
        s, cells[row["cell"]] = json.loads(raw), hashlib.sha256(raw).hexdigest()
        inp.setdefault((row["K"], row["seed"]), set()).add(s["input_hash"])
        bucket = rnd.setdefault(row["seed"], set()) if row["rotate_mode"] == "random" else fixed.setdefault(
            row["rotate_deg"], set())
        bucket.add(s["assignment_hash"])
        order.add(tuple(t[1] for t in s["assignment_tuples"]))
    targets = _one(order, "the split's target order")
    derive = lambda deg: C.eval_FLAC.canonical_stream_hash([[i, t, shift_of(deg)] for i, t in enumerate(targets)])  # noqa
    observed = {deg: _one(fixed.get(deg, set()), f"assignment_hash at {deg} deg") for deg in OBSERVED}
    bad = [deg for deg in OBSERVED if derive(deg) != observed[deg]]
    if bad:
        raise ValueError(f"the derivation rule does not reproduce the observed fixed assignments at {bad} deg")
    return {"schema_version": 1,
            "source": {"pin": pin, "helper": os.path.relpath(os.path.abspath(__file__), REPO), "cells": cells},
            "input_hash": {str(k): {str(s): _one(inp.get((k, s), set()), f"input_hash K{k} s{s}") for s in SEEDS}
                           for k in (1, 8)},
            "assignment_hash_random": {str(s): _one(rnd.get(s, set()), f"assignment_hash random s{s}") for s in SEEDS},
            "assignment_hash_fixed": {str(d): observed[d] if d in observed else derive(d) for d in OBSERVED + DERIVED},
            "derived": {"angles": list(DERIVED),
                        "rule": "canonical_stream_hash([[i, target_i, yaw_column_shift(deg, 512)] for the split's "
                                "target order]) -- reproduces every observed angle"}}


def render(registry):
    return json.dumps(registry, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", default=os.path.expanduser("~/codespace/exp-25-depth-canon"))
    ap.add_argument("--cells", default=None)
    ap.add_argument("--pin", default=EXP25_PIN)
    ap.add_argument("--farm-root", default=None)
    ap.add_argument("--dump-root", default=None)
    ap.add_argument("--canon-ckpt-manifest", default=None)
    ap.add_argument("--out", default=OUT)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    C = collector()
    manifest = C.load_cells(args.cells or C.CELLS_JSON)
    index = C.load_canon_index(args.canon_ckpt_manifest or C.find_canon_manifest(),
                               manifest["arms"]["CANON"]["ckpt"]["run_ns"])
    text = render(build_registry(C, manifest, base_dir=args.base_dir, canon_index=index, pin=args.pin,
                                 farm_root=args.farm_root, dump_root=args.dump_root))
    if args.check:
        with open(args.out) as f:
            same = f.read() == text
        print("stream registry OK" if same else f"stream registry DRIFTED from a rebuild: {args.out}")
        return 0 if same else 1
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
