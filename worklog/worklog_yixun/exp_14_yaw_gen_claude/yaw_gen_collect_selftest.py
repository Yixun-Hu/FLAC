#!/usr/bin/env python3
"""exp_14 — run the collector over SYNTHETIC campaigns and save the transcripts.

Four scenarios, chosen because each is a way the campaign can go wrong and each
must produce a *visible refusal* rather than a plausible number:

1. ``complete``  — a fabricated but well-formed 106-cell grid: every section
   renders, gates pass, the H-readouts carry verdicts;
2. ``pending``   — one seed of one block never landed: that block renders
   PENDING and the endpoint contrast that needed it renders PENDING too;
3. ``blocked``   — one arm's rotated cell evaluated a different item stream: the
   §3.3 hash equality fails, G4 FAILs and the cross-arm contrast renders BLOCKED;
4. ``gatefail``  — the VANL@90 positive control does not degrade: G2 FAILs and
   every H-readout is SUPPRESSED.

The fixtures are the ones the test-suite builds (``src/tests/test_yaw_gen_collect``
is imported for its factory), so the transcripts are evidence about the SAME
artifacts the tests pin — not a second, prettier mock. Nothing here touches the
campaign's real output tree: every scenario is written under a temporary
directory that is removed on exit.

Usage
-----
    python3 yaw_gen_collect_selftest.py [--outdir DIR]
"""
import argparse
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "tests"))
sys.path.insert(0, _REPO_ROOT)

import test_yaw_gen_collect as T                                    # noqa: E402

C, V = T.C, T.V


def _scenario_complete(root):
    T.write_grid(root)


def _scenario_pending(root):
    """One seed of C32's rotated K=8 block never landed."""
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 46 and c.k == 8)]
    T.write_grid(root, grid)


def _scenario_blocked(root):
    """C32's rotated seed-43 cell evaluated a DIFFERENT item stream."""
    grid = [c for c in V.expected_grid()
            if not (c.arm == "C32" and c.cell == "rgen" and c.seed == 43 and c.k == 8)]
    T.write_grid(root, grid)
    T.write_cell(root, V.Cell("C32", "rgen", V.STEP, 43, 8, None),
                 targets=[f"wrong/rir_{i}.wav" for i in range(T.COUNT)])


def _scenario_gatefail(root):
    """The VANL@90 positive control reads exactly like VANL@0."""
    grid = [c for c in V.expected_grid() if not (c.arm == "VANL" and c.cell == "vctl")]
    T.write_grid(root, grid)
    T.write_cell(root, V.Cell("VANL", "vctl", V.STEP, 42, 8, 90.0),
                 metrics=T.synthetic_metrics(
                     V.Cell("VANL", "zref", V.STEP, 42, 8, None)))


SCENARIOS = (("complete", _scenario_complete), ("pending", _scenario_pending),
             ("blocked", _scenario_blocked), ("gatefail", _scenario_gatefail))


def run(outdir):
    import json
    written = []
    workdir = tempfile.mkdtemp(prefix="exp14_collect_selftest_")
    try:
        expect = os.path.join(workdir, "ckpt_expect.json")
        with open(expect, "w") as fh:
            json.dump({"step": V.STEP,
                       "arms": {a: {"sha256": s} for a, s in T.CKPT_SHA.items()}}, fh)
        for name, build in SCENARIOS:
            root = os.path.join(workdir, name)
            os.makedirs(root)
            build(root)
            out = os.path.join(outdir, f"yaw_gen_collect_selftest_{name}.txt")
            rc = C.main(["--output-root", root, "--pin", T.PIN,
                         "--expected-count", str(T.COUNT), "--ckpt-expect", expect,
                         "--out", out])
            # The exit code is part of the transcript: a wrapper reads it, and a
            # scenario that "printed something" while claiming success would be
            # the failure mode this self-test exists to make visible.
            with open(out, "a") as fh:
                fh.write(f"\n<!-- selftest scenario: {name} · collector exit code: "
                         f"{rc} · synthetic fixtures, {T.COUNT}-position stream -->\n")
            print(f"{name}: exit {rc} -> {out}")
            written.append((name, rc, out))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args(argv)
    written = run(args.outdir)
    expected = {"complete": 0, "pending": 4, "blocked": 3, "gatefail": 3}
    bad = [(n, rc) for n, rc, _ in written if rc != expected[n]]
    if bad:
        print(f"UNEXPECTED exit codes: {bad}", file=sys.stderr)
        return 1
    print(f"{len(written)} scenario transcripts written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
