"""Aggregate the exp_17 C4 grid into a rotation-robustness table.

The question is not "how good is Yaw-Aug" but "how much does its quality MOVE
when the scene is rotated". So each row is one (checkpoint, K) orbit and reports
the value at 0 degrees alongside the **spread** (max - min) across the four
angles. A rotation-invariant arm has a spread of zero; the vanilla anchor does
not (exp_07's A6 head-to-head measured EDT +5.83 at 90 degrees).

Preconditions are enforced rather than assumed: a cell whose recorded protocol
disagrees with the grid, or whose recorded angle contradicts its filename, is a
hard error — a silently permuted orbit would produce a plausible spread that
means nothing. An incomplete orbit yields NO row: max-min over three of four
angles is a smaller, flattering number, not the C4 spread.

Usage:
    python -m src.tools.exp17_rotation_table --dir outputs_FLAC/exp17_YAWAUG_roteval
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

ROTATIONS = (0, 90, 180, 270)
COND_METHOD = "vanilla"
COND_AUTOCAST = "bf16"

# The keys as they exist IN the metric JSON ("T60", "C50", ...), which are NOT
# the stdout labels ("Test/T60 (%)"). Verified against a live grid record.
METRIC_KEYS = ("T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1")
LOWER_IS_BETTER = {"T60", "C50", "EDT", "FD"}

# The eval_name is embedded mid-filename, NOT at the end: for non-zero angles
# eval_FLAC appends its OWN rotation suffix after the seed, so real files read
# `..._rot90_seed42_rot90.json` while 0-degree files read `..._rot0_seed42.json`.
# An end-anchored pattern therefore matches only the 0-degree quarter of the
# grid and silently drops every rotated cell (found live at 41/128 cells; the
# fixture had idealized the names -- same lesson as the termination marker).
_NAME = re.compile(r"_S(\d+)_K(\d+)_rot(\d+)_seed(\d+)")
_EVAL_SUFFIX = re.compile(r"_seed\d+_rot(\d+)\.json$")


class Cell(NamedTuple):
    step: int
    k: int
    rotate_deg: int
    metrics: dict


class OrbitRow(NamedTuple):
    step: int
    k: int
    at_0: dict
    spread: dict


def load_cells(directory: Path) -> list[Cell]:
    """Every admissible metric JSON in ``directory``, or raise on a bad one."""
    cells = []
    for path in sorted(Path(directory).glob("*.json")):
        if ".stream." in path.name or "_predictions_" in path.name:
            continue                       # sidecars, never metric records
        m = _NAME.search(path.name)
        if not m:
            continue                       # not a grid cell
        step, k, rot = int(m.group(1)), int(m.group(2)), int(m.group(3))
        suf = _EVAL_SUFFIX.search(path.name)
        if suf and int(suf.group(1)) != rot:
            raise ValueError(
                f"{path.name}: the evaluator-appended rotation suffix _rot{suf.group(1)} "
                f"contradicts the rot{rot} in the eval name -- the file was produced "
                f"under a different angle than its cell claims"
            )
        rec = json.loads(path.read_text())

        if rec.get("cond_method") != COND_METHOD:
            raise ValueError(
                f"{path.name}: cond_method={rec.get('cond_method')!r} is not the "
                f"grid's {COND_METHOD!r}; this cell is not part of this experiment"
            )
        if rec.get("cond_autocast") != COND_AUTOCAST:
            raise ValueError(
                f"{path.name}: cond_autocast={rec.get('cond_autocast')!r} is not "
                f"the registered {COND_AUTOCAST!r}; its numbers are not comparable "
                f"to model_comparison.md"
            )
        if float(rec.get("rotate_deg", -1)) != float(rot):
            raise ValueError(
                f"{path.name}: recorded rotate_deg={rec.get('rotate_deg')} "
                f"contradicts the {rot} in its name — the orbit would be permuted"
            )
        missing = [key for key in METRIC_KEYS if key not in rec["metrics"]]
        if missing:
            raise ValueError(f"{path.name}: metrics record lacks {missing}")
        cells.append(Cell(step, k, rot, rec["metrics"]))
    return cells


def orbit_rows(cells: list[Cell]) -> list[OrbitRow]:
    """One row per COMPLETE (step, K) orbit, ordered by step then K."""
    groups: dict[tuple[int, int], dict[int, dict]] = {}
    for c in cells:
        groups.setdefault((c.step, c.k), {})[c.rotate_deg] = c.metrics

    rows = []
    for (step, k), by_rot in sorted(groups.items()):
        if set(by_rot) != set(ROTATIONS):
            continue                       # incomplete orbit -> no row, by design
        at_0, spread = {}, {}
        for key in METRIC_KEYS:
            vals = [by_rot[r][key] for r in ROTATIONS]
            at_0[key] = by_rot[0][key]
            spread[key] = max(vals) - min(vals)
        rows.append(OrbitRow(step, k, at_0, spread))
    return rows


def render(rows: list[OrbitRow]) -> str:
    short = {"T60": "T60", "C50": "C50", "EDT": "EDT", "FD": "FD",
             "RIR_to_GT_RIR_R@1": "R@1"}
    head = "| step | K | " + " | ".join(
        f"{short[m]}@0 | Δ{short[m]}" for m in METRIC_KEYS) + " |"
    sep = "|" + "---|" * (2 + 2 * len(METRIC_KEYS))
    out = [
        "Δ is the C4 spread (max − min over 0/90/180/270°). A rotation-invariant",
        "arm has Δ = 0; the value at 0° is the ordinary quality number.",
        "", head, sep,
    ]
    for r in rows:
        cells = []
        for m in METRIC_KEYS:
            cells += [f"{r.at_0[m]:.3f}", f"{r.spread[m]:.3f}"]
        out.append(f"| {r.step} | {r.k} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path)
    args = ap.parse_args(argv)

    cells = load_cells(args.dir)
    rows = orbit_rows(cells)
    complete = len(rows) * len(ROTATIONS)
    print(f"# exp_17 C4 rotation robustness — {len(cells)} cells, "
          f"{len(rows)} complete orbits ({complete} cells in rows)")
    if len(cells) != complete:
        print(f"# NOTE: {len(cells) - complete} cell(s) belong to incomplete "
              f"orbits and are deliberately excluded — a 3-of-4 max-min is not "
              f"a C4 spread.")
    print()
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
