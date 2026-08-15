"""Plan the exp_17 C4 rotation-evaluation grid.

Yixun, 2026-08-15: every finished checkpoint of the FULL 40k run, evaluated at
0/90/180/270 degrees, at K=1 and K=8 — 16 x 4 x 2 = **128 cells** — executed
only AFTER training finishes, so the run keeps both A6000s to itself.

This module only PLANS. It builds the cell list and the exact ``eval_FLAC.py``
argv for each, so the protocol can be unit-tested instead of trusted: at ~11 min
a cell, a mistake repeated 128 times is a day of GPU time. Execution lives in
``exp17_rotation_grid_run.sh``, which refuses to start while training is alive.

See ``src/tests/test_exp17_rotation_grid.py`` for why each pin exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

# The C4 orbit. 45 degrees is the negative control used in exp_07's A6 test and
# is deliberately NOT part of this grid.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)

# K is a property of the DATASET CONFIG (its modalities.acoustic_context
# .max_context), never a flag. These are the published unseen configs: all
# 6,337 items / 17 rooms. Never substitute a reduced or invented eval set.
_EVAL_DIR = "src/configs/dataset_configs/AR/eval"
K_CONFIGS: dict[int, str] = {
    1: f"{_EVAL_DIR}/acousticroom_unseeneval_1.json",
    8: f"{_EVAL_DIR}/acousticroom_unseeneval.json",
}

ENDPOINT_STEPS = 40_000
CADENCE = 2_500
EXPECTED_STEPS: tuple[int, ...] = tuple(range(CADENCE, ENDPOINT_STEPS + 1, CADENCE))

# The scoring protocol every row of model_comparison.md was produced with.
CFG_SCALE = "1.0"
STEPS = "1"
SEED = "42"
COND_METHOD = "vanilla"     # the arm is vanilla-conditioned (announcement 05)
ROTATE_MODE = "fixed"
# model_comparison.md declares the protocol every row was produced under:
# "full published unseen split (6,337 items / 17 rooms), EMA weights, cfg 1.0,
# bf16 cond-autocast". eval_FLAC.py DEFAULTS to 'default', not bf16 -- so
# omitting this flag silently produces numbers comparable to no existing row.
COND_AUTOCAST = "bf16"


class Cell(NamedTuple):
    step: int
    k: int
    rotate_deg: int
    ckpt_path: str


def build_grid(ckpts: dict[int, str]) -> list[Cell]:
    """Every (checkpoint, K, rotation) cell, or raise if the set is not the one.

    A 15/16 grid is not the registered experiment, and an off-cadence checkpoint
    means we are looking at a different run's directory. Both are refused loudly
    rather than quietly evaluated.
    """
    have, want = set(ckpts), set(EXPECTED_STEPS)
    if missing := sorted(want - have):
        raise ValueError(
            f"checkpoint set is incomplete: missing steps {missing}. The grid is "
            f"defined over all {len(EXPECTED_STEPS)} cadence checkpoints."
        )
    if extra := sorted(have - want):
        raise ValueError(
            f"unexpected off-cadence checkpoints {extra}: expected exactly "
            f"{CADENCE}..{ENDPOINT_STEPS} step {CADENCE}. Is this the right run?"
        )
    return [
        Cell(step=s, k=k, rotate_deg=r, ckpt_path=ckpts[s])
        for s in EXPECTED_STEPS
        for k in sorted(K_CONFIGS)
        for r in ROTATIONS
    ]


def cell_name(c: Cell) -> str:
    """Unique identity for a cell; the metric JSON is named from it.

    Two cells sharing a name would overwrite each other's results silently.
    """
    return f"exp17_YAWAUG_S{c.step}_K{c.k}_rot{c.rotate_deg}_seed{SEED}"


def cell_argv(c: Cell, *, model_config: str) -> list[str]:
    """The exact command line for one cell.

    Every protocol flag is passed EXPLICITLY. Announcement 05: a mismatched
    eval-protocol flag produces plausible-looking, catastrophically wrong
    numbers in both directions, and relying on a default is how exp_09's
    protocol error happened.

    ``--rotate-seed`` is deliberately absent: ``eval_FLAC.resolve_rotation_plan``
    raises on a seed in fixed mode, because a fixed angle draws nothing.
    """
    return [
        "python", "eval_FLAC.py",
        "--model-config", model_config,
        "--dataset-config", K_CONFIGS[c.k],
        "--ckpt-path", c.ckpt_path,
        "--cond-method", COND_METHOD,
        "--cond-autocast", COND_AUTOCAST,
        "--rotate-mode", ROTATE_MODE,
        "--rotate-deg", str(float(c.rotate_deg)),
        "--cfg-scale", CFG_SCALE,
        "--steps", STEPS,
        "--seed", SEED,
        "--eval-name", cell_name(c),
    ]


def cell_is_complete(c: Cell, *, out_dir: Path) -> bool:
    """Is there an ADMISSIBLE result for this cell?

    "Non-empty file exists" is not enough (Codex review): a truncated JSON, a
    ``{}``, or an output produced under the WRONG protocol would all suppress a
    rerun and then be read as a result. So the record must parse, carry metrics,
    and agree with the cell on every protocol field it records — which makes this
    the artifact-admission gate as well as the resume predicate.
    """
    for path in Path(out_dir).rglob(f"*{cell_name(c)}*.json"):
        if not (path.is_file() and path.stat().st_size > 0):
            continue
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue                       # truncated/crashed: rerun it
        if not isinstance(rec, dict) or not rec.get("metrics"):
            continue
        if rec.get("cond_method") != COND_METHOD:
            continue
        if rec.get("cond_autocast") not in (None, COND_AUTOCAST):
            continue                       # produced under a different protocol
        if float(rec.get("rotate_deg", -1)) != float(c.rotate_deg):
            continue                       # a different angle wrote this file
        return True
    return False


def pending_cells(grid: list[Cell], *, out_dir: Path) -> list[Cell]:
    """Cells with no ADMISSIBLE result yet — 128 cells will not run uninterrupted."""
    return [c for c in grid if not cell_is_complete(c, out_dir=out_dir)]
