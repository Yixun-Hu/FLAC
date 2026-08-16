"""exp_17 — the C4 rotation-evaluation grid over the FULL run's checkpoints.

Yixun, 2026-08-15: evaluate every finished checkpoint at 0/90/180/270 degrees,
at K=1 and K=8, and run it only AFTER training finishes so the 40k run keeps
both A6000s to itself.

That is 16 checkpoints x 4 rotations x 2 K = **128 cells**, ~11 min each, so the
planner is not a convenience: a mistake repeated 128 times is a day of GPU time.
The things that can silently ruin the grid, and are therefore pinned here:

* **The eval-protocol flags are part of the experiment** (announcement 05).
  ``--cond-method`` must be ``vanilla`` because the arm is vanilla-conditioned;
  evaluating a checkpoint under the wrong conditioning produced exp_09's
  protocol error and forced a retraction. It is passed explicitly, never
  defaulted.
* **K is selected by the dataset config, never by a flag** — ``max_context``
  lives in ``acousticroom_unseeneval{,_1,_4}.json``. Pairing a K label with the
  wrong config mislabels every number in the row.
* **The eval set is the full published one**: all 6,337 items / 17 rooms. No
  subsampling, ever (standing directive).
* **Cell identity must be unique.** The metric JSON is named from the eval name;
  two cells sharing a name overwrite each other and the loss is silent.
* ``--rotate-seed`` is meaningless in fixed mode and ``eval_FLAC.py`` raises on
  it; the planner must not emit it.

Written by the main session seat (Claude Opus 5, max effort).
"""
import json

import pytest

from src.tools.exp17_rotation_grid import (
    ROTATIONS, K_CONFIGS, EXPECTED_STEPS, build_grid, cell_argv, cell_name,
    pending_cells,
)


CKPT_TEMPLATE = "/w/outputs/checkpoints/epoch={e}-step={s}.ckpt"
CKPTS = {s: CKPT_TEMPLATE.format(e=s // 4550, s=s) for s in EXPECTED_STEPS}
MODEL_CFG = "/w/FLAC_AR_YAWAUG_A6000.json"


# --------------------------------------------------------------------------- #
# the grid itself
# --------------------------------------------------------------------------- #
def test_the_grid_is_exactly_the_registered_size():
    grid = build_grid(CKPTS)
    assert len(grid) == 16 * 4 * 2 == 128


def test_every_cell_is_unique():
    names = [cell_name(c) for c in build_grid(CKPTS)]
    assert len(set(names)) == len(names), "two cells share an eval name; one would overwrite the other"


def test_the_registered_rotations_are_the_C4_orbit():
    assert ROTATIONS == (0, 90, 180, 270)


def test_both_K_are_covered_for_every_checkpoint_and_rotation():
    grid = build_grid(CKPTS)
    for step in EXPECTED_STEPS:
        for rot in ROTATIONS:
            ks = {c.k for c in grid if c.step == step and c.rotate_deg == rot}
            assert ks == set(K_CONFIGS), f"step {step} rot {rot} covers {ks}"


def test_an_incomplete_checkpoint_set_is_refused():
    """A 15/16 grid is not the registered experiment; say so rather than run it."""
    short = {s: p for s, p in CKPTS.items() if s != 37500}
    with pytest.raises(ValueError, match="37500"):
        build_grid(short)


def test_an_off_cadence_checkpoint_is_refused():
    extra = dict(CKPTS)
    extra[42500] = CKPT_TEMPLATE.format(e=9, s=42500)
    with pytest.raises(ValueError, match="42500"):
        build_grid(extra)


# --------------------------------------------------------------------------- #
# K comes from the dataset config, not from a label
# --------------------------------------------------------------------------- #
def test_each_K_maps_to_its_published_unseen_config():
    assert K_CONFIGS[1].endswith("acousticroom_unseeneval_1.json")
    assert K_CONFIGS[8].endswith("acousticroom_unseeneval.json")


def test_the_cell_carries_the_config_matching_its_own_K():
    for c in build_grid(CKPTS):
        argv = cell_argv(c, model_config=MODEL_CFG)
        i = argv.index("--dataset-config")
        assert argv[i + 1] == K_CONFIGS[c.k], f"K={c.k} paired with {argv[i+1]}"


def test_no_reduced_or_invented_eval_set_can_be_used():
    """Standing directive: the full published unseen set, all 6,337 items."""
    for cfg in K_CONFIGS.values():
        assert "unseeneval" in cfg
        assert "sub" not in cfg and "small" not in cfg


# --------------------------------------------------------------------------- #
# the protocol flags (announcement 05)
# --------------------------------------------------------------------------- #
def test_conditioning_is_explicitly_vanilla_on_every_cell():
    for c in build_grid(CKPTS):
        argv = cell_argv(c, model_config=MODEL_CFG)
        assert argv[argv.index("--cond-method") + 1] == "vanilla"


def test_the_rotation_is_passed_as_a_fixed_angle():
    argv = cell_argv(build_grid(CKPTS)[0], model_config=MODEL_CFG)
    assert argv[argv.index("--rotate-mode") + 1] == "fixed"


def test_rotate_seed_is_never_emitted_in_fixed_mode():
    """eval_FLAC.resolve_rotation_plan RAISES on a seed in fixed mode."""
    for c in build_grid(CKPTS):
        assert "--rotate-seed" not in cell_argv(c, model_config=MODEL_CFG)


def test_every_registered_angle_reaches_the_command_line():
    seen = {float(cell_argv(c, model_config=MODEL_CFG)[
        cell_argv(c, model_config=MODEL_CFG).index("--rotate-deg") + 1])
        for c in build_grid(CKPTS)}
    assert seen == {float(r) for r in ROTATIONS}


def test_the_scoring_protocol_is_pinned_to_the_comparison_table():
    """cfg-scale 1.0 / steps 1 / seed 42 — what every model_comparison row used."""
    argv = cell_argv(build_grid(CKPTS)[0], model_config=MODEL_CFG)
    assert argv[argv.index("--cfg-scale") + 1] == "1.0"
    assert argv[argv.index("--steps") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == "42"


def test_the_registered_bf16_cond_autocast_is_passed_explicitly():
    """model_comparison.md: every row is 'cfg 1.0, bf16 cond-autocast'.

    eval_FLAC.py DEFAULTS to 'default', not bf16 — so omitting this flag makes
    all 128 numbers comparable to no existing row in the table.
    """
    for c in build_grid(CKPTS):
        argv = cell_argv(c, model_config=MODEL_CFG)
        assert argv[argv.index("--cond-autocast") + 1] == "bf16"


def test_the_checkpoint_path_is_the_one_for_that_cell():
    for c in build_grid(CKPTS):
        argv = cell_argv(c, model_config=MODEL_CFG)
        assert argv[argv.index("--ckpt-path") + 1] == CKPTS[c.step]


# --------------------------------------------------------------------------- #
# cell identity
# --------------------------------------------------------------------------- #
def test_the_eval_name_encodes_step_K_and_rotation():
    grid = build_grid(CKPTS)
    c = next(x for x in grid if x.step == 20000 and x.k == 8 and x.rotate_deg == 90)
    n = cell_name(c)
    assert "20000" in n and "K8" in n and "90" in n


def test_names_that_differ_only_by_rotation_are_distinct():
    grid = build_grid(CKPTS)
    a = next(x for x in grid if x.step == 2500 and x.k == 1 and x.rotate_deg == 0)
    b = next(x for x in grid if x.step == 2500 and x.k == 1 and x.rotate_deg == 180)
    assert cell_name(a) != cell_name(b)


# --------------------------------------------------------------------------- #
# resumability — 128 cells will not survive an uninterrupted 12 h
# --------------------------------------------------------------------------- #
def _record(c, **over):
    rec = {"metrics": {"T60": 1.0}, "cond_method": "vanilla",
           "cond_autocast": "bf16", "rotate_deg": float(c.rotate_deg)}
    rec.update(over)
    return json.dumps(rec)


def test_completed_cells_are_not_rerun(tmp_path):
    grid = build_grid(CKPTS)
    done = grid[0]
    (tmp_path / f"{cell_name(done)}.json").write_text(_record(done))
    remaining = pending_cells(grid, out_dir=tmp_path)
    assert done not in remaining
    assert len(remaining) == len(grid) - 1


def test_an_empty_json_object_does_not_count_as_a_result(tmp_path):
    """Rev 1's test explicitly blessed `{}` — Codex called that out."""
    grid = build_grid(CKPTS)
    (tmp_path / f"{cell_name(grid[0])}.json").write_text("{}")
    assert grid[0] in pending_cells(grid, out_dir=tmp_path)


def test_a_truncated_json_does_not_count_as_a_result(tmp_path):
    grid = build_grid(CKPTS)
    (tmp_path / f"{cell_name(grid[0])}.json").write_text('{"metrics": {"T60"')
    assert grid[0] in pending_cells(grid, out_dir=tmp_path)


@pytest.mark.parametrize("wrong", [
    {"cond_method": "fa_invariant"},
    {"cond_autocast": "default"},
    {"rotate_deg": 45.0},
])
def test_a_result_produced_under_the_wrong_protocol_is_rerun(tmp_path, wrong):
    """The resume predicate is also the artifact-admission gate."""
    grid = build_grid(CKPTS)
    c = grid[0]
    (tmp_path / f"{cell_name(c)}.json").write_text(_record(c, **wrong))
    assert c in pending_cells(grid, out_dir=tmp_path)


def test_nothing_is_skipped_when_no_output_exists(tmp_path):
    grid = build_grid(CKPTS)
    assert pending_cells(grid, out_dir=tmp_path) == grid


def test_an_empty_output_file_does_not_count_as_done(tmp_path):
    """A crashed eval can leave a zero-byte JSON; that is not a result."""
    grid = build_grid(CKPTS)
    (tmp_path / f"{cell_name(grid[0])}.json").write_text("")
    assert grid[0] in pending_cells(grid, out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# the P1 control arm — same grid, distinct identity
# --------------------------------------------------------------------------- #
def test_the_control_arm_yields_distinct_names_for_every_cell():
    """A YAWAUG cell and its P1-control twin must never share a metric filename."""
    from src.tools.exp17_rotation_grid import DEFAULT_ARM, P1_CONTROL_ARM
    grid = build_grid(CKPTS)
    yaw = {cell_name(c, arm=DEFAULT_ARM) for c in grid}
    ctl = {cell_name(c, arm=P1_CONTROL_ARM) for c in grid}
    assert yaw.isdisjoint(ctl)
    assert len(ctl) == len(grid)


def test_the_control_arm_reaches_the_eval_name_on_the_command_line():
    from src.tools.exp17_rotation_grid import P1_CONTROL_ARM
    c = build_grid(CKPTS)[0]
    argv = cell_argv(c, model_config=MODEL_CFG, arm=P1_CONTROL_ARM)
    assert argv[argv.index("--eval-name") + 1].startswith("exp17_P1CTRL_")


def test_the_default_arm_is_byte_identical_to_the_pre_parameterisation_names():
    """The RUNNING grid depends on these names; the default must not move."""
    c = next(x for x in build_grid(CKPTS) if x.step == 2500 and x.k == 1 and x.rotate_deg == 0)
    assert cell_name(c) == "exp17_YAWAUG_S2500_K1_rot0_seed42"


def test_completion_of_one_arm_does_not_mark_the_other_done(tmp_path):
    """The admission gate must scope by arm, or the control would skip cells
    the YAWAUG grid already produced."""
    from src.tools.exp17_rotation_grid import P1_CONTROL_ARM
    grid = build_grid(CKPTS)
    c = grid[0]
    (tmp_path / f"{cell_name(c)}.json").write_text(_record(c))   # YAWAUG artifact
    assert c in pending_cells(grid, out_dir=tmp_path, arm=P1_CONTROL_ARM)
