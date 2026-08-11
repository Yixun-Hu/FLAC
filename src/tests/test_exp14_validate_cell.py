"""Tests for exp_14's per-cell artifact validator (plan §5.4/§5.5).

``exp14_validate_cell.py`` answers ONE question: are this cell's artifacts a
complete, self-consistent, protocol-compliant record of the cell the campaign
registered? It is the same predicate in three places, which is the point —

* the screen driver runs it before emitting SCREENRESULT, so a cell that cannot
  be validated never announces a result;
* the wave submitter runs it for dedup, where the rule is validate-BEFORE-skip
  (review B6): an artifact that exists but fails any check halts for triage
  instead of being silently skipped or overwritten;
* the collector runs it before any contrast.

Everything here is byte-level: synthetic JSON fixtures, no torch, no GPU. The
validator itself must stay torch-free (the submitter classifies ~100 cells on a
shared login node), so it carries LOCAL copies of two rules that live in
``eval_FLAC``: the canonical stream serialization and the output-path shape.
Local copies can drift, so two tests pin them to the originals — those are the
only tests here that import ``eval_FLAC``.
"""
import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_EXPDIR = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_14_yaw_gen_claude")


def _load(name):
    if _EXPDIR not in sys.path:
        sys.path.insert(0, _EXPDIR)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EXPDIR, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("exp14_validate_cell")

PIN = "a" * 40
CKPT_SHA = "b" * 64
COUNT = 6337
IMG_W = 512


# --------------------------------------------------------------------------
# fixtures: a VALID cell of each class, which every negative case then breaks
# --------------------------------------------------------------------------
def _stream(cell, count=COUNT, offset_for=lambda i: 0):
    """A .stream.json payload whose hashes really are its tuples' hashes."""
    input_tuples = [[i, f"{i}|room/rir_{i}.wav", [f"c{i}"], IMG_W] for i in range(count)]
    offsets = [offset_for(i) for i in range(count)]
    assignment_tuples = [[i, f"{i}|room/rir_{i}.wav", offsets[i]] for i in range(count)]
    rec = {
        "schema_version": 1,
        "fingerprint_schema": 1,
        "rotate_mode": "random" if cell == "rgen" else "fixed",
        "rotate_seed": 42 if cell == "rgen" else None,
        "rotate_deg": None if cell == "rgen" else (90.0 if cell == "vctl" else 0.0),
        "img_w": IMG_W,
        "stream_count": count,
        "input_tuples": input_tuples,
        "offsets": offsets,
        "assignment_tuples": assignment_tuples,
    }
    rec["input_hash"] = V.canonical_stream_hash(input_tuples)
    rec["assignment_hash"] = V.canonical_stream_hash(assignment_tuples)
    return rec


def _metrics(cell, arm="C4L", seed=42, k=8):
    rec = {
        "metrics": {"T60_error": 1.0, "RIR_to_GT_RIR_R@1": 0.5},
        "ckpt_path": f"/o/exp11_{arm}/checkpoints/epoch=8-step=40000.ckpt",
        "rotate_deg": 0.0,
        "cond_method": "vanilla" if arm == "VANL" else "fa_invariant",
        "frame_avg_angles": None if arm == "VANL" else [0.0, 90.0, 180.0, 270.0],
        "cond_autocast": "bf16",
        "orbit_execution": "n/a" if arm == "VANL" else "batched",
        "source_sha": PIN,
        "batch_size": 64,
        "n_samples": COUNT,
        "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
        if k == 8
        else "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
        "seed": seed,
        "cfg_scale": 1.0,
        "steps": 1,
        "eval_name": V.eval_name(V.Cell(arm, cell, 40000, seed, k, _deg(cell))),
        "weights_source": "ema",
        "device": "cuda",
    }
    if cell == "rgen":
        # random mode APPENDS provenance and nulls the angle (eval_FLAC's own rule)
        rec["rotate_deg"] = None
        rec["rotate_mode"] = "random"
        rec["rotate_seed"] = seed
        rec["input_hash"] = "0" * 64
        rec["assignment_hash"] = "1" * 64
        rec["stream_count"] = COUNT
        rec["img_w"] = IMG_W
    elif cell == "vctl":
        rec["rotate_deg"] = 90.0
    return rec


def _screenmeta(cell, arm="C4L", seed=42, k=8):
    return {
        "arm": arm, "step": 40000, "seed": seed, "K": k,
        "eval_name": V.eval_name(V.Cell(arm, cell, 40000, seed, k, _deg(cell))),
        "cfg_scale": 1.0, "steps": 1,
        "model_config": f"worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_{arm}.json",
        "model_config_sha256": "c" * 64,
        "dataset_config": "x.json",
        "ckpt_path": f"/o/exp11_{arm}/checkpoints/epoch=8-step=40000.ckpt",
        "ckpt_sha256": CKPT_SHA, "use_ema": True,
        "frame_avg_angles": None if arm == "VANL" else [0.0, 90.0, 180.0, 270.0],
        "cond_method": "vanilla" if arm == "VANL" else "fa_invariant",
        "cond_autocast": "bf16", "commit": PIN,
        "cell": cell, "training_orbit": 0 if arm == "VANL" else 4,
        "eval_orbit": 0 if arm == "VANL" else 4,
        "rotate_mode": "random" if cell == "rgen" else "fixed",
        "rotate_deg": None if cell == "rgen" else (90.0 if cell == "vctl" else 0.0),
        "rotate_seed": seed if cell == "rgen" else None,
        "expected_stream_count": COUNT, "record_stream": True,
        "stream_sidecar": "m.stream.json",
        "batch_size": 64, "num_workers": 4,
    }


def _deg(cell):
    return 90.0 if cell == "vctl" else None


def _cell(cell, arm="C4L", seed=42, k=8):
    return V.Cell(arm, cell, 40000, seed, k, _deg(cell))


def _write_cell(tmp_path, cell, arm="C4L", seed=42, k=8,
                metrics=None, meta=None, stream=None, count=COUNT):
    """Write the three artifact files a landed cell consists of; return the metrics path."""
    shift = {"rgen": lambda i: i % IMG_W, "vctl": lambda i: 128, "zref": lambda i: 0}[cell]
    m = _metrics(cell, arm, seed, k) if metrics is None else metrics
    sm = _screenmeta(cell, arm, seed, k) if meta is None else meta
    st = _stream(cell, count, shift) if stream is None else stream
    # The record and its sidecar are two views of ONE stream, so a VALID fixture
    # must agree with itself; a broken-stream case then breaks exactly one of them.
    if "input_hash" in m:
        m["input_hash"], m["assignment_hash"] = st["input_hash"], st["assignment_hash"]
    p = tmp_path / "epoch=8-step=40000_metrics_1_1.0_x.json"
    p.write_text(json.dumps(m))
    (tmp_path / (p.name + ".screenmeta.json")).write_text(json.dumps(sm))
    (tmp_path / p.name.replace(".json", ".stream.json")).write_text(json.dumps(st))
    return str(p)


# --------------------------------------------------------------------------
# 1. the registered grid
# --------------------------------------------------------------------------
def test_expected_grid_is_exactly_106_unique_cells():
    g = V.expected_grid()
    assert len(g) == 106
    assert len(set(g)) == 106


def test_expected_grid_block_sizes():
    g = V.expected_grid()
    assert sum(1 for c in g if c.cell == "rgen") == 50
    assert sum(1 for c in g if c.cell == "zref") == 50
    assert sum(1 for c in g if c.cell == "vctl") == 6


def test_expected_grid_vctl_tuples_are_the_six_registered_ones():
    got = {(c.arm, c.rotate_deg) for c in V.expected_grid() if c.cell == "vctl"}
    assert got == {("C4L", 90.0), ("C8", 90.0), ("C16", 90.0), ("C32", 90.0),
                   ("VANL", 90.0), ("C4L", 45.0)}


def test_expected_grid_vctl_is_s42_k8_only():
    for c in V.expected_grid():
        if c.cell == "vctl":
            assert (c.seed, c.k) == (42, 8)


def test_expected_grid_has_no_vanl_at_45():
    assert ("VANL", 45.0) not in {(c.arm, c.rotate_deg)
                                  for c in V.expected_grid() if c.cell == "vctl"}


def test_expected_grid_rgen_and_zref_cover_five_arms_two_k_five_seeds():
    for cell in ("rgen", "zref"):
        cells = [c for c in V.expected_grid() if c.cell == cell]
        assert {c.arm for c in cells} == {"VANL", "C4L", "C8", "C16", "C32"}
        assert {c.k for c in cells} == {1, 8}
        assert {c.seed for c in cells} == {42, 43, 44, 45, 46}
        assert all(c.rotate_deg is None for c in cells)


def test_expected_grid_is_all_at_step_40000():
    assert {c.step for c in V.expected_grid()} == {40000}


def test_wave_cells_partition_the_grid():
    waves = [V.wave_cells(w) for w in ("vctl", "zref", "rgen")]
    assert [len(w) for w in waves] == [6, 50, 50]
    assert sorted(c for w in waves for c in w) == sorted(V.expected_grid())
    assert sorted(V.wave_cells("all")) == sorted(V.expected_grid())


def test_wave_cells_rejects_an_unknown_wave():
    with pytest.raises(ValueError):
        V.wave_cells("conf")


# --------------------------------------------------------------------------
# 2. naming (the same rule the sbatch renders in shell)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell,arm,seed,k,deg,want", [
    ("rgen", "C32", 44, 8, None, "exp14_C32_rgen_S40000_s44_K8_rotrand44"),
    ("zref", "C32", 44, 8, None, "exp14_C32_zref_S40000_s44_K8"),
    ("vctl", "C4L", 42, 8, 45.0, "exp14_C4L_vctl_S40000_s42_K8_rot45"),
    ("vctl", "VANL", 42, 8, 90.0, "exp14_VANL_vctl_S40000_s42_K8_rot90"),
])
def test_eval_name_shape(cell, arm, seed, k, deg, want):
    assert V.eval_name(V.Cell(arm, cell, 40000, seed, k, deg)) == want


def test_eval_names_are_injective_over_the_whole_grid():
    names = [V.eval_name(c) for c in V.expected_grid()]
    assert len(set(names)) == len(names) == 106


def test_parse_eval_name_round_trips_every_registered_cell():
    for c in V.expected_grid():
        assert V.parse_eval_name(V.eval_name(c)) == c


@pytest.mark.parametrize("bad", [
    "exp11_C8_screen_S10000_s42_K8",          # another campaign
    "exp14_C8_conf_S40000_s42_K8",            # unregistered cell type
    "exp14_C8_rgen_S40000_s42_K4",            # unregistered K
    "exp14_FA1_rgen_S40000_s42_K8_rotrand42",  # unregistered arm
    "exp14_C8_rgen_S40000_s42_K8",            # rgen without its seed token
    "exp14_C8_zref_S40000_s42_K8_rot90",      # zref with a rotation token
    "exp14_C8_vctl_S40000_s42_K8",            # vctl without its angle
])
def test_parse_eval_name_rejects_unregistered_names(bad):
    with pytest.raises(ValueError):
        V.parse_eval_name(bad)


# --------------------------------------------------------------------------
# 3. rules mirrored from eval_FLAC — pinned to the originals so they cannot drift
# --------------------------------------------------------------------------
def test_canonical_stream_hash_matches_eval_FLAC():
    import eval_FLAC  # the only import of the heavy module in this suite
    tuples = [[0, "0|a.wav", ["c0", "c1"], 512], [1, "1|b.wav", [], 512]]
    assert V.canonical_stream_hash(tuples) == eval_FLAC.canonical_stream_hash(tuples)


def test_metrics_path_rule_matches_eval_FLAC_build_output_paths():
    import eval_FLAC
    ckpt = "/o/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt"
    for cell, arm, seed, k, deg, cond, n in [
        ("rgen", "C4L", 43, 8, None, "fa_invariant", 4),
        ("zref", "VANL", 46, 1, None, "vanilla", 0),
        ("vctl", "C4L", 42, 8, 45.0, "fa_invariant", 4),
    ]:
        c = V.Cell(arm, cell, 40000, seed, k, deg)
        want = eval_FLAC.build_output_paths(
            ckpt, 1, 1.0, V.eval_name(c), cond_method=cond,
            rotate_deg=0.0 if deg is None else deg, n_angles=n,
            rotate_mode="random" if cell == "rgen" else "fixed",
            rotate_seed=seed if cell == "rgen" else None,
        )["metrics"]
        assert V.metrics_path(ckpt, c) == want


def test_expected_column_shift_matches_yaw_column_shift():
    import math
    from src.data.yaw_rotation import yaw_column_shift
    for deg in (0.0, 45.0, 90.0):
        assert V.expected_column_shift(deg, IMG_W) == yaw_column_shift(
            math.radians(deg), IMG_W)


# --------------------------------------------------------------------------
# 4. metrics-record validation
# --------------------------------------------------------------------------
def test_valid_cells_of_every_class_report_no_reasons(tmp_path):
    for cell in ("rgen", "zref", "vctl"):
        d = tmp_path / cell
        d.mkdir()
        p = _write_cell(d, cell)
        assert V.validate_cell(p, _cell(cell), pin=PIN, ckpt_sha=CKPT_SHA,
                               expected_count=COUNT) == []


def test_a_random_cell_missing_its_rotate_mode_is_rejected(tmp_path):
    m = _metrics("rgen")
    del m["rotate_mode"]
    p = _write_cell(tmp_path, "rgen", metrics=m)
    reasons = V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA)
    assert any("rotate_mode" in r for r in reasons)


def test_a_fixed_cell_carrying_random_provenance_is_rejected(tmp_path):
    # A zref record with rotate_mode set is not the frozen fixed-mode record: the
    # cell ran a different protocol than the one it claims.
    m = _metrics("zref")
    m["rotate_mode"] = "random"
    m["rotate_seed"] = 42
    p = _write_cell(tmp_path, "zref", metrics=m)
    reasons = V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA)
    assert any("rotate_mode" in r for r in reasons)


def test_a_random_cell_with_a_non_null_rotate_deg_is_rejected(tmp_path):
    m = _metrics("rgen")
    m["rotate_deg"] = 0.0          # an unrotated cell reads exactly like this
    p = _write_cell(tmp_path, "rgen", metrics=m)
    assert any("rotate_deg" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_zref_cell_with_a_nonzero_angle_is_rejected(tmp_path):
    m = _metrics("zref")
    m["rotate_deg"] = 90.0
    p = _write_cell(tmp_path, "zref", metrics=m)
    assert any("rotate_deg" in r for r in
               V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_vctl_cell_at_the_wrong_angle_is_rejected(tmp_path):
    m = _metrics("vctl")
    m["rotate_deg"] = 45.0                    # the cell asked for 90
    p = _write_cell(tmp_path, "vctl", metrics=m)
    assert any("rotate_deg" in r for r in
               V.validate_cell(p, _cell("vctl"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_rotation_seed_that_is_not_the_eval_seed_is_rejected(tmp_path):
    m = _metrics("rgen")
    m["rotate_seed"] = 99
    p = _write_cell(tmp_path, "rgen", metrics=m)
    assert any("rotate_seed" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


@pytest.mark.parametrize("field,value,token", [
    ("cond_autocast", "default", "cond_autocast"),
    ("cfg_scale", 3.0, "cfg_scale"),
    ("steps", 8, "steps"),
    ("batch_size", 32, "batch_size"),
    ("seed", 43, "seed"),
    ("cond_method", "vanilla", "cond_method"),
    ("frame_avg_angles", [0.0, 45.0], "frame_avg_angles"),
    ("eval_name", "exp14_C4L_rgen_S40000_s42_K8", "eval_name"),
])
def test_every_protocol_field_is_checked(tmp_path, field, value, token):
    m = _metrics("rgen")
    m[field] = value
    p = _write_cell(tmp_path, "rgen", metrics=m)
    reasons = V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA)
    assert any(token in r for r in reasons), reasons


def test_an_empty_metrics_block_is_rejected(tmp_path):
    m = _metrics("zref")
    m["metrics"] = {}
    p = _write_cell(tmp_path, "zref", metrics=m)
    assert any("metrics" in r for r in
               V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_the_vanilla_arm_must_carry_no_orbit(tmp_path):
    m = _metrics("zref", arm="VANL")
    m["cond_method"] = "fa_invariant"
    m["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    p = _write_cell(tmp_path, "zref", arm="VANL", metrics=m)
    assert V.validate_cell(p, _cell("zref", arm="VANL"), pin=PIN, ckpt_sha=CKPT_SHA)


# --------------------------------------------------------------------------
# 5. the .stream.json assignment audit
# --------------------------------------------------------------------------
def test_a_missing_stream_sidecar_is_rejected(tmp_path):
    p = _write_cell(tmp_path, "rgen")
    os.remove(p.replace(".json", ".stream.json"))
    assert any("stream" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_missing_screenmeta_sidecar_is_rejected(tmp_path):
    p = _write_cell(tmp_path, "rgen")
    os.remove(p + ".screenmeta.json")
    assert any("screenmeta" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_short_stream_is_rejected(tmp_path):
    p = _write_cell(tmp_path, "rgen", count=6000)
    reasons = V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA,
                              expected_count=COUNT)
    assert any("6000" in r for r in reasons), reasons


def test_a_wrong_schema_version_is_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: i % IMG_W)
    st["schema_version"] = 2
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("schema_version" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_stored_input_hash_that_is_not_its_tuples_hash_is_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: i % IMG_W)
    st["input_hash"] = "f" * 64
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("input_hash" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_stored_assignment_hash_that_is_not_its_tuples_hash_is_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: i % IMG_W)
    st["assignment_hash"] = "f" * 64
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("assignment_hash" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_offsets_disagreeing_with_the_assignment_tuples_are_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: i % IMG_W)
    st["offsets"] = [0] * COUNT            # hashes still self-consistent
    st["assignment_hash"] = V.canonical_stream_hash(st["assignment_tuples"])
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("offset" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_out_of_order_positions_are_rejected(tmp_path):
    st = _stream("zref", COUNT, lambda i: 0)
    st["input_tuples"][5][0] = 99
    st["input_hash"] = V.canonical_stream_hash(st["input_tuples"])
    p = _write_cell(tmp_path, "zref", stream=st)
    assert any("position" in r for r in
               V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_zref_stream_with_a_nonzero_offset_is_rejected(tmp_path):
    st = _stream("zref", COUNT, lambda i: 7)
    p = _write_cell(tmp_path, "zref", stream=st)
    assert any("offset" in r for r in
               V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_vctl_stream_must_carry_the_constant_shift_of_its_angle(tmp_path):
    st = _stream("vctl", COUNT, lambda i: 129)      # 90 deg over 512 columns is 128
    p = _write_cell(tmp_path, "vctl", stream=st)
    assert any("offset" in r for r in
               V.validate_cell(p, _cell("vctl"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_random_stream_of_one_constant_offset_is_rejected(tmp_path):
    # 6,337 independent draws over 512 columns are never all equal; a constant
    # stream means the random path did not run.
    st = _stream("rgen", COUNT, lambda i: 3)
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("offset" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_an_offset_outside_the_column_grid_is_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: IMG_W if i == 3 else i % IMG_W)
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("offset" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


def test_a_stream_whose_protocol_contradicts_the_record_is_rejected(tmp_path):
    st = _stream("rgen", COUNT, lambda i: i % IMG_W)
    st["rotate_seed"] = 77
    st["assignment_hash"] = V.canonical_stream_hash(st["assignment_tuples"])
    p = _write_cell(tmp_path, "rgen", stream=st)
    assert any("rotate_seed" in r for r in
               V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA))


# --------------------------------------------------------------------------
# 6. provenance: pin, checkpoint identity, runtime pins
# --------------------------------------------------------------------------
def test_a_cell_from_another_pin_is_rejected(tmp_path):
    p = _write_cell(tmp_path, "zref")
    reasons = V.validate_cell(p, _cell("zref"), pin="c" * 40, ckpt_sha=CKPT_SHA)
    assert any("pin" in r or "commit" in r for r in reasons), reasons


def test_a_cell_from_another_checkpoint_is_rejected(tmp_path):
    p = _write_cell(tmp_path, "zref")
    reasons = V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha="d" * 64)
    assert any("ckpt" in r for r in reasons), reasons


def test_the_screenmeta_must_agree_with_the_cell_it_claims(tmp_path):
    sm = _screenmeta("zref")
    sm["K"] = 1
    p = _write_cell(tmp_path, "zref", meta=sm)
    assert V.validate_cell(p, _cell("zref"), pin=PIN, ckpt_sha=CKPT_SHA)


@pytest.mark.parametrize("field,value", [
    ("batch_size", 32),
    ("num_workers", 6),
    ("expected_stream_count", 100),
    ("record_stream", False),
    ("cond_autocast", "default"),
])
def test_the_screenmeta_runtime_pins_are_checked(tmp_path, field, value):
    sm = _screenmeta("rgen")
    sm[field] = value
    p = _write_cell(tmp_path, "rgen", meta=sm)
    reasons = V.validate_cell(p, _cell("rgen"), pin=PIN, ckpt_sha=CKPT_SHA)
    assert any(field in r for r in reasons), reasons


# --------------------------------------------------------------------------
# 7. failure modes that must be NAMED, never crashes
# --------------------------------------------------------------------------
def test_a_missing_metrics_file_is_a_named_reason(tmp_path):
    reasons = V.validate_cell(str(tmp_path / "nope.json"), _cell("zref"), pin=PIN)
    assert len(reasons) == 1 and "missing" in reasons[0]


def test_malformed_json_is_a_named_reason(tmp_path):
    p = tmp_path / "epoch=8-step=40000_metrics_1_1.0_x.json"
    p.write_text("{not json")
    reasons = V.validate_cell(str(p), _cell("zref"), pin=PIN)
    assert any("parse" in r or "JSON" in r for r in reasons), reasons


def test_an_unregistered_cell_is_refused_before_any_file_is_read(tmp_path):
    with pytest.raises(ValueError):
        V.validate_cell(str(tmp_path / "x.json"),
                        V.Cell("VANL", "vctl", 40000, 42, 8, 45.0), pin=PIN)


def test_reasons_are_strings_and_the_valid_case_is_the_empty_list(tmp_path):
    p = _write_cell(tmp_path, "vctl")
    out = V.validate_cell(p, _cell("vctl"), pin=PIN, ckpt_sha=CKPT_SHA)
    assert out == [] and isinstance(out, list)
    bad = V.validate_cell(p, _cell("vctl"), pin="e" * 40, ckpt_sha=CKPT_SHA)
    assert bad and all(isinstance(r, str) for r in bad)


# --------------------------------------------------------------------------
# 8. the CLI the kit and the submitter actually call
# --------------------------------------------------------------------------
def _run(argv):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = V.main(argv)
    return rc, buf.getvalue()


def test_cli_grid_prints_the_whole_registered_grid():
    rc, out = _run(["grid"])
    lines = [l for l in out.splitlines() if l.strip()]
    assert rc == 0 and len(lines) == 106
    assert len(set(lines)) == 106


def test_cli_grid_wave_prints_only_that_wave():
    for wave, n in (("vctl", 6), ("zref", 50), ("rgen", 50)):
        rc, out = _run(["grid", "--wave", wave])
        lines = [l for l in out.splitlines() if l.strip()]
        assert rc == 0 and len(lines) == n
        assert all(f" {wave} " in " " + l + " " for l in lines)


def test_cli_check_exits_zero_on_a_valid_cell(tmp_path):
    p = _write_cell(tmp_path, "rgen")
    rc, out = _run(["check", "--metrics", p, "--arm", "C4L", "--cell", "rgen",
                    "--step", "40000", "--seed", "42", "--k", "8",
                    "--pin", PIN, "--ckpt-sha", CKPT_SHA,
                    "--expected-count", str(COUNT)])
    assert rc == 0, out


def test_cli_check_exits_nonzero_and_names_the_reason(tmp_path):
    m = _metrics("rgen")
    m["cond_autocast"] = "default"
    p = _write_cell(tmp_path, "rgen", metrics=m)
    rc, out = _run(["check", "--metrics", p, "--arm", "C4L", "--cell", "rgen",
                    "--step", "40000", "--seed", "42", "--k", "8",
                    "--pin", PIN, "--ckpt-sha", CKPT_SHA])
    assert rc == 1 and "cond_autocast" in out


def test_cli_check_exits_three_when_the_artifact_is_absent(tmp_path):
    rc, out = _run(["check", "--metrics", str(tmp_path / "gone.json"),
                    "--arm", "C4L", "--cell", "zref", "--step", "40000",
                    "--seed", "42", "--k", "8", "--pin", PIN])
    assert rc == 3, out


def test_cli_check_refuses_an_unregistered_cell(tmp_path):
    rc, out = _run(["check", "--metrics", str(tmp_path / "x.json"),
                    "--arm", "VANL", "--cell", "vctl", "--step", "40000",
                    "--seed", "42", "--k", "8", "--rotate-deg", "45",
                    "--pin", PIN])
    assert rc == 2 and ("not registered" in out or "unregistered" in out)


def test_cli_classify_reports_one_line_per_cell_with_a_status(tmp_path):
    rc, out = _run(["classify", "--wave", "vctl", "--output-root", str(tmp_path),
                    "--pin", PIN])
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 6
    # no checkpoints under a synthetic output root -> every cell is MISSING
    assert all(" MISSING" in l for l in lines), lines
    assert rc == 0


# --------------------------------------------------------------------------
# 9. FB1 (review B1): the argv the SCREEN DRIVER actually renders
#
# Every completed rgen/zref job used to hand the validator `--rotate-deg 0`,
# which argparse kept as the STRING "0"; `"0" not in (None, 0, 0.0)` made the
# check a usage error, so 100 of 106 cells would have failed validation AFTER
# spending their GPU. Two independent fixes, both pinned here: the driver passes
# no angle outside vctl, and the parser is tolerant of every spelling of "no
# angle" it could still be handed.
# --------------------------------------------------------------------------
def test_check_argv_carries_an_angle_only_for_vctl():
    for c in V.expected_grid():
        argv = V.check_argv(c, "/m.json", pin=PIN, ckpt_sha=CKPT_SHA)
        assert ("--rotate-deg" in argv) == (c.cell == "vctl"), c


def test_check_argv_round_trips_through_main(tmp_path):
    for cell in ("rgen", "zref", "vctl"):
        d = tmp_path / cell
        d.mkdir()
        p = _write_cell(d, cell)
        c = _cell(cell)
        rc, out = _run(V.check_argv(c, p, pin=PIN, ckpt_sha=CKPT_SHA,
                                    expected_count=COUNT))
        assert rc == 0, (cell, out)


@pytest.mark.parametrize("cell,deg", [
    ("rgen", "0"), ("zref", "0"), ("rgen", "0.0"), ("zref", None),
])
def test_cli_check_accepts_every_spelling_of_no_angle(tmp_path, cell, deg):
    # The string "0" is what the shell renders; it is NOT a usage error.
    p = _write_cell(tmp_path, cell)
    argv = ["check", "--metrics", p, "--arm", "C4L", "--cell", cell,
            "--step", "40000", "--seed", "42", "--k", "8", "--pin", PIN,
            "--ckpt-sha", CKPT_SHA, "--expected-count", str(COUNT)]
    if deg is not None:
        argv += ["--rotate-deg", deg]
    rc, out = _run(argv)
    assert rc == 0, out


@pytest.mark.parametrize("deg", ["45", "45.0"])
def test_cli_check_parses_a_vctl_angle_as_a_float(tmp_path, deg):
    m = _metrics("vctl")
    m["rotate_deg"] = 45.0
    sm = _screenmeta("vctl")
    sm["rotate_deg"] = 45.0
    sm["eval_name"] = m["eval_name"] = "exp14_C4L_vctl_S40000_s42_K8_rot45"
    st = _stream("vctl")
    st["rotate_deg"] = 45.0
    st["offsets"] = [64] * COUNT
    st["assignment_tuples"] = [[i, t[1], 64] for i, t in enumerate(st["input_tuples"])]
    st["assignment_hash"] = V.canonical_stream_hash(st["assignment_tuples"])
    p = _write_cell(tmp_path, "vctl", metrics=m, meta=sm, stream=st)
    rc, out = _run(["check", "--metrics", p, "--arm", "C4L", "--cell", "vctl",
                    "--step", "40000", "--seed", "42", "--k", "8",
                    "--rotate-deg", deg, "--pin", PIN, "--ckpt-sha", CKPT_SHA,
                    "--expected-count", str(COUNT)])
    assert rc == 0, out


def test_cli_check_still_refuses_a_real_angle_on_a_random_cell(tmp_path):
    p = _write_cell(tmp_path, "rgen")
    rc, out = _run(["check", "--metrics", p, "--arm", "C4L", "--cell", "rgen",
                    "--step", "40000", "--seed", "42", "--k", "8",
                    "--rotate-deg", "45", "--pin", PIN, "--ckpt-sha", CKPT_SHA])
    assert rc == 2, out


def test_cli_argv_prints_the_same_argv_the_driver_must_render():
    rc, out = _run(["argv", "--arm", "C4L", "--cell", "vctl", "--step", "40000",
                    "--seed", "42", "--k", "8", "--rotate-deg", "45",
                    "--metrics", "/m.json", "--pin", PIN, "--ckpt-sha", CKPT_SHA,
                    "--expected-count", str(COUNT)])
    assert rc == 0
    printed = out.split()
    c45 = V.Cell("C4L", "vctl", 40000, 42, 8, 45.0)
    assert printed == V.check_argv(c45, "/m.json", pin=PIN,
                                   ckpt_sha=CKPT_SHA, expected_count=COUNT)
