"""Tests for the exp_11 row validator (plan §5, Yixun Q7 standing request).

Every number that reaches ``model_comparison.md`` must be provably from the
protocol it claims: right arm, right orbit, right checkpoint step, right K, cfg
1.0, bf16 conditioning autocast, EMA weights, the five eval seeds exactly once,
and — since exp_11 — the batched orbit execution with its cap and source SHA, so
a legacy-loop row can never be averaged in with batched ones.

The validator is fail-closed by construction: it reads the metrics JSON *and* the
screen sidecar written next to it, and anything it cannot prove is a problem, not
a default. These tests drive each check from a known-good row by mutating exactly
one field at a time.
"""
import importlib.util
import json
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_VALIDATOR_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude", "exp11_validate_rows.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_validate_rows", _VALIDATOR_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_VALIDATOR_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_module()

CKPT = ("outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/"
        "epoch=2-step=10000.ckpt")


def _record(**over):
    rec = {
        "metrics": {"T60": 12.3, "C50": 1.1},
        "ckpt_path": CKPT,
        "rotate_deg": 0.0,
        "cond_method": "fa_invariant",
        "frame_avg_angles": [k * 360.0 / 8 for k in range(8)],
        "cond_autocast": "bf16",
        "orbit_execution": "batched",
        "frame_avg_fwd_cap": 64,
        "source_sha": "a" * 40,
        "batch_size": 64,
        "n_samples": 6337,
    }
    rec.update(over)
    return rec


def _sidecar(**over):
    side = {
        "arm": "C8", "step": 10000, "seed": 42, "K": 8,
        "eval_name": "exp11_C8_screen_S10000_s42_K8",
        "cfg_scale": 1.0, "steps": 1,
        "model_config": "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json",
        "model_config_sha256": "b" * 64,
        "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
        "ckpt_path": CKPT, "ckpt_sha256": "c" * 64,
        "use_ema": True,
        "frame_avg_angles": [k * 360.0 / 8 for k in range(8)],
        "cond_method": "fa_invariant", "cond_autocast": "bf16",
        "commit": "d" * 40,
    }
    side.update(over)
    return side


def _write_row(tmp_path, rec=None, side=None, name=None):
    name = name or "epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json"
    path = tmp_path / name
    path.write_text(json.dumps(rec if rec is not None else _record()))
    if side is not False:
        V.sidecar_path_for(str(path))
        with open(V.sidecar_path_for(str(path)), "w") as fh:
            json.dump(side if side is not None else _sidecar(), fh)
    return str(path)


# --------------------------------------------------------------------------- #
# 1. eval-name schema (plan §4)
# --------------------------------------------------------------------------- #
def test_parse_eval_name_screen():
    got = V.parse_eval_name("exp11_C16_screen_S12500_s42_K8")
    assert got == {"arm": "C16", "cell": "screen", "step": 12500, "seed": 42, "K": 8}


def test_parse_eval_name_backfill():
    got = V.parse_eval_name("exp11_C4backfill_S20000_s42_K8")
    assert got["arm"] == "C4BACKFILL" and got["step"] == 20000 and got["K"] == 8


@pytest.mark.parametrize("bad", [
    "exp11_C8_screen_S10000_s42",          # no K
    "exp11_C8_S10000_s42_K8",              # no cell
    "exp07_C8_screen_S10000_s42_K8",       # wrong experiment
    "exp11_C7_screen_S10000_s42_K8",       # not an arm
    "",
])
def test_parse_eval_name_rejects_malformed(bad):
    with pytest.raises(ValueError):
        V.parse_eval_name(bad)


def test_orbit_for_each_arm():
    assert V.orbit_for("C4L") == [0.0, 90.0, 180.0, 270.0]
    assert V.orbit_for("C4BACKFILL") == [0.0, 90.0, 180.0, 270.0]
    assert len(V.orbit_for("C32")) == 32 and V.orbit_for("C32")[1] == pytest.approx(11.25)
    with pytest.raises(ValueError):
        V.orbit_for("VAN")


# --------------------------------------------------------------------------- #
# 2. a good row validates
# --------------------------------------------------------------------------- #
def test_good_row_passes(tmp_path):
    row, problems = V.validate_row(_write_row(tmp_path))
    assert problems == [], problems
    assert row["arm"] == "C8" and row["step"] == 10000 and row["seed"] == 42 and row["K"] == 8


def test_sidecar_is_required(tmp_path):
    path = _write_row(tmp_path, side=False)
    _row, problems = V.validate_row(path)
    assert problems and any("sidecar" in p for p in problems)


# --------------------------------------------------------------------------- #
# 3. each protocol field is actually checked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,bad,needle", [
    ("cond_method", "vanilla", "cond_method"),
    ("cond_autocast", "default", "cond_autocast"),
    ("orbit_execution", "loop", "orbit_execution"),
    ("frame_avg_fwd_cap", 8, "frame_avg_fwd_cap"),
    ("source_sha", "", "source_sha"),
])
def test_metrics_record_fields_are_checked(tmp_path, field, bad, needle):
    path = _write_row(tmp_path, rec=_record(**{field: bad}))
    _row, problems = V.validate_row(path)
    assert any(needle in p for p in problems), problems


def test_wrong_orbit_for_the_arm_is_rejected(tmp_path):
    c4_angles = [0.0, 90.0, 180.0, 270.0]                  # C8 row carrying C4 angles
    path = _write_row(tmp_path, rec=_record(frame_avg_angles=c4_angles),
                      side=_sidecar(frame_avg_angles=c4_angles))
    _row, problems = V.validate_row(path)
    assert any("frame_avg_angles" in p for p in problems), problems


def test_integer_angles_are_rejected(tmp_path):
    """45 is not 45.0: the arms' orbits are floats and a row must match exactly."""
    ints = [0, 45, 90, 135, 180, 225, 270, 315]
    path = _write_row(tmp_path, rec=_record(frame_avg_angles=ints), side=_sidecar(frame_avg_angles=ints))
    _row, problems = V.validate_row(path)
    assert any("frame_avg_angles" in p for p in problems)


@pytest.mark.parametrize("field,bad,needle", [
    ("cfg_scale", 3.0, "cfg_scale"),
    ("steps", 8, "steps"),
    ("use_ema", False, "use_ema"),
    ("K", 1, "K"),
    ("seed", 43, "seed"),
    ("step", 12500, "step"),
    ("arm", "C4L", "arm"),
    ("dataset_config", "src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json", "dataset_config"),
])
def test_sidecar_fields_are_checked(tmp_path, field, bad, needle):
    path = _write_row(tmp_path, side=_sidecar(**{field: bad}))
    _row, problems = V.validate_row(path)
    assert any(needle in p for p in problems), problems


def test_k_must_match_the_dataset_config(tmp_path):
    """K=1 must use the _1 dataset config; claiming K=1 on the K=8 split fails."""
    path = _write_row(tmp_path, side=_sidecar(
        K=1, eval_name="exp11_C8_screen_S10000_s42_K1"),
        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K1_fa_invariant_a8.json")
    _row, problems = V.validate_row(path)
    assert any("dataset_config" in p or "K" in p for p in problems)


def test_ckpt_step_must_match_the_claimed_step(tmp_path):
    other = CKPT.replace("step=10000", "step=12500")
    path = _write_row(tmp_path, rec=_record(ckpt_path=other), side=_sidecar(ckpt_path=other))
    _row, problems = V.validate_row(path)
    assert any("ckpt" in p for p in problems), problems


def test_ckpt_must_live_in_the_arms_own_run_dir(tmp_path):
    foreign = CKPT.replace("exp11_C8", "exp11_C4L")
    path = _write_row(tmp_path, rec=_record(ckpt_path=foreign), side=_sidecar(ckpt_path=foreign))
    _row, problems = V.validate_row(path)
    assert any("ckpt" in p for p in problems), problems


def test_metrics_and_sidecar_must_agree(tmp_path):
    """The sidecar cannot claim a protocol the metrics record contradicts."""
    path = _write_row(tmp_path, side=_sidecar(cond_autocast="off"))
    _row, problems = V.validate_row(path)
    assert any("disagree" in p or "cond_autocast" in p for p in problems)


def test_empty_metrics_are_rejected(tmp_path):
    path = _write_row(tmp_path, rec=_record(metrics={}))
    _row, problems = V.validate_row(path)
    assert any("metrics" in p for p in problems)


# --------------------------------------------------------------------------- #
# 4. a CELL: the five eval seeds exactly once
# --------------------------------------------------------------------------- #
def _seed_row(tmp_path, seed, arm="C8", step=10000, k=8):
    name = (f"epoch=2-step={step}_metrics_1_1.0_exp11_{arm}_screen_S{step}_s{seed}_K{k}"
            f"_fa_invariant_a8.json")
    return _write_row(
        tmp_path,
        rec=_record(),
        side=_sidecar(seed=seed, eval_name=f"exp11_{arm}_screen_S{step}_s{seed}_K{k}"),
        name=name)


def test_cell_with_all_five_seeds_passes(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45, 46)]
    rows, problems = V.validate_cell(paths, arm="C8", step=10000, expected_seeds=(42, 43, 44, 45, 46), k=8)
    assert problems == [], problems
    assert sorted(r["seed"] for r in rows) == [42, 43, 44, 45, 46]


def test_cell_missing_a_seed_fails(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, expected_seeds=(42, 43, 44, 45, 46), k=8)
    assert any("46" in p for p in problems)


def test_cell_with_a_duplicated_seed_fails(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45, 46)]
    dup = _write_row(tmp_path, side=_sidecar(seed=46, eval_name="exp11_C8_screen_S10000_s46_K8"),
                     name="dup_metrics_1_1.0_exp11_C8_screen_S10000_s46_K8_fa_invariant_a8.json")
    _rows, problems = V.validate_cell(paths + [dup], arm="C8", step=10000,
                                      expected_seeds=(42, 43, 44, 45, 46), k=8)
    assert any("more than once" in p or "duplicate" in p for p in problems)


def test_single_seed_screen_cell_passes(tmp_path):
    """A futility screen is one seed by design (plan §4)."""
    rows, problems = V.validate_cell([_seed_row(tmp_path, 42)], arm="C8", step=10000,
                                     expected_seeds=(42,), k=8)
    assert problems == [] and len(rows) == 1


def test_cell_rejects_a_row_from_another_arm(tmp_path):
    paths = [_seed_row(tmp_path, 42), _seed_row(tmp_path, 43, arm="C4L")]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, expected_seeds=(42, 43), k=8)
    assert problems


def test_main_returns_nonzero_on_a_bad_cell(tmp_path, capsys):
    paths = [_seed_row(tmp_path, 42)]
    rc = V.main(["--arm", "C8", "--step", "10000", "--k", "8", "--seeds", "42,43", *paths])
    assert rc != 0
    assert "43" in capsys.readouterr().out


def test_main_returns_zero_on_a_good_cell(tmp_path, capsys):
    paths = [_seed_row(tmp_path, s) for s in (42, 43)]
    rc = V.main(["--arm", "C8", "--step", "10000", "--k", "8", "--seeds", "42,43", *paths])
    assert rc == 0
    assert "VALIDATED" in capsys.readouterr().out
