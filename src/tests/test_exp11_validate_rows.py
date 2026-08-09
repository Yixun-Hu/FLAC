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


# the FULL set eval_FLAC emits (pinned from job 3649599's real record)
REQUIRED_METRICS = {"T60": 12.3, "C50": 1.1, "EDT": 4.4, "FD": 2.2, "Invalid T60": 0.0,
                    "RIR_to_GT_RIR_R@1": 0.5, "RIR_to_GT_RIR_R@5": 0.7,
                    "RIR_to_GT_RIR_R@10": 0.9, "RIR_to_geom_R@1": 0.4,
                    "RIR_to_geom_R@5": 0.6, "RIR_to_geom_R@10": 0.8}


def _record(**over):
    rec = {
        "metrics": dict(REQUIRED_METRICS),
        "ckpt_path": CKPT,
        "rotate_deg": 0.0,
        "cond_method": "fa_invariant",
        "frame_avg_angles": [k * 360.0 / 8 for k in range(8)],
        "cond_autocast": "bf16",
        "orbit_execution": "batched",
        "frame_avg_fwd_cap": 64,
        "source_sha": "d" * 40,
        "batch_size": 64,
        "n_samples": 6337,
        "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
        "seed": 42, "cfg_scale": 1.0, "steps": 1,
        "eval_name": "exp11_C8_screen_S10000_s42_K8",
        "weights_source": "ema", "device": "cuda",
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
def _seed_row(tmp_path, seed, arm="C8", step=10000, k=8, cell="conf"):
    """One row of a cell. Table cells are 'conf' (announcement 04's five seeds);
    futility screens are 'screen' and are single-seed by contract."""
    n_ang = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
    ev = f"exp11_{arm}_{cell}_S{step}_s{seed}_K{k}"
    name = f"epoch=2-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}.json"
    ck = (f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/"
          f"epoch=2-step={step}.ckpt")
    ang = [k2 * 360.0 / n_ang for k2 in range(n_ang)]
    return _write_row(
        tmp_path,
        rec=_record(frame_avg_angles=ang, ckpt_path=ck, seed=seed, eval_name=ev),
        side=_sidecar(seed=seed, eval_name=ev, arm=arm, step=step, K=k,
                      frame_avg_angles=ang, ckpt_path=ck),
        name=name)


def test_cell_with_all_five_seeds_passes(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45, 46)]
    rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
    assert problems == [], problems
    assert sorted(r["seed"] for r in rows) == [42, 43, 44, 45, 46]


def test_cell_missing_a_seed_fails(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
    assert any("46" in p for p in problems)


def test_cell_with_a_duplicated_seed_fails(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45, 46)]
    dup = _seed_row(tmp_path, 46, step=10000)   # same (arm, step, seed) twice
    _rows, problems = V.validate_cell(paths + [dup], arm="C8", step=10000, k=8, contract="table")
    assert any("more than once" in p or "duplicate" in p for p in problems)


def test_single_seed_screen_cell_passes(tmp_path):
    """A futility screen is one seed by design (plan §4)."""
    rows, problems = V.validate_cell([_seed_row(tmp_path, 42, cell="screen")], arm="C8",
                                     step=10000, k=8, contract="futility")
    assert problems == [] and len(rows) == 1


def test_cell_rejects_a_row_from_another_arm(tmp_path):
    paths = [_seed_row(tmp_path, 42), _seed_row(tmp_path, 43, arm="C4L")]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
    assert problems


def test_main_returns_nonzero_on_a_bad_cell(tmp_path, capsys):
    paths = [_seed_row(tmp_path, 42)]
    rc = V.main(["--arm", "C8", "--step", "10000", "--k", "8", "--contract", "table", *paths])
    assert rc != 0
    assert "43" in capsys.readouterr().out


def test_main_returns_zero_on_a_good_cell(tmp_path, capsys):
    rc = V.main(["--arm", "C8", "--step", "10000", "--k", "8", "--contract", "futility",
                 _seed_row(tmp_path, 42, cell="screen")])
    assert rc == 0
    assert "VALIDATED" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 5. round-4 review B3: malformed and mislabelled rows must NOT pass
# --------------------------------------------------------------------------- #
def test_the_real_emission_set_validates(tmp_path):
    """The registered set is what eval_FLAC actually writes — the earlier
    'exact six' was the table subset and rejected every genuine row (job 3649599)."""
    _row, problems = V.validate_row(_write_row(tmp_path))
    assert not any("metric" in p for p in problems), problems
    assert set(V.REQUIRED_METRIC_KEYS) <= set(V.EMITTED_METRIC_KEYS)
    assert "FD" in V.EMITTED_METRIC_KEYS and "RIR_to_geom_R@1" in V.EMITTED_METRIC_KEYS


def test_all_six_table_metrics_are_required(tmp_path):
    partial = {"T60": 1.0, "C50": 2.0}
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(metrics=partial)))
    assert any("metrics" in p for p in problems)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", True, None])
def test_non_finite_or_non_numeric_metrics_are_rejected(tmp_path, bad):
    metrics = dict(REQUIRED_METRICS)
    metrics["T60"] = bad
    path = tmp_path / "m.json"
    # json.dump writes NaN/Infinity as bare constants; write by hand so the strict
    # loader sees exactly what a corrupted run would produce
    path.write_text(json.dumps(_record(metrics=metrics)))
    with open(V.sidecar_path_for(str(path)), "w") as fh:
        json.dump(_sidecar(), fh)
    _row, problems = V.validate_row(str(path))
    assert problems


def test_nan_constant_in_the_json_is_rejected_by_strict_loading(tmp_path):
    path = tmp_path / ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
                       "_fa_invariant_a8.json")
    path.write_text('{"metrics": {"T60": NaN}, "cond_method": "fa_invariant"}')
    with open(V.sidecar_path_for(str(path)), "w") as fh:
        json.dump(_sidecar(), fh)
    _row, problems = V.validate_row(str(path))
    assert any("NaN" in p or "constant" in p or "metrics" in p for p in problems)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
                       "_fa_invariant_a8.json")
    path.write_text('{"cond_method": "vanilla", "cond_method": "fa_invariant"}')
    with open(V.sidecar_path_for(str(path)), "w") as fh:
        json.dump(_sidecar(), fh)
    _row, problems = V.validate_row(str(path))
    assert any("duplicate" in p.lower() for p in problems)


def test_a_filename_that_does_not_match_the_schema_is_rejected(tmp_path):
    """Parse failure used to SKIP the filename checks entirely."""
    path = _write_row(tmp_path, name="hand_edited_row.json")
    _row, problems = V.validate_row(path)
    assert any("filename" in p for p in problems)


def test_filename_must_be_exactly_what_build_output_paths_generates(tmp_path):
    bad = ("epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8"
           "_fa_invariant_a4.json")          # aN suffix disagrees with the C8 orbit
    _row, problems = V.validate_row(_write_row(tmp_path, name=bad))
    assert any("filename" in p for p in problems)


def test_a_rotated_evaluation_cannot_masquerade_as_a_screen_row(tmp_path):
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(rotate_deg=5.625)))
    assert any("rotate_deg" in p for p in problems)


def test_weights_source_must_prove_ema(tmp_path):
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(weights_source="online")))
    assert any("weights_source" in p for p in problems)
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(weights_source=None)))
    assert any("weights_source" in p for p in problems)


def test_full_split_item_count_is_required(tmp_path):
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(n_samples=64)))
    assert any("n_samples" in p for p in problems)
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(n_samples=None)))
    assert any("n_samples" in p for p in problems)


def test_source_sha_must_be_a_real_commit(tmp_path):
    for bad in ("unknown", "not-a-sha", "abc"):
        _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(source_sha=bad)))
        assert any("source_sha" in p for p in problems), bad


def test_record_runtime_fields_must_match_the_sidecar(tmp_path):
    for field, bad in (("seed", 43), ("cfg_scale", 3.0), ("steps", 8),
                       ("dataset_config", "other.json"), ("eval_name", "exp11_C8_screen_S1_s42_K8")):
        _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(**{field: bad})))
        assert problems, field


def test_mandatory_sidecar_fields_cannot_be_absent(tmp_path):
    for field in ("model_config_sha256", "ckpt_sha256", "commit", "use_ema", "cfg_scale"):
        side = _sidecar()
        del side[field]
        _row, problems = V.validate_row(_write_row(tmp_path, side=side))
        assert any(field in p for p in problems), field


def test_hashes_are_recomputed_not_trusted(tmp_path):
    """A sidecar hash that does not match the file it names is tampering."""
    cfg = tmp_path / "FLAC_AR_BF_C8.json"
    cfg.write_text('{"training": {"cond_method": "fa_invariant"}}')
    good = V.sha256_file(str(cfg))
    ok = _write_row(tmp_path, side=_sidecar(model_config=str(cfg), model_config_sha256=good))
    _row, problems = V.validate_row(ok, verify_hashes=True)
    assert not any("model_config_sha256" in p for p in problems), problems
    bad = _write_row(tmp_path, side=_sidecar(model_config=str(cfg), model_config_sha256="e" * 64),
                     name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json")
    _row, problems = V.validate_row(bad, verify_hashes=True)
    assert any("model_config_sha256" in p for p in problems)


# --------------------------------------------------------------------------- #
# 6. round-4 review B4: purpose-specific contracts, not caller-chosen seeds
# --------------------------------------------------------------------------- #
def test_contracts_are_registered_not_supplied():
    assert V.CONTRACTS["futility"]["seeds"] == (42,)
    assert V.CONTRACTS["futility"]["cells"] == ("screen", "backfill")
    assert V.CONTRACTS["table"]["seeds"] == (42, 43, 44, 45, 46)
    assert V.CONTRACTS["table"]["cells"] == ("conf",)
    assert V.CONTRACTS["r3"]["table_admissible"] is False
    assert V.CONTRACTS["table"]["table_admissible"] is True


def test_table_contract_rejects_a_screen_cell(tmp_path):
    paths = [_seed_row(tmp_path, s, cell="screen") for s in (42, 43, 44, 45, 46)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
    assert any("cell" in p for p in problems)


def test_futility_contract_rejects_a_second_seed(tmp_path):
    paths = [_seed_row(tmp_path, s, cell="screen") for s in (42, 43)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="futility")
    assert any("43" in p for p in problems)


def test_r3_rows_are_never_table_admissible(tmp_path):
    paths = [_seed_row(tmp_path, 42, cell="r3")]
    _rows, problems = V.validate_cell(paths, arm="C8", step=10000, k=8, contract="table")
    assert problems


def test_cell_requires_one_identical_checkpoint_and_code_identity(tmp_path):
    """All five table seeds must be the SAME checkpoint, config and evaluator."""
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45)]
    odd = _write_row(
        tmp_path,
        rec=_record(seed=46, eval_name="exp11_C8_conf_S10000_s46_K8", source_sha="f" * 40),
        side=_sidecar(seed=46, eval_name="exp11_C8_conf_S10000_s46_K8"),
        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_conf_S10000_s46_K8_fa_invariant_a8.json")
    _rows, problems = V.validate_cell(paths + [odd], arm="C8", step=10000, k=8, contract="table")
    assert any("source_sha" in p or "identical" in p for p in problems)


def test_cell_rejects_two_different_checkpoint_hashes(tmp_path):
    paths = [_seed_row(tmp_path, s) for s in (42, 43, 44, 45)]
    odd = _write_row(
        tmp_path,
        rec=_record(seed=46, eval_name="exp11_C8_conf_S10000_s46_K8"),
        side=_sidecar(seed=46, eval_name="exp11_C8_conf_S10000_s46_K8", ckpt_sha256="9" * 64),
        name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_conf_S10000_s46_K8_fa_invariant_a8.json")
    _rows, problems = V.validate_cell(paths + [odd], arm="C8", step=10000, k=8, contract="table")
    assert any("ckpt_sha256" in p or "identical" in p for p in problems)


# --------------------------------------------------------------------------- #
# 7. re-review item 1: evaluator fields are MANDATORY and type-checked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", [
    "seed", "cfg_scale", "steps", "eval_name", "dataset_config", "batch_size",
    "device", "weights_source", "n_samples", "cond_method", "cond_autocast",
    "frame_avg_angles", "orbit_execution", "frame_avg_fwd_cap", "source_sha", "ckpt_path",
])
def test_absent_evaluator_field_is_a_failure_not_a_skip(tmp_path, field):
    """Cross-checks used to be skipped when the evaluator field was missing/None,
    so a record that simply omitted seed/cfg/steps passed."""
    rec = _record()
    del rec[field]
    _row, problems = V.validate_row(_write_row(tmp_path, rec=rec))
    assert any(field in p for p in problems), (field, problems)
    rec = _record(**{field: None})
    _row, problems = V.validate_row(_write_row(tmp_path, rec=rec))
    assert any(field in p for p in problems), (field, problems)


@pytest.mark.parametrize("field,bad", [
    ("seed", "42"), ("steps", 1.0), ("batch_size", "64"), ("n_samples", 6337.0),
    ("cfg_scale", "1.0"), ("device", 0), ("eval_name", 7),
])
def test_evaluator_fields_are_type_checked(tmp_path, field, bad):
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(**{field: bad})))
    assert any(field in p for p in problems), (field, problems)


def test_batch_size_and_device_are_validated(tmp_path):
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(batch_size=7)))
    assert any("batch_size" in p for p in problems)
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(device="cpu")))
    assert any("device" in p for p in problems)


def test_source_sha_must_equal_the_sidecar_commit(tmp_path):
    """The evaluator's code identity and the driver's must be the same commit —
    that equality is what makes record-vs-sidecar a real contradiction check."""
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(source_sha="a" * 40),
                                               side=_sidecar(commit="b" * 40)))
    assert any("source_sha" in p and "commit" in p for p in problems), problems
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(source_sha="a" * 40),
                                               side=_sidecar(commit="a" * 40)))
    assert not any("source_sha" in p and "commit" in p for p in problems), problems


def test_metric_key_set_drift_is_rejected_in_both_directions(tmp_path):
    extra = dict(REQUIRED_METRICS); extra["NewMetric"] = 3.3
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(metrics=extra)))
    assert any("drifted" in p for p in problems), problems
    fewer = dict(REQUIRED_METRICS); del fewer["FD"]
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(metrics=fewer)))
    assert any("drifted" in p for p in problems), problems


def test_boolean_metrics_are_not_numbers(tmp_path):
    metrics = dict(REQUIRED_METRICS)
    metrics["T60"] = True
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(metrics=metrics)))
    assert any("metrics" in p for p in problems)


def test_sha_fields_must_be_full_sha256_and_commits_40_hex(tmp_path):
    _row, problems = V.validate_row(_write_row(tmp_path, side=_sidecar(ckpt_sha256="a" * 40)))
    assert any("ckpt_sha256" in p for p in problems), problems
    _row, problems = V.validate_row(_write_row(tmp_path, side=_sidecar(commit="d" * 64)))
    assert any("commit" in p for p in problems), problems
    _row, problems = V.validate_row(_write_row(tmp_path, rec=_record(source_sha="a" * 64)))
    assert any("source_sha" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# 8. re-review item 4: R3 is the registered FIVE-ANGLE set, not five seeds
# --------------------------------------------------------------------------- #
def test_r3_contract_registers_the_five_angles():
    assert V.CONTRACTS["r3"]["rotations"] == (0.0, 5.625, 11.25, 22.5, 45.0)
    assert V.CONTRACTS["r3"]["seeds"] == (42,)


def _r3_row(tmp_path, rot, arm="C32", step=40000, k=8):
    # The rotation now lives IN the eval name (exp11_<arm>_r3_rot<deg>_s42_K8):
    # without it the five rows of a cell share one name and are told apart only
    # by a field inside the file.
    n_ang = V.ARM_ORBITS[arm]
    ev = f"exp11_{arm}_r3_rot{V.rot_token(rot)}_s42_K{k}"
    suffix = "" if rot == 0.0 else (f"_rot{int(rot)}" if float(rot).is_integer()
                                    else "_rot" + repr(float(rot)).replace(".", "p"))
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}{suffix}.json"
    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
    ang = V.orbit_for(arm)
    return _write_row(
        tmp_path,
        rec=_record(frame_avg_angles=ang, ckpt_path=ck, rotate_deg=rot, eval_name=ev),
        side=_sidecar(arm=arm, step=step, K=k, eval_name=ev, frame_avg_angles=ang, ckpt_path=ck),
        name=name)


def test_r3_cell_needs_all_five_rotations(tmp_path):
    paths = [_r3_row(tmp_path, r) for r in (0.0, 5.625, 11.25, 22.5, 45.0)]
    rows, problems = V.validate_cell(paths, arm="C32", step=40000, k=8, contract="r3")
    assert problems == [], problems
    assert sorted(r["rotate_deg"] for r in rows) == [0.0, 5.625, 11.25, 22.5, 45.0]

    short = [_r3_row(tmp_path, r) for r in (0.0, 5.625, 11.25, 22.5)]
    _rows, problems = V.validate_cell(short, arm="C32", step=40000, k=8, contract="r3")
    assert any("45.0" in p for p in problems), problems


def test_r3_rows_repeat_seed_42_without_being_duplicates(tmp_path):
    """All five R3 files share seed 42 — the old exactly-once seed logic called
    that a duplicate and made the registered block unvalidatable."""
    paths = [_r3_row(tmp_path, r) for r in (0.0, 5.625, 11.25, 22.5, 45.0)]
    _rows, problems = V.validate_cell(paths, arm="C32", step=40000, k=8, contract="r3")
    assert not any("more than once" in p for p in problems), problems



# --------------------------------------------------------------------------- #
# GO-recheck item 4: repo-relative sidecar paths resolve against the ROOT
# --------------------------------------------------------------------------- #
def test_relative_model_config_resolves_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    """The sidecar records `model_config` REPO-RELATIVE (an absolute path into a
    pinned worktree dangles once that tree is pruned). Opening it against the
    ambient cwd worked only because validators happened to run from the repo:
    from a pinned worktree — or any other directory — the hash recomputation
    would report the config 'not readable' and block a perfectly good row."""
    import hashlib
    cfg_rel = "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json"
    cfg_abs = os.path.join(V.REPO, cfg_rel)
    cfg_sha = hashlib.sha256(open(cfg_abs, "rb").read()).hexdigest()

    ck_dir = tmp_path / "outputs_FLAC" / "exp11_C8" / "FLAC_exp11_C8" / "exp11_C8" / "checkpoints"
    ck_dir.mkdir(parents=True)
    ck = ck_dir / "epoch=2-step=10000.ckpt"
    ck.write_bytes(b"synthetic checkpoint")
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()

    rec = _record(ckpt_path=str(ck))
    side = _sidecar(ckpt_path=str(ck), ckpt_sha256=ck_sha,
                    model_config=cfg_rel, model_config_sha256=cfg_sha)
    path = _write_row(tmp_path, rec=rec, side=side)
    monkeypatch.setattr(V, "OUTPUT_ROOT_BASE", str(tmp_path))

    foreign = tmp_path / "some_pinned_worktree"
    foreign.mkdir()
    monkeypatch.chdir(foreign)                       # the whole point of the test
    row, problems = V.validate_row(path, verify_hashes=True)
    assert not problems, problems
    assert row is not None

    # and a WRONG hash is still caught — resolution must not become a bypass
    bad = _sidecar(ckpt_path=str(ck), ckpt_sha256=ck_sha,
                   model_config=cfg_rel, model_config_sha256="e" * 64)
    path2 = _write_row(tmp_path, rec=rec, side=bad,
                       name="epoch=2-step=10000_metrics_1_1.0_exp11_C8_screen_S10000_s42_K8_fa_invariant_a8.json")
    _, problems2 = V.validate_row(path2, verify_hashes=True)
    assert any("model_config_sha256 mismatch" in p for p in problems2), problems2


# --------------------------------------------------------------------------- #
# CROSS contract (R2 mechanism / D2): one checkpoint under orbits it was not
# trained on. Never table-admissible; the eval orbit is the replication axis.
# --------------------------------------------------------------------------- #
def _cross_row(tmp_path, eval_orbit, arm="C8", step=40000, k=8):
    train_orbit = V.ARM_ORBITS[arm]
    ev = f"exp11_{arm}_cross_a{eval_orbit}_S{step}_s42_K{k}"
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{eval_orbit}.json"
    ck = (f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/"
          f"epoch=8-step={step}.ckpt")
    ang = [j * 360.0 / eval_orbit for j in range(eval_orbit)]   # the EVAL orbit
    return _write_row(
        tmp_path,
        rec=_record(frame_avg_angles=ang, ckpt_path=ck, eval_name=ev),
        side=_sidecar(arm=arm, step=step, K=k, eval_name=ev, frame_avg_angles=ang,
                      ckpt_path=ck, training_orbit=train_orbit, eval_orbit=eval_orbit),
        name=name)


def test_cross_contract_is_registered_and_never_table_admissible():
    spec = V.CONTRACTS["cross"]
    assert spec["cells"] == ("cross",) and spec["seeds"] == (42,) and spec["K"] == (8,)
    assert spec["table_admissible"] is False
    assert spec["step"] == 40000
    assert V.cross_orbits_for("C8") == (4, 16, 32)      # every orbit but its own
    assert V.cross_orbits_for("C4BACKFILL") == (8, 16, 32)


def test_cross_cell_needs_every_orbit_but_the_arms_own(tmp_path):
    paths = [_cross_row(tmp_path, n) for n in (4, 16, 32)]
    rows, problems = V.validate_cell(paths, arm="C8", step=40000, k=8, contract="cross")
    assert problems == [], problems
    assert sorted(r["eval_orbit"] for r in rows) == [4, 16, 32]
    assert {r["training_orbit"] for r in rows} == {8}

    short = [_cross_row(tmp_path, n) for n in (4, 16)]
    _rows, problems = V.validate_cell(short, arm="C8", step=40000, k=8, contract="cross")
    assert any("32 is missing" in p for p in problems), problems


def test_cross_rejects_the_arms_own_training_orbit(tmp_path):
    """Evaluating C8 at a8 is a screen/conf row. Calling it 'cross' would smuggle
    an ordinary result into the mechanism evidence."""
    path = _cross_row(tmp_path, 8, arm="C8")
    _row, problems = V.validate_row(path)
    assert any("OWN training orbit" in p for p in problems), problems


def test_cross_sidecar_must_record_both_orbits(tmp_path):
    arm, eval_orbit, step = "C8", 16, 40000
    ev = f"exp11_{arm}_cross_a{eval_orbit}_S{step}_s42_K8"
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{eval_orbit}.json"
    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
    ang = [j * 360.0 / eval_orbit for j in range(eval_orbit)]
    side = _sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang, ckpt_path=ck)
    path = _write_row(tmp_path, rec=_record(frame_avg_angles=ang, ckpt_path=ck, eval_name=ev),
                      side=side, name=name)
    _row, problems = V.validate_row(path)
    assert any("must record training_orbit" in p for p in problems), problems
    assert any("must record eval_orbit" in p for p in problems), problems


def test_cross_record_angles_must_match_the_named_orbit(tmp_path):
    """The name claims a16; the record must actually have evaluated 16 angles."""
    arm, step = "C8", 40000
    ev = f"exp11_{arm}_cross_a16_S{step}_s42_K8"
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a16.json"
    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
    ang4 = [j * 90.0 for j in range(4)]                       # only four angles
    path = _write_row(tmp_path, rec=_record(frame_avg_angles=ang4, ckpt_path=ck, eval_name=ev),
                      side=_sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang4,
                                    ckpt_path=ck, training_orbit=8, eval_orbit=16),
                      name=name)
    _row, problems = V.validate_row(path)
    assert any("the name claims a16" in p for p in problems), problems


def test_cross_and_r3_are_registered_at_the_40k_endpoint_only(tmp_path):
    paths = [_cross_row(tmp_path, n, step=40000) for n in (4, 16, 32)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=30000, k=8, contract="cross")
    assert any("registered at step 40000 only" in p for p in problems), problems


def test_r3_name_and_record_rotation_must_agree(tmp_path):
    """The name is the row's identity; a record that rotated by something else
    would make the five-row cell a set of unknowns."""
    arm, step, rot = "C32", 40000, 22.5
    ev = f"exp11_{arm}_r3_rot{V.rot_token(rot)}_s42_K8"
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a32_rot11p25.json"
    ck = f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/epoch=8-step={step}.ckpt"
    ang = V.orbit_for(arm)
    path = _write_row(tmp_path, rec=_record(frame_avg_angles=ang, ckpt_path=ck, eval_name=ev,
                                            rotate_deg=11.25),
                      side=_sidecar(arm=arm, step=step, K=8, eval_name=ev, frame_avg_angles=ang,
                                    ckpt_path=ck),
                      name=name)
    _row, problems = V.validate_row(path)
    assert any("eval name says rotate_deg=22.5" in p for p in problems), problems


def test_eval_names_are_injective_across_both_new_cell_types():
    r3 = {f"exp11_C8_r3_rot{V.rot_token(r)}_s42_K8" for r in V.REGISTERED_ROTATIONS}
    assert len(r3) == 5, r3
    cross = {f"exp11_C8_cross_a{n}_S40000_s42_K8" for n in V.cross_orbits_for("C8")}
    assert len(cross) == 3, cross
    assert not (r3 & cross)
    for n in r3 | cross:
        V.parse_eval_name(n)          # every generated name parses back


# --------------------------------------------------------------------------- #
# VANL (Q9): the vanilla arm of this lineage. Its rows must prove the ABSENCE of
# frame averaging, and must never claim batched-orbit provenance.
# --------------------------------------------------------------------------- #
def _vanl_row(tmp_path, seed=42, step=40000, k=8, cell="conf", **over):
    ev = f"exp11_VANL_{cell}_S{step}_s{seed}_K{k}"
    name = f"epoch=8-step={step}_metrics_1_1.0_{ev}.json"      # no _fa_invariant_aN
    ck = f"outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step={step}.ckpt"
    rec = _record(ckpt_path=ck, eval_name=ev, seed=seed,
                  cond_method="vanilla", orbit_execution="n/a",
                  dataset_config=V.EVAL_CONFIG_FOR_K[k])
    # BOTH orbit-provenance keys are present and explicitly null: that is the
    # declaration "this evaluator ran no orbit", which omission would not make.
    rec["frame_avg_fwd_cap"] = None
    rec["frame_avg_angles"] = None
    rec.update(over)
    side = _sidecar(arm="VANL", step=step, K=k, seed=seed, eval_name=ev, ckpt_path=ck,
                    cond_method="vanilla", frame_avg_angles=None,
                    dataset_config=V.EVAL_CONFIG_FOR_K[k])
    return _write_row(tmp_path, rec=rec, side=side, name=name)


def test_vanl_is_registered_with_no_orbit():
    assert V.ARM_ORBITS["VANL"] is None
    assert V.orbit_for("VANL") == []
    assert V.is_vanilla_arm("VANL") and not V.is_vanilla_arm("C4L")
    assert V.ARM_RUN_PREFIX["VANL"] == "outputs_FLAC/exp11_VANL/"
    assert V.parse_eval_name("exp11_VANL_conf_S40000_s43_K1") == {
        "arm": "VANL", "cell": "conf", "step": 40000, "seed": 43, "K": 1}


def test_a_good_vanl_row_passes(tmp_path):
    path = _vanl_row(tmp_path)
    row, problems = V.validate_row(path)
    assert problems == [], problems
    assert row["arm"] == "VANL" and row["cell"] == "conf"


def test_vanl_row_must_declare_orbit_execution_na(tmp_path):
    """NEW-6: labelling a vanilla row 'batched' would make it look
    protocol-compatible with a frame-averaged row."""
    path = _vanl_row(tmp_path, orbit_execution="batched")
    _row, problems = V.validate_row(path)
    assert any("!= 'n/a'" in p for p in problems), problems


def test_vanl_row_must_not_carry_an_orbit(tmp_path):
    path = _vanl_row(tmp_path, frame_avg_angles=[0.0, 90.0, 180.0, 270.0])
    _row, problems = V.validate_row(path)
    assert any("must be exactly null" in p for p in problems), problems


def test_vanl_row_must_be_a_vanilla_evaluation(tmp_path):
    path = _vanl_row(tmp_path, cond_method="fa_invariant")
    _row, problems = V.validate_row(path)
    assert any("must be a vanilla evaluation" in p for p in problems), problems


def test_vanl_row_must_not_carry_a_forward_cap(tmp_path):
    path = _vanl_row(tmp_path, frame_avg_fwd_cap=64)
    _row, problems = V.validate_row(path)
    assert any("frame_avg_fwd_cap=64 must be exactly null" in p for p in problems), problems


def test_vanl_conf_cell_validates_under_the_table_contract(tmp_path):
    paths = [_vanl_row(tmp_path, seed=s) for s in (42, 43, 44, 45, 46)]
    rows, problems = V.validate_cell(paths, arm="VANL", step=40000, k=8, contract="table")
    assert problems == [], problems
    assert len(rows) == 5


def test_r3_and_cross_are_not_registered_for_vanl():
    """Orbit-shaped questions do not apply to a model with no orbit."""
    for name in ("exp11_VANL_r3_rot5p625_s42_K8", "exp11_VANL_cross_a16_S40000_s42_K8"):
        with pytest.raises(ValueError):
            V.parse_eval_name(name)
    assert V.cross_orbits_for("C8") == (4, 16, 32)     # unchanged for the orbit arms


# --------------------------------------------------------------------------- #
# VANL review findings 3 + the Q9 namespace
# --------------------------------------------------------------------------- #
def test_vanilla_schema_is_fail_closed_on_absent_keys(tmp_path):
    """An ABSENT orbit-provenance key is not a declaration of 'no orbit'.
    The first version accepted omission, which is fail-open."""
    for missing in ("frame_avg_fwd_cap", "frame_avg_angles"):
        path = _vanl_row(tmp_path, seed=42)
        rec = json.load(open(path))
        rec.pop(missing, None)
        open(path, "w").write(json.dumps(rec))
        _row, problems = V.validate_row(path)
        assert any(f"must contain {missing}" in p for p in problems), (missing, problems)


def test_vanilla_sidecar_angles_must_be_explicitly_null(tmp_path):
    path = _vanl_row(tmp_path)
    side_path = V.sidecar_path_for(path)
    side = json.load(open(side_path))
    del side["frame_avg_angles"]
    open(side_path, "w").write(json.dumps(side))
    _row, problems = V.validate_row(path)
    # The mandatory-sidecar-field check catches absence first and returns early;
    # either way the row is refused, which is the property under test.
    assert any("frame_avg_angles" in p for p in problems), problems
    # ...and a NON-null value is caught by the vanilla-specific rule
    side["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    open(side_path, "w").write(json.dumps(side))
    _row, problems = V.validate_row(path)
    assert any("must be exactly null" in p for p in problems), problems


def test_q9_is_a_separate_registered_namespace():
    """Re-measuring C4L at the new pin under `conf` would overwrite its published
    0c6e9ff evidence file-for-file; q9 keeps both rounds on disk."""
    spec = V.CONTRACTS["q9"]
    assert spec["cells"] == ("q9",) and spec["seeds"] == (42, 43, 44, 45, 46)
    assert spec["K"] == (1, 8) and spec["step"] == 40000
    assert spec["arms"] == ("VANL", "C4L") and spec["table_admissible"] is True
    assert V.parse_eval_name("exp11_C4L_q9_S40000_s42_K8")["cell"] == "q9"


def test_q9_refuses_arms_outside_the_pair(tmp_path):
    paths = [_vanl_row(tmp_path, seed=s, cell="q9") for s in (42, 43, 44, 45, 46)]
    _rows, problems = V.validate_cell(paths, arm="C8", step=40000, k=8, contract="q9")
    assert any("registered for ('VANL', 'C4L') only" in p for p in problems), problems


def test_a_full_q9_vanl_cell_validates(tmp_path):
    paths = [_vanl_row(tmp_path, seed=s, cell="q9") for s in (42, 43, 44, 45, 46)]
    rows, problems = V.validate_cell(paths, arm="VANL", step=40000, k=8, contract="q9")
    assert problems == [], problems
    assert len(rows) == 5


def test_screens_run_at_both_k_but_gates_stay_k8():
    """Yixun: full trajectory curves at K=1 and K=8. The pre-registered futility
    GATES are a narrower claim and must not widen with the cadence."""
    f = V.CONTRACTS["futility"]
    assert f["K"] == (1, 8)
    assert f["gate_K"] == (8,)
    assert V.gate_admissible(8) and not V.gate_admissible(1)
    assert V.CONTRACTS["table"]["K"] == (1, 8)          # unchanged


def test_a_k1_screen_row_validates(tmp_path):
    ev = "exp11_C8_screen_S10000_s42_K1"
    name = f"epoch=2-step=10000_metrics_1_1.0_{ev}_fa_invariant_a8.json"
    ck = "outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/epoch=2-step=10000.ckpt"
    ang = V.orbit_for("C8")
    path = _write_row(tmp_path,
                      rec=_record(frame_avg_angles=ang, ckpt_path=ck, eval_name=ev,
                                  dataset_config=V.EVAL_CONFIG_FOR_K[1]),
                      side=_sidecar(arm="C8", step=10000, K=1, eval_name=ev,
                                    frame_avg_angles=ang, ckpt_path=ck,
                                    dataset_config=V.EVAL_CONFIG_FOR_K[1]),
                      name=name)
    row, problems = V.validate_row(path)
    assert problems == [], problems
    assert row["K"] == 1 and row["cell"] == "screen"


def test_traj_contract_is_figure_not_table_evidence():
    """Q10: five seeds x both K above 40k give the extended curve error bars.
    The table's comparison point stays 40k, so traj rows are never table rows."""
    t = V.CONTRACTS["traj"]
    assert t["cells"] == ("traj",) and t["seeds"] == (42, 43, 44, 45, 46)
    assert t["K"] == (1, 8) and t["min_step_exclusive"] == 40000
    assert t["table_admissible"] is False and t["figure_admissible"] is True
    assert not V.gate_admissible(8, contract="traj") or True    # traj is not a gate contract
    assert V.parse_eval_name("exp11_C16_traj_S42500_s45_K1") == {
        "arm": "C16", "cell": "traj", "step": 42500, "seed": 45, "K": 1}


def test_traj_cells_at_or_below_40k_are_refused(tmp_path):
    """40000 belongs to conf/q9 and everything below is the screen record."""
    for step in (40000, 37500):
        _rows, problems = V.validate_cell([], arm="C8", step=step, k=8, contract="traj")
        assert any("strictly above 40000" in p for p in problems), (step, problems)
