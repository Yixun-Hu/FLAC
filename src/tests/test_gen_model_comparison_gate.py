"""Tests for the model-comparison generator's exp_11 validation gate (review B5/B7).

Round-4 review: the validator was advisory — the table generator globbed raw
JSONs and aggregated whatever it found, so an unvalidated (or unprovable) exp_11
row could reach `model_comparison.md`. It is now a gate: an exp_11 row whose cell
fails `exp11_validate_rows` is refused and rendered as blocked, never as numbers.

The second half of the review's concern is disclosure (B7): rows evaluated under
the legacy per-angle loop and rows evaluated under the batched orbit are not
interchangeable, so the table must say which each row is instead of labelling
everything "fa eval".
"""
import importlib.util
import json
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_GEN_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "gen_model_comparison.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_model_comparison", _GEN_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_GEN_PY}"
    mod = importlib.util.module_from_spec(spec)
    # the module writes the table at import time only under __main__
    spec.loader.exec_module(mod)
    return mod


G = _load_module()


def _row_spec(label, proto, k, pats, **extra):
    return (label, proto, k, pats, extra) if extra else (label, proto, k, pats)


# --------------------------------------------------------------------------- #
# 1. exp_11 rows are recognised and carry their execution label
# --------------------------------------------------------------------------- #
def test_importing_the_generator_does_not_rewrite_the_table():
    """Re-review item 6: at module scope the write ran on IMPORT, so merely
    running pytest could regenerate the frozen table."""
    table = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "model_comparison.md")
    before = open(table, "rb").read()
    _load_module()                      # a second import, deliberately
    assert open(table, "rb").read() == before, "importing the generator rewrote the table"
    assert hasattr(G, "main"), "table generation must live behind main()"


def test_exp11_rows_are_detected_from_their_patterns():
    assert G.is_exp11_row(["outputs_FLAC/exp11_C8/**/*exp11_C8_conf_S40000*.json"])
    assert not G.is_exp11_row(["outputs_FLAC/exp07_BF/**/*exp10_BF40_K8*.json"])
    assert not G.is_exp11_row(["weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json"])


def test_every_registered_row_declares_its_orbit_execution():
    """B7: 'fa eval' alone is not a protocol. Every fa row must say whether it was
    produced by the legacy per-angle loop or the batched orbit."""
    for row in G.ROWS:
        proto = row[1]
        if "fa" in proto:
            assert ("legacy-loop" in proto) or ("batched" in proto), (
                f"row {row[0]!r} says {proto!r} — it must disclose loop vs batched")


def test_historical_fa_rows_are_labelled_legacy_loop():
    labels = {row[0]: row[1] for row in G.ROWS}
    for label, proto in labels.items():
        if "fa" in proto and not G.is_exp11_row(row_pats(label)):
            assert "legacy-loop" in proto, f"historical row {label!r} is not labelled legacy-loop"


def row_pats(label):
    for row in G.ROWS:
        if row[0] == label:
            return row[3]
    return []


# --------------------------------------------------------------------------- #
# 2. the gate itself
# --------------------------------------------------------------------------- #
def _write_valid_cell(tmp_path, arm="C8", step=40000, k=8, seeds=(42, 43, 44, 45, 46)):
    """Five conf rows that pass exp11_validate_rows (sidecars included)."""
    spec = importlib.util.spec_from_file_location(
        "v", os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                          "exp11_validate_rows.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    n_ang = V.ARM_ORBITS[arm]
    angles = V.orbit_for(arm)
    ck = (f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/"
          f"epoch=8-step={step}.ckpt")
    metrics = {"T60": 12.0, "C50": 1.0, "EDT": 4.0, "RIR_to_GT_RIR_R@1": 0.5,
               "RIR_to_GT_RIR_R@5": 0.7, "RIR_to_GT_RIR_R@10": 0.9}
    paths = []
    for seed in seeds:
        ev = f"exp11_{arm}_conf_S{step}_s{seed}_K{k}"
        name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}.json"
        rec = {"metrics": metrics, "ckpt_path": ck, "rotate_deg": 0.0,
               "cond_method": "fa_invariant", "frame_avg_angles": angles,
               "cond_autocast": "bf16", "orbit_execution": "batched",
               "frame_avg_fwd_cap": 64, "source_sha": "a" * 40, "batch_size": 64,
               "n_samples": 6337,
               "dataset_config": V.EVAL_CONFIG_FOR_K[k], "seed": seed, "cfg_scale": 1.0,
               "steps": 1, "eval_name": ev, "weights_source": "ema", "device": "cuda"}
        side = {"arm": arm, "step": step, "seed": seed, "K": k, "eval_name": ev,
                "cfg_scale": 1.0, "steps": 1,
                "model_config": f"worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_{arm}.json",
                "model_config_sha256": "b" * 64, "dataset_config": V.EVAL_CONFIG_FOR_K[k],
                "ckpt_path": ck, "ckpt_sha256": "c" * 64, "use_ema": True,
                "frame_avg_angles": angles, "cond_method": "fa_invariant",
                "cond_autocast": "bf16", "commit": "d" * 40}
        p = tmp_path / name
        p.write_text(json.dumps(rec))
        (tmp_path / (name + ".screenmeta.json")).write_text(json.dumps(side))
        paths.append(str(p))
    return paths


def test_gate_passes_a_validated_exp11_cell(tmp_path):
    paths = _write_valid_cell(tmp_path)
    ok, problems = G.validate_exp11_cell(paths)
    assert ok, problems


def test_gate_refuses_a_cell_with_a_missing_sidecar(tmp_path):
    paths = _write_valid_cell(tmp_path)
    os.remove(paths[0] + ".screenmeta.json")
    ok, problems = G.validate_exp11_cell(paths)
    assert not ok and any("sidecar" in p for p in problems)


def test_gate_refuses_a_legacy_loop_row_in_an_exp11_cell(tmp_path):
    paths = _write_valid_cell(tmp_path)
    rec = json.load(open(paths[0]))
    rec["orbit_execution"] = "loop"
    open(paths[0], "w").write(json.dumps(rec))
    ok, problems = G.validate_exp11_cell(paths)
    assert not ok and any("orbit_execution" in p for p in problems)


def test_gate_refuses_a_four_seed_table_cell(tmp_path):
    paths = _write_valid_cell(tmp_path, seeds=(42, 43, 44, 45))
    ok, problems = G.validate_exp11_cell(paths)
    assert not ok and any("46" in p for p in problems)


def test_gate_refuses_a_screen_cell_as_a_table_row(tmp_path):
    """A single-seed futility screen must never become a table row."""
    paths = _write_valid_cell(tmp_path, seeds=(42,))
    ok, problems = G.validate_exp11_cell(paths)
    assert not ok


def test_gate_reports_empty_cells_as_not_blocking():
    """No files yet is 'pending', which the generator already renders — the gate
    must not turn an empty cell into a failure."""
    ok, problems = G.validate_exp11_cell([])
    assert ok and problems == []


def test_render_blocks_an_invalid_exp11_row(tmp_path):
    paths = _write_valid_cell(tmp_path)
    rec = json.load(open(paths[0]))
    rec["n_samples"] = 64                        # a partial-split row
    open(paths[0], "w").write(json.dumps(rec))
    line, blocked = G.render_row("C8 @40k", "fa eval (batched)", 8, paths)
    assert blocked
    assert "BLOCKED" in line and "12.000" not in line, line


def test_render_emits_numbers_for_a_valid_row(tmp_path):
    paths = _write_valid_cell(tmp_path)
    line, blocked = G.render_row("C8 @40k", "fa eval (batched)", 8, paths)
    assert not blocked and "12.000" in line
