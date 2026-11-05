"""exp_35 round 5a (i): the exp_25 stream registry and its helper ``exp35_make_stream_registry.py``
(plan_orbitrir_yaw_evidence.md r2 §5 item 5): the input / assignment hashes every exp_35 cell must reproduce,
frozen from exp_25's VERIFIED P1-pin, CANON and YAW-pin unseen 40k cells, which must agree across arms.

The source cells are the shared fixture's synthetic exp_25 cells (N = 8 items, real evaluator functions).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import copy
import importlib.util
import json
import math
import os

import pytest

import eval_FLAC
import test_depth_canon_collect as T

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
HELPER = os.path.join(T.REPO, "worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude",
                      "exp35_make_stream_registry.py")


def _load():
    spec = importlib.util.spec_from_file_location("exp35_make_stream_registry", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()
SOURCE = ["T", "C4+45", "R", "P1-pin", "YAW-pin"]


def _build(tmp, **kw):
    return R.build_registry(C, MANIFEST, base_dir=str(tmp), pin=PIN, canon_index=T.canon_index(),
                            farm_root=T.FARM, dump_root=os.path.join(str(tmp), "dumps"), **kw)


def _campaign(tmp, skip=()):
    T.write_campaign(tmp, SOURCE, skip=skip)


def _stream(tmp, cell):
    r = next(c for c in MANIFEST["cells"] if c["cell"] == cell)
    p = C.cell_paths(r, os.path.join(T.FARM, r["arm"], T.ckpt_base(r) + ".ckpt"), "d")
    return json.load(open(os.path.join(str(tmp), p.stream)))


def test_the_registry_holds_every_stream_hash_of_the_verified_cells(tmp_path):
    _campaign(tmp_path)
    reg = _build(tmp_path)
    assert list(reg) == ["schema_version", "source", "input_hash", "assignment_hash_random", "assignment_hash_fixed",
                         "derived"]
    for k in (1, 8):
        for s in (42, 43, 44, 45, 46):
            want = _stream(tmp_path, f"exp25_P1_T_S40000_K{k}_s{s}")["input_hash"]
            assert reg["input_hash"][str(k)][str(s)] == want == _stream(tmp_path, f"exp25_YAW_R_S40000_K{k}_s{s}")["input_hash"]
    for s in (42, 43, 44, 45, 46):
        assert reg["assignment_hash_random"][str(s)] == _stream(tmp_path, f"exp25_CANON_R_S40000_K1_s{s}")["assignment_hash"]
    assert reg["assignment_hash_fixed"]["0"] == _stream(tmp_path, "exp25_P1_T_S40000_K8_s44")["assignment_hash"]
    assert reg["assignment_hash_fixed"]["45"] == _stream(tmp_path, "exp25_CANON_C4_S40000_K1_s42_th45")["assignment_hash"]
    assert set(reg["assignment_hash_fixed"]) == {"0", "90", "180", "270", "45", "135", "225"}
    assert reg["source"]["pin"] == PIN and len(reg["source"]["cells"]) == 84          # 3 arms x 28 cells


def test_the_derived_angles_are_what_the_evaluator_would_record(tmp_path):
    """135 / 225 have no exp_25 cell: derived from the split's target order + yaw_column_shift -- the same
    tuples eval_FLAC's fixed-mode stream records (and the derivation reproduces every observed angle)."""
    _campaign(tmp_path)
    reg = _build(tmp_path)
    for deg in (135, 225):
        row = dict(next(c for c in MANIFEST["cells"] if c["cell"] == "exp25_P1_T_S40000_K1_s42"), rotate_deg=deg)
        assert reg["assignment_hash_fixed"][str(deg)] == T.build_stream(row).assignment_hash()
        assert eval_FLAC.yaw_column_shift(math.radians(deg), 512) == {135: 192, 225: 320}[deg]
    assert reg["derived"]["angles"] == [135, 225]


def test_cells_that_disagree_across_arms_refuse_the_registry(tmp_path):
    odd = next(c for c in MANIFEST["cells"] if c["cell"] == "exp25_YAW_R_S40000_K8_s45")
    _campaign(tmp_path, skip={odd["cell"]})
    T.write_cell(tmp_path, odd, repeats=(1, 2, 3))                          # another context draw
    with pytest.raises(ValueError, match="disagree"):
        _build(tmp_path)


def test_an_unverified_or_missing_source_cell_refuses_the_registry(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _campaign(a, skip={"exp25_P1_C4_S40000_K8_s42_th270"})
    with pytest.raises(ValueError, match="exp25_P1_C4_S40000_K8_s42_th270"):
        _build(a)
    _campaign(b)
    T.edit_json(os.path.join(str(b), C.cell_paths(
        next(c for c in MANIFEST["cells"] if c["cell"] == "exp25_P1_R_S40000_K1_s43"),
        os.path.join(T.FARM, "P1", "epoch=8-step=40000.ckpt"), "d").receipt), lambda d: d.update(rc=1))
    with pytest.raises(ValueError, match="does not verify"):
        _build(b)


def test_a_wrong_derivation_rule_is_caught_by_the_observed_angles(tmp_path, monkeypatch):
    _campaign(tmp_path)
    monkeypatch.setattr(R, "shift_of", lambda deg: 7)
    with pytest.raises(ValueError, match="does not reproduce"):
        _build(tmp_path)


def test_cli_writes_and_checks_the_registry(tmp_path, capsys):
    _campaign(tmp_path)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    out = tmp_path / "reg.json"
    argv = ["--cells", str(cells), "--base-dir", str(tmp_path), "--pin", PIN, "--farm-root", T.FARM,
            "--dump-root", os.path.join(str(tmp_path), "dumps"), "--canon-ckpt-manifest", T._canon_manifest_file(tmp_path),
            "--out", str(out)]
    assert R.main(argv + ["--write"]) == 0 and json.loads(out.read_text()) == _build(tmp_path)
    assert out.read_text() == json.dumps(_build(tmp_path), indent=1) + "\n"
    assert R.main(argv + ["--check"]) == 0
    out.write_text(out.read_text().replace('"schema_version": 1', '"schema_version": 2'))
    assert R.main(argv + ["--check"]) == 1 and "DRIFTED" in capsys.readouterr().out
