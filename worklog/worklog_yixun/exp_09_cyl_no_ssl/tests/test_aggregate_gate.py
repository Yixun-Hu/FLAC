"""exp-09 Stage B: named test-first round for ``aggregate_gate.py`` (plan §2/§4).

``eval_FLAC.py`` / ``compare_predictions.py`` write metric JSONs directly and never
exit nonzero on thresholds; ``aggregate_gate.py`` is the reviewed D-stage wrapper
that reads the per-gate verdicts, serialises ONE aggregate verdict atomically, and
exits nonzero on any failure. This file pins its contract, each case RED-first +
mutation-swept:

* incomplete-input rejection (a required gate has no verdict);
* non-finite rejection (a NaN/Inf anywhere in a verdict);
* atomic tmp+rename with NO partial verdict on an induced write failure;
* exit 0 on all-pass / nonzero on any fail  [mutation-sweep (e)].
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

# The exp-09 conftest already puts the runner dir (where aggregate_gate.py lives) on
# sys.path; belt-and-braces in case this file is collected standalone.
_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

import aggregate_gate as ag  # noqa: E402


# ------------------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------------------- #
def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
    return str(path)


def _verdict(gate, passed=True, **extra):
    d = {"gate": gate, "pass": passed}
    d.update(extra)
    return d


def _tmp_siblings(directory):
    return [n for n in os.listdir(directory) if n.startswith(ag.OUTPUT_TMP_PREFIX)]


# ------------------------------------------------------------------------------------- #
# happy path: all-pass -> exit 0, aggregate written atomically and finite
# ------------------------------------------------------------------------------------- #
def test_all_pass_exits_zero_and_writes_finite_aggregate(tmp_path):
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True, metrics={"T60": 0.05}))
    p2 = _write(tmp_path / "d2.json", _verdict("d2_cond", True, metrics={"rel": 1e-6}))
    out = tmp_path / "verdict.json"
    rc = ag.main(["--out", str(out), "--require", "d1_task,d2_cond", p1, p2])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["all_pass"] is True
    assert set(rec["gates"]) == {"d1_task", "d2_cond"}
    assert rec["failed_gates"] == []
    # the written aggregate is finite JSON (allow_nan default rejects NaN on read too)
    assert json.loads(out.read_text()) == rec


# ------------------------------------------------------------------------------------- #
# exit code: any fail -> nonzero  [mutation-sweep (e): inverting the exit flips both]
# ------------------------------------------------------------------------------------- #
def test_any_fail_exits_nonzero(tmp_path):
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True))
    p2 = _write(tmp_path / "d2.json", _verdict("d2_cond", False))
    out = tmp_path / "verdict.json"
    rc = ag.main(["--out", str(out), "--require", "d1_task,d2_cond", p1, p2])
    assert rc != 0
    rec = json.loads(out.read_text())
    assert rec["all_pass"] is False
    assert rec["failed_gates"] == ["d2_cond"]


def test_exit_code_is_zero_only_when_every_gate_passes(tmp_path):
    """Directly pins the mapping the exit-inversion mutant breaks: rc==0 iff all_pass."""
    for passed, expect_rc in ((True, 0), (False, 1)):
        p1 = _write(tmp_path / f"d1_{passed}.json", _verdict("d1_task", passed))
        out = tmp_path / f"v_{passed}.json"
        rc = ag.main(["--out", str(out), "--require", "d1_task", p1])
        assert (rc == 0) == passed, (passed, rc)
        assert rc == expect_rc


# ------------------------------------------------------------------------------------- #
# incomplete-input rejection
# ------------------------------------------------------------------------------------- #
def test_incomplete_input_is_rejected_and_writes_no_verdict(tmp_path):
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True))
    out = tmp_path / "verdict.json"
    # require two gates but only supply one
    rc = ag.main(["--out", str(out), "--require", "d1_task,d2_cond", p1])
    assert rc != 0
    assert not out.exists(), "a partial/aggregate verdict must not be written on incomplete input"
    assert _tmp_siblings(tmp_path) == [], "atomic-writer temp left behind"


def test_incomplete_input_raises_at_the_function_level(tmp_path):
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True))
    with pytest.raises(ag.GateError, match="d2_cond"):
        ag.aggregate([p1], required_gates=["d1_task", "d2_cond"])


def test_unexpected_extra_gate_is_rejected(tmp_path):
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True))
    p2 = _write(tmp_path / "x.json", _verdict("mystery_gate", True))
    with pytest.raises(ag.GateError, match="mystery_gate"):
        ag.aggregate([p1, p2], required_gates=["d1_task"])


def test_duplicate_gate_is_rejected(tmp_path):
    p1 = _write(tmp_path / "d1a.json", _verdict("d1_task", True))
    p2 = _write(tmp_path / "d1b.json", _verdict("d1_task", True))
    with pytest.raises(ag.GateError, match="[Dd]uplicate"):
        ag.aggregate([p1, p2], required_gates=["d1_task"])


# ------------------------------------------------------------------------------------- #
# non-finite rejection
# ------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_is_rejected(tmp_path, bad):
    # json.dump would emit NaN/Infinity tokens; write them so load sees a real non-finite
    p1 = _write(tmp_path / "d1.json", _verdict("d1_task", True))
    with open(tmp_path / "d2.json", "w") as f:
        json.dump(_verdict("d2_cond", True, metrics={"rel": bad}), f)  # allow_nan default True
    p2 = str(tmp_path / "d2.json")
    out = tmp_path / "verdict.json"
    rc = ag.main(["--out", str(out), "--require", "d1_task,d2_cond", p1, p2])
    assert rc != 0
    assert not out.exists(), "no verdict may be written when an input is non-finite"


def test_non_finite_raises_at_the_function_level(tmp_path):
    with open(tmp_path / "d.json", "w") as f:
        json.dump(_verdict("d1_task", True, metrics={"rel": float("nan")}), f)
    with pytest.raises(ag.GateError, match="finite"):
        ag.load_verdict(str(tmp_path / "d.json"))


# ------------------------------------------------------------------------------------- #
# atomicity: tmp+rename, no partial verdict on induced failure
# ------------------------------------------------------------------------------------- #
def test_write_failure_leaves_no_partial_and_no_tmp(tmp_path):
    """A record carrying a non-finite value (bypassing the input pre-check) must make
    the atomic write raise via allow_nan=False, leave the destination untouched, and
    remove its temp file — no partial verdict."""
    out = tmp_path / "verdict.json"
    with pytest.raises(ValueError):
        ag.write_verdict(str(out), {"all_pass": True, "poison": float("inf")})
    assert not out.exists()
    assert _tmp_siblings(tmp_path) == []


def test_induced_replace_failure_leaves_destination_and_removes_tmp(tmp_path, monkeypatch):
    out = tmp_path / "verdict.json"
    _write(out, {"pre_existing": True})  # a valid prior verdict must survive a failed rewrite
    before = out.read_text()

    def boom(src, dst):
        raise OSError("induced os.replace failure")

    monkeypatch.setattr(ag.os, "replace", boom)
    with pytest.raises(OSError, match="induced"):
        ag.write_verdict(str(out), {"all_pass": True})
    assert out.read_text() == before, "destination was clobbered on a failed rename"
    assert _tmp_siblings(tmp_path) == [], "temp file left behind after a failed rename"


# ------------------------------------------------------------------------------------- #
# schema validation
# ------------------------------------------------------------------------------------- #
def test_missing_pass_field_is_rejected(tmp_path):
    p = _write(tmp_path / "d.json", {"gate": "d1_task", "metrics": {"x": 1.0}})
    with pytest.raises(ag.GateError, match="pass"):
        ag.load_verdict(p)


def test_non_bool_pass_field_is_rejected(tmp_path):
    p = _write(tmp_path / "d.json", {"gate": "d1_task", "pass": "yes"})
    with pytest.raises(ag.GateError, match="pass"):
        ag.load_verdict(p)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ag.GateError):
        ag.load_verdict(str(tmp_path / "does_not_exist.json"))
