"""exp_35 round 6a (iv): the supervisor's real world, its collector gates, the dry run and the CLI (plan r2 §11 R8:
"a supervisor dry run that plans the 33 chunks and exercises the identity check read-only against the live GPU state
without spawning anything"; r4 §13 NIT 3: a read-only helper printing the sha256 of his queue script, evaluator and
admission helper). No real nvidia-smi and no real collector run here (both are replaced); one trivial child process.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from test_exp35_supervisor_lanes import G0, G1, S, make


def _snapshot(root):
    return {os.path.join(d, f): os.lstat(os.path.join(d, f)).st_mtime_ns for d, _, fs in os.walk(root) for f in fs}


# --------------------------------------------------------------------------- #
# the real world
# --------------------------------------------------------------------------- #
def test_the_real_world_reads_nvidia_smi_fail_closed(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=f"0, {G0}\n1, {G1}\n", stderr="")
    monkeypatch.setattr(S.subprocess, "run", run)
    w = S.World()
    assert S.gpu_map(w.nvsmi("map")) == {0: G0, 1: G1}
    assert calls[-1] == ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
    w.nvsmi("apps")
    assert calls[-1] == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]
    monkeypatch.setattr(S.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 9, "", "NVML error"))
    with pytest.raises(ValueError, match="rc 9"):
        w.nvsmi("apps")

    def missing(argv, **kw):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(S.subprocess, "run", missing)
    with pytest.raises(ValueError, match="nvidia-smi"):
        w.nvsmi("map")


def test_the_real_world_spawns_into_the_chunk_log(tmp_path):
    log = tmp_path / "chunk.log"
    p = S.World().spawn([sys.executable, "-c", "import os, sys; print('hello', os.getcwd()); sys.exit(3)"],
                        dict(os.environ), str(log))
    assert p.wait(timeout=60) == 3 and log.read_text().strip() == f"hello {S.REPO}"


# --------------------------------------------------------------------------- #
# the collector gates: the collector's own CLIs at the pin
# --------------------------------------------------------------------------- #
def _gates(tmp_path, rc=0, out="ok"):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")
    cfg = {"h35": "a" * 40, "repo": str(tmp_path), "cells": os.path.join(S.REPO, S.CELLS)}
    return S.CollectorGates(cfg, run=run), calls


def test_the_gates_call_the_collector_clis_at_the_pin(tmp_path):
    g, calls = _gates(tmp_path)
    assert g.e1(8) == (0, "ok") and calls[-1][0] == [S.PY, S.COLLECT, "--exp35-ctl-check", "8", "--pin", "a" * 40]
    kw = calls[-1][1]
    assert kw["cwd"] == str(tmp_path) and kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and "ALLOW_SHARED_GPU" not in kw["env"]
    g.e2()
    assert calls[-1][0] == [S.PY, S.COLLECT, "--exp35-e2", "--pin", "a" * 40]
    g.finalize("worklog/x/orbitrir_bf_cond_probe_t.log", "d" * 64)
    assert calls[-1][0] == [S.PY, S.COLLECT, "--finalize-bf-probe", S.PROBE_JSON, "--probe-log",
                            "worklog/x/orbitrir_bf_cond_probe_t.log", "--campaign-digest", "d" * 64, "--pin", "a" * 40]


def _receipt_path(g, cell):
    C = S.collector_module()
    m = C.load_cells(g.cfg["cells"])
    row = next(r for r in m["cells"] if r["cell"] == cell)
    target, _ = C.expected_checkpoint(row, m, None)
    return C.cell_paths(row, os.path.join(m["farm_root"], row["arm"], os.path.basename(target)), m["dump_root"])


@pytest.mark.parametrize("case", ["valid", "invalid", "no_receipt"])
def test_verify_reads_the_receipt_and_runs_verify_cell_under_its_digest(tmp_path, case):
    g, calls = _gates(tmp_path, rc=0 if case == "valid" else 1, out="VALID x" if case == "valid" else "INVALID x")
    cell = "exp35_BF_T_S40000_K1_s42"
    p = _receipt_path(g, cell)
    if case != "no_receipt":
        os.makedirs(os.path.dirname(os.path.join(str(tmp_path), p.receipt)), exist_ok=True)
        with open(os.path.join(str(tmp_path), p.receipt), "w") as f:
            json.dump({"cell": cell, "campaign_digest": "d" * 64}, f)
    ok, info = g.verify({"cell": cell})
    if case == "valid":
        raw = open(os.path.join(str(tmp_path), p.receipt), "rb").read()
        assert ok and info == {"receipt_sha256": hashlib.sha256(raw).hexdigest(), "campaign_digest": "d" * 64}
        assert calls[-1][0] == [S.PY, S.COLLECT, "--verify-cell", p.stem, "--pin", "a" * 40, "--campaign-digest", "d" * 64]
    else:
        assert not ok and info["reasons"] and (calls == [] if case == "no_receipt" else len(calls) == 1)


# --------------------------------------------------------------------------- #
# the dry run: read-only against the live state
# --------------------------------------------------------------------------- #
def test_the_dry_run_plans_the_33_chunks_and_checks_identity_read_only(tmp_path, capsys):
    w, _, cfg = make(tmp_path)
    before = _snapshot(tmp_path)
    assert S.dry_run(cfg, w) == 0
    out = capsys.readouterr().out
    assert sum(line.startswith("  chunk ") for line in out.splitlines()) == 33
    for needle in ("--only exp35_P1_T_S40000_K1_s42", "--only exp35_BF_C4_S40000_K8_s42_th45", S.PROBE,
                   "identity OK", "guard OK", "predictor", "reservations OK", "DRY RUN OK"):
        assert needle in out, needle
    assert w.spawned == [] and _snapshot(tmp_path) == before and not os.path.exists(cfg["state"])


def test_the_dry_run_reports_a_mismatch_and_still_writes_nothing(tmp_path, capsys):
    w, _, cfg = make(tmp_path)
    w.apps.append((99999, G1))
    before = _snapshot(tmp_path)
    assert S.dry_run(cfg, w) == 1
    out = capsys.readouterr().out
    assert "IDENTITY MISMATCH" in out and "99999" in out and "DRY RUN REFUSED" in out
    assert w.spawned == [] and _snapshot(tmp_path) == before


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_external_shas_helper(tmp_path, monkeypatch, capsys):
    files = {k: tmp_path / f"{k}.txt" for k in ("queue", "evaluator", "admission")}
    for k, p in files.items():
        p.write_text(k)
    monkeypatch.setattr(S, "EXTERNAL", {k: str(p) for k, p in files.items()})
    assert S.main(["--external-shas"]) == 0
    assert json.loads(capsys.readouterr().out) == {k: {"path": str(p), "sha256": hashlib.sha256(k.encode()).hexdigest()}
                                                   for k, p in files.items()}
    files["evaluator"].unlink()
    assert S.main(["--external-shas"]) == 1
    assert set(S.EXTERNAL) == {"queue", "evaluator", "admission"}


@pytest.mark.parametrize("argv", [["--h35", "abc", "--approved", "x", "--approved-sha256", "0" * 64],
                                  ["--h35", "a" * 40, "--approved", "x", "--approved-sha256", "nope"],
                                  ["--approved", "x", "--approved-sha256", "0" * 64]])
def test_the_cli_refuses_malformed_pins(argv, capsys):
    assert S.main(argv + ["--dry-run"]) == 2 and "SUPERVISOR REFUSED" in capsys.readouterr().out


def test_the_cli_dry_run_creates_nothing(tmp_path, monkeypatch, capsys):
    w, _, cfg = make(tmp_path)
    monkeypatch.setattr(S, "World", lambda: w)
    monkeypatch.setattr(S, "PROC", cfg["proc"])
    res = tmp_path / "no_reservations_dir_yet"
    argv = ["--h35", cfg["h35"], "--approved", cfg["approved"], "--approved-sha256", cfg["approved_sha256"],
            "--state", cfg["state"], "--reservations", str(res), "--his-log", cfg["his_log"], "--logdir", cfg["logdir"]]
    assert S.main(argv + ["--dry-run"]) == 0 and not res.exists() and not os.path.exists(cfg["state"])
    assert "DRY RUN OK" in capsys.readouterr().out and w.spawned == []


def test_the_cli_run_refuses_a_head_other_than_h35(tmp_path, monkeypatch, capsys):
    w, _, cfg = make(tmp_path)
    monkeypatch.setattr(S, "World", lambda: w)
    argv = ["--h35", "b" * 40, "--approved", cfg["approved"], "--approved-sha256", cfg["approved_sha256"],
            "--state", cfg["state"], "--reservations", str(tmp_path / "res"), "--his-log", cfg["his_log"]]
    assert S.main(argv) == 3 and "HEAD" in capsys.readouterr().out
    assert w.spawned == [] and not os.path.exists(cfg["state"]) and not (tmp_path / "res").exists()
