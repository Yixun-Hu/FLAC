"""exp_35 round 1c: the evaluation runner for exp_35 cells (plan_orbitrir_yaw_evidence.md r2 §3b / §11 B1,
r3 §12 B1.1 / H4 (c)) -- run as a subprocess against the throwaway clone of test_depth_canon_eval_run.py in
harness mode (stub evaluator, stub nvidia-smi / findmnt / df, fake checkpoints; nothing real is evaluated).

An exp_35 cell (plan entry ``no_flash``) is spawned through the committed no-flash launcher with
``PYTHONPATH=<repo>/overlays/no_flash`` and ``PYTHONDONTWRITEBYTECODE=1``; the launcher's CPU pre-flight runs
before any spawn; after rc 0 the cell log must carry the backend sentinels or the campaign hard-stops with
no receipt; the receipt is schema 2. With ``EXP35_APPROVED_COTENANTS`` the foreign set of the lane must equal
the approved identity tuple at the lane selection AND at the final pre-spawn check. exp_25 cells: unchanged.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import json
import os
import subprocess
import sys

import pytest

from test_depth_canon_eval_run import (C, UUID, _apps, _calls, _entry, _git, _ran, _verify, clone, drive,  # noqa: F401
                                       harness, sleepers)

CTL1, CTL8 = "exp35_P1_T_S40000_K1_s42", "exp35_P1_T_S40000_K8_s42"
BF_R = "exp35_BF_R_S40000_K1_s43"
EXP25 = "exp25_P1_T_S40000_K1_s42"
IDENTITY = os.path.join("worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude", "exp35_gpu_identity.py")


def _receipt(clone, harness, cell):
    return json.load(open(_entry(clone, harness, cell)["receipt"]))


def _clog(harness, cell):
    logs = harness["dir"] / "logs"
    hits = sorted(p for p in logs.iterdir() if p.name.endswith(f"_eval_{cell}.log"))
    assert hits, list(logs.iterdir())
    return hits[-1].read_text()


def test_an_exp35_cell_runs_through_the_launcher_and_gets_receipt_2(clone, harness):
    rc, out = drive(clone, harness, "--only", f"{CTL1},{BF_R}", "--lanes", "0",
                    env={"PYTHONPATH": "/tmp/an-inherited-entry"})
    assert rc == 0 and "CAMPAIGN SELECTION COMPLETE" in out, out
    assert "no-flash pre-flight OK" in out and "no-flash overlay + launcher OK" in out
    overlay = str(clone / "overlays" / "no_flash")
    for call in _calls(harness):
        assert call["no_flash"] is True and call["dont_write"] == "1", call       # launched, SDPA path asserted
        assert call["pythonpath"] == overlay and call["path"] == [overlay, str(clone)]   # §12 H4 (c); no inherit
    for cell in (CTL1, BF_R):
        rec = _receipt(clone, harness, cell)
        assert rec["schema_version"] == 2 and rec["attention_backend"] == "sdpa", cell
        assert (rec["no_flash_overlay_sha256"], rec["no_flash_launcher_sha256"]) == (
            C.NO_FLASH["overlay_sha256"], C.NO_FLASH["launcher_sha256"])
        assert C.backend_log_reasons(_clog(harness, cell), target=str(harness["dir"] / "stub_eval.py")) == []
        assert _verify(clone, harness, cell, backend_target=str(harness["dir"] / "stub_eval.py")) == []
    rc, out = drive(clone, harness, "--only", f"{CTL1},{BF_R}", "--lanes", "0")
    assert rc == 0 and out.count("SKIP (complete and verified)") == 2 and len(_calls(harness)) == 2, out


def test_an_exp25_cell_is_spawned_exactly_as_before(clone, harness):
    rc, out = drive(clone, harness, "--only", EXP25, env={"PYTHONPATH": "/tmp/an-inherited-entry"})
    assert rc == 0, out
    (call,) = _calls(harness)
    assert call["no_flash"] is False and call["pythonpath"] == "/tmp/an-inherited-entry"   # the old spawn
    rec = _receipt(clone, harness, EXP25)
    assert rec["schema_version"] == 1 and list(rec) == list(C.RECEIPT_KEYS)
    assert "no-flash pre-flight" not in out                                    # no exp_35 cell: no pre-flight


@pytest.mark.parametrize("mode", ["flash_missing"])
def test_a_cell_log_without_the_backend_evidence_hard_stops_the_campaign(clone, harness, mode):
    rc, out = drive(clone, harness, "--only", f"{CTL1},{CTL8}", "--lanes", "0", env={"STUB_MODE": mode})
    assert rc == 2 and "BACKEND" in out and "ABORTED" in out, out
    assert len(_calls(harness)) == 1                                            # the second cell never spawned
    assert not os.path.exists(_entry(clone, harness, CTL1)["receipt"])
    assert not os.path.exists(_entry(clone, harness, CTL8)["receipt"])


def test_dry_run_runs_the_preflight_and_plans_without_spawning(clone, harness):
    rc, out = drive(clone, harness, "--only", f"{CTL1},{BF_R}", env={"DRY_RUN": "1"})
    assert rc == 0 and "no-flash pre-flight OK" in out and out.count("would run") == 2, out
    assert "spawned through overlays/no_flash/exp35_no_flash_launch.py" in out
    assert not (harness["dir"] / "calls.log").exists() and not (harness["dir"] / "nvsmi.log").exists()
    pre = sorted((harness["dir"] / "logs").glob("*_no_flash_preflight.log"))
    assert pre and C.backend_log_reasons(pre[-1].read_text(), target="--preflight") == []


def test_ignored_bytecode_under_the_overlay_fails_the_preflight(clone, harness):
    """Codex r1 R2.2: an ignored .pyc never enters the tree digest -- the launcher refuses it."""
    cache = clone / "overlays" / "no_flash" / "__pycache__"
    cache.mkdir()
    try:
        (cache / "flash_attn.cpython-310.pyc").write_bytes(b"\x00")
        assert not _git(clone, "status", "--porcelain", "--untracked-files=all", "--", "overlays")   # ignored
        rc, out = drive(clone, harness, "--only", CTL1, env={"DRY_RUN": "1"})
        assert rc == 2 and "NO-FLASH PRE-FLIGHT FAILED" in out and not _calls(harness), out
    finally:
        (cache / "flash_attn.cpython-310.pyc").unlink()
        cache.rmdir()


def test_an_edited_overlay_or_launcher_is_refused(clone, harness):
    path = clone / "overlays" / "no_flash" / "flash_attn.py"
    original = path.read_bytes()
    head = _git(clone, "rev-parse", "HEAD")
    try:
        path.write_bytes(original + b"# edit\n")
        rc, out = drive(clone, harness, "--only", CTL1, env={"DRY_RUN": "1"})
        assert rc == 2 and "NOT COMMITTED" in out and "overlays/no_flash" in out, out      # uncommitted
        _git(clone, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-am", "edit overlay")
        rc, out = drive(clone, harness, "--only", CTL1, env={"DRY_RUN": "1",
                                                              "EXPECT_HEAD": _git(clone, "rev-parse", "HEAD")})
        assert rc == 2 and "NO-FLASH PIN FAILED" in out, out                                # committed, unpinned
    finally:
        _git(clone, "reset", "-q", "--hard", head)


def test_an_edit_of_the_launcher_between_two_cells_is_pin_drift(clone, harness):
    launcher = clone / "overlays" / "no_flash" / "exp35_no_flash_launch.py"
    original = launcher.read_bytes()
    try:
        rc, out = drive(clone, harness, "--only", f"{CTL1},{CTL8}", "--lanes", "0",
                        env={"GUARD_MUTATE_AT": "3", "GUARD_MUTATE_FILE": str(launcher)})
    finally:
        launcher.write_bytes(original)
    assert rc == 4 and "PIN DRIFT" in out and len(_ran(harness)) == 1, out


# ---- §12 B1.1: the approved identity tuple, enforced by the runner too ---- #
def _tuple(clone, tmp_path, pids, gpu=1, edit=None):
    p = subprocess.run([sys.executable, str(clone / IDENTITY), "--snapshot", "--gpu-uuid", UUID[gpu]]
                       + [x for pid in pids for x in ("--pid", str(pid))], capture_output=True, text=True,
                       check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    doc = json.loads(p.stdout)
    if edit:
        edit(doc)
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(doc))
    return str(path)


def test_sharing_an_exp35_cell_needs_the_identity_tuple(clone, harness, sleepers, tmp_path):
    _apps(harness, [(sleepers(), 1)])
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1", env={"ALLOW_SHARED_GPU": "1"})
    assert rc == 2 and "EXP35_APPROVED_COTENANTS" in out and not _calls(harness), out
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"EXP35_APPROVED_COTENANTS": _tuple(clone, tmp_path, [os.getpid()])})
    assert rc == 2 and "ALLOW_SHARED_GPU=1" in out and not _calls(harness), out


def test_the_approved_cotenant_shares_the_lane(clone, harness, sleepers, tmp_path):
    pid = sleepers()
    _apps(harness, [(pid, 1)])
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", "EXP35_APPROVED_COTENANTS": _tuple(clone, tmp_path, [pid])})
    assert rc == 0 and "identity tuple" in out and [c["cvd"] for c in _calls(harness)] == ["1"], out


@pytest.mark.parametrize("case", ["extra", "missing", "replaced", "pid_reuse", "other_gpu"])
def test_any_other_foreign_set_refuses_the_cell(clone, harness, sleepers, tmp_path, case):
    pid = sleepers()
    approved = _tuple(clone, tmp_path, [pid], gpu=0 if case == "other_gpu" else 1,
                      edit=(lambda d: d["processes"][0].update(start_time=d["processes"][0]["start_time"] + 1))
                      if case == "pid_reuse" else None)
    rows = {"extra": [(pid, 1), (sleepers(), 1)], "missing": [], "replaced": [(sleepers(), 1)]}.get(case, [(pid, 1)])
    _apps(harness, rows)
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", "EXP35_APPROVED_COTENANTS": approved})
    assert rc == 2 and "GPU GATE FAILED" in out and "identity" in out and not _calls(harness), (case, out)


def test_a_change_at_the_final_pre_spawn_check_refuses_the_cell(clone, harness, sleepers, tmp_path):
    """The tuple holds at the lane selection, a newcomer joins before the spawn: no evaluator runs."""
    pid = sleepers()
    _apps(harness, [(pid, 1)], name="apps.1")                                 # the selection's query
    _apps(harness, [(pid, 1), (sleepers(), 1)])                               # every later query
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", "EXP35_APPROVED_COTENANTS": _tuple(clone, tmp_path, [pid])})
    assert rc == 2 and "identity" in out and not _calls(harness), out
    assert not os.path.exists(_entry(clone, harness, CTL1)["receipt"])
