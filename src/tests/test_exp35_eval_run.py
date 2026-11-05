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
    import hashlib                                     # Codex C2 BLOCKING 1: the launch entry pins its bytes
    return {"EXP35_APPROVED_COTENANTS": str(path),
            "EXP35_APPROVED_COTENANTS_SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_sharing_an_exp35_cell_needs_the_identity_tuple(clone, harness, sleepers, tmp_path):
    _apps(harness, [(sleepers(), 1)])
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1", env={"ALLOW_SHARED_GPU": "1"})
    assert rc == 2 and "EXP35_APPROVED_COTENANTS" in out and not _calls(harness), out
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env=_tuple(clone, tmp_path, [os.getpid()]))
    assert rc == 2 and "ALLOW_SHARED_GPU=1" in out and not _calls(harness), out


def test_the_approved_cotenant_shares_the_lane(clone, harness, sleepers, tmp_path):
    pid = sleepers()
    _apps(harness, [(pid, 1)])
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", **_tuple(clone, tmp_path, [pid])})
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
                    env={"ALLOW_SHARED_GPU": "1", **approved})
    assert rc == 2 and "GPU GATE FAILED" in out and "identity" in out and not _calls(harness), (case, out)


def test_a_change_at_the_final_pre_spawn_check_refuses_the_cell(clone, harness, sleepers, tmp_path):
    """The tuple holds at the lane selection, a newcomer joins before the spawn: no evaluator runs."""
    pid = sleepers()
    _apps(harness, [(pid, 1)], name="apps.1")                                 # the selection's query
    _apps(harness, [(pid, 1), (sleepers(), 1)])                               # every later query
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", **_tuple(clone, tmp_path, [pid])})
    assert rc == 2 and "identity" in out and not _calls(harness), out
    assert not os.path.exists(_entry(clone, harness, CTL1)["receipt"])


# ---- Codex C2 HIGH 2: an exp_25-only selection never depends on the exp_35 files ---- #
def test_an_exp25_only_selection_runs_without_any_exp35_file(clone, harness):
    from test_exp35_collect import _collector_at
    head = _git(clone, "rev-parse", "HEAD")
    try:
        _git(clone, "rm", "-q", "-r", "overlays", IDENTITY)
        _git(clone, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-m", "no exp_35 files")
        env = {"EXPECT_HEAD": _git(clone, "rev-parse", "HEAD")}
        rc, out = drive(clone, harness, "--only", EXP25, env=dict(env, DRY_RUN="1"))
        assert rc == 0 and "no-flash" not in out, out
        rc, out = drive(clone, harness, "--only", EXP25, env=env)
        assert rc == 0 and "CAMPAIGN SELECTION COMPLETE" in out, out
        (call,) = _calls(harness)
        assert call["no_flash"] is False and _receipt(clone, harness, EXP25)["schema_version"] == 1
        plan = json.load(open(sorted((harness["dir"] / "logs").glob("*_eval_plan.json"))[-1]))
        old = _collector_at("682df5db")
        m = harness["manifest"]
        index = old.load_canon_index(str(harness["dir"] / "canon_ckpts.sha256"), m["arms"]["CANON"]["ckpt"]["run_ns"])
        want = old.plan_cells(m, old.select_cells(m, only=[EXP25]), farm_root=str(harness["dir"] / "farm"),
                              dump_root=str(harness["dir"] / "dumps"), canon_index=index, no_dump_blocks=[])
        assert json.dumps(plan["cells"]) == json.dumps(want)                      # the old plan, byte for byte
        rc, out = drive(clone, harness, "--only", CTL1, env=dict(env, DRY_RUN="1"))
        assert rc == 2 and not _calls(harness)[1:], out                          # an exp_35 cell needs them
    finally:
        _git(clone, "reset", "-q", "--hard", head)


def test_the_cotenant_tuple_is_an_exp35_gate(clone, harness, sleepers, tmp_path):
    pid = sleepers()
    _apps(harness, [(pid, 1)])
    rc, out = drive(clone, harness, "--only", EXP25, "--lanes", "1",
                    env={"ALLOW_SHARED_GPU": "1", **_tuple(clone, tmp_path, [pid])})
    assert rc == 2 and "exp_35" in out and not _calls(harness), out


# ---- Codex C2 BLOCKING 1: a frozen tuple, and the identity checked at the real pre-exec boundary ---- #
def test_the_tuple_must_carry_its_launch_entry_digest(clone, harness, sleepers, tmp_path):
    pid = sleepers()
    _apps(harness, [(pid, 1)])
    env = dict(_tuple(clone, tmp_path, [pid]), ALLOW_SHARED_GPU="1")
    for sha, needle in (("", "needs EXP35_APPROVED_COTENANTS_SHA256"), ("0" * 64, "the launch entry pins")):
        rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                        env=dict(env, EXP35_APPROVED_COTENANTS_SHA256=sha))
        assert rc == 2 and needle in out and not _calls(harness), (sha, out)


@pytest.mark.parametrize("at", [2, 3], ids=["main_loop_recheck", "in_run_cell"])
def test_a_tuple_replaced_between_checks_refuses_the_cell(clone, harness, sleepers, tmp_path, at):
    pid = sleepers()
    _apps(harness, [(pid, 1)])
    env = dict(_tuple(clone, tmp_path, [pid]), ALLOW_SHARED_GPU="1")
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "1",
                    env=dict(env, GUARD_MUTATE_AT=str(at), GUARD_MUTATE_FILE=env["EXP35_APPROVED_COTENANTS"]))
    assert rc == 2 and "the launch entry pins" in out and not _calls(harness), out
    assert not os.path.exists(_entry(clone, harness, CTL1)["receipt"])


def test_a_process_joining_after_the_checkpoint_hash_is_caught_before_the_launch(clone, harness, sleepers, tmp_path):
    """The selection and the main-loop re-check see the tuple; the farm + cache re-hash follow; a newcomer
    then joins -- the final check inside run_cell, immediately before the launcher, refuses (hard stop)."""
    pid = sleepers()
    for name in ("apps.1", "apps.2"):
        _apps(harness, [(pid, 1)], name=name)
    _apps(harness, [(pid, 1), (sleepers(), 1)])                               # query 3 onward
    rc, out = drive(clone, harness, "--only", f"{CTL1},{CTL8}", "--lanes", "1",
                    env=dict(_tuple(clone, tmp_path, [pid]), ALLOW_SHARED_GPU="1"))
    assert rc == 2 and "LANE CHANGED" in out and "ABORTED" in out and not _calls(harness), out
    assert not os.path.exists(_entry(clone, harness, CTL1)["receipt"])
    assert "RUN exp35_P1_T_S40000_K8_s42" not in out                         # no second cell after the stop


def test_an_unshared_exp35_lane_is_rechecked_before_the_launch(clone, harness, sleepers):
    """GPU 0 after the hand-over: ALLOW_SHARED_GPU unset; a foreign process appearing after the main-loop
    re-check is still caught by the in-run_cell check."""
    for name in ("apps.1", "apps.2"):
        _apps(harness, [], name=name)
    _apps(harness, [(sleepers(), 0)])
    rc, out = drive(clone, harness, "--only", CTL1, "--lanes", "0")
    assert rc == 2 and "LANE CHANGED" in out and not _calls(harness), out


def test_the_runner_selects_the_e2_fail_block_by_name(clone, harness):
    """plan §14 A1.3: the Planner's explicit phase-(ii) selection after an E2 FAIL needs no code change."""
    rc, out = drive(clone, harness, "--blocks", "E2-FAIL", "--limit", "2", env={"DRY_RUN": "1"})
    assert rc == 0 and out.count("would run") == 2, out
    assert "exp35_P1_OFF_S40000_K1_s43_th45" in out and "exp35_P1_OFF_S40000_K1_s44_th45" in out


BF_CFG = os.path.join("worklog", "worklog_yixun", "exp_07_fa_scratch_claude", "FLAC_AR_BF.json")


@pytest.mark.parametrize("change", ["removed", "edited_committed", "edited_uncommitted"])
def test_an_exp25_only_selection_ignores_the_bf_config(clone, harness, change):
    """Codex C3a HIGH 1: the B-F model config is in neither an exp_25-only selection's relevant tree nor its config
    pin check (both are derived from the selection); an exp_35 B-F selection still pins it."""
    head, path = _git(clone, "rev-parse", "HEAD"), clone / BF_CFG
    try:
        if change == "removed":
            _git(clone, "rm", "-q", BF_CFG)
        else:
            path.write_text(path.read_text().replace("{", '{"exp35_test_edit": 1, ', 1))
        if change != "edited_uncommitted":
            _git(clone, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-am", f"BF config {change}")
        env = {"EXPECT_HEAD": _git(clone, "rev-parse", "HEAD")}
        rc, out = drive(clone, harness, "--only", EXP25, env=dict(env, DRY_RUN="1"))
        assert rc == 0 and out.count("would run") == 1 and "FLAC_AR_BF" not in out, out
        rc, out = drive(clone, harness, "--only", BF_R, env=dict(env, DRY_RUN="1"))
        assert rc == 2 and ("CONFIG" in out or "NOT COMMITTED" in out) and "FLAC_AR_BF" in out, out
    finally:
        _git(clone, "reset", "-q", "--hard", head)


def test_every_exp35_selection_shares_one_relevant_tree(clone, harness):
    """The supervisor runs each exp_35 cell as its own selection, so the relevant tree (hence the campaign digest)
    must not depend on the selected arm -- one digest for the campaign (plan §5 item 4). A guard on the HIGH 1 fix."""
    import re
    digests = set()
    for cell in (CTL1, BF_R):
        rc, out = drive(clone, harness, "--only", cell, env={"DRY_RUN": "1"})
        assert rc == 0, out
        digests |= set(re.findall(r"evaluator tree digest ([0-9a-f]{64})", out))
    assert len(digests) == 1, digests
