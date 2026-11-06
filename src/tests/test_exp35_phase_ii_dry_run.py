"""exp_35 phase (ii) kit, part 4b (plan_orbitrir_yaw_evidence.md §15 item 7; Codex plan §15 r1 S4, S6): the dry-run
transaction -- from 'reserved' only, with this kit commit, under the kit lock: every precondition and the held
records again, then the EXACT runner command under DRY_RUN=1 in the launch environment (proved from the stub
runner's own /proc environ, whatever the caller's environment holds), its whole transcript parsed exactly, the
runner's own log byte-identical to it, its plan JSON the re-derived 68-entry plan, its pre-flight log the pinned
three lines; the record (environment, cwd, argv, rc, the log hashes, the planned set, all 68 argv) and the state
'dry_run_passed' -- or 'dry_run_failed', retained (never re-run, never cleared).

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, X25, FakeWorld, git, make_box

K = SB.load_kit()


def _box(tmp_path, **dry):
    stub = SB.install_stub(tmp_path)
    b = make_box(tmp_path, runner=stub)
    SB.stub_control(tmp_path, dry=dry)
    rc, msg = K.reserve(FakeWorld(), b.cfg, kit_commit=b.commit, h35=b.head, lanes=(0,), report=b.report,
                        report_sha256=b.report_sha)
    assert rc == 0, msg
    b.orch = msg.split()[1].rstrip(":")
    return b


def _dry(b, world=None, **kw):
    args = dict(kit_commit=b.commit, h35=b.head, orch=b.orch)
    args.update(kw)
    return K.dry_run(world or FakeWorld(t=1_800_000_200.0), b.cfg, **args)


def _state(b):
    return json.load(open(os.path.join(b.orch, "state.json")))


def test_the_dry_run_runs_the_exact_command_and_records_everything(tmp_path, monkeypatch):
    b = _box(tmp_path)
    for k, v in (("ALLOW_SHARED_GPU", "1"), ("EXP35_APPROVED_COTENANTS", "/tmp/x.json"), ("PYTHONPATH", "/x"),
                 ("EXP35_APPROVED_COTENANTS_SHA256", "0" * 64), ("GPU_MAX_POLLS", "1"), ("DRY_RUN", "0")):
        monkeypatch.setenv(k, v)                                              # a contaminated caller environment
    rc, msg = _dry(b)
    assert rc == 0 and msg.startswith("DRY RUN PASSED"), msg
    st = _state(b)
    d = st["dry_run"]
    assert st["phase"] == "dry_run_passed" and d["reasons"] == [] and d["rc"] == 0
    argv, env, cwd = K.runner_command(b.cfg, (0,), dry=True)
    assert (d["argv"], d["env"], d["cwd"]) == (argv, env, cwd)
    [call] = SB.stub_calls(tmp_path)
    assert call["argv"] == argv[2:] and call["dry"] is True and os.path.realpath(call["cwd"]) == os.path.realpath(b.run)
    assert SB.stub_env(tmp_path, call["pid"]) == dict(K.LAUNCH_ENV, EXPECT_HEAD=b.head, DRY_RUN="1")   # its /proc environ
    outer = open(d["outer_log"]["path"], "rb").read()
    assert os.path.dirname(d["outer_log"]["path"]) == b.orch and d["outer_log"]["sha256"] == K.sha256_bytes(outer)
    assert d["internal_log"]["path"] == os.path.join(b.run, X25, f"depth_canon_{d['ts']}_eval_run.log")
    assert d["internal_log"]["sha256"] == d["outer_log"]["sha256"]
    assert d["plan"]["path"].endswith(f"depth_canon_{d['ts']}_eval_plan.json") and d["plan"]["sha256"]
    assert d["preflight"]["sha256"] == K.sha256_bytes(K.PREFLIGHT.encode())
    _, m, rows = K.selection(SB.real_cells())
    assert d["n_ids"] == 68 and d["ids_sha256"] == K.NAMES_SHA256 and d["benign_lines"] == 2
    assert d["argv_records"] == [e["argv"] for e in K.expected_plan(m, rows, run=b.run, digest=b.digest)["cells"]]
    assert d["before"]["reasons"] == [] and d["after"]["reasons"] == []
    assert K.held_reasons(st["reservations"][0]) == [] and st["failure"] is None
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"


@pytest.mark.parametrize("case, reason", [("kit", "40-hex"), ("h35", "registered"), ("outside", "not an orchestration"),
                                          ("twice", "not reserved"), ("lock", "is held"), ("other-kit", "reserved with")])
def test_a_refusal_before_the_state_is_judged_changes_nothing(tmp_path, case, reason):
    b, kw, holder = _box(tmp_path), {}, None
    if case == "kit":
        kw["kit_commit"] = "x"
    elif case == "h35":
        kw["h35"] = "0" * 40
    elif case == "outside":
        kw["orch"] = os.path.join(b.tmp, "elsewhere")
    elif case == "twice":
        assert _dry(b)[0] == 0
    elif case == "other-kit":
        st = _state(b)
        st["kit"]["commit"] = "f" * 40
        K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    else:
        holder = subprocess.Popen([sys.executable, "-c", SB.HOLDER, os.path.join(b.run, K.KIT_LOCK), "30"],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    before, calls = open(os.path.join(b.orch, "state.json"), "rb").read(), len(SB.stub_calls(tmp_path))
    try:
        rc, msg = _dry(b, **kw)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    assert rc == 1 and "REFUSED" in msg and reason in msg, msg
    assert open(os.path.join(b.orch, "state.json"), "rb").read() == before and len(SB.stub_calls(tmp_path)) == calls


@pytest.mark.parametrize("case, reason", [("busy", "compute process"), ("replaced", "not our record"),
                                          ("moved", "is not H35"), ("report", "evidence changed")])
def test_a_failed_recheck_is_a_retained_dry_run_failure_and_the_runner_never_runs(tmp_path, case, reason):
    b, w = _box(tmp_path), FakeWorld(t=1_800_000_200.0)
    if case == "busy":
        w.answers["apps"] = f"4242, {GPU0}\n"
    elif case == "replaced":
        os.unlink(b.res())
        open(b.res(), "w").write(json.dumps({"holder": "exp22"}))
    elif case == "moved":
        git(b.run, "commit", "-q", "--allow-empty", "-m", "moved")
    else:
        st = _state(b)
        st["evidence"]["eta"] = "a different ETA"
        K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    rc, msg = _dry(b, w)
    st = _state(b)
    assert rc == 1 and "DRY RUN FAILED" in msg and reason in msg, msg
    assert st["phase"] == "dry_run_failed" and any(reason in x for x in st["failure"]["reasons"])
    assert SB.stub_calls(tmp_path) == [] and st["dry_run"]["reasons"] == st["failure"]["reasons"]
    if case == "replaced":
        assert json.load(open(b.res())) == {"holder": "exp22"}
    assert _dry(b, w)[0] == 1 and "not reserved" in _dry(b, w)[1]                    # never re-run, never cleared


@pytest.mark.parametrize("fault, reason", [
    ({"abort": True}, "rc 2"), ({"rc": 3}, "rc 3"), ({"internal_extra": "only in the runner's own log"}, "internal log"),
    ({"plan_tamper": True}, "plan"), ({"preflight_extra": "a fourth line\n"}, "preflight"),
    ({"skip": "exp35_P1_OFF_S40000_K1_s42_th135"}, "a SKIP line"), ({"e2fail": True}, "an E2-FAIL cell"),
    ({"quarantine": "exp35_BF_OFF_S40000_K8_s46_th225"}, "a quarantine line")])
def test_every_runner_fault_is_a_retained_dry_run_failure(tmp_path, fault, reason):
    b = _box(tmp_path, **fault)
    rc, msg = _dry(b)
    st = _state(b)
    assert rc == 1 and reason in msg, msg
    assert st["phase"] == "dry_run_failed" and st["dry_run"]["rc"] == {"abort": 2, "rc": 3}.get(next(iter(fault)), 0)
    assert len(SB.stub_calls(tmp_path)) == 1 and K.held_reasons(st["reservations"][0]) == []   # the record stays
