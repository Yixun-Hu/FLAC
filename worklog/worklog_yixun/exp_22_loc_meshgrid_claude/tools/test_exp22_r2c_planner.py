"""exp_22 A7 (plan §9 A7 r2 (v)(2), r3 (e) (i)): the no-exp22-run-active check, the Planner's release after a
stop or a crash (no live supervisor / line child by pid + start time, our exact record only) and the launcher's
identity confirmation of the detached supervisor."""
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, HOLDER, PIN2, RUN, FakeWorld, cfg_for, load, res_file, reserve, setup

E = load("exp22_r2c_evidence")
S = load("exp22_r2c_supervisor")
DEAD = {"pid": 4194000, "start_time": 1, "cwd": "/", "cmdline_sha256": "0" * 64}   # never a live process here


def _stop(orch, phase="stopped"):
    """The state as a stopped (or crashed) supervisor leaves it: a supervisor identity, the reservation kept."""
    st = json.load(open(orch / "state.json"))
    st.update(phase=phase, supervisor=DEAD, stop={"reason": "a test stop", "time": "t"} if phase == "stopped" else None)
    E.write_json_atomic(str(orch / "state.json"), st)
    return st


# --------------------------------------------------------------------------- #
# no exp22 run active; the Planner's release; the launcher's confirmation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phase", ["stopped", "line_complete", "child_identified", "supervisor_started", "reserved"])
def test_a_stopped_or_crashed_run_stays_active_until_the_planner_releases_it(tmp_path, phase):
    """Codex A7-C1 HIGH 2 (A7 r3 (e)): only "released" is resolved -- a stopped or crashed state blocks every new run
    even when its reservation file has disappeared."""
    cfg, w, wt, orch, seal = setup(tmp_path)
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    if phase != "reserved":
        _stop(orch, phase)
    os.unlink(res_file(cfg))
    assert any(f"phase {phase}" in x for x in S.active_reasons(cfg, str(wt)))
    assert S.planner_release(cfg, str(wt), str(orch))[0] == 0
    assert S.active_reasons(cfg, str(wt)) == [] and json.load(open(orch / "state.json"))["phase"] == "released"


def test_a_foreign_replacement_waits_for_its_owner_then_the_absent_record_release_closes_the_old_state(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    _stop(orch)
    os.unlink(res_file(cfg))
    foreign = json.dumps({"holder": "exp35", "h35": "x"})
    open(res_file(cfg), "w").write(foreign)                                   # another holder took the card
    rc, msg = S.planner_release(cfg, str(wt), str(orch))
    assert rc == 1 and "nothing unlinked" in msg and open(res_file(cfg)).read() == foreign
    assert any("phase stopped" in x for x in S.active_reasons(cfg, str(wt)))   # the old state still waits
    os.unlink(res_file(cfg))                                                   # its owner removes it
    rc, msg = S.planner_release(cfg, str(wt), str(orch))
    assert rc == 0 and "already absent" in msg and S.active_reasons(cfg, str(wt)) == []


def test_an_active_exp22_run_is_seen_by_its_reservation_state_lock_or_process(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    assert S.active_reasons(cfg, str(wt)) == []
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    why = S.active_reasons(cfg, str(wt))
    assert any("holder exp22" in x for x in why) and any("reserved" in x for x in why)
    assert S.active_reasons(cfg, str(wt), except_orch=str(orch)) and \
        not any("r2_preflight" in x for x in S.active_reasons(cfg, str(wt), except_orch=str(orch)))
    _stop(orch)
    os.unlink(res_file(cfg))                                                   # removed outside the release path
    assert any("phase stopped" in x for x in S.active_reasons(cfg, str(wt)))  # Codex A7-C1 HIGH 2: NOT resolved
    rc, msg = S.planner_release(cfg, str(wt), str(orch))
    assert rc == 0 and "already absent" in msg, msg
    assert S.active_reasons(cfg, str(wt)) == []                               # only "released" resolves a run
    lock = wt / "outputs_loc" / "exp22" / "r2_preflight.lock"
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        assert any("lock" in x for x in S.active_reasons(cfg, str(wt)))
    finally:
        p.kill()
        p.wait()
    (orch / "state.json").write_text("{broken")
    assert any("unreadable" in x for x in S.active_reasons(cfg, str(wt)))


def test_the_planner_release_needs_no_live_supervisor_or_child_and_our_exact_record(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    cfg = cfg._replace(proc="/proc")                                           # the identities are real processes
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    st = json.load(open(orch / "state.json"))
    me = C.proc_identity(os.getpid())
    dead = {k: me[k] for k in ("pid", "start_time", "cwd", "cmdline_sha256")}
    dead["start_time"] += 7                                                    # a crashed supervisor: gone
    st.update(phase="child_identified", supervisor=dead,
              current={"child": dict(me, expected_cmdline_sha256=me["cmdline_sha256"])})
    E.write_json_atomic(str(orch / "state.json"), st)
    rc, msg = S.planner_release(cfg, str(wt), str(orch))
    assert rc == 1 and "alive" in msg and os.path.exists(res_file(cfg))       # our "child" (this process) lives
    st["current"]["child"]["start_time"] += 1                                  # PID reuse: another process
    E.write_json_atomic(str(orch / "state.json"), st)
    rc, msg = S.planner_release(cfg, str(wt), str(orch))
    assert rc == 0, msg
    st = json.load(open(orch / "state.json"))
    assert st["phase"] == "released" and st["released"]["by"] == "planner" and not os.path.exists(res_file(cfg))
    assert S.planner_release(cfg, str(wt), str(orch))[0] == 1                  # once only


def test_the_confirmation_binds_the_detached_supervisor_by_pid_start_time_and_cmdline(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    cfg = cfg._replace(proc="/proc")
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    st = json.load(open(orch / "state.json"))
    st.update(phase="supervisor_started", supervisor={k: v for k, v in C.proc_identity(os.getpid()).items()
                                                      if k in ("pid", "start_time", "cwd", "cmdline_sha256")})
    E.write_json_atomic(str(orch / "state.json"), st)
    lock = wt / "outputs_loc" / "exp22" / "r2_preflight.lock"
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert S.confirm(cfg, str(wt), str(orch), os.getpid(), timeout=1)[0] == 0
        assert S.confirm(cfg, str(wt), str(orch), os.getpid() + 1, timeout=1)[0] == 1           # another pid
        st["supervisor"]["start_time"] += 1                                                      # PID reuse
        E.write_json_atomic(str(orch / "state.json"), st)
        assert S.confirm(cfg, str(wt), str(orch), os.getpid(), timeout=1)[0] == 1
    finally:
        os.close(fd)
    st["supervisor"]["start_time"] -= 1
    E.write_json_atomic(str(orch / "state.json"), st)
    assert S.confirm(cfg, str(wt), str(orch), os.getpid(), timeout=1)[0] == 1                   # the lock not held


class Crash(BaseException):
    pass


def test_a_crashed_reserve_leaves_an_orphan_that_counts_as_an_active_exp22_run(tmp_path, monkeypatch):
    cfg, w, wt, orch, seal = setup(tmp_path)
    real = S.create_reservation
    monkeypatch.setattr(S, "create_reservation", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(Crash()))[0])
    with pytest.raises(Crash):
        reserve(cfg, w, wt, orch, seal)
    monkeypatch.undo()
    assert any("holder exp22" in x for x in S.active_reasons(cfg, str(wt)))   # no state, yet the run is visible
