"""exp_35 phase (ii) kit, part 5b (plan_orbitrir_yaw_evidence.md §15 items 6-8; Codex plan §15 r1 S4, S5, S6 "before
launch, reauthenticate every reservation, recheck the cards and runner lock, and require the matching successful
dry run"): the launch -- from 'dry_run_passed' only, the dry run of THIS exact command and fresh, the recheck,
'launch_pending' written BEFORE any spawn, the last look, then the wrapper DETACHED (setsid + nohup, exactly the
launch environment, cwd = the run worktree, stdin /dev/null, the kit lock's fd inherited) and its identity confirmed
read-only; a refused launch is retained ('launch_refused'), never retried in place; nothing is ever relaunched.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys
import time

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, FakeWorld, fake_proc, make_box

K = SB.load_kit()
T_DRY = 1_800_000_100.0


class Crash(BaseException):
    """A simulated kill: not an Exception, so no handler of the code under test swallows it."""


def _ready(tmp_path, **run):
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path))
    SB.stub_control(tmp_path, dry={}, run=run)
    rc, msg = K.reserve(FakeWorld(), b.cfg, kit_commit=b.commit, h35=b.head, lanes=(0,), report=b.report,
                        report_sha256=b.report_sha)
    assert rc == 0, msg
    b.orch = msg.split()[1].rstrip(":")
    rc, msg = K.dry_run(FakeWorld(t=T_DRY), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
    assert rc == 0, msg
    return b


def _state(b):
    return json.load(open(os.path.join(b.orch, "state.json")))


def _launch(b, world=None, **kw):
    args = dict(kit_commit=b.commit, h35=b.head, orch=b.orch)
    args.update(kw)
    return K.launch(world or FakeWorld(t=T_DRY + 60), b.cfg, **args)


def _until_exited(b, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = _state(b)
        if st["phase"] == "exited" and st["exit"]["logs"] is not None:
            return st
        time.sleep(0.1)
    raise AssertionError(f"the detached wrapper did not finish: {_state(b)['phase']}")


def _reap(pid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
        except ChildProcessError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} did not end")


def _stat_fields(pid):
    text = open(f"/proc/{pid}/stat").read()
    return text[text.rindex(")") + 2:].split()


def test_the_launch_detaches_the_wrapper_which_runs_the_exact_runner_to_its_recorded_rc(tmp_path):
    hold = os.path.join(str(tmp_path), "hold")
    open(hold, "w").close()
    b = _ready(tmp_path, hold=hold)
    try:
        rc, msg = _launch(b)
        assert rc == 0 and msg.startswith("LAUNCHED"), msg
        st = _state(b)
        w, r = st["wrapper"], st["runner"]
        assert st["phase"] == "running" and st["launch"]["wrapper"]["argv"][:2] == ["/usr/bin/setsid", "/usr/bin/nohup"]
        env = sorted(x for x in open(f"/proc/{w['pid']}/environ", "rb").read().decode().split("\0") if x)
        assert env == sorted(f"{k}={v}" for k, v in K.LAUNCH_ENV.items())        # env -i: exactly the launch environment
        assert int(_stat_fields(w["pid"])[3]) == w["pid"]                        # setsid: its own session
        assert os.readlink(f"/proc/{w['pid']}/fd/0") == "/dev/null" and w["cwd"] == os.path.realpath(b.run)
        assert int(_stat_fields(r["pid"])[1]) == w["pid"] and r["environ_is_the_spec"] and r["sharing_absent"]
        assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "held"           # the wrapper (and runner) hold it
        assert st["launch"]["before"]["reasons"] == [] and st["launch"]["runner"]["argv"] == K.runner_command(
            b.cfg, (0,), dry=False)[0]
    finally:
        if os.path.exists(hold):
            os.unlink(hold)
    st = _until_exited(b)
    _reap(st["wrapper"]["pid"])
    assert st["exit"]["rc"] == 0 and K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"
    assert [e["what"].split(":")[0] for e in st["events"]][-5:] == [
        "dry_run_passed", "launch_pending", "spawning", "running", "exited"]
    assert "running" in open(os.path.join(b.orch, "wrapper.log")).read()


@pytest.mark.parametrize("case, reason", [("kit", "40-hex"), ("h35", "registered"), ("no-dry-run", "not dry_run_passed"),
                                          ("lock", "is held")])
def test_a_refusal_before_the_state_is_judged_changes_nothing(tmp_path, case, reason):
    b, kw, holder = _ready(tmp_path), {}, None
    if case == "kit":
        kw["kit_commit"] = "x"
    elif case == "h35":
        kw["h35"] = "0" * 40
    elif case == "no-dry-run":
        st = _state(b)
        st["phase"] = "reserved"
        K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    else:
        holder = subprocess.Popen([sys.executable, "-c", SB.HOLDER, os.path.join(b.run, K.KIT_LOCK), "30"],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    before = open(os.path.join(b.orch, "state.json"), "rb").read()
    try:
        rc, msg = _launch(b, **kw)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    assert rc == 1 and "REFUSED" in msg and reason in msg, msg
    assert open(os.path.join(b.orch, "state.json"), "rb").read() == before
    assert not os.path.exists(os.path.join(b.orch, "wrapper.log"))


def test_an_exited_campaign_is_never_relaunched(tmp_path):
    b = _ready(tmp_path)
    rc, msg = _launch(b)
    assert rc == 0, msg
    st = _until_exited(b)
    _reap(st["wrapper"]["pid"])
    rc, msg = _launch(b)
    assert rc == 1 and "not dry_run_passed" in msg and _state(b)["phase"] == "exited"
    assert len(SB.stub_calls(tmp_path)) == 2


@pytest.mark.parametrize("case, reason", [("stale", "older than"), ("other-command", "not this exact command"),
                                          ("busy", "compute process"), ("replaced", "not our record"),
                                          ("runner-lock", "runner lock is held"), ("supervisor", "exp35_supervisor.py")])
def test_a_failed_launch_check_is_a_retained_refusal_and_nothing_is_spawned(tmp_path, case, reason):
    b, w, holder = _ready(tmp_path), FakeWorld(t=T_DRY + 60), None
    if case == "stale":
        w = FakeWorld(t=T_DRY + b.cfg.dry_max_age_s + 1)
    elif case == "other-command":
        st = _state(b)
        st["dry_run"]["argv"] = st["dry_run"]["argv"][:-2]
        K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    elif case == "busy":
        w.answers["apps"] = f"4242, {GPU0}\n"
    elif case == "replaced":
        os.unlink(b.res())
        open(b.res(), "w").write(json.dumps({"holder": "exp22"}))
    elif case == "runner-lock":
        holder = subprocess.Popen([sys.executable, "-c", SB.HOLDER, os.path.join(b.run, K.RUNNER_LOCK), "30"],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    else:
        fake_proc(b.cfg.scan_proc, 4321, ["/py", "/x/exp35_supervisor.py", "--h35", "a" * 40])
    try:
        rc, msg = _launch(b, w)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    st = _state(b)
    assert rc == 1 and "LAUNCH REFUSED" in msg and reason in msg, msg
    assert st["phase"] == "launch_refused" and st["launch"] is None and st["wrapper"] is None
    assert not os.path.exists(os.path.join(b.orch, "wrapper.log")) and len(SB.stub_calls(tmp_path)) == 1
    assert _launch(b, FakeWorld(t=T_DRY + 60))[0] == 1                               # never retried in place


def test_a_change_at_the_last_look_refuses_after_launch_pending_without_any_spawn(tmp_path):
    b, w = _ready(tmp_path), FakeWorld(t=T_DRY + 60)
    calls = []
    w.answers["apps"] = lambda: calls.append(1) or ("" if len(calls) == 1 else f"4242, {GPU0}\n")
    rc, msg = _launch(b, w)
    st = _state(b)
    assert rc == 1 and "at the last look" in msg and st["phase"] == "launch_refused" and st["launch"] is not None
    assert st["wrapper"] is None and not os.path.exists(os.path.join(b.orch, "wrapper.log"))
    assert [e["what"].split(":")[0] for e in st["events"]][-2:] == ["launch_pending", "launch_refused"]


def test_a_wrapper_that_cannot_be_spawned_is_a_durable_failure(tmp_path, monkeypatch):
    b = _ready(tmp_path)
    monkeypatch.setattr(K, "spawn", lambda *a, **k: (_ for _ in ()).throw(OSError("no setsid")))
    rc, msg = _launch(b)
    assert rc == 1 and _state(b)["phase"] == "wrapper_spawn_failed" and "no setsid" in msg


def test_a_crash_between_launch_pending_and_the_spawn_leaves_launch_pending(tmp_path, monkeypatch):
    b = _ready(tmp_path)
    monkeypatch.setattr(K, "spawn", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        _launch(b)
    monkeypatch.undo()
    st = _state(b)
    assert st["phase"] == "launch_pending" and st["wrapper"] is None and st["launch"] is not None
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"


def test_a_wrapper_that_never_reports_is_not_confirmed_and_never_signalled(tmp_path, monkeypatch):
    b = _ready(tmp_path)
    b.cfg = b.cfg._replace(confirm_s=1)
    real, started = K.spawn, []

    def mute(argv, **kw):                                     # a "wrapper" that never writes the state
        started.append(real(["sleep", "3"], **kw))
        return started[-1]
    monkeypatch.setattr(K, "spawn", mute)
    rc, msg = _launch(b)
    assert rc == 1 and "NOT CONFIRMED" in msg and _state(b)["phase"] == "launch_pending"
    assert started[0].poll() is None                         # still alive: never signalled
    started[0].wait()


@pytest.mark.parametrize("case, ok", [("alive", True), ("other-pid", False), ("runner-gone", False),
                                      ("runner-unidentified", False), ("wrapper-reused", False)])
def test_the_confirmation_needs_the_spawned_pid_and_both_identities_alive(tmp_path, case, ok):
    b = _ready(tmp_path)
    me = K.proc_identity(os.getpid(), "/proc")
    me = {k: me[k] for k in K.ID_KEYS}
    st = _state(b)
    st.update(phase="running", wrapper=dict(me), runner=dict(me))
    pid = os.getpid()
    if case == "other-pid":
        pid += 1
    elif case == "runner-gone":
        st["runner"]["pid"] = 2 ** 22 + 17                                   # no such process
    elif case == "runner-unidentified":
        st["runner"] = {"pid": os.getpid(), "unidentified": "did not show its command"}
    elif case == "wrapper-reused":
        st["wrapper"]["start_time"] -= 1                                    # the pid now names another process
    K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    rc, msg = K.confirm(b.cfg, b.orch, pid)
    assert (rc == 0 and msg.startswith("LAUNCHED:")) if ok else (rc == 1 and "NOT CONFIRMED" in msg), msg
