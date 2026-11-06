"""exp_35 phase (ii) kit, part 5a (plan_orbitrir_yaw_evidence.md §15 items 7-8; Codex plan §15 r1 S5, S6 "launch
through a detached wrapper / supervisor that durably records the runner's exact rc; record complete identities --
PID, /proc start time, cwd, cmdline / argv SHA-256 -- for the detached wrapper and actual runner"): the wrapper,
run here as a plain subprocess of the sandbox kit copy (the detachment itself is part 5b) -- it proves it holds the
kit lock, records its own identity and environment, spawns the runner with the lock passed on, records the runner's
identity and its /proc environment, waits without ever signalling and records the EXACT rc (0 / 1 / 2 / 4; rc 1 with
the later cells still run); crashes before / after the spawn and before the identity / rc publication leave the state
for review.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys
import time

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import X25, FakeWorld, make_box

K = SB.load_kit()


class Crash(BaseException):
    """A simulated kill: not an Exception, so no handler of the code under test swallows it."""


def _pending(tmp_path, **run):
    """A box reserved, dry-run-passed and put at launch_pending exactly as launch() writes it (its own launch_spec),
    the kit lock held by the test (as the launcher holds it for the wrapper)."""
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path))
    SB.stub_control(tmp_path, dry={}, run=run)
    rc, msg = K.reserve(FakeWorld(), b.cfg, kit_commit=b.commit, h35=b.head, lanes=(0,), report=b.report,
                        report_sha256=b.report_sha)
    assert rc == 0, msg
    b.orch = msg.split()[1].rstrip(":")
    rc, msg = K.dry_run(FakeWorld(t=1_800_000_100.0), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
    assert rc == 0, msg
    b.fd = K.kit_lock(b.cfg)
    st = _state(b)
    st.update(phase="launch_pending", launch=dict(K.launch_spec(b.cfg, st, b.orch, b.fd), started=1.0, before=None))
    K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    return b


def _state(b):
    return json.load(open(os.path.join(b.orch, "state.json")))


def _wrapper(b, env=None, fd=None):
    fd = b.fd if fd is None else fd
    argv = [sys.executable, "-B", b.cfg.helper, "wrap", "--orch", b.orch, "--lock-fd", str(fd)]
    return subprocess.run(argv, cwd=b.run, env=env or dict(K.LAUNCH_ENV), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, pass_fds=(fd,), timeout=120)


def test_the_launch_spec_is_the_exact_runner_command_and_the_detached_wrapper(tmp_path):
    b = _pending(tmp_path)
    try:
        spec = _state(b)["launch"]
        argv, env, cwd = K.runner_command(b.cfg, (0,), dry=False)
        assert spec["runner"] == {"argv": argv, "env": env, "cwd": cwd, "log": os.path.join(b.orch, "runner.log")}
        assert spec["wrapper"] == {"argv": ["/usr/bin/setsid", "/usr/bin/nohup", b.cfg.py, "-B", b.cfg.helper, "wrap",
                                            "--orch", b.orch, "--lock-fd", str(b.fd)],
                                   "env": K.LAUNCH_ENV, "cwd": b.cfg.run, "log": os.path.join(b.orch, "wrapper.log")}
        assert not set(K.SHARING) & set(spec["runner"]["env"]) and "DRY_RUN" not in spec["runner"]["env"]
    finally:
        os.close(b.fd)


@pytest.mark.parametrize("run, rc", [({}, 0), ({"fail": ["exp35_P1_OFF_S40000_K8_s43_th225"]}, 1), ({"rc": 2}, 2),
                                     ({"rc": 4}, 4)])
def test_the_wrapper_records_the_runners_exact_rc_and_both_identities(tmp_path, run, rc):
    b = _pending(tmp_path, **run)
    try:
        p = _wrapper(b)
    finally:
        os.close(b.fd)
    assert p.returncode == 0, p.stdout + p.stderr
    st = _state(b)
    assert st["phase"] == "exited" and st["exit"]["rc"] == rc and st["exit"]["signal"] is None
    [_, call] = SB.stub_calls(tmp_path)                                          # the dry run's, then the launch's
    w, r = st["wrapper"], st["runner"]
    argv = st["launch"]["runner"]["argv"]
    assert r["pid"] == call["pid"] and r["cwd"] == os.path.realpath(b.run) and type(r["start_time"]) is int
    assert r["cmdline_sha256"] == K.sha256_bytes(b"".join(a.encode() + b"\0" for a in argv))
    assert r["environ_is_the_spec"] is True and r["sharing_absent"] is True
    assert SB.stub_env(tmp_path, call["pid"]) == dict(K.LAUNCH_ENV, EXPECT_HEAD=b.head)        # no DRY_RUN, no sharing
    assert w["environ_is_the_launch_env"] is True and w["sharing_absent"] is True and w["pid"] != r["pid"]
    outer = open(st["launch"]["runner"]["log"], "rb").read()
    assert st["exit"]["logs"]["outer"]["sha256"] == K.sha256_bytes(outer)
    ts = outer.decode().splitlines()[0].split(" -- ")[1]
    assert st["exit"]["logs"]["internal"]["path"] == os.path.join(b.run, X25, f"depth_canon_{ts}_eval_run.log")
    assert [e["what"].split(":")[0] for e in st["events"]][-3:] == ["spawning", "running", "exited"]
    lines = outer.decode().splitlines()
    if rc == 1:                                                                  # the later cells still ran
        i = next(k for k, ln in enumerate(lines) if ln.startswith("  !! FAILED exp35_P1_OFF_S40000_K8_s43_th225"))
        assert any(ln.startswith("  DONE exp35_CANON_OFF_S40000_K8_s46_th45") for ln in lines[i:])
        assert lines[-1] == "CAMPAIGN SELECTION INCOMPLETE - re-run to resume"
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"


def test_a_wrapper_without_the_kit_lock_refuses_and_writes_nothing(tmp_path):
    b = _pending(tmp_path)
    other = os.open(os.path.join(b.tmp, "not_the_lock"), os.O_RDWR | os.O_CREAT, 0o644)
    before = open(os.path.join(b.orch, "state.json"), "rb").read()
    try:
        p = _wrapper(b, fd=other)
    finally:
        os.close(other)
        os.close(b.fd)
    assert p.returncode == 3 and "WRAPPER REFUSED (nothing written)" in p.stdout
    assert open(os.path.join(b.orch, "state.json"), "rb").read() == before and len(SB.stub_calls(tmp_path)) == 1


def test_a_wrapper_finding_another_phase_refuses_and_writes_nothing(tmp_path):
    b = _pending(tmp_path)
    st = _state(b)
    st["phase"] = "dry_run_passed"
    K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    before = open(os.path.join(b.orch, "state.json"), "rb").read()
    try:
        p = _wrapper(b)
    finally:
        os.close(b.fd)
    assert p.returncode == 3 and open(os.path.join(b.orch, "state.json"), "rb").read() == before


def test_a_contaminated_wrapper_environment_is_refused_before_any_spawn(tmp_path):
    b = _pending(tmp_path)
    try:
        p = _wrapper(b, env=dict(K.LAUNCH_ENV, ALLOW_SHARED_GPU="1"))
    finally:
        os.close(b.fd)
    st = _state(b)
    assert p.returncode == 3 and st["phase"] == "wrapper_refused" and len(SB.stub_calls(tmp_path)) == 1
    assert any("environment" in x for x in st["failure"]["reasons"]) and st["wrapper"]["sharing_absent"] is False


# --------------------------------------------------------------------------- #
# crashes (S6 "crashes before / after spawn and before identity / rc publication"): in-process, no signal anywhere
# --------------------------------------------------------------------------- #
def _in_process(monkeypatch):
    real = K.environ_of
    monkeypatch.setattr(K, "environ_of", lambda pid, proc: dict(K.LAUNCH_ENV) if pid == os.getpid() else real(pid, proc))


def _wait_gone(b, pid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
        except ChildProcessError:
            return
        time.sleep(0.05)
    raise AssertionError(f"the stub runner {pid} did not end")


def test_a_crash_before_the_spawn_leaves_spawning_without_a_runner(tmp_path, monkeypatch):
    b = _pending(tmp_path)
    _in_process(monkeypatch)
    monkeypatch.setattr(K.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    try:
        with pytest.raises(Crash):
            K.wrap(b.cfg, orch=b.orch, lock_fd=b.fd)
    finally:
        os.close(b.fd)
    st = _state(b)
    assert st["phase"] == "spawning" and st["runner"] is None and st["exit"] is None and st["wrapper"]["pid"] == os.getpid()
    assert len(SB.stub_calls(tmp_path)) == 1


def test_a_spawn_error_is_a_durable_runner_spawn_failure(tmp_path, monkeypatch):
    b = _pending(tmp_path)
    _in_process(monkeypatch)
    monkeypatch.setattr(K.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("exec failed")))
    try:
        assert K.wrap(b.cfg, orch=b.orch, lock_fd=b.fd) == 3
    finally:
        os.close(b.fd)
    assert _state(b)["phase"] == "runner_spawn_failed" and "exec failed" in _state(b)["failure"]["reasons"][0]


@pytest.mark.parametrize("point", ["identify", "wait"])
def test_a_crash_after_the_spawn_leaves_the_runner_running_and_the_state_for_review(tmp_path, monkeypatch, point):
    hold = os.path.join(str(tmp_path), "hold")
    open(hold, "w").close()
    b = _pending(tmp_path, hold=hold)
    _in_process(monkeypatch)
    spawned = []
    real_popen = K.subprocess.Popen
    monkeypatch.setattr(K.subprocess, "Popen", lambda *a, **k: spawned.append(real_popen(*a, **k)) or spawned[-1])
    monkeypatch.setattr(K, "identify" if point == "identify" else "wait_for",
                        lambda *a, **k: (_ for _ in ()).throw(Crash()))
    try:
        with pytest.raises(Crash):
            K.wrap(b.cfg, orch=b.orch, lock_fd=b.fd)
        os.close(b.fd)
        [p] = spawned
        st = _state(b)
        assert st["phase"] == ("spawning" if point == "identify" else "running") and st["exit"] is None
        assert (st["runner"] is None) == (point == "identify")
        assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "held"             # the runner inherited the lock
    finally:
        if os.path.exists(hold):
            os.unlink(hold)                                   # the stub finishes by itself: nothing is ever signalled
        for q in spawned:
            _wait_gone(b, q.pid)
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld" and _state(b)["exit"] is None


def test_an_unidentified_runner_is_still_waited_for_and_its_rc_recorded(tmp_path, monkeypatch):
    b = _pending(tmp_path)
    _in_process(monkeypatch)
    monkeypatch.setattr(K, "identify", lambda p, spec, cfg: (None, "did not show its command"))
    try:
        assert K.wrap(b.cfg, orch=b.orch, lock_fd=b.fd) == 0
    finally:
        os.close(b.fd)
    st = _state(b)
    assert st["phase"] == "exited" and st["exit"]["rc"] == 0 and st["runner"]["unidentified"] == "did not show its command"
    assert "start_time" not in st["runner"]


def test_the_runner_environment_proof_is_read_from_proc_not_from_the_spec(tmp_path, monkeypatch):
    b = _pending(tmp_path)
    real = K.environ_of
    me = os.getpid()

    def contaminated(pid, proc):                              # /proc says the runner holds a sharing variable
        return dict(K.LAUNCH_ENV) if pid == me else dict(real(pid, proc), ALLOW_SHARED_GPU="1")
    monkeypatch.setattr(K, "environ_of", contaminated)
    try:
        assert K.wrap(b.cfg, orch=b.orch, lock_fd=b.fd) == 0
    finally:
        os.close(b.fd)
    r = _state(b)["runner"]
    assert r["environ_is_the_spec"] is False and r["sharing_absent"] is False and "ALLOW_SHARED_GPU" in r["environ_keys"]
