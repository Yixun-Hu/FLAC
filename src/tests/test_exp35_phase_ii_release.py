"""exp_35 phase (ii) kit, part 6a (plan_orbitrir_yaw_evidence.md §15 items 8-9; Codex plan §15 r1 S3, S6: "status
must be read-only and report identity validity, held locks / reservations, log / summary state, receipt counts for
the exact 68 cells, and final rc"; "release must require a durable exit / refusal state, no live wrapper / runner /
evaluator descendant, and no inherited phase-ii or runner lock ... authenticate each exact record immediately before
unlinking and fsync the reservation directory afterward"; never "release on ambiguous process / state evidence").

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys
import time

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, GPU1, FakeWorld, fake_proc, make_box

K = SB.load_kit()
T_DRY = 1_800_000_100.0


def _box(tmp_path, upto="reserved", lanes=(0,), **run):
    kw = {"lanes": "0,1", "decision": "Planner: both cards"} if len(lanes) == 2 else {}
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path), **kw)
    if len(lanes) == 2:
        with open(b.cfg.queue_log, "a") as f:
            f.write(SB.EVAL_DONE_S43)
    SB.stub_control(tmp_path, dry={}, run=run)
    rc, msg = K.reserve(FakeWorld(), b.cfg, kit_commit=b.commit, h35=b.head, lanes=lanes, report=b.report,
                        report_sha256=b.report_sha)
    assert rc == 0, msg
    b.orch = msg.split()[1].rstrip(":")
    if upto in ("dry_run_passed", "exited"):
        assert K.dry_run(FakeWorld(t=T_DRY), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)[0] == 0
    if upto == "exited":
        rc, msg = K.launch(FakeWorld(t=T_DRY + 60), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
        assert rc == 0, msg
        deadline = time.time() + 90
        while not (_state(b)["phase"] == "exited" and _state(b)["exit"]["logs"]) and time.time() < deadline:
            time.sleep(0.1)
        pid = _state(b)["wrapper"]["pid"]
        while time.time() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    break
            except ChildProcessError:
                break
            time.sleep(0.05)
    return b


def _state(b):
    return json.load(open(os.path.join(b.orch, "state.json")))


def _put(b, **kw):
    st = _state(b)
    st.update(**kw)
    K.write_json_atomic(os.path.join(b.orch, "state.json"), st)
    return st


def _release(b, **kw):
    args = dict(kit_commit=b.commit, h35=b.head, orch=b.orch)
    args.update(kw)
    return K.release(FakeWorld(t=T_DRY + 9000), b.cfg, **args)


# --------------------------------------------------------------------------- #
# release
# --------------------------------------------------------------------------- #
def test_an_exited_campaign_releases_its_exact_record(tmp_path):
    b = _box(tmp_path, "exited")
    assert _state(b)["exit"]["rc"] == 0 and os.path.exists(b.res())
    rc, msg = _release(b)
    st = _state(b)
    assert rc == 0 and msg.startswith("RELEASED") and not os.path.lexists(b.res())
    assert st["phase"] == "released" and st["reservations"] == [] and st["released"]["by"] == "planner"
    assert st["released"]["prior_phase"] == "exited" and st["released"]["records"] == [b.res()]
    assert _release(b)[0] == 1 and "not reserved or" in _release(b)[1]               # released once, never twice


@pytest.mark.parametrize("upto", ["reserved", "dry_run_passed"])
def test_a_never_launched_reservation_is_withdrawn(tmp_path, upto):
    b = _box(tmp_path, upto)
    rc, msg = _release(b)
    assert rc == 0 and not os.path.lexists(b.res()) and _state(b)["released"]["prior_phase"] == upto


@pytest.mark.parametrize("phase", ["dry_run_failed", "launch_refused", "wrapper_spawn_failed", "wrapper_refused",
                                   "runner_spawn_failed"])
def test_every_durable_refusal_state_is_releasable(tmp_path, phase):
    b = _box(tmp_path)
    _put(b, phase=phase, failure={"reasons": ["x"], "time": "t"})
    assert _release(b)[0] == 0 and not os.path.lexists(b.res())


@pytest.mark.parametrize("phase", ["launch_pending", "spawning", "running", "reserving", "reserve_refused",
                                   "reserve_failed", "released", "release_failed"])
def test_a_crash_or_unresolvable_state_is_never_released(tmp_path, phase):
    b = _box(tmp_path)
    before = _put(b, phase=phase)
    rc, msg = _release(b)
    assert rc == 1 and "RELEASE REFUSED (nothing unlinked)" in msg and os.path.exists(b.res())
    assert _state(b) == before


def _me():
    me = K.proc_identity(os.getpid(), "/proc")
    return {k: me[k] for k in K.ID_KEYS}


@pytest.mark.parametrize("case, reason", [
    ("live-runner", "may be alive"), ("live-wrapper", "may be alive"), ("unidentified-no-exit", "never identified"),
    ("supervisor", "exp35_supervisor.py"), ("runner-proc", "depth_canon_eval_run.sh"), ("kit-lock", "kit lock"),
    ("runner-lock", "runner lock is held"), ("missing", "gone"), ("bytes", "other bytes"), ("inode", "another inode")])
def test_every_ambiguous_or_live_evidence_refuses_the_release(tmp_path, case, reason):
    b = _box(tmp_path)
    _put(b, phase="exited", exit={"rc": 0, "signal": None, "ended": 1.0, "time": "t", "logs": None})
    holder = None
    if case == "live-runner":
        _put(b, runner=_me())
    elif case == "live-wrapper":
        _put(b, wrapper=_me())
    elif case == "unidentified-no-exit":
        _put(b, phase="wrapper_refused", runner={"pid": 4242, "unidentified": "x"}, exit=None)
    elif case == "supervisor":
        fake_proc(b.cfg.scan_proc, 777, ["/py", "/x/exp35_supervisor.py"])
    elif case == "runner-proc":
        fake_proc(b.cfg.scan_proc, 778, ["bash", "worklog/x/depth_canon_eval_run.sh", "--blocks", "OFF-P1"])
    elif case in ("kit-lock", "runner-lock"):                 # an evaluator descendant that inherited the lock
        lock = K.KIT_LOCK if case == "kit-lock" else K.RUNNER_LOCK
        holder = subprocess.Popen([sys.executable, "-c", SB.HOLDER, os.path.join(b.run, lock), "30"],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    elif case == "missing":
        os.unlink(b.res())
    elif case == "bytes":
        raw = open(b.res(), "rb").read()
        open(b.res(), "wb").write(raw.replace(b'"phase": "ii"', b'"phase": "x"'))
    else:
        raw = open(b.res(), "rb").read()
        open(b.res() + ".new", "wb").write(raw)
        os.replace(b.res() + ".new", b.res())
    before = open(os.path.join(b.orch, "state.json"), "rb").read()
    try:
        rc, msg = _release(b)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    assert rc == 1 and "RELEASE REFUSED (nothing unlinked)" in msg and reason in msg, msg
    assert open(os.path.join(b.orch, "state.json"), "rb").read() == before
    assert os.path.lexists(b.res()) == (case != "missing")


def test_a_reused_pid_is_another_process_and_does_not_block(tmp_path):
    b = _box(tmp_path)
    fake = os.path.join(b.tmp, "fakeproc")
    fake_proc(fake, 31337, ["bash", "x"], start=99)                                # the pid now names another process
    ident = {"pid": 31337, "start_time": 42, "cwd": "/r", "cmdline_sha256": "0" * 64}
    _put(b, phase="exited", wrapper=ident, runner=dict(ident),
         exit={"rc": 1, "signal": None, "ended": 1.0, "time": "t", "logs": None})
    b.cfg = b.cfg._replace(proc=fake)
    assert _release(b)[0] == 0 and not os.path.lexists(b.res())


def test_two_lanes_are_authenticated_together_before_any_unlink(tmp_path):
    b = _box(tmp_path, lanes=(0, 1))
    os.unlink(b.res(GPU1))
    open(b.res(GPU1), "w").write(json.dumps({"holder": "exp22"}))
    rc, msg = _release(b)
    assert rc == 1 and os.path.exists(b.res(GPU0)) and json.load(open(b.res(GPU1))) == {"holder": "exp22"}


def test_a_record_lost_at_its_unlink_is_a_retained_release_failure(tmp_path, monkeypatch):
    b = _box(tmp_path, lanes=(0, 1))
    real = K.release_reservation
    monkeypatch.setattr(K, "release_reservation", lambda h: real(h) if h["index"] == 0 else ["replaced at the unlink"])
    rc, msg = _release(b)
    st = _state(b)
    assert rc == 1 and st["phase"] == "release_failed" and not os.path.lexists(b.res(GPU0))
    assert os.path.exists(b.res(GPU1)) and st["reservations"][0]["unreleased"] == ["replaced at the unlink"]


# --------------------------------------------------------------------------- #
# status: read-only, informative
# --------------------------------------------------------------------------- #
def _tree(root):
    out = {}
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            p = os.path.join(d, n)
            st = os.lstat(p)
            out[p] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino, os.readlink(p) if os.path.islink(p) else None)
    return out


def test_status_writes_nothing_before_any_orchestration(tmp_path):
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path))
    before = _tree(b.tmp)
    rc, text = K.status(FakeWorld(), b.cfg, kit_commit=b.commit)
    assert rc == 0 and _tree(b.tmp) == before and not os.path.exists(os.path.join(b.run, K.KIT_LOCK))
    assert f"phase (i): state sha256 {b.state_sha}" in text and K.NAMES_SHA256 in text
    assert "kit: the reviewed commit" in text and "orchestrations: none" in text


def test_status_reports_an_exited_campaign_without_writing(tmp_path):
    b = _box(tmp_path, "exited", fail=["exp35_BF_OFF_S40000_K1_s44_th135"])
    before = _tree(b.tmp)
    rc, text = K.status(FakeWorld(t=T_DRY + 9000), b.cfg, orch=b.orch)
    assert rc == 0 and _tree(b.tmp) == before
    for want in ("phase exited", "our exact record", "wrapper: gone", "runner: gone", "DONE 67", "FAILED 1",
                 "receipts: 67/68", "rc 1 (a cell failed (later cells may still have run))",
                 "CAMPAIGN SELECTION INCOMPLETE - re-run to resume", "kit lock unheld", "runner lock unheld"):
        assert want in text, (want, text)


def test_status_reports_a_live_run_and_its_held_lock(tmp_path):
    hold = os.path.join(str(tmp_path), "hold")
    open(hold, "w").close()
    b = _box(tmp_path, "dry_run_passed", hold=hold)
    try:
        rc, msg = K.launch(FakeWorld(t=T_DRY + 60), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
        assert rc == 0, msg
        log, deadline = os.path.join(b.orch, "runner.log"), time.time() + 60
        while "CELL exp35_P1_OFF_S40000_K1_s42_th135" not in open(log).read():   # the stub now waits on the hold
            assert time.time() < deadline
            time.sleep(0.05)
        before = _tree(b.tmp)
        rc, text = K.status(FakeWorld(t=T_DRY + 70), b.cfg, orch=b.orch)
        assert rc == 0 and _tree(b.tmp) == before
        assert "phase running" in text and "wrapper: alive" in text and "runner: alive" in text
        assert "kit lock held" in text and "receipts: 0/68" in text
    finally:
        os.unlink(hold)
    deadline = time.time() + 90
    while _state(b)["phase"] != "exited" or not _state(b)["exit"]["logs"]:
        assert time.time() < deadline
        time.sleep(0.1)
    pid = _state(b)["wrapper"]["pid"]
    while time.time() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                break
        except ChildProcessError:
            break
        time.sleep(0.05)


def test_an_unreadable_exp35_folder_is_unreadable_not_a_crash(tmp_path):
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path))
    cfg = b.cfg._replace(run=os.path.join(b.tmp, "no-such-run-worktree"))
    with pytest.raises(K.Unreadable):
        K.orchestrations(cfg)
    rc, text = K.status(FakeWorld(), cfg)
    assert rc == 0 and "orchestrations: unreadable" in text
