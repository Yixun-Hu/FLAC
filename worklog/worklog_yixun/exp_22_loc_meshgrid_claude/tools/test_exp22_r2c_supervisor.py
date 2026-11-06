"""exp_22 A7 (plan §9 A7 r2 (iv) (v)(2), r3 (a) (c) (e) (f) (g) (i), r4, r5): the supervisor's run over the REGISTERED
fifteen lines on a sandbox checkout (the real overlay + launcher bytes, the real pytest, stand-in targets; CPU only):
the fsynced state machine, the pre-line and during-line re-checks, the backend and T rules, O_EXCL dispatch records in
the orchestration directory only, stop at the first failure (the nonzero exit recorded first), a foreign process
during a running line (the child untouched, no later line), fault injection at every state / spawn / release
transition, a real kill of the supervisor with a surviving child, the exact-record release."""
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import types

import pytest

from conftest import C, GPU0, MAP, FakeWorld, cfg_for, load, sandbox_repo

E = load("exp22_r2c_evidence")
S = load("exp22_r2c_supervisor")
RUN = "20261106T130501"
NAMES = ["T", "GV", "V_GV", "G1", "V_G1", "G2", "V_G2", "G3", "V_G3", "G3D", "V_G3D", "G0", "V_G0", "P", "A"]


class Crash(BaseException):
    """A simulated kill (not an Exception: nothing under test may swallow it)."""


class LineWorld(FakeWorld):
    def spawn(self, argv, env, cwd, log_fd, lock_fd):
        return S.spawn_line(argv, env, cwd, log_fd, lock_fd)


def _reserve(w, pin2, cfg, world):
    orch = os.path.join(w, "outputs_loc", "exp22", f"r2_preflight_{RUN}")
    os.mkdir(orch)
    seal = os.path.join(orch, "seal.txt")
    with open(seal, "w") as f:
        f.write(f"pin2 {pin2}\nrun {RUN}\ncard GPU0 {GPU0} 0\n")
    rc, msg = S.reserve(world, cfg, worktree=w, orch=orch, pin2=pin2, card="GPU0", index=0, seal=seal,
                        seal_sha256=E.sha256_bytes(open(seal, "rb").read()))
    assert rc == 0, msg
    return orch


@pytest.fixture
def box(tmp_path):
    w, pin2 = sandbox_repo(tmp_path)
    cfg = cfg_for(tmp_path)._replace(proc="/proc")
    world = LineWorld()
    orch = _reserve(w, pin2, cfg, world)
    fd = os.open(S.lock_path(w), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)                              # as the launcher holds it for the supervisor
    b = types.SimpleNamespace(w=w, pin2=pin2, cfg=cfg, orch=orch, world=world, fd=fd)
    b.sup = lambda **kw: S.Supervisor(world, cfg, orch, fd, worktree=w, child_poll_s=0.05, relist=(2, 0.05), **kw)
    b.state = lambda: json.load(open(os.path.join(orch, "state.json")))
    b.control = lambda d: open(os.path.join(w, "a7_control.json"), "w").write(json.dumps(d))
    b.res = os.path.join(cfg.reservations, f"{GPU0}.json")
    b.path = lambda rel: os.path.join(w, rel)
    b.lines = E.expand_lines(json.load(open(S.LINES_JSON)), run=RUN, index=0, worktree=w, py=C.PY,
                             pytest_main=C.PYTEST_MAIN)
    yield b
    os.close(fd)


def _sha(path):
    return E.sha256_bytes(open(path, "rb").read())


def test_the_fifteen_registered_lines_run_and_the_exact_record_is_released(box):
    assert box.sup().run() == 0
    st = box.state()
    assert st["phase"] == "released" and st["released"]["by"] == "supervisor" and st["current"] is None
    assert [ln["name"] for ln in st["lines"]] == NAMES and all(ln["admitted"] and ln["rc"] == 0 for ln in st["lines"])
    assert not os.path.lexists(box.res)                                     # the exact record, released
    me = C.proc_identity(os.getpid())
    assert st["supervisor"] == {k: me[k] for k in S.ID_KEYS}
    orch_files = sorted(os.listdir(box.orch))
    assert [f for f in orch_files if f.startswith("dispatch_")] == [f"dispatch_{i:02d}_{n}.json" for i, n in
                                                                   enumerate(NAMES, 1)]
    for ln, rec in zip(box.lines, st["lines"]):
        doc = json.load(open(box.path(ln["record"])))
        assert rec["record_sha256"] == _sha(box.path(ln["record"])) and doc["admitted"] and doc["refusals"] == []
        assert doc["argv"] == ln["argv"] and doc["env"] == ln["env"] and doc["cwd"] == box.w and doc["pin2"] == box.pin2
        assert doc["card"] == {"name": "GPU0", "uuid": GPU0, "index": 0} and doc["exit_code"] == 0
        assert doc["log"]["sha256"] == _sha(box.path(ln["log"])) and doc["reservation"] == st["reservation"]
        assert doc["sentinel"] == (E.sentinel_line(ln["target"]) if ln["wrapped"] else None)
        assert [o["path"] for o in doc["output_paths"]] == ln["outputs"]
        assert all(o["sha256"] == _sha(box.path(o["path"])) for o in doc["output_paths"] + doc["input_paths"])
        assert doc["child"]["cmdline_sha256"] == doc["child"]["expected_cmdline_sha256"] and doc["child"]["start_time"]
        assert doc["launcher_sha256"] == E.LAUNCHER_SHA256 and doc["overlay_sha256"] == E.OVERLAY_SHA256
        if ln["kind"] in ("gate", "probe"):                                  # the target saw exactly its argv (runpy)
            text = open(box.path(ln["log"])).read()
            assert json.loads(re.search(r"^stand-in \S+ argv (.*)$", text, re.M)[1]) == ln["argv"][3:]
    t = json.load(open(box.path(box.lines[0]["record"])))
    assert t["selected_node_ids"] == list(E.T_NODE_IDS) and E.SUMMARY_RE.fullmatch(t["pytest_summary"])
    assert t["junit_xml"] == {"path": box.lines[0]["junit"], "sha256": _sha(box.path(box.lines[0]["junit"]))}
    assert t["pytest_passed_lines_sha256"] and t["output_paths"] == []
    gates, probe = box.path(f"outputs_loc/exp22/r2_gates_{RUN}"), box.path(f"outputs_loc/exp22/r2_probe_BF_{RUN}")
    assert sorted(os.listdir(gates)) == sorted(f"{x}.json" for x in ("vanilla_cache_parity", "fa_split_parity",
                                                                     "fa_cache_parity", "fa_production_parity",
                                                                     "fa_production_drift", "vanilla_replay_regression"))
    assert os.listdir(probe) == ["diagnostics_BF_bathrooms14.json"]           # only the drivers' scientific outputs
    assert json.load(open(os.path.join(box.orch, "heartbeat.json")))["phase"] == "released"


def _phases(monkeypatch):
    seen, real = [], S.E.write_json_atomic

    def spy(path, doc):
        if path.endswith("state.json"):
            seen.append((doc["phase"], (doc["current"] or {}).get("name"), (doc["current"] or {}).get("rc")))
        return real(path, doc)
    monkeypatch.setattr(S.E, "write_json_atomic", spy)
    return seen


def test_every_transition_is_saved_in_order(box, monkeypatch):
    seen = _phases(monkeypatch)
    assert box.sup().run() == 0
    phases = [p for p, _, _ in seen]
    assert phases[0] == "supervisor_started" and phases[-1] == "released"
    t = [(p, n) for p, n, _ in seen if n == "T"]
    assert [p for p, _ in t][:3] == ["line_pending", "child_identified", "line_complete"]


def test_stop_at_the_first_failure_with_the_nonzero_exit_recorded_first(box, monkeypatch):
    box.control({"fa_split_parity": {"rc": 1}})
    seen = _phases(monkeypatch)
    assert box.sup().run() == 3
    st = box.state()
    assert st["phase"] == "stopped" and "G1" in st["stop"]["reason"] and [ln["name"] for ln in st["lines"]] == NAMES[:4]
    assert st["lines"][-1]["rc"] == 1 and not st["lines"][-1]["admitted"]
    assert seen.index(("line_complete", "G1", 1)) < [p for p, _, _ in seen].index("stopped")
    doc = json.load(open(box.path(box.lines[3]["record"])))
    assert doc["exit_code"] == 1 and not doc["admitted"] and "exit code 1" in doc["refusals"]
    assert not os.path.exists(box.path(box.lines[4]["log"])) and os.path.exists(box.res)     # nothing after; kept


@pytest.mark.parametrize("control, stop_at, needle", [
    ({"vanilla_cache_parity": {"print": ["ModuleNotFoundError: No module named 'flash_attn'"]}}, "GV", "No module named"),
    ({"vanilla_cache_parity": {"print": [E.sentinel_line("localize_meshgrid.py")]}}, "GV", "not exactly one"),
    ({"T": {"test_skipping_the_metric_stack_changes_the_draw": "skip"}}, "T", "skipped"),
    ({"T": {"test_our_call_graph_matches_an_eval_flac_faithful_reference": "fail"}}, "T", "exit code 1"),
    ({"vanilla_cache_parity": {"pass": False}}, "V_GV", "RE-VERIFIED"),
    ({"fa_cache_parity": {"no_output": True}}, "G2", "missing"),
    ({"diagnostics_BF_bathrooms14": {"total": 200.0}}, "A", "WITHIN THE CAP"),
], ids=["missing-module", "second-sentinel", "t-skip", "t-fail", "reverify", "no-output", "a5-above-cap"])
def test_a_refused_line_stops_the_preflight_and_keeps_the_reservation(box, control, stop_at, needle):
    box.control(control)
    assert box.sup().run() == 3
    st = box.state()
    names = [ln["name"] for ln in st["lines"]]
    assert st["phase"] == "stopped" and names == NAMES[:NAMES.index(stop_at) + 1] and not st["lines"][-1]["admitted"]
    doc = json.load(open(box.path(box.lines[NAMES.index(stop_at)]["record"])))
    assert needle in " ".join(doc["refusals"]) and os.path.exists(box.res)


def test_a_foreign_process_during_a_running_line_stops_further_lines_and_never_touches_the_child(box):
    box.control({"fa_production_parity": {"sleep": 1.5}})
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    g3 = box.lines[NAMES.index("G3")]
    try:
        box.world.answers["apps"] = lambda: (f"{foreign.pid}, {GPU0}\n" if os.path.exists(box.path(g3["log"]))
                                             and not os.path.exists(box.path(g3["record"])) else "")
        assert box.sup().run() == 3
    finally:
        foreign.kill()
        foreign.wait()
    st = box.state()
    doc = json.load(open(box.path(g3["record"])))
    assert doc["exit_code"] == 0 and not doc["admitted"] and str(foreign.pid) in " ".join(doc["refusals"])
    assert "stand-in fa_production_parity" in open(box.path(g3["log"])).read()      # it ran to its own end
    assert os.path.exists(box.path(g3["outputs"][0])) and st["lines"][-1]["name"] == "G3"
    assert not os.path.exists(box.path(box.lines[NAMES.index("V_G3")]["log"]))        # no later line
    assert any("LATCH" in e["what"] for e in st["events"])


def _after(box, name, action):
    """Run ``action`` once, inside the first nvidia-smi query after line ``name``'s record exists."""
    done, ln = [], box.lines[NAMES.index(name)]

    def apps():
        if not done and os.path.exists(box.path(ln["record"])):
            done.append(action())
        return ""
    box.world.answers["apps"] = apps


@pytest.mark.parametrize("change, trigger, before", [
    ("replaced-reservation", "GV", "V_GV"), ("lock-held", "GV", "V_GV"), ("head-moved", "GV", "G1"),
    ("dirty-tree", "GV", "G1"), ("clobber", "GV", "G1"), ("earlier-output", "V_GV", "V_G1")])
def test_a_change_between_lines_stops_before_the_next_one(box, change, trigger, before):
    holder = []

    def act():
        if change == "replaced-reservation":
            raw = open(box.res, "rb").read()
            open(box.res + ".new", "wb").write(raw)
            os.replace(box.res + ".new", box.res)
        elif change == "head-moved":
            subprocess.run(["git", "-C", box.w, "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-q",
                            "--allow-empty", "-m", "moved"], check=True)
        elif change == "dirty-tree":
            open(box.path("localize_meshgrid.py"), "a").write("\n# edited\n")
        elif change == "earlier-output":
            open(box.path(box.lines[1]["outputs"][0]), "a").write(" ")
        elif change == "clobber":
            open(box.path(box.lines[3]["outputs"][0]), "w").write("{}")
        else:
            from conftest import HOLDER
            p = subprocess.Popen([sys.executable, "-c", HOLDER, box.cfg.exp35_lock], stdout=subprocess.PIPE, text=True)
            assert p.stdout.readline().strip() == "locked"
            holder.append(p)
        return True
    _after(box, trigger, act)
    try:
        assert box.sup().run() == 3
    finally:
        for p in holder:
            p.kill()
            p.wait()
    st = box.state()
    assert st["phase"] == "stopped" and f"before line {NAMES.index(before) + 1} {before}" in st["stop"]["reason"]
    assert not os.path.exists(box.path(box.lines[NAMES.index(before)]["log"]))
    assert os.path.lexists(box.res)                                           # kept (or not ours: never unlinked)


@pytest.mark.parametrize("pre", ["gates-dir", "first-log", "junit"])
def test_nothing_is_clobbered(box, pre):
    p = {"gates-dir": f"outputs_loc/exp22/r2_gates_{RUN}", "first-log": box.lines[0]["log"],
         "junit": box.lines[0]["junit"]}[pre]
    os.mkdir(box.path(p)) if pre == "gates-dir" else open(box.path(p), "w").write("prior")
    if pre == "gates-dir":
        assert box.sup().run() == 3 and box.state()["lines"][-1]["name"] == "T"     # stopped before GV
    else:
        assert box.sup().run() == 3 and box.state()["lines"] == []                  # stopped before T
        assert open(box.path(p)).read() == "prior"


@pytest.mark.parametrize("point", ["spawn", "identify", "watch", "evidence", "record", "release", "after-unlink"])
def test_a_crash_at_any_transition_leaves_everything_for_the_planner(box, monkeypatch, point):
    """A7 r3 (e): no crash path releases or adopts; the restart refuses; a surviving child runs to its end."""
    targets = {"spawn": (box.world, "spawn"), "identify": (S.Supervisor, "identify"), "watch": (S.Supervisor, "watch"),
               "evidence": (S.Supervisor, "evidence")}
    if point in targets:
        obj, name = targets[point]
        monkeypatch.setattr(obj, name, lambda *a, **k: (_ for _ in ()).throw(Crash()))
    elif point == "record":
        real = S.E.write_exclusive
        monkeypatch.setattr(S.E, "write_exclusive", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(Crash()))[0])
    else:
        real = S.release_reservation
        monkeypatch.setattr(S, "release_reservation", (lambda h: (_ for _ in ()).throw(Crash())) if point == "release"
                            else (lambda h: (real(h), (_ for _ in ()).throw(Crash()))[0]))
    with pytest.raises(Crash):
        box.sup().run()
    monkeypatch.undo()
    st = box.state()
    child = (st["current"] or {}).get("child")
    deadline = time.time() + 60
    while child and os.path.exists(f"/proc/{child['pid']}") and time.time() < deadline:   # the child is never signalled
        time.sleep(0.05)
    if point == "spawn":
        assert st["phase"] == "line_pending" and st["current"]["name"] == "T" and "child" not in st["current"]
    elif point == "identify":
        assert st["phase"] == "line_pending" and os.path.exists(box.path(box.lines[0]["log"]))
    elif point == "watch":
        assert st["phase"] == "child_identified" and child["cmdline_sha256"] == child["expected_cmdline_sha256"]
    elif point in ("evidence", "record"):
        assert st["phase"] == "line_complete" and st["current"]["rc"] == 0 and st["lines"] == []
        assert os.path.exists(box.path(box.lines[0]["record"])) == (point == "record")
    else:
        assert st["phase"] == "line_complete" and len(st["lines"]) == 15 and st["released"] is None
    assert os.path.lexists(box.res) == (point != "after-unlink")
    before = open(os.path.join(box.orch, "state.json"), "rb").read()
    assert box.sup().run() == 3 and open(os.path.join(box.orch, "state.json"), "rb").read() == before
    assert any(box.orch in x for x in S.active_reasons(box.cfg, box.w))      # unresolved until the Planner acts


def test_a_restart_refuses_after_a_complete_run(box):
    assert box.sup().run() == 0
    before = open(os.path.join(box.orch, "state.json"), "rb").read()
    assert box.sup().run() == 3 and open(os.path.join(box.orch, "state.json"), "rb").read() == before


def test_the_supervisor_refuses_without_the_inherited_lock(box, tmp_path):
    other = os.open(str(tmp_path / "not-the-lock"), os.O_RDWR | os.O_CREAT, 0o644)
    second = os.open(S.lock_path(box.w), os.O_RDWR)                         # another open file description
    try:
        for fd in (other, second):
            assert S.Supervisor(box.world, box.cfg, box.orch, fd, worktree=box.w, child_poll_s=0.05).run() == 3
    finally:
        os.close(other)
        os.close(second)
    assert box.state()["phase"] == "reserved"


def test_the_heartbeat_beats_while_a_line_runs(box, monkeypatch):
    box.control({"vanilla_cache_parity": {"sleep": 1.0}})
    beats, real = [], S.Supervisor.heartbeat
    monkeypatch.setattr(S.Supervisor, "heartbeat", lambda self: (beats.append(time.time()), real(self))[1])
    assert box.sup().run() == 0
    assert len(beats) > 15 and max(b - a for a, b in zip(beats, beats[1:])) < 1.0
    assert C.CHILD_POLL_S <= 60                                              # the registered heartbeat bound


# --------------------------------------------------------------------------- #
# process level: the CLI in the sandbox copy; a real kill with a surviving child
# --------------------------------------------------------------------------- #
@pytest.fixture
def cli(tmp_path):
    cfg = cfg_for(tmp_path)
    nv = tmp_path / "nvidia-smi"
    (tmp_path / "apps.txt").write_text("")
    nv.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '%s' '{MAP}';; *) cat {tmp_path / 'apps.txt'};; esac\n")
    nv.chmod(0o755)
    w, pin2 = sandbox_repo(tmp_path, common=dict(
        RESERVATIONS=cfg.reservations, EXP35_LOCK=cfg.exp35_lock, QUEUE_LOG=cfg.queue_log, PROC="/proc",
        NVIDIA_SMI=str(nv), POLL_S=60, WINDOW_POLLS=31, POLL_SLACK_S=15, FRESH_S=600, PY=sys.executable,
        PYTEST_MAIN=C.PYTEST_MAIN, CHILD_POLL_S=0.05))
    sup = os.path.join(w, "worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude", "tools", "exp22_r2c_supervisor.py")
    orch = os.path.join(w, "outputs_loc", "exp22", f"r2_preflight_{RUN}")
    os.mkdir(orch)
    seal = os.path.join(orch, "seal.txt")
    open(seal, "w").write(f"pin2 {pin2}\nrun {RUN}\ncard GPU0 {GPU0} 0\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    run = lambda *a, **kw: subprocess.run([sys.executable, "-B", sup, *a], capture_output=True, text=True, env=env,
                                          timeout=300, **kw)
    p = run("reserve", "--orch", orch, "--pin2", pin2, "--card", "GPU0", "--index", "0", "--seal", seal,
            "--seal-sha256", E.sha256_bytes(open(seal, "rb").read()))
    assert p.returncode == 0 and p.stdout.startswith("RESERVED"), p.stdout + p.stderr
    yield types.SimpleNamespace(w=w, cfg=cfg, sup=sup, orch=orch, run=run, env=env,
                                res=os.path.join(cfg.reservations, f"{GPU0}.json"),
                                state=lambda: json.load(open(os.path.join(orch, "state.json"))))


def _locked_fd(w):
    fd = os.open(S.lock_path(w), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_the_cli_runs_the_lines_to_the_release(cli):
    fd = _locked_fd(cli.w)
    try:
        p = cli.run("run", "--orch", cli.orch, "--lock-fd", str(fd), pass_fds=(fd,))
    finally:
        os.close(fd)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert cli.state()["phase"] == "released" and not os.path.lexists(cli.res)
    p = cli.run("active")
    assert p.returncode == 0, p.stdout


def test_a_killed_supervisor_leaves_a_surviving_child_that_keeps_the_lock_and_the_reservation(cli):
    open(os.path.join(cli.w, "a7_control.json"), "w").write(json.dumps({"fa_production_parity": {"sleep": 4}}))
    fd = _locked_fd(cli.w)
    sup = subprocess.Popen([sys.executable, "-B", cli.sup, "run", "--orch", cli.orch, "--lock-fd", str(fd)],
                           env=cli.env, pass_fds=(fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(fd)                                                       # from here only the supervisor holds it
    deadline = time.time() + 120
    while time.time() < deadline:
        st = cli.state()
        if st["phase"] == "child_identified" and st["current"]["name"] == "G3":
            break
        time.sleep(0.05)
    child = st["current"]["child"]
    sup.send_signal(signal.SIGKILL)                                    # OUR test supervisor, never anyone else's
    sup.wait()
    assert os.path.exists(f"/proc/{child['pid']}")                     # the line child survives, untouched
    assert C.lock_status(S.lock_path(cli.w)) == "held"                 # ... and keeps the exp22 lock held
    p = cli.run("active")
    assert p.returncode == 1 and "lock" in p.stdout
    p = cli.run("release", "--orch", cli.orch)
    assert p.returncode == 1 and os.path.exists(cli.res)               # the Planner cannot release yet
    deadline = time.time() + 60
    while os.path.exists(f"/proc/{child['pid']}") and time.time() < deadline:
        time.sleep(0.05)
    assert C.lock_status(S.lock_path(cli.w)) == "unheld"
    assert cli.state()["phase"] == "child_identified"                  # no crash path adopts or releases
    p = cli.run("release", "--orch", cli.orch)
    assert p.returncode == 0 and not os.path.exists(cli.res), p.stdout
    assert cli.state()["phase"] == "released" and cli.state()["released"]["by"] == "planner"
