"""exp_38 — the training supervisor's process ownership (exp38_supervise.py; Codex code r1 HIGH 4, 7, 11).

Synthetic children only (no GPU, no train.py): a child's exit status passes through; TERM to the
supervisor stops OUR process group (grandchildren included) within the bounded grace and leaves a
finalized record; a TERM-ignoring child is SIGKILLed after the grace; a failed / timed-out / pre-empted
start gate terminates the group (exit 4); the card lock passed down with ``pass_fds`` outlives a
SIGKILLed supervisor (so a card can never be released under a running trainer) and frees when the
child ends; a reaped child is never signalled again; the GPU / allocation gates and the child
environment are exact.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
sys.path.insert(0, EXP)
import exp38_supervise as sv  # noqa: E402

PY = sys.executable
SLEEPER = textwrap.dedent("""
    import subprocess, sys, time
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    open(sys.argv[1], "w").write(str(p.pid))
    time.sleep(120)
""")
IGNORER = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(120)"


def _harness(tmp_path, cmd, start="none", grace=30, lock=False, start_timeout=60):
    rec = tmp_path / "record.json"
    code = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {EXP!r})
        import exp38_supervise as s
        s.install_signal_handlers()
        rec = s.Record([{str(rec)!r}], status="running")
        fds = s.lock_cards([0], {str(tmp_path)!r}) if {lock!r} else []
        child = s.OwnedChild({cmd!r}, {str(tmp_path)!r}, dict(os.environ), {str(tmp_path / 'child.log')!r}, pass_fds=fds)
        s.start_lock_holder(child.pgid, fds)
        rec.update(pid=child.pid)
        print("CHILD", child.pid, flush=True)
        checks = {{"none": None, "fail": lambda: ["bad first step"], "never": lambda: None}}
        sys.exit(s.supervise(child, rec, start_check=checks[{start!r}], start_timeout={start_timeout},
                             grace={grace}, poll=0.1))
    """)
    p = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("CHILD"), (line, p.stderr.read() if p.poll() is not None else "")
    return p, int(line.split()[1]), rec


def _gone(pid, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            st = open(f"/proc/{pid}/stat").read()
        except OSError:
            return True
        if st[st.rindex(")") + 2] == "Z":           # a zombie awaiting its (dead) parent's init reap
            return True
        time.sleep(0.05)
    return False


def _wait_file(path, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path) and open(path).read().strip():
            return open(path).read().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_the_child_exit_status_passes_through(tmp_path):
    p, pid, rec = _harness(tmp_path, [PY, "-c", "import sys; sys.exit(3)"])
    assert p.wait(timeout=60) == 3
    assert json.load(open(rec))["child_rc"] == 3


def test_a_signal_killed_child_maps_to_128_plus_n(tmp_path):
    p, pid, rec = _harness(tmp_path, [PY, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"])
    assert p.wait(timeout=60) == 128 + 9


def test_sigterm_stops_our_group_including_grandchildren(tmp_path):
    gc_file = tmp_path / "grandchild.pid"
    p, pid, rec = _harness(tmp_path, [PY, "-c", SLEEPER, str(gc_file)])
    gc = int(_wait_file(gc_file))
    assert os.getpgid(gc) == pid                     # the grandchild is in OUR group
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 128 + signal.SIGTERM
    assert _gone(pid) and _gone(gc)
    r = json.load(open(rec))
    assert r["status"].startswith("terminated by signal 15") and "finished_at" in r


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP])
def test_int_and_hup_are_forwarded_too(tmp_path, sig):
    p, pid, rec = _harness(tmp_path, [PY, "-c", "import time; time.sleep(120)"])
    time.sleep(0.3)
    p.send_signal(sig)
    assert p.wait(timeout=60) == 128 + sig and _gone(pid)


def test_a_term_ignoring_child_is_killed_after_the_bounded_grace(tmp_path):
    p, pid, rec = _harness(tmp_path, [PY, "-c", IGNORER], grace=1)
    _wait_file(tmp_path / "child.log")
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=60) == 128 + signal.SIGTERM
    assert time.time() - t0 < 20 and _gone(pid)
    assert json.load(open(rec))["child_rc"] == -signal.SIGKILL


@pytest.mark.parametrize("start,cmd,why", [
    ("fail", [PY, "-c", "import time; time.sleep(120)"], "bad first step"),
    ("never", [PY, "-c", "import time; time.sleep(120)"], "no start gate result within 1 s"),
    ("never", [PY, "-c", "print('bye')"], "exited before the start gate"),
])
def test_a_failed_start_gate_terminates_our_group(tmp_path, start, cmd, why):
    p, pid, rec = _harness(tmp_path, cmd, start=start, start_timeout=1)
    assert p.wait(timeout=60) == sv.EXIT_START and _gone(pid)
    r = json.load(open(rec))
    assert any(why in x for x in r["start_gate"]) and "gate failed" in r["status"]


def test_the_card_lock_outlives_a_killed_supervisor(tmp_path):
    p, pid, rec = _harness(tmp_path, [PY, "-c", "import time; time.sleep(4)"], lock=True)
    p.send_signal(signal.SIGKILL)                    # no handler can run: the child keeps the lock
    p.wait(timeout=30)
    fd = os.open(tmp_path / ".exp38_card0.lock", os.O_WRONLY)
    try:
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _gone(pid, timeout=30)
        time.sleep(0.2)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # the child ended: the card is free
    finally:
        os.close(fd)


def test_a_second_lock_holder_is_refused(tmp_path):
    fds = sv.lock_cards([0, 1], str(tmp_path))
    try:
        with pytest.raises(sv.Refused):
            sv.lock_cards([1], str(tmp_path))
    finally:
        for f in fds:
            os.close(f)
    for f in sv.lock_cards([1], str(tmp_path)):
        os.close(f)


def test_a_reaped_child_is_never_signalled(tmp_path, monkeypatch):
    c = sv.OwnedChild([PY, "-c", "pass"], str(tmp_path), dict(os.environ), str(tmp_path / "x.log"))
    assert c.pgid == c.pid and c.start == sv.proc_start_time(c.pid)
    while not c.exited():
        time.sleep(0.05)
    assert c.owned()                                 # exited but unreaped: still ours (a zombie)
    assert c.reap() == 0 and not c.owned()
    sent = []
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(a))
    c.signal_group(signal.SIGTERM)
    assert sent == [] and sv.proc_start_time(2 ** 22 + 12345) is None


def test_the_record_is_rewritten_atomically(tmp_path):
    r = sv.Record([str(tmp_path / "a.json"), str(tmp_path / "b.json")], x=1)
    r.update(y=2)
    for n in ("a.json", "b.json"):
        assert json.load(open(tmp_path / n)) == {"x": 1, "y": 2}
    assert sorted(os.listdir(tmp_path)) == ["a.json", "b.json"]


# ------------------------------------------------------------------ CLI and gates
@pytest.mark.parametrize("argv", [
    ["--platform", "a6000", "--mode", "FULL", "--run", "de_yaw_f025", "--dry-save-root", "/x"],
    ["--platform", "a6000", "--mode", "SMOKE", "--run", "de_yaw_f025", "--lockdir", "/x"],
    ["--platform", "neuronic", "--mode", "SMOKE_DDP", "--run", "de_yaw_f025"],
    ["--platform", "neuronic", "--mode", "FULL", "--run", "de_yaw_f025", "--save-dir", "/x"],
    ["--platform", "neuronic", "--mode", "FULL", "--run", "de_yaw_f025", "--token", "xyz", "--save-dir", "/x"],
    ["--platform", "a6000", "--mode", "SMOKE", "--run", "de_yaw_f025", "--smoke-steps", "500"],
])
def test_cli_refusals(argv):
    with pytest.raises(SystemExit):
        sv.parse(argv)


def test_unregistered_runs_are_refused_before_any_work(capsys):
    assert sv.main(["--platform", "a6000", "--mode", "DRY_RUN", "--run", "de_p1_f050"]) == sv.EXIT_GATE
    assert "not a registered exp_38 run id" in capsys.readouterr().err


ROWS = [["0", "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "48000", "NVIDIA RTX A6000"],
        ["1", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da", "48000", "NVIDIA RTX A6000"]]


def test_a6000_gpu_gate():
    assert sv.a6000_gpu_gate([0, 1], "idle", 21900, ROWS, []) == sorted([r[1], r[3]] for r in ROWS)
    bad = [
        ([ROWS[0]], []),                                                     # one GPU visible
        ([ROWS[0], [*ROWS[1][:3], "NVIDIA L40"]], []),                       # not an A6000
        ([ROWS[0], ["1", "GPU-ffff", "48000", "NVIDIA RTX A6000"]], []),     # not this box's card
        (ROWS, [[ROWS[1][1], "1234", "900"]]),                               # foreign process, idle policy
        ([ROWS[0], [*ROWS[1][:2], "1000", ROWS[1][3]]], []),                 # too little free memory
    ]
    for rows, apps in bad:
        with pytest.raises(sv.Refused):
            sv.a6000_gpu_gate([0, 1], "idle", 21900, rows, apps)
    assert sv.a6000_gpu_gate([0, 1], "shared", 21900, ROWS, [[ROWS[1][1], "1234", "900"]])


def test_neuronic_allocation_gate_refusals(monkeypatch):
    for k in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(sv.Refused):
        sv.neuronic_allocation_gate(2, dict(os.environ))                     # not in an allocation
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    for cvd in ("0", "0,0", "0,1,2", ""):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
        with pytest.raises(sv.Refused):
            sv.neuronic_allocation_gate(2, dict(os.environ))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    with pytest.raises(sv.Refused):
        sv.neuronic_allocation_gate(2, dict(os.environ))
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    with pytest.raises(sv.Refused):
        sv.neuronic_allocation_gate(2, dict(os.environ))
    monkeypatch.setenv("SLURM_NTASKS", "1")
    with pytest.raises(sv.Refused):                                          # CPU-only here: torch sees none
        sv.neuronic_allocation_gate(2, {**os.environ, "CUDA_VISIBLE_DEVICES": ""})


LAUNCHER_ENV = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4", "GROUP_RANK": "0", "LOCAL_WORLD_SIZE": "2",
                "MASTER_ADDR": "x", "MASTER_PORT": "1", "NODE_RANK": "1", "TORCHELASTIC_RUN_ID": "r",
                "PL_TRAINER_DEVICES": "1", "PL_GLOBAL_SEED": "7", "LIGHTNING_X": "1"}


def test_child_env_is_exact(monkeypatch, tmp_path):
    """Codex code r2 BLOCKING 4: only the chosen launcher creates rank variables."""
    monkeypatch.setenv("PYTHONPATH", "/nowhere")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    for k, v in LAUNCHER_ENV.items():
        monkeypatch.setenv(k, v)
    e = sv.child_env("neuronic", "flash", str(tmp_path))
    assert "PYTHONPATH" not in e and "SLURM_PROCID" not in e and e["SLURM_JOB_ID"] == "7"
    assert not set(LAUNCHER_ENV) & set(e)
    assert (e["HF_HUB_OFFLINE"], e["PYTHONNOUSERSITE"], e["PYTHONUNBUFFERED"]) == ("1", "1", "1")
    e = sv.child_env("a6000", "sdpa", str(tmp_path), "0,1")
    assert e["PYTHONPATH"] == str(tmp_path / "overlays" / "no_flash") and e["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert e["PYTHONDONTWRITEBYTECODE"] == "1" and not any(k.startswith("SLURM_") for k in e)
    assert not set(LAUNCHER_ENV) & set(e)


@pytest.mark.parametrize("var", sorted(LAUNCHER_ENV) + ["SLURM_PROCID"])
def test_inherited_launcher_variables_refuse_the_launch(monkeypatch, var):
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv(var, "1")
    with pytest.raises(sv.Refused, match="launcher variables"):
        sv.interpreter_gate("a6000")
    if not var.startswith("SLURM_"):
        with pytest.raises(sv.Refused, match="launcher variables"):
            sv.interpreter_gate("neuronic", sys.executable)
    else:
        sv.interpreter_gate("neuronic", sys.executable)                        # expected inside a job


def test_the_lock_holder_outlives_a_rank_that_did_not_inherit_the_lock(tmp_path):
    """Codex code r2 HIGH 1: Lightning's launcher starts rank 1 with a plain Popen (close_fds=True),
    so rank 1 does not inherit the lock fd. Here the direct child spawns such a grandchild and exits;
    the supervisor is SIGKILLed: the card stays locked until the grandchild is gone."""
    gc_file = tmp_path / "rank1.pid"
    code = ("import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(6)']); "
            f"open({str(gc_file)!r}, 'w').write(str(p.pid)); time.sleep(0.5)")
    p, pid, rec = _harness(tmp_path, [PY, "-c", code], lock=True)
    gc = int(_wait_file(gc_file))
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=30)
    assert _gone(pid, timeout=20)                     # rank 0 (the direct child) has exited
    fd = os.open(tmp_path / ".exp38_card0.lock", os.O_WRONLY)
    try:
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                     # rank 1 still runs
        assert _gone(gc, timeout=30)
        end = time.time() + 15
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                 # freed within the holder's poll
                break
            except OSError:
                assert time.time() < end, "the lock holder never released the card"
                time.sleep(0.5)
    finally:
        os.close(fd)


def test_the_authoritative_record_must_be_written(tmp_path):
    """Codex code r2 HIGH 8: the first copy is authoritative at creation and at terminal transitions."""
    good, bad = str(tmp_path / "a.json"), str(tmp_path / "missing_dir" / "b.json")
    r = sv.Record([good, bad], x=1)                                            # a secondary failure is tolerated
    r.update(strict=True, y=2)
    assert json.load(open(good)) == {"x": 1, "y": 2}
    with pytest.raises(OSError):
        sv.Record([bad, good], x=1)                                            # no authoritative copy: no launch
    d = tmp_path / "ro"
    d.mkdir()
    r2 = sv.Record([str(d / "c.json")], x=1)
    os.chmod(d, 0o500)
    try:
        r2.update(y=3)                                                         # non-terminal: reported, not fatal
        with pytest.raises(OSError):
            r2.update(strict=True, status="finished")                          # terminal: fatal
    finally:
        os.chmod(d, 0o700)


def test_backend_gate_resolves_like_the_child():
    env = sv.child_env("a6000", "sdpa", REPO, "")
    got = sv.backend_gate("sdpa", REPO, env)
    assert got["flash_attn"] == os.path.realpath(os.path.join(REPO, "overlays/no_flash/flash_attn.py"))
    with pytest.raises(sv.Refused):
        sv.backend_gate("flash", REPO, env)                                  # the overlay shadows flash
    env = sv.child_env("a6000", "flash", REPO, "")
    try:
        import importlib.util
        real = importlib.util.find_spec("flash_attn") is not None
    except ValueError:
        real = False
    if real:
        assert sv.backend_gate("flash", REPO, env)["attention_backend"] == "flash"
        with pytest.raises(sv.Refused):
            sv.backend_gate("sdpa", REPO, env)                               # a real flash_attn is reachable


# ------------------------------------------------------------------ the FULL lifecycle with a fake trainer
FAKE_TRAIN = textwrap.dedent("""
    import json, os, sys, time
    mode = {mode!r}
    print({log!r}, flush=True)
    if mode == "badstart":
        sys.exit(0)
    for name, rec in {evidence!r}.items():
        os.makedirs({evdir!r}, exist_ok=True)
        json.dump(rec, open(os.path.join({evdir!r}, name), "w"))
    time.sleep(0.5)
    print({end!r}, flush=True)
    sys.exit(1 if mode == "crash" else 0)
""")


def _campaign(tmp_path, monkeypatch, mode="ok", audit_ok=True, smode="FULL", e0_head="f" * 40):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_exp38_receipts as tr
    reg = sv.reg
    run = "de_bf_f025"
    monkeypatch.setattr(reg, "REPO", str(tmp_path))
    (tmp_path / reg.EXP_DIR).mkdir(parents=True)
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(tmp_path / "nas"))
    (tmp_path / "nas").mkdir()
    monkeypatch.setitem(sv.SNAPSHOT_ROOT, "a6000", str(tmp_path / "snaps"))
    gpus = sorted([u, "NVIDIA RTX A6000"] for u in reg.A6000_UUIDS)
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("bf", "025", "2x32", PY, {"argv": []}, {}))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: ("f" * 40, "d" * 64, "c" * 64))
    monkeypatch.setattr(sv, "lock_cards", lambda cards, lockdir: [])
    monkeypatch.setattr(sv, "a6000_gpu_gate", lambda *x: gpus)
    monkeypatch.setattr(sv, "backend_gate", lambda *x: {})
    monkeypatch.setattr(reg, "check_e0_receipt", lambda p: {"sha256": "e" * 64, "head": e0_head})
    monkeypatch.setattr(reg, "check_smoke_receipt", lambda *x, **k: {"smoke_receipt": "s.json", "sha256": "6" * 64})
    monkeypatch.setattr(reg, "storage_gate", lambda *x, **k: {"fstype": "cifs"})
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": "c" * 64})
    monkeypatch.setattr(reg, "_gpu_rows", lambda: [["0", reg.A6000_UUIDS[0], "NVIDIA RTX A6000", "570.86"]])
    import exp38_content_manifest as cm
    content_calls = []
    monkeypatch.setattr(cm, "verify", lambda *x, **k: content_calls.append(k.get("want_sha")) or [])
    retained_calls = []
    monkeypatch.setattr(reg, "retain_receipts", lambda rd_, paths: retained_calls.append(sorted(paths)) or
                        {k: os.path.basename(v) for k, v in paths.items()})
    rd = tmp_path / "nas" / run
    ev = {f"run_evidence.json.rank{r}.json": e for r, e in tr.records(run, max_steps=40000).items()}
    log = tr.smoke_log(run, steps=40000, losses=2, end=False)
    if mode == "badstart":
        log = log.replace("MEMBER: 2/2", "MEMBER: 1/2")

    def fake_snapshot(run_, dest, repo=None):
        os.makedirs(dest)
        open(os.path.join(dest, "train.py"), "w").write(FAKE_TRAIN.format(
            mode=mode, log=log, evidence=ev, evdir=str(rd / "evidence"), end=reg.FIT_END.format(n=40000)))
        return {"snapshot": dest, "files": 1, "snapshot_sha256": "5" * 64}
    monkeypatch.setattr(reg, "snapshot_tools", fake_snapshot)
    monkeypatch.setattr(reg, "verify_snapshot", lambda dest, want=None: {"verified": True})
    monkeypatch.setattr(reg, "build_train_argv", lambda *x, **k: ["train.py"])
    monkeypatch.setattr(reg, "build_smoke_argv", lambda *x, **k: ["train.py"])
    calls = []

    import inspect
    real_sig = inspect.signature(reg.write_training_receipt)

    def fake_receipt(*args, **kw):
        real_sig.bind(*args, **kw)                               # the supervisor calls the real signature
        assert args[12] == "d" * 64 and kw == {"repo": kw["repo"], "tree": str(tmp_path)}
        assert sorted(args[13]) == ["e0", "env", "smoke"] and args[14] == reg.AR_CONTENT_SHA256
        assert content_calls == [reg.AR_CONTENT_SHA256] and retained_calls == [["e0", "env", "smoke"]]
        calls.append(args)
        if not audit_ok:
            raise reg.GateError("end gates: synthetic failure")
        return {"final_ckpt_sha256": "9" * 64}
    monkeypatch.setattr(reg, "write_training_receipt", fake_receipt)
    smoke_sig = inspect.signature(reg.write_smoke_receipt)

    def fake_smoke(*args, **kw):
        smoke_sig.bind(*args, **kw)
        assert args[11] == "d" * 64 and kw["out_repo"] == str(tmp_path) and kw["snapshot_dir"] == kw["repo"]
        calls.append(args)
        return {"verdict": "PASS" if audit_ok else "FAIL"}
    monkeypatch.setattr(reg, "write_smoke_receipt", fake_smoke)
    a = sv.parse(["--platform", "a6000", "--mode", smode, "--run", run, "--expect-sha", "f" * 40, "--grace", "2"])
    code = sv.mode_campaign(a, "a6000")
    copies = [f for f in os.listdir(tmp_path / reg.EXP_DIR) if f.startswith("at_launch_")]
    assert len(copies) == 1
    rec = json.load(open(tmp_path / reg.EXP_DIR / copies[0]))
    rd = __import__("pathlib").Path(rec["run_dir"])
    assert json.load(open(rd / copies[0])) == rec               # the run-dir copy is identical
    return code, rec, rd, calls


def test_a_full_run_launches_from_the_snapshot_gates_and_publishes(tmp_path, monkeypatch):
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch)
    assert code == 0 and rec["status"] == "COMPLETE" and rec["start_gate"] == "PASS" and rec["evidence_gate"] == "PASS"
    assert rec["cwd"].startswith(str(tmp_path / "snaps")) and rec["child_rc"] == 0 and rec["supervisor_pid"] == os.getpid()
    assert len(calls) == 1 and not (rd / ".exp38_lock").exists()
    assert [n for n in os.listdir(rd) if n.startswith(".exp38_lock.released_")]


@pytest.mark.parametrize("mode,audit_ok,want", [("crash", True, 1), ("badstart", True, sv.EXIT_START),
                                                ("ok", False, sv.EXIT_END)])
def test_a_failed_full_run_keeps_its_crash_marker(tmp_path, monkeypatch, mode, audit_ok, want):
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, mode, audit_ok)
    assert code == want and (rd / ".exp38_lock").is_dir() and "finished_at" in rec
    assert rec["status"] != "COMPLETE" and len(calls) == (1 if mode == "ok" else 0)
    with pytest.raises(sv.reg.GateError):
        sv.reg.check_run_dir(str(rd), "de_bf_f025", 0)          # never relaunched into: archive first


@pytest.mark.parametrize("audit_ok,want", [(True, 0), (False, sv.EXIT_END)])
def test_a_smoke_ddp_seals_its_receipt_from_the_snapshot(tmp_path, monkeypatch, audit_ok, want):
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, audit_ok=audit_ok, smode="SMOKE_DDP")
    assert code == want and len(calls) == 1 and str(rd).startswith(str(tmp_path / "outputs_FLAC" / "exp38_smoke"))
    assert "start_gate" not in rec and rec["mode"] == "SMOKE_DDP"


def test_a_full_run_refuses_an_e0_receipt_of_another_head(tmp_path, monkeypatch):
    with pytest.raises(sv.Refused, match="E0 campaign"):
        _campaign(tmp_path, monkeypatch, e0_head="a" * 40)


def test_a_full_run_refuses_unverified_ar_content(tmp_path, monkeypatch):
    import exp38_content_manifest as cm
    orig = cm.verify
    try:
        monkeypatch.setattr(cm, "verify", lambda *x, **k: ["3 group(s) differ from the content manifest"])
        monkeypatch.setattr(sv.reg, "AR_CONTENT_SHA256", sv.reg.AR_CONTENT_SHA256)
        with pytest.raises(sv.Refused, match="AR content"):
            _campaign_no_content_patch(tmp_path, monkeypatch)
    finally:
        cm.verify = orig


def _campaign_no_content_patch(tmp_path, monkeypatch):
    import exp38_content_manifest as cm
    bad = cm.verify
    real_setattr = monkeypatch.setattr

    def keep_verify(obj, name, value, *a, **k):                # let _campaign patch everything but cm.verify
        if obj is cm and name == "verify":
            return real_setattr(obj, name, bad)
        return real_setattr(obj, name, value, *a, **k)
    monkeypatch.setattr = keep_verify
    try:
        return _campaign(tmp_path, monkeypatch)
    finally:
        monkeypatch.setattr = real_setattr
