"""exp_46 (forked from test_exp38_supervise.py) — the training supervisor's process ownership (exp46_supervise.py; Codex code r1 HIGH 4, 7, 11).

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
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_supervise as sv  # noqa: E402

PY = sys.executable


@pytest.fixture(autouse=True)
def _exp45_ledger_stand_in(tmp_path, monkeypatch):
    """C3 (plan r3 §3.2 "Storage") reads exp_45's REAL ledger path at run time; here an absent stand-in (unless a test
    writes one), so the real exp_45 state never decides a test."""
    monkeypatch.setattr(sv.reg, "EXP45_LEDGER", str(tmp_path / "exp45_ledger_stand_in" / ".exp45_storage.json"))
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
        import exp46_supervise as s
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
def test_the_single_rank_smoke_is_refused_before_any_work(monkeypatch, capsys):
    """Codex code r1 finding 11: exp_46 registers no single-rank path (plan r3 §5 L5: 2 x L40, 2 x 32): the a6000
    one-rank SMOKE is refused explicitly, before any gate, card lock or directory."""
    touched = []
    monkeypatch.setattr(sv, "common_gates", lambda *a, **k: touched.append("gates") or ("c4f", "025", "2x32", PY, {}, {}))
    monkeypatch.setattr(sv, "lock_cards", lambda *a, **k: touched.append("locks") or [])
    a = sv.parse(["--platform", "a6000", "--mode", "SMOKE", "--run", "exp46_C4Ff025_l40_2x32"])
    assert sv._dispatch(a) == sv.Refused("x").code
    err = capsys.readouterr().err
    assert "single-rank SMOKE" in err and "2 x 32 only" in err and touched == []


@pytest.mark.parametrize("argv", [
    ["--platform", "a6000", "--mode", "FULL", "--run", "exp46_C4Ff025_l40_2x32", "--dry-save-root", "/x"],
    ["--platform", "a6000", "--mode", "SMOKE", "--run", "exp46_C4Ff025_l40_2x32", "--lockdir", "/x"],
    ["--platform", "neuronic", "--mode", "SMOKE_DDP", "--run", "exp46_C4Ff025_l40_2x32"],
    ["--platform", "neuronic", "--mode", "FULL", "--run", "exp46_C4Ff025_l40_2x32", "--save-dir", "/x"],
    ["--platform", "neuronic", "--mode", "FULL", "--run", "exp46_C4Ff025_l40_2x32", "--token", "xyz", "--save-dir", "/x"],
    ["--platform", "a6000", "--mode", "SMOKE", "--run", "exp46_C4Ff025_l40_2x32", "--smoke-steps", "500"],
])
def test_cli_refusals(argv):
    with pytest.raises(SystemExit):
        sv.parse(argv)


def test_unregistered_runs_are_refused_before_any_work(capsys):
    assert sv.main(["--platform", "a6000", "--mode", "DRY_RUN", "--run", "de_yaw_f025"]) == sv.EXIT_GATE   # exp_38's id
    assert "not a registered exp_46 run id" in capsys.readouterr().err


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
    for k in list(os.environ):                     # e.g. PL_GLOBAL_SEED left by an earlier seed_everything
        if k in sv.reg.LAUNCHER_VARS or k.startswith(sv.reg.LAUNCHER_PREFIXES) or k.startswith("SLURM_"):
            monkeypatch.delenv(k)
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
    for rel, data in {logger!r}.items():                    # what PL's CSV logger leaves in the run dir
        path = os.path.join(sys.argv[1], rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(data)
    time.sleep(0.5)
    print({end!r}, flush=True)
    sys.exit(1 if mode == "crash" else 0)
""")


DELTA_KW = []                          # the delta_verification each fake training-receipt call got (f100)


def _campaign(tmp_path, monkeypatch, mode="ok", audit_ok=True, smode="FULL", e0_head="f" * 40, content_after_ok=True,
              logger=None, reserve=True, run="exp46_C4Ff025_l40_2x32", after_receipt=None):
    """mode_campaign end to end with every gate and the child faked. ``after_receipt``: called right after the fake
    training receipt is written (fix round 1: a signal that arrives between the receipt and the commit marker)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_exp46_receipts as tr
    import exp46_fixtures as fx
    v0 = "lightning_logs/version_0/"
    if logger is None:                                          # PL's CSV logger: FULL 400 rows, SMOKE none
        logger = {v0 + "hparams.yaml": "{}\n"}
        if smode == "FULL":
            logger[v0 + "metrics.csv"] = fx.loss_csv(sv.reg).decode()
    reg = sv.reg
    DELTA_KW.clear()
    tag = reg.parse_run_id(run)[1]
    monkeypatch.setattr(reg, "REPO", str(tmp_path))
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")     # exercise the local lifecycle (D2 itself: below)
    (tmp_path / reg.EXP_DIR).mkdir(parents=True)
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(tmp_path / "nas"))
    (tmp_path / "nas").mkdir()
    monkeypatch.setitem(sv.SNAPSHOT_ROOT, "a6000", str(tmp_path / "snaps"))
    gpus = sorted([u, "NVIDIA RTX A6000"] for u in reg.A6000_UUIDS)
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("c4f", tag, "2x32", PY, {"argv": []}, {}))
    monkeypatch.setattr(reg, "check_f100_delta_pinned", lambda repo=None: {"pinned": "stub"})
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: ("f" * 40, "d" * 64, "c" * 64))
    monkeypatch.setattr(sv, "lock_cards", lambda cards, lockdir: [])
    monkeypatch.setattr(sv, "a6000_gpu_gate", lambda *x: gpus)
    monkeypatch.setattr(sv, "backend_gate", lambda *x: {})
    monkeypatch.setattr(reg, "check_e0_receipt", lambda p: {"sha256": "e" * 64, "head": e0_head})
    smoke_rec = tmp_path / "smoke_receipt.json"
    smoke_rec.write_text(json.dumps({"log": str(tmp_path / "smoke.log")}))
    monkeypatch.setattr(reg, "check_smoke_receipt", lambda *x, **k: {"smoke_receipt": str(smoke_rec), "sha256": "6" * 64})
    monkeypatch.setattr(reg, "storage_gate", lambda *x, **k: {"fstype": "cifs"})
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)                # fake room; the ledger is real
    if smode == "FULL" and reserve:                       # what the submitter did before sbatch (plan r3 §3.2)
        reg.reserve_storage(run, "a" * 32, str(tmp_path / "nas" / run), str(tmp_path / "nas"))
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": "c" * 64})
    monkeypatch.setattr(reg, "_gpu_rows", lambda: [["0", reg.A6000_UUIDS[0], "NVIDIA RTX A6000", "570.86"]])
    import exp38_content_manifest as cm
    content_calls = []
    monkeypatch.setattr(cm, "verify", lambda *x, **k: content_calls.append(k.get("want_sha")) or
                        ([] if (len(content_calls) == 1 or content_after_ok) else ["1 group(s) differ"]))
    retained_calls = []
    monkeypatch.setattr(reg, "retain_receipts", lambda rd_, paths, plain=None: retained_calls.append(sorted(paths)) or
                        {k: os.path.basename(v) for k, v in paths.items()})
    commits = []
    monkeypatch.setattr(reg, "write_commit_marker", lambda rd_, run_, sha_, path_: commits.append(
        (json.load(open(path_))["status"], json.load(open(path_))["training_receipt_sha256"] == sha_)))
    rd = tmp_path / "nas" / run
    ev = {f"run_evidence.json.rank{r}.json": e for r, e in tr.records(run, max_steps=40000).items()}
    log = tr.smoke_log(run, steps=40000, losses=2, end=False)
    if mode == "badstart":
        log = log.replace("MEMBER: 2/2", "MEMBER: 1/2")

    def fake_snapshot(run_, dest, repo=None):
        os.makedirs(dest)
        open(os.path.join(dest, "train.py"), "w").write(FAKE_TRAIN.format(
            mode=mode, log=log, evidence=ev, evdir=str(rd / "evidence"), end=reg.FIT_END.format(n=40000), logger=logger))
        return {"snapshot": dest, "files": 1, "snapshot_sha256": "5" * 64}
    monkeypatch.setattr(reg, "snapshot_tools", fake_snapshot)
    monkeypatch.setattr(reg, "verify_snapshot", lambda dest, want=None: {"verified": True})
    monkeypatch.setattr(reg, "build_train_argv", lambda *x, **k: ["train.py", str(rd)])
    monkeypatch.setattr(reg, "build_smoke_argv", lambda *x, **k: ["train.py", x[2]])      # x[2] = the smoke dir
    calls = []

    import inspect
    real_sig = inspect.signature(reg.write_training_receipt)

    def fake_receipt(*args, **kw):
        real_sig.bind(*args, **kw)                               # the supervisor calls the real signature
        DELTA_KW.append(kw.get("delta_verification"))
        assert args[12] == "d" * 64 and {k: v for k, v in kw.items() if k != "delta_verification"} == \
            {"repo": kw["repo"], "tree": str(tmp_path)}
        assert ("delta_verification" in kw) and (kw["delta_verification"] is None) is (tag != reg.FULL_TAG)
        assert sorted(args[13]) == ["e0", "env", "smoke"] and args[14] == reg.AR_CONTENT_SHA256
        assert content_calls == [reg.AR_CONTENT_SHA256] * 2 and retained_calls == [["e0", "env", "smoke"]]
        open(os.path.join(args[4], reg.TRAINING_RECEIPT), "w").write("{}")
        calls.append(args)
        if after_receipt is not None:
            after_receipt()
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
    platform = "neuronic" if smode == "SMOKE" else "a6000"
    argv = ["--platform", platform, "--mode", smode, "--run", run, "--expect-sha", "f" * 40, "--grace", "2",
            "--token", "a" * 32]
    if platform == "neuronic":          # exp_46: the SMOKE lifecycle runs the neuronic branch (no A6000 SMOKE path)
        import importlib.metadata
        import exp46_submit_neuronic as sub
        monkeypatch.setitem(reg.PLATFORMS["neuronic"], "save_root", str(tmp_path / "nas"))
        monkeypatch.setattr(importlib.metadata, "version", lambda name: reg.FLASH_VERSION)
        monkeypatch.setattr(sub, "claim_attempt", lambda *x: str(tmp_path / "owner"))
        monkeypatch.setattr(sub, "finish_attempt", lambda owner, code: None)
        monkeypatch.setattr(sv, "neuronic_allocation_gate", lambda ngpu, env: gpus)
        monkeypatch.setattr(sv, "launch_cmd", lambda platform_, ngpu, argv_: [PY, *argv_])
        argv += ["--save-dir", str(tmp_path / "nas" / f"{run}_smoke"), "--expect-flash-version", reg.FLASH_VERSION]
    a = sv.parse(argv)
    code = sv.mode_campaign(a, platform)
    copies = [f for f in os.listdir(tmp_path / reg.EXP_DIR) if f.startswith("at_launch_")]
    assert len(copies) == 1
    rec = json.load(open(tmp_path / reg.EXP_DIR / copies[0]))
    rd = __import__("pathlib").Path(rec["run_dir"])
    assert json.load(open(rd / copies[0])) == rec               # the run-dir copy is identical
    calls.append(("commits", commits))
    return code, rec, rd, calls


def test_a_full_run_launches_from_the_snapshot_gates_and_publishes(tmp_path, monkeypatch):
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch)
    assert code == 0 and rec["status"] == "COMPLETE" and rec["start_gate"] == "PASS" and rec["evidence_gate"] == "PASS"
    assert rec["cwd"].startswith(str(tmp_path / "snaps")) and rec["child_rc"] == 0 and rec["supervisor_pid"] == os.getpid()
    assert len(calls) == 2 and calls[-1] == ("commits", [("COMPLETE", True)]) and not (rd / ".exp46_lock").exists()
    assert [n for n in os.listdir(rd) if n.startswith(".exp46_lock.released_")]
    # the FULL branch moved PL's CSV loss record into receipts/ before the end audit (plan r3 §3.2)
    assert not (rd / "lightning_logs").exists() and rec["loss_record"]["loss_record_rows"] == 400
    assert sorted(os.listdir(rd / "receipts")) == ["loss_record.csv", "loss_record_hparams.yaml"]
    assert (rd / "receipts" / "loss_record_hparams.yaml").read_text() == "{}\n"
    # sealed: the run's storage reservation is released (plan r3 §3.2) -- as the ledger's own entry, never written into
    # the sealed launch record (Codex code r1 finding 1)
    assert "storage_released" not in rec and rec["storage"]["reservation"]["reserved_bytes"] == sv.reg.RUN_RESERVE_BYTES
    led = json.load(open(tmp_path / "nas" / sv.reg.STORAGE_LEDGER))
    assert led["reservations"] == {} and [(r["run"], r["reason"]) for r in led["releases"]] == [
        ("exp46_C4Ff025_l40_2x32", "sealed")]


@pytest.mark.parametrize("mode,audit_ok,want", [("crash", True, 1), ("badstart", True, sv.EXIT_START),
                                                ("ok", False, sv.EXIT_END)])
def test_a_failed_full_run_keeps_its_crash_marker(tmp_path, monkeypatch, mode, audit_ok, want):
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, mode, audit_ok)
    assert code == want and (rd / ".exp46_lock").is_dir() and "finished_at" in rec
    assert rec["status"] != "COMPLETE" and len(calls) == (2 if mode == "ok" else 1) and calls[-1] == ("commits", [])
    with pytest.raises(sv.reg.GateError):
        sv.reg.check_run_dir(str(rd), "exp46_C4Ff025_l40_2x32", 0)          # never relaunched into: archive first
    # a crash keeps the storage hold (released only on seal / abandon)
    held = json.load(open(tmp_path / "nas" / sv.reg.STORAGE_LEDGER))["reservations"]
    assert list(held) == ["exp46_C4Ff025_l40_2x32"] and "storage_released" not in rec


def test_a_full_run_without_its_reservation_is_refused_before_any_child(tmp_path, monkeypatch):
    with pytest.raises(sv.reg.GateError, match="no storage reservation"):
        _campaign(tmp_path, monkeypatch, reserve=False)


def test_c3_the_full_start_refuses_before_any_child_once_exp45_holds_a_reservation(tmp_path, monkeypatch):
    """exp_46 plan r3 §3.2 "Storage" (C3): exp_45 reserving after exp_46's submitter did -- the supervisor's FULL-start
    re-check reads exp_45's ledger (read-only) and refuses before any child; exp_46's own hold stays, exp_45's ledger
    is untouched."""
    ledger = tmp_path / "exp45_root" / ".exp45_storage.json"
    ledger.parent.mkdir()
    monkeypatch.setattr(sv.reg, "EXP45_LEDGER", str(ledger))
    held = json.dumps({"schema": 1, "reservations": {"exp45_RFf025_l40_2x32": {"token": "b" * 32}}, "releases": []})
    real = sv.reg.reserve_storage

    def reserve_then_exp45_reserves(*a, **k):                  # the submitter's step, then exp_45 reserves
        out = real(*a, **k)
        ledger.write_text(held)
        return out
    monkeypatch.setattr(sv.reg, "reserve_storage", reserve_then_exp45_reserves)
    with pytest.raises(sv.reg.GateError, match="C3: exp_45 holds 1 outstanding"):
        _campaign(tmp_path, monkeypatch)
    assert ledger.read_text() == held and sorted(os.listdir(ledger.parent)) == [".exp45_storage.json"]
    mine = json.load(open(tmp_path / "nas" / sv.reg.STORAGE_LEDGER))["reservations"]
    assert list(mine) == ["exp46_C4Ff025_l40_2x32"]
    assert not (tmp_path / "nas" / "exp46_C4Ff025_l40_2x32" / sv.reg.TRAINING_RECEIPT).exists()


@pytest.mark.parametrize("audit_ok,want", [(True, 0), (False, sv.EXIT_END)])
def test_a_neuronic_smoke_seals_its_receipt_from_the_snapshot(tmp_path, monkeypatch, audit_ok, want):
    """exp_46: the SMOKE lifecycle is the neuronic one (exp_45's A6000 SMOKE_DDP path is refused, test below)."""
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, audit_ok=audit_ok, smode="SMOKE")
    assert code == want and len(calls) == 2 and rd == tmp_path / "nas" / "exp46_C4Ff025_l40_2x32_smoke"
    assert "start_gate" not in rec and rec["mode"] == "SMOKE" and rec["cwd"] == str(rd / ".exp46_tools")
    # the SMOKE branch moves nothing: its logger dir stays where PL wrote it (the SMOKE gate checks it)
    assert sv.reg.logger_entries(str(rd)) == ["version_0", "version_0/hparams.yaml"] and "loss_record" not in rec
    assert not (rd / "receipts").exists() and not (tmp_path / "outputs_FLAC").exists()


def test_the_a6000_smoke_ddp_is_refused_before_any_gate_or_directory(tmp_path, monkeypatch):
    """exp_46 plan r3 §3.2 (Codex plan r2 finding 1): the inherited A6000 training-SMOKE destination
    outputs_FLAC/exp38_smoke is refused; nothing is gated, claimed, locked or created."""
    with pytest.raises(sv.Refused, match="no A6000 training path"):
        _campaign(tmp_path, monkeypatch, smode="SMOKE_DDP")
    assert not (tmp_path / "outputs_FLAC").exists() and os.listdir(str(tmp_path / "nas")) == []

V0 = "lightning_logs/version_0/"


@pytest.mark.parametrize("name,logger", [
    ("extra version", {V0 + "hparams.yaml": "{}\n", V0 + "metrics.csv": "LOSS", "lightning_logs/version_1/hparams.yaml": "{}\n"}),
    ("extra file", {V0 + "hparams.yaml": "{}\n", V0 + "metrics.csv": "LOSS", V0 + "events.out.tfevents.1": "x"}),
    ("missing csv", {V0 + "hparams.yaml": "{}\n"}),
    ("no logger dir", {}),
    ("short csv", {V0 + "hparams.yaml": "{}\n", V0 + "metrics.csv": "step,train/loss\n99,0.5\n"}),
])
def test_the_full_branch_moves_exactly_one_csv_logger_version_or_fails_its_end_audit(tmp_path, monkeypatch, name, logger):
    import exp46_fixtures as fx
    logger = {k: (fx.loss_csv(sv.reg).decode() if v == "LOSS" else v) for k, v in logger.items()}
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, logger=logger)
    assert code == sv.EXIT_END and rec["status"].startswith("end audit failed") and "loss_record" not in rec
    assert calls == [("commits", [])] and (rd / ".exp46_lock").is_dir()             # no receipt; crash marker kept
    assert sorted(sv.reg.logger_entries(str(rd))) == sorted({os.path.normpath(os.path.relpath(k, "lightning_logs"))
                                                            for k in logger} | ({os.path.dirname(os.path.relpath(k, "lightning_logs"))
                                                                                for k in logger} - {""}))
    assert not (rd / "receipts" / "loss_record.csv").exists()                     # nothing moved


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


def test_ar_content_changed_during_the_run_publishes_nothing(tmp_path, monkeypatch):
    """Codex code r3 BLOCKING 1 (defense in depth): the post-run re-hash refuses the terminal receipt."""
    code, rec, rd, calls = _campaign(tmp_path, monkeypatch, content_after_ok=False)
    assert code == sv.EXIT_END and "AR content changed during the run" in rec["status"]
    assert calls == [("commits", [])] and (rd / ".exp46_lock").is_dir()


@pytest.mark.parametrize("where", ["holder", "record"])
def test_an_exception_after_spawn_stops_our_group(tmp_path, monkeypatch, where):
    """Codex code r3 HIGH 2: a failure after the child exists (lock-holder start, a record write inside
    a gate) terminates and reaps the owned group before the exception propagates."""
    spawned = []
    real_child = sv.OwnedChild

    class Tracked(real_child):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            spawned.append(self)
    monkeypatch.setattr(sv, "OwnedChild", Tracked)
    if where == "holder":
        monkeypatch.setattr(sv, "start_lock_holder", lambda pgid, fds: (_ for _ in ()).throw(OSError("no holder")))
    else:
        real_update = sv.Record.update

        def flaky(self, strict=False, **fields):
            if "start_gate" in fields:
                raise RuntimeError("record write failed inside the start gate")
            return real_update(self, strict=strict, **fields)
        monkeypatch.setattr(sv.Record, "update", flaky)
    with pytest.raises((OSError, RuntimeError)):
        _campaign(tmp_path, monkeypatch)
    assert spawned and all(c.rc is not None for c in spawned) and all(_gone(c.pid) for c in spawned)


def test_d2_full_refuses_off_the_training_platform(monkeypatch):
    """D1 (exp_46) / D2 (exp_38): every training runs on neuronic; a FULL on this box refuses before any work."""
    assert sv.reg.TRAINING_PLATFORM == "neuronic"
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("c4f", "025", "2x32", PY, {}, {}))
    a = sv.parse(["--platform", "a6000", "--mode", "FULL", "--run", "exp46_C4Ff025_l40_2x32", "--expect-sha", "f" * 40])
    with pytest.raises(sv.Refused, match="D2"):
        sv.mode_campaign(a, "a6000")


# ------------------------------------------------------------------ the FULL completion, end to end (Codex code r1 finding 1)
E2E_TRAIN = textwrap.dedent("""
    import json, os, sys, time
    rd = sys.argv[sys.argv.index("--save-dir") + 1]
    print({log!r}, flush=True)
    for name in {ckpts!r}:
        open(os.path.join(rd, name), "wb").write(("ckpt " + name).encode())
    os.makedirs(os.path.join(rd, "evidence"), exist_ok=True)
    for name, rec in {evidence!r}.items():
        json.dump(rec, open(os.path.join(rd, "evidence", name), "w"))
    for rel, data in {logger!r}.items():                    # PL's CSV logger output
        path = os.path.join(rd, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(data)
    time.sleep(0.5)
    print({end!r}, flush=True)
""")


def _completed_full_run(tmp_path, monkeypatch, release=None):
    """mode_campaign of one FULL run to completion with the REAL retain_receipts, write_training_receipt and
    write_commit_marker (only the GPU / git / tree gates, the checkpoint audit and the snapshot copy are faked)."""
    import shutil
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_exp46_receipts as tr
    import exp46_fixtures as fx
    reg = sv.reg
    run = "exp46_C4Ff050_l40_2x32"
    arm, tag, rung = reg.parse_run_id(run)
    tree = tmp_path / "tree"
    (tree / reg.EXP_DIR).mkdir(parents=True)
    monkeypatch.setattr(reg, "REPO", str(tree))
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": tr.CLOSURE, "n_files": 1})
    monkeypatch.setattr(reg, "check_subset_audit", lambda repo=None: {"sha256": tr.AUDIT_SHA})
    rels = [reg.model_config_rel(arm, rung)] + [reg.FRACS[t][k] for t in reg.FRACS for k in ("split", "dataset_config")]
    for rel in rels:
        os.makedirs(tree / os.path.dirname(rel), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), tree / rel)
    paths, digests = tr._admitted(tree, monkeypatch, run)       # the sealed E0 / env / SMOKE_DDP receipts
    nas = tmp_path / "nas"
    nas.mkdir()
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(nas))
    monkeypatch.setitem(sv.SNAPSHOT_ROOT, "a6000", str(tmp_path / "snaps"))
    monkeypatch.setattr(sv, "common_gates", lambda a, p: (arm, tag, rung, PY, {"argv": []}, {}))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: (tr.HEAD, digests["env"], tr.CLOSURE))
    monkeypatch.setattr(sv, "lock_cards", lambda cards, lockdir: [])
    monkeypatch.setattr(sv, "a6000_gpu_gate", lambda *x: tr.A6000_GPUS)
    monkeypatch.setattr(sv, "backend_gate", lambda *x: {})
    monkeypatch.setattr(reg, "check_e0_receipt", lambda p: {"sha256": digests["e0"], "head": tr.HEAD})
    monkeypatch.setattr(reg, "check_smoke_receipt", lambda *x, **k: {"smoke_receipt": paths["smoke"],
                                                                     "sha256": digests["smoke"]})
    monkeypatch.setattr(reg, "storage_gate", lambda *x, **k: {"fstype": "cifs"})
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)
    monkeypatch.setattr(reg, "_gpu_rows", lambda: [["0", reg.A6000_UUIDS[0], "NVIDIA RTX A6000", "570.86"]])
    monkeypatch.setattr(reg, "audit_ckpt", lambda path, run, step, repo=None, run_dir=None: {
        "sha256": reg.sha256_file(path), "bytes": os.path.getsize(path), "global_step": step})
    import exp38_content_manifest as cm
    monkeypatch.setattr(cm, "verify", lambda *x, **k: [])
    token = "a" * 32
    rd = nas / run
    reg.reserve_storage(run, token, str(rd), str(nas))          # what the submitter did before sbatch
    ev = {f"run_evidence.json.rank{r}.json": e for r, e in tr.records(run, max_steps=40000).items()}
    log = tr.smoke_log(run, steps=40000, losses=60, end=False)
    v0 = "lightning_logs/version_0/"
    logger = {v0 + "hparams.yaml": "{}\n", v0 + "metrics.csv": fx.loss_csv(reg, tag).decode()}

    def fake_snapshot(run_, dest, repo=None):
        os.makedirs(dest)
        open(os.path.join(dest, "train.py"), "w").write(E2E_TRAIN.format(
            log=log, ckpts=[reg.ckpt_name(s, tag) for s in reg.cadence_steps()], evidence=ev, logger=logger,
            end=reg.FIT_END.format(n=40000)))
        for rel in rels:                                      # what the receipt writer re-hashes in the snapshot
            os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
            shutil.copyfile(tree / rel, os.path.join(dest, rel))
        return {"snapshot": dest, "files": len(rels) + 1, "snapshot_sha256": fx.write_snapshot_manifests(reg, dest)}
    monkeypatch.setattr(reg, "snapshot_tools", fake_snapshot)
    monkeypatch.setattr(reg, "verify_snapshot", lambda dest, want=None: {"verified": True})
    if release is not None:
        monkeypatch.setattr(reg, "release_storage", release)
    a = sv.parse(["--platform", "a6000", "--mode", "FULL", "--run", run, "--expect-sha", tr.HEAD, "--grace", "2",
                  "--token", token])
    return a, run, rd, nas, tr, digests


def _farm_copy(reg, rd, farm):
    """The run's records as the evaluation farm keeps them (exp46_eval.prepare's copy set): the offline consumer's input."""
    import shutil
    farm.mkdir()
    lr = json.load(open(rd / reg.COMMIT_MARKER))["launch_record"]
    for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256", reg.COMMIT_MARKER,
              reg.COMMIT_MARKER + ".sha256", lr):
        shutil.copyfile(rd / f, farm / f)
    for sub in (reg.RECEIPTS_DIR, "evidence"):
        shutil.copytree(rd / sub, farm / sub)


def test_a_completed_full_run_passes_online_and_offline_admission(tmp_path, monkeypatch):
    """Codex code r1 finding 1 (BLOCKING): a FULL run that mode_campaign() completes is admissible -- the sealed
    commit marker binds the terminal launch record's CURRENT bytes, so check_training_receipt passes online (the run
    dir) and offline (the farm copy, verify_bytes=False). The storage release is the ledger's own entry, never a
    rewrite of the sealed launch record."""
    a, run, rd, nas, tr, digests = _completed_full_run(tmp_path, monkeypatch)
    reg = sv.reg
    assert sv.mode_campaign(a, "a6000") == 0
    cm = json.load(open(rd / reg.COMMIT_MARKER))
    lr = rd / cm["launch_record"]
    assert reg.sha256_file(str(lr)) == cm["launch_record_sha256"]
    rec = json.load(open(lr))
    assert rec["status"] == "COMPLETE" and "storage_released" not in rec
    assert json.load(open(tmp_path / "tree" / reg.EXP_DIR / cm["launch_record"])) == rec      # both copies final
    got, digest = reg.check_training_receipt(run, str(rd), closure=tr.CLOSURE, e0_sha=digests["e0"], platform="a6000")
    assert got["loss_record_rows"] == 400 and digest == cm["training_receipt_sha256"]
    farm = tmp_path / "farm"
    _farm_copy(reg, rd, farm)
    reg.check_training_receipt(run, str(rd), closure=tr.CLOSURE, e0_sha=digests["e0"], platform="a6000",
                               receipt_dir=str(farm), verify_bytes=False)
    led = json.load(open(nas / reg.STORAGE_LEDGER))
    assert led["reservations"] == {}
    assert [(r["run"], r["token"], r["reason"]) for r in led["releases"]] == [(run, "a" * 32, "sealed")]


@pytest.mark.parametrize("exc", [sv.reg.GateError("ledger unreadable"), KeyboardInterrupt()])
def test_nothing_rewrites_the_launch_record_after_the_commit_marker(tmp_path, monkeypatch, exc):
    """Codex code r1 finding 1: once the commit marker sealed the launch record's hash, a failing release -- reported,
    or an interrupt that unwinds the supervisor -- leaves the record byte-identical: the run stays admissible and its
    reservation stays held (released later by hand, never by rewriting the record)."""
    def boom(*x, **k):
        raise exc
    a, run, rd, nas, tr, digests = _completed_full_run(tmp_path, monkeypatch, release=boom)
    reg = sv.reg
    if isinstance(exc, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            sv.mode_campaign(a, "a6000")
    else:
        assert sv.mode_campaign(a, "a6000") == 0
    cm = json.load(open(rd / reg.COMMIT_MARKER))
    assert reg.sha256_file(str(rd / cm["launch_record"])) == cm["launch_record_sha256"]
    reg.check_training_receipt(run, str(rd), closure=tr.CLOSURE, e0_sha=digests["e0"], platform="a6000")
    assert list(json.load(open(nas / reg.STORAGE_LEDGER))["reservations"]) == [run]


# ------------------------------------------------------------------ the first SMOKEs and the environment receipt (finding 14)
def test_three_concurrent_first_smokes_create_exactly_one_environment_receipt(tmp_path, monkeypatch):
    """Codex code r1 finding 14: the neuronic sheet submits the three SMOKEs together, so up to three jobs may find no
    environment receipt at once. Its creation is serialised under an exclusive lock: exactly one job writes the sealed
    receipt, the others re-check it under the same lock -- all three pass with the one receipt's sha256 and the
    receipt is written ONCE (the snapshot takes 0.5 s, so every job has seen "no receipt" before the first write;
    unserialised, identical snapshots would be published two or three times over each other)."""
    import multiprocessing as mp
    reg = sv.reg
    (tmp_path / reg.EXP_DIR).mkdir(parents=True)
    monkeypatch.setattr(reg, "REPO", str(tmp_path))
    monkeypatch.setattr(sv, "git", lambda *args, repo=None: "f" * 40 if args[0] == "rev-parse" else "")
    monkeypatch.setattr(reg, "check_untracked", lambda platform, repo=None: [])
    monkeypatch.setattr(reg, "env_problems", lambda: [])
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": "c" * 64})
    snap = {k: k for k in reg.ENV_RECEIPT_KEYS}
    snap.update(schema=1, platform="neuronic", packages={"torch": "2.1.0"})

    def slow_snapshot(platform):
        time.sleep(0.5)
        return json.loads(json.dumps(snap))
    monkeypatch.setattr(reg, "env_snapshot", slow_snapshot)
    writes, real_write = tmp_path / "receipt_writes.log", reg.write_sealed

    def counted_write(path, rec):                      # one O_APPEND line per write of the receipt, fork-safe
        if path == reg._abs(reg.env_receipt_rel("neuronic")):
            fd = os.open(str(writes), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
        return real_write(path, rec)
    monkeypatch.setattr(reg, "write_sealed", counted_write)
    ctx = mp.get_context("fork")
    barrier, q = ctx.Barrier(3), ctx.Queue()

    def first_smoke(i):
        a = sv.parse(["--platform", "neuronic", "--mode", "SMOKE", "--run", reg.required_runs()[i], "--expect-sha",
                      "f" * 40, "--token", str(i) * 32, "--save-dir", "/x"])
        barrier.wait()
        try:
            q.put((i, "ok", sv.campaign_gates(a, "neuronic")[1]))
        except BaseException as err:                       # noqa: BLE001 -- reported to the parent
            q.put((i, "error", repr(err)))
    procs = [ctx.Process(target=first_smoke, args=(i,)) for i in range(3)]
    for p in procs:
        p.start()
    got = sorted(q.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    assert len(writes.read_text().split()) == 1, writes.read_text()       # created exactly once
    assert [g[1] for g in got] == ["ok"] * 3, got
    _, digest = reg.read_sealed(reg._abs(reg.env_receipt_rel("neuronic")), reg.ENV_RECEIPT_KEYS)
    assert {g[2] for g in got} == {digest}
    assert reg.UNTRACKED_RECORDS.match(reg.ENV_RECEIPT_LOCK)     # the lock file is a registered record name



def test_a_failure_after_the_marker_rename_never_rewrites_the_sealed_record(tmp_path, monkeypatch):
    """Codex code r2 finding 5: the terminal launch record is frozen BEFORE the commit marker is published, so a
    failure after the marker's rename -- here the directory fsync of its publication -- cannot rewrite it: the
    published marker still binds the record's bytes, the run stays admissible, and the hold stays (`release`)."""
    a, run, rd, nas, tr, digests = _completed_full_run(tmp_path, monkeypatch)
    reg = sv.reg
    real = reg._fsync_dir

    def fail_after_rename(d):
        if os.path.exists(os.path.join(d, reg.COMMIT_MARKER)):
            raise OSError("injected: the directory fsync after the marker's rename failed")
        return real(d)
    monkeypatch.setattr(reg, "_fsync_dir", fail_after_rename)
    with pytest.raises(OSError, match="injected"):
        sv.mode_campaign(a, "a6000")
    monkeypatch.setattr(reg, "_fsync_dir", real)
    cm = json.load(open(rd / reg.COMMIT_MARKER))
    assert reg.sha256_file(str(rd / cm["launch_record"])) == cm["launch_record_sha256"]
    assert json.load(open(rd / cm["launch_record"]))["status"] == "COMPLETE"
    reg.check_training_receipt(run, str(rd), closure=tr.CLOSURE, e0_sha=digests["e0"], platform="a6000")
    assert list(json.load(open(nas / reg.STORAGE_LEDGER))["reservations"]) == [run]
