"""exp_35 round 6a (iii-a): the supervisor's lane ownership (plan_orbitrir_yaw_evidence.md r3 §12 B1, r4 §13 B1) --
``Supervisor.ready`` (the guard, the GPU tables, the approved identity tuple, the predictor; waits that never
spawn) and ``Supervisor.handover`` (GPU 0 free on two checks 60 s apart -> its O_EXCL reservation -> the state ->
the GPU 1 file removed), the runner / probe commands; and the fake world the loop tests (iii-b) share: fake time,
a fake nvidia-smi table, a fake /proc read by the REAL identity helper, Yixun's log read by the REAL guard.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import hashlib
import json
import os
import shutil

import pytest

import test_exp35_supervisor_core as SC
import test_exp35_training_guard as TG

S = SC.S
G0, G1 = SC.G0, SC.G1
TR0, TR1 = 3027719, 3028196
CMD = b"/home/yixunhu/miniconda3/envs/flac/bin/python\x00train.py\x00--seed\x0043\x00"
HISLOG = "/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/train_s43.log"
N_CHUNKS = 33


def proc_entry(root, pid, *, ppid=1, start=887766, cmdline=CMD, cwd="/home/yixunhu/codespace/flac-fork-attn"):
    base = root / str(pid)
    shutil.rmtree(base, ignore_errors=True)
    (base / "fd").mkdir(parents=True)
    fields = ["S", str(ppid)] + ["7"] * 17 + [str(start)] + ["0"] * 30           # field 4 ppid, field 22 start
    (base / "stat").write_text(f"{pid} (python x) " + " ".join(fields) + "\n")
    (base / "cmdline").write_bytes(cmdline)
    os.symlink(cwd, base / "cwd")
    for fd in ("1", "2"):
        os.symlink(HISLOG, base / "fd" / fd)


class FakeProc:
    """A spawned runner / probe: alive for ``polls`` world sleeps, its evaluator (pid + 1) on ``uuid`` meanwhile."""

    def __init__(self, world, pid, polls, rc, uuid, on_exit):
        self.w, self.pid, self.left, self.rc, self.uuid, self.on_exit = world, pid, polls, rc, uuid, on_exit
        self.returncode = None
        proc_entry(world.procdir, pid, ppid=1, cmdline=b"bash\x00runner\x00")
        proc_entry(world.procdir, pid + 1, ppid=pid, cmdline=b"python\x00eval_FLAC.py\x00")
        world.apps.append((pid + 1, uuid))

    def tick(self):
        if self.returncode is None:
            self.left -= 1
            if self.left <= 0:
                self.w.apps.remove((self.pid + 1, self.uuid))
                shutil.rmtree(self.w.procdir / str(self.pid + 1))
                shutil.rmtree(self.w.procdir / str(self.pid))
                self.returncode = self.rc
                if self.on_exit:
                    self.on_exit(self)

    def poll(self):
        return self.returncode


class FakeWorld:
    def __init__(self, tmp):
        self.tmp, self.procdir, self.res = tmp, tmp / "proc", tmp / "reservations"
        self.procdir.mkdir()
        self.res.mkdir()
        self.log = tmp / "train_s43.log"
        TG.write_log(self.log, [4.23, 4.23, 4.23])
        self.t = os.stat(self.log).st_mtime
        self.gpus = {0: G0, 1: G1}
        self.apps = [(TR0, G0), (TR1, G1)]
        proc_entry(self.procdir, TR0)
        proc_entry(self.procdir, TR1)
        self.procs, self.spawned, self.sleeps, self.hooks, self.pid = [], [], [], [], 50000
        self.behaviour = lambda argv: (2, 0)

    def now(self):
        return self.t

    def sleep(self, s):
        if len(self.sleeps) >= 600:                   # a supervisor that would wait for ever fails, not hangs
            raise RuntimeError("the supervisor waited past the fake world's horizon (600 sleeps)")
        self.t += s
        self.sleeps.append(s)
        for p in list(self.procs):
            p.tick()
        for hook in [h for h in self.hooks if h[0](self)]:
            self.hooks.remove(hook)
            hook[1](self)

    def nvsmi(self, query):
        rows = self.gpus.items() if query == "map" else self.apps
        return "".join(f"{a}, {b}\n" for a, b in rows)

    def spawn(self, argv, env, log):
        self.pid += 10
        polls, rc = self.behaviour(argv)
        uuid = self.gpus[int(env["CUDA_VISIBLE_DEVICES"])] if "--lanes" not in argv else self.gpus[
            int(argv[argv.index("--lanes") + 1])]
        self.spawned.append({"argv": argv, "env": env, "log": log, "t": self.t, "uuid": uuid, "sleeps": len(self.sleeps),
                             "files": sorted(p.name for p in self.res.iterdir())})
        with open(log, "w") as f:                       # the probe's log carries the launcher's backend evidence
            f.write(SC.T.backend_log(SC.T.C.PROBE_CONTRACT_BF["target"]) if "--arm" in argv else f"fake {argv[1]}\n")
        p = FakeProc(self, self.pid, polls, rc, uuid, self.probe_exit if "--arm" in argv else None)
        self.procs.append(p)
        return p

    def probe_exit(self, proc):
        """The probe's own outputs as write_report leaves them: a B-F report awaiting its finalization, and its sha256
        sidecar (finalization is the gates' business)."""
        text = json.dumps({"arm": "BF", "provenance": {}}, indent=1) + "\n"
        (self.tmp / "orbitrir_bf_cond_probe.json").write_text(text)
        (self.tmp / "orbitrir_bf_cond_probe.json.sha256").write_text(
            f"{hashlib.sha256(text.encode()).hexdigest()}  orbitrir_bf_cond_probe.json\n")

    def at_spawn(self, k, fn):
        """Run ``fn`` once the k-th spawn happened (on the next sleep)."""
        self.hooks.append((lambda w: len(w.spawned) >= k, fn))

    def remove_trainers(self, gpu0=True):
        self.apps = [a for a in self.apps if a[0] != TR1 and (not gpu0 or a[0] != TR0)]
        shutil.rmtree(self.procdir / str(TR1))
        if gpu0:
            shutil.rmtree(self.procdir / str(TR0))


class FakeGates:
    def __init__(self):
        self.calls, self.e1_rc, self.e2_rc, self.finalize_rc, self.bad_verify = [], {1: 0, 8: 0}, 0, 0, set()
        self.probe_json = None

    def verify(self, chunk):
        self.calls.append(("verify", chunk["cell"]))
        if chunk["cell"] in self.bad_verify:
            return False, {"reasons": ["INVALID"]}
        return True, {"receipt_sha256": hashlib.sha256(chunk["cell"].encode()).hexdigest(), "campaign_digest": "d" * 64}

    def e1(self, k):
        self.calls.append(("E1", k))
        return self.e1_rc[k], "E1 out"

    def e2(self):
        self.calls.append(("E2",))
        return self.e2_rc, "E2 out"

    def finalize(self, log, digest):
        """The REAL collector finalizer over the probe's report (its production rule -- idempotent only on the identical
        record, Codex C3d HIGH 3); the contract after it is ``finalize_rc``."""
        self.calls.append(("finalize", log, digest))
        if self.finalize_rc != 0:
            return self.finalize_rc, "BF PROBE INVALID"
        try:
            SC.T.C.finalize_bf_probe(self.probe_json, log, campaign_digest=digest, base_dir=S.REPO)
        except (ValueError, OSError, TypeError) as err:
            return 1, f"BF PROBE FINALIZATION REFUSED: {err}"
        return 0, "BF PROBE VALID"


def tuple_file(world):
    I = S.identity_module()
    doc = {"schema_version": 1, "gpu_uuid": G1, "processes": [I.proc_identity(TR1, str(world.procdir))]}
    path = world.tmp / "approved.json"
    path.write_bytes(json.dumps(doc).encode())
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def make(tmp_path, monkeypatch=None):
    w = FakeWorld(tmp_path)
    approved, sha = tuple_file(w)
    cfg = {"h35": "a" * 40, "state": str(tmp_path / "state.json"), "approved": approved, "approved_sha256": sha,
           "reservations": str(w.res), "his_log": str(w.log), "cells_sha256": "c" * 64, "supervisor_sha256": "d" * 64,
           "proc": str(w.procdir), "logdir": str(tmp_path / "logs"), "probe_json": str(tmp_path / "orbitrir_bf_cond_probe.json")}
    os.makedirs(cfg["logdir"])
    g = FakeGates()
    g.probe_json = cfg["probe_json"]
    return w, g, cfg


def sup(w, gates, cfg):
    return S.Supervisor(cfg, w, gates)


def state(cfg):
    return json.load(open(cfg["state"]))


def _lane(rec):
    return rec["uuid"], rec["env"].get("ALLOW_SHARED_GPU"), rec["env"].get("EXP35_APPROVED_COTENANTS")


# --------------------------------------------------------------------------- #
# ready() and handover() on their own
# --------------------------------------------------------------------------- #
CH1 = S.chunk_plan()[0]


def _sup(tmp_path):
    w, g, cfg = make(tmp_path)
    s = S.Supervisor(cfg, w, g)
    s.st = S.new_state(h35=cfg["h35"], cells_sha256=cfg["cells_sha256"], supervisor_sha256=cfg["supervisor_sha256"],
                       approved_sha256=cfg["approved_sha256"])
    s.uuids = tuple(sorted((G0, G1)))                                          # as run() sets them from the GPU map
    return w, cfg, s


def _hold_gpu1(cfg, w, s):
    p1, sha = S.create_reservation(cfg["reservations"], G1, epoch=1, h35=cfg["h35"], now=w.t)
    s.st.update(reservation={"uuid": G1, "epoch": 1, "path": p1, "sha256": sha}, lane={"uuid": G1, "shared": True})
    return p1


def test_the_approved_trainer_alone_gives_the_shared_gpu1(tmp_path):
    w, cfg, s = _sup(tmp_path)
    assert s.ready(CH1) == (G1, 1, True) and w.sleeps == [] and s.st["his_log_anchor"]["size"] > 0


def test_an_extra_process_on_gpu1_waits_while_the_trainer_runs(tmp_path):
    w, cfg, s = _sup(tmp_path)
    w.apps.append((99999, G1))
    w.hooks.append((lambda world: len(world.sleeps) >= 3, lambda world: world.apps.remove((99999, G1))))
    assert s.ready(CH1) == (G1, 1, True) and w.sleeps == [S.WAIT_POLL_S] * 3 and w.spawned == []
    notes = [e["what"] for e in s.st["events"]]
    assert notes.count("no shared spawn: GPU 1 holds more than the approved trainer") == 1          # one note per reason


def test_the_trainer_gone_hands_over_to_gpu0_in_the_registered_order(tmp_path, monkeypatch):
    """§12 B1.4: (a) GPU 0 free twice 60 s apart, (b) its file O_EXCL, (c) the state with epoch + 1, (d) the GPU 1 file
    removed -- so our reservation is never absent and the state never names a file that does not exist."""
    w, cfg, s = _sup(tmp_path)
    _hold_gpu1(cfg, w, s)
    w.remove_trainers()
    order, real = [], (S.create_reservation, S.write_state, S.release_reservation)
    monkeypatch.setattr(S, "create_reservation", lambda *a, **k: (order.append(("create", a[1])), real[0](*a, **k))[1])
    monkeypatch.setattr(S, "write_state", lambda path, doc: (order.append(("state", (doc["reservation"] or {}).get(
        "epoch"))), real[1](path, doc))[1])
    monkeypatch.setattr(S, "release_reservation", lambda held, h35: (
        order.append(("remove", os.path.basename(held["path"]))), real[2](held, h35))[1])
    assert s.ready(CH1) == (G0, 0, False)
    steps = [o for o in order if o[0] != "state" or o[1] == 2]
    assert steps[steps.index(("create", G0)):][:3] == [("create", G0), ("state", 2), ("remove", f"{G1}.json")]
    assert [p.name for p in w.res.iterdir()] == [f"{G0}.json"] and s.st["lane"] == {"uuid": G0, "shared": False}
    free = [e for e in s.st["events"] if e["what"].startswith("GPU 0 free")]
    assert len(free) == 2 and free[1]["t"] - free[0]["t"] >= S.HANDOVER_GAP_S


def test_a_busy_gpu0_or_another_holder_makes_the_handover_wait(tmp_path):
    w, cfg, s = _sup(tmp_path)
    w.remove_trainers(gpu0=False)                                               # rank 0 still on GPU 0
    assert s.handover(*s.tables()) is False and "GPU 0 busy" in s.st["events"][-1]["what"]
    w.apps.remove((TR0, G0))
    (w.res / f"{G0}.json").write_text(json.dumps({"holder": "exp22", "gpu_uuid": G0}))
    assert s.handover(*s.tables()) is False and "another holder" in s.st["events"][-1]["what"]
    (w.res / f"{G0}.json").unlink()
    assert s.handover(*s.tables()) is True and s.st["reservation"]["epoch"] == 1        # no GPU 1 file held yet


# --------------------------------------------------------------------------- #
# Codex C3d HIGH 1: the held record authenticated before / after the gap, before the commit, before the removal
# --------------------------------------------------------------------------- #
HANDOVER_SPOILS = {
    "deleted": lambda p: os.unlink(p),
    "replaced_by_another_holder": lambda p: open(p, "w").write(json.dumps({"holder": "exp22", "gpu_uuid": G1})),
    "another_epoch": lambda p: SC._rewrite(p, epoch=5),
    "corrupt": lambda p: open(p, "w").write("{not json"),
}


def _content(p):
    return open(p, "rb").read() if os.path.exists(p) else None


@pytest.mark.parametrize("how", sorted(HANDOVER_SPOILS))
def test_a_reservation_spoiled_during_the_handover_gap_latches_and_unlinks_nothing(tmp_path, how):
    w, cfg, s = _sup(tmp_path)
    p1 = _hold_gpu1(cfg, w, s)
    w.remove_trainers()
    seen = {}

    def spoil(world):
        HANDOVER_SPOILS[how](p1)
        seen["after"] = _content(p1)
    w.hooks.append((lambda world: world.sleeps[-1:] == [S.HANDOVER_GAP_S], spoil))         # inside the 60-s gap
    assert s.ready(CH1) is None and "reservation" in s.st["stop_latch"]["reason"]
    assert _content(p1) == seen["after"]                                       # the foreign record kept as it is
    assert not os.path.exists(S.reservation_path(cfg["reservations"], G0)) and s.st["reservation"]["uuid"] == G1
    assert len([e for e in s.st["events"] if e["what"].startswith("GPU 0 free")]) == 1


@pytest.mark.parametrize("when", ["before_the_commit", "before_the_removal"])
@pytest.mark.parametrize("how", sorted(HANDOVER_SPOILS))
def test_a_reservation_spoiled_at_the_commit_or_the_removal_latches_and_unlinks_nothing(tmp_path, monkeypatch, how, when):
    w, cfg, s = _sup(tmp_path)
    p1 = _hold_gpu1(cfg, w, s)
    w.remove_trainers()
    seen, real = {}, (S.create_reservation, S.write_state)

    def spoil():
        if "after" not in seen:
            HANDOVER_SPOILS[how](p1)
            seen["after"] = _content(p1)
    if when == "before_the_commit":                            # right after GPU 0's file is created
        monkeypatch.setattr(S, "create_reservation", lambda *a, **k: (real[0](*a, **k), spoil())[0])
    else:                                                      # right after the committed state is written
        monkeypatch.setattr(S, "write_state", lambda path, doc: (real[1](path, doc), (doc["reservation"] or {}).get(
            "uuid") == G0 and spoil())[0])
    assert s.ready(CH1) is None and "nothing unlinked" in s.st["stop_latch"]["reason"]
    assert _content(p1) == seen["after"] and os.path.exists(S.reservation_path(cfg["reservations"], G0))
    assert s.st["reservation"]["uuid"] == (G1 if when == "before_the_commit" else G0)
    assert not any(e["what"] == "hand-over to GPU 0" for e in s.st["events"])


# --------------------------------------------------------------------------- #
# Codex C3d HIGH 2: a listed PID whose identity cannot be read is never the approved trainer
# --------------------------------------------------------------------------- #
UNREADABLE = {
    "deleted": lambda w: shutil.rmtree(w.procdir / str(TR1)),
    "corrupt_stat": lambda w: (w.procdir / str(TR1) / "stat").write_text("garbage\n"),
    "no_fd_link": lambda w: os.unlink(w.procdir / str(TR1) / "fd" / "1"),
}


@pytest.mark.parametrize("how", sorted(UNREADABLE))
def test_a_listed_trainer_with_an_unreadable_identity_latches_before_a_spawn(tmp_path, how):
    w, cfg, s = _sup(tmp_path)
    UNREADABLE[how](w)                                                         # still listed on GPU 1
    assert s.ready(CH1) is None and str(TR1) in s.st["stop_latch"]["reason"]
    assert "unreadable" in s.st["stop_latch"]["reason"] and w.sleeps == []
    assert not any(e["what"].startswith("GPU 0 free") for e in s.st["events"]) and list(w.res.iterdir()) == []


def test_an_unreadable_trainer_no_longer_listed_at_the_requery_is_gone(tmp_path):
    """It exited between the two observations -- listed, unreadable, and the re-queried table no longer lists it: gone,
    never a latch (the hand-over to GPU 0 follows). A regression guard: green before and after the fix."""
    w, cfg, s = _sup(tmp_path)
    w.apps.remove((TR0, G0))
    shutil.rmtree(w.procdir / str(TR0))                                       # rank 0 has exited ...
    shutil.rmtree(w.procdir / str(TR1))                                       # ... rank 1 is leaving: still listed
    real, queries = w.nvsmi, []

    def nvsmi(query):
        if query == "apps":
            queries.append(query)
            if len(queries) == 2:                                             # the re-query: it has left the table
                w.apps.remove((TR1, G1))
        return real(query)
    w.nvsmi = nvsmi
    assert s.ready(CH1) == (G0, 0, False) and s.st["stop_latch"] is None


def test_a_foreign_process_on_gpu0_latches_before_a_spawn(tmp_path):
    w, cfg, s = _sup(tmp_path)
    s.st["lane"] = {"uuid": G0, "shared": False}
    w.apps.append((66666, G0))
    assert s.ready(CH1) is None and "GPU 0" in s.st["stop_latch"]["reason"] and "66666" in s.st["stop_latch"]["reason"]


@pytest.mark.parametrize("bad", ["guard", "tables", "tuple"])
def test_ready_latches_when_it_cannot_judge(tmp_path, bad):
    w, cfg, s = _sup(tmp_path)
    if bad == "guard":
        s.st["baseline"] = {"time": w.t, "step": 0, "log": {}}
        TG.write_log(w.log, [12.0, 4.23])                                       # every window past the limits
    elif bad == "tables":
        w.gpus = {0: G0}                                                       # not this box's two cards
    else:
        with open(cfg["approved"], "a") as f:
            f.write(" ")                                                       # the bytes no longer hash to the pin
    assert s.ready(CH1) is None and s.st["stop_latch"] is not None
    assert {"guard": "guard", "tables": "GPU map", "tuple": "launch entry pins"}[bad] in s.st["stop_latch"]["reason"]


def test_the_predictor_holds_a_shared_spawn(tmp_path, monkeypatch):
    w, cfg, s = _sup(tmp_path)
    monkeypatch.setattr(S, "MAX_STEPS", 3 * TG.T + 500)
    w.hooks.append((lambda world: len(world.sleeps) >= 2, lambda world: world.remove_trainers()))
    assert s.ready(CH1) == (G0, 0, False)                                       # never GPU 1 near s43's end
    assert any(e["what"].startswith("predictor: no shared spawn") for e in s.st["events"])


def test_the_runner_and_probe_commands(monkeypatch):
    monkeypatch.setenv("ALLOW_SHARED_GPU", "1")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    cfg = {"h35": "a" * 40, "approved": "/x/approved.json", "approved_sha256": "e" * 64}
    argv, env = S.runner_command(cfg, {"cell": "exp35_BF_R_S40000_K1_s44"}, 1, True)
    assert argv == ["bash", S.RUNNER, "--lanes", "1", "--only", "exp35_BF_R_S40000_K1_s44"]
    assert {k: env[k] for k in ("EXPECT_HEAD", "GPU_MAX_POLLS", "GPU_POLL_SECONDS", "ALLOW_SHARED_GPU",
                                "EXP35_APPROVED_COTENANTS", "EXP35_APPROVED_COTENANTS_SHA256")} == {
        "EXPECT_HEAD": "a" * 40, "GPU_MAX_POLLS": "1", "GPU_POLL_SECONDS": "0", "ALLOW_SHARED_GPU": "1",
        "EXP35_APPROVED_COTENANTS": "/x/approved.json", "EXP35_APPROVED_COTENANTS_SHA256": "e" * 64}
    _, env = S.runner_command(cfg, {"cell": "c"}, 0, False)
    assert not any(k in env for k in ("ALLOW_SHARED_GPU", "EXP35_APPROVED_COTENANTS", "PYTHONPATH"))   # GPU 0: unset
    argv, env = S.probe_command(1)
    assert argv == [S.PY, S.LAUNCHER, S.PROBE, "--arm", "BF", "--ckpt-path", S.BF_FARM_LINK]
    assert (env["CUDA_VISIBLE_DEVICES"], env["PYTHONPATH"], env["PYTHONDONTWRITEBYTECODE"], env["HF_HUB_OFFLINE"]) == (
        "1", os.path.join(S.REPO, "overlays", "no_flash"), "1", "1") and "ALLOW_SHARED_GPU" not in env
