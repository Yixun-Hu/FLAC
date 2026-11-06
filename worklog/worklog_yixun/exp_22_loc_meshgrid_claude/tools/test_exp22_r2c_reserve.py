"""exp_22 A7 (plan §9 A7 r2 (v)(2), r3 (b) (e) (h) (i)): the reservation and the first durable state -- the O_EXCL
record (holder exp22, PIN2, the card, epoch, created, run), the re-checks immediately before AND after the
acquisition, the authenticated exact-record release, fault injection at every reservation / state transition,
O_EXCL races, the no-exp22-run-active check, the Planner's release and the launcher's identity confirmation."""
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, HOLDER, PIN2, RUN, FakeWorld, cfg_for, load, res_file, reserve, setup

E = load("exp22_r2c_evidence")
S = load("exp22_r2c_supervisor")
class Crash(BaseException):
    """A simulated kill: not an Exception, so no handler of the code under test can swallow it."""


def test_a_reservation_is_our_exact_record_and_the_state_starts_reserved(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    rc, msg = reserve(cfg, w, wt, orch, seal)
    assert rc == 0, msg
    raw = open(res_file(cfg), "rb").read()
    doc = json.loads(raw)
    assert list(doc) == ["holder", "pin2", "gpu_uuid", "epoch", "created", "run"]
    assert (doc["holder"], doc["pin2"], doc["gpu_uuid"], doc["epoch"], doc["run"]) == ("exp22", PIN2, GPU0, 1, RUN)
    st = json.load(open(orch / "state.json"))
    assert st["phase"] == "reserved" and st["supervisor"] is None and st["lines"] == [] and st["current"] is None
    held = st["reservation"]
    assert held["path"] == res_file(cfg) and held["sha256"] == E.sha256_bytes(raw)
    assert [held["dev"], held["inode"]] == [os.stat(res_file(cfg)).st_dev, os.stat(res_file(cfg)).st_ino]
    assert st["card"] == {"name": "GPU0", "uuid": GPU0, "index": 0} and st["run"] == RUN and st["pin2"] == PIN2
    assert st["seal"] == {"path": str(seal), "sha256": E.sha256_bytes(seal.read_bytes())}
    assert S.state_reasons(st, cfg) == []


@pytest.mark.parametrize("prior", ["exp35", "exp22", "malformed"])
def test_a_card_reserved_by_anyone_is_refused_before_the_acquisition(tmp_path, prior):
    cfg, w, wt, orch, seal = setup(tmp_path)
    raw = b"{" if prior == "malformed" else json.dumps({"holder": prior}).encode()
    open(res_file(cfg), "wb").write(raw)
    rc, msg = reserve(cfg, w, wt, orch, seal)
    assert rc == 3 and "reservation" in msg and open(res_file(cfg), "rb").read() == raw
    assert not (orch / "state.json").exists()


def test_an_o_excl_race_lost_at_the_acquisition_touches_nothing(tmp_path, monkeypatch):
    cfg, w, wt, orch, seal = setup(tmp_path)
    real = C.observe

    def observe_then_lose(*a, **k):                     # another holder wins between the check and the O_EXCL
        o = real(*a, **k)
        if not os.path.exists(res_file(cfg)):
            open(res_file(cfg), "w").write(json.dumps({"holder": "exp35"}))
        return o
    monkeypatch.setattr(C, "observe", observe_then_lose)
    rc, msg = reserve(cfg, w, wt, orch, seal)
    assert rc == 3 and "O_EXCL" in msg and json.load(open(res_file(cfg))) == {"holder": "exp35"}
    assert not (orch / "state.json").exists()


@pytest.mark.parametrize("change", ["foreign-process", "exp35-lock", "replaced-record"])
def test_a_change_seen_right_after_the_acquisition_releases_only_our_record(tmp_path, monkeypatch, change):
    cfg, w, wt, orch, seal = setup(tmp_path)
    real, calls, holder = C.observe, [], []

    def observe(world, cfg_, card, held=None, ours=()):
        calls.append(held)
        if held is not None and change == "foreign-process":
            world.answers["apps"] = f"4242, {GPU0}\n"
        if held is not None and change == "exp35-lock" and not holder:
            p = subprocess.Popen([sys.executable, "-c", HOLDER, cfg.exp35_lock], stdout=subprocess.PIPE, text=True)
            assert p.stdout.readline().strip() == "locked"
            holder.append(p)
        if held is not None and change == "replaced-record":
            os.unlink(held["path"])
            open(held["path"], "w").write(json.dumps({"holder": "someone-else"}))
        return real(world, cfg_, card, held=held, ours=ours)
    monkeypatch.setattr(C, "observe", observe)
    try:
        rc, msg = reserve(cfg, w, wt, orch, seal)
    finally:
        for p in holder:
            p.kill()
            p.wait()
    assert rc == 3 and calls[0] is None and calls[1] is not None and not (orch / "state.json").exists()
    if change == "replaced-record":                       # not ours any more: never unlinked
        assert json.load(open(res_file(cfg))) == {"holder": "someone-else"} and "nothing unlinked" in msg
    else:
        assert not os.path.exists(res_file(cfg)) and "released our own record" in msg


@pytest.mark.parametrize("point", ["after-create", "after-recheck", "in-state-write"])
def test_a_crash_at_any_reserve_transition_leaves_everything_in_place(tmp_path, monkeypatch, point):
    """A7 r3 (e): no crash path releases or adopts: the reservation stays, a restart refuses, the Planner resolves."""
    cfg, w, wt, orch, seal = setup(tmp_path)
    if point == "after-create":
        real = S.create_reservation
        monkeypatch.setattr(S, "create_reservation", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(Crash()))[0])
    elif point == "after-recheck":
        real = C.observe
        monkeypatch.setattr(C, "observe", lambda *a, **k: (real(*a, **k), (_ for _ in ()).throw(Crash()))[0]
                            if k.get("held") else real(*a, **k))
    else:
        monkeypatch.setattr(S.E, "write_exclusive", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        reserve(cfg, w, wt, orch, seal)
    monkeypatch.undo()
    raw = open(res_file(cfg), "rb").read()
    assert json.loads(raw)["holder"] == "exp22" and json.loads(raw)["run"] == RUN       # self-describing orphan
    assert not (orch / "state.json").exists()
    assert reserve(cfg, w, wt, orch, seal)[0] == 3                                       # a restart refuses
    assert open(res_file(cfg), "rb").read() == raw


def test_a_restart_refuses_while_the_state_exists(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    assert reserve(cfg, w, wt, orch, seal)[0] == 0
    os.unlink(res_file(cfg))
    rc, msg = reserve(cfg, w, wt, orch, seal)
    assert rc == 3 and "state" in msg and not os.path.exists(res_file(cfg))


@pytest.mark.parametrize("bad", ["sha", "pin2", "run", "orch"])
def test_the_seal_and_the_orchestration_directory_bind_the_reservation(tmp_path, bad):
    cfg, w, wt, orch, seal = setup(tmp_path)
    kw = dict(worktree=str(wt), orch=str(orch), pin2=PIN2, card="GPU0", index=0, seal=str(seal),
              seal_sha256=E.sha256_bytes(seal.read_bytes()))
    if bad == "sha":
        kw["seal_sha256"] = "0" * 64
    elif bad == "pin2":
        kw["pin2"] = "b" * 40
    elif bad == "run":
        seal.write_text(f"pin2 {PIN2}\nrun 20261106T999999\ncard GPU0 {GPU0} 0\n")
        kw["seal_sha256"] = E.sha256_bytes(seal.read_bytes())
    else:
        other = orch.parent / "elsewhere"
        other.mkdir()
        kw["orch"] = str(other)
    rc, msg = S.reserve(w, cfg, **kw)
    assert rc == 3 and not os.path.exists(res_file(cfg)) and not (orch / "state.json").exists()


def test_the_index_must_match_the_live_map(tmp_path):
    cfg, w, wt, orch, seal = setup(tmp_path)
    seal.write_text(f"pin2 {PIN2}\nrun {RUN}\ncard GPU0 {GPU0} 1\n")       # sealed at index 1, GPU 0 now at 0
    rc, msg = reserve(cfg, w, wt, orch, seal, index=1)
    assert rc == 3 and "index" in msg and not os.path.exists(res_file(cfg))


# --------------------------------------------------------------------------- #
# the authenticated release
# --------------------------------------------------------------------------- #
def _held(tmp_path):
    cfg = cfg_for(tmp_path)
    return cfg, S.create_reservation(cfg.reservations, GPU0, pin2=PIN2, run=RUN, now=1.0)


def test_the_exact_record_is_released(tmp_path):
    cfg, held = _held(tmp_path)
    assert S.release_reservation(held) == [] and not os.path.exists(held["path"])
    assert S.release_reservation(held)                                       # gone: refused, nothing to unlink


@pytest.mark.parametrize("tamper", ["bytes", "inode", "symlink"])
def test_a_record_that_is_not_ours_is_never_unlinked(tmp_path, tamper):
    cfg, held = _held(tmp_path)
    raw = open(held["path"], "rb").read()
    if tamper == "bytes":
        open(held["path"], "wb").write(raw.replace(b'"epoch": 1', b'"epoch": 2'))
    elif tamper == "inode":                                                  # identical bytes, another file
        open(held["path"] + ".new", "wb").write(raw)                         # (a new inode while the old lives)
        os.replace(held["path"] + ".new", held["path"])
    else:
        target = tmp_path / "elsewhere.json"
        target.write_bytes(raw)
        os.unlink(held["path"])
        os.symlink(target, held["path"])
    assert S.release_reservation(held) and os.path.lexists(held["path"])


def test_create_is_o_excl(tmp_path):
    cfg, held = _held(tmp_path)
    with pytest.raises(FileExistsError):
        S.create_reservation(cfg.reservations, GPU0, pin2=PIN2, run=RUN, now=2.0)
