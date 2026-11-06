"""exp_35 phase (ii) kit, part 3b (plan_orbitrir_yaw_evidence.md §15 items 5, 6; Codex plan §15 r1 S3, S6): the
reserve transaction -- the preconditions at TWO observations gap_s apart (index <-> uuid each time), a fresh
orchestration with its durable state 'reserving' (O_EXCL), each lane's O_EXCL record (a two-lane acquisition is
all-or-nothing), the observation again AFTER the acquisition, then 'reserved'; on any loss or change only OUR exact
records are released ('reserve_refused'), a record that is no longer ours is retained ('reserve_failed'); a crash
leaves everything in place; an unresolved earlier orchestration, or a launched one the report does not name, refuses.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, GPU1, MAP, FakeWorld, make_box

K = SB.load_kit()


class Crash(BaseException):
    """A simulated kill: not an Exception, so no handler of the code under test swallows it."""


def _reserve(b, world=None, lanes=(0,), **kw):
    args = dict(kit_commit=b.commit, h35=b.head, lanes=lanes, report=b.report, report_sha256=b.report_sha)
    args.update(kw)
    return K.reserve(world or FakeWorld(), b.cfg, **args)


def _orchs(b):
    base = os.path.join(b.run, SB.X35)
    return sorted(os.path.join(base, n) for n in os.listdir(base) if n.startswith("exp35_phase_ii_"))


def _state(orch):
    return json.load(open(os.path.join(orch, "state.json")))


def test_a_reserve_writes_the_exact_records_after_two_observations_and_a_recheck(tmp_path):
    b, w = make_box(tmp_path), FakeWorld()
    rc, msg = _reserve(b, w)
    assert rc == 0 and msg.startswith("RESERVED"), msg
    [orch] = _orchs(b)
    st = _state(orch)
    assert list(st) == list(K.STATE_KEYS) and st["phase"] == "reserved" and st["schema_version"] == 1
    assert os.path.basename(orch) == "exp35_phase_ii_" + st["run"] and K.RUN_RE.fullmatch(st["run"])
    [held] = st["reservations"]
    raw = open(b.res(), "rb").read()
    assert held == {"index": 0, "uuid": GPU0, "path": b.res(), "sha256": K.sha256_bytes(raw),
                    "dev": os.stat(b.res()).st_dev, "inode": os.stat(b.res()).st_ino}
    assert json.loads(raw) == {"holder": "exp35", "phase": "ii", "h35": b.head, "gpu_uuid": GPU0, "gpu_index": 0,
                               "created": w.t, "run": st["run"]}
    assert w.slept == [b.cfg.gap_s] and len(st["observations"]) == 3
    t = [o["t"] for o in st["observations"]]
    assert t[1] - t[0] == b.cfg.gap_s and all(o["reasons"] == [] for o in st["observations"])
    assert st["lanes"] == [{"index": 0, "uuid": GPU0}] and st["kit"]["commit"] == b.commit
    assert st["evidence"] == K.evidence(b.cfg, lanes=(0,), report=b.report, report_sha256=b.report_sha,
                                        prior_runs=())[1]
    assert [e["what"].split(":")[0] for e in st["events"]] == ["reserving", "reserved"]
    assert (st["dry_run"], st["launch"], st["wrapper"], st["runner"], st["exit"], st["failure"]) == (None,) * 6
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"


def test_two_recorded_lanes_are_reserved_together(tmp_path):
    b = make_box(tmp_path, lanes="0,1", decision="Planner 2026-11-06T11:40-05:00 (Yixun: both cards)")
    with open(b.cfg.queue_log, "a") as f:
        f.write(SB.EVAL_DONE_S43)
    rc, msg = _reserve(b, lanes=(0, 1))
    assert rc == 0, msg
    assert [h["uuid"] for h in _state(_orchs(b)[0])["reservations"]] == [GPU0, GPU1]
    assert os.path.exists(b.res(GPU0)) and os.path.exists(b.res(GPU1))


def _apps_sequence(*answers):
    seq = list(answers)
    return lambda: seq.pop(0) if len(seq) > 1 else seq[0]


@pytest.mark.parametrize("case, reason", [
    ("kit", "40-hex"), ("h35", "registered"), ("latched", "stop latch"), ("busy-first", "compute process"),
    ("busy-second", "compute process"), ("map-change", "GPU map"), ("unresolved", "unresolved"),
    ("unnamed-prior-run", "prior-runs"), ("unreadable", "unreadable"), ("lock", "is held")])
def test_every_refusal_before_the_acquisition_writes_nothing(tmp_path, case, reason):
    b, w, kw = make_box(tmp_path), FakeWorld(), {}
    holder = None
    if case == "kit":
        kw["kit_commit"] = "abc"
    elif case == "h35":
        kw["h35"] = "b" * 40
    elif case == "latched":
        SB.write_phase_i(b.run, b.head, b.digest, stop_latch={"reason": "the guard", "time": "t"})
    elif case == "busy-first":
        w.answers["apps"] = f"4242, {GPU0}\n"
    elif case == "busy-second":
        w.answers["apps"] = _apps_sequence("", f"4242, {GPU0}\n")
    elif case == "map-change":
        w.answers["map"] = _apps_sequence(MAP, f"0, {GPU1}\n1, {GPU0}\n")
    elif case in ("unresolved", "unnamed-prior-run"):
        d = os.path.join(b.run, SB.X35, "exp35_phase_ii_20261101T000000")
        os.mkdir(d)
        st = dict.fromkeys(K.STATE_KEYS)
        st.update(schema_version=1, campaign="exp35_phase_ii", run="20261101T000000", events=[],
                  phase="reserved" if case == "unresolved" else "released",
                  launch=None if case == "unresolved" else {"started": 1.0})
        open(os.path.join(d, "state.json"), "w").write(json.dumps(st))
    elif case == "unreadable":
        w.answers["apps"] = lambda: (_ for _ in ()).throw(K.Unreadable("nvidia-smi --query-compute-apps: rc 9"))
    else:
        holder = subprocess.Popen([sys.executable, "-c", SB.HOLDER, os.path.join(b.run, K.KIT_LOCK), "30"],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "locked"
    try:
        before = set(_orchs(b))
        rc, msg = _reserve(b, w, **kw)
    finally:
        if holder:
            holder.kill()
            holder.wait()
    assert rc == 1 and "REFUSED" in msg and reason in msg, msg
    assert set(_orchs(b)) == before and not os.path.lexists(b.res()) and not os.path.lexists(b.res(GPU1))


def test_a_lost_o_excl_race_touches_nothing_of_the_winner(tmp_path, monkeypatch):
    b = make_box(tmp_path)
    real = K.create_reservation

    def lose(cfg, index, **kw):                       # another holder wins between the observation and the O_EXCL
        open(K.reservation_path(cfg, K.CARDS[index]), "w").write(json.dumps({"holder": "exp22", "run": "r"}))
        return real(cfg, index, **kw)
    monkeypatch.setattr(K, "create_reservation", lose)
    rc, msg = _reserve(b)
    st = _state(_orchs(b)[0])
    assert rc == 1 and "lost race" in msg and json.load(open(b.res())) == {"holder": "exp22", "run": "r"}
    assert st["phase"] == "reserve_refused" and st["reservations"] == [] and "lost race" in st["failure"]["reasons"][0]


def test_a_partial_two_lane_acquisition_releases_our_first_record_only(tmp_path, monkeypatch):
    b = make_box(tmp_path, lanes="0,1", decision="Planner: both cards")
    with open(b.cfg.queue_log, "a") as f:
        f.write(SB.EVAL_DONE_S43)
    real = K.create_reservation

    def second_lost(cfg, index, **kw):
        if index == 1:
            open(K.reservation_path(cfg, GPU1), "w").write(json.dumps({"holder": "exp22", "run": "r"}))
        return real(cfg, index, **kw)
    monkeypatch.setattr(K, "create_reservation", second_lost)
    rc, msg = _reserve(b, lanes=(0, 1))
    st = _state(_orchs(b)[0])
    assert rc == 1 and not os.path.lexists(b.res(GPU0)) and json.load(open(b.res(GPU1)))["holder"] == "exp22"
    assert st["phase"] == "reserve_refused" and st["reservations"] == [] and len(st["released"]["records"]) == 1


def test_an_occupation_seen_after_the_acquisition_releases_our_record(tmp_path):
    b, w = make_box(tmp_path), FakeWorld()
    w.answers["apps"] = _apps_sequence("", "", f"4242, {GPU0}\n")
    rc, msg = _reserve(b, w)
    st = _state(_orchs(b)[0])
    assert rc == 1 and "after the acquisition" in msg and not os.path.lexists(b.res())
    assert st["phase"] == "reserve_refused" and any("compute process" in x for x in st["failure"]["reasons"])
    assert len(st["observations"]) == 3 and st["observations"][2]["reasons"]


def test_a_record_replaced_right_after_the_acquisition_is_never_unlinked(tmp_path, monkeypatch):
    b = make_box(tmp_path)
    real = K.create_reservation

    def replaced(cfg, index, **kw):
        held = real(cfg, index, **kw)
        os.unlink(held["path"])
        open(held["path"], "w").write(json.dumps({"holder": "someone-else"}))
        return held
    monkeypatch.setattr(K, "create_reservation", replaced)
    rc, msg = _reserve(b)
    st = _state(_orchs(b)[0])
    assert rc == 1 and json.load(open(b.res())) == {"holder": "someone-else"}
    assert st["phase"] == "reserve_failed" and len(st["reservations"]) == 1 and st["reservations"][0]["unreleased"]


@pytest.mark.parametrize("point", ["before-the-record", "after-the-record"])
def test_a_crash_inside_the_transaction_leaves_everything_for_review(tmp_path, monkeypatch, point):
    b = make_box(tmp_path)
    real = K.create_reservation

    def crash(cfg, index, **kw):
        if point == "after-the-record":
            real(cfg, index, **kw)
        raise Crash()
    monkeypatch.setattr(K, "create_reservation", crash)
    with pytest.raises(Crash):
        _reserve(b)
    monkeypatch.undo()
    [orch] = _orchs(b)
    assert _state(orch)["phase"] == "reserving" and os.path.lexists(b.res()) == (point == "after-the-record")
    raw = open(b.res(), "rb").read() if point == "after-the-record" else None
    assert K.lock_status(os.path.join(b.run, K.KIT_LOCK)) == "unheld"
    rc, msg = _reserve(b, FakeWorld(t=1_800_000_100.0))                      # a restart refuses: unresolved
    assert rc == 1 and "unresolved" in msg and _orchs(b) == [orch]
    assert (open(b.res(), "rb").read() if raw else None) == raw


def test_a_resolved_earlier_orchestration_does_not_block_and_a_named_prior_run_passes(tmp_path):
    b, w = make_box(tmp_path, prior="20261101T000000"), FakeWorld()
    d = os.path.join(b.run, SB.X35, "exp35_phase_ii_20261101T000000")
    os.mkdir(d)
    st = dict.fromkeys(K.STATE_KEYS)
    st.update(schema_version=1, campaign="exp35_phase_ii", run="20261101T000000", phase="released", events=[],
              launch={"started": 1.0})
    open(os.path.join(d, "state.json"), "w").write(json.dumps(st))
    rc, msg = _reserve(b, w)
    assert rc == 0, msg
    assert _state([o for o in _orchs(b) if o != d][0])["evidence"]["prior_runs"] == ["20261101T000000"]


def test_a_malformed_earlier_state_is_unresolved(tmp_path):
    b = make_box(tmp_path)
    d = os.path.join(b.run, SB.X35, "exp35_phase_ii_20261101T000000")
    os.mkdir(d)
    open(os.path.join(d, "state.json"), "w").write("{")
    rc, msg = _reserve(b)
    assert rc == 1 and "unresolved" in msg and not os.path.lexists(b.res())
