"""exp_35 phase (ii) kit, part 2b (plan_orbitrir_yaw_evidence.md §15 items 6, 9; Codex plan §15 r1 S3, S6): the
O_EXCL reservation records (holder exp35, phase ii) with their exact-bytes + device / inode authentication -- a
record that is not ours is never unlinked -- and the one card observation every mode makes (index <-> uuid, compute
processes, reservation files of any holder, his table job on GPU 1, the locks, the live exp_35 processes).

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import subprocess
import sys

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, GPU1, MAP, FakeWorld, cfg_for, fake_proc

K = SB.load_kit()


# --------------------------------------------------------------------------- #
# the O_EXCL reservation records (§15 item 6: holder exp35, phase ii, H35, uuid + index, created, run)
# --------------------------------------------------------------------------- #
def test_a_reservation_is_an_exact_o_excl_record(tmp_path):
    cfg = cfg_for(tmp_path)
    held = K.create_reservation(cfg, 0, h35=cfg.h35, run="20261106T143000", now=1.5)
    raw = open(held["path"], "rb").read()
    doc = json.loads(raw)
    assert list(doc) == list(K.RES_KEYS) == ["holder", "phase", "h35", "gpu_uuid", "gpu_index", "created", "run"]
    assert doc == {"holder": "exp35", "phase": "ii", "h35": cfg.h35, "gpu_uuid": GPU0, "gpu_index": 0, "created": 1.5,
                   "run": "20261106T143000"}
    st = os.stat(held["path"])
    assert held == {"index": 0, "uuid": GPU0, "path": os.path.join(cfg.reservations, f"{GPU0}.json"),
                    "sha256": K.sha256_bytes(raw), "dev": st.st_dev, "inode": st.st_ino}
    assert K.held_reasons(held) == [] and K.reservation_status(cfg, GPU0) == "present (holder exp35)"
    with pytest.raises(FileExistsError):
        K.create_reservation(cfg, 0, h35=cfg.h35, run="20261106T143001", now=2.0)
    assert open(held["path"], "rb").read() == raw


@pytest.mark.parametrize("tamper", ["bytes", "inode", "symlink", "missing"])
def test_a_record_that_is_not_ours_is_never_unlinked(tmp_path, tamper):
    cfg = cfg_for(tmp_path)
    held = K.create_reservation(cfg, 0, h35=cfg.h35, run="20261106T143000", now=1.0)
    raw = open(held["path"], "rb").read()
    if tamper == "bytes":
        open(held["path"], "wb").write(raw.replace(b'"phase": "ii"', b'"phase": "i"'))
    elif tamper == "inode":                                                    # identical bytes, another file
        open(held["path"] + ".new", "wb").write(raw)
        os.replace(held["path"] + ".new", held["path"])
    elif tamper == "symlink":
        (tmp_path / "elsewhere.json").write_bytes(raw)
        os.unlink(held["path"])
        os.symlink(str(tmp_path / "elsewhere.json"), held["path"])
    else:
        os.unlink(held["path"])
    assert K.held_reasons(held) and K.release_reservation(held)
    assert os.path.lexists(held["path"]) == (tamper != "missing")


def test_our_exact_record_is_released_and_the_directory_synced(tmp_path):
    cfg = cfg_for(tmp_path)
    held = K.create_reservation(cfg, 1, h35=cfg.h35, run="20261106T143000", now=1.0)
    assert K.release_reservation(held) == [] and not os.path.lexists(held["path"])
    assert K.release_reservation(held)                                         # gone: refused, nothing to unlink


def test_a_record_replaced_between_its_authentication_and_the_unlink_is_never_unlinked(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    held = K.create_reservation(cfg, 0, h35=cfg.h35, run="20261106T143000", now=1.0)
    real = K.held_reasons

    def racing(h):                                   # authenticated, then replaced (same bytes, another inode)
        why = real(h)
        raw = open(h["path"], "rb").read()
        open(h["path"] + ".new", "wb").write(raw)
        os.replace(h["path"] + ".new", h["path"])
        return why
    monkeypatch.setattr(K, "held_reasons", racing)
    assert any("replaced while it was authenticated" in x for x in K.release_reservation(held))
    assert os.path.lexists(held["path"])


@pytest.mark.parametrize("content, status", [(None, "absent"), (b'{"holder": "exp22", "run": "x"}', "present (holder exp22)"),
                                             (b"{", "present (malformed)"), (b"[1]", "present (malformed)"),
                                             ("dangling", "present (a dangling link, or removed while it was read)")])
def test_the_reservation_status_of_any_file(tmp_path, content, status):
    cfg = cfg_for(tmp_path)
    p = os.path.join(cfg.reservations, f"{GPU1}.json")
    if content == "dangling":
        os.symlink(str(tmp_path / "nowhere"), p)
    elif content is not None:
        open(p, "wb").write(content)
    assert K.reservation_status(cfg, GPU1) == status


def test_an_unreadable_reservation_directory_is_unreadable(tmp_path):
    cfg = cfg_for(tmp_path, reservations=str(tmp_path / "absent-dir"))
    with pytest.raises(K.Unreadable):
        K.reservation_status(cfg, GPU0)


# --------------------------------------------------------------------------- #
# the card observation (§15 item 6; S6 "process / reservation observations, index <-> uuid each time")
# --------------------------------------------------------------------------- #
def _locks(cfg):
    for rel in (K.SUP_LOCK, K.RUNNER_LOCK):
        os.makedirs(os.path.dirname(os.path.join(cfg.run, rel)), exist_ok=True)
        open(os.path.join(cfg.run, rel), "w").close()


def test_a_free_gpu0_lane_is_observed_free(tmp_path):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    o = K.observe(FakeWorld(apps=f"777, {GPU1}\n"), cfg, (0,))                         # his table job on GPU 1
    assert o["reasons"] == [] and o["map"] == {"0": GPU0, "1": GPU1} and o["cards"]["0"]["pids"] == []
    assert o["apps_sha256"] == K.sha256_bytes(f"777, {GPU1}\n".encode()) and o["locks"] == {
        "supervisor": "unheld", "runner": "unheld"}


@pytest.mark.parametrize("case, reason", [
    ("busy", "compute process"), ("reserved-exp22", "reservation file exists"), ("reserved-malformed", "reservation"),
    ("exp35-other-card", "exp35 reservation"), ("malformed-other-card", "unexplained"), ("swapped-map", "registered"),
    ("third-card", "registered"), ("unknown-uuid", "unknown"), ("supervisor-proc", "exp35_supervisor.py"),
    ("runner-proc", "depth_canon_eval_run.sh"), ("gpu1-table-job", "EVAL_DONE")])
def test_every_occupied_or_inconsistent_observation_is_refused(tmp_path, case, reason):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    w, lanes = FakeWorld(), (0,)
    res = lambda u: os.path.join(cfg.reservations, f"{u}.json")  # noqa: E731
    if case == "busy":
        w.answers["apps"] = f"4242, {GPU0}\n"
    elif case == "reserved-exp22":
        open(res(GPU0), "w").write(json.dumps({"holder": "exp22", "run": "x"}))
    elif case == "reserved-malformed":
        open(res(GPU0), "w").write("{")
    elif case == "exp35-other-card":
        open(res(GPU1), "w").write(json.dumps({"holder": "exp35", "h35": "a" * 40}))     # phase (i)'s, left over
    elif case == "malformed-other-card":
        open(res(GPU1), "w").write("not json")
    elif case == "swapped-map":
        w.answers["map"] = f"0, {GPU1}\n1, {GPU0}\n"
    elif case == "third-card":
        w.answers["map"] = MAP + "2, GPU-ffff\n"
    elif case == "unknown-uuid":
        w.answers["apps"] = "5, GPU-ffff\n"
    elif case == "supervisor-proc":
        fake_proc(cfg.scan_proc, 99, ["/py", "/x/exp35_supervisor.py"])
    elif case == "runner-proc":
        fake_proc(cfg.scan_proc, 98, ["bash", "x/depth_canon_eval_run.sh"])
    else:
        lanes = (0, 1)
    o = K.observe(w, cfg, lanes)
    assert o["reasons"] and any(reason in x for x in o["reasons"]), o["reasons"]


def test_an_exp22_record_on_the_other_card_is_no_obstacle(tmp_path):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    open(os.path.join(cfg.reservations, f"{GPU1}.json"), "w").write(json.dumps({"holder": "exp22", "run": "r"}))
    assert K.observe(FakeWorld(), cfg, (0,))["reasons"] == []


def test_a_gpu1_lane_needs_the_table_job_terminally_complete(tmp_path):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    with open(cfg.queue_log, "a") as f:
        f.write(SB.EVAL_DONE_S43)
    assert K.observe(FakeWorld(), cfg, (0, 1))["reasons"] == []


@pytest.mark.parametrize("which", ["supervisor", "runner"])
def test_a_held_supervisor_or_runner_lock_refuses(tmp_path, which):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    lock = os.path.join(cfg.run, K.SUP_LOCK if which == "supervisor" else K.RUNNER_LOCK)
    p = subprocess.Popen([sys.executable, "-c", SB.HOLDER, lock, "30"], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        o = K.observe(FakeWorld(), cfg, (0,))
    finally:
        p.kill()
        p.wait()
    assert any(f"the {which} lock is held" in x for x in o["reasons"]) and o["locks"][which] == "held"


def test_with_held_records_the_lane_file_must_be_our_exact_record(tmp_path):
    cfg = cfg_for(tmp_path)
    _locks(cfg)
    held = K.create_reservation(cfg, 0, h35=cfg.h35, run="20261106T143000", now=1.0)
    assert K.observe(FakeWorld(), cfg, (0,), held=[held])["reasons"] == []
    assert any("reservation file exists" in x for x in K.observe(FakeWorld(), cfg, (0,))["reasons"])
    os.unlink(held["path"])
    open(held["path"], "w").write(json.dumps({"holder": "exp22"}))
    assert any("not our record" in x for x in K.observe(FakeWorld(), cfg, (0,), held=[held])["reasons"])


@pytest.mark.parametrize("bad", ["map", "apps"])
def test_a_malformed_table_is_unreadable(tmp_path, bad):
    cfg = cfg_for(tmp_path)
    w = FakeWorld()
    w.answers[bad] = "garbage\n"
    with pytest.raises(K.Unreadable):
        K.observe(w, cfg, (0,))
