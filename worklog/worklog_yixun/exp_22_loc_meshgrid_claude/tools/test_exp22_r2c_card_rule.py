"""exp_22 A7 (plan §9 A7 r2 (v)(1), r3 (h)): /proc (processes by name, identity, descendants), Yixun's table-job
rule and the card rule itself, on tmp_path fakes only."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, HOLDER, MAP, FakeWorld, cfg_for, fake_proc

# --------------------------------------------------------------------------- #
# /proc: processes by name, identity, descendants
# --------------------------------------------------------------------------- #
def test_processes_are_found_by_the_exact_basename_of_an_argv_element(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_proc(proc, 11, ["bash", "/kit/exp30_attn_eval.sh", "s43"])
    fake_proc(proc, 12, ["bash", "/kit/exp30_attn_eval.sh.bak"])
    fake_proc(proc, 13, ["python", "x.py", "--note", "exp30_attn_eval.sh-like"])
    (proc / "14").mkdir()                                                 # vanished between listdir and read
    (proc / "self").mkdir()
    assert C.procs_named("exp30_attn_eval.sh", str(proc)) == [11]
    fake_proc(proc, 15, ["/bin/sh", "-c", "exp30_attn_eval.sh"])
    assert C.procs_named("exp30_attn_eval.sh", str(proc)) == [11, 15]


def test_an_unreadable_cmdline_is_unreadable_not_skipped(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    d = fake_proc(proc, 21, ["bash", "/kit/exp30_attn_eval.sh"])
    (d / "cmdline").chmod(0)
    try:
        with pytest.raises(C.Unreadable, match="21"):
            C.procs_named("exp30_attn_eval.sh", str(proc))
    finally:
        (d / "cmdline").chmod(0o644)


def test_proc_identity_binds_pid_start_time_cwd_and_cmdline(tmp_path):
    me = C.proc_identity(os.getpid())
    with open(f"/proc/{os.getpid()}/cmdline", "rb") as f:
        assert me["cmdline_sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert me["pid"] == os.getpid() and me["cwd"] == os.getcwd() and type(me["start_time"]) is int
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_proc(proc, 31, ["python", "a.py"], start=4242, cwd="/w")
    fake = C.proc_identity(31, str(proc))
    assert fake["start_time"] == 4242 and fake["cwd"] == "/w" and fake["fd1"] == "/dev/null"
    with pytest.raises(ValueError):
        C.proc_identity(32, str(proc))


def test_descendants_follow_the_ppid_tree(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_proc(proc, 40, ["python", "sup.py"], ppid=1)
    fake_proc(proc, 41, ["python", "child.py"], ppid=40)
    fake_proc(proc, 42, ["python", "worker"], ppid=41)
    fake_proc(proc, 43, ["other"], ppid=1)
    assert C.descendants(41, str(proc)) == {41, 42}
    assert C.descendants(99, str(proc)) == set()


# --------------------------------------------------------------------------- #
# the table-job rule (GPU 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, done", [
    ("EVAL_DONE ACEattn s44 cells=40 2026-11-04T16:01:40-05:00\n", False),
    ("EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n", True),
    ("x EVAL_DONE ACEattn s43 cells=40\n", False),
    ("EVAL_DONE ACEattn s430 cells=40\n", False),
    ("EVAL_DONE ACEattn s43\n", False),
])
def test_the_table_job_is_done_only_on_its_exact_eval_done_line(tmp_path, text, done):
    cfg = cfg_for(tmp_path)
    with open(cfg.queue_log, "a") as f:
        f.write(text)
    ok, why = C.table_job_status(cfg.queue_log, cfg.proc)
    assert ok is done and (why == [] if done else why)


def test_the_table_job_is_not_done_while_its_evaluator_runs_or_the_log_is_unreadable(tmp_path):
    cfg = cfg_for(tmp_path)
    with open(cfg.queue_log, "a") as f:
        f.write("EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n")
    fake_proc(tmp_path / "proc", 77, ["bash", "/kit/exp30_attn_eval.sh", "ACEattn", "s45"])
    ok, why = C.table_job_status(cfg.queue_log, cfg.proc)
    assert not ok and "77" in why[0]
    with pytest.raises(C.Unreadable, match="queue log"):
        C.table_job_status(str(tmp_path / "missing.log"), cfg.proc)


# --------------------------------------------------------------------------- #
# the card rule
# --------------------------------------------------------------------------- #
def test_an_empty_gpu0_with_no_reservation_and_no_exp35_lock_is_free(tmp_path):
    w, cfg = FakeWorld(apps=f"3028196, {GPU1}\n"), cfg_for(tmp_path)
    o = C.observe(w, cfg, "GPU0")
    assert o["free"] is True and o["reasons"] == [] and o["uuid"] == GPU0 and o["index"] == 0
    assert o["apps_raw"] == f"3028196, {GPU1}\n" and o["apps_sha256"] == hashlib.sha256(o["apps_raw"].encode()).hexdigest()
    assert o["gpu_map_raw"] == MAP and o["t"] == w.t and o["pids_on_card"] == []
    assert o["reservation"] == "absent" and o["rule"] == "exp35 lock absent"


def test_a_compute_process_or_any_reservation_makes_a_card_busy(tmp_path):
    cfg = cfg_for(tmp_path)
    o = C.observe(FakeWorld(apps=f"5, {GPU0}\n"), cfg, "GPU0")
    assert not o["free"] and o["pids_on_card"] == [5] and "compute process" in o["reasons"][0]
    (tmp_path / "reservations" / f"{GPU0}.json").write_text(json.dumps({"holder": "exp35"}))
    o = C.observe(FakeWorld(), cfg, "GPU0")
    assert not o["free"] and "exp_35" in " ".join(o["reasons"])          # a phase-(ii) intent / lane
    (tmp_path / "reservations" / f"{GPU0}.json").write_text(json.dumps({"holder": "exp22"}))
    assert not C.observe(FakeWorld(), cfg, "GPU0")["free"]


def test_gpu0_is_busy_while_the_exp35_lock_is_held_and_free_on_a_leftover(tmp_path):
    cfg = cfg_for(tmp_path)
    lock = tmp_path / "exp35_supervisor_state.json.lock"
    lock.write_text("")
    o = C.observe(FakeWorld(), cfg, "GPU0")
    assert o["free"] and o["rule"] == "exp35 lock unheld"
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        o = C.observe(FakeWorld(), cfg, "GPU0")
        assert not o["free"] and o["rule"] == "exp35 lock HELD" and "hand-over" in " ".join(o["reasons"])
        assert C.observe(FakeWorld(apps=""), cfg_for(tmp_path), "GPU1")["rule"] != "exp35 lock HELD"
    finally:
        p.kill()
        p.wait()


def test_gpu1_waits_for_the_table_job(tmp_path):
    cfg = cfg_for(tmp_path)
    o = C.observe(FakeWorld(), cfg, "GPU1")
    assert not o["free"] and "EVAL_DONE ACEattn s43" in " ".join(o["reasons"]) and o["index"] == 1
    with open(cfg.queue_log, "a") as f:
        f.write("EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n")
    assert C.observe(FakeWorld(), cfg, "GPU1")["free"]


@pytest.mark.parametrize("world", [
    FakeWorld(gpu_map=f"0, {GPU0}\n"),                                    # one card
    FakeWorld(gpu_map=f"0, {GPU0}\n1, GPU-ffffffff-0000-0000-0000-000000000000\n"),
    FakeWorld(apps="garbage\n"),
    FakeWorld(apps=lambda: (_ for _ in ()).throw(C.Unreadable("nvidia-smi --query-compute-apps: rc 9"))),
], ids=["one-card", "foreign-uuid", "malformed-apps", "nvsmi-fails"])
def test_an_unreadable_or_foreign_gpu_table_is_unreadable(tmp_path, world):
    with pytest.raises(C.Unreadable):
        C.observe(world, cfg_for(tmp_path), "GPU0")


def test_the_index_follows_the_uuid_not_the_name(tmp_path):
    o = C.observe(FakeWorld(gpu_map=f"0, {GPU1}\n1, {GPU0}\n"), cfg_for(tmp_path), "GPU0")
    assert o["uuid"] == GPU0 and o["index"] == 1


def test_an_unknown_card_name_is_refused(tmp_path):
    with pytest.raises(ValueError):
        C.observe(FakeWorld(), cfg_for(tmp_path), "GPU2")
