"""exp_22 A7 (plan §9 A7 r2 (v), r3 (b) (h)): the READ-ONLY inputs of the card rule -- nvidia-smi tables,
reservations, the exp_35 lock probe -- on tmp_path fakes only (never the live reservation directory or lock)."""
import fcntl
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, HOLDER, MAP, cfg_for

# --------------------------------------------------------------------------- #
# nvidia-smi tables
# --------------------------------------------------------------------------- #
def test_gpu_map_and_compute_apps_parse_the_csv_forms():
    assert C.gpu_map(MAP) == {0: GPU0, 1: GPU1}
    assert C.compute_apps(f"3027719, {GPU0}\n740388, {GPU1}\n") == [(3027719, GPU0), (740388, GPU1)]
    assert C.compute_apps("") == [] and C.compute_apps("\n") == []
    for bad in ("", "0, GPU-x\n0, GPU-y\n", "zero, GPU-x\n", f"0, {GPU0}\n1, {GPU0}\n"):
        with pytest.raises(ValueError):
            C.gpu_map(bad)
    for bad in ("No running processes found\n", "[N/A], GPU-x\n", f"12, {GPU0}, extra\n"):
        with pytest.raises(ValueError):
            C.compute_apps(bad)


def test_the_real_world_wraps_every_nvidia_smi_failure_as_unreadable(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\ncase \"$1\" in --query-gpu=*) echo '0, GPU-a';; *) exit 9;; esac\n")
    fake.chmod(0o755)
    w = C.World(str(fake))
    assert w.nvsmi("map") == "0, GPU-a\n"
    with pytest.raises(C.Unreadable, match="rc 9"):
        w.nvsmi("apps")
    with pytest.raises(C.Unreadable):
        C.World(str(tmp_path / "absent")).nvsmi("map")
    with pytest.raises(ValueError):
        w.nvsmi("anything-else")


# --------------------------------------------------------------------------- #
# reservations (read only)
# --------------------------------------------------------------------------- #
def test_any_reservation_file_makes_the_card_reserved_whoever_holds_it(tmp_path):
    cfg = cfg_for(tmp_path)
    assert C.reservation_status(cfg.reservations, GPU0) == "absent"
    p = tmp_path / "reservations" / f"{GPU0}.json"
    p.write_text(json.dumps({"holder": "exp35", "h35": "x"}))
    assert C.reservation_status(cfg.reservations, GPU0) == "present (holder exp35)"
    p.write_text("{not json")
    assert C.reservation_status(cfg.reservations, GPU0) == "present (malformed)"
    p.write_text(json.dumps(["no", "holder"]))
    assert C.reservation_status(cfg.reservations, GPU0) == "present (malformed)"
    p.unlink()
    os.symlink(tmp_path / "nowhere", p)                       # a dangling link is a present file
    assert C.reservation_status(cfg.reservations, GPU0).startswith("present")


def test_an_unreadable_reservation_file_or_directory_is_unreadable(tmp_path):
    cfg = cfg_for(tmp_path)
    p = tmp_path / "reservations" / f"{GPU1}.json"
    p.write_text(json.dumps({"holder": "exp22"}))
    p.chmod(0)
    try:
        with pytest.raises(C.Unreadable, match="reservation"):
            C.reservation_status(cfg.reservations, GPU1)
    finally:
        p.chmod(0o644)
    with pytest.raises(C.Unreadable):
        C.reservation_status(str(tmp_path / "no-such-dir"), GPU1)


# --------------------------------------------------------------------------- #
# the exp_35 lock, probed read-only
# --------------------------------------------------------------------------- #
def test_the_lock_probe_tells_absent_unheld_held_and_unreadable(tmp_path):
    lock = tmp_path / "exp35_supervisor_state.json.lock"
    assert C.lock_status(str(lock)) == "absent" and not lock.exists()     # never created by the probe
    lock.write_text("")
    assert C.lock_status(str(lock)) == "unheld"                          # a leftover file: no supervisor
    p = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        assert C.lock_status(str(lock)) == "held"
    finally:
        p.kill()
        p.wait()
    assert C.lock_status(str(lock)) == "unheld"
    lock.chmod(0)
    try:
        with pytest.raises(C.Unreadable, match="lock"):
            C.lock_status(str(lock))
    finally:
        lock.chmod(0o644)


def test_the_lock_probe_takes_no_lock_that_outlives_it(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("")
    assert C.lock_status(str(lock)) == "unheld"
    fd = os.open(str(lock), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                   # an exp_35 supervisor could still start
    finally:
        os.close(fd)


def test_the_registered_config_is_the_plan_literals():
    cfg = C.registered_config()
    assert cfg.reservations == "/home/yixunhu/codespace/.gpu_reservations"
    assert cfg.exp35_lock == ("/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/"
                              "exp_35_orbitrir_yaw_evidence_claude/exp35_supervisor_state.json.lock")
    assert cfg.queue_log == ("/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/"
                             "queue_20261101T233607_741947.log")
    assert cfg.proc == "/proc" and cfg.nvidia_smi == "nvidia-smi"
    assert C.CARDS == {"GPU0": "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e",
                       "GPU1": "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"}
    assert (C.POLL_S, C.WINDOW_POLLS) == (60, 31) and C.EVAL_DONE == "EVAL_DONE ACEattn s43 "
