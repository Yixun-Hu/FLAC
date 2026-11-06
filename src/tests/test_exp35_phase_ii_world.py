"""exp_35 phase (ii) kit, part 2a (plan_orbitrir_yaw_evidence.md §15 items 4, 6, 9; Codex plan §15 r1 S3, S6): the
read-only world the kit observes -- nvidia-smi's index <-> uuid map and compute table (every failure unreadable), the
lock probes (never created, never held past the probe), the process scans (a supervisor or a runner anywhere, our
detached wrapper, an evaluator in the run worktree), his s43 table-job rule for GPU 1, and complete /proc identities
(PID reuse = another process). All on sandbox paths and a fake /proc tree.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import os
import subprocess
import sys

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import GPU0, GPU1, MAP, cfg_for, fake_proc

K = SB.load_kit()


# --------------------------------------------------------------------------- #
# nvidia-smi: the tables and the real-world query (fail closed)
# --------------------------------------------------------------------------- #
def test_the_gpu_map_and_the_compute_table_parse():
    assert K.gpu_map(MAP) == {0: GPU0, 1: GPU1}
    assert K.compute_apps(f"4242, {GPU1}\n17, {GPU0}\n\n") == [(4242, GPU1), (17, GPU0)] and K.compute_apps("") == []


@pytest.mark.parametrize("text", ["", "0, GPU-x\n0, GPU-y\n", "0, GPU-x\n1, GPU-x\n", "zero, GPU-x\n", "0 GPU-x\n"])
def test_a_malformed_gpu_map_is_refused(text):
    with pytest.raises(ValueError):
        K.gpu_map(text)


@pytest.mark.parametrize("text", ["4242\n", "x, GPU-a\n", "1, 2, 3\n"])
def test_a_malformed_compute_table_is_refused(text):
    with pytest.raises(ValueError):
        K.compute_apps(text)


def _smi(tmp_path, body):
    p = tmp_path / "nvidia-smi"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(0o755)
    return str(p)


def test_the_world_queries_nvidia_smi_and_every_failure_is_unreadable(tmp_path):
    w = K.World(_smi(tmp_path, f"case \"$1\" in --query-gpu=*) printf '%s' '{MAP}';; *) echo '4242, {GPU1}';; esac\n"))
    assert w.nvsmi("map") == MAP and w.nvsmi("apps") == f"4242, {GPU1}\n"
    with pytest.raises(K.Unreadable):
        K.World(_smi(tmp_path, "echo boom >&2; exit 9\n")).nvsmi("map")
    with pytest.raises(K.Unreadable):
        K.World(str(tmp_path / "absent")).nvsmi("apps")
    with pytest.raises(ValueError):
        w.nvsmi("--kill")


# --------------------------------------------------------------------------- #
# lock probes: absent / unheld / held, never created, nothing outlives the probe
# --------------------------------------------------------------------------- #
def test_the_lock_probe_never_creates_and_sees_a_holder(tmp_path):
    lock = str(tmp_path / "x.lock")
    assert K.lock_status(lock) == "absent" and not os.path.exists(lock)
    open(lock, "w").close()
    assert K.lock_status(lock) == "unheld" and K.lock_status(lock) == "unheld"
    p = subprocess.Popen([sys.executable, "-c", SB.HOLDER, lock, "30"], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        assert K.lock_status(lock) == "held"
    finally:
        p.kill()
        p.wait()
    assert K.lock_status(lock) == "unheld"


def test_an_unreadable_lock_is_unreadable(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("")
    lock.chmod(0o000)
    try:
        with pytest.raises(K.Unreadable):
            K.lock_status(str(lock))
    finally:
        lock.chmod(0o644)


# --------------------------------------------------------------------------- #
# the process scans (S6: "absence of live phase-(i) identities"; release: no live runner / wrapper / child)
# --------------------------------------------------------------------------- #
def test_the_scan_finds_every_exp35_process_and_nothing_else(tmp_path):
    cfg = cfg_for(tmp_path)
    proc = cfg.scan_proc
    fake_proc(proc, 101, ["/py", "/x/exp35_supervisor.py", "--h35", "a" * 40])
    fake_proc(proc, 102, ["bash", "worklog/x/depth_canon_eval_run.sh", "--blocks", "OFF-P1"], cwd="/elsewhere")
    fake_proc(proc, 103, ["/py", "-B", "/kit/exp35_phase_ii.py", "wrap", "--orch", "/o", "--lock-fd", "9"])
    fake_proc(proc, 104, ["/py", "overlays/no_flash/exp35_no_flash_launch.py", "eval_FLAC.py"], cwd=cfg.run)
    fake_proc(proc, 105, ["/py", "eval_FLAC.py", "--seed", "43"], cwd="/home/yixunhu/codespace/flac-fork-attn")
    fake_proc(proc, 106, ["/py", "-B", "/kit/exp35_phase_ii.py", "status"])               # a status: not a wrapper
    fake_proc(proc, 107, ["/py", "depth_canon_cond_probe.py", "--arm", "BF"], cwd=os.path.join(cfg.run, "sub"))
    fake_proc(proc, os.getpid(), ["/py", "/x/exp35_supervisor.py"])                       # this process: never
    os.makedirs(os.path.join(proc, "108"))                                               # vanished: skipped
    os.makedirs(os.path.join(proc, "self"))
    assert [p for p, _ in K.live_procs(cfg)] == [101, 102, 103, 104, 107]


def test_an_unreadable_cmdline_or_cwd_is_unreadable(tmp_path):
    cfg = cfg_for(tmp_path)
    d = fake_proc(cfg.scan_proc, 201, ["/py", "eval_FLAC.py"], cwd=cfg.run)
    os.unlink(os.path.join(d, "cwd"))
    os.makedirs(os.path.join(d, "cwd"))                                    # readlink fails with EINVAL, not ENOENT
    with pytest.raises(K.Unreadable):
        K.live_procs(cfg)
    os.rmdir(os.path.join(d, "cwd"))
    os.unlink(os.path.join(d, "cmdline"))
    os.makedirs(os.path.join(d, "cmdline"))                                # IsADirectoryError
    with pytest.raises(K.Unreadable):
        K.live_procs(cfg)


def test_the_table_job_rule_for_gpu1(tmp_path):
    cfg = cfg_for(tmp_path)
    assert any("EVAL_DONE" in x for x in K.table_job_reasons(cfg))                   # s43 ended, its table job not
    with open(cfg.queue_log, "a") as f:
        f.write(SB.EVAL_DONE_S43)
    assert K.table_job_reasons(cfg) == []
    fake_proc(cfg.scan_proc, 301, ["bash", "worklog/x/exp30_attn_eval.sh", "1", "43", "ckpt"])
    assert any("exp30_attn_eval.sh" in x for x in K.table_job_reasons(cfg))
    os.unlink(cfg.queue_log)
    with pytest.raises(K.Unreadable):
        K.table_job_reasons(cfg)


# --------------------------------------------------------------------------- #
# complete identities; PID reuse is another process (never a pid alone)
# --------------------------------------------------------------------------- #
def test_a_live_identity_is_complete_and_matches_only_its_own_process(tmp_path):
    proc = str(tmp_path / "proc")
    fake_proc(proc, 401, ["bash", "run.sh"], start=5555, cwd="/run")
    ident = K.proc_identity(401, proc)
    assert {k: ident[k] for k in K.ID_KEYS} == {"pid": 401, "start_time": 5555, "cwd": "/run",
                                                "cmdline_sha256": K.sha256_bytes(b"bash\0run.sh\0")}
    assert K.identity_status(ident, proc) == "alive"
    assert K.identity_status(dict(ident, pid=402), proc) == "gone"
    assert K.identity_status(dict(ident, start_time=5554), proc) == "reused"              # same pid, another process
    with pytest.raises(ValueError):
        K.identity_status({"pid": 401}, proc)                                             # incomplete: no identity


def test_an_identity_of_this_very_process_reads_through_the_real_proc():
    me = K.proc_identity(os.getpid(), "/proc")
    assert me["pid"] == os.getpid() and K.identity_status(me, "/proc") == "alive"
