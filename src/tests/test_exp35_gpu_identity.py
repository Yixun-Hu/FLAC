"""exp_35 §12 B1.1-B1.2: the approved co-tenant identity tuple (``exp35_gpu_identity.py``) -- PID, start
time (/proc stat field 22), cwd, cmdline sha256 and the fd/1, fd/2 links, read mechanically from /proc.

A fake /proc tree under tmp_path drives the mismatch cases; one case reads this test's own real process.
Nothing is signalled, nothing outside tmp_path is written.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r3 §12.
"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDENTITY = os.path.join(REPO, "worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude",
                        "exp35_gpu_identity.py")


def _load():
    spec = importlib.util.spec_from_file_location("exp35_gpu_identity", IDENTITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load()
UUID = "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"
CMDLINE = b"/home/yixunhu/miniconda3/envs/flac/bin/python\x00train.py\x00--seed\x0043\x00"


def fake_proc(root, pid, *, start=887766, cwd="/home/yixunhu/codespace/flac-fork-attn", cmdline=CMDLINE,
              fd1="/home/yixunhu/codespace/flac-fork-attn/logs/train_s43.log", fd2=None, comm="pt_main (thr) 1"):
    """One /proc/<pid> entry: stat (a comm with spaces and parentheses), cmdline, cwd and fd links."""
    base = root / str(pid)
    if base.exists():
        shutil.rmtree(base)
    (base / "fd").mkdir(parents=True)
    fields = ["S"] + ["7"] * 18 + [str(start)] + ["0"] * 30          # fields 3 .. 52; field 22 = start
    (base / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
    (base / "cmdline").write_bytes(cmdline)
    os.symlink(cwd, base / "cwd")
    os.symlink(fd1, base / "fd" / "1")
    os.symlink(fd2 or fd1, base / "fd" / "2")
    return root


@pytest.fixture()
def proc(tmp_path):
    root = tmp_path / "proc"
    fake_proc(root, 3028196)
    return root


def _tuple(proc, *pids):
    return {"schema_version": 1, "gpu_uuid": UUID, "processes": [G.proc_identity(p, str(proc)) for p in pids]}


def test_the_identity_is_read_from_proc(proc):
    ident = G.proc_identity(3028196, str(proc))
    assert ident == {"pid": 3028196, "start_time": 887766, "cwd": "/home/yixunhu/codespace/flac-fork-attn",
                     "cmdline_sha256": hashlib.sha256(CMDLINE).hexdigest(),
                     "fd1": "/home/yixunhu/codespace/flac-fork-attn/logs/train_s43.log",
                     "fd2": "/home/yixunhu/codespace/flac-fork-attn/logs/train_s43.log"}
    assert list(ident) == list(G.FIELDS)
    with pytest.raises(ValueError, match="unreadable"):
        G.proc_identity(4242, str(proc))                                     # gone


def test_the_identity_of_a_real_process():
    ident = G.proc_identity(os.getpid())
    with open("/proc/self/stat") as f:
        stat = f.read()
    assert ident["start_time"] == int(stat[stat.rindex(")") + 2:].split()[19])
    assert ident["cwd"] == os.path.realpath(os.getcwd())
    with open("/proc/self/cmdline", "rb") as f:
        assert ident["cmdline_sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert ident["fd1"] == os.readlink("/proc/self/fd/1") and ident["fd2"] == os.readlink("/proc/self/fd/2")


@pytest.mark.parametrize("swap", ["start_time", "pid"])
def test_an_entry_that_changes_during_the_sample_is_refused(proc, monkeypatch, swap):
    """Codex C2 HIGH 1: stat is read before AND after cmdline / cwd / fd; a PID reused in between (its
    entry rewritten after the first stat read) never yields a mixed identity."""
    real, state = os.readlink, {"done": False}

    def readlink(path, *a, **kw):
        if not state["done"]:                              # first link read: the process is replaced NOW
            state["done"] = True
            stat = proc / "3028196" / "stat"
            text = stat.read_text()
            new = text.replace(" 887766 ", " 999999 ") if swap == "start_time" else text.replace(
                "3028196 (", "3028197 (", 1)
            assert new != text
            stat.write_text(new)
        return real(path, *a, **kw)
    monkeypatch.setattr(G.os, "readlink", readlink)
    with pytest.raises(ValueError, match="changed during"):
        G.proc_identity(3028196, str(proc))


def test_the_exact_tuple_passes(proc):
    assert G.check_reasons([3028196], UUID, _tuple(proc, 3028196), str(proc)) == []


MISMATCH = {   # (observed pids, gpu uuid, change to the fake /proc after the tuple was frozen)
    "extra_process": ([3028196, 4100000], UUID, None),
    "trainer_exited": ([], UUID, lambda p: shutil.rmtree(p / "3028196")),
    "replaced_by_table_job": ([4100000], UUID, lambda p: fake_proc(p, 4100000, cmdline=b"python\x00eval.py\x00")),
    "pid_reuse_new_start_time": ([3028196], UUID, lambda p: fake_proc(p, 3028196, start=999999)),
    "pid_reuse_other_cmdline": ([3028196], UUID, lambda p: fake_proc(p, 3028196, cmdline=b"python\x00x.py\x00")),
    "other_cwd": ([3028196], UUID, lambda p: fake_proc(p, 3028196, cwd="/tmp")),
    "other_log": ([3028196], UUID, lambda p: fake_proc(p, 3028196, fd1="/tmp/other.log")),
    "other_stderr": ([3028196], UUID, lambda p: fake_proc(p, 3028196, fd2="/dev/null")),
    "other_gpu": ([3028196], "GPU-00000000-aaaa", None),
    "duplicate_observation": ([3028196, 3028196], UUID, None),
}


@pytest.mark.parametrize("name", list(MISMATCH))
def test_any_extra_missing_or_changed_process_is_a_mismatch(proc, name):
    observed, uuid, change = MISMATCH[name]
    approved = _tuple(proc, 3028196)
    if change:
        change(proc)
    assert G.check_reasons(observed, uuid, approved, str(proc)), name


MALFORMED = {
    "keys": lambda d: d.pop("gpu_uuid"), "schema": lambda d: d.update(schema_version=2),
    "uuid": lambda d: d.update(gpu_uuid="6ac2604a"), "empty": lambda d: d.update(processes=[]),
    "pid_type": lambda d: d["processes"][0].update(pid="3028196"),
    "start_type": lambda d: d["processes"][0].update(start_time=1.5),
    "cmdline_hex": lambda d: d["processes"][0].update(cmdline_sha256="x" * 64),
    "field_missing": lambda d: d["processes"][0].pop("fd2"),
    "duplicate_pid": lambda d: d["processes"].append(copy.deepcopy(d["processes"][0])),
}


@pytest.mark.parametrize("name", list(MALFORMED))
def test_a_malformed_tuple_is_refused(proc, tmp_path, name):
    doc = _tuple(proc, 3028196)
    assert G.tuple_reasons(doc) == []
    MALFORMED[name](doc)
    assert G.tuple_reasons(doc), name
    path = tmp_path / "t.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="malformed"):
        G.load_tuple(str(path))


def _cli(*args):
    p = subprocess.run([sys.executable, IDENTITY, *args], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return p.returncode, p.stdout + p.stderr


def test_cli_snapshot_validate_and_check(tmp_path):
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        rc, out = _cli("--snapshot", "--gpu-uuid", UUID, "--pid", str(sleeper.pid))
        assert rc == 0, out
        doc = json.loads(out)
        assert doc["processes"][0]["pid"] == sleeper.pid and G.tuple_reasons(doc) == []
        path = tmp_path / "approved.json"
        path.write_text(out)
        assert _cli("--validate", str(path))[0] == 0
        assert _cli("--check", "--approved", str(path), "--gpu-uuid", UUID, "--pids", str(sleeper.pid))[0] == 0
        rc, out = _cli("--check", "--approved", str(path), "--gpu-uuid", UUID, "--pids", f"{sleeper.pid},1")
        assert rc == 1 and "IDENTITY MISMATCH" in out
        rc, out = _cli("--check", "--approved", str(path), "--gpu-uuid", UUID, "--pids", "")
        assert rc == 1 and "missing" in out
        (tmp_path / "bad.json").write_text("{}")
        assert _cli("--check", "--approved", str(tmp_path / "bad.json"), "--gpu-uuid", UUID, "--pids", "1")[0] == 2
        assert _cli("--validate", str(tmp_path / "absent.json"))[0] == 2
    finally:
        sleeper.kill()                                 # our own child only
        sleeper.wait()
    rc, out = _cli("--check", "--approved", str(path), "--gpu-uuid", UUID, "--pids", str(sleeper.pid))
    assert rc == 1 and "unreadable" in out                                  # the approved process is gone
