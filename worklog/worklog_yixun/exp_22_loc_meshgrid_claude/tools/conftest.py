"""exp_22 A7 tools tests (CPU only): the tools are imported by path, never written as bytecode, and every
test uses tmp_path fakes -- never the live reservation directory, the live exp_35 lock or a GPU."""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    """The tool module ``name`` (a file beside this conftest), imported fresh under a private name."""
    spec = importlib.util.spec_from_file_location(f"_a7_{name}", os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load("exp22_r2c_common")
GPU0, GPU1 = C.CARDS["GPU0"], C.CARDS["GPU1"]
MAP = f"0, {GPU0}\n1, {GPU1}\n"


class FakeWorld:
    """nvidia-smi answers from dicts (a callable answer may raise); a fake clock."""

    def __init__(self, gpu_map=MAP, apps="", t=1_800_000_000.0):
        self.answers = {"map": gpu_map, "apps": apps}
        self.t = t
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s

    def nvsmi(self, query):
        a = self.answers[query]
        return a() if callable(a) else a


def cfg_for(tmp_path, **kw):
    res = tmp_path / "reservations"
    res.mkdir(exist_ok=True)
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    q = tmp_path / "queue.log"
    if not q.exists():
        q.write_text("2026-11-04T06:12:55-05:00 START training ACEattn s43 gpus 0,1\n")
    out = dict(reservations=str(res), exp35_lock=str(tmp_path / "exp35_supervisor_state.json.lock"),
               queue_log=str(q), proc=str(proc), nvidia_smi="nvidia-smi")
    out.update(kw)
    return C.Config(**out)


def fake_proc(root, pid, argv, ppid=1, start=1000, cwd="/somewhere"):
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 10
    (d / "stat").write_text(f"{pid} ({os.path.basename(argv[0])}) " + " ".join(fields) + "\n")
    os.symlink(cwd, d / "cwd")
    os.symlink("/dev/null", d / "fd" / "1")
    os.symlink("/dev/null", d / "fd" / "2")
    return d


HOLDER = r'''
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(60)
'''
