"""exp_22 A7 tools tests (CPU only): the tools are imported by path, never written as bytecode, and every
test uses tmp_path fakes -- never the live reservation directory, the live exp_35 lock or a GPU."""
import ast
import importlib.util
import os
import shlex
import shutil
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
        self.t += max(0.0, s)

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


BEGIN, END = "# --- BEGIN REGISTERED CONFIG", "# --- END REGISTERED CONFIG ---"
TOOLS = ("exp22_r2c_common.py", "exp22_r2c_evidence.py", "exp22_r2c_supervisor.py", "exp22_free_card_detect.sh",
         "exp22_r2c_preflight.sh", "exp22_r2c_preflight_lines.json")


def _block(text):
    i = text.index(BEGIN)
    j = text.index(END, i) + len(END)
    return i, j


def block_names(text, shell):
    """The names the registered-config block of ``text`` assigns."""
    i, j = _block(text)
    body = text[i:j]
    if shell:
        return sorted({ln.split("=", 1)[0] for ln in body.splitlines()[1:-1] if ln and not ln.startswith("#")})
    return sorted({t.id for node in ast.parse(body).body for t in
                   (node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets)})


def replace_block(text, values, shell):
    """``text`` with ONLY its registered-config block replaced by ``values`` -- which must cover exactly its names."""
    names = block_names(text, shell)
    assert sorted(values) == names, (sorted(values), names)
    i, j = _block(text)
    first = text[i:text.index("\n", i)]
    body = "".join(f"{k}={shlex.quote(str(v))}\n" if shell else f"{k} = {v!r}\n" for k, v in values.items())
    return text[:i] + first + "\n" + body + END + text[j:]


def sandbox_tools(root, common=None, shells=None):
    """Copies of the tools under ``root``'s worklog layout: ``exp22_r2c_common.py``'s config block replaced by
    ``common`` and each shell script's by ``shells[name]`` (exactly their names, nothing else changed)."""
    dest = os.path.join(str(root), "worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude", "tools")
    os.makedirs(dest, exist_ok=True)
    for name in TOOLS:
        src = os.path.join(HERE, name)
        if not os.path.exists(src):
            continue
        with open(src) as f:
            text = f.read()
        values = common if name == "exp22_r2c_common.py" else (shells or {}).get(name)
        if values is not None:
            text = replace_block(text, values, name.endswith(".sh"))
        with open(os.path.join(dest, name), "w") as f:
            f.write(text)
        shutil.copymode(src, os.path.join(dest, name))
    return dest
