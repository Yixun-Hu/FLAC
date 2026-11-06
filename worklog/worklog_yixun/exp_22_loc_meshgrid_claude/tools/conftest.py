"""exp_22 A7 tools tests (CPU only): the tools are imported by path, never written as bytecode, and every
test uses tmp_path fakes -- never the live reservation directory, the live exp_35 lock or a GPU."""
import ast
import fcntl
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import types

import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    """The tool module ``name`` (a file beside this conftest) under its canonical name -- ONE copy per process, as
    the tools load each other (``exp22_r2c_evidence.common()``), so their exception classes are shared."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


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


# the reservation fixtures shared by the reserve / planner tests (the supervisor loaded lazily)
RUN, PIN2 = "20261106T130501", "a" * 40


def setup(tmp_path, apps=""):
    cfg = cfg_for(tmp_path)
    w = tmp_path / "W"
    orch = w / "outputs_loc" / "exp22" / f"r2_preflight_{RUN}"
    orch.mkdir(parents=True)
    seal = orch / "seal.txt"
    seal.write_text(f"pin2 {PIN2}\nrun {RUN}\ncard GPU0 {GPU0} 0\n")
    return cfg, FakeWorld(apps=apps), w, orch, seal


def reserve(cfg, world, w, orch, seal, card="GPU0", index=0):
    sup, ev = load("exp22_r2c_supervisor"), load("exp22_r2c_evidence")
    return sup.reserve(world, cfg, worktree=str(w), orch=str(orch), pin2=PIN2, card=card, index=index,
                       seal=str(seal), seal_sha256=ev.sha256_bytes(seal.read_bytes()))


def res_file(cfg, uuid=GPU0):
    return os.path.join(cfg.reservations, f"{uuid}.json")


# --------------------------------------------------------------------------- #
# a sandbox CHECKOUT for the supervisor and the launcher (CPU, no torch): the real PIN2 files, a stand-in transformer
# carrying the real fallback block, stand-ins for localize_meshgrid.py, meshgrid_fa_parity and the eleven T tests,
# all driven by an untracked a7_control.json; git-committed, so HEAD = its "PIN2"
# --------------------------------------------------------------------------- #
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PIN_PATHS = ("overlays/no_flash/flash_attn.py", "overlays/no_flash/exp35_no_flash_launch.py",
             "src/tests/test_exp35_no_flash_launch.py", "pyproject.toml")
STANDIN_TARGET = r'''"""A7 sandbox stand-in for localize_meshgrid.py: the registered evidence name into --out-dir."""
import json, os, sys, time
a = sys.argv[1:]
if "--probe" in a:
    name = "diagnostics_BF_bathrooms14"
elif "--cache-parity-check" in a:
    name = "vanilla_cache_parity" if "--expect-arm" in a else "fa_cache_parity"
else:
    name = {"--fa-split-parity-check": "fa_split_parity", "--fa-production-parity-check": "fa_production_parity",
            "--fa-production-drift-measurement": "fa_production_drift",
            "--replay-check": "vanilla_replay_regression"}[next(x for x in a if x.endswith(("-check", "-measurement")))]
me = (json.load(open("a7_control.json")) if os.path.exists("a7_control.json") else {}).get(name, {})
time.sleep(me.get("sleep", 0))
out = a[a.index("--out-dir") + 1]
os.makedirs(out, exist_ok=True)
doc = ({"projection": {"projected_gpu_hours": {"total": me.get("total", 100.0)}}} if name.startswith("diagnostics")
       else {"name": name, "pass": me.get("pass", True)})
if not me.get("no_output"):
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(doc, f)
print("stand-in", name, "argv", json.dumps(a), flush=True)
for line in me.get("print", []):
    print(line, flush=True)
sys.exit(me.get("rc", 0))
'''
STANDIN_PARITY = 'def reverify_gate_evidence(doc):\n    return [] if doc.get("pass") else ["the stand-in evidence does not pass"]\n'
STANDIN_TEST_HEAD = '''import json, os, pytest


def _behave(name):
    kind = (json.load(open("a7_control.json")) if os.path.exists("a7_control.json") else {}).get("T", {}).get(name)
    if kind == "fail":
        assert False, "stand-in failure"
    if kind == "skip":
        pytest.skip("stand-in skip")


'''


def _fallback_block():
    with open(os.path.join(REPO, "src", "models", "transformer.py")) as f:
        text = f.read()
    start = text.index("try:\n    from flash_attn import flash_attn_func\n")
    end = text.index("    flash_attn_func = None\n", start) + len("    flash_attn_func = None\n")
    return text[start:end]


def _git(w, *args):
    return subprocess.run(["git", "-C", str(w), "-c", "user.email=a7@sandbox", "-c", "user.name=a7", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def sandbox_repo(root, common=None, shells=None):
    """``(W, pin2)``: the sandbox checkout under ``root``."""
    w = os.path.join(str(root), "W")
    for rel in PIN_PATHS:
        os.makedirs(os.path.dirname(os.path.join(w, rel)) or w, exist_ok=True)
        shutil.copy(os.path.join(REPO, rel), os.path.join(w, rel))
    files = {"src/__init__.py": "", "src/models/__init__.py": "", "src/models/transformer.py": _fallback_block(),
             "src/localization/__init__.py": "", "src/localization/meshgrid_fa_parity.py": STANDIN_PARITY,
             "localize_meshgrid.py": STANDIN_TARGET}
    ids = load("exp22_r2c_evidence").T_NODE_IDS
    for path in sorted({i.split("::")[0] for i in ids}):
        body, seen = STANDIN_TEST_HEAD, set()
        for nid in ids:
            p, name = nid.split("[")[0].split("::")
            if p != path or name in seen:
                continue
            seen.add(name)
            param = "[" in nid
            body += (("@pytest.mark.parametrize('n', [4, 64])\n" if param else "")
                     + f"def {name}({'n' if param else ''}):\n    _behave({name!r})\n\n\n")
        files[path] = body
    for rel, text in files.items():
        os.makedirs(os.path.dirname(os.path.join(w, rel)) or w, exist_ok=True)
        with open(os.path.join(w, rel), "w") as f:
            f.write(text)
    if common is not None:                       # tools only with a test config (never runnable production copies)
        sandbox_tools(w, common=common, shells=shells)
    os.makedirs(os.path.join(w, "outputs_loc", "exp22"))
    _git(w, "init", "-q")
    _git(w, "add", "-A", ".")
    _git(w, "commit", "-q", "-m", "sandbox PIN2")
    return w, _git(w, "rev-parse", "HEAD")


# --------------------------------------------------------------------------- #
# the supervisor's sandbox box (shared by the supervisor and snapshot tests)
# --------------------------------------------------------------------------- #
E, S = load("exp22_r2c_evidence"), load("exp22_r2c_supervisor")
NAMES = ["T", "GV", "V_GV", "G1", "V_G1", "G2", "V_G2", "G3", "V_G3", "G3D", "V_G3D", "G0", "V_G0", "P", "A"]




class LineWorld(FakeWorld):
    def spawn(self, argv, env, cwd, log_fd, lock_fd):
        return S.spawn_line(argv, env, cwd, log_fd, lock_fd)


def reserve_box(w, pin2, cfg, world):
    orch = os.path.join(w, "outputs_loc", "exp22", f"r2_preflight_{RUN}")
    os.mkdir(orch)
    seal = os.path.join(orch, "seal.txt")
    with open(seal, "w") as f:
        f.write(f"pin2 {pin2}\nrun {RUN}\ncard GPU0 {GPU0} 0\n")
    rc, msg = S.reserve(world, cfg, worktree=w, orch=orch, pin2=pin2, card="GPU0", index=0, seal=seal,
                        seal_sha256=E.sha256_bytes(open(seal, "rb").read()))
    assert rc == 0, msg
    return orch


@pytest.fixture
def box(tmp_path):
    w, pin2 = sandbox_repo(tmp_path)
    cfg = cfg_for(tmp_path)._replace(proc="/proc")
    world = LineWorld()
    orch = reserve_box(w, pin2, cfg, world)
    fd = os.open(S.lock_path(w), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)                              # as the launcher holds it for the supervisor
    b = types.SimpleNamespace(w=w, pin2=pin2, cfg=cfg, orch=orch, world=world, fd=fd)
    b.sup = lambda **kw: S.Supervisor(world, cfg, orch, fd, worktree=w, child_poll_s=0.05, relist=(2, 0.05), **kw)
    b.state = lambda: json.load(open(os.path.join(orch, "state.json")))
    b.control = lambda d: open(os.path.join(w, "a7_control.json"), "w").write(json.dumps(d))
    b.res = os.path.join(cfg.reservations, f"{GPU0}.json")
    b.path = lambda rel: os.path.join(w, rel)
    b.lines = E.expand_lines(json.load(open(S.LINES_JSON)), run=RUN, index=0, worktree=w, py=C.PY,
                             pytest_main=C.PYTEST_MAIN)
    yield b
    os.close(fd)


def phases(monkeypatch):
    seen, real = [], S.E.write_json_atomic

    def spy(path, doc):
        if path.endswith("state.json"):
            seen.append((doc["phase"], (doc["current"] or {}).get("name"), (doc["current"] or {}).get("rc")))
        return real(path, doc)
    monkeypatch.setattr(S.E, "write_json_atomic", spy)
    return seen
