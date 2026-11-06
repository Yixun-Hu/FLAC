"""exp_37 round B-a item 2, part 2: the REAL no-flash launcher on THIS checkout (CPU only), the packaging
exclusion, and the import closure of ``localize_FLAC.py`` (no Python child process can escape the overlay).

Ported from the reviewed exp_35 integration cases (localization-exp ``src/tests/test_exp35_no_flash_launch.py``);
the closure scan now starts at ``localize_FLAC.py`` and follows every repo-local import (``src``, the ``AGREE``
namespace package, top-level modules such as ``eval_FLAC``). Written by the exp_37 Coder (Claude Opus 5.5, max
effort) per the Coder brief r1, round B-a.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERLAY_DIR = os.path.join(REPO, "overlays", "no_flash")
OVERLAY = os.path.join(OVERLAY_DIR, "flash_attn.py")
LAUNCHER = os.path.join(OVERLAY_DIR, "exp37_no_flash_launch.py")
PY = sys.executable
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"
MESSAGE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
FALLBACK = "flash_attn not installed, disabling Flash Attention"
SENTINEL = "EXP37_NO_FLASH_LAUNCHER"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _real_env(**extra):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "EXP37_NO_FLASH_LAUNCH")}
    e.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
             PYTHONPATH=OVERLAY_DIR, **extra)
    return e


def _sentinel(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.startswith(SENTINEL + " ")]
    assert len(lines) == 1, stdout
    return dict(tok.split("=", 1) for tok in lines[0].split()[1:])


def test_overlays_holds_no_package_marker_and_only_the_two_tracked_files():
    assert sorted(os.listdir(OVERLAY_DIR)) == ["exp37_no_flash_launch.py", "flash_attn.py"]
    for root, dirs, files in os.walk(os.path.join(REPO, "overlays")):
        assert "__init__.py" not in files and "__pycache__" not in dirs, root
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "overlays"], capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked == ["overlays/no_flash/exp37_no_flash_launch.py", "overlays/no_flash/flash_attn.py"]


def test_packaging_excludes_the_overlay_tree(tmp_path):
    """setuptools' namespace discovery (where=["."]) would otherwise ship overlays.no_flash."""
    import setuptools
    import tomli
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        find = tomli.load(f)["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["."] and find["exclude"] == ["overlays*"]
    (tmp_path / "overlays" / "no_flash").mkdir(parents=True)
    (tmp_path / "overlays" / "no_flash" / "flash_attn.py").write_text("")
    (tmp_path / "src" / "models").mkdir(parents=True)
    (tmp_path / "src" / "models" / "__init__.py").write_text("")
    assert "overlays.no_flash" in setuptools.find_namespace_packages(where=str(tmp_path))
    kept = setuptools.find_namespace_packages(where=str(tmp_path), exclude=find["exclude"])
    assert not [p for p in kept if p.startswith("overlays")] and "src.models" in kept


def test_the_real_launcher_preflight_on_this_checkout():
    p = subprocess.run([PY, LAUNCHER, "--preflight"], env=_real_env(), cwd=REPO, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert lines.count(MESSAGE) == 1 and lines.count(FALLBACK) == 1, p.stdout
    assert _sentinel(p.stdout) == {"attention_backend": "sdpa", "overlay_sha256": OVERLAY_SHA256,
                                   "launcher_sha256": _sha(LAUNCHER), "target": "--preflight"}
    assert "No module named 'flash_attn'" not in p.stdout + p.stderr
    assert sorted(os.listdir(OVERLAY_DIR)) == ["exp37_no_flash_launch.py", "flash_attn.py"]   # nothing written


def test_a_fresh_python_child_with_the_runner_environment_hits_the_overlay():
    """PYTHONPATH=<repo>/overlays/no_flash alone puts the overlay before site-packages for any fresh
    interpreter, even one started from the repo root."""
    p = subprocess.run([PY, "-c", "import flash_attn"], env=_real_env(), cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0 and f"ImportError: {MESSAGE}" in p.stderr, p.stderr


WORKERS = r'''
import importlib.util, json, os, sys
import torch
from torch.utils.data import DataLoader, Dataset


class Probe(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        import src.models.transformer as T
        spec = importlib.util.find_spec("flash_attn")
        return json.dumps({"pid": os.getpid(), "transformer": T.__file__, "flash_none": T.flash_attn_func is None,
                           "origin": spec.origin if spec else None})


if __name__ == "__main__":
    out = {"parent": os.getpid()}
    for ctx in ("fork", "spawn"):
        loader = DataLoader(Probe(), batch_size=None, num_workers=1, multiprocessing_context=ctx)
        out[ctx] = json.loads(next(iter(loader)))
    with open(os.environ["TARGET_REPORT"], "w") as f:
        json.dump(out, f)
'''


def test_dataloader_fork_and_spawn_workers_run_the_sdpa_path(tmp_path):
    """A forked worker inherits the asserted modules, a spawned one rebuilds them from the parent's explicit
    sys.path -- both report this checkout's transformer and flash_attn_func None."""
    target = tmp_path / "workers.py"
    target.write_text(WORKERS)
    report = tmp_path / "workers.json"
    p = subprocess.run([PY, LAUNCHER, str(target)], env=_real_env(TARGET_REPORT=str(report)), cwd=REPO,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr
    rep = json.loads(report.read_text())
    want = os.path.join(REPO, "src", "models", "transformer.py")
    for ctx in ("fork", "spawn"):
        w = rep[ctx]
        assert w["pid"] != rep["parent"], ctx
        assert os.path.realpath(w["transformer"]) == want and w["flash_none"] is True, (ctx, w)
        assert os.path.realpath(w["origin"]) == OVERLAY, (ctx, w)
    assert p.stdout.count(MESSAGE) == 2                     # the parent's import and the spawned worker's


# --------------------------------------------------------------------------- #
# the localize_FLAC.py tree starts no Python child process
# --------------------------------------------------------------------------- #
ALLOWED_ARGV0 = {"git", "nvidia-smi", "ps"}       # literal argv lists of non-Python tools only
SPAWNERS = {"subprocess": {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
            "os": {"system", "popen", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
                   "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                   "posix_spawn", "posix_spawnp", "fork", "forkpty"},
            "pty": {"spawn", "fork"}, "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
            "multiprocessing": {"Process", "Pool", "get_context", "set_start_method"}}


def _resolve(name):
    """A dotted repo-local module name -> its file (module or package ``__init__``), else None; a namespace
    directory (``AGREE``) resolves to None but does not stop its submodules."""
    base = os.path.join(REPO, *name.split("."))
    for cand in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def _import_closure(roots):
    """Every repo-local file reachable from ``roots`` through ANY import statement (lazy ones too)."""
    seen, todo = set(), list(roots)
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        rel = os.path.relpath(path, REPO)[:-len(".py")].replace(os.sep, ".")
        pkg = rel.rsplit(".", 1)[0] if not rel.endswith("__init__") else rel[:-len(".__init__")]
        names = []
        for node in ast.walk(ast.parse(open(path).read(), path)):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")[:len(pkg.split(".")) - node.level + 1]
                    base = ".".join(parts + ([node.module] if node.module else []))
                names += [base] + [f"{base}.{a.name}" for a in node.names]
        for name in names:
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                hit = _resolve(".".join(parts[:i]))
                if hit and not os.path.relpath(hit, REPO).startswith(os.path.join("src", "tests")):
                    todo.append(hit)
    return seen


def _spawn_violations(path):
    """Process-spawning calls in ``path`` that are not a literal argv list of an allowed non-Python tool."""
    out, aliases = [], {}
    tree = ast.parse(open(path).read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in SPAWNERS:
            out += [f"{path}:{node.lineno}: from {node.module} import {a.name}" for a in node.names
                    if a.name in SPAWNERS[node.module]]
        if isinstance(node, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in node.names if a.name in SPAWNERS})
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)):
            continue
        mod = aliases.get(node.func.value.id, node.func.value.id)
        if mod in SPAWNERS and node.func.attr in SPAWNERS[mod]:
            argv = node.args[0] if node.args else None
            tool = (isinstance(argv, (ast.List, ast.Tuple)) and argv.elts and isinstance(argv.elts[0], ast.Constant)
                    and argv.elts[0].value in ALLOWED_ARGV0)
            if not (mod == "subprocess" and tool):
                out.append(f"{path}:{node.lineno}: {mod}.{node.func.attr}(...)")
    return out


def test_the_localize_flac_tree_starts_no_python_child():
    closure = _import_closure([os.path.join(REPO, "localize_FLAC.py")])
    for rel in ("src/models/transformer.py", "src/data/yaw_rotation.py", "src/localization/runner.py",
                "src/localization/engine.py", "eval_FLAC.py", "AGREE/AGREE/model.py"):
        assert os.path.join(REPO, rel) in closure, rel
    assert len(closure) > 30
    monitor = os.path.join(REPO, "src", "localization", "exp37_cotenancy.py")      # the one reviewed exception:
    assert [v for p in sorted(closure) if p != monitor for v in _spawn_violations(p)] == []   # the isolated monitor
    # child (pinned by test_exp37_runner_monitor.py::test_the_only_python_child_of_the_localize_flac_tree_...)


def test_the_spawn_scanner_bites(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess, os\nimport subprocess as sp\nsubprocess.run(['git', 'status'])\n"
                   "subprocess.run([sys.executable, '-c', 'x'])\nsp.Popen(['python'])\nos.execv('/bin/sh', [])\n"
                   "from multiprocessing import Process\nsubprocess.run(['nvidia-smi', '-L'])\n"
                   "subprocess.run(argv)\n")
    got = _spawn_violations(str(bad))
    assert len(got) == 5 and not any(":3:" in g or ":8:" in g for g in got), got


def _spawner_references(path):
    """Every ``subprocess.X`` / ``os.X`` spawner attribute in ``path``, called or merely referenced."""
    found = set()
    for node in ast.walk(ast.parse(open(path).read(), path)):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in ("subprocess", "os") and node.attr in SPAWNERS[node.value.id]):
            found.add(f"{node.value.id}.{node.attr}")
    return found


def test_the_only_spawner_reference_is_the_host_record_with_literal_tool_argv():
    """The scanner above sees direct calls only; the host record calls ``subprocess.run`` through an injectable
    default. Pin that this is the closure's ONLY spawner reference and that its argv lists are literal tools."""
    closure = _import_closure([os.path.join(REPO, "localize_FLAC.py")])
    refs = {(os.path.relpath(p, REPO), ref) for p in closure for ref in _spawner_references(p)}
    assert refs - {("src/localization/exp37_cotenancy.py", "subprocess.Popen")} == {
        ("src/localization/run_provenance.py", "subprocess.run")}                  # + the monitor child (pinned elsewhere)
    tree = ast.parse(open(os.path.join(REPO, "src", "localization", "run_provenance.py")).read())
    argvs = [node.args[1] for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_command"]
    assert len(argvs) == 2 and all(isinstance(a, ast.List) and all(isinstance(e, ast.Constant) for e in a.elts)
                                   for a in argvs)
    assert sorted(a.elts[0].value for a in argvs) == ["nvidia-smi", "ps"]
