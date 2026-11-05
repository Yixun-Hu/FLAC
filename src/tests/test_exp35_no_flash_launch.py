"""exp_35 round 1a: the committed no-flash overlay and its same-interpreter launcher
(plan_orbitrir_yaw_evidence.md r2 §3b content, §11 B1 steps (1)-(7) + packaging H5, r3 §12 H4).

Every launch runs the launcher as a SUBPROCESS: the launcher refuses any bytecode under its own
directory, so no test ever imports it (or the overlay) in-process. Most cases use a throwaway repo
under tmp_path (the real overlay + launcher bytes, a stand-in ``src/models/transformer.py`` carrying
the real file's flash-attention fallback block, so no torch import is needed); the integration cases
run the real launcher on this checkout (the real transformer, CPU only).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERLAY_DIR = os.path.join(REPO, "overlays", "no_flash")
OVERLAY = os.path.join(OVERLAY_DIR, "flash_attn.py")
LAUNCHER = os.path.join(OVERLAY_DIR, "exp35_no_flash_launch.py")
PY = sys.executable
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"   # plan §3b, verified
MESSAGE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
FALLBACK = "flash_attn not installed, disabling Flash Attention"
SENTINEL = "EXP35_NO_FLASH_LAUNCHER"
REFUSED = 86


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _fallback_block():
    """The real transformer.py's flash-attention import block, verbatim (lines 12-18 at the pin)."""
    with open(os.path.join(REPO, "src", "models", "transformer.py")) as f:
        text = f.read()
    start = text.index("try:\n    from flash_attn import flash_attn_func\n")
    end = text.index("    flash_attn_func = None\n", start) + len("    flash_attn_func = None\n")
    return text[start:end]


TARGET = r'''
import json, os, sys
import src.models.transformer as T
report = {"argv": sys.argv, "name": __name__, "path": sys.path[:2], "pythonpath": os.environ.get("PYTHONPATH"),
          "flash_none": T.flash_attn_func is None, "transformer": T.__file__, "dont_write": sys.dont_write_bytecode,
          "path_all": sys.path}
with open(os.environ["TARGET_REPORT"], "w") as f:
    json.dump(report, f)
if os.environ.get("TARGET_EXIT"):
    sys.exit(int(os.environ["TARGET_EXIT"]))
'''


def fake_repo(tmp_path, transformer=None):
    """A throwaway checkout: the real overlay + launcher, a stand-in transformer, a target."""
    root = tmp_path / "repo"
    ov = root / "overlays" / "no_flash"
    ov.mkdir(parents=True)
    shutil.copy(OVERLAY, ov / "flash_attn.py")
    shutil.copy(LAUNCHER, ov / "exp35_no_flash_launch.py")
    (root / "src" / "models").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "models" / "__init__.py").write_text("")
    (root / "src" / "models" / "transformer.py").write_text(transformer if transformer is not None
                                                             else _fallback_block())
    (root / "target.py").write_text(TARGET)
    return root


def launch(root, *args, env=None, launcher=None):
    """Run the launcher of ``root`` (cwd = root); ``env`` values of None delete the variable."""
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TARGET_REPORT=str(root.parent / "report.json"))
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    report = root.parent / "report.json"
    if report.exists():
        report.unlink()
    p = subprocess.run([PY, str(launcher or root / "overlays" / "no_flash" / "exp35_no_flash_launch.py"), *args],
                       env=e, cwd=str(root), capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr, (json.loads(report.read_text()) if report.exists() else None)


def _sentinel(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.startswith(SENTINEL + " ")]
    assert len(lines) == 1, stdout
    return dict(tok.split("=", 1) for tok in lines[0].split()[1:])


def _pair_then_sentinel(stdout):
    lines = stdout.splitlines()
    msg = [i for i, ln in enumerate(lines) if ln == MESSAGE]
    fb = [i for i, ln in enumerate(lines) if ln == FALLBACK]
    sent = [i for i, ln in enumerate(lines) if ln.startswith(SENTINEL + " ")]
    return len(msg) == len(fb) == len(sent) == 1 and fb[0] == msg[0] + 1 and sent[0] > fb[0]


# --------------------------------------------------------------------------- #
# the committed files: the overlay's exact bytes, the layout, packaging (H5)
# --------------------------------------------------------------------------- #
def test_the_overlay_is_the_registered_four_lines():
    assert _sha(OVERLAY) == OVERLAY_SHA256
    with open(OVERLAY) as f:
        text = f.read()
    assert text.count("\n") == 4 and text.endswith(f'raise ImportError("{MESSAGE}")\n')


def test_overlays_holds_no_package_marker_and_only_the_two_files():
    assert sorted(os.listdir(OVERLAY_DIR)) == ["exp35_no_flash_launch.py", "flash_attn.py"]
    for root, dirs, files in os.walk(os.path.join(REPO, "overlays")):
        assert "__init__.py" not in files and "__pycache__" not in dirs, root
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "overlays"], capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked == ["overlays/no_flash/exp35_no_flash_launch.py", "overlays/no_flash/flash_attn.py"]


def _exclude_from_pyproject():
    import tomli
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        find = tomli.load(f)["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["."]
    return find["exclude"]


def test_packaging_excludes_the_overlay_tree(tmp_path):
    """H5: setuptools' namespace discovery (where=["."]) would ship overlays.no_flash otherwise."""
    import setuptools
    exclude = _exclude_from_pyproject()
    assert exclude == ["overlays*"]
    root = fake_repo(tmp_path)
    found = setuptools.find_namespace_packages(where=str(root))
    assert "overlays.no_flash" in found                                   # what the exclusion prevents
    kept = setuptools.find_namespace_packages(where=str(root), exclude=exclude)
    assert not [p for p in kept if p.startswith("overlays")] and "src.models" in kept


# --------------------------------------------------------------------------- #
# (1)-(7) on a throwaway checkout
# --------------------------------------------------------------------------- #
def test_the_target_runs_in_this_interpreter_with_its_argv_after_one_sentinel(tmp_path):
    root = fake_repo(tmp_path)
    rc, out, err, rep = launch(root, "target.py", "--alpha", "1", "--beta")
    assert rc == 0, err
    assert rep["argv"] == ["target.py", "--alpha", "1", "--beta"] and rep["name"] == "__main__"
    assert rep["path"] == [str(root / "overlays" / "no_flash"), str(root)]       # (3) overlay, then repo
    assert rep["flash_none"] is True and rep["dont_write"] is True
    assert os.path.realpath(rep["transformer"]) == str(root / "src" / "models" / "transformer.py")
    assert rep["pythonpath"] == str(root / "overlays" / "no_flash")              # children: overlay only
    assert _pair_then_sentinel(out), out
    assert _sentinel(out) == {"attention_backend": "sdpa", "overlay_sha256": OVERLAY_SHA256,
                              "launcher_sha256": _sha(LAUNCHER), "target": "target.py"}
    assert not list((root / "overlays").rglob("__pycache__")) and not list(root.rglob("*.pyc"))


def test_preflight_checks_everything_and_runs_nothing(tmp_path):
    root = fake_repo(tmp_path)
    rc, out, err, rep = launch(root, "--preflight")
    assert rc == 0 and rep is None, err
    assert _pair_then_sentinel(out) and _sentinel(out)["target"] == "--preflight"
    rc, _, err, _ = launch(root, "--preflight", "extra")
    assert rc == REFUSED and "--preflight" in err


def test_the_exit_code_of_the_target_is_its_own(tmp_path):
    rc, out, _, rep = launch(fake_repo(tmp_path), "target.py", env={"TARGET_EXIT": "7"})
    assert rc == 7 and rep is not None and _pair_then_sentinel(out)


def _refused(rc, out, err, rep, needle):
    assert rc == REFUSED and rep is None, (rc, out, err)
    assert "NO-FLASH LAUNCH REFUSED" in err and needle in err, err
    assert SENTINEL not in out


def test_bytecode_writing_must_be_disabled_in_the_environment(tmp_path):
    _refused(*launch(fake_repo(tmp_path), "target.py", env={"PYTHONDONTWRITEBYTECODE": None}),
             "PYTHONDONTWRITEBYTECODE")


def test_an_inherited_pythonpath_is_dropped(tmp_path):
    """(3): an inherited entry never reaches sys.path -- even one holding an importable flash_attn."""
    root = fake_repo(tmp_path)
    foreign = tmp_path / "foreign"
    (foreign / "flash_attn").mkdir(parents=True)
    (foreign / "flash_attn" / "__init__.py").write_text("flash_attn_func = object()\n")
    rc, out, err, rep = launch(root, "target.py", env={"PYTHONPATH": f"{foreign}{os.pathsep}{root}"})
    assert rc == 0, err
    assert str(foreign) not in rep["path_all"] and rep["flash_none"] is True
    assert rep["path_all"][:2] == [str(root / "overlays" / "no_flash"), str(root)]
    assert rep["path_all"].count(str(root)) == 1 and rep["pythonpath"] == str(root / "overlays" / "no_flash")


@pytest.mark.parametrize("shadow", ["flash_attn.py", "flash_attn/__init__.py", "flash_attn/x.txt",
                                    "flash_attn.cpython-310-x86_64-linux-gnu.so", "flash_attn.pyc"])
def test_a_repo_root_shadow_is_refused(tmp_path, shadow):
    root = fake_repo(tmp_path)
    (root / shadow).parent.mkdir(parents=True, exist_ok=True)
    (root / shadow).write_text("flash_attn_func = object()\n")
    _refused(*launch(root, "target.py"), "repo root")


@pytest.mark.parametrize("stray", ["__pycache__/flash_attn.cpython-310.pyc", "flash_attn.pyc",
                                   "__pycache__/exp35_no_flash_launch.cpython-310.pyc", "flash_attn/__init__.py",
                                   "notes.txt"])
def test_anything_besides_the_two_files_under_the_overlay_is_refused(tmp_path, stray):
    root = fake_repo(tmp_path)
    path = root / "overlays" / "no_flash" / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00stray")
    _refused(*launch(root, "target.py"), "overlays/no_flash")


@pytest.mark.parametrize("stray", ["__init__.py", "__pycache__/x.cpython-310.pyc", "other/__init__.py"])
def test_a_package_marker_or_bytecode_anywhere_under_overlays_is_refused(tmp_path, stray):
    root = fake_repo(tmp_path)
    (root / "overlays" / stray).parent.mkdir(parents=True, exist_ok=True)
    (root / "overlays" / stray).write_text("")
    _refused(*launch(root, "target.py"), "overlays")


def test_an_edited_overlay_is_refused(tmp_path):
    root = fake_repo(tmp_path)
    with open(root / "overlays" / "no_flash" / "flash_attn.py", "a") as f:
        f.write("# edited\n")
    _refused(*launch(root, "target.py"), "pinned")


@pytest.mark.parametrize("hook", [
    # flash_attn already imported (from elsewhere) before the launcher ran
    "import sys, types\nm = types.ModuleType('flash_attn'); m.__file__ = '/elsewhere/flash_attn.py'\n"
    "sys.modules['flash_attn'] = m\n",
    # a meta-path finder that resolves flash_attn to another origin
    "import sys, importlib.machinery as M\nclass F:\n    @staticmethod\n    def find_spec(name, path=None, target=None):\n"
    "        return M.ModuleSpec(name, M.SourceFileLoader(name, '/elsewhere/flash_attn.py'), origin='/elsewhere/flash_attn.py') "
    "if name == 'flash_attn' else None\nsys.meta_path.insert(0, F)\n"], ids=["preimported", "meta_path"])
def test_a_wrong_spec_origin_is_refused(tmp_path, hook):
    """(4): whatever the route (a startup hook here), flash_attn must resolve to the overlay file."""
    root = fake_repo(tmp_path)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(hook)
    _refused(*launch(root, "target.py", env={"PYTHONPATH": str(hooks)}), "flash_attn")


def test_a_transformer_with_flash_attention_enabled_is_refused(tmp_path):
    root = fake_repo(tmp_path, transformer="flash_attn_func = object()\n")
    _refused(*launch(root, "target.py"), "flash_attn_func")


def test_a_src_package_from_outside_the_repo_is_refused(tmp_path):
    """(5): a ``src`` that is not this checkout's (here: put first on the path by a startup hook; the
    flac env also holds a stale installed ``src`` in site-packages, which the same check refuses)."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "src" / "models").mkdir(parents=True)
    (elsewhere / "src" / "__init__.py").write_text("")
    (elsewhere / "src" / "models" / "__init__.py").write_text("")
    (elsewhere / "src" / "models" / "transformer.py").write_text(_fallback_block())
    root = fake_repo(tmp_path)
    shutil.rmtree(root / "src")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(f"import sys\nsys.path.insert(0, {str(elsewhere)!r})\n")
    rc, out, err, rep = launch(root, "target.py", env={"PYTHONPATH": str(hooks)})
    _refused(rc, out, err, rep, "src.models.transformer is")
    assert f"the package src is {elsewhere}" in err


def test_a_transformer_that_fails_to_import_is_refused(tmp_path):
    _refused(*launch(fake_repo(tmp_path, transformer="raise RuntimeError('boom')\n"), "target.py"),
             "src.models.transformer")


def test_a_missing_target_or_no_argument_is_refused(tmp_path):
    root = fake_repo(tmp_path)
    _refused(*launch(root, "absent.py"), "absent.py")
    _refused(*launch(root), "usage")


# --------------------------------------------------------------------------- #
# integration: the real launcher, the real transformer (this checkout, CPU only)
# --------------------------------------------------------------------------- #
def _real_env(**extra):
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
             PYTHONPATH=OVERLAY_DIR, **extra)
    return e


def test_the_real_launcher_preflight_on_this_checkout():
    p = subprocess.run([PY, LAUNCHER, "--preflight"], env=_real_env(), cwd=REPO, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    assert _pair_then_sentinel(p.stdout), p.stdout
    assert _sentinel(p.stdout) == {"attention_backend": "sdpa", "overlay_sha256": OVERLAY_SHA256,
                                   "launcher_sha256": _sha(LAUNCHER), "target": "--preflight"}
    assert "No module named 'flash_attn'" not in p.stdout + p.stderr
    assert sorted(os.listdir(OVERLAY_DIR)) == ["exp35_no_flash_launch.py", "flash_attn.py"]   # nothing written


def test_a_fresh_python_child_with_the_runner_environment_hits_the_overlay():
    """§12 H4 (c): PYTHONPATH=<repo>/overlays/no_flash alone puts the overlay before site-packages
    for any fresh interpreter, even one started from the repo root."""
    p = subprocess.run([PY, "-c", "import flash_attn"], env=_real_env(), cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0 and f"ImportError: {MESSAGE}" in p.stderr, p.stderr
    assert sorted(os.listdir(OVERLAY_DIR)) == ["exp35_no_flash_launch.py", "flash_attn.py"]


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
    """§12 H4 (b): a forked worker inherits the asserted modules, a spawned one rebuilds them from the
    parent's explicit sys.path -- both report this checkout's transformer and flash_attn_func None."""
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
# §12 H4 (a): the pinned evaluator / probe tree starts no Python child process
# --------------------------------------------------------------------------- #
PROBE = os.path.join(REPO, "worklog", "worklog_yixun", "exp_25_depth_canon_claude", "depth_canon_cond_probe.py")
SPAWNERS = {"subprocess": {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
            "os": {"system", "popen", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
                   "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                   "posix_spawn", "posix_spawnp", "fork", "forkpty"},
            "pty": {"spawn", "fork"}, "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
            "multiprocessing": {"Process", "Pool", "get_context", "set_start_method"}}


def _resolve(name):
    """``src.a.b`` -> its file under this checkout (module or package), else None."""
    base = os.path.join(REPO, *name.split("."))
    for cand in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def _import_closure(roots):
    """Every ``src`` file reachable from ``roots`` through ANY import statement (lazy ones too), with
    each imported module's parent packages."""
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
            if parts[0] != "src":
                continue
            for i in range(1, len(parts) + 1):
                hit = _resolve(".".join(parts[:i]))
                if hit and not os.path.relpath(hit, REPO).startswith(os.path.join("src", "tests")):
                    todo.append(hit)
    return seen


def _spawn_violations(path):
    """Process-spawning calls in ``path`` that are not a literal ``["git", ...]`` argv."""
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
            git = (isinstance(argv, (ast.List, ast.Tuple)) and argv.elts and isinstance(argv.elts[0], ast.Constant)
                   and argv.elts[0].value == "git")
            if not (mod == "subprocess" and git):
                out.append(f"{path}:{node.lineno}: {mod}.{node.func.attr}(...)")
    return out


def test_the_evaluator_and_probe_tree_starts_no_python_child():
    closure = _import_closure([os.path.join(REPO, "eval_FLAC.py"), PROBE])
    assert os.path.join(REPO, "src", "models", "transformer.py") in closure
    assert os.path.join(REPO, "src", "data", "yaw_rotation.py") in closure and len(closure) > 20
    assert [v for p in sorted(closure) for v in _spawn_violations(p)] == []
    git_calls = [p for p in closure if "subprocess" in open(p).read()]
    assert sorted(os.path.relpath(p, REPO) for p in git_calls) == [
        "eval_FLAC.py", "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py"]


def test_the_spawn_scanner_bites(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess, os\nimport subprocess as sp\nsubprocess.run(['git', 'status'])\n"
                   "subprocess.run([sys.executable, '-c', 'x'])\nsp.Popen(['python'])\nos.execv('/bin/sh', [])\n"
                   "from multiprocessing import Process\n")
    got = _spawn_violations(str(bad))
    assert len(got) == 4 and not any(":3:" in g for g in got), got
