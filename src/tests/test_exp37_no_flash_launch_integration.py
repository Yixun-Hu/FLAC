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


