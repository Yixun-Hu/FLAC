"""exp_37 round B-a item 2: the no-flash overlay and its same-interpreter launcher for ``localize_FLAC.py``
(plan_loc_efficiency.md §3 / §10 "Provenance"; a port of the reviewed exp_35 ``overlays/no_flash/`` pattern of
``localization-exp``, plus the in-process provenance hand-off the runner records as ``attention_backend``).

Every launch runs the launcher as a SUBPROCESS: the launcher refuses any bytecode under its own directory, so no
test ever imports it (or the overlay) in-process. Most cases use a throwaway repo under tmp_path (the real overlay +
launcher bytes, a stand-in ``src/models/transformer.py`` carrying the real file's flash-attention fallback block,
so no torch import is needed); the integration cases (part 2) run the real launcher on this checkout, CPU only.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""
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
LAUNCHER = os.path.join(OVERLAY_DIR, "exp37_no_flash_launch.py")
PY = sys.executable
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"   # the reviewed exp_35 bytes
MESSAGE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
FALLBACK = "flash_attn not installed, disabling Flash Attention"
SENTINEL = "EXP37_NO_FLASH_LAUNCHER"
LAUNCH_ENV = "EXP37_NO_FLASH_LAUNCH"
REFUSED = 86


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _fallback_block():
    """The real transformer.py's flash-attention import block, verbatim."""
    with open(os.path.join(REPO, "src", "models", "transformer.py")) as f:
        text = f.read()
    start = text.index("try:\n    from flash_attn import flash_attn_func\n")
    end = text.index("    flash_attn_func = None\n", start) + len("    flash_attn_func = None\n")
    return text[start:end]


TARGET = r'''
import json, os, sys
report = {"argv": sys.argv, "name": __name__, "file": __file__, "file_abs": os.path.abspath(__file__),
          "cwd": os.getcwd(), "pid": os.getpid(),
          "path": sys.path[:2], "pythonpath": os.environ.get("PYTHONPATH"), "dont_write": sys.dont_write_bytecode,
          "path_all": sys.path, "launch_env": os.environ.get("EXP37_NO_FLASH_LAUNCH")}
if os.environ.get("TARGET_NO_IMPORT") != "1":   # (a direct invocation would meet the real flash_attn)
    import src.models.transformer as T
    report.update(flash_none=T.flash_attn_func is None, transformer=T.__file__)
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
    shutil.copy(LAUNCHER, ov / "exp37_no_flash_launch.py")
    (root / "src" / "models").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "models" / "__init__.py").write_text("")
    (root / "src" / "models" / "transformer.py").write_text(transformer if transformer is not None
                                                             else _fallback_block())
    (root / "target.py").write_text(TARGET)
    return root


def launch(root, *args, env=None, launcher=None):
    """Run the launcher of ``root`` (cwd = root); ``env`` values of None delete the variable."""
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", LAUNCH_ENV)}
    e.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TARGET_REPORT=str(root.parent / "report.json"))
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    report = root.parent / "report.json"
    if report.exists():
        report.unlink()
    p = subprocess.Popen([PY, str(launcher or root / "overlays" / "no_flash" / "exp37_no_flash_launch.py"), *args],
                         env=e, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=300)
    rep = json.loads(report.read_text()) if report.exists() else None
    if rep is not None:
        rep["launcher_pid"] = p.pid                     # the process the launcher ran in
    return p.returncode, out, err, rep


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
# the committed overlay: the reviewed exp_35 bytes, unchanged
# --------------------------------------------------------------------------- #
def test_the_overlay_is_the_reviewed_four_lines():
    assert _sha(OVERLAY) == OVERLAY_SHA256
    with open(OVERLAY) as f:
        text = f.read()
    assert text.count("\n") == 4 and text.endswith(f'raise ImportError("{MESSAGE}")\n')


# --------------------------------------------------------------------------- #
# (1)-(7) on a throwaway checkout
# --------------------------------------------------------------------------- #
def test_the_target_runs_in_this_interpreter_with_its_argv_after_one_sentinel(tmp_path):
    root = fake_repo(tmp_path)
    rc, out, err, rep = launch(root, "target.py", "--alpha", "1", "--beta")
    assert rc == 0, err
    assert rep["argv"] == ["target.py", "--alpha", "1", "--beta"] and rep["name"] == "__main__"
    assert rep["pid"] == rep["launcher_pid"] and rep["file"] == "target.py" and rep["cwd"] == str(root)
    assert rep["path"] == [str(root / "overlays" / "no_flash"), str(root)]       # (3) overlay, then repo
    assert rep["flash_none"] is True and rep["dont_write"] is True
    assert os.path.realpath(rep["transformer"]) == str(root / "src" / "models" / "transformer.py")
    assert rep["pythonpath"] == str(root / "overlays" / "no_flash")              # children: overlay only
    assert _pair_then_sentinel(out), out
    assert _sentinel(out) == {"attention_backend": "sdpa", "overlay_sha256": OVERLAY_SHA256,
                              "launcher_sha256": _sha(LAUNCHER), "target": "target.py"}
    assert not list((root / "overlays").rglob("__pycache__")) and not list(root.rglob("*.pyc"))


def test_the_target_receives_the_in_process_provenance(tmp_path):
    """The runner records this hand-off in its run manifest: the backend the launcher asserted, both digests,
    the target, and the pid of the ONE process the launcher and the target share (a stale exported copy of
    the variable belongs to another pid and is refused by the runner)."""
    root = fake_repo(tmp_path)
    stale = json.dumps({"attention_backend": "sdpa", "pid": 1, "overlay_sha256": "0" * 64})
    rc, out, err, rep = launch(root, "target.py", env={LAUNCH_ENV: stale})
    assert rc == 0, err
    got = json.loads(rep["launch_env"])
    assert got == {"attention_backend": "sdpa", "flash_attn_func_is_none": True, "overlay_sha256": OVERLAY_SHA256,
                   "launcher_sha256": _sha(LAUNCHER), "pid": rep["pid"], "target": "target.py",
                   "sentinel": SENTINEL}
    assert rep["pid"] == rep["launcher_pid"]


def test_the_target_sees_what_a_direct_invocation_sees(tmp_path):
    """runpy.run_path in the launcher's own process: the arguments, ``__name__``, the script's resolved path and
    the cwd of ``python target.py ...`` from the same cwd (``__file__`` stays as given, which every consumer --
    localize_FLAC.py's ``Path(__file__).resolve()`` -- resolves against the unchanged cwd)."""
    root = fake_repo(tmp_path)
    rc, _, err, via = launch(root, "target.py", "--x", "y", env={"TARGET_NO_IMPORT": "1"})
    assert rc == 0, err
    report = tmp_path / "direct.json"
    p = subprocess.run([PY, "target.py", "--x", "y"], cwd=str(root), capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TARGET_NO_IMPORT="1",
                                TARGET_REPORT=str(report)))
    assert p.returncode == 0, p.stderr
    direct = json.loads(report.read_text())
    for key in ("name", "file_abs", "cwd"):
        assert via[key] == direct[key], key
    assert via["argv"][1:] == direct["argv"][1:] == ["--x", "y"]
    assert via["file"] == "target.py" and direct["file"] == str(root / "target.py")       # the one difference
    assert via["pid"] == via["launcher_pid"] != direct["pid"]


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


