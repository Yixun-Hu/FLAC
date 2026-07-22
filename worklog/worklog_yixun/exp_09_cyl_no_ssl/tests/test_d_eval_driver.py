"""exp-09 D-stage: d_eval_driver.sh script-contract tests (CPU-only).

The driver is a GPU-eval launcher template; these tests NEVER execute a real eval or the
(model-building) pin gate. The happy path is exercised only through DRY_RUN (prints the
assembled pin-gate + eval commands and exits 0 BEFORE any GPU gate, mkdir, or exec); every
non-DRY_RUN test drives a REFUSAL that exits before the eval_FLAC.py launch. nvidia-smi is a
fake on PATH; CUDA_VISIBLE_DEVICES="".

Contract asserted:
  * MANDATORY flags: --cond-method fa_invariant AND --frame-avg-angles 0 (vanilla-trap guard);
  * --rotate-deg <ROTATE_DEG> (D2 sweep parameter);
  * PYTHONPATH-first eval env: PYTHONPATH=<cyl>/src prepended, EVAL_PYTHON parameterized;
  * EMBEDDED PIN GATE (F2): assert_arm_configs_exp09.py runs bound to the ACTUAL config +
    its auto-detected variant (base / online);
  * `set -o pipefail` verbatim AND uncommented at line start (F7c);
  * frozen MIN_FREE gate reads c1_frozen_min_free.txt (absent/non-numeric refused);
  * refuses missing args with exit 3 + usage, no `set -u` unbound trip;
  * external log-dir arg: an in-worktree path is refused BEFORE any mkdir (F6, no dir left).
"""
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

_EXP09_DIR = Path(__file__).resolve().parents[1]
_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
_DRIVER = _EXP09_DIR / "d_eval_driver.sh"
_BASE_CFG = _EXP09_DIR / "FLAC_AR_exp09.json"
_ONLINE_CFG = _EXP09_DIR / "FLAC_AR_exp09_online_eval.json"

_FAKE_SMI = """\
#!{python}
import os, sys
args = sys.argv[1:]
def has(s): return any(s in a for a in args)
if os.environ.get("FAKE_SMI_FAIL") == "1":
    sys.exit(1)
if has("--query-compute-apps"):
    sys.exit(0)
if has("--query-gpu=memory.free"):
    idx = None
    for i, a in enumerate(args):
        if a == "-i" and i + 1 < len(args):
            idx = args[i + 1]
    free = os.environ.get("FAKE_FREE_MB_%s" % idx, os.environ.get("FAKE_FREE_MB", "999999"))
    sys.stdout.write(free + "\\n"); sys.exit(0)
if has("index"):
    for g in (0, 1):
        sys.stdout.write("%d, 1000, 1000\\n" % g)
    sys.exit(0)
sys.exit(0)
"""


def _fake_bin(tmp_path):
    d = tmp_path / "bin"; d.mkdir(exist_ok=True)
    smi = d / "nvidia-smi"
    smi.write_text(_FAKE_SMI.format(python=sys.executable))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _frozen(tmp_path, value="27552"):
    f = tmp_path / "c1_frozen_min_free.txt"; f.write_text(value + "\n")
    return f


def _run(args, *, tmp_path, env_extra=None, timeout=120):
    env = dict(os.environ)
    env["PATH"] = f"{_fake_bin(tmp_path)}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", str(_DRIVER), *map(str, args)],
                          cwd=str(_WORKTREE_ROOT), env=env, capture_output=True,
                          text=True, timeout=timeout)


def _ok_args(tmp_path, cfg=_BASE_CFG, name="exp09_D1_K8_seed42"):
    ckpt = tmp_path / "step=67500.ckpt"; ckpt.write_text("x")
    logdir = tmp_path / "extlog"; logdir.mkdir(exist_ok=True)
    return [str(ckpt), str(cfg), name, str(logdir)]


def _dry(tmp_path, cfg=_BASE_CFG, env_extra=None):
    e = {"DRY_RUN": "1", "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))}
    if env_extra:
        e.update(env_extra)
    return _run(_ok_args(tmp_path, cfg), tmp_path=tmp_path, env_extra=e)


# --------------------------------------------------------------------------------------- #
# existence + pipefail (F7c: uncommented, line-start)
# --------------------------------------------------------------------------------------- #
def test_driver_exists():
    assert _DRIVER.exists()


def test_pipefail_literal_uncommented_at_line_start():
    lines = _DRIVER.read_text().splitlines()
    assert any(re.match(r"^set -o pipefail\b", ln) for ln in lines), \
        "`set -o pipefail` must be present, uncommented, at the start of a line"


# --------------------------------------------------------------------------------------- #
# DRY_RUN: mandatory eval flags (mutation m4 target)
# --------------------------------------------------------------------------------------- #
def test_dry_run_mandatory_cond_flags(tmp_path):
    out = (lambda p: p.stdout + p.stderr)(_dry(tmp_path))
    assert "--cond-method fa_invariant" in out
    assert "--frame-avg-angles 0" in out


def test_dry_run_rotate_deg_sweep_param(tmp_path):
    out = (lambda p: p.stdout + p.stderr)(_dry(tmp_path, env_extra={"ROTATE_DEG": "90"}))
    assert "--rotate-deg 90" in out


def test_dry_run_pythonpath_first_eval_env(tmp_path):
    p = _dry(tmp_path, env_extra={"CYL_DINOV3_SRC": "/opt/cyl/src",
                                  "EVAL_PYTHON": "/opt/rir2rir/bin/python"})
    out = p.stdout + p.stderr
    assert "PYTHONPATH=/opt/cyl/src" in out and "/opt/rir2rir/bin/python" in out


def test_dry_run_passes_ckpt_config_evalname(tmp_path):
    args = _ok_args(tmp_path)
    p = _run(args, tmp_path=tmp_path,
             env_extra={"DRY_RUN": "1", "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = p.stdout + p.stderr
    assert "eval_FLAC.py" in out and "--eval-name exp09_D1_K8_seed42" in out and args[0] in out


# --------------------------------------------------------------------------------------- #
# EMBEDDED PIN GATE (F2): bound to the actual config + variant
# --------------------------------------------------------------------------------------- #
def test_dry_run_embeds_pin_gate_base_variant(tmp_path):
    out = (lambda p: p.stdout + p.stderr)(_dry(tmp_path, cfg=_BASE_CFG))
    assert "assert_arm_configs_exp09.py" in out
    assert f"--config {_BASE_CFG}" in out
    assert "--config-variant base" in out


def test_dry_run_embeds_pin_gate_online_variant(tmp_path):
    """The online-eval config auto-detects the 'online' variant (its registered delta differs)."""
    out = (lambda p: p.stdout + p.stderr)(_dry(tmp_path, cfg=_ONLINE_CFG))
    assert "assert_arm_configs_exp09.py" in out
    assert f"--config {_ONLINE_CFG}" in out
    assert "--config-variant online" in out


def test_config_variant_env_override(tmp_path):
    out = (lambda p: p.stdout + p.stderr)(_dry(tmp_path, cfg=_BASE_CFG,
                                               env_extra={"CONFIG_VARIANT": "online"}))
    assert "--config-variant online" in out


def test_driver_text_names_pin_gate_and_eval():
    text = _DRIVER.read_text()
    assert "assert_arm_configs_exp09.py" in text and "eval_FLAC.py" in text
    assert "c1_frozen_min_free.txt" in text


# --------------------------------------------------------------------------------------- #
# missing-arg refusals: exit 3 + usage, NO set -u unbound trip
# --------------------------------------------------------------------------------------- #
def test_refuses_no_args_exit3_usage(tmp_path):
    p = _run([], tmp_path=tmp_path)
    out = p.stdout + p.stderr
    assert p.returncode == 3 and "usage" in out.lower()
    assert "unbound variable" not in out.lower()


def test_refuses_missing_logdir_exit3(tmp_path):
    ckpt = tmp_path / "c.ckpt"; ckpt.write_text("x")
    p = _run([str(ckpt), str(_BASE_CFG), "nm"], tmp_path=tmp_path)
    out = p.stdout + p.stderr
    assert p.returncode == 3
    assert "usage" in out.lower() or "log" in out.lower()
    assert "unbound variable" not in out.lower()


# --------------------------------------------------------------------------------------- #
# F6: in-worktree log dir refused BEFORE any mkdir (nonexistent path leaves NO directory)
# --------------------------------------------------------------------------------------- #
def test_refuses_logdir_inside_worktree_before_mkdir(tmp_path):
    ckpt = tmp_path / "c.ckpt"; ckpt.write_text("x")
    inside = _EXP09_DIR / "d_driver_inworktree_probe_dir"
    assert not inside.exists()
    p = _run([str(ckpt), str(_BASE_CFG), "nm", str(inside)], tmp_path=tmp_path,
             env_extra={"C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = p.stdout + p.stderr
    try:
        assert p.returncode == 3 and "worktree" in out.lower()
        assert not inside.exists(), "an in-worktree refusal must NOT create the directory (F6)"
    finally:
        if inside.exists():
            inside.rmdir()


# --------------------------------------------------------------------------------------- #
# frozen MIN_FREE gate + VRAM gate (non-DRY_RUN refusals)
# --------------------------------------------------------------------------------------- #
def test_refuses_absent_frozen_file(tmp_path):
    p = _run(_ok_args(tmp_path), tmp_path=tmp_path,
             env_extra={"C1_FROZEN_MIN_FREE_FILE": str(tmp_path / "nope.txt")})
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "frozen" in out.lower()


def test_refuses_non_numeric_frozen(tmp_path):
    p = _run(_ok_args(tmp_path), tmp_path=tmp_path,
             env_extra={"C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path, "TBD_FROM_C1_FIT"))})
    assert p.returncode != 0


def test_frozen_value_gates_eval_gpu(tmp_path):
    p = _run(_ok_args(tmp_path), tmp_path=tmp_path,
             env_extra={"C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path, "27552")),
                        "FAKE_FREE_MB": "27000"})
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "27552" in out


def test_fail_closed_on_nvidia_smi_error(tmp_path):
    p = _run(_ok_args(tmp_path), tmp_path=tmp_path,
             env_extra={"C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path, "27552")),
                        "FAKE_SMI_FAIL": "1"})
    assert p.returncode != 0
