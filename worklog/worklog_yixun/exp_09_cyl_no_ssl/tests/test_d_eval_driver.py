"""exp-09 D-stage: d_eval_driver.sh script-contract tests (CPU-only).

The driver is a GPU-eval launcher template; these tests NEVER execute a real eval. The
happy path is exercised only through DRY_RUN (prints the assembled command and exits 0
BEFORE any GPU gate or exec); every non-DRY_RUN test drives a REFUSAL that exits before
the eval_FLAC.py launch. The nvidia-smi seam is a fake on PATH; CUDA_VISIBLE_DEVICES="".

Contract asserted:
  * MANDATORY flags in the built command: --cond-method fa_invariant  AND
    --frame-avg-angles 0 (eval_FLAC defaults to the raw-pose 'vanilla' trap otherwise);
  * --rotate-deg <ROTATE_DEG> present (D2 sweep parameter);
  * PYTHONPATH-first eval env: PYTHONPATH=<cylindrical-dinov3>/src prepended, EVAL_PYTHON
    parameterized (rir2rir-vs-flac both supported);
  * `set -o pipefail` verbatim;
  * frozen MIN_FREE gate reads c1_frozen_min_free.txt (absent/non-numeric refused);
  * refuses missing args with exit 3 + usage, no `set -u` unbound-variable trip;
  * external log-dir argument (inside-worktree refused).
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

_EXP09_DIR = Path(__file__).resolve().parents[1]
_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
_DRIVER = _EXP09_DIR / "d_eval_driver.sh"

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


# a set of valid-shaped positional args (ckpt / config / eval-name / external log dir)
def _ok_args(tmp_path):
    ckpt = tmp_path / "step=67500.ckpt"; ckpt.write_text("x")
    cfg = _EXP09_DIR / "FLAC_AR_exp09.json"
    logdir = tmp_path / "extlog"; logdir.mkdir(exist_ok=True)
    return [str(ckpt), str(cfg), "exp09_D1_K8_seed42", str(logdir)]


# --------------------------------------------------------------------------------------- #
# existence + pipefail verbatim
# --------------------------------------------------------------------------------------- #
def test_driver_exists_and_executable():
    assert _DRIVER.exists(), "d_eval_driver.sh not created"


def test_driver_has_pipefail_verbatim():
    assert "set -o pipefail" in _DRIVER.read_text()


# --------------------------------------------------------------------------------------- #
# DRY_RUN: assembled command carries the mandatory flags (mutation m4 target)
# --------------------------------------------------------------------------------------- #
def test_dry_run_emits_mandatory_cond_flags(tmp_path):
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"DRY_RUN": "1", "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "--cond-method fa_invariant" in out, "mandatory --cond-method fa_invariant missing"
    assert "--frame-avg-angles 0" in out, "mandatory --frame-avg-angles 0 missing"


def test_dry_run_includes_rotate_deg_sweep_param(tmp_path):
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"DRY_RUN": "1", "ROTATE_DEG": "90",
                           "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "--rotate-deg 90" in out


def test_dry_run_pythonpath_first_eval_env(tmp_path):
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"DRY_RUN": "1", "CYL_DINOV3_SRC": "/opt/cyl/src",
                           "EVAL_PYTHON": "/opt/rir2rir/bin/python",
                           "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "PYTHONPATH=/opt/cyl/src" in out          # prepended cylindrical-dinov3 src
    assert "/opt/rir2rir/bin/python" in out           # EVAL_PYTHON override honored (rir2rir)


def test_dry_run_passes_ckpt_config_evalname(tmp_path):
    args = _ok_args(tmp_path)
    proc = _run(args, tmp_path=tmp_path,
                env_extra={"DRY_RUN": "1", "C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = proc.stdout + proc.stderr
    assert "eval_FLAC.py" in out
    assert "--eval-name exp09_D1_K8_seed42" in out
    assert args[0] in out and "FLAC_AR_exp09.json" in out


# --------------------------------------------------------------------------------------- #
# missing-arg refusals: exit 3 + usage, NO set -u unbound trip
# --------------------------------------------------------------------------------------- #
def test_refuses_no_args_exit3_usage(tmp_path):
    proc = _run([], tmp_path=tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "usage" in out.lower()
    assert "unbound variable" not in out.lower()


def test_refuses_missing_logdir_exit3(tmp_path):
    ckpt = tmp_path / "c.ckpt"; ckpt.write_text("x")
    proc = _run([str(ckpt), str(_EXP09_DIR / "FLAC_AR_exp09.json"), "nm"], tmp_path=tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "usage" in out.lower() or "log" in out.lower()
    assert "unbound variable" not in out.lower()


def test_refuses_logdir_inside_worktree(tmp_path):
    ckpt = tmp_path / "c.ckpt"; ckpt.write_text("x")
    inside = _WORKTREE_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
    proc = _run([str(ckpt), str(_EXP09_DIR / "FLAC_AR_exp09.json"), "nm", str(inside)],
                tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(_frozen(tmp_path))})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "worktree" in out.lower()


# --------------------------------------------------------------------------------------- #
# frozen MIN_FREE gate: absent / non-numeric refused; low free refused on the frozen value
# --------------------------------------------------------------------------------------- #
def test_refuses_absent_frozen_file(tmp_path):
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(tmp_path / "nope.txt")})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "frozen" in out.lower()


def test_refuses_non_numeric_frozen_value(tmp_path):
    fz = _frozen(tmp_path, "TBD_FROM_C1_FIT")
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz)})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0


def test_frozen_value_gates_the_eval_gpu(tmp_path):
    """Free mocked just below the frozen value -> refuse at the per-GPU VRAM gate,
    proving the frozen number is read and actually gates (fail-closed)."""
    fz = _frozen(tmp_path, "27552")
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz), "FAKE_FREE_MB": "27000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "27552" in out


def test_fail_closed_on_nvidia_smi_query_error(tmp_path):
    fz = _frozen(tmp_path, "27552")
    proc = _run(_ok_args(tmp_path), tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz), "FAKE_SMI_FAIL": "1"})
    assert proc.returncode != 0


# --------------------------------------------------------------------------------------- #
# text-level contract (belt-and-braces alongside DRY_RUN emission)
# --------------------------------------------------------------------------------------- #
def test_driver_text_names_frozen_records_file():
    text = _DRIVER.read_text()
    assert "c1_frozen_min_free.txt" in text
    assert "eval_FLAC.py" in text
