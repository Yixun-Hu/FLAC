"""exp-09 C1/launch/screen shell-script gate tests (integrative-review findings 2,3,4,6).

CPU-only. Every test drives a REFUSAL path that exits BEFORE any `python train.py` /
`eval_FLAC.py` launch, so no GPU is ever used and no training runs. The nvidia-smi seam
is a fake on PATH that emits controlled free/used memory; `CUDA_VISIBLE_DEVICES=""`.

Covered:
* c1_fit.sh   — requires an EXTERNAL log dir ($1); bootstrap gate free >= 21900/GPU.
* c1_smoke.sh — refuses absent / TBD / non-numeric FROZEN MIN_FREE_MB; gates both GPUs.
* exp09_launch.sh — binds MIN_FREE_MB to the frozen-records value EXACTLY (mismatched
  numeric refused; absent frozen file refused); --ckpt-path passthrough documented.
* exp09_screen.sh — free-memory gate refuses low headroom; C2 co-tenant-prohibition note.

Mutation sweep (named, RED, restored) lives in test_c1_mutation_sweep.py.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_EXP09_DIR = Path(__file__).resolve().parents[1]
_WORKTREE_ROOT = Path(__file__).resolve().parents[4]

_FIT = _EXP09_DIR / "c1_fit.sh"
_SMOKE = _EXP09_DIR / "c1_smoke.sh"
_LAUNCH = _EXP09_DIR / "exp09_launch.sh"
_SCREEN = _EXP09_DIR / "exp09_screen.sh"


# ------------------------------------------------------------------------------------- #
# fake nvidia-smi on PATH + a run helper
# ------------------------------------------------------------------------------------- #
_FAKE_SMI = """\
#!{python}
import os, sys
args = sys.argv[1:]
def has(s): return any(s in a for a in args)
if os.environ.get("FAKE_SMI_FAIL") == "1":
    sys.exit(1)                                   # simulate a query error (fail-closed path)
if has("--query-compute-apps"):
    sys.exit(0)                                   # no co-tenants
if has("--query-gpu=memory.free"):
    idx = None
    for i, a in enumerate(args):
        if a == "-i" and i + 1 < len(args):
            idx = args[i + 1]
    free = os.environ.get("FAKE_FREE_MB_%s" % idx, os.environ.get("FAKE_FREE_MB", "999999"))
    sys.stdout.write(free + "\\n")
    sys.exit(0)
if has("index"):                                  # index,memory.used[,...] disclosures
    used = os.environ.get("FAKE_USED_MB", "1000")
    for g in (0, 1):
        sys.stdout.write("%d, %s, %s\\n" % (g, used, used))
    sys.exit(0)
sys.exit(0)
"""


def _make_fake_smi(tmp_path: Path) -> Path:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    smi = d / "nvidia-smi"
    smi.write_text(_FAKE_SMI.format(python=sys.executable))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _run(script: Path, args, *, tmp_path, env_extra=None, timeout=120):
    fake_bin = _make_fake_smi(tmp_path)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["LOGGER"] = "none"                              # skip the wandb identity gate
    env["EXP09_LOG_DIR"] = str(tmp_path)               # never tee into the worktree
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        ["bash", str(script), *map(str, args)],
        cwd=str(_WORKTREE_ROOT), env=env, capture_output=True, text=True, timeout=timeout,
    )
    return proc


# ------------------------------------------------------------------------------------- #
# c1_fit.sh
# ------------------------------------------------------------------------------------- #
def test_c1_fit_exists_and_has_pipefail_verbatim():
    assert _FIT.exists(), "c1_fit.sh not created"
    assert "set -o pipefail" in _FIT.read_text(), "c1_fit.sh must contain `set -o pipefail` verbatim"


def test_c1_fit_refuses_absent_log_dir(tmp_path):
    proc = _run(_FIT, [], tmp_path=tmp_path)
    assert proc.returncode != 0
    assert "log" in (proc.stdout + proc.stderr).lower()


def test_c1_fit_refuses_log_dir_inside_worktree(tmp_path):
    inside = _WORKTREE_ROOT / "worklog" / "worklog_yixun" / "exp_09_cyl_no_ssl"
    proc = _run(_FIT, [str(inside)], tmp_path=tmp_path)
    assert proc.returncode != 0
    assert "worktree" in (proc.stdout + proc.stderr).lower()


def test_c1_fit_bootstrap_gate_refuses_low_free(tmp_path):
    logdir = tmp_path / "extlog"; logdir.mkdir()
    proc = _run(_FIT, [str(logdir)], tmp_path=tmp_path, env_extra={"FAKE_FREE_MB": "1000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "21900" in out and "refus" in out.lower()


def test_c1_fit_bootstrap_threshold_is_21900(tmp_path):
    assert "21900" in _FIT.read_text()


# ------------------------------------------------------------------------------------- #
# c1_smoke.sh — refuses absent / TBD / non-numeric MIN_FREE_MB
# r2 LOW: absent args must hit the intended REFUSE (exit 3 + usage), NOT a `set -u`
# 'unbound variable' trip (exit 1). Assert the exit CODE and the message, not just nonzero.
# ------------------------------------------------------------------------------------- #
def test_c1_smoke_refuses_absent_min_free(tmp_path):
    proc = _run(_SMOKE, [], tmp_path=tmp_path)               # $1 (MIN_FREE_MB) omitted entirely
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "usage" in out.lower()
    assert "unbound variable" not in out.lower(), "set -u trip instead of the intended refusal"


def test_c1_smoke_refuses_missing_logdir(tmp_path):
    proc = _run(_SMOKE, ["27552"], tmp_path=tmp_path)        # $1 given, $2 (log dir) omitted
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "log dir" in out.lower() or "usage" in out.lower()
    assert "unbound variable" not in out.lower()


def test_c1_smoke_refuses_tbd_min_free(tmp_path):
    logdir = tmp_path / "extlog"; logdir.mkdir()
    proc = _run(_SMOKE, ["TBD_FROM_C1_FIT_PROBE", str(logdir)], tmp_path=tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "numeric" in out.lower() or "TBD" in out
    assert "unbound variable" not in out.lower()


def test_c1_smoke_refuses_non_numeric_min_free(tmp_path):
    logdir = tmp_path / "extlog"; logdir.mkdir()
    proc = _run(_SMOKE, ["not_a_number", str(logdir)], tmp_path=tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "unbound variable" not in out.lower()


def test_c1_smoke_gates_both_gpus_on_frozen_value(tmp_path):
    """A numeric frozen MIN_FREE_MB that exceeds the (mocked) free memory must refuse at the
    per-GPU gate — proving the frozen number actually gates both GPUs."""
    logdir = tmp_path / "extlog"; logdir.mkdir()
    proc = _run(_SMOKE, ["27552", str(logdir)], tmp_path=tmp_path,
                env_extra={"FAKE_FREE_MB": "1000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "27552" in out


def test_c1_smoke_refuses_low_free_on_gpu1_only(tmp_path):
    """Fail-closed per-GPU: GPU0 has headroom, GPU1 does not -> still refuse."""
    logdir = tmp_path / "extlog"; logdir.mkdir()
    proc = _run(_SMOKE, ["20000", str(logdir)], tmp_path=tmp_path,
                env_extra={"FAKE_FREE_MB_0": "999999", "FAKE_FREE_MB_1": "1000"})
    assert proc.returncode != 0


# ------------------------------------------------------------------------------------- #
# exp09_launch.sh — exact-match frozen binding
# ------------------------------------------------------------------------------------- #
def _frozen_file(tmp_path, value="27552"):
    f = tmp_path / "c1_frozen_min_free.txt"
    f.write_text(value + "\n")
    return f


def test_launch_refuses_when_frozen_file_absent(tmp_path):
    proc = _run(_LAUNCH, [], tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(tmp_path / "nope.txt")})
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "frozen" in out.lower()


def test_launch_refuses_mismatched_numeric_min_free(tmp_path):
    """The core of finding 3/5: an ARBITRARY numeric override != the frozen value is refused
    (the old gate accepted any numeric MIN_FREE_MB)."""
    fz = _frozen_file(tmp_path, "27552")
    proc = _run(_LAUNCH, [], tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz), "MIN_FREE_MB": "21900"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "27552" in out and "21900" in out


def test_launch_binds_exact_frozen_value(tmp_path):
    """MIN_FREE_MB unset -> bound to the frozen value; proven by the per-GPU VRAM gate then
    refusing against the FROZEN number (free mocked just below it)."""
    fz = _frozen_file(tmp_path, "27552")
    proc = _run(_LAUNCH, [], tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz), "FAKE_FREE_MB": "27000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "27552" in out           # the gate used the frozen value, not B-F's 21900


def test_launch_accepts_matching_numeric_min_free(tmp_path):
    """MIN_FREE_MB set == frozen is accepted (proceeds to the VRAM gate, which then refuses
    only because free is mocked low) — distinct from the mismatch refusal."""
    fz = _frozen_file(tmp_path, "27552")
    proc = _run(_LAUNCH, [], tmp_path=tmp_path,
                env_extra={"C1_FROZEN_MIN_FREE_FILE": str(fz), "MIN_FREE_MB": "27552",
                           "FAKE_FREE_MB": "1000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "free 1000" in out.lower() or "1000 mib" in out.lower() or "27552" in out


def test_launch_documents_ckpt_path_passthrough():
    text = _LAUNCH.read_text()
    assert "--ckpt-path" in text, "launch must document/pass --ckpt-path (train.py:230 resume)"
    assert "CKPT_PATH" in text


# ------------------------------------------------------------------------------------- #
# exp09_screen.sh — free-memory gate + co-tenant prohibition note
# ------------------------------------------------------------------------------------- #
def test_screen_refuses_absent_headroom(tmp_path):
    proc = _run(_SCREEN, ["2500"], tmp_path=tmp_path)   # step given, headroom missing
    assert proc.returncode != 0


def test_screen_refuses_non_numeric_headroom(tmp_path):
    proc = _run(_SCREEN, ["2500", "abc"], tmp_path=tmp_path)
    assert proc.returncode != 0


def test_screen_gate_refuses_low_headroom(tmp_path):
    """The new free-memory gate refuses when GPU1 free < required headroom, BEFORE any eval."""
    proc = _run(_SCREEN, ["2500", "20000"], tmp_path=tmp_path,
                env_extra={"FAKE_FREE_MB": "1000"})
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "20000" in out or "free" in out.lower()


def test_screen_has_cotenant_prohibition_note():
    text = _SCREEN.read_text().upper()
    assert "C1" in text and "PROHIBIT" in text and "CO-TENANT" in text.replace(" ", "-")
