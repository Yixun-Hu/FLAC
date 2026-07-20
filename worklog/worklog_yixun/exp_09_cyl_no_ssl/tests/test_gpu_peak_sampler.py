"""exp-09 C1 external GPU peak sampler tests (integrative-review finding 2, item 1).

CPU-only, mocked nvidia-smi seam (``NVIDIA_SMI`` env -> a fake that emits controlled
CSV). No GPU is ever touched. Covers the review's named cases:

* sampler zero-sample FAILURE (fail-closed): any GPU with no valid sample -> refuse;
* sampler peak math: per-GPU peak == max used; derived gate == max(peaks) + 4096;
* peak tracks the running MAX across ticks (a transient mid-run peak is captured);
* end-to-end main(): atomic JSON written, exit code semantics.

Mutation sweep (named, RED, restored) lives in test_c1_mutation_sweep.py.
"""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

import gpu_peak_sampler as gps  # noqa: E402


# ------------------------------------------------------------------------------------- #
# fake nvidia-smi seam
# ------------------------------------------------------------------------------------- #
_FAKE_TEMPLATE = """\
#!{python}
import os, sys
# Emit `index, used_mib` CSV lines (--format=csv,noheader,nounits) for GPUs 0 and 1.
if os.environ.get("FAKE_FAIL") == "1":
    sys.exit(1)                      # a dead/failed nvidia-smi -> no valid sample
seq0 = [int(x) for x in os.environ.get("FAKE_SEQ_0", "0").split(",")]
seq1 = [int(x) for x in os.environ.get("FAKE_SEQ_1", "0").split(",")]
state = os.environ["FAKE_STATE"]
try:
    with open(state) as f:
        n = int(f.read().strip() or "0")
except OSError:
    n = 0
with open(state, "w") as f:
    f.write(str(n + 1))
i0 = min(n, len(seq0) - 1)           # clamp at the last value once the sequence runs out
i1 = min(n, len(seq1) - 1)
sys.stdout.write("0, %d\\n1, %d\\n" % (seq0[i0], seq1[i1]))
"""


def _write_fake_nvidia_smi(tmp_path: Path, seq0="0", seq1="0") -> dict:
    fake = tmp_path / "nvidia-smi"
    fake.write_text(_FAKE_TEMPLATE.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    state = tmp_path / "fake_state.txt"
    env = dict(os.environ)
    env["NVIDIA_SMI"] = str(fake)
    env["FAKE_STATE"] = str(state)
    env["FAKE_SEQ_0"] = seq0
    env["FAKE_SEQ_1"] = seq1
    env.pop("FAKE_FAIL", None)
    return env


# ------------------------------------------------------------------------------------- #
# 1. fail-closed: zero valid samples on a GPU -> sampler_ok False, no derived gate
# ------------------------------------------------------------------------------------- #
def test_zero_sample_failure_pure():
    """assemble_record: if ANY requested GPU has 0 valid samples the sampler is NOT ok and
    the derived gate is None — a dead sampler must never yield a (tiny) usable gate."""
    rec = gps.assemble_record(
        gpus=[0, 1], peaks={0: 12345, 1: None}, counts={0: 5, 1: 0},
        child_exit=0, interval_s=1.0, gate_margin_mib=4096,
        child_argv=["true"], started_at=0.0, ended_at=1.0,
    )
    assert rec["sampler_ok"] is False
    assert rec["derived_gate_mib"] is None
    assert rec["samples_per_gpu"] == {"0": 5, "1": 0}


def test_zero_sample_failure_end_to_end(tmp_path):
    """main() with a nvidia-smi that always fails: JSON is still written (atomically),
    sampler_ok False, derived gate null, and the process exits with the fail-closed code 3."""
    env = _write_fake_nvidia_smi(tmp_path)
    env["FAKE_FAIL"] = "1"
    out = tmp_path / "peak.json"
    old = os.environ.copy()
    try:
        os.environ.clear(); os.environ.update(env)
        rc = gps.main(["--out", str(out), "--interval", "0.02", "--gpus", "0,1",
                       "--", sys.executable, "-c", "import time; time.sleep(0.15)"])
    finally:
        os.environ.clear(); os.environ.update(old)
    assert rc == gps.SAMPLER_FAIL_EXITCODE
    rec = json.loads(out.read_text())
    assert rec["sampler_ok"] is False
    assert rec["derived_gate_mib"] is None
    assert rec["samples_per_gpu"]["0"] == 0 and rec["samples_per_gpu"]["1"] == 0


# ------------------------------------------------------------------------------------- #
# 2. peak math: derived gate == max(peak0, peak1) + 4096
# ------------------------------------------------------------------------------------- #
def test_peak_math_and_plus_4096_pure():
    rec = gps.assemble_record(
        gpus=[0, 1], peaks={0: 1000, 1: 2000}, counts={0: 7, 1: 7},
        child_exit=0, interval_s=1.0, gate_margin_mib=gps.DEFAULT_GATE_MARGIN_MIB,
        child_argv=["true"], started_at=0.0, ended_at=1.0,
    )
    assert rec["sampler_ok"] is True
    assert rec["per_gpu_peaks_mib"] == {"0": 1000, "1": 2000}
    # max(1000, 2000) + 4096
    assert rec["derived_gate_mib"] == 2000 + 4096 == 6096


def test_default_gate_margin_is_4096():
    assert gps.DEFAULT_GATE_MARGIN_MIB == 4096


# ------------------------------------------------------------------------------------- #
# 3. peak tracks the running MAX across ticks (transient mid-run peak captured)
# ------------------------------------------------------------------------------------- #
def test_peak_tracks_running_max(monkeypatch):
    """Drive _Sampler._sample_once with a controlled sequence: a transient spike in the
    MIDDLE must be the recorded peak even though the final sample is lower."""
    seq = {0: [1000, 5000, 2000, 2000], 1: [1500, 6000, 2500, 2500]}
    calls = {"n": 0}

    def fake_query(gpus):
        i = min(calls["n"], len(seq[0]) - 1)
        calls["n"] += 1
        return {g: seq[g][i] for g in gpus}

    monkeypatch.setattr(gps, "query_used_mib", fake_query)
    s = gps._Sampler([0, 1], interval_s=999)   # never auto-fires; we call manually
    for _ in range(4):
        s._sample_once()
    assert s.peaks == {0: 5000, 1: 6000}
    assert s.counts == {0: 4, 1: 4}


def test_end_to_end_peak_via_fake_nvidia_smi(tmp_path):
    """Full main() path through a real subprocess + the fake nvidia-smi emitting a
    spike sequence: peaks == the sequence max, derived gate == max+4096, exit == child_exit."""
    env = _write_fake_nvidia_smi(tmp_path, seq0="1000,5000,2000", seq1="1500,6000,2500")
    out = tmp_path / "peak.json"
    old = os.environ.copy()
    try:
        os.environ.clear(); os.environ.update(env)
        rc = gps.main(["--out", str(out), "--interval", "0.01", "--gpus", "0,1",
                       "--", sys.executable, "-c", "import time; time.sleep(0.4)"])
    finally:
        os.environ.clear(); os.environ.update(old)
    rec = json.loads(out.read_text())
    assert rec["sampler_ok"] is True
    assert rec["samples_per_gpu"]["0"] >= 3 and rec["samples_per_gpu"]["1"] >= 3
    assert rec["per_gpu_peaks_mib"] == {"0": 5000, "1": 6000}
    assert rec["derived_gate_mib"] == 6000 + 4096
    assert rec["child_exit"] == 0
    assert rc == 0


def test_child_failure_propagates_exit_code(tmp_path):
    """When the sampler is healthy but the CHILD fails, main() propagates the child exit."""
    env = _write_fake_nvidia_smi(tmp_path, seq0="1000", seq1="2000")
    out = tmp_path / "peak.json"
    old = os.environ.copy()
    try:
        os.environ.clear(); os.environ.update(env)
        rc = gps.main(["--out", str(out), "--interval", "0.01", "--gpus", "0,1",
                       "--", sys.executable, "-c", "import sys; sys.exit(7)"])
    finally:
        os.environ.clear(); os.environ.update(old)
    rec = json.loads(out.read_text())
    assert rec["sampler_ok"] is True       # sampler worked
    assert rec["child_exit"] == 7
    assert rc == 7                          # child failure surfaced


# ------------------------------------------------------------------------------------- #
# 4. no partial output / arg validation
# ------------------------------------------------------------------------------------- #
def test_missing_child_command_errors(tmp_path):
    with pytest.raises(SystemExit):
        gps.main(["--out", str(tmp_path / "x.json"), "--interval", "0.1", "--gpus", "0,1"])


def test_atomic_no_temp_left_behind(tmp_path):
    env = _write_fake_nvidia_smi(tmp_path, seq0="1000", seq1="2000")
    out = tmp_path / "peak.json"
    old = os.environ.copy()
    try:
        os.environ.clear(); os.environ.update(env)
        gps.main(["--out", str(out), "--interval", "0.01", "--gpus", "0,1",
                  "--", sys.executable, "-c", "pass"])
    finally:
        os.environ.clear(); os.environ.update(old)
    leftover = [n for n in os.listdir(out.parent) if n.startswith(gps.OUTPUT_TMP_PREFIX)]
    assert leftover == [], f"atomic-writer temp left behind: {leftover}"
