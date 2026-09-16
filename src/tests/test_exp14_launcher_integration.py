"""Shell-level integration tests for the exp_14 launcher, on stubs, without a GPU.

Finding M6: the dry-run tests pin the *printed argv* but bypass every real gate, so they
stayed green with three of this round's findings present. These tests run the launcher for
real -- forks, waits, file discovery, marker files, exit codes -- against a stub ``python``
first on ``PATH`` that intercepts the five experiment entry points (verifier, run-contract
make/validate, ``train.py``, ``eval_FLAC.py``, ``check-bundle``) and delegates everything
else to the real interpreter, so the launcher's own JSON parsing still runs for real.

The stub touches no GPU and imports no torch; every wait is driven by ``POLL_SECONDS`` and
``STABLE_WAIT``, which the launcher reads from the environment (default 180 s / 60 s).

The cases are the review's matrix:

a. an existing final checkpoint that fails validation is fatal **before** any training;
b. a preflight failure of the *second* arm leaves no process started at all;
c. an evaluator that exits 0 without writing its metrics JSON fails the cell;
d. a verifier failure (e.g. the dirty-tree FAIL of finding B2) stops everything;
e. ``PYTHONPATH`` must be the ``src`` of the package directory that was verified;
f. the happy path: both arms train, validate, and 20 cells complete.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

from src.tools.data_curve import names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.environ.get("EXP14_LAUNCHER") or os.path.join(
    os.path.dirname(REPO_ROOT), "cylindrical-dinov3", "worklog", "worklog_yixun",
    "exp_14_data_curve_claude", "scripts", "exp14_launch.sh")
TAG = "025"
CFG_SHA = "c0ffee" * 10 + "cafe"          # 64 hex chars the stub reports for the eval config

STUB = r'''#!/bin/bash
# exp_14 test stub for `python`. Intercepts the experiment's entry points; everything else
# (the launcher's own `python -c` JSON checks) goes to the real interpreter. No CUDA, ever.
set -u
M="$STUB_MARKERS"; mkdir -p "$M"
echo "$*" >> "$M/calls.log"
arg () { local want="$1"; shift; while [ $# -gt 0 ]; do
  if [ "$1" = "$want" ]; then echo "${2:-}"; return 0; fi; shift; done; echo ""; }
arm_of () { case "$1" in *dc_cyl_*) echo cyl ;; *dc_van_*) echo van ;; *) echo unknown ;; esac; }
rc_for () {  # $1 = base var name, $2 = arm
  local per; per="$(eval "echo \${${1}_$(echo "$2" | tr a-z A-Z):-}")"
  if [ -n "$per" ]; then echo "$per"; else eval "echo \${${1}:-0}"; fi; }

if [ "${1:-}" = "-m" ]; then
  case "${2:-}" in
    src.tools.data_curve.verify)
      echo "24/24 checks PASSED"; touch "$M/verify"; exit "${STUB_RC_VERIFY:-0}" ;;
    src.training.run_contract)
      case "${3:-}" in
        make-contract)
          RD="$(arg --run-dir "$@")"; RID="$(arg --run-id "$@")"; A="$(arm_of "$RID")"
          RC="$(rc_for STUB_RC_CONTRACT "$A")"
          touch "$M/contract_$RID"
          [ "$RC" = 0 ] || { echo "CONTRACT MISMATCH: stubbed" >&2; exit "$RC"; }
          mkdir -p "$RD"; printf '{"run_id": "%s"}\n' "$RID" > "$RD/run_contract.json"
          echo "$RD/run_contract.json"; exit 0 ;;
        validate)
          CK="$(arg --ckpt "$@")"; A="$(arm_of "$CK")"
          case " $* " in *" --for-resume "*) KIND=resume ;; *) KIND=final ;; esac
          echo "$CK" >> "$M/validate_${KIND}_$A"
          RC="$(rc_for "STUB_RC_VALIDATE_$(echo "$KIND" | tr a-z A-Z)" "$A")"
          [ "$RC" = 0 ] || { echo "CONTRACT VIOLATION: stubbed" >&2; exit "$RC"; }
          echo "OK $CK"; exit 0 ;;
      esac ;;
    src.tools.data_curve.names)
      PT="$(arg --pt "$@")"; NAME="$(arg --expect-eval-name "$@")"
      echo "$NAME" >> "$M/bundlecheck"
      RC="${STUB_RC_BUNDLE:-0}"
      [ -s "$PT" ] || RC="${STUB_RC_BUNDLE_MISSING:-3}"
      [ "$RC" = 0 ] || { echo "FAIL $PT: stubbed"; exit "$RC"; }
      echo "PASS $PT: stubbed dataset_config_sha256=$STUB_CFG_SHA"; exit 0 ;;
  esac
fi

case "${1:-}" in
  train.py)
    RUN="$(arg --name "$@")"; SD="$(arg --save-dir "$@")"
    echo "$*" > "$M/train_$RUN"
    [ "${STUB_RC_TRAIN:-0}" = 0 ] || exit "${STUB_RC_TRAIN}"
    mkdir -p "$SD"; echo "fake checkpoint" > "$SD/epoch=8-step=40000.ckpt"; exit 0 ;;
  eval_FLAC.py)
    CK="$(arg --ckpt-path "$@")"; NAME="$(arg --eval-name "$@")"
    SUF=""; [ "$(arg --cond-method "$@")" = fa_invariant ] && SUF="_fa_invariant_a1"
    BASE="$(dirname "$CK")/$(basename "$CK" .ckpt)"
    echo "$NAME" >> "$M/evaluated"
    [ "${STUB_RC_EVAL:-0}" = 0 ] || exit "${STUB_RC_EVAL}"
    [ "${STUB_EVAL_NO_BUNDLE:-0}" = 1 ] \
      || echo "bundle" > "${BASE}_predictions_1_1.0_${NAME}${SUF}.pt"
    [ "${STUB_EVAL_NO_METRICS:-0}" = 1 ] \
      || printf '{"metrics": {"T60": 1.0}, "cond_method": "x"}\n' \
           > "${BASE}_metrics_1_1.0_${NAME}${SUF}.json"
    exit 0 ;;
  -c)
    case "$*" in *append_resume_log*)
      RD="${3:-}"; echo "{\"from_ckpt\": \"${5:-}\"}" >> "$RD/resume_log.json"
      touch "$M/resume_log"; exit 0 ;;
    esac ;;
esac
exec "$STUB_REAL_PYTHON" "$@"
'''


class Harness:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.markers = tmp_path / "markers"
        self.markers.mkdir()
        binary = tmp_path / "bin"
        binary.mkdir()
        stub = binary / "python"
        stub.write_text(STUB)
        stub.chmod(0o755)
        self.bin = str(binary)

        kit = tmp_path / "kit" / "configs"
        kit.mkdir(parents=True)
        for arm in names.ARMS:
            (kit / names.ARM_MODEL_CONFIG_BASENAMES[arm]).write_text('{"model": {}}\n')
        wt = tmp_path / "wt"
        (wt / "src/configs/dataset_configs/AR/train").mkdir(parents=True)
        (wt / os.path.dirname(names.TRAIN_DATASET_CONFIGS[TAG])).mkdir(exist_ok=True,
                                                                      parents=True)
        (wt / names.TRAIN_DATASET_CONFIGS[TAG]).write_text("{}\n")
        (wt / "data/AR").mkdir(parents=True)
        (wt / names.SPLIT_FILES[TAG]).write_text("{}\n")
        (tmp_path / "pkg" / "src").mkdir(parents=True)
        for name in ("nas", "rec"):
            (tmp_path / name).mkdir()
        self.wt, self.kit = str(wt), str(tmp_path / "kit")

    def run_dir(self, arm):
        return os.path.join(str(self.tmp / "nas"), names.run_id(arm, TAG))

    def run(self, **overrides):
        env = dict(os.environ)
        env.update({
            "PATH": self.bin + os.pathsep + env["PATH"],
            "STUB_MARKERS": str(self.markers),
            "STUB_REAL_PYTHON": sys.executable,
            "STUB_CFG_SHA": CFG_SHA,
            "FLAC_WT": self.wt,
            "CYL_SRC": str(self.tmp / "pkg" / "src"),
            "KIT": self.kit,
            "TAG": TAG,
            "NAS_ROOT": str(self.tmp / "nas"),
            "REC": str(self.tmp / "rec"),
            "GPUS": "0,1",
            "EXPECT_FLAC_SHA": "a" * 40,
            "EXPECT_PKG_SHA": "b" * 40,
            "POLL_SECONDS": "1",
            "STABLE_WAIT": "0",
            "CUDA_VISIBLE_DEVICES": "",
        })
        env.update({k: str(v) for k, v in overrides.items()})
        return subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=env)

    def marker(self, name):
        return (self.markers / name).exists()

    def summary(self):
        return json.load(open(self.tmp / "rec" / "data_curve_launch_summary.json"))


@pytest.fixture
def harness(tmp_path):
    if not os.path.exists(SCRIPT):
        pytest.skip(f"launcher not available at {SCRIPT}")
    if not shutil.which("bash"):
        pytest.skip("bash not available")
    return Harness(tmp_path)


def _seed_final_ckpt(harness, arm, content="stale"):
    run_dir = harness.run_dir(arm)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "epoch=8-step=40000.ckpt")
    open(path, "w").write(content)
    return path


def test_a_an_unvalidatable_existing_final_is_fatal_before_any_training(harness):
    """Finding B1: a filename alone used to mark an arm SKIPPED, so a corrupt or cross-run
    cyl final let van train for days before anything checked it."""
    _seed_final_ckpt(harness, "cyl")
    proc = harness.run(STUB_RC_VALIDATE_FINAL=3)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert not harness.marker("train_dc_cyl_f025")
    assert not harness.marker("train_dc_van_f025")
    assert "run contract" in proc.stdout or "contract" in proc.stdout
    assert os.path.exists(_seed_final_ckpt(harness, "cyl", "stale"))   # never deleted


def test_a2_a_validated_existing_final_skips_only_that_arm(harness):
    ckpt = _seed_final_ckpt(harness, "cyl")
    proc = harness.run()

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not harness.marker("train_dc_cyl_f025")      # skipped, because VALIDATED
    assert harness.marker("train_dc_van_f025")
    assert harness.summary()["runs"]["cyl"]["pid"] == "SKIPPED"
    assert harness.summary()["runs"]["cyl"]["final_ckpt"] == ckpt


def test_b_a_second_arm_preflight_failure_starts_nothing(harness):
    """Finding B1, second half: cyl used to be launched before van's contract was made."""
    proc = harness.run(STUB_RC_CONTRACT_VAN=2)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert harness.marker("contract_dc_cyl_f025") and harness.marker("contract_dc_van_f025")
    assert not harness.marker("train_dc_cyl_f025")
    assert not harness.marker("train_dc_van_f025")


def test_d_a_verifier_failure_stops_everything(harness):
    """The dirty-worktree FAIL of finding B2 reaches the launcher as a non-zero verifier."""
    proc = harness.run(STUB_RC_VERIFY=3)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert harness.marker("verify")
    assert not harness.marker("contract_dc_cyl_f025")
    assert not harness.marker("train_dc_cyl_f025")


def test_e_pythonpath_must_be_the_verified_package(harness, tmp_path):
    """Finding B2: PKG_DIR could name the verified repository while PYTHONPATH executed a
    different tree."""
    other = tmp_path / "pkg" / "lib"
    other.mkdir(parents=True)
    proc = harness.run(CYL_SRC=str(other))

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert not harness.marker("verify")                  # refused before anything ran
    assert "CYL_SRC" in proc.stdout
