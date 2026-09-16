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
f. the happy path: both arms train, validate, and 20 cells complete;
g. a second invocation skips the completed cells;
h. a lane that is killed mid-way fails the pipeline, whatever the cells before it did
   (round-F finding 1: bare ``wait`` returned success for a lane that never finished);
i. an artifact naming a foreign checkpoint is refused (round-F finding 2);
j. ... and is therefore re-evaluated rather than skipped;
k. two launchers for the same TAG cannot run at once (round-F finding 5) -- including
   two that use different REC directories, because the lock belongs to the run target;
l. artifacts with the right checkpoint PATH but a stale embedded digest are refused and
   the cell is re-evaluated (codex full-r2 finding 1).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

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
      case "${3:-}" in
        check-bundle)
          PT="$(arg --pt "$@")"; NAME="$(arg --expect-eval-name "$@")"
          XC="$(arg --expect-ckpt "$@")"; XS="$(arg --expect-ckpt-sha256 "$@")"
          echo "$NAME" >> "$M/bundlecheck"
          # round-F 2: the launcher must bind every artifact to the validated checkpoint
          [ -n "$XC" ] && [ -n "$XS" ] || { echo "FAIL $PT: unbound check-bundle"; exit 2; }
          RC="${STUB_RC_BUNDLE:-0}"; FROM=""; FROMSHA=""
          [ -s "$PT" ] || RC="${STUB_RC_BUNDLE_MISSING:-3}"
          # the stub evaluator writes, as the bundle's body, the checkpoint it scored
          # and the digest of the bytes it loaded (full-r2 1): "<path> <sha256>"
          [ -s "$PT" ] && { read -r FROM FROMSHA < "$PT" || true; }
          [ -z "$FROM" ] || [ "$FROM" = "$XC" ] || RC=3
          [ -z "$FROM" ] || [ "$FROMSHA" = "$XS" ] || RC=3
          if [ -e "$XC" ] && [ "$XS" != "$(sha256sum "$XC" | cut -d" " -f1)" ]; then RC=3; fi
          [ "$RC" = 0 ] || { echo "FAIL $PT: stubbed (from='$FROM' expected='$XC')"; exit "$RC"; }
          echo "PASS $PT: stubbed dataset_config_sha256=$STUB_CFG_SHA ckpt_sha256=$XS"
          exit 0 ;;
        check-metrics)
          J="$(arg --json "$@")"; XC="$(arg --expect-ckpt "$@")"
          XS="$(arg --expect-ckpt-sha256 "$@")"
          echo "$J" >> "$M/metricscheck"
          [ -n "$XC" ] && [ -n "$XS" ] || { echo "FAIL $J: unbound check-metrics"; exit 2; }
          [ -s "$J" ] || { echo "FAIL $J: no metrics JSON"; exit 3; }
          grep -q "\"ckpt_path\": \"$XC\"" "$J" \
            || { echo "FAIL $J: metrics ckpt_path is not $XC"; exit 3; }
          grep -q "\"ckpt_sha256\": \"$XS\"" "$J" \
            || { echo "FAIL $J: metrics ckpt_sha256 is not $XS"; exit 3; }
          echo "PASS $J: stubbed"; exit 0 ;;
      esac ;;
  esac
fi

case "${1:-}" in
  train.py)
    RUN="$(arg --name "$@")"; SD="$(arg --save-dir "$@")"
    echo "$*" > "$M/train_$RUN"
    if [ -n "${STUB_TRAIN_WAIT:-}" ]; then      # hold the launcher inside its lock
      N=0; while [ ! -e "$STUB_TRAIN_WAIT" ] && [ "$N" -lt 600 ]; do sleep 0.1
        N=$((N + 1)); done; fi
    [ "${STUB_RC_TRAIN:-0}" = 0 ] || exit "${STUB_RC_TRAIN}"
    mkdir -p "$SD"; echo "fake checkpoint" > "$SD/epoch=8-step=40000.ckpt"; exit 0 ;;
  eval_FLAC.py)
    CK="$(arg --ckpt-path "$@")"; NAME="$(arg --eval-name "$@")"
    CM="$(arg --cond-method "$@")"; SUF=""
    XS="$(arg --expect-ckpt-sha256 "$@")"
    [ "$CM" = fa_invariant ] && SUF="_fa_invariant_a1"
    BASE="$(dirname "$CK")/$(basename "$CK" .ckpt)"
    echo "$NAME" >> "$M/evaluated"
    # the real evaluator hashes --ckpt-path BEFORE loading it and refuses a mismatch,
    # then stamps the digest of the bytes it loaded into both artifacts (full-r2 1).
    echo "$NAME $XS" >> "$M/evaluated_sha"
    [ -n "$XS" ] || { echo "eval_FLAC: no --expect-ckpt-sha256" >&2; exit 2; }
    if [ "${STUB_EVAL_SWAP_CKPT:-0}" = 1 ]; then    # replaced after the launcher hashed it
      printf 'another contract-valid step=40000 checkpoint' > "$CK"; fi
    SHA="$(sha256sum "$CK" | cut -d" " -f1)"
    [ "$XS" = "$SHA" ] || { echo "eval_FLAC: REFUSED $CK is $SHA not $XS" >&2; exit 4; }
    # a lane that dies mid-way: the evaluator kills the lane subshell that started it
    if [ "${STUB_EVAL_KILL_AT:-}" = "$NAME" ]; then
      echo "$NAME" >> "$M/killed_lane"; kill -9 "$PPID" 2>/dev/null; exit 137; fi
    [ "${STUB_RC_EVAL:-0}" = 0 ] || exit "${STUB_RC_EVAL}"
    WROTE="$CK"
    [ "${STUB_EVAL_FOREIGN_CKPT:-0}" = 1 ] \
      && WROTE="$(dirname "$CK")/foreign_step=40000.ckpt"
    [ "${STUB_EVAL_NO_BUNDLE:-0}" = 1 ] \
      || echo "$WROTE $SHA" > "${BASE}_predictions_1_1.0_${NAME}${SUF}.pt"
    [ "${STUB_EVAL_NO_METRICS:-0}" = 1 ] \
      || printf '{"metrics": {"T60": 1.0}, "ckpt_path": "%s", "ckpt_sha256": "%s", "cond_method": "%s"}\n' \
           "$WROTE" "$SHA" "$CM" > "${BASE}_metrics_1_1.0_${NAME}${SUF}.json"
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
        env.update(self._env(**overrides))
        return subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=env)

    def _env(self, **overrides):
        env = {
            "PATH": self.bin + os.pathsep + os.environ["PATH"],
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
            "EXPECT_KIT_SHA": "c" * 40,
            "POLL_SECONDS": "1",
            "STABLE_WAIT": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
        env.update({k: str(v) for k, v in overrides.items()})
        return env

    def spawn(self, **overrides):
        """The same invocation, backgrounded, for the two-launcher cases."""
        env = dict(os.environ)
        env.update(self._env(**overrides))
        return subprocess.Popen(["bash", SCRIPT], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)

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
    ckpt = _seed_final_ckpt(harness, "cyl")
    proc = harness.run(STUB_RC_VALIDATE_FINAL=3)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert not harness.marker("train_dc_cyl_f025")
    assert not harness.marker("train_dc_van_f025")
    assert "NOT" in proc.stdout and ckpt in proc.stdout     # names what it refused to trust
    assert os.path.exists(ckpt) and open(ckpt).read() == "stale"   # never deleted or touched


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

def test_c_an_evaluator_that_writes_no_metrics_json_fails_the_cell(harness):
    """Finding H3: only the bundle was checked after an evaluator run, so a cell could be
    recorded done without the metrics JSON the results table is built from."""
    proc = harness.run(STUB_EVAL_NO_METRICS=1)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "metrics" in proc.stdout
    summary = harness.summary()
    assert summary["cells_evaluated"] == 0
    assert len(summary["failed_cells"]) == 20


def test_c2_an_evaluator_that_writes_no_bundle_fails_the_cell(harness):
    proc = harness.run(STUB_EVAL_NO_BUNDLE=1)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert harness.summary()["cells_evaluated"] == 0


def test_f_the_happy_path_trains_validates_and_completes_twenty_cells(harness):
    proc = harness.run()

    assert proc.returncode == 0, proc.stdout + proc.stderr
    for arm in names.ARMS:
        assert harness.marker(f"train_dc_{arm}_f{TAG}")
    evaluated = open(harness.markers / "evaluated").read().split()
    assert len(evaluated) == 20 and len(set(evaluated)) == 20
    assert set(evaluated) == {names.eval_name(arm, TAG, k, seed)
                              for arm in names.ARMS for k in names.K_VALUES
                              for seed in names.SEEDS}
    summary = harness.summary()
    assert summary["status"] == "complete"
    assert summary["cells_evaluated"] == 20 and summary["failed_cells"] == []
    # every validated final was checked at step 40000, one per arm (B1 preflight)
    assert len(open(harness.markers / "validate_final_cyl").read().split()) == 1
    # M4: the per-cell record carries the eval config's sha computed at check time
    done = [line for line in open(harness.markers / "bundlecheck").read().split() if line]
    assert len(done) == 40          # once in the lane, once in the final enumeration
    cells = summary["cells"]
    assert len(cells) == 20 and all(CFG_SHA in entry for entry in cells)
    # round-F 1 + 2: the verdict is a re-check of all twenty planned cells, each bound to
    # the digest the launcher computed when it validated that arm's final checkpoint.
    assert summary["cells_final"] == 20
    assert "final enumeration: 20/20" in proc.stdout
    for arm in names.ARMS:
        digest = summary["runs"][arm]["final_ckpt_sha256"]
        assert len(digest) == 64, digest


def _seed_cell_artifacts(harness, arm, ckpt, body, digest=None):
    """Write the two artifacts of every cell of one arm, naming ``body`` as their source
    and ``digest`` as the checkpoint bytes they claim to have been scored from."""
    base = ckpt[:-len(".ckpt")]
    suffix = "_fa_invariant_a1" if arm == "cyl" else ""
    if digest is None:
        digest = hashlib.sha256(open(body, "rb").read()).hexdigest() \
            if os.path.exists(body) else "0" * 64
    for k in names.K_VALUES:
        for seed in names.SEEDS:
            name = names.eval_name(arm, TAG, k, seed)
            open(f"{base}_predictions_1_1.0_{name}{suffix}.pt", "w").write(
                f"{body} {digest}\n")
            open(f"{base}_metrics_1_1.0_{name}{suffix}.json", "w").write(
                '{"metrics": {"T60": 1.0}, "ckpt_path": "%s", "ckpt_sha256": "%s", '
                '"cond_method": "x"}\n' % (body, digest))


def test_h_a_lane_that_dies_after_n_cells_fails_the_pipeline(harness):
    """Round-F finding 1: both lanes were backgrounded and awaited with a bare `wait`,
    which reports success even for a lane that was killed -- so the launcher could write
    COMPLETE with four of ten K=1 cells present and no failure marker anywhere."""
    victim = names.eval_name("cyl", TAG, 1, 44)          # the 5th cell of the K=1 lane
    proc = harness.run(STUB_EVAL_KILL_AT=victim)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "COMPLETE" not in proc.stdout
    assert open(harness.markers / "killed_lane").read().split() == [victim]
    summary = harness.summary()
    assert summary["status"] == "failed"
    assert summary["cells_final"] < 20
    # the K=8 lane ran to the end, so the failure is the lane's, not a cell's
    assert len(open(harness.markers / "evaluated").read().split()) < 20
    assert "lane" in proc.stdout


def test_i_an_artifact_naming_a_foreign_checkpoint_fails_the_cell(harness):
    """Round-F finding 2: metrics_ok proved only that the JSON parsed, so a result from
    another checkpoint was counted as this cell's."""
    proc = harness.run(STUB_EVAL_FOREIGN_CKPT=1)

    assert proc.returncode == 3, proc.stdout + proc.stderr
    summary = harness.summary()
    assert summary["cells_evaluated"] == 0
    assert len(summary["failed_cells"]) == 20
    assert summary["cells_final"] == 0


def test_j_stale_artifacts_from_another_checkpoint_are_re_evaluated_not_skipped(harness):
    for arm in names.ARMS:
        ckpt = _seed_final_ckpt(harness, arm)
        _seed_cell_artifacts(harness, arm, ckpt,
                             os.path.join(os.path.dirname(ckpt), "foreign_step=40000.ckpt"))
    proc = harness.run()

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(open(harness.markers / "evaluated").read().split()) == 20   # none skipped
    summary = harness.summary()
    assert summary["cells_skipped"] == 0 and summary["cells_evaluated"] == 20


def test_k_two_launchers_for_the_same_tag_cannot_run_at_once(harness):
    """Round-F finding 5: an atomic per-tag lock -- and a stale lock is reported, never
    removed by the launcher itself."""
    lock = harness.tmp / "nas" / f".lock_f{TAG}"
    lock.mkdir()
    (lock / "pid").write_text(f"{os.getpid()} this test\n")
    held = harness.run()
    assert held.returncode == 3, held.stdout + held.stderr
    assert not harness.marker("verify")            # refused before anything ran
    assert str(os.getpid()) in held.stdout + held.stderr

    (lock / "pid").write_text("4194303 a launcher that was killed\n")
    stale = harness.run()
    assert stale.returncode == 3
    assert "dead" in stale.stdout + stale.stderr
    assert lock.is_dir()                           # this script deletes nothing of ours

    lock.joinpath("pid").unlink()
    lock.rmdir()
    assert harness.run().returncode == 0           # and the lock was only ever advisory


def test_g_a_second_invocation_skips_the_completed_cells(harness):
    assert harness.run().returncode == 0
    first = len(open(harness.markers / "evaluated").read().split())

    again = harness.run()
    assert again.returncode == 0, again.stdout + again.stderr
    assert len(open(harness.markers / "evaluated").read().split()) == first   # none re-run
    assert again.stdout.count("SKIPPED") >= 20
    assert harness.summary()["cells_skipped"] == 20


def test_f2_every_eval_is_pinned_to_the_validated_checkpoint_digest(harness):
    """Codex full-r2 finding 1: the launcher hands the evaluator the digest it computed
    when it validated that arm's final checkpoint, and the evaluator refuses anything
    else. The stub mirrors both halves, so an unpinned argv fails the whole run."""
    proc = harness.run()
    assert proc.returncode == 0, proc.stdout + proc.stderr

    pinned = [line.split() for line in
              open(harness.markers / "evaluated_sha").read().splitlines() if line]
    assert len(pinned) == 20
    summary = harness.summary()
    for arm in names.ARMS:
        digest = summary["runs"][arm]["final_ckpt_sha256"]
        assert len(digest) == 64
        cells = [sha for name, sha in pinned if name.startswith(f"dc_{arm}_")]
        assert len(cells) == 10 and set(cells) == {digest}


def test_l_artifacts_with_the_right_path_but_stale_bytes_are_refused(harness):
    """THE round-2 case: same pathname, another contract-valid step-40000 checkpoint.
    Path and protocol still match; only the digest the artifacts carry disagrees, so the
    cells must be re-evaluated rather than skipped."""
    for arm in names.ARMS:
        ckpt = _seed_final_ckpt(harness, arm)
        _seed_cell_artifacts(harness, arm, ckpt, ckpt, digest="0" * 64)
    proc = harness.run()

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(open(harness.markers / "evaluated").read().split()) == 20   # none skipped
    summary = harness.summary()
    assert summary["cells_skipped"] == 0 and summary["cells_evaluated"] == 20
    assert summary["cells_final"] == 20


def test_l2_a_checkpoint_replaced_after_validation_stops_every_cell(harness):
    """And if the bytes change between the launcher's digest and the evaluator's, the
    evaluator itself refuses: no cell may be scored from a checkpoint nobody validated."""
    proc = harness.run(STUB_EVAL_SWAP_CKPT=1)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert harness.summary()["cells_final"] == 0


def test_k2_two_launchers_with_different_records_dirs_share_one_lock(harness, tmp_path):
    """Round-2 finding 2: the lock used to live under REC, so two launchers writing the
    same NAS run directories from different records dirs both got in."""
    gate = tmp_path / "let-training-finish"
    (tmp_path / "rec2").mkdir()
    first = harness.spawn(STUB_TRAIN_WAIT=str(gate))
    lock = tmp_path / "nas" / f".lock_f{TAG}"
    try:
        deadline = time.time() + 60
        while not (lock / "pid").exists() and time.time() < deadline:
            assert first.poll() is None, "the first launcher exited before taking the lock"
            time.sleep(0.05)
        assert (lock / "pid").exists(), "no lock was taken under NAS_ROOT"

        second = harness.run(REC=str(tmp_path / "rec2"))
        assert second.returncode == 3, second.stdout + second.stderr
        blob = second.stdout + second.stderr
        assert str(first.pid) in blob and "already holds" in blob, blob
    finally:
        gate.write_text("go\n")
        out = first.communicate(timeout=180)[0]
    assert first.returncode == 0, out
    assert not lock.exists(), "the holder's EXIT trap must release its own lock"
