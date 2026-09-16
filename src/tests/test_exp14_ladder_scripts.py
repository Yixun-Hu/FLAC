"""The exp_14 ladder scripts, exercised on stubs: a rung may not mark itself done.

Codex round-F finding 4: ``rung5_smoke.sh`` / ``rung5b_probe.sh`` / ``rung6_fit_probe.sh``
/ ``rung7_nas_ckpt.sh`` / ``rung5b67_chain.sh`` sequenced commands without ``set -e`` and
wrote their ``.done`` marker unconditionally, so a rung that had failed every one of its
acceptance conditions still published a completion marker -- and the chain published
``CHAIN_DONE`` on top of it. The recorded evidence of the real (GPU) ladder run stands; what
is tested here is the *bookkeeping*, on a stub ``python`` and a stub conda profile, with no
GPU, no NAS and no network.

Each script now takes its paths from the environment (the production defaults are
unchanged), which is what lets this harness point one at ``tmp_path``.
"""
import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LADDER = os.path.join(REPO_ROOT, "worklog", "worklog_yixun", "exp_14_data_curve")
SCRIPTS = ("rung5_smoke.sh", "rung5b_probe.sh", "rung6_fit_probe.sh", "rung7_nas_ckpt.sh",
           "rung5b67_chain.sh")

STUB_PYTHON = r'''#!/bin/bash
# Ladder-test stub for `python`: the two entry points the rungs call, and nothing else.
set -u
M="$STUB_MARKERS"; mkdir -p "$M"
echo "$*" >> "$M/calls.log"
arg () { local want="$1"; shift; while [ $# -gt 0 ]; do
  if [ "$1" = "$want" ]; then echo "${2:-}"; return 0; fi; shift; done; echo ""; }
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "src.training.run_contract" ]; then
  case "${3:-}" in
    make-contract)
      RD="$(arg --run-dir "$@")"; mkdir -p "$RD"
      printf '{"run_id": "smoke7_cyl"}\n' > "$RD/run_contract.json"
      echo "$RD/run_contract.json"; exit "${STUB_RC_CONTRACT:-0}" ;;
    validate)
      echo "validate rc=${STUB_RC_VALIDATE:-0}"; exit "${STUB_RC_VALIDATE:-0}" ;;
  esac
fi
if [ "${1:-}" = "train.py" ]; then
  NAME="$(arg --name "$@")"; SD="$(arg --save-dir "$@")"
  echo "$NAME" >> "$M/trained"
  O="STUB_OUT_$NAME"; echo "${!O:-${STUB_TRAIN_STDOUT:-max_steps=3 reached}}"
  C="STUB_CKPT_$NAME"
  if [ "${!C:-${STUB_TRAIN_CKPT:-0}}" = 1 ]; then
    mkdir -p "$SD"; printf 'checkpoint bytes' > "$SD/epoch=0-step=5.ckpt"; fi
  V="STUB_RC_$NAME"; exit "${!V:-${STUB_RC_TRAIN:-0}}"
fi
exit 0
'''

STUB_SAMPLER = '#!/bin/bash\nwhile kill -0 "$2" 2>/dev/null; do sleep 0.2; done\n'
STUB_CONDA = 'conda () { return 0; }\n'


class Ladder:
    """A scratch worktree + records dir holding copies of the five ladder scripts."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.markers = tmp_path / "markers"
        self.markers.mkdir()
        binary = tmp_path / "bin"
        binary.mkdir()
        (binary / "python").write_text(STUB_PYTHON)
        (binary / "python").chmod(0o755)
        self.bin = str(binary)

        self.wt = tmp_path / "wt"
        self.rec = self.wt / "worklog" / "worklog_yixun" / "exp_14_data_curve"
        self.rec.mkdir(parents=True)
        for name in SCRIPTS:
            shutil.copy(os.path.join(LADDER, name), self.rec / name)
        (self.rec / "gpu_sampler.sh").write_text(STUB_SAMPLER)
        (tmp_path / "conda.sh").write_text(STUB_CONDA)
        kit = tmp_path / "kit" / "configs"
        kit.mkdir(parents=True)
        for arm in ("cyl", "van"):
            (kit / f"FLAC_AR_exp14_{arm}S.json").write_text("{}\n")
        self.kit = str(tmp_path / "kit")
        self.nas = tmp_path / "nas"
        self.nas.mkdir()
        env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
        (self.wt / "README").write_text("scratch\n")
        for args in (["init", "-q"], ["add", "-A"],
                     ["-c", "user.email=t@t", "-c", "user.name=T", "commit", "-qm", "wt"]):
            done = subprocess.run(["git", "-C", str(self.wt), *args], capture_output=True,
                                  text=True, env=env)
            assert done.returncode == 0, done.stderr

    def run(self, script, **overrides):
        env = dict(os.environ)
        env.update({
            "PATH": self.bin + os.pathsep + env["PATH"],
            "STUB_MARKERS": str(self.markers),
            "WT": str(self.wt), "REC": str(self.rec), "KIT": self.kit,
            "NAS": str(self.nas), "RUN": str(self.nas / "smoke7_cyl"),
            "CONDA_SH": str(self.tmp / "conda.sh"),
            "PKG_SRC": str(self.tmp / "pkg" / "src"), "PKG_DIR": str(self.wt),
            "STABLE_WAIT": "0", "SETTLE": "0",
            "CUDA_VISIBLE_DEVICES": "",
        })
        env.update({k: str(v) for k, v in overrides.items()})
        return subprocess.run(["bash", str(self.rec / script)], capture_output=True,
                              text=True, env=env)

    def markers_named(self, suffix):
        return sorted(p.name for p in self.rec.glob(f"*{suffix}"))


@pytest.fixture
def ladder(tmp_path):
    if not shutil.which("bash") or not shutil.which("git"):
        pytest.skip("bash and git are required")
    return Ladder(tmp_path)


def test_every_ladder_script_is_syntactically_valid_bash():
    for name in SCRIPTS:
        path = os.path.join(LADDER, name)
        assert subprocess.run(["bash", "-n", path], capture_output=True).returncode == 0, name


# ----------------------------------------------------------------------------- rung 5
def test_rung5_marks_done_only_when_both_arms_pass(ladder):
    proc = ladder.run("rung5_smoke.sh")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ladder.markers_named(".done") and not ladder.markers_named(".failed")
    assert open(ladder.markers / "trained").read().split() == ["smoke5_dc_cyl", "smoke5_dc_van"]


@pytest.mark.parametrize("override,needle", [
    ({"STUB_RC_smoke5_dc_van": 1}, "van"),                  # the second arm crashed
    ({"STUB_TRAIN_STDOUT": "Epoch 0: 0%"}, "banner"),       # rc 0 but never reached max_steps
])
def test_rung5_writes_failed_and_no_done_when_an_arm_misses_its_acceptance(
        ladder, override, needle):
    proc = ladder.run("rung5_smoke.sh", **override)
    assert proc.returncode != 0, proc.stdout
    assert not ladder.markers_named(".done")
    failed = ladder.markers_named(".failed")
    assert len(failed) == 1
    assert needle in (ladder.rec / failed[0]).read_text()


# ---------------------------------------------------------------------------- rung 5b
def _contract_violation(**extra):
    env = {"STUB_RC_TRAIN": 1,
           "STUB_TRAIN_STDOUT": "src.data.dataset.DatasetContractError: no in-split context"}
    env.update(extra)
    return env


def test_rung5b_marks_done_on_a_clean_contract_violation(ladder):
    proc = ladder.run("rung5b_probe.sh", **_contract_violation())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ladder.markers_named(".done") and not ladder.markers_named(".failed")


@pytest.mark.parametrize("override,needle", [
    ({"STUB_RC_TRAIN": 0}, "exited 0"),                     # the violation did not terminate
    (_contract_violation(STUB_TRAIN_STDOUT="loss=1.0"), "DatasetContractError"),
    (_contract_violation(STUB_TRAIN_CKPT=1), "checkpoint"),
])
def test_rung5b_writes_failed_when_any_of_its_four_conditions_breaks(ladder, override, needle):
    proc = ladder.run("rung5b_probe.sh", **override)
    assert proc.returncode != 0, proc.stdout
    assert not ladder.markers_named(".done")
    failed = ladder.markers_named(".failed")
    assert len(failed) == 1 and needle in (ladder.rec / failed[0]).read_text()


# ------------------------------------------------------------------------ rungs 6 and 7
def test_rung6_requires_both_arms_to_exit_zero(ladder):
    assert ladder.run("rung6_fit_probe.sh").returncode == 0
    assert ladder.markers_named(".done") and not ladder.markers_named(".failed")

    again = ladder.run("rung6_fit_probe.sh", STUB_RC_smoke6_dc_van=4)
    assert again.returncode != 0, again.stdout
    failed = ladder.markers_named(".failed")
    assert len(failed) == 1 and "van(rc=4)" in (ladder.rec / failed[0]).read_text()


def test_rung7_requires_a_stable_checkpoint_and_a_clean_validate(ladder):
    assert ladder.run("rung7_nas_ckpt.sh", STUB_TRAIN_CKPT=1).returncode == 0
    assert ladder.markers_named(".done") and not ladder.markers_named(".failed")

    bad = ladder.run("rung7_nas_ckpt.sh", STUB_TRAIN_CKPT=1, STUB_RC_VALIDATE=3)
    assert bad.returncode != 0, bad.stdout
    assert any("validate exited 3" in (ladder.rec / f).read_text()
               for f in ladder.markers_named(".failed"))


def test_rung7_fails_when_training_wrote_no_checkpoint(ladder):
    proc = ladder.run("rung7_nas_ckpt.sh", STUB_TRAIN_CKPT=0)
    assert proc.returncode != 0, proc.stdout
    assert not ladder.markers_named(".done")
    assert any("no step-5 checkpoint" in (ladder.rec / f).read_text()
               for f in ladder.markers_named(".failed"))


# ------------------------------------------------------------------------------- chain
#: The one stub configuration in which all three chained rungs meet their acceptance:
#: 5b must terminate on a contract violation without a checkpoint, 6 and 7 must succeed,
#: and only 7 may leave a checkpoint behind.
CHAIN_HAPPY = {
    "STUB_RC_smoke5b_probe": 1,
    "STUB_OUT_smoke5b_probe": "src.data.dataset.DatasetContractError: no in-split context",
    "STUB_CKPT_smoke7_cyl": 1,
}


def test_the_chain_stops_at_the_first_rung_that_misses_its_acceptance(ladder):
    """The old chain ran the three rungs with `;` and then wrote CHAIN_DONE regardless."""
    proc = ladder.run("rung5b67_chain.sh")        # rung 5b's stub train exits 0: not a probe
    assert proc.returncode != 0, proc.stdout
    assert not [m for m in ladder.markers_named(".done") if m.startswith("rung5b67")]
    assert [m for m in ladder.markers_named(".failed") if m.startswith("rung5b67")]
    trained = open(ladder.markers / "trained").read().split()
    assert trained == ["smoke5b_probe"], trained   # rungs 6 and 7 never ran


def test_the_chain_marks_done_only_after_all_three_rungs(ladder):
    proc = ladder.run("rung5b67_chain.sh", **CHAIN_HAPPY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert [m for m in ladder.markers_named(".done") if m.startswith("rung5b67")]
    assert not ladder.markers_named(".failed")
    trained = open(ladder.markers / "trained").read().split()
    assert trained[0] == "smoke5b_probe" and trained[-1] == "smoke7_cyl"
    assert set(trained[1:3]) == {"smoke6_dc_cyl", "smoke6_dc_van"}   # rung 6 is concurrent


def test_the_chain_fails_when_a_later_rung_fails(ladder):
    proc = ladder.run("rung5b67_chain.sh", **dict(CHAIN_HAPPY, STUB_RC_smoke7_cyl=2))
    assert proc.returncode != 0, proc.stdout
    assert not [m for m in ladder.markers_named(".done") if m.startswith("rung5b67")]
    chain = [m for m in ladder.markers_named(".failed") if m.startswith("rung5b67")]
    assert len(chain) == 1 and "rung7_nas_ckpt.sh" in (ladder.rec / chain[0]).read_text()
