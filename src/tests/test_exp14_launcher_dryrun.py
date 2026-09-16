"""The exp_14 launcher's dry run, checked against the helper it must agree with.

``scripts/exp14_launch.sh`` (kit repo ``cylindrical-dinov3``) is the only thing that will
ever type these command lines, and it runs unattended for ~4.5 GPU-days per pair. Its
``DRY_RUN=1`` mode prints every command it would run and executes none of them, so this
test can assert -- without a GPU, without the NAS and without launching anything -- that:

* the two training commands are exactly ``names.train_argv`` (including ``--ckpt-path``
  on a resume, and only then),
* the twenty evaluation commands are exactly ``names.eval_argv``, in the planned
  interleaving (K=1 on the first GPU, K=8 on the second, seeds ascending, arms alternating),
* no run id or eval name carries the ``exp14_`` substring (finding r3-3),
* the at-launch record and the pid line are written, and
* **any** non-zero exit of the verifier, of ``make-contract`` or of ``validate`` stops the
  pipeline with exit 3 before anything downstream happens (the D1 review's addendum),
  simulated through the launcher's ``DRY_RUN_FAIL`` hook rather than by running the real
  CLIs against real data.

``FLAC_WT`` is this worktree (the launcher hashes the real dataset config and split), while
the kit, the NAS root and the records dir are ``tmp_path``, so nothing is written into the
repository.
"""
import json
import os
import subprocess

import pytest

from src.tools.data_curve import names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.environ.get("EXP14_LAUNCHER") or os.path.join(
    os.path.dirname(REPO_ROOT), "cylindrical-dinov3", "worklog", "worklog_yixun",
    "exp_14_data_curve_claude", "scripts", "exp14_launch.sh")
TAG = "025"


def _layout(tmp_path):
    """A kit / NAS / records / package layout the dry run can be pointed at."""
    kit = tmp_path / "kit" / "configs"
    kit.mkdir(parents=True)
    for arm in names.ARMS:
        (kit / names.ARM_MODEL_CONFIG_BASENAMES[arm]).write_text('{"model": {}}\n')
    for name in ("nas", "rec", "pkg/src"):
        (tmp_path / name).mkdir(parents=True)
    return {
        "FLAC_WT": REPO_ROOT,
        "CYL_SRC": str(tmp_path / "pkg" / "src"),
        "KIT": str(tmp_path / "kit"),
        "TAG": TAG,
        "NAS_ROOT": str(tmp_path / "nas"),
        "REC": str(tmp_path / "rec"),
        "GPUS": "0,1",
        "EXPECT_FLAC_SHA": "a" * 40,
        "EXPECT_PKG_SHA": "b" * 40,
        "DRY_RUN": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _run(env):
    if not os.path.exists(SCRIPT):
        pytest.skip(f"launcher not available at {SCRIPT}")
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=full)


def _tagged(stdout, kind):
    """``[dc f025] <kind> <label> | <command> | <ts>`` -> ``[(label, command), …]``."""
    out = []
    for line in stdout.splitlines():
        body = line.split("] ", 1)[-1]
        parts = body.split(" | ")
        if len(parts) < 3 or not parts[0].startswith(kind + " "):
            continue
        out.append((parts[0][len(kind) + 1:], parts[1]))
    return out


def _cfg(env, arm):
    return os.path.join(env["KIT"], "configs", names.ARM_MODEL_CONFIG_BASENAMES[arm])


def _run_dir(env, arm):
    return os.path.join(env["NAS_ROOT"], names.run_id(arm, TAG))


def test_the_launcher_is_syntactically_valid_bash():
    if not os.path.exists(SCRIPT):
        pytest.skip(f"launcher not available at {SCRIPT}")
    assert subprocess.run(["bash", "-n", SCRIPT], capture_output=True).returncode == 0


def test_dry_run_prints_exactly_the_helper_s_training_commands(tmp_path):
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    printed = dict(_tagged(proc.stdout, "ARGV"))
    for arm in names.ARMS:
        run = names.run_id(arm, TAG)
        expected = names.train_argv(
            arm, TAG, _cfg(env, arm), names.TRAIN_DATASET_CONFIGS[TAG],
            _run_dir(env, arm), os.path.join(_run_dir(env, arm), "run_contract.json"))
        assert printed[f"train:{run}"].split() == expected
        assert "--ckpt-path" not in printed[f"train:{run}"]


def test_dry_run_prints_the_twenty_eval_cells_in_the_planned_interleaving(tmp_path):
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    printed = [(label, cmd) for label, cmd in _tagged(proc.stdout, "ARGV")
               if label.startswith("eval:")]
    expected = []
    for K in (1, 8):                       # K=1 lane first, then the K=8 lane
        for seed in names.SEEDS:           # seeds ascending
            for arm in names.ARMS:         # arms interleaved inside a lane
                ckpt = os.path.join(_run_dir(env, arm), "dryrun_step=40000.ckpt")
                expected.append((f"eval:{names.eval_name(arm, TAG, K, seed)}",
                                 names.eval_argv(arm, TAG, K, seed, _cfg(env, arm), ckpt)))
    assert [label for label, _ in printed] == [label for label, _ in expected]
    assert [cmd.split() for _, cmd in printed] == [argv for _, argv in expected]


def test_dry_run_names_never_contain_the_forbidden_substring(tmp_path):
    proc = _run(_layout(tmp_path))
    for label, cmd in _tagged(proc.stdout, "ARGV"):
        tokens = cmd.split()
        for flag in ("--name", "--experiment-name", "--eval-name"):
            if flag in tokens:
                assert names.FORBIDDEN_SUBSTRING not in tokens[tokens.index(flag) + 1]
        assert names.FORBIDDEN_SUBSTRING not in label


def test_dry_run_records_the_launch_and_writes_a_summary(tmp_path):
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    records = dict(_tagged(proc.stdout, "RECORD"))
    pids = dict(_tagged(proc.stdout, "PID"))
    for arm in names.ARMS:
        run = names.run_id(arm, TAG)
        path = os.path.join(env["REC"], f"at_launch_{run}.txt")
        assert records[run] == path
        assert pids[run] == "DRYRUN"
        record = open(path).read()
        for needle in (f"run: {run}", "host: ", f"flac_sha: {'a' * 40}",
                       f"package_sha: {'b' * 40}", "argv: python train.py",
                       f"split: {names.SPLIT_FILES[TAG]}", "pid: DRYRUN"):
            assert needle in record, record

    summary = json.load(open(os.path.join(env["REC"], "data_curve_launch_summary.json")))
    assert summary["tag"] == TAG and summary["fraction"] == 0.25
    assert summary["dry_run"] is True and summary["status"] == "complete"
    assert summary["runs"]["cyl"]["run_id"] == "dc_cyl_f025"
    assert summary["failed_cells"] == []


@pytest.mark.parametrize("step,rc,after", [
    ("verify", 3, "contract:"),                      # nothing may be created
    ("contract:dc_cyl_f025", 2, "train:"),           # no training may start
    ("validate:dc_cyl_f025", 3, "eval:"),            # no cell may be evaluated
])
def test_any_non_zero_exit_of_a_gate_is_fatal(tmp_path, step, rc, after):
    """The D1 addendum: make-contract exits 2 on an identity mismatch, validate 3 on a
    checkpoint violation, and the launcher must treat **both** as fatal."""
    env = _layout(tmp_path)
    env.update(DRY_RUN_FAIL=step, DRY_RUN_FAIL_RC=str(rc))
    proc = _run(env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "STOP:" in proc.stdout
    assert not [label for label, _ in _tagged(proc.stdout, "ARGV") if after in label]
    assert not [label for label, _ in _tagged(proc.stdout, "CMD") if after in label]


def test_a_half_finished_run_is_never_resumed_without_RESUME(tmp_path):
    env = _layout(tmp_path)
    run_dir = _run_dir(env, "cyl")
    os.makedirs(run_dir)
    ckpt = os.path.join(run_dir, "epoch=0-step=2500.ckpt")
    open(ckpt, "wb").write(b"not a real checkpoint")

    refused = _run(env)
    assert refused.returncode == 3
    assert "RESUME=1" in refused.stdout
    assert not _tagged(refused.stdout, "ARGV")

    env["RESUME"] = "1"
    resumed = _run(env)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    printed = dict(_tagged(resumed.stdout, "ARGV"))
    assert printed["train:dc_cyl_f025"].split() == names.train_argv(
        "cyl", TAG, _cfg(env, "cyl"), names.TRAIN_DATASET_CONFIGS[TAG], run_dir,
        os.path.join(run_dir, "run_contract.json"), ckpt_path=ckpt)
    # the resume is validated with the resume-only state check, at the step it claims
    validate = dict(_tagged(resumed.stdout, "CMD"))["validate-resume:dc_cyl_f025"]
    assert "--for-resume" in validate and "--expect-step 2500" in validate
    # ... and the van arm, which has no checkpoints, still starts fresh
    assert "--ckpt-path" not in printed["train:dc_van_f025"]
