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
  CLIs against real data,
* the round-F gates are printed where they will run: the verifier pins the kit and the
  launcher's own bytes, the twenty planned cells are re-enumerated after the lanes, and
  every marker, record and summary this invocation writes carries its unique launch id.

``FLAC_WT`` is this worktree (the launcher hashes the real dataset config and split), while
the kit, the NAS root and the records dir are ``tmp_path``, so nothing is written into the
repository.
"""
import hashlib
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
        "EXPECT_KIT_SHA": "c" * 40,
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


def _launcher_sha256():
    return hashlib.sha256(open(SCRIPT, "rb").read()).hexdigest()


def _one_summary(env):
    """The invocation's own summary, reached through the `latest` pointer."""
    pointer = os.path.join(env["REC"], "data_curve_launch_summary.json")
    assert os.path.islink(pointer), "the stable name must be a pointer, never the record"
    return json.load(open(pointer))


def test_the_launcher_is_syntactically_valid_bash():
    if not os.path.exists(SCRIPT):
        pytest.skip(f"launcher not available at {SCRIPT}")
    assert subprocess.run(["bash", "-n", SCRIPT], capture_output=True).returncode == 0


def test_dry_run_prints_exactly_the_helper_s_training_commands(tmp_path):
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stderr == "", proc.stderr   # a dry run touches nothing and warns about nothing

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
                                 names.eval_argv(arm, TAG, K, seed, _cfg(env, arm), ckpt,
                                                 names.DRYRUN_SHA256)))
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
        path = records[run]
        # N7/round-F 5: the record is named for THIS invocation, so a second launcher
        # cannot overwrite the first one's record of what it started.
        assert os.path.basename(path).startswith(f"at_launch_{run}_"), path
        assert pids[run] == "DRYRUN"
        record = open(path).read()
        for needle in (f"run: {run}", "host: ", f"flac_sha: {'a' * 40}",
                       f"package_sha: {'b' * 40}", "argv: python train.py",
                       f"split: {names.SPLIT_FILES[TAG]}", "pid: DRYRUN",
                       "final_ckpt_sha256: "):
            assert needle in record, record

    summary = _one_summary(env)
    assert summary["tag"] == TAG and summary["fraction"] == 0.25
    assert summary["dry_run"] is True and summary["status"] == "complete"
    assert summary["runs"]["cyl"]["run_id"] == "dc_cyl_f025"
    assert summary["failed_cells"] == []
    assert summary["launch_ts"] and summary["launcher_sha256"] == _launcher_sha256()
    assert summary["kit_sha"] == "c" * 40
    assert summary["cells_final"] == 20


def test_dry_run_verifies_the_kit_and_its_own_bytes_before_anything_else(tmp_path):
    """Round-F finding 3: the kit holds the launcher and both arm configs, and the running
    launcher proves it IS the kit's launcher by passing its own sha256."""
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    verify = dict(_tagged(proc.stdout, "CMD"))["verify"]
    assert f"--expect-kit-sha {'c' * 40}" in verify
    assert f"--expect-launcher-sha256 {_launcher_sha256()}" in verify
    assert f"--kit-dir {env['KIT']}" in verify


def test_the_launcher_refuses_to_run_without_the_kit_sha(tmp_path):
    env = _layout(tmp_path)
    del env["EXPECT_KIT_SHA"]
    full = dict(os.environ)
    full.pop("EXPECT_KIT_SHA", None)
    full.update(env)
    proc = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=full)
    assert proc.returncode != 0
    assert "EXPECT_KIT_SHA" in proc.stderr


def test_dry_run_enumerates_the_twenty_planned_cells_after_the_lanes(tmp_path):
    """Round-F finding 1: `wait` without job ids let the launcher write COMPLETE with fewer
    than 20 evaluated cells. The verdict is now a re-check of every planned cell."""
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    checked = [label[len("finalcheck:"):] for label, _ in _tagged(proc.stdout, "CMD")
               if label.startswith("finalcheck:")]
    planned = [names.eval_name(arm, TAG, k, seed)
               for k in names.K_VALUES for seed in names.SEEDS for arm in names.ARMS]
    # metrics JSON and bundle, for each of the twenty planned cells
    assert sorted(set(checked)) == sorted(planned)
    assert len(checked) == 40
    commands = [cmd for label, cmd in _tagged(proc.stdout, "CMD")
                if label.startswith("finalcheck:")]
    assert sum("check-metrics" in c for c in commands) == 20
    assert sum("check-bundle" in c for c in commands) == 20
    # every cell check is bound to its arm's validated final checkpoint (round-F 2)
    for arm in names.ARMS:
        ckpt = os.path.join(_run_dir(env, arm), "dryrun_step=40000.ckpt")
        assert sum(f"--expect-ckpt {ckpt} " in c + " " for c in commands) == 20
    # ... and to the digest of those bytes (codex full-r2 1): BOTH gates, all 40 commands
    assert sum("--expect-ckpt-sha256 " in c for c in commands) == 40


def test_two_invocations_never_overwrite_each_other_s_summary(tmp_path):
    """Finding N7 / round-F 5: `>` to a fixed name, and a same-second timestamp, lost one
    invocation's record. Names now carry the timestamp AND the pid."""
    env = _layout(tmp_path)
    assert _run(env).returncode == 0
    assert _run(env).returncode == 0
    summaries = [p for p in os.listdir(env["REC"])
                 if p.startswith("data_curve_launch_summary_f")]
    assert len(summaries) == 2, summaries
    assert os.path.islink(os.path.join(env["REC"], "data_curve_launch_summary.json"))
    # a record on a shared box is readable by the other session, as `>` used to leave it
    for name in summaries:
        mode = os.stat(os.path.join(env["REC"], name)).st_mode & 0o777
        assert mode & 0o044 == 0o044, (name, oct(mode))


def test_a_second_launcher_for_the_same_tag_is_refused(tmp_path):
    """Round-F finding 5: an atomic per-tag lock, with a stale-pid check that never
    deletes anything by itself."""
    env = _layout(tmp_path)
    lock = os.path.join(env["NAS_ROOT"], f".lock_f{TAG}")
    os.makedirs(lock)
    with open(os.path.join(lock, "pid"), "w") as fout:
        fout.write(f"{os.getpid()} held-by-this-test\n")
    held = _run(env)
    assert held.returncode == 3, held.stdout + held.stderr
    assert str(os.getpid()) in held.stdout + held.stderr
    assert not _tagged(held.stdout, "CMD")          # nothing was even printed

    with open(os.path.join(lock, "pid"), "w") as fout:
        fout.write("4194303 a launcher that was killed\n")   # > /proc/sys/kernel/pid_max
    stale = _run(env)
    assert stale.returncode == 3
    assert "dead" in (stale.stdout + stale.stderr)
    assert os.path.isdir(lock)                      # refused, never removed for us


def test_a_completed_invocation_releases_its_lock(tmp_path):
    env = _layout(tmp_path)
    assert _run(env).returncode == 0
    assert not os.path.exists(os.path.join(env["NAS_ROOT"], f".lock_f{TAG}"))


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


# ------------------------------------------- the checkpoint digest (codex full-r2 1)
def test_dry_run_pins_the_checkpoint_digest_on_every_eval(tmp_path):
    """Every eval argv carries --expect-ckpt-sha256, so the evaluator re-hashes the
    checkpoint before loading it and refuses bytes replaced at the same pathname."""
    env = _layout(tmp_path)
    proc = _run(env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    evals = [cmd for label, cmd in _tagged(proc.stdout, "ARGV") if label.startswith("eval:")]
    assert len(evals) == 20
    for cmd in evals:
        tokens = cmd.split()
        assert tokens.count("--expect-ckpt-sha256") == 1, cmd
        # a dry run has no checkpoint to hash, so the one explicit sentinel is printed
        assert tokens[tokens.index("--expect-ckpt-sha256") + 1] == names.DRYRUN_SHA256
    # the digest is announced once per arm, next to the checkpoint it belongs to
    announced = [line for line in proc.stdout.splitlines() if " SHA256 " in line]
    assert len(announced) == 2, announced


def test_the_launcher_refuses_a_run_target_it_cannot_lock(tmp_path):
    """Round-2 finding 2: the lock is global per run target, so it must live under
    NAS_ROOT -- and a NAS_ROOT that cannot be created or written is fatal (exit 3),
    never a silently un-locked run."""
    env = _layout(tmp_path)
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    env["NAS_ROOT"] = str(parent / "nas")
    try:
        proc = _run(env)
    finally:
        parent.chmod(0o700)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "NAS_ROOT" in proc.stdout + proc.stderr
    assert not _tagged(proc.stdout, "CMD")          # nothing ran, nothing was verified


def test_the_lock_is_the_run_target_not_the_records_dir(tmp_path):
    """Two launchers with different REC directories write the same NAS run dirs, so a
    lock under REC lets both in. A stale lock under the records dir must not block."""
    env = _layout(tmp_path)
    (tmp_path / "rec2").mkdir()
    rec_lock = os.path.join(env["REC"], f".lock_f{TAG}")
    os.makedirs(rec_lock)
    with open(os.path.join(rec_lock, "pid"), "w") as fout:
        fout.write(f"{os.getpid()} an old per-REC lock\n")

    assert _run(env).returncode == 0                # the per-REC lock is not the lock
    nas_lock = os.path.join(env["NAS_ROOT"], f".lock_f{TAG}")
    os.makedirs(nas_lock)
    with open(os.path.join(nas_lock, "pid"), "w") as fout:
        fout.write(f"{os.getpid()} held-by-this-test\n")
    env["REC"] = str(tmp_path / "rec2")             # a different records dir ...
    blocked = _run(env)
    assert blocked.returncode == 3, blocked.stdout  # ... is still the same run target
    assert str(os.getpid()) in blocked.stdout + blocked.stderr
