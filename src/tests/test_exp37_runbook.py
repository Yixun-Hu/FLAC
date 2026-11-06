"""exp_37 fix round (Codex BLOCKING 5): the executable runbook ``exp37_runbook.sh``.

``list`` enumerates every step in execution order with its exact command (unique ids; every registered spec run has
exactly one producing step; seed-42 B2 passes and the latency passes timed, seeds 43 / 44 and B3 accuracy-only with
inline dumps, every replay naming its timing pass; every model step through the no-flash launcher with SDPA
required); ``run`` refuses a dirty tracked tree or an existing output (exit 2), logs HEAD + the command, and returns
the command's own exit status, stopping at the first failure; B3 synthesis is an explicit external placeholder.
CPU; a scratch git repository stands in for the checkout. Written by the exp_37 Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tools import exp37_build_b2b3_spec as builder

REPO = Path(__file__).resolve().parents[2]
D = "worklog/worklog_yixun/exp_37_loc_efficiency_claude"
SCRIPT = REPO / D / "exp37_runbook.sh"


def _bash(*args, env=None, cwd=REPO):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=120, cwd=cwd,
                          env={**os.environ, **(env or {})})


@pytest.fixture(scope="module")
def steps():
    out = _bash("list")
    assert out.returncode == 0, out.stderr
    rows = {}
    for line in out.stdout.splitlines():
        sid, phase, *command = line.split(None, 2)
        rows[sid] = (phase, shlex.split(command[0]) if command and sid != "B3S" else command)
    return rows, out.stdout


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_every_step_is_unique_and_every_registered_run_has_one_producer(steps):
    rows, text = steps
    assert len(rows) == len(text.splitlines())                                     # unique ids
    outputs = [_flag(a, "--output-dir") for _p, a in rows.values() if _flag(a, "--output-dir")]
    assert len(outputs) == len(set(outputs))
    stems = {Path(o).name for o in outputs}
    inputs = json.loads((REPO / D / "b2b3" / "b2b3_inputs.json").read_text())
    spec = builder.build_spec(inputs, Path(D) / "runs", check_exists=False, require_timing_replays=True)
    wanted = {Path(i[k]).name for i in spec["b2_runs"] + spec["b2_timing"] + spec["b3_latency"]["runs"]
              for k in ("dir", "replay") if i.get(k)} | {Path(i["dir"]).name for i in spec["b3_runs"]}
    assert wanted <= stems, sorted(wanted - stems)


def test_timing_accuracy_and_replay_steps_carry_their_registered_flags(steps):
    rows, _ = steps
    for sid, (_phase, argv) in rows.items():
        out = _flag(argv, "--output-dir")
        if out is None:
            continue
        stem = Path(out).name
        assert "overlays/no_flash/exp37_no_flash_launch.py" in argv and _flag(argv, "--require-attention-backend") == "sdpa"
        timed, dumped, replay = "--measure-core-forward" in argv, _flag(argv, "--dump-waveforms"), _flag(argv, "--replay-of")
        assert timed != bool(dumped), sid                                           # timing passes never dump
        if dumped:
            assert Path(dumped).name == stem and Path(dumped).parent == Path("/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms")
        if replay:
            assert stem == f"{Path(replay).name}_replay"
        if stem.startswith("b2_") and not stem.endswith("_replay"):
            assert timed is stem.endswith("_s42"), sid                             # seed 42 timed; 43 / 44 accuracy
        if stem.startswith("b3_") and "_lat_" not in stem:
            assert dumped and not replay, sid
            assert ("--rotate-deg" in argv) is ("_t90_" in stem)


def test_the_synthesis_step_is_an_explicit_external_placeholder(steps):
    _rows, text = steps
    assert text.splitlines()[-1].startswith("B3S") and "PENDING" in text.splitlines()[-1]
    refused = _bash("run", "B3S")
    assert refused.returncode == 2 and "external" in refused.stderr


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "tracked.txt").write_text("x\n")
    for argv in (["git", "init", "-q"], ["git", "add", "tracked.txt"],
                 ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"]):
        subprocess.run(argv, cwd=repo, check=True)
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\nexit 7\n")
    fake.chmod(0o755)
    return repo, fake


def test_a_dry_run_logs_head_and_the_exact_command(checkout):
    repo, _fake = checkout
    done = _bash("run", "G0.YAW.run", env={"REPO_DIR": str(repo), "DRY_RUN": "1"})
    assert done.returncode == 0, done.stderr
    logs = sorted((repo / D).glob("loc_efficiency_*_G0.YAW.run.log"))
    text = logs[0].read_text()
    assert text.startswith("HEAD ") and "COMMAND env CUDA_VISIBLE_DEVICES=0" in text and "DRY_RUN: not executed" in text


def test_a_dirty_tree_or_an_existing_output_is_refused(checkout):
    repo, _fake = checkout
    (repo / "tracked.txt").write_text("changed\n")
    assert _bash("run", "SEL.0", env={"REPO_DIR": str(repo), "DRY_RUN": "1"}).returncode == 2
    subprocess.run(["git", "checkout", "-q", "tracked.txt"], cwd=repo, check=True)
    (repo / D / "runs" / "g0_YAW").mkdir(parents=True)
    done = _bash("run", "G0.YAW.run", env={"REPO_DIR": str(repo), "DRY_RUN": "1"})
    assert done.returncode == 2 and "exists" in done.stderr


def test_the_command_s_exit_status_is_returned_and_stops_the_sequence(checkout):
    repo, fake = checkout
    done = _bash("run", "SEL.0", "B1.0", env={"REPO_DIR": str(repo), "PY": str(fake)})
    assert done.returncode == 7 and "STOPPED at SEL.0 (exit 7)" in done.stderr
    assert "EXIT 7" in sorted((repo / D).glob("loc_efficiency_*_SEL.0.log"))[0].read_text()
    assert not list((repo / D).glob("loc_efficiency_*_B1.0.log"))
    assert _bash("phase", "NOPE", env={"REPO_DIR": str(repo)}).returncode == 2
