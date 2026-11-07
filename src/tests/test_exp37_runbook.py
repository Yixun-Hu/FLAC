"""exp_37 fix round 2 (Codex code review 2, BLOCKING 4 / 5, HIGH 8, NIT 1): the executable runbook ``exp37_runbook.sh``.

``list`` prints the registered sequence (98 steps: P0 / G0 / SEL / B1 / B2 / B3 timed phases, the mandatory TR replays
of every timing pass, the B2A / B3A accuracy runs incl. the seed 43 / 44 c2f runs, the two B1 summaries consuming every
replay with --require-replays, the gated B3S synthesis (host gate, the Planner's leaf chain b3 at its launch commit,
the synthesis section) and the aggregation with --require-timing-replays) with each step's kind and exact command.
``run`` / ``all`` enforce the order through hashed receipts (every earlier step done; every earlier campaign step --
P0, the gates, the model and leaf steps -- at this HEAD), a pre-step exp_36 gate run immediately before every timed /
selector / leaf step and its fresh receipt (bound to the gate record), a tracked tree that is clean and untracked
content only under the allowlist, an exclusive log whose tee status is checked (a logging failure is exit 74), and
the command's own exit status; ``skip`` only for optional steps (the K_ctx 1 splits and the B1 summary variant);
``clean`` quarantines a failed model step; ``check-leaf`` verifies the leaf worktree.
CPU; a scratch git repository stands in for the checkout. Written by the exp_37 Coder (Claude Opus 5.5, max effort).
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools import exp37_build_b2b3_spec as builder

REPO = Path(__file__).resolve().parents[2]
D = "worklog/worklog_yixun/exp_37_loc_efficiency_claude"
SCRIPT = REPO / D / "exp37_runbook.sh"
LEAF_COMMIT = "d30137349027914af7f04462b05347f0f9e03c02"


def _bash(*args, env=None):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=300, cwd=REPO,
                          env={**os.environ, "RPY": sys.executable, **(env or {})})


@pytest.fixture(scope="module")
def steps():
    out = _bash("list")
    assert out.returncode == 0, out.stderr
    rows = {}
    for line in out.stdout.splitlines():
        sid, phase, kind, command = line.split(None, 3)
        rows[sid] = (phase, kind, shlex.split(command))
    return rows, out.stdout


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def test_the_registered_sequence(steps):
    rows, text = steps
    assert len(rows) == len(text.splitlines()) == 98
    phases = [p for p, _k, _c in rows.values()]
    order = ["P0", "G0", "SEL", "B1", "B2", "B3", "TR", "B2A", "B3A", "B1S", "B3S", "AGG"]
    assert [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p] == order
    assert list(rows)[-5:] == ["B3S.gate", "B3S.run", "B3S.cells", "AGG.spec", "AGG.run"]
    assert {sid for sid, (_p, kind, _c) in rows.items() if kind.endswith("?")} == {          # optional steps
        "B1.split_k1.P1", "B1.split_k1.YAW", "B1.split_k1.BF", "TR.split_k1.P1", "TR.split_k1.YAW", "TR.split_k1.BF",
        "B1S.all", "B1S.noK1"}


def test_every_registered_run_and_replay_has_one_producer(steps):
    rows, _ = steps
    outputs = [_flag(c, "--output-dir") for _p, _k, c in rows.values() if _flag(c, "--output-dir")]
    assert len(outputs) == len(set(outputs))
    stems = {Path(o).name for o in outputs}
    inputs = json.loads((REPO / D / "b2b3" / "b2b3_inputs.json").read_text())
    spec = builder.build_spec(inputs, Path(D) / "runs", check_exists=False)
    wanted = {Path(i[k]).name for i in spec["b2_runs"] + spec["b2_timing"] + spec["b3_latency"]["runs"]
              for k in ("dir", "replay") if i.get(k)} | {Path(i["dir"]).name for i in spec["b3_runs"] + spec["b2_c2f_runs"]}
    assert wanted <= stems, sorted(wanted - stems)
    timed = {Path(_flag(c, "--output-dir")).name for _p, k, c in rows.values() if k.rstrip("?") == "timed"}
    replays = {Path(_flag(c, "--replay-of")).name for _p, _k, c in rows.values() if _flag(c, "--replay-of")}
    assert timed == replays                                                  # D-c: every timing pass is replayed


def test_every_model_step_carries_its_registered_flags(steps):
    rows, _ = steps
    for sid, (_phase, kind, argv) in rows.items():
        out = _flag(argv, "--output-dir")
        if out is None:
            continue
        stem = Path(out).name
        assert "overlays/no_flash/exp37_no_flash_launch.py" in argv and _flag(argv, "--require-attention-backend") == "sdpa"
        timed, dumped, replay = "--measure-core-forward" in argv, _flag(argv, "--dump-waveforms"), _flag(argv, "--replay-of")
        assert timed == (kind.rstrip("?") == "timed") and timed != bool(dumped), sid
        if dumped:
            assert Path(dumped) == Path("/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms") / stem
        if replay:
            assert stem == f"{Path(replay).name}_replay"
        if stem.startswith("b2_") and not stem.endswith("_replay"):
            assert timed is stem.endswith("_s42"), sid                       # D-a
        if stem.startswith("b2a_c2f_"):
            assert _flag(argv, "--candidate-subset") == "c2f" and _flag(argv, "--c2f-top-k") == "5" and dumped


def test_the_summaries_consume_every_replay(steps):
    rows, _ = steps
    full, no_k1, spec = rows["B1S.all"][2], rows["B1S.noK1"][2], rows["AGG.spec"][2]
    assert full.count("--replay") == 6 and full.count("--split-replay") == 6 and "--require-replays" in full
    assert no_k1.count("--replay") == 6 and no_k1.count("--split-replay") == 3 and "--require-replays" in no_k1
    assert "--require-timing-replays" in spec and _flag(spec, "--synthesis") == f"{D}/b2b3/synthesis_cells.json"


def test_b3s_runs_the_planners_leaf_chain_at_its_launch_commit(steps):
    rows, _ = steps
    phase, kind, argv = rows["B3S.run"]
    assert (phase, kind) == ("B3S", "leaf")
    assert argv == ["cd", "/home/yixunhu/codespace/exp-37-b3-synthesis", "&&", "env", f"EXPECT_HEAD={LEAF_COMMIT}",
                    "bash", "worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh", "b3"]
    assert rows["B3S.gate"][1] == "gate" and rows["B3S.cells"][2][-2:] == ["--output", f"{D}/b2b3/synthesis_cells.json"]


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    (repo / D).mkdir(parents=True)
    (repo / "tracked.txt").write_text("x\n")
    (repo / D / "exp37_assets.sh").write_text("exit 0\n")
    (repo / "AcousticRooms").mkdir()                                         # the pinned dataset entry
    (repo / ".gitignore").write_text("AcousticRooms\n")
    for argv in (["git", "init", "-q"], ["git", "add", "tracked.txt", ".gitignore", f"{D}/exp37_assets.sh"],
                 ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"]):
        subprocess.run(argv, cwd=repo, check=True)
    ok, seven = tmp_path / "ok", tmp_path / "seven"
    ok.write_text(FAKE_PY)
    seven.write_text("#!/bin/sh\nexit 7\n")
    ok.chmod(0o755)
    seven.chmod(0o755)
    calls = tmp_path / "calls.txt"
    return repo, {"REPO_DIR": str(repo), "PY": str(ok), "TOOLS": str(REPO / "tools"), "CALLS": str(calls)}, str(seven)


# a stand-in interpreter: logs its argv; a gate (--gate-only --output X) writes its record and exits GATE_STATUS
FAKE_PY = """#!/bin/sh
echo "$*" >> "$CALLS"
out=""; gate=0; prev=""
for a in "$@"; do
  [ "$prev" = "--output" ] && out=$a
  [ "$a" = "--gate-only" ] && gate=1
  prev=$a
done
if [ $gate = 1 ]; then
  mkdir -p "$(dirname "$out")"
  if [ "${GATE_STATUS:-0}" = 0 ]; then echo '{"gate": "clean"}' > "$out"; else echo '{"gate": "exp36_present"}' > "$out"; fi
  exit "${GATE_STATUS:-0}"
fi
exit 0
"""


def _receipts(repo):
    return sorted(p.stem for p in (repo / D / "receipts").glob("*.json"))


def test_the_order_is_enforced_by_receipts(checkout):
    repo, env, _seven = checkout
    done = _bash("run", "P0.2", env=env)
    assert done.returncode == 2 and "predecessor" in done.stderr
    assert _bash("run", "P0.1", "P0.2", "P0.3", "P0.4", "G0.YAW.run", env=env).returncode == 0
    assert _receipts(repo) == ["G0.YAW.run", "P0.1", "P0.2", "P0.3", "P0.4"]
    assert "predecessor" in _bash("run", "G0.P1.run", env=env).stderr        # G0.YAW.cmp comes first
    assert _bash("run", "P0.1", env=env).returncode == 2                      # done steps are never re-run


def test_all_runs_the_sequence_and_stops_at_the_first_refusal(checkout):
    repo, env, _seven = checkout
    done = _bash("all", env=env)
    assert done.returncode == 2 and "G0.YAW.cmp" in done.stderr               # its run manifest is missing
    assert _receipts(repo) == ["G0.YAW.run", "P0.1", "P0.2", "P0.3", "P0.4"]


def test_the_command_status_is_returned_and_writes_no_receipt(checkout):
    repo, env, seven = checkout
    assert _bash("run", "P0.1", env=env).returncode == 0
    done = _bash("run", "P0.2", env={**env, "PY": seven})
    assert done.returncode == 7 and "P0.2" not in _receipts(repo)
    assert "EXIT 7" in sorted((repo / D).glob("loc_efficiency_*_P0.2.log"))[0].read_text()


def test_a_dirty_tree_stray_untracked_content_or_an_unwritable_log_refuses(checkout):
    repo, env, _seven = checkout
    (repo / "stray.txt").write_text("?")
    done = _bash("run", "P0.1", env=env)
    assert done.returncode == 2 and "untracked" in done.stderr
    (repo / "stray.txt").unlink()
    (repo / "tracked.txt").write_text("changed\n")
    assert _bash("run", "P0.1", env=env).returncode == 2
    subprocess.run(["git", "checkout", "-q", "tracked.txt"], cwd=repo, check=True)
    (repo / D).chmod(0o555)
    try:
        assert _bash("run", "P0.1", env=env).returncode == 74                # the audit log cannot be created
    finally:
        (repo / D).chmod(0o755)


def test_campaign_steps_share_one_head(checkout):
    repo, env, _seven = checkout
    assert _bash("run", "P0.1", "P0.2", "P0.3", env=env).returncode == 0
    assert _bash("skip", "P0.4", env=env).returncode == 2                    # a gate is never skipped
    receipt = repo / D / "receipts" / "P0.4.json"
    subprocess.run([sys.executable, str(REPO / "tools" / "exp37_receipt.py"), "write", "--path", str(receipt), "--step",
                    "P0.4", "--phase", "P0", "--status", "ok", "--head", "0" * 40, "--command", "x", "--log", "x",
                    "--started", "2026-01-01T00:00:00+00:00", "--finished", "2026-01-01T00:00:00+00:00"], check=True)
    done = _bash("run", "G0.YAW.run", env=env)                                 # the P0 gate ran at another HEAD
    assert done.returncode == 2 and "HEAD" in done.stderr


def test_every_timed_step_is_gated_immediately_before_it_starts(checkout):
    repo, env, _seven = checkout
    calls = Path(env["CALLS"])
    assert _bash("run", "P0.1", "P0.2", "P0.3", "P0.4", env=env).returncode == 0
    done = _bash("run", "G0.YAW.run", env={**env, "GATE_STATUS": "3"})    # exp_36 is on the host
    assert done.returncode == 2 and "gate" in done.stderr and "G0.YAW.run" not in _receipts(repo)
    assert "localize_FLAC.py" not in calls.read_text()                       # the timed pass never started
    assert len(list((repo / D / "gates" / "pre").glob("G0.YAW.run.*.json"))) == 1
    assert _bash("run", "G0.YAW.run", env=env).returncode == 0              # a clean gate on retry
    assert "localize_FLAC.py" in calls.read_text() and "G0.YAW.run" in _receipts(repo)
    receipts = list((repo / D / "receipts" / "pregate").glob("G0.YAW.run.*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "ok" and payload["step"] == "G0.YAW.run.pregate" and len(payload["record_sha256"]) == 64


def test_clean_quarantines_a_failed_model_step_only(checkout, tmp_path):
    repo, env, _seven = checkout
    run_dir = repo / D / "runs" / "g0_YAW"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{}")
    done = _bash("clean", "G0.YAW.run", env={**env, "W": str(tmp_path / "waves")})
    assert done.returncode == 0, done.stderr
    assert not run_dir.exists() and list((repo / D / "runs" / ".quarantine").glob("g0_YAW.*.json"))
    assert _bash("clean", "P0.1", env=env).returncode == 2                    # not a model step


def test_check_leaf_verifies_the_launch_commit(tmp_path):
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "f").write_text("x")
    for argv in (["git", "init", "-q"], ["git", "add", "f"],
                 ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"]):
        subprocess.run(argv, cwd=leaf, check=True)
    done = _bash("check-leaf", env={"LEAF_ROOT": str(leaf)})
    assert done.returncode == 2 and "launch commit" in done.stderr
