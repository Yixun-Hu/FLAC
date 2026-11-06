"""exp_35 phase (ii) kit, part 6b (plan_orbitrir_yaw_evidence.md §15 item 10; Codex plan §15 r1 S6, binding): the
Planner-run CLI ``exp35_phase_ii.sh`` (reserve | dry-run | launch | status | release) end to end in the sandbox --
the self-pin by EXPECT_KIT_COMMIT, the phase-(i) launch environment for the helper whatever the caller's environment,
a log per mutating mode (never for status), the detached wrapper outliving its launcher -- and the never-list as
static checks of the kit's own source (no signal, kill or attach; no git write; no install; no sharing; no
--no-dump-blocks; no E2-FAIL; no exp_22; nvidia-smi only for its two queries).

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import make_box

K = SB.load_kit()


def _lab(tmp_path):
    smi = SB.fake_nvidia_smi(tmp_path)
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path), nvidia_smi=smi, gap_s=0.2)
    SB.stub_control(tmp_path, dry={}, run={})
    b.sh = os.path.join(b.kit_dir, "exp35_phase_ii.sh")
    return b


def _sh(b, *args, expect=True, extra=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **(extra or {})}
    if expect:
        env["EXPECT_KIT_COMMIT"] = b.commit
    return subprocess.run(["bash", b.sh, *args], env=env, capture_output=True, text=True, timeout=600)


def _logs(b):
    return sorted(n for n in os.listdir(b.kit_dir) if "_phase_ii_" in n and n.endswith(".log"))


def test_the_five_modes_end_to_end_through_the_cli(tmp_path):
    b = _lab(tmp_path)
    dirty = {"ALLOW_SHARED_GPU": "1", "EXP35_APPROVED_COTENANTS": "/x.json", "PYTHONPATH": "/x", "DRY_RUN": "1",
             "GIT_DIR": os.path.join(b.tmp, "not-a-git-dir"), "GPU_MAX_POLLS": "1"}  # env -i drops all of them
    p = _sh(b, "reserve", b.head, "--lanes", "0", "--report", b.report, "--report-sha256", b.report_sha, extra=dirty)
    assert p.returncode == 0 and "RESERVED " in p.stdout, p.stdout + p.stderr
    outputs = [p.stdout]
    orch = re.search(r"^RESERVED (\S+):", p.stdout, re.M)[1]
    p = _sh(b, "dry-run", b.head, "--orch", orch, extra=dirty)
    assert p.returncode == 0 and "DRY RUN PASSED" in p.stdout, p.stdout[-3000:]
    outputs.append(p.stdout)
    p = _sh(b, "launch", b.head, "--orch", orch, extra=dirty)
    assert p.returncode == 0 and "LAUNCHED: wrapper pid" in p.stdout, p.stdout[-3000:]
    outputs.append(p.stdout)
    deadline = time.time() + 120
    while not (json.load(open(os.path.join(orch, "state.json")))["phase"] == "exited"
               and json.load(open(os.path.join(orch, "state.json")))["exit"]["logs"]):
        assert time.time() < deadline, "the detached wrapper did not finish"
        time.sleep(0.2)
    st = json.load(open(os.path.join(orch, "state.json")))
    assert st["exit"]["rc"] == 0 and st["runner"]["sharing_absent"] and st["wrapper"]["environ_is_the_launch_env"]
    [_, run] = SB.stub_calls(tmp_path)
    assert SB.stub_env(tmp_path, run["pid"]) == dict(K.LAUNCH_ENV, EXPECT_HEAD=b.head)     # the caller's env dropped
    while os.path.exists(f"/proc/{st['wrapper']['pid']}"):                       # detached: not our child to reap
        assert time.time() < deadline
        time.sleep(0.1)
    logs = _logs(b)
    p = _sh(b, "status", "--orch", orch)
    assert p.returncode == 0 and "phase exited" in p.stdout and "receipts: 68/68" in p.stdout and _logs(b) == logs
    p = _sh(b, "release", b.head, "--orch", orch)
    assert p.returncode == 0 and "RELEASED" in p.stdout and not os.path.lexists(b.res())
    outputs.append(p.stdout)
    modes = [re.search(r"_phase_ii_(\w[\w-]*)\.log$", n)[1] for n in _logs(b)]
    assert sorted(modes) == ["dry-run", "launch", "release", "reserve"]
    for out in outputs:                       # Codex P2K-1 NIT 1: the FINAL file's digest, printed outside the log
        digest, path = re.search(r"^log sha256 ([0-9a-f]{64})  (\S+) \(the final file", out, re.M).groups()
        raw = open(path, "rb").read()
        assert hashlib.sha256(raw).hexdigest() == digest and raw.decode().splitlines()[-1] == f"rc 0; log {path}"
        assert digest not in raw.decode()


def test_without_the_reviewed_commit_a_mode_refuses(tmp_path):
    b = _lab(tmp_path)
    p = _sh(b, "reserve", b.head, "--lanes", "0", "--report", b.report, "--report-sha256", b.report_sha, expect=False)
    assert p.returncode == 1 and "40-hex" in p.stdout and not os.path.lexists(b.res())


@pytest.mark.parametrize("args", [[], ["nonsense"], ["reserve"], ["reserve", "x" * 40], ["dry-run", "x" * 40],
                                  ["reserve", "a" * 40, "--lanes", "0,0", "--report", "r", "--report-sha256", "0" * 64]])
def test_bad_arguments_are_refused_with_rc_2(tmp_path, args):
    b = _lab(tmp_path)
    assert _sh(b, *args).returncode == 2 and not os.path.lexists(b.res())


def test_the_cli_environment_is_the_launch_environment():
    text = open(SB.KIT_SH).read()
    body = text[text.index("ENV=(") + 5:text.index(")", text.index("ENV=("))]
    assert dict(x.split("=", 1) for x in body.split()) == K.LAUNCH_ENV
    assert re.search(r"^PY=/home/yixunhu/miniconda3/envs/flac/bin/python$", text, re.M)


# --------------------------------------------------------------------------- #
# the never-list (S6), statically, on the kit's own source
# --------------------------------------------------------------------------- #
SRC = {p: open(p).read() for p in (SB.KIT_PY, SB.KIT_SH)}


def _code(path):
    """The source without docstrings and comments (what executes)."""
    tree = ast.parse(SRC[path])
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.unparse(tree)


def test_the_kit_never_signals_kills_or_attaches():
    code = _code(SB.KIT_PY)
    for bad in ("os.kill", "killpg", "import signal", "signal.", ".kill(", ".terminate(", "send_signal", "ptrace",
                "pkill", "kill -", "os.abort"):
        assert bad not in code, bad
    sh = "\n".join(ln for ln in SRC[SB.KIT_SH].splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\b(kill|pkill|killall|timeout|trap)\b", sh)


KILLING = {"run", "call", "check_call", "check_output", "communicate", "wait"}   # a timeout here kills the child


def test_no_production_call_can_kill_a_child_through_a_timeout():
    """Codex P2K-1 BLOCKING 1: subprocess.run(..., timeout=) / communicate(timeout=) / wait(timeout=) kill (or
    abandon to a kill) the child on expiry -- never in the kit. A non-killing observation deadline (a monotonic
    clock compared in a loop) stays allowed."""
    bad = [ast.unparse(c) for c in ast.walk(ast.parse(SRC[SB.KIT_PY])) if isinstance(c, ast.Call)
           and isinstance(c.func, ast.Attribute) and c.func.attr in KILLING
           and (any(k.arg == "timeout" for k in c.keywords) or (c.func.attr in ("communicate", "wait") and c.args))]
    assert bad == []


def test_every_git_call_is_a_read():
    tree = ast.parse(SRC[SB.KIT_PY])
    used = {c.args[1].value for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "git"
            and len(c.args) > 1 and isinstance(c.args[1], ast.Constant)}
    assert used == {"rev-parse", "status", "ls-files", "cat-file"}
    assert '"git", "--no-optional-locks"' in SRC[SB.KIT_PY]


def test_subprocesses_run_only_git_nvidia_smi_the_runner_and_the_wrapper():
    tree = ast.parse(SRC[SB.KIT_PY])
    owners = set()
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef):
            for c in ast.walk(f):
                if isinstance(c, ast.Attribute) and isinstance(c.value, ast.Name) and c.value.id == "subprocess" and \
                        c.attr in ("run", "Popen", "call", "check_call", "check_output"):
                    owners.add(f.name)
    assert owners == {"git", "nvsmi", "spawn", "dry_spawn"}
    assert K.World.QUERIES == {"map": "--query-gpu=index,uuid", "apps": "--query-compute-apps=pid,gpu_uuid"}


@pytest.mark.parametrize("token", [r"\bpip\b", r"\bconda\b", r"\binstall\b", "--no-dump-blocks", "exp22",
                                   "ALLOW_SHARED_GPU=1", "EXP35_APPROVED_COTENANTS="])
def test_nothing_installs_shares_suppresses_dumps_or_starts_exp22(token):
    assert not re.search(token, _code(SB.KIT_PY))
    assert not re.search(token, "\n".join(ln for ln in SRC[SB.KIT_SH].splitlines() if not ln.lstrip().startswith("#")))
    assert "E2-FAIL" not in K.BLOCKS


@pytest.mark.parametrize("path", [SB.KIT_PY, SB.KIT_SH] + sorted(
    os.path.join(SB.HERE, n) for n in os.listdir(SB.HERE) if n.startswith(("test_exp35_phase_ii", "exp35_phase_ii"))))
def test_one_final_newline_and_no_trailing_whitespace(path):
    raw = open(path, "rb").read()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert not [i for i, ln in enumerate(raw.split(b"\n"), 1) if ln != ln.rstrip()]
