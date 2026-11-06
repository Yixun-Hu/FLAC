"""exp_35 phase (ii) kit RETIRED (exp_36 code review III HIGH 2): Yixun superseded phase (ii) on 2026-11-06 15:30 EST
("Seeds 43-46 at these angles are NOT wanted"); its 24-cell seed-42 off-orbit replacement runs only through the exp_36
chain, and the generic exp_25 runner this kit reaches would quarantine the H35 OFF dumps that chain shares by name.
``exp35_phase_ii.sh`` refuses reserve / dry-run / launch with rc 1 BEFORE any log, helper invocation, orchestration,
reservation, runner or nvidia-smi query -- proven by spies on the helper's interpreter and on nvidia-smi, the stub
runner's call log and a byte-level snapshot of the whole sandbox -- and the helper's own CLI (``main``) refuses them
first too, before its parser, config or world exist, and never dispatches them; status and release keep working.
Sandbox only (``exp35_phase_ii_sandbox``): never the real kit, runner, reservations or a GPU.

Written by the exp_35 kit-retirement Coder (Claude Opus 5.5, max effort).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pytest

import exp35_phase_ii_sandbox as SB
from exp35_phase_ii_sandbox import FakeWorld, make_box

K = SB.load_kit()
RETIRED, T_DRY = ("reserve", "dry-run", "launch"), 1_800_000_100.0
UPTO = {"reserve": None, "dry-run": "reserved", "launch": "dry_run_passed"}     # what each mode needed to proceed
DIRTY = {"ALLOW_SHARED_GPU": "1", "EXP35_APPROVED_COTENANTS": "/x.json", "PYTHONPATH": "/x", "DRY_RUN": "1",
         "GPU_MAX_POLLS": "1"}
LINE = re.compile(r"REFUSED \(RETIRED\): (exp35_phase_ii\.(?:sh|py)) (reserve|dry-run|launch) -- (.+)\n")
PHRASES = ("the exp_35 phase-(ii) kit is RETIRED", "Yixun superseded phase (ii) on 2026-11-06 15:30 EST",
           "'Seeds 43-46 at these angles are NOT wanted'", "ONLY through the exp_36 chain",
           "worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh", "exp_36 code review III HIGH 2",
           "nothing was logged, reserved, spawned or queried", "status and release remain")


def refusal_body(out, prog, mode):
    """The body of the ONE refusal line ``out`` must be (the shell and the helper print the same text)."""
    m = LINE.fullmatch(out)
    assert m and (m[1], m[2]) == (prog, mode), out
    assert all(p in m[3] for p in PHRASES), m[3]
    return m[3]


def tree(root, skip=None):
    """Every path under ``root`` and root itself: lstat mode / size / mtime / inode, link target, the bytes' sha256."""
    out = {}
    for d, dirs, files in os.walk(root):
        for p in [d] + [os.path.join(d, n) for n in dirs + files]:
            if skip and (p == skip or p.startswith(skip + os.sep)):
                continue
            st, link = os.lstat(p), os.path.islink(p)
            data = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) and not link else None
            out[p] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino, os.readlink(p) if link else None, data)
    return out


def spies(tmp_path):
    """The helper's interpreter and nvidia-smi as spies: every call is appended to <tmp>/spy/<name>.calls."""
    spy, out = os.path.join(str(tmp_path), "spy"), {}
    os.makedirs(spy)
    for name, target in (("py", sys.executable), ("nvidia-smi", SB.fake_nvidia_smi(tmp_path))):
        out[name] = os.path.join(spy, name)
        with open(out[name], "w") as f:
            f.write(f"#!/bin/sh\necho \"$*\" >> '{spy}/{name}.calls'\nexec '{target}' \"$@\"\n")
        os.chmod(out[name], 0o755)
    return spy, out


def wait_exited(b):
    """The detached wrapper's exit recorded with its logs, then the wrapper (our child here) reaped."""
    deadline = time.time() + 120
    while not (_state(b)["phase"] == "exited" and _state(b)["exit"]["logs"]):
        assert time.time() < deadline, "the detached wrapper did not finish"
        time.sleep(0.1)
    pid = _state(b)["wrapper"]["pid"]
    while time.time() < deadline:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                break
        except ChildProcessError:
            break
        time.sleep(0.05)


def box(tmp_path, upto=None, spy=True):
    """A sandbox kit checkout (the kit's files, their config blocks only replaced) and run worktree; ``upto`` carries
    a legacy orchestration that far through the dormant library machinery first (no CLI)."""
    b_spy, s = spies(tmp_path) if spy else (None, {"py": sys.executable, "nvidia-smi": SB.fake_nvidia_smi(tmp_path)})
    b = make_box(tmp_path, runner=SB.install_stub(tmp_path), nvidia_smi=s["nvidia-smi"], py=s["py"], gap_s=0.2)
    SB.stub_control(tmp_path, dry={}, run={})
    b.sh, b.spy, b.orch = os.path.join(b.kit_dir, "exp35_phase_ii.sh"), b_spy, None
    if upto:
        rc, msg = K.reserve(FakeWorld(), b.cfg, kit_commit=b.commit, h35=b.head, lanes=(0,), report=b.report,
                            report_sha256=b.report_sha)
        assert rc == 0, msg
        b.orch = msg.split()[1].rstrip(":")
    if upto in ("dry_run_passed", "exited"):
        rc, msg = K.dry_run(FakeWorld(t=T_DRY), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
        assert rc == 0, msg
    if upto == "exited":
        rc, msg = K.launch(FakeWorld(t=T_DRY + 60), b.cfg, kit_commit=b.commit, h35=b.head, orch=b.orch)
        assert rc == 0, msg
        wait_exited(b)
    return b


def _state(b):
    return json.load(open(os.path.join(b.orch, "state.json")))


def sh(b, *args, commit=True):
    env = dict(DIRTY, PATH=os.environ["PATH"], HOME=os.environ["HOME"])
    env.update({"EXPECT_KIT_COMMIT": b.commit} if commit else {})
    return subprocess.run(["bash", b.sh, *args], env=env, capture_output=True, text=True, timeout=600)


def args_of(b, mode):
    """A retired mode's complete arguments -- valid here: the kit before its retirement would have run them."""
    return [b.head, "--lanes", "0", "--report", b.report, "--report-sha256", b.report_sha] if mode == "reserve" else [
        b.head, "--orch", b.orch]


# --------------------------------------------------------------------------- #
# the shell CLI refuses the three modes before anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", ["valid", "valid without EXPECT_KIT_COMMIT", "bare", "help", "junk"])
@pytest.mark.parametrize("mode", RETIRED)
def test_the_cli_refuses_each_retired_mode_before_anything(tmp_path, mode, shape):
    b = box(tmp_path, UPTO[mode])
    args = {"valid": args_of(b, mode), "valid without EXPECT_KIT_COMMIT": args_of(b, mode), "bare": [],
            "help": ["--help"], "junk": ["x" * 40, "--orch", "/", "--lanes", "0,1"]}[shape]
    calls, before = len(SB.stub_calls(tmp_path)), tree(b.tmp)
    p = sh(b, mode, *args, commit=shape != "valid without EXPECT_KIT_COMMIT")
    assert (p.returncode, p.stderr) == (1, ""), p.stdout + p.stderr
    assert tree(b.tmp) == before                     # no log, orchestration, state, lock, reservation or runner file
    assert sorted(os.listdir(b.spy)) == ["nvidia-smi", "py"]                 # the helper and nvidia-smi never ran
    assert len(SB.stub_calls(tmp_path)) == calls                               # the runner never ran
    assert refusal_body(p.stdout, "exp35_phase_ii.sh", mode) == K.RETIRED_BODY


def test_the_shell_gate_is_its_first_decision():
    """One exact arm, preceded only by `set`, MODE and the message: before the config block, HERE / HELPER / ENV,
    the other modes, the log and the helper."""
    lines = open(SB.KIT_SH).read().splitlines()
    code = [i for i, ln in enumerate(lines) if ln.strip() and not ln.lstrip().startswith("#")]
    gate = [i for i in code if "reserve|dry-run|launch)" in lines[i]]
    assert len(gate) == 1 and lines[gate[0]] == ('case "$MODE" in reserve|dry-run|launch) echo "REFUSED (RETIRED): '
                                                 'exp35_phase_ii.sh ${MODE} -- ${RETIRED_BODY}"; exit 1 ;; esac')
    pre = [lines[i] for i in code if i < gate[0]]
    assert pre[:2] == ["set -uo pipefail", 'MODE="${1:-}"'] and pre[2:] and all(
        ln.startswith(("RETIRED_BODY=", "RETIRED_BODY+=")) for ln in pre[2:])


# --------------------------------------------------------------------------- #
# status and release keep working (the dormant library machinery made the legacy orchestrations)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("commit", [True, False])
def test_status_still_works_through_the_cli_and_names_the_retirement(tmp_path, commit):
    b = box(tmp_path)
    before = tree(b.tmp, skip=b.spy)
    p = sh(b, "status", commit=commit)
    assert p.returncode == 0 and tree(b.tmp, skip=b.spy) == before, p.stdout + p.stderr    # read-only, no log
    out = p.stdout.splitlines()
    assert out[0].startswith("exp_35 phase (ii) status at ") and "orchestrations: none" in out
    assert out[1] == "RETIRED: reserve / dry-run / launch refuse -- " + K.RETIRED_BODY
    assert any(ln.startswith("kit: the reviewed commit ") for ln in out) is commit
    assert open(os.path.join(b.spy, "py.calls")).read().split()[:3] == ["-B", b.cfg.helper, "status"]


@pytest.mark.parametrize("upto", ["reserved", "dry_run_passed", "exited"])
def test_release_still_works_through_the_cli(tmp_path, upto):
    b = box(tmp_path, upto, spy=False)
    assert os.path.exists(b.res())
    p = sh(b, "release", b.head, "--orch", b.orch)
    assert p.returncode == 0 and p.stdout.splitlines()[0] == f"RELEASED {b.orch}: {b.res()} (prior phase {upto})", (
        p.stdout + p.stderr)
    st = _state(b)
    assert st["phase"] == "released" and st["released"]["prior_phase"] == upto and not os.path.lexists(b.res())
    [log] = [n for n in os.listdir(b.kit_dir) if "_phase_ii_" in n and n.endswith(".log")]
    digest, path = re.search(r"^log sha256 ([0-9a-f]{64})  (\S+) \(the final file", p.stdout, re.M).groups()
    assert path == os.path.join(b.kit_dir, log) and log.endswith("_phase_ii_release.log")
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest
