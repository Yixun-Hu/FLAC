"""exp_22 A7 (plan §9 A7 r2 (v)(1), r3 (d) (h) (i)): the READ-ONLY free-card detector -- its 60-s polls and 30-min
window, GPU 0 first, the late-poll reset, the bounded runtime and a two-hour re-arm, unreadable inputs, and that it
writes nothing; the bash wrapper in a sandbox copy whose ONLY change is the registered-config block."""
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, MAP, FakeWorld, cfg_for, sandbox_tools

T0 = 1_800_000_000.0
DONE = "EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n"


class Scripted(FakeWorld):
    """A fake clock whose compute table is a function of time; ``late`` adds seconds inside the n-th apps query."""

    def __init__(self, apps_at=lambda t: "", late=None, **kw):
        super().__init__(t=T0, **kw)
        self.apps_at, self.late, self.calls, self.polls = apps_at, dict(late or {}), 0, []
        self.answers["apps"] = self._apps

    def _apps(self):
        self.calls += 1
        self.polls.append(self.t)
        self.t += self.late.pop(self.calls, 0)
        return self.apps_at(self.t)


def free_cfg(tmp_path, gpu1_done=True):
    cfg = cfg_for(tmp_path)
    if gpu1_done:
        with open(cfg.queue_log, "a") as f:
            f.write(DONE)
    return cfg


def test_a_card_free_for_31_polls_is_a_candidate_after_exactly_30_minutes(tmp_path):
    w = Scripted(apps_at=lambda t: f"9, {GPU1}\n")                    # GPU 1 busy, GPU 0 idle
    rc, line = C.detect(w, free_cfg(tmp_path), 7200)
    assert rc == 0 and line.startswith(f"EVENT CANDIDATE GPU0 {GPU0} index 0: free at 31 consecutive polls")
    assert w.t == T0 + 1800 and "\n" not in line
    gaps = {round(b - a) for a, b in zip(w.polls[::2], w.polls[2::2])}
    assert gaps == {60}                                                  # one poll per minute, both cards each time


def test_one_busy_poll_restarts_the_window(tmp_path):
    w = Scripted(apps_at=lambda t: f"7, {GPU0}\n" if T0 + 600 <= t < T0 + 660 else f"9, {GPU1}\n")
    rc, line = C.detect(w, free_cfg(tmp_path), 7200)
    assert rc == 0 and "GPU0" in line and w.t == T0 + 660 + 1800


def test_gpu0_is_preferred_when_both_cards_qualify_together(tmp_path):
    rc, line = C.detect(Scripted(), free_cfg(tmp_path), 7200)
    assert rc == 0 and line.startswith("EVENT CANDIDATE GPU0 ")


def test_gpu1_alone_qualifies_while_the_exp35_lock_holds_gpu0(tmp_path):
    from conftest import HOLDER
    cfg = free_cfg(tmp_path)
    p = subprocess.Popen([sys.executable, "-c", HOLDER, cfg.exp35_lock], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "locked"
        rc, line = C.detect(Scripted(), cfg, 7200)
    finally:
        p.kill()
        p.wait()
    assert rc == 0 and line.startswith(f"EVENT CANDIDATE GPU1 {GPU1} index 1")


def test_gpu1_never_qualifies_before_the_table_job_is_done(tmp_path):
    w = Scripted(apps_at=lambda t: f"9, {GPU0}\n")
    rc, line = C.detect(w, free_cfg(tmp_path, gpu1_done=False), 3600)
    assert rc == 3 and line.startswith("EVENT TIMEOUT") and "GPU1 0/31" in line


def test_the_runtime_is_bounded_and_no_poll_lies_past_the_bound(tmp_path):
    w = Scripted(apps_at=lambda t: f"1, {GPU0}\n2, {GPU1}\n")
    rc, line = C.detect(w, free_cfg(tmp_path), 600)
    assert rc == 3 and "after 600 s" in line and max(w.polls) <= T0 + 600 and w.t <= T0 + 600 + 1


def test_a_late_poll_resets_every_window_without_a_catch_up_burst(tmp_path):
    w = Scripted(late={10: 900})                                         # the 10th apps query stalls 15 min
    rc, line = C.detect(w, free_cfg(tmp_path), 7200)
    assert rc == 0
    times = sorted(set(round(t) for t in w.polls))
    assert all(b - a >= 59 for a, b in zip(times, times[1:]))            # never two polls within a minute
    assert w.t >= T0 + 900 + 1800


def test_a_two_hour_re_arm_starts_a_fresh_window(tmp_path):
    """A7 r3 (d): re-armed at most every two hours; the detector keeps no memory, so the window restarts."""
    cfg = free_cfg(tmp_path)
    busy_until = T0 + 6900
    w = Scripted(apps_at=lambda t: f"5, {GPU0}\n5, {GPU1}\n" if t < busy_until else "")
    rc, line = C.detect(w, cfg, 7200)
    assert rc == 3 and line.startswith("EVENT TIMEOUT")
    w.t = T0 + 7210                                                      # the Planner re-arms ten seconds later
    rc, line = C.detect(w, cfg, 7200)
    assert rc == 0 and w.t == T0 + 7210 + 1800                           # not busy_until + 1800


@pytest.mark.parametrize("breakage", ["nvsmi", "lock", "queue", "reservation"])
def test_an_unreadable_input_ends_the_detector_with_one_event(tmp_path, breakage):
    cfg = free_cfg(tmp_path)
    w = Scripted()
    if breakage == "nvsmi":
        w.answers["map"] = lambda: (_ for _ in ()).throw(C.Unreadable("nvidia-smi --query-gpu=index,uuid: rc 6"))
    elif breakage == "lock":
        open(cfg.exp35_lock, "w").close()
        os.chmod(cfg.exp35_lock, 0)
    elif breakage == "queue":
        os.unlink(cfg.queue_log)
    else:
        path = os.path.join(cfg.reservations, f"{GPU1}.json")
        open(path, "w").write("{}")
        os.chmod(path, 0)
    try:
        rc, line = C.detect(w, cfg, 7200)
    finally:
        for p in (cfg.exp35_lock, os.path.join(cfg.reservations, f"{GPU1}.json")):
            if os.path.exists(p):
                os.chmod(p, 0o644)
    assert rc == 4 and line.startswith("EVENT UNREADABLE") and "\n" not in line


def _tree(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.lstat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def test_the_detector_writes_nothing(tmp_path):
    cfg = free_cfg(tmp_path)
    open(cfg.exp35_lock, "w").close()
    before = _tree(tmp_path)
    C.detect(Scripted(), cfg, 7200)
    C.detect(Scripted(apps_at=lambda t: f"1, {GPU0}\n"), cfg, 120)
    assert _tree(tmp_path) == before


# --------------------------------------------------------------------------- #
# the bash wrapper, in a sandbox copy (only the config blocks differ)
# --------------------------------------------------------------------------- #
def _wrapper(tmp_path, apps=""):
    cfg = free_cfg(tmp_path)
    nv = tmp_path / "nvidia-smi"
    (tmp_path / "apps.txt").write_text(apps)
    nv.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '%s' '{MAP}';; "
                  f"*) cat {tmp_path / 'apps.txt'};; esac\n")
    nv.chmod(0o755)
    tools = sandbox_tools(tmp_path / "W", common=dict(
        RESERVATIONS=cfg.reservations, EXP35_LOCK=cfg.exp35_lock, QUEUE_LOG=cfg.queue_log, PROC=cfg.proc,
        NVIDIA_SMI=str(nv), POLL_S=0.05, WINDOW_POLLS=4, POLL_SLACK_S=5, FRESH_S=600, PY=sys.executable,
        PYTEST_MAIN=C.PYTEST_MAIN, CHILD_POLL_S=0.05),
        shells={"exp22_free_card_detect.sh": {"PY": sys.executable}})
    return os.path.join(tools, "exp22_free_card_detect.sh")


def _run(script, *args):
    return subprocess.run(["bash", script, *args], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


def test_the_wrapper_prints_exactly_one_event_line(tmp_path):
    p = _run(_wrapper(tmp_path, apps=f"9, {GPU1}\n"), "60")
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("\n") == 1 and p.stdout.startswith("EVENT CANDIDATE GPU0 ")


def test_the_wrapper_times_out_with_one_event_line(tmp_path):
    p = _run(_wrapper(tmp_path, apps=f"1, {GPU0}\n2, {GPU1}\n"), "1")
    assert p.returncode == 3 and p.stdout.count("\n") == 1 and p.stdout.startswith("EVENT TIMEOUT")


@pytest.mark.parametrize("args", [[], ["0"], ["7201"], ["abc"], ["60", "x"], ["-5"]])
def test_the_wrapper_refuses_a_bad_bound(tmp_path, args):
    p = _run(_wrapper(tmp_path), *args)
    assert p.returncode == 2 and p.stdout.startswith("EVENT ARGUMENT ERROR") and p.stdout.count("\n") == 1


def test_the_registered_wrapper_runs_the_flac_interpreter_on_the_detector():
    text = open(os.path.join(os.path.dirname(C.__file__), "exp22_free_card_detect.sh")).read()
    assert "PY=/home/yixunhu/miniconda3/envs/flac/bin/python\n" in text
    assert '"$PY" -B "$HERE/exp22_r2c_common.py" detect --max-seconds "$1"' in text
    assert "exec " not in text.split("# --- END REGISTERED CONFIG ---")[1]     # the bash parent stays visible


def test_the_cli_observe_prints_one_json_observation(tmp_path):
    w = _wrapper(tmp_path, apps=f"9, {GPU0}\n")
    common = os.path.join(os.path.dirname(w), "exp22_r2c_common.py")
    p = subprocess.run([sys.executable, "-B", common, "observe", "--card", "GPU0"], capture_output=True, text=True)
    assert p.returncode == 1 and json.loads(p.stdout)["pids_on_card"] == [9]
    p = subprocess.run([sys.executable, "-B", common, "observe", "--card", "GPU1"], capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout)["free"] is True
