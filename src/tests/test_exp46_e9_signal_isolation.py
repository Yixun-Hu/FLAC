"""exp_46 E9 / E10 / E11 (L1 attempt 2; Yixun 2026-11-19, "Fix as recommended"): the cross-test leak that hung the
merged L1 suite.

The supervisor tests' test_unregistered_runs_are_refused_before_any_work -- in test_exp38_supervise.py (E9),
test_exp45_supervise.py (E10) and test_exp46_supervise.py (E11) -- calls sv.main() in the pytest process. main()
installs process-wide SIGTERM / SIGINT / SIGHUP handlers (<kit>_supervise._on_signal, which only record the signal)
and nothing restored them. Every later fork of the process inherits them. A fork Pool's terminate() relies on SIGTERM
to end a worker that is still blocked on the task queue's lock (the parent has taken that lock); the inherited
handler swallows the signal and the pool's join waits for the worker forever. The merged suite hung exactly so in
test_exp40_kit.py::test_concurrent_submitters_from_two_checkouts_share_one_control_dir; on one CPU the parent wins
that lock race almost every time. Each edit (test_exp46_provenance.supervise_restore) restores the three handlers at
that node's teardown.

* test_a_supervisor_refusal_node_then_the_fork_pool_test_finish_and_pass_in_one_process -- RED until each edit lands:
  ONE pytest process, pinned to one CPU, in its own session and killed as a group at the timeout, runs the refusal
  node, then a probe (SIGTERM still stops a forked child; the three handlers are back at their collection-time
  values), then the fork Pool test. Unfixed, the probe fails at once and the fork test hangs.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import os
import signal
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUPERVISE = {"E9": "src/tests/test_exp38_supervise.py", "E10": "src/tests/test_exp45_supervise.py",
             "E11": "src/tests/test_exp46_supervise.py"}
NODE = "test_unregistered_runs_are_refused_before_any_work"
FORK = "src/tests/test_exp40_kit.py::test_concurrent_submitters_from_two_checkouts_share_one_control_dir"
TIMEOUT_S = 120
PIN = ("import os, sys; os.sched_setaffinity(0, {int(sys.argv[1])}); "
       "os.execv(sys.executable, [sys.executable, *sys.argv[2:]])")
PROBE = '''"""exp_46 E9 / E10 / E11 probe (written by test_exp46_e9_signal_isolation.py): Pool.terminate() relies on SIGTERM
ending each forked worker; check exactly that on a forked child, and that SIGTERM / SIGINT / SIGHUP are back at their
collection-time handlers."""
import multiprocessing
import signal
import time

SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
BEFORE = {s.name: signal.getsignal(s) for s in SIGNALS}            # at collection, before any test ran


def test_sigterm_still_stops_a_forked_child_and_the_handlers_are_back():
    now = {s.name: signal.getsignal(s) for s in SIGNALS}
    p = multiprocessing.get_context("fork").Process(target=time.sleep, args=(60,))
    p.start()
    time.sleep(0.5)
    p.terminate()                                  # what Pool.terminate() sends each worker
    p.join(10)
    survived = p.exitcode is None
    if survived:
        p.kill()
        p.join()
    assert not survived and p.exitcode == -signal.SIGTERM, ("a forked child survived SIGTERM", now)
    assert now == BEFORE, now
'''


def _session(*args):
    """(exit status, or None when it hung; its output) of ONE pytest process over ``args`` in order, pinned to one CPU.
    A hung session is killed with its whole process group: the pool workers and the manager it forked go with it."""
    cpu = min(os.sched_getaffinity(0))
    p = subprocess.Popen([sys.executable, "-I", "-c", PIN, str(cpu), "-B", "-I", "-m", "pytest", "-q", "-p",
                          "no:cacheprovider", f"--rootdir={REPO}", *args], cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out = p.communicate(timeout=TIMEOUT_S)[0]
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        return None, p.communicate()[0]


@pytest.mark.parametrize("tag", sorted(SUPERVISE))
def test_a_supervisor_refusal_node_then_the_fork_pool_test_finish_and_pass_in_one_process(tag, tmp_path):
    probe = tmp_path / "test_exp46_signal_probe.py"
    probe.write_text(PROBE)
    rc, out = _session(f"{SUPERVISE[tag]}::{NODE}", str(probe), FORK)
    assert rc == 0 and "3 passed" in out, (tag, rc if rc is not None else f"hung: killed after {TIMEOUT_S} s",
                                           out[-3000:])
