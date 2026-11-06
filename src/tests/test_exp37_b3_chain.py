"""exp_37 B3: the chain's fail-closed mode ``b3`` -- the dump preflight (``--pin EXPECT_HEAD``: its authenticated-receipt
rule, inherited from H36_I3), the replay exp37_BF_C4_S40000_K8_s42_th90 (on
--lanes 0,1, with its dump), its replay check (exactly 0 with an equal verdict), the dump preflight of the 8 BF-ORB cells,
then the 8 cells; no publication (exp_37's outputs stay in the run checkout's farm). Driven with the stub runner /
comparator of test_exp36_chain; the real runner plans the 9 cells in harness mode.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) for exp_37 B3.
"""
import os

import pytest

import test_depth_canon_eval_run as R
from test_depth_canon_eval_run import clone, harness  # noqa: F401
from test_exp36_chain import CANON_CKPTS, RUN_ENV, _verdicts, calls, ch, chain  # noqa: F401

REPLAY = "exp37_BF_C4_S40000_K8_s42_th90"
ORB = [f"exp37_BF_ORB{o}_S40000_K{k}_s42_th{d}" for o in (1, 2) for k in (1, 8) for d in (0, 90)]
TAIL = ["--lanes", "0,1", "--canon-ckpt-manifest", CANON_CKPTS]


def test_chain_b3_runs_its_replay_its_check_then_the_8_orbit_cells(clone, ch):
    rc, out = chain(clone, ch, args=("b3",))
    assert rc == 0 and "CHAIN COMPLETE (b3)" in out and "--exp36-publish" not in out, out
    got = calls(ch)
    assert [k for k, _, _ in got] == ["PREFLIGHT", "RUNNER", "COLLECT", "PREFLIGHT", "RUNNER"]
    pin = ["--pin", ch["head"]]                                  # inherited from H36_I3 (Codex code review III HIGH 1)
    assert got[0][2] == ["--exp36-dump-preflight", "--only", REPLAY, "--canon-ckpt-manifest", CANON_CKPTS] + pin
    assert got[1][2] == ["--only", REPLAY] + TAIL and got[1][1] == RUN_ENV.format(0, ch["head"])
    assert got[2][2][:3] == ["--exp36-replay-check", "--exp36-replay", REPLAY]
    assert got[3][2] == ["--exp36-dump-preflight", "--only", ",".join(ORB), "--canon-ckpt-manifest", CANON_CKPTS] + pin
    assert got[4][2] == ["--only", ",".join(ORB)] + TAIL
    (verdict,) = _verdicts(ch)
    assert verdict.endswith(f"_b3_verdict_{REPLAY}.json")


@pytest.mark.parametrize("env, rc_want, kinds", [
    ({"STUB_CHECK_RC": "1"}, 4, ["PREFLIGHT", "RUNNER", "COLLECT"]),
    ({"STUB_CHECK_RC": "2"}, 5, ["PREFLIGHT", "RUNNER", "COLLECT"]),
    ({"STUB_PREFLIGHT_RC_2": "2"}, 2, ["PREFLIGHT", "RUNNER", "COLLECT", "PREFLIGHT"]),
    ({"STUB_RUNNER_RC_1": "2"}, 3, ["PREFLIGHT", "RUNNER"])])
def test_chain_b3_stops_before_the_orbit_cells_on_any_failure(clone, ch, env, rc_want, kinds):
    rc, out = chain(clone, ch, env=env, args=("b3",))
    assert rc == rc_want and "CHAIN COMPLETE" not in out, out
    assert [k for k, _, _ in calls(ch)] == kinds


def test_the_runner_dry_run_plans_the_replay_and_the_8_orbit_cells_with_dumps(clone, harness):
    rc, out = R.drive(clone, harness, "--only", ",".join([REPLAY] + ORB), "--lanes", "0,1", env={"DRY_RUN": "1"})
    assert rc == 0 and "(9 exp_35 cell(s))" in out and out.count("spawned through overlays/no_flash/") == 9, out
    m, h = harness["manifest"], harness["dir"]
    got = R._dry_argv(out)
    assert sorted(got) == sorted([REPLAY] + ORB)
    for name, argv in got.items():
        assert argv == R.manifest_row(m, name)["eval_args"] + ["--ckpt-path", f"{h}/farm/BF/epoch=8-step=40000.ckpt",
                                                                "--dump-scored-waveforms", f"{h}/dumps"], name
