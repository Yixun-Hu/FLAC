"""exp_36 yaw_c4_spread, cycle 5: the ONE fail-closed launch chain ``yaw_c4_spread_chain.sh`` and the runner's
binding of it (plan_yaw_c4_spread.md r0 §2 "Run" as amended by the Codex plan review B4 / H5 / H6).

Part 1 drives the REAL runner ``depth_canon_eval_run.sh`` in harness mode (stub evaluator, stub nvidia-smi / findmnt /
df, fake checkpoints) against test_depth_canon_eval_run's throwaway clone (its sparse checkout includes the exp_36
folder: the chain belongs to every no-flash selection's relevant tree, which must be hashable): a YAW-C4S cell
spawns through the no-flash launcher on lane 1 (with --no-dump-blocks: a runner capability the chain no longer uses --
since the Planner's addendum every chain cell, every replay included, dumps) and gets receipt 2; the chain's own runner
calls are dry-run with their dumps; and the chain script belongs to the relevant tree of every no-flash selection (an
uncommitted edit stops it; an exp_25-only selection is unaffected).
Part 2 drives the chain itself with a stub runner and a stub comparator: (1) the replay -> (2) the check -> (3) the
24 only on an exact 0; every other outcome stops before (3); DRY_RUN never runs the check.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import json
import os
import subprocess

import pytest

import test_depth_canon_eval_run as R
from test_depth_canon_eval_run import clone, harness  # noqa: F401 -- the throwaway clone (exp_36 folder included)

C = R.C
EXP36REL = os.path.join("worklog", "worklog_yixun", "exp_36_yaw_c4_spread_claude")
CHAIN_REL = os.path.join(EXP36REL, "yaw_c4_spread_chain.sh")
REPLAY = "exp36_YAW_C4_S40000_K8_s42_th90"
CELLS24 = [f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)]
# amendment A1, chain I ("offorbit"): the four replays, then the 24 off-orbit s42 production cells
REPLAYS4 = [f"exp36_{a}_C4_S40000_K8_s42_th90" for a in ("YAW", "P1", "BF", "CANON")]
OFF24 = ([f"exp35_{a}_OFF_S40000_K{k}_s42_th{d}" for a in ("P1", "BF", "CANON") for k in (1, 8) for d in (135, 225)]
         + [f"exp35_{a}_OFF_S40000_K{k}_s42_th315" for a in ("P1", "BF", "CANON") for k in (1, 8)]
         + [f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" for k in (1, 8) for d in (135, 225, 315)])
CANON_CKPTS = os.path.join(R.EXPREL, "depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256")


# --------------------------------------------------------------------------- #
# part 1: the real runner, YAW-C4S cells
# --------------------------------------------------------------------------- #
def test_the_runner_dry_run_spawns_the_replay_through_the_no_flash_launcher(clone, harness):
    rc, out = R.drive(clone, harness, "--only", REPLAY, "--no-dump-blocks", "YAW-C4S", "--lanes", "1",
                      env={"DRY_RUN": "1"})
    assert rc == 0 and "no-flash pre-flight OK" in out and "(1 exp_35 cell(s))" in out, out
    row = R.manifest_row(harness["manifest"], REPLAY)
    link = f"{harness['dir']}/farm/YAW/epoch=8-step=40000.ckpt"
    assert R._dry_argv(out) == {REPLAY: row["eval_args"] + ["--ckpt-path", link]}           # no dump flag
    assert "spawned through overlays/no_flash/exp35_no_flash_launch.py" in out and "lanes 1" in out
    assert not (harness["dir"] / "calls.log").exists()


def test_the_runner_runs_yaw_c4s_cells_on_lane_1_without_dumps_and_receipt_2(clone, harness):
    cells = [REPLAY, CELLS24[0], CELLS24[-1]]
    rc, out = R.drive(clone, harness, "--only", ",".join(cells), "--no-dump-blocks", "YAW-C4S", "--lanes", "1")
    assert rc == 0 and "CAMPAIGN SELECTION COMPLETE" in out and "3/3" in out, out
    calls = R._calls(harness)
    assert sorted(R._ran(harness)) == sorted(cells) and {c["cvd"] for c in calls} == {"1"}
    assert all(c["no_flash"] is True and "--dump-scored-waveforms" not in c["argv"] for c in calls)
    for cell in cells:
        e = R._entry(clone, harness, cell)
        rec = json.load(open(e["receipt"]))
        assert (rec["schema_version"], rec["attention_backend"], rec["dump"], rec["lane"]) == (2, "sdpa", False, 1)
        assert not os.path.exists(e["dump_dir"])
        assert R._verify(clone, harness, cell, expect_dump=False,
                         backend_target=str(harness["dir"] / "stub_eval.py")) == [], cell


def test_the_runner_dry_run_of_chain_i_plans_the_4_replays_and_the_24_off_orbit_cells(clone, harness):
    """As the chain calls it (addendum: no --no-dump-blocks, every cell dumps into the dump root)."""
    rc, out = R.drive(clone, harness, "--only", ",".join(REPLAYS4), "--lanes", "0,1", env={"DRY_RUN": "1"})
    assert rc == 0 and "(4 exp_35 cell(s))" in out and out.count("spawned through overlays/no_flash/") == 4, out
    m, h = harness["manifest"], harness["dir"]
    want = {n: R.manifest_row(m, n)["eval_args"] + ["--ckpt-path", f"{h}/farm/{R.manifest_row(m, n)['arm']}/epoch=8-step"
                                                                      "=40000.ckpt", "--dump-scored-waveforms", f"{h}/dumps"]
            for n in REPLAYS4}
    assert R._dry_argv(out) == want and "dump root OK under DRY_RUN" in out
    rc, out = R.drive(clone, harness, "--only", ",".join(OFF24), "--lanes", "0,1", env={"DRY_RUN": "1"})
    assert rc == 0 and "(24 exp_35 cell(s))" in out and out.count("spawned through overlays/no_flash/") == 24, out
    got = R._dry_argv(out)
    assert sorted(got) == sorted(OFF24) and all(a[-2:] == ["--dump-scored-waveforms", f"{h}/dumps"] for a in got.values())


def test_the_runner_runs_the_4_replays_on_both_lanes_with_receipt_2_and_their_dumps(clone, harness):
    rc, out = R.drive(clone, harness, "--only", ",".join(REPLAYS4), "--lanes", "0,1")
    assert rc == 0 and "4/4" in out and sorted(R._ran(harness)) == sorted(REPLAYS4), out
    assert {c["cvd"] for c in R._calls(harness)} == {"0", "1"} and all(c["no_flash"] for c in R._calls(harness))
    for cell in REPLAYS4:
        e = R._entry(clone, harness, cell)
        rec = json.load(open(e["receipt"]))
        assert (rec["schema_version"], rec["attention_backend"], rec["dump"]) == (2, "sdpa", True), cell
        assert os.path.isfile(os.path.join(e["dump_dir"], "COMPLETE")), cell
        assert R._verify(clone, harness, cell, backend_target=str(harness["dir"] / "stub_eval.py")) == [], cell


def test_the_chain_script_is_in_the_relevant_tree_of_a_no_flash_selection(clone, harness):
    path = clone / CHAIN_REL
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"# drift\n")
        rc, out = R.drive(clone, harness, "--only", REPLAY, "--no-dump-blocks", "YAW-C4S", "--lanes", "1",
                          env={"DRY_RUN": "1"})
        assert rc == 2 and "NOT COMMITTED" in out and CHAIN_REL in out, out
        rc, out = R.drive(clone, harness, "--only", "exp25_P1_T_S40000_K1_s42", env={"DRY_RUN": "1"})
        assert rc == 0, out                                              # an exp_25-only selection: not bound
    finally:
        path.write_bytes(original)


# --------------------------------------------------------------------------- #
# part 2: the chain, with a stub runner and a stub comparator
# --------------------------------------------------------------------------- #
STUB_RUNNER = r"""#!/usr/bin/env bash
H="$EXP36_CHAIN_HARNESS"
n=$(( $(cat "$H/runner.count" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$H/runner.count"
{ echo "RUNNER"; echo "env DRY_RUN=${DRY_RUN-unset} EXPECT_HEAD=${EXPECT_HEAD-unset} ALLOW_SHARED_GPU=${ALLOW_SHARED_GPU-unset} COT=${EXP35_APPROVED_COTENANTS-unset}"
  printf '%s\n' "$@"; echo END; } >> "$H/calls.txt"
echo "=== exp_25 depth_canon eval campaign -- 2026-11-06_20-00-0${n} -- HEAD x -- dry_run=${DRY_RUN} -- harness=false ==="
if [ "${STUB_NO_PLAN:-0}" != 1 ]; then echo '{}' > "$H/plan_${n}.json"; echo "plan: 1 cell(s) -> $H/plan_${n}.json"; fi
rc_var="STUB_RUNNER_RC_${n}"; exit "${!rc_var:-0}"
"""
STUB_COLLECT = r"""import json, os, sys
H, a = os.environ["EXP36_CHAIN_HARNESS"], sys.argv[1:]
kind = "PREFLIGHT" if "--exp36-dump-preflight" in a else "COLLECT"
with open(os.path.join(H, "calls.txt"), "a") as f:
    f.write(kind + "\nenv CVD=%s\n" % os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + "\n".join(a) + "\nEND\n")
n = len([x for x in os.listdir(H) if x.startswith(kind.lower() + ".")]) + 1
open(os.path.join(H, f"{kind.lower()}.{n}"), "w").close()
if kind == "PREFLIGHT":
    rc = int(os.environ.get(f"STUB_PREFLIGHT_RC_{n}", "0"))
    print("EXP36 DUMP PREFLIGHT " + ("OK" if rc == 0 else "REFUSED"))
    sys.exit(rc)
rc = int(os.environ.get(f"STUB_CHECK_RC_{n}", os.environ.get("STUB_CHECK_RC", "0")))
verdict = os.environ.get("STUB_VERDICT", {0: "equal", 1: "mismatch"}.get(rc, "refused"))
if verdict != "none":
    with open(a[a.index("--verdict-out") + 1], "x") as f:
        json.dump({"result": verdict}, f)
print("EXP36 REPLAY " + {0: "EQUAL", 1: "MISMATCH"}.get(rc, "REFUSED"))
sys.exit(rc)
"""


@pytest.fixture()
def ch(tmp_path, clone):
    h = tmp_path / "ch"
    h.mkdir()
    (h / ".exp36_chain_harness").write_text("")
    (h / "stub_runner.sh").write_text(STUB_RUNNER)
    (h / "stub_collect.py").write_text(STUB_COLLECT)
    return {"dir": h, "head": R._git(clone, "rev-parse", "HEAD")}


def chain(clone, ch, env=None, args=("A",)):
    h = ch["dir"]
    e = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "EXPECT_HEAD", "ALLOW_SHARED_GPU",
                                                            "EXP35_APPROVED_COTENANTS", "DEPTH_CANON_EVAL_HARNESS")}
    e.update(EXP36_CHAIN_HARNESS=str(h), EXP36_CHAIN_RUNNER=str(h / "stub_runner.sh"),
             EXP36_CHAIN_COLLECT=str(h / "stub_collect.py"), EXPECT_HEAD=ch["head"])
    e.update(env or {})
    p = subprocess.run(["bash", str(clone / CHAIN_REL), *args], env=e, capture_output=True, text=True, timeout=120,
                       cwd=str(h))
    return p.returncode, p.stdout + p.stderr


def calls(ch):
    path = ch["dir"] / "calls.txt"
    blocks = path.read_text().split("END\n")[:-1] if path.exists() else []
    return [(b.splitlines()[0], b.splitlines()[1], b.splitlines()[2:]) for b in blocks]


RUN_ENV = "env DRY_RUN={} EXPECT_HEAD={} ALLOW_SHARED_GPU=0 COT=unset"


def _verdicts(ch):
    return sorted(p for p in os.listdir(ch["dir"]) if "_verdict_" in p)


def test_chain_ii_runs_the_yaw_replay_its_check_then_the_24(clone, ch):
    rc, out = chain(clone, ch, env={"ALLOW_SHARED_GPU": "0"})
    assert rc == 0 and "CHAIN COMPLETE" in out, out
    (p1, _, pa1), (k1, e1, a1), (k2, e2, a2), (p3, _, pa3), (k3, e3, a3) = calls(ch)
    assert (p1, k1, k2, p3, k3) == ("PREFLIGHT", "RUNNER", "COLLECT", "PREFLIGHT", "RUNNER")
    assert e1 == e3 == RUN_ENV.format(0, ch["head"]) and e2 == "env CVD="
    tail = ["--lanes", "0,1", "--canon-ckpt-manifest", CANON_CKPTS]                      # amendment A1: both cards
    pin = ["--pin", ch["head"]]                                                           # review III HIGH 1
    assert pa1 == ["--exp36-dump-preflight", "--only", REPLAY, "--canon-ckpt-manifest", CANON_CKPTS] + pin
    assert pa3 == ["--exp36-dump-preflight", "--only", ",".join(CELLS24), "--canon-ckpt-manifest", CANON_CKPTS] + pin
    assert a1 == ["--only", REPLAY] + tail                                                # dumps ON (addendum)
    (verdict,) = _verdicts(ch)
    assert verdict.endswith(f"_A_verdict_{REPLAY}.json")
    assert a2 == ["--exp36-replay-check", "--exp36-replay", REPLAY, "--pin", ch["head"], "--plan-json",
                  str(ch["dir"] / "plan_1.json"), "--verdict-out", str(ch["dir"] / verdict),
                  "--canon-ckpt-manifest", CANON_CKPTS]
    assert a3 == ["--only", ",".join(CELLS24)] + tail
    _publication_pointer(out, ch["dir"] / verdict)
    logs = [p for p in os.listdir(ch["dir"]) if p.endswith("_A_chain.log")]
    assert len(logs) == 1 and "CHAIN COMPLETE" in (ch["dir"] / logs[0]).read_text()


def test_chain_i_runs_the_4_replays_all_4_checks_then_the_24_off_orbit_cells(clone, ch):
    rc, out = chain(clone, ch, args=("offorbit",))
    assert rc == 0 and "CHAIN COMPLETE" in out, out
    got = calls(ch)
    assert [k for k, _, _ in got] == ["PREFLIGHT", "RUNNER"] + ["COLLECT"] * 4 + ["PREFLIGHT", "RUNNER"]
    tail = ["--lanes", "0,1", "--canon-ckpt-manifest", CANON_CKPTS]
    assert got[0][2] == ["--exp36-dump-preflight", "--only", ",".join(REPLAYS4), "--canon-ckpt-manifest", CANON_CKPTS,
                         "--pin", ch["head"]]
    assert got[1][2] == ["--only", ",".join(REPLAYS4)] + tail                             # dumps ON (addendum)
    assert [a[a.index("--exp36-replay") + 1] for _, _, a in got[2:6]] == REPLAYS4
    assert all(a[a.index("--plan-json") + 1] == str(ch["dir"] / "plan_1.json") for _, _, a in got[2:6])
    assert got[6][2] == ["--exp36-dump-preflight", "--only", ",".join(OFF24), "--canon-ckpt-manifest", CANON_CKPTS,
                         "--pin", ch["head"]]
    assert got[7][2] == ["--only", ",".join(OFF24)] + tail
    assert len(_verdicts(ch)) == 4 and all("_offorbit_verdict_" in v for v in _verdicts(ch))
    yaw = [v for v in _verdicts(ch) if REPLAY in v][0]
    _publication_pointer(out, ch["dir"] / yaw)


def _publication_pointer(out, verdict):
    """Review III NIT 1: the chain cannot know the later publisher pin, so it prints only a pointer to the reviewed
    command document (and the verdict that command needs); any printed publication command would have to carry
    --publisher-pin, --base-dir and the publisher checkout -- an incomplete one fails this test."""
    for ln in (ln for ln in out.splitlines() if "--exp36-publish" in ln):
        assert "--publisher-pin" in ln and "--base-dir" in ln and "cd /home/yixunhu/codespace/exp-36-yaw-c4-spread" in ln
    assert "yaw_c4_spread_command.md" in out and str(verdict) in out


@pytest.mark.parametrize("n, rc_want", [(1, 4), (2, 4), (3, 5), (4, 5)])
def test_chain_i_stops_at_the_first_failed_check(clone, ch, n, rc_want):
    rc, out = chain(clone, ch, env={f"STUB_CHECK_RC_{n}": "1" if rc_want == 4 else "2"}, args=("offorbit",))
    assert rc == rc_want and "CHAIN COMPLETE" not in out, out
    assert [k for k, _, _ in calls(ch)] == ["PREFLIGHT", "RUNNER"] + ["COLLECT"] * n     # nothing past it


@pytest.mark.parametrize("mode", ["A", "offorbit"])
@pytest.mark.parametrize("n, kinds", [(1, ["PREFLIGHT"]), (2, None)])
def test_a_dump_collision_stops_the_chain_before_its_runner_step(clone, ch, mode, n, kinds):
    """The addendum: every cell dumps; a target dump directory this checkout did not publish stops the chain before
    the runner step it would collide in (the replays' or the production cells')."""
    rc, out = chain(clone, ch, env={f"STUB_PREFLIGHT_RC_{n}": "2"}, args=(mode,))
    assert rc == 2 and "DUMP PREFLIGHT REFUSED" in out and "CHAIN COMPLETE" not in out, out
    checks = ["COLLECT"] * (4 if mode == "offorbit" else 1)
    assert [k for k, _, _ in calls(ch)] == (kinds or ["PREFLIGHT", "RUNNER", *checks, "PREFLIGHT"])


@pytest.mark.parametrize("args", [(), ("B",), ("A", "offorbit"), ("--dry-run",)])
def test_the_chain_takes_exactly_one_mode(clone, ch, args):
    rc, out = chain(clone, ch, args=args)
    assert rc == 2 and "mode" in out and not calls(ch), out


def test_a_changed_canon_checkpoint_manifest_is_refused(clone, ch):
    path = clone / CANON_CKPTS
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        rc, out = chain(clone, ch, args=("offorbit",))
        assert rc == 2 and "CANON checkpoint manifest" in out and not calls(ch), out
    finally:
        path.write_bytes(original)


_R, _C, _P = "RUNNER", "COLLECT", "PREFLIGHT"


@pytest.mark.parametrize("env, rc_want, needle, kinds", [
    ({"STUB_CHECK_RC": "1"}, 4, "REPLAY MISMATCH", [_P, _R, _C]),
    ({"STUB_CHECK_RC": "2"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),
    ({"STUB_CHECK_RC": "3"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),
    ({"STUB_RUNNER_RC_1": "2"}, 3, "the replay run exited 2", [_P, _R]),
    ({"STUB_RUNNER_RC_1": "1"}, 3, "the replay run exited 1", [_P, _R]),
    ({"STUB_NO_PLAN": "1"}, 3, "no single plan", [_P, _R]),
    ({"STUB_RUNNER_RC_2": "1"}, 6, "re-run this chain to resume", [_P, _R, _C, _P, _R]),
    ({"STUB_RUNNER_RC_2": "4"}, 6, "the production run exited 4", [_P, _R, _C, _P, _R]),
    ({"STUB_VERDICT": "none"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),                  # rc 0 without a verdict
    ({"STUB_VERDICT": "mismatch"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),              # rc 0, not equal
    ({"STUB_CHECK_RC": "1", "STUB_VERDICT": "none"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),   # a crash
    ({"STUB_CHECK_RC": "1", "STUB_VERDICT": "equal"}, 5, "REPLAY CHECK REFUSED", [_P, _R, _C]),
])
def test_every_other_outcome_is_a_hard_stop_before_the_24(clone, ch, env, rc_want, needle, kinds):
    rc, out = chain(clone, ch, env=env)
    assert rc == rc_want and needle in out and "CHAIN COMPLETE" not in out, out
    assert [k for k, _, _ in calls(ch)] == kinds


@pytest.mark.parametrize("mode", ["A", "offorbit"])
def test_dry_run_plans_both_runner_steps_and_never_runs_the_check(clone, ch, mode):
    rc, out = chain(clone, ch, env={"DRY_RUN": "1"}, args=(mode,))
    assert rc == 0 and "DRY_RUN complete" in out and "step 2/3 (DRY_RUN: not run)" in out, out
    got = calls(ch)
    assert [(k, e) for k, e, _ in got if k == "RUNNER"] == [("RUNNER", RUN_ENV.format(1, ch["head"]))] * 2
    assert [k for k, _, _ in got] == ["PREFLIGHT", "RUNNER", "PREFLIGHT", "RUNNER"]   # the preflights run in DRY_RUN
    assert not _verdicts(ch)


@pytest.mark.parametrize("env, needle", [
    ({"EXPECT_HEAD": "0" * 40}, "HEAD PIN"), ({"EXPECT_HEAD": ""}, "EXPECT_HEAD"), ({"DRY_RUN": "2"}, "DRY_RUN"),
    ({"ALLOW_SHARED_GPU": "1"}, "ALLOW_SHARED_GPU"), ({"EXP35_APPROVED_COTENANTS": "/tmp/x"}, "co-tenant"),
    ({"EXP36_CHAIN_HARNESS": "/var/tmp"}, "HARNESS"), ({"EXP36_CHAIN_RUNNER": "/bin/true"}, "HARNESS")])
def test_the_chain_gates_refuse_before_any_step(clone, ch, env, needle):
    rc, out = chain(clone, ch, env=env)
    assert rc == 2 and needle in out and not calls(ch), out


def test_outside_the_harness_a_test_variable_is_refused(clone, ch):
    e = {"EXP36_CHAIN_HARNESS": "", "DEPTH_CANON_EVAL_HARNESS": str(ch["dir"])}
    rc, out = chain(clone, ch, env=e)
    assert rc == 2 and "test-harness variable" in out and not calls(ch), out


def test_an_uncommitted_chain_is_refused(clone, ch):
    path = clone / CHAIN_REL
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# drift\n")
        rc, out = chain(clone, ch)
        assert rc == 2 and "not committed" in out and not calls(ch), out
    finally:
        path.write_bytes(original)
