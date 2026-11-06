"""exp_36 yaw_c4_spread, cycle 5: the ONE fail-closed launch chain ``yaw_c4_spread_chain.sh`` and the runner's
binding of it (plan_yaw_c4_spread.md r0 §2 "Run" as amended by the Codex plan review B4 / H5 / H6).

Part 1 drives the REAL runner ``depth_canon_eval_run.sh`` in harness mode (stub evaluator, stub nvidia-smi / findmnt /
df, fake checkpoints) against a throwaway clone whose sparse checkout includes the exp_36 folder: a YAW-C4S cell
spawns through the no-flash launcher on lane 1 without a dump and gets receipt 2, and the chain script belongs to the
relevant tree of every no-flash selection (an uncommitted edit stops it; an exp_25-only selection is unaffected).
Part 2 drives the chain itself with a stub runner and a stub comparator: (1) the replay -> (2) the check -> (3) the
24 only on an exact 0; every other outcome stops before (3); DRY_RUN never runs the check.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import json
import os
import subprocess

import pytest

import test_depth_canon_eval_run as R

C = R.C
EXP36REL = os.path.join("worklog", "worklog_yixun", "exp_36_yaw_c4_spread_claude")
CHAIN_REL = os.path.join(EXP36REL, "yaw_c4_spread_chain.sh")
REPLAY = "exp36_YAW_C4_S40000_K8_s42_th90"
CELLS24 = [f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)]


@pytest.fixture(scope="module")
def clone36(tmp_path_factory):
    """R.clone's throwaway clone, its sparse checkout widened to the exp_36 folder (the chain script overlaid too)."""
    root = tmp_path_factory.mktemp("exp36run") / "repo"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", R.REPO, str(root)], check=True)
    R._git(root, "sparse-checkout", "set", "--cone", "src", "data/AR", "worklog/worklog_yixun/exp_07_fa_scratch_claude",
           "worklog/worklog_yixun/exp_15_yaw_aug_claude", R.EXPREL, "overlays", R.EXP35REL, EXP36REL)
    R._git(root, "checkout", "-q", "--detach", R._git(R.REPO, "rev-parse", "HEAD"))
    rels = [os.path.join(R.EXPREL, f) for f in R.KIT] + list(R.KIT35) + [CHAIN_REL]
    for rel in rels:                                       # the working-tree versions, like R.clone's kit
        src = os.path.join(R.REPO, rel)
        if os.path.exists(src):
            with open(src, "rb") as a, open(root / rel, "wb") as b:
                b.write(a.read())
    R._git(root, "add", "-A")
    R._git(root, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "--allow-empty", "-m", "kit")
    return root


@pytest.fixture()
def harness36(tmp_path, clone36):
    """R.harness's body for clone36: a harness dir (stub bins, stub evaluator) and its N = 8 manifest whose
    checkpoints are fake files (+ their NAS / CANON manifests)."""
    h = tmp_path / "h"
    (h / "bin").mkdir(parents=True)
    (h / ".depth_canon_eval_harness").write_text("")
    for name, body in (("findmnt", R.STUB_FINDMNT), ("nvidia-smi", R.STUB_NVSMI), ("df", R.STUB_DF)):
        (h / "bin" / name).write_text(body)
        os.chmod(h / "bin" / name, 0o755)
    (h / "stub_eval.py").write_text(R.STUB_EVAL.replace("@TESTMOD@", R.TESTMOD))
    m = json.loads(json.dumps(R.MC.build_manifest()))
    for split in m["splits"].values():
        split["expected_stream_count"] = 8
    lines = []
    for arm in ("P1", "YAW", "BF"):
        ck = m["arms"][arm]["ckpt"]
        ck["path"] = str(h / "nas" / arm / "epoch=8-step=40000.ckpt")
        ck["sha256"] = R._fake_ckpt(ck["path"], f"fake {arm}".encode())
        ck["nas_manifest"] = str(h / "nas" / "MANIFEST.sha256")
        lines.append(f"{ck['sha256']}  {ck['nas_manifest_key']}\n")
    (h / "nas" / "MANIFEST.sha256").write_text("".join(lines))
    ns = m["arms"]["CANON"]["ckpt"]["run_ns"] = str(h / "nas" / "canon")
    (h / "canon_ckpts.sha256").write_text("".join(
        f"{R._fake_ckpt(f'{ns}/checkpoints/epoch={s // 4551}-step={s}.ckpt', str(s).encode())}  "
        f"{ns}/checkpoints/epoch={s // 4551}-step={s}.ckpt\n" for s in range(2500, 40001, 2500)))
    for c in m["cells"]:
        c["expected_stream_count"] = 8
        c["eval_args"][c["eval_args"].index("--expected-stream-count") + 1] = "8"
    (h / "cells.json").write_text(json.dumps(m))
    return {"dir": h, "manifest": m, "head": R._git(clone36, "rev-parse", "HEAD")}


# --------------------------------------------------------------------------- #
# part 1: the real runner, YAW-C4S cells
# --------------------------------------------------------------------------- #
def test_the_runner_dry_run_spawns_the_replay_through_the_no_flash_launcher(clone36, harness36):
    rc, out = R.drive(clone36, harness36, "--only", REPLAY, "--no-dump-blocks", "YAW-C4S", "--lanes", "1",
                      env={"DRY_RUN": "1"})
    assert rc == 0 and "no-flash pre-flight OK" in out and "(1 exp_35 cell(s))" in out, out
    row = R.manifest_row(harness36["manifest"], REPLAY)
    link = f"{harness36['dir']}/farm/YAW/epoch=8-step=40000.ckpt"
    assert R._dry_argv(out) == {REPLAY: row["eval_args"] + ["--ckpt-path", link]}           # no dump flag
    assert "spawned through overlays/no_flash/exp35_no_flash_launch.py" in out and "lanes 1" in out
    assert not (harness36["dir"] / "calls.log").exists()


def test_the_runner_runs_yaw_c4s_cells_on_lane_1_without_dumps_and_receipt_2(clone36, harness36):
    cells = [REPLAY, CELLS24[0], CELLS24[-1]]
    rc, out = R.drive(clone36, harness36, "--only", ",".join(cells), "--no-dump-blocks", "YAW-C4S", "--lanes", "1")
    assert rc == 0 and "CAMPAIGN SELECTION COMPLETE" in out and "3/3" in out, out
    calls = R._calls(harness36)
    assert sorted(R._ran(harness36)) == sorted(cells) and {c["cvd"] for c in calls} == {"1"}
    assert all(c["no_flash"] is True and "--dump-scored-waveforms" not in c["argv"] for c in calls)
    for cell in cells:
        e = R._entry(clone36, harness36, cell)
        rec = json.load(open(e["receipt"]))
        assert (rec["schema_version"], rec["attention_backend"], rec["dump"], rec["lane"]) == (2, "sdpa", False, 1)
        assert not os.path.exists(e["dump_dir"])
        assert R._verify(clone36, harness36, cell, expect_dump=False,
                         backend_target=str(harness36["dir"] / "stub_eval.py")) == [], cell


def test_the_chain_script_is_in_the_relevant_tree_of_a_no_flash_selection(clone36, harness36):
    path = clone36 / CHAIN_REL
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"# drift\n")
        rc, out = R.drive(clone36, harness36, "--only", REPLAY, "--no-dump-blocks", "YAW-C4S", "--lanes", "1",
                          env={"DRY_RUN": "1"})
        assert rc == 2 and "NOT COMMITTED" in out and CHAIN_REL in out, out
        rc, out = R.drive(clone36, harness36, "--only", "exp25_P1_T_S40000_K1_s42", env={"DRY_RUN": "1"})
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
with open(os.path.join(H, "calls.txt"), "a") as f:
    f.write("COLLECT\nenv CVD=%s\n" % os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + "\n".join(a) + "\nEND\n")
rc = int(os.environ.get("STUB_CHECK_RC", "0"))
with open(a[a.index("--verdict-out") + 1], "x") as f:
    json.dump({"result": {0: "equal", 1: "mismatch"}.get(rc, "refused")}, f)
print("EXP36 REPLAY " + {0: "EQUAL", 1: "MISMATCH"}.get(rc, "REFUSED"))
sys.exit(rc)
"""


@pytest.fixture()
def ch(tmp_path, clone36):
    h = tmp_path / "ch"
    h.mkdir()
    (h / ".exp36_chain_harness").write_text("")
    (h / "stub_runner.sh").write_text(STUB_RUNNER)
    (h / "stub_collect.py").write_text(STUB_COLLECT)
    return {"dir": h, "head": R._git(clone36, "rev-parse", "HEAD")}


def chain(clone, ch, env=None):
    h = ch["dir"]
    e = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "EXPECT_HEAD", "ALLOW_SHARED_GPU",
                                                            "EXP35_APPROVED_COTENANTS", "DEPTH_CANON_EVAL_HARNESS")}
    e.update(EXP36_CHAIN_HARNESS=str(h), EXP36_CHAIN_RUNNER=str(h / "stub_runner.sh"),
             EXP36_CHAIN_COLLECT=str(h / "stub_collect.py"), EXPECT_HEAD=ch["head"])
    e.update(env or {})
    p = subprocess.run(["bash", str(clone / CHAIN_REL)], env=e, capture_output=True, text=True, timeout=120, cwd=str(h))
    return p.returncode, p.stdout + p.stderr


def calls(ch):
    path = ch["dir"] / "calls.txt"
    blocks = path.read_text().split("END\n")[:-1] if path.exists() else []
    return [(b.splitlines()[0], b.splitlines()[1], b.splitlines()[2:]) for b in blocks]


RUN_ENV = "env DRY_RUN={} EXPECT_HEAD={} ALLOW_SHARED_GPU=0 COT=unset"


def test_the_chain_runs_replay_check_then_the_24(clone36, ch):
    rc, out = chain(clone36, ch, env={"ALLOW_SHARED_GPU": "0"})
    assert rc == 0 and "CHAIN COMPLETE" in out, out
    (k1, e1, a1), (k2, e2, a2), (k3, e3, a3) = calls(ch)
    assert (k1, k2, k3) == ("RUNNER", "COLLECT", "RUNNER")
    assert e1 == e3 == RUN_ENV.format(0, ch["head"]) and e2 == "env CVD="
    assert a1 == ["--only", REPLAY, "--no-dump-blocks", "YAW-C4S", "--lanes", "1"]
    verdict = [p for p in os.listdir(ch["dir"]) if p.endswith("_replay_verdict.json")]
    assert a2 == ["--exp36-replay-check", "--pin", ch["head"], "--plan-json", str(ch["dir"] / "plan_1.json"),
                  "--verdict-out", str(ch["dir"] / verdict[0])] and len(verdict) == 1
    assert a3 == ["--only", ",".join(CELLS24), "--no-dump-blocks", "YAW-C4S", "--lanes", "1"]
    assert f"--exp36-publish --pin {ch['head']} --exp36-verdict {ch['dir'] / verdict[0]}" in out
    logs = [p for p in os.listdir(ch["dir"]) if p.endswith("_chain.log")]
    assert len(logs) == 1 and "CHAIN COMPLETE" in (ch["dir"] / logs[0]).read_text()


@pytest.mark.parametrize("env, rc_want, needle, kinds", [
    ({"STUB_CHECK_RC": "1"}, 4, "REPLAY MISMATCH", ["RUNNER", "COLLECT"]),
    ({"STUB_CHECK_RC": "2"}, 5, "REPLAY CHECK REFUSED", ["RUNNER", "COLLECT"]),
    ({"STUB_CHECK_RC": "3"}, 5, "REPLAY CHECK REFUSED", ["RUNNER", "COLLECT"]),
    ({"STUB_RUNNER_RC_1": "2"}, 3, "the replay run exited 2", ["RUNNER"]),
    ({"STUB_RUNNER_RC_1": "1"}, 3, "the replay run exited 1", ["RUNNER"]),
    ({"STUB_NO_PLAN": "1"}, 3, "no single plan", ["RUNNER"]),
    ({"STUB_RUNNER_RC_2": "1"}, 6, "re-run this chain to resume", ["RUNNER", "COLLECT", "RUNNER"]),
    ({"STUB_RUNNER_RC_2": "4"}, 6, "the 24-cell run exited 4", ["RUNNER", "COLLECT", "RUNNER"]),
])
def test_every_other_outcome_is_a_hard_stop_before_the_24(clone36, ch, env, rc_want, needle, kinds):
    rc, out = chain(clone36, ch, env=env)
    assert rc == rc_want and needle in out and "CHAIN COMPLETE" not in out, out
    assert [k for k, _, _ in calls(ch)] == kinds


def test_dry_run_plans_both_runner_steps_and_never_runs_the_check(clone36, ch):
    rc, out = chain(clone36, ch, env={"DRY_RUN": "1"})
    assert rc == 0 and "DRY_RUN complete" in out and "step 2/3 (DRY_RUN: not run)" in out, out
    assert [(k, e) for k, e, _ in calls(ch)] == [("RUNNER", RUN_ENV.format(1, ch["head"]))] * 2
    assert not [p for p in os.listdir(ch["dir"]) if p.endswith("_replay_verdict.json")]


@pytest.mark.parametrize("env, needle", [
    ({"EXPECT_HEAD": "0" * 40}, "HEAD PIN"), ({"EXPECT_HEAD": ""}, "EXPECT_HEAD"), ({"DRY_RUN": "2"}, "DRY_RUN"),
    ({"ALLOW_SHARED_GPU": "1"}, "ALLOW_SHARED_GPU"), ({"EXP35_APPROVED_COTENANTS": "/tmp/x"}, "co-tenant"),
    ({"EXP36_CHAIN_HARNESS": "/var/tmp"}, "HARNESS"), ({"EXP36_CHAIN_RUNNER": "/bin/true"}, "HARNESS")])
def test_the_chain_gates_refuse_before_any_step(clone36, ch, env, needle):
    rc, out = chain(clone36, ch, env=env)
    assert rc == 2 and needle in out and not calls(ch), out


def test_outside_the_harness_a_test_variable_is_refused(clone36, ch):
    e = {"EXP36_CHAIN_HARNESS": "", "DEPTH_CANON_EVAL_HARNESS": str(ch["dir"])}
    rc, out = chain(clone36, ch, env=e)
    assert rc == 2 and "test-harness variable" in out and not calls(ch), out


def test_an_uncommitted_chain_is_refused(clone36, ch):
    path = clone36 / CHAIN_REL
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# drift\n")
        rc, out = chain(clone36, ch)
        assert rc == 2 and "not committed" in out and not calls(ch), out
    finally:
        path.write_bytes(original)
