"""exp_45 (forked from test_exp38_launchers.py, B6/B7) — launch paths: the neuronic sbatch
(EXP38_SBATCH_DRYRUN), a thin wrapper that execs exp45_supervise.py, and the exactly-once neuronic
submitter + in-job attempt claim (fake scheduler; exp_38 Codex code r1 HIGH 6).

Fork edit (exp_45 commit B): exp_38's A6000 local launcher exp38_train_local.sh is not forked (plan r3
§3.2 copy list; D2: every training runs on neuronic), so its DRY_RUN / refusal tests are dropped; the
untracked-importable check runs through the sbatch hook instead.

Nothing here touches a GPU, the NAS run namespace or a scheduler: the sbatch test hook refuses under a
real allocation, the submitter is driven with a fake runner.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
EXP38 = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")   # read-only reuse
SBATCH = os.path.join(EXP, "exp45_train.sbatch")
ENV_PREFIX = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))


def _load(name):
    sys.path.insert(0, EXP)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sub = _load("exp45_submit_neuronic")
reg = sub.reg

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(EXP38, "data_efficiency_subset_audit.json")),
                                reason="the subset-audit artifact is produced at ladder rung 3")


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EXP38_", "MODE", "RUN", "RESUME", "SLURM_")
                                                                       + reg.LAUNCHER_PREFIXES) and k not in reg.LAUNCHER_VARS}
    env["PATH"] = os.path.join(ENV_PREFIX, "bin") + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _sbatch_dry(run="exp45_RFf025_l40_2x32", mode="FULL", save_dir="/x", **kw):
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode, EXPECT_SHA="a" * 40,
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION, SAVE_DIR=save_dir, **kw)
    env.pop("SLURM_JOB_ID", None)
    return subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def test_launcher_body_is_one_function_called_last():
    lines = open(SBATCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"] and "main() {" in lines


@pytest.mark.parametrize("run,mode", [("exp45_RFf025_l40_2x32", "FULL"), ("exp45_RFf050_l40_2x32", "SMOKE"), ("exp45_RFf075_l40_2x32", "FULL")])
def test_sbatch_dry_run_prints_the_torchrun_lines(run, mode, tmp_path):
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode, EXPECT_SHA="a" * 40,
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION,
               SAVE_DIR=str(tmp_path / run), ALLOW_NONMATCHED_RUNG=0)
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    n = 2
    full = [l for l in r.stdout.split("\n") if l.startswith("FULL (cwd = the closure snapshot):")][0]
    assert f"-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={n} train.py" in full
    assert full.endswith(" ".join(reg.build_train_argv(run, "neuronic")))
    smoke = [l for l in r.stdout.split("\n") if l.startswith("SMOKE (cwd = the closure snapshot):")][0]
    assert "--max-steps 30" in smoke and f"/exp45_rf_data_curve/{run}_smoke" in smoke


def test_sbatch_refuses_unregistered_runs_and_an_unallowed_rung():
    base = dict(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, MODE="FULL", EXPECT_SHA="a" * 40,
                EXP38_TOKEN="0" * 32, SAVE_DIR="/x")
    for run, needle in (("de_yaw_f025", "not a registered"), ("exp45_RFf025_l40_8x8", "not a registered"),
                        ("exp45_RFf100_l40_2x32", "not a registered"), ("exp45_RFf025_l40_2x32_r8x8", "not a registered")):
        env = _env(RUN=run, **base)
        env.pop("SLURM_JOB_ID", None)
        r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 2 and needle in r.stderr, (run, r.stderr)


def test_sbatch_hook_is_refused_under_a_real_allocation():
    env = _env(EXP38_SBATCH_DRYRUN=1, SLURM_JOB_ID=123)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "test hook" in r.stderr


# ------------------------------------------------------------------ the submitter
class FakeScheduler:
    def __init__(self, squeue="", sacct="", sbatch="4242;cluster", fail=None):
        self.out = {"squeue": squeue, "sacct": sacct, "sbatch": sbatch}
        self.fail = fail or set()
        self.calls = []

    def __call__(self, argv, capture_output=True, text=True):
        self.calls.append(argv)
        tool = os.path.basename(argv[0])
        return types.SimpleNamespace(returncode=1 if tool in self.fail else 0, stdout=self.out.get(tool, ""),
                                     stderr="boom" if tool in self.fail else "")


@pytest.fixture
def no_preflight(monkeypatch):
    monkeypatch.setattr(sub, "preflight", lambda run, mode, allow, dry, backend: "a" * 40)
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)              # a fake, roomy filesystem
    monkeypatch.setattr(sub, "SUBLOG", "worklog/worklog_yixun/exp_45_rf_data_curve_claude/.test_submissions.log")
    yield
    p = os.path.join(reg.REPO, sub.SUBLOG)
    if os.path.exists(p):
        os.remove(p)


def _submit(tmp_path, run="exp45_RFf025_l40_2x32", mode="FULL", fake=None, **kw):
    return sub.submit(run, mode, "3-00:00:00", "flash", reg.FLASH_VERSION, run.endswith("_r8x8"), False,
                      root=str(tmp_path), runner=fake or FakeScheduler(), **kw)


@pytest.mark.parametrize("mode", ["FULL", "SMOKE"])
def test_submit_is_exactly_once_per_run_and_mode(tmp_path, no_preflight, mode):
    fake = FakeScheduler()
    out = _submit(tmp_path, mode=mode, fake=fake)
    assert out["job_id"] == "4242" and len(fake.calls) == 1
    argv = fake.calls[0]
    assert f"--job-name=exp38-exp45_RFf025_l40_2x32-{mode.lower()}-{out['token'][:8]}" in argv and "--gres=gpu:l40:2" in argv
    assert "--no-requeue" in argv and any(a.startswith(f"--export=ALL,RUN=exp45_RFf025_l40_2x32,MODE={mode}") for a in argv)
    od, intent = sub.find_intent("exp45_RFf025_l40_2x32", mode, root=str(tmp_path))
    assert intent["state"] == "submitted" and intent["job_id"] == "4242" and set(intent) == sub.INTENT_KEYS
    assert intent["save_dir"] == str(tmp_path / ("exp45_RFf025_l40_2x32" if mode == "FULL" else "exp45_RFf025_l40_2x32_smoke"))
    held = json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"] if mode == "FULL" else {}
    assert (out["storage"] is None) == (mode == "SMOKE") and sorted(held) == (["exp45_RFf025_l40_2x32"] if mode == "FULL" else [])
    assert mode == "SMOKE" or (held["exp45_RFf025_l40_2x32"]["token"] == out["token"]
                               and held["exp45_RFf025_l40_2x32"]["reserved_bytes"] == reg.RUN_RESERVE_BYTES)
    with pytest.raises(reg.GateError):                          # the owner record refuses a second submission
        _submit(tmp_path, mode=mode, fake=fake)
    assert len(fake.calls) == 1


def _claim(tmp_path, monkeypatch, out, run="exp45_RFf025_l40_2x32", mode="FULL", **over):
    od, intent = sub.find_intent(run, mode, root=str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", str(over.pop("jid", intent["job_id"] or "4242")))
    monkeypatch.setenv("SLURM_JOB_NAME", over.pop("jname", intent["job_name"]))
    args = dict(token=out["token"], head="a" * 40, backend="flash", sd=intent["save_dir"],
                flash_version=reg.FLASH_VERSION, allow_nonmatched=False)
    args.update(over)
    return sub.claim_attempt(run, mode, args["token"], args["head"], args["backend"], args["sd"],
                             args["flash_version"], args["allow_nonmatched"], root=str(tmp_path))


@pytest.mark.parametrize("over", [{"token": "1" * 32}, {"head": "b" * 40}, {"backend": "sdpa"},
                                  {"flash_version": "2.8.0"}, {"allow_nonmatched": True}, {"sd": "/elsewhere"},
                                  {"jid": "999"}, {"jname": "exp38-exp45_RFf025_l40_2x32-full-deadbeef"}])
def test_the_job_must_match_its_owner_intent_exactly(tmp_path, monkeypatch, no_preflight, over):
    out = _submit(tmp_path)
    with pytest.raises(reg.GateError):
        _claim(tmp_path, monkeypatch, out, **over)


def test_one_attempt_per_owner_and_its_record(tmp_path, monkeypatch, no_preflight):
    out = _submit(tmp_path, mode="SMOKE")
    path = _claim(tmp_path, monkeypatch, out, mode="SMOKE")
    assert json.load(open(path))["state"] == "running"
    with pytest.raises(reg.GateError):                          # a duplicate / requeued start
        _claim(tmp_path, monkeypatch, out, mode="SMOKE")
    sub.finish_attempt(path, 5)
    rec = json.load(open(path))
    assert rec["state"] == "finished" and rec["rc"] == 5


def test_a_job_started_before_the_id_was_recorded_may_claim(tmp_path, monkeypatch, no_preflight):
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="exp45_RFf050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    od, intent = sub.find_intent("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path))
    assert intent["state"] == "submitting" and intent["job_id"] is None
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_NAME", intent["job_name"])
    sub.claim_attempt("exp45_RFf050_l40_2x32", "FULL", intent["token"], "a" * 40, "flash", intent["save_dir"],
                      reg.FLASH_VERSION, False, root=str(tmp_path))


def test_reconcile_resolves_only_an_unambiguous_state(tmp_path, no_preflight):
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="exp45_RFf050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    _, intent = sub.find_intent("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path))
    name = intent["job_name"]
    with pytest.raises(sub.SubmitError):                       # zero jobs: refuse, never guess
        sub.reconcile("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler())
    two = FakeScheduler(squeue="777|PENDING\n778|PENDING\n")
    with pytest.raises(sub.SubmitError):                       # two jobs carry the token
        sub.reconcile("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=two)
    fake = FakeScheduler(squeue="777|PENDING\n", sacct=f"777|{name}|PENDING\n")
    out = sub.reconcile("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=fake)
    assert out["intent"]["state"] == "submitted" and out["intent"]["job_id"] == "777" and out["intent"]["reconciled"]
    with pytest.raises(sub.SubmitError):                       # recorded 777 but the scheduler shows 778
        sub.reconcile("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler(squeue="778|RUNNING\n"))
    with pytest.raises(sub.SubmitError):                       # live job: never abandon
        sub.abandon("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=fake)
    done = FakeScheduler(sacct=f"777|{name}|FAILED\n")
    (tmp_path / "exp45_RFf050_l40_2x32").mkdir()
    assert "exp45_RFf050_l40_2x32" in json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"]   # held while owned
    moved = sub.abandon("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=done)
    assert moved["storage_released"] and json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"] == {}
    assert not os.path.exists(tmp_path / "exp45_RFf050_l40_2x32") and os.path.isdir(moved["run_dir_moved_to"])
    assert os.path.isdir(moved["owner_moved_to"]) and "/abandoned/" in moved["owner_moved_to"]
    with pytest.raises(sub.SubmitError):                       # an abandoned record is never found again
        sub.find_intent("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path))
    _submit(tmp_path, run="exp45_RFf050_l40_2x32")                        # ... and the run may be submitted afresh


def test_a_full_submission_reserves_before_any_owner_record_and_refuses_below_the_floor(tmp_path, monkeypatch, no_preflight):
    """plan r3 §3.2: the reservation precedes sbatch AND the owner intent; below the floor nothing is written."""
    monkeypatch.setattr(reg, "free_bytes", lambda path: reg.STORAGE_FLOOR_BYTES + reg.RUN_RESERVE_BYTES - 1)
    fake = FakeScheduler()
    with pytest.raises(reg.GateError, match="floor"):
        _submit(tmp_path, fake=fake)
    assert fake.calls == [] and not os.path.exists(tmp_path / sub.OWNERS)
    assert not os.path.exists(tmp_path / reg.STORAGE_LEDGER)
    monkeypatch.setattr(reg, "free_bytes", lambda path: reg.STORAGE_FLOOR_BYTES + reg.RUN_RESERVE_BYTES)
    assert _submit(tmp_path, fake=fake)["storage"]["reserved_bytes"] == reg.RUN_RESERVE_BYTES
    assert _submit(tmp_path, mode="SMOKE", fake=fake)["storage"] is None          # a SMOKE writes no checkpoint
    with pytest.raises(reg.GateError, match="floor"):                             # the next candidate: no room
        _submit(tmp_path, run="exp45_RFf050_l40_2x32", fake=fake)
    assert not os.path.exists(os.path.join(sub.owner_dir("exp45_RFf050_l40_2x32", "FULL", str(tmp_path))))


def test_scheduler_queries_fail_closed(tmp_path):
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(fail={"sacct"}))
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(sacct="1|x|OUT_OF_ME+\n"))   # truncated state


def test_abandon_refuses_when_the_recorded_job_is_invisible(tmp_path, no_preflight):
    _submit(tmp_path, run="exp45_RFf050_l40_2x32", fake=FakeScheduler(sbatch="99"))
    with pytest.raises(sub.SubmitError):
        sub.abandon("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler())


def test_run_dirs_owner_dirs_and_resources():
    assert sub.run_dir("exp45_RFf025_l40_2x32", "SMOKE", "/r") == "/r/exp45_RFf025_l40_2x32_smoke"
    assert sub.run_dir("exp45_RFf025_l40_2x32", "FULL", "/r") == "/r/exp45_RFf025_l40_2x32"
    assert sub.owner_dir("exp45_RFf025_l40_2x32", "SMOKE", "/r") == "/r/.exp38_owners/exp45_RFf025_l40_2x32__SMOKE"
    assert sub.resources("exp45_RFf025_l40_2x32") == {"ngpu": 2, "cpus": 22, "mem": "64G"}
    with pytest.raises(reg.GateError):
        sub.run_dir("de_p1_f050", "FULL", "/r")
    with pytest.raises(sub.SubmitError):
        sub.submit("exp45_RFf025_l40_2x32", "FULL", "forever", "flash", "", False, True)


def _claim_proc(args):
    tmp, token, jid, name, sd = args
    os.environ["SLURM_JOB_ID"], os.environ["SLURM_JOB_NAME"] = jid, name
    try:
        sub.claim_attempt("exp45_RFf050_l40_2x32", "FULL", token, "a" * 40, "flash", sd, reg.FLASH_VERSION, False, root=tmp)
        return jid
    except reg.GateError:
        return None


def test_two_simultaneous_jobs_of_one_owner_cannot_both_claim(tmp_path, no_preflight):
    """Codex code r2 HIGH 2: an interrupted submission (job id unknown) and a duplicate job start at
    once: the fixed owner-scoped claim admits exactly one."""
    import multiprocessing as mp
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="exp45_RFf050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    od, intent = sub.find_intent("exp45_RFf050_l40_2x32", "FULL", root=str(tmp_path))
    args = [(str(tmp_path), intent["token"], str(1000 + i), intent["job_name"], intent["save_dir"]) for i in range(6)]
    with mp.get_context("fork").Pool(6) as pool:
        got = pool.map(_claim_proc, args)
    assert len([g for g in got if g]) == 1
    assert len([n for n in os.listdir(od) if n.startswith("attempt_")]) == 1


def test_an_untracked_importable_next_to_the_kit_refuses_every_mode(tmp_path):
    shadow = os.path.join(EXP, f"zz_exp45_shadow_test_{os.getpid()}.py")
    open(shadow, "w").write("raise SystemExit(99)\n")
    try:
        r = _sbatch_dry(save_dir=str(tmp_path / "exp45_RFf025_l40_2x32"))
        assert r.returncode == 3 and "untracked importable files in the kit folder" in r.stderr
    finally:
        os.remove(shadow)
