"""exp_38 B6/B7 — launch paths: the A6000 launcher (DRY_RUN + refusals) and the neuronic sbatch
(EXP38_SBATCH_DRYRUN), both thin wrappers that exec exp38_supervise.py, and the exactly-once
neuronic submitter + in-job attempt claim (fake scheduler; Codex code r1 HIGH 6).

Nothing here touches a GPU, the NAS run namespace or a scheduler: DRY_RUN spawns nothing, the
sbatch test hook refuses under a real allocation, the submitter is driven with a fake runner.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
LAUNCH = os.path.join(EXP, "exp38_train_local.sh")
SBATCH = os.path.join(EXP, "exp38_train.sbatch")
ENV_PREFIX = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))


def _load(name):
    sys.path.insert(0, EXP)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sub = _load("exp38_submit_neuronic")
reg = sub.reg

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(EXP, "data_efficiency_subset_audit.json")),
                                reason="the subset-audit artifact is produced at ladder rung 3")


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EXP38_", "MODE", "RUN", "RESUME"))}
    env["PATH"] = os.path.join(ENV_PREFIX, "bin") + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _launch(**kw):
    return subprocess.run(["bash", LAUNCH], cwd=REPO, env=_env(**kw), capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("run", ["de_yaw_f025", "de_bf_f075", "de_p1_f025"])
def test_dry_run_prints_the_registered_commands_and_spawns_nothing(run, tmp_path):
    r = _launch(MODE="DRY_RUN", RUN=run, EXP38_DRY_SAVE_ROOT=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert f"DRY_RUN OK {run} on a6000 (nothing spawned)" in r.stdout
    full = [l for l in r.stdout.split("\n") if l.startswith("FULL (cwd = the closure snapshot):")][0]
    want = " ".join(reg.build_train_argv(run, "a6000", save_root=str(tmp_path)))
    assert full.endswith(" " + want) and "--ckpt-path" not in full
    smoke = [l for l in r.stdout.split("\n") if l.startswith("SMOKE_DDP (cwd = the closure snapshot):")][0]
    assert "--max-steps 30" in smoke and "--num-gpus 2" in smoke and "--sync-batchnorm true" in smoke
    assert "E0 receipt (FULL needs it):" in r.stdout and not os.listdir(tmp_path)


@pytest.mark.parametrize("kw,needle", [
    ({"MODE": "LAUNCH", "RUN": "de_yaw_f025"}, "MODE must be"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025_r8x8"}, "RUN must be a registered 2x32"),
    ({"MODE": "DRY_RUN", "RUN": "dc_van_f025"}, "RUN must be a registered 2x32"),
    ({"MODE": "DRY_RUN", "RUN": "de_p1_f050"}, "not a registered exp_38 run id"),
    ({"MODE": "FULL", "RUN": "de_yaw_f025", "RESUME_STEP": "2500"}, "there is no resume"),
    ({"MODE": "SMOKE", "RUN": "de_yaw_f025", "EXP38_LOCKDIR": "/tmp/x"}, "DRY_RUN test hooks"),
    ({"MODE": "FULL", "RUN": "de_yaw_f025", "EXP38_DRY_SAVE_ROOT": "/tmp/x"}, "DRY_RUN test hooks"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025", "GPU_POLICY": "maybe"}, "GPU_POLICY"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025", "ATTN_BACKEND": "xformers"}, "ATTN_BACKEND"),
    ({"MODE": "FULL", "RUN": "de_yaw_f025"}, "runs only from the campaign tree"),
])
def test_launcher_refusals(kw, needle):
    r = _launch(**kw)
    assert r.returncode == 2 and needle in r.stderr, (r.returncode, r.stderr)


def test_dry_run_refuses_a_non_fresh_run_dir(tmp_path):
    d = tmp_path / "de_bf_f025"
    d.mkdir()
    (d / "epoch=2-step=2500.ckpt").write_bytes(b"x")
    r = _launch(MODE="DRY_RUN", RUN="de_bf_f025", EXP38_DRY_SAVE_ROOT=str(tmp_path))
    assert r.returncode == 2 and "epoch=2-step=2500.ckpt" in r.stderr and "for state 0" in r.stderr


def test_launcher_sanitises_the_environment_before_the_supervisor(tmp_path):
    r = _launch(MODE="DRY_RUN", RUN="de_yaw_f025", EXP38_DRY_SAVE_ROOT=str(tmp_path), PYTHONOPTIMIZE="1",
                PYTHONPATH="/nowhere", PYTHONNOUSERSITE="0")
    assert r.returncode == 0, r.stderr                          # unset / exported by the wrapper


def test_launcher_body_is_one_function_called_last():
    lines = open(LAUNCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"] and "main() {" in lines
    lines = open(SBATCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"]


@pytest.mark.parametrize("run,mode", [("de_yaw_f025", "FULL"), ("de_bf_f050", "SMOKE"), ("de_bf_f025_r8x8", "FULL")])
def test_sbatch_dry_run_prints_the_torchrun_lines(run, mode, tmp_path):
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode, EXPECT_SHA="a" * 40,
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION,
               SAVE_DIR=str(tmp_path / run), ALLOW_NONMATCHED_RUNG=1 if run.endswith("_r8x8") else 0)
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    n = 8 if run.endswith("_r8x8") else 2
    full = [l for l in r.stdout.split("\n") if l.startswith("FULL (cwd = the closure snapshot):")][0]
    assert f"-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={n} train.py" in full
    assert full.endswith(" ".join(reg.build_train_argv(run, "neuronic", allow_nonmatched=n == 8)))
    smoke = [l for l in r.stdout.split("\n") if l.startswith("SMOKE (cwd = the closure snapshot):")][0]
    assert "--max-steps 30" in smoke and f"/exp38_data_efficiency/{run}_smoke" in smoke


def test_sbatch_refuses_unregistered_runs_and_an_unallowed_rung():
    base = dict(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, MODE="FULL", EXPECT_SHA="a" * 40,
                EXP38_TOKEN="0" * 32, SAVE_DIR="/x")
    for run, needle in (("de_p1_f050", "not a registered"), ("de_p1_f025_r8x8", "not a registered"),
                        ("de_bf_f025_r8x8", "ALLOW_NONMATCHED_RUNG")):
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
    monkeypatch.setattr(sub, "SUBLOG", "worklog/worklog_yixun/exp_38_data_efficiency_claude/.test_submissions.log")
    yield
    p = os.path.join(reg.REPO, sub.SUBLOG)
    if os.path.exists(p):
        os.remove(p)


def _submit(tmp_path, run="de_yaw_f025", mode="FULL", fake=None, **kw):
    return sub.submit(run, mode, "3-00:00:00", "flash", reg.FLASH_VERSION, run.endswith("_r8x8"), False,
                      root=str(tmp_path), runner=fake or FakeScheduler(), **kw)


@pytest.mark.parametrize("mode", ["FULL", "SMOKE"])
def test_submit_is_exactly_once_per_run_and_mode(tmp_path, no_preflight, mode):
    fake = FakeScheduler()
    out = _submit(tmp_path, mode=mode, fake=fake)
    assert out["job_id"] == "4242" and len(fake.calls) == 1
    argv = fake.calls[0]
    assert f"--job-name=exp38-de_yaw_f025-{mode.lower()}-{out['token'][:8]}" in argv and "--gres=gpu:l40:2" in argv
    assert "--no-requeue" in argv and any(a.startswith(f"--export=ALL,RUN=de_yaw_f025,MODE={mode}") for a in argv)
    od, intent = sub.find_intent("de_yaw_f025", mode, root=str(tmp_path))
    assert intent["state"] == "submitted" and intent["job_id"] == "4242" and set(intent) == sub.INTENT_KEYS
    assert intent["save_dir"] == str(tmp_path / ("de_yaw_f025" if mode == "FULL" else "de_yaw_f025_smoke"))
    with pytest.raises(reg.GateError):                          # the owner record refuses a second submission
        _submit(tmp_path, mode=mode, fake=fake)
    assert len(fake.calls) == 1


def _claim(tmp_path, monkeypatch, out, run="de_yaw_f025", mode="FULL", **over):
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
                                  {"jid": "999"}, {"jname": "exp38-de_yaw_f025-full-deadbeef"}])
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
        _submit(tmp_path, run="de_bf_f050", fake=FakeScheduler(fail={"sbatch"}))
    od, intent = sub.find_intent("de_bf_f050", "FULL", root=str(tmp_path))
    assert intent["state"] == "submitting" and intent["job_id"] is None
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_NAME", intent["job_name"])
    sub.claim_attempt("de_bf_f050", "FULL", intent["token"], "a" * 40, "flash", intent["save_dir"],
                      reg.FLASH_VERSION, False, root=str(tmp_path))


def test_reconcile_resolves_only_an_unambiguous_state(tmp_path, no_preflight):
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="de_bf_f050", fake=FakeScheduler(fail={"sbatch"}))
    _, intent = sub.find_intent("de_bf_f050", "FULL", root=str(tmp_path))
    name = intent["job_name"]
    with pytest.raises(sub.SubmitError):                       # zero jobs: refuse, never guess
        sub.reconcile("de_bf_f050", "FULL", root=str(tmp_path), runner=FakeScheduler())
    two = FakeScheduler(squeue="777|PENDING\n778|PENDING\n")
    with pytest.raises(sub.SubmitError):                       # two jobs carry the token
        sub.reconcile("de_bf_f050", "FULL", root=str(tmp_path), runner=two)
    fake = FakeScheduler(squeue="777|PENDING\n", sacct=f"777|{name}|PENDING\n")
    out = sub.reconcile("de_bf_f050", "FULL", root=str(tmp_path), runner=fake)
    assert out["intent"]["state"] == "submitted" and out["intent"]["job_id"] == "777" and out["intent"]["reconciled"]
    with pytest.raises(sub.SubmitError):                       # recorded 777 but the scheduler shows 778
        sub.reconcile("de_bf_f050", "FULL", root=str(tmp_path), runner=FakeScheduler(squeue="778|RUNNING\n"))
    with pytest.raises(sub.SubmitError):                       # live job: never abandon
        sub.abandon("de_bf_f050", "FULL", root=str(tmp_path), runner=fake)
    done = FakeScheduler(sacct=f"777|{name}|FAILED\n")
    (tmp_path / "de_bf_f050").mkdir()
    moved = sub.abandon("de_bf_f050", "FULL", root=str(tmp_path), runner=done)
    assert not os.path.exists(tmp_path / "de_bf_f050") and os.path.isdir(moved["run_dir_moved_to"])
    assert os.path.isdir(moved["owner_moved_to"]) and "/abandoned/" in moved["owner_moved_to"]
    with pytest.raises(sub.SubmitError):                       # an abandoned record is never found again
        sub.find_intent("de_bf_f050", "FULL", root=str(tmp_path))
    _submit(tmp_path, run="de_bf_f050")                        # ... and the run may be submitted afresh


def test_scheduler_queries_fail_closed(tmp_path):
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(fail={"sacct"}))
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(sacct="1|x|OUT_OF_ME+\n"))   # truncated state


def test_abandon_refuses_when_the_recorded_job_is_invisible(tmp_path, no_preflight):
    _submit(tmp_path, run="de_yaw_f050", fake=FakeScheduler(sbatch="99"))
    with pytest.raises(sub.SubmitError):
        sub.abandon("de_yaw_f050", "FULL", root=str(tmp_path), runner=FakeScheduler())


def test_run_dirs_owner_dirs_and_resources():
    assert sub.run_dir("de_bf_f025", "SMOKE", "/r") == "/r/de_bf_f025_smoke"
    assert sub.run_dir("de_bf_f025", "FULL", "/r") == "/r/de_bf_f025"
    assert sub.owner_dir("de_bf_f025", "SMOKE", "/r") == "/r/.exp38_owners/de_bf_f025__SMOKE"
    assert sub.resources("de_bf_f025_r8x8") == {"ngpu": 8, "cpus": 64, "mem": "108G"}
    with pytest.raises(reg.GateError):
        sub.run_dir("de_p1_f050", "FULL", "/r")
    with pytest.raises(sub.SubmitError):
        sub.submit("de_bf_f025", "FULL", "forever", "flash", "", False, True)
