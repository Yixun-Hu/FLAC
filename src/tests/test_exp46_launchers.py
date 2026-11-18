"""exp_46 (forked from test_exp38_launchers.py, B6/B7) — launch paths: the neuronic sbatch
(EXP38_SBATCH_DRYRUN), a thin wrapper that execs exp46_supervise.py, and the exactly-once neuronic
submitter + in-job attempt claim (fake scheduler; exp_38 Codex code r1 HIGH 6).

Fork edit (exp_46 commit B): exp_38's A6000 local launcher exp38_train_local.sh is not forked (plan r3
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
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
EXP38 = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")   # read-only reuse
SBATCH = os.path.join(EXP, "exp46_train.sbatch")
ENV_PREFIX = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
RF050 = "exp46_C4Ff050_l40_2x32"


def _load(name):
    sys.path.insert(0, EXP)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sub = _load("exp46_submit_neuronic")
reg = sub.reg


@pytest.fixture(autouse=True)
def _exp45_ledger_stand_in(tmp_path, monkeypatch):
    """C3 (plan r3 §3.2 "Storage") reads exp_45's REAL ledger path at run time; here an absent stand-in, so the real
    exp_45 state never decides a test."""
    monkeypatch.setattr(reg, "EXP45_LEDGER", str(tmp_path / "exp45_ledger_stand_in" / ".exp45_storage.json"))

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(EXP38, "data_efficiency_subset_audit.json")),
                                reason="the subset-audit artifact is produced at ladder rung 3")


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EXP38_", "MODE", "RUN", "RESUME", "SLURM_")
                                                                       + reg.LAUNCHER_PREFIXES) and k not in reg.LAUNCHER_VARS}
    env["PATH"] = os.path.join(ENV_PREFIX, "bin") + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _sbatch_dry(run="exp46_C4Ff025_l40_2x32", mode="FULL", save_dir="/x", **kw):
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode, EXPECT_SHA="a" * 40,
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION, SAVE_DIR=save_dir, **kw)
    env.pop("SLURM_JOB_ID", None)
    return subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def test_launcher_body_is_one_function_called_last():
    lines = open(SBATCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"] and "main() {" in lines


@pytest.mark.parametrize("run,mode", [("exp46_C4Ff025_l40_2x32", "FULL"), ("exp46_C4Ff050_l40_2x32", "SMOKE"), ("exp46_C4Ff075_l40_2x32", "FULL")])
def test_sbatch_dry_run_prints_the_torchrun_lines(run, mode, tmp_path):
    """Codex code r1 finding 10: the sbatch test hook routes the dry run through an ISOLATED save root
    (EXP38_DRY_SAVE_ROOT -> --dry-save-root), so the result does not depend on the live run directories."""
    iso = tmp_path / "dry_save_root"
    iso.mkdir()
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode, EXPECT_SHA="a" * 40,
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION,
               SAVE_DIR=str(tmp_path / run), ALLOW_NONMATCHED_RUNG=0, EXP38_DRY_SAVE_ROOT=str(iso))
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    n = 2
    full = [l for l in r.stdout.split("\n") if l.startswith("FULL (cwd = the closure snapshot):")][0]
    assert f"-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={n} train.py" in full
    assert full.endswith(" ".join(reg.build_train_argv(run, "neuronic", str(iso))))
    smoke = [l for l in r.stdout.split("\n") if l.startswith("SMOKE (cwd = the closure snapshot):")][0]
    assert "--max-steps 30" in smoke and f"{iso}/{run}_smoke" in smoke
    assert f"run dir {iso}/{run}: fresh" in r.stdout
    assert reg.PLATFORMS["neuronic"]["save_root"] not in full + smoke                # nothing of the live root


def test_the_dry_run_checks_its_isolated_root_even_with_the_production_run_dir_occupied(tmp_path, monkeypatch, capsys):
    """Codex code r1 finding 10: with the production FULL directory occupied (a run's checkpoint in it), the dry run
    on an isolated save root still passes -- it checks, and prints, only the isolated root -- while the same dry run
    against the production root refuses."""
    sv = _load("exp46_supervise")
    run = "exp46_C4Ff025_l40_2x32"
    prod, iso = tmp_path / "production_save_root", tmp_path / "dry_save_root"
    (prod / run).mkdir(parents=True)
    (prod / run / reg.ckpt_name(2500, "025")).write_bytes(b"a live run's checkpoint")
    iso.mkdir()
    monkeypatch.setitem(reg.PLATFORMS["neuronic"], "save_root", str(prod))
    monkeypatch.setattr(sv, "interpreter_gate", lambda platform, expect=None: sys.executable)
    import exp38_subset_audit as sa
    monkeypatch.setattr(sa, "verify", lambda *a, **k: [])
    monkeypatch.setattr(reg, "vae_check", lambda *a, **k: {})
    monkeypatch.setattr(reg, "vit_pin_check", lambda *a, **k: {})
    monkeypatch.setattr(sv, "backend_gate", lambda *a, **k: {})
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": "c" * 64})
    a = sv.parse(["--platform", "neuronic", "--mode", "DRY_RUN", "--run", run, "--dry-save-root", str(iso)])
    assert sv.mode_dry_run(a, "neuronic") == 0
    out = capsys.readouterr().out
    assert f"run dir {iso}/{run}: fresh" in out and f"--save-dir {iso}/{run}" in out and f"{iso}/{run}_smoke" in out
    assert str(prod) not in out
    with pytest.raises(reg.GateError, match="checkpoints"):
        sv.mode_dry_run(sv.parse(["--platform", "neuronic", "--mode", "DRY_RUN", "--run", run]), "neuronic")


def test_sbatch_refuses_unregistered_runs_and_an_unallowed_rung():
    base = dict(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, MODE="FULL", EXPECT_SHA="a" * 40,
                EXP38_TOKEN="0" * 32, SAVE_DIR="/x")
    for run, needle in (("de_yaw_f025", "not a registered"), ("exp46_C4Ff025_l40_8x8", "not a registered"),
                        ("exp46_C4Ff090_l40_2x32", "not a registered"), ("exp46_C4Ff025_l40_2x32_r8x8", "not a registered")):
        env = _env(RUN=run, **base)
        env.pop("SLURM_JOB_ID", None)
        r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 2 and needle in r.stderr, (run, r.stderr)


def test_sbatch_hook_is_refused_under_a_real_allocation():
    env = _env(EXP38_SBATCH_DRYRUN=1, SLURM_JOB_ID=123)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "test hook" in r.stderr


# ------------------------------------------------------------------ the submitter
SCONTROL_ALL = "PartitionName=all AllowGroups=ALL Default=YES MaxNodes=UNLIMITED MaxTime=7-00:00:00 MinNodes=0 State=UP"


class FakeScheduler:
    def __init__(self, squeue="", sacct="", sbatch="4242;cluster", fail=None, scontrol=SCONTROL_ALL):
        self.out = {"squeue": squeue, "sacct": sacct, "sbatch": sbatch, "scontrol": scontrol}
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
    monkeypatch.setattr(sub, "SUBLOG", "worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/.test_submissions.log")
    yield
    p = os.path.join(reg.REPO, sub.SUBLOG)
    if os.path.exists(p):
        os.remove(p)


def _submit(tmp_path, run="exp46_C4Ff025_l40_2x32", mode="FULL", fake=None, **kw):
    return sub.submit(run, mode, "3-00:00:00", "flash", reg.FLASH_VERSION, run.endswith("_r8x8"), False,
                      root=str(tmp_path), runner=fake or FakeScheduler(), **kw)


@pytest.mark.parametrize("mode", ["FULL", "SMOKE"])
def test_submit_is_exactly_once_per_run_and_mode(tmp_path, no_preflight, mode):
    fake = FakeScheduler()
    out = _submit(tmp_path, mode=mode, fake=fake)
    sbatches = lambda: [c for c in fake.calls if c[0] == "sbatch"]  # noqa: E731
    assert out["job_id"] == "4242" and len(sbatches()) == 1
    argv = sbatches()[0]
    assert f"--job-name=exp46-exp46_C4Ff025_l40_2x32-{mode.lower()}-{out['token'][:8]}" in argv and "--gres=gpu:l40:2" in argv
    assert f"--comment=exp46:{out['token']}" in argv and argv[-1].endswith("/exp46_train.sbatch")
    assert "--no-requeue" in argv and any(a.startswith(f"--export=ALL,RUN=exp46_C4Ff025_l40_2x32,MODE={mode}") for a in argv)
    od, intent = sub.find_intent("exp46_C4Ff025_l40_2x32", mode, root=str(tmp_path))
    assert intent["state"] == "submitted" and intent["job_id"] == "4242" and set(intent) == sub.INTENT_KEYS
    assert intent["save_dir"] == str(tmp_path / ("exp46_C4Ff025_l40_2x32" if mode == "FULL" else "exp46_C4Ff025_l40_2x32_smoke"))
    held = json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"] if mode == "FULL" else {}
    assert (out["storage"] is None) == (mode == "SMOKE") and sorted(held) == (["exp46_C4Ff025_l40_2x32"] if mode == "FULL" else [])
    assert mode == "SMOKE" or (held["exp46_C4Ff025_l40_2x32"]["token"] == out["token"]
                               and held["exp46_C4Ff025_l40_2x32"]["reserved_bytes"] == reg.RUN_RESERVE_BYTES)
    assert intent["partition_max_time"] == "7-00:00:00" and intent["time"] == "3-00:00:00"
    with pytest.raises(reg.GateError):                          # the owner record refuses a second submission
        _submit(tmp_path, mode=mode, fake=fake)
    assert len(sbatches()) == 1


def _claim(tmp_path, monkeypatch, out, run="exp46_C4Ff025_l40_2x32", mode="FULL", **over):
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
                                  {"jid": "999"}, {"jname": "exp46-exp46_C4Ff025_l40_2x32-full-deadbeef"}])
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
        _submit(tmp_path, run="exp46_C4Ff050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    od, intent = sub.find_intent("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path))
    assert intent["state"] == "submitting" and intent["job_id"] is None
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("SLURM_JOB_NAME", intent["job_name"])
    sub.claim_attempt("exp46_C4Ff050_l40_2x32", "FULL", intent["token"], "a" * 40, "flash", intent["save_dir"],
                      reg.FLASH_VERSION, False, root=str(tmp_path))


def test_reconcile_resolves_only_an_unambiguous_state(tmp_path, no_preflight):
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="exp46_C4Ff050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    _, intent = sub.find_intent("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path))
    name = intent["job_name"]
    with pytest.raises(sub.SubmitError):                       # zero jobs: refuse, never guess
        sub.reconcile("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler())
    two = FakeScheduler(squeue="777|PENDING\n778|PENDING\n")
    with pytest.raises(sub.SubmitError):                       # two jobs carry the token
        sub.reconcile("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=two)
    fake = FakeScheduler(squeue="777|PENDING\n", sacct=f"777|{name}|PENDING\n")
    out = sub.reconcile("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=fake)
    assert out["intent"]["state"] == "submitted" and out["intent"]["job_id"] == "777" and out["intent"]["reconciled"]
    with pytest.raises(sub.SubmitError):                       # recorded 777 but the scheduler shows 778
        sub.reconcile("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler(squeue="778|RUNNING\n"))
    with pytest.raises(sub.SubmitError):                       # live job: never abandon
        sub.abandon("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=fake)
    done = FakeScheduler(sacct=f"777|{name}|FAILED\n")
    (tmp_path / "exp46_C4Ff050_l40_2x32").mkdir()
    assert "exp46_C4Ff050_l40_2x32" in json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"]   # held while owned
    moved = sub.abandon("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=done)
    assert moved["storage_released"] and json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"] == {}
    assert not os.path.exists(tmp_path / "exp46_C4Ff050_l40_2x32") and os.path.isdir(moved["run_dir_moved_to"])
    assert os.path.isdir(moved["owner_moved_to"]) and "/abandoned/" in moved["owner_moved_to"]
    with pytest.raises(sub.SubmitError):                       # an abandoned record is never found again
        sub.find_intent("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path))
    _submit(tmp_path, run="exp46_C4Ff050_l40_2x32")                        # ... and the run may be submitted afresh


def test_a_full_submission_reserves_before_any_owner_record_and_refuses_below_the_floor(tmp_path, monkeypatch, no_preflight):
    """plan r3 §3.2: the reservation precedes sbatch AND the owner intent; below the floor nothing is written."""
    monkeypatch.setattr(reg, "free_bytes", lambda path: reg.STORAGE_FLOOR_BYTES + reg.RUN_RESERVE_BYTES - 1)
    fake = FakeScheduler()
    with pytest.raises(reg.GateError, match="floor"):
        _submit(tmp_path, fake=fake)
    assert [c for c in fake.calls if c[0] == "sbatch"] == [] and not os.path.exists(tmp_path / sub.OWNERS)
    assert not os.path.exists(tmp_path / reg.STORAGE_LEDGER)
    monkeypatch.setattr(reg, "free_bytes", lambda path: reg.STORAGE_FLOOR_BYTES + reg.RUN_RESERVE_BYTES)
    assert _submit(tmp_path, fake=fake)["storage"]["reserved_bytes"] == reg.RUN_RESERVE_BYTES
    assert _submit(tmp_path, mode="SMOKE", fake=fake)["storage"] is None          # a SMOKE writes no checkpoint
    with pytest.raises(reg.GateError, match="floor"):                             # the next candidate: no room
        _submit(tmp_path, run="exp46_C4Ff050_l40_2x32", fake=fake)
    assert not os.path.exists(os.path.join(sub.owner_dir("exp46_C4Ff050_l40_2x32", "FULL", str(tmp_path))))


@pytest.mark.parametrize("time_limit,scontrol,ok", [
    ("7-00:00:00", SCONTROL_ALL, True), ("7-00:00:01", SCONTROL_ALL, False), ("168:00:01", SCONTROL_ALL, False),
    ("8-00:00:00", SCONTROL_ALL.replace("MaxTime=7-00:00:00", "MaxTime=UNLIMITED"), True),
    ("1-00:00:00", "PartitionName=all State=UP", False),                       # no MaxTime: fail closed
    ("1-00:00:00", SCONTROL_ALL.replace("7-00:00:00", "7days"), False),        # unparseable: fail closed
    ("1-00:60:00", SCONTROL_ALL, False),
])
def test_the_walltime_is_refused_above_the_partition_maxtime_before_any_owner_record(tmp_path, no_preflight,
                                                                                     time_limit, scontrol, ok):
    """plan r3 §3.2 / L6 (Codex plan r1 finding 14): the request is checked against `all`'s MaxTime before any
    reservation or owner intent, recorded in the owner record, never clamped and never chained."""
    fake = FakeScheduler(scontrol=scontrol)
    if ok:
        out = sub.submit(RF050, "FULL", time_limit, "flash", reg.FLASH_VERSION, False, False, root=str(tmp_path), runner=fake)
        assert sub.find_intent(RF050, "FULL", root=str(tmp_path))[1]["time"] == time_limit and out["job_id"]
        return
    with pytest.raises(sub.SubmitError):
        sub.submit(RF050, "FULL", time_limit, "flash", reg.FLASH_VERSION, False, False, root=str(tmp_path), runner=fake)
    assert [c[0] for c in fake.calls if c[0] == "sbatch"] == [] and not os.path.exists(tmp_path / sub.OWNERS)
    assert not os.path.exists(tmp_path / reg.STORAGE_LEDGER)


def test_scontrol_failure_refuses_the_submission(tmp_path, no_preflight):
    with pytest.raises(sub.SubmitError, match="MaxTime"):
        _submit(tmp_path, fake=FakeScheduler(fail={"scontrol"}))
    assert not os.path.exists(tmp_path / sub.OWNERS)


def test_the_submitter_names_exp46_and_fetches_the_exp46_branch():
    assert sub.job_name(RF050, "FULL", "ab" * 16) == f"exp46-{RF050}-full-abababab" and sub.OWNERS == ".exp46_owners"
    assert sub.BRANCH == "exp-46-c4flip-data-curve" and sub.PARTITION == "all"
    import inspect
    src = inspect.getsource(sub.preflight)
    assert '"fetch", "-q", "origin", BRANCH' in src and "exp-38-data-efficiency" not in open(sub.__file__).read()


def test_scheduler_queries_fail_closed(tmp_path):
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(fail={"sacct"}))
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(sacct="1|x|OUT_OF_ME+\n"))   # truncated state


def test_abandon_refuses_when_the_recorded_job_is_invisible(tmp_path, no_preflight):
    _submit(tmp_path, run="exp46_C4Ff050_l40_2x32", fake=FakeScheduler(sbatch="99"))
    with pytest.raises(sub.SubmitError):
        sub.abandon("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path), runner=FakeScheduler())


def test_run_dirs_owner_dirs_and_resources():
    assert sub.run_dir("exp46_C4Ff025_l40_2x32", "SMOKE", "/r") == "/r/exp46_C4Ff025_l40_2x32_smoke"
    assert sub.run_dir("exp46_C4Ff025_l40_2x32", "FULL", "/r") == "/r/exp46_C4Ff025_l40_2x32"
    assert sub.owner_dir("exp46_C4Ff025_l40_2x32", "SMOKE", "/r") == "/r/.exp46_owners/exp46_C4Ff025_l40_2x32__SMOKE"
    assert sub.resources("exp46_C4Ff025_l40_2x32") == {"ngpu": 2, "cpus": 22, "mem": "64G"}
    with pytest.raises(reg.GateError):
        sub.run_dir("de_p1_f050", "FULL", "/r")
    with pytest.raises(sub.SubmitError):
        sub.submit("exp46_C4Ff025_l40_2x32", "FULL", "forever", "flash", "", False, True)


def _claim_proc(args):
    tmp, token, jid, name, sd = args
    os.environ["SLURM_JOB_ID"], os.environ["SLURM_JOB_NAME"] = jid, name
    try:
        sub.claim_attempt("exp46_C4Ff050_l40_2x32", "FULL", token, "a" * 40, "flash", sd, reg.FLASH_VERSION, False, root=tmp)
        return jid
    except reg.GateError:
        return None


def test_two_simultaneous_jobs_of_one_owner_cannot_both_claim(tmp_path, no_preflight):
    """Codex code r2 HIGH 2: an interrupted submission (job id unknown) and a duplicate job start at
    once: the fixed owner-scoped claim admits exactly one."""
    import multiprocessing as mp
    with pytest.raises(sub.SubmitError):
        _submit(tmp_path, run="exp46_C4Ff050_l40_2x32", fake=FakeScheduler(fail={"sbatch"}))
    od, intent = sub.find_intent("exp46_C4Ff050_l40_2x32", "FULL", root=str(tmp_path))
    args = [(str(tmp_path), intent["token"], str(1000 + i), intent["job_name"], intent["save_dir"]) for i in range(6)]
    with mp.get_context("fork").Pool(6) as pool:
        got = pool.map(_claim_proc, args)
    assert len([g for g in got if g]) == 1
    assert len([n for n in os.listdir(od) if n.startswith("attempt_")]) == 1


def test_an_untracked_importable_next_to_the_kit_refuses_every_mode(tmp_path):
    shadow = os.path.join(EXP, f"zz_exp46_shadow_test_{os.getpid()}.py")
    open(shadow, "w").write("raise SystemExit(99)\n")
    try:
        r = _sbatch_dry(save_dir=str(tmp_path / "exp46_C4Ff025_l40_2x32"))
        assert r.returncode == 3 and "untracked importable files in the kit folder" in r.stderr
    finally:
        os.remove(shadow)


# ------------------------------------------------------------------ release: a sealed run's stranded hold (Codex code r2 finding 8)
def _sealed_full(tmp_path, run="exp46_C4Ff025_l40_2x32"):
    """A FULL run sealed by the kit's own marker writer: the run dir holds a training receipt and a COMPLETE
    launch record of this owner's attempt, bound by TRAINING_COMMITTED.json."""
    od, intent = sub.find_intent(run, "FULL", root=str(tmp_path))
    rd = tmp_path / run
    rd.mkdir()
    (rd / reg.TRAINING_RECEIPT).write_text('{"sealed": "training receipt"}\n')
    digest = reg.sha256_file(str(rd / reg.TRAINING_RECEIPT))
    lr = rd / f"at_launch_{run}_neuronic_FULL_x.json"
    json.dump({"run": run, "status": "COMPLETE", "training_receipt_sha256": digest,
               "owner": os.path.join(od, f"attempt_{intent['job_id']}.json")}, open(lr, "w"))
    reg.write_commit_marker(str(rd), run, digest, str(lr))
    return rd, od, intent


def _tree_state(*dirs):
    return [sorted((os.path.relpath(os.path.join(d, f), str(top)), os.lstat(os.path.join(d, f)).st_mtime_ns)
                   for d, _, fs in os.walk(str(top)) for f in fs) for top in dirs]


def test_release_frees_only_the_ledger_hold_of_a_sealed_terminal_run(tmp_path, no_preflight):
    out = _submit(tmp_path)
    rd, od, intent = _sealed_full(tmp_path)
    name = intent["job_name"]
    before = _tree_state(rd, od)
    live = FakeScheduler(squeue="4242|RUNNING\n", sacct=f"4242|{name}|RUNNING\n")
    with pytest.raises(sub.SubmitError, match="not terminal"):
        sub.release("exp46_C4Ff025_l40_2x32", root=str(tmp_path), runner=live)
    assert list(json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"]) == ["exp46_C4Ff025_l40_2x32"]
    done = FakeScheduler(sacct=f"4242|{name}|COMPLETED\n")
    got = sub.release("exp46_C4Ff025_l40_2x32", root=str(tmp_path), runner=done)
    led = json.load(open(tmp_path / reg.STORAGE_LEDGER))
    assert got["released"] and led["reservations"] == {} and led["releases"][-1]["token"] == out["token"]
    assert "sealed" in led["releases"][-1]["reason"]
    assert _tree_state(rd, od) == before                         # the run dir and the owner record are untouched
    assert sub.release("exp46_C4Ff025_l40_2x32", root=str(tmp_path), runner=done)["released"] is False   # idempotent
    with pytest.raises(SystemExit):
        sub.main(["release"])                                     # the CLI verb exists (and needs --run)


@pytest.mark.parametrize("what", ["no marker", "record changed", "receipt changed", "another attempt", "foreign token",
                                  "job not visible"])
def test_release_refuses_an_unsealed_foreign_or_unproven_run(tmp_path, no_preflight, what):
    _submit(tmp_path)
    rd, od, intent = _sealed_full(tmp_path)
    name = intent["job_name"]
    runner = FakeScheduler(sacct=f"4242|{name}|COMPLETED\n")
    lr = [f for f in os.listdir(rd) if f.startswith("at_launch_")][0]
    if what == "no marker":
        for f in (reg.COMMIT_MARKER, reg.COMMIT_MARKER + ".sha256"):
            os.remove(rd / f)
    elif what == "record changed":
        rec = json.load(open(rd / lr))
        rec["note"] = "rewritten after the seal"
        json.dump(rec, open(rd / lr, "w"))
    elif what == "receipt changed":
        (rd / reg.TRAINING_RECEIPT).write_text('{"other": 1}\n')
    elif what == "another attempt":
        cm = json.load(open(rd / reg.COMMIT_MARKER))
        rec = json.load(open(rd / lr))
        rec["owner"] = os.path.join(od, "attempt_999.json")
        for f in (reg.COMMIT_MARKER, reg.COMMIT_MARKER + ".sha256"):
            os.remove(rd / f)
        json.dump(rec, open(rd / lr, "w"))
        reg.write_commit_marker(str(rd), "exp46_C4Ff025_l40_2x32", cm["training_receipt_sha256"], str(rd / lr))
    elif what == "foreign token":
        p = tmp_path / reg.STORAGE_LEDGER
        led = json.load(open(p))
        led["reservations"]["exp46_C4Ff025_l40_2x32"]["token"] = "9" * 32
        json.dump(led, open(p, "w"))
    else:
        runner = FakeScheduler()                                   # the recorded job is not visible
    with pytest.raises(reg.GateError):
        sub.release("exp46_C4Ff025_l40_2x32", root=str(tmp_path), runner=runner)
    assert list(json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"]) == ["exp46_C4Ff025_l40_2x32"]


def test_the_launcher_writes_no_bytecode_for_the_reused_exp38_tools(tmp_path):
    """Codex code r2 finding 13: the importing driver (the supervisor the sbatch execs) runs with bytecode writing
    disabled, so a fresh launch checkout never acquires __pycache__ files under exp_38's folder. The dry-run hook
    imports the reused exp_38 tools; with PYTHONPYCACHEPREFIX pointed at a scratch dir, any bytecode Python would
    have written lands there instead of in the repo -- and none for exp_38's folder may appear (the supervisor's own
    probe subprocesses, which import no exp_38 tool, may still cache stdlib bytecode there)."""
    prefix = tmp_path / "pycache_prefix"
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN="exp46_C4Ff025_l40_2x32", MODE="FULL",
               EXPECT_SHA="a" * 40, EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION,
               SAVE_DIR="/x", ALLOW_NONMATCHED_RUNG=0, EXP38_DRY_SAVE_ROOT=str(tmp_path), PYTHONPYCACHEPREFIX=str(prefix))
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    written = [os.path.join(d, f) for d, _, fs in os.walk(prefix) for f in fs] if prefix.exists() else []
    assert not [w for w in written if "exp_38_data_efficiency_claude" in w], written[:5]
    assert not [w for w in written if "/worklog/worklog_yixun/" in w], written[:5]      # nor of any kit / tool folder
    lines = open(SBATCH).read().split("\n")
    execs = [ln for ln in lines if ln.strip().startswith("exec ")]
    assert len(execs) == 2 and all('exec "$PY" -B ' in ln for ln in execs), execs



# ------------------------------------------------------------------ fix round 2, Codex code r2 finding 9: sealed runs
def test_abandon_refuses_a_sealed_run_and_release_frees_its_stranded_hold(tmp_path, no_preflight):
    """A FULL job cancelled after its commit marker (before its reservation was released) is a valid SEALED run:
    abandon refuses it -- nothing moved, nothing released, exit 2 at the CLI -- and names `release --run`, which frees
    only the stranded hold (the run dir and the owner record untouched)."""
    run = "exp46_C4Ff025_l40_2x32"
    _submit(tmp_path)
    rd, od, intent = _sealed_full(tmp_path)
    cancelled = FakeScheduler(sacct=f"4242|{intent['job_name']}|CANCELLED\n")
    before = _tree_state(rd, od)
    with pytest.raises(sub.SubmitError, match=rf"SEALED.*release --run {run}"):
        sub.abandon(run, "FULL", root=str(tmp_path), runner=cancelled)
    assert _tree_state(rd, od) == before and sorted(os.listdir(tmp_path / sub.OWNERS)) == [f"{run}__FULL"]
    assert not [p for p in os.listdir(tmp_path) if ".abandoned_" in p]
    assert list(json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"]) == [run]
    assert sub.release(run, root=str(tmp_path), runner=cancelled)["released"]
    assert _tree_state(rd, od) == before and json.load(open(tmp_path / reg.STORAGE_LEDGER))["reservations"] == {}


def test_an_unsealed_cancelled_attempt_is_still_abandoned(tmp_path, no_preflight):
    run = "exp46_C4Ff025_l40_2x32"
    _submit(tmp_path)
    od, intent = sub.find_intent(run, "FULL", root=str(tmp_path))
    (tmp_path / run).mkdir()                                     # a partial run dir: no commit marker
    (tmp_path / run / reg.TRAINING_RECEIPT).write_text("{}")
    got = sub.abandon(run, "FULL", root=str(tmp_path),
                      runner=FakeScheduler(sacct=f"4242|{intent['job_name']}|CANCELLED\n"))
    assert got["run_dir_moved_to"] and got["storage_released"] and not os.path.lexists(tmp_path / run)
