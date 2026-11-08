"""exp_38 B6/B7 — launch paths: the A6000 launcher (DRY_RUN + refusals), the neuronic sbatch
(EXP38_SBATCH_DRYRUN) and the exactly-once neuronic submitter (fake scheduler).

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


@pytest.mark.parametrize("run", ["de_yaw_f025", "de_bf_f075"])
def test_dry_run_prints_the_registered_argv_and_spawns_nothing(run, tmp_path):
    r = _launch(MODE="DRY_RUN", RUN=run, EXP38_DRY_SAVE_ROOT=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert f"DRY_RUN OK {run} (nothing spawned)" in r.stdout
    full = [l for l in r.stdout.split("\n") if l.startswith("FULL argv:")][0]
    want = " ".join(reg.build_train_argv(run, "a6000", save_root=str(tmp_path)))
    assert full.endswith("python " + want + " ")
    assert "E0 receipt (FULL needs it):" in r.stdout and not os.listdir(tmp_path)


@pytest.mark.parametrize("kw,needle", [
    ({"MODE": "LAUNCH", "RUN": "de_yaw_f025"}, "MODE must be"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025_r8x8"}, "RUN must be a 2x32"),
    ({"MODE": "DRY_RUN", "RUN": "dc_van_f025"}, "RUN must be a 2x32"),
    ({"MODE": "FULL", "RUN": "de_yaw_f025", "RESUME_STEP": "2500"}, "there is no resume"),
    ({"MODE": "SMOKE", "RUN": "de_yaw_f025", "EXP38_LOCKDIR": "/tmp/x"}, "DRY_RUN test hook"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025", "GPU_POLICY": "maybe"}, "GPU_POLICY"),
    ({"MODE": "DRY_RUN", "RUN": "de_yaw_f025", "ATTN_BACKEND": "xformers"}, "ATTN_BACKEND"),
])
def test_launcher_refusals(kw, needle):
    r = _launch(**kw)
    assert r.returncode == 2 and needle in r.stderr, (r.returncode, r.stderr)


def test_dry_run_refuses_a_non_fresh_run_dir(tmp_path):
    d = tmp_path / "de_bf_f025"
    d.mkdir()
    (d / "epoch=2-step=2500.ckpt").write_bytes(b"x")
    r = _launch(MODE="DRY_RUN", RUN="de_bf_f025", EXP38_DRY_SAVE_ROOT=str(tmp_path))
    assert r.returncode == 2 and "epoch=2-step=2500.ckpt" in r.stderr and "for a start at step 0" in r.stderr


def test_launcher_body_is_one_function_called_last():
    lines = open(LAUNCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"] and "main() {" in lines
    lines = open(SBATCH).read().rstrip("\n").split("\n")
    assert lines[-2:] == ['main "$@"', "exit $?"]


@pytest.mark.parametrize("run,mode", [("de_yaw_f025", "FULL"), ("de_bf_f050", "SMOKE")])
def test_sbatch_dry_run_prints_the_torchrun_line(run, mode, tmp_path):
    sd = str(tmp_path / run)
    env = _env(EXP38_SBATCH_DRYRUN=1, EXP38_ENV_PREFIX=ENV_PREFIX, RUN=run, MODE=mode,
               EXPECT_SHA=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               EXP38_TOKEN="0" * 32, ATTN_BACKEND="flash", EXPECT_FLASH_VERSION=reg.FLASH_VERSION, SAVE_DIR=sd)
    env.pop("SLURM_JOB_ID", None)
    r = subprocess.run(["bash", SBATCH], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    clean = not subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
    if not clean:
        assert r.returncode == 2 and "tracked tree is not clean" in r.stderr
        return
    if mode == "FULL":                      # FULL's SAVE_DIR must be the registry's neuronic run dir
        assert r.returncode == 2 and "SAVE_DIR" in r.stderr
        return
    assert r.returncode == 0, r.stderr
    line = [l for l in r.stdout.split("\n") if l.startswith("DRYRUN SMOKE")][0]
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py" in line and "--max-steps 30" in line


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
    monkeypatch.setattr(sub, "preflight", lambda run, mode, allow, dry: "a" * 40)
    monkeypatch.setattr(sub, "SUBLOG", "worklog/worklog_yixun/exp_38_data_efficiency_claude/.test_submissions.log")
    yield
    p = os.path.join(reg.REPO, sub.SUBLOG)
    if os.path.exists(p):
        os.remove(p)


def test_submit_is_exactly_once(tmp_path, no_preflight):
    fake = FakeScheduler()
    out = sub.submit("de_yaw_f025", "FULL", "3-00:00:00", "flash", reg.FLASH_VERSION, False, False,
                     save_root=str(tmp_path), runner=fake)
    assert out["job_id"] == "4242" and len(fake.calls) == 1
    argv = fake.calls[0]
    assert f"--job-name=exp38-de_yaw_f025-full-{out['token'][:8]}" in argv and "--gres=gpu:l40:2" in argv
    assert any(a.startswith("--export=ALL,RUN=de_yaw_f025,MODE=FULL") for a in argv)
    intent = json.load(open(tmp_path / "de_yaw_f025" / ".exp38_submitted" / "intent.json"))
    assert intent["state"] == "submitted" and intent["job_id"] == "4242"
    with pytest.raises((sub.SubmitError, reg.GateError)):   # the marker refuses a second submission
        sub.submit("de_yaw_f025", "FULL", "3-00:00:00", "flash", reg.FLASH_VERSION, False, False,
                   save_root=str(tmp_path), runner=fake)
    assert len(fake.calls) == 1


def test_failed_sbatch_leaves_a_submitting_intent_that_reconcile_resolves(tmp_path, no_preflight):
    with pytest.raises(sub.SubmitError):
        sub.submit("de_bf_f050", "FULL", "24:00:00", "flash", reg.FLASH_VERSION, False, False,
                   save_root=str(tmp_path), runner=FakeScheduler(fail={"sbatch"}))
    _, intent = sub.find_intent("de_bf_f050", "FULL", save_root=str(tmp_path))
    assert intent["state"] == "submitting"
    name = intent["job_name"]
    fake = FakeScheduler(squeue=f"777|PENDING\n", sacct=f"777|{name}|PENDING\n")
    out = sub.reconcile("de_bf_f050", "FULL", save_root=str(tmp_path), runner=fake)
    assert out["intent"]["state"] == "submitted" and out["intent"]["job_id"] == "777"
    with pytest.raises(sub.SubmitError):                       # live job: never abandon
        sub.abandon("de_bf_f050", "FULL", save_root=str(tmp_path), runner=fake)
    done = FakeScheduler(sacct=f"777|{name}|FAILED\n")
    moved = sub.abandon("de_bf_f050", "FULL", save_root=str(tmp_path), runner=done)
    assert not os.path.exists(tmp_path / "de_bf_f050") and os.path.isdir(moved["moved_to"])


def test_scheduler_queries_fail_closed(tmp_path):
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(fail={"sacct"}))
    with pytest.raises(sub.SubmitError):
        sub.scheduler_state("x", runner=FakeScheduler(sacct="1|x|OUT_OF_ME+\n"))   # truncated state


def test_abandon_refuses_when_the_recorded_job_is_invisible(tmp_path, no_preflight):
    sub.submit("de_yaw_f050", "FULL", "24:00:00", "flash", reg.FLASH_VERSION, False, False,
               save_root=str(tmp_path), runner=FakeScheduler(sbatch="99"))
    with pytest.raises(sub.SubmitError):
        sub.abandon("de_yaw_f050", "FULL", save_root=str(tmp_path), runner=FakeScheduler())


def test_smoke_and_full_use_distinct_run_dirs_and_resources():
    t = "ab" * 16
    assert sub.run_dir("de_bf_f025", "SMOKE", t, "/r") == f"/r/de_bf_f025_smoke_{t[:8]}"
    assert sub.run_dir("de_bf_f025", "FULL", t, "/r") == "/r/de_bf_f025"
    assert sub.resources("de_bf_f025_r8x8") == {"ngpu": 8, "cpus": 64, "mem": "108G"}
    with pytest.raises(sub.SubmitError):
        sub.submit("de_bf_f025", "FULL", "forever", "flash", "", False, True)
