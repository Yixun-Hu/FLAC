"""exp_40 yaw_aug_c16 -- the training kit (exp40_kit.py) and the evidence driver.

CPU only, synthetic inputs, no GPU, no NAS, no Slurm: every external command the kit runs
is injected (stub runners) and every filesystem effect lands under ``tmp_path``. One test
builds a real run tree from this checkout's HEAD (``git archive``), one runs the real
prefigure parse, one runs PyTorch Lightning twice on the tiny CPU model (init parity).
"""
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load("exp40_kit", _EXP / "exp40_kit.py")
drv = _load("exp40_evidence_driver", _EXP / "exp40_evidence_driver.py")
cap = _load("exp40_yaw_aug_golden_capture", _REPO / "src/tests/exp40_yaw_aug_golden_capture.py")
RID = "exp40_C16_a6000_2x32"
STAMP = "2026-11-07_22-00-00"


# --------------------------------------------------------------------------- #
# argv / recipe projection / prefigure parse
# --------------------------------------------------------------------------- #
def test_full_argv_is_the_exp15_shape_with_the_c16_config():
    a = kit.train_argv("a6000", "2x32", "FULL", RID, "/abs/save")
    assert a[:2] == ["--model-config", kit.ARM_CONFIG]
    pairs = dict(zip(a[::2], a[1::2]))
    assert pairs["--batch-size"] == "32" and pairs["--num-gpus"] == "2" and pairs["--accum-batches"] == "1"
    assert pairs["--strategy"] == "ddp_find_unused_parameters_true" and pairs["--sync-batchnorm"] == "true"
    assert pairs["--max-steps"] == "40000" and pairs["--checkpoint-every"] == "2500"
    assert pairs["--seed"] == "42" and pairs["--precision"] == "bf16-mixed" and pairs["--val-every"] == "-1"
    assert pairs["--experiment-name"] == RID and pairs["--save-dir"] == "/abs/save"


def test_ddp_smoke_differs_from_full_only_in_steps_and_namespace():
    full = kit.train_argv("a6000", "2x32", "FULL", RID, "/abs/a")
    smoke = kit.train_argv("a6000", "2x32", "DDP_SMOKE", RID, "/abs/b", STAMP)
    assert kit.recipe_projection(full) == kit.recipe_projection(smoke)
    assert dict(zip(smoke[::2], smoke[1::2]))["--checkpoint-every"] == "2500"      # plan §14 H5
    one = kit.train_argv("a6000", "2x32", "SMOKE", RID, "/abs/b", STAMP)
    assert "--sync-batchnorm" not in one and dict(zip(one[::2], one[1::2]))["--num-gpus"] == "1"


@pytest.mark.parametrize("args,needle", [
    (("lambda", "2x32", "FULL", RID), "platform"),
    (("a6000", "8x8", "FULL", "exp40_C16_a6000_8x8"), "rung"),
    (("a6000", "2x32", "PILOT", RID), "mode"),
    (("neuronic", "2x32", "SMOKE", "exp40_C16_neuronic_2x32"), "mode"),
    (("a6000", "2x32", "FULL", "exp40_C16_neuronic_2x32"), "names"),
    (("a6000", "2x32", "FULL", "exp14_x"), "does not match"),
])
def test_identity_refusals(args, needle):
    with pytest.raises(kit.Refused, match=needle):
        kit.train_argv(*args, save_root="/abs")


def test_relative_save_root_refused():
    with pytest.raises(kit.Refused, match="absolute"):
        kit.train_argv("a6000", "2x32", "FULL", RID, "rel/save")


def test_prefigure_parse_matches_the_expected_values_type_strict():
    for mode, stamp in (("FULL", ""), ("DDP_SMOKE", STAMP), ("SMOKE", STAMP)):
        argv = kit.train_argv("a6000", "2x32", mode, RID, "/abs/s", stamp)
        want = kit.expected_parse("a6000", "2x32", mode, RID, "/abs/s", stamp)
        assert kit.parsed_argv_reasons(str(_REPO), argv, want, sys.executable) == []
    bad = dict(want, batch_size=8)
    assert kit.parsed_argv_reasons(str(_REPO), argv, bad, sys.executable)


# --------------------------------------------------------------------------- #
# immutable run tree (real git archive of HEAD)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def runtree(tmp_path_factory):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    base = tmp_path_factory.mktemp("rt")
    ar, vae = base / "AR", base / "vae.safetensors"
    ar.mkdir()
    vae.write_bytes(b"vae")
    links = {"AcousticRooms": str(ar), "weights/FLAC/VAE.safetensors": str(vae)}
    tree = base / "tree"
    kit.build_runtree(str(_REPO), head, str(tree), links)
    return head, tree, links


def _writable(path):
    os.chmod(path.parent, 0o755)
    if path.exists() and not path.is_symlink():
        os.chmod(path, 0o644)


def test_runtree_verifies(runtree):
    head, tree, links = runtree
    assert kit.verify_runtree(str(_REPO), head, str(tree), links) == []
    assert kit.verify_runtree(None, head, str(tree), links) == []          # carried manifest alone


def test_runtree_refuses_rebuild_and_wrong_links(runtree, tmp_path):
    head, tree, links = runtree
    with pytest.raises(kit.Refused, match="built once"):
        kit.build_runtree(str(_REPO), head, str(tree), links)
    with pytest.raises(kit.Refused, match="exactly"):
        kit.build_runtree(str(_REPO), head, str(tmp_path / "x"), {"AcousticRooms": links["AcousticRooms"]})


def test_runtree_detects_a_modified_file(runtree, tmp_path):
    head, tree, links = runtree
    f = tree / "train.py"
    _writable(f)
    orig = f.read_bytes()
    try:
        f.write_bytes(orig + b"\n# shadow\n")
        r = kit.verify_runtree(str(_REPO), head, str(tree), links)
        assert any("train.py differs" in x for x in r) and any("writable" in x for x in r)
    finally:
        f.write_bytes(orig)
        os.chmod(f, 0o444)
        os.chmod(f.parent, 0o555)


def test_runtree_detects_an_extra_file_and_a_retargeted_link(runtree, tmp_path):
    head, tree, links = runtree
    extra = tree / "src" / "sitecustomize.py"
    os.chmod(extra.parent, 0o755)
    try:
        extra.write_text("x = 1\n")
        os.chmod(extra, 0o444)
        assert any("untracked file src/sitecustomize.py" in x
                   for x in kit.verify_runtree(str(_REPO), head, str(tree), links))
    finally:
        os.chmod(extra, 0o644)
        extra.unlink()
        os.chmod(extra.parent, 0o555)
    other = dict(links, AcousticRooms=str(tmp_path))
    assert any("links" in x or "->" in x for x in kit.verify_runtree(str(_REPO), head, str(tree), other))


# --------------------------------------------------------------------------- #
# start watcher
# --------------------------------------------------------------------------- #
B = kit.BANNER
STEP1 = "Epoch 0:   0%|          | 1/4550 [00:04<6:16:50,  0.20it/s, train/loss=2.3]"
MEM = "Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {m}/2"


def test_start_gate_pass_and_pending():
    ok = "\n".join([MEM.format(r=0, m=1), MEM.format(r=1, m=2), B, "Epoch 0:   0%| | 0/4550 [", STEP1])
    assert kit.start_gate(ok, 2)[0] == "pass"
    assert kit.start_gate("\n".join([MEM.format(r=0, m=1), B]), 2)[0] == "pending"


@pytest.mark.parametrize("text", [
    "\n".join([MEM.format(r=0, m=1), MEM.format(r=1, m=2), STEP1, B]),           # step first
    "\n".join([MEM.format(r=0, m=1), "yaw_aug ENABLED img_w=512 seed=42", STEP1]),  # exp_15 banner
    "\n".join([MEM.format(r=0, m=1), B, STEP1]),                                  # one rank only
    "\n".join(["Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8", B, STEP1]),  # wrong world
])
def test_start_gate_failures(text):
    assert kit.start_gate(text, 2)[0] == "fail"


def test_watch_terminates_only_its_own_session(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("\n".join([STEP1]))
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        rc = kit.watch(str(log), 1, proc.pid, proc.pid, timeout_s=30, poll_s=0.1)
        assert rc == 8
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    with pytest.raises(kit.Refused, match="watcher itself"):
        kit.watch(str(log), 1, os.getpid(), os.getpgrp(), timeout_s=1)


# --------------------------------------------------------------------------- #
# GPU gate and cooperative per-card locks
# --------------------------------------------------------------------------- #
U0, U1 = "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"
GPUS = f"0, {U0}, NVIDIA RTX A6000, 48000\n1, {U1}, NVIDIA RTX A6000, 48000\n"


def test_gpu_gate():
    g = kit.parse_gpu_query(GPUS)
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g, [], 2) == []
    busy = kit.parse_apps_query(f"{U1}, 4242, 3000\n")
    assert any("4242" in r for r in kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g, busy, 2))
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "shared", g, [], 2)[0].startswith("a primary FULL")
    assert kit.gpu_reasons("a6000", "SMOKE", "2x32", [0], "idle", g, busy, 2) == []
    assert kit.gpu_reasons("a6000", "SMOKE", "2x32", [0, 1], "idle", g, [], 2)        # 2 lanes for 1 rank
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g[:1], [], 2)      # count
    wrong = kit.parse_gpu_query(f"0, {U0}, NVIDIA L40, 48000\n1, {U1}, NVIDIA L40, 48000\n")
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", wrong, [], 2)
    stray = kit.parse_apps_query("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 1, 1\n")
    assert any("unknown" in r for r in kit.gpu_reasons("a6000", "SMOKE", "2x32", [0], "idle", g, stray, 2))
    with pytest.raises(kit.Refused):
        kit.parse_gpu_query("garbage")


def test_gpu_locks_race_rollback_and_owner(tmp_path):
    t1, p1 = kit.acquire_gpu_locks(str(tmp_path), [U1], {"who": "a"})
    with pytest.raises(kit.Refused, match="held"):
        kit.acquire_gpu_locks(str(tmp_path), [U0, U1], {"who": "b"})           # U0 taken first, then U1 fails
    assert not (tmp_path / f"GPU_LOCK_{U0}").exists(), "the partial acquisition was not rolled back"
    with pytest.raises(kit.Refused, match="another token"):
        kit.release_gpu_locks(p1, "not-the-owner")
    kit.release_gpu_locks(p1, t1)
    assert not (tmp_path / f"GPU_LOCK_{U1}").exists()
    with pytest.raises(kit.Refused):
        kit.release_gpu_locks(p1, t1)                                          # already released
    t2, p2 = kit.acquire_gpu_locks(str(tmp_path), [U0, U1], {"who": "c"})
    assert len(p2) == 2
    with pytest.raises(kit.Refused):
        kit.acquire_gpu_locks(str(tmp_path), ["not-a-uuid"], {})


# --------------------------------------------------------------------------- #
# neuronic: execution-identity claims and the submission state machine
# --------------------------------------------------------------------------- #
def test_claim_is_per_execution_and_exclusive(tmp_path):
    claims = os.path.join(_control(tmp_path), "claims")
    p = kit.claim(claims, "exp40_C16_neuronic_2x32", "FULL", "101", "intentA")
    assert os.path.basename(p) == "exp40_C16_neuronic_2x32__FULL.claim" and "intentA" not in p
    with pytest.raises(kit.Refused, match="must not train"):
        kit.claim(claims, "exp40_C16_neuronic_2x32", "FULL", "102", "intentB")


def test_a_second_control_namespace_cannot_admit_the_same_execution(tmp_path, capsys):
    """code-r2 B1: exactly ONE control directory; another one is refused for submission and
    for the in-job claim, and the CLI offers no override."""
    _control(tmp_path)
    other = os.path.realpath(str(tmp_path / "other_control"))
    with pytest.raises(kit.Refused, match="not THE exp_40 control directory"):
        kit.neuronic_submit(other, "exp40_C16_neuronic_2x32", "FULL", "exp40_C16_neuronic_2x32", _argv, Sched(),
                            relaxation="post-run-publish", probe_ok=True)
    with pytest.raises(kit.Refused, match="claims dir"):
        kit.claim(os.path.join(other, "claims"), "exp40_C16_neuronic_2x32", "FULL", "7", "i")
    kit.claim(os.path.join(kit.ALLOWED_CONTROL, "claims"), "exp40_C16_neuronic_2x32", "FULL", "7", "i")
    with pytest.raises(SystemExit):
        kit.main(["neuronic-submit", "--control-dir", other, "--run-id", "x", "--mode", "FULL",
                  "--execution-id", "x", "--sbatch-json", "{}", "--probe-script", "s", "--expect-sha", "a" * 40,
                  "--maxtime-raw", "1:00:00", "--time-min", "1", "--source-checkout", "/x"])
    for name in ("exp40_submit_neuronic.sh", "exp40_train.sbatch", "exp40_claim_probe.sbatch"):
        text = (_EXP / name).read_text()
        assert "/n/fs/gatrdp/exp40_control" in text, name


class Sched:
    """A stub Slurm: ``jobs`` name -> list of (id, state); ``visible_after`` delays sacct."""

    def __init__(self, fail=None):
        self.jobs, self.calls, self.fail, self.next_id, self.hidden = {}, [], fail, 1000, set()

    def __call__(self, cmd):
        self.calls.append(cmd)
        if self.fail and cmd[0] == self.fail:
            return 1, "", f"{cmd[0]} unavailable"
        if cmd[0] == "sbatch":
            name = cmd[cmd.index("--job-name") + 1]
            self.next_id += 1
            self.jobs.setdefault(name, []).append((str(self.next_id), "PENDING"))
            return 0, f"{self.next_id}\n", ""
        if cmd[0] == "squeue":
            name = cmd[cmd.index("-n") + 1]
            return 0, "".join(f"{j}|{s}\n" for j, s in self.jobs.get(name, [])
                              if s in ("PENDING", "RUNNING") and name not in self.hidden), ""
        if cmd[0] == "sacct":
            name = cmd[cmd.index("--name") + 1]
            if name in self.hidden:
                self.hidden.discard(name)          # appears after the grace interval
                return 0, "", ""
            return 0, "".join(f"{j}|{s}\n" for j, s in self.jobs.get(name, [])), ""
        raise AssertionError(cmd)


def _argv(name, intent):
    return ["sbatch", "--parsable", "--job-name", name, "--comment", intent, "script"]


@pytest.fixture(autouse=True)
def _restore_allowed_control():
    saved = kit.ALLOWED_CONTROL
    yield
    kit.ALLOWED_CONTROL = saved


def _control(tmp_path):
    """THE control directory of this test (code-r2 B1: the kit accepts exactly one)."""
    kit.ALLOWED_CONTROL = os.path.realpath(str(tmp_path / "control"))
    return kit.ALLOWED_CONTROL


def _submit(tmp_path, sched, exec_id, mode="PILOT", **kw):
    kw.setdefault("relaxation", "post-run-publish")
    kw.setdefault("probe_ok", True)
    return kit.neuronic_submit(_control(tmp_path), "exp40_C16_neuronic_2x32", mode, exec_id, _argv, sched,
                               grace_s=0, sleep=lambda s: None, **kw)


def _state(tmp_path, exec_id, mode="PILOT"):
    return kit.read_json(os.path.join(_control(tmp_path), "states", f"{exec_id}__{mode}.state.json"))


def _job_claims(tmp_path, exec_id, jid, mode="PILOT"):
    """What the job's first step does: the O_EXCL claim in the control dir."""
    return kit.claim(os.path.join(_control(tmp_path), "claims"), exec_id, mode, jid, "intent")


def test_submit_records_the_attempt_boundary(tmp_path):
    sched = Sched()
    action, st = _submit(tmp_path, sched, "e1")
    assert action == "submitted" and st["job_ids"] == ["1001"]
    rec = _state(tmp_path, "e1")
    assert rec["status"] == "submitted" and rec["sbatch"][3] == "exp40-e1-PILOT"
    assert rec["control_dir"] == _control(tmp_path)


def test_control_dir_must_be_canonical_absolute(tmp_path):
    with pytest.raises(kit.Refused, match="canonical"):
        kit.neuronic_submit("relative/control", "exp40_C16_neuronic_2x32", "PILOT", "e", _argv, Sched())
    with pytest.raises(kit.Refused, match="canonical"):
        kit.neuronic_submit(str(tmp_path) + "/a/../control", "exp40_C16_neuronic_2x32", "PILOT", "e", _argv, Sched())


def test_live_or_ambiguous_execution_blocks_any_new_one(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")                                             # PENDING
    with pytest.raises(kit.Refused, match="live"):
        _submit(tmp_path, sched, "e2")
    assert _state(tmp_path, "e1")["status"] == "live"                          # durable transition
    sched.jobs["exp40-e1-PILOT"] = []                                          # vanished: ambiguous
    with pytest.raises(kit.Refused, match="ambiguous"):
        _submit(tmp_path, sched, "e2")


def test_post_sbatch_crash_is_adopted_not_resubmitted(tmp_path):
    sched = Sched()
    os.makedirs(os.path.join(_control(tmp_path), "states"))
    kit.atomic_write_json(os.path.join(_control(tmp_path), "states", "e1__PILOT.state.json"),
                          {"run_id": "exp40_C16_neuronic_2x32", "mode": "PILOT", "execution_id": "e1",
                           "status": "sbatch_attempted", "intent": "x", "job_ids": []})
    sched.jobs["exp40-e1-PILOT"] = [("777", "RUNNING")]                        # sbatch DID happen
    with pytest.raises(kit.Refused, match="live"):
        _submit(tmp_path, sched, "e2")
    assert not [c for c in sched.calls if c[0] == "sbatch"]
    assert _state(tmp_path, "e1")["job_ids"] == ["777"]


def test_delayed_accounting_then_completed_owner(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")
    _job_claims(tmp_path, "e1", "1001")
    sched.jobs["exp40-e1-PILOT"] = [("1001", "COMPLETED")]
    sched.hidden.add("exp40-e1-PILOT")
    action, _ = _submit(tmp_path, sched, "e2")                                 # grace re-query finds it
    assert action == "submitted"
    st = _state(tmp_path, "e1")
    assert st["status"] == "terminal" and st["outcome"] == "completed" and st["owner_job"] == "1001"
    with pytest.raises(kit.Refused, match="fresh execution identity"):
        sched.jobs["exp40-e2-PILOT"] = [("1002", "FAILED")]
        _submit(tmp_path, sched, "e2")


def test_claim_owner_completed_with_a_refused_duplicate_is_completed(tmp_path):
    """H6: an accidental duplicate job that exits 3 on the claim is a recorded duplicate,
    not a failure of the execution."""
    sched = Sched()
    _submit(tmp_path, sched, "R", mode="FULL")
    _job_claims(tmp_path, "R", "1001", mode="FULL")
    sched.jobs["exp40-R-FULL"] = [("1001", "COMPLETED"), ("1005", "FAILED")]
    with pytest.raises(kit.Refused, match="already COMPLETED"):
        _submit(tmp_path, sched, "R_r1", mode="FULL")
    st = _state(tmp_path, "R", "FULL")
    assert st["outcome"] == "completed" and st["duplicates"] == ["1005"] and st["job_ids"] == ["1001", "1005"]


def test_owner_unknown_to_the_scheduler_or_completed_without_claim_is_ambiguous(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")
    _job_claims(tmp_path, "e1", "4242")
    sched.jobs["exp40-e1-PILOT"] = [("1001", "FAILED")]
    with pytest.raises(kit.Refused, match="ambiguous"):
        _submit(tmp_path, sched, "e2")
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    sched2 = Sched()
    _submit(tmp2, sched2, "e1")
    sched2.jobs["exp40-e1-PILOT"] = [("1001", "COMPLETED")]                    # no claim at all
    with pytest.raises(kit.Refused, match="ambiguous"):
        _submit(tmp2, sched2, "e2")


def test_failed_before_claiming_is_terminal_failed(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")
    sched.jobs["exp40-e1-PILOT"] = [("1001", "FAILED")]
    action, _ = _submit(tmp_path, sched, "e2")
    assert action == "submitted" and _state(tmp_path, "e1")["outcome"] == "failed"


@pytest.mark.parametrize("payload", ["{trunc", "[]", '{"run_id": "exp40_C16_neuronic_2x32"}',
                                     '{"run_id": "x", "mode": "PILOT", "execution_id": "e", "status": "weird", '
                                     '"intent": "i", "job_ids": []}'])
def test_malformed_or_truncated_state_is_an_explicit_ambiguous_refusal(tmp_path, payload):
    os.makedirs(os.path.join(_control(tmp_path), "states"))
    with open(os.path.join(_control(tmp_path), "states", "bad__PILOT.state.json"), "w") as f:
        f.write(payload)
    with pytest.raises(kit.Refused, match="ambiguous"):
        _submit(tmp_path, Sched(), "e2")


class FileSched:
    """A picklable stub Slurm whose jobs live in a JSON file (shared across processes)."""

    def __init__(self, path):
        self.path = path

    def __call__(self, cmd):
        import fcntl
        with open(self.path + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            jobs = json.load(open(self.path)) if os.path.exists(self.path) else {}
            if cmd[0] == "sbatch":
                time.sleep(0.5)                          # widen the race window
                name = cmd[cmd.index("--job-name") + 1]
                jid = str(2000 + sum(len(v) for v in jobs.values()))
                jobs.setdefault(name, []).append([jid, "PENDING"])
                json.dump(jobs, open(self.path, "w"))
                return 0, jid + "\n", ""
            key = "-n" if cmd[0] == "squeue" else "--name"
            name = cmd[cmd.index(key) + 1]
            return 0, "".join(f"{j}|{st}\n" for j, st in jobs.get(name, [])), ""


def _race_submit(args):
    control, sched_path, exec_id, barrier = args
    barrier.wait()
    try:
        action, _ = kit.neuronic_submit(control, "exp40_C16_neuronic_2x32", "FULL", exec_id, _argv,
                                        FileSched(sched_path), grace_s=0, sleep=lambda s: None,
                                        relaxation="post-run-publish", probe_ok=True)
        return action
    except kit.Refused as err:
        return f"REFUSED {err}"


def test_concurrent_submitters_from_two_checkouts_share_one_control_dir(tmp_path):
    """B2: two submitters (as from two checkouts: separate processes, nothing shared but the
    control directory) race the same RUN_ID: exactly one sbatch happens."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    control, sched_path = _control(tmp_path), str(tmp_path / "sched.json")
    barrier = ctx.Manager().Barrier(2)
    with ctx.Pool(2) as pool:
        got = pool.map(_race_submit, [(control, sched_path, "exp40_C16_neuronic_2x32", barrier)] * 2)
    assert sorted(g.split()[0] for g in got) == ["REFUSED", "submitted"], got
    assert sum(len(v) for v in json.load(open(sched_path)).values()) == 1
    later = _race_submit((control, sched_path, "exp40_C16_neuronic_2x32_r1", types.SimpleNamespace(wait=lambda: None)))
    assert later.startswith("REFUSED") and "live" in later


def _race_claim(args):
    claims, i, barrier = args
    kit.ALLOWED_CONTROL = os.path.dirname(claims)
    barrier.wait()
    try:
        kit.claim(claims, "exp40_C16_neuronic_2x32", "FULL", str(100 + i), f"intent{i}")
        return True
    except kit.Refused:
        return False


def test_claim_is_exclusive_under_true_concurrency(tmp_path):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    barrier = ctx.Manager().Barrier(8)
    with ctx.Pool(8) as pool:
        got = pool.map(_race_claim, [(os.path.join(_control(tmp_path), "claims"), i, barrier) for i in range(8)])
    assert sum(got) == 1


@pytest.mark.parametrize("raw,want", [("OUT_OF_ME+", "OUT_OF_MEMORY"), ("COMPLETED", "COMPLETED"),
                                      ("CANCELLED by 1", "CANCELLED"), ("PEND", "PENDING")])
def test_state_normalisation(raw, want):
    assert kit.normalise_state(raw) == want


@pytest.mark.parametrize("raw", ["C", "CO", "", "WEIRD"])
def test_state_normalisation_ambiguous_is_fatal(raw):
    with pytest.raises(kit.Refused):
        kit.normalise_state(raw)


def test_query_failure_is_fatal(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")
    with pytest.raises(kit.Refused, match="sacct failed"):
        _submit(tmp_path, _failing(sched, "sacct"), "e2")


def _failing(sched, which):
    def run(cmd):
        if cmd[0] == which:
            return 1, "", "down"
        return sched(cmd)
    return run


def test_full_gates_relaxation_probe_and_completion(tmp_path):
    sched = Sched()
    with pytest.raises(kit.Refused, match="relaxation"):
        _submit(tmp_path, sched, "R", mode="FULL", relaxation=None)
    with pytest.raises(kit.Refused, match="probe"):
        _submit(tmp_path, sched, "R", mode="FULL", probe_ok=False)
    _submit(tmp_path, sched, "R", mode="FULL")
    sched.jobs["exp40-R-FULL"] = [("1001", "COMPLETED")]
    with pytest.raises(kit.Refused):
        _submit(tmp_path, sched, "R_r1", mode="FULL")


def test_time_request_and_partition_limit():
    assert kit.parse_slurm_time("1-00:00:00") == 1440 and kit.parse_slurm_time("infinite") is None
    # M4: every sinfo line, the most conservative finite one; empty refused
    assert kit.parse_maxtime_lines("7-00:00:00\n1-00:00:00\ninfinite\n") == 1440
    assert kit.parse_maxtime_lines("infinite\nUNLIMITED\n") is None
    with pytest.raises(kit.Refused):
        kit.parse_maxtime_lines("\n")
    with pytest.raises(kit.Refused):
        kit.parse_maxtime_lines("1-00:00:00\nbogus")
    need = kit.full_time_request(2.0, None)
    assert need == int(-(-(40000 * 2.0 * 1.15 / 60 + 45) // 1))
    with pytest.raises(kit.Refused, match="MaxTime"):
        kit.full_time_request(2.0, 1440)


# --------------------------------------------------------------------------- #
# save-time NAS publication (plan §14 B2)
# --------------------------------------------------------------------------- #
def _ckpt(dirpath, step, payload=b"x"):
    p = dirpath / f"epoch=0-step={step}.ckpt"
    p.write_bytes(payload * 100)
    return p


def test_publish_saved_acknowledges_and_is_idempotent(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    man = kit.read_manifest(nas / RID / "MANIFEST.sha256")
    assert man == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))            # idempotent
    assert not list((nas / RID).glob(".*partial*"))


def test_publish_adopts_an_identical_orphan_and_refuses_a_clobber(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    (nas / RID).mkdir(parents=True)
    p = _ckpt(src, 5000)
    kit.mark_saved(str(p))
    (nas / RID / p.name).write_bytes(p.read_bytes())                            # crash orphan
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    q = _ckpt(src, 7500, b"y")
    kit.mark_saved(str(q))
    (nas / RID / q.name).write_bytes(b"different")
    with pytest.raises(kit.Refused, match="clobber"):
        kit.publish_saved(str(q), RID, timeout_s=60, dest_root=str(nas))


def test_publish_refuses_a_changed_source_and_noreplace_holds(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    p.write_bytes(b"z" * 7)
    with pytest.raises(kit.Refused):
        kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    with pytest.raises(kit.Refused, match="NOREPLACE"):
        kit.rename_noreplace(str(a), str(b))
    assert b.read_text() == "b"


def test_pulled_manifest_publication(tmp_path):
    src, nas = tmp_path / "pulled", tmp_path / "nas"
    src.mkdir()
    entries = {}
    for s in (2500, 5000):
        p = _ckpt(src, s, bytes([s % 251]))
        entries[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    kit.write_manifest(str(tmp_path / "CHECKPOINTS.partial.sha256"), entries)
    done = kit.publish_pass(str(src), RID, dest_root=str(nas),
                            pulled_manifest=str(tmp_path / "CHECKPOINTS.partial.sha256"))
    assert sorted(done) == sorted(entries)
    assert kit.read_manifest(nas / RID / "MANIFEST.sha256") == entries


def test_save_hook_fails_every_rank_when_publication_fails(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    calls = []
    monkeypatch.setattr(pl.callbacks.ModelCheckpoint, "_save_checkpoint",
                        lambda self, trainer, fp: calls.append(fp))
    fake_kit = types.SimpleNamespace(mark_saved=lambda p: None,
                                     publish_saved=lambda *a, **k: (_ for _ in ()).throw(OSError("NAS down")),
                                     record_saved_local=lambda p: None)
    monkeypatch.setitem(sys.modules, "exp40_kit", fake_kit)
    ev = drv.Evidence(str(tmp_path), verbatim=True)
    ev.bind_rank(0)
    drv.install_publication(ev, "nas")
    trainer = types.SimpleNamespace(is_global_zero=True,
                                    strategy=types.SimpleNamespace(broadcast=lambda obj, src=0: obj))
    with pytest.raises(RuntimeError, match="not acknowledged"):
        pl.callbacks.ModelCheckpoint._save_checkpoint(object(), trainer, "/x/epoch=0-step=2500.ckpt")
    assert calls == ["/x/epoch=0-step=2500.ckpt"]
    peer = types.SimpleNamespace(is_global_zero=False,
                                 strategy=types.SimpleNamespace(broadcast=lambda obj, src=0: (False, "rank0 failed")))
    with pytest.raises(RuntimeError, match="rank0 failed"):
        pl.callbacks.ModelCheckpoint._save_checkpoint(object(), peer, "/x/epoch=0-step=2500.ckpt")


# --------------------------------------------------------------------------- #
# evidence driver: attribution, observer, init parity
# --------------------------------------------------------------------------- #
def _cls(module, name="C"):
    return type(name, (), {"__module__": module, "forward": lambda self, *a: a,
                           "__call__": lambda self, *a: a})


def test_checkpoint_owner_classification():
    dino = _cls("transformers.models.dinov3_vit.modeling_dinov3_vit", "DINOv3ViTLayer")()
    dit = _cls("src.models.transformer", "TransformerBlock")()
    assert drv.checkpoint_owner_class(dino.forward) == "dinov3"
    assert drv.checkpoint_owner_class(functools.partial(dino.forward, 1)) == "dinov3"
    assert drv.checkpoint_owner_class(dit) == "dit"
    assert drv.checkpoint_owner_class(lambda x: x) == "unclassified"
    assert drv.checkpoint_owner_class(_cls("other.mod")()) == "unclassified"


def test_counted_checkpoint_separates_reentrant_and_default(monkeypatch, tmp_path):
    import torch.utils.checkpoint as tcp
    seen = []
    monkeypatch.setattr(tcp, "checkpoint", lambda f, *a, use_reentrant=None, **k: seen.append(use_reentrant) or f(*a))
    ev = drv.Evidence(str(tmp_path), verbatim=True)
    import src.models.transformer as T
    import src.training.diffusion as D
    for name in ("flash_attn_func",):
        monkeypatch.setattr(T, name, None)
    snapshot = {k: getattr(D.DiffusionCondTrainingWrapper, k) for k in
                ("_apply_yaw_aug", "on_fit_start", "on_train_end", "training_step")}
    rot = D.rotate_scene_metadata
    try:
        drv.install(ev)
        dino = _cls("transformers.models.dinov3_vit.modeling_dinov3_vit")()
        tcp.checkpoint(dino.forward, 1, use_reentrant=False)
        tcp.checkpoint(dino.forward, 1, use_reentrant=True)
        tcp.checkpoint(dino.forward, 1)
        tcp.checkpoint(lambda x: x, 1, use_reentrant=False)
        assert ev.counts["ckpt_dinov3"] == 1 and ev.counts["ckpt_reentrant"] == 1
        assert ev.counts["ckpt_default"] == 1 and ev.counts["ckpt_unclassified"] == 1
    finally:
        for k, v in snapshot.items():
            setattr(D.DiffusionCondTrainingWrapper, k, v)
        D.rotate_scene_metadata = rot


def test_loss_observer_has_no_per_step_host_sync():
    import ast
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(drv.Evidence.record_loss))).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        fn.body = fn.body[1:]                                   # drop the docstring
    code = ast.unparse(fn)
    for bad in (".item(", ".cpu(", "float(", ".tolist(", ".numpy("):
        assert bad not in code, bad


def test_loss_observer_returns_the_original_object_and_changes_nothing(tmp_path, monkeypatch):
    import src.models.transformer as T
    import src.training.diffusion as D
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()

    def run(observe):
        cap._mods()
        torch.manual_seed(5)
        w = cap.build_wrapper(m, cap.tiny_config(16))
        cap.attach_stub_trainer(w, 3, 0)
        ev = drv.Evidence(str(tmp_path / f"o{observe}"), verbatim=False)
        os.makedirs(ev.dir, exist_ok=True)
        g = torch.Generator().manual_seed(2)
        batch = (torch.randn(2, 4, 64, generator=g), [cap.make_md(m, i, 16) for i in range(2)])
        torch.manual_seed(9)
        out = D.DiffusionCondTrainingWrapper.training_step(w, batch, 0)
        if observe:
            ev.bind_rank(0)
            ev.record_loss(3, out)
        out.backward()
        return out, {n: p.grad.clone() for n, p in w.named_parameters() if p.grad is not None}, \
            torch.get_rng_state(), ev
    a, ga, ra, ev = run(True)
    b, gb, rb, _ = run(False)
    assert torch.equal(a, b) and torch.equal(ra, rb)
    assert ga.keys() == gb.keys() and all(torch.equal(ga[k], gb[k]) for k in ga)
    assert bool(ev.finite) is True and ev.counts["training_step"] == 1


def test_init_signature_parity_through_lightning(tmp_path, monkeypatch):
    """plan §14 H5: two real Lightning setups differing only in name / save dir / max steps
    (offline W&B logger + ModelCheckpoint at the production cadence) reach on_fit_start with
    identical state, parameter, RNG and Sobol signatures."""
    import pytorch_lightning as pl
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    m = cap._mods()
    sigs = []

    class Sig(pl.Callback):
        def on_fit_start(self, trainer, module):
            sigs.append((drv.state_signature(module), drv.param_signature(module), drv.rng_signature(module)))
            raise KeyboardInterrupt

    def batches():
        g = torch.Generator().manual_seed(3)
        return [(torch.randn(2, 4, 64, generator=g), [cap.make_md(m, i, 16) for i in range(2)])]

    for name, steps in (("a_smoke", 30), ("b_full", 40000)):
        pl.seed_everything(42, workers=True)
        w = cap.build_wrapper(m, cap.tiny_config(16, {"offset_step": 4}))
        logger = pl.loggers.WandbLogger(project="p", name=name, save_dir=str(tmp_path / name))
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=steps, logger=logger,
                             enable_progress_bar=False, enable_model_summary=False,
                             callbacks=[pl.callbacks.ModelCheckpoint(dirpath=str(tmp_path / name / "ck"),
                                                                     every_n_train_steps=2500, save_top_k=-1),
                                        Sig()])
        try:
            trainer.fit(w, torch.utils.data.DataLoader(batches(), batch_size=None))
        except KeyboardInterrupt:
            pass
    assert len(sigs) == 2 and sigs[0] == sigs[1]


# --------------------------------------------------------------------------- #
# rank-private evidence verdicts (synthetic, real recomputation)
# --------------------------------------------------------------------------- #
def _evidence(steps=6, world=2, micro=32):
    exp = kit.expected_stream(str(_REPO), steps, range(world), micro, verbatim=True)
    fit = {"kind": "fit_start", "world_size": world, "yaw_aug": kit.YAW_BLOCK,
           "flash_attn_func_is_none": False, "global_step": 0, "syncbn_modules": 4 if world > 1 else 0,
           "gpu_name": "NVIDIA RTX A6000", "state_signature": "s", "param_signature": "p",
           "rng_signature": {"python": "r"}, "numerics": {"x": 1}}
    recs, sums = {}, {}
    for r in range(world):
        rows = [dict(fit, rank=r)] + [{"kind": "offsets", "rank": r, "step": s, "n": micro,
                                       "offsets": list(exp[r]["verbatim"][s]), "t": float(s)} for s in range(steps)]
        recs[r] = rows
        sums[r] = {"rank": r, "counts": {"apply_yaw_aug": steps, "training_step": steps, "ckpt_reentrant": 0,
                                         "ckpt_default": 0, "ckpt_unclassified": 0, "flash_calls": 3,
                                         "ckpt_dinov3": 5, "ckpt_dit": 5, "checkpoints_published": 0},
                   "offset_digest": exp[r]["digest"], "offset_histogram": exp[r]["hist"],
                   "n_offsets": exp[r]["n"], "n_steps": steps, "steps_contiguous_from_0": True,
                   "sampled_offsets": exp[r]["sampled"], "all_losses_finite": True, "global_step": steps}
    log = "\n".join([MEM.format(r=0, m=1), MEM.format(r=1, m=2), B, STEP1])
    return recs, sums, log, exp


def test_evidence_pass():
    recs, sums, log, exp = _evidence()
    assert kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda r, s: s[1]["counts"].update(ckpt_reentrant=1), "ckpt_reentrant"),
    (lambda r, s: s[0]["counts"].update(ckpt_unclassified=2), "ckpt_unclassified"),
    (lambda r, s: s[0]["counts"].update(ckpt_dinov3=0), "ckpt_dinov3"),
    (lambda r, s: s[0]["counts"].update(flash_calls=0), "flash_calls"),
    (lambda r, s: s[1].update(all_losses_finite=False), "all_losses_finite"),
    (lambda r, s: s[0].update(offset_digest="0" * 64), "offset stream"),
    (lambda r, s: r[1][2]["offsets"].__setitem__(0, 1), "verbatim"),
    (lambda r, s: r[0][0].update(state_signature="other"), "state_signature"),
    (lambda r, s: r[1][0].update(syncbn_modules=0), "SyncBatchNorm"),
    (lambda r, s: r[0][0].update(yaw_aug={"enabled": True}), "yaw_aug"),
    (lambda r, s: s.pop(1), "ranks"),
])
def test_evidence_failures(mutate, needle):
    recs, sums, log, exp = _evidence()
    mutate(recs, sums)
    assert any(needle in x for x in kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None))


def test_evidence_reference_and_banner():
    recs, sums, log, exp = _evidence()
    ref = {"kind": "ddp_smoke", "fit_start": {str(r): dict(recs[r][0]) for r in range(2)}}
    assert kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None, ref) == []
    ref["fit_start"]["1"]["rng_signature"] = {"python": "different"}
    assert any("rng_signature" in x for x in kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None, ref))
    assert any("banner" in x for x in kit.evidence_reasons(recs, sums, log + "\n" + B, "DDP_SMOKE", "2x32", 6, exp, None))


# --------------------------------------------------------------------------- #
# full-state checkpoint audit (synthetic tiny checkpoint)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_ckpt(monkeypatch):
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()
    cfg = cap.tiny_config(16, {"offset_step": 4})
    cfg["training"]["use_ema"] = True
    w = kit.build_wrapper_cpu(str(_REPO), cfg)
    opt = torch.optim.AdamW(w.diffusion.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-3)
    params = list(w.diffusion.parameters())
    state = {i: {"step": torch.tensor(40000.0), "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
             for i, p in enumerate(params) if p.requires_grad}
    sd = w.state_dict()
    sd["diffusion_ema.initted"] = torch.tensor([1.0])
    sd["diffusion_ema.step"] = torch.tensor([40000])
    key = ("ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2500, "
           "'every_n_epochs': 0, 'train_time_interval': None}")
    ck = {"global_step": 40000, "epoch": 8, "model_config": cfg, "state_dict": sd,
          "optimizer_states": [{"state": state, "param_groups": [{"lr": 5e-05, "betas": (0.9, 0.999),
                                                                  "weight_decay": 0.001,
                                                                  "params": list(range(len(params)))}]}],
          "lr_schedulers": [dict(kit.SCHED)],
          "loops": {"fit_loop": {"epoch_loop.automatic_optimization.optim_progress":
                                 {"optimizer": {"step": {"total": {"completed": 40000}}}}}},
          "callbacks": {key: {"dirpath": "/ck"}}}
    return ck, cfg, w


def test_checkpoint_state_pass(tiny_ckpt):
    ck, cfg, w = tiny_ckpt
    assert kit.checkpoint_state_reasons(ck, cfg, "/ck", w) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda ck: ck.update(global_step=39999), "global_step"),
    (lambda ck: ck["model_config"]["training"].update(seed=1), "model_config"),
    (lambda ck: ck["lr_schedulers"][0].update(last_epoch=1), "InverseLR"),
    (lambda ck: ck["state_dict"].update({"diffusion_ema.initted": torch.tensor([0.0])}), "EMA state"),
    (lambda ck: ck["callbacks"].clear(), "callbacks"),
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].update(lr=1.0), "AdamW"),
    (lambda ck: ck["optimizer_states"][0]["state"].popitem(), "optimizer state"),
])
def test_checkpoint_state_failures(tiny_ckpt, mutate, needle):
    ck, cfg, w = tiny_ckpt
    import copy
    ck = copy.deepcopy(ck)
    mutate(ck)
    assert any(needle in x for x in kit.checkpoint_state_reasons(ck, cfg, "/ck", w))


# --------------------------------------------------------------------------- #
# sealed HF hub and the AR gates
# --------------------------------------------------------------------------- #
def _fake_hub(tmp_path, cfg=b"{}", weights=b"w"):
    src = tmp_path / "src" / kit.DINO_REPO_DIR / "snapshots" / kit.DINO_REVISION
    src.mkdir(parents=True)
    (src / "config.json").write_bytes(cfg)
    (src / "model.safetensors").write_bytes(weights)
    return tmp_path / "src"


def test_sealed_hub_build_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(kit, "DINO_CONFIG_SHA256", hashlib.sha256(b"{}").hexdigest())
    monkeypatch.setattr(kit, "DINO_SHA256", hashlib.sha256(b"w").hexdigest())
    dest = tmp_path / "sealed"
    kit.build_hf_sealed(str(_fake_hub(tmp_path)), str(dest))
    assert kit.hf_sealed_reasons(str(dest)) == []
    ref = dest / "hub" / kit.DINO_REPO_DIR / "refs" / "main"
    os.chmod(ref.parent, 0o755)
    os.chmod(ref, 0o644)
    ref.write_text("0" * 40)
    assert kit.hf_sealed_reasons(str(dest))
    with pytest.raises(kit.Refused, match="pinned"):
        kit.build_hf_sealed(str(_fake_hub(tmp_path / "bad", weights=b"corrupt")), str(tmp_path / "s2"))


def test_ar_counts_and_verify_receipt(tmp_path):
    ar = tmp_path / "AR"
    for sub in kit.AR_SUBDIRS:
        d = ar / sub / "Cafe" / "Cafe_idx_0"
        d.mkdir(parents=True)
        (d / "a.bin").write_bytes(b"1")
    man = kit.ar_manifest(str(ar), [("Cafe", "Cafe_idx_0")], content=True, workers=1)
    assert kit.ar_counts_reasons(str(ar), [("Cafe", "Cafe_idx_0")], man) == []
    (ar / "single_channel_ir_1" / "Cafe" / "Cafe_idx_0" / "b.bin").write_bytes(b"2")      # added file
    assert kit.ar_counts_reasons(str(ar), [("Cafe", "Cafe_idx_0")], man)
    (ar / "single_channel_ir_1" / "Cafe" / "Cafe_idx_0" / "b.bin").unlink()
    (ar / "metadata" / "Cafe" / "Cafe_idx_0" / "a.bin").write_bytes(b"9")                # same-size mutation
    again = kit.ar_manifest(str(ar), [("Cafe", "Cafe_idx_0")], content=True, workers=1)
    assert again != man, "a same-size content change must change the full content manifest"
    pinned = {"rooms": man, "digest": kit.digest_obj(man)}
    base = {"verdict": "PASS", "manifest_digest": kit.digest_obj(man), "got_digest": kit.digest_obj(man),
            "ar_root": os.path.realpath(str(ar)), "t": time.time(), "seconds": 1.0, "differing_rooms": []}
    r = str(tmp_path / "r.json")
    kit.atomic_write_json(r, base)
    assert kit.ar_verify_reasons(r, pinned, str(ar)) == []
    now = base["t"]
    mutations = [("t", now - 49 * 3600), ("t", now + 3600), ("t", True), ("t", "1"),
                 ("verdict", "FAIL"), ("differing_rooms", ["Cafe/Cafe_idx_0"]),
                 ("manifest_digest", "0" * 64), ("got_digest", "0" * 64), ("ar_root", "/elsewhere")]
    for key, val in mutations:
        kit.atomic_write_json(r, {**base, key: val})
        assert kit.ar_verify_reasons(r, pinned, str(ar), now=now), (key, val)
    kit.atomic_write_json(r, {**base, "extra": 1})
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)
    kit.atomic_write_json(r, {k: v for k, v in base.items() if k != "seconds"})
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)
    kit.atomic_write_json(r, base)
    assert kit.ar_verify_reasons(r, {"rooms": man, "digest": "0" * 64}, str(ar), now=now)
    (tmp_path / "r.json").write_text("{trunc")
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)


def test_driver_pin_is_enforced_on_a6000_and_recorded_on_neuronic():
    assert kit.driver_reasons("a6000", kit.DRIVER_A6000) == []
    assert kit.driver_reasons("a6000", "580.1")
    assert kit.driver_reasons("a6000", None)
    assert kit.driver_reasons("neuronic", "550.54.15") == []
    assert kit.driver_reasons("neuronic", "")


def _assets(hub="/sealed"):
    snap = os.path.join(hub, "hub", kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)
    return kit.current_assets(hub, [os.path.join(snap, "model.safetensors"), os.path.join(snap, "config.json")])


def _admission(tmp_path, **over):
    run_id = "exp40_C16_a6000_2x32"
    nas = tmp_path / "nas" / run_id
    nas.mkdir(parents=True)
    ck = {}
    for name in kit.EXPECTED_CKPTS:
        (nas / name).write_bytes(name.encode())
        ck[name] = hashlib.sha256(name.encode()).hexdigest()
    kit.write_manifest(str(nas / "MANIFEST.sha256"), ck)
    final = kit.ckpt_name(kit.ENDPOINT)
    rec = {"kind": "exp40_admission", "admitted": True, "arm": "C16", "run_id": run_id,
           "platform": "a6000", "rung": "2x32", "train_sha": "a" * 40, "runtree_digest": "1" * 64,
           "env_digest": "2" * 64, "nas_dir": str(nas), "checkpoints": ck,
           "manifest_sha256": kit.sha256_file(str(nas / "MANIFEST.sha256")),
           "ckpt_path": str(nas / final), "sha256": ck[final], "bytes": 9, "global_step": 40000,
           "model_config": kit.ARM_CONFIG, "model_config_sha256": kit.PINS[kit.ARM_CONFIG],
           "embedded_config_sha256": "3" * 64, "ema_inventory_sha256": "4" * 64,
           "strict_load": True, "full_evidence_sha256": "5" * 64, "check_sha256": "6" * 64,
           "audit_sha256": "7" * 64, "historical": dict(kit.HISTORICAL), "current_assets": _assets(),
           "trajectory_equivalence_claim_allowed": False, "note": "", "t": 1.0}
    rec.update(over)
    path = tmp_path / "admission.json"
    kit.atomic_write_json(str(path), rec)
    return str(path), rec, nas


def test_ckpt_names_are_the_lightning_epoch_step_names():
    assert kit.ckpt_name(2500) == "epoch=0-step=2500.ckpt"
    assert kit.ckpt_name(5000) == "epoch=1-step=5000.ckpt"
    assert kit.ckpt_name(40000) == "epoch=8-step=40000.ckpt"
    assert len(kit.EXPECTED_CKPTS) == 16


def test_admission_validator_accepts_the_exact_record_and_rehashes(tmp_path):
    path, rec, nas = _admission(tmp_path)
    for mode in (None, "final", "all"):
        reasons, got, sha = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=mode)
        assert reasons == [] and got == rec and sha == kit.sha256_file(path), (mode, reasons)


@pytest.mark.parametrize("key,val", [
    ("admitted", 1), ("arm", "YAW"), ("kind", "admission"), ("run_id", "exp40_C16_neuronic_2x32"),
    ("platform", "neuronic"), ("rung", "8x8"), ("train_sha", "a" * 39), ("sha256", "0" * 64),
    ("ckpt_path", "/elsewhere/epoch=8-step=40000.ckpt"), ("global_step", 40000.0), ("global_step", True),
    ("bytes", 0), ("model_config", "x.json"), ("strict_load", False), ("audit_sha256", "Z" * 64),
    ("trajectory_equivalence_claim_allowed", None), ("note", None), ("t", True)])
def test_admission_validator_rejects_every_field_mutation(tmp_path, key, val):
    path, rec, nas = _admission(tmp_path, **{key: val})
    reasons, got, _ = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=None)
    assert reasons and got is None


def test_admission_validator_rejects_schema_and_nas_mutations(tmp_path):
    path, rec, nas = _admission(tmp_path)
    root = str(tmp_path / "nas")
    kit.atomic_write_json(path, {**rec, "extra": 1})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, {k: v for k, v in rec.items() if k != "note"})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, {**rec, "checkpoints": {k: v for k, v in list(rec["checkpoints"].items())[1:]}})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, rec)
    for name in kit.EXPECTED_CKPTS:                       # every cadence file, one at a time
        p = nas / name
        old = p.read_bytes()
        p.write_bytes(old + b"x")
        assert kit.admission_reasons(path, nas_root=root, rehash="all")[0], name
        if name == kit.ckpt_name(kit.ENDPOINT):
            assert kit.admission_reasons(path, nas_root=root, rehash="final")[0]
        p.write_bytes(old)
    assert kit.admission_reasons(path, nas_root=root, rehash="all")[0] == []
    (nas / "epoch=8-step=40000.ckpt.partial").write_bytes(b"")
    assert kit.admission_reasons(path, nas_root=root, rehash="all")[0]
    (nas / "epoch=8-step=40000.ckpt.partial").unlink()
    (tmp_path / "admission.json").write_text("{trunc")
    assert kit.admission_reasons(path, nas_root=root)[0]


# --------------------------------------------------------------------------- #
# code review 1: neuronic allocated-GPU gate (H5), claim probe (B3)
# --------------------------------------------------------------------------- #
L40 = [{"index": i, "uuid": f"GPU-0000000{i}-0000-0000-0000-000000000000", "name": "NVIDIA L40",
        "free_mib": 46000} for i in range(4)]


def test_slurm_gate_inspects_only_the_allocated_devices_by_uuid():
    alloc = [L40[3]["uuid"], L40[1]["uuid"]]                  # Slurm gave physical 3 and 1
    busy0 = [{"uuid": L40[0]["uuid"], "pid": 7, "used": "100"}]  # someone on an UNallocated card
    assert kit.slurm_gpu_reasons(alloc, 2, L40, busy0) == []
    busy3 = [{"uuid": L40[3]["uuid"], "pid": 9, "used": "100"}]
    assert any("local rank 0" in r and "9" in r for r in kit.slurm_gpu_reasons(alloc, 2, L40, busy3))
    assert kit.slurm_gpu_reasons(alloc[:1], 2, L40, [])
    assert kit.slurm_gpu_reasons([alloc[0], alloc[0]], 2, L40, [])
    assert kit.slurm_gpu_reasons(["GPU-ffffffff-0000-0000-0000-000000000000", alloc[1]], 2, L40, [])
    a6 = [dict(g, name="NVIDIA RTX A6000") for g in L40]
    assert kit.slurm_gpu_reasons(alloc, 2, a6, [])


def _race_probe(args):
    d, rank = args
    return kit.probe_task(d, rank, f"host{rank}", rounds=kit.PROBE_ROUNDS, timeout_s=30)


def _probe(tmp_path):
    control = _control(tmp_path)
    d = os.path.join(control, "claims", "probe_123")
    os.makedirs(d)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(2) as pool:
        ranks = pool.map(_race_probe, [(d, 0), (d, 1)])
    script = tmp_path / "exp40_claim_probe.sbatch"
    script.write_text("#!/bin/bash\n")
    return control, d, ranks, str(script)


def test_claim_probe_barrier_rounds_and_receipt_validation(tmp_path):
    control, d, ranks, script = _probe(tmp_path)
    assert all(r["barrier_ok"] for r in ranks)
    assert all(a != b for a, b in zip(ranks[0]["won"], ranks[1]["won"]))      # one winner per round
    rec = kit.probe_verdict(d, control, "123", script, "a" * 40, 0)
    assert rec["verdict"] == "PASS", rec["reasons"]
    out = kit.control_paths(control)["probe"]
    os.makedirs(os.path.dirname(out))
    kit.atomic_write_json(out, rec)
    here = ranks[0]["mount"]
    SHA = "a" * 40
    assert kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here) == []
    for key, val in [("verdict", "FAIL"), ("hosts", ["h", "h"]), ("rounds", 19), ("winners", [0] * 19),
                     ("control_dir", "/elsewhere"), ("dir", "/elsewhere/claims/probe_1"),
                     ("mounts", {"0": {"source": "a", "fstype": "nfs"}, "1": {"source": "b", "fstype": "nfs"}}),
                     ("job_id", ""), ("source_sha", "x"), ("script_sha256", "0" * 64),
                     ("t", time.time() - 15 * 86400), ("t", time.time() + 3600), ("reasons", ["x"])]:
        kit.atomic_write_json(out, {**rec, key: val})
        assert kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here), key
    kit.atomic_write_json(out, {**rec, "extra": 1})
    assert kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here)
    kit.atomic_write_json(out, rec)
    # code-r2 B2: bound to EXPECT_SHA and to an exact, successful, CURRENT mount
    assert any("not EXPECT_SHA" in x for x in kit.probe_receipt_reasons(out, control, script, "b" * 40,
                                                                          current_mount=here))
    m0 = rec["mounts"]["0"]
    for bad, needle in [({**m0, "source": ""}, "empty or malformed"), ({**m0, "findmnt_rc": 1}, "findmnt failed"),
                        ({k: v for k, v in m0.items() if k != "dev"}, "exact schema"),
                        ({**m0, "fstype": "nfs4"}, "different mounts")]:
        kit.atomic_write_json(out, {**rec, "mounts": {"0": bad, "1": rec["mounts"]["1"]}})
        assert any(needle in x for x in kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here)), needle
    empty = {"source": "", "fstype": "", "dev": 0, "findmnt_rc": 1}
    kit.atomic_write_json(out, {**rec, "mounts": {"0": empty, "1": empty}})              # two failed records
    assert any("findmnt failed" in x for x in kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here))
    kit.atomic_write_json(out, rec)
    moved = {**here, "source": "//elsewhere/share", "fstype": "cifs"}
    assert any("now on" in x for x in kit.probe_receipt_reasons(out, control, script, SHA, current_mount=moved))
    assert kit.probe_receipt_reasons(out, control, script, SHA) == []                  # real current mount
    other = tmp_path / "other.json"
    kit.atomic_write_json(str(other), rec)
    assert any("must be" in x for x in kit.probe_receipt_reasons(str(other), control, script, SHA, current_mount=here))
    script_text = open(script).read()
    open(script, "w").write(script_text + "# edited\n")
    assert any("probe script" in x for x in kit.probe_receipt_reasons(out, control, script, SHA, current_mount=here))


def test_claim_probe_verdict_fails_on_one_host_two_winners_or_outside_dir(tmp_path):
    control, d, ranks, script = _probe(tmp_path)
    r1 = kit.read_json(os.path.join(d, "rank1.json"))
    kit.atomic_write_json(os.path.join(d, "rank1.json"), {**r1, "host": "host0"})
    assert any("not a two-node" in r for r in kit.probe_verdict(d, control, "1", script, "a" * 40, 0)["reasons"])
    r0 = kit.read_json(os.path.join(d, "rank0.json"))
    for rec, name in ((r0, "rank0.json"), (r1, "rank1.json")):           # both claim every round
        kit.atomic_write_json(os.path.join(d, name), {**rec, "won": [True] * kit.PROBE_ROUNDS})
    assert any("2 winners" in r for r in kit.probe_verdict(d, control, "1", script, "a" * 40, 0)["reasons"])
    kit.atomic_write_json(os.path.join(d, "rank0.json"), r0)
    flipped = [not w for w in r1["won"]]                                   # rank files vs claim files
    kit.atomic_write_json(os.path.join(d, "rank1.json"), {**r1, "won": flipped})
    kit.atomic_write_json(os.path.join(d, "rank0.json"), {**r0, "won": [not w for w in r0["won"]]})
    assert any("claim file names" in r for r in kit.probe_verdict(d, control, "1", script, "a" * 40, 0)["reasons"])
    kit.atomic_write_json(os.path.join(d, "rank0.json"), r0)
    kit.atomic_write_json(os.path.join(d, "rank1.json"), r1)
    assert kit.probe_verdict(d, control, "1", script, "a" * 40, 1)["verdict"] == "FAIL"     # srun rc
    elsewhere = os.path.realpath(str(tmp_path / "other_control"))
    with pytest.raises(kit.Refused, match="not THE exp_40 control directory"):
        kit.probe_verdict(d, elsewhere, "1", script, "a" * 40, 0)


# --------------------------------------------------------------------------- #
# code review 1: bounded group termination (B4), lock-record failure (H3)
# --------------------------------------------------------------------------- #
def test_terminate_group_escalates_to_sigkill_for_a_term_ignoring_group():
    proc = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 60 & wait"], start_new_session=True)
    try:
        time.sleep(0.3)
        assert kit.group_alive(proc.pid)
        assert kit.terminate_group(proc.pid, grace_s=0.5, kill_wait_s=5, poll_s=0.05) is True
        assert not kit.group_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    with pytest.raises(kit.Refused):
        kit.terminate_group(os.getpgrp())


def test_gpu_lock_record_failure_releases_the_locks(tmp_path, capsys):
    hs = tmp_path / "hs"
    hs.mkdir()
    rc = kit.main(["gpu-lock", "--uuids", U0, "--handshake-dir", str(hs), "--payload-json", "{}",
                   "--out", str(tmp_path / "missing_dir" / "locks.json")])
    assert rc == 2 and "released" in capsys.readouterr().out
    assert not (hs / f"GPU_LOCK_{U0}").exists()                                # nothing leaked
    assert kit.main(["gpu-lock", "--uuids", U0, "--handshake-dir", str(hs), "--payload-json", "{}",
                     "--out", str(tmp_path / "locks.json")]) == 0
    assert (hs / f"GPU_LOCK_{U0}").exists()
    assert kit.main(["gpu-unlock", "--lock-json", str(tmp_path / "locks.json")]) == 0


def test_launchers_install_signal_safe_cleanup_before_taking_locks():
    for name in ("exp40_train_local.sh", "exp40_train.sbatch"):
        text = (_EXP / name).read_text()
        trap = text.index("trap cleanup EXIT")
        assert "trap 'on_signal INT' INT" in text and "trap 'on_signal TERM' TERM" in text
        assert "stop-group --pgid \"$TRAIN_PID\"" in text
        if name == "exp40_train_local.sh":
            assert trap < text.index('"$PY" "$KIT" gpu-lock'), "the trap must precede lock acquisition"
            assert "gpu-unlock" in text[text.index("cleanup() {"):trap]
            assert 'bad=1' in text and "exit 12" in text
        assert trap < text.index("exec setsid")


def test_local_launcher_interrupt_releases_locks_and_kills_only_its_group(tmp_path):
    """B4 end-to-end on a stub: the launcher's cleanup block, extracted verbatim, run under a
    fake kit; SIGTERM while waiting -> the owned group is terminated and the lock released."""
    text = (_EXP / "exp40_train_local.sh").read_text()
    block = text[text.index("TRAIN_PID=\"\"; WATCH_PID=\"\""):text.index('"$PY" "$KIT" gpu-lock')]
    fake = tmp_path / "fakekit.py"
    fake.write_text(
        "import sys, os, json\n"
        "sys.path.insert(0, %r)\n" % str(_EXP) +
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('k', %r)\n" % str(_EXP / "exp40_kit.py") +
        "k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\n"
        "sys.exit(k.main(sys.argv[1:]))\n")
    hs = tmp_path / "hs"
    hs.mkdir()
    script = tmp_path / "launcher.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"PY={sys.executable}; KIT={fake}; MODE=FULL; TMPDIR={tmp_path / 'tmpdir'}; mkdir -p $TMPDIR\n"
        f"LOCK_JSON={tmp_path / 'locks.json'}\n"
        "die() { echo \"REFUSED: $*\"; exit 2; }\n" + block +
        f"\"$PY\" \"$KIT\" gpu-lock --uuids {U0} --handshake-dir {hs} --payload-json '{{}}' --out \"$LOCK_JSON\" || die lock\n"
        "( exec setsid sleep 300 ) &\nTRAIN_PID=$!\n"
        f"echo $TRAIN_PID > {tmp_path / 'pid'}\n"
        "wait \"$TRAIN_PID\"\n")
    proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        if (tmp_path / "pid").exists() and (hs / f"GPU_LOCK_{U0}").exists():
            break
        time.sleep(0.1)
    pid = int((tmp_path / "pid").read_text())
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 130, out
    assert "INTERRUPTED by SIGTERM" in out
    assert not kit.group_alive(pid)
    assert not (hs / f"GPU_LOCK_{U0}").exists() and (tmp_path / "locks_released.json").exists()
    assert not (tmp_path / "tmpdir").exists()


# --------------------------------------------------------------------------- #
# code review 1 H2: re-hash manifested destinations, orphan adoption, killable copy, NAS probe
# --------------------------------------------------------------------------- #
def test_an_already_manifested_but_mutated_nas_file_is_not_acknowledged(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    (nas / RID / p.name).write_bytes(b"rotted" * 50)
    with pytest.raises(kit.Refused, match="re-hash"):
        kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))


def test_orphan_adoption_rehashes_the_source(tmp_path, monkeypatch):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    (nas / RID).mkdir(parents=True)
    p = _ckpt(src, 5000)
    (nas / RID / p.name).write_bytes(p.read_bytes())
    real, calls = kit.sha256_file, []

    def flaky(path, *a):
        calls.append(path)
        return "0" * 64 if (path == str(p) and len(calls) > 2) else real(path, *a)
    monkeypatch.setattr(kit, "sha256_file", flaky)
    with pytest.raises(kit.Refused, match="orphan was adopted"):
        kit.publish_one(str(src), p.name, str(nas / RID))


def test_publication_timeout_kills_the_copy_process_group(tmp_path, monkeypatch):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    slow = tmp_path / "slowpy"
    slow.write_text(f"#!/bin/bash\necho $$ > {tmp_path / 'copy.pid'}\nsleep 120\n")
    slow.chmod(0o755)
    t0 = time.time()
    with pytest.raises(kit.Refused, match="exceeded"):
        kit.publish_saved(str(p), RID, timeout_s=1.0, dest_root=str(nas), python=str(slow))
    assert time.time() - t0 < 60
    pid = int((tmp_path / "copy.pid").read_text())
    assert not kit.group_alive(pid), "the timed-out publication must not keep writing"


def test_nas_capability_probe_and_receipt_validation(tmp_path):
    root = os.path.realpath(str(tmp_path))
    rec = kit.nas_capability_probe(root, size=1 << 16)
    assert rec["verdict"] == "PASS" and len(rec["checks"]) == 5, rec
    path = str(tmp_path / "nas_capability_a6000.json")
    kit.atomic_write_json(path, rec)
    assert kit.nas_receipt_reasons(path, dest_root=root) == []
    for key, val, needle in [("verdict", "FAIL", "did not PASS"), ("dest_root", "/x", "tested"),
                             ("t", time.time() - 49 * 3600, "age"), ("t", time.time() + 600, "age"),
                             ("checks", {**rec["checks"], "o_excl_create": "FAILED"}, "each True"),
                             ("checks", {f"renamed_{i}": True for i in range(5)}, "not exactly"),
                             ("mount", {**rec["mount"], "findmnt_rc": 2}, "findmnt failed"),
                             ("mount", {**rec["mount"], "source": ""}, "empty or malformed"),
                             ("mount", {"source": rec["mount"]["source"]}, "exact schema")]:
        kit.atomic_write_json(path, {**rec, key: val})
        assert any(needle in x for x in kit.nas_receipt_reasons(path, dest_root=root)), (key, needle)
    kit.atomic_write_json(path, {**rec, "more": 1})
    assert any("exact schema" in x for x in kit.nas_receipt_reasons(path, dest_root=root))
    kit.atomic_write_json(path, rec)
    moved = {**rec["mount"], "source": "//10.0.0.9/other", "fstype": "cifs"}
    assert any("now on" in x for x in kit.nas_receipt_reasons(path, dest_root=root, current_mount=moved))


def test_mount_identity_takes_the_innermost_mount_of_an_autofs_path(monkeypatch, tmp_path):
    monkeypatch.setattr(kit, "run", lambda cmd, **k: (0, "/etc/auto.nas autofs\n//10.0.0.1/share cifs\n", ""))
    m = kit.mount_identity(str(tmp_path))
    assert (m["source"], m["fstype"]) == ("//10.0.0.1/share", "cifs")


def test_atomic_write_json_leaves_no_temp_on_failure(tmp_path):
    with pytest.raises(TypeError):
        kit.atomic_write_json(str(tmp_path / "x.json"), {"bad": object()})
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
# code review 1 B5 / B6 / M6: strict FULL evidence, audit, run result, derived admission
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_stream():
    return kit.expected_stream(kit.ROOT, kit.ENDPOINT, range(2), 32, verbatim=False)


def _fit(r, world=2):
    sig = hashlib.sha256(b"state").hexdigest()
    return {"kind": "fit_start", "rank": r, "world_size": world, "local_rank": r,
            "gpu_name": "NVIDIA RTX A6000", "gpu_uuid": [U0, U1][r][4:], "syncbn_modules": 20,
            "batchnorm_modules": 20, "yaw_aug": dict(kit.YAW_BLOCK), "flash_attn_func_is_none": False,
            "state_signature": sig, "param_signature": hashlib.sha256(b"param").hexdigest(),
            "rng_signature": {k: hashlib.sha256(k.encode()).hexdigest() for k in sorted(kit.RNG_KEYS)},
            "global_step": 0,
            "numerics": {"matmul_allow_tf32": True, "cudnn_allow_tf32": True, "cudnn_benchmark": False,
                         "cudnn_deterministic": False, "float32_matmul_precision": "medium"}}


def _summary(r, e, steps, published):
    keys = sorted(str(t) for t in kit.SAMPLED_STEPS if t < steps)
    return {"rank": r, "counts": {"apply_yaw_aug": steps, "training_step": steps, "ckpt_reentrant": 0,
                                  "ckpt_default": 0, "ckpt_unclassified": 0, "flash_calls": 9,
                                  "ckpt_dinov3": 9, "ckpt_dit": 9, "checkpoints_published": published},
            "offset_digest": e["digest"], "offset_histogram": e["hist"], "n_offsets": e["n"],
            "sampled_offsets": {k: e["sampled"][k] for k in keys},
            "sampled_losses": {k: (2.5).hex() for k in keys}, "n_steps": steps, "global_step": steps,
            "steps_contiguous_from_0": True, "all_losses_finite": True, "pid": 1000 + r}


_SMOKE = {}


def _real_smoke_stream():
    if "s" not in _SMOKE:
        _SMOKE["s"] = kit.expected_stream(kit.ROOT, kit.SMOKE_STEPS["DDP_SMOKE"], range(2), 32, verbatim=False)
    return _SMOKE["s"]


@pytest.fixture(scope="module")
def smoke_stream():
    return _real_smoke_stream()


def _streams(full_stream):
    """expected_stream stand-in: the real 40k stream for FULL, the real 30-step one otherwise."""
    return lambda tree, steps, *a, **k: full_stream if steps == kit.ENDPOINT else _real_smoke_stream()


def _full_records(tmp_path, stream, run_id="exp40_C16_a6000_2x32", smoke=None):
    """The exact producer shapes: the check record, the DDP_SMOKE reference receipt and the
    FULL evidence of a 2x32 A6000 run, bound by sha256."""
    sha = "b" * 40
    argv = kit.train_argv("a6000", "2x32", "FULL", run_id, "/save")
    sargv = kit.train_argv("a6000", "2x32", "DDP_SMOKE", run_id, "/smoke", "2026-11-08_00-00-00")
    fits = {str(r): _fit(r) for r in range(2)}
    links = {"AcousticRooms": "/ar", kit.VAE: "/vae"}
    check = {"verdict": "PASS", "reasons": [], "sha": sha, "tree_digest": "1" * 64, "env_digest": "2" * 64,
             "argv": argv, "links": links}
    rec_dir = tmp_path / "receipts"
    rec_dir.mkdir(exist_ok=True)
    kit.atomic_write_json(str(rec_dir / "check.json"), check)
    kst = smoke if smoke is not None else _real_smoke_stream()
    ref = {"verdict": "PASS", "kind": "ddp_smoke", "platform": "a6000", "rung": "2x32", "mode": "DDP_SMOKE",
           "run_id": run_id, "stamp": "2026-11-08_00-00-00", "sha": sha, "tree_digest": "1" * 64,
           "env_digest": "2" * 64, "recipe_projection": kit.recipe_projection(sargv), "argv": sargv,
           "train_rc": 0, "watcher_rc": 0, "publisher_rc": 0, "reference_sha256": None,
           "evidence": {"verdict": "PASS", "reasons": []}, "fit_start": fits,
           "summaries": {str(r): _summary(r, kst[r], 30, 0) for r in range(2)}, "check_sha256": "3" * 64}
    kit.atomic_write_json(str(rec_dir / "ref.json"), ref)
    ev = {"verdict": "PASS", "kind": "full_evidence", "platform": "a6000", "rung": "2x32", "mode": "FULL",
          "run_id": run_id, "stamp": "2026-11-08_01-00-00", "sha": sha, "tree_digest": "1" * 64,
          "env_digest": "2" * 64, "recipe_projection": kit.recipe_projection(argv), "argv": argv,
          "train_rc": 0, "watcher_rc": 0, "publisher_rc": 0,
          "reference_sha256": kit.sha256_file(str(rec_dir / "ref.json")),
          "evidence": {"verdict": "PASS", "reasons": []}, "fit_start": fits,
          "summaries": {str(r): _summary(r, stream[r], 40000, 16 if r == 0 else 0) for r in range(2)},
          "check_sha256": kit.sha256_file(str(rec_dir / "check.json"))}
    kit.atomic_write_json(str(rec_dir / "ev.json"), ev)
    return rec_dir, ev


def test_the_reference_receipt_validator_is_exact(tmp_path, full_stream, smoke_stream):
    """code-r2 B3: the DDP_SMOKE / PILOT receipt -- every produced field, nested exactly."""
    d, ev = _full_records(tmp_path, full_stream, smoke=smoke_stream)
    ref = kit.read_json(str(d / "ref.json"))
    args = ("ddp_smoke", "a6000", "2x32", "b" * 40, "1" * 64, "2" * 64, ev["recipe_projection"])
    assert kit.receipt_reasons(ref, *args) == []
    import copy
    cases = [
        (lambda x: x.pop("publisher_rc"), "exact schema"), (lambda x: x.update(extra=1), "exact schema"),
        (lambda x: x.update(mode="FULL"), "mode"), (lambda x: x.update(run_id="exp40_C16_neuronic_2x32"), "run_id"),
        (lambda x: x.update(stamp=""), "stamp"), (lambda x: x.update(publisher_rc=1), "publisher_rc"),
        (lambda x: x.update(reference_sha256="0" * 64), "reference_sha256"),
        (lambda x: x["argv"].__setitem__(x["argv"].index("--max-steps") + 1, "29"), "--max-steps"),
        (lambda x: x["fit_start"]["1"].pop("gpu_uuid"), "fit_start keys"),
        (lambda x: x["fit_start"]["1"].update(gpu_uuid=x["fit_start"]["0"]["gpu_uuid"]), "same GPU"),
        (lambda x: x["fit_start"]["0"].update(gpu_name="NVIDIA L40"), "gpu_name"),
        (lambda x: x["fit_start"]["0"].update(syncbn_modules=0), "syncbn_modules"),
        (lambda x: x["fit_start"]["0"].update(batchnorm_modules=21), "syncbn_modules"),
        (lambda x: x["fit_start"]["1"].update(local_rank=0), "local_rank"),
        (lambda x: x["fit_start"]["0"].update(rank=True), "rank"),
        (lambda x: x["fit_start"]["0"]["numerics"].pop("cudnn_benchmark"), "numerics"),
        (lambda x: x["fit_start"]["0"]["yaw_aug"].update(enabled=1), "yaw_aug"),
        (lambda x: x["summaries"]["1"].pop("pid"), "summary keys"),
        (lambda x: x["summaries"]["1"].update(rank=0), "summary of rank"),
        (lambda x: x["summaries"]["0"]["counts"].pop("ckpt_dit"), "counters"),
        (lambda x: x["summaries"]["0"]["counts"].update(ckpt_dit=True), "counters"),
        (lambda x: x["summaries"]["0"]["offset_histogram"].__setitem__(0, 0), "histogram"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"].pop("3"), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"]["0"].__setitem__(0, 33), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_losses"].update({"0": "nan"}), "sampled losses"),
        (lambda x: x["summaries"]["0"]["sampled_losses"].update({"0": (float("inf")).hex()}), "sampled losses"),
    ]
    for mutate, needle in cases:
        bad = copy.deepcopy(ref)
        mutate(bad)
        got = kit.receipt_reasons(bad, *args)
        assert any(needle in x for x in got), (needle, got)
    pilot = dict(copy.deepcopy(ref), kind="pilot", mode="PILOT", platform="neuronic", run_id="exp40_C16_neuronic_2x32")
    for v in (None, 0, -1.0, float("nan"), True, "2.0"):
        bad = dict(pilot, sec_per_step=v)
        assert any("sec_per_step" in x or "rate" in x for x in
                   kit.receipt_reasons(bad, "pilot", "neuronic", "2x32", "b" * 40, "1" * 64, "2" * 64,
                                       ev["recipe_projection"])), v


def test_full_evidence_validator_accepts_the_bound_record(tmp_path, full_stream, monkeypatch):
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    d, ev = _full_records(tmp_path, full_stream)
    r, sha, rec = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"),
                                            str(d / "ref.json"), recompute=True)
    assert r == [] and rec == ev and sha == kit.sha256_file(str(d / "ev.json"))


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.update(verdict="FAIL"), "verdict"), (lambda e: e.update(kind="ddp_smoke"), "kind"),
    (lambda e: e.update(mode="PILOT"), "mode"), (lambda e: e.update(platform="neuronic"), "platform"),
    (lambda e: e.update(rung="8x8"), "rung"), (lambda e: e.update(sha="z"), "40-hex"),
    (lambda e: e.update(tree_digest="x"), "tree_digest"), (lambda e: e.update(train_rc=1), "train_rc"),
    (lambda e: e.update(watcher_rc=False), "watcher_rc"), (lambda e: e.update(publisher_rc=3), "publisher_rc"),
    (lambda e: e.update(evidence={"verdict": "PASS", "reasons": ["x"]}), "evidence"),
    (lambda e: e["argv"].__setitem__(e["argv"].index("--seed") + 1, "43"), "recipe projection"),
    (lambda e: e["summaries"]["0"]["counts"].update(checkpoints_published=15), "checkpoints_published"),
    (lambda e: e["summaries"]["1"]["counts"].update(checkpoints_published=1), "checkpoints_published"),
    (lambda e: e["summaries"]["1"]["counts"].update(ckpt_reentrant=1), "ckpt_reentrant"),
    (lambda e: e["summaries"]["0"]["counts"].update(flash_calls=0), "flash_calls"),
    (lambda e: e["summaries"]["0"]["counts"].update(apply_yaw_aug=39999), "apply_yaw_aug"),
    (lambda e: e["summaries"]["1"].update(all_losses_finite=None), "all_losses_finite"),
    (lambda e: e["summaries"]["0"].update(steps_contiguous_from_0=False), "contiguous"),
    (lambda e: e["summaries"]["1"].update(offset_digest="0" * 64), "recomputation"),
    (lambda e: e["summaries"].pop("1"), "exactly ranks"),
    (lambda e: e["fit_start"]["0"].update(yaw_aug={"enabled": True}), "yaw_aug"),
    (lambda e: e["fit_start"]["1"].update(rng_signature={"python": "other"}), "rng_signature"),
    (lambda e: e.update(reference_sha256="0" * 64), "reference receipt is not the one"),
    (lambda e: e.update(check_sha256="0" * 64), "check record is not the one"),
    (lambda e: e.update(extra=1), "exact schema"), (lambda e: e.pop("stamp"), "exact schema"),
    (lambda e: e.update(stamp="x"), "stamp"),
    (lambda e: e["fit_start"]["0"].pop("gpu_uuid"), "fit_start keys"),
    (lambda e: e["fit_start"]["1"].pop("batchnorm_modules"), "fit_start keys"),
    (lambda e: e["fit_start"]["0"].update(gpu_name="NVIDIA L40"), "gpu_name"),
    (lambda e: e["fit_start"]["1"].update(rank=0), "fit_start rank"),
    (lambda e: e["summaries"]["0"].pop("sampled_losses"), "summary keys"),
    (lambda e: e["summaries"]["1"].pop("rank"), "summary keys"),
    (lambda e: e["summaries"]["0"]["counts"].update(extra_counter=0), "counters"),
])
def test_full_evidence_validator_rejects_every_critical_mutation(tmp_path, full_stream, monkeypatch, mutate, needle):
    import copy
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    d, ev = _full_records(tmp_path, full_stream)
    bad = copy.deepcopy(ev)
    mutate(bad)
    kit.atomic_write_json(str(d / "ev.json"), bad)
    r, _, rec = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"),
                                          str(d / "ref.json"), recompute=True)
    assert rec is None and any(needle in x for x in r), (needle, r)


def test_full_evidence_binding_to_check_and_reference(tmp_path, full_stream):
    d, ev = _full_records(tmp_path, full_stream)
    chk = kit.read_json(str(d / "check.json"))
    kit.atomic_write_json(str(d / "check.json"), {**chk, "env_digest": "9" * 64})
    assert kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"))[0]
    kit.atomic_write_json(str(d / "check.json"), chk)
    ref = kit.read_json(str(d / "ref.json"))
    kit.atomic_write_json(str(d / "ref.json"), {**ref, "tree_digest": "9" * 64})
    assert kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], None, str(d / "ref.json"))[0]
    assert kit.full_evidence_reasons(str(d / "ev.json"), "exp40_C16_a6000_2x32_r1")[0]


@pytest.fixture(scope="module")
def sealed_hub(tmp_path_factory):
    """A real sealed hub built from this box's HF cache (code-r3 B1 tests)."""
    src = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(os.path.join(src, kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)):
        pytest.skip("the pinned DINOv3 snapshot is not in this box's HF cache")
    return kit.build_hf_sealed(src, str(tmp_path_factory.mktemp("sealed") / "hf_sealed"))


def _inject(tree, w, sealed):
    """In-process stand-ins for the admission tree and its subprocess (the real ones are tested
    separately): the tiny wrapper, interpreted from ``tree``; DINOv3 'resolved' in ``sealed``."""
    def builder(repo, sha, check_rec, ev, parent):
        assert check_rec["links"] and sha == ev["sha"]
        return str(tree)

    def state(t, ckpt, dirpath, strict_load=True, hf_sealed=None):
        assert t == str(tree) and hf_sealed == sealed
        st = kit.checkpoint_admission_state(ckpt, dirpath, t, strict_load=strict_load)
        snap = os.path.join(os.path.realpath(sealed), "hub", kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)
        st.update(hf_sealed=os.path.realpath(sealed),
                  dino_resolved=[os.path.join(snap, "model.safetensors"), os.path.join(snap, "config.json")])
        return st
    return {"tree_builder": builder, "state_fn": state, "hf_sealed": sealed}


def _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream):
    """16 cadence checkpoints (the 40k one a real tiny full-state checkpoint), save markers,
    NAS publication through the real protocol, and the bound evidence records."""
    ck, cfg, w = tiny_ckpt
    tree = tmp_path / "tree"
    (tree / kit.EXPREL).mkdir(parents=True)
    (tree / kit.ARM_CONFIG).write_text(json.dumps(cfg))
    (tree / kit.AR_MANIFEST).write_bytes((_REPO / kit.AR_MANIFEST).read_bytes())
    monkeypatch.setattr(kit, "build_wrapper_cpu", lambda t, c: w)
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    run_id = "exp40_C16_a6000_2x32"
    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    for n in kit.EXPECTED_CKPTS:
        if n == kit.ckpt_name(kit.ENDPOINT):
            torch.save(ck, str(ckdir / n))
        else:
            (ckdir / n).write_bytes(n.encode() * 10)
        kit.mark_saved(str(ckdir / n))
    nas_root = os.path.realpath(str(tmp_path / "nas"))
    for n in kit.EXPECTED_CKPTS:
        kit.publish_pass(str(ckdir), run_id, dest_root=nas_root, only=[n])
    d, ev = _full_records(tmp_path, full_stream, run_id)
    return tree, ckdir, nas_root, d, ev


def test_full_audit_and_derived_admission_end_to_end(tmp_path, tiny_ckpt, monkeypatch, full_stream, sealed_hub):
    tree, ckdir, nas_root, d, ev = _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream)
    nas = os.path.join(nas_root, ev["run_id"])
    v = kit.audit(str(ckdir), str(tree), nas_dir=nas, evidence=str(d / "ev.json"), run_id=ev["run_id"],
                  expected_dirpath="/ck", check_path=str(d / "check.json"), reference_path=str(d / "ref.json"))
    assert v["verdict"] == "PASS", v["reasons"]
    kit.atomic_write_json(str(d / "audit.json"), v)
    rr = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", {"check": str(d / "check.json"), "reference": str(d / "ref.json"),
                                                         "evidence": str(d / "ev.json"), "audit": str(d / "audit.json")},
                        "/ck", str(ckdir), str(tree), nas_root=nas_root)
    kit.atomic_write_json(str(d / "run_result.json"), rr)
    rec = kit.admit(str(d / "run_result.json"), str(d), str(_REPO), note="test", nas_root=nas_root,
                    **_inject(tree, tiny_ckpt[2], sealed_hub))
    assert set(rec) == kit.ADMISSION_KEYS
    assert (rec["train_sha"], rec["runtree_digest"], rec["env_digest"]) == (ev["sha"], "1" * 64, "2" * 64)
    kit.atomic_write_json(str(tmp_path / "admission.json"), rec)
    assert kit.admission_reasons(str(tmp_path / "admission.json"), nas_root=nas_root, rehash="all")[0] == []
    # every NAS checkpoint, mutated one at a time, blocks admission
    for n in kit.EXPECTED_CKPTS:
        p = os.path.join(nas, n)
        os.chmod(p, 0o644)
        old = open(p, "rb").read()
        open(p, "wb").write(old + b"!")
        with pytest.raises(kit.Refused):
            kit.admit(str(d / "run_result.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
        open(p, "wb").write(old)
    open(os.path.join(nas, ".epoch=8-step=40000.ckpt.partial-1"), "wb").write(b"")
    with pytest.raises(kit.Refused, match="beyond"):
        kit.admit(str(d / "run_result.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))


def test_admission_refuses_unbound_or_altered_records(tmp_path, tiny_ckpt, monkeypatch, full_stream, sealed_hub):
    tree, ckdir, nas_root, d, ev = _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream)
    nas = os.path.join(nas_root, ev["run_id"])
    v = kit.audit(str(ckdir), str(tree), nas_dir=nas, evidence=str(d / "ev.json"), run_id=ev["run_id"],
                  expected_dirpath="/ck", check_path=str(d / "check.json"), reference_path=str(d / "ref.json"))
    kit.atomic_write_json(str(d / "audit.json"), v)
    files = {"check": str(d / "check.json"), "reference": str(d / "ref.json"), "evidence": str(d / "ev.json"),
             "audit": str(d / "audit.json")}
    rr = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", files, "/ck",
                        str(ckdir), str(tree), nas_root=nas_root)
    kit.atomic_write_json(str(d / "rr.json"), rr)
    kit.atomic_write_json(str(d / "audit.json"), {**v, "full_evidence_sha256": "0" * 64})
    with pytest.raises(kit.Refused, match="not the recorded bytes"):           # run result binds the audit
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    kit.atomic_write_json(str(d / "rr.json"), {**rr, "verdict": "FAIL"})
    with pytest.raises(kit.Refused, match="verdict"):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    kit.atomic_write_json(str(d / "rr.json"), {**rr, "platform": "neuronic"})
    with pytest.raises(kit.Refused):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    with pytest.raises(kit.Refused, match="note"):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), note={"admitted": False}, nas_root=nas_root,
                  **_inject(tree, tiny_ckpt[2], sealed_hub))


def test_audit_refuses_wrong_names_markers_and_strays(tmp_path):
    for s in kit.CKPT_STEPS:
        (tmp_path / kit.ckpt_name(s)).write_bytes(b"x")
        kit.mark_saved(str(tmp_path / kit.ckpt_name(s)))
    (tmp_path / "epoch=0-step=2500.ckpt.saved").unlink()
    (tmp_path / "stray.txt").write_text("x")
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False)
    assert v["verdict"] == "FAIL"
    assert any("save marker" in r for r in v["reasons"]) and any("unexpected" in r for r in v["reasons"])
    os.rename(tmp_path / kit.ckpt_name(5000), tmp_path / "epoch=0-step=5000.ckpt")   # wrong epoch
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False)
    assert any("missing" in r for r in v["reasons"])


# --------------------------------------------------------------------------- #
# code review 2: admission interprets the run in an immutable tree at the training commit (H2)
# --------------------------------------------------------------------------- #
def _head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_admission_tree_reproduces_the_training_tree_digest_or_refuses(tmp_path):
    sha = _head()
    links = {"AcousticRooms": "/ar/at/training/time", kit.VAE: "/vae/at/training/time"}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": "exp40_C16_a6000_2x32", "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    assert kit.verify_runtree(str(_REPO), sha, tree, {r: os.path.realpath(os.path.join(_REPO, r))
                                                       for r in kit.LINKS}) == []
    assert open(os.path.join(tree, kit.EXPREL, "exp40_kit.py"), "rb").read() == subprocess.run(
        ["git", "show", f"{sha}:{kit.EXPREL}/exp40_kit.py"], cwd=_REPO, capture_output=True, check=True).stdout
    with pytest.raises(kit.Refused, match="run-tree digest"):
        kit.build_admission_tree(str(_REPO), sha, {"links": links}, dict(ev, tree_digest="0" * 64), str(tmp_path))
    with pytest.raises(kit.Refused, match="training-time links"):
        kit.build_admission_tree(str(_REPO), sha, {}, ev, str(tmp_path))


def _require_head_kit(sha):
    """The admission tree runs HEAD's COMMITTED kit: these tests need it to be this revision."""
    if b"def dino_resolved_paths" not in subprocess.run(["git", "show", f"{sha}:{kit.EXPREL}/exp40_kit.py"],
                                                         cwd=_REPO, capture_output=True, check=True).stdout:
        pytest.skip("HEAD's committed kit predates the sealed-hub admission-state (run after committing)")


def test_admission_state_runs_in_the_admission_trees_own_kit(tmp_path, tiny_ckpt, sealed_hub):
    sha = _head()
    _require_head_kit(sha)
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": "exp40_C16_a6000_2x32", "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    ck, cfg, w = tiny_ckpt
    torch.save(ck, str(tmp_path / "tiny.ckpt"))
    st = kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False,
                                        hf_sealed=sealed_hub)
    assert st["tree"] == os.path.realpath(tree) and st["sha256"] == kit.sha256_file(str(tmp_path / "tiny.ckpt"))
    assert any("model_config" in r for r in st["reasons"])     # the tiny config is not the C16 arm config


def test_an_already_manifested_name_rehashes_the_source_too(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_pass(str(src), RID, dest_root=str(nas), only=[p.name])
    p.write_bytes(b"y" * len(p.read_bytes()))                          # same-size source mutation
    with pytest.raises(kit.Refused):
        kit.publish_pass(str(src), RID, dest_root=str(nas), only=[p.name])


def _pilot(tmp_path, full_stream, sec=2.0):
    import copy
    run_id = "exp40_C16_neuronic_2x32"
    d, ev = _full_records(tmp_path, full_stream)
    ref = copy.deepcopy(kit.read_json(str(d / "ref.json")))
    argv = kit.train_argv("neuronic", "2x32", "PILOT", run_id, "/s", "2026-11-08_00-00-00")
    for r in ("0", "1"):
        ref["fit_start"][r]["gpu_name"] = "NVIDIA L40"
        e = full_stream[int(r)]
        ref["summaries"][r] = _summary(int(r), dict(e, n=300 * 32, hist=[600] * 16), 300, 0)
    ref.update(kind="pilot", mode="PILOT", platform="neuronic", run_id=run_id, argv=argv,
               recipe_projection=kit.recipe_projection(argv), sec_per_step=sec,
               pilot_window={"rank": 0, "steps": [100, 299], "t": [1000.0, 1000.0 + 199 * sec]})
    path = str(tmp_path / "pilot.json")
    kit.atomic_write_json(path, ref)
    return path, ref


def _time_setup(tmp_path, full_stream):
    """A prospective FULL run tree at HEAD, the PILOT's bound check record (environment =
    what this interpreter measures now) and an exact PILOT receipt of HEAD."""
    sha = _head()
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    tree = str(tmp_path / "fulltree")
    kit.build_runtree(str(_REPO), sha, tree, links)
    light = kit.measure_environment_light(sys.executable)
    env = {**light, "cuda_runtime": "12.4", "cudnn": 90100, "nccl": [2, 21, 5], "origins": {}}
    chk = {"verdict": "PASS", "reasons": [], "sha": sha, "tree_digest": kit.tree_digest(str(_REPO), sha, links),
           "environment": env, "driver_version": "550.54.15", "env_digest": kit.environment_digest(env, "550.54.15")}
    chk_path = str(tmp_path / "pilot_check.json")
    kit.atomic_write_json(chk_path, chk)
    path, ref = _pilot(tmp_path, full_stream)
    ref.update(sha=sha, tree_digest=chk["tree_digest"], env_digest=chk["env_digest"],
               check_sha256=kit.sha256_file(chk_path))
    kit.atomic_write_json(path, ref)
    return sha, links, tree, chk_path, chk, path, ref


def _time(path, chk_path, sha, rung, run_id, tree, links, maxtime="7-00:00:00\n"):
    return kit.main(["neuronic-time", "--mode", "FULL", "--receipt", path, "--pilot-check", chk_path,
                     "--expect-sha", sha, "--rung", rung, "--run-id", run_id, "--tree", tree,
                     *sum((["--link", f"{k}={v}"] for k, v in links.items()), []), "--maxtime-raw", maxtime])


def test_neuronic_time_request_binds_independent_identity(tmp_path, full_stream, capsys, monkeypatch):
    """code-r3 H3: rung, RUN_ID, the prospective tree, the PILOT check record's environment and
    the interpreter measured now are supplied independently of the receipt."""
    monkeypatch.setitem(kit.PLATFORMS["neuronic"], "python", sys.executable)
    sha, links, tree, chk_path, chk, path, ref = _time_setup(tmp_path, full_stream)
    R = "exp40_C16_neuronic_2x32"
    assert kit.neuronic_time_reasons(path, chk_path, sha, "2x32", R, tree, links, python=sys.executable)[0] == []
    assert _time(path, chk_path, sha, "2x32", R, tree, links) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(kit.full_time_request(2.0, 7 * 1440))
    assert kit.neuronic_time_reasons(path, chk_path, sha, "2x32", R + "_r1", tree, links, python=sys.executable)[0] == []

    def refused(needle, **kw):
        args = dict(receipt=path, check=chk_path, sha=sha, rung="2x32", run_id=R, tree=tree, links=links)
        args.update(kw)
        got, _ = kit.neuronic_time_reasons(args["receipt"], args["check"], args["sha"], args["rung"], args["run_id"],
                                           args["tree"], args["links"], python=sys.executable)
        assert any(needle in x for x in got), (needle, got)
    refused("selects '8x8'", rung="8x8")
    refused("is not the PILOT's", run_id="exp40_C16_neuronic_2x32_x")
    refused("prospective FULL run tree", links={**links, "AcousticRooms": "/elsewhere"})
    other = str(tmp_path / "othertree")
    kit.build_runtree(str(_REPO), sha, other, {**links, "AcousticRooms": os.path.realpath(str(tmp_path))})
    refused("digest differs", tree=other, links={**links, "AcousticRooms": os.path.realpath(str(tmp_path))})
    kit.atomic_write_json(chk_path, {**chk, "env_digest": "0" * 64})
    refused("bound to")                                                     # the check record changed
    bad_env = {**chk["environment"], "dists": {**chk["environment"]["dists"], "torch": "2.6.0"}}
    kit.atomic_write_json(chk_path, {**chk, "environment": bad_env, "env_digest": kit.environment_digest(bad_env, "550.54.15")})
    kit.atomic_write_json(path, {**ref, "check_sha256": kit.sha256_file(chk_path),
                                 "env_digest": kit.environment_digest(bad_env, "550.54.15")})
    refused("distribution torch now")
    kit.atomic_write_json(chk_path, chk)
    kit.atomic_write_json(path, {**ref, "sec_per_step": 1.0})
    refused("derived from its timing window")                               # a rate its timestamps contradict
    kit.atomic_write_json(path, {**ref, "pilot_window": {**ref["pilot_window"], "steps": [0, 299]}})
    refused("timing window")
    kit.atomic_write_json(path, ref)
    refused("not a PASS at EXPECT_SHA", sha="c" * 40)
    assert kit.main(["neuronic-time", "--mode", "PILOT", "--maxtime-raw", "1:00:00\n"]) == 2  # 120 > 60 min


# --------------------------------------------------------------------------- #
# code review 3: RNG schema, balanced histogram corruption, recomputed smoke stream (H1)
# --------------------------------------------------------------------------- #
def test_reference_receipt_rng_schema_bins_and_recomputed_stream(tmp_path, full_stream, smoke_stream):
    import copy
    d, ev = _full_records(tmp_path, full_stream, smoke=smoke_stream)
    ref = kit.read_json(str(d / "ref.json"))
    args = ("ddp_smoke", "a6000", "2x32", "b" * 40, "1" * 64, "2" * 64, ev["recipe_projection"])
    assert kit.receipt_reasons(ref, *args) == [] and kit.receipt_stream_reasons(ref, kit.ROOT, "DDP_SMOKE", "2x32") == []
    for mutate in (lambda f: f.pop("sobol"), lambda f: f.update(extra="0" * 64), lambda f: f.update(numpy="x")):
        bad = copy.deepcopy(ref)
        mutate(bad["fit_start"]["0"]["rng_signature"])
        assert any("rng_signature" in x for x in kit.receipt_reasons(bad, *args))
    bad = copy.deepcopy(ref)
    h = bad["summaries"]["0"]["offset_histogram"]
    h[0], h[1] = -1, h[1] + h[0] + 1                                         # balanced: the sum is unchanged
    assert any("histogram" in x for x in kit.receipt_reasons(bad, *args))
    bad = copy.deepcopy(ref)                                                  # a different VALID stream: rank 1's
    bad["summaries"]["0"] = dict(bad["summaries"]["1"], rank=0)
    assert kit.receipt_reasons(bad, *args) == []
    assert any("recomputation" in x for x in kit.receipt_stream_reasons(bad, kit.ROOT, "DDP_SMOKE", "2x32"))
    bad = copy.deepcopy(ref)
    bad["summaries"]["1"]["sampled_offsets"]["2"] = list(reversed(bad["summaries"]["1"]["sampled_offsets"]["2"]))
    assert any("recomputation" in x for x in kit.receipt_stream_reasons(bad, kit.ROOT, "DDP_SMOKE", "2x32"))
    # and the FULL validator recomputes the reference's stream when asked to
    kit.atomic_write_json(str(d / "ref.json"), bad)
    ev2 = kit.read_json(str(d / "ev.json"))
    kit.atomic_write_json(str(d / "ev.json"), {**ev2, "reference_sha256": kit.sha256_file(str(d / "ref.json"))})
    r, _, _ = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"), str(d / "ref.json"),
                                        recompute=True, tree=kit.ROOT)
    assert any("DDP_SMOKE receipt rank 1" in x for x in r), r


# --------------------------------------------------------------------------- #
# code review 3: the probe runs the spooled script from an immutable tree (H2)
# --------------------------------------------------------------------------- #
def test_probe_preflight_refuses_queued_checkout_drift(tmp_path):
    sha = _head()
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    tree = str(tmp_path / "probetree")
    kit.build_runtree(str(_REPO), sha, tree, links)
    committed = os.path.join(tree, kit.EXPREL, "exp40_claim_probe.sbatch")
    spooled = tmp_path / "slurm_script"
    spooled.write_bytes(open(committed, "rb").read())
    digest = kit.sha256_file(str(spooled))
    assert kit.probe_preflight_reasons(str(_REPO), sha, tree, str(spooled), digest) == []
    drifted = tmp_path / "slurm_script_drifted"                               # the checkout moved meanwhile
    drifted.write_bytes(open(committed, "rb").read() + b"\n# a later revision\n")
    got = kit.probe_preflight_reasons(str(_REPO), sha, tree, str(drifted), kit.sha256_file(str(drifted)))
    assert any("executed probe script" in x for x in got), got
    got = kit.probe_preflight_reasons(str(_REPO), sha, tree, str(spooled), "0" * 64)
    assert any("recorded at submission" in x for x in got), got
    parent = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=_REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    assert any("probe tree" in x for x in kit.probe_preflight_reasons(str(_REPO), parent, tree, str(spooled), digest))
    text = (_EXP / "exp40_claim_probe.sbatch").read_text()
    assert 'probe-preflight' in text and '--script-self "$0"' in text and '--script "$0"' in text
    assert '"$TREE/$EXPREL/exp40_kit.py"' in text.replace("KIT=", "") or 'KIT="$TREE/$EXPREL/exp40_kit.py"' in text


# --------------------------------------------------------------------------- #
# code review 3: admission resolves DINOv3 ONLY from the sealed hub (B1); provenance (M2)
# --------------------------------------------------------------------------- #
def test_sealed_env_removes_every_competing_hf_variable():
    base = {"HF_HOME": "/home/x/.cache/hf", "HF_HUB_CACHE": "/c", "HUGGINGFACE_HUB_CACHE": "/d",
            "TRANSFORMERS_CACHE": "/e", "HF_DATASETS_CACHE": "/f", "PYTHONPATH": "/p", "PATH": "/bin"}
    e = kit.sealed_env("/sealed", base)
    assert e["HF_HOME"] == "/sealed" and e["HF_HUB_CACHE"] == "/sealed/hub" and e["HF_HUB_OFFLINE"] == "1"
    assert e["TRANSFORMERS_OFFLINE"] == "1" and e["PATH"] == "/bin" and "PYTHONPATH" not in e
    assert not {"HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"} & set(e)


def test_admission_state_resolves_dinov3_only_under_the_sealed_hub(tmp_path, tiny_ckpt, sealed_hub, monkeypatch):
    sha = _head()
    _require_head_kit(sha)
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": "exp40_C16_a6000_2x32", "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    ck, cfg, w = tiny_ckpt
    torch.save(ck, str(tmp_path / "tiny.ckpt"))
    decoy = tmp_path / "decoy_hf"                                              # a competing ordinary cache
    decoy.mkdir()
    monkeypatch.setenv("HF_HOME", str(decoy))
    monkeypatch.setenv("HF_HUB_CACHE", str(decoy / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(decoy / "hub"))
    st = kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False, hf_sealed=sealed_hub)
    hub = os.path.join(os.path.realpath(sealed_hub), "hub") + os.sep
    assert len(st["dino_resolved"]) == 2 and all(p.startswith(hub) for p in st["dino_resolved"]), st
    assert st["hf_sealed"] == os.path.realpath(sealed_hub) and st["sha256"] == kit.sha256_file(str(tmp_path / "tiny.ckpt"))
    assert kit.current_assets_reasons(kit.current_assets(sealed_hub, st["dino_resolved"])) == []
    # in a process whose environment is NOT the sealed hub, admission-state refuses before loading
    got = kit.admission_state(str(tmp_path / "tiny.ckpt"), "/ck", tree=tree, strict_load=False, hf_sealed=sealed_hub)
    assert got["sha256"] is None and any("sealed hub" in x or "competing" in x for x in got["reasons"]), got
    with pytest.raises(kit.Refused, match="sealed HF hub"):
        kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False)


def test_admit_requires_a_valid_sealed_hub(tmp_path):
    with pytest.raises(kit.Refused, match="--hf-sealed"):
        kit.admit("rr.json", str(tmp_path), str(_REPO), hf_sealed=None)
    (tmp_path / "not_sealed" / "hub").mkdir(parents=True)
    with pytest.raises(kit.Refused, match="sealed HF hub"):
        kit.admit("rr.json", str(tmp_path), str(_REPO), hf_sealed=str(tmp_path / "not_sealed"))
    with pytest.raises(SystemExit):
        kit.main(["admit", "--run-result", "x", "--receipts-dir", "y", "--out", "z"])     # flag is mandatory


@pytest.mark.parametrize("key,val,needle", [
    ("historical", {}, "historical"), ("historical", {**kit.HISTORICAL, "exp17_launch_commit": "41aa31dc"}, "historical"),
    ("current_assets", {}, "current_assets keys"),
    ("dinov3_revision", "0" * 40, "dinov3_revision"), ("dinov3_config_sha256", "0" * 64, "dinov3_config_sha256"),
    ("dinov3_weights_sha256", "0" * 64, "dinov3_weights_sha256"), ("vae_sha256", "0" * 64, "vae_sha256"),
    ("acousticrooms_manifest_digest", "0" * 64, "acousticrooms_manifest_digest"),
    ("hf_sealed_hub", "relative/hub", "hf_sealed_hub"), ("dinov3_resolved", ["/elsewhere/model.safetensors"], "dinov3_resolved"),
    ("t", time.time() + 3600, "non-future"), ("t", float("nan"), "non-future"), ("t", 0, "non-future"),
])
def test_admission_validator_checks_provenance_fields(tmp_path, key, val, needle):
    if key in kit.CURRENT_ASSET_KEYS:
        path, rec, nas = _admission(tmp_path, current_assets={**_assets(), key: val})
    else:
        path, rec, nas = _admission(tmp_path, **{key: val})
    reasons, got, _ = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=None)
    assert got is None and any(needle in x for x in reasons), (needle, reasons)
