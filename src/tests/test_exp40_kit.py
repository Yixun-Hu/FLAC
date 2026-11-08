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
    p = kit.claim(str(tmp_path), "exp40_C16_neuronic_2x32", "FULL", "101", "intentA")
    assert os.path.basename(p) == "exp40_C16_neuronic_2x32__FULL.claim" and "intentA" not in p
    with pytest.raises(kit.Refused, match="must not train"):
        kit.claim(str(tmp_path), "exp40_C16_neuronic_2x32", "FULL", "102", "intentB")


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


def _submit(states, sched, exec_id, mode="PILOT", **kw):
    kw.setdefault("relaxation", "post-run-publish")
    kw.setdefault("probe_ok", True)
    return kit.neuronic_submit(str(states), "exp40_C16_neuronic_2x32", mode, exec_id, _argv, sched,
                               grace_s=0, sleep=lambda s: None, **kw)


def test_submit_records_the_attempt_boundary(tmp_path):
    sched = Sched()
    action, st = _submit(tmp_path, sched, "e1")
    assert action == "submitted" and st["job_ids"] == ["1001"]
    rec = kit.read_json(tmp_path / "e1__PILOT.state.json")
    assert rec["status"] == "submitted" and rec["sbatch"][3] == "exp40-e1-PILOT"


def test_live_or_ambiguous_execution_blocks_any_new_one(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")                                             # PENDING
    with pytest.raises(kit.Refused, match="live"):
        _submit(tmp_path, sched, "e2")
    sched.jobs["exp40-e1-PILOT"] = []                                          # vanished: ambiguous
    with pytest.raises(kit.Refused, match="ambiguous"):
        _submit(tmp_path, sched, "e2")


def test_post_sbatch_crash_is_adopted_not_resubmitted(tmp_path):
    sched = Sched()
    kit.atomic_write_json(tmp_path / "e1__PILOT.state.json",
                          {"run_id": "exp40_C16_neuronic_2x32", "mode": "PILOT", "execution_id": "e1",
                           "status": "sbatch_attempted", "intent": "x", "job_ids": []})
    sched.jobs["exp40-e1-PILOT"] = [("777", "RUNNING")]                        # sbatch DID happen
    with pytest.raises(kit.Refused, match="live"):
        _submit(tmp_path, sched, "e2")
    assert not [c for c in sched.calls if c[0] == "sbatch"]


def test_delayed_accounting_then_completed(tmp_path):
    sched = Sched()
    _submit(tmp_path, sched, "e1")
    sched.jobs["exp40-e1-PILOT"] = [("1001", "COMPLETED")]
    sched.hidden.add("exp40-e1-PILOT")
    action, _ = _submit(tmp_path, sched, "e2")                                 # grace re-query finds it
    assert action == "submitted"
    with pytest.raises(kit.Refused, match="fresh execution identity"):
        sched.jobs["exp40-e2-PILOT"] = [("1002", "FAILED")]
        _submit(tmp_path, sched, "e2")


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
        _submit(tmp_path, Sched.__new__(Sched) if False else _failing(sched, "sacct"), "e2")


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


def test_cadence_and_save_markers(tmp_path):
    for s in kit.CKPT_STEPS:
        (tmp_path / f"epoch=0-step={s}.ckpt").write_bytes(b"x")
        (tmp_path / f"epoch=0-step={s}.ckpt.saved").write_text("{}")
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False) if False else None
    names = sorted(os.listdir(tmp_path))
    steps = kit.ckpt_steps(n for n in names if kit._CKPT_RE.match(n))
    assert sorted(steps) == list(kit.CKPT_STEPS)
    (tmp_path / "epoch=0-step=2500.ckpt.saved").unlink()
    (tmp_path / "stray.txt").write_text("x")
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False)
    assert v["verdict"] == "FAIL"
    assert any("save marker" in r for r in v["reasons"]) or any("unexpected" in r for r in v["reasons"])


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
           "audit_sha256": "7" * 64, "historical": {}, "current_assets": {},
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
