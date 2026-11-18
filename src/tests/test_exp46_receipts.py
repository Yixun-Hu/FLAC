"""exp_46 (forked from test_exp38_receipts.py) — the receipts every gate consumes (exp46_registry.py; Codex code r1 BLOCKING 1, 2, 4, HIGH 9,
NIT 1, 2, and the H11 mutation-coverage requirement).

* first-step evidence: the exact 24-key schema, every value pinned -- the RNG digests per topology
  (one rank vs DDP), strategy, local_rank, the init state, SyncBN, the attention backend's call
  counts, the ViT checkpoint calls, the conditioner calls -- one mutation per field is refused;
* sealed records: exact schema, sidecar binding, written once, atomically;
* SMOKE_DDP receipts: the writer RECOMPUTES the log / evidence / checkpoint gates (a caller cannot
  hand it a verdict); the consumer checks every field against the current tree, environment and
  request (an SDPA smoke cannot authorise a flash FULL; the newest receipt decides);
* TRAINING_COMPLETE: published only after the end gates + the run-dir / checkpoint audits, bound
  to the closure, the manifest and the final checkpoint's sha + bytes;
* the environment receipt fails closed and binds interpreter, packages, flags and env vars;
* untracked links are validated by content (and on this box by target); the storage gate;
* train.py's logger metadata without the flag is the pre-flag dict.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp46_fixtures as fx  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

SMOKES = os.path.join(REPO, "outputs_FLAC", "exp38_smoke")
BASE = {"vanilla": os.path.join(SMOKES, "de_yaw_f025_2026-11-07_21-15-35", "evidence"),
        "fa_invariant": os.path.join(SMOKES, "de_bf_f025_2026-11-07_21-10-54", "evidence")}
have_smokes = pytest.mark.skipif(not all(os.path.isdir(d) for d in BASE.values()),
                                 reason="the GPU-0 smoke evidence lives in this checkout's outputs_FLAC only")
# the pinned values themselves, so the synthetic records below do not depend on the smoke files
PINNED_SINGLE = {
    "accumulate_grad_batches": 1, "batchnorm_nonsync_modules": 20, "flash_attn_func_set": True, "flash_calls": 48,
    "global_step_at_fit_start": 0, "global_step_recorded": 1, "gradient_clip_val": 0.0, "local_rank": 0,
    "micro_batch": 32, "n_state_tensors": 1279, "num_nodes": 1, "precision": "bf16-mixed", "rank": 0, "sdpa_calls": 0,
    "sobol": reg.INIT_SOBOL, "state_dict_sha256": reg.INIT_STATE_SHA256, "strategy": "SingleDeviceStrategy",
    "syncbn_modules": 0, "vit_ckpt_wrapped_layers": 12, "world_size": 1}


def records(run, ngpu=2, backend="flash", max_steps=30):
    arm, tag, rung = reg.parse_run_id(run)
    cond = reg.ARMS[arm]["cond_method"]
    micro = reg.RUNGS[rung]["micro"] if ngpu > 1 else 32
    out = {}
    for r in range(ngpu):
        e = dict(PINNED_SINGLE, rank=r, local_rank=r, world_size=ngpu, max_steps=max_steps, micro_batch=micro,
                 strategy="DDPStrategy" if ngpu > 1 else "SingleDeviceStrategy",
                 syncbn_modules=20 if ngpu > 1 else 0, batchnorm_nonsync_modules=0 if ngpu > 1 else 20,
                 rng_at_fit_start=dict(reg.RNG_AT_FIT_START["ddp" if ngpu > 1 else "single"]),
                 checkpoint_calls=[{"use_reentrant": False}] * reg.VIT_CKPT_CALLS[cond],
                 conditioner_calls=[[micro, None]] + ([[micro, ["source_vit", "context_poses_vit"]]] * 3
                                                      if cond == "fa_invariant" else []))
        if backend == "sdpa":
            e.update(flash_attn_func_set=False, flash_calls=0, sdpa_calls=48)
        out[r] = json.loads(json.dumps(e))
    return out


def write_evidence(d, recs):
    os.makedirs(d, exist_ok=True)
    for r, e in recs.items():
        json.dump(e, open(os.path.join(d, f"run_evidence.json.rank{r}.json"), "w"))


# ------------------------------------------------------------------ evidence
@have_smokes
@pytest.mark.parametrize("cond,run", [("vanilla", "de_yaw_f025")])   # C4F shares the vanilla pins
def test_the_real_gpu0_smoke_evidence_passes_the_exact_gate(cond, run):
    recs = reg.load_evidence(BASE[cond], 1)
    assert reg.evidence_problems(recs, reg.required_runs()[0], 1, "flash", 20) == []   # exp_38's YAW smoke
    for k, v in PINNED_SINGLE.items():
        assert recs[0][k] == v, k
    assert recs[0]["rng_at_fit_start"] == reg.RNG_AT_FIT_START["single"]


@pytest.mark.parametrize("run,ngpu", [(r, 2) for r in reg.all_runs()])
def test_synthetic_ddp_records_pass_for_every_registered_run(run, ngpu):
    assert reg.evidence_problems(records(run, ngpu), run, ngpu, "flash", 30) == []
    assert reg.evidence_problems(records(run, ngpu, "sdpa"), run, ngpu, "sdpa", 30) == []


def _m(fn):
    return fn


EVIDENCE_MUTATIONS = {
    "missing key": lambda e: e.pop("sobol"), "extra key": lambda e: e.update(extra=1),
    "rng python": lambda e: e["rng_at_fit_start"].update(python="0" * 64),
    "rng numpy": lambda e: e["rng_at_fit_start"].update(numpy="0" * 64),
    "rng torch_cpu (single-rank value under DDP)": lambda e: e["rng_at_fit_start"].update(
        torch_cpu=reg.RNG_AT_FIT_START["single"]["torch_cpu"]),
    "rng torch_cuda": lambda e: e["rng_at_fit_start"].update(torch_cuda="0" * 64),
    "rng missing component": lambda e: e["rng_at_fit_start"].pop("torch_cuda"),
    "strategy": lambda e: e.update(strategy="SingleDeviceStrategy"), "local_rank": lambda e: e.update(local_rank=0),
    "rank": lambda e: e.update(rank=0), "world": lambda e: e.update(world_size=4), "nodes": lambda e: e.update(num_nodes=2),
    "n_state_tensors": lambda e: e.update(n_state_tensors=1278), "init": lambda e: e.update(state_dict_sha256="0" * 64),
    "sobol drawn": lambda e: e["sobol"].update(num_generated=32), "sobol state": lambda e: e["sobol"].update(state_sha256="0" * 64),
    "syncbn": lambda e: e.update(syncbn_modules=19), "plain bn": lambda e: e.update(batchnorm_nonsync_modules=1),
    "flash calls": lambda e: e.update(flash_calls=47), "sdpa calls": lambda e: e.update(sdpa_calls=1),
    "flash flag type": lambda e: e.update(flash_attn_func_set=1),
    "ckpt calls count": lambda e: e.update(checkpoint_calls=e["checkpoint_calls"][:-1]),
    "ckpt reentrant": lambda e: e["checkpoint_calls"][0].update(use_reentrant=True),
    "ckpt unset": lambda e: e["checkpoint_calls"][3].update(use_reentrant="UNSET"),
    "conditioner": lambda e: e.update(conditioner_calls=[[32, None]] * 2),
    "micro": lambda e: e.update(micro_batch=16), "precision": lambda e: e.update(precision="32-true"),
    "clip": lambda e: e.update(gradient_clip_val=1.0), "accum": lambda e: e.update(accumulate_grad_batches=2),
    "recorded step": lambda e: e.update(global_step_recorded=2), "fit start": lambda e: e.update(global_step_at_fit_start=2500),
    "vit layers": lambda e: e.update(vit_ckpt_wrapped_layers=11), "max steps": lambda e: e.update(max_steps=40000),
    "int vs float": lambda e: e.update(gradient_clip_val=0),
}


@pytest.mark.parametrize("name", sorted(EVIDENCE_MUTATIONS))
@pytest.mark.parametrize("run", ["exp46_C4Ff050_l40_2x32", "exp46_C4Ff075_l40_2x32"])
def test_each_evidence_mutation_on_one_rank_is_refused(name, run):
    recs = records(run)
    EVIDENCE_MUTATIONS[name](recs[1])
    assert reg.evidence_problems(recs, run, 2, "flash", 30), name


def test_the_backend_is_part_of_the_evidence():
    assert reg.evidence_problems(records("exp46_C4Ff025_l40_2x32"), "exp46_C4Ff025_l40_2x32", 2, "sdpa", 30)
    assert reg.evidence_problems(records("exp46_C4Ff025_l40_2x32", backend="sdpa"), "exp46_C4Ff025_l40_2x32", 2, "flash", 30)


def test_load_evidence_needs_exactly_the_rank_files(tmp_path):
    d = tmp_path / "evidence"
    write_evidence(str(d), records("exp46_C4Ff025_l40_2x32"))
    assert sorted(reg.load_evidence(str(d), 2)) == [0, 1]
    with pytest.raises(reg.GateError):
        reg.load_evidence(str(d), 8)
    (d / "run_evidence.json.rank0.json.tmp").write_text("{}")
    with pytest.raises(reg.GateError):
        reg.load_evidence(str(d), 2)
    os.remove(d / "run_evidence.json.rank0.json.tmp")
    (d / "run_evidence.json.rank1.json").write_text("{not json")
    with pytest.raises(reg.GateError):
        reg.load_evidence(str(d), 2)


# ------------------------------------------------------------------ sealed records
def test_sealed_records(tmp_path):
    p = str(tmp_path / "r.json")
    digest = reg.write_sealed(p, {"a": 1, "b": [2]})
    rec, d2 = reg.read_sealed(p, {"a", "b"})
    assert rec == {"a": 1, "b": [2]} and d2 == digest and sorted(os.listdir(tmp_path)) == ["r.json", "r.json.sha256"]
    with pytest.raises(reg.GateError):
        reg.write_sealed(p, {"a": 2, "b": []})                  # written once
    with pytest.raises(reg.GateError):
        reg.read_sealed(p, {"a"})                               # exact schema
    with pytest.raises(reg.GateError):
        reg.read_sealed(p, {"a", "b", "c"})
    side = tmp_path / "r.json.sha256"
    side.write_text(side.read_text().replace("r.json", "q.json"))
    with pytest.raises(reg.GateError):
        reg.read_sealed(p, {"a", "b"})                          # the sidecar names this file
    side.write_text(f"{digest}  r.json\n")
    open(p, "ab").write(b" ")
    with pytest.raises(reg.GateError):
        reg.read_sealed(p, {"a", "b"})
    with pytest.raises(reg.GateError):
        reg.read_sealed(str(tmp_path / "absent.json"), {"a"})


# ------------------------------------------------------------------ SMOKE_DDP receipts
CLOSURE, ENV_SHA, AUDIT_SHA, HEAD = fx.FIX_CLOSURE, "d" * 64, "a" * 64, "f" * 40
A6000_GPUS = sorted([u, "NVIDIA RTX A6000"] for u in reg.A6000_UUIDS)
RF050 = "exp46_C4Ff050_l40_2x32"


def smoke_log(run, ngpu=2, steps=30, losses=3, end=True, backend="flash"):
    arm, tag, _ = reg.parse_run_id(run)
    parts = ["Seed set to 42"] + [f"Found {reg.FRACS[tag]['n_items']} files in 243 subfolders"] * ngpu
    if backend == "sdpa":
        parts += [reg.FLASH_MISSING + ", disabling Flash Attention"] * ngpu
    parts += [f"Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {r + 1}/{ngpu}" for r in range(ngpu)]
    parts.append(reg.ARMS[arm]["banner"])                       # one C4F line (rank 0)
    spe = reg.FRACS[tag]["steps_per_epoch"]
    parts += [f"Epoch 0:   0%|          | {i + 1}/{spe} [00:05<1:00:00,  0.30it/s, train/loss=0.5{i}]" for i in range(losses)]
    text = "\n".join(parts)
    return text + ("\r" + reg.FIT_END.format(n=steps) if end else "")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": CLOSURE, "n_files": 1})
    monkeypatch.setattr(reg, "env_receipt_sha", lambda platform, repo=None: ENV_SHA)
    monkeypatch.setattr(reg, "check_subset_audit", lambda repo=None: {"sha256": AUDIT_SHA})
    (tmp_path / reg.EXP_DIR).mkdir(parents=True)
    import shutil
    for arm, rung in (("c4f", "2x32"),):
        rel = reg.model_config_rel(arm, rung)
        os.makedirs(tmp_path / os.path.dirname(rel), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), tmp_path / rel)
    for t in reg.FRACS:
        for k in ("split", "dataset_config"):
            os.makedirs(tmp_path / os.path.dirname(reg.FRACS[t][k]), exist_ok=True)
            shutil.copyfile(os.path.join(REPO, reg.FRACS[t][k]), tmp_path / reg.FRACS[t][k])
    return tmp_path


def _smoke(tree, run="exp46_C4Ff025_l40_2x32", ts="2026-11-08_04-00-00", rc=0, ckpt=False, recs=None, log=None, backend="flash",
           platform="a6000", gpus=None, logger=None):
    sd = tree / f"smoke_{ts}"
    for rel, data in ({"lightning_logs/version_0/hparams.yaml": "{}\n"} if logger is None else logger).items():
        (sd / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        (sd / rel).write_text(data)                              # what PL's CSV logger leaves after 30 steps
    write_evidence(str(sd / "evidence"), recs if recs is not None else records(run, backend=backend))
    if ckpt:
        (sd / "epoch=0-step=1.ckpt").write_bytes(b"x")
    lp = tree / f"smoke_{ts}.log"
    lp.write_text(log if log is not None else smoke_log(run, backend=backend))
    snap = tree / f"snap_{ts}"
    snap_sha = fx.fake_snapshot(reg, str(snap))
    return reg.write_smoke_receipt(run, platform, backend, HEAD, str(sd), str(lp), rc, gpus or A6000_GPUS, snap_sha,
                                   "2026-11-08T04:00:00-0500", ts, ENV_SHA, repo=str(tree), out_repo=str(tree),
                                   snapshot_dir=str(snap))


def test_the_smoke_writer_recomputes_every_gate(tree):
    assert _smoke(tree)["verdict"] == "PASS"
    assert _smoke(tree, ts="t1", rc=1)["verdict"] == "FAIL"
    assert _smoke(tree, ts="t2", ckpt=True)["verdict"] == "FAIL"
    bad = records("exp46_C4Ff025_l40_2x32")
    bad[1]["rng_at_fit_start"]["torch_cuda"] = "0" * 64
    out = _smoke(tree, ts="t3", recs=bad)
    assert out["verdict"] == "FAIL" and out["evidence_gate"] != "PASS"
    out = _smoke(tree, ts="t4", log=smoke_log("exp46_C4Ff025_l40_2x32", end=False))
    assert out["verdict"] == "FAIL" and out["log_gate"] != "PASS"
    out = _smoke(tree, ts="t5", recs={0: records("exp46_C4Ff025_l40_2x32")[0]})
    assert out["verdict"] == "FAIL"                               # a missing rank's evidence
    assert _smoke(tree, ts="t6")["logger_gate"] == "PASS"
    v0 = "lightning_logs/version_0/"
    for i, logger in enumerate(({}, {v0 + "hparams.yaml": "{}\n", v0 + "events.out.tfevents.1.host": "x"},
                                {v0 + "hparams.yaml": "{}\n", v0 + "metrics.csv": "step\n"},
                                {v0 + "hparams.yaml": "{}\n", "lightning_logs/version_1/hparams.yaml": "{}\n"})):
        out = _smoke(tree, ts=f"t7{i}", logger=logger)          # not exactly the CSV logger's version_0 dir
        assert out["verdict"] == "FAIL" and out["logger_gate"] != "PASS", logger


def test_the_smoke_logger_gate_checks_file_types_like_the_full_mover(tree):
    """Codex code r1 finding 16: the SMOKE logger gate requires PL's CSV-logger OUTPUT, not just its names --
    lightning_logs/version_0 a real directory and version_0/hparams.yaml a regular file, neither a symlink (the FULL
    mover's own check). An empty directory or a symlink named hparams.yaml, or a symlinked version_0, is refused."""
    v0 = "lightning_logs/version_0"
    elsewhere = tree / "elsewhere"
    (elsewhere / "version_0").mkdir(parents=True)
    (elsewhere / "version_0" / "hparams.yaml").write_text("{}\n")
    (elsewhere / "hparams.yaml").write_text("{}\n")

    def regular(sd):
        (sd / v0).mkdir(parents=True)
        (sd / v0 / "hparams.yaml").write_text("{}\n")

    def hp_dir(sd):
        (sd / v0 / "hparams.yaml").mkdir(parents=True)                # an empty directory under the CSV name

    def hp_link(sd):
        (sd / v0).mkdir(parents=True)
        os.symlink(elsewhere / "hparams.yaml", sd / v0 / "hparams.yaml")

    def v0_link(sd):
        (sd / "lightning_logs").mkdir(parents=True)
        os.symlink(elsewhere / "version_0", sd / v0)

    for name, make in (("regular", regular), ("hparams dir", hp_dir), ("hparams symlink", hp_link),
                       ("version_0 symlink", v0_link)):
        sd = tree / f"logger_{name.replace(' ', '_')}"
        make(sd)
        probs = reg.smoke_logger_problems(str(sd))
        assert (probs == []) == (name == "regular"), (name, probs)
    # through the SMOKE writer: the sealed receipt's logger gate and verdict
    out = _smoke(tree, ts="t_link", logger={})
    assert out["verdict"] == "FAIL"                                     # no logger dir at all (unchanged)
    sd = tree / "smoke_t_hp"
    (sd / v0 / "hparams.yaml").mkdir(parents=True)
    out = _smoke(tree, ts="t_hp", logger={})
    assert out["verdict"] == "FAIL" and out["logger_gate"] != "PASS"


def test_the_writers_refuse_an_environment_receipt_changed_during_the_run(tree, monkeypatch):
    monkeypatch.setattr(reg, "env_receipt_sha", lambda platform, repo=None: "9" * 64)
    with pytest.raises(reg.GateError, match="environment receipt changed"):
        _smoke(tree)


def test_the_smoke_consumer_accepts_the_newest_matching_pass(tree):
    out = _smoke(tree)
    got = reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))
    assert got["smoke_receipt"] == out["smoke_receipt"]
    _smoke(tree, ts="2026-11-08_05-00-00", rc=1)                  # a newer FAIL
    with pytest.raises(reg.GateError):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))


def _reseal(path, fn):
    rec = json.load(open(path))
    fn(rec)
    os.remove(path)
    os.remove(path + ".sha256")
    reg.write_sealed(path, rec)


SMOKE_MUTATIONS = {
    "backend": lambda r: r.update(backend="sdpa"), "head": lambda r: r.update(head="e" * 40),
    "closure": lambda r: r.update(closure_sha256="0" * 64), "env": lambda r: r.update(env_receipt_sha256="0" * 64),
    "audit": lambda r: r.update(subset_audit_sha256="0" * 64), "run": lambda r: r.update(run="exp46_C4Ff050_l40_2x32"),
    "platform": lambda r: r.update(platform="neuronic"), "mode": lambda r: r.update(mode="SMOKE"),
    "argv": lambda r: r.update(argv_projection=r["argv_projection"][:-2]),
    "model cfg": lambda r: r.update(model_config_sha256="0" * 64), "data cfg": lambda r: r.update(dataset_config_sha256="0" * 64),
    "split": lambda r: r.update(split_sha256="0" * 64), "rc": lambda r: r.update(rc=1),
    "ckpts": lambda r: r.update(checkpoints_written=1), "log gate": lambda r: r.update(log_gate=["x"]),
    "evidence gate": lambda r: r.update(evidence_gate=["x"]), "verdict": lambda r: r.update(verdict="FAIL"),
    "gpus": lambda r: r.update(gpus=[["GPU-ffff", "NVIDIA RTX A6000"]] * 2),
    "evidence content": lambda r: r["evidence"]["1"]["rng_at_fit_start"].update(torch_cpu="0" * 64),
    "evidence rank": lambda r: r["evidence"].pop("1"), "evidence sdpa": lambda r: r["evidence"]["0"].update(
        flash_attn_func_set=False, flash_calls=0, sdpa_calls=48),
    "snapshot": lambda r: r.update(snapshot_sha256="nope"), "schema": lambda r: r.update(schema=2),
    "snapshot sha (valid-looking)": lambda r: r.update(snapshot_sha256="7" * 64),
    "snapshot dir": lambda r: r.update(snapshot_dir="/nonexistent/snapshot"),
    "log sha (valid-looking)": lambda r: r.update(log_sha256="7" * 64), "log path": lambda r: r.update(log="/nonexistent.log"),
    "extra key": lambda r: r.update(extra=1), "missing key": lambda r: r.pop("finished_at"),
    "logger gate": lambda r: r.update(logger_gate=["x"]), "smoke dir": lambda r: r.update(smoke_dir="/nonexistent/smoke"),
}


@pytest.mark.parametrize("name", sorted(SMOKE_MUTATIONS))
def test_the_smoke_consumer_refuses_each_field_mutation(tree, name):
    out = _smoke(tree)
    _reseal(out["smoke_receipt"], SMOKE_MUTATIONS[name])
    with pytest.raises(reg.GateError):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))


def test_the_smoke_consumer_re_opens_its_log_and_snapshot(tree):
    """Durable bindings are re-hashed (Codex code r2 HIGH 3): an edited log or snapshot file refuses
    although the sealed receipt is untouched; a log that no longer passes its gate refuses too."""
    out = _smoke(tree)
    rec = json.load(open(out["smoke_receipt"]))
    reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))
    open(os.path.join(rec["snapshot_dir"], "x.txt"), "a").write("tampered\n")
    with pytest.raises(reg.GateError, match="snapshot"):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))
    out = _smoke(tree, ts="2026-11-08_05-00-00")
    rec = json.load(open(out["smoke_receipt"]))
    (__import__("pathlib").Path(rec["smoke_dir"]) / "lightning_logs" / "version_1").mkdir()
    with pytest.raises(reg.GateError, match="smoke logger"):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))
    out = _smoke(tree, ts="2026-11-08_05-30-00")
    rec = json.load(open(out["smoke_receipt"]))
    open(rec["log"], "a").write("one more line\n")
    with pytest.raises(reg.GateError, match="log"):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))


def test_the_smoke_consumer_checks_the_request(tree, monkeypatch):
    out = _smoke(tree)
    for args in (("exp46_C4Ff025_l40_2x32", "a6000", "sdpa", HEAD), ("exp46_C4Ff025_l40_2x32", "a6000", "flash", "e" * 40),
                 ("exp46_C4Ff050_l40_2x32", "a6000", "flash", HEAD)):
        with pytest.raises(reg.GateError):
            reg.check_smoke_receipt(*args, A6000_GPUS, repo=str(tree))
    monkeypatch.setattr(reg, "env_receipt_sha", lambda platform, repo=None: "9" * 64)   # a recreated env receipt
    with pytest.raises(reg.GateError):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, A6000_GPUS, repo=str(tree))
    open(out["smoke_receipt"], "ab").write(b" ")
    with pytest.raises(reg.GateError):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "flash", HEAD, None, repo=str(tree))


def test_a_neuronic_smoke_receipt_needs_its_l40_allocation(tree):
    gpus = [[f"GPU-{i}", "NVIDIA L40"] for i in range(2)]
    _smoke(tree, platform="neuronic", gpus=gpus)
    assert reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "neuronic", "flash", HEAD, repo=str(tree))
    _smoke(tree, ts="2026-11-08_06-00-00", platform="neuronic", gpus=[["GPU-0", "NVIDIA A100"]] * 2)
    with pytest.raises(reg.GateError):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "neuronic", "flash", HEAD, repo=str(tree))


# ------------------------------------------------------------------ TRAINING_COMPLETE
import exp46_eval  # noqa: E402  (E0_KEYS for the retained E0 copy)


def _admitted(tree, monkeypatch, run="exp46_C4Ff050_l40_2x32", backend="flash"):
    """The sealed E0 / environment / SMOKE_DDP receipts a FULL launch validated, in the campaign tree,
    mutually consistent; returns (paths, digests)."""
    exp = tree / reg.EXP_DIR
    e0d = reg.write_sealed(str(exp / "c4flip_data_curve_e0_receipt.json"), fx.synthetic_e0(exp46_eval, CLOSURE))
    pk = {"flash-attn": reg.FLASH_VERSION}
    _, envd = fx.sealed(reg, str(exp / "c4flip_data_curve_env_receipt_a6000.json"), reg.ENV_RECEIPT_KEYS, platform="a6000",
                        packages=pk, packages_sha256=reg.hashlib.sha256(reg.canonical(pk).encode()).hexdigest(),
                        flags={"optimize": 0, "no_user_site": 1})
    monkeypatch.setattr(reg, "env_receipt_sha", lambda platform, repo=None: envd)
    argv = reg.build_train_argv(run, "a6000")
    smk_path = str(exp / f"c4flip_data_curve_smoke_{run}_a6000_2026-11-08_04-00-00.json")
    slog = exp / f"c4flip_data_curve_2026-11-08_03-00-00_{run}_a6000_SMOKE_DDP_train.log"
    slog.write_text(smoke_log(run, backend=backend))
    _, smkd = fx.sealed(reg, smk_path, reg.SMOKE_KEYS, schema=1, mode="SMOKE_DDP", run=run, platform="a6000",
                        backend=backend, head=HEAD, closure_sha256=CLOSURE, env_receipt_sha256=envd, rc=0,
                        checkpoints_written=0, log_gate="PASS", evidence_gate="PASS", verdict="PASS",
                        logger_gate="PASS", smoke_dir=str(tree / "smoke"),
                        argv_projection=reg.argv_projection(argv), gpus=A6000_GPUS, snapshot_sha256="7" * 64,
                        log=str(slog), log_sha256=reg.sha256_file(str(slog)),
                        launched_at="2026-11-06T03:00:00-0500", finished_at="2026-11-06T03:30:00-0500",
                        evidence={str(r): e for r, e in records(run, backend=backend).items()})
    paths = {"e0": str(exp / "c4flip_data_curve_e0_receipt.json"), "smoke": smk_path,
             "env": str(exp / "c4flip_data_curve_env_receipt_a6000.json")}
    return paths, {"e0": e0d, "env": envd, "smoke": smkd, "smoke_log": str(slog)}


def _full_run(tree, monkeypatch, run="exp46_C4Ff050_l40_2x32", losses=60, rc=0, drop=None, backend="flash"):
    arm, tag, _ = reg.parse_run_id(run)
    rd = tree / run
    rd.mkdir()
    for s_ in reg.cadence_steps():
        if s_ != drop:
            (rd / reg.ckpt_name(s_, tag)).write_bytes(f"ckpt {s_}".encode())
    (rd / ".exp46_lock").mkdir()
    write_evidence(str(rd / "evidence"), records(run, max_steps=40000, backend=backend))
    log = tree / f"{run}.log"
    log.write_text(smoke_log(run, steps=40000, losses=losses, backend=backend))
    monkeypatch.setattr(reg, "audit_ckpt", lambda path, run, step, repo=None, run_dir=None: {
        "sha256": reg.sha256_file(path), "bytes": os.path.getsize(path), "global_step": step})
    paths, digests = _admitted(tree, monkeypatch, run, backend=backend)
    retained = reg.retain_receipts(str(rd), paths, plain={"smoke_log": digests["smoke_log"]})
    digests["snapshot"] = fx.write_snapshot_manifests(reg, str(tree))     # the launched snapshot's manifests
    digests["loss"] = fx.write_loss_record(reg, str(rd / reg.RECEIPTS_DIR), tag)   # what the supervisor moved in
    return rd, log, retained, paths, digests


def _write_tr(tree, rd, log, retained, paths, digests, run="exp46_C4Ff050_l40_2x32", rc=0, content=None, commit=True,
              backend="flash"):
    rec = reg.write_training_receipt(run, "a6000", backend, HEAD, str(rd), str(log), rc, digests["snapshot"],
                                     digests["e0"], {"smoke_receipt": paths["smoke"], "sha256": digests["smoke"]},
                                     A6000_GPUS, "2026-11-07T04:00:00-0500", digests["env"], retained,
                                     reg.AR_CONTENT_SHA256 if content is None else content,
                                     repo=str(tree), tree=str(tree))
    if commit:
        fx.commit(reg, str(rd), run, reg.sha256_file(str(rd / reg.TRAINING_RECEIPT)))
    return rec


def test_the_training_receipt_is_published_only_after_every_end_audit(tree, monkeypatch):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    rec = _write_tr(tree, rd, log, retained, paths, digests)
    assert rec["final_ckpt_bytes"] == len(b"ckpt 40000") and rec["rc"] == 0
    assert sorted(os.listdir(rd / reg.RECEIPTS_DIR)) == sorted(
        [os.path.basename(p) for p in paths.values()] + [os.path.basename(p) + ".sha256" for p in paths.values()]
        + ["train.log", "smoke_log.log", "SNAPSHOT.sha256", "SNAPSHOT.links", reg.LOSS_RECORD, reg.LOSS_RECORD_HPARAMS])
    assert (rec["loss_record_sha256"], rec["loss_record_rows"]) == (digests["loss"]["loss_record_sha256"], 400)
    got, digest = reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"],
                                             platform="a6000")
    assert got == rec and len(digest) == 64
    for kw in ({"closure": "0" * 64}, {"e0_sha": "0" * 64}, {"platform": "neuronic"}):
        with pytest.raises(reg.GateError):
            reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), **kw)
    with pytest.raises(reg.GateError):
        _write_tr(tree, rd, log, retained, paths, digests)                       # written once
    with pytest.raises(reg.GateError):
        reg.retain_receipts(str(rd), paths)                                      # retained once


def test_a_consistent_sdpa_training_chain_is_refused_at_admission(tree, monkeypatch):
    """Codex code r2 finding 12: plan r3 F2 / L5 register backend flash for training (exp_38's seven neuronic FULL
    launch records at e5a012fa all say backend flash). A fully consistent SDPA chain -- SMOKE_DDP receipt, evidence,
    logs, training receipt all SDPA -- is refused at admission, and so is any request for an SDPA SMOKE."""
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch, backend="sdpa")
    _write_tr(tree, rd, log, retained, paths, digests, backend="sdpa")
    with pytest.raises(reg.GateError, match="flash"):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"],
                                   platform="a6000")
    monkeypatch.setattr(reg, "env_receipt_sha", lambda platform, repo=None: ENV_SHA)   # the SMOKE helper's receipt
    _smoke(tree, backend="sdpa")
    with pytest.raises(reg.GateError, match="flash"):
        reg.check_smoke_receipt("exp46_C4Ff025_l40_2x32", "a6000", "sdpa", HEAD, A6000_GPUS, repo=str(tree))


def test_an_sdpa_launch_is_refused_before_any_work(tmp_path, monkeypatch):
    """Codex code r2 finding 12: the submitter and the supervisor refuse a non-flash SMOKE / SMOKE_DDP / FULL before
    any scheduler call, lock, record or directory (DRY_RUN stays a diagnostic)."""
    sys.path.insert(0, EXP)
    import exp46_submit_neuronic as sub
    import exp46_supervise as sv
    import types
    called = []
    monkeypatch.setattr(sub, "partition_max_time", lambda runner=None: called.append("scontrol") or "7-00:00:00")
    monkeypatch.setattr(sub, "preflight", lambda *a: called.append("preflight") or "a" * 40)
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)

    def no_scheduler(argv, capture_output=True, text=True):      # never a real sbatch, whatever the code does
        called.append(os.path.basename(argv[0]))
        return types.SimpleNamespace(returncode=1, stdout="", stderr="no scheduler in this test")
    for mode in ("SMOKE", "FULL"):
        with pytest.raises(sub.SubmitError, match="flash"):
            sub.submit("exp46_C4Ff025_l40_2x32", mode, "01:00:00", "sdpa", "", False, False, root=str(tmp_path / "r"),
                       runner=no_scheduler)
    assert called == [] and not (tmp_path / "r").exists()
    for argv in (["--platform", "neuronic", "--mode", "SMOKE", "--token", "a" * 32, "--save-dir", "/x"],
                 ["--platform", "neuronic", "--mode", "FULL", "--token", "a" * 32, "--save-dir", "/x"],
                 ["--platform", "a6000", "--mode", "SMOKE_DDP"], ["--platform", "a6000", "--mode", "FULL"]):
        with pytest.raises(SystemExit):
            sv.parse(argv + ["--run", "exp46_C4Ff025_l40_2x32", "--backend", "sdpa"])
    sv.parse(["--platform", "neuronic", "--mode", "DRY_RUN", "--run", "exp46_C4Ff025_l40_2x32", "--backend", "sdpa"])


@pytest.mark.parametrize("case", ["rc", "few losses", "missing ckpt", "bad evidence", "sdpa evidence", "content"])
def test_the_training_receipt_refuses_a_failed_run(tree, monkeypatch, case):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch, losses=10 if case == "few losses" else 60,
                                                  drop=20000 if case == "missing ckpt" else None)
    if case == "bad evidence":
        write_evidence(str(rd / "evidence"), {**records("exp46_C4Ff050_l40_2x32", max_steps=40000),
                                              1: dict(records("exp46_C4Ff050_l40_2x32", max_steps=40000)[1], local_rank=0)})
    if case == "sdpa evidence":
        write_evidence(str(rd / "evidence"), records("exp46_C4Ff050_l40_2x32", backend="sdpa", max_steps=40000))
    with pytest.raises(reg.GateError):
        _write_tr(tree, rd, log, retained, paths, digests, rc=1 if case == "rc" else 0,
                  content="0" * 64 if case == "content" else None)
    assert not (rd / reg.TRAINING_RECEIPT).exists()


def _reseal_tr(rd, fn):
    path = str(rd / reg.TRAINING_RECEIPT)
    rec = json.load(open(path))
    fn(rec)
    os.remove(path)
    os.remove(path + ".sha256")
    reg.write_sealed(path, rec)


TR_MUTATIONS = {
    "platform": lambda r: r.update(platform="neuronic"), "backend": lambda r: r.update(backend="sdpa"),
    "head": lambda r: r.update(head="e" * 40), "head malformed": lambda r: r.update(head="abc"),
    "argv": lambda r: r.update(argv=r["argv"][:-2]), "argv batch": lambda r: r["argv"].__setitem__(r["argv"].index("--batch-size") + 1, "16"),
    "model cfg": lambda r: r.update(model_config_sha256="0" * 64), "data cfg": lambda r: r.update(dataset_config_sha256="0" * 64),
    "split": lambda r: r.update(split_sha256="0" * 64), "content": lambda r: r.update(data_content_sha256="0" * 64),
    "rc": lambda r: r.update(rc=1), "log gate": lambda r: r.update(log_gate=["x"]), "evidence gate": lambda r: r.update(evidence_gate=["x"]),
    "env sha": lambda r: r.update(env_receipt_sha256="1" * 64), "smoke sha": lambda r: r.update(smoke_receipt_sha256="1" * 64),
    "e0 sha": lambda r: r.update(e0_receipt_sha256="1" * 64), "snapshot": lambda r: r.update(snapshot_sha256="x"),
    "log sha": lambda r: r.update(log_sha256="1" * 64), "evidence sha": lambda r: r.update(evidence_sha256="1" * 64),
    "manifest sha": lambda r: r.update(checkpoints_sha256="1" * 64), "final sha": lambda r: r.update(final_ckpt_sha256="1" * 64),
    "final bytes": lambda r: r.update(final_ckpt_bytes=1), "final bytes type": lambda r: r.update(final_ckpt_bytes=10.0),
    "final name": lambda r: r.update(final_ckpt="epoch=0-step=40000.ckpt"),
    "gpus": lambda r: r.update(gpus=[["GPU-x", "NVIDIA RTX A6000"]] * 2), "gpus l40": lambda r: r.update(gpus=[["GPU-0", "NVIDIA L40"]] * 2),
    "receipts map": lambda r: r["receipts"].pop("smoke"), "receipts name": lambda r: r["receipts"].update(env="other.json"),
    "snapshot sha (valid-looking)": lambda r: r.update(snapshot_sha256="9" * 64),
    "closure (valid-looking)": lambda r: r.update(closure_sha256="9" * 64),
    "time order": lambda r: r.update(launched_at="2099-11-10T04:00:00-0500"),
    "content checks": lambda r: r.update(data_content_checks=["pre-launch"]),
    "time": lambda r: r.update(finished_at=None), "schema": lambda r: r.update(schema=2), "run": lambda r: r.update(run="exp46_C4Ff025_l40_2x32"),
    "loss sha": lambda r: r.update(loss_record_sha256="1" * 64), "loss rows": lambda r: r.update(loss_record_rows=399),
    "loss rows type": lambda r: r.update(loss_record_rows=400.0), "loss sha malformed": lambda r: r.update(loss_record_sha256="x"),
    "receipts loss name": lambda r: r["receipts"].update(loss_record="metrics.csv"),
    "receipts loss missing": lambda r: r["receipts"].pop("loss_record_hparams"),
}


@pytest.mark.parametrize("name", sorted(TR_MUTATIONS))
def test_the_training_receipt_consumer_refuses_each_field_mutation(tree, monkeypatch, name):
    """Codex code r2 BLOCKING 1: every field of the sealed terminal receipt is checked on consumption."""
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests)
    _reseal_tr(rd, TR_MUTATIONS[name])
    with pytest.raises(reg.GateError):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"])


@pytest.mark.parametrize("what", ["e0 copy", "env copy", "smoke copy resealed", "log copy", "evidence file",
                                  "smoke copy evidence", "smoke log", "snapshot manifest", "e0 copy resealed",
                                  "env copy resealed", "no commit", "commit record status"])
def test_the_training_receipt_re_opens_its_retained_records(tree, monkeypatch, what):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests)
    rdir = rd / reg.RECEIPTS_DIR
    if what == "e0 copy":
        p = rdir / retained["e0"]
        p.write_bytes(p.read_bytes().replace(b'"PASS"', b'"FAIL"'))
    elif what == "env copy":
        p = rdir / retained["env"]
        p.write_bytes(p.read_bytes() + b" ")
    elif what in ("smoke copy resealed", "smoke copy evidence"):
        p = str(rdir / retained["smoke"])
        rec = json.load(open(p))
        if what == "smoke copy resealed":
            rec["backend"] = "sdpa"
        else:
            rec["evidence"]["1"]["rng_at_fit_start"]["torch_cuda"] = "0" * 64
        os.remove(p)
        os.remove(p + ".sha256")
        reg.write_sealed(p, rec)
    elif what == "log copy":
        (rdir / "train.log").write_text("Seed set to 42\n")
    elif what == "smoke log":
        (rdir / "smoke_log.log").write_text("Seed set to 42\n")
    elif what == "snapshot manifest":
        (rdir / "SNAPSHOT.sha256").write_text(fx.SNAP_TEXT + "0" * 64 + "  src/extra.py\n")
    elif what in ("e0 copy resealed", "env copy resealed"):       # internally inconsistent, freshly sealed
        p = str(rdir / retained["e0" if what.startswith("e0") else "env"])
        rec = json.load(open(p))
        if what.startswith("e0"):
            rec["checkpoints"]["E0b"]["bytes"] = 1
        else:
            rec["packages"]["torch"] = "9.9"
        os.remove(p)
        os.remove(p + ".sha256")
        reg.write_sealed(p, rec)
    elif what == "no commit":
        os.remove(rd / reg.COMMIT_MARKER)
    elif what == "commit record status":
        lr = rd / json.load(open(rd / reg.COMMIT_MARKER))["launch_record"]
        rec = json.load(open(lr))
        rec["status"] = "running"
        json.dump(rec, open(lr, "w"))
    else:
        f = rd / "evidence" / "run_evidence.json.rank1.json"
        e = json.load(open(f))
        e["n_state_tensors"] = 1
        json.dump(e, open(f, "w"))
    with pytest.raises(reg.GateError):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"])


# ------------------------------------------------------------------ the loss record (plan r3 §3.2, §3.3)
def test_check_loss_record_accepts_pls_csv_in_any_column_order(tmp_path):
    p = tmp_path / "metrics.csv"
    p.write_bytes(fx.loss_csv(reg))
    got = reg.check_loss_record(str(p))
    assert got == {"loss_record_sha256": reg.sha256_file(str(p)), "loss_record_rows": 400}
    p.write_bytes(fx.loss_csv(reg, header=list(reversed(fx.LOSS_HEADER))))      # PL builds the order from a set
    assert reg.check_loss_record(str(p))["loss_record_rows"] == 400


@pytest.mark.parametrize("name,kw", [
    ("399 rows", {"rows": 399}), ("401 rows", {"rows": 401}),
    ("step gap", {"fn": lambda k, r: r.update(step="1199") if k == 10 else None}),       # 1099 missing
    ("step float", {"fn": lambda k, r: r.update(step=r["step"] + ".0") if k == 0 else None}),
    ("nan loss", {"fn": lambda k, r: r.update(**{"train/loss": "nan"}) if k == 7 else None}),
    ("inf loss", {"fn": lambda k, r: r.update(**{"train/loss": "inf"}) if k == 399 else None}),
    ("empty loss", {"fn": lambda k, r: r.update(**{"train/loss": ""}) if k == 3 else None}),
    ("extra column", {"header": fx.LOSS_HEADER + ["val/loss"]}),
    ("missing column", {"header": fx.LOSS_HEADER[:-1]}),
    ("duplicate column", {"header": fx.LOSS_HEADER + ["train/loss"]}),
])
def test_check_loss_record_refuses(tmp_path, name, kw):
    p = tmp_path / "metrics.csv"
    p.write_bytes(fx.loss_csv(reg, **kw))
    with pytest.raises(reg.GateError):
        reg.check_loss_record(str(p))


def test_check_loss_record_refuses_a_missing_or_empty_file(tmp_path):
    with pytest.raises(reg.GateError):
        reg.check_loss_record(str(tmp_path / "absent.csv"))
    (tmp_path / "e.csv").write_bytes(b"")
    with pytest.raises(reg.GateError):
        reg.check_loss_record(str(tmp_path / "e.csv"))


@pytest.mark.parametrize("offline", [False, True])
def test_a_loss_record_changed_after_sealing_is_refused(tree, monkeypatch, offline):
    """The training receipt binds the loss record's sha256 / rows; online (the run dir) and offline (a copy of
    the run's records, as the evaluation farm keeps it) the CSV is re-opened and re-validated."""
    import shutil
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests)
    farm = tree / "farm_copy"
    farm.mkdir()
    for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256", reg.COMMIT_MARKER,
              reg.COMMIT_MARKER + ".sha256"):
        shutil.copyfile(rd / f, farm / f)
    lr = json.load(open(rd / reg.COMMIT_MARKER))["launch_record"]
    shutil.copyfile(rd / lr, farm / lr)
    for sub in (reg.RECEIPTS_DIR, "evidence"):
        shutil.copytree(rd / sub, farm / sub)
    kw = {"receipt_dir": str(farm), "verify_bytes": False} if offline else {}
    src = farm if offline else rd
    reg.check_training_receipt(RF050, str(rd), closure=CLOSURE, e0_sha=digests["e0"], **kw)
    csv = src / reg.RECEIPTS_DIR / reg.LOSS_RECORD
    good = csv.read_bytes()
    csv.write_bytes(good.replace(b"4.9e-05", b"4.8e-05", 1))                         # a valid CSV, other bytes
    with pytest.raises(reg.GateError, match="loss record"):
        reg.check_training_receipt(RF050, str(rd), closure=CLOSURE, e0_sha=digests["e0"], **kw)
    csv.write_bytes(good)
    os.remove(src / reg.RECEIPTS_DIR / reg.LOSS_RECORD_HPARAMS)
    with pytest.raises(reg.GateError, match="hparams"):
        reg.check_training_receipt(RF050, str(rd), closure=CLOSURE, e0_sha=digests["e0"], **kw)


def test_the_training_receipt_needs_the_moved_loss_record(tree, monkeypatch):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    os.remove(rd / reg.RECEIPTS_DIR / reg.LOSS_RECORD)
    with pytest.raises(reg.GateError, match="loss record"):
        _write_tr(tree, rd, log, retained, paths, digests)
    assert not (rd / reg.TRAINING_RECEIPT).exists()


def test_the_training_receipt_binds_manifest_sha_and_bytes(tree, monkeypatch):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests)
    final = rd / reg.ckpt_name(40000, "050")
    final.write_bytes(b"ckpt 40001")                                             # same length, other bytes
    with pytest.raises(reg.GateError):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE)
    reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, verify_bytes=False)
    man = rd / "CHECKPOINTS.sha256"
    man.write_text(man.read_text().replace("-step=2500.ckpt", "-step=2501.ckpt"))
    with pytest.raises(reg.GateError):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, verify_bytes=False)


# ------------------------------------------------------------------ environment receipt
@pytest.mark.skipif(not os.path.isdir("/media/diskstation/yixunhu/FLAC"), reason="A6000 box only (nvidia-smi, A6000s)")
def test_the_environment_snapshot_is_complete_on_this_box():
    rec = reg.env_snapshot("a6000")
    assert set(rec) == reg.ENV_RECEIPT_KEYS and rec["packages"]["flash-attn"] == reg.FLASH_VERSION
    assert rec["executable"] == os.path.realpath(sys.executable) and rec["nccl"] and rec["driver"]
    assert rec["gpu_names"] == ["NVIDIA RTX A6000"] and rec["flags"]["optimize"] == 0


def test_the_environment_snapshot_fails_closed(monkeypatch):
    import torch
    monkeypatch.setattr(reg, "_gpu_rows", lambda: (_ for _ in ()).throw(reg.GateError("nvidia-smi failed")))
    with pytest.raises(reg.GateError):
        reg.env_snapshot("a6000")
    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (_ for _ in ()).throw(RuntimeError("no nccl")))
    with pytest.raises(reg.GateError):
        reg.env_snapshot("a6000")


def test_the_environment_receipt_is_written_once_and_detects_drift(tmp_path, monkeypatch):
    snap = {k: None for k in reg.ENV_RECEIPT_KEYS}
    snap.update(packages={"torch": "2.7.0", "numpy": "1.26.4"}, platform="a6000")
    state = {"now": json.loads(json.dumps(snap))}
    monkeypatch.setattr(reg, "env_snapshot", lambda platform: json.loads(json.dumps(state["now"])))
    (tmp_path / reg.EXP_DIR).mkdir(parents=True)
    out = reg.env_receipt("a6000", write=True, repo=str(tmp_path))
    assert reg.env_receipt("a6000", repo=str(tmp_path))["sha256"] == out["sha256"]
    with pytest.raises(reg.GateError):
        reg.env_receipt("a6000", write=True, repo=str(tmp_path))
    state["now"]["packages"]["numpy"] = "2.0.0"
    with pytest.raises(reg.GateError, match="numpy"):
        reg.env_receipt("a6000", repo=str(tmp_path))
    with pytest.raises(reg.GateError):
        reg.env_receipt("neuronic", repo=str(tmp_path))          # no receipt for that platform


def test_env_problems():
    ok = {"PYTHONNOUSERSITE": "1"}
    assert reg.env_problems(ok) == []
    for bad in ({}, {**ok, "PYTHONOPTIMIZE": "1"}, {**ok, "ENABLE_TORCH_COMPILE": "1"}, {**ok, "LD_PRELOAD": "x.so"},
                {**ok, "PYTHONSTARTUP": "/x.py"}, {**ok, "CUDA_LAUNCH_BLOCKING": "1"}):
        assert reg.env_problems(bad), bad


# ------------------------------------------------------------------ untracked links by content
def _git_tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "weights" / "FLAC").mkdir(parents=True)
    return root


def test_neuronic_links_are_validated_by_content(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root = _git_tree(tmp_path)
    data = tmp_path / "AR"
    data.mkdir()
    (root / "AcousticRooms").symlink_to(data)
    fake_vae = tmp_path / "vae.safetensors"
    fake_vae.write_bytes(b"not the vae")
    (root / reg.VAE_PATH).symlink_to(fake_vae)
    probs = reg.check_untracked("neuronic", str(root))
    assert any(reg.VAE_PATH in p and "pinned" in p for p in probs)
    (root / reg.VAE_PATH).unlink()
    (root / reg.VAE_PATH).symlink_to(os.path.realpath(os.path.join(REPO, reg.VAE_PATH)))
    assert reg.check_untracked("neuronic", str(root)) == []
    (root / "AcousticRooms").unlink()
    (root / "AcousticRooms").symlink_to(fake_vae)                # a file, not the dataset tree
    assert reg.check_untracked("neuronic", str(root))
    (root / "AcousticRooms").unlink()
    assert any("required link AcousticRooms" in p for p in reg.check_untracked("neuronic", str(root)))
    (root / "AcousticRooms").symlink_to(data)
    (root / "stray.py").write_text("x = 1\n")
    (root / "src").mkdir()
    (root / "src" / "flash_attn.py").write_text("")
    probs = reg.check_untracked("neuronic", str(root))
    assert any("stray.py" in p for p in probs) and any("import shadow" in p for p in probs)


PAUSED_WRITER = """
import json, os, sys, time
sys.path.insert(0, sys.argv[2])
import exp46_registry as reg
reg.env_problems = lambda: []
reg.env_snapshot = lambda platform: {k: k for k in reg.ENV_RECEIPT_KEYS}
real_replace = os.replace

def paused_replace(src, dst):                         # the first publication pauses with its temp file on disk
    print("PAUSED " + os.path.basename(src), flush=True)
    time.sleep(120)
    return real_replace(src, dst)
os.replace = paused_replace
reg.ensure_env_receipt("neuronic", repo=sys.argv[1])
"""
RESUMABLE_WRITER = """
import os, sys
sys.path.insert(0, sys.argv[2])
import exp46_registry as reg
reg.env_problems = lambda: []
reg.env_snapshot = lambda platform: {k: k for k in reg.ENV_RECEIPT_KEYS}
real_replace, paused = os.replace, []

def paused_replace(src, dst):                         # the publication's first rename waits for a line on stdin
    if not paused:
        paused.append(src)
        print("PAUSED " + os.path.basename(src), flush=True)
        sys.stdin.readline()
    return real_replace(src, dst)
os.replace = paused_replace
reg.ensure_env_receipt("neuronic", repo=sys.argv[1])
print("PUBLISHED", flush=True)
"""


def test_the_real_hygiene_gate_tolerates_a_live_writers_receipt_temp_file_only(tmp_path, monkeypatch):
    """Codex code r2 finding 11: while another SMOKE publishes the environment receipt (it holds the receipt lock, its
    temp file .c4flip_data_curve_env_receipt_neuronic.json.sha256.tmp.<pid> on disk), the REAL check_untracked of an
    arriving SMOKE passes; once that writer is dead (lock free) the same temp file is refused as stale, by name.
    Codex code r3 finding 2: a publication that COMPLETES between the scan's git listing and its inspection (renamed,
    unlocked, exited) leaves a listed name that has vanished -- ignored, never reported stale."""
    import signal
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root = _git_tree(tmp_path)
    data = tmp_path / "AR"
    data.mkdir()
    (root / "AcousticRooms").symlink_to(data)
    (root / reg.VAE_PATH).symlink_to(os.path.realpath(os.path.join(REPO, reg.VAE_PATH)))
    (root / reg.EXP_DIR).mkdir(parents=True)
    assert reg.check_untracked("neuronic", str(root)) == []
    child = subprocess.Popen([sys.executable, "-I", "-c", PAUSED_WRITER, str(root), EXP], stdout=subprocess.PIPE,
                             text=True)
    try:
        line = child.stdout.readline().strip()
        assert line.startswith("PAUSED .c4flip_data_curve_env_receipt_neuronic.json"), line
        tmp_name = line.split()[1]
        assert os.path.exists(root / reg.EXP_DIR / tmp_name)
        assert reg.check_untracked("neuronic", str(root)) == []               # a live writer's temp file
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    assert os.path.exists(root / reg.EXP_DIR / tmp_name)                      # the killed writer's leftover
    probs = reg.check_untracked("neuronic", str(root))
    assert len(probs) == 1 and tmp_name in probs[0] and "stale" in probs[0], probs
    (root / reg.EXP_DIR / ".c4flip_data_curve_env_receipt_neuronic.json.tmp.junk").write_text("x")
    assert any(".tmp.junk" in p for p in reg.check_untracked("neuronic", str(root)))   # only the kit's exact pattern
    for name in (tmp_name, ".c4flip_data_curve_env_receipt_neuronic.json.tmp.junk"):        # (removed after checking)
        os.remove(root / reg.EXP_DIR / name)
    assert reg.check_untracked("neuronic", str(root)) == []
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", RESUMABLE_WRITER, str(root), EXP],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    real_git_out, live = reg._git_out, []

    def listing_then_the_publication_completes(repo, *args):
        out = real_git_out(repo, *args)
        if "--others" in args and "--ignored" not in args and child.poll() is None:
            assert live[0].encode() in out                                # git listed the live writer's temp name
            child.stdin.write("go\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "PUBLISHED"         # renamed, unlocked ...
            child.wait(timeout=60)                                        # ... and gone
        return out
    try:
        line = child.stdout.readline().strip()
        assert line.startswith("PAUSED .c4flip_data_curve_env_receipt_neuronic.json"), line
        live.append(line.split()[1])
        monkeypatch.setattr(reg, "_git_out", listing_then_the_publication_completes)
        assert reg.check_untracked("neuronic", str(root)) == []
        assert child.returncode == 0 and not os.path.lexists(root / reg.EXP_DIR / live[0])
        assert os.path.exists(root / reg.env_receipt_rel("neuronic"))
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)


@pytest.mark.skipif(not all(os.path.exists(t) for t in reg.A6000_LINK_TARGETS.values()), reason="A6000 box only")
def test_a6000_links_need_the_registered_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root = _git_tree(tmp_path)
    for rel, target in reg.A6000_LINK_TARGETS.items():
        (root / rel).symlink_to(target)
    assert reg.check_untracked("a6000", str(root)) == []
    (root / "AcousticRooms").unlink()
    (root / "AcousticRooms").symlink_to(os.path.realpath(reg.A6000_LINK_TARGETS["AcousticRooms"]) + "/")
    assert reg.check_untracked("a6000", str(root))                # same tree, another spelling


# ------------------------------------------------------------------ storage
def test_the_storage_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "mount_fstype", lambda path: ("ext4", "/"))
    with pytest.raises(reg.GateError):
        reg.storage_gate("a6000", str(tmp_path), 1)               # a local disk in place of the NAS
    monkeypatch.setattr(reg, "mount_fstype", lambda path: ("cifs", "/media/diskstation"))
    out = reg.storage_gate("a6000", str(tmp_path), 1)
    assert out["fstype"] == "cifs" and os.listdir(tmp_path) == []  # the probe cleans up
    with pytest.raises(reg.GateError):
        reg.storage_gate("a6000", str(tmp_path), 2 ** 62)          # the block budget
    with pytest.raises(reg.GateError):
        reg.storage_gate("a6000", str(tmp_path / "unmounted"), 1)  # never created
    monkeypatch.setattr(reg, "mount_fstype", lambda path: ("nfs4", "/n/fs"))
    assert reg.storage_gate("neuronic", str(tmp_path), 1)["fstype"] == "nfs4"
    with pytest.raises(reg.GateError):
        reg.storage_gate("a6000", str(tmp_path), 1)


@pytest.mark.skipif(not os.path.isdir("/media/diskstation/yixunhu/FLAC"), reason="NAS not mounted")
def test_mount_fstype_sees_the_nas_behind_autofs():
    assert reg.mount_fstype("/media/diskstation/yixunhu/FLAC/checkpoints")[0] == "cifs"


# ------------------------------------------------------------------ train.py logger metadata (NIT 1)
def test_logger_metadata_without_the_flag_is_the_pre_flag_dict():
    sys.path.insert(0, REPO)
    import argparse
    import train
    a = argparse.Namespace(seed=42, evidence_json="")
    d = train.logger_args_dict(a)
    assert d == {"seed": 42} and not hasattr(a, "evidence_json") and train.extra_callbacks(a) == []
    b = argparse.Namespace(seed=42, evidence_json="/x/run_evidence.json")
    assert train.logger_args_dict(b) == {"seed": 42, "evidence_json": "/x/run_evidence.json"}


def test_the_commit_marker_needs_the_terminal_record(tree, monkeypatch):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests, commit=False)
    with pytest.raises(reg.GateError, match="committed"):
        reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"])
    digest = reg.sha256_file(str(rd / reg.TRAINING_RECEIPT))
    lr = rd / "at_launch_x.json"
    json.dump({"run": "exp46_C4Ff050_l40_2x32", "status": "end audit failed", "training_receipt_sha256": digest}, open(lr, "w"))
    with pytest.raises(reg.GateError):
        reg.write_commit_marker(str(rd), "exp46_C4Ff050_l40_2x32", digest, str(lr))   # never for a non-COMPLETE record
    fx.commit(reg, str(rd), "exp46_C4Ff050_l40_2x32", digest)
    reg.check_training_receipt("exp46_C4Ff050_l40_2x32", str(rd), closure=CLOSURE, e0_sha=digests["e0"])


def test_the_publication_cli_needs_its_campaign_context(tree, monkeypatch, capsys):
    rd, log, retained, paths, digests = _full_run(tree, monkeypatch)
    _write_tr(tree, rd, log, retained, paths, digests)
    with pytest.raises(SystemExit):
        reg._cli(["training-check", "exp46_C4Ff050_l40_2x32", "--run-dir", str(rd)])
    base = ["training-check", "exp46_C4Ff050_l40_2x32", "--run-dir", str(rd), "--platform", "a6000"]
    assert reg._cli(base + ["--closure", CLOSURE, "--e0-sha", digests["e0"]]) == 0
    assert reg._cli(base + ["--closure", "0" * 64, "--e0-sha", digests["e0"]]) == 2
    assert reg._cli(["training-check", "exp46_C4Ff050_l40_2x32", "--run-dir", str(rd), "--platform", "neuronic",
                     "--closure", CLOSURE, "--e0-sha", digests["e0"]]) == 2


def test_duplicate_l40_uuids_are_not_a_topology():
    assert reg._gpus_ok("neuronic", 2, [["GPU-a", "NVIDIA L40"], ["GPU-b", "NVIDIA L40"]])
    assert not reg._gpus_ok("neuronic", 2, [["GPU-a", "NVIDIA L40"], ["GPU-a", "NVIDIA L40"]])
    assert not reg._gpus_ok("neuronic", 2, [["x", "NVIDIA L40"], ["y", "NVIDIA L40"]])


def test_record_identity_hashes_the_current_installed_bytes(tmp_path):
    """Codex code r3 HIGH 4: a file edited while its RECORD stays the same refuses; an unhashed
    entry still enters the identity through its current bytes."""
    import base64
    site = tmp_path / "site"
    (site / "pkg").mkdir(parents=True)
    (site / "pkg" / "__init__.py").write_text("x = 1\n")
    (site / "pkg" / "data.bin").write_bytes(b"abc")
    b64 = lambda data: base64.urlsafe_b64encode(reg.hashlib.sha256(data).digest()).decode().rstrip("=")
    record = (f"pkg/__init__.py,sha256={b64(b'x = 1' + bytes([10]))},6\n"
              f"pkg/data.bin,,3\npkg/__pycache__/x.cpython-310.pyc,,\npkg-1.0.dist-info/RECORD,,\n")
    (site / "pkg-1.0.dist-info").mkdir()
    (site / "pkg-1.0.dist-info" / "RECORD").write_text(record)
    a = reg.record_identity(str(site), record)
    (site / "pkg" / "data.bin").write_bytes(b"abd")                      # an UNHASHED entry changes
    assert reg.record_identity(str(site), record) != a
    (site / "pkg" / "__init__.py").write_text("x = 2\n")               # a HASHED entry changes
    with pytest.raises(reg.GateError, match="RECORD hash"):
        reg.record_identity(str(site), record)
    os.remove(site / "pkg" / "__init__.py")
    with pytest.raises(reg.GateError, match="missing"):
        reg.record_identity(str(site), record)


def test_an_sdpa_only_environment_records_flash_attn_as_absent(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "flash_attn" else real(name, *a))
    origins, records_ = reg.module_identity()
    assert origins["flash_attn"] == records_["flash_attn"] == "absent" and records_["torch"].startswith("record:")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "einops" else real(name, *a))
    with pytest.raises(reg.GateError):
        reg.module_identity()                                             # a required module may not be absent
