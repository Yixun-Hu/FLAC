"""exp_38 B8/B9 — the evaluation driver (exp38_eval.py) and the collector (exp38_collect.py).

Cell plan and argv (the reference cells' exact shape incl. ``--num-workers 4``), output names
(must equal eval_FLAC's own ``build_output_paths``), recursive type-strict equality for E0, the
exact reference set, the sealed E0 receipt (every field mutated), the transactional cell admission
(one complete synthetic cell incl. a real scored-waveform dump, then one mutation per field: metrics,
by-scene, protocol, campaign, checkpoint, stream, receipt, dump), the inlined dump predicate's
equivalence with eval_FLAC's own ``is_complete``, aggregation, PENDING, and the result pair's
publication order (Codex code r1 BLOCKING 3-7, HIGH 10, 11).
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

import exp38_fixtures as fx

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(name):
    import sys
    sys.path.insert(0, EXP)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_load("exp38_registry")
import exp38_collect as col  # noqa: E402  (one module graph: col.ev is ev, col.reg is ev.reg)
import exp38_eval as ev  # noqa: E402


def _opt(argv, flag):
    return argv[argv.index(flag) + 1]


def test_plan_has_8_e0_and_60_result_cells_with_unique_outputs():
    cells = ev.all_cells()                      # D2 = neuronic: the P1 f025 bridge is part of the campaign
    assert ev.reg.TRAINING_PLATFORM == "neuronic" and len(cells) == 78 and len(ev.all_cells(bridge=True)) == 78
    assert len({c["name"] for c in cells}) == 78 and len({c["metrics"] for c in cells}) == 78
    assert sum(c["e0"] is not None for c in cells) == 8 and sum(c["run"] == "de_p1_f025" for c in cells) == 10
    assert all(c["dump"] == (c["e0"] is None) for c in cells)


@pytest.mark.parametrize("run,cond", [("de_yaw_f050", "vanilla"), ("de_bf_f075", "fa_invariant")])
def test_argv_is_the_reference_cells_shape(run, cond):
    c = [c for c in ev.run_cells(run) if c["K"] == 1 and c["seed"] == 44][0]
    a = ev.eval_argv(c)
    assert _opt(a, "--cond-method") == cond and _opt(a, "--num-workers") == "4" and _opt(a, "--batch-size") == "64"
    assert _opt(a, "--dataset-config").endswith("acousticroom_unseeneval_1.json") and _opt(a, "--seed") == "44"
    assert _opt(a, "--expected-stream-count") == "6337" and "--record-per-scene" in a and "--record-stream" in a
    assert ("--frame-avg-max-fwd-samples" in a) == (cond == "fa_invariant")
    if cond == "fa_invariant":
        assert _opt(a, "--frame-avg-angles") == "0,90,180,270" and _opt(a, "--frame-avg-max-fwd-samples") == "64"
    assert _opt(a, "--dump-scored-waveforms") == ev.DUMP_ROOT
    assert _opt(a, "--model-config") == c["config"]


def test_output_names_equal_eval_flacs_own_builder():
    import eval_FLAC
    for c in ev.all_cells():
        want = eval_FLAC.build_output_paths(c["ckpt_link"], 1, 1.0, c["name"], cond_method=c["cond"], n_angles=4)
        assert want["metrics"] == c["metrics"], c["name"]


def test_strict_equal_is_recursive_and_type_strict():
    assert ev.strict_equal({"a": [1.0, {"b": 2}]}, {"a": [1.0, {"b": 2}]}) is None
    assert ev.strict_equal({"a": 1}, {"a": 1.0})
    assert ev.strict_equal({"a": True}, {"a": 1})
    assert ev.strict_equal({"a": 1}, {"a": 1, "b": 2})
    assert ev.strict_equal([1, 2], [1, 2, 3])
    assert ev.strict_equal({"x": {"y": 0.30000000000000004}}, {"x": {"y": 0.3}})


def test_input_hashes_and_references_are_the_pinned_real_ones():
    assert len(ev.INPUT_HASH) == 10
    stream = os.path.join(REPO, "outputs_FLAC/exp25_depth_canon/P1/epoch=8-step=40000_metrics_1_1.0_exp25_P1_T_S40000_K8_s44.stream.json")
    if os.path.exists(stream):
        assert json.load(open(stream))["input_hash"] == ev.INPUT_HASH[(8, 44)]
    refs = ev.check_refs()
    assert len(refs) == 68 and sorted(refs) == ev.ref_set()
    assert ev.reg.sha256_file(os.path.join(EXP, "ref_cells/E0/E0a_K8_s42.json")) == \
        "4136ac75d570f84f76d34c5cac0f1e05e681897f4aefe53ffce12dd381f13180"
    assert ev.reg.sha256_file(os.path.join(EXP, "ref_cells/E0/E0b_K8_s42.json")) == \
        "f167470e3501bcc7c4f116b0c26164523a8bd140b55c5e72e935d88d3f373adb"


def test_reference_set_is_exact(tmp_path, monkeypatch):
    """Codex code r1 BLOCKING 7: exactly the 68 registered files -- no extra, missing or duplicate line."""
    import shutil
    root = tmp_path / "ref_cells"
    shutil.copytree(os.path.join(EXP, "ref_cells"), root)
    monkeypatch.setattr(ev, "EXP", str(tmp_path / "x"))
    monkeypatch.setattr(ev, "REF_MANIFEST", str(root / "REF_MANIFEST.sha256"))
    (tmp_path / "x").mkdir()
    os.symlink(root, tmp_path / "x" / "ref_cells")
    assert len(ev.check_refs()) == 68
    man = root / "REF_MANIFEST.sha256"
    lines = man.read_text().strip().split("\n")
    for bad in (lines[:-1], lines + [lines[0]], lines[:-1] + [lines[-1].replace(".json", "_x.json")]):
        man.write_text("\n".join(bad) + "\n")
        with pytest.raises(ev.EvalError):
            ev.check_refs()
    man.write_text("\n".join(lines) + "\n")
    (root / "anchors" / "BF" / "BF_K8_s47.json").write_text("{}")
    with pytest.raises(ev.EvalError):
        ev.check_refs()                                          # an unmanifested anchor on disk
    os.remove(root / "anchors" / "BF" / "BF_K8_s47.json")
    f = root / "E0" / "E0a_K8_s42.json"
    f.write_bytes(f.read_bytes() + b" ")
    with pytest.raises(ev.EvalError):
        ev.check_refs()


# ------------------------------------------------------------------ the sealed E0 receipt
@pytest.fixture
def e0(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "E0_RECEIPT", str(tmp_path / "e0_receipt.json"))
    monkeypatch.setattr(ev, "FARM", str(tmp_path / "farm"))           # every cell path lands in tmp
    monkeypatch.setattr(ev.reg, "closure_digest", lambda repo=None: {"closure_sha256": fx.CAMPAIGN["closure_sha256"]})
    monkeypatch.setattr(ev.reg, "env_receipt_sha", lambda platform, repo=None: fx.CAMPAIGN["env_receipt_sha256"])
    monkeypatch.setattr(ev, "eval_snapshot", lambda closure, create=False: (str(tmp_path / "snap"),
                                                                           fx.CAMPAIGN["eval_snapshot_sha256"]))
    monkeypatch.setattr(ev, "live_eval_pins", lambda: {"agree_sha256": ev.reg.AGREE_SHA256,
                                                       "vae_ckpt_sha256": ev.reg.VAE_CKPT_SHA256, "vit_rev": ev.reg.VIT_REV})
    campaign, rec = fx.make_e0_receipt(ev, log_dir=str(tmp_path / "logs"))
    return campaign, rec


def test_e0_receipt_valid_online_and_offline(e0):
    campaign, _ = e0
    for offline in (False, True):
        got = ev.validate_e0_receipt(offline=offline)
        assert got["sha256"] == campaign["sha256"] and got["head"] == campaign["head"]


def _reseal(path, rec):
    for p in (path, path + ".sha256"):
        os.remove(p)
    ev.reg.write_sealed(path, rec)


E0_MUTATIONS = {
    "verdict": lambda r: r.update(verdict="FAIL"), "failed": lambda r: r.update(failed=["de_E0a_K8_s42"]),
    "cell missing": lambda r: r["cells"].pop("de_E0b_K1_s42"),
    "cell unequal": lambda r: r["cells"]["de_E0c_K8_s42"].update(equal=False),
    "cell difference": lambda r: r["cells"]["de_E0c_K8_s42"].update(difference="$.metrics.T60"),
    "cell problems": lambda r: r["cells"]["de_E0d_K1_s42"].update(problems=["x"]),
    "cell reference": lambda r: r["cells"]["de_E0a_K1_s42"].update(reference="E0/E0b_K1_s42.json"),
    "cell extra key": lambda r: r["cells"]["de_E0a_K1_s42"].update(extra=1),
    "checkpoint sha": lambda r: r["checkpoints"]["E0b"].update(sha256="0" * 64),
    "checkpoint bytes": lambda r: r["checkpoints"]["E0b"].update(bytes=1),
    "input hash": lambda r: r["input_hash"].update(K8_s42="0" * 64),
    "agree pin": lambda r: r["eval_pins"].update(agree_sha256="0" * 64),
    "references": lambda r: r["references"].update({"E0/E0a_K8_s42.json": "0" * 64}),
    "ref manifest": lambda r: r.update(reference_manifest_sha256="0" * 64),
    "evaluator set": lambda r: r["evaluator_files"].pop("eval_FLAC.py"),
    "closure": lambda r: r.update(closure_sha256="0" * 64), "env": lambda r: r.update(env_receipt_sha256="0" * 64),
    "evaluator file": lambda r: r["evaluator_files"].update({"eval_FLAC.py": "0" * 64}),
    "head": lambda r: r.update(head="short"), "extra key": lambda r: r.update(extra=1),
    "missing key": lambda r: r.pop("written_at"),
    # valid-looking but wrong cell bindings (Codex code r2 HIGH 3): re-hashed, not syntax-checked
    "cell metrics sha": lambda r: r["cells"]["de_E0b_K8_s42"].update(metrics_sha256="a" * 64),
    "cell receipt sha": lambda r: r["cells"]["de_E0c_K1_s42"].update(receipt_sha256="b" * 64),
    "snapshot": lambda r: r.update(eval_snapshot_sha256="6" * 64),
    "head (other campaign)": lambda r: r.update(head="e" * 40),
}


@pytest.mark.parametrize("name", sorted(E0_MUTATIONS))
def test_e0_receipt_refuses_each_mutation(e0, name):
    _, rec = e0
    rec = json.loads(json.dumps(rec))
    E0_MUTATIONS[name](rec)
    _reseal(ev.E0_RECEIPT, rec)
    offline_ok = name == "evaluator file"       # the evaluator-file VALUES bind THIS tree: online-only
    with pytest.raises(ev.reg.GateError):
        ev.validate_e0_receipt()
    if offline_ok:
        ev.validate_e0_receipt(offline=True)
    else:
        with pytest.raises(ev.reg.GateError):
            ev.validate_e0_receipt(offline=True)


def test_e0_cells_are_re_admitted_and_re_compared(e0):
    """The E0 cells themselves (not just the receipt): a changed metric or a changed cell receipt
    refuses even though the sealed E0 receipt is untouched."""
    c = [c for c in ev.e0_cells() if c["name"] == "de_E0d_K8_s42"][0]
    rec = json.load(open(ev._abs(c["metrics"])))
    rec["metrics"]["T60"] += 1e-9
    json.dump(rec, open(ev._abs(c["metrics"]), "w"))
    for kw in ({"offline": False}, {"offline": True}, {"platform": "neuronic"}):
        with pytest.raises(ev.reg.GateError):
            ev.validate_e0_receipt(**kw)                   # neuronic re-admits the copied cells too (r3 HIGH 5)
    with pytest.raises(ev.reg.GateError):
        ev.reg.check_e0_receipt("neuronic")


def test_the_neuronic_gate_re_admits_the_copied_e0_cells(e0):
    campaign, rec = e0
    assert ev.reg.check_e0_receipt("neuronic")["sha256"] == campaign["sha256"]
    c = ev.e0_cells()[3]
    os.remove(ev._abs(c["receipt"]))                       # a cell not copied to neuronic
    with pytest.raises(ev.reg.GateError):
        ev.reg.check_e0_receipt("neuronic")


def test_e0_receipt_sidecar_is_binding(e0):
    path = ev.E0_RECEIPT
    open(path, "ab").write(b" ")
    with pytest.raises(ev.reg.GateError):
        ev.validate_e0_receipt(offline=True)


# ------------------------------------------------------------------ transactional admission
@pytest.fixture
def world(tmp_path, monkeypatch, e0):
    campaign, _ = e0
    monkeypatch.setattr(ev, "DUMP_ROOT", str(tmp_path / "dumps"))
    monkeypatch.setattr(ev, "DUMP_SHAPE", [6337, 4])
    monkeypatch.setitem(ev.reg.PLATFORMS["a6000"], "save_root", str(tmp_path / "nas"))
    (tmp_path / "dumps").mkdir()
    return {"tmp": tmp_path, "campaign": campaign}


def _cell(world, run="de_bf_f025", K=8, seed=43, dump=True):
    tmp = world["tmp"]
    cell = [c for c in ev.run_cells(run) if c["K"] == K and c["seed"] == seed][0]
    farm = os.path.dirname(cell["ckpt_link"])
    rd = tmp / "nas" / run
    if not (rd / ev.reg.TRAINING_RECEIPT).exists():
        fx.make_training_receipt(ev, run, str(rd), world["campaign"]["closure_sha256"],
                                 e0_sha=world["campaign"]["sha256"], e0_path=world["campaign"]["e0_receipt"])
        fx.copy_to_farm(ev, str(rd), farm)
        final = ev.reg.ckpt_name(ev.reg.MAX_STEPS, ev.reg.parse_run_id(run)[1])
        os.symlink(str(rd / final), os.path.join(farm, final))          # what `prepare` links
    tr, digest = ev.reg.read_sealed(str(rd / ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
    return fx.make_cell(ev, cell, world["campaign"], (tr["final_ckpt_sha256"], tr["final_ckpt_bytes"], digest),
                        str(tmp / "logs"), dump=dump)


def test_validate_cell_admits_a_complete_cell(world):
    cell = _cell(world)
    assert ev.validate_cell(cell) == []
    assert ev.validate_cell(cell, world["campaign"]) == []
    vcell = _cell(world, run="de_yaw_f050", K=1, seed=46)
    assert ev.validate_cell(vcell) == []


def _mut(path, fn):
    rec = json.load(open(path))
    fn(rec)
    json.dump(rec, open(path, "w"))


METRIC_MUTATIONS = {
    "metrics key missing": lambda r: r["metrics"].pop("FD"), "metrics key extra": lambda r: r["metrics"].update(X=1.0),
    "metric nan": lambda r: r["metrics"].update(T60=float("nan")), "metric string": lambda r: r["metrics"].update(T60="8.2"),
    "metric None": lambda r: r["metrics"].update(C50=None), "metric bool": lambda r: r["metrics"].update(EDT=True),
    "by_scene missing": lambda r: r["by_scene"].pop("Cafe"), "by_scene extra": lambda r: r["by_scene"].update(Garage={}),
    "by_scene inner key": lambda r: r["by_scene"]["Office"].pop("FD"),
    "by_scene inner string": lambda r: r["by_scene"]["Office"].update(FD="x"),
    "by_scene inner inf": lambda r: r["by_scene"]["Office"].update(FD=float("inf")),
    "top extra": lambda r: r.update(extra=1), "top missing": lambda r: r.pop("weights_source"),
    "cond": lambda r: r.update(cond_method="vanilla"), "n_samples": lambda r: r.update(n_samples=6336),
    "cap": lambda r: r.update(frame_avg_fwd_cap=32), "angles": lambda r: r.update(frame_avg_angles=[0.0, 90.0]),
    "autocast": lambda r: r.update(cond_autocast="off"), "seed": lambda r: r.update(seed=44),
    "seed type": lambda r: r.update(seed=43.0), "rotate": lambda r: r.update(rotate_deg=45.0),
    "weights": lambda r: r.update(weights_source="online"), "ckpt path": lambda r: r.update(ckpt_path="/x.ckpt"),
    "ckpt sha": lambda r: r.update(ckpt_sha256_loaded="b" * 64), "ckpt bytes": lambda r: r.update(ckpt_bytes_loaded=1),
    "ckpt bytes type": lambda r: r.update(ckpt_bytes_loaded=float(r["ckpt_bytes_loaded"])),
    "source sha": lambda r: r.update(source_sha="e" * 40),
}


@pytest.mark.parametrize("name", sorted(METRIC_MUTATIONS))
def test_validate_cell_refuses_each_metrics_mutation(world, name):
    cell = _cell(world)
    _mut(cell["metrics"], METRIC_MUTATIONS[name])
    assert ev.validate_cell(cell) != []


def _receipt_mutations():
    out = {k: (lambda r, k=k: r.pop(k)) for k in sorted(ev.RECEIPT_KEYS)}
    out = {f"missing {k}": v for k, v in out.items()}
    out.update({
        "extra": lambda r: r.update(extra=1), "rc": lambda r: r.update(rc=1),
        "sentinel": lambda r: r.update(sentinel="EXP35_NO_FLASH_LAUNCHER attention_backend=flash"),
        "argv": lambda r: r.update(argv=r["argv"][:-2]), "cell": lambda r: r.update(cell="de_bf_f025_K8_s44"),
        "head": lambda r: r.update(head="e" * 40), "closure": lambda r: r.update(closure_sha256="0" * 64),
        "e0": lambda r: r.update(e0_receipt_sha256="0" * 64), "e0 none": lambda r: r.update(e0_receipt_sha256=None),
        "env": lambda r: r.update(env_receipt_sha256="0" * 64), "agree": lambda r: r.update(agree_sha256="0" * 64),
        "vae": lambda r: r.update(vae_ckpt_sha256="0" * 64), "vit": lambda r: r.update(vit_rev="main"),
        "model cfg": lambda r: r.update(model_config_sha256="0" * 64),
        "data cfg": lambda r: r.update(dataset_config_sha256="0" * 64),
        "split": lambda r: r.update(unseen_split_sha256="0" * 64), "metrics sha": lambda r: r.update(metrics_sha256="0" * 64),
        "stream sha": lambda r: r.update(stream_sha256="0" * 64), "dump sha": lambda r: r.update(dump_manifest_sha256="0" * 64),
        "ckpt sha": lambda r: r.update(ckpt_sha256="0" * 64), "ckpt bytes": lambda r: r.update(ckpt_bytes=2),
        "training receipt": lambda r: r.update(training_receipt_sha256="0" * 64),
        "gpu name": lambda r: r.update(gpu_name="NVIDIA L40"), "gpu uuid": lambda r: r.update(gpu_uuid=ev.reg.A6000_UUIDS[0]),
        "gpu index": lambda r: r.update(gpu=2), "gpu bool": lambda r: r.update(gpu=True),
        "log sha": lambda r: r.update(log_sha256="0" * 64), "log gone": lambda r: r.update(log="/nonexistent.log"),
        "eval snapshot": lambda r: r.update(eval_snapshot_sha256="6" * 64),
        "time": lambda r: r.update(started_at=None)})
    return out


@pytest.mark.parametrize("name", sorted(_receipt_mutations()))
def test_validate_cell_refuses_each_receipt_mutation(world, name):
    cell = _cell(world)
    _mut(cell["receipt"], _receipt_mutations()[name])
    assert ev.validate_cell(cell) != []


def test_admission_checks_the_current_evaluator_weights(world, monkeypatch):
    cell = _cell(world)
    assert ev.validate_cell(cell) == []
    for k in ("agree_sha256", "vae_ckpt_sha256", "vit_rev"):
        live = {"agree_sha256": ev.reg.AGREE_SHA256, "vae_ckpt_sha256": ev.reg.VAE_CKPT_SHA256, "vit_rev": ev.reg.VIT_REV}
        live[k] = "0" * 64
        monkeypatch.setattr(ev, "live_eval_pins", lambda live=live: dict(live))
        assert ev.validate_cell(cell) != [], k
        assert ev.validate_cell(cell, offline=True) == []        # offline: the receipt's own pins only


def test_a_cell_is_claimed_exactly_once(world):
    cell = _cell(world, seed=46)
    ev.claim_cell(cell)
    with pytest.raises(ev.EvalError, match="claimed"):
        ev.claim_cell(cell)                                       # a concurrent / crashed run
    import multiprocessing as mp
    other = [c for c in ev.run_cells("de_bf_f025") if c["seed"] == 44 and c["K"] == 1][0]
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        got = pool.map(_try_claim, [other] * 4)
    assert sorted(got) == [False, False, False, True]


def _try_claim(cell):
    try:
        ev.claim_cell(cell)
        return True
    except ev.EvalError:
        return False


def test_run_one_executes_from_the_snapshot_while_the_tree_changes(world, monkeypatch, tmp_path):
    """A fake evaluator mutates the campaign checkout DURING the cell: the cell ran the snapshot's
    bytes, the snapshot verifies after the cell, and the claim is released only on admission."""
    run = "de_bf_f025"
    cell = [c for c in ev.run_cells(run) if c["K"] == 8 and c["seed"] == 45][0]
    _cell(world, seed=43)                                        # the training receipt + farm copies
    for p in (cell["metrics"], cell["stream"], cell["receipt"]):
        assert not os.path.exists(p)
    snap = tmp_path / "evalsnap"
    snap.mkdir()
    (snap / "eval_FLAC.py").write_text("ORIGINAL\n")
    tree_file = tmp_path / "tree_eval_FLAC.py"
    tree_file.write_text("ORIGINAL\n")
    calls = {}
    monkeypatch.setattr(ev, "eval_snapshot", lambda closure, create=False: (str(snap), fx.CAMPAIGN["eval_snapshot_sha256"]))
    monkeypatch.setattr(ev.reg, "verify_snapshot", lambda d, want=None: {"snapshot_sha256": fx.CAMPAIGN["eval_snapshot_sha256"],
                                                                          "verified": (snap / "eval_FLAC.py").read_text() == "ORIGINAL\n"})
    monkeypatch.setattr(ev.reg, "eval_pins", lambda repo=None: {"agree_sha256": ev.reg.AGREE_SHA256,
                                                                "vae_ckpt_sha256": ev.reg.VAE_CKPT_SHA256, "vit_rev": ev.reg.VIT_REV})
    monkeypatch.setattr(ev, "gpu_row", lambda gpu: (ev.reg.A6000_UUIDS[gpu], ev.GPU_NAME))
    monkeypatch.setattr(ev, "EVAL_LOG_DIR", str(tmp_path / "eval_logs"))
    monkeypatch.setattr(ev, "eval_env", lambda gpu: {"GIT_DIR": "/x"})
    tr, digest = ev.reg.read_sealed(str(world["tmp"] / "nas" / run / ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)

    def fake_run(cmd, cwd, env, log, lock_fds=(), grace=60):
        calls["cwd"], calls["ran"] = cwd, open(os.path.join(cwd, "eval_FLAC.py")).read()
        tree_file.write_text("MUTATED\n")                         # the checkout changes mid-cell
        open(log, "a").write(ev.SENTINEL + "\n")
        fx.make_cell(ev, cell, world["campaign"], (tr["final_ckpt_sha256"], tr["final_ckpt_bytes"], digest),
                     str(world["tmp"] / "logs2"))
        os.remove(cell["receipt"])                                 # the driver writes the receipt
        return 0
    monkeypatch.setattr(ev, "run_owned", fake_run)
    monkeypatch.setattr(ev, "validate_cell", lambda c, camp=None, offline=False: [] if os.path.exists(c["receipt"]) else ["absent"])
    assert ev.run_one(cell, 1, world["campaign"]) == "done"
    assert calls["cwd"] == str(snap) and calls["ran"] == "ORIGINAL\n" and tree_file.read_text() == "MUTATED\n"
    rec = json.load(open(cell["receipt"]))
    assert rec["eval_snapshot_sha256"] == fx.CAMPAIGN["eval_snapshot_sha256"] and not os.path.exists(ev.claim_path(cell))


def test_validate_cell_refuses_stream_and_campaign_mutations(world):
    cell = _cell(world)
    _mut(cell["stream"], lambda r: r.update(input_hash="0" * 64))
    assert ev.validate_cell(cell) != []
    cell = _cell(world, seed=44)
    other = dict(world["campaign"], closure_sha256="0" * 64)
    assert ev.validate_cell(cell, other) != []                  # a different campaign
    os.remove(ev.E0_RECEIPT)
    assert ev.validate_cell(cell) != []                         # no E0 receipt, no admission


def _dump_mutations():
    import numpy as np
    def npy(d, arr):
        np.save(os.path.join(d, "waveforms.npy"), arr, allow_pickle=False)
    return {
        "COMPLETE missing": lambda d: os.remove(os.path.join(d, "COMPLETE")),
        "COMPLETE wrong": lambda d: open(os.path.join(d, "COMPLETE"), "w").write("ok"),
        "sidecar wrong": lambda d: open(os.path.join(d, "waveforms.sha256"), "w").write("0" * 64 + "  waveforms.npy\n"),
        "npy missing": lambda d: os.remove(os.path.join(d, "waveforms.npy")),
        "npy bytes": lambda d: npy(d, np.zeros((6337, 4), dtype=np.float32)),
        "npy dtype": lambda d: npy(d, np.zeros((6337, 4), dtype=np.float64)),
        "npy rows": lambda d: npy(d, np.zeros((6336, 4), dtype=np.float32)),
    }


@pytest.mark.parametrize("name", sorted(_dump_mutations()))
def test_validate_cell_refuses_each_dump_mutation_like_eval_flacs_is_complete(world, name):
    import eval_FLAC
    cell = _cell(world)
    assert eval_FLAC.ScoredWaveformDump.is_complete(cell["dump_dir"], expected_n=6337)
    _dump_mutations()[name](cell["dump_dir"])
    assert ev.validate_cell(cell) != []
    assert not eval_FLAC.ScoredWaveformDump.is_complete(cell["dump_dir"], expected_n=6337)   # the same verdict
    assert ev.DUMP_SCHEMA_VERSION == eval_FLAC.SCORED_DUMP_SCHEMA_VERSION


def test_dump_manifest_shape_stem_and_cross_links(world, monkeypatch):
    cell = _cell(world)
    man = os.path.join(cell["dump_dir"], "manifest.json")
    for fn in (lambda m: m.update(metrics_stem="other"), lambda m: m.update(source_sha="e" * 40),
               lambda m: m.update(metrics_sha256="0" * 64), lambda m: m.update(schema_version=2)):
        c = _cell(world, seed=45)
        man = os.path.join(c["dump_dir"], "manifest.json")
        _mut(man, fn)
        open(ev.dump_manifest_copy(c), "wb").write(open(man, "rb").read())
        _mut(c["receipt"], lambda r: r.update(dump_manifest_sha256=ev.reg.sha256_file(man)))
        assert ev.validate_cell(c) != []
        import shutil
        shutil.rmtree(c["dump_dir"])
    monkeypatch.setattr(ev, "DUMP_SHAPE", [6337, 5])             # the registered T binds too
    assert ev.validate_cell(cell) != []


def test_training_receipt_binds_the_checkpoint(world):
    cell = _cell(world)
    rd = world["tmp"] / "nas" / "de_bf_f025"
    rec, _ = ev.reg.read_sealed(str(rd / ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
    rec["final_ckpt_bytes"] += 1
    _reseal(str(rd / ev.reg.TRAINING_RECEIPT), rec)
    assert ev.validate_cell(cell) != []
    with pytest.raises(ev.reg.GateError):
        ev.reg.check_training_receipt("de_bf_f025", str(rd), closure=fx.CAMPAIGN["closure_sha256"])


def test_offline_validation_needs_no_nas(world):
    import shutil
    cell = _cell(world)
    shutil.rmtree(world["tmp"] / "nas")
    shutil.rmtree(cell["dump_dir"])
    assert ev.validate_cell(cell) != []                          # online needs the NAS
    assert ev.validate_cell(cell, offline=True) == []
    open(ev.dump_manifest_copy(cell), "a").write(" ")
    assert ev.validate_cell(cell, offline=True) != []            # the copy is sha-bound


def test_e0_cells_validate_under_the_current_campaign(world, monkeypatch):
    cell = [c for c in ev.e0_cells() if c["e0"] == "E0b" and c["K"] == 8][0]
    camp = dict(fx.CAMPAIGN, sha256=None)
    monkeypatch.setattr(ev, "current_campaign", lambda: camp)
    assert ev.validate_cell(cell) == []
    _mut(cell["metrics"], lambda r: r.update(ckpt_bytes_loaded=ev.E0_BYTES["E0a"]))
    assert ev.validate_cell(cell) != []


def test_prepare_requires_the_training_receipt(world, monkeypatch):
    rd = world["tmp"] / "nas" / "de_yaw_f025"
    with pytest.raises(ev.reg.GateError):
        ev.prepare("de_yaw_f025")                                # no TRAINING_COMPLETE
    fx.make_training_receipt(ev, "de_yaw_f025", str(rd), "0" * 64)
    with pytest.raises(ev.reg.GateError):
        ev.prepare("de_yaw_f025")                                # trained at another closure
    rd2 = world["tmp"] / "nas" / "de_yaw_f050"
    fx.make_training_receipt(ev, "de_yaw_f050", str(rd2), fx.CAMPAIGN["closure_sha256"])
    with pytest.raises(ev.reg.GateError, match="another E0"):
        ev.prepare("de_yaw_f050")                                # trained under another E0 receipt


# ------------------------------------------------------------------ collector
def test_aggregate_and_seed_set():
    recs = {s: {k: float(s) for k in ev.METRIC_KEYS} for s in ev.SEEDS}
    st = col.aggregate(recs)
    assert st["T60"][0] == 44.0 and abs(st["T60"][1] - 1.5811388300841898) < 1e-12 and st["T60"][2] == 5
    with pytest.raises(ValueError):
        col.aggregate({s: recs[s] for s in (42, 43, 44, 45)})


def test_reference_rows_reproduce_table_3b_and_bind_to_e0(e0):
    campaign, rec = e0
    rows, used = col.reference_rows(campaign)
    assert len(used) == 60
    t = {(m, f): rows[(m, f, 8)]["stats"] for (m, f, K) in rows if K == 8}
    assert round(t[("p1", "025")]["EDT"][0], 3) == 42.914 and round(t[("p1", "025")]["T60"][0], 2) == 10.25
    assert round(t[("p1", "100")]["EDT"][0], 3) == 40.650 and round(t[("p1", "100")]["T60"][0], 3) == 8.993
    assert round(t[("yaw", "100")]["T60"][0], 3) == 7.965 and round(t[("bf", "100")]["EDT"][0], 3) == 38.794
    rec["references"]["anchors/P1/P1_K8_s42.json"] = "0" * 64
    json.dump(rec, open(campaign["e0_receipt"], "w"))
    with pytest.raises(ev.EvalError):
        col.reference_rows(campaign)


def test_run_block_pending_on_any_problem_and_render(e0):
    campaign, _ = e0
    rows, probs, man = col.run_block("de_yaw_f025", campaign, validate=lambda c: [f"{c['name']}: missing"])
    assert rows == {} and man == {} and len(probs) == 10
    md = col.render(col.reference_rows(campaign)[0], probs)
    assert md.count("PENDING") >= 12 and "de_yaw_f025_K8_s42: missing" in md


def test_publish_order_markdown_first_json_last(tmp_path, e0, monkeypatch):
    campaign, _ = e0
    exp = tmp_path / col.reg.EXP_DIR
    exp.mkdir(parents=True)
    rows = col.reference_rows(campaign)[0]
    sha = col.publish(rows, [], {"m": 1}, repo=str(tmp_path))
    j = (tmp_path / col.OUT_JSON).read_bytes()
    assert hashlib.sha256(j).hexdigest() == sha and sha in (tmp_path / col.OUT_MD).read_text()
    assert json.loads(j)["manifest"] == {"m": 1}
    assert not [p for p in exp.iterdir() if ".tmp" in p.name]
    order, real = [], col.reg.atomic_write_bytes

    def spy(path, data):
        order.append(os.path.basename(path))
        if path.endswith(".json"):
            raise OSError("crash before the commit point")
        real(path, data)
    monkeypatch.setattr(col.reg, "atomic_write_bytes", spy)
    with pytest.raises(OSError):
        col.publish(rows, ["x"], {"m": 2}, repo=str(tmp_path))
    assert order == [os.path.basename(col.OUT_MD), os.path.basename(col.OUT_JSON)]
    assert (tmp_path / col.OUT_JSON).read_bytes() == j           # the old JSON stands; the MD names a sha it lacks
    assert hashlib.sha256(j).hexdigest() not in (tmp_path / col.OUT_MD).read_text()


def test_pending_cells_fail_unless_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(col, "collect", lambda bridge=False: ({}, ["de_yaw_f025: pending"], {}))
    monkeypatch.setattr(col, "publish", lambda rows, pending, manifest, repo=None: "0" * 64)
    assert col.main([]) == 1
    assert col.main(["--partial"]) == 0
    monkeypatch.setattr(col, "collect", lambda bridge=False: ({}, [], {}))
    assert col.main([]) == 0


def test_the_result_manifest_lists_every_admitted_artifact(world, monkeypatch):
    """Codex code r2 HIGH 7: a path -> sha256 entry for every file the admission consumed."""
    for K in (8, 1):
        for s_ in ev.SEEDS:
            _cell(world, K=K, seed=s_)
    campaign = world["campaign"]
    rows, probs, arts = col.run_block("de_bf_f025", campaign)
    assert probs == [] and len(rows) == 2
    cells = ev.run_cells("de_bf_f025")
    farm = os.path.dirname(cells[0]["ckpt_link"])
    for c in cells:
        rec = json.load(open(c["receipt"]))
        for path in (c["metrics"], c["stream"], c["receipt"], rec["log"], ev.dump_manifest_copy(c),
                     os.path.join(c["dump_dir"], "waveforms.npy"), os.path.join(c["dump_dir"], "COMPLETE")):
            assert path in arts, path
    for f in ("CHECKPOINTS.sha256", ev.reg.TRAINING_RECEIPT, ev.reg.TRAINING_RECEIPT + ".sha256", "receipts/train.log",
              "evidence/run_evidence.json.rank1.json"):
        assert os.path.join(farm, f) in arts, f
    assert all(len(v) == 64 for v in arts.values())
    assert os.path.realpath(cells[0]["ckpt_link"]) in arts and os.path.join(farm, ev.reg.COMMIT_MARKER) in arts
    monkeypatch.setattr(col.reg, "campaign_runs", lambda: ["de_bf_f025"])
    snapdir = world["tmp"] / "evalsnap_manifests"
    fx.write_snapshot_manifests(ev.reg, str(snapdir))
    monkeypatch.setattr(ev, "eval_snapshot_dir", lambda closure: str(snapdir))
    real_digest = ev.reg.snapshot_digest
    monkeypatch.setattr(ev.reg, "snapshot_digest", lambda d: campaign["eval_snapshot_sha256"] if d == str(snapdir)
                        else real_digest(d))
    env_rel = str(world["tmp"] / "env_receipt.json")
    fx.sealed(ev.reg, env_rel, ev.reg.ENV_RECEIPT_KEYS, platform="a6000")
    monkeypatch.setattr(ev.reg, "env_receipt_rel", lambda platform: env_rel)
    rows, pending, manifest = col.collect()
    assert pending == [] and manifest["runs"]["de_bf_f025"] == arts
    art = manifest["artifacts"]
    assert campaign["e0_receipt"] in art and campaign["e0_receipt"] + ".sha256" in art
    assert all(ev._abs(c["metrics"]) in art for c in ev.e0_cells())
    assert len([k for k in art if "/ref_cells/" in k]) == 69                 # 68 cells + the manifest
    for c in ev.e0_cells():                                                   # E0 logs, weights, snapshot manifests
        assert json.load(open(c["receipt"]))["log"] in art
    assert os.path.realpath(ev._abs(ev.reg.AGREE_REL)) in art and os.path.realpath(ev._abs(ev.reg.VAE_CKPT_REL)) in art
    assert str(snapdir / "SNAPSHOT.sha256") in art and str(snapdir / "SNAPSHOT.links") in art


@pytest.mark.parametrize("what", ["waveforms", "checkpoint", "e0 log", "cell log"])
def test_the_manifest_refuses_an_artifact_changed_after_admission(world, monkeypatch, what):
    """Codex code r3 HIGH 6: manifest hashes are CURRENT bytes, compared with what admission accepted."""
    cell = _cell(world, seed=42)
    if what == "waveforms":
        import numpy as np
        np.save(os.path.join(cell["dump_dir"], "waveforms.npy"), np.zeros(ev.DUMP_SHAPE, dtype=np.float32))
        fn = lambda: ev.cell_artifacts(cell)
    elif what == "checkpoint":
        open(os.path.realpath(cell["ckpt_link"]), "ab").write(b"x")
        fn = lambda: ev.run_artifacts("de_bf_f025")
    elif what == "cell log":
        open(json.load(open(cell["receipt"]))["log"], "a").write("x\n")
        fn = lambda: ev.cell_artifacts(cell)
    else:
        c = ev.e0_cells()[0]
        open(json.load(open(c["receipt"]))["log"], "a").write("x\n")
        env_rel = str(world["tmp"] / "env_receipt.json")
        fx.sealed(ev.reg, env_rel, ev.reg.ENV_RECEIPT_KEYS, platform="a6000")
        monkeypatch.setattr(ev.reg, "env_receipt_rel", lambda platform: env_rel)
        fn = lambda: ev.campaign_artifacts(world["campaign"])
    with pytest.raises(ev.EvalError, match="changed since its admission"):
        fn()


def test_the_evaluator_env_keeps_one_git_dir(monkeypatch):
    for k, v in {"GIT_WORK_TREE": "/x", "GIT_OBJECT_DIRECTORY": "/y", "GIT_COMMON_DIR": "/z", "GIT_DIR": "/w",
                 "RANK": "1", "PL_GLOBAL_SEED": "1", "SLURM_PROCID": "0"}.items():
        monkeypatch.setenv(k, v)
    e = ev.eval_env(1)
    assert [k for k in e if k.startswith("GIT_")] == ["GIT_DIR"] and e["GIT_DIR"].endswith("/worktrees/exp-38-data-efficiency") \
        or e["GIT_DIR"].endswith("/.git")
    assert not {"RANK", "PL_GLOBAL_SEED", "SLURM_PROCID"} & set(e) and e["CUDA_VISIBLE_DEVICES"] == "1"


def _owned_harness(tmp_path, cmd):
    import subprocess
    import textwrap
    code = textwrap.dedent(f"""
        import fcntl, os, sys
        sys.path.insert(0, {EXP!r})
        import exp38_eval as ev
        fd = os.open({str(tmp_path / 'card.lock')!r}, os.O_WRONLY | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("LOCKED", flush=True)
        sys.exit(ev.run_owned({cmd!r}, {str(tmp_path)!r}, dict(os.environ), {str(tmp_path / 'e.log')!r}, (fd,)))
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "LOCKED"
    return p


def test_a_killed_driver_cannot_release_the_card_under_its_evaluator(tmp_path):
    """Codex code r3 HIGH 2: the evaluator is an owned child; its group keeps the card lock."""
    import fcntl
    import signal
    import time
    gc = tmp_path / "gc.pid"
    p = _owned_harness(tmp_path, [sys.executable, "-c",
                                  "import subprocess, sys, time; q = subprocess.Popen([sys.executable, '-c', "
                                  "'import time; time.sleep(5)']); "
                                  f"open({str(gc)!r}, 'w').write(str(q.pid)); time.sleep(5)"])
    end = time.time() + 20
    while not (gc.exists() and gc.read_text()):
        assert time.time() < end
        time.sleep(0.1)
    p.send_signal(signal.SIGKILL)
    p.wait(timeout=30)
    fd = os.open(tmp_path / "card.lock", os.O_WRONLY)
    try:
        with pytest.raises(OSError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        end = time.time() + 30
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                assert time.time() < end, "the evaluator's lock was never released"
                time.sleep(0.5)
    finally:
        os.close(fd)


def test_run_owned_passes_the_status_and_stops_the_group_on_an_exception(tmp_path, monkeypatch):
    import exp38_supervise as sv
    assert ev.run_owned([sys.executable, "-c", "import sys; sys.exit(7)"], str(tmp_path), dict(os.environ),
                        str(tmp_path / "a.log")) == 7
    made = []
    real = sv.OwnedChild

    class T(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    monkeypatch.setattr(sv, "OwnedChild", T)
    monkeypatch.setattr(sv, "start_lock_holder", lambda pgid, fds: (_ for _ in ()).throw(OSError("no holder")))
    with pytest.raises(OSError):
        ev.run_owned([sys.executable, "-c", "import time; time.sleep(60)"], str(tmp_path), dict(os.environ),
                     str(tmp_path / "b.log"), grace=2)
    assert made and made[0].rc is not None


# ------------------------------------------------------------------ Yixun's E0a decision (2026-11-08)
def _decision_copy(tmp_path, monkeypatch, fn=None):
    src = ev._abs(ev.E0A_DECISION)
    rec = json.load(open(src))
    if fn:
        fn(rec)
    dst = str(tmp_path / "data_efficiency_e0a_decision.json")
    for p in (dst, dst + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    ev.reg.write_sealed(dst, rec)
    monkeypatch.setattr(ev, "E0A_DECISION", dst)
    return dst


def test_the_committed_e0a_decision_is_narrow_and_pinned():
    acc, sha = ev.load_e0a_decision(required=True)
    assert set(acc) == set(ev.E0A_CELLS)
    assert all(set(keys) == {"RIR_to_GT_RIR_R@1", "RIR_to_geom_R@1"} for keys in acc.values())   # R@1 only, as decided
    refs = ev.check_refs()
    for c in ev.e0_cells():
        if c["name"] in acc:
            ref, _ = ev.e0_reference(c, refs)
            assert all(v["reference"] == ref["metrics"][k] for k, v in acc[c["name"]].items())
    d = json.load(open(ev._abs(ev.E0A_DECISION)))
    import re
    assert d["decided_by"] == "Yixun Hu" and d["gate"] == "E0a"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", json.dumps(d))   # no account email


@pytest.mark.parametrize("name,fn", [
    ("acoustic key", lambda r: r["accepted"]["de_E0a_K8_s42"].update(T60={"ours": 1.0, "reference": 10.252})),
    ("other gate cell", lambda r: r["accepted"].update(de_E0b_K8_s42={"RIR_to_GT_RIR_R@1": {"ours": 1.0, "reference": 2.0}})),
    ("wrong reference", lambda r: r["accepted"]["de_E0a_K1_s42"]["RIR_to_geom_R@1"].update(reference=3.5)),
    ("no deviation", lambda r: r["accepted"]["de_E0a_K1_s42"]["RIR_to_geom_R@1"].update(ours=3.5032)),
    ("bool value", lambda r: r["accepted"]["de_E0a_K8_s42"]["RIR_to_geom_R@1"].update(ours=True)),
    ("gate", lambda r: r.update(gate="E0b")), ("extra key", lambda r: r.update(extra=1)),
    ("empty", lambda r: r.update(accepted={})),
])
def test_an_out_of_scope_decision_is_refused(tmp_path, monkeypatch, name, fn):
    _decision_copy(tmp_path, monkeypatch, fn)
    with pytest.raises(ev.reg.GateError):
        ev.load_e0a_decision(required=True)


def _e0a_with(ours):
    c = [c for c in ev.e0_cells() if c["name"] == "de_E0a_K8_s42"][0]
    ref, _ = ev.e0_reference(c, ev.check_refs())
    return c, {"metrics": dict(ref["metrics"], **ours)}, ref


def test_the_judge_accepts_exactly_the_decided_deviations():
    decision, _ = ev.load_e0a_decision(required=True)
    pinned = {k: v["ours"] for k, v in decision["de_E0a_K8_s42"].items()}
    c, got, ref = _e0a_with(pinned)
    ok, equal, diff, acc = ev.judge_e0(c, got, ref, decision)
    assert ok and equal is False and set(acc) == set(pinned) and "RIR_to_GT_RIR_R@1" in diff
    for extra in ({"RIR_to_GT_RIR_R@5": 99.0}, {"T60": 1.0}, {"FD": 0.1}, {"RIR_to_geom_R@1": pinned["RIR_to_geom_R@1"] + 1e-4}):
        c, got, ref = _e0a_with(dict(pinned, **extra))
        assert not ev.judge_e0(c, got, ref, decision)[0], extra
    c, got, ref = _e0a_with(pinned)
    assert not ev.judge_e0(c, got, ref, None)[0]                         # without the decision: strict
    c, got, ref = _e0a_with({"RIR_to_GT_RIR_R@1": pinned["RIR_to_GT_RIR_R@1"]})
    ok, equal, diff, acc = ev.judge_e0(c, got, ref, decision)              # a subset of the decided keys: fine
    assert ok and list(acc) == ["RIR_to_GT_RIR_R@1"]
    e0b = [c for c in ev.e0_cells() if c["name"] == "de_E0b_K8_s42"][0]
    ref_b, _ = ev.e0_reference(e0b, ev.check_refs())
    bad = {"metrics": dict(ref_b["metrics"], **{"RIR_to_GT_RIR_R@1": 1.0}), "by_scene": ref_b["by_scene"]}
    assert not ev.judge_e0(e0b, bad, ref_b, decision)[0]                  # E0b-d stay strict


@pytest.fixture
def decided(tmp_path, monkeypatch):
    """A campaign whose E0a cells carry exactly the decided R@1 deviations."""
    monkeypatch.setattr(ev, "E0_RECEIPT", str(tmp_path / "e0_receipt.json"))
    monkeypatch.setattr(ev, "FARM", str(tmp_path / "farm"))
    monkeypatch.setattr(ev.reg, "closure_digest", lambda repo=None: {"closure_sha256": fx.CAMPAIGN["closure_sha256"]})
    monkeypatch.setattr(ev.reg, "env_receipt_sha", lambda platform, repo=None: fx.CAMPAIGN["env_receipt_sha256"])
    monkeypatch.setattr(ev, "eval_snapshot", lambda closure, create=False: (str(tmp_path / "snap"),
                                                                           fx.CAMPAIGN["eval_snapshot_sha256"]))
    monkeypatch.setattr(ev, "live_eval_pins", lambda: {"agree_sha256": ev.reg.AGREE_SHA256,
                                                       "vae_ckpt_sha256": ev.reg.VAE_CKPT_SHA256, "vit_rev": ev.reg.VIT_REV})
    decision = ev.load_e0a_decision(required=True)
    ours = {n: {k: v["ours"] for k, v in keys.items()} for n, keys in decision[0].items()}
    campaign, rec = fx.make_e0_receipt(ev, log_dir=str(tmp_path / "logs"), e0a_ours=ours, decision=decision)
    return campaign, rec, decision


def test_a_decided_e0_receipt_validates_online_offline_and_on_neuronic(decided):
    campaign, rec, decision = decided
    assert rec["e0a_decision_sha256"] == decision[1] and rec["verdict"] == "PASS"
    assert rec["cells"]["de_E0a_K8_s42"]["equal"] is False and rec["cells"]["de_E0b_K8_s42"]["accepted_deviations"] == {}
    for kw in ({}, {"offline": True}, {"platform": "neuronic"}):
        assert ev.validate_e0_receipt(**kw)["sha256"] == campaign["sha256"]
    assert ev.e0_static_problems(rec) == []


@pytest.mark.parametrize("fn", [
    lambda r: r["accepted"]["de_E0a_K8_s42"].pop("RIR_to_geom_R@1"),        # a narrower decision
    lambda r: r.update(answer=r["answer"] + " (edited)"),                    # any other edit: another sha
])
def test_the_e0_receipt_binds_its_decision_file(decided, tmp_path, monkeypatch, fn):
    _decision_copy(tmp_path, monkeypatch, fn)
    with pytest.raises(ev.reg.GateError):
        ev.validate_e0_receipt()


def test_a_decided_receipt_needs_the_decision_and_e0b_stays_strict(decided, monkeypatch):
    campaign, rec, decision = decided
    bad = json.loads(json.dumps(rec))
    bad["e0a_decision_sha256"] = None
    assert ev.e0_static_problems(bad)
    bad = json.loads(json.dumps(rec))
    bad["cells"]["de_E0b_K8_s42"].update(equal=False, difference="x", accepted_deviations={"RIR_to_GT_RIR_R@1": {}})
    assert ev.e0_static_problems(bad)


CAMPAIGN_E0A = "/home/yixunhu/codespace/exp-38-campaign/outputs_FLAC/exp38_data_efficiency/eval/E0a"


@pytest.mark.skipif(not os.path.isdir(CAMPAIGN_E0A), reason="the 2026-11-08 E0a replay lives in the campaign tree")
def test_the_real_e0a_replay_also_deviates_at_r5_and_r10():
    """Read-only evidence: the bfc4ea4b E0a replay differs in R@5 / R@10 too, which the R@1-only decision does
    not accept -- so a re-sealed E0 under it FAILS until the decision is widened (reported to the coordinator)."""
    decision, _ = ev.load_e0a_decision(required=True)
    refs = ev.check_refs()
    for c in ev.e0_cells():
        if c["e0"] != "E0a":
            continue
        got = json.load(open(os.path.join(CAMPAIGN_E0A, os.path.basename(c["metrics"]))))
        ref, _ = ev.e0_reference(c, refs)
        diff = ev.metric_diff(got["metrics"], ref["metrics"])
        assert set(diff) <= set(ev.RETRIEVAL_KEYS)                         # the acoustics and FD are exact
        assert {k for k in diff if k.endswith("_R@1")} == set(decision[c["name"]])
        assert {k: diff[k][0] for k in decision[c["name"]]} == {k: v["ours"] for k, v in decision[c["name"]].items()}
        assert not ev.judge_e0(c, got, ref, decision)[0]                   # R@5 / R@10 deviations remain


def test_flac_rows_carry_the_exp14_retrieval_label(decided, tmp_path):
    campaign, rec, decision = decided
    rows, used = col.reference_rows(campaign)
    for (m, f, K), v in rows.items():
        assert v["retrieval_evaluator"] == ("exp_14" if m == "p1" and f in ("025", "050", "075") else None)
    md = col.render(rows, [])
    flac25 = [ln for ln in md.split("\n") if ln.startswith("| FLAC (P1) | 25 %")]
    assert flac25 and all(ln.count("†") == 1 for ln in flac25) and "exp_14's own evaluator" in md
    assert all("†" not in ln for ln in md.split("\n") if ln.startswith("| FLAC (P1) | 100 %"))
    (tmp_path / col.reg.EXP_DIR).mkdir(parents=True)
    col.publish(rows, [], {}, repo=str(tmp_path))
    payload = json.load(open(tmp_path / col.OUT_JSON))
    labels = {(r["model"], r["fraction"]): r["retrieval_evaluator"] for r in payload["rows"]}
    assert labels[("p1", "050")] == "exp_14" and labels[("p1", "100")] == "exp_38" and labels[("yaw", "100")] == "exp_38"


def test_d2_neuronic_campaign_runs_and_admission(world, monkeypatch):
    assert ev.reg.campaign_runs() == ev.reg.required_runs() + ["de_p1_f025"]
    rd = world["tmp"] / "nas" / "de_yaw_f050"
    fx.make_training_receipt(ev, "de_yaw_f050", str(rd), fx.CAMPAIGN["closure_sha256"], e0_sha=world["campaign"]["sha256"],
                             e0_path=world["campaign"]["e0_receipt"], platform="a6000")
    with pytest.raises(ev.EvalError, match="D2"):
        ev.prepare("de_yaw_f050")                                        # trained on the wrong platform
