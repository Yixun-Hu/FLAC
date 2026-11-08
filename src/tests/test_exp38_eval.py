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
    cells = ev.all_cells()
    assert len(cells) == 68 and len(ev.all_cells(bridge=True)) == 78
    assert len({c["name"] for c in cells}) == 68 and len({c["metrics"] for c in cells}) == 68
    assert sum(c["e0"] is not None for c in cells) == 8
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
    monkeypatch.setattr(ev.reg, "closure_digest", lambda repo=None: {"closure_sha256": fx.CAMPAIGN["closure_sha256"]})
    monkeypatch.setattr(ev.reg, "env_receipt_sha", lambda platform, repo=None: fx.CAMPAIGN["env_receipt_sha256"])
    campaign, rec = fx.make_e0_receipt(ev)
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
}


@pytest.mark.parametrize("name", sorted(E0_MUTATIONS))
def test_e0_receipt_refuses_each_mutation(e0, name):
    _, rec = e0
    rec = json.loads(json.dumps(rec))
    E0_MUTATIONS[name](rec)
    _reseal(ev.E0_RECEIPT, rec)
    offline_ok = name in ("closure", "env", "evaluator file")     # bindings to THIS tree are online-only
    with pytest.raises(ev.reg.GateError):
        ev.validate_e0_receipt()
    if offline_ok:
        ev.validate_e0_receipt(offline=True)
    else:
        with pytest.raises(ev.reg.GateError):
            ev.validate_e0_receipt(offline=True)


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
    farm = tmp / "farm" / run
    farm.mkdir(parents=True, exist_ok=True)
    for k in ("ckpt_link", "metrics", "stream", "receipt"):
        cell[k] = str(farm / os.path.basename(cell[k]))
    cell["dump_dir"] = str(tmp / "dumps" / cell["stem"])
    rd = tmp / "nas" / run
    if not (rd / ev.reg.TRAINING_RECEIPT).exists():
        tr, digest = fx.make_training_receipt(ev, run, str(rd), world["campaign"]["closure_sha256"],
                                              e0_sha=world["campaign"]["sha256"])
        for f in ("CHECKPOINTS.sha256", ev.reg.TRAINING_RECEIPT, ev.reg.TRAINING_RECEIPT + ".sha256"):
            open(farm / f, "wb").write(open(rd / f, "rb").read())
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
        "time": lambda r: r.update(started_at=None)})
    return out


@pytest.mark.parametrize("name", sorted(_receipt_mutations()))
def test_validate_cell_refuses_each_receipt_mutation(world, name):
    cell = _cell(world)
    _mut(cell["receipt"], _receipt_mutations()[name])
    assert ev.validate_cell(cell) != []


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
    tmp = world["tmp"]
    cell = [c for c in ev.e0_cells() if c["e0"] == "E0b" and c["K"] == 8][0]
    for k in ("ckpt_link", "metrics", "stream", "receipt"):
        cell[k] = str(tmp / "farm" / "E0b" / os.path.basename(cell[k]))
    camp = dict(fx.CAMPAIGN, sha256=None)
    fx.make_cell(ev, cell, camp, (ev.E0["E0b"]["sha"], ev.E0_BYTES["E0b"], None), str(tmp / "logs"))
    monkeypatch.setattr(ev, "current_campaign", lambda: camp)
    assert ev.validate_cell(cell) == []
    _mut(cell["metrics"], lambda r: r.update(ckpt_bytes_loaded=ev.E0_BYTES["E0a"]))
    assert ev.validate_cell(cell) != []


def test_prepare_requires_the_training_receipt(world, monkeypatch):
    rd = world["tmp"] / "nas" / "de_yaw_f025"
    monkeypatch.setattr(ev, "FARM", str(world["tmp"] / "farm2"))
    with pytest.raises(ev.reg.GateError):
        ev.prepare("de_yaw_f025")                                # no TRAINING_COMPLETE
    fx.make_training_receipt(ev, "de_yaw_f025", str(rd), "0" * 64, e0_sha=world["campaign"]["sha256"])
    with pytest.raises(ev.reg.GateError):
        ev.prepare("de_yaw_f025")                                # trained at another closure
    rd2 = world["tmp"] / "nas" / "de_yaw_f050"
    fx.make_training_receipt(ev, "de_yaw_f050", str(rd2), fx.CAMPAIGN["closure_sha256"], e0_sha="0" * 64)
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
