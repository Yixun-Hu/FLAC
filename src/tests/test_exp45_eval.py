"""exp_45 (forked from test_exp38_eval.py, B8/B9) — the evaluation driver (exp45_eval.py) and the collector
(exp45_collect.py).

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

import exp45_fixtures as fx

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
EXP38 = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")   # read-only reuse


def _load(name):
    import sys
    sys.path.insert(0, EXP)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_load("exp45_registry")
import exp45_collect as col  # noqa: E402  (one module graph: col.ev is ev, col.reg is ev.reg)
import exp45_eval as ev  # noqa: E402


RF025 = "exp45_RFf025_l40_2x32"


def _opt(argv, flag):
    return argv[argv.index(flag) + 1]


def test_plan_has_8_e0_30_result_and_10_bridge_cells_with_unique_outputs():
    cells = ev.all_cells()                      # E0 + the three RF runs + the RF-100 bridge (no P1 bridge run)
    assert ev.reg.TRAINING_PLATFORM == "neuronic" and len(cells) == 48
    assert len({c["name"] for c in cells}) == 48 and len({c["metrics"] for c in cells}) == 48
    assert sum(c["e0"] is not None for c in cells) == 8 and sum(bool(c["bridge"]) for c in cells) == 10
    assert {c["run"] for c in cells if not c["e0"]} == set(ev.reg.campaign_runs()) | {ev.BRIDGE_RUN}
    assert all(c["dump"] == (c["e0"] is None) for c in cells)


@pytest.mark.parametrize("run,cond", [("exp45_RFf050_l40_2x32", "vanilla"), ("exp45_RFf075_l40_2x32", "vanilla")])
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
    assert ev.reg.sha256_file(os.path.join(EXP38, "ref_cells/E0/E0a_K8_s42.json")) == \
        "4136ac75d570f84f76d34c5cac0f1e05e681897f4aefe53ffce12dd381f13180"
    assert ev.reg.sha256_file(os.path.join(EXP38, "ref_cells/E0/E0b_K8_s42.json")) == \
        "f167470e3501bcc7c4f116b0c26164523a8bd140b55c5e72e935d88d3f373adb"


def test_the_fork_writes_its_own_farm_and_dumps_and_shares_exp38s_card_locks():
    """plan r3 §3.2 (Codex plan r2 finding 1): exp_45's farm / dump roots; exp_38's card-lock names, so trees 38 and
    45 serialize on the same physical cards."""
    assert ev.FARM == "outputs_FLAC/exp45_rf_data_curve/eval"
    assert ev.DUMP_ROOT == "/media/diskstation/yixunhu/FLAC/exp45_pred_waveforms"
    assert ev.LOCKDIR == "/home/yixunhu/codespace" and ev.REF_DIR == f"{ev.reg.EXP38_DIR}/ref_cells"
    import inspect
    src = inspect.getsource(ev)
    assert src.count('f".exp38_card{gpu}.lock"') == 1 and all(c["metrics"].startswith(ev.FARM + "/") for c in ev.all_cells())


def test_no_reference_reader_opens_a_path_under_exp45s_folder(e0, monkeypatch):
    """Codex plan r2 finding 1: every reader / enumerator of the reference cells (check_refs, load_ref,
    e0_reference, the E0a decision, the E0 re-admission) reads exp_38's ref_cells/ -- never EXPREL/ref_cells."""
    import builtins
    opened, real_open, real_walk = [], builtins.open, os.walk
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: opened.append(os.path.realpath(str(f))) or real_open(f, *a, **k))
    monkeypatch.setattr(os, "walk", lambda top, *a, **k: opened.append(os.path.realpath(str(top))) or real_walk(top, *a, **k))
    refs = ev.check_refs()
    for c in ev.e0_cells():
        ev.e0_reference(c, refs)
    ev.load_e0a_decision(required=True)
    ev.validate_e0_receipt(offline=True)
    mine = os.path.realpath(os.path.join(REPO, ev.reg.EXP_DIR, "ref_cells"))
    theirs = os.path.realpath(os.path.join(REPO, ev.reg.REF_DIR))
    assert opened and not [p for p in opened if p == mine or p.startswith(mine + os.sep)]
    assert len({p for p in opened if p.startswith(theirs + os.sep)}) >= 69          # the 68 cells + the manifest


def test_reference_set_is_exact(tmp_path, monkeypatch):
    """Codex code r1 BLOCKING 7: exactly the 68 registered files -- no extra, missing or duplicate line."""
    import shutil
    root = tmp_path / "ref_cells"
    shutil.copytree(os.path.join(EXP38, "ref_cells"), root)
    monkeypatch.setattr(ev, "REF_DIR", str(root))              # every reader goes through REF_DIR
    monkeypatch.setattr(ev, "REF_MANIFEST", str(root / "REF_MANIFEST.sha256"))
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


def _cell(world, run="exp45_RFf025_l40_2x32", K=8, seed=43, dump=True):
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
    vcell = _cell(world, run="exp45_RFf050_l40_2x32", K=1, seed=46)
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
    "cond": lambda r: r.update(cond_method="fa_invariant"), "n_samples": lambda r: r.update(n_samples=6336),
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
        "argv": lambda r: r.update(argv=r["argv"][:-2]), "cell": lambda r: r.update(cell="exp45_RFf025_l40_2x32_K8_s44"),
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
    other = [c for c in ev.run_cells("exp45_RFf025_l40_2x32") if c["seed"] == 44 and c["K"] == 1][0]
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
    run = "exp45_RFf025_l40_2x32"
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


# exp_38's preliminary L40 stream sidecars on neuronic: the source of the ASSIGNMENT_HASH pins (a data property)
PRELIM = "/n/fs/gatrdp/codespace/_sessions/exp40_c16_runner/prelim_eval/de_yaw_f025"


def test_the_assignment_hash_pins_are_one_data_property():
    """plan r3 §3.2 (Codex plan r1 finding 13): ASSIGNMENT_HASH[(K, seed)] for K in {8, 1} x seeds 42-46; at
    rotate-deg 0 every item's offset is 0, so all ten equal exp_43's A6000 RF-100 K8 s42 value (brief §5.1)."""
    assert sorted(ev.ASSIGNMENT_HASH) == sorted(ev.INPUT_HASH) and len(ev.ASSIGNMENT_HASH) == 10
    assert set(ev.ASSIGNMENT_HASH.values()) == {"4894d0e119f71ed6f5d566aa01b69d8fc3f8a6cb88fb66f27f121ba59f0a8442"}


@pytest.mark.skipif(not os.path.isdir(PRELIM), reason="the exp_38 preliminary sidecars live on neuronic only")
def test_the_pins_equal_the_preliminary_neuronic_sidecars():
    import glob
    for (K, s), want in ev.ASSIGNMENT_HASH.items():
        files = glob.glob(os.path.join(PRELIM, f"*_metrics_1_1.0_de_yaw_f025_K{K}_s{s}.stream.json"))
        assert len(files) == 1, (K, s)
        d = json.load(open(files[0]))
        assert (d["assignment_hash"], d["input_hash"], d["stream_count"]) == (want, ev.INPUT_HASH[(K, s)], 6337)


@pytest.mark.parametrize("fn", [lambda r: r.update(assignment_hash="0" * 64), lambda r: r.pop("assignment_hash"),
                                lambda r: r.update(assignment_hash=None),
                                lambda r: r.update(assignment_hash=ev.ASSIGNMENT_HASH[(8, 43)].upper())])
def test_validate_cell_refuses_a_missing_or_mutated_assignment_hash(world, fn):
    cell = _cell(world)
    assert ev.validate_cell(cell) == [] and ev.validate_cell(cell, offline=True) == []
    _mut(cell["stream"], fn)
    assert any("assignment_hash" in p for p in ev.validate_cell(cell))
    assert any("assignment_hash" in p for p in ev.validate_cell(cell, offline=True))


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
    rd = world["tmp"] / "nas" / "exp45_RFf025_l40_2x32"
    rec, _ = ev.reg.read_sealed(str(rd / ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
    rec["final_ckpt_bytes"] += 1
    _reseal(str(rd / ev.reg.TRAINING_RECEIPT), rec)
    assert ev.validate_cell(cell) != []
    with pytest.raises(ev.reg.GateError):
        ev.reg.check_training_receipt("exp45_RFf025_l40_2x32", str(rd), closure=fx.CAMPAIGN["closure_sha256"])


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
    rd = world["tmp"] / "nas" / "exp45_RFf025_l40_2x32"
    with pytest.raises(ev.reg.GateError):
        ev.prepare("exp45_RFf025_l40_2x32")                                # no TRAINING_COMPLETE
    fx.make_training_receipt(ev, "exp45_RFf025_l40_2x32", str(rd), "0" * 64)
    with pytest.raises(ev.reg.GateError):
        ev.prepare("exp45_RFf025_l40_2x32")                                # trained at another closure
    rd2 = world["tmp"] / "nas" / "exp45_RFf050_l40_2x32"
    fx.make_training_receipt(ev, "exp45_RFf050_l40_2x32", str(rd2), fx.CAMPAIGN["closure_sha256"])
    with pytest.raises(ev.reg.GateError, match="another E0"):
        ev.prepare("exp45_RFf050_l40_2x32")                                # trained under another E0 receipt


# ------------------------------------------------------------------ collector
def test_aggregate_and_seed_set():
    recs = {s: {k: float(s) for k in ev.METRIC_KEYS} for s in ev.SEEDS}
    st = col.aggregate(recs)
    assert st["T60"][0] == 44.0 and abs(st["T60"][1] - 1.5811388300841898) < 1e-12 and st["T60"][2] == 5
    with pytest.raises(ValueError):
        col.aggregate({s: recs[s] for s in (42, 43, 44, 45)})


def _rows():
    """Synthetic rf_rows() output (the publication order test needs no admitted cells)."""
    st = {k: (1.0, 0.1, 5) for k in ev.METRIC_KEYS}
    return {(m, f, K): {"stats": st, "source": "x"} for m, f in [("rf", t) for t in ("025", "050", "075")] + [("rf_bridge", "100")]
            for K in (8, 1)}


def test_render_shows_the_three_fractions_and_the_bridge():
    md = col.render(_rows(), [])
    assert md.count("| Roll + Flip (RF) | ") == 6 and md.count("(bridge) | 100 % |") == 2 and "PENDING" not in md


def test_publish_order_markdown_first_json_last(tmp_path, e0, monkeypatch):
    exp = tmp_path / col.reg.EXP_DIR
    exp.mkdir(parents=True)
    rows = _rows()
    sha = col.publish(rows, [], {"m": 1}, repo=str(tmp_path))
    j = (tmp_path / col.OUT_JSON).read_bytes()
    assert hashlib.sha256(j).hexdigest() == sha and sha in (tmp_path / col.OUT_MD).read_text()
    assert json.loads(j)["manifest"] == {"m": 1} and len(json.loads(j)["rows"]) == 8
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


def test_the_collector_refuses_rather_than_publishing_partially(monkeypatch, tmp_path):
    published = []
    monkeypatch.setattr(col, "publish", lambda rows, pending, manifest, repo=None: published.append(rows) or "0" * 64)
    monkeypatch.setattr(col, "collect", lambda: (_ for _ in ()).throw(ev.EvalError("3 cell problem(s)")))
    assert col.main([]) == 2 and published == []
    monkeypatch.setattr(col, "collect", lambda: ({}, [], {}))
    assert col.main([]) == 0 and published == [{}]
    with pytest.raises(SystemExit):
        col.main(["--partial"])                                   # no partial publication exists


def _training(world, run):
    """A run's sealed training receipt in the (fake) NAS run dir + what `prepare` copies / links into the farm."""
    tmp = world["tmp"]
    farm = os.path.dirname(ev.run_cells(run)[0]["ckpt_link"])
    rd = tmp / "nas" / run
    fx.make_training_receipt(ev, run, str(rd), world["campaign"]["closure_sha256"], e0_sha=world["campaign"]["sha256"],
                             e0_path=world["campaign"]["e0_receipt"])
    fx.copy_to_farm(ev, str(rd), farm)
    final = ev.reg.ckpt_name(ev.reg.MAX_STEPS, ev.reg.parse_run_id(run)[1])
    os.symlink(str(rd / final), os.path.join(farm, final))
    return ev.reg.read_sealed(str(rd / ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)


def _cmd_run_world(world, monkeypatch, tmp_path):
    """cmd_run() -- the production entry point -- with a fake evaluator writing what eval_FLAC writes."""
    for run in ev.reg.campaign_runs():
        _training(world, run)
    snap = tmp_path / "evalsnap"
    snap.mkdir()
    pins = {"agree_sha256": ev.reg.AGREE_SHA256, "vae_ckpt_sha256": ev.reg.VAE_CKPT_SHA256, "vit_rev": ev.reg.VIT_REV}
    for name, fn in {"campaign_gates": lambda: None, "gpu_foreign": lambda gpu: [], "eval_env": lambda gpu: {},
                     "gpu_row": lambda gpu: (ev.reg.A6000_UUIDS[gpu], ev.GPU_NAME), "git_head": lambda repo=None: fx.CAMPAIGN["head"],
                     "eval_snapshot": lambda closure, create=False: (str(snap), fx.CAMPAIGN["eval_snapshot_sha256"]),
                     "EVAL_LOG_DIR": str(tmp_path / "eval_logs")}.items():
        monkeypatch.setattr(ev, name, fn)
    monkeypatch.setattr(ev.reg, "storage_gate", lambda *a, **k: {})
    monkeypatch.setattr(ev.reg, "eval_pins", lambda repo=None: dict(pins))
    monkeypatch.setattr(ev.reg, "verify_snapshot", lambda d, want=None: {"snapshot_sha256": fx.CAMPAIGN["eval_snapshot_sha256"]})
    cells = {c["name"]: c for c in ev.all_cells()}

    def fake_eval(cmd, cwd, env, log, lock_fds=(), grace=60):
        cell = cells[cmd[cmd.index("--eval-name") + 1]]
        ck = ev.cell_checkpoint(cell, world["campaign"])
        fx.make_cell(ev, cell, world["campaign"], ck, str(tmp_path / "unused_logs"))
        os.remove(cell["receipt"])                                # the driver writes the receipt itself
        open(log, "a").write(ev.SENTINEL + "\n")
        return 0
    monkeypatch.setattr(ev, "run_owned", fake_eval)
    names = [c["name"] for c in ev.all_cells() if not c["e0"]]
    ev.cmd_run(1, names, lockdir=str(tmp_path))
    return names


def _seal_bridge(world, monkeypatch, mutate=None):
    """The sealed rf_data_curve_bridge.json of the cmd_run world's ten bridge cells (exp_43's references = ours)."""
    monkeypatch.setattr(ev, "BRIDGE_RESULT", str(world["tmp"] / "rf_data_curve_bridge.json"))
    return ev.cmd_bridge(str(_bundle(world, mutate, make=False)))


def test_collect_admits_all_30_run_and_10_bridge_cells_produced_by_cmd_run(world, monkeypatch, tmp_path):
    """plan r3 §3.2 (Codex plan r2 finding 3): the 40 cells come from cmd_run(); collect() admits every one OFFLINE
    and lists {path: sha256} of every file that admission consumed -- the sealed bridge comparison included."""
    names = _cmd_run_world(world, monkeypatch, tmp_path)
    assert len(names) == 40
    _seal_bridge(world, monkeypatch)
    rows, pending, manifest = col.collect()
    assert pending == [] and sorted(rows) == sorted((m, f, K) for m, f in [("rf", t) for t in ("025", "050", "075")]
                                                    + [("rf_bridge", "100")] for K in (8, 1))
    assert all(rows[k]["stats"]["T60"][2] == 5 for k in rows)
    assert sorted(manifest["runs"]) == sorted(ev.reg.campaign_runs() + [ev.BRIDGE_RUN])
    for run in ev.reg.campaign_runs():
        arts = manifest["runs"][run]
        for c in ev.run_cells(run):
            for p in (c["metrics"], c["stream"], c["receipt"], ev.dump_manifest_copy(c)):
                assert os.path.relpath(ev._abs(p), ev.reg.REPO) in arts, p
        farm = os.path.relpath(os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"])), ev.reg.REPO)
        for f in ("CHECKPOINTS.sha256", ev.reg.TRAINING_RECEIPT, "receipts/loss_record.csv", "evidence/run_evidence.json.rank1.json"):
            assert os.path.join(farm, f) in arts, f
        assert not [k for k in arts if k.startswith("/media") or k.endswith(".ckpt") or k.endswith(".npy")]
    assert len(manifest["runs"][ev.BRIDGE_RUN]) == 42                     # 10 cells x 4 + the bridge result + sidecar
    assert manifest["bridge"]["verdict"] == "equal" and manifest["bridge"]["discrepancies"] == {}
    assert manifest["bridge"]["sha256"] == ev.reg.sha256_file(ev.BRIDGE_RESULT)
    assert len(manifest["artifacts"]) == 3 + 24
    (tmp_path / col.reg.EXP_DIR).mkdir(parents=True, exist_ok=True)
    sha = col.publish(rows, pending, manifest, repo=str(tmp_path))
    assert json.load(open(tmp_path / col.OUT_JSON))["manifest"] == json.loads(json.dumps(manifest)) and len(sha) == 64


def _reseal_bridge(fn):
    rec = json.load(open(ev.BRIDGE_RESULT))
    fn(rec)
    for p in (ev.BRIDGE_RESULT, ev.BRIDGE_RESULT + ".sha256"):
        os.remove(p)
    ev.reg.write_sealed(ev.BRIDGE_RESULT, rec)


BRIDGE_RESULT_MUTATIONS = {
    "another campaign": lambda r: r["campaign"].update(head="e" * 40),
    "another descriptor": lambda r: r.update(descriptor_sha256="0" * 64),
    "a changed descriptor": lambda r: r["descriptor"].update(bytes=1),
    "a stale cell binding": lambda r: r["cells"]["exp45_RF100bridge_K1_s45"].update(metrics_sha256="0" * 64),
    "a missing cell": lambda r: r["cells"].pop("exp45_RF100bridge_K8_s46"),
    "an equal verdict over a difference": lambda r: r["cells"]["exp45_RF100bridge_K8_s42"].update(
        differences={"metrics.T60": {"ours": 1.0, "reference": 1.1}}),
    "an equal overall verdict over a discrepancy": lambda r: r["cells"]["exp45_RF100bridge_K8_s42"].update(
        verdict="evaluator discrepancy", differences={"metrics.T60": {"ours": 1.0, "reference": 1.1}}),
    "another admitted checkpoint": lambda r: r["admission_record"].update(ckpt_sha256="0" * 64),
    "an unbundled admission record": lambda r: r["admission_record"].update(sha256="0" * 64),
    "an unbundled reference": lambda r: r["cells"]["exp45_RF100bridge_K1_s42"].update(reference_metrics_sha256="0" * 64),
}


@pytest.mark.parametrize("what", ["absent", "sidecar"] + sorted(BRIDGE_RESULT_MUTATIONS))
def test_collect_requires_the_sealed_bridge_comparison_of_this_campaign_and_these_cells(world, monkeypatch, tmp_path, what):
    """Codex code r1 finding 4: forty admissible cells are not enough -- the collector publishes only with the sealed
    rf_data_curve_bridge.json bound to THIS campaign, the registered descriptor, exactly the ten bridge cells' current
    metrics bytes, the bundle's reference / admission hashes and verdicts that follow from the differences."""
    _cmd_run_world(world, monkeypatch, tmp_path)
    _seal_bridge(world, monkeypatch)
    if what == "absent":
        for p in (ev.BRIDGE_RESULT, ev.BRIDGE_RESULT + ".sha256"):
            os.remove(p)
    elif what == "sidecar":
        open(ev.BRIDGE_RESULT, "ab").write(b" ")
    else:
        _reseal_bridge(BRIDGE_RESULT_MUTATIONS[what])
    with pytest.raises((ev.EvalError, ev.reg.GateError), match="bridge"):
        col.collect()


def _refresh_cell_links(c):
    """Re-bind a cell's receipt and dump-manifest copy to its CURRENT metrics / stream bytes (what an attacker who
    edits a cell after the bridge seal would do, so that the cell still passes its own offline admission)."""
    copy = ev.dump_manifest_copy(c)
    _mut(copy, lambda m: m.update(stream_sidecar_sha256=ev.reg.sha256_file(c["stream"]),
                                  metrics_sha256=ev.reg.sha256_file(c["metrics"])))
    _mut(c["receipt"], lambda r: r.update(stream_sha256=ev.reg.sha256_file(c["stream"]),
                                          metrics_sha256=ev.reg.sha256_file(c["metrics"]),
                                          dump_manifest_sha256=ev.reg.sha256_file(copy)))


def test_a_bridge_stream_changed_after_the_seal_refuses_collection(world, monkeypatch, tmp_path):
    """Codex code r2 finding 3: the sealed comparison binds the sha256 of every compared exp_45 stream and metrics file
    and of the cell's receipt; a stream stripped of its preimages after the seal -- its pins kept and its receipt /
    dump-manifest links refreshed so the cell alone still admits offline -- refuses collection."""
    _cmd_run_world(world, monkeypatch, tmp_path)
    rec = _seal_bridge(world, monkeypatch)
    c = [c for c in ev.bridge_cells() if c["name"] == "exp45_RF100bridge_K8_s44"][0]
    assert {"stream_sha256", "receipt_sha256", "metrics_sha256"} <= set(rec["cells"][c["name"]])
    s = json.load(open(c["stream"]))
    for k in ev.STREAM_ITEMS:
        s.pop(k)
    json.dump(s, open(c["stream"], "w"))
    _refresh_cell_links(c)
    assert ev.validate_cell(c, world["campaign"], offline=True) == []         # the cell alone still admits
    with pytest.raises((ev.EvalError, ev.reg.GateError), match="bridge"):
        col.collect()


def test_each_bridge_bundle_file_is_read_exactly_once(bridge_world, monkeypatch):
    """Codex code r2 finding 4: every bundle file (the manifest, the admission record, the 20 reference files) is
    read ONCE; the captured bytes are verified against the manifest and are the bytes parsed and compared."""
    import builtins
    b = _bundle(bridge_world)
    opened, real_open = {}, builtins.open

    def counting_open(f, *a, **k):
        p = os.path.realpath(str(f)) if isinstance(f, (str, bytes, os.PathLike)) else None
        if p and p.startswith(os.path.realpath(str(b)) + os.sep):
            opened[os.path.relpath(p, os.path.realpath(str(b)))] = opened.get(os.path.relpath(p, os.path.realpath(str(b))), 0) + 1
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", counting_open)
    rec = ev.cmd_bridge(str(b))
    monkeypatch.setattr(builtins, "open", real_open)
    assert rec["verdict"] == "equal" and len(opened) == 22
    assert {k: v for k, v in opened.items() if v != 1} == {}, opened


@pytest.mark.parametrize("which", ["reference metrics", "reference stream", "admission record"])
def test_a_bundle_file_changed_after_its_capture_cannot_steer_the_comparison(bridge_world, monkeypatch, which):
    """Codex code r2 finding 4: a reference changed on disk right after it was captured (and verified) is never the
    one compared -- the comparison uses the captured bytes, so the sealed hashes describe what was compared."""
    import builtins
    b = _bundle(bridge_world)
    stem = ev.bridge_ref_stem("exp43_RF_T_S40000_K8_s42")
    rel = {"reference metrics": f"cells/{stem}.json", "reference stream": f"cells/{stem}.stream.json",
           "admission record": "admission/exp43_admission_RF.json"}[which]
    target = os.path.realpath(str(b / rel))
    real_open, state = builtins.open, {"done": False}

    def swap_after_first_read(f, *a, **k):
        fh = real_open(f, *a, **k)
        if not state["done"] and isinstance(f, (str, os.PathLike)) and os.path.realpath(str(f)) == target \
                and ("r" in (a[0] if a else k.get("mode", "r"))):
            state["done"] = True
            data = fh.read()
            fh.close()
            if which == "reference metrics":
                d = json.loads(data)
                d["metrics"]["T60"] += 1.0                     # would show as a discrepancy if it were compared
                new = json.dumps(d).encode()
            elif which == "reference stream":
                d = json.loads(data)
                d.pop("input_tuples")                          # would refuse if it were compared
                new = json.dumps(d).encode()
            else:
                new = b"{}"
            with real_open(target, "wb") as w:
                w.write(new)
            import io
            return io.BytesIO(data) if "b" in (a[0] if a else k.get("mode", "r")) else io.StringIO(data.decode()
                                                                                                 if isinstance(data, bytes) else data)
        return fh
    monkeypatch.setattr(builtins, "open", swap_after_first_read)
    rec = ev.cmd_bridge(str(b))
    monkeypatch.setattr(builtins, "open", real_open)
    assert state["done"] and rec["verdict"] == "equal" and rec["cells"]["exp45_RF100bridge_K8_s42"]["differences"] == {}


def test_an_evaluator_discrepancy_is_published_as_a_discrepancy_never_as_equality(world, monkeypatch, tmp_path):
    """Codex code r1 finding 4: a recorded evaluator discrepancy reaches the results JSON and Markdown as one."""
    _cmd_run_world(world, monkeypatch, tmp_path)

    def bump(c, m, s):
        if c["name"] == "exp45_RF100bridge_K8_s43":
            m["metrics"]["C50"] += 0.25
    _seal_bridge(world, monkeypatch, bump)
    rows, pending, manifest = col.collect()
    br = manifest["bridge"]
    assert br["verdict"] == "evaluator discrepancy" and sorted(br["discrepancies"]) == ["exp45_RF100bridge_K8_s43"]
    assert sorted(br["discrepancies"]["exp45_RF100bridge_K8_s43"]) == ["metrics.C50"]
    assert all("evaluator discrepancy" in rows[("rf_bridge", "100", K)]["source"] for K in (8, 1))
    (tmp_path / col.reg.EXP_DIR).mkdir(parents=True, exist_ok=True)
    col.publish(rows, pending, manifest, repo=str(tmp_path))
    md = (tmp_path / col.OUT_MD).read_text()
    assert "evaluator discrepancy" in md and "exp45_RF100bridge_K8_s43" in md and "metrics.C50" in md
    assert "equal on all ten" not in md
    assert json.load(open(tmp_path / col.OUT_JSON))["manifest"]["bridge"]["verdict"] == "evaluator discrepancy"


@pytest.mark.parametrize("what", ["missing run cell", "missing bridge cell", "unadmitted bridge cell", "e0 cell"])
def test_collect_refuses_any_missing_or_unadmitted_cell(world, monkeypatch, tmp_path, what):
    _cmd_run_world(world, monkeypatch, tmp_path)
    _seal_bridge(world, monkeypatch)
    if what == "missing run cell":
        os.remove(ev.run_cells(ev.reg.campaign_runs()[1])[7]["stream"])
    elif what == "missing bridge cell":
        os.remove(ev.bridge_cells()[9]["metrics"])
    elif what == "unadmitted bridge cell":
        _mut(ev.bridge_cells()[2]["receipt"], lambda r: r.update(bridge_descriptor_sha256="0" * 64))
    else:
        _mut(ev.e0_cells()[1]["metrics"], lambda r: r["metrics"].update(T60=r["metrics"]["T60"] + 1e-9))
    with pytest.raises((ev.EvalError, ev.reg.GateError)):
        col.collect()


def _publication(tmp_path):
    """A published results bundle in a fresh root: the JSON + Markdown (exp45_collect.publish) whose manifest names a
    few evidence files, and the generator_static() files -- each at its relative path."""
    root = tmp_path / "published"
    (root / col.reg.EXP_DIR).mkdir(parents=True)
    ev_files = {f"{ev.FARM}/E0a/x_metrics.json": b"e0 cell", f"{ev.FARM}/exp45_RFf025_l40_2x32/y.stream.json": b"stream",
                f"{col.reg.EXP_DIR}/rf_data_curve_bridge.json": b"bridge"}
    for rel, data in list(ev_files.items()) + [(rel, b"static " + rel.encode()) for rel in col.generator_static()]:
        os.makedirs(root / os.path.dirname(rel), exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {"artifacts": {rel: hashlib.sha256(d).hexdigest() for rel, d in list(ev_files.items())[:1]},
                "runs": {"exp45_RFf025_l40_2x32": {rel: hashlib.sha256(d).hexdigest() for rel, d in list(ev_files.items())[1:]}}}
    col.publish(_rows(), [], manifest, repo=str(root))
    return root, ev_files


def _seed_exp38_files(root, dest):
    """exp_38's read-only files, byte-identical, in the destination (they arrive there only with exp_38's own
    publication; exp_45's materialise verifies and never writes them -- Codex code r2 finding 6)."""
    owned = [rel for rel in col.generator_static() if rel.startswith(col.reg.EXP38_DIR + "/")]
    for rel in owned:
        os.makedirs(dest / os.path.dirname(rel), exist_ok=True)
        (dest / rel).write_bytes((root / rel).read_bytes())
    return owned


def test_verify_and_materialise_carry_a_publication_to_another_root_sha256_verified(tmp_path):
    """Codex code r1 finding 12: ``verify`` re-hashes every file the results JSON names (and the Markdown names the
    JSON's sha256); ``materialise`` copies the publication + generator_static() byte for byte to another root,
    identical-or-absent, then verifies it there."""
    root, ev_files = _publication(tmp_path)
    got = col.verify(str(root))
    assert set(got) == set(ev_files) | {col.OUT_JSON, col.OUT_MD}
    dest = tmp_path / "generator_root"
    dest.mkdir()
    owned = _seed_exp38_files(root, dest)
    out = col.materialise(str(dest), root=str(root))
    assert out["copied"] == len(ev_files) + 2 + len(col.generator_static()) - len(owned)
    assert out["already_identical"] == len(owned) > 0
    assert col.verify(str(dest)) == got
    assert all((dest / rel).read_bytes() == (root / rel).read_bytes() for rel in col.generator_static())
    assert col.materialise(str(dest), root=str(root))["already_identical"] == out["files"]          # idempotent
    victim = dest / list(ev_files)[1]
    victim.write_bytes(b"other bytes")
    with pytest.raises(ev.EvalError, match="other bytes"):
        col.materialise(str(dest), root=str(root))                         # never overwritten
    with pytest.raises(ev.EvalError, match="recorded sha256"):
        col.verify(str(dest))
    (root / list(ev_files)[0]).write_bytes(b"changed")
    with pytest.raises(ev.EvalError, match="recorded sha256"):
        col.verify(str(root))
    md = root / col.OUT_MD
    md.write_text(md.read_text().replace("results JSON sha256", "results JSON"))
    with pytest.raises(ev.EvalError, match="does not name"):
        col.verify(str(root))
    with pytest.raises(ev.EvalError, match="no published"):
        col.verify(str(tmp_path / "nothing_here"))


@pytest.mark.parametrize("case", ["dangling symlink", "tracked but missing", "created during the race"])
def test_materialise_never_clobbers_a_destination(tmp_path, monkeypatch, case):
    """Codex code r2 finding 9: an existing destination entry (a dangling symlink included, by lexists) refuses; a
    path tracked in the destination repo's index but missing from its worktree refuses (git ls-files
    --error-unmatch, read-only); an absent file is published with a no-clobber primitive, so a destination created
    between the check and the write refuses and keeps its bytes."""
    import builtins
    import subprocess
    root, ev_files = _publication(tmp_path)
    dest = tmp_path / "generator_root"
    dest.mkdir()
    _seed_exp38_files(root, dest)
    rel = list(ev_files)[1]
    target = dest / rel
    os.makedirs(target.parent, exist_ok=True)
    if case == "dangling symlink":
        target.symlink_to(tmp_path / "nowhere")
    elif case == "tracked but missing":
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        target.write_bytes(b"tracked")
        subprocess.run(["git", "-C", str(dest), "add", "-f", rel], check=True)
        os.remove(target)                                            # tracked in the index, absent on disk
    else:
        real_open, src, seen = builtins.open, os.path.realpath(str(root / rel)), []

        def racing_open(f, *a, **k):
            if isinstance(f, (str, os.PathLike)) and os.path.realpath(str(f)) == src:
                seen.append(1)
                if len(seen) == 2 and not target.exists():           # its copy read: after verify's hashing and
                    target.write_bytes(b"a concurrent writer's bytes")   # the existence check, before the write
            return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", racing_open)
    with pytest.raises(ev.EvalError):
        col.materialise(str(dest), root=str(root))
    monkeypatch.undo()
    if case == "dangling symlink":
        assert os.path.islink(target) and not os.path.exists(target)
    elif case == "tracked but missing":
        assert not os.path.lexists(target)
    else:
        assert target.read_bytes() == b"a concurrent writer's bytes"


def test_the_collector_cli_verifies_and_materialises(tmp_path, monkeypatch, capsys):
    root, _ = _publication(tmp_path)
    monkeypatch.chdir(os.getcwd())                               # main() chdirs to REPO: restored afterwards
    monkeypatch.setattr(col.reg, "REPO", str(root))
    assert col.main(["verify"]) == 0 and "verified" in capsys.readouterr().out
    dest = tmp_path / "dest"
    dest.mkdir()
    _seed_exp38_files(root, dest)
    assert col.main(["materialise", "--dest", str(dest)]) == 0 and col.verify(str(dest))
    assert col.main(["verify", "--root", str(tmp_path / "absent")]) == 2


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
        fn = lambda: ev.run_artifacts("exp45_RFf025_l40_2x32")
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
    import subprocess                       # exp_45: this checkout's own git dir (worktree or main checkout)
    gitdir = subprocess.run(["git", "-C", REPO, "rev-parse", "--absolute-git-dir"], capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")}).stdout.strip()
    assert [k for k in e if k.startswith("GIT_")] == ["GIT_DIR"] and e["GIT_DIR"] == gitdir
    assert not {"RANK", "PL_GLOBAL_SEED", "SLURM_PROCID"} & set(e) and e["CUDA_VISIBLE_DEVICES"] == "1"


def _owned_harness(tmp_path, cmd):
    import subprocess
    import textwrap
    code = textwrap.dedent(f"""
        import fcntl, os, sys
        sys.path.insert(0, {EXP!r})
        import exp45_eval as ev
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
    import exp45_supervise as sv
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
    dst = str(tmp_path / "data_efficiency_e0a_decision.json")   # exp_38's file name
    for p in (dst, dst + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    ev.reg.write_sealed(dst, rec)
    monkeypatch.setattr(ev, "E0A_DECISION", dst)
    return dst


# the bfc4ea4b E0a replay (the campaign tree's cells, copied; sha256 == the bfc4ea4b E0 receipt's metrics_sha256)
REPLAY = os.path.join(REPO, "src", "tests", "fixtures", "exp38_e0a_replay_bfc4ea4b")
REPLAY_SHA = {"de_E0a_K8_s42": "9b1c537876031ac711d149fb8f2f96f8d777c9a01d3fc71e91ab6b4dcebdbbca",
              "de_E0a_K1_s42": "e3c148f6e4850e3ad2f6367ca7d3601ab14079d7e9b864fe45f838f3fbca546e"}
WIDENED = {"de_E0a_K8_s42": {"RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5",
                             "RIR_to_geom_R@10"},
           "de_E0a_K1_s42": {"RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_geom_R@1", "RIR_to_geom_R@5",
                             "RIR_to_geom_R@10"}}


def _replay(name):
    path = os.path.join(REPLAY, f"{name}.metrics.json")
    assert ev.reg.sha256_file(path) == REPLAY_SHA[name]
    c = [c for c in ev.e0_cells() if c["name"] == name][0]
    ref, _ = ev.e0_reference(c, ev.check_refs())
    return c, json.load(open(path)), ref


def test_the_committed_e0a_decision_is_narrow_and_pinned():
    """Yixun 2026-11-08: "Accept, label R@1", widened the same day to "Widen to R@1/5/10" -- exactly the ten
    measured (ours, reference) pairs of the bfc4ea4b replay, nothing else."""
    acc, sha = ev.load_e0a_decision(required=True)
    assert set(acc) == set(ev.E0A_CELLS)
    assert {n: set(keys) for n, keys in acc.items()} == WIDENED
    for n in ev.E0A_CELLS:
        c, got, ref = _replay(n)
        assert {k: (v["ours"], v["reference"]) for k, v in acc[n].items()} == ev.metric_diff(got["metrics"], ref["metrics"])
    refs = ev.check_refs()
    for c in ev.e0_cells():
        if c["name"] in acc:
            ref, _ = ev.e0_reference(c, refs)
            assert all(v["reference"] == ref["metrics"][k] for k, v in acc[c["name"]].items())
    d = json.load(open(ev._abs(ev.E0A_DECISION)))
    import re
    assert d["decided_by"] == "Yixun Hu" and d["gate"] == "E0a" and d["schema"] == 2
    assert '"Widen to R@1/5/10"' in d["answer"] and "Accept, label R@1" in d["answer"]
    assert "vae_sample" in d["reason"] and "randn_like" in d["reason"] and "exp_37" in d["reason"]
    assert d["supersedes"]["sha256"] == "332338a3dd2e05ffb3a370342b880e4dc74b313d4abfadbc5fdd5d2e1d219068"
    assert not re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", json.dumps(d))   # no account email


@pytest.mark.parametrize("name,fn", [
    ("acoustic key", lambda r: r["accepted"]["de_E0a_K8_s42"].update(T60={"ours": 1.0, "reference": 10.252})),
    ("other gate cell", lambda r: r["accepted"].update(de_E0b_K8_s42={"RIR_to_GT_RIR_R@1": {"ours": 1.0, "reference": 2.0}})),
    ("wrong reference", lambda r: r["accepted"]["de_E0a_K1_s42"]["RIR_to_geom_R@1"].update(reference=3.5)),
    ("no deviation", lambda r: r["accepted"]["de_E0a_K1_s42"]["RIR_to_geom_R@1"].update(ours=3.5032)),
    ("bool value", lambda r: r["accepted"]["de_E0a_K8_s42"]["RIR_to_geom_R@1"].update(ours=True)),
    ("gate", lambda r: r.update(gate="E0b")), ("extra key", lambda r: r.update(extra=1)),
    ("empty", lambda r: r.update(accepted={})), ("schema 1", lambda r: r.update(schema=1)),
    ("no reason", lambda r: r.update(reason=" ")), ("no supersedes", lambda r: r.pop("supersedes")),
    ("empty cell", lambda r: r["accepted"].update(de_E0a_K1_s42={})),
    ("FD key", lambda r: r["accepted"]["de_E0a_K1_s42"].update(FD={"ours": 1.0, "reference": 2.0})),
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
    """A campaign whose E0a cells carry exactly the decided retrieval deviations."""
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


def test_the_real_bfc4ea4b_e0a_replay_passes_under_the_widened_decision():
    """The real evidence: both bfc4ea4b E0a replays pass with exactly their five retrieval deviations accepted;
    a changed acoustic value, an off-pin decided retrieval value or a changed undecided key still fails."""
    decision, _ = ev.load_e0a_decision(required=True)
    for name in ev.E0A_CELLS:
        c, got, ref = _replay(name)
        diff = ev.metric_diff(got["metrics"], ref["metrics"])
        assert set(diff) == WIDENED[name] and set(diff) <= set(ev.RETRIEVAL_KEYS)     # acoustics + FD exact
        ok, equal, text, acc = ev.judge_e0(c, got, ref, decision)
        assert ok and equal is False and acc == decision[name]
        assert not ev.judge_e0(c, got, ref, None)[0]                                  # strict without it
        for k in ("T60", "Invalid T60", "C50", "EDT", "FD"):                          # any acoustic change fails
            bad = json.loads(json.dumps(got))
            bad["metrics"][k] = got["metrics"][k] + 1e-4
            assert not ev.judge_e0(c, bad, ref, decision)[0], (name, k)
            bad["metrics"][k] = float(got["metrics"][k]) if isinstance(got["metrics"][k], int) else str(got["metrics"][k])
            assert not ev.judge_e0(c, bad, ref, decision)[0], (name, k, "type")
        for k in WIDENED[name]:                                                       # off-pin retrieval fails
            for delta in (1e-4, -1e-4):
                bad = json.loads(json.dumps(got))
                bad["metrics"][k] = round(got["metrics"][k] + delta, 4)
                assert not ev.judge_e0(c, bad, ref, decision)[0], (name, k, delta)
            bad = json.loads(json.dumps(got))
            bad["metrics"][k] = ref["metrics"][k]                                     # partly reverted: a subset
            ok, _, _, acc = ev.judge_e0(c, bad, ref, decision)
            assert ok and set(acc) == WIDENED[name] - {k}
        for k in set(ev.RETRIEVAL_KEYS) - WIDENED[name]:                              # an undecided key moves: fail
            bad = json.loads(json.dumps(got))
            bad["metrics"][k] = round(got["metrics"][k] + 0.0158, 4)
            assert not ev.judge_e0(c, bad, ref, decision)[0], (name, k)
        other = [o for o in ev.E0A_CELLS if o != name][0]                             # pins are per cell
        assert not ev.judge_e0(c, got, ref, {other: decision[other]})[0]


def test_d2_neuronic_campaign_runs_and_admission(world, monkeypatch):
    assert ev.reg.campaign_runs() == ev.reg.required_runs()
    rd = world["tmp"] / "nas" / "exp45_RFf050_l40_2x32"
    fx.make_training_receipt(ev, "exp45_RFf050_l40_2x32", str(rd), fx.CAMPAIGN["closure_sha256"], e0_sha=world["campaign"]["sha256"],
                             e0_path=world["campaign"]["e0_receipt"], platform="a6000")
    with pytest.raises(ev.EvalError, match="D2"):
        ev.prepare("exp45_RFf050_l40_2x32")                                        # trained on the wrong platform


# ------------------------------------------------------------------ the RF-100 bridge (plan r3 §3.2, N5)
def test_the_bridge_descriptor_and_its_ten_cells():
    d = ev.RF100_BRIDGE
    assert d["checkpoint"] == "/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug/exp43_RF_a6000_2x32/epoch=8-step=40000.ckpt"
    assert (d["sha256"], d["bytes"], d["model_config_sha256"]) == (
        "ae62977e6a446c696f4b8f34e9393d335406bc8cd6404e7cc9caceaf55d1b6f6", 723922731,
        "d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d")
    assert ev.reg.sha256_file(os.path.join(REPO, ev.reg.ARMS["rf"]["config"])) == d["model_config_sha256"]
    cells = ev.bridge_cells()
    assert [c["name"] for c in cells] == list(d["cells"]) and len(cells) == 10
    assert set(d["cells"].values()) == {f"exp43_RF_T_S40000_K{K}_s{s}" for K in (8, 1) for s in ev.SEEDS}
    assert all(c["bridge"] and c["dump"] and c["e0"] is None and c["config"] == ev.reg.ARMS["rf"]["config"]
               and c["cond"] == "vanilla" for c in cells)
    names = [c["name"] for c in ev.all_cells()]
    assert len(names) == len(set(names)) == 48 and set(d["cells"]) <= set(names)
    run = [c for c in ev.run_cells(RF025) if c["K"] == 8 and c["seed"] == 42][0]
    a, b = ev.eval_argv(cells[0]), ev.eval_argv(run)                     # a run cell's argv but name + checkpoint
    assert len(a) == len(b) and [x for x, y in zip(a, b) if x != y] == [cells[0]["name"], cells[0]["ckpt_link"]]
    assert ev.BRIDGE_DESCRIPTOR_SHA256 == hashlib.sha256(ev.reg.canonical(d).encode()).hexdigest()


def test_cmd_run_accepts_bridge_names(monkeypatch):
    monkeypatch.setattr(ev, "campaign_gates", lambda: (_ for _ in ()).throw(ev.EvalError("gates reached")))
    with pytest.raises(ev.EvalError, match="gates reached"):
        ev.cmd_run(0, ["exp45_RF100bridge_K8_s42", RF025 + "_K8_s42"])     # bridge + run cells: one kind
    with pytest.raises(ev.EvalError, match="unknown cells"):
        ev.cmd_run(0, ["exp45_RF100bridge_K8_s47"])
    with pytest.raises(ev.EvalError, match="separate calls"):
        ev.cmd_run(0, ["exp45_RF100bridge_K8_s42", "de_E0a_K8_s42"])


def test_one_checkpoint_dispatcher_for_run_one_and_validate_cell(world, monkeypatch):
    b = ev.bridge_cells()[0]
    want = (ev.RF100_BRIDGE["sha256"], ev.RF100_BRIDGE["bytes"], None)
    assert ev.cell_checkpoint(b, world["campaign"]) == want
    seen = []
    monkeypatch.setattr(ev, "expected_ckpt", lambda cell, campaign, offline=False: seen.append((cell["name"], offline)) or ("s", 1, "t"))
    c = ev.run_cells(RF025)[0]
    assert ev.cell_checkpoint(c, world["campaign"], True) == ("s", 1, "t") and seen == [(c["name"], True)]
    assert ev.cell_checkpoint(b, world["campaign"], True) == want and len(seen) == 1
    import inspect
    for fn in (ev.run_one, ev.validate_cell):                           # both go through the dispatcher only
        src = inspect.getsource(fn)
        assert "cell_checkpoint(cell, campaign" in src and "expected_ckpt(" not in src


def _bridge_cell(world, K=8, seed=42, dump=True, stream_fn=None):
    cell = [c for c in ev.bridge_cells() if c["K"] == K and c["seed"] == seed][0]
    return fx.make_cell(ev, cell, world["campaign"], ev.bridge_checkpoint(), str(world["tmp"] / "logs"), dump=dump,
                        stream_fn=stream_fn)


def _exp43_kit():
    """exp_43's own kit (its admission validator), loaded independently of the module under test."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp43_kit_eval_test", os.path.join(
        REPO, "worklog", "worklog_yixun", "exp_43_std_aug_claude", "exp43_kit.py"))
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    return kit


KIT43 = _exp43_kit()


def test_validate_cell_admits_a_bridge_cell_online_and_offline(world):
    cell = _bridge_cell(world)
    assert ev.validate_cell(cell) == [] and ev.validate_cell(cell, offline=True) == []
    rec = json.load(open(cell["receipt"]))
    assert (rec["checkpoint_source"], rec["bridge_descriptor_sha256"], rec["training_receipt_sha256"]) == \
        ("exp43_external", ev.BRIDGE_DESCRIPTOR_SHA256, None)


@pytest.mark.parametrize("name,fn", [
    ("source", lambda r: r.update(checkpoint_source="exp45_training")),
    ("descriptor", lambda r: r.update(bridge_descriptor_sha256="0" * 64)),
    ("no source", lambda r: r.pop("checkpoint_source")), ("no descriptor", lambda r: r.pop("bridge_descriptor_sha256")),
    ("training receipt", lambda r: r.update(training_receipt_sha256="0" * 64)),
    ("ckpt sha", lambda r: r.update(ckpt_sha256="0" * 64)), ("ckpt bytes", lambda r: r.update(ckpt_bytes=1)),
])
def test_validate_cell_refuses_each_bridge_receipt_mutation(world, name, fn):
    cell = _bridge_cell(world)
    _mut(cell["receipt"], fn)
    assert ev.validate_cell(cell) != [] and ev.validate_cell(cell, offline=True) != []


def test_a_run_cell_receipt_may_not_carry_the_bridge_keys(world):
    cell = _cell(world)
    _mut(cell["receipt"], lambda r: r.update(checkpoint_source="exp43_external", bridge_descriptor_sha256=ev.BRIDGE_DESCRIPTOR_SHA256))
    assert any("receipt schema" in p for p in ev.validate_cell(cell))
    b = _bridge_cell(world, seed=43)
    _mut(b["metrics"], lambda r: r.update(ckpt_sha256_loaded="1" * 64))                # not the descriptor's checkpoint
    assert any("ckpt sha" in p for p in ev.validate_cell(b))


def test_prepare_bridge_verifies_sha_bytes_and_the_embedded_config(world, monkeypatch, tmp_path):
    import io
    import torch
    cfg = json.load(open(os.path.join(REPO, ev.reg.ARMS["rf"]["config"])))
    path = tmp_path / "epoch=8-step=40000.ckpt"

    def save(model_config):
        buf = io.BytesIO()
        torch.save({"model_config": model_config, "state_dict": {}}, buf)
        path.write_bytes(buf.getvalue())
        monkeypatch.setitem(ev.RF100_BRIDGE, "sha256", hashlib.sha256(buf.getvalue()).hexdigest())
        monkeypatch.setitem(ev.RF100_BRIDGE, "bytes", len(buf.getvalue()))
    save(cfg)
    out = ev._prepare_bridge(str(path), torch, io)
    link = ev.bridge_cells()[0]["ckpt_link"]
    assert out["sha256"] == ev.RF100_BRIDGE["sha256"] and os.readlink(link) == str(path)
    os.remove(link)
    other = json.loads(json.dumps(cfg))
    other["training"]["yaw_aug"]["flip"] = False                                    # exp_15's YAW config
    save(other)
    with pytest.raises(ev.EvalError, match="embedded model config"):
        ev._prepare_bridge(str(path), torch, io)
    save(cfg)
    monkeypatch.setitem(ev.RF100_BRIDGE, "bytes", ev.RF100_BRIDGE["bytes"] + 1)
    with pytest.raises(ev.EvalError, match="sha256 / bytes"):
        ev._prepare_bridge(str(path), torch, io)
    save(cfg)
    monkeypatch.setitem(ev.RF100_BRIDGE, "sha256", "0" * 64)
    with pytest.raises(ev.EvalError, match="sha256 / bytes"):
        ev._prepare_bridge(str(path), torch, io)
    save(cfg)
    monkeypatch.setitem(ev.RF100_BRIDGE, "model_config_sha256", "0" * 64)
    with pytest.raises(ev.EvalError, match="model config"):
        ev._prepare_bridge(str(path), torch, io)
    assert not os.path.lexists(link)


def _bundle(world, mutate=None, admission=None, stream_fn=None, make=True):
    """Our ten admitted bridge cells + exp_43's reference bundle (references = our values with exp_43's metadata;
    the admission record = exp_43's exact admission of the RF100_BRIDGE checkpoint unless ``admission`` is given)."""
    b = world["tmp"] / "bundle"
    (b / "cells").mkdir(parents=True)
    (b / "admission").mkdir()
    for c in ev.bridge_cells():
        cell = _bridge_cell(world, c["K"], c["seed"], stream_fn=(lambda s, c=c: stream_fn(c, s)) if stream_fn else None) \
            if make else c
        stem = ev.bridge_ref_stem(ev.RF100_BRIDGE["cells"][c["name"]])
        m, s = json.load(open(cell["metrics"])), fx.stream_sidecar(c["K"], c["seed"])      # exp_43's own (pristine) stream
        m.update(eval_name=ev.RF100_BRIDGE["cells"][c["name"]], ckpt_path="/home/yixunhu/x/farm/RF/epoch=8-step=40000.ckpt",
                 source_sha="5f5b493a8ac1" + "0" * 28)                             # metadata only: not compared
        if mutate:
            mutate(c, m, s)
        json.dump(m, open(b / "cells" / f"{stem}.json", "w"))
        json.dump(s, open(b / "cells" / f"{stem}.stream.json", "w"))
    if admission is None:
        admission = fx.exp43_admission(KIT43, ev.RF100_BRIDGE["sha256"], ev.RF100_BRIDGE["bytes"])
    json.dump(admission, open(b / "admission" / "exp43_admission_RF.json", "w"))
    rels = sorted(os.path.relpath(os.path.join(d, f), b) for d, _, fs in os.walk(b) for f in fs)
    (b / ev.BRIDGE_MANIFEST).write_text("".join(f"{ev.reg.sha256_file(str(b / r))}  {r}\n" for r in rels))
    return b


@pytest.fixture
def bridge_world(world, monkeypatch):
    monkeypatch.setattr(ev, "BRIDGE_RESULT", str(world["tmp"] / "rf_data_curve_bridge.json"))
    return world


def test_the_bridge_seals_equal_cells_and_records_the_bundle_hashes(bridge_world):
    b = _bundle(bridge_world)
    rec = ev.cmd_bridge(str(b))
    assert rec["verdict"] == "equal" and sorted(rec["cells"]) == sorted(ev.RF100_BRIDGE["cells"])
    assert all(v["verdict"] == "equal" and v["differences"] == {} for v in rec["cells"].values())
    adm = b / "admission" / "exp43_admission_RF.json"
    assert len(rec["bundle_files"]) == 21 and rec["admission_record"] == {
        "path": "admission/exp43_admission_RF.json", "sha256": ev.reg.sha256_file(str(adm)), "run_id": "exp43_RF_a6000_2x32",
        "ckpt_path": ev.RF100_BRIDGE["checkpoint"], "ckpt_sha256": ev.RF100_BRIDGE["sha256"],
        "bytes": ev.RF100_BRIDGE["bytes"], "global_step": 40000, "model_config": ev.reg.ARMS["rf"]["config"],
        "model_config_sha256": ev.RF100_BRIDGE["model_config_sha256"]}
    assert rec["admission_record"]["sha256"] == rec["bundle_files"]["admission/exp43_admission_RF.json"]
    assert rec["bundle_manifest_sha256"] == ev.reg.sha256_file(str(b / ev.BRIDGE_MANIFEST))
    sealed, _ = ev.reg.read_sealed(ev.BRIDGE_RESULT, ev.BRIDGE_KEYS)
    assert sealed == json.loads(json.dumps(rec)) and sealed["descriptor_sha256"] == ev.BRIDGE_DESCRIPTOR_SHA256
    with pytest.raises(ev.reg.GateError):
        ev.cmd_bridge(str(b))                                                      # sealed once


def test_a_numeric_difference_is_an_evaluator_discrepancy_not_a_failure(bridge_world):
    def bump(c, m, s):
        if c["name"] == "exp45_RF100bridge_K1_s44":
            m["metrics"]["T60"] += 1e-6
            m["by_scene"]["Cafe"]["EDT"] += 1e-6
    rec = ev.cmd_bridge(str(_bundle(bridge_world, bump)))
    assert rec["verdict"] == "evaluator discrepancy"
    bad = rec["cells"]["exp45_RF100bridge_K1_s44"]
    assert bad["verdict"] == "evaluator discrepancy" and sorted(bad["differences"]) == ["by_scene.Cafe.EDT", "metrics.T60"]
    assert [k for k, v in rec["cells"].items() if v["verdict"] != "equal"] == ["exp45_RF100bridge_K1_s44"]


@pytest.mark.parametrize("name", ["ref ckpt sha", "ref ckpt bytes", "ref input hash", "ref assignment hash", "ref count",
                                  "ref eval name", "ref seed", "ref cond", "stream offsets", "ref metric nan"])
def test_every_bridge_integrity_failure_refuses(bridge_world, name):
    target = "exp45_RF100bridge_K8_s45"

    def mutate(c, m, s):
        if c["name"] != target:
            return
        {"ref ckpt sha": lambda: m.update(ckpt_sha256_loaded="0" * 64), "ref ckpt bytes": lambda: m.update(ckpt_bytes_loaded=1),
         "ref input hash": lambda: s.update(input_hash="0" * 64), "ref assignment hash": lambda: s.update(assignment_hash="0" * 64),
         "ref count": lambda: s.update(stream_count=6336), "ref eval name": lambda: m.update(eval_name="exp43_RF_T_S40000_K8_s46"),
         "ref seed": lambda: m.update(seed=46), "ref cond": lambda: m.update(cond_method="fa_invariant"),
         "stream offsets": lambda: s.update(offsets=[1]), "ref metric nan": lambda: m["metrics"].update(C50=float("nan")),
         }[name]()
    with pytest.raises(ev.EvalError, match="bridge integrity"):
        ev.cmd_bridge(str(_bundle(bridge_world, mutate)))
    assert not os.path.exists(ev.BRIDGE_RESULT)


def test_the_bundle_itself_is_integrity_checked(bridge_world):
    b = _bundle(bridge_world)
    f = b / "cells" / (ev.bridge_ref_stem("exp43_RF_T_S40000_K8_s42") + ".json")
    good = f.read_bytes()
    f.write_bytes(good + b" ")
    with pytest.raises(ev.EvalError, match="manifest sha256"):
        ev.cmd_bridge(str(b))
    f.write_bytes(good)
    (b / "cells" / "extra.json").write_text("{}")
    with pytest.raises(ev.EvalError, match="unmanifested"):
        ev.cmd_bridge(str(b))
    os.remove(b / "cells" / "extra.json")
    man = (b / ev.BRIDGE_MANIFEST).read_text().split("\n")
    (b / ev.BRIDGE_MANIFEST).write_text("\n".join(ln for ln in man if "K1_s46.stream" not in ln))
    with pytest.raises(ev.EvalError, match="20 reference cell files"):
        ev.cmd_bridge(str(b))
    assert not os.path.exists(ev.BRIDGE_RESULT)


def _adm(**over):
    return fx.exp43_admission(KIT43, ev.RF100_BRIDGE["sha256"], ev.RF100_BRIDGE["bytes"], **over)


def _drop(key):
    rec = _adm()
    rec.pop(key)
    return rec


@pytest.mark.parametrize("name,admission", [
    ("only names the checkpoint", {"run": "exp43_RF_a6000_2x32", "note": "ckpt " + "ae62977e6a446c696f4b8f34e9393d335406bc8cd6404e7cc9caceaf55d1b6f6"}),
    ("not admitted", _adm(admitted=False)), ("another arm", _adm(arm="YAW")), ("another kind", _adm(kind="exp40_admission")),
    ("another config sha", _adm(model_config_sha256="0" * 64)), ("another config", _adm(model_config="x.json")),
    ("another checkpoint", _adm(sha256="0" * 64)),
    ("another checkpoint, consistent", dict(_adm(sha256="0" * 64), checkpoints=dict(_adm()["checkpoints"],
                                                                                   **{"epoch=8-step=40000.ckpt": "0" * 64}))),
    ("other bytes", _adm(bytes=723922730)), ("another step", _adm(global_step=39999)),
    ("another run", _adm(run_id="exp43_RF_a6000_2x32_r1")), ("not a strict load", _adm(strict_load=False)),
    ("a missing key", _drop("audit_sha256")), ("an extra key", dict(_adm(), extra=1)), ("no history", _adm(train_history=None)),
])
def test_the_admission_record_is_exp43s_exact_admission_of_the_descriptors_checkpoint(bridge_world, name, admission):
    """Codex code r1 finding 2 (deviation 5 rejected): the bundle's record must pass exp_43's own validator
    (exp43_kit.admission_reasons, rehash=None -- the exact ADMISSION_KEYS schema, admitted, RF, the run grammar, the
    16 cadence names, the pinned config, strict load, the W&B / history bindings) and admit exactly RF100_BRIDGE's
    checkpoint: sha256, bytes, config + its sha, step 40000, path. Substring presence of the sha proves nothing."""
    with pytest.raises(ev.EvalError, match="admission"):
        ev.cmd_bridge(str(_bundle(bridge_world, admission=admission)))
    assert not os.path.exists(ev.BRIDGE_RESULT)


def test_the_fixture_admission_record_passes_exp43s_own_validator(tmp_path):
    """The negative cases above are meaningful: the fixture record itself is exp_43-valid (record only)."""
    p = tmp_path / "adm.json"
    json.dump(_adm(), open(p, "w"))
    reasons, rec, _ = KIT43.admission_reasons(str(p), rehash=None)
    assert reasons == [] and set(rec) == KIT43.ADMISSION_KEYS and rec["sha256"] == ev.RF100_BRIDGE["sha256"]
    assert ev.RF100_BRIDGE["global_step"] == rec["global_step"] == 40000


def _strip(s):
    for k in ev.STREAM_ITEMS:
        s.pop(k)


def _short(s):
    for k in ev.STREAM_ITEMS:
        s[k] = s[k][:-1]


def _retarget(s):                                     # a changed item, input AND assignment tuple alike
    s["input_tuples"][7][1] = "Cafe/Cafe_idx_0/S000_R999_hybrid_IR.wav"
    s["assignment_tuples"][7][1] = "Cafe/Cafe_idx_0/S000_R999_hybrid_IR.wav"


def _offsets(s):
    s["offsets"][3] = 1


def _position(s):
    s["input_tuples"][0][0] = 1
    s["assignment_tuples"][0][0] = 1


@pytest.mark.parametrize("side,fn", [("both", _strip), ("ours", _strip), ("reference", _strip), ("both", _short),
                                     ("reference", _short), ("both", _retarget), ("ours", _retarget),
                                     ("reference", _offsets), ("both", _position)])
def test_incomplete_or_inconsistent_stream_preimages_are_integrity_failures(bridge_world, side, fn):
    """Codex code r1 finding 3: the bridge compares COMPLETE per-item preimages -- the 6,337-item input tuples,
    assignment tuples and offsets, structurally valid and re-hashing to the sidecar's input / assignment hash. Missing
    on both sides they no longer compare equal; changed identically on both sides they fail their own hashes."""
    target = "exp45_RF100bridge_K8_s44"
    ours = (lambda c, s: fn(s) if c["name"] == target else None) if side in ("ours", "both") else None
    ref = (lambda c, m, s: fn(s) if c["name"] == target else None) if side in ("reference", "both") else None
    with pytest.raises(ev.EvalError, match="bridge integrity"):
        ev.cmd_bridge(str(_bundle(bridge_world, ref, stream_fn=ours)))
    assert not os.path.exists(ev.BRIDGE_RESULT)


def test_an_unadmitted_bridge_cell_refuses_the_comparison(bridge_world):
    b = _bundle(bridge_world)
    c = ev.bridge_cells()[3]
    _mut(c["receipt"], lambda r: r.update(checkpoint_source="exp45_training"))
    with pytest.raises(ev.EvalError, match="bridge integrity"):
        ev.cmd_bridge(str(b))
