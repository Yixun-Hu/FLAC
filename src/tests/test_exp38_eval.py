"""exp_38 B8/B9 — the evaluation driver (exp38_eval.py) and the collector (exp38_collect.py).

Cell plan and argv (the reference cells' exact shape incl. ``--num-workers 4``), output names
(must equal eval_FLAC's own ``build_output_paths``), recursive type-strict equality for E0, the
transactional cell admission (one synthetic valid cell, then one mutation per failure), the pinned
input hashes and imported references (real data), aggregation, PENDING, the atomic result pair.
"""
import hashlib
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(name):
    import sys
    sys.path.insert(0, EXP)
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp38_eval")
col = _load("exp38_collect")


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
    assert len(refs) == 68
    assert ev.reg.sha256_file(os.path.join(EXP, "ref_cells/E0/E0a_K8_s42.json")) == \
        "4136ac75d570f84f76d34c5cac0f1e05e681897f4aefe53ffce12dd381f13180"
    assert ev.reg.sha256_file(os.path.join(EXP, "ref_cells/E0/E0b_K8_s42.json")) == \
        "f167470e3501bcc7c4f116b0c26164523a8bd140b55c5e72e935d88d3f373adb"


# ------------------------------------------------------------------ transactional admission
def _valid_cell(tmp_path, monkeypatch, run="de_bf_f025", K=8, seed=43):
    monkeypatch.setattr(ev, "DUMP_ROOT", str(tmp_path / "dumps"))
    monkeypatch.setitem(ev.reg.PLATFORMS["a6000"], "save_root", str(tmp_path / "nas"))
    cell = [c for c in ev.run_cells(run) if c["K"] == K and c["seed"] == seed][0]
    for k in ("ckpt_link", "metrics", "stream", "receipt"):
        cell[k] = str(tmp_path / "farm" / os.path.basename(cell[k]))
    (tmp_path / "farm").mkdir()
    ck_name = os.path.basename(cell["ckpt_link"])
    nas = tmp_path / "nas" / run
    nas.mkdir(parents=True)
    (nas / "CHECKPOINTS.sha256").write_text(f"{'a' * 64}  {ck_name}\n")
    fa = cell["cond"] == "fa_invariant"
    metrics = {k: 1.0 for k in ev.METRIC_KEYS}
    rec = {"metrics": metrics, "by_scene": {s: dict(metrics) for s in ev.SCENES}, "cond_method": cell["cond"],
           "cond_autocast": "bf16", "rotate_deg": 0.0, "seed": seed, "steps": 1, "cfg_scale": 1.0,
           "batch_size": 64, "n_samples": 6337, "device": "cuda", "dataset_config": ev.EVAL_CONFIGS[K],
           "eval_name": cell["name"], "weights_source": "ema", "per_scene_schema": 1, "scene_count": 10,
           "frame_avg_angles": [0.0, 90.0, 180.0, 270.0] if fa else None, "frame_avg_fwd_cap": 64 if fa else None,
           "orbit_execution": "batched" if fa else "n/a", "source_sha": "f" * 40, "ckpt_sha256_loaded": "a" * 64}
    json.dump(rec, open(cell["metrics"], "w"))
    json.dump({"stream_count": 6337, "input_hash": ev.INPUT_HASH[(K, seed)]}, open(cell["stream"], "w"))
    cell["dump_dir"] = str(tmp_path / "dumps" / cell["stem"])
    os.makedirs(cell["dump_dir"])
    msha = ev.reg.sha256_file(cell["metrics"])
    ssha = ev.reg.sha256_file(cell["stream"])
    json.dump({"n": 6337, "dtype": "float32", "metrics_sha256": msha, "stream_sidecar_sha256": ssha,
               "input_hash": ev.INPUT_HASH[(K, seed)]}, open(os.path.join(cell["dump_dir"], "manifest.json"), "w"))
    open(os.path.join(cell["dump_dir"], "COMPLETE"), "w").write("ok")
    receipt = {"rc": 0, "sentinel": ev.SENTINEL, "argv": ev.eval_argv(cell), "metrics_sha256": msha,
               "stream_sha256": ssha, "model_config_sha256": ev.reg.sha256_file(ev._abs(cell["config"])),
               "dataset_config_sha256": ev.EVAL_CONFIG_SHA256[K], "unseen_split_sha256": ev.UNSEEN_SPLIT[1],
               "dump_manifest_sha256": ev.reg.sha256_file(os.path.join(cell["dump_dir"], "manifest.json"))}
    json.dump(receipt, open(cell["receipt"], "w"))
    return cell


def test_validate_cell_admits_a_complete_cell(tmp_path, monkeypatch):
    cell = _valid_cell(tmp_path, monkeypatch)
    assert ev.validate_cell(cell, "f" * 40) == []


@pytest.mark.parametrize("mutation", [
    ("metrics", lambda r: r["metrics"].pop("FD")), ("metrics", lambda r: r["metrics"].update(T60=float("nan"))),
    ("metrics", lambda r: r.update(cond_method="vanilla")), ("metrics", lambda r: r.update(n_samples=6336)),
    ("metrics", lambda r: r.update(frame_avg_fwd_cap=32)), ("metrics", lambda r: r.update(ckpt_sha256_loaded="b" * 64)),
    ("metrics", lambda r: r["by_scene"].pop("Cafe")), ("metrics", lambda r: r.update(source_sha="e" * 40)),
    ("stream", lambda r: r.update(input_hash="0" * 64)), ("receipt", lambda r: r.update(rc=1)),
    ("receipt", lambda r: r.update(sentinel="EXP35_NO_FLASH_LAUNCHER attention_backend=flash")),
    ("receipt", lambda r: r.update(argv=r["argv"][:-2])),
])
def test_validate_cell_refuses_each_mutation(tmp_path, monkeypatch, mutation):
    cell = _valid_cell(tmp_path, monkeypatch)
    which, fn = mutation
    rec = json.load(open(cell[which]))
    fn(rec)
    json.dump(rec, open(cell[which], "w"))
    assert ev.validate_cell(cell, "f" * 40) != []


def test_validate_cell_refuses_a_broken_dump(tmp_path, monkeypatch):
    cell = _valid_cell(tmp_path, monkeypatch)
    os.remove(os.path.join(cell["dump_dir"], "COMPLETE"))
    assert ev.validate_cell(cell, "f" * 40) != []


# ------------------------------------------------------------------ collector
def test_aggregate_and_seed_set():
    recs = {s: {k: float(s) for k in ev.METRIC_KEYS} for s in ev.SEEDS}
    st = col.aggregate(recs)
    assert st["T60"][0] == 44.0 and abs(st["T60"][1] - 1.5811388300841898) < 1e-12 and st["T60"][2] == 5
    with pytest.raises(ValueError):
        col.aggregate({s: recs[s] for s in (42, 43, 44, 45)})


def test_reference_rows_reproduce_table_3b():
    rows = col.reference_rows()
    t = {(m, f): rows[(m, f, 8)]["stats"] for (m, f, K) in rows if K == 8}
    assert round(t[("p1", "025")]["EDT"][0], 3) == 42.914 and round(t[("p1", "025")]["T60"][0], 2) == 10.25
    assert round(t[("p1", "100")]["EDT"][0], 3) == 40.650 and round(t[("p1", "100")]["T60"][0], 3) == 8.993
    assert round(t[("yaw", "100")]["T60"][0], 3) == 7.965 and round(t[("bf", "100")]["EDT"][0], 3) == 38.794


def test_run_block_pending_on_any_problem_and_render(tmp_path, monkeypatch):
    rows, probs = col.run_block("de_yaw_f025", validate=lambda c: [f"{c['name']}: missing"])
    assert rows == {} and len(probs) == 10
    md = col.render(col.reference_rows(), probs)
    assert md.count("PENDING") >= 12 and "de_yaw_f025_K8_s42: missing" in md


def test_publish_writes_the_pair_atomically(tmp_path):
    exp = tmp_path / col.reg.EXP_DIR
    exp.mkdir(parents=True)
    sha = col.publish(col.reference_rows(), [], repo=str(tmp_path))
    j = (tmp_path / col.OUT_JSON).read_bytes()
    assert hashlib.sha256(j).hexdigest() == sha and sha in (tmp_path / col.OUT_MD).read_text()
    assert not list(exp.glob("*.tmp"))
