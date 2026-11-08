"""exp_40 yaw_aug_c16 -- the evaluation driver (exp40_eval.py), CPU only, synthetic farms.

Covers the 85-cell list (and its exact derivation from the exp_25 YAW-pin rows), the off-grid
readout formula, the per-cell verification contract, the E0 replay comparator and gate, the
runtime-input manifest, the comparator registry and the figure contract. Real evaluator
helpers (eval_FLAC.build_output_paths / canonical_stream_hash) are used, never re-implemented.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp40_eval", _EXP / "exp40_eval.py")
import eval_FLAC  # noqa: E402  (the conftest puts the repo root first)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((_EXP / "exp40_cells.json").read_text())


@pytest.fixture(scope="module")
def exp25():
    return json.loads((_REPO / ev.EXP25_CELLS).read_text())


# --------------------------------------------------------------------------- #
# the cell list
# --------------------------------------------------------------------------- #
def test_cells_file_is_the_generator_output(manifest, exp25):
    assert ev.render(ev.build_cells(exp25)) == (_EXP / "exp40_cells.json").read_text()


def test_cell_counts(manifest):
    assert manifest["count"] == 85
    assert manifest["block_counts"] == {"E0": 3, "T": 10, "C4": 30, "G45": 8, "OFF": 8, "R": 10, "FILL": 16}
    ks = [c["K"] for c in manifest["cells"]]
    assert ks.count(1) == 42 and ks.count(8) == 43


def test_off_grid_angles_are_the_balanced_half_step_coset(manifest):
    off = sorted({c["rotate_deg"] for c in manifest["cells"] if c["block"] in ("OFF", "FILL")})
    assert off == [11.25, 101.25, 191.25, 281.25]
    for d in off:
        assert (d * 512 / 360) % 32 == 16                       # exactly half a C16 step
    g45 = sorted({c["rotate_deg"] for c in manifest["cells"] if c["block"] == "G45"})
    assert g45 == [45.0, 135.0, 225.0, 315.0] and all((d * 512 / 360) % 32 == 0 for d in g45)


def _strip(argv, *flags):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in flags:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def test_every_argv_is_the_exp25_template_except_the_substituted_flags(manifest, exp25):
    rows = {c["cell"]: c for c in exp25["cells"]}
    sub = ("--model-config", "--seed", "--rotate-mode", "--rotate-deg", "--rotate-seed", "--eval-name")
    for c in manifest["cells"]:
        t = rows[c["template"]]
        assert _strip(c["eval_args"], *sub) == _strip(t["eval_args"], *sub), c["cell"]
        a = dict(zip(c["eval_args"][::2], c["eval_args"][1::2]))
        assert a["--model-config"] == ev.ARMS[c["arm"]]["config"]
        assert a["--seed"] == str(c["seed"]) and a["--eval-name"] == c["cell"]
        assert a["--dataset-config"] == c["dataset_config"] == t["dataset_config"]
        if c["rotate_mode"] == "random":
            assert a["--rotate-seed"] == str(c["seed"]) and "--rotate-deg" not in a
        else:
            assert float(a["--rotate-deg"]) == c["rotate_deg"] and "--rotate-seed" not in a


def test_e0_cells_replay_the_archived_exp25_cells_exactly(manifest, exp25):
    rows = {c["cell"]: c for c in exp25["cells"]}
    e0 = [c for c in manifest["cells"] if c["block"] == "E0"]
    assert sorted(c["replay_of"] for c in e0) == sorted(ev.E0.values())
    for c in e0:
        assert _strip(c["eval_args"], "--eval-name") == _strip(rows[c["replay_of"]]["eval_args"], "--eval-name")


def test_stems_are_unique_and_use_rot_tokens(manifest):
    stems = [ev.stem_of(c, str(_REPO)) for c in manifest["cells"]]
    assert len(set(stems)) == len(stems)
    off = [ev.stem_of(c, str(_REPO)) for c in manifest["cells"] if c["cell"].endswith("th11p25")]
    assert off and all(s.endswith("_rot11p25") for s in off)


# --------------------------------------------------------------------------- #
# off-grid readout (plan §1)
# --------------------------------------------------------------------------- #
def _grid(c16_off_extra=0.0):
    x = {}
    for arm in ("C16", "YAW"):
        for k in ev.KS:
            for m in ev.ORIENT:
                for t in ev.OFF + (0.0, 90.0, 180.0, 270.0):
                    base = 10.0 + (0.1 if t in (90.0, 270.0) else 0.0)
                    bump = c16_off_extra if (arm == "C16" and t in ev.OFF) else 0.0
                    x[(arm, k, m, t)] = base + bump
    return x


def test_offgrid_readout_formula_and_labels():
    flat = ev.offgrid_readout(_grid())
    for v in flat.values():
        assert v["label"] == "within YAW's C4 spread" and math.isclose(v["S"], 0.1)
    worse = ev.offgrid_readout(_grid(c16_off_extra=1.0))
    assert worse[(8, "T60")]["label"] == "C16 worse off-grid"            # T60 up = worse
    assert worse[(8, "RIR_to_GT_RIR_R@1")]["label"] == "C16 better off-grid"   # R@1 up = better
    assert math.isclose(worse[(8, "T60")]["P"], -(1.0 + 0.0) + 0.0 - 0.0 + 0.05 - 0.05, abs_tol=1e-9) or \
        worse[(8, "T60")]["P"] < 0


def test_offgrid_readout_pending_on_any_missing_cell():
    x = _grid()
    del x[("YAW", 1, "EDT", 191.25)]
    assert ev.offgrid_readout(x)[(1, "EDT")] == {"label": "PENDING"}


# --------------------------------------------------------------------------- #
# per-cell verification and the E0 comparator (synthetic farm)
# --------------------------------------------------------------------------- #
def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def cell_env(tmp_path, manifest):
    cell = copy.deepcopy([c for c in manifest["cells"] if c["cell"] == "exp40_C16_C4_S40000_K8_s43_th90"][0])
    farm, dumps = tmp_path / "farm", tmp_path / "dumps"
    stem = ev.stem_of(cell, str(_REPO))
    inputs = [[i, f"t{i}", [f"c{i}"], 512] for i in range(6337)]
    assign = [[i, f"t{i}", 128] for i in range(6337)]
    stream = {"schema_version": 1, "stream_count": 6337, "img_w": 512, "rotate_mode": "fixed",
              "rotate_deg": 90.0, "rotate_seed": None, "input_tuples": inputs, "assignment_tuples": assign,
              "offsets": [128] * 6337, "input_hash": eval_FLAC.canonical_stream_hash(inputs),
              "assignment_hash": eval_FLAC.canonical_stream_hash(assign), "fingerprint_schema": 1}
    rec = {"cond_method": "vanilla", "cond_autocast": "bf16", "frame_avg_angles": None,
           "frame_avg_fwd_cap": None, "orbit_execution": "n/a", "batch_size": 64, "cfg_scale": 1.0,
           "steps": 1, "device": "cuda", "weights_source": "ema", "n_samples": 6337, "scene_count": 10,
           "per_scene_schema": 1, "seed": 43, "rotate_deg": 90.0, "dataset_config": cell["dataset_config"],
           "source_sha": "e" * 40, "ckpt_sha256_loaded": "a" * 64, "ckpt_bytes_loaded": 7,
           "eval_name": cell["cell"], "metrics": {k: 1.0 for k in ev.METRIC_KEYS},
           "by_scene": {s: {} for s in ev.SCENES}}
    msha = _write(farm / "C16" / f"{stem}.json", rec)
    ssha = _write(farm / "C16" / f"{stem}.stream.json", stream)
    d = dumps / stem
    d.mkdir(parents=True)
    (d / "waveforms.npy").write_bytes(b"w" * 10)
    wsha = hashlib.sha256(b"w" * 10).hexdigest()
    (d / "waveforms.sha256").write_text(f"{wsha}  waveforms.npy\n")
    (d / "COMPLETE").write_text(wsha + "\n")
    (d / "manifest.json").write_text(json.dumps({"n": 6337, "shape": [6337, 10240], "dtype": "float32",
                                                 "input_hash": stream["input_hash"], "metrics_stem": stem,
                                                 "metrics_sha256": msha, "stream_sidecar_sha256": ssha,
                                                 "sha256": wsha}))
    log = tmp_path / "cell.log"
    log.write_text(f"EXP35_NO_FLASH_LAUNCHER backend=sdpa overlay={ev.EVAL_PINS['overlays/no_flash/flash_attn.py']} x\n")
    hashes = {(8, 43): (stream["input_hash"], "unused")}
    return cell, farm, dumps, log, hashes, stem


def _verify(cell_env, **over):
    cell, farm, dumps, log, hashes, stem = cell_env
    kw = dict(head="e" * 40, ckpt_sha="a" * 64, ckpt_bytes=7)
    kw.update(over)
    r, _ = ev.verify_cell(cell, str(farm), str(_REPO), kw["head"], kw["ckpt_sha"], kw["ckpt_bytes"],
                          str(log), hashes, dump_root=str(dumps))
    return r


def test_verify_cell_passes(cell_env):
    assert _verify(cell_env) == []


@pytest.mark.parametrize("field", ["source_sha", "ckpt_sha", "ckpt_bytes"])
def test_verify_cell_binds_identity(cell_env, field):
    over = {"source_sha": {"head": "f" * 40}, "ckpt_sha": {"ckpt_sha": "b" * 64}, "ckpt_bytes": {"ckpt_bytes": 8}}[field]
    assert _verify(cell_env, **over)


def _edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(cond_method="fa_invariant"), "cond_method"),
    (lambda d: d.update(cond_autocast="default"), "cond_autocast"),
    (lambda d: d.update(n_samples=6336), "n_samples"),
    (lambda d: d["metrics"].pop("FD"), "metric keys"),
    (lambda d: d["metrics"].update(T60=float("nan")), "finiteness"),
    (lambda d: d["by_scene"].pop("Cafe"), "by_scene"),
    (lambda d: d.update(rotate_deg=90), "rotate_deg"),                     # int, not float: type-strict
])
def test_verify_cell_metrics_contract(cell_env, mutate, needle):
    cell, farm, dumps, log, hashes, stem = cell_env
    _edit(farm / "C16" / f"{stem}.json", mutate)
    assert any(needle in r for r in _verify(cell_env))


def test_verify_cell_stream_and_dump_contract(cell_env):
    cell, farm, dumps, log, hashes, stem = cell_env
    _edit(farm / "C16" / f"{stem}.stream.json", lambda d: d["offsets"].__setitem__(5, 0))
    assert any("offsets" in r for r in _verify(cell_env))
    _edit(farm / "C16" / f"{stem}.stream.json", lambda d: d["offsets"].__setitem__(5, 128))
    _edit(farm / "C16" / f"{stem}.stream.json", lambda d: d.update(input_hash="0" * 64))
    r = _verify(cell_env)
    assert any("recompute" in x for x in r) and any("input_hash" in x for x in r)


def test_verify_cell_dump_and_sentinel(cell_env):
    cell, farm, dumps, log, hashes, stem = cell_env
    (dumps / stem / "waveforms.npy").write_bytes(b"tampered")
    assert any("re-hash" in r for r in _verify(cell_env))
    log.write_text("no sentinel here\n")
    assert any("sentinel" in r for r in _verify(cell_env))


def test_e0_gate(tmp_path):
    p = tmp_path / "e0.json"
    assert ev.e0_gate_reasons(str(p), "a" * 40, "c")
    p.write_text(json.dumps({"verdict": "EQUAL", "eval_sha": "a" * 40, "closure": "c", "cells": sorted(ev.E0)}))
    assert ev.e0_gate_reasons(str(p), "a" * 40, "c") == []
    assert ev.e0_gate_reasons(str(p), "b" * 40, "c")                  # another EVAL_SHA
    assert ev.e0_gate_reasons(str(p), "a" * 40, "d")                  # another closure
    p.write_text(json.dumps({"verdict": "EQUAL", "eval_sha": "a" * 40, "closure": "c", "cells": sorted(ev.E0)[:2]}))
    assert ev.e0_gate_reasons(str(p), "a" * 40, "c")                  # not all three replays


def test_runtime_inputs_are_hash_pinned(tmp_path):
    f = tmp_path / "admission.json"
    f.write_text('{"admitted": true}')
    spec = tmp_path / "inputs.json"
    spec.write_text(json.dumps({"admission": {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}}))
    inputs, sha = ev.runtime_inputs(str(spec))
    assert inputs["admission"]["path"] == str(f) and len(sha) == 64
    f.write_text('{"admitted": false}')
    with pytest.raises(ev.Refused, match="does not hash"):
        ev.runtime_inputs(str(spec))


def test_c16_checkpoint_needs_the_runtime_admission():
    with pytest.raises(ev.Refused, match="runtime-inputs"):
        ev.arm_checkpoint("C16", str(_REPO), None)
    assert ev.arm_checkpoint("YAW", str(_REPO))[1].startswith("ac1f2603")


# --------------------------------------------------------------------------- #
# figure contract
# --------------------------------------------------------------------------- #
def test_figure_contract_covers_every_arm_angle_and_k():
    c = json.loads((_EXP / "exp40_figure_contract.json").read_text())
    assert c["arms"] == ["C16", "YAW", "P1"] and c["Ks"] == [1, 8]
    assert len(c["cells"]) == 3 * 2 * len(c["angles"])
    rows = [dict(e, **{m: 1.0 for m in ev.METRIC_KEYS}) for e in c["cells"]]
    assert ev.figure_reasons(rows, c) == []
    assert ev.figure_reasons(rows[1:], c)                              # a missing cell
    assert ev.figure_reasons(rows + rows[:1], c)                       # a duplicate


# --------------------------------------------------------------------------- #
# comparator registry (synthetic exp_25 / exp_36 farms)
# --------------------------------------------------------------------------- #
def _farms(tmp_path, exp36_pin="9" * 40, drop=None, nodump=None):
    import itertools
    want = []
    for arm in ("YAW", "P1"):
        for k in ev.KS:
            for s in ev.SEEDS:
                want += [("exp25", arm, f"exp25_{arm}_T_S40000_K{k}_s{s}", s, 0.0),
                         ("exp25", arm, f"exp25_{arm}_R_S40000_K{k}_s{s}", s, None)]
            want += [("exp25", arm, f"exp25_{arm}_C4_S40000_K{k}_s42_th{d}", 42, float(d)) for d in (90, 180, 270, 45)]
            want += [("exp36", arm, (f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" if arm == "YAW"
                                     else f"exp35_P1_OFF_S40000_K{k}_s42_th{d}"), 42, float(d)) for d in (135, 225, 315)]
    for k in ev.KS:
        want += [("exp36", "YAW", f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}", s, float(d)) for s in (43, 44, 45, 46) for d in (90, 180, 270)]
    dumps = tmp_path / "dumps"
    for i, (src, arm, name, s, deg) in enumerate(want):
        if name == drop:
            continue
        farm = tmp_path / src / arm
        farm.mkdir(parents=True, exist_ok=True)
        stem = f"stem{i}"
        (farm / f"{stem}.json").write_text(json.dumps({
            "source_sha": "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e" if src == "exp25" else exp36_pin,
            "cond_method": "vanilla", "seed": s, "rotate_deg": deg, "n_samples": 6337,
            "ckpt_sha256_loaded": ev.ARMS[arm]["ckpt_sha256"], "metrics": {}}))
        (farm / f"{stem}.stream.json").write_text(json.dumps({"input_hash": "i", "assignment_hash": "a"}))
        (farm / f"{stem}.cell.json").write_text(json.dumps({"cell": name, "stem": stem, "dump": name != nodump,
                                                            "dump_root": str(dumps)}))
        (dumps / stem).mkdir(parents=True)
        (dumps / stem / "waveforms.sha256").write_text("w" * 64 + "  waveforms.npy\n")
        (dumps / stem / "manifest.json").write_text("{}")
    return tmp_path / "exp25", tmp_path / "exp36", len(want)


def test_registry_pins_exactly_the_registered_mix(tmp_path):
    a, b, n = _farms(tmp_path)
    reg = ev.register_comparators(str(a), str(b))
    assert reg["count"] == n == 2 * (2 * 5 * 2 + 2 * 4 + 2 * 3) + 2 * 4 * 3
    assert reg["exp36_pin"] == "9" * 40 and ev.registry_reasons(reg) == []
    Path(reg["cells"][0]["metrics_path"]).write_text("{}")
    assert ev.registry_reasons(reg)


def test_registry_refuses_a_missing_cell_and_a_dumpless_one(tmp_path):
    a, b, _ = _farms(tmp_path / "x", drop="exp25_P1_T_S40000_K8_s44")
    with pytest.raises(ev.Refused, match="0 receipts"):
        ev.register_comparators(str(a), str(b))
    a, b, _ = _farms(tmp_path / "y", nodump="exp25_YAW_C4_S40000_K1_s45_th180")
    with pytest.raises(ev.Refused, match="no dump"):
        ev.register_comparators(str(a), str(b))


def test_validate_row_refuses_unregistered_files(tmp_path):
    f = tmp_path / "not_ours_metrics.json"
    f.write_text("{}")
    ok, problems = ev.validate_row([str(f)], str(tmp_path), str(_REPO))
    assert not ok and "not a registered" in problems[0]


# --------------------------------------------------------------------------- #
# model_comparison.md contract exp40 (plan §5.4)
# --------------------------------------------------------------------------- #
def test_generator_registers_two_exp40_rows_and_blocks_unverified_files(tmp_path):
    gen = _load("gen_model_comparison_exp40", _REPO / "worklog/worklog_yixun/gen_model_comparison.py")
    rows = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp40"]
    assert sorted(r[2] for r in rows) == [1, 8] and all(r[0] == gen.EXP40_LABEL for r in rows)
    assert all("exp40_C16_T_S40000_K" in r[3][0] for r in rows)
    assert gen.validate_exp40_cell([]) == (True, [])
    f = tmp_path / "x_metrics_1_1.0_exp40_C16_T_S40000_K1_s42.json"
    f.write_text("{}")
    ok, problems = gen.validate_exp40_cell([str(f)])
    assert not ok and problems
