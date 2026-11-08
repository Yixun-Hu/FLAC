"""exp_40 yaw_aug_c16 -- the evaluation driver (exp40_eval.py), CPU only, synthetic farms.

Covers the 85-cell list (and its exact derivation from the exp_25 YAW-pin rows), the off-grid
readout formula (exact values), the strict per-cell verification contract (metrics / per-scene /
stream / dump / sentinel schemas), the committed E0 reference and the replay comparator, the
strict E0 and cell receipts shared by run / verify / collect / table, the runtime-input manifest,
the AR and environment gates, the content-addressed checkpoint cache under true thread
concurrency, the lane locks and owned-children termination, worker exceptions, the detached
worktree staying clean (no bytecode), the comparator registry, collect / tables / paper import /
frozen CSV, the offline table validator and the generator's exp40 contract. Real evaluator
helpers (eval_FLAC.build_output_paths / canonical_stream_hash, src.data.yaw_rotation) are used,
never re-implemented. Written by the exp_40 Coder seat (Claude Opus 5.5) for code review 1.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
TREE = str(_REPO)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp40_eval", _EXP / "exp40_eval.py")
kit = ev.kit
import eval_FLAC  # noqa: E402  (the conftest puts the repo root first)

UUIDS = [f"GPU-{i:08d}-0000-0000-0000-000000000000" for i in range(2)]
TARGET = os.path.join(TREE, "eval_FLAC.py")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((_EXP / "exp40_cells.json").read_text())


@pytest.fixture(scope="module")
def exp25():
    return json.loads((_REPO / ev.EXP25_CELLS).read_text())


def _cell(manifest, name):
    return copy.deepcopy([c for c in manifest["cells"] if c["cell"] == name][0])


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


def test_exp25_cells_file_is_pinned():
    assert kit.sha256_file(str(_REPO / ev.EXP25_CELLS)) == ev.EXP25_CELLS_SHA256


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
    stems = [ev.stem_of(c, TREE) for c in manifest["cells"]]
    assert len(set(stems)) == len(stems)
    off = [ev.stem_of(c, TREE) for c in manifest["cells"] if c["cell"].endswith("th11p25")]
    assert off and all(s.endswith("_rot11p25") for s in off)


def test_eval_argv_runs_eval_flac_through_the_no_flash_launcher(manifest):
    c = _cell(manifest, "exp40_C16_T_S40000_K1_s42")
    a = ev.eval_argv(c, TREE, "/farm/C16/epoch=8-step=40000.ckpt", "/py", ev.DUMP_ROOT)
    assert a[:3] == ["/py", os.path.join(TREE, ev.NO_FLASH_LAUNCHER), TARGET]
    assert a[3:-4] == c["eval_args"] and a[-4:] == ["--ckpt-path", "/farm/C16/epoch=8-step=40000.ckpt",
                                                      "--dump-scored-waveforms", ev.DUMP_ROOT]


# --------------------------------------------------------------------------- #
# B7: the evaluator of ONE tree per process, and the detached worktree stays clean
# --------------------------------------------------------------------------- #
def test_evaluator_refuses_a_second_tree(tmp_path):
    ev.evaluator(TREE)
    with pytest.raises(ev.Refused, match="second evaluation tree"):
        ev.evaluator(str(tmp_path))


def test_stem_generation_leaves_a_detached_worktree_clean(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    blob = subprocess.run(["git", "-C", TREE, "archive", "HEAD", "--", "eval_FLAC.py", "src", "AGREE"],
                          capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(wt)], input=blob, check=True)
    git = ["git", "-C", str(wt), "-c", "user.name=exp40", "-c", "user.email=exp40"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "tree"], check=True)
    subprocess.run(git + ["checkout", "-q", "--detach", "HEAD"], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}       # no -B, no env help
    snippet = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('e', {str(_EXP / 'exp40_eval.py')!r})\n"
        "ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)\n"
        f"cells = json.load(open({str(_EXP / 'exp40_cells.json')!r}))['cells']\n"
        "stems = [ev.stem_of(c, sys.argv[1]) for c in cells]\n"
        "offs = ev.expected_offsets([c for c in cells if c['rotate_mode'] == 'random'][0], sys.argv[1])\n"
        "print('STEMS', len(set(stems)), len(offs))\n")
    for _ in range(2):
        p = subprocess.run([sys.executable, "-c", snippet, str(wt)], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=600)
        assert p.returncode == 0 and "STEMS 85 6337" in p.stdout, p.stderr[-2000:]
        status = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=all"],
                                capture_output=True, text=True, check=True).stdout
        assert status == "", status
    assert not list(wt.rglob("__pycache__"))
    # control: an unprotected import of the same tree DOES write bytecode (the test is not vacuous)
    subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import eval_FLAC",
                    str(wt)], cwd=str(tmp_path), env=env, capture_output=True, check=True, timeout=600)
    assert list(wt.rglob("__pycache__"))


# --------------------------------------------------------------------------- #
# off-grid readout (plan §1): exact values
# --------------------------------------------------------------------------- #
GRID = (0.0, 90.0, 180.0, 270.0)


def _x(yaw_grid, yaw_off, c16_off, p1=True):
    """x(arm, K, metric, theta): grid values per angle, OFF values = grid(theta - 11.25) + bump."""
    x = {}
    for k in ev.KS:
        for m in ev.ORIENT:
            for i, g in enumerate(GRID):
                for arm, bump in (("YAW", yaw_off), ("C16", c16_off)) + ((("P1", 0.3),) if p1 else ()):
                    x[(arm, k, m, g)] = yaw_grid[i]
                    x[(arm, k, m, g + 11.25)] = yaw_grid[i] + bump
    return x


def test_offgrid_readout_exact_values():
    x = _x((10.0, 10.05, 10.0, 10.05), 0.5, 0.1)
    out = ev.offgrid_readout(x)
    t60, r1 = out[(8, "T60")], out[(8, "RIR_to_GT_RIR_R@1")]
    assert math.isclose(t60["d_YAW"], -0.5) and math.isclose(t60["d_C16"], -0.1)       # q = -1 (lower is better)
    assert math.isclose(t60["d_P1"], -0.3) and math.isclose(t60["P"], 0.4) and math.isclose(t60["S"], 0.05)
    assert t60["label"] == "C16 better off-grid"
    assert math.isclose(r1["d_YAW"], 0.5) and math.isclose(r1["d_C16"], 0.1) and math.isclose(r1["P"], -0.4)
    assert r1["label"] == "C16 worse off-grid"
    within = ev.offgrid_readout(_x((10.0, 10.6, 10.2, 10.4), 0.5, 0.2))[(1, "EDT")]
    assert math.isclose(within["P"], 0.3) and math.isclose(within["S"], 0.6) and within["label"] == "within YAW's C4 spread"


def test_offgrid_readout_pending_and_p1_optional():
    x = _x((10.0, 10.05, 10.0, 10.05), 0.5, 0.1)
    del x[("YAW", 1, "EDT", 191.25)]
    assert ev.offgrid_readout(x)[(1, "EDT")] == {"label": "PENDING"}
    nop1 = ev.offgrid_readout(_x((10.0, 10.05, 10.0, 10.05), 0.5, 0.1, p1=False))
    assert nop1[(8, "T60")]["d_P1"] is None and nop1[(8, "T60")]["label"] == "C16 better off-grid"


# --------------------------------------------------------------------------- #
# synthetic artifacts with the exact exp_25 schemas
# --------------------------------------------------------------------------- #
_INPUTS = {}


def _inputs(k, seed):
    if (k, seed) not in _INPUTS:
        _INPUTS[(k, seed)] = [[i, f"{i}|single_channel_ir_1/Cafe/t{seed}_{i}.wav",
                               [f"{seed}.{j},{i}.0,0.000000" for j in range(k)], 512] for i in range(6337)]
    return _INPUTS[(k, seed)]


def _rand_cell(seed):
    return {"rotate_mode": "random", "rotate_seed": seed, "rotate_deg": None}


@pytest.fixture(scope="module")
def hashes():
    out = {}
    for k in ev.KS:
        for s in ev.SEEDS:
            it = _inputs(k, s)
            offs = ev.expected_offsets(_rand_cell(s), TREE)
            out[(k, s)] = (eval_FLAC.canonical_stream_hash(it),
                           eval_FLAC.canonical_stream_hash([[i, t[1], o] for i, (t, o) in enumerate(zip(it, offs))]))
    return out


def _metrics_block(base=1.0):
    return {k: base + 0.01 * i for i, k in enumerate(ev.METRIC_KEYS)}


def make_artifacts(cell, farm, dumps, log_path, head, link, ckpt_sha, ckpt_bytes, target=TARGET):
    p = ev.cell_paths(cell, str(farm), TREE, str(dumps))
    offs = ev.expected_offsets(cell, TREE)
    it = _inputs(cell["K"], cell["seed"])
    at = [[i, t[1], o] for i, (t, o) in enumerate(zip(it, offs))]
    stream = {"schema_version": 1, "fingerprint_schema": 1, "rotate_mode": cell["rotate_mode"],
              "rotate_seed": cell["rotate_seed"], "rotate_deg": cell["rotate_deg"], "img_w": 512,
              "stream_count": 6337, "input_tuples": it, "offsets": list(offs), "assignment_tuples": at,
              "input_hash": eval_FLAC.canonical_stream_hash(it), "assignment_hash": eval_FLAC.canonical_stream_hash(at)}
    rec = {"cond_method": "vanilla", "cond_autocast": "bf16", "frame_avg_angles": None,
           "frame_avg_fwd_cap": None, "orbit_execution": "n/a", "batch_size": 64, "cfg_scale": 1.0,
           "steps": 1, "device": "cuda", "weights_source": "ema", "n_samples": 6337, "scene_count": 10,
           "per_scene_schema": 1, "seed": cell["seed"], "rotate_deg": cell["rotate_deg"],
           "dataset_config": cell["dataset_config"], "source_sha": head, "ckpt_sha256_loaded": ckpt_sha,
           "ckpt_bytes_loaded": ckpt_bytes, "eval_name": cell["cell"], "ckpt_path": link,
           "metrics": _metrics_block(1.0 + cell["seed"] / 100.0),
           "by_scene": {s: _metrics_block(2.0 + i) for i, s in enumerate(ev.SCENES)}}
    if cell["rotate_mode"] == "random":
        rec.update(rotate_mode="random", rotate_seed=cell["seed"], stream_count=6337, img_w=512,
                   input_hash=stream["input_hash"], assignment_hash=stream["assignment_hash"])
    Path(p["metrics"]).parent.mkdir(parents=True, exist_ok=True)
    mb, sb = json.dumps(rec).encode(), json.dumps(stream).encode()
    Path(p["metrics"]).write_bytes(mb)
    Path(p["stream"]).write_bytes(sb)
    d = Path(p["dump"])
    d.mkdir(parents=True)
    npy = f"wave-{cell['cell']}".encode()
    (d / "waveforms.npy").write_bytes(npy)
    wsha = hashlib.sha256(npy).hexdigest()
    (d / "waveforms.sha256").write_text(f"{wsha}  waveforms.npy\n")
    (d / "COMPLETE").write_text(wsha + "\n")
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "n": 6337, "dtype": "float32", "shape": [6337, 10240], "sha256": wsha,
        "input_hash": stream["input_hash"], "metrics_stem": p["stem"], "source_sha": head, "ckpt_path": link,
        "metrics_path": p["metrics"], "metrics_sha256": hashlib.sha256(mb).hexdigest(),
        "stream_sidecar_path": p["stream"], "stream_sidecar_sha256": hashlib.sha256(sb).hexdigest(),
        "canon_sidecar_path": None, "canon_sidecar_sha256": None}, indent=2))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text("starting\n" + ev.sentinel_line(target) + "\nEvaluation complete!\n")
    return p


HEAD = "e" * 40
LINK_C16 = "/farm/C16/epoch=8-step=40000.ckpt"


@pytest.fixture
def cell_env(tmp_path, manifest, hashes):
    cell = _cell(manifest, "exp40_C16_C4_S40000_K8_s43_th90")
    farm, dumps, log = tmp_path / "farm", tmp_path / "dumps", tmp_path / "cell.log"
    p = make_artifacts(cell, farm, dumps, log, HEAD, LINK_C16, "a" * 64, 7)
    return cell, farm, dumps, log, hashes, p


def _verify(cell_env, **over):
    cell, farm, dumps, log, hashes, p = cell_env
    kw = dict(head=HEAD, ckpt_sha="a" * 64, ckpt_bytes=7, link=LINK_C16, target=TARGET)
    kw.update(over)
    r, _ = ev.verify_cell(cell, str(farm), TREE, kw["head"], kw["ckpt_sha"], kw["ckpt_bytes"], kw["link"],
                          str(log), kw["target"], hashes, dump_root=str(dumps))
    return r


def _edit(path, fn):
    d = json.loads(Path(path).read_text())
    fn(d)
    Path(path).write_text(json.dumps(d))


def test_verify_cell_passes(cell_env):
    assert _verify(cell_env) == []


@pytest.mark.parametrize("over", [{"head": "f" * 40}, {"ckpt_sha": "b" * 64}, {"ckpt_bytes": 8},
                                  {"link": "/elsewhere.ckpt"}, {"target": "/x/eval_FLAC.py"}])
def test_verify_cell_binds_identity(cell_env, over):
    assert _verify(cell_env, **over)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(cond_method="fa_invariant"), "cond_method"),
    (lambda d: d.update(cond_autocast="default"), "cond_autocast"),
    (lambda d: d.update(n_samples=6336), "n_samples"),
    (lambda d: d["metrics"].pop("FD"), "metric keys"),
    (lambda d: d["metrics"].update(T60=float("nan")), "finiteness"),
    (lambda d: d["metrics"].update(T60=True), "finiteness"),                  # a bool is not a metric
    (lambda d: d["by_scene"].pop("Cafe"), "by_scene"),
    (lambda d: d["by_scene"]["Cafe"].pop("EDT"), "by_scene Cafe"),
    (lambda d: d["by_scene"]["Office"].update(C50=float("inf")), "by_scene Office"),
    (lambda d: d["by_scene"]["Office"].update(extra=1.0), "by_scene Office"),
    (lambda d: d.update(rotate_deg=90), "rotate_deg"),                     # int, not float: type-strict
    (lambda d: d.update(batch_size=64.0), "batch_size"),
    (lambda d: d.update(extra_key=1), "metrics keys"),
    (lambda d: d.pop("orbit_execution"), "metrics keys"),
    (lambda d: d.update(rotate_mode="fixed"), "metrics keys"),               # a random-only key on a fixed cell
])
def test_verify_cell_metrics_contract(cell_env, mutate, needle):
    _edit(cell_env[5]["metrics"], mutate)
    assert any(needle in r for r in _verify(cell_env)), _verify(cell_env)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["offsets"].__setitem__(5, 0), "offsets"),
    (lambda d: (d["offsets"].__setitem__(5, 0), d["assignment_tuples"][5].__setitem__(2, 0)), "offsets"),
    (lambda d: d["input_tuples"][7][2].pop(), "tuple 7"),                     # K contexts, not K - 1
    (lambda d: d["assignment_tuples"][9].__setitem__(1, "other"), "tuple 9"),
    (lambda d: d["input_tuples"][3].__setitem__(0, 4), "tuple 3"),
    (lambda d: d["offsets"].__setitem__(2, True), "tuple 2"),
    (lambda d: d["offsets"].pop(), "counts"),
    (lambda d: d.update(input_hash="0" * 64), "recompute"),
    (lambda d: d.update(img_w=511), "img_w"),
    (lambda d: d.update(fingerprint_schema=2), "fingerprint_schema"),
    (lambda d: d.update(extra=1), "stream keys"),
])
def test_verify_cell_stream_contract(cell_env, mutate, needle):
    _edit(cell_env[5]["stream"], mutate)
    assert any(needle in r for r in _verify(cell_env)), _verify(cell_env)


def test_verify_cell_input_hash_must_be_the_exp25_one(cell_env):
    cell, farm, dumps, log, hashes, p = cell_env
    bad = dict(hashes)
    bad[(8, 43)] = ("0" * 64, bad[(8, 43)][1])
    r, _ = ev.verify_cell(cell, str(farm), TREE, HEAD, "a" * 64, 7, LINK_C16, str(log), TARGET, bad,
                          dump_root=str(dumps))
    assert any("exp_25 YAW cell" in x for x in r)


def test_random_cell_offsets_must_be_the_eval_seed_draw(tmp_path, manifest, hashes):
    cell = _cell(manifest, "exp40_C16_R_S40000_K8_s44")
    farm, dumps, log = tmp_path / "farm", tmp_path / "dumps", tmp_path / "r.log"
    p = make_artifacts(cell, farm, dumps, log, HEAD, LINK_C16, "a" * 64, 7)
    args = (cell, str(farm), TREE, HEAD, "a" * 64, 7, LINK_C16, str(log), TARGET, hashes)
    assert ev.verify_cell(*args, dump_root=str(dumps))[0] == []
    # a consistent stream with the draw of ANOTHER seed (tuples + hashes recomputed) is still refused
    other = ev.expected_offsets(_rand_cell(45), TREE)

    def swap(d):
        d["offsets"] = list(other)
        for i, t in enumerate(d["assignment_tuples"]):
            t[2] = other[i]
        d["assignment_hash"] = eval_FLAC.canonical_stream_hash(d["assignment_tuples"])
    _edit(p["stream"], swap)
    r = ev.verify_cell(*args, dump_root=str(dumps))[0]
    assert any("eval-seed draw" in x for x in r) and any("assignment_hash" in x for x in r)


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.update(source_sha="0" * 40), "source_sha"),
    (lambda d: d.update(n=6336), "dump manifest n"),
    (lambda d: d.update(shape=[6337, 10241]), "shape"),
    (lambda d: d.update(metrics_stem="x"), "metrics_stem"),
    (lambda d: d.update(canon_sidecar_path="x"), "canon_sidecar_path"),
    (lambda d: d.update(metrics_path="/elsewhere.json"), "metrics_path"),
    (lambda d: d.update(extra=1), "dump manifest keys"),
])
def test_verify_cell_dump_manifest_contract(cell_env, mutate, needle):
    _edit(Path(cell_env[5]["dump"]) / "manifest.json", mutate)
    assert any(needle in r for r in _verify(cell_env)), _verify(cell_env)


def test_verify_cell_dump_bytes_marker_and_sentinel(cell_env):
    cell, farm, dumps, log, hashes, p = cell_env
    d = Path(p["dump"])
    wsha = (d / "COMPLETE").read_text().strip()
    (d / "COMPLETE").write_text(f"done {wsha}\n")                                # a substring is not enough
    assert any("COMPLETE" in r for r in _verify(cell_env))
    (d / "COMPLETE").write_text(wsha + "\n")
    (d / "waveforms.sha256").write_text(f"{wsha} waveforms.npy\n")
    assert any("waveforms.sha256" in r for r in _verify(cell_env))
    (d / "waveforms.sha256").write_text(f"{wsha}  waveforms.npy\n")
    (d / "waveforms.npy").write_bytes(b"tampered")
    assert any("re-hash" in r for r in _verify(cell_env))
    (d / "waveforms.npy").write_bytes(f"wave-{cell['cell']}".encode())
    Path(str(d) + ".staging-123").mkdir()
    assert any("staging" in r for r in _verify(cell_env))
    Path(str(d) + ".staging-123").rmdir()
    assert _verify(cell_env) == []
    old = (f"EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa overlay_sha256={ev.EVAL_PINS[ev.NO_FLASH_OVERLAY]} "
           f"target={TARGET}")                                                  # no launcher sha
    for text in ("no sentinel here\n", old + "\n", ev.sentinel_line(TARGET) + "\n" + ev.sentinel_line(TARGET) + "\n",
                 ev.sentinel_line("/other/eval_FLAC.py") + "\n",
                 ev.sentinel_line(TARGET) + "\nNO-FLASH LAUNCH REFUSED\n"):
        log.write_text(text)
        assert any("sentinel" in r or "refused" in r for r in _verify(cell_env)), text


# --------------------------------------------------------------------------- #
# E0: the committed reference (B9), replay semantics, the strict E0 receipt
# --------------------------------------------------------------------------- #
def test_e0_reference_is_committed_pinned_and_strict(tmp_path):
    ref, sha = ev.e0_reference(TREE)
    assert sha == ev.E0_REFERENCE_SHA256 == kit.sha256_file(str(_REPO / ev.E0_REFERENCE))
    assert sorted(ref["e0"]) == sorted(ev.E0) and len(ref["expected"]) == 10
    h = ev.expected_hashes(ref)
    assert h[(8, 42)] == (ref["expected"]["K8_s42"]["input_hash"], ref["expected"]["K8_s42"]["assignment_hash_random"])
    for e in ref["e0"].values():                     # every pin is a sha256, every path absolute
        for k in ("metrics", "stream", "receipt"):
            assert ev._hex64(e[k]["sha256"]) and os.path.isabs(e[k]["path"])
        assert ev._hex64(e["dump"]["waveforms_sha256"])
    bad = tmp_path / ev.E0_REFERENCE
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({**ref, "e0": {}}))
    with pytest.raises(ev.Refused, match="sha256"):
        ev.e0_reference(str(tmp_path))
    with pytest.raises(ev.Refused, match="schema"):
        ev.e0_reference(str(tmp_path), pinned=kit.sha256_file(str(bad)))


def _fake_ref(tmp_path, rec):
    mb = json.dumps(rec).encode()
    mp = tmp_path / "ref_metrics.json"
    mp.write_bytes(mb)
    name = "exp40_E0_YAW_T_S40000_K1_s42"
    return name, {"e0": {name: {"reference": ev.E0[name], "metrics": {"path": str(mp), "sha256": hashlib.sha256(mb).hexdigest()},
                                "stream": {"sha256": "5" * 64}, "dump": {"waveforms_sha256": "6" * 64}}}}, mp


def test_replay_reasons_semantics_and_reference_byte_check(tmp_path, manifest):
    rec = {"metrics": {"T60": 9.3665}, "by_scene": {"Cafe": {"T60": 1.0}}, "seed": 42, "source_sha": "d" * 40,
           "eval_name": "exp25_YAW_T_S40000_K1_s42", "ckpt_path": "a.ckpt", "n_samples": 6337}
    name, ref, mp = _fake_ref(tmp_path, rec)
    cell = _cell(manifest, name)
    new = {**rec, "source_sha": "e" * 40, "eval_name": name, "ckpt_path": "b.ckpt"}     # the three ignored keys
    d = {"record": new, "stream_sha256": "5" * 64, "waveforms_sha256": "6" * 64}
    assert ev.replay_reasons(cell, d, ref) == []
    for mut in ({"metrics": {"T60": 9.3666}}, {"metrics": {"T60": 9}}, {"by_scene": {"Cafe": {"T60": 1}}},
                {"n_samples": 6336}):
        assert ev.replay_reasons(cell, {**d, "record": {**new, **mut}}, ref)
    assert ev.replay_reasons(cell, {**d, "record": {**new, "extra": 1}}, ref)
    assert any("stream" in r for r in ev.replay_reasons(cell, {**d, "stream_sha256": "7" * 64}, ref))
    assert any("waveforms" in r for r in ev.replay_reasons(cell, {**d, "waveforms_sha256": "7" * 64}, ref))
    mp.write_text(json.dumps({**rec, "seed": 43}))                           # the reference itself moved
    assert any("changed since they were pinned" in r for r in ev.replay_reasons(cell, d, ref))


def _ident(tmp_path, hashes, admission=None, **over):
    i = ev.identity(HEAD, "c" * 64, ev.E0_REFERENCE_SHA256, hashes, str(tmp_path / "repo" / ev.FARM), TREE,
                    dump_root=str(tmp_path / "dumps"), env_digest="d" * 64, e0_sha=None, admission=admission)
    i.update(over)
    return i


def _write_e0(tmp_path, ident, manifest, **over):
    shas = {}
    for c in [c for c in manifest["cells"] if c["block"] == "E0"]:
        rp = Path(ev.cell_paths(c, ident["farm"], TREE, ident["dump_root"])["receipt"])
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"stand-in": c["cell"]}))
        shas[c["cell"]] = kit.sha256_file(str(rp))
    doc = ev.e0_receipt_doc("EQUAL", ident, None, "a" * 64, "b" * 64, shas, [])
    doc.update(over)
    path = ev.e0_receipt_path(str(tmp_path / "repo" / ev.EXPREL / "receipts"), ident["eval_sha"], ident["closure"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kit.atomic_write_json(path, doc)
    return path, doc


def test_e0_gate_is_strict(tmp_path, manifest, hashes):
    ident = _ident(tmp_path, hashes)
    path, doc = _write_e0(tmp_path, ident, manifest)
    r, sha, got = ev.e0_gate_reasons(path, ident)
    assert r == [] and sha == kit.sha256_file(path) and got["env_digest"] == "d" * 64
    assert ev.e0_gate_reasons(path, {**ident, "eval_sha": "b" * 40})[0]
    assert ev.e0_gate_reasons(path, {**ident, "closure": "d" * 64})[0]
    assert ev.e0_gate_reasons(path, {**ident, "env_digest": "9" * 64})[0]           # environment changed
    assert ev.e0_gate_reasons(path, {**ident, "env_digest": None})[0] == []        # verify / collect take E0's
    assert ev.e0_gate_reasons(path, {**ident, "e0_reference_sha": "0" * 64})[0]
    for over in ({"verdict": "PARTIAL"}, {"verdict": "NOT_EQUAL"}, {"cells": sorted(ev.E0)[:2]},
                 {"failures": [["x", ["y"]]]}, {"ckpt_bytes": 1}, {"eval_pins": {}},
                 {"cell_receipts": {k: "0" * 64 for k in ev.E0}}, {"input_hash_map": {}}):
        p2, _ = _write_e0(tmp_path, ident, manifest, **over)
        assert ev.e0_gate_reasons(p2, ident)[0], over
    p3, _ = _write_e0(tmp_path, ident, manifest)
    d = json.loads(Path(p3).read_text())
    d["extra"] = 1
    Path(p3).write_text(json.dumps(d))
    assert ev.e0_gate_reasons(p3, ident)[0]
    p4, _ = _write_e0(tmp_path, ident, manifest)
    c = [c for c in manifest["cells"] if c["block"] == "E0"][0]
    Path(ev.cell_paths(c, ident["farm"], TREE, ident["dump_root"])["receipt"]).write_text("{}")   # receipt moved
    assert any("changed" in x for x in ev.e0_gate_reasons(p4, ident)[0])
    assert ev.e0_gate_reasons(str(tmp_path / "missing.json"), ident)[0]


# --------------------------------------------------------------------------- #
# runtime inputs, admission, AR gate, environment probe
# --------------------------------------------------------------------------- #
def _inputs_file(tmp_path, entries):
    spec = tmp_path / "inputs.json"
    spec.write_text(json.dumps(entries))
    return str(spec)


def _pin(path):
    return {"path": str(path), "sha256": kit.sha256_file(str(path))}


def test_runtime_inputs_are_exact_and_hash_pinned(tmp_path):
    ar = tmp_path / "ar.json"
    ar.write_text("{}")
    adm = tmp_path / "adm.json"
    adm.write_text('{"admitted": true}')
    inputs, sha = ev.runtime_inputs(_inputs_file(tmp_path, {"ar_verify": _pin(ar), "admission": _pin(adm)}), tree=TREE)
    assert inputs["admission"]["path"] == str(adm) and len(sha) == 64
    for bad in ({"admission": _pin(adm)}, {"ar_verify": _pin(ar), "extra": _pin(adm)},
                {"ar_verify": {"path": "ar.json", "sha256": _pin(ar)["sha256"]}},
                {"ar_verify": {**_pin(ar), "note": 1}}, {"ar_verify": {"path": str(ar), "sha256": "x"}}):
        with pytest.raises(ev.Refused):
            ev.runtime_inputs(_inputs_file(tmp_path, bad), tree=TREE)
    inside = _REPO / ev.EXPREL / "exp40_cells.json"
    with pytest.raises(ev.Refused, match="inside the eval worktree"):
        ev.runtime_inputs(_inputs_file(tmp_path, {"ar_verify": _pin(inside)}), tree=TREE)
    spec = _inputs_file(tmp_path, {"ar_verify": _pin(ar), "admission": _pin(adm)})
    adm.write_text('{"admitted": false}')
    with pytest.raises(ev.Refused, match="does not hash"):
        ev.runtime_inputs(spec, tree=TREE)


def _ckpt(tmp_path, arm="C16"):
    with open(_REPO / ev.ARMS[arm]["config"]) as f:
        cfg = json.load(f)
    p = tmp_path / f"src_{arm}.ckpt"
    torch.save({"model_config": cfg, "state_dict": {}}, str(p))
    return str(p), kit.sha256_file(str(p)), os.path.getsize(p)


def _admission_file(tmp_path, ckpt_sha, nbytes, nas_root=None, **over):
    run_id = "exp40_C16_a6000_2x32"
    nas = os.path.join(nas_root or kit.NAS_ROOT, run_id)
    ck = {n: hashlib.sha256(n.encode()).hexdigest() for n in kit.EXPECTED_CKPTS}
    ck[kit.ckpt_name(40000)] = ckpt_sha
    rec = {"kind": "exp40_admission", "admitted": True, "arm": "C16", "run_id": run_id, "platform": "a6000",
           "rung": "2x32", "train_sha": "a" * 40, "runtree_digest": "1" * 64, "env_digest": "2" * 64,
           "nas_dir": nas, "checkpoints": ck, "manifest_sha256": "3" * 64,
           "ckpt_path": os.path.join(nas, kit.ckpt_name(40000)), "sha256": ckpt_sha, "bytes": nbytes,
           "global_step": 40000, "model_config": kit.ARM_CONFIG, "model_config_sha256": kit.PINS[kit.ARM_CONFIG],
           "embedded_config_sha256": "4" * 64, "ema_inventory_sha256": "5" * 64, "strict_load": True,
           "full_evidence_sha256": "6" * 64, "check_sha256": "7" * 64, "audit_sha256": "8" * 64,
           "historical": {}, "current_assets": {}, "trajectory_equivalence_claim_allowed": False, "note": "", "t": 1.0}
    rec.update(over)
    path = tmp_path / "admission.json"
    kit.atomic_write_json(str(path), rec)
    return str(path), rec


def test_admission_goes_through_the_kit_validator(tmp_path):
    src, sha, nbytes = _ckpt(tmp_path)
    nas_root = str(tmp_path / "nas")
    path, rec = _admission_file(tmp_path, sha, nbytes, nas_root=nas_root)
    Path(rec["ckpt_path"]).parent.mkdir(parents=True)
    Path(rec["ckpt_path"]).write_bytes(Path(src).read_bytes())
    got, asha = ev.admission_of({"admission": _pin(path)}, nas_root=nas_root)
    assert got == rec and asha == kit.sha256_file(path)
    assert ev.arm_checkpoint("C16", (got, asha)) == (rec["ckpt_path"], sha, nbytes)
    with pytest.raises(ev.Refused, match="changed after"):
        ev.admission_of({"admission": {"path": path, "sha256": "0" * 64}}, nas_root=nas_root)
    Path(rec["ckpt_path"]).write_bytes(b"other")                               # the NAS checkpoint moved
    with pytest.raises(ev.Refused, match="admission refused"):
        ev.admission_of({"admission": _pin(path)}, nas_root=nas_root)
    path2, _ = _admission_file(tmp_path, sha, nbytes, nas_root=nas_root, arm="YAW")
    with pytest.raises(ev.Refused, match="admission refused"):
        ev.admission_of({"admission": _pin(path2)}, nas_root=nas_root, rehash=None)
    with pytest.raises(ev.Refused, match="runtime-inputs"):
        ev.admission_of({"ar_verify": {}})
    with pytest.raises(ev.Refused, match="runtime-inputs"):
        ev.arm_checkpoint("C16", None)
    assert ev.arm_checkpoint("YAW")[1].startswith("ac1f2603") and ev.arm_checkpoint("P1")[2] == 723927531


def test_ar_gate_requires_the_receipt_and_the_eval_sample(monkeypatch):
    calls = {}
    monkeypatch.setattr(kit, "ar_verify_reasons", lambda path, man, root, now=None: calls.setdefault("v", (path, root)) and [])
    monkeypatch.setattr(kit, "ar_sample_digest", lambda tree, split, root: calls.setdefault("s", split) and kit.AR_EVAL_SAMPLE_DIGEST)
    inputs = {"ar_verify": {"path": "/r/ar.json", "sha256": "0" * 64}}
    assert ev.ar_gate_reasons(TREE, inputs) == []
    assert calls["v"] == ("/r/ar.json", os.path.join(TREE, "AcousticRooms")) and calls["s"] == "data/AR/unseen_eval.json"
    assert ev.ar_gate_reasons(TREE, {})
    monkeypatch.setattr(kit, "ar_sample_digest", lambda tree, split, root: "0" * 64)
    assert any("sample digest" in r for r in ev.ar_gate_reasons(TREE, inputs))
    monkeypatch.setattr(kit, "ar_verify_reasons", lambda *a, **k: ["stale receipt"])
    assert "stale receipt" in ev.ar_gate_reasons(TREE, inputs)


def _measured(py):
    pins = kit.ENV_PINS_A6000
    return {"python": pins["python"], "executable": os.path.realpath(py), "prefix": "/env",
            "dists": {k: v for k, v in pins.items() if k not in ("python", "cuda_runtime", "cudnn", "nccl")},
            "cuda_runtime": pins["cuda_runtime"], "cudnn": pins["cudnn"], "nccl": pins["nccl"],
            "origins": {"torch": "/env/lib/python3.10/site-packages/torch/__init__.py",
                        "flash_attn": "/env/lib/python3.10/site-packages/flash_attn/__init__.py"},
            "transformer_module": os.path.realpath(os.path.join(TREE, "src/models/transformer.py")),
            "flash_attn_func_is_none": False}


def test_eval_environment_reasons():
    py = sys.executable
    m = _measured(py)
    assert ev.eval_environment_reasons(m, TREE, py) == []
    for mut in ({"executable": "/other/python"}, {"python": "3.11.0"}, {"cudnn": 1},
                {"origins": {"torch": "/usr/lib/torch/__init__.py"}}, {"origins": {"torch": None}},
                {"transformer_module": "/x/src/models/transformer.py"},
                {"dists": {**m["dists"], "torch": "2.6.0"}}, {"dists": {**m["dists"], "flash-attn": None}}):
        assert ev.eval_environment_reasons({**m, **mut}, TREE, py), mut


def test_eval_environment_probe_preflight_hf_and_driver(monkeypatch, tmp_path):
    py = sys.executable
    m = _measured(py)
    monkeypatch.setattr(kit, "hf_sealed_reasons", lambda sealed: [])
    monkeypatch.setattr(kit, "hf_resolution_reasons", lambda sealed, py_, env: [])
    state = {"pre": ev.sentinel_line("--preflight"), "drv": "590.48.01\n590.48.01\n"}

    def runner(cmd, cwd=None, env=None):
        if cmd[-1] == "--preflight":
            assert env["PYTHONPATH"] == os.path.join(TREE, "overlays/no_flash") and env["PYTHONDONTWRITEBYTECODE"] == "1"
            return 0, "x\n" + state["pre"] + "\n", ""
        assert cmd[0] == "nvidia-smi"
        return 0, state["drv"], ""

    def measure(tree, py_, env):
        assert "PYTHONPATH" not in env and env["HF_HUB_OFFLINE"] == "1"
        return m
    r, info = ev.eval_environment(TREE, py, str(tmp_path / "hf"), runner=runner, measure=measure)
    assert r == [] and info["env_digest"] == kit.environment_digest(m, "590.48.01") and info["driver"] == "590.48.01"
    state["pre"] = state["pre"].replace(ev.EVAL_PINS[ev.NO_FLASH_LAUNCHER], "0" * 64)
    assert any("preflight" in x for x in ev.eval_environment(TREE, py, str(tmp_path), runner=runner, measure=measure)[0])
    state["pre"] = ev.sentinel_line("--preflight")
    for drv in ("580.1\n", "590.48.01\n580.1\n", ""):
        state["drv"] = drv
        assert any("driver" in x for x in ev.eval_environment(TREE, py, str(tmp_path), runner=runner, measure=measure)[0])
    monkeypatch.setattr(kit, "hf_resolution_reasons", lambda sealed, py_, env: ["resolves outside"])
    state["drv"] = "590.48.01\n"
    assert "resolves outside" in ev.eval_environment(TREE, py, str(tmp_path), runner=runner, measure=measure)[0]


# --------------------------------------------------------------------------- #
# checkpoint cache (B8): true thread concurrency, one winner, identical adoption
# --------------------------------------------------------------------------- #
def test_cache_under_concurrent_threads_has_one_entry_and_one_link(tmp_path):
    src, sha, nbytes = _ckpt(tmp_path)
    farm = tmp_path / "farm"
    barrier, out, errs = threading.Barrier(8), [], []

    def go():
        try:
            barrier.wait()
            out.append(ev.cache_checkpoint("C16", TREE, str(farm), src, sha, nbytes))
        except Exception as err:          # pragma: no cover - reported below
            errs.append(err)
    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errs == [] and len(out) == 8 and len(set(out)) == 1
    cache = farm / ".ckpt_cache"
    assert sorted(p.name for p in cache.iterdir()) == sorted([f"{sha}.ckpt", f"{sha}.lock"])
    link = Path(out[0][0])
    assert os.path.samestat(link.stat(), (cache / f"{sha}.ckpt").stat()) and not link.is_symlink()
    assert (cache / f"{sha}.ckpt").stat().st_mode & 0o222 == 0


def test_cache_refuses_wrong_bytes_config_and_stale_links(tmp_path):
    src, sha, nbytes = _ckpt(tmp_path)
    with pytest.raises(ev.Refused, match="not the pinned"):
        ev.cache_checkpoint("C16", TREE, str(tmp_path / "f1"), src, "0" * 64, nbytes)
    with pytest.raises(ev.Refused, match="not the pinned"):
        ev.cache_checkpoint("C16", TREE, str(tmp_path / "f1"), src, sha, nbytes + 1)
    ysrc, ysha, ybytes = _ckpt(tmp_path, arm="YAW")
    with pytest.raises(ev.Refused, match="embedded model_config"):
        ev.cache_checkpoint("C16", TREE, str(tmp_path / "f2"), ysrc, ysha, ybytes)
    farm = tmp_path / "f3"
    (farm / "C16").mkdir(parents=True)
    (farm / "C16" / "epoch=8-step=40000.ckpt").write_bytes(b"stale")
    with pytest.raises(ev.Refused, match="stale farm entry"):
        ev.cache_checkpoint("C16", TREE, str(farm), src, sha, nbytes)
    assert not [p for p in (farm / ".ckpt_cache").iterdir() if ".tmp-" in p.name]


# --------------------------------------------------------------------------- #
# lanes, locks, owned children
# --------------------------------------------------------------------------- #
def _query(n=2, apps=(), name=None, free=48000):
    def q():
        return ([{"index": i, "uuid": UUIDS[i], "name": name or ev.GPU_NAME, "free_mib": free} for i in range(n)],
                [dict(a) for a in apps])
    return q


def test_lane_state_and_acquire(tmp_path):
    hs = tmp_path / "hs"
    hs.mkdir()
    assert ev.lane_state(0, _query()) == (UUIDS[0], [])
    assert ev.lane_state(1, _query(apps=[{"uuid": UUIDS[1], "pid": 7, "used": "1"}]))[1]
    assert ev.lane_state(0, _query(name="NVIDIA L40"))[1]
    assert ev.lane_state(0, _query(free=100))[1]
    assert ev.lane_state(3, _query())[1]
    got, r = ev.acquire_lane(0, {"x": 1}, _query(), str(hs))
    assert r == [] and got[0] == UUIDS[0] and (hs / f"GPU_LOCK_{UUIDS[0]}").exists()
    assert ev.acquire_lane(0, {"x": 2}, _query(), str(hs))[0] is None             # held by the first
    kit.release_gpu_locks(got[2], got[1])
    seq = iter([_query()(), _query(apps=[{"uuid": UUIDS[0], "pid": 9, "used": "1"}])()])
    got2, r2 = ev.acquire_lane(0, {"x": 3}, lambda: next(seq), str(hs))
    assert got2 is None and r2 and not (hs / f"GPU_LOCK_{UUIDS[0]}").exists()       # re-check failed: released


def test_children_terminate_only_owned_groups_and_stop_spawning(tmp_path):
    ch = ev.Children()
    rc = {}
    t = threading.Thread(target=lambda: rc.setdefault("rc", ch.run(["sleep", "30"], str(tmp_path), dict(os.environ),
                                                                     str(tmp_path / "s.log"))))
    t0 = time.time()
    t.start()
    while not ch._procs and time.time() - t0 < 10:
        time.sleep(0.05)
    (pid,) = list(ch._procs)
    assert os.getpgid(pid) == pid != os.getpgid(0)                                  # its own session
    ch.terminate(grace_s=5.0)
    t.join(10)
    assert rc["rc"] == -15 and time.time() - t0 < 10 and ch.interrupted
    with pytest.raises(ev.Refused, match="stopping"):
        ch.run(["true"], str(tmp_path), dict(os.environ), str(tmp_path / "t.log"))


def test_hold_cards_lock_is_exclusive(tmp_path):
    p = str(tmp_path / "cards.lock")
    fd = ev.hold_cards_lock(p)
    with pytest.raises(ev.Refused, match="holds"):
        ev.hold_cards_lock(p)
    os.close(fd)
    os.close(ev.hold_cards_lock(p))


# --------------------------------------------------------------------------- #
# run_cells with a fake evaluator: receipts, skip, orphans, exceptions, final check (B8, H8)
# --------------------------------------------------------------------------- #
@pytest.fixture
def run_env(tmp_path, hashes, monkeypatch):
    monkeypatch.setattr(ev, "eval_worktree_reasons", lambda *a, **k: [])
    src, sha, nbytes = _ckpt(tmp_path)
    adm_path, adm = _admission_file(tmp_path, sha, nbytes)
    adm = {**adm, "ckpt_path": src}                            # the synthetic NAS copy is the local file
    adm_sha = kit.sha256_file(adm_path)
    ident = _ident(tmp_path, hashes, admission=(adm, adm_sha), e0_sha="f" * 64)
    inputs = {"ar_verify": {"path": "/r/ar.json", "sha256": "b" * 64}, "admission": {"path": adm_path, "sha256": adm_sha}}
    calls = []

    def spawn(ctx, cell, link, lane, log_path):
        calls.append((cell["cell"], lane))
        hs = Path(ctx["handshake_dir"])
        assert (hs / f"GPU_LOCK_{UUIDS[lane]}").exists()              # held for the evaluator's lifetime
        make_artifacts(cell, ident["farm"], ident["dump_root"], log_path, HEAD, link, sha, nbytes)
        return 0
    (tmp_path / "hs").mkdir()
    ctx = {"tree": TREE, "farm": ident["farm"], "py": sys.executable, "hf_sealed": str(tmp_path / "hf"),
           "inputs": inputs, "inputs_sha": "a" * 64, "ident": ident, "ref": None, "logdir": str(tmp_path / "logs"),
           "lockrec_dir": str(tmp_path / "lockrec"), "lanes": [0, 1], "dump_root": ident["dump_root"],
           "query": _query(), "sleep": lambda s: None, "poll_s": 1.0, "max_wait_s": 0.0,
           "handshake_dir": str(tmp_path / "hs"), "children": ev.Children(), "spawn": spawn}
    return ctx, calls


def _c16(manifest, *names):
    return [_cell(manifest, n) for n in names]


def test_run_cells_writes_strict_receipts_that_reverify(run_env, manifest):
    ctx, calls = run_env
    cells = _c16(manifest, "exp40_C16_T_S40000_K1_s42", "exp40_C16_C4_S40000_K8_s43_th90", "exp40_C16_R_S40000_K8_s44")
    assert ev.run_cells(cells, ctx) == []
    assert sorted(c for c, _ in calls) == sorted(c["cell"] for c in cells)
    for c in cells:
        r, rsha = ev.full_cell_reasons(c, ctx["ident"])
        assert r == [] and ev._hex64(rsha)
    hs = Path(ctx["handshake_dir"])
    assert not list(hs.glob("GPU_LOCK_*")) and len(list((hs / "released").iterdir())) == 3
    assert not list(Path(ctx["lockrec_dir"]).iterdir())
    calls.clear()
    assert ev.run_cells(cells, ctx) == [] and calls == []                       # skip only re-verified cells
    rp = Path(ev.cell_paths(cells[0], ctx["farm"], TREE, ctx["dump_root"])["receipt"])
    rp.unlink()
    f = ev.run_cells(cells[:1], ctx)
    assert f and "artifacts exist without a receipt" in f[0][1][0] and calls == []


def test_receipt_validator_rejects_every_identity_mutation(run_env, manifest):
    ctx, _ = run_env
    cell = _c16(manifest, "exp40_C16_T_S40000_K1_s42")[0]
    assert ev.run_cells([cell], ctx) == []
    rec = json.loads(Path(ev.cell_paths(cell, ctx["farm"], TREE, ctx["dump_root"])["receipt"]).read_text())
    ident = ctx["ident"]
    assert ev.receipt_reasons(rec, cell, ident) == []
    muts = [("eval_sha", "b" * 40), ("closure", "0" * 64), ("env_digest", "0" * 64), ("e0_reference_sha256", "0" * 64),
            ("e0_receipt_sha256", None), ("admission_sha256", "0" * 64), ("ckpt_sha256", "0" * 64),
            ("ckpt_bytes", rec["ckpt_bytes"] + 1), ("ckpt_link", "/x.ckpt"), ("target", "/x/eval_FLAC.py"),
            ("lane", True), ("kind", "receipt"), ("schema_version", 1), ("stem", "x"), ("input_hash", "z"),
            ("runtime_inputs", {"admission": rec["runtime_inputs"]["admission"]}),
            ("ar_receipt_sha256", "0" * 64), ("cell", {**cell, "seed": 43})]
    for key, val in muts:
        assert ev.receipt_reasons({**rec, key: val}, cell, ident), key
    assert ev.receipt_reasons({**rec, "extra": 1}, cell, ident)
    assert ev.receipt_reasons({k: v for k, v in rec.items() if k != "log"}, cell, ident)
    assert ev.receipt_reasons(rec, cell, {**ident, "e0_sha": "9" * 64})
    assert ev.receipt_reasons(rec, cell, {**ident, "admission": None})


def test_full_cell_reasons_catch_artifact_drift_after_the_receipt(run_env, manifest):
    ctx, _ = run_env
    cell = _c16(manifest, "exp40_C16_C4_S40000_K8_s43_th90")[0]
    assert ev.run_cells([cell], ctx) == []
    p = ev.cell_paths(cell, ctx["farm"], TREE, ctx["dump_root"])
    rec = json.loads(Path(p["receipt"]).read_text())
    Path(rec["log"]).write_text(Path(rec["log"]).read_text() + "late line\n")       # the log changed
    assert any("log_sha256" in r for r in ev.full_cell_reasons(cell, ctx["ident"])[0])


def test_worker_exceptions_and_failed_verification_are_failures(run_env, manifest):
    ctx, calls = run_env
    cells = _c16(manifest, "exp40_C16_T_S40000_K1_s43", "exp40_C16_T_S40000_K1_s44")
    good = ctx["spawn"]

    def spawn(c2, cell, link, lane, log_path):
        if cell["cell"].endswith("s43"):
            raise RuntimeError("boom")
        return good(c2, cell, link, lane, log_path)
    ctx["spawn"] = spawn
    f = dict(ev.run_cells(cells, ctx))
    assert "worker exception RuntimeError: boom" in f["exp40_C16_T_S40000_K1_s43"][0]
    assert "exp40_C16_T_S40000_K1_s44" not in f
    assert not list(Path(ctx["handshake_dir"]).glob("GPU_LOCK_*"))                 # released despite the raise
    ctx["spawn"] = lambda *a: 0                                                     # rc 0, nothing written
    f2 = ev.run_cells(_c16(manifest, "exp40_C16_T_S40000_K1_s45"), ctx)
    assert f2 and "artifacts unreadable" in " ".join(f2[0][1])
    ctx["spawn"] = lambda *a: 3
    f3 = ev.run_cells(_c16(manifest, "exp40_C16_T_S40000_K1_s46"), ctx)
    assert f3 == [("exp40_C16_T_S40000_K1_s46", ["eval rc 3"])]


def test_busy_lane_waits_then_fails_without_spawning(run_env, manifest):
    ctx, calls = run_env
    ctx.update(lanes=[0], query=_query(apps=[{"uuid": UUIDS[0], "pid": 99, "used": "500"}]), max_wait_s=3.0)
    slept = []
    ctx["sleep"] = slept.append
    f = ev.run_cells(_c16(manifest, "exp40_C16_T_S40000_K8_s42"), ctx)
    assert f and "not free" in f[0][1][0] and calls == [] and slept == [1.0, 1.0, 1.0]


def test_worktree_drift_stops_every_lane(run_env, manifest, monkeypatch):
    ctx, calls = run_env
    monkeypatch.setattr(ev, "eval_worktree_reasons", lambda *a, **k: ["eval worktree not clean: '?? x'"])
    f = ev.run_cells(_c16(manifest, "exp40_C16_T_S40000_K8_s43", "exp40_C16_T_S40000_K8_s44"), ctx)
    assert f and any("drifted" in rs[0] for _, rs in f) and calls == [] and ctx["children"].stop.is_set()
    assert not list(Path(ctx["handshake_dir"]).glob("GPU_LOCK_*"))


# --------------------------------------------------------------------------- #
# comparator registry (synthetic sources; the authoritative validator stubbed)
# --------------------------------------------------------------------------- #
def _sources(tmp_path, monkeypatch, drop=None, nodump=None, refuse=None):
    ref, _ = ev.e0_reference(TREE)
    h = ev.expected_hashes(ref)
    dumps = tmp_path / "compdumps"
    monkeypatch.setattr(ev, "COMP_DUMP_ROOT", str(dumps))
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = {s: {**v, "base_dir": str(tmp_path / s)} for s, v in ev.SOURCES.items()}
    verdicts = {}
    for key in ev.EXP36_VERDICTS:
        vp = tmp_path / f"verdict_{key[0]}_{key[1]}.json"
        vp.write_text(json.dumps({"check": "exp36_replay", "result": "equal", "pin": ev.SOURCES["exp36"]["pin"]}))
        verdicts[key] = (str(vp), kit.sha256_file(str(vp)))
    for i, w in enumerate(ev.registry_want()):
        if w["cell"] == drop:
            continue
        farm = Path(sources[w["source"]]["base_dir"]) / ev.COMP_FARM_ROOT / w["arm"]
        farm.mkdir(parents=True, exist_ok=True)
        stem = f"epoch=8-step=40000_metrics_1_1.0_{w['cell']}_{i}"
        rec = {k: None for k in ev.METRICS_BASE_KEYS}
        rec.update(source_sha=sources[w["source"]]["pin"], cond_method="vanilla", seed=w["seed"],
                   rotate_deg=w["rotate_deg"], ckpt_sha256_loaded=ev.ARMS[w["arm"]]["ckpt_sha256"],
                   ckpt_bytes_loaded=ev.ARMS[w["arm"]]["ckpt_bytes"], n_samples=6337,
                   dataset_config=ev.DATASET_OF_K[w["K"]], eval_name=w["cell"], metrics=_metrics_block(3.0 + i / 100))
        if w["rotate_mode"] == "random":
            rec.update({k: 0 for k in ev.METRICS_RANDOM_KEYS})
        mb = json.dumps(rec).encode()
        st = {"input_hash": h[(w["K"], w["seed"])][0],
              "assignment_hash": h[(w["K"], w["seed"])][1] if w["rotate_mode"] == "random" else "a" * 64}
        sb = json.dumps(st).encode()
        (farm / f"{stem}.json").write_bytes(mb)
        (farm / f"{stem}.stream.json").write_bytes(sb)
        (farm / f"{stem}.cell.json").write_text(json.dumps({
            "cell": w["cell"], "stem": stem, "arm": w["arm"], "dump": w["cell"] != nodump, "dump_root": str(dumps),
            "ckpt_sha256": ev.ARMS[w["arm"]]["ckpt_sha256"], "model_config": ev.ARMS[w["arm"]]["config"],
            "model_config_sha256": ev.ARMS[w["arm"]]["config_sha256"], "dataset_config": ev.DATASET_OF_K[w["K"]],
            "dataset_config_sha256": ev.EVAL_PINS[ev.DATASET_OF_K[w["K"]]],
            **({"expect_head": ev.SOURCES["exp36"]["pin"]} if w["source"] == "exp36" else {})}))
        d = dumps / stem
        d.mkdir(parents=True)
        npy = f"npy{i}".encode()
        wsha = hashlib.sha256(npy).hexdigest()
        (d / "waveforms.npy").write_bytes(npy)
        (d / "waveforms.sha256").write_text(f"{wsha}  waveforms.npy\n")
        (d / "COMPLETE").write_text(wsha + "\n")
        (d / "manifest.json").write_text(json.dumps({"sha256": wsha, "metrics_sha256": hashlib.sha256(mb).hexdigest(),
                                                     "stream_sidecar_sha256": hashlib.sha256(sb).hexdigest(),
                                                     "metrics_stem": stem, "n": 6337}))

    def validate(src, stems):
        return {s: (["authoritative refusal"] if refuse and refuse in s else []) for s in stems}
    return sources, verdicts, validate


def test_registry_pins_exactly_the_registered_mix(tmp_path, monkeypatch):
    sources, verdicts, validate = _sources(tmp_path, monkeypatch)
    reg = ev.register_comparators(TREE, sources=sources, validate=validate, verdicts=verdicts)
    assert reg["count"] == 92 == 2 * (2 * 5 * 2 + 2 * 4 + 2 * 3) + 2 * 4 * 3
    assert reg["exp36_pin"] == ev.SOURCES["exp36"]["pin"] and reg["e0_reference_sha256"] == ev.E0_REFERENCE_SHA256
    assert all(set(e) == ev.REG_CELL_KEYS for e in reg["cells"])
    assert all((e["replay_verdict"] is None) == (e["source"] == "exp25") for e in reg["cells"])
    assert ev.registry_reasons(reg, ev.E0_REFERENCE_SHA256, sources=sources) == []
    e = reg["cells"][5]
    for path in (e["metrics_path"], os.path.join(e["dump_dir"], "waveforms.npy"), e["receipt_path"]):
        old = Path(path).read_bytes()
        Path(path).write_bytes(old + b" ")
        assert ev.registry_reasons(reg, ev.E0_REFERENCE_SHA256, sources=sources), path
        Path(path).write_bytes(old)
    assert ev.registry_reasons({**reg, "cells": reg["cells"][:-1]}, ev.E0_REFERENCE_SHA256, sources=sources)
    assert ev.registry_reasons({**reg, "cells": reg["cells"] + reg["cells"][:1], "count": 73},
                               ev.E0_REFERENCE_SHA256, sources=sources)
    assert ev.registry_reasons(reg, "0" * 64, sources=sources)
    assert ev.registry_reasons({**reg, "exp36_pin": "0" * 40}, ev.E0_REFERENCE_SHA256, sources=sources)


def test_registry_refusals(tmp_path, monkeypatch):
    s, v, val = _sources(tmp_path / "a", monkeypatch, drop="exp25_P1_T_S40000_K8_s44")
    with pytest.raises(ev.Refused, match="0 receipts"):
        ev.register_comparators(TREE, sources=s, validate=val, verdicts=v)
    s, v, val = _sources(tmp_path / "b", monkeypatch, nodump="exp25_YAW_C4_S40000_K1_s42_th180")
    with pytest.raises(ev.Refused, match="receipt does not match"):
        ev.register_comparators(TREE, sources=s, validate=val, verdicts=v)
    s, v, val = _sources(tmp_path / "c", monkeypatch, refuse="exp35_P1_OFF_S40000_K8_s42_th225")
    with pytest.raises(ev.Refused, match="authoritative validator refused"):
        ev.register_comparators(TREE, sources=s, validate=val, verdicts=v)
    s, v, val = _sources(tmp_path / "d", monkeypatch)
    v = {**v, ("A", "YAW"): (v[("A", "YAW")][0], "0" * 64)}
    with pytest.raises(ev.Refused, match="replay verdict"):
        ev.register_comparators(TREE, sources=s, validate=val, verdicts=v)


def test_authoritative_validator_pins_are_the_source_blobs():
    for src, s in ev.SOURCES.items():
        assert kit.git(TREE, "rev-parse", f"{s['pin']}:{ev.VALIDATOR}").strip() == s["validator_blob"]


# --------------------------------------------------------------------------- #
# tables, paper import, frozen CSV (H12)
# --------------------------------------------------------------------------- #
def _row(arm, block, k, seed, mode, deg, v=1.0):
    r = {"cell": f"{arm}_{block}_{k}_{seed}_{deg}", "arm": arm, "block": block, "K": k, "seed": seed,
         "rotate_mode": mode, "rotate_deg": deg, "metrics_path": f"/m/{arm}_{k}_{seed}_{deg}.json",
         "metrics_sha256": "1" * 64, "stream_sha256": "2" * 64, "receipt_sha256": "3" * 64,
         "source_sha": "e" * 40, "origin": "exp40"}
    r.update({m: v + seed / 1000 for m in ev.METRIC_KEYS})
    return r


def _all_rows():
    rows = []
    for arm in ("C16", "YAW", "P1"):
        for k in ev.KS:
            rows += [_row(arm, "T", k, s, "fixed", 0.0) for s in ev.SEEDS]
            rows += [_row(arm, "R", k, s, "random", None) for s in ev.SEEDS]
            c4_seeds = (42,) if arm == "P1" else ev.SEEDS
            rows += [_row(arm, "C4", k, s, "fixed", float(d)) for d in ev.C4 for s in c4_seeds]
            rows += [_row(arm, "G45", k, 42, "fixed", float(d)) for d in ev.G45]
            rows += [_row(arm, "OFF" if arm == "C16" else "FILL", k, 42, "fixed", d, 1.5) for d in ev.OFF]
    return rows


def test_tables_render_every_block_and_pending():
    rows = _all_rows()
    text = ev.render_tables(rows, [])
    for title, _, _ in ev.TABLES:
        assert f"## {title}" in text
    assert "PENDING" not in text and "d P1" in text
    assert "| P1 | 8 | 90 | 1 |" in text and "| YAW | 8 | 90 | 5 |" in text and "| C16 | 1 | 11.25 | 1 |" in text
    rows2 = [r for r in rows if not (r["arm"] == "YAW" and r["block"] == "C4" and r["seed"] == 44 and r["rotate_deg"] == 180.0)]
    assert "| YAW | 1 | 180 | 4 | PENDING" in ev.render_tables(rows2, [])


def test_paper_import_csv_has_the_paper_archive_schema():
    text = ev.render_paper_import(_all_rows())
    lines = text.splitlines()
    assert lines[0] == ("method,K,angle_deg,seed,source_json,T60,C50,EDT,FD,RIR_to_GT_RIR_R@1,RIR_to_GT_RIR_R@5,"
                        "RIR_to_GT_RIR_R@10,RIR_to_geom_R@1,RIR_to_geom_R@5,RIR_to_geom_R@10")
    paper = Path("/home/yixunhu/codespace/-Neurips-2026-workshop-Sound-Localization/fig_inputs/appendix/yaw_c4_off_orbit_raw.csv")
    if paper.exists():
        assert lines[0] == paper.read_text().splitlines()[0]
    assert all(",random," not in ln for ln in lines) and any(",11.25,42," in ln for ln in lines)
    assert any(ln.startswith("YawAug C16@40k (exp_40),8,90,43,") for ln in lines)


def test_collect_outputs_freeze_and_the_figure_refuses_a_stale_csv(tmp_path):
    rows = _all_rows()
    prov = {"eval_sha": HEAD}
    paths = ev.write_collect(rows, [], prov, str(tmp_path))
    assert [os.path.basename(p) for p in paths] == [ev.OUT_CSV, ev.OUT_TABLES, ev.OUT_PAPER, ev.OUT_COLLECT]
    parsed, csv_sha, _ = ev.frozen_rows(rows, prov, str(tmp_path))
    assert len(parsed) == len(rows) and csv_sha == kit.sha256_file(str(tmp_path / ev.OUT_CSV))
    assert sorted(ev.render_csv(parsed).splitlines()) == sorted(ev.render_csv(rows).splitlines())
    with pytest.raises(ev.Refused, match="provenance"):
        ev.frozen_rows(rows, {"eval_sha": "f" * 40}, str(tmp_path))
    with pytest.raises(ev.Refused, match="stale"):
        ev.frozen_rows(rows[1:], prov, str(tmp_path))
    contract = json.loads((_EXP / "exp40_figure_contract.json").read_text())
    sel, reasons = ev.figure_selection(parsed, contract)
    assert reasons == [] and len(sel) == len(contract["cells"])
    assert ev.figure_reasons([r for r in parsed if not (r["arm"] == "P1" and r["rotate_deg"] == 45.0)], contract)
    dup = parsed + [dict(parsed[[i for i, r in enumerate(parsed) if r["rotate_deg"] == 11.25][0]])]
    assert any("duplicate" in r for r in ev.figure_reasons(dup, contract))


def test_figure_contract_covers_every_arm_angle_and_k():
    c = json.loads((_EXP / "exp40_figure_contract.json").read_text())
    assert c["arms"] == ["C16", "YAW", "P1"] and c["Ks"] == [1, 8]
    assert len(c["cells"]) == 3 * 2 * len(c["angles"])
    rows = [dict(e, **{m: 1.0 for m in ev.METRIC_KEYS}) for e in c["cells"]]
    assert ev.figure_reasons(rows, c) == []
    assert ev.figure_reasons(rows[1:], c)                              # a missing cell
    assert ev.figure_reasons(rows + rows[:1], c)                       # a duplicate


# --------------------------------------------------------------------------- #
# collect end-to-end, the table evidence and the offline row validator (B10, M7)
# --------------------------------------------------------------------------- #
def test_collect_freeze_and_validate_row_end_to_end(tmp_path, monkeypatch, manifest, hashes, run_env):
    ctx, _ = run_env
    sources, verdicts, validate = _sources(tmp_path / "src", monkeypatch)
    reg = ev.register_comparators(TREE, sources=sources, validate=validate, verdicts=verdicts)
    reg_path = tmp_path / "comparators.json"
    kit.atomic_write_json(str(reg_path), reg)
    monkeypatch.setattr(ev, "SOURCES", sources)
    monkeypatch.setattr(ev, "tree_identity_reasons", lambda *a: [])
    monkeypatch.setattr(ev, "closure_digest", lambda *a: "c" * 64)
    monkeypatch.setattr(ev, "expected_hashes", lambda ref: hashes)
    monkeypatch.setattr(ev, "DUMP_ROOT", ctx["dump_root"])
    ident = ctx["ident"]
    repo = Path(ident["farm"]).parents[2]
    e0_path, _ = _write_e0(repo.parent, {**ident, "env_digest": "d" * 64}, manifest)
    ident["e0_sha"] = kit.sha256_file(e0_path)
    t_cells = [c for c in manifest["cells"] if c["block"] == "T"]
    assert ev.run_cells(t_cells + _c16(manifest, "exp40_C16_R_S40000_K1_s42"), ctx) == []
    adm_path = ctx["inputs"]["admission"]["path"]
    adm_rec = json.loads(Path(adm_path).read_text())
    assert ident["admission"][1] == kit.sha256_file(adm_path) and adm_rec["sha256"] == ident["admission"][0]["sha256"]
    rows, pending, prov = ev.collect_rows(str(repo), TREE, HEAD, adm_path, registry_path=str(reg_path))
    exp40 = [r for r in rows if r["origin"] == "exp40"]
    assert len(exp40) == 11 and len(rows) == 11 + 92 and len(pending) == 85 - 3 - 11
    assert prov["e0_receipt_sha256"] == ident["e0_sha"] and prov["admission_sha256"] == ident["admission"][1]
    ev_doc = ev.freeze_table_evidence(str(repo), TREE, HEAD, adm_path, registry_path=str(reg_path))
    evp = tmp_path / "table_evidence.json"
    kit.atomic_write_json(str(evp), ev_doc)
    files = [ev.cell_paths(c, ident["farm"], TREE, ctx["dump_root"])["metrics"] for c in t_cells if c["K"] == 1]
    assert ev.validate_row(files, str(repo), TREE, evidence_path=str(evp)) == (True, [])
    assert not ev.validate_row(files[:4], str(repo), TREE, evidence_path=str(evp))[0]
    assert not ev.validate_row(files + files[:1], str(repo), TREE, evidence_path=str(evp))[0]
    k8 = [ev.cell_paths(c, ident["farm"], TREE, ctx["dump_root"])["metrics"] for c in t_cells if c["K"] == 8]
    assert not ev.validate_row(files[:4] + k8[:1], str(repo), TREE, evidence_path=str(evp))[0]
    copy_ = tmp_path / os.path.basename(files[0])
    copy_.write_bytes(Path(files[0]).read_bytes())
    assert not ev.validate_row([str(copy_)] + files[1:], str(repo), TREE, evidence_path=str(evp))[0]
    for path in (files[2], files[2][:-len(".json")] + ".exp40_receipt.json", files[2][:-len(".json")] + ".stream.json",
                 adm_path, e0_path):
        old = Path(path).read_bytes()
        Path(path).write_bytes(old + b" ")
        assert not ev.validate_row(files, str(repo), TREE, evidence_path=str(evp))[0], path
        Path(path).write_bytes(old)
    assert ev.validate_row(files, str(repo), TREE, evidence_path=str(evp)) == (True, [])
    assert not ev.validate_row(files, str(repo), TREE, evidence_path=str(tmp_path / "none.json"))[0]
    # collect refuses a cell whose receipt no longer re-verifies
    Path(files[1][:-len(".json")] + ".stream.json").write_text("{}")
    with pytest.raises(ev.Refused, match="does not re-verify"):
        ev.collect_rows(str(repo), TREE, HEAD, adm_path, registry_path=str(reg_path))


def test_validate_row_refuses_unregistered_files(tmp_path):
    f = tmp_path / "not_ours_metrics.json"
    f.write_text("{}")
    ok, problems = ev.validate_row([str(f)], str(tmp_path), TREE, evidence_path=str(tmp_path / "missing.json"))
    assert not ok and "table evidence" in problems[0]


# --------------------------------------------------------------------------- #
# model_comparison.md contract exp40 (plan §5.4, M7)
# --------------------------------------------------------------------------- #
def test_generator_registers_two_exp40_rows_and_requires_the_five_canonical_files(tmp_path):
    gen = _load("gen_model_comparison_exp40", _REPO / "worklog/worklog_yixun/gen_model_comparison.py")
    rows = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp40"]
    assert sorted(r[2] for r in rows) == [1, 8] and all(r[0] == gen.EXP40_LABEL for r in rows)
    assert all("exp40_C16_T_S40000_K" in r[3][0] for r in rows)
    assert gen.validate_exp40_cell([]) == (True, [])
    canon = gen.exp40_canonical_files(str(tmp_path), 1)
    for p in canon:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("{}")
    assert [os.path.basename(p)[-8:] for p in canon] == [f"s{s}.json"[-8:] for s in range(42, 47)]
    for files, k in ((canon[:4], 1), (canon + canon[:1], 1), (canon, 8), (canon, None)):
        ok, problems = gen.validate_exp40_cell(files, repo_root=str(tmp_path), expected_k=k)
        assert not ok and problems
    dup = tmp_path / "elsewhere" / os.path.basename(canon[0])
    dup.parent.mkdir()
    dup.write_text("{}")
    assert not gen.validate_exp40_cell([str(dup)] + canon[1:], repo_root=str(tmp_path), expected_k=1)[0]
    ok, problems = gen.validate_exp40_cell(canon, repo_root=str(tmp_path), expected_k=1)
    assert not ok and "cannot run the exp_40 validator" in problems[0]          # names pass; the validator gates


# --------------------------------------------------------------------------- #
# the run / verify CLI end-to-end (E0 first, its receipt, the gate, the rest, verify)
# --------------------------------------------------------------------------- #
def test_run_cli_e0_then_rest_then_verify(tmp_path, monkeypatch, manifest, hashes):
    ysrc, ysha, ybytes = _ckpt(tmp_path, arm="YAW")
    monkeypatch.setitem(ev.ARMS, "YAW", {**ev.ARMS["YAW"], "ckpt": ysrc, "ckpt_sha256": ysha, "ckpt_bytes": ybytes})
    csrc, csha, cbytes = _ckpt(tmp_path, arm="C16")
    adm_path, adm = _admission_file(tmp_path, csha, cbytes)
    adm = {**adm, "ckpt_path": csrc}
    ar = tmp_path / "ar_verify.json"
    ar.write_text("{}")
    inputs = _inputs_file(tmp_path, {"ar_verify": _pin(ar), "admission": _pin(adm_path)})
    monkeypatch.setattr(ev, "eval_worktree_reasons", lambda *a, **k: [])
    monkeypatch.setattr(ev, "tree_identity_reasons", lambda *a, **k: [])
    monkeypatch.setattr(ev, "ar_gate_reasons", lambda *a, **k: [])
    monkeypatch.setattr(ev, "eval_environment", lambda *a, **k: ([], {"env_digest": "d" * 64}))
    monkeypatch.setattr(ev, "closure_digest", lambda *a: "c" * 64)
    monkeypatch.setattr(ev, "expected_hashes", lambda ref: hashes)
    monkeypatch.setattr(ev, "admission_of", lambda inp, nas_root=None, rehash="final": (adm, kit.sha256_file(adm_path)))
    monkeypatch.setattr(ev, "replay_reasons", lambda cell, d, ref: [])
    monkeypatch.setattr(ev, "DUMP_ROOT", str(tmp_path / "dumps"))
    monkeypatch.setattr(kit, "CARDS_LOCK", str(tmp_path / "cards.lock"))
    monkeypatch.setattr(kit, "HANDSHAKE_DIR", str(tmp_path / "hs"))
    monkeypatch.setattr(kit, "query_gpus", _query())
    (tmp_path / "hs").mkdir()
    repo = tmp_path / "repo"
    spawned = []

    def spawn(ctx, cell, link, lane, log_path):
        spawned.append(cell["cell"])
        sha, nb = (ysha, ybytes) if cell["arm"] == "YAW" else (csha, cbytes)
        make_artifacts(cell, ctx["farm"], ctx["dump_root"], log_path, HEAD, link, sha, nb)
        return 0
    monkeypatch.setattr(ev, "default_spawn", spawn)
    base = ["--repo", str(repo), "--eval-worktree", TREE, "--eval-sha", HEAD, "--runtime-inputs", inputs]
    run = base + ["--hf-sealed", str(tmp_path / "hf"), "--lanes", "0,1"]
    c16 = "exp40_C16_T_S40000_K1_s42,exp40_C16_R_S40000_K8_s43"
    assert ev.main(["run"] + run + ["--only", c16]) == 2                         # no E0 receipt yet: refused
    assert spawned == []
    assert ev.main(["run"] + run + ["--blocks", "E0"]) == 0
    e0p = ev.e0_receipt_path(str(repo / ev.EXPREL / "receipts"), HEAD, "c" * 64)
    e0 = json.loads(Path(e0p).read_text())
    assert e0["verdict"] == "EQUAL" and e0["env_digest"] == "d" * 64 and sorted(e0["cell_receipts"]) == sorted(ev.E0)
    assert ev.main(["run"] + run + ["--only", c16]) == 0
    assert sorted(spawned) == sorted(list(ev.E0) + c16.split(","))
    rec = json.loads(Path(ev.cell_paths(_cell(manifest, "exp40_C16_T_S40000_K1_s42"), str(repo / ev.FARM), TREE,
                                        str(tmp_path / "dumps"))["receipt"]).read_text())
    assert rec["e0_receipt_sha256"] == kit.sha256_file(e0p) and rec["admission_sha256"] == kit.sha256_file(adm_path)
    assert ev.main(["run"] + run + ["--only", c16]) == 0 and len(spawned) == 5      # skip: re-verified, not re-run
    assert ev.main(["verify"] + base + ["--only", c16 + ",exp40_E0_YAW_T_S40000_K1_s42"]) == 0
    assert ev.main(["verify"] + base + ["--only", "exp40_C16_T_S40000_K8_s42"]) == 1  # never run: missing
    s = ev.cell_paths(_cell(manifest, "exp40_C16_R_S40000_K8_s43"), str(repo / ev.FARM), TREE, str(tmp_path / "dumps"))
    Path(s["stream"]).write_text("{}")
    assert ev.main(["verify"] + base + ["--only", c16]) == 1
    monkeypatch.setattr(ev, "eval_environment", lambda *a, **k: ([], {"env_digest": "9" * 64}))
    assert ev.main(["run"] + run + ["--only", "exp40_C16_T_S40000_K8_s42"]) == 2  # the environment changed since E0
    fd = ev.hold_cards_lock(str(tmp_path / "cards.lock"))
    monkeypatch.setattr(ev, "eval_environment", lambda *a, **k: ([], {"env_digest": "d" * 64}))
    assert ev.main(["run"] + run + ["--only", "exp40_C16_T_S40000_K8_s42"]) == 2  # another exp_40 user holds the cards
    os.close(fd)
    assert ev.main(["run"] + run + ["--dry-run", "--only", "exp40_C16_T_S40000_K8_s42"]) == 0
    assert len(spawned) == 5
