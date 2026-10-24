"""exp_25 depth_canon round 3b -- the pre-FULL tie audit
(``worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py``,
plan_depth_canon.md v8 §7 rung 3).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §7/§8.

CPU only, no dataset needed: synthetic scalar maps come from ``test_depth_canon``'s
seeded helper, split JSONs are written in the real ``{scene: {scene_id: [file, ...]}}``
schema of ``data/AR/*.json`` into a tmp tree.
"""
import importlib.util
import os
import sys
import zlib

import numpy as np
import pytest
import torch

from src.data import yaw_rotation as yr
from test_depth_canon import _ar_md, _scalar_depth

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AUDIT_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_25_depth_canon_claude",
                         "depth_canon_tie_audit.py")
W = 512


def _load_audit():
    spec = importlib.util.spec_from_file_location("depth_canon_tie_audit", _AUDIT_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # importable by name: Pool pickles _task by reference
    spec.loader.exec_module(module)
    return module


audit = _load_audit()

# (name, peaks, period, expected raw tie class or None for a survivor)
FIXTURES = [
    ("unique", [(40, 300)], None, "none"),
    ("same_column", [(10, 77), (200, 77)], None, "same_column"),
    ("row_tie_sep1", [(90, 5), (90, 6)], None, "cross_column_image"),
    ("row_tie_wrap64", [(120, 480), (120, 32)], None, "cross_column_image"),
    ("row_tie_sep256", [(7, 100), (7, 356)], None, "cross_column_image"),
    ("cross_row_tie", [(30, 50), (220, 400)], None, "cross_column_image"),
    ("periodic_256", [(60, 11)], 256, None),
    ("periodic_128", [(60, 11)], 128, None),
]


def _fixture(name, peaks, period, seed=3):
    return _scalar_depth(256, W, seed=seed, period=period, peaks=peaks)


# --------------------------------------------------------------------------- #
# raw-side selector (pure numpy, independent of the proxy implementation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, peaks, period, want", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_raw_select_agrees_with_the_operator_selector(name, peaks, period, want):
    d = _fixture(name, peaks, period)
    j, tie_class, cands, survivors = audit.raw_select(d.numpy())
    P = d.max(dim=0).values
    assert cands == torch.nonzero(P == P.max()).flatten().tolist()
    if want is None:
        with pytest.raises(yr.StabilizedTieError) as err:
            yr.canonical_column(d)
        assert (j, tie_class) == (None, None) and tuple(survivors) == err.value.candidates
    else:
        assert (j, tie_class) == yr.canonical_column(d) and tie_class == want
        assert survivors == [j]


@pytest.mark.parametrize("roll", [1, 64, 255, 256, 511])
@pytest.mark.parametrize("name, peaks, period, want", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_raw_select_co_rotates(name, peaks, period, want, roll):
    raw = _fixture(name, peaks, period).numpy()
    j, tie_class, cands, survivors = audit.raw_select(raw)
    j2, tie_class2, cands2, survivors2 = audit.raw_select(np.roll(raw, roll, axis=1))
    assert tie_class2 == tie_class
    assert sorted(cands2) == sorted((c + roll) % W for c in cands)
    assert sorted(survivors2) == sorted((s + roll) % W for s in survivors)
    assert j2 == (None if j is None else (j + roll) % W)


def test_raw_select_uses_the_rooted_image_not_the_first_index():
    """A cross-column tie whose rooted-image winner is the LARGER column index."""
    for seed in range(20):
        raw = _fixture("t", [(9, 20), (9, 300)], None, seed=seed).numpy()
        j, tie_class, cands, _ = audit.raw_select(raw)
        if j == 300:
            assert cands == [20, 300] and tie_class == "cross_column_image"
            return
    pytest.fail("no seed gave a non-first-index winner")


@pytest.mark.parametrize("bad", [np.zeros((4,)), np.zeros((2, 3, 4)), np.zeros((0, 4)),
                                 np.zeros((4, 4), dtype=np.int64),
                                 np.array([[1.0, np.nan], [0.0, 1.0]]), [[1.0, 2.0]]])
def test_raw_select_refuses_malformed_maps(bad):
    with pytest.raises(ValueError):
        audit.raw_select(bad)


@pytest.mark.parametrize("cands, want", [([300], 0.0), ([0, 256], 180.0), ([480, 32], 45.0),
                                         ([1, 2, 511], 3 * 360.0 / W), ([5, 6], 360.0 / W)])
def test_max_circular_separation_in_degrees(cands, want):
    assert audit.max_circular_sep_deg(cands, W) == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- #
# split item -> depth map: the loader's own path logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fname, receiver", [("S001_R0044_hybrid_IR.wav", 44),
                                             ("S0071_R000_hybrid_IR.wav", 0),
                                             ("S009_R001_hybrid_IR.wav", 1)])
def test_resolve_depth_path_is_the_path_ar_md_loads(tmp_path, monkeypatch, fname, receiver):
    root = str(tmp_path / "AcousticRooms")
    audio = os.path.join(root, "single_channel_ir_1", "Cafe", "Cafe_idx_3", fname)
    loaded = []

    def spy(path, *a, **k):
        loaded.append(path)
        return np.ones((256, 512))

    monkeypatch.setattr(np, "load", spy)
    info = {"path": audio, "relpath": os.path.relpath(audio, root),
            "modalities": audit.DEPTH_ONLY_MODALITIES}
    md = _ar_md().get_custom_metadata(info, None)
    assert set(md) == {"scene", "depth"} and loaded and len(loaded) == 1
    path, key = audit.resolve_depth_path(audio, root)
    assert path == loaded[0]
    assert key == os.path.join("depth_map", "Cafe", "Cafe_idx_3", f"{receiver}.npy")
    assert os.path.normpath(path) == os.path.join(root, key)


# --------------------------------------------------------------------------- #
# split JSONs (real schema) -> unique maps, with the inputs ledger
# --------------------------------------------------------------------------- #
_PROD_TRAIN_CFG = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR", "train",
                               "acousticroom_train.json")
SPLITS = {
    "train": {"Cafe": {"Cafe_idx_0": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav",
                                      "S001_R0012_hybrid_IR.wav"],
                       "Cafe_idx_1": ["S003_R001_hybrid_IR.wav"]}},
    "unseen": {"Office": {"Office_idx_2": ["S0010_R005_hybrid_IR.wav", "S004_R006_hybrid_IR.wav"]}},
    "seen": {"Cafe": {"Cafe_idx_0": ["S009_R001_hybrid_IR.wav", "S0010_R003_hybrid_IR.wav"]}},
}


def _write_tree(tmp_path, splits=SPLITS, maps=None):
    """A tmp AcousticRooms tree: split JSONs + dataset configs in the production schema
    (the real train config with only the paths replaced) + 256x512 maps per receiver."""
    import json
    root = tmp_path / "AcousticRooms"
    template = json.load(open(_PROD_TRAIN_CFG))
    cfgs = {}
    for name, split in splits.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(split))
        cfg = json.loads(json.dumps(template))
        cfg["datasets"][0].update(path=str(root), json_file_path=str(tmp_path / f"{name}.json"),
                                  custom_metadata_module=_ar_md().__file__)
        (tmp_path / f"{name}_cfg.json").write_text(json.dumps(cfg, indent=4))
        cfgs[name] = str(tmp_path / f"{name}_cfg.json")
        for scene, rooms in split.items():
            for sid, files in rooms.items():
                for f in files:
                    key = audit.resolve_depth_path(str(root / "x" / scene / sid / f), str(root))[1]
                    p = root / key
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if not p.exists():
                        arr = (maps or {}).get(key)
                        if arr is None:
                            arr = _scalar_depth(256, W, seed=zlib.crc32(key.encode()) % 1000).numpy()
                        np.save(p, arr)
    return cfgs


def test_collect_maps_dedupes_by_resolved_path_and_counts_items(tmp_path):
    cfgs = _write_tree(tmp_path)
    splits = {n: audit.read_split_config(p) for n, p in cfgs.items()}
    maps, ledger = audit.collect_maps(splits)
    assert sorted(maps) == ["depth_map/Cafe/Cafe_idx_0/1.npy", "depth_map/Cafe/Cafe_idx_0/12.npy",
                            "depth_map/Cafe/Cafe_idx_0/3.npy", "depth_map/Cafe/Cafe_idx_1/1.npy",
                            "depth_map/Office/Office_idx_2/5.npy", "depth_map/Office/Office_idx_2/6.npy"]
    assert maps["depth_map/Cafe/Cafe_idx_0/1.npy"]["counts"] == {"train": 2, "unseen": 0, "seen": 1}
    assert maps["depth_map/Cafe/Cafe_idx_0/3.npy"]["counts"] == {"train": 0, "unseen": 0, "seen": 1}
    rep = maps["depth_map/Cafe/Cafe_idx_0/1.npy"]["rep"]   # smallest item path, any split
    assert rep == str(tmp_path / "AcousticRooms" / "single_channel_ir_1" / "Cafe" / "Cafe_idx_0"
                      / "S001_R001_hybrid_IR.wav")
    assert audit.resolve_depth_path(rep, splits["train"]["root_abs"])[1] == "depth_map/Cafe/Cafe_idx_0/1.npy"
    assert [(r["split"], r["n_items"], r["n_unique_maps"]) for r in ledger] == [
        ("train", 4, 3), ("unseen", 2, 2), ("seen", 2, 2)]


def test_inputs_ledger_carries_the_config_and_split_hashes(tmp_path):
    import hashlib
    cfgs = _write_tree(tmp_path)
    split = audit.read_split_config(cfgs["unseen"])
    assert split["dataset_config_sha256"] == hashlib.sha256(open(cfgs["unseen"], "rb").read()).hexdigest()
    assert split["json_sha256"] == hashlib.sha256(open(tmp_path / "unseen.json", "rb").read()).hexdigest()
    assert split["custom_metadata_module_sha256"] == \
        hashlib.sha256(open(_ar_md().__file__, "rb").read()).hexdigest()
    assert split["folder_name"] == "single_channel_ir_1"


def test_relative_config_paths_resolve_against_the_repo_root():
    split = audit.read_split_config("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json")
    assert split["json_abs"] == os.path.join(_REPO_ROOT, "data", "AR", "unseen_eval.json")
    assert split["root_abs"] == os.path.join(_REPO_ROOT, "AcousticRooms")


@pytest.mark.parametrize("mutate, needle", [
    (lambda c: c["datasets"].append(dict(c["datasets"][0])), "exactly one dataset"),
    (lambda c: c["datasets"][0].pop("json_file_path"), "json_file_path")])
def test_read_split_config_refuses_unexpected_configs(tmp_path, mutate, needle):
    import json
    cfg = json.load(open(_PROD_TRAIN_CFG))
    mutate(cfg)
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=needle):
        audit.read_split_config(str(tmp_path / "c.json"))


def test_collect_maps_refuses_splits_with_different_roots(tmp_path):
    cfgs = _write_tree(tmp_path)
    splits = {n: audit.read_split_config(p) for n, p in cfgs.items()}
    splits["seen"]["root_abs"] = str(tmp_path / "Elsewhere" / "AcousticRooms")
    with pytest.raises(ValueError, match="same dataset root"):
        audit.collect_maps(splits)


# --------------------------------------------------------------------------- #
# per-map audit: the treatment path (AR_md cloud -> operator) vs the raw path
# --------------------------------------------------------------------------- #
def _one_map_tree(tmp_path, raw, scene="Cafe", sid="Cafe_idx_0", fname="S001_R007_hybrid_IR.wav"):
    split = {scene: {sid: [fname]}}
    key = audit.resolve_depth_path(f"/r/x/{scene}/{sid}/{fname}", "/r")[1]
    cfgs = _write_tree(tmp_path, splits={"unseen": split}, maps={key: raw})
    splits = {"unseen": audit.read_split_config(cfgs["unseen"])}
    maps, _ = audit.collect_maps(splits)
    return key, maps[key], splits["unseen"]


def _row(tmp_path, raw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    key, entry, split = _one_map_tree(tmp_path, raw)
    module = audit.load_metadata_module(split["custom_metadata_module_abs"])
    return audit.audit_map(key, entry, module, split["root_abs"]), entry


@pytest.mark.parametrize("name, peaks, period, want", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_audit_map_rows(tmp_path, name, peaks, period, want):
    import hashlib
    raw = _fixture(name, peaks, period).numpy()
    row, entry = _row(tmp_path, raw)
    assert row["error"] is None and row["path"] == "depth_map/Cafe/Cafe_idx_0/7.npy"
    assert row["sha256"] == hashlib.sha256(open(entry["path"], "rb").read()).hexdigest()
    j, raw_class, cands, survivors = audit.raw_select(raw)
    assert row["candidates"] == cands and row["n_candidates"] == len(cands)
    assert row["max_sep_deg"] == audit.max_circular_sep_deg(cands, W)
    assert row["raw_agree_set"] is True and row["raw_agree_col"] is True
    assert row["counts"] == {"unseen": 1}
    if want is None:
        assert row["survivor"] is True and row["survivors"] == survivors
        assert (row["j_star"], row["dj"], row["tie_class"]) == (None, None, None)
    else:
        assert row["survivor"] is False and row["tie_class"] == want == raw_class
        assert row["j_star"] == j and row["dj"] == (W - j) % W and row["survivors"] == [j]


def test_audit_map_records_a_processing_error(tmp_path):
    raw = _fixture("unique", [(40, 300)], None).numpy()
    raw[3, 4] = np.nan
    row, _ = _row(tmp_path, raw)
    assert row["error"] and "non-finite" in row["error"] and row["survivor"] is False
    assert row["sha256"] is not None and row["tie_class"] is None


def test_audit_map_refuses_content_other_than_what_ar_md_loads(tmp_path, monkeypatch):
    key, entry, split = _one_map_tree(tmp_path, _fixture("u", [(40, 300)], None).numpy())
    other = tmp_path / "other.npy"
    np.save(other, _fixture("u", [(41, 301)], None, seed=9).numpy())
    module = audit.load_metadata_module(split["custom_metadata_module_abs"])
    row = audit.audit_map(key, dict(entry, path=str(other)), module, split["root_abs"])
    assert row["error"] and "AR_md" in row["error"]


def test_raw_and_proxy_can_disagree_and_the_flags_say_so(tmp_path):
    """A raw tie across two ROWS at a non-power-of-two value: z = -d sin(phi) is rounded
    per row, so the proxy may break the raw tie. The flags must report it."""
    for v in np.linspace(7.6, 7.99, 40):
        raw = _fixture("u", [], None).numpy()
        raw[17, 100] = raw[201, 400] = v
        row, _ = _row(tmp_path / f"v{v:.6f}", raw)
        if row["raw_agree_set"] is False:
            assert row["error"] is None and audit.raw_candidates(raw) == [100, 400]
            assert row["n_candidates"] == 1
            assert row["raw_agree_col"] is (audit.raw_select(raw)[3] == row["survivors"])
            return
    pytest.fail("no raw tie broken by the proxy in the search space")


# --------------------------------------------------------------------------- #
# the whole audit: report + sidecar, the gate, subsets, determinism under workers
# --------------------------------------------------------------------------- #
PERIODIC_KEY = "depth_map/Office/Office_idx_2/5.npy"


def _report(tmp_path, maps=None, names=("train", "unseen", "seen"), limit=None, workers=1,
            out="audit.json"):
    """What main() does, step by step, on a tmp tree."""
    import json
    cfgs = _write_tree(tmp_path, maps=maps)
    splits = {n: audit.read_split_config(cfgs[n]) for n in names}
    found, ledger = audit.collect_maps(splits)
    module = splits[names[0]]["custom_metadata_module_abs"]
    rows = audit.run_rows(found, module, splits[names[0]]["root_abs"], workers, limit)
    rep = audit.build_report(rows, ledger, list(names), len(found), limit, module)
    audit.write_report(rep, str(tmp_path / out))
    assert json.loads((tmp_path / out).read_text()) == json.loads(json.dumps(rep))
    return rep, tmp_path / out


def test_a_clean_complete_audit_passes_and_publishes_json_and_sidecar(tmp_path):
    import hashlib
    rep, out = _report(tmp_path)
    assert rep["gate"] == {"verdict": "PASS", "exit_code": 0, "n_survivors": 0, "n_errors": 0}
    assert rep["schema_version"] == audit.SCHEMA_VERSION
    assert rep["coverage"] == {"splits": ["train", "unseen", "seen"], "limit": None, "subset": False,
                               "n_maps_union": 6, "n_maps_audited": 6, "complete": True}
    assert [r["path"] for r in rep["rows"]] == sorted(r["path"] for r in rep["rows"])
    assert rep["survivors"] == [] and rep["errors"] == []
    assert [(r["split"], r["n_items"], r["n_unique_maps"]) for r in rep["inputs"]] == [
        ("train", 4, 3), ("unseen", 2, 2), ("seen", 2, 2)]
    sidecar = (tmp_path / "audit.json.sha256").read_text()
    assert sidecar == f"{hashlib.sha256(out.read_bytes()).hexdigest()}  audit.json\n"
    per = rep["per_split"]["train"]
    assert per["maps"]["none"] + per["maps"]["same_column"] + per["maps"]["cross_column_image"] == 3
    assert sum(per["items"].values()) == 4 and sum(per["dj_histogram_items"]) == 4
    assert sum(per["dj_histogram_maps"]) == 3 and len(per["dj_histogram_maps"]) == W
    assert rep["agreement"]["raw_agree_set"] == {"true": 6, "false": 0}
    assert rep["tool"]["script_sha256"] == hashlib.sha256(open(_AUDIT_PY, "rb").read()).hexdigest()


def test_a_periodic_panorama_is_listed_and_fails_the_gate(tmp_path):
    periodic = _scalar_depth(256, W, seed=5, period=128, peaks=[(60, 11)]).numpy()
    rep, _ = _report(tmp_path, maps={PERIODIC_KEY: periodic})
    assert rep["gate"]["verdict"] == "FAIL" and rep["gate"]["exit_code"] == 1
    assert rep["survivors"] == [{"path": PERIODIC_KEY, "candidates": [11, 139, 267, 395],
                                 "survivors": [11, 139, 267, 395],
                                 "counts": {"train": 0, "unseen": 1, "seen": 0}}]
    assert rep["per_split"]["unseen"]["maps"]["survivor"] == 1


@pytest.mark.parametrize("kw", [dict(limit=4), dict(names=("unseen", "seen"))])
def test_a_subset_never_passes(tmp_path, kw):
    rep, _ = _report(tmp_path, **kw)
    assert rep["gate"]["verdict"] == "REFUSED_SUBSET" and rep["gate"]["exit_code"] == 4
    assert rep["coverage"]["subset"] is True and rep["coverage"]["complete"] is False


def test_a_survivor_inside_a_subset_still_fails(tmp_path):
    periodic = _scalar_depth(256, W, seed=5, period=256, peaks=[(60, 11)]).numpy()
    rep, _ = _report(tmp_path, maps={PERIODIC_KEY: periodic}, names=("unseen",))
    assert rep["gate"]["verdict"] == "FAIL"


def test_an_unprocessed_map_fails_the_gate(tmp_path):
    bad = _scalar_depth(256, W, seed=5).numpy()
    bad[0, 0] = np.inf
    rep, _ = _report(tmp_path, maps={PERIODIC_KEY: bad})
    assert rep["gate"]["verdict"] == "INCOMPLETE" and rep["gate"]["exit_code"] == 3
    assert [e["path"] for e in rep["errors"]] == [PERIODIC_KEY]


def test_workers_do_not_change_a_single_byte(tmp_path):
    _, one = _report(tmp_path, workers=1, out="w1.json")
    _, three = _report(tmp_path, workers=3, out="w3.json")
    assert one.read_bytes() == three.read_bytes()


# --------------------------------------------------------------------------- #
# CLI: exit code == gate, markdown summary on stdout, argument refusals
# --------------------------------------------------------------------------- #
def _main(tmp_path, monkeypatch, capsys, *args, maps=None):
    cfgs = _write_tree(tmp_path, maps=maps)
    monkeypatch.setattr(audit, "SPLIT_CONFIGS", cfgs)
    rc = audit.main(["--out", str(tmp_path / "cli.json"), "--workers", "2"] + list(args))
    return rc, capsys.readouterr().out


@pytest.mark.parametrize("args, maps, rc_want, needle", [
    ([], None, 0, "gate PASS (exit 0)"),
    ([], {PERIODIC_KEY: _scalar_depth(256, W, seed=5, period=256, peaks=[(60, 11)]).numpy()}, 1,
     f"{PERIODIC_KEY} candidates [11, 267]"),
    (["--limit", "2"], None, 4, "gate REFUSED_SUBSET (exit 4)"),
    (["--splits", "seen,unseen"], None, 4, "coverage: splits unseen,seen"),
    ([], {PERIODIC_KEY: np.full((256, W), np.inf)}, 3, "errors (1):")])
def test_main_exit_code_is_the_gate_and_stdout_has_the_summary(
        tmp_path, monkeypatch, capsys, args, maps, rc_want, needle):
    rc, text = _main(tmp_path, monkeypatch, capsys, *args, maps=maps)
    assert rc == rc_want and needle in text
    assert "| split | items | maps |" in text and "timing:" in text and "maps/s" in text
    assert (tmp_path / "cli.json").exists() and (tmp_path / "cli.json.sha256").exists()


@pytest.mark.parametrize("args", [["--splits", "train,bogus"], ["--splits", "seen,seen"],
                                  ["--limit", "0"]])
def test_main_refuses_malformed_arguments_before_any_data_access(args, monkeypatch):
    monkeypatch.setattr(audit, "read_split_config", lambda *a: pytest.fail("data was read"))
    with pytest.raises(SystemExit) as err:
        audit.main(args)
    assert err.value.code == 2


def test_the_script_runs_as_a_program():
    import subprocess
    proc = subprocess.run([sys.executable, _AUDIT_PY, "--splits", "nope"], capture_output=True,
                          text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 2 and "--splits must be distinct names" in proc.stderr
