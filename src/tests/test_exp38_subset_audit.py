"""exp_38 round A4 — the exhaustive subset audit (exp38_subset_audit.py) on toy AR trees.

The audit's candidate pool must be AR_md's own candidate list (captured from the real
``get_ir_and_location_for_other_sources``), its ordered target list ``get_audio_filenames``'s,
and its failure modes (empty restricted pool, split file missing on disk) must be reported.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("exp38_subset_audit", os.path.join(EXP, "exp38_subset_audit.py"))
ar_md = _load("ar_md_under_test", os.path.join(REPO, "src/configs/dataset_configs/custom_metadata/AR_md.py"))


def _tree(tmp_path, rooms):
    """rooms: {(scene, room): [(src, rec), ...]} -> dataset root path."""
    root = tmp_path / "AcousticRooms"
    for (scene, room), pairs in rooms.items():
        d = root / audit.FOLDER / scene / room
        d.mkdir(parents=True)
        for s, r in pairs:
            (d / f"S00{s}_R00{r}_hybrid_IR.wav").write_bytes(b"")   # real AR naming: S000 ... S0011
    return str(root)


class _Captured(Exception):
    pass


def _ar_md_pool(path, allowed=None):
    """The candidate list AR_md hands to np.random.choice for this target."""
    seen = {}
    real = np.random.choice

    def spy(pool, *a, **k):
        seen["pool"] = [os.path.basename(p) for p in pool]
        raise _Captured
    np.random.choice = spy
    try:
        ar_md.get_ir_and_location_for_other_sources(path, 8, "unused", allowed_basenames=allowed)
    except _Captured:
        pass
    finally:
        np.random.choice = real
    return seen["pool"]


def test_candidate_pool_is_ar_mds_own_list_in_ar_mds_order(tmp_path):
    pairs = [(s, r) for s in range(1, 10) for r in range(1, 4) if (s, r) not in {(3, 2), (7, 1)}]
    root = _tree(tmp_path, {("Toy", "Toy_idx_0"): pairs})
    d = os.path.join(root, audit.FOLDER, "Toy", "Toy_idx_0")
    listing = sorted(os.listdir(d))
    for s, r in pairs:
        name = f"S00{s}_R00{r}_hybrid_IR.wav"
        assert audit.candidate_pool(listing, name) == _ar_md_pool(os.path.join(d, name))


def test_candidate_pool_never_reaches_a_S010_file(tmp_path):
    """AR_md rebuilds names as S00{int}: a file named S010_... parses to 10 and is rebuilt as
    S0010_..., which does not exist -> unreachable as a context (exp_14's sampler-faithful rule)."""
    pairs = [(s, 1) for s in range(0, 10)]
    root = _tree(tmp_path, {("Toy", "Toy_idx_0"): pairs})
    d = os.path.join(root, audit.FOLDER, "Toy", "Toy_idx_0")
    open(os.path.join(d, "S010_R001_hybrid_IR.wav"), "wb").close()
    listing = sorted(os.listdir(d))
    pool = audit.candidate_pool(listing, "S001_R001_hybrid_IR.wav")
    assert len(pool) == 9 and not any(c.startswith(("S010_", "S0010_")) for c in pool)
    for name in ("S001_R001_hybrid_IR.wav", "S010_R001_hybrid_IR.wav"):
        assert audit.candidate_pool(listing, name) == _ar_md_pool(os.path.join(d, name))


def _split(tmp_path, content, name="split.json"):
    p = tmp_path / name
    p.write_text(json.dumps(content))
    return str(p)


def test_audit_split_counts_histograms_and_order(tmp_path):
    pairs = [(s, r) for s in range(1, 6) for r in range(1, 3)]
    root = _tree(tmp_path, {("Toy", "Toy_idx_0"): pairs, ("Toy", "Toy_idx_1"): pairs})
    split = {"Toy": {"Toy_idx_0": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav"],
                     "Toy_idx_1": [f"S00{s}_R002_hybrid_IR.wav" for s in range(1, 6)]}}
    summary, per_room, problems = audit.audit_split(_split(tmp_path, split), True, root)
    assert problems == []
    assert summary["n_targets"] == 7 and summary["rooms"] == 2
    assert per_room == {"Toy_idx_0": {"0": 0, "1-7": 2, ">=8": 0}, "Toy_idx_1": {"0": 0, "1-7": 5, ">=8": 0}}
    # room 0: each target has 4 on-disk candidates at R001, 1 in-split -> 3 removed, twice
    assert summary["candidates_removed_by_restriction"] == summary["on_disk_candidates_outside_split"] == 6


def test_audit_split_reports_an_empty_restricted_pool_and_a_missing_file(tmp_path):
    root = _tree(tmp_path, {("Toy", "Toy_idx_0"): [(1, 1), (2, 1), (1, 2)]})
    split = {"Toy": {"Toy_idx_0": ["S001_R002_hybrid_IR.wav", "S009_R009_hybrid_IR.wav"]}}
    summary, _, problems = audit.audit_split(_split(tmp_path, split), True, root)
    assert summary["empty_restricted_pools"] == 1
    assert any("absent on disk" in p for p in problems) and any("empty restricted pool" in p for p in problems)


def test_unrestricted_split_counts_outside_candidates_without_failing(tmp_path):
    root = _tree(tmp_path, {("Toy", "Toy_idx_0"): [(1, 1), (2, 1), (3, 1)]})
    split = {"Toy": {"Toy_idx_0": ["S001_R001_hybrid_IR.wav"]}}
    summary, per_room, problems = audit.audit_split(_split(tmp_path, split), False, root)
    assert problems == [] and summary["on_disk_candidates_outside_split"] == 2
    assert summary["candidates_removed_by_restriction"] == 0
    assert per_room["Toy_idx_0"]["0"] == 1                       # in-split pool of the only target is empty


def test_function_source_hashes_pin_the_ported_code():
    for (path, name), want in audit.EXP14_FN_SHA256.items():
        assert audit.fn_source_sha256(open(os.path.join(REPO, path)).read(), name) == want, name
    assert audit.fn_source_sha256("def f():\n    return 1\n", "g") is None


def test_audit_refuses_to_overwrite(tmp_path):
    out = tmp_path / "a.json"
    out.write_text("{}")
    assert audit.main(["--out", str(out)]) == 2
