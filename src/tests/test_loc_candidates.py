"""Tests for ``src.localization.candidates`` (exp_18 loc_invert, round 1).

Written test-first (announcement 02). The contracts under test are
``loc_invert_impl_contracts.md`` §4.2 plus the Rev 3 §4 deltas: numeric,
naming-tolerant id parsing (the wav namespace and the metadata-file namespace
are separate -- ``S008_R089_hybrid_IR.wav`` vs ``S008_R0089.json`` -- so every
lookup matches on parsed numeric identity, never on a reconstructed fixed
format), metadata pair JSONs as the candidate authority, and a shallow-copy
metadata variant that swaps only ``source``/``source_vit``.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

from src.localization.candidates import (
    enumerate_metadata_sources,
    find_pair_metadata,
    parse_ir_filename,
    project_to_camera,
)


# --------------------------------------------------------------------------- #
# fixture helpers: a metadata room directory of pair JSONs
# --------------------------------------------------------------------------- #
def _write_pair(room_dir, src, rec, src_loc, rec_loc, name=None):
    """Write one pair JSON. ``name`` overrides the file name so a fixture can mix
    naming conventions (release style is the literal ``"S00" + str(node)``)."""
    room_dir.mkdir(parents=True, exist_ok=True)
    fname = name if name is not None else f"S00{src}_R00{rec}.json"
    path = room_dir / fname
    path.write_text(json.dumps({
        "sim_src_id": f"src-{src}", "sim_rec_id": f"rec-{rec}",
        "src_loc": [float(v) for v in src_loc],
        "rec_loc": [float(v) for v in rec_loc],
        "IR_norm": 1.0,
    }))
    return path


_SRC_LOCS = {0: (0.0, 0.0, 1.0), 7: (1.5, -2.0, 1.25), 10: (-3.0, 4.5, 0.75)}
_REC_LOCS = {0: (0.25, 0.5, 1.0), 3: (-1.0, 2.0, 1.1)}


def _build_room(tmp_path, name_fn=None):
    """Metadata room with sources {0, 7, 10} x receivers {0, 3}."""
    room = tmp_path / "metadata" / "Cafe" / "Cafe_idx_1"
    for src, src_loc in _SRC_LOCS.items():
        for rec, rec_loc in _REC_LOCS.items():
            fname = None if name_fn is None else name_fn(src, rec)
            _write_pair(room, src, rec, src_loc, rec_loc, name=fname)
    return room


# --------------------------------------------------------------------------- #
# parse_ir_filename
# --------------------------------------------------------------------------- #
def test_parse_ir_filename_standard():
    assert parse_ir_filename("S008_R089_hybrid_IR.wav") == (8, 89)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("S000_R000_hybrid_IR.wav", (0, 0)),
        ("S010_R0010_hybrid_IR.wav", (10, 10)),      # differing zero-padding
        ("S00100_R7_hybrid_IR.wav", (100, 7)),
        ("S0_R0123_hybrid_IR.wav", (0, 123)),
    ],
)
def test_parse_ir_filename_padding_variants(name, expected):
    """Digit counts vary between rooms/nodes; identity is the parsed integer."""
    assert parse_ir_filename(name) == expected


def test_parse_ir_filename_accepts_full_path():
    got = parse_ir_filename("/data/AR/single_channel_ir_1/Cafe/Cafe_idx_1/S008_R089_hybrid_IR.wav")
    assert got == (8, 89)


@pytest.mark.parametrize(
    "bad",
    [
        "S008_R089.wav",              # not an IR file name
        "S008_R089_hybrid_IR.json",   # wrong extension
        "R089_S008_hybrid_IR.wav",    # swapped roles
        "S008_hybrid_IR.wav",         # missing receiver
        "SXXX_R089_hybrid_IR.wav",    # non-numeric
        "",
    ],
)
def test_parse_ir_filename_malformed_raises(bad):
    with pytest.raises(ValueError):
        parse_ir_filename(bad)


# --------------------------------------------------------------------------- #
# find_pair_metadata -- naming-tolerant, matched on numeric identity
# --------------------------------------------------------------------------- #
def test_find_pair_metadata_release_naming(tmp_path):
    room = _build_room(tmp_path)                       # "S00" + str(node)
    got = find_pair_metadata(room, 10, 3)
    assert got is not None
    assert os.path.basename(got) == "S0010_R003.json"


def test_find_pair_metadata_zero_padded_naming(tmp_path):
    room = _build_room(tmp_path, name_fn=lambda s, r: f"S{s:03d}_R{r:03d}.json")
    got = find_pair_metadata(room, 10, 3)
    assert got is not None
    assert os.path.basename(got) == "S010_R003.json"


def test_find_pair_metadata_missing_returns_none(tmp_path):
    room = _build_room(tmp_path)
    assert find_pair_metadata(room, 4, 3) is None       # no such source
    assert find_pair_metadata(room, 10, 99) is None     # no such receiver


def test_find_pair_metadata_ambiguous_naming_raises(tmp_path):
    """Two file names parsing to the same numeric identity is not resolvable."""
    room = _build_room(tmp_path)
    _write_pair(room, 10, 3, _SRC_LOCS[10], _REC_LOCS[3], name="S010_R003.json")
    with pytest.raises(ValueError):
        find_pair_metadata(room, 10, 3)


# --------------------------------------------------------------------------- #
# enumerate_metadata_sources -- the candidate authority (C7)
# --------------------------------------------------------------------------- #
def test_enumerate_metadata_sources_unique_nodes(tmp_path):
    room = _build_room(tmp_path)
    sources = enumerate_metadata_sources(room)
    assert sorted(sources) == [0, 7, 10]                # unique despite 2 receivers each
    for node, xyz in sources.items():
        assert isinstance(xyz, np.ndarray) and xyz.shape == (3,) and xyz.dtype == np.float64
        np.testing.assert_array_equal(xyz, np.asarray(_SRC_LOCS[node], dtype=np.float64))


def test_enumerate_metadata_sources_naming_tolerant(tmp_path):
    """Zero-padded names give the identical node->xyz map as release naming."""
    a = enumerate_metadata_sources(_build_room(tmp_path / "a"))
    b = enumerate_metadata_sources(
        _build_room(tmp_path / "b", name_fn=lambda s, r: f"S{s:03d}_R{r:03d}.json"))
    assert sorted(a) == sorted(b)
    for node in a:
        np.testing.assert_array_equal(a[node], b[node])


def test_enumerate_metadata_sources_ignores_foreign_files(tmp_path):
    room = _build_room(tmp_path)
    (room / "README.txt").write_text("not a pair file")
    (room / "S0010_R003_hybrid_IR.wav").write_bytes(b"")
    assert sorted(enumerate_metadata_sources(room)) == [0, 7, 10]


def test_enumerate_metadata_sources_inconsistent_across_receivers_raises(tmp_path):
    room = _build_room(tmp_path)
    _write_pair(room, 7, 3, (9.0, 9.0, 9.0), _REC_LOCS[3])   # overwrite: moved source
    with pytest.raises(ValueError):
        enumerate_metadata_sources(room)


def test_enumerate_metadata_sources_missing_key_raises(tmp_path):
    room = _build_room(tmp_path)
    (room / "S007_R003.json").write_text(json.dumps({"rec_loc": [0.0, 0.0, 0.0]}))
    with pytest.raises(ValueError):
        enumerate_metadata_sources(room)


def test_enumerate_metadata_sources_empty_or_missing_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        enumerate_metadata_sources(empty)
    with pytest.raises(ValueError):
        enumerate_metadata_sources(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# project_to_camera -- parity with the release loader's projection
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AR_MD_PATH = os.path.join(
    _REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata", "AR_md.py")


def _load_ar_md():
    """Load the release metadata module exactly as ``src/data/dataset.py`` does."""
    spec = importlib.util.spec_from_file_location("metadata_module", _AR_MD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_to_camera_matches_ar_md_release_math():
    """Bit-exact parity with ``AR_md.get_3d_point_camera_coord`` on 100 random
    points. Both are the same single subtraction per axis, so exact float
    equality (rtol=atol=0) is the contract -- any drift means the candidate
    conditioning would not reproduce the loader's ``md['source']``."""
    ar_md = _load_ar_md()
    rng = np.random.default_rng(18)
    for _ in range(100):
        rec_loc = (rng.uniform(-10.0, 10.0, size=3)).tolist()
        xyz = (rng.uniform(-10.0, 10.0, size=3)).tolist()
        expected = ar_md.get_3d_point_camera_coord(source_pose=rec_loc, point_3d=xyz)
        got = project_to_camera(rec_loc, xyz)
        np.testing.assert_array_equal(got, np.asarray(expected, dtype=np.float64))


def test_project_to_camera_is_translation():
    rec = np.array([1.0, -2.0, 0.5])
    np.testing.assert_array_equal(project_to_camera(rec, rec), np.zeros(3))
    np.testing.assert_array_equal(
        project_to_camera(rec, np.array([2.0, 0.0, 1.5])), np.array([1.0, 2.0, 1.0]))


def test_project_to_camera_dtype_and_shape():
    out = project_to_camera([0, 0, 0], [1, 2, 3])
    assert isinstance(out, np.ndarray) and out.shape == (3,) and out.dtype == np.float64


def test_project_to_camera_rejects_bad_shape():
    with pytest.raises(ValueError):
        project_to_camera([0.0, 0.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        project_to_camera([0.0, 0.0, 0.0], [1.0, 2.0])
