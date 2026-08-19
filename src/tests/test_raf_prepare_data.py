"""Tests for ``data/RAF/prepare_data.py`` — RAF -> FLAC-runtime preparation.

exp_19 (RAF finetune), contract section B. TDD cycles:

* cycle 4 — ``load_room_index`` + per-capture cross-check
* cycle 5 — ``group_captures`` (canonical 7-tuple key, exactly-36 invariant) and
  placement clustering
* cycle 6 — ``select_splits``, split-JSON emission, splits record
* cycle 7 — ``resample_and_write`` + amplitude audit, runtime metadata, CLI
* cycle 13 — the trailing all-NaN ``all_rx`` sentinel rule (contracts Amendment 1)

Every fixture here is SYNTHETIC (pytest ``tmp_path``). The real RAF corpus under
/media/diskstation is read-only and is never touched by the test-suite.
"""
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import prepare_data as raf_prepare  # noqa: E402
import raf_common  # noqa: E402

assert os.path.dirname(os.path.abspath(raf_prepare.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# synthetic mini-room fixture
# --------------------------------------------------------------------------- #
N_MICS = 36


def _mic_array():
    """A rigid 36-point array: 3 x 3 x 4 lattice spanning 1.0 x 1.0 x 1.2 m (RAF axes).

    The z spacing is deliberately asymmetric (0, 0.5, 0.9, 1.2 -> mean 0.65) so the
    lattice point nearest the centroid is unique by a clear margin: an evenly
    spaced axis puts two points at exactly the same distance and the hand-derived
    FPS oracle would then turn on a 1-ULP difference rather than on geometry.
    """
    pts = []
    for x in (0.0, 0.5, 1.0):
        for y in (0.0, 0.5, 1.0):
            for z in (0.0, 0.5, 0.9, 1.2):
                pts.append((x, y, z))
    assert len(pts) == N_MICS
    return np.array(pts, dtype=np.float64)


def _default_groups(n_groups):
    """(quat, tx_xyz, placement_centre) per group, in RAF world coordinates."""
    out = []
    for g in range(n_groups):
        quat = (round(0.1 * (g + 1), 6), 0.9, 0.0, 0.1)
        tx = (round(1.0 * g, 6), 1.5, round(0.5 * g, 6))
        centre = (round(2.0 + 0.7 * g, 6), 0.0, round(-1.0 + 0.3 * g, 6))
        out.append((quat, tx, centre))
    return out


def _rir(seed, n=4800, sr=48000, peak=0.01):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    sig = rng.normal(size=n) * np.exp(-t * 20.0)
    sig = sig / np.abs(sig).max() * peak
    return sig.astype(np.float32)


def write_room(root, room, groups=None, n_mics=N_MICS, extra_rx_lines=(),
               extra_tx_lines=(), rir_peak=0.01, rir_len=4800):
    """Write one synthetic RAF room under ``<root>/archived/<room>/``.

    Returns the list of (line_index, capture_id, tx_line, rx_line) tuples written.
    """
    if groups is None:
        groups = _default_groups(3)
    room_dir = os.path.join(root, "archived", room)
    os.makedirs(os.path.join(room_dir, "metadata"), exist_ok=True)
    array = _mic_array()

    tx_lines, rx_lines, records = [], [], []
    idx = 0
    for quat, tx, centre in groups:
        tx_line = ",".join(f"{v:.6f}" for v in list(quat) + list(tx))
        for m in range(n_mics):
            rx = array[m] + np.array(centre)
            rx_line = ",".join(f"{v:.6f}" for v in rx)
            cid = f"{idx:06d}"
            cdir = os.path.join(room_dir, "data", cid)
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "tx_pos.txt"), "w") as f:
                f.write(tx_line + "\n")
            with open(os.path.join(cdir, "rx_pos.txt"), "w") as f:
                f.write(rx_line + "\n")
            sf.write(os.path.join(cdir, "rir.wav"), _rir(idx, n=rir_len, peak=rir_peak),
                     48000, subtype="FLOAT")
            tx_lines.append(tx_line)
            rx_lines.append(rx_line)
            records.append((idx, cid, tx_line, rx_line))
            idx += 1

    with open(os.path.join(room_dir, "metadata", "all_tx_pos.txt"), "w") as f:
        f.write("\n".join(tx_lines + list(extra_tx_lines)) + "\n")
    with open(os.path.join(room_dir, "metadata", "all_rx_pos.txt"), "w") as f:
        f.write("\n".join(rx_lines + list(extra_rx_lines)) + "\n")
    return records


@pytest.fixture
def mini_room(tmp_path):
    """3 groups x 36 captures, one room."""
    write_room(str(tmp_path), "EmptyRoom")
    return os.path.join(str(tmp_path), "archived", "EmptyRoom")


# --------------------------------------------------------------------------- #
# load_room_index (cycle 4)
# --------------------------------------------------------------------------- #
def test_load_room_index_returns_one_record_per_capture_in_id_order(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    assert len(index) == 3 * N_MICS
    assert [r["capture_id"] for r in index[:3]] == ["000000", "000001", "000002"]
    assert index[0]["index"] == 0
    assert index[0]["tx_xyz"].tolist() == [0.0, 1.5, 0.0]
    assert index[0]["quat"].tolist() == [0.1, 0.9, 0.0, 0.1]
    # first mic of the first group: array point (0,0,0) + centre (2, 0, -1)
    assert index[0]["rx_xyz"].tolist() == [2.0, 0.0, -1.0]
    assert index[N_MICS]["tx_xyz"].tolist() == [1.0, 1.5, 0.5]


def test_load_room_index_tolerates_trailing_blank_lines(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    for name in ("all_tx_pos.txt", "all_rx_pos.txt"):
        p = os.path.join(room_dir, "metadata", name)
        with open(p, "a") as f:
            f.write("\n   \n\n")
    assert len(raf_prepare.load_room_index(room_dir)) == N_MICS


def test_load_room_index_accepts_the_observed_trailing_nan_sentinel(tmp_path):
    """Contracts Amendment 1 (D1): the released corpus ships exactly one trailing
    ``nan,nan,nan`` line in all_rx_pos.txt. It is dropped IFF it is the final line,
    all three fields are NaN, and the counts line up after the drop — and the drop
    is RECORDED, never silent."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_rx_lines=("nan,nan,nan",))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    index = raf_prepare.load_room_index(room_dir)
    assert len(index) == N_MICS
    assert index.rx_trailing_sentinel_dropped is True
    assert index[-1]["capture_id"] == f"{N_MICS - 1:06d}"
    assert np.isfinite(index[-1]["rx_xyz"]).all()


def test_load_room_index_records_no_sentinel_when_there_is_none(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    assert raf_prepare.load_room_index(room_dir).rx_trailing_sentinel_dropped is False


def test_load_room_index_aborts_on_two_trailing_nan_lines(tmp_path):
    """EXACTLY one sentinel is tolerated; two means something else is wrong."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_rx_lines=("nan,nan,nan", "nan,nan,nan"))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    with pytest.raises(ValueError) as exc:
        raf_prepare.load_room_index(room_dir)
    assert "all_rx_pos.txt" in str(exc.value)


def test_load_room_index_aborts_when_counts_are_still_wrong_after_the_drop(tmp_path):
    """A trailing sentinel does not license a count mismatch: here all_tx is also
    one line short, so dropping the sentinel still leaves rx != tx."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_rx_lines=("nan,nan,nan",))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    p = os.path.join(room_dir, "metadata", "all_tx_pos.txt")
    lines = open(p).read().splitlines()
    open(p, "w").write("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_aborts_on_a_trailing_nan_line_in_all_tx(tmp_path):
    """The sentinel rule is rx-only: nothing licenses a stray tx line."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_tx_lines=("nan,nan,nan,nan,nan,nan,nan",))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


@pytest.mark.parametrize("filename,replacement", [
    ("all_rx_pos.txt", "nan,nan,nan"),
    ("all_rx_pos.txt", "1.0,nan,3.0"),
    ("all_tx_pos.txt", "nan,nan,nan,nan,nan,nan,nan"),
    ("all_tx_pos.txt", "0.1,0.9,0.0,0.1,1.0,nan,0.0"),
])
def test_load_room_index_aborts_on_a_mid_file_nan(tmp_path, filename, replacement):
    """NaN anywhere other than the single trailing rx sentinel still aborts."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    p = os.path.join(room_dir, "metadata", filename)
    lines = open(p).read().splitlines()
    lines[7] = replacement
    open(p, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_does_not_drop_a_final_nan_that_is_a_data_line(tmp_path):
    """Counts already match, so the last line is DATA — a NaN there is a corrupt
    capture, not a sentinel, and must abort rather than silently shorten the room."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    p = os.path.join(room_dir, "metadata", "all_rx_pos.txt")
    lines = open(p).read().splitlines()
    lines[-1] = "nan,nan,nan"
    open(p, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_does_not_drop_a_trailing_inf_line(tmp_path):
    """The sentinel is NaN specifically; inf is a corrupt value, not a sentinel."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_rx_lines=("inf,inf,inf",))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_sentinel_flag_read_is_fail_closed():
    """A plain list carries no provenance: reading the flag off one must raise,
    never report a comfortable False."""
    with pytest.raises(AttributeError):
        raf_prepare.sentinel_flag_of([])


def test_load_room_index_aborts_when_a_capture_dir_is_missing(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    import shutil
    shutil.rmtree(os.path.join(room_dir, "data", "000005"))
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_aborts_on_non_contiguous_capture_ids(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    os.rename(os.path.join(room_dir, "data", "000005"),
              os.path.join(room_dir, "data", "000099"))
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_aborts_on_a_malformed_line(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    p = os.path.join(room_dir, "metadata", "all_tx_pos.txt")
    lines = open(p).read().splitlines()
    lines[3] = "1,2,3"
    open(p, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        raf_prepare.load_room_index(room_dir)


def test_load_room_index_aborts_on_missing_metadata_file(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    os.remove(os.path.join(room_dir, "metadata", "all_tx_pos.txt"))
    with pytest.raises((ValueError, FileNotFoundError)):
        raf_prepare.load_room_index(room_dir)


# --------------------------------------------------------------------------- #
# crosscheck_captures (cycle 4)
# --------------------------------------------------------------------------- #
def test_crosscheck_passes_on_a_consistent_room(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    report = raf_prepare.crosscheck_captures(mini_room, index, n_sample=20, seed=0)
    assert report["checked"] == 20
    assert report["mode"] == "sample"
    assert report["mismatches"] == 0


def test_crosscheck_full_checks_every_capture(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    report = raf_prepare.crosscheck_captures(mini_room, index, full=True)
    assert report["checked"] == len(index)
    assert report["mode"] == "full"


def test_crosscheck_sample_size_is_capped_at_the_corpus_size(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    report = raf_prepare.crosscheck_captures(mini_room, index, n_sample=10 ** 6, seed=0)
    assert report["checked"] == len(index)


def test_crosscheck_aborts_on_a_per_capture_tx_mismatch(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    p = os.path.join(mini_room, "data", "000000", "tx_pos.txt")
    open(p, "w").write("0.1,0.9,0.0,0.1,9.999999,1.500000,0.000000\n")
    with pytest.raises(ValueError) as exc:
        raf_prepare.crosscheck_captures(mini_room, index, full=True)
    assert "000000" in str(exc.value)


def test_crosscheck_aborts_on_a_per_capture_rx_mismatch(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    p = os.path.join(mini_room, "data", "000007", "rx_pos.txt")
    open(p, "w").write("9.0,9.0,9.0\n")
    with pytest.raises(ValueError) as exc:
        raf_prepare.crosscheck_captures(mini_room, index, full=True)
    assert "000007" in str(exc.value)


def test_crosscheck_sampling_is_seeded_and_reproducible(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    a = raf_prepare.crosscheck_captures(mini_room, index, n_sample=5, seed=3)
    b = raf_prepare.crosscheck_captures(mini_room, index, n_sample=5, seed=3)
    c = raf_prepare.crosscheck_captures(mini_room, index, n_sample=5, seed=4)
    assert a["capture_ids"] == b["capture_ids"]
    assert a["capture_ids"] != c["capture_ids"]
    # a sample is a sample: a corruption outside it is (by construction) not seen
    assert set(a["capture_ids"]).issubset({r["capture_id"] for r in index})


# --------------------------------------------------------------------------- #
# group_captures (cycle 5)
# --------------------------------------------------------------------------- #
def _index_entry(i, quat, tx, rx):
    return {
        "index": i,
        "capture_id": f"{i:06d}",
        "quat": np.array(quat, dtype=np.float64),
        "tx_xyz": np.array(tx, dtype=np.float64),
        "rx_xyz": np.array(rx, dtype=np.float64),
    }


def _synthetic_index(specs, n_mics=N_MICS, array=None):
    """specs: list of (quat, tx, centre); emits n_mics captures per spec."""
    array = _mic_array() if array is None else array
    index, i = [], 0
    for quat, tx, centre in specs:
        for m in range(n_mics):
            index.append(_index_entry(i, quat, tx, array[m] + np.array(centre)))
            i += 1
    return index


def test_group_captures_groups_by_the_full_seven_tuple(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, report = raf_prepare.group_captures(index)
    assert len(groups) == 3
    assert report["n_groups"] == 3
    assert [len(g["capture_ids"]) for g in groups] == [N_MICS] * 3
    assert groups[0]["capture_ids"][0] == "000000"
    assert groups[1]["capture_ids"][0] == f"{N_MICS:06d}"
    assert groups[0]["tx_xyz"].tolist() == [0.0, 1.5, 0.0]
    assert report["nonuniform"] == []


def test_group_captures_orders_groups_by_first_capture(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, _ = raf_prepare.group_captures(index)
    firsts = [int(g["capture_ids"][0]) for g in groups]
    assert firsts == sorted(firsts)


def test_group_captures_merges_sign_flipped_quaternions():
    """q and -q are the same rotation: they must land in ONE group."""
    q = (0.1, 0.9, 0.0, 0.1)
    minus_q = tuple(-v for v in q)
    index = _synthetic_index([(q, (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))], n_mics=18)
    index += _synthetic_index([(minus_q, (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))], n_mics=18)
    for i, rec in enumerate(index):  # renumber after concatenation
        rec["index"] = i
        rec["capture_id"] = f"{i:06d}"
    groups, report = raf_prepare.group_captures(index)
    assert len(groups) == 1
    assert len(groups[0]["capture_ids"]) == 36
    assert groups[0]["quat_canon"][0] > 0


def test_group_captures_separates_same_xyz_different_orientation():
    """Grouping is by the FULL pose line, not by position (plan C2 audit)."""
    index = _synthetic_index([
        ((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0)),
        ((0.9, 0.1, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0)),
    ])
    for i, rec in enumerate(index):
        rec["index"], rec["capture_id"] = i, f"{i:06d}"
    groups, _ = raf_prepare.group_captures(index)
    assert len(groups) == 2
    assert groups[0]["group_key"] != groups[1]["group_key"]


def test_group_key_is_filesystem_safe_and_stable():
    index = _synthetic_index([((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))])
    key_a = raf_prepare.group_captures(index)[0][0]["group_key"]
    key_b = raf_prepare.group_captures(index)[0][0]["group_key"]
    assert key_a == key_b
    assert len(key_a) == 16
    assert all(c in "0123456789abcdef" for c in key_a)


def test_group_captures_records_the_canonical_tuple():
    index = _synthetic_index([((-0.1, -0.9, 0.0, -0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))])
    groups, _ = raf_prepare.group_captures(index)
    assert groups[0]["group_tuple"] == [0.1, 0.9, -0.0, 0.1, 1.0, 1.5, 2.0]


def test_group_captures_aborts_on_a_group_that_is_not_36():
    index = _synthetic_index([((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))],
                             n_mics=35, array=_mic_array()[:35])
    with pytest.raises(ValueError) as exc:
        raf_prepare.group_captures(index)
    assert "35" in str(exc.value)


def test_group_captures_allow_nonuniform_downgrades_to_a_recorded_warning():
    """FurnishedRoom has exactly one 72-capture group (measured 2026-08-19)."""
    array = np.vstack([_mic_array(), _mic_array() + 0.01])
    index = _synthetic_index([((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))],
                             n_mics=72, array=array)
    groups, report = raf_prepare.group_captures(index, allow_nonuniform=True)
    assert len(groups) == 1
    assert report["nonuniform"] == [{"group_key": groups[0]["group_key"], "size": 72}]


def test_group_captures_rejects_an_empty_index():
    with pytest.raises(ValueError):
        raf_prepare.group_captures([])


# --------------------------------------------------------------------------- #
# placement clustering (cycle 5)
# --------------------------------------------------------------------------- #
def test_placement_key_clusters_groups_re_occupying_a_placement():
    """Two tx poses over the same array placement (centroids within 1 cm)."""
    index = _synthetic_index([
        ((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0)),
        ((0.9, 0.1, 0.0, 0.1), (3.0, 1.5, 2.0), (0.004, 0.0, 0.0)),  # +4 mm
    ])
    for i, rec in enumerate(index):
        rec["index"], rec["capture_id"] = i, f"{i:06d}"
    groups, report = raf_prepare.group_captures(index)
    assert groups[0]["placement_key"] == groups[1]["placement_key"]
    assert report["n_placements"] == 1
    assert len(report["placements"][groups[0]["placement_key"]]) == 2


def test_placement_key_separates_distinct_placements():
    index = _synthetic_index([
        ((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0)),
        ((0.9, 0.1, 0.0, 0.1), (3.0, 1.5, 2.0), (0.5, 0.0, 0.0)),
    ])
    for i, rec in enumerate(index):
        rec["index"], rec["capture_id"] = i, f"{i:06d}"
    groups, report = raf_prepare.group_captures(index)
    assert groups[0]["placement_key"] != groups[1]["placement_key"]
    assert report["n_placements"] == 2


def test_group_rx_centroid_is_recorded_in_the_pipeline_frame():
    """Positions handed on to the runtime metadata are already pipeline-frame."""
    index = _synthetic_index([((0.1, 0.9, 0.0, 0.1), (1.0, 1.5, 2.0), (0.0, 0.0, 0.0))])
    groups, _ = raf_prepare.group_captures(index)
    raf_centroid = _mic_array().mean(axis=0)              # RAF (X, Y, Z)
    expected = raf_common.RAF_TO_PIPELINE @ raf_centroid  # -> (X, Z, Y)
    np.testing.assert_allclose(groups[0]["rx_centroid_p"], expected, atol=1e-9)
    np.testing.assert_allclose(groups[0]["tx_xyz_p"], [1.0, 2.0, 1.5], atol=1e-9)


# --------------------------------------------------------------------------- #
# select_splits (cycle 6)
# --------------------------------------------------------------------------- #
@pytest.fixture
def mini_groups(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, report = raf_prepare.group_captures(index)
    return groups, report


def test_select_splits_group_order_is_hand_checked_fps_over_tx(mini_groups):
    """tx (pipeline) = (0, 0, 1.5), (1, 0.5, 1.5), (2, 1.0, 1.5).

    centroid = (1, 0.5, 1.5) -> group 1 is exactly on it            -> start 1
    distances from group 1: [1.118, 0, 1.118] -> tie 0 vs 2 -> index 0
    remaining                                                        -> 2
    So with n_groups=2 / n_val_groups=1: train/test = {g1, g0}, val = {g2}.
    """
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    assert split["train_test_groups"] == [groups[1]["group_key"], groups[0]["group_key"]]
    assert split["val_groups"] == [groups[2]["group_key"]]
    assert split["reserve_groups"] == []


def test_select_splits_counts(mini_groups):
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    assert sum(len(v) for v in split["train_ids"].values()) == 2 * 12
    assert sum(len(v) for v in split["test_ids"].values()) == 2 * 24
    assert sum(len(v) for v in split["val_ids"].values()) == 36
    # every selected group carries a 12-mic support pool, including the val group:
    # a val item's acoustic context is drawn from its own group's support.
    assert set(split["support_ids"]) == set(split["train_test_groups"] + split["val_groups"])
    assert all(len(v) == 12 for v in split["support_ids"].values())


def test_select_splits_first_support_mic_is_hand_checked_fps_over_rx(mini_groups):
    """The 3x3x4 lattice's centroid is (0.5, 0.5, 0.65) in RAF axes.

    Nearest lattice point: x = 0.5 and y = 0.5 are exact hits; on z the candidates
    are 0.5 (0.15 away) and 0.9 (0.25 away), so (0.5, 0.5, 0.5) wins outright. With
    index = 12*ix + 4*iy + iz that is 12*1 + 4*1 + 1 = 17, i.e. capture 000017 of
    the first group (the next-nearest point is 0.10 further out, so the oracle does
    not hinge on floating-point detail).
    """
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    gk = groups[0]["group_key"]
    assert split["train_ids"][gk][0] == "000017"


def test_select_splits_is_group_atomic_and_disjoint(mini_groups):
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    train = [i for v in split["train_ids"].values() for i in v]
    test = [i for v in split["test_ids"].values() for i in v]
    val = [i for v in split["val_ids"].values() for i in v]
    assert len(set(train) & set(test)) == 0
    assert len(set(train) & set(val)) == 0
    assert len(set(test) & set(val)) == 0
    assert len(train) + len(test) + len(val) == len(set(train + test + val))
    # no capture of a val group appears in train/test in any role
    val_group_ids = set(groups[2]["capture_ids"])
    assert not (set(train) | set(test)) & val_group_ids


def test_select_splits_train_and_test_partition_their_group(mini_groups):
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    for gk in split["train_test_groups"]:
        g = next(x for x in groups if x["group_key"] == gk)
        assert sorted(split["train_ids"][gk] + split["test_ids"][gk]) == sorted(g["capture_ids"])


def test_select_splits_is_deterministic(mini_groups):
    groups, _ = mini_groups
    a = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    b = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    assert a["train_ids"] == b["train_ids"] and a["val_ids"] == b["val_ids"]


def test_select_splits_lists_reserve_groups(mini_groups):
    groups, _ = mini_groups
    split = raf_prepare.select_splits(groups, n_groups=1, n_val_groups=1, n_train=12)
    assert len(split["reserve_groups"]) == 1
    assert split["roles"][split["reserve_groups"][0]] == "reserve"


@pytest.mark.parametrize("kwargs", [
    {"n_groups": 3, "n_val_groups": 1},   # 4 > 3 available groups
    {"n_groups": 0, "n_val_groups": 1},
    {"n_groups": 2, "n_val_groups": 0},
    {"n_groups": 2, "n_val_groups": 1, "n_train": 37},
    {"n_groups": 2, "n_val_groups": 1, "n_train": 0},
])
def test_select_splits_rejects_impossible_parameters(mini_groups, kwargs):
    groups, _ = mini_groups
    kwargs.setdefault("n_train", 12)
    with pytest.raises(ValueError):
        raf_prepare.select_splits(groups, **kwargs)


# --------------------------------------------------------------------------- #
# split JSON emission + splits record (cycle 6)
# --------------------------------------------------------------------------- #
def _one_room_payload(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, group_report = raf_prepare.group_captures(index)
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    return {"EmptyRoom": {"groups": groups, "split": split, "group_report": group_report,
                          "crosscheck": {"mode": "full", "checked": len(index),
                                         "mismatches": 0},
                          "rx_trailing_sentinel_dropped":
                              raf_prepare.sentinel_flag_of(index)}}


def test_assemble_split_jsons_has_the_haa_shape(mini_room):
    payload = _one_room_payload(mini_room)
    jsons = raf_prepare.assemble_split_jsons(payload)
    assert set(jsons) == {"train", "val", "test"}
    assert list(jsons["train"]) == ["EmptyRoom"]
    assert len(jsons["train"]["EmptyRoom"]) == 24
    assert len(jsons["test"]["EmptyRoom"]) == 48
    assert len(jsons["val"]["EmptyRoom"]) == 36
    for name in jsons["train"]["EmptyRoom"]:
        assert name.endswith(".wav") and len(name) == 10 and name[:6].isdigit()
    assert jsons["train"]["EmptyRoom"] == sorted(jsons["train"]["EmptyRoom"])


def test_write_split_files_round_trips(tmp_path, mini_room):
    payload = _one_room_payload(mini_room)
    jsons = raf_prepare.assemble_split_jsons(payload)
    out = tmp_path / "splits"
    paths = raf_prepare.write_split_files(str(out), jsons)
    assert sorted(os.path.basename(p) for p in paths.values()) == [
        "test_base.json", "train_base.json", "val_base.json"]
    with open(paths["train"]) as f:
        assert json.load(f) == jsons["train"]


def test_splits_record_carries_the_preregistration_fields(mini_room):
    payload = _one_room_payload(mini_room)
    params = {"seed": 0, "n_groups": 2, "n_val_groups": 1, "n_train": 12,
              "rooms": ["EmptyRoom"]}
    record = raf_prepare.build_splits_record(payload, params)
    assert record["params"] == params
    room = record["rooms"]["EmptyRoom"]
    assert room["counts"] == {"train": 24, "test": 48, "val": 36,
                              "train_test_groups": 2, "val_groups": 1, "reserve_groups": 0}
    assert len(room["train_test_groups"]) == 2
    assert len(room["reserve_groups"]) == 0
    assert room["placements"]["n_placements"] == 3
    assert room["group_report"]["nonuniform"] == []
    assert room["crosscheck"]["mismatches"] == 0
    stats = room["distances"]["test_to_nearest_support"]
    assert stats["count"] == 48
    assert stats["min"] > 0.0
    assert stats["max"] <= 2.0   # the synthetic array spans 1.0 x 1.0 x 1.2 m
    assert set(stats) == {"count", "min", "p25", "median", "p75", "max", "mean"}
    assert "support_pairwise" in room["distances"]
    assert "git_describe" in record
    json.dumps(record)  # must be JSON-serialisable as-is


def test_splits_record_group_entries_carry_the_canonical_tuple(mini_room):
    payload = _one_room_payload(mini_room)
    record = raf_prepare.build_splits_record(payload, {"seed": 0})
    entry = record["rooms"]["EmptyRoom"]["group_details"][0]
    assert len(entry["group_tuple"]) == 7
    assert entry["role"] in {"train_test", "val", "reserve"}
    assert entry["placement_key"]
    assert len(entry["tx_xyz_p"]) == 3


# --------------------------------------------------------------------------- #
# resample_and_write + amplitude audit (cycle 7)
# --------------------------------------------------------------------------- #
def test_resample_writes_float32_wavs_at_22050(tmp_path, mini_room):
    out_room = tmp_path / "runtime" / "EmptyRoom"
    audit = raf_prepare.resample_and_write(mini_room, str(out_room),
                                           ["000000", "000001", "000002"])
    assert audit["n_files"] == 3
    assert audit["target_sr"] == 22050
    assert audit["subtype"] == "FLOAT"
    for cid in ("000000", "000001", "000002"):
        path = out_room / "mono_rirs_22050Hz" / f"{cid}.wav"
        assert path.exists()
        info = sf.info(str(path))
        assert info.samplerate == 22050
        assert info.channels == 1
        assert info.subtype == "FLOAT"
        data, _ = sf.read(str(path), dtype="float32")
        # 4800 samples at 48 kHz -> 2205 at 22.05 kHz (+- the resampler's edge)
        assert abs(len(data) - 2205) <= 2
        assert np.isfinite(data).all()


def test_resample_audit_records_peaks_and_silence_flags(tmp_path, mini_room):
    out_room = tmp_path / "runtime" / "EmptyRoom"
    audit = raf_prepare.resample_and_write(mini_room, str(out_room), ["000000"])
    entry = audit["files"]["000000"]
    assert 0.005 < entry["peak"] < 0.02          # synthetic RIRs peak at 0.01
    assert -50.0 < entry["dbfs"] < -30.0
    assert entry["silent_at_threshold"] is False
    assert entry["dbfs_crop"] <= entry["dbfs"] + 1e-6
    assert audit["silence_threshold_db"] == -60.0
    assert audit["n_silent"] == 0
    assert set(audit["peak_stats"]) == {"count", "min", "p25", "median", "p75",
                                        "max", "mean"}


def test_resample_audit_flags_a_sub_60db_file_without_aborting(tmp_path):
    """The loader silently substitutes items below -60 dBFS; the audit must count them."""
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=1e-5)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    audit = raf_prepare.resample_and_write(room_dir, str(tmp_path / "runtime" / "EmptyRoom"),
                                           ["000000"])
    assert audit["files"]["000000"]["silent_at_threshold"] is True
    assert audit["n_silent"] == 1


def test_resample_aborts_on_wrong_source_rate(tmp_path, mini_room):
    sf.write(os.path.join(mini_room, "data", "000000", "rir.wav"),
             _rir(0), 44100, subtype="FLOAT")
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(mini_room, str(tmp_path / "out"), ["000000"])


def test_resample_aborts_on_nan_input(tmp_path, mini_room):
    sig = _rir(0)
    sig[10] = np.nan
    sf.write(os.path.join(mini_room, "data", "000000", "rir.wav"), sig, 48000,
             subtype="FLOAT")
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(mini_room, str(tmp_path / "out"), ["000000"])


def test_resample_aborts_on_clipping(tmp_path, mini_room):
    sig = _rir(0)
    sig[10] = 1.5
    sf.write(os.path.join(mini_room, "data", "000000", "rir.wav"), sig, 48000,
             subtype="FLOAT")
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(mini_room, str(tmp_path / "out"), ["000000"])


def test_resample_aborts_on_multichannel_input(tmp_path, mini_room):
    stereo = np.stack([_rir(0), _rir(1)], axis=1)
    sf.write(os.path.join(mini_room, "data", "000000", "rir.wav"), stereo, 48000,
             subtype="FLOAT")
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(mini_room, str(tmp_path / "out"), ["000000"])


# --------------------------------------------------------------------------- #
# runtime metadata (cycle 7)
# --------------------------------------------------------------------------- #
def test_runtime_metadata_shapes_and_roles(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, _ = raf_prepare.group_captures(index)
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    poses, groups_meta = raf_prepare.build_runtime_metadata(index, groups, split)

    # only captures of SELECTED groups are runtime-visible
    assert len(poses) == 3 * N_MICS
    assert set(groups_meta) == set(split["train_test_groups"] + split["val_groups"])

    entry = poses["000000"]
    assert set(entry) == {"tx_xyz_p", "quat_raw", "rx_p", "group_key", "split_role"}
    assert len(entry["tx_xyz_p"]) == 3 and len(entry["rx_p"]) == 3
    assert len(entry["quat_raw"]) == 4
    assert entry["tx_xyz_p"] == [0.0, 0.0, 1.5]           # RAF (0, 1.5, 0) -> (X, Z, Y)
    assert entry["rx_p"] == [2.0, -1.0, 0.0]              # RAF (2, 0, -1)  -> (X, Z, Y)
    assert entry["group_key"] == groups[0]["group_key"]
    assert entry["split_role"] in {"train", "test", "val"}
    assert all(isinstance(k, str) and len(k) == 6 for k in poses)

    gm = groups_meta[groups[0]["group_key"]]
    assert set(gm) == {"tx_xyz_p", "depth_file", "train_ids", "role"}
    assert gm["depth_file"] == f"{groups[0]['group_key']}_depth_image.npy"
    assert len(gm["train_ids"]) == 12
    assert gm["role"] == "train_test"


def test_runtime_metadata_roles_agree_with_the_split(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, _ = raf_prepare.group_captures(index)
    split = raf_prepare.select_splits(groups, n_groups=2, n_val_groups=1, n_train=12)
    poses, _ = raf_prepare.build_runtime_metadata(index, groups, split)
    for gk, ids in split["train_ids"].items():
        assert all(poses[c]["split_role"] == "train" for c in ids)
    for gk, ids in split["test_ids"].items():
        assert all(poses[c]["split_role"] == "test" for c in ids)
    for gk, ids in split["val_ids"].items():
        assert all(poses[c]["split_role"] == "val" for c in ids)


def test_runtime_metadata_is_json_serialisable(mini_room):
    index = raf_prepare.load_room_index(mini_room)
    groups, _ = raf_prepare.group_captures(index)
    split = raf_prepare.select_splits(groups, n_groups=1, n_val_groups=1, n_train=12)
    poses, groups_meta = raf_prepare.build_runtime_metadata(index, groups, split)
    json.dumps({"poses": poses, "groups": groups_meta})


# --------------------------------------------------------------------------- #
# CLI (cycle 7)
# --------------------------------------------------------------------------- #
def _run_cli(tmp_path, extra=()):
    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    write_room(str(raf_root), "FurnishedRoom")
    out = tmp_path / "runtime" / "RAF"
    split_dir = tmp_path / "splits"
    argv = ["--raf-root", str(raf_root), "--output-dir", str(out),
            "--split-dir", str(split_dir), "--rooms", "EmptyRoom", "FurnishedRoom",
            "--n-groups", "2", "--n-val-groups", "1", "--n-train", "12",
            "--full-crosscheck"] + list(extra)
    raf_prepare.main(argv)
    return out, split_dir


def test_cli_emits_every_artifact(tmp_path):
    out, split_dir = _run_cli(tmp_path)
    for name in ("train_base.json", "val_base.json", "test_base.json",
                 "raf_splits_record.json", "raf_amplitude_audit.json"):
        assert (split_dir / name).exists(), name
    for room in ("EmptyRoom", "FurnishedRoom"):
        assert (out / room / "metadata" / "poses_metadata.json").exists()
        assert (out / room / "metadata" / "groups_metadata.json").exists()
        wavs = sorted((out / room / "mono_rirs_22050Hz").glob("*.wav"))
        assert len(wavs) == 3 * N_MICS   # all captures of the 3 selected groups


def test_cli_split_files_cover_both_rooms_and_match_the_wavs(tmp_path):
    out, split_dir = _run_cli(tmp_path)
    with open(split_dir / "train_base.json") as f:
        train = json.load(f)
    with open(split_dir / "test_base.json") as f:
        test = json.load(f)
    with open(split_dir / "val_base.json") as f:
        val = json.load(f)
    assert set(train) == set(test) == set(val) == {"EmptyRoom", "FurnishedRoom"}
    assert len(train["EmptyRoom"]) == 24 and len(test["EmptyRoom"]) == 48
    assert len(val["EmptyRoom"]) == 36
    for room in ("EmptyRoom", "FurnishedRoom"):
        for name in train[room] + test[room] + val[room]:
            assert (out / room / "mono_rirs_22050Hz" / name).exists()


def test_cli_runtime_metadata_covers_every_split_item(tmp_path):
    out, split_dir = _run_cli(tmp_path)
    with open(split_dir / "test_base.json") as f:
        test = json.load(f)
    with open(out / "EmptyRoom" / "metadata" / "poses_metadata.json") as f:
        poses = json.load(f)
    with open(out / "EmptyRoom" / "metadata" / "groups_metadata.json") as f:
        groups_meta = json.load(f)
    for name in test["EmptyRoom"]:
        cid = name[:-4]
        assert cid in poses
        gk = poses[cid]["group_key"]
        assert gk in groups_meta
        assert cid not in groups_meta[gk]["train_ids"]   # test items are not support


def test_cli_still_aborts_on_a_count_mismatch_the_sentinel_rule_does_not_cover(tmp_path):
    """Amendment 1 licenses ONE trailing all-NaN rx line and nothing else: two of
    them still stop the CLI before anything is written."""
    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", extra_rx_lines=("nan,nan,nan", "nan,nan,nan"))
    with pytest.raises(ValueError):
        raf_prepare.main(["--raf-root", str(raf_root),
                          "--output-dir", str(tmp_path / "out"),
                          "--split-dir", str(tmp_path / "splits"),
                          "--rooms", "EmptyRoom",
                          "--n-groups", "2", "--n-val-groups", "1"])


def test_cli_is_idempotent(tmp_path):
    out, split_dir = _run_cli(tmp_path)
    with open(split_dir / "train_base.json") as f:
        first = json.load(f)
    raf_root = tmp_path / "raf"
    raf_prepare.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                      "--split-dir", str(split_dir), "--rooms", "EmptyRoom",
                      "FurnishedRoom", "--n-groups", "2", "--n-val-groups", "1",
                      "--n-train", "12", "--full-crosscheck"])
    with open(split_dir / "train_base.json") as f:
        assert json.load(f) == first


# --------------------------------------------------------------------------- #
# trailing-sentinel provenance end to end (cycle 13)
# --------------------------------------------------------------------------- #
def test_splits_record_carries_the_sentinel_flag(mini_room):
    payload = _one_room_payload(mini_room)
    record = raf_prepare.build_splits_record(payload, {"seed": 0})
    assert record["rooms"]["EmptyRoom"]["rx_trailing_sentinel_dropped"] is False


def test_splits_record_requires_the_sentinel_flag(mini_room):
    """Fail-closed: a payload that never observed the pose files cannot claim
    'no sentinel dropped' by omission."""
    payload = _one_room_payload(mini_room)
    del payload["EmptyRoom"]["rx_trailing_sentinel_dropped"]
    with pytest.raises(KeyError):
        raf_prepare.build_splits_record(payload, {"seed": 0})


def test_cli_records_the_sentinel_drop_per_room(tmp_path):
    """One room ships the sentinel, the other does not: the record must say so
    per room, since it is the only evidence a line was dropped at all."""
    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", extra_rx_lines=("nan,nan,nan",))
    write_room(str(raf_root), "FurnishedRoom")
    split_dir = tmp_path / "splits"
    raf_prepare.main(["--raf-root", str(raf_root),
                      "--output-dir", str(tmp_path / "runtime" / "RAF"),
                      "--split-dir", str(split_dir),
                      "--rooms", "EmptyRoom", "FurnishedRoom",
                      "--n-groups", "2", "--n-val-groups", "1", "--n-train", "12",
                      "--full-crosscheck"])
    with open(split_dir / "raf_splits_record.json") as f:
        record = json.load(f)
    assert record["rooms"]["EmptyRoom"]["rx_trailing_sentinel_dropped"] is True
    assert record["rooms"]["FurnishedRoom"]["rx_trailing_sentinel_dropped"] is False
    # the sentinel room still has every capture: the dropped line was not data
    assert record["rooms"]["EmptyRoom"]["group_report"]["n_captures"] == 3 * N_MICS
