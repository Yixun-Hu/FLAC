"""Tests for ``data/RAF/prepare_data.py`` — RAF -> FLAC-runtime preparation.

exp_19 (RAF finetune), contract section B. TDD cycles:

* cycle 4 — ``load_room_index`` + per-capture cross-check
* cycle 5 — ``group_captures`` (canonical 7-tuple key, exactly-36 invariant) and
  placement clustering
* cycle 6 — ``select_splits``, split-JSON emission, splits record
* cycle 7 — ``resample_and_write`` + amplitude audit, runtime metadata, CLI

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
    """A rigid 36-point array: 3 x 3 x 4 lattice spanning 1.0 x 1.0 x 1.2 m (RAF axes)."""
    pts = []
    for x in (0.0, 0.5, 1.0):
        for y in (0.0, 0.5, 1.0):
            for z in (0.0, 0.4, 0.8, 1.2):
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


def test_load_room_index_aborts_on_the_observed_trailing_nan_rx_line(tmp_path):
    """The real corpus ships one extra ``nan,nan,nan`` line in all_rx_pos.txt.

    Fail-closed by contract (plan Rev 2 section 2): the off-by-one is resolved by a
    human at the readback rung, never silently absorbed here.
    """
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1),
               extra_rx_lines=("nan,nan,nan",))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    with pytest.raises(ValueError) as exc:
        raf_prepare.load_room_index(room_dir)
    msg = str(exc.value)
    assert "all_rx_pos.txt" in msg
    assert str(N_MICS + 1) in msg and str(N_MICS) in msg


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
