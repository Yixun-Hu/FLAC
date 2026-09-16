"""The contract that must never drift (exp_14 round E, plan §3 "Amendment 1").

``src/tools/make_ar_train_subsets.sampler_candidates`` decides which entries of a room can
serve as another entry's acoustic context, and the whole data curve rests on that answer
being the *pinned sampler's* answer: the subset generator uses it to decide starvation and
what to top up, the verifier uses it to re-check the committed splits, and at training time
``AR_md.get_ir_and_location_for_other_sources`` is what actually draws. If the two ever
disagree, a split that looks healthy raises ``DatasetContractError`` mid-run (or, worse,
silently trains on a different distribution than the manifest describes).

This file pins them together from both sides:

* ``_pinned_valid_other_src_ir_paths`` is the candidate-building block copied **verbatim**
  out of ``AR_md.py`` at commit ``7bbd8aa`` — the sampler the 100 % anchors trained with;
* ``_live_candidates`` runs the *current* ``AR_md.get_ir_and_location_for_other_sources`` on
  a real directory tree and captures the list it hands to ``np.random.choice``.

Both are compared against ``sampler_candidates`` on trees that make the difference between
the sampler's rule and raw token equality visible — above all the ``S001…S010`` room shape
of the 111 AR training rooms, where node 10 is rebuilt as ``S0010`` and so is unreachable.

The tests compare candidate *sets*: the sampler iterates ``list(set_of_ints)``, whose order
CPython does not promise, while ``sampler_candidates`` returns ascending node order. Order
is irrelevant to eligibility (the only thing this tool asks) and is asserted only as
"a permutation of the sampler's list".
"""
import importlib.util
import json
import os

import numpy as np
import pytest

from src.data.dataset import DatasetContractError
from src.tools.make_ar_train_subsets import sampler_candidates, sampler_node, parse_nodes

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AR_MD_PATH = os.path.join(
    REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata", "AR_md.py"
)
AR_ROOT = os.path.join(REPO_ROOT, "AcousticRooms", "single_channel_ir_1")
TRAIN_JSON = os.path.join(REPO_ROOT, "data", "AR", "train.json")


# ======================================================================================
# Reference 1 — the pinned sampler's candidate block, copied verbatim from 7bbd8aa
# ======================================================================================
def _pinned_valid_other_src_ir_paths(ir_file_path):
    """Lines 91-102 of ``AR_md.py`` @ ``7bbd8aa``, unmodified except for the return.

    Do not "clean this up": its value is that it is character-for-character the code that
    ran during the 100 % anchor trainings.
    """
    dir_name = os.path.dirname(ir_file_path)
    ir_file_name = ir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    all_src_node = set([int(fn.split("_")[0][1:]) for fn in os.listdir(dir_name)])
    remain_src_node = list(all_src_node.difference(set([src_node])))
    valid_other_src_ir_paths = []
    for node in remain_src_node:
        rec_n = ir_file_name.split("_")[1]
        src_n = f"S00{node}"
        other_src_ir_path = os.path.join(dir_name, f"{src_n}_{rec_n}_hybrid_IR.wav")
        if os.path.exists(other_src_ir_path):
            valid_other_src_ir_paths.append(other_src_ir_path)
    return valid_other_src_ir_paths


# ======================================================================================
# Reference 2 — the LIVE hook, caught in the act of handing its pool to np.random.choice
# ======================================================================================
class _Captured(Exception):
    """Raised by the stubbed ``np.random.choice`` once its argument has been recorded."""


def load_ar_md():
    spec = importlib.util.spec_from_file_location("ar_md_equivalence_probe", AR_MD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _live_candidates(monkeypatch, ir_file_path, allowed_basenames=None):
    """The pool the *current* hook would draw from, without loading a single wav.

    The pinned body wraps the ``replace=False`` draw in ``except Exception`` and retries with
    ``replace=True``, so the stub is hit twice with the same list; the second raise escapes.
    Asserting that shape here is deliberate — it pins the retry branch too.
    """
    ar_md = load_ar_md()
    seen = []

    def fake_choice(pool, size, replace=False):
        seen.append(list(pool))
        raise _Captured()

    monkeypatch.setattr(np.random, "choice", fake_choice)
    with pytest.raises(_Captured):
        ar_md.get_ir_and_location_for_other_sources(
            ir_file_path, 8, "unused", allowed_basenames=allowed_basenames
        )
    assert len(seen) == 2 and seen[0] == seen[1]   # replace=False then replace=True
    return seen[0]


def _basenames(paths):
    return [os.path.basename(p) for p in paths]


def _make_room(tmp_path, basenames, room="Room_idx_0"):
    room_dir = tmp_path / "AcousticRooms" / "Scene" / room
    room_dir.mkdir(parents=True)
    for name in basenames:
        (room_dir / name).write_bytes(b"")     # existence is all the candidate block reads
    return room_dir


def _assert_agrees(monkeypatch, room_dir, room_files, target):
    """``sampler_candidates`` == the pinned block == the live hook, for one target."""
    path = str(room_dir / target)
    pinned = _basenames(_pinned_valid_other_src_ir_paths(path))
    live = _basenames(_live_candidates(monkeypatch, path))
    ours = sampler_candidates(target, room_files)
    assert set(ours) == set(pinned) == set(live)
    assert sorted(ours) == sorted(pinned)                       # no duplicates either way
    assert ours == sorted(ours, key=lambda f: sampler_node(parse_nodes(f)[0]))
    return ours


# ======================================================================================
# The S001…S010 room shape — 111 of the 243 AR training rooms
# ======================================================================================
TEN_SOURCE = [f"S{s:03d}_R{r:03d}_hybrid_IR.wav" for r in (1, 2) for s in range(1, 11)]


def test_ten_source_room_agrees_and_source_ten_is_unreachable(tmp_path, monkeypatch):
    room_dir = _make_room(tmp_path, TEN_SOURCE)
    got = _assert_agrees(monkeypatch, room_dir, TEN_SOURCE, "S007_R001_hybrid_IR.wav")
    assert "S010_R001_hybrid_IR.wav" not in got      # the whole point: node 10 -> "S0010"
    assert len(got) == 8


def test_the_unreachable_source_is_still_a_valid_target(tmp_path, monkeypatch):
    room_dir = _make_room(tmp_path, TEN_SOURCE)
    got = _assert_agrees(monkeypatch, room_dir, TEN_SOURCE, "S010_R001_hybrid_IR.wav")
    assert len(got) == 9


@pytest.mark.parametrize("target", [f"S{s:03d}_R{r:03d}_hybrid_IR.wav"
                                    for s in range(1, 11) for r in (1, 2)])
def test_every_target_of_a_ten_source_room_agrees(tmp_path, monkeypatch, target):
    room_dir = _make_room(tmp_path, TEN_SOURCE)
    _assert_agrees(monkeypatch, room_dir, TEN_SOURCE, target)


# ======================================================================================
# Shapes that stress the reconstruction itself
# ======================================================================================
def test_the_S00n_spelling_of_the_normal_rooms_agrees(tmp_path, monkeypatch):
    """The 132 rooms that spell sources ``S000, S002, …, S0039`` — the spelling the sampler
    rebuilds exactly, so everything there IS reachable."""
    files = [f"S00{s}_R00{r}_hybrid_IR.wav" for r in (0, 11) for s in (0, 2, 7, 11, 39)]
    room_dir = _make_room(tmp_path, files)
    got = _assert_agrees(monkeypatch, room_dir, files, "S007_R0011_hybrid_IR.wav")
    assert len(got) == 4                                   # every other source is reachable


def test_two_tokens_sharing_one_integer_node_agree(tmp_path, monkeypatch):
    """``S012`` and ``S0012`` collapse to node 12 for the sampler; only the rebuilt spelling
    is reachable, and a target of that node has neither as a candidate."""
    files = ["S012_R001_hybrid_IR.wav", "S0012_R001_hybrid_IR.wav", "S003_R001_hybrid_IR.wav"]
    room_dir = _make_room(tmp_path, files)
    for target in files:
        _assert_agrees(monkeypatch, room_dir, files, target)
    assert sampler_candidates("S003_R001_hybrid_IR.wav", files) == ["S0012_R001_hybrid_IR.wav"]


def test_a_foreign_tail_contributes_a_node_but_is_never_a_candidate(tmp_path, monkeypatch):
    """The node universe comes from *every* file in the room, but only ``…_hybrid_IR.wav``
    can be drawn — so a stray tail can add a node whose rebuild does not exist."""
    files = ["S001_R001_hybrid_IR.wav", "S004_R001_other_IR.wav", "S005_R001_hybrid_IR.wav"]
    room_dir = _make_room(tmp_path, files)
    got = _assert_agrees(monkeypatch, room_dir, files, "S001_R001_hybrid_IR.wav")
    assert got == ["S005_R001_hybrid_IR.wav"]


def test_a_lone_receiver_has_an_empty_pool_on_both_sides(tmp_path, monkeypatch):
    files = ["S001_R001_hybrid_IR.wav", "S010_R001_hybrid_IR.wav"]
    room_dir = _make_room(tmp_path, files)
    assert _assert_agrees(monkeypatch, room_dir, files, "S001_R001_hybrid_IR.wav") == []


# ======================================================================================
# The restricted path: the split filter is applied AFTER the reconstruction
# ======================================================================================
def test_restriction_cannot_resurrect_an_unreachable_token(tmp_path, monkeypatch):
    """A split may list ``S010_R001`` as ``S007_R001``'s only company; the sampler still
    cannot reach it, and answers with the fatal ``DatasetContractError``. This is exactly
    the state the generator now calls *starved*."""
    room_dir = _make_room(tmp_path, TEN_SOURCE)
    ar_md = load_ar_md()
    allowed = frozenset({"S007_R001_hybrid_IR.wav", "S010_R001_hybrid_IR.wav"})
    assert sampler_candidates("S007_R001_hybrid_IR.wav", allowed) == []
    with pytest.raises(DatasetContractError):
        ar_md.get_ir_and_location_for_other_sources(
            str(room_dir / "S007_R001_hybrid_IR.wav"), 8, "unused", allowed_basenames=allowed
        )


def test_restricted_pool_equals_our_candidates_intersected_with_the_split(tmp_path, monkeypatch):
    room_dir = _make_room(tmp_path, TEN_SOURCE)
    allowed = frozenset(f"S{s:03d}_R001_hybrid_IR.wav" for s in (2, 5, 7, 10))
    live = _basenames(_live_candidates(
        monkeypatch, str(room_dir / "S007_R001_hybrid_IR.wav"), allowed_basenames=allowed))
    assert set(live) == set(sampler_candidates("S007_R001_hybrid_IR.wav", TEN_SOURCE)) & allowed
    assert set(live) == {"S002_R001_hybrid_IR.wav", "S005_R001_hybrid_IR.wav"}
    # and it is what judging against the retained room alone gives, which is what the
    # generator's histogram does
    assert set(live) == set(sampler_candidates("S007_R001_hybrid_IR.wav", allowed))


# ======================================================================================
# The real dataset, when it is on this box
# ======================================================================================
def _real_rooms():
    with open(TRAIN_JSON) as fin:
        split = json.load(fin)
    unreachable, normal = None, None
    for scene in sorted(split):
        for room in sorted(split[scene]):
            files = split[scene][room]
            tokens = {parse_nodes(f)[0] for f in files}
            broken = any(tok != f"S00{sampler_node(tok)}" for tok in tokens)
            if broken and unreachable is None:
                unreachable = (scene, room, files)
            if not broken and normal is None:
                normal = (scene, room, files)
    return unreachable, normal


@pytest.mark.parametrize("which", [0, 1])
def test_real_ar_rooms_agree_with_the_pinned_sampler(monkeypatch, which):
    """One real room of each spelling, straight out of ``train.json``. 111 of the 243 AR
    training rooms carry a source token the sampler cannot rebuild; the other 132 do not."""
    if not os.path.isdir(AR_ROOT) or not os.path.isfile(TRAIN_JSON):
        pytest.skip("AcousticRooms / train.json not available on this box")
    picked = _real_rooms()[which]
    assert picked is not None, "train.json should contain rooms of both spellings"
    scene, room, files = picked
    room_dir = os.path.join(AR_ROOT, scene, room)
    receivers = sorted({parse_nodes(f)[1] for f in files})
    targets = [f for f in sorted(files) if parse_nodes(f)[1] in receivers[:2]][:12]
    assert targets
    for target in targets:
        path = os.path.join(room_dir, target)
        pinned = set(_basenames(_pinned_valid_other_src_ir_paths(path)))
        live = set(_basenames(_live_candidates(monkeypatch, path)))
        assert set(sampler_candidates(target, files)) == pinned == live


def _first_room_with_extras(broken_spelling):
    """The first room of the requested spelling whose directory holds MORE files than
    ``train.json`` lists (114 of the 243 do)."""
    with open(TRAIN_JSON) as fin:
        split = json.load(fin)
    for scene in sorted(split):
        for room in sorted(split[scene]):
            files = split[scene][room]
            tokens = {parse_nodes(f)[0] for f in files}
            if any(tok != f"S00{sampler_node(tok)}" for tok in tokens) != broken_spelling:
                continue
            on_disk = os.listdir(os.path.join(AR_ROOT, scene, room))
            if len(on_disk) > len(files):
                return scene, room, files, on_disk
    return None


@pytest.mark.parametrize("broken_spelling", [True, False])
def test_the_train_json_room_list_is_the_right_node_universe(monkeypatch, broken_spelling):
    """``sampler_candidates`` is given ``train.json``'s room list, while the sampler's own
    universe is ``os.listdir`` of the room directory — a strict superset in 114 rooms.

    They still agree at every *training* receiver, which is the only place this experiment
    asks: a rebuilt name has to exist on disk AND lie in the split to be usable, and the
    receiver-level anchor audit (plan §3 D3) established that no out-of-split file sits at a
    receiver ``train.json`` trains on. This test is that argument, executed."""
    if not os.path.isdir(AR_ROOT) or not os.path.isfile(TRAIN_JSON):
        pytest.skip("AcousticRooms / train.json not available on this box")
    picked = _first_room_with_extras(broken_spelling)
    assert picked is not None, "expected rooms of both spellings to have extra on-disk files"
    scene, room, files, on_disk = picked
    assert len(on_disk) > len(files)                      # the fixture is meaningful
    room_dir = os.path.join(AR_ROOT, scene, room)
    receivers = sorted({parse_nodes(f)[1] for f in files})
    targets = [f for f in sorted(files) if parse_nodes(f)[1] in receivers[:2]][:12]
    assert targets
    for target in targets:
        pinned = set(_basenames(_pinned_valid_other_src_ir_paths(os.path.join(room_dir, target))))
        assert set(sampler_candidates(target, files)) == pinned
        assert pinned <= set(files)                       # nothing out-of-split was reachable
