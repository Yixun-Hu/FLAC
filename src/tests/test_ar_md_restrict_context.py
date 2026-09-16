"""Tests for exp_14 round B — restricting the acoustic-context pool to the training split.

The data-curve experiment trains FLAC on 25 / 50 / 75 % subsets of ``data/AR/train.json``.
"25 % data" is only honest if the withheld 75 % is invisible in *every* role, so the
K-reference context sampler in ``src/configs/dataset_configs/custom_metadata/AR_md.py``
(which draws from ``os.listdir`` of the room directory, i.e. from the whole on-disk
dataset) must be restricted to the basenames the split actually lists. Two contracts
follow, and this file is their executable statement (plan_data_curve.md §4 "Round B",
test ids B1-B8):

* **Backwards compatibility.** With ``restrict_to_split`` absent/false the sampler must be
  *bitwise* the pinned implementation (commit ``7bbd8aa``) under the same numpy seed — the
  100 % anchors (exp07_P1@40k, exp09_cylNoSSL@40k) were trained with it and stay valid
  anchors only if the default path is untouched (B1).
* **Fail-closed.** A restricted pool that comes out empty, or a room the split does not
  list, is a *dataset-contract* violation, not a bad file: ``SampleDataset.__getitem__``
  answers every exception by silently returning a different random sample, which would
  turn the violation into a quiet change of the sampling distribution. Contract violations
  therefore raise ``DatasetContractError``, which the generic fallback must let through,
  all the way out of the worker processes and out of the training process (B3, B5, B7, B8).

Fixtures are hermetic: a toy ``AcousticRooms`` tree in ``tmp_path`` (the directory name
must contain "AcousticRooms" — ``json_scandir`` asserts it), silent-free 22.05 kHz mono
IRs carrying a unique impulse marker so a drawn context can be identified bitwise, the
metadata JSONs ``get_receiver_source_location`` expects, and 256x512 depth maps.
"""
import importlib.util
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import torchaudio

from src.data.dataset import (
    DatasetContractError,
    LocalDatasetConfig,
    SampleDataset,
    create_dataloader_from_config,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AR_MD_PATH = os.path.join(
    REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata", "AR_md.py"
)

SCENE = "ToyScene"
SR = 22050
IR_LEN = 9600


# ======================================================================================
# Fixture helpers — toy AcousticRooms tree
# ======================================================================================
def load_ar_md():
    """Load ``AR_md.py`` exactly as ``create_dataloader_from_config`` does (by file path)."""
    spec = importlib.util.spec_from_file_location("metadata_module_under_test", AR_MD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ar_md():
    return load_ar_md()


def ir_basename(src, rec):
    return f"S00{src}_R00{rec}_hybrid_IR.wav"


def marker_index(room_i, src, rec):
    """Unique (room, source, receiver) -> impulse position, well inside ``IR_LEN``."""
    return 100 * room_i + 10 * src + rec


def build_toy_tree(root, rooms=("ToyScene_idx_0",), sources=(1, 2, 3), receivers=(1, 2)):
    """Write a toy AcousticRooms tree; return the paths + the marker <-> file mapping."""
    dataset_dir = os.path.join(str(root), "AcousticRooms")
    marker_of = {}
    file_of_marker = {}
    for room_i, room in enumerate(rooms):
        ir_dir = os.path.join(dataset_dir, "single_channel_ir_1", SCENE, room)
        md_dir = os.path.join(dataset_dir, "metadata", SCENE, room)
        depth_dir = os.path.join(dataset_dir, "depth_map", SCENE, room)
        for d in (ir_dir, md_dir, depth_dir):
            os.makedirs(d, exist_ok=True)
        for rec in receivers:
            np.save(os.path.join(depth_dir, f"{rec}.npy"), np.ones((256, 512), dtype=np.float32))
            for src in sources:
                marker = marker_index(room_i, src, rec)
                wav = torch.zeros(1, IR_LEN)
                wav[0, marker] = 1.0
                torchaudio.save(os.path.join(ir_dir, ir_basename(src, rec)), wav, SR)
                with open(os.path.join(md_dir, f"S00{src}_R00{rec}.json"), "w") as fout:
                    json.dump({"src_loc": [float(src), float(rec), 0.0],
                               "rec_loc": [0.0, 0.0, 0.0]}, fout)
                marker_of[(room, ir_basename(src, rec))] = marker
                file_of_marker[marker] = (room, ir_basename(src, rec))
    splits_dir = os.path.join(str(root), "splits")
    os.makedirs(splits_dir, exist_ok=True)
    return {
        "root": str(root),
        "dataset_dir": dataset_dir,
        "splits_dir": splits_dir,
        "rooms": list(rooms),
        "sources": list(sources),
        "receivers": list(receivers),
        "marker_of": marker_of,
        "file_of_marker": file_of_marker,
    }


def ir_path(tree, room, src, rec):
    return os.path.join(tree["dataset_dir"], "single_channel_ir_1", SCENE, room,
                        ir_basename(src, rec))


def metadata_dir(tree):
    return os.path.join(tree["dataset_dir"], "metadata")


def write_split(tree, name, rooms_to_files):
    """Write ``{SCENE: {room: [basename, ...]}}`` and return its path."""
    path = os.path.join(tree["splits_dir"], name)
    with open(path, "w") as fout:
        json.dump({SCENE: {room: list(files) for room, files in rooms_to_files.items()}}, fout)
    return path


def full_split(tree, rooms=None):
    rooms = tree["rooms"] if rooms is None else rooms
    return {room: [ir_basename(s, r) for r in tree["receivers"] for s in tree["sources"]]
            for room in rooms}


def modalities(max_context=8, restrict=None, max_len=IR_LEN):
    acoustic = {"load": True, "max_context": max_context, "max_len": max_len}
    if restrict is not None:
        acoustic["restrict_to_split"] = restrict
    return {"acoustic_context": acoustic, "depth": {"load": True}, "poses": {"load": True}}


def make_sample_dataset(tree, split_path, custom_metadata_fn, conditioning=None):
    config = LocalDatasetConfig(
        id="AcousticRooms",
        path=tree["dataset_dir"],
        custom_metadata_fn=custom_metadata_fn,
        json_file_path=split_path,
        folder_name="single_channel_ir_1",
        conditioning=conditioning if conditioning is not None else modalities(),
    )
    return SampleDataset([config], sample_size=10240, sample_rate=SR, augs=False)


@pytest.fixture
def tree(tmp_path):
    return build_toy_tree(tmp_path)


# ======================================================================================
# B5 — SampleDataset.__getitem__ : contract violations terminate, other failures resample
# ======================================================================================
def test_B5_dataset_contract_error_propagates_without_resampling(tree):
    split_path = write_split(tree, "full.json", full_split(tree))
    seen = []

    def metadata_fn(info, audio):
        seen.append(info["idx"])
        raise DatasetContractError("no in-split context for <toy>")

    dataset = make_sample_dataset(tree, split_path, metadata_fn)
    assert len(dataset) == 6

    with pytest.raises(DatasetContractError):
        dataset[0]

    # __getitem__ was entered exactly once: no replacement sample was ever drawn.
    assert seen == [0]


def test_B5_plain_exception_still_resamples(tree, capsys):
    split_path = write_split(tree, "full.json", full_split(tree))
    seen = []

    def metadata_fn(info, audio):
        seen.append(info["idx"])
        if info["idx"] == 0:
            raise RuntimeError("transient read failure")
        return {"scene": SCENE}

    dataset = make_sample_dataset(tree, split_path, metadata_fn)
    random.seed(20260916)
    audio, info = dataset[0]

    assert audio.shape == (1, 10240)
    assert seen[0] == 0
    assert len(seen) > 1 and seen[-1] != 0, "the pre-existing resample fallback must survive"
    assert "Couldn't load file" in capsys.readouterr().out


# ======================================================================================
# The pinned implementation (commit 7bbd8aa), copied VERBATIM — B1's reference.
# Only the sampler's name changes (`_pinned_reference`); the two helpers it calls are
# copied verbatim too, so the reference is independent of any later edit to AR_md.py.
# Trailing whitespace on three blank lines was normalized away (git diff --check): what
# B1 pins is semantic equivalence — the returned tensors and the numpy RNG state — not
# the whitespace of the source.
# ======================================================================================
def get_3d_point_camera_coord(source_pose, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = source_pose[0], source_pose[1], source_pose[2]
    camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]

def get_receiver_source_location(ir_file_path, metadata_path):
    scene_name = ir_file_path.split("/")[-3]
    scene_id = ir_file_path.split("/")[-2]
    ir_file_name = ir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
    metadata_file_path = os.path.join(metadata_path, scene_name, scene_id, json_file_name)
    with open(metadata_file_path, "r") as fin:
        meta_info = json.load(fin)
    src_loc = meta_info["src_loc"]
    rec_loc = meta_info["rec_loc"]
    return src_loc, rec_loc

def _pinned_reference(ir_file_path, num_ref_sources, metadata_path, max_len=9600):
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
    try:
        select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=False)
    except Exception as e:
        select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=True)
    all_ref_irs = []
    all_ref_src_pos = []

    for fp in select_other_src_ir_paths:
        ref_wav, rate = torchaudio.load(fp)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if ref_wav.shape[1] < max_len:
            ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])], dim=1)
        else:
            ref_wav = ref_wav[:, :max_len]
        ref_wav = ref_wav.unsqueeze(0) # C=1
        all_ref_irs.append(ref_wav)

        src_loc, rec_loc = get_receiver_source_location(fp, metadata_path=metadata_path)

        proj_src_loc = get_3d_point_camera_coord(rec_loc, src_loc)

        all_ref_src_pos.append(torch.Tensor(proj_src_loc).float())
    all_ref_irs = torch.cat(all_ref_irs, dim=0)
    all_ref_src_pos = torch.vstack(all_ref_src_pos)
    return all_ref_irs, all_ref_src_pos


def assert_identical_rng_state(actual, expected):
    """The numpy RNG must be left in EXACTLY the state the pinned implementation leaves.

    Equal return values are not enough: an implementation that consumed one extra draw
    after the selection would still return the pinned tensors for a single call, but
    every later sample of that epoch would diverge from the pinned stream.
    """
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if isinstance(want, np.ndarray):
            assert isinstance(got, np.ndarray)
            assert got.dtype == want.dtype
            assert np.array_equal(got, want)
        else:
            assert type(got) is type(want)
            assert got == want


def drawn_markers(ref_irs):
    """Impulse position of every drawn context IR (shape [N, 1, max_len]) -> its identity."""
    assert ref_irs.dim() == 3 and ref_irs.shape[1] == 1
    return [int(torch.argmax(row[0])) for row in ref_irs]


# ======================================================================================
# B1 — allowed_basenames=None is bitwise the pinned implementation, seed for seed
# ======================================================================================
# num_ref_sources=2 with 2 candidates -> replace=False branch;
# num_ref_sources=8 with 2 candidates -> the ValueError is caught -> replace=True branch.
@pytest.mark.parametrize("num_ref_sources", [2, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("receiver", [1, 2])
def test_B1_default_path_is_bitwise_the_pinned_implementation(
    tree, ar_md, num_ref_sources, seed, receiver
):
    target = ir_path(tree, tree["rooms"][0], 1, receiver)

    np.random.seed(seed)
    ref_irs, ref_pos = _pinned_reference(
        target, num_ref_sources, metadata_dir(tree), max_len=IR_LEN
    )
    reference_state = np.random.get_state()

    np.random.seed(seed)
    new_irs, new_pos = ar_md.get_ir_and_location_for_other_sources(
        target,
        num_ref_sources=num_ref_sources,
        metadata_path=metadata_dir(tree),
        max_len=IR_LEN,
        allowed_basenames=None,
    )
    new_state = np.random.get_state()

    assert torch.equal(new_irs, ref_irs)
    assert torch.equal(new_pos, ref_pos)
    assert_identical_rng_state(new_state, reference_state)
    assert new_irs.shape == (num_ref_sources, 1, IR_LEN)
    assert new_pos.shape == (num_ref_sources, 3)
    if num_ref_sources > 2:  # the replacement branch really was the one exercised
        assert len(set(drawn_markers(new_irs))) < num_ref_sources


def test_B1_default_path_is_the_default_argument(tree, ar_md):
    """Existing call sites pass no allowed_basenames at all."""
    target = ir_path(tree, tree["rooms"][0], 2, 1)

    np.random.seed(7)
    ref_irs, ref_pos = _pinned_reference(target, 8, metadata_dir(tree), max_len=IR_LEN)
    reference_state = np.random.get_state()

    np.random.seed(7)
    new_irs, new_pos = ar_md.get_ir_and_location_for_other_sources(
        target, num_ref_sources=8, metadata_path=metadata_dir(tree), max_len=IR_LEN
    )
    new_state = np.random.get_state()

    assert torch.equal(new_irs, ref_irs)
    assert torch.equal(new_pos, ref_pos)
    assert_identical_rng_state(new_state, reference_state)


# ======================================================================================
# B2 — restricted pool: every draw is in-split, the target is never drawn
# ======================================================================================
def test_B2_restricted_pool_without_replacement(tree, ar_md):
    room = tree["rooms"][0]
    target = ir_path(tree, room, 1, 1)
    allowed = frozenset({ir_basename(2, 1), ir_basename(3, 1)})  # 2 of 3 sources, no target

    np.random.seed(0)
    irs, pos = ar_md.get_ir_and_location_for_other_sources(
        target,
        num_ref_sources=2,
        metadata_path=metadata_dir(tree),
        max_len=IR_LEN,
        allowed_basenames=allowed,
    )

    drawn = [tree["file_of_marker"][m] for m in drawn_markers(irs)]
    assert {b for _, b in drawn} == set(allowed)
    assert all(r == room for r, _ in drawn)
    assert ir_basename(1, 1) not in {b for _, b in drawn}


def test_B2_restricted_pool_smaller_than_k_uses_replacement(tree, ar_md):
    room = tree["rooms"][0]
    target = ir_path(tree, room, 1, 1)
    # 2 of the room's 3 sources are in-split, but one of them IS the target -> pool of 1.
    allowed = frozenset({ir_basename(1, 1), ir_basename(2, 1)})

    np.random.seed(0)
    irs, pos = ar_md.get_ir_and_location_for_other_sources(
        target,
        num_ref_sources=8,
        metadata_path=metadata_dir(tree),
        max_len=IR_LEN,
        allowed_basenames=allowed,
    )

    assert irs.shape == (8, 1, IR_LEN)
    drawn = [tree["file_of_marker"][m] for m in drawn_markers(irs)]
    assert {b for _, b in drawn} == {ir_basename(2, 1)}  # replacement branch, all in-split
    assert torch.equal(pos, torch.tensor([[2.0, 1.0, 0.0]]).repeat(8, 1))


def test_B2_restriction_ignores_out_of_split_and_other_receivers(tree, ar_md):
    """A plain `set` works too, and files of other receivers are never reachable anyway."""
    room = tree["rooms"][0]
    target = ir_path(tree, room, 3, 2)
    allowed = {ir_basename(1, 2), ir_basename(1, 1), ir_basename(2, 1)}

    np.random.seed(3)
    irs, _ = ar_md.get_ir_and_location_for_other_sources(
        target,
        num_ref_sources=4,
        metadata_path=metadata_dir(tree),
        max_len=IR_LEN,
        allowed_basenames=allowed,
    )

    drawn = [tree["file_of_marker"][m] for m in drawn_markers(irs)]
    assert {b for _, b in drawn} == {ir_basename(1, 2)}  # only in-split file at receiver 2


# ======================================================================================
# B3 — an empty restricted pool is a fatal contract violation
# ======================================================================================
@pytest.mark.parametrize(
    "allowed",
    [
        frozenset(),                              # nothing in split
        frozenset({"S001_R001_hybrid_IR.wav"}),   # only the target itself
        frozenset({"S002_R002_hybrid_IR.wav"}),   # only another receiver's file
    ],
)
def test_B3_empty_filtered_pool_raises_dataset_contract_error(tree, ar_md, allowed):
    target = ir_path(tree, tree["rooms"][0], 1, 1)

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md.get_ir_and_location_for_other_sources(
            target,
            num_ref_sources=8,
            metadata_path=metadata_dir(tree),
            max_len=IR_LEN,
            allowed_basenames=allowed,
        )

    assert "no in-split context" in str(excinfo.value)
    assert target in str(excinfo.value)
    assert isinstance(excinfo.value, RuntimeError)


# ======================================================================================
# B4 — get_custom_metadata: the flag, the cached split index, the missing-room contract
# ======================================================================================
def make_info(tree, room, src, rec, split_path, conditioning):
    return {
        "path": ir_path(tree, room, src, rec),
        "relpath": os.path.join("single_channel_ir_1", SCENE, room, ir_basename(src, rec)),
        "modalities": conditioning,
        "json_file_path": split_path,
    }


def test_B4_split_room_index_maps_scene_room_to_basenames(tree, ar_md):
    room = tree["rooms"][0]
    split_path = write_split(tree, "two.json", {room: [ir_basename(1, 1), ir_basename(2, 1)]})

    index = ar_md._split_room_index(os.path.realpath(split_path))

    assert index == {(SCENE, room): frozenset({ir_basename(1, 1), ir_basename(2, 1)})}
    assert isinstance(index[(SCENE, room)], frozenset)


@pytest.mark.parametrize("restrict", [None, False])
def test_B4_without_the_flag_the_index_is_never_touched(tree, ar_md, monkeypatch, restrict):
    room = tree["rooms"][0]
    split_path = write_split(tree, "full.json", full_split(tree))

    def never(*args, **kwargs):
        pytest.fail("the split index must not be consulted when restrict_to_split is off")

    monkeypatch.setattr(ar_md, "_split_room_index", never)
    monkeypatch.setattr(ar_md, "_load_split_room_index", never)

    np.random.seed(0)
    md = ar_md.get_custom_metadata(
        make_info(tree, room, 1, 1, split_path, modalities(restrict=restrict)), None
    )

    assert md["context_audio"].shape == (8, 1, IR_LEN)
    assert md["scene"] == SCENE
    assert md["depth"].shape == (3, 256, 512)


def test_B4_flag_restricts_the_drawn_contexts(tree, ar_md):
    room = tree["rooms"][0]
    # receiver 1 keeps sources 1 and 2 only; receiver 2 keeps everything.
    split_path = write_split(tree, "part.json", {room: [
        ir_basename(1, 1), ir_basename(2, 1),
        ir_basename(1, 2), ir_basename(2, 2), ir_basename(3, 2),
    ]})

    np.random.seed(0)
    md = ar_md.get_custom_metadata(
        make_info(tree, room, 1, 1, split_path, modalities(restrict=True)), None
    )

    drawn = [tree["file_of_marker"][m] for m in drawn_markers(md["context_audio"])]
    assert {b for _, b in drawn} == {ir_basename(2, 1)}  # 3 is out of split, 1 is the target


def test_B4_index_is_loaded_once_per_path_relative_or_absolute(tree, ar_md, monkeypatch):
    room = tree["rooms"][0]
    split_path = write_split(tree, "full.json", full_split(tree))
    loaded = []
    original_loader = ar_md._load_split_room_index

    def counting_loader(path):
        loaded.append(path)
        return original_loader(path)

    monkeypatch.setattr(ar_md, "_load_split_room_index", counting_loader)
    ar_md._split_room_index.cache_clear()

    for _ in range(2):
        np.random.seed(0)
        ar_md.get_custom_metadata(
            make_info(tree, room, 1, 1, split_path, modalities(restrict=True)), None
        )

    assert len(loaded) == 1, "lru_cache must keep the split JSON from being re-parsed"

    # The production config carries a RELATIVE split path; it must share the cache entry.
    monkeypatch.chdir(tree["root"])
    relative = os.path.join("splits", "full.json")
    assert not os.path.isabs(relative)
    np.random.seed(0)
    ar_md.get_custom_metadata(
        make_info(tree, room, 1, 1, relative, modalities(restrict=True)), None
    )

    assert len(loaded) == 1, "relative and absolute spellings must canonicalize to one entry"
    info = ar_md._split_room_index.cache_info()
    assert (info.misses, info.hits) == (1, 2)


def test_B4_room_missing_from_the_split_raises_dataset_contract_error(tree, ar_md):
    room = tree["rooms"][0]
    split_path = write_split(tree, "other_room.json", {"ToyScene_idx_99": [ir_basename(1, 1)]})

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md.get_custom_metadata(
            make_info(tree, room, 1, 1, split_path, modalities(restrict=True)), None
        )

    message = str(excinfo.value)
    assert "does not list room" in message
    assert f"{SCENE}/{room}" in message
    assert os.path.realpath(split_path) in message


def test_B4_empty_pool_under_the_flag_is_fatal(tree, ar_md):
    room = tree["rooms"][0]
    split_path = write_split(tree, "one_source.json", {room: [ir_basename(1, 1)]})

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md.get_custom_metadata(
            make_info(tree, room, 1, 1, split_path, modalities(restrict=True)), None
        )

    assert "no in-split context" in str(excinfo.value)


def test_B4_flag_without_a_split_path_is_fatal(tree, ar_md):
    """Fail closed rather than crashing inside realpath() and being resampled away."""
    room = tree["rooms"][0]
    info = make_info(tree, room, 1, 1, None, modalities(restrict=True))

    with pytest.raises(DatasetContractError):
        ar_md.get_custom_metadata(info, None)


# ======================================================================================
# B6-B8 — integration through the real DataLoader, worker and process boundaries
# ======================================================================================
def make_dataset_config(tree, split_path, restrict, max_context=8):
    """The shape of ``src/configs/dataset_configs/AR/train/acousticroom_train.json``."""
    return {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": "AcousticRooms",
                "path": tree["dataset_dir"],
                "json_file_path": split_path,
                "custom_metadata_module": AR_MD_PATH,
                "folder_name": "single_channel_ir_1",
            }
        ],
        "random_crop": False,
        "augs": False,
        "force_channels": "mono",
        "drop_last": False,
        "modalities": modalities(max_context=max_context, restrict=restrict),
    }


def build_loader(tree, split_path, restrict, num_workers=2, max_context=8):
    return create_dataloader_from_config(
        make_dataset_config(tree, split_path, restrict, max_context=max_context),
        batch_size=1,
        sample_size=10240,
        sample_rate=SR,
        audio_channels=1,
        num_workers=num_workers,
        shuffle=False,
    )


def count_index_loads(loader, tree, counter_path):
    """Wrap the split-index loader in the hook module the loader just exec'd (pre-fork)."""
    metadata_fn = loader.dataset.custom_metadata_fns[tree["dataset_dir"]]
    module_globals = metadata_fn.__globals__
    original_loader = module_globals["_load_split_room_index"]

    def counting_loader(path, _original=original_loader, _counter=counter_path):
        with open(_counter, "a") as fout:
            fout.write(f"{os.getpid()}\n")
        return _original(path)

    module_globals["_load_split_room_index"] = counting_loader


def forbid_resampling(monkeypatch, sentinel_path):
    """Record any call to the resample fallback's RNG (inherited by forked workers)."""
    original_randrange = random.randrange

    def recording_randrange(*args, _original=original_randrange, _sentinel=sentinel_path, **kwargs):
        with open(_sentinel, "a") as fout:
            fout.write(f"{os.getpid()}\n")
        return _original(*args, **kwargs)

    monkeypatch.setattr(random, "randrange", recording_randrange)


@pytest.fixture
def big_tree(tmp_path):
    """Two rooms, so both workers get batches."""
    return build_toy_tree(tmp_path, rooms=("ToyScene_idx_0", "ToyScene_idx_1"))


def test_B6_two_workers_two_epochs_only_draw_in_split_contexts(big_tree, tmp_path, capfd):
    split = {room: [ir_basename(s, r) for r in big_tree["receivers"] for s in (1, 2)]
             for room in big_tree["rooms"]}          # 2 of the 3 sources are in-split
    split_path = write_split(big_tree, "two_of_three.json", split)
    counter_path = str(tmp_path / "index_loads.txt")

    loader = build_loader(big_tree, split_path, restrict=True)
    count_index_loads(loader, big_tree, counter_path)
    assert len(loader.dataset) == 8  # only in-split files are targets

    seen_targets = 0
    for _ in range(2):  # persistent workers: a second epoch must not re-parse the split
        for audio, infos in loader:
            for info in infos:
                target = os.path.basename(info["path"])
                room = info["relpath"].split("/")[-2]
                drawn = [big_tree["file_of_marker"][m] for m in drawn_markers(info["context_audio"])]
                assert info["context_audio"].shape == (8, 1, IR_LEN)
                assert all(drawn_room == room for drawn_room, _ in drawn)
                assert all(basename in split[room] for _, basename in drawn)
                assert all(basename != target for _, basename in drawn)
                seen_targets += 1

    assert seen_targets == 16
    assert "Couldn't load file" not in capfd.readouterr().out

    with open(counter_path) as fin:
        pids = [line.strip() for line in fin if line.strip()]
    assert sorted(pids) == sorted(set(pids)), "each worker must parse the split exactly once"
    assert len(pids) == 2, f"expected one parse in each of the 2 workers, got {pids}"
    assert str(os.getpid()) not in pids, "the parent must never parse the split"


def test_B7a_room_removed_from_the_split_after_construction_is_fatal(
    big_tree, tmp_path, monkeypatch
):
    present, removed = big_tree["rooms"][1], big_tree["rooms"][0]
    split_path = write_split(big_tree, "swap.json", full_split(big_tree, rooms=[removed]))
    sentinel = str(tmp_path / "resampled.txt")
    forbid_resampling(monkeypatch, sentinel)

    loader = build_loader(big_tree, split_path, restrict=True)
    assert len(loader.dataset) == 6  # the targets of the room that is about to disappear

    # Atomically swap in a split that no longer lists the room the targets live in.
    temporary = split_path + ".tmp"
    with open(temporary, "w") as fout:
        json.dump({SCENE: full_split(big_tree, rooms=[present])}, fout)
    os.replace(temporary, split_path)

    with pytest.raises(DatasetContractError) as excinfo:
        next(iter(loader))

    assert "does not list room" in str(excinfo.value)
    assert not os.path.exists(sentinel), "no replacement sample may be drawn"


def test_B7b_empty_context_pool_in_a_worker_is_fatal(big_tree, tmp_path, monkeypatch):
    # Every receiver keeps exactly one source => every restricted context pool is empty.
    split = {room: [ir_basename(1, r) for r in big_tree["receivers"]] for room in big_tree["rooms"]}
    split_path = write_split(big_tree, "one_source_per_receiver.json", split)
    sentinel = str(tmp_path / "resampled.txt")
    forbid_resampling(monkeypatch, sentinel)

    loader = build_loader(big_tree, split_path, restrict=True)
    assert len(loader.dataset) == 4

    with pytest.raises(DatasetContractError) as excinfo:
        next(iter(loader))

    assert "no in-split context" in str(excinfo.value)
    assert not os.path.exists(sentinel), "no replacement sample may be drawn"


def test_B7b_without_the_flag_the_same_split_trains_happily(big_tree):
    """Control: the fatal path is the restriction, not the fixture."""
    split = {room: [ir_basename(1, r) for r in big_tree["receivers"]] for room in big_tree["rooms"]}
    split_path = write_split(big_tree, "one_source_per_receiver.json", split)

    loader = build_loader(big_tree, split_path, restrict=False)
    audio, infos = next(iter(loader))

    assert audio.shape == (1, 1, 10240)
    assert infos[0]["context_audio"].shape == (8, 1, IR_LEN)


def test_B8_training_process_exits_non_zero_on_a_contract_violation(big_tree, tmp_path):
    split = {room: [ir_basename(1, r) for r in big_tree["receivers"]] for room in big_tree["rooms"]}
    split_path = write_split(big_tree, "one_source_per_receiver.json", split)

    config_path = tmp_path / "dataset_config.json"
    with open(config_path, "w") as fout:
        json.dump(make_dataset_config(big_tree, split_path, restrict=True), fout)

    driver_path = tmp_path / "driver.py"
    driver_path.write_text(
        "import json, sys\n"
        "from src.data.dataset import create_dataloader_from_config\n"
        "config = json.load(open(sys.argv[1]))\n"
        "loader = create_dataloader_from_config(config, batch_size=1, sample_size=10240,\n"
        "                                       sample_rate=22050, audio_channels=1,\n"
        "                                       num_workers=2, shuffle=False)\n"
        "for batch in loader:\n"
        "    print('PRODUCED A BATCH', flush=True)\n"
        "    break\n"
    )

    environment = dict(os.environ, PYTHONPATH=REPO_ROOT, CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, str(driver_path), str(config_path)],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=300,
    )

    assert completed.returncode != 0
    assert "DatasetContractError" in completed.stderr
    assert "PRODUCED A BATCH" not in completed.stdout


# ======================================================================================
# B9 (round B-fix) — EVERY split-index failure is fail-closed
# ======================================================================================
# A split we cannot read or cannot trust must terminate the run. Raw OSError /
# JSONDecodeError / AttributeError / TypeError would all be swallowed by
# SampleDataset.__getitem__'s generic fallback and silently resampled away, which is the
# exact failure mode round B exists to close.
def write_raw_split(tree, name, text):
    path = os.path.join(tree["splits_dir"], name)
    with open(path, "w") as fout:
        fout.write(text)
    return path


@pytest.mark.parametrize(
    "name, payload",
    [
        ("not_json.json", "{ this is not json"),
        ("empty.json", ""),
        ("root_is_a_list.json", '["ToyScene_idx_0"]'),
        ("root_is_a_string.json", '"ToyScene"'),
        ("scene_not_a_dict.json", '{"ToyScene": ["S001_R001_hybrid_IR.wav"]}'),
        ("scene_is_null.json", '{"ToyScene": null}'),
        ("room_not_a_list.json", '{"ToyScene": {"ToyScene_idx_0": "S001_R001_hybrid_IR.wav"}}'),
        ("room_is_a_dict.json", '{"ToyScene": {"ToyScene_idx_0": {"S001": 1}}}'),
        ("filename_not_a_str.json", '{"ToyScene": {"ToyScene_idx_0": [123]}}'),
        ("filename_is_null.json", '{"ToyScene": {"ToyScene_idx_0": [null]}}'),
    ],
)
def test_B9_malformed_split_raises_dataset_contract_error(tree, ar_md, name, payload):
    path = write_raw_split(tree, name, payload)

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md._load_split_room_index(path)

    assert path in str(excinfo.value)


def test_B9_invalid_utf8_split_raises_dataset_contract_error(tree, ar_md):
    """(codex round-C nit) A split whose *bytes* are not valid UTF-8 fails while being
    decoded, before any JSON parsing: ``json.load`` raises ``UnicodeDecodeError``, which
    is a ``ValueError`` but **not** a ``json.JSONDecodeError``, so the malformed-payload
    matrix above never exercised it. The production path already names
    ``UnicodeDecodeError`` in its except clause (``AR_md._load_split_room_index``); this
    pins it, because dropping it would let a raw ``UnicodeDecodeError`` escape into
    ``SampleDataset.__getitem__``'s generic fallback and be silently resampled away.
    Both the loader and the cached accessor must be fail-closed."""
    path = os.path.join(tree["splits_dir"], "invalid_utf8.json")
    with open(path, "wb") as fout:
        fout.write(b"\xff\xfe{")  # UTF-16-LE BOM + "{": not decodable as UTF-8

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md._load_split_room_index(path)
    assert path in str(excinfo.value)
    assert "not valid JSON" in str(excinfo.value)

    with pytest.raises(DatasetContractError):
        ar_md._split_room_index(path)  # the cached accessor is fail-closed too


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_B9_unreadable_split_raises_dataset_contract_error(tree, ar_md, kind):
    if kind == "missing":
        path = os.path.join(tree["splits_dir"], "does_not_exist.json")
    else:
        path = os.path.join(tree["splits_dir"], "a_directory.json")
        os.makedirs(path)

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md._load_split_room_index(path)

    assert path in str(excinfo.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a 0o000 file regardless")
def test_B9_permission_denied_split_raises_dataset_contract_error(tree, ar_md):
    path = write_split(tree, "locked.json", full_split(tree))
    os.chmod(path, 0o000)
    try:
        with pytest.raises(DatasetContractError):
            ar_md._load_split_room_index(path)
    finally:
        os.chmod(path, 0o644)


def test_B9_scene_not_a_dict_is_fatal_through_get_custom_metadata(tree, ar_md):
    """The HAA-style scene -> [files] schema is not a valid AR restriction index."""
    room = tree["rooms"][0]
    path = write_raw_split(tree, "haa_style.json", '{"ToyScene": ["S001_R001_hybrid_IR.wav"]}')

    with pytest.raises(DatasetContractError) as excinfo:
        ar_md.get_custom_metadata(
            make_info(tree, room, 1, 1, path, modalities(restrict=True)), None
        )

    assert "scene/room split" in str(excinfo.value)


def test_B9_split_replaced_by_malformed_json_after_construction_is_fatal(
    big_tree, tmp_path, monkeypatch
):
    room = big_tree["rooms"][0]
    split_path = write_split(big_tree, "swap_malformed.json", full_split(big_tree, rooms=[room]))
    sentinel = str(tmp_path / "resampled.txt")
    forbid_resampling(monkeypatch, sentinel)

    loader = build_loader(big_tree, split_path, restrict=True)
    assert len(loader.dataset) == 6

    temporary = split_path + ".tmp"
    with open(temporary, "w") as fout:
        fout.write("{ truncated json")
    os.replace(temporary, split_path)

    with pytest.raises(DatasetContractError) as excinfo:
        next(iter(loader))

    assert "not valid JSON" in str(excinfo.value)
    assert not os.path.exists(sentinel), "no replacement sample may be drawn"
