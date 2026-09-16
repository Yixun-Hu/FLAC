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

import numpy as np
import pytest
import torch
import torchaudio

from src.data.dataset import (
    DatasetContractError,
    LocalDatasetConfig,
    SampleDataset,
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

    np.random.seed(seed)
    new_irs, new_pos = ar_md.get_ir_and_location_for_other_sources(
        target,
        num_ref_sources=num_ref_sources,
        metadata_path=metadata_dir(tree),
        max_len=IR_LEN,
        allowed_basenames=None,
    )

    assert torch.equal(new_irs, ref_irs)
    assert torch.equal(new_pos, ref_pos)
    assert new_irs.shape == (num_ref_sources, 1, IR_LEN)
    assert new_pos.shape == (num_ref_sources, 3)
    if num_ref_sources > 2:  # the replacement branch really was the one exercised
        assert len(set(drawn_markers(new_irs))) < num_ref_sources


def test_B1_default_path_is_the_default_argument(tree, ar_md):
    """Existing call sites pass no allowed_basenames at all."""
    target = ir_path(tree, tree["rooms"][0], 2, 1)

    np.random.seed(7)
    ref_irs, ref_pos = _pinned_reference(target, 8, metadata_dir(tree), max_len=IR_LEN)

    np.random.seed(7)
    new_irs, new_pos = ar_md.get_ir_and_location_for_other_sources(
        target, num_ref_sources=8, metadata_path=metadata_dir(tree), max_len=IR_LEN
    )

    assert torch.equal(new_irs, ref_irs)
    assert torch.equal(new_pos, ref_pos)


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
