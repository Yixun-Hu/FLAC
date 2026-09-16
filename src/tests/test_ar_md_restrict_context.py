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
