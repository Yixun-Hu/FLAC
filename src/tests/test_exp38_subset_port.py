"""exp_38 round A — the port of exp_14's data side onto ``localization-exp`` is exact.

exp_38 trains Yaw-Aug FLAC and OrbitRIR on "EXACTLY the same nested subsets as exp_14's
dc_van runs (same subset definition/manifests, same context restriction)". exp_14's data
path lives only on the ``exp-14-data-curve`` branch (launch commits ``66901577`` /
``f609976d`` / ``6a4b8b08``, all byte-identical on these files); exp_38 ports it onto
``localization-exp`` (plan_data_efficiency.md §4-§5). These tests are the executable
statement of "the port is exactly exp_14's data path":

* every ported file hashes to exp_14's launch blob (the constants below were read with
  ``git show 66901577:<path> | sha256sum`` and agree at all three launch commits);
* (the base-commit check -- ``AR_md.py`` was byte-identical at ``7bbd8aa`` and at the
  exp_38 base, so the ported file IS exp_14's diff -- needs git history and therefore lives in
  ``exp38_subset_audit.py``, not here: these tests are self-contained, Codex plan r1 NIT 6);
* the three splits are the registered nested subsets of ``data/AR/train.json``;
* the three dataset configs differ from ``acousticroom_train.json`` only by the split path
  and ``restrict_to_split: true``;
* ``SampleDataset`` lets ``DatasetContractError`` through and still resamples any other
  failure (the behaviour of exp_14's two ``dataset.py`` hunks).
"""
import hashlib
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# path -> sha256 of the exp_14 launch blob (66901577 == f609976d == 6a4b8b08 == 46a9c3d0).
EXP14_BLOBS = {
    "data/AR/train_frac025_s2026.json": "ce33f34f74b055d98a576c5aa5398d058c698d4cce9049e9648bcf3edf27122d",
    "data/AR/train_frac050_s2026.json": "bf33084671130a87735a47f97ff9c5851676f65fe1828bc8f8916e55af113ce4",
    "data/AR/train_frac075_s2026.json": "57388ce870f63d9d64fda0c6722f85625580f5572b44b85add4130a8788d26c1",
    "data/AR/train_frac_manifest_s2026.json": "553534ea7fec8efe65207d6f2a81c7bbb92b9e1ac8e72e14a7cf8defea3048ae",
    "data/AR/train_frac_manifest_s2026.sha256": "812dccf386926a52299779c5d6fb64577c89f02f0aca9b39f263721f5fd28c4c",
    "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json": "7ac8243f94cf9abc652e1a667c51263436041ea7a1fb0e8ed868c7e07fae7a21",
    "src/configs/dataset_configs/AR/train/acousticroom_train_frac050.json": "b74687f7a8f314436a4220eb461a0bf222f65276fb33eff4486a51e6d32a4025",
    "src/configs/dataset_configs/AR/train/acousticroom_train_frac075.json": "cc0e5fb8b5f221f393f19122d60c862763688f088329bcf0e42fb4ac338a17b5",
    "src/configs/dataset_configs/custom_metadata/AR_md.py": "fb2f68821e24f3e9d7f5c38ab2ecaee0235d4c9a36734c5bf4fe35fa86198310",
    "src/tests/test_ar_md_restrict_context.py": "5f63591eeb895a56d67434edf8b26fcc2e4d52d8be7e02faf71df4d30af68ca6",
}
TRAIN_JSON_SHA256 = "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03"

# tag -> (registered item count, rooms) from exp_14's manifest / data_curve_params_set_up.md.
FRACTIONS = {"025": (73473, 243), "050": (145665, 243), "075": (218463, 243)}


def _sha256(rel):
    with open(os.path.join(REPO, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _load(rel):
    with open(os.path.join(REPO, rel)) as fh:
        return json.load(fh)


def _rooms(split):
    """{(scene, room): set(basenames)} of a scene -> room -> [files] split."""
    return {(s, r): {os.path.basename(f) for f in files}
            for s, rooms in split.items() for r, files in rooms.items()}


@pytest.mark.parametrize("rel", sorted(EXP14_BLOBS))
def test_ported_file_is_the_exp14_launch_blob(rel):
    assert _sha256(rel) == EXP14_BLOBS[rel], f"{rel} is not exp_14's launch blob"


def test_train_json_is_the_one_exp14_subsampled():
    assert _sha256("data/AR/train.json") == TRAIN_JSON_SHA256


def test_manifest_sha256_file_lists_the_three_splits_and_the_manifest():
    lines = open(os.path.join(REPO, "data/AR/train_frac_manifest_s2026.sha256")).read().split("\n")
    entries = dict(reversed(line.split("  ", 1)) for line in lines if line.strip())
    for name, digest in entries.items():
        assert _sha256(os.path.join("data/AR", name)) == digest, name
    assert sorted(entries) == ["train_frac025_s2026.json", "train_frac050_s2026.json",
                               "train_frac075_s2026.json", "train_frac_manifest_s2026.json"]


@pytest.mark.parametrize("tag", sorted(FRACTIONS))
def test_split_counts_rooms_and_nesting(tag):
    n_items, n_rooms = FRACTIONS[tag]
    split = _rooms(_load(f"data/AR/train_frac{tag}_s2026.json"))
    full = _rooms(_load("data/AR/train.json"))
    assert sum(len(v) for v in split.values()) == n_items
    assert len(split) == n_rooms == len(full)
    for key, names in split.items():
        assert names <= full[key], f"{tag}: {key} lists files outside train.json"


def test_fractions_are_nested():
    f25, f50, f75 = (_rooms(_load(f"data/AR/train_frac{t}_s2026.json")) for t in ("025", "050", "075"))
    for key in f25:
        assert f25[key] <= f50[key] <= f75[key], key


@pytest.mark.parametrize("tag", sorted(FRACTIONS))
def test_frac_dataset_config_differs_from_full_only_by_split_and_restriction(tag):
    full = _load("src/configs/dataset_configs/AR/train/acousticroom_train.json")
    frac = _load(f"src/configs/dataset_configs/AR/train/acousticroom_train_frac{tag}.json")
    assert frac["datasets"][0]["json_file_path"] == f"data/AR/train_frac{tag}_s2026.json"
    assert frac["modalities"]["acoustic_context"]["restrict_to_split"] is True
    frac["datasets"][0]["json_file_path"] = full["datasets"][0]["json_file_path"]
    del frac["modalities"]["acoustic_context"]["restrict_to_split"]
    assert frac == full


def test_dataset_contract_error_is_reraised_before_the_generic_resample():
    src = open(os.path.join(REPO, "src/data/dataset.py")).read()
    from src.data.dataset import DatasetContractError
    assert issubclass(DatasetContractError, RuntimeError)
    reraise = src.index("except DatasetContractError:")
    generic = src.index("except Exception as e:\n            print(f'Couldn\\'t load file")
    assert reraise < generic
    assert src[reraise:generic].split("\n")[2].strip() == "raise"
