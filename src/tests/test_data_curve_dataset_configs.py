"""The three exp_14 fraction dataset configs differ from the base in exactly two fields.

Plan §4 Round D: ``acousticroom_train_frac{025,050,075}.json`` are the base training
config with (a) ``json_file_path`` pointing at that fraction's split and (b)
``modalities.acoustic_context.restrict_to_split: true`` -- the round-B flag that keeps the
other (1-f) of the data invisible in the *context* role too, which is what makes "25 % of
the data" an honest claim.

Any third difference would silently change the recipe between the arms/fractions and the
100 % anchors, so it is a test failure, not a review comment. The base config must also
stay free of the flag: default off means byte-identical upstream behaviour.
"""
import copy
import json
import os

import pytest

from src.tools.data_curve import names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO_ROOT, "src/configs/dataset_configs/AR/train/acousticroom_train.json")


def _load(path):
    with open(path) as fin:
        return json.load(fin)


def test_base_training_config_has_no_restriction_flag():
    assert "restrict_to_split" not in _load(BASE)["modalities"]["acoustic_context"]


@pytest.mark.parametrize("tag", ["025", "050", "075"])
def test_fraction_config_is_the_base_plus_exactly_two_fields(tag):
    base = _load(BASE)
    variant = _load(os.path.join(REPO_ROOT, names.TRAIN_DATASET_CONFIGS[tag]))

    expected = copy.deepcopy(base)
    expected["datasets"][0]["json_file_path"] = names.SPLIT_FILES[tag]
    expected["modalities"]["acoustic_context"]["restrict_to_split"] = True
    assert variant == expected


@pytest.mark.parametrize("tag", ["025", "050", "075"])
def test_fraction_config_points_at_a_committed_split(tag):
    variant = _load(os.path.join(REPO_ROOT, names.TRAIN_DATASET_CONFIGS[tag]))
    split = variant["datasets"][0]["json_file_path"]
    assert split == f"data/AR/train_frac{tag}_s2026.json"
    assert os.path.exists(os.path.join(REPO_ROOT, split)), f"{split} is not committed"
