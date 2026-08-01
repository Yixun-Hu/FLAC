"""Static and construction contracts for the Yaw-equi-DiT config."""

import copy
import gc
import json
from pathlib import Path

from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
V0_MODEL = CONFIG_ROOT / "model_configs/FLAC/AR/FLAC_AR_YawPhaseDiT_C16.json"
PARENT_TRAIN = CONFIG_ROOT / "dataset_configs/AR/train/acousticroom_train.json"
K1_TRAIN = CONFIG_ROOT / "dataset_configs/AR/train/acousticroom_train_1.json"


def _load(path):
    with path.open() as handle:
        return json.load(handle)


def _conditioners(config):
    return {
        item["id"]: item
        for item in config["model"]["conditioning"]["configs"]
    }


def test_v0_static_routing_is_fixed_k1_34_token_contract():
    config = _load(V0_MODEL)
    diffusion = config["model"]["diffusion"]
    conds = _conditioners(config)

    assert diffusion["cross_attention_cond_ids"] == [
        "source_vit",
        "context_poses_vit",
        "context_poses",
        "context_audio",
    ]
    assert diffusion["global_cond_ids"] == ["source"]
    assert diffusion["query_phase_cond_id"] == "source"
    assert diffusion["phase_aliases"] == {"context_audio": "context_poses"}
    assert diffusion["config"]["azimuth_num_freqs"] == 8
    assert config["training"]["cond_method"] == "relative_phase"
    assert conds["source"]["type"] == "yaw_pose"
    assert conds["source"]["config"]["pose_role"] == "target"
    assert conds["context_poses"]["type"] == "yaw_pose"
    assert conds["context_poses"]["config"]["pose_role"] == "context"
    assert conds["source_vit"]["config"]["token_pool"] == "azimuth"
    assert conds["context_poses_vit"]["config"]["token_pool"] == "azimuth"

    target_w = (
        conds["source_vit"]["config"]["ViT"]["img_w"]
        // conds["source_vit"]["config"]["ViT"]["patch_w"]
    )
    context_w = (
        conds["context_poses_vit"]["config"]["ViT"]["img_w"]
        // conds["context_poses_vit"]["config"]["ViT"]["patch_w"]
    )
    assert target_w + context_w + 1 + 1 == 34


def test_k1_train_config_is_full_train_split_with_only_max_context_changed():
    parent = _load(PARENT_TRAIN)
    actual = _load(K1_TRAIN)
    expected = copy.deepcopy(parent)
    expected["modalities"]["acoustic_context"]["max_context"] = 1

    assert actual == expected
    dataset_path = actual["datasets"][0]["json_file_path"]
    assert dataset_path == "data/AR/train.json"
    assert "eval" not in dataset_path.lower()
    assert "subset" not in dataset_path.lower()


def _assert_complete_yaw_phase_model(model, *, azimuth_num_freqs):
    assert model.pretransform.enable_grad is False
    assert all(
        not parameter.requires_grad
        for parameter in model.pretransform.parameters()
    )
    assert model.query_phase_cond_id == "source"
    assert model.phase_aliases == {"context_audio": "context_poses"}
    assert model.cross_attn_cond_ids == [
        "source_vit",
        "context_poses_vit",
        "context_poses",
        "context_audio",
    ]
    assert set(model.cross_attn_type_embeddings) == set(model.cross_attn_cond_ids)
    conditioners = model.conditioner.conditioners
    assert type(conditioners["source"]).__name__ == "YawPoseConditioner"
    assert type(conditioners["context_poses"]).__name__ == "YawPoseConditioner"
    assert (
        conditioners["source"].yaw_pose_proj
        is conditioners["context_poses"].yaw_pose_proj
    )
    assert model.model.model.azimuth_num_freqs == azimuth_num_freqs


def test_complete_m8_config_constructs_on_cpu():
    config = _load(V0_MODEL)
    model = create_model_from_config(config)
    _assert_complete_yaw_phase_model(model, azimuth_num_freqs=8)
    wrapper = create_training_wrapper_from_config(config, model)
    assert wrapper.cond_method == "relative_phase"
    assert not hasattr(wrapper, "frame_avg_angles")
    del wrapper, model
    gc.collect()
