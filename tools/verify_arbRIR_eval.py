"""
Verification harness for the arbitrary-RIR eval modules
(plan/eval_arbRIR_v0_vs_baseline_K1_K8.md, Verification #2).

Loads AR_md_arbRIR_v0_eval_{A,B}.py exactly as the dataloader does
(importlib.spec_from_file_location), runs them on a real seen query at
K=8 and K=1, and asserts the shape/scheme/nesting/A-vs-B contract.
"""
import importlib.util
import json
import os

import torch

MOD_DIR = "src/configs/dataset_configs/custom_metadata"


def load_mod(name):
    p = os.path.join(MOD_DIR, name)
    spec = importlib.util.spec_from_file_location("metadata_module", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_info(scene, scene_id, wav, max_context):
    relpath = f"single_channel_ir_1/{scene}/{scene_id}/{wav}"
    return {
        "path": f"AcousticRooms/{relpath}",
        "relpath": relpath,
        "json_file_path": "data/AR/seen_eval.json",
        "seeneval": True,
        "unseeneval": False,
        "modalities": {
            "acoustic_context": {"load": True, "max_context": max_context, "max_len": 9600},
            "depth": {"load": True},
            "poses": {"load": True},
        },
    }


def main():
    split = json.load(open("data/AR/seen_eval.json"))
    scene = next(iter(split))
    scene_id = next(iter(split[scene]))
    wav = split[scene][scene_id][0]
    print(f"sample: {scene}/{scene_id}/{wav}")

    A = load_mod("AR_md_arbRIR_v0_eval_A.py")
    B = load_mod("AR_md_arbRIR_v0_eval_B.py")

    expected_keys = {
        "scene", "source", "source_vit", "depth",
        "context_audio", "context_poses_vit", "context_fused_pose", "context_poses",
    }

    for K in (8, 1):
        mdA = A.get_custom_metadata(make_info(scene, scene_id, wav, K), None)
        mdB = B.get_custom_metadata(make_info(scene, scene_id, wav, K), None)

        assert set(mdA) == expected_keys, (K, set(mdA) ^ expected_keys)
        assert set(mdB) == expected_keys, (K, set(mdB) ^ expected_keys)

        # shapes
        assert mdA["context_audio"].shape == (K, 1, 9600), mdA["context_audio"].shape
        for key in ("context_poses_vit", "context_poses"):
            assert mdA[key].shape == (K, 3), (key, mdA[key].shape)
        for sub in ("pair_local", "src_qrel", "recs_qrel"):
            assert mdA["context_fused_pose"][sub].shape == (K, 3)
        assert mdA["source"].shape == (3,)
        assert mdA["source_vit"].shape == (1, 3)
        assert mdA["depth"].shape == (3, 256, 512), mdA["depth"].shape

        # scheme semantics
        assert torch.equal(mdA["context_poses"], mdA["context_poses_vit"]), "A: ctx_poses==ctx_poses_vit"
        assert torch.equal(mdA["context_poses"], mdA["context_fused_pose"]["src_qrel"]), "A: ==src_qrel"
        assert torch.equal(mdB["context_poses"], mdB["context_fused_pose"]["pair_local"]), "B: ==pair_local"
        assert not torch.equal(mdB["context_poses"], mdB["context_poses_vit"]), \
            "B: context_poses should differ from poses_vit on arbitrary context"

        # A vs B: identical context, only context_poses differs
        assert torch.equal(mdA["context_audio"], mdB["context_audio"]), "A/B context_audio identical"
        assert torch.allclose(mdA["context_poses_vit"], mdB["context_poses_vit"], rtol=0, atol=1e-7)
        for sub in ("pair_local", "src_qrel", "recs_qrel"):
            assert torch.allclose(
                mdA["context_fused_pose"][sub], mdB["context_fused_pose"][sub], rtol=0, atol=1e-7
            ), f"A/B fused_pose[{sub}]"

        print(f"K={K}: keys/shapes/scheme/A-vs-B OK "
              f"(ctxA[0]={mdA['context_poses'][0].tolist()}, "
              f"ctxB[0]={mdB['context_poses'][0].tolist()})")

    # nesting: K=1 == K=8 sliced to [:1]
    md8 = A.get_custom_metadata(make_info(scene, scene_id, wav, 8), None)
    md1 = A.get_custom_metadata(make_info(scene, scene_id, wav, 1), None)
    assert torch.equal(md1["context_audio"], md8["context_audio"][:1]), "nesting: audio"
    assert torch.equal(md1["context_poses"], md8["context_poses"][:1]), "nesting: poses"
    for sub in ("pair_local", "src_qrel", "recs_qrel"):
        assert torch.equal(md1["context_fused_pose"][sub], md8["context_fused_pose"][sub][:1]), \
            f"nesting: fused[{sub}]"
    print("nesting K=1 == K=8[:1] OK")
    print("\nVerification #2 PASSED")


if __name__ == "__main__":
    main()
