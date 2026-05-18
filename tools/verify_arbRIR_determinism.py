"""
Determinism HARD GATE (plan/eval_arbRIR_v0_vs_baseline_K1_K8.md, Verif. #1/#3/#4).

MANDATORY before the eval matrix. Exits non-zero on any failure.

1. Full-split key dry-run (Verification #1, complete): for EVERY query in
   data/AR/{seen,unseen}_eval.json, the scene-triple key the module computes
   from info['relpath'] is present in the frozen manifest -> zero KeyError.
2. Plumbing-independence (Verification #3): build the seen K=8 dataloader
   twice with different (num_workers, batch_size) — (1, 16) vs (4, 32) — and
   assert per-sample context_audio is bit-identical via torch.equal over a
   multi-scene subset. (1 vs 4, not 0 vs 4: dataset.py:411 hardcodes
   persistent_workers=True, which forbids num_workers=0.)
3. A-vs-B identity: same subset, Scheme-A vs Scheme-B — context_audio
   torch.equal; context_poses_vit / context_fused_pose allclose(rtol=0,
   atol=1e-7); context_poses differs (the only intended difference).
4. Cross-model key check (Verification #4): FLAC_AR.json's required
   conditioning ids are a subset of the keys the modules emit (so
   MultiConditioner cannot raise on a missing context_poses).
"""
import importlib.util
import json
import os
import sys

import torch

from src.data.dataset import create_dataloader_from_config

MOD_DIR = "src/configs/dataset_configs/custom_metadata"
EVAL_DIR = "src/configs/dataset_configs/AR/eval"
N_SUBSET = 640  # spans Cafe_idx_1 (200) + into later scenes -> multi-scene-dir


def load_mod(name):
    spec = importlib.util.spec_from_file_location("metadata_module", os.path.join(MOD_DIR, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def step1_full_split_keys():
    A = load_mod("AR_md_arbRIR_v0_eval_A.py")
    total = 0
    for split in ("seen", "unseen"):
        manifest = json.load(open(f"data/AR/arbRIR_v0_eval_manifest_{split}.json"))
        sd = json.load(open(f"data/AR/{split}_eval.json"))
        miss = 0
        for scene in sd:
            for scene_id in sd[scene]:
                for wav in sd[scene][scene_id]:
                    total += 1
                    relpath = f"single_channel_ir_1/{scene}/{scene_id}/{wav}"
                    # exactly the module's key derivation
                    s = relpath.split("/")[-3]
                    sid = relpath.split("/")[-2]
                    fnm = relpath.split("/")[-1].split(".")[0]
                    key = f"{s}/{sid}/{fnm}"
                    if key not in manifest:
                        miss += 1
        assert miss == 0, f"{split}: {miss} queries missing a manifest key"
        # also confirm the resolver picks the right manifest fail-loud path
        info = {"json_file_path": f"data/AR/{split}_eval.json",
                "seeneval": split == "seen", "unseeneval": split == "unseen"}
        assert A._resolve_manifest_path(info).endswith(f"manifest_{split}.json")
    print(f"[1] full-split key dry-run OK: {total} queries, 0 KeyError, resolver OK")


def collect_per_sample(cfg_name, num_workers, batch_size, limit):
    cfg = json.load(open(os.path.join(EVAL_DIR, cfg_name)))
    dl = create_dataloader_from_config(
        cfg, batch_size=batch_size, num_workers=num_workers,
        sample_rate=22050, sample_size=10240, audio_channels=1, shuffle=False,
    )
    out = []
    for _, md in dl:
        for m in md:
            out.append(m)
            if len(out) >= limit:
                return out
    return out


def step2_plumbing_independence():
    a = collect_per_sample("acousticroom_seeneval_arbRIR_v0evalA_8.json", 1, 16, N_SUBSET)
    b = collect_per_sample("acousticroom_seeneval_arbRIR_v0evalA_8.json", 4, 32, N_SUBSET)
    n = min(len(a), len(b))
    assert n >= N_SUBSET, f"only collected {n} samples"
    for i in range(n):
        assert torch.equal(a[i]["context_audio"], b[i]["context_audio"]), \
            f"context_audio differs at sample {i} across (nw=1,bs=16) vs (nw=4,bs=32)"
        assert torch.equal(a[i]["context_poses"], b[i]["context_poses"]), f"context_poses @ {i}"
    print(f"[2] plumbing-independence OK: {n} samples bit-identical "
          f"across (nw=1,bs=16) vs (nw=4,bs=32)")


def step3_AB_identity():
    a = collect_per_sample("acousticroom_seeneval_arbRIR_v0evalA_8.json", 4, 32, N_SUBSET)
    b = collect_per_sample("acousticroom_seeneval_arbRIR_v0evalB_8.json", 4, 32, N_SUBSET)
    n = min(len(a), len(b))
    diff_poses = 0
    for i in range(n):
        assert torch.equal(a[i]["context_audio"], b[i]["context_audio"]), f"A/B audio @ {i}"
        assert torch.allclose(a[i]["context_poses_vit"], b[i]["context_poses_vit"], rtol=0, atol=1e-7)
        for s in ("pair_local", "src_qrel", "recs_qrel"):
            assert torch.allclose(a[i]["context_fused_pose"][s], b[i]["context_fused_pose"][s],
                                  rtol=0, atol=1e-7), f"A/B fused[{s}] @ {i}"
        # A: context_poses == src_qrel ; B: == pair_local
        assert torch.equal(a[i]["context_poses"], a[i]["context_fused_pose"]["src_qrel"])
        assert torch.equal(b[i]["context_poses"], b[i]["context_fused_pose"]["pair_local"])
        if not torch.equal(a[i]["context_poses"], b[i]["context_poses"]):
            diff_poses += 1
    assert diff_poses > 0, "A and B context_poses never differ — scheme switch ineffective?"
    print(f"[3] A-vs-B identity OK: {n} samples, context identical, "
          f"context_poses differs on {diff_poses}/{n} (expected: arbitrary r_i != r_q)")


def step4_cross_model_keys():
    emitted = {"scene", "source", "source_vit", "depth",
               "context_audio", "context_poses_vit", "context_fused_pose", "context_poses"}
    for mc in ("FLAC_AR.json", "FLAC_AR_arbRIR_v0.json"):
        c = json.load(open(f"src/configs/model_configs/FLAC/AR/{mc}"))
        diff = c["model"]["diffusion"]
        req = set(diff.get("cross_attention_cond_ids", [])) | set(diff.get("global_cond_ids", []))
        missing = req - emitted
        assert not missing, f"{mc}: required cond ids not emitted: {missing}"
        print(f"[4] {mc}: required ids {sorted(req)} ⊆ emitted — MultiConditioner won't raise")


if __name__ == "__main__":
    try:
        step1_full_split_keys()
        step2_plumbing_independence()
        step3_AB_identity()
        step4_cross_model_keys()
    except AssertionError as e:
        print(f"\nHARD GATE FAILED: {e}")
        sys.exit(1)
    print("\nDETERMINISM HARD GATE PASSED — safe to run the matrix")
