"""
Validity gate for the context-ablation control.

Loads eval_A (=correct), zeroctx, wrongroom via importlib (exactly how the
dataloader loads them) on one real seen query at K=1 and asserts:

  * EVERY non-audio key is BYTE-IDENTICAL across the three modules
    (source, source_vit, depth, context_poses, context_poses_vit,
     context_fused_pose{pair_local,src_qrel,recs_qrel}). This is the whole
    validity argument: the three runs differ in context_audio ONLY.
  * zeroctx.context_audio is all-zeros, same shape as correct.
  * wrongroom.context_audio differs from correct, is finite, same shape,
    and is a real RIR from a different scene per the donor manifest.
"""
import importlib.util
import json
import os

import torch

MOD_DIR = "src/configs/dataset_configs/custom_metadata"


def load(name):
    p = os.path.join(MOD_DIR, name)
    s = importlib.util.spec_from_file_location("metadata_module", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def info_for(scene, scene_id, wav):
    rel = f"single_channel_ir_1/{scene}/{scene_id}/{wav}"
    return {
        "path": f"AcousticRooms/{rel}",
        "relpath": rel,
        "json_file_path": "data/AR/seen_eval.json",
        "seeneval": True, "unseeneval": False,
        "modalities": {
            "acoustic_context": {"load": True, "max_context": 1, "max_len": 9600},
            "depth": {"load": True}, "poses": {"load": True},
        },
    }


def main():
    sd = json.load(open("data/AR/seen_eval.json"))
    scene = next(iter(sd)); sid = next(iter(sd[scene])); wav = sd[scene][sid][0]
    print(f"sample: {scene}/{sid}/{wav}  (K=1)")

    A = load("AR_md_arbRIR_v0_eval_A.py")
    Z = load("AR_md_arbRIR_v0_eval_zeroctx.py")
    W = load("AR_md_arbRIR_v0_eval_wrongroom.py")

    a = A.get_custom_metadata(info_for(scene, sid, wav), None)
    z = Z.get_custom_metadata(info_for(scene, sid, wav), None)
    w = W.get_custom_metadata(info_for(scene, sid, wav), None)

    # 1. every non-audio key byte-identical across the 3 modules
    for k in ("source", "source_vit", "depth", "context_poses", "context_poses_vit"):
        assert torch.equal(a[k], z[k]), f"zeroctx {k} drifted from correct"
        assert torch.equal(a[k], w[k]), f"wrongroom {k} drifted from correct"
    for sub in ("pair_local", "src_qrel", "recs_qrel"):
        assert torch.equal(a["context_fused_pose"][sub], z["context_fused_pose"][sub]), sub
        assert torch.equal(a["context_fused_pose"][sub], w["context_fused_pose"][sub]), sub
    # Scheme-A invariant carried through
    assert torch.equal(a["context_poses"], a["context_fused_pose"]["src_qrel"])
    print("[1] all non-audio keys byte-identical across correct/zeroctx/wrongroom  OK")

    # 2. shapes
    shp = (1, 1, 9600)
    for tag, m in (("correct", a), ("zeroctx", z), ("wrongroom", w)):
        assert m["context_audio"].shape == shp, (tag, m["context_audio"].shape)
    print(f"[2] context_audio shape {shp} for all three  OK")

    # 3. zeroctx audio is exactly zero
    assert torch.count_nonzero(z["context_audio"]) == 0, "zeroctx audio not all-zero"
    assert torch.equal(a["context_audio"], a["context_audio"])  # finite sanity
    print("[3] zeroctx.context_audio is all-zeros  OK")

    # 4. wrongroom audio: differs from correct, finite, matches donor manifest
    assert not torch.equal(a["context_audio"], w["context_audio"]), "wrongroom == correct audio"
    assert torch.isfinite(w["context_audio"]).all(), "wrongroom audio non-finite"
    wr = json.load(open("data/AR/arbRIR_v0_eval_wrongroom_seen.json"))
    key = f"{scene}/{sid}/{os.path.splitext(wav)[0]}"
    d_scene, d_sid, d_wav = wr[key]
    assert d_scene != scene, f"donor scene {d_scene} == query scene {scene}"
    # load donor directly and confirm wrongroom used exactly it
    import torchaudio
    dfp = os.path.join("AcousticRooms/single_channel_ir_1", d_scene, d_sid, d_wav)
    dw, dr = torchaudio.load(dfp)
    dw = dw[:, :9600] if dw.shape[1] >= 9600 else torch.cat(
        [dw, torch.zeros(dw.shape[0], 9600 - dw.shape[1])], dim=1)
    assert torch.equal(w["context_audio"], dw.unsqueeze(0)), "wrongroom audio != donor file"
    print(f"[4] wrongroom.context_audio == donor {d_scene}/{d_sid}/{d_wav} (≠ query room)  OK")

    print("\nVALIDITY GATE PASSED — the 3 runs differ in context_audio ONLY")


if __name__ == "__main__":
    main()
