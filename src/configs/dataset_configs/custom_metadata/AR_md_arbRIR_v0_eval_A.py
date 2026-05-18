"""
SUPERSET eval metadata module — Scheme A (frozen-manifest reader).

Plan: plan/eval_arbRIR_v0_vs_baseline_K1_K8.md

Clone of AR_md_arbRIR_v0.py's query side, with the random context sampler
REPLACED by a deterministic lookup into a frozen manifest built offline by
tools/build_arbRIR_eval_manifest.py. No `np.random.choice` here — context is
manifest-bound, so every eval run / both schemes / both models see byte-
identical context regardless of num_workers/batch_size/fork (see the
"Serious Issue" and "Manifest-key contract" sections of the plan).

Emits BOTH cross-attention key families from the same picked (s_i, r_i) set:
  - md['context_audio']      : RIR waveforms                 (both models)
  - md['context_poses_vit']  : s_i - r_q                      (both models, ViT)
  - md['context_fused_pose'] : {pair_local, src_qrel, recs_qrel}  (ablation)
  - md['context_poses']      : s_i - r_q   <-- SCHEME A       (baseline)

The V3 ablation model ignores `context_poses`; the baseline ignores
`context_fused_pose`. MultiConditioner drops keys not named in each model's
conditioning ids, so this one module serves the ablation AND baseline-A runs.

Scheme A: context_poses = s_i - r_q (query-receiver-centered source position) —
the quantity the baseline effectively learned (training had r_i = r_q, so
s_i - r_i collapsed to s_i - r_q); equals the V3 src_qrel exactly.
"""
import os
import json
import numpy as np
import torch
import torchaudio

# ---- Scheme switch (the ONLY semantic difference vs. _eval_B.py) ----
_SCHEME = "A"  # context_poses = src_qrel (s_i - r_q)

# Lazy per-process manifest cache. persistent_workers=True keeps this alive
# per worker, so the JSON is read at most once per worker.
_MANIFEST_CACHE = {}


def _get_manifest(manifest_path):
    m = _MANIFEST_CACHE.get(manifest_path)
    if m is None:
        with open(manifest_path) as f:
            m = json.load(f)
        _MANIFEST_CACHE[manifest_path] = m
    return m


def _resolve_manifest_path(info):
    """Fail-loud seen/unseen resolver (Manifest-key contract). Primary signal
    is the split-json basename; the seeneval/unseeneval flags are used only as
    a non-contradiction cross-check. Never silently picks a manifest."""
    jfp = info.get('json_file_path') or ''
    base = os.path.basename(jfp)
    seen_flag = bool(info.get('seeneval', False))
    unseen_flag = bool(info.get('unseeneval', False))

    if base == 'unseen_eval.json':
        split = 'unseen'
    elif base == 'seen_eval.json':
        split = 'seen'
    else:
        raise RuntimeError(
            f"[arbRIR_v0_eval_{_SCHEME}] cannot resolve split from "
            f"json_file_path={jfp!r}; expected basename seen_eval.json or "
            f"unseen_eval.json"
        )
    # Cross-check: an explicitly-True opposite flag means the config is
    # inconsistent — refuse rather than evaluate on the wrong distribution.
    if split == 'seen' and unseen_flag and not seen_flag:
        raise RuntimeError(
            f"[arbRIR_v0_eval_{_SCHEME}] split ambiguity: json={jfp} -> seen "
            f"but info['unseeneval']=True"
        )
    if split == 'unseen' and seen_flag and not unseen_flag:
        raise RuntimeError(
            f"[arbRIR_v0_eval_{_SCHEME}] split ambiguity: json={jfp} -> unseen "
            f"but info['seeneval']=True"
        )

    path = f"data/AR/arbRIR_v0_eval_manifest_{split}.json"
    if not os.path.exists(path):
        raise RuntimeError(
            f"[arbRIR_v0_eval_{_SCHEME}] manifest not found: {path}. Run "
            f"`python tools/build_arbRIR_eval_manifest.py` first."
        )
    return path


def get_custom_metadata(info, audio):
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]
    metadata_path = os.path.join(dataset_folder, 'metadata')

    modalities = info['modalities']
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    scene_name = rel_path.split("/")[-3]
    scene_id = rel_path.split("/")[-2]
    filename = rel_path.split("/")[-1].split(".")[0]
    # AR filename pattern: "S{src}_R{rec}_hybrid_IR" with "S00"+str(src), "R00"+str(rec).
    src_node_query = int(filename.split("_")[0][1:])
    rec_node_query = int(filename.split("_")[1][1:])
    md['scene'] = scene_name

    # Load query (s_q, r_q) in world coordinates.
    src_loc_query, rec_loc_query = get_receiver_source_location(rel_path, metadata_path)
    src_loc_query_np = np.asarray(src_loc_query, dtype=np.float32)
    rec_loc_query_np = np.asarray(rec_loc_query, dtype=np.float32)

    if pose_config.get('load', False):
        # query token: s_q - r_q in receiver-centered frame.
        proj_source_pos = src_loc_query_np - rec_loc_query_np
        proj_source_pos_t = torch.from_numpy(proj_source_pos).float()
        md['source'] = proj_source_pos_t                  # [3]
        md['source_vit'] = proj_source_pos_t.unsqueeze(0)  # [1, 3]

    if acoustic_context_config.get('load', False):
        max_len_cond = acoustic_context_config.get('max_len', 9600)
        num_ref = acoustic_context_config.get('max_context', 8)

        manifest = _get_manifest(_resolve_manifest_path(info))
        context_audio, pair_local, src_qrel, recs_qrel = load_context_from_manifest(
            manifest=manifest,
            scene_name=scene_name,
            scene_id=scene_id,
            filename=filename,
            query_audio_path=full_audio_path,
            metadata_path=metadata_path,
            rec_loc_query=rec_loc_query_np,
            num_ref=num_ref,
            max_len=max_len_cond,
        )
        # V3 cross-attention contract (consumed by the ablation model):
        md['context_audio'] = context_audio          # [N, 1, max_len]
        md['context_poses_vit'] = src_qrel           # [N, 3]
        md['context_fused_pose'] = {
            'pair_local': pair_local,  # [N, 3]  s_i - r_i
            'src_qrel':   src_qrel,    # [N, 3]  s_i - r_q
            'recs_qrel':  recs_qrel,   # [N, 3]  r_i - r_q
        }
        # Baseline cross-attention contract — SCHEME A: s_i - r_q.
        md['context_poses'] = src_qrel               # [N, 3]

    if depth_config.get('load', False):
        pano_depth_path = dataset_folder + 'depth_map'
        pano_depth = np.load(os.path.join(pano_depth_path, scene_name, scene_id, f"{rec_node_query}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512)  # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1)  # [3, H, W]

    return md


############# UTILS #############
def convert_equirect_to_camera_coord(depth_map, img_h, img_w):
    """Unproject equirectangular depth panorama to per-pixel 3D points in the
    receiver-centered camera frame. Identical to AR_md.py."""
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack(
        [depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi],
        dim=-1,
    )


def get_receiver_source_location(ir_file_path, metadata_path):
    """Identical contract to AR_md.py: returns (src_loc, rec_loc) world coords."""
    scene_name = ir_file_path.split("/")[-3]
    scene_id = ir_file_path.split("/")[-2]
    ir_file_name = ir_file_path.split("/")[-1]
    src_node = int(ir_file_name.split("_")[0][1:])
    rec_node = int(ir_file_name.split("_")[1][1:])
    json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
    metadata_file_path = os.path.join(metadata_path, scene_name, scene_id, json_file_name)
    with open(metadata_file_path, "r") as fin:
        meta_info = json.load(fin)
    return meta_info["src_loc"], meta_info["rec_loc"]


def load_context_from_manifest(manifest, scene_name, scene_id, filename,
                               query_audio_path, metadata_path,
                               rec_loc_query, num_ref, max_len):
    """
    Deterministic replacement for AR_md_arbRIR_v0.py:sample_arbitrary_context.
    Looks the query up in the frozen manifest by the scene-triple key and loads
    the first `num_ref` context RIRs in listed order. K=1 is therefore the
    prefix of the K=8 list (nested by construction).

    Returns
    -------
    context_audio : torch.Tensor [N, 1, max_len]
    pair_local    : torch.Tensor [N, 3]  s_i - r_i   (frame-invariant)
    src_qrel      : torch.Tensor [N, 3]  s_i - r_q   (query-receiver frame)
    recs_qrel     : torch.Tensor [N, 3]  r_i - r_q   (query-receiver frame)
    """
    key = f"{scene_name}/{scene_id}/{filename}"
    if key not in manifest:
        raise KeyError(
            f"[arbRIR_v0_eval_{_SCHEME}] query {key!r} not in frozen manifest "
            f"(rebuild with tools/build_arbRIR_eval_manifest.py if the split "
            f"changed)."
        )
    select = manifest[key][:num_ref]
    if len(select) < num_ref:
        raise RuntimeError(
            f"[arbRIR_v0_eval_{_SCHEME}] manifest entry for {key!r} has only "
            f"{len(select)} contexts, need {num_ref}"
        )

    dir_name = os.path.dirname(query_audio_path)
    rec_loc_query = np.asarray(rec_loc_query, dtype=np.float32)

    audio_list, pair_local_list, src_qrel_list, recs_qrel_list = [], [], [], []
    for fn in select:
        fp = os.path.join(dir_name, fn)
        ref_wav, rate = torchaudio.load(fp)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if ref_wav.shape[1] < max_len:
            ref_wav = torch.cat(
                [ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])], dim=1
            )
        else:
            ref_wav = ref_wav[:, :max_len]
        audio_list.append(ref_wav.unsqueeze(0))  # [1, 1, max_len]

        src_loc_i, rec_loc_i = get_receiver_source_location(fp, metadata_path)
        src_loc_i = np.asarray(src_loc_i, dtype=np.float32)
        rec_loc_i = np.asarray(rec_loc_i, dtype=np.float32)

        pair_local_list.append(torch.from_numpy(src_loc_i - rec_loc_i).float())
        src_qrel_list.append(torch.from_numpy(src_loc_i - rec_loc_query).float())
        recs_qrel_list.append(torch.from_numpy(rec_loc_i - rec_loc_query).float())

    context_audio = torch.cat(audio_list, dim=0)         # [N, 1, max_len]
    pair_local = torch.vstack(pair_local_list)            # [N, 3]
    src_qrel = torch.vstack(src_qrel_list)                # [N, 3]
    recs_qrel = torch.vstack(recs_qrel_list)              # [N, 3]
    return context_audio, pair_local, src_qrel, recs_qrel
