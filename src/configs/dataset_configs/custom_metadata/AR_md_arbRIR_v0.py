"""
Custom metadata module for the arbitrary-RIR ablation (FLAC V3, geometry-fused
coordinate formulation). Differs from AR_md.py in two ways:

1. Context RIRs are sampled from arbitrary (s_i, r_i) pairs in the scene
   (not restricted to the query receiver). Pairs sharing the query source
   s_q are excluded — user-confirmed source-exclusion policy.

2. Three per-context cross-attention tokens are produced:
     - md['context_audio']     : RIR waveforms          (RIRConditioner, unchanged)
     - md['context_poses_vit'] : s_i - r_q              (GeometryConditioner against G_{r_q})
     - md['context_fused_pose']: dict bundling the three v0 pose vectors
                                  (FusedPoseConditioner)
   This matches the baseline cross-attention sequence length (3N) while
   replacing the geometric Fourier stream with the fused-pose token.

Query side (`source`, `source_vit`, `depth`, `scene`) is identical to AR_md.py.
"""
import os
import json
import numpy as np
import torch
import torchaudio


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

        context_audio, pair_local, src_qrel, recs_qrel = sample_arbitrary_context(
            full_audio_path,
            metadata_path=metadata_path,
            src_node_query=src_node_query,
            rec_loc_query=rec_loc_query_np,
            num_ref=num_ref,
            max_len=max_len_cond,
        )
        # V3 cross-attention contract:
        #   - audio stream  -> RIRConditioner   (existing)
        #   - ViT stream    -> GeometryConditioner over depth at r_q with s_i - r_q
        #   - fused-pose    -> new FusedPoseConditioner over the three pose vectors
        md['context_audio'] = context_audio          # [N, 1, max_len]
        md['context_poses_vit'] = src_qrel           # [N, 3]
        md['context_fused_pose'] = {
            'pair_local': pair_local,  # [N, 3]  s_i - r_i
            'src_qrel':   src_qrel,    # [N, 3]  s_i - r_q
            'recs_qrel':  recs_qrel,   # [N, 3]  r_i - r_q
        }

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


def sample_arbitrary_context(query_audio_path, metadata_path, src_node_query,
                             rec_loc_query, num_ref, max_len):
    """
    Sample `num_ref` context RIRs from arbitrary (s_i, r_i) pairs in the same
    scene_id directory, excluding all pairs with s_i == src_node_query.

    Returns
    -------
    context_audio : torch.Tensor [N, 1, max_len]
    pair_local    : torch.Tensor [N, 3]  s_i - r_i   (frame-invariant)
    src_qrel      : torch.Tensor [N, 3]  s_i - r_q   (query-receiver frame)
    recs_qrel     : torch.Tensor [N, 3]  r_i - r_q   (query-receiver frame)
    """
    dir_name = os.path.dirname(query_audio_path)
    all_filenames = os.listdir(dir_name)

    # Candidate pool: every audio file in this scene_id whose source differs
    # from the query source. Source-exclusion policy: see plan.
    candidates = []
    for fn in all_filenames:
        if not fn.endswith("_hybrid_IR.wav"):
            continue
        parts = fn.split("_")
        try:
            src_node = int(parts[0][1:])
        except ValueError:
            continue
        if src_node == src_node_query:
            continue
        candidates.append(fn)

    if len(candidates) == 0:
        raise RuntimeError(
            f"No context candidates after excluding src={src_node_query} in {dir_name}"
        )
    # Same replace-False / replace-True fallback as AR_md.py:103-106.
    replace = len(candidates) < num_ref
    select = np.random.choice(candidates, num_ref, replace=replace)

    audio_list, pair_local_list, src_qrel_list, recs_qrel_list = [], [], [], []
    rec_loc_query = np.asarray(rec_loc_query, dtype=np.float32)
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
