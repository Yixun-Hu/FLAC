import functools
import os
import numpy as np
import json
import torch 
import torchaudio

from src.data.dataset import DatasetContractError


def get_custom_metadata(info, audio): 
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]
    metadata_path = os.path.join(dataset_folder, 'metadata')

    # Get Config Info
    modalities = info['modalities'] 
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    # Get Instance Information
    scene_name = rel_path.split("/")[-3]
    scene_id = rel_path.split("/")[-2]
    filename = rel_path.split("/")[-1].split(".")[0]
    receiver_idx, source_idx = int(filename.split("_")[1][1:]), int(filename.split("_")[0][1:])
    md['scene'] = scene_name

    # Load Positions
    if pose_config.get('load', False):
        source_pos, listener_pos = get_receiver_source_location(rel_path, metadata_path)
        proj_source_pos = get_3d_point_camera_coord(listener_pos, source_pos)
        proj_source_pos = torch.Tensor(proj_source_pos).float()
        proj_listener_pos = torch.Tensor([0., 0., 0.])
        source_listener_pos = torch.cat([proj_source_pos.unsqueeze(0), proj_listener_pos.unsqueeze(0)], dim=0) # [2, 3]
        md['source'] = proj_source_pos 
        md['source_vit'] = proj_source_pos.unsqueeze(0) # [1, 3]

    # Load Acoustic Context
    if acoustic_context_config.get('load', False):
        max_len_cond = acoustic_context_config.get('max_len', 9600)
        allowed_basenames = None
        if acoustic_context_config.get('restrict_to_split', False):
            allowed_basenames = get_allowed_basenames(info.get('json_file_path', None), scene_name, scene_id)
        all_ref_irs, all_ref_src_pos = get_ir_and_location_for_other_sources(full_audio_path, num_ref_sources=acoustic_context_config.get('max_context', 8), metadata_path=metadata_path, max_len=max_len_cond, allowed_basenames=allowed_basenames)
        md['context_poses'] = all_ref_src_pos # [N, 3]  
        md['context_poses_vit'] = all_ref_src_pos
        md['context_audio'] = all_ref_irs # [N, 1, max_len_cond]

    # Load Depth
    if depth_config.get('load', False):
        pano_depth_path = dataset_folder + 'depth_map'
        pano_depth = np.load(os.path.join(pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512) # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1) # [3, H, W]
    
    return md


############# UTILS #############
def _load_split_room_index(json_path_canonical):
    """Parse a split JSON (scene -> room -> [basenames]) into {(scene, room): frozenset}.

    Fail-closed: a split we cannot read, or cannot trust, must terminate the run, so every
    failure becomes a DatasetContractError. A raw OSError / JSONDecodeError / AttributeError
    / TypeError would be caught by SampleDataset.__getitem__ and silently resampled away.
    """
    try:
        with open(json_path_canonical, "r") as fin:
            split_dict = json.load(fin)
    except OSError as err:
        raise DatasetContractError(f"split {json_path_canonical} cannot be read: {err}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise DatasetContractError(f"split {json_path_canonical} is not valid JSON: {err}") from None
    if not isinstance(split_dict, dict):
        raise DatasetContractError(f"split {json_path_canonical} is not a scene/room split: root is {type(split_dict).__name__}")
    index = {}
    for scene, rooms in split_dict.items():
        if not isinstance(rooms, dict):
            raise DatasetContractError(f"split {json_path_canonical} is not a scene/room split: scene {scene} is {type(rooms).__name__}")
        for room, files in rooms.items():
            if not isinstance(files, list):
                raise DatasetContractError(f"split {json_path_canonical} room {scene}/{room} is not a list of filenames: {type(files).__name__}")
            for fn in files:
                if not isinstance(fn, str):
                    raise DatasetContractError(f"split {json_path_canonical} room {scene}/{room} lists a non-string filename: {fn!r}")
            index[(scene, room)] = frozenset(os.path.basename(fn) for fn in files)
    return index

@functools.lru_cache(maxsize=8)
def _split_room_index(json_path_canonical):
    """Cached {(scene, room): frozenset(basenames)} index of one split file (per process)."""
    return _load_split_room_index(json_path_canonical)

def get_allowed_basenames(json_file_path, scene_name, scene_id):
    """In-split basenames of one room; every failure is fatal, never a resampled sample."""
    if not json_file_path:
        raise DatasetContractError("restrict_to_split is set but the dataset config has no json_file_path")
    canonical = os.path.realpath(json_file_path)
    try:
        return _split_room_index(canonical)[(scene_name, scene_id)]
    except KeyError:
        raise DatasetContractError(f"split {canonical} does not list room {scene_name}/{scene_id}") from None

def convert_equirect_to_camera_coord(depth_map, img_h, img_w): # 3D point cloud per pixel
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)

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

def get_ir_and_location_for_other_sources(ir_file_path, num_ref_sources, metadata_path, max_len=9600, allowed_basenames=None):
    """Draw ``num_ref_sources`` reference IRs from the other sources at this receiver.

    ``allowed_basenames`` (exp_14): when None the candidate list and the draws are the
    pinned upstream ones; when a set/frozenset of split basenames, candidates outside the
    training split are removed first, and an empty pool is a fatal contract violation.
    """
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
    if allowed_basenames is not None:
        valid_other_src_ir_paths = [p for p in valid_other_src_ir_paths
                                    if os.path.basename(p) in allowed_basenames]
        if len(valid_other_src_ir_paths) == 0:
            raise DatasetContractError(f"no in-split context for {ir_file_path}")
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