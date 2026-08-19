import collections
import importlib.util
import os
import numpy as np
import json
import torch
import torchaudio


def get_custom_metadata(info, audio):
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]

    # Get Config Info
    modalities = info['modalities'] # Modalities to load
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    # Get Instance Information
    scene_name = rel_path.split("/")[-3]
    md['scene'] = scene_name
    capture_id = rel_path.split("/")[-1].split(".")[0]
    # RAF metadata is per-room (one poses/groups pair per room), unlike HAA's
    # single dataset-level metadata folder.
    metadata_path = os.path.join(dataset_folder, scene_name, 'metadata')

    poses_metadata = load_json_cached(os.path.join(metadata_path, 'poses_metadata.json'))
    groups_metadata = load_json_cached(os.path.join(metadata_path, 'groups_metadata.json'))
    capture = poses_metadata[capture_id]
    group = groups_metadata[capture["group_key"]]

    source_pos = capture["tx_xyz_p"]

    # Load Positions
    if pose_config.get('load', False):
        # Mapping H (plan Rev 2 section 3): the frame is centred on the source (tx)
        # and the receiver goes into the source/source_vit slots, exactly as HAA
        # does — the name is kept so the AR-pretrained weights transfer.
        proj_listener_pos = get_3d_point_camera_coord(source_pos, capture["rx_p"])
        proj_listener_pos = torch.Tensor(proj_listener_pos).float()
        md['source'] = proj_listener_pos
        md['source_vit'] = proj_listener_pos.unsqueeze(0) # [1, 3]

    # Load Acoustic Context
    if acoustic_context_config.get('load', False):
        all_ref_irs, all_ref_receiver_pos, context_ids = get_ir_and_location_for_other_receivers(
            full_audio_path,
            capture_id=capture_id,
            scene_name=scene_name,
            source_pos=source_pos,
            poses_metadata=poses_metadata,
            train_ids=group["train_ids"],
            num_ref_receivers=acoustic_context_config.get('max_context', 8),
            max_len=acoustic_context_config.get('max_len', 9600),
            deterministic=acoustic_context_config.get('deterministic', False),
        )
        md['context_poses'] = all_ref_receiver_pos # [N, 3]
        md['context_poses_vit'] = all_ref_receiver_pos
        md['context_audio'] = all_ref_irs # [N, 1, max_len]
        # Provenance (plan Rev 2 section 6, C7): RAF receivers are near-duplicates
        # across placements, so a position fingerprint cannot identify a context
        # set — the capture IDs can. int64 tensors, because the collation path
        # stacks tensors and would leave strings as a ragged tuple.
        md['context_capture_ids'] = torch.tensor([int(c) for c in context_ids],
                                                 dtype=torch.int64) # [N]
        md['sample_target_id'] = torch.tensor(int(capture_id), dtype=torch.int64)

    # Load Depth
    if depth_config.get('load', False):
        depth_path = os.path.join(dataset_folder, scene_name, "depth_images",
                                  group["depth_file"])
        pano_depth = load_depth_cached(depth_path) # [H, W]
        # NO flipud (unlike HAA_md): render_depth.py emits rows in exactly the
        # order convert_equirect_to_camera_coord assumes, row 0 = zenith.
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512) # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1) # [3, H, W]

    return md


##### Utils #####
# convert_equirect_to_camera_coord and get_3d_point_camera_coord are copied
# verbatim from HAA_md.py: each custom-metadata hook is loaded standalone by
# src/data/dataset.py and carries its own helpers (AR_md.py does the same).
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


# Per-worker caches. Each dataloader worker gets its own copy of this module's
# globals, so these are per-process and never shared/mutated across workers.
_JSON_CACHE = {}
_DEPTH_CACHE = collections.OrderedDict()
_DEPTH_CACHE_MAX = 64


def load_json_cached(path):
    """Load a metadata JSON once per worker (they are large and read every item)."""
    if path not in _JSON_CACHE:
        with open(path) as f:
            _JSON_CACHE[path] = json.load(f)
    return _JSON_CACHE[path]


def load_depth_cached(path):
    """Bounded LRU cache of RAW depth maps.

    The raw [H, W] map is cached rather than the converted [3, H, W] point cloud:
    it is a third of the memory, and every caller then gets a freshly built tensor
    that no other sample aliases.
    """
    if path in _DEPTH_CACHE:
        _DEPTH_CACHE.move_to_end(path)
        return _DEPTH_CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RAF depth map not found: {path}")
    depth = np.load(path)
    _DEPTH_CACHE[path] = depth
    while len(_DEPTH_CACHE) > _DEPTH_CACHE_MAX:
        _DEPTH_CACHE.popitem(last=False)
    return depth


def _raf_common():
    """Load ``data/RAF/raf_common.py`` (the single source of the context seed).

    This hook is exec'd by file path, so it cannot rely on package imports; the
    repo root is five directories up from
    ``src/configs/dataset_configs/custom_metadata/RAF_md.py``.
    """
    module = globals().get("_RAF_COMMON")
    if module is None:
        repo_root = os.path.abspath(__file__)
        for _ in range(5):
            repo_root = os.path.dirname(repo_root)
        path = os.path.join(repo_root, "data", "RAF", "raf_common.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"raf_common.py not found at {path}")
        spec = importlib.util.spec_from_file_location("raf_common_for_md", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals()["_RAF_COMMON"] = module
    return module


def select_context_ids(scene_name, capture_id, train_ids, num_ref_receivers,
                       deterministic):
    """Choose the context capture ids for one target.

    The pool is the target's own group support minus the target itself, sorted so
    the draw never depends on the order the metadata JSON happens to list.

    ``deterministic=False`` (training) keeps HAA's stochastic ``np.random.choice``.
    ``deterministic=True`` (eval) draws from a ``torch.Generator`` seeded by
    ``stable_context_seed(room, capture id)``, which makes the context set a
    function of the item alone: identical across worker topologies, checkpoints and
    eval seeds, so the 5 eval seeds vary diffusion noise only.
    """
    pool = sorted(c for c in train_ids if c != capture_id)
    if len(pool) < num_ref_receivers:
        raise ValueError(
            f"capture {capture_id} of scene {scene_name}: context pool holds "
            f"{len(pool)} captures, need {num_ref_receivers}")
    if not deterministic:
        return [str(c) for c in np.random.choice(pool, num_ref_receivers, replace=False)]
    generator = torch.Generator()
    generator.manual_seed(_raf_common().stable_context_seed(scene_name, capture_id))
    picks = torch.randperm(len(pool), generator=generator)[:num_ref_receivers]
    return [pool[int(i)] for i in picks]


def get_ir_and_location_for_other_receivers(ir_file_path, capture_id, scene_name,
                                            source_pos, poses_metadata, train_ids,
                                            num_ref_receivers, max_len=9600,
                                            deterministic=False):
    dir_name = os.path.dirname(ir_file_path)
    context_ids = select_context_ids(scene_name, capture_id, train_ids,
                                     num_ref_receivers, deterministic)

    all_ref_irs = []
    all_ref_receiver_pos = []
    for context_id in context_ids:
        ref_wav, rate = torchaudio.load(os.path.join(dir_name, f"{context_id}.wav"))
        assert rate == 22050, "IR sampling rate must be 22050!"
        if ref_wav.shape[1] < max_len:
            ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])], dim=1)
        else:
            ref_wav = ref_wav[:, :max_len]
        ref_wav = ref_wav.unsqueeze(0) # C=1
        all_ref_irs.append(ref_wav)

        proj_rec_loc = get_3d_point_camera_coord(source_pos,
                                                 poses_metadata[context_id]["rx_p"])
        all_ref_receiver_pos.append(torch.Tensor(proj_rec_loc).float())

    all_ref_irs = torch.cat(all_ref_irs, dim=0)
    all_ref_receiver_pos = torch.vstack(all_ref_receiver_pos)
    return all_ref_irs, all_ref_receiver_pos, context_ids
