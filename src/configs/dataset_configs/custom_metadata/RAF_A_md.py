import collections
import importlib
import importlib.util
import sys
import os
import numpy as np
import json
import torch
import torchaudio


# exp_21 (Mapping A). AR_md's semantics, exactly: the frame is centred on the
# LISTENER, the source position fills the source/source_vit slots, each context
# subtracts ITS OWN capture's receiver, and the depth panorama is rendered at the
# TARGET RECEIVER. That per-context own-rx subtraction is the part a "nominal mic
# position" shortcut would quietly get wrong -- see AR_md.get_ir_and_location_for_
# other_sources, which passes each reference file's own rec_loc.
#
# Unlike Mapping H, the context is FIXED IN THE MANIFEST rather than drawn at load
# time: every arm and seed must condition on identical references for the cross-arm
# ranking to be a paired comparison, and a draw -- however deterministic -- is one
# more thing that could differ between two runs.
_RAF_A_MD_TEST_MODE = False
_PUBLICATION_POINTER = "raf_publication.json"
_PUBLICATION_CHECKED = {}

_JSON_CACHE = {}
_DEPTH_CACHE = collections.OrderedDict()
_DEPTH_CACHE_MAX = 64

DEPTH_SHAPE = (256, 512)
METADATA_NAME = "mappingA_metadata.json"


def get_custom_metadata(info, audio):
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]

    modalities = info['modalities']
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    scene_name = rel_path.split("/")[-3]
    md['scene'] = scene_name
    capture_id = rel_path.split("/")[-1].split(".")[0]
    metadata_path = os.path.join(dataset_folder, scene_name, 'metadata')

    publication = assert_published_once(dataset_folder)
    if publication is not None:
        # exp_21 r3 P3: the generation the gate just attested travels with the
        # sample, so a per-item metric row can name the corpus it scored -- a
        # config path can be republished under the same name, a generation cannot.
        # r4 Q2: BOTH kinds. They are published separately and can move
        # independently, so a depth republish between two arms would otherwise
        # leave their recorded corpus identity unchanged.
        md['publication_prepare_generation'] = (
            publication["kinds"]["mappingA_prepare"]["generation"])
        md['publication_depth_generation'] = (
            publication["kinds"]["mappingA_depth"]["generation"])

    items = load_json_cached(os.path.join(metadata_path, METADATA_NAME))
    item = items[capture_id]
    rx_target = item["rx_target_p"]

    # exp_21 r2 N7: the item's identity travels with the sample so a per-item
    # metric row can be PAIRED across arms and seeds by id rather than by
    # position -- position pairing is exactly what item substitution breaks.
    md['item_id'] = item["item_id"]
    md['placement_id'] = item["placement_id"]
    md['mic_slot'] = int(item["mic_slot"])

    # Poses -- AR_md's formulas verbatim.
    if pose_config.get('load', False):
        proj_source_pos = get_3d_point_camera_coord(rx_target, item["tx_p"])
        proj_source_pos = torch.Tensor(proj_source_pos).float()
        md['source'] = proj_source_pos
        md['source_vit'] = proj_source_pos.unsqueeze(0)  # [1, 3]

    # Acoustic context -- other SOURCES at this microphone.
    if acoustic_context_config.get('load', False):
        max_len = acoustic_context_config.get('max_len', 9600)
        k = acoustic_context_config.get('max_context', 8)
        context = item["context"]
        if len(context) != k:
            raise ValueError(
                f"item {item['item_id']} holds {len(context)} context captures but "
                f"the config asks for {k}: the manifest is the contract")
        dir_name = os.path.dirname(full_audio_path)
        all_ref_irs, all_ref_src_pos, ids = [], [], []
        for entry in context:
            ref_wav, rate = torchaudio.load(
                os.path.join(dir_name, f"{entry['capture_id']}.wav"))
            assert rate == 22050, "IR sampling rate must be 22050!"
            if ref_wav.shape[1] < max_len:
                ref_wav = torch.cat(
                    [ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])],
                    dim=1)
            else:
                ref_wav = ref_wav[:, :max_len]
            all_ref_irs.append(ref_wav.unsqueeze(0))  # C=1
            # AR_md semantics: each reference subtracts ITS OWN receiver, not the
            # target's and not a nominal slot position. The two differ by the
            # measured rx displacement, which the manifest records per context.
            all_ref_src_pos.append(torch.Tensor(
                get_3d_point_camera_coord(entry["rx_p"], entry["tx_p"])).float())
            ids.append(int(entry["capture_id"]))
        md['context_poses'] = torch.vstack(all_ref_src_pos)   # [N, 3]
        md['context_poses_vit'] = md['context_poses']
        md['context_audio'] = torch.cat(all_ref_irs, dim=0)   # [N, 1, max_len]
        md['context_capture_ids'] = torch.tensor(ids, dtype=torch.int64)
        md['sample_target_id'] = torch.tensor(int(capture_id), dtype=torch.int64)

    # Depth -- the panorama at the TARGET RECEIVER (AR's listener-centred map).
    if depth_config.get('load', False):
        depth_path = os.path.join(dataset_folder, scene_name, "depth_images",
                                  item["depth_file"])
        pano_depth = load_depth_cached(depth_path)  # [H, W]
        # NO flipud: render_depth emits row 0 = zenith already.
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth),
                                                       256, 512)  # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1)  # [3, H, W]

    return md


##### Utils #####
# Copied verbatim from AR_md.py: each custom-metadata hook is exec'd standalone by
# src/data/dataset.py and carries its own helpers.
def convert_equirect_to_camera_coord(depth_map, img_h, img_w):  # 3D point cloud per pixel
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


def load_json_cached(path):
    """Load the item manifest once per worker (it is read for every item)."""
    if path not in _JSON_CACHE:
        with open(path) as f:
            _JSON_CACHE[path] = json.load(f)
    return _JSON_CACHE[path]


def validate_depth_map(depth, path):
    """Fail-closed contract check on a loaded depth map.

    ``SampleDataset.__getitem__`` swallows exceptions and substitutes a RANDOM
    other item, so a malformed map would silently shrink and reshuffle the
    evaluation set. The distinctive wording is the only trace left in the log.
    """
    problems = []
    if depth.shape != DEPTH_SHAPE:
        problems.append(f"shape {tuple(depth.shape)} != {DEPTH_SHAPE}")
    if depth.dtype != np.float32:
        problems.append(f"dtype {depth.dtype} != float32")
    if not np.isfinite(depth).all():
        problems.append("holds non-finite values")
    elif not (depth > 0).all():
        problems.append(f"holds non-positive distances (min {float(depth.min())})")
    if problems:
        raise ValueError(
            f"RAF depth map contract violated for {path}: " + "; ".join(problems))
    return depth


def load_depth_cached(path):
    """Bounded LRU cache of RAW depth maps (a fresh tensor is built per call)."""
    if path in _DEPTH_CACHE:
        _DEPTH_CACHE.move_to_end(path)
        return _DEPTH_CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mapping-A depth map not found: {path}")
    depth = validate_depth_map(np.load(path), path)
    _DEPTH_CACHE[path] = depth
    while len(_DEPTH_CACHE) > _DEPTH_CACHE_MAX:
        _DEPTH_CACHE.popitem(last=False)
    return depth


def _publication_error_type():
    """``src.data.dataset.RAFPublicationError`` -- the one exception the loader's
    substitution handler re-raises instead of swallowing."""
    module = sys.modules.get("src.data.dataset")
    if module is None:
        module = importlib.import_module("src.data.dataset")
    return module.RAFPublicationError


def _raf_module(name):
    """Load a sibling data/RAF module by path (this hook is exec'd, not imported)."""
    repo_root = os.path.abspath(__file__)
    for _ in range(5):
        repo_root = os.path.dirname(repo_root)
    path = os.path.join(repo_root, "data", "RAF", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"raf_{name}_for_mappingA_md", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_directory(a, b):
    """Same directory by INODE, not by string: symlinks and bind mounts differ."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def _verify_publication(root):
    """Verify the MAPPING-A combined publication the runtime pointer describes.

    Mapping H's verifier keeps checking Mapping H; this one checks the mappingA
    flavor, on that flavor's own disjoint roots, so a tree carrying both stays
    valid for both.
    """
    pointer_path = os.path.join(root, _PUBLICATION_POINTER)
    if not os.path.isfile(pointer_path):
        return {"published": False,
                "reason": f"no {_PUBLICATION_POINTER} in {root}: the tree was never "
                          "published by data/RAF/prepare_mappingA.py"}
    try:
        with open(pointer_path) as f:
            pointer = json.load(f)
        split_dir, output_dir = pointer["split_dir"], pointer["output_dir"]
        rooms = list(pointer["rooms"])
        flavor = pointer.get("flavor")
    except (ValueError, KeyError, TypeError) as e:
        return {"published": False,
                "reason": f"{pointer_path} is not a valid publication pointer ({e})"}

    if flavor != "mappingA":
        return {"published": False,
                "reason": f"{pointer_path} declares flavor {flavor!r}, not 'mappingA': "
                          "a Mapping-A config may not consume a Mapping-H tree"}
    if not pointer.get("canonical"):
        return {"published": False,
                "reason": f"{pointer_path} declares a NON-CANONICAL publication"}
    if not _same_directory(output_dir, root):
        return {"published": False,
                "reason": f"{pointer_path} points at output_dir {output_dir!r}, which "
                          f"is not the tree being loaded ({root!r})"}

    publish = _raf_module("publish")
    try:
        report = publish.verify_combined_publication(
            split_dir, output_dir, rooms=rooms, canonical=True, flavor="mappingA")
    except (ValueError, OSError) as e:
        return {"published": False, "reason": f"combined verification failed: {e}"}
    report["pointer"] = pointer
    return report


def assert_published_once(dataset_folder):
    """First-load publication gate, cached per process."""
    if _RAF_A_MD_TEST_MODE:
        return None
    root = os.path.abspath(dataset_folder.rstrip(os.sep) or os.sep)
    if root in _PUBLICATION_CHECKED:
        return _PUBLICATION_CHECKED[root]
    report = _verify_publication(root)
    if not report.get("published"):
        raise _publication_error_type()(
            f"Mapping-A publication check failed for {root}: {report.get('reason')}. "
            "The tree is not an attested Mapping-A publication.")
    _PUBLICATION_CHECKED[root] = report
    return report
