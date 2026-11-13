"""exp_43 (std_aug): the horizontal mirror flip of the UniFuse / PanoFormer augmentation, as a
reflection ``y -> -y`` of FLAC's conditioning. Own module: ``src/data/yaw_rotation.py`` keeps its
exp_15 bytes (exp_25's audit gate pins that file's sha256).

AR's panorama grid is ``theta_j = (j + 0.5) * 2pi / W - pi`` with per-pixel vectors
``(d cos(phi) cos(theta), d cos(phi) sin(theta), -d sin(phi))``. In real arithmetic
``theta_{W-1-j} = -theta_j``, so reversing the columns and negating ``y`` is the reflection
``M = diag(1, -1, 1)`` across the plane ``y = 0``; the pose vectors (``src_loc - rec_loc`` in
world axes) get the same ``M``. Exactness: :func:`mirror_scene_metadata` is a sign change and a
column permutation, so it is a bitwise involution and preserves norms bitwise; agreement with
re-converting a column-reversed scalar depth map is numerical, because the float32 grid is not
bit-symmetric (plan F6). With the yaw roll ``roll_d`` it generates the dihedral group D_W:
``flip . roll_d = roll_{-d} . flip``.

Written by the exp_43 Coder seat (Claude Opus 5.5) per plan_std_aug.md §2.1 / §3.1.
"""
import torch

from .yaw_rotation import POSE_KEYS

MIRROR_AXIS = 1


def _check_img_w(img_w):
    if isinstance(img_w, bool) or not isinstance(img_w, int) or img_w <= 0:
        raise ValueError(f"img_w must be a positive int, got {img_w!r}")


def _check_pose_keys(pose_keys):
    if not isinstance(pose_keys, tuple) or not all(isinstance(k, str) for k in pose_keys):
        raise ValueError(f"pose_keys must be a tuple of str, got {pose_keys!r}")


def _check_depth(depth, img_w):
    if not torch.is_tensor(depth):
        raise ValueError(f"depth must be a torch.Tensor, got {type(depth).__name__}")
    if not depth.is_floating_point():
        raise ValueError(f"depth must be a floating tensor, got {depth.dtype}")
    if depth.ndim != 3 or depth.shape[0] != 3:
        raise ValueError(f"depth must have shape [3, H, W], got {list(depth.shape)}")
    if int(depth.shape[2]) != img_w:
        raise ValueError(f"depth is {int(depth.shape[2])} columns wide but img_w is {img_w}")
    if not bool(torch.isfinite(depth).all()):
        raise ValueError("depth contains non-finite values")


def _check_pose(key, pose):
    if not torch.is_tensor(pose):
        raise ValueError(f"{key} must be a torch.Tensor, got {type(pose).__name__}")
    if not pose.is_floating_point():
        raise ValueError(f"{key} must be a floating tensor, got {pose.dtype}")
    # ndim first: a 0-d pose would make shape[-1] raise IndexError.
    if pose.ndim < 1 or pose.shape[-1] != 3:
        raise ValueError(f"{key} must have trailing dimension 3, got shape {list(pose.shape)}")


def check_mirror_metadata(md, img_w, pose_keys=POSE_KEYS):
    """The fail-closed input contract of :func:`mirror_scene_metadata`; returns ``None``.

    ``md`` a dict; ``img_w`` a positive int (``bool`` refused); ``pose_keys`` a tuple of str.
    A present ``depth`` must be a finite floating ``[3, H, img_w]`` tensor; a present pose a
    floating tensor with trailing dimension 3. Absent fields are allowed (the same absent-key
    semantics as ``rotate_scene_metadata``). Consumes no randomness, mutates nothing.
    """
    if not isinstance(md, dict):
        raise ValueError(f"metadata must be a dict, got {type(md).__name__}")
    _check_img_w(img_w)
    _check_pose_keys(pose_keys)
    if "depth" in md:
        _check_depth(md["depth"], img_w)
    for key in pose_keys:
        if key in md:
            _check_pose(key, md[key])


def draw_mirror_flips(n, generator):
    """``n`` fair bits from the CALLER's generator: ``torch.randint(0, 2, (n,))`` as int64.

    ``generator`` must be a ``torch.Generator`` (``None`` would silently draw from the global
    RNG); ``n`` a non-negative int (``bool`` refused). ``n == 0`` consumes nothing.
    """
    if not isinstance(generator, torch.Generator):
        raise ValueError(
            "draw_mirror_flips requires a dedicated torch.Generator, got "
            f"{type(generator).__name__}: None would draw from the GLOBAL RNG")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    return torch.randint(0, 2, (n,), generator=generator, dtype=torch.long)


def mirror_scene_metadata(md, img_w, pose_keys=POSE_KEYS):
    """Reflect one sample's conditioning across the plane ``y = 0``.

    ``depth [3, H, W]``: columns reversed, channel 1 negated. Each present field of
    ``pose_keys``: component 1 negated. Every other field (target RIR, ``context_audio``,
    ``padding_mask``, ``scene``, loader ids) passes through as the same object. Returns a
    shallow copy; the input dict and its tensors are never mutated, and every replaced field
    lives in fresh storage (``source`` / ``source_vit`` may share one storage, AR_md builds
    them so). Raises ``ValueError`` per :func:`check_mirror_metadata`.
    """
    check_mirror_metadata(md, img_w, pose_keys)
    out = dict(md)
    if "depth" in md:
        flipped = torch.flip(md["depth"], dims=[2])          # new storage
        flipped[MIRROR_AXIS] = -flipped[MIRROR_AXIS]
        out["depth"] = flipped
    for key in pose_keys:
        if key in md:
            pose = md[key].clone()                           # never write through an alias
            pose[..., MIRROR_AXIS] = -pose[..., MIRROR_AXIS]
            out[key] = pose
    return out
