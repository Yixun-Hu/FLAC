"""Physically-consistent azimuthal (yaw) rotation of FLAC scene conditioning.

For a mono room impulse response, rigidly rotating the whole scene (room geometry
+ source + listener) about the vertical axis leaves the ground-truth RIR unchanged.
This module applies that rotation to the *conditioning* metadata only: the
equirectangular depth panorama (a horizontal circular roll together with a rotation
of its stored per-pixel 3D vectors) and the source/context pose vectors (a rotation
about z). ``context_audio`` (the reference RIR waveforms) and the target RIR are
yaw-invariant and are intentionally left untouched.

Used both by the standalone rotation-invariance diagnostic and by ``eval_FLAC.py``
(via its optional ``--rotate-deg`` flag).
"""
import math
from typing import Dict, Tuple

import torch

POSE_KEYS: Tuple[str, ...] = ("source", "source_vit", "context_poses", "context_poses_vit")


def wrap_angle(phi):
    """
    Wrap an angle into the half-open interval ``(-pi, pi]``.

    The interval is closed at ``+pi`` and open at ``-pi``, so an input of exactly
    ``-pi`` maps to ``+pi`` (and ``+pi`` stays ``+pi``). Works elementwise on
    Python floats and on ``torch.Tensor`` inputs alike.

    Parameters
    ----------
    phi : float or torch.Tensor
        Angle(s) in radians.

    Returns
    -------
    float or torch.Tensor
        The wrapped angle(s) in ``(-pi, pi]``, same type as the input.
    """
    return math.pi - (math.pi - phi) % (2.0 * math.pi)


def yaw_pose_content_and_phase(
    md: Dict[str, object], eps: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """Split target/context Cartesian poses into invariant content and phase.

    The reference is the target azimuth when it is defined, otherwise the
    largest-radius pose in the joint target/context bundle, and zero when every
    radius is below ``eps``. A
    degenerate context pose is assigned the shared reference phase so its
    relative phase is exactly zero.

    Unlike :func:`cylindrical_pose_features`, this helper does not replace any
    metadata fields.  It returns fresh tensors in a separate bundle and leaves
    ``md`` (including the Cartesian fields used by geometry conditioners)
    untouched.  Content retains the input floating dtype/device; phase outputs
    are explicitly float32 for the relative-phase attention path.
    """

    source = md.get("source")
    context = md.get("context_poses")
    if not isinstance(source, torch.Tensor):
        raise ValueError("yaw pose preprocessing requires tensor field 'source'")
    if source.shape == (1, 3):
        source = source[0]
    if source.shape != (3,):
        raise ValueError(
            f"source must have shape [3], got {tuple(source.shape)}"
        )
    if not source.is_floating_point():
        raise TypeError("source pose must be floating point")

    if context is None:
        context_tensor = source.new_empty((0, 3))
    else:
        if not isinstance(context, torch.Tensor):
            raise ValueError(
                "yaw pose preprocessing requires tensor field 'context_poses'"
            )
        context_tensor = context
        if context_tensor.shape == (3,):
            context_tensor = context_tensor.unsqueeze(0)
        if context_tensor.ndim != 2 or context_tensor.shape[-1] != 3:
            raise ValueError(
                "context_poses must have shape [K, 3], got "
                f"{tuple(context_tensor.shape)}"
            )
        if not context_tensor.is_floating_point():
            raise TypeError("context poses must be floating point")
        if context_tensor.device != source.device:
            raise ValueError("source and context poses must be on the same device")
        if context_tensor.dtype != source.dtype:
            raise ValueError("source and context poses must have the same dtype")

    sx, sy, sz = source.unbind(dim=-1)
    target_radius = torch.sqrt(sx * sx + sy * sy)
    target_azimuth = torch.atan2(sy, sx)
    target_content = torch.stack((target_radius, sz))

    if context_tensor.shape[0] > 0:
        cx, cy, cz = context_tensor.unbind(dim=-1)
        context_radii = torch.sqrt(cx * cx + cy * cy)
        context_azimuths = torch.atan2(cy, cx)
        context_content = torch.stack((context_radii, cz), dim=-1)
    else:
        context_radii = source.new_empty((0,))
        context_azimuths = source.new_empty((0,))
        context_content = source.new_empty((0, 2))

    if bool(target_radius >= eps):
        reference = target_azimuth
    else:
        candidate_radii = torch.cat((target_radius.reshape(1), context_radii))
        candidate_phases = torch.cat(
            (target_azimuth.reshape(1), context_azimuths)
        )
        largest = int(torch.argmax(candidate_radii))
        if bool(candidate_radii[largest] >= eps):
            reference = candidate_phases[largest]
        else:
            reference = torch.zeros_like(target_azimuth)

    if context_radii.numel() > 0:
        context_phases = torch.where(
            context_radii >= eps,
            context_azimuths,
            reference.expand_as(context_azimuths),
        )
    else:
        context_phases = context_azimuths

    return {
        "target_content": target_content,
        "target_phase": reference.float(),
        "context_content": context_content,
        "context_phases": context_phases.float(),
    }


def azimuth_rotation_matrix(alpha_rad: float) -> torch.Tensor:
    """
    Build a 3x3 rotation matrix about the vertical (z) axis.

    Parameters
    ----------
    alpha_rad : float
        Rotation angle in radians. Positive values rotate the (x, y) plane
        counter-clockwise, which corresponds to increasing azimuth.

    Returns
    -------
    torch.Tensor
        A ``[3, 3]`` float32 rotation matrix leaving the z component unchanged.
    """
    c = math.cos(alpha_rad)
    s = math.sin(alpha_rad)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def rotate_scene_metadata(
    md: Dict[str, object],
    alpha_rad: float,
    img_w: int,
    pose_keys: Tuple[str, ...] = POSE_KEYS,
) -> Dict[str, object]:
    """
    Apply a physically-consistent yaw rotation to a single sample's metadata.

    The equirectangular depth map is rolled horizontally by the integer number of
    columns closest to ``alpha_rad`` and its stored per-pixel 3D vectors are rotated
    about z. All pose vectors are rotated about z by the same effective angle.
    ``context_audio``, ``scene`` and any other field pass through unchanged. The
    rotation is exact (no interpolation) because the column shift is quantised and
    the rotation matrix uses the matching effective angle.

    Parameters
    ----------
    md : Dict[str, object]
        One per-sample metadata dict as produced by the AR/HAA metadata module.
        Expected tensor fields: ``depth`` of shape ``[3, H, W]`` and pose fields
        (``source``, ``source_vit``, ``context_poses``, ``context_poses_vit``)
        whose last dimension is 3.
    alpha_rad : float
        Requested yaw rotation angle in radians.
    img_w : int
        Panorama width in pixels (number of azimuth columns), e.g. 512.
    pose_keys : Tuple[str, ...], optional
        Which pose fields to rotate. Defaults to all of ``POSE_KEYS``. Restricting
        it (e.g. to the ``*_vit`` keys only) leaves
        the unlisted pose fields bit-identical; ``depth`` handling is unaffected.

    Returns
    -------
    Dict[str, object]
        A shallow-copied metadata dict with ``depth`` and the selected pose fields
        replaced by their rotated versions. The original dict is not mutated.
    """
    dj = int(round(alpha_rad * img_w / (2.0 * math.pi))) % img_w
    alpha_eff = dj * 2.0 * math.pi / img_w
    rot = azimuth_rotation_matrix(alpha_eff)

    out: Dict[str, object] = dict(md)

    if "depth" in md:
        depth = md["depth"]
        assert isinstance(depth, torch.Tensor)
        depth = torch.roll(depth, shifts=dj, dims=2)
        rot_d = rot.to(device=depth.device, dtype=depth.dtype)
        out["depth"] = torch.einsum("ij,jhw->ihw", rot_d, depth)

    for key in pose_keys:
        if key in md:
            pose = md[key]
            assert isinstance(pose, torch.Tensor)
            rot_p = rot.to(device=pose.device, dtype=pose.dtype)
            out[key] = torch.einsum("ij,...j->...i", rot_p, pose)

    return out


def yaw_transform_consistency(
    depth: torch.Tensor, img_w: int, angles_deg: Tuple[float, ...]
) -> Dict[float, float]:
    """
    Verify that yaw-rotating a depth panorama keeps it a valid equirectangular map.

    A consistent equirectangular depth point cloud stores, at column ``j``, a 3D
    vector whose horizontal azimuth equals that column's azimuth ``theta_j`` (because
    ``x = d*cos(phi)*cos(theta)`` and ``y = d*cos(phi)*sin(theta)``). A physically
    correct yaw rotation --- a horizontal roll of ``dj`` columns together with a
    matching rotation of the stored vectors about z --- must preserve this property.
    If the roll direction and the rotation-matrix sign disagree, the rotated map is
    no longer a valid panorama and the per-column azimuth deviation becomes large
    (about twice the rotation angle). This is the automated sign sanity-check.

    Parameters
    ----------
    depth : torch.Tensor
        A single depth point cloud of shape ``[3, H, W]`` in the panorama camera frame.
    img_w : int
        Panorama width in pixels (number of azimuth columns).
    angles_deg : Tuple[float, ...]
        Yaw angles in degrees to test for consistency (0 deg is always included as a
        control).

    Returns
    -------
    Dict[float, float]
        Mapping from angle in degrees to the maximum per-pixel azimuth deviation in
        radians, excluding near-pole pixels where azimuth is undefined. Values near
        zero indicate a geometrically consistent transform.
    """
    j = torch.arange(img_w, device=depth.device, dtype=depth.dtype)
    theta = ((j + 0.5) * 2.0 * math.pi / img_w - math.pi).unsqueeze(0)

    def max_dev(pc: torch.Tensor) -> float:
        x, y = pc[0], pc[1]
        az = torch.atan2(y, x)
        diff = (az - theta + math.pi) % (2.0 * math.pi) - math.pi
        mag = torch.sqrt(x * x + y * y)
        mask = mag > 1e-3
        if bool(mask.sum() == 0):
            return float("nan")
        return float(diff[mask].abs().max().item())

    report = {0.0: max_dev(depth)}
    for deg in angles_deg:
        rotated = rotate_scene_metadata({"depth": depth}, math.radians(deg), img_w)["depth"]
        assert isinstance(rotated, torch.Tensor)
        report[float(deg)] = max_dev(rotated)
    return report
