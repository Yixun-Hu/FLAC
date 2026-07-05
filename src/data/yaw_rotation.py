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

# Yaw subgroup G = C4 aligned with the panorama columns (90 deg = 128 columns of
# W=512 -> exact roll). Single source of truth for the frame-averaging angles.
DEFAULT_FRAME_ANGLES: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)


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


def cylindrical_pose_features(
    md: Dict[str, object], eps: float = 1e-6
) -> Dict[str, object]:
    """
    Replace absolute pose vectors with yaw-invariant cylindrical features.

    Each pose ``(x, y, z)`` becomes ``(r, z, dphi)`` where ``r = sqrt(x^2 + y^2)``
    and ``dphi`` is the azimuth measured *relative* to a scene-intrinsic reference
    azimuth. Because every azimuth in a rigidly yaw-rotated scene shifts by the
    same amount, the relative ``dphi`` (and ``r``, ``z``) are invariant under any
    yaw rotation, making these features an exact invariant of the C-infinity yaw
    group.

    The reference azimuth is the target source's azimuth. If the source lies on
    the vertical axis (``r_s < eps``, azimuth undefined) the reference falls back
    to the azimuth of the largest-``r`` pose among ``{source, context poses}`` --
    still a scene-intrinsic quantity, so invariance is preserved in the fallback.
    If *every* pose is degenerate (all on the z-axis) every ``dphi`` is ``0``.

    Only ``'source'`` and ``'context_poses'`` are transformed; ``'*_vit'`` poses,
    ``'depth'`` and all other keys pass through untouched (same objects). The
    input dict is not mutated: a shallow copy is returned with fresh tensors for
    the replaced keys, preserving each pose tensor's dtype and device.

    Parameters
    ----------
    md : Dict[str, object]
        One per-sample metadata dict. ``'source'`` (shape ``[3]``) and, optionally,
        ``'context_poses'`` (shape ``[N, 3]``) are read; the last dimension is
        ordered ``(x, y, z)``.
    eps : float, optional
        Radius below which a pose azimuth is treated as undefined (default 1e-6).

    Returns
    -------
    Dict[str, object]
        A shallow copy of ``md`` with ``'source'`` -> ``(r_s, z_s, 0.0)`` and each
        ``'context_poses'`` row -> ``(r_i, z_i, wrap(phi_i - phi_ref))``.
    """
    out: Dict[str, object] = dict(md)

    source = md.get("source")
    context = md.get("context_poses")

    # Candidate poses for the fallback reference azimuth: source + context rows.
    cand_r = []
    cand_phi = []

    r_s = phi_s = sz = None
    if source is not None:
        assert isinstance(source, torch.Tensor)
        sx, sy, sz = source[..., 0], source[..., 1], source[..., 2]
        r_s = torch.sqrt(sx * sx + sy * sy)
        phi_s = torch.atan2(sy, sx)
        cand_r.append(r_s.reshape(1))
        cand_phi.append(phi_s.reshape(1))

    r_c = phi_c = cz = None
    if context is not None:
        assert isinstance(context, torch.Tensor)
        cx, cy, cz = context[..., 0], context[..., 1], context[..., 2]
        r_c = torch.sqrt(cx * cx + cy * cy)
        phi_c = torch.atan2(cy, cx)
        cand_r.append(r_c.reshape(-1))
        cand_phi.append(phi_c.reshape(-1))

    # Reference azimuth: target source, unless degenerate -> largest-r pose.
    if r_s is not None and bool(r_s >= eps):
        phi_ref = phi_s
    elif cand_r:
        all_r = torch.cat(cand_r)
        all_phi = torch.cat(cand_phi)
        phi_ref = all_phi[int(torch.argmax(all_r))]
    else:
        phi_ref = None

    if source is not None:
        out["source"] = torch.stack([r_s, sz, torch.zeros_like(r_s)])

    if context is not None:
        if phi_ref is not None:
            dphi = wrap_angle(phi_c - phi_ref)
        else:
            dphi = torch.zeros_like(r_c)
        out["context_poses"] = torch.stack([r_c, cz, dphi], dim=-1)

    return out


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
        Which pose fields to rotate. Defaults to all of ``POSE_KEYS`` (the
        exp_02 behaviour). Restricting it (e.g. to the ``*_vit`` keys only) leaves
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
