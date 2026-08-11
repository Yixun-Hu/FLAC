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
from typing import Dict, List, Sequence, Tuple

import torch

POSE_KEYS: Tuple[str, ...] = ("source", "source_vit", "context_poses", "context_poses_vit")

# Yaw subgroup G = C4 aligned with the panorama columns (90 deg = 128 columns of
# W=512 -> exact roll). Single source of truth for the frame-averaging angles.
DEFAULT_FRAME_ANGLES: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)

# Largest number of samples fed to the ViT conditioners in ONE frame-averaging
# forward. The orbit used to run one angle at a time: at the training micro-batch
# (8) that is a ~8-sample, ~512-token ViT forward per angle per conditioner, and
# exp_11's P0 profiling measured every orbit cell at only 28-34% GPU utilisation
# -- latency/kernel-launch bound, not compute bound. Stacking the rotated
# variants along the batch dimension turns C small forwards into a few large
# ones. 64 is a PER-RANK conditioner batch equal to the effective global batch,
# so one orbit chunk costs about one ordinary step's worth of activations; the
# batched peak is requalified by the 8x8 P0 spot cells (real forward + backward +
# checkpoint recompute), not inferred from the sequential measurement.
#
# DISCLOSED RECIPE CHANGE (exp_11, approved as such). The averaging ARITHMETIC is
# identical -- same terms, same order, same divisor. The augmentation SCHEDULE is
# not: train-mode DINOv3 draws a random RoPE position rescale once per forward
# (pos_embed_rescale 2.0), so angles sharing a chunk now share one draw where the
# per-angle loop gave them independent draws, and fewer draws are taken overall.
# This is applied identically to every arm INCLUDING the contemporaneous C4L
# bridge, which is therefore the sole inferential comparator; historical rows
# evaluated under the per-angle loop are labelled legacy-loop. In eval mode the
# rescale is off, so eval-mode batched and loop results agree numerically.
FRAME_AVG_MAX_FWD_SAMPLES: int = 64

# Recorded in evaluation provenance so a metrics row states which orbit execution
# produced it ('batched' here, 'loop' for the legacy per-angle path).
ORBIT_EXECUTION: str = "batched"


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


def draw_yaw_offsets(n: int, img_w: int, generator: torch.Generator) -> torch.Tensor:
    """
    Draw ``n`` independent panorama-column yaw offsets ``d ~ Uniform{0..img_w-1}``.

    The unit of a *drawn* rotation is one panorama column, never a degree: an
    integer column shift is applied by :func:`rotate_scene_metadata` with no
    interpolation, so the offset that is recorded is exactly the offset that is
    applied (``dj == d``; pinned by test). Angles come from
    :func:`offsets_to_radians`.

    The draw is taken from the CALLER'S generator and from nothing else. Two
    properties follow, and both are load-bearing for exp_14's design:

    * **Global-RNG isolation.** The evaluation's own sampling noise is drawn from
      the global RNG seeded by ``pl.seed_everything(eval_seed)``. If the yaw draw
      advanced that stream, a rotated cell and its paired unrotated cell would
      sample different diffusion noise and the seed-paired contrast would be void.
    * **Chunk independence.** Consecutive calls consume one contiguous stream, so
      any batch partition of the same item order yields the same per-item
      assignment; the batch size is pinned anyway, but the assignment does not
      depend on that pin holding.

    Parameters
    ----------
    n : int
        Number of offsets (e.g. the current batch size). ``0`` is legal — a tail
        batch of zero items consumes no randomness.
    img_w : int
        Panorama width in columns (the support is ``{0, ..., img_w-1}``).
    generator : torch.Generator
        A dedicated CPU generator, seeded with the rotation seed. Required: a
        ``None`` here would silently fall back to the global RNG.

    Returns
    -------
    torch.Tensor
        ``int64`` tensor of shape ``[n]`` with values in ``[0, img_w)``.
    """
    if generator is None:
        raise ValueError(
            "draw_yaw_offsets requires a dedicated torch.Generator: passing None "
            "would draw from the GLOBAL RNG and perturb the evaluation's own "
            "sampling noise (see the docstring)."
        )
    n = int(n)
    img_w = int(img_w)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if img_w <= 0:
        raise ValueError(f"img_w must be > 0, got {img_w}")
    return torch.randint(0, img_w, (n,), generator=generator, dtype=torch.long)


def offsets_to_radians(offsets: Sequence, img_w: int) -> List[float]:
    """
    Convert panorama-column offsets to yaw angles in radians: ``d * 2*pi / img_w``.

    Exact by construction on the column grid — :func:`rotate_scene_metadata`
    re-quantises the angle back to ``round(alpha * img_w / 2pi) % img_w`` and
    recovers the same ``d`` for every offset in the support (pinned by test), so
    the recorded offset and the applied rotation can never disagree.

    Parameters
    ----------
    offsets : Sequence
        Column offsets: a ``torch.Tensor``, list, or any iterable of integers.
    img_w : int
        Panorama width in columns.

    Returns
    -------
    List[float]
        One angle in radians per offset, in the input order.
    """
    img_w = int(img_w)
    if img_w <= 0:
        raise ValueError(f"img_w must be > 0, got {img_w}")
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    return [float(int(d)) * 2.0 * math.pi / float(img_w) for d in offsets]


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
    If *every* pose is degenerate (largest ``r`` also ``< eps``) every ``dphi``
    is exactly ``0``.

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
    # The fallback pose must itself be non-degenerate (its azimuth defined);
    # if even the largest r is < eps the scene is all-degenerate -> dphi == 0.
    phi_ref = None
    if r_s is not None and bool(r_s >= eps):
        phi_ref = phi_s
    elif cand_r:
        all_r = torch.cat(cand_r)
        all_phi = torch.cat(cand_phi)
        max_idx = int(torch.argmax(all_r))
        if bool(all_r[max_idx] >= eps):
            phi_ref = all_phi[max_idx]

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


def invariant_conditioning(
    conditioner,
    metadata: "list",
    device,
    angles: Tuple[float, ...] = DEFAULT_FRAME_ANGLES,
    vit_ids: Tuple[str, ...] = ("source_vit", "context_poses_vit"),
) -> Dict[str, object]:
    """
    Compute Route-1 yaw-symmetrized FLAC conditioning.

    Two sub-paths are symmetrized independently (see the exp_03 plan §2b REVISED):

    1. **Pose path** (``source`` / ``context_poses`` dist-embedders): the absolute
       ``(x, y, z)`` poses are replaced by intrinsically yaw-invariant cylindrical
       features ``(r, z, dphi)`` via :func:`cylindrical_pose_features`. These are
       exactly invariant at *any* yaw angle.
    2. **ViT path** (``vit_ids``, e.g. the DINOv3 depth-panorama conditioners): a
       **frame average** ``(1/|G|) sum_g f(g . x)`` over ``angles``. The depth
       map and the ``*_vit`` pose keys are rotated *together* by each frame angle;
       averaging their conditioner outputs makes the result exactly invariant on
       the panorama-aligned yaw subgroup G.

    All remaining conditioners (e.g. the ``context_audio`` RIR encoder, which is
    already yaw-invariant and may carry BatchNorm running stats) are run **exactly
    once** — only the ``vit_ids`` conditioners see the rotated frames. Those are
    executed in BATCHED chunks of at most :data:`FRAME_AVG_MAX_FWD_SAMPLES`
    samples rather than one forward per angle (the orbit passes are
    latency-bound; see the constant). The averaging arithmetic, the accumulation
    order and the returned structure are unchanged; train-mode DINOv3's random
    RoPE rescale becomes chunk-shared, a disclosed recipe change applied
    identically to every arm. The caller's
    ``metadata`` is never mutated (its ``source`` / ``depth`` stay raw for the
    downstream metric callback).

    Parameters
    ----------
    conditioner : MultiConditioner
        The FLAC conditioner. Must accept ``conditioner(batch, device)`` and the
        optional ``only_ids=`` keyword (to re-run a subset of conditioners).
    metadata : list of dict
        Per-sample metadata dicts. ``depth`` (shape ``[3, H, W]``), the pose keys
        and any conditioner inputs are read but not modified.
    device : torch.device or str
        Device passed through to ``conditioner``.
    angles : Tuple[float, ...], optional
        Frame-average angles in **degrees**; ``angles[0]`` must be ``0.0`` (the
        identity / base pass). Defaults to :data:`DEFAULT_FRAME_ANGLES`.
    vit_ids : Tuple[str, ...], optional
        Conditioning ids whose outputs are frame-averaged (the ViT depth path).

    Returns
    -------
    Dict[str, object]
        The conditioner output dict ``{id: [tensor, mask]}``; the ``vit_ids``
        tensors are replaced by their frame average, all masks come from the base
        (angle-0) pass, and all other entries are the single base-pass outputs.

    Raises
    ------
    ValueError
        If ``angles`` is empty or ``angles[0] != 0.0``.
    """
    if len(angles) == 0:
        raise ValueError("angles must be non-empty")
    if angles[0] != 0.0:
        raise ValueError(f"angles[0] must be 0.0 (the identity pass), got {angles[0]}")

    # 1. cylindrical (yaw-invariant) pose features; never mutates the caller's md.
    md_inv = [cylindrical_pose_features(md) for md in metadata]

    # 2. single full pass -> all conditioning entries (non-ViT ids run once here).
    base = conditioner(md_inv, device)

    present = [i for i in vit_ids if i in base]
    if not present or len(angles) == 1 or "depth" not in metadata[0]:
        return base

    # 3. frame-average ONLY the ViT conditioners over the rotation subgroup.
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = _orbit_average_batched(conditioner, md_inv, base, present, angles, img_w, device)

    for i in present:
        base[i][0] = accum[i] / float(len(angles))

    return base


def _rotated_variants(md_inv, deg, img_w, present):
    """The rotated metadata for one orbit angle (depth and the ``*_vit`` poses
    move together; every other field passes through)."""
    return [
        rotate_scene_metadata(m, math.radians(deg), img_w, pose_keys=tuple(present))
        for m in md_inv
    ]


def _orbit_average_loop(conditioner, md_inv, base, present, angles, img_w, device):
    """Reference orbit accumulation: ONE conditioner forward per angle.

    This is the pre-batching execution order, kept as the equivalence reference
    for :func:`_orbit_average_batched` (tests and the exp_11 equivalence probe
    compare against *this code*, not against a copy of it). It is not used on the
    training path any more.

    Returns ``{id: summed_tensor}`` over the whole orbit, angle 0 first.
    """
    accum = {i: base[i][0].clone() for i in present}
    for g in angles[1:]:
        part = conditioner(_rotated_variants(md_inv, g, img_w, present), device,
                           only_ids=present)
        for i in present:
            accum[i] = accum[i] + part[i][0]
    return accum


def _orbit_average_batched(conditioner, md_inv, base, present, angles, img_w, device):
    """Same orbit sum as :func:`_orbit_average_loop`, executed as a few large forwards.

    The rotated variants of consecutive angles are concatenated along the batch
    dimension into chunks of at most :data:`FRAME_AVG_MAX_FWD_SAMPLES` samples
    (whole angles per chunk, so a chunk is ``k * B`` samples), each chunk is one
    conditioner call, and the stacked output is split back per angle. Terms are
    accumulated in the INPUT ANGLE ORDER starting from the angle-0 base, so the
    additions are never reassociated: the averaging arithmetic is identical to
    the loop.

    What is NOT identical is the augmentation schedule under train-mode DINOv3:
    its random RoPE rescale is drawn once per forward, so a chunk's angles share
    a draw (see :data:`FRAME_AVG_MAX_FWD_SAMPLES` for the disclosure). In eval
    mode there is no such draw and the two paths agree numerically.

    Only the ``present`` (ViT) ids are re-run; masks come from the base pass.
    A batch larger than the cap is rejected rather than silently exceeding it,
    because a chunk cannot be smaller than one angle.
    """
    accum = {i: base[i][0].clone() for i in present}
    rest = angles[1:]
    if not rest:
        return accum
    batch = len(md_inv)
    if batch > FRAME_AVG_MAX_FWD_SAMPLES:
        raise ValueError(
            f"batch {batch} exceeds FRAME_AVG_MAX_FWD_SAMPLES "
            f"({FRAME_AVG_MAX_FWD_SAMPLES}): a frame-average chunk is whole angles, so it "
            "cannot be split below one angle -- lower the batch or raise the cap deliberately"
        )
    angles_per_chunk = max(1, FRAME_AVG_MAX_FWD_SAMPLES // batch)
    for start in range(0, len(rest), angles_per_chunk):
        chunk = rest[start:start + angles_per_chunk]
        stacked = []
        for g in chunk:
            stacked.extend(_rotated_variants(md_inv, g, img_w, present))
        part = conditioner(stacked, device, only_ids=present)
        for i in present:
            out = part[i][0]
            if out.shape[0] != len(chunk) * batch:
                raise RuntimeError(
                    f"{i}: batched orbit forward returned {out.shape[0]} rows for "
                    f"{len(chunk)} angle(s) x {batch} sample(s)"
                )
            for k in range(len(chunk)):          # ascending angle order preserved
                accum[i] = accum[i] + out[k * batch:(k + 1) * batch]
    return accum


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
