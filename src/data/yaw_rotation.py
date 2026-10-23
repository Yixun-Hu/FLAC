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
#
# This is the DEFAULT, not the only value: announcement 06 requires every FA arm
# to declare its effective chunk plan, and exp_14 makes the cap a per-arm config
# key (``training.frame_avg_max_fwd_samples``) passed to
# :func:`invariant_conditioning` as ``max_fwd_samples``. Nothing assigns to this
# name at runtime -- see :func:`resolve_frame_avg_cap`.
FRAME_AVG_MAX_FWD_SAMPLES: int = 64

# Recorded in evaluation provenance so a metrics row states which orbit execution
# produced it ('batched' here, 'loop' for the legacy per-angle path).
ORBIT_EXECUTION: str = "batched"


def resolve_frame_avg_cap(max_fwd_samples=None) -> int:
    """The frame-average forward cap this call must use.

    ``None`` -> :data:`FRAME_AVG_MAX_FWD_SAMPLES` (64), i.e. exactly the
    behaviour every recipe already in the record was run under. Anything else is
    validated and returned as-is; the module global is never rebound, because the
    cap is a per-arm property (exp_14's ``training.frame_avg_max_fwd_samples``)
    and a process-wide assignment would (a) leak into whatever else the process
    later conditions and (b) survive a mid-step crash as a cap nobody declared.

    Validation is on the RAW value and fail-closed. ``bool`` is rejected before
    ``int`` because ``isinstance(True, int)`` is True and ``max(1, 64 // True)``
    would quietly become a 64-angle chunk plan; a float or a numeric string is
    rejected rather than coerced, because ``int("32")`` succeeding on a typo is
    how an arm gets trained under a cap its config never stated.
    """
    if max_fwd_samples is None:
        return FRAME_AVG_MAX_FWD_SAMPLES
    if isinstance(max_fwd_samples, bool) or not isinstance(max_fwd_samples, int):
        raise ValueError(
            f"max_fwd_samples must be an int (the frame-average per-forward sample "
            f"cap), got {max_fwd_samples!r}"
        )
    if max_fwd_samples < 1:
        raise ValueError(f"max_fwd_samples must be >= 1, got {max_fwd_samples}")
    return int(max_fwd_samples)


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


def yaw_column_shift(alpha_rad: float, img_w: int) -> int:
    """
    The integer panorama-column shift a yaw angle quantises to: ``round(a*W/2pi) % W``.

    This is the rule :func:`rotate_scene_metadata` applies, extracted so that code
    which must *record* the applied shift (exp_14's assignment audit, which stores
    a fixed-mode cell's constant offset) reads it from the same place the rotation
    does. Two copies of this rule could drift, and a drifted copy would label a
    cell with an offset it never applied --- the one thing the audit exists to make
    impossible.

    Parameters
    ----------
    alpha_rad : float
        Requested yaw rotation angle in radians (any sign, any magnitude).
    img_w : int
        Panorama width in columns.

    Returns
    -------
    int
        The column shift in ``[0, img_w)``.
    """
    return int(round(alpha_rad * img_w / (2.0 * math.pi))) % int(img_w)


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
    dj = yaw_column_shift(alpha_rad, img_w)
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
    max_fwd_samples: "int | None" = None,
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
    executed in BATCHED chunks of at most ``max_fwd_samples`` samples (default
    :data:`FRAME_AVG_MAX_FWD_SAMPLES`) rather than one forward per angle (the
    orbit passes are latency-bound; see the constant). That partition IS the
    method under announcement 06 -- it decides how many angles share one
    train-mode RoPE draw -- so exp_14 declares it per arm rather than inheriting
    a module constant. The averaging arithmetic, the accumulation
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
    max_fwd_samples : int or None, optional
        Largest number of samples in ONE frame-averaging forward, i.e. the thing
        that decides how many angles share a chunk (and therefore share a
        train-mode RoPE draw — announcement 06). ``None`` (the default) means
        :data:`FRAME_AVG_MAX_FWD_SAMPLES`, exactly today's behaviour. exp_14
        passes the arm's ``training.frame_avg_max_fwd_samples`` here so the chunk
        plan is a declared, checkpoint-embedded property of the arm rather than
        an ambient constant.

    Returns
    -------
    Dict[str, object]
        The conditioner output dict ``{id: [tensor, mask]}``; the ``vit_ids``
        tensors are replaced by their frame average, all masks come from the base
        (angle-0) pass, and all other entries are the single base-pass outputs.

    Raises
    ------
    ValueError
        If ``angles`` is empty, ``angles[0] != 0.0``, or ``max_fwd_samples`` is
        not a positive integer.
    """
    if len(angles) == 0:
        raise ValueError("angles must be non-empty")
    if angles[0] != 0.0:
        raise ValueError(f"angles[0] must be 0.0 (the identity pass), got {angles[0]}")
    # Validated HERE, before the short-circuit returns below: an invalid cap must
    # abort even on an orbit that happens to need no chunking, or a typo'd arm
    # could train for days without the treatment ever being read.
    cap = resolve_frame_avg_cap(max_fwd_samples)

    # 1. cylindrical (yaw-invariant) pose features; never mutates the caller's md.
    md_inv = [cylindrical_pose_features(md) for md in metadata]

    # 2. single full pass -> all conditioning entries (non-ViT ids run once here).
    base = conditioner(md_inv, device)

    present = [i for i in vit_ids if i in base]
    if not present or len(angles) == 1 or "depth" not in metadata[0]:
        return base

    # 3. frame-average ONLY the ViT conditioners over the rotation subgroup.
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = _orbit_average_batched(conditioner, md_inv, base, present, angles, img_w, device,
                                   cap=cap, cap_is_explicit=max_fwd_samples is not None)

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


def _orbit_average_batched(conditioner, md_inv, base, present, angles, img_w, device,
                           cap=None, cap_is_explicit=False):
    """Same orbit sum as :func:`_orbit_average_loop`, executed as a few large forwards.

    The rotated variants of consecutive angles are concatenated along the batch
    dimension into chunks of at most ``cap`` samples (default
    :data:`FRAME_AVG_MAX_FWD_SAMPLES`; whole angles per chunk, so a chunk is
    ``k * B`` samples), each chunk is one conditioner call, and the stacked output
    is split back per angle. Terms are accumulated in the INPUT ANGLE ORDER
    starting from the angle-0 base, so the additions are never reassociated: the
    averaging arithmetic is identical to the loop.

    ``cap_is_explicit`` only selects which knob the over-cap error names — the
    module constant, or the config key that overrode it. A caller that hands us a
    cap it read from a config must not be told to edit a module global.

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
    if cap is None:
        cap = FRAME_AVG_MAX_FWD_SAMPLES
    cap_name = ("training.frame_avg_max_fwd_samples" if cap_is_explicit
                else "FRAME_AVG_MAX_FWD_SAMPLES")
    batch = len(md_inv)
    if batch > cap:
        raise ValueError(
            f"batch {batch} exceeds {cap_name} "
            f"({cap}): a frame-average chunk is whole angles, so it "
            "cannot be split below one angle -- lower the batch or raise the cap deliberately"
        )
    angles_per_chunk = max(1, cap // batch)
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


# --------------------------------------------------------------------------- #
# exp_25 depth_canon: deepest-pixel yaw canonicalization (plan_depth_canon.md
# §2.1 / §8). Additions only -- nothing above this line is changed.
# --------------------------------------------------------------------------- #
def _check_depth_cloud(depth) -> None:
    """Fail-closed contract of a metadata ``depth`` point cloud ``[3, H, W]``."""
    if not isinstance(depth, torch.Tensor):
        raise ValueError(f"depth must be a torch.Tensor, got {type(depth).__name__}")
    if not depth.is_floating_point():
        raise ValueError(f"depth must be a floating tensor, got {depth.dtype}")
    if depth.ndim != 3 or depth.shape[0] != 3:
        raise ValueError(f"depth must have shape [3, H, W], got {list(depth.shape)}")
    H, W = int(depth.shape[1]), int(depth.shape[2])
    if H == 0 or W == 0 or H % 2 != 0:
        raise ValueError(
            f"depth needs an even H > 0 and W > 0, got H={H}, W={W} (an odd H puts a "
            "panorama row at phi = 0, where the depth proxy's denominator is zero)")
    if not bool(torch.isfinite(depth).all()):
        raise ValueError("depth contains non-finite values")


def recover_depth_proxy(depth: torch.Tensor) -> torch.Tensor:
    """
    ``[3, H, W]`` camera-frame point cloud -> ``[H, W]`` recovered-depth PROXY.

    ``D[h, w] = depth[2, h, w] / (-sin(phi_h))`` with
    ``phi_h = (h + 0.5) * pi / H - pi / 2`` built exactly as
    ``AR_md.convert_equirect_to_camera_coord`` builds its row angle
    (``torch.arange`` -> float32 angle -> ``torch.sin``), cast to ``depth.dtype``
    only at the division.

    This is a **bitwise yaw-stable proxy**, NOT an exact inverse of the raw depth
    map: AR_md stores ``z = -d * sin(phi)`` in float64, and dividing back agrees
    with the raw ``.npy`` only to ~1e-15 (the planning probe found 8.9e-16 and 91 %
    of pixels bitwise). What IS exact is its behaviour under
    :func:`rotate_scene_metadata`: a yaw rotation leaves the z channel bitwise
    unchanged (matrix row ``[0, 0, 1]``) and ``sin(phi_h)`` is a per-row constant,
    so the proxy of a rotated map, un-rolled, equals the original proxy bitwise --
    which the per-pixel L2 norm of the point cloud does not.

    Raises ``ValueError`` unless ``depth`` is a finite floating ``[3, H, W]``
    tensor with even ``H > 0`` and ``W > 0``, or if any row has ``sin(phi) == 0``.
    """
    _check_depth_cloud(depth)
    H = int(depth.shape[1])
    phi = (torch.arange(H) + 0.5) * math.pi / H - math.pi / 2  # float32, as in AR_md
    s = torch.sin(phi)
    if not bool(s.abs().min() > 0.0):
        raise ValueError("a panorama row has sin(phi) == 0: the depth proxy is undefined")
    return depth[2] / (-s.to(device=depth.device, dtype=depth.dtype))[:, None]


def column_profile(D: torch.Tensor) -> torch.Tensor:
    """Per-column maximum ``P[w] = max_h D[h, w]`` of an ``[H, W]`` depth proxy -> ``[W]``."""
    if not isinstance(D, torch.Tensor) or not D.is_floating_point() or D.ndim != 2 \
            or D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError(
            f"column_profile needs a non-empty floating [H, W] tensor, got "
            f"{getattr(D, 'dtype', type(D).__name__)} {list(getattr(D, 'shape', []))}")
    return D.max(dim=0).values


TIE_CLASSES: Tuple[str, ...] = ("none", "same_column", "cross_column_image",
                                "cross_column_source", "cross_column_context")


class StabilizedTieError(RuntimeError):
    """Plan §2.1 C-4: several candidate columns survive EVERY tie-break key.

    Only an exactly periodic depth proxy with azimuth-degenerate source and context
    poses gets here -- a joint input with a nontrivial cyclic stabilizer, on which no
    deterministic selector is equivariant (``min`` of {0, 256} rolled by 300 is 44,
    not 300). Canonicalization therefore FAILS CLOSED; this is an exception, never a
    tie class. ``candidates`` holds the surviving columns in ascending order."""

    def __init__(self, candidates):
        self.candidates = tuple(int(c) for c in candidates)
        super().__init__(
            f"stabilized canonicalization tie: candidate columns {list(self.candidates)} "
            "survive the rooted-image, source and context keys; no equivariant choice "
            "exists, failing closed (plan_depth_canon.md §2.1 C-4)")

    def __reduce__(self):  # picklable despite the non-message constructor argument
        return (type(self), (self.candidates,))


def _check_pose(name: str, pose, shape: Tuple) -> None:
    """Exact-shape (``None`` = free size), floating, finite pose contract."""
    if not isinstance(pose, torch.Tensor) or not pose.is_floating_point():
        raise ValueError(f"{name} must be a floating torch.Tensor, got "
                         f"{getattr(pose, 'dtype', type(pose).__name__)}")
    if pose.ndim != len(shape) or any(s is not None and int(n) != s
                                      for n, s in zip(pose.shape, shape)):
        raise ValueError(f"{name} must have shape {list(shape)} (None = any), got "
                         f"{list(pose.shape)}")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"{name} contains non-finite values")


def _lex_compare(a: torch.Tensor, b: torch.Tensor) -> int:
    """Exact lexicographic order of two equal-length 1-D tensors: -1, 0 or +1."""
    diff = torch.nonzero(a != b)
    if diff.numel() == 0:
        return 0
    k = int(diff[0, 0])  # nonzero() returns indices in increasing order
    return 1 if bool(a[k] > b[k]) else -1


def _image_key_survivors(D: torch.Tensor, cands: List[int]) -> List[int]:
    """Step C-1: the candidates whose ROOTED image -- columns c, c+1, ... (mod W),
    each read top to bottom -- is lexicographically LARGEST (exact comparison).
    The rooted image is read from the candidate anchor, so it co-rotates."""
    H = int(D.shape[0])
    flat = D.t().contiguous().view(-1)  # column-major: column w is flat[w*H:(w+1)*H]
    best, best_img = [cands[0]], torch.roll(flat, shifts=-cands[0] * H)
    for c in cands[1:]:
        img = torch.roll(flat, shifts=-c * H)
        order = _lex_compare(img, best_img)
        if order > 0:
            best, best_img = [c], img
        elif order == 0:
            best.append(c)
    return best


# Radius below which a pose's azimuth is undefined (as in cylindrical_pose_features).
_AZIMUTH_EPS: float = 1e-6


def _relative_pose_keys(poses: torch.Tensor, c: int, W: int) -> List[Tuple[float, float, float]]:
    """Per pose row, in ORIGINAL row order: ``(r, z, (atan2(y, x) - theta_c) mod 2pi)``
    in float64 with ``theta_c = (c + 0.5) * 2pi / W - pi`` (column ``c``'s centre
    azimuth); the azimuth is ``+inf`` when ``r < 1e-6``. The wrap is taken on the
    DIFFERENCE, so atan2's +-pi branch cut never enters a comparison."""
    p = poses.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 3)
    r = torch.hypot(p[:, 0], p[:, 1])
    theta_c = (c + 0.5) * 2.0 * math.pi / W - math.pi
    rel = torch.remainder(torch.atan2(p[:, 1], p[:, 0]) - theta_c, 2.0 * math.pi)
    rel = torch.where(r < _AZIMUTH_EPS, torch.full_like(rel, math.inf), rel)
    return list(zip(r.tolist(), p[:, 2].tolist(), rel.tolist()))


def _min_key_survivors(cands: List[int], key) -> List[int]:
    """The candidates whose ``key(c)`` is the (exact) minimum."""
    keys = {c: key(c) for c in cands}
    best = min(keys.values())
    return [c for c in cands if keys[c] == best]


def canonical_column(D: torch.Tensor, source=None, context=None) -> Tuple[int, str]:
    """
    The canonical ("deepest") column ``j*`` of an ``[H, W]`` depth proxy, with its
    tie class (plan_depth_canon.md §2.1 Steps B-C).

    Candidates are the columns whose maximum equals the global maximum (exact
    equality). A single candidate is returned with class ``"same_column"`` when
    several pixels of that column share the global maximum, else ``"none"``.
    Several candidates are resolved by keys that are rotation-invariant objects
    read from each candidate's anchor, so the choice co-rotates with the panorama:

    1. the full rooted image (largest wins) -> ``"cross_column_image"``;
    2. the target source's relative azimuth ``(atan2(y, x) - theta_c) mod 2pi``,
       ``theta_c = (c + 0.5) * 2pi / W - pi`` (smallest wins; skipped when the
       source radius is ``< 1e-6``) -> ``"cross_column_source"``;
    3. the ORDERED, pair-preserving context key: for each context row in its
       original order the tuple ``(r_i, z_i, relative azimuth_i)`` (azimuth
       ``+inf`` when ``r_i < 1e-6``), compared lexicographically over the rows
       (smallest wins) -> ``"cross_column_context"``. Rows are never permuted,
       so the pairing with the untouched ``context_audio`` rows is preserved;

    and otherwise (exactly periodic proxy, degenerate source, identical ordered
    contexts) the selector fails closed with :class:`StabilizedTieError`.
    Keys 2-3 are reached only on an exactly periodic proxy. There the selector
    has the measure-zero discontinuity every canonicalizer has: a deciding pose
    within float32 rotation round-off (~1e-7 rad) of a candidate's column centre
    (relative azimuth ~0 vs ~2pi) may resolve differently after a rotation.

    ``source`` (``[3]``) and ``context`` (``[N, 3]``) are validated when given.
    Every comparison is exact; there is no tolerance anywhere.
    """
    P = column_profile(D)
    if not bool(torch.isfinite(D).all()):
        raise ValueError("depth proxy contains non-finite values")
    if source is not None:
        _check_pose("source", source, (3,))
    if context is not None:
        _check_pose("context", context, (None, 3))
    M = P.max()
    cands = torch.nonzero(P == M).flatten().tolist()
    if len(cands) == 1:
        n_max = int((D[:, cands[0]] == M).sum())
        return int(cands[0]), ("same_column" if n_max > 1 else "none")
    cands = _image_key_survivors(D, cands)
    if len(cands) == 1:
        return int(cands[0]), "cross_column_image"
    W = int(D.shape[1])
    # source radius is anchor-independent: +inf at one anchor <=> degenerate source
    if source is not None and _relative_pose_keys(source, 0, W)[0][2] < math.inf:
        cands = _min_key_survivors(cands, lambda c: _relative_pose_keys(source, c, W)[0][2])
        if len(cands) == 1:
            return int(cands[0]), "cross_column_source"
    if context is not None and int(context.shape[0]) > 0:
        cands = _min_key_survivors(cands, lambda c: _relative_pose_keys(context, c, W))
        if len(cands) == 1:
            return int(cands[0]), "cross_column_context"
    raise StabilizedTieError(cands)


def canonical_shift(j_star: int, img_w: int) -> int:
    """The roll ``dj = (img_w - j*) mod img_w`` that moves column ``j*`` to column 0
    (:func:`rotate_scene_metadata` rolls by ``+dj``). ``dj == 0`` iff ``j* == 0``."""
    for name, v in (("j_star", j_star), ("img_w", img_w)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be an int, got {v!r}")
    if img_w <= 0 or not 0 <= j_star < img_w:
        raise ValueError(f"need img_w > 0 and 0 <= j_star < img_w, got {j_star}, {img_w}")
    return (img_w - j_star) % img_w


# Exact pose shapes of the AR metadata contract (None = the context count N >= 1).
_CANON_POSE_SHAPES: Dict[str, Tuple] = {"source": (3,), "source_vit": (1, 3),
                                        "context_poses": (None, 3),
                                        "context_poses_vit": (None, 3)}


def _check_canon_inputs(md, img_w) -> None:
    """Fail-closed input contract of :func:`canonicalize_scene_metadata`."""
    if not isinstance(md, dict):
        raise ValueError(f"metadata must be a dict, got {type(md).__name__}")
    if isinstance(img_w, bool) or not isinstance(img_w, int) or img_w <= 0:
        raise ValueError(f"img_w must be a positive int, got {img_w!r}")
    if "depth" not in md:
        raise ValueError("metadata has no 'depth' field to canonicalize")
    _check_depth_cloud(md["depth"])
    if int(md["depth"].shape[2]) != img_w:
        raise ValueError(f"depth width {int(md['depth'].shape[2])} != img_w {img_w}")
    counts = set()
    for key, shape in _CANON_POSE_SHAPES.items():
        if key in md:
            _check_pose(key, md[key], shape)
            if shape[0] is None:
                counts.add(int(md[key].shape[0]))
    if 0 in counts or len(counts) > 1:
        raise ValueError(f"context_poses / context_poses_vit need the same N >= 1 rows, "
                         f"got {sorted(counts)}")


def canonicalize_scene_metadata(
    md: Dict[str, object], img_w: int, pose_keys: Tuple[str, ...] = POSE_KEYS
) -> Tuple[Dict[str, object], int, str]:
    """
    Deepest-pixel yaw canonicalization of ONE sample's metadata (plan §2.1 Step D).

    ``j*`` comes from :func:`canonical_column` on the :func:`recover_depth_proxy` of
    ``md['depth']`` (tie keys ``md['source']``, ``md['context_poses']``); the reviewed
    yaw operator is then applied ONCE, ``rotate_scene_metadata(md, dj * 2pi / img_w,
    img_w, pose_keys)`` with ``dj = canonical_shift(j*, img_w)``: column ``j*``
    becomes column 0 and the depth vectors and pose fields rotate about z by the
    same angle. ``context_audio``, ``scene``, ``padding_mask`` and every other field
    pass through as the same objects; the input dict and its tensors are not mutated.

    Fail-closed: ``ValueError`` on a contract violation (``depth`` a finite floating
    ``[3, H, W]`` with even ``H`` and ``W == img_w``; when present ``source [3]``,
    ``source_vit [1, 3]``, ``context_poses`` / ``context_poses_vit [N, 3]`` with one
    ``N >= 1``, finite floating); :class:`StabilizedTieError` propagates (never
    caught); ``RuntimeError`` when the post-condition -- the selector re-run on the
    output gives ``j* == 0`` and the same tie class -- fails.

    Returns ``(md_canon, dj, tie_class)``.
    """
    _check_canon_inputs(md, img_w)
    j_star, tie_class = canonical_column(recover_depth_proxy(md["depth"]), md.get("source"),
                                         md.get("context_poses"))
    dj = canonical_shift(j_star, img_w)
    out = rotate_scene_metadata(md, offsets_to_radians([dj], img_w)[0], img_w, pose_keys)
    j_check, class_check = canonical_column(recover_depth_proxy(out["depth"]),
                                            out.get("source"), out.get("context_poses"))
    if j_check != 0 or class_check != tie_class:
        raise RuntimeError(
            f"canonicalization post-condition failed: shifting j*={j_star} ({tie_class}) "
            f"by dj={dj} left the selector at j*={j_check} ({class_check}), expected 0")
    return out, dj, tie_class
