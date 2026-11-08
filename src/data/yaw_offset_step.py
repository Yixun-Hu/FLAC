"""exp_40 (yaw_aug_c16): the discrete-grid option of the exp_15 yaw augmentation.

Kept in its own module so ``src/data/yaw_rotation.py`` stays byte-identical to the exp_15
state (exp_25's audit gate pins that file's sha256). ``yaw_aug.offset_step`` absent => none
of this runs and the exp_15 draw is used unchanged.
"""
import torch


def check_yaw_offset_step(offset_step, img_w) -> int:
    """Validate a discrete grid step for the yaw augmentation (exp_40); return it.

    ``offset_step`` must be a plain ``int`` (``bool`` refused), ``1 <= offset_step <
    img_w`` and a divisor of ``img_w``. Only then does :func:`quantize_yaw_offsets` map
    a uniform column draw onto an EXACTLY uniform grid of ``img_w // offset_step``
    offsets (each grid point has ``offset_step`` preimages). ``offset_step == img_w``
    would leave the single offset 0 -- no augmentation at all -- and is refused rather
    than silently disabling the treatment. Nothing is coerced: ``32.0`` and ``"32"``
    are errors, like every other ``yaw_aug`` field.
    """
    if isinstance(img_w, bool) or not isinstance(img_w, int) or img_w <= 0:
        raise ValueError(f"img_w must be a positive int, got {img_w!r}")
    if isinstance(offset_step, bool) or not isinstance(offset_step, int):
        raise ValueError(f"offset_step must be an int, got {offset_step!r}")
    if not 1 <= offset_step < img_w:
        raise ValueError(f"offset_step must be in [1, {img_w}), got {offset_step}")
    if img_w % offset_step != 0:
        raise ValueError(
            f"offset_step {offset_step} does not divide img_w {img_w}: the snapped "
            "offsets would not be uniform over the grid")
    return offset_step


def quantize_yaw_offsets(offsets: torch.Tensor, img_w: int, offset_step: int) -> torch.Tensor:
    """Snap column offsets DOWN onto the ``offset_step`` grid: ``d -> s * (d // s)``.

    exp_40's discrete (C_{img_w/s}) yaw augmentation. Applied to the draw of
    :func:`draw_yaw_offsets`, so for a uniform ``d`` on ``{0..img_w-1}`` the result is
    exactly uniform on ``{0, s, ..., img_w - s}`` and, sample by sample, equal to the
    continuous draw rounded down onto the grid. Pure: consumes no randomness, returns
    a new ``int64`` tensor on the input's device, never modifies ``offsets``.
    """
    check_yaw_offset_step(offset_step, img_w)
    if not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.long or offsets.ndim != 1:
        raise ValueError(
            "offsets must be a 1-D int64 tensor (the output of draw_yaw_offsets), got "
            f"{type(offsets).__name__} "
            f"{getattr(offsets, 'dtype', None)} {list(getattr(offsets, 'shape', []))}")
    if offsets.numel() and (int(offsets.min()) < 0 or int(offsets.max()) >= img_w):
        raise ValueError(f"offsets must lie in [0, {img_w}), got [{int(offsets.min())}, "
                         f"{int(offsets.max())}]")
    return torch.div(offsets, offset_step, rounding_mode="floor") * offset_step
