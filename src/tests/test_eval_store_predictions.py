"""Tests for ``--store_predictions`` storing the *exactly-as-scored* tensor
(exp_14 round C; announcement 08 ``worklog/worklog_yixun/announcement/08_save_pred_waveforms.md``).

At pin ``7bbd8aa`` ``eval_FLAC``'s per-batch loop appended ``fakes.cpu()`` to
``decoded_samples`` **before** the clamp/pad block, so the stored bundle was the
raw decoder output, not the tensor the metric callback scored. Round C extracts
that block into :func:`eval_FLAC.clamp_and_pad` and moves the append after it.

RED first, one cycle per commit:
  C1 ``eval_FLAC.clamp_and_pad`` does not exist (AttributeError).
  C2 the stored tensor is the pre-clamp/pad one (not bitwise the scored input).
  C3 the bundle meta lacks ``ckpt_path`` / ``n_items`` / ``eval_name`` / ``steps``
     / ``cfg_scale`` / ``stored_after_clamp_pad`` / ``artifact_contract``.

The artifact contract pinned here: the stored tensor is the **clamped/padded
callback input** -- the metric callback's own ``float()`` cast and its
``max_len`` (8000-sample) crop happen *inside* scoring and are deliberately not
part of the artifact.
"""
import torch

import pytest

import eval_FLAC  # noqa: E402  (heavy but side-effect-free at import)


# --------------------------------------------------------------------------- #
# Pinned reference: the clamp/pad block as it stood at 7bbd8aa
# (``git show 7bbd8aa:eval_FLAC.py``, lines 305-311), copied verbatim into a
# function. Only whitespace is normalized (the pin has a trailing space after
# ``clamp(-1.0, 1.0)``); no token differs. DO NOT EDIT -- this is the
# equivalence oracle for the extraction.
# --------------------------------------------------------------------------- #
def _pinned_clamp_and_pad(fakes, reals):
    # Clamp and pad if necessary
    fakes = fakes.clamp(-1.0, 1.0)
    if fakes.shape != reals.shape:
        if fakes.shape[-1] < reals.shape[-1]:
            fakes = torch.nn.functional.pad(fakes, (0, reals.shape[-1] - fakes.shape[-1]))
        else:
            reals = torch.nn.functional.pad(reals, (0, fakes.shape[-1] - reals.shape[-1]))
    return fakes, reals


# --------------------------------------------------------------------------- #
# C1: clamp_and_pad semantics
# --------------------------------------------------------------------------- #
def test_clamp_and_pad_clamps_out_of_range_values():
    """Values outside [-1, 1] are clamped; in-range values pass through."""
    fakes = torch.tensor([[[-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 7.5]]])
    reals = torch.zeros(1, 1, 7)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert torch.equal(
        out_f, torch.tensor([[[-1.0, -1.0, -0.25, 0.0, 0.5, 1.0, 1.0]]])
    )
    assert out_f.min() >= -1.0 and out_f.max() <= 1.0
    assert torch.equal(out_r, reals)


def test_clamp_and_pad_short_fakes_are_zero_padded_to_reals():
    """fakes shorter than reals -> fakes zero-padded on the right; reals untouched."""
    fakes = torch.full((2, 1, 5), 2.0)
    reals = torch.arange(2 * 1 * 8, dtype=torch.float32).reshape(2, 1, 8)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.shape == reals.shape
    assert torch.equal(out_f[..., :5], torch.ones(2, 1, 5))   # clamped 2.0 -> 1.0
    assert torch.equal(out_f[..., 5:], torch.zeros(2, 1, 3))  # zero padding
    assert out_r is reals


def test_clamp_and_pad_long_fakes_pad_reals():
    """fakes longer than reals -> reals zero-padded instead; fakes keep their length."""
    fakes = torch.full((2, 1, 11), -4.0)
    reals = torch.ones(2, 1, 8)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.shape == (2, 1, 11)
    assert out_r.shape == (2, 1, 11)
    assert torch.equal(out_f, torch.full((2, 1, 11), -1.0))
    assert torch.equal(out_r[..., :8], torch.ones(2, 1, 8))
    assert torch.equal(out_r[..., 8:], torch.zeros(2, 1, 3))


def test_clamp_and_pad_equal_lengths_leaves_reals_identical():
    """Equal shapes -> no padding at all; reals is returned as the same object."""
    fakes = torch.rand(3, 2, 16) * 0.5
    reals = torch.rand(3, 2, 16)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_r is reals
    assert torch.equal(out_f, fakes)  # already in range: clamp is a no-op


def test_clamp_and_pad_does_not_mutate_its_inputs():
    """Pure: the caller's tensors are unchanged (clamp/pad both return copies)."""
    fakes = torch.tensor([[[-3.0, 0.5, 9.0]]])
    reals = torch.zeros(1, 1, 5)
    fakes_before, reals_before = fakes.clone(), reals.clone()
    eval_FLAC.clamp_and_pad(fakes, reals)
    assert torch.equal(fakes, fakes_before)
    assert torch.equal(reals, reals_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clamp_and_pad_preserves_dtype_and_device(dtype):
    fakes = (torch.randn(2, 1, 5) * 3).to(dtype)
    reals = torch.randn(2, 1, 8).to(dtype)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.dtype == dtype and out_r.dtype == dtype
    assert out_f.device == fakes.device and out_r.device == reals.device


@pytest.mark.parametrize(
    "shape_f,shape_r",
    [
        ((2, 1, 8), (2, 1, 8)),    # equal
        ((2, 1, 5), (2, 1, 8)),    # fakes shorter -> pad fakes
        ((2, 1, 11), (2, 1, 8)),   # fakes longer  -> pad reals
        ((1, 1, 1), (1, 1, 4)),    # degenerate single sample
        ((3, 2, 4), (3, 2, 4)),    # multi-channel, equal
    ],
)
def test_clamp_and_pad_matches_pinned_block(shape_f, shape_r):
    """Bitwise equivalence with the verbatim 7bbd8aa block on random tensors
    (scaled x3 so clamping actually fires)."""
    torch.manual_seed(0)
    fakes = torch.randn(*shape_f) * 3.0
    reals = torch.randn(*shape_r)
    got_f, got_r = eval_FLAC.clamp_and_pad(fakes.clone(), reals.clone())
    exp_f, exp_r = _pinned_clamp_and_pad(fakes.clone(), reals.clone())
    assert torch.equal(got_f, exp_f)
    assert torch.equal(got_r, exp_r)
