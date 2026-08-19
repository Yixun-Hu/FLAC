"""Tests for ``src.localization.agree_embed`` (exp_18 loc_invert, round 2).

Written test-first (announcement 02). Contracts: ``loc_invert_impl_contracts.md``
§4.4 + the Rev 3 deltas, serving plan §2.4 -- the *deterministic VAE-mean*
readout is the registered scorer (AGREE's stock ``encode_audio`` samples the VAE
bottleneck and consumes the global RNG), and the preprocessing is the established
AR metric route: clamp to [-1, 1], first ``max_len`` = 8000 samples, then the
tower's 10,240 padding.
"""
import pytest
import torch

from src.localization.agree_embed import MAX_LEN, TOWER_LEN, preprocess_for_scoring


# --------------------------------------------------------------------------- #
# preprocess_for_scoring
# --------------------------------------------------------------------------- #
def test_preprocess_shape_and_dtype_contract():
    out = preprocess_for_scoring(torch.zeros(3, 1, 5000))
    assert tuple(out.shape) == (3, 1, TOWER_LEN)
    assert out.dtype == torch.float32
    assert MAX_LEN == 8000 and TOWER_LEN == 10240


def test_preprocess_pads_short_input_without_touching_content():
    wav = torch.rand(2, 1, 5000) * 0.5
    out = preprocess_for_scoring(wav)
    assert torch.equal(out[..., :5000], wav.float())
    assert torch.all(out[..., 5000:] == 0.0)


def test_preprocess_truncates_at_max_len_before_padding():
    """Content past sample 8000 never reaches the scorer: the established route
    slices at ``max_len`` first and only then pads up to the tower length."""
    wav = torch.ones(1, 1, 9000) * 0.25
    out = preprocess_for_scoring(wav)
    assert torch.all(out[..., :MAX_LEN] == 0.25)
    assert torch.all(out[..., MAX_LEN:] == 0.0)


def test_preprocess_truncates_input_longer_than_the_tower_length():
    wav = torch.rand(1, 1, 12000)
    out = preprocess_for_scoring(wav)
    assert tuple(out.shape) == (1, 1, TOWER_LEN)
    assert torch.equal(out[..., :MAX_LEN], wav[..., :MAX_LEN].float())
    assert torch.all(out[..., MAX_LEN:] == 0.0)


def test_preprocess_clamps_to_unit_range():
    wav = torch.tensor([[[-3.0, -1.0, 0.25, 1.0, 7.0]]])
    out = preprocess_for_scoring(wav)
    assert torch.equal(out[0, 0, :5], torch.tensor([-1.0, -1.0, 0.25, 1.0, 1.0]))


def test_preprocess_casts_to_float32_and_keeps_batch_rows_independent():
    wav = torch.stack([torch.full((1, 100), 0.5, dtype=torch.float64),
                       torch.full((1, 100), -2.0, dtype=torch.float64)])
    out = preprocess_for_scoring(wav)
    assert out.dtype == torch.float32
    assert torch.all(out[0, 0, :100] == 0.5) and torch.all(out[1, 0, :100] == -1.0)


def test_preprocess_matches_the_real_metric_route_composition():
    """C6: our single call must equal the composition of the ACTUAL code paths --
    ``fakes.clamp(-1.0, 1.0)`` (eval_FLAC.py:1313 / src/training/diffusion.py:885),
    then the ``pred[index, ..., :self.max_len]`` slice with ``max_len`` = 8000 for
    AcousticRooms (src/metrics/metric_callback.py:113-114 and :287), then the
    padding branch of ``Retrieval.compute_audio_features``:
    ``if h.shape[-1] < 10240: h = torch.nn.functional.pad(h, (0, 10240 - h.shape[-1]))``
    (src/metrics/modules/Retrieval.py:46-47). The pad expression is replicated
    here rather than instantiating Retrieval, which would need a real AGREE model.
    """
    torch.manual_seed(18)
    for length in (4000, 8000, 9600, 12000):
        wav = torch.randn(3, 1, length) * 1.5          # deliberately outside [-1, 1]

        route = wav.clamp(-1.0, 1.0)                                     # eval_FLAC.py:1313
        route = torch.stack([route[i, ..., :8000] for i in range(route.shape[0])])  # :287
        if route.shape[-1] < 10240:                                      # Retrieval.py:46-47
            route = torch.nn.functional.pad(route, (0, 10240 - route.shape[-1]))

        assert torch.equal(preprocess_for_scoring(wav), route.float())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_preprocess_rejects_nonfinite_input(bad):
    wav = torch.zeros(1, 1, 100)
    wav[0, 0, 7] = bad
    with pytest.raises(ValueError):
        preprocess_for_scoring(wav)


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 100),        # missing channel axis
    torch.zeros(2, 2, 100),     # not mono
    torch.zeros(1, 1, 0),       # empty
    torch.zeros(1, 1, 10, 2),   # too many axes
])
def test_preprocess_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        preprocess_for_scoring(bad)
