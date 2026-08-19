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

from src.localization.agree_embed import MAX_LEN, TOWER_LEN, embed_rirs, preprocess_for_scoring


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


# --------------------------------------------------------------------------- #
# embed_rirs -- the registered deterministic VAE-mean readout (plan §2.4)
#
# The stub mirrors ``OobleckEncoder`` (AGREE/AGREE/audio_model.py:199-204) and
# uses the REAL ``VAEBottleneck``, so the sampled path under test is the stock
# stochastic one, not a re-implementation. ``scale_value`` controls the
# bottleneck's stdev = softplus(scale) + 1e-4: -30 gives ~1e-4 (sampling is
# effectively off), +5 gives ~5 (sampling dominates).
# --------------------------------------------------------------------------- #
from AGREE.AGREE.audio_model import VAEBottleneck   # noqa: E402


class _StubLayers(torch.nn.Module):
    def __init__(self, latent, length, scale_value):
        super().__init__()
        self.latent, self.length, self.scale_value = latent, length, scale_value

    def forward(self, x):
        b = x.shape[0]
        mean = x.reshape(b, -1)[:, : self.latent * self.length].reshape(b, self.latent, self.length)
        return torch.cat([mean, torch.full_like(mean, self.scale_value)], dim=1)


class _StubAudio(torch.nn.Module):
    def __init__(self, latent=4, length=5, embed=6, scale_value=-30.0, seed=0):
        super().__init__()
        self.layers = _StubLayers(latent, length, scale_value)
        self.bottleneck = VAEBottleneck()
        self.project = torch.nn.Linear(latent * length, embed)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.project.weight.copy_(torch.randn(embed, latent * length, generator=g))
            self.project.bias.copy_(torch.randn(embed, generator=g))

    def forward(self, x):                      # mirrors OobleckEncoder.forward
        x = self.layers(x)
        latents = self.bottleneck.encode(x, return_info=False)
        latents = latents.view(x.size(0), -1)
        return self.project(latents)


class _StubModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.audio = _StubAudio(**kwargs)
        self.eval()                            # the scorer is always frozen + eval


def _wavs(b=4, t=9000, seed=18):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 1, t, generator=g) * 0.4


def test_embed_rirs_shape_dtype_and_normalization():
    out = embed_rirs(_StubModel(embed=6), _wavs(), "cpu")
    assert tuple(out.shape) == (4, 6)
    assert out.dtype == torch.float32 and out.device.type == "cpu"
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert out.requires_grad is False
    assert out.is_inference() is False          # usable as an ordinary tensor downstream


def test_embed_rirs_mean_readout_is_bitwise_deterministic():
    model, wavs = _StubModel(), _wavs()
    assert torch.equal(embed_rirs(model, wavs, "cpu"), embed_rirs(model, wavs, "cpu"))


def test_embed_rirs_mean_readout_does_not_touch_the_global_rng():
    """O2: the registered readout must draw no randomness at all -- otherwise it
    would both jitter scores and desynchronize the driver's noise bank."""
    model, wavs = _StubModel(), _wavs()
    before = torch.random.get_rng_state()
    embed_rirs(model, wavs, "cpu", readout="mean")
    assert torch.equal(torch.random.get_rng_state(), before)

    # contrast: the stock sampled path DOES consume the global stream
    before = torch.random.get_rng_state()
    embed_rirs(model, wavs, "cpu", readout="sample")
    assert not torch.equal(torch.random.get_rng_state(), before)


def test_embed_rirs_sample_matches_mean_when_the_bottleneck_stdev_vanishes():
    model, wavs = _StubModel(scale_value=-30.0), _wavs()      # stdev ~ 1e-4
    mean_emb = embed_rirs(model, wavs, "cpu", readout="mean")
    sampled = embed_rirs(model, wavs, "cpu", readout="sample")
    assert torch.allclose(mean_emb, sampled, atol=1e-3)


def test_embed_rirs_sample_diverges_from_mean_at_a_large_bottleneck_stdev():
    """Proves the mean path really bypasses sampling rather than coincidentally
    agreeing with it."""
    model, wavs = _StubModel(scale_value=5.0), _wavs()        # stdev ~ 5
    mean_emb = embed_rirs(model, wavs, "cpu", readout="mean")
    sampled = embed_rirs(model, wavs, "cpu", readout="sample")
    assert (mean_emb - sampled).abs().max() > 1e-2
    assert torch.equal(mean_emb, embed_rirs(model, wavs, "cpu", readout="mean"))


def test_embed_rirs_is_batch_size_invariant():
    model, wavs = _StubModel(), _wavs(b=8)
    batched = embed_rirs(model, wavs, "cpu")
    one_by_one = torch.cat([embed_rirs(model, wavs[i:i + 1], "cpu") for i in range(8)])
    assert torch.allclose(batched, one_by_one, atol=1e-6)


def test_embed_rirs_rejects_bad_readout_and_nonfinite_input():
    model = _StubModel()
    with pytest.raises(ValueError):
        embed_rirs(model, _wavs(), "cpu", readout="median")
    bad = _wavs()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        embed_rirs(model, bad, "cpu")
    with pytest.raises(ValueError):
        embed_rirs(model, torch.zeros(2, 100), "cpu")
