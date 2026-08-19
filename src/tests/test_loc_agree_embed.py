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

from src.localization.agree_embed import (
    MAX_LEN,
    TOWER_LEN,
    embed_rirs,
    load_agree_audio,
    preprocess_for_scoring,
    sha256_file,
)


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


@pytest.mark.parametrize("length", [1, 100, 7999, 8000, 8001, 10239, 10240, 10241, 20000])
def test_preprocess_output_length_is_invariant_and_pads_only(length):
    """O18: the release route pads but never crops (Retrieval.py:46-47 has no
    else-branch), so the tower length must be reached by padding alone -- which
    holds because the max_len slice runs first. Whatever T is, the output is
    exactly TOWER_LEN and the content window is the first min(T, MAX_LEN)
    samples, with zeros after it."""
    keep = min(length, MAX_LEN)
    wav = torch.full((1, 1, length), 0.75)
    out = preprocess_for_scoring(wav)
    assert tuple(out.shape) == (1, 1, TOWER_LEN)
    assert torch.all(out[..., :keep] == 0.75)
    assert torch.all(out[..., keep:] == 0.0)


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


# --------------------------------------------------------------------------- #
# load_agree_audio -- wrapper over metric_callback.loading_AGREE_model
#
# The audio tower loads ``audio_cfg['pretrained']`` = "weights/FLAC/VAE.ckpt"
# through a CWD-RELATIVE torch.load at construction time (AGREE/AGREE/
# audio_model.py:185-192, config AGREE/AGREE/model_configs/dinoV3.json), so a
# wrong working directory must be refused up front, not diagnosed 30 s later.
# --------------------------------------------------------------------------- #
import hashlib   # noqa: E402
import os        # noqa: E402


def test_sha256_file_matches_hashlib_and_is_chunk_invariant(tmp_path):
    payload = os.urandom(3 * 1024 * 1024 + 17)         # spans several read chunks
    path = tmp_path / "ckpt.pt"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_load_agree_audio_refuses_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_agree_audio(tmp_path / "not_here.pt", "cpu")


def test_load_agree_audio_refuses_a_wrong_working_directory(tmp_path, monkeypatch):
    """The guard must fire BEFORE the model is constructed -- asserted with a spy
    that fails if loading_AGREE_model is reached."""
    import src.metrics.metric_callback as mc

    def _never(*args, **kwargs):
        raise AssertionError("loading_AGREE_model must not run when the CWD is wrong")

    monkeypatch.setattr(mc, "loading_AGREE_model", _never)
    ckpt = tmp_path / "AGREE_AR.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    monkeypatch.chdir(tmp_path)                        # no weights/FLAC/VAE.ckpt here
    with pytest.raises(FileNotFoundError, match="working directory"):
        load_agree_audio(ckpt, "cpu")


def test_load_agree_audio_refuses_an_unknown_model_config(tmp_path):
    ckpt = tmp_path / "AGREE_AR.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    with pytest.raises(ValueError):
        load_agree_audio(ckpt, "cpu", config_name="no_such_config")


# --------------------------------------------------------------------------- #
# INTEGRATION -- the real AGREE scorer (CPU, small batches, ~6 s total).
# Skipped unless the checkpoints and the gated DINOv3 HF cache are present; the
# weight paths are relative, so a run from outside the repo root also skips.
# --------------------------------------------------------------------------- #
_AGREE_CKPT = "weights/AGREE/AGREE_AR.pt"
_VAE_CKPT = "weights/FLAC/VAE.ckpt"


def _dinov3_cache_present():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    return os.path.isdir(os.path.join(hub, "models--facebook--dinov3-vits16-pretrain-lvd1689m"))


_HAVE_ASSETS = (os.path.isfile(_AGREE_CKPT) and os.path.isfile(_VAE_CKPT)
                and _dinov3_cache_present())
integration = pytest.mark.skipif(
    not _HAVE_ASSETS,
    reason="AGREE/VAE checkpoints or the gated DINOv3 HF cache absent (or CWD is not the repo root)")


@pytest.fixture(scope="module")
def real_agree():
    return load_agree_audio(_AGREE_CKPT, "cpu")


@pytest.fixture(scope="module")
def real_wavs():
    g = torch.Generator().manual_seed(2618)
    return torch.randn(2, 1, 9000, generator=g) * 0.3


@integration
def test_integration_load_agree_audio_is_frozen_and_eval(real_agree):
    assert real_agree.model.training is False
    assert not any(p.requires_grad for p in real_agree.model.parameters())
    assert real_agree.device == "cpu" and real_agree.ckpt_path == _AGREE_CKPT
    assert len(real_agree.ckpt_sha256) == 64
    assert real_agree.ckpt_sha256 == sha256_file(_AGREE_CKPT)


@integration
def test_integration_mean_readout_is_deterministic_and_unit_norm(real_agree, real_wavs):
    first = embed_rirs(real_agree.model, real_wavs, "cpu")
    second = embed_rirs(real_agree.model, real_wavs, "cpu")
    assert tuple(first.shape) == (2, 512) and first.dtype == torch.float32
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.equal(first, second)


@integration
def test_integration_sampled_readout_is_stochastic(real_agree, real_wavs):
    """Witnesses on the real model why the sampled path cannot be the scorer."""
    first = embed_rirs(real_agree.model, real_wavs, "cpu", readout="sample")
    second = embed_rirs(real_agree.model, real_wavs, "cpu", readout="sample")
    assert not torch.equal(first, second)
    assert (first - second).abs().max() > 1e-6


@integration
def test_integration_mean_readout_leaves_the_global_rng_untouched(real_agree, real_wavs):
    before = torch.random.get_rng_state()
    embed_rirs(real_agree.model, real_wavs, "cpu", readout="mean")
    assert torch.equal(torch.random.get_rng_state(), before)

    before = torch.random.get_rng_state()
    embed_rirs(real_agree.model, real_wavs, "cpu", readout="sample")
    assert not torch.equal(torch.random.get_rng_state(), before)
