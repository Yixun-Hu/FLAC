"""AGREE scorer wiring for exp_18 (loc_invert): loading, preprocessing, readout.

The registered scorer (plan §2.4) is the **deterministic VAE-mean readout**.
AGREE's audio tower ends in a sampling ``VAEBottleneck``
(``AGREE/AGREE/audio_model.py:275-300``: ``mean, scale = x.chunk(2, dim=1)`` then
``randn_like(mean) * stdev + mean``, with no eval guard), so the stock
``encode_audio`` is stochastic AND draws from the global RNG stream. This module
reproduces the tower's arithmetic with the mean substituted for the sample --
without editing any AGREE code -- and keeps the stochastic path available as a
labelled diagnostic.

Preprocessing is the established AR metric route, not a new convention: clamp to
[-1, 1] (``eval_FLAC.py:1313`` / ``src/training/diffusion.py:885``), take the
first ``max_len`` = 8000 samples (``src/metrics/metric_callback.py:113-114``,
``:287``), then pad to the tower's 10,240
(``src/metrics/modules/Retrieval.py:46-47``).
"""
import torch

#: AcousticRooms ``max_len`` used by the metric callback for retrieval inputs.
MAX_LEN = 8000
#: audio length the AGREE tower is configured for (``audio_cfg.audio_length``).
TOWER_LEN = 10240


def _require_finite(tensor, what):
    """Fail closed on NaN / +-Inf (a threshold guard alone would let NaN pass)."""
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{what} must be finite (no NaN or Inf)")
    return tensor


def preprocess_for_scoring(wavs):
    """``[B, 1, T]`` waveforms -> ``[B, 1, 10240]`` float32 scorer input.

    Clamp to [-1, 1], truncate to the first ``MAX_LEN`` samples, then pad to
    ``TOWER_LEN`` -- the same three steps, in the same order, that the release
    metric route applies before ``encode_audio``. Truncation happens at
    ``MAX_LEN``, so nothing past sample 8000 can influence a score.
    """
    if not isinstance(wavs, torch.Tensor):
        raise ValueError(f"wavs must be a torch.Tensor, got {type(wavs).__name__}")
    if wavs.ndim != 3 or wavs.shape[1] != 1 or wavs.shape[-1] == 0:
        raise ValueError(f"wavs must be a non-empty [B, 1, T] tensor, got shape {tuple(wavs.shape)}")
    _require_finite(wavs, "wavs")

    out = wavs.float().clamp(-1.0, 1.0)[..., :MAX_LEN]
    if out.shape[-1] < TOWER_LEN:
        out = torch.nn.functional.pad(out, (0, TOWER_LEN - out.shape[-1]))
    return out


def embed_rirs(model, wavs, device, readout="mean"):
    """Embed ``[B, 1, T]`` RIRs with the AGREE audio tower -> ``[B, D]`` float32 (CPU).

    ``readout='mean'`` is the REGISTERED scorer: it runs the tower's own layers
    and projection but substitutes the bottleneck's *mean* for its sample --
    ``x = audio.layers(h)``, ``mean, scale = x.chunk(2, dim=1)``,
    ``audio.project(mean.reshape(B, -1))`` -- reproducing
    ``OobleckEncoder.forward`` (AGREE/AGREE/audio_model.py:199-204) with
    ``vae_sample``'s ``randn_like`` removed. It is deterministic and consumes no
    randomness, so scores cannot jitter and the driver's noise bank cannot be
    desynchronized.

    ``readout='sample'`` is the stock stochastic path (``model.audio(x)``, which
    is exactly what ``encode_audio(x, normalize=True)`` runs -- AGREE/AGREE/
    model.py:288-290) and is a labelled diagnostic only.

    Embeddings are L2-normalized, as the retrieval route normalizes them.
    """
    if readout not in ("mean", "sample"):
        raise ValueError(f"unknown readout {readout!r} (expected 'mean' or 'sample')")
    if getattr(model, "training", False):
        raise ValueError("the scorer must be in eval mode; call load_agree_audio (plan §2.4)")

    x = preprocess_for_scoring(wavs).to(device)
    with torch.inference_mode():
        if readout == "mean":
            audio = model.audio
            hidden = audio.layers(x)
            mean, _scale = hidden.chunk(2, dim=1)
            features = audio.project(mean.reshape(hidden.shape[0], -1))
        else:
            features = model.audio(x)
        embeddings = torch.nn.functional.normalize(features.float(), dim=-1).cpu()
    # leave inference mode behind: the driver stores, stacks and scores these.
    return embeddings.clone()
