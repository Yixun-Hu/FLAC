"""exp_22 I1 -- the frozen mesh-grid localization engine (inherited plan §1.4/§1.5).

One query is an observed RIR ``h_obs``; the engine regenerates one RIR per
mesh-valid grid candidate under the frozen Vanilla FLAC checkpoint, scores each
against ``h_obs`` in AGREE's audio embedding space, and predicts the argmax
candidate. Everything that decides a number -- the noise a candidate is drawn
from, the context tensors it is conditioned on, the candidate set it competes in
-- is bound to a registered artifact (the D1 context manifest, the G1 candidate
manifests, the checkpoint, the scorer) and refused when it does not match.

Two protocol points are RECORDED DEVIATIONS from the inherited plan text and are
stamped into every run's provenance rather than being silently taken:

* ``SCORER_READOUT`` -- §1.4 pins ``encode_audio(..., normalize=True)``. That
  path samples AGREE's VAE bottleneck, which exp_18 measured as ~7e-5 of cosine
  noise per call and which consumes the global RNG stream. The registered
  scorer here is exp_18/exp_20's deterministic mean readout, which is the same
  arithmetic with the bottleneck's mean substituted for its sample.
* ``NOISE_KEY_POLICY`` -- §1.1 says candidates of a query share their seeds
  (common random numbers); the dispatched contract keys the draw by
  ``(seed, query_id, candidate_index, k)``. The dispatched key is the default,
  and the shared-across-candidates alternative is implemented and selectable so
  a ruling either way costs no code round.
"""
import hashlib
import json
import math

import numpy as np
import torch

from src.localization.reaggregate import encode_sims
from src.localization.scoring import aggregate
from src.localization import scoring as _scoring

#: the three reported nested prefixes; K=1 is sample 0, K=4 samples 0-3 (§1.4).
K_PREFIXES = (1, 4, 8)
#: length of the one generated sequence all three prefixes are read from.
NUM_SAMPLES = 8
#: the fixed score temperature (§1.4; Yixun 2026-08-21).
TAU = 0.1
#: the registered noise seed.
SEED = 42
#: sampler settings mirroring the release evaluation path (§1.4).
STEPS = 1
CFG_SCALE = 1.0

#: conditioning branches. The context branch is computed once per QUERY and the
#: source branch once per (receiver, candidate) -- the split of §1.5.
CONTEXT_COND_IDS = ("context_poses_vit", "context_poses", "context_audio")
SOURCE_COND_IDS = ("source", "source_vit")

#: the registered readout and the measurement that justifies the deviation.
SCORER_READOUT = "mean"
SCORER_READOUT_DEVIATION = (
    "inherited plan §1.4 names encode_audio(..., normalize=True); this run uses the "
    "deterministic VAE-mean readout of exp_18/exp_20 (src/localization/agree_embed.py), "
    "because the sampled path draws from AGREE's VAE bottleneck -- measured by exp_18 at "
    "~7e-5 cosine noise per call -- and consumes the global RNG stream")

#: the scorer's leakage caveat (Yixun 2026-08-24 decision 2c), stamped everywhere.
AGREE_LEAKAGE_CAVEAT = (
    "AGREE_fullAR saw the full dataset including the unseen rooms; acceptable here because "
    "the scorer is frozen, identical across arms and candidates, and pinned by the approved "
    "exp_09 protocol -- but absolute levels are NOT leak-free and must never be compared "
    "against AGREE_AR-scored exp_18/exp_20 rows without this label")

NOISE_KEY_POLICIES = ("per_candidate", "shared_across_candidates")
#: the dispatched default (see the module docstring's deviation note).
NOISE_KEY_POLICY = "per_candidate"


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def noise_key(seed, query_id, candidate_index, k):
    """Deterministic generator seed for ``(query, candidate, draw)``.

    sha256 over a canonical JSON payload -- never Python's salted ``hash()`` --
    so a resumed pass, a second machine and an offline replay all draw the same
    latent for the same candidate.
    """
    payload = json.dumps(["loc_meshgrid_noise_key", int(seed), str(query_id),
                          int(candidate_index), int(k)], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def noise_key_for(policy, seed, query_id, candidate_index, k):
    """The draw's generator seed under the selected keying policy."""
    if policy == "per_candidate":
        return noise_key(seed, query_id, candidate_index, k)
    if policy == "shared_across_candidates":
        return _scoring.noise_key(seed, query_id, k)
    raise ValueError(f"unknown noise policy {policy!r} (expected one of {list(NOISE_KEY_POLICIES)})")


def noise_block(seed, query_id, candidate_indices, num_samples, latent_shape,
                policy=NOISE_KEY_POLICY, device="cpu"):
    """The candidate-major latent noise for one query slice: ``[M * K, C, T]``.

    Row ``m * K + k`` is candidate ``candidate_indices[m]`` with draw ``k``, each
    from its own CPU ``torch.Generator`` -- never the global stream, which the
    loader and conditioners also advance. Because every row is keyed
    independently, generating a slice of candidates yields exactly the rows the
    whole block would have carried, which is what makes the pass batch-size and
    chunking invariant.
    """
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples (K) must be >= 1, got {num_samples}")
    shape = tuple(int(s) for s in latent_shape)
    if len(shape) != 2 or any(s < 1 for s in shape):
        raise ValueError(f"latent_shape must be [channels, samples] with positive dims, got {shape}")
    indices = [int(i) for i in candidate_indices]
    if not indices:
        raise ValueError("candidate_indices must be a non-empty sequence")
    if len(set(indices)) != len(indices):
        raise ValueError("candidate_indices must be unique: a repeated candidate would be "
                         "generated twice under the same key and scored twice")

    draws = []
    for index in indices:
        for k in range(num_samples):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(noise_key_for(policy, seed, query_id, index, k))
            draws.append(torch.randn(shape, generator=generator))
    return torch.stack(draws).to(device)


# --------------------------------------------------------------------------- #
# scoring: the nested prefixes and the registered prediction rule
# --------------------------------------------------------------------------- #
def nested_scores(sims, tau=TAU, prefixes=K_PREFIXES):
    """``{K: {scores, mean_scores}}`` over nested prefixes of one ``[M, K]`` block.

    All three settings read the SAME generated sequence (§1.4), so they cost one
    K=8 execution and cannot disagree about a sample.
    """
    if not isinstance(sims, torch.Tensor) or sims.ndim != 2:
        raise ValueError(f"sims must be an [M, K] tensor, got {type(sims).__name__}")
    if not bool(torch.isfinite(sims).all()):
        raise ValueError("sims must be finite (no NaN or Inf)")
    prefixes = tuple(int(k) for k in prefixes)
    biggest = max(prefixes)
    if sims.shape[-1] < biggest:
        raise ValueError(f"sims carries {sims.shape[-1]} samples but the registered prefixes "
                         f"{list(prefixes)} need {biggest}")
    out = {}
    for k in prefixes:
        head = sims[:, :k].contiguous()
        out[k] = {"scores": aggregate(head, method="lme", tau=tau),
                  "mean_scores": head.mean(dim=-1)}
    return out


def argmax_by_global_index(scores, candidate_indices):
    """Highest score, ties broken by the SMALLEST global candidate index (§1.4).

    Returns the ROW of the winner. The manifest's indices are ascending, so this
    is normally the first maximal row -- but the rule is written on the global
    index so that a re-ordered candidate slice cannot change the prediction.
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores)
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty [M] tensor, got shape {tuple(scores.shape)}")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite (no NaN or Inf)")
    indices = [int(i) for i in candidate_indices]
    if len(indices) != scores.numel():
        raise ValueError(f"candidate_indices has {len(indices)} entries for {scores.numel()} "
                         "scores")
    best = float(scores.max())
    winners = [row for row in range(len(indices)) if float(scores[row]) == best]
    return min(winners, key=lambda row: indices[row])


def score_query(sims, candidate_indices, coordinates, tau=TAU, prefixes=K_PREFIXES):
    """One query's full score block: every prefix's scores, prediction and mean.

    ``scores_hex`` is the exact float32 hex codec exp_18 published its
    similarities in, so an offline re-aggregation reproduces the online numbers
    bit for bit.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    indices = [int(i) for i in candidate_indices]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must be [M, 3], got shape {coordinates.shape}")
    if coordinates.shape[0] != len(indices):
        raise ValueError(f"{coordinates.shape[0]} coordinates for {len(indices)} candidates")

    by_k = {}
    for k, block in nested_scores(sims, tau=tau, prefixes=prefixes).items():
        row = argmax_by_global_index(block["scores"], indices)
        mean_row = argmax_by_global_index(block["mean_scores"], indices)
        by_k[k] = {
            "prediction_row": row,
            "prediction_index": indices[row],
            "prediction_xyz": coordinates[row].tolist(),
            "scores_hex": encode_sims(block["scores"].reshape(1, -1))[0],
            "mean_prediction_row": mean_row,
            "mean_prediction_index": indices[mean_row],
            "mean_prediction_xyz": coordinates[mean_row].tolist(),
            "mean_scores_hex": encode_sims(block["mean_scores"].reshape(1, -1))[0],
        }
    return {"by_k": by_k, "n_candidates": len(indices),
            "num_samples": int(sims.shape[-1]), "tau": float(tau)}
