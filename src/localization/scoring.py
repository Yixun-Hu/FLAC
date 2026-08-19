"""Pure scoring / metric / baseline functions for exp_18 (loc_invert).

No I/O, no model, no global RNG: everything here is a deterministic function of
its arguments, so the driver's numbers can be re-derived offline from the logged
similarities.
"""
import math

import torch

#: tolerance on ``|‖v‖ - 1|`` for the cosine-similarity norm guard.
NORM_TOL = 1e-4


def cosine_sims(obs, gen):
    """Cosine similarities between one observation and ``M x K`` generations.

    ``obs`` is ``[D]``, ``gen`` is ``[M, K, D]``; both must already be
    L2-normalized (the embedder's contract) -- a norm deviating from 1 by more
    than ``NORM_TOL`` raises ``ValueError`` rather than silently returning a
    quantity that is not a cosine.
    """
    if obs.ndim != 1:
        raise ValueError(f"obs must be [D], got shape {tuple(obs.shape)}")
    if gen.ndim != 3:
        raise ValueError(f"gen must be [M, K, D], got shape {tuple(gen.shape)}")
    if gen.shape[-1] != obs.shape[0]:
        raise ValueError(f"embedding dim mismatch: obs {obs.shape[0]} vs gen {gen.shape[-1]}")

    obs_dev = (obs.norm(dim=-1) - 1.0).abs().max().item()
    if obs_dev > NORM_TOL:
        raise ValueError(f"obs is not L2-normalized (|‖v‖-1| = {obs_dev:g} > {NORM_TOL:g})")
    gen_dev = (gen.norm(dim=-1) - 1.0).abs().max().item()
    if gen_dev > NORM_TOL:
        raise ValueError(f"gen is not L2-normalized (max |‖v‖-1| = {gen_dev:g} > {NORM_TOL:g})")

    return gen @ obs


def aggregate(sims, method="lme", tau=None):
    """Aggregate ``sims`` ``[M, K]`` over the K samples into per-candidate scores ``[M]``.

    ``lme`` is the registered method: ``tau * (logsumexp(s / tau) - log K)``,
    i.e. a log-mean-exp -- it tends to ``max`` as ``tau -> 0+`` and to ``mean``
    as ``tau`` grows, and goes through ``torch.logsumexp`` so the ``s / tau``
    scaling (50x at the registered tau = 0.02) cannot overflow.
    """
    if sims.ndim != 2:
        raise ValueError(f"sims must be [M, K], got shape {tuple(sims.shape)}")
    if method == "mean":
        return sims.mean(dim=-1)
    if method == "max":
        return sims.max(dim=-1).values
    if method == "lme":
        if tau is None:
            raise ValueError("method 'lme' requires tau")
        tau = float(tau)
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0 for method 'lme', got {tau}")
        return tau * (torch.logsumexp(sims / tau, dim=-1) - math.log(sims.shape[-1]))
    raise ValueError(f"unknown aggregation method {method!r} (expected 'lme', 'mean' or 'max')")


def predict_index(scores):
    """Index of the highest score, ties broken by lowest index (registered)."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty [M] tensor, got shape {tuple(scores.shape)}")
    winners = torch.nonzero(scores == scores.max(), as_tuple=False)
    return int(winners[0].item())


def softmax_map(scores, T):
    """Temperature-``T`` softmax over candidates: a [M] distribution summing to 1."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty [M] tensor, got shape {tuple(scores.shape)}")
    T = float(T)
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")
    return torch.softmax(scores / T, dim=-1)


def _point(value, what):
    xyz = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if xyz.numel() != 3:
        raise ValueError(f"{what} must be 3 coordinates, got {tuple(torch.as_tensor(value).shape)}")
    return xyz


def localization_error(pred_xyz, gt_xyz):
    """Euclidean distance (metres) between a predicted and the true position."""
    return float(torch.linalg.norm(_point(pred_xyz, "pred_xyz") - _point(gt_xyz, "gt_xyz")))


def success_within(error, radius):
    """``True`` iff ``error <= radius`` -- the boundary counts as a success."""
    radius = float(radius)
    if radius < 0.0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    return bool(float(error) <= radius)
