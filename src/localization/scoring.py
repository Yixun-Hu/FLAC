"""Pure scoring / metric / baseline functions for exp_18 (loc_invert).

No I/O, no model, no global RNG: everything here is a deterministic function of
its arguments, so the driver's numbers can be re-derived offline from the logged
similarities.
"""
import hashlib
import json
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


def _points(value, what):
    xyz = torch.as_tensor(value, dtype=torch.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"{what} must be a non-empty [N, 3] array, got shape {tuple(xyz.shape)}")
    return xyz


def _gt_index(cand, gt):
    """Index of the GT row in the candidate set (GT is in C by construction)."""
    hits = torch.nonzero((cand == gt).all(dim=-1), as_tuple=False).reshape(-1)
    if hits.numel() != 1:
        raise ValueError(
            f"expected exactly one candidate equal to gt_xyz {gt.tolist()}, found {hits.numel()}")
    return int(hits[0].item())


def _expectations(distances, radii):
    """Exact expectations of a uniform draw over the eligible candidates."""
    n = int(distances.numel())
    for r in radii:
        if float(r) < 0.0:
            raise ValueError(f"radius must be >= 0, got {r}")
    return {
        "n_eligible": n,
        "distances": [float(d) for d in distances],
        "mean_error": float(distances.mean()),
        "success": {float(r): float((distances <= float(r)).double().mean()) for r in radii},
        "top1": 1.0 / n,
    }


def uniform_baseline(cand_xyz, gt_xyz, radii=(0.5, 1.0)):
    """Uniform-over-C baseline: exact expectations, no sampling.

    ``distances`` is the pooled per-candidate distance list; the per-query weight
    of 1/M over those distances is applied by :func:`summarize`.
    """
    cand = _points(cand_xyz, "cand_xyz")
    gt = _point(gt_xyz, "gt_xyz")
    _gt_index(cand, gt)
    out = _expectations(torch.linalg.norm(cand - gt, dim=-1), radii)
    out["n_candidates"] = cand.shape[0]
    return out


def context_conditioned_baseline(cand_xyz, gt_xyz, context_member_mask, radii=(0.5, 1.0)):
    """Information-matched baseline (C1): uniform over the NON-context candidates.

    The 8 context source positions are visible in the conditioning, so a guesser
    that only eliminates them is the registered comparison target. GT is never a
    context member (excluded by construction) -- a mask that says otherwise is a
    wiring bug and raises. When every non-GT candidate is in the context the
    eligible set is ``{GT}``: the returned ``gt_only`` flag marks that
    zero-headroom case (top-1 = 1.0) so callers can exclude or label it.
    """
    cand = _points(cand_xyz, "cand_xyz")
    gt = _point(gt_xyz, "gt_xyz")
    mask = torch.as_tensor(context_member_mask, dtype=torch.bool).reshape(-1)
    if mask.numel() != cand.shape[0]:
        raise ValueError(
            f"context_member_mask has {mask.numel()} entries for {cand.shape[0]} candidates")
    gt_idx = _gt_index(cand, gt)
    if bool(mask[gt_idx]):
        raise ValueError(
            "GT candidate is flagged as a context member; context is drawn from OTHER "
            "sources by construction, so this is a wiring bug")
    eligible = cand[~mask]
    out = _expectations(torch.linalg.norm(eligible - gt, dim=-1), radii)
    out["n_candidates"] = cand.shape[0]
    out["gt_only"] = out["n_eligible"] == 1
    return out


def nearest_context_baseline(cand_xyz, ctx_xyz, ctx_sims):
    """Non-generative control (O10): index of the candidate nearest the
    best-matching context source (no generator involved).

    ``ctx_sims`` are similarities between the observed RIR and each context RIR;
    both the best-context and the nearest-candidate choices break ties by lowest
    index, as everywhere else in the readout.
    """
    cand = _points(cand_xyz, "cand_xyz")
    ctx = _points(ctx_xyz, "ctx_xyz")
    sims = torch.as_tensor(ctx_sims, dtype=torch.float64).reshape(-1)
    if sims.numel() != ctx.shape[0]:
        raise ValueError(f"ctx_sims has {sims.numel()} entries for {ctx.shape[0]} context sources")
    best_ctx = predict_index(sims)
    return predict_index(-torch.linalg.norm(cand - ctx[best_ctx], dim=-1))


def noise_key(seed, query_id, k):
    """Deterministic ``torch.Generator`` seed for sample ``k`` of ``query_id``.

    sha256 over a canonical JSON payload -- never Python's ``hash()``, which is
    salted per interpreter -- so the noise bank is identical across processes,
    machines and resumed runs, and the K draws are shared across candidates
    (common random numbers, C10).
    """
    payload = json.dumps(
        ["loc_invert_noise_key", int(seed), str(query_id), int(k)], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def power_statistic(sims):
    """Between- over within-candidate variance of ``sims`` ``[M, K]`` (plan §2.8.2).

    ``var_m(mean_k s_{m,k}) / mean_m(var_k s_{m,k})`` with unbiased variances:
    with common random numbers this must exceed 1 for candidate identity -- not
    sampling noise -- to be what moves the similarities.
    """
    if sims.ndim != 2:
        raise ValueError(f"sims must be [M, K], got shape {tuple(sims.shape)}")
    if sims.shape[0] < 2:
        raise ValueError("power_statistic needs at least 2 candidates")
    if sims.shape[1] < 2:
        raise ValueError("power_statistic needs at least 2 samples per candidate")
    s = sims.double()
    between = torch.var(s.mean(dim=-1))
    within = torch.var(s, dim=-1).mean()
    return float(between / within)
