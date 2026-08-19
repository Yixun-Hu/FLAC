"""Pure scoring / metric / baseline functions for exp_18 (loc_invert).

No I/O, no model, no global RNG: everything here is a deterministic function of
its arguments, so the driver's numbers can be re-derived offline from the logged
similarities.
"""
import hashlib
import json
import math

import numpy as np
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


#: name of the registered primary metric (plan §2.6).
PRIMARY_NAME = "pooled_median_e_loc"
DEFAULT_RADII = (0.5, 1.0)
ROOM_KEY = "room_id"


def _record_values(rec):
    """``(values, weights)`` of one query record; every query weighs 1 in total.

    A FLAC record carries one ``e_loc``; a baseline record carries the whole
    per-candidate ``distances`` list, each candidate weighted 1/M -- so both are
    summarized under identical conventions.
    """
    if not isinstance(rec, dict) or ROOM_KEY not in rec:
        raise ValueError(f"record must be a dict with a {ROOM_KEY!r} key, got {rec!r}")
    has_dist, has_e = "distances" in rec, "e_loc" in rec
    if has_dist == has_e:
        raise ValueError(f"record needs exactly one of 'e_loc' / 'distances': {rec!r}")
    if has_dist:
        values = np.asarray(rec["distances"], dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"record has an empty 'distances' list: {rec!r}")
    else:
        values = np.asarray([float(rec["e_loc"])], dtype=np.float64)
    return values, np.full(values.shape, 1.0 / values.size, dtype=np.float64)


def _weighted_median(values, weights):
    """Weighted median; reduces exactly to ``np.median`` for equal weights."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    half = 0.5 * cum[-1]
    i = int(np.searchsorted(cum, half, side="left"))
    if i + 1 < v.size and abs(cum[i] - half) <= 1e-12 * cum[-1]:
        return float(0.5 * (v[i] + v[i + 1]))
    return float(v[i])


def _weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


def _optional_column(records, key):
    """Per-query column for an optional field: ``None`` if absent everywhere,
    ``ValueError`` if only some records carry it (silent mixing would bias it)."""
    present = [key in rec for rec in records]
    if all(present):
        return np.asarray([float(rec[key]) for rec in records], dtype=np.float64)
    if any(present):
        raise ValueError(f"field {key!r} is present in only {sum(present)}/{len(records)} records")
    return None


def _block(values, weights, radii, top1, rr, n_queries):
    return {
        "n_queries": n_queries,
        "median_e_loc": _weighted_median(values, weights),
        "mean_e_loc": _weighted_mean(values, weights),
        "success": {float(r): _weighted_mean((values <= float(r)).astype(np.float64), weights)
                    for r in radii},
        "top1": None if top1 is None else float(np.mean(top1)),
        "mrr": None if rr is None else float(np.mean(rr)),
    }


def summarize(records, radii=DEFAULT_RADII):
    """Pooled primary + labelled secondaries over per-query records.

    Primary (C3) is the pooled median ``e_loc`` over all query records; pooled
    mean and success@r accompany it. Secondaries are labelled: per-room stats
    (``room_id`` = ``scene/scene_id``), the equal-room macro averages including
    the mean of per-room medians, top-1 accuracy and the MRR of the GT candidate.
    Baseline records are summarized through this same function, so the weighting
    and the boundary conventions are identical by construction.
    """
    records = list(records)
    if not records:
        raise ValueError("summarize needs at least one record")
    for r in radii:
        if float(r) < 0.0:
            raise ValueError(f"radius must be >= 0, got {r}")

    per_query = [_record_values(rec) for rec in records]
    top1 = _optional_column(records, "top1")
    rr = _optional_column(records, "rr")

    values = np.concatenate([v for v, _ in per_query])
    weights = np.concatenate([w for _, w in per_query])
    pooled = _block(values, weights, radii, top1, rr, len(records))

    rooms = {}
    for idx, rec in enumerate(records):
        rooms.setdefault(str(rec[ROOM_KEY]), []).append(idx)
    per_room = {}
    for room in sorted(rooms):
        idxs = rooms[room]
        per_room[room] = _block(
            np.concatenate([per_query[i][0] for i in idxs]),
            np.concatenate([per_query[i][1] for i in idxs]),
            radii,
            None if top1 is None else top1[idxs],
            None if rr is None else rr[idxs],
            len(idxs),
        )

    blocks = list(per_room.values())
    macro = {
        "n_rooms": len(per_room),
        "mean_of_room_medians": float(np.mean([b["median_e_loc"] for b in blocks])),
        "mean_of_room_means": float(np.mean([b["mean_e_loc"] for b in blocks])),
        "success": {float(r): float(np.mean([b["success"][float(r)] for b in blocks]))
                    for r in radii},
        "top1": None if top1 is None else float(np.mean([b["top1"] for b in blocks])),
        "mrr": None if rr is None else float(np.mean([b["mrr"] for b in blocks])),
    }
    return {
        "primary_name": PRIMARY_NAME,
        "primary": pooled["median_e_loc"],
        "pooled": pooled,
        "per_room": per_room,
        "macro": macro,
        "n_queries": len(records),
        "n_rooms": len(per_room),
    }


def _cluster_bootstrap(payloads, statistic, n, seed):
    """Resample whole clusters with replacement; return the ``n`` statistics."""
    n = int(n)
    if n < 1:
        raise ValueError(f"n (bootstrap resamples) must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    k = len(payloads)
    draws = rng.integers(0, k, size=(n, k))
    return np.asarray([statistic([payloads[j] for j in row]) for row in draws], dtype=np.float64)


def _percentile_ci(samples, alpha):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    lo, hi = np.percentile(samples, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def clustered_bootstrap_ci(records, by=ROOM_KEY, n=10000, seed=0, alpha=0.05):
    """Room-clustered percentile CI on the pooled-median primary (C3).

    Queries within a room are not independent, so the resampling unit is the
    room. With a single cluster the interval collapses onto the point estimate --
    honest, since there is no between-room information to resample.
    """
    records = list(records)
    if not records:
        raise ValueError("clustered_bootstrap_ci needs at least one record")
    clusters = {}
    for rec in records:
        if by not in rec:
            raise ValueError(f"record has no cluster key {by!r}: {rec!r}")
        clusters.setdefault(str(rec[by]), []).append(_record_values(rec))
    payloads = [(np.concatenate([v for v, _ in items]), np.concatenate([w for _, w in items]))
                for _key, items in sorted(clusters.items())]

    def statistic(picked):
        return _weighted_median(np.concatenate([v for v, _ in picked]),
                                np.concatenate([w for _, w in picked]))

    samples = _cluster_bootstrap(payloads, statistic, n, seed)
    lo, hi = _percentile_ci(samples, alpha)
    return {
        "stat": PRIMARY_NAME,
        "point": statistic(payloads),
        "lo": lo,
        "hi": hi,
        "alpha": float(alpha),
        "n_boot": int(n),
        "n_clusters": len(payloads),
        "n_queries": len(records),
        "by": by,
    }


def _query_point_estimate(rec):
    """One scalar per query: ``e_loc``, or a baseline record's expected error."""
    values, weights = _record_values(rec)
    return _weighted_mean(values, weights)


def paired_room_clustered_test(records_a, records_b, by=ROOM_KEY, n=10000, seed=0, stat="median"):
    """Room-clustered bootstrap on per-query paired differences ``a - b`` (O12).

    Records are paired by ``query_id`` (1:1, same room on both sides); the
    bootstrap resamples rooms, and ``p_value`` is the two-sided bootstrap
    proportion crossing zero.
    """
    if stat not in ("median", "mean"):
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
    records_a, records_b = list(records_a), list(records_b)
    if len(records_a) != len(records_b):
        raise ValueError(f"paired inputs differ in length: {len(records_a)} vs {len(records_b)}")

    def indexed(records, side):
        out = {}
        for rec in records:
            if "query_id" not in rec:
                raise ValueError(f"record in {side} has no 'query_id': {rec!r}")
            qid = str(rec["query_id"])
            if qid in out:
                raise ValueError(f"duplicate query_id {qid!r} in {side}")
            out[qid] = rec
        return out

    left, right = indexed(records_a, "records_a"), indexed(records_b, "records_b")
    if set(left) != set(right):
        raise ValueError("records_a and records_b do not cover the same query ids")

    clusters = {}
    for qid in sorted(left):
        rec_a, rec_b = left[qid], right[qid]
        if by not in rec_a or by not in rec_b:
            raise ValueError(f"record has no cluster key {by!r} for query {qid!r}")
        if str(rec_a[by]) != str(rec_b[by]):
            raise ValueError(
                f"query {qid!r} is in {rec_a[by]!r} on one side and {rec_b[by]!r} on the other")
        clusters.setdefault(str(rec_a[by]), []).append(
            _query_point_estimate(rec_a) - _query_point_estimate(rec_b))
    payloads = [np.asarray(diffs, dtype=np.float64) for _key, diffs in sorted(clusters.items())]
    reduce = np.median if stat == "median" else np.mean

    def statistic(picked):
        return float(reduce(np.concatenate(picked)))

    samples = _cluster_bootstrap(payloads, statistic, n, seed)
    lo, hi = _percentile_ci(samples, 0.05)
    p_value = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
    return {
        "stat": f"{stat}_paired_difference",
        "point": statistic(payloads),
        "lo": lo,
        "hi": hi,
        "p_value": min(1.0, p_value),
        "n_boot": int(n),
        "n_clusters": len(payloads),
        "n_queries": len(records_a),
        "by": by,
    }
