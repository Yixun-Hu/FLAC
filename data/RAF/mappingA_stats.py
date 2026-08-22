"""Placement-level statistics for the Mapping-A cross-arm comparison.

exp_21, contract section E / plan section 6. The clustering unit is the PLACEMENT,
not the item: the 36 items of a placement share one room position, one array, one
target source and largely overlapping context, so treating them as 1,152
independent observations would understate every interval by roughly the square root
of that dependence. So: aggregate within placement first, then bootstrap over
placements (room-stratified), and contrast arms by a PAIRED placement-level
randomization -- the arms saw identical items, and throwing that pairing away would
be a strictly weaker test.

No item-i.i.d. intervals, and no generalization claims beyond these two rooms.
"""
import hashlib
import itertools
import math

import numpy as np

EXACT_RANDOMIZATION_LIMIT = 20   # 2**20 sign assignments is still cheap and EXACT


def aggregate_within_placement(records, metric):
    """{(room, placement): mean of ``metric`` over that placement's items}."""
    sums, counts = {}, {}
    for record in records:
        key = (record["room"], record["placement_id"])
        value = record["metrics"][metric]
        if value is None or not np.isfinite(value):
            raise ValueError(
                f"{key} item {record.get('item_id')}: metric {metric} is {value!r}; "
                "an aggregate over silently dropped items is a different estimand")
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}, counts


def macro_two_room(placement_means):
    """Mean over placements within each room, then the equal-room macro.

    Equal-room on purpose: the two rooms are the population, and weighting by
    placement count would let whichever room contributed more placements set the
    headline.
    """
    by_room = {}
    for (room, _placement), value in placement_means.items():
        by_room.setdefault(room, []).append(value)
    room_means = {room: float(np.mean(values)) for room, values in sorted(by_room.items())}
    return {
        "rooms": room_means,
        "macro": float(np.mean([room_means[r] for r in sorted(room_means)])),
        "n_placements": {room: len(values) for room, values in sorted(by_room.items())},
    }


def _stable_seed(label):
    return int(hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)


def cluster_bootstrap(placement_means, n_resamples=10000, alpha=0.05, label="mappingA"):
    """Room-stratified cluster bootstrap over PLACEMENTS.

    Resampling placements (with replacement, within room) is what makes the
    interval reflect the dependence the design actually has; resampling items would
    treat 36 recordings of one array position as 36 independent facts.
    """
    by_room = {}
    for (room, _placement), value in placement_means.items():
        by_room.setdefault(room, []).append(value)
    rooms = sorted(by_room)
    arrays = {room: np.asarray(by_room[room], dtype=np.float64) for room in rooms}
    rng = np.random.default_rng(_stable_seed(label))

    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        room_means = []
        for room in rooms:
            values = arrays[room]
            picks = rng.integers(0, len(values), size=len(values))
            room_means.append(values[picks].mean())
        draws[i] = float(np.mean(room_means))

    point = macro_two_room(placement_means)["macro"]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": float(alpha),
        "n_resamples": int(n_resamples),
        "unit": "placement",
        "stratified_by": "room",
        "n_placements": {room: len(arrays[room]) for room in rooms},
    }


def paired_randomization(arm_a, arm_b, n_resamples=10000, label="mappingA"):
    """Paired placement-level randomization test for an arm contrast.

    The arms evaluated identical items under identical conditioning, so the pairing
    is real and the null is a per-placement sign flip. Exact enumeration for up to
    ``EXACT_RANDOMIZATION_LIMIT`` placements -- with 32 placements the sampled
    p-value is used and reported as such.
    """
    keys = sorted(set(arm_a) & set(arm_b))
    missing = sorted(set(arm_a) ^ set(arm_b))
    if missing:
        raise ValueError(
            f"the two arms do not cover the same placements: {missing}. A paired "
            "test over different placements is not a paired test.")
    if not keys:
        raise ValueError("no placements to compare")

    diffs = np.array([arm_a[k] - arm_b[k] for k in keys], dtype=np.float64)
    observed = float(np.mean(diffs))
    n = len(diffs)

    if n <= EXACT_RANDOMIZATION_LIMIT:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = (signs * diffs).mean(axis=1)
        exact = True
        n_used = len(stats)
    else:
        rng = np.random.default_rng(_stable_seed(label))
        signs = rng.choice([1.0, -1.0], size=(n_resamples, n))
        stats = (signs * diffs).mean(axis=1)
        exact = False
        n_used = n_resamples

    p_value = float(np.mean(np.abs(stats) >= abs(observed) - 1e-15))
    return {
        "observed_difference": observed,
        "p_value": p_value,
        "exact": exact,
        "n_assignments": int(n_used),
        "n_placements": n,
        "unit": "placement",
        "placements": keys,
    }


def seed_variability(per_seed_macros):
    """Monte-Carlo variability of the diffusion seed, reported SEPARATELY.

    It is not a confidence interval about the arm: it says how much one arm's
    number moves when only the sampler's noise changes.
    """
    values = np.asarray(list(per_seed_macros.values()), dtype=np.float64)
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "note": "Monte-Carlo seed variability, not an interval about the estimand",
    }
