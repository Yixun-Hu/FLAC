"""Mapping-A foundations: physical placements and microphone correspondence.

exp_21, contract section A. Mapping A conditions a target RIR on other sources
heard by THE SAME microphone, so "the same microphone" must be a measured fact
before a single item is built. The inherited placement key -- a receiver centroid
rounded to 1 cm, explicitly informational -- cannot carry that weight: rounding
splits one re-occupation across adjacent bins and merges neighbouring placements,
and a distance cutoff alone never establishes a one-to-one 36-way correspondence
(Codex M2).

So placements are clustered by COMPLETE linkage (every pair within the cap, no
transitive chaining) and each tx-group is matched to its placement's medoid
template by a Hungarian assignment whose displacement statistics and ambiguity
margins are recorded and gated. A group that fails is excluded BEFORE eligibility;
nothing downstream silently shrinks.
"""
import hashlib

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Registered tolerances (plan Rev 2 section 2). Sub-centimetre matching keeps the
# same-microphone claim inside the acoustic scale that waveform metrics care about.
PLACEMENT_CAP_M = 0.05          # complete-linkage cap between re-occupations
MATCH_P95_M = 0.01              # 95th percentile matched displacement
MATCH_MAX_M = 0.02              # hard per-mic anomaly cap
MATCH_AMBIGUITY_MARGIN = 3.0    # next-nearest-mic distance / matched displacement
CANONICAL_ARRAY_SIZE = 36
MATCH_ALGORITHM_VERSION = "mappingA-correspondence-1"


def _as_points(array, name, expected_n=None):
    points = np.asarray(array, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}")
    if expected_n is not None and points.shape[0] != expected_n:
        raise ValueError(
            f"{name} must hold {expected_n} points, got {points.shape[0]}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} holds non-finite coordinates")
    return points


def match_mics(template_rx, group_rx, expected_n=CANONICAL_ARRAY_SIZE):
    """One-to-one microphone correspondence between a tx-group and its template.

    A Hungarian assignment minimises total displacement, which is what makes the
    result a PERMUTATION rather than a per-mic nearest-neighbour lookup: greedy
    nearest matching can map two template slots onto one group mic and leave
    another unmatched, and that error is invisible in the displacement summary.

    Three registered gates, all recorded either way:

    * ``p95`` matched displacement <= ``MATCH_P95_M``;
    * every matched displacement <= ``MATCH_MAX_M`` (one anomalous mic must fail
      even when the percentile is clean);
    * ambiguity margin >= ``MATCH_AMBIGUITY_MARGIN``, where the margin for a slot
      is the distance to the SECOND-nearest group mic divided by its matched
      displacement. A 3 mm displacement between mics 6 mm apart is inside every
      distance tolerance and still a coin flip -- only the margin sees that.

    Returns the report regardless of outcome; ``passed`` and ``reasons`` carry the
    verdict.
    """
    template = _as_points(template_rx, "template_rx", expected_n)
    group = _as_points(group_rx, "group_rx", expected_n)
    if template.shape[0] != group.shape[0]:
        raise ValueError(
            f"template ({template.shape[0]}) and group ({group.shape[0]}) hold "
            "different numbers of microphones")
    n = template.shape[0]
    if n < 2:
        raise ValueError("a correspondence needs at least two microphones")

    cost = np.linalg.norm(template[:, None, :] - group[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    assignment = [int(c) for c in cols[np.argsort(rows)]]
    displacements = [float(cost[slot, assignment[slot]]) for slot in range(n)]

    margins = []
    for slot in range(n):
        distances = np.sort(cost[slot])
        second = float(distances[1])
        matched = displacements[slot]
        margins.append(second / matched if matched > 0 else float("inf"))

    p50 = float(np.percentile(displacements, 50))
    p95 = float(np.percentile(displacements, 95))
    worst = float(np.max(displacements))
    min_margin = float(np.min(margins))

    reasons = []
    if p95 > MATCH_P95_M:
        reasons.append(f"p95 displacement {p95:.4f} m > {MATCH_P95_M} m")
    if worst > MATCH_MAX_M:
        reasons.append(f"max displacement {worst:.4f} m > {MATCH_MAX_M} m")
    if min_margin < MATCH_AMBIGUITY_MARGIN:
        reasons.append(
            f"ambiguity margin {min_margin:.2f} < {MATCH_AMBIGUITY_MARGIN}: the "
            "matched mic is not decisively nearer than the next one")

    return {
        "algorithm_version": MATCH_ALGORITHM_VERSION,
        "n": n,
        "assignment": assignment,
        "displacements_m": displacements,
        "ambiguity_margins": margins,
        "p50_m": p50,
        "p95_m": p95,
        "max_m": worst,
        "min_ambiguity_margin": min_margin,
        "tolerances": {"p95_m": MATCH_P95_M, "max_m": MATCH_MAX_M,
                       "ambiguity_margin": MATCH_AMBIGUITY_MARGIN},
        "passed": not reasons,
        "reasons": reasons,
    }


def _complete_linkage(centroids, cap):
    """Agglomerative complete linkage under a distance cap.

    Complete linkage on purpose: single linkage would chain A-B-C into one
    "placement" whenever consecutive re-occupations are close, even though A and C
    are far apart. Deterministic: the closest admissible pair merges first, ties
    broken by the lowest index pair.
    """
    clusters = [[i] for i in range(len(centroids))]
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    while True:
        best, best_pair = None, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                linkage = max(distances[i, j] for i in clusters[a] for j in clusters[b])
                if linkage <= cap and (best is None or linkage < best - 1e-12):
                    best, best_pair = linkage, (a, b)
        if best_pair is None:
            return clusters
        a, b = best_pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        clusters.pop(b)


def cluster_placements(groups, cap=PLACEMENT_CAP_M, expected_n=CANONICAL_ARRAY_SIZE):
    """Group tx-groups into physical array placements.

    ``groups`` are exp_19 group dicts (``group_key``, ``rx_xyz_p``). Returns one
    entry per placement with its members, a deterministic medoid template, and the
    template's centroid. The medoid -- the member closest to the others -- is used
    rather than a synthetic mean array so the template is a REAL measured
    placement, which is what the per-group correspondence is then measured against.
    """
    if not groups:
        return []
    keys, arrays, centroids = [], [], []
    for group in groups:
        if "group_key" not in group or "rx_xyz_p" not in group:
            raise ValueError(
                f"group {group.get('group_key', '?')!r} carries no rx_xyz_p: the "
                "placement clustering needs the receiver array")
        array = _as_points(group["rx_xyz_p"], f"{group['group_key']}.rx_xyz_p",
                           expected_n)
        keys.append(str(group["group_key"]))
        arrays.append(array)
        centroids.append(array.mean(axis=0))
    centroids = np.vstack(centroids)

    members = _complete_linkage(centroids, cap)
    # Deterministic order: by the lexicographically first member key, so a shuffled
    # input yields identical placement ids.
    members.sort(key=lambda idx: sorted(keys[i] for i in idx)[0])

    clusters = []
    for position, indices in enumerate(members):
        within = np.linalg.norm(centroids[indices][:, None, :]
                                - centroids[indices][None, :, :], axis=-1).sum(axis=1)
        # ties -> lowest member key, so the template never depends on input order
        order = sorted(range(len(indices)),
                       key=lambda k: (within[k], keys[indices[k]]))
        medoid = indices[order[0]]
        clusters.append({
            "placement_id": f"p{position:03d}",
            "member_keys": sorted(keys[i] for i in indices),
            "medoid_key": keys[medoid],
            "template_rx": arrays[medoid],
            "centroid_p": centroids[medoid],
            "n_members": len(indices),
            "cap_m": float(cap),
            "algorithm_version": MATCH_ALGORITHM_VERSION,
        })
    return clusters


# --------------------------------------------------------------------------- #
# Target and context selection
# --------------------------------------------------------------------------- #
DEFAULT_K = 8


def source_xyz_key(xyz):
    """Canonical identity of a SOURCE POSITION: the 6-decimal rendering.

    RAF's pose text carries exactly six decimals, so this is lossless on parsed
    values, and ``-0.0`` is normalised so a sign flip cannot spell one position two
    ways. Two tx-groups with this key equal are the same loudspeaker position --
    including quaternion-only duplicates, which is exactly what "unseen source
    POSITION" has to exclude (M5).
    """
    values = np.asarray(xyz, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"source xyz must have shape (3,), got {values.shape}")
    values = np.where(values == 0.0, 0.0, values)
    return ",".join(f"{v:.6f}" for v in values)


def target_digest(room, placement_id, pose_key, seed=0):
    """sha256 of the registered target-selection payload (stable across machines)."""
    payload = f"mappingA-target|{seed}|{room}|{placement_id}|{pose_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_target(room, placement_id, poses, seed=0):
    """Hash-uniform target pose for one placement (M8).

    The estimand is GENERAL unseen-source performance, so the target is drawn
    uniformly by a stable hash rather than by a spatial rule: farthest-point
    selection would systematically pick the most extreme source in each placement
    and silently turn the row into a spatial-stress test. FPS is retained only for
    placement coverage, one level up.
    """
    if not poses:
        raise ValueError(
            f"{room}/{placement_id}: no eligible poses to select a target from")
    ranked = sorted(poses, key=lambda p: (target_digest(room, placement_id,
                                                        str(p["group_key"]), seed),
                                          str(p["group_key"])))
    return ranked[0]


def context_digest(room, placement_id, mic_slot, target_key, seed=0):
    """sha256 of the registered per-item context payload."""
    payload = (f"mappingA-context|{seed}|{room}|{placement_id}|{int(mic_slot)}|"
               f"{target_key}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_item_context(room, placement_id, mic_slot, target, candidates,
                        k=DEFAULT_K, seed=0):
    """Deterministic K-context for one Mapping-A item.

    The draw is a function of (room, placement, mic slot, target) alone -- never of
    worker topology, ambient RNG, checkpoint or eval seed -- so every arm and seed
    conditions on exactly the same references and an arm contrast is a paired
    comparison rather than a comparison of different problems.

    Excluded from the pool: the target itself and EVERY group sharing its source
    xyz (quaternion-only duplicates included). A same-position source in the
    context would make the "unseen source position" claim false.
    """
    target_key = str(target["group_key"])
    excluded_key = source_xyz_key(target["tx_xyz"])
    pool, n_excluded = [], 0
    for candidate in candidates:
        if source_xyz_key(candidate["tx_xyz"]) == excluded_key:
            n_excluded += 1
            continue
        pool.append(candidate)
    pool.sort(key=lambda c: str(c["group_key"]))

    if len(pool) < k:
        raise ValueError(
            f"{room}/{placement_id} slot {mic_slot}: context pool holds "
            f"{len(pool)} source-distinct groups, need {k}")

    generator = torch.Generator()
    generator.manual_seed(
        int(context_digest(room, placement_id, mic_slot, target_key, seed)[:16], 16)
        & ((1 << 63) - 1))
    picks = torch.randperm(len(pool), generator=generator)[:k]
    context = [pool[int(i)] for i in picks]
    return {
        "context": context,
        "context_keys": [str(c["group_key"]) for c in context],
        "pool_size": len(pool),
        "n_excluded_same_xyz": n_excluded,
        "target_key": target_key,
        "target_xyz_key": excluded_key,
        "seed": seed,
        "digest": context_digest(room, placement_id, mic_slot, target_key, seed),
    }
