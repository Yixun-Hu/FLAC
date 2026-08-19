"""Offline re-aggregation of exp_18 rows (R1's tau/aggregation/K' selection).

The driver logs every per-sample similarity at full float32 precision, so tau,
the aggregation method and K' can be re-selected from the rows alone -- no
regeneration, no GPU. R1's selection is therefore reviewed code with tests rather
than an after-the-fact script (plan Rev 3.1 §3).

Registered selection (plan §2.5): LME with K'=8, objective = dev pooled MEAN
e_loc (the median is a step function of top-1 at M~10), ties broken towards the
smallest tau.
"""
import json

import numpy as np
import torch

from src.localization.scoring import aggregate, localization_error, predict_index, summarize

#: the dev grid the plan registers for tau selection.
DEFAULT_TAUS = (0.02, 0.05, 0.1, 0.2, 0.5)
DEFAULT_METHODS = ("lme", "mean", "max")
DEFAULT_K_PRIMES = (1, 2, 4, 8)
#: the registered objective and cell.
REGISTERED_OBJECTIVE = "pooled_mean_e_loc"
REGISTERED_METHOD = "lme"
REGISTERED_K_PRIME = 8


def encode_sims(sims):
    """``[M, K]`` similarities as exact hex floats (``float.hex``).

    Widening a float32 to float64 is exact and ``float.fromhex`` inverts it bit
    for bit, so an offline re-aggregation reproduces the online scores exactly
    (O18). Defined here, next to the code that consumes it, and re-exported by the
    driver so there is only one codec.
    """
    return [[float(v).hex() for v in row] for row in sims.detach().cpu().float()]


def decode_sims(payload):
    """Inverse of :func:`encode_sims` -> float32 ``[M, K]``."""
    return torch.tensor([[float.fromhex(v) for v in row] for row in payload], dtype=torch.float32)


def decode_scores(payload):
    """Inverse of a row's ``scores_hex`` -> float32 ``[M]``."""
    return decode_sims([payload])[0]


def read_rows(path):
    """Read one JSONL row file."""
    rows = []
    with open(str(path), "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recompute_row(row, method, tau, k_prime):
    """Re-derive one query's prediction under ``(method, tau, K')``.

    ``K'`` takes the FIRST K' samples in generation order -- the noise bank is
    keyed by ``k``, so the prefix of a K=8 run is exactly what a K'=4 run would
    have drawn.
    """
    sims = decode_sims(row["sims_hex"])
    k_prime = int(k_prime)
    if k_prime < 1 or k_prime > sims.shape[1]:
        raise ValueError(f"K'={k_prime} is outside the logged {sims.shape[1]} samples")
    scores = aggregate(sims[:, :k_prime], method, tau if method == "lme" else None)

    available = row.get("candidate_available") or [True] * sims.shape[0]
    usable = [i for i, flag in enumerate(available) if flag]
    if not usable:
        raise ValueError(f"query {row.get('query_id')!r} has no available candidate")
    pred_index = usable[predict_index(scores.index_select(0, torch.tensor(usable,
                                                                         dtype=torch.long)))]
    candidates = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
    gt = np.asarray(row["gt_xyz_world"], dtype=np.float64)
    return {"pred_index": int(pred_index),
            "e_loc": localization_error(candidates[pred_index], gt),
            "top1": 1.0 if int(pred_index) == int(row["gt_index"]) else 0.0}


def _configs(taus, methods, k_primes):
    """The grid, without tau duplicates for methods that ignore tau."""
    for method in methods:
        for k_prime in k_primes:
            if method == "lme":
                for tau in taus:
                    yield method, float(tau), int(k_prime)
            else:
                yield method, None, int(k_prime)


def sweep(rows, taus=DEFAULT_TAUS, methods=DEFAULT_METHODS, k_primes=DEFAULT_K_PRIMES):
    """Pooled mean/median e_loc (and top-1) for every configuration of the grid."""
    rows = list(rows)
    if not rows:
        raise ValueError("sweep needs at least one row")
    sample_counts = {int(row["n_samples"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError(f"rows disagree on the sample count: {sorted(sample_counts)}; "
                         "K' semantics would not be comparable across them")
    logged_k = sample_counts.pop()
    usable_k_primes = tuple(int(k) for k in k_primes if int(k) <= logged_k)
    if not usable_k_primes:
        raise ValueError(f"no requested K' fits the {logged_k} logged samples")

    results = []
    for method, tau, k_prime in _configs(taus, methods, usable_k_primes):
        records = []
        for row in rows:
            outcome = recompute_row(row, method, tau, k_prime)
            records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                            "e_loc": outcome["e_loc"], "top1": outcome["top1"]})
        pooled = summarize(records)["pooled"]
        results.append({"method": method, "tau": tau, "k_prime": k_prime,
                        "pooled_mean_e_loc": pooled["mean_e_loc"],
                        "pooled_median_e_loc": pooled["median_e_loc"],
                        "top1": pooled["top1"], "n_queries": len(records)})
    return results


def select_registered(results, method=REGISTERED_METHOD, k_prime=REGISTERED_K_PRIME):
    """The registered choice: minimum pooled MEAN e_loc, smallest tau on a tie."""
    cell = [r for r in results if r["method"] == method and int(r["k_prime"]) == int(k_prime)]
    if not cell:
        raise ValueError(f"the registered cell (method={method!r}, K'={k_prime}) is not in the "
                         "sweep; tau cannot be selected")
    best = min(cell, key=lambda r: (r[REGISTERED_OBJECTIVE], float(r["tau"])))
    chosen = dict(best)
    chosen["objective"] = REGISTERED_OBJECTIVE
    return chosen


def reaggregate(row_files, taus=DEFAULT_TAUS, methods=DEFAULT_METHODS,
                k_primes=DEFAULT_K_PRIMES, select_k_prime=None):
    """Full offline report: the grid plus the registered selection."""
    row_files = [str(p) for p in row_files]
    rows = []
    for path in row_files:
        rows.extend(read_rows(path))
    results = sweep(rows, taus=taus, methods=methods, k_primes=k_primes)
    available_k = sorted({int(r["k_prime"]) for r in results})
    if select_k_prime is None:
        # the registered cell is K'=8; if the rows carry fewer samples, select at the
        # largest K' they actually support and say so in the report.
        select_k_prime = REGISTERED_K_PRIME if REGISTERED_K_PRIME in available_k \
            else max(available_k)
    return {
        "row_files": row_files,
        "n_rows": len(rows),
        "taus": [float(t) for t in taus],
        "methods": list(methods),
        "k_primes_requested": [int(k) for k in k_primes],
        "k_primes_evaluated": available_k,
        "sweep": results,
        "selected": select_registered(results, k_prime=select_k_prime),
        "registered_rule": (f"LME at K'={select_k_prime}, objective = dev pooled MEAN e_loc, "
                            "smallest-tau tie-break (plan §2.5)"),
    }
