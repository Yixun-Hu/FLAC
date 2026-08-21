"""exp_18 R4 -- offline aggregation of the published metric streams.

Reads what the registered passes already wrote (the replay rows and their
metrics-JSONL sibling) and turns them into the numbers plan_loc_invert_R4 asks
for: per family and seed, the localization block of §1; the metric-matched
retrieval control of §2 recomputed from the recorded context distances; the §3
controls; and the §4 comparisons with Holm-corrected paired tests.

Nothing here re-scores anything: every distance was computed by the registered
unseen passes and is read back bit-exactly from its hex payload. The two
conventions the campaign already fixed -- the reciprocal-rank tie-break and the
nearest-context retrieval geometry -- are reused, not restated: the retrieval
control delegates to :func:`scoring.nearest_context_baseline` through
:func:`rir_metrics.metric_matched_retrieval`, and the rank convention is pinned
against the driver's own function by test.
"""
import json

import numpy as np
import torch

from src.localization.reaggregate import decode_sims
from src.localization.rir_metrics import (aggregate_over_k, metric_matched_retrieval,
                                          predict_from_distances)
from src.localization.scoring import nearest_context_baseline, summarize

#: the five registered families; everything else the rows carry is a DECLARED
#: secondary and is labelled as one wherever it is reported.
PRIMARY_FAMILIES = ("m1", "m2", "m3", "m4", "m5")
SECONDARY_FAMILIES = ("m2_complex", "m3_band", "m3_hilbert", "m5_gcc")
REPORT_FAMILIES = PRIMARY_FAMILIES + SECONDARY_FAMILIES
#: the mandated Delta = 0 alignment-sensitivity rows (reported, never promoted).
ALIGNMENT_SENSITIVITY_FAMILIES = ("m1_delta0", "m5_delta0")

#: K aggregation: mean is primary (plan §1), the rest are declared secondaries.
PRIMARY_AGGREGATION = "mean"
SECONDARY_AGGREGATIONS = ("min", "median", "lme")
AGGREGATIONS = (PRIMARY_AGGREGATION,) + SECONDARY_AGGREGATIONS

#: fixed campaign references (R-1b / R2, K_ctx = 8), quoted not recomputed.
AGREE_RETRIEVAL_REFERENCE = 0.689
AGREE_CONTEXT_MEMBER_RATE = 0.376


def decode_matrix(payload):
    """``[M][K]`` hex payload -> float32 ``[M, K]`` array, bit-exactly."""
    return decode_sims(payload).numpy()


def decode_vector(payload):
    """``[N]`` hex payload -> float32 ``[N]`` array, bit-exactly."""
    return decode_sims([list(payload)]).numpy()[0]


def reciprocal_rank(distances, gt_index):
    """Reciprocal rank of the GT candidate under DISTANCES (lower is better).

    The registered convention (``eval_localization.gt_reciprocal_rank``) ranks
    similarities and breaks ties by lowest index; this is the same rule read for
    distances, and a test pins the two against each other.
    """
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    gt_index = int(gt_index)
    gt_value = distances[gt_index]
    better = int((distances < gt_value).sum())
    tied_before = int((distances[:gt_index] == gt_value).sum())
    return 1.0 / (better + tied_before + 1)


def family_scores(row, family, aggregation=PRIMARY_AGGREGATION):
    """The recorded per-candidate score ``[M]`` of one family and aggregation."""
    block = row["families"][family]
    aggregations = block.get("aggregations") or {}
    if aggregation in aggregations:
        return decode_vector(aggregations[aggregation])
    # a row that predates the aggregation block still carries the raw [M, K]
    return aggregate_over_k(torch.from_numpy(decode_matrix(block["candidates_hex"])),
                            aggregation).numpy()


def _geometry(row):
    world = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
    gt = np.asarray(row["gt_xyz_world"], dtype=np.float64)
    return world, gt


def _record(row, pred_index, scores=None):
    """One query record in the campaign's own shape (``scoring.summarize``)."""
    world, gt = _geometry(row)
    pred_index = int(pred_index)
    gt_index = int(row["gt_index"])
    record = {
        "query_id": row["query_id"], "room_id": row["room_id"],
        "e_loc": float(np.linalg.norm(world[pred_index] - gt)),
        "top1": 1.0 if pred_index == gt_index else 0.0,
        "pred_index": pred_index,
        "context_member_pred": bool(row["context_member"][pred_index]),
    }
    record["rr"] = (1.0 if scores is None and pred_index == gt_index else
                    (reciprocal_rank(scores, gt_index) if scores is not None else 0.0))
    return record


def family_record(row, family, aggregation=PRIMARY_AGGREGATION):
    """The localization record of one family on one query."""
    scores = family_scores(row, family, aggregation)
    return _record(row, predict_from_distances(torch.from_numpy(scores)), scores=scores)


def _distance_to_chosen_context(replay_row, pred_reference_index):
    """Candidate ranking of a retrieval control: distance to the chosen source."""
    cand = np.asarray(replay_row["candidate_xyz_cam"], dtype=np.float64)
    ctx = np.asarray(replay_row["context_xyz_cam"], dtype=np.float64)
    return np.linalg.norm(cand - ctx[int(pred_reference_index)], axis=-1)


def retrieval_record(row, replay_row, family, masked=False):
    """Plan §2's metric-matched control, recomputed from the recorded distances.

    Ranking for the reciprocal rank is the control's own ordering: geometric
    distance to the context source the metric chose.
    """
    ctx_distances = decode_vector(row["families"][family]["context_hex"])
    eligible = [not member for member in row["context_member"]] if masked else None
    pred = metric_matched_retrieval(replay_row["candidate_xyz_cam"],
                                    replay_row["context_xyz_cam"], ctx_distances,
                                    eligible_mask=eligible)
    best_ctx = int(np.argmin(ctx_distances))
    record = _record(row, pred, scores=_distance_to_chosen_context(replay_row, best_ctx))
    record["best_context"] = best_ctx
    return record


def agree_retrieval_record(replay_row, masked=False, row=None):
    """The registered AGREE nearest-context control, per query.

    This is the reference the fixed 0.689 came from; recomputing it per query is
    what makes comparison (a) a PAIRED test rather than a difference of two
    pooled numbers.
    """
    row = row or replay_row
    sims = decode_vector(replay_row["context_sims_hex"])
    eligible = [not member for member in row["context_member"]] if masked else None
    pred = nearest_context_baseline(replay_row["candidate_xyz_cam"],
                                    replay_row["context_xyz_cam"], torch.from_numpy(sims),
                                    eligible_mask=eligible)
    best_ctx = int(np.argmax(sims))
    record = _record(row, pred, scores=_distance_to_chosen_context(replay_row, best_ctx))
    record["best_context"] = best_ctx
    return record


def agree_record(row):
    """The registered FLAC+AGREE prediction of this query, as recorded."""
    return _record(row, int(row["agree_pred_index"]))


def summarize_records(records):
    """The campaign's own summary block (pooled + per-room + macro)."""
    return summarize([{"query_id": r["query_id"], "room_id": r["room_id"],
                       "e_loc": r["e_loc"], "top1": r["top1"], "rr": r["rr"]}
                      for r in records])


def iter_rows(path):
    """Stream a JSONL file one decoded row at a time (the files are ~350 MB)."""
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_joined(metrics_path, rows_path):
    """Stream ``(metrics_row, replay_row)`` pairs, refusing any misalignment.

    The two files are written by the same loop, so a difference in order or
    length means one of them is not the sibling of the other -- every downstream
    number would silently pair the wrong query.
    """
    metrics_stream, rows_stream = iter_rows(metrics_path), iter_rows(rows_path)
    position = 0
    while True:
        metrics_row = next(metrics_stream, None)
        replay_row = next(rows_stream, None)
        if metrics_row is None and replay_row is None:
            return
        if metrics_row is None or replay_row is None:
            raise ValueError(
                f"{metrics_path!r} and {rows_path!r} differ in length at position {position}; "
                "the metrics stream and the rows stream must be siblings of one pass")
        if str(metrics_row["query_id"]) != str(replay_row["query_id"]):
            raise ValueError(
                f"stream mismatch at position {position}: metrics row is "
                f"{metrics_row['query_id']!r} but the replay row is "
                f"{replay_row['query_id']!r}")
        yield metrics_row, replay_row
        position += 1


def scan_seed(metrics_path, rows_path, families=REPORT_FAMILIES,
              aggregations=AGGREGATIONS, secondary_aggregation_families=PRIMARY_FAMILIES):
    """One streaming pass over a seed: every per-query record the report needs.

    Returns compact per-query records only -- the [M, K] payloads are decoded,
    used and dropped, so peak memory stays independent of the file size.
    """
    families = tuple(families)
    records = {family: {agg: [] for agg in aggregations
                        if agg == PRIMARY_AGGREGATION
                        or family in secondary_aggregation_families}
               for family in families}
    retrieval = {family: {"raw": [], "masked": []} for family in families}
    agree, agree_retrieval = [], {"raw": [], "masked": []}
    rooms, n_queries = set(), 0

    for metrics_row, replay_row in iter_joined(metrics_path, rows_path):
        present = [family for family in families if family in metrics_row["families"]]
        for family in present:
            for aggregation in records[family]:
                records[family][aggregation].append(
                    family_record(metrics_row, family, aggregation))
            retrieval[family]["raw"].append(
                retrieval_record(metrics_row, replay_row, family, masked=False))
            retrieval[family]["masked"].append(
                retrieval_record(metrics_row, replay_row, family, masked=True))
        agree.append(agree_record(metrics_row))
        for mode in ("raw", "masked"):
            agree_retrieval[mode].append(
                agree_retrieval_record(replay_row, masked=(mode == "masked"),
                                       row=metrics_row))
        rooms.add(metrics_row["room_id"])
        n_queries += 1

    if not n_queries:
        raise ValueError(f"{metrics_path!r} contains no rows")
    return {"records": {f: b for f, b in records.items() if b[PRIMARY_AGGREGATION]},
            "retrieval": {f: b for f, b in retrieval.items() if b["raw"]},
            "agree": agree, "agree_retrieval": agree_retrieval,
            "n_queries": n_queries, "n_rooms": len(rooms),
            "families_present": [f for f in families if records[f][PRIMARY_AGGREGATION]],
            "metrics_path": str(metrics_path), "rows_path": str(rows_path)}
