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

#: the campaign's registered resampling settings (scoring.clustered_bootstrap_ci).
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 0
ALPHA = 0.05


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


def compare_records(records_a, records_b, label, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """One §4 comparison: paired per-query differences with clustered CIs.

    Two quantities, both paired by query and bootstrapped over the 17 rooms: the
    campaign's primary (``e_loc``, median paired difference) and the top-1
    indicator (mean paired difference), which is the quantity the conclusion
    questions are phrased in. Negative ``e_loc`` and positive ``top1`` both mean
    "A is better than B".
    """
    from src.localization.scoring import paired_room_clustered_test

    def as_records(records, key):
        return [{"query_id": r["query_id"], "room_id": r["room_id"],
                 "e_loc": float(r[key])} for r in records]

    e_loc = paired_room_clustered_test(as_records(records_a, "e_loc"),
                                       as_records(records_b, "e_loc"),
                                       n=n_boot, seed=seed, stat="median")
    top1 = paired_room_clustered_test(as_records(records_a, "top1"),
                                      as_records(records_b, "top1"),
                                      n=n_boot, seed=seed, stat="mean")
    mean_top1_a = float(np.mean([r["top1"] for r in records_a]))
    mean_top1_b = float(np.mean([r["top1"] for r in records_b]))
    return {"label": label, "e_loc": e_loc, "top1": top1,
            "quantity": {"e_loc": "e_loc_metres", "top1": "top1_indicator"},
            "top1_a": mean_top1_a, "top1_b": mean_top1_b,
            "top1_delta": mean_top1_a - mean_top1_b,
            "n_queries": len(records_a)}


def holm_over(comparisons, quantity="e_loc", alpha=ALPHA):
    """Holm-Bonferroni over the registered primary tests, via the R4 helper."""
    from src.localization.rir_metrics import holm_bonferroni

    return holm_bonferroni({str(c["label"]): float(c[quantity]["p_value"])
                            for c in comparisons}, alpha=alpha)


def power_by_family(row, family):
    """Between-candidate over within-sample variation of one family's [M, K].

    ``None`` when the family cannot support the statistic on this query (a
    degenerate or all-NaN block, which M4 produces when its validity mask empties
    a query) -- reported as a skip rather than silently averaged as zero.
    """
    from src.localization.scoring import power_statistic

    matrix = decode_matrix(row["families"][family]["candidates_hex"])
    if matrix.shape[0] < 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        return None
    return float(power_statistic(torch.from_numpy(matrix)))


def context_split(records):
    """Plan §3: performance where the PREDICTION is a context member, and where
    it is not (the split the r4m3 review fixed: splitting on the GT gives an
    always-empty bucket, since a GT in its own context aborts upstream)."""
    buckets = {"context": [], "non_context": []}
    for record in records:
        buckets["context" if record["context_member_pred"] else "non_context"].append(record)
    out = {}
    for name, bucket in buckets.items():
        out[name] = {
            "n_queries": len(bucket),
            "top1": float(np.mean([r["top1"] for r in bucket])) if bucket else None,
            "median_e_loc": (float(np.median([r["e_loc"] for r in bucket]))
                             if bucket else None),
            "mean_e_loc": float(np.mean([r["e_loc"] for r in bucket])) if bucket else None,
        }
    out["context_member_rate"] = (float(np.mean([r["context_member_pred"] for r in records]))
                                  if records else None)
    out["n_queries"] = len(records)
    return out


class M4Accumulator:
    """Streaming M4 diagnostics: per-feature discrimination and drop statistics.

    Same formulas the seen calibration pass reported (between/within variance of
    the feature across candidates, and the feature's own top-1), evaluated here
    on whichever split is being aggregated.
    """

    def __init__(self, names=None):
        from src.localization.rir_metrics import M4_FEATURES

        self.names = tuple(names or M4_FEATURES)
        self.n_queries = 0
        self.between = [[] for _ in self.names]
        self.within = [[] for _ in self.names]
        self.hits = [0 for _ in self.names]
        self.totals = [0 for _ in self.names]
        self.dropped_per_feature = {name: 0 for name in self.names}
        self.n_with_drop = 0
        self.total_dropped = 0
        self.causes = {}

    def add(self, row):
        block = row.get("m4") or {}
        if not block.get("features"):
            return
        self.n_queries += 1
        features = np.asarray(block["features"], dtype=np.float64)
        obs = np.asarray(block["obs_features"], dtype=np.float64).reshape(-1)
        gt_index = int(row["gt_index"])
        dropped = block.get("dropped") or {}
        names_dropped = list(dropped.get("dropped") or [])
        if names_dropped:
            self.n_with_drop += 1
            self.total_dropped += len(names_dropped)
            for name in names_dropped:
                self.dropped_per_feature[name] = self.dropped_per_feature.get(name, 0) + 1
            for name, cause in (dropped.get("causes") or {}).items():
                self.causes.setdefault(name, {"n": 0, "example": cause})["n"] += 1

        for index in range(min(features.shape[-1], len(self.names))):
            column = features[..., index]                       # [M, K]
            if not np.isfinite(column).all():
                continue
            per_candidate = column.mean(axis=-1)
            self.between[index].append(float(np.var(per_candidate)))
            self.within[index].append(float(np.mean(np.var(column, axis=-1))))
            if np.isfinite(obs[index]):
                distances = np.abs(per_candidate - obs[index])
                self.hits[index] += int(int(np.argmin(distances)) == gt_index)
                self.totals[index] += 1

    def result(self):
        per_feature = []
        for index, name in enumerate(self.names):
            between = float(np.mean(self.between[index])) if self.between[index] else float("nan")
            within = float(np.mean(self.within[index])) if self.within[index] else float("nan")
            per_feature.append({
                "feature": name, "between_var": between, "within_var": within,
                "power": (between / within) if within else float("inf"),
                "top1": (self.hits[index] / self.totals[index]
                         if self.totals[index] else float("nan")),
                "n_queries": self.totals[index],
            })
        return {"n_queries": self.n_queries, "per_feature": per_feature,
                "dropped": {"n_queries_with_a_drop": self.n_with_drop,
                            "total_dropped": self.total_dropped,
                            "per_feature": dict(self.dropped_per_feature),
                            "causes": self.causes}}


def sensitivity_summary(rows, aggregation=PRIMARY_AGGREGATION):
    """Plan §3's battery, summarized: what each perturbation did to each family.

    Reported per variant and family: the top-1 under the perturbed distances, the
    rate at which the prediction moved at all, and the mean absolute change of
    the per-candidate score. Only rows that actually carry a battery are counted,
    and the baseline is those same rows -- never the whole split.
    """
    from src.localization.rir_metrics import SENSITIVITY_VARIANTS

    rows = list(rows)
    carriers = [row for row in rows if row.get("sensitivities")]
    summary = {"declared_variants": list(SENSITIVITY_VARIANTS),
               "n_rows": len(rows), "n_rows_with_battery": len(carriers),
               "aggregation": aggregation, "baseline_top1": {}, "variants": {},
               "status": "not computed: no row carries a sensitivity battery"}
    if not carriers:
        return summary

    # only families the battery AND the baseline row both carry can be compared
    baseline_hits, baseline_pred, families = {}, {}, set()
    for row in carriers:
        for block in row["sensitivities"].values():
            families.update(set(block) & set(row["families"]))
    families = sorted(families)

    for family in families:
        hits, preds = [], []
        for row in carriers:
            scores = family_scores(row, family, aggregation)
            pred = int(predict_from_distances(torch.from_numpy(scores)))
            hits.append(1.0 if pred == int(row["gt_index"]) else 0.0)
            preds.append((pred, scores))
        baseline_hits[family] = float(np.mean(hits))
        baseline_pred[family] = preds
    summary["baseline_top1"] = baseline_hits

    variants = {}
    for variant in summary["declared_variants"]:
        block = {}
        for family in families:
            hits, changed, deltas = [], [], []
            for position, row in enumerate(carriers):
                payload = (row["sensitivities"].get(variant) or {}).get(family)
                if payload is None:
                    continue
                matrix = decode_matrix(payload)
                scores = aggregate_over_k(torch.from_numpy(matrix), aggregation).numpy()
                pred = int(predict_from_distances(torch.from_numpy(scores)))
                base_pred, base_scores = baseline_pred[family][position]
                hits.append(1.0 if pred == int(row["gt_index"]) else 0.0)
                changed.append(1.0 if pred != base_pred else 0.0)
                deltas.append(float(np.mean(np.abs(scores - base_scores))))
            if hits:
                block[family] = {"n_queries": len(hits), "top1": float(np.mean(hits)),
                                 "prediction_change_rate": float(np.mean(changed)),
                                 "mean_abs_score_change": float(np.mean(deltas))}
        if block:
            variants[variant] = block
    summary["variants"] = variants
    summary["status"] = "computed"
    return summary


def seen_vs_unseen(seen_summaries, unseen_summaries):
    """The seen-only-scorer detector: the same family on both splits."""
    table = {}
    for family, unseen in sorted(unseen_summaries.items()):
        seen = seen_summaries.get(family)
        entry = {
            "unseen_top1": float(unseen["pooled"]["top1"]),
            "unseen_median_e_loc": float(unseen["pooled"]["median_e_loc"]),
            "unseen_mean_e_loc": float(unseen["pooled"]["mean_e_loc"]),
            "seen_top1": None, "seen_median_e_loc": None, "seen_mean_e_loc": None,
            "top1_gap": None,
            "status": "seen and unseen",
        }
        if seen is None:
            entry["status"] = ("no seen counterpart: the calibration pass did not compute "
                               "this family")
        else:
            entry.update({
                "seen_top1": float(seen["pooled"]["top1"]),
                "seen_median_e_loc": float(seen["pooled"]["median_e_loc"]),
                "seen_mean_e_loc": float(seen["pooled"]["mean_e_loc"]),
                "top1_gap": float(seen["pooled"]["top1"]) - float(unseen["pooled"]["top1"]),
            })
        table[family] = entry
    return table


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
    power = {family: [] for family in families}
    agree, agree_retrieval = [], {"raw": [], "masked": []}
    rooms, n_queries = set(), 0
    m4 = M4Accumulator()
    battery_rows = []

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
            power[family].append(power_by_family(metrics_row, family))
        m4.add(metrics_row)
        if metrics_row.get("sensitivities"):
            # the battery lands on every Nth query only, so keeping those rows
            # costs nothing and lets the summary use the exact same baseline
            battery_rows.append(metrics_row)
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
            "power": {f: v for f, v in power.items() if v},
            "m4": m4.result(),
            "sensitivity": sensitivity_summary(battery_rows or []),
            "agree": agree, "agree_retrieval": agree_retrieval,
            "n_queries": n_queries, "n_rooms": len(rooms),
            "families_present": [f for f in families if records[f][PRIMARY_AGGREGATION]],
            "metrics_path": str(metrics_path), "rows_path": str(rows_path)}
