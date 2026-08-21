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
import os

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


def reciprocal_rank(distances, gt_index, eligible=None):
    """Reciprocal rank of the GT candidate under DISTANCES (lower is better).

    The registered convention (``eval_localization.gt_reciprocal_rank``) ranks
    similarities and breaks ties by lowest index; this is the same rule read for
    distances, and a test pins the two against each other.

    ``eligible`` restricts the ranking to the candidates the prediction was
    restricted to (r4m6 finding 6): a masked control that cannot predict an
    ineligible candidate must not be charged for it out-ranking the GT either.
    """
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    gt_index = int(gt_index)
    if eligible is not None:
        keep = [index for index, flag in enumerate(eligible) if flag]
        if gt_index not in keep:
            raise ValueError(
                f"the GT candidate {gt_index} is not in the eligible set; its rank under the "
                "masked control is undefined (a GT inside its own context aborts upstream)")
        distances = distances[keep]
        gt_index = keep.index(gt_index)
    gt_value = distances[gt_index]
    better = int((distances < gt_value).sum())
    tied_before = int((distances[:gt_index] == gt_value).sum())
    return 1.0 / (better + tied_before + 1)


def family_scores(row, family, aggregation=PRIMARY_AGGREGATION):
    """The recorded per-candidate score ``[M]`` of one family and aggregation.

    RECORDED, never re-derived (r4m6 finding 3): computing an aggregation the
    registered pass did not write would report a number that run never produced.
    """
    block = row["families"][family]
    aggregations = block.get("aggregations") or {}
    if aggregation not in aggregations:
        raise ValueError(
            f"query {row.get('query_id')!r}: family {family!r} records no {aggregation!r} "
            f"aggregation (it has {sorted(aggregations)}); the report reads what the "
            "registered pass wrote and never re-derives it")
    return decode_vector(aggregations[aggregation])


def _geometry(row):
    world = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
    gt = np.asarray(row["gt_xyz_world"], dtype=np.float64)
    return world, gt


def _record(row, pred_index, scores=None, eligible=None):
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
                    (reciprocal_rank(scores, gt_index, eligible=eligible)
                     if scores is not None else 0.0))
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
    record = _record(row, pred, scores=_distance_to_chosen_context(replay_row, best_ctx),
                     eligible=eligible)
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
    record = _record(row, pred, scores=_distance_to_chosen_context(replay_row, best_ctx),
                     eligible=eligible)
    record["best_context"] = best_ctx
    return record


def oracle_record(row, oracle_row, family):
    """Plan §3's measured-candidate oracle ceiling for one family and query.

    The published control stores the compact distances of the candidates whose
    measured RIR exists, and the prediction already mapped back to the original
    candidate order; the GT is ranked among those AVAILABLE candidates, which is
    the set the oracle could choose from.
    """
    block = oracle_row["families"][family]
    available = [bool(flag) for flag in oracle_row["candidate_available"]]
    usable = [index for index, flag in enumerate(available) if flag]
    gt_index = int(row["gt_index"])
    distances = decode_vector(block["oracle_hex"])
    if len(usable) != distances.size:
        raise ValueError(
            f"query {row['query_id']!r}: the oracle stores {distances.size} distances for "
            f"{len(usable)} available candidates")
    record = _record(row, int(block["oracle_pred_index"]))
    record["rr"] = (reciprocal_rank(distances, usable.index(gt_index))
                    if gt_index in usable else 0.0)
    record["gt_available"] = gt_index in usable
    record["n_available"] = len(usable)
    if bool(block.get("oracle_correct")) != bool(record["top1"]):
        raise ValueError(
            f"query {row['query_id']!r}: the oracle control records "
            f"correct={block['oracle_correct']} but its prediction "
            f"{block['oracle_pred_index']} against GT {gt_index} says otherwise")
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

        mask = block.get("mask")
        for index in range(min(features.shape[-1], len(self.names))):
            # a feature the registered validity rule dropped for this query did
            # not take part in the distance and may not be scored (r4m6 F6)
            if mask is not None and not bool(mask[index]):
                continue
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


#: fields that describe the QUERY itself and must therefore agree between the
#: metrics row and the replay row of the same pass (r4m6 finding 3).
BOUND_QUERY_FIELDS = ("room_id", "gt_index", "gt_node", "candidate_nodes",
                      "gt_xyz_world", "candidate_xyz_world", "context_member")


def _bind_query(metrics_row, replay_row, position):
    """Refuse a pair that does not describe the same query and geometry."""
    if str(metrics_row["query_id"]) != str(replay_row["query_id"]):
        raise ValueError(
            f"stream mismatch at position {position}: metrics row is "
            f"{metrics_row['query_id']!r} but the replay row is {replay_row['query_id']!r}")
    for field in BOUND_QUERY_FIELDS:
        if field not in metrics_row or field not in replay_row:
            continue
        if metrics_row[field] != replay_row[field]:
            raise ValueError(
                f"query {metrics_row['query_id']!r} at position {position}: {field} differs "
                f"between the metrics stream ({metrics_row[field]!r}) and the replay stream "
                f"({replay_row[field]!r}); the two are not describing the same query")


def _bind_seed(replay_row, seed, position):
    """Prove the replay row was drawn with the DECLARED seed.

    Every seed scores the same identities, so a query id cannot detect that seed
    42's metrics were paired with seed 43's rows -- but the recorded noise keys
    are a function of (seed, query_id, k) and can (r4m6 finding 3).
    """
    from src.localization.scoring import noise_key

    recorded = replay_row.get("noise_keys")
    if not recorded:
        raise ValueError(
            f"query {replay_row['query_id']!r} at position {position} records no noise_keys; "
            f"the declared seed {seed} cannot be verified against the stream")
    expected = [noise_key(int(seed), str(replay_row["query_id"]), k)
                for k in range(len(recorded))]
    if [int(v) for v in recorded] != expected:
        raise ValueError(
            f"query {replay_row['query_id']!r} at position {position} was NOT drawn with the "
            f"declared seed {seed}: its noise keys belong to another seed's pass")


def _sibling_summary(path):
    """The summary JSON a published stream is a sibling of, or ``None``."""
    for suffix in ("_metrics.jsonl", "_rows.jsonl", ".jsonl"):
        if str(path).endswith(suffix):
            candidate = str(path)[: -len(suffix)] + "_summary.json"
            if os.path.isfile(candidate):
                with open(candidate) as handle:
                    return json.load(handle).get("provenance") or {}
            return None
    return None


#: provenance the two streams of one pass must agree on, and the report records.
BOUND_PROVENANCE_FIELDS = ("seed", "context_stream_digest", "registration_sha",
                           "metric_registration_sha_resolved", "split_hash",
                           "candidate_manifest_sha256")


def bind_provenance(metrics_path, rows_path, seed=None, require=False):
    """Bind the pass's own provenance: seed, context draw and registration."""
    provenances = {"metrics": _sibling_summary(metrics_path),
                   "rows": _sibling_summary(rows_path)}
    present = {name: p for name, p in provenances.items() if p}
    if not present:
        if require:
            raise ValueError(
                f"no sibling summary was found beside {metrics_path!r} or {rows_path!r}; a "
                "published pass carries one, and the report binds seed, context-stream digest "
                "and registration sha to it")
        return {}
    bound = {}
    for field in BOUND_PROVENANCE_FIELDS:
        values = {name: p.get(field) for name, p in present.items() if p.get(field) is not None}
        if len(set(map(str, values.values()))) > 1:
            raise ValueError(
                f"the two streams disagree on {field}: {values}; they are not the metrics and "
                "rows of one pass")
        if values:
            bound[field] = list(values.values())[0]
    if seed is not None and "seed" in bound and int(bound["seed"]) != int(seed):
        raise ValueError(f"the pass declares seed {seed} but its summary records "
                         f"seed {bound['seed']}")
    return bound


def iter_joined(metrics_path, rows_path, seed=None, oracle_path=None):
    """Stream ``(metrics_row, replay_row, oracle_row)`` triples, fail-closed.

    The streams are written by the same loop over the same registered split, so
    a difference in order, length, identity, geometry or seed means they are not
    siblings -- and every downstream number would silently pair the wrong query.
    """
    metrics_stream, rows_stream = iter_rows(metrics_path), iter_rows(rows_path)
    oracle_stream = iter_rows(oracle_path) if oracle_path else None
    position = 0
    while True:
        metrics_row = next(metrics_stream, None)
        replay_row = next(rows_stream, None)
        if metrics_row is None and replay_row is None:
            if oracle_stream is not None and next(oracle_stream, None) is not None:
                raise ValueError(f"{oracle_path!r} is longer than the pass it belongs to")
            return
        if metrics_row is None or replay_row is None:
            raise ValueError(
                f"{metrics_path!r} and {rows_path!r} differ in length at position {position}; "
                "the metrics stream and the rows stream must be siblings of one pass")
        _bind_query(metrics_row, replay_row, position)
        if seed is not None:
            _bind_seed(replay_row, seed, position)
        oracle_row = None
        if oracle_stream is not None:
            oracle_row = next(oracle_stream, None)
            if oracle_row is None:
                raise ValueError(
                    f"{oracle_path!r} ends at position {position} but the pass has more "
                    "queries; the oracle control must cover the whole registered split")
            _bind_query(metrics_row, oracle_row, position)
        yield metrics_row, replay_row, oracle_row
        position += 1


def scan_seed(metrics_path, rows_path, families=REPORT_FAMILIES,
              aggregations=AGGREGATIONS, secondary_aggregation_families=PRIMARY_FAMILIES,
              seed=None, oracle_path=None, expect_queries=None, require_provenance=False):
    """One streaming pass over a seed: every per-query record the report needs.

    Fail-closed (r4m6 finding 3): identities must be unique and, when a count is
    declared, exactly that many; the streams are bound query by query; the
    declared seed is proved against the recorded noise keys; and the sibling
    summaries must agree on seed, context-stream digest and registration sha.

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
    oracle = {family: [] for family in families}
    rooms, identities = set(), set()
    n_queries = 0
    m4 = M4Accumulator()
    battery_rows = []
    provenance = bind_provenance(metrics_path, rows_path, seed=seed,
                                 require=require_provenance)

    for metrics_row, replay_row, oracle_row in iter_joined(metrics_path, rows_path, seed=seed,
                                                           oracle_path=oracle_path):
        identity = str(metrics_row["query_id"])
        if identity in identities:
            raise ValueError(f"duplicate query id {identity!r} in {metrics_path!r}; the "
                             "registered split scores every identity exactly once")
        identities.add(identity)
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
            if oracle_row is not None and family in oracle_row.get("families", {}):
                oracle[family].append(oracle_record(metrics_row, oracle_row, family))
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
    if expect_queries is not None and n_queries != int(expect_queries):
        raise ValueError(
            f"{metrics_path!r} carries {n_queries} queries but the registered split declares "
            f"{int(expect_queries)}; the report covers the whole split or refuses")
    return {"records": {f: b for f, b in records.items() if b[PRIMARY_AGGREGATION]},
            "oracle": {f: v for f, v in oracle.items() if v},
            "provenance": provenance, "seed": None if seed is None else int(seed),
            "retrieval": {f: b for f, b in retrieval.items() if b["raw"]},
            "power": {f: v for f, v in power.items() if v},
            "m4": m4.result(),
            "sensitivity": sensitivity_summary(battery_rows or []),
            "agree": agree, "agree_retrieval": agree_retrieval,
            "n_queries": n_queries, "n_rooms": len(rooms),
            "families_present": [f for f in families if records[f][PRIMARY_AGGREGATION]],
            "metrics_path": str(metrics_path), "rows_path": str(rows_path)}


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def _summary_or_none(records):
    return summarize_records(records) if records else None


def build_seed_report(scan, seed, n_boot=BOOTSTRAP_N, boot_seed=BOOTSTRAP_SEED,
                      primary_families=PRIMARY_FAMILIES):
    """Everything one seed answers: §1 blocks, §2 controls, §3 and the §4 tests.

    The Holm correction covers exactly the registered primary tests -- the five
    families times {vs the AGREE retrieval reference, vs the family's own matched
    control} -- and the declared secondaries are reported beside them, never
    inside the correction.
    """
    families = list(scan["families_present"])
    agree_records = scan["agree"]
    agree_retrieval = {mode: _summary_or_none(records)
                       for mode, records in scan["agree_retrieval"].items()}

    blocks, comparisons, primary_labels = {}, [], []
    for family in families:
        family_records = scan["records"][family]
        primary = family_records[PRIMARY_AGGREGATION]
        power_values = [v for v in scan["power"].get(family, []) if v is not None]
        blocks[family] = {
            "is_primary": family in primary_families,
            "label": "primary" if family in primary_families else "declared-secondary",
            "primary": summarize_records(primary),
            "aggregations": {agg: summarize_records(records)
                             for agg, records in family_records.items()},
            "retrieval": {mode: _summary_or_none(records)
                          for mode, records in scan["retrieval"][family].items()},
            "context_split": context_split(primary),
            "power": {"n_queries": len(power_values),
                      "n_skipped": len(scan["power"].get(family, [])) - len(power_values),
                      "mean": float(np.mean(power_values)) if power_values else None,
                      "median": float(np.median(power_values)) if power_values else None},
        }
        against = (("agree_retrieval", scan["agree_retrieval"]["masked"]),
                   ("matched_retrieval", scan["retrieval"][family]["masked"]))
        for name, reference in against:
            label = f"{family}_vs_{name}"
            comparisons.append(compare_records(primary, reference, label,
                                               n_boot=n_boot, seed=boot_seed))
            if family in primary_families:
                primary_labels.append(label)

    primary_comparisons = [c for c in comparisons if c["label"] in primary_labels]
    # q6 asks whether a family is right where the registered AGREE readout is
    # wrong, so the two correctness columns are kept paired per query
    pairs = {}
    for family in families:
        primary = scan["records"][family][PRIMARY_AGGREGATION]
        pairs[family] = [(float(a["top1"]), float(b["top1"]))
                         for a, b in zip(primary, agree_records)
                         if a["query_id"] == b["query_id"]]
    return {
        "seed": int(seed),
        "_pairs": pairs,
        "n_queries": scan["n_queries"], "n_rooms": scan["n_rooms"],
        "families": blocks,
        "agree": summarize_records(agree_records),
        "agree_context_member_rate": context_split(agree_records)["context_member_rate"],
        "agree_retrieval": agree_retrieval,
        "comparisons": comparisons,
        "primary_comparisons": primary_comparisons,
        "holm": {quantity: holm_over(primary_comparisons, quantity=quantity)
                 for quantity in ("e_loc", "top1")},
        "m4": scan["m4"], "sensitivity": scan["sensitivity"],
        "inputs": {"metrics_path": scan["metrics_path"], "rows_path": scan["rows_path"]},
    }


def _mean_sd(values):
    """Mean and SAMPLE SD (ddof = 1), exp_18's published three-seed convention.

    One value has no sample SD, and saying so is more honest than reporting the
    zero a population SD would print (r4m6 finding 4).
    """
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values)}


#: the seed table's columns: where each number lives inside a seed report.
SEED_TABLE_COLUMNS = (
    ("top1", ("primary", "pooled", "top1")),
    ("median_e_loc", ("primary", "pooled", "median_e_loc")),
    ("mean_e_loc", ("primary", "pooled", "mean_e_loc")),
    ("mrr", ("primary", "pooled", "mrr")),
    ("macro_mean_e_loc", ("primary", "macro", "mean_of_room_means")),
    ("macro_top1", ("primary", "macro", "top1")),
    ("retrieval_masked_top1", ("retrieval", "masked", "pooled", "top1")),
    ("retrieval_masked_macro_top1", ("retrieval", "masked", "macro", "top1")),
    ("context_member_rate", ("context_split", "context_member_rate")),
    ("power_mean", ("power", "mean")),
)


def _dig(block, path):
    for key in path:
        if block is None:
            return None
        block = block.get(key)
    return block


def seed_table(seed_reports, families=REPORT_FAMILIES):
    """Per family: every reported number as per-seed values plus mean +- SD."""
    table = {}
    for family in families:
        entries = [(report["seed"], report["families"][family])
                   for report in seed_reports if family in report["families"]]
        if not entries:
            continue
        columns = {}
        for name, path in SEED_TABLE_COLUMNS:
            per_seed = {str(seed): _dig(block, path) for seed, block in entries}
            columns[name] = dict(_mean_sd(per_seed.values()), per_seed=per_seed)
        columns["label"] = entries[0][1]["label"]
        for column in ("success_0.5", "success_1.0"):
            radius = float(column.split("_")[1])
            per_seed = {str(seed): _dig(block, ("primary", "pooled", "success")).get(radius)
                        for seed, block in entries}
            columns[column] = dict(_mean_sd(per_seed.values()), per_seed=per_seed)
        table[family] = columns
    return table


def _comparison(seed_reports, label):
    return [c for report in seed_reports for c in report["comparisons"] if c["label"] == label]


def conclusions(seed_reports, families=PRIMARY_FAMILIES,
                reference=AGREE_RETRIEVAL_REFERENCE,
                context_reference=AGREE_CONTEXT_MEMBER_RATE,
                seed_tolerance=0.01, room_agreement=0.8, seen_gaps=None, sensitivity=None):
    """The six questions of the R4 directive, computed rather than narrated.

    Every verdict states the rule it applied; the numbers behind it are in the
    same block, so a reader can disagree with a threshold without having to
    recompute anything.
    """
    table = seed_table(seed_reports, families=families)
    answers = {}

    q1 = {}
    macro_reference = _mean_sd([_dig(report, ("agree_retrieval", "masked", "macro", "top1"))
                                for report in seed_reports])
    pooled_reference = _mean_sd([_dig(report, ("agree_retrieval", "masked", "pooled", "top1"))
                                 for report in seed_reports])
    for family, columns in table.items():
        pooled, macro = columns["top1"], columns["macro_top1"]
        paired = _comparison(seed_reports, f"{family}_vs_agree_retrieval")
        q1[family] = {
            # 0.689 is the EQUAL-ROOM MACRO top-1 of the R-1b masked control, so
            # the verdict is macro-to-macro; pooled is reported, never swapped in
            "reference_convention": "macro (equal-room mean of per-room top-1), as R-1b",
            "macro_top1_mean": macro["mean"], "macro_top1_sd": macro["sd"],
            "macro_per_seed": macro["per_seed"],
            "pooled_top1_mean": pooled["mean"], "pooled_top1_sd": pooled["sd"],
            "pooled_per_seed": pooled["per_seed"],
            "reference": float(reference),
            "recomputed_reference_macro_top1": macro_reference["mean"],
            "recomputed_reference_pooled_top1": pooled_reference["mean"],
            "delta_vs_reference": (None if macro["mean"] is None
                                   else macro["mean"] - float(reference)),
            "delta_vs_recomputed_macro": (None if macro["mean"] is None
                                          or macro_reference["mean"] is None
                                          else macro["mean"] - macro_reference["mean"]),
            "delta_vs_recomputed_pooled": (None if pooled["mean"] is None
                                           or pooled_reference["mean"] is None
                                           else pooled["mean"] - pooled_reference["mean"]),
            "paired_top1_p_values": [c["top1"]["p_value"] for c in paired],
            "paired_e_loc_p_values": [c["e_loc"]["p_value"] for c in paired],
            "exceeds": None if macro["mean"] is None else bool(macro["mean"] > float(reference)),
            "exceeds_pooled": (None if pooled["mean"] is None or pooled_reference["mean"] is None
                               else bool(pooled["mean"] > pooled_reference["mean"])),
        }
    answers["q1_exceeds_agree_retrieval"] = dict(
        q1, rule=(f"exceeds := equal-room MACRO top-1 averaged over seeds > the fixed AGREE "
                  f"retrieval reference {reference}, which is itself a macro number (R-1b, "
                  "K_ctx=8; its pooled value is 0.6317). The same control recomputed per query "
                  "on these very rows is reported in both conventions, and exceeds_pooled "
                  "answers the same question pooled."))

    q2 = {}
    for family in table:
        paired = _comparison(seed_reports, f"{family}_vs_matched_retrieval")
        deltas = [c["top1_delta"] for c in paired]
        holm = [next((t for t in report["holm"]["top1"]["tests"]
                      if t["label"] == f"{family}_vs_matched_retrieval"), None)
                for report in seed_reports]
        q2[family] = {
            "top1_delta_mean": float(np.mean(deltas)) if deltas else None,
            "top1_delta_per_seed": {str(r["seed"]): c["top1_delta"]
                                    for r, c in zip(seed_reports, paired)},
            "e_loc_paired_median": [c["e_loc"]["point"] for c in paired],
            "top1_p_values": [c["top1"]["p_value"] for c in paired],
            "holm_adjusted_p": [None if t is None else t["p_adjusted"] for t in holm],
            "holm_rejected": [None if t is None else t["rejected"] for t in holm],
            "beats": (bool(deltas and float(np.mean(deltas)) > 0.0
                           and all(bool(t and t["rejected"]) for t in holm))),
        }
    answers["q2_beats_own_matched_control"] = dict(
        q2, rule=("beats := mean paired top-1 difference over the family's own matched control "
                  "is positive AND the Holm-corrected top-1 test rejects in every seed"))

    q3 = {}
    for family, columns in table.items():
        top1 = columns["top1"]
        per_room = []
        for report in seed_reports:
            block = report["families"].get(family)
            if block is None or not block["retrieval"]["masked"]:
                continue
            control_rooms = block["retrieval"]["masked"]["per_room"]
            for room, stats in block["primary"]["per_room"].items():
                if room in control_rooms:
                    per_room.append(float(stats["top1"]) - float(control_rooms[room]["top1"]))
        wins = float(np.mean([d > 0 for d in per_room])) if per_room else None
        pooled_sign = (None if top1["mean"] is None else
                       float(np.sign((top1["mean"] or 0.0)
                                     - (columns["retrieval_masked_top1"]["mean"] or 0.0))))
        agreeing = (None if not per_room or pooled_sign is None else
                    float(np.mean([float(np.sign(d)) == pooled_sign for d in per_room])))
        q3[family] = {
            "top1_sd_over_seeds": top1["sd"], "top1_per_seed": top1["per_seed"],
            "n_rooms": len(per_room),
            "room_win_rate_vs_matched": wins,
            "room_sign_agreement": agreeing,
            "consistent": (None if top1["sd"] is None or agreeing is None else
                           bool(top1["sd"] <= seed_tolerance and agreeing >= room_agreement)),
        }
    answers["q3_seed_and_room_consistent"] = dict(
        q3, rule=(f"consistent := seed SD of pooled top-1 <= {seed_tolerance} AND at least "
                  f"{room_agreement:.0%} of rooms move in the same direction as the pooled "
                  "difference against the family's matched control"))

    q4 = {}
    for family, columns in table.items():
        rate = columns["context_member_rate"]
        observed = _mean_sd([report["agree_context_member_rate"] for report in seed_reports])
        q4[family] = {
            "context_member_rate": rate["mean"], "per_seed": rate["per_seed"],
            "agree_reference": float(context_reference),
            "agree_rate_on_these_rows": observed["mean"],
            "delta_vs_reference": (None if rate["mean"] is None
                                   else rate["mean"] - float(context_reference)),
            "reduces": (None if rate["mean"] is None
                        else bool(rate["mean"] < float(context_reference))),
        }
    answers["q4_reduces_context_member_failure"] = dict(
        q4, rule=(f"reduces := the family's context-member prediction rate is below AGREE's "
                  f"registered {context_reference}; the rate AGREE shows on these very rows is "
                  "reported beside it"))

    q5 = {}
    # the declared battery is computed on the SEEN calibration pass only, so its
    # summary is passed in; the unseen seed reports carry none by construction
    batteries = ([sensitivity] if sensitivity is not None
                 else [report["sensitivity"] for report in seed_reports])
    for family in table:
        variants, worst = {}, None
        for battery in batteries:
            for variant, block in ((battery or {}).get("variants") or {}).items():
                if family in block:
                    entry = variants.setdefault(variant, {"top1": [], "change": []})
                    entry["top1"].append(block[family]["top1"])
                    entry["change"].append(block[family]["prediction_change_rate"])
        for variant, entry in variants.items():
            change = float(np.mean(entry["change"]))
            if worst is None or change > worst[1]:
                worst = (variant, change)
        seen_gap = [(seen_gaps or {}).get(family)]
        m4_drop = [report["m4"]["dropped"]["n_queries_with_a_drop"] / max(1, report["m4"]
                                                                         ["n_queries"])
                   for report in seed_reports if report["m4"]["n_queries"]]
        q5[family] = {
            "sensitivity": {variant: {"top1_mean": float(np.mean(entry["top1"])),
                                      "prediction_change_rate":
                                          float(np.mean(entry["change"]))}
                            for variant, entry in sorted(variants.items())},
            "worst_variant": None if worst is None else worst[0],
            "worst_prediction_change_rate": None if worst is None else worst[1],
            "n_battery_rows": ((sensitivity or {}).get("n_rows_with_battery")
                               if sensitivity is not None else None),
            "m4_query_drop_rate": float(np.mean(m4_drop)) if m4_drop else None,
            "seen_unseen_top1_gap": _mean_sd([g for g in seen_gap if g is not None])["mean"],
            "caveat": (None if worst is None else
                       bool(worst[1] > 0.0)),
        }
    answers["q5_robustness_caveats"] = dict(
        q5, rule=("caveat := at least one declared seen-battery perturbation moves the "
                  "prediction on some query; the per-variant rates, the M4 drop rate and the "
                  "seen-unseen gap are the evidence"))

    q6 = {}
    for family in table:
        contingency = {"both_right": 0, "family_right_agree_wrong": 0,
                       "family_wrong_agree_right": 0, "both_wrong": 0}
        union = 0
        total = 0
        for report in seed_reports:
            pairs = report.get("_pairs", {}).get(family)
            if not pairs:
                continue
            for family_hit, agree_hit in pairs:
                total += 1
                union += int(bool(family_hit) or bool(agree_hit))
                if family_hit and agree_hit:
                    contingency["both_right"] += 1
                elif family_hit:
                    contingency["family_right_agree_wrong"] += 1
                elif agree_hit:
                    contingency["family_wrong_agree_right"] += 1
                else:
                    contingency["both_wrong"] += 1
        agree_wrong = (contingency["family_right_agree_wrong"] + contingency["both_wrong"])
        agree_right = (contingency["family_wrong_agree_right"] + contingency["both_right"])
        q6[family] = {
            "n_paired": total,
            "contingency": contingency,
            "agreement_rate": (None if not total else
                               (contingency["both_right"] + contingency["both_wrong"]) / total),
            "rescue_rate": (None if not agree_wrong else
                            contingency["family_right_agree_wrong"] / agree_wrong),
            "loss_rate": (None if not agree_right else
                          contingency["family_wrong_agree_right"] / agree_right),
            "union_top1": None if not total else union / total,
            "agree_top1": (None if not total else agree_right / total),
            "adds_information": (None if not total else
                                 bool(contingency["family_right_agree_wrong"] > 0
                                      and union / total > agree_right / total)),
        }
    answers["q6_adds_information_vs_different_scorer"] = dict(
        q6, rule=("adds_information := the family is right on queries where the registered "
                  "AGREE readout is wrong, so their union beats AGREE alone; the full 2x2 "
                  "contingency and the agreement rate are reported, not just the verdict"))
    return answers


def file_sha256(path, chunk=1 << 20):
    """sha256 of a published artifact, streamed (these files are ~350 MB)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(seed_reports, families=REPORT_FAMILIES, seen_scan=None,
                 n_boot=BOOTSTRAP_N, hash_inputs=False, extra_provenance=None):
    """The whole R4 answer: seed tables, controls, comparisons and conclusions."""
    from datetime import datetime, timezone

    families = tuple(families)
    seen_summaries = {}
    seen_block = None
    if seen_scan is not None:
        seen_summaries = {family: summarize_records(block[PRIMARY_AGGREGATION])
                          for family, block in seen_scan["records"].items()}
        seen_block = {
            "n_queries": seen_scan["n_queries"], "n_rooms": seen_scan["n_rooms"],
            "families": {f: s for f, s in sorted(seen_summaries.items())},
            "m4": seen_scan["m4"], "sensitivity": seen_scan["sensitivity"],
            "inputs": {"metrics_path": seen_scan["metrics_path"],
                       "rows_path": seen_scan["rows_path"]},
        }
    unseen_summaries = {}
    for family in families:
        blocks = [r["families"][family]["primary"] for r in seed_reports
                  if family in r["families"]]
        if blocks:
            unseen_summaries[family] = blocks[0]
    split_table = seen_vs_unseen(seen_summaries, unseen_summaries) if seen_scan else {}

    answers = conclusions(
        seed_reports,
        families=tuple(f for f in PRIMARY_FAMILIES
                       if any(f in r["families"] for r in seed_reports)),
        seen_gaps={f: entry.get("top1_gap") for f, entry in split_table.items()},
        sensitivity=(seen_block or {}).get("sensitivity"))

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap": {"n": int(n_boot), "seed": int(BOOTSTRAP_SEED), "alpha": ALPHA,
                      "clustered_by": "room_id"},
        "aggregation": {"primary": PRIMARY_AGGREGATION,
                        "declared_secondaries": list(SECONDARY_AGGREGATIONS)},
        "references": {"agree_retrieval_top1": AGREE_RETRIEVAL_REFERENCE,
                       "agree_context_member_rate": AGREE_CONTEXT_MEMBER_RATE},
        "seeds": [r["seed"] for r in seed_reports],
        "inputs": [r["inputs"] for r in seed_reports],
        "status": "PRELIMINARY -- pending review",
    }
    if seen_scan is not None:
        provenance["seen_inputs"] = seen_block["inputs"]
    if hash_inputs:
        paths = [p for entry in provenance["inputs"] for p in entry.values()]
        if seen_scan is not None:
            paths += list(provenance["seen_inputs"].values())
        provenance["input_sha256"] = {path: file_sha256(path) for path in sorted(set(paths))}
    if extra_provenance:
        provenance.update(dict(extra_provenance))

    return {
        "mode": "metrics-report",
        "families": {"primary": list(PRIMARY_FAMILIES),
                     "declared_secondary": list(SECONDARY_FAMILIES)},
        "seed_table": seed_table(seed_reports, families=families),
        "seeds": [{k: v for k, v in report.items() if k != "_pairs"}
                  for report in seed_reports],
        "seen": seen_block,
        "seen_vs_unseen": split_table,
        "conclusions": answers,
        "provenance": provenance,
    }


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report):
    """The compact `_results.md` block: the tables a reader needs, nothing else."""
    lines = ["### exp_18 R4 -- non-AGREE metric families (PRELIMINARY, pending review)", ""]
    provenance = report["provenance"]
    lines.append(f"Seeds {provenance['seeds']}; K aggregation "
                 f"{provenance['aggregation']['primary']} (primary); "
                 f"{provenance['bootstrap']['n']} room-clustered bootstrap resamples; "
                 f"fixed AGREE retrieval reference "
                 f"{provenance['references']['agree_retrieval_top1']}.")
    lines.append("")

    lines.append("| family | kind | pooled top-1 (mean +- SD) | macro top-1 | median e_loc | "
                 "mean e_loc | s@0.5 | s@1.0 | MRR | matched control (pooled) | "
                 "matched control (macro) | ctx-member rate |")
    lines.append("|" + "---|" * 12)
    for family, columns in report["seed_table"].items():
        lines.append("| {} | {} | {} +- {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            family, columns["label"],
            _fmt(columns["top1"]["mean"]), _fmt(columns["top1"]["sd"]),
            _fmt(columns["macro_top1"]["mean"]),
            _fmt(columns["median_e_loc"]["mean"]), _fmt(columns["mean_e_loc"]["mean"]),
            _fmt(columns["success_0.5"]["mean"]), _fmt(columns["success_1.0"]["mean"]),
            _fmt(columns["mrr"]["mean"]),
            _fmt(columns["retrieval_masked_top1"]["mean"]),
            _fmt(columns["retrieval_masked_macro_top1"]["mean"]),
            _fmt(columns["context_member_rate"]["mean"])))
    lines.append("")

    lines.append("| test (primary, Holm over the 10) | top-1 delta | median e_loc delta | "
                 "p (top-1) | p_adj (top-1) | rejected |")
    lines.append("|---|---|---|---|---|---|")
    first = report["seeds"][0] if report["seeds"] else None
    if first:
        adjusted = {t["label"]: t for t in first["holm"]["top1"]["tests"]}
        for comparison in first["primary_comparisons"]:
            test = adjusted.get(comparison["label"], {})
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                comparison["label"], _fmt(comparison["top1_delta"]),
                _fmt(comparison["e_loc"]["point"]), _fmt(comparison["top1"]["p_value"]),
                _fmt(test.get("p_adjusted")), _fmt(test.get("rejected"))))
        lines.append("")
        lines.append(f"(seed {first['seed']} shown; every seed is in the report JSON)")
    lines.append("")

    lines.append("| question | family | answer | evidence |")
    lines.append("|---|---|---|---|")
    verdicts = (("q1_exceeds_agree_retrieval", "exceeds", "delta_vs_reference"),
                ("q2_beats_own_matched_control", "beats", "top1_delta_mean"),
                ("q3_seed_and_room_consistent", "consistent", "room_sign_agreement"),
                ("q4_reduces_context_member_failure", "reduces", "context_member_rate"),
                ("q5_robustness_caveats", "caveat", "worst_prediction_change_rate"),
                ("q6_adds_information_vs_different_scorer", "adds_information", "union_top1"))
    for key, verdict_key, evidence_key in verdicts:
        block = report["conclusions"].get(key, {})
        for family, entry in sorted(block.items()):
            if family == "rule":
                continue
            lines.append("| {} | {} | {} | {} = {} |".format(
                key.split("_")[0], family, _fmt(entry.get(verdict_key)), evidence_key,
                _fmt(entry.get(evidence_key))))
    lines.append("")
    for key in [k for k, _v, _e in verdicts]:
        rule = report["conclusions"].get(key, {}).get("rule")
        if rule:
            lines.append(f"- **{key.split('_')[0]} rule** -- {rule}")
    return "\n".join(lines) + "\n"
