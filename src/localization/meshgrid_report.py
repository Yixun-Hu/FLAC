"""exp_22 R1 -- the mesh-grid localization report (inherited plan §2).

The I1 engine publishes one authenticated row per query and one float16
similarity sidecar beside it. This module turns that artifact set into the
numbers §2 registers -- and refuses to turn anything else into them.

Nothing is computed until the artifacts have proven they are the registered
ones. In order: the published run binding must hash to its own content, the G1
audit chain must re-verify, the D1 manifest must re-verify its stream hashes and
its census, every row must match its own ``row_sha256`` and its sidecar's
``sims_sha256``, and the row set must be EXACTLY the registered census -- the
5,337 queries of the 16 mesh-available rooms, once each, at their registered
stream positions, summing to the registered candidate-query-pair and generated-
waveform totals. Every one of those is a fail-closed ``ValueError`` naming what
moved.

Ground truth is read HERE and nowhere upstream. The engine is structurally
unable to read ``md['source']`` (:class:`~src.localization.meshgrid_engine.
GuardedMetadata`) and the G1 manifests publish only the oracle DISTANCE, never
``x*_s``; reporting is post hoc, so it resolves the continuous truth from the
dataset's own pair metadata -- the same seam the G1 audit used
(``candidates.find_pair_metadata``) -- and cross-checks the oracle it re-derives
against the value G1 published. A disagreement there means the report is not
looking at the same query the engine scored, and it refuses.

Three readouts are kept apart on purpose:

* the **headline** is the plan's log-mean-exp score
  ``S = tau * (logsumexp(s / tau) - log K)`` at ``tau = 0.1``;
* ``S_mean`` is published beside it as a **declared diagnostic** (§2), never as
  a headline;
* the float16 sidecar is a **diagnostic precision** (the engine's
  ``SIMS_PRECISION_CAVEAT``): the report recomputes both aggregators from it and
  cross-checks the argmax against the row, but every published number is taken
  from the row's float32 ``scores_hex``.

Aggregation is room-first (§2): a statistic is computed inside each of the 16
rooms and the 16 room values are then averaged, with a 95% percentile interval
from a room bootstrap (10,000 resamples, pre-registered seed 20260825). Pooled
per-query values are also reported, labelled as a secondary.
"""
import argparse
import hashlib
import json
import math
import os

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_geometry as mg
from src.localization import meshgrid_queries as mq
from src.localization import scoring as sc
from src.localization.reaggregate import decode_scores

# --------------------------------------------------------------------------- #
# the registered constants of §2
# --------------------------------------------------------------------------- #
#: the pre-registered room-bootstrap settings (§2 "95% room-bootstrap CIs").
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_N = 10000
BOOTSTRAP_ALPHA = 0.05

#: the registered success radii, in metres (§2).
SUCCESS_RADII = (0.5, 1.0)

#: the pre-registered uniform-random-candidate baseline seeds (§2 "repeated with
#: pre-registered seeds"). Each is ONE independent full repetition.
RANDOM_BASELINE_SEEDS = (101, 102, 103, 104, 105)

#: the two aggregators §2 asks for side by side. ``lme`` is the PDF-controlled
#: headline; ``mean`` is the declared diagnostic and never replaces it.
AGGREGATORS = ("lme", "mean")
HEADLINE_AGGREGATOR = "lme"
#: the cell the visualization quantiles are selected on (§2 + the r9 dispatch).
HEADLINE_K = 8

#: the oracle-coverage threshold §2 reports the fraction above.
ORACLE_THRESHOLD = mg.ORACLE_THRESHOLD

#: the scope label every table carries.
SUBSET_LABEL = ("mesh-available preflight subset (5,337/16 rooms; canonical-heading "
                "diagnostic only)")

#: the heatmap softmax temperature: VISUALIZATION ONLY, uncalibrated, and unable
#: to affect any prediction (inherited plan §1.4).
VISUALIZATION_T = 0.1
VISUALIZATION_T_LABEL = ("uncalibrated visualization softmax temperature T = 0.1; it scales a "
                         "score map for display and cannot affect any predicted candidate "
                         "(inherited plan §1.4)")

#: how the pre-registered visualization cases are chosen, verbatim.
VISUALIZATION_RULE = (
    "pre-registered quantile selection, computed AFTER every query is scored and a pure "
    "function of the results: the lowest-e_loc (sharp), the median-e_loc (ambiguous) and the "
    "highest-e_loc (failure) query of the headline cell (log-mean-exp, K = 8). Queries are "
    "ordered by (e_loc, global stream position), so ties are broken deterministically by the "
    "smaller position; the median is the lower median at index (n - 1) // 2. Nothing is "
    "hand-picked")

#: role of each aggregator, stamped into the report so a table cannot be read as
#: promoting the diagnostic.
AGGREGATOR_ROLES = {
    "lme": "HEADLINE -- the PDF Eq. (3) score S = tau * (logsumexp(s / tau) - log K), tau = 0.1",
    "mean": "DECLARED DIAGNOSTIC -- S_mean = mean_k s[x, k] (§2 'diagnostics only'); reported "
            "beside the headline and never in place of it",
}

#: the re-derived oracle must equal the one G1 published to this many metres.
#: Both sides are float64 minima over the SAME coordinate array, so the only
#: admissible difference is summation order.
ORACLE_TOLERANCE = 1e-9
#: the metadata receiver must be the manifest receiver to this many metres.
RECEIVER_TOLERANCE = 1e-6

#: What to do when the float16 sidecar's recomputed argmax differs from the
#: row's float32 one.
#:
#: ``"explained"`` (registered default) applies the engine's own argmax-stability
#: rule (:data:`~src.localization.meshgrid_engine.ARGMAX_STABILITY_FACTOR`): a
#: per-score bound ``eps`` moves a top-1 GAP by up to ``2 eps``, so a
#: disagreement is admitted only when the row's own top-1 margin is within twice
#: the measured sidecar deviation, and is then counted, named and published.
#: ``"strict"`` refuses every disagreement, including the ones the declared
#: float16 precision explains. Either way the published number always comes from
#: the row's float32 score, never from the sidecar.
#:
#: Be precise about what "explained" can and cannot catch. If two score vectors
#: over the SAME candidates disagree about the argmax, then the leader must have
#: lost and the runner-up gained at least the gap between them, so the larger of
#: the two moves is at least ``margin / 2`` -- i.e. ``margin <= 2 * deviation``
#: holds for EVERY possible flip, by arithmetic. The "explained" branch is
#: therefore a deliberate strictness setting, not a corruption detector; the
#: corruption detectors are the row digest, the sidecar digest and the exact
#: checks in :func:`evaluate_query` (which also pin the row's recorded ``margin``
#: to its own scores, so a flip cannot be excused by an inflated margin).
SIDECAR_ARGMAX_POLICIES = ("explained", "strict")
SIDECAR_ARGMAX_POLICY = "explained"
SIDECAR_ARGMAX_NOTE = (
    "per-sample similarities are published as float16 (the engine's SIMS_PRECISION_CAVEAT), so "
    "an aggregate recomputed from the sidecar differs from the row's float32 aggregate by about "
    "one float16 ulp. Under the engine's own stability rule a per-score bound eps can move a "
    "top-1 gap by 2 eps, so a recomputed argmax may legitimately differ from the row's exactly "
    "when the row's margin is <= 2x the measured sidecar deviation -- which, for two score "
    "vectors over the same candidates, is every flip there can be. Those cases are COUNTED and "
    "NAMED here; every published number is the row's float32 value, never the sidecar's, and "
    "the row's own recorded margin is pinned to its own scores so it cannot excuse a flip")

#: §2 controls this report does NOT contain, named so a reader cannot mistake
#: its silence for a null result. Both are published by their own tools.
CONTROLS_ELSEWHERE = {
    "off_grid_truth_probe": "src/localization/meshgrid_offgrid_probe.py -- generates at the "
                            "continuous truth on the sixteen registered probe queries and "
                            "reports its score/rank against that query's grid candidates",
    "real_vs_generated_agree_calibration":
        "src/localization/meshgrid_offgrid_probe.py -- cos(E(h_obs), E(h_real,other)) against "
        "cos(E(h_obs), E(h_generated)) on the same sixteen queries",
    "agree_oracle_retrieval_over_the_metadata_bank":
        "NOT IMPLEMENTED in the r9 deliverables. §2 also registers an AGREE oracle retrieval "
        "over real candidate-bank RIRs where an exact dataset RIR exists, labelled "
        "sparse/metadata-bank and never confused with the dense-grid model oracle. It needs a "
        "real-RIR bank this reporting path does not build; it is outstanding, not null",
    "score_ablations": "deferred by §2 unless separately approved (waveform / multiscale STFT)",
}

#: the report's own artifact names.
REPORT_JSON = "meshgrid_r1_report.json"
REPORT_MARKDOWN = "meshgrid_r1_report.md"
CASES_JSON = "meshgrid_r1_visualization_cases.json"


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def radius_key(radius):
    """The JSON key a success radius is published under."""
    return str(float(radius))


def _finite(value, what):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN or Inf), got {number}")
    return number


def jsonable(value):
    """Recursively coerce numpy scalars/arrays so ``json.dumps`` cannot fail."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


# --------------------------------------------------------------------------- #
# the gates: nothing is computed until the artifacts prove what they are
# --------------------------------------------------------------------------- #
def load_published_binding(run_dir):
    """The run's published binding, RECOMPUTED from its own content.

    A stored digest is what a tampered directory would keep saying, so it is
    evidence of nothing on its own (the merge's gate 7 makes the same argument,
    ``meshgrid_engine._read_shard``).
    """
    path = os.path.join(str(run_dir), me.BINDING_FILENAME)
    if not os.path.isfile(path):
        raise ValueError(f"{run_dir} publishes no {me.BINDING_FILENAME}; a report may not "
                         "describe artifacts whose provenance is unknown")
    with open(path) as handle:
        published = json.load(handle)
    missing = [field for field in me.RUN_BINDING_FIELDS if field not in published]
    if missing:
        raise ValueError(f"{path} is missing the registered binding fields {missing}; the "
                         "binding cannot be recomputed, so it cannot be trusted")
    recomputed = me.binding_sha256({field: published[field]
                                    for field in me.RUN_BINDING_FIELDS})
    if published.get("binding_sha256") != recomputed:
        raise ValueError(f"{path} does not match its own content: it stores "
                         f"{str(published.get('binding_sha256'))[:12]}... but hashes to "
                         f"{recomputed[:12]}...; it was edited after publication")
    return published, recomputed


def iter_row_paths(run_dir):
    """Every published row file of a run, in room / position order."""
    root = os.path.join(str(run_dir), me.ROWS_DIRNAME)
    if not os.path.isdir(root):
        raise ValueError(f"{run_dir} has no {me.ROWS_DIRNAME}/ directory; there are no rows to "
                         "report on")
    for room in sorted(os.listdir(root)):
        room_dir = os.path.join(root, room)
        if not os.path.isdir(room_dir):
            continue
        for name in sorted(os.listdir(room_dir)):
            if name.endswith(".json"):
                yield os.path.join(room_dir, name)


def verify_rows(run_dir, binding_sha256):
    """Re-accept every row from its own bytes -- row digest AND sidecar digest.

    Returns the rows in ascending stream position. A single rejection refuses
    the whole report: a partial artifact set cannot be aggregated into a census-
    complete number.
    """
    rows, rejected = [], []
    for path in iter_row_paths(run_dir):
        verdict = me.verify_query_artifact(path, binding_sha256=binding_sha256)
        if not verdict["ok"]:
            rejected.append({"row": path, "reason": verdict["reason"],
                             "query_id": verdict.get("query_id")})
            continue
        with open(path) as handle:
            row = json.load(handle)
        row["_row_path"] = path
        rows.append(row)
    if rejected:
        raise ValueError(
            f"{len(rejected)} published row(s) do not re-verify and the report refuses to "
            f"aggregate a partial artifact set; first {rejected[:3]}")
    if not rows:
        raise ValueError(f"{run_dir} publishes no rows")
    return sorted(rows, key=lambda row: int(row["position"]))


def assert_census(rows, records, totals=None):
    """The row set IS the registered subset -- exactly, once each, in place.

    Checks, in order and each with its own message: no duplicate query, no query
    outside the registered subset, no registered query missing, every row at the
    position and in the room the D1 manifest registers, the registered room and
    query counts, and the registered candidate-query-pair / generated-waveform
    totals the G1 cost gate published.
    """
    totals = dict(me.REGISTERED_TOTALS if totals is None else totals)
    expected = {record["query_id"]: (int(record["position"]), str(record["room_id"]))
                for record in records}
    if len(expected) != len(records):
        raise ValueError("the context manifest itself carries duplicate query ids; the census "
                         "cannot be taken against it")

    seen = {}
    duplicates = []
    for row in rows:
        query_id = row["query_id"]
        if query_id in seen:
            duplicates.append(query_id)
        seen[query_id] = row
    if duplicates:
        raise ValueError(f"{len(duplicates)} query id(s) are published more than once "
                         f"(first {sorted(duplicates)[:3]}); a report scores each query once")

    extra = sorted(set(seen) - set(expected))
    if extra:
        raise ValueError(f"{len(extra)} published row(s) are not in the registered subset "
                         f"(first {extra[:3]}); the subset is the {mq.FILTERED_COUNT}-query "
                         f"mesh-available preflight set with {mq.EXCLUDED_ROOM!r} excluded")
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} registered queries have no published row "
                         f"(first {missing[:3]}); a report covers the complete subset or "
                         "refuses")

    for query_id in sorted(seen):
        row = seen[query_id]
        position, room_id = expected[query_id]
        if int(row["position"]) != position or str(row["room_id"]) != room_id:
            raise ValueError(f"{query_id} is published at position {row['position']} in "
                             f"{row['room_id']!r} but the context manifest registers position "
                             f"{position} in {room_id!r}")

    rooms = sorted({str(row["room_id"]) for row in rows})
    per_room = {room: 0 for room in rooms}
    for row in rows:
        per_room[str(row["room_id"])] += 1
    expected_per_room = {}
    for record in records:
        expected_per_room[str(record["room_id"])] = \
            expected_per_room.get(str(record["room_id"]), 0) + 1
    if per_room != expected_per_room:
        differing = sorted(room for room in set(per_room) | set(expected_per_room)
                           if per_room.get(room) != expected_per_room.get(room))
        raise ValueError(f"the per-room query counts differ from the context manifest on "
                         f"{differing[:3]}; the census is per room as well as in total")

    if totals.get("queries") is not None and len(rows) != int(totals["queries"]):
        raise ValueError(f"the run publishes {len(rows):,} queries but the registered census is "
                         f"{int(totals['queries']):,}")
    if totals.get("rooms") is not None and len(rooms) != int(totals["rooms"]):
        raise ValueError(f"the run publishes {len(rooms)} rooms but the registered census is "
                         f"{int(totals['rooms'])}")

    pairs = sum(int(row["n_candidates"]) for row in rows)
    waveforms = sum(int(row["n_candidates"]) * int(row["num_samples"]) for row in rows)
    for name, found, wanted in (("candidate_query_pairs", pairs,
                                 totals.get("candidate_query_pairs")),
                                ("generated_waveforms", waveforms,
                                 totals.get("generated_waveforms"))):
        if wanted is not None and int(found) != int(wanted):
            raise ValueError(f"the census fails on {name}: the rows account for {found:,} but "
                             f"the registered total is {int(wanted):,}")
    return {"n_queries": len(rows), "n_rooms": len(rooms), "rooms": rooms,
            "queries_per_room": per_room, "candidate_query_pairs": int(pairs),
            "generated_waveforms": int(waveforms),
            "registered_totals": {key: int(value) for key, value in totals.items()
                                  if value is not None},
            "excluded_room": mq.EXCLUDED_ROOM, "n_excluded": mq.EXCLUDED_COUNT}


def assert_row_protocol(rows, binding):
    """Every row was produced under the protocol the binding pins.

    The binding authenticates the RUN; this authenticates that each row inside it
    actually carries the registered tau, seed, sample count, prefixes and noise
    policy, so a cell cannot be aggregated across two protocols.
    """
    wanted = {"tau": float(binding["tau"]), "seed": int(binding["seed"]),
              "num_samples": int(binding["num_samples"]),
              "noise_policy": str(binding["noise_policy"]),
              "k_prefixes": [int(k) for k in binding["k_prefixes"]],
              "branch": str(binding["branch"]),
              "scorer_readout": str(binding["scorer_readout"])}
    for row in rows:
        found = {"tau": float(row["tau"]), "seed": int(row["seed"]),
                 "num_samples": int(row["num_samples"]),
                 "noise_policy": str(row["noise_policy"]),
                 "k_prefixes": [int(k) for k in row["k_prefixes"]],
                 "branch": str(row["branch"]),
                 "scorer_readout": str(row.get("scorer_readout"))}
        differing = sorted(key for key in wanted if wanted[key] != found[key])
        if differing:
            raise ValueError(f"{row['query_id']}: the row was produced under "
                             f"{ {key: found[key] for key in differing} } but the run binding "
                             f"pins { {key: wanted[key] for key in differing} }")
        if sorted(int(k) for k in row["by_k"]) != sorted(wanted["k_prefixes"]):
            raise ValueError(f"{row['query_id']}: the row publishes prefixes "
                             f"{sorted(row['by_k'])} for the binding's {wanted['k_prefixes']}")
    return wanted


# --------------------------------------------------------------------------- #
# ground truth: resolved here, post hoc, from the dataset's own metadata
# --------------------------------------------------------------------------- #
class TruthResolver:
    """``query_id -> (receiver_global, source_global)`` from the pair metadata.

    The same seam the G1 audit used (``audit_meshgrid_geometry._metadata_for``):
    the pair file is found by PARSED NUMERIC IDENTITY over the directory listing,
    never by reconstructing a name, because the release writes receiver 19 as
    ``S007_R0019.json`` and receiver 8 as ``S007_R008.json``.

    Reading ``src_loc`` is legitimate HERE and only here: the engine is
    structurally forbidden from it (``GuardedMetadata``) and the G1 manifests
    publish only the oracle distance, so the report is the first and only place
    the continuous truth enters -- after every prediction is already frozen.
    """

    def __init__(self, metadata_root):
        self.metadata_root = str(metadata_root)
        self._cache = {}

    def _pair_path(self, room_id, src_node, rec_node):
        from src.localization.candidates import find_pair_metadata

        key = (room_id, int(src_node), int(rec_node))
        if key not in self._cache:
            scene, scene_id = room_id.split("/")
            room_dir = os.path.join(self.metadata_root, scene, scene_id)
            self._cache[key] = find_pair_metadata(room_dir, int(src_node), int(rec_node))
        return self._cache[key]

    def resolve(self, record):
        from src.localization.candidates import parse_ir_filename

        room_id = str(record["room_id"])
        relpath = record.get("relpath") or record.get("path")
        if not relpath:
            raise ValueError(f"{record.get('query_id')!r}: the context manifest record carries "
                             "no relpath, so its pair metadata cannot be resolved")
        src_node, rec_node = parse_ir_filename(os.path.basename(str(relpath)))
        path = self._pair_path(room_id, src_node, rec_node)
        if path is None:
            scene, scene_id = room_id.split("/")
            raise ValueError(f"{record['query_id']}: no pair metadata for (S{src_node}, "
                             f"R{rec_node}) under "
                             f"{os.path.join(self.metadata_root, scene, scene_id)!r}")
        with open(path) as handle:
            payload = json.load(handle)
        receiver = np.asarray(payload["rec_loc"], dtype=np.float64).reshape(3)
        source = np.asarray(payload["src_loc"], dtype=np.float64).reshape(3)
        if not (np.isfinite(receiver).all() and np.isfinite(source).all()):
            raise ValueError(f"{record['query_id']}: {path} carries a non-finite coordinate")
        return receiver, source


def assert_receiver_matches(query_id, metadata_receiver, manifest_receiver,
                            tolerance=RECEIVER_TOLERANCE):
    """The metadata pair file must describe the query the manifest describes."""
    drift = float(np.abs(np.asarray(metadata_receiver, dtype=np.float64)
                         - np.asarray(manifest_receiver, dtype=np.float64)).max())
    if drift > tolerance:
        raise ValueError(
            f"{query_id}: the pair metadata's receiver "
            f"{np.asarray(metadata_receiver).tolist()} is {drift:.6g} m from the candidate "
            f"manifest's {np.asarray(manifest_receiver).tolist()}; the report is not resolving "
            "the same query the engine scored")
    return drift


# --------------------------------------------------------------------------- #
# one query
# --------------------------------------------------------------------------- #
def _stored_scores(block, aggregator):
    key = "scores_hex" if aggregator == "lme" else "mean_scores_hex"
    return decode_scores(block[key])


def _stored_prediction(block, aggregator):
    if aggregator == "lme":
        return int(block["prediction_row"]), int(block["prediction_index"]), \
            block["prediction_xyz"]
    return int(block["mean_prediction_row"]), int(block["mean_prediction_index"]), \
        block["mean_prediction_xyz"]


def evaluate_query(row, sims, coordinates, truth, *, tau=None, radii=SUCCESS_RADII,
                   oracle_tolerance=ORACLE_TOLERANCE,
                   sidecar_argmax_policy=SIDECAR_ARGMAX_POLICY):
    """Every §2 readout for one query, from artifacts that must agree with each other.

    ``coordinates`` is the query's candidate coordinate block ``[M, 3]``, taken
    from the G1 npz (never from the row), and ``truth`` is the continuous source
    position from the pair metadata.

    Three cross-checks run before any number is kept:

    1. the row's own float32 score vector must reproduce the row's argmax under
       the registered tie-break (``argmax_by_global_index``), the candidate
       coordinate it names and the top-1 margin it records -- all exact, and none
       of them may ever fail;
    2. the aggregators recomputed from the float16 sidecar must agree with the
       row, with a disagreement admitted only when the declared sidecar precision
       explains it (see :data:`SIDECAR_ARGMAX_NOTE`);
    3. the oracle re-derived from the candidate block and the metadata truth must
       equal the oracle G1 published in the row.
    """
    if sidecar_argmax_policy not in SIDECAR_ARGMAX_POLICIES:
        raise ValueError(f"unknown sidecar_argmax_policy {sidecar_argmax_policy!r} "
                         f"(expected one of {list(SIDECAR_ARGMAX_POLICIES)})")
    tau = float(row["tau"]) if tau is None else float(tau)
    indices = [int(i) for i in row["candidate_indices"]]
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(indices), 3):
        raise ValueError(f"{row['query_id']}: {coordinates.shape[0]} candidate coordinates for "
                         f"{len(indices)} published candidate indices")
    truth = np.asarray(truth, dtype=np.float64).reshape(3)

    sims_t = torch.as_tensor(np.asarray(sims, dtype=np.float32))
    if tuple(sims_t.shape) != (len(indices), int(row["num_samples"])):
        raise ValueError(f"{row['query_id']}: the sidecar is {tuple(sims_t.shape)} but the row "
                         f"declares ({len(indices)}, {row['num_samples']})")
    prefixes = tuple(int(k) for k in row["k_prefixes"])
    recomputed = me.nested_scores(sims_t, tau=tau, prefixes=prefixes)

    # (3) the oracle, re-derived from the candidate block and the metadata truth
    e_oracle = float(np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1).min())
    published_oracle = float(row["e_oracle"])
    oracle_delta = abs(e_oracle - published_oracle)
    if oracle_delta > float(oracle_tolerance):
        raise ValueError(
            f"{row['query_id']}: the oracle re-derived from the candidate block and the pair "
            f"metadata is {e_oracle:.9f} m but the G1 manifest published {published_oracle:.9f} "
            f"m (|delta| = {oracle_delta:.3g} > {oracle_tolerance:g}); the report's ground "
            "truth is not the one the audit measured")
    oracle_row = int(np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1).argmin())

    out = {"query_id": row["query_id"], "room_id": row["room_id"],
           "position": int(row["position"]), "receiver_id": row.get("receiver_id"),
           "n_candidates": len(indices), "num_samples": int(row["num_samples"]),
           "e_oracle": e_oracle, "e_oracle_published": published_oracle,
           "e_oracle_delta": oracle_delta,
           "oracle_candidate_index": int(indices[oracle_row]),
           "oracle_candidate_xyz": coordinates[oracle_row].tolist(),
           "truth_xyz": truth.tolist(),
           "by": {aggregator: {} for aggregator in AGGREGATORS},
           "sidecar": {aggregator: {} for aggregator in AGGREGATORS},
           "latency_s": {name: float(value)
                         for name, value in (row.get("timings_s") or {}).items()}}

    for k in prefixes:
        block = row["by_k"][str(k)]
        for aggregator in AGGREGATORS:
            stored = _stored_scores(block, aggregator)
            stored_row, stored_index, stored_xyz = _stored_prediction(block, aggregator)

            # (1) the row's own score vector must reproduce the row's argmax
            derived_row = me.argmax_by_global_index(stored, indices)
            if derived_row != stored_row or int(indices[stored_row]) != stored_index:
                raise ValueError(
                    f"{row['query_id']} K={k} {aggregator}: the row's published scores select "
                    f"row {derived_row} (candidate {indices[derived_row]}) under the registered "
                    f"tie-break, but the row records row {stored_row} (candidate "
                    f"{stored_index}); the row is internally inconsistent")
            if not np.array_equal(coordinates[stored_row],
                                  np.asarray(stored_xyz, dtype=np.float64)):
                raise ValueError(
                    f"{row['query_id']} K={k} {aggregator}: the row's prediction_xyz "
                    f"{list(stored_xyz)} is not the G1 coordinate of candidate {stored_index} "
                    f"({coordinates[stored_row].tolist()}); JSON float round-trip is exact, so "
                    "this is a different candidate array")

            derived_margin = me.top1_margin(stored)
            if aggregator == "lme" and float(block["margin"]) != derived_margin:
                raise ValueError(
                    f"{row['query_id']} K={k}: the row records a top-1 margin of "
                    f"{float(block['margin']):.6g} but its own published scores have "
                    f"{derived_margin:.6g}; an inflated margin would excuse an argmax that "
                    "could in fact flip, so it is refused rather than used")

            # (2) the sidecar recompute
            recomputed_scores = (recomputed[k]["scores"] if aggregator == "lme"
                                 else recomputed[k]["mean_scores"])
            deviation = float((recomputed_scores - stored).abs().max())
            recomputed_row = me.argmax_by_global_index(recomputed_scores, indices)
            margin = derived_margin
            explained = bool(margin <= me.ARGMAX_STABILITY_FACTOR * deviation)
            agrees = recomputed_row == stored_row
            if not agrees and (sidecar_argmax_policy == "strict" or not explained):
                raise ValueError(
                    f"{row['query_id']} K={k} {aggregator}: the argmax recomputed from the "
                    f"float16 sidecar is candidate {indices[recomputed_row]} but the row "
                    f"records {stored_index}; the row's top-1 margin is {margin:.3g} and the "
                    f"measured sidecar deviation is {deviation:.3g} "
                    f"({'within' if explained else 'NOT within'} the "
                    f"{me.ARGMAX_STABILITY_FACTOR}x stability bound, policy "
                    f"{sidecar_argmax_policy!r}). {SIDECAR_ARGMAX_NOTE}")
            out["sidecar"][aggregator][k] = {
                "max_abs_delta": deviation, "argmax_agrees": bool(agrees),
                "margin": margin, "explained_by_precision": explained}

            best = float(stored[stored_row])
            e_loc = float(np.linalg.norm(coordinates[stored_row] - truth))
            e_excess = max(0.0, e_loc - e_oracle)
            out["by"][aggregator][k] = {
                "prediction_index": stored_index,
                "prediction_row": stored_row,
                "prediction_xyz": coordinates[stored_row].tolist(),
                "best_score": best,
                "margin": margin,
                "e_loc": e_loc,
                "e_excess": e_excess,
                "success_raw": {radius_key(r): float(e_loc <= float(r)) for r in radii},
                "success_oracle_normalized": {radius_key(r): float(e_excess <= float(r))
                                              for r in radii},
            }
    return out


# --------------------------------------------------------------------------- #
# room-first aggregation and the room bootstrap
# --------------------------------------------------------------------------- #
def flat_stat_names(radii=SUCCESS_RADII):
    """The flat statistic names a cell publishes, in a pinned order."""
    names = ["median_e_loc", "mean_e_loc", "median_e_excess", "mean_e_excess"]
    names += [f"success_raw@{radius_key(r)}" for r in radii]
    names += [f"success_oracle_normalized@{radius_key(r)}" for r in radii]
    return names


def room_block(values, radii=SUCCESS_RADII):
    """One room's statistics from its per-query ``(e_loc, e_excess)`` lists."""
    e_loc = np.asarray(values["e_loc"], dtype=np.float64)
    e_excess = np.asarray(values["e_excess"], dtype=np.float64)
    if e_loc.size == 0:
        raise ValueError("a room block needs at least one query")
    if not (np.isfinite(e_loc).all() and np.isfinite(e_excess).all()):
        raise ValueError("a room block must be finite (no NaN or Inf)")
    block = {"n_queries": int(e_loc.size),
             "median_e_loc": float(np.median(e_loc)),
             "mean_e_loc": float(e_loc.mean()),
             "median_e_excess": float(np.median(e_excess)),
             "mean_e_excess": float(e_excess.mean())}
    for r in radii:
        block[f"success_raw@{radius_key(r)}"] = float((e_loc <= float(r)).mean())
        block[f"success_oracle_normalized@{radius_key(r)}"] = \
            float((e_excess <= float(r)).mean())
    return block


def room_bootstrap_draws(n_rooms, seed=BOOTSTRAP_SEED, n=BOOTSTRAP_N):
    """The resample index matrix ``[n, n_rooms]`` -- built ONCE and shared.

    Every interval in the report is read off the SAME pre-registered resampling
    of the 16 rooms, so two statistics of one cell cannot come from different
    draws, and the whole report is reproducible from ``(seed, n, n_rooms)``.
    """
    n_rooms, n = int(n_rooms), int(n)
    if n_rooms < 1:
        raise ValueError("a room bootstrap needs at least one room")
    if n < 1:
        raise ValueError(f"n (bootstrap resamples) must be >= 1, got {n}")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n_rooms, size=(n, n_rooms))


def percentile_ci(samples, alpha=BOOTSTRAP_ALPHA):
    """Two-sided percentile interval, interpolation pinned to ``linear``.

    Stated rather than inherited from NumPy's default, exactly as
    ``scoring._percentile_ci`` pins it, so the registered endpoints cannot
    silently change with the library.
    """
    alpha = _finite(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    low, high = np.percentile(np.asarray(samples, dtype=np.float64),
                              [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
                              method="linear")
    return float(low), float(high)


def across_rooms(per_room, names=None, draws=None, seed=BOOTSTRAP_SEED, n=BOOTSTRAP_N,
                 alpha=BOOTSTRAP_ALPHA):
    """Room-first aggregation: the mean over rooms, with a room-bootstrap CI.

    ``per_room`` is ``{room_id: {stat: value}}``. Rooms are ordered by id so the
    bootstrap draw matrix indexes the same room every time.
    """
    rooms = sorted(per_room)
    if not rooms:
        raise ValueError("across_rooms needs at least one room")
    names = list(flat_stat_names() if names is None else names)
    draws = room_bootstrap_draws(len(rooms), seed=seed, n=n) if draws is None else draws
    if draws.shape[1] != len(rooms):
        raise ValueError(f"the bootstrap draw matrix is over {draws.shape[1]} rooms but this "
                         f"cell has {len(rooms)}")
    out = {}
    for name in names:
        values = np.asarray([float(per_room[room][name]) for room in rooms], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"statistic {name!r} is not finite in every room")
        resampled = values[draws].mean(axis=1)
        low, high = percentile_ci(resampled, alpha=alpha)
        out[name] = {"point": float(values.mean()), "ci_lo": low, "ci_hi": high,
                     "sd_over_rooms": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                     "min_room": float(values.min()), "max_room": float(values.max())}
    out["_settings"] = {"n_rooms": len(rooms), "rooms": rooms, "bootstrap_seed": int(seed),
                        "n_boot": int(draws.shape[0]), "alpha": float(alpha),
                        "method": "percentile (linear interpolation), rooms resampled with "
                                  "replacement",
                        "aggregation": "room-first: the statistic is computed inside each room "
                                       "and the room values are then averaged (§2)"}
    return out


def pooled_block(e_loc, e_excess, radii=SUCCESS_RADII):
    """The per-query pooled block -- a LABELLED SECONDARY, never the headline."""
    return dict(room_block({"e_loc": e_loc, "e_excess": e_excess}, radii=radii),
                label="pooled over queries (secondary; the registered aggregation is "
                      "room-first)")


def build_cell(results, aggregator, k, draws=None, radii=SUCCESS_RADII, **bootstrap):
    """One ``(aggregator, K)`` cell: per room, across rooms, and pooled."""
    by_room = {}
    all_loc, all_excess = [], []
    for result in results:
        entry = result["by"][aggregator][k]
        bucket = by_room.setdefault(str(result["room_id"]), {"e_loc": [], "e_excess": []})
        bucket["e_loc"].append(entry["e_loc"])
        bucket["e_excess"].append(entry["e_excess"])
        all_loc.append(entry["e_loc"])
        all_excess.append(entry["e_excess"])
    per_room = {room: room_block(values, radii=radii) for room, values in by_room.items()}
    return {"aggregator": aggregator, "k": int(k), "role": AGGREGATOR_ROLES[aggregator],
            "per_room": per_room,
            "across_rooms": across_rooms(per_room, names=flat_stat_names(radii), draws=draws,
                                         **bootstrap),
            "pooled": pooled_block(all_loc, all_excess, radii=radii)}


def oracle_report(results, draws=None, threshold=ORACLE_THRESHOLD, **bootstrap):
    """The oracle distribution: per room, across rooms, pooled (§2)."""
    by_room = {}
    for result in results:
        by_room.setdefault(str(result["room_id"]), []).append(float(result["e_oracle"]))
    per_room, names = {}, ["median_e_oracle", "mean_e_oracle",
                           f"fraction_e_oracle_over_{radius_key(threshold)}"]
    for room in sorted(by_room):
        values = np.asarray(by_room[room], dtype=np.float64)
        per_room[room] = {
            "n_queries": int(values.size),
            "median_e_oracle": float(np.median(values)),
            "mean_e_oracle": float(values.mean()),
            "min_e_oracle": float(values.min()),
            "max_e_oracle": float(values.max()),
            "p95_e_oracle": float(np.percentile(values, 95.0, method="linear")),
            f"n_e_oracle_over_{radius_key(threshold)}":
                int((values > float(threshold)).sum()),
            f"fraction_e_oracle_over_{radius_key(threshold)}":
                float((values > float(threshold)).mean())}
    pooled_values = np.asarray([float(r["e_oracle"]) for r in results], dtype=np.float64)
    return {"threshold_m": float(threshold),
            "per_room": per_room,
            "across_rooms": across_rooms(per_room, names=names, draws=draws, **bootstrap),
            "pooled": {"n_queries": int(pooled_values.size),
                       "median_e_oracle": float(np.median(pooled_values)),
                       "mean_e_oracle": float(pooled_values.mean()),
                       "min_e_oracle": float(pooled_values.min()),
                       "max_e_oracle": float(pooled_values.max()),
                       f"n_e_oracle_over_{radius_key(threshold)}":
                           int((pooled_values > float(threshold)).sum()),
                       f"fraction_e_oracle_over_{radius_key(threshold)}":
                           float((pooled_values > float(threshold)).mean()),
                       "label": "pooled over queries (secondary; the registered aggregation is "
                                "room-first)"},
            "note": "e_oracle is a property of the candidate GEOMETRY and the continuous truth, "
                    "so it is identical for every aggregator and every K; it is the denominator "
                    "the oracle-normalized success is measured against"}


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
#: the per-query components an I1 row stamps (``_score_one_query``'s timer).
ROW_TIMING_COMPONENTS = ("conditioning", "sampling", "decode", "embed", "scoring")

LATENCY_SCOPE_NOTE = (
    "the row's timings_s covers exactly the per-query generation+scoring loop -- conditioning "
    "assembly, sampling, VAE decode, AGREE embedding and the cosine -- because that is what the "
    "engine stamps into a row (meshgrid_engine._build_row). The per-QUERY context branch and the "
    "per-RECEIVER source-cache build are billed to the run and to the receiver group "
    "respectively and are recorded only in run_summary.json / the throughput probe, so they are "
    "NOT included here; a wall-clock cost must add them separately")


def latency_report(results, components=ROW_TIMING_COMPONENTS):
    """Latency per query, per candidate and per generated RIR, from the rows."""
    totals = {name: 0.0 for name in components}
    per_query, n_pairs, n_waveforms = [], 0, 0
    missing = []
    for result in results:
        timings = result.get("latency_s") or {}
        if not timings:
            missing.append(result["query_id"])
            continue
        seconds = 0.0
        for name in components:
            value = float(timings.get(name, 0.0))
            totals[name] += value
            seconds += value
        per_query.append(seconds)
        n_pairs += int(result["n_candidates"])
        n_waveforms += int(result["n_candidates"]) * int(result["num_samples"])
    if missing:
        raise ValueError(f"{len(missing)} rows carry no timings_s (first {missing[:3]}); "
                         "§2 registers latency per query, candidate and generated RIR")
    values = np.asarray(per_query, dtype=np.float64)
    total = float(values.sum())
    return {"n_queries": int(values.size),
            "candidate_query_pairs": int(n_pairs),
            "generated_waveforms": int(n_waveforms),
            "total_seconds": total,
            "seconds_per_query": {"mean": float(values.mean()),
                                  "median": float(np.median(values)),
                                  "min": float(values.min()), "max": float(values.max())},
            "seconds_per_candidate": total / n_pairs if n_pairs else 0.0,
            "seconds_per_generated_rir": total / n_waveforms if n_waveforms else 0.0,
            "component_seconds": {name: float(totals[name]) for name in components},
            "component_fraction": {name: (float(totals[name]) / total if total else 0.0)
                                   for name in components},
            "scope_note": LATENCY_SCOPE_NOTE}


# --------------------------------------------------------------------------- #
# the deterministic uniform-random candidate baseline
# --------------------------------------------------------------------------- #
def baseline_key(seed, query_id):
    """Deterministic draw key for ``(seed, query)`` -- sha256, never ``hash()``.

    Keying the draw on the query rather than on a sequential stream makes the
    baseline independent of the order the report happens to walk the rooms in, so
    a re-run, a re-shard or a partial replay draws the same candidate for the
    same query -- the same argument the engine makes for its noise keys
    (``meshgrid_engine.noise_key``).
    """
    payload = json.dumps(["loc_meshgrid_random_baseline", int(seed), str(query_id)],
                         separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def draw_baseline_candidate(seed, query_id, n_candidates):
    """One uniform draw over a query's IDENTICAL valid candidate manifest."""
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError(f"{query_id}: a baseline draw needs a non-empty candidate set")
    rng = np.random.default_rng(baseline_key(seed, query_id))
    return int(rng.integers(0, n_candidates))


def baseline_repetition(records, seed, draws=None, radii=SUCCESS_RADII, bootstrap=None):
    """One full independent repetition of the random baseline, room-first.

    ``bootstrap`` is passed as a dict rather than ``**kwargs`` on purpose: the
    baseline's own ``seed`` and the room bootstrap's ``seed`` are two different
    registered numbers and must not be able to collide in one call.
    """
    bootstrap = dict(bootstrap or {})
    by_room, all_loc, all_excess = {}, [], []
    for record in records:
        e_loc = float(record["by_seed"][int(seed)])
        e_excess = max(0.0, e_loc - float(record["e_oracle"]))
        bucket = by_room.setdefault(str(record["room_id"]), {"e_loc": [], "e_excess": []})
        bucket["e_loc"].append(e_loc)
        bucket["e_excess"].append(e_excess)
        all_loc.append(e_loc)
        all_excess.append(e_excess)
    per_room = {room: room_block(values, radii=radii) for room, values in by_room.items()}
    return {"seed": int(seed), "per_room": per_room,
            "across_rooms": across_rooms(per_room, names=flat_stat_names(radii), draws=draws,
                                         **bootstrap),
            "pooled": pooled_block(all_loc, all_excess, radii=radii)}


def baseline_report(results, seeds=RANDOM_BASELINE_SEEDS, draws=None, radii=SUCCESS_RADII,
                    bootstrap=None):
    """Every repetition plus the summary over repetitions (§2 controls)."""
    bootstrap = dict(bootstrap or {})
    seeds = [int(s) for s in seeds]
    records = [{"query_id": r["query_id"], "room_id": r["room_id"],
                "e_oracle": r["e_oracle"], "by_seed": r["baseline_e_loc"]}
               for r in results]
    repetitions = [baseline_repetition(records, seed, draws=draws, radii=radii,
                                       bootstrap=bootstrap)
                   for seed in seeds]

    names = flat_stat_names(radii)
    summary = {}
    for name in names:
        values = np.asarray([rep["across_rooms"][name]["point"] for rep in repetitions],
                            dtype=np.float64)
        summary[name] = {"mean": float(values.mean()),
                         "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                         "min": float(values.min()), "max": float(values.max()),
                         "per_seed": [float(v) for v in values]}

    # the pooled-over-repetitions view: every query contributes all len(seeds)
    # draws, each with weight 1/len(seeds), so a query cannot count more than one
    pooled_by_room, pooled_loc, pooled_excess = {}, [], []
    for record in records:
        bucket = pooled_by_room.setdefault(str(record["room_id"]),
                                           {"e_loc": [], "e_excess": []})
        for seed in seeds:
            e_loc = float(record["by_seed"][seed])
            bucket["e_loc"].append(e_loc)
            bucket["e_excess"].append(max(0.0, e_loc - float(record["e_oracle"])))
            pooled_loc.append(e_loc)
            pooled_excess.append(max(0.0, e_loc - float(record["e_oracle"])))
    pooled_per_room = {room: room_block(values, radii=radii)
                       for room, values in pooled_by_room.items()}
    return {
        "rule": ("uniform draw over the query's IDENTICAL published valid candidate set; the "
                 "draw is keyed by sha256(seed, query_id) so it is independent of iteration "
                 "order, and each pre-registered seed is one independent full repetition"),
        "seeds": seeds,
        "repetitions": repetitions,
        "summary_over_repetitions": summary,
        "all_draws": {"per_room": pooled_per_room,
                      "across_rooms": across_rooms(pooled_per_room, names=names, draws=draws,
                                                   **bootstrap),
                      "pooled": pooled_block(pooled_loc, pooled_excess, radii=radii),
                      "label": f"all {len(seeds)} repetitions pooled (each query contributes "
                               f"{len(seeds)} draws)"},
    }


# --------------------------------------------------------------------------- #
# score / candidate-count associations (§2 "report score/candidate-count
# associations")
# --------------------------------------------------------------------------- #
def _rankdata(values):
    """Average-tie ranks -- Spearman's transform, without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = 0.5 * (start + index - 1) + 1.0
            start = index
    return ranks


def correlation(x, y):
    """Pearson and Spearman correlation, with the degenerate cases named."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"correlation needs paired inputs, got {x.size} and {y.size}")
    if x.size < 2:
        return {"n": int(x.size), "pearson": None, "spearman": None,
                "reason": "fewer than two observations"}
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("correlation inputs must be finite (no NaN or Inf)")

    def _pearson(a, b):
        sd_a, sd_b = a.std(), b.std()
        if sd_a == 0.0 or sd_b == 0.0:
            return None
        return float((((a - a.mean()) * (b - b.mean())).mean()) / (sd_a * sd_b))

    return {"n": int(x.size), "pearson": _pearson(x, y),
            "spearman": _pearson(_rankdata(x), _rankdata(y))}


def association_report(results, aggregator=HEADLINE_AGGREGATOR, k=HEADLINE_K, n_buckets=4):
    """How the score and the errors move with the candidate-set size M.

    A candidate set's size is a property of the ROOM's geometry and of the
    per-query guards, not of the model, so a score that tracks it is a scale
    artefact rather than evidence about localization. §2 asks for the association
    to be reported; it is a diagnostic and decides nothing.
    """
    counts = np.asarray([float(r["n_candidates"]) for r in results], dtype=np.float64)
    best = np.asarray([float(r["by"][aggregator][k]["best_score"]) for r in results],
                      dtype=np.float64)
    margin = np.asarray([float(r["by"][aggregator][k]["margin"]) for r in results],
                        dtype=np.float64)
    e_loc = np.asarray([float(r["by"][aggregator][k]["e_loc"]) for r in results],
                       dtype=np.float64)
    e_excess = np.asarray([float(r["by"][aggregator][k]["e_excess"]) for r in results],
                          dtype=np.float64)
    e_oracle = np.asarray([float(r["e_oracle"]) for r in results], dtype=np.float64)
    finite_margin = np.isfinite(margin)

    pooled = {
        "n_candidates_vs_best_score": correlation(counts, best),
        "n_candidates_vs_e_loc": correlation(counts, e_loc),
        "n_candidates_vs_e_excess": correlation(counts, e_excess),
        "n_candidates_vs_e_oracle": correlation(counts, e_oracle),
        "n_candidates_vs_top1_margin": correlation(counts[finite_margin],
                                                   margin[finite_margin]),
        "best_score_vs_e_loc": correlation(best, e_loc),
    }

    edges = np.quantile(counts, np.linspace(0.0, 1.0, int(n_buckets) + 1), method="linear")
    buckets = []
    for index in range(int(n_buckets)):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (counts >= low) & (counts <= high) if index == int(n_buckets) - 1 \
            else (counts >= low) & (counts < high)
        if not mask.any():
            buckets.append({"quantile": index + 1, "n_candidates_range": [low, high],
                            "n_queries": 0})
            continue
        buckets.append({"quantile": index + 1, "n_candidates_range": [low, high],
                        "n_queries": int(mask.sum()),
                        "mean_n_candidates": float(counts[mask].mean()),
                        "mean_best_score": float(best[mask].mean()),
                        "median_e_loc": float(np.median(e_loc[mask])),
                        "mean_e_loc": float(e_loc[mask].mean()),
                        "median_e_excess": float(np.median(e_excess[mask])),
                        "median_e_oracle": float(np.median(e_oracle[mask]))})

    per_room = {}
    by_room = {}
    for index, result in enumerate(results):
        by_room.setdefault(str(result["room_id"]), []).append(index)
    for room in sorted(by_room):
        rows = np.asarray(by_room[room], dtype=np.int64)
        per_room[room] = {
            "n_queries": int(rows.size),
            "mean_n_candidates": float(counts[rows].mean()),
            "n_candidates_vs_best_score": correlation(counts[rows], best[rows]),
            "n_candidates_vs_e_loc": correlation(counts[rows], e_loc[rows])}

    return {"aggregator": aggregator, "k": int(k),
            "n_candidates": {"mean": float(counts.mean()), "median": float(np.median(counts)),
                             "min": float(counts.min()), "max": float(counts.max())},
            "pooled": pooled, "by_candidate_count_quantile": buckets, "per_room": per_room,
            "note": "diagnostic only: the candidate-set size is fixed by room geometry and the "
                    "per-query receiver/context/z-band guards, so an association with the score "
                    "is a scale effect and never a localization result"}


# --------------------------------------------------------------------------- #
# the pre-registered visualization cases
# --------------------------------------------------------------------------- #
def select_visualization_cases(results, aggregator=HEADLINE_AGGREGATOR, k=HEADLINE_K):
    """The three quantile cases, as a pure function of the finished results.

    Ordering is ``(e_loc, position)``; ties therefore go to the smaller global
    stream position, and the median is the LOWER median at ``(n - 1) // 2`` so an
    even-sized set has one named winner rather than an interpolated non-query.
    """
    results = list(results)
    if not results:
        raise ValueError("visualization cases need at least one scored query")
    ordered = sorted(results,
                     key=lambda r: (float(r["by"][aggregator][k]["e_loc"]), int(r["position"])))
    picks = (("lowest_e_loc", 0),
             ("median_e_loc", (len(ordered) - 1) // 2),
             ("highest_e_loc", len(ordered) - 1))
    cases = []
    for label, index in picks:
        result = ordered[index]
        entry = result["by"][aggregator][k]
        cases.append({"quantile": label, "rank": int(index), "n_ranked": len(ordered),
                      "query_id": result["query_id"], "room_id": result["room_id"],
                      "position": int(result["position"]),
                      "e_loc": float(entry["e_loc"]),
                      "e_excess": float(entry["e_excess"]),
                      "e_oracle": float(result["e_oracle"])})
    return {"rule": VISUALIZATION_RULE, "aggregator": aggregator, "k": int(k), "cases": cases}


def build_case_payload(selection, rows_by_id, plans_by_id, results_by_id, run_dir, plan,
                       softmax_t=VISUALIZATION_T):
    """The renderer's payload for the selected cases -- and a registered dump list.

    The file doubles as a ``--dump-cases`` list for ``localize_meshgrid.py``: its
    top-level ``query_ids`` is exactly what
    ``meshgrid_engine.load_dump_cases`` reads, so the announcement-08 waveform
    exemption can be extended to these three cases by registering this file's
    sha256 -- no second, hand-written list.
    """
    aggregator, k = selection["aggregator"], int(selection["k"])
    payload_cases = []
    for case in selection["cases"]:
        query_id = case["query_id"]
        row, query, result = rows_by_id[query_id], plans_by_id[query_id], results_by_id[query_id]
        block = row["by_k"][str(k)]
        scores = _stored_scores(block, aggregator)
        coordinates = np.asarray(query.coordinates, dtype=np.float64)
        entry = result["by"][aggregator][k]
        payload_cases.append({
            "quantile": case["quantile"], "rank": case["rank"],
            "query_id": query_id, "room_id": row["room_id"], "position": int(row["position"]),
            "receiver_id": row.get("receiver_id"),
            "receiver_xyz": [float(v) for v in row["receiver_xyz"]],
            "branch": row["branch"], "z_band": row.get("z_band"),
            "n_candidates": int(row["n_candidates"]), "num_samples": int(row["num_samples"]),
            "candidate_indices": [int(i) for i in row["candidate_indices"]],
            "candidate_xyz": coordinates.tolist(),
            "scores": [float(v) for v in scores],
            "score_softmax": [float(v) for v in sc.softmax_map(scores, softmax_t)],
            "score_softmax_T": float(softmax_t),
            "score_softmax_label": VISUALIZATION_T_LABEL,
            "prediction_index": int(entry["prediction_index"]),
            "prediction_xyz": [float(v) for v in entry["prediction_xyz"]],
            "truth_xyz": [float(v) for v in result["truth_xyz"]],
            "oracle_candidate_index": int(result["oracle_candidate_index"]),
            "oracle_candidate_xyz": [float(v) for v in result["oracle_candidate_xyz"]],
            "e_loc": float(entry["e_loc"]), "e_excess": float(entry["e_excess"]),
            "e_oracle": float(result["e_oracle"]),
            "mesh": ((plan.report.get("rooms") or {}).get(row["room_id"]) or {}).get("mesh"),
            "row_path": os.path.relpath(row["_row_path"], str(run_dir)),
            "sims_path": row.get("sims_path"),
        })
    # the dump authority is a SET of queries: on a subset small enough for two
    # quantiles to land on one query the case list still names three cases, but
    # the authority names each query once
    authority = list(dict.fromkeys(case["query_id"] for case in payload_cases))
    return {
        "experiment": "exp_22 loc_meshgrid R1 visualization cases",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # the key meshgrid_engine.load_dump_cases reads
        "query_ids": authority,
        "selection": {"rule": VISUALIZATION_RULE, "aggregator": aggregator, "k": k},
        "subset": SUBSET_LABEL,
        "agree_leakage_caveat": me.AGREE_LEAKAGE_CAVEAT,
        "scorer_readout_deviation": me.SCORER_READOUT_DEVIATION,
        "cases": payload_cases,
    }


# --------------------------------------------------------------------------- #
# the whole report
# --------------------------------------------------------------------------- #
def evaluate_run(run_dir, audit_report, context_manifest, metadata_root, totals=None,
                 radii=SUCCESS_RADII, sidecar_argmax_policy=SIDECAR_ARGMAX_POLICY,
                 baseline_seeds=RANDOM_BASELINE_SEEDS, oracle_tolerance=ORACLE_TOLERANCE,
                 require_manifest_census=True, on_query=None):
    """Gate the artifacts, then evaluate every query. Returns the raw material.

    The gates run first and in full -- binding, G1 chain, D1 manifest, row and
    sidecar digests, the census -- and only then does a single room-major pass
    compute anything. Inside a room, every query's row is authenticated against
    the G1 plan BEFORE that room's first metric is taken.

    ``require_manifest_census`` exists for fixtures, exactly as
    ``meshgrid_queries.build_manifest``'s does, and is never relaxed by the
    production path: the CLI leaves it on, so a report can only ever be published
    against the registered 6,337 -> 5,337 context manifest.
    """
    run_dir = str(run_dir)
    binding, binding_sha = load_published_binding(run_dir)
    plan = me.load_audit_plan(audit_report, branch=binding["branch"])
    manifest = mq.load_manifest(context_manifest, require_census=require_manifest_census)
    records = manifest["records"]
    rows = verify_rows(run_dir, binding_sha)
    census = assert_census(rows, records, totals=totals)
    protocol = assert_row_protocol(rows, binding)

    if sorted(plan.rooms) != census["rooms"]:
        raise ValueError(f"the audit publishes rooms {sorted(plan.rooms)[:3]}... but the run "
                         f"publishes {census['rooms'][:3]}...; a report joins one audit to one "
                         "run")

    rows_by_id = {row["query_id"]: row for row in rows}
    records_by_id = {record["query_id"]: record for record in records}
    resolver = TruthResolver(metadata_root)
    seeds = [int(s) for s in baseline_seeds]

    results, plans_by_id, receiver_drift = [], {}, []
    for room_id in sorted(plan.rooms):
        room_plan = me.load_room_plan(plan, room_id)
        # phase A: every row of this room is authenticated against the G1 plan
        # before phase B takes a single number out of it
        for query in room_plan.queries:
            me.assert_published_matches(run_dir, query, binding_sha256=binding_sha)
        for query in room_plan.queries:
            row = rows_by_id[query.query_id]
            metadata_receiver, truth = resolver.resolve(records_by_id[query.query_id])
            receiver_drift.append(assert_receiver_matches(query.query_id, metadata_receiver,
                                                          query.receiver_xyz))
            sims_path = os.path.join(run_dir, str(row["sims_path"]))
            sims = np.load(sims_path)
            result = evaluate_query(row, sims, query.coordinates, truth, tau=protocol["tau"],
                                    radii=radii, oracle_tolerance=oracle_tolerance,
                                    sidecar_argmax_policy=sidecar_argmax_policy)
            result["baseline_e_loc"] = {
                seed: float(np.linalg.norm(
                    np.asarray(query.coordinates, dtype=np.float64)[
                        draw_baseline_candidate(seed, query.query_id, query.n_candidates)]
                    - np.asarray(truth, dtype=np.float64)))
                for seed in seeds}
            results.append(result)
            plans_by_id[query.query_id] = query
            if on_query is not None:
                on_query(result)

    results.sort(key=lambda result: int(result["position"]))
    return {"binding": binding, "binding_sha256": binding_sha, "plan": plan,
            "manifest": manifest, "records": records, "rows": rows,
            "rows_by_id": rows_by_id, "plans_by_id": plans_by_id, "results": results,
            "census": census, "protocol": protocol,
            "max_receiver_drift_m": float(max(receiver_drift)) if receiver_drift else 0.0,
            "baseline_seeds": seeds}


def sidecar_summary(results):
    """What the float16 recompute found -- counted and named, never absorbed."""
    out = {"policy_note": SIDECAR_ARGMAX_NOTE, "by_cell": {}}
    named = []
    for aggregator in AGGREGATORS:
        out["by_cell"][aggregator] = {}
        for k in sorted(results[0]["sidecar"][aggregator]):
            deltas = np.asarray([r["sidecar"][aggregator][k]["max_abs_delta"]
                                 for r in results], dtype=np.float64)
            disagreements = [r for r in results
                             if not r["sidecar"][aggregator][k]["argmax_agrees"]]
            out["by_cell"][aggregator][str(k)] = {
                "n_queries": len(results),
                "max_abs_delta": float(deltas.max()), "mean_abs_delta": float(deltas.mean()),
                "n_argmax_disagreements": len(disagreements),
                "all_explained_by_precision": all(
                    r["sidecar"][aggregator][k]["explained_by_precision"]
                    for r in disagreements)}
            for result in disagreements:
                entry = result["sidecar"][aggregator][k]
                named.append({"query_id": result["query_id"], "aggregator": aggregator,
                              "k": int(k), "margin": entry["margin"],
                              "sidecar_max_abs_delta": entry["max_abs_delta"],
                              "explained_by_precision": entry["explained_by_precision"]})
    out["argmax_disagreements"] = named
    out["n_argmax_disagreements"] = len(named)
    return out


def oracle_crosscheck_summary(results):
    """How far the re-derived oracle sat from the one G1 published."""
    deltas = np.asarray([float(r["e_oracle_delta"]) for r in results], dtype=np.float64)
    return {"n_queries": int(deltas.size), "max_abs_delta_m": float(deltas.max()),
            "mean_abs_delta_m": float(deltas.mean()), "tolerance_m": float(ORACLE_TOLERANCE),
            "note": "the report re-derives e_oracle = min_c ||c - x*_s|| from the G1 candidate "
                    "block and the pair metadata's src_loc, and requires it to equal the value "
                    "the G1 audit published in the row"}


def build_report(evaluated, run_dir, audit_report, context_manifest, metadata_root,
                 radii=SUCCESS_RADII, bootstrap_seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_N,
                 alpha=BOOTSTRAP_ALPHA, sidecar_argmax_policy=SIDECAR_ARGMAX_POLICY):
    """The machine-readable R1 report: every §2 readout, every stamp."""
    results = evaluated["results"]
    census = evaluated["census"]
    prefixes = [int(k) for k in evaluated["protocol"]["k_prefixes"]]
    draws = room_bootstrap_draws(census["n_rooms"], seed=bootstrap_seed, n=n_boot)
    bootstrap = {"seed": bootstrap_seed, "n": n_boot, "alpha": alpha}

    metrics = {aggregator: {str(k): build_cell(results, aggregator, k, draws=draws,
                                               radii=radii, **bootstrap)
                            for k in prefixes}
               for aggregator in AGGREGATORS}

    merge_path = os.path.join(str(run_dir), "merge_report.json")
    report = {
        "experiment": "exp_22 loc_meshgrid R1 report",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": {
            "subset": SUBSET_LABEL,
            "agree_leakage_caveat": me.AGREE_LEAKAGE_CAVEAT,
            "scorer_readout_deviation": me.SCORER_READOUT_DEVIATION,
            "sims_precision_caveat": me.SIMS_PRECISION_CAVEAT,
            "batching_caveat": me.BATCHING_CAVEAT,
            "aggregator_roles": AGGREGATOR_ROLES,
        },
        "controls_elsewhere": CONTROLS_ELSEWHERE,
        "provenance": {
            "run_dir": str(run_dir),
            "binding_sha256": evaluated["binding_sha256"],
            "binding": evaluated["binding"],
            "audit_report": str(audit_report),
            "audit_report_sha256": evaluated["plan"].report_sha256,
            "branch": evaluated["plan"].branch,
            "context_manifest": str(context_manifest),
            "context_manifest_sha256": me.file_sha256(context_manifest),
            "context_manifest_filtered_stream_sha256":
                evaluated["manifest"].get("filtered_stream_sha256"),
            "metadata_root": str(metadata_root),
            "merge_report_sha256": (me.file_sha256(merge_path)
                                    if os.path.isfile(merge_path) else None),
        },
        "protocol": {
            "tau": float(evaluated["protocol"]["tau"]),
            "seed": int(evaluated["protocol"]["seed"]),
            "num_samples": int(evaluated["protocol"]["num_samples"]),
            "k_prefixes": prefixes,
            "noise_policy": evaluated["protocol"]["noise_policy"],
            "scorer_readout": evaluated["protocol"]["scorer_readout"],
            "success_radii_m": [float(r) for r in radii],
            "oracle_threshold_m": float(ORACLE_THRESHOLD),
            "aggregation": "room-first, then averaged over rooms (§2)",
            "bootstrap": {"seed": int(bootstrap_seed), "n_boot": int(n_boot),
                          "alpha": float(alpha), "unit": "room",
                          "method": "percentile (linear interpolation)"},
            "headline_cell": {"aggregator": HEADLINE_AGGREGATOR, "k": HEADLINE_K},
            "baseline_seeds": [int(s) for s in evaluated["baseline_seeds"]],
            "sidecar_argmax_policy": str(sidecar_argmax_policy),
        },
        "census": census,
        # These are not assertions the report makes about itself: each gate is a
        # fail-closed refusal inside evaluate_run, so a report exists only if all
        # of them passed. They are recorded so the artifact says WHICH checks the
        # number behind it survived.
        "gates": {
            "note": "every entry below is a gate evaluate_run refuses on; a published report is "
                    "proof they passed, not a claim that they did",
            "binding_recomputed_from_content": True,
            "g1_audit_chain_reverified": True,
            "d1_manifest_stream_and_census_reverified": True,
            "rows_and_sidecars_digest_verified": True,
            "rows_authenticated_against_g1_plan": True,
            "row_protocol_matches_binding": True,
            "max_receiver_drift_m": evaluated["max_receiver_drift_m"],
        },
        "metrics": metrics,
        "oracle": oracle_report(results, draws=draws, **bootstrap),
        "latency": latency_report(results),
        "random_baseline": baseline_report(results, seeds=evaluated["baseline_seeds"],
                                           draws=draws, radii=radii, bootstrap=bootstrap),
        "associations": association_report(results),
        "crosscheck": {"sidecar": sidecar_summary(results),
                       "oracle": oracle_crosscheck_summary(results)},
    }
    report["visualization_cases"] = select_visualization_cases(results)
    return report


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def format_number(value, digits=4):
    """One number as a table cell: ``None`` -> ``n/a``, integers grouped, inf named."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    number = float(value)
    if not math.isfinite(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _ci(entry, digits=4):
    return (f"{format_number(entry['point'], digits)} "
            f"[{format_number(entry['ci_lo'], digits)}, {format_number(entry['ci_hi'], digits)}]")


def _stamp(report):
    """The three stamps every table carries: binding, scope, leakage caveat."""
    provenance = report["provenance"]
    return (f"_binding_ `{provenance['binding_sha256']}` · "
            f"_subset_ {report['labels']['subset']} · "
            f"_AGREE leakage_ {report['labels']['agree_leakage_caveat']}")


def render_markdown(report):
    """The human-readable summary. Every table carries the three stamps."""
    lines = []
    provenance, protocol, census = report["provenance"], report["protocol"], report["census"]
    lines.append("# exp_22 R1 — mesh-grid localization report")
    lines.append("")
    lines.append(f"Generated {report['created_utc']} from `{provenance['run_dir']}`.")
    lines.append("")
    lines.append(f"- **Scope:** {report['labels']['subset']}")
    lines.append(f"- **Run binding:** `{provenance['binding_sha256']}`")
    lines.append(f"- **G1 audit:** `{provenance['audit_report_sha256'][:16]}...` "
                 f"(branch `{provenance['branch']}`)")
    lines.append(f"- **D1 manifest:** `{str(provenance['context_manifest_sha256'])[:16]}...`")
    lines.append(f"- **AGREE leakage caveat:** {report['labels']['agree_leakage_caveat']}")
    lines.append(f"- **Scorer readout deviation:** "
                 f"{report['labels']['scorer_readout_deviation']}")
    lines.append(f"- **Similarity precision:** {report['labels']['sims_precision_caveat']}")
    lines.append(f"- **Batching:** {report['labels']['batching_caveat']}")
    lines.append("")
    lines.append(f"Census: {census['n_queries']:,} queries over {census['n_rooms']} rooms, "
                 f"{census['candidate_query_pairs']:,} candidate-query pairs, "
                 f"{census['generated_waveforms']:,} generated waveforms; "
                 f"`{census['excluded_room']}` excluded ({census['n_excluded']:,} queries).")
    lines.append("")
    lines.append(f"Aggregation is room-first and intervals are 95% percentile intervals from "
                 f"{protocol['bootstrap']['n_boot']:,} room resamples at pre-registered seed "
                 f"{protocol['bootstrap']['seed']}.")
    lines.append("")

    for aggregator in AGGREGATORS:
        lines.append(f"## {aggregator.upper()} — {report['labels']['aggregator_roles'][aggregator]}")
        lines.append("")
        lines.append(_stamp(report))
        lines.append("")
        header = ("| K | median e_loc (m) | mean e_loc (m) | median e_excess (m) | "
                  "success@0.5 | success@1.0 | oracle-norm@0.5 | oracle-norm@1.0 |")
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in protocol["k_prefixes"]:
            cell = report["metrics"][aggregator][str(k)]["across_rooms"]
            lines.append(
                f"| {k} | {_ci(cell['median_e_loc'], 3)} | {_ci(cell['mean_e_loc'], 3)} | "
                f"{_ci(cell['median_e_excess'], 3)} | "
                f"{_ci(cell['success_raw@0.5'], 3)} | {_ci(cell['success_raw@1.0'], 3)} | "
                f"{_ci(cell['success_oracle_normalized@0.5'], 3)} | "
                f"{_ci(cell['success_oracle_normalized@1.0'], 3)} |")
        lines.append("")

    lines.append(f"## Per-room — headline cell "
                 f"({protocol['headline_cell']['aggregator']}, K = "
                 f"{protocol['headline_cell']['k']})")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append("| room | n | median e_loc | mean e_loc | median e_excess | success@0.5 | "
                 "oracle-norm@0.5 | median e_oracle | frac e_oracle>0.5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    headline = report["metrics"][protocol["headline_cell"]["aggregator"]][
        str(protocol["headline_cell"]["k"])]["per_room"]
    oracle_rooms = report["oracle"]["per_room"]
    for room in sorted(headline):
        block, oracle = headline[room], oracle_rooms[room]
        lines.append(
            f"| {room} | {block['n_queries']:,} | {format_number(block['median_e_loc'], 3)} | "
            f"{format_number(block['mean_e_loc'], 3)} | {format_number(block['median_e_excess'], 3)} | "
            f"{format_number(block['success_raw@0.5'], 3)} | "
            f"{format_number(block['success_oracle_normalized@0.5'], 3)} | "
            f"{format_number(oracle['median_e_oracle'], 3)} | "
            f"{format_number(oracle[f'fraction_e_oracle_over_{radius_key(ORACLE_THRESHOLD)}'], 4)} |")
    lines.append("")

    oracle = report["oracle"]
    lines.append("## Continuous-grid oracle")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append(f"- median e_oracle (room-first): {_ci(oracle['across_rooms']['median_e_oracle'], 4)} m")
    lines.append(f"- mean e_oracle (room-first): {_ci(oracle['across_rooms']['mean_e_oracle'], 4)} m")
    lines.append(
        "- fraction e_oracle > "
        f"{oracle['threshold_m']} m (room-first): "
        f"{_ci(oracle['across_rooms'][f'fraction_e_oracle_over_{radius_key(ORACLE_THRESHOLD)}'], 5)}")
    lines.append(f"- pooled median / mean / max: "
                 f"{format_number(oracle['pooled']['median_e_oracle'], 4)} / "
                 f"{format_number(oracle['pooled']['mean_e_oracle'], 4)} / "
                 f"{format_number(oracle['pooled']['max_e_oracle'], 4)} m")
    lines.append("")

    baseline = report["random_baseline"]
    lines.append("## Deterministic uniform-random candidate baseline")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append(f"{baseline['rule']}. Seeds {baseline['seeds']}.")
    lines.append("")
    lines.append("| seed | median e_loc (m) | mean e_loc (m) | success@0.5 | oracle-norm@0.5 |")
    lines.append("|---|---|---|---|---|")
    for repetition in baseline["repetitions"]:
        cell = repetition["across_rooms"]
        lines.append(f"| {repetition['seed']} | {_ci(cell['median_e_loc'], 3)} | "
                     f"{_ci(cell['mean_e_loc'], 3)} | {_ci(cell['success_raw@0.5'], 4)} | "
                     f"{_ci(cell['success_oracle_normalized@0.5'], 4)} |")
    summary = baseline["summary_over_repetitions"]
    lines.append(f"| **pooled** | {format_number(summary['median_e_loc']['mean'], 3)} "
                 f"± {format_number(summary['median_e_loc']['sd'], 3)} | "
                 f"{format_number(summary['mean_e_loc']['mean'], 3)} "
                 f"± {format_number(summary['mean_e_loc']['sd'], 3)} | "
                 f"{format_number(summary['success_raw@0.5']['mean'], 4)} "
                 f"± {format_number(summary['success_raw@0.5']['sd'], 4)} | "
                 f"{format_number(summary['success_oracle_normalized@0.5']['mean'], 4)} "
                 f"± {format_number(summary['success_oracle_normalized@0.5']['sd'], 4)} |")
    lines.append("")

    latency = report["latency"]
    lines.append("## Latency")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append(f"- per query: mean {format_number(latency['seconds_per_query']['mean'], 3)} s, "
                 f"median {format_number(latency['seconds_per_query']['median'], 3)} s")
    lines.append(f"- per candidate: {format_number(latency['seconds_per_candidate'] * 1e3, 4)} ms")
    lines.append(f"- per generated RIR: "
                 f"{format_number(latency['seconds_per_generated_rir'] * 1e3, 4)} ms")
    lines.append(f"- scope: {latency['scope_note']}")
    lines.append("")

    association = report["associations"]
    lines.append(f"## Score / candidate-count association "
                 f"({association['aggregator']}, K = {association['k']}) — diagnostic")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append("| pair | Pearson | Spearman |")
    lines.append("|---|---|---|")
    for name, entry in association["pooled"].items():
        lines.append(f"| {name} | {format_number(entry['pearson'], 4)} | {format_number(entry['spearman'], 4)} |")
    lines.append("")
    lines.append(f"{association['note']}")
    lines.append("")

    crosscheck = report["crosscheck"]
    lines.append("## Cross-checks")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append(f"- oracle re-derivation vs G1: max |delta| "
                 f"{crosscheck['oracle']['max_abs_delta_m']:.3g} m "
                 f"(tolerance {crosscheck['oracle']['tolerance_m']:g} m)")
    lines.append(f"- float16 sidecar recompute: "
                 f"{crosscheck['sidecar']['n_argmax_disagreements']} argmax disagreement(s) "
                 f"over {len(AGGREGATORS) * len(protocol['k_prefixes'])} cells, all explained "
                 f"by the declared precision: "
                 f"{format_number(all(entry['explained_by_precision'] for entry in crosscheck['sidecar']['argmax_disagreements']) if crosscheck['sidecar']['argmax_disagreements'] else True)}")
    lines.append(f"- max receiver drift (pair metadata vs candidate manifest): "
                 f"{report['gates']['max_receiver_drift_m']:.3g} m")
    lines.append("")

    lines.append("## §2 controls that are NOT in this report")
    lines.append("")
    for name, where in sorted(report["controls_elsewhere"].items()):
        lines.append(f"- **{name}** — {where}")
    lines.append("")

    cases = report["visualization_cases"]
    lines.append("## Pre-registered visualization cases")
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    lines.append(f"{cases['rule']}")
    lines.append("")
    lines.append("| quantile | query | room | e_loc (m) | e_excess (m) | e_oracle (m) |")
    lines.append("|---|---|---|---|---|---|")
    for case in cases["cases"]:
        lines.append(f"| {case['quantile']} | `{case['query_id']}` | {case['room_id']} | "
                     f"{format_number(case['e_loc'], 3)} | {format_number(case['e_excess'], 3)} | "
                     f"{format_number(case['e_oracle'], 3)} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_report(out_dir, report, case_payload):
    """Publish the three artifacts atomically and return their paths + digests."""
    os.makedirs(str(out_dir), exist_ok=True)
    paths = {"json": os.path.join(str(out_dir), REPORT_JSON),
             "markdown": os.path.join(str(out_dir), REPORT_MARKDOWN),
             "cases": os.path.join(str(out_dir), CASES_JSON)}
    me.write_json(paths["json"], jsonable(report))
    me.write_json(paths["cases"], jsonable(case_payload))
    tmp = paths["markdown"] + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(render_markdown(report))
    os.replace(tmp, paths["markdown"])
    return {"paths": paths, "sha256": {name: me.file_sha256(path)
                                       for name, path in paths.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True,
                        help="the MERGED I1 run directory (rows/, sidecars, run_binding.json)")
    parser.add_argument("--audit-report",
                        default=os.path.join("outputs_loc", "exp22", "g1_audit",
                                             "geometry_audit_report.json"))
    parser.add_argument("--context-manifest",
                        default=os.path.join("outputs_loc", "exp22",
                                             "d1_context_manifest.json"))
    parser.add_argument("--metadata-root",
                        default=os.path.join("AcousticRooms", "metadata"),
                        help="the dataset metadata root the continuous truth is read from")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--baseline-seeds", type=int, nargs="+",
                        default=list(RANDOM_BASELINE_SEEDS))
    parser.add_argument("--sidecar-argmax-policy", default=SIDECAR_ARGMAX_POLICY,
                        choices=list(SIDECAR_ARGMAX_POLICIES))
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """The registered settings are the defaults; a change must be deliberate."""
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED or int(args.n_boot) != BOOTSTRAP_N:
        print(f"NOTE: the bootstrap is being run at seed {args.bootstrap_seed} x "
              f"{args.n_boot} resamples, not the pre-registered "
              f"{BOOTSTRAP_SEED} x {BOOTSTRAP_N}; this report is a sensitivity check, "
              "not the registered one")
    if [int(s) for s in args.baseline_seeds] != list(RANDOM_BASELINE_SEEDS):
        print(f"NOTE: the random baseline is being run at seeds {args.baseline_seeds}, not the "
              f"pre-registered {list(RANDOM_BASELINE_SEEDS)}")
    if int(args.n_boot) < 1:
        _refuse("--n-boot must be at least 1")
    return True


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print(f"AGREE LEAKAGE CAVEAT: {me.AGREE_LEAKAGE_CAVEAT}")
    print(f"SUBSET: {SUBSET_LABEL}")

    evaluated = evaluate_run(args.run_dir, args.audit_report, args.context_manifest,
                             args.metadata_root, baseline_seeds=args.baseline_seeds,
                             sidecar_argmax_policy=args.sidecar_argmax_policy)
    print(f"gates passed: binding {evaluated['binding_sha256'][:12]}..., "
          f"{evaluated['census']['n_queries']:,} queries / "
          f"{evaluated['census']['n_rooms']} rooms")
    report = build_report(evaluated, args.run_dir, args.audit_report, args.context_manifest,
                          args.metadata_root, bootstrap_seed=args.bootstrap_seed,
                          n_boot=args.n_boot,
                          sidecar_argmax_policy=args.sidecar_argmax_policy)
    cases = build_case_payload(report["visualization_cases"], evaluated["rows_by_id"],
                               evaluated["plans_by_id"],
                               {r["query_id"]: r for r in evaluated["results"]},
                               args.run_dir, evaluated["plan"])
    published = write_report(args.out_dir, report, cases)

    headline = report["metrics"][HEADLINE_AGGREGATOR][str(HEADLINE_K)]["across_rooms"]
    print(f"\nHEADLINE ({HEADLINE_AGGREGATOR}, K={HEADLINE_K}), room-first over "
          f"{report['census']['n_rooms']} rooms:")
    for name in flat_stat_names():
        print(f"  {name:36s} {_ci(headline[name], 4)}")
    print(f"\nrandom baseline (pooled over {len(report['random_baseline']['seeds'])} seeds): "
          f"median e_loc "
          f"{format_number(report['random_baseline']['summary_over_repetitions']['median_e_loc']['mean'], 3)}"
          f" m")
    for name, path in published["paths"].items():
        print(f"  {name:9s} -> {path}  sha256 {published['sha256'][name]}")
    print(f"\nregister the case list with: --dump-cases {published['paths']['cases']} "
          f"--dump-cases-sha256 {published['sha256']['cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
