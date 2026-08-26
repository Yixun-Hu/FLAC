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
import io
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
    "highest-e_loc (failure) query of the headline cell (log-mean-exp, K = 8). Each quantile "
    "first fixes an e_loc VALUE -- the minimum, the lower median at index (n - 1) // 2 of the "
    "ascending errors, and the maximum -- and then names the query attaining that value with "
    "the SMALLEST global stream position. The tie-break is therefore the same in all three "
    "cases, including the highest-error one. Nothing is hand-picked")

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
#: the loader's float32 ``md['source']`` vs the float64 pair-metadata difference.
TRUTH_VECTOR_TOLERANCE = 1e-4

#: The registered protocol constants (inherited plan §1.4; K fixed by Yixun
#: 2026-08-21). A run binding that differs on any of these is a SENSITIVITY
#: CHECK, not the canonical pass, so the report refuses it unless the deviation
#: is explicitly allowed -- and then every artifact says so instead of continuing
#: to call the settings pre-registered (Codex r9 review, finding 6).
REGISTERED_PROTOCOL = {
    "tau": me.TAU,
    "k_prefixes": list(me.K_PREFIXES),
    "num_samples": me.NUM_SAMPLES,
    "seed": me.SEED,
    "noise_policy": me.REGISTERED_NOISE_POLICY,
    "steps": me.STEPS,
    "cfg_scale": me.CFG_SCALE,
    "cond_method": "vanilla",
    "scorer_readout": me.SCORER_READOUT,
    "cond_autocast": "default",
}

#: Artifact identities §1.4 pins BY DIGEST and that the published P1 binding
#: reproduces, so they are enforced here rather than merely recorded. The
#: dataset config joins them per the r9c review: it is what the observed-RIR
#: loader is built from, so a changed sample rate or size would move every
#: score while the binding still passed.
REGISTERED_ARTIFACT_SHA256 = {
    "agree_ckpt_sha256": "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787",
    "model_config_sha256": "f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9",
    "dataset_config_sha256": "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55",
}

#: The REGISTERED admissible-arm checkpoint registry (Planner RULING 1,
#: 2026-08-25) -- the sha256 of each of ``weights/exp20/{P1,BF,YAW}_40k.ckpt``.
#:
#: r9c recorded ``ckpt_sha256`` without pinning it, on the reading that §1.4's
#: ``da12748586...`` names the EMA EXTRACT while Yixun's decision 2d admits our
#: wrapped 40k checkpoints "hash-checked against their EMA extract on arrival".
#: The r9c review found that cross-check never happened -- the rsync did not
#: arrive -- so the justification did not hold and ANY checkpoint counted as
#: registered. The Planner resolved the identity BY AUTHORITY instead
#: ("P1_40k_clean_hybrid_EMA.ckpt is our trained P1 40k checkpoint") and pinned
#: the three admissible arms by their real digests. P1 is byte-identical to the
#: checkpoint the published P1 binding names.
REGISTERED_CKPT_SHA256 = {
    "P1": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
    "BF": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
    "YAW": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
}

CKPT_SHA256_NOTE = (
    "the registered admissible arms are weights/exp20/{P1,BF,YAW}_40k.ckpt, pinned by digest "
    "(Planner RULING 1, 2026-08-25). The inherited plan §1.4 names da12748586..., which is the "
    "EMA EXTRACT's digest; decision 2d admits our wrapped 40k checkpoints and Yixun resolved "
    "their identity by authority after the extract rsync never arrived, so the wrapped digests "
    "are the registered ones. A canonical report refuses any other checkpoint")

#: A run that is not the census-gated merge of every shard is not the canonical
#: pass, and the difference has to be visible in the artifact.
SINGLE_SHARD_NOTE = (
    "SINGLE-SHARD MODE: this report was built from a directory that publishes no "
    "merge_report.json, so the merge-only gates -- disjoint declared rooms whose union is the "
    "registered set, one pinned advisory batching across the whole pass, and the G1-derived "
    "source-row census -- were NOT applied. The artifact-hash joins, the identity join, the "
    "row/sidecar digests and the row-derived batching and source-row derivations still were. "
    "These numbers are a shard-local diagnostic and are not the canonical 5,337-query P1 result")

#: Why a merge receipt is re-derived rather than believed.
MERGE_DERIVATION_NOTE = (
    "merge_report.json is a RECEIPT and receipts copy: a directory assembled by hand can carry a "
    "genuine one and never have met a single merge gate (Codex r9c review, B1). So everything "
    "the receipt claims is re-derived from the rows themselves before it is believed -- the "
    "candidate-query pairs and generated waveforms from the row contents, the source-row census "
    "from the per-receiver union of the rows' own candidate index lists, and the effective "
    "batching from every row's stamp -- and the receipt has to agree with all of it. The G1 plan "
    "yields the same source-row count a second, independent way, and that must agree too")

#: How the report authenticates the continuous truth it reads.
#:
#: Planner RULING 2 (2026-08-25) settles what this can and cannot be: full
#: independence from the AcousticRooms pair-metadata tree is impossible, because
#: that tree IS the truth authority -- G1's oracle and the loader's
#: ``md['source']`` both derive from the same JSONs (``AR_md.py:31``), so the
#: probe's vector check proves the two readings agree, not that either is right.
#: The honest closure is PRE-REGISTRATION rather than provenance.
TRUTH_BINDING_NOTE = (
    "the continuous truth x*_s is pinned by no run artifact -- the engine is structurally unable "
    "to read it and G1 publishes only the oracle DISTANCE -- and it cannot be pinned by an "
    "independent witness either, because the AcousticRooms pair-metadata tree IS the authority "
    "the loader's md['source'] and G1's oracle both read (Planner RULING 2). What closes it is "
    "PRE-REGISTRATION: the metadata-bank digest is computed over that tree and committed BEFORE "
    "the merged run exists and before any localization quality has been read, so no post-hoc "
    "selection of a favourable truth is possible, and a canonical report REQUIRES that "
    "pre-registered digest. On top of it the pair file's receiver must be the candidate "
    "manifest's, the re-derived dense-grid oracle must equal the audit's (a SCALAR, and so not "
    "injective on its own), and where a loader stream exists the truth is checked as a full "
    "VECTOR -- which detects a tree edited after registration, and is circular as an origin "
    "argument, which is why it is not offered as one")

#: What a run that is not the canonical P1 result must say about itself.
NON_CANONICAL_NOTE = (
    "NON-CANONICAL: at least one gate that makes a report THE registered 5,337-query P1 result "
    "was relaxed or could not be met. The reasons are listed beside this note; every number "
    "below is a diagnostic and none may be quoted as the canonical result")

#: How the metadata-bank digest is pre-registered.
METADATA_BANK_PREREGISTRATION_NOTE = (
    "compute the digest with `python -m src.localization.meshgrid_report --print-metadata-bank-"
    "digest --context-manifest <D1> --metadata-root <tree>`, commit the value, and pass it back "
    "as --expect-metadata-bank-sha256 on every canonical run. The digest's power comes from "
    "WHEN it is committed, not from where it is computed: registered before the merge exists "
    "and before any quality is read, it makes an adversarially chosen truth impossible")

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
#: Be precise about what the margin rule can and cannot catch. If two score
#: vectors over the SAME candidates disagree about the argmax, then the leader
#: must have lost and the runner-up gained at least the gap between them, so the
#: larger of the two moves is at least ``margin / 2`` -- i.e.
#: ``margin <= 2 * deviation`` holds for EVERY possible flip, by arithmetic. That
#: is why the flag is named :data:`argmax_flip_within_2dev` and not
#: "explained_by_precision" (Codex r9 review, finding 7): the inequality states
#: which flips are POSSIBLE, never that this one came from float16 rounding.
#:
#: What actually establishes the precision claim is a separate, ABSOLUTE check
#: that runs for every cell whether or not an argmax moved: the sidecar must
#: declare and carry float16, and the recomputed aggregate must sit inside the
#: half-ulp bound a float16 quantization of the row's own similarities could
#: produce (:func:`float16_quantization_bound`). A deviation above that bound
#: means the sidecar is not a quantization of what the row was scored from, and
#: it is a hard refusal under either policy.
SIDECAR_ARGMAX_POLICIES = ("explained", "strict")
SIDECAR_ARGMAX_POLICY = "explained"

#: slack over the pure float16 half-ulp bound, absorbing the float32 rounding of
#: the two aggregations themselves (the engine's and this recompute's).
SIDECAR_FLOAT32_SLACK = 1e-5

SIDECAR_ARGMAX_NOTE = (
    "per-sample similarities are published as float16 (the engine's SIMS_PRECISION_CAVEAT), so "
    "an aggregate recomputed from the sidecar differs from the row's float32 aggregate by at "
    "most one float16 half-ulp plus float32 rounding -- an ABSOLUTE bound that is checked for "
    "every cell, so a sidecar that is not a quantization of the row's own similarities is "
    "refused even when no argmax moves. Separately, the engine's stability rule says a per-score "
    "bound eps can move a top-1 gap by 2 eps; a recomputed argmax can therefore differ from the "
    "row's whenever the row's margin is <= 2x the measured deviation, which for two score "
    "vectors over the same candidates is every flip there can be. Those cases are COUNTED and "
    "NAMED as argmax_flip_within_2dev; every published number is the row's float32 value, never "
    "the sidecar's, and the row's own recorded margin is pinned to its own scores so it cannot "
    "excuse a flip")

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
        "src/localization/meshgrid_retrieval_control.py -- built (r9b), run pending. AGREE "
        "nearest-neighbour retrieval over the real dataset RIRs that exist at each query's own "
        "receiver (other sources; the query's own pair excluded), labelled sparse/metadata-bank "
        "and never confused with the dense-grid model oracle: its candidate set is not the "
        "grid and its oracle floor is the sparse bank's own. When it has been run, its "
        "retrieval_control_handoff.json carries the numbers this entry should name",
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


def assert_artifact_hashes(binding, plan, context_manifest):
    """The files this report was HANDED are the ones the run was bound to.

    Without this the binding authenticates only itself: a second, perfectly valid
    G1 audit or D1 manifest could be passed on the command line and its geometry
    and its context draws would silently decide every number, while the report
    kept stamping the run's binding digest (Codex r9 review, finding 1). The
    three digests are recomputed exactly as ``localize_meshgrid.build_run_binding``
    computed them, so a match means byte identity and nothing weaker.
    """
    differing = {}
    found = {"d1_manifest_sha256": me.file_sha256(context_manifest),
             "g1_report_sha256": plan.report_sha256}
    for field, value in found.items():
        if binding.get(field) != value:
            differing[field] = {"binding": binding.get(field), "supplied": value}

    supplied_rooms = {room: me.file_sha256(path) for room, path in plan.rooms.items()}
    bound_rooms = binding.get("room_manifest_sha256") or {}
    if supplied_rooms != bound_rooms:
        rooms = sorted(set(supplied_rooms) | set(bound_rooms))
        first = next((room for room in rooms
                      if supplied_rooms.get(room) != bound_rooms.get(room)), None)
        differing["room_manifest_sha256"] = {
            "n_rooms_differing": sum(1 for room in rooms
                                     if supplied_rooms.get(room) != bound_rooms.get(room)),
            "first_room": first,
            "binding": bound_rooms.get(first), "supplied": supplied_rooms.get(first)}
    if differing:
        raise ValueError(
            f"the artifacts this report was given are not the ones the run was bound to: "
            f"{sorted(differing)} differ. First mismatch: {sorted(differing)[0]} = "
            f"{differing[sorted(differing)[0]]!r}. A different -- even perfectly valid -- G1 "
            "audit, D1 manifest or room manifest would decide the geometry and the context "
            "draws behind every number while the report kept naming this run's binding")
    return {"d1_manifest_sha256": found["d1_manifest_sha256"],
            "g1_report_sha256": found["g1_report_sha256"],
            "n_room_manifests": len(supplied_rooms)}


def derive_run_facts(rows):
    """Re-derive from the ROWS what a merge receipt would otherwise be believed on.

    ``source_rows`` is the engine's per-(receiver, candidate) conditioner-call
    census. ``merge_shards`` derives it from the G1 plan; here it is derived a
    second, independent way -- the union of each receiver's own published
    candidate index lists -- so a receipt that claims a number no row supports is
    caught (Codex r9c review, B1). The effective batching every row stamps is
    collected too: a directory assembled from shards run at different
    ``batch_rows`` would carry a genuine-looking receipt and mixed arithmetic.
    """
    unions, pairs, waveforms, batchings = {}, 0, 0, {}
    for row in rows:
        indices = row["candidate_indices"]
        pairs += len(indices)
        waveforms += len(indices) * int(row["num_samples"])
        unions.setdefault(str(row["receiver_id"]), set()).update(int(i) for i in indices)
        stamp = json.dumps(row.get("batching") or {}, sort_keys=True)
        batchings.setdefault(stamp, []).append(row["query_id"])
    return {"candidate_query_pairs": int(pairs),
            "generated_waveforms": int(waveforms),
            "source_rows": int(sum(len(members) for members in unions.values())),
            "n_receivers": len(unions),
            "batching_stamps": {stamp: {"n_rows": len(queries),
                                        "query_ids": sorted(queries)[:3]}
                                for stamp, queries in batchings.items()},
            "note": MERGE_DERIVATION_NOTE}


def plan_source_rows(plan, rooms=None):
    """The G1 plan's own source-row census -- ``merge_shards``'s derivation."""
    return int(sum(len(group.union)
                   for room_id in sorted(plan.rooms if rooms is None else rooms)
                   for group in me.receiver_groups(me.load_room_plan(plan, room_id))))


def assert_uniform_batching(rows, advisory):
    """Every row carries a COMPLETE batching stamp, and it is the one the run pins.

    ``merge_shards`` makes this check per row, but only a directory that actually
    went through it has been checked; a hand-assembled one has not (Codex r9c
    review, B1). Under the engine's own BATCHING_CAVEAT a changed batch shape
    moves a score by about one float16 ulp, so mixed stamps mean the cells are
    not comparable by construction.

    Completeness is the point of this revision. r9d compared only the keys a row
    happened to carry and skipped the comparison entirely for an empty stamp, so
    a re-signed row with its ``batching`` stripped -- or with ``source_chunk``
    removed -- canonicalised (Codex r9f review, B1). The engine stamps both
    advisory fields into every row it writes, so anything less is a row that was
    edited, and it is refused rather than partially checked. The run's own
    advisory must be complete for the same reason: a canonical pass states the
    batching it ran at, it does not leave it null.
    """
    fields = list(me.RUN_BINDING_ADVISORY)
    wanted = {key: (advisory or {}).get(key) for key in fields}
    unpinned = sorted(key for key in fields if wanted[key] is None)
    if unpinned:
        raise ValueError(
            f"the published binding does not pin the advisory batching {unpinned}; a run that "
            f"does not state the batch shapes it ran at cannot be compared cell to cell. "
            f"{me.BATCHING_CAVEAT}")

    incomplete = []
    stamps = {}
    for row in rows:
        stamp = row.get("batching")
        if not isinstance(stamp, dict) or sorted(stamp) != sorted(fields):
            incomplete.append({"query_id": row["query_id"],
                               "batching": stamp,
                               "missing": sorted(set(fields) - set(stamp or {}))})
            continue
        stamps.setdefault(json.dumps(stamp, sort_keys=True), []).append(row["query_id"])
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} row(s) carry no complete batching stamp (first "
            f"{incomplete[:3]}); the engine stamps {fields} into every row it writes, so a "
            "missing or partial stamp is an edited row and is refused rather than skipped")
    if not stamps:
        raise ValueError("no rows were offered to the batching check")
    if len(stamps) > 1:
        summary = {stamp: len(queries) for stamp, queries in sorted(stamps.items())}
        raise ValueError(
            f"the rows were produced at {len(stamps)} different batchings ({summary}); a merged "
            f"run states ONE. {me.BATCHING_CAVEAT}")
    found = json.loads(next(iter(stamps)))
    if found != wanted:
        raise ValueError(f"every row is stamped with batching {found} but the published binding "
                         f"pins {wanted}; the advisory values are not the ones the pass ran at")
    return {"batching": found, "advisory": wanted, "n_rows": len(rows),
            "fields": fields}


def assert_merge_report(run_dir, binding, binding_sha256, plan, totals=None, derived=None):
    """A run presented as the canonical pass carries its census-gated merge.

    ``merge_shards`` is where the merge-only gates live -- disjoint declared
    rooms whose union is the registered set, ONE pinned advisory batching across
    the whole pass, and the G1-derived source-row census. A hand-assembled
    directory can satisfy every per-row check and still have skipped all of
    them, so the merge report is required and re-joined here rather than merely
    hashed if it happens to exist (Codex r9 review, finding 1).

    ``derived`` closes the copyability the r9c review found: a receipt is
    evidence only once every number in it has been re-derived from the rows, so
    the caller passes :func:`derive_run_facts` and the receipt must agree with
    it AND with the G1 plan's own source-row census.
    """
    totals = dict(me.REGISTERED_TOTALS if totals is None else totals)
    path = os.path.join(str(run_dir), "merge_report.json")
    if not os.path.isfile(path):
        raise ValueError(
            f"{run_dir} publishes no merge_report.json. The canonical R1 report is built on the "
            "census-gated merge of every shard; a directory that was never merged has not "
            "passed the disjoint-room, pinned-advisory or source-row gates. Pass "
            "--single-shard to publish a shard-local diagnostic instead, which relaxes ONLY "
            "this requirement")
    with open(path) as handle:
        report = json.load(handle)

    reasons = []
    if report.get("ok") is not True:
        reasons.append(f"the merge report does not claim success (ok={report.get('ok')!r})")
    if report.get("binding_sha256") != binding_sha256:
        reasons.append(f"it was written under binding {str(report.get('binding_sha256'))[:12]}"
                       f"... but this directory publishes {binding_sha256[:12]}...")
    declared = sorted(report.get("declared_rooms") or [])
    if declared != sorted(plan.rooms):
        reasons.append(f"its declared rooms are not the audit's {len(plan.rooms)} "
                       f"(declared {len(declared)}, first difference "
                       f"{sorted(set(declared) ^ set(plan.rooms))[:1]})")
    bound_rooms = sorted(binding.get("declared_rooms") or [])
    if bound_rooms and bound_rooms != declared:
        reasons.append("the merged binding's declared_rooms and the merge report's disagree")
    advisory = report.get("advisory")
    if advisory != binding.get("advisory"):
        reasons.append(f"the advisory batching it pins ({advisory!r}) is not the one the merged "
                       f"binding publishes ({binding.get('advisory')!r})")
    if totals.get("queries") is not None and int(report.get("n_rows", -1)) != int(
            totals["queries"]):
        reasons.append(f"it merged {report.get('n_rows')} rows for a registered census of "
                       f"{int(totals['queries'])}")
    merged_totals = report.get("totals") or {}
    for name in ("candidate_query_pairs", "source_rows", "generated_waveforms"):
        wanted = totals.get(name)
        if wanted is None:
            continue
        if int(merged_totals.get(name, -1)) != int(wanted):
            reasons.append(f"its {name} census is {merged_totals.get(name)} for a registered "
                           f"{int(wanted)}")
    # the receipt is only evidence once the rows say the same thing
    plan_rows = plan_source_rows(plan)
    if derived is not None:
        for name in ("candidate_query_pairs", "generated_waveforms", "source_rows"):
            if int(merged_totals.get(name, -1)) != int(derived[name]):
                reasons.append(f"it claims {name} = {merged_totals.get(name)} but the rows "
                               f"themselves yield {derived[name]}")
        if int(derived["source_rows"]) != plan_rows:
            reasons.append(f"the rows' per-receiver candidate unions yield "
                           f"{derived['source_rows']} source rows but the G1 plan yields "
                           f"{plan_rows}; the two independent derivations disagree")
    elif int(merged_totals.get("source_rows", -1)) != plan_rows:
        reasons.append(f"it claims source_rows = {merged_totals.get('source_rows')} but the G1 "
                       f"plan yields {plan_rows}")

    if reasons:
        trailer = f" (and {len(reasons) - 1} more: {reasons[1:3]})" if len(reasons) > 1 else ""
        raise ValueError(f"{path} does not authenticate this directory as the canonical merged "
                         f"pass: {reasons[0]}{trailer}")
    return {"merge_report_sha256": me.file_sha256(path), "declared_rooms": declared,
            "advisory": advisory, "n_rows": int(report.get("n_rows", 0)),
            "totals": {name: merged_totals.get(name)
                       for name in ("candidate_query_pairs", "source_rows",
                                    "generated_waveforms")},
            "source_rows_derived_from": report.get("source_rows_derived_from"),
            "source_rows_from_g1_plan": plan_rows,
            "source_rows_from_rows": (None if derived is None
                                      else int(derived["source_rows"])),
            "receipt_cross_checked_against_rows": derived is not None,
            "derivation_note": MERGE_DERIVATION_NOTE}


def registered_arm(ckpt_sha256):
    """Which admissible arm a checkpoint digest is, or ``None``."""
    for arm, digest in REGISTERED_CKPT_SHA256.items():
        if str(ckpt_sha256) == digest:
            return arm
    return None


def assert_registered_protocol(binding, expect_ckpt_sha256=None, allow_deviation=False):
    """The run binding IS the registered protocol, or the deviation is declared.

    ``assert_row_protocol`` proves every row equals the binding; this proves the
    binding equals the REGISTERED constants -- tau, the K ladder, the sample
    count, the seed, the noise policy, the sampler settings, the conditioning
    method and readout, the artifact digests §1.4 pins, the dataset config the
    observed RIRs are loaded through, and the CHECKPOINT, which must be one of
    the three admissible arms (Planner RULING 1; r9c left any digest passing).
    A deviation refuses unless it is explicitly allowed, and when it is allowed
    the verdict travels into the report and the markdown so no artifact can keep
    calling the setting pre-registered.
    """
    deviations = {}
    for field, wanted in REGISTERED_PROTOCOL.items():
        found = binding.get(field)
        if isinstance(wanted, list):
            found = list(found or [])
        if found != wanted:
            deviations[field] = {"registered": wanted, "run": binding.get(field)}
    for field, wanted in REGISTERED_ARTIFACT_SHA256.items():
        if binding.get(field) != wanted:
            deviations[field] = {"registered": wanted, "run": binding.get(field)}

    arm = registered_arm(binding.get("ckpt_sha256"))
    if arm is None:
        deviations["ckpt_sha256"] = {"registered": dict(REGISTERED_CKPT_SHA256),
                                     "run": binding.get("ckpt_sha256")}
    if expect_ckpt_sha256 and binding.get("ckpt_sha256") != str(expect_ckpt_sha256):
        deviations["ckpt_sha256"] = {"registered": str(expect_ckpt_sha256),
                                     "run": binding.get("ckpt_sha256")}
    if deviations and not allow_deviation:
        raise ValueError(
            f"the run binding is not the registered protocol: {sorted(deviations)} differ. "
            f"First: {sorted(deviations)[0]} = {deviations[sorted(deviations)[0]]!r}. A report "
            "over a different tau, K ladder, seed, noise policy, sampler setting, scorer, "
            "checkpoint or pinned artifact is a SENSITIVITY CHECK, not the canonical R1 result; "
            f"pass --allow-protocol-deviation to publish it as one. {CKPT_SHA256_NOTE}")
    return {"is_registered": not deviations, "deviations": deviations,
            "checked": sorted(list(REGISTERED_PROTOCOL) + list(REGISTERED_ARTIFACT_SHA256)
                              + ["ckpt_sha256"]),
            "ckpt_sha256": binding.get("ckpt_sha256"),
            "arm": arm,
            "registered_arms": dict(REGISTERED_CKPT_SHA256),
            "ckpt_sha256_pinned": bool(arm is not None or expect_ckpt_sha256),
            "ckpt_sha256_note": CKPT_SHA256_NOTE,
            "deviation_allowed": bool(allow_deviation)}


def plan_query_identities(plan, rooms=None):
    """``{query_id: (room_id, position)}`` over the whole audit, dup-free.

    A G1 manifest that named one query twice would let that query be scored --
    and weighted -- twice, and one that quietly lost a query would produce
    metrics over 5,336 while the census still said 5,337 (Codex r9 review,
    finding 2). Duplicates are refused inside a room and across rooms.

    This is the same rule ``meshgrid_retrieval_control.query_index`` applies per
    room; kept here because the R1 report needs it over the WHOLE audit, and
    pinned against that function by a test so the two cannot drift.
    """
    identities = {}
    for room_id in sorted(plan.rooms if rooms is None else rooms):
        room_plan = me.load_room_plan(plan, room_id)
        seen = set()
        for query in room_plan.queries:
            if query.query_id in seen:
                raise ValueError(f"{room_id}: the candidate manifest names "
                                 f"{query.query_id!r} twice; the report cannot tell which entry "
                                 "authenticates the query, and a duplicate would be weighted "
                                 "twice")
            seen.add(query.query_id)
            if query.query_id in identities:
                raise ValueError(f"{query.query_id!r} is published by both "
                                 f"{identities[query.query_id][0]!r} and {room_id!r}; a query "
                                 "belongs to exactly one room")
            identities[query.query_id] = (room_id, int(query.position))
    return identities


def assert_identity_join(plan, records, rows):
    """D1 identities == G1 plan queries == published rows, exactly and once each.

    The census compares the ROWS to the D1 manifest and the room NAMES to the
    audit; nothing compared the audit's actual query set to either, so a G1 plan
    missing one query yielded partial metrics under a full-looking census
    (Codex r9 review, finding 2). This is the final join, and it runs before a
    single metric is taken.

    It costs a second parse of the room manifests -- ~330 MB in production -- and
    that is the price of the ordering: the join has to be complete before any
    number is computed, and the metric pass reads the rooms one at a time.
    """
    plan_ids = plan_query_identities(plan)
    expected = {str(record["query_id"]): (str(record["room_id"]), int(record["position"]))
                for record in records}
    published = {str(row["query_id"]): (str(row["room_id"]), int(row["position"]))
                 for row in rows}

    for label, left, right in (("G1 audit", plan_ids, expected),
                               ("published rows", published, expected)):
        missing = sorted(set(right) - set(left))
        extra = sorted(set(left) - set(right))
        if missing or extra:
            raise ValueError(
                f"the {label} does not cover exactly the registered D1 subset: {len(missing)} "
                f"registered queries are absent (first {missing[:3]}) and {len(extra)} are "
                f"present that the context manifest does not register (first {extra[:3]}). "
                "Metrics may not be taken over a set that is not the registered one")
        differing = sorted(key for key in right if left[key] != right[key])
        if differing:
            first = differing[0]
            raise ValueError(
                f"the {label} places {first!r} at {left[first]} but the context manifest "
                f"registers {right[first]}; {len(differing)} identities disagree on room or "
                "stream position")
    return {"n_queries": len(expected), "n_rooms": len({room for room, _ in expected.values()}),
            "joined": ["d1_context_manifest", "g1_candidate_manifests", "published_rows"]}


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


#: what the verify-and-parse contract promises, stated where it is relied on.
SINGLE_READ_NOTE = (
    "a row and its similarity sidecar are each read EXACTLY ONCE: the bytes are hashed and the "
    "object is parsed out of that same buffer, and the metrics path then consumes the parsed "
    "object rather than reopening the file. r9j's report verified a row and then reopened it, "
    "and reopened its sidecar again for the metrics, which left two windows in which a "
    "coordinated row+sidecar substitution could change accepted predictions and e_loc while "
    "every digest still matched (Codex r9l review, item 3). The windows are closed by "
    "construction rather than by a tighter check")


def read_verified_query_artifact(row_path, binding_sha256=None):
    """One read of the row, one of its sidecar, verified from those buffers.

    Mirrors ``meshgrid_engine.verify_query_artifact`` check for check -- and a
    test pins the two to the same verdict on a healthy artifact and on every
    tampering it distinguishes -- but returns the PARSED row and sidecar so the
    caller never has to open either again.
    """
    row_path = str(row_path)
    verdict = {"ok": False, "query_id": None, "row_path": row_path, "row": None, "sims": None}
    try:
        with open(row_path, "rb") as handle:
            raw = handle.read()
        row = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:   # noqa: BLE001 -- a verdict
        return dict(verdict, reason=f"the row is unreadable: {error}")
    verdict["query_id"] = row.get("query_id")

    out_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(row_path))))
    sims_path = os.path.join(out_dir, str(row.get("sims_path", "")))
    if not os.path.isfile(sims_path):
        return dict(verdict, reason=f"the sims sidecar {row.get('sims_path')!r} is missing")
    with open(sims_path, "rb") as handle:
        sims_raw = handle.read()
    if hashlib.sha256(sims_raw).hexdigest() != row.get("sims_sha256"):
        return dict(verdict, reason="the sims sidecar does not match the digest the row records")
    try:
        sims = np.load(io.BytesIO(sims_raw), allow_pickle=False)
    except ValueError as error:                                  # noqa: BLE001 -- a verdict
        return dict(verdict, reason=f"the sims sidecar does not load: {error}")

    shape = [int(v) for v in sims.shape] if sims.ndim == 2 else list(sims.shape)
    if shape != list(row.get("sims_shape", [])):
        return dict(verdict, reason=f"the sims sidecar is {shape}, not the recorded "
                                    f"{row.get('sims_shape')}")
    if shape != [int(row.get("n_candidates", -1)), int(row.get("num_samples", -1))]:
        return dict(verdict, reason=f"the sims sidecar is {shape} but the row declares "
                                    f"{row.get('n_candidates')} candidates x "
                                    f"{row.get('num_samples')} samples")
    prefixes = row.get("k_prefixes") or me.K_PREFIXES
    if sorted(str(k) for k in (row.get("by_k") or {})) != sorted(str(k) for k in prefixes):
        return dict(verdict, reason=f"the row publishes prefixes {sorted(row.get('by_k') or {})} "
                                    f"for a declared {list(prefixes)}")
    if "row_sha256" not in row:
        return dict(verdict, reason="the row carries no row_sha256; its predictions, oracle "
                                    "and candidate indices are unauthenticated")
    if me.row_digest(row) != row.get("row_sha256"):
        return dict(verdict, reason="the row does not match its own row_sha256; a prediction, "
                                    "oracle, candidate index, receiver or binding field was "
                                    "edited after publication")
    if binding_sha256 is not None and row.get("binding_sha256") != str(binding_sha256):
        return dict(verdict, reason=f"the row was produced under binding "
                                    f"{str(row.get('binding_sha256'))[:12]}... but this run is "
                                    f"{str(binding_sha256)[:12]}...")
    if row.get("waveform_path"):
        dump_path = os.path.join(out_dir, str(row["waveform_path"]))
        if not os.path.isfile(dump_path):
            return dict(verdict, reason=f"the waveform dump {row['waveform_path']!r} the row "
                                        "names is missing")
        with open(dump_path, "rb") as handle:
            if hashlib.sha256(handle.read()).hexdigest() != row.get("waveform_sha256"):
                return dict(verdict, reason="the waveform dump does not match the digest the "
                                            "row records")
    return dict(verdict, ok=True, reason=None, row=row, sims=sims,
                position=row.get("position"), room_id=row.get("room_id"))


def verify_rows_with_sidecars(run_dir, binding_sha256):
    """Every row AND its sidecar, parsed from the buffers that were verified.

    Returns ``(rows, sims_by_query_id)``. The sidecars are held in memory -- 142
    MB of float16 over the registered subset -- precisely so the metrics pass
    cannot reopen one.
    """
    rows, sims, rejected = [], {}, []
    for path in iter_row_paths(run_dir):
        verdict = read_verified_query_artifact(path, binding_sha256=binding_sha256)
        if not verdict["ok"]:
            rejected.append({"row": path, "reason": verdict["reason"],
                             "query_id": verdict.get("query_id")})
            continue
        row = verdict["row"]
        row["_row_path"] = path
        rows.append(row)
        sims[str(row["query_id"])] = verdict["sims"]
    if rejected:
        raise ValueError(
            f"{len(rejected)} published row(s) do not re-verify and the report refuses to "
            f"aggregate a partial artifact set; first {rejected[:3]}")
    if not rows:
        raise ValueError(f"{run_dir} publishes no rows")
    return sorted(rows, key=lambda row: int(row["position"])), sims


def verify_rows(run_dir, binding_sha256):
    """Re-accept every row from its own bytes -- row digest AND sidecar digest.

    The rows only; :func:`verify_rows_with_sidecars` is the form the evaluation
    uses, because it keeps the sidecar it verified.
    """
    return verify_rows_with_sidecars(run_dir, binding_sha256)[0]


def assert_row_matches_plan(row, query):
    """A published row IS this query's row -- from the row already in hand.

    The in-memory form of ``meshgrid_engine.assert_published_matches``, which
    re-reads the file. Same comparisons, same refusal; a test pins the two
    against each other so they cannot drift.
    """
    published = [int(i) for i in row.get("candidate_indices", [])]
    expected = [int(i) for i in query.candidate_indices]
    mismatches = [name for name, left, right in (
        ("query_id", row.get("query_id"), query.query_id),
        ("room_id", row.get("room_id"), query.room_id),
        ("position", int(row.get("position", -1)), query.position),
        ("receiver_id", row.get("receiver_id"), query.receiver_id),
        ("branch", row.get("branch"), query.branch),
        ("n_candidates", int(row.get("n_candidates", -1)), query.n_candidates),
        ("candidate_indices", published, expected)) if left != right]
    if mismatches:
        raise ValueError(f"{query.query_id}: the published row does not match the candidate "
                         f"manifest on {mismatches}; a report may not take a number out of a row "
                         "that describes a different query")
    return True


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

    def __init__(self, metadata_root, expected=None):
        self.metadata_root = str(metadata_root)
        #: ``{query_id: sha256}`` from the frozen pair-metadata bank. When given,
        #: every pair file this resolver reads must be one the bank covered AND
        #: must still hash to what the bank recorded -- so a runtime truth cannot
        #: come from a file the gate never saw, nor from one edited after it
        #: (Codex r9l review, item 1). The digest compared is the one taken from
        #: the very buffer the coordinates are parsed out of.
        self.expected = None if expected is None else dict(expected)
        self._cache = {}
        #: ``query_id -> {path, sha256}`` for every pair file this resolver read.
        #: The truth is not pinned by any run artifact, so the FILES it came out
        #: of are digested and published; :func:`metadata_bank_digest` folds them
        #: into one value an operator can register (Codex r9 review, finding 3).
        self.pair_files = {}

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
        # ONE read. The digest and the coordinates come out of the SAME buffer,
        # so there is no window between hashing a file and parsing it in which
        # the file could be swapped -- the r9i review's pair-JSON swap
        # (meshgrid_report.py:924) was exactly that window, and closing it is a
        # matter of construction rather than of a tighter check.
        with open(path, "rb") as handle:
            raw = handle.read()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"{record['query_id']}: {path} is not readable as JSON "
                             f"({error}); the truth cannot be parsed out of the bytes that "
                             "were digested") from error
        if self.expected is not None:
            query_id = str(record["query_id"])
            wanted = self.expected.get(query_id)
            if wanted is None:
                raise ValueError(
                    f"{query_id}: the pre-registered pair-metadata bank does not cover this "
                    "query, so no file on disk may supply its truth")
            if digest != str(wanted):
                raise ValueError(
                    f"{query_id}: {path} hashes to {digest[:16]}... but the pre-registered bank "
                    f"records {str(wanted)[:16]}...; the truth being read is not the registered "
                    f"one. {TRUTH_BINDING_NOTE}")
        receiver = np.asarray(payload["rec_loc"], dtype=np.float64).reshape(3)
        source = np.asarray(payload["src_loc"], dtype=np.float64).reshape(3)
        if not (np.isfinite(receiver).all() and np.isfinite(source).all()):
            raise ValueError(f"{record['query_id']}: {path} carries a non-finite coordinate")
        self.pair_files[str(record["query_id"])] = {
            "path": os.path.relpath(path, self.metadata_root),
            "sha256": digest}
        return receiver, source

    def metadata_bank_digest(self):
        """One digest over every pair file this resolver's truths came out of.

        Canonical and order-free: the map is sorted by query id and covers the
        root-relative path plus the file's exact bytes, so an edited ``src_loc``
        anywhere in the bank changes it. This is what makes the truth pinnable
        offline -- the scalar oracle check cannot separate two truths mirrored
        inside one lattice cell, and this can.
        """
        from src.localization.crossarm import canonical_sha256

        if not self.pair_files:
            raise ValueError("no pair metadata has been resolved yet; there is nothing to digest")
        return canonical_sha256({query_id: [entry["path"], entry["sha256"]]
                                 for query_id, entry in sorted(self.pair_files.items())})


def compute_metadata_bank_digest(context_manifest, metadata_root,
                                 require_manifest_census=True, records=None):
    """The pre-registration entry point: the bank digest, computed on its own.

    Deterministic and independent of any run -- it needs only the D1 manifest and
    the metadata tree -- so the value can be computed and COMMITTED before the
    merged run exists and before any localization quality has been read. That
    ordering is the whole argument (Planner RULING 2): the tree is the truth
    authority and cannot be corroborated from outside, but a digest registered
    before there are results to choose between makes an adversarially selected
    truth impossible.

    ``records`` lets a caller that has already loaded (and census-verified) the
    D1 manifest hand its records over instead of parsing the 16 MB file twice;
    the digest is identical either way, because it is a pure function of the
    query set and the tree.
    """
    if records is None:
        records = mq.load_manifest(context_manifest,
                                   require_census=require_manifest_census)["records"]
    resolver = TruthResolver(metadata_root)
    for record in records:
        resolver.resolve(record)
    return {"metadata_bank_sha256": resolver.metadata_bank_digest(),
            # per-query digests, so the runtime truth path can be held to the
            # very files the bank covered (Codex r9l review, item 1)
            "queries": {query_id: dict(entry)
                        for query_id, entry in sorted(resolver.pair_files.items())},
            "n_pair_files": len(resolver.pair_files),
            "n_records": len(records),
            "context_manifest": str(context_manifest),
            "context_manifest_sha256": me.file_sha256(context_manifest),
            "metadata_root": str(metadata_root),
            "how_to_register": METADATA_BANK_PREREGISTRATION_NOTE,
            "note": TRUTH_BINDING_NOTE}


def assert_metadata_bank(found, expected=None, allow_unpinned=False):
    """The truths came out of the PRE-REGISTERED pair-metadata bank.

    A canonical report requires ``expected``. Recording the digest and feeding it
    back on the next run proves stability, not origin (Codex r9c review, B3), so
    trust-on-first-use is not a canonical mode: without a pre-registered value
    the caller must say ``allow_unpinned`` and the whole report is stamped
    non-canonical.
    """
    if expected and str(expected) != str(found):
        raise ValueError(
            f"the pair-metadata bank this report read hashes to {str(found)[:16]}... but the "
            f"registered bank is {str(expected)[:16]}...; the continuous truths behind every "
            "e_loc, every success and every baseline error are not the registered ones")
    if not expected and not allow_unpinned:
        raise ValueError(
            "a canonical report requires the PRE-REGISTERED pair-metadata bank digest, and none "
            f"was supplied. The bank this run reads hashes to {str(found)}. "
            f"{METADATA_BANK_PREREGISTRATION_NOTE}. Recording a digest now and feeding it back "
            "later would prove only that the tree did not change in between, which is why "
            "trust-on-first-use is not a canonical mode; pass --non-canonical to publish a "
            "diagnostic instead")
    return {"metadata_bank_sha256": str(found), "pinned": bool(expected),
            "preregistration_note": METADATA_BANK_PREREGISTRATION_NOTE,
            "note": TRUTH_BINDING_NOTE}


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


def assert_grid_oracle(query_id, coordinates, published_oracle, truth,
                       tolerance=ORACLE_TOLERANCE):
    """The truth reproduces the dense-grid oracle the G1 audit published.

    The shared form of the rule ``meshgrid_retrieval_control.assert_grid_oracle``
    applies to a ``QueryPlan``; a test pins the two to the same verdict so they
    cannot drift. Its strength is stated rather than implied: the oracle is a
    SCALAR, so it catches a truth that moved off the query's own neighbourhood
    and not every possible substitution -- two truths mirrored inside one lattice
    cell share it. The injective checks are the metadata-bank digest
    (:func:`assert_metadata_bank`) and, where a loader stream exists,
    :func:`assert_truth_vector`.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    derived = float(np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1).min())
    delta = abs(derived - float(published_oracle))
    if delta > float(tolerance):
        raise ValueError(
            f"{query_id}: the oracle re-derived from the candidate block and the pair metadata "
            f"is {derived:.9f} m but the G1 manifest published {float(published_oracle):.9f} m "
            f"(|delta| = {delta:.3g} > {float(tolerance):g}); the report's ground truth is not "
            "the one the audit measured")
    return delta


def assert_truth_vector(query_id, truth, receiver, source_camera,
                        tolerance=TRUTH_VECTOR_TOLERANCE):
    """The truth is the loader's OWN target vector -- the injective check.

    ``AR_md`` builds ``md['source']`` as ``src_loc - rec_loc`` in float32, so
    ``md['source'] + receiver`` is the continuous truth expressed by a witness
    that is not the pair file. Comparing the full 3-vector (not a distance)
    cannot be satisfied by a mirrored-in-cell substitution, which is exactly the
    hole a scalar oracle leaves (Codex r9 review, finding 3).

    Reading the target here is legitimate for the same reason reading
    ``src_loc`` is: this runs post hoc, after every prediction is frozen, in a
    tool that produces no prediction.
    """
    recovered = (np.asarray(source_camera, dtype=np.float64).reshape(3)
                 + np.asarray(receiver, dtype=np.float64).reshape(3))
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    drift = float(np.abs(recovered - truth).max())
    if drift > float(tolerance):
        raise ValueError(
            f"{query_id}: the pair metadata's source {truth.tolist()} is {drift:.6g} m from the "
            f"loader's own target {recovered.tolist()} (md['source'] + receiver); the truth this "
            "report would measure against is not the one the query was held out from")
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


def float16_half_ulp(sims):
    """Half the LARGER gap to either float16 neighbour, per stored sample.

    Round-to-nearest puts the original value inside
    ``[y - gap_below / 2, y + gap_above / 2]``, and at a binade boundary those
    two gaps differ by a factor of two. ``np.spacing`` reports only one of them
    -- the gap away from zero -- which is the larger one for a positive boundary
    like ``0.5`` and the SMALLER one for a negative boundary like ``-0.5``. Using
    it alone therefore halves the bound on negative boundaries and refuses honest
    roundoff (Codex r9c review, M7). Both neighbours are consulted here.
    """
    array = np.asarray(sims, dtype=np.float16)
    if array.size == 0:
        raise ValueError("a quantization bound needs at least one sample")
    up = np.abs(np.nextafter(array, np.float16(np.inf)).astype(np.float64)
                - array.astype(np.float64))
    down = np.abs(array.astype(np.float64)
                  - np.nextafter(array, np.float16(-np.inf)).astype(np.float64))
    return float(0.5 * float(np.maximum(up, down).max()))


def float16_quantization_bound(sims, slack=SIDECAR_FLOAT32_SLACK):
    """The most a float16 sidecar can move either registered aggregate.

    Round-to-nearest puts every stored sample within half an ulp of the value the
    engine scored from (:func:`float16_half_ulp`, which takes the larger of the
    two adjacent gaps). Both aggregates are 1-Lipschitz in the sup-norm of their
    samples -- the mean obviously, and the log-mean-exp because its gradient is a
    softmax whose weights are non-negative and sum to one -- so the aggregate
    moves by at most that same half-ulp. ``slack`` absorbs the float32 rounding
    of the two aggregations themselves.

    A measured deviation ABOVE this bound is not a precision effect: it means the
    sidecar is not a float16 quantization of the similarities the row was scored
    from, which is the absolute check the r9 review found missing (finding 7).
    """
    return float(float16_half_ulp(sims) + float(slack))


def assert_sidecar_dtype(row, sims):
    """The sidecar IS the declared float16 array, in both the row and the bytes."""
    declared = row.get("sims_dtype")
    if declared != me.SIMS_DTYPE:
        raise ValueError(f"{row['query_id']}: the row declares sims_dtype {declared!r}, not the "
                         f"engine's {me.SIMS_DTYPE!r}; the float16 precision bound this report "
                         "checks against would not apply")
    found = np.asarray(sims).dtype
    if found != np.dtype(me.SIMS_DTYPE):
        raise ValueError(f"{row['query_id']}: the sidecar array is {found}, not the declared "
                         f"{me.SIMS_DTYPE}; a widened or re-encoded sidecar is not the artifact "
                         "the row's digest authenticates")
    return str(found)


def evaluate_query(row, sims, coordinates, truth, *, tau=None, radii=SUCCESS_RADII,
                   oracle_tolerance=ORACLE_TOLERANCE,
                   sidecar_argmax_policy=SIDECAR_ARGMAX_POLICY):
    """Every §2 readout for one query, from artifacts that must agree with each other.

    ``coordinates`` is the query's candidate coordinate block ``[M, 3]``, taken
    from the G1 npz (never from the row), and ``truth`` is the continuous source
    position from the pair metadata.

    Four cross-checks run before any number is kept:

    1. the row's own float32 score vector must reproduce the row's argmax under
       the registered tie-break (``argmax_by_global_index``), the candidate
       coordinate it names and the top-1 margin it records -- all exact, and none
       of them may ever fail;
    2. the sidecar must BE the declared float16 array, and each aggregate
       recomputed from it must sit inside the absolute half-ulp bound a float16
       quantization could produce -- checked whether or not an argmax moved;
    3. an argmax that does move is classified against the engine's own stability
       rule and either counted (``"explained"``) or refused (``"strict"``), and
       is never allowed to change a published number (see
       :data:`SIDECAR_ARGMAX_NOTE`);
    4. the oracle re-derived from the candidate block and the metadata truth must
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

    sims_dtype = assert_sidecar_dtype(row, sims)
    sims_f16 = np.asarray(sims, dtype=np.float16)
    sims_t = torch.as_tensor(sims_f16.astype(np.float32))
    if tuple(sims_t.shape) != (len(indices), int(row["num_samples"])):
        raise ValueError(f"{row['query_id']}: the sidecar is {tuple(sims_t.shape)} but the row "
                         f"declares ({len(indices)}, {row['num_samples']})")
    prefixes = tuple(int(k) for k in row["k_prefixes"])
    recomputed = me.nested_scores(sims_t, tau=tau, prefixes=prefixes)

    # (4) the oracle, re-derived from the candidate block and the metadata truth
    published_oracle = float(row["e_oracle"])
    oracle_delta = assert_grid_oracle(row["query_id"], coordinates, published_oracle, truth,
                                      tolerance=oracle_tolerance)
    distances = np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1)
    e_oracle = float(distances.min())
    oracle_row = int(distances.argmin())

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

            # (2) the sidecar recompute, against an ABSOLUTE float16 bound
            recomputed_scores = (recomputed[k]["scores"] if aggregator == "lme"
                                 else recomputed[k]["mean_scores"])
            deviation = float((recomputed_scores - stored).abs().max())
            bound = float16_quantization_bound(sims_f16[:, :k])
            if deviation > bound:
                raise ValueError(
                    f"{row['query_id']} K={k} {aggregator}: the aggregate recomputed from the "
                    f"sidecar differs from the row's by {deviation:.3g}, above the "
                    f"{bound:.3g} a float16 quantization of the row's own similarities could "
                    f"produce (half-ulp + {SIDECAR_FLOAT32_SLACK:g} float32 slack). The sidecar "
                    "is therefore not a quantization of what this row was scored from, and no "
                    "argmax agreement can make it one")

            recomputed_row = me.argmax_by_global_index(recomputed_scores, indices)
            margin = derived_margin
            within_2dev = bool(margin <= me.ARGMAX_STABILITY_FACTOR * deviation)
            agrees = recomputed_row == stored_row
            if not agrees and (sidecar_argmax_policy == "strict" or not within_2dev):
                raise ValueError(
                    f"{row['query_id']} K={k} {aggregator}: the argmax recomputed from the "
                    f"float16 sidecar is candidate {indices[recomputed_row]} but the row "
                    f"records {stored_index}; the row's top-1 margin is {margin:.3g} and the "
                    f"measured sidecar deviation is {deviation:.3g} "
                    f"({'within' if within_2dev else 'NOT within'} the "
                    f"{me.ARGMAX_STABILITY_FACTOR}x stability bound, policy "
                    f"{sidecar_argmax_policy!r}). {SIDECAR_ARGMAX_NOTE}")
            out["sidecar"][aggregator][k] = {
                "max_abs_delta": deviation, "argmax_agrees": bool(agrees),
                "margin": margin,
                # named for what the inequality states -- which flips are
                # POSSIBLE -- never for a cause it cannot establish (r9 finding 7)
                "argmax_flip_within_2dev": within_2dev,
                "float16_bound": bound,
                "within_float16_bound": bool(deviation <= bound),
                "sims_dtype": sims_dtype}

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


#: the latency statistics the room-first aggregation and its bootstrap cover.
LATENCY_STAT_NAMES = ("mean_seconds_per_query", "median_seconds_per_query",
                      "seconds_per_candidate", "seconds_per_generated_rir")

LATENCY_COMPLETENESS_NOTE = (
    "a row whose timings_s is missing one of the five generation components would silently "
    "under-report if the gap were read as a zero, so incomplete rows are NAMED and counted and "
    "are excluded from the aggregate rather than folded into it. n_incomplete and "
    "missing_components below say exactly which rows, which components and which rooms; the "
    "pooled totals cover the complete rows only")

LATENCY_NON_CANONICAL_NOTE = (
    "NON-CANONICAL LATENCY: at least one row was excluded for a missing timing component, so "
    "this endpoint is NOT the registered latency of the pass. Excluding rows is not neutral -- "
    "missingness can be selective (a room, a shard, a slow receiver group) and can bias the "
    "per-room means or drop a room out of the room-first average entirely (Codex r9c review, "
    "M8). The exclusions are named per component and per room below; a canonical latency "
    "endpoint requires every row to carry all five components")


def _latency_room_block(bucket):
    """One room's latency block from its complete rows."""
    seconds = np.asarray(bucket["seconds"], dtype=np.float64)
    pairs = float(sum(bucket["pairs"]))
    waveforms = float(sum(bucket["waveforms"]))
    total = float(seconds.sum())
    return {"n_queries": int(seconds.size),
            "mean_seconds_per_query": float(seconds.mean()),
            "median_seconds_per_query": float(np.median(seconds)),
            "min_seconds_per_query": float(seconds.min()),
            "max_seconds_per_query": float(seconds.max()),
            "seconds_per_candidate": total / pairs if pairs else 0.0,
            "seconds_per_generated_rir": total / waveforms if waveforms else 0.0,
            "total_seconds": total,
            "candidate_query_pairs": int(pairs),
            "generated_waveforms": int(waveforms)}


def latency_report(results, components=ROW_TIMING_COMPONENTS, seed=BOOTSTRAP_SEED,
                   n=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA):
    """Latency per query, candidate and generated RIR -- room-first, bootstrapped.

    §2 registers latency beside the localization readouts, so it is aggregated
    the same way they are: inside each room first, then averaged over rooms with
    a room-bootstrap interval (Codex r9 review, finding 8). Its draw matrix is
    built from its OWN room count, because a room can legitimately drop out of
    this readout -- if every one of its rows is missing a timing component --
    without dropping out of the metrics; when the room sets agree the matrix is
    identical to the shared one, since both come from ``(seed, n, n_rooms)``.
    """
    components = tuple(components)
    results = list(results)
    by_room, missing_components, incomplete = {}, {}, []
    totals = {name: 0.0 for name in components}
    rooms_seen = set()
    for result in results:
        rooms_seen.add(str(result["room_id"]))
        timings = result.get("latency_s") or {}
        absent = [name for name in components if name not in timings]
        if absent:
            for name in absent:
                entry = missing_components.setdefault(
                    name, {"n_rows": 0, "query_ids": [], "by_room": {}})
                entry["n_rows"] += 1
                entry["by_room"][str(result["room_id"])] = \
                    entry["by_room"].get(str(result["room_id"]), 0) + 1
                if len(entry["query_ids"]) < 5:
                    entry["query_ids"].append(result["query_id"])
            incomplete.append({"query_id": result["query_id"], "room_id": result["room_id"],
                               "missing": absent})
            continue
        seconds = 0.0
        for name in components:
            value = float(timings[name])
            totals[name] += value
            seconds += value
        bucket = by_room.setdefault(str(result["room_id"]),
                                    {"seconds": [], "pairs": [], "waveforms": []})
        bucket["seconds"].append(seconds)
        bucket["pairs"].append(int(result["n_candidates"]))
        bucket["waveforms"].append(int(result["n_candidates"]) * int(result["num_samples"]))

    if not by_room:
        raise ValueError(
            f"no row carries all of {list(components)} in timings_s ({len(incomplete)} rows are "
            "incomplete); §2 registers latency per query, candidate and generated RIR, and it "
            "may not be reported over rows whose components were read as zeros")

    per_room = {room: _latency_room_block(bucket) for room, bucket in by_room.items()}
    complete = int(sum(block["n_queries"] for block in per_room.values()))
    total = float(sum(block["total_seconds"] for block in per_room.values()))
    pairs = int(sum(block["candidate_query_pairs"] for block in per_room.values()))
    waveforms = int(sum(block["generated_waveforms"] for block in per_room.values()))
    rooms_present = sorted(per_room)
    rooms_dropped = sorted(rooms_seen - set(rooms_present))
    canonical = not incomplete
    return {
        # a latency endpoint built on a subset of the rows is never a clean
        # canonical block, however few were dropped (Codex r9c review, M8)
        "canonical": canonical,
        "non_canonical_note": None if canonical else LATENCY_NON_CANONICAL_NOTE,
        "n_queries": complete,
        "n_rows_offered": len(results),
        "n_incomplete": len(incomplete),
        "incomplete_rows": incomplete[:10],
        "missing_components": missing_components,
        "rooms_without_a_complete_row": rooms_dropped,
        "candidate_query_pairs": pairs,
        "generated_waveforms": waveforms,
        "total_seconds": total,
        "per_room": per_room,
        "across_rooms": across_rooms(per_room, names=list(LATENCY_STAT_NAMES), draws=None,
                                     seed=seed, n=n, alpha=alpha),
        "pooled": {"seconds_per_query": {
                       "mean": total / complete if complete else 0.0,
                       "total": total},
                   "seconds_per_candidate": total / pairs if pairs else 0.0,
                   "seconds_per_generated_rir": total / waveforms if waveforms else 0.0,
                   "label": "pooled over queries (secondary; the registered aggregation is "
                            "room-first)"},
        "component_seconds": {name: float(totals[name]) for name in components},
        "component_fraction": {name: (float(totals[name]) / total if total else 0.0)
                               for name in components},
        "components": list(components),
        "completeness_note": LATENCY_COMPLETENESS_NOTE,
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
    registered = seeds == [int(s) for s in RANDOM_BASELINE_SEEDS]
    return {
        "rule": ("uniform draw over the query's IDENTICAL published valid candidate set; the "
                 "draw is keyed by sha256(seed, query_id) so it is independent of iteration "
                 "order, and each seed is one independent full repetition"
                 + (". The seeds below ARE the pre-registered ones" if registered else
                    ". SENSITIVITY CHECK: these are NOT the pre-registered seeds "
                    f"{list(RANDOM_BASELINE_SEEDS)}")),
        "seeds": seeds,
        # stated, not implied: an artifact may not call a setting pre-registered
        # when it is not (Codex r9 review, finding 6)
        "seeds_are_registered": bool(registered),
        "registered_seeds": [int(s) for s in RANDOM_BASELINE_SEEDS],
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

    Each quantile fixes an e_loc VALUE first -- the minimum, the lower median at
    ``(n - 1) // 2`` of the ascending errors, and the maximum -- and then names
    the query attaining it with the SMALLEST global stream position. Doing it in
    that order is what makes the tie-break uniform: taking the last element of an
    ascending sort would hand the highest-error case to the LARGEST tied
    position, contradicting the rule the report prints beside the table (Codex r9
    review, finding 10).
    """
    results = list(results)
    if not results:
        raise ValueError("visualization cases need at least one scored query")
    errors = sorted(float(r["by"][aggregator][k]["e_loc"]) for r in results)
    wanted = (("lowest_e_loc", errors[0]),
              ("median_e_loc", errors[(len(errors) - 1) // 2]),
              ("highest_e_loc", errors[-1]))
    cases = []
    for label, value in wanted:
        attaining = [r for r in results
                     if float(r["by"][aggregator][k]["e_loc"]) == value]
        result = min(attaining, key=lambda r: int(r["position"]))
        entry = result["by"][aggregator][k]
        cases.append({"quantile": label, "e_loc_quantile_value": float(value),
                      "n_attaining": len(attaining), "n_ranked": len(results),
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
            "quantile": case["quantile"],
            "e_loc_quantile_value": case["e_loc_quantile_value"],
            "n_attaining": case["n_attaining"], "n_ranked": case["n_ranked"],
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
        "sims_precision_caveat": me.SIMS_PRECISION_CAVEAT,
        # every artifact this round emits carries the same disclosures, so a case
        # file read on its own cannot lose them (Codex r9 review, finding 10)
        "latency_scope_note": LATENCY_SCOPE_NOTE,
        "controls_elsewhere": CONTROLS_ELSEWHERE,
        "cases": payload_cases,
    }


# --------------------------------------------------------------------------- #
# the whole report
# --------------------------------------------------------------------------- #
def evaluate_run(run_dir, audit_report, context_manifest, metadata_root, totals=None,
                 radii=SUCCESS_RADII, sidecar_argmax_policy=SIDECAR_ARGMAX_POLICY,
                 baseline_seeds=RANDOM_BASELINE_SEEDS, oracle_tolerance=ORACLE_TOLERANCE,
                 require_manifest_census=True, single_shard=False,
                 expect_ckpt_sha256=None, expect_metadata_bank_sha256=None,
                 allow_protocol_deviation=False, allow_unpinned_metadata_bank=False,
                 source_provider=None, on_query=None):
    """Gate the artifacts, then evaluate every query. Returns the raw material.

    The gates run first and in full, in this order, and every one of them is a
    refusal:

    1. the published binding is recomputed from its own content;
    2. the D1 manifest, the G1 report and every room manifest THIS REPORT WAS
       HANDED are the ones the binding pins (``assert_artifact_hashes``);
    3. the binding is the registered protocol, including the admissible-arm
       checkpoint registry (``assert_registered_protocol``);
    4. every row and sidecar re-verifies against its own digest, and every row
       carries the SAME effective batching, which is the one the run pins
       (``assert_uniform_batching``);
    5. the merge receipt is re-derived from the rows -- pairs, waveforms and the
       per-receiver source-row union -- and must agree with them and with the G1
       plan (``derive_run_facts`` + ``assert_merge_report``), unless
       ``single_shard`` explicitly downgrades the report to a shard-local
       diagnostic, which relaxes ONLY the receipt and never a hash join or a
       derivation;
    6. the census holds, per room and in total;
    7. the D1 identities, the G1 plan's queries and the published rows are the
       SAME set, once each (``assert_identity_join``);
    8. the pair-metadata bank is the PRE-REGISTERED one
       (``assert_metadata_bank``), which a canonical report requires.

    Only then does a single room-major pass compute anything, and inside a room
    every row is authenticated against the G1 plan before that room's first
    metric is taken.

    ``source_provider`` is an optional ``query_id -> md['source']`` callable. When
    supplied, each truth is additionally checked as a VECTOR against the loader's
    own target (:func:`assert_truth_vector`) -- the injective check the scalar
    oracle cannot make. The report records whether it ran.

    ``require_manifest_census`` exists for fixtures, exactly as
    ``meshgrid_queries.build_manifest``'s does, and is never relaxed by the
    production path.
    """
    run_dir = str(run_dir)
    binding, binding_sha = load_published_binding(run_dir)
    plan = me.load_audit_plan(audit_report, branch=binding["branch"])
    artifacts = assert_artifact_hashes(binding, plan, context_manifest)
    registered = assert_registered_protocol(binding, expect_ckpt_sha256=expect_ckpt_sha256,
                                            allow_deviation=allow_protocol_deviation)

    manifest = mq.load_manifest(context_manifest, require_census=require_manifest_census)
    records = manifest["records"]
    rows, sims_by_id = verify_rows_with_sidecars(run_dir, binding_sha)
    # the receipt is checked against the ROWS, so the rows come first now
    derived = derive_run_facts(rows)
    batching = assert_uniform_batching(rows, binding.get("advisory"))
    merge = (None if single_shard
             else assert_merge_report(run_dir, binding, binding_sha, plan, totals=totals,
                                      derived=derived))
    census = assert_census(rows, records, totals=totals)
    protocol = assert_row_protocol(rows, binding)

    if sorted(plan.rooms) != census["rooms"]:
        raise ValueError(f"the audit publishes rooms {sorted(plan.rooms)[:3]}... but the run "
                         f"publishes {census['rooms'][:3]}...; a report joins one audit to one "
                         "run")
    identity_join = assert_identity_join(plan, records, rows)

    rows_by_id = {row["query_id"]: row for row in rows}
    records_by_id = {record["query_id"]: record for record in records}
    resolver = TruthResolver(metadata_root)
    seeds = [int(s) for s in baseline_seeds]

    results, plans_by_id, receiver_drift, truth_drift = [], {}, [], []
    for room_id in sorted(plan.rooms):
        room_plan = me.load_room_plan(plan, room_id)
        # phase A: every row of this room is authenticated against the G1 plan
        # before phase B takes a single number out of it -- from the row already
        # verified and parsed, never by reopening the file (Codex r9l, item 3)
        for query in room_plan.queries:
            assert_row_matches_plan(rows_by_id[query.query_id], query)
        for query in room_plan.queries:
            row = rows_by_id[query.query_id]
            metadata_receiver, truth = resolver.resolve(records_by_id[query.query_id])
            receiver_drift.append(assert_receiver_matches(query.query_id, metadata_receiver,
                                                          query.receiver_xyz))
            if source_provider is not None:
                truth_drift.append(assert_truth_vector(
                    query.query_id, truth, query.receiver_xyz,
                    source_provider(query.query_id)))
            # the sidecar parsed out of the bytes that were verified -- there is
            # no second open of it anywhere in this function
            sims = sims_by_id[query.query_id]
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

    metadata_bank = assert_metadata_bank(resolver.metadata_bank_digest(),
                                         expected=expect_metadata_bank_sha256,
                                         allow_unpinned=allow_unpinned_metadata_bank)
    results.sort(key=lambda result: int(result["position"]))
    return {"binding": binding, "binding_sha256": binding_sha, "plan": plan,
            "manifest": manifest, "records": records, "rows": rows,
            "rows_by_id": rows_by_id, "plans_by_id": plans_by_id, "results": results,
            "census": census, "protocol": protocol, "artifacts": artifacts,
            "registered_protocol": registered, "merge": merge, "derived": derived,
            "batching": batching,
            "single_shard": bool(single_shard), "identity_join": identity_join,
            "metadata_bank": metadata_bank, "sims_by_id": sims_by_id,
            "single_read_note": SINGLE_READ_NOTE,
            "truth_vector_checked": source_provider is not None,
            "max_truth_vector_drift_m": (float(max(truth_drift)) if truth_drift else None),
            "max_receiver_drift_m": float(max(receiver_drift)) if receiver_drift else 0.0,
            "baseline_seeds": seeds}


def canonical_status(evaluated, report=None):
    """Whether this is THE registered result, and every reason it is not.

    One place decides it, so the JSON, the markdown and the console can never
    disagree about whether a number may be quoted as canonical.
    """
    reasons = []
    if evaluated["single_shard"]:
        reasons.append({"gate": "merge_report",
                        "why": "the directory publishes no census-gated merge receipt",
                        "note": SINGLE_SHARD_NOTE})
    if not evaluated["registered_protocol"]["is_registered"]:
        reasons.append({"gate": "registered_protocol",
                        "why": f"the run binding deviates on "
                               f"{sorted(evaluated['registered_protocol']['deviations'])}",
                        "note": CKPT_SHA256_NOTE})
    if not evaluated["metadata_bank"]["pinned"]:
        reasons.append({"gate": "metadata_bank",
                        "why": "no pre-registered pair-metadata bank digest was supplied",
                        "note": METADATA_BANK_PREREGISTRATION_NOTE})
    if report is not None:
        if not report["protocol"]["bootstrap"]["is_registered"]:
            reasons.append({"gate": "bootstrap",
                            "why": "the room bootstrap is not the pre-registered seed x n",
                            "note": None})
        if not report["protocol"]["baseline_seeds_are_registered"]:
            reasons.append({"gate": "baseline_seeds",
                            "why": "the random baseline did not run the pre-registered seeds",
                            "note": None})
        if not report["latency"]["canonical"]:
            reasons.append({"gate": "latency_completeness",
                            "why": f"{report['latency']['n_incomplete']} row(s) lack a timing "
                                   "component, so the latency endpoint is non-canonical",
                            "note": LATENCY_NON_CANONICAL_NOTE})
    return {"canonical": not reasons, "reasons": reasons,
            "note": None if not reasons else NON_CANONICAL_NOTE}


def sidecar_summary(results):
    """What the float16 recompute found -- counted and named, never absorbed."""
    out = {"policy_note": SIDECAR_ARGMAX_NOTE, "by_cell": {}}
    named = []
    for aggregator in AGGREGATORS:
        out["by_cell"][aggregator] = {}
        for k in sorted(results[0]["sidecar"][aggregator]):
            deltas = np.asarray([r["sidecar"][aggregator][k]["max_abs_delta"]
                                 for r in results], dtype=np.float64)
            bounds = np.asarray([r["sidecar"][aggregator][k]["float16_bound"]
                                 for r in results], dtype=np.float64)
            disagreements = [r for r in results
                             if not r["sidecar"][aggregator][k]["argmax_agrees"]]
            out["by_cell"][aggregator][str(k)] = {
                "n_queries": len(results),
                "max_abs_delta": float(deltas.max()), "mean_abs_delta": float(deltas.mean()),
                "max_float16_bound": float(bounds.max()),
                "max_delta_over_bound": float((deltas / bounds).max()),
                "all_within_float16_bound": bool(all(
                    r["sidecar"][aggregator][k]["within_float16_bound"] for r in results)),
                "n_argmax_disagreements": len(disagreements),
                "all_flips_within_2dev": all(
                    r["sidecar"][aggregator][k]["argmax_flip_within_2dev"]
                    for r in disagreements)}
            for result in disagreements:
                entry = result["sidecar"][aggregator][k]
                named.append({"query_id": result["query_id"], "aggregator": aggregator,
                              "k": int(k), "margin": entry["margin"],
                              "sidecar_max_abs_delta": entry["max_abs_delta"],
                              "float16_bound": entry["float16_bound"],
                              "argmax_flip_within_2dev": entry["argmax_flip_within_2dev"]})
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
            "metadata_bank": evaluated["metadata_bank"],
            "artifact_hash_join": evaluated["artifacts"],
            "merge": evaluated["merge"],
            "merge_report_sha256": ((evaluated["merge"] or {}).get("merge_report_sha256")
                                    if evaluated["merge"] else
                                    (me.file_sha256(merge_path)
                                     if os.path.isfile(merge_path) else None)),
            "single_shard": evaluated["single_shard"],
            "single_shard_note": SINGLE_SHARD_NOTE if evaluated["single_shard"] else None,
            "derived_from_rows": evaluated["derived"],
            "effective_batching": evaluated["batching"],
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
                          "method": "percentile (linear interpolation)",
                          # stated, not implied (Codex r9 review, finding 6)
                          "is_registered": bool(int(bootstrap_seed) == BOOTSTRAP_SEED
                                                and int(n_boot) == BOOTSTRAP_N),
                          "registered": {"seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_N}},
            "headline_cell": {"aggregator": HEADLINE_AGGREGATOR, "k": HEADLINE_K},
            "baseline_seeds": [int(s) for s in evaluated["baseline_seeds"]],
            "baseline_seeds_are_registered":
                bool([int(s) for s in evaluated["baseline_seeds"]]
                     == [int(s) for s in RANDOM_BASELINE_SEEDS]),
            "registered_protocol": evaluated["registered_protocol"],
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
            "supplied_artifacts_match_the_binding_hashes": True,
            "binding_matches_the_registered_protocol":
                evaluated["registered_protocol"]["is_registered"],
            "merge_report_gates_applied": not evaluated["single_shard"],
            "merge_receipt_rederived_from_rows": bool(
                (evaluated["merge"] or {}).get("receipt_cross_checked_against_rows")),
            "effective_batching_uniform_and_pinned": True,
            "g1_audit_chain_reverified": True,
            "d1_manifest_stream_and_census_reverified": True,
            "rows_and_sidecars_digest_verified": True,
            "sidecar_dtype_and_float16_bound_checked": True,
            "rows_authenticated_against_g1_plan": True,
            "row_protocol_matches_binding": True,
            "d1_g1_rows_identity_join": evaluated["identity_join"],
            "metadata_bank_pinned": evaluated["metadata_bank"]["pinned"],
            "truth_vector_checked_against_the_loader":
                evaluated["truth_vector_checked"],
            "max_truth_vector_drift_m": evaluated["max_truth_vector_drift_m"],
            "max_receiver_drift_m": evaluated["max_receiver_drift_m"],
            "truth_binding_note": TRUTH_BINDING_NOTE,
        },
        "metrics": metrics,
        "oracle": oracle_report(results, draws=draws, **bootstrap),
        "latency": latency_report(results, seed=bootstrap_seed, n=n_boot, alpha=alpha),
        "random_baseline": baseline_report(results, seeds=evaluated["baseline_seeds"],
                                           draws=draws, radii=radii, bootstrap=bootstrap),
        "associations": association_report(results),
        "crosscheck": {"sidecar": sidecar_summary(results),
                       "oracle": oracle_crosscheck_summary(results)},
    }
    report["visualization_cases"] = select_visualization_cases(results)
    # one authority for "may this be quoted as the registered result", read by
    # the JSON, the markdown and the console alike
    report["canonical_status"] = canonical_status(evaluated, report)
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


def _sensitivity_banners(report):
    """Loud, unmissable lines whenever a setting is not the registered one.

    Mirrors the retrieval control's banner path: an artifact may not keep calling
    a setting pre-registered once it is not (Codex r9 review, finding 6).
    """
    protocol, provenance = report["protocol"], report["provenance"]
    lines = []
    status = report["canonical_status"]
    if not status["canonical"]:
        lines.append(f"> **{status['note']}**")
        lines.append(">")
        for reason in status["reasons"]:
            lines.append(f"> - `{reason['gate']}` — {reason['why']}")
        lines.append("")
    registered = protocol["registered_protocol"]
    if not registered["is_registered"]:
        lines.append(f"> **SENSITIVITY CHECK, not the registered protocol:** the run binding "
                     f"differs from the registered constants on "
                     f"{sorted(registered['deviations'])}. "
                     f"{ {k: v for k, v in list(registered['deviations'].items())[:3]} }")
        lines.append("")
    if not protocol["bootstrap"]["is_registered"]:
        lines.append(f"> **SENSITIVITY CHECK, not the registered bootstrap:** the pre-registered "
                     f"settings are seed {protocol['bootstrap']['registered']['seed']} x "
                     f"{protocol['bootstrap']['registered']['n_boot']:,} resamples.")
        lines.append("")
    if not protocol["baseline_seeds_are_registered"]:
        lines.append(f"> **SENSITIVITY CHECK, not the registered baseline seeds:** the "
                     f"pre-registered seeds are {list(RANDOM_BASELINE_SEEDS)}.")
        lines.append("")
    if provenance["single_shard"]:
        lines.append(f"> **{provenance['single_shard_note']}**")
        lines.append("")
    if not provenance["metadata_bank"]["pinned"]:
        lines.append(f"> **NON-CANONICAL — the pair-metadata bank is not PRE-REGISTERED:** it "
                     f"hashes to `{provenance['metadata_bank']['metadata_bank_sha256']}`. "
                     "Recording it here and feeding it back later would prove only that the tree "
                     "did not change in between, not where the truth came from. "
                     f"{provenance['metadata_bank']['preregistration_note']}")
        lines.append("")
    if not report["gates"]["truth_vector_checked_against_the_loader"]:
        lines.append("> **The injective truth check did not run here:** no loader stream was "
                     "supplied, so md['source'] + receiver was not compared against the pair "
                     "metadata. The off-grid probe runs that check on its sixteen queries.")
        lines.append("")
    return lines


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
                 f"{protocol['bootstrap']['n_boot']:,} room resamples at seed "
                 f"{protocol['bootstrap']['seed']}.")
    lines.append("")
    lines.extend(_sensitivity_banners(report))

    for aggregator in AGGREGATORS:
        lines.append(f"## {aggregator.upper()} — {report['labels']['aggregator_roles'][aggregator]}")
        lines.append("")
        lines.append(_stamp(report))
        lines.append("")
        header = ("| K | median e_loc (m) | mean e_loc (m) | median e_excess (m) | "
                  "mean e_excess (m) | success@0.5 | success@1.0 | oracle-norm@0.5 | "
                  "oracle-norm@1.0 |")
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for k in protocol["k_prefixes"]:
            cell = report["metrics"][aggregator][str(k)]["across_rooms"]
            lines.append(
                f"| {k} | {_ci(cell['median_e_loc'], 3)} | {_ci(cell['mean_e_loc'], 3)} | "
                f"{_ci(cell['median_e_excess'], 3)} | {_ci(cell['mean_e_excess'], 3)} | "
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
    lines.append("| seed | median e_loc (m) | mean e_loc (m) | success@0.5 | success@1.0 | "
                 "oracle-norm@0.5 | oracle-norm@1.0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for repetition in baseline["repetitions"]:
        cell = repetition["across_rooms"]
        lines.append(f"| {repetition['seed']} | {_ci(cell['median_e_loc'], 3)} | "
                     f"{_ci(cell['mean_e_loc'], 3)} | {_ci(cell['success_raw@0.5'], 4)} | "
                     f"{_ci(cell['success_raw@1.0'], 4)} | "
                     f"{_ci(cell['success_oracle_normalized@0.5'], 4)} | "
                     f"{_ci(cell['success_oracle_normalized@1.0'], 4)} |")
    summary = baseline["summary_over_repetitions"]
    cells = " | ".join(
        f"{format_number(summary[name]['mean'], 4)} ± {format_number(summary[name]['sd'], 4)}"
        for name in ("median_e_loc", "mean_e_loc", "success_raw@0.5", "success_raw@1.0",
                     "success_oracle_normalized@0.5", "success_oracle_normalized@1.0"))
    lines.append(f"| **pooled** | {cells} |")
    lines.append("")

    latency = report["latency"]
    across = latency["across_rooms"]
    lines.append("## Latency — room-first"
                 + ("" if latency["canonical"] else " (NON-CANONICAL)"))
    lines.append("")
    lines.append(_stamp(report))
    lines.append("")
    if not latency["canonical"]:
        lines.append(f"> **{latency['non_canonical_note']}**")
        lines.append("")
    lines.append("| statistic | room-first point [95% CI] |")
    lines.append("|---|---|")
    lines.append(f"| mean s / query | {_ci(across['mean_seconds_per_query'], 4)} |")
    lines.append(f"| median s / query | {_ci(across['median_seconds_per_query'], 4)} |")
    lines.append(f"| ms / candidate | "
                 f"{format_number(across['seconds_per_candidate']['point'] * 1e3, 4)} "
                 f"[{format_number(across['seconds_per_candidate']['ci_lo'] * 1e3, 4)}, "
                 f"{format_number(across['seconds_per_candidate']['ci_hi'] * 1e3, 4)}] |")
    lines.append(f"| ms / generated RIR | "
                 f"{format_number(across['seconds_per_generated_rir']['point'] * 1e3, 4)} "
                 f"[{format_number(across['seconds_per_generated_rir']['ci_lo'] * 1e3, 4)}, "
                 f"{format_number(across['seconds_per_generated_rir']['ci_hi'] * 1e3, 4)}] |")
    lines.append("")
    lines.append(f"Pooled (secondary): {format_number(latency['pooled']['seconds_per_candidate'] * 1e3, 4)}"
                 f" ms / candidate over {latency['n_queries']:,} complete rows.")
    if latency["n_incomplete"]:
        lines.append("")
        lines.append(f"**{latency['n_incomplete']:,} of {latency['n_rows_offered']:,} row(s) "
                     f"were excluded for a missing timing component:**")
        lines.append("")
        lines.append("| component | rows missing it | by room |")
        lines.append("|---|---|---|")
        for name, block in sorted(latency["missing_components"].items()):
            by_room = ", ".join(f"{room} x{count}"
                                for room, count in sorted(block["by_room"].items()))
            lines.append(f"| {name} | {block['n_rows']:,} | {by_room} |")
        lines.append("")
        if latency["rooms_without_a_complete_row"]:
            lines.append(f"Rooms with NO complete row (dropped from the room-first average): "
                         f"{latency['rooms_without_a_complete_row']}")
            lines.append("")
        lines.append(f"First offenders: "
                     f"{[entry['query_id'] for entry in latency['incomplete_rows'][:3]]}. "
                     f"{latency['completeness_note']}")
    lines.append("")
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
    flips = crosscheck["sidecar"]["argmax_disagreements"]
    worst_ratio = max((block["max_delta_over_bound"]
                       for by_k in crosscheck["sidecar"]["by_cell"].values()
                       for block in by_k.values()), default=0.0)
    lines.append(f"- float16 sidecar: every cell is inside the absolute half-ulp bound "
                 f"(worst deviation is {format_number(worst_ratio, 3)}x the bound), and "
                 f"{crosscheck['sidecar']['n_argmax_disagreements']} argmax disagreement(s) "
                 f"over {len(AGGREGATORS) * len(protocol['k_prefixes'])} cells are all within "
                 f"the 2x stability bound: "
                 f"{format_number(all(entry['argmax_flip_within_2dev'] for entry in flips) if flips else True)}")
    lines.append(f"- max receiver drift (pair metadata vs candidate manifest): "
                 f"{report['gates']['max_receiver_drift_m']:.3g} m")
    lines.append(f"- pair-metadata bank: "
                 f"`{provenance['metadata_bank']['metadata_bank_sha256'][:16]}...` "
                 f"({'pinned' if provenance['metadata_bank']['pinned'] else 'recorded only'})")
    lines.append(f"- injective truth-vector check against the loader: "
                 f"{format_number(report['gates']['truth_vector_checked_against_the_loader'])}")
    lines.append(f"- artifact hash join (D1 / G1 / room manifests vs the binding): passed over "
                 f"{provenance['artifact_hash_join']['n_room_manifests']} room manifests")
    lines.append(f"- identity join (D1 == G1 == rows): "
                 f"{report['gates']['d1_g1_rows_identity_join']['n_queries']:,} queries over "
                 f"{report['gates']['d1_g1_rows_identity_join']['n_rooms']} rooms")
    lines.append("")

    lines.append("## §2 controls that are NOT in this report")
    lines.append("")
    for name, where in sorted(report["controls_elsewhere"].items()):
        lines.append(f"- **{name}** — {where}")
    lines.append("")
    lines.append(f"_Latency scope for every table above:_ {latency['scope_note']}")
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
    parser.add_argument("--run-dir", default=None,
                        help="the MERGED I1 run directory (rows/, sidecars, run_binding.json). "
                             "Required for a report; omitted by "
                             "--print-metadata-bank-digest, which needs no run")
    parser.add_argument("--print-metadata-bank-digest", action="store_true",
                        help="PRE-REGISTRATION MODE: compute the pair-metadata bank digest from "
                             "--context-manifest and --metadata-root, print it and exit. Commit "
                             "the value BEFORE the merged run exists, then pass it back as "
                             "--expect-metadata-bank-sha256 on every canonical run")
    parser.add_argument("--audit-report",
                        default=os.path.join("outputs_loc", "exp22", "g1_audit",
                                             "geometry_audit_report.json"))
    parser.add_argument("--context-manifest",
                        default=os.path.join("outputs_loc", "exp22",
                                             "d1_context_manifest.json"))
    parser.add_argument("--metadata-root",
                        default=os.path.join("AcousticRooms", "metadata"),
                        help="the dataset metadata root the continuous truth is read from")
    parser.add_argument("--out-dir", default=None,
                        help="where the report is published; required unless "
                             "--print-metadata-bank-digest")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--baseline-seeds", type=int, nargs="+",
                        default=list(RANDOM_BASELINE_SEEDS))
    parser.add_argument("--sidecar-argmax-policy", default=SIDECAR_ARGMAX_POLICY,
                        choices=list(SIDECAR_ARGMAX_POLICIES))
    parser.add_argument("--single-shard", action="store_true",
                        help="publish a SHARD-LOCAL diagnostic from a directory that carries no "
                             "merge_report.json. Relaxes only the merge-only gates; the "
                             "artifact-hash joins, the identity join and every digest still "
                             "apply, and the artifact is stamped as non-canonical")
    parser.add_argument("--expect-ckpt-sha256", default=None,
                        help="narrow the admissible checkpoint to exactly this digest; by "
                             "default any of the three registered arms is accepted")
    parser.add_argument("--expect-metadata-bank-sha256", default=None,
                        help="the PRE-REGISTERED pair-metadata bank digest the continuous truths "
                             "must come out of. Required for a canonical report; obtain it with "
                             "--print-metadata-bank-digest and commit it before any result "
                             "exists")
    parser.add_argument("--non-canonical", action="store_true",
                        help="publish a diagnostic without a pre-registered metadata-bank "
                             "digest. Trust-on-first-use is not a canonical mode, so the report "
                             "and the markdown are stamped NON-CANONICAL throughout")
    parser.add_argument("--allow-protocol-deviation", action="store_true",
                        help="publish even though the run binding is not the registered "
                             "protocol; the report and the markdown are then stamped as a "
                             "sensitivity check throughout")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """The registered settings are the defaults; a change must be DECLARED.

    A deviation is no longer a console note that scrolls away: it is refused
    unless ``--allow-protocol-deviation`` is passed, and when it is passed the
    published artifacts stop calling the setting pre-registered (Codex r9 review,
    finding 6).
    """
    if args.print_metadata_bank_digest:
        if args.run_dir or args.out_dir:
            print("NOTE: --print-metadata-bank-digest needs neither --run-dir nor --out-dir; "
                  "they are ignored")
        return True
    for name in ("run_dir", "out_dir"):
        if not getattr(args, name):
            _refuse(f"--{name.replace('_', '-')} is required to publish a report")
    if not args.expect_metadata_bank_sha256 and not args.non_canonical:
        _refuse("a canonical report requires the PRE-REGISTERED pair-metadata bank digest. "
                f"{METADATA_BANK_PREREGISTRATION_NOTE}. Pass --non-canonical to publish a "
                "diagnostic instead")
    deviating = []
    if int(args.bootstrap_seed) != BOOTSTRAP_SEED or int(args.n_boot) != BOOTSTRAP_N:
        deviating.append(f"the bootstrap ({args.bootstrap_seed} x {args.n_boot} vs the "
                         f"pre-registered {BOOTSTRAP_SEED} x {BOOTSTRAP_N})")
    if [int(s) for s in args.baseline_seeds] != list(RANDOM_BASELINE_SEEDS):
        deviating.append(f"the baseline seeds ({list(args.baseline_seeds)} vs the "
                         f"pre-registered {list(RANDOM_BASELINE_SEEDS)})")
    if deviating and not args.allow_protocol_deviation:
        _refuse(f"{' and '.join(deviating)} are not the pre-registered settings. Pass "
                "--allow-protocol-deviation to publish this as a sensitivity check; the "
                "artifacts will be stamped as one throughout")
    for line in deviating:
        print(f"SENSITIVITY CHECK: {line}")
    if int(args.n_boot) < 1:
        _refuse("--n-boot must be at least 1")
    return True


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.print_metadata_bank_digest:
        verdict = compute_metadata_bank_digest(args.context_manifest, args.metadata_root)
        print(json.dumps(jsonable(verdict), indent=2, sort_keys=True))
        print(f"\nmetadata_bank_sha256 = {verdict['metadata_bank_sha256']}")
        print(f"  over {verdict['n_pair_files']:,} pair files for "
              f"{verdict['n_records']:,} registered queries")
        print(f"\n{METADATA_BANK_PREREGISTRATION_NOTE}")
        return 0
    print(f"AGREE LEAKAGE CAVEAT: {me.AGREE_LEAKAGE_CAVEAT}")
    print(f"SUBSET: {SUBSET_LABEL}")
    if args.single_shard:
        print(f"\n{SINGLE_SHARD_NOTE}\n")

    evaluated = evaluate_run(args.run_dir, args.audit_report, args.context_manifest,
                             args.metadata_root, baseline_seeds=args.baseline_seeds,
                             sidecar_argmax_policy=args.sidecar_argmax_policy,
                             single_shard=args.single_shard,
                             expect_ckpt_sha256=args.expect_ckpt_sha256,
                             expect_metadata_bank_sha256=args.expect_metadata_bank_sha256,
                             allow_protocol_deviation=args.allow_protocol_deviation,
                             allow_unpinned_metadata_bank=args.non_canonical)
    print(f"gates passed: binding {evaluated['binding_sha256'][:12]}..., "
          f"{evaluated['census']['n_queries']:,} queries / "
          f"{evaluated['census']['n_rooms']} rooms, identity join over "
          f"{evaluated['identity_join']['n_queries']:,} D1 == G1 == row identities")
    print(f"pair-metadata bank {evaluated['metadata_bank']['metadata_bank_sha256']} "
          f"({'PINNED' if evaluated['metadata_bank']['pinned'] else 'recorded only'})")
    report = build_report(evaluated, args.run_dir, args.audit_report, args.context_manifest,
                          args.metadata_root, bootstrap_seed=args.bootstrap_seed,
                          n_boot=args.n_boot,
                          sidecar_argmax_policy=args.sidecar_argmax_policy)
    cases = build_case_payload(report["visualization_cases"], evaluated["rows_by_id"],
                               evaluated["plans_by_id"],
                               {r["query_id"]: r for r in evaluated["results"]},
                               args.run_dir, evaluated["plan"])
    published = write_report(args.out_dir, report, cases)

    status = report["canonical_status"]
    if status["canonical"]:
        print("\nCANONICAL: every registered gate passed")
    else:
        print(f"\n{status['note']}")
        for reason in status["reasons"]:
            print(f"  - {reason['gate']}: {reason['why']}")

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
