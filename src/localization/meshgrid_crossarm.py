"""exp_22 R2 -- the paired cross-arm comparison on the mesh grid (plan r2 §1, §3 R2-A).

The confirmatory question (plan r2 §0): does the generator's training policy
change how much invertible source-position information the mesh-grid
analysis-by-synthesis localizer recovers? Two treatment arms -- BF (C4 frame-
averaged) and YAW (yaw-augmented) -- are each compared with the published P1
comparator on the frozen 5,337-query / 16-room mesh protocol.

**Estimand (Codex B1).** Exactly the P1 report's room-first headline: inside each
of the 16 registered rooms take the median ``e_loc`` and ``success_raw@1.0`` (the
fraction with ``e_loc <= 1.0``) at K = 8 under the headline log-mean-exp score;
pair the 16 room values (treatment - P1) and average them unweighted. Per-query
joining is required for ADMISSION and room membership; it is not the reduction --
no pooled median, no median of per-query differences.

**Inference (Codex B2).** ONE room-resample index matrix from the report's
registered constants (``default_rng(20260825)``, 10,000 draws over the 16 rooms)
is reused for all four tests; each draw recomputes the paired room-delta mean.
Raw p-values are bootstrap tail proportions, draws exactly 0 counted as crossing,
no finite-sample correction (the exp_20 convention): BF one-sided toward
improvement (``P(draw >= 0)`` for the e_loc delta, ``P(draw <= 0)`` for the success
delta), YAW two-sided (``min(1, 2 min(P(draw <= 0), P(draw >= 0)))``). Holm over
exactly the four registered tests at alpha = 0.05. Percentile 95 % intervals of the
same draws are DESCRIPTIVE, never a verdict gate.

**Admission (Codex B3/H1).** Every per-query number is taken from the report's own
admission path (``meshgrid_report.evaluate_run``) -- never from raw rows and never
from an aggregate table -- and two runs are paired only when they agree on every
axis outside an explicit whitelist (the checkpoint, the arm, the arm's registered
method fields, the launch records, timings and every generated output).
"""
import argparse
import hashlib
import os
import sys

import numpy as np

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_report as mr

CONTROL_ARM = "P1"
TREATMENT_ARMS = ("BF", "YAW")
PRIMARY_AGGREGATOR = mr.HEADLINE_AGGREGATOR
PRIMARY_K = mr.HEADLINE_K
PRIMARY_ENDPOINTS = ("median_e_loc", "success_raw@1.0")
#: the direction of IMPROVEMENT of each endpoint's treatment - P1 delta
IMPROVEMENT = {"median_e_loc": -1, "success_raw@1.0": +1}
SIDEDNESS = {"BF": "one_sided", "YAW": "two_sided"}
REGISTERED_FAMILY = tuple((arm, stat) for arm in TREATMENT_ARMS for stat in PRIMARY_ENDPOINTS)
FAMILY_ALPHA = 0.05
BOOTSTRAP_SEED = mr.BOOTSTRAP_SEED
BOOTSTRAP_N = mr.BOOTSTRAP_N
CI_ALPHA = mr.BOOTSTRAP_ALPHA

#: descriptive only: these K and statistics get a delta and an interval, never a p
SECONDARY_KS = (1, 4, 8)
SECONDARY_STATS = ("median_e_loc", "success_raw@1.0", "success_raw@0.5",
                   "success_oracle_normalized@1.0")

INCOMPLETE_FAMILY_VERDICT = (
    "NO VERDICT -- INCOMPLETE FAMILY: the registered Holm family is exactly four tests over "
    "BOTH treatment arms; with one arm present only raw p-values are reported")
CI_ROLE = ("DESCRIPTIVE -- 95 % percentile interval of the shared room-resample draws; "
           "unadjusted, never a verdict gate (plan r2 §1)")


def family_label(arm, stat, k=PRIMARY_K):
    return f"{arm}_vs_{CONTROL_ARM}_{stat}_K{int(k)}"


def room_statistics(results, aggregator=PRIMARY_AGGREGATOR, k=PRIMARY_K):
    """``{room: report room_block}`` from AUTHENTICATED per-query results."""
    by_room = {}
    for result in results:
        entry = result["by"][aggregator][int(k)]
        bucket = by_room.setdefault(str(result["room_id"]), {"e_loc": [], "e_excess": []})
        bucket["e_loc"].append(float(entry["e_loc"]))
        bucket["e_excess"].append(float(entry["e_excess"]))
    return {room: mr.room_block(values) for room, values in sorted(by_room.items())}


def paired_room_deltas(treatment, control, stat):
    """``(rooms, treatment - control)`` over the SAME rooms, ordered by id."""
    rooms = sorted(control)
    if sorted(treatment) != rooms:
        differing = sorted(set(treatment) ^ set(control))[:3]
        raise ValueError(f"the arms do not cover the same rooms: {differing} differ; paired "
                         "room deltas need one room set")
    return rooms, np.asarray([float(treatment[r][stat]) - float(control[r][stat])
                              for r in rooms], dtype=np.float64)


def shared_draws(n_rooms, seed=BOOTSTRAP_SEED, n=BOOTSTRAP_N):
    """The ONE resample matrix every test reads (the report's own constructor)."""
    return mr.room_bootstrap_draws(n_rooms, seed=seed, n=n)


def draws_digest(draws):
    array = np.ascontiguousarray(np.asarray(draws, dtype=np.int64))
    return hashlib.sha256(repr(array.shape).encode() + array.tobytes()).hexdigest()


def endpoint_test(deltas, draws, arm, stat):
    """One registered test: the room-delta mean, its raw p, its descriptive CI."""
    deltas = np.asarray(deltas, dtype=np.float64)
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != deltas.size:
        raise ValueError(f"the resample matrix is over {draws.shape[-1]} rooms but there are "
                         f"{deltas.size} room deltas")
    if not np.isfinite(deltas).all():
        raise ValueError("a room delta is not finite")
    boot = deltas[draws].mean(axis=1)
    p_le, p_ge = float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0))
    if SIDEDNESS[arm] == "one_sided":
        p_raw = p_ge if IMPROVEMENT[stat] < 0 else p_le
    else:
        p_raw = min(1.0, 2.0 * min(p_le, p_ge))
    low, high = mr.percentile_ci(boot, alpha=CI_ALPHA)
    return {"label": family_label(arm, stat), "arm": arm, "stat": stat, "k": PRIMARY_K,
            "delta": float(deltas.mean()), "room_deltas": [float(d) for d in deltas],
            "p_raw": float(p_raw), "p_draw_le_zero": p_le, "p_draw_ge_zero": p_ge,
            "sidedness": SIDEDNESS[arm],
            "improvement": "negative delta" if IMPROVEMENT[stat] < 0 else "positive delta",
            "ci95": [low, high], "ci_role": CI_ROLE,
            "n_boot": int(draws.shape[0]), "n_rooms": int(deltas.size),
            "draws_sha256": draws_digest(draws), "p_adjusted": None, "rejected": None}


def holm_family(tests, alpha=FAMILY_ALPHA):
    """Holm over EXACTLY the four registered tests -- no more, no fewer."""
    from src.localization.rir_metrics import holm_bonferroni

    registered = sorted(family_label(arm, stat) for arm, stat in REGISTERED_FAMILY)
    supplied = sorted(str(test["label"]) for test in tests)
    if supplied != registered:
        raise ValueError(f"the confirmatory family must be exactly {registered}; got {supplied}")
    return holm_bonferroni({test["label"]: float(test["p_raw"]) for test in tests},
                           alpha=alpha)


def _improves(test):
    return (test["delta"] < 0.0) if IMPROVEMENT[test["stat"]] < 0 else (test["delta"] > 0.0)


def bf_verdict(test_e, test_s):
    """SUPERIOR iff both reject after Holm; PARTIAL iff one; NOT SUPERIOR iff none."""
    rejected = [test for test in (test_e, test_s) if test["rejected"]]
    for test in rejected:
        if not _improves(test):
            raise ValueError(f"{test.get('label', test['stat'])} rejected with the wrong sign "
                             f"(delta {test['delta']!r}); a one-sided test cannot, so the "
                             "registered procedure has been breached")
    return {2: "SUPERIOR", 1: "PARTIAL", 0: "NOT SUPERIOR"}[len(rejected)]


def yaw_verdict(test_e, test_s):
    """The four exhaustive two-sided classes of plan r2 §1."""
    n_rejected = sum(bool(test["rejected"]) for test in (test_e, test_s))
    if n_rejected == 0:
        return "NOT DIFFERENT"
    if n_rejected == 1:
        return "PARTIAL"
    if test_e["delta"] == 0.0 or test_s["delta"] == 0.0:
        raise ValueError("a rejected two-sided test has a delta of exactly 0; its direction is "
                         "undefined and the registered verdict cannot be assigned")
    return "DIFFERENT-CONSISTENT" if _improves(test_e) == _improves(test_s) else \
        "DIFFERENT-DISCORDANT"


def confirmatory(room_stats, draws=None):
    """The registered family over ``{arm: {room: stats}}`` (P1 + one or both treatments)."""
    if CONTROL_ARM not in room_stats:
        raise ValueError(f"the comparator {CONTROL_ARM} is required")
    present = [arm for arm in TREATMENT_ARMS if arm in room_stats]
    if not present:
        raise ValueError(f"at least one treatment arm of {list(TREATMENT_ARMS)} is required")
    rooms = sorted(room_stats[CONTROL_ARM])
    registered_draws = shared_draws(len(rooms))
    draws = registered_draws if draws is None else np.asarray(draws)
    tests = []
    for arm in present:
        for stat in PRIMARY_ENDPOINTS:
            _rooms, deltas = paired_room_deltas(room_stats[arm], room_stats[CONTROL_ARM], stat)
            tests.append(endpoint_test(deltas, draws, arm, stat))
    complete = len(present) == len(TREATMENT_ARMS)
    holm, verdicts = None, {arm: INCOMPLETE_FAMILY_VERDICT for arm in present}
    if complete:
        holm = holm_family(tests)
        adjusted = {entry["label"]: entry for entry in holm["tests"]}
        for test in tests:
            test["p_adjusted"] = float(adjusted[test["label"]]["p_adjusted"])
            test["rejected"] = bool(adjusted[test["label"]]["rejected"])
        by_arm = {arm: [t for t in tests if t["arm"] == arm] for arm in present}
        verdicts = {"BF": bf_verdict(*by_arm["BF"]), "YAW": yaw_verdict(*by_arm["YAW"])}
    return {"family_complete": complete, "arms": present, "rooms": rooms, "tests": tests,
            "holm": holm, "verdicts": verdicts, "alpha": FAMILY_ALPHA,
            "draws": {"seed": BOOTSTRAP_SEED, "n_boot": int(draws.shape[0]),
                      "n_rooms": int(draws.shape[1]), "sha256": draws_digest(draws),
                      "registered": bool(np.array_equal(draws, registered_draws))}}


def secondary_descriptives(results_by_arm, draws=None):
    """K = 1 / 4 / 8 x the secondary statistics: a delta and an interval, never a p."""
    control = results_by_arm[CONTROL_ARM]
    out = {}
    for arm in [a for a in TREATMENT_ARMS if a in results_by_arm]:
        out[arm] = {}
        for k in SECONDARY_KS:
            treated = room_statistics(results_by_arm[arm], k=k)
            base = room_statistics(control, k=k)
            matrix = shared_draws(len(base)) if draws is None else np.asarray(draws)
            cells = {}
            for stat in SECONDARY_STATS:
                if k == PRIMARY_K and stat in PRIMARY_ENDPOINTS:
                    continue
                _rooms, deltas = paired_room_deltas(treated, base, stat)
                low, high = mr.percentile_ci(deltas[matrix].mean(axis=1), alpha=CI_ALPHA)
                cells[stat] = {"delta": float(deltas.mean()), "ci95": [low, high],
                               "treatment": float(np.mean([treated[r][stat] for r in treated])),
                               "control": float(np.mean([base[r][stat] for r in base])),
                               "role": CI_ROLE}
            out[arm][f"K{k}"] = cells
    return out


# --------------------------------------------------------------------------- #
# admission through the report, and the pairing gate (Codex B3 / H1)
# --------------------------------------------------------------------------- #
LEGACY_EXCEPTION_NOTE = (
    "P1 LEGACY-PROVENANCE EXCEPTION (plan r2 §1, Codex B4/N3): the P1 comparator is the "
    "published Aug-2026 run, which predates full-pass launch records. It is admitted as "
    "canonical AS PUBLISHED only because --allow-legacy-p1-provenance was passed; the exception "
    "does not extend to BF or YAW, whose every row must resolve to a verified launch record")

#: what two admitted runs MAY differ on -- everything else is a pairing axis.
PAIRING_WHITELIST = {
    "binding": ("ckpt_sha256", "binding_sha256") + tuple(mr.ARM_PROTOCOL_FIELDS),
    "row": ("by_k", "timings_s", "launch_record_sha256", "binding_sha256", "row_sha256",
            "sims_sha256", "_row_path", "waveform_path", "waveform_sha256",
            "waveform_candidate_indices", "waveform_content_rule"),
    "run": ("arm", "launch_provenance", "predictions_scores_margins", "sidecar_disagreements",
            "latency", "registered_protocol.arm_protocol"),
}
PAIRING_AXES = ("query_identities", "row.<every field outside the row whitelist>",
                "by_k_prefixes", "candidate_coordinates", "truth_xyz",
                "binding.<every field outside the binding whitelist>", "effective_batching",
                "sidecar_argmax_policy", "merge_receipt", "census", "metadata_bank_sha256",
                "noise_keys", "aggregate_policy", "single_shard", "protocol_deviation",
                "metadata_bank_unpinned", "non_canonical", "launch_provenance", "arm")
MERGE_RECEIPT_FIELDS = ("declared_rooms", "advisory", "n_rows", "totals",
                        "source_rows_derived_from", "source_rows_from_g1_plan",
                        "source_rows_from_rows", "receipt_cross_checked_against_rows")
AGGREGATE_POLICY = {"aggregators": list(mr.AGGREGATORS),
                    "headline": [mr.HEADLINE_AGGREGATOR, mr.HEADLINE_K],
                    "tie_break": "smallest global candidate index"}


def whitelist_names():
    return ([f"binding.{field}" for field in PAIRING_WHITELIST["binding"]]
            + [f"row.{field}" for field in PAIRING_WHITELIST["row"]]
            + list(PAIRING_WHITELIST["run"]))


def _digest(value):
    from src.localization.crossarm import canonical_sha256

    return canonical_sha256(mr.jsonable(value))


def noise_keys_digest(binding, results, rows):
    """The derivable CRN keys of every query's K draws (seed, policy, query, draw)."""
    keys = {}
    for result in results:
        first = int(rows[result["query_id"]]["candidate_indices"][0])
        keys[result["query_id"]] = [int(me.noise_key_for(binding["noise_policy"],
                                                         int(binding["seed"]),
                                                         result["query_id"], first, k))
                                    for k in range(int(binding["num_samples"]))]
    return _digest(keys)


def admit_run(run_dir, arm, *, audit_report, context_manifest, metadata_root,
              expect_metadata_bank_sha256, allow_legacy_p1_provenance=False, ckpt_path=None,
              totals=None, require_manifest_census=True,
              sidecar_argmax_policy=mr.SIDECAR_ARGMAX_POLICY):
    """One arm, admitted through the report's own gates, canonical or refused."""
    if not expect_metadata_bank_sha256:
        raise ValueError("a cross-arm comparison requires the pre-registered pair-metadata "
                         "bank digest; the truths behind every e_loc must be the registered ones")
    evaluated = mr.evaluate_run(
        run_dir, audit_report, context_manifest, metadata_root, totals=totals,
        sidecar_argmax_policy=sidecar_argmax_policy,
        require_manifest_census=require_manifest_census, single_shard=False,
        expect_ckpt_sha256=mr.REGISTERED_CKPT_SHA256[arm], expect_arm=arm,
        expect_metadata_bank_sha256=expect_metadata_bank_sha256,
        allow_protocol_deviation=False, allow_unpinned_metadata_bank=False, ckpt_path=ckpt_path)
    # the comparison applies the sparse-retrieval gate ITSELF, once (rA-F1)
    report = mr.build_report(evaluated, run_dir, audit_report, context_manifest, metadata_root,
                             sidecar_argmax_policy=sidecar_argmax_policy,
                             defer_retrieval_gate=True)
    status = report["canonical_status"]
    if not status["canonical"]:
        raise ValueError(f"the {arm} run at {run_dir} is not canonical: "
                         + "; ".join(f"{r['gate']}: {r['why']}" for r in status["reasons"]))
    provenance = evaluated["launch_provenance"]
    if provenance["legacy_provenance"] and not allow_legacy_p1_provenance:
        raise ValueError(f"the {arm} run carries no launch records; it may enter only as the "
                         "published P1 comparator under the registered exception, which must "
                         f"be requested explicitly with --allow-legacy-p1-provenance. "
                         f"{LEGACY_EXCEPTION_NOTE}")
    results, rows = evaluated["results"], evaluated["rows_by_id"]
    plans = evaluated["plans_by_id"]
    return {"arm": arm, "run_dir": str(run_dir), "binding": evaluated["binding"],
            "binding_sha256": evaluated["binding_sha256"], "results": results, "rows": rows,
            "candidate_coordinates_sha256": {
                qid: hashlib.sha256(np.ascontiguousarray(
                    np.asarray(plan.coordinates, dtype=np.float64)).tobytes()).hexdigest()
                for qid, plan in plans.items()},
            "truth_xyz": {r["query_id"]: [float(v) for v in r["truth_xyz"]] for r in results},
            "noise_keys_sha256": noise_keys_digest(evaluated["binding"], results, rows),
            "census": evaluated["census"], "merge": evaluated["merge"],
            "batching": evaluated["batching"], "metadata_bank": evaluated["metadata_bank"],
            "registered_protocol": evaluated["registered_protocol"],
            "launch_provenance": provenance, "canonical_status": status,
            "single_shard": evaluated["single_shard"],
            "sidecar_argmax_policy": str(sidecar_argmax_policy),
            "legacy_provenance_allowed": bool(allow_legacy_p1_provenance),
            "aggregate_policy": AGGREGATE_POLICY,
            "report_headline": report["metrics"][PRIMARY_AGGREGATOR][str(PRIMARY_K)][
                "across_rooms"],
            "report_per_room": report["metrics"][PRIMARY_AGGREGATOR][str(PRIMARY_K)][
                "per_room"],
            "sidecar_summary": report["crosscheck"]["sidecar"]}


def _run_flags(run):
    """Absence of every diagnostic / relaxed gate, and verified launch provenance."""
    registered, provenance = run["registered_protocol"], run["launch_provenance"]
    failed = []
    if run["single_shard"]:
        failed.append(("single_shard", "a single-shard diagnostic is not a canonical run"))
    if registered.get("deviation_allowed") or not registered.get("is_registered"):
        failed.append(("protocol_deviation", "the binding is a protocol deviation"))
    if not run["metadata_bank"].get("pinned"):
        failed.append(("metadata_bank_unpinned", "the pair-metadata bank was not pinned"))
    if not run["canonical_status"].get("canonical"):
        failed.append(("non_canonical", "the run's own report is non-canonical"))
    if not (provenance.get("complete") or (provenance.get("legacy_provenance")
                                           and run.get("legacy_provenance_allowed"))):
        failed.append(("launch_provenance", "rows do not all resolve to verified launch "
                                            "records and no permitted legacy exception applies"))
    return [{"axis": axis, "detail": f"{run['arm']}: {why}"} for axis, why in failed]


def pair_runs(a, b):
    """Two admitted runs may be paired only if they agree outside the whitelist (H1)."""
    mismatches, whitelisted = [], set()

    def check(axis, left, right, detail=None):
        if left != right:
            mismatches.append({"axis": axis, "detail": detail or f"{a['arm']} != {b['arm']}"})

    for run in (a, b):
        mismatches += _run_flags(run)
    check("arm", a["arm"] == b["arm"], False, f"both runs are arm {a['arm']}")
    ids = [[(r["query_id"], int(r["position"])) for r in run["results"]] for run in (a, b)]
    for run, identities in zip((a, b), ids):
        if len({qid for qid, _ in identities}) != len(identities):
            check("query_identities", "duplicates", None, f"{run['arm']} repeats a query")
    check("query_identities", ids[0], ids[1], "the ordered (query, position) streams differ")

    binding_skip = set(PAIRING_WHITELIST["binding"])
    for key in sorted(set(a["binding"]) | set(b["binding"])):
        if key in binding_skip:
            if a["binding"].get(key) != b["binding"].get(key):
                whitelisted.add(f"binding.{key}")
            continue
        check(f"binding.{key}", a["binding"].get(key), b["binding"].get(key))

    row_skip, by_axis = set(PAIRING_WHITELIST["row"]), {}
    common = [qid for qid, _ in ids[0] if qid in a["rows"] and qid in b["rows"]]
    for qid in common:
        ra, rb = a["rows"][qid], b["rows"][qid]
        for key in set(ra) | set(rb):
            if key in row_skip:
                if ra.get(key) != rb.get(key):
                    whitelisted.add(f"row.{key}")
            elif ra.get(key) != rb.get(key):
                by_axis.setdefault(f"row.{key}", []).append(qid)
        if sorted(ra.get("by_k") or {}) != sorted(rb.get("by_k") or {}):
            by_axis.setdefault("by_k_prefixes", []).append(qid)
        for axis, field in (("candidate_coordinates", "candidate_coordinates_sha256"),
                            ("truth_xyz", "truth_xyz")):
            if a[field].get(qid) != b[field].get(qid):
                by_axis.setdefault(axis, []).append(qid)
    for axis, queries in sorted(by_axis.items()):
        mismatches.append({"axis": axis, "detail": f"{len(queries)} queries differ "
                                                   f"(first {queries[:2]})"})

    check("effective_batching", a["batching"].get("batching"), b["batching"].get("batching"))
    check("sidecar_argmax_policy", a["sidecar_argmax_policy"], b["sidecar_argmax_policy"])
    check("merge_receipt", {f: (a["merge"] or {}).get(f) for f in MERGE_RECEIPT_FIELDS},
          {f: (b["merge"] or {}).get(f) for f in MERGE_RECEIPT_FIELDS})
    check("census", a["census"], b["census"])
    check("metadata_bank_sha256", a["metadata_bank"].get("metadata_bank_sha256"),
          b["metadata_bank"].get("metadata_bank_sha256"))
    check("noise_keys", a["noise_keys_sha256"], b["noise_keys_sha256"])
    check("aggregate_policy", a["aggregate_policy"], b["aggregate_policy"])
    if a["arm"] != b["arm"]:
        whitelisted.add("arm")
    if _digest(a["launch_provenance"].get("records")) != _digest(
            b["launch_provenance"].get("records")):
        whitelisted.add("launch_provenance")
    return {"paired": not mismatches, "arms": [a["arm"], b["arm"]], "mismatches": mismatches,
            "n_queries": len(ids[0]), "whitelisted_differences": sorted(whitelisted),
            "axes": list(PAIRING_AXES), "whitelist": PAIRING_WHITELIST}


# --------------------------------------------------------------------------- #
# the comparison, its artifacts and its CLI
# --------------------------------------------------------------------------- #
OUTPUT_JSON = "crossarm_r2.json"
OUTPUT_MARKDOWN = "crossarm_r2.md"
CLAIM_BOUNDARY = (
    "CLAIM BOUNDARY (plan r2 §1, Codex H5): no per-arm off-grid probe or calibration is run in "
    "R2, so this comparison makes NO claim for BF or YAW about continuous-truth score/rank or "
    "sub-lattice sharpness, grid densification, real-vs-generated AGREE calibration / domain "
    "gap, or off-grid behaviour; those conclusions remain P1-only. The arm-independent controls "
    "(random baseline, sparse retrieval) are reused, not rerun")


def assert_per_room_agreement(arm, stats, run):
    """EVERY registered room, both endpoints: crossarm == the arm's own report.

    The across-room check alone would pass compensating per-room differences
    (Codex rA NIT); both sides apply ``mr.room_block`` to the same authenticated
    results in the same order, so the values must be EQUAL, not merely close.
    """
    reported = run["report_per_room"]
    registered = sorted(run["census"]["rooms"])
    if sorted(stats) != registered or sorted(reported) != registered:
        raise ValueError(f"{arm}: the per-room sets differ -- registered {registered[:3]}..., "
                         f"crossarm {sorted(stats)[:3]}..., report {sorted(reported)[:3]}...")
    for room in registered:
        for stat in PRIMARY_ENDPOINTS:
            if float(stats[room][stat]) != float(reported[room][stat]):
                raise ValueError(f"{arm}: the per-room {stat} of {room} recomputed here "
                                 f"({stats[room][stat]!r}) is not the arm's own report's "
                                 f"({reported[room][stat]!r}); every room must agree")


def run_crossarm(p1, yaw=None, bf=None):
    """Pair every admitted treatment with P1 (and with each other), then test the family."""
    from datetime import datetime, timezone

    admitted = {arm: run for arm, run in (("P1", p1), ("YAW", yaw), ("BF", bf)) if run}
    pairs = [(arm, CONTROL_ARM) for arm in ("YAW", "BF") if arm in admitted]
    if len(pairs) == 2:
        pairs.append(("BF", "YAW"))
    pairing = {f"{t}_vs_{c}": pair_runs(admitted[c], admitted[t]) for t, c in pairs}
    refused = {name: verdict["mismatches"][:5] for name, verdict in pairing.items()
               if not verdict["paired"]}
    if refused:
        raise ValueError(f"the pairing gate refused {sorted(refused)}: {refused}")
    room_stats = {arm: room_statistics(run["results"]) for arm, run in admitted.items()}
    draws = shared_draws(len(room_stats[CONTROL_ARM]))
    per_arm = {}
    for arm, stats in room_stats.items():
        per_arm[arm] = {stat: float(np.mean([stats[r][stat] for r in stats]))
                        for stat in PRIMARY_ENDPOINTS}
        per_arm[arm].update(
            per_room=stats, n_rooms=len(stats), n_queries=len(admitted[arm]["results"]),
            census={key: admitted[arm]["census"][key] for key in
                    ("n_queries", "n_rooms", "candidate_query_pairs", "generated_waveforms")},
            run_dir=admitted[arm]["run_dir"],
            ckpt_sha256=admitted[arm]["binding"]["ckpt_sha256"],
            binding_sha256=admitted[arm]["binding_sha256"],
            legacy_provenance=bool(admitted[arm]["launch_provenance"]["legacy_provenance"]),
            launch_records=[{key: record.get(key) for key in
                             ("launch_record_sha256", "shard", "n_rows", "position_ranges")}
                            for record in admitted[arm]["launch_provenance"]["records"]])
        headline = admitted[arm]["report_headline"]
        for stat in PRIMARY_ENDPOINTS:
            if abs(per_arm[arm][stat] - float(headline[stat]["point"])) > 1e-12:
                raise ValueError(f"{arm}: the room-first {stat} recomputed here differs from the "
                                 "arm's own report headline; the two derivations disagree")
        assert_per_room_agreement(arm, stats, admitted[arm])
    return {"experiment": "exp_22 loc_meshgrid R2 paired cross-arm comparison",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "estimand": "unweighted mean over the registered rooms of the paired room deltas "
                        "(treatment - P1) of the within-room median e_loc and success_raw@1.0, "
                        f"{PRIMARY_AGGREGATOR} score, K = {PRIMARY_K} (plan r2 §1)",
            "confirmatory": confirmatory(room_stats, draws=draws),
            "secondary": secondary_descriptives(
                {arm: run["results"] for arm, run in admitted.items()}, draws=draws),
            "per_arm": per_arm, "pairing": pairing,
            "legacy_p1_provenance": per_arm[CONTROL_ARM]["legacy_provenance"],
            "legacy_exception_note": (LEGACY_EXCEPTION_NOTE
                                      if per_arm[CONTROL_ARM]["legacy_provenance"] else None),
            "claim_boundary": CLAIM_BOUNDARY,
            "labels": {"subset": mr.SUBSET_LABEL,
                       "agree_leakage_caveat": me.AGREE_LEAKAGE_CAVEAT}}


def _cell(value, digits=4, signed=True):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def render_markdown(payload):
    confirmatory = payload["confirmatory"]
    lines = ["# exp_22 R2 — paired cross-arm comparison on the mesh grid", "",
             f"Generated {payload['created_utc']}. {payload['estimand']}.", "",
             f"- **Scope:** {payload['labels']['subset']}",
             f"- **AGREE leakage caveat:** {payload['labels']['agree_leakage_caveat']}",
             f"- **Inference:** one shared room-resample matrix (seed "
             f"{confirmatory['draws']['seed']}, {confirmatory['draws']['n_boot']:,} draws over "
             f"{confirmatory['draws']['n_rooms']} rooms, sha256 "
             f"`{confirmatory['draws']['sha256'][:16]}...`); BF one-sided, YAW two-sided; "
             "draws exactly 0 count as crossing; no finite-sample correction; Holm over "
             f"exactly 4 tests at alpha = {confirmatory['alpha']}. CIs are descriptive.", ""]
    if payload.get("legacy_exception_note"):
        lines += [f"> **{payload['legacy_exception_note']}**", ""]
    lines += [f"> **{payload['claim_boundary']}**", ""]
    lines += ["## Registered confirmatory tests (room-first, K = 8)", "",
              "| contrast | Δ | 95 % CI | p_raw | p_adj | verdict |", "|---|---|---|---|---|---|"]
    for test in confirmatory["tests"]:
        outcome = ("raw p only (family incomplete)" if test["rejected"] is None
                   else "REJECTED" if test["rejected"] else "not rejected")
        lines.append(f"| `{test['label']}` | {_cell(test['delta'])} | "
                     f"[{_cell(test['ci95'][0])}, {_cell(test['ci95'][1])}] | "
                     f"{_cell(test['p_raw'], signed=False)} | "
                     f"{_cell(test['p_adjusted'], signed=False)} | {outcome} |")
    lines.append("")
    for arm, verdict in confirmatory["verdicts"].items():
        lines.append(f"- **{arm}: {verdict}**")
    lines += ["", "## Per-arm room-first values (K = 8)", "",
              "| arm | mean within-room median e_loc (m) | mean success_raw@1.0 | rooms | "
              "queries | legacy provenance | launch records |", "|---|---|---|---|---|---|---|"]
    for arm, summary in payload["per_arm"].items():
        lines.append(f"| {arm} | {_cell(summary['median_e_loc'], signed=False)} | "
                     f"{_cell(summary['success_raw@1.0'], signed=False)} | {summary['n_rooms']} | "
                     f"{summary['n_queries']:,} | {'yes' if summary['legacy_provenance'] else 'no'}"
                     f" | {len(summary['launch_records'])} |")
    lines += ["", "## Secondary readouts (DESCRIPTIVE: Δ and interval only, no test)", "",
              "| arm | K | statistic | Δ | 95 % CI |", "|---|---|---|---|---|"]
    for arm, by_k in payload["secondary"].items():
        for k, cells in by_k.items():
            for stat, cell in cells.items():
                lines.append(f"| {arm} | {k} | {stat} | {_cell(cell['delta'])} | "
                             f"[{_cell(cell['ci95'][0])}, {_cell(cell['ci95'][1])}] |")
    lines += ["", "## Pairing gate", "", "| pair | paired | queries | whitelisted differences |",
              "|---|---|---|---|"]
    for name, verdict in payload["pairing"].items():
        lines.append(f"| {name} | {'yes' if verdict['paired'] else 'NO'} | "
                     f"{verdict['n_queries']:,} | "
                     f"{', '.join(verdict['whitelisted_differences']) or '—'} |")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, payload):
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = os.path.join(str(out_dir), OUTPUT_JSON)
    me.write_json(json_path, mr.jsonable(payload))
    markdown_path = os.path.join(str(out_dir), OUTPUT_MARKDOWN)
    with open(markdown_path + ".tmp", "w") as handle:
        handle.write(render_markdown(payload))
    os.replace(markdown_path + ".tmp", markdown_path)
    return {"json": json_path, "markdown": markdown_path}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-p1", required=True, help="the MERGED P1 comparator run")
    parser.add_argument("--run-yaw", default=None, help="the MERGED YAW run")
    parser.add_argument("--run-bf", default=None, help="the MERGED BF run")
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--expect-metadata-bank-sha256", default=None,
                        help="the PRE-REGISTERED pair-metadata bank digest (required)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-legacy-p1-provenance", action="store_true",
                        help="admit the published P1 comparator, which predates launch "
                             "records, under the registered exception (printed prominently)")
    for arm in ("p1", "yaw", "bf"):
        parser.add_argument(f"--ckpt-{arm}", default=None,
                            help=f"optional: the {arm.upper()} checkpoint, so its embedded "
                                 "training protocol is checked too")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    if not args.expect_metadata_bank_sha256:
        _refuse("--expect-metadata-bank-sha256 is required: the truths behind every e_loc must "
                "come from the pre-registered bank")
    if not (args.run_yaw or args.run_bf):
        _refuse("at least one treatment run (--run-yaw / --run-bf) is required")
    runs = {os.path.abspath(str(d)) for d in (args.run_p1, args.run_yaw, args.run_bf) if d}
    if os.path.abspath(str(args.out_dir)) in runs:
        _refuse("--out-dir may not be one of the run directories it reports on")
    return True


def main(argv=None, admission_kwargs=None):
    """CLI. ``admission_kwargs`` is a fixture seam (totals / manifest census), never a flag."""
    args = parse_args(argv)
    validate_args(args)
    common = dict(audit_report=args.audit_report, context_manifest=args.context_manifest,
                  metadata_root=args.metadata_root,
                  expect_metadata_bank_sha256=args.expect_metadata_bank_sha256,
                  **(admission_kwargs or {}))
    admitted = {}
    for arm, run_dir, ckpt in (("P1", args.run_p1, args.ckpt_p1),
                               ("YAW", args.run_yaw, args.ckpt_yaw),
                               ("BF", args.run_bf, args.ckpt_bf)):
        if not run_dir:
            continue
        try:
            admitted[arm] = admit_run(
                run_dir, arm, ckpt_path=ckpt,
                allow_legacy_p1_provenance=(arm == CONTROL_ARM
                                            and args.allow_legacy_p1_provenance), **common)
        except ValueError as error:
            _refuse(f"{arm}: {error}")
        print(f"admitted {arm}: {run_dir} (binding {admitted[arm]['binding_sha256'][:12]}...)")
    try:
        payload = run_crossarm(admitted["P1"], yaw=admitted.get("YAW"), bf=admitted.get("BF"))
    except ValueError as error:
        _refuse(str(error))
    if payload["legacy_exception_note"]:
        print(f"\n*** {payload['legacy_exception_note']} ***\n")
    print(f"{CLAIM_BOUNDARY}\n")
    for test in payload["confirmatory"]["tests"]:
        print(f"  {test['label']:34s} delta {test['delta']:+.4f}  p_raw {test['p_raw']:.4f}  "
              f"p_adj {_cell(test['p_adjusted'], signed=False)}")
    for arm, verdict in payload["confirmatory"]["verdicts"].items():
        print(f"VERDICT {arm}: {verdict}")
    for name, path in write_outputs(args.out_dir, payload).items():
        print(f"  {name:9s} -> {path}  sha256 {me.file_sha256(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
