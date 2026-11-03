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
import hashlib

import numpy as np

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
