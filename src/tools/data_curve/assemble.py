"""exp_14: turn the data-curve's metric JSONs into one table, one verdict, one provenance.

Six runs (two backbones x three data fractions), ten evaluation cells each, plus the two
100 % anchors measured in exp_13 -- and a hypothesis ("CylDINO is a data-efficient
inductive bias") that is easy to read into a table by eye. So the reading is not done by
eye. Everything the results artifacts say is computed here by the rules pre-registered in
``plan_data_curve.md`` §1, before any number existed:

* **Orientation.** ``B_f = van - cyl`` for the error metrics (T60/C50/EDT) and
  ``cyl - van`` for recall, so "positive = CylDINO better" holds for every metric. One
  sign slip inverts the conclusion while the table still looks entirely reasonable.
* **Paired benefit.** Both arms are scored on the same five eval seeds, so the benefit is
  a per-seed difference ``d_s`` and its sd is *eval*-seed noise (one training seed per
  cell -- exp_12 Amendment 2: the verdict is descriptive, not an uncertainty statement).
* **The ordered verdict.** NOT SUPPORTED -> MIXED -> SUPPORTED -> PARTIAL, first match
  wins. The branches overlap on purpose, so the order carries the meaning.
* **Data equivalence.** An upward scan with first match over g(f) = cyl(f) - van@100 %,
  oriented; exactly one of four outcomes fires, and a non-monotone curve reports only its
  first crossing.

Decisions are taken at the reporting precision (4 decimals) so a reader can reproduce the
crossing fraction from the printed g vector; the aggregation is the pinned evaluator's AR
convention -- one item-weighted value over the whole 6,337-item unseen split per eval seed,
then mean +- sd over seeds 42-46. Never per-scene (plan §1, §11).
"""
import statistics

#: Scored metric -> its key in a metrics JSON's ``metrics`` object. FD and the
#: ``RIR_to_geom_R@k`` family are reported by the evaluator but are not endpoints here.
METRIC_KEYS = {
    "T60": "T60",
    "C50": "C50",
    "EDT": "EDT",
    "R@1": "RIR_to_GT_RIR_R@1",
    "R@5": "RIR_to_GT_RIR_R@5",
    "R@10": "RIR_to_GT_RIR_R@10",
}
METRICS = tuple(METRIC_KEYS)
#: The error metrics: smaller is better, so the benefit is ``van - cyl``.
LOWER_IS_BETTER = ("T60", "C50", "EDT")

#: The pre-registered primary endpoints of the verdict (plan §1). Everything else --
#: K=1, C50, R@k -- is reported with the same machinery and never overrides this.
PRIMARY_METRICS = ("T60", "EDT")
PRIMARY_K = 8

#: The measured fractions, in the scan order the data-equivalence rule requires.
FRACTION_PCTS = (25, 50, 75, 100)
#: The fraction that is *not* a new run: the exp_13 anchors at the full split.
ANCHOR_PCT = 100
#: Decimals the tables print, and therefore the precision equality is decided at.
REPORT_DP = 4


def oriented_benefit(metric, van_value, cyl_value):
    """Benefit of CylDINO over stock DINOv3 on ``metric``, oriented so positive = better.

    ``ValueError`` for a metric this experiment does not score: FD and the geometry
    recalls are printed by the evaluator but were never pre-registered as endpoints, and
    silently orienting one of them would invent a result.
    """
    if metric not in METRIC_KEYS:
        raise ValueError(f"{metric!r} is not a scored endpoint; expected one of {METRICS}")
    if metric in LOWER_IS_BETTER:
        return float(van_value) - float(cyl_value)
    return float(cyl_value) - float(van_value)


def mean_sd(values):
    """``(mean, sample sd)`` over the eval seeds; sd of a single seed is 0.0, not an error.

    Sample (n-1) sd, the convention of ``gen_model_comparison.py`` and of exp_13's
    ``assemble_curve.py``, so a number here is comparable with the published tables.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("cannot aggregate an empty list of seed values")
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _round(value):
    """Round to the reporting precision, normalising ``-0.0`` so ``>= 0`` reads honestly."""
    out = round(float(value), REPORT_DP)
    return 0.0 if out == 0 else out


def verdict(benefit_means, primary_metrics=PRIMARY_METRICS, fractions=FRACTION_PCTS):
    """The pre-registered verdict of plan §1, evaluated in ORDER, first match wins.

    ``benefit_means`` is ``{metric: {fraction %: B_f}}`` for the primary metrics at
    K = PRIMARY_K. The branches are deliberately overlapping, so this function is the rule:

    1. **NOT SUPPORTED** -- every primary metric has ``B_f <= 0`` at some fraction (not
       necessarily the same one).
    2. **MIXED** -- exactly one does.
    3. **SUPPORTED** -- neither of the above, and every primary benefit is non-increasing
       in the data fraction (B_25 >= B_50 >= B_75 >= B_100).
    4. **PARTIAL** -- otherwise; annotated "benefit shrinking" for each metric whose
       B_25 < B_100.

    A missing cell raises: a verdict over four of the eight primary numbers is not the
    pre-registered verdict, and rendering it as one would be the error this rule exists
    to prevent.
    """
    missing = [(m, f) for m in primary_metrics for f in fractions
               if benefit_means.get(m, {}).get(f) is None]
    if missing:
        raise ValueError(
            "the verdict is a function of every primary cell; missing " + ", ".join(
                f"{m} @ {f}%" for m, f in missing))
    non_positive = {m: [f for f in fractions if benefit_means[m][f] <= 0]
                    for m in primary_metrics}
    failing = [m for m in primary_metrics if non_positive[m]]
    monotone = {m: all(benefit_means[m][a] >= benefit_means[m][b]
                       for a, b in zip(fractions, fractions[1:]))
                for m in primary_metrics}
    annotations = []
    if len(failing) == len(primary_metrics):
        label, rule = "NOT SUPPORTED", "1 (every primary metric is non-positive somewhere)"
    elif failing:
        label, rule = "MIXED", "2 (exactly one primary metric is non-positive somewhere)"
    elif all(monotone.values()):
        label, rule = "SUPPORTED", "3 (all positive and non-increasing in the data fraction)"
    else:
        label, rule = "PARTIAL", "4 (all positive, monotonicity fails for >= 1 metric)"
        annotations = [f"{m}: benefit shrinking (B_{fractions[0]} < B_{fractions[-1]})"
                       for m in primary_metrics
                       if benefit_means[m][fractions[0]] < benefit_means[m][fractions[-1]]]
    return {
        "verdict": label,
        "rule": rule,
        "primary_metrics": list(primary_metrics),
        "primary_K": PRIMARY_K,
        "non_positive": non_positive,
        "monotone": monotone,
        "annotations": annotations,
        "benefits": {m: {f: benefit_means[m][f] for f in fractions} for m in primary_metrics},
    }


def data_equivalence(g_by_fraction, fractions=FRACTION_PCTS):
    """"How little data does CylDINO need to match stock DINOv3 at 100 %?", per plan §1.

    ``g_by_fraction`` is ``{fraction %: g(f)}`` with g already oriented so that ``g >= 0``
    means "CylDINO at f is at least as good as stock DINOv3 at 100 %". Upward scan, first
    match, exactly one of four outcomes:

    1. ``<= 25 %`` -- g at the smallest measured fraction is already non-negative, so the
       answer is censored by the design (we never measured below it).
    2. ``exact f_i`` -- g is zero there *to the reporting precision*.
    3. ``crossing in (f_{i-1}, f_i)`` -- g turns positive; f* by linear interpolation.
    4. ``no crossing through 100 %``.

    Decisions and the interpolation both use the rounded values, so f* is reproducible
    from the g vector printed beside it. A non-monotone curve therefore reports only its
    first crossing -- the whole vector is returned so the reader sees the rest.
    """
    g = {f: _round(g_by_fraction[f]) for f in fractions}
    base = {"g": g, "fractions": list(fractions)}
    first = fractions[0]
    if g[first] >= 0:
        return dict(base, kind="censored_at_min", outcome=f"<= {first} %", f_star=None,
                    bracket=None)
    previous = first
    for current in fractions[1:]:
        if g[previous] < 0:
            if g[current] == 0:
                return dict(base, kind="exact", outcome=f"exact {current} %",
                            f_star=float(current), bracket=None)
            if g[current] > 0:
                span = current - previous
                f_star = previous + span * (0.0 - g[previous]) / (g[current] - g[previous])
                return dict(base, kind="crossing",
                            outcome=f"crossing in ({previous} %, {current} %)",
                            f_star=f_star, bracket=[previous, current],
                            bracket_g=[g[previous], g[current]])
        previous = current
    return dict(base, kind="none", outcome=f"no crossing through {fractions[-1]} %",
                f_star=None, bracket=None)
