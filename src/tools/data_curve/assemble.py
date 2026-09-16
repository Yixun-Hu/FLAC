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
import glob
import json
import math
import os
import re
import statistics

from src.tools.data_curve import names

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


#: Optimizer steps x global batch = target draws per run; the numerator of "effective
#: epochs" (plan §1: this is a FIXED-COMPUTE curve, so the 25 % arm revisits each RIR ~4x
#: more often than the 100 % arm, and that ratio is reported, not hidden).
GLOBAL_BATCH = names.MICRO_BATCH * names.NUM_GPUS * names.ACCUM_BATCHES
#: The 100 % anchors come from exp_13's reference file, arm by arm.
ANCHOR_REFERENCE_KEYS = {"van": "P1_vanilla_S", "cyl": "CylDINO_core_S"}
#: tier_S_reference.json states its own n (five eval seeds) in ``_provenance``; it holds
#: no per-seed values, so a marginal anchor's n is *declared*, never recounted here.
ANCHOR_DECLARED_SEEDS = 5
#: Placed-by-hand anchor cells live beside the raw JSONs in this file (D10: P1's seed-42
#: K=8 anchor is the screen cell, whose basename names neither K nor seed).
ANCHOR_MANIFEST_BASENAME = "anchor_cells.json"
_ANCHOR_CELL_RE = re.compile(r"_K(\d+)_s(\d+)(?=[_.])")


class DataCurveError(RuntimeError):
    """The artifacts on disk cannot answer the question that was asked of them."""


def final_checkpoint(run_dir, step=names.MAX_STEPS):
    """The run's single step-``step`` checkpoint -- the file every cell was scored from.

    Zero means the run is not finished; more than one means the directory cannot say which
    bytes produced the cells beside it. Both are refusals, never a "pick the newest".
    """
    hits = sorted(glob.glob(os.path.join(glob.escape(run_dir), f"*step={step}.ckpt")))
    if not hits:
        raise DataCurveError(f"no step-{step} checkpoint in {run_dir}: the run is unfinished")
    if len(hits) > 1:
        raise DataCurveError(
            f"{len(hits)} step-{step} checkpoints in {run_dir} "
            f"({', '.join(os.path.basename(h) for h in hits)}): which one produced the "
            "cells beside them is not decidable here")
    return hits[0]


def cell_protocol(arm):
    """The four conditioning flags every cell of ``arm`` must carry (announcement 05)."""
    return {"cond_method": names.ARM_COND_METHOD[arm],
            "frame_avg_angles": [float(names.FRAME_AVG_ANGLES)] if arm == "cyl" else None,
            "rotate_deg": float(names.ROTATE_DEG),
            "cond_autocast": names.COND_AUTOCAST}


def check_cell_protocol(record, arm):
    """Violations of one metrics JSON against its arm's protocol and this experiment's keys.

    Deliberately not ``names.check_metrics``: that one binds a cell to a checkpoint digest
    the *launcher* holds, which the assembler does not have. What is checkable from the
    record alone is checked here -- and a cell scored under the other arm's conditioning is
    the failure that produced exp_09's retracted conclusion, so it is refused, not noted.
    """
    if not isinstance(record, dict):
        return [f"is a {type(record).__name__}, not a metrics record"]
    want = cell_protocol(arm)
    bad = []
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        bad.append(f"metrics is {metrics!r}, expected a non-empty object")
    else:
        for key in METRIC_KEYS.values():
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                bad.append(f"metrics.{key} is {value!r}, expected a finite number")
    if record.get("cond_method") != want["cond_method"]:
        bad.append(f"cond_method is {record.get('cond_method')!r}, "
                   f"expected {want['cond_method']!r} for the {arm} arm")
    angles = record.get("frame_avg_angles")
    if angles is not None:
        try:
            angles = [float(a) for a in angles]
        except (TypeError, ValueError):
            pass
    if angles != want["frame_avg_angles"]:
        bad.append(f"frame_avg_angles is {angles!r}, expected {want['frame_avg_angles']!r}")
    rotate = record.get("rotate_deg")
    if isinstance(rotate, bool) or not isinstance(rotate, (int, float)) \
            or float(rotate) != want["rotate_deg"]:
        bad.append(f"rotate_deg is {rotate!r}, expected {want['rotate_deg']}")
    if record.get("cond_autocast") != want["cond_autocast"]:
        bad.append(f"cond_autocast is {record.get('cond_autocast')!r}, "
                   f"expected {want['cond_autocast']!r}")
    return bad


def _read_cell(path, arm):
    """``(metrics, record, violations)`` for one metrics JSON; metrics is None if refused."""
    try:
        with open(path) as fin:
            record = json.load(fin)
    except (OSError, ValueError) as err:
        return None, None, [f"cannot be read as JSON ({type(err).__name__}: {err})"]
    bad = check_cell_protocol(record, arm)
    if bad:
        return None, record, bad
    return ({metric: float(record["metrics"][key]) for metric, key in METRIC_KEYS.items()},
            record, [])


def load_run(nas_root, arm, tag, seeds=names.SEEDS, ks=names.K_VALUES):
    """Every cell of one training run, off the NAS, with the reasons any of them is absent.

    A run whose cells disagree about ``ckpt_sha256`` is emptied, not patched: the digest is
    the evaluator's own record of which bytes it scored, so two of them in one run means
    the directory cannot say what these ten numbers are, and no subset of them is safe.
    """
    run = names.run_id(arm, tag)
    run_dir = os.path.join(nas_root, run)
    out = {"run_id": run, "arm": arm, "fraction_tag": tag, "run_dir": run_dir, "ckpt": None,
           "ckpt_sha256": None, "protocol": cell_protocol(arm),
           "cells": {K: {} for K in ks}, "violations": []}
    try:
        out["ckpt"] = final_checkpoint(run_dir)
    except DataCurveError as err:
        out["violations"].append(f"{run}: {err}")
        return out
    digests = {}
    for K in ks:
        for seed in seeds:
            where = f"{run} K{K} s{seed}"
            path = names.metrics_json_path(out["ckpt"], arm, tag, K, seed)
            if not os.path.exists(path):
                out["violations"].append(f"{where}: no metrics JSON at {path}")
                continue
            metrics, record, bad = _read_cell(path, arm)
            if metrics is None:
                out["violations"] += [f"{where}: {b}" for b in bad]
                continue
            digest = record.get("ckpt_sha256")
            if digest is None:
                out["violations"].append(
                    f"{where}: the record carries no ckpt_sha256, so it cannot be bound to "
                    "the checkpoint bytes the launcher validated")
            else:
                digests.setdefault(str(digest), []).append(f"K{K} s{seed}")
            out["cells"][K][seed] = {"path": path, "metrics": metrics, "ckpt_sha256": digest}
    if len(digests) > 1:
        out["violations"].append(
            f"{run}: its cells name {len(digests)} different ckpt_sha256 values (" + "; ".join(
                f"{d[:12]}...: {', '.join(c)}" for d, c in sorted(digests.items()))
            + ") -- this run's numbers do not come from one checkpoint and are all dropped")
        out["cells"] = {K: {} for K in ks}
    elif digests:
        out["ckpt_sha256"] = next(iter(digests))
    return out


def aggregate(values_by_seed, expect_seeds=names.SEEDS):
    """mean +- sd over the eval seeds present, with ``complete`` iff all five are there.

    Nothing is dropped and nothing is extrapolated: a four-seed row keeps its numbers and
    is *marked*, so the renderer can show it as incomplete rather than let it stand beside
    five-seed rows as if it were the same quantity (plan §1; strict mode makes it fatal).
    """
    seeds = sorted(values_by_seed)
    if not seeds:
        return {"mean": None, "sd": None, "n": 0, "seeds": [], "complete": False}
    mean, sd = mean_sd([values_by_seed[seed] for seed in seeds])
    return {"mean": mean, "sd": sd, "n": len(seeds), "seeds": seeds,
            "complete": seeds == sorted(expect_seeds)}


def paired_benefit(metric, cyl_by_seed, van_by_seed, expect_seeds=names.SEEDS):
    """B_f from per-seed differences d_s, or ``None`` when the arms are not paired.

    Both arms are evaluated on the same seeds 42-46, so the difference is taken *within*
    a seed and its sd is the spread of the benefit itself -- much tighter, and much more
    informative, than a difference of two marginal means whose sds share the same
    seed-to-seed wobble. Refused unless both arms carry exactly the expected seeds
    (plan §1: "the assembler refuses a paired sd unless exactly one cell per seed exists").
    """
    seeds = sorted(cyl_by_seed)
    if not seeds or seeds != sorted(van_by_seed) or seeds != sorted(expect_seeds):
        return None
    diffs = [oriented_benefit(metric, van_by_seed[s], cyl_by_seed[s]) for s in seeds]
    mean, sd = mean_sd(diffs)
    return {"mean": mean, "sd": sd, "n": len(seeds), "seeds": seeds, "form": "paired",
            "per_seed": dict(zip(seeds, diffs))}


def marginal_benefit(metric, cyl_agg, van_agg):
    """B_f as a difference of means -- the only form available from marginal anchors.

    ``sd`` is ``None`` on purpose: the two sds cannot be combined into the sd of a paired
    difference, and printing a pooled number there would claim a precision nobody measured.
    """
    if not cyl_agg or not van_agg:
        return None
    if cyl_agg.get("mean") is None or van_agg.get("mean") is None:
        return None
    return {"mean": oriented_benefit(metric, van_agg["mean"], cyl_agg["mean"]), "sd": None,
            "n": min(cyl_agg.get("n") or 0, van_agg.get("n") or 0), "seeds": None,
            "form": "marginal"}


def load_anchor_reference(path, arms=names.ARMS, ks=names.K_VALUES):
    """The 100 % anchors in marginal form, from exp_13's ``tier_S_reference.json``."""
    with open(path) as fin:
        reference = json.load(fin)
    out = {}
    for arm in arms:
        block = reference[ANCHOR_REFERENCE_KEYS[arm]]
        out[arm] = {K: {metric: {"mean": float(block[f"K{K}"][metric][0]),
                                 "sd": float(block[f"K{K}"][metric][1]),
                                 "n": ANCHOR_DECLARED_SEEDS, "seeds": None,
                                 "complete": True, "form": "marginal"}
                        for metric in METRICS}
                    for K in ks}
    return out


def load_anchor_cells(directory, arm, seeds=names.SEEDS, ks=names.K_VALUES):
    """``({K: {seed: cell}}, violations)`` for the raw 100 % anchor JSONs of one arm.

    Placing a cell is the whole difficulty. Most anchor basenames carry ``_K<k>_s<seed>``,
    but D10's seed-42 K=8 P1 anchor is the screen cell ``..._exp07_P1_screen_S40000_ema``,
    whose name says neither -- so an ``anchor_cells.json`` beside the JSONs pins those by
    hand and a cell that is neither parseable nor pinned is refused, never guessed. Two
    cells claiming one slot is likewise a refusal: "the newest wins" is how an anchor
    quietly becomes a different measurement.
    """
    violations = []
    cells = {K: {} for K in ks}
    manifest = {}
    manifest_path = os.path.join(directory, ANCHOR_MANIFEST_BASENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as fin:
            manifest = json.load(fin)
    claims = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(directory), "*.json"))):
        base = os.path.basename(path)
        if base == ANCHOR_MANIFEST_BASENAME:
            continue
        slot = None
        if base in manifest:
            slot = (int(manifest[base]["K"]), int(manifest[base]["seed"]))
        else:
            hit = _ANCHOR_CELL_RE.search(base)
            if hit:
                slot = (int(hit.group(1)), int(hit.group(2)))
        if slot is None or slot[0] not in ks or slot[1] not in seeds:
            violations.append(
                f"{base}: names no K/seed this experiment evaluates and is not placed by "
                f"{ANCHOR_MANIFEST_BASENAME} -- the 100 % anchors are pinned, never guessed")
            continue
        claims.setdefault(slot, []).append(path)
    for (K, seed), paths in sorted(claims.items()):
        if len(paths) > 1:
            violations.append(f"K{K} s{seed}: more than one cell claims this anchor slot ("
                              + ", ".join(os.path.basename(p) for p in paths) + ")")
            continue
        metrics, _, bad = _read_cell(paths[0], arm)
        if metrics is None:
            violations += [f"K{K} s{seed} ({os.path.basename(paths[0])}): {b}" for b in bad]
            continue
        cells[K][seed] = {"path": paths[0], "metrics": metrics}
    for K in ks:
        for seed in seeds:
            if seed not in cells[K]:
                violations.append(f"K{K} s{seed}: no anchor cell for the {arm} arm in "
                                  f"{directory}; the paired 100 % form needs all of them")
    return cells, violations


def effective_epochs(manifest, max_steps=names.MAX_STEPS, global_batch=GLOBAL_BATCH):
    """``({fraction %: effective epochs}, violations)`` -- the fixed-compute disclosure.

    Recomputed here as draws / |S_f| for every fraction including 100 %, then cross-checked
    against the number Round A recorded: one formula for all four points means the 25 %
    arm's ~4x revisit rate cannot come out of a different arithmetic than the anchor's.
    """
    draws = max_steps * global_batch
    epochs, violations = {}, []
    blocks = manifest.get("fractions") or {}
    for tag, fraction in sorted(names.FRACTIONS.items()):
        entry = next((blocks[key] for key in (str(fraction), f"{fraction:g}", tag)
                      if key in blocks), None)
        if entry is None:
            violations.append(f"the split manifest has no block for fraction {fraction}")
            continue
        pct = int(round(fraction * 100))
        epochs[pct] = draws / int(entry["final"])
        recorded = entry.get("effective_epochs_at_40k_x64")
        if recorded is None or not math.isclose(float(recorded), epochs[pct],
                                                rel_tol=1e-9, abs_tol=1e-6):
            violations.append(
                f"fraction {fraction}: the manifest records {recorded!r} effective epochs, "
                f"but {draws} draws / {entry['final']} targets = {epochs[pct]:.6f}")
    histogram = manifest.get("full_split_context_histogram")
    if isinstance(histogram, dict) and histogram:
        epochs[ANCHOR_PCT] = draws / sum(int(v) for v in histogram.values())
    else:
        violations.append("the split manifest carries no full_split_context_histogram, so "
                          f"the {ANCHOR_PCT} % anchor's effective epochs cannot be computed")
    return epochs, violations
