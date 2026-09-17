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
import argparse
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys

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
#: The raw anchor cells are pinned beside them in this file: one entry per JSON, each
#: carrying ``sha256`` and (optionally, when the basename does not say) ``K``/``seed``.
#: D10's P1 seed-42 K=8 anchor is the screen cell, whose basename names neither.
ANCHOR_MANIFEST_BASENAME = "anchor_cells.json"
#: The sha256 of that screen cell in the FLAC checkout, quoted by plan D10 and by the
#: codex D3 review. Recorded here so the pin can be checked without opening the plan;
#: the assembler never hard-codes which FILE an anchor is, only what its bytes must be.
P1_SCREEN_K8_S42_SHA256 = \
    "8bd130a70442fff9f247677a046efaee9c8bfd983a2630ca8380a33cb0276f04"
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


def check_cell_endpoints(record):
    """Violations about the six scored endpoints of one metrics record."""
    metrics = record.get("metrics") if isinstance(record, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        return [f"metrics is {metrics!r}, expected a non-empty object"]
    bad = []
    for key in METRIC_KEYS.values():
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            bad.append(f"metrics.{key} is {value!r}, expected a finite number")
    return bad


def check_cell_protocol(record, arm):
    """Violations of one record against its arm's protocol and this experiment's endpoints.

    Used for the 100 % ANCHOR cells, which were scored years-of-commits ago from another
    checkpoint entirely; the new cells go through ``names.check_metrics`` instead, which
    additionally binds them to a specific checkpoint and digest.
    """
    if not isinstance(record, dict):
        return [f"is a {type(record).__name__}, not a metrics record"]
    want = cell_protocol(arm)
    bad = check_cell_endpoints(record)
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


def _load_json(path):
    try:
        with open(path) as fin:
            return json.load(fin), []
    except (OSError, ValueError) as err:
        return None, [f"cannot be read as JSON ({type(err).__name__}: {err})"]


def _read_cell(path, arm):
    """``(metrics, record, violations)`` for one ANCHOR cell; metrics is None if refused."""
    record, bad = _load_json(path)
    if record is None:
        return None, None, bad
    bad = check_cell_protocol(record, arm)
    if bad:
        return None, record, bad
    return ({metric: float(record["metrics"][key]) for metric, key in METRIC_KEYS.items()},
            record, [])


def load_run(nas_root, arm, tag, seeds=names.SEEDS, ks=names.K_VALUES, expect_n=None):
    """Every cell of one training run, bound to the checkpoint the run directory holds.

    The binding is the point (codex D3 finding 1). The final checkpoint is discovered and
    hashed ONCE, and every cell is then required to name that file, to carry the digest the
    evaluator stamped in when it loaded it, and to come with a sibling prediction bundle
    that agrees about seed, K, split and protocol -- plan §2's completion contract, verbatim.
    A cell that fails any of those is DROPPED, not inserted with a note beside it: a row
    that keeps five numbers is a row that can still produce a verdict, and a verdict over
    numbers of unknown provenance is the failure this whole gate exists to prevent.

    Ten cells that uniformly carry the *wrong* digest used to pass, because only
    disagreement among them was checked; now the comparison is against the bytes on disk.

    ``expect_n`` is for fixtures only -- production leaves it ``None`` so the split size is
    ``names.N_ITEMS_UNSEEN``. Checking the bundles costs one torch.load per cell (~260 MB
    each in production); that is the price of knowing the numbers are the ones scored.
    """
    n_items = names.N_ITEMS_UNSEEN if expect_n is None else expect_n
    run = names.run_id(arm, tag)
    run_dir = os.path.join(nas_root, run)
    out = {"run_id": run, "arm": arm, "fraction_tag": tag, "run_dir": run_dir, "ckpt": None,
           "ckpt_sha256": None, "protocol": cell_protocol(arm),
           "cells": {K: {} for K in ks}, "violations": []}
    try:
        out["ckpt"] = final_checkpoint(run_dir)
        out["ckpt_sha256"] = names.file_sha256(out["ckpt"])
    except (DataCurveError, OSError) as err:
        out["violations"].append(f"{run}: {err}")
        return out
    for K in ks:
        for seed in seeds:
            where = f"{run} K{K} s{seed}"
            path = names.metrics_json_path(out["ckpt"], arm, tag, K, seed)
            bundle = names.predictions_pt_path(out["ckpt"], arm, tag, K, seed)
            if not os.path.exists(path):
                out["violations"].append(f"{where}: no metrics JSON at {path}")
                continue
            bad = names.check_metrics(path, out["ckpt"], out["ckpt_sha256"],
                                      names.ARM_COND_METHOD[arm], names.FRAME_AVG_ANGLES,
                                      names.ROTATE_DEG, names.COND_AUTOCAST)
            record, unreadable = _load_json(path)
            bad += unreadable or check_cell_endpoints(record)
            if not os.path.exists(bundle):
                bad.append(f"no prediction bundle at {bundle}: plan §2 counts a cell only "
                           "when its metrics JSON and a loadable bundle both exist")
            else:
                bad += names.check_bundle(
                    bundle, n_items, seed, K, arm,
                    expect_eval_name=names.eval_name(arm, tag, K, seed),
                    expect_ckpt=out["ckpt"], expect_ckpt_sha256=out["ckpt_sha256"])
            if bad:
                out["violations"] += [f"{where}: {line}" for line in bad]
                continue
            out["cells"][K][seed] = {
                "path": path, "bundle": bundle, "ckpt_sha256": out["ckpt_sha256"],
                "metrics": {m: float(record["metrics"][k]) for m, k in METRIC_KEYS.items()}}
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

    Two things have to be true of a cell before it may move the fixed end of this curve
    (codex D3 finding 5). It must be PLACED: most basenames carry ``_K<k>_s<seed>``, but
    D10's seed-42 K=8 P1 anchor is the screen cell ``..._exp07_P1_screen_S40000_ema``,
    whose name says neither -- so ``anchor_cells.json`` places those by hand, and anything
    neither parseable nor placed is refused rather than guessed. And it must be PINNED: the
    same manifest carries a ``sha256`` per file, checked against the bytes on disk, because
    a protocol-compatible cell from another step renamed into a slot would otherwise change
    B_100, the data-equivalence scan and possibly the verdict, and nothing would say so.

    No manifest at all means no pins, which means no paired anchors -- not "fall back to
    names".
    """
    violations = []
    cells = {K: {} for K in ks}
    manifest_path = os.path.join(directory, ANCHOR_MANIFEST_BASENAME)
    manifest, unreadable = (_load_json(manifest_path) if os.path.exists(manifest_path)
                            else (None, [f"{manifest_path} does not exist"]))
    if not isinstance(manifest, dict):
        return cells, [f"{directory}: no usable {ANCHOR_MANIFEST_BASENAME} "
                       f"({'; '.join(unreadable) or 'not an object'}), so no anchor cell is "
                       "pinned by sha256 and the paired 100 % form is unavailable"]
    claims = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(directory), "*.json"))):
        base = os.path.basename(path)
        if base == ANCHOR_MANIFEST_BASENAME:
            continue
        entry = manifest.get(base)
        if not isinstance(entry, dict):
            violations.append(f"{base}: has no entry in {ANCHOR_MANIFEST_BASENAME}, so its "
                              "bytes are not pinned and it cannot serve as an anchor")
            continue
        slot = None
        if "K" in entry and "seed" in entry:
            slot = (int(entry["K"]), int(entry["seed"]))
        else:
            hit = _ANCHOR_CELL_RE.search(base)
            if hit:
                slot = (int(hit.group(1)), int(hit.group(2)))
        if slot is None or slot[0] not in ks or slot[1] not in seeds:
            violations.append(
                f"{base}: names no K/seed this experiment evaluates and its "
                f"{ANCHOR_MANIFEST_BASENAME} entry does not place it -- the 100 % anchors "
                "are pinned, never guessed")
            continue
        claims.setdefault(slot, []).append((path, entry))
    for (K, seed), found in sorted(claims.items()):
        if len(found) > 1:
            violations.append(f"K{K} s{seed}: more than one cell claims this anchor slot ("
                              + ", ".join(os.path.basename(p) for p, _ in found) + ")")
            continue
        path, entry = found[0]
        base = os.path.basename(path)
        pinned = entry.get("sha256")
        if not isinstance(pinned, str) or len(pinned) != 64:
            violations.append(f"{base}: its {ANCHOR_MANIFEST_BASENAME} entry carries no "
                              f"sha256 ({pinned!r}); an unpinned anchor is not an anchor")
            continue
        try:
            digest = names.file_sha256(path)
        except OSError as err:
            violations.append(f"{base}: cannot be hashed, so its sha256 pin cannot be "
                              f"checked ({err})")
            continue
        if digest != pinned:
            violations.append(f"{base}: hashes to sha256 {digest}, not the pinned {pinned} "
                              "-- these are not the bytes the anchor was measured from")
            continue
        metrics, _, bad = _read_cell(path, arm)
        if metrics is None:
            violations += [f"K{K} s{seed} ({base}): {line}" for line in bad]
            continue
        cells[K][seed] = {"path": path, "sha256": digest, "metrics": metrics}
    for K in ks:
        for seed in seeds:
            if seed not in cells[K]:
                violations.append(f"K{K} s{seed}: no pinned anchor cell for the {arm} arm "
                                  f"in {directory}; the paired 100 % form needs all of them")
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


#: Exit codes. 4 is "the artifacts on disk do not yet support the table that was asked
#: for" -- a state the launcher and the analyst both need to be able to branch on.
EXIT_OK, EXIT_INPUT_ERROR, EXIT_INCOMPLETE = 0, 2, 4
#: Round A's manifest, next to the split files it describes.
DEFAULT_SPLIT_MANIFEST = os.path.join(names.REPO_ROOT, "data", "AR",
                                      "train_frac_manifest_s2026.json")
#: Plan §11, carried into every results artifact verbatim rather than remembered.
DISCLOSURES = (
    f"Fixed-compute estimand -- a fixed-compute efficiency curve under jointly reduced "
    f"target and context diversity: every run is {names.MAX_STEPS:,} optimizer steps x a global "
    f"batch of {GLOBAL_BATCH} = {names.MAX_STEPS * GLOBAL_BATCH:,} target draws, so the "
    "25 % arm revisits each RIR ~4x more often than the 100 % arm (effective epochs per "
    "fraction are tabulated below). This is NOT a fixed-epoch learning curve.",
    f"Aggregation is the pinned evaluator's AR convention: one item-weighted value over "
    f"the whole {names.N_ITEMS_UNSEEN:,}-item unseen split per eval seed, then mean +- sd "
    f"over seeds {names.SEEDS[0]}-{names.SEEDS[-1]}. No per-scene averaging anywhere in "
    "this experiment.",
    "One training seed per cell: sd(d_s) is EVAL-seed noise, not training-run uncertainty "
    "(exp_12 Amendment 2), so the verdict is descriptive. The known step-to-step band on "
    "this stack is ~ +-0.5 T60.",
    "Nested subsets with jointly reduced target AND context diversity; both arms are "
    "scored under their own conditioning protocol with all four flags explicit "
    "(announcement 05).",
    "Frame-average chunking is N/A on this pin (announcement 06): the cyl arm trains and "
    "scores with frame_avg_angles = [0], orbit size 1.",
)


def _stringify(obj):
    """JSON uses string keys; make that true in memory too, so a reload compares equal."""
    if isinstance(obj, dict):
        return {str(key): _stringify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(value) for value in obj]
    return obj


def _row(metric, pct, cyl_by_seed, van_by_seed, seeds, source):
    """One (metric, K, fraction) row: both arms aggregated and their oriented benefit.

    The benefit falls back to the marginal form when the arms are not exactly paired --
    and the row is marked incomplete in the same breath, so nothing downstream can mistake
    a difference of four-seed means for the pre-registered B_f.
    """
    cyl_agg = aggregate(cyl_by_seed, seeds)
    van_agg = aggregate(van_by_seed, seeds)
    benefit = paired_benefit(metric, cyl_by_seed, van_by_seed, seeds) \
        or marginal_benefit(metric, cyl_agg, van_agg)
    return {"fraction_pct": pct, "cyl": cyl_agg, "van": van_agg, "benefit": benefit,
            "complete": bool(cyl_agg["complete"] and van_agg["complete"]), "source": source}


def build_curve(nas_root, anchors_path, split_manifest_path=None, anchor_cell_dirs=None,
                fraction_tags=None, ks=names.K_VALUES, seeds=names.SEEDS, metrics=METRICS,
                generated_at=None, expect_n=None):
    """The whole data-curve document: six runs, two anchors, one verdict, one provenance.

    The 100 % point is *paired* only when raw per-seed anchor cells are supplied for both
    arms and complete (plan D10); otherwise it degrades to the marginal form and says so.
    An incomplete primary cell makes the verdict PENDING rather than a verdict computed
    over whichever cells happen to exist.
    """
    tags = sorted(fraction_tags or names.FRACTIONS)
    pcts = [int(round(names.FRACTIONS[tag] * 100)) for tag in tags] + [ANCHOR_PCT]
    violations = []

    runs = {}
    for arm in names.ARMS:
        for tag in tags:
            runs[(arm, tag)] = load_run(nas_root, arm, tag, seeds, ks, expect_n)
            violations += runs[(arm, tag)]["violations"]

    reference = load_anchor_reference(anchors_path, ks=ks)
    anchor_cells, anchor_form = {}, "marginal"
    anchor_reason = "no raw per-seed anchor cells were supplied for both arms (plan D10)"
    if anchor_cell_dirs:
        refusals = []
        for arm in names.ARMS:
            directory = anchor_cell_dirs.get(arm)
            if not directory:
                refusals.append(f"{arm}: no directory given")
                continue
            anchor_cells[arm], bad = load_anchor_cells(directory, arm, seeds, ks)
            refusals += [f"{arm}: {b}" for b in bad]
        violations += [f"anchor {line}" for line in refusals]
        anchor_form = "paired" if not refusals else "marginal"
        anchor_reason = ("every raw anchor cell verified against its pinned sha256"
                         if not refusals else
                         "the raw anchor cells did not verify -- " + "; ".join(refusals[:3])
                         + (f" (+{len(refusals) - 3} more)" if len(refusals) > 3 else ""))

    epochs = {}
    if split_manifest_path:
        with open(split_manifest_path) as fin:
            epochs, bad = effective_epochs(json.load(fin))
        violations += bad

    curve = {}
    for K in ks:
        curve[f"K{K}"] = {}
        for metric in metrics:
            rows = {}
            for tag, pct in zip(tags, pcts):
                by_seed = {arm: {seed: cell["metrics"][metric] for seed, cell
                                 in runs[(arm, tag)]["cells"][K].items()}
                           for arm in names.ARMS}
                rows[str(pct)] = _row(metric, pct, by_seed["cyl"], by_seed["van"], seeds,
                                      source=f"dc_*_f{tag} @ {nas_root}")
            if anchor_form == "paired":
                by_seed = {arm: {seed: cell["metrics"][metric]
                                 for seed, cell in anchor_cells[arm][K].items()}
                           for arm in names.ARMS}
                rows[str(ANCHOR_PCT)] = _row(metric, ANCHOR_PCT, by_seed["cyl"],
                                             by_seed["van"], seeds,
                                             source="raw 100 % anchor cells (exp_13 tier S)")
            else:
                cyl_agg, van_agg = (dict(reference[arm][K][metric]) for arm in ("cyl", "van"))
                rows[str(ANCHOR_PCT)] = {
                    "fraction_pct": ANCHOR_PCT, "cyl": cyl_agg, "van": van_agg,
                    "benefit": marginal_benefit(metric, cyl_agg, van_agg), "complete": True,
                    "source": os.path.basename(anchors_path)}
            curve[f"K{K}"][metric] = rows

    primary = curve[f"K{PRIMARY_K}"]
    pending = [f"{m} @ {pct} %" for m in PRIMARY_METRICS for pct in pcts
               if not primary[m][str(pct)]["complete"]
               or primary[m][str(pct)]["benefit"] is None]
    if pending:
        verdict_block = {"verdict": "PENDING", "rule": None, "primary_K": PRIMARY_K,
                         "primary_metrics": list(PRIMARY_METRICS), "annotations": [],
                         "reason": "not every primary cell is complete: " + ", ".join(pending)}
    else:
        verdict_block = verdict(
            {m: {pct: primary[m][str(pct)]["benefit"]["mean"] for pct in pcts}
             for m in PRIMARY_METRICS}, fractions=tuple(pcts))

    equivalence = {}
    for K in ks:
        equivalence[f"K{K}"] = {}
        for metric in metrics:
            rows = curve[f"K{K}"][metric]
            van_100 = rows[str(ANCHOR_PCT)]["van"]["mean"]
            g = {pct: oriented_benefit(metric, van_100, rows[str(pct)]["cyl"]["mean"])
                 for pct in pcts
                 if van_100 is not None and rows[str(pct)]["cyl"]["mean"] is not None}
            if len(g) != len(pcts):
                equivalence[f"K{K}"][metric] = {
                    "kind": "pending", "outcome": "pending (missing cells)", "g": g,
                    "f_star": None, "bracket": None, "fractions": list(pcts)}
            else:
                equivalence[f"K{K}"][metric] = data_equivalence(g, fractions=tuple(pcts))

    disclosures = list(DISCLOSURES)
    disclosures.append(
        "The 100 % point is PAIRED: raw per-seed anchor cells were supplied for both arms, "
        "so B_100 carries a per-seed sd like every other fraction."
        if anchor_form == "paired" else
        "The 100 % point is reported in the MARGINAL form (`anchor_form: marginal`; "
        "mean +- sd per arm, B_100 = difference of means, no paired sd), because "
        + anchor_reason + ". The verdict rule uses B only, so it still evaluates.")

    document = {
        "schema": "exp_14_data_curve/1",
        "generated_at": generated_at or datetime.datetime.now().astimezone().isoformat(),
        "sources": {"nas_root": nas_root, "anchor_reference": anchors_path,
                    "split_manifest": split_manifest_path,
                    "anchor_cell_dirs": dict(anchor_cell_dirs or {})},
        "anchor_form": anchor_form,
        "anchor_form_reason": anchor_reason,
        "fractions_pct": pcts,
        "effective_epochs": epochs,
        "curve": curve,
        "verdict": verdict_block,
        "data_equivalence": equivalence,
        "provenance": {
            "runs": {run["run_id"]: {
                "arm": run["arm"], "fraction_tag": run["fraction_tag"],
                "run_dir": run["run_dir"], "ckpt": run["ckpt"],
                "ckpt_sha256": run["ckpt_sha256"], "protocol": run["protocol"],
                "cells": {f"K{K}": {seed: cell["path"] for seed, cell
                                    in sorted(run["cells"][K].items())} for K in ks},
            } for run in runs.values()},
            "anchor_cells": {arm: {f"K{K}": {seed: cell["path"] for seed, cell
                                             in sorted(cells[K].items())} for K in ks}
                             for arm, cells in anchor_cells.items()},
        },
        "disclosures": disclosures,
        "violations": violations,
        "complete": not violations and verdict_block["verdict"] != "PENDING",
    }
    return _stringify(document)


def _fmt(cell, dp=REPORT_DP):
    """``mean +- sd`` at the reporting precision; an absent sd prints as a bare mean."""
    if not cell or cell.get("mean") is None:
        return "--"
    if cell.get("sd") is None:
        return f"{cell['mean']:.{dp}f}"
    return f"{cell['mean']:.{dp}f} +- {cell['sd']:.{dp}f}"


def _benefit_header(metric):
    return ("B_f = van - cyl" if metric in LOWER_IS_BETTER else "B_f = cyl - van")


def render_markdown(doc):
    """The human-readable twin of ``data_curve.json`` -- same numbers, same caveats.

    Every table states its own orientation in the column header, every row says whether it
    is complete, and the provenance section names the file each number came from: the
    point is that a reader can re-derive the verdict without trusting this script.
    """
    out = ["# exp_14 -- data-efficiency curve: CylDINO core S vs stock DINOv3 S "
           f"(@{names.MAX_STEPS//1000}k, AR unseen)", "",
           f"Generated {doc['generated_at']} by `src.tools.data_curve.assemble` from "
           f"`{doc['sources']['nas_root']}`.", "",
           "## Estimand and disclosures", ""]
    out += [f"- {line}" for line in doc["disclosures"]]
    epochs = doc.get("effective_epochs") or {}
    if epochs:
        out += ["", "### Effective epochs per fraction", "",
                "| fraction | " + " | ".join(f"{pct} %" for pct in doc["fractions_pct"]) + " |",
                "|---" * (len(doc["fractions_pct"]) + 1) + "|",
                "| effective epochs | " + " | ".join(
                    f"{epochs[str(pct)]:.2f}" if str(pct) in epochs else "--"
                    for pct in doc["fractions_pct"]) + " |"]
    for key, block in sorted(doc["curve"].items()):
        out += ["", f"## K = {key[1:]}", ""]
        for metric in METRICS:
            better = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
            out += [f"### {metric} ({better})", "",
                    f"| fraction | eff. epochs | CylDINO core S | stock DINOv3 S | "
                    f"{_benefit_header(metric)} | form | n | complete |",
                    "|---" * 8 + "|"]
            for pct in doc["fractions_pct"]:
                row = block[metric][str(pct)]
                benefit = row["benefit"] or {}
                out.append(
                    f"| {pct} % | "
                    + (f"{epochs[str(pct)]:.2f}" if str(pct) in epochs else "--")
                    + f" | {_fmt(row['cyl'])} | {_fmt(row['van'])} | {_fmt(benefit)} | "
                    + f"{benefit.get('form', '--')} | {row['cyl'].get('n', 0)} | "
                    + ("yes" if row["complete"] else "**NO**") + " |")
            out.append("")
    v = doc["verdict"]
    out += ["## Verdict (pre-registered, plan §1)", "",
            f"**{v['verdict']}** -- rule {v.get('rule') or v.get('reason', '')}", ""]
    if v.get("annotations"):
        out += [f"- {a}" for a in v["annotations"]] + [""]
    if v.get("benefits"):
        out += [f"| metric (K = {v['primary_K']}) | "
                + " | ".join(f"B_{pct}" for pct in doc["fractions_pct"]) + " | non-positive at |",
                "|---" * (len(doc["fractions_pct"]) + 2) + "|"]
        for metric in v["primary_metrics"]:
            out.append(f"| {metric} | " + " | ".join(
                f"{v['benefits'][metric][str(pct)]:+.{REPORT_DP}f}"
                for pct in doc["fractions_pct"])
                + " | " + (", ".join(f"{p} %" for p in v["non_positive"][metric]) or "--") + " |")
        out.append("")
    out += ["## Data equivalence -- how little data CylDINO needs to match stock @100 %", "",
            "g(f) = cyl(f) vs stock@100 %, oriented so g >= 0 means \"at least as good\"; "
            "upward scan, first match.", "",
            "| K | metric | " + " | ".join(f"g({pct})" for pct in doc["fractions_pct"])
            + " | outcome | f* |", "|---" * (len(doc["fractions_pct"]) + 4) + "|"]
    for key, block in sorted(doc["data_equivalence"].items()):
        for metric in METRICS:
            eq = block[metric]
            star = "--" if eq.get("f_star") is None else f"{eq['f_star']:.1f} %"
            out.append(f"| {key[1:]} | {metric} | " + " | ".join(
                (f"{eq['g'][str(pct)]:+.{REPORT_DP}f}" if str(pct) in eq["g"] else "--")
                for pct in doc["fractions_pct"]) + f" | {eq['outcome']} | {star} |")
    out += ["", "## Provenance", ""]
    for run_id, run in sorted(doc["provenance"]["runs"].items()):
        protocol = ", ".join(f"{k}={v}" for k, v in sorted(run["protocol"].items()))
        out += [f"### {run_id} ({run['arm']} arm, fraction {run['fraction_tag']})", "",
                f"- checkpoint: `{run['ckpt']}`",
                f"- ckpt_sha256: `{run['ckpt_sha256']}`",
                f"- protocol: {protocol}", "- cells:", ""]
        out += [f"  - `{path}`" for K in sorted(run["cells"])
                for _, path in sorted(run["cells"][K].items())]
        out.append("")
    for arm, cells in sorted(doc["provenance"].get("anchor_cells", {}).items()):
        out += [f"### 100 % anchor cells ({arm} arm)", ""]
        out += [f"  - `{path}`" for K in sorted(cells) for _, path in sorted(cells[K].items())]
        out.append("")
    if doc["violations"]:
        out += ["## Open violations (this table is NOT complete)", ""]
        out += [f"- {line}" for line in doc["violations"]] + [""]
    return "\n".join(out) + "\n"


def main(argv=None):
    """CLI: assemble ``data_curve.json`` + ``data_curve.md`` from the NAS run dirs."""
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.data_curve.assemble",
        description="Assemble exp_14's data-efficiency curve from the cells on the NAS.")
    parser.add_argument("--nas-root", required=True,
                        help="the directory holding the dc_<arm>_f<tag> run directories")
    parser.add_argument("--anchors", required=True,
                        help="exp_13's tier_S_reference.json (the 100 %% anchors)")
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST,
                        help="Round A's train_frac_manifest_s2026.json (effective epochs)")
    parser.add_argument("--anchor-cells-p1", default=None,
                        help="directory of the 10 raw P1 (van) 100 %% cells; with "
                             "--anchor-cells-cyl this promotes B_100 to the paired form")
    parser.add_argument("--anchor-cells-cyl", default=None,
                        help="directory of the 10 raw cylNoSSL 100 %% cells")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if anything is missing, mismatched or unpaired")
    args = parser.parse_args(argv)

    dirs = {arm: path for arm, path in (("van", args.anchor_cells_p1),
                                        ("cyl", args.anchor_cells_cyl)) if path}
    try:
        doc = build_curve(args.nas_root, args.anchors,
                          split_manifest_path=args.split_manifest,
                          anchor_cell_dirs=dirs or None)
    except (OSError, ValueError, KeyError) as err:
        print(f"assemble: {type(err).__name__}: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.out_json:
        with open(args.out_json, "w") as fout:
            json.dump(doc, fout, indent=1, sort_keys=True)
    if args.out_md:
        with open(args.out_md, "w") as fout:
            fout.write(render_markdown(doc))
    print(f"verdict: {doc['verdict']['verdict']} | anchors: {doc['anchor_form']} | "
          f"complete: {doc['complete']} | violations: {len(doc['violations'])}")
    for line in doc["violations"]:
        print(f"  ! {line}", file=sys.stderr)
    if args.strict and doc["violations"]:
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":     # pragma: no cover - exercised through main() in tests
    sys.exit(main())
