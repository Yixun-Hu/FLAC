#!/usr/bin/env python3
"""exp_14 — collect the yaw-generalization campaign into readouts (plan §5.6).

The campaign is 106 cells (plan §3.1). This module turns the ones that landed
into the pre-registered readouts of plan §4 — absolute robustness ``m_R``, the
paired degradation ``Δ = m_R − m_Z``, the endpoint contrasts H-P/H-M/H-S, the
descriptive adjacent contrasts — and, just as importantly, refuses to turn the
rest into anything.

Three refusals are the whole design:

* **A cell that cannot be proven is not evidence.** Every artifact goes through
  ``exp14_validate_cell`` (the same predicate the screen driver and the wave
  submitter run) before it can enter a mean. The rules are IMPORTED from there,
  never restated here: one validator, three callers.
* **A partial block is PENDING, never a number.** Four seeds of a five-seed block
  do not average into a four-seed estimate that later gets read as the cell.
  ``aggregate_cell`` carries no values at all until the block is complete.
* **An unmatched contrast is BLOCKED.** The cross-arm comparison is only a
  comparison because every arm saw the same items with the same rotations
  (plan §3.3); if a hash equality fails, the affected contrast renders BLOCKED
  with the named reason instead of a plausible-looking number.

Usage
-----
    python3 yaw_gen_collect.py --output-root outputs_FLAC --pin <sha> \\
        [--out yaw_gen_results.md] [--json yaw_gen_results.json] \\
        [--expected-count 6337] [--ckpt-expect exp14_ckpt_expect.json]
"""
import argparse
import importlib.util
import json
import math
import os
import statistics as st
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def _load_validator(path=None):
    """Import ``exp14_validate_cell`` from an explicit path (never by sys.path).

    The collector is the third caller of the round-2 validator, and it must be
    the SAME predicate: a cell the submitter would have refused to skip cannot
    become a number here."""
    path = os.path.abspath(path or os.path.join(_HERE, "exp14_validate_cell.py"))
    spec = importlib.util.spec_from_file_location("exp14_validate_cell", path)
    if spec is None or spec.loader is None:            # pragma: no cover - unreachable
        raise ImportError(f"cannot load the exp_14 validator from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_validator()

# The registered campaign is the VALIDATOR's, re-exported rather than restated:
# a collector with its own copy of the grid would report on cells nobody ran.
expected_grid = V.expected_grid
ARMS = V.ARMS
SEEDS = V.SEEDS
KS = V.KS
STEP = V.STEP

# Plan §4: the fixed arm order for adjacent contrasts. Never data-dependent.
ARM_ORDER = ("VANL", "C4L", "C8", "C16", "C32")
assert set(ARM_ORDER) == set(ARMS)


class ArtifactError(ValueError):
    """A named reason an artifact cannot even be read as this cell's."""


# --------------------------------------------------------------------------- #
# what one landed cell is
# --------------------------------------------------------------------------- #
CellArtifact = namedtuple("CellArtifact", "cell path record screenmeta stream")

# The slim view kept after validation: everything the cross-cell reasoning needs
# and nothing that would keep 106 × 6,337 tuples alive at once.
CellData = namedtuple("CellData",
                      "cell path metrics flat_metrics input_hash assignment_hash "
                      "offsets source_sha")

Expectation = namedtuple("Expectation", "pin ckpt_sha expected_count")


class CampaignExpectation:
    """The campaign pin, the audited per-arm checkpoint digests, the stream count.

    Fail-closed BY CONSTRUCTION (the round-2 review's B4 lesson, one level up): a
    validator can be handed ``pin=None`` and answer "valid" for everything it was
    able to check. A collector run that never learned which commit and which
    checkpoint produced its numbers must not be able to start at all.
    """

    def __init__(self, pin, ckpt_sha_by_arm, expected_count=V.EXPECTED_COUNT):
        if not pin or not isinstance(pin, str):
            raise ValueError("a campaign pin (source commit sha) is required: a number "
                             "whose evaluator commit is unknown is not a measurement")
        missing = [a for a in ARMS
                   if not isinstance((ckpt_sha_by_arm or {}).get(a), str)
                   or len(ckpt_sha_by_arm[a]) != 64]
        if missing:
            raise ValueError(f"audited checkpoint sha256 missing for arm(s) {missing}: "
                             "publish them with exp14_hash_ckpts.py --write")
        if int(expected_count) <= 0:
            raise ValueError(f"expected stream count must be positive, got {expected_count}")
        self.pin = pin
        self.ckpt_sha_by_arm = dict(ckpt_sha_by_arm)
        self.expected_count = int(expected_count)

    def for_cell(self, cell):
        return Expectation(self.pin, self.ckpt_sha_by_arm[cell.arm], self.expected_count)

    @classmethod
    def from_files(cls, pin, ckpt_expect=None, expected_count=V.EXPECTED_COUNT):
        return cls(pin, V.load_ckpt_expect(ckpt_expect or V.CKPT_EXPECT),
                   expected_count=expected_count)


def _read_json_object(path, label):
    if not os.path.isfile(path):
        raise ArtifactError(f"{label} artifact missing: {path}")
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"could not parse {label} JSON {path}: {exc}")
    if not isinstance(obj, dict):
        raise ArtifactError(f"{label} {path} is not a JSON object at the top level "
                            f"(got {type(obj).__name__})")
    return obj


def parse_cell_artifact(path):
    """Read one cell's three files and say WHICH registered cell they claim to be.

    Identity comes from the record's own ``eval_name`` through the validator's
    ``parse_eval_name``, so an artifact that names an unregistered cell is
    refused here rather than quietly aggregated under a cell it resembles.
    """
    record = _read_json_object(path, "metrics")
    for key in ("eval_name", "metrics"):
        if key not in record:
            raise ArtifactError(f"metrics record {path} has no {key!r}: it cannot be "
                                "read as an exp_14 cell")
    if not isinstance(record["metrics"], dict) or not record["metrics"]:
        raise ArtifactError(f"metrics record {path} carries an empty metrics block")
    try:
        cell = V.parse_eval_name(record["eval_name"])
    except ValueError as exc:
        raise ArtifactError(f"{path}: {exc}")
    screenmeta = _read_json_object(V.screenmeta_path(path), "screenmeta")
    stream = _read_json_object(V.stream_path(path), "stream")
    # Schema versions are refused BEFORE any field is read under them: a sidecar
    # written under another contract is not a sidecar with a wrong value in it,
    # it is a payload whose fields mean something else.
    if stream.get("schema_version") != V.STREAM_SCHEMA_VERSION:
        raise ArtifactError(
            f"{V.stream_path(path)}: stream schema_version {stream.get('schema_version')!r}"
            f" != {V.STREAM_SCHEMA_VERSION} — written under another contract")
    if stream.get("fingerprint_schema") != V.FINGERPRINT_SCHEMA:
        raise ArtifactError(
            f"{V.stream_path(path)}: fingerprint_schema {stream.get('fingerprint_schema')!r}"
            f" != {V.FINGERPRINT_SCHEMA} — the context fingerprint means something else")
    return CellArtifact(cell=cell, path=path, record=record, screenmeta=screenmeta,
                        stream=stream)


def validate_cell_provenance(artifact, expected):
    """Named reasons this cell is not the registered cell it claims ([] = valid).

    Every rule is the validator's; this function only supplies the three parsed
    payloads it already read, so the artifacts are not read twice (the stream
    sidecar is ~6,337 tuples and there are 106 of them).
    """
    cell = artifact.cell
    reasons = list(V.validate_metrics_record(artifact.record, cell, pin=expected.pin,
                                             expected_count=expected.expected_count))
    reasons += V.validate_screenmeta(artifact.screenmeta, cell, pin=expected.pin,
                                     ckpt_sha=expected.ckpt_sha,
                                     expected_count=expected.expected_count)
    reasons += V.validate_stream_record(artifact.stream, cell,
                                        expected_count=expected.expected_count,
                                        record=artifact.record)
    return reasons


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


# --- the per-metric aggregation ruling (Planner, 2026-08-11, PRE-REGISTERED ---
# before any cell has run). Recorded verbatim because it decides what every
# number below MEANS:
#
#   "per-scene mean applies to the ACOUSTIC-PARAMETER family only — T60 (incl.
#    Invalid-T60 handling), C50, EDT — matching the paper convention the plan §4
#    intended. RETRIEVAL (RIR_to_GT_RIR_R@k, and the quarantined RIR_to_geom_R@k)
#    and FD use the SPLIT-LEVEL global metrics: within-scene retrieval among ~370
#    items is a different, easier task whose levels are incomparable to every
#    previously published number in this program, and exp_01's noise-floor
#    calibration against released Table-1 was on the global quantity; one-room
#    Frechet is additionally small-sample biased. Co-primaries therefore: T60%
#    (per-scene mean) + RIR_to_GT_RIR_R@1 (split-level)."
#
# Both sources exist in every exp_14 record — the flat `metrics` block and the
# `by_scene` block — so this is a reading rule, not a measurement change, and
# by_scene stays REQUIRED for every cell because the acoustic family needs it.
ACOUSTIC_METRICS = ("T60", "C50", "EDT", "Invalid T60")
SPLIT_LEVEL_METRICS = ("FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
                       "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5",
                       "RIR_to_geom_R@10")
AGGREGATION_SOURCE = dict(
    [(m, "scene-mean") for m in ACOUSTIC_METRICS]
    + [(m, "split") for m in SPLIT_LEVEL_METRICS])


def aggregation_source(metric):
    """``'scene-mean'`` or ``'split'`` — no default.

    A metric with no ruled source cannot be aggregated at all: which of two
    different quantities it names would be decided by accident.
    """
    name = canonical_metric(metric)
    if name not in AGGREGATION_SOURCE:
        raise KeyError(f"no ruled aggregation source for metric {metric!r}: the "
                       "per-metric ruling decides scene-mean vs split-level, and "
                       "guessing would silently choose one of two different numbers")
    return AGGREGATION_SOURCE[name]


def _scene_column(by_scene, metric):
    """``(mean, reasons)`` over the scenes — the acoustic family's observation."""
    reasons, column = [], []
    for scene in sorted(by_scene):
        payload = by_scene[scene]
        if not isinstance(payload, dict) or metric not in payload:
            return None, [f"scene {scene!r} does not report {metric}: the per-scene "
                          "mean is undefined over a missing scene"]
        if not _finite(payload[metric]):
            return None, [f"scene {scene!r} reports {metric}={payload[metric]!r}, "
                          "which is not a finite number"]
        column.append(float(payload[metric]))
    return (st.mean(column) if column else None), reasons


def flat_observation(record, metrics=None):
    """``(values, reasons)`` — the split-level payload, validated for G5's use."""
    metrics = tuple(G5_METRICS if metrics is None else metrics)
    flat = record.get("metrics")
    if not isinstance(flat, dict):
        return {}, ["metrics block is missing or is not an object"]
    reasons, values = [], {}
    for metric in metrics:
        if metric not in flat:
            reasons.append(f"the split-level metrics do not report {metric}, which the "
                           "G5 reproduction check compares against exp_11")
        elif not _finite(flat[metric]):
            reasons.append(f"split-level {metric}={flat[metric]!r} is not a finite number")
        else:
            values[metric] = float(flat[metric])
    return values, reasons


def cell_observation(record, required=None, optional=None):
    """``(metrics, reasons)`` — each metric read from ITS ruled source.

    The payload is validated HERE, at the consumer (review B5), on whichever side
    the metric is read from: every reported metric must be present and finite, or
    the cell is refused by name. A metrics object that is merely non-empty passes
    the per-cell validator and then raises KeyError — or carries a NaN into a
    mean, a CI and a verdict — halfway through a contrast.
    """
    required = tuple(HEADLINE_METRICS if required is None else required)
    optional = tuple(CONFOUNDED_METRICS if optional is None else optional)
    by_scene = record.get("by_scene")
    flat = record.get("metrics")
    if not isinstance(flat, dict):
        return {}, ["metrics block is missing or is not an object"]
    # by_scene is required for EVERY cell: the acoustic family is read from it,
    # and a cell that never recorded one did not measure that estimand.
    if not isinstance(by_scene, dict) or not by_scene:
        return {}, ["by_scene is missing or empty: the acoustic family's observation "
                    "is the PER-SCENE mean, so a cell without per-scene results did "
                    "not measure the estimand"]
    reasons, values = [], []
    values = {}
    for metric in required:
        if aggregation_source(metric) == "scene-mean":
            mean, why = _scene_column(by_scene, metric)
            if why:
                reasons += why
            else:
                values[metric] = mean
            continue
        if metric not in flat:
            reasons.append(f"the split-level metrics do not report {metric}, which "
                           "this campaign reads from them by ruling")
        elif not _finite(flat[metric]):
            reasons.append(f"split-level {metric}={flat[metric]!r} is not a finite "
                           "number")
        else:
            values[metric] = float(flat[metric])
    for metric in optional:
        # descriptive-only (quarantined): absence is not a campaign failure, but a
        # value that is present must still be publishable.
        if aggregation_source(metric) != "split" or metric not in flat:
            continue
        if not _finite(flat[metric]):
            reasons.append(f"split-level {metric}={flat[metric]!r} is not a finite "
                           "number")
            continue
        values[metric] = float(flat[metric])
    return values, reasons


def slim(artifact):
    """Drop the tuple stream; keep the ROUTED observation and the hashes.

    ``(CellData, reasons)`` — a cell whose per-scene payload cannot be reduced to
    the metrics this collector reports comes back as reasons, never as data.
    """
    stream = artifact.stream
    metrics, reasons = cell_observation(artifact.record)
    flat, flat_reasons = flat_observation(artifact.record)
    reasons = list(reasons) + list(flat_reasons)
    if reasons:
        return None, reasons
    return CellData(cell=artifact.cell, path=artifact.path, metrics=metrics,
                    flat_metrics=flat,
                    input_hash=stream.get("input_hash"),
                    assignment_hash=stream.get("assignment_hash"),
                    offsets=tuple(stream.get("offsets") or ()),
                    source_sha=artifact.record.get("source_sha")), []


def load_cell(path, campaign):
    """``(CellData, reasons)`` — data only when every check passed."""
    try:
        artifact = parse_cell_artifact(path)
    except ArtifactError as exc:
        return None, [str(exc)]
    reasons = validate_cell_provenance(artifact, campaign.for_cell(artifact.cell))
    if reasons:
        return None, reasons
    return slim(artifact)


# --------------------------------------------------------------------------- #
# §3.3 assignment integrity — the equalities BETWEEN cells
# --------------------------------------------------------------------------- #
Violation = namedtuple("Violation", "kind scope detail blocks")


def _by_cell(cells):
    return {(c.cell.arm, c.cell.cell, int(c.cell.k), int(c.cell.seed)): c for c in cells}


def match_assignments(cells):
    """The plan §3.3 equality checks over whatever cells are present.

    (a) Across arms within one (cell type, K, seed): the ``input_hash`` must
        agree — otherwise the arms did not see the same items and the same
        context draws, and no cross-arm contrast at that K is a contrast. For
        the rotated blocks the ``assignment_hash`` must agree too: same items,
        same yaw per item, or the arms are not rotation-matched.
    (b) Within one (arm, K, seed): ``Z.input_hash == R.input_hash``. Z and R
        differ only in rotation, so a differing input stream means the pair is
        not a pair and that arm's Δ is unmeasurable at that seed.

    Returns a list of :class:`Violation`; each carries the scopes it BLOCKS, so
    the renderer can refuse exactly the affected contrasts and no others.
    """
    violations = []
    # Two different groupings, because they answer two different questions.
    # WHICH ITEMS ran is a property of the split and must agree across every cell
    # at one (K, seed) — including the validity controls, which evaluate the same
    # split. WHICH YAW each item received is only comparable among cells that ran
    # the SAME rotation protocol: C4L@45 and C4L@90 are both registered validity
    # cells and their assignments differ BY DESIGN, so grouping them together
    # would report the campaign's own design as an integrity failure.
    inputs, assignments = {}, {}
    for c in cells:
        key = (c.cell.cell, int(c.cell.k), int(c.cell.seed))
        inputs.setdefault(key, []).append(c)
        if c.cell.cell != "zref":        # the unrotated block assigns nothing
            assignments.setdefault(key + (c.cell.rotate_deg,), []).append(c)
    for groups, field, kind in ((inputs, "input_hash", "cross_arm_input_hash"),
                                (assignments, "assignment_hash",
                                 "cross_arm_assignment_hash")):
        for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
            celltype, k, seed = key[0], key[1], key[2]
            if len(members) < 2:
                continue                 # nothing to compare against; not a violation
            seen = {}
            for c in members:
                seen.setdefault(getattr(c, field), []).append(c.cell.arm)
            if len(seen) > 1:
                detail = "; ".join(
                    f"{(h or 'None')[:12]}: {sorted(arms)}" for h, arms in sorted(
                        seen.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))))
                angle = "" if len(key) < 4 or key[3] is None else f" @{V.fmt_deg(key[3])}°"
                violations.append(Violation(
                    kind=kind, scope=f"{celltype}{angle} K={k} s{seed}",
                    detail=(f"{celltype}{angle} cells at K={k} seed {seed} disagree on "
                            f"{field} — {detail}"),
                    blocks=(("cross_arm", k),)))
    paired = _by_cell(cells)
    for (arm, celltype, k, seed), zcell in sorted(paired.items()):
        if celltype != "zref":
            continue
        rcell = paired.get((arm, "rgen", k, seed))
        if rcell is None:
            continue
        if zcell.input_hash != rcell.input_hash:
            violations.append(Violation(
                kind="z_r_input_hash", scope=f"{arm} K={k} s{seed}",
                detail=(f"{arm} K={k} seed {seed}: the unrotated cell's input_hash "
                        f"{(zcell.input_hash or 'None')[:12]} != the rotated cell's "
                        f"{(rcell.input_hash or 'None')[:12]} — Z and R are not a pair"),
                blocks=(("paired", arm, k), ("cross_arm", k))))
    return violations


# --------------------------------------------------------------------------- #
# blocks: seeds, pairing, aggregation
# --------------------------------------------------------------------------- #
def pair_seeds(z_cells, r_cells):
    """``({seed: (z, r)}, problems)`` for one (arm, K).

    A duplicate seed does not pick a winner and an orphan is never silently
    dropped: both are named, because either one means the block on disk is not
    the block the campaign registered.
    """
    problems = []

    def _index(cells, label):
        idx, dupes = {}, set()
        for c in cells:
            seed = int(c.cell.seed)
            if seed in idx:
                dupes.add(seed)
            idx.setdefault(seed, []).append(c)
        for seed in sorted(dupes):
            problems.append(f"duplicate {label} cell for seed {seed}: "
                            + ", ".join(sorted(x.path for x in idx[seed])))
        return {s: v[0] for s, v in idx.items() if s not in dupes}

    def _homogeneous(cells, label):
        arms = {c.cell.arm for c in cells}
        ks = {int(c.cell.k) for c in cells}
        if len(arms) > 1 or len(ks) > 1:
            problems.append(f"{label} cells span more than one block "
                            f"(arms {sorted(arms)}, K {sorted(ks)}): a pairing is "
                            "defined within ONE (arm, K)")
        return arms, ks

    z_arms, z_ks = _homogeneous(z_cells, "unrotated (Z)") if z_cells else (set(), set())
    r_arms, r_ks = _homogeneous(r_cells, "rotated (R)") if r_cells else (set(), set())
    stray = (r_arms - z_arms) | (z_arms - r_arms)
    if z_arms and r_arms and stray:
        problems.append(f"orphan cells: arms {sorted(stray)} appear on only one side "
                        "of the Z/R pairing")
    if z_ks and r_ks and z_ks != r_ks:
        problems.append(f"orphan cells: K {sorted(z_ks ^ r_ks)} appears on only one side")
    zi, ri = _index(z_cells, "unrotated (Z)"), _index(r_cells, "rotated (R)")
    pairs = {}
    for seed in sorted(set(zi) | set(ri)):
        z, r = zi.get(seed), ri.get(seed)
        if z is None or r is None:
            problems.append(f"seed {seed} has no "
                            + ("unrotated (Z)" if z is None else "rotated (R)")
                            + " partner: the seed-paired Δ needs both")
            continue
        if z.cell.arm != r.cell.arm or int(z.cell.k) != int(r.cell.k):
            problems.append(f"seed {seed} pairs {z.cell.arm} K={z.cell.k} with "
                            f"{r.cell.arm} K={r.cell.k}: not the same block")
            continue
        pairs[seed] = (z, r)
    return pairs, problems


Aggregate = namedtuple("Aggregate", "status n seeds values per_seed reasons")


def aggregate_cell(seed_records, seeds=SEEDS):
    """Mean ± std over the five registered seeds, or PENDING carrying NO numbers.

    The emptiness on the PENDING path is deliberate. Rendering is not the only
    place a four-seed mean could leak — a JSON bundle, a later plot, a copied
    cell in a message — so an incomplete block never computes one.
    """
    records = list(seed_records)
    got = sorted(int(r.cell.seed) for r in records)
    reasons = []
    blocks = {(r.cell.arm, r.cell.cell, int(r.cell.k)) for r in records}
    if len(blocks) > 1:
        reasons.append(f"block spans more than one (arm, cell, K): {sorted(blocks)}")
    if len(set(got)) != len(got):
        reasons.append(f"duplicate seeds present: {got}")
    missing = [s for s in seeds if s not in got]
    if missing:
        reasons.append(f"{len(got)}/{len(seeds)} seeds on disk; missing {missing}")
    if reasons:
        return Aggregate(status="PENDING", n=len(records), seeds=tuple(got),
                         values={}, per_seed={}, reasons=tuple(reasons))
    keys = sorted(set.intersection(*[set(r.metrics) for r in records]))
    per_seed = {k: {int(r.cell.seed): float(r.metrics[k]) for r in records} for k in keys}
    values = {k: (st.mean(per_seed[k].values()), st.stdev(per_seed[k].values()))
              for k in keys}
    return Aggregate(status="OK", n=len(records), seeds=tuple(got), values=values,
                     per_seed=per_seed, reasons=())


# --------------------------------------------------------------------------- #
# the golden assignment (gate G3's collector half)
# --------------------------------------------------------------------------- #
def _golden_offsets_impl(seed, n, img_w=V.IMG_W):
    """The offsets rotation seed ``seed`` MUST have produced, recomputed here.

    Recomputed with ``draw_yaw_offsets`` itself — the function the evaluator
    calls — rather than compared against a stored constant, so the gate proves
    the landed cell was drawn by the registered algorithm and not merely that it
    matches a number somebody once wrote down.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    import torch                                   # local: the collector is torch-free
    from src.data.yaw_rotation import draw_yaw_offsets
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return [int(x) for x in draw_yaw_offsets(int(n), int(img_w), gen).tolist()]


_GOLDEN_CACHE = {}


def golden_offsets(seed, n, img_w=V.IMG_W):
    """:func:`_golden_offsets_impl`, memoised — 50 rgen cells share 5 sequences."""
    key = (int(seed), int(n), int(img_w))
    if key not in _GOLDEN_CACHE:
        _GOLDEN_CACHE[key] = _golden_offsets_impl(*key)
    return list(_GOLDEN_CACHE[key])


# --------------------------------------------------------------------------- #
# statistics (plan §4): 5 seed-paired observations, df = 4, two-sided 95%
# --------------------------------------------------------------------------- #
# Pre-registered in the plan, so it is a CONSTANT here and not a library call:
# the campaign's interval must not depend on which numerical stack is installed
# on the machine that renders the table.
T_CRITICAL = {(0.05, 4): 2.7764451051977987}

try:                                                  # optional, cross-checked by test
    from scipy import stats as _scipy_stats
    STATS_BACKEND = "scipy"
except Exception:                                     # pragma: no cover - env-dependent
    _scipy_stats = None
    STATS_BACKEND = "pure-python (regularised incomplete beta)"


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    maxit, eps, fpmin = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_sf(t, df):
    """``P(T > t)`` for Student's t with ``df`` degrees of freedom, no scipy."""
    df = float(df)
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + float(t) ** 2))
    return tail if t >= 0 else 1.0 - tail


def _student_t_ppf(q, df):
    """Inverse of :func:`_student_t_sf`'s CDF, by bisection (fallback path)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    lo, hi = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - _student_t_sf(mid, df)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_critical(alpha, df):
    """Two-sided critical value; the campaign's (0.05, 4) is the plan's constant."""
    key = (round(float(alpha), 10), int(df))
    if key in T_CRITICAL:
        return T_CRITICAL[key]
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df))
    return _student_t_ppf(1.0 - alpha / 2.0, df)       # pragma: no cover - env-dependent


def _t_sf(t, df):
    if _scipy_stats is not None:
        return float(_scipy_stats.t.sf(t, df))
    return _student_t_sf(t, df)                        # pragma: no cover - env-dependent


TResult = namedtuple("TResult", "n df mean sd se t p lo hi alpha")


def paired_t_ci(diffs, alpha=0.05):
    """Two-sided paired-t estimate over the per-seed differences (plan §4).

    Zero spread is not an error and not a licence: five identical non-zero
    differences are a real effect with p → 0, five identical zeros are no effect
    with p = 1. Reporting either as "undefined" would silently drop a co-primary
    from the Holm family.
    """
    values = [float(d) for d in diffs]
    n = len(values)
    if n < 2:
        raise ValueError(f"a paired-t estimate needs at least two observations, got {n}")
    df = n - 1
    mean = st.mean(values)
    sd = st.stdev(values)
    se = sd / math.sqrt(n)
    crit = t_critical(alpha, df)
    if se == 0.0:
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p = 1.0 if mean == 0.0 else 0.0
        return TResult(n, df, mean, sd, se, t, p, mean, mean, alpha)
    t = mean / se
    p = 2.0 * _t_sf(abs(t), df)
    return TResult(n, df, mean, sd, se, t, min(p, 1.0),
                   mean - crit * se, mean + crit * se, alpha)


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, in the input order (ties included)."""
    values = [float(p) for p in pvals]
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# Plan §4 metric directions. "Lower is better" for the error metrics, "higher is
# better" for retrieval. There is no default: a metric with no registered
# direction cannot be scored "better" at all, so metric_direction RAISES.
METRIC_DIRECTION = {
    "T60": "lower", "C50": "lower", "EDT": "lower", "FD": "lower",
    "RIR_to_GT_RIR_R@1": "higher", "RIR_to_GT_RIR_R@5": "higher",
    "RIR_to_GT_RIR_R@10": "higher",
    "RIR_to_geom_R@1": "higher", "RIR_to_geom_R@5": "higher",
    "RIR_to_geom_R@10": "higher",
}
# The plan writes these as T60% and R@k; the metrics JSON spells them T60 and
# RIR_to_GT_RIR_R@k. Both names reach the same direction, one table.
METRIC_ALIAS = {"T60%": "T60", "T60 (%)": "T60", "R@1": "RIR_to_GT_RIR_R@1",
                "R@5": "RIR_to_GT_RIR_R@5", "R@10": "RIR_to_GT_RIR_R@10",
                "C50 (dB)": "C50", "EDT (ms)": "EDT"}
# Co-primaries (plan §4, restored by the round-1 review's B3): the reported R@1
# is RIR_to_GT_RIR_R@1, audio-to-audio. Only RIR_to_geom_R@k embeds the ROTATED
# point cloud through a non-yaw-invariant AGREE, so only those are confounded in
# a rotated cell — reported, but never in a headline table.
CO_PRIMARY = ("T60", "RIR_to_GT_RIR_R@1")
# G5 is a REPRODUCTION check against exp_11's committed rows, and those rows only
# ever carried the flat split-level numbers. So G5 compares FLAT to FLAT — the
# scene-mean is this campaign's estimand, not a quantity exp_11 ever measured
# (round-3 closure B6).
G5_METRICS = CO_PRIMARY
CONFOUNDED_METRICS = ("RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
HEADLINE_METRICS = ("T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1",
                    "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10")
ALPHA = 0.05


def canonical_metric(metric):
    return METRIC_ALIAS.get(metric, metric)


def metric_direction(metric):
    """``'lower'`` or ``'higher'`` — the complete plan §4 table, no default."""
    name = canonical_metric(metric)
    if name not in METRIC_DIRECTION:
        raise KeyError(f"no registered direction for metric {metric!r}: it cannot be "
                       "scored better-or-worse")
    return METRIC_DIRECTION[name]


Contrast = namedtuple("Contrast",
                      "metric better n df mean lo hi t p favors_first significant")


def contrast(metric, diffs, better="metric", alpha=ALPHA):
    """One paired contrast ``first − second`` on one metric.

    ``better`` is explicit because the answer is not always the metric's own
    direction: an H-M contrast compares |Δ| — a MAGNITUDE of change — where
    smaller is flatter no matter which way the underlying metric points.
    """
    if better == "metric":
        better = metric_direction(metric)
    if better not in ("lower", "higher"):
        raise ValueError(f"better must be 'lower', 'higher' or 'metric', got {better!r}")
    res = paired_t_ci(diffs, alpha=alpha)
    favors = res.mean < 0 if better == "lower" else res.mean > 0
    return Contrast(metric=canonical_metric(metric), better=better, n=res.n, df=res.df,
                    mean=res.mean, lo=res.lo, hi=res.hi, t=res.t, p=res.p,
                    favors_first=favors, significant=res.p < alpha)


def verdict(flags):
    """Plan §4: both co-primaries → SUPPORTED, exactly one → PARTIAL, else NEGATIVE."""
    wins = sum(1 for f in flags if f)
    if wins == len(flags) and wins > 0:
        return "SUPPORTED"
    return "PARTIAL" if wins else "NEGATIVE"


def endpoint_contrast(name, first, second, seeds, better="metric",
                      metrics=CO_PRIMARY, alpha=ALPHA, win="favor"):
    """A labelled hypothesis: co-primary contrasts + Holm over exactly those two.

    ``first``/``second`` are ``{metric: {seed: value}}``. ``win='favor'`` scores a
    co-primary as won when it is significant AND points the hypothesised way
    (H-P, H-M); ``win='nonzero'`` scores any significant difference (H-S, whose
    claim is ``Δ ≠ 0``).
    """
    seeds = [int(s) for s in seeds]
    rows, contrasts = {}, []
    for metric in metrics:
        diffs = [float(first[metric][s]) - float(second[metric][s]) for s in seeds]
        contrasts.append(contrast(metric, diffs, better=better, alpha=alpha))
    for c, p_holm in zip(contrasts, holm_adjust([c.p for c in contrasts])):
        won = (p_holm < alpha) and (c.favors_first or win == "nonzero")
        rows[c.metric] = {"mean": c.mean, "lo": c.lo, "hi": c.hi, "t": c.t, "p": c.p,
                          "p_holm": p_holm, "df": c.df, "n": c.n, "better": c.better,
                          "favors_first": c.favors_first, "won": won}
    return {"name": name, "seeds": seeds, "alpha": alpha, "win_rule": win,
            "metrics": rows, "verdict": verdict([r["won"] for r in rows.values()])}


# --------------------------------------------------------------------------- #
# discovery: which registered cells actually landed
# --------------------------------------------------------------------------- #
class CellStore:
    """Every registered cell, sorted into proven / absent / refused."""

    def __init__(self, output_root, campaign, cells, missing, rejected):
        self.output_root = output_root
        self.campaign = campaign
        self.cells = list(cells)
        self.missing = list(missing)
        self.rejected = list(rejected)
        self.index = _by_cell(self.cells)

    def get(self, arm, celltype, k, seed):
        return self.index.get((arm, celltype, int(k), int(seed)))

    def block(self, arm, celltype, k):
        """The per-seed records of one (arm, cell type, K), seed-ordered."""
        return [self.index[(arm, celltype, int(k), s)] for s in SEEDS
                if (arm, celltype, int(k), s) in self.index]

    def vctl(self, arm, deg):
        for c in self.cells:
            if (c.cell.cell == "vctl" and c.cell.arm == arm
                    and float(c.cell.rotate_deg) == float(deg)):
                return c
        return None


def collect_cells(output_root, campaign, grid=None, step=STEP):
    """Read every registered cell that landed; refuse the rest, by name."""
    cells, missing, rejected = [], [], []
    ckpts = {}
    for cell in (grid or expected_grid()):
        if cell.arm not in ckpts:
            ckpts[cell.arm] = V.checkpoint_path(output_root, cell.arm, step)
        ckpt = ckpts[cell.arm]
        if ckpt is None:
            missing.append({"cell": V.eval_name(cell), "path": None,
                            "reason": f"no unique step={step} checkpoint under "
                                      f"{output_root}/exp11_{cell.arm}"})
            continue
        path = V.metrics_path(ckpt, cell)
        if not os.path.isfile(path):
            missing.append({"cell": V.eval_name(cell), "path": path,
                            "reason": "not run yet"})
            continue
        data, reasons = load_cell(path, campaign)
        if reasons:
            rejected.append({"cell": V.eval_name(cell), "path": path, "reasons": reasons})
        else:
            cells.append(data)
    return CellStore(output_root, campaign, cells, missing, rejected)


# --------------------------------------------------------------------------- #
# gates G1-G4 (executable, blocking) and G5 (a check, never a gate)
# --------------------------------------------------------------------------- #
GATE_NAMES = ("G1", "G2", "G3", "G4")
CN_ARMS = tuple(a for a in ARM_ORDER if a != "VANL")


def _z_std(store, arm, metric, k=8):
    """σ̂ over the arm's five unrotated seeds at K — the gates' own yardstick."""
    agg = aggregate_cell(store.block(arm, "zref", k))
    if agg.status != "OK":
        return None, agg
    return agg.values[metric][1], agg


def gate_g1(store, k=8, seed=42, tolerance=0.5):
    """In-group floor: rotating a Cn arm by its OWN group angle must not move a
    co-primary by more than half that arm's own five-seed spread (plan §4 G1)."""
    failures, details, pending = [], [], []
    for arm in CN_ARMS:
        control = store.vctl(arm, 90.0)
        reference = store.get(arm, "zref", k, seed)
        if control is None or reference is None:
            pending.append(f"{arm}: missing "
                           + ("the 90° validity cell" if control is None
                              else f"the unrotated seed-{seed} reference"))
            continue
        for metric in CO_PRIMARY:
            sigma, agg = _z_std(store, arm, metric, k)
            if sigma is None:
                pending.append(f"{arm}: the K={k} unrotated block is {agg.status} "
                               f"({agg.n}/5 seeds) — no σ̂, so no tolerance")
                break
            diff = abs(float(control.metrics[metric]) - float(reference.metrics[metric]))
            bound = tolerance * sigma
            row = {"arm": arm, "metric": metric, "diff": diff, "sigma": sigma,
                   "bound": bound, "passed": diff <= bound}
            details.append(row)
            if not row["passed"]:
                failures.append(f"{arm} {metric}: |Δ|={diff:.4f} > {tolerance}·σ̂="
                                f"{bound:.4f} (σ̂={sigma:.4f})")
    status = "PENDING" if pending else ("FAIL" if failures else "PASS")
    return {"name": "G1", "status": status, "failures": failures, "pending": pending,
            "details": details,
            "definition": "|m(V@90°,s42,K8) − m(Z,s42,K8)| ≤ 0.5·σ̂(arm's 5 Z seeds), "
                          "each co-primary read through its ruled source "
                          "(T60 scene-mean, R@1 split-level)"}


def gate_g2(store, k=8, seed=42, factor=5.0, metric="T60"):
    """Positive control: VANL@90° must degrade far past VANL's own seed noise."""
    control, reference = store.vctl("VANL", 90.0), store.get("VANL", "zref", k, seed)
    if control is None or reference is None:
        return {"name": "G2", "status": "PENDING",
                "pending": ["VANL 90° validity cell or unrotated reference missing"],
                "failures": [], "details": [],
                "definition": "m_T60(VANL V@90°) − m_T60(VANL Z) ≥ 5·σ̂_T60(VANL), T60 read as the SCENE-MEAN (ruled source)"}
    sigma, agg = _z_std(store, "VANL", metric, k)
    if sigma is None:
        return {"name": "G2", "status": "PENDING",
                "pending": [f"VANL K={k} unrotated block is {agg.status} ({agg.n}/5)"],
                "failures": [], "details": [],
                "definition": "m_T60(VANL V@90°) − m_T60(VANL Z) ≥ 5·σ̂_T60(VANL), T60 read as the SCENE-MEAN (ruled source)"}
    delta = float(control.metrics[metric]) - float(reference.metrics[metric])
    bound = factor * sigma
    passed = delta >= bound
    return {"name": "G2", "status": "PASS" if passed else "FAIL",
            "failures": ([] if passed else
                         [f"VANL {metric}: degradation {delta:.4f} < {factor}·σ̂="
                          f"{bound:.4f} — the harness is not detecting non-invariance"]),
            "pending": [],
            "details": [{"arm": "VANL", "metric": metric, "delta": delta,
                         "sigma": sigma, "bound": bound, "passed": passed}],
            "definition": "m_T60(VANL V@90°) − m_T60(VANL Z) ≥ 5·σ̂_T60(VANL), T60 read as the SCENE-MEAN (ruled source)"}


def gate_g3(store):
    """Golden assignment: every rotated cell's offsets are RECOMPUTED here from
    its rotation seed, so the gate proves the registered draw reached
    ``rotate_scene_metadata`` — not that a stored sequence matches itself."""
    failures, details = [], []
    rgen = [c for c in store.cells if c.cell.cell == "rgen"]
    if not rgen:
        return {"name": "G3", "status": "PENDING", "failures": [],
                "pending": ["no rotated (rgen) cell has landed yet"], "details": [],
                "definition": "cell offsets == draw_yaw_offsets(n, 512, gen(rotate_seed))"}
    for c in sorted(rgen, key=lambda c: (c.cell.arm, c.cell.k, c.cell.seed)):
        want = golden_offsets(int(c.cell.seed), len(c.offsets), V.IMG_W)
        ok = list(c.offsets) == want
        details.append({"cell": V.eval_name(c.cell), "n": len(c.offsets), "passed": ok})
        if not ok:
            first = next((i for i, (a, b) in enumerate(zip(c.offsets, want)) if a != b),
                         min(len(c.offsets), len(want)))
            failures.append(f"{V.eval_name(c.cell)}: offsets differ from the seed-"
                            f"{c.cell.seed} draw at position {first}")
    return {"name": "G3", "status": "FAIL" if failures else "PASS", "failures": failures,
            "pending": [], "details": details,
            "definition": "cell offsets == draw_yaw_offsets(n, 512, gen(rotate_seed))"}


def _assignment_comparisons(cells):
    """How many §3.3 equalities there actually were to check.

    Zero is the reason G4 cannot report PASS on an empty campaign: "nothing
    disagreed" is not evidence when nothing was compared.
    """
    groups, pairs = {}, 0
    keys = set()
    for c in cells:
        key = (c.cell.cell, int(c.cell.k), int(c.cell.seed))
        groups[key] = groups.get(key, 0) + 1
        keys.add((c.cell.arm, c.cell.cell, int(c.cell.k), int(c.cell.seed)))
    for arm, celltype, k, seed in keys:
        if celltype == "zref" and (arm, "rgen", k, seed) in keys:
            pairs += 1
    return sum(n - 1 for n in groups.values() if n > 1) + pairs


def gate_g4(store):
    """Assignment integrity: every plan §3.3 hash equality (see match_assignments)."""
    violations = match_assignments(store.cells)
    blocked = sorted({tuple(b) for v in violations for b in v.blocks})
    comparisons = _assignment_comparisons(store.cells)
    status = "FAIL" if violations else ("PASS" if comparisons else "PENDING")
    return {"name": "G4", "status": status, "comparisons": comparisons,
            "failures": [v.detail for v in violations],
            "pending": ([] if comparisons else
                        ["no cell pair or cross-arm group has landed yet: there is no "
                         "hash equality to check, and 'nothing disagreed' is not "
                         "evidence when nothing was compared"]),
            "violations": [{"kind": v.kind, "scope": v.scope, "detail": v.detail,
                            "blocks": [list(b) for b in v.blocks]} for v in violations],
            "blocked_scopes": [list(b) for b in blocked],
            "definition": "cross-arm input/assignment hashes equal within (K, seed); "
                          "Z.input_hash == R.input_hash within (arm, K, seed)"}


def exp11_conf_files(exp11_root, arm, k, step=STEP):
    """exp_11's committed θ=0 conf rows for one (arm, K) — read-only, cross-pin."""
    import glob as _glob
    orbit = V.TRAIN_ORBIT[arm]
    suffix = "" if orbit == 0 else f"_fa_invariant_a{orbit}"
    pattern = os.path.join(
        exp11_root, f"exp11_{arm}", f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints",
        f"*step={step}_metrics_1_1.0_exp11_{arm}_conf_S{step}_s4[2-6]_K{k}{suffix}.json")
    return sorted(_glob.glob(pattern))


def gate_g5(store, exp11_root=None, k=8, metrics=CO_PRIMARY, factor=3.0):
    """External reproduction — a CHECK, never a gate (plan §4 G5).

    exp_14 re-measures θ=0 at ITS OWN pin; exp_11's committed conf rows were
    measured at another one. A discrepancy is worth disclosing, and is worth
    nothing as a halt: the two are not the same measurement.
    """
    root = exp11_root or store.output_root
    rows, notes = [], []
    for arm in CN_ARMS:
        files = exp11_conf_files(root, arm, k)
        if len(files) < len(SEEDS):
            notes.append(f"{arm}: {len(files)}/5 exp_11 conf rows found under {root}")
            continue
        block = store.block(arm, "zref", k)
        ours = aggregate_cell(block)
        if ours.status != "OK":
            notes.append(f"{arm}: exp_14 Z block at K={k} is {ours.status}")
            continue
        # FLAT to FLAT (round-3 closure B6). exp_11's committed rows carry only
        # the split-level numbers, so comparing them against exp_14's scene-mean
        # would report the AGGREGATION as a reproduction discrepancy — or hide a
        # real one behind it.
        flat_seed = {m: [c.flat_metrics[m] for c in block if m in c.flat_metrics]
                     for m in metrics}
        for metric in metrics:
            theirs = []
            for path in files:
                try:
                    with open(path) as fh:
                        theirs.append(float(json.load(fh)["metrics"][metric]))
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    notes.append(f"{arm}: unreadable exp_11 row {path} ({exc})")
                    theirs = []
                    break
            if len(theirs) != len(SEEDS):
                continue
            m11, s11 = st.mean(theirs), st.stdev(theirs)
            mine = flat_seed.get(metric) or []
            if len(mine) != len(SEEDS):
                notes.append(f"{arm}: exp_14 K={k} has {len(mine)}/{len(SEEDS)} "
                             f"split-level {metric} values, so the flat-to-flat "
                             "reproduction check cannot be made")
                continue
            m14, s14 = st.mean(mine), st.stdev(mine)
            bound = factor * math.sqrt(s11 ** 2 + s14 ** 2) / math.sqrt(len(SEEDS))
            rows.append({"arm": arm, "metric": metric, "exp11_mean": m11,
                         "exp11_std": s11, "exp14_mean": m14, "exp14_std": s14,
                         "diff": m14 - m11, "bound": bound,
                         "source": "split-level (both sides)",
                         "beyond": abs(m14 - m11) > bound})
    return {"name": "G5", "status": "CHECK" if rows else "UNAVAILABLE", "rows": rows,
            "notes": notes, "gates": False,
            "definition": "exp_14 Z vs exp_11 conf @40k, SPLIT-LEVEL on both sides "
                          "(exp_11 never measured a scene-mean); disclose "
                          "|Δ| > 3·√(σ11²+σ14²)/√5 (cross-pin: reported, never a halt)"}


def evaluate_gates(store, exp11_root=None):
    """G1–G4 (blocking) plus the G5 check, with the scopes G4 blocks."""
    report = {"gate_names": GATE_NAMES}
    report["G1"] = gate_g1(store)
    report["G2"] = gate_g2(store)
    report["G3"] = gate_g3(store)
    report["G4"] = gate_g4(store)
    report["G5"] = gate_g5(store, exp11_root=exp11_root)
    # "Not evaluated" may never read as "passed": a PENDING gate suppresses the
    # H-readouts exactly like a failing one, and says which it was.
    report["all_passed"] = all(report[g]["status"] == "PASS" for g in GATE_NAMES)
    report["blocked_scopes"] = report["G4"]["blocked_scopes"]
    report["summary"] = ", ".join(f"{g}={report[g]['status']}" for g in GATE_NAMES)
    return report


# --------------------------------------------------------------------------- #
# results: what the campaign says, and what it refuses to say
# --------------------------------------------------------------------------- #
# The per-seed observation, stated in the output itself, PER METRIC (the
# Planner's 2026-08-11 ruling, quoted verbatim above AGGREGATION_SOURCE and
# pre-registered before any cell ran).
AGGREGATION = {
    "ruling": ("per-scene mean applies to the ACOUSTIC-PARAMETER family only — T60 "
               "(incl. Invalid-T60 handling), C50, EDT — matching the paper convention "
               "plan §4 intended. RETRIEVAL (RIR_to_GT_RIR_R@k, and the quarantined "
               "RIR_to_geom_R@k) and FD use the SPLIT-LEVEL global metrics: within-scene "
               "retrieval among ~370 items is a different, easier task whose levels are "
               "incomparable to every previously published number in this program, and "
               "exp_01's noise-floor calibration against released Table-1 was on the "
               "global quantity; one-room Frechet is additionally small-sample biased"),
    "co_primary": ("T60% (per-scene mean) + RIR_to_GT_RIR_R@1 (split-level)"),
    "scene_mean": list(ACOUSTIC_METRICS),
    "split_level": list(SPLIT_LEVEL_METRICS),
    "note": ("both sources exist in every exp_14 record — the flat `metrics` block and "
             "the `by_scene` block — so this is a reading rule, not a measurement "
             "change; by_scene stays REQUIRED for every cell because the acoustic "
             "family is read from it"),
}
RESULTS_SCHEMA_VERSION = 1
ADJACENT_PAIRS = tuple(zip(ARM_ORDER[1:], ARM_ORDER[:-1]))   # (later, earlier)


def suppress_validity_cells(rows):
    """The V block is QA (plan §3.1) — it never appears in a headline table."""
    out = []
    for row in rows:
        celltype = (row.get("cell_type") if isinstance(row, dict)
                    else getattr(row.cell, "cell", None))
        if celltype != "vctl":
            out.append(row)
    return out


def _block(agg, arm, k, blocked_reason=None):
    row = {"arm": arm, "K": int(k), "status": agg.status, "n": agg.n,
           "seeds": list(agg.seeds), "reasons": list(agg.reasons),
           "values": {m: [v[0], v[1]] for m, v in agg.values.items()},
           "per_seed": {m: {str(s): v for s, v in d.items()}
                        for m, d in agg.per_seed.items()}}
    if blocked_reason:
        row.update({"status": "BLOCKED", "values": {}, "per_seed": {},
                    "reasons": [blocked_reason]})
    return row


def _pending_note(agg):
    return f"{agg.n}/{len(SEEDS)} seeds"


def build_results(store, gates=None, generated_at=None, exp11_root=None, alpha=ALPHA):
    """Everything the campaign supports, and a named reason for everything it does not."""
    import datetime
    gates = gates or evaluate_gates(store, exp11_root=exp11_root)
    blocked = {tuple(b) for b in gates["blocked_scopes"]}
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "campaign": {"pin": store.campaign.pin, "output_root": store.output_root,
                     "expected_count": store.campaign.expected_count, "step": STEP,
                     "alpha": alpha, "stats_backend": STATS_BACKEND,
                     "arm_order": list(ARM_ORDER), "seeds": list(SEEDS),
                     "co_primary": list(CO_PRIMARY)},
        "aggregation": dict(AGGREGATION),
        "gates": gates,
    }
    counts = {"random-yaw (R)": 0, "unrotated (Z)": 0, "validity (V)": 0}
    label = {"rgen": "random-yaw (R)", "zref": "unrotated (Z)", "vctl": "validity (V)"}
    for c in store.cells:
        counts[label[c.cell.cell]] += 1
    results["inventory"] = {
        "registered": len(expected_grid()), "valid": len(store.cells), "counts": counts,
        "missing": store.missing, "rejected": store.rejected}

    # --- absolute blocks, per arm and K -------------------------------------
    aggs = {}
    for kind, celltype in (("R", "rgen"), ("Z", "zref")):
        results.setdefault("blocks", {})[kind] = {}
        for arm in ARM_ORDER:
            results["blocks"][kind][arm] = {}
            for k in KS:
                agg = aggregate_cell(store.block(arm, celltype, k))
                aggs[(kind, arm, k)] = agg
                results["blocks"][kind][arm][str(k)] = _block(agg, arm, k)

    # --- paired degradation, per arm and K ----------------------------------
    results["paired"] = {}
    deltas = {}
    for arm in ARM_ORDER:
        results["paired"][arm] = {}
        for k in KS:
            z, r = aggs[("Z", arm, k)], aggs[("R", arm, k)]
            pairs, problems = pair_seeds(store.block(arm, "zref", k),
                                         store.block(arm, "rgen", k))
            entry = {"arm": arm, "K": int(k), "metrics": {}, "abs_metrics": {},
                     "pairs": len(pairs), "reasons": list(problems)}
            if ("paired", arm, int(k)) in blocked:
                entry["status"] = "BLOCKED"
                entry["reasons"].append("Z and R are not a matched pair at this "
                                        "(arm, K) — see gate G4")
            elif z.status != "OK" or r.status != "OK" or len(pairs) != len(SEEDS):
                entry["status"] = "PENDING"
                entry["reasons"] += [f"Z block {z.status} ({z.n}/5)",
                                     f"R block {r.status} ({r.n}/5)"]
            else:
                entry["status"] = "OK"
                for metric in HEADLINE_METRICS + CONFOUNDED_METRICS:
                    if metric not in z.values or metric not in r.values:
                        continue
                    per_seed = {s: r.per_seed[metric][s] - z.per_seed[metric][s]
                                for s in SEEDS}
                    res = paired_t_ci(list(per_seed.values()), alpha=alpha)
                    entry["metrics"][metric] = {
                        "mean": res.mean, "lo": res.lo, "hi": res.hi, "t": res.t,
                        "p": res.p, "df": res.df,
                        "per_seed": {str(s): v for s, v in per_seed.items()}}
                    entry["abs_metrics"][metric] = {
                        "per_seed": {str(s): abs(v) for s, v in per_seed.items()},
                        "mean": st.mean(abs(v) for v in per_seed.values())}
                    deltas[(arm, int(k), metric)] = per_seed
            results["paired"][arm][str(k)] = entry

    # --- the labelled hypotheses (plan §4) ----------------------------------
    results["hypotheses"] = _hypotheses(results, aggs, deltas, gates, blocked, alpha, k=8)
    results["hypotheses_K1"] = _hypotheses(results, aggs, deltas, gates, blocked, alpha,
                                           k=1, descriptive=True)
    results["adjacent"] = _adjacent(aggs, deltas, blocked, alpha)
    results["geometry_retrieval"] = {
        "disclosure": ("rotated-gallery retrieval: in an R cell the gallery embeds the "
                       "ROTATED point cloud through a non-yaw-invariant AGREE, so these "
                       "numbers mix model robustness with AGREE's own yaw sensitivity. "
                       "Cross-arm comparisons stay internally valid (the galleries are "
                       "rotation-matched) but the level is confounded — descriptive only, "
                       "never a headline or a co-primary."),
        "metrics": list(CONFOUNDED_METRICS),
        "blocks": {kind: {arm: {str(k): {
            "status": aggs[(kind, arm, k)].status,
            "values": {m: list(aggs[(kind, arm, k)].values[m])
                       for m in CONFOUNDED_METRICS if m in aggs[(kind, arm, k)].values}}
            for k in KS} for arm in ARM_ORDER} for kind in ("R", "Z")},
    }
    results["validity_cells"] = _validity_cells(store)
    return results


def _hypotheses(results, aggs, deltas, gates, blocked, alpha, k=8, descriptive=False):
    """H-P / H-M / H-S at one K. Numbers exist only when the gates let them."""
    out = {"K": int(k), "descriptive": bool(descriptive),
           "suppressed": not gates["all_passed"],
           "suppression_reason": ("" if gates["all_passed"] else
                                  f"gates did not pass ({gates['summary']}); plan §4: "
                                  "H-readouts render only when G1–G4 pass")}

    def _status(needs_cross_arm, arms):
        if needs_cross_arm and ("cross_arm", int(k)) in blocked:
            return "BLOCKED", ("arms are not rotation-matched at K="
                               f"{k} — see gate G4")
        for arm in arms:
            if ("paired", arm, int(k)) in blocked:
                return "BLOCKED", f"{arm}: Z and R are not a matched pair at K={k}"
        for arm in arms:
            if aggs[("R", arm, k)].status != "OK":
                return "PENDING", (f"{arm} R block at K={k} is "
                                   f"{aggs[('R', arm, k)].status} "
                                   f"({aggs[('R', arm, k)].n}/5 seeds)")
            if aggs[("Z", arm, k)].status != "OK":
                return "PENDING", (f"{arm} Z block at K={k} is "
                                   f"{aggs[('Z', arm, k)].status} "
                                   f"({aggs[('Z', arm, k)].n}/5 seeds)")
        if not gates["all_passed"]:
            return "SUPPRESSED", out["suppression_reason"]
        return "OK", ""

    def _per_seed(kind, arm):
        return {m: {s: aggs[(kind, arm, k)].per_seed[m][s] for s in SEEDS}
                for m in CO_PRIMARY if m in aggs[(kind, arm, k)].per_seed}

    def _abs_delta(arm):
        return {m: {s: abs(deltas[(arm, int(k), m)][s]) for s in SEEDS}
                for m in CO_PRIMARY if (arm, int(k), m) in deltas}

    def _delta(arm):
        return {m: dict(deltas[(arm, int(k), m)]) for m in CO_PRIMARY
                if (arm, int(k), m) in deltas}

    # H-P (PRIMARY): absolute robustness, C32 vs VANL
    label = "H-P (PRIMARY): m_R(C32) vs m_R(VANL)"
    status, why = _status(True, ("C32", "VANL"))
    out["H-P"] = {"name": label, "status": status, "reason": why, "K": int(k)}
    if status == "OK":
        out["H-P"].update(endpoint_contrast(
            label, _per_seed("R", "C32"), _per_seed("R", "VANL"), SEEDS,
            better="metric", alpha=alpha))
    # H-M (mechanism): flatness, |Δ|(C32) vs |Δ|(C4L)
    label = "H-M (mechanism): |Δ|(C32) vs |Δ|(C4L)"
    status, why = _status(True, ("C32", "C4L"))
    out["H-M"] = {"name": label, "status": status, "reason": why, "K": int(k)}
    if status == "OK":
        out["H-M"].update(endpoint_contrast(
            label, _abs_delta("C32"), _abs_delta("C4L"), SEEDS, better="lower",
            alpha=alpha))
        side, why_side = _status(True, ("VANL", "C4L"))
        if side == "OK":
            out["H-M"]["alongside"] = endpoint_contrast(
                "|Δ|(VANL) vs |Δ|(C4L)", _abs_delta("VANL"), _abs_delta("C4L"), SEEDS,
                better="lower", alpha=alpha)
    # H-S (sanity): vanilla is not yaw-robust
    label = "H-S (sanity): Δ(VANL) ≠ 0"
    status, why = _status(False, ("VANL",))
    out["H-S"] = {"name": label, "status": status, "reason": why, "K": int(k)}
    if status == "OK":
        zero = {m: {s: 0.0 for s in SEEDS} for m in CO_PRIMARY}
        out["H-S"].update(endpoint_contrast(label, _delta("VANL"), zero, SEEDS,
                                            better="metric", alpha=alpha, win="nonzero"))
    return out


def _adjacent(aggs, deltas, blocked, alpha):
    """Adjacent contrasts in the FIXED plan order — descriptive, never a verdict."""
    out = {"order": list(ARM_ORDER),
           "note": ("fixed-order adjacent contrasts on the plan's arm order, unadjusted "
                    "and descriptive: no verdict attaches to them and the observed "
                    "ranking is never turned into a confirmatory test"),
           "absolute": [], "abs_delta": []}
    for k in KS:
        for later, earlier in ADJACENT_PAIRS:
            for metric in CO_PRIMARY:
                row = {"pair": f"{earlier}→{later}", "K": int(k), "metric": metric}
                if ("cross_arm", int(k)) in blocked:
                    row["status"] = "BLOCKED"
                    row["reason"] = f"arms are not rotation-matched at K={k} (gate G4)"
                    out["absolute"].append(dict(row))
                    out["abs_delta"].append(dict(row))
                    continue
                a, b = aggs[("R", later, k)], aggs[("R", earlier, k)]
                if a.status != "OK" or b.status != "OK" or metric not in a.per_seed:
                    row["status"] = "PENDING"
                    row["reason"] = f"{a.n}/5 and {b.n}/5 seeds on disk"
                    out["absolute"].append(dict(row))
                    out["abs_delta"].append(dict(row))
                    continue
                c = contrast(metric, [a.per_seed[metric][s] - b.per_seed[metric][s]
                                      for s in SEEDS], better="metric", alpha=alpha)
                out["absolute"].append(dict(row, status="OK", mean=c.mean, lo=c.lo,
                                            hi=c.hi, p=c.p))
                if (later, int(k), metric) in deltas and (earlier, int(k), metric) in deltas:
                    d = contrast(metric,
                                 [abs(deltas[(later, int(k), metric)][s])
                                  - abs(deltas[(earlier, int(k), metric)][s])
                                  for s in SEEDS], better="lower", alpha=alpha)
                    out["abs_delta"].append(dict(row, status="OK", mean=d.mean, lo=d.lo,
                                                 hi=d.hi, p=d.p))
                else:
                    out["abs_delta"].append(dict(row, status="PENDING",
                                                 reason="paired Δ unavailable"))
    return out


def _validity_cells(store):
    """The six V cells, with their own unrotated reference — QA, never headline."""
    rows = []
    for arm, deg in V.VCTL_TUPLES:
        cell = store.vctl(arm, deg)
        reference = store.get(arm, "zref", 8, 42)
        row = {"arm": arm, "rotate_deg": float(deg), "role": (
            "positive control (exp_02 prior: vanilla degrades)" if arm == "VANL"
            else "in-group floor (G1)" if float(deg) == 90.0
            else "off-group mechanism control — NO gate role")}
        if cell is None:
            row["status"] = "MISSING"
        else:
            row["status"] = "OK"
            row["metrics"] = {m: cell.metrics.get(m) for m in CO_PRIMARY}
            if reference is not None:
                row["vs_unrotated"] = {m: cell.metrics.get(m, 0.0)
                                       - reference.metrics.get(m, 0.0)
                                       for m in CO_PRIMARY}
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _arrow(metric):
    return "↓" if metric_direction(metric) == "lower" else "↑"


def _metric_label(metric):
    """``T60 (scene-mean)`` / ``RIR_to_GT_RIR_R@1 (split)``.

    The source is part of the metric's name in every table: two readers comparing
    a T60 and an R@1 from this report are comparing quantities aggregated
    differently, on purpose, and the column has to say so.
    """
    return f"{canonical_metric(metric)} ({aggregation_source(metric)})"


def _num(value, digits=3):
    return f"{value:.{digits}f}"


def _md_row(cells):
    """One markdown row. An EMPTY cell renders as ``| |`` — the columns a refusal
    leaves unfilled must look unfilled, not like a cell someone forgot to read.

    Pipes inside a cell are escaped: G1's definition is literally
    ``|m(V@90°) − m(Z)| ≤ 0.5·σ̂``, and unescaped those four bars would split the
    row into extra columns and silently mangle the gate table.
    """
    out = []
    for c in cells:
        text = str(c).replace("|", "\\|")
        out.append(f" {text} " if text != "" else " ")
    return "|" + "|".join(out) + "|"


def render_block_table(rows, metrics):
    """One markdown table of per-arm blocks. PENDING and BLOCKED occupy the first
    metric column and leave the rest empty — there is no cell in which a number
    could be mistaken for a measured one."""
    head = ["arm", "K", "n"] + [f"{_metric_label(m)} {_arrow(m)}" for m in metrics]
    out = [_md_row(head), "|" + "---|" * len(head)]
    for row in rows:
        arm, k, status = row["arm"], row["K"], row["status"]
        if status == "OK":
            cells = [_num(row["values"][m][0]) + " ± " + _num(row["values"][m][1])
                     if m in row["values"] else "" for m in metrics]
            out.append(_md_row([arm, k, row["n"]] + cells))
            continue
        reason = (row.get("reasons") or [""])[0]
        if status == "PENDING":
            note = f"*PENDING ({row['n']}/{len(SEEDS)} {row.get('unit', 'seeds')})*"
            count = "—"
        else:
            note = f"**BLOCKED — {reason}**"
            count = str(row["n"])
        out.append(_md_row([arm, k, count, note] + [""] * (len(metrics) - 1)))
    return "\n".join(out) + "\n"


def render_gate_report(gates):
    """The gate table plus one line per failure or pending reason."""
    out = [_md_row(["gate", "status", "definition"]), "|---|---|---|"]
    notes = []
    for name in gates["gate_names"]:
        g = gates[name]
        out.append(_md_row([name, g["status"], g.get("definition", "")]))
        for line in list(g.get("failures") or []) + list(g.get("pending") or []):
            notes.append(f"**{name} {g['status']}** — {line}")
    if notes:
        out += [""] + notes
    return "\n".join(out) + "\n"


def _rows_for(results, kind, k):
    return [results["blocks"][kind][arm][str(k)] for arm in ARM_ORDER]


def _delta_rows(results, k):
    rows = []
    for arm in ARM_ORDER:
        entry = results["paired"][arm][str(k)]
        complete = entry["status"] == "OK"
        # the count is the number of MATCHED PAIRS (review N6): a block with four
        # pairs and a missing fifth seed reported 0/5, because the count used to
        # come from the metrics dict that PENDING deliberately leaves empty
        row = {"arm": arm, "K": int(k), "status": entry["status"],
               "n": len(SEEDS) if complete else entry.get("pairs", 0),
               "unit": "pairs", "reasons": entry.get("reasons", []), "values": {}}
        if complete:
            # ± half the 95% CI width, so the Δ table's spread column is the
            # interval the contrast actually used rather than a second statistic.
            row["values"] = {m: [v["mean"], (v["hi"] - v["lo"]) / 2.0]
                             for m, v in entry["metrics"].items()}
        rows.append(row)
    return rows


def _render_hypothesis(entry):
    """One labelled hypothesis: its co-primaries with Holm-adjusted p, or the
    named reason there is nothing to print."""
    out = [f"**{entry['name']}** (K={entry['K']})"]
    if entry["status"] != "OK":
        out.append(f"- {entry['status']} — {entry.get('reason') or 'no reason recorded'}")
        return "\n".join(out) + "\n"
    out += ["", "| metric | mean Δ | 95% CI | p | p (Holm) | favours first | won |",
            "|---|---|---|---|---|---|---|"]
    for metric, row in entry["metrics"].items():
        out.append(f"| {_metric_label(metric)} | {_num(row['mean'], 4)} | "
                   f"[{_num(row['lo'], 4)}, {_num(row['hi'], 4)}] | {row['p']:.4g} | "
                   f"{row['p_holm']:.4g} | {'yes' if row['favors_first'] else 'no'} | "
                   f"{'yes' if row['won'] else 'no'} |")
    out += ["", f"**Verdict: {entry['verdict']}**"]
    if "alongside" in entry:
        side = entry["alongside"]
        out.append(f"- alongside — {side['name']}: verdict {side['verdict']}")
    return "\n".join(out) + "\n"


def _render_adjacent(rows, title):
    out = [f"*{title}*", "",
           "| pair | K | metric | mean Δ | 95% CI | p |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row["status"] != "OK":
            out.append(f"| {row['pair']} | {row['K']} | "
                       f"{_metric_label(row['metric'])} | "
                       f"{row['status']} — {row.get('reason', '')} | | |")
            continue
        out.append(f"| {row['pair']} | {row['K']} | {_metric_label(row['metric'])} | "
                   f"{_num(row['mean'], 4)} | [{_num(row['lo'], 4)}, "
                   f"{_num(row['hi'], 4)}] | {row['p']:.4g} |")
    return "\n".join(out) + "\n"


def render_tables(results):
    """The whole readout, in the plan's order, with every refusal visible."""
    camp = results["campaign"]
    gates = results["gates"]
    out = ["# exp_14 — random-yaw generalization: collected readouts", "",
           f"Generated {results['generated_at']} by `yaw_gen_collect.py` — every cell is "
           "re-validated from its own artifacts on each run; numbers never live in this file.",
           "",
           f"- campaign pin `{camp['pin']}` · step {camp['step']} · seeds {camp['seeds']} "
           f"· α={camp['alpha']} · stats backend: {camp['stats_backend']}",
           f"- co-primary metrics: {', '.join(camp['co_primary'])} "
           "(Holm over the two, within each labelled hypothesis)",
           "- **per-metric aggregation (Planner ruling, pre-registered "
           f"{results['generated_at'][:10]}):** "
           f"{results['aggregation']['ruling']}. Co-primaries: "
           f"{results['aggregation']['co_primary']}.",
           f"  - scene-mean: {', '.join(results['aggregation']['scene_mean'])} · "
           f"split-level: {', '.join(results['aggregation']['split_level'])}",
           f"  - {results['aggregation']['note']}.",
           "",
           "## 1. Cell inventory", ""]
    inv = results["inventory"]
    out += [f"- registered {inv['registered']} · validated {inv['valid']} · "
            f"missing {len(inv['missing'])} · refused {len(inv['rejected'])}",
            "- " + " · ".join(f"{k}: {v}" for k, v in inv["counts"].items()), "",
            "## 2. Validity gates (G1–G4) and the G5 check", "",
            render_gate_report(gates), ""]
    g5 = gates["G5"]
    out += [f"*G5 (external reproduction — a CHECK, never a gate): {g5['status']}.* "
            f"{g5['definition']}", ""]
    if g5["rows"]:
        out += ["| arm | metric | exp_11 conf | exp_14 Z | Δ | 3σ bound | beyond |",
                "|---|---|---|---|---|---|---|"]
        for row in g5["rows"]:
            out.append(f"| {row['arm']} | {row['metric']} | "
                       f"{_num(row['exp11_mean'])} ± {_num(row['exp11_std'])} | "
                       f"{_num(row['exp14_mean'])} ± {_num(row['exp14_std'])} | "
                       f"{_num(row['diff'], 4)} | {_num(row['bound'], 4)} | "
                       f"{'YES' if row['beyond'] else 'no'} |")
        out.append("")
    for note in g5["notes"]:
        out.append(f"- {note}")
    out += ["", "## 3. Absolute robustness m_R (the PRIMARY criterion)", ""]
    for k in KS:
        out += [f"**K={k}** ({'confirmatory' if k == 8 else 'descriptive'})", "",
                render_block_table(_rows_for(results, "R", k), HEADLINE_METRICS), ""]
        if ("cross_arm", int(k)) in {tuple(b) for b in gates["blocked_scopes"]}:
            out += [f"> **Cross-arm comparison at K={k} is BLOCKED** — the arms are not "
                    "rotation-matched (gate G4). The per-arm levels above stand; no "
                    "contrast between them may be read.", ""]
    out += ["## 4. Paired degradation Δ = m_R − m_Z (mean over 5 seed-paired diffs, ±½·95% CI)", ""]
    for k in KS:
        out += [f"**K={k}**", "", render_block_table(_delta_rows(results, k),
                                                     HEADLINE_METRICS), ""]
    out += ["## 5. Endpoint contrasts (H-P / H-M / H-S)", ""]
    hyp = results["hypotheses"]
    if hyp["suppressed"]:
        out += [f"> **SUPPRESSED — {hyp['suppression_reason']}.** No verdict is printed "
                "below; fix the gate, do not read past it.", ""]
    for name in ("H-P", "H-M", "H-S"):
        out += [_render_hypothesis(hyp[name]), ""]
    out += ["*K=1 (descriptive repeat):*", ""]
    for name in ("H-P", "H-M", "H-S"):
        entry = results["hypotheses_K1"][name]
        verdict_text = entry.get("verdict", entry["status"])
        out.append(f"- {entry['name']} → {verdict_text}"
                   + ("" if entry["status"] == "OK" else f" ({entry.get('reason', '')})"))
    adj = results["adjacent"]
    out += ["", "## 6. Adjacent fixed-order contrasts", "", f"{adj['note']}", "",
            _render_adjacent(adj["absolute"], "absolute m_R, fixed order "
                             + "→".join(ARM_ORDER)), "",
            _render_adjacent(adj["abs_delta"], "|Δ| (flatness), same fixed order"), "",
            "## 7. Geometry retrieval (rotated-gallery, confounded — descriptive only)",
            "", results["geometry_retrieval"]["disclosure"], ""]
    geom = results["geometry_retrieval"]
    out += ["| block | arm | K | "
            + " | ".join(_metric_label(m) for m in CONFOUNDED_METRICS) + " |",
            "|---|---|---|" + "---|" * len(CONFOUNDED_METRICS)]
    for kind in ("R", "Z"):
        for arm in ARM_ORDER:
            for k in KS:
                cell = geom["blocks"][kind][arm][str(k)]
                if cell["status"] != "OK":
                    out.append(f"| {kind} | {arm} | {k} | *{cell['status']}* | | |")
                    continue
                vals = " | ".join(_num(cell["values"][m][0]) + " ± "
                                  + _num(cell["values"][m][1])
                                  if m in cell["values"] else ""
                                  for m in CONFOUNDED_METRICS)
                out.append(f"| {kind} | {arm} | {k} | {vals} |")
    out += ["", "## 8. Validity cells (V) — QA only, never a headline", "",
            "| arm | angle | role | status | Δ vs unrotated (co-primaries) |",
            "|---|---|---|---|---|"]
    for row in results["validity_cells"]:
        delta = row.get("vs_unrotated") or {}
        detail = ", ".join(f"{m}: {_num(v, 4)}" for m, v in delta.items()) or "—"
        out.append(f"| {row['arm']} | {_num(row['rotate_deg'], 0)}° | {row['role']} | "
                   f"{row['status']} | {detail} |")
    out += ["", "## 9. Refused and missing cells", ""]
    if not inv["rejected"]:
        out.append("- no cell was refused.")
    for row in inv["rejected"]:
        out.append(f"- **REFUSED** `{row['cell']}` — " + "; ".join(row["reasons"]))
    missing = inv["missing"]
    out.append(f"- {len(missing)} registered cell(s) have not landed"
               + (": " + ", ".join(sorted(m["cell"] for m in missing[:12]))
                  + (" …" if len(missing) > 12 else "") if missing else "."))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", required=True,
                    help="the tree holding exp11_<ARM>/… run directories")
    ap.add_argument("--pin", required=True,
                    help="the campaign pin (source commit sha) every cell must carry")
    ap.add_argument("--ckpt-expect", default=V.CKPT_EXPECT,
                    help="audited arm -> checkpoint sha256 map (exp14_ckpt_expect.json)")
    ap.add_argument("--expected-count", type=int, default=V.EXPECTED_COUNT)
    ap.add_argument("--exp11-root", default=None,
                    help="where exp_11's committed conf rows live (G5 check only)")
    ap.add_argument("--out", default=None, help="markdown output (default: stdout)")
    ap.add_argument("--json", dest="json_out", default=None, help="JSON bundle output")
    args = ap.parse_args(argv)
    try:
        campaign = CampaignExpectation.from_files(args.pin, args.ckpt_expect,
                                                  expected_count=args.expected_count)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    store = collect_cells(args.output_root, campaign)
    results = build_results(store, exp11_root=args.exp11_root)
    text = render_tables(results)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")
    # The exit code is what a wrapper reads, so 0 must mean "the campaign produced
    # its readouts" — not merely "the collector ran". A gate failure and an
    # incomplete grid are different problems and get different codes.
    gates = results["gates"]
    if not gates["all_passed"]:
        print(f"gates did not pass ({gates['summary']}): H-readouts are SUPPRESSED",
              file=sys.stderr)
        return 3
    unfinished = {n: results["hypotheses"][n]["status"] for n in ("H-P", "H-M", "H-S")
                  if results["hypotheses"][n]["status"] != "OK"}
    if unfinished:
        print(f"gates passed, but the readouts are not complete: {unfinished}",
              file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
