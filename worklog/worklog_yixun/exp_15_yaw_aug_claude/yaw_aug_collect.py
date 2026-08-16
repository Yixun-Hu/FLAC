#!/usr/bin/env python3
"""exp_15 — collect the 42 eval cells into results (plan §5, §6.8).

The last thing between a landed cell and a number in a table. Everything here is
fail-closed: a contrast whose inputs cannot be proven equal renders BLOCKED, an
incomplete seed block renders PENDING, and neither ever renders a mean.

WHAT IS REUSED RATHER THAN REWRITTEN
------------------------------------
The verified, campaign-agnostic machinery is IMPORTED from exp_14's collector
(``yaw_gen_collect.py``), which is review-closed and ran a 106-cell campaign:
the t-distribution (``t_critical``/``_t_sf``, incomplete-beta based), the paired-t
estimate, Holm step-down, the metric direction/alias tables, the per-metric
aggregation ruling, the per-cell observation router, seed pairing, block
aggregation and the golden-offset recomputation. Re-implementing any of it would
create a second definition that can disagree with the campaign it is being
compared against — exp_15's external check reads exp_14's own rows.

Per-cell VALIDATION is likewise not duplicated: ``exp15_validate_cell`` is the one
source of truth for the grid, the protocol contract and the metric schema, and
this module calls it rather than restating any of it.

What is exp_15's own: the arms and cell classes, gates G1–G5 in this plan's
numbering, hypotheses H1/H2/H3, the verdict vocabulary, and the rendering.

THE ONE AMBIGUITY IN plan §5, AND HOW IT IS RESOLVED
----------------------------------------------------
Plan §5 says "the per-seed observation is the per-scene-mean aggregate for that
(arm, K, seed) cell" — flatly, for every metric. Read literally that would take a
per-scene mean of FD and of the retrieval metrics too. exp_14 pre-registered the
opposite ruling for those, before any of its cells ran, and recorded why:

    per-scene mean applies to the ACOUSTIC-PARAMETER family only — T60 (incl.
    Invalid-T60 handling), C50, EDT — matching the paper convention. RETRIEVAL
    (RIR_to_GT_RIR_R@k, and the quarantined RIR_to_geom_R@k) and FD use the
    SPLIT-LEVEL global metrics: within-scene retrieval among ~370 items is a
    different, easier task whose levels are incomparable to every previously
    published number in this program, and exp_01's noise-floor calibration
    against released Table-1 was on the global quantity; one-room Frechet is
    additionally small-sample biased.

exp_15 adopts that ruling, for the reason above AND because plan §5 pre-declares
an external check of exp_15's VANL rows against exp_14's Z rows: two campaigns
that aggregate the same artifact differently cannot be compared at all, so the
literal reading would silently void a check the plan requires. Recorded here, in
the code that acts on it, rather than left to a reader to reconstruct.

Retrieval is read as ``RIR_to_GT_RIR_R@k`` (audio-to-audio). ``RIR_to_geom_R@k``
is CONFOUNDED under a rotation experiment — it retrieves against the geometry
that the R block rotates — so it is carried descriptively and never scored.

Usage
-----
    python3 yaw_aug_collect.py report --output-root outputs_FLAC --pin <sha>
    python3 yaw_aug_collect.py bundle --output-root outputs_FLAC --pin <sha> \\
        --json yaw_aug_results.json
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKLOG = os.path.dirname(_HERE)
_EXP14 = os.path.join(_WORKLOG, "exp_14_yaw_gen_claude")
for _p in (_HERE, _EXP14):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp15_validate_cell as V              # noqa: E402  the ONE grid/schema source
import yaw_gen_collect as G14                # noqa: E402  the verified primitives

SCHEMA_VERSION = 1
ALPHA = 0.05

# --- re-exported primitives (imported, never re-implemented) -----------------
paired_t_ci = G14.paired_t_ci
t_critical = G14.t_critical
canonical_metric = G14.canonical_metric
aggregation_source = G14.aggregation_source


def metric_direction(metric):
    """exp_14's direction table, extended (not mutated) with §13's Invalid T60."""
    name = canonical_metric(metric)
    if name in EXTRA_DIRECTION:
        return EXTRA_DIRECTION[name]
    return G14.metric_direction(name)
cell_observation = G14.cell_observation
pair_seeds = G14.pair_seeds
aggregate_cell = G14.aggregate_cell
golden_offsets = G14.golden_offsets
CONFOUNDED_METRICS = G14.CONFOUNDED_METRICS
HEADLINE_METRICS = G14.HEADLINE_METRICS
# §13 puts Invalid T60 in the ACOUSTIC family (ten-room-family mean). exp_14's
# HEADLINE_METRICS predates that ratification and omits it, so exp_15 carries its
# own descriptive set rather than editing exp_14's (integrative review F4). It is
# descriptive ONLY — never in the confirmatory family.
DESCRIPTIVE_METRICS = ("T60", "Invalid T60", "C50", "EDT", "FD",
                       "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10")
# ...and its direction, which exp_14's table also lacks. Fewer invalid T60
# estimates is better. Looked up through metric_direction() below so exp_14's
# dict is read, never mutated.
EXTRA_DIRECTION = {"Invalid T60": "lower"}


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order (exp_14's, renamed).

    Named ``holm`` because plan §6.8's function inventory names it that; the
    implementation is exp_14's ``holm_adjust``, unmodified.
    """
    return G14.holm_adjust(pvals)


# --- exp_15's own campaign shape ---------------------------------------------
ARMS = V.ARMS                                  # ("YAWAUG", "VANL")
ARM_ORDER = ("YAWAUG", "VANL")
SEEDS = V.SEEDS
KS = V.KS
STEP = V.STEP
T_BLOCK, R_BLOCK, V_BLOCK = "tbl", "rrob", "vctl"
CO_PRIMARY = ("T60", "RIR_to_GT_RIR_R@1")      # plan §5: T60% and R@1
CONFIRMATORY_K = 8                             # K=8 confirmatory; K=1 descriptive
GATE_NAMES = ("G1", "G2", "G3", "G4", "G5")
VERDICTS = ("YAWAUG-SUPERIOR", "YAWAUG-INFERIOR", "NOT STATISTICALLY RESOLVED")
# Plan §5 retires equivalence for this experiment: "NOT STATISTICALLY RESOLVED"
# is the only non-directional verdict, and it is a statement about power, not a
# claim that the arms are the same.
NOT_RESOLVED = "NOT STATISTICALLY RESOLVED"


class ArtifactError(ValueError):
    """A cell's artifacts cannot be read as the cell they claim to be."""


Artifact = namedtuple("Artifact", "path cell record meta stream")
Completeness = namedtuple("Completeness", "status seeds missing")


def registered_cells():
    """The 42 registered cells — from the validator, not a second list."""
    return V.expected_grid()


# --------------------------------------------------------------------------- #
# parse_cell
# --------------------------------------------------------------------------- #
def cell_from_metrics_path(path):
    """The registered cell whose metrics artifact this path IS, or raise.

    Matched by reconstructing each registered cell's expected basename with this
    file's checkpoint stem and comparing — 42 exact comparisons. An unregistered
    artifact therefore cannot be parsed into "some cell" at all, which is the
    point of a registered grid.
    """
    base = os.path.basename(path)
    marker = "_metrics_"
    if marker not in base:
        raise ArtifactError(f"{base!r} is not a metrics artifact name")
    ckpt_stem = base.split(marker, 1)[0]
    fake_ckpt = os.path.join(os.path.dirname(path), ckpt_stem + ".ckpt")
    for cell in registered_cells():
        if os.path.basename(V.metrics_path(fake_ckpt, cell)) == base:
            return cell
    raise ArtifactError(
        f"{base!r} does not name any of the {len(registered_cells())} registered "
        "exp_15 cells: an artifact the plan never registered must not be read as "
        "evidence")


def _read_json_object(path, label):
    if not os.path.isfile(path):
        raise ArtifactError(f"{label} artifact missing: {path}")
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (ValueError, OSError) as exc:
        raise ArtifactError(f"{label} {path} could not be parsed: {exc}")
    if not isinstance(obj, dict):
        raise ArtifactError(f"{label} {path} is not a JSON object "
                            f"(got {type(obj).__name__})")
    return obj


def parse_cell(path):
    """Read one cell's three artifacts; every failure NAMES itself."""
    cell = cell_from_metrics_path(path)
    record = _read_json_object(path, "metrics")
    meta = _read_json_object(V.screenmeta_path(path), "screenmeta")
    stream = _read_json_object(V.stream_path(path), "stream")
    return Artifact(path=path, cell=cell, record=record, meta=meta, stream=stream)


# --------------------------------------------------------------------------- #
# validate_protocol — delegated, not duplicated
# --------------------------------------------------------------------------- #
def validate_protocol(path, pin=None, ckpt_sha=None,
                      expected_count=V.EXPECTED_COUNT):
    """Named reasons this cell is not protocol-conformant ([] = conformant).

    This is exp15_validate_cell.validate_cell verbatim. The collector must not
    own a second copy of the protocol contract or the metric schema: the whole
    point of the per-cell validator being run by the driver, by the wave
    submitter's dedup AND here is that all three ask the same question.
    """
    cell = cell_from_metrics_path(path)
    return V.validate_cell(path, cell, pin=pin, ckpt_sha=ckpt_sha,
                           expected_count=expected_count)


# --------------------------------------------------------------------------- #
# verify_hashes — plan §4.3, fail-closed
# --------------------------------------------------------------------------- #
def _hashes(art):
    """``(input_hash, assignment_hash)`` from the STREAM sidecar.

    Read from the stream, not the record: eval_FLAC writes the hash provenance
    into the metrics record only in random mode, so a T cell's input_hash exists
    only in its .stream.json — which is exactly why --record-stream is mandatory
    for every cell and not just the rotated ones.
    """
    return art.stream.get("input_hash"), art.stream.get("assignment_hash")


def verify_hashes(cells):
    """Every §4.3 equality as NAMED messages ([] = the contrast may proceed).

    One implementation, two shapes: :func:`hash_violations` carries the scope each
    violation blocks (which contrast, which K), this returns just the messages.
    """
    return [v["message"] for v in hash_violations(cells)]


# --------------------------------------------------------------------------- #
# check_completeness
# --------------------------------------------------------------------------- #
def check_completeness(got, block, seeds=SEEDS):
    """``Completeness`` for ONE (arm, cell, K) block; PENDING carries no numbers."""
    arm, kind, k = block
    if arm not in ARMS or kind not in V.CELLS:
        raise ValueError(f"block {block} is not registered in the exp_15 grid")
    for row in got:
        if row[0] not in ARMS or row[1] not in V.CELLS:
            raise ValueError(f"cell {row} is not registered in the exp_15 grid")
    present = sorted({int(r[3]) for r in got
                      if (r[0], r[1], int(r[2])) == (arm, kind, int(k))})
    missing = tuple(s for s in seeds if s not in present)
    return Completeness(status=("OK" if not missing else "PENDING"),
                        seeds=tuple(present), missing=missing)


# --------------------------------------------------------------------------- #
# orientation / verdicts
# --------------------------------------------------------------------------- #
def orient_metric(name, values):
    """Orient so POSITIVE always means WORSE (plan §5's δ convention).

    Lower-is-better metrics pass through; higher-is-better metrics flip sign. A
    metric with no registered direction RAISES — scoring "better" on a quantity
    whose direction nobody declared is how a sign error becomes a conclusion.
    """
    direction = metric_direction(name)          # raises KeyError if unregistered
    factor = 1.0 if direction == "lower" else -1.0
    return [factor * float(v) for v in values]


def verdict_for(metric, mean, p_holm, alpha=ALPHA):
    """Plan §5's per-metric verdict for a YAWAUG − VANL difference.

    No equivalence branch exists: a non-significant result is NOT STATISTICALLY
    RESOLVED, which says the study did not resolve the question — not that the
    arms are the same.
    """
    if p_holm >= alpha:
        return NOT_RESOLVED
    favours_yawaug = (mean < 0) if metric_direction(metric) == "lower" else (mean > 0)
    return "YAWAUG-SUPERIOR" if favours_yawaug else "YAWAUG-INFERIOR"


def contrast_rows(first, second, seeds=SEEDS, metrics=CO_PRIMARY, alpha=ALPHA):
    """Seed-paired ``first − second`` on each metric, with Holm over the family."""
    seeds = [int(s) for s in seeds]
    stats = []
    for metric in metrics:
        diffs = [float(first[metric][s]) - float(second[metric][s]) for s in seeds]
        stats.append((metric, paired_t_ci(diffs, alpha=alpha)))
    adjusted = holm([r.p for _, r in stats])
    rows = []
    for (metric, res), p_holm in zip(stats, adjusted):
        rows.append({"metric": metric, "n": res.n, "df": res.df, "mean": res.mean,
                     "sd": res.sd, "se": res.se, "lo": res.lo, "hi": res.hi,
                     "t": res.t, "p": res.p, "p_holm": p_holm,
                     "verdict": verdict_for(metric, res.mean, p_holm, alpha)})
    return rows


def secondary_rows(first, second, seeds=SEEDS, metrics=CO_PRIMARY, alpha=ALPHA):
    """As :func:`contrast_rows`, but SECONDARY: CIs, no confirmatory verdicts.

    Plan §5 puts exactly ONE family in the confirmatory bucket (H1's two
    co-primaries). H2 and H3 get the same paired machinery and the same CIs, and
    are labelled so no reader can mistake them for confirmatory claims — so no
    Holm adjustment and no verdict vocabulary is attached to them.
    """
    seeds = [int(s) for s in seeds]
    rows = []
    for metric in metrics:
        diffs = [float(first[metric][s]) - float(second[metric][s]) for s in seeds]
        res = paired_t_ci(diffs, alpha=alpha)
        rows.append({"metric": metric, "n": res.n, "df": res.df, "mean": res.mean,
                     "lo": res.lo, "hi": res.hi, "t": res.t, "p": res.p,
                     "family": "SECONDARY (not confirmatory; unadjusted)"})
    return rows


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def gate_g1(vctl_t60, tbl_t60_by_seed, factor=5.0, seeds=SEEDS, reference_seed=42):
    """G1 positive control: VANL@90° must degrade by ≥ factor·σ̂ of VANL's T block.

    THE COMPARATOR IS SEED 42, not the five-seed mean (eval-r2 review finding 1).
    Plan §5 pre-registers it literally: ``m_T60(VANL V@90°, s42, K8) − m_T60(VANL
    T, s42, K8)``. The V cell IS a seed-42 cell, so the seed-paired difference is
    the quantity the plan named; subtracting the block mean silently compares two
    different draws. The five-seed SD is still what sets the threshold — that is
    what σ̂ means here — so all five seeds remain required.

    A harness that cannot detect non-invariance in a model known not to have it
    cannot be trusted to detect its absence anywhere else, so a FAIL HALTS the
    readout rather than annotating it.
    """
    present = sorted(int(s) for s in tbl_t60_by_seed)
    missing = [s for s in seeds if s not in present]
    if missing:
        return {"gate": "G1", "status": "PENDING", "missing_seeds": missing,
                "detail": f"needs all {len(seeds)} T-cell seeds to estimate sigma; "
                          f"missing {missing}"}
    values = [float(tbl_t60_by_seed[s]) for s in present]
    sigma = st.stdev(values)
    reference = float(tbl_t60_by_seed[int(reference_seed)])
    observed = float(vctl_t60) - reference
    threshold = factor * sigma
    return {"gate": "G1", "status": "PASS" if observed >= threshold else "FAIL",
            "observed": observed, "sigma": sigma, "factor": factor,
            "threshold": threshold, "reference": reference,
            "reference_seed": int(reference_seed),
            "detail": (f"VANL@90° T60 {vctl_t60:.4f} − VANL T seed-{reference_seed} "
                       f"T60 {reference:.4f} = {observed:.4f}; need ≥ {factor}·"
                       f"σ̂({sigma:.4f}) = {threshold:.4f}")}


def gate_g2(artifact, expected_count=None):
    """G2 golden assignment: the offsets are the ones the registered draw makes.

    Recomputed with exp_14's ``draw_yaw_offsets`` — the function the evaluator
    itself calls — so this proves the cell was drawn by the registered algorithm,
    not merely that it matches a number somebody wrote down.
    """
    if artifact.cell.cell != R_BLOCK:
        return {"gate": "G2", "status": "N/A",
                "detail": "only a random-yaw cell has a drawn assignment"}
    offsets = artifact.stream.get("offsets") or []
    n = int(expected_count or len(offsets))
    seed = int(artifact.cell.seed)
    try:
        want = list(golden_offsets(seed, n))
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        return {"gate": "G2", "status": "PENDING",
                "detail": f"could not recompute the golden offsets: {exc}"}
    if list(offsets) == want:
        return {"gate": "G2", "status": "PASS", "n": n, "seed": seed,
                "detail": f"all {n} offsets equal the seed-{seed} golden draw"}
    bad = [i for i, (a, b) in enumerate(zip(offsets, want)) if a != b]
    return {"gate": "G2", "status": "FAIL", "n": n, "seed": seed,
            "mismatches": len(bad), "first_mismatch": bad[0] if bad else None,
            "detail": (f"{len(bad)} of {n} offsets differ from the seed-{seed} "
                       "golden draw — the evaluated rotations are not the "
                       "registered ones")}


def g3_obligations():
    """Every §4.3 equality the REGISTERED grid owes, as scope-tagged keys.

    Enumerated from the grid, not from what landed: a gate that becomes PASS the
    moment one pair agrees has not checked the campaign, it has checked a pair
    (eval-r2 review finding 4).
    """
    out = []
    kinds = {}
    for cell in registered_cells():
        kinds.setdefault((cell.cell, int(cell.k), int(cell.seed)), set()).add(cell.arm)
    for (kind, k, seed), arms in sorted(kinds.items()):
        # V owes NOTHING here (plan §5: YAWAUG@90 "carries no gate role";
        # integrative review F3). Its artifacts are still validated and its hashes
        # are still reported — as v_cell_problems, which suppress only the V
        # mechanism readout. Requiring them let a missing or mismatched YAWAUG V
        # cell hold the entire inference hostage.
        if kind == V_BLOCK or len(arms) < 2:
            continue
        out.append(("input_hash", kind, k, seed))
        # cross-arm assignment equality is a ROTATION-matching claim, so only the
        # registered R pairs owe it (plan §4.3).
        if kind == R_BLOCK:
            out.append(("assignment_hash", kind, k, seed))
    pairs = {}
    for cell in registered_cells():
        pairs.setdefault((cell.arm, int(cell.k), int(cell.seed)), set()).add(cell.cell)
    for (arm, k, seed), ks in sorted(pairs.items()):
        if {T_BLOCK, R_BLOCK} <= ks:
            out.append(("pairing", arm, k, seed))
    return tuple(out)


def hash_violations(cells):
    """Structured §4.3 violations: ``message`` plus the scope each one blocks."""
    found = []
    by_key = {}
    for art in cells:
        by_key.setdefault((int(art.cell.k), int(art.cell.seed), art.cell.cell),
                          []).append(art)
    for (k, seed, kind), arts in sorted(by_key.items()):
        if len({a.cell.arm for a in arts}) < 2:
            continue
        ih = {a.cell.arm: _hashes(a)[0] for a in arts}
        if len(set(ih.values())) > 1:
            found.append({"kind": "input_hash", "cell_class": kind, "k": k,
                          "seed": seed,
                          "message": (f"input_hash differs across arms at (K={k}, "
                                      f"seed={seed}, {kind}): {ih} — the arms did "
                                      "not evaluate the same items, so the "
                                      "cross-arm contrast is BLOCKED")})
        if kind in (R_BLOCK, V_BLOCK):
            ah = {a.cell.arm: _hashes(a)[1] for a in arts}
            if len(set(ah.values())) > 1:
                found.append({"kind": "assignment_hash", "cell_class": kind, "k": k,
                              "seed": seed,
                              "message": (f"assignment_hash differs across arms at "
                                          f"(K={k}, seed={seed}, {kind}): {ah} — the "
                                          "arms did not receive the same rotations, "
                                          "so the contrast is not rotation-matched "
                                          "and is BLOCKED")})
    by_arm = {}
    for art in cells:
        by_arm.setdefault((art.cell.arm, int(art.cell.k), int(art.cell.seed)),
                          {})[art.cell.cell] = art
    for (arm, k, seed), kinds in sorted(by_arm.items()):
        t, r = kinds.get(T_BLOCK), kinds.get(R_BLOCK)
        if t is None or r is None:
            continue
        if _hashes(t)[0] != _hashes(r)[0]:
            found.append({"kind": "pairing", "arm": arm, "cell_class": None, "k": k,
                          "seed": seed,
                          "message": (f"T<->R pairing invalid for {arm} (K={k}, "
                                      f"seed={seed}): the unrotated cell's "
                                      f"input_hash {_hashes(t)[0]} != the rotated "
                                      f"cell's {_hashes(r)[0]} — a seed-paired Δ "
                                      "over different item sets is not a paired "
                                      "difference, so it is BLOCKED")})
    return found


def g3_checked(cells):
    """The subset of :func:`g3_obligations` for which two artifacts are present."""
    have = set()
    by_key = {}
    for art in cells:
        by_key.setdefault((art.cell.cell, int(art.cell.k), int(art.cell.seed)),
                          set()).add(art.cell.arm)
    for (kind, k, seed), arms in by_key.items():
        if kind == V_BLOCK or len(arms) < 2:
            continue
        have.add(("input_hash", kind, k, seed))
        if kind == R_BLOCK:
            have.add(("assignment_hash", kind, k, seed))
    pairs = {}
    for art in cells:
        pairs.setdefault((art.cell.arm, int(art.cell.k), int(art.cell.seed)),
                         set()).add(art.cell.cell)
    for (arm, k, seed), ks in pairs.items():
        if {T_BLOCK, R_BLOCK} <= ks:
            have.add(("pairing", arm, k, seed))
    return have


def gate_g3(cells):
    """G3 assignment integrity, against the COMPLETE registered obligation set.

    PASS only when every registered equality has actually been compared. "No
    violations found" over one landed pair is not the campaign's integrity, and a
    gate that verified nothing at all is PENDING, never PASS.
    """
    all_violations = hash_violations(cells)
    # V-cell violations are REPORTED but do not fail the gate: they suppress the V
    # mechanism readout only (integrative review F3).
    violations = [v for v in all_violations if v.get("cell_class") != V_BLOCK]
    v_only = [v for v in all_violations if v.get("cell_class") == V_BLOCK]
    expected = set(g3_obligations())
    checked = g3_checked(cells) & expected
    base = {"gate": "G3", "checked": len(checked), "expected": len(expected),
            "problems": [v["message"] for v in violations], "violations": violations,
            "v_cell_problems": [v["message"] for v in v_only]}
    if violations:
        return dict(base, status="FAIL",
                    detail=(f"{len(violations)} integrity violation(s) over "
                            f"{len(checked)}/{len(expected)} checked obligations; "
                            "the affected contrasts are BLOCKED"))
    if len(checked) < len(expected):
        return dict(base, status="PENDING",
                    detail=(f"only {len(checked)} of {len(expected)} registered hash "
                            "equalities have both artifacts present; the rest have "
                            "not been TESTED"))
    return dict(base, status="PASS",
                detail=f"all {len(expected)} registered hash equalities hold")


def gate_g4(cells, control=None, registry=None):
    """G4 admission: every REGISTERED cell evaluated the pre-registered 40k file.

    Both the arms AND the cells are enumerated from the grid (eval-r2 review
    findings 4): driving either from what happens to be on disk let a single
    landed cell turn the gate green while 41 obligations were unmet, and let
    YAWAUG's missing admission record hide behind an absent cell.

    The DEEP recomputation is exp15_admit_ckpt's (it needs torch and runs in the
    job, before the GPU is spent). What is checked here is that each landed cell
    RECORDS the admitted digest.
    """
    problems, expected = [], {}
    for arm in ARMS:
        try:
            expected[arm] = V.admission_expectation(
                arm, control or V.CONTROL_ADMISSION, registry or V.LAUNCH_REGISTRY)
        except ValueError as exc:
            problems.append(f"{arm}: {exc}")
    landed = {V.eval_name(a.cell): a for a in cells}
    # The OBLIGATION set is the 40 hypothesis cells (T and R). A V cell that has
    # not landed must not make this gate PENDING and thereby suppress the
    # readout — that is the same V-gating leak review F3 closed in G3, one gate
    # over (plan §5: YAWAUG@90 "carries no gate role").
    #
    # A V cell that HAS landed is still compared, and a digest MISMATCH anywhere —
    # V included — is still a FAIL: evaluating the wrong checkpoint is a defect
    # wherever it happens. Only its absence is tolerated.
    obligations = [V.eval_name(c) for c in registered_cells()
                   if c.cell in (T_BLOCK, R_BLOCK)]
    checked = 0
    required = set(obligations)
    for name, art in sorted(landed.items()):
        exp = expected.get(art.cell.arm)
        if exp is None:
            continue
        if name in required:
            checked += 1
        got = art.meta.get("ckpt_sha256")
        if got != exp["sha256"]:
            problems.append(f"{name} evaluated ckpt {got} but the committed record "
                            f"admits {exp['sha256']}")
    base = {"gate": "G4", "problems": problems, "checked": checked,
            "expected": len(obligations)}
    if problems:
        return dict(base, status="FAIL",
                    detail=f"{len(problems)} admission problem(s)")
    if checked < len(obligations):
        return dict(base, status="PENDING",
                    detail=(f"only {checked} of {len(obligations)} registered cells "
                            "have been checked against an admission record"))
    return dict(base, status="PASS",
                detail=f"all {checked} registered cells evaluated the admitted checkpoint")


def registered_blocks():
    """The eight seed blocks a complete campaign has: 2 arms × {T,R} × 2 K."""
    return tuple((arm, kind, k) for arm in ARM_ORDER
                 for kind in (T_BLOCK, R_BLOCK) for k in KS)


def gate_g5(blocks, seeds=SEEDS):
    """G5 completeness: 5/5 seeds per (arm, block, K) before any mean ± std.

    Enumerated over the REGISTERED blocks, so a block with no cells at all is
    PENDING rather than invisible. Driving this from the blocks present made an
    empty campaign report PASS.
    """
    pending = {}
    for block in registered_blocks():
        got = {int(x) for x in blocks.get(block, ())}
        missing = [s for s in seeds if s not in got]
        if missing:
            pending["/".join(str(b) for b in block)] = missing
    return {"gate": "G5", "status": "PASS" if not pending else "PENDING",
            "pending": pending,
            "detail": ("every registered block has all 5 seeds" if not pending else
                       f"{len(pending)} of {len(registered_blocks())} block(s) "
                       f"incomplete")}


def external_tolerance(sd_a, sd_b, factor=3.0, n=5):
    """The pre-declared external-check bound: ``factor·√(σa² + σb²)/√n``."""
    return factor * math.sqrt(float(sd_a) ** 2 + float(sd_b) ** 2) / math.sqrt(n)


def external_check(metric, ours, theirs, sd_ours, sd_theirs, factor=3.0, n=5):
    """A NON-HALTING cross-campaign comparison (plan §5).

    An honestly measured discrepancy against exp_11's or exp_14's rows does not
    invalidate a within-pin exp_15 contrast — the two were measured at different
    pins — so this is disclosed, never enforced. ``halting`` is False by
    construction and is reported so a reader does not have to infer it.
    """
    diff = float(ours) - float(theirs)
    tol = external_tolerance(sd_ours, sd_theirs, factor=factor, n=n)
    return {"metric": canonical_metric(metric), "ours": float(ours),
            "theirs": float(theirs), "difference": diff, "tolerance": tol,
            "exceeds": abs(diff) > tol, "halting": False,
            "detail": (f"|{diff:.4f}| vs {factor}·√(σ²+σ²)/√{n} = {tol:.4f} "
                       "(disclosed, non-halting: cross-pin)")}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
_SHORT = {"RIR_to_GT_RIR_R@1": "R@1", "RIR_to_GT_RIR_R@5": "R@5",
          "RIR_to_GT_RIR_R@10": "R@10"}


def _label(metric):
    name = canonical_metric(metric)
    arrow = "↓" if metric_direction(name) == "lower" else "↑"
    return f"{_SHORT.get(name, name)} {arrow}"


def _f(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_block_table(rows, metrics):
    """Mean ± std per block, or PENDING — never a four-seed mean."""
    head = ["| arm | K | status | " + " | ".join(_label(m) for m in metrics) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in metrics) + " |"]
    for row in rows:
        if row.get("status") != "OK":
            seen = row.get("seeds", ())
            cells = [f"PENDING ({len(seen)}/5)"] * len(metrics)
        else:
            cells = []
            for m in metrics:
                mean, sd = row["values"][m]
                cells.append(f"{mean:.3f} ± {sd:.3f}")
        head.append("| {a} | {k} | {s} | ".format(a=row["arm"], k=row["K"],
                                                  s=row.get("status", "?"))
                    + " | ".join(cells) + " |")
    return "\n".join(head) + "\n"


def render_gate_report(gates):
    out = ["| gate | status | detail |", "| --- | --- | --- |"]
    for name in GATE_NAMES:
        g = gates.get(name) or {"status": "PENDING", "detail": "not evaluated"}
        out.append(f"| {name} | {g.get('status')} | {g.get('detail', '')} |")
    return "\n".join(out) + "\n"


def _json_safe(obj):
    """Make a payload STRICTLY valid JSON without losing what it said.

    ``paired_t_ci`` deliberately returns ±inf / p=0 for the zero-spread case (five
    identical non-zero differences ARE a real effect), and ``json.dumps`` renders
    that as the bare token ``Infinity`` — accepted by Python's own loader and
    REJECTED by ``JSON.parse``, i.e. by the HTML page this bundle exists to feed.
    Non-finite floats are therefore rendered as their names, as strings.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def results_bundle(payload):
    """The JSON bundle the HTML page reads; versioned so a reader can tell."""
    bundle = {"schema_version": SCHEMA_VERSION, "experiment": "exp_15",
              "alpha": ALPHA, "co_primary": list(CO_PRIMARY),
              "confirmatory_family": "H1 co-primaries only (Holm-2)",
              "verdict_vocabulary": list(VERDICTS),
              "aggregation": {
                  "scene_mean": list(G14.ACOUSTIC_METRICS),
                  "split_level": list(G14.SPLIT_LEVEL_METRICS),
                  "ruling": "exp_14's pre-registered per-metric ruling, adopted",
              },
              "confounded_descriptive_only": list(CONFOUNDED_METRICS)}
    bundle.update(payload)
    return _json_safe(bundle)


# --------------------------------------------------------------------------- #
# discovery + the whole report
# --------------------------------------------------------------------------- #
SCOPE_OF_INFERENCE = (
    "**Scope of inference (mandatory, plan §5).** Both arms have exactly ONE "
    "training run each (seed 42). The five eval seeds estimate EVALUATION-time "
    "variability — diffusion sampling and, in the R block, the rotation "
    "assignment — NOT training-run variability (init, data order, hardware "
    "nondeterminism) and NOT checkpoint-band position: matching seed and step "
    "aligns the two draws' schedules but cannot pair away band variance. All "
    "inference is conditional on these two specific training trajectories at the "
    "pre-registered 40,000-step endpoint; no checkpoint selection was performed.\n\n"
    "**Chained-vs-monolithic disclosure (plan §12).** The YAWAUG arm was trained "
    "as a 16-leg chain, the VANL control monolithically. A chained run is not "
    "bit-equivalent to a monolithic one: PL restores optimizer/scheduler/EMA/loop "
    "state but not RNG streams, so data-order and dropout streams re-seed per "
    "leg. The yaw-augmentation draws are exempt (counter-based on (seed, "
    "global_step, rank, index), resume-exact). This asymmetry is disclosed, not "
    "corrected for.\n\n"
    "**Aggregation routing (plan §13, ratified pre-data).** T60/C50/EDT/Invalid "
    "T60 are per-seed means over the ten room-family groups; FD and all retrieval "
    "metrics are split-level values. R@1 := RIR_to_GT_RIR_R@1. RIR_to_geom_R@k is "
    "quarantined descriptive-only under rotation."
)

# Gates whose FAILURE halts every hypothesis readout (plan §5: "Failure ⇒ HALT").
HALTING_GATES = ("G1", "G2", "G4")
REQUIRED_GATES = GATE_NAMES


def gate_disposition(gates):
    """What the gates permit: ``(halt, pending, g3_scopes)``.

    ONE place decides whether a number may be emitted (eval-r2 review finding 1).
    Previously only G3/G4 failures blocked H1, so a campaign could publish a
    contrast while G1 had explicitly FAILED and G2/G5 were still PENDING — the
    readout would have been read as validated when nothing had validated it.

    * G1/G2/G4 FAIL  ⇒ HALT: every hypothesis renders BLOCKED, no numbers.
    * ANY required gate PENDING ⇒ no hypothesis numbers (the campaign is not
      finished being checked, which is different from having failed).
    * G3 violations ⇒ block the SPECIFIC contrasts their scope touches.
    """
    halt = [f"{n} {gates.get(n, {}).get('status')}: {gates.get(n, {}).get('detail', '')}"
            for n in HALTING_GATES if gates.get(n, {}).get("status") == "FAIL"]
    if gates.get("G3", {}).get("status") == "FAIL":
        pass                                  # scoped below, not a global halt
    pending = [f"{n} PENDING" for n in REQUIRED_GATES
               if gates.get(n, {}).get("status") == "PENDING"]
    scopes = {}
    for violation in gates.get("G3", {}).get("violations", []):
        k = int(violation.get("k"))
        kind, klass = violation["kind"], violation.get("cell_class")
        touched = set()
        if kind == "input_hash":
            # a CROSS-ARM input mismatch means the two arms did not evaluate the
            # same items, which invalidates the contrast on that block AND the
            # flatness contrast that reads it
            touched |= {("H1", k), ("H2", k)} if klass == T_BLOCK else set()
            touched |= {("H3", k), ("H2", k)} if klass == R_BLOCK else set()
        elif kind == "assignment_hash":
            touched |= {("H2", k), ("H3", k)}
        elif kind == "pairing":
            # a WITHIN-ARM T<->R mismatch breaks only the seed-paired Δ, i.e. H2
            touched |= {("H2", k)}
        for scope in touched:
            scopes.setdefault(scope, []).append(violation["message"])
    return {"halt": halt, "pending": pending, "g3_scopes": scopes}


def collect_cells(output_root, pin=None, expected_count=V.EXPECTED_COUNT):
    """``(artifacts, missing, rejected)`` over the registered grid.

    Parses each cell ONCE and validates EXACTLY the payloads it retains
    (eval-r2 review finding 5): the path-based validator re-reads all three
    files, so a concurrent replacement could be validated as version B while
    version A's numbers went on to the table.
    """
    artifacts, missing, rejected = [], [], []
    ckpts = {}
    for cell in registered_cells():
        if cell.arm not in ckpts:
            ckpts[cell.arm] = V.checkpoint_path(output_root, cell.arm, cell.step)
        ckpt = ckpts[cell.arm]
        if ckpt is None:
            missing.append((cell, f"no unique step={cell.step} checkpoint"))
            continue
        path = V.metrics_path(ckpt, cell)
        if not os.path.isfile(path):
            missing.append((cell, path))
            continue
        try:
            art = parse_cell(path)
        except ArtifactError as exc:
            rejected.append((cell, str(exc)))
            continue
        try:
            reasons = V.validate_payloads(
                art.record, art.meta, art.stream, cell,
                pin=pin, ckpt_sha=art.meta.get("ckpt_sha256"),
                expected_count=expected_count)
        except ValueError as exc:
            rejected.append((cell, str(exc)))
            continue
        if reasons:
            rejected.append((cell, "; ".join(reasons)))
            continue
        artifacts.append(art)
    return artifacts, missing, rejected


def route_observations(cells, metrics=None):
    """``{(arm, kind, K): {metric: {seed: value}}}`` — each metric from ITS source.

    The routing is plan §13's, applied by exp_14's ``cell_observation``: acoustic
    from the ten-group scene mean, FD and retrieval from the split-level block.
    """
    metrics = tuple(DESCRIPTIVE_METRICS if metrics is None else metrics)
    out = {}
    for art in cells:
        c = art.cell
        values, reasons = cell_observation(art.record, required=metrics,
                                           optional=CONFOUNDED_METRICS)
        if reasons:
            continue
        block = out.setdefault((c.arm, c.cell, int(c.k)), {})
        for metric, value in values.items():
            block.setdefault(metric, {})[int(c.seed)] = value
    return out


def block_rows(routed, kind, k, metrics):
    """Aggregate rows (mean ± std over the five seeds) for one cell class."""
    rows = []
    for arm in ARM_ORDER:
        per_seed = routed.get((arm, kind, int(k)), {})
        seeds = sorted(set.intersection(*[set(per_seed[m]) for m in metrics])
                       if all(m in per_seed for m in metrics) else set())
        if len(seeds) != len(SEEDS):
            rows.append({"arm": arm, "K": k, "status": "PENDING", "seeds": tuple(seeds)})
            continue
        values = {m: (st.mean([per_seed[m][s] for s in seeds]),
                      st.stdev([per_seed[m][s] for s in seeds])) for m in metrics}
        rows.append({"arm": arm, "K": k, "status": "OK", "seeds": tuple(seeds),
                     "values": values})
    return rows


def _complete(per_seed, metrics):
    return all(m in per_seed and len(per_seed[m]) == len(SEEDS) for m in metrics)


def hypotheses(routed, disposition, k, metrics=CO_PRIMARY, alpha=ALPHA):
    """H1 (confirmatory at K=8), H2 and H3 (secondary) for one K.

    Every one of them passes through the SAME disposition: a halting gate failure
    blocks all three, any pending required gate suppresses all numbers, and a G3
    violation blocks exactly the hypotheses its scope touches.
    """
    def blocked_for(name):
        if disposition["halt"]:
            return "gates HALTED: " + "; ".join(disposition["halt"])
        scoped = disposition["g3_scopes"].get((name, int(k)))
        if scoped:
            return "; ".join(scoped)
        return None

    def pending_note():
        return ("gates not yet complete: " + ", ".join(disposition["pending"])
                if disposition["pending"] else None)

    out = {}
    yt = routed.get(("YAWAUG", T_BLOCK, k), {})
    vt = routed.get(("VANL", T_BLOCK, k), {})
    yr = routed.get(("YAWAUG", R_BLOCK, k), {})
    vr = routed.get(("VANL", R_BLOCK, k), {})

    # --- H1: the clean cost/benefit, m_T(YAWAUG) vs m_T(VANL) ----------------
    # THE CONFIRMATORY FAMILY IS K=8 ONLY (plan §5: "K=8 confirmatory; K=1 repeats
    # everything descriptively"). K=1 gets the same paired machinery and the same
    # CIs with NO Holm adjustment and NO superiority/inferiority verdict —
    # rendering it as a second confirmatory family would inflate the multiplicity
    # the plan deliberately fixed at two tests (integrative review F2).
    confirmatory = int(k) == CONFIRMATORY_K
    h1 = {"name": "H1", "k": k,
          "family": ("CONFIRMATORY (Holm over 2 co-primaries)" if confirmatory
                     else "DESCRIPTIVE (K=1 repeat; unadjusted, no verdicts)"),
          "confirmatory": confirmatory,
          "blocked": blocked_for("H1"), "pending": pending_note(), "rows": []}
    if not h1["blocked"] and not h1["pending"]:
        if _complete(yt, metrics) and _complete(vt, metrics):
            h1["rows"] = (contrast_rows(yt, vt, metrics=metrics, alpha=alpha)
                          if confirmatory
                          else secondary_rows(yt, vt, metrics=metrics, alpha=alpha))
        else:
            h1["pending"] = "the K=%s T blocks do not have all five seeds" % k
    out["H1"] = h1

    # --- H2: does the augmentation buy FLATNESS? -----------------------------
    # delta(a,s) = orient(m_R - m_T) so POSITIVE means worse under rotation;
    # d_s = delta(VANL,s) - delta(YAWAUG,s), expected > 0 if YAWAUG is flatter.
    h2 = {"name": "H2", "k": k, "family": "SECONDARY (not confirmatory; unadjusted)",
          "blocked": blocked_for("H2"), "pending": pending_note(), "rows": [],
          "definition": ("δ(a,s) = orient(m_R(a,s) − m_T(a,s)); "
                         "d_s = δ(VANL,s) − δ(YAWAUG,s); d > 0 ⇒ YAWAUG flatter")}
    if not h2["blocked"] and not h2["pending"]:
        if all(_complete(b, metrics) for b in (yt, vt, yr, vr)):
            first, second = {}, {}
            for metric in metrics:
                dv = orient_metric(metric, [vr[metric][s] - vt[metric][s] for s in SEEDS])
                dy = orient_metric(metric, [yr[metric][s] - yt[metric][s] for s in SEEDS])
                first[metric] = {s: dv[i] for i, s in enumerate(SEEDS)}
                second[metric] = {s: dy[i] for i, s in enumerate(SEEDS)}
            h2["rows"] = secondary_rows(first, second, metrics=metrics, alpha=alpha)
        else:
            h2["pending"] = "H2 needs complete T and R blocks for both arms at K=%s" % k
    out["H2"] = h2

    # --- H3: absolute deployment comparison under rotation -------------------
    h3 = {"name": "H3", "k": k, "family": "SECONDARY (not confirmatory; unadjusted)",
          "blocked": blocked_for("H3"), "pending": pending_note(), "rows": [],
          "definition": "m_R(YAWAUG) − m_R(VANL), seed-paired"}
    if not h3["blocked"] and not h3["pending"]:
        if _complete(yr, metrics) and _complete(vr, metrics):
            h3["rows"] = secondary_rows(yr, vr, metrics=metrics, alpha=alpha)
        else:
            h3["pending"] = "the K=%s R blocks do not have all five seeds" % k
    out["H3"] = h3
    return out


# --------------------------------------------------------------------------- #
# external checks (non-halting, plan §5)
# --------------------------------------------------------------------------- #
def exp14_z_rows(output_root, k, arm="VANL", metrics=CO_PRIMARY, step=STEP):
    """exp_14's VANL zref rows at this K, read from its committed artifacts."""
    ckpt = V.checkpoint_path(output_root, arm, step)
    if ckpt is None:
        return {}
    directory, stem = os.path.dirname(ckpt), os.path.basename(ckpt)[:-len(".ckpt")]
    per_seed = {m: {} for m in metrics}
    for seed in SEEDS:
        name = (f"{stem}_metrics_1_1.0_exp14_{arm}_zref_S{step}_s{seed}_K{k}.json")
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        try:
            record = _read_json_object(path, "exp_14 zref")
        except ArtifactError:
            continue
        values, reasons = cell_observation(record, required=metrics, optional=())
        if reasons:
            continue
        for m in metrics:
            per_seed[m][seed] = values[m]
    return per_seed


def exp11_q9_rows(output_root, k, metrics=CO_PRIMARY):
    """exp_11's Q9 VANL rows — the OTHER pre-declared external reference.

    Discovered by the same glob shape ``gen_model_comparison.py`` registers for
    the Q9 contract (``*exp11_VANL_q9_S40000_s4[2-6]_K<k>.json``), so the two
    read the same evidence rather than two descriptions of it. Descriptive and
    non-halting, exactly like the exp_14 comparison (integrative review F5).
    """
    pattern = os.path.join(output_root, "exp11_VANL", "**",
                           f"*exp11_VANL_q9_S{STEP}_s4[2-6]_K{k}.json")
    per_seed = {m: {} for m in metrics}
    for path in sorted(glob.glob(pattern, recursive=True)):
        if path.endswith(".screenmeta.json") or path.endswith(".stream.json"):
            continue
        try:
            record = _read_json_object(path, "exp_11 Q9")
        except ArtifactError:
            continue
        seed = record.get("seed")
        if not isinstance(seed, int):
            continue
        # SPLIT-LEVEL ONLY, deliberately. exp_11's Q9 cells predate
        # --record-per-scene and carry no `by_scene` block at all, so their T60 is
        # the split-level quantity — a DIFFERENT estimand from our ten-room-family
        # mean under §13. Comparing the two would manufacture a discrepancy out of
        # an aggregation difference, so the acoustic family is reported
        # UNAVAILABLE for this source rather than compared.
        values, reasons = G14.flat_observation(record, metrics=metrics)
        if reasons:
            continue
        for m in metrics:
            if aggregation_source(m) != "split" or m not in values:
                continue
            per_seed[m][int(seed)] = values[m]
    return per_seed


def _one_external(label, ours, theirs, metrics):
    checks = []
    for metric in metrics:
        a, b = ours.get(metric, {}), theirs.get(metric, {})
        if len(a) != len(SEEDS) or len(b) != len(SEEDS):
            why = (f"needs 5 seeds on both sides; have {len(a)} (ours) and "
                   f"{len(b)} ({label})")
            if label == "exp_11 Q9" and aggregation_source(metric) == "scene-mean":
                why = ("exp_11's Q9 cells predate --record-per-scene and carry no "
                       "by_scene block, so their value is SPLIT-LEVEL — a different "
                       "estimand from §13's ten-room-family mean. Not comparable, "
                       "so not compared.")
            checks.append({"metric": canonical_metric(metric), "source": label,
                           "status": "UNAVAILABLE", "halting": False,
                           "detail": why})
            continue
        av = [a[s] for s in SEEDS]
        bv = [b[s] for s in SEEDS]
        chk = external_check(metric, st.mean(av), st.mean(bv),
                             st.stdev(av), st.stdev(bv))
        chk["source"] = label
        chk["status"] = "EXCEEDS" if chk["exceeds"] else "CONSISTENT"
        checks.append(chk)
    return checks


def external_checks(routed, output_root, k=CONFIRMATORY_K, metrics=CO_PRIMARY):
    """exp_15's VANL T rows vs exp_14's Z rows — DISCLOSED, never halting."""
    ours = routed.get(("VANL", T_BLOCK, k), {})
    return (_one_external("exp_14 Z", ours, exp14_z_rows(output_root, k, metrics=metrics),
                          metrics)
            + _one_external("exp_11 Q9", ours, exp11_q9_rows(output_root, k, metrics),
                            metrics))


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_contrast_table(rows, title, blocked=None, pending=None, holm=True):
    if blocked:
        return f"**{title}: BLOCKED** — {blocked}\n\nNo numbers are reported for a blocked contrast.\n"
    if pending:
        return f"**{title}: PENDING** — {pending}\n\nNo numbers are reported for an incomplete block.\n"
    cols = ["metric", "Δ", "95% CI", "p"] + (["p (Holm-2)", "verdict"] if holm else [])
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        cells = [_label(row["metric"]), _f(row["mean"]),
                 f"[{_f(row['lo'])}, {_f(row['hi'])}]", _f(row["p"], 4)]
        if holm:
            cells += [_f(row["p_holm"], 4), row["verdict"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_h1_table(rows, blocked=None, pending=None, k=CONFIRMATORY_K,
                    confirmatory=None):
    """H1's table for one K: confirmatory at K=8, descriptive at K=1.

    The K=1 table deliberately has no Holm column and no verdict column — those
    belong to the single registered confirmatory family, and printing them beside
    a descriptive repeat is how a second family gets read into the record.
    """
    if confirmatory is None:
        confirmatory = int(k) == CONFIRMATORY_K
    tag = ("K=%s, co-primaries" % k if confirmatory
           else "K=%s, DESCRIPTIVE repeat" % k)
    if blocked:
        return (f"**H1 ({tag}): BLOCKED** — {blocked}\n\n"
                "No numbers are reported for a blocked contrast.\n")
    if pending:
        return (f"**H1 ({tag}): PENDING** — {pending}\n\n"
                "No numbers are reported for an incomplete block.\n")
    cols = (["metric", "Δ (YAWAUG − VANL)", "95% CI", "p", "p (Holm-2)", "verdict"]
            if confirmatory else
            ["metric", "Δ (YAWAUG − VANL)", "95% CI", "p"])
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        cells = [_label(row["metric"]), _f(row["mean"]),
                 f"[{_f(row['lo'])}, {_f(row['hi'])}]", _f(row["p"], 4)]
        if confirmatory:
            cells += [_f(row["p_holm"], 4), row["verdict"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def render_external_table(checks):
    out = ["| metric | source | ours | theirs | Δ | tolerance | status |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in checks:
        if c.get("status") == "UNAVAILABLE":
            out.append(f"| {c['metric']} | {c.get('source','')} | — | — | — | — | "
                       f"UNAVAILABLE ({c['detail']}) |")
            continue
        out.append("| {m} | {s} | {a} | {b} | {d} | {t} | {st} |".format(
            m=_label(c["metric"]), s=c.get("source", ""), a=_f(c["ours"]),
            b=_f(c["theirs"]), d=_f(c["difference"]), t=_f(c["tolerance"]),
            st=c["status"]))
    return "\n".join(out) + "\nNon-halting by construction: these are cross-pin comparisons.\n"


def render_v_readouts(v_rows):
    """A defective V cell renders the WORD, never the number."""
    out = ["| arm | role | T60 | note |", "| --- | --- | --- | --- |"]
    for row in v_rows:
        if row.get("withheld"):
            value = f"WITHHELD (hash defect: {row['withheld']})"
        else:
            value = _f(row.get("T60"))
        out.append(f"| {row['arm']} | {row['role']} | {value} | {row['note']} |")
    return "\n".join(out) + "\n"


def render_report(results):
    """The whole §5/§6.8 report, in the order a reader must meet it."""
    gates = results.get("gates", {})
    out = ["# exp_15 yaw_aug — results", "",
           "## Validity gates (read before any number)", "",
           render_gate_report(gates), ""]
    disp = results.get("disposition", {})
    if disp.get("halt"):
        out += ["> **HALTED.** " + "; ".join(disp["halt"]), ""]
    if disp.get("pending"):
        out += ["> **Gates incomplete:** " + ", ".join(disp["pending"])
                + " — no hypothesis numbers are reported.", ""]
    for k in (CONFIRMATORY_K, 1):
        tag = "confirmatory" if k == CONFIRMATORY_K else "descriptive repeat"
        hyp = (results.get("hypotheses") or {}).get(str(k), {})
        out += [f"## K = {k} ({tag})", "",
                "### H1 — clean cost/benefit (m_T YAWAUG vs VANL)", ""]
        h1 = hyp.get("H1", {})
        out += [f"_{h1.get('family', '')}_", "",
                render_h1_table(h1.get("rows") or [], blocked=h1.get("blocked"),
                                pending=h1.get("pending"), k=k,
                                confirmatory=h1.get("confirmatory")), ""]
        for name, heading in (("H2", "H2 — does augmentation buy flatness? (secondary)"),
                              ("H3", "H3 — absolute deployment under rotation (secondary)")):
            h = hyp.get(name, {})
            out += [f"### {heading}", "", f"_{h.get('definition', '')}_", "",
                    render_contrast_table(h.get("rows") or [], f"{name} (K={k})",
                                          blocked=h.get("blocked"),
                                          pending=h.get("pending"), holm=False), ""]
        for kind, label in ((T_BLOCK, "T block (θ=0)"), (R_BLOCK, "R block (random yaw)")):
            rows = (results.get("blocks") or {}).get(f"{kind}/{k}") or []
            out += [f"### {label} — mean ± std over seeds", "",
                    render_block_table(rows, DESCRIPTIVE_METRICS), ""]
    out += ["## Validity-control cells (V block)", "",
            render_v_readouts(results.get("v_readouts") or []), "",
            "## External reproduction checks (non-halting)", "",
            render_external_table(results.get("externals") or []), "",
            "## Quarantined, descriptive only — RIR_to_geom_R@k", "",
            "_Confounded by construction: it retrieves against the geometry the R "
            "block rotates (plan §13)._", "",
            render_block_table((results.get("blocks") or {}).get(f"{R_BLOCK}/8") or [],
                               CONFOUNDED_METRICS), "",
            "## Scope of inference", "", SCOPE_OF_INFERENCE, ""]
    return "\n".join(out)


def build_results(output_root, pin=None, expected_count=V.EXPECTED_COUNT,
                  control=None, registry=None):
    """Every gate, then the disposition, then — only if it permits — the numbers."""
    cells, missing, rejected = collect_cells(output_root, pin=pin,
                                             expected_count=expected_count)
    seeds_by_block = {}
    for art in cells:
        seeds_by_block.setdefault((art.cell.arm, art.cell.cell, int(art.cell.k)),
                                  []).append(int(art.cell.seed))

    gates = {"G3": gate_g3(cells), "G4": gate_g4(cells, control, registry),
             "G5": gate_g5({b: tuple(s) for b, s in seeds_by_block.items()
                            if b[1] in (T_BLOCK, R_BLOCK)})}
    # The ladder names ONE probe: YAWAUG / rrob / K=8 / seed 42 (plan §7-8).
    # Accepting "the first K=8 seed-42 R cell from either arm" could have gated
    # the campaign on VANL's draw instead (integrative review F8).
    probe_cell = V.Cell("YAWAUG", R_BLOCK, STEP, 42, CONFIRMATORY_K, None)
    probe_name = V.eval_name(probe_cell)
    r_probe = next((a for a in cells if a.cell == probe_cell), None)
    if r_probe is None:
        gates["G2"] = {"gate": "G2", "status": "PENDING", "probe": probe_name,
                       "detail": f"the registered probe {probe_name} has not landed"}
    else:
        gates["G2"] = dict(gate_g2(r_probe, expected_count), probe=probe_name)
        gates["G2"]["detail"] = f"{probe_name}: {gates['G2']['detail']}"
    routed = route_observations(cells)
    vanl_t = routed.get(("VANL", T_BLOCK, CONFIRMATORY_K), {}).get("T60", {})
    vctl = next((a for a in cells if a.cell.arm == "VANL"
                 and a.cell.cell == V_BLOCK), None)
    if vctl is None:
        gates["G1"] = {"gate": "G1", "status": "PENDING",
                       "detail": "the VANL@90° positive control has not landed"}
    else:
        vals, _ = cell_observation(vctl.record, required=("T60",), optional=())
        gates["G1"] = gate_g1(vals.get("T60", float("nan")), vanl_t)

    disposition = gate_disposition(gates)
    hyp = {str(k): hypotheses(routed, disposition, k) for k in (CONFIRMATORY_K, 1)}
    blocks = {}
    for kind in (T_BLOCK, R_BLOCK):
        for k in KS:
            blocks[f"{kind}/{k}"] = block_rows(routed, kind, k,
                                               DESCRIPTIVE_METRICS + CONFOUNDED_METRICS)
    # A V cell with a hash defect must not publish a NUMBER (re-review finding 4).
    # Excluding its defects from G3 was right — they must not block inference —
    # but the readout itself is exactly what those defects invalidate, so it is
    # WITHHELD rather than printed. Reported per arm, keyed by the cell's own
    # eval name so a reader can see which artifact is implicated.
    v_defects = {}
    for violation in (gates["G3"].get("v_cell_problems") or []):
        for art in cells:
            if art.cell.cell == V_BLOCK and V.eval_name(art.cell) in violation:
                v_defects.setdefault(art.cell.arm, []).append(violation)
                break
        else:
            v_defects.setdefault("*", []).append(violation)
    v_readouts = []
    for art in sorted((a for a in cells if a.cell.cell == V_BLOCK),
                      key=lambda a: a.cell.arm):
        vals, _ = cell_observation(art.record, required=("T60",), optional=())
        defects = v_defects.get(art.cell.arm, []) + v_defects.get("*", [])
        row = {
            "arm": art.cell.arm, "eval_name": V.eval_name(art.cell),
            "role": ("G1 positive control" if art.cell.arm == "VANL"
                     else "mechanism readout"),
            "note": ("gates the harness's ability to detect non-invariance"
                     if art.cell.arm == "VANL" else
                     "DESCRIPTIVE ONLY — carries no gate role (plan §5, review F3)")}
        if defects:
            row["T60"] = None
            row["withheld"] = "; ".join(defects)
            row["note"] = ("WITHHELD (hash defect) — " + row["note"])
        else:
            row["T60"] = vals.get("T60")
            row["withheld"] = None
        v_readouts.append(row)
    return {"cells": [V.eval_name(a.cell) for a in cells],
            "missing": [[V.eval_name(c), why] for c, why in missing],
            "rejected": [[V.eval_name(c), why] for c, why in rejected],
            "blocks": blocks, "gates": gates, "disposition": disposition,
            "hypotheses": hyp, "v_readouts": v_readouts,
            "externals": external_checks(routed, output_root),
            "scope_of_inference": SCOPE_OF_INFERENCE,
            "h1": hyp[str(CONFIRMATORY_K)]["H1"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("report", "bundle"):
        sp = sub.add_parser(name)
        sp.add_argument("--output-root", required=True)
        sp.add_argument("--pin", default=None)
        sp.add_argument("--expected-count", type=int, default=V.EXPECTED_COUNT)
        sp.add_argument("--json", default=None)
        sp.set_defaults(cmd=name)
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    results = build_results(args.output_root, pin=args.pin,
                            expected_count=args.expected_count)
    if args.cmd == "report":
        print(render_report(results))
    else:
        # ensure_ascii=False: the bundle carries §/±/θ from the plan and the
        # report, and an escaped \u00a7 is not what a reader or the HTML page wants.
        text = json.dumps(results_bundle(results), indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False)
        if args.json:
            open(args.json, "w").write(text + "\n")
            print(f"wrote {args.json}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
