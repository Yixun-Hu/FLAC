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
metric_direction = G14.metric_direction
aggregation_source = G14.aggregation_source
cell_observation = G14.cell_observation
pair_seeds = G14.pair_seeds
aggregate_cell = G14.aggregate_cell
golden_offsets = G14.golden_offsets
CONFOUNDED_METRICS = G14.CONFOUNDED_METRICS
HEADLINE_METRICS = G14.HEADLINE_METRICS


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
    """Every §4.3 equality, as NAMED problems ([] = the contrast may proceed)."""
    problems = []
    by_key = {}
    for art in cells:
        by_key.setdefault((int(art.cell.k), int(art.cell.seed)), []).append(art)

    # (a) across arms within (K, seed): input_hash equal for both arms, and
    #     assignment_hash equal for the R cells.
    for (k, seed), arts in sorted(by_key.items()):
        for kind in (T_BLOCK, R_BLOCK, V_BLOCK):
            same = [a for a in arts if a.cell.cell == kind]
            if len({a.cell.arm for a in same}) < 2:
                continue
            ih = {a.cell.arm: _hashes(a)[0] for a in same}
            if len(set(ih.values())) > 1:
                problems.append(
                    f"input_hash differs across arms at (K={k}, seed={seed}, "
                    f"{kind}): {ih} — the arms did not evaluate the same items, so "
                    "the cross-arm contrast is BLOCKED")
            if kind in (R_BLOCK, V_BLOCK):
                ah = {a.cell.arm: _hashes(a)[1] for a in same}
                if len(set(ah.values())) > 1:
                    problems.append(
                        f"assignment_hash differs across arms at (K={k}, "
                        f"seed={seed}, {kind}): {ah} — the arms did not receive the "
                        "same rotations, so the contrast is not rotation-matched "
                        "and is BLOCKED")

    # (b) within (arm, K, seed): T.input_hash == R.input_hash — pairing validity.
    by_arm = {}
    for art in cells:
        by_arm.setdefault((art.cell.arm, int(art.cell.k), int(art.cell.seed)),
                          {})[art.cell.cell] = art
    for (arm, k, seed), kinds in sorted(by_arm.items()):
        t, r = kinds.get(T_BLOCK), kinds.get(R_BLOCK)
        if t is None or r is None:
            continue
        if _hashes(t)[0] != _hashes(r)[0]:
            problems.append(
                f"T<->R pairing invalid for {arm} (K={k}, seed={seed}): the "
                f"unrotated cell's input_hash {_hashes(t)[0]} != the rotated "
                f"cell's {_hashes(r)[0]} — a seed-paired Δ over different item "
                "sets is not a paired difference, so it is BLOCKED")
    return problems


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
def gate_g1(vctl_t60, tbl_t60_by_seed, factor=5.0, seeds=SEEDS):
    """G1 positive control: VANL@90° must degrade by ≥ factor·σ̂ of VANL's T block.

    A harness that cannot detect non-invariance in a model known not to have it
    cannot be trusted to detect its absence anywhere else, so a FAIL here HALTS
    the readout rather than annotating it.
    """
    present = sorted(int(s) for s in tbl_t60_by_seed)
    missing = [s for s in seeds if s not in present]
    if missing:
        return {"gate": "G1", "status": "PENDING", "missing_seeds": missing,
                "detail": f"needs all {len(seeds)} T-cell seeds to estimate sigma; "
                          f"missing {missing}"}
    values = [float(tbl_t60_by_seed[s]) for s in present]
    sigma = st.stdev(values)
    baseline = st.mean(values)
    observed = float(vctl_t60) - baseline
    threshold = factor * sigma
    return {"gate": "G1", "status": "PASS" if observed >= threshold else "FAIL",
            "observed": observed, "sigma": sigma, "factor": factor,
            "threshold": threshold, "baseline": baseline,
            "detail": (f"VANL@90° T60 {vctl_t60:.4f} − T-block mean {baseline:.4f} "
                       f"= {observed:.4f}; need ≥ {factor}·σ̂({sigma:.4f}) "
                       f"= {threshold:.4f}")}


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


def _comparable_groups(cells):
    """How many §4.3 equalities there are actually two artifacts to compare."""
    n = 0
    by_key = {}
    for art in cells:
        by_key.setdefault((int(art.cell.k), int(art.cell.seed), art.cell.cell),
                          set()).add(art.cell.arm)
    n += sum(1 for arms in by_key.values() if len(arms) > 1)
    by_arm = {}
    for art in cells:
        by_arm.setdefault((art.cell.arm, int(art.cell.k), int(art.cell.seed)),
                          set()).add(art.cell.cell)
    n += sum(1 for kinds in by_arm.values() if {T_BLOCK, R_BLOCK} <= kinds)
    return n


def gate_g3(cells):
    """G3 assignment integrity: every §4.3 equality (see :func:`verify_hashes`).

    A gate that COMPARED NOTHING is PENDING, never PASS. "No violations found"
    over an empty set is vacuously true and would render green on a campaign
    where not a single cell has landed — the fail-open reading this whole design
    exists to refuse.
    """
    problems = verify_hashes(cells)
    compared = _comparable_groups(cells)
    if problems:
        return {"gate": "G3", "status": "FAIL", "problems": problems,
                "compared": compared,
                "detail": (f"{len(problems)} integrity violation(s); the affected "
                           "contrasts are BLOCKED")}
    if compared == 0:
        return {"gate": "G3", "status": "PENDING", "problems": [], "compared": 0,
                "detail": ("no hash equality has been TESTED yet: fewer than two "
                           "comparable cells have landed, so this gate has "
                           "verified nothing")}
    return {"gate": "G3", "status": "PASS", "problems": [], "compared": compared,
            "detail": f"all {compared} cross-arm and T↔R hash equalities hold"}


def gate_g4(cells, control=None, registry=None):
    """G4 admission: every cell's checkpoint is the pre-registered 40k file.

    The DEEP recomputation is exp15_admit_ckpt's (it needs torch and runs in the
    job, before the GPU is spent). What is checked here is that each landed cell
    RECORDS the admitted digest — the collector's job is to refuse a number whose
    provenance disagrees with the committed record, not to re-hash 724 MB × 42.
    """
    problems = []
    expected = {}
    # Enumerated over the REGISTERED arms, not over whichever arms happen to have
    # landed: while the training chain is unfinished YAWAUG has no admission
    # record at all, and driving this loop from the cells on disk would hide that
    # entirely whenever no YAWAUG cell had landed — which is exactly the state the
    # campaign is in before it starts.
    for arm in ARMS:
        try:
            expected[arm] = V.admission_expectation(
                arm, control or V.CONTROL_ADMISSION, registry or V.LAUNCH_REGISTRY)
        except ValueError as exc:
            problems.append(f"{arm}: {exc}")
    checked = 0
    for art in cells:
        exp = expected.get(art.cell.arm)
        if exp is None:
            continue
        checked += 1
        got = art.meta.get("ckpt_sha256")
        if got != exp["sha256"]:
            problems.append(
                f"{V.eval_name(art.cell)} evaluated ckpt {got} but the committed "
                f"record admits {exp['sha256']}")
    if problems:
        return {"gate": "G4", "status": "FAIL", "problems": problems,
                "checked": checked,
                "detail": f"{len(problems)} admission problem(s)"}
    if checked == 0:
        return {"gate": "G4", "status": "PENDING", "problems": [], "checked": 0,
                "detail": ("no cell has been checked against an admission record "
                           "yet: this gate has verified nothing")}
    return {"gate": "G4", "status": "PASS", "problems": [], "checked": checked,
            "detail": f"all {checked} cell(s) evaluated the admitted checkpoint"}


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


def render_h1_table(rows, blocked=None):
    """H1's confirmatory table, or the word BLOCKED and no numbers at all."""
    if blocked:
        # Deliberately not a table: a blocked contrast must not be rendered in a
        # shape that invites reading a number out of it.
        return (f"**H1 (K=8, co-primaries): BLOCKED** — {blocked}\n\n"
                "No numbers are reported for a blocked contrast.\n")
    out = ["| metric | Δ (YAWAUG − VANL) | 95% CI | p | p (Holm-2) | verdict |",
           "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        out.append("| {m} | {d} | [{lo}, {hi}] | {p} | {ph} | {v} |".format(
            m=_label(row["metric"]), d=_f(row["mean"]),
            lo=_f(row["lo"]), hi=_f(row["hi"]),
            p=_f(row["p"], 4), ph=_f(row["p_holm"], 4), v=row["verdict"]))
    return "\n".join(out) + "\n"


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
    return bundle


# --------------------------------------------------------------------------- #
# discovery + the whole report
# --------------------------------------------------------------------------- #
def collect_cells(output_root, pin=None, expected_count=V.EXPECTED_COUNT):
    """``(artifacts, missing, rejected)`` over the registered grid."""
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
            sha = art.meta.get("ckpt_sha256")
            reasons = V.validate_cell(path, cell, pin=pin, ckpt_sha=sha,
                                      expected_count=expected_count)
        except ValueError as exc:
            rejected.append((cell, str(exc)))
            continue
        if reasons:
            rejected.append((cell, "; ".join(reasons)))
            continue
        artifacts.append(art)
    return artifacts, missing, rejected


def _per_seed(cells, arm, kind, k, metrics):
    """``{metric: {seed: value}}`` for one block, each metric from ITS source."""
    out = {m: {} for m in metrics}
    for art in cells:
        c = art.cell
        if (c.arm, c.cell, int(c.k)) != (arm, kind, int(k)):
            continue
        values, reasons = cell_observation(art.record, required=metrics, optional=())
        if reasons:
            continue
        for m in metrics:
            if m in values:
                out[m][int(c.seed)] = values[m]
    return out


def build_results(output_root, pin=None, expected_count=V.EXPECTED_COUNT,
                  control=None, registry=None):
    """Every gate, then every contrast — in that order, and never the reverse."""
    cells, missing, rejected = collect_cells(output_root, pin=pin,
                                             expected_count=expected_count)
    blocks = {}
    for art in cells:
        blocks.setdefault((art.cell.arm, art.cell.cell, int(art.cell.k)),
                          []).append(int(art.cell.seed))

    gates = {"G3": gate_g3(cells), "G4": gate_g4(cells, control, registry),
             "G5": gate_g5({b: tuple(s) for b, s in blocks.items()
                            if b[1] in (T_BLOCK, R_BLOCK)})}
    r_probe = next((a for a in cells if a.cell.cell == R_BLOCK
                    and int(a.cell.seed) == 42 and int(a.cell.k) == CONFIRMATORY_K), None)
    gates["G2"] = (gate_g2(r_probe, expected_count) if r_probe else
                   {"gate": "G2", "status": "PENDING",
                    "detail": "no K=8 seed-42 random-yaw cell has landed yet"})
    vanl_t = _per_seed(cells, "VANL", T_BLOCK, CONFIRMATORY_K, ("T60",))["T60"]
    vctl = next((a for a in cells if a.cell.arm == "VANL"
                 and a.cell.cell == V_BLOCK), None)
    if vctl is None:
        gates["G1"] = {"gate": "G1", "status": "PENDING",
                       "detail": "the VANL@90° positive control has not landed"}
    else:
        vals, _ = cell_observation(vctl.record, required=("T60",), optional=())
        gates["G1"] = gate_g1(vals.get("T60", float("nan")), vanl_t)

    blocked = None
    if gates["G3"]["status"] == "FAIL":
        blocked = "; ".join(gates["G3"]["problems"][:2])
    elif gates["G4"]["status"] == "FAIL":
        blocked = "; ".join(gates["G4"]["problems"][:2])

    h1 = {"blocked": blocked, "rows": []}
    if blocked is None:
        y = _per_seed(cells, "YAWAUG", T_BLOCK, CONFIRMATORY_K, CO_PRIMARY)
        v = _per_seed(cells, "VANL", T_BLOCK, CONFIRMATORY_K, CO_PRIMARY)
        if all(len(y[m]) == len(SEEDS) and len(v[m]) == len(SEEDS) for m in CO_PRIMARY):
            h1["rows"] = contrast_rows(y, v)
        else:
            h1["blocked"] = None
            h1["pending"] = True
    return {"cells": [V.eval_name(a.cell) for a in cells],
            "missing": [[V.eval_name(c), why] for c, why in missing],
            "rejected": [[V.eval_name(c), why] for c, why in rejected],
            "blocks": {"/".join(str(x) for x in b): sorted(s)
                       for b, s in sorted(blocks.items())},
            "gates": gates, "h1": h1}


def render_report(results):
    out = ["## Gates (read before any number)", "",
           render_gate_report(results.get("gates", {})), "",
           "## H1 — clean cost/benefit at K=8 (confirmatory: Holm over 2 co-primaries)",
           ""]
    h1 = results.get("h1") or {}
    if h1.get("pending"):
        out.append("**PENDING** — not every (arm, K=8) T block has all 5 seeds.\n")
    else:
        out.append(render_h1_table(h1.get("rows") or [], blocked=h1.get("blocked")))
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    for name in ("report", "bundle"):
        s = sub.add_parser(name)
        s.add_argument("--output-root", required=True)
        s.add_argument("--pin", default=None)
        s.add_argument("--expected-count", type=int, default=V.EXPECTED_COUNT)
        s.add_argument("--json", default=None)
        s.set_defaults(cmd=name)
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    results = build_results(args.output_root, pin=args.pin,
                            expected_count=args.expected_count)
    if args.cmd == "report":
        print(render_report(results))
    else:
        bundle = results_bundle(results)
        text = json.dumps(bundle, indent=2, sort_keys=True)
        if args.json:
            open(args.json, "w").write(text + "\n")
            print(f"wrote {args.json}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
