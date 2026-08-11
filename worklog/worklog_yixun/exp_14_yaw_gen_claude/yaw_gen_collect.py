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
                      "cell path metrics input_hash assignment_hash offsets source_sha")

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


def slim(artifact):
    """Drop the tuple stream, keep what cross-cell reasoning needs."""
    stream = artifact.stream
    return CellData(cell=artifact.cell, path=artifact.path,
                    metrics=dict(artifact.record["metrics"]),
                    input_hash=stream.get("input_hash"),
                    assignment_hash=stream.get("assignment_hash"),
                    offsets=tuple(stream.get("offsets") or ()),
                    source_sha=artifact.record.get("source_sha"))


def load_cell(path, campaign):
    """``(CellData, reasons)`` — data only when every check passed."""
    try:
        artifact = parse_cell_artifact(path)
    except ArtifactError as exc:
        return None, [str(exc)]
    reasons = validate_cell_provenance(artifact, campaign.for_cell(artifact.cell))
    if reasons:
        return None, reasons
    return slim(artifact), []


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
    groups = {}
    for c in cells:
        groups.setdefault((c.cell.cell, int(c.cell.k), int(c.cell.seed)), []).append(c)
    for (celltype, k, seed), members in sorted(groups.items()):
        if len(members) < 2:
            continue                     # nothing to compare against; not a violation
        for field, kind in (("input_hash", "cross_arm_input_hash"),
                            ("assignment_hash", "cross_arm_assignment_hash")):
            if kind.endswith("assignment_hash") and celltype == "zref":
                continue                 # the unrotated block assigns nothing
            seen = {}
            for c in members:
                seen.setdefault(getattr(c, field), []).append(c.cell.arm)
            if len(seen) > 1:
                detail = "; ".join(
                    f"{(h or 'None')[:12]}: {sorted(arms)}" for h, arms in sorted(
                        seen.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))))
                violations.append(Violation(
                    kind=kind, scope=f"{celltype} K={k} s{seed}",
                    detail=(f"{celltype} cells at K={k} seed {seed} disagree on "
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
def golden_offsets(seed, n, img_w=V.IMG_W):
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
