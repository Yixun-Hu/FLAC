#!/usr/bin/env python3
"""exp-09 Stage D atomic gate aggregator (plan §2 / §4).

``eval_FLAC.py`` and ``compare_predictions.py`` write their metric/verdict JSONs
directly and NEVER exit nonzero on a threshold breach. This reviewed wrapper is the
single point where the D-stage acceptance decision is (a) made total, (b) serialised
atomically, and (c) turned into a process exit code. Every D-stage acceptance run
goes through it.

Contract
--------
Inputs: one per-gate verdict JSON per gate, each of the shape::

    {"gate": "<name>", "pass": <bool>, "metrics": {<finite numbers>}, ...}

``--require`` lists the FULL set of gate names that must be present. The aggregator:

* rejects INCOMPLETE input   — a required gate has no verdict (or an unexpected/
  duplicate gate appears);
* rejects NON-FINITE input   — any NaN/Inf anywhere in a verdict;
* rejects a malformed verdict — missing/`non-bool ``pass``, missing file, bad JSON;
* writes ONE aggregate verdict ATOMICALLY (temp in the same dir, ``fsync``,
  ``os.replace``; ``allow_nan=False``) so a failure never leaves a partial verdict;
* exits 0 iff every gate passed, nonzero otherwise (and on any rejection above).

Design deliberately mirrors the Stage-A ``audit_convention.py`` atomic-writer and
finiteness helpers so the two never drift.

Run:  python aggregate_gate.py --out <verdict.json> --require d1_task d2_cond ... <inputs...>
"""
import argparse
import json
import math
import os
import sys
import tempfile
import typing as tp

# Atomic-writer temp-file naming — kept as module constants so tests can assert no
# temp sibling is left behind after a failed write. Temps are
# ``<out_dir>/.aggregate_gate.<rand>.tmp``.
OUTPUT_TMP_PREFIX = ".aggregate_gate."
OUTPUT_TMP_SUFFIX = ".tmp"

SCHEMA_VERSION = 1


class GateError(Exception):
    """Any refusal to aggregate: incomplete/unexpected/duplicate gates, a non-finite
    or malformed verdict, or a missing/unreadable input file. Fail-closed by
    construction (an explicit raise, so it survives ``python -O``)."""


# ============================================================================================
# Finiteness (mutation target: dropping this lets a NaN verdict through).
# ============================================================================================
def collect_non_finite(obj: tp.Any, path: str = "") -> tp.List[str]:
    """Recursively list json-paths of any non-finite float (NaN/Inf) in ``obj``."""
    bad: tp.List[str] = []
    if isinstance(obj, bool):
        return bad
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(collect_non_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(collect_non_finite(v, f"{path}[{i}]"))
    return bad


# ============================================================================================
# Atomic serialisation (mutation target: a non-atomic write leaves partial verdicts).
# ============================================================================================
def atomic_write_json(path: str, record: tp.Mapping[str, tp.Any]) -> None:
    """Serialise ``record`` to ``path`` atomically: temp file in the same directory,
    ``fsync``, then ``os.replace`` (atomic on POSIX). ``allow_nan=False`` -> any
    non-finite number raises BEFORE the destination is touched, and the temp file is
    removed. On ANY failure the destination is left untouched (no partial write)."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=OUTPUT_TMP_PREFIX, suffix=OUTPUT_TMP_SUFFIX)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, allow_nan=False, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic rename
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_verdict(path: str, record: tp.Mapping[str, tp.Any]) -> None:
    """Public alias for the atomic writer (name used by the D-stage callers/tests)."""
    atomic_write_json(path, record)


# ============================================================================================
# Verdict loading + validation.
# ============================================================================================
def load_verdict(path: str) -> tp.Dict[str, tp.Any]:
    """Load and validate one per-gate verdict JSON. Raises ``GateError`` on a missing
    file, unreadable/invalid JSON, a missing/non-string ``gate``, a missing/non-bool
    ``pass``, or ANY non-finite value anywhere in the record."""
    if not os.path.isfile(path):
        raise GateError(f"verdict file not found: {path}")
    try:
        with open(path) as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        raise GateError(f"could not read verdict JSON {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise GateError(f"verdict {path} is not a JSON object")

    gate = record.get("gate")
    if not isinstance(gate, str) or not gate:
        raise GateError(f"verdict {path} has a missing/invalid 'gate' name: {gate!r}")
    if "pass" not in record or not isinstance(record["pass"], bool):
        raise GateError(
            f"verdict {path} (gate {gate!r}) has a missing/non-bool 'pass' field: "
            f"{record.get('pass')!r}"
        )
    bad = collect_non_finite(record)
    if bad:
        raise GateError(f"verdict {path} (gate {gate!r}) has non-finite values at: {bad}")
    return record


def aggregate(
    verdict_paths: tp.Sequence[str], required_gates: tp.Sequence[str]
) -> tp.Dict[str, tp.Any]:
    """Load every verdict, enforce the required-gate set EXACTLY (missing, unexpected
    and duplicate gates all raise ``GateError``), and build the aggregate record.
    Raises before any output is produced, so a rejected aggregation writes nothing."""
    required = list(required_gates)
    required_set = set(required)
    if len(required_set) != len(required):
        raise GateError(f"--require lists a duplicate gate name: {required}")

    gates: tp.Dict[str, tp.Any] = {}
    for path in verdict_paths:
        record = load_verdict(path)
        name = record["gate"]
        if name in gates:
            raise GateError(f"duplicate verdict for gate {name!r} (paths collide)")
        if name not in required_set:
            raise GateError(
                f"unexpected gate {name!r} in {path}: not in the required set {sorted(required_set)}"
            )
        gates[name] = record

    missing = [g for g in required if g not in gates]
    if missing:
        raise GateError(f"incomplete input: required gate(s) with no verdict: {missing}")

    failed = [g for g in required if not gates[g]["pass"]]
    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "aggregate_gate.py",
        "required_gates": required,
        "n_gates": len(required),
        "gates": {g: gates[g] for g in required},  # deterministic order
        "failed_gates": failed,
        "all_pass": len(failed) == 0,
    }
    # Finiteness backstop over the assembled record (the per-verdict check already
    # ran, but this guards any field we synthesised).
    bad = collect_non_finite(record)
    if bad:
        raise GateError(f"assembled aggregate has non-finite values at: {bad}")
    return record


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    """Aggregate, write atomically, and return an exit code: 0 iff every gate passed,
    1 on any gate failure OR any rejection (incomplete/non-finite/malformed input).
    A rejection writes NO verdict; a completed aggregation always writes one."""
    parser = argparse.ArgumentParser(description="exp-09 Stage D atomic gate aggregator")
    parser.add_argument("--out", required=True, help="output aggregate verdict JSON path")
    parser.add_argument(
        "--require", required=True, metavar="G1,G2,...",
        help="comma-separated full set of gate names that must each have a verdict",
    )
    parser.add_argument("inputs", nargs="+", help="per-gate verdict JSON paths")
    args = parser.parse_args(argv)

    required = [g for g in args.require.split(",") if g]
    try:
        record = aggregate(args.inputs, required)
    except GateError as exc:
        print(f"aggregate_gate: REJECTED — {exc}", file=sys.stderr)
        return 2  # no verdict written on a rejection

    write_verdict(args.out, record)
    status = "PASS" if record["all_pass"] else "FAIL"
    print(
        f"aggregate_gate: {status} — {record['n_gates']} gate(s), "
        f"failed={record['failed_gates']} -> {args.out}"
    )
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
