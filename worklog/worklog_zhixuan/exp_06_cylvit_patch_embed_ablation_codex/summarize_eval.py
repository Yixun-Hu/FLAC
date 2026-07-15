"""Summarize exp06 Table-1 clean metrics and the K=1 C16 yaw gate.

The evaluator writes one JSON beside each checkpoint.  This script recognizes
only the canonical exp06 eval-name embedded by ``run_eval_one.sh``; that name
contains the patch-embedding variant, train seed/step, K, generation seed, and
the lossless yaw tag (for example ``yaw22p5``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
VARIANTS = ("linear", "cnn")
VARIANT_LABELS = {"linear": "CylViT-MLP/Linear", "cnn": "CylViT-CNN"}
TABLE1_SEEDS = (42, 43, 44, 45, 46)
C16_ANGLES = tuple(index * 22.5 for index in range(16))

# Paper Table-1 order.  The callback's JSON key ``FD`` is the paper's FD_G;
# retrieval uses generated-RIR -> GT-RIR, not the extra geometry recalls.
METRICS = (
    ("T60", "T60 (%) ↓", True),
    ("C50", "C50 (dB) ↓", True),
    ("EDT", "EDT (ms) ↓", True),
    ("RIR_to_GT_RIR_R@1", "R@1 (%) ↑", False),
    ("RIR_to_GT_RIR_R@5", "R@5 (%) ↑", False),
    ("RIR_to_GT_RIR_R@10", "R@10 (%) ↑", False),
    ("FD", "FD_G ↓", True),
)

EVAL_NAME_RE = re.compile(
    r"exp06_cylvit_pe_(?P<variant>linear|cnn)"
    r"_trainS(?P<train_seed>[0-9]+)"
    r"_step(?P<train_step>[0-9]+)"
    r"_K(?P<k>1|8)"
    r"_evalS(?P<eval_seed>[0-9]+)"
    r"_yaw(?P<yaw_tag>[0-9]+(?:p[0-9]+)?)"
)


@dataclass(frozen=True, order=True)
class Condition:
    variant: str
    train_seed: int
    train_step: int
    k: int
    eval_seed: int
    yaw: float


@dataclass(frozen=True)
class EvalRecord:
    condition: Condition
    metrics: dict[str, float]
    checkpoint: str
    path: Path


def find_flac_root() -> Path:
    for candidate in (EXP_DIR, *EXP_DIR.parents):
        if (candidate / "eval_FLAC.py").is_file() and (candidate / "outputs_FLAC").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the FLAC root above summarize_eval.py")


def yaw_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def yaw_text(yaw: float) -> str:
    return str(int(yaw)) if float(yaw).is_integer() else f"{yaw:g}"


def condition_text(condition: Condition) -> str:
    return (
        f"{condition.variant}/trainS{condition.train_seed}/step{condition.train_step}/"
        f"K{condition.k}/evalS{condition.eval_seed}/yaw{yaw_text(condition.yaw)}"
    )


def _as_float(value: object, *, key: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {key!r} is not numeric in {path}: {value!r}") from exc


def load_record(path: Path, match: re.Match[str]) -> EvalRecord:
    condition = Condition(
        variant=match.group("variant"),
        train_seed=int(match.group("train_seed")),
        train_step=int(match.group("train_step")),
        k=int(match.group("k")),
        eval_seed=int(match.group("eval_seed")),
        yaw=yaw_from_tag(match.group("yaw_tag")),
    )
    with path.open() as handle:
        payload = json.load(handle)
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"Missing metrics object in {path}")

    missing_keys = [key for key, _, _ in METRICS if key not in raw_metrics]
    if missing_keys:
        raise ValueError(f"Missing Table-1 metrics in {path}: {missing_keys}")
    metrics = {
        key: _as_float(raw_metrics[key], key=key, path=path)
        for key, _, _ in METRICS
    }

    recorded_yaw = payload.get("rotate_deg")
    if recorded_yaw is not None and abs(float(recorded_yaw) - condition.yaw) > 1e-9:
        raise ValueError(
            f"Filename/payload yaw mismatch in {path}: {condition.yaw} vs {recorded_yaw}"
        )
    cond_method = payload.get("cond_method")
    if cond_method not in (None, "vanilla"):
        raise ValueError(f"Expected vanilla conditioning in {path}, got {cond_method!r}")

    return EvalRecord(
        condition=condition,
        metrics=metrics,
        checkpoint=str(payload.get("ckpt_path", "")),
        path=path,
    )


def scan_records(
    directories: dict[str, Path], train_seed: int, train_step: int
) -> tuple[dict[Condition, EvalRecord], list[str]]:
    grouped: dict[Condition, list[EvalRecord]] = {}
    notes: list[str] = []
    for expected_variant, directory in directories.items():
        if not directory.is_dir():
            notes.append(f"Missing results directory for {expected_variant}: `{directory}`")
            continue
        for path in sorted(directory.glob("*metrics*.json")):
            match = EVAL_NAME_RE.search(path.name)
            if match is None:
                continue
            record = load_record(path, match)
            condition = record.condition
            if condition.train_seed != train_seed or condition.train_step != train_step:
                continue
            if condition.variant != expected_variant:
                raise ValueError(
                    f"Variant-tagged {condition.variant} result found in the "
                    f"{expected_variant} directory: {path}"
                )
            grouped.setdefault(condition, []).append(record)

    duplicates = {condition: rows for condition, rows in grouped.items() if len(rows) > 1}
    if duplicates:
        details = []
        for condition, rows in sorted(duplicates.items()):
            details.append(condition_text(condition) + ":")
            details.extend(f"  {row.path}" for row in rows)
        raise ValueError(
            "Multiple JSONs encode the same evaluation condition; choose one checkpoint/output "
            "set instead of silently mixing them:\n" + "\n".join(details)
        )
    return {condition: rows[0] for condition, rows in grouped.items()}, notes


def table1_conditions(train_seed: int, train_step: int) -> set[Condition]:
    return {
        Condition(variant, train_seed, train_step, k, eval_seed, 0.0)
        for variant in VARIANTS
        for k in (1, 8)
        for eval_seed in TABLE1_SEEDS
    }


def c16_conditions(train_seed: int, train_step: int, eval_seed: int) -> set[Condition]:
    return {
        Condition(variant, train_seed, train_step, 1, eval_seed, yaw)
        for variant in VARIANTS
        for yaw in C16_ANGLES
    }


def expected_conditions(
    suite: str, train_seed: int, train_step: int, c16_eval_seed: int
) -> set[Condition]:
    if suite == "table1":
        return table1_conditions(train_seed, train_step)
    if suite == "c16":
        return c16_conditions(train_seed, train_step, c16_eval_seed)
    return table1_conditions(train_seed, train_step) | c16_conditions(
        train_seed, train_step, c16_eval_seed
    )


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} ± {statistics.stdev(values):.4f} (n={len(values)})"


def table1_values(
    records: dict[Condition, EvalRecord],
    variant: str,
    train_seed: int,
    train_step: int,
    k: int,
    metric_key: str,
) -> list[float]:
    values = []
    for eval_seed in TABLE1_SEEDS:
        condition = Condition(variant, train_seed, train_step, k, eval_seed, 0.0)
        if condition in records:
            values.append(records[condition].metrics[metric_key])
    return values


def build_markdown(
    records: dict[Condition, EvalRecord],
    notes: list[str],
    expected: set[Condition],
    suite: str,
    train_seed: int,
    train_step: int,
    c16_eval_seed: int,
    directories: dict[str, Path],
) -> str:
    found_expected = expected & records.keys()
    missing = sorted(expected - records.keys())
    generated = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Exp06 CylViT Patch-Embedding Evaluation",
        "",
        f"Generated: {generated}",
        "",
        f"Protocol: train seed {train_seed}, train step {train_step}; full 6337-item unseen "
        "AcousticRooms split; cfg scale 1.0; one diffusion step; vanilla conditioning. "
        "Table-1 clean results use K in {1,8}, yaw 0, and generation seeds 42–46. "
        f"The C16 gate uses K=1 and generation seed {c16_eval_seed}.",
        "",
        f"Completion for suite `{suite}`: **{len(found_expected)}/{len(expected)} unique conditions**.",
        "",
        "Five Table-1 seeds are generation seeds for one trained checkpoint; they do not "
        "measure variance across independently trained models.",
        "",
        "## Table 1 — Clean Accuracy at Yaw 0",
        "",
        "Release-style all-sample aggregation is used. Lower is better for T60/C50/EDT/FD_G; "
        "higher is better for retrieval.",
        "",
        "| Variant | K | G | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---:|:---:|" + "---:|" * len(METRICS),
    ]

    for variant in VARIANTS:
        for k in (1, 8):
            cells = [
                fmt_mean_std(
                    table1_values(records, variant, train_seed, train_step, k, metric_key)
                )
                for metric_key, _, _ in METRICS
            ]
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {k} | ✓ | " + " | ".join(cells) + " |"
            )

    lines += [
        "",
        "## Clean CNN − Linear Difference",
        "",
        "Differences are computed from each variant's available five-seed mean. Negative is "
        "better for ↓ metrics; positive is better for ↑ metrics.",
        "",
        "| K | Metric | Linear mean | CNN mean | CNN − Linear |",
        "|---:|---|---:|---:|---:|",
    ]
    for k in (1, 8):
        for key, label, _ in METRICS:
            linear_values = table1_values(records, "linear", train_seed, train_step, k, key)
            cnn_values = table1_values(records, "cnn", train_seed, train_step, k, key)
            if linear_values and cnn_values:
                linear_mean = statistics.mean(linear_values)
                cnn_mean = statistics.mean(cnn_values)
                lines.append(
                    f"| {k} | {label} | {linear_mean:.4f} | {cnn_mean:.4f} | "
                    f"{cnn_mean - linear_mean:+.4f} |"
                )
            else:
                lines.append(f"| {k} | {label} | - | - | - |")

    lines += [
        "",
        f"## C16 Absolute Metrics — K=1, Eval Seed {c16_eval_seed}",
        "",
        "| Variant | Yaw (°) | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    for variant in VARIANTS:
        for yaw in C16_ANGLES:
            condition = Condition(variant, train_seed, train_step, 1, c16_eval_seed, yaw)
            row = records.get(condition)
            cells = [fmt(row.metrics[key]) if row else "-" for key, _, _ in METRICS]
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {yaw_text(yaw)} | "
                + " | ".join(cells)
                + " |"
            )

    lines += [
        "",
        "## C16 Delta from Each Variant's Yaw 0",
        "",
        "For ↓ metrics, positive deltas are worse. For ↑ metrics, negative deltas are worse.",
        "",
        "| Variant | Yaw (°) | "
        + " | ".join("Δ" + label.rsplit(" ", 1)[0] for _, label, _ in METRICS)
        + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    for variant in VARIANTS:
        base_condition = Condition(variant, train_seed, train_step, 1, c16_eval_seed, 0.0)
        base = records.get(base_condition)
        for yaw in C16_ANGLES[1:]:
            condition = Condition(variant, train_seed, train_step, 1, c16_eval_seed, yaw)
            row = records.get(condition)
            if base and row:
                cells = [f"{row.metrics[key] - base.metrics[key]:+.4f}" for key, _, _ in METRICS]
            else:
                cells = ["-"] * len(METRICS)
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {yaw_text(yaw)} | "
                + " | ".join(cells)
                + " |"
            )

    lines += [
        "",
        "## C16 Robustness Summary",
        "",
        "Mean absolute delta and worst delta exclude yaw 0; population std uses all available "
        "angles, including yaw 0.",
        "",
        "| Variant | Metric | Yaw-0 | Mean abs delta | Worst delta | Std over yaw | Angles |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        base_condition = Condition(variant, train_seed, train_step, 1, c16_eval_seed, 0.0)
        base = records.get(base_condition)
        for key, label, lower_is_better in METRICS:
            available = [
                records[Condition(variant, train_seed, train_step, 1, c16_eval_seed, yaw)]
                for yaw in C16_ANGLES
                if Condition(variant, train_seed, train_step, 1, c16_eval_seed, yaw) in records
            ]
            if base is None or len(available) < 2:
                lines.append(
                    f"| {VARIANT_LABELS[variant]} | {label} | "
                    f"{fmt(base.metrics[key]) if base else '-'} | - | - | - | {len(available)} |"
                )
                continue
            deltas = [
                record.metrics[key] - base.metrics[key]
                for record in available
                if record.condition.yaw != 0.0
            ]
            values = [record.metrics[key] for record in available]
            mean_abs = statistics.mean(abs(value) for value in deltas)
            worst = max(deltas) if lower_is_better else min(deltas)
            yaw_std = statistics.pstdev(values)
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {label} | {base.metrics[key]:.4f} | "
                f"{mean_abs:.4f} | {worst:+.4f} | {yaw_std:.4f} | {len(available)} |"
            )

    if missing:
        lines += ["", "## Missing Conditions", ""]
        lines.extend(f"- `{condition_text(condition)}`" for condition in missing)

    if notes:
        lines += ["", "## Scan Notes", ""]
        lines.extend(f"- {note}" for note in notes)

    lines += [
        "",
        "## Source Directories",
        "",
        f"- Linear: `{directories['linear']}`",
        f"- CNN: `{directories['cnn']}`",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    root = find_flac_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("all", "table1", "c16"), default="all")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-step", type=int, default=30000)
    parser.add_argument("--c16-eval-seed", type=int, default=42)
    parser.add_argument(
        "--linear-dir",
        type=Path,
        default=root / "outputs_FLAC/exp06_cylvit_pe_linear_trainS42",
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=root / "outputs_FLAC/exp06_cylvit_pe_cnn_trainS42",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 after writing the report when any expected condition is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directories = {
        "linear": args.linear_dir.expanduser().resolve(),
        "cnn": args.cnn_dir.expanduser().resolve(),
    }
    records, notes = scan_records(directories, args.train_seed, args.train_step)
    expected = expected_conditions(
        args.suite, args.train_seed, args.train_step, args.c16_eval_seed
    )
    report = build_markdown(
        records,
        notes,
        expected,
        args.suite,
        args.train_seed,
        args.train_step,
        args.c16_eval_seed,
        directories,
    )
    output = args.output
    if output is None:
        output = EXP_DIR / f"eval_summary_trainS{args.train_seed}_step{args.train_step}.md"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n")

    missing = expected - records.keys()
    print(f"Wrote {output}")
    print(f"Expected conditions: {len(expected)}; found: {len(expected) - len(missing)}; missing: {len(missing)}")
    if missing and args.require_complete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
