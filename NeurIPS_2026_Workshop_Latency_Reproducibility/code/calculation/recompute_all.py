#!/usr/bin/env python3
"""Recompute every repeat summary and the final latency table from bundled data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


METHOD_DIRECTORIES = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
)


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate_repeat(repeat_dir: Path) -> None:
    for method in METHOD_DIRECTORIES:
        method_dir = repeat_dir / method
        manifest = method_dir / "run_manifest.json"
        query_count = len(list((method_dir / "queries").glob("query_*.json")))
        if not manifest.is_file() or query_count != 128:
            raise RuntimeError(
                f"{repeat_dir.name}/{method} is incomplete: "
                f"manifest={manifest.is_file()}, query_json_count={query_count}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute the K_ctx=8, K_gen=1 latency summaries."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <bundle>/recomputed)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="do not compare the recomputed numeric table with expected/summary_final.json",
    )
    args = parser.parse_args()

    bundle = Path(__file__).resolve().parents[2]
    data = bundle / "data"
    output = (args.output_dir or bundle / "recomputed").resolve()
    output.mkdir(parents=True, exist_ok=True)

    summarize = bundle / "code" / "calculation" / "summarize_core_forward_latency.py"
    aggregate = bundle / "code" / "calculation" / "aggregate_kctx8_kgen1_latency.py"
    repeat_summaries: list[Path] = []

    run([sys.executable, str(bundle / "code" / "calculation" / "audit_bundle.py")])

    for ordinal in (1,):
        repeat_name = f"repeat_{ordinal:03d}"
        repeat_dir = data / "repeats" / repeat_name
        validate_repeat(repeat_dir)
        repeat_output = output / repeat_name
        repeat_output.mkdir(parents=True, exist_ok=True)
        summary_json = repeat_output / "summary.json"
        summary_md = repeat_output / "summary.md"
        run(
            [
                sys.executable,
                str(summarize),
                "--selection",
                str(data / "selection" / "frozen_16room_128.json"),
                "--vanilla-dir",
                str(repeat_dir / "vanilla_flac"),
                "--fa-bf-dir",
                str(repeat_dir / "fa_bf_flac"),
                "--yawaug-dir",
                str(repeat_dir / "yawaug_flac"),
                "--few-shot-dir",
                str(repeat_dir / "few_shot_rir"),
                "--fem-primary-dir",
                str(data / "fem" / "primary_97"),
                "--fem-oversized-dir",
                str(data / "fem" / "oversized_9"),
                "--fem-external-runtime",
                str(data / "fem" / "external_server_6query_runtime_recovery.json"),
                "--fem-fallback-runtime",
                str(data / "fem" / "fem_random_fallback_latency_16.json"),
                "--selector-latency",
                str(data / "selector" / "selector_latency_128.json"),
                "--output-json",
                str(summary_json),
                "--output-md",
                str(summary_md),
            ]
        )
        repeat_summaries.append(summary_json)

    final_json = output / "summary_final.json"
    final_md = output / "summary_final.md"
    command = [sys.executable, str(aggregate)]
    for summary in repeat_summaries:
        command.extend(("--summary", str(summary)))
    command.extend(("--output-json", str(final_json), "--output-md", str(final_md)))
    run(command)

    if not args.skip_verification:
        run(
            [
                sys.executable,
                str(bundle / "code" / "calculation" / "verify_numeric_equivalence.py"),
                "--expected",
                str(bundle / "expected" / "summary_final.json"),
                "--actual",
                str(final_json),
            ]
        )
    print(f"Recomputed final table: {final_md}")


if __name__ == "__main__":
    main()
