#!/usr/bin/env python3
"""Profile five localization methods at K_ctx=8 and one RIR per candidate.

The program deliberately orchestrates the established method implementations instead
of copying their inference logic.  Every measured repeat uses one frozen selection,
one GPU assignment, one serial method schedule, and a common result layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
METHOD_ORDER = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
    "fem_omp",
)
STAGE_ORDER = METHOD_ORDER[:-1]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: tuple[str, ...]
    environment: dict[str, str]


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_executable_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    """Make an executable absolute without dereferencing a virtualenv symlink."""

    path = Path(value).expanduser()
    joined = path if path.is_absolute() else base / path
    return Path(os.path.abspath(os.fspath(joined)))


def load_selection(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise ValueError(f"selection SHA-256 mismatch: {path}")
    if len(payload.get("records", ())) != int(payload.get("query_count", -1)):
        raise ValueError("selection query count is inconsistent")
    indices = [int(record["index"]) for record in payload["records"]]
    if len(indices) != len(set(indices)):
        raise ValueError("selection contains duplicate query indices")
    return payload


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 2:
        raise ValueError("latency benchmark config must use schema_version=2")
    required = (
        "python",
        "selection",
        "context_manifest",
        "geometry_audit",
        "dataset_root",
        "output_root",
        "agree_checkpoint",
        "fem_primary_dir",
        "fem_oversized_dir",
        "fem_external_runtime",
        "fem_fallback_runtime",
        "selector_latency",
        "methods",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"latency benchmark config is missing {missing}")
    for method in ("vanilla_flac", "fa_bf_flac", "yawaug_flac", "few_shot_rir"):
        if method not in config["methods"]:
            raise ValueError(f"latency benchmark config is missing method {method}")
    return config


def validate_config(config: dict) -> dict[str, Path]:
    paths = {
        "python": resolve_executable_path(config["python"]),
        "selection": resolve_path(config["selection"]),
        "context_manifest": resolve_path(config["context_manifest"]),
        "geometry_audit": resolve_path(config["geometry_audit"]),
        "dataset_root": resolve_path(config["dataset_root"]),
        "output_root": resolve_path(config["output_root"]),
        "agree_checkpoint": resolve_path(config["agree_checkpoint"]),
        "fem_primary_dir": resolve_path(config["fem_primary_dir"]),
        "fem_oversized_dir": resolve_path(config["fem_oversized_dir"]),
        "fem_external_runtime": resolve_path(config["fem_external_runtime"]),
        "fem_fallback_runtime": resolve_path(config["fem_fallback_runtime"]),
        "selector_latency": resolve_path(config["selector_latency"]),
    }
    for method, method_config in config["methods"].items():
        for field in ("model_config", "checkpoint"):
            if field in method_config:
                paths[f"{method}.{field}"] = resolve_path(method_config[field])
    for label, path in paths.items():
        if label == "output_root":
            continue
        if label == "dataset_root":
            if not path.is_dir():
                raise FileNotFoundError(f"{label}: {path}")
        elif label in ("fem_primary_dir", "fem_oversized_dir"):
            if not path.is_dir():
                raise FileNotFoundError(f"{label}: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    try:
        paths["output_root"].relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("output_root must remain inside the localization repository") from error
    load_selection(paths["selection"])
    return paths


def _common_learned_args(paths: dict[str, Path], selection: Path, output: Path) -> list[str]:
    return [
        "--agree-ckpt",
        str(paths["agree_checkpoint"]),
        "--context-manifest",
        str(paths["context_manifest"]),
        "--geometry-audit",
        str(paths["geometry_audit"]),
        "--pilot-manifest",
        str(selection),
        "--dataset-root",
        str(paths["dataset_root"]),
        "--output-dir",
        str(output),
    ]


def build_run_commands(
    config: dict,
    paths: dict[str, Path],
    *,
    selection: Path,
    output_dir: Path,
    warmup_query_count: int = 0,
) -> list[CommandSpec]:
    python = str(paths["python"])
    device = str(config.get("device", "cuda:0"))
    gpu = str(config.get("cuda_visible_devices", "0"))
    candidate_batch_size = int(config.get("candidate_batch_size", 64))
    score_seed = int(config.get("score_seed", 42))
    tau = float(config.get("tau", 0.1))
    gpu_environment = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "MPLCONFIGDIR": str(config.get("mplconfigdir", "/tmp/matplotlib-five-method-latency")),
    }
    commands: list[CommandSpec] = []
    for method in ("vanilla_flac", "fa_bf_flac", "yawaug_flac"):
        method_config = config["methods"][method]
        command = (
            python,
            str(REPO_ROOT / "localize_FLAC.py"),
            "--model-config",
            str(paths[f"{method}.model_config"]),
            "--ckpt-path",
            str(paths[f"{method}.checkpoint"]),
            *_common_learned_args(paths, selection, output_dir / method),
            "--device",
            device,
            "--cond-method",
            str(method_config["conditioning_method"]),
            "--candidate-batch-size",
            str(candidate_batch_size),
            "--sample-seed",
            str(score_seed),
            "--tau",
            str(tau),
            "--score-sample-counts",
            "1",
            "--synchronize-latency",
            "--warmup-query-count",
            str(warmup_query_count),
            "--measure-core-forward",
        )
        commands.append(CommandSpec(method, command, gpu_environment))

    few = config["methods"]["few_shot_rir"]
    few_command = (
        python,
        str(REPO_ROOT / "localize_baseline.py"),
        "--method",
        "few_shot_rir_waveform",
        "--model-config",
        str(paths["few_shot_rir.model_config"]),
        "--ckpt-path",
        str(paths["few_shot_rir.checkpoint"]),
        *_common_learned_args(paths, selection, output_dir / "few_shot_rir"),
        "--device",
        device,
        "--candidate-batch-size",
        str(candidate_batch_size),
        "--random-seed",
        str(score_seed),
        "--context-counts",
        "8",
        "--synchronize-latency",
        "--warmup-query-count",
        str(warmup_query_count),
        "--measure-core-forward",
    )
    commands.append(CommandSpec("few_shot_rir", few_command, gpu_environment))
    if tuple(spec.name for spec in commands) != STAGE_ORDER:
        raise RuntimeError("internal method ordering changed")
    return commands


def build_summary_command(
    paths: dict[str, Path], *, selection: Path, output_dir: Path
) -> CommandSpec:
    command = (
        str(paths["python"]),
        str(REPO_ROOT / "tools" / "summarize_core_forward_latency.py"),
        "--selection",
        str(selection),
        "--vanilla-dir",
        str(output_dir / "vanilla_flac"),
        "--fa-bf-dir",
        str(output_dir / "fa_bf_flac"),
        "--yawaug-dir",
        str(output_dir / "yawaug_flac"),
        "--few-shot-dir",
        str(output_dir / "few_shot_rir"),
        "--fem-primary-dir",
        str(paths["fem_primary_dir"]),
        "--fem-oversized-dir",
        str(paths["fem_oversized_dir"]),
        "--fem-external-runtime",
        str(paths["fem_external_runtime"]),
        "--fem-fallback-runtime",
        str(paths["fem_fallback_runtime"]),
        "--selector-latency",
        str(paths["selector_latency"]),
        "--output-json",
        str(output_dir / "summary.json"),
        "--output-md",
        str(output_dir / "summary.md"),
    )
    return CommandSpec("summarize", command, {})


def build_aggregate_command(
    paths: dict[str, Path], *, summaries: list[Path], output_root: Path
) -> CommandSpec:
    command = [
        str(paths["python"]),
        str(REPO_ROOT / "tools" / "aggregate_kctx8_kgen1_latency.py"),
    ]
    for summary in summaries:
        command.extend(("--summary", str(summary)))
    command.extend(
        (
            "--output-json",
            str(output_root / "summary_final.json"),
            "--output-md",
            str(output_root / "summary_final.md"),
        )
    )
    return CommandSpec("aggregate", tuple(command), {})


def ensure_fresh_output(output_dir: Path, *, resume: bool, dry_run: bool) -> None:
    if dry_run or resume or not output_dir.exists():
        return
    occupied = [output_dir / stage for stage in STAGE_ORDER if (output_dir / stage).exists()]
    if occupied:
        raise RuntimeError(
            f"measured output already exists under {output_dir}; use --resume or a new output_root"
        )


def completed_run(output_dir: Path, *, expected_queries: int) -> bool:
    """Return whether a learned-method output already has full query coverage."""

    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    query_indices = manifest.get("identity", {}).get("query_indices", ())
    if len(query_indices) != expected_queries:
        return False
    query_dir = output_dir / "queries"
    return all(
        (query_dir / f"query_{int(query_index):05d}.json").is_file()
        for query_index in query_indices
    )


def execute(spec: CommandSpec, *, output_dir: Path, dry_run: bool) -> dict:
    rendered = shlex.join(spec.command)
    print(f"[{spec.name}] {rendered}", flush=True)
    if dry_run:
        return {"name": spec.name, "status": "dry_run", "command": list(spec.command)}
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    environment = os.environ.copy()
    environment.update(spec.environment)
    started = time.perf_counter()
    with log_path.open("a") as log:
        log.write(f"COMMAND {rendered}\n")
        log.flush()
        process = subprocess.run(
            spec.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"{spec.name} failed with exit {process.returncode}; see {log_path}")
    return {
        "name": spec.name,
        "status": "completed",
        "wall_seconds": seconds,
        "log": str(log_path),
        "command": list(spec.command),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--warmup-query-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = validate_config(config)
    repeat_count = int(
        args.repeat_count if args.repeat_count is not None else config.get("repeat_count", 1)
    )
    warmup_count = int(
        args.warmup_query_count
        if args.warmup_query_count is not None
        else config.get("warmup_query_count", 0)
    )
    if repeat_count <= 0 or warmup_count < 0:
        raise ValueError("repeat count must be positive and warmup count nonnegative")

    source_selection = load_selection(paths["selection"])
    output_root = paths["output_root"]
    outcomes = []
    repeat_summaries = []
    for repeat in range(1, repeat_count + 1):
        repeat_dir = output_root / f"repeat_{repeat:03d}"
        ensure_fresh_output(repeat_dir, resume=args.resume, dry_run=args.dry_run)
        if args.stage in ("run", "all"):
            for spec in build_run_commands(
                config,
                paths,
                selection=paths["selection"],
                output_dir=repeat_dir,
                warmup_query_count=warmup_count,
            ):
                if args.resume and completed_run(
                    repeat_dir / spec.name,
                    expected_queries=int(source_selection["query_count"]),
                ):
                    print(f"[{spec.name}] skip completed 128-query run", flush=True)
                    outcomes.append({"name": spec.name, "status": "skipped_completed"})
                    continue
                outcomes.append(execute(spec, output_dir=repeat_dir, dry_run=args.dry_run))
        if args.stage in ("summarize", "all"):
            summary_spec = build_summary_command(
                paths,
                selection=paths["selection"],
                output_dir=repeat_dir,
            )
            outcomes.append(execute(summary_spec, output_dir=repeat_dir, dry_run=args.dry_run))
            repeat_summaries.append(str(repeat_dir / "summary.json"))

    final_summary = None
    if args.stage in ("summarize", "all"):
        aggregate_spec = build_aggregate_command(
            paths,
            summaries=[Path(path) for path in repeat_summaries],
            output_root=output_root,
        )
        outcomes.append(execute(aggregate_spec, output_dir=output_root, dry_run=args.dry_run))
        final_summary = str(output_root / "summary_final.json")

    manifest_body = {
        "schema_version": 2,
        "program": str(Path(__file__).resolve()),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "selection": str(paths["selection"]),
        "selection_sha256": source_selection["sha256"],
        "query_count": int(source_selection["query_count"]),
        "method_order": list(METHOD_ORDER),
        "execution_stage_order": list(STAGE_ORDER),
        "repeat_count": repeat_count,
        "warmup_query_count": warmup_count,
        "latency_protocol": {
            "context_count": 8,
            "generated_rirs_per_candidate": 1,
            "timing_boundary": (
                "conditioning and acoustic forward plus AGREE/OMP localization scoring "
                "and candidate selection"
            ),
            "fem_source": (
                "112 reused observed runtime_seconds.total values plus 16 measured "
                "strict-failure detection and random-candidate fallback values"
            ),
        },
        "stage": args.stage,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "repeat_summaries": repeat_summaries,
        "final_summary": final_summary,
        "outcomes": outcomes,
    }
    manifest = {**manifest_body, "sha256": canonical_sha256(manifest_body)}
    if not args.dry_run:
        atomic_json(output_root / "benchmark_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
