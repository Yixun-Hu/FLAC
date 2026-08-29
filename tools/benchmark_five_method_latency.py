#!/usr/bin/env python3
"""Run and summarize the five localization methods through one audited entrypoint.

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
    "fem_agree",
)
STAGE_ORDER = (*METHOD_ORDER[:-1], "fem_forward", "fem_agree")


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
    if config.get("schema_version") != 1:
        raise ValueError("latency benchmark config must use schema_version=1")
    required = (
        "python",
        "selection",
        "context_manifest",
        "geometry_audit",
        "dataset_root",
        "output_root",
        "agree_checkpoint",
        "mkl_runtime",
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
        "mkl_runtime": resolve_path(config["mkl_runtime"]),
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
        elif not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    try:
        paths["output_root"].relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("output_root must remain inside the localization repository") from error
    load_selection(paths["selection"])
    return paths


def make_warmup_selection(source: dict, count: int) -> dict:
    if not 0 < count <= int(source["query_count"]):
        raise ValueError("warmup query count must lie inside the frozen selection")
    payload = {key: value for key, value in source.items() if key != "sha256"}
    payload["records"] = list(source["records"][:count])
    payload["query_count"] = count
    payload["room_count"] = len({record["room"] for record in payload["records"]})
    payload["candidate_query_pairs"] = sum(
        int(record["candidate_count"]) for record in payload["records"]
    )
    payload["parent_selection_sha256"] = source["sha256"]
    payload["purpose"] = "latency_warmup_prefix"
    payload["sha256"] = canonical_sha256(payload)
    return payload


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
) -> list[CommandSpec]:
    python = str(paths["python"])
    device = str(config.get("device", "cuda:0"))
    gpu = str(config.get("cuda_visible_devices", "0"))
    candidate_batch_size = int(config.get("candidate_batch_size", 64))
    agree_batch_size = int(config.get("agree_candidate_batch_size", 32))
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
    )
    commands.append(CommandSpec("few_shot_rir", few_command, gpu_environment))

    fem_forward = output_dir / "fem_forward"
    cpu_environment = {
        "MKL_RT": str(paths["mkl_runtime"]),
        "MPLCONFIGDIR": str(config.get("mplconfigdir", "/tmp/matplotlib-five-method-latency")),
    }
    fem_command = (
        python,
        str(REPO_ROOT / "tools" / "run_depth_aabb_matched_pilot.py"),
        "--selection",
        str(selection),
        "--context-manifest",
        str(paths["context_manifest"]),
        "--geometry-audit",
        str(paths["geometry_audit"]),
        "--dataset-root",
        str(paths["dataset_root"]),
        "--output-dir",
        str(fem_forward),
        "--workers",
        str(int(config.get("fem_workers", 1))),
        "--solver-threads",
        str(int(config.get("fem_solver_threads", 24))),
        "--mkl-runtime",
        str(paths["mkl_runtime"]),
    )
    commands.append(CommandSpec("fem_forward", fem_command, cpu_environment))

    fem_agree_command = (
        python,
        str(REPO_ROOT / "tools" / "run_depth_aabb_fem_agree.py"),
        "--stage",
        "all",
        "--selection",
        str(selection),
        "--context-manifest",
        str(paths["context_manifest"]),
        "--geometry-audit",
        str(paths["geometry_audit"]),
        "--dataset-root",
        str(paths["dataset_root"]),
        "--source-result-dir",
        str(fem_forward),
        "--agree-ckpt",
        str(paths["agree_checkpoint"]),
        "--output-dir",
        str(output_dir / "fem_agree"),
        "--device",
        device,
        "--candidate-batch-size",
        str(agree_batch_size),
        "--score-seed",
        str(score_seed),
        "--tau",
        str(tau),
        "--solver-backend",
        "mkl_pardiso",
        "--solver-threads",
        str(int(config.get("fem_solver_threads", 24))),
        "--mkl-runtime",
        str(paths["mkl_runtime"]),
    )
    commands.append(
        CommandSpec("fem_agree", fem_agree_command, {**gpu_environment, **cpu_environment})
    )
    if tuple(spec.name for spec in commands) != STAGE_ORDER:
        raise RuntimeError("internal method ordering changed")
    return commands


def build_summary_command(
    paths: dict[str, Path], *, selection: Path, output_dir: Path
) -> CommandSpec:
    command = (
        str(paths["python"]),
        str(REPO_ROOT / "tools" / "summarize_five_method_latency.py"),
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
        "--fem-omp-dir",
        str(output_dir / "fem_forward"),
        "--fem-agree-dir",
        str(output_dir / "fem_agree" / "results"),
        "--output-json",
        str(output_dir / "summary.json"),
        "--output-md",
        str(output_dir / "summary.md"),
    )
    return CommandSpec("summarize", command, {})


def ensure_fresh_output(output_dir: Path, *, resume: bool, dry_run: bool) -> None:
    if dry_run or resume or not output_dir.exists():
        return
    occupied = [output_dir / stage for stage in STAGE_ORDER if (output_dir / stage).exists()]
    if occupied:
        raise RuntimeError(
            f"measured output already exists under {output_dir}; use --resume or a new output_root"
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
    if args.stage in ("run", "all") and warmup_count:
        warmup_selection = make_warmup_selection(source_selection, warmup_count)
        warmup_path = output_root / "warmup_selection.json"
        if not args.dry_run:
            atomic_json(warmup_path, warmup_selection)
        ensure_fresh_output(output_root / "warmup", resume=args.resume, dry_run=args.dry_run)
        for spec in build_run_commands(
            config,
            paths,
            selection=warmup_path,
            output_dir=output_root / "warmup",
        ):
            outcomes.append(execute(spec, output_dir=output_root / "warmup", dry_run=args.dry_run))

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
            ):
                outcomes.append(execute(spec, output_dir=repeat_dir, dry_run=args.dry_run))
        if args.stage in ("summarize", "all"):
            summary_spec = build_summary_command(
                paths, selection=paths["selection"], output_dir=repeat_dir
            )
            outcomes.append(execute(summary_spec, output_dir=repeat_dir, dry_run=args.dry_run))
            repeat_summaries.append(str(repeat_dir / "summary.json"))

    manifest_body = {
        "schema_version": 1,
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
        "stage": args.stage,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "repeat_summaries": repeat_summaries,
        "outcomes": outcomes,
    }
    manifest = {**manifest_body, "sha256": canonical_sha256(manifest_body)}
    if not args.dry_run:
        atomic_json(output_root / "benchmark_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
