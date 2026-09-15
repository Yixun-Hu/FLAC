#!/usr/bin/env python3
"""Audit bundled artifact counts, embedded JSON hashes, and FEM response hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


METHODS = ("vanilla_flac", "fa_bf_flac", "yawaug_flac", "few_shot_rir")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def require_count(label: str, paths: list[Path], expected: int) -> None:
    if len(paths) != expected:
        raise RuntimeError(f"{label}: expected {expected} files, found {len(paths)}")


def main() -> None:
    bundle = Path(__file__).resolve().parents[2]
    data = bundle / "data"

    checked_json = 0
    for path in sorted(data.rglob("*.json")) + sorted((bundle / "expected").glob("*.json")):
        payload = json.loads(path.read_text())
        if "sha256" in payload:
            if payload["sha256"] != canonical_hash(payload):
                raise RuntimeError(f"invalid embedded JSON hash: {path}")
            checked_json += 1

    primary = sorted((data / "fem" / "primary_97").glob("query_*_depth_aabb_result.json"))
    oversized = sorted((data / "fem" / "oversized_9").glob("query_*_depth_aabb_result.json"))
    require_count("FEM primary", primary, 97)
    require_count("FEM oversized", oversized, 9)

    response_dir = data / "selector" / "fem_response_cache_97"
    response_files = sorted(response_dir.glob("query_*_response.npz"))
    response_audits = sorted(response_dir.glob("query_*.json"))
    require_count("FEM response NPZ", response_files, 97)
    require_count("FEM response audit JSON", response_audits, 97)
    for audit_path in response_audits:
        audit = json.loads(audit_path.read_text())
        response_path = response_dir / audit["response_file"]
        if digest(response_path) != audit["response_file_sha256"]:
            raise RuntimeError(f"FEM response hash mismatch: {response_path}")

    for ordinal in (1,):
        repeat = data / "repeats" / f"repeat_{ordinal:03d}"
        for method in METHODS:
            query_dir = repeat / method / "queries"
            require_count(
                f"repeat_{ordinal:03d}/{method} JSON",
                sorted(query_dir.glob("query_*.json")),
                128,
            )
            require_count(
                f"repeat_{ordinal:03d}/{method} NPZ",
                sorted(query_dir.glob("query_*.npz")),
                128,
            )
            if not (repeat / method / "run_manifest.json").is_file():
                raise RuntimeError(f"missing run manifest: repeat_{ordinal:03d}/{method}")

    selection = json.loads(
        (data / "selection" / "frozen_16room_128.json").read_text()
    )
    strict = json.loads(
        (data / "selection" / "depth_aabb_matched_16room_112.json").read_text()
    )
    selector = json.loads((data / "selector" / "selector_latency_128.json").read_text())
    external = json.loads(
        (data / "fem" / "external_server_6query_runtime_recovery.json").read_text()
    )
    fallback = json.loads(
        (data / "fem" / "fem_random_fallback_latency_16.json").read_text()
    )
    observed = {
        int(json.loads(path.read_text())["query_index"])
        for path in primary + oversized
    } | {int(row["query_index"]) for row in external["queries"]}
    full = {int(row["index"]) for row in selection["records"]}
    strict_indices = {int(row["index"]) for row in strict["records"]}
    fallback_indices = {int(row["query_index"]) for row in fallback["queries"]}
    if not (
        len(full) == 128
        and len(strict_indices) == 112
        and len(selector["queries"]) == 128
        and len(external["queries"]) == 6
        and len(fallback_indices) == 16
        and observed == strict_indices
        and fallback_indices == full - strict_indices
    ):
        raise RuntimeError("selection/FEM/selector coverage sets do not agree")

    print(
        "PASS: bundle audit complete: "
        f"{checked_json} embedded JSON hashes, 1 repeat × 4 methods × 128 queries, "
        "97+9+6 FEM successes, 16 fallbacks, and 97 FEM response caches."
    )


if __name__ == "__main__":
    main()
