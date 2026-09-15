"""Sparse real-RIR AGREE upper-bound diagnostics for exp_09."""

from __future__ import annotations

import json
import hashlib
import math
import re
import copy
from pathlib import Path

import numpy as np


RIR_NAME = re.compile(r"^S(?P<source>\d+)_R(?P<receiver>\d+)_hybrid_IR\.wav$")


def deterministic_agree_seed(base_seed: int, query_index: int, role: str) -> int:
    """Derive an order-independent seed for AGREE's stochastic VAE encoder."""

    if not role:
        raise ValueError("AGREE seed role must be nonempty")
    payload = f"{int(base_seed)}:{int(query_index)}:{role}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (
        2**63 - 1
    )


def _coordinates(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise RuntimeError(f"{label} must be a finite three-dimensional coordinate")
    return result


def discover_real_rir_bank(record: dict, dataset_root: Path | str) -> dict:
    """Return every real source RIR sharing the query's receiver."""

    root = Path(dataset_root)
    query_relpath = Path(record["query_id"])
    match = RIR_NAME.fullmatch(query_relpath.name)
    if match is None:
        raise RuntimeError(f"invalid AcousticRooms RIR name: {query_relpath.name}")
    target_source = int(match.group("source"))
    receiver_id = int(match.group("receiver"))
    query_path = root / query_relpath
    if not query_path.is_file():
        raise FileNotFoundError(query_path)
    room_audio = query_path.parent
    metadata_dir = root / "metadata" / record["scene"] / record["room"]
    expected_receiver = _coordinates(record["receiver_global"], "query receiver")
    candidates = []
    for path in room_audio.glob(f"S*_R{receiver_id:03d}_hybrid_IR.wav"):
        candidate_match = RIR_NAME.fullmatch(path.name)
        if candidate_match is None or int(candidate_match.group("receiver")) != receiver_id:
            continue
        source_id = int(candidate_match.group("source"))
        metadata_path = metadata_dir / f"S00{source_id}_R00{receiver_id}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        source = _coordinates(metadata.get("src_loc"), "candidate source")
        receiver = _coordinates(metadata.get("rec_loc"), "candidate receiver")
        if not np.allclose(receiver, expected_receiver, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"candidate receiver coordinate mismatch: {path}")
        candidates.append((source_id, path, source))
    candidates.sort(key=lambda item: item[0])
    source_ids = [item[0] for item in candidates]
    if len(source_ids) < 2 or len(source_ids) != len(set(source_ids)):
        raise RuntimeError("real-RIR bank must contain at least two unique sources")
    if target_source not in source_ids:
        raise RuntimeError("real-RIR bank does not contain the target source")
    target_index = source_ids.index(target_source)
    if candidates[target_index][1].resolve() != query_path.resolve():
        raise RuntimeError("target candidate path is not the observed RIR")
    positions = np.stack([item[2] for item in candidates])
    expected_source = _coordinates(record["source_global"], "query source")
    if not np.allclose(positions[target_index], expected_source, atol=1e-6, rtol=0.0):
        raise RuntimeError("target source coordinate mismatch")
    return {
        "receiver_id": receiver_id,
        "source_ids": source_ids,
        "rir_paths": [str(item[1].relative_to(root)) for item in candidates],
        "positions_global": positions.astype(float).tolist(),
        "target_index": target_index,
    }


def summarize_oracle_scores(
    candidate_positions,
    scores,
    *,
    target_index: int,
    temperature: float = 0.1,
) -> dict:
    """Summarize real-RIR retrieval and AGREE-space ambiguity for one query."""

    positions = np.asarray(candidate_positions, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("candidate_positions must have shape [M, 3] with M >= 2")
    if values.shape != (len(positions),) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector aligned with candidates")
    if not np.isfinite(positions).all():
        raise ValueError("candidate positions must be finite")
    if target_index < 0 or target_index >= len(positions):
        raise ValueError("target_index is outside the candidate bank")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    prediction_index = int(np.argmax(values))
    negative_indices = np.delete(np.arange(len(values)), target_index)
    hardest_negative_index = int(
        negative_indices[np.argmax(values[negative_indices])]
    )
    target_position = positions[target_index]
    distances = np.linalg.norm(positions - target_position, axis=1)
    shifted = (values - values.max()) / temperature
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    normalized_entropy = entropy / math.log(len(probabilities))
    order = np.lexsort((np.arange(len(values)), -values))
    target_rank = int(np.flatnonzero(order == target_index)[0]) + 1
    localization_error = float(distances[prediction_index])
    return {
        "candidate_count": len(values),
        "prediction_index": prediction_index,
        "target_rank": target_rank,
        "hardest_negative_index": hardest_negative_index,
        "localization_error_m": localization_error,
        "success_0_5m": int(localization_error <= 0.5),
        "success_1_0m": int(localization_error <= 1.0),
        "target_score": float(values[target_index]),
        "hardest_negative_score": float(values[hardest_negative_index]),
        "target_margin": float(values[target_index] - values[hardest_negative_index]),
        "hardest_negative_distance_m": float(distances[hardest_negative_index]),
        "target_probability": float(probabilities[target_index]),
        "normalized_entropy": float(normalized_entropy),
        "probability_mass_0_5m": float(probabilities[distances <= 0.5].sum()),
        "probability_mass_1_0m": float(probabilities[distances <= 1.0].sum()),
        "probabilities": probabilities.astype(float).tolist(),
    }


def aggregate_oracle_rows(rows: list[dict]) -> dict:
    """Aggregate the fixed real-RIR diagnostic query set."""

    if not rows:
        raise ValueError("oracle rows must be nonempty")
    margins = np.asarray([item["target_margin"] for item in rows], dtype=np.float64)
    errors = np.asarray([item["localization_error_m"] for item in rows], dtype=np.float64)
    candidate_counts = np.asarray(
        [item["candidate_count"] for item in rows], dtype=np.int64
    )
    return {
        "query_count": len(rows),
        "room_count": len({(item.get("scene", ""), item["room"]) for item in rows}),
        "candidate_count_min": int(candidate_counts.min()),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_max": int(candidate_counts.max()),
        "target_recall_at_1": float(np.mean([item["target_rank"] == 1 for item in rows])),
        "mean_localization_error_m": float(errors.mean()),
        "median_localization_error_m": float(np.median(errors)),
        "success_0_5m": float(np.mean([item["success_0_5m"] for item in rows])),
        "success_1_0m": float(np.mean([item["success_1_0m"] for item in rows])),
        "mean_target_score": float(np.mean([item["target_score"] for item in rows])),
        "mean_hardest_negative_score": float(
            np.mean([item["hardest_negative_score"] for item in rows])
        ),
        "mean_target_margin": float(margins.mean()),
        "median_target_margin": float(np.median(margins)),
        "target_margin_p10": float(np.quantile(margins, 0.1)),
        "target_margin_p90": float(np.quantile(margins, 0.9)),
        "mean_target_probability": float(
            np.mean([item["target_probability"] for item in rows])
        ),
        "median_target_probability": float(
            np.median([item["target_probability"] for item in rows])
        ),
        "mean_normalized_entropy": float(
            np.mean([item["normalized_entropy"] for item in rows])
        ),
        "mean_hardest_negative_distance_m": float(
            np.mean([item["hardest_negative_distance_m"] for item in rows])
        ),
        "median_hardest_negative_distance_m": float(
            np.median([item["hardest_negative_distance_m"] for item in rows])
        ),
        "mean_probability_mass_0_5m": float(
            np.mean([item["probability_mass_0_5m"] for item in rows])
        ),
        "mean_probability_mass_1_0m": float(
            np.mean([item["probability_mass_1_0m"] for item in rows])
        ),
    }


def select_representative_cases(rows: list[dict]) -> list[dict]:
    """Select up to four deterministic, distinct real-RIR score-field examples."""

    if not rows or len({item["query_id"] for item in rows}) != len(rows):
        raise ValueError("case selection requires unique queries")
    median_margin = float(np.median([item["target_margin"] for item in rows]))
    remaining = list(rows)
    policies = (
        ("sharp", lambda item: (-item["target_margin"], item["query_id"])),
        ("ambiguous", lambda item: (item["target_margin"], item["query_id"])),
        ("diffuse", lambda item: (-item["normalized_entropy"], item["query_id"])),
        (
            "typical",
            lambda item: (abs(item["target_margin"] - median_margin), item["query_id"]),
        ),
    )
    selected = []
    for category, key in policies[: min(4, len(rows))]:
        choice = min(remaining, key=key)
        selected.append({"category": category, **choice})
        remaining = [item for item in remaining if item["query_id"] != choice["query_id"]]
    return selected


def resolve_oracle_records(
    labeled_pilots: list[tuple[str, dict]], context_manifest: dict
) -> list[dict]:
    """Join multiple non-overlapping pilots to their frozen query metadata."""

    if not labeled_pilots:
        raise ValueError("at least one pilot is required")
    context_by_index = {
        int(item["index"]): item for item in context_manifest.get("records", ())
    }
    output = []
    seen_indices: set[int] = set()
    seen_query_ids: set[str] = set()
    for label, pilot in labeled_pilots:
        if pilot.get("context_manifest_sha256") != context_manifest.get("sha256"):
            raise ValueError("pilot/context manifest mismatch")
        if len(pilot.get("records", ())) != int(pilot.get("query_count", -1)):
            raise ValueError("pilot query count is inconsistent")
        for selected in pilot["records"]:
            index = int(selected["index"])
            query_id = selected["query_id"]
            if index in seen_indices or query_id in seen_query_ids:
                raise ValueError("pilot query overlap detected")
            record = context_by_index.get(index)
            if record is None or record.get("query_id") != query_id:
                raise ValueError("pilot query is absent from the frozen context manifest")
            joined = copy.deepcopy(record)
            joined["batch"] = str(label)
            joined["pilot_manifest_sha256"] = pilot["sha256"]
            output.append(joined)
            seen_indices.add(index)
            seen_query_ids.add(query_id)
    return output


def render_oracle_markdown(payload: dict) -> str:
    """Render the compact scientific report for the real-RIR upper bound."""

    summary = payload["summary"]
    lines = [
        "# Exp_09 real-RIR AGREE diagnostic upper bound",
        "",
        f"Scope: {payload['query_count']} queries / {payload['room_count']} rooms; "
        f"score K={payload['score_sample_counts']} with tau={payload['tau']:.3g}; "
        f"visualization temperature T={payload['temperature']:.3g}.",
        "",
        "> This is a sparse metadata-bank **ground-truth-RIR upper bound**. It replaces "
        "FLAC output with released real candidate RIRs at the same receiver. The observed "
        "RIR and every candidate copy are independently passed through AGREE's stochastic "
        "VAE audio encoder with fixed, recorded seeds; rank-1/error-zero is therefore not "
        "assumed.",
        "The T-scaled softmax mass is a visualization diagnostic, not a calibrated probability.",
        "",
        "| K | Target R@1 | Median / mean error | Success@0.5 / 1.0 m | Mean / median margin |",
        "|---:|---:|---:|---:|---:|",
    ]
    for count in payload["score_sample_counts"]:
        item = payload["summary_by_k"][str(count)]
        lines.append(
            f"| {count} | {item['target_recall_at_1']:.3f} | "
            f"{item['median_localization_error_m']:.3f} / {item['mean_localization_error_m']:.3f} m | "
            f"{item['success_0_5m']:.3f} / {item['success_1_0m']:.3f} | "
            f"{item['mean_target_margin']:.4f} / {item['median_target_margin']:.4f} |"
        )
    lines += [
        "",
        f"The remaining diagnostics and figures use the pre-registered primary nested K={payload['primary_score_sample_count']} score.",
        "",
        "| Primary-K diagnostic | Value |",
        "|---|---:|",
        f"| Real candidates per receiver, min / median / max | {summary['candidate_count_min']} / {summary['candidate_count_median']:.0f} / {summary['candidate_count_max']} |",
        f"| Mean target K-score | {summary['mean_target_score']:.4f} |",
        f"| Mean hardest-negative K-score | {summary['mean_hardest_negative_score']:.4f} |",
        f"| Target margin, mean / median | {summary['mean_target_margin']:.4f} / {summary['median_target_margin']:.4f} |",
        f"| Target margin, p10 / p90 | {summary['target_margin_p10']:.4f} / {summary['target_margin_p90']:.4f} |",
        f"| Diagnostic target softmax mass, mean / median | {summary['mean_target_probability']:.3f} / {summary['median_target_probability']:.3f} |",
        f"| Mean normalized entropy | {summary['mean_normalized_entropy']:.3f} |",
        f"| Hardest-negative distance, mean / median | {summary['mean_hardest_negative_distance_m']:.3f} / {summary['median_hardest_negative_distance_m']:.3f} m |",
        f"| Mean probability mass within 0.5 / 1.0 m | {summary['mean_probability_mass_0_5m']:.3f} / {summary['mean_probability_mass_1_0m']:.3f} |",
        "",
        "## Deterministic representative cases",
        "",
        "| Case | Batch | Room / target | Margin | Target mass | Entropy | Hardest-negative distance |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in payload["representative_cases"]:
        lines.append(
            f"| {item['category']} | {item['batch']} | {item['room']} / "
            f"{Path(item['query_id']).name} | {item['target_margin']:.4f} | "
            f"{item['target_probability']:.3f} | {item['normalized_entropy']:.3f} | "
            f"{item['hardest_negative_distance_m']:.3f} m |"
        )
    lines += [
        "",
        "![Aggregate ambiguity diagnostics](real_rir_oracle_summary.png)",
        "",
        "![Representative real-RIR score fields](real_rir_oracle_cases.png)",
        "",
    ]
    return "\n".join(lines)
