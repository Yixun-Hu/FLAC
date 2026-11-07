#!/usr/bin/env python3
"""exp_37 AGREE selector latency over an ARBITRARY query / candidate-count manifest (plan_loc_efficiency.md §10).

The bundle's ``benchmark_localization_selector_latency.py`` hard-codes the 128 / 112 / 97 FEM accounting; this tool
keeps ITS timed function (``benchmark_agree_query``, imported unchanged: observed encode, zeros ``[b, 1, 10240]``
generated batches of ``--candidate-batch-size``, similarity, concatenation + argmax) and its protocol (one
unreported warm-up batch, ``--repeat-count`` repeats per query, per-field medians; ``scoring_total_seconds`` is the
selector term of L_q) for any hashed manifest whose records carry ``index / query_id / room / candidate_count``:
the 128 latency selection, the 64-query pilot, the 0.75 m pilot, or the nested 1.0 m subset counts. The selector's
own peak memory (allocated / reserved, absolute) is recorded per query, outside the timers. A CUDA run is bound
before its measurement (``make_binding``) and attested by the run-bound co-tenancy monitor in ``<output>.cotenancy``
(Codex code review 1, BLOCKING 3; judged by ``exp37_latency.selector_cotenancy``).
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.engine import encode_audio_features, load_agree_retrieval, load_frozen_query  # noqa: E402
from src.localization.exp37_cotenancy import Monitor  # noqa: E402
from src.localization.pilot import canonical_sha256  # noqa: E402
from src.localization.run_provenance import host_snapshot, process_start_ticks, producer_identity  # noqa: E402
from tools.benchmark_localization_selector_latency import benchmark_agree_query  # noqa: E402

RECORD_TYPE = "exp37_agree_selector_latency"
TIMED_FUNCTION = "tools/benchmark_localization_selector_latency.py::benchmark_agree_query"


def _hashed(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return payload


def load_selection(path: Path) -> dict:
    """A hashed manifest whose records name each query and its candidate count (no duplicate index)."""
    selection = _hashed(path)
    records = selection["records"]
    indices = [int(r["index"]) for r in records]
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"{path}: duplicate query index")
    if int(selection.get("query_count", len(records))) != len(records):
        raise RuntimeError(f"{path}: query_count does not match its records")
    if any(int(r["candidate_count"]) <= 0 for r in records):
        raise RuntimeError(f"{path}: every candidate_count must be positive")
    return selection


@torch.inference_mode()
def measure_selector(selection: dict, context_records: dict, dataset_root: Path, retrieval, *, device: str,
                     batch_size: int = 64, repeat_count: int = 3, load_query=load_frozen_query) -> list[dict]:
    cuda = torch.device(device).type == "cuda"
    warmup = torch.zeros((batch_size, 1, 10240), device=device)       # one unreported warm-up, as the bundle
    encode_audio_features(retrieval, warmup)
    if cuda:
        torch.cuda.synchronize(device)
    rows = []
    for selected in sorted(selection["records"], key=lambda r: int(r["index"])):
        index = int(selected["index"])
        record = context_records.get(index)
        if record is None or record["query_id"] != selected["query_id"]:
            raise RuntimeError(f"query identity mismatch at {index}")
        observed, _metadata = load_query(record, dataset_root)       # input loading: outside every timer
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        agree = benchmark_agree_query(retrieval, observed, candidate_count=int(selected["candidate_count"]),
                                      batch_size=batch_size, device=device, repeat_count=repeat_count)
        rows.append({"query_index": index, "query_id": selected["query_id"], "room": selected["room"],
                     "candidate_count": int(selected["candidate_count"]),
                     "candidate_indices_sha256": selected.get("candidate_indices_sha256"), "agree": agree,
                     "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if cuda else None,
                     "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if cuda else None})
        print(f"[{len(rows)}/{len(selection['records'])}] selector latency query {index}", flush=True)
    return rows


def make_binding(*, selection: dict, context_sha256: str, agree_sha256: str, device: str, batch_size: int,
                 repeat_count: int, output: Path, producer: dict | None = None) -> dict:
    """BLOCKING 3 / review 2 HIGH 1: what the co-tenancy monitor binds to, fixed BEFORE the measurement -- the
    exp37_latency.SELECTOR_BOUND fields, the output, and the producer (pid + /proc start time, host, card)."""
    import socket
    producer = producer or {"pid": os.getpid(), "pid_start_ticks": process_start_ticks(),
                            "hostname": socket.gethostname(), "card_uuid": None}
    body = {"record_type": "exp37_selector_binding", "selection_sha256": selection["sha256"],
            "context_manifest_sha256": context_sha256, "agree_checkpoint_file_sha256": agree_sha256, "device": device,
            "candidate_batch_size": int(batch_size), "repeat_count": int(repeat_count), "output": str(output),
            **producer, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return {**body, "sha256": canonical_sha256(body)}


def build_payload(*, selection: dict, context_sha256: str, agree_path: Path, agree_sha256: str, device: str,
                  batch_size: int, repeat_count: int, rows: list[dict], host: dict, binding: dict | None = None) -> dict:
    body = {"schema_version": 1, "record_type": RECORD_TYPE, "timed_function": TIMED_FUNCTION,
            "selection_sha256": selection["sha256"], "context_manifest_sha256": context_sha256,
            "agree_checkpoint": str(agree_path), "agree_checkpoint_file_sha256": agree_sha256, "device": device,
            "candidate_batch_size": int(batch_size), "repeat_count": int(repeat_count), "query_count": len(rows),
            "host": host, "binding": binding, "queries": rows}
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--agree-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_batch_size <= 0 or args.repeat_count <= 0:
        raise ValueError("batch size and repeat count must be positive")
    if args.output.exists():
        raise RuntimeError(f"{args.output} exists (selector vectors are written once)")
    attestation = Path(f"{args.output}.cotenancy")
    if attestation.exists():
        raise RuntimeError(f"{attestation} exists (a selector vector's cotenancy directory is written once)")
    selection = load_selection(args.selection)
    context = _hashed(args.context_manifest)
    if selection.get("context_manifest_sha256", context["sha256"]) != context["sha256"]:
        raise RuntimeError("the selection and the context manifest do not match")
    agree_path = args.agree_checkpoint.resolve()
    agree_sha256 = hashlib.sha256(agree_path.read_bytes()).hexdigest()
    cuda = torch.device(args.device).type == "cuda"     # a CPU smoke is never a timing source (no attestation)
    binding = make_binding(selection=selection, context_sha256=context["sha256"], agree_sha256=agree_sha256,
                           device=args.device, batch_size=args.candidate_batch_size, repeat_count=args.repeat_count,
                           output=args.output, producer=producer_identity(args.device) if cuda else None)
    monitor = None
    if cuda:
        monitor = Monitor(attestation, binding_sha256=binding["sha256"], card_uuid=binding["card_uuid"]).start()
    retrieval = load_agree_retrieval(agree_path, args.device)
    if monitor is not None:
        monitor.marker("gpu_resident")
        monitor.marker("timed_start")
    rows = measure_selector(selection, {int(r["index"]): r for r in context["records"]}, args.dataset_root.resolve(),
                            retrieval, device=args.device, batch_size=args.candidate_batch_size,
                            repeat_count=args.repeat_count)
    if monitor is not None:
        monitor.marker("timed_end")
        monitor.stop()
    payload = build_payload(selection=selection, context_sha256=context["sha256"], agree_path=agree_path,
                            agree_sha256=agree_sha256, device=args.device, batch_size=args.candidate_batch_size,
                            repeat_count=args.repeat_count, rows=rows, host=host_snapshot(args.device),
                            binding=binding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "sha256": payload["sha256"], "queries": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
