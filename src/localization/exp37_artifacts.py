"""exp_37 untimed artifact executor and artifact replay (announcement 08; Codex code review 1, BLOCKING 1).

``execute_query_artifacts`` computes exactly what the timed headline computes -- the same context and candidate
caches, full-grid seeds, batches, ``generate_rir_batch`` and AGREE scoring -- but never times, and dumps each query's
scored waveforms (the exact ``prepare_generated_audio`` output) through a ``WaveformDump``. Accuracy-only runs use it
inline. Timing passes never dump; their ARTIFACT REPLAY runs this executor with ``reference`` = the timing run's stored
similarities and refuses unless the regenerated similarities are bitwise identical. A coarse-to-fine pass is replayed
with ``c2f_replay_plan`` (its coarse batches, then its new refined batches; output in union order).
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS, FA_CONTEXT_CONDITIONING_IDS, FA_DYNAMIC_CONDITIONING_IDS, SOURCE_CONDITIONING_IDS,
    candidate_metadata, candidate_seed, encode_audio_features, filter_frozen_query_candidates, generate_rir_batch,
    load_frozen_query,
)
from src.localization.exp37_grid import CHEBYSHEV_RADIUS_M, TOLERANCE_M, nested_subset, query_scores, top_k
from src.localization.pilot import canonical_sha256
from src.localization.runner import _build_query_result, _cache_branch, _repeat_branch, validate_frame_angles

REPLAY_EXEMPT = ("synchronize_timing", "warmup_query_count", "measure_core_forward", "stage_split")
_MISSING = object()


def check_dump_request(output_dir, dump_waveforms, replay_of, *, timed: bool, exp37: bool, root) -> None:
    """The waveform refusals of a run request, before anything is written."""
    if replay_of is not None and timed:
        raise ValueError("an artifact replay is untimed: no --measure-core-forward / --synchronize-latency / "
                         "--stage-split / warm-up")
    if dump_waveforms is not None and timed:
        raise ValueError("timing passes never dump waveforms: replay them untimed with --replay-of")
    if replay_of is not None and dump_waveforms is None:
        raise ValueError("an artifact replay dumps its waveforms: --dump-waveforms is required")
    if exp37 and not timed and dump_waveforms is None:
        raise ValueError("an accuracy-only exp_37 run must dump its scored waveforms: --dump-waveforms <root>/<run stem>")
    if dump_waveforms is None:
        return
    if (Path(output_dir) / "run_manifest.json").exists():
        raise RuntimeError(f"a waveform-dump run is never resumed: {output_dir}/run_manifest.json exists")
    dump_dir = Path(dump_waveforms)
    if dump_dir.parent.resolve() != Path(root).resolve() or dump_dir.name != Path(output_dir).name:
        raise ValueError(f"--dump-waveforms must be {root}/<run stem>: a directory named after the output directory "
                         f"({Path(output_dir).name!r}) directly under the registered root")
    if dump_dir.exists():
        raise RuntimeError(f"{dump_dir} exists (waveform dumps are write-once)")


def _verified(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"replay: {path} fails its own sha256")
    return payload


def load_replay_source(run_dir, identity: dict) -> dict:
    """The hashed manifest of the timing run a replay reproduces: a COMPLETE timing pass whose identity equals the
    replay's outside the timing flags; RuntimeError otherwise (before the replay writes anything)."""
    run_dir = Path(run_dir)
    manifest = _verified(run_dir / "run_manifest.json")
    original = manifest["identity"]
    if not (original.get("measure_core_forward") and original.get("synchronize_timing")):
        raise RuntimeError(f"replay: {run_dir} is not a timing pass (an accuracy-only run dumps inline)")
    keys = (set(original) | set(identity)) - set(REPLAY_EXEMPT)
    differ = sorted(k for k in keys if original.get(k, _MISSING) != identity.get(k, _MISSING))
    if differ:
        raise RuntimeError(f"replay: the identity differs from the timing run's in {differ}")
    for index in identity["query_indices"]:
        path = run_dir / "queries" / f"query_{int(index):05d}.json"
        if not path.is_file() or _verified(path).get("run_manifest_sha256") != manifest["sha256"]:
            raise RuntimeError(f"replay: the timing run is incomplete or foreign at query {index}")
    return manifest


def replay_inputs(run_dir, manifest: dict, index: int):
    """(reference similarities, c2f batch plan or None) of one query of the replayed timing run, hash-verified."""
    run_dir = Path(run_dir)
    query = _verified(run_dir / "queries" / f"query_{int(index):05d}.json")
    arrays = run_dir / query["arrays_file"]
    if query.get("run_manifest_sha256") != manifest["sha256"] or \
            hashlib.sha256(arrays.read_bytes()).hexdigest() != query.get("arrays_sha256"):
        raise RuntimeError(f"replay: query {index} of {run_dir} is not that run's or its arrays changed")
    with np.load(arrays) as stored:
        similarities = np.array(stored["similarities"])
    return similarities, (c2f_replay_plan(query["c2f"]) if "c2f" in query else None)


def c2f_replay_plan(c2f: dict):
    """The batch plan of a recorded c2f pass: its coarse (nested) batches, then its new refined batches."""
    union = np.asarray(c2f["union_indices"])

    def plan(full, batch_size):
        coarse = nested_subset(full)
        new = np.setdiff1d(union, coarse)
        batches = [coarse[i:i + batch_size] for i in range(0, len(coarse), batch_size)]
        return batches + [new[i:i + batch_size] for i in range(0, len(new), batch_size)], union
    return plan


@torch.inference_mode()
def execute_query_artifacts(*, module, retrieval, selected, record, audit, room_base, dataset_root, device, cond_method,
                            candidate_batch_size, sample_seed, tau, run_sha256, dump, score_sample_counts=(1,),
                            synchronize_timing=False, measure_core_forward=False, context_count=8,
                            frame_angles=DEFAULT_FRAME_ANGLES, candidate_subset=None, reference=None, plan=None,
                            c2f_top_k=None):
    if measure_core_forward or synchronize_timing:
        raise ValueError("the artifact executor is untimed (timing passes never dump; replay them instead)")
    if tuple(score_sample_counts) != (1,):
        raise ValueError("exp_37 artifacts are K_gen = 1")
    if c2f_top_k is not None and (plan is not None or candidate_subset is not None or int(c2f_top_k) <= 0):
        raise ValueError("an adaptive c2f run has no plan / candidate subset and a positive top-k")
    validate_frame_angles(cond_method, frame_angles)
    device_type, started = torch.device(device).type, time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    full = filter_frozen_query_candidates(record, audit, room_base)
    if len(full) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    def chunks(indices):
        return [indices[i:i + candidate_batch_size] for i in range(0, len(indices), candidate_batch_size)]
    if plan is not None:
        batches, output = plan(full, candidate_batch_size)
    elif c2f_top_k is not None:                   # adaptive c2f: the coarse stage first, the rest after selection
        batches, output = chunks(nested_subset(full)), None
    else:
        output = np.asarray(candidate_subset(full)) if candidate_subset is not None else np.arange(len(full))
        batches = chunks(output)
    observation_features = encode_audio_features(retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32))
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [metadata], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    scores, waves = {}, {}

    def run(batch_list):
        for batch in batch_list:
            items = candidate_metadata(metadata, full[batch], record["receiver_global"])
            with torch.amp.autocast(device_type):
                source = _cache_branch(module, items, SOURCE_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
                dynamic = (_cache_branch(module, items, FA_DYNAMIC_CONDITIONING_IDS, device, cond_method,
                                         frame_angles=frame_angles) if cond_method == "fa_invariant" else None)
            seeds = [candidate_seed(sample_seed, int(record["index"]), int(i), 0) for i in batch]
            generated = generate_rir_batch(module, _repeat_branch(source, 1), context, seeds,
                                           dynamic_branch=_repeat_branch(dynamic, 1) if dynamic else None)
            batch_scores = (encode_audio_features(retrieval, generated) @ observation_features.T).reshape(-1).cpu()
            generated = generated.cpu()
            for row, index in enumerate(batch):
                scores[int(index)], waves[int(index)] = batch_scores[row], generated[row]
    run(batches)
    c2f_record = None
    if c2f_top_k is not None:                     # = exp37_c2f.execute_query_c2f's selection, untimed, no repeats
        coarse = nested_subset(full)
        coarse_scores = np.full(len(full), np.nan)
        coarse_scores[coarse] = query_scores(torch.stack([scores[int(i)] for i in coarse]).float().numpy()[:, None], tau)
        top = top_k(coarse_scores, coarse, int(c2f_top_k))
        near = np.zeros(len(full), dtype=bool)
        for index in top:
            near |= np.abs(full.astype(np.float64) - full[index].astype(np.float64)).max(axis=1) \
                <= CHEBYSHEV_RADIUS_M + TOLERANCE_M
        refined = np.flatnonzero(near)
        new = np.setdiff1d(refined, coarse)
        run(chunks(new))
        output = np.union1d(coarse, refined)
        union_scores = torch.stack([scores[int(i)] for i in output]).float().numpy()[:, None]
        c2f_record = {"full_candidate_count": len(full), "coarse_count": len(coarse), "top_indices": top.tolist(),
                      "refined_count": len(refined), "new_count": len(new), "union_indices": output.tolist(),
                      "generated_count": len(coarse) + len(new),
                      "winner_full_index": int(output[int(np.argmax(query_scores(union_scores, tau)))]),
                      "union_indices_sha256": hashlib.sha256(output.astype("<u4").tobytes()).hexdigest(),
                      "top_k": int(c2f_top_k), "chebyshev_radius_m": CHEBYSHEV_RADIUS_M}
    similarities = torch.stack([scores[int(i)] for i in output]).float().reshape(-1, 1)
    if reference is not None:
        expected = np.asarray(reference(int(record["index"])))
        if expected.dtype != np.float32 or not np.array_equal(similarities.numpy(), expected):
            delta = float(np.max(np.abs(similarities.numpy().astype(np.float64) - expected.astype(np.float64)))) \
                if expected.shape == tuple(similarities.shape) else float("nan")
            raise RuntimeError(f"replay: query {record['index']} similarities are not bitwise the timing run's "
                               f"(max |delta| {delta:.3e})")
    entry = dump.write(int(record["index"]), torch.stack([waves[int(i)] for i in output]).numpy(), output)
    peak = int(torch.cuda.max_memory_allocated(device)) if device_type == "cuda" else 0
    result = _build_query_result(selected=selected, record=record, candidates=full[output], similarities=similarities,
                                 run_sha256=run_sha256, elapsed_seconds=time.perf_counter() - started,
                                 peak_memory_bytes=peak, tau=tau, random_seed=sample_seed, score_sample_counts=(1,))
    if candidate_subset is not None or plan is not None or c2f_record is not None:
        active = hashlib.sha256(np.asarray(output).astype("<u4").tobytes()).hexdigest()
        result["parent_candidate_indices_sha256"] = result["candidate_indices_sha256"]
        result["candidate_indices_sha256"] = active
        if candidate_subset is not None:
            result["candidate_subset"] = {"name": candidate_subset.__name__, "full_candidate_count": len(full),
                                          "indices_sha256": active}
    if c2f_record is not None:
        result["c2f"] = c2f_record
    result["waveforms"] = {"dump_dir": str(dump.dir), **entry}
    return result, full[output], similarities.numpy()
