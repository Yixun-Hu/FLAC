"""exp_37 round B-c: the validated inputs of the B2 / B3 aggregation (plan §4, §5, §10).

- ``exp37_grid.readout_run(..., full_only=True)``: the full-set readout of a run whose grid is not the 0.5 m lattice
  (the 0.75 m runs) or that is rotated (B3 θ = 90);
- ``exp37_latency.load_arm_run`` binds a nested-1.0 m timing pass to its subset manifest (counts and the per-query
  subset sha256), checks the orbit (frame angles) of every run, and refuses a rotated run as a timing source;
- ``exp37_latency.load_c2f_run`` validates a coarse-to-fine sequential pass.
CPU, synthetic runs. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import hashlib

import numpy as np
import pytest

from src.localization import exp37_grid as grid
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

C4 = [0.0, 90.0, 180.0, 270.0]


def _pilot(counts):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c} for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def _run(tmp_path, name, arm, pilot, *, identity=None, per_query=None, counts=None):
    ident = {"conditioning_method": lat.ARM_PROTOCOL[arm]["cond"], "checkpoint_sha256": lat.ARM_PROTOCOL[arm]["ckpt"],
             "n_context": 8, "pilot_manifest_sha256": pilot["sha256"], "query_indices": [r["index"] for r in pilot["records"]],
             "frame_average_angles": C4 if arm == "BF" else None, "tau": 0.1, "score_sample_counts": [1],
             **lat.TIMING_IDENTITY, **(identity or {})}
    run = tmp_path / name
    manifest = initialize_run(run, ident)
    for record in pilot["records"]:
        n = (counts or {}).get(record["index"], record["candidate_count"])
        candidates = np.array([[0.75 * k, 0.0, 1.0] for k in range(n)], dtype=np.float32)
        sims = np.linspace(0.1, 0.9, n, dtype=np.float32).reshape(n, 1)
        metrics = {"prediction_index": n - 1, "localization_error_m": float(np.linalg.norm(candidates[-1] - [0.0, 0.0, 1.0]))}
        result = {"run_manifest_sha256": manifest["sha256"], "query_index": record["index"], "query_id": record["query_id"],
                  "room": "r", "candidate_count": n, "peak_memory_bytes": 10, "source_global": [0.0, 0.0, 1.0],
                  "metrics": {"1": metrics},
                  "latency_protocol": {"name": "fem_core_aligned_kctx8_kgen1", "context_count": 8},
                  "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": 0.2,
                                      "core_forward_total": 0.3}, **((per_query or {}).get(record["index"], {}))}
        save_query_result(run, result=result, candidates=candidates, similarities=sims)
    return run


def test_a_full_only_readout_needs_no_half_metre_lattice(tmp_path):
    pilot = _pilot({1: 5, 2: 7})
    rows = grid.readout_run(_run(tmp_path, "r075", "P1", pilot), pilot, full_only=True)
    assert set(rows) == {"full"} and [r["candidate_count"] for r in rows["full"]] == [5, 7]
    with pytest.raises(ValueError, match="0.5 m lattice"):
        grid.readout_run(_run(tmp_path, "r075b", "P1", pilot), pilot)                 # the nested readout needs it


def _subset(pilot, counts):
    records = [{"index": r["index"], "query_id": r["query_id"], "room": r["room"], "candidate_count": counts[r["index"]],
                "candidate_indices_sha256": f"s{r['index']}"} for r in pilot["records"]]
    body = {"subset": "nested_1m_integer", "parent_pilot_sha256": pilot["sha256"], "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def test_a_nested_timing_pass_is_bound_to_its_subset_manifest(tmp_path):
    pilot = _pilot({1: 9, 2: 12})
    subset = _subset(pilot, {1: 2, 2: 3})
    per_query = {i: {"candidate_subset": {"name": "nested_subset", "indices_sha256": f"s{i}"}} for i in (1, 2)}
    run = _run(tmp_path, "nested", "P1", pilot, identity={"candidate_subset": "nested_1m"}, per_query=per_query,
               counts={1: 2, 2: 3})
    rows = lat.load_arm_run(run, arm="P1", selection=pilot, context_count=8, subset_manifest=subset)
    assert rows[2]["candidate_count"] == 3
    with pytest.raises(RuntimeError, match="candidate_subset"):
        lat.load_arm_run(run, arm="P1", selection=pilot, context_count=8)            # not a headline run
    bad = _run(tmp_path, "nested_bad", "P1", pilot, identity={"candidate_subset": "nested_1m"}, counts={1: 2, 2: 3},
               per_query={1: per_query[1], 2: {"candidate_subset": {"indices_sha256": "other"}}})
    with pytest.raises(RuntimeError, match="subset"):
        lat.load_arm_run(bad, arm="P1", selection=pilot, context_count=8, subset_manifest=subset)


def test_the_orbit_is_checked_and_a_rotated_run_is_never_a_timing_source(tmp_path):
    pilot = _pilot({1: 4})
    orbit = _run(tmp_path, "orbit0", "BF", pilot, identity={"frame_average_angles": [0.0]})
    assert lat.load_arm_run(orbit, arm="BF", selection=pilot, context_count=8, frame_angles=(0.0,))
    with pytest.raises(RuntimeError, match="frame_average_angles"):
        lat.load_arm_run(orbit, arm="BF", selection=pilot, context_count=8)            # the default orbit is C4
    turned = _run(tmp_path, "turned", "BF", pilot, identity={"rotate_deg": 90.0})
    with pytest.raises(RuntimeError, match="rotate_deg"):
        lat.load_arm_run(turned, arm="BF", selection=pilot, context_count=8)


def test_a_c2f_pass_is_validated_and_returns_its_sequential_totals(tmp_path):
    pilot = _pilot({1: 30, 2: 40})
    per_query = {i: {"latency_protocol": {"name": "exp37_c2f_sequential_kctx8_kgen1", "context_count": 8, "top_k": 5},
                     "latency_seconds": {"c2f_total": 0.5 * i, "core_forward_total": 0.4 * i},
                     "c2f": {"generated_count": 9 + i, "union_indices": list(range(8 + i))}} for i in (1, 2)}
    run = _run(tmp_path, "c2f", "BF", pilot, identity={"candidate_subset": "c2f", "c2f_top_k": 5,
                                                         "c2f_chebyshev_radius_m": 0.5}, per_query=per_query,
               counts={1: 9, 2: 10})
    rows = lat.load_c2f_run(run, arm="BF", selection=pilot, top_k=5)
    assert rows[2] == {"c2f_total": 1.0, "core": 0.8, "union_count": 10, "generated_count": 11}
    with pytest.raises(RuntimeError, match="c2f_top_k"):
        lat.load_c2f_run(run, arm="BF", selection=pilot, top_k=3)
    head = _run(tmp_path, "head", "BF", pilot)
    with pytest.raises(RuntimeError, match="candidate_subset"):
        lat.load_c2f_run(head, arm="BF", selection=pilot, top_k=5)
