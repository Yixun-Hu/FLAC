"""exp_37 round B-b: the 3-arm latency summary (plan_loc_efficiency.md §1 formulas unchanged, §10 B1 / CPU
co-tenancy). L_q = core_forward_total_q + the AGREE selector's median scoring_total_q; s/query mean / median / P90
over the queries; ms/candidate dataset-wide = 1000 * sum L / sum |C|, query median / P90 of 1000 * L_q / |C_q|.
The bundle's own repeat_001 + selector vector must reproduce its expected summary EXACTLY (an oracle, no GPU).

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility"
BUNDLE_ARMS = {"P1": "vanilla_flac", "BF": "fa_bf_flac", "YAW": "yawaug_flac"}


def _bundle_queries(arm):
    out = {}
    for path in sorted((LB / "data/repeats/repeat_001" / BUNDLE_ARMS[arm] / "queries").glob("query_*.json")):
        payload = json.loads(path.read_text())
        out[int(payload["query_index"])] = payload
    return out


@pytest.mark.parametrize("arm", ["P1", "BF", "YAW"])
def test_the_bundle_rows_are_reproduced_exactly(arm):
    selection = json.loads((LB / "data/selection/frozen_16room_128.json").read_text())
    selector = lat.load_selector(LB / "data/selector/selector_latency_128.json", selection)
    queries = _bundle_queries(arm)
    core = {i: q["latency_seconds"]["core_forward_total"] for i, q in queries.items()}
    counts = {int(r["index"]): int(r["candidate_count"]) for r in selection["records"]}
    stats = lat.latency_statistics(core, {i: row["scoring_total_seconds"] for i, row in selector.items()}, counts)
    expected = json.loads((LB / "expected/summary_final.json").read_text())
    assert stats["overall"] == expected["overall"][BUNDLE_ARMS[arm]]
    assert stats["per_candidate"] == expected["per_candidate"][BUNDLE_ARMS[arm]]


def test_statistics_follow_the_bundle_formulas_on_a_toy():
    stats = lat.latency_statistics({1: 1.0, 2: 3.0}, {1: 0.5, 2: 0.5}, {1: 10, 2: 30})
    assert stats["overall"]["mean_seconds"] == 2.5 and stats["overall"]["summed_seconds"] == 5.0
    assert stats["overall"]["p90_seconds"] == pytest.approx(1.5 + 0.9 * 2.0)
    assert stats["per_candidate"]["amortized_seconds"] == 5.0 / 40
    assert stats["per_candidate"]["median_seconds"] == pytest.approx((0.15 + 3.5 / 30) / 2)
    assert stats["per_candidate"]["candidate_evaluations"] == 40
    with pytest.raises(RuntimeError, match="coverage"):
        lat.latency_statistics({1: 1.0}, {1: 0.5, 2: 0.5}, {1: 10, 2: 30})


def _selection(counts):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c} for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def _selector_file(tmp_path, selection, repeat_count=3, rows=None):
    rows = rows if rows is not None else [
        {"query_index": r["index"], "query_id": r["query_id"], "room": r["room"], "candidate_count": r["candidate_count"],
         "agree": {"median_seconds": {"scoring_total_seconds": 0.25}}, "peak_memory_allocated_bytes": 100}
        for r in selection["records"]]
    body = {"record_type": "exp37_agree_selector_latency", "selection_sha256": "anything", "repeat_count": repeat_count,
            "candidate_batch_size": 64, "queries": rows}
    path = tmp_path / "selector.json"
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


def test_the_selector_vector_may_cover_more_queries_but_must_match_each_one(tmp_path):
    big = _selection({1: 10, 2: 20, 3: 30})
    small = _selection({1: 10, 3: 30})
    rows = lat.load_selector(_selector_file(tmp_path, big), small)
    assert sorted(rows) == [1, 3] and rows[3]["scoring_total_seconds"] == 0.25 and rows[3]["peak_memory_allocated_bytes"] == 100
    with pytest.raises(RuntimeError, match="coverage"):
        lat.load_selector(_selector_file(tmp_path, small), big)
    with pytest.raises(RuntimeError, match="identity"):
        lat.load_selector(_selector_file(tmp_path, _selection({1: 11, 3: 30})), small)
    with pytest.raises(RuntimeError, match="repeat"):
        lat.load_selector(_selector_file(tmp_path, small, repeat_count=1), small)


def _host(run_dir, label, uuid, apps, top=()):
    path = run_dir / "host_records" / f"{label}_20261106T000000000000Z_pid42.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {"pid": 42, "cuda_device": {"uuid": uuid} if uuid is not None else None,
                "gpu_compute_apps": {"lines": list(apps)}, "top_cpu_processes": {"lines": list(top)}}
    path.write_text(json.dumps({"label": label, "snapshot": snapshot}))


def test_co_tenancy_from_the_host_records(tmp_path):
    run = tmp_path / "run"
    _host(run, "start", "6ac2604a-1111", [])
    _host(run, "end", "6ac2604a-1111", ["GPU-6ac2604a-1111, 42, python, 1200 MiB", "GPU-0a1aaa90-2222, 77, python, 9 MiB"],
          top=["42 1 me 99.0 1.0 10:00 python", "77 1 me 95.0 1.0 10:00 python", "5 1 root 0.5 0.0 1:00 init"])
    verdict = lat.host_cotenancy(run)
    assert verdict["timing_valid"] is True and verdict["gpu_cotenants"] == []
    assert verdict["cpu_cotenants"] == ["77 1 me 95.0 1.0 10:00 python"]        # recorded, never silently ignored
    _host(run, "end", "6ac2604a-1111", ["GPU-6ac2604a-1111, 42, python, 1 MiB", "GPU-6ac2604a-1111, 99, python, 5 MiB"])
    assert lat.host_cotenancy(run)["timing_valid"] is False
    _host(tmp_path / "nouuid", "start", None, [])
    assert lat.host_cotenancy(tmp_path / "nouuid")["timing_valid"] is False   # an unknown card is never valid
    assert lat.host_cotenancy(tmp_path / "absent")["timing_valid"] is False


def _arm_run(tmp_path, name, arm, selection, *, context_count=8, over=None, query_over=None):
    identity = {"conditioning_method": lat.ARM_PROTOCOL[arm]["cond"], "checkpoint_sha256": lat.ARM_PROTOCOL[arm]["ckpt"],
                "n_context": context_count, "score_sample_counts": [1], "measure_core_forward": True,
                "synchronize_timing": True, "warmup_query_count": 1, "candidate_batch_size": 64, "tau": 0.1,
                "attention_backend": "sdpa", "pilot_manifest_sha256": selection["sha256"],
                "query_indices": [r["index"] for r in selection["records"]], **(over or {})}
    run = tmp_path / name
    manifest = initialize_run(run, identity)
    for r in selection["records"]:
        n = r["candidate_count"]
        result = {"schema_version": 1, "run_manifest_sha256": manifest["sha256"], "query_index": r["index"],
                  "query_id": r["query_id"], "room": r["room"], "candidate_count": n, "peak_memory_bytes": 1000 + n,
                  "latency_protocol": {"name": f"fem_core_aligned_kctx{context_count}_kgen1",
                                       "context_count": context_count},
                  "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": n / 100,
                                      "core_forward_total": 0.1 + n / 100}, **(query_over or {})}
        save_query_result(run, result=result, candidates=np.zeros((n, 3)), similarities=np.zeros((n, 1)))
    return run


def test_an_arm_run_is_validated_against_its_protocol(tmp_path):
    selection = _selection({1: 10, 2: 20})
    rows = lat.load_arm_run(_arm_run(tmp_path, "ok", "BF", selection), arm="BF", selection=selection, context_count=8)
    assert rows[2]["core"] == pytest.approx(0.3) and rows[2]["peak_memory_bytes"] == 1020
    for name, arm, over, needle in [("cond", "BF", {"conditioning_method": "vanilla"}, "conditioning_method"),
                                    ("flash", "P1", {"attention_backend": "flash_attn"}, "attention_backend"),
                                    ("k", "YAW", {"n_context": 1}, "n_context"),
                                    ("pilot", "P1", {"pilot_manifest_sha256": "x"}, "pilot_manifest_sha256")]:
        with pytest.raises(RuntimeError, match=needle):
            lat.load_arm_run(_arm_run(tmp_path, name, arm, selection, over=over), arm=arm, selection=selection,
                             context_count=8)
    k1 = _arm_run(tmp_path, "k1", "P1", selection, context_count=1)
    assert sorted(lat.load_arm_run(k1, arm="P1", selection=selection, context_count=1)) == [1, 2]
    with pytest.raises(RuntimeError, match="latency_protocol"):
        lat.load_arm_run(_arm_run(tmp_path, "proto", "P1", selection, query_over={"latency_protocol": {}}),
                         arm="P1", selection=selection, context_count=8)
