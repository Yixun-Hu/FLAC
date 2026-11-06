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

from exp37_fakes import attest_run, attest_selector, example_identity, selector_binding
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility"
BUNDLE_ARMS = {"P1": "vanilla_flac", "BF": "fa_bf_flac", "YAW": "yawaug_flac"}
AUDIT = {"sha256": "a" * 64, "mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}


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
    body = {"record_type": "exp37_agree_selector_latency", "selection_sha256": selection["sha256"], "device": "cuda:0",
            "repeat_count": repeat_count, "candidate_batch_size": 64, "queries": rows,
            "agree_checkpoint_file_sha256": ident.AGREE_SHA256, "context_manifest_sha256": ident.CONTEXT_MANIFEST_SHA256}
    path = tmp_path / "selector.json"
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


def test_the_selector_vector_may_cover_more_queries_but_must_match_each_one(tmp_path, monkeypatch):
    big = _selection({1: 10, 2: 20, 3: 30})
    small = _selection({1: 10, 3: 30})
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], (big["sha256"],))
    rows = lat.load_selector(_selector_file(tmp_path, big), small)
    assert sorted(rows) == [1, 3] and rows[3]["scoring_total_seconds"] == 0.25 and rows[3]["peak_memory_allocated_bytes"] == 100
    with pytest.raises(RuntimeError, match="superset"):                      # a smaller vector never serves
        lat.load_selector(_selector_file(tmp_path, small), big)
    wrong = _selection({1: 11, 3: 30})
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], (wrong["sha256"],))
    with pytest.raises(RuntimeError, match="identity"):
        lat.load_selector(_selector_file(tmp_path, wrong), small)
    with pytest.raises(RuntimeError, match="repeat"):
        lat.load_selector(_selector_file(tmp_path, small, repeat_count=1), small)


def _arm_run(tmp_path, name, arm, selection, *, context_count=8, over=None, query_over=None):
    expected = ident.registered_identity(arm, selection=selection, audit=AUDIT, seed=42, timing=True,
                                         context_count=context_count)
    identity = {**example_identity(expected, AUDIT), **(over or {})}
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
    rows = lat.load_arm_run(_arm_run(tmp_path, "ok", "BF", selection), arm="BF", selection=selection, context_count=8, audit=AUDIT)
    assert rows[2]["core"] == pytest.approx(0.3) and rows[2]["peak_memory_bytes"] == 1020
    for name, arm, over, needle in [("cond", "BF", {"conditioning_method": "vanilla"}, "conditioning_method"),
                                    ("flash", "P1", {"attention_backend": "flash_attn"}, "attention_backend"),
                                    ("k", "YAW", {"n_context": 1}, "n_context"),
                                    ("pilot", "P1", {"pilot_manifest_sha256": "x"}, "pilot_manifest_sha256")]:
        with pytest.raises(RuntimeError, match=needle):
            lat.load_arm_run(_arm_run(tmp_path, name, arm, selection, over=over), arm=arm, selection=selection,
                             context_count=8, audit=AUDIT)
    k1 = _arm_run(tmp_path, "k1", "P1", selection, context_count=1)
    assert sorted(lat.load_arm_run(k1, arm="P1", selection=selection, context_count=1, audit=AUDIT)) == [1, 2]
    with pytest.raises(RuntimeError, match="latency_protocol"):
        lat.load_arm_run(_arm_run(tmp_path, "proto", "P1", selection, query_over={"latency_protocol": {}}),
                         arm="P1", selection=selection, context_count=8, audit=AUDIT)


def _rows(cores, stage_sums=None):
    return {i: {"core": c, "stage_sum": (stage_sums or {}).get(i, c)} for i, c in cores.items()}


def test_the_split_is_publishable_only_within_two_percent_per_query_and_in_aggregate():
    head = _rows({1: 1.0, 2: 10.0})
    ok = lat.split_comparison(head, _rows({1: 1.015, 2: 10.1}))
    assert ok["split_publishable"] is True and ok["aggregate_relative_difference"] == pytest.approx(0.115 / 11)
    one_query_out = lat.split_comparison(head, _rows({1: 1.03, 2: 10.0}))
    assert one_query_out["aggregate_rule_pass"] is True and one_query_out["strict_rule_pass"] is False
    assert one_query_out["split_publishable"] is False and one_query_out["per_query_within_fraction"] == 0.5
    aggregate_out = lat.split_comparison(head, _rows({1: 1.0, 2: 10.3}))
    assert aggregate_out["aggregate_rule_pass"] is False and aggregate_out["split_publishable"] is False
    internal = lat.split_comparison(head, _rows({1: 1.0, 2: 10.0}, stage_sums={2: 9.9}))
    assert internal["internal_stage_sum_vs_core_max_abs"] == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="coverage"):
        lat.split_comparison(head, _rows({1: 1.0}))


def _split_query_over(n):
    return {"latency_protocol": {"name": "fem_core_aligned_kctx8_kgen1", "context_count": 8, "stage_split": True},
            "stage_seconds": {"context_conditioning": 0.1, "candidate_conditioning": n / 400, "dit": n / 400,
                              "vae_decode": n / 200},
            "stage_sum_seconds": 0.1 + n / 100,
            "stage_peak_memory": {"dit": {"allocated_bytes": 900, "reserved_bytes": 1000},
                                  "whole_query": {"allocated_bytes": 950, "reserved_bytes": 1000}}}


def test_headline_and_split_runs_are_never_confused(tmp_path):
    selection = _selection({1: 10})
    split_dir = _arm_run(tmp_path, "split", "P1", selection, over={"stage_split": True},
                         query_over=_split_query_over(10))
    rows = lat.load_arm_run(split_dir, arm="P1", selection=selection, context_count=8, stage_split=True, audit=AUDIT)
    assert rows[1]["stage_seconds"]["vae_decode"] == pytest.approx(0.05) and rows[1]["stage_sum"] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="stage_split"):
        lat.load_arm_run(split_dir, arm="P1", selection=selection, context_count=8, audit=AUDIT)          # not a headline
    head_dir = _arm_run(tmp_path, "head", "P1", selection)
    with pytest.raises(RuntimeError, match="stage_split"):
        lat.load_arm_run(head_dir, arm="P1", selection=selection, context_count=8, stage_split=True, audit=AUDIT)


def test_the_stage_breakdown_adds_the_selector_and_reports_absolute_peaks():
    split = {1: {"stage_seconds": {"context_conditioning": 0.1, "candidate_conditioning": 0.2, "dit": 0.3,
                                   "vae_decode": 0.4},
                 "stage_peak_memory": {"dit": {"allocated_bytes": 900, "reserved_bytes": 1000},
                                       "whole_query": {"allocated_bytes": 950, "reserved_bytes": 1100}}},
             2: {"stage_seconds": {"context_conditioning": 0.1, "candidate_conditioning": 0.6, "dit": 0.7,
                                   "vae_decode": 0.8},
                 "stage_peak_memory": {"dit": {"allocated_bytes": 1200, "reserved_bytes": 1300},
                                       "whole_query": {"allocated_bytes": 1250, "reserved_bytes": 1300}}}}
    out = lat.stage_breakdown(split, {1: 0.5, 2: 0.5}, {1: 10, 2: 30})
    assert out["seconds"]["dit"]["summed_seconds"] == pytest.approx(1.0)
    assert out["seconds"]["agree_selector"]["summed_seconds"] == pytest.approx(1.0)
    assert out["seconds"]["vae_decode"]["ms_per_candidate_dataset_wide"] == pytest.approx(1000 * 1.2 / 40)
    assert sum(v["fraction"] for v in out["seconds"].values()) == pytest.approx(1.0)
    assert out["peak_memory"]["dit"] == {"allocated_bytes": 1200, "reserved_bytes": 1300}          # max, not sum
    assert out["peak_memory"]["whole_query"]["allocated_bytes"] == 1250


def test_a_run_s_timing_is_valid_only_with_its_monitor_and_bound_host_records(tmp_path):
    """BLOCKING 3: the verdict is the run-bound monitor's over the whole timed interval plus hashed start / loaded /
    end host records bound to the run manifest -- never a snapshot."""
    selection = _selection({1: 10})
    run = attest_run(_arm_run(tmp_path, "ok", "P1", selection))
    verdict = lat.host_cotenancy(run)
    assert verdict["timing_valid"] is True and verdict["reasons"] == [] and len(verdict["host_records"]) == 3
    assert verdict["monitor"]["foreign_cpu_cores_mean"] == pytest.approx(0.2)
    busy = attest_run(_arm_run(tmp_path, "busy", "P1", selection), co_tenant=True)
    assert lat.host_cotenancy(busy)["timing_valid"] is False
    exp36 = attest_run(_arm_run(tmp_path, "exp36", "P1", selection), exp36=True)
    assert any("exp_36" in r for r in lat.host_cotenancy(exp36)["reasons"])
    bare = _arm_run(tmp_path, "bare", "P1", selection)                         # no monitor, no host records
    assert lat.host_cotenancy(bare)["timing_valid"] is False
    loose = attest_run(_arm_run(tmp_path, "loose", "P1", selection))
    record = next((loose / "host_records").glob("loaded_*.json"))
    record.write_text(json.dumps({**json.loads(record.read_text()), "run_manifest_sha256": "x"}))   # unhashed / foreign
    assert lat.host_cotenancy(loose)["timing_valid"] is False
    missing = attest_run(_arm_run(tmp_path, "missing", "P1", selection))
    next((missing / "host_records").glob("end_*.json")).unlink()
    assert lat.host_cotenancy(missing)["timing_valid"] is False
    assert lat.host_cotenancy(tmp_path / "absent")["timing_valid"] is False


def test_the_selector_vector_needs_its_own_bound_monitor(tmp_path):
    selection = _selection({1: 10})
    path = _selector_file(tmp_path, selection)
    payload = json.loads(path.read_text())
    assert lat.selector_cotenancy(path)["timing_valid"] is False                 # no binding (e.g. the bundle's)
    binding = selector_binding(payload, path)
    body = {**{k: v for k, v in payload.items() if k != "sha256"}, "binding": binding}
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    assert lat.selector_cotenancy(path)["timing_valid"] is False                 # no monitor yet
    attest_selector(path, binding)
    verdict = lat.selector_cotenancy(path)
    assert verdict["timing_valid"] is True and verdict["binding_sha256"] == binding["sha256"]
    other = tmp_path / "other"
    other.mkdir()
    wrong = _selector_file(other, _selection({1: 11}))
    body = {**{k: v for k, v in json.loads(wrong.read_text()).items() if k != "sha256"}, "binding": binding}
    wrong.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))     # another run's binding
    attest_selector(wrong, binding)
    assert lat.selector_cotenancy(wrong)["timing_valid"] is False
    busy = tmp_path / "busy"
    busy.mkdir()
    shared = _selector_file(busy, selection)
    binding = selector_binding(json.loads(shared.read_text()), shared)
    body = {**{k: v for k, v in json.loads(shared.read_text()).items() if k != "sha256"}, "binding": binding}
    shared.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    attest_selector(shared, binding, co_tenant=True)
    assert lat.selector_cotenancy(shared)["timing_valid"] is False
