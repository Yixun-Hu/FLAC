"""exp_37 B2 / B3 aggregation on the complete synthetic registered world (Codex code review 1, BLOCKING 4, HIGH 3).

Every accuracy number comes from a validated source with waveform coverage (accuracy-only runs: their inline dump;
seed-42 timing passes: their bitwise replay); every accuracy statistic is n = 3 (seeds 42-44); times come from
attested timing passes only; the summary carries an inventory of every source's hashes (inputs, selectors with their
bindings and monitor heads, run / replay / dump manifests, per-query artifacts, host attestations, the pinned exp_23
summary and bundle latency CSV, the synthesis metrics files). Publication outputs (JSON, markdown, PNG + PDF with the
Pareto frontier) are written only when every timing is attested and the synthesis section is complete; otherwise a
separate DIAGNOSTIC JSON / markdown without any frontier or figure, exit 3. Outputs are write-once.
CPU only. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1 (replaces the round B-c tests).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from exp37_fakes import write_attestation
from exp37_world import World, registered_world
from tools import exp37_aggregate_b2b3 as tool
from tools import exp37_build_b2b3_spec as builder

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = registered_world(tmp_path, monkeypatch)                          # every registered run + synthesis stand-ins
    summary_sha, csv_sha = w.external()
    monkeypatch.setattr(tool, "EXP23_SUMMARY_SHA256", summary_sha)      # the synthetic world's own references
    monkeypatch.setattr(tool, "LATENCY_CSV_SHA256", csv_sha)
    return w


def _spec(w, synthesis=True):
    return builder.build_spec(w.inputs, w.runs, synthesis=w.root / "synthesis.json" if synthesis else None)


def test_a_complete_world_is_publishable_with_n3_everywhere(world):
    out = tool.build_summary(_spec(world))
    assert out["publishable"] is True and out["reasons"] == [] and out["record_type"] == "exp37_b2_b3_summary"
    points = {(p["arm"], p["grid"]): p for p in out["b2"]["points"]}
    assert set(points) == {(a, g) for a in ("P1", "BF") for g in ("0.50", "0.75", "1.00_nested", "c2f_k5")}
    assert all(p["accuracy"][m]["n"] == 3 for p in points.values() for m in tool.METRICS)
    assert points[("P1", "0.50")]["seconds_per_query"] == pytest.approx(2.5)                # 2.0 core + 0.5 selector
    assert points[("BF", "1.00_nested")]["seconds_per_query"] == pytest.approx(0.5)
    assert points[("BF", "c2f_k5")]["seconds_per_query"] == pytest.approx(0.9)               # the sequential total
    assert points[("P1", "0.75")]["seconds_per_query"] == pytest.approx(0.8)
    assert "seconds_per_query_repeats" not in points[("P1", "0.50")]                         # D-a: seed 42 only
    assert all(p["timing_valid"] for p in points.values()) and out["b2"]["frontier"]["success_1_0m"]
    c2f = points[("BF", "c2f_k5")]["accuracy"]                                               # sequential c2f runs
    assert c2f["seeds"] == [42, 43, 44] and c2f["median_localization_error_m"]["n"] == 3
    parity = out["b2"]["c2f_parity"]["BF"]                                                   # the offline cross-check
    assert set(parity) == {"42", "43", "44"} and all(v["union_equal"] == v["queries"] == 2 for v in parity.values())
    assert {s["grid"] for s in out["b2"]["sensitivity"]} == {"c2f_k1", "c2f_k3", "c2f_k10", "c2f_k5_offline"}
    cells = {(r["orbit"], r["theta"]): r for r in out["b3"]["accuracy"]}
    assert set(cells) == {(o, t) for o in ("0", "0_180", "c4") for t in (0, 90)}
    assert all(r["accuracy"]["median_localization_error_m"]["n"] == 3 for r in cells.values())
    c4 = next(p for p in out["b2"]["points"] if (p["arm"], p["grid"]) == ("BF", "0.50"))
    assert cells[("c4", 0)]["accuracy"]["success_1_0m"] == c4["accuracy"]["success_1_0m"]   # C4 theta 0 = B2 B-F 0.50
    assert sorted(r["orbit"] for r in out["b3"]["latency"]) == ["0", "0_180", "c4"]
    assert out["b3"]["synthesis"]["status"] == "complete" and len(out["b3"]["synthesis"]["cells"]) == 12


def test_the_inventory_binds_every_source(world):
    spec = _spec(world)
    out = tool.build_summary(spec)
    inv = out["inventory"]
    dirs = {i["dir"] for i in spec["b2_runs"] + spec["b2_timing"] + spec["b2_c2f_runs"] + spec["b3_runs"]
            + spec["b3_latency"]["runs"]}
    assert set(inv["runs"]) == dirs and len(dirs) == 38
    assert all("replay" in inv["runs"][i["dir"]] for i in spec["b2_timing"] + spec["b3_latency"]["runs"])   # D-c
    seed42 = inv["runs"][next(i["dir"] for i in spec["b2_runs"] if (i["arm"], i["grid"], i["seed"]) == ("BF", "0.50", 42))]
    assert seed42["replay"]["waveforms"]["mode"] == "replay" and len(seed42["attestation"]["monitor_head_sha256"]) == 64
    acc = inv["runs"][next(i["dir"] for i in spec["b3_runs"])]
    assert acc["waveforms"]["mode"] == "inline" and len(acc["waveforms"]["dump_manifest_sha256"]) == 64
    assert set(inv["inputs"]) == set(tool.INPUT_KEYS) - {"selector_0p50", "selector_0p75", "selector_nested"} | {
        "latency_selection"}
    assert set(inv["selectors"]) == {"selector_0p50", "selector_0p75", "selector_nested", "latency_selector"}
    assert all(len(s["binding_sha256"]) == 64 for s in inv["selectors"].values())
    assert inv["external"]["exp23_summary"]["sha256"] == tool.EXP23_SUMMARY_SHA256
    assert inv["external"]["latency_csv"]["file_sha256"] == tool.LATENCY_CSV_SHA256
    assert len(inv["synthesis"]) == 12 and out["spec_sha256"]


def test_an_unattested_timing_withholds_publication_and_the_frontier(world, tmp_path):
    run = world.runs / "b2_0p50_BF_s42"
    shutil.rmtree(run / "cotenancy")
    write_attestation(run / "cotenancy", json.loads((run / "run_manifest.json").read_text())["sha256"], co_tenant=True)
    out = tool.build_summary(_spec(world))
    assert out["publishable"] is False and out["b2"]["frontier"] is None
    assert out["record_type"] == "exp37_b2_b3_DIAGNOSTIC" and any("BF 0.50" in r for r in out["reasons"])


def test_a_pending_synthesis_is_diagnostic_only(world):
    out = tool.build_summary(_spec(world, synthesis=None))
    assert out["publishable"] is False and any("synthesis" in r for r in out["reasons"])
    assert out["b3"]["synthesis"]["status"] == "pending"


def test_a_tampered_synthesis_artifact_is_refused(world):
    spec = _spec(world)
    Path(spec["b3_synthesis"]["cells"][3]["metrics_json"]).write_text(json.dumps({"T60": 0.0}))
    with pytest.raises(RuntimeError, match="sha256"):
        tool.build_summary(spec)


def test_a_tampered_waveform_is_refused(world):
    spec = _spec(world)
    target = world.waves / "b2_0p50_P1_s43" / "query_00001.npy"
    target.chmod(0o644)                                                   # r2: published files are read-only
    np.save(target, np.zeros((15, 1, 10240), np.float32))
    with pytest.raises(RuntimeError, match="sha256"):
        tool.build_summary(spec)


def test_a_claimed_cell_must_be_the_run_s_identity(world):
    spec = _spec(world)
    a, b = spec["b3_runs"][0], spec["b3_runs"][3]                        # {0} theta 0 s42 <-> {0} theta 90 s42
    a["dir"], b["dir"] = b["dir"], a["dir"]
    with pytest.raises(RuntimeError, match="rotate_deg"):
        tool.build_summary(spec)
    spec = _spec(world)
    p1, bf = (next(i for i in spec["b2_runs"] if (i["arm"], i["grid"], i["seed"]) == (arm, "0.75", 43)) for arm in ("P1", "BF"))
    p1["dir"], bf["dir"] = bf["dir"], p1["dir"]
    with pytest.raises(RuntimeError, match="checkpoint_sha256"):
        tool.build_summary(spec)


def test_the_external_references_are_pinned_to_the_real_files():
    real = {"accuracy": str(REPO / "worklog/worklog_yixun/exp_23_five_method_64_random_fallback/summary.json"),
            "latency_csv": str(REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility/expected/latency_per_query.csv")}
    pilot = json.loads((E9 / "pilot_manifest_seed42_4_per_room.json").read_text())
    inventory = {}
    markers = tool.load_external(real, pilot, inventory)
    assert set(markers) == {"few_shot_rir", "fem_omp"} and inventory["exp23_summary"]["sha256"] == tool.EXP23_SUMMARY_SHA256
    assert tool.LATENCY_CSV_SHA256 == "3cac4a9c92f0943a57c0a228f94ca4d8d9357ea244214e202238cdb01b0191b8"


def test_unpinned_external_references_are_refused(world, monkeypatch):
    monkeypatch.setattr(tool, "EXP23_SUMMARY_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="pinned"):
        tool.build_summary(_spec(world))


def _main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["exp37_aggregate_b2b3.py"] + [str(a) for a in argv])
    return tool.main()


def test_the_cli_publishes_once_and_otherwise_writes_only_a_diagnostic(world, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_spec(world)))
    out, stem = tmp_path / "out" / "b2b3.json", tmp_path / "out" / "b2_pareto"
    argv = ["--spec", spec, "--output-json", out, "--output-md", out.with_suffix(".md"), "--figure-stem", stem]
    assert _main(monkeypatch, argv) == 0
    assert stem.with_suffix(".png").stat().st_size > 10000 and stem.with_suffix(".pdf").stat().st_size > 1000
    assert "Pareto frontier" in out.with_suffix(".md").read_text() and json.loads(out.read_text())["publishable"]
    assert _main(monkeypatch, argv) == 2                                                    # write-once
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(_spec(world, synthesis=None)))
    out2 = tmp_path / "out2" / "b2b3.json"
    argv2 = ["--spec", pending, "--output-json", out2, "--output-md", out2.with_suffix(".md"), "--figure-stem",
             tmp_path / "out2" / "fig"]
    assert _main(monkeypatch, argv2) == 3
    assert not out2.exists() and not (tmp_path / "out2" / "fig.png").exists()
    diagnostic = json.loads((tmp_path / "out2" / "b2b3.DIAGNOSTIC.json").read_text())
    assert diagnostic["publishable"] is False and diagnostic["b2"]["frontier"] is None
    text = (tmp_path / "out2" / "b2b3.DIAGNOSTIC.md").read_text()
    assert text.startswith("# DIAGNOSTIC") and "Pareto frontier" not in text


def test_the_cli_refuses_an_incomplete_spec_before_writing(world, tmp_path, monkeypatch):
    body = _spec(world)
    body["b2_runs"].pop()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(body))
    out = tmp_path / "o" / "b2b3.json"
    assert _main(monkeypatch, ["--spec", spec, "--output-json", out, "--output-md", out.with_suffix(".md")]) == 2
    assert not (tmp_path / "o").exists()
