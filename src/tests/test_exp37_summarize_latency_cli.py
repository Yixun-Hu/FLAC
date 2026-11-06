"""exp_37 round B-b: the 3-arm latency summarizer CLI (``tools/exp37_summarize_latency.py``): arms x K_ctx rows with
the bundle's statistics, peak memory as separate quantities (whole-query peak, selector peak, their MAXIMUM, the
loaded-weight footprint), co-tenancy validity, and the stage split only when it agrees with the headline within
2 %. CPU only, synthetic runs. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-b.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from exp37_fakes import attest_run, attest_selector, example_identity, selector_binding, write_attestation
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result
from tools import exp37_summarize_latency as summ

REPO = Path(__file__).resolve().parents[2]
COUNTS = {1: 10, 2: 30}
_AUDIT_BODY = {"mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}
AUDIT = {**_AUDIT_BODY, "sha256": canonical_sha256(_AUDIT_BODY)}


def _write_hashed(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


@pytest.fixture
def world(tmp_path):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c} for i, c in COUNTS.items()]
    selection = {"schema_version": 1, "query_count": 2, "records": records}
    selection = {**selection, "sha256": canonical_sha256(selection)}
    sel_path = _write_hashed(tmp_path / "selection.json", {k: v for k, v in selection.items() if k != "sha256"})
    rows = [{**r, "agree": {"median_seconds": {"scoring_total_seconds": 0.5}}, "peak_memory_allocated_bytes": 3000,
             "query_index": r["index"]} for r in records]
    fields = {"record_type": "exp37_agree_selector_latency", "repeat_count": 3, "candidate_batch_size": 64,
              "queries": rows, "selection_sha256": selection["sha256"], "device": "cuda:0",
              "agree_checkpoint_file_sha256": ident.AGREE_SHA256, "context_manifest_sha256": ident.CONTEXT_MANIFEST_SHA256}
    binding = selector_binding(fields, tmp_path / "selector.json")
    selector = _write_hashed(tmp_path / "selector.json", {**fields, "binding": binding})
    attest_selector(selector, binding, pid=43)
    _write_hashed(tmp_path / "audit.json", _AUDIT_BODY)

    def run(name, arm, k=8, scale=1.0, split=False, cotenant=False):
        expected = ident.registered_identity(arm, selection=selection, audit=AUDIT, seed=42, timing=True,
                                             context_count=k, stage_split=split)
        identity = example_identity(expected, AUDIT)
        run_dir = tmp_path / name
        manifest = initialize_run(run_dir, identity)
        for i, n in COUNTS.items():
            core = (0.1 + n / 100) * scale
            result = {"run_manifest_sha256": manifest["sha256"], "query_index": i, "query_id": f"q{i}.wav",
                      "candidate_count": n, "peak_memory_bytes": 1000 + n,
                      "latency_protocol": {"name": f"fem_core_aligned_kctx{k}_kgen1", "context_count": k,
                                           **({"stage_split": True} if split else {})},
                      "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": core - 0.1,
                                          "core_forward_total": core}}
            if split:
                result.update(stage_seconds={"context_conditioning": 0.1, "candidate_conditioning": (core - 0.1) / 4,
                                             "dit": (core - 0.1) / 4, "vae_decode": (core - 0.1) / 2},
                              stage_sum_seconds=core, stage_peak_memory={
                                  "dit": {"allocated_bytes": 2000, "reserved_bytes": 2500},
                                  "whole_query": {"allocated_bytes": 2100, "reserved_bytes": 2500}})
            save_query_result(run_dir, result=result, candidates=np.zeros((n, 3)), similarities=np.zeros((n, 1)))
        attest_run(run_dir, co_tenant=cotenant, loaded_extra={"memory_after_flac_load": {"memory_allocated_bytes": 700},
                                                              "memory_after_agree_load": {"memory_allocated_bytes": 1100}})
        return run_dir
    return selection, sel_path, selector, run


def test_rows_carry_the_bundle_statistics_memory_and_validity(world):
    selection, _sel_path, selector, run = world
    out = summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, run("p1", "P1")), ("BF", 1, run("bf1", "BF", k=1))])
    p1 = out["rows"][0]
    assert (p1["arm"], p1["context_count"], p1["timing_valid"]) == ("P1", 8, True)
    assert p1["overall"]["summed_seconds"] == pytest.approx(0.2 + 0.4 + 1.0)
    assert p1["memory"] == {"query_peak_allocated_max_bytes": 1030, "query_peak_allocated_median_bytes": 1020.0,
                            "selector_peak_allocated_max_bytes": 3000, "combined_inference_peak_allocated_bytes": 3000,
                            "weights_footprint": {"memory_after_flac_load": {"memory_allocated_bytes": 700},
                                                  "memory_after_agree_load": {"memory_allocated_bytes": 1100},
                                                  "loaded_record_count": 1}}
    assert out["rows"][1]["context_count"] == 1 and "stage_split" not in p1
    assert out["sha256"] == canonical_sha256({k: v for k, v in out.items() if k != "sha256"})


def test_the_split_is_published_within_two_percent_and_withheld_beyond(world):
    selection, _sel_path, selector, run = world
    head = run("head", "YAW")
    near = summ.build_summary(selection, selector, audit=AUDIT, runs=[("YAW", 8, head)], splits=[("YAW", 8, run("s1", "YAW", scale=1.01, split=True))])
    assert near["rows"][0]["stage_split"]["published"] is True and "breakdown" in near["rows"][0]["stage_split"]
    far = summ.build_summary(selection, selector, audit=AUDIT, runs=[("YAW", 8, head)], splits=[("YAW", 8, run("s2", "YAW", scale=1.05, split=True))])
    withheld = far["rows"][0]["stage_split"]
    assert withheld["published"] is False and "breakdown" not in withheld
    assert withheld["comparison"]["aggregate_relative_difference"] == pytest.approx(0.05)
    assert far["rows"][0]["overall"] == near["rows"][0]["overall"]                  # the headline is never withheld


def test_bad_combinations_are_refused(world):
    selection, _sel_path, selector, run = world
    p1 = run("p1", "P1")
    with pytest.raises(ValueError, match="duplicate"):
        summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1), ("P1", 8, p1)])
    with pytest.raises(ValueError, match="no headline"):
        summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1)], splits=[("BF", 8, p1)])


def test_the_cli_writes_json_and_markdown_and_exits_3_on_a_co_tenant(world, tmp_path):
    _selection, sel_path, selector, run = world
    out_json, out_md = tmp_path / "s.json", tmp_path / "s.md"
    argv = [sys.executable, str(REPO / "tools/exp37_summarize_latency.py"), "--selection", str(sel_path),
            "--selector", str(selector), "--geometry-audit", str(tmp_path / "audit.json"),
            "--run", "P1", "8", str(run("p1", "P1")), "--output-json", str(out_json),
            "--output-md", str(out_md)]
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
    done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr
    assert json.loads(out_json.read_text())["rows"][0]["arm"] == "P1" and "Vanilla FLAC" in out_md.read_text()
    argv[argv.index("--run") + 3] = str(run("busy", "P1", cotenant=True))
    out_json.unlink()
    done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 3 and json.loads(out_json.read_text())["rows"][0]["timing_valid"] is False


def test_the_summary_records_the_selector_validity(world, tmp_path):
    selection, _sel_path, selector, run = world
    p1 = run("p1", "P1")
    assert summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1)])["selector_timing_valid"] is True
    body = {k: v for k, v in json.loads(selector.read_text()).items() if k not in ("sha256", "binding")}
    bare = _write_hashed(tmp_path / "bare_selector.json", body)
    out = summ.build_summary(selection, bare, audit=AUDIT, runs=[("P1", 8, p1)])
    assert out["selector_host"]["timing_valid"] is False and out["selector_timing_valid"] is False   # no binding
